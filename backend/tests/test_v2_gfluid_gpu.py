"""v2 流体 2D の GPU 版 (es_sim.gfluid.gpu) のテスト (prompts/126)。CUDA が無ければ skip。

GPU 版は CPU 版 (CartesianFluid2dSimulation) と同じ離散化・同じ BiCGSTAB 漸化式なので、状態・履歴・
時間平均・位相分解が丸め誤差の範囲で一致し、反復数も同じになることを確かめる。加えて、状態の
読み書き (デバイスが正、読むとホストへ写す)、Poisson の 3 経路 (密な直接法・同期なしの GMG-PCG・
特異な問題の同期版)、Boltzmann 係数モデル、続き実行、server / 振り分けを確かめる。
合成格子 (AMR) 版 (GpuAmrFluid2dSimulation: 輸送は同じカーネル、Poisson は GPU の AMG-PCG) も
CPU の AmrFluid2dSimulation と一致することを確かめる (prompts/128)。サブステップ数の上限・サブステップ間の
停止 (デバイスの状態の巻き戻し)・Joule 緩和時間 (電子がほぼ空の節点を除く) も v1 と同じ判定になること。
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from es_sim.device import cuda_available

pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA が使えません")

if cuda_available():
    import es_sim.gfluid.gpu as gpu_mod
    from es_sim.gfluid import CartesianFluid2dSimulation, GpuCartesianFluid2dSimulation, make_fluid2d_simulation
    from es_sim.gfluid.amr import AmrFluid2dSimulation, GpuAmrFluid2dSimulation

from es_sim.fluid_coeffs import build_fluid_reactions
from es_sim.pic1d_presets import edupic_ar_processes
from es_sim.schema import Project

W, H = 0.02, 0.01
RF = {"amplitude": 100.0, "freq_hz": 13.56e6}
PIN = {"id": "pin", "type": "conductor", "voltage": 0.0, "see_gamma": 0.05,
       "shape": {"kind": "circle", "center": [0.0137, 0.0068], "radius": 0.0011}}
BLOCK = {"id": "blk", "type": "dielectric", "eps_r": 4.0, "see_gamma": 0.1,
         "polygon": [[0.008, 0.0], [0.012, 0.0], [0.012, 0.003], [0.008, 0.003]]}
CCP_BND = [
    {"edges": [3], "type": "dirichlet", "voltage": 0.0, "voltage_rf": RF, "see_gamma": 0.05},
    {"edges": [1], "type": "dirichlet", "voltage": 0.0, "see_gamma": 0.05},
    {"edges": [0, 2], "type": "symmetry"},
]


AMR1 = {"max_level": 1, "buffer_cells": 2}


def _project(regions=(PIN, BLOCK), boundaries=CCP_BND, size=0.5e-3, coord="xy", w=W, h=H, amr=None,
             **fluid) -> Project:
    f = {"init_density_m3": 5e14, "init_te_ev": 3.0, "gas_pressure_pa": 30.0, "n_steps": 20,
         "frame_every": 10**9}
    f.update(fluid)
    return Project.model_validate({
        "coord": coord,
        "geometry": {"domain": {"polygon": [[0, 0], [w, 0], [w, h], [0, h]]},
                     "boundaries": list(boundaries), "regions": list(regions)},
        "mesh": {"size": size, "mode": "cartesian", **({"amr": amr} if amr else {})},
        "fluid2d": f,
    })


def _run_pair(project: Project, amr: bool = False):
    cpu = (AmrFluid2dSimulation if amr else CartesianFluid2dSimulation)(project.model_copy(deep=True), device="cpu")
    gpu = (GpuAmrFluid2dSimulation if amr else GpuCartesianFluid2dSimulation)(project.model_copy(deep=True))
    cpu.run_batch(store_frames=False)
    gpu.run_batch(store_frames=False)
    return cpu, gpu


def _rel(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(a))), 1e-300))


def _assert_same(cpu, gpu, tol=1e-9) -> None:
    for k in ("n_e", "n_i", "w", "phi", "q_surf"):      # q_surf: 誘電体の表面電荷 (prompts/129)
        assert _rel(getattr(cpu, k), getattr(gpu, k)) < tol, k
    for k in ("n_e_total", "n_i_total", "wall_e", "wall_i", "gen_total", "surf_q"):
        assert _rel(cpu.history[k], gpu.history[k]) < tol, k
    assert cpu.timing["solver_iters"] == gpu.timing["solver_iters"]


# ---- CPU 版との一致 --------------------------------------------------------------------


def test_matches_cpu_ccp_with_solids_fields_and_cycle():
    """固体入り CCP (RF・SEE・Frost・位相分解): 状態・履歴・時間平均・位相分解・反復数が CPU 版と一致。"""
    cpu, gpu = _run_pair(_project(n_steps=200, avg_steps=100, phase_bins=8))
    assert gpu._g.poisson.solver.direct            # 861 節点: 密な直接法の経路
    _assert_same(cpu, gpu)
    for k in ("phi", "e_abs", "n_e", "n_i", "t_e", "ionization"):
        assert _rel(cpu.fields[k], gpu.fields[k]) < 1e-9, k
    assert cpu.fields["avg_steps"] == gpu.fields["avg_steps"] == 100
    for k in ("phi", "n_e", "n_i", "t_e"):
        assert _rel(cpu.cycle[k], gpu.cycle[k]) < 1e-9, k
    assert gpu.warnings == cpu.warnings == []


def test_matches_cpu_with_gmg_poisson_and_const_mobility():
    """未知数 4096 超 (同期なしの GMG-PCG をグラフで再生) と移動度一定のイオン。"""
    cpu, gpu = _run_pair(_project(size=0.2e-3, n_steps=40, ion_mobility_model="const"))
    assert not gpu._g.poisson.solver.direct and gpu.poisson_iters > 0
    _assert_same(cpu, gpu, tol=1e-8)       # Poisson は 1e-10 まで反復で解く (CPU は LU)


def test_matches_cpu_axisymmetric():
    bnd = [{"edges": [0], "type": "symmetry"},
           {"edges": [1], "type": "dirichlet", "voltage": 0.0, "voltage_rf": RF, "see_gamma": 0.05},
           {"edges": [2], "type": "dirichlet", "voltage": 0.0, "see_gamma": 0.05},
           {"edges": [3], "type": "dirichlet", "voltage": 0.0, "see_gamma": 0.05}]
    reg = [{"id": "d", "type": "dielectric", "eps_r": 4.0,
            "polygon": [[0.004, 0.0], [0.0065, 0.0], [0.0065, 0.0027], [0.004, 0.0027]]}]
    cpu, gpu = _run_pair(_project(reg, bnd, size=1e-3, coord="rz", init_density_m3=1e14, n_steps=300))
    _assert_same(cpu, gpu)


def test_matches_cpu_boltzmann_coefficients():
    """electron_model="boltzmann" (ε̄ をキーにした表、e_ion/e_exc も表) でも CPU 版と一致。"""
    procs, _ = edupic_ar_processes()
    r = build_fluid_reactions(procs)
    te = r.te_grid_ev
    n = len(te)
    table = {
        "en_td": list(np.linspace(1.0, 300.0, n)), "mean_energy_ev": list(1.5 * te),
        "mobility_n": list(r.transport.mobility_n), "k_ion": list(r.k_ion), "k_exc": list(r.k_exc),
        "e_ion_ev": list(np.full(n, 15.8)), "e_exc_ev": list(np.full(n, 11.5)),
        "eedf_eps_ev": [0.0, 1.0], "eedf": [[0.0, 0.0]] * n, "source_hash": "synthetic",
    }
    cpu, gpu = _run_pair(_project(n_steps=150, electron_model="boltzmann", boltz_table=table))
    _assert_same(cpu, gpu)


def test_debug_flags_boltzmann_equilibrium_on_gpu():
    """イオン固定・反射壁・源と電子エネルギー無し: n_e ∝ exp(φ/Te) に緩和 (状態の代入がデバイスへ届く)。"""
    length, te0, n_i0 = 0.02, 2.0, 1.0e14
    bnd = [{"edges": [e], "type": "dirichlet", "voltage": 0.0} for e in range(4)]
    obst = {"id": "d", "type": "dielectric", "eps_r": 3.0,
            "shape": {"kind": "circle", "center": [0.0123, 0.0087], "radius": 0.0031}}
    sim = GpuCartesianFluid2dSimulation(_project([obst], bnd, size=1e-3, w=length, h=length,
                                                 init_density_m3=n_i0, init_te_ev=te0, gas_pressure_pa=50.0))
    sim.debug_source_enabled = False
    sim.debug_energy_enabled = False
    sim.debug_ions_enabled = False
    sim.debug_reflective_walls = True
    x, y = sim.mesh.nodes[:, 0], sim.mesh.nodes[:, 1]
    act = sim.active_idx
    ni = np.zeros(sim.n_nodes)
    ni[act] = n_i0 * (1.0 + 0.5 * np.cos(math.pi * x[act] / length) * np.cos(math.pi * y[act] / length))
    sim.n_i = ni
    sim.n_e[act] = n_i0              # 読んだ配列のその場の書き換えも次のステップで反映される
    sim.w = 1.5 * sim.n_e * te0
    total0 = float(np.sum(sim.n_e[act] * sim.node_vol))
    for _ in range(1500):
        sim.step()
    ne = sim.n_e
    assert float(np.sum(ne[act] * sim.node_vol)) == pytest.approx(total0, rel=1e-6)
    assert np.array_equal(sim.n_i, ni)                                   # イオンは動かない
    ref = act[np.argmin(np.hypot(x[act] - 0.004, y[act] - 0.004))]
    d = np.log(ne[act] / ne[ref]) - (sim.phi[act] - sim.phi[ref]) / te0
    assert np.max(np.abs(d)) < 0.02


def test_particle_balance_on_gpu():
    """(電離生成 − 壁損失) と全量変化が一致 (反復法の収束誤差の範囲)。"""
    bnd = [{"edges": [e], "type": "dirichlet", "voltage": 0.0, "see_gamma": 0.05} for e in range(4)]
    sim = GpuCartesianFluid2dSimulation(_project(boundaries=bnd, init_density_m3=1e15, gas_pressure_pa=50.0,
                                                 dt=5e-11))
    act = sim.active_idx
    ne0 = float(np.sum(sim.n_e[act] * sim.node_vol))
    ni0 = float(np.sum(sim.n_i[act] * sim.node_vol))
    for _ in range(200):
        sim.step()
    ne1 = float(np.sum(sim.n_e[act] * sim.node_vol))
    ni1 = float(np.sum(sim.n_i[act] * sim.node_vol))
    assert (ne1 - ne0) == pytest.approx(sim.gen_total - sim.wall["electron"], rel=1e-6)
    assert (ni1 - ni0) == pytest.approx(sim.gen_total - sim.wall["ion"], rel=1e-6)
    assert sim.history["n_e_total"][-1] == pytest.approx(ne1, rel=1e-12)


def test_singular_poisson_and_continue_on_gpu():
    """Dirichlet の無い問題 (特異な Poisson を同期版で) と、続き実行で履歴・時間平均が作り直されること。"""
    sim = GpuCartesianFluid2dSimulation(_project([], [], size=1e-3, n_steps=30, avg_steps=10))
    assert sim._g.poisson.solver.singular
    hist, _ = sim.run_batch(store_frames=False)
    assert len(hist["t"]) == 30 and np.all(np.isfinite(sim.phi))
    sim.prepare_continue(20, avg_steps=5)
    hist, _ = sim.run_batch(store_frames=False)
    assert len(hist["t"]) == 20 and hist["step"][0] == 31
    assert sim.fields is not None and sim.fields["avg_steps"] == 5


def test_frames_carry_device_state():
    sim = GpuCartesianFluid2dSimulation(_project(n_steps=30, frame_every=10))
    _, frames = sim.run_batch()
    assert [f["step"] for f in frames] == [10, 20, 30]
    last = frames[-1]
    assert np.allclose(last["n_e"], sim.n_e) and np.allclose(last["phi"], sim.phi)
    assert last["counts"]["n_e_total"] == sim.history["n_e_total"][-1]


# ---- サブステップの制御 (v1 fluid2d の上限・停止・τ_J と同じ判定) ---------------------------------


def test_dc_cathode_sheath_substeps_match_cpu():
    """健全な DC 放電: 陰極シースのほぼ空の節点を τ_J から除き、サブステップ数がステップごとに CPU 版と
    同じで少ない (除外を忘れると 1 ステップ数千万回を要求して上限で ValueError になる)。"""
    bnd = [{"edges": [3], "type": "dirichlet", "voltage": 0.0}, {"edges": [1], "type": "dirichlet", "voltage": 100.0},
           {"edges": [0, 2], "type": "symmetry"}]
    p = _project([], bnd, size=2e-3, w=0.1, h=0.05, init_density_m3=1e15, init_te_ev=2.0, gas_pressure_pa=50.0)
    cpu = CartesianFluid2dSimulation(p.model_copy(deep=True), device="cpu")
    gpu = GpuCartesianFluid2dSimulation(p.model_copy(deep=True))
    counts = {"cpu": [], "gpu": []}

    def counting(name, f):
        def wrapped(*a, **kw):
            counts[name][-1] += 1
            return f(*a, **kw)
        return wrapped

    cpu._step_once = counting("cpu", cpu._step_once)
    gpu._g.substep = counting("gpu", gpu._g.substep)
    for _ in range(200):
        for name, sim in (("cpu", cpu), ("gpu", gpu)):
            counts[name].append(0)
            sim.step()
    assert counts["gpu"] == counts["cpu"] and max(counts["gpu"]) <= 5
    assert _rel(cpu.phi, gpu.phi) < 1e-9 and _rel(cpu.n_e, gpu.n_e) < 1e-9
    assert float(np.max(gpu.phi)) < 110.0


def test_substep_limit_raises_without_changing_device_state():
    sim = GpuCartesianFluid2dSimulation(_project(init_density_m3=1e15, gas_pressure_pa=50.0, dt=1e-9))
    sim.step()
    before = {k: getattr(sim, k).copy() for k in ("n_e", "n_i", "w", "phi", "q_surf")}
    assert np.any(before["q_surf"])                      # 誘電体ブロックが帯電している (prompts/129)
    wall = dict(sim.wall)
    sim._max_substeps = 1
    with pytest.raises(ValueError, match="上限 1 "):
        sim.step()
    assert sim.step_count == 1 and len(sim.history["step"]) == 1
    for k, arr in before.items():
        assert np.array_equal(getattr(sim, k), arr), k
    assert sim.wall == wall


def test_stop_between_substeps_rolls_back_device_state_and_continues_identically():
    """サブステップの合間の停止要求: デバイスの状態 (壁損失・生成の積算を含む) をステップ開始時に戻して
    None を返し、続きは中断なしの実行とビット一致する (決定的な集約・密な直接法の Poisson)。"""
    p = _project(init_density_m3=1e15, gas_pressure_pa=50.0, dt=1e-9)
    sim = GpuCartesianFluid2dSimulation(p.model_copy(deep=True))
    ref = GpuCartesianFluid2dSimulation(p.model_copy(deep=True))
    assert sim._g.poisson.direct
    sim.step()
    ref.step()
    keys = ("n_e", "n_i", "w", "phi", "q_surf")          # q_surf: 誘電体の表面電荷 (prompts/129)
    before = {k: getattr(sim, k).copy() for k in keys}
    assert np.any(before["q_surf"])
    for k in keys:                                       # ref も同じく読んで (ホスト → デバイスの戻し) 揃える
        getattr(ref, k)
    n_checks = 0

    def stop_at_third_check():
        nonlocal n_checks
        n_checks += 1
        return n_checks >= 3

    assert sim.step(stop_at_third_check) is None
    assert n_checks == 3 and sim.step_count == 1 and len(sim.history["t"]) == 1
    for k in keys:
        assert np.array_equal(getattr(sim, k), before[k]), k
    sim.step()
    ref.step()
    for k in keys:
        assert np.array_equal(getattr(sim, k), getattr(ref, k)), k
    assert sim.wall == ref.wall and sim.gen_total == ref.gen_total


def test_run_batch_stop_request_within_a_step_on_gpu():
    """run_batch の停止要求がステップの途中 (サブステップの合間) でも効き、履歴は完了したステップだけ。"""
    sim = GpuCartesianFluid2dSimulation(_project(init_density_m3=1e15, gas_pressure_pa=50.0, dt=1e-9, n_steps=5))
    calls = 0

    def stop_soon():
        nonlocal calls
        calls += 1
        return calls > 4                                  # 1 ステップ目の途中で停止
    hist, _ = sim.run_batch(should_stop=stop_soon, store_frames=False)
    assert sim.step_count == 0 and hist["t"] == []
    assert np.all(np.isfinite(sim.n_e))


# ---- 合成格子 (AMR) 版 ------------------------------------------------------------------


def test_amr_matches_cpu_direct_poisson():
    """小さい合成格子 (Poisson は密な逆行列): 状態・履歴・時間平均・反復数が CPU の AMR 版と一致。"""
    cpu, gpu = _run_pair(_project(amr=AMR1, n_steps=200, avg_steps=100, phase_bins=4), amr=True)
    assert isinstance(gpu._g.poisson, gpu_mod._AmrPoisson) and gpu._g.poisson.direct
    assert gpu.n_nodes == cpu.n_nodes > 861
    _assert_same(cpu, gpu)
    for k in ("phi", "e_abs", "n_e", "t_e"):
        assert _rel(cpu.fields[k], gpu.fields[k]) < 1e-9, k
    assert gpu.warnings == cpu.warnings == []


def test_amr_matches_cpu_with_amg_pcg_poisson():
    """未知数 4096 超の合成格子 (GPU の AMG-PCG を同期なしで積む。CPU は LU)。"""
    cpu, gpu = _run_pair(_project(size=0.25e-3, amr={"max_level": 2, "buffer_cells": 2}, n_steps=30), amr=True)
    assert not gpu._g.poisson.direct and gpu.poisson_iters > 0
    _assert_same(cpu, gpu, tol=1e-8)


def test_amr_singular_poisson_on_gpu():
    """Dirichlet の無い合成格子 (特異): GPU の AMG-PCG で解け、全量・電位差が CPU (pyamg) と合う。"""
    amr = {"max_level": 1, "regions": [{"p1": [0.006, 0.002], "p2": [0.012, 0.008], "level": 1}]}
    p = _project([], [], size=1e-3, amr=amr, n_steps=30)
    cpu, gpu = _run_pair(p, amr=True)
    assert gpu._g.poisson.solver.singular
    for k in ("n_e_total", "n_i_total"):
        assert _rel(cpu.history[k], gpu.history[k]) < 1e-6, k
    dphi_c = cpu.phi - cpu.phi.mean()
    dphi_g = gpu.phi - gpu.phi.mean()
    assert _rel(dphi_c, dphi_g) < 1e-6


# ---- 振り分け・server -------------------------------------------------------------------


def test_factory_routes_by_size_and_solver(monkeypatch):
    big = _project(size=0.2e-3)                          # 5151 節点
    assert isinstance(make_fluid2d_simulation(big), GpuCartesianFluid2dSimulation)
    small = _project(size=1e-3)                          # 231 節点
    assert type(make_fluid2d_simulation(small)) is CartesianFluid2dSimulation
    direct = _project(size=0.2e-3, linear_solver="direct")
    assert type(make_fluid2d_simulation(direct)) is CartesianFluid2dSimulation
    assert type(make_fluid2d_simulation(big, explicit=True)) is CartesianFluid2dSimulation
    amr = _project(amr=AMR1)                             # 葉セル 1000 以上の合成格子
    assert type(make_fluid2d_simulation(amr)) is GpuAmrFluid2dSimulation
    assert type(make_fluid2d_simulation(amr, explicit=True)) is AmrFluid2dSimulation
    monkeypatch.setenv("ES_SIM_DEVICE", "cpu")
    assert type(make_fluid2d_simulation(big)) is CartesianFluid2dSimulation
    assert type(make_fluid2d_simulation(amr)) is AmrFluid2dSimulation


def test_ws_fluid2d_runs_gpu_version(monkeypatch):
    from fastapi.testclient import TestClient

    import es_sim.server as server

    monkeypatch.setattr(gpu_mod, "GPU_MIN_NODES", 0)
    server._last_simfluid2d = None
    client = TestClient(server.app)
    proj = _project(size=1e-3, n_steps=20, frame_every=10, avg_steps=10)
    with client.websocket_connect("/ws/fluid2d") as ws:
        ws.send_text(json.dumps({"cmd": "start", "project": proj.model_dump(mode="json")}))
        started = ws.receive_json()
        assert started["type"] == "started" and len(started["mesh"]["nodes"]) == 231
        while True:
            msg = ws.receive_json()
            assert msg["type"] != "error", msg.get("detail")
            if msg["type"] == "done":
                break
        assert len(msg["result"]["history"]["t"]) == 20
    assert isinstance(server._last_simfluid2d, GpuCartesianFluid2dSimulation)
    server._last_simfluid2d = None
