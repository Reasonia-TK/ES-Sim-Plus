"""阻止コンデンサ (自己バイアス、prompts/134) の流体 2D (v2 の一様格子・AMR、v1 の三角形メッシュ) のテスト。

1. 電極の容量: 平行平板 (平面) は ε0·H/L、軸対称の導体は静電場 (/solve) の容量と一致する。LU と GMG で同じ。
2. 真空の分圧 (伝導電流 0): V_e = bias + V_s·C_b/(C_b + C)。
3. 電荷の恒等式: 毎サブステップ Q_N = Q_e(φ) + C_b (V_e − V_s)。
4. 電荷の保存: 全ての電極にコンデンサを付けると、プラズマの電荷 + 誘電体の表面電荷 + Σ Q_N が一定
   (導体の表面・外周の辺・電極に接した誘電体の縁の伝導電流が漏れなく電極へ入る)。
5. C_b → ∞ で直結と一致する。細長い矩形 (上下 symmetry) の自己バイアスが流体 1D と一致する。
6. 続き・サブステップの合間の停止 (回路の状態の巻き戻し)・結果とフレーム。
7. GPU 版が CPU 版と丸め誤差で一致する (CUDA が無ければ skip)。
8. AMR (合成格子) と v1 (三角形メッシュ): 電極の容量が静電場 (AMR・v1 の /solve) と一致し、電荷が保存され、
   3 つの実装の電極の電位がそろう。AMR の GPU 版も CPU 版と一致する。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import es_sim.gfluid.simulation as gsim
from es_sim.device import cuda_available
from es_sim.fem import EPS0
from es_sim.field.electrostatic import solve_electrostatic
from es_sim.fluid1d import Fluid1dSimulation
from es_sim.fluid2d import build_fluid2d_result
from es_sim.gfluid import CartesianFluid2dSimulation
from es_sim.particles import QE
from es_sim.schema import (
    BlockingCapacitor1d,
    Domain,
    Fluid1dSettings,
    Geometry,
    MeshSettings,
    Pic1dElectrode,
    Project,
    VoltageRF,
)

F0 = 13.56e6
W, H = 0.02, 0.01
RF = {"amplitude": 100.0, "freq_hz": F0}
PIN = {"id": "pin", "type": "conductor", "voltage": 0.0, "see_gamma": 0.05,
       "shape": {"kind": "circle", "center": [0.0137, 0.0068], "radius": 0.0011}}
# 左の電極 (x = 0) に接した誘電体: その縁の小片の一部は電極の固定節点が受け持つ (電極へ流れる扱い)
CORNER = {"id": "corner", "type": "dielectric", "eps_r": 4.0, "see_gamma": 0.1,
          "polygon": [[0.0, 0.0], [0.0031, 0.0], [0.0031, 0.0029], [0.0, 0.0029]]}


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def _cap(c, bias=0.0):
    return {"capacitance": c, "initial_bias_v": bias}


def _project(regions=(), left_cap=None, right_cap=None, coord="xy", w=W, h=H, size=0.5e-3, rf=RF, **fluid):
    f = {"init_density_m3": 5e14, "init_te_ev": 3.0, "gas_pressure_pa": 30.0, "n_steps": 20, "frame_every": 10**9}
    f.update(fluid)
    return Project.model_validate({
        "coord": coord,
        "geometry": {
            "domain": {"polygon": [[0, 0], [w, 0], [w, h], [0, h]]},
            "boundaries": [
                {"edges": [3], "type": "dirichlet", "voltage": 0.0, "voltage_rf": rf, "see_gamma": 0.05,
                 "blocking_capacitor": left_cap},
                {"edges": [1], "type": "dirichlet", "voltage": 0.0, "see_gamma": 0.05,
                 "blocking_capacitor": right_cap},
                {"edges": [0, 2], "type": "symmetry"},
            ],
            "regions": list(regions),
        },
        "mesh": {"size": size, "mode": "cartesian"},
        "fluid2d": f,
    })


def _rz_project(cap=_cap(5e-9), size=0.5e-3, **fluid):
    """軸対称 (左辺が軸): 円板の電極と接地した容器 (GEC サンプルを小さくした形)。"""
    f = {"init_density_m3": 1.0, "gas_pressure_pa": 20.0, "n_steps": 1}
    f.update(fluid)
    return Project.model_validate({
        "coord": "rz_x0",
        "geometry": {
            "domain": {"polygon": [[0, 0], [0.05, 0], [0.05, 0.04], [0, 0.04]]},
            "regions": [{"id": "rf", "type": "conductor", "voltage": 1.0, "voltage_rf": RF,
                         "polygon": [[0, 0.01], [0.02, 0.01], [0.02, 0.015], [0, 0.015]],
                         "blocking_capacitor": cap}],
            "boundaries": [{"edges": [0, 1, 2], "type": "dirichlet", "voltage": 0.0},
                           {"edges": [3], "type": "symmetry"}],
        },
        "mesh": {"size": size, "mode": "cartesian"},
        "fluid2d": f,
    })


def _charge_now(sim, phi):
    """今の密度と φ での電極の電荷 (_cap_charges を解き直した φ で)。"""
    q = QE * np.asarray(sim.graph.charge_map @ (sim.n_i - sim.n_e)).ravel()
    b_q = sim._q_static + q + (sim.q_surf / (2.0 * np.pi) if sim.rz else sim.q_surf)
    return sim._cap_charges(phi, sim._v_now, b_q)


# ---- 1〜2. 容量・真空の分圧 ------------------------------------------------------------------------


def test_parallel_plate_capacitance_and_vacuum_divider():
    cb = 1e-12
    proj = _project(left_cap=_cap(cb, -20.0), w=0.02, h=1e-3, size=1e-3, init_density_m3=1.0)
    sim = CartesianFluid2dSimulation(proj, device="cpu")
    c = sim.circuit.c[0, 0]
    assert c == pytest.approx(EPS0 * 1e-3 / 0.02, rel=1e-12)
    sim.debug_reflective_walls = True
    for _ in range(100):
        sim.step()
        expected = -20.0 + sim.circuit.v_src[0] * cb / (cb + c)
        assert sim.circuit.v[0] == pytest.approx(expected, abs=1e-5)


def test_conductor_capacitance_matches_static_solve_rz(monkeypatch):
    """軸対称の円板の電極: ψ から求めた容量が静電場 (/solve) の容量と一致する (LU と GMG のどちらでも)。"""
    proj = _rz_project()
    static = solve_electrostatic(proj).capacitance
    lu = CartesianFluid2dSimulation(proj, device="cpu")
    assert lu._lu is not None
    assert lu.circuit.c[0, 0] == pytest.approx(static, rel=1e-9)
    monkeypatch.setattr(gsim, "DIRECT_MAX", 0)
    mg = CartesianFluid2dSimulation(proj.model_copy(deep=True), device="cpu")
    assert mg._lu is None
    assert mg.circuit.c[0, 0] == pytest.approx(static, rel=1e-7)


# ---- 3〜4. 電荷の恒等式・保存 -------------------------------------------------------------------------


@pytest.mark.parametrize("direct", [True, False])
def test_charge_identity_every_substep(direct, monkeypatch):
    if not direct:
        monkeypatch.setattr(gsim, "DIRECT_MAX", 0)
    proj = _project([PIN, CORNER], left_cap=_cap(1e-10, 5.0), init_density_m3=1e15, gas_pressure_pa=50.0)
    sim = CartesianFluid2dSimulation(proj, device="cpu")
    assert (sim._lu is not None) == direct
    solve_phi = sim._solve_phi
    worst = []

    def checked(t):
        phi = solve_phi(t)
        r = sim.circuit.residual(_charge_now(sim, phi))
        worst.append(float(np.max(np.abs(r))) / float(np.max(np.abs(sim.circuit.q_node))))
        return phi

    sim._solve_phi = checked
    for _ in range(150):
        sim.step()
    assert max(worst) < (1e-10 if direct else 1e-7)


def test_total_charge_is_conserved_when_every_electrode_has_a_capacitor():
    """プラズマの電荷 + 誘電体の表面電荷 + Σ Q_N が一定 (伝導電流の行き先の漏れが無い)。

    輸送は直接法 (反復法の収束判定の分だけ粒子の収支がずれるのを避ける)。誘電体の縁の小片の分 (全体の
    数 %) を落とすと、壁へ流れた電荷の 1e-2 程度ずれる。
    """
    proj = _project([PIN, CORNER], left_cap=_cap(1e-10), right_cap=_cap(3e-10),
                    init_density_m3=1e15, gas_pressure_pa=50.0, linear_solver="direct")
    proj.geometry.regions[0].blocking_capacitor = proj.geometry.boundaries[0].blocking_capacitor.model_copy()
    sim = CartesianFluid2dSimulation(proj, device="cpu")
    assert sim.circuit.labels == ["edge3", "edge1", "pin"]
    pieces, _loc, _w, elec_of = sim._cap_ends
    elec = np.bincount(elec_of, minlength=3)
    assert np.all(elec > 0)
    # 左の電極に接した誘電体の縁: 誘電体の小片のうち電極へ流れるもの
    g = sim.graph
    diel_to_left = g.wall_dielectric[pieces] & (elec_of == 0)
    assert np.any(diel_to_left)

    def total():
        a = sim.active_idx
        plasma = QE * float(np.sum((sim.n_i[a] - sim.n_e[a]) * sim.node_vol))
        return plasma + float(sim.q_surf.sum()) + float(np.sum(sim.circuit.q_node))

    sim.step()
    t0 = total()
    for _ in range(300):
        sim.step()
    lost = sim.wall["ion"] * QE  # 比べる大きさ (壁へ流れたイオンの電荷)
    assert abs(total() - t0) < 1e-9 * lost


# ---- 5. 極限・流体 1D -------------------------------------------------------------------------------


def test_large_capacitance_matches_direct_coupling_2d():
    direct = CartesianFluid2dSimulation(_project([PIN, CORNER], init_density_m3=1e15), device="cpu")
    cap = CartesianFluid2dSimulation(_project([PIN, CORNER], left_cap=_cap(1e6), init_density_m3=1e15), device="cpu")
    for _ in range(200):
        direct.step()
        cap.step()
    assert np.max(np.abs(direct.phi - cap.phi)) < 1e-8
    assert np.max(np.abs(direct.n_e - cap.n_e)) < 1e-9 * np.max(direct.n_e)


def test_self_bias_matches_fluid1d_in_a_strip():
    """細長い矩形 (上下 symmetry) の自己バイアスが、容量を面積あたりにした流体 1D と一致する (EAE、θ = 0°)。

    v2 の流体 2D は双対セルの体積、流体 1D は半セルの端を使うので、もともと数 % ずれる (test_v2_gfluid の
    断面平均の比較と同じ)。
    """
    gap, n_cells = 0.02, 20
    h = gap / n_cells
    rf = [VoltageRF(amplitude=100.0, freq_hz=F0, phase_deg=90.0), VoltageRF(amplitude=100.0, freq_hz=2 * F0, phase_deg=90.0)]
    n_steps = 6 * 2000
    c1d = 1e-8                        # [F/m²]、2D は奥行きあたり c1d·h [F/m]
    s1 = Fluid1dSettings(
        gap_m=gap, n_cells=n_cells,
        left=Pic1dElectrode(v_dc=0.0, voltage_rf=rf, see_gamma=0.05, blocking_capacitor=BlockingCapacitor1d(capacitance=c1d)),
        right=Pic1dElectrode(v_dc=0.0, see_gamma=0.05),
        init_density_m3=1e15, init_te_ev=2.0, gas_pressure_pa=50.0, gas_temperature_k=300.0,
        n_steps=n_steps, frame_every=n_steps, avg_steps=1000,
    )
    dummy = Geometry(domain=Domain(polygon=[(0, 0), (1, 0), (1, 1), (0, 1)]))
    sim1 = Fluid1dSimulation(Project(geometry=dummy, mesh=MeshSettings(size=0.1), fluid1d=s1))
    sim1.run_batch(store_frames=False)
    proj = _project(left_cap=_cap(c1d * h), w=gap, h=h, size=h, rf=[r.model_dump() for r in rf],
                    init_density_m3=1e15, init_te_ev=2.0, gas_pressure_pa=50.0, n_steps=n_steps, avg_steps=1000)
    sim2 = CartesianFluid2dSimulation(proj, device="cpu")
    assert (sim2.grid.nx, sim2.grid.ny) == (20, 1)
    assert sim2.circuit.c[0, 0] / h == pytest.approx(sim1.circuit.c[0, 0], rel=1e-12)
    sim2.run_batch(store_frames=False)
    v1 = np.array([r[0] for r in sim1.circuit.history["v_dc"]])
    v2 = np.array([r[0] for r in sim2.circuit.history["v_dc"]])
    assert len(v1) == len(v2) == 6
    assert v1[-1] < -10.0                    # θ = 0° は負の自己バイアス (EAE。6 周期ではまだ落ち着く途中)
    assert np.max(np.abs(v2 - v1)) < 0.05 * abs(v1[-1])


# ---- 6. 続き・途中停止・結果 ---------------------------------------------------------------------------


def test_stop_between_substeps_rolls_back_circuit():
    proj = _project([PIN, CORNER], left_cap=_cap(1e-10), init_density_m3=1e15, gas_pressure_pa=50.0, dt=1e-9)
    sim = CartesianFluid2dSimulation(proj, device="cpu")
    ref = CartesianFluid2dSimulation(proj.model_copy(deep=True), device="cpu")
    sim.step()
    ref.step()
    q0, v0 = sim.circuit.q_node.copy(), sim.circuit.v.copy()
    n_checks = 0

    def stop_at_third_check():
        nonlocal n_checks
        n_checks += 1
        return n_checks >= 3

    assert sim.step(stop_at_third_check) is None
    assert np.array_equal(sim.circuit.q_node, q0) and np.array_equal(sim.circuit.v, v0)
    for _ in range(3):
        sim.step()
        ref.step()
    assert np.array_equal(sim.phi, ref.phi)
    assert np.array_equal(sim.circuit.q_node, ref.circuit.q_node)
    assert sim.circuit.history == ref.circuit.history


def test_continue_result_and_frames_rz():
    proj = _rz_project(cap=_cap(5e-10, -10.0), size=1e-3, init_density_m3=1e15, init_te_ev=3.0,
                       gas_pressure_pa=50.0, n_steps=2500, avg_steps=500)
    proj.fluid2d.frame_every = 500
    full = CartesianFluid2dSimulation(proj, device="cpu")
    _, frames = full.run_batch()
    part_proj = proj.model_copy(deep=True)
    part_proj.fluid2d.n_steps = 1700
    part = CartesianFluid2dSimulation(part_proj, device="cpu")
    part.run_batch(store_frames=False)
    part.prepare_continue(extra_steps=800, avg_steps=500)
    part.run_batch(store_frames=False)
    assert np.array_equal(full.phi, part.phi)
    assert np.array_equal(full.circuit.q_node, part.circuit.q_node)
    assert full.circuit.history["v_dc"][-1:] == part.circuit.history["v_dc"]   # 2000 ステップ目で閉じる 1 周期
    assert frames[-1]["circuit"][0]["label"] == "rf"
    res = build_fluid2d_result(full, elapsed_s=1.0)["circuit"]
    assert res["units"] == {"capacitance": "F", "charge": "C", "current": "A"}
    e = res["electrodes"][0]
    assert e["capacitance"] == 5e-10 and e["initial_bias_v"] == -10.0
    assert e["c_self"] == pytest.approx(full.circuit.c[0, 0]) and 0.0 < e["rf_division"] < 1.0
    assert len(e["v_dc"]) == 1 and e["last_period"] is not None


# ---- 7. GPU ------------------------------------------------------------------------------------------


@pytest.mark.skipif(not cuda_available(), reason="CUDA が使えません")
def test_gpu_matches_cpu_with_two_capacitors():
    from es_sim.gfluid import GpuCartesianFluid2dSimulation

    pin = dict(PIN, blocking_capacitor=_cap(2e-11, -5.0))
    # dt を RF 周期の 1/200 にして、300 ステップで周期の集計も比べる
    proj = _project([pin, CORNER], left_cap=_cap(1e-10), init_density_m3=5e14, n_steps=300, dt=1.0 / (F0 * 200))
    cpu = CartesianFluid2dSimulation(proj.model_copy(deep=True), device="cpu")
    gpu = GpuCartesianFluid2dSimulation(proj.model_copy(deep=True))
    cpu.run_batch(store_frames=False)
    gpu.run_batch(store_frames=False)

    def rel(a, b):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(a))), 1e-300))

    assert cpu.circuit.labels == gpu.circuit.labels == ["edge3", "pin"]
    assert rel(cpu.circuit.c, gpu.circuit.c) < 1e-8          # ψ: CPU は LU、GPU は GMG
    for k in ("n_e", "n_i", "phi", "q_surf"):
        assert rel(getattr(cpu, k), getattr(gpu, k)) < 1e-8, k
    assert rel(cpu.circuit.q_node, gpu.circuit.q_node) < 1e-8
    assert rel(cpu.circuit.v, gpu.circuit.v) < 1e-8
    assert len(cpu.circuit.history["v_dc"]) == 1
    assert rel(cpu.circuit.history["v_dc"], gpu.circuit.history["v_dc"]) < 1e-8


# ---- 8. AMR (合成格子) と v1 (三角形メッシュ) ------------------------------------------------------

AMR1 = {"max_level": 1, "buffer_cells": 2}
# AMR・v1 の比較用の導体 (格子に揃った正方形: 三角形メッシュでも同じ形になる)
BOX = {"id": "pin", "type": "conductor", "voltage": 1.0, "see_gamma": 0.05,
       "polygon": [[0.012, 0.004], [0.015, 0.004], [0.015, 0.007], [0.012, 0.007]]}


def _variant(kind: str, regions, **kw) -> Project:
    """kind: "uniform" (v2 の一様格子)・"amr" (v2 の合成格子)・"v1" (構造格子の三角形メッシュ)。"""
    p = _project(regions, **kw)
    if kind == "amr":
        p.mesh.amr = AMR1
    if kind == "v1":
        p.mesh.mode = "structured"
    return Project.model_validate(p.model_dump())


def _make(kind: str, p: Project):
    from es_sim.fluid2d import Fluid2dSimulation
    from es_sim.gfluid.amr import AmrFluid2dSimulation

    if kind == "v1":
        return Fluid2dSimulation(p)
    return (AmrFluid2dSimulation if kind == "amr" else CartesianFluid2dSimulation)(p, device="cpu")


@pytest.mark.parametrize("kind", ["amr", "v1"])
def test_amr_and_v1_capacitance_match_static_solvers(kind):
    """ψ から求めた電極の容量が、それぞれの静電場の解法 (AMR の合成格子・v1 の FEM) の容量と一致する。"""
    from es_sim.amr.electrostatic import solve_electrostatic_amr
    from es_sim.fem import solve as fem_solve
    from es_sim.meshing import generate_mesh

    p = _variant(kind, [dict(BOX, blocking_capacitor=_cap(1e-10))], init_density_m3=1.0)
    sim = _make(kind, p)
    static = solve_electrostatic_amr(p).capacitance if kind == "amr" else fem_solve(p, generate_mesh(p)).capacitance
    assert sim.circuit.c[0, 0] == pytest.approx(static, rel=1e-9)


@pytest.mark.parametrize("kind", ["amr", "v1"])
def test_amr_and_v1_conserve_charge_with_capacitors(kind):
    """全ての電極 (左右の辺・導体) にコンデンサ: プラズマの電荷 + 表面電荷 + Σ Q_N が一定 (輸送は直接法)。"""
    p = _variant(kind, [dict(BOX, blocking_capacitor=_cap(1e-10)), CORNER], left_cap=_cap(1e-10),
                 right_cap=_cap(3e-10), init_density_m3=1e15, gas_pressure_pa=50.0, linear_solver="direct")
    sim = _make(kind, p)
    assert sim.circuit.labels == ["edge3", "edge1", "pin"]
    assert np.all(np.bincount(sim._cap_ends[3], minlength=3) > 0)

    def total():
        a = sim.active_idx
        plasma = QE * float(np.sum((sim.n_i[a] - sim.n_e[a]) * sim.node_vol))
        return plasma + float(sim.q_surf.sum()) + float(np.sum(sim.circuit.q_node))

    sim.step()
    t0 = total()
    for _ in range(150):
        sim.step()
    assert abs(total() - t0) < 1e-9 * sim.wall["ion"] * QE


def test_uniform_amr_and_v1_agree():
    """同じ形 (格子に揃った導体・誘電体) で、3 つの実装の電極の電位 (回路が決めたもの) がそろう。"""
    vs = {}
    for kind in ("uniform", "amr", "v1"):
        p = _variant(kind, [dict(BOX, blocking_capacitor=_cap(1e-10)), CORNER], left_cap=_cap(1e-10),
                     init_density_m3=1e15, gas_pressure_pa=50.0)
        sim = _make(kind, p)
        for _ in range(200):
            sim.step()
        vs[kind] = sim.circuit.v
    scale = max(float(np.max(np.abs(v))) for v in vs.values())
    for kind in ("amr", "v1"):
        assert np.max(np.abs(vs[kind] - vs["uniform"])) < 0.03 * scale, (kind, vs)


@pytest.mark.skipif(not cuda_available(), reason="CUDA が使えません")
def test_gpu_amr_matches_cpu_amr_with_capacitors():
    from es_sim.gfluid.amr import AmrFluid2dSimulation, GpuAmrFluid2dSimulation

    p = _variant("amr", [dict(BOX, blocking_capacitor=_cap(2e-11, -5.0)), CORNER], left_cap=_cap(1e-10),
                 init_density_m3=5e14, n_steps=300, dt=1.0 / (F0 * 200))
    cpu = AmrFluid2dSimulation(p.model_copy(deep=True), device="cpu")
    gpu = GpuAmrFluid2dSimulation(p.model_copy(deep=True))
    cpu.run_batch(store_frames=False)
    gpu.run_batch(store_frames=False)

    def rel(a, b):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(a))), 1e-300))

    assert rel(cpu.circuit.c, gpu.circuit.c) < 1e-10
    for k in ("n_e", "n_i", "phi", "q_surf"):
        assert rel(getattr(cpu, k), getattr(gpu, k)) < 1e-8, k
    assert rel(cpu.circuit.q_node, gpu.circuit.q_node) < 1e-8
    assert len(cpu.circuit.history["v_dc"]) == 1
    assert rel(cpu.circuit.history["v_dc"], gpu.circuit.history["v_dc"]) < 1e-8
