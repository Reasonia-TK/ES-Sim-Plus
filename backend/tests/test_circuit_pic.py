"""阻止コンデンサ (自己バイアス、prompts/134) の PIC のテスト: PIC 1D と v1 PIC (三角形メッシュ)。

PIC は電極の電荷を Green の相反定理 (誘導電荷 −ψ̃·q) で求め、Poisson の前に電極の電位を決める
(circuit.BlockingCircuit.solve_induced)。

1. 真空の分圧: V_e = initial_bias + V_s·C_b/(C_b + C)。C は 1D で ε0/L、2D で静電場 (/solve) の容量。
2. 電荷の恒等式: 毎ステップ Q_N = Q_e(φ) + C_b (V_e − V_s)。Q_e は解いた φ の電束 (1D)・残差 (2D) の形で、
   相反定理の形と丸め誤差で一致する。
3. 電荷の保存: 全ての電極にコンデンサを付けると Σ Q_N + 粒子 + 誘電体の表面電荷が丸め誤差で一定。
4. C_b → ∞ で直結と一致する (短い計算。PIC はカオス的なので長くは比べない)。
5. 続き実行はビット一致し、コンデンサの状態を引き継ぐ。結果とフレームに circuit が載る。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from es_sim.batch import _build_results_bundle, self_bias_metrics
from es_sim.fem import EPS0, assemble
from es_sim.fem import solve as fem_solve
from es_sim.pic import PicSimulation
from es_sim.pic1d import Pic1dSimulation, build_pic1d_result, electrode_voltage
from es_sim.pic1d_presets import edupic_ar_processes
from es_sim.schema import (
    BlockingCapacitor1d,
    Domain,
    Geometry,
    MeshSettings,
    Pic1dElectrode,
    Pic1dSettings,
    Project,
    VoltageRF,
)

F0 = 13.56e6
GAP = 0.025


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def _mcc(seed: int = 0) -> dict:
    e_procs, i_procs = edupic_ar_processes()
    return {"gas": {"name": "Ar", "pressure_pa": 10.0, "temperature_k": 350.0},
            "electron_processes": [p.model_dump() for p in e_procs],
            "ion_processes": [p.model_dump() for p in i_procs], "seed": seed}


# ---- PIC 1D ---------------------------------------------------------------------------------------


def _project_1d(left=None, right=None, dens=1e15, mcc=True, n_macro=4000, gamma=0.1, **kw) -> Project:
    """eduPIC の Ar 条件 (10 Pa・ギャップ 2.5 cm) に左 250 V・13.56 MHz。両側の γ = 0.1。"""
    s = Pic1dSettings(
        gap_m=GAP, n_cells=128,
        left=Pic1dElectrode(v_dc=0.0, voltage_rf=VoltageRF(amplitude=250.0, freq_hz=F0), see_gamma=gamma,
                            blocking_capacitor=left),
        right=Pic1dElectrode(v_dc=0.0, see_gamma=gamma, blocking_capacitor=right),
        init_density_m3=dens, init_te_ev=2.0, init_ti_ev=0.03, ion_mass_amu=39.948, n_macro=n_macro,
        dt=1.0 / (400.0 * F0), n_steps=100, frame_every=1_000_000, phase_bins=0,
        mcc=_mcc() if mcc else None, **kw,
    )
    return Project(geometry=Geometry(domain=Domain(polygon=[(0, 0), (1, 0), (1, 1), (0, 1)])),
                   mesh=MeshSettings(size=0.1), pic1d=s)


def _total_charge_1d(sim: Pic1dSimulation) -> float:
    q = sum(sp.q * float(sp.w.sum()) for sp in sim.species.values())
    return q + float(np.sum(sim.circuit.q_node))


def test_vacuum_divider_pic1d():
    cb = 1.0e-8
    sim = Pic1dSimulation(_project_1d(BlockingCapacitor1d(capacitance=cb, initial_bias_v=-20.0), dens=1e3,
                                      mcc=False, n_macro=10))
    c = EPS0 / GAP
    assert sim.circuit.c[0, 0] == pytest.approx(c, rel=1e-12)
    for _ in range(300):
        t = sim.t
        phi = sim.step()
        expected = -20.0 + electrode_voltage(sim.s.left, t) * cb / (cb + c)
        assert phi[0] == pytest.approx(expected, abs=1e-8)
        assert phi[0] == sim.circuit.v[0]


def test_charge_identity_and_conservation_pic1d():
    """両側にコンデンサ (MCC・二次電子あり): 恒等式は電束の形で丸め誤差、粒子 + Q_N の電荷は保存。"""
    sim = Pic1dSimulation(_project_1d(BlockingCapacitor1d(capacitance=1.0e-7, initial_bias_v=5.0),
                                      BlockingCapacitor1d(capacitance=3.0e-7)))
    solve = sim._solve_phi
    worst = []

    def checked(f_dep, t):
        phi = solve(f_dep, t)
        r = sim.circuit.residual(sim.electrode_charges(phi, f_dep))
        worst.append(float(np.max(np.abs(r))) / float(np.max(np.abs(sim.circuit.q_node))))
        return phi

    sim._solve_phi = checked
    q0 = _total_charge_1d(sim)
    for _ in range(1500):
        sim.step()
    assert max(worst) < 1e-11
    absorbed = sum(sum(w.values()) for w in sim.wall.values()) * 1.602176634e-19
    assert absorbed > 1e-7 and sim.see_events > 0.0
    assert abs(_total_charge_1d(sim) - q0) < 1e-13 * absorbed


def test_fn_emission_charges_the_electrode_pic1d():
    """FN 電界放出の電子の分 +e·w が電極の Q_N に入り、粒子 + Q_N の電荷が保存する (ギャップ 2.5 mm・−3 kV)。"""
    s = Pic1dSettings(
        gap_m=0.0025, n_cells=64,
        left=Pic1dElectrode(v_dc=-3000.0, fn={"phi_ev": 4.5, "beta": 2500.0, "init_energy_ev": 0.5, "macro_weight": 1e9},
                            blocking_capacitor=BlockingCapacitor1d(capacitance=1e-5)),
        right=Pic1dElectrode(v_dc=0.0, blocking_capacitor=BlockingCapacitor1d(capacitance=1e-5)),
        init_density_m3=1e12, init_te_ev=1.0, n_macro=200, dt=1e-12, n_steps=10, frame_every=1_000_000, phase_bins=0,
    )
    sim = Pic1dSimulation(Project(geometry=Geometry(domain=Domain(polygon=[(0, 0), (1, 0), (1, 1), (0, 1)])),
                                  mesh=MeshSettings(size=0.1), pic1d=s))
    sim.step()
    q0 = _total_charge_1d(sim)
    for _ in range(60):
        sim.step()
    emitted = sim.fn_total_w["left"] * 1.602176634e-19
    assert emitted > 1e-6
    assert abs(_total_charge_1d(sim) - q0) < 1e-12 * emitted


def test_large_capacitance_matches_direct_coupling_pic1d():
    direct = Pic1dSimulation(_project_1d())
    cap = Pic1dSimulation(_project_1d(BlockingCapacitor1d(capacitance=1.0e6)))
    for _ in range(200):
        p_direct = direct.step()
        p_cap = cap.step()
    assert np.max(np.abs(p_direct - p_cap)) < 1e-6
    np.testing.assert_allclose(direct.species["electron"].x, cap.species["electron"].x, rtol=0, atol=1e-9)


def test_continue_and_result_pic1d():
    cap = BlockingCapacitor1d(capacitance=1.0e-7, initial_bias_v=-10.0)
    full = Pic1dSimulation(_project_1d(cap, n_macro=2000))
    full.s.n_steps = 1000
    full.run_batch(store_frames=False)
    part = Pic1dSimulation(_project_1d(cap, n_macro=2000))
    part.s.n_steps = 450
    part.s.frame_every = 150
    _, frames = part.run_batch()
    assert frames[-1]["circuit"][0]["label"] == "left"
    part.prepare_continue(extra_steps=550)
    part.run_batch(store_frames=False)
    np.testing.assert_array_equal(full.species["electron"].x, part.species["electron"].x)
    np.testing.assert_array_equal(full.circuit.q_node, part.circuit.q_node)
    # 1 周期 = 400 ステップ: 450 ステップ目は 2 周期目の途中。続きで 2 周期目が閉じる
    assert full.circuit.history["t"][-1:] == part.circuit.history["t"]
    assert full.circuit.history["v_dc"][-1:] == part.circuit.history["v_dc"]
    bundle = build_pic1d_result(part, elapsed_s=1.0)
    res = bundle["circuit"]
    assert res["units"]["capacitance"] == "F/m^2" and res["period_s"] == pytest.approx(1.0 / F0)
    e = res["electrodes"][0]
    assert e["label"] == "left" and e["c_self"] == pytest.approx(EPS0 / GAP) and len(e["v_dc"]) == 1
    # スイープのケースの一覧に載せる要約 (batch.self_bias_metrics)
    assert self_bias_metrics({"version": 1, "pic1d": bundle}) == [
        {"label": "left", "v_dc": e["v_dc"][-1], "v1": e["v1"][-1]}
    ]
    plain = Pic1dSimulation(_project_1d(n_macro=100))
    plain.step()
    assert build_pic1d_result(plain, elapsed_s=0.0)["circuit"] is None


# ---- v1 PIC (三角形メッシュ) ------------------------------------------------------------------------

W, H = 0.02, 0.01


def _project_2d(cap_left=None, cap_right=None, cap_pin=None, dens=5e14, coord="xy", mcc=True, n_macro=4000,
                **pic) -> Project:
    """左の辺 (RF 100 V)・右の辺 (0 V)・導体 pin (1 V)・誘電体の角、上下は対称。"""
    p = {"initial_plasma": {"density": dens, "te_ev": 2.0, "ti_ev": 0.03, "seed": 1}, "n_macro": n_macro,
         "dt": 1.0 / (400 * F0), "n_steps": 20, "frame_every": 1_000_000, "phase_bins": 0, "see_energy_ev": 2.0}
    if mcc:
        p["mcc"] = {**_mcc(3), "gas": {"name": "Ar", "pressure_pa": 20.0, "temperature_k": 300.0}}
    p.update(pic)
    return Project.model_validate({
        "coord": coord,
        "geometry": {
            "domain": {"polygon": [[0, 0], [W, 0], [W, H], [0, H]]},
            "boundaries": [
                {"edges": [3], "type": "dirichlet", "voltage": 0.0,
                 "voltage_rf": {"amplitude": 100.0, "freq_hz": F0}, "see_gamma": 0.1, "blocking_capacitor": cap_left},
                {"edges": [1], "type": "dirichlet", "voltage": 0.0, "see_gamma": 0.1, "blocking_capacitor": cap_right},
                {"edges": [0, 2], "type": "symmetry"},
            ],
            "regions": [
                {"id": "pin", "type": "conductor", "voltage": 1.0, "see_gamma": 0.1,
                 "polygon": [[0.012, 0.004], [0.015, 0.004], [0.015, 0.007], [0.012, 0.007]],
                 "blocking_capacitor": cap_pin},
                {"id": "corner", "type": "dielectric", "eps_r": 4.0, "see_gamma": 0.1,
                 "polygon": [[0.0, 0.0], [0.003, 0.0], [0.003, 0.003], [0.0, 0.003]]},
            ],
        },
        "mesh": {"size": 0.6e-3},
        "pic": p,
    })


@pytest.mark.parametrize("coord", ["xy", "rz"])
def test_capacitance_and_vacuum_divider_pic2d(coord):
    """導体の容量が静電場 (/solve) の容量と一致し、真空では容量の分圧になる。"""
    p = _project_2d(cap_pin={"capacitance": 1e-10}, dens=1.0, coord=coord, mcc=False, n_macro=10)
    sim = PicSimulation(p)
    assert sim.circuit.c[0, 0] == pytest.approx(fem_solve(p, sim.mesh).capacitance, rel=1e-12)
    cb = 3e-11 if coord == "xy" else 3e-12
    sim = PicSimulation(_project_2d(cap_left={"capacitance": cb, "initial_bias_v": -20.0}, dens=1.0, coord=coord,
                                    mcc=False, n_macro=10))
    c = sim.circuit.c[0, 0]
    edge_nodes = sim._cap_nodes[0]
    for _ in range(100):
        phi = sim.step()
        expected = -20.0 + sim.circuit.v_src[0] * cb / (cb + c)
        assert sim.circuit.v[0] == pytest.approx(expected, abs=1e-9)
        assert np.all(phi[edge_nodes] == sim.circuit.v[0])


def _caps(coord: str) -> dict:
    s = 1.0 if coord == "xy" else 0.1
    return {"cap_left": {"capacitance": 3e-11 * s, "initial_bias_v": 5.0}, "cap_right": {"capacitance": 1e-10 * s},
            "cap_pin": {"capacitance": 2e-11 * s}}


@pytest.mark.parametrize("coord", ["xy", "rz"])
def test_charge_identity_pic2d(coord):
    """残差の形の電極の電荷 Σ (Kφ − f) で Q_N = Q_e + C_b (V_e − V_s) が丸め誤差で成り立つ。"""
    p = _project_2d(**_caps(coord), coord=coord)
    sim = PicSimulation(p)
    k_full = assemble(p, sim.mesh)[0].tocsr()
    factor = 2 * np.pi if sim.rz else 1.0
    solve = sim._solve_phi
    worst = []

    def checked(f_dep, t):
        phi = solve(f_dep, t)
        f = sim.f_static + (f_dep / (2 * np.pi) if sim.rz else f_dep)
        f = f + (sim.q_surf / (2 * np.pi) if sim.rz else sim.q_surf)
        r = k_full @ phi - f
        q_e = factor * np.array([r[nodes].sum() for nodes in sim._cap_nodes])
        worst.append(float(np.max(np.abs(sim.circuit.residual(q_e)))) / float(np.max(np.abs(sim.circuit.q_node))))
        return phi

    sim._solve_phi = checked
    for _ in range(150):
        sim.step()
    assert sim.circuit.labels == ["edge3", "edge1", "pin"]
    assert max(worst) < 1e-11


@pytest.mark.parametrize("coord", ["xy", "rz"])
def test_charge_conservation_pic2d(coord):
    """全ての電極にコンデンサ: Σ Q_N + 粒子 + 誘電体の表面電荷が丸め誤差で一定 (吸収・二次電子・誘電体あり)。

    電極でも誘電体でもない辺は対称 (反射) なので、壁へ行った電荷は全て Q_N か表面電荷に入る。軸対称では軸
    (y = 0) と外周 (y = H) が対称の辺: 押し出しが回転法になる前は、軸のすぐ近くの粒子が遠心力の項 vθ²/r で
    1 ステップに領域の何倍も飛び、外周で反射しきれずに対称の辺で消えて保存が破れていた。
    """
    sim = PicSimulation(_project_2d(**_caps(coord), coord=coord))

    def total():
        q = sum(sp.q * float(sp.w.sum()) for sp in sim.species.values())
        return q + float(np.sum(sim.circuit.q_node)) + float(sim.q_surf.sum())

    sim.step()
    q0 = total()
    for _ in range(800):
        sim.step()
    w0 = float(sim.species["ion"].w.mean())
    n_abs = sim.species["electron"].wall_absorbed + sim.species["ion"].wall_absorbed
    absorbed = n_abs * w0 * 1.602176634e-19  # 平面は C/m、軸対称は C (リングの重みは小さい)
    assert sim.see_events > 0 and float(sim.q_surf.sum()) != 0.0 and n_abs > 500
    assert abs(total() - q0) < 1e-12 * absorbed
    for sp in sim.species.values():  # どの粒子も領域の中 (walk の許容誤差の分だけ余裕を見る)
        assert np.all((sp.x >= -1e-9) & (sp.x <= [W + 1e-9, H + 1e-9]))


def test_large_capacitance_matches_direct_coupling_pic2d():
    direct = PicSimulation(_project_2d())
    cap = PicSimulation(_project_2d(cap_left={"capacitance": 1e6}))
    for _ in range(100):
        p_direct = direct.step()
        p_cap = cap.step()
    assert np.max(np.abs(p_direct - p_cap)) < 1e-9


def test_result_frame_and_continue_pic2d():
    p = _project_2d(cap_pin={"capacitance": 2e-11, "initial_bias_v": -3.0}, n_macro=1500)
    full = PicSimulation(p)
    full.pic.n_steps = 900
    full.run_batch(store_frames=False)
    part = PicSimulation(_project_2d(cap_pin={"capacitance": 2e-11, "initial_bias_v": -3.0}, n_macro=1500))
    part.pic.n_steps = 500
    part.pic.frame_every = 250
    _, frames = part.run_batch()
    assert frames[-1]["circuit"][0]["label"] == "pin"
    part.prepare_continue(400)
    part.run_batch(store_frames=False)
    np.testing.assert_array_equal(full.circuit.q_node, part.circuit.q_node)
    np.testing.assert_array_equal(full.species["electron"].x, part.species["electron"].x)
    assert full.circuit.history["v_dc"][-1:] == part.circuit.history["v_dc"]
    bundle = _build_results_bundle(part, 500, elapsed_s=1.0)["pic"]
    c = bundle["circuit"]
    assert c["units"]["capacitance"] == "F/m" and c["electrodes"][0]["label"] == "pin"
    assert len(c["electrodes"][0]["v_dc"]) == 1
    plain = PicSimulation(_project_2d(n_macro=100))
    plain.run_batch(store_frames=False)
    assert _build_results_bundle(plain, 0, elapsed_s=0.0)["pic"]["circuit"] is None


def test_capacitor_on_an_edge_after_an_arc():
    """外周に円弧があると境界条件の辺は弦の番号になるが、電極は元の辺の番号のラベル (edge3) で引く。

    容量は静電場で接地した電極 (edge1) に誘導される電荷の符号を変えたものと一致する。
    """
    p = Project.model_validate({
        "geometry": {
            "domain": {"polygon": [[0, 0], [W, 0], [W, H], [0, H]], "bulges": [0, 0, 0.3, 0]},
            "boundaries": [
                {"edges": [3], "type": "dirichlet", "voltage": 1.0, "blocking_capacitor": {"capacitance": 1e-10}},
                {"edges": [1], "type": "dirichlet", "voltage": 0.0},
            ],
        },
        "mesh": {"size": 1e-3},
        "pic": {"dt": 1e-11},
    })
    assert p.geometry.boundaries[0].edges != [3]  # 弦に分けたあとの番号
    sim = PicSimulation(p)
    assert sim.circuit.labels == ["edge3"]
    charges = {label: q for label, _, q in fem_solve(p, sim.mesh).charges}
    assert sim.circuit.c[0, 0] == pytest.approx(-charges["edge1"], rel=1e-12)
