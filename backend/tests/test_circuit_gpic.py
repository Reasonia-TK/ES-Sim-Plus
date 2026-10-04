"""阻止コンデンサ (自己バイアス、prompts/134) の v2 GPU PIC (一様格子・AMR) のテスト。

回路はデバイスで進める (誘導電荷 → 電極の電位をグループ電位へ → Poisson 1 回、境界のカーネルが電極ごとの
吸収・放出の電荷を数える)。周期の集計はホストの BlockingCircuit が履歴を読むときに行う。

1. 容量が静電場 (/solve の直交格子・AMR) と一致し、真空では容量の分圧になる (平面・軸対称)。
2. 電荷の恒等式: 小さい格子 (Poisson は密な逆行列で厳密) で Q_N = Q_e(φ) + C_b (V_e − V_s)。Q_e は解いた φ の
   電束 (一様格子) か合成格子の電荷の汎関数 (AMR) で、相反定理の形と丸め誤差で一致する。
3. 電荷の保存: 全ての電極にコンデンサを付けると Σ Q_N + 粒子 + 誘電体の表面電荷が丸め誤差で一定。
4. 動的再格子化で W と容量行列を作り直す (容量は新しい階層の静電場の値)。Q_N は続き、電荷は保存する。
5. v1 PIC と同じ CCP ストリップ (電気的非対称効果) で、自己バイアスの推移が統計の誤差の範囲で一致する。
6. 結果・フレーム・続き。

GPU (CUDA) が無い環境では skip する。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from es_sim.device import cuda_available
from es_sim.particles import QE
from es_sim.pic1d_presets import edupic_ar_processes
from es_sim.schema import Project

pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA (CuPy) が使えない環境")

W, H = 0.02, 0.01
F0 = 13.56e6
PATCH = {"max_level": 1, "buffer_cells": 2, "regions": [{"p1": [0.010, 0.002], "p2": [0.017, 0.009], "level": 1}]}


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def _mcc() -> dict:
    e_procs, i_procs = edupic_ar_processes()
    return {"gas": {"name": "Ar", "pressure_pa": 20.0, "temperature_k": 300.0},
            "electron_processes": [x.model_dump() for x in e_procs],
            "ion_processes": [x.model_dump() for x in i_procs], "seed": 3}


def _project(cap_left=None, cap_right=None, cap_pin=None, dens=5e14, coord="xy", mcc=True, n_macro=20000,
             amr=None, dielectric=True, **pic) -> Project:
    """左の辺 (RF 100 V)・右の辺 (0 V)・導体 pin (1 V)・誘電体の角、上下は対称 (test_circuit_pic と同じ形)。"""
    p = {"initial_plasma": {"density": dens, "te_ev": 2.0, "ti_ev": 0.03, "seed": 1}, "n_macro": n_macro,
         "dt": 1.0 / (400 * F0), "n_steps": 20, "frame_every": 1_000_000, "phase_bins": 0, "see_energy_ev": 2.0}
    if mcc:
        p["mcc"] = _mcc()
    p.update(pic)
    mesh = {"size": 0.5e-3, "mode": "cartesian"}
    if amr:
        mesh["amr"] = amr
    regions = [{"id": "pin", "type": "conductor", "voltage": 1.0, "see_gamma": 0.1,
                "polygon": [[0.012, 0.004], [0.015, 0.004], [0.015, 0.007], [0.012, 0.007]],
                "blocking_capacitor": cap_pin}]
    if dielectric:
        regions.append({"id": "corner", "type": "dielectric", "eps_r": 4.0, "see_gamma": 0.1,
                        "polygon": [[0.0, 0.0], [0.003, 0.0], [0.003, 0.003], [0.0, 0.003]]})
    return Project.model_validate({
        "coord": coord,
        "geometry": {
            "domain": {"polygon": [[0, 0], [W, 0], [W, H], [0, H]]},
            "boundaries": [
                {"edges": [3], "type": "dirichlet", "voltage": 0.0, "voltage_rf": {"amplitude": 100.0, "freq_hz": F0},
                 "see_gamma": 0.1, "blocking_capacitor": cap_left},
                {"edges": [1], "type": "dirichlet", "voltage": 0.0, "see_gamma": 0.1, "blocking_capacitor": cap_right},
                {"edges": [0, 2], "type": "symmetry"},
            ],
            "regions": regions,
        },
        "mesh": mesh,
        "pic": p,
    })


def _sim(project: Project):
    from es_sim.gpic import GpuPicSimulation

    return GpuPicSimulation(project)


def _caps(coord: str) -> dict:
    s = 1.0 if coord == "xy" else 0.1
    return {"cap_left": {"capacitance": 3e-11 * s, "initial_bias_v": 5.0}, "cap_right": {"capacitance": 1e-10 * s},
            "cap_pin": {"capacitance": 2e-11 * s}}


def _total_charge(sim) -> float:
    q = sum(sp.q * float(sim.get_particles(name)["w"].sum()) for name, sp in sim.species.items())
    return q + float(np.sum(sim.circuit.q_node)) + sim._two_pi * float(sim._q_surf.sum())


@pytest.mark.parametrize("coord", ["xy", "rz"])
@pytest.mark.parametrize("grid", ["uniform", "amr"])
def test_capacitance_and_vacuum_divider_gpu(grid, coord):
    from es_sim.field.compat import cartesian_solve

    amr = PATCH if grid == "amr" else None
    p = _project(cap_pin={"capacitance": 1e-10}, dens=1.0, coord=coord, mcc=False, n_macro=10, amr=amr)
    sim = _sim(p)
    assert (sim.amr is not None) == (grid == "amr")
    static = cartesian_solve(p, device="cpu")[0].capacitance
    assert sim.circuit.c[0, 0] == pytest.approx(static, rel=1e-12)
    cb = 3e-11 if coord == "xy" else 3e-12
    sim = _sim(_project(cap_left={"capacitance": cb, "initial_bias_v": -20.0}, dens=1.0, coord=coord, mcc=False,
                        n_macro=10, amr=amr))
    c = sim.circuit.c[0, 0]
    for _ in range(60):
        sim.step()
        assert sim.circuit.v[0] == pytest.approx(-20.0 + sim.circuit.v_src[0] * cb / (cb + c), abs=1e-9)


@pytest.mark.parametrize("grid", ["uniform", "amr"])
def test_charge_identity_gpu(grid):
    """Poisson が厳密 (小さい格子は密な逆行列) なら、解いた φ の電極の電荷で Q_N = Q_e + C_b (V_e − V_s)。

    ステップの Poisson の時点の Q_N は、ステップの終わりの値からそのステップの伝導電流の分を引いたもの。
    誘電体の表面電荷はステップの中で変わるので、誘電体の無い形で確かめる。
    """
    from es_sim.eb.build import MASK_FIXED
    from es_sim.gpic.simulation import HISTORY_ROWS

    sim = _sim(_project(**_caps("xy"), amr=PATCH if grid == "amr" else None, dielectric=False))
    assert sim.solver.direct
    m = sim.circuit.m
    worst = []
    for _ in range(300):
        sim.step()
        st = sim._cap_st.get()
        row = sim._cap_hist.get()[(sim.step_count - 1) % HISTORY_ROWS]
        q_node = st[:m] - row[m:] * sim.dt
        v, vs = st[m:2 * m], st[2 * m:3 * m]
        phi = sim._phi.get().ravel()
        vg = sim._vg.get()
        q_all = (sim._q_static + sim._rho + sim._q_surf).get().ravel()
        if grid == "uniform":
            op = sim.op
            group_elec = np.full(op.coupling.shape[1], -1)
            for j, groups in enumerate(sim._cap_groups):
                group_elec[groups] = j
            coo = op.coupling.tocoo()
            ej = group_elec[coo.col]
            sel = ej >= 0
            flux = np.bincount(ej[sel], weights=(coo.data * (vg[coo.col] - phi[coo.row]))[sel], minlength=m)
            fixed = (op.mask == MASK_FIXED).ravel()
            fe = np.where(fixed, group_elec[np.maximum(op.fixed_group.ravel(), 0)], -1)
            own = np.bincount(fe[fe >= 0], weights=q_all[fe >= 0], minlength=m)
            q_e = flux - own
        else:
            from es_sim.amr.composite import electrode_charge_functionals

            lay = sim.amr
            a, b, c = electrode_charge_functionals(lay.op, sim._cap_groups)
            q_e = a @ phi + b @ vg[: lay.n_groups] + c @ q_all
        r = q_node - (q_e + sim.circuit.c_b * (v - vs))
        worst.append(float(np.max(np.abs(r)) / np.max(np.abs(q_node))))
    assert sim.history["wall_i"][-1] > 0 and sim.history["see_events"][-1] > 0
    assert max(worst) < 1e-10


@pytest.mark.parametrize("coord", ["xy", "rz"])
@pytest.mark.parametrize("grid", ["uniform", "amr"])
def test_charge_conservation_gpu(grid, coord):
    """全ての電極にコンデンサ (MCC・二次電子・誘電体の表面電荷あり): 電荷の和が丸め誤差で一定。軸対称は半径に
    比例した重み (既定、prompts/136) の分割・併合も電荷を厳密に保つ。"""
    sim = _sim(_project(**_caps(coord), coord=coord, amr=PATCH if grid == "amr" else None))
    assert sim._rw == (coord == "rz")
    sim.step()
    q0 = _total_charge(sim)
    for _ in range(600):
        sim.step()
    moved = QE * (sim.history["wall_e"][-1] + sim.history["wall_i"][-1]) * sim._w0
    assert sim.history["see_events"][-1] > 0 and float(sim._q_surf.sum()) != 0.0 and moved > (1e-9 if coord == "xy" else 1e-10)
    if coord == "rz":
        assert sim.history["split"][-1] > 0 and sim.history["merged"][-1] > 0
    assert abs(_total_charge(sim) - q0) < 1e-12 * moved


def test_regrid_rebuilds_capacitor_weights():
    """動的再格子化 (prompts/123) で W と容量行列を作り直す: 容量は新しい合成格子の静電場の値になり、Q_N は
    そのまま、電荷の和は再格子化をまたいで保存する (test_v2_amr_regrid と同じ 10 mm の箱・1e16 m⁻³)。"""
    import scipy.sparse.linalg as spla

    from es_sim.amr.composite import energy_and_charges

    def box(rf: bool) -> dict:
        left = {"edges": [3], "type": "dirichlet", "voltage": 0.0 if rf else 1.0}
        if rf:
            left |= {"voltage_rf": {"amplitude": 30.0, "freq_hz": F0}, "blocking_capacitor": {"capacitance": 1e-10}}
        return {
            "geometry": {"domain": {"polygon": [[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01]]},
                         "boundaries": [left, {"edges": [0, 1, 2], "type": "dirichlet", "voltage": 0.0,
                                               "blocking_capacitor": {"capacitance": 3e-10} if rf else None}]},
            "mesh": {"size": 0.0004, "mode": "cartesian",
                     "amr": {"max_level": 2, "refine_boundaries": False, "blocking_factor": 4,
                             "pic_regrid_every": 200, "pic_h_over_debye": 1.5}},
            "pic": {"initial_plasma": {"density": 1e16, "te_ev": 3.0, "ti_ev": 0.0, "ion_mass_amu": 40.0,
                                       "immobile_ions": True, "seed": 2},
                    "n_macro": 200000, "n_steps": 150, "avg_steps": 50, "frame_every": 1_000_000, "phase_bins": 0},
        }

    sim = _sim(Project.model_validate(box(True)))
    c_before = sim.circuit.c.copy()
    sim.run_batch(store_frames=False)
    q0 = _total_charge(sim)
    q_node = sim._cap_st.get()[:2].copy()
    for _ in range(2):
        sim._launch_tag_deposit()
    assert sim._regrid() and sim.amr.hier.n_levels > 1
    np.testing.assert_array_equal(sim._cap_st.get()[:2], q_node)
    assert not np.allclose(sim.circuit.c, c_before, rtol=1e-6, atol=0.0)
    # 新しい合成格子で左の辺 1 V・ほか 0 V を直接法で解き、静電場の AMR と同じ電荷 (energy_and_charges) と比べる
    lay = sim.amr
    vk = np.zeros(lay.n_groups)
    vk[sim._cap_groups[0]] = 1.0
    x = spla.spsolve(lay.op.A_c.tocsc(), np.asarray(lay.PTq[:, lay.n_nodes:] @ vk).ravel())
    phi = np.asarray(lay.PC @ np.concatenate([x, vk])).ravel()
    q_left = energy_and_charges(sim.model, lay.op, phi, vk)[1][sim._cap_groups[0]].sum()
    assert sim.circuit.c[0, 0] == pytest.approx(q_left, rel=1e-9)
    ainv = sim._cap_par.get()[:4].reshape(2, 2)
    np.testing.assert_allclose(ainv, np.linalg.inv(sim.circuit.c + np.diag(sim.circuit.c_b)), rtol=1e-12)
    sim.pic.n_steps = 50
    sim.run_batch(store_frames=False)
    moved = QE * sim.history["wall_e"][-1] * sim._w0
    assert moved > 0.0
    assert abs(_total_charge(sim) - q0) < 1e-12 * max(moved, abs(q0))


def test_self_bias_matches_v1_pic_on_ccp_strip():
    """CCP ストリップ (eduPIC の Ar 10 Pa、上下反射) に 13.56 + 27.12 MHz (θ = 0°) を阻止コンデンサ越しに:
    v1 PIC と自己バイアスの推移 (RF 1 周期ごと) が統計の誤差の範囲で一致する (1 周期目から負へ)。"""
    from es_sim.pic import PicSimulation

    e_procs, i_procs = edupic_ar_processes()
    L, Hs = 0.025, 0.002

    def project(mode: str) -> Project:
        return Project.model_validate({
            "geometry": {
                "domain": {"polygon": [[0, 0], [L, 0], [L, Hs], [0, Hs]]},
                "boundaries": [
                    {"edges": [3], "type": "dirichlet", "voltage": 0.0,
                     "voltage_rf": [{"amplitude": 100.0, "freq_hz": F0, "phase_deg": 90.0},
                                    {"amplitude": 100.0, "freq_hz": 2 * F0, "phase_deg": 90.0}],
                     "blocking_capacitor": {"capacitance": 2e-12}},
                    {"edges": [1], "type": "dirichlet", "voltage": 0.0},
                ],
            },
            "mesh": {"size": L / 100, "mode": mode},
            "pic": {
                "initial_plasma": {"density": 5e14, "te_ev": 2.0, "ti_ev": 0.026, "ion_mass_amu": 39.948, "seed": 1},
                "n_macro": 40000, "dt": 1.0 / (F0 * 400), "n_steps": 10 * 400, "avg_steps": 400,
                "frame_every": 1_000_000, "phase_bins": 0, "reflect_edges": [0, 2],
                "mcc": {"gas": {"name": "Ar", "pressure_pa": 10.0, "temperature_k": 300.0},
                        "electron_processes": [q.model_dump() for q in e_procs],
                        "ion_processes": [q.model_dump() for q in i_procs], "seed": 3},
            },
        })

    v1 = PicSimulation(project("structured"))
    v1.run_batch(store_frames=False)
    v2 = _sim(project("cartesian"))
    v2.run_batch(store_frames=False)
    assert v2.circuit.c[0, 0] == pytest.approx(v1.circuit.c[0, 0], rel=1e-6)
    a = np.array([r[0] for r in v1.circuit.history["v_dc"]])
    b = np.array([r[0] for r in v2.circuit.history["v_dc"]])
    assert len(a) == len(b) == 10
    assert np.all(a < -10.0) and np.all(b < -10.0)
    assert np.max(np.abs(a - b)) < 2.0
    assert abs(np.mean(a[5:] - b[5:])) < 1.0


def test_result_frame_and_continue_gpu():
    from es_sim.batch import _build_results_bundle

    sim = _sim(_project(cap_pin={"capacitance": 2e-11, "initial_bias_v": -3.0}, n_steps=500, frame_every=250))
    _, frames = sim.run_batch()
    assert frames[-1]["circuit"][0]["label"] == "pin" and frames[-1]["circuit"][0]["v_e"] is not None
    assert len(sim.circuit.history["v_dc"]) == 1  # 1 周期 = 400 ステップ
    q_before = sim.circuit.q_node.copy()
    sim.prepare_continue(400)
    assert sim.circuit.history["t"] == []
    sim.run_batch(store_frames=False)
    assert len(sim.circuit.history["v_dc"]) == 1 and not np.array_equal(sim.circuit.q_node, q_before)
    bundle = _build_results_bundle(sim, 500, elapsed_s=1.0)["pic"]
    c = bundle["circuit"]
    assert c["units"]["capacitance"] == "F/m" and c["electrodes"][0]["label"] == "pin"
    assert c["electrodes"][0]["c_self"] == pytest.approx(sim.circuit.c[0, 0])
    plain = _sim(_project(n_macro=100))
    plain.run_batch(store_frames=False)
    assert _build_results_bundle(plain, 0, elapsed_s=0.0)["pic"]["circuit"] is None
