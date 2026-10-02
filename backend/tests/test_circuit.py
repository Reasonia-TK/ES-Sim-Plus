"""阻止コンデンサ (自己バイアス、prompts/134) のテスト: 回路の集計 (circuit.py) と流体 1D。

1. 周期の集計: 区間を周期の境で分けて、V_dc・|V1|・I_dc を正しく求める (dt が周期を割り切らなくても)。
2. 真空の分圧: 伝導電流 0 なら V_e = initial_bias + V_s·C_b/(C_b + C)、C = ε0/L。
3. 電荷の恒等式: 毎サブステップ Q_N = Q_e(φ) + C_b (V_e − V_s) (解き直した φ の電極の電荷で)。
4. C_b → ∞ で直結 (+ initial_bias のずれ) と一致する。
5. 対称な放電 (両側の γ が同じ、1 周波) では自己バイアスが小さく、0 へ向かう。
6. 電気的非対称効果 (EAE、13.56 + 27.12 MHz): θ = 0° で負、90° で正。
7. 続き実行はビット一致し、コンデンサの状態を引き継ぐ。結果とフレームに circuit が載る。
8. スキーマの制約と、まだ対応していないエンジンのエラー。
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pydantic
import pytest

from es_sim.circuit import BlockingCircuit, CapacitorSpec, blocking_capacitor_labels
from es_sim.fem import EPS0
from es_sim.fluid1d import Fluid1dSimulation, build_fluid1d_result
from es_sim.pic1d import Pic1dSimulation, electrode_voltage
from es_sim.schema import (
    BlockingCapacitor1d,
    Domain,
    Fluid1dSettings,
    Geometry,
    MeshSettings,
    Pic1dElectrode,
    Pic1dSettings,
    Project,
    VoltageRF,
)

F0 = 13.56e6
GAP = 0.025
_GEOMETRY = Geometry(domain=Domain(polygon=[(0, 0), (1, 0), (1, 1), (0, 1)]))
_MESH = MeshSettings(size=0.1)


def _project(fluid1d: Fluid1dSettings) -> Project:
    return Project(geometry=_GEOMETRY, mesh=_MESH, fluid1d=fluid1d)


def _settings(cap=None, rf=None, n_steps=1, n_cells=60, gamma_right=0.05, **kw) -> Fluid1dSettings:
    """13.56 MHz・Ar 50 Pa・ギャップ 2.5 cm (test_fluid1d の CCP スモークと同じ条件)。左が RF。"""
    rf = rf if rf is not None else VoltageRF(amplitude=150.0, freq_hz=F0, phase_deg=0.0)
    return Fluid1dSettings(
        gap_m=GAP, n_cells=n_cells,
        left=Pic1dElectrode(v_dc=0.0, voltage_rf=rf, see_gamma=0.05, blocking_capacitor=cap),
        right=Pic1dElectrode(v_dc=0.0, see_gamma=gamma_right),
        init_density_m3=1.0e15, init_te_ev=3.0, gas_pressure_pa=50.0, gas_temperature_k=300.0,
        n_steps=n_steps, frame_every=1_000_000, phase_bins=20, **kw,
    )


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # 係数テーブルの範囲外 (真空の Te) の警告
        yield


# ---- 1. 周期の集計 -------------------------------------------------------------------------


@pytest.mark.parametrize("steps_per_period", [100, 37.3])
def test_period_statistics_split_at_boundaries(steps_per_period):
    """V(t) = D + A sin(ωt + p) を区間一定で与えると、各周期の V_dc ≈ D・|V1| ≈ A・I_dc = I になる。"""
    period = 1.0 / F0
    c = BlockingCircuit([CapacitorSpec("e", 1e-9)], [[1e-11]], period)
    c.solve([0.0], [0.0], [0.0])
    dt = period / steps_per_period
    d, a, p, i = -40.0, 80.0, 0.7, 2.5e-3
    t = 0.0
    while t < 5.2 * period:
        tm = t + 0.5 * dt  # 区間の中点の値で一定とする
        c.v = np.array([d + a * math.sin(2 * math.pi * F0 * tm + p)])
        c.conduct([i], t, dt)
        t += dt
    h = c.history
    assert len(h["t"]) == 5
    assert h["t"][0] == pytest.approx(period)
    for row in h["v_dc"]:
        assert row[0] == pytest.approx(d, abs=1e-3 * a)
    for row in h["v1"]:
        assert row[0] == pytest.approx(a, rel=2e-3)
    for row in h["i_dc"]:
        assert row[0] == pytest.approx(i, rel=1e-12)
    w = c.last_wave
    assert len(w["t"]) == len(w["v"][0]) and 0.0 <= w["t"][0] < w["t"][-1] < period


# ---- 2〜4. 流体 1D の回路 --------------------------------------------------------------------


def test_vacuum_divider_1d():
    """密度がほぼ 0・反射壁 (伝導電流 0): V_e = bias + V_s·C_b/(C_b + ε0/L)。"""
    cb = 1.0e-8
    s = _settings(BlockingCapacitor1d(capacitance=cb, initial_bias_v=-20.0),
                  rf=VoltageRF(amplitude=100.0, freq_hz=F0))
    s.init_density_m3 = 1.0
    sim = Fluid1dSimulation(_project(s))
    sim.debug_reflective_walls = True
    c = EPS0 / GAP
    assert sim.circuit.c[0, 0] == pytest.approx(c, rel=1e-12)
    for _ in range(300):
        sim.step()
        expected = -20.0 + sim.circuit.v_src[0] * cb / (cb + c)
        # 床の値の密度 (1e6 m⁻³) の空間電荷の分だけずれる (〜1e-7 V)
        assert sim.circuit.v[0] == pytest.approx(expected, abs=1e-6)
        assert sim.phi[0] == pytest.approx(expected, abs=1e-6)


def test_charge_identity_with_plasma_1d():
    """放電中も毎サブステップ Q_N = Q_e(φ) + C_b (V_e − V_s) が丸め誤差で成り立つ。"""
    sim = Fluid1dSimulation(_project(_settings(BlockingCapacitor1d(capacitance=1.0e-7, initial_bias_v=5.0))))
    solve_phi = sim._solve_phi
    worst = []

    def checked(n_e, n_i, t):
        phi = solve_phi(n_e, n_i, t)
        r = sim.circuit.residual(sim._electrode_charges(phi, n_e, n_i))
        worst.append(float(np.max(np.abs(r))) / float(np.max(np.abs(sim.circuit.q_node))))
        # 電極の節点の電位は回路の V_e そのもの
        assert phi[0] == sim.circuit.v[0]
        return phi

    sim._solve_phi = checked
    for _ in range(600):
        sim.step()
    assert worst and max(worst) < 1e-10


def test_large_capacitance_matches_direct_coupling_1d():
    """C_b → ∞ では V_e = V_s + bias。bias 0 なら直結の計算と一致し、bias は直流分のずれになる。"""
    direct = Fluid1dSimulation(_project(_settings()))
    cap = Fluid1dSimulation(_project(_settings(BlockingCapacitor1d(capacitance=1.0e6))))
    for _ in range(1500):
        direct.step()
        cap.step()
    assert np.max(np.abs(direct.phi - cap.phi)) < 1e-9
    assert np.max(np.abs(direct.n_e - cap.n_e)) < 1e-10 * np.max(direct.n_e)

    biased = Fluid1dSimulation(_project(_settings(BlockingCapacitor1d(capacitance=1.0e6, initial_bias_v=-30.0))))
    for _ in range(200):
        t = biased.t
        biased.step()
        assert biased.phi[0] - electrode_voltage(biased.s.left, t) == pytest.approx(-30.0, abs=1e-6)


# ---- 5〜6. 自己バイアス -------------------------------------------------------------------------


def _bias_history(sim: Fluid1dSimulation) -> np.ndarray:
    return np.array([row[0] for row in sim.circuit.history["v_dc"]])


def test_symmetric_discharge_has_small_bias_1d():
    """両側が同じ γ・1 周波の対称な放電: 立ち上がりの片寄り (最初の周期で数 V) が 0 へ戻っていく。"""
    s = _settings(BlockingCapacitor1d(capacitance=1.0e-7), n_steps=10 * 2000)
    sim = Fluid1dSimulation(_project(s))
    sim.run_batch(store_frames=False)
    v_dc = _bias_history(sim)
    assert len(v_dc) == 10
    assert abs(v_dc[-1]) < 0.02 * 150.0
    assert abs(v_dc[-1]) < abs(v_dc[0])


def test_electrical_asymmetry_effect_sign_1d():
    """V0[cos(ωt+θ) + cos(2ωt)] (sin の位相は θ+90°・90°): θ = 0° は負、θ = 90° は正の自己バイアス。

    理想値 η = −(φmax + φmin)/2 は θ = 0° で −0.44 V0、90° で +0.44 V0 (Heil ら 2008)。短い計算なので
    符号と大きさの目安 (理想値の 1 割以上) だけ見る (収束後の比べ方は prompts/134 の検証の記録)。
    """
    v0 = 100.0
    out = {}
    for th in (0.0, 90.0):
        rf = [VoltageRF(amplitude=v0, freq_hz=F0, phase_deg=th + 90.0),
              VoltageRF(amplitude=v0, freq_hz=2 * F0, phase_deg=90.0)]
        s = _settings(BlockingCapacitor1d(capacitance=1.0e-8), rf=rf, n_steps=8 * 2000)
        sim = Fluid1dSimulation(_project(s))
        sim.run_batch(store_frames=False)
        out[th] = _bias_history(sim)[-1]
    assert out[0.0] < -0.044 * v0
    assert out[90.0] > 0.044 * v0


# ---- 7. 続き・結果 -------------------------------------------------------------------------------


def test_continue_is_bit_exact_and_keeps_capacitor_state():
    cap = BlockingCapacitor1d(capacitance=1.0e-7, initial_bias_v=-10.0)
    full = Fluid1dSimulation(_project(_settings(cap, n_steps=6500, avg_steps=500)))
    full.run_batch(store_frames=False)
    part = Fluid1dSimulation(_project(_settings(cap, n_steps=2100, avg_steps=500)))
    part.run_batch(store_frames=False)
    part.prepare_continue(extra_steps=4400, avg_steps=500)
    part.run_batch(store_frames=False)
    np.testing.assert_array_equal(full.phi, part.phi)
    np.testing.assert_array_equal(full.n_e, part.n_e)
    np.testing.assert_array_equal(full.circuit.q_node, part.circuit.q_node)
    np.testing.assert_array_equal(full.circuit.v, part.circuit.v)
    # 続きの履歴は追加区間で閉じた周期だけ (1 周期 = 2000 ステップ。2100 ステップ目は 2 周期目の途中で、
    # 続きで 2・3 周期目が閉じる。途中の周期の積分は引き継ぐので、全体を通した計算と同じ値)
    assert full.circuit.history["t"][-2:] == part.circuit.history["t"]
    assert full.circuit.history["v_dc"][-2:] == part.circuit.history["v_dc"]


def test_result_and_frame_carry_circuit_1d():
    s = _settings(BlockingCapacitor1d(capacitance=2.0e-7), n_steps=2 * 2000, avg_steps=500)
    s.frame_every = 1000
    sim = Fluid1dSimulation(_project(s))
    _, frames = sim.run_batch()
    assert frames[-1]["circuit"][0]["label"] == "left"
    assert frames[-1]["circuit"][0]["v_dc"] == sim.circuit.history["v_dc"][-1][0]
    res = build_fluid1d_result(sim, elapsed_s=1.0)
    c = res["circuit"]
    assert c["units"]["capacitance"] == "F/m^2"
    assert c["period_s"] == pytest.approx(1.0 / F0)
    e = c["electrodes"][0]
    assert e["label"] == "left" and e["capacitance"] == 2.0e-7
    assert e["c_self"] == pytest.approx(EPS0 / GAP)
    assert len(e["t"]) == len(e["v_dc"]) == len(e["v1"]) == len(e["i_dc"]) == 2
    assert 0.0 < len(e["last_period"]["t"]) <= 400
    # コンデンサが無ければ None (今までの結果と同じ形)
    plain = Fluid1dSimulation(_project(_settings()))
    plain.step()
    assert build_fluid1d_result(plain, elapsed_s=0.0)["circuit"] is None


# ---- 8. スキーマ・未対応のエンジン ----------------------------------------------------------------


def _project_2d(regions, boundaries):
    return Project.model_validate({
        "geometry": {"domain": {"polygon": [[0, 0], [0.02, 0], [0.02, 0.01], [0, 0.01]]},
                     "regions": regions, "boundaries": boundaries},
        "mesh": {"size": 1e-3, "mode": "cartesian"},
    })


def test_schema_allows_capacitor_only_on_electrodes():
    cap = {"capacitance": 5e-9}
    with pytest.raises(pydantic.ValidationError):
        _project_2d([{"id": "d", "type": "dielectric", "polygon": [[0, 0], [1e-3, 0], [1e-3, 1e-3]],
                      "blocking_capacitor": cap}], [])
    with pytest.raises(pydantic.ValidationError):
        _project_2d([], [{"edges": [0], "type": "symmetry", "blocking_capacitor": cap}])
    with pytest.raises(pydantic.ValidationError):
        BlockingCapacitor1d(capacitance=0.0)
    p = _project_2d(
        [{"id": "rf", "type": "conductor", "voltage": 0.0, "polygon": [[0, 0], [1e-3, 0], [1e-3, 1e-3]],
          "blocking_capacitor": cap}],
        [{"edges": [1, 3], "type": "dirichlet", "blocking_capacitor": cap}],
    )
    assert blocking_capacitor_labels(p) == ["edge1", "edge3", "rf"]


def test_engines_without_support_reject_capacitor():
    """PIC (1D・2D) はまだ対応していないので実行の初めにエラー。流体 2D は全て (一様格子・AMR・v1) 対応。"""
    pic1d = Pic1dSettings(
        gap_m=GAP, init_density_m3=1e15,
        left=Pic1dElectrode(blocking_capacitor=BlockingCapacitor1d(capacitance=1e-7)),
    )
    with pytest.raises(ValueError, match="阻止コンデンサ"):
        Pic1dSimulation(Project(geometry=_GEOMETRY, mesh=_MESH, pic1d=pic1d))

    from es_sim.gfluid import CartesianFluid2dSimulation, make_fluid2d_simulation
    from es_sim.gfluid.amr import AmrFluid2dSimulation
    from es_sim.gpic import make_pic_simulation

    base = _project_2d(
        [{"id": "rf", "type": "conductor", "voltage": 0.0, "polygon": [[0.008, 0.004], [0.012, 0.004], [0.012, 0.006]],
          "blocking_capacitor": {"capacitance": 5e-9}}],
        [{"edges": [1, 3], "type": "dirichlet"}],
    ).model_dump()
    pic = Project.model_validate({**base, "pic": {}})
    for mode in ("cartesian", "structured"):
        pic.mesh.mode = mode
        with pytest.raises(ValueError, match="PIC はまだ阻止コンデンサ"):
            make_pic_simulation(pic)
    fluid = Project.model_validate({**base, "fluid2d": {"init_density_m3": 1e15, "gas_pressure_pa": 30.0}})
    assert CartesianFluid2dSimulation(fluid, device="cpu").circuit.labels == ["rf"]
    amr = fluid.model_copy(deep=True)
    amr.mesh.amr = {"max_level": 1, "buffer_cells": 2}
    amr = Project.model_validate(amr.model_dump())
    assert AmrFluid2dSimulation(amr, device="cpu").circuit.labels == ["rf"]
    fluid.mesh.mode = "structured"
    assert make_fluid2d_simulation(fluid).circuit.labels == ["rf"]


# ---- 9. スイープのケースの自己バイアス ---------------------------------------------------------------


def test_sweep_summary_carries_self_bias():
    """流体 1D の阻止コンデンサの容量をスイープすると、ケースの要約に最後の周期の自己バイアスが載る。"""
    import time

    from fastapi.testclient import TestClient

    import es_sim.server as server

    s = _settings(BlockingCapacitor1d(capacitance=1e-7), n_steps=2100, avg_steps=100)
    project = _project(s).model_dump(mode="json")
    client = TestClient(server.app)
    body = {"kind": "sweep", "project": project,
            "options": {"param_path": "fluid1d.left.blocking_capacitor.capacitance", "values": [1e-8, 1e-7], "parallel": 2}}
    jid = client.post("/v2/jobs", json=body).json()["id"]
    t0 = time.time()
    while client.get(f"/v2/jobs/{jid}").json()["state"] not in ("done", "error"):
        assert time.time() - t0 < 180, "timeout"
        time.sleep(0.05)
    res = client.get(f"/v2/jobs/{jid}/result").json()
    assert res["module"] == "fluid1d" and [c["ok"] for c in res["summary"]] == [True, True]
    for c in res["summary"]:
        (bias,) = c["self_bias"]
        assert bias["label"] == "left" and bias["v1"] > 100.0 and abs(bias["v_dc"]) < 150.0
    # 容量が小さいほど自己バイアスが早く動く (最初の周期の片寄りが大きい)
    assert abs(res["summary"][0]["self_bias"][0]["v_dc"]) > abs(res["summary"][1]["self_bias"][0]["v_dc"])
    client.delete(f"/v2/jobs/{jid}")
