"""1D PIC/MCC (1d3v) の専用一様格子ソルバーのテスト (prompts/91)。

1. Poisson 精度: 粒子なし・両端 Dirichlet → φ が線形、E が一様
2. リープフロッグ: 一様 E 中の単電子 (MCC なし) の軌道が解析解と一致
   (定数加速度なのでリープフロッグは数値的に厳密 — 2次精度の上限を確認する)
3. 電荷保存: CIC 堆積後の Σ f_dep = Σ q·w (端節点の半体積を含む密度変換をしても
   総量は不変)
4. CCP スモーク (eduPIC Ar 解析式断面積): シース形成・電離発生を確認する
   頑健なアサーションのみ (定常前でも成り立つもの)
5. 続きから実行: run(n) → continue(m) が run(n+m) とビット一致
6. SEE: see_gamma=1.0 でイオン吸収の重み和 ≈ SEE 放出の重み和
7. validator: gap_m<=0、eedf_regions 5個、use_dsmc_gas=True で ValueError
8. プリセット: edupic_ar の断面積が閾値未満で0・閾値超で正、
   Turner Case 1 の数値 (gap, pressure, freq) が仕様通り
9. RF 重畳 (voltage_rf、prompts/93): 電極電圧が2D と同じ式どおり、CSV サンプル波形
   との等価性、cycle 基本周波数の優先順位 (voltage_rf 優先)、デュアル周波数の疎通
"""

import json
import math

import numpy as np
import pydantic
import pytest
from fastapi.testclient import TestClient

import es_sim.server as server
from es_sim.mcc import KB
from es_sim.particles import ME, QE
from es_sim.pic1d import Pic1dSimulation
from es_sim.pic1d_presets import edupic_ar_processes, get_presets
from es_sim.schema import Geometry, Domain, MeshSettings, Pic1dSettings, Project, VoltageWaveform

# ダミーの geometry/mesh (1D PIC はこれらを一切参照しないが、Project スキーマ上必須)
_DUMMY_GEOMETRY = Geometry(domain=Domain(polygon=[(0, 0), (1, 0), (1, 1), (0, 1)]))
_DUMMY_MESH = MeshSettings(size=0.1)


def _project(pic1d: Pic1dSettings) -> Project:
    return Project(geometry=_DUMMY_GEOMETRY, mesh=_DUMMY_MESH, pic1d=pic1d)


# ---- 1. Poisson 精度 ---------------------------------------------------------


def test_poisson_linear_no_particles():
    gap = 0.02
    s = Pic1dSettings(
        gap_m=gap, n_cells=40, init_density_m3=1.0e10, n_macro=1, dt=1e-9, n_steps=1,
        left={"v_dc": 100.0}, right={"v_dc": 0.0},
    )
    sim = Pic1dSimulation(_project(s))
    f0 = np.zeros(sim.n_nodes)  # 粒子なし (電荷ゼロ)
    phi = sim._solve_phi(f0, 0.0)
    expected = 100.0 * (1.0 - sim.xg / gap)
    assert np.allclose(phi, expected, rtol=1e-12, atol=1e-9)

    ex = sim._e_field(phi)
    e_expected = 100.0 / gap
    assert np.allclose(ex, e_expected, rtol=1e-10)


# ---- 2. リープフロッグ (定数加速度、解析解と一致) -----------------------------------


def test_leapfrog_matches_analytic_uniform_field():
    gap = 0.02
    v_bias = 10.0
    s = Pic1dSettings(
        gap_m=gap, n_cells=50, init_density_m3=1.0e6, n_macro=1, dt=1e-10, n_steps=1,
        left={"v_dc": v_bias}, right={"v_dc": 0.0},
    )
    sim = Pic1dSimulation(_project(s))
    # 単電子のみ (イオンは重み0で空間電荷への寄与をゼロにする)。中央から出発
    x0 = gap / 2.0
    sim.species["electron"].x = np.array([x0])
    sim.species["electron"].v = np.array([[0.0, 0.0, 0.0]])
    sim.species["electron"].w = np.array([0.0])
    sim.species["ion"].x = np.array([x0])
    sim.species["ion"].v = np.array([[0.0, 0.0, 0.0]])
    sim.species["ion"].w = np.array([0.0])
    # 粒子を差し替えたので初期半ステップ後退キックをやり直す (重み0なので空間電荷影響なし)
    f0 = sim._deposit()
    phi0 = sim._solve_phi(f0, 0.0)
    ex0 = sim._e_field(phi0)
    e_at = sim._cic_gather(ex0, sim.species["electron"].x)
    sp = sim.species["electron"]
    sp.v[:, 0] -= 0.5 * sim.dt * (sp.q / sp.m) * e_at

    e_uniform = v_bias / gap  # V_L>V_R の線形電位 → E = +v_bias/gap (一様)
    a = (-QE / ME) * e_uniform
    n = 100  # a<0 で x0=gap/2 から左壁まで届かない範囲 (解析的に確認済み)
    for _ in range(n):
        sim.step()
    t = sim.dt * n
    x_analytic = x0 + 0.5 * a * t * t
    assert sim.species["electron"].x[0] == pytest.approx(x_analytic, rel=1e-9, abs=1e-15)


# ---- 3. 電荷保存 ---------------------------------------------------------------


def test_charge_deposit_conserves_total_charge():
    s = Pic1dSettings(gap_m=0.01, n_cells=17, init_density_m3=1.0e14, n_macro=777, dt=1e-10, n_steps=1, seed=5)
    sim = Pic1dSimulation(_project(s))
    for sp in sim.species.values():
        f = sim._cic_deposit(sp.x, sp.q * sp.w)
        total_q = float(np.sum(sp.q * sp.w))
        assert f.sum() == pytest.approx(total_q, rel=1e-12, abs=1e-30)
        # 密度変換 (端節点は半体積) をしても Σρ·vol は堆積した総電荷と厳密に一致する
        rho = f / sim.node_vol
        assert np.sum(rho * sim.node_vol) == pytest.approx(total_q, rel=1e-12, abs=1e-30)


# ---- 4. CCP スモーク (eduPIC Ar 解析式断面積) --------------------------------------


def _edupic_smoke_settings(n_steps: int) -> Pic1dSettings:
    preset = get_presets()["edupic_ar"]["pic1d"]
    d = dict(preset)
    d["n_cells"] = 64
    d["n_macro"] = 4000
    d["n_steps"] = n_steps
    d["frame_every"] = n_steps
    d["avg_steps"] = None
    return Pic1dSettings(**d)


def test_ccp_smoke_edupic_ar():
    # 13.56 MHz・dt=1/(400f) で 2400 ステップ ≈ 6 RF 周期 (CI で数秒程度)
    s = _edupic_smoke_settings(2400)
    sim = Pic1dSimulation(_project(s))
    hist, _ = sim.run_batch()

    assert sim.fields is not None
    for key in ("phi", "e", "n_e", "n_i", "t_e", "ionization"):
        assert np.all(np.isfinite(sim.fields[key])), key

    n_e = sim.fields["n_e"]
    n_i = sim.fields["n_i"]
    # 両壁近傍数セルではシース形成により電子密度がイオン密度を下回る
    assert np.all(n_e[:3] < n_i[:3])
    assert np.all(n_e[-3:] < n_i[-3:])
    # バルク中央のイオン密度は壁近傍より大きい
    mid = sim.n_nodes // 2
    assert n_i[mid] > n_i[0]
    assert n_i[mid] > n_i[-1]
    # 電離イベントが実際に発生している
    assert hist["ion_events"][-1] > 0
    assert float(sim.fields["ionization"].max()) > 0.0


# ---- 5. 続きから実行 (ビット一致) --------------------------------------------------


def test_continue_is_bit_identical_to_single_run():
    def make(n_steps: int) -> Pic1dSettings:
        preset = get_presets()["edupic_ar"]["pic1d"]
        d = dict(preset)
        d["n_cells"] = 32
        d["n_macro"] = 500
        d["n_steps"] = n_steps
        d["frame_every"] = n_steps
        return Pic1dSettings(**d)

    # A: 1200 ステップ連続 (avg_steps=300 を明示指定 → 平均区間 901..1200)
    sim_a = Pic1dSimulation(_project(make(1200)))
    sim_a.s.avg_steps = 300
    hist_a, _ = sim_a.run_batch()

    # B: 600 ステップ → continue で 600 ステップ (avg_steps=300 → 同じ平均区間 901..1200)
    sim_b = Pic1dSimulation(_project(make(600)))
    sim_b.run_batch()
    sim_b.prepare_continue(600, avg_steps=300)
    hist_b, _ = sim_b.run_batch()

    for name in ("electron", "ion"):
        sa, sb = sim_a.species[name], sim_b.species[name]
        assert len(sa.x) == len(sb.x) and len(sa.x) > 0
        assert np.array_equal(sa.x, sb.x)
        assert np.array_equal(sa.v, sb.v)
        assert np.array_equal(sa.w, sb.w)

    assert sim_a.coll_e == sim_b.coll_e
    assert sim_a.ion_events == sim_b.ion_events
    assert sim_a.t == sim_b.t
    assert sim_a.step_count == sim_b.step_count == 1200
    assert hist_a["n_e"][-1] == hist_b["n_e"][-1]

    assert sim_a.fields is not None and sim_a.fields["avg_steps"] == 300
    assert sim_b.fields is not None and sim_b.fields["avg_steps"] == 300
    assert np.array_equal(sim_a.fields["phi"], sim_b.fields["phi"])
    assert np.array_equal(sim_a.fields["n_e"], sim_b.fields["n_e"])
    assert np.array_equal(sim_a.fields["n_i"], sim_b.fields["n_i"])


# ---- 6. SEE ---------------------------------------------------------------------


def test_see_gamma_one_matches_ion_absorption():
    s = Pic1dSettings(
        gap_m=0.01, n_cells=20, init_density_m3=1.0e14, init_te_ev=2.0, init_ti_ev=5.0,
        n_macro=3000, dt=2e-10, n_steps=300, frame_every=300, seed=1,
        left={"v_dc": 0.0, "see_gamma": 1.0}, right={"v_dc": 0.0, "see_gamma": 1.0},
    )
    sim = Pic1dSimulation(_project(s))
    sim.run_batch()
    wall_i_total = sim.wall["left"]["ion"] + sim.wall["right"]["ion"]
    assert wall_i_total > 0.0  # 実際にイオンが壁に吸収されている
    assert sim.see_events == pytest.approx(wall_i_total, rel=1e-9)


# ---- 7. validator -----------------------------------------------------------------


def test_validators_raise():
    with pytest.raises(pydantic.ValidationError):
        Pic1dSettings(gap_m=0.0, init_density_m3=1e14)
    with pytest.raises(pydantic.ValidationError):
        Pic1dSettings(gap_m=0.0, init_density_m3=1e14, n_cells=8)  # 念のため別値でも確認
    with pytest.raises(pydantic.ValidationError):
        Pic1dSettings(
            gap_m=0.01, init_density_m3=1e14,
            eedf_regions=[{"x1": 0.0, "x2": 0.001} for _ in range(5)],
        )
    with pytest.raises(pydantic.ValidationError):
        Pic1dSettings(
            gap_m=0.01, init_density_m3=1e14,
            mcc={"gas": {"pressure_pa": 1.0}, "use_dsmc_gas": True},
        )


# ---- 8. プリセット ------------------------------------------------------------------


def test_edupic_ar_processes_thresholds():
    e_procs, i_procs = edupic_ar_processes()
    by_kind = {p.kind: p for p in e_procs}

    exc = by_kind["excitation"]
    e = np.asarray(exc.energy_ev)
    sig = np.asarray(exc.sigma_m2)
    assert exc.threshold_ev == pytest.approx(11.5)
    assert np.all(sig[e <= 11.5] == 0.0)
    assert np.any(e > 11.5)
    assert np.all(sig[e > 11.5 + 1e-6] > 0.0)

    ionz = by_kind["ionization"]
    e2 = np.asarray(ionz.energy_ev)
    sig2 = np.asarray(ionz.sigma_m2)
    assert ionz.threshold_ev == pytest.approx(15.8)
    assert np.all(sig2[e2 <= 15.8] == 0.0)
    assert np.all(sig2[e2 > 15.8 + 1e-6] > 0.0)

    elastic = by_kind["elastic"]
    assert np.all(np.asarray(elastic.sigma_m2) >= 0.0)
    assert elastic.mass_ratio > 0.0

    assert {p.kind for p in i_procs} == {"isotropic", "backscat"}
    for p in i_procs:
        assert np.all(np.asarray(p.sigma_m2) >= 0.0)
        assert np.asarray(p.energy_ev).min() > 0.0  # E^-0.5 特異点を避けるため 0 を含まない


def test_turner_preset_values():
    presets = get_presets()
    turner = presets["turner_he_case1"]
    assert "note" in turner and ("LXCat" in turner["note"] or "断面積" in turner["note"])
    p = turner["pic1d"]
    assert p["gap_m"] == pytest.approx(0.067)
    assert p["mcc"]["electron_processes"] == []
    assert p["mcc"]["ion_processes"] == []

    n_he = 9.64e20
    expected_pressure = n_he * KB * 300.0
    assert p["mcc"]["gas"]["pressure_pa"] == pytest.approx(expected_pressure)

    left_rf = p["left"]["voltage_rf"]
    assert left_rf["freq_hz"] == pytest.approx(13.56e6)
    assert left_rf["amplitude"] == pytest.approx(450.0)
    assert left_rf["phase_deg"] == pytest.approx(0.0)

    expected_amu = 6.67e-27 / 1.66054e-27
    assert p["ion_mass_amu"] == pytest.approx(expected_amu)


def test_edupic_preset_values():
    presets = get_presets()
    edupic = presets["edupic_ar"]
    p = edupic["pic1d"]
    assert p["gap_m"] == pytest.approx(0.025)
    assert p["mcc"]["gas"]["pressure_pa"] == pytest.approx(10.0)
    assert p["mcc"]["gas"]["temperature_k"] == pytest.approx(350.0)
    left_rf = p["left"]["voltage_rf"]
    assert left_rf["freq_hz"] == pytest.approx(13.56e6)
    assert left_rf["amplitude"] == pytest.approx(250.0)
    assert len(p["mcc"]["electron_processes"]) == 3
    assert len(p["mcc"]["ion_processes"]) == 2


# ---- GET /pic1d/presets (簡易 API 疎通確認) ------------------------------------------


def test_pic1d_presets_endpoint():
    client = TestClient(server.app)
    resp = client.get("/pic1d/presets")
    assert resp.status_code == 200
    body = resp.json()
    assert "edupic_ar" in body and "turner_he_case1" in body
    # プリセットの pic1d 部分がそのまま Pic1dSettings として検証できること
    Pic1dSettings(**body["edupic_ar"]["pic1d"])
    Pic1dSettings(**body["turner_he_case1"]["pic1d"])


# ---- /ws/pic1d プロトコル (start → frame → done、continue) ---------------------------


def _empty_pic1d_project(n_steps: int) -> dict:
    s = Pic1dSettings(
        gap_m=0.01, n_cells=10, init_density_m3=1.0e12, n_macro=20,
        dt=1e-9, n_steps=n_steps, frame_every=max(1, n_steps // 2),
    )
    return _project(s).model_dump(mode="json")


def _recv_until_done(ws) -> dict:
    while True:
        msg = ws.receive_json()
        assert msg["type"] != "error", msg.get("detail")
        if msg["type"] == "done":
            return msg


def test_ws_pic1d_start_frame_done_and_continue():
    server._last_sim1d = None
    client = TestClient(server.app)
    with client.websocket_connect("/ws/pic1d") as ws:
        project = _empty_pic1d_project(10)
        ws.send_text(json.dumps({"cmd": "start", "project": project}))
        started = ws.receive_json()
        assert started["type"] == "started"
        assert started["n_steps"] == 10
        assert started["step_offset"] == 0
        assert isinstance(started["dt"], float)
        assert len(started["x"]) == 11  # n_cells=10 → n_nodes=11
        assert "threads" not in started

        done1 = _recv_until_done(ws)
        result = done1["result"]
        assert len(result["history"]["t"]) == 10
        assert result["elapsed_s"] > 0.0
        assert "total" in result["timing"]

        ws.send_text(json.dumps({"cmd": "continue", "extra_steps": 5}))
        started2 = ws.receive_json()
        assert started2["type"] == "started" and started2["n_steps"] == 5
        assert started2["step_offset"] == 10
        done2 = _recv_until_done(ws)
        assert len(done2["result"]["history"]["t"]) == 5

    server._last_sim1d = None


def test_ws_pic1d_continue_without_state_errors():
    server._last_sim1d = None
    client = TestClient(server.app)
    with client.websocket_connect("/ws/pic1d") as ws:
        ws.send_text(json.dumps({"cmd": "continue", "extra_steps": 5}))
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "保持" in msg["detail"] or "start" in msg["detail"]


# ---- 9. RF 重畳 (voltage_rf, prompts/93) ------------------------------------------


def test_electrode_voltage_rf_matches_2d_formula():
    """V(t) = v_dc + Σ amplitude·sin(2π·freq_hz·t + phase_deg·π/180) (pic.py の
    _dirichlet_values と完全に同じ式) どおりの値になることを t=0, T/4, T で確認する。
    """
    amp, freq, phase_deg = 100.0, 5.0e6, 30.0
    s = Pic1dSettings(
        gap_m=0.01, n_cells=10, init_density_m3=1.0e10, n_macro=1, dt=1e-9, n_steps=1,
        left={"v_dc": 10.0, "voltage_rf": {"amplitude": amp, "freq_hz": freq, "phase_deg": phase_deg}},
        right={"v_dc": 0.0},
    )
    sim = Pic1dSimulation(_project(s))
    period = 1.0 / freq
    for t in (0.0, period / 4.0, period):
        expected = 10.0 + amp * math.sin(2.0 * math.pi * freq * t + math.radians(phase_deg))
        assert sim._electrode_voltage(sim.s.left, t) == pytest.approx(expected, rel=1e-12, abs=1e-9)


def test_voltage_rf_matches_sampled_waveform_profile():
    """voltage_rf {A, f, phase=0} と、同じ正弦を2000点サンプルした VoltageWaveform とで、
    短い実行の時間平均プロファイルが近いこと (rtol 緩め、線形補間誤差分) を確認する。
    """
    freq = 13.56e6
    amp = 50.0
    n = 2000
    phase = (np.arange(n) / n).tolist()
    v = (amp * np.sin(2.0 * np.pi * np.arange(n) / n)).tolist()
    wf = VoltageWaveform(freq_hz=freq, phase=phase, v=v)

    def make(left_electrode: dict) -> Pic1dSettings:
        return Pic1dSettings(
            gap_m=0.01, n_cells=32, init_density_m3=1.0e14, n_macro=1000,
            dt=1.0 / (400.0 * freq), n_steps=400, frame_every=400, seed=3,
            left=left_electrode, right={"v_dc": 0.0},
        )

    s_rf = make({"v_dc": 0.0, "voltage_rf": {"amplitude": amp, "freq_hz": freq, "phase_deg": 0.0}})
    s_wf = make({"v_dc": 0.0, "waveforms": [wf.model_dump()]})

    sim_rf = Pic1dSimulation(_project(s_rf))
    sim_wf = Pic1dSimulation(_project(s_wf))
    sim_rf.run_batch()
    sim_wf.run_batch()

    assert sim_rf.fields is not None and sim_wf.fields is not None
    for key in ("phi", "n_e", "n_i"):
        np.testing.assert_allclose(
            sim_rf.fields[key], sim_wf.fields[key], rtol=1e-3, atol=1e-6
        )


def test_cycle_freq_prefers_voltage_rf_over_waveform():
    """cycle 基本周波数は voltage_rf を優先する (pic1d.py の Pic1dSimulation.__init__
    のコメント参照)。voltage_rf のみ指定でも cycle が有効になり freq_hz が一致すること、
    および voltage_rf と異なる周波数の waveform が同時にあっても voltage_rf 側が
    優先されることを確認する。
    """
    rf_freq = 27.12e6
    wf_freq = 1.0e6
    n = 100
    wf = VoltageWaveform(
        freq_hz=wf_freq,
        phase=(np.arange(n) / n).tolist(),
        v=(10.0 * np.sin(2.0 * np.pi * np.arange(n) / n)).tolist(),
    )
    s = Pic1dSettings(
        gap_m=0.01, n_cells=16, init_density_m3=1.0e12, n_macro=200,
        dt=1e-10, n_steps=20, frame_every=20, phase_bins=8,
        left={
            "v_dc": 0.0,
            "voltage_rf": {"amplitude": 20.0, "freq_hz": rf_freq, "phase_deg": 0.0},
            "waveforms": [wf.model_dump()],
        },
        right={"v_dc": 0.0},
    )
    sim = Pic1dSimulation(_project(s))
    assert sim._cycle_enabled
    assert sim._cycle_freq == pytest.approx(rf_freq)
    sim.run_batch()
    assert sim.cycle is not None
    assert sim.cycle["freq_hz"] == pytest.approx(rf_freq)


def test_dual_frequency_voltage_rf_runs():
    """voltage_rf をリスト2成分 (デュアル周波数) で指定して実行できること (有限値チェック)。"""
    s = Pic1dSettings(
        gap_m=0.01, n_cells=16, init_density_m3=1.0e12, n_macro=200,
        dt=1e-10, n_steps=50, frame_every=50,
        left={
            "v_dc": 0.0,
            "voltage_rf": [
                {"amplitude": 50.0, "freq_hz": 13.56e6, "phase_deg": 0.0},
                {"amplitude": 10.0, "freq_hz": 2.0e6, "phase_deg": 45.0},
            ],
        },
        right={"v_dc": 0.0},
    )
    sim = Pic1dSimulation(_project(s))
    sim.run_batch()
    assert sim.fields is not None
    for key in ("phi", "e", "n_e", "n_i", "t_e", "ionization"):
        assert np.all(np.isfinite(sim.fields[key])), key
