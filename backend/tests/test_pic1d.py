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
10. FN 電界放出 (prompts/95): 放出重みが実際に使われた表面電流密度の積分と一致
    (端数キャリー1個未満の誤差)、逆向き電界からは放出ゼロ、続きから実行がビット一致、
    validator (phi_ev<=0 等)
11. Brinkmann シースエッジ検出 (prompts/97): 段差プロファイル・線形ランプの解析解
    (後者は scipy.optimize.brentq による独立な数値根探索でも確認)、右電極の鏡映対称性、
    プラズマ無しで None、CCP スモーク結果への搭載
12. シース振動 FFT (prompts/100): 合成系列 (基本波+第3高調波) の振幅定量検証
    (整数周期トリムが効いている証拠、rtol 1e-2)、内部孤立 NaN の線形補間、
    有効率50%未満での縮退 (None)、f0 無指定時の帯域上限、RF付き CCP スモークでの
    result.sheath_fft/sheath_ts の搭載確認
"""

import json
import math

import numpy as np
import pydantic
import pytest
from fastapi.testclient import TestClient
from scipy import integrate, optimize

import es_sim.server as server
from es_sim.mcc import KB
from es_sim.particles import ME, QE
from es_sim.pic1d import Pic1dSimulation, brinkmann_sheath_edge, build_pic1d_result, sheath_fft
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


# ---- 10. FN 電界放出 (prompts/95) --------------------------------------------------
#
# 空間電荷の影響を除くため初期プラズマ (electron/ion) の重みをゼロにし、電界を
# 境界電圧だけで決まる値 (± FN 放出電子自身の空間電荷による僅かな摂動のみ) にする
# (test_leapfrog_matches_analytic_uniform_field と同じ手法)。


def _zero_background_weight(sim: Pic1dSimulation) -> None:
    for name in ("electron", "ion"):
        sp = sim.species[name]
        sp.w = np.zeros_like(sp.w)


def _fn_accounting_settings(n_steps: int) -> Pic1dSettings:
    # gap=1um・V=100V (beta=50 でF~1e8 V/m級) → 有意な FN 電流が出る条件 (prompts/95 の例)。
    # 右電極のみ fn を設定 (左は not-emitting 方向の検証に使う)
    return Pic1dSettings(
        gap_m=1e-6, n_cells=10, init_density_m3=1.0e6, n_macro=2,
        dt=1e-16, n_steps=n_steps, frame_every=n_steps,
        left={"v_dc": 0.0},
        right={"v_dc": -100.0, "fn": {"phi_ev": 4.5, "beta": 50.0, "macro_weight": 1e12}},
    )


def test_fn_emission_matches_integrated_current():
    """放出重みの積算 ≈ 実際に使われた J(F_t) を積分した値 (誤差はマクロ1個未満、prompts/95)。

    quota_t = J_t·dt/(e·w) + frac_{t-1}、n_emit_t = floor(quota_t) の毎ステップ更新を
    Σ で telescope すると Σn_emit_t = ΣJ_t·dt/(e) /w  - frac_T (frac_0=0) になるので、
    総放出重み = w·Σn_emit_t は Σ J_t·dt/e よりちょうど [0, w) 小さいはずである。
    """
    sim = Pic1dSimulation(_project(_fn_accounting_settings(10)))
    _zero_background_weight(sim)

    captured_j_right: list[float] = []
    orig_emit = sim._emit_fn

    def spy(ex: np.ndarray):
        result = orig_emit(ex)
        captured_j_right.append(result[1])
        return result

    sim._emit_fn = spy
    sim.run_batch()

    assert sim.fn_events["right"] > 0  # 実際に何か放出されていること
    expected_w = sum(captured_j_right) * sim.dt / QE
    w_fn = sim._fn_w["right"]
    diff = expected_w - sim.fn_total_w["right"]
    assert 0.0 <= diff < w_fn * (1.0 + 1e-9)

    # 左電極は fn 未設定なので放出処理自体が発生しない
    assert sim.fn_events["left"] == 0
    assert sim.fn_total_w["left"] == 0.0


def test_fn_no_emission_when_field_points_wrong_way():
    """引き出し電界が逆向き (右電極側から見て F<0) の電極からは放出ゼロ (prompts/95)。"""
    s = _fn_accounting_settings(10)
    # 左右電圧を反転し、右電極 (fn 設定側) の表面電界を「電子を引き込む」向きにする
    s.left.v_dc, s.right.v_dc = s.right.v_dc, s.left.v_dc
    sim = Pic1dSimulation(_project(s))
    _zero_background_weight(sim)
    sim.run_batch()
    assert sim.fn_events["right"] == 0
    assert sim.fn_total_w["right"] == 0.0
    assert all(w == 0.0 for w in sim.history["fn_right"])


def test_fn_continue_is_bit_identical_to_single_run():
    """FN ありでも run(n)+continue(m) が run(n+m) とビット一致する (端数キャリーも維持、prompts/95)。"""

    def make(n_steps: int) -> Pic1dSettings:
        return Pic1dSettings(
            gap_m=1e-6, n_cells=10, init_density_m3=1.0e6, n_macro=2,
            dt=1e-16, n_steps=n_steps, frame_every=n_steps,
            left={"v_dc": 0.0, "fn": {"phi_ev": 4.5, "beta": 40.0, "macro_weight": 5e11}},
            right={"v_dc": 100.0, "fn": {"phi_ev": 4.5, "beta": 40.0, "macro_weight": 5e11}},
        )

    sim_a = Pic1dSimulation(_project(make(20)))
    _zero_background_weight(sim_a)
    sim_a.s.avg_steps = 10
    sim_a.run_batch()

    sim_b = Pic1dSimulation(_project(make(10)))
    _zero_background_weight(sim_b)
    sim_b.run_batch()
    sim_b.prepare_continue(10, avg_steps=10)
    sim_b.run_batch()

    for name in ("electron", "ion"):
        sa, sb = sim_a.species[name], sim_b.species[name]
        assert len(sa.x) == len(sb.x) and len(sa.x) > 0
        assert np.array_equal(sa.x, sb.x)
        assert np.array_equal(sa.v, sb.v)
        assert np.array_equal(sa.w, sb.w)

    assert sim_a.fn_events == sim_b.fn_events
    assert sim_a.fn_total_w == sim_b.fn_total_w
    assert sim_a.fn_events["left"] > 0  # 実際に FN 放出が起きた条件であること (退化ケース回避)
    assert sim_a.fn == sim_b.fn


def test_fn_validators_raise():
    with pytest.raises(pydantic.ValidationError):
        Pic1dSettings(
            gap_m=0.01, init_density_m3=1e14, left={"fn": {"phi_ev": 0.0}},
        )
    with pytest.raises(pydantic.ValidationError):
        Pic1dSettings(
            gap_m=0.01, init_density_m3=1e14, right={"fn": {"beta": 0.0}},
        )
    with pytest.raises(pydantic.ValidationError):
        Pic1dSettings(
            gap_m=0.01, init_density_m3=1e14, left={"fn": {"init_energy_ev": -1.0}},
        )
    with pytest.raises(pydantic.ValidationError):
        Pic1dSettings(
            gap_m=0.01, init_density_m3=1e14, left={"fn": {"macro_weight": 0.0}},
        )


# ---- 11. Brinkmann シースエッジ検出 (prompts/97) ------------------------------------


def test_brinkmann_step_profile_exact_root():
    """段差プロファイル (n_i=n0 一様、n_e は d を境に 0→n0) → s=d に一致 (rtol 1e-9)。

    d をセル中点に置くのがポイント: 台形則は区間の両端値の平均×幅で積分するため、
    真の (連続な) 段差がちょうど区間の中点にあるとき、区間全体の真の面積
    n0×(区間幅/2) と台形則の見積り (0+n0)/2×区間幅 = n0×区間幅/2 が厳密に一致する。
    このため d が節点上になくても離散実装で厳密に d を再現でき、rtol 1e-9 の
    厳しい比較ができる (d を区間内の任意点にすると離散化誤差が乗り、雑な近似
    比較になってしまう)。
    """
    n0 = 3.0e15
    n = 21
    length = 0.02
    x = np.linspace(0.0, length, n)
    k = 7
    d = 0.5 * (x[k] + x[k + 1])  # セル中点 (節点上には無い)
    n_i = np.full(n, n0)
    n_e = np.where(x < d, 0.0, n0)

    s = brinkmann_sheath_edge(x, n_e, n_i, True, length)
    assert s is not None
    assert s == pytest.approx(d, rel=1e-9)


def test_brinkmann_linear_ramp_matches_hand_derivation():
    """線形ランプ n_i=n0 (一様)、n_e=n0·x/x_b (0≤x≤x_b) の根を手計算で導出し検証する。

    G(s) = ∫₀ˢ n_e dx − ∫ₛ^{x_b} (n_i−n_e) dx
         = n0 s²/(2x_b) − [ n0(x_b−s) − n0(x_b²−s²)/(2x_b) ]
         = n0 s²/(2x_b) + n0(x_b²−s²)/(2x_b) − n0(x_b−s)
    s² の項に着目すると n0 s²/(2x_b) − n0 s²/(2x_b) = 0 で厳密に打ち消し合うため、
    G(s) は s の1次式になる:
         = n0 x_b²/(2x_b) − n0(x_b−s) = n0 x_b/2 − n0 x_b + n0 s = n0 (s − x_b/2)
    ⇒ 閉形式の根は厳密に s = x_b/2 (プロンプト中の「x_b(√2−1)」という仮の見立ては
    誤りで、実際には s²項が打ち消し合うため単純な中点になる)。

    上の手計算とは独立に、連続関数を scipy.integrate.quad で数値積分し
    scipy.optimize.brentq で G(s)=0 を直接根探索した値でも同じ x_b/2 になることを
    確認し (雑な近似比較にしないための二重チェック)、その上で離散実装 (台形則、
    かつ x_b をあえて格子点からずらすケース) の結果とも突き合わせる。
    """
    n0 = 2.0e15
    x_b = 0.01

    def n_e_cont(xv: float) -> float:
        return n0 * xv / x_b

    def n_i_cont(_xv: float) -> float:
        return n0

    def g_cont(s: float) -> float:
        int_e, _ = integrate.quad(n_e_cont, 0.0, s)
        int_net, _ = integrate.quad(lambda xv: n_i_cont(xv) - n_e_cont(xv), s, x_b)
        return int_e - int_net

    s_numeric = optimize.brentq(g_cont, 1e-9 * x_b, x_b * (1.0 - 1e-9))
    assert s_numeric == pytest.approx(x_b / 2.0, rel=1e-9)

    # 離散実装: 格子は x_b を超えて延びる (x_b が節点上に無いケースも兼ねて検証)。
    # n_e = n0·x/x_b という同一の1次式を x_b の外側までそのまま延長する (x_b 以遠は
    # 積分に使われないので物理的な意味は不要)。これにより x_b をまたぐ区間の
    # 内挿点でも「同一の直線」上の値になり、折れ線 (別の関数への切り替え) による
    # 補間誤差が入らない — 台形則・線形補間はどちらも1次式に対して厳密なため
    n = 33
    length = 0.023
    x = np.linspace(0.0, length, n)
    n_i = np.full(n, n0)
    n_e = n0 * x / x_b

    s = brinkmann_sheath_edge(x, n_e, n_i, True, x_b)
    assert s is not None
    assert s == pytest.approx(x_b / 2.0, rel=1e-9)
    assert s == pytest.approx(s_numeric, rel=1e-9)


def test_brinkmann_right_electrode_mirrors_left():
    """左のケース (段差プロファイル) を反転した配列を渡すと、右電極判定が同じ s を返す。

    x が [0,L] の等間隔格子のとき x[::-1] の位置は L−x と一致するため、n_e/n_i を
    そのまま反転した配列は「右電極からの距離で見た元のプロファイル」と同一になる。
    brinkmann_sheath_edge(..., from_left=False) は内部で距離 d=x[-1]−x に鏡映するため、
    反転済み配列を渡すとちょうど元の (反転前の) 配列に戻り、左電極判定と同じ根に
    なるはずである。
    """
    n0 = 3.0e15
    n = 21
    length = 0.02
    x = np.linspace(0.0, length, n)
    k = 7
    d = 0.5 * (x[k] + x[k + 1])
    n_i = np.full(n, n0)
    n_e = np.where(x < d, 0.0, n0)

    s_left = brinkmann_sheath_edge(x, n_e, n_i, True, length)
    s_right = brinkmann_sheath_edge(x, n_e[::-1].copy(), n_i[::-1].copy(), False, length)
    assert s_left is not None
    assert s_right == pytest.approx(s_left, rel=1e-9)


def test_brinkmann_no_plasma_returns_none():
    """n_e=n_i=0 (プラズマ未形成) は左右どちらの判定でも None を返す。"""
    x = np.linspace(0.0, 0.02, 21)
    zeros = np.zeros_like(x)
    assert brinkmann_sheath_edge(x, zeros, zeros, True, 0.01) is None
    assert brinkmann_sheath_edge(x, zeros, zeros, False, 0.01) is None


def test_brinkmann_sheath_edge_in_ccp_result():
    """CCP スモーク (既存の短い Ar 実行を流用) で result.sheath / cycle.sheath を確認する。"""
    s = _edupic_smoke_settings(2400)
    sim = Pic1dSimulation(_project(s))
    sim.run_batch()
    result = build_pic1d_result(sim, 0.0)

    assert result["sheath"] is not None
    half_gap = sim.gap / 2.0
    left_s = result["sheath"]["left_s"]
    right_s = result["sheath"]["right_s"]
    assert left_s is not None and 0.0 < left_s < half_gap
    assert right_s is not None and 0.0 < right_s < half_gap

    if result["cycle"] is not None:
        cyc_sheath = result["cycle"]["sheath"]
        bins = result["cycle"]["bins"]
        assert len(cyc_sheath["s_left"]) == bins
        assert len(cyc_sheath["s_right"]) == bins


# ---- 12. シース振動 FFT (prompts/100) --------------------------------------------------


def _synthetic_sheath_series(f0: float, n_total: int, s0: float, a1: float, a2: float, dt: float) -> np.ndarray:
    """s(t) = s0 + A1·sin(2π f0 t) + A2·sin(2π·3f0 t) の合成系列 (窓はあえて f0 の非整数周期)。"""
    t = np.arange(n_total) * dt
    return s0 + a1 * np.sin(2.0 * np.pi * f0 * t) + a2 * np.sin(2.0 * np.pi * 3.0 * f0 * t)


def test_sheath_fft_synthetic_amplitudes():
    """合成系列 (基本波 f0 + 第3高調波) の FFT が mean≈s0、f0/3f0 ビンの振幅≈A1/A2
    (rtol 1e-2)、他ビンは A1 の1%未満になることを検証する (整数周期トリムが効いている
    証拠: 窓長 5432 サンプルは f0 の 1 周期=1000 サンプルの非整数倍だが、トリム後は
    ちょうど整数周期になるためリークがほぼ皆無になる)。
    """
    f0 = 1.0e6
    dt = 1.0e-9  # 1周期=1000サンプル (きれいな整数)
    n_total = 5432  # 1000 の非倍数 (非整数周期の窓であることを保証)
    s0, a1, a2 = 1.0e-3, 2.0e-4, 5.0e-5
    s = _synthetic_sheath_series(f0, n_total, s0, a1, a2, dt)

    result = sheath_fft(s, dt, f0)
    assert result is not None
    assert result["mean"] == pytest.approx(s0, rel=1e-2)
    assert result["f0_hz"] == pytest.approx(f0)
    assert result["n_samples"] < n_total  # 実際にトリムされていること

    freq = result["freq_hz"]
    amp = result["amp"]
    df = result["df_hz"]
    idx1 = int(round(f0 / df))
    idx2 = int(round(3.0 * f0 / df))
    assert freq[idx1] == pytest.approx(f0, rel=1e-6)
    assert freq[idx2] == pytest.approx(3.0 * f0, rel=1e-6)
    assert amp[idx1] == pytest.approx(a1, rel=1e-2)
    assert amp[idx2] == pytest.approx(a2, rel=1e-2)

    other = np.delete(amp, [0, idx1, idx2])
    assert np.all(other < 0.01 * a1)


def test_sheath_fft_nan_interpolation_and_degenerate_below_half():
    """内部の孤立 NaN を数点入れても振幅検証が成立すること (線形補間)、
    および NaN が過半数を占めれば None (縮退) になることを確認する。"""
    f0 = 1.0e6
    dt = 1.0e-9
    n_total = 5432
    s0, a1, a2 = 1.0e-3, 2.0e-4, 5.0e-5
    s = _synthetic_sheath_series(f0, n_total, s0, a1, a2, dt)

    s_nan = s.copy()
    rng = np.random.default_rng(0)
    # 先頭/末尾を除く内部に孤立 NaN を数点だけ入れる (有効率は50%を大きく上回る)
    idx = rng.choice(np.arange(10, n_total - 10), size=8, replace=False)
    s_nan[idx] = np.nan

    result = sheath_fft(s_nan, dt, f0)
    assert result is not None
    assert result["mean"] == pytest.approx(s0, rel=1e-2)
    freq = result["freq_hz"]
    amp = result["amp"]
    df = result["df_hz"]
    idx1 = int(round(f0 / df))
    idx2 = int(round(3.0 * f0 / df))
    assert amp[idx1] == pytest.approx(a1, rel=1e-2)
    assert amp[idx2] == pytest.approx(a2, rel=1e-2)

    # 過半数を NaN にすると有効率50%未満になり None を返す。ただし先頭/末尾の連続 NaN は
    # トリムされてしまい「有効率」の分母に入らないため、両端は有効値のままにして
    # 内部にランダムに散らす (先頭寄りの連続区間を NaN にする単純な書き方では、
    # 単にトリムされて残りが全部有効になってしまい意図した縮退にならない)
    s_degenerate = s.copy()
    interior = np.arange(1, n_total - 1)
    n_nan = int(0.6 * n_total)
    nan_idx = rng.choice(interior, size=n_nan, replace=False)
    s_degenerate[nan_idx] = np.nan
    assert sheath_fft(s_degenerate, dt, f0) is None


def test_sheath_fft_all_nan_returns_none():
    assert sheath_fft(np.full(100, np.nan), 1e-9, 1e6) is None


def test_sheath_fft_no_f0_uses_full_series_and_caps_bins():
    """f0=None なら整数周期トリムをせず全系列をそのまま使い、返す帯域は
    低周波側 2048 ビンまでに絞る (Nyquist 全帯域を返す必要はないため)。"""
    dt = 1.0e-9
    n_total = 6000  # rfft のビン数は 3001 (> 2048) になる想定
    t = np.arange(n_total) * dt
    s = 1.0e-3 + 2.0e-4 * np.sin(2.0 * np.pi * 1.0e6 * t)
    result = sheath_fft(s, dt, None)
    assert result is not None
    assert result["f0_hz"] is None
    assert result["n_samples"] == n_total
    assert len(result["freq_hz"]) == 2048
    assert len(result["amp"]) == 2048


def test_sheath_fft_and_ts_in_ccp_result():
    """RF付き CCP スモークで result.sheath_fft が非null、freq_hz の上限≈40·f0、
    amp配列長が freq_hz と一致、result.sheath_ts が ≤2048 点になること (prompts/100)。
    """
    s = _edupic_smoke_settings(2400)
    sim = Pic1dSimulation(_project(s))
    sim.run_batch()
    result = build_pic1d_result(sim, 0.0)

    fft = result["sheath_fft"]
    assert fft is not None
    f0 = fft["f0_hz"]
    assert f0 is not None and f0 == pytest.approx(sim._cycle_freq)
    assert len(fft["freq_hz"]) == len(fft["amp_left"]) == len(fft["amp_right"])
    # 上限は 40·f0 (端数ビンの丸めで多少前後する)。実際に高調波帯域まで返っていることも確認
    assert fft["freq_hz"][-1] <= 40.0 * f0 + fft["df_hz"] + 1e-6
    assert fft["freq_hz"][-1] > 30.0 * f0
    assert fft["mean_left"] is not None and math.isfinite(fft["mean_left"])
    assert fft["mean_right"] is not None and math.isfinite(fft["mean_right"])

    ts = result["sheath_ts"]
    assert ts is not None
    assert len(ts["t"]) <= 2048
    assert len(ts["t"]) == len(ts["s_left"]) == len(ts["s_right"])
