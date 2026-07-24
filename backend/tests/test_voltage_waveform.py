"""電極電位の CSV 波形インポート (1周期ループ) のテスト (prompts/73)。

1. 波形評価の単体: 三角波1周期を CSV 相当のサンプルで与え、周期折返しを
   含む複数時刻で線形補間+ループの期待値と一致すること
2. PIC スモーク: Dirichlet 辺に voltage_waveform を設定して数十ステップ実行し、
   エラーなく完走・診断値が有限であること
3. schema validator: phase 非昇順 / 長さ不一致 / freq 0 が ValidationError
"""

import numpy as np
import pytest
from pydantic import ValidationError

from es_sim.pic import PicSimulation, _eval_waveform
from es_sim.schema import Project, VoltageWaveform

L = 0.02
H = 0.01

# 三角波: phase=[0, 0.25, 0.75] (昇順・[0,1))、v=[0, 1, -1]。
# 折返し (phase[-1]=0.75 → phase[0]+1=1.0) は v=-1 → v=0 の線形補間で連続に繋がる
TRI_PHASE = [0.0, 0.25, 0.75]
TRI_V = [0.0, 1.0, -1.0]


# ---- 1. 波形評価の単体テスト ----------------------------------------------------


def test_eval_waveform_triangle_periodic():
    """t = 0, T/4, T/2, 3T/4, T, 1.25T の V_wf が期待値と一致する (freq_hz=1 なので T=1)。"""
    freq_hz = 1.0
    ts = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    # 期待値:
    #  t=0    (phase 0.00): サンプル通り v=0
    #  t=0.25 (phase 0.25): サンプル通り v=1
    #  t=0.5  (phase 0.50): 0.25→0.75 (v=1→-1) の中点 = 0
    #  t=0.75 (phase 0.75): サンプル通り v=-1
    #  t=1.0  (phase 0.00 に周期折返し): v=0 (t=0 と同値)
    #  t=1.25 (phase 0.25 に周期折返し): v=1 (t=T/4 と同値)
    expected = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0]
    for t, exp in zip(ts, expected):
        got = _eval_waveform(TRI_PHASE, TRI_V, freq_hz, t)
        assert got == pytest.approx(exp, abs=1e-12)

    # ベクトル化呼び出し (t を配列で渡す) も同じ結果になること
    got_vec = _eval_waveform(TRI_PHASE, TRI_V, freq_hz, np.asarray(ts))
    assert np.allclose(got_vec, expected, atol=1e-12)

    # 周波数を変えても V_wf(t) = f(t·freq_hz) としてスケールするだけ
    freq_hz2 = 2.0e6
    period2 = 1.0 / freq_hz2
    for frac, exp in zip([0.0, 0.25, 0.5, 0.75, 1.0, 1.25], expected):
        got = _eval_waveform(TRI_PHASE, TRI_V, freq_hz2, frac * period2)
        assert got == pytest.approx(exp, abs=1e-9)


# ---- 2. PIC スモーク ------------------------------------------------------------


def _waveform_project(n_steps: int = 30) -> Project:
    return Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [L, 0], [L, H], [0, H]]},
                "boundaries": [
                    {
                        "edges": [3],
                        "type": "dirichlet",
                        "voltage": 0.0,
                        "voltage_waveform": {
                            "freq_hz": 5.0e6,
                            "phase": TRI_PHASE,
                            "v": [0.0, 50.0, -50.0],
                        },
                    },
                    {"edges": [1], "type": "dirichlet", "voltage": 0.0},
                ],
            },
            "mesh": {"size": 1.5e-3},
            "pic": {
                "initial_plasma": {
                    "density": 1e14, "te_ev": 2.0, "ti_ev": 0.03,
                    "ion_mass_amu": 40.0, "seed": 0,
                },
                "n_macro": 2000,
                "dt": 1e-10,
                "n_steps": n_steps,
                "frame_every": n_steps,
            },
        }
    )


def test_pic_smoke_with_voltage_waveform():
    """CSV 波形付き Dirichlet 辺で数十ステップ実行し、診断値が有限であること。"""
    sim = PicSimulation(_waveform_project(n_steps=30))
    history, _ = sim.run_batch()
    for key in ("ke_e", "fe", "phi_min", "phi_max"):
        arr = np.asarray(history[key])
        assert np.all(np.isfinite(arr)), f"{key} に非有限値"

    # 波形電極の V(t) が実際に時間変化していること (定数ではない)
    wf_rows = np.nonzero(sim._wf_index >= 0)[0]
    assert len(wf_rows) > 0
    vals = [sim._dirichlet_values(t)[wf_rows[0]] for t in (0.0, 1e-8, 2e-8, 3e-8)]
    assert len(set(np.round(vals, 6))) > 1


def test_pic_dirichlet_values_matches_waveform_eval():
    """PIC の Dirichlet 値が _eval_waveform の直接評価と一致する。"""
    sim = PicSimulation(_waveform_project(n_steps=5))
    wf_rows = np.nonzero(sim._wf_index >= 0)[0]
    assert len(wf_rows) > 0
    freq_hz = 5.0e6
    for t in (0.0, 3.0e-8, 7.3e-8):
        vd = sim._dirichlet_values(t)
        expected = _eval_waveform(TRI_PHASE, [0.0, 50.0, -50.0], freq_hz, t)
        assert np.allclose(vd[wf_rows], expected, atol=1e-9)


# ---- 3. schema validator --------------------------------------------------------


def test_schema_phase_not_ascending():
    with pytest.raises(ValidationError):
        VoltageWaveform(freq_hz=1.0e6, phase=[0.0, 0.75, 0.25], v=[0.0, 1.0, -1.0])


def test_schema_length_mismatch():
    with pytest.raises(ValidationError):
        VoltageWaveform(freq_hz=1.0e6, phase=[0.0, 0.5], v=[0.0, 1.0, -1.0])


def test_schema_freq_zero():
    with pytest.raises(ValidationError):
        VoltageWaveform(freq_hz=0.0, phase=[0.0, 0.5], v=[0.0, 1.0])


def test_schema_too_few_points():
    with pytest.raises(ValidationError):
        VoltageWaveform(freq_hz=1.0e6, phase=[0.0], v=[0.0])


def test_schema_phase_out_of_range():
    with pytest.raises(ValidationError):
        VoltageWaveform(freq_hz=1.0e6, phase=[0.0, 1.0], v=[0.0, 1.0])
