"""収束の判定の部品のテスト (convergence.py、prompts/137)。人工の周期平均の列で判定の振る舞いを確かめる。"""

import json
import math

import numpy as np
import pytest

from es_sim.convergence import (
    MIN_SAMPLES,
    TOL_FLUID,
    TOL_PIC,
    BlockMap,
    ConvergenceMonitor,
    PeriodSums,
    circuit_vdc,
    make_monitor,
)
from es_sim.schema import ConvergenceSettings


def _monitor(kind: str = "fluid", t: float = 0.0, **kw) -> ConvergenceMonitor:
    return ConvergenceMonitor(ConvergenceSettings(**kw), kind=kind, rf_period_s=1.0, dt=0.01, frame_every=10, t=t)


def _feed(mon: ConvergenceMonitor, samples: list[dict], blocks: BlockMap | None = None, circuit=None) -> None:
    """周期ごとの値 (量の名前 → 周期平均) を 1 周期ずつ足して閉じる。"""
    for s in samples:
        for name, v in s.items():
            mon.sums.add(name, v)
        t = (mon._k + 1) * mon.period_s
        assert mon.boundary_passed(t)
        mon.close_period(t, round(t / mon.dt), blocks=blocks, circuit=circuit)


def test_slow_exponential_approach_waits_for_the_remaining_change():
    """時定数 300 周期で近づく量: 前の周期との比較 (0.1%) は 300 周期より前に満たすが、残りの変化が 0.1% に
    なる周期 (≈ τ ln 500 ≈ 1864) まで収束としない。"""
    tau = 300.0
    vals = [1.0 - 0.5 * math.exp(-k / tau) for k in range(1, 2201)]
    naive = next(k for k in range(1, len(vals)) if abs(vals[k] - vals[k - 1]) / vals[k] <= TOL_FLUID)
    assert naive < 300
    mon = _monitor()
    _feed(mon, [{"N_e": v} for v in vals])
    assert mon.converged
    expected = tau * math.log(0.5 / TOL_FLUID)
    assert expected - 5 < mon.converged_at["period"] < expected + 15
    # 窓は 1 周期 (決定的で雑音が無い)。残りの変化は減り方からの外挿 (decay)
    assert {w for w in mon.history["window"] if w is not None} == {1}
    assert mon.history["r_kind"]["N_e"][-1] == "decay"


def test_slow_decay_early_in_the_run_is_not_passed():
    """時定数 2 万周期で近づく量 (1 周期の変化 2.5e-5、残り約 50%): 外挿はこれまでの長さの 10 倍の先まで見るので、
    初めの 40 周期でも合格させない (これまでの長さだけなら 7 周期で 1.75e-4 と合格してしまう)。"""
    tau = 2.0e4
    mon = _monitor()
    _feed(mon, [{"N_e": 1.0 - 0.5 * math.exp(-k / tau)} for k in range(1, 41)])
    assert not mon.converged
    judged = [s for s in mon.history["status"] if s in ("pass", "fail")]
    assert judged and set(judged) == {"fail"}
    assert all(r > TOL_FLUID for r in mon.history["r"]["N_e"] if r is not None)


def test_white_noise_widens_the_window_and_converges():
    """周期ごとに 1% の白色雑音だけの量 (PIC 相当、閾値 1%) は、窓が雑音に合わせて広がって収束する。"""
    rng = np.random.default_rng(1)
    mon = _monitor(kind="pic")
    assert mon.tol == TOL_PIC
    _feed(mon, [{"N_e": 1.0 + 0.01 * rng.standard_normal()} for _ in range(200)])
    assert mon.converged
    assert mon.converged_at["period"] < 100
    windows = [w for w in mon.history["window"] if w is not None]
    assert 3 <= max(windows) <= mon.max_window
    # 雑音だけなので、雑音を差し引いた変化 D は閾値より十分小さいことが多い
    d = [v for v in mon.history["d"]["N_e"] if v is not None]
    assert np.median(d) < 0.5 * TOL_PIC


def test_slow_drift_hidden_in_noise_is_found_with_longer_windows():
    """0.5% の雑音に 1 周期 0.1% の傾き: 雑音から決めた窓 (2 周期) の変化は閾値より小さいが、窓を広げて傾きを
    見つけ (減っていない)、収束としない。傾きが止まってから収束する。"""
    rng = np.random.default_rng(2)
    mon = _monitor(kind="pic")
    drift = [1.0 + 1.0e-3 * k + 0.005 * rng.standard_normal() for k in range(300)]
    _feed(mon, [{"N_e": v} for v in drift])
    assert not mon.converged
    # 減っていると言えない傾き: 今の傾きでこれまでの長さの 10 倍 (trend)
    assert mon.history["r_kind"]["N_e"][-1] == "trend"
    f = mon.frame()
    assert f["worst_name"] == "N_e" and f["worst_kind"] == "trend" and f["worst"] > TOL_PIC
    flat = [drift[-1] + 0.005 * rng.standard_normal() for _ in range(200)]
    _feed(mon, [{"N_e": v} for v in flat])
    assert mon.converged
    assert mon.converged_at["period"] > 300


def test_constant_quantity_converges_after_hold_checks():
    """変わらない量は、判定に要る周期 (MIN_SAMPLES) から合格し、hold 回続いたところで収束。"""
    mon = _monitor(hold=3)
    _feed(mon, [{"N_e": 2.0, "N_i": 2.0} for _ in range(12)])
    status = mon.history["status"]
    assert status[: MIN_SAMPLES - 1] == ["warming"] * (MIN_SAMPLES - 1)
    assert status[MIN_SAMPLES - 1:] == ["pass"] * (12 - MIN_SAMPLES + 1)
    assert mon.converged_at["period"] == MIN_SAMPLES - 1 + 2
    assert mon.history["r_kind"]["N_e"][-1] == "none"               # 有意な変化が無い
    assert mon.history["r_kind"]["N_e"][0] is None                    # 判定中は無し


def test_one_failing_quantity_blocks_and_resets_the_hold():
    """1 つの量が跳ぶと合格しない。跳んだ所が雑音の見積もりや比べる窓に入っている間は合格せず、外れてから合格に
    戻る。収束の記録は最初の 1 回のまま。"""
    mon = _monitor(hold=3)
    jump = MIN_SAMPLES + 2                          # 跳ぶ前に 3 回合格する
    samples = [{"N_e": 1.0, "N_i": 1.0} for _ in range(jump)]
    samples.append({"N_e": 1.0, "N_i": 1.5})       # イオンだけ跳ぶ → 合格しない
    samples += [{"N_e": 1.0, "N_i": 1.5} for _ in range(30)]
    _feed(mon, samples)
    status = mon.history["status"]
    assert status[jump] in ("fail", "noisy")      # 跳びは三階差分を大きくするので「判定できない」こともある
    assert "pass" not in status[jump:jump + 8]
    assert "fail" in status[jump:jump + 12]        # 窓が跳びをまたぐ間は残りの変化を見積もって合格しない
    assert mon.converged          # 跳ぶ前に 3 回続けて合格していた
    assert mon.converged_at["period"] == MIN_SAMPLES - 1 + 2
    assert mon.now_passing         # 跳んだあとまた落ち着いた


def test_fields_are_compared_on_blocks_and_zero_fields_pass():
    """場はブロックの体積平均で比べる。0 の場 (プラズマが消えた) は変化 0 で合格。"""
    x = np.linspace(0.0, 1.0, 101)
    vol = np.full(101, 0.01)
    blocks = BlockMap(x, vol, 10)
    mon = _monitor()
    phi = np.sin(np.pi * x)
    _feed(mon, [{"phi": phi * (1.0 + 0.1 * 0.5**k), "n_e": np.zeros(101)} for k in range(30)], blocks=blocks)
    assert mon.converged
    assert all(v == 0.0 for v in mon.history["d"]["n_e"] if v is not None)
    assert mon.blocks == (10,)


def test_block_map_volume_weighted_means():
    rng = np.random.default_rng(3)
    xy = rng.random((500, 2)) * [2.0, 1.0]
    vol = rng.random(500) + 0.1
    bm = BlockMap(xy, vol, 32)
    nx, ny = bm.shape
    assert abs(nx / ny - 2.0) < 0.6 and abs(nx * ny - 32) <= 4
    f = rng.random(500)
    proj = bm.project(f)
    for b in range(bm.n):
        sel = bm.index == b
        if sel.any():
            assert proj[b] == pytest.approx(np.sum(vol[sel] * f[sel]) / np.sum(vol[sel]))
            assert bm.weights[b] == pytest.approx(np.sum(vol[sel]))
        else:
            assert proj[b] == 0.0 and bm.weights[b] == 0.0


class _FakeCircuit:
    def __init__(self, period: float):
        self.period = period
        self.labels = ["rf"]
        self.history = {"t": [], "v_dc": [], "v1": [], "i_dc": []}

    def add(self, t: float, v_dc: float, v1: float) -> None:
        self.history["t"].append(t)
        self.history["v_dc"].append([v_dc])
        self.history["v1"].append([v1])
        self.history["i_dc"].append([0.0])


def test_circuit_vdc_selects_the_rf_periods_of_the_sampling_period():
    c = _FakeCircuit(1.0)
    for k in range(1, 7):
        c.add(float(k), -10.0 * k, 100.0 + k)
    assert circuit_vdc(c, 2.0, 4.0) == [("rf", -35.0, 103.5)]
    assert circuit_vdc(c, 6.0, 8.0) == []
    assert circuit_vdc(None, 0.0, 1.0) == []


def test_self_bias_is_scaled_by_the_rf_amplitude():
    """V_dc は |V1| を尺度に比べる (V_dc が 0 の近くでも判定できる)。時定数 50 周期で 0 → −80 V、|V1| = 100 V。"""
    c = _FakeCircuit(1.0)
    mon = _monitor()
    for k in range(1, 801):
        c.add(float(k), -80.0 * (1.0 - math.exp(-k / 50.0)), 100.0)
        mon.sums.add("N_e", 1.0)
        mon.close_period(float(k), k, circuit=c)
    assert mon.converged
    # 残りの変化 80 e^{-k/50} / 100 が 0.1% になる周期 ≈ 50 ln 800 ≈ 334
    assert 330 < mon.converged_at["period"] < 345


def test_stop_only_when_convergence_is_found_during_the_run():
    mon = _monitor(stop=True, hold=2)
    mon.begin_run()
    _feed(mon, [{"N_e": 1.0} for _ in range(MIN_SAMPLES + 1)])
    assert mon.converged and mon.stop_now
    assert mon.stop_end(step_count=500, end=10_000, avg_steps=300, accum_start=9_701) == 800
    assert mon.stopped and not mon.stop_now
    assert mon.run_end == 800                       # ジョブの進捗の総ステップ数 (この実行の中だけ)
    # 平均区間で終わりが早まらなければ止めない。時間平均の区間がもう始まっていても止めない (今の終わりまで)
    mon2 = _monitor(stop=True, hold=2)
    mon2.begin_run()
    _feed(mon2, [{"N_e": 1.0} for _ in range(MIN_SAMPLES + 1)])
    assert mon2.stop_end(step_count=500, end=600, avg_steps=300, accum_start=301) is None and not mon2.stopped
    assert mon2.stop_end(step_count=500, end=10_000, avg_steps=300, accum_start=400) is None and not mon2.stopped
    assert mon2.run_end is None
    # 続きの実行 (収束したあと) では止めない
    mon.begin_run()
    assert mon.run_end is None
    _feed(mon, [{"N_e": 1.0} for _ in range(3)])
    assert not mon.stop_now
    # avg_steps が無ければ判定の周期 10 個分
    assert mon.default_avg_steps() == 1000


def test_partial_first_period_is_discarded():
    mon = _monitor(t=0.5)
    mon.sums.add("N_e", 1.0)
    assert not mon.boundary_passed(0.9) and mon.boundary_passed(1.0)
    mon.close_period(1.0, 100)
    assert mon.history["t"] == []
    _feed(mon, [{"N_e": 1.0}])
    assert mon.history["period"] == [1]


def test_period_from_steps_without_rf_and_pic_stride():
    s = ConvergenceSettings(steps=500)
    mon = ConvergenceMonitor(s, kind="pic", rf_period_s=None, dt=1e-9, frame_every=20)
    assert mon.basis == "steps" and mon.period_s == pytest.approx(500e-9)
    assert mon.stride == 500 // 64
    mon2 = ConvergenceMonitor(ConvergenceSettings(), kind="fluid", rf_period_s=None, dt=1e-9, frame_every=20)
    assert mon2.period_s == pytest.approx(200e-9) and mon2.stride == 1
    assert make_monitor(ConvergenceSettings(enabled=False), kind="fluid", rf_period_s=1.0, dt=0.01,
                        frame_every=1) is None


def test_period_sums_copy_the_first_value():
    ps = PeriodSums()
    a = np.ones(3)
    ps.add("x", a)
    a[:] = 5.0                     # 足したあとで元を書き換えても影響しない
    ps.add("x", np.full(3, 3.0))
    ps.add("s", 1.0)
    ps.add("s", 2.0)
    m = ps.means()
    assert np.allclose(m["x"], 2.0) and m["s"] == pytest.approx(1.5)
    ps.clear()
    assert len(ps) == 0


def test_result_and_frame_are_json_serialisable():
    rng = np.random.default_rng(4)
    mon = _monitor(kind="pic")
    _feed(mon, [{"N_e": 1.0 + 0.02 * k + 0.01 * rng.standard_normal()} for k in range(40)])
    r = mon.result()
    json.dumps(r, allow_nan=False)
    json.dumps(mon.frame(), allow_nan=False)
    assert len(r["t"]) == len(r["status"]) == len(r["d"]["N_e"]) == 40
    assert not r["converged"]
