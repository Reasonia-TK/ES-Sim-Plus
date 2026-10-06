"""流体 1D・2D の収束の判定 (prompts/137)。

生成・損失を止めて壁を反射にした閉じた系 (電子だけが RF の電位の中で落ち着く) はすぐ周期的な定常になるので、
それで判定の周期・記録・止める・続きを確かめる。v2 の GPU 版は CPU 版と同じ判定になる (CUDA が無ければ skip)。
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from es_sim.convergence import MIN_SAMPLES
from es_sim.device import cuda_available
from es_sim.fluid1d import Fluid1dSimulation, build_fluid1d_result
from es_sim.fluid2d import Fluid2dSimulation, build_fluid2d_result
from es_sim.schema import Project

F0 = 13.56e6
T = 1.0 / F0
SPP = 50
#: 閉じた系 (流体 1D) が収束と判定される周期 (判定を始める MIN_SAMPLES 周期の 1 回目は雑音の見積もりが落ち着かず、
#: その次から hold = 3 回続けて合格)。2D は始めの揺れで 1〜2 周期遅れることがある
CONVERGED_PERIOD = MIN_SAMPLES + 2
PERIODS_2D = 13


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def _closed(sim):
    """生成・損失の無い閉じた系にする (すぐ周期的な定常になる)。"""
    sim.debug_source_enabled = False
    sim.debug_energy_enabled = False
    sim.debug_ions_enabled = False
    sim.debug_reflective_walls = True
    return sim


def _p1d(periods: int, cap: dict | None = None, **conv) -> Project:
    return Project.model_validate({
        "geometry": {"domain": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}}, "mesh": {"size": 0.1},
        "fluid1d": {
            "gap_m": 0.01, "n_cells": 40,
            "left": {"voltage_rf": {"amplitude": 50.0, "freq_hz": F0}, "blocking_capacitor": cap},
            "init_density_m3": 1e13, "init_te_ev": 2.0, "gas_pressure_pa": 100.0, "dt": T / SPP,
            "n_steps": periods * SPP, "frame_every": SPP, "avg_steps": 2 * SPP, "phase_bins": 0,
            "convergence": conv,
        },
    })


def _p2d(periods: int, kind: str = "cartesian", **conv) -> Project:
    mesh = {"size": 0.5e-3, "mode": "structured" if kind == "v1" else "cartesian"}
    if kind == "amr":
        mesh["amr"] = {"max_level": 1, "buffer_cells": 2}
    return Project.model_validate({
        "geometry": {
            "domain": {"polygon": [[0, 0], [0.01, 0], [0.01, 0.005], [0, 0.005]]},
            "boundaries": [
                {"edges": [3], "type": "dirichlet", "voltage": 0.0, "voltage_rf": {"amplitude": 50.0, "freq_hz": F0}},
                {"edges": [1], "type": "dirichlet", "voltage": 0.0},
                {"edges": [0, 2], "type": "symmetry"},
            ],
            "regions": [{"id": "pin", "type": "conductor", "voltage": 0.0,
                         "polygon": [[0.006, 0.002], [0.007, 0.002], [0.007, 0.003], [0.006, 0.003]]}],
        },
        "mesh": mesh,
        "fluid2d": {"init_density_m3": 1e13, "init_te_ev": 2.0, "gas_pressure_pa": 100.0, "dt": T / SPP,
                    "n_steps": periods * SPP, "frame_every": SPP, "avg_steps": 2 * SPP, "convergence": conv},
    })


def _make2d(kind: str, p: Project):
    """kind: "v1" (三角形メッシュ)・"cartesian" (v2 の一様格子、CPU)・"amr" (v2 の合成格子、CPU)。"""
    from es_sim.gfluid import CartesianFluid2dSimulation
    from es_sim.gfluid.amr import AmrFluid2dSimulation

    if kind == "v1":
        return Fluid2dSimulation(p)
    return (AmrFluid2dSimulation if kind == "amr" else CartesianFluid2dSimulation)(p, device="cpu")


# ---- 流体 1D ---------------------------------------------------------------------------------------


def test_fluid1d_judges_each_rf_period_and_reports_it():
    frames = []
    sim = _closed(Fluid1dSimulation(_p1d(10)))
    sim.run_batch(callback=frames.append, store_frames=False)
    r = build_fluid1d_result(sim, 0.0)["convergence"]
    json.dumps(r, allow_nan=False)
    assert r["basis"] == "rf" and r["period_s"] == pytest.approx(T)
    assert r["period"] == list(range(10))
    assert np.allclose(np.asarray(r["t"]) / T, np.arange(1, 11), rtol=1e-9)
    assert r["step"] == [SPP * (k + 1) for k in range(10)]
    assert set(r["d"]) == {"phi", "n_e", "N_e", "N_i"}
    assert r["blocks"] == [41]                      # 41 節点 (64 より少ない) は 1 節点ずつのブロック
    assert r["converged"] and r["converged_period"] == CONVERGED_PERIOD
    assert r["converged_step"] == SPP * (CONVERGED_PERIOD + 1) and not r["stopped"]
    # フレームに今の状態 (判定が進むにつれ変わる)
    assert [f["convergence"]["checks"] for f in frames] == list(range(1, 11))
    assert frames[-1]["convergence"]["converged"] and frames[-1]["convergence"]["status"] == "pass"


def test_fluid1d_stop_takes_the_averaging_window_after_convergence():
    sim = _closed(Fluid1dSimulation(_p1d(40, stop=True)))
    sim.s.avg_steps = 150
    history, _ = sim.run_batch(store_frames=False)
    r = build_fluid1d_result(sim, 0.0)["convergence"]
    assert r["converged"] and r["stopped"]
    assert sim.step_count == r["converged_step"] + 150 < 40 * SPP
    assert len(history["step"]) == sim.step_count
    assert sim.fields["avg_steps"] == 150          # 時間平均は収束のあとの平均区間


def test_job_progress_ends_at_the_convergence_stop(monkeypatch):
    """ジョブで走らせて収束で止めると、進捗の総ステップ数は早めた終わりまで (設定の n_steps は上限だった。CV-c)。"""
    import time

    from fastapi.testclient import TestClient

    import es_sim.server as server
    from es_sim.jobs.runners import Fluid1dRunner

    make = Fluid1dRunner.make
    monkeypatch.setattr(Fluid1dRunner, "make", lambda self, p: _closed(make(self, p)))
    client = TestClient(server.app)
    project = _p1d(40, stop=True).model_dump(mode="json")
    jid = client.post("/v2/jobs", json={"kind": "fluid1d", "project": project}).json()["id"]
    deadline = time.monotonic() + 60.0
    while (s := client.get(f"/v2/jobs/{jid}").json())["state"] not in ("done", "error"):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    end = SPP * (CONVERGED_PERIOD + 1) + 2 * SPP   # 収束のステップ + 平均区間 (avg_steps)
    assert s["state"] == "done"
    assert s["progress"]["step"] == s["progress"]["n_steps"] == end < 40 * SPP
    assert client.get(f"/v2/jobs/{jid}/result").json()["convergence"]["stopped"]
    client.delete(f"/v2/jobs/{jid}")


def test_fluid1d_continue_keeps_the_judgement():
    sim = _closed(Fluid1dSimulation(_p1d(8, stop=True)))
    sim.run_batch(store_frames=False)
    assert not sim.conv.converged
    sim.prepare_continue(6 * SPP)
    sim.run_batch(store_frames=False)
    r = build_fluid1d_result(sim, 0.0)["convergence"]
    assert r["converged_period"] == CONVERGED_PERIOD
    # 続きの実行の中で収束したので止める: 収束から平均区間 (avg_steps = 2 周期) で終わる
    assert r["stopped"] and sim.step_count == r["converged_step"] + 2 * SPP
    assert r["period"] == list(range(sim.step_count // SPP))   # 続きの実行でも周期がつながる
    # 収束したあとの続きでは止めない
    sim.prepare_continue(3 * SPP)
    sim.run_batch(store_frames=False)
    assert sim.step_count == r["converged_step"] + 5 * SPP


def test_fluid1d_judges_the_self_bias():
    sim = _closed(Fluid1dSimulation(_p1d(10, cap={"capacitance": 1e-3, "initial_bias_v": -5.0})))
    sim.run_batch(store_frames=False)
    r = build_fluid1d_result(sim, 0.0)["convergence"]
    assert "V_dc:left" in r["d"]
    assert r["converged"]


def test_disabled_judgement_adds_nothing():
    sim = _closed(Fluid1dSimulation(_p1d(3, enabled=False)))
    sim.run_batch(store_frames=False)
    assert sim.conv is None and build_fluid1d_result(sim, 0.0)["convergence"] is None


# ---- 流体 2D (v1 の三角形メッシュ・v2 の一様格子・AMR) ------------------------------------------------


@pytest.mark.parametrize("kind", ["v1", "cartesian", "amr"])
def test_fluid2d_judges_each_rf_period(kind):
    sim = _closed(_make2d(kind, _p2d(PERIODS_2D, kind)))
    sim.run_batch(store_frames=False)
    r = build_fluid2d_result(sim, 0.0)["convergence"]
    json.dumps(r, allow_nan=False)
    assert r["period"] == list(range(PERIODS_2D))
    assert np.allclose(np.asarray(r["t"]) / T, np.arange(1, PERIODS_2D + 1), rtol=1e-9)
    assert len(r["blocks"]) == 2
    assert r["converged"] and CONVERGED_PERIOD <= r["converged_period"] <= PERIODS_2D - 2


def test_fluid2d_stop_takes_the_averaging_window_after_convergence():
    sim = _closed(_make2d("cartesian", _p2d(40, stop=True)))
    sim.run_batch(store_frames=False)
    r = build_fluid2d_result(sim, 0.0)["convergence"]
    assert r["converged"] and r["stopped"]
    assert sim.step_count == r["converged_step"] + 2 * SPP
    assert sim.fields["avg_steps"] == 2 * SPP


@pytest.mark.skipif(not cuda_available(), reason="CUDA が使えません")
def test_fluid2d_gpu_judges_like_cpu():
    """GPU 版 (場をデバイスで足し込む) は CPU 版と同じ周期で収束と判定し、変化も丸め誤差の範囲で一致する。"""
    from es_sim.gfluid.gpu import GpuCartesianFluid2dSimulation

    p = _p2d(PERIODS_2D)
    cpu = _closed(_make2d("cartesian", p))
    cpu.run_batch(store_frames=False)
    gpu = _closed(GpuCartesianFluid2dSimulation(p))
    gpu.run_batch(store_frames=False)
    rc = build_fluid2d_result(cpu, 0.0)["convergence"]
    rg = build_fluid2d_result(gpu, 0.0)["convergence"]
    assert rc["converged"] and rg["converged_period"] == rc["converged_period"]
    assert rg["status"] == rc["status"]
    for n in ("phi", "N_e"):
        a = np.array([v if v is not None else np.nan for v in rc["d"][n]])
        b = np.array([v if v is not None else np.nan for v in rg["d"][n]])
        assert np.allclose(a, b, rtol=1e-3, atol=1e-12, equal_nan=True)
