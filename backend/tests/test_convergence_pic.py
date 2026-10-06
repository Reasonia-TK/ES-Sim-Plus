"""PIC 1D・v1 PIC・v2 GPU PIC の収束の判定 (prompts/137)。

判定そのもの (雑音・窓・残りの変化) は test_convergence.py で確かめてあるので、ここではエンジンへの配線を見る:
判定の周期が RF 周期にそろうこと、量 (φ・n_e・総数・自己バイアス) とフレーム、止める設定 (判定の部品を差し替えて
決まった周期で収束としたとき、そこから平均区間を取って終わり、時間平均・最後の 1 周期の粒子の写しがその区間に
なる)。v2 は CUDA が無ければ skip。
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from es_sim.batch import _build_results_bundle
from es_sim.convergence import TOL_PIC
from es_sim.device import cuda_available
from es_sim.pic import PicSimulation
from es_sim.pic1d import Pic1dSimulation, build_pic1d_result
from es_sim.schema import Project

F0 = 13.56e6
T = 1.0 / F0
SPP = 200


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def _force_convergence(sim, period: int) -> None:
    """判定の結果に関係なく、period を閉じたところで収束とする (エンジンの止める配線だけを確かめる)。"""
    mon = sim.conv
    judge = mon._judge

    def forced(t, step, k_closed):
        judge(t, step, k_closed)
        if k_closed == period and not mon.converged:
            mon.converged = True
            mon.converged_at = {"t": t, "step": step, "period": k_closed}
            mon.stop_now = mon.stop and not mon._converged_at_run_start

    mon._judge = forced


def _p1d(periods: int, **conv) -> Project:
    return Project.model_validate({
        "geometry": {"domain": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}}, "mesh": {"size": 0.1},
        "pic1d": {
            "gap_m": 0.02, "n_cells": 64, "n_macro": 4000, "init_density_m3": 1e15, "init_te_ev": 2.0,
            "left": {"voltage_rf": {"amplitude": 100.0, "freq_hz": F0},
                     "blocking_capacitor": {"capacitance": 1e-2, "initial_bias_v": 0.0}},
            "dt": T / SPP, "n_steps": periods * SPP, "frame_every": SPP, "avg_steps": 2 * SPP,
            "convergence": conv,
        },
    })


def _p2d(periods: int, mode: str = "structured", **conv) -> Project:
    return Project.model_validate({
        "geometry": {
            "domain": {"polygon": [[0, 0], [0.02, 0], [0.02, 0.01], [0, 0.01]]},
            "boundaries": [
                {"edges": [3], "type": "dirichlet", "voltage": 0.0, "voltage_rf": {"amplitude": 100.0, "freq_hz": F0}},
                {"edges": [1], "type": "dirichlet", "voltage": 0.0},
                {"edges": [0, 2], "type": "symmetry"},
            ],
        },
        "mesh": {"size": 0.5e-3, "mode": mode},
        "pic": {"initial_plasma": {"density": 1e15, "te_ev": 2.0}, "n_macro": 4000, "dt": T / SPP,
                "n_steps": periods * SPP, "frame_every": SPP, "avg_steps": 2 * SPP, "phase_bins": 8,
                "convergence": conv},
    })


def _make2d(p: Project):
    if p.mesh.mode == "cartesian":
        from es_sim.gpic import make_pic_simulation

        return make_pic_simulation(p)
    return PicSimulation(p)


def _check_periods(r: dict, n: int) -> None:
    json.dumps(r, allow_nan=False)
    assert r["basis"] == "rf" and r["tol"] == TOL_PIC
    assert r["period"] == list(range(n))
    assert np.allclose(np.asarray(r["t"]) / T, np.arange(1, n + 1), rtol=1e-9)


# ---- PIC 1D ----------------------------------------------------------------------------------------


def test_pic1d_judges_each_rf_period():
    frames = []
    sim = Pic1dSimulation(_p1d(8))
    sim.run_batch(callback=frames.append, store_frames=False)
    r = build_pic1d_result(sim, 0.0)["convergence"]
    _check_periods(r, 8)
    assert set(r["d"]) == {"phi", "n_e", "N_e", "N_i", "V_dc:left"}
    assert sim.conv.stride == SPP // 64            # 1 周期に 64 回ほど足し込む
    assert r["window"][-1] is not None              # 雑音から窓を決めている
    assert [f["convergence"]["checks"] for f in frames] == list(range(1, 9))


def test_pic1d_stop_takes_the_averaging_window():
    sim = Pic1dSimulation(_p1d(40, stop=True))
    _force_convergence(sim, 3)
    sim.run_batch(store_frames=False)
    r = build_pic1d_result(sim, 0.0)["convergence"]
    assert r["converged"] and r["stopped"] and r["converged_period"] == 3
    assert sim.step_count == r["converged_step"] + 2 * SPP == 6 * SPP
    assert len(sim.history["step"]) == sim.step_count
    assert sim.fields["avg_steps"] == 2 * SPP


# ---- 2D PIC (v1 の三角形メッシュ・v2 の GPU) ---------------------------------------------------------


_MODES = ["structured", pytest.param("cartesian", marks=pytest.mark.skipif(not cuda_available(),
                                                                            reason="CUDA が使えません"))]


@pytest.mark.parametrize("mode", _MODES)
def test_pic2d_judges_each_rf_period(mode):
    frames = []
    sim = _make2d(_p2d(6, mode))
    sim.run_batch(callback=frames.append, store_frames=False)
    r = _build_results_bundle(sim, 0, 0.0)["pic"]["convergence"]
    _check_periods(r, 6)
    assert set(r["d"]) == {"phi", "n_e", "N_e", "N_i"}
    assert len(r["blocks"]) == 2
    assert frames[-1]["convergence"]["checks"] == 6


@pytest.mark.parametrize("mode", _MODES)
def test_pic2d_stop_takes_the_averaging_window(mode):
    sim = _make2d(_p2d(40, mode, stop=True))
    _force_convergence(sim, 3)
    sim.run_batch(store_frames=False)
    r = _build_results_bundle(sim, 0, 0.0)["pic"]["convergence"]
    assert r["converged"] and r["stopped"] and r["converged_period"] == 3
    assert sim.step_count == r["converged_step"] + 2 * SPP == 6 * SPP
    assert sim.fields["avg_steps"] == 2 * SPP
    # 最後の 1 周期の粒子の写しも、新しい終わりの前の 1 周期で取れている
    assert all(len(snap) > 0 for snap in sim.cycle["particles"]["electron"])


def test_disabled_judgement_adds_nothing():
    sim = Pic1dSimulation(_p1d(2, enabled=False))
    sim.run_batch(store_frames=False)
    assert sim.conv is None and build_pic1d_result(sim, 0.0)["convergence"] is None
    sim2 = PicSimulation(_p2d(1, enabled=False))
    sim2.run_batch(store_frames=False)
    assert _build_results_bundle(sim2, 0, 0.0)["pic"]["convergence"] is None
