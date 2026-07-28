"""PIC の続き実行 (prepare_continue / WS continue) のテスト (prompts/32)。

1. ビット一致: MCC・RF・誘電体付き小ケースで「200 ステップ連続」と
   「100 ステップ → continue で 100 ステップ」の粒子状態・q_surf が完全一致
2. 時刻の連続性: continue 後の history["t"] が前回最終時刻から連続
3. 保持なしエラー: 状態なしで continue するとエラー (WS TestClient)。
   併せて start → done → continue → done の WS フローを確認
4. fields 再計算: continue 区間の avg_steps 指定で fields が返る
   (連続実行と同じ平均区間なら fields もビット一致)
"""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import es_sim.server as server
from es_sim.pic import PicSimulation
from es_sim.schema import Project

DT = 2e-11  # 200 ステップで電子が失われ尽くさない程度の刻み


def _mcc_rf_project(n_steps: int) -> Project:
    """RF + MCC (合成電離) + 中央誘電体ブロック付きの小ケース。"""
    return Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01]]},
                "regions": [
                    {
                        "id": "block",
                        "type": "dielectric",
                        "polygon": [[0.004, 0.004], [0.006, 0.004], [0.006, 0.006], [0.004, 0.006]],
                        "eps_r": 2.0,
                    }
                ],
                "boundaries": [
                    {
                        "edges": [3],
                        "voltage": 0.0,
                        "voltage_rf": {"amplitude": 50.0, "freq_hz": 12.5e6, "phase_deg": 0.0},
                    },
                    {"edges": [1], "voltage": 0.0},
                ],
            },
            "mesh": {"size": 8e-4},
            "pic": {
                "initial_plasma": {
                    "density": 1.0e14,
                    "te_ev": 3.0,  # 電離閾値 5 eV を超える裾を持たせる
                    "ti_ev": 0.03,
                    "ion_mass_amu": 40.0,
                    "seed": 3,
                },
                "n_macro": 800,
                "dt": DT,
                "n_steps": n_steps,
                "frame_every": n_steps,
                "mcc": {
                    "gas": {"name": "Ar", "pressure_pa": 100.0, "temperature_k": 300.0},
                    "electron_processes": [
                        {
                            "kind": "ionization",
                            "label": "synthetic ionization",
                            "threshold_ev": 5.0,
                            "energy_ev": [5.0, 5.0 + 1e-6, 1000.0],
                            "sigma_m2": [0.0, 2.0e-20, 2.0e-20],
                        }
                    ],
                    "ion_processes": [],
                    "seed": 9,
                },
            },
        }
    )


def _empty_project(n_steps: int) -> Project:
    return Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01]]},
                "boundaries": [{"edges": [0, 1, 2, 3], "voltage": 0.0}],
            },
            "mesh": {"size": 1e-3},
            "pic": {"dt": 1e-9, "n_steps": n_steps, "n_macro": 10, "frame_every": n_steps},
        }
    )


# ---- 1 & 4. ビット一致 + fields 再計算 --------------------------------------------


def test_continue_is_bit_identical_to_single_run():
    """乱数 Generator と粒子状態を保持するため、200 ステップ連続実行と
    100 ステップ → continue(100) の粒子状態はビット単位で一致する。"""
    # A: 200 ステップ連続 (avg_steps 既定 = 最後の 25% = 50 → 平均区間 151..200)
    sim_a = PicSimulation(_mcc_rf_project(200))
    hist_a, _ = sim_a.run_batch()

    # B: 100 ステップ → continue で 100 ステップ (avg_steps=50 → 同じ平均区間 151..200)
    sim_b = PicSimulation(_mcc_rf_project(100))
    sim_b.run_batch()
    sim_b.prepare_continue(100, avg_steps=50)
    hist_b, _ = sim_b.run_batch()

    # 粒子状態 (位置・速度・重み・所属要素) がビット単位で一致
    for name in ("electron", "ion"):
        sa, sb = sim_a.species[name], sim_b.species[name]
        assert len(sa.x) == len(sb.x) and len(sa.x) > 0
        assert np.array_equal(sa.x, sb.x)
        assert np.array_equal(sa.v, sb.v)
        assert np.array_equal(sa.w, sb.w)
        assert np.array_equal(sa.elem, sb.elem)
        assert sa.wall_absorbed == sb.wall_absorbed

    # 表面電荷 (誘電体) もビット一致し、実際に蓄積されている
    assert np.array_equal(sim_a.q_surf, sim_b.q_surf)
    assert float(np.abs(sim_a.q_surf).sum()) > 0.0

    # 累計カウンタ・時刻・ステップ数も一致
    assert sim_a.coll_e == sim_b.coll_e
    assert sim_a.ion_events == sim_b.ion_events
    assert sim_a.t == sim_b.t
    assert sim_a.step_count == sim_b.step_count == 200
    assert hist_a["n_e"][-1] == hist_b["n_e"][-1]

    # 4. fields 再計算: continue 区間の avg_steps 指定で fields が返り、
    #    平均区間 (151..200) が同一なのでビット一致する
    assert sim_b.fields is not None and sim_b.fields["avg_steps"] == 50
    assert sim_a.fields is not None and sim_a.fields["avg_steps"] == 50
    assert np.array_equal(sim_a.fields["phi"], sim_b.fields["phi"])
    assert np.array_equal(sim_a.fields["n_e"], sim_b.fields["n_e"])
    assert np.array_equal(sim_a.fields["n_i"], sim_b.fields["n_i"])


# ---- 2. 時刻の連続性 ---------------------------------------------------------------


def test_continue_history_time_is_contiguous():
    dt = 1e-9
    sim = PicSimulation(_empty_project(20))
    hist1, _ = sim.run_batch()
    t1 = list(hist1["t"])
    assert len(t1) == 20

    sim.prepare_continue(10)
    hist2, _ = sim.run_batch()
    t2 = np.asarray(hist2["t"])

    # history は追加区間分のみで、t は通算時刻のまま前回最終時刻から連続・単調増加
    assert len(t2) == 10
    assert t2[0] == pytest.approx(t1[-1] + dt, rel=1e-12)
    assert np.all(np.diff(t2) > 0.0)
    assert sim.step_count == 30


# ---- 3. WS プロトコル (保持なしエラー / start → continue フロー) --------------------


def _recv_until_done(ws) -> dict:
    while True:
        msg = ws.receive_json()
        assert msg["type"] != "error", msg.get("detail")
        if msg["type"] == "done":
            return msg


def test_ws_continue_without_state_errors_and_full_flow():
    server._last_sim = None  # 保持スロットを空にしてから検証する
    client = TestClient(server.app)

    with client.websocket_connect("/ws/pic") as ws:
        # 保持状態なしの continue はエラー
        ws.send_text(json.dumps({"cmd": "continue", "n_steps": 10}))
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "保持" in msg["detail"] or "start" in msg["detail"]

        # start → done (10 ステップ)
        project = _empty_project(10)
        ws.send_text(json.dumps({"cmd": "start", "project": project.model_dump(mode="json")}))
        started = ws.receive_json()
        assert started["type"] == "started" and started["n_steps"] == 10
        done1 = _recv_until_done(ws)
        assert len(done1["history"]["t"]) == 10
        # 実行時間のリアルタイム表示 (prompts/86): done に run_batch の壁時計秒が載る
        assert done1["elapsed_s"] > 0

        # continue → started (n_steps=追加分) → done (history は追加区間分・通算時刻)
        ws.send_text(json.dumps({"cmd": "continue", "n_steps": 5, "frame_every": 5}))
        started2 = ws.receive_json()
        assert started2["type"] == "started" and started2["n_steps"] == 5
        done2 = _recv_until_done(ws)
        t2 = done2["history"]["t"]
        assert len(t2) == 5
        assert t2[0] == pytest.approx(done1["history"]["t"][-1] + 1e-9, rel=1e-9)
        assert done2["elapsed_s"] > 0

    server._last_sim = None  # 後続テストへ状態を持ち越さない
