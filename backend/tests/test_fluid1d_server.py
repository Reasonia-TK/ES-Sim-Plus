"""流体 (1D) の server / sweep / batch 配線のテスト (prompts/107)。

pic1d の配線テスト (test_pic1d.py の /ws/pic1d プロトコル部分、test_sweep.py の
pic1d スイープテスト、test_batch.py の --module auto テスト) と同じ構造をそのまま
1D 流体 (fluid1d) に写経する。

1. /ws/fluid1d: start → frame → done のスモーク (小規模設定、TestClient)。
2. /ws/fluid1d: continue で step_offset・結果 (history 行数) が整合すること。
3. sweep: fluid1d 2ケース (パス "fluid1d.init_density_m3") が回り、
   GET /sweep/result/{i} に fluid1d キーと上書き値が入ること。
4. batch: --module auto が fluid1d 専用 project で fluid1d を選ぶこと。
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import es_sim.server as server
from es_sim.batch import run_files
from es_sim.schema import Project


def _tiny_fluid1d_project(init_density: float = 1.0e14, n_steps: int = 40) -> dict:
    """1D 流体の縮小ケース。geometry/mesh は fluid1d が一切参照しないダミー
    (Project スキーマ上必須なだけ、test_pic1d.py/test_sweep.py の pic1d ケースと同じ方針)。
    n_cells は下限の16、n_steps=40 なら数秒未満で完了する (RF 無し・小 dt)。
    """
    return {
        "geometry": {"domain": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}},
        "mesh": {"size": 0.1},
        "fluid1d": {
            "gap_m": 0.02,
            "n_cells": 16,
            "init_density_m3": init_density,
            "gas_pressure_pa": 10.0,
            "n_steps": n_steps,
            "frame_every": 10,
            "avg_steps": 10,
            "left": {"v_dc": 50.0},
            "right": {"v_dc": 0.0},
        },
    }


def _recv_until_done(ws) -> dict:
    while True:
        msg = ws.receive_json()
        assert msg["type"] != "error", msg.get("detail")
        if msg["type"] == "done":
            return msg


# ---- 1. /ws/fluid1d: start → frame → done のスモーク -----------------------------


def test_ws_fluid1d_start_frame_done():
    server._last_simfluid1d = None
    client = TestClient(server.app)
    with client.websocket_connect("/ws/fluid1d") as ws:
        project = _tiny_fluid1d_project(n_steps=40)
        ws.send_text(json.dumps({"cmd": "start", "project": project}))

        started = ws.receive_json()
        assert started["type"] == "started"
        assert started["n_steps"] == 40
        assert started["step_offset"] == 0
        assert isinstance(started["dt"], float) and started["dt"] > 0
        assert len(started["x"]) == 17  # n_cells=16 → n_nodes=17
        assert isinstance(started["warnings"], list)

        saw_frame = False
        while True:
            msg = ws.receive_json()
            assert msg["type"] != "error", msg.get("detail")
            if msg["type"] == "frame":
                saw_frame = True
                # 粒子サンプルは無い (流体なので)、代わりに t_e[]/counts を持つ
                assert set(("step", "t", "phi", "n_e", "n_i", "t_e", "counts", "elapsed_s")) <= set(msg)
                assert len(msg["phi"]) == 17
                assert len(msg["t_e"]) == 17
                assert "sample" not in msg
            elif msg["type"] == "done":
                done = msg
                break
        assert saw_frame

        result = done["result"]
        assert len(result["history"]["t"]) == 40
        assert result["elapsed_s"] > 0.0
        assert "total" in result["timing"]
        assert result["profiles"] is not None
        assert result["settings"]["gap_m"] == 0.02

    server._last_simfluid1d = None


# ---- 2. continue: step_offset・結果整合 ------------------------------------------


def test_ws_fluid1d_continue_step_offset_and_history():
    server._last_simfluid1d = None
    client = TestClient(server.app)
    with client.websocket_connect("/ws/fluid1d") as ws:
        project = _tiny_fluid1d_project(n_steps=20)
        ws.send_text(json.dumps({"cmd": "start", "project": project}))
        started = ws.receive_json()
        assert started["type"] == "started" and started["n_steps"] == 20 and started["step_offset"] == 0

        done1 = _recv_until_done(ws)
        assert len(done1["result"]["history"]["t"]) == 20

        ws.send_text(json.dumps({"cmd": "continue", "extra_steps": 10}))
        started2 = ws.receive_json()
        assert started2["type"] == "started"
        assert started2["n_steps"] == 10
        assert started2["step_offset"] == 20  # 通算ステップ (start の20分から継続)

        done2 = _recv_until_done(ws)
        assert len(done2["result"]["history"]["t"]) == 10

    server._last_simfluid1d = None


def test_ws_fluid1d_continue_without_state_errors():
    server._last_simfluid1d = None
    client = TestClient(server.app)
    with client.websocket_connect("/ws/fluid1d") as ws:
        ws.send_text(json.dumps({"cmd": "continue", "extra_steps": 5}))
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "保持" in msg["detail"] or "start" in msg["detail"]


# ---- 3. sweep: fluid1d 2ケース ----------------------------------------------------


def test_ws_sweep_fluid1d_two_values_full_flow_and_result_endpoint():
    client = TestClient(server.app)
    project = _tiny_fluid1d_project(init_density=1.0e14)

    with client.websocket_connect("/ws/sweep") as ws:
        ws.send_json(
            {
                "cmd": "start",
                "project": project,
                "param_path": "fluid1d.init_density_m3",
                "values": [1.0e14, 2.0e14],
                "parallel": 2,
            }
        )
        started = ws.receive_json()
        assert started["type"] == "started"
        assert started["n_cases"] == 2
        assert started["module"] == "fluid1d"  # module 未指定 → param_path の接頭辞から自動判定

        case_done: dict[int, dict] = {}
        done_msg = None
        while done_msg is None:
            msg = ws.receive_json()
            assert msg["type"] != "error", msg.get("detail")
            if msg["type"] == "case_done":
                case_done[msg["case"]] = msg
            elif msg["type"] == "done":
                done_msg = msg

        assert set(case_done) == {0, 1}
        assert all(ev["ok"] for ev in case_done.values())

    resp0 = client.get("/sweep/result/0")
    assert resp0.status_code == 200
    obj0 = resp0.json()
    assert obj0["fluid1d"]["init_density_m3"] == 1.0e14
    assert obj0["results"]["version"] == 1
    assert "pic" not in obj0["results"] and "pic1d" not in obj0["results"]
    fluid1d_result0 = obj0["results"]["fluid1d"]
    assert fluid1d_result0 is not None
    assert set(fluid1d_result0) == {
        "history", "profiles", "sheath", "cycle", "wall_iedf", "walls", "gen_total", "elapsed_s", "timing", "settings",
        "circuit",
    }
    assert fluid1d_result0["settings"]["init_density_m3"] == 1.0e14

    resp1 = client.get("/sweep/result/1")
    assert resp1.status_code == 200
    obj1 = resp1.json()
    assert obj1["fluid1d"]["init_density_m3"] == 2.0e14
    assert obj1["results"]["fluid1d"]["settings"]["init_density_m3"] == 2.0e14


def test_resolve_sweep_module_auto_detects_fluid1d_prefix():
    from es_sim.sweep import resolve_sweep_module

    assert resolve_sweep_module("fluid1d.init_density_m3", None) == "fluid1d"
    assert resolve_sweep_module("fluid1d.left.v_dc", None) == "fluid1d"
    # 明示指定は接頭辞より優先される
    assert resolve_sweep_module("fluid1d.n_steps", "pic1d") == "pic1d"


# ---- 4. batch: --module auto が fluid1d を選ぶこと ---------------------------------


def _write_case(tmp_path: Path, name: str, project: dict) -> Path:
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(project), encoding="utf-8")
    return p


def test_batch_run_module_auto_selects_fluid1d_only_project(tmp_path: Path):
    case = _write_case(tmp_path, "case_fluid1d", _tiny_fluid1d_project(n_steps=20))
    rc = run_files([str(case)], parallel=1, out_dir=str(tmp_path))
    assert rc == 0

    obj = json.loads((tmp_path / "case_fluid1d_results.json").read_text(encoding="utf-8"))
    assert "pic" not in obj["results"] and "pic1d" not in obj["results"]
    result = obj["results"]["fluid1d"]
    assert result is not None
    assert set(result) == {
        "history", "profiles", "sheath", "cycle", "wall_iedf", "walls", "gen_total", "elapsed_s", "timing", "settings",
        "circuit",
    }
    assert result["settings"]["n_steps"] == 20
    assert result["elapsed_s"] > 0

    # results を除いた部分は pydantic Project として再検証できる (pic1d 側と同じ設計)
    project_only = {k: v for k, v in obj.items() if k != "results"}
    project = Project.model_validate(project_only)
    assert project.fluid1d is not None
    assert project.pic is None and project.pic1d is None
