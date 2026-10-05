"""流体 (2D) の server / sweep / batch 配線のテスト (prompts/112)。

fluid1d の配線テスト (test_fluid1d_server.py) と同じ構造をそのまま 2D 流体
(fluid2d) に写経する。fluid1d と違い geometry/mesh からメッシュを生成する
必要があるため、CI 予算内に収まるよう極小の構造格子 (structured, 数ノード) を
使う (test_fluid2d.py の test_matches_fluid1d_cross_section_average と同じ
「細長い矩形 + 上下 symmetry」のパターンを、さらに縮小したもの)。

1. /ws/fluid2d: start → frame → done のスモーク (小規模設定、TestClient)。
   frame は mesh を含まず (prompts/112 の指示どおり)、フィールド節点配列
   (phi/n_e/n_i/t_e) が全節点長で入っていることを確認する。
2. /ws/fluid2d: continue で step_offset・結果 (history 行数) が整合すること。
3. sweep: fluid2d 2ケース (パス "fluid2d.init_density_m3") が回り、
   GET /sweep/result/{i} に fluid2d キーと上書き値が入ること。
4. batch: --module auto が fluid2d 専用 project で fluid2d を選ぶこと。
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import es_sim.server as server
from es_sim.batch import run_files
from es_sim.schema import Project

# 細長い矩形 (gap × h、上下 symmetry・左右 dirichlet)。structured モードで
# size=h を指定すると x 方向に gap/h セル、y 方向に1セルの極小メッシュになる
# (test_fluid2d.py のテスト1と同じ幾何パターン、CI 用にさらに小さくしたもの)。
_GAP = 0.01
_H = 0.005


def _tiny_fluid2d_project(init_density: float = 1.0e14, n_steps: int = 20) -> dict:
    return {
        "geometry": {
            "domain": {"polygon": [[0, 0], [_GAP, 0], [_GAP, _H], [0, _H]]},
            "boundaries": [
                {"edges": [3], "type": "dirichlet", "voltage": 50.0},  # 左辺
                {"edges": [1], "type": "dirichlet", "voltage": 0.0},   # 右辺
                {"edges": [0], "type": "symmetry"},                    # 下辺
                {"edges": [2], "type": "symmetry"},                    # 上辺
            ],
        },
        "mesh": {"size": _H, "mode": "structured"},
        "fluid2d": {
            "init_density_m3": init_density,
            "gas_pressure_pa": 10.0,
            "n_steps": n_steps,
            "frame_every": 10,
            "avg_steps": 10,
        },
    }


def _recv_until_done(ws) -> dict:
    while True:
        msg = ws.receive_json()
        assert msg["type"] != "error", msg.get("detail")
        if msg["type"] == "done":
            return msg


# ---- 1. /ws/fluid2d: start → frame → done のスモーク -----------------------------


def test_ws_fluid2d_start_frame_done():
    server._last_simfluid2d = None
    client = TestClient(server.app)
    with client.websocket_connect("/ws/fluid2d") as ws:
        project = _tiny_fluid2d_project(n_steps=20)
        ws.send_text(json.dumps({"cmd": "start", "project": project}))

        started = ws.receive_json()
        assert started["type"] == "started"
        assert started["n_steps"] == 20
        assert started["step_offset"] == 0
        assert isinstance(started["dt"], float) and started["dt"] > 0
        assert isinstance(started["warnings"], list)
        # 2D PIC と異なり mesh は同梱しない (フロントは既存の /mesh 結果を使う、prompts/112)
        assert "mesh" not in started
        assert "x" not in started

        n_nodes = None
        saw_frame = False
        while True:
            msg = ws.receive_json()
            assert msg["type"] != "error", msg.get("detail")
            if msg["type"] == "frame":
                saw_frame = True
                assert set(("step", "t", "phi", "n_e", "n_i", "t_e", "counts", "elapsed_s")) <= set(msg)
                assert "mesh" not in msg
                n_nodes = len(msg["phi"])
                assert len(msg["n_e"]) == n_nodes
                assert len(msg["n_i"]) == n_nodes
                assert len(msg["t_e"]) == n_nodes
            elif msg["type"] == "done":
                done = msg
                break
        assert saw_frame
        assert n_nodes is not None and n_nodes > 0

        result = done["result"]
        assert len(result["history"]["t"]) == 20
        assert result["elapsed_s"] > 0.0
        assert "total" in result["timing"]
        assert result["fields"] is not None
        assert len(result["fields"]["phi"]) == n_nodes
        assert result["settings"]["init_density_m3"] == 1.0e14

    server._last_simfluid2d = None


# ---- 2. continue: step_offset・結果整合 ------------------------------------------


def test_ws_fluid2d_continue_step_offset_and_history():
    server._last_simfluid2d = None
    client = TestClient(server.app)
    with client.websocket_connect("/ws/fluid2d") as ws:
        project = _tiny_fluid2d_project(n_steps=20)
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

    server._last_simfluid2d = None


def test_ws_fluid2d_continue_without_state_errors():
    server._last_simfluid2d = None
    client = TestClient(server.app)
    with client.websocket_connect("/ws/fluid2d") as ws:
        ws.send_text(json.dumps({"cmd": "continue", "extra_steps": 5}))
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "保持" in msg["detail"] or "start" in msg["detail"]


# ---- 3. sweep: fluid2d 2ケース ----------------------------------------------------


def test_ws_sweep_fluid2d_two_values_full_flow_and_result_endpoint():
    client = TestClient(server.app)
    project = _tiny_fluid2d_project(init_density=1.0e14)

    with client.websocket_connect("/ws/sweep") as ws:
        ws.send_json(
            {
                "cmd": "start",
                "project": project,
                "param_path": "fluid2d.init_density_m3",
                "values": [1.0e14, 2.0e14],
                "parallel": 2,
            }
        )
        started = ws.receive_json()
        assert started["type"] == "started"
        assert started["n_cases"] == 2
        assert started["module"] == "fluid2d"  # module 未指定 → param_path の接頭辞から自動判定

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
    assert obj0["fluid2d"]["init_density_m3"] == 1.0e14
    assert obj0["results"]["version"] == 1
    assert "pic" not in obj0["results"] and "pic1d" not in obj0["results"] and "fluid1d" not in obj0["results"]
    fluid2d_result0 = obj0["results"]["fluid2d"]
    assert fluid2d_result0 is not None
    assert set(fluid2d_result0) == {
        "history", "fields", "cycle", "walls", "gen_total", "elapsed_s", "timing", "settings", "circuit", "convergence",
    }
    assert fluid2d_result0["settings"]["init_density_m3"] == 1.0e14

    resp1 = client.get("/sweep/result/1")
    assert resp1.status_code == 200
    obj1 = resp1.json()
    assert obj1["fluid2d"]["init_density_m3"] == 2.0e14
    assert obj1["results"]["fluid2d"]["settings"]["init_density_m3"] == 2.0e14


def test_resolve_sweep_module_auto_detects_fluid2d_prefix():
    from es_sim.sweep import resolve_sweep_module

    assert resolve_sweep_module("fluid2d.init_density_m3", None) == "fluid2d"
    assert resolve_sweep_module("fluid2d.gas_pressure_pa", None) == "fluid2d"
    # 明示指定は接頭辞より優先される
    assert resolve_sweep_module("fluid2d.n_steps", "pic1d") == "pic1d"


# ---- 4. batch: --module auto が fluid2d を選ぶこと ---------------------------------


def _write_case(tmp_path: Path, name: str, project: dict) -> Path:
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(project), encoding="utf-8")
    return p


def test_batch_run_module_auto_selects_fluid2d_only_project(tmp_path: Path):
    case = _write_case(tmp_path, "case_fluid2d", _tiny_fluid2d_project(n_steps=20))
    rc = run_files([str(case)], parallel=1, out_dir=str(tmp_path))
    assert rc == 0

    obj = json.loads((tmp_path / "case_fluid2d_results.json").read_text(encoding="utf-8"))
    assert "pic" not in obj["results"] and "pic1d" not in obj["results"] and "fluid1d" not in obj["results"]
    result = obj["results"]["fluid2d"]
    assert result is not None
    assert set(result) == {
        "history", "fields", "cycle", "walls", "gen_total", "elapsed_s", "timing", "settings", "circuit", "convergence",
    }
    assert result["settings"]["n_steps"] == 20
    assert result["elapsed_s"] > 0

    # results を除いた部分は pydantic Project として再検証できる (fluid1d 側と同じ設計)
    project_only = {k: v for k, v in obj.items() if k != "results"}
    project = Project.model_validate(project_only)
    assert project.fluid2d is not None
    assert project.pic is None and project.pic1d is None and project.fluid1d is None
