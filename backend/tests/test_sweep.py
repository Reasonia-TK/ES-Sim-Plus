"""GUI パラメータスイープのテスト (prompts/79、1D 対応は prompts/96)。

set_by_path (パス指定での数値上書き) の単体テストと、/ws/sweep の WS プロトコル
(started → progress → case_done×N → done) + GET /sweep/result/{i} の統合テストを行う。
ケース実行は es_sim.batch._worker (prompts/78) をそのまま使うため、test_batch.py と
同様に極小 PIC ケースで実行する (multiprocessing spawn のため数秒かかる)。
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from es_sim import server as srv
from es_sim.sweep import build_sweep_cases, resolve_sweep_module, set_by_path

L = 0.01
H = 0.02


def _tiny_pic_project(voltage: float = 50.0, n_steps: int = 8) -> dict:
    """test_batch.py の縮小 CCP ケースと同型 (数秒で完了する)。"""
    return {
        "geometry": {
            "domain": {"polygon": [[0, 0], [L, 0], [L, H], [0, H]]},
            "boundaries": [
                {"edges": [3], "type": "dirichlet", "voltage": voltage},
                {"edges": [1], "type": "dirichlet", "voltage": 0.0},
            ],
        },
        "mesh": {"size": 2.0e-3},
        "pic": {
            "initial_plasma": {
                "density": 1.0e14,
                "te_ev": 2.0,
                "ti_ev": 0.03,
                "ion_mass_amu": 40.0,
                "immobile_ions": False,
                "seed": 1,
            },
            "n_macro": 300,
            "dt": 5e-10,
            "n_steps": n_steps,
            "frame_every": 4,
        },
    }


def _tiny_pic1d_project(init_density: float = 1.0e14, n_steps: int = 100) -> dict:
    """1D PIC/MCC の縮小ケース (prompts/96)。n_cells 16 / n_macro 500 / MCC無しで数秒以内に完了する。
    geometry/mesh は pic1d が一切参照しないダミー (Project スキーマ上必須なだけ、test_pic1d.py と同じ方針)。
    """
    return {
        "geometry": {"domain": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}},
        "mesh": {"size": 0.1},
        "pic1d": {
            "gap_m": 0.02,
            "n_cells": 16,
            "init_density_m3": init_density,
            "n_macro": 500,
            "dt": 1.0e-10,
            "n_steps": n_steps,
            "frame_every": 20,
            "left": {"v_dc": 50.0},
            "right": {"v_dc": 0.0},
        },
    }


# ---- 1. set_by_path 単体テスト (正常系) -----------------------------------------


def test_set_by_path_nested_dict_field():
    obj = {"pic": {"n_macro": 100}}
    set_by_path(obj, "pic.n_macro", 500.0)
    assert obj["pic"]["n_macro"] == 500.0


def test_set_by_path_array_index():
    obj = {"geometry": {"boundaries": [{"voltage": 0.0}, {"voltage": 10.0}]}}
    set_by_path(obj, "geometry.boundaries.1.voltage", 100.0)
    assert obj["geometry"]["boundaries"][1]["voltage"] == 100.0
    assert obj["geometry"]["boundaries"][0]["voltage"] == 0.0  # 他は変わらない


def test_set_by_path_rf_component_amplitude():
    obj = {"geometry": {"regions": [{"voltage_rf": [{"amp": 1.0}, {"amp": 2.0}]}]}}
    set_by_path(obj, "geometry.regions.0.voltage_rf.0.amp", 42.0)
    assert obj["geometry"]["regions"][0]["voltage_rf"][0]["amp"] == 42.0
    assert obj["geometry"]["regions"][0]["voltage_rf"][1]["amp"] == 2.0


def test_set_by_path_bfield_component():
    obj = {"b_field": {"bx": 0.0, "by": 0.0, "bz": 0.0}}
    set_by_path(obj, "b_field.bz", 0.02)
    assert obj["b_field"] == {"bx": 0.0, "by": 0.0, "bz": 0.02}


def test_set_by_path_pic_scalar_field():
    obj = {"pic": {"n_macro": 20000, "dt": None}}
    set_by_path(obj, "pic.n_macro", 5000.0)
    assert obj["pic"]["n_macro"] == 5000.0


# ---- set_by_path 単体テスト (異常系) --------------------------------------------


def test_set_by_path_missing_key_raises():
    obj = {"pic": {"n_macro": 100}}
    with pytest.raises(ValueError):
        set_by_path(obj, "pic.no_such_field", 1.0)


def test_set_by_path_missing_intermediate_raises():
    obj = {"pic": {}}
    with pytest.raises(ValueError):
        set_by_path(obj, "pic.mcc.gas.pressure_pa", 1.0)


def test_set_by_path_non_numeric_terminal_raises():
    obj = {"geometry": {"regions": [{"id": "r1"}]}}
    with pytest.raises(ValueError):
        set_by_path(obj, "geometry.regions.0.id", 1.0)


def test_set_by_path_bool_terminal_raises():
    # bool は int のサブクラスだが、True/False を数値スイープの対象にすべきではない
    obj = {"pic": {"initial_plasma": {"immobile_ions": False}}}
    with pytest.raises(ValueError):
        set_by_path(obj, "pic.initial_plasma.immobile_ions", 1.0)


def test_set_by_path_index_out_of_range_raises():
    obj = {"geometry": {"boundaries": [{"voltage": 0.0}]}}
    with pytest.raises(ValueError):
        set_by_path(obj, "geometry.boundaries.5.voltage", 1.0)


def test_set_by_path_empty_path_raises():
    with pytest.raises(ValueError):
        set_by_path({}, "", 1.0)


def test_build_sweep_cases_applies_values_without_mutating_base():
    base = _tiny_pic_project(voltage=50.0)
    cases = build_sweep_cases(base, "geometry.boundaries.0.voltage", [10.0, 20.0])
    assert len(cases) == 2
    assert cases[0]["geometry"]["boundaries"][0]["voltage"] == 10.0
    assert cases[1]["geometry"]["boundaries"][0]["voltage"] == 20.0
    assert base["geometry"]["boundaries"][0]["voltage"] == 50.0  # deepcopy なので base は不変


# ---- 2. WS プロトコル (started → progress → case_done×N → done) + GET /sweep/result --------


def test_ws_sweep_two_values_full_flow_and_result_endpoint():
    client = TestClient(srv.app)
    project = _tiny_pic_project(voltage=50.0)

    with client.websocket_connect("/ws/sweep") as ws:
        ws.send_json(
            {
                "cmd": "start",
                "project": project,
                "param_path": "geometry.boundaries.0.voltage",
                "values": [50.0, 100.0],
                "parallel": 2,
            }
        )
        started = ws.receive_json()
        assert started["type"] == "started"
        assert started["n_cases"] == 2
        assert started["param_path"] == "geometry.boundaries.0.voltage"
        assert started["values"] == [50.0, 100.0]

        case_done: dict[int, dict] = {}
        done_msg = None
        while done_msg is None:
            msg = ws.receive_json()
            assert msg["type"] != "error", msg.get("detail")
            if msg["type"] == "progress":
                assert msg["case"] in (0, 1)
                assert 0 < msg["step"] <= msg["n_steps"]
            elif msg["type"] == "case_done":
                case_done[msg["case"]] = msg
            elif msg["type"] == "done":
                done_msg = msg
            else:
                raise AssertionError(f"想定外のメッセージ: {msg['type']}")

        assert set(case_done) == {0, 1}
        assert all(ev["ok"] for ev in case_done.values())
        assert len(done_msg["summary"]) == 2
        for s in done_msg["summary"]:
            assert s["ok"]
        summary_by_case = {s["case"]: s for s in done_msg["summary"]}
        assert summary_by_case[0]["value"] == 50.0
        assert summary_by_case[1]["value"] == 100.0

    # GET /sweep/result/{i}: 各ケースの project にスイープ値が反映されていること
    resp0 = client.get("/sweep/result/0")
    assert resp0.status_code == 200
    obj0 = resp0.json()
    assert obj0["geometry"]["boundaries"][0]["voltage"] == 50.0
    assert obj0["results"]["version"] == 1
    assert obj0["results"]["pic"] is not None

    resp1 = client.get("/sweep/result/1")
    assert resp1.status_code == 200
    obj1 = resp1.json()
    assert obj1["geometry"]["boundaries"][0]["voltage"] == 100.0

    # 存在しないケース番号は404
    resp_missing = client.get("/sweep/result/2")
    assert resp_missing.status_code == 404


def test_ws_sweep_rejects_use_dsmc_gas_before_starting_any_case():
    client = TestClient(srv.app)
    project = _tiny_pic_project()
    project["pic"]["mcc"] = {"gas": {"pressure_pa": 1.0}, "use_dsmc_gas": True}

    with client.websocket_connect("/ws/sweep") as ws:
        ws.send_json(
            {
                "cmd": "start",
                "project": project,
                "param_path": "geometry.boundaries.0.voltage",
                "values": [10.0, 20.0],
                "parallel": 1,
            }
        )
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "use_dsmc_gas" in msg["detail"]


def test_ws_sweep_invalid_param_path_reports_error():
    client = TestClient(srv.app)
    project = _tiny_pic_project()

    with client.websocket_connect("/ws/sweep") as ws:
        ws.send_json(
            {
                "cmd": "start",
                "project": project,
                "param_path": "pic.no_such_field",
                "values": [1.0, 2.0],
                "parallel": 1,
            }
        )
        msg = ws.receive_json()
        assert msg["type"] == "error"


# ---- 3. stop: 開始直後に stop してエラーなく閉じること ----------------------------


def test_ws_sweep_stop_immediately_closes_without_error():
    client = TestClient(srv.app)
    # 停止がケース完了より先に効くよう、少し長めのステップ数にする
    project = _tiny_pic_project(n_steps=500)

    with client.websocket_connect("/ws/sweep") as ws:
        ws.send_json(
            {
                "cmd": "start",
                "project": project,
                "param_path": "geometry.boundaries.0.voltage",
                "values": [10.0, 20.0, 30.0],
                "parallel": 1,
            }
        )
        started = ws.receive_json()
        assert started["type"] == "started"
        ws.send_json({"cmd": "stop"})

        saw_done = False
        while True:
            msg = ws.receive_json()
            assert msg["type"] != "error", msg.get("detail")
            if msg["type"] == "done":
                saw_done = True
                break
        assert saw_done


# ---- 4. set_by_path: pic1d パス (prompts/96) --------------------------------------
# set_by_path は汎用実装なので pic1d.* パスも既存機能でそのまま通る (backend 側の
# 追加実装は不要)。既存の pic.* 系テストと対になる形で 1D 側も確認しておく。


def test_set_by_path_pic1d_scalar_field():
    obj = {"pic1d": {"gap_m": 0.01}}
    set_by_path(obj, "pic1d.gap_m", 0.03)
    assert obj["pic1d"]["gap_m"] == 0.03


def test_set_by_path_pic1d_nested_electrode_field():
    obj = {"pic1d": {"left": {"v_dc": 0.0}, "right": {"v_dc": 0.0}}}
    set_by_path(obj, "pic1d.left.v_dc", 100.0)
    assert obj["pic1d"]["left"]["v_dc"] == 100.0
    assert obj["pic1d"]["right"]["v_dc"] == 0.0  # 他は変わらない


def test_set_by_path_pic1d_missing_field_raises():
    obj = {"pic1d": {"gap_m": 0.01}}
    with pytest.raises(ValueError):
        set_by_path(obj, "pic1d.no_such_field", 1.0)


# ---- 5. resolve_sweep_module: module の自動判定・明示指定 (prompts/96) --------------


def test_resolve_sweep_module_auto_detects_pic1d_prefix():
    assert resolve_sweep_module("pic1d.n_steps", None) == "pic1d"
    assert resolve_sweep_module("pic1d.left.v_dc", None) == "pic1d"


def test_resolve_sweep_module_auto_defaults_to_pic_for_other_paths():
    # "pic." 始まりに限らず、pic1d 以外のパスは全て従来互換で "pic" になる
    assert resolve_sweep_module("pic.n_steps", None) == "pic"
    assert resolve_sweep_module("geometry.boundaries.0.voltage", None) == "pic"


def test_resolve_sweep_module_explicit_value_overrides_path_prefix():
    # UI 確定値 (明示指定) が最優先。パスの接頭辞と矛盾していても明示指定が勝つ
    assert resolve_sweep_module("pic.n_steps", "pic1d") == "pic1d"
    assert resolve_sweep_module("pic1d.n_steps", "pic") == "pic"


# ---- 6. 1D スイープ: 2ケース実行、bundle に pic1d 結果が入ること (prompts/96) ----------


def test_ws_sweep_pic1d_two_values_full_flow_and_result_endpoint():
    client = TestClient(srv.app)
    project = _tiny_pic1d_project(init_density=1.0e14)

    with client.websocket_connect("/ws/sweep") as ws:
        ws.send_json(
            {
                "cmd": "start",
                "project": project,
                "param_path": "pic1d.init_density_m3",
                "values": [1.0e14, 2.0e14],
                "parallel": 2,
            }
        )
        started = ws.receive_json()
        assert started["type"] == "started"
        assert started["n_cases"] == 2
        assert started["module"] == "pic1d"  # module 未指定 → param_path の接頭辞から自動判定

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

    # GET /sweep/result/{i}: 各ケースの project にスイープ値が反映され、
    # 結果バンドルに pic1d 結果 (server.py の _pic1d_result = build_pic1d_result と同形) が入っていること
    resp0 = client.get("/sweep/result/0")
    assert resp0.status_code == 200
    obj0 = resp0.json()
    assert obj0["pic1d"]["init_density_m3"] == 1.0e14
    assert obj0["results"]["version"] == 1
    assert "pic" not in obj0["results"]  # 2D 結果キーは含まれない (pic1d 実行のため)
    pic1d_result0 = obj0["results"]["pic1d"]
    assert pic1d_result0 is not None
    assert set(pic1d_result0) == {
        "history", "profiles", "sheath", "cycle", "sheath_fft", "sheath_ts",
        "eedf", "wall_iedf", "walls", "fn", "elapsed_s", "timing", "settings",
    }
    assert pic1d_result0["settings"]["init_density_m3"] == 1.0e14  # 上書きした値が反映されている

    resp1 = client.get("/sweep/result/1")
    assert resp1.status_code == 200
    obj1 = resp1.json()
    assert obj1["pic1d"]["init_density_m3"] == 2.0e14
    assert obj1["results"]["pic1d"]["settings"]["init_density_m3"] == 2.0e14


def test_ws_sweep_explicit_module_is_echoed_in_started_and_overrides_default():
    """module を明示指定すると started にそのまま反映される (自動判定に頼らない、prompts/96)。"""
    client = TestClient(srv.app)
    project = _tiny_pic1d_project()

    with client.websocket_connect("/ws/sweep") as ws:
        ws.send_json(
            {
                "cmd": "start",
                "project": project,
                "param_path": "pic1d.n_macro",
                "values": [500.0],
                "parallel": 1,
                "module": "pic1d",
            }
        )
        started = ws.receive_json()
        assert started["type"] == "started"
        assert started["module"] == "pic1d"

        done = False
        while not done:
            msg = ws.receive_json()
            assert msg["type"] != "error", msg.get("detail")
            done = msg["type"] == "done"


def test_ws_sweep_rejects_invalid_module_value():
    client = TestClient(srv.app)
    project = _tiny_pic_project()

    with client.websocket_connect("/ws/sweep") as ws:
        ws.send_json(
            {
                "cmd": "start",
                "project": project,
                "param_path": "geometry.boundaries.0.voltage",
                "values": [10.0, 20.0],
                "parallel": 1,
                "module": "bogus",
            }
        )
        msg = ws.receive_json()
        assert msg["type"] == "error"
