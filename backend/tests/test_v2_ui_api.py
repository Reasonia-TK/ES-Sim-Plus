"""UI v2 (ui/) が使うバックエンドの API のテスト (prompts/130)。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import es_sim
import es_sim.server as server
from es_sim.schema import Project

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_v2_schema_describes_the_project_model():
    client = TestClient(server.app)
    res = client.get("/v2/schema")
    assert res.status_code == 200
    body = res.json()
    assert body["version"] == es_sim.__version__
    schema = body["project"]
    assert {"geometry", "mesh", "pic", "pic1d", "fluid1d", "fluid2d", "dsmc", "tl"} <= set(schema["properties"])
    defs = schema["$defs"]
    assert {"Fluid2dSettings", "AmrSettings", "Region", "BoundaryCondition"} <= set(defs)
    # 範囲・既定値の制約がそのまま載る (UI のフォームの検証に使う)
    max_level = defs["AmrSettings"]["properties"]["max_level"]
    assert max_level["minimum"] == 0 and max_level["maximum"] == 6 and max_level["default"] == 0
    assert defs["Fluid2dSettings"]["properties"]["gas_pressure_pa"]["exclusiveMinimum"] == 0
    # 2 回目はキャッシュ (同じ内容)
    assert client.get("/v2/schema").json() == body


def test_health_reports_a_stable_instance_id():
    """UI v2 は instance が変わったらバックエンドが再起動したとみなしてスキーマを取り直す。"""
    client = TestClient(server.app)
    a = client.get("/health").json()
    b = client.get("/health").json()
    assert isinstance(a["instance"], str) and len(a["instance"]) == 16
    assert a["instance"] == b["instance"] == server.SERVER_INSTANCE


def test_every_numeric_field_declares_its_unit():
    """UI の設定フォームは x-unit で単位を出し、x-geom の長さは表示単位 (mm/µm) で入出力する。"""
    from es_sim.ui_schema import project_schema

    schema = project_schema()
    missing = []
    for name, d in [("Project", schema), *schema["$defs"].items()]:
        for key, prop in d.get("properties", {}).items():
            types = {prop.get("type")} | {a.get("type") for a in prop.get("anyOf", [])}
            if "number" in types and "x-unit" not in prop:
                missing.append(f"{name}.{key}")
    assert missing == []
    props = schema["$defs"]["PicSettings"]["properties"]
    assert props["dt"]["x-unit"] == "s" and props["threads"]["x-advanced"] is True
    assert schema["$defs"]["CircleShape"]["properties"]["center"]["x-geom"] is True
    # 分子の直径は幾何の長さではない (mm で表示しない)
    assert "x-geom" not in schema["$defs"]["DsmcGas"]["properties"]["d_ref_m"]


def test_ui_schema_snapshot_is_current():
    """ui/src/schema/project.schema.json が schema.py と一致する (違えば python -m es_sim.ui_schema で書き直す)。"""
    from es_sim.ui_schema import SNAPSHOT, snapshot_text

    assert SNAPSHOT.exists(), "python -m es_sim.ui_schema で ui/src/schema/project.schema.json を作ってください"
    assert json.loads(SNAPSHOT.read_text(encoding="utf-8")) == json.loads(snapshot_text()), (
        "UI に同梱したスキーマが古い: python -m es_sim.ui_schema で書き直してください"
    )


def test_examples_bundled_in_the_ui_are_valid_projects():
    files = sorted(EXAMPLES.glob("*.json"))
    assert files, "examples/*.json が見つかりません"
    for f in files:
        Project.model_validate(json.loads(f.read_text(encoding="utf-8")))


def test_old_documents_with_the_removed_solver_setting_still_load():
    """初版の "solver": {"backend": ...} (計算に効いていなかった) は消したが、それを含む古い文書も読める
    (知らないキーとして捨てる)。"""
    doc = json.loads((EXAMPLES / "parallel_plates.json").read_text(encoding="utf-8"))
    doc["solver"] = {"backend": "cupy"}
    assert "solver" not in Project.model_validate(doc).model_dump()
    assert "solver" not in Project.model_json_schema()["properties"]
    res = TestClient(server.app).post("/solve", json=doc)
    assert res.status_code == 200, res.text
    assert res.json()["capacitance"] > 0
