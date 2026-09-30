"""パラメータと式の束縛 (prompts/132 P7f)。

- 式の文法 (数と単位・演算・関数・名前) と、パラメータの計算の順・循環・名前の検査 (UI と共通の試験データ)
- スキーマの検査 (名前・知らない名前・束縛の式)
- apply_params: 束縛の場所 (キー・番号・{"id"}・{"edge"}) に計算した値を書く、場所が無い・数値でない欄はエラー
- スイープ: "params.<名前>" で束縛した欄ごと計算し直す。平行平板のドメインの幅をパラメータにして静電容量が 1/幅
"""

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from es_sim.fem import EPS0, solve
from es_sim.meshing import generate_mesh
from es_sim.params import ExprError, apply_params, evaluate, evaluate_params, name_problem, names_in
from es_sim.schema import Project
from es_sim.sweep import build_sweep_cases

CASES = json.loads((Path(__file__).parent / "data" / "param_expr_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES["expressions"], ids=lambda c: c["expr"] or "(empty)")
def test_shared_expressions(case):
    if "error" in case:
        with pytest.raises(ExprError) as info:
            evaluate(case["expr"], case.get("vars"))
        assert info.value.kind == case["error"]
    else:
        assert evaluate(case["expr"], case.get("vars")) == pytest.approx(case["value"], rel=1e-12)


@pytest.mark.parametrize("case", CASES["params"], ids=lambda c: ",".join(v["name"] for v in c["vars"]))
def test_shared_params(case):
    if "error" in case:
        with pytest.raises(ExprError) as info:
            evaluate_params(case["vars"], case.get("overrides"))
        assert info.value.kind == case["error"]
    else:
        values = evaluate_params(case["vars"], case.get("overrides"))
        assert values == pytest.approx(case["values"], rel=1e-12)


def test_shared_names():
    for c in CASES["names"]:
        assert (name_problem(c["name"]) is None) == c["ok"], c["name"]


def test_names_are_collected_without_evaluating():
    # 途中で 0 で割っても、後ろの名前まで集める
    assert names_in("1/(a-1) + b*sin(c) + 2 mm") == {"a", "b", "c"}


def _plates(**extra) -> dict:
    return {
        "geometry": {
            "domain": {"polygon": [[0.0, 0.0], [0.1, 0.0], [0.1, 0.05], [0.0, 0.05]], "edge_ids": ["e1", "e2", "e3", "e4"]},
            "regions": [
                {"id": "a", "type": "conductor", "voltage": 5.0, "polygon": [[0.02, 0.02], [0.03, 0.02], [0.03, 0.03]]},
                {"id": "d", "type": "dielectric", "eps_r": 2.0, "polygon": [[0.05, 0.01], [0.06, 0.01], [0.06, 0.02]]},
            ],
            "boundaries": [{"edges": [3], "voltage": 0.0}, {"edges": [1], "voltage": 100.0}],
        },
        "mesh": {"size": 0.005},
        **extra,
    }


def test_schema_checks_params():
    ok = Project.model_validate(_plates(params={"vars": [{"name": "V0", "expr": "100"}], "bindings": [{"path": ["geometry", "regions", {"id": "a"}, "voltage"], "expr": "V0/2"}]}))
    assert ok.params is not None and ok.params.vars[0].name == "V0"
    bad = {
        "は単位の記号": {"vars": [{"name": "V", "expr": "1"}]},
        "b というパラメータはありません": {"vars": [{"name": "a", "expr": "b"}]},
        "循環": {"vars": [{"name": "a", "expr": "b"}, {"name": "b", "expr": "a"}]},
        "束縛 x: q というパラメータ": {"bindings": [{"path": ["x"], "expr": "q"}]},
        "重複": {"vars": [{"name": "x", "expr": "1"}, {"name": "x", "expr": "2"}]},
    }
    for msg, params in bad.items():
        with pytest.raises(ValidationError, match=msg):
            Project.model_validate(_plates(params=params))
    # 領域のレイヤ (UI だけが使う) は受け付ける
    assert Project.model_validate(_plates()).geometry.regions[0].layer is None
    p = _plates()
    p["geometry"]["regions"][0]["layer"] = "L1"
    assert Project.model_validate(p).geometry.regions[0].layer == "L1"


def test_apply_params_writes_bound_fields():
    doc = _plates(
        params={
            "vars": [{"name": "V0", "expr": "100"}, {"name": "gap", "expr": "2 mm"}, {"name": "eps", "expr": "4"}],
            "bindings": [
                {"path": ["geometry", "regions", {"id": "a"}, "voltage"], "expr": "V0/2"},
                {"path": ["geometry", "boundaries", {"edge": "e2"}, "voltage"], "expr": "V0"},
                {"path": ["geometry", "regions", {"id": "d"}, "eps_r"], "expr": "eps"},
                {"path": ["geometry", "regions", {"id": "a"}, "polygon", 1, 0], "expr": "0.02 + gap"},
                {"path": ["mesh", "size"], "expr": "gap"},
            ],
        }
    )
    values = apply_params(doc, {"V0": 300.0})
    assert values == {"V0": 300.0, "gap": 0.002, "eps": 4.0}
    geo = doc["geometry"]
    assert geo["regions"][0]["voltage"] == 150.0
    assert geo["boundaries"][1]["voltage"] == 300.0
    assert geo["boundaries"][0]["voltage"] == 0.0
    assert geo["regions"][1]["eps_r"] == 4.0
    assert geo["regions"][0]["polygon"][1][0] == pytest.approx(0.022, rel=1e-12)
    assert doc["mesh"]["size"] == 0.002
    assert doc["params"]["vars"][0] == {"name": "V0", "expr": "300.0", "value": 300.0}
    assert doc["params"]["vars"][1]["value"] == 0.002
    Project.model_validate(doc)


def test_apply_params_reports_broken_bindings():
    for path, msg in [
        (["geometry", "regions", {"id": "zz"}, "voltage"], "id が 'zz' の要素がありません"),
        (["geometry", "boundaries", {"edge": "e9"}, "voltage"], "辺 'e9' がありません"),
        (["geometry", "boundaries", {"edge": "e1"}, "voltage"], "辺 'e1' の境界条件がありません"),
        (["geometry", "regions", 5, "voltage"], "番号 5 の要素がありません"),
        (["geometry", "regions", {"id": "a"}, "type"], "数値の欄ではありません"),
        (["mesh", "nothing"], "'nothing' がありません"),
    ]:
        doc = _plates(params={"vars": [{"name": "v", "expr": "1"}], "bindings": [{"path": path, "expr": "v"}]})
        with pytest.raises(ValueError, match=msg):
            apply_params(doc)
    with pytest.raises(ExprError, match="パラメータがありません"):
        apply_params(_plates(), {"x": 1.0})
    assert apply_params(_plates()) == {}


def test_sweep_over_a_parameter_recomputes_bound_fields():
    doc = _plates(
        params={
            "vars": [{"name": "V0", "expr": "100"}, {"name": "Vh", "expr": "V0/2"}],
            "bindings": [{"path": ["geometry", "regions", {"id": "a"}, "voltage"], "expr": "Vh"}],
        }
    )
    cases = build_sweep_cases(doc, "params.V0", [10.0, 20.0])
    assert [c["geometry"]["regions"][0]["voltage"] for c in cases] == [5.0, 10.0]
    assert [c["params"]["vars"][1]["value"] for c in cases] == [5.0, 10.0]
    # 元の文書は変えない
    assert doc["geometry"]["regions"][0]["voltage"] == 5.0
    with pytest.raises(ExprError):
        build_sweep_cases(doc, "params.nothing", [1.0])


def test_swept_domain_width_scales_the_capacitance():
    """平行平板 (左 0 V・右 100 V) の間隔 L をパラメータにして右の辺の 2 頂点の x を束縛: C = ε0 H / L。"""
    doc = _plates(
        params={
            "vars": [{"name": "L", "expr": "100 mm"}],
            "bindings": [
                {"path": ["geometry", "domain", "polygon", 1, 0], "expr": "L"},
                {"path": ["geometry", "domain", "polygon", 2, 0], "expr": "L"},
            ],
        }
    )
    doc["geometry"]["regions"] = []
    caps = []
    for case in build_sweep_cases(doc, "params.L", [0.1, 0.05]):
        p = Project.model_validate(case)
        sol = solve(p, generate_mesh(p))
        caps.append({label: q for label, _, q in sol.charges}["edge1"] / 100.0)
    h = 0.05
    for L, c in zip([0.1, 0.05], caps):
        assert c == pytest.approx(EPS0 * h / L, rel=1e-6)
    assert caps[1] / caps[0] == pytest.approx(2.0, rel=1e-6)
    assert math.isfinite(caps[0])
