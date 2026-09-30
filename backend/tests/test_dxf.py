"""DXF の読み書き (prompts/132 P7g、es_sim.dxf)。

- 読み込み: 線・円弧 (押し出し -Z は向きが逆)・円・LWPOLYLINE と 2D POLYLINE の bulge・楕円とスプライン (折れ線)・
  ブロックの展開 (回転・拡大、中のレイヤ 0 は INSERT のレイヤ、非一様な拡大の円は楕円 → 折れ線)・単位
  ($INSUNITS・指定・無ければ mm とみなす)・レイヤの色と表示・形でないものの数・重なった点の整理
- 書き出し → 読み込みで形 (ドメイン・領域の外周と穴・円・スケッチ) とレイヤが戻る
- サーバの口 (base64 で受ける・誤りは 422)
"""

import base64
import io
import math

import ezdxf
import pytest
from fastapi.testclient import TestClient

from es_sim.dxf import read_dxf, write_dxf
from es_sim.server import app


def _text(doc) -> bytes:
    s = io.StringIO()
    doc.write(s)
    return s.getvalue().encode("utf-8")


def _by_kind(r, kind):
    return [e for e in r["entities"] if e["kind"] == kind]


def test_reads_entities_in_metres_with_arcs_and_bulges():
    doc = ezdxf.new("R2010", units=4)  # mm
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 0))
    msp.add_arc((0, 0), 5, 0, 90)
    msp.add_arc((0, 0), 5, 0, 90, dxfattribs={"extrusion": (0, 0, -1)})
    msp.add_arc((30, 0), 2, 0, 360)
    msp.add_circle((20, 20), 3)
    msp.add_lwpolyline([(0, 0, 0.5), (10, 0, 0), (10, 5, 0)], format="xyb", close=True)
    msp.add_lwpolyline([(0, 10, 1.0), (4, 10, 0)], format="xyb")  # 2 点の開いた形 = 円弧
    pl = msp.add_polyline2d([(0, 20), (5, 20), (5, 25)], format="xy", close=True)
    pl.vertices[0].dxf.bulge = -0.25
    msp.add_text("label")
    msp.add_line((1, 1), (1, 1))  # 長さ 0 は飛ばす
    r = read_dxf(_text(doc))
    assert r["unit"] == "mm" and r["assumed"] is False
    assert r["skipped"] == {"TEXT": 1}
    line = _by_kind(r, "line")[0]
    assert line["a"] == [0.0, 0.0] and line["b"] == pytest.approx([0.01, 0.0])
    arcs = _by_kind(r, "arc")
    q = math.tan(math.pi / 8)
    assert arcs[0]["a"] == pytest.approx([0.005, 0.0]) and arcs[0]["bulge"] == pytest.approx(q)
    # 押し出し -Z: x が鏡映され時計回り
    assert arcs[1]["a"] == pytest.approx([-0.005, 0.0]) and arcs[1]["bulge"] == pytest.approx(-q)
    # 2 点の LWPOLYLINE は円弧
    assert arcs[2]["a"] == pytest.approx([0.0, 0.01]) and arcs[2]["bulge"] == pytest.approx(1.0)
    circles = _by_kind(r, "circle")
    assert [c["r"] for c in circles] == pytest.approx([0.002, 0.003])
    assert circles[1]["center"] == pytest.approx([0.02, 0.02])
    polys = _by_kind(r, "polyline")
    assert polys[0]["closed"] is True and polys[0]["bulges"] == [0.5, 0.0, 0.0]
    assert polys[1]["closed"] is True and polys[1]["bulges"][0] == -0.25
    assert all(e["layer"] == "0" for e in r["entities"])


def test_flattens_curves_and_explodes_blocks_with_layer_zero_inheritance():
    doc = ezdxf.new("R2010", units=6)  # m
    doc.layers.add("EL", color=1)
    msp = doc.modelspace()
    msp.add_ellipse((0, 0), major_axis=(1, 0), ratio=0.5)
    msp.add_spline([(0, 2), (1, 3), (2, 2)])
    blk = doc.blocks.new("B")
    blk.add_circle((1, 0), 0.5)  # レイヤ 0 → INSERT のレイヤ
    blk.add_line((0, 0), (1, 0), dxfattribs={"layer": "EL"})
    msp.add_blockref("B", (10, 0), dxfattribs={"xscale": 2, "yscale": 2, "rotation": 90, "layer": "EL"})
    msp.add_blockref("B", (20, 0), dxfattribs={"xscale": 2, "yscale": 1})
    r = read_dxf(_text(doc))
    assert r["unit"] == "m"
    polys = _by_kind(r, "polyline")
    ellipse = polys[0]
    assert ellipse["closed"] is True and len(ellipse["points"]) > 20
    assert max(abs(p[0]) for p in ellipse["points"]) == pytest.approx(1.0, rel=1e-6)
    spline = polys[1]
    assert spline["closed"] is False and spline["points"][0] == pytest.approx([0.0, 2.0])
    c = _by_kind(r, "circle")[0]
    assert c["center"] == pytest.approx([10.0, 2.0]) and c["r"] == pytest.approx(1.0) and c["layer"] == "EL"
    # 非一様な拡大の円は楕円 → 閉じた折れ線
    assert polys[2]["closed"] is True and max(p[0] for p in polys[2]["points"]) == pytest.approx(23.0, rel=1e-6)
    assert {e["layer"] for e in _by_kind(r, "line")} == {"EL"}
    assert {la["name"]: la["color"] for la in r["layers"]} == {"0": None, "EL": "#ff0000"}


def test_units_from_header_argument_or_assumed():
    for code, scale in [(1, 0.0254), (2, 0.3048), (5, 0.01), (13, 1e-6)]:
        doc = ezdxf.new("R2010", units=code)
        doc.modelspace().add_line((0, 0), (1, 0))
        assert read_dxf(_text(doc))["entities"][0]["b"][0] == pytest.approx(scale)
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 0
    doc.modelspace().add_line((0, 0), (1, 0))
    r = read_dxf(_text(doc))
    assert r["assumed"] is True and r["unit"] == "mm" and r["entities"][0]["b"][0] == pytest.approx(1e-3)
    assert read_dxf(_text(doc), "m")["entities"][0]["b"][0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="単位"):
        read_dxf(_text(doc), "furlong")
    with pytest.raises(ValueError, match="DXF を読めません"):
        read_dxf(b"this is not a dxf file")


def test_cleans_polylines_and_reports_layer_visibility():
    doc = ezdxf.new("R2010", units=4)
    doc.layers.add("HID", color=5).off()
    doc.layers.add("FRZ", color=2).freeze()
    msp = doc.modelspace()
    # 続けて同じ点、最後の点が最初の点と同じ (閉じた形とみなす)
    msp.add_lwpolyline([(0, 0), (5, 0), (5, 0), (5, 5), (0, 0)], dxfattribs={"layer": "HID"})
    msp.add_line((0, 0), (1, 0), dxfattribs={"layer": "FRZ"})
    r = read_dxf(_text(doc))
    poly = _by_kind(r, "polyline")[0]
    assert poly["closed"] is True and len(poly["points"]) == 3 and len(poly["bulges"]) == 3
    vis = {la["name"]: la["visible"] for la in r["layers"]}
    assert vis == {"HID": False, "FRZ": False}


def _project():
    t = math.tan(math.pi / 8)
    return {
        "geometry": {
            "domain": {"polygon": [[0, 0], [0.1, 0], [0.1, 0.05], [0, 0.05]], "bulges": [0, 0, 0.2, 0]},
            "regions": [
                {"id": "a", "type": "conductor", "voltage": 0, "polygon": [[0.01, 0.01], [0.04, 0.01], [0.04, 0.04], [0.01, 0.04]], "holes": [{"polygon": [[0.03, 0.025], [0.02, 0.025]], "bulges": [1, 1]}], "layer": "L1"},
                {"id": "c", "type": "dielectric", "shape": {"kind": "circle", "center": [0.07, 0.025], "radius": 0.005}},
            ],
            "boundaries": [],
        },
        "mesh": {"size": 0.002},
        "cad": {
            "layers": [{"id": "0", "name": "0"}, {"id": "L1", "name": "電極/A", "color": "#ff8800", "visible": False}],
            "sketch": [
                {"id": "s1", "kind": "line", "a": [0, 0.06], "b": [0.1, 0.06]},
                {"id": "s2", "kind": "arc", "a": [0.01, 0.07], "b": [0.02, 0.08], "bulge": -t, "layer": "L1"},
                {"id": "s3", "kind": "circle", "center": [0.05, 0.07], "r": 0.004},
                {"id": "s4", "kind": "polyline", "points": [[0, 0.09], [0.01, 0.09], [0.02, 0.1]], "bulges": [0.3, 0, 0], "closed": False},
            ],
        },
    }


def test_export_then_import_gives_the_same_shapes():
    text = write_dxf(_project(), "mm")
    doc = ezdxf.read(io.StringIO(text))
    assert doc.header["$INSUNITS"] == 4
    assert {la.dxf.name for la in doc.layers} >= {"0", "ES_DOMAIN", "電極_A"}
    lay = doc.layers.get("電極_A")
    assert lay.rgb == (255, 136, 0) and lay.is_off()
    r = read_dxf(text.encode("utf-8"))
    by_layer = {}
    for e in r["entities"]:
        by_layer.setdefault(e["layer"], []).append(e)
    dom = by_layer["ES_DOMAIN"][0]
    assert dom["kind"] == "polyline" and dom["closed"] and dom["bulges"] == pytest.approx([0, 0, 0.2, 0])
    assert dom["points"][2] == pytest.approx([0.1, 0.05])
    outer, hole, arc = by_layer["電極_A"]
    assert outer["points"][1] == pytest.approx([0.04, 0.01]) and outer["closed"]
    assert hole["points"][0] == pytest.approx([0.03, 0.025]) and hole["points"][1] == pytest.approx([0.02, 0.025])
    assert hole["bulges"] == pytest.approx([1, 1])
    # 時計回りの円弧は反時計回りの ARC (端点が入れ替わる) として戻る
    assert arc["kind"] == "arc" and arc["bulge"] == pytest.approx(math.tan(math.pi / 8))
    assert arc["a"] == pytest.approx([0.02, 0.08]) and arc["b"] == pytest.approx([0.01, 0.07])
    zero = by_layer["0"]
    kinds = [e["kind"] for e in zero]
    assert kinds == ["circle", "line", "circle", "polyline"]
    assert zero[0]["center"] == pytest.approx([0.07, 0.025]) and zero[0]["r"] == pytest.approx(0.005)
    assert zero[3]["closed"] is False and zero[3]["bulges"][0] == pytest.approx(0.3)
    # m でも書ける
    assert read_dxf(write_dxf(_project(), "m").encode("utf-8"))["unit"] == "m"
    with pytest.raises(ValueError):
        write_dxf(_project(), "parsec")


def test_endpoints():
    client = TestClient(app)
    data = base64.b64encode(write_dxf(_project(), "mm").encode("utf-8")).decode("ascii")
    res = client.post("/v2/cad/dxf/import", json={"data": data})
    assert res.status_code == 200
    body = res.json()
    assert body["unit"] == "mm" and body["counts"]["polyline"] >= 3
    assert client.post("/v2/cad/dxf/import", json={"data": "not base64!"}).status_code == 422
    assert client.post("/v2/cad/dxf/import", json={"data": base64.b64encode(b"garbage").decode()}).status_code == 422
    res = client.post("/v2/cad/dxf/export", json={"project": _project(), "unit": "mm"})
    assert res.status_code == 200 and "LWPOLYLINE" in res.text and res.headers["content-type"].startswith("application/dxf")
    assert client.post("/v2/cad/dxf/export", json={"project": _project(), "unit": "x"}).status_code == 422
