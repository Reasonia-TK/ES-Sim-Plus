"""DXF の読み書き (CAD v2 P7g、prompts/132)。ezdxf を使う。

読み込み (read_dxf): モデル空間の LINE・ARC・CIRCLE・LWPOLYLINE・POLYLINE (2D。3D は x, y だけ)・ELLIPSE と SPLINE
(大きさの 1e-4 の幅の折れ線に)・INSERT (ブロックを展開。ブロックの中のレイヤ 0 の形は INSERT のレイヤ) を、UI の
スケッチの形 (線・円弧・円・ポリライン、座標は m、円弧は bulge) とレイヤ (名前・色・表示) にする。単位は指定、無ければ
$INSUNITS、それも無ければ mm とみなす (assumed)。OCS の押し出しが -Z の形は鏡映として扱い、Z 軸に平行でない形は
飛ばす。文字・寸法・ハッチなど形でないものは種類ごとの数だけ返す。

書き出し (write_dxf): ドメイン (レイヤ ES_DOMAIN)・領域 (外周と穴は閉じた LWPOLYLINE、円は CIRCLE)・スケッチ
(LINE・ARC・CIRCLE・LWPOLYLINE) を、UI のレイヤ (名前・色・表示) ごとに書く。単位は mm か m など ($INSUNITS)。
"""

from __future__ import annotations

import io
import math
import os
import re
import tempfile
from collections import Counter
from typing import Any

import ezdxf
from ezdxf import colors, recover
from ezdxf.bbox import extents

from .paths import arc_of, bulge_of

# 単位の名前 → ($INSUNITS の番号, 1 単位の m)
UNITS: dict[str, tuple[int, float]] = {
    "mm": (4, 1e-3),
    "cm": (5, 1e-2),
    "m": (6, 1.0),
    "um": (13, 1e-6),
    "in": (1, 0.0254),
    "ft": (2, 0.3048),
}
# $INSUNITS の番号 → 1 単位の m (0 は単位なし)
_INSUNITS_M = {1: 0.0254, 2: 0.3048, 4: 1e-3, 5: 1e-2, 6: 1.0, 8: 2.54e-8, 9: 2.54e-5, 10: 0.9144, 11: 1e-10, 12: 1e-9, 13: 1e-6, 14: 0.1}
_UNIT_NAME = {4: "mm", 5: "cm", 6: "m", 13: "um", 1: "in", 2: "ft"}

DOMAIN_LAYER = "ES_DOMAIN"
_MAX_DEPTH = 8


def _hex(rgb: Any) -> str:
    r, g, b = (int(c) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _dist(p: list[float], q: list[float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _clean_polyline(points: list[list[float]], bulges: list[float], closed: bool, tol: float) -> tuple[list[list[float]], list[float], bool]:
    """続けて同じ点 (長さ 0 の辺) を除き、最後の点が最初の点と同じなら閉じた形にする。"""
    pts = list(points)
    bs = list(bulges)
    k = 0
    while k < len(pts) - 1:
        if _dist(pts[k], pts[k + 1]) <= tol:
            del pts[k + 1]
            del bs[k]
        else:
            k += 1
    if len(pts) >= 3 and _dist(pts[0], pts[-1]) <= tol:
        del pts[-1]
        del bs[-1]
        closed = True
    return pts, bs, closed


def read_dxf(data: bytes, unit: str | None = None) -> dict[str, Any]:
    """DXF (テキストかバイナリ) → スケッチの形とレイヤ。unit は "mm" など (None なら $INSUNITS、無ければ mm)。"""
    fd, path = tempfile.mkstemp(suffix=".dxf")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        try:
            doc, _auditor = recover.readfile(path)
        except (ezdxf.DXFStructureError, OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"DXF を読めません: {exc}") from exc
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    insunits = int(doc.header.get("$INSUNITS", 0) or 0)
    assumed = False
    if unit is not None:
        if unit not in UNITS:
            raise ValueError(f"単位 {unit!r} は使えません ({', '.join(UNITS)})")
        scale = UNITS[unit][1]
        used = unit
    elif insunits in _INSUNITS_M:
        scale = _INSUNITS_M[insunits]
        used = _UNIT_NAME.get(insunits, f"insunits{insunits}")
    else:
        scale, used, assumed = 1e-3, "mm", True
    msp = doc.modelspace()
    try:
        box = extents(msp, fast=True)
        size = max(box.size.x, box.size.y) if box.has_data else 0.0
    except Exception:  # noqa: BLE001 (壊れた形があっても読み込みは続ける)
        size = 0.0
    flat = (size if size > 0 else 1.0) * 1e-4  # 楕円・スプラインを折れ線にする幅 (図の単位)
    tol = (size if size > 0 else 1.0) * 1e-9  # 同じ点とみなす距離 (図の単位)

    out: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    used_layers: set[str] = set()

    def pt(v: Any) -> list[float]:
        return [float(v[0]) * scale, float(v[1]) * scale]

    def add(e: dict[str, Any], layer: str) -> None:
        e["layer"] = layer
        used_layers.add(layer)
        out.append(e)

    def polyline(points: list[Any], bulges: list[float], closed: bool, layer: str) -> None:
        pts, bs, closed = _clean_polyline([[float(p[0]), float(p[1])] for p in points], list(bulges), closed, tol)
        if len(pts) < 2 or (closed and len(pts) < 3 and not any(bs)):
            return
        if not closed and len(pts) == 2:
            b = bs[0]
            add({"kind": "arc", "a": pt(pts[0]), "b": pt(pts[1]), "bulge": b} if b else {"kind": "line", "a": pt(pts[0]), "b": pt(pts[1])}, layer)
            return
        add({"kind": "polyline", "points": [pt(p) for p in pts], "bulges": [float(b) for b in bs], "closed": closed}, layer)

    def visit(e: Any, inherited: str | None, depth: int) -> None:
        t = e.dxftype()
        layer = str(e.dxf.get("layer", "0") or "0")
        if layer == "0" and inherited is not None:
            layer = inherited
        if t == "INSERT":
            if depth >= _MAX_DEPTH:
                skipped[t] += 1
                return
            try:
                children = list(e.virtual_entities())
            except Exception:  # noqa: BLE001 (展開できないブロックは飛ばす)
                skipped[t] += 1
                return
            for c in children:
                visit(c, layer, depth + 1)
            return
        ext = e.dxf.get("extrusion", (0.0, 0.0, 1.0))
        if t in ("ARC", "CIRCLE", "LWPOLYLINE", "POLYLINE") and (abs(ext[0]) > 1e-9 or abs(ext[1]) > 1e-9):
            skipped[t] += 1
            return
        flip = -1.0 if float(ext[2]) < 0 else 1.0
        if t == "LINE":
            a, b = e.dxf.start, e.dxf.end
            if _dist([a[0], a[1]], [b[0], b[1]]) > tol:
                add({"kind": "line", "a": pt(a), "b": pt(b)}, layer)
        elif t == "ARC":
            sweep = (float(e.dxf.end_angle) - float(e.dxf.start_angle)) % 360.0
            r = float(e.dxf.radius)
            if r <= 0:
                return
            if sweep <= 1e-12 or sweep >= 360.0 - 1e-9:
                add({"kind": "circle", "center": pt(e.ocs().to_wcs(e.dxf.center)), "r": r * scale}, layer)
            else:
                add({"kind": "arc", "a": pt(e.start_point), "b": pt(e.end_point), "bulge": flip * math.tan(math.radians(sweep) / 4.0)}, layer)
        elif t == "CIRCLE":
            r = float(e.dxf.radius)
            if r > 0:
                add({"kind": "circle", "center": pt(e.ocs().to_wcs(e.dxf.center)), "r": r * scale}, layer)
        elif t == "LWPOLYLINE":
            pts = list(e.vertices_in_wcs())
            bs = [flip * float(b) for (_x, _y, b) in e.get_points("xyb")]
            polyline(pts, bs, bool(e.closed), layer)
        elif t == "POLYLINE":
            if e.is_2d_polyline:
                pts = list(e.points_in_wcs())
                bs = [flip * float(v.dxf.get("bulge", 0.0) or 0.0) for v in e.vertices]
                polyline(pts, bs, bool(e.is_closed), layer)
            elif e.is_3d_polyline:
                pts = [v.dxf.location for v in e.vertices]
                polyline(pts, [0.0] * len(pts), bool(e.is_closed), layer)
            else:
                skipped[t] += 1
        elif t in ("ELLIPSE", "SPLINE"):
            try:
                pts = list(e.flattening(flat))
            except Exception:  # noqa: BLE001
                skipped[t] += 1
                return
            closed = bool(getattr(e, "closed", False))
            polyline(pts, [0.0] * len(pts), closed, layer)
        else:
            skipped[t] += 1

    for e in msp:
        visit(e, None, 0)

    layers = []
    for lay in doc.layers:
        name = str(lay.dxf.name)
        if name not in used_layers:
            continue
        rgb = lay.rgb
        aci = abs(int(lay.color))
        # 7 (白/黒) と BYBLOCK・BYLAYER は既定の色
        color = _hex(rgb) if rgb else (None if aci in (0, 7, 256) else _hex(colors.aci2rgb(aci)))
        layers.append({"name": name, "color": color, "visible": not (lay.is_off() or lay.is_frozen())})
    known = {lay["name"] for lay in layers}
    for name in sorted(used_layers - known):
        layers.append({"name": name, "color": None, "visible": True})
    counts = Counter(e["kind"] for e in out)
    return {"unit": used, "assumed": assumed, "entities": out, "layers": layers, "counts": dict(counts), "skipped": dict(skipped)}


# ---- 書き出し ----

_BAD_NAME = re.compile(r'[<>/\\":;?*|=`]')


def _layer_name(name: str) -> str:
    s = _BAD_NAME.sub("_", name).strip()
    return s[:255] or "0"


def write_dxf(project: dict[str, Any], unit: str = "mm") -> str:
    """文書 (dict) → DXF のテキスト (R2010)。unit は "mm" など。"""
    if unit not in UNITS:
        raise ValueError(f"単位 {unit!r} は使えません ({', '.join(UNITS)})")
    code, per = UNITS[unit]
    k = 1.0 / per  # m → 図の単位
    doc = ezdxf.new("R2010", units=code)
    msp = doc.modelspace()
    cad = project.get("cad") or {}
    ui_layers = {str(la.get("id")): la for la in (cad.get("layers") or []) if isinstance(la, dict)}

    def layer_of(item: dict[str, Any]) -> str:
        lid = item.get("layer")
        if lid is None or lid == "0" or lid not in ui_layers:
            info = ui_layers.get("0")
            name = "0"
        else:
            info = ui_layers[lid]
            name = _layer_name(str(info.get("name") or lid))
        if name not in doc.layers:
            doc.layers.add(name)
        dxf_layer = doc.layers.get(name)
        if info:
            if info.get("color"):
                c = str(info["color"]).lstrip("#")
                if len(c) == 6:
                    dxf_layer.rgb = (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
            if info.get("visible") is False:
                dxf_layer.off()
        return name

    def p(q: Any) -> tuple[float, float]:
        return (float(q[0]) * k, float(q[1]) * k)

    def closed_path(points: list[Any], bulges: list[float] | None, layer: str) -> None:
        n = len(points)
        pts = [(*p(points[i]), bulge_of(bulges, i)) for i in range(n)]
        msp.add_lwpolyline(pts, format="xyb", close=True, dxfattribs={"layer": layer})

    def arc(a: Any, b: Any, bulge: float, layer: str) -> None:
        (cx, cy), r, a0, theta = arc_of((float(a[0]), float(a[1])), (float(b[0]), float(b[1])), bulge)
        start, end = (a0, a0 + theta) if theta > 0 else (a0 + theta, a0)
        msp.add_arc((cx * k, cy * k), r * k, math.degrees(start), math.degrees(end), dxfattribs={"layer": layer})

    geo = project.get("geometry") or {}
    dom = geo.get("domain") or {}
    if dom.get("polygon"):
        if DOMAIN_LAYER not in doc.layers:
            doc.layers.add(DOMAIN_LAYER, color=8)
        closed_path(dom["polygon"], dom.get("bulges"), DOMAIN_LAYER)
    for r in geo.get("regions") or []:
        layer = layer_of(r)
        shape = r.get("shape")
        if shape:
            msp.add_circle(p(shape["center"]), float(shape["radius"]) * k, dxfattribs={"layer": layer})
            continue
        if not r.get("polygon"):
            continue
        closed_path(r["polygon"], r.get("bulges"), layer)
        for h in r.get("holes") or []:
            closed_path(h["polygon"], h.get("bulges"), layer)
    for e in cad.get("sketch") or []:
        if not isinstance(e, dict):
            continue
        layer = layer_of(e)
        kind = e.get("kind")
        if kind == "line":
            msp.add_line(p(e["a"]), p(e["b"]), dxfattribs={"layer": layer})
        elif kind == "arc":
            arc(e["a"], e["b"], float(e.get("bulge") or 0.0), layer) if e.get("bulge") else msp.add_line(p(e["a"]), p(e["b"]), dxfattribs={"layer": layer})
        elif kind == "circle":
            msp.add_circle(p(e["center"]), float(e["r"]) * k, dxfattribs={"layer": layer})
        elif kind == "polyline":
            pts = e.get("points") or []
            bs = e.get("bulges")
            vals = [(*p(pts[i]), bulge_of(bs, i)) for i in range(len(pts))]
            msp.add_lwpolyline(vals, format="xyb", close=bool(e.get("closed")), dxfattribs={"layer": layer})
    s = io.StringIO()
    doc.write(s)
    return s.getvalue()
