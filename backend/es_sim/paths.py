"""閉じた経路 (頂点 + 辺ごとの bulge) の計算 — CAD v2 (prompts/132)。

経路は ``polygon`` (頂点列、閉じは暗黙) と ``bulges`` (辺ごと、0 か None は直線) で表す。辺 i は頂点 i → i+1
(最後は頂点 0 へ)。円弧の bulge は tan(θ/4) (θ は中心角、正は反時計回り) で、DXF の LWPOLYLINE と同じ。

ソルバーは直線だけの多角形を前提にしているので、``Project`` の検証の段階で円弧をメッシュ幅に合わせた弦に分ける
(``flatten_path``)。分ける密度は円の多角形化 (meshing._circle_polygon) と同じ: 1 周 ceil(2πr/h) 本を
[CIRCLE_SEGMENTS_MIN, CIRCLE_SEGMENTS_MAX] に収め、円弧は中心角の割合 (切り上げ)。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# 円の多角形化の分割数の下限・上限 (meshing.CIRCLE_SEGMENTS_MIN / MAX と同じ値。meshing は gmsh を読み込むので
# スキーマから参照できるよう、ここに置いて meshing が使う)
CIRCLE_SEGMENTS_MIN = 24
CIRCLE_SEGMENTS_MAX = 720

# bulge を 0 (直線) とみなす大きさ
BULGE_EPS = 1e-12

Pt = tuple[float, float]


def bulge_of(bulges: Sequence[float] | None, i: int) -> float:
    if bulges is None:
        return 0.0
    b = float(bulges[i])
    return b if abs(b) > BULGE_EPS else 0.0


def has_arcs(bulges: Sequence[float] | None) -> bool:
    return bulges is not None and any(abs(float(b)) > BULGE_EPS for b in bulges)


def arc_of(p: Pt, q: Pt, bulge: float) -> tuple[Pt, float, float, float]:
    """bulge の円弧 p → q の (中心, 半径, 始点の角度, 符号付きの中心角)。"""
    theta = 4.0 * math.atan(bulge)
    dx, dy = q[0] - p[0], q[1] - p[1]
    chord = math.hypot(dx, dy)
    r = abs(chord / (2.0 * math.sin(theta / 2.0)))
    # 弦の中点から中心へ: 弦の左の法線方向に (chord/2)/tan(θ/2) (θ > π や時計回りでは負 = 右側)
    h = (chord / 2.0) / math.tan(theta / 2.0)
    cx = (p[0] + q[0]) / 2.0 - dy / chord * h
    cy = (p[1] + q[1]) / 2.0 + dx / chord * h
    a0 = math.atan2(p[1] - cy, p[0] - cx)
    return (cx, cy), r, a0, theta


def arc_segment_count(radius: float, sweep: float, h: float) -> int:
    """中心角 |sweep| の円弧を分ける弦の本数 (円の多角形化と同じ密度)。"""
    n_full = math.ceil(2.0 * math.pi * radius / h)
    n_full = max(CIRCLE_SEGMENTS_MIN, min(CIRCLE_SEGMENTS_MAX, n_full))
    return max(1, math.ceil(n_full * abs(sweep) / (2.0 * math.pi) - 1e-9))


def flatten_path(polygon: Sequence[Pt], bulges: Sequence[float] | None, h: float) -> tuple[list[Pt], list[int]]:
    """円弧を弦に分けた頂点列と、分けた後の辺 k が元のどの辺か (origin[k])。

    元の頂点はそのまま残り (円弧の端点は元の頂点と一致する)、円弧の途中の点だけを足す。
    """
    n = len(polygon)
    pts: list[Pt] = []
    origin: list[int] = []
    for i in range(n):
        p = (float(polygon[i][0]), float(polygon[i][1]))
        q = (float(polygon[(i + 1) % n][0]), float(polygon[(i + 1) % n][1]))
        pts.append(p)
        origin.append(i)
        b = bulge_of(bulges, i)
        if b == 0.0:
            continue
        (cx, cy), r, a0, theta = arc_of(p, q, b)
        m = arc_segment_count(r, theta, h)
        for j in range(1, m):
            a = a0 + theta * j / m
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
            origin.append(i)
    return pts, origin


def path_area(polygon: Sequence[Pt], bulges: Sequence[float] | None) -> float:
    """符号付き面積 (反時計回りが正、円弧の弓形 r²(θ - sin θ)/2 を含める)。"""
    n = len(polygon)
    s = 0.0
    for i in range(n):
        p = polygon[i]
        q = polygon[(i + 1) % n]
        s += (p[0] * q[1] - q[0] * p[1]) / 2.0
        b = bulge_of(bulges, i)
        if b != 0.0:
            _, r, _, theta = arc_of((p[0], p[1]), (q[0], q[1]), b)
            s += r * r * (theta - math.sin(theta)) / 2.0
    return s


def check_path(polygon: Sequence[Pt], bulges: Sequence[float] | None, what: str) -> None:
    """経路として使えるかの検査 (ValueError)。

    - bulges は頂点と同じ数・有限
    - 直線だけなら 3 頂点以上、円弧を含めば 2 頂点以上
    - 円弧の両端は別の点、面積が 0 でない
    """
    n = len(polygon)
    if bulges is not None:
        if len(bulges) != n:
            raise ValueError(f"{what}: bulges は頂点と同じ数 ({n}) が必要です (今は {len(bulges)})")
        if any(not math.isfinite(float(b)) for b in bulges):
            raise ValueError(f"{what}: bulges に有限でない値があります")
    arcs = has_arcs(bulges)
    if n < 3 and not arcs:
        raise ValueError(f"{what}: 直線だけの経路は 3 頂点以上が必要です")
    if n < 2:
        raise ValueError(f"{what}: 経路は 2 頂点以上が必要です")
    for i in range(n):
        if bulge_of(bulges, i) != 0.0:
            p, q = polygon[i], polygon[(i + 1) % n]
            if p[0] == q[0] and p[1] == q[1]:
                raise ValueError(f"{what}: 円弧の辺 {i} の両端が同じ点です")
    if arcs and path_area(polygon, bulges) == 0.0:
        raise ValueError(f"{what}: 経路の面積が 0 です")
