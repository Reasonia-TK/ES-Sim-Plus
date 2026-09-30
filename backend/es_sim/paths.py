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

import numpy as np

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


def _extent(polygon: Sequence[Pt]) -> float:
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    return max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0


def _fine_points(polygon: Sequence[Pt], bulges: Sequence[float] | None, h: float) -> list[Pt]:
    if has_arcs(bulges):
        return flatten_path(polygon, bulges, h)[0]
    return [(float(p[0]), float(p[1])) for p in polygon]


def point_in_path(q: Pt, polygon: Sequence[Pt], bulges: Sequence[float] | None) -> bool:
    """点が閉じた経路の内側か (円弧は経路の大きさの 1/512 の幅で弦に分けて偶奇規則で判定。境界上は不定)。"""
    pts = _fine_points(polygon, bulges, _extent(polygon) / 512.0)
    x, y = float(q[0]), float(q[1])
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])


def _point_segment_d2(p: np.ndarray, s0: np.ndarray, s1: np.ndarray) -> np.ndarray:
    e = s1 - s0
    l2 = np.sum(e * e, axis=-1)
    t = np.clip(np.sum((p - s0) * e, axis=-1) / np.where(l2 > 0.0, l2, 1.0), 0.0, 1.0)
    d = p - (s0 + t[..., None] * e)
    return np.sum(d * d, axis=-1)


def rings_touch(a: Sequence[Pt], b: Sequence[Pt], tol: float) -> bool:
    """2 つの閉じた折れ線 (頂点列) の辺どうしが交わるか、tol 以内に近づくか。"""
    a0 = np.asarray(a, dtype=np.float64)
    b0 = np.asarray(b, dtype=np.float64)
    a1 = np.roll(a0, -1, axis=0)
    b1 = np.roll(b0, -1, axis=0)
    B0, B1 = b0[None, :, :], b1[None, :, :]
    step = max(1, 1_000_000 // max(1, len(b0)))  # 1 度に作る表を 100 万組ほどに抑える
    for i in range(0, len(a0), step):
        A0, A1 = a0[i : i + step, None, :], a1[i : i + step, None, :]
        d1, d2 = _cross(B0, B1, A0), _cross(B0, B1, A1)
        d3, d4 = _cross(A0, A1, B0), _cross(A0, A1, B1)
        if np.any((d1 * d2 < 0.0) & (d3 * d4 < 0.0)):
            return True
        near = np.minimum(
            np.minimum(_point_segment_d2(A0, B0, B1), _point_segment_d2(A1, B0, B1)),
            np.minimum(_point_segment_d2(B0, A0, A1), _point_segment_d2(B1, A0, A1)),
        )
        if np.any(near <= tol * tol):
            return True
    return False


def check_holes(
    polygon: Sequence[Pt], bulges: Sequence[float] | None, holes: Sequence[tuple[Sequence[Pt], Sequence[float] | None]], what: str
) -> None:
    """穴の検査 (ValueError): 外周の内側にあり、外周・ほかの穴と交わらず接せず、ほかの穴の中にない。

    円弧は外周の大きさの 1/512 の幅で弦に分けて調べる。
    """
    if not holes:
        return
    extent = _extent(polygon)
    h = extent / 512.0
    tol = 1e-9 * extent
    rings = [_fine_points(polygon, bulges, h), *[_fine_points(hp, hb, h) for hp, hb in holes]]
    for k, (hp, _) in enumerate(holes):
        if not all(point_in_path(q, polygon, bulges) for q in hp):
            raise ValueError(f"{what}: 穴 {k + 1} が外周の外にはみ出しています")
        if rings_touch(rings[0], rings[k + 1], tol):
            raise ValueError(f"{what}: 穴 {k + 1} が外周と交わるか接しています")
    for j in range(len(holes)):
        for k in range(j + 1, len(holes)):
            if rings_touch(rings[j + 1], rings[k + 1], tol):
                raise ValueError(f"{what}: 穴 {j + 1} と穴 {k + 1} が交わるか接しています")
            if point_in_path(holes[j][0][0], holes[k][0], holes[k][1]) or point_in_path(holes[k][0][0], holes[j][0], holes[j][1]):
                raise ValueError(f"{what}: 穴 {j + 1} と穴 {k + 1} の一方がもう一方の中にあります")
