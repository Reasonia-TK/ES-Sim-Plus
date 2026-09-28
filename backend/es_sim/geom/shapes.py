"""2D 形状 (多角形・円) の解析的な問い合わせ — v2 直交格子エンジン用 (prompts/119)。

埋め込み境界 (EB) の構築に必要な 3 種類の問い合わせだけを、全点・全線分一括の
numpy ベクトル演算で提供する:

- ``contains(x, y, tol)``: 点の内包判定 (境界から tol 以内も「内側」とみなす。v1 の
  構造格子 meshing._points_in_region と同じ規約 — 格子線上に乗った導体の辺の節点は
  導体側 (Dirichlet) に分類される)。
- ``crossings_h(y0, xa, xb)`` / ``crossings_v(x0, ya, yb)``: 軸平行な線分群と境界の
  交点。格子の辺・誘電体チューブの標本線はすべて軸平行なので、一般の線分交差は不要。
  戻り値は (線分番号の配列, 交点座標の配列) の平坦な組 (線分あたり交点数は可変)。
- ``boundary_points(spacing)``: 境界上の標本点 (セルの「境界が通るか」判定用)。

多角形の交差判定は点内包判定 (ray casting) と同じ半開区間規則
``(py > y0) != (qy > y0)`` を使う。これにより頂点をかすめる格子線でも交点数の偶奇が
内包判定と矛盾しない。円は厳密な二次方程式の解 (多角形近似しない — v1 の gmsh 経路が
円を多角形化していたのに対し、EB では曲率を保ったまま扱える)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _empty_hits() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)


@dataclass(frozen=True)
class PolygonShape:
    """単純多角形 (自己交差なし、向きは問わない)。vertices は (n, 2)、閉じは暗黙。"""

    vertices: np.ndarray

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices, dtype=np.float64)
        if v.ndim != 2 or v.shape[1] != 2 or len(v) < 3:
            raise ValueError("PolygonShape には 3 頂点以上の (n, 2) 配列が必要です")
        object.__setattr__(self, "vertices", v)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        v = self.vertices
        return float(v[:, 0].min()), float(v[:, 1].min()), float(v[:, 0].max()), float(v[:, 1].max())

    def _edges(self) -> tuple[np.ndarray, np.ndarray]:
        p = self.vertices
        q = np.roll(p, -1, axis=0)
        return p, q

    def contains(self, x: np.ndarray, y: np.ndarray, tol: float = 0.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        shape = np.broadcast(x, y).shape
        xf = np.broadcast_to(x, shape).ravel()
        yf = np.broadcast_to(y, shape).ravel()
        inside = np.zeros(xf.shape, dtype=bool)
        xmin, ymin, xmax, ymax = self.bbox
        cand = np.nonzero(
            (xf >= xmin - tol) & (xf <= xmax + tol) & (yf >= ymin - tol) & (yf <= ymax + tol)
        )[0]
        if cand.size == 0:
            return inside.reshape(shape)
        px_all, qx_all = self._edges()
        # 候補点を y でソートし、各辺はその y 範囲 (±tol) に入る点だけを調べる
        # (頂点数 n・点数 N に対し O(N log N + Σ辺ごとの該当点数)。円を細かく多角形化した
        # 形状でも全点×全辺の O(N·n) にならない)
        order = np.argsort(yf[cand], kind="stable")
        cand = cand[order]
        cx, cy = xf[cand], yf[cand]
        odd = np.zeros(cand.size, dtype=bool)
        on = np.zeros(cand.size, dtype=bool)
        for (px, py), (qx, qy) in zip(px_all, qx_all):
            vlo, vhi = (py, qy) if py < qy else (qy, py)
            lo = int(np.searchsorted(cy, vlo - tol, side="left"))
            hi = int(np.searchsorted(cy, vhi + tol, side="right"))
            if hi <= lo:
                continue
            sy = cy[lo:hi]
            sx = cx[lo:hi]
            # 半開区間規則の ray casting (+x 方向へのレイ)
            straddle = (py > sy) != (qy > sy)
            if np.any(straddle):
                idx = np.nonzero(straddle)[0]
                xc = px + (sy[idx] - py) * (qx - px) / (qy - py)
                hit = idx[sx[idx] < xc]
                odd[lo + hit] ^= True
            if tol > 0.0:
                # 線分までの距離 ≤ tol なら境界上 (内側扱い)
                ex, ey = qx - px, qy - py
                l2 = ex * ex + ey * ey
                if l2 > 0.0:
                    t = np.clip(((sx - px) * ex + (sy - py) * ey) / l2, 0.0, 1.0)
                    dx = sx - (px + t * ex)
                    dy = sy - (py + t * ey)
                    on[lo:hi] |= dx * dx + dy * dy <= tol * tol
        inside[cand] = odd | on
        return inside.reshape(shape)

    def crossings_h(self, y0: np.ndarray, xa: np.ndarray, xb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """水平線分 {y=y0[k], xa[k] < x < xb[k]} と境界の交点 (k, x)。xa < xb を仮定。"""
        return _polygon_axis_crossings(self.vertices, y0, xa, xb, axis=0)

    def crossings_v(self, x0: np.ndarray, ya: np.ndarray, yb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """鉛直線分 {x=x0[k], ya[k] < y < yb[k]} と境界の交点 (k, y)。"""
        return _polygon_axis_crossings(self.vertices[:, ::-1], x0, ya, yb, axis=0)

    def boundary_points(self, spacing: float) -> tuple[np.ndarray, np.ndarray]:
        p, q = self._edges()
        xs, ys = [], []
        for (px, py), (qx, qy) in zip(p, q):
            length = float(np.hypot(qx - px, qy - py))
            n = max(1, int(np.ceil(length / spacing)))
            t = np.arange(n + 1) / n
            xs.append(px + t * (qx - px))
            ys.append(py + t * (qy - py))
        return np.concatenate(xs), np.concatenate(ys)


def _polygon_axis_crossings(
    verts: np.ndarray, c0: np.ndarray, a_lo: np.ndarray, a_hi: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    """``verts`` の列 (u, v) について、線分 {v = c0[k], a_lo[k] < u < a_hi[k]} との交点。

    crossings_h は (u, v) = (x, y)、crossings_v は列を入れ替えて (u, v) = (y, x) で呼ぶ。
    """
    del axis
    c0 = np.asarray(c0, dtype=np.float64).ravel()
    a_lo = np.asarray(a_lo, dtype=np.float64).ravel()
    a_hi = np.asarray(a_hi, dtype=np.float64).ravel()
    if c0.size == 0:
        return _empty_hits()
    # 問い合わせ線分を v (= c0) でソートし、多角形の各辺はその v 範囲に入る線分だけ調べる
    # (頂点数の多い多角形 × 全格子辺の総当たりを避ける)
    order = np.argsort(c0, kind="stable")
    c0s = c0[order]
    p = verts
    q = np.roll(verts, -1, axis=0)
    ks, us = [], []
    for (pu, pv), (qu, qv) in zip(p, q):
        vlo, vhi = (pv, qv) if pv < qv else (qv, pv)
        ulo, uhi = (pu, qu) if pu < qu else (qu, pu)
        lo = int(np.searchsorted(c0s, vlo, side="left"))
        hi = int(np.searchsorted(c0s, vhi, side="right"))
        if hi <= lo:
            continue
        idx = order[lo:hi]
        # u 範囲でさらに絞る (bbox 前処理)
        idx = idx[(a_hi[idx] >= ulo) & (a_lo[idx] <= uhi)]
        if idx.size == 0:
            continue
        cv = c0[idx]
        straddle = (pv > cv) != (qv > cv)
        if not np.any(straddle):
            continue
        idx = idx[straddle]
        cv = cv[straddle]
        uc = pu + (cv - pv) * (qu - pu) / (qv - pv)
        ok = (uc > a_lo[idx]) & (uc < a_hi[idx])
        ks.append(idx[ok])
        us.append(uc[ok])
    if not ks:
        return _empty_hits()
    return np.concatenate(ks).astype(np.int64), np.concatenate(us)


@dataclass(frozen=True)
class CircleShape:
    """円 (中心 (cx, cy)、半径 r)。"""

    cx: float
    cy: float
    r: float

    def __post_init__(self) -> None:
        if not (self.r > 0.0):
            raise ValueError("CircleShape の半径は正である必要があります")

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.cx - self.r, self.cy - self.r, self.cx + self.r, self.cy + self.r

    def contains(self, x: np.ndarray, y: np.ndarray, tol: float = 0.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return np.hypot(x - self.cx, y - self.cy) <= self.r + tol

    def crossings_h(self, y0: np.ndarray, xa: np.ndarray, xb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _circle_axis_crossings(self.cx, self.cy, self.r, y0, xa, xb)

    def crossings_v(self, x0: np.ndarray, ya: np.ndarray, yb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _circle_axis_crossings(self.cy, self.cx, self.r, x0, ya, yb)

    def boundary_points(self, spacing: float) -> tuple[np.ndarray, np.ndarray]:
        n = max(16, int(np.ceil(2.0 * np.pi * self.r / spacing)))
        th = 2.0 * np.pi * np.arange(n) / n
        return self.cx + self.r * np.cos(th), self.cy + self.r * np.sin(th)


def _circle_axis_crossings(
    cu: float, cv: float, r: float, c0: np.ndarray, a_lo: np.ndarray, a_hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    c0 = np.asarray(c0, dtype=np.float64).ravel()
    a_lo = np.asarray(a_lo, dtype=np.float64).ravel()
    a_hi = np.asarray(a_hi, dtype=np.float64).ravel()
    d = c0 - cv
    disc = r * r - d * d
    sel = np.nonzero(disc > 0.0)[0]
    if sel.size == 0:
        return _empty_hits()
    s = np.sqrt(disc[sel])
    ks, us = [], []
    for sign in (-1.0, 1.0):
        u = cu + sign * s
        ok = (u > a_lo[sel]) & (u < a_hi[sel])
        ks.append(sel[ok])
        us.append(u[ok])
    return np.concatenate(ks).astype(np.int64), np.concatenate(us)


Shape = PolygonShape | CircleShape
