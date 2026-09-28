"""v2 GPU PIC の粒子用幾何 — セル状態・固体形状・吸収位置と法線・表示用三角形分割 (prompts/119)。

- セル状態 (ny, nx) uint8: 0 = 気体のみ、1 = 固体のみ (導体・誘電体)、2 = 境界が通る
  (カット)。粒子の吸収判定はカットセルでだけ形状との厳密な内包判定を GPU で行う。
- 固体形状は GPU カーネル用に平坦化 (多角形の頂点列 + オフセット、円 (cx, cy, r))。
- 吸収された粒子 (1 ステップあたり全体のごく一部) の衝突点・法線・衝突相手は
  ホスト (numpy) で厳密に求める (線分と多角形辺/円の最初の交点)。
- v1 UI 互換の表示用メッシュ: 直交格子のセルを市松の対角線で 2 三角形に分割し、
  導体内のセルを除く (v1 の構造格子と同じ規約)。節点 = 格子節点 (全点)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..eb.build import _mark_boundary_cells
from ..eb.grid import CartesianGrid
from ..geom.model import GeometryModel
from ..geom.shapes import CircleShape, PolygonShape

# 衝突相手の種類
HIT_SIDE = 0       # domain 外周
HIT_CONDUCTOR = 1  # 導体 (index = 導体番号)
HIT_DIELECTRIC = 2  # 誘電体 (index = model.others の番号)

# 外周の辺番号 (boundary カーネルの status 1..4 と対応)
SIDE_OF_STATUS = {1: "left", 2: "right", 3: "bottom", 4: "top"}
SIDE_INWARD_NORMAL = {"left": (1.0, 0.0), "right": (-1.0, 0.0), "bottom": (0.0, 1.0), "top": (0.0, -1.0)}


@dataclass
class _SolidShape:
    kind: int          # HIT_CONDUCTOR / HIT_DIELECTRIC
    index: int         # 導体番号 or others 番号
    shape: PolygonShape | CircleShape


@dataclass
class DisplayMesh:
    """v1 UI 互換の表示用三角形メッシュ (server が started メッセージで送る)。"""

    nodes: np.ndarray            # (N, 2)
    triangles: np.ndarray        # (M, 3)
    tri_region: np.ndarray       # (M,) geometry.regions の番号 (-1 = 真空)
    tri_cell: np.ndarray         # (M,) 平坦セル番号 j*nx + i


class ParticleGeometry:
    """粒子の境界処理に使う幾何 (ホスト側の定義。GPU 配列は simulation が転送する)。"""

    def __init__(self, model: GeometryModel, grid: CartesianGrid):
        self.model = model
        self.grid = grid
        solids: list[_SolidShape] = []
        for k, c in enumerate(model.conductors):
            solids.append(_SolidShape(HIT_CONDUCTOR, k, c.shape))
        for k, o in enumerate(model.others):
            if o.type == "dielectric":
                solids.append(_SolidShape(HIT_DIELECTRIC, k, o.shape))
        self.solids = solids

        # ---- セル状態 ---------------------------------------------------------------
        nx, ny = grid.nx, grid.ny
        xc = grid.xs[:-1] + 0.5 * grid.dx
        yc = grid.ys[:-1] + 0.5 * grid.dy
        XC, YC = np.meshgrid(xc, yc)
        gas_center = model.gas_at(XC, YC)
        cut = _mark_boundary_cells([s.shape for s in solids], grid) if solids else np.zeros((ny, nx), dtype=bool)
        state = np.where(gas_center, 0, 1).astype(np.uint8)
        state[cut] = 2
        self.cell_state = state

        # ---- GPU 用の形状表 ----------------------------------------------------------
        polys = [s.shape for s in solids if isinstance(s.shape, PolygonShape)]
        circs = [s.shape for s in solids if isinstance(s.shape, CircleShape)]
        off = [0]
        xy = []
        for p in polys:
            xy.append(p.vertices.ravel())
            off.append(off[-1] + len(p.vertices))
        self.poly_off = np.asarray(off, dtype=np.int32)
        self.poly_xy = np.concatenate(xy) if xy else np.zeros(2)
        self.n_poly = len(polys)
        self.circ = np.asarray([[c.cx, c.cy, c.r] for c in circs], dtype=np.float64).ravel() if circs else np.zeros(3)
        self.n_circ = len(circs)

    # ---- 吸収粒子の衝突点 -----------------------------------------------------------

    def solid_hits(self, p0: np.ndarray, p1: np.ndarray):
        """線分 p0→p1 (各行 1 粒子) が最初に入る固体の (t, 衝突点, 内向き法線, 種類, 番号)。

        法線は粒子の来た側 (気体側) を向く単位ベクトル。交点が見つからない (数値的に
        既に内部から出発した等) 粒子は t=1 (現在位置)、法線 = -進行方向。
        """
        n = len(p0)
        d = p1 - p0
        best_t = np.full(n, np.inf)
        best_n = np.zeros((n, 2))
        best_kind = np.full(n, -1, dtype=np.int64)
        best_idx = np.full(n, -1, dtype=np.int64)
        for s in self.solids:
            if isinstance(s.shape, CircleShape):
                c = np.array([s.shape.cx, s.shape.cy])
                f = p0 - c
                a = np.sum(d * d, axis=1)
                b = 2.0 * np.sum(f * d, axis=1)
                cc = np.sum(f * f, axis=1) - s.shape.r ** 2
                disc = b * b - 4.0 * a * cc
                ok = (disc >= 0.0) & (a > 0.0)
                t = np.full(n, np.inf)
                sq = np.sqrt(np.where(ok, disc, 0.0))
                t0 = np.where(ok, (-b - sq) / np.where(a > 0, 2.0 * a, 1.0), np.inf)
                t = np.where((t0 >= 0.0) & (t0 <= 1.0), t0, np.inf)
                upd = t < best_t
                if np.any(upd):
                    hit = p0[upd] + t[upd, None] * d[upd]
                    nrm = hit - c
                    nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-300)[:, None]
                    best_t[upd] = t[upd]
                    best_n[upd] = nrm
                    best_kind[upd] = s.kind
                    best_idx[upd] = s.index
            else:
                v = s.shape.vertices
                q0 = v
                q1 = np.roll(v, -1, axis=0)
                e = q1 - q0                                          # (m, 2)
                # p0 + t d = q0 + u e を全粒子×全辺で解く
                den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]   # (n, m)
                w = q0[None, :, :] - p0[:, None, :]                                  # (n, m, 2)
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = (w[:, :, 0] * e[None, :, 1] - w[:, :, 1] * e[None, :, 0]) / den
                    u = (w[:, :, 0] * d[:, None, 1] - w[:, :, 1] * d[:, None, 0]) / den
                ok = (np.abs(den) > 0.0) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
                t = np.where(ok, t, np.inf)
                j = np.argmin(t, axis=1)
                tmin = t[np.arange(n), j]
                upd = tmin < best_t
                if np.any(upd):
                    ee = e[j[upd]]
                    nrm = np.stack([ee[:, 1], -ee[:, 0]], axis=1)
                    nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-300)[:, None]
                    best_t[upd] = tmin[upd]
                    best_n[upd] = nrm
                    best_kind[upd] = s.kind
                    best_idx[upd] = s.index
        none = ~np.isfinite(best_t)
        if np.any(none):
            best_t[none] = 1.0
            dn = -d[none]
            best_n[none] = dn / np.maximum(np.linalg.norm(dn, axis=1), 1e-300)[:, None]
        # 法線を気体側 (進行方向と逆) へ向ける
        flip = np.sum(best_n * d, axis=1) > 0.0
        best_n[flip] *= -1.0
        hit = p0 + best_t[:, None] * d
        return best_t, hit, best_n, best_kind, best_idx

    # ---- 表示用メッシュ ----------------------------------------------------------------

    def display_mesh(self) -> DisplayMesh:
        g = self.grid
        nx, ny = g.nx, g.ny
        X, Y = np.meshgrid(g.xs, g.ys)
        nodes = np.stack([X.ravel(), Y.ravel()], axis=1)
        xc = g.xs[:-1] + 0.5 * g.dx
        yc = g.ys[:-1] + 0.5 * g.dy
        XC, YC = np.meshgrid(xc, yc)
        keep = self.model.classify_conductor(XC, YC, 0.0) < 0     # 導体内のセルは穴 (v1 と同じ)
        J, I = np.nonzero(keep)
        a = J * (nx + 1) + I
        b = a + 1
        c = b + (nx + 1)
        d = a + (nx + 1)
        even = ((I + J) % 2 == 0)[:, None]
        t1 = np.where(even, np.stack([a, b, c], axis=1), np.stack([a, b, d], axis=1))
        t2 = np.where(even, np.stack([a, c, d], axis=1), np.stack([b, c, d], axis=1))
        tris = np.empty((2 * len(a), 3), dtype=np.int64)
        tris[0::2] = t1
        tris[1::2] = t2
        cell = np.repeat(J * nx + I, 2)
        cent = nodes[tris].mean(axis=1)
        other = self.model.classify_other(cent[:, 0], cent[:, 1])
        region_idx = np.array([o.index for o in self.model.others] + [-1], dtype=np.int64)
        tri_region = np.where(other >= 0, region_idx[np.maximum(other, 0)], -1)
        return DisplayMesh(nodes=nodes, triangles=tris, tri_region=tri_region, tri_cell=cell)


def triangle_gradients(nodes: np.ndarray, tris: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """三角形ごとの線形補間の勾配 ∇φ (M, 2)。phi は (N,) または (B, N)。"""
    p = nodes[tris]
    x, y = p[:, :, 0], p[:, :, 1]
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)
    det = x[:, 0] * b[:, 0] + x[:, 1] * b[:, 1] + x[:, 2] * b[:, 2]
    vt = phi[..., tris]                          # (..., M, 3)
    gx = np.sum(vt * b, axis=-1) / det
    gy = np.sum(vt * c, axis=-1) / det
    return np.stack([gx, gy], axis=-1)
