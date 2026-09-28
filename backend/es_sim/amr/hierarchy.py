"""AMReX 型のブロック構造 AMR 階層 (prompts/121)。

- レベル l の仮想格子は domain 全体を (nx0·2^l) × (ny0·2^l) セルで覆う (格子幅 h/2^l)。
  実際に使うのは各レベルの「領域」だけで、領域は bf × bf セルのブロックの和集合
  (bf = blocking_factor。AMReX と同じく細分化の単位を揃える)。
- ``refined[l]`` はレベル l のブロックのうち、より細かいレベル (l+1) に覆われるもの。
  レベル l+1 の領域 = refined[l] の各ブロックを 2×2 に分割したもの。
- **proper nesting**: 細分化されたレベル l+1 のブロックの 8 近傍はすべてレベル l+1 の領域内
  (domain 外周では打ち切り、周期方向は巻き戻し)。これで、レベル l+2 のセルとレベル l の
  セルの間に必ずレベル l+1 のブロック 1 個分の緩衝帯が入り、隣接する葉セルのレベル差は
  高々 1 になる (2:1 バランス。合成格子のぶら下がり節点が 1 段で済む)。
- 葉セル: そのレベルの領域内で、より細かいレベルに覆われていないセル。全レベルの葉セルが
  domain を重なりなく覆う。

タグ付け (細分化したいセル):
- refine_boundaries: 導体・誘電体の境界が通るセルから buffer_cells セル以内 (各レベルの
  セル単位)。境界近傍は max_level まで細分化される。
- regions: 矩形 (対角 2 点) と目標レベル。その矩形に重なるセルを目標レベルまで細分化する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..eb.build import _mark_boundary_cells
from ..eb.grid import CartesianGrid
from ..geom.model import GeometryModel


def _pow2_divisor(n: int, cap: int) -> int:
    """n を割り切る 2 のべき (cap 以下) の最大値。"""
    b = 1
    while b * 2 <= cap and n % (b * 2) == 0:
        b *= 2
    return b


def _shift(a: np.ndarray, s: int, axis: int, periodic: bool) -> np.ndarray:
    if periodic:
        return np.roll(a, s, axis=axis)
    out = np.zeros_like(a)
    n = a.shape[axis]
    if abs(s) >= n:
        return out
    src = [slice(None)] * a.ndim
    dst = [slice(None)] * a.ndim
    if s > 0:
        src[axis] = slice(0, n - s)
        dst[axis] = slice(s, n)
    else:
        src[axis] = slice(-s, n)
        dst[axis] = slice(0, n + s)
    out[tuple(dst)] = a[tuple(src)]
    return out


def dilate(mask: np.ndarray, r: int, px: bool = False, py: bool = False) -> np.ndarray:
    """正方形の構造要素 (半径 r) による膨張。周期方向は巻き戻す。"""
    if r <= 0:
        return mask.copy()
    out = mask.copy()
    for s in range(1, r + 1):
        out |= _shift(mask, s, 1, px) | _shift(mask, -s, 1, px)
    row = out.copy()
    for s in range(1, r + 1):
        out |= _shift(row, s, 0, py) | _shift(row, -s, 0, py)
    return out


def _upsample(blocks: np.ndarray, k: int) -> np.ndarray:
    """ブロック/セルのマスクを各方向 k 倍に複製する。"""
    return np.repeat(np.repeat(blocks, k, axis=0), k, axis=1)


def _block_any(cells: np.ndarray, bf: int) -> np.ndarray:
    ny, nx = cells.shape
    return cells.reshape(ny // bf, bf, nx // bf, bf).any(axis=(1, 3))


@dataclass(frozen=True)
class AmrSpec:
    max_level: int = 0
    refine_boundaries: bool = True
    buffer_cells: int = 2
    blocking_factor: int = 8
    regions: tuple = ()      # ((x0, y0, x1, y1, level), ...)

    @classmethod
    def from_settings(cls, amr) -> "AmrSpec":
        if amr is None:
            return cls()
        regs = []
        for r in amr.regions:
            x0, x1 = sorted((float(r.p1[0]), float(r.p2[0])))
            y0, y1 = sorted((float(r.p1[1]), float(r.p2[1])))
            regs.append((x0, y0, x1, y1, int(r.level)))
        return cls(
            max_level=int(amr.max_level),
            refine_boundaries=bool(amr.refine_boundaries),
            buffer_cells=int(amr.buffer_cells),
            blocking_factor=int(amr.blocking_factor),
            regions=tuple(regs),
        )


class AmrHierarchy:
    """ブロック構造の細分化階層。レベル 0 は base (domain 全体の一様格子)。"""

    def __init__(self, model: GeometryModel, base: CartesianGrid, spec: AmrSpec):
        self.model = model
        self.base = base
        self.spec = spec
        px, py = model.periodic_x, model.periodic_y
        self.px, self.py = px, py
        # regions の目標レベルが max_level を超える場合は max_level を引き上げる
        L = max([spec.max_level] + [r[4] for r in spec.regions]) if spec.regions else spec.max_level
        self.max_level = L
        self.bf = _pow2_divisor(math.gcd(base.nx, base.ny), max(1, spec.blocking_factor))
        bf = self.bf

        # ---- 1. タグ → 細分化ブロック (粗いレベルから順に) ------------------------------
        refined: list[np.ndarray] = []
        region_blocks = np.ones((base.ny // bf, base.nx // bf), dtype=bool)
        shapes = [c.shape for c in model.conductors] + [o.shape for o in model.others]
        for lvl in range(L):
            g = self.level_grid(lvl)
            region_cells = _upsample(region_blocks, bf)
            tags = np.zeros((g.ny, g.nx), dtype=bool)
            if spec.refine_boundaries and lvl < spec.max_level and shapes:
                tags |= dilate(_mark_boundary_cells(shapes, g), spec.buffer_cells, px, py)
            for (x0, y0, x1, y1, target) in spec.regions:
                if lvl < target:
                    i0 = max(0, int(math.floor((x0 - g.x0) / g.dx)))
                    i1 = min(g.nx, int(math.ceil((x1 - g.x0) / g.dx)))
                    j0 = max(0, int(math.floor((y0 - g.y0) / g.dy)))
                    j1 = min(g.ny, int(math.ceil((y1 - g.y0) / g.dy)))
                    if i1 > i0 and j1 > j0:
                        tags[j0:j1, i0:i1] = True
            tags &= region_cells
            b = _block_any(tags, bf) & region_blocks
            refined.append(b)
            region_blocks = _upsample(b, 2)

        # ---- 2. proper nesting (細かいレベルから粗いレベルへ領域を広げる) ----------------
        for lvl in range(L - 2, -1, -1):
            need = dilate(refined[lvl + 1], 1, px, py)
            ny_b, nx_b = need.shape
            parent = need.reshape(ny_b // 2, 2, nx_b // 2, 2).any(axis=(1, 3))
            refined[lvl] = refined[lvl] | parent
        # 領域の確定 (レベル 0 は全体)
        self.refined = refined
        self.region_blocks = [np.ones((base.ny // bf, base.nx // bf), dtype=bool)]
        for lvl in range(L):
            self.region_blocks.append(_upsample(refined[lvl], 2))
        # 細分化が空になったレベルより上は捨てる
        top = 0
        for lvl in range(L):
            if refined[lvl].any():
                top = lvl + 1
        self.max_level = top
        self.refined = self.refined[:top]
        self.region_blocks = self.region_blocks[: top + 1]

    # ---- 格子・セル ------------------------------------------------------------------

    @property
    def n_levels(self) -> int:
        return self.max_level + 1

    def level_grid(self, lvl: int) -> CartesianGrid:
        b = self.base
        return CartesianGrid(b.x0, b.y0, b.x1, b.y1, b.nx << lvl, b.ny << lvl)

    def region_cells(self, lvl: int) -> np.ndarray:
        return _upsample(self.region_blocks[lvl], self.bf)

    def leaf_mask(self, lvl: int) -> np.ndarray:
        """レベル lvl の葉セル (領域内かつ細かいレベルに覆われない) の (ny_l, nx_l) マスク。"""
        m = self.region_cells(lvl)
        if lvl < self.max_level:
            m = m & ~_upsample(self.refined[lvl], self.bf)
        return m

    def leaf_cells(self, lvl: int) -> tuple[np.ndarray, np.ndarray]:
        j, i = np.nonzero(self.leaf_mask(lvl))
        return i.astype(np.int64), j.astype(np.int64)

    def n_leaf_cells(self) -> list[int]:
        return [int(self.leaf_mask(lvl).sum()) for lvl in range(self.n_levels)]

    def level_boxes(self, lvl: int) -> list[tuple[float, float, float, float]]:
        """レベル lvl の領域を行ごとに連結したブロックの矩形 (物理座標) — 表示・報告用。"""
        g = self.level_grid(lvl)
        bf = self.bf
        blocks = self.region_blocks[lvl]
        out = []
        for bj in range(blocks.shape[0]):
            row = blocks[bj]
            bi = 0
            while bi < row.size:
                if row[bi]:
                    b0 = bi
                    while bi < row.size and row[bi]:
                        bi += 1
                    out.append((g.x0 + b0 * bf * g.dx, g.y0 + bj * bf * g.dy,
                                g.x0 + bi * bf * g.dx, g.y0 + (bj + 1) * bf * g.dy))
                else:
                    bi += 1
        return out
