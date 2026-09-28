"""v2 直交格子 (節点中心) の定義と、マルチグリッド向けのセル数選択 (prompts/119)。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..geom.model import DomainRect

#: セル数 n = m·2^k の仮数 m の上限。最粗レベルが m×m セル以下になり直接法で解ける
MAX_MANTISSA = 15


def choose_cells(length: float, h: float, max_mantissa: int = MAX_MANTISSA) -> int:
    """length/h 以上で m·2^k (1 ≤ m ≤ max_mantissa) の形をとる最小のセル数。

    要求メッシュ幅 h を超えない (最大 1/max_mantissa ≈ 7〜12% 細かくなる) 範囲で、
    幾何マルチグリッドの粗視化 (2 倍ずつ) を最粗レベルまで続けられる数を選ぶ
    (AMReX の blocking_factor と同じ発想)。
    """
    if not (length > 0.0 and h > 0.0):
        raise ValueError("choose_cells: length と h は正である必要があります")
    n_min = max(1, math.ceil(length / h - 1e-9))
    best: int | None = None
    k = 0
    while True:
        m = math.ceil(n_min / 2**k)
        if m <= max_mantissa:
            cand = m * 2**k
            best = cand if best is None else min(best, cand)
        if 2**k >= n_min:
            break
        k += 1
    assert best is not None
    return best


@dataclass(frozen=True)
class CartesianGrid:
    """節点中心の一様直交格子。節点 (i, j) = (x0 + i·dx, y0 + j·dy)、i=0..nx、j=0..ny。

    配列は (ny+1, nx+1) の C 順 (x が連続方向)。セル (i, j) は [x_i, x_{i+1}]×[y_j, y_{j+1}]。
    """

    x0: float
    y0: float
    x1: float
    y1: float
    nx: int
    ny: int

    @property
    def dx(self) -> float:
        return (self.x1 - self.x0) / self.nx

    @property
    def dy(self) -> float:
        return (self.y1 - self.y0) / self.ny

    @property
    def xs(self) -> np.ndarray:
        return np.linspace(self.x0, self.x1, self.nx + 1)

    @property
    def ys(self) -> np.ndarray:
        return np.linspace(self.y0, self.y1, self.ny + 1)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny + 1, self.nx + 1)

    @property
    def n_nodes(self) -> int:
        return (self.nx + 1) * (self.ny + 1)

    def coarsening(self, min_cells: int = 2) -> tuple[int, int]:
        """次のレベルへの粗視化係数 (fx, fy) ∈ {1, 2}²。どちらも 1 なら粗視化不可。"""
        fx = 2 if (self.nx % 2 == 0 and self.nx // 2 >= min_cells) else 1
        fy = 2 if (self.ny % 2 == 0 and self.ny // 2 >= min_cells) else 1
        return fx, fy

    def coarsened(self, fx: int, fy: int) -> "CartesianGrid":
        return CartesianGrid(self.x0, self.y0, self.x1, self.y1, self.nx // fx, self.ny // fy)


def make_grid(domain: DomainRect, h: float, *, max_mantissa: int = MAX_MANTISSA) -> CartesianGrid:
    """domain を覆う格子を、要求メッシュ幅 h 以下で GMG に適したセル数で作る。"""
    nx = choose_cells(domain.width, h, max_mantissa)
    ny = choose_cells(domain.height, h, max_mantissa)
    return CartesianGrid(domain.x0, domain.y0, domain.x1, domain.y1, nx, ny)
