"""AMR 階層での点の所属葉セルと双一次重み (CPU、prompts/122)。

最細レベルの整数添字 (I, J) で与えた点は整数演算だけで厳密に葉セルを決められる (電場ステンシル・
表示・PIC の表の構築用)。物理座標の点は最細添字の実数座標に直してから同じ手順で求める。
"""

from __future__ import annotations

import numpy as np

from .hierarchy import AmrHierarchy


def leaf_level_map(hier: AmrHierarchy) -> list[np.ndarray]:
    """レベルごとの葉セルマスク (ny_l, nx_l) のリスト。"""
    return [hier.leaf_mask(lvl) for lvl in range(hier.n_levels)]


def locate_fine(hier: AmrHierarchy, I: np.ndarray, J: np.ndarray, leaves: list[np.ndarray] | None = None):
    """最細添字の実数座標 (I, J) (0 ≤ I ≤ nxf, 0 ≤ J ≤ nyf) を含む葉セル。

    戻り値: (lvl, i, j, wx, wy)。境界上の点は添字の小さい側ではなく floor の側のセル
    (右端・上端はクランプして wx=1 / wy=1)。どちらのセルでも辺上の値は同じ (辺上で線形)。
    """
    I = np.asarray(I, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    L = hier.max_level
    leaves = leaves if leaves is not None else leaf_level_map(hier)
    lvl = np.full(I.shape, -1, dtype=np.int64)
    ci = np.zeros(I.shape, dtype=np.int64)
    cj = np.zeros(I.shape, dtype=np.int64)
    wx = np.zeros(I.shape)
    wy = np.zeros(I.shape)
    todo = np.ones(I.shape, dtype=bool)
    for lv in range(L, -1, -1):
        if not np.any(todo):
            break
        s = float(1 << (L - lv))
        nx_l, ny_l = hier.base.nx << lv, hier.base.ny << lv
        idx = np.nonzero(todo)[0]
        fx = I[idx] / s
        fy = J[idx] / s
        i = np.clip(np.floor(fx).astype(np.int64), 0, nx_l - 1)
        j = np.clip(np.floor(fy).astype(np.int64), 0, ny_l - 1)
        ok = leaves[lv][j, i]
        if not np.any(ok):
            continue
        k = idx[ok]
        lvl[k] = lv
        ci[k] = i[ok]
        cj[k] = j[ok]
        wx[k] = np.clip(fx[ok] - i[ok], 0.0, 1.0)
        wy[k] = np.clip(fy[ok] - j[ok], 0.0, 1.0)
        todo[k] = False
    return lvl, ci, cj, wx, wy


def to_fine_index(hier: AmrHierarchy, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """物理座標 → 最細添字の実数座標 (domain 外はクランプ)。"""
    b = hier.base
    L = hier.max_level
    nxf, nyf = b.nx << L, b.ny << L
    I = np.clip((np.asarray(x, dtype=np.float64) - b.x0) / (b.x1 - b.x0) * nxf, 0.0, nxf)
    J = np.clip((np.asarray(y, dtype=np.float64) - b.y0) / (b.y1 - b.y0) * nyf, 0.0, nyf)
    return I, J


def corner_fine_index(hier: AmrHierarchy, lvl: np.ndarray, i: np.ndarray, j: np.ndarray):
    """葉セル (lvl, i, j) の 4 隅の最細添字 (巻き戻し前): (I00, J00, I10, J10, I01, J01, I11, J11)。"""
    s = (hier.max_level - np.asarray(lvl)).astype(np.int64)
    i0, j0 = np.asarray(i) << s, np.asarray(j) << s
    i1, j1 = (np.asarray(i) + 1) << s, (np.asarray(j) + 1) << s
    return i0, j0, i1, j0, i0, j1, i1, j1
