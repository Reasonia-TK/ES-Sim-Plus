"""AMR 階層の葉セルを計算セルとして使う粒子法 (GPU DSMC) の表 (prompts/127)。

PIC の AmrPicLayout (合成格子の節点・行列) と違い、セル中心の量だけを扱う:

- セル番号 = ブロック番号·bf² + ブロック内の局所番号 (kernels/dsmc.cu の ds_leaf_cell が
  ブロック表を上から辿って求める。葉でないセル番号は体積 0 で分子が入らない)
- 葉セルの気体体積 (軸対称は 2π 込み)、表示用メッシュ (葉セルごとに 2 三角形) と三角形 → セル
- 平滑化の近傍: 面を共有する葉セルの組と重み (レベル差は 2:1 バランスで高々 1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geom.model import GeometryModel
from .composite import cell_integral
from .electrostatic import display_mesh
from .hierarchy import AmrHierarchy
from .locate import leaf_level_map, locate_fine, to_fine_index
from .pic_layout import block_index, cell_id


@dataclass
class AmrCellLayout:
    hier: AmrHierarchy
    dd: np.ndarray                # float64[4]: x0, y0, 1/dx0, 1/dy0
    di: np.ndarray                # int64[4 + 2 (L+1)]
    refined: np.ndarray           # uint8 (ブロック表)
    blockid: np.ndarray           # int32
    n_blocks: int
    bf: int
    cell_vol_gas: np.ndarray      # (n_cells,) 葉セルの気体体積 (葉でないセルは 0)
    cell_level: np.ndarray        # (n_cells,) 葉セルのレベル (葉でないセルは -1)
    cell_xy: np.ndarray           # (n_cells, 2) 葉セルの中心
    disp_nodes: np.ndarray
    disp_tris: np.ndarray
    disp_tri_region: np.ndarray
    tri_cell: np.ndarray          # (M,) 三角形 → セル番号
    nb_a: np.ndarray              # 平滑化の近傍の組 (セル番号)
    nb_b: np.ndarray
    nb_w: np.ndarray              # 重み = min(V_a, V_b) × 共有面の長さ / 短い方の面の長さ
    h_min: float                  # 最細レベルの格子幅 (小さい方向)
    block_of_level: list

    @property
    def n_cells(self) -> int:
        return self.n_blocks * self.bf * self.bf


def cell_of_points(lay: AmrCellLayout, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """点の所属葉セルのセル番号 (CPU。GPU の ds_leaf_cell と同じ規約: 境界上は floor 側)。"""
    hier = lay.hier
    I, J = to_fine_index(hier, x, y)
    lvl, i, j, _, _ = locate_fine(hier, I, J)
    out = np.empty(lvl.size, dtype=np.int64)
    for lv in np.unique(lvl):
        m = lvl == lv
        out[m] = cell_id(hier, lay.block_of_level, int(lv), i[m], j[m])
    return out


def smooth_cells(lay: AmrCellLayout, q: np.ndarray, passes: int, theta: float = 0.2) -> np.ndarray:
    """セルの密度量 q (n_cells, k) を近傍の体積重み対称拡散で均す (総量 Σ q V を保存・正値)。"""
    vol = lay.cell_vol_gas
    gas = vol > 0.0
    safe = np.where(gas, vol, 1.0)[:, None]
    q = np.where(gas[:, None], q, 0.0)
    a, b, w = lay.nb_a, lay.nb_b, lay.nb_w[:, None]
    for _ in range(passes):
        flux = w * (q[b] - q[a])
        dq = np.zeros_like(q)
        np.add.at(dq, a, flux)
        np.add.at(dq, b, -flux)
        q = q + theta * dq / safe
        q[~gas] = 0.0
    return q


#: 動的再格子化のヒステリシス: 既に細分化されている所は 格子幅/λ がこの倍率 × 閾値を超える限り保つ
REGRID_KEEP = 0.7


def mfp_tags(lay: AmrCellLayout, lam: np.ndarray, hier: AmrHierarchy, h_over_mfp: float,
             keep: AmrHierarchy | None = None) -> list[np.ndarray]:
    """DSMC の動的再格子化のタグ (prompts/127): 格子幅/平均自由行程 が閾値を超えるセル。

    hier (候補の階層) のレベル l の領域のセルごとに、中心と 4 隅 (少し内側) を含む旧格子 lay の葉セルの
    平均自由行程 lam (セル番号ごと、分子の居ないセルは inf) の最小で判定する。keep (現在の階層) で
    細分化されているセルは閾値 × REGRID_KEEP まで保つ。上限は spec.max_level。
    """
    cap = hier.spec.max_level
    b = hier.base
    tags = [np.zeros((b.ny << lvl, b.nx << lvl), dtype=bool) for lvl in range(cap)]
    for lvl in range(min(hier.n_levels, cap)):
        cj, ci = np.nonzero(hier.region_cells(lvl))
        if ci.size == 0:
            continue
        g = hier.level_grid(lvl)
        lam_c = np.full(ci.size, np.inf)
        for fx, fy in ((0.5, 0.5), (0.01, 0.01), (0.99, 0.01), (0.01, 0.99), (0.99, 0.99)):
            ids = cell_of_points(lay, g.x0 + (ci + fx) * g.dx, g.y0 + (cj + fy) * g.dy)
            lam_c = np.minimum(lam_c, lam[ids])
        ratio = max(g.dx, g.dy) / lam_c
        hot = ratio > h_over_mfp
        if keep is not None and lvl < keep.max_level:
            ref = np.repeat(np.repeat(keep.refined[lvl], keep.bf, axis=0), keep.bf, axis=1)[cj, ci]
            hot |= ref & (ratio > REGRID_KEEP * h_over_mfp)
        tags[lvl][cj[hot], ci[hot]] = True
    return tags


def build_cell_layout(model: GeometryModel, hier: AmrHierarchy) -> AmrCellLayout:
    di, refined, blockid, n_blocks, block_of_level = block_index(hier)
    bf = hier.bf
    n_cells = n_blocks * bf * bf
    b = hier.base
    L = hier.max_level
    two_pi = 2.0 * np.pi if model.radial_axis() is not None else 1.0
    vol = np.zeros(n_cells)
    level = np.full(n_cells, -1, dtype=np.int64)
    xy = np.zeros((n_cells, 2))
    for lvl in range(hier.n_levels):
        ci, cj = hier.leaf_cells(lvl)
        if ci.size == 0:
            continue
        cid = cell_id(hier, block_of_level, lvl, ci, cj)
        vol[cid] = two_pi * cell_integral(model, hier, lvl, ci, cj, model.gas_at)
        level[cid] = lvl
        g = hier.level_grid(lvl)
        xy[cid, 0] = g.x0 + (ci + 0.5) * g.dx
        xy[cid, 1] = g.y0 + (cj + 0.5) * g.dy

    # ---- 平滑化の近傍: 各葉セルの +x・+y の面を 1/4・3/4 の 2 点で向こう側の葉セルへ -------
    leaves = leaf_level_map(hier)
    nxf, nyf = b.nx << L, b.ny << L
    eps = 1e-3
    pa, pb, frac = [], [], []
    for lvl in range(hier.n_levels):
        ci, cj = hier.leaf_cells(lvl)
        if ci.size == 0:
            continue
        cid = cell_id(hier, block_of_level, lvl, ci, cj)
        s = 1 << (L - lvl)
        for axis in (0, 1):
            for t in (0.25, 0.75):
                if axis == 0:
                    I = (ci + 1) * s + eps
                    J = (cj + t) * s
                    ok = (ci + 1) * s < nxf
                else:
                    I = (ci + t) * s
                    J = (cj + 1) * s + eps
                    ok = (cj + 1) * s < nyf
                if not np.any(ok):
                    continue
                lv2, i2, j2, _, _ = locate_fine(hier, I[ok], J[ok], leaves)
                nb = np.empty(lv2.size, dtype=np.int64)
                for lv in np.unique(lv2):
                    m = lv2 == lv
                    nb[m] = cell_id(hier, block_of_level, int(lv), i2[m], j2[m])
                own = cid[ok]
                # 面の半分ずつ: 細かい側の面の長さに対する比 (同じレベル・粗い隣: 0.5、細かい隣: 1.0)
                finer = lv2 > lvl
                pa.append(own)
                pb.append(nb)
                frac.append(np.where(finer, 1.0, 0.5))
    if pa:
        a = np.concatenate(pa)
        bb = np.concatenate(pb)
        f = np.concatenate(frac)
        key = a * n_cells + bb
        uk, inv = np.unique(key, return_inverse=True)
        fsum = np.bincount(inv, weights=f)
        a, bb = uk // n_cells, uk % n_cells
        w = np.minimum(vol[a], vol[bb]) * np.minimum(fsum, 1.0)
        keep = (w > 0.0) & (a != bb)
        nb_a, nb_b, nb_w = a[keep], bb[keep], w[keep]
    else:
        nb_a = nb_b = np.zeros(0, dtype=np.int64)
        nb_w = np.zeros(0)

    nodes, tris, tri_region, _keys, tri_cells = display_mesh(model, hier, return_cells=True)
    if tri_cells.size:
        tri_cell = np.empty(tri_cells.shape[0], dtype=np.int64)
        for lv in np.unique(tri_cells[:, 0]):
            m = tri_cells[:, 0] == lv
            tri_cell[m] = cell_id(hier, block_of_level, int(lv), tri_cells[m, 1], tri_cells[m, 2])
    else:
        tri_cell = np.zeros(0, dtype=np.int64)

    dd = np.array([b.x0, b.y0, 1.0 / b.dx, 1.0 / b.dy], dtype=np.float64)
    return AmrCellLayout(
        hier=hier, dd=dd, di=di, refined=refined, blockid=blockid, n_blocks=n_blocks, bf=bf,
        cell_vol_gas=vol, cell_level=level, cell_xy=xy, disp_nodes=nodes, disp_tris=tris,
        disp_tri_region=tri_region, tri_cell=tri_cell, nb_a=nb_a, nb_b=nb_b, nb_w=nb_w,
        h_min=min(b.dx, b.dy) / (1 << L), block_of_level=block_of_level,
    )
