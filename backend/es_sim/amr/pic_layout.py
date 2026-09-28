"""AMR 階層上の GPU PIC に必要な表・行列 (CPU で構築、prompts/122)。

全節点 (合成格子の正準節点: 未知・ぶら下がり・固定) を 1 本の配列 (番号 = op.keys の順) で持ち、
1 ステップの場の計算を次の疎行列で表す (列の後ろ K 個はグループ電位 V):

- 右辺      b_c   = PTq [q_all; V]      (PTq = [P_nodeᵀ | Coup_c]、q_all = 静電荷 + 粒子 + 表面電荷)
- 電位      φ_all = PC  [x_c; V]        (PC  = [P_node | C_node]、固定節点は自分のグループの単位)
- 節点電場  E     = G   [φ_all; V]      (fieldops.build_efield_stencil)

粒子の所属セルは kernels/pic.cu の es_locate が次のブロック表で求める:
refined (ブロックが細分化されているか)・blockid (レベル領域内のブロック番号) をレベルごとの
ブロック解像度で連結し、ブロックごとに (bf+1)² 個の格子点の節点番号表 (tab) を持つ。
セル番号 = block·bf² + 局所番号 (フレームの要素密度用)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ..geom.model import GeometryModel
from .composite import CompositeOperator, _canon_key, assemble_composite, cell_integral, dual_integral
from .electrostatic import display_mesh
from .fieldops import build_efield_stencil
from .hierarchy import AmrHierarchy


@dataclass
class AmrPicLayout:
    hier: AmrHierarchy
    op: CompositeOperator
    n_nodes: int
    n_c: int
    n_groups: int                 # K (列の拡張に使うグループ数、≥ 1)
    PTq: sp.csr_matrix            # (n_c, N + K)
    PC: sp.csr_matrix             # (N, n_c + K)
    Gx: sp.csr_matrix             # (N, N + K)
    Gy: sp.csr_matrix
    edge_u: np.ndarray            # 場のエネルギー用の辺 (節点番号)
    edge_v: np.ndarray
    edge_g: np.ndarray
    coup_u: np.ndarray            # Dirichlet 結合 (節点番号, グループ, g)
    coup_grp: np.ndarray
    coup_g: np.ndarray
    q_static: np.ndarray          # (N,) 静電荷 (固定節点は 0)
    xy: np.ndarray                # (N, 2)
    # es_locate の表
    dd: np.ndarray                # float64[4]
    di: np.ndarray                # int64[4 + 2 (L+1)]
    refined: np.ndarray           # uint8
    blockid: np.ndarray           # int32
    tab: np.ndarray               # int32 (n_blocks·(bf+1)²)
    n_blocks: int
    bf: int
    # 体積 (物理量: 軸対称は 2π 込み)
    node_vol_gas: np.ndarray      # (N,) 節点の双対セルの気体体積
    cell_vol_gas: np.ndarray      # (n_blocks·bf²,) 葉セルの気体体積 (葉でないセルは 0)
    total_gas_volume: float
    # 表示用メッシュ (v1 UI 互換)
    disp_nodes: np.ndarray
    disp_tris: np.ndarray
    disp_tri_region: np.ndarray
    disp_idx: np.ndarray          # 表示節点 → 全節点番号
    tri_cell: np.ndarray          # 三角形 → セル番号
    h_min: float                  # 最細レベルの格子幅 (小さい方向)

    @property
    def n_cells(self) -> int:
        return self.n_blocks * self.bf * self.bf


def _block_tables(hier: AmrHierarchy, op: CompositeOperator):
    L = hier.max_level
    bf = hier.bf
    b = hier.base
    ref_parts, bid_parts, tabs = [], [], []
    di = [L, bf, b.nx, b.ny]
    off = 0
    n_blocks = 0
    block_of_level = []
    s1 = bf + 1
    loc = np.arange(s1)
    for lvl in range(hier.n_levels):
        region = hier.region_blocks[lvl]
        nby, nbx = region.shape
        refined = hier.refined[lvl] if lvl < L else np.zeros_like(region)
        bid = np.full(region.shape, -1, dtype=np.int64)
        bj, bi = np.nonzero(region)
        bid[bj, bi] = n_blocks + np.arange(bj.size)
        block_of_level.append(bid)
        # ブロックの (bf+1)² 格子点 → 最細添字 → 節点番号
        s = L - lvl
        I = ((bi[:, None, None] * bf + loc[None, None, :]) << s) * np.ones((1, s1, 1), dtype=np.int64)
        J = ((bj[:, None, None] * bf + loc[None, :, None]) << s) * np.ones((1, 1, s1), dtype=np.int64)
        keys = _canon_key(hier, I.ravel(), J.ravel())
        tabs.append(op.node_index(keys).astype(np.int32))
        ref_parts.append(refined.astype(np.uint8).ravel())
        bid_parts.append(bid.astype(np.int32).ravel())
        di += [off, nbx]
        off += nbx * nby
        n_blocks += bj.size
    return (np.asarray(di, dtype=np.int64), np.concatenate(ref_parts), np.concatenate(bid_parts),
            np.concatenate(tabs) if tabs else np.zeros(0, dtype=np.int32), n_blocks, block_of_level)


def build_pic_layout(model: GeometryModel, hier: AmrHierarchy, op: CompositeOperator | None = None) -> AmrPicLayout:
    op = op if op is not None else assemble_composite(model, hier)
    N = op.keys.size
    n_c = op.A_c.shape[0]
    K = op.coup_full.shape[1]
    unk = np.nonzero(op.fixed_group < 0)[0]
    node_of_u = np.empty(unk.size, dtype=np.int64)
    node_of_u[op.u_of_node[unk]] = unk

    # ---- 行列 --------------------------------------------------------------------------
    Pc = op.P.tocoo()
    P_node = sp.csr_matrix((Pc.data, (node_of_u[Pc.row], Pc.col)), shape=(N, n_c))
    Cc = op.C.tocoo()
    fixed = np.nonzero(op.fixed_group >= 0)[0]
    C_node = sp.csr_matrix(
        (np.concatenate([Cc.data, np.ones(fixed.size)]),
         (np.concatenate([node_of_u[Cc.row], fixed]), np.concatenate([Cc.col, op.fixed_group[fixed]]))),
        shape=(N, K))
    PC = sp.hstack([P_node, C_node], format="csr")
    PTq = sp.hstack([P_node.T.tocsr(), op.coup_c], format="csr")
    Gx, Gy = build_efield_stencil(model, hier, op)
    q_static = np.zeros(N)
    q_static[node_of_u] = op.q_full

    # ---- es_locate の表 ------------------------------------------------------------------
    di, refined, blockid, tab, n_blocks, block_of_level = _block_tables(hier, op)
    b = hier.base
    dd = np.array([b.x0, b.y0, 1.0 / b.dx, 1.0 / b.dy], dtype=np.float64)

    # ---- 体積 ----------------------------------------------------------------------------
    two_pi = 2.0 * np.pi if model.radial_axis() is not None else 1.0

    def node_of(I, J):
        return op.node_index(_canon_key(hier, I, J))

    node_vol_gas = two_pi * dual_integral(model, hier, N, node_of, model.gas_at)
    bf = hier.bf
    cell_vol_gas = np.zeros(n_blocks * bf * bf)
    for lvl in range(hier.n_levels):
        ci, cj = hier.leaf_cells(lvl)
        if ci.size == 0:
            continue
        cid = block_of_level[lvl][cj // bf, ci // bf] * bf * bf + (cj % bf) * bf + (ci % bf)
        cell_vol_gas[cid] = two_pi * cell_integral(model, hier, lvl, ci, cj, model.gas_at)

    # ---- 表示用メッシュ --------------------------------------------------------------------
    nodes, tris, tri_region, keys, tri_cells = display_mesh(model, hier, return_cells=True)
    disp_idx = op.node_index(keys) if keys.size else np.zeros(0, dtype=np.int64)
    if tri_cells.size:
        lv, ci, cj = tri_cells[:, 0], tri_cells[:, 1], tri_cells[:, 2]
        tri_cell = np.empty(lv.size, dtype=np.int64)
        for lvl in np.unique(lv):
            m = lv == lvl
            tri_cell[m] = block_of_level[lvl][cj[m] // bf, ci[m] // bf] * bf * bf + (cj[m] % bf) * bf + (ci[m] % bf)
    else:
        tri_cell = np.zeros(0, dtype=np.int64)

    L = hier.max_level
    return AmrPicLayout(
        hier=hier, op=op, n_nodes=N, n_c=n_c, n_groups=K, PTq=PTq, PC=PC, Gx=Gx, Gy=Gy,
        edge_u=node_of_u[op.edge_u], edge_v=node_of_u[op.edge_v], edge_g=op.edge_g,
        coup_u=node_of_u[op.coup_u], coup_grp=op.coup_grp, coup_g=op.coup_g,
        q_static=q_static, xy=op.xy,
        dd=dd, di=di, refined=refined, blockid=blockid, tab=tab, n_blocks=n_blocks, bf=bf,
        node_vol_gas=node_vol_gas, cell_vol_gas=cell_vol_gas, total_gas_volume=float(node_vol_gas.sum()),
        disp_nodes=nodes, disp_tris=tris, disp_tri_region=tri_region, disp_idx=disp_idx, tri_cell=tri_cell,
        h_min=min(b.dx, b.dy) / (1 << L),
    )
