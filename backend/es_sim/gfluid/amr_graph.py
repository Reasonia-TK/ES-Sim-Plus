"""AMR 階層 (合成格子) 上の流体 2D の輸送グラフ (prompts/128)。

合成格子の全節点 (ぶら下がり節点を含む) を輸送の未知数とし、葉セルの Delaunay 適合三角形分割の
Voronoi 有限体積で輸送グラフを組む:

- 細分化の境目に接しない葉セルは対角線で 2 つの直角三角形に分ける。直角三角形の外心は斜辺の中点なので、
  Voronoi 領域は双対セル (1/4 セルずつ) そのもの、辺の重み (Voronoi 面の長さ / 辺長 = EAFE の余接公式)
  も軸方向の辺だけに乗る — 一様格子版 (gfluid.geometry) と同じ離散化になる。
- 辺の中点にぶら下がり節点を持つ粗い葉セル (遷移セル) は、角と中点の Delaunay 三角形分割にする
  (16 通り。すべて鈍角の無い三角形なので重みは非負で、行列は常に M 行列)。
- 固体は一様格子版と同じ埋め込み境界: 境界が通る葉セルの三角形では、Voronoi 領域の気体の割合を標本点で、
  Voronoi 面の開口を「辺に平行な標本線が端から端まで気体中を通るか」で求める。固体内部の節点に
  かかる気体の小片と壁は隣のアクティブ節点へ併合、固体表面は小片に分けて最寄りの節点に持たせる。
- 軸対称は r 重み (Voronoi 面は ∫ r ds、領域は ∫ r dA) を厳密に、2π は最後に掛ける。
- Poisson は P4 の合成格子の演算子 (build_pic_layout の PTq / PC) を使う: 電荷は節点の気体体積 ×
  e(n_i − n_e) を Pᵀ で未知へ集約、電位は PC で全節点 (ぶら下がり節点は補間) に戻す。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay, cKDTree

from ..amr.composite import _canon_key
from ..amr.hierarchy import AmrHierarchy
from ..amr.pic_layout import AmrPicLayout, _cic, build_pic_layout
from ..eb.build import _mark_boundary_cells
from ..geom.model import GeometryModel
from .geometry import EPS_REL, SEG_FRAC, FluidGraph, _lines_all_gas, _near_gas, merge_targets_graph, solid_pieces

#: 境界が通るセルの三角形で、辺ごとの Voronoi 面にとる標本線の本数
N_FACE = 6
#: 軸に平行でない辺 (遷移セルの中) の標本線の気体判定に使う点の数
N_LINE = 9
#: 外周の辺 1 本あたりの標本点 (偶数: 前半が始点、後半が終点の Voronoi 領域)
N_SIDE = 8
#: Voronoi 領域の小三角形 1 つあたりの標本点の一辺の数 (n² 点)
N_TRI = 4

# 遷移セルの基準点: 角 0..3 = (0,0) (1,0) (1,1) (0,1)、辺の中点 4..7 = 下・右・上・左
_REF = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5]], dtype=np.float64)


@dataclass
class AmrFluidGraph(FluidGraph):
    """AMR 版の輸送グラフ (節点番号 = 合成格子の節点番号)。"""

    lay: AmrPicLayout | None = None
    nodes: np.ndarray | None = None       # (N, 2)
    tris: np.ndarray | None = None        # 表示用 (導体内の葉セルを除く) の三角形 (M, 3)
    tri_region: np.ndarray | None = None
    h_min: float = 0.0


def _patterns(ax: float, ay: float) -> list[np.ndarray]:
    """辺の中点の有無 (4 bit: 下・右・上・左) ごとの Delaunay 三角形分割 (基準点の番号、反時計回り)。"""
    out = []
    scale = np.array([ax, ay])
    for mask in range(16):
        idx = [0, 1, 2, 3] + [4 + e for e in range(4) if mask >> e & 1]
        pts = _REF[idx] * scale
        tri = Delaunay(pts).simplices
        loc = np.asarray(idx)[tri]
        p = _REF[loc] * scale
        cross = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
        loc = np.where((cross < 0)[:, None], loc[:, [0, 2, 1]], loc)
        out.append(loc)
    return out


def _tri_samples(n: int) -> np.ndarray:
    """三角形を n² 個の小三角形に分けた重心の重心座標 (n², 3)。"""
    pts = []
    for i in range(n):
        for j in range(n - i):
            pts.append(((i + 1 / 3) / n, (j + 1 / 3) / n))
            if i + j < n - 1:
                pts.append(((i + 2 / 3) / n, (j + 2 / 3) / n))
    a = np.asarray(pts)
    return np.stack([a[:, 0], a[:, 1], 1.0 - a[:, 0] - a[:, 1]], axis=1)


def _circumcenter(p: np.ndarray) -> np.ndarray:
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    d = 2.0 * (a[:, 0] * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1]) + c[:, 0] * (a[:, 1] - b[:, 1]))
    a2, b2, c2 = np.sum(a * a, axis=1), np.sum(b * b, axis=1), np.sum(c * c, axis=1)
    ux = (a2 * (b[:, 1] - c[:, 1]) + b2 * (c[:, 1] - a[:, 1]) + c2 * (a[:, 1] - b[:, 1])) / d
    uy = (a2 * (c[:, 0] - b[:, 0]) + b2 * (a[:, 0] - c[:, 0]) + c2 * (b[:, 0] - a[:, 0])) / d
    return np.stack([ux, uy], axis=1)


def _tri_volume(p0, p1, p2, ridx):
    """三角形の面積 (xy) / ∫ r dA (軸対称、2π なし)。各引数は (n, 2)。"""
    area = 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]))
    if ridx is None:
        return area
    return area * (p0[:, ridx] + p1[:, ridx] + p2[:, ridx]) / 3.0


def _cell_states(model: GeometryModel, hier: AmrHierarchy):
    """レベルごとのセル状態 (0 = 気体のみ、1 = 固体のみ、2 = 境界が通る) を返す関数。"""
    shapes = [c.shape for c in model.conductors] + [o.shape for o in model.others if o.type == "dielectric"]
    cache: dict[int, np.ndarray] = {}

    def state(lvl: int, ci: np.ndarray, cj: np.ndarray) -> np.ndarray:
        g = hier.level_grid(lvl)
        if lvl not in cache:
            cache[lvl] = _mark_boundary_cells(shapes, g) if shapes else np.zeros((g.ny, g.nx), dtype=bool)
        xc = g.x0 + (ci + 0.5) * g.dx
        yc = g.y0 + (cj + 0.5) * g.dy
        st = np.where(model.gas_at(xc, yc), 0, 1).astype(np.int8)
        st[cache[lvl][cj, ci]] = 2
        return st

    return state


def build_amr_fluid_graph(model: GeometryModel, hier: AmrHierarchy, lay: AmrPicLayout | None = None) -> AmrFluidGraph:
    lay = lay if lay is not None else build_pic_layout(model, hier)
    op = lay.op
    warnings: list[str] = []
    ridx = model.radial_axis()
    rz = ridx is not None
    two_pi = 2.0 * math.pi if rz else 1.0
    L = hier.max_level
    b = hier.base
    h_min = min(b.dx, b.dy) / (1 << L)
    eps = EPS_REL * h_min
    N = lay.n_nodes
    xy = lay.xy

    # ---- 1. 葉セルの三角形分割 (ぶら下がり節点の有無でパターンを選ぶ) ------------------------
    pats = _patterns(b.dx, b.dy)
    cell_state = _cell_states(model, hier)
    tri_nodes, tri_state, tri_center_x, tri_center_y = [], [], [], []
    for lvl in range(hier.n_levels):
        ci, cj = hier.leaf_cells(lvl)
        if ci.size == 0:
            continue
        s = 1 << (L - lvl)
        I0, J0 = ci * s, cj * s
        ptsI = np.stack([I0 + (_REF[k, 0] * s).astype(np.int64) for k in range(8)], axis=1)
        ptsJ = np.stack([J0 + (_REF[k, 1] * s).astype(np.int64) for k in range(8)], axis=1)
        keys = _canon_key(hier, ptsI, ptsJ)
        mask = np.zeros(ci.size, dtype=np.int64)
        if s >= 2:
            for e in range(4):
                k = keys[:, 4 + e]
                pos = np.clip(np.searchsorted(op.keys, k), 0, op.keys.size - 1)
                mask |= (op.keys[pos] == k).astype(np.int64) << e
        st = cell_state(lvl, ci, cj)
        g = hier.level_grid(lvl)
        xc = g.x0 + (ci + 0.5) * g.dx
        yc = g.y0 + (cj + 0.5) * g.dy
        for m in np.unique(mask):
            sel = np.nonzero(mask == m)[0]
            loc = pats[int(m)]                                   # (T, 3)
            kk = keys[sel][:, loc]                               # (n, T, 3)
            tri_nodes.append(op.node_index(kk.reshape(-1)).reshape(-1, 3))
            tri_state.append(np.repeat(st[sel], loc.shape[0]))
            tri_center_x.append(np.repeat(xc[sel], loc.shape[0]))
            tri_center_y.append(np.repeat(yc[sel], loc.shape[0]))
    tri = np.concatenate(tri_nodes)
    tri_st = np.concatenate(tri_state)
    cell_cx = np.concatenate(tri_center_x)
    cell_cy = np.concatenate(tri_center_y)
    live = tri_st != 1                                           # 固体だけのセルの三角形は寄与なし

    # ---- 2. 三角形ごとの Voronoi 面・領域 (純粋な気体のセルは厳密、境界のセルは標本) -----------
    P = xy[tri]                                                  # (T, 3, 2)
    O = _circumcenter(P)
    vg = np.zeros(N)
    e_u, e_v, e_w = [], [], []
    samp = _tri_samples(N_TRI)
    for k in range(3):
        i, j = (k + 1) % 3, (k + 2) % 3
        Pi, Pj, Pk = P[:, i], P[:, j], P[:, k]
        M = 0.5 * (Pi + Pj)
        ev = Pj - Pi
        elen = np.hypot(ev[:, 0], ev[:, 1])
        fvec = O - M
        flen = np.hypot(fvec[:, 0], fvec[:, 1])
        same_side = np.sum(fvec * (Pk - M), axis=1) >= -1e-12 * elen * elen
        if not np.all(same_side[live]):
            warnings.append("流体 2D (AMR): 鈍角の三角形があります (該当する辺の重みを 0 にしました)")
        # 直角三角形の斜辺 (外心 = 斜辺の中点) は丸め誤差の分を 0 に (輸送の辺に残さない)
        flen = np.where(same_side & (flen > 1e-9 * elen), flen, 0.0)
        rw = 1.0 if ridx is None else 0.5 * (M[:, ridx] + O[:, ridx])
        w = np.where(live, flen * rw / np.where(elen > 0.0, elen, 1.0), 0.0)
        mixed = np.nonzero((tri_st == 2) & (flen > 0.0))[0]
        if mixed.size:
            # Voronoi 面 M → O 上の標本点ごとに、辺に平行で同じ長さの線分が端から端まで気体か
            t = (np.arange(N_FACE) + 0.5) / N_FACE
            q = M[mixed][:, None, :] + t[None, :, None] * fvec[mixed][:, None, :]      # (m, F, 2)
            half = 0.5 * ev[mixed][:, None, :]
            a0p, a1p = (q - half).reshape(-1, 2), (q + half).reshape(-1, 2)
            horiz = np.abs(ev[mixed, 1]) <= 1e-12 * elen[mixed]
            vert = np.abs(ev[mixed, 0]) <= 1e-12 * elen[mixed]
            ok = np.zeros(a0p.shape[0], dtype=bool)
            hh = np.repeat(horiz, N_FACE)
            vv = np.repeat(vert, N_FACE)
            # 端は eps だけ内側へ (端点が固体の表面上にある辺で、丸め誤差の微小な区間を固体と数えない)
            if np.any(hh):
                x_lo = np.minimum(a0p[hh, 0], a1p[hh, 0]) + eps
                x_hi = np.maximum(a0p[hh, 0], a1p[hh, 0]) - eps
                ok[hh] = _lines_all_gas(model, 0, a0p[hh, 1], x_lo, x_hi)
            if np.any(vv):
                y_lo = np.minimum(a0p[vv, 1], a1p[vv, 1]) + eps
                y_hi = np.maximum(a0p[vv, 1], a1p[vv, 1]) - eps
                ok[vv] = _lines_all_gas(model, 1, a0p[vv, 0], y_lo, y_hi)
            dd = ~(hh | vv)
            if np.any(dd):
                s_ = (np.arange(N_LINE) + 0.5) / N_LINE
                pts = a0p[dd][:, None, :] + s_[None, :, None] * (a1p[dd] - a0p[dd])[:, None, :]
                ok[dd] = np.all(model.gas_at(pts[..., 0], pts[..., 1]), axis=1)
            ok = ok.reshape(-1, N_FACE)
            rq = np.ones(ok.shape) if ridx is None else q[..., ridx]
            w[mixed] = np.sum(np.where(ok, rq, 0.0), axis=1) * (flen[mixed] / N_FACE) / elen[mixed]
        e_u.append(tri[:, i])
        e_v.append(tri[:, j])
        e_w.append(w)

        # 頂点 k の Voronoi 領域 (k の 2 辺の中点と外心で囲む四角形 = 小三角形 2 つ)
        Ma = 0.5 * (P[:, k] + P[:, (k + 1) % 3])
        Mb = 0.5 * (P[:, k] + P[:, (k + 2) % 3])
        vk = np.zeros(tri.shape[0])
        for A, B in ((Ma, O), (O, Mb)):
            vol = _tri_volume(P[:, k], A, B, ridx)
            vk += np.where(tri_st == 0, vol, 0.0)
            mix = np.nonzero(tri_st == 2)[0]
            if mix.size:
                pts = (samp[None, :, 0:1] * P[mix, k][:, None, :] + samp[None, :, 1:2] * A[mix][:, None, :]
                       + samp[None, :, 2:3] * B[mix][:, None, :])
                gas = model.gas_at(pts[..., 0], pts[..., 1]).astype(np.float64)
                if ridx is None:
                    frac = gas.mean(axis=1)
                else:
                    rr = pts[..., ridx]
                    frac = np.sum(gas * rr, axis=1) / np.maximum(np.sum(rr, axis=1), 1e-300)
                vk[mix] += frac * vol[mix]
        np.add.at(vg, tri[:, k], vk)

    # ---- 3. 辺の重みを節点の組ごとに合計 ----------------------------------------------------
    eu = np.concatenate(e_u)
    ev_ = np.concatenate(e_v)
    ew = np.concatenate(e_w)
    lo_, hi_ = np.minimum(eu, ev_), np.maximum(eu, ev_)
    key = lo_ * N + hi_
    uk, inv = np.unique(key, return_inverse=True)
    wsum = np.bincount(inv, weights=ew)
    pa, pb = uk // N, uk % N

    # ---- 4. アクティブ節点・小片の併合 ----------------------------------------------------
    active = _near_gas(model, xy[:, 0], xy[:, 1], eps) & (vg > 0.0)
    if not np.any(active):
        raise ValueError("流体輸送領域が空です (domain 全体が固体です)")

    # ---- 5. 壁 (外周の辺と固体の表面) ------------------------------------------------------
    walls = _side_walls_amr(model, xy, tri, eps)
    tree = cKDTree(xy)
    for mid, area, n_in, gamma, group, pg in solid_pieces(model, SEG_FRAC * h_min, eps):
        _, node = tree.query(pg)
        walls.append((node.astype(np.int64), area, n_in, np.full(node.size, gamma),
                      np.full(node.size, group, dtype=np.int64), mid[:, 0], mid[:, 1]))
    if walls:
        node, area, nrm, gamma, group, wx, wy = (np.concatenate(a) for a in zip(*walls))
    else:
        node = np.zeros(0, dtype=np.int64)
        area = gamma = wx = wy = np.zeros(0)
        nrm = np.zeros((0, 2))
        group = np.zeros(0, dtype=np.int64)
    has_wall = np.zeros(N, dtype=bool)
    has_wall[node] = True
    o, orphan = merge_targets_graph(N, active, vg, has_wall, pa, pb)
    node_vol = two_pi * np.asarray(o.T @ vg).ravel()
    node_vol[~active] = 0.0
    charge_map = (sp.diags(vg) @ o).tocsr()
    dropped = two_pi * float(vg[orphan].sum())
    gas_volume = two_pi * float(vg.sum())
    if dropped > 1e-3 * gas_volume:
        warnings.append(
            f"流体 2D: 固体に囲まれて輸送節点に併合できない気体の小片が総体積の "
            f"{100.0 * dropped / gas_volume:.2g}% あります (格子幅より狭い隙間)"
        )
    sub = o[node].tocoo()
    r = sub.row
    node = sub.col.astype(np.int64)
    area = two_pi * area[r] * sub.data
    nrm, gamma, group, wx, wy = nrm[r], gamma[r], group[r], wx[r], wy[r]

    # 壁向き電場のプローブ: W と、法線の逆向きに局所の格子幅 1 つ分入った A (合成格子の葉セルで双一次補間)
    from ..amr.locate import locate_fine, to_fine_index

    if nrm.size:
        I, J = to_fine_index(hier, wx - 2 * eps * nrm[:, 0], wy - 2 * eps * nrm[:, 1])
        lvl_w, *_ = locate_fine(hier, I, J)
        hl = min(b.dx, b.dy) / (1 << lvl_w).astype(np.float64)
        ax = np.clip(wx - hl * nrm[:, 0], b.x0, b.x1)
        ay = np.clip(wy - hl * nrm[:, 1], b.y0, b.y1)
        wl = np.maximum((wx - ax) * nrm[:, 0] + (wy - ay) * nrm[:, 1], 0.25 * hl)
        w_idx, w_wt = _cic(lay, wx, wy)
        a_idx, a_wt = _cic(lay, ax, ay)
    else:
        wl = np.zeros(0)
        w_idx = a_idx = np.zeros((0, 4), dtype=np.int64)
        w_wt = a_wt = np.zeros((0, 4))

    keep = (wsum > 0.0) & active[pa] & active[pb]           # 輸送の辺は両端がアクティブなものだけ
    # 表示用: 導体の内部の葉セル (穴) の三角形を除く
    in_cond = model.classify_conductor(cell_cx, cell_cy, 0.0) >= 0 if model.conductors else np.zeros(tri.shape[0], bool)
    disp = tri[~in_cond]
    cent = xy[disp].mean(axis=1)
    other = model.classify_other(cent[:, 0], cent[:, 1])
    region_idx = np.array([o_.index for o_ in model.others] + [-1], dtype=np.int64)
    tri_region = np.where(other >= 0, region_idx[np.maximum(other, 0)], -1)
    return AmrFluidGraph(
        grid=b,
        active=active,
        node_vol=node_vol,
        charge_map=charge_map,
        edge_p=pa[keep],
        edge_q=pb[keep],
        edge_w=two_pi * wsum[keep],
        edge_len=np.hypot(*(xy[pb[keep]] - xy[pa[keep]]).T),
        wall_node=node,
        wall_area=area,
        wall_normal=nrm,
        wall_gamma=gamma,
        wall_group=group,
        wall_w_idx=w_idx,
        wall_w_wt=w_wt,
        wall_a_idx=a_idx,
        wall_a_wt=a_wt,
        wall_len=wl,
        gas_volume=gas_volume,
        dropped_volume=dropped,
        warnings=warnings,
        lay=lay,
        nodes=xy,
        tris=disp,
        tri_region=tri_region,
        h_min=h_min,
    )


def _side_walls_amr(model: GeometryModel, xy: np.ndarray, tri: np.ndarray, eps: float) -> list:
    """外周の壁: 三角形分割の外周の辺ごとに、前半を始点・後半を終点の節点へ (気体に接する部分だけ)。"""
    dom = model.domain
    ridx = model.radial_axis()
    tol = model.tol
    # 外周の辺 = 1 つの三角形にしか属さない辺
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    k = np.sort(e, axis=1)
    uk, cnt = np.unique(k[:, 0] * xy.shape[0] + k[:, 1], return_counts=True)
    bnd = uk[cnt == 1]
    u, v = bnd // xy.shape[0], bnd % xy.shape[0]
    out = []
    t_mid = (np.arange(N_SIDE) + 0.5) / N_SIDE
    for side, sbc in model.sides.items():
        if sbc.kind in ("symmetry", "periodic"):
            continue
        if (ridx == 1 and side == "bottom" and abs(dom.y0) <= tol) or (ridx == 0 and side == "left" and abs(dom.x0) <= tol):
            continue
        along_x = side in ("bottom", "top")
        c = {"bottom": dom.y0, "top": dom.y1, "left": dom.x0, "right": dom.x1}[side]
        coord = 1 if along_x else 0
        on = (np.abs(xy[u, coord] - c) <= tol) & (np.abs(xy[v, coord] - c) <= tol)
        if not np.any(on):
            continue
        uu, vv = u[on], v[on]
        inward = {"bottom": (0.0, 1.0), "top": (0.0, -1.0), "left": (1.0, 0.0), "right": (-1.0, 0.0)}[side]
        pa, pb = xy[uu], xy[vv]
        pts = pa[:, None, :] + t_mid[None, :, None] * (pb - pa)[:, None, :]            # (n, S, 2)
        seg = np.hypot(*(pb - pa).T) / N_SIDE
        gas = model.gas_at(pts[..., 0] + eps * inward[0], pts[..., 1] + eps * inward[1])
        rw = np.ones(pts.shape[:2]) if ridx is None else pts[..., ridx]
        a = np.where(gas, rw, 0.0) * seg[:, None]
        half = N_SIDE // 2
        area_u = a[:, :half].sum(axis=1)
        area_v = a[:, half:].sum(axis=1)
        gamma = 0.0
        if sbc.kind == "dirichlet" and side in model.side_group:
            gamma = model.groups[model.side_group[side]].see_gamma
        nodes = np.concatenate([uu, vv])
        areas = np.concatenate([area_u, area_v])
        keep = areas > 0.0
        nodes, areas = nodes[keep], areas[keep]
        # 同じ節点に 2 本の辺から入る分を合計する (壁の点 W は節点の位置)
        un, inv = np.unique(nodes, return_inverse=True)
        asum = np.bincount(inv, weights=areas)
        nrm = np.tile(np.array([-inward[0], -inward[1]]), (un.size, 1))
        out.append((un.astype(np.int64), asum, nrm, np.full(un.size, gamma), np.full(un.size, -1, dtype=np.int64),
                    xy[un, 0], xy[un, 1]))
    return out
