"""v2 直交格子 + 埋め込み境界 (EB) 上の流体 2D の輸送グラフ (prompts/125)。

v1 fluid2d (EAFE/FEM-SG) と同じ「節点中心の有限体積 + エッジ重み w_ij」の形で、直交格子の
双対セルから輸送グラフを組む。直角三角形分割に EAFE を適用すると斜辺の重みが 0 になり
5 点ステンシルに一致するので、固体の無い領域では v1 の構造格子と同じ離散化になる
(ただし節点体積は双対セルそのもの。v1 の P1 集中質量は市松の対角線で 4/3·h² と 2/3·h² が
交互に並ぶが、こちらは一様な h²)。

- 制御体積: 双対セルのうち気体の部分 (EB Poisson の vol_gas と同じ積分、軸対称は 2π r 重み)。
- アクティブ節点: 固体の厳密な内部に無い節点 (境界上の節点は壁節点として残す。v1 の
  「固体表面の節点も輸送節点」と同じ)。固体内部の節点の双対セルにかかる気体の小片は、隣接
  するアクティブ節点へ体積比で併合する (小セル併合: 総気体体積を保つ。Poisson の電荷は
  小片の元の双対セルに置く)。
- エッジ重み: 辺を挟む 2 つの半チューブ (EB Poisson と同じ) の面積重みのうち、辺に平行な
  標本線が端から端まで気体中を通る部分だけを開口として数える (薄い固体越しに漏れない)。
  全て非負なので行列は常に M 行列 (v1 の非 Delaunay メッシュで出る負の重みは起きない)。
- 壁: domain 外周 (symmetry・rz 対称軸・periodic 以外) と固体 (導体・誘電体) の表面。表面は
  格子幅の 1/4 以下の小片に分け、気体側の点を含む双対セルの節点 (固体内部なら併合先) に
  面積 (軸対称は 2π r 倍)・法線 (気体 → 壁)・SEE γ を持たせる。
- 壁向きの電場 E·n: 壁上の点 W と、そこから法線の逆向きに 1 格子間隔 L だけ気体側へ入った
  点 A の電位差 (φ(A) − φ(W))/L (φ は節点値の双一次補間、導体の φ(W) は電極電位そのもの)。
  外周の辺では A はちょうど隣の節点になり、v1 の「壁に接する要素の E」と同じ量になる。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..eb.build import _edge_set, _strip_area
from ..eb.grid import CartesianGrid
from ..geom.model import GeometryModel
from ..geom.shapes import CircleShape

#: 境界が通る半チューブを分割する標本線の本数
N_STRIPS = 8
#: 外周の辺の、節点 1 つ分の区間あたりの標本点数 (固体が外周に接する部分を除くため)
N_SIDE = 8
#: 境界上の点の分類に使う微小距離 (格子幅比)
EPS_REL = 1e-6
#: 固体表面を分割する小片の長さ (格子幅比)
SEG_FRAC = 0.25


@dataclass
class FluidGraph:
    """直交格子の流体輸送グラフ (節点番号は格子の平坦番号 j·(nx+1) + i)。"""

    grid: CartesianGrid
    active: np.ndarray            # (N,) bool
    node_vol: np.ndarray          # (N,) 制御体積 [m³] (xy は奥行き 1 m、軸対称は 2π 込み)。非アクティブは 0
    charge_map: sp.csr_matrix     # (N, N) 電荷 q = e·charge_map @ (n_i − n_e) (2π なし、EB Poisson の右辺の単位)
    edge_p: np.ndarray            # (E,) 端点の節点番号
    edge_q: np.ndarray
    edge_w: np.ndarray            # (E,) 開口面積 / 辺長 (2π 込み)
    edge_len: np.ndarray          # (E,)
    wall_node: np.ndarray         # (W,) 壁を受け持つアクティブ節点
    wall_area: np.ndarray         # (W,) 面積 (2π 込み)
    wall_normal: np.ndarray       # (W, 2) 単位法線 (気体 → 壁)
    wall_gamma: np.ndarray        # (W,) SEE γ
    wall_group: np.ndarray        # (W,) 導体表面なら Dirichlet グループ番号、それ以外 -1
    wall_w_idx: np.ndarray        # (W, 4) 壁上の点 W の双一次補間の節点
    wall_w_wt: np.ndarray         # (W, 4)
    wall_a_idx: np.ndarray        # (W, 4) 気体側の点 A = W − L·n の双一次補間の節点
    wall_a_wt: np.ndarray         # (W, 4)
    wall_len: np.ndarray          # (W,) 法線方向の距離 L
    gas_volume: float             # 気体の総体積 (2π 込み)
    dropped_volume: float         # 併合先が無く捨てた小片の体積 (2π 込み)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return self.grid.n_nodes


# ---- 点・線の分類 ------------------------------------------------------------------


def _near_gas(model: GeometryModel, x: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
    """点のまわり (距離 eps の 8 方向) のどこかが気体か (= 固体の厳密な内部ではない)。"""
    s = math.sqrt(0.5)
    out = np.zeros(np.broadcast(x, y).shape, dtype=bool)
    for ux, uy in ((1, 0), (-1, 0), (0, 1), (0, -1), (s, s), (s, -s), (-s, s), (-s, -s)):
        out |= model.gas_at(x + eps * ux, y + eps * uy)
    return out


def _lines_all_gas(model: GeometryModel, axis: int, c: np.ndarray, a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
    """軸平行な線分 (axis=0: y=c, x∈[a0,a1]、axis=1: x=c, y∈[a0,a1]) が端から端まで気体か。

    線分を全形状の境界との厳密な交点で区間に切り、各区間の中点を gas_at で分類する
    (領域の優先順位は gas_at がそのまま扱う)。
    """
    n = c.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    ks = [np.arange(n), np.arange(n)]
    us = [a0, a1]
    for shape in [r.shape for r in model.conductors] + [r.shape for r in model.others]:
        k, u = shape.crossings_h(c, a0, a1) if axis == 0 else shape.crossings_v(c, a0, a1)
        if k.size:
            ks.append(k)
            us.append(u)
    k = np.concatenate(ks)
    u = np.concatenate(us)
    order = np.lexsort((u, k))
    k, u = k[order], u[order]
    same = k[1:] == k[:-1]
    seg_k = k[:-1][same]
    mid = 0.5 * (u[1:] + u[:-1])[same]
    cc = c[seg_k]
    gas = model.gas_at(mid, cc) if axis == 0 else model.gas_at(cc, mid)
    blocked = np.bincount(seg_k[~gas], minlength=n) > 0
    return ~blocked


def _bilinear(grid: CartesianGrid, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """点 (x, y) の節点値の双一次補間 (節点番号 (n, 4) と重み (n, 4))。domain 外は端へ寄せる。"""
    fx = np.clip((x - grid.x0) / grid.dx, 0.0, grid.nx)
    fy = np.clip((y - grid.y0) / grid.dy, 0.0, grid.ny)
    i = np.minimum(np.floor(fx).astype(np.int64), grid.nx - 1)
    j = np.minimum(np.floor(fy).astype(np.int64), grid.ny - 1)
    wx = fx - i
    wy = fy - j
    a = j * (grid.nx + 1) + i
    idx = np.stack([a, a + 1, a + grid.nx + 1, a + grid.nx + 2], axis=1)
    wt = np.stack([(1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy], axis=1)
    return idx, wt


# ---- 小セル併合 --------------------------------------------------------------------


def _merge_targets(
    grid: CartesianGrid, active: np.ndarray, vol_gas: np.ndarray, need: np.ndarray
) -> tuple[sp.csr_matrix, np.ndarray]:
    """節点 → 受け持つアクティブ節点の重み行列 O (N, N) と、併合先の無い節点のマスク。

    アクティブ節点は自分自身 (重み 1)。固体内部の節点のうち気体の小片か壁を持つもの
    (vol_gas > 0 または need) は、4 近傍のアクティブ節点 (無ければ斜め 4 近傍) へ、その
    気体体積に比例して配分する。
    """
    nx1 = grid.nx + 1
    n = grid.n_nodes
    rows = [np.nonzero(active)[0]]
    cols = [rows[0]]
    vals = [np.ones(rows[0].size)]
    sliver = np.nonzero(~active & ((vol_gas > 0.0) | need))[0]
    orphan = np.zeros(n, dtype=bool)
    if sliver.size:
        j, i = np.divmod(sliver, nx1)
        remaining = np.ones(sliver.size, dtype=bool)
        for offsets in (((1, 0), (-1, 0), (0, 1), (0, -1)), ((1, 1), (1, -1), (-1, 1), (-1, -1))):
            r_l, c_l, v_l = [], [], []
            for di, dj in offsets:
                ii, jj = i + di, j + dj
                ok = remaining & (ii >= 0) & (ii <= grid.nx) & (jj >= 0) & (jj <= grid.ny)
                nb = np.where(ok, jj * nx1 + ii, 0)
                ok &= active[nb]
                r_l.append(np.nonzero(ok)[0])
                c_l.append(nb[ok])
                v_l.append(vol_gas[nb[ok]])
            rr = np.concatenate(r_l)
            if rr.size == 0:
                continue
            cc = np.concatenate(c_l)
            vv = np.concatenate(v_l)
            tot = np.bincount(rr, weights=vv, minlength=sliver.size)
            rows.append(sliver[rr])
            cols.append(cc)
            vals.append(vv / tot[rr])
            remaining[rr] = False
        orphan[sliver[remaining]] = True
    o = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )
    return o, orphan


# ---- エッジ ------------------------------------------------------------------------


def _edges(model: GeometryModel, grid: CartesianGrid, cell_state: np.ndarray, active: np.ndarray, axis: int):
    es = _edge_set(grid, axis, False, False)
    area = np.zeros(es.p.size)
    cand = active[es.p] & active[es.q]
    for exists, row, col, lo, hi in es.halves:
        sel = np.nonzero(exists & cand)[0]
        if sel.size == 0:
            continue
        st = cell_state[row[sel], col[sel]]
        pure = st == 0
        if np.any(pure):
            g = sel[pure]
            area[g] += _strip_area(model.coord, axis, lo[g], hi[g], es.r_face[g])
        cut = st == 2
        if np.any(cut):
            g = sel[cut]
            m = g.size
            wdt = ((hi[g] - lo[g]) / N_STRIPS)[:, None]
            s_lo = lo[g][:, None] + np.arange(N_STRIPS)[None, :] * wdt
            s_hi = s_lo + wdt
            c_mid = 0.5 * (s_lo + s_hi)
            open_ = _lines_all_gas(
                model, axis, c_mid.ravel(), np.repeat(es.a0[g], N_STRIPS), np.repeat(es.a1[g], N_STRIPS)
            ).reshape(m, N_STRIPS)
            rf = np.repeat(es.r_face[g], N_STRIPS).reshape(m, N_STRIPS)
            a_s = _strip_area(model.coord, axis, s_lo, s_hi, rf)
            area[g] += np.sum(np.where(open_, a_s, 0.0), axis=1)
    keep = area > 0.0
    return es.p[keep], es.q[keep], area[keep] / es.length, np.full(int(keep.sum()), es.length)


# ---- 壁 ----------------------------------------------------------------------------


def _side_walls(model: GeometryModel, grid: CartesianGrid, eps: float):
    """外周の壁 (節点ごとの辺の区間のうち気体に接する部分)。2π は掛けない。"""
    dom = model.domain
    ridx = model.radial_axis()
    tol = model.tol
    t_mid = (np.arange(N_SIDE) + 0.5) / N_SIDE
    t_lo = np.arange(N_SIDE) / N_SIDE
    out = []
    for side, sbc in model.sides.items():
        if sbc.kind in ("symmetry", "periodic"):
            continue
        if (ridx == 1 and side == "bottom" and abs(dom.y0) <= tol) or (
            ridx == 0 and side == "left" and abs(dom.x0) <= tol
        ):
            continue  # 軸対称の対称軸 (r=0) は自然境界 (v1 と同じ)
        along_x = side in ("bottom", "top")
        u = grid.xs if along_x else grid.ys
        du = grid.dx if along_x else grid.dy
        v_side = {"bottom": dom.y0, "top": dom.y1, "left": dom.x0, "right": dom.x1}[side]
        inward = {"bottom": (0.0, 1.0), "top": (0.0, -1.0), "left": (1.0, 0.0), "right": (-1.0, 0.0)}[side]
        lo = np.maximum(u - 0.5 * du, u[0])
        hi = np.minimum(u + 0.5 * du, u[-1])
        span = (hi - lo)[:, None]
        p_lo = lo[:, None] + span * t_lo[None, :]
        p_hi = p_lo + span / N_SIDE
        p_mid = lo[:, None] + span * t_mid[None, :]
        if along_x:
            sx, sy = p_mid, np.full(p_mid.shape, v_side + eps * inward[1])
        else:
            sx, sy = np.full(p_mid.shape, v_side + eps * inward[0]), p_mid
        gas = model.gas_at(sx, sy)
        r_along = (ridx == 0 and along_x) or (ridx == 1 and not along_x)
        if ridx is None:
            a = p_hi - p_lo
        elif r_along:
            a = 0.5 * (p_hi * p_hi - p_lo * p_lo)
        else:
            a = (p_hi - p_lo) * v_side
        area = np.sum(np.where(gas, a, 0.0), axis=1)
        k = np.arange(u.size)
        if along_x:
            jj = 0 if side == "bottom" else grid.ny
            node = jj * (grid.nx + 1) + k
            wx, wy = u, np.full(u.size, v_side)
        else:
            ii = 0 if side == "left" else grid.nx
            node = k * (grid.nx + 1) + ii
            wx, wy = np.full(u.size, v_side), u
        gamma = 0.0
        if sbc.kind == "dirichlet" and side in model.side_group:
            gamma = model.groups[model.side_group[side]].see_gamma
        keep = area > 0.0
        nrm = np.tile(np.array([-inward[0], -inward[1]]), (int(keep.sum()), 1))
        out.append((node[keep], area[keep], nrm, np.full(int(keep.sum()), gamma),
                    np.full(int(keep.sum()), -1, dtype=np.int64), wx[keep], wy[keep]))
    return out


def _solid_walls(model: GeometryModel, grid: CartesianGrid, eps: float):
    """固体 (導体・誘電体) の表面の小片。2π は掛けない (r 重みのみ)。"""
    dom = model.domain
    ridx = model.radial_axis()
    seg_max = SEG_FRAC * min(grid.dx, grid.dy)
    solids = []
    for k, c in enumerate(model.conductors):
        g = model.conductor_group[k]
        solids.append((c.shape, model.groups[g].see_gamma, g))
    for o in model.others:
        if o.type == "dielectric":
            solids.append((o.shape, float(o.region.see_gamma), -1))
    out = []
    for shape, gamma, group in solids:
        if isinstance(shape, CircleShape):
            n = max(32, int(math.ceil(2.0 * math.pi * shape.r / seg_max)))
            th = 2.0 * math.pi * (np.arange(n) + 0.5) / n
            radial = np.stack([np.cos(th), np.sin(th)], axis=1)
            mid = np.array([shape.cx, shape.cy]) + shape.r * radial
            length = np.full(n, 2.0 * math.pi * shape.r / n)
            n_in = -radial
        else:
            v = shape.vertices
            w = np.roll(v, -1, axis=0)
            signed = 0.5 * float(np.sum(v[:, 0] * w[:, 1] - w[:, 0] * v[:, 1]))
            mids, lens, nrms = [], [], []
            for p, q in zip(v, w):
                d = q - p
                ln = float(np.hypot(d[0], d[1]))
                if ln <= 0.0:
                    continue
                m = max(1, int(math.ceil(ln / seg_max)))
                t = (np.arange(m) + 0.5) / m
                mids.append(p[None, :] + t[:, None] * d[None, :])
                lens.append(np.full(m, ln / m))
                left = np.array([-d[1], d[0]]) / ln
                nrms.append(np.tile(left if signed > 0.0 else -left, (m, 1)))
            if not mids:
                continue
            mid = np.concatenate(mids)
            length = np.concatenate(lens)
            n_in = np.concatenate(nrms)
        pg = mid - eps * n_in
        ps = mid + eps * n_in
        inside = (pg[:, 0] > dom.x0) & (pg[:, 0] < dom.x1) & (pg[:, 1] > dom.y0) & (pg[:, 1] < dom.y1)
        keep = inside & model.gas_at(pg[:, 0], pg[:, 1]) & ~model.gas_at(ps[:, 0], ps[:, 1])
        if not np.any(keep):
            continue
        mid, length, n_in, pg = mid[keep], length[keep], n_in[keep], pg[keep]
        area = length if ridx is None else length * mid[:, ridx]
        i = np.clip(np.rint((pg[:, 0] - grid.x0) / grid.dx).astype(np.int64), 0, grid.nx)
        j = np.clip(np.rint((pg[:, 1] - grid.y0) / grid.dy).astype(np.int64), 0, grid.ny)
        node = j * (grid.nx + 1) + i
        out.append((node, area, n_in, np.full(node.size, float(gamma)),
                    np.full(node.size, group, dtype=np.int64), mid[:, 0], mid[:, 1]))
    return out


# ---- 本体 --------------------------------------------------------------------------


def build_fluid_graph(
    model: GeometryModel, grid: CartesianGrid, cell_state: np.ndarray, vol_gas: np.ndarray
) -> FluidGraph:
    """格子・セル状態 (gpic.geometry.ParticleGeometry.cell_state)・双対セルの気体体積
    (eb.build の vol_gas、2π なし) から流体の輸送グラフを組む。"""
    warnings: list[str] = []
    rz = model.radial_axis() is not None
    two_pi = 2.0 * math.pi if rz else 1.0
    eps = EPS_REL * min(grid.dx, grid.dy)
    vg = np.asarray(vol_gas, dtype=np.float64).ravel()

    X, Y = np.meshgrid(grid.xs, grid.ys)
    active = _near_gas(model, X, Y, eps).ravel() & (vg > 0.0)
    if not np.any(active):
        raise ValueError("流体輸送領域が空です (domain 全体が固体です)")

    parts = _side_walls(model, grid, eps) + _solid_walls(model, grid, eps)
    if parts:
        node, area, nrm, gamma, group, wx, wy = (np.concatenate(a) for a in zip(*parts))
    else:
        node = np.zeros(0, dtype=np.int64)
        area = gamma = wx = wy = np.zeros(0)
        nrm = np.zeros((0, 2))
        group = np.zeros(0, dtype=np.int64)
    has_wall = np.zeros(grid.n_nodes, dtype=bool)
    has_wall[node] = True

    o, orphan = _merge_targets(grid, active, vg, has_wall)
    node_vol = two_pi * np.asarray(o.T @ vg).ravel()
    node_vol[~active] = 0.0
    charge_map = (sp.diags(vg) @ o).tocsr()
    dropped = two_pi * float(vg[orphan].sum())
    gas_volume = two_pi * float(vg.sum())
    if dropped > 1e-3 * gas_volume:
        warnings.append(
            f"流体 2D: 固体に囲まれて輸送節点に併合できない気体の小片が総体積の "
            f"{100.0 * dropped / gas_volume:.2g}% あります (格子幅より狭い隙間。mesh.size を細かくしてください)"
        )

    ps, qs, ws, ls = [], [], [], []
    for axis in (0, 1):
        p, q, w, ln = _edges(model, grid, cell_state, active, axis)
        ps.append(p)
        qs.append(q)
        ws.append(two_pi * w)
        ls.append(ln)

    # 固体内部の節点に落ちた壁は、小片と同じ併合先へ面積を配分する
    sub = o[node].tocoo()
    lost = float(area.sum() - np.sum(area[sub.row] * sub.data)) if area.size else 0.0
    if area.size and lost > 1e-3 * float(area.sum()):
        warnings.append(
            f"流体 2D: 輸送節点に割り当てられない壁が総面積の {100.0 * lost / float(area.sum()):.2g}% あります"
        )
    r = sub.row
    node = sub.col.astype(np.int64)
    area = two_pi * area[r] * sub.data
    nrm, gamma, group, wx, wy = nrm[r], gamma[r], group[r], wx[r], wy[r]

    # 壁向き電場のプローブ: W と、法線の逆向きに 1 格子間隔 (外周ではちょうど隣の節点) 入った A
    lw = 1.0 / np.sqrt((nrm[:, 0] / grid.dx) ** 2 + (nrm[:, 1] / grid.dy) ** 2) if nrm.size else np.zeros(0)
    ax = np.clip(wx - lw * nrm[:, 0], grid.x0, grid.x1) if nrm.size else np.zeros(0)
    ay = np.clip(wy - lw * nrm[:, 1], grid.y0, grid.y1) if nrm.size else np.zeros(0)
    if nrm.size:
        lw = np.maximum((wx - ax) * nrm[:, 0] + (wy - ay) * nrm[:, 1], 0.25 * lw)
    w_idx, w_wt = _bilinear(grid, wx, wy)
    a_idx, a_wt = _bilinear(grid, ax, ay)

    return FluidGraph(
        grid=grid,
        active=active,
        node_vol=node_vol,
        charge_map=charge_map,
        edge_p=np.concatenate(ps),
        edge_q=np.concatenate(qs),
        edge_w=np.concatenate(ws),
        edge_len=np.concatenate(ls),
        wall_node=node,
        wall_area=area,
        wall_normal=nrm,
        wall_gamma=gamma,
        wall_group=group,
        wall_w_idx=w_idx,
        wall_w_wt=w_wt,
        wall_a_idx=a_idx,
        wall_a_wt=a_wt,
        wall_len=lw,
        gas_volume=gas_volume,
        dropped_volume=dropped,
        warnings=warnings,
    )
