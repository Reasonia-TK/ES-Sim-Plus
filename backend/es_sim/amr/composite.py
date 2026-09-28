"""合成格子 (AMR) 上の節点有限体積 Poisson — ぶら下がり節点を Pᵀ A P で消去する (prompts/121)。

## 構成

v2 の節点 FV 演算子 (eb.build) は「辺のコンダクタンス = 辺を挟む 2 セルの半チューブの和」なので、
**セルごとの寄与の和**に分解できる。そこで全レベルの葉セル (hierarchy.leaf_cells) について、
各セルの 4 辺それぞれの半チューブの寄与を現行と同じ規則 (誘電体の直列/並列合成・導体の
ゴーストフルイド θ・外周 Dirichlet・軸対称の r 重み) で求め、角の節点どうしの結合として
全体行列 A_full に積む。節点は最細レベルの添字空間 (I, J) で同定する (粗い節点は同じ位置の
細かい節点と同一。周期方向は巻き戻して同一視)。

細かい領域の境界で粗い葉セルの辺の中点にある節点 (ぶら下がり節点) は、その粗い辺の両端の
値で φ_h = w_a φ_a + w_b φ_b (w_a + w_b = 1) と拘束する。重みはぶら下がり節点に細かい側から
接する 2 本の半チューブの ∫ε dA の比 (一様な直交座標では線形補間 w = ½、軸対称の径方向の辺は
½ ∓ h/(8 r_m)、誘電体界面上では ε 比。理由は assemble_composite のコメント)。端点が Dirichlet
節点なら定数項 (電極電位の線形結合) になるので、拘束を φ_U = P φ_c + C V (V: Dirichlet
グループの電位) と書き、

    A_c = Pᵀ A_full P,   b_c = Pᵀ (q + Coup·V − A_full C V)

を解く。A_full が対称正定値なら A_c も対称正定値、Pᵀ による右辺の集約は電荷を保存する
(有限要素のぶら下がり節点消去と同じ構成)。細分化が無ければ A_c は eb.build の一様格子の
演算子と一致する。

## 求解

A_c を陽に疎行列で組み、pyamg の smoothed aggregation 代数マルチグリッドを前処理とする CG で
解く (P4a は CPU。GPU への移植は P4b)。Dirichlet 節点を持たない問題は右辺を和 0 に整合化して
解き、解の平均を 0 にする。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..eb.build import N_STRIPS, N_SUB, THETA_MIN, _T_EPS, _eps_series_lines, _mark_boundary_cells, _strip_area
from ..geom.model import GeometryModel
from .hierarchy import AmrHierarchy

# 半辺の種類 (セルのどの辺か)
_BOTTOM, _TOP, _LEFT, _RIGHT = 0, 1, 2, 3


@dataclass
class _HalfEdges:
    """全レベルの葉セルの半辺 (セル 1 つにつき 4 本)。配列は全て同じ長さ。"""

    lvl: np.ndarray
    kind: np.ndarray
    axis: np.ndarray
    ci: np.ndarray
    cj: np.ndarray
    pI: np.ndarray    # 端点 P (辺の始点) の最細添字 (巻き戻し前)
    pJ: np.ndarray
    qI: np.ndarray    # 端点 Q
    qJ: np.ndarray
    c_line: np.ndarray
    a0: np.ndarray
    a1: np.ndarray
    lo: np.ndarray    # 半チューブの横方向区間
    hi: np.ndarray
    length: np.ndarray
    r_face: np.ndarray


def _collect_half_edges(hier: AmrHierarchy) -> _HalfEdges:
    L = hier.max_level
    parts: dict[str, list[np.ndarray]] = {k: [] for k in _HalfEdges.__dataclass_fields__}
    for lvl in range(hier.n_levels):
        ci, cj = hier.leaf_cells(lvl)
        if ci.size == 0:
            continue
        g = hier.level_grid(lvl)
        s = L - lvl
        xs = g.x0 + (g.x1 - g.x0) * np.arange(g.nx + 1) / g.nx
        ys = g.y0 + (g.y1 - g.y0) * np.arange(g.ny + 1) / g.ny
        dx, dy = g.dx, g.dy
        n = ci.size
        for kind in (_BOTTOM, _TOP, _LEFT, _RIGHT):
            if kind in (_BOTTOM, _TOP):
                jj = cj if kind == _BOTTOM else cj + 1
                pi, pj, qi, qj = ci, jj, ci + 1, jj
                axis = 0
                c_line = ys[jj]
                a0, a1 = xs[ci], xs[ci + 1]
                lo = ys[jj] if kind == _BOTTOM else ys[jj] - 0.5 * dy
                hi = lo + 0.5 * dy
                length = np.full(n, dx)
            else:
                ii = ci if kind == _LEFT else ci + 1
                pi, pj, qi, qj = ii, cj, ii, cj + 1
                axis = 1
                c_line = xs[ii]
                a0, a1 = ys[cj], ys[cj + 1]
                lo = xs[ii] if kind == _LEFT else xs[ii] - 0.5 * dx
                hi = lo + 0.5 * dx
                length = np.full(n, dy)
            parts["lvl"].append(np.full(n, lvl, dtype=np.int64))
            parts["kind"].append(np.full(n, kind, dtype=np.int64))
            parts["axis"].append(np.full(n, axis, dtype=np.int64))
            parts["ci"].append(ci)
            parts["cj"].append(cj)
            parts["pI"].append(pi << s)
            parts["pJ"].append(pj << s)
            parts["qI"].append(qi << s)
            parts["qJ"].append(qj << s)
            parts["c_line"].append(np.asarray(c_line, dtype=np.float64))
            parts["a0"].append(np.asarray(a0, dtype=np.float64))
            parts["a1"].append(np.asarray(a1, dtype=np.float64))
            parts["lo"].append(np.asarray(lo, dtype=np.float64))
            parts["hi"].append(np.asarray(hi, dtype=np.float64))
            parts["length"].append(length)
            parts["r_face"].append(0.5 * (np.asarray(a0) + np.asarray(a1)))
    return _HalfEdges(**{k: np.concatenate(v) if v else np.zeros(0) for k, v in parts.items()})


class _HalfEdgeIndex:
    """(レベル, 辺の種類, セル添字) → 半辺の番号 (_collect_half_edges の並び: レベル → 種類 → 葉セル)。"""

    def __init__(self, hier: AmrHierarchy, he: _HalfEdges):
        self.hier = hier
        self.seg: dict[tuple[int, int], tuple[int, np.ndarray]] = {}
        for lvl in range(hier.n_levels):
            nx_l = hier.base.nx << lvl
            for kind in (_BOTTOM, _TOP, _LEFT, _RIGHT):
                idx = np.nonzero((he.lvl == lvl) & (he.kind == kind))[0]
                if idx.size:
                    # 葉セルは行優先 (j, i) の昇順に並ぶのでキーは昇順
                    self.seg[(lvl, kind)] = (int(idx[0]), he.cj[idx] * nx_l + he.ci[idx])

    def find(self, lvl: int, kind: int, ci: np.ndarray, cj: np.ndarray) -> np.ndarray:
        """該当する葉セルの半辺の番号 (無ければ −1)。周期方向は巻き戻す。"""
        out = np.full(ci.shape, -1, dtype=np.int64)
        seg = self.seg.get((lvl, kind))
        if seg is None:
            return out
        g = self.hier.level_grid(lvl)
        ci = np.mod(ci, g.nx) if self.hier.px else ci
        cj = np.mod(cj, g.ny) if self.hier.py else cj
        ok = (ci >= 0) & (ci < g.nx) & (cj >= 0) & (cj < g.ny)
        start, cell_keys = seg
        k = cj * g.nx + ci
        pos = np.clip(np.searchsorted(cell_keys, k), 0, cell_keys.size - 1)
        hit = ok & (cell_keys[pos] == k)
        out[hit] = start + pos[hit]
        return out


@dataclass
class CompositeOperator:
    """合成格子の演算子と、解の再構成・診断に必要な情報。"""

    hier: AmrHierarchy
    keys: np.ndarray            # (N,) 正準節点キー (昇順)
    xy: np.ndarray              # (N, 2) 節点座標 (正準位置)
    fixed_group: np.ndarray     # (N,) Dirichlet グループ (-1 = 未知)
    u_of_node: np.ndarray       # (N,) 未知番号 (-1 = 固定)
    hanging: np.ndarray         # (nU,) ぶら下がり節点か
    A_full: sp.csr_matrix       # (nU, nU)
    coup_full: sp.csr_matrix    # (nU, K)
    q_full: np.ndarray          # (nU,)
    P: sp.csr_matrix            # (nU, nC)
    C: sp.csr_matrix            # (nU, K)
    A_c: sp.csr_matrix
    coup_c: sp.csr_matrix
    q_c: np.ndarray
    # エネルギー・電荷用: 通常の半辺 (未知番号の組と g) と Dirichlet 結合 (未知番号, グループ, g)
    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_g: np.ndarray
    coup_u: np.ndarray
    coup_grp: np.ndarray
    coup_g: np.ndarray
    n_groups: int
    singular: bool
    setup_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def n_unknowns(self) -> int:
        return int(self.A_c.shape[0])

    def canon(self, I: np.ndarray, J: np.ndarray) -> np.ndarray:
        return _canon_key(self.hier, I, J)

    def node_index(self, keys: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.keys, keys)
        idx = np.clip(idx, 0, self.keys.size - 1)
        if not np.all(self.keys[idx] == keys):
            raise KeyError("合成格子に存在しない節点キーがあります")
        return idx


def _fine_dims(hier: AmrHierarchy) -> tuple[int, int]:
    return hier.base.nx << hier.max_level, hier.base.ny << hier.max_level


def _canon_key(hier: AmrHierarchy, I: np.ndarray, J: np.ndarray) -> np.ndarray:
    nxf, nyf = _fine_dims(hier)
    Iw = np.where(I == nxf, 0, I) if hier.px else I
    Jw = np.where(J == nyf, 0, J) if hier.py else J
    return Jw.astype(np.int64) * (nxf + 1) + Iw.astype(np.int64)


def _coords(hier: AmrHierarchy, I: np.ndarray, J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = hier.base
    nxf, nyf = _fine_dims(hier)
    return b.x0 + (b.x1 - b.x0) * I / nxf, b.y0 + (b.y1 - b.y0) * J / nyf


def _occurrence_groups(model: GeometryModel, hier: AmrHierarchy, I: np.ndarray, J: np.ndarray) -> np.ndarray:
    """節点の出現位置 (巻き戻し前) ごとの Dirichlet グループ (外周 Dirichlet 辺 → 導体で上書き)。"""
    nxf, nyf = _fine_dims(hier)
    grp = np.full(I.shape, -1, dtype=np.int64)
    on_side = {"left": I == 0, "right": I == nxf, "bottom": J == 0, "top": J == nyf}
    for side, gi in model.side_group.items():
        grp[on_side[side]] = gi
    if model.conductors:
        x, y = _coords(hier, I, J)
        c = model.classify_conductor(x, y)
        cg = np.asarray(model.conductor_group, dtype=np.int64)
        hit = c >= 0
        grp[hit] = cg[c[hit]]
    return grp


def _edge_hits(model: GeometryModel, axis: np.ndarray, c_line, a0, a1, length):
    """辺と導体境界の交点 (最初/最後の t と導体番号、交点数)。"""
    n = axis.size
    n_hit = np.zeros(n, dtype=np.int64)
    t_first = np.full(n, np.inf)
    t_last = np.full(n, -np.inf)
    ci_first = np.full(n, -1, dtype=np.int64)
    ci_last = np.full(n, -1, dtype=np.int64)
    if not model.conductors or n == 0:
        return n_hit, t_first, ci_first, t_last, ci_last
    ks, ts, cs = [], [], []
    for ax in (0, 1):
        sel = np.nonzero(axis == ax)[0]
        if sel.size == 0:
            continue
        for ci, cond in enumerate(model.conductors):
            fn = cond.shape.crossings_h if ax == 0 else cond.shape.crossings_v
            k, u = fn(c_line[sel], a0[sel], a1[sel])
            if k.size:
                e = sel[k]
                ks.append(e)
                ts.append((u - a0[e]) / length[e])
                cs.append(np.full(k.size, ci, dtype=np.int64))
    if not ks:
        return n_hit, t_first, ci_first, t_last, ci_last
    he = np.concatenate(ks)
    ht = np.concatenate(ts)
    hc = np.concatenate(cs)
    np.add.at(n_hit, he, 1)
    order = np.lexsort((ht, he))
    he, ht, hc = he[order], ht[order], hc[order]
    first = np.ones(he.size, dtype=bool)
    first[1:] = he[1:] != he[:-1]
    last = np.ones(he.size, dtype=bool)
    last[:-1] = he[1:] != he[:-1]
    t_first[he[first]] = ht[first]
    ci_first[he[first]] = hc[first]
    t_last[he[last]] = ht[last]
    ci_last[he[last]] = hc[last]
    return n_hit, t_first, ci_first, t_last, ci_last


def _group_at(model: GeometryModel, axis, c_line, a0, length, t, fallback_ci) -> np.ndarray:
    u = a0 + t * length
    x = np.where(axis == 0, u, c_line)
    y = np.where(axis == 0, c_line, u)
    ci = model.classify_conductor(x, y)
    ci = np.where(ci >= 0, ci, np.maximum(fallback_ci, 0))
    return np.asarray(model.conductor_group, dtype=np.int64)[ci]


def _eps_area_half(model: GeometryModel, he: _HalfEdges, sel: np.ndarray, t0, t1, pure: np.ndarray,
                   eps_pure: np.ndarray) -> np.ndarray:
    """選択した半辺の区間 [t0, t1] の ∫ε_series dA (その半チューブ 1 つ分)。"""
    out = np.zeros(sel.size)
    if sel.size == 0:
        return out
    axis = he.axis[sel]
    lo, hi, rf = he.lo[sel], he.hi[sel], he.r_face[sel]
    area = np.where(axis == 0, _strip_area(model.coord, 0, lo, hi, rf), _strip_area(model.coord, 1, lo, hi, rf))
    p = pure[sel]
    out[p] = eps_pure[sel][p] * area[p]
    mix = np.nonzero(~p)[0]
    if mix.size:
        gm = sel[mix]
        a0 = he.a0[gm] + t0[mix] * he.length[gm]
        a1 = he.a0[gm] + t1[mix] * he.length[gm]
        s = (np.arange(N_STRIPS) + 0.5) / N_STRIPS
        lo_m, hi_m = he.lo[gm][:, None], he.hi[gm][:, None]
        w = (hi_m - lo_m) / N_STRIPS
        s_lo = lo_m + np.arange(N_STRIPS)[None, :] * w
        s_hi = s_lo + w
        c_mid = (lo_m + s[None, :] * (hi_m - lo_m))
        rfm = np.repeat(he.r_face[gm][:, None], N_STRIPS, axis=1)
        vals = np.zeros(mix.size)
        for ax in (0, 1):
            k = np.nonzero(he.axis[gm] == ax)[0]
            if k.size == 0:
                continue
            eps_s = _eps_series_lines(
                model, ax, c_mid[k].ravel(), np.repeat(a0[k], N_STRIPS), np.repeat(a1[k], N_STRIPS)
            ).reshape(k.size, N_STRIPS)
            a_s = _strip_area(model.coord, ax, s_lo[k], s_hi[k], rfm[k])
            vals[k] = np.sum(eps_s * a_s, axis=1)
        out[mix] = vals
    return out


def _cell_eps_and_purity(model: GeometryModel, hier: AmrHierarchy, he: _HalfEdges):
    """半辺ごとに、その半チューブを含むセルが単一材料か (pure) と、そのセルの ε。"""
    pure = np.ones(he.lvl.size, dtype=bool)
    eps = np.zeros(he.lvl.size)
    others = [o.shape for o in model.others]
    for lvl in range(hier.n_levels):
        sel = np.nonzero(he.lvl == lvl)[0]
        if sel.size == 0:
            continue
        g = hier.level_grid(lvl)
        if others:
            mixed = _mark_boundary_cells(others, g)
            pure[sel] = ~mixed[he.cj[sel], he.ci[sel]]
        xc = g.x0 + (he.ci[sel] + 0.5) * g.dx
        yc = g.y0 + (he.cj[sel] + 0.5) * g.dy
        eps[sel] = model.eps_at(xc, yc)
    return pure, eps


def _static_charge(model: GeometryModel, hier: AmrHierarchy, n_nodes: int, node_of) -> np.ndarray:
    """葉セルの四分体ごとの ∫ρ dV を角の節点へ (節点番号の配列を返す関数 node_of(I, J))。"""
    q = np.zeros(n_nodes)
    if not model.has_charge:
        return q
    rad = model.radial_axis()
    shapes = [c.shape for c in model.conductors] + [o.shape for o in model.others]
    L = hier.max_level
    s_pts = (np.arange(N_SUB) + 0.5) / N_SUB
    for lvl in range(hier.n_levels):
        ci, cj = hier.leaf_cells(lvl)
        if ci.size == 0:
            continue
        g = hier.level_grid(lvl)
        mixed = _mark_boundary_cells(shapes, g)[cj, ci]
        sh = L - lvl
        for a in (0, 1):
            for b in (0, 1):
                xlo = g.x0 + (ci + 0.5 * a) * g.dx
                ylo = g.y0 + (cj + 0.5 * b) * g.dy
                xhi, yhi = xlo + 0.5 * g.dx, ylo + 0.5 * g.dy
                if rad is None:
                    vq = (xhi - xlo) * (yhi - ylo)
                elif rad == 1:
                    vq = (xhi - xlo) * 0.5 * (yhi**2 - ylo**2)
                else:
                    vq = (yhi - ylo) * 0.5 * (xhi**2 - xlo**2)
                val = model.rho_at(0.5 * (xlo + xhi), 0.5 * (ylo + yhi)) * vq
                m = np.nonzero(mixed)[0]
                if m.size:
                    sx = xlo[m][:, None, None] + (xhi - xlo)[m][:, None, None] * s_pts[None, None, :]
                    sy = ylo[m][:, None, None] + (yhi - ylo)[m][:, None, None] * s_pts[None, :, None]
                    sx, sy = np.broadcast_arrays(sx, sy)
                    f = model.rho_at(sx, sy)
                    wr = 1.0 if rad is None else (sy if rad == 1 else sx)
                    sub = ((xhi - xlo)[m] * (yhi - ylo)[m] / N_SUB**2)[:, None, None]
                    val[m] = np.sum(f * wr * sub, axis=(1, 2))
                idx = node_of((ci + a) << sh, (cj + b) << sh)
                np.add.at(q, idx, val)
    return q


def assemble_composite(model: GeometryModel, hier: AmrHierarchy) -> CompositeOperator:
    """合成格子の演算子を組む (ぶら下がり節点の消去まで)。"""
    t0 = time.perf_counter()
    warnings: list[str] = []
    L = hier.max_level
    he = _collect_half_edges(hier)
    n_he = he.lvl.size
    K = max(len(model.groups), 1)

    # ---- 節点 (正準キー) と Dirichlet グループ ---------------------------------------------
    occ_I = np.concatenate([he.pI, he.qI])
    occ_J = np.concatenate([he.pJ, he.qJ])
    occ_key = _canon_key(hier, occ_I, occ_J)
    keys, inv = np.unique(occ_key, return_inverse=True)
    n_nodes = keys.size
    grp_occ = _occurrence_groups(model, hier, occ_I, occ_J)
    fixed_group = np.full(n_nodes, -1, dtype=np.int64)
    np.maximum.at(fixed_group, inv, grp_occ)       # 周期の両側で片方でも固定なら固定
    p_node, q_node = inv[:n_he], inv[n_he:]
    nxf, _ = _fine_dims(hier)
    kI = keys % (nxf + 1)
    kJ = keys // (nxf + 1)
    xy = np.stack(_coords(hier, kI, kJ), axis=1)

    unknown = fixed_group < 0
    u_of_node = np.full(n_nodes, -1, dtype=np.int64)
    u_of_node[unknown] = np.arange(int(unknown.sum()))
    nU = int(unknown.sum())

    # ---- 半辺ごとの寄与 (eb.build.build_level と同じ場合分けを半チューブ単位で) ------------
    pure, eps_cell = _cell_eps_and_purity(model, hier, he)
    pf = fixed_group[p_node] >= 0
    qf = fixed_group[q_node] >= 0
    n_hit, t_first, ci_first, t_last, ci_last = _edge_hits(model, he.axis, he.c_line, he.a0, he.a1, he.length)
    hits = n_hit > 0
    live = ~(pf & qf)

    edge_u, edge_v, edge_g = [], [], []
    normal = np.nonzero(live & ~pf & ~qf & ~hits)[0]
    if normal.size:
        g = _eps_area_half(model, he, normal, np.zeros(normal.size), np.ones(normal.size), pure, eps_cell)
        g = g / he.length[normal]
        edge_u.append(u_of_node[p_node[normal]])
        edge_v.append(u_of_node[q_node[normal]])
        edge_g.append(g)

    coup_u, coup_grp, coup_g = [], [], []
    for side in ("p", "q"):
        if side == "p":
            sel = np.nonzero(live & ~pf & (qf | hits))[0]
            if sel.size == 0:
                continue
            has = hits[sel]
            theta = np.where(has, t_first[sel], 1.0)
            grp = np.where(
                has,
                _group_at(model, he.axis[sel], he.c_line[sel], he.a0[sel], he.length[sel],
                          np.clip(theta + _T_EPS, 0.0, 1.0), ci_first[sel]) if model.conductors else -1,
                fixed_group[q_node[sel]],
            )
            theta = np.maximum(theta, THETA_MIN)
            ta, tb = np.zeros(sel.size), theta
            node = p_node[sel]
        else:
            sel = np.nonzero(live & ~qf & (pf | hits))[0]
            if sel.size == 0:
                continue
            has = hits[sel]
            theta = np.where(has, 1.0 - t_last[sel], 1.0)
            grp = np.where(
                has,
                _group_at(model, he.axis[sel], he.c_line[sel], he.a0[sel], he.length[sel],
                          np.clip(t_last[sel] - _T_EPS, 0.0, 1.0), ci_last[sel]) if model.conductors else -1,
                fixed_group[p_node[sel]],
            )
            theta = np.maximum(theta, THETA_MIN)
            ta, tb = 1.0 - theta, np.ones(sel.size)
            node = q_node[sel]
        bad = grp < 0
        if np.any(bad):
            warnings.append(f"Dirichlet グループを特定できない結合が {int(bad.sum())} 本あります (無視しました)")
            keep = ~bad
            sel, grp, ta, tb, node = sel[keep], grp[keep], ta[keep], tb[keep], node[keep]
        g = _eps_area_half(model, he, sel, ta, tb, pure, eps_cell) / (he.length[sel] * (tb - ta))
        coup_u.append(u_of_node[node])
        coup_grp.append(grp)
        coup_g.append(g)

    cat = lambda xs, dt=np.float64: np.concatenate(xs) if xs else np.zeros(0, dtype=dt)  # noqa: E731
    eu, ev, eg = cat(edge_u, np.int64), cat(edge_v, np.int64), cat(edge_g)
    cu, cgp, cgv = cat(coup_u, np.int64), cat(coup_grp, np.int64), cat(coup_g)

    rows = np.concatenate([eu, ev, eu, ev, cu])
    cols = np.concatenate([eu, ev, ev, eu, cu])
    vals = np.concatenate([eg, eg, -eg, -eg, cgv])
    A_full = sp.csr_matrix((vals, (rows, cols)), shape=(nU, nU))
    coup_full = sp.csr_matrix((cgv, (cu, cgp)), shape=(nU, K))

    def node_of(I, J):
        idx = np.searchsorted(keys, _canon_key(hier, I, J))
        return idx

    q_nodes = _static_charge(model, hier, n_nodes, node_of)
    q_full = q_nodes[unknown]

    # ---- ぶら下がり節点 --------------------------------------------------------------------
    # 拘束 φ_h = w_a φ_a + (1 − w_a) φ_b。Pᵀ は同じ重みでぶら下がり節点の方程式 (細かい側から
    # 入る流束) を端点へ配る。その流束が通る面 (ぶら下がり節点に細かい側から接する 2 本の半チューブ)
    # は粗い側の双対セルの境目 (粗い辺の中点) で a 側と b 側に分かれるので、重みを 2 本の半チューブの
    # ∫ε dA (軸対称は r 重み) の比 w_a = A_a / (A_a + A_b) にする。直交座標の一様な ε なら線形補間
    # (½)、軸対称の径方向の辺なら ½ − h/(8 r_m)、誘電体界面がぶら下がり節点を通るなら ε 比。
    # 線形補間のままだと、軸 (r = 0) に接する界面や誘電体界面で界面に平行な一様場の流束収支が
    # 合わず、隣の節点に O(h) の誤差が出る (tests/test_v2_amr.py)。
    hidx = _HalfEdgeIndex(hier, he)
    hang_u, hang_a, hang_b, hang_w = [], [], [], []
    for lvl in range(L):
        ci, cj = hier.leaf_cells(lvl)
        if ci.size == 0:
            continue
        s = L - lvl
        h = 1 << (s - 1)
        # (ぶら下がり節点, 端点 a, 端点 b, a 側の細かいセルと半辺の種類, b 側)
        for mI, mJ, aI, aJ, bI, bJ, (fai, faj, ka), (fbi, fbj, kb) in (
            # 下辺 (細かい側は下): a 側 = 左下のセルの右辺、b 側 = 右下のセルの左辺
            ((2 * ci + 1) * h, cj << s, ci << s, cj << s, (ci + 1) << s, cj << s,
             (2 * ci, 2 * cj - 1, _RIGHT), (2 * ci + 1, 2 * cj - 1, _LEFT)),
            # 上辺 (細かい側は上)
            ((2 * ci + 1) * h, (cj + 1) << s, ci << s, (cj + 1) << s, (ci + 1) << s, (cj + 1) << s,
             (2 * ci, 2 * cj + 2, _RIGHT), (2 * ci + 1, 2 * cj + 2, _LEFT)),
            # 左辺 (細かい側は左): a 側 = 左下のセルの上辺、b 側 = 左上のセルの下辺
            (ci << s, (2 * cj + 1) * h, ci << s, cj << s, ci << s, (cj + 1) << s,
             (2 * ci - 1, 2 * cj, _TOP), (2 * ci - 1, 2 * cj + 1, _BOTTOM)),
            # 右辺 (細かい側は右)
            ((ci + 1) << s, (2 * cj + 1) * h, (ci + 1) << s, cj << s, (ci + 1) << s, (cj + 1) << s,
             (2 * ci + 2, 2 * cj, _TOP), (2 * ci + 2, 2 * cj + 1, _BOTTOM)),
        ):
            mk = _canon_key(hier, mI, mJ)
            pos = np.clip(np.searchsorted(keys, mk), 0, n_nodes - 1)
            present = keys[pos] == mk
            present[present] = unknown[pos[present]]
            sel = np.nonzero(present)[0]
            if sel.size == 0:
                continue
            ia = hidx.find(lvl + 1, ka, fai[sel], faj[sel])
            ib = hidx.find(lvl + 1, kb, fbi[sel], fbj[sel])
            w = np.full(sel.size, 0.5)
            good = np.nonzero((ia >= 0) & (ib >= 0))[0]
            if good.size:
                both = np.concatenate([ia[good], ib[good]])
                area = _eps_area_half(model, he, both, np.zeros(both.size), np.ones(both.size), pure, eps_cell)
                area_a, area_b = area[: good.size], area[good.size:]
                tot = area_a + area_b
                w[good] = np.where(tot > 0.0, area_a / np.where(tot > 0.0, tot, 1.0), 0.5)
            hang_u.append(u_of_node[pos[sel]])
            hang_a.append(np.searchsorted(keys, _canon_key(hier, aI[sel], aJ[sel])))
            hang_b.append(np.searchsorted(keys, _canon_key(hier, bI[sel], bJ[sel])))
            hang_w.append(w)
    hanging = np.zeros(nU, dtype=bool)
    if hang_u:
        hu = np.concatenate(hang_u)
        ha = np.concatenate(hang_a)
        hb = np.concatenate(hang_b)
        hw = np.concatenate(hang_w)
        hu, first = np.unique(hu, return_index=True)
        ha, hb, hw = ha[first], hb[first], hw[first]
        hanging[hu] = True
    else:
        hu = ha = hb = np.zeros(0, dtype=np.int64)
        hw = np.zeros(0)

    # ---- 拘束 φ_U = P φ_c + C V -------------------------------------------------------------
    comp = ~hanging
    c_of_u = np.full(nU, -1, dtype=np.int64)
    c_of_u[comp] = np.arange(int(comp.sum()))
    nC = int(comp.sum())
    pr, pc, pv = [np.nonzero(comp)[0]], [c_of_u[comp]], [np.ones(nC)]
    cr, cc, cv = [], [], []
    for ends, wts in ((ha, hw), (hb, 1.0 - hw)):
        eu_ = u_of_node[ends]
        is_fixed = eu_ < 0
        if np.any(eu_[~is_fixed] >= 0) and np.any(hanging[eu_[~is_fixed]]):
            raise RuntimeError("ぶら下がり節点の端点がぶら下がり節点です (proper nesting が崩れています)")
        k = ~is_fixed
        pr.append(hu[k])
        pc.append(c_of_u[eu_[k]])
        pv.append(wts[k])
        cr.append(hu[is_fixed])
        cc.append(fixed_group[ends[is_fixed]])
        cv.append(wts[is_fixed])
    P = sp.csr_matrix((np.concatenate(pv), (np.concatenate(pr), np.concatenate(pc))), shape=(nU, nC))
    C = sp.csr_matrix((cat(cv), (cat(cr, np.int64), cat(cc, np.int64))), shape=(nU, K))

    PT = P.T.tocsr()
    A_c = (PT @ A_full @ P).tocsr()
    coup_c = (PT @ (coup_full - A_full @ C)).tocsr()
    q_c = PT @ q_full

    return CompositeOperator(
        hier=hier, keys=keys, xy=xy, fixed_group=fixed_group, u_of_node=u_of_node, hanging=hanging,
        A_full=A_full, coup_full=coup_full, q_full=q_full, P=P, C=C, A_c=A_c, coup_c=coup_c, q_c=q_c,
        edge_u=eu, edge_v=ev, edge_g=eg, coup_u=cu, coup_grp=cgp, coup_g=cgv,
        n_groups=len(model.groups), singular=not bool(np.any(fixed_group >= 0)),
        setup_s=time.perf_counter() - t0, warnings=warnings,
    )


@dataclass
class AmrSolveInfo:
    iterations: int
    relative_residual: float
    converged: bool
    elapsed_s: float


def solve_composite(op: CompositeOperator, v_groups: np.ndarray, *, tol: float = 1e-10,
                    max_iter: int = 500) -> tuple[np.ndarray, AmrSolveInfo]:
    """A_c x = q_c + coup_c V を AMG-CG で解き、全節点の電位 (固定節点は V) を返す。"""
    import pyamg

    t0 = time.perf_counter()
    vg = np.asarray(v_groups, dtype=np.float64)
    vK = np.zeros(op.coup_c.shape[1])
    vK[: vg.size] = vg
    b = op.q_c + op.coup_c @ vK
    if op.singular and b.size:
        b = b - b.mean()
    if b.size == 0:
        x = np.zeros(0)
        info = AmrSolveInfo(0, 0.0, True, 0.0)
    else:
        ml = pyamg.smoothed_aggregation_solver(op.A_c, symmetry="symmetric", max_coarse=500)
        res: list[float] = []
        x = ml.solve(b, tol=tol, accel="cg", maxiter=max_iter, residuals=res)
        if op.singular:
            x = x - x.mean()
        bn = float(np.linalg.norm(b))
        rel = float(np.linalg.norm(b - op.A_c @ x)) / bn if bn > 0 else 0.0
        info = AmrSolveInfo(max(len(res) - 1, 0), rel, rel <= max(tol * 10, 1e-12), time.perf_counter() - t0)
    phi_u = op.P @ x + op.C @ vK
    phi = np.empty(op.keys.size)
    fixed = op.fixed_group >= 0
    phi[fixed] = vK[op.fixed_group[fixed]]
    phi[~fixed] = phi_u[op.u_of_node[~fixed]]
    return phi, info


def energy_and_charges(model: GeometryModel, op: CompositeOperator, phi: np.ndarray,
                       v_groups: np.ndarray) -> tuple[float, np.ndarray]:
    """二次形式のエネルギーと電極ごとの誘起電荷。軸対称は 2π 倍。

    電荷 = Dirichlet 結合のフラックス + 端点が Dirichlet のぶら下がり節点の拘束反力 (Cᵀ r、
    r = A_full φ_U − Coup·V − q は拘束で満たされない節点方程式の残差)。後者を入れると
    Q_k = ∂W/∂V_k となり、総電荷の保存 (ΣQ = −Σq) と 2W = Σ Q_k V_k が厳密に成り立つ。
    """
    unk = np.nonzero(op.fixed_group < 0)[0]
    phi_u = np.empty(unk.size)
    phi_u[op.u_of_node[unk]] = phi[unk]
    vg = np.asarray(v_groups, dtype=np.float64)
    d = phi_u[op.edge_u] - phi_u[op.edge_v]
    w = 0.5 * float(np.sum(op.edge_g * d * d))
    dc = phi_u[op.coup_u] - vg[op.coup_grp] if op.coup_u.size else np.zeros(0)
    w += 0.5 * float(np.sum(op.coup_g * dc * dc))
    q = np.bincount(op.coup_grp, weights=op.coup_g * (-dc), minlength=len(vg)) if len(vg) else np.zeros(0)
    if op.C.nnz and len(vg):
        vK = np.zeros(op.coup_full.shape[1])
        vK[: vg.size] = vg
        r = op.A_full @ phi_u - op.coup_full @ vK - op.q_full
        q = q + (op.C.T @ r)[: vg.size]
    factor = 2.0 * np.pi if model.radial_axis() is not None else 1.0
    return factor * w, factor * q
