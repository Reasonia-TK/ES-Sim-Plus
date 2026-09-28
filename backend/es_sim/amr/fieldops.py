"""合成格子の節点ステンシル: 電場 E = G φ_all + D V と、誤差指標の二階差分 (prompts/122)。

一様格子の GPU PIC (kernels/pic.cu の es_deriv) と同じ差分を、各節点を角に持つ最も細かい
レベルの格子幅 h で組む:

- ±h の点 m が節点ならその値、節点でなければ (粗細界面の粗い側) m を含む葉セルの双一次補間。
  整数の最細添字で厳密に位置を決めるので、線形場では厳密。
- 未知節点: 線分 [n, m] が導体境界と交わるなら交点 (距離 θh、θ ≥ THETA_MIN) の導体電位を
  使う (ゴーストフルイド)。両側が揃えば不等間隔の中心差分、片側しか無い (外周 Neumann) なら 0。
- 固定節点: 未知の隣への片側差分 (隣から見た交点までの距離 θh) の平均 (es_deriv と同じ)。

電場は列が [φ_all (N 個); V (K 個)] の疎行列 G_x, G_y (N × (N+K)) で、E = G [φ; V]。
細分化の無い階層 (L = 0) では一様格子の efield_nodes と一致する (tests/test_v2_amr_pic.py)。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..eb.build import THETA_MIN, _T_EPS
from ..geom.model import GeometryModel
from .composite import CompositeOperator, _canon_key, _coords, _edge_hits, _fine_dims, _group_at
from .hierarchy import AmrHierarchy
from .locate import leaf_level_map, locate_fine


def node_levels(hier: AmrHierarchy, op: CompositeOperator) -> np.ndarray:
    """各節点を角に持つ葉セルの最大レベル (その節点の格子幅を決める)。"""
    lev = np.full(op.keys.size, -1, dtype=np.int64)
    L = hier.max_level
    for lvl in range(hier.n_levels):
        ci, cj = hier.leaf_cells(lvl)
        if ci.size == 0:
            continue
        s = L - lvl
        for a in (0, 1):
            for b in (0, 1):
                idx = op.node_index(_canon_key(hier, (ci + a) << s, (cj + b) << s))
                np.maximum.at(lev, idx, lvl)
    return lev


def _point_value(hier: AmrHierarchy, op: CompositeOperator, Im: np.ndarray, Jm: np.ndarray, leaves):
    """最細添字の格子点 (Im, Jm) (巻き戻し済み) の電位を節点の線形結合で表す: (cols (n,4), w (n,4))。"""
    n = Im.size
    cols = np.zeros((n, 4), dtype=np.int64)
    w = np.zeros((n, 4))
    key = _canon_key(hier, Im, Jm)
    pos = np.clip(np.searchsorted(op.keys, key), 0, op.keys.size - 1)
    is_node = op.keys[pos] == key
    cols[is_node, 0] = pos[is_node]
    w[is_node, 0] = 1.0
    other = np.nonzero(~is_node)[0]
    if other.size:
        lvl, i, j, wx, wy = locate_fine(hier, Im[other].astype(np.float64), Jm[other].astype(np.float64), leaves)
        s = (hier.max_level - lvl).astype(np.int64)
        for c, (a, b, wgt) in enumerate(((0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)),
                                         (0, 1, (1 - wx) * wy), (1, 1, wx * wy))):
            cols[other, c] = op.node_index(_canon_key(hier, (i + a) << s, (j + b) << s))
            w[other, c] = wgt
    return cols, w, is_node


def _neighbor_sides(model: GeometryModel, hier: AmrHierarchy, op: CompositeOperator, lev: np.ndarray):
    """軸ごと・側 (±1) ごとの隣の点 m = n ± h e の情報。戻り値: [(h, {+1: dict, −1: dict}), (…)]。

    dict: exists (domain 内または周期), cols/w (m の電位の節点結合), m_is_node, m_fixed,
    hit (線分 [n, m] が導体境界と交わる), th_n/th_m (n/m から最初の交点までの割合、≥ THETA_MIN),
    grp (n 側の最初の交点の導体グループ)。
    """
    N = op.keys.size
    L = hier.max_level
    nxf, nyf = _fine_dims(hier)
    kI = op.keys % (nxf + 1)
    kJ = op.keys // (nxf + 1)
    step = (np.int64(1) << (L - lev)).astype(np.int64)
    b = hier.base
    dxf = (b.x1 - b.x0) / nxf
    dyf = (b.y1 - b.y0) / nyf
    fixed = op.fixed_group >= 0
    leaves = leaf_level_map(hier)
    xn, yn = _coords(hier, kI, kJ)
    out = []
    for axis in (0, 1):
        n_ax = nxf if axis == 0 else nyf
        per = hier.px if axis == 0 else hier.py
        h = step * (dxf if axis == 0 else dyf)
        sides = {}
        for sgn in (1, -1):
            c_ax = (kI if axis == 0 else kJ) + sgn * step
            exists = np.ones(N, dtype=bool) if per else (c_ax >= 0) & (c_ax <= n_ax)
            c_wrap = np.mod(c_ax, n_ax) if per else np.clip(c_ax, 0, n_ax)
            Im = c_wrap if axis == 0 else kI
            Jm = kJ if axis == 0 else c_wrap
            cols, w, m_is_node = _point_value(hier, op, Im, Jm, leaves)
            m_fixed = m_is_node & fixed[cols[:, 0]]
            # 導体との交差: 軸方向の区間 [a0, a1] (a0 < a1) で調べる
            c_line = yn if axis == 0 else xn
            a_n = xn if axis == 0 else yn
            a_m = a_n + sgn * h
            a0 = np.minimum(a_n, a_m)
            a1 = np.maximum(a_n, a_m)
            ax = np.full(N, axis, dtype=np.int64)
            n_hit, t_first, ci_first, t_last, ci_last = _edge_hits(model, ax, c_line, a0, a1, h)
            hit = exists & (n_hit > 0)
            if sgn > 0:
                th_n = np.where(hit, t_first, 1.0)          # n から最初の交点
                th_m = np.where(hit, 1.0 - t_last, 1.0)     # m から最初の交点
                t_in, ci_in = t_first + _T_EPS, ci_first
            else:
                th_n = np.where(hit, 1.0 - t_last, 1.0)
                th_m = np.where(hit, t_first, 1.0)
                t_in, ci_in = t_last - _T_EPS, ci_last
            grp = np.full(N, -1, dtype=np.int64)
            if np.any(hit) and model.conductors:
                k = np.nonzero(hit)[0]
                grp[k] = _group_at(model, ax[k], c_line[k], a0[k], h[k], np.clip(t_in[k], 0.0, 1.0),
                                   np.maximum(ci_in[k], 0))
            sides[sgn] = dict(exists=exists, cols=cols, w=w, m_is_node=m_is_node, m_fixed=m_fixed, hit=hit,
                              th_n=np.maximum(th_n, THETA_MIN), th_m=np.maximum(th_m, THETA_MIN), grp=grp)
        out.append((h, sides))
    return out


class _Triplets:
    def __init__(self):
        self.r, self.c, self.v = [], [], []

    def add(self, r, c, v):
        keep = v != 0.0
        self.r.append(r[keep])
        self.c.append(c[keep])
        self.v.append(v[keep])

    def csr(self, shape) -> sp.csr_matrix:
        cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dtype=dt)  # noqa: E731
        return sp.csr_matrix((cat(self.v, np.float64), (cat(self.r, np.int64), cat(self.c, np.int64))), shape=shape)


def build_efield_stencil(model: GeometryModel, hier: AmrHierarchy, op: CompositeOperator,
                         levels: np.ndarray | None = None) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """節点電場のステンシル (G_x, G_y)。形 (N, N+K)、E = G [φ_all; V]。"""
    N = op.keys.size
    K = max(op.coup_full.shape[1], 1)
    lev = levels if levels is not None else node_levels(hier, op)
    fixed = op.fixed_group >= 0
    out = []
    for h, sides in _neighbor_sides(model, hier, op, lev):
        t = _Triplets()
        # ---- 未知節点: 不等間隔中心差分 ----------------------------------------------------
        side_val = {}
        for sgn in (1, -1):
            sd = sides[sgn]
            ghost = sd["hit"] & (sd["grp"] >= 0)
            avail = sd["exists"] & (ghost | ~sd["hit"])
            side_val[sgn] = (avail, ghost, np.where(ghost, sd["th_n"] * h, h))
        r = np.nonzero(~fixed & side_val[1][0] & side_val[-1][0])[0]
        if r.size:
            hp, hm = side_val[1][2][r], side_val[-1][2][r]
            den = hp * hm * (hp + hm)
            # E = −dφ/dx、dφ/dx = [hm² f₊ + (hp² − hm²) f₀ − hp² f₋] / den
            coef = {1: -hm * hm / den, -1: hp * hp / den}
            t.add(r, r, -(hp * hp - hm * hm) / den)
            for sgn in (1, -1):
                sd = sides[sgn]
                ghost = side_val[sgn][1][r]
                c_s = coef[sgn]
                g = np.nonzero(ghost)[0]
                t.add(r[g], N + sd["grp"][r[g]], c_s[g])
                o = np.nonzero(~ghost)[0]
                for c in range(4):
                    t.add(r[o], sd["cols"][r[o], c], c_s[o] * sd["w"][r[o], c])
        # ---- 固定節点: 未知の隣への片側差分の平均 ----------------------------------------------
        fx_nodes = np.nonzero(fixed)[0]
        if fx_nodes.size:
            usable = {sgn: sides[sgn]["exists"][fx_nodes] & ~sides[sgn]["m_fixed"][fx_nodes] for sgn in (1, -1)}
            cnt = usable[1].astype(np.int64) + usable[-1].astype(np.int64)
            for sgn in (1, -1):
                sd = sides[sgn]
                ok = usable[sgn] & (cnt > 0)
                r = fx_nodes[ok]
                if r.size == 0:
                    continue
                dist = sd["th_m"][r] * h[r]
                # 寄与 = sgn·(f_m − f_n)/dist を平均し、E = −平均
                c_s = -sgn / (dist * cnt[ok])
                t.add(r, r, -c_s)
                for c in range(4):
                    t.add(r, sd["cols"][r, c], c_s * sd["w"][r, c])
        out.append(t.csr((N, N + K)))
    return out[0], out[1]


def build_second_difference(model: GeometryModel, hier: AmrHierarchy, op: CompositeOperator,
                            levels: np.ndarray | None = None) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """誤差指標用の二階差分 D2 φ = φ(n+h) − 2φ(n) + φ(n−h) (軸ごと、N × N)。

    未知節点で両隣があり導体と交わらない所だけ行を持つ (他は 0)。滑らかな φ では ≈ h² φ''
    (双一次補間の誤差の 8 倍)。粗細界面の粗い側の点は葉セルの補間なので、補間誤差も拾う。
    """
    N = op.keys.size
    lev = levels if levels is not None else node_levels(hier, op)
    fixed = op.fixed_group >= 0
    out = []
    for _h, sides in _neighbor_sides(model, hier, op, lev):
        t = _Triplets()
        ok = ~fixed
        for sgn in (1, -1):
            ok &= sides[sgn]["exists"] & ~sides[sgn]["hit"]
        r = np.nonzero(ok)[0]
        if r.size:
            t.add(r, r, np.full(r.size, -2.0))
            for sgn in (1, -1):
                sd = sides[sgn]
                for c in range(4):
                    t.add(r, sd["cols"][r, c], sd["w"][r, c])
        out.append(t.csr((N, N)))
    return out[0], out[1]
