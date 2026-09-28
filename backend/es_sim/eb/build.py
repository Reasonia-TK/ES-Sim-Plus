"""埋め込み境界 (EB) 付き節点有限体積の離散化 — 1 レベル分の演算子を組む (prompts/119)。

## 離散化

∇·(ε∇φ) = −ρ を**節点中心の有限体積** (制御体積 = 双対セル) で離散化する。
節点 P の釣り合いは

    Σ_{辺 PQ} G_PQ (φ_P − φ_Q) + Σ_{Dirichlet 結合 k} G_Pk (φ_P − V_k) = q_P

(q_P = ∫_{双対セル} ρ dV)。G は辺の「コンダクタンス」[F/m] (xy は奥行き 1 m あたり、
軸対称は 2π を落とした r 重み — v1 fem.assemble と同じ流儀) で、行列は対称正定値 (SPD)。
5 点ステンシルなので、x 辺 (ny+1, nx) と y 辺 (ny, nx+1) のコンダクタンス配列と
対角 (ny+1, nx+1) だけで演算子が決まる (行列を陽に作らない matrix-free)。

### 導体 (EB Dirichlet): Gibou らの対称ゴーストフルイド法

格子辺 PQ が導体表面と P から距離 θ·L (0 < θ ≤ 1) で交わるとき、P からの流束を
G/θ · (V_k − φ_P) とする (F. Gibou et al., J. Comput. Phys. 176, 205 (2002) の対称版)。
交点は CAD 形状 (多角形・円) と格子辺の**解析的な交点**から求めるので、境界は階段近似
されず 2 次精度になる。θ は THETA_MIN でクリップする (境界が節点にほぼ重なる場合の
条件数悪化を防ぐ。境界位置の誤差は ≤ THETA_MIN·h)。格子線に沿った導体辺 (v1 の構造格子
と同じく節点が境界上に乗る場合) は節点自体が Dirichlet になる (θ = 1 と等価)。

### 誘電体: 半チューブの「線方向に直列・線間で並列」合成

辺 PQ のコンダクタンスは、辺を挟む 2 つの半チューブ (辺を含むセルの半分ずつ) の和。
半チューブが単一材料なら G = ε·A/L (A: 双対面の面積重み、L: 辺長)。材料境界が通る
半チューブは辺に平行な N_STRIPS 本の標本線に分け、各線で ε の**調和平均** (直列) を
境界との厳密な交点から求め、線間は面積重みで**算術平均** (並列) する。界面が辺に垂直な
層状誘電体 (直列) でも平行な場合 (並列) でも、格子と非整合な界面を正しく表す。
ε の材料分類は導体を無視する (geom.model 参照)。

### 軸対称 (rz: y=r、rz_x0: x=r)

面積重み A は r で重み付ける: 横方向が r なら ∫ r dr (軸上の半チューブは ∫_0^{h/2} r dr)、
辺の方向が r (径方向の辺) なら辺中点の r × 幅。体積も ∫ r dA。2π は落とす (v1 と同じ)。

### 外周境界

- Dirichlet 辺: 辺上の節点が Dirichlet (ただし導体の指定が優先)
- Neumann / symmetry: 半双対セル (自然境界)
- periodic: 右端列 (上端行) の節点を左端 (下端) のスレーブとし、辺と半チューブを巻き戻す

## 出力 (LevelOperator)

- cx, cy, diag: 未知節点どうしの結合と対角 (Dirichlet 結合も対角に含む)
- mask: 0 = 未知、1 = 固定 (Dirichlet)、2 = 周期スレーブ
- fixed_group: 固定節点の Dirichlet グループ番号 (-1 = 固定でない)
- (最細レベルのみ) coupling: (節点数 × グループ数) の疎行列 G_Pk。右辺は
  b = coupling @ V(t) + q_static + q_particles。cut_theta/cut_group: 節点ごと 4 方向
  (W, E, S, N) の Dirichlet 結合の θ とグループ (電場の再構成用)。q_static・vol・vol_gas。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..geom.model import EPS0, GeometryModel
from .grid import CartesianGrid

#: ゴーストフルイドの θ の下限 (境界が節点にほぼ重なる場合のクリップ)
THETA_MIN = 1e-3
#: 混合材料の半チューブを分割する標本線の本数
N_STRIPS = 8
#: 混合セルの四分体 (双対セルの 1/4) あたりの標本点数 (1 辺あたり)。ρ・気体体積の積分用
N_SUB = 6
#: 導体の交点の「すぐ先」を分類するための辺長比の微小量
_T_EPS = 1e-7

MASK_UNKNOWN = 0
MASK_FIXED = 1
MASK_SLAVE = 2

# cut_theta / cut_group の方向番号
DIR_W, DIR_E, DIR_S, DIR_N = 0, 1, 2, 3


@dataclass
class LevelOperator:
    """1 レベル分の離散演算子 (ホスト上の numpy 配列)。"""

    grid: CartesianGrid
    periodic_x: bool
    periodic_y: bool
    cx: np.ndarray           # (ny+1, nx) x 辺のコンダクタンス (未知節点どうしの結合のみ)
    cy: np.ndarray           # (ny, nx+1) y 辺
    diag: np.ndarray         # (ny+1, nx+1)
    mask: np.ndarray         # (ny+1, nx+1) uint8
    fixed_group: np.ndarray  # (ny+1, nx+1) int32
    coupling: sp.csr_matrix | None = None
    q_static: np.ndarray | None = None
    vol: np.ndarray | None = None
    vol_gas: np.ndarray | None = None
    cut_theta: np.ndarray | None = None
    cut_group: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def n_unknowns(self) -> int:
        return int(np.count_nonzero(self.mask == MASK_UNKNOWN))


# ---- 補助: 面積重み・ε の直列平均・混合セル検出 --------------------------------------


def _strip_area(coord: str, axis: int, lo: np.ndarray, hi: np.ndarray, r_face: np.ndarray) -> np.ndarray:
    """双対面の一部 (横方向区間 [lo, hi]) の面積重み (xy は幅、軸対称は r 重み、2π なし)。

    axis: 辺の向き (0 = x 辺、1 = y 辺)。r_face: 辺の中点の r (径方向の辺でのみ使う)。
    """
    if coord == "xy":
        return hi - lo
    r_is_transverse = (coord == "rz" and axis == 0) or (coord == "rz_x0" and axis == 1)
    if r_is_transverse:
        return 0.5 * (hi * hi - lo * lo)
    return r_face * (hi - lo)


def _eps_series_lines(
    model: GeometryModel, axis: int, c: np.ndarray, a0: np.ndarray, a1: np.ndarray
) -> np.ndarray:
    """軸平行な標本線群に沿った ε の調和平均 (直列合成)。

    axis=0: 水平線 y=c, x∈[a0,a1]。axis=1: 鉛直線 x=c, y∈[a0,a1]。
    線と誘電体/電荷領域の境界の厳密な交点で区間を切り、各区間の中点の材料で ε を決める。
    """
    n = c.size
    if n == 0:
        return np.zeros(0)
    ks = [np.arange(n), np.arange(n)]
    us = [a0, a1]
    for o in model.others:
        k, u = o.shape.crossings_h(c, a0, a1) if axis == 0 else o.shape.crossings_v(c, a0, a1)
        if k.size:
            ks.append(k)
            us.append(u)
    k = np.concatenate(ks)
    u = np.concatenate(us)
    order = np.lexsort((u, k))
    k, u = k[order], u[order]
    same = k[1:] == k[:-1]
    seg_k = k[:-1][same]
    seg_len = (u[1:] - u[:-1])[same]
    mid = 0.5 * (u[1:] + u[:-1])[same]
    if axis == 0:
        eps = model.eps_at(mid, c[seg_k])
    else:
        eps = model.eps_at(c[seg_k], mid)
    resist = np.bincount(seg_k, weights=seg_len / eps, minlength=n)
    length = a1 - a0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(resist > 0.0, length / np.where(resist > 0.0, resist, 1.0), 0.0)
    # 長さ 0 の線 (発生しない想定) は端点の ε
    zero = length <= 0.0
    if np.any(zero):
        pts = (a0[zero], c[zero]) if axis == 0 else (c[zero], a0[zero])
        out[zero] = model.eps_at(*pts)
    return out


def _mark_boundary_cells(shapes: list, grid: CartesianGrid) -> np.ndarray:
    """形状の境界が通るセル (ny, nx) を True にする (境界上の密な標本点で塗る)。"""
    mixed = np.zeros((grid.ny, grid.nx), dtype=bool)
    spacing = 0.25 * min(grid.dx, grid.dy)
    for s in shapes:
        bx, by = s.boundary_points(spacing)
        i = np.floor((bx - grid.x0) / grid.dx).astype(np.int64)
        j = np.floor((by - grid.y0) / grid.dy).astype(np.int64)
        ok = (i >= 0) & (i < grid.nx) & (j >= 0) & (j < grid.ny)
        mixed[j[ok], i[ok]] = True
    return mixed


# ---- 辺の処理 ----------------------------------------------------------------------


@dataclass
class _EdgeSet:
    """1 方向の辺群 (x 辺または y 辺) の幾何。配列は全て辺数 E の 1 次元。"""

    axis: int
    shape: tuple[int, int]      # 辺配列の形 ((ny+1, nx) または (ny, nx+1))
    valid: np.ndarray           # 処理対象 (周期スレーブ行/列の辺を除く)
    p: np.ndarray               # 端点 P の正準節点番号 (平坦)
    q: np.ndarray               # 端点 Q の正準節点番号
    c_line: np.ndarray          # 辺の横方向座標 (x 辺なら y_j)
    a0: np.ndarray              # 辺の始点座標 (沿う方向)
    a1: np.ndarray
    length: float
    r_face: np.ndarray          # 辺の中点の沿う方向座標 (径方向の辺の面積重み用)
    # 2 つの半チューブ: (存在, セル行, セル列, 横方向下端, 横方向上端)
    halves: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


def _canon_node(i: np.ndarray, j: np.ndarray, grid: CartesianGrid, px: bool, py: bool) -> np.ndarray:
    ii = np.where(i == grid.nx, 0, i) if px else i
    jj = np.where(j == grid.ny, 0, j) if py else j
    return jj * (grid.nx + 1) + ii


def _edge_set(grid: CartesianGrid, axis: int, px: bool, py: bool) -> _EdgeSet:
    nx, ny = grid.nx, grid.ny
    xs, ys = grid.xs, grid.ys
    dx, dy = grid.dx, grid.dy
    if axis == 0:
        J, I = np.meshgrid(np.arange(ny + 1), np.arange(nx), indexing="ij")
        J, I = J.ravel(), I.ravel()
        valid = np.ones(J.size, dtype=bool)
        if py:
            valid &= J != ny
        p = _canon_node(I, J, grid, px, py)
        q = _canon_node(I + 1, J, grid, px, py)
        c_line = ys[J]
        a0, a1 = xs[I], xs[I + 1]
        length = dx
        # 下側の半チューブ: セル (J-1, I)、周期 y で J=0 ならセル (ny-1, I) (上端側)
        ex_lo = (J >= 1) | (py & (J == 0))
        row_lo = np.where(J >= 1, J - 1, ny - 1)
        c_ref_lo = np.where(J >= 1, ys[J], ys[ny])
        halves = [
            (ex_lo, row_lo, I, c_ref_lo - 0.5 * dy, c_ref_lo),
            (J < ny, np.minimum(J, ny - 1), I, c_line, c_line + 0.5 * dy),
        ]
        shape = (ny + 1, nx)
    else:
        J, I = np.meshgrid(np.arange(ny), np.arange(nx + 1), indexing="ij")
        J, I = J.ravel(), I.ravel()
        valid = np.ones(J.size, dtype=bool)
        if px:
            valid &= I != nx
        p = _canon_node(I, J, grid, px, py)
        q = _canon_node(I, J + 1, grid, px, py)
        c_line = xs[I]
        a0, a1 = ys[J], ys[J + 1]
        length = dy
        ex_lo = (I >= 1) | (px & (I == 0))
        col_lo = np.where(I >= 1, I - 1, nx - 1)
        c_ref_lo = np.where(I >= 1, xs[I], xs[nx])
        halves = [
            (ex_lo, J, col_lo, c_ref_lo - 0.5 * dx, c_ref_lo),
            (I < nx, J, np.minimum(I, nx - 1), c_line, c_line + 0.5 * dx),
        ]
        shape = (ny, nx + 1)
    r_face = 0.5 * (a0 + a1)
    return _EdgeSet(axis, shape, valid, p, q, c_line, a0, a1, length, r_face, halves)


def _eps_area(
    model: GeometryModel,
    es: _EdgeSet,
    sel: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    cell_eps: np.ndarray,
    mixed_eps: np.ndarray,
) -> np.ndarray:
    """選択した辺 sel の区間 [t0, t1] (辺長比) の ∫ ε_series dA (2 つの半チューブの和)。"""
    out = np.zeros(sel.size)
    if sel.size == 0:
        return out
    a0 = es.a0[sel] + t0 * es.length
    a1 = es.a0[sel] + t1 * es.length
    for exists, row, col, lo, hi in es.halves:
        ex = exists[sel]
        if not np.any(ex):
            continue
        loc = np.nonzero(ex)[0]
        gi = sel[loc]
        r, c = row[gi], col[gi]
        area = _strip_area(model.coord, es.axis, lo[gi], hi[gi], es.r_face[gi])
        pure = ~mixed_eps[r, c]
        if np.any(pure):
            out[loc[pure]] += cell_eps[r[pure], c[pure]] * area[pure]
        mix = ~pure
        if np.any(mix):
            lm = loc[mix]
            gm = gi[mix]
            n_m = lm.size
            s = (np.arange(N_STRIPS) + 0.5) / N_STRIPS
            lo_m = lo[gm][:, None]
            hi_m = hi[gm][:, None]
            w = (hi_m - lo_m) / N_STRIPS
            s_lo = lo_m + np.arange(N_STRIPS)[None, :] * w
            s_hi = s_lo + w
            c_mid = (lo_m + s[None, :] * (hi_m - lo_m)).ravel()
            aa0 = np.repeat(a0[lm], N_STRIPS)
            aa1 = np.repeat(a1[lm], N_STRIPS)
            eps_s = _eps_series_lines(model, es.axis, c_mid, aa0, aa1).reshape(n_m, N_STRIPS)
            rf = np.repeat(es.r_face[gm], N_STRIPS).reshape(n_m, N_STRIPS)
            a_s = _strip_area(model.coord, es.axis, s_lo, s_hi, rf)
            out[lm] += np.sum(eps_s * a_s, axis=1)
    return out


def _conductor_hits(
    model: GeometryModel, es: _EdgeSet, sel: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """選択した辺と導体境界の交点。戻り値: (辺番号, t (辺長比), 導体番号)。"""
    ks, ts, cs = [], [], []
    c = es.c_line[sel]
    a0 = es.a0[sel]
    a1 = es.a1[sel]
    for ci, cond in enumerate(model.conductors):
        k, u = cond.shape.crossings_h(c, a0, a1) if es.axis == 0 else cond.shape.crossings_v(c, a0, a1)
        if k.size:
            ks.append(sel[k])
            ts.append((u - a0[k]) / es.length)
            cs.append(np.full(k.size, ci, dtype=np.int64))
    if not ks:
        z = np.zeros(0, dtype=np.int64)
        return z, np.zeros(0), z
    return np.concatenate(ks), np.concatenate(ts), np.concatenate(cs)


def _group_at(model: GeometryModel, es: _EdgeSet, e: np.ndarray, t: np.ndarray, fallback_ci: np.ndarray) -> np.ndarray:
    """辺 e の位置 t にある導体の Dirichlet グループ番号 (優先順位適用)。"""
    u = es.a0[e] + t * es.length
    if es.axis == 0:
        ci = model.classify_conductor(u, es.c_line[e])
    else:
        ci = model.classify_conductor(es.c_line[e], u)
    ci = np.where(ci >= 0, ci, fallback_ci)
    return np.asarray(model.conductor_group, dtype=np.int64)[ci]


# ---- 本体 ----------------------------------------------------------------------------


def build_level(model: GeometryModel, grid: CartesianGrid, *, full: bool = True) -> LevelOperator:
    """1 レベル分の演算子を組む。full=True (最細レベル) なら右辺用の情報も作る。"""
    nx, ny = grid.nx, grid.ny
    px, py = model.periodic_x, model.periodic_y
    shape = grid.shape
    n_nodes = grid.n_nodes
    X, Y = np.meshgrid(grid.xs, grid.ys)
    warnings: list[str] = []

    # ---- 1. 固定節点 (外周 Dirichlet 辺 → 導体で上書き) -------------------------------
    fixed_group = np.full(shape, -1, dtype=np.int32)
    side_slices = {
        "bottom": (0, slice(None)),
        "top": (ny, slice(None)),
        "left": (slice(None), 0),
        "right": (slice(None), nx),
    }
    for side, g in model.side_group.items():
        fixed_group[side_slices[side]] = g
    if model.conductors:
        cond = model.classify_conductor(X, Y)
        cg = np.asarray(model.conductor_group, dtype=np.int32)
        hit = cond >= 0
        fixed_group[hit] = cg[cond[hit]]
    # 周期: マスター/スレーブのどちらかが固定なら両方固定 (マスター優先)
    if px:
        m, s = fixed_group[:, 0], fixed_group[:, nx]
        merged = np.where(m >= 0, m, s)
        if np.any((m >= 0) != (s >= 0)):
            warnings.append("周期境界 (左右) の片側だけに Dirichlet 節点があります (両側を固定扱いにしました)")
        fixed_group[:, 0] = merged
        fixed_group[:, nx] = merged
    if py:
        m, s = fixed_group[0, :], fixed_group[ny, :]
        merged = np.where(m >= 0, m, s)
        if np.any((m >= 0) != (s >= 0)):
            warnings.append("周期境界 (上下) の片側だけに Dirichlet 節点があります (両側を固定扱いにしました)")
        fixed_group[0, :] = merged
        fixed_group[ny, :] = merged
    mask = np.where(fixed_group >= 0, MASK_FIXED, MASK_UNKNOWN).astype(np.uint8)
    if px:
        mask[:, nx] = np.where(mask[:, nx] == MASK_FIXED, MASK_FIXED, MASK_SLAVE)
    if py:
        mask[ny, :] = np.where(mask[ny, :] == MASK_FIXED, MASK_FIXED, MASK_SLAVE)
    fixed_flat = (mask == MASK_FIXED).ravel()
    group_flat = fixed_group.ravel()

    # ---- 2. セルの材料 (ε 用、導体は無視) ------------------------------------------
    xc = grid.xs[:-1] + 0.5 * grid.dx
    yc = grid.ys[:-1] + 0.5 * grid.dy
    XC, YC = np.meshgrid(xc, yc)
    cell_mat = model.classify_other(XC, YC)
    cell_eps = EPS0 * model.eps_r_table()[cell_mat + 1]
    mixed_eps = _mark_boundary_cells([o.shape for o in model.others], grid)

    diag = np.zeros(n_nodes)
    cx = np.zeros((ny + 1) * nx)
    cy = np.zeros(ny * (nx + 1))
    coup_rows: list[np.ndarray] = []
    coup_cols: list[np.ndarray] = []
    coup_vals: list[np.ndarray] = []
    cut_theta = np.zeros((4, n_nodes)) if full else None
    cut_group = np.full((4, n_nodes), -1, dtype=np.int64) if full else None

    for axis, carr in ((0, cx), (1, cy)):
        es = _edge_set(grid, axis, px, py)
        e_all = np.nonzero(es.valid)[0]
        p_fixed = fixed_flat[es.p[e_all]]
        q_fixed = fixed_flat[es.q[e_all]]
        live = ~(p_fixed & q_fixed)
        e_all = e_all[live]
        p_fixed, q_fixed = p_fixed[live], q_fixed[live]

        # 導体との交点 (辺ごとの最初/最後)
        n_hit = np.zeros(es.valid.size, dtype=np.int64)
        t_first = np.full(es.valid.size, np.inf)
        t_last = np.full(es.valid.size, -np.inf)
        ci_first = np.full(es.valid.size, -1, dtype=np.int64)
        ci_last = np.full(es.valid.size, -1, dtype=np.int64)
        if model.conductors and e_all.size:
            he, ht, hc = _conductor_hits(model, es, e_all)
            if he.size:
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

        hits = n_hit[e_all] > 0
        # (a) 通常の辺: 両端未知・交点なし
        normal = (~p_fixed) & (~q_fixed) & (~hits)
        en = e_all[normal]
        if en.size:
            g = _eps_area(model, es, en, np.zeros(en.size), np.ones(en.size), cell_eps, mixed_eps) / es.length
            carr[en] = g
            np.add.at(diag, es.p[en], g)
            np.add.at(diag, es.q[en], g)

        # (b) P 側の Dirichlet 結合: P 未知 かつ (Q 固定 または 交点あり)
        for side in ("p", "q"):
            if side == "p":
                need = (~p_fixed) & (q_fixed | hits)
                e = e_all[need]
                if e.size == 0:
                    continue
                has = n_hit[e] > 0
                theta = np.where(has, t_first[e], 1.0)
                grp = np.where(
                    has,
                    _group_at(model, es, e, np.clip(theta + _T_EPS, 0.0, 1.0), np.maximum(ci_first[e], 0))
                    if model.conductors else -1,
                    group_flat[es.q[e]],
                )
                theta = np.maximum(theta, THETA_MIN)
                t0, t1 = np.zeros(e.size), theta
                node = es.p[e]
                dirn = DIR_E if axis == 0 else DIR_N
            else:
                need = (~q_fixed) & (p_fixed | hits)
                e = e_all[need]
                if e.size == 0:
                    continue
                has = n_hit[e] > 0
                theta = np.where(has, 1.0 - t_last[e], 1.0)
                grp = np.where(
                    has,
                    _group_at(model, es, e, np.clip(t_last[e] - _T_EPS, 0.0, 1.0), np.maximum(ci_last[e], 0))
                    if model.conductors else -1,
                    group_flat[es.p[e]],
                )
                theta = np.maximum(theta, THETA_MIN)
                t0, t1 = 1.0 - theta, np.ones(e.size)
                node = es.q[e]
                dirn = DIR_W if axis == 0 else DIR_S
            bad = grp < 0
            if np.any(bad):
                # 固定節点なのにグループ不明 (発生しない想定) は結合しない
                warnings.append(f"Dirichlet グループを特定できない結合が {int(bad.sum())} 本あります (無視しました)")
                e, theta, t0, t1, node, grp = e[~bad], theta[~bad], t0[~bad], t1[~bad], node[~bad], grp[~bad]
            g = _eps_area(model, es, e, t0, t1, cell_eps, mixed_eps) / (es.length * (t1 - t0))
            np.add.at(diag, node, g)
            coup_rows.append(node)
            coup_cols.append(grp)
            coup_vals.append(g)
            if full:
                cut_theta[dirn, node] = theta
                cut_group[dirn, node] = grp

    # ---- 3. 対角・マスクの仕上げ -------------------------------------------------------
    unknown = (mask.ravel() == MASK_UNKNOWN)
    isolated = unknown & (diag <= 0.0)
    if np.any(isolated):
        warnings.append(f"結合を持たない未知節点が {int(isolated.sum())} 個あります (固定 0 V 扱い)")
        m = mask.ravel()
        m[isolated] = MASK_FIXED
        mask = m.reshape(shape)
    diag = np.where(mask.ravel() == MASK_UNKNOWN, diag, 1.0).reshape(shape)

    op = LevelOperator(
        grid=grid,
        periodic_x=px,
        periodic_y=py,
        cx=cx.reshape(ny + 1, nx),
        cy=cy.reshape(ny, nx + 1),
        diag=diag,
        mask=mask,
        fixed_group=fixed_group,
        warnings=warnings,
    )
    if not full:
        return op

    n_groups = len(model.groups)
    if coup_rows:
        rows = np.concatenate(coup_rows)
        cols = np.concatenate(coup_cols)
        vals = np.concatenate(coup_vals)
    else:
        rows = cols = np.zeros(0, dtype=np.int64)
        vals = np.zeros(0)
    op.coupling = sp.csr_matrix((vals, (rows, cols)), shape=(n_nodes, max(n_groups, 1)))
    op.cut_theta = cut_theta.reshape(4, *shape)
    op.cut_group = cut_group.reshape(4, *shape)

    mixed_any = mixed_eps | _mark_boundary_cells([c.shape for c in model.conductors], grid)
    op.vol = _dual_integral(model, grid, None, mixed_any)
    op.vol_gas = _dual_integral(model, grid, model.gas_at, mixed_any)
    if model.has_charge:
        op.q_static = _dual_integral(model, grid, model.rho_at, mixed_any)
    else:
        op.q_static = np.zeros(shape)
    return op


def _dual_integral(model: GeometryModel, grid: CartesianGrid, fn, mixed: np.ndarray) -> np.ndarray:
    """節点の双対セル (domain 内) での ∫ fn dV (軸対称は ∫ fn r dA、2π なし)。

    セルを 4 つの四分体に分け、各四分体を隣接する角の節点へ加える。純セル (境界が通らない)
    はセル中心の値 × 四分体体積、混合セルは N_SUB×N_SUB の中点則で積分する。
    fn=None は fn≡1 (体積そのもの、常に厳密)。周期スレーブへの寄与はマスターへ寄せる。
    """
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    xs, ys = grid.xs, grid.ys
    rad = model.radial_axis()
    out = np.zeros(grid.shape)

    def quad_volume(xlo, xhi, ylo, yhi):
        if rad is None:
            return (xhi - xlo) * (yhi - ylo)
        if rad == 1:
            return (xhi - xlo) * 0.5 * (yhi * yhi - ylo * ylo)
        return (yhi - ylo) * 0.5 * (xhi * xhi - xlo * xlo)

    J, I = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    xm = xs[:-1] + 0.5 * dx
    ym = ys[:-1] + 0.5 * dy
    if fn is not None:
        XC, YC = np.meshgrid(xm, ym)
        center_val = fn(XC, YC).astype(np.float64)
    for a in (0, 1):
        for b in (0, 1):
            xlo = np.where(a == 0, xs[:-1], xm)[None, :] * np.ones((ny, 1))
            xhi = np.where(a == 0, xm, xs[1:])[None, :] * np.ones((ny, 1))
            ylo = np.where(b == 0, ys[:-1], ym)[:, None] * np.ones((1, nx))
            yhi = np.where(b == 0, ym, ys[1:])[:, None] * np.ones((1, nx))
            vq = quad_volume(xlo, xhi, ylo, yhi)
            if fn is None:
                val = vq
            else:
                val = center_val * vq
                if np.any(mixed):
                    jm, im = np.nonzero(mixed)
                    s = (np.arange(N_SUB) + 0.5) / N_SUB
                    sx = xlo[jm, im][:, None, None] + (xhi - xlo)[jm, im][:, None, None] * s[None, None, :]
                    sy = ylo[jm, im][:, None, None] + (yhi - ylo)[jm, im][:, None, None] * s[None, :, None]
                    sx, sy = np.broadcast_arrays(sx, sy)
                    f = fn(sx, sy).astype(np.float64)
                    sub = ((xhi - xlo)[jm, im] * (yhi - ylo)[jm, im] / (N_SUB * N_SUB))[:, None, None]
                    if rad == 1:
                        w = sy
                    elif rad == 0:
                        w = sx
                    else:
                        w = 1.0
                    val[jm, im] = np.sum(f * w * sub, axis=(1, 2))
            np.add.at(out, (J + b, I + a), val)
    if model.periodic_x:
        out[:, 0] += out[:, nx]
        out[:, nx] = out[:, 0]
    if model.periodic_y:
        out[0, :] += out[ny, :]
        out[ny, :] = out[0, :]
    return out
