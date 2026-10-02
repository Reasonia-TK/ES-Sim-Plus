"""Fowler–Nordheim (FN) 電界放出 (prompts/46)。

Murphy-Good 型 FN 式 + Forbes (2006) の Nordheim 関数近似で、電極表面の
局所電界 F [V/m] から放出電流密度 J [A/m^2] を計算する:

    f    = c F / φ²                       (c = e/(4πε0) = 1.439964e-9 V m)
    v(f) ≈ 1 − f + (f/6) ln f             (Forbes の単純良近似)
    t(f) ≈ 1 + f/9 − (f/18) ln f
    J    = A F² / (φ t(f)²) · exp(−B φ^{3/2} v(f) / F)

    A = 1.541434e-6 A eV V⁻²、B = 6.830890e9 eV^{-3/2} V m⁻¹
    φ: 仕事関数 [eV]、F: β·(表面幾何電界)、β: 電界増倍係数

放出面は電極 (Dirichlet) 表面の境界メッシュエッジ列として表す。放出方向は
メッシュ内部 (真空側) への単位法線。表面電界は隣接要素の E = −∇φ の法線成分
のうち「電子を真空側へ引き出す向き」(E·n̂ < 0、n̂ = 放出方向) のみを使う。

trace (固定場) と PIC (毎ステップの場) の両方から共用する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .meshing import Mesh, _region_rings
from .schema import FnEmission, Project

# FN 定数 (Forbes & Deane 2007 の標準値)
FN_A = 1.541434e-6   # [A eV V^-2]
FN_B = 6.830890e9    # [eV^-3/2 V m^-1]
FN_C = 1.439964e-9   # e/(4πε0) [V m] — Schottky 障壁低下比 f = c F / φ²


def fn_current_density(f_surf: np.ndarray, phi_ev: float, beta: float = 1.0) -> np.ndarray:
    """表面幾何電界 F [V/m] (≤0 は放出なし) から FN 電流密度 J [A/m²] を返す。

    ベクトル化されており、F ≤ 0 の要素は J = 0。f = cβF/φ² は 1 でクリップする
    (f ≥ 1 は障壁が完全に消えた極限。v(1) = 0 で指数項が 1 になる)。
    """
    f_local = beta * np.asarray(f_surf, dtype=np.float64)
    j = np.zeros_like(f_local)
    pos = f_local > 0.0
    if not np.any(pos):
        return j
    fp = f_local[pos]
    f = np.minimum(FN_C * fp / (phi_ev * phi_ev), 1.0)
    lnf = np.log(f)  # fp > 0 なので f > 0
    v = 1.0 - f + (f / 6.0) * lnf
    t = 1.0 + f / 9.0 - (f / 18.0) * lnf
    j[pos] = (
        FN_A * fp * fp / (phi_ev * t * t)
        * np.exp(-FN_B * phi_ev**1.5 * v / fp)
    )
    return j


@dataclass
class FnSurface:
    """FN 放出面 (電極表面の境界メッシュエッジ列) の前計算テーブル。"""

    elem: np.ndarray    # (S,) 隣接要素 (真空側)
    pa: np.ndarray      # (S, 2) エッジ端点1
    pb: np.ndarray      # (S, 2) エッジ端点2
    mid: np.ndarray     # (S, 2) 中点
    nrm: np.ndarray     # (S, 2) 放出方向 (真空側の単位法線)
    length: np.ndarray  # (S,) エッジ長
    delta: np.ndarray   # (S,) 境界からわずかに内側へ置くオフセット量
    n1: np.ndarray | None = None  # (S,) 端点1 (pa) の節点番号 (阻止コンデンサの電極の電荷の行き先、prompts/134)
    n2: np.ndarray | None = None  # (S,) 端点2 (pb) の節点番号


def _both_on_segment(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray, tol: float
) -> np.ndarray:
    """メッシュエッジ両端 p1, p2 (K,2) が線分 q1-q2 上にあるか (K,) bool。"""
    seg = q2 - q1
    seg_len = float(np.hypot(seg[0], seg[1]))
    if seg_len <= 0.0:
        return np.zeros(len(p1), dtype=bool)

    def _on(p: np.ndarray) -> np.ndarray:
        d = p - q1
        dist = np.abs(d[:, 0] * seg[1] - d[:, 1] * seg[0]) / seg_len
        t = (d[:, 0] * seg[0] + d[:, 1] * seg[1]) / (seg_len * seg_len)
        return (dist <= tol) & (t >= -1e-9) & (t <= 1.0 + 1e-9)

    return _on(p1) & _on(p2)


def build_fn_surface(
    project: Project, mesh: Mesh, adjacency: np.ndarray, fn: FnEmission
) -> FnSurface | None:
    """FN 放出面テーブルを構築する。該当エッジが無ければ None。

    対象: 境界メッシュエッジ (隣接 = -1) のうち、両端節点がともに Dirichlet
    (電極) 節点で、かつ指定ソース上にあるもの:
      - fn.edges:   domain 外周のエッジ番号 (頂点 i → i+1 の線分上)
      - fn.regions: conductor 領域 id (領域輪郭ポリゴンの線分上。穴の輪郭も含む)
    """
    tris = mesh.triangles
    nodes = mesh.nodes
    ts, loc = np.nonzero(adjacency == -1)
    n1 = tris[ts, (loc + 1) % 3]
    n2 = tris[ts, (loc + 2) % 3]
    n_opp = tris[ts, loc]
    p1, p2, po = nodes[n1], nodes[n2], nodes[n_opp]

    # 両端がともに Dirichlet (電極) 節点であること
    dir_mask = np.zeros(len(nodes), dtype=bool)
    if mesh.dirichlet:
        dir_mask[np.fromiter(mesh.dirichlet.keys(), dtype=np.int64)] = True
    cand = dir_mask[n1] & dir_mask[n2]

    scale = float(np.max(np.abs(nodes))) if len(nodes) else 1.0
    tol = 1e-8 * (scale if scale > 0.0 else 1.0)

    sel = np.zeros(len(ts), dtype=bool)
    poly = np.asarray(project.geometry.domain.polygon, dtype=np.float64)
    nv = len(poly)
    for e in fn.edges:
        sel |= _both_on_segment(p1, p2, poly[e % nv], poly[(e + 1) % nv], tol)

    if fn.regions:
        local = {ls.region: ls.size for ls in project.mesh.local_sizes}
        by_id = {r.id: r for r in project.geometry.regions}
        for rid in fn.regions:
            region = by_id.get(rid)
            if region is None:
                raise ValueError(f"fn.regions の領域 id '{rid}' が見つかりません")
            if region.type != "conductor":
                raise ValueError(f"fn.regions の領域 '{rid}' は conductor ではありません")
            # 円は多角形化してから照合する (メッシュはこの多角形に沿っている)。穴の輪郭も放出面
            for ring in _region_rings(region, local.get(rid, project.mesh.size)):
                rpoly = np.asarray(ring, dtype=np.float64)
                for i in range(len(rpoly)):
                    sel |= _both_on_segment(p1, p2, rpoly[i], rpoly[(i + 1) % len(rpoly)], tol)

    sel &= cand
    if not np.any(sel):
        return None

    p1s, p2s, pos_ = p1[sel], p2[sel], po[sel]
    mid = 0.5 * (p1s + p2s)
    t_vec = p2s - p1s
    perp = np.stack([-t_vec[:, 1], t_vec[:, 0]], axis=1)
    # 放出方向 = 対頂点の側 (メッシュ内部 = 真空側)。domain 外周電極でも
    # conductor 穴の輪郭でも、隣接要素は常に真空側にあるためこの規約で正しい
    sgn = np.where(np.sum(perp * (pos_ - mid), axis=1) >= 0.0, 1.0, -1.0)
    length = np.linalg.norm(perp, axis=1)
    nrm = perp * (sgn / length)[:, None]
    h = np.abs(np.sum((pos_ - mid) * nrm, axis=1))  # エッジから対頂点までの高さ

    return FnSurface(
        elem=ts[sel],
        pa=p1s,
        pb=p2s,
        mid=mid,
        nrm=nrm,
        length=length,
        delta=1e-3 * h,
        n1=n1[sel],
        n2=n2[sel],
    )


def fn_segment_currents(
    surf: FnSurface,
    e_elem: np.ndarray,
    fn: FnEmission,
    coord: str = "xy",
) -> tuple[np.ndarray, np.ndarray]:
    """各放出セグメントの表面幾何電界 F [V/m] と放出電流 I を返す。

    e_elem: (M, 2) 要素ごとの E = −∇φ。
    I の単位: 平面2D (xy) = [A/m] (奥行き1m換算)、軸対称 (rz / rz_x0) = [A]
    (2πr̄ を掛けて周方向に積分)。
    """
    e_at = np.asarray(e_elem)[surf.elem]
    en = np.sum(e_at * surf.nrm, axis=1)
    # 電子 (電荷 -e) を真空側 (+n̂) へ引き出すのは E·n̂ < 0 の場のみ
    f_surf = np.maximum(0.0, -en)
    j = fn_current_density(f_surf, fn.phi_ev, fn.beta)
    current = j * surf.length
    ridx = {"rz": 1, "rz_x0": 0}.get(coord)
    if ridx is not None:
        current = current * (2.0 * np.pi * surf.mid[:, ridx])
    return f_surf, current


def distribute_particles(current: np.ndarray, n: int) -> np.ndarray:
    """総数 n のマクロ粒子を電流比例で各セグメントへ配分する (最大剰余法)。

    電流 0 のセグメントには配分しない。全電流 0 なら全ゼロを返す。
    """
    counts = np.zeros(len(current), dtype=np.int64)
    total = float(current.sum())
    if total <= 0.0 or n <= 0:
        return counts
    quota = current / total * n
    counts = np.floor(quota).astype(np.int64)
    rem = n - int(counts.sum())
    if rem > 0:
        # 剰余の大きい順に1個ずつ配る (同値は np.argsort の安定順で決定的)
        order = np.argsort(-(quota - counts), kind="stable")
        counts[order[:rem]] += 1
    # 電流 0 のセグメントへ剰余が回らないよう保証 (quota=0 → 剰余0 なので通常発生しない)
    counts[current <= 0.0] = 0
    return counts
