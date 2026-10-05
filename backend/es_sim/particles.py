"""荷電粒子軌道追跡 (フェーズ2)。仕様書 §8 参照。

- 粒子位置 → 所属三角形の特定: walk 探索 + 前回要素キャッシュ
- E 補間: P1 要素なので要素内一定 (将来、節点平均場の重心座標補間に変更可)
- 積分器: リープフロッグ (kick-drift-kick / velocity-Verlet 形式)。一様磁場 (b_field、平面 2D のみ) は
  Boris 回転 (_boris_matrix、prompts/51)。軸対称 (rz / rz_x0) の移動は回転法
  (局所座標の3D直線移動 → 子午面へ回転、PIC・DSMC と同じ)
- 全粒子を numpy 一括で進め、walk 探索は Numba (_numba_kernels.walk_step。Numba が無ければ同じ結果の
  numpy 実装)。CPU だけで動く (v2 の GPU には移していない)
- 電極・外周到達で吸収し、衝突位置・エネルギーを記録。対称境界は鏡面反射、周期境界は反対側へ移す
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import _numba_kernels
from .fem import Solution, _radial_index
from .fn import build_fn_surface, distribute_particles, fn_segment_currents
from .meshing import Mesh
from .schema import Emitter, ParticleSettings, Project, Species

QE = 1.602176634e-19   # 電気素量 [C]
ME = 9.1093837015e-31  # 電子質量 [kg]
MP = 1.67262192369e-27 # 陽子質量 [kg]

# walk 探索の反復上限 (1ステップの移動量が要素サイズに比べて十分小さければ
# ほぼ1〜2回で収束する。安全のため上限を設ける)
_MAX_WALK_ITERS = 64
# 重心座標が「内部」とみなす許容誤差 (数値誤差対策)
_TOL = 1e-9


@dataclass
class TraceOutput:
    """trace() の生データ (numpy)。server.py で TraceResult(pydantic) に変換する。"""

    trajectories: np.ndarray       # (n_particles, n_frames, 2)  初期位置を含む
    absorbed: np.ndarray           # (n_particles,) bool
    tof: np.ndarray                # (n_particles,) float。alive なら nan
    final_energy_ev: np.ndarray    # (n_particles,)
    final_angle_deg: np.ndarray    # (n_particles,) 最終速度の向き [度] (atan2(vy, vx))
    dt: float                      # 実際に使った dt [s]
    # FN 電界放出 (prompts/46、fn 指定時のみ非 None)
    currents: np.ndarray | None = None  # (n_particles,) 粒子ごとの担持電流
    fn_current: float | None = None     # 総放出電流 (xy: [A/m]、rz: [A])


def b_vector(project: Project) -> np.ndarray | None:
    """project.b_field を (3,) ベクトル [T] にする。未指定・全成分 0 なら None。"""
    b = project.b_field
    if b is None or b.is_zero():
        return None
    return np.array([b.bx, b.by, b.bz], dtype=np.float64)


def _boris_matrix(q: float, m: float, dt: float, b: np.ndarray) -> np.ndarray:
    """Boris 回転の 3×3 行列 R を返す (v⁺ = R v⁻、prompts/51)。

    t = (q dt / 2m) B、s = 2t/(1+|t|²) として
        v' = v⁻ + v⁻×t、v⁺ = v⁻ + v'×s
    は線形写像なので、一様磁場・固定 dt では行列を前計算できる。
    v×t = -[t]×v より R = I - [s]× + [s]×[t]× (厳密にノルム保存)。
    """
    t = (q * dt / (2.0 * m)) * b
    s = 2.0 * t / (1.0 + float(t @ t))

    def cross_mat(a: np.ndarray) -> np.ndarray:
        return np.array(
            [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]]
        )

    st = cross_mat(s)
    return np.eye(3) - st + st @ cross_mat(t)


def _species_qm(species: Species) -> tuple[float, float]:
    """粒子種から (q, m) を決定する。"""
    if species.preset == "electron":
        return -QE, ME
    if species.preset == "proton":
        return QE, MP
    assert species.q is not None and species.m is not None  # schema で保証済み
    return species.q, species.m


# ---- メッシュ前処理 ---------------------------------------------------------


def _barycentric_coeffs(nodes: np.ndarray, tris: np.ndarray):
    """P1 重心座標 L_i(x,y) = (a_i + b_i x + c_i y) / det の係数を全要素一括で返す。

    (a, b, c, det) はいずれも (M,) または (M,3) の配列。
    det は符号付き2倍面積 (三角形の頂点順序に依存)。向きに関わらず
    L_i は分子・分母を同じ符号規約で計算すれば正しい値になる。
    """
    p = nodes[tris]                      # (M, 3, 2)
    x, y = p[:, :, 0], p[:, :, 1]
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)
    a = np.stack([
        x[:, 1] * y[:, 2] - x[:, 2] * y[:, 1],
        x[:, 2] * y[:, 0] - x[:, 0] * y[:, 2],
        x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0],
    ], axis=1)
    det = x[:, 0] * b[:, 0] + x[:, 1] * b[:, 1] + x[:, 2] * b[:, 2]
    return a, b, c, det


def _adjacency(triangles: np.ndarray) -> np.ndarray:
    """隣接三角形配列を構築する。

    adjacency[t, i] = 要素 t の頂点 i の対辺 (頂点 i+1, i+2) を挟んで
    隣接する要素番号。境界 (隣接要素なし) なら -1。
    メッシュ全体で1回だけ行う前処理なので、要素数分の python ループで実装する
    (粒子ループではないので制約に抵触しない)。
    """
    m = len(triangles)
    edge_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for t in range(m):
        v0, v1, v2 = int(triangles[t, 0]), int(triangles[t, 1]), int(triangles[t, 2])
        # local i の対辺は (i+1, i+2) 番目の頂点
        local_edges = ((v1, v2), (v2, v0), (v0, v1))
        for i, (u, v) in enumerate(local_edges):
            key = (u, v) if u < v else (v, u)
            edge_map.setdefault(key, []).append((t, i))

    adjacency = np.full((m, 3), -1, dtype=np.int64)
    for lst in edge_map.values():
        if len(lst) == 2:
            (t0, i0), (t1, i1) = lst
            adjacency[t0, i0] = t1
            adjacency[t1, i1] = t0
    return adjacency


def _locate_initial(coeffs, points: np.ndarray) -> np.ndarray:
    """初期所属要素を全要素総当たりで決定する (初回のみ)。"""
    a, b, c, det = coeffs
    x = points[:, 0].reshape(-1, 1, 1)
    y = points[:, 1].reshape(-1, 1, 1)
    # L: (n_particles, n_elements, 3)
    l_all = (a[None, :, :] + b[None, :, :] * x + c[None, :, :] * y) / det[None, :, None]
    inside = np.all(l_all >= -_TOL, axis=2)
    any_inside = inside.any(axis=1)
    elem = np.where(any_inside, np.argmax(inside, axis=1), 0)
    if not np.all(any_inside):
        # 領域外 (エミッタ設定ミス等) のフォールバック: 違反が最小の要素を採用
        violation = np.clip(-l_all, 0.0, None).max(axis=2)
        fallback = np.argmin(violation, axis=1)
        elem = np.where(any_inside, elem, fallback)
    return elem.astype(np.int64)


def _pack_coeffs(coeffs) -> np.ndarray:
    """重心座標係数を (M, 10) = [a|b|c|det] に詰める (walk の gather を1回にする)。"""
    a, b, c, det = coeffs
    return np.concatenate([a, b, c, det[:, None]], axis=1)


def _walk_step(
    coeffs,
    adjacency: np.ndarray,
    elem0: np.ndarray,
    x_new: np.ndarray,
    l_out: np.ndarray | None = None,
    packed: np.ndarray | None = None,
):
    """walk の実装切り替え (numba あり/なし)。

    毎呼び出しで _numba_kernels.HAVE_NUMBA を見て分岐する (import 時に一度だけ
    束縛しないのは、テストで monkeypatch してフォールバック経路を強制できる
    ようにするため、prompts/76)。両実装はビット単位で同じ結果を返す
    (test_particles.py の等価性テストで担保)。
    """
    if _numba_kernels.HAVE_NUMBA:
        return _numba_kernels.walk_step(coeffs, adjacency, elem0, x_new, l_out, packed)
    return _walk_step_numpy(coeffs, adjacency, elem0, x_new, l_out, packed)


def _walk_step_numpy(
    coeffs,
    adjacency: np.ndarray,
    elem0: np.ndarray,
    x_new: np.ndarray,
    l_out: np.ndarray | None = None,
    packed: np.ndarray | None = None,
):
    """現在の要素から x_new へ向けて重心座標 walk を行う (numpy ベクトル化実装)。

    戻り値:
        elem: 新しい所属要素 (吸収された粒子は最後にいた要素のまま)
        absorbed: 境界 (adjacency == -1) を越えて出たかどうか
        b_elem, b_loc: absorbed 時に境界を検出した要素・ローカルエッジ番号
                       (tof 補間に使う。absorbed=False の要素では無意味)

    l_out に (n, 3) 配列を渡すと、最終所属要素での重心座標を書き込む
    (電荷堆積での再計算を省くキャッシュ用。absorbed 粒子の行は未定義)。
    packed には _pack_coeffs(coeffs) の前計算を渡せる (ループ呼び出しの高速化)。
    """
    if packed is None:
        packed = _pack_coeffs(coeffs)
    n = len(x_new)
    elem = elem0.copy()
    absorbed = np.zeros(n, dtype=bool)
    b_elem = np.zeros(n, dtype=np.int64)
    b_loc = np.zeros(n, dtype=np.int64)

    # アクティブ集合はインデックス配列で直接引き継ぐ (毎反復の全粒子 nonzero を回避)。
    # 反復0 は全粒子が対象なので idx=None として fancy index のコピーを省く
    idx: np.ndarray | None = None
    for _ in range(_MAX_WALK_ITERS):
        if idx is None:
            xp, yp = x_new[:, 0], x_new[:, 1]
            e_cur = elem
        else:
            if idx.size == 0:
                break
            xp, yp = x_new[idx, 0], x_new[idx, 1]
            e_cur = elem[idx]
        # 係数は (M,10) 詰め込み配列から1回の gather で取得する (a[e] 等の4回より高速)
        g = packed[e_cur]
        # 加算順は (a + b·x) + c·y のまま in-place で温存する (ビット一致のため)
        l_loc = g[:, 0:3] + g[:, 3:6] * xp[:, None]
        l_loc += g[:, 6:9] * yp[:, None]
        l_loc /= g[:, 9:10]  # (K,3)
        # 内外判定は列ごとの比較で行い (axis 縮約より高速)、
        # argmin は外に出た少数の粒子に限定する
        outside = (
            (l_loc[:, 0] < -_TOL) | (l_loc[:, 1] < -_TOL) | (l_loc[:, 2] < -_TOL)
        )

        if not np.any(outside):
            # 全員が現要素内 (イオン等で頻出の高速パス): マスクコピーなしで書き出す
            if l_out is not None:
                if idx is None:
                    np.copyto(l_out, l_loc)
                else:
                    l_out[idx] = l_loc
            idx = np.zeros(0, dtype=np.int64)
            break

        inside = ~outside
        if l_out is not None:
            done = np.nonzero(inside)[0] if idx is None else idx[inside]
            if done.size:
                l_out[done] = l_loc[inside]

        o_idx = np.nonzero(outside)[0] if idx is None else idx[outside]
        o_elem = elem[o_idx]
        o_loc = np.argmin(l_loc[outside], axis=1)
        neighbor = adjacency[o_elem, o_loc]
        left = neighbor == -1

        abs_idx = o_idx[left]
        if abs_idx.size:
            absorbed[abs_idx] = True
            b_elem[abs_idx] = o_elem[left]
            b_loc[abs_idx] = o_loc[left]

        idx = o_idx[~left]
        if idx.size:
            elem[idx] = neighbor[~left]
            # 次の反復で新しい要素を再チェック
    else:
        idx = np.zeros(0, dtype=np.int64) if idx is None else idx

    if l_out is not None and idx.size:
        # 反復上限に達して残った粒子 (稀): 最終要素で重心座標を計算
        g = packed[elem[idx]]
        l_out[idx] = (
            g[:, 0:3]
            + g[:, 3:6] * x_new[idx, 0][:, None]
            + g[:, 6:9] * x_new[idx, 1][:, None]
        ) / g[:, 9:10]

    return elem, absorbed, b_elem, b_loc


# ---- 固体領域 (粒子が侵入できない要素) の判定 (prompts/24) ---------------------


def _solid_elements(project: Project, mesh: Mesh) -> np.ndarray | None:
    """粒子が侵入できない固体要素のマスク (M,) を返す。固体が無ければ None。

    領域種別ごとの粒子透過性:
      - conductor: 穴 (要素が無いので従来通り壁として吸収)
      - dielectric: 固体 (場は εr 付きで解くが、粒子は表面で吸収する)
      - charge: 透過 (空間電荷雲なので従来通り通過)
    """
    solid = np.zeros(len(mesh.triangles), dtype=bool)
    for i, region in enumerate(project.geometry.regions):
        if region.type == "dielectric":
            solid |= mesh.tri_region == i
    return solid if bool(solid.any()) else None


# ---- 境界メッシュエッジの分類 (対称反射・周期ラップ、prompts/22) ---------------


@dataclass
class _BoundaryTables:
    """境界メッシュエッジ (adjacency == -1) の反射/周期分類表。

    いずれも (要素, ローカルエッジ) で引ける形。対象エッジが無いフィールドは None。
    """

    reflect: np.ndarray | None = None       # (M, 3) bool 鏡面反射エッジ
    refl_normal: np.ndarray | None = None   # (M, 3, 2) 内向き単位法線
    refl_point: np.ndarray | None = None    # (M, 3, 2) エッジ上の基準点 (符号付き距離用)
    periodic: np.ndarray | None = None      # (M, 3) bool 周期エッジ
    shift: np.ndarray | None = None         # (M, 3, 2) 周期ラップの平行移動ベクトル


def _build_boundary_tables(
    polygon,
    mesh: Mesh,
    adjacency: np.ndarray,
    reflect_edges,
    periodic_pairs,
) -> _BoundaryTables | None:
    """domain 外周エッジの指定から境界メッシュエッジの分類表を構築する。

    reflect_edges: 鏡面反射する domain エッジ番号列 (symmetry / pic.reflect_edges)
    periodic_pairs: 周期境界の domain エッジ番号対 [(e1, e2), ...]

    境界メッシュエッジの両端節点が指定 domain エッジ (頂点 i → i+1 の線分) 上に
    乗っている場合に対象とする。分類対象が無ければ None を返す
    (従来の吸収のみの経路と完全に一致させるため)。
    """
    if not reflect_edges and not periodic_pairs:
        return None
    poly = np.asarray(polygon, dtype=np.float64)
    nv = len(poly)
    tris = mesh.triangles
    nodes = mesh.nodes
    ts, loc = np.nonzero(adjacency == -1)
    n1 = tris[ts, (loc + 1) % 3]
    n2 = tris[ts, (loc + 2) % 3]
    n_opp = tris[ts, loc]
    p1, p2, po = nodes[n1], nodes[n2], nodes[n_opp]
    scale = float(np.max(np.abs(poly)))
    tol = 1e-8 * (scale if scale > 0.0 else 1.0)

    def _on_edge(e: int) -> np.ndarray:
        q1 = poly[e % nv]
        q2 = poly[(e + 1) % nv]
        seg = q2 - q1
        seg_len = float(np.hypot(seg[0], seg[1]))
        if seg_len <= 0.0:
            return np.zeros(len(ts), dtype=bool)

        def _on_segment(p: np.ndarray) -> np.ndarray:
            # 線分 q1-q2 への共線判定 (距離 tol 以内) + パラメータ範囲チェック
            d = p - q1
            dist = np.abs(d[:, 0] * seg[1] - d[:, 1] * seg[0]) / seg_len
            t = (d[:, 0] * seg[0] + d[:, 1] * seg[1]) / (seg_len * seg_len)
            return (dist <= tol) & (t >= -1e-9) & (t <= 1.0 + 1e-9)

        return _on_segment(p1) & _on_segment(p2)

    m = len(tris)
    tables = _BoundaryTables()

    # 内向き単位法線 = 対頂点の側 (反射・周期の両方で使う規約)
    t_vec = p2 - p1
    perp = np.stack([-t_vec[:, 1], t_vec[:, 0]], axis=1)
    mid = 0.5 * (p1 + p2)
    sgn = np.where(np.sum(perp * (po - mid), axis=1) >= 0.0, 1.0, -1.0)
    nrm = perp * (sgn / np.linalg.norm(perp, axis=1))[:, None]

    if reflect_edges:
        on_reflect = np.zeros(len(ts), dtype=bool)
        for e in reflect_edges:
            on_reflect |= _on_edge(e)
        if np.any(on_reflect):
            tables.reflect = np.zeros((m, 3), dtype=bool)
            tables.refl_normal = np.zeros((m, 3, 2))
            tables.refl_point = np.zeros((m, 3, 2))
            sel = on_reflect
            tables.reflect[ts[sel], loc[sel]] = True
            tables.refl_normal[ts[sel], loc[sel]] = nrm[sel]
            tables.refl_point[ts[sel], loc[sel]] = p1[sel]

    if periodic_pairs:
        on_per = np.zeros(len(ts), dtype=bool)
        shift = np.zeros((len(ts), 2))
        for e1, e2 in periodic_pairs:
            m1 = 0.5 * (poly[e1 % nv] + poly[(e1 + 1) % nv])
            m2 = 0.5 * (poly[e2 % nv] + poly[(e2 + 1) % nv])
            for ea, sh in ((e1, m2 - m1), (e2, m1 - m2)):
                on = _on_edge(ea)
                if np.any(on):
                    shift[on] = sh
                    on_per |= on
        if np.any(on_per):
            tables.periodic = np.zeros((m, 3), dtype=bool)
            tables.shift = np.zeros((m, 3, 2))
            tables.periodic[ts[on_per], loc[on_per]] = True
            tables.shift[ts[on_per], loc[on_per]] = shift[on_per]

    if tables.reflect is None and tables.periodic is None:
        return None
    return tables


def _apply_trace_boundaries(
    tables: _BoundaryTables,
    coeffs,
    packed: np.ndarray,
    adjacency: np.ndarray,
    x_new: np.ndarray,
    v_new: np.ndarray,
    elem: np.ndarray,
    absorbed: np.ndarray,
    b_elem: np.ndarray,
    b_loc: np.ndarray,
) -> None:
    """壁に達した粒子のうち、対称エッジは鏡面反射・周期エッジはラップする。

    全引数を in-place 更新する。反射は位置を境界線について折り返し、速度の
    法線成分 (vx, vy) を反転する。ラップは位置を周期ベクトル分平行移動して
    反対側へ移す (速度不変)。処理後に所属要素を再特定し、コーナーで別の壁に
    達した粒子は次の反復で処理する。対象外の壁に達した粒子は absorbed のまま
    残す (通常の吸収として扱われる)。
    """
    for _ in range(8):
        idx = np.nonzero(absorbed)[0]
        if idx.size == 0:
            return
        changed = False

        if tables.reflect is not None:
            refl = tables.reflect[b_elem[idx], b_loc[idx]]
            if np.any(refl):
                r_idx = idx[refl]
                ea, el = b_elem[r_idx], b_loc[r_idx]
                nrm = tables.refl_normal[ea, el]
                # 符号付き距離 d < 0 = 境界の外側。折り返して境界内へ戻す
                d = np.sum((x_new[r_idx] - tables.refl_point[ea, el]) * nrm, axis=1)
                x_new[r_idx] -= 2.0 * np.minimum(d, 0.0)[:, None] * nrm
                vn = np.sum(v_new[r_idx, :2] * nrm, axis=1)
                v_new[r_idx, :2] -= 2.0 * vn[:, None] * nrm
                e2, a2, be2, bl2 = _walk_step(
                    coeffs, adjacency, ea, x_new[r_idx], packed=packed
                )
                elem[r_idx] = e2
                absorbed[r_idx] = a2
                b_elem[r_idx] = be2
                b_loc[r_idx] = bl2
                changed = True
                idx = np.nonzero(absorbed)[0]
                if idx.size == 0:
                    return

        if tables.periodic is not None:
            per = tables.periodic[b_elem[idx], b_loc[idx]]
            if np.any(per):
                p_idx = idx[per]
                ea, el = b_elem[p_idx], b_loc[p_idx]
                x_new[p_idx] += tables.shift[ea, el]
                # ラップした少数粒子のみ総当たりで所属要素を再特定し、walk で確定
                elem0 = _locate_initial(coeffs, x_new[p_idx])
                e2, a2, be2, bl2 = _walk_step(
                    coeffs, adjacency, elem0, x_new[p_idx], packed=packed
                )
                elem[p_idx] = e2
                absorbed[p_idx] = a2
                b_elem[p_idx] = be2
                b_loc[p_idx] = bl2
                changed = True

        if not changed:
            return


# ---- エミッタ・dt推定 -------------------------------------------------------


def _init_particles(
    emitter: Emitter, m: float, vtheta: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """エミッタ設定から初期位置・初期速度を作る。

    energy_dist == "mono" (既定) の場合は乱数を使わず等間隔割り振り (従来動作)。
    energy_dist == "maxwell" の場合はドリフト速度 (energy_ev, direction_deg から。
    spread_deg は無視) に熱速度成分 (2D Maxwell分布) を加算する。乱数は
    np.random.default_rng(seed) を使い、同じ seed で完全再現する。

    vtheta=True (rz 軸対称モード用) では速度を3成分 (vz, vr, vθ) で返す:
    maxwell は第3成分 vθ も熱速度から抽選し、mono は vθ = 0 とする。
    既定 (False) は従来通り (n, 2) を返す (既存呼び出しは完全不変)。
    """
    n = emitter.n
    if emitter.kind == "point":
        p1 = np.asarray(emitter.p1, dtype=np.float64)
        pos = np.tile(p1, (n, 1))
    else:
        p1 = np.asarray(emitter.p1, dtype=np.float64)
        p2 = np.asarray(emitter.p2, dtype=np.float64)
        t = np.linspace(0.0, 1.0, n)
        pos = p1[None, :] + t[:, None] * (p2 - p1)[None, :]

    n_comp = 3 if vtheta else 2
    if emitter.energy_dist == "maxwell":
        # ドリフト速度は全粒子共通 (spread_deg は無視)。vθ 方向のドリフトは無し
        angle = math.radians(emitter.direction_deg)
        speed = math.sqrt(2.0 * emitter.energy_ev * QE / m) if emitter.energy_ev > 0 else 0.0
        drift = speed * np.array([math.cos(angle), math.sin(angle), 0.0][:n_comp])
        vel = np.tile(drift, (n, 1))

        # 熱速度: 各成分 ~ Normal(0, sigma), sigma = sqrt(kT * q_e / m)
        sigma = math.sqrt(emitter.temperature_ev * QE / m)
        rng = np.random.default_rng(emitter.seed)
        vel = vel + rng.normal(0.0, sigma, size=(n, n_comp))
        return pos, vel

    if n == 1:
        angles_deg = np.array([emitter.direction_deg])
    else:
        angles_deg = np.linspace(
            emitter.direction_deg - emitter.spread_deg,
            emitter.direction_deg + emitter.spread_deg,
            n,
        )
    angles = np.radians(angles_deg)

    speed = math.sqrt(2.0 * emitter.energy_ev * QE / m) if emitter.energy_ev > 0 else 0.0
    vel = speed * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    if vtheta:
        vel = np.column_stack([vel, np.zeros(n)])  # mono の vθ は 0
    return pos, vel


def _estimate_dt(mesh: Mesh, e_field: np.ndarray, v0: np.ndarray, q: float, m: float) -> float:
    """dt 自動推定。

    h_min: 最小要素の外接半径 (= 辺長の積 / (4×面積))。
    v = max(初期速度, sqrt(2|q|E_max h_min / m))、dt = 0.3 h_min / v。
    """
    p = mesh.nodes[mesh.triangles]
    d0 = np.linalg.norm(p[:, 1] - p[:, 2], axis=1)
    d1 = np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
    d2 = np.linalg.norm(p[:, 0] - p[:, 1], axis=1)
    x, y = p[:, :, 0], p[:, :, 1]
    det = x[:, 0] * (y[:, 1] - y[:, 2]) + x[:, 1] * (y[:, 2] - y[:, 0]) + x[:, 2] * (y[:, 0] - y[:, 1])
    area = 0.5 * np.abs(det)
    circum_r = (d0 * d1 * d2) / (4.0 * area)
    h_min = float(circum_r.min())

    e_abs = np.sqrt(np.sum(e_field ** 2, axis=1))
    e_max = float(e_abs.max()) if len(e_abs) else 0.0

    v0_max = float(np.max(np.sqrt(np.sum(v0 ** 2, axis=1)))) if len(v0) else 0.0
    v_th = math.sqrt(2.0 * abs(q) * e_max * h_min / m) if e_max > 0.0 else 0.0
    v_rep = max(v0_max, v_th, 1e-9)  # ゼロ割回避 (静止・無電場の極端なケース)
    return 0.3 * h_min / v_rep


def _rz_rotate(
    xr: np.ndarray, yt: np.ndarray, vr: np.ndarray, vt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """軸対称の回転法: 局所直交座標で (径 xr、周 yt) へ動いた点の半径と、そこでの (vr, vθ) を返す。

    半径 r' = √(xr² + yt²) (≥ 0)。速度の (径, 周) 成分を位置と同じ角
    (cos = xr/r'、sin = yt/r') だけ回して、動いた先の局所座標の成分に直す
    (r'·vθ' = xr·vt − yt·vr)。r' = 0 (軸の上) では回さない。
    演算順は v2 の GPU PIC (kernels/pic.cu の push) の回転と同じ。
    """
    r_new = np.sqrt(xr * xr + yt * yt)
    moved = r_new > 0.0
    r_div = np.where(moved, r_new, 1.0)
    cos_a = np.where(moved, xr / r_div, 1.0)
    sin_a = np.where(moved, yt / r_div, 0.0)
    return r_new, cos_a * vr + sin_a * vt, -sin_a * vr + cos_a * vt


# ---- 本体 -------------------------------------------------------------------


def trace(project: Project, mesh: Mesh, sol: Solution) -> TraceOutput:
    """フェーズ1の解 (固定場) を使って粒子軌道を積分する。"""
    settings: ParticleSettings | None = project.particles
    if settings is None:
        raise ValueError("project.particles が指定されていません")

    if settings.fn is not None:
        q, m = -QE, ME  # FN 電界放出は常に電子 (species は無視)
    else:
        q, m = _species_qm(settings.species)

    tris = mesh.triangles
    coeffs = _barycentric_coeffs(mesh.nodes, tris)
    adjacency = _adjacency(tris)
    e_field = sol.e_field  # (M, 2) 要素内一定

    # 対称 (鏡面反射)・周期 (ラップ) 境界の分類表 (指定が無ければ None = 従来経路)
    refl_edges: set[int] = set()
    periodic_pairs: list[tuple[int, int]] = []
    for bc in project.geometry.boundaries:
        if bc.type == "symmetry":
            refl_edges.update(bc.edges)
        elif bc.type == "periodic":
            periodic_pairs.append((bc.edges[0], bc.edges[1]))
    tables = _build_boundary_tables(
        project.geometry.domain.polygon, mesh, adjacency,
        sorted(refl_edges), periodic_pairs,
    )

    # 誘電体 (固体) 要素のマスク (無ければ None = 従来経路)
    solid = _solid_elements(project, mesh)

    # 軸対称モード (prompts/39, 41): 面内速度2成分 + 第3成分 vθ。
    # 径方向座標インデックス ridx (rz: y=1、rz_x0: x=0) で径成分を一般化する。
    # 速度は粒子の位置の局所直交座標 (軸・径・周) の成分で、移動は回転法 (v1/v2 の
    # PIC・DSMC と同じ): E だけで半キック → 局所座標の3D直線移動 (径 r + vr·dt、
    # 周 vθ·dt) → 新しい半径 r' = √((r + vr·dt)² + (vθ·dt)²) へ移し (vr, vθ) を同じ角
    # だけ回す → 新しい位置の E で半キック (_rz_rotate)。角運動量 r·vθ を厳密に保存し、
    # 無電場では3D直線運動を丸め誤差でなぞる。r ≥ 0 のままなので軸 (境界メッシュ
    # エッジ) で吸収されず、vθ = 0 では軸の鏡映 (r → −r、vr → −vr) と一致する。
    # 遠心力の項 vθ²/r は使わない (軸の近くで L²/r³ の蹴りが発散するため)
    ridx = _radial_index(project.coord)
    rz = ridx is not None
    # 一様磁場 (prompts/51)。schema の validator で rz + B は弾かれている。
    # B ありでは面内成分と vz が結合するため xy でも速度を3成分で扱う
    b_vec = b_vector(project)
    three_v = rz or b_vec is not None
    fn_currents: np.ndarray | None = None
    fn_total: float | None = None
    if settings.fn is not None:
        # FN 電界放出 (prompts/46): 電極表面の電界から放出電流を計算し、
        # fn.n 個のマクロ電子を電流比例で放出面に配置する
        n_comp = 3 if three_v else 2
        surf = build_fn_surface(project, mesh, adjacency, settings.fn)
        if surf is None:
            seg_i = np.zeros(0)
            counts = np.zeros(0, dtype=np.int64)
            fn_total = 0.0
        else:
            _f_surf, seg_i = fn_segment_currents(surf, e_field, settings.fn, project.coord)
            fn_total = float(seg_i.sum())
            counts = distribute_particles(seg_i, settings.fn.n)
        total = int(counts.sum())
        if total == 0:
            x0 = np.zeros((0, 2))
            v0 = np.zeros((0, n_comp))
            fn_currents = np.zeros(0)
        else:
            seg_idx = np.repeat(np.arange(len(counts)), counts)
            # セグメント内は (j+0.5)/k の等間隔配置 (乱数不使用で決定的)
            offs = np.concatenate(
                [(np.arange(k) + 0.5) / k for k in counts if k > 0]
            )
            x0 = (
                surf.pa[seg_idx]
                + offs[:, None] * (surf.pb[seg_idx] - surf.pa[seg_idx])
                + surf.delta[seg_idx, None] * surf.nrm[seg_idx]
            )
            speed = (
                math.sqrt(2.0 * settings.fn.init_energy_ev * QE / m)
                if settings.fn.init_energy_ev > 0.0
                else 0.0
            )
            v0 = np.zeros((total, n_comp))
            v0[:, :2] = speed * surf.nrm[seg_idx]
            fn_currents = seg_i[seg_idx] / counts[seg_idx]
    else:
        x0, v0 = _init_particles(settings.emitter, m, vtheta=three_v)
    n = len(x0)

    dt = settings.dt
    if dt is None:
        dt = _estimate_dt(mesh, e_field, v0, q, m)
        if b_vec is not None:
            # ジャイロ運動の解像: dt ≤ 0.2/ωc (Boris は無条件安定だが精度確保)
            wc = abs(q) * float(np.linalg.norm(b_vec)) / m
            if wc > 0.0:
                dt = min(dt, 0.2 / wc)
    dt = float(dt)

    # Boris 回転行列 (一様磁場・固定 dt なので前計算できる)。転置を持ち v @ R.T で適用
    boris_rt = _boris_matrix(q, m, dt, b_vec).T if b_vec is not None else None

    n_steps = settings.n_steps
    save_every = settings.save_every

    elem = _locate_initial(coeffs, x0)
    x = x0.copy()
    v = v0.copy()
    alive = np.ones(n, dtype=bool)
    tof = np.full(n, np.nan)

    frames = [x.copy()]
    t_elapsed = 0.0
    qm = q / m

    a_coef, b_coef, c_coef, det_coef = coeffs
    packed = _pack_coeffs(coeffs)  # walk 用の詰め込み係数 (ループ外で1回)

    for step in range(1, n_steps + 1):
        idx_active = np.nonzero(alive)[0]
        if idx_active.size:
            e_at = e_field[elem[idx_active]]
            a_cur = qm * e_at
            x_prev = x[idx_active]
            if rz:
                # 回転法の前半キックは E だけ (E は周成分を持たないので vθ は変わらない)
                v_half = v[idx_active, :2] + 0.5 * dt * a_cur
                vth = v[idx_active, 2]
            elif boris_rt is not None:
                # 一様磁場 (prompts/51): Boris 法。半キック (E) → 回転 (B) の順で
                # 3成分速度 v3 を作り、ドリフト・境界処理は面内成分ビューで行う
                v3 = v[idx_active].copy()
                v3[:, :2] += 0.5 * dt * a_cur
                v3 = v3 @ boris_rt
                v_half = v3[:, :2]  # ビュー: 反射等の面内変更は v3 に反映される
            else:
                v_half = v[idx_active] + 0.5 * dt * a_cur
            x_new = x_prev + dt * v_half

            if rz:
                # 局所座標の3D直線移動 (径 = x_new の径成分 r + vr·dt、周 vθ·dt) の後の半径へ移し、
                # 速度 (vr, vθ) を移動後の位置の局所座標へ回す。軸を越える移動 (r + vr·dt < 0) も
                # r ≥ 0 のまま扱える。所属要素は移動後の位置への walk で追従し、反射・後半キックは
                # 移動後の局所座標の速度に対して行う
                r_new, vr_rot, vth = _rz_rotate(
                    x_new[:, ridx], dt * vth, v_half[:, ridx], vth
                )
                x_new[:, ridx] = r_new
                v_half[:, ridx] = vr_rot

            new_elem, absorbed_mask, b_elem, b_loc = _walk_step(
                coeffs, adjacency, elem[idx_active], x_new, packed=packed
            )

            # 対称エッジは鏡面反射、周期エッジは反対側へラップ (吸収しない)
            if tables is not None and np.any(absorbed_mask):
                _apply_trace_boundaries(
                    tables, coeffs, packed, adjacency,
                    x_new, v_half, new_elem, absorbed_mask, b_elem, b_loc,
                )

            # 誘電体 (固体) 要素へ入った粒子は表面で吸収する (prompts/24)
            hit_solid = None
            if solid is not None:
                hit_solid = ~absorbed_mask & solid[new_elem]
                if not np.any(hit_solid):
                    hit_solid = None

            cont = ~absorbed_mask if hit_solid is None else ~(absorbed_mask | hit_solid)
            cont_idx = idx_active[cont]
            if cont_idx.size:
                x[cont_idx] = x_new[cont]
                elem[cont_idx] = new_elem[cont]
                e_new = e_field[elem[cont_idx]]
                a_new = qm * e_new
                if rz:
                    # 後半キック (E だけ、移動後の位置の局所座標)。vθ は回転で更新済み
                    v[cont_idx, :2] = v_half[cont] + 0.5 * dt * a_new
                    v[cont_idx, 2] = vth[cont]
                elif boris_rt is not None:
                    # 後半キック (E のみ)。vz は回転で更新済み
                    v[cont_idx] = v3[cont]
                    v[cont_idx, :2] += 0.5 * dt * a_new
                else:
                    v[cont_idx] = v_half[cont] + 0.5 * dt * a_new

            abs_idx = idx_active[absorbed_mask]
            if abs_idx.size:
                # 境界を検出したエッジの重心座標 L を x_prev, x_new で評価し、
                # L=0 (境界線) を通過する時刻を線形補間する。
                ea = b_elem[absorbed_mask]
                eloc = b_loc[absorbed_mask]
                bb = b_coef[ea, eloc]
                cc = c_coef[ea, eloc]
                aa = a_coef[ea, eloc]
                dd = det_coef[ea]
                xp_a = x_prev[absorbed_mask]
                xn_a = x_new[absorbed_mask]
                l0 = (aa + bb * xp_a[:, 0] + cc * xp_a[:, 1]) / dd
                l1 = (aa + bb * xn_a[:, 0] + cc * xn_a[:, 1]) / dd
                denom = l0 - l1
                denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
                frac = np.clip(l0 / denom, 0.0, 1.0)

                x[abs_idx] = xp_a + frac[:, None] * (xn_a - xp_a)
                if rz:
                    # 衝突時刻の速度: x_n の局所座標で E の蹴りを frac だけ進め (平面と同じ)、
                    # 3D直線移動のその時刻の位置 (径 r + frac·vr·dt、周 frac·vθ·dt。vr は
                    # 前半キック後の移動の速度) の局所座標へ回す。|v| は回転で変わらない
                    v_abs = v[abs_idx]
                    a_abs = a_cur[absorbed_mask]
                    u = v_abs[:, :2] + frac[:, None] * dt * a_abs
                    vr_drift = v_abs[:, ridx] + 0.5 * dt * a_abs[:, ridx]
                    _, u_r, u_t = _rz_rotate(
                        xp_a[:, ridx] + frac * (dt * vr_drift),
                        frac * (dt * v_abs[:, 2]),
                        u[:, ridx],
                        v_abs[:, 2],
                    )
                    u[:, ridx] = u_r
                    v[abs_idx, :2] = u
                    v[abs_idx, 2] = u_t
                elif boris_rt is not None:
                    # 半キック+回転後の速度を衝突時速度として採用する (近似)
                    v[abs_idx] = v3[absorbed_mask]
                else:
                    v[abs_idx] = v[abs_idx] + frac[:, None] * dt * a_cur[absorbed_mask]
                tof[abs_idx] = t_elapsed + frac * dt
                alive[abs_idx] = False

            if hit_solid is not None:
                # 誘電体表面での吸収: 最終位置は侵入直前〜現在位置の間で良いため
                # 現在位置 (1ステップ分だけ内側) を採用する (厳密な交点補間は不要)
                hit_idx = idx_active[hit_solid]
                x[hit_idx] = x_new[hit_solid]
                if rz:
                    # 移動後の位置の局所座標に回した速度 (後半キック前) を採用する
                    v[hit_idx, :2] = v_half[hit_solid]
                    v[hit_idx, 2] = vth[hit_solid]
                elif boris_rt is not None:
                    v[hit_idx] = v3[hit_solid]
                else:
                    v[hit_idx] = v_half[hit_solid]
                tof[hit_idx] = t_elapsed + dt
                alive[hit_idx] = False

        t_elapsed += dt
        if step % save_every == 0:
            frames.append(x.copy())

    trajectories = np.stack(frames, axis=1)  # (n, n_frames, 2)
    speed2 = np.sum(v * v, axis=1)
    final_energy_ev = 0.5 * m * speed2 / QE
    absorbed = ~alive
    # 衝突角度: absorbed 粒子は衝突時刻に線形補間した速度 (v[abs_idx] を参照)、
    # alive 粒子は最終ステップの速度から向きを求める
    final_angle_deg = np.degrees(np.arctan2(v[:, 1], v[:, 0]))

    return TraceOutput(
        trajectories=trajectories,
        absorbed=absorbed,
        tof=tof,
        final_energy_ev=final_energy_ev,
        final_angle_deg=final_angle_deg,
        dt=dt,
        currents=fn_currents,
        fn_current=fn_total,
    )
