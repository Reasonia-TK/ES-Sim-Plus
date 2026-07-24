"""gmsh (OCC カーネル + boolean fragment) による三角形メッシュ生成。仕様書 §5 / prompts/22 参照。

方針:
- domain・各領域のポリゴン (円は多角形化) を OCC でサーフェス化し、`occ.fragment`
  で全て分割する
- フラグメント面 → 領域の同定は fragment の親子対応 (out_map) で行う
  - domain 外のフラグメントは破棄する (= 領域は黙って domain にクリップされる)
  - conductor 内のフラグメントは穴としてメッシュ化しない
- conductor の Dirichlet 節点は、conductor フラグメントと残存面が共有する
  境界曲線上の節点とする
- 外周エッジの境界条件は、フラグメント後の外周曲線の中点が元の domain エッジ i
  上にあるかで対応付ける (電極が外枠に重なった区間は電極の Dirichlet を優先)
- periodic 境界 (対辺2本) は fragment 後に `mesh.setPeriodic` を適用して対辺の
  メッシュを一致させ、スレーブ→マスターの節点対応 (periodic_map) を返す
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import gmsh
import numpy as np

from .schema import Project, Region, rf_components

# 円形状の多角形分割数の下限・上限 (仕様書 §8 スキーマ契約参照)
CIRCLE_SEGMENTS_MIN = 24
CIRCLE_SEGMENTS_MAX = 720


@dataclass
class Mesh:
    nodes: np.ndarray          # (N, 2) float64
    triangles: np.ndarray      # (M, 3) int64
    tri_region: np.ndarray     # (M,) int64  regions のインデックス。-1 は背景 (真空)
    dirichlet: dict[int, float] = field(default_factory=dict)  # 節点番号 -> 電位 [V] (直流分)
    # 節点番号 -> RF 成分列 ((振幅 [V], 周波数 [Hz], 位相 [deg]), ...)。
    # デュアル周波数対応 (prompts/49) で成分のタプル列。PIC のみ使用 (フェーズ1/2 は無視)
    dirichlet_rf: dict[int, tuple[tuple[float, float, float], ...]] = field(default_factory=dict)
    # 節点番号 -> 二次電子放出係数 γ (>0 の節点のみ)。PIC のみ使用
    see_gamma: dict[int, float] = field(default_factory=dict)
    # 周期境界の正準化写像 (N,)。periodic_map[i] = 節点 i のマスター節点番号
    # (スレーブ以外は自分自身)。periodic 境界が無ければ None
    periodic_map: np.ndarray | None = None
    # 節点番号 -> 電極ラベル ("edge0".."edge3" (domain 外周の Dirichlet エッジ) または
    # conductor 領域の region id)。dirichlet 節点の帰属先を、生成時に既に判明している
    # 情報 (どの境界/領域から Dirichlet が付いたか) からそのまま記録したもの。
    # fem.py の残差法による電極電荷の計算に使う (幾何再判定より確実なため)。
    # dirichlet と同じ優先順位 (領域 > エッジ) で上書きされる
    electrode: dict[int, str] = field(default_factory=dict)


def _circle_polygon(center: tuple[float, float], radius: float, h: float) -> list[tuple[float, float]]:
    """円を多角形近似する。分割数は特性長 h から

        n = clamp(ceil(2*pi*r / h), CIRCLE_SEGMENTS_MIN, CIRCLE_SEGMENTS_MAX)

    で決め、開始角0・反時計回りで頂点列を返す (仕様書 §8 スキーマ契約参照)。
    """
    n = math.ceil(2.0 * math.pi * radius / h)
    n = max(CIRCLE_SEGMENTS_MIN, min(CIRCLE_SEGMENTS_MAX, n))
    cx, cy = center
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return [(cx + radius * math.cos(t), cy + radius * math.sin(t)) for t in theta]


def _region_polygon(region: Region, h: float) -> list[tuple[float, float]]:
    """領域の輪郭ポリゴンを解決する。

    polygon 指定の領域はそのまま返し、shape (circle) 指定の領域は
    その領域の特性長 h (local_size があればそれ、なければ mesh.size) で
    多角形化する。
    """
    if region.polygon is not None:
        return region.polygon
    assert region.shape is not None  # schema のバリデーションで保証済み
    return _circle_polygon(region.shape.center, region.shape.radius, h)


def _occ_polygon_surface(points) -> int:
    """点列から OCC の閉じた平面サーフェスを作る (サーフェスタグを返す)。"""
    occ = gmsh.model.occ
    pts = [occ.addPoint(x, y, 0.0) for x, y in points]
    curves = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    loop = occ.addCurveLoop(curves)
    return occ.addPlaneSurface([loop])


def _points_on_segment(pts: np.ndarray, q1: np.ndarray, q2: np.ndarray, tol: float) -> np.ndarray:
    """点列 pts (K, 2) が線分 q1-q2 上 (距離 tol 以内・パラメータ範囲内) にあるか。"""
    seg = q2 - q1
    seg_len = float(np.hypot(seg[0], seg[1]))
    if seg_len <= 0.0 or len(pts) == 0:
        return np.zeros(len(pts), dtype=bool)
    d = pts - q1
    dist = np.abs(d[:, 0] * seg[1] - d[:, 1] * seg[0]) / seg_len
    t = (d[:, 0] * seg[0] + d[:, 1] * seg[1]) / (seg_len * seg_len)
    return (dist <= tol) & (t >= -1e-9) & (t <= 1.0 + 1e-9)


def generate_mesh(project: Project) -> Mesh:
    """メッシュ生成の入口。mesh.mode に応じて非構造 (gmsh) / 構造格子を切り替える。"""
    if project.mesh.mode == "structured":
        return _generate_structured(project)
    return _generate_unstructured(project)


def _generate_unstructured(project: Project) -> Mesh:
    geo = project.geometry
    lc = project.mesh.size
    local = {ls.region: ls.size for ls in project.mesh.local_sizes}
    domain_poly = np.asarray(geo.domain.polygon, dtype=np.float64)
    n_edges = len(domain_poly)
    scale = float(np.max(np.abs(domain_poly)))
    tol = 1e-8 * (scale if scale > 0.0 else 1.0)

    # interruptible=False: シグナルハンドラを登録しない
    # (FastAPI はワーカースレッドで実行するため必須)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("es_sim")
        occ = gmsh.model.occ

        # ---- OCC サーフェス化 → boolean fragment ----------------------------
        s_domain = _occ_polygon_surface(geo.domain.polygon)
        region_lcs: list[float] = []
        tool_surfs: list[int] = []
        for region in geo.regions:
            r_lc = local.get(region.id, lc)
            region_lcs.append(r_lc)
            tool_surfs.append(_occ_polygon_surface(_region_polygon(region, r_lc)))

        if tool_surfs:
            _, out_map = occ.fragment([(2, s_domain)], [(2, s) for s in tool_surfs])
        else:
            out_map = [[(2, s_domain)]]
        occ.synchronize()

        # ---- フラグメント面 → 領域の同定 (fragment の親子対応で判定) ---------
        # parents[面タグ] = 親入力のインデックス列 (0: domain、1+i: regions[i])
        parents: dict[int, list[int]] = {}
        for pi, children in enumerate(out_map):
            for dim, tag in children:
                if dim != 2:
                    continue
                ps = parents.setdefault(tag, [])
                if pi not in ps:
                    ps.append(pi)

        kept: list[tuple[int, int]] = []            # (面タグ, 領域インデックス。-1 は背景)
        conductor_surfs: dict[int, list[int]] = {}  # 領域インデックス -> 面タグ列
        removed_surfs: list[int] = []               # メッシュ化しない面 (domain 外・conductor 内)
        for tag, ps in parents.items():
            if 0 not in ps:
                removed_surfs.append(tag)  # domain 外 → 黙ってクリップ
                continue
            ridx = [p - 1 for p in ps if p >= 1]
            cond = [r for r in ridx if geo.regions[r].type == "conductor"]
            if cond:
                conductor_surfs.setdefault(cond[0], []).append(tag)
                removed_surfs.append(tag)  # conductor 内は穴として除外
            elif ridx:
                kept.append((tag, ridx[0]))
            else:
                kept.append((tag, -1))
        if not kept:
            raise ValueError("メッシュ化できる面がありません (domain 全体が conductor に覆われています)")

        # ---- 曲線・点の分類 (面を削除する前に行う) ---------------------------
        def _bnd_curves(surface_tag: int) -> list[int]:
            return [
                abs(t)
                for d, t in gmsh.model.getBoundary([(2, surface_tag)], oriented=False)
                if d == 1
            ]

        kept_curves: list[int] = []
        kept_curve_set: set[int] = set()
        for tag, _ in kept:
            for c in _bnd_curves(tag):
                if c not in kept_curve_set:
                    kept_curve_set.add(c)
                    kept_curves.append(c)

        # conductor の Dirichlet 曲線 = conductor フラグメントと残存面が共有する曲線
        conductor_curves: list[tuple[str, float, object, float, list[int]]] = []
        for i, region in enumerate(geo.regions):
            if region.type != "conductor":
                continue
            if region.voltage is None:
                raise ValueError(f"conductor '{region.id}' に voltage がありません")
            curves: list[int] = []
            for tag in conductor_surfs.get(i, []):
                for c in _bnd_curves(tag):
                    if c in kept_curve_set and c not in curves:
                        curves.append(c)
            conductor_curves.append(
                (region.id, region.voltage, region.voltage_rf, region.see_gamma, curves)
            )

        # 局所メッシュサイズ用: 各領域のフラグメント境界点 (従来相当の挙動維持)
        region_size_pts: list[tuple[list[int], float]] = []
        for i in range(len(geo.regions)):
            r_lc = region_lcs[i]
            if r_lc == lc:
                continue  # 全体特性長と同じなら個別設定は不要
            tags = [t for t, r in kept if r == i] + conductor_surfs.get(i, [])
            pts: set[int] = set()
            for tag in tags:
                for d, p in gmsh.model.getBoundary(
                    [(2, tag)], recursive=True, oriented=False
                ):
                    if d == 0:
                        pts.add(p)
            if pts:
                region_size_pts.append((sorted(pts), r_lc))

        # ---- 不要な面を除去 → 特性長設定 -------------------------------------
        if removed_surfs:
            gmsh.model.removeEntities([(2, t) for t in removed_surfs], recursive=False)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
        for pts, r_lc in region_size_pts:
            gmsh.model.mesh.setSize([(0, p) for p in pts], r_lc)

        # ---- 外周曲線 → domain エッジの対応付け (曲線中点がエッジ上か) --------
        if kept_curves:
            mids = np.array(
                [occ.getCenterOfMass(1, c)[:2] for c in kept_curves], dtype=np.float64
            )
        else:
            mids = np.zeros((0, 2))
        curve_edge = np.full(len(kept_curves), -1, dtype=np.int64)
        for i in range(n_edges):
            on = _points_on_segment(
                mids, domain_poly[i], domain_poly[(i + 1) % n_edges], tol
            )
            curve_edge[on & (curve_edge < 0)] = i
        edge_curves: dict[int, list[int]] = {}
        for c, e in zip(kept_curves, curve_edge):
            if e >= 0:
                edge_curves.setdefault(int(e), []).append(c)

        # ---- periodic 境界: 対辺の曲線を対応付けて setPeriodic ----------------
        periodic_slave_curves: list[int] = []
        for bc in geo.boundaries:
            if bc.type != "periodic":
                continue
            e_m, e_s = bc.edges  # [マスター辺, スレーブ辺] とみなす
            seg_mid_m = 0.5 * (domain_poly[e_m] + domain_poly[(e_m + 1) % n_edges])
            seg_mid_s = 0.5 * (domain_poly[e_s] + domain_poly[(e_s + 1) % n_edges])
            t_vec = seg_mid_s - seg_mid_m  # マスター辺 → スレーブ辺の平行移動
            masters = edge_curves.get(e_m, [])
            slaves = edge_curves.get(e_s, [])
            if not masters or len(masters) != len(slaves):
                raise ValueError(
                    f"periodic 境界 (エッジ {e_m}, {e_s}) の対辺で曲線分割が一致しません。"
                    "周期辺には領域を重ねないでください"
                )
            master_mids = {
                cm: np.asarray(occ.getCenterOfMass(1, cm)[:2]) for cm in masters
            }
            ordered_masters: list[int] = []
            for cs in slaves:
                target = np.asarray(occ.getCenterOfMass(1, cs)[:2]) - t_vec
                match = [
                    cm
                    for cm, mm in master_mids.items()
                    if float(np.hypot(*(target - mm))) <= 10.0 * tol
                ]
                if len(match) != 1:
                    raise ValueError(
                        f"periodic 境界 (エッジ {e_m}, {e_s}) の曲線対応が取れません。"
                        "対辺のメッシュ分割が平行移動で一致する形状にしてください"
                    )
                ordered_masters.append(match[0])
            affine = [
                1.0, 0.0, 0.0, float(t_vec[0]),
                0.0, 1.0, 0.0, float(t_vec[1]),
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ]
            gmsh.model.mesh.setPeriodic(1, slaves, ordered_masters, affine)
            periodic_slave_curves.extend(slaves)

        gmsh.model.mesh.generate(2)

        # ---- 節点 ---------------------------------------------------------
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=np.int64)
        nodes = np.asarray(coords, dtype=np.float64).reshape(-1, 3)[:, :2]
        tag_to_index = np.zeros(node_tags.max() + 1, dtype=np.int64)
        tag_to_index[node_tags] = np.arange(len(node_tags))

        # ---- 要素 (残存面ごとに領域タグを付ける) -----------------------------
        tri_list, region_list = [], []
        for surface_tag, region_index in kept:
            etypes, _, enodes = gmsh.model.mesh.getElements(2, surface_tag)
            for etype, conn in zip(etypes, enodes):
                if etype != 2:  # 3節点三角形のみ
                    continue
                tris = tag_to_index[np.asarray(conn, dtype=np.int64)].reshape(-1, 3)
                tri_list.append(tris)
                region_list.append(np.full(len(tris), region_index, dtype=np.int64))

        triangles = np.concatenate(tri_list)
        tri_region = np.concatenate(region_list)

        # ---- Dirichlet 節点 ------------------------------------------------
        dirichlet: dict[int, float] = {}
        dirichlet_rf: dict[int, tuple[tuple[float, float, float], ...]] = {}
        see_gamma: dict[int, float] = {}
        electrode: dict[int, str] = {}

        def _curve_nodes(curve_tag: int) -> np.ndarray:
            tags, _, _ = gmsh.model.mesh.getNodes(1, curve_tag, includeBoundary=True)
            return tag_to_index[np.asarray(tags, dtype=np.int64)]

        def _assign(n: int, voltage: float, rf, label: str) -> None:
            """節点に直流分・RF 成分列・電極ラベルを設定する (RF なしなら既存 RF を消して上書き)。

            label は fem.py の残差法による電荷計算用の帰属先 ("edge0".."edge3" または
            conductor の region id)。dirichlet と同じ呼び出し順で上書きするため、
            電極の優先順位 (領域 > エッジ) も自動的に一致する。
            """
            dirichlet[n] = voltage
            electrode[n] = label
            comps = rf_components(rf)
            if comps:
                dirichlet_rf[n] = tuple((c.amplitude, c.freq_hz, c.phase_deg) for c in comps)
            else:
                dirichlet_rf.pop(n, None)

        def _assign_gamma(n: int, gamma: float) -> None:
            """節点に SEE 係数 γ を設定する (共有節点は最大値を採用)。"""
            if gamma > 0.0:
                see_gamma[n] = max(see_gamma.get(n, 0.0), gamma)

        # 外周エッジの境界条件 (Dirichlet のみ。symmetry / periodic は自然境界)
        for bc in geo.boundaries:
            if bc.type != "dirichlet":
                continue
            for edge in bc.edges:
                for c in edge_curves.get(edge, []):
                    for n in _curve_nodes(c):
                        _assign(int(n), bc.voltage, bc.voltage_rf, f"edge{edge}")
                        _assign_gamma(int(n), bc.see_gamma)

        # 電極輪郭 (電極の指定を優先して上書き。外枠に重なった区間も電極が勝つ)
        for region_id, voltage, rf, gamma, curves in conductor_curves:
            for c in curves:
                for n in _curve_nodes(c):
                    _assign(int(n), voltage, rf, region_id)
                    _assign_gamma(int(n), gamma)

        # ---- 周期節点対応 (スレーブ → マスター) ------------------------------
        pairs: dict[int, int] = {}
        for c in periodic_slave_curves:
            mtag, ntags, mntags, _ = gmsh.model.mesh.getPeriodicNodes(1, c)
            if mtag == c or len(ntags) == 0:
                continue
            for s_t, m_t in zip(ntags, mntags):
                si = int(tag_to_index[int(s_t)])
                mi = int(tag_to_index[int(m_t)])
                if si != mi:
                    pairs[si] = mi

        # ---- 未参照節点の除去・再番号付け・周期正準化 (共通後処理) -------------
        return _finalize_mesh(
            nodes, triangles, tri_region, dirichlet, dirichlet_rf, see_gamma, pairs, electrode
        )
    finally:
        gmsh.finalize()


def _finalize_mesh(
    nodes: np.ndarray,
    triangles: np.ndarray,
    tri_region: np.ndarray,
    dirichlet: dict[int, float],
    dirichlet_rf: dict[int, tuple[tuple[float, float, float], ...]],
    see_gamma: dict[int, float],
    pairs: dict[int, int],
    electrode: dict[int, str] | None = None,
) -> Mesh:
    """メッシュの共通後処理: 未参照節点の除去と再番号付け、周期対応の正準化。

    (conductor 内・domain 外の節点を落とし、periodic のスレーブ→マスター写像
    periodic_map を構築して Dirichlet / γ / 電極ラベルをマスターへ伝播する)
    """
    electrode = {} if electrode is None else electrode
    used = np.unique(triangles)
    remap = np.full(len(nodes), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    nodes = nodes[used]
    triangles = remap[triangles]

    dirichlet = {int(remap[n]): v for n, v in dirichlet.items() if remap[n] >= 0}
    dirichlet_rf = {int(remap[n]): v for n, v in dirichlet_rf.items() if remap[n] >= 0}
    see_gamma = {int(remap[n]): v for n, v in see_gamma.items() if remap[n] >= 0}
    electrode = {int(remap[n]): v for n, v in electrode.items() if remap[n] >= 0}

    periodic_map: np.ndarray | None = None
    if pairs:
        canon = np.arange(len(nodes), dtype=np.int64)
        for s, m in pairs.items():
            if remap[s] >= 0 and remap[m] >= 0:
                canon[remap[s]] = remap[m]
        # 二重周期の角などの連鎖を推移的に解決する
        for _ in range(8):
            c2 = canon[canon]
            if np.array_equal(c2, canon):
                break
            canon = c2
        if np.any(canon != np.arange(len(nodes))):
            periodic_map = canon
            # スレーブに付いた Dirichlet / γ / 電極ラベルをマスターへも伝播する (角の整合。
            # fem.py の残差法はマスター節点で評価するため、電極ラベルもマスター側に無いと
            # そのマスター節点の残差が電荷合計に計上されなくなる)
            for n in list(dirichlet):
                m = int(canon[n])
                if m != n and m not in dirichlet:
                    dirichlet[m] = dirichlet[n]
                    if n in dirichlet_rf:
                        dirichlet_rf[m] = dirichlet_rf[n]
                    if n in electrode:
                        electrode[m] = electrode[n]
            for n in list(see_gamma):
                m = int(canon[n])
                if m != n:
                    see_gamma[m] = max(see_gamma.get(m, 0.0), see_gamma[n])

    return Mesh(nodes=nodes, triangles=triangles,
                tri_region=tri_region, dirichlet=dirichlet,
                dirichlet_rf=dirichlet_rf, see_gamma=see_gamma,
                periodic_map=periodic_map, electrode=electrode)


# ---- 構造格子メッシュ (prompts/34) --------------------------------------------


def _points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """点列 pts (K, 2) がポリゴン内部にあるか (ray casting、偶奇則)。

    境界上の点の判定は不定 (要素中心の判定用。節点の境界含む判定は
    _points_in_region を使う)。
    """
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    n = len(poly)
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        if y1 == y2:
            continue  # 水平エッジは交差判定に寄与しない
        cond = (y1 > y) != (y2 > y)
        x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (x < x_cross)
    return inside


def _points_in_region(region: Region, pts: np.ndarray, tol: float) -> np.ndarray:
    """点列が領域に内包されるか (境界上を含む、許容誤差 tol)。

    circle は中心距離、polygon は ray casting + 辺への距離判定。
    """
    if region.shape is not None:
        cx, cy = region.shape.center
        dist = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        return dist <= region.shape.radius + tol
    poly = np.asarray(region.polygon, dtype=np.float64)
    inc = _points_in_polygon(pts, poly)
    for k in range(len(poly)):
        inc |= _points_on_segment(pts, poly[k], poly[(k + 1) % len(poly)], tol)
    return inc


def _region_centroid_mask(region: Region, pts: np.ndarray) -> np.ndarray:
    """要素中心の領域内包判定 (polygon: ray casting、circle: 中心距離)。"""
    if region.shape is not None:
        cx, cy = region.shape.center
        return np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) <= region.shape.radius
    return _points_in_polygon(pts, np.asarray(region.polygon, dtype=np.float64))


def _generate_structured(project: Project) -> Mesh:
    """軸平行矩形 domain 用の等間隔構造格子メッシュ (三角形2分割) を生成する。

    - 格子: nx = max(1, round(W/size)), ny = max(1, round(H/size))。
      各セルを対角線で2三角形に分割 (向きは市松に交互 = 等方性向上、反時計回り)
    - 領域割り当て: 要素中心の点内包判定 (曲線境界は階段近似、
      domain 外へのはみ出しは自然にクリップ)
    - conductor: 内包要素を穴として除去し、conductor に内包される節点
      (境界上含む、許容誤差 1e-12 相対) のうち残存要素から参照されるものを
      Dirichlet にする
    - periodic: 対辺の節点が格子で完全一致するため座標対応で periodic_map を構築
    - local_sizes は構造格子では非対応 (指定されていても無視する)
    """
    geo = project.geometry
    size = project.mesh.size
    poly = np.asarray(geo.domain.polygon, dtype=np.float64)
    scale = float(np.max(np.abs(poly)))
    scale = scale if scale > 0.0 else 1.0
    tol = 1e-12 * scale  # 節点内包・エッジ判定の相対許容誤差

    # ---- 前提検査: 軸平行の矩形 (4頂点、各辺が x/y 軸に平行) -----------------
    if len(poly) != 4:
        raise ValueError(
            "構造格子 (mesh.mode='structured') には4頂点の矩形 domain が必要です"
        )
    for k in range(4):
        d = poly[(k + 1) % 4] - poly[k]
        ax_x = abs(d[0]) <= tol  # x が一定 (縦辺)
        ax_y = abs(d[1]) <= tol  # y が一定 (横辺)
        if ax_x == ax_y:  # 両方 True (退化) か両方 False (斜め辺)
            raise ValueError(
                "構造格子 (mesh.mode='structured') には軸平行の矩形 domain が必要です "
                f"(辺 {k} が x/y 軸に平行ではありません)"
            )

    x0, x1 = float(poly[:, 0].min()), float(poly[:, 0].max())
    y0, y1 = float(poly[:, 1].min()), float(poly[:, 1].max())
    w, h = x1 - x0, y1 - y0
    nx = max(1, int(round(w / size)))
    ny = max(1, int(round(h / size)))

    # ---- 節点 (等間隔格子) と三角形 (市松に対角線を交互) ----------------------
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    gx, gy = np.meshgrid(xs, ys)  # (ny+1, nx+1)
    nodes = np.stack([gx.ravel(), gy.ravel()], axis=1)

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    ii, jj = ii.ravel(), jj.ravel()
    a = jj * (nx + 1) + ii        # 左下
    b = a + 1                     # 右下
    c = b + (nx + 1)              # 右上
    d = a + (nx + 1)              # 左上
    even = ((ii + jj) % 2 == 0)[:, None]
    # 偶数セルは a-c 対角 (abc, acd)、奇数セルは b-d 対角 (abd, bcd)。全て反時計回り
    t1 = np.where(even, np.stack([a, b, c], axis=1), np.stack([a, b, d], axis=1))
    t2 = np.where(even, np.stack([a, c, d], axis=1), np.stack([b, c, d], axis=1))
    triangles = np.empty((2 * len(a), 3), dtype=np.int64)
    triangles[0::2] = t1
    triangles[1::2] = t2

    # ---- 領域割り当て (要素中心の内包判定。conductor を優先) ------------------
    centroids = nodes[triangles].mean(axis=1)
    tri_region = np.full(len(triangles), -1, dtype=np.int64)
    conductor_ids = [i for i, r in enumerate(geo.regions) if r.type == "conductor"]
    other_ids = [i for i, r in enumerate(geo.regions) if r.type != "conductor"]
    for i in conductor_ids + other_ids:
        sel = _region_centroid_mask(geo.regions[i], centroids) & (tri_region == -1)
        tri_region[sel] = i

    # conductor 内包要素は穴として除去する
    for i in conductor_ids:
        if geo.regions[i].voltage is None:
            raise ValueError(f"conductor '{geo.regions[i].id}' に voltage がありません")
    if conductor_ids:
        keep = ~np.isin(tri_region, conductor_ids)
        triangles = triangles[keep]
        tri_region = tri_region[keep]
    if len(triangles) == 0:
        raise ValueError("メッシュ化できる要素がありません (domain 全体が conductor に覆われています)")

    # ---- Dirichlet 節点 -------------------------------------------------------
    dirichlet: dict[int, float] = {}
    dirichlet_rf: dict[int, tuple[tuple[float, float, float], ...]] = {}
    see_gamma: dict[int, float] = {}
    electrode: dict[int, str] = {}

    def _assign(n: int, voltage: float, rf, label: str) -> None:
        """節点に直流分・RF 成分列・電極ラベルを設定する (RF なしなら既存 RF を消して上書き)。

        label は fem.py の残差法による電荷計算用の帰属先 ("edge0".."edge3" または
        conductor の region id)。
        """
        dirichlet[n] = voltage
        electrode[n] = label
        comps = rf_components(rf)
        if comps:
            dirichlet_rf[n] = tuple((c.amplitude, c.freq_hz, c.phase_deg) for c in comps)
        else:
            dirichlet_rf.pop(n, None)

    def _assign_gamma(n: int, gamma: float) -> None:
        """節点に SEE 係数 γ を設定する (共有節点は最大値を採用)。"""
        if gamma > 0.0:
            see_gamma[n] = max(see_gamma.get(n, 0.0), gamma)

    # 外周エッジの境界条件 (Dirichlet のみ。symmetry / periodic は自然境界)
    for bc in geo.boundaries:
        if bc.type != "dirichlet":
            continue
        for edge in bc.edges:
            q1, q2 = poly[edge % 4], poly[(edge + 1) % 4]
            for n in np.nonzero(_points_on_segment(nodes, q1, q2, tol))[0]:
                _assign(int(n), bc.voltage, bc.voltage_rf, f"edge{edge}")
                _assign_gamma(int(n), bc.see_gamma)

    # 電極: conductor に内包される節点 (電極の指定を優先して上書き)。
    # 残存要素から参照されない節点 (電極内部) は後処理で除去される
    for i in conductor_ids:
        region = geo.regions[i]
        for n in np.nonzero(_points_in_region(region, nodes, tol))[0]:
            _assign(int(n), region.voltage, region.voltage_rf, region.id)
            _assign_gamma(int(n), region.see_gamma)

    # ---- periodic: 対辺の節点は格子で完全一致するので座標対応で組む -----------
    pairs: dict[int, int] = {}
    for bc in geo.boundaries:
        if bc.type != "periodic":
            continue
        e_m, e_s = bc.edges  # [マスター辺, スレーブ辺] とみなす
        seg_m = (poly[e_m % 4], poly[(e_m + 1) % 4])
        seg_s = (poly[e_s % 4], poly[(e_s + 1) % 4])
        t_vec = 0.5 * (seg_s[0] + seg_s[1]) - 0.5 * (seg_m[0] + seg_m[1])
        m_idx = np.nonzero(_points_on_segment(nodes, seg_m[0], seg_m[1], tol))[0]
        s_idx = np.nonzero(_points_on_segment(nodes, seg_s[0], seg_s[1], tol))[0]
        if len(m_idx) != len(s_idx) or len(m_idx) == 0:
            raise ValueError(
                f"periodic 境界 (エッジ {e_m}, {e_s}) の対辺で節点数が一致しません"
            )
        # 座標の辞書順で並べれば平行移動で一対一に対応する
        m_srt = m_idx[np.lexsort((nodes[m_idx, 1], nodes[m_idx, 0]))]
        s_srt = s_idx[np.lexsort((nodes[s_idx, 1], nodes[s_idx, 0]))]
        if not np.allclose(nodes[s_srt] - t_vec, nodes[m_srt], atol=10.0 * tol + 1e-15):
            raise ValueError(
                f"periodic 境界 (エッジ {e_m}, {e_s}) の対辺節点が平行移動で一致しません"
            )
        for s, m in zip(s_srt, m_srt):
            if int(s) != int(m):
                pairs[int(s)] = int(m)

    # ---- 共通後処理 (未参照節点の除去・再番号付け・周期正準化) ----------------
    return _finalize_mesh(
        nodes, triangles, tri_region, dirichlet, dirichlet_rf, see_gamma, pairs, electrode
    )
