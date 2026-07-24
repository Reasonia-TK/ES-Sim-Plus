"""P1 (線形三角形) 要素による静電場 FEM。仕様書 §6 参照。

∇·(ε∇V) = -ρ を弱形式で解く。
組み立ては全要素一括のベクトル化。Dirichlet は対称性を保つ縮約方式。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .meshing import Mesh
from .schema import Project

EPS0 = 8.8541878128e-12  # 真空の誘電率 [F/m]


def _radial_index(coord: str) -> int | None:
    """軸対称モードの径方向座標インデックスを返す (rz: y=1、rz_x0: x=0。xy は None)。"""
    return {"rz": 1, "rz_x0": 0}.get(coord)


@dataclass
class Solution:
    v: np.ndarray        # (N,) 節点電位 [V]
    e_field: np.ndarray  # (M, 2) 要素ごとの E = -∇V [V/m]
    energy: float        # 蓄積エネルギー [J/m] (奥行き単位長あたり)
    # 電極ごとの誘起電荷 (label, voltage [V], q)。xy: [C/m]、軸対称: [C]。
    # label は "edge0".."edge3" (domain 外周の Dirichlet エッジ) または conductor の region id
    charges: list[tuple[str, float, float]] = field(default_factory=list)
    # 静電容量 [F/m] (xy) / [F] (軸対称)。電極の電位がちょうど2水準かつ空間電荷が
    # 全域 0 の場合のみ定義。それ以外 (3水準以上・電荷あり・電極なし) は None
    capacitance: float | None = None


def _element_geometry(nodes: np.ndarray, tris: np.ndarray):
    """P1 要素の形状関数勾配と面積 (全要素一括)。"""
    p = nodes[tris]                      # (M, 3, 2)
    x, y = p[:, :, 0], p[:, :, 1]
    # b_i = y_j - y_k, c_i = x_k - x_j  (i, j, k は巡回)
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)
    det = x[:, 0] * b[:, 0] + x[:, 1] * b[:, 1] + x[:, 2] * b[:, 2]
    area = 0.5 * np.abs(det)
    return b, c, area                    # (M,3), (M,3), (M,)


def _material_arrays(project: Project, mesh: Mesh):
    """要素ごとの ε と ρ。"""
    eps = np.full(len(mesh.triangles), EPS0)
    rho = np.zeros(len(mesh.triangles))
    for i, region in enumerate(project.geometry.regions):
        mask = mesh.tri_region == i
        if region.type == "dielectric":
            eps[mask] = EPS0 * region.eps_r
        elif region.type == "charge":
            rho[mask] = region.rho
    return eps, rho


def assemble(project: Project, mesh: Mesh):
    """剛性行列 K (csr) と右辺 f を返す。

    軸対称モード (prompts/39, 41) では弱形式
        ∫ ε ∇V·∇W r dr dz = ∫ ρ W r dr dz
    を使う (径方向座標は coord="rz" なら y、"rz_x0" なら x)。剛性は平面の Ke に
    要素重心半径 r̄ = (r_i + r_j + r_k)/3 を乗じる標準近似、右辺は線形 r を
    厳密に積分する。
    """
    tris = mesh.triangles
    b, c, area = _element_geometry(mesh.nodes, tris)
    eps, rho = _material_arrays(project, mesh)

    # 周期境界: スレーブ節点をマスターへ置換した正準節点番号で組み立てる
    # (要素幾何は元の節点座標で評価済みなので係数は変わらない)
    idx = tris if mesh.periodic_map is None else mesh.periodic_map[tris]
    n = len(mesh.nodes)
    f = np.zeros(n)

    ridx = _radial_index(project.coord)
    if ridx is not None:
        # 軸対称剛性: Ke[i,j] = ε·r̄·(b_i b_j + c_i c_j)/(4A) (r̄ による標準近似。
        # 軸上 r=0 を含む要素でも r̄ > 0 なので特異にならない)
        r_nodes = mesh.nodes[tris][:, :, ridx]         # (M, 3) 各頂点の r
        r_bar = r_nodes.mean(axis=1)                   # (M,) 要素重心半径
        coef = (eps * r_bar / (4.0 * area))[:, None, None]
        ke = coef * (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :])
        # 右辺 f_i = ρ ∫ N_i r dA = ρ·(A/12)·(2 r_i + r_j + r_k) (線形 r の厳密積分)
        fw = (rho * area / 12.0)[:, None] * (r_nodes + 3.0 * r_bar[:, None])
        np.add.at(f, idx.ravel(), fw.ravel())
    else:
        # 平面2D: Ke[i,j] = eps * (b_i b_j + c_i c_j) / (4A)
        coef = (eps / (4.0 * area))[:, None, None]
        ke = coef * (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :])
        # 一様電荷密度の P1 右辺: 各節点へ rho*A/3
        np.add.at(f, idx.ravel(), np.repeat(rho * area / 3.0, 3))

    rows = np.repeat(idx, 3, axis=1).ravel()           # i index
    cols = np.tile(idx, (1, 3)).ravel()                # j index
    k = sp.coo_matrix((ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    return k, f


def _label_order(project: Project) -> list[str]:
    """電極ラベルの表示順を決める (エッジ境界の指定順 → conductor 領域の定義順)。

    charges の並びを毎回決定的にするため (dict の走査順に依存させない)。
    """
    order: list[str] = []
    seen: set[str] = set()
    for bc in project.geometry.boundaries:
        if bc.type != "dirichlet":
            continue
        for edge in bc.edges:
            label = f"edge{edge}"
            if label not in seen:
                seen.add(label)
                order.append(label)
    for region in project.geometry.regions:
        if region.type == "conductor" and region.id not in seen:
            seen.add(region.id)
            order.append(region.id)
    return order


def _electrode_charges(
    project: Project, mesh: Mesh, k: sp.csr_matrix, f: np.ndarray, v: np.ndarray
) -> tuple[list[tuple[str, float, float]], float | None]:
    """電極ごとの誘起電荷と (定義できれば) 静電容量を、残差法で計算する。

    残差 r = Kφ - f は、BC 適用前の全体剛性・右辺に対する各節点の「不釣り合い力」で、
    Dirichlet 節点上ではこれがその節点に集中する電荷 (弱形式の ∮ε∂φ/∂n) に一致する
    (フラックスを要素境界で積分するより高精度な標準手法)。電極 (edge/conductor) ごとに
    帰属節点の残差を合計すれば、その電極の全電荷が得られる。

    周期境界のスレーブ節点は assemble() で行が丸ごとマスターへ寄せられているため
    (mesh.periodic_map)、K のスレーブ行は全て 0 → 残差も 0 になり、二重計上の
    心配なく単純に合計してよい (mesh.electrode にスレーブ節点が含まれていても影響しない)。

    軸対称モードの fem.py の剛性は 2π を含まない r̄ 重み近似 (assemble 参照) なので、
    残差もそのままでは 2π 分小さい。ここで 2π を掛けて物理電荷 [C] に直す
    (xy は奥行き1m あたりの [C/m] のまま)。
    """
    r = k @ v - f
    ridx = _radial_index(project.coord)
    factor = 2.0 * np.pi if ridx is not None else 1.0

    groups: dict[str, list[float]] = {}  # label -> [voltage, q_sum]
    for node, label in mesh.electrode.items():
        q = factor * float(r[node])
        if label in groups:
            groups[label][1] += q
        else:
            groups[label] = [mesh.dirichlet[node], q]

    charges = [(label, groups[label][0], groups[label][1])
               for label in _label_order(project) if label in groups]

    # 静電容量: 電極電位がちょうど2水準、かつ空間電荷 (charge 領域の rho) が全域 0 の場合のみ
    capacitance: float | None = None
    _, rho = _material_arrays(project, mesh)
    if len(charges) >= 2 and not np.any(rho != 0.0):
        voltages = sorted({voltage for _, voltage, _ in charges})
        if len(voltages) == 2:
            v_lo, v_hi = voltages
            q_hi = sum(q for _, voltage, q in charges if voltage == v_hi)
            capacitance = q_hi / (v_hi - v_lo)

    return charges, capacitance


def solve(project: Project, mesh: Mesh) -> Solution:
    k, f = assemble(project, mesh)
    n = len(mesh.nodes)

    fixed = np.fromiter(mesh.dirichlet.keys(), dtype=np.int64)
    v_fixed = np.fromiter(mesh.dirichlet.values(), dtype=np.float64)
    canon = mesh.periodic_map
    if canon is None:
        free = np.setdiff1d(np.arange(n), fixed)
    else:
        # 周期スレーブ節点は自由度から除外する (剛性行列の行がマスターへ寄っている)
        slaves = np.nonzero(canon != np.arange(n))[0]
        free = np.setdiff1d(np.arange(n), np.union1d(fixed, slaves))

    v = np.zeros(n)
    v[fixed] = v_fixed
    if len(free):
        k_ff = k[free][:, free].tocsc()
        rhs = f[free] - k[free][:, fixed] @ v_fixed
        # 直接法 (LU)。PIC で右辺のみ更新する再解析に備え splu を使う
        v[free] = spla.splu(k_ff).solve(rhs)
    if canon is not None:
        v = v[canon]        # スレーブ節点へマスター値をコピー (表示互換)
        v[fixed] = v_fixed  # Dirichlet 値は厳密に保持

    # E = -∇V (要素内一定)
    tris = mesh.triangles
    b, c, area = _element_geometry(mesh.nodes, tris)
    vt = v[tris]                                        # (M, 3)
    inv2a = 1.0 / (2.0 * area)
    ex = -np.sum(vt * b, axis=1) * inv2a
    ey = -np.sum(vt * c, axis=1) * inv2a
    e_field = np.stack([ex, ey], axis=1)

    eps, _ = _material_arrays(project, mesh)
    ridx = _radial_index(project.coord)
    if ridx is not None:
        # 軸対称エネルギー W = ½ ∫ ε|E|²·2πr dA [J]
        # (E は要素内一定なので ∫ r dA = r̄·A で厳密。xy モードは [J/m])
        r_bar = mesh.nodes[tris][:, :, ridx].mean(axis=1)
        energy = float(np.sum(0.5 * eps * (ex**2 + ey**2) * 2.0 * np.pi * r_bar * area))
    else:
        energy = float(np.sum(0.5 * eps * (ex**2 + ey**2) * area))

    charges, capacitance = _electrode_charges(project, mesh, k, f, v)
    return Solution(v=v, e_field=e_field, energy=energy, charges=charges, capacitance=capacitance)
