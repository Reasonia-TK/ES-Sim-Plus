"""v2 直交格子 (EB) 静電場ソルバーの高レベル API (prompts/119)。

v1 の ``fem.solve`` (P1 FEM、非構造三角形) と同じ量を同じ単位で返す:

- 電位 φ (節点)、電場 E = −∇φ (節点)
- 蓄積エネルギー W = ½∫ε|E|² (xy: [J/m]、軸対称: [J])
- 電極ごとの誘起電荷 (label, V, Q) (xy: [C/m]、軸対称: [C]) と静電容量
  (電極電位がちょうど 2 水準かつ空間電荷が全域 0 のときのみ。v1 と同じ条件)

電荷は Dirichlet 結合のフラックス Q_k = Σ_P G_Pk (V_k − φ_P) (導体から出る電束 = 表面電荷、
ガウスの法則そのもの)。エネルギーは離散演算子と整合する二次形式
W = ½ Σ_辺 G (Δφ)² + ½ Σ_結合 G (φ − V)² (厳密解ではどちらも ½ΣQV に一致)。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..device import Device, get_device
from ..eb.build import (
    DIR_E,
    DIR_N,
    DIR_S,
    DIR_W,
    MASK_FIXED,
    MASK_SLAVE,
    MASK_UNKNOWN,
    LevelOperator,
)
from ..eb.grid import CartesianGrid, make_grid
from ..geom.model import GeometryModel
from ..schema import Project
from .gmg import GMGSolver, SolveInfo


@dataclass
class StaticSolution:
    grid: CartesianGrid
    phi: np.ndarray                  # (ny+1, nx+1) [V]
    ex: np.ndarray                   # (ny+1, nx+1) [V/m]
    ey: np.ndarray
    energy: float
    charges: list[tuple[str, float, float]]
    capacitance: float | None
    info: SolveInfo
    device: str
    n_levels: int
    timing: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def group_voltages(model: GeometryModel, t: float | None = None) -> np.ndarray:
    """Dirichlet グループの電位ベクトル。t=None は直流分のみ (v1 /solve と同じ)。"""
    if t is None:
        return np.array([g.voltage for g in model.groups], dtype=np.float64)
    return np.array([g.value(t) for g in model.groups], dtype=np.float64)


def fill_fixed(op: LevelOperator, phi_unknown: np.ndarray, v_groups: np.ndarray) -> np.ndarray:
    """未知節点の解に、固定節点の Dirichlet 値と周期スレーブ値を埋めた全節点の電位を返す。"""
    phi = np.array(phi_unknown, dtype=np.float64, copy=True)
    fixed = op.mask == MASK_FIXED
    if np.any(fixed):
        phi[fixed] = v_groups[op.fixed_group[fixed]]
    nx, ny = op.grid.nx, op.grid.ny
    if op.periodic_x:
        slave = op.mask[:, nx] == MASK_SLAVE
        phi[slave, nx] = phi[slave, 0]
    if op.periodic_y:
        slave = op.mask[ny, :] == MASK_SLAVE
        phi[ny, slave] = phi[0, slave]
    return phi


def _axis_derivative(
    phi: np.ndarray,
    op: LevelOperator,
    v_groups: np.ndarray,
    axis: int,
) -> np.ndarray:
    """∂φ/∂(axis) を節点で求める (2 次精度の不等間隔中心差分、Dirichlet 結合は θh の距離)。

    - 両側に値があれば不等間隔中心差分 (境界との交点は距離 θh・値 V_k)。
    - 未知節点で片側が欠ける (Neumann 外周) なら法線微分 0。
    - 固定節点は取れる側の差分 (外周・導体表面の表示用。導体内部は隣も同電位なので 0)。
    """
    h = op.grid.dx if axis == 1 else op.grid.dy
    periodic = op.periodic_x if axis == 1 else op.periodic_y
    n = phi.shape[axis] - 1
    th = op.cut_theta
    gr = op.cut_group
    d_minus, d_plus = (DIR_W, DIR_E) if axis == 1 else (DIR_S, DIR_N)

    if periodic:
        core = np.take(phi, np.arange(n), axis=axis)          # スレーブを除いたマスター配列
        f_plus = np.roll(core, -1, axis=axis)
        f_minus = np.roll(core, 1, axis=axis)
        # スレーブ位置 (index n) へはマスター (index 0) の値を複製
        f_plus = np.concatenate([f_plus, np.take(f_plus, [0], axis=axis)], axis=axis)
        f_minus = np.concatenate([f_minus, np.take(f_minus, [0], axis=axis)], axis=axis)
        has_plus = np.ones(phi.shape, dtype=bool)
        has_minus = np.ones(phi.shape, dtype=bool)
    else:
        f_plus = np.roll(phi, -1, axis=axis)
        f_minus = np.roll(phi, 1, axis=axis)
        idx = np.arange(n + 1)
        shape = [1, 1]
        shape[axis] = n + 1
        has_plus = np.broadcast_to((idx < n).reshape(shape), phi.shape).copy()
        has_minus = np.broadcast_to((idx > 0).reshape(shape), phi.shape).copy()
    h_plus = np.full(phi.shape, h)
    h_minus = np.full(phi.shape, h)
    for d, fv, hv, hasv in ((d_plus, f_plus, h_plus, has_plus), (d_minus, f_minus, h_minus, has_minus)):
        cut = th[d] > 0.0
        if np.any(cut):
            fv[cut] = v_groups[gr[d][cut]]
            hv[cut] = th[d][cut] * h
            hasv[cut] = True

    both = has_plus & has_minus
    out = np.zeros(phi.shape)
    hp, hm = h_plus[both], h_minus[both]
    out[both] = (hm * hm * (f_plus[both] - phi[both]) + hp * hp * (phi[both] - f_minus[both])) / (
        hp * hm * (hp + hm)
    )
    fixed = op.mask == MASK_FIXED
    only_p = fixed & has_plus & ~has_minus
    only_m = fixed & has_minus & ~has_plus
    out[only_p] = (f_plus[only_p] - phi[only_p]) / h_plus[only_p]
    out[only_m] = (phi[only_m] - f_minus[only_m]) / h_minus[only_m]
    return out


def node_field(op: LevelOperator, phi: np.ndarray, v_groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """節点の電場 (Ex, Ey) = −∇φ。"""
    ex = -_axis_derivative(phi, op, v_groups, axis=1)
    ey = -_axis_derivative(phi, op, v_groups, axis=0)
    return ex, ey


def _energy_and_charges(
    model: GeometryModel, op: LevelOperator, phi: np.ndarray, v_groups: np.ndarray
) -> tuple[float, np.ndarray]:
    nx, ny = op.grid.nx, op.grid.ny
    # 通常の辺 (未知-未知) の二次形式
    phi_e = phi[:, 1:] if not op.periodic_x else np.concatenate([phi[:, 1:nx], phi[:, :1]], axis=1)
    w = 0.5 * float(np.sum(op.cx * (phi[:, :nx] - phi_e) ** 2))
    phi_n = phi[1:, :] if not op.periodic_y else np.concatenate([phi[1:ny, :], phi[:1, :]], axis=0)
    w += 0.5 * float(np.sum(op.cy * (phi[:ny, :] - phi_n) ** 2))
    # Dirichlet 結合
    coo = op.coupling.tocoo()
    dphi = phi.ravel()[coo.row] - v_groups[coo.col]
    w += 0.5 * float(np.sum(coo.data * dphi * dphi))
    q = np.bincount(coo.col, weights=coo.data * (-dphi), minlength=len(v_groups))
    factor = 2.0 * np.pi if model.radial_axis() is not None else 1.0
    return factor * w, factor * q


def solve_electrostatic(
    project: Project,
    *,
    h: float | None = None,
    device: Device | str | None = None,
    tol: float = 1e-10,
    t: float | None = None,
    grid: CartesianGrid | None = None,
) -> StaticSolution:
    """プロジェクトの静電場を v2 直交格子 EB エンジンで解く。

    h: 要求メッシュ幅 [m] (None なら project.mesh.size)。実際の格子は GMG に適した
    セル数に切り上げる (eb.grid.choose_cells)。t: RF/波形を含む時刻 t の電位で解く
    (None は直流分のみ = v1 /solve と同じ)。
    """
    t_start = time.perf_counter()
    dev = device if isinstance(device, Device) else get_device(device)
    model = GeometryModel(project)
    g = grid or make_grid(model.domain, float(h if h is not None else project.mesh.size))
    solver = GMGSolver(model, g, dev)
    op = solver.finest
    v = group_voltages(model, t)
    xp = dev.xp
    b_host = op.q_static.ravel() + (op.coupling @ v if len(v) else 0.0)
    b = dev.asarray(b_host.reshape(g.shape))
    t_solve = time.perf_counter()
    x, info = solver.solve(b, tol=tol)
    dev.synchronize()
    t_done = time.perf_counter()
    phi = fill_fixed(op, dev.to_host(x), v)
    ex, ey = node_field(op, phi, v)
    energy, q = _energy_and_charges(model, op, phi, v)
    charges = [(grp.label, float(grp.voltage if t is None else v[k]), float(q[k]))
               for k, grp in enumerate(model.groups)]
    capacitance = None
    if len(charges) >= 2 and not model.has_charge:
        levels = sorted({vv for _, vv, _ in charges})
        if len(levels) == 2:
            lo, hi = levels
            q_hi = sum(qq for _, vv, qq in charges if vv == hi)
            capacitance = q_hi / (hi - lo)
    del xp
    return StaticSolution(
        grid=g,
        phi=phi,
        ex=ex,
        ey=ey,
        energy=energy,
        charges=charges,
        capacitance=capacitance,
        info=info,
        device=dev.kind,
        n_levels=solver.n_levels,
        timing={
            "setup_s": solver.setup_s,
            "solve_s": t_done - t_solve,
            "total_s": time.perf_counter() - t_start,
        },
        warnings=list(op.warnings),
    )


def unknown_mask(op: LevelOperator) -> np.ndarray:
    return op.mask == MASK_UNKNOWN
