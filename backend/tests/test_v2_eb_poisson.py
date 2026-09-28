"""v2 直交格子 + 埋め込み境界 (EB) Poisson ソルバーの検証 (prompts/119 P2)。

解析解 (平行平板・直列/並列の誘電体・同軸円筒・同心球 (RZ)・一様場中の誘電体円柱・
周期境界) と v1 FEM との比較、GMG-PCG の格子非依存な収束、CPU/GPU の一致を確かめる。
GPU のテストは CUDA が使える環境でのみ実行する (CI は CPU 経路のみ)。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from es_sim.device import cuda_available
from es_sim.eb import choose_cells
from es_sim.field import solve_electrostatic
from es_sim.schema import Project

EPS0 = 8.8541878128e-12
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

needs_cuda = pytest.mark.skipif(not cuda_available(), reason="CUDA (CuPy) が使えない環境")


def _project(domain, regions=(), boundaries=(), h=0.002, coord="xy") -> Project:
    return Project.model_validate(
        {
            "version": 1,
            "unit": "m",
            "coord": coord,
            "geometry": {"domain": {"polygon": domain}, "regions": list(regions), "boundaries": list(boundaries)},
            "mesh": {"size": h},
        }
    )


def _plates(h=0.001, regions=()):
    return _project(
        [[0, 0], [0.1, 0], [0.1, 0.05], [0, 0.05]],
        regions,
        [
            {"edges": [3], "type": "dirichlet", "voltage": 0.0},
            {"edges": [1], "type": "dirichlet", "voltage": 100.0},
        ],
        h=h,
    )


def _ring_pieces(b: float, big: float, n: int = 2880) -> list[list[tuple[float, float]]]:
    """正方形 [-big, big]² から円板 r<b を除いた領域を覆う 4 つの単純多角形 (外導体用)。"""
    corners = [(big, big), (-big, big), (-big, -big), (big, -big)]
    pieces = []
    for q in range(4):
        a0 = math.pi / 4 + q * math.pi / 2
        th = np.linspace(a0, a0 + math.pi / 2, n // 4 + 1)
        arc = [(b * math.cos(t), b * math.sin(t)) for t in th]
        pieces.append(arc + [corners[(q + 1) % 4], corners[q]])
    return pieces


def _coax(h, a=0.01, b=0.04, L=0.05):
    regions = [{"id": "inner", "type": "conductor", "voltage": 1.0,
                "shape": {"kind": "circle", "center": [0.0, 0.0], "radius": a}}]
    for k, poly in enumerate(_ring_pieces(b, 1.2 * L)):
        regions.append({"id": f"outer{k}", "type": "conductor", "voltage": 0.0, "polygon": poly})
    return _project([[-L, -L], [L, -L], [L, L], [-L, L]], regions, [], h=h)


def _coax_errors(sol, a=0.01, b=0.04):
    X, Y = np.meshgrid(sol.grid.xs, sol.grid.ys)
    r = np.hypot(X, Y)
    sel = (r > a) & (r < b)
    exact = np.log(r[sel] / b) / math.log(a / b)
    c_exact = 2.0 * math.pi * EPS0 / math.log(b / a)
    return float(np.max(np.abs(sol.phi[sel] - exact))), (sol.capacitance - c_exact) / c_exact


# ---- 格子の選択 -----------------------------------------------------------------


def test_choose_cells_is_m_times_power_of_two_and_not_coarser():
    for length, h in [(0.1, 0.001), (0.067, 0.00052), (1.0, 0.3), (0.05, 0.05), (0.1, 1e-4)]:
        n = choose_cells(length, h)
        assert length / n <= h * (1 + 1e-12)          # 要求メッシュ幅以下
        m = n
        while m % 2 == 0 and m > 15:
            m //= 2
        assert m <= 15                                 # m·2^k (m ≤ 15)
        # 過剰に細かくしない: 最小の m·2^k を選ぶので切り上げは最大 (m+1)/m ≤ 9/8 (m ≥ 8)
        assert n <= math.ceil(length / h) * 9 / 8 + 1


# ---- 解析解 (CPU) ----------------------------------------------------------------


def test_parallel_plates_exact():
    s = solve_electrostatic(_plates(0.001), device="cpu")
    exact = 1000.0 * s.grid.xs
    assert np.max(np.abs(s.phi - exact[None, :])) < 1e-7
    assert np.allclose(s.ex, -1000.0, rtol=1e-8)
    assert np.allclose(s.ey, 0.0, atol=1e-6)
    c_exact = EPS0 * 0.05 / 0.1
    assert s.capacitance == pytest.approx(c_exact, rel=1e-8)
    assert s.energy == pytest.approx(0.5 * c_exact * 100.0**2, rel=1e-8)
    labels = [c[0] for c in s.charges]
    assert labels == ["edge3", "edge1"]              # v1 fem._label_order と同じ順
    assert s.charges[0][2] == pytest.approx(-s.charges[1][2], rel=1e-8)


def test_dielectric_series_non_aligned_interfaces_exact():
    """界面が格子と非整合な直列の誘電体スラブ: 1D 解析解に丸め誤差レベルで一致。"""
    a, b, er = 0.0313, 0.0671, 4.0
    slab = [{"id": "d", "type": "dielectric", "eps_r": er,
             "polygon": [[a, -0.01], [b, -0.01], [b, 0.06], [a, 0.06]]}]
    s = solve_electrostatic(_plates(0.001, slab), device="cpu")
    xs = s.grid.xs
    ed = 100.0 / (er * (a + 0.1 - b) + (b - a))
    ev = er * ed
    exact = np.where(xs < a, ev * xs, np.where(xs < b, ev * a + ed * (xs - a), ev * a + ed * (b - a) + ev * (xs - b)))
    assert np.max(np.abs(s.phi - exact[None, :])) < 1e-6
    assert s.capacitance == pytest.approx(EPS0 * 0.05 / (a + 0.1 - b + (b - a) / er), rel=1e-7)


def test_dielectric_parallel_non_aligned_capacitance():
    """電場と平行な誘電体層 (界面 y=c は格子と非整合): E は一様、容量は並列合成。"""
    c, er = 0.01737, 5.0
    layer = [{"id": "d", "type": "dielectric", "eps_r": er,
              "polygon": [[-0.01, -0.01], [0.11, -0.01], [0.11, c], [-0.01, c]]}]
    s = solve_electrostatic(_plates(0.001, layer), device="cpu")
    assert np.max(np.abs(s.phi - 1000.0 * s.grid.xs[None, :])) < 1e-6
    c_exact = EPS0 * ((0.05 - c) + er * c) / 0.1
    assert s.capacitance == pytest.approx(c_exact, rel=2e-3)   # 標本線 8 本の並列合成の分解能


def test_coax_second_order_convergence_and_capacitance():
    """円 (厳密 EB) の内導体 + 多角形の外導体: 電位誤差・容量誤差とも 2 次収束。"""
    errs, cerrs = [], []
    for n in (32, 64, 128):
        s = solve_electrostatic(_coax(0.1 / n), device="cpu")
        e, ce = _coax_errors(s)
        errs.append(e)
        cerrs.append(abs(ce))
    assert errs[-1] < 6e-4
    assert cerrs[-1] < 5e-4                     # v1 (h=2mm で 1% 以内) より桁違いに高精度
    assert errs[1] / errs[2] > 2.5              # ≈ 2 次 (理論 4、粗い側は漸近前)
    assert cerrs[1] / cerrs[2] > 2.5


def test_rz_concentric_spheres():
    """軸対称 (rz, y=r): 同心球コンデンサ φ = V(1/r − 1/b)/(1/a − 1/b)、C = 4πε0ab/(b−a)。"""
    a, b, L = 0.01, 0.04, 0.05
    th = np.linspace(math.pi, 0.0, 1441)
    outer = [(-1.2 * L, 0.0), (-b, 0.0)] + [(b * math.cos(t), b * math.sin(t)) for t in th[1:-1]] + \
        [(b, 0.0), (1.2 * L, 0.0), (1.2 * L, 1.2 * L), (-1.2 * L, 1.2 * L)]
    regions = [
        {"id": "inner", "type": "conductor", "voltage": 1.0, "shape": {"kind": "circle", "center": [0.0, 0.0], "radius": a}},
        {"id": "outer", "type": "conductor", "voltage": 0.0, "polygon": outer},
    ]
    s = solve_electrostatic(_project([[-L, 0], [L, 0], [L, L], [-L, L]], regions, [], h=L / 64, coord="rz"), device="cpu")
    Z, R = np.meshgrid(s.grid.xs, s.grid.ys)
    rho = np.hypot(Z, R)
    sel = (rho > a) & (rho < b)
    exact = (1 / rho[sel] - 1 / b) / (1 / a - 1 / b)
    assert np.max(np.abs(s.phi[sel] - exact)) < 3e-3
    c_exact = 4.0 * math.pi * EPS0 * a * b / (b - a)
    assert s.capacitance == pytest.approx(c_exact, rel=1.5e-3)


def test_dielectric_cylinder_in_uniform_field():
    """一様場中の誘電体円柱: 内部の場は一様で E_in = 2/(1+εr)·E0 (円柱半径 ≪ 領域)。"""
    er, R, L = 4.0, 0.004, 0.05
    regions = [{"id": "cyl", "type": "dielectric", "eps_r": er, "shape": {"kind": "circle", "center": [0.0, 0.0], "radius": R}}]
    p = _project(
        [[-L, -L], [L, -L], [L, L], [-L, L]], regions,
        [{"edges": [3], "type": "dirichlet", "voltage": 0.0}, {"edges": [1], "type": "dirichlet", "voltage": 100.0}],
        h=L / 256,
    )
    s = solve_electrostatic(p, device="cpu")
    X, Y = np.meshgrid(s.grid.xs, s.grid.ys)
    inside = np.hypot(X, Y) < 0.6 * R
    e0 = 100.0 / (2 * L)
    e_in = -s.ex[inside]
    assert np.mean(e_in) == pytest.approx(2.0 / (1.0 + er) * e0, rel=0.02)
    assert np.std(e_in) / np.mean(e_in) < 0.02          # 内部は一様
    assert np.max(np.abs(s.ey[inside])) < 0.02 * e0


def test_periodic_x_with_space_charge_matches_1d():
    """左右周期・上下 Dirichlet・x 方向に一様な電荷層: 解は x に依存せず 1D 解析解に一致。"""
    H, rho0, y1, y2 = 0.05, 1e-7, 0.0123, 0.0321
    regions = [{"id": "q", "type": "charge", "rho": rho0,
                "polygon": [[-0.01, y1], [0.11, y1], [0.11, y2], [-0.01, y2]]}]
    p = _project(
        [[0, 0], [0.1, 0], [0.1, H], [0, H]], regions,
        [
            {"edges": [3, 1], "type": "periodic"},
            {"edges": [0], "type": "dirichlet", "voltage": 0.0},
            {"edges": [2], "type": "dirichlet", "voltage": 10.0},
        ],
        h=0.001,
    )
    s = solve_electrostatic(p, device="cpu")
    assert np.max(np.abs(s.phi - s.phi[:, :1])) < 1e-8            # x に依存しない
    ys = s.grid.ys

    # 1D: -ε0 φ'' = ρ(y), φ(0)=0, φ(H)=10 の解析解
    def phi1d(y):
        # 電荷の一次・二次モーメントで区分二次式を組む
        def f(y):
            s1 = np.clip(y, y1, y2) - y1
            return -rho0 / EPS0 * (np.where(y > y1, 0.5 * s1 * s1 + s1 * np.clip(y - y2, 0, None), 0.0))
        fh = f(np.array(H))
        return f(y) + (10.0 - fh) * y / H

    exact = phi1d(ys)
    assert np.max(np.abs(s.phi[:, 5] - exact)) < 2e-3 * np.max(np.abs(exact))


def test_all_neumann_periodic_problem_is_solved_with_zero_mean():
    """Dirichlet 節点の無い問題 (全周期) は零空間を除いて解け、解は体積平均 0。"""
    regions = [
        {"id": "p", "type": "charge", "rho": 1e-8, "shape": {"kind": "circle", "center": [0.03, 0.025], "radius": 0.008}},
        {"id": "n", "type": "charge", "rho": -1e-8, "shape": {"kind": "circle", "center": [0.07, 0.025], "radius": 0.008}},
    ]
    p = _project(
        [[0, 0], [0.1, 0], [0.1, 0.05], [0, 0.05]], regions,
        [{"edges": [3, 1], "type": "periodic"}, {"edges": [0, 2], "type": "periodic"}],
        h=0.001,
    )
    s = solve_electrostatic(p, device="cpu")
    assert s.info.converged
    assert abs(np.mean(s.phi[:-1, :-1])) < 1e-6 * np.max(np.abs(s.phi))
    # 正電荷の中心が高電位、負電荷の中心が低電位 (対称)
    j = np.argmin(np.abs(s.grid.ys - 0.025))
    i_p = np.argmin(np.abs(s.grid.xs - 0.03))
    i_n = np.argmin(np.abs(s.grid.xs - 0.07))
    assert s.phi[j, i_p] > 0 > s.phi[j, i_n]
    assert s.phi[j, i_p] == pytest.approx(-s.phi[j, i_n], rel=2e-2)


# ---- v1 FEM との比較 ----------------------------------------------------------------


def test_matches_v1_fem_on_parallel_plates_example():
    """examples/parallel_plates.json (誘電体ブロック入り) で v1 P1-FEM と一致。"""
    from scipy.interpolate import LinearNDInterpolator

    from es_sim.fem import solve as fem_solve
    from es_sim.meshing import generate_mesh

    data = json.loads((EXAMPLES / "parallel_plates.json").read_text(encoding="utf-8"))
    data["mesh"]["size"] = 0.0007
    project = Project.model_validate(data)
    v1 = fem_solve(project, generate_mesh(project))
    s = solve_electrostatic(project, h=0.0007, device="cpu")
    mesh = generate_mesh(project)
    interp = LinearNDInterpolator(mesh.nodes, v1.v)
    X, Y = np.meshgrid(s.grid.xs, s.grid.ys)
    v1_on_grid = interp(X, Y)
    ok = np.isfinite(v1_on_grid)
    diff = np.abs(s.phi[ok] - v1_on_grid[ok])
    assert np.max(diff) < 0.3            # V (電圧範囲 100 V の 0.3%)
    assert np.mean(diff) < 0.03
    assert s.energy == pytest.approx(v1.energy, rel=5e-3)
    v1q = {lab: q for lab, _, q in v1.charges}
    for lab, _, q in s.charges:
        assert q == pytest.approx(v1q[lab], rel=5e-3)


# ---- ソルバーの性質 -------------------------------------------------------------------


def test_gmg_pcg_iterations_are_mesh_independent():
    its = []
    for n in (64, 256):
        s = solve_electrostatic(_coax(0.1 / n), device="cpu")
        assert s.info.converged
        its.append(s.info.iterations)
    assert max(its) <= 12
    assert its[1] <= its[0] + 3


@needs_cuda
def test_gpu_matches_cpu():
    p = _coax(0.1 / 128)
    s_cpu = solve_electrostatic(p, device="cpu", tol=1e-12)
    s_gpu = solve_electrostatic(p, device="cuda", tol=1e-12)
    assert s_gpu.device == "cuda"
    assert np.max(np.abs(s_gpu.phi - s_cpu.phi)) < 1e-9
    assert s_gpu.capacitance == pytest.approx(s_cpu.capacitance, rel=1e-9)


@needs_cuda
def test_gpu_large_grid_converges_fast():
    s = solve_electrostatic(_coax(0.1 / 1024), device="cuda")
    assert s.grid.nx == 1024
    assert s.info.converged and s.info.iterations <= 12
    e, ce = _coax_errors(s)
    assert abs(ce) < 5e-5
