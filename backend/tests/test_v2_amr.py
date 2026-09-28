"""v2 AMR (ブロック構造の局所細分化) の合成格子静電場ソルバーの検証 (prompts/121)。

- 細分化なし / 領域全体の細分化は一様格子の EB ソルバーと一致する
- 粗細界面の整合: 線形解の厳密再現、電場に平行な誘電体層が界面を横切っても厳密、
  二次式の厳密解で界面を含めて 2 次収束 (直交座標・軸対称で軸に接する界面)
- 演算子の対称性、電荷保存 (ΣQ = −Σq)、2W = Σ Q_k V_k
- 細線同軸・同心球 (RZ)・誘電体円柱: 一様格子より少ない未知数で同等以上の精度
- 周期境界 (巻き戻しの proper nesting)・Dirichlet の無い特異問題
- 階層の不変条件 (proper nesting・葉セルの分割・2:1 バランス) を乱数の入力で確かめる
すべて CPU (pyamg の AMG-CG)。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from es_sim.amr import AmrSpec, assemble_composite, build_hierarchy, dilate, solve_electrostatic_amr
from es_sim.field import solve_electrostatic
from es_sim.geom.model import GeometryModel
from es_sim.schema import Project

EPS0 = 8.8541878128e-12
PLATES = [[0, 0], [0.1, 0], [0.1, 0.05], [0, 0.05]]
LR = [{"edges": [3], "type": "dirichlet", "voltage": 0.0}, {"edges": [1], "type": "dirichlet", "voltage": 100.0}]
BIG = [[-0.01, -0.01], [0.11, -0.01], [0.11, 0.11], [-0.01, 0.11]]   # domain を覆う多角形 (一様な電荷など)


def _project(domain, regions=(), boundaries=(), h=0.002, coord="xy", amr=None) -> Project:
    mesh: dict = {"size": h, "mode": "cartesian"}
    if amr is not None:
        mesh["amr"] = amr
    return Project.model_validate({
        "version": 1, "unit": "m", "coord": coord,
        "geometry": {"domain": {"polygon": domain}, "regions": list(regions), "boundaries": list(boundaries)},
        "mesh": mesh,
    })


def _ring_pieces(b: float, big: float, n: int = 2880) -> list[list[tuple[float, float]]]:
    """正方形 [-big, big]² から円板 r<b を除いた領域を覆う 4 つの単純多角形 (外導体用)。"""
    corners = [(big, big), (-big, big), (-big, -big), (big, -big)]
    pieces = []
    for q in range(4):
        a0 = math.pi / 4 + q * math.pi / 2
        th = np.linspace(a0, a0 + math.pi / 2, n // 4 + 1)
        pieces.append([(b * math.cos(t), b * math.sin(t)) for t in th] + [corners[(q + 1) % 4], corners[q]])
    return pieces


def _coax(h, a=0.01, b=0.04, L=0.05, amr=None) -> Project:
    regions = [{"id": "inner", "type": "conductor", "voltage": 1.0,
                "shape": {"kind": "circle", "center": [0.0, 0.0], "radius": a}}]
    for k, poly in enumerate(_ring_pieces(b, 1.2 * L)):
        regions.append({"id": f"outer{k}", "type": "conductor", "voltage": 0.0, "polygon": poly})
    return _project([[-L, -L], [L, -L], [L, L], [-L, L]], regions, [], h=h, amr=amr)


def _coax_err(nodes, phi, a=0.01, b=0.04) -> float:
    r = np.hypot(nodes[:, 0], nodes[:, 1])
    sel = (r > a) & (r < b)
    exact = np.log(r[sel] / b) / math.log(a / b)
    return float(np.max(np.abs(phi[sel] - exact)))


def _spec(level, regions, bf=4):
    return AmrSpec(max_level=level, refine_boundaries=False, blocking_factor=bf, regions=tuple(regions))


# ---- 一様格子との一致 --------------------------------------------------------------------


def test_no_refinement_equals_uniform_eb_solver():
    p = _coax(0.1 / 64)
    u = solve_electrostatic(p, device="cpu")
    s = solve_electrostatic_amr(p, spec=AmrSpec(max_level=0))
    assert s.hier.n_levels == 1 and not s.op.hanging.any()
    X, Y = np.meshgrid(u.grid.xs, u.grid.ys)
    phi, _, _ = s.sample(X.ravel(), Y.ravel())
    assert np.max(np.abs(phi - u.phi.ravel())) < 1e-7
    assert s.capacitance == pytest.approx(u.capacitance, rel=1e-8)
    assert s.energy == pytest.approx(u.energy, rel=1e-8)


def test_full_domain_refinement_equals_uniform_fine_grid():
    """領域全体をレベル 2 に細分化 = 4 倍細かい一様格子 (葉セルは全て最細レベル)。"""
    s = solve_electrostatic_amr(_coax(0.1 / 32), spec=_spec(2, [(-1, -1, 1, 1, 2)], bf=8))
    u = solve_electrostatic(_coax(0.1 / 128), device="cpu")
    assert s.leaf_counts[:2] == [0, 0]
    X, Y = np.meshgrid(u.grid.xs, u.grid.ys)
    phi, _, _ = s.sample(X.ravel(), Y.ravel())
    assert np.max(np.abs(phi - u.phi.ravel())) < 1e-7
    assert s.capacitance == pytest.approx(u.capacitance, rel=1e-8)


# ---- 粗細界面の整合 ------------------------------------------------------------------------


def test_refinement_region_reproduces_linear_field_exactly():
    p = _project(PLATES, [], LR, h=0.002,
                 amr={"max_level": 2, "regions": [{"p1": [0.03, 0.01], "p2": [0.06, 0.03], "level": 2}]})
    s = solve_electrostatic_amr(p)
    assert s.op.hanging.sum() > 0 and s.leaf_counts[2] > 0
    assert np.max(np.abs(s.phi - 1000.0 * s.op.xy[:, 0])) < 1e-6
    assert np.max(np.abs(s.phi_display - 1000.0 * s.nodes[:, 0])) < 1e-6
    assert s.capacitance == pytest.approx(EPS0 * 0.05 / 0.1, rel=1e-9)
    A = s.op.A_c
    assert abs(A - A.T).max() <= 1e-14 * abs(A).max()


@pytest.mark.parametrize("c", [5 * (0.1 / 32) / 2, 0.01737])
def test_dielectric_layer_parallel_to_field_across_interface_is_exact(c):
    """電場に平行な誘電体層 (界面 y=c) が縦の粗細界面を横切る: 解は φ = 1000x のまま。

    c = 7.8125 mm はぶら下がり節点の高さ。拘束の重みを線形補間 (½) にすると、界面に平行な
    流束の配分が ε の比と合わず 0.3 V ほどずれる (重みは半チューブの ∫ε dA の比)。
    """
    layer = [{"id": "d", "type": "dielectric", "eps_r": 4.0,
              "polygon": [[-0.01, -0.01], [0.11, -0.01], [0.11, c], [-0.01, c]]}]
    s = solve_electrostatic_amr(_project(PLATES, layer, LR, h=0.1 / 32),
                                spec=_spec(1, [(0.03, 0.0, 0.06, 0.05, 1)]))
    assert s.op.hanging.sum() > 0
    assert np.max(np.abs(s.phi - 1000.0 * s.op.xy[:, 0])) < 1e-6
    c_exact = EPS0 * (4.0 * c + (0.05 - c)) / 0.1
    assert s.capacitance == pytest.approx(c_exact, rel=2e-3)     # 標本線 8 本の並列合成の分解能


def _space_charge_plates(n: int, axis: str) -> Project:
    edges = [3, 1] if axis == "x" else [0, 2]
    bnd = [{"edges": [e], "type": "dirichlet", "voltage": 0.0} for e in edges]
    sc = [{"id": "sc", "type": "charge", "rho": 1e-6, "polygon": BIG}]
    return _project([[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]], sc, bnd, h=0.1 / n)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_second_order_convergence_across_coarse_fine_interfaces(axis):
    """一様な空間電荷の平行平板 φ = ρ/(2ε0)·s(0.1−s): 一様格子は厳密、AMR は界面の補間誤差が
    2 次で減る。"""
    errs = []
    for n in (16, 32, 64):
        p = _space_charge_plates(n, axis)
        s = solve_electrostatic_amr(p, spec=_spec(1, [(0.03, 0.03, 0.06, 0.06, 1)]))
        coord = s.op.xy[:, 0] if axis == "x" else s.op.xy[:, 1]
        exact = 1e-6 / (2 * EPS0) * coord * (0.1 - coord)
        errs.append(float(np.max(np.abs(s.phi - exact))) / float(np.max(exact)))
    u = solve_electrostatic(_space_charge_plates(32, axis), device="cpu")
    X, Y = np.meshgrid(u.grid.xs, u.grid.ys)
    cu = X if axis == "x" else Y
    assert np.max(np.abs(u.phi - 1e-6 / (2 * EPS0) * cu * (0.1 - cu))) < 1e-9 * np.max(np.abs(u.phi))
    assert errs[-1] < 3e-4
    assert errs[0] / errs[1] > 3.5 and errs[1] / errs[2] > 3.5


@pytest.mark.parametrize("coord", ["rz", "rz_x0"])
def test_rz_interface_meeting_axis_is_second_order(coord):
    """軸対称で軸 (r=0) に接する粗細界面: 軸方向に変化する厳密解 φ = ρ/(2ε0)·z(Z−z) が 2 次収束。

    径方向の辺のぶら下がり節点を線形補間で拘束すると軸上の節点で流束収支が合わず 1 次になる
    (w_a = ½ − h/(8 r_m) で解消)。
    """
    Z, R = 0.1, 0.05
    if coord == "rz":
        dom, edges, reg = [[0, 0], [Z, 0], [Z, R], [0, R]], [3, 1], (0.03, 0.0, 0.06, 0.02, 1)
    else:
        dom, edges, reg = [[0, 0], [R, 0], [R, Z], [0, Z]], [0, 2], (0.0, 0.03, 0.02, 0.06, 1)
    errs = []
    for n in (16, 32, 64):
        p = _project(dom, [{"id": "sc", "type": "charge", "rho": 1e-6, "polygon": BIG}],
                     [{"edges": [e], "type": "dirichlet", "voltage": 0.0} for e in edges], h=R / n, coord=coord)
        s = solve_electrostatic_amr(p, spec=_spec(1, [reg]))
        z = s.op.xy[:, 0] if coord == "rz" else s.op.xy[:, 1]
        exact = 1e-6 / (2 * EPS0) * z * (Z - z)
        errs.append(float(np.max(np.abs(s.phi - exact))) / float(np.max(exact)))
    assert errs[-1] < 1e-4
    assert errs[0] / errs[1] > 3.5 and errs[1] / errs[2] > 3.5


# ---- 保存則 --------------------------------------------------------------------------------


def test_charges_conserved_and_consistent_with_energy():
    """細分化領域が Dirichlet 辺に接する (端点が電極のぶら下がり節点がある) 場合も
    Q_左 + Q_右 = 0、2W = Σ Q_k V_k が厳密に成り立ち、容量は細かい一様格子と一致する。"""
    diel = [{"id": "d", "type": "dielectric", "eps_r": 4.0,
             "polygon": [[0.004, 0.012], [0.02, 0.012], [0.02, 0.026], [0.004, 0.026]]}]
    p = _project(PLATES, diel, LR, h=0.002, amr={"max_level": 2, "refine_boundaries": False,
                                                "regions": [{"p1": [0.0, 0.01], "p2": [0.03, 0.03], "level": 2}]})
    s = solve_electrostatic_amr(p)
    assert s.op.C.nnz > 0                                   # 端点が電極のぶら下がり節点
    q = np.array([qq for _, _, qq in s.charges])
    v = np.array([vv for _, vv, _ in s.charges])
    assert abs(q.sum()) < 1e-8 * abs(q).max()
    assert 2.0 * s.energy == pytest.approx(float(np.dot(q, v)), rel=1e-10)
    u = solve_electrostatic(_project(PLATES, diel, LR, h=0.0005), device="cpu")
    assert s.capacitance == pytest.approx(u.capacitance, rel=1e-4)


def test_space_charge_is_balanced_by_electrode_charges():
    """電荷の円板の一部だけ細分化: ΣQ_k = −Σq (厳密) ≈ −ρ·πR² (円の求積の精度)。"""
    blob = [{"id": "q", "type": "charge", "rho": 1e-6, "shape": {"kind": "circle", "center": [0.05, 0.025], "radius": 0.008}}]
    p = _project(PLATES, blob, LR, h=0.002, amr={"max_level": 2, "refine_boundaries": False,
                                                "regions": [{"p1": [0.05, 0.02], "p2": [0.07, 0.04], "level": 2}]})
    s = solve_electrostatic_amr(p)
    q = sum(qq for _, _, qq in s.charges)
    total = 1e-6 * math.pi * 0.008**2
    assert abs(q + s.op.q_full.sum()) < 1e-8 * total
    assert q == pytest.approx(-total, rel=2e-3)


# ---- 精度と未知数 ------------------------------------------------------------------------


def test_thin_wire_amr_beats_uniform_grid_with_fewer_unknowns():
    """細線 (a=1mm) の同軸: 線の近傍だけ 3 レベル細分化すると、節点数の多い一様格子 (128²) より
    はるかに高精度 (一様 512² と同程度)。"""
    a, b = 0.001, 0.04
    c_exact = 2 * math.pi * EPS0 / math.log(b / a)
    s = solve_electrostatic_amr(_coax(0.1 / 64, a=a), spec=_spec(3, [(-0.004, -0.004, 0.004, 0.004, 3)], bf=8))
    err_amr = _coax_err(s.nodes, s.phi_display, a=a)
    u = solve_electrostatic(_coax(0.1 / 128, a=a), device="cpu")
    X, Y = np.meshgrid(u.grid.xs, u.grid.ys)
    err_uni = _coax_err(np.stack([X.ravel(), Y.ravel()], 1), u.phi.ravel(), a=a)
    assert s.n_unknowns < u.grid.n_nodes
    assert err_amr < 6e-4 and err_amr < err_uni / 4
    assert abs(s.capacitance - c_exact) / c_exact < 5e-4
    assert s.info.converged and s.info.iterations < 20


def test_coax_boundary_refinement_improves_accuracy():
    """境界近傍の細分化で誤差が減り、容量は細分化なしより 1 桁以上正確になる。"""
    c_exact = 2 * math.pi * EPS0 / math.log(4.0)
    errs, cerrs = [], []
    for lvl in (0, 1, 2):
        s = solve_electrostatic_amr(_coax(0.1 / 64), spec=AmrSpec(max_level=lvl))
        errs.append(_coax_err(s.nodes, s.phi_display))
        cerrs.append(abs(s.capacitance - c_exact) / c_exact)
    assert errs[2] < errs[1] < errs[0]
    assert cerrs[2] < cerrs[0] / 4 and cerrs[1] < cerrs[0] / 4


def test_rz_concentric_spheres_with_boundary_refinement():
    a, b, L = 0.01, 0.04, 0.05
    th = np.linspace(math.pi, 0.0, 1441)
    outer = [(-1.2 * L, 0.0), (-b, 0.0)] + [(b * math.cos(t), b * math.sin(t)) for t in th[1:-1]] + \
        [(b, 0.0), (1.2 * L, 0.0), (1.2 * L, 1.2 * L), (-1.2 * L, 1.2 * L)]
    regions = [
        {"id": "inner", "type": "conductor", "voltage": 1.0, "shape": {"kind": "circle", "center": [0.0, 0.0], "radius": a}},
        {"id": "outer", "type": "conductor", "voltage": 0.0, "polygon": outer},
    ]
    c_exact = 4.0 * math.pi * EPS0 * a * b / (b - a)
    errs = []
    for lvl in (1, 2):
        s = solve_electrostatic_amr(_project([[-L, 0], [L, 0], [L, L], [-L, L]], regions, [], h=L / 32, coord="rz",
                                             amr={"max_level": lvl}))
        rho = np.hypot(s.nodes[:, 0], s.nodes[:, 1])
        sel = (rho > a) & (rho < b)
        exact = (1 / rho[sel] - 1 / b) / (1 / a - 1 / b)
        errs.append(float(np.max(np.abs(s.phi_display[sel] - exact))))
        assert abs(s.capacitance - c_exact) / c_exact < 6e-4
        q = [qq for _, _, qq in s.charges]
        assert abs(q[0] + q[1]) < 1e-7 * abs(q[0])
    assert errs[1] < errs[0] and errs[1] < 1.2e-3
    assert s.n_unknowns < 257 * 129 / 3                 # 同じ最細幅の一様格子 (L/128) の 1/3 未満


def test_dielectric_cylinder_with_interface_refinement():
    """一様場中の誘電体円柱 (界面近傍を細分化): 内部の場は一様で E_in = 2/(1+εr)·E0。"""
    er, R, L = 4.0, 0.004, 0.05
    cyl = [{"id": "cyl", "type": "dielectric", "eps_r": er, "shape": {"kind": "circle", "center": [0.0, 0.0], "radius": R}}]
    xs = np.linspace(-0.6 * R, 0.6 * R, 41)
    X, Y = np.meshgrid(xs, xs)
    m = np.hypot(X, Y) < 0.6 * R
    e0 = 100.0 / (2 * L)
    stds = []
    for lvl in (0, 2):
        s = solve_electrostatic_amr(_project([[-L, -L], [L, -L], [L, L], [-L, L]], cyl, LR, h=2 * L / 64,
                                             amr={"max_level": lvl}))
        _, ex, ey = s.sample(X[m], Y[m])
        e_in = -ex
        stds.append(float(np.std(e_in) / np.mean(e_in)))
        assert np.mean(e_in) == pytest.approx(2.0 / (1.0 + er) * e0, rel=0.025)
    assert stds[1] < 2e-3 and stds[1] < stds[0] / 3
    assert np.max(np.abs(ey)) < 2e-3 * e0
    assert s.n_unknowns < 257 * 257 / 10


# ---- 解に基づく適応細分化 (prompts/122) ----------------------------------------------------


def test_adaptive_refinement_concentrates_on_space_charge():
    """接地平板間の電荷円板: 適応細分化は電荷の周りだけを最大レベルまで細かくし、誤差指標が減る。"""
    blob = [{"id": "q", "type": "charge", "rho": 1e-6, "shape": {"kind": "circle", "center": [0.05, 0.025], "radius": 0.006}}]
    bnd = [{"edges": [3], "type": "dirichlet", "voltage": 0.0}, {"edges": [1], "type": "dirichlet", "voltage": 0.0}]
    p = _project(PLATES, blob, bnd, h=0.1 / 32, amr={"max_level": 3, "refine_boundaries": False,
                                                    "adaptive": True, "adapt_tol": 1e-3, "adapt_iters": 4})
    s = solve_electrostatic_amr(p, device="cpu")
    hist = s.adapt_history
    assert s.hier.n_levels == 4 and len(hist) >= 3
    assert all(a["eta_rel"] > b["eta_rel"] for a, b in zip(hist, hist[1:]))
    # 最細レベルは電荷の近傍だけ
    for x0, y0, x1, y1 in s.hier.level_boxes(3):
        assert x0 >= 0.05 - 0.0125 and x1 <= 0.05 + 0.0125
    # 細かい一様格子 (最細と同じ幅) の解との差は、適応なし (基本格子) より小さい
    ref = solve_electrostatic(_project(PLATES, blob, bnd, h=0.1 / 256), device="cpu")
    X, Y = np.meshgrid(ref.grid.xs, ref.grid.ys)
    coarse = solve_electrostatic_amr(_project(PLATES, blob, bnd, h=0.1 / 32), spec=AmrSpec(max_level=0), device="cpu")
    scale = float(np.max(np.abs(ref.phi)))
    err = lambda sol: float(np.nanmax(np.abs(sol.sample(X.ravel(), Y.ravel())[0] - ref.phi.ravel()))) / scale  # noqa: E731
    assert err(s) < err(coarse) / 5
    assert s.n_unknowns < ref.grid.n_nodes / 3


def test_adaptive_refinement_on_coax_beats_boundary_only_refinement():
    c_exact = 2 * math.pi * EPS0 / math.log(4.0)
    bnd = solve_electrostatic_amr(_coax(0.1 / 64), spec=AmrSpec(max_level=3), device="cpu")
    ada = solve_electrostatic_amr(_coax(0.1 / 64), spec=AmrSpec(max_level=3, refine_boundaries=False, adaptive=True,
                                                               adapt_tol=1e-3, adapt_iters=4), device="cpu")
    err_b = abs(bnd.capacitance - c_exact) / c_exact
    err_a = abs(ada.capacitance - c_exact) / c_exact
    assert ada.n_unknowns < 0.6 * bnd.n_unknowns
    assert err_a < err_b and err_a < 1e-4
    assert _coax_err(ada.nodes, ada.phi_display) < 3e-4


# ---- 周期・特異 ----------------------------------------------------------------------------


def test_periodic_refinement_wraps_and_matches_1d():
    H, rho0, y1, y2 = 0.05, 1e-7, 0.0123, 0.0321
    lay = [{"id": "q", "type": "charge", "rho": rho0, "polygon": [[-0.01, y1], [0.11, y1], [0.11, y2], [-0.01, y2]]}]
    bnd = [{"edges": [3, 1], "type": "periodic"}, {"edges": [0], "type": "dirichlet", "voltage": 0.0},
           {"edges": [2], "type": "dirichlet", "voltage": 10.0}]
    p = _project(PLATES, lay, bnd, h=0.1 / 64)
    s = solve_electrostatic_amr(p, spec=_spec(2, [(0.09, 0.01, 0.1, 0.03, 2)], bf=8))
    assert s.hier.refined[0][:, 0].any() and s.hier.refined[0][:, -1].any()    # 周期の反対側へ緩衝帯

    def f(y):
        s1 = np.clip(y, y1, y2) - y1
        return -rho0 / EPS0 * np.where(y > y1, 0.5 * s1 * s1 + s1 * np.clip(y - y2, 0, None), 0.0)

    y = s.op.xy[:, 1]
    exact = f(y) + (10.0 - f(np.array(H))) * y / H
    assert np.max(np.abs(s.phi - exact)) < 2e-3 * np.max(np.abs(exact))


def test_all_periodic_singular_problem():
    regions = [
        {"id": "p", "type": "charge", "rho": 1e-8, "shape": {"kind": "circle", "center": [0.03, 0.025], "radius": 0.008}},
        {"id": "n", "type": "charge", "rho": -1e-8, "shape": {"kind": "circle", "center": [0.07, 0.025], "radius": 0.008}},
    ]
    p = _project(PLATES, regions, [{"edges": [3, 1], "type": "periodic"}, {"edges": [0, 2], "type": "periodic"}],
                 h=0.002, amr={"max_level": 1})
    s = solve_electrostatic_amr(p)
    assert s.op.singular and s.info.converged
    ph, _, _ = s.sample(np.array([0.03, 0.07]), np.array([0.025, 0.025]))
    assert ph[0] > 0 > ph[1]
    assert ph[0] == pytest.approx(-ph[1], rel=1e-6)


# ---- 表示用メッシュ -------------------------------------------------------------------------


def test_display_mesh_covers_domain_and_matches_sample():
    p = _project(PLATES, [{"id": "d", "type": "dielectric", "eps_r": 3.0,
                           "shape": {"kind": "circle", "center": [0.05, 0.025], "radius": 0.01}}], LR,
                 h=0.004, amr={"max_level": 2})
    s = solve_electrostatic_amr(p)
    P = s.nodes[s.triangles]
    area = 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                        - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    assert area.sum() == pytest.approx(0.1 * 0.05, rel=1e-12)
    assert len(set(np.round(area, 14))) >= 3                        # 3 レベルの三角形
    assert set(s.tri_region.tolist()) == {-1, 0}
    phi, _, _ = s.sample(s.nodes[:, 0], s.nodes[:, 1])
    assert np.max(np.abs(phi - s.phi_display)) < 1e-9
    assert len(s.e_tri) == len(s.triangles)


# ---- 階層の不変条件 ------------------------------------------------------------------------


def _leaf_level_map(hier) -> np.ndarray:
    L = hier.max_level
    nyf, nxf = hier.base.ny << L, hier.base.nx << L
    level = np.full((nyf, nxf), -1, dtype=np.int64)
    count = np.zeros((nyf, nxf), dtype=np.int64)
    for lvl in range(hier.n_levels):
        k = 1 << (L - lvl)
        m = np.repeat(np.repeat(hier.leaf_mask(lvl), k, axis=0), k, axis=1)
        level[m] = lvl
        count += m
    assert np.all(count == 1), "葉セルが domain を重なりなく覆っていない"
    return level


def _random_project(rng, periodic_x: bool, periodic_y: bool) -> Project:
    regions = []
    for k in range(int(rng.integers(1, 4))):
        cx, cy, r = rng.uniform(0.01, 0.09), rng.uniform(0.01, 0.04), rng.uniform(0.002, 0.008)
        if rng.random() < 0.5:
            regions.append({"id": f"c{k}", "type": "conductor", "voltage": float(rng.uniform(-5, 5)),
                            "shape": {"kind": "circle", "center": [cx, cy], "radius": r}})
        else:
            regions.append({"id": f"d{k}", "type": "dielectric", "eps_r": 3.0,
                            "polygon": [[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]]})
    bnd = []
    bnd.append({"edges": [3, 1], "type": "periodic"} if periodic_x else
               {"edges": [3], "type": "dirichlet", "voltage": 0.0})
    if periodic_y:
        bnd.append({"edges": [0, 2], "type": "periodic"})
    amr_regions = []
    for _ in range(int(rng.integers(0, 3))):
        x0, y0 = rng.uniform(0.0, 0.1), rng.uniform(0.0, 0.05)
        amr_regions.append({"p1": [x0, y0], "p2": [x0 + rng.uniform(0.001, 0.03), y0 + rng.uniform(0.001, 0.02)],
                            "level": int(rng.integers(1, 4))})
    amr = {"max_level": int(rng.integers(0, 4)), "refine_boundaries": bool(rng.random() < 0.7),
           "buffer_cells": int(rng.integers(0, 4)), "blocking_factor": int(rng.choice([2, 4, 8])),
           "regions": amr_regions}
    return _project(PLATES, regions, bnd, h=float(rng.choice([0.1 / 32, 0.1 / 48, 0.1 / 64])), amr=amr)


@pytest.mark.parametrize("seed", range(10))
def test_hierarchy_invariants_random(seed):
    rng = np.random.default_rng(seed)
    p = _random_project(rng, periodic_x=bool(seed % 2), periodic_y=bool(seed % 3 == 0))
    model = GeometryModel(p)
    hier = build_hierarchy(p, model=model)
    px, py = hier.px, hier.py
    # proper nesting: 細分化されたレベル l+1 のブロックの 8 近傍はレベル l+1 の領域内
    for lvl in range(hier.max_level - 1):
        need = dilate(hier.refined[lvl + 1], 1, px, py)
        region = np.repeat(np.repeat(hier.refined[lvl], 2, axis=0), 2, axis=1)
        assert not np.any(need & ~region)
    # 葉セルは domain を重なりなく覆う (面積も一致)
    level = _leaf_level_map(hier)
    b = hier.base
    area = sum(n * (b.dx * b.dy) / 4**lvl for lvl, n in enumerate(hier.n_leaf_cells()))
    assert area == pytest.approx((b.x1 - b.x0) * (b.y1 - b.y0), rel=1e-12)
    # 2:1 バランス: 辺・角で接する葉セルのレベル差は高々 1
    for dj, di in ((0, 1), (1, 0), (1, 1), (1, -1)):
        other = level
        other = np.roll(other, -di, axis=1) if px else _shift_fill(other, -di, 1)
        other = np.roll(other, -dj, axis=0) if py else _shift_fill(other, -dj, 0)
        valid = other >= 0
        assert np.all(np.abs(level - other)[valid] <= 1)
    # 合成格子: ぶら下がり節点の端点は通常節点 (組めること)、A_c は対称、拘束の重みの和は 1
    op = assemble_composite(model, hier)
    A = op.A_c
    if A.nnz:
        assert abs(A - A.T).max() <= 1e-12 * abs(A).max()
    rows = np.asarray(op.P.sum(axis=1)).ravel() + np.asarray(op.C.sum(axis=1)).ravel()
    assert np.allclose(rows, 1.0)


def _shift_fill(a: np.ndarray, s: int, axis: int) -> np.ndarray:
    """非周期のずらし (はみ出した所は −1)。"""
    out = np.full_like(a, -1)
    n = a.shape[axis]
    src = [slice(None)] * 2
    dst = [slice(None)] * 2
    if s > 0:
        src[axis], dst[axis] = slice(0, n - s), slice(s, n)
    elif s < 0:
        src[axis], dst[axis] = slice(-s, n), slice(0, n + s)
    else:
        return a.copy()
    out[tuple(dst)] = a[tuple(src)]
    return out
