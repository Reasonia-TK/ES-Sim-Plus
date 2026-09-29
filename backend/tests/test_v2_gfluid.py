"""v2 流体 2D (直交格子 + 埋め込み境界、es_sim.gfluid) のテスト (prompts/125)。

v1 (tests/test_fluid2d.py) と同じ検証を直交格子版で行い、さらに輸送グラフの幾何
(気体体積・壁面積の解析値との一致)、v1 構造格子との突き合わせ、Poisson の LU と GMG の一致、
server / batch の配線を確かめる。GPU は無くてもよい (Poisson の GMG は CPU でも動く)。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import es_sim.gfluid.simulation as gsim
import es_sim.server as server
from es_sim.batch import run_files
from es_sim.fluid1d import Fluid1dSimulation
from es_sim.fluid2d import Fluid2dSimulation, build_fluid2d_result
from es_sim.gfluid import CartesianFluid2dSimulation, make_fluid2d_simulation
from es_sim.schema import (
    BoundaryCondition,
    Domain,
    Fluid1dSettings,
    Fluid2dSettings,
    Geometry,
    MeshSettings,
    Pic1dElectrode,
    Project,
    VoltageRF,
)

W, H = 0.02, 0.01
RF = {"amplitude": 100.0, "freq_hz": 13.56e6}


def _project(regions=(), boundaries=None, coord="xy", size=0.5e-3, w=W, h=H, fluid=None, mode="cartesian",
             amr=None) -> Project:
    if boundaries is None:
        boundaries = [
            {"edges": [3], "type": "dirichlet", "voltage": 0.0, "voltage_rf": RF, "see_gamma": 0.05},
            {"edges": [1], "type": "dirichlet", "voltage": 0.0, "see_gamma": 0.05},
            {"edges": [0, 2], "type": "symmetry"},
        ]
    f = {"init_density_m3": 5e14, "init_te_ev": 3.0, "gas_pressure_pa": 30.0, "n_steps": 20,
         "frame_every": 100000}
    f.update(fluid or {})
    mesh = {"size": size, "mode": mode}
    if amr is not None:
        mesh["amr"] = amr
    return Project.model_validate({
        "coord": coord,
        "geometry": {"domain": {"polygon": [[0, 0], [w, 0], [w, h], [0, h]]},
                     "boundaries": boundaries, "regions": list(regions)},
        "mesh": mesh,
        "fluid2d": f,
    })


PIN_R = 0.0023
PIN = {"id": "pin", "type": "conductor", "voltage": 0.0, "see_gamma": 0.1,
       "shape": {"kind": "circle", "center": [0.0101, 0.0052], "radius": PIN_R}}
BLOCK = {"id": "blk", "type": "dielectric", "eps_r": 4.0, "see_gamma": 0.2,
         "polygon": [[0.0031, 0.0], [0.0057, 0.0], [0.0057, 0.0033], [0.0031, 0.0033]]}


# ---- 輸送グラフの幾何 ----------------------------------------------------------------


def test_graph_volumes_walls_and_edges_match_geometry():
    """格子に揃っていない円 (導体) と多角形 (誘電体): 気体体積・壁面積が解析値に一致する。"""
    sim = CartesianFluid2dSimulation(_project([PIN, BLOCK], size=0.25e-3), device="cpu")
    g = sim.graph
    exact = W * H - math.pi * PIN_R**2 - 0.0026 * 0.0033
    assert g.gas_volume == pytest.approx(exact, rel=2e-4)
    assert float(g.node_vol.sum()) == pytest.approx(g.gas_volume, rel=1e-12)   # 小片は全て併合された
    assert g.dropped_volume == 0.0
    # 電荷の写像は元の双対セルの気体体積 (総量 = 気体体積、2π なし)
    assert float(g.charge_map.sum()) == pytest.approx(g.gas_volume, rel=1e-12)

    pin = g.wall_group >= 0
    blk = g.wall_gamma == 0.2
    assert float(g.wall_area[pin].sum()) == pytest.approx(2.0 * math.pi * PIN_R, rel=1e-9)
    # 誘電体は底辺 (外周の symmetry 辺) に接しているので露出面は 3 辺
    assert float(g.wall_area[blk].sum()) == pytest.approx(2 * 0.0033 + 0.0026, rel=1e-9)
    side = ~pin & ~blk
    assert float(g.wall_area[side].sum()) == pytest.approx(2 * H, rel=1e-12)      # 左右の電極のみ (上下は symmetry)
    assert np.allclose(g.wall_gamma[side], 0.05)
    # 法線は単位ベクトルで、円の壁は中心を向く (気体 → 固体)
    assert np.allclose(np.hypot(g.wall_normal[:, 0], g.wall_normal[:, 1]), 1.0)
    assert np.all(g.edge_w > 0.0)

    # 固体の厳密な内部の節点は輸送節点にならない
    nodes = sim.mesh.nodes
    d = np.hypot(nodes[:, 0] - 0.0101, nodes[:, 1] - 0.0052)
    inside_pin = d < PIN_R - 1e-9
    assert not np.any(g.active & inside_pin)


@pytest.mark.parametrize("coord", ["rz", "rz_x0"])
def test_graph_axisymmetric_volume_and_walls(coord):
    """軸対称の円筒 (軸上に誘電体の円柱): 体積 π R² L・壁面積 (側面・端面・円柱の露出面) が厳密に合う。"""
    radius, length, rd, ld = 0.01, 0.02, 0.0031, 0.004
    if coord == "rz":   # r = y
        w, h = length, radius
        bnd = [{"edges": [0], "type": "symmetry"}] + [
            {"edges": [e], "type": "dirichlet", "voltage": 0.0} for e in (1, 2, 3)]
        poly = [[0.008, 0], [0.008 + ld, 0], [0.008 + ld, rd], [0.008, rd]]
    else:               # r = x
        w, h = radius, length
        bnd = [{"edges": [3], "type": "symmetry"}] + [
            {"edges": [e], "type": "dirichlet", "voltage": 0.0} for e in (0, 1, 2)]
        poly = [[0, 0.008], [rd, 0.008], [rd, 0.008 + ld], [0, 0.008 + ld]]
    reg = [{"id": "d", "type": "dielectric", "eps_r": 2.0, "polygon": poly}]
    sim = CartesianFluid2dSimulation(_project(reg, bnd, coord=coord, size=0.25e-3, w=w, h=h), device="cpu")
    g = sim.graph
    vol = math.pi * radius**2 * length - math.pi * rd**2 * ld
    assert g.gas_volume == pytest.approx(vol, rel=2e-4)
    assert float(g.node_vol.sum()) == pytest.approx(g.gas_volume, rel=1e-12)
    wall = 2 * math.pi * radius * length + 2 * math.pi * radius**2 + 2 * math.pi * rd * ld + 2 * math.pi * rd**2
    assert float(g.wall_area.sum()) == pytest.approx(wall, rel=1e-9)   # 対称軸 (r=0) は壁にならない


# ---- v1 と同じ検証 (tests/test_fluid2d.py) -----------------------------------------------


def test_matches_fluid1d_cross_section_average():
    """細長い矩形 (上下 symmetry) の断面平均が同条件の fluid1d と一致する (v1 テスト 1 と同じ条件)。"""
    gap, n_cells = 0.02, 20
    h = gap / n_cells
    rf = VoltageRF(amplitude=100.0, freq_hz=13.56e6, phase_deg=0.0)
    n_steps, avg = 6000, 1500
    s1 = Fluid1dSettings(
        gap_m=gap, n_cells=n_cells,
        left=Pic1dElectrode(v_dc=0.0, voltage_rf=rf, see_gamma=0.05),
        right=Pic1dElectrode(v_dc=0.0, see_gamma=0.05),
        init_density_m3=1e14, init_te_ev=2.0, gas_pressure_pa=20.0, gas_temperature_k=300.0,
        n_steps=n_steps, frame_every=n_steps, avg_steps=avg,
    )
    dummy = Geometry(domain=Domain(polygon=[(0, 0), (1, 0), (1, 1), (0, 1)]))
    sim1 = Fluid1dSimulation(Project(geometry=dummy, mesh=MeshSettings(size=0.1), fluid1d=s1))
    sim1.run_batch(store_frames=False)
    geo = Geometry(
        domain=Domain(polygon=[(0, 0), (gap, 0), (gap, h), (0, h)]),
        boundaries=[
            BoundaryCondition(edges=[3], type="dirichlet", voltage=0.0, voltage_rf=rf, see_gamma=0.05),
            BoundaryCondition(edges=[1], type="dirichlet", voltage=0.0, see_gamma=0.05),
            BoundaryCondition(edges=[0], type="symmetry"),
            BoundaryCondition(edges=[2], type="symmetry"),
        ],
    )
    s2 = Fluid2dSettings(init_density_m3=1e14, init_te_ev=2.0, gas_pressure_pa=20.0, gas_temperature_k=300.0,
                         n_steps=n_steps, frame_every=n_steps, avg_steps=avg)
    sim2 = CartesianFluid2dSimulation(
        Project(geometry=geo, mesh=MeshSettings(size=h, mode="cartesian"), fluid2d=s2), device="cpu")
    assert (sim2.grid.nx, sim2.grid.ny) == (20, 1)
    sim2.run_batch(store_frames=False)
    f1, f2 = sim1.fields, sim2.fields
    xs = sim2.mesh.nodes[:, 0]
    ux = np.unique(np.round(xs, 12))

    def cross(field):
        return np.array([field[np.isclose(xs, x)].mean() for x in ux])

    ne1, te1, ph1 = (np.interp(ux, sim1.xg, f1[k]) for k in ("n_e", "t_e", "phi"))
    ne2, te2, ph2 = cross(f2["n_e"]), cross(f2["t_e"]), cross(f2["phi"])
    assert np.max(np.abs(ne2 - ne1) / np.maximum(np.maximum(ne1, ne2), 1.0)) < 0.10
    assert np.max(np.abs(ph2 - ph1)) < 0.05 * (ph1.max() - ph1.min())
    bulk = ne1 > 0.2 * ne1.max()
    assert bulk.sum() >= 5
    assert np.max(np.abs(te2[bulk] - te1[bulk]) / te1[bulk]) < 0.10


def test_matches_v1_structured_with_grid_aligned_solids():
    """格子に揃った誘電体ブロック・導体の角柱: v1 構造格子 (同じ格子) と場・全量が一致する。

    違いは節点体積 (v1 は P1 集中質量で市松に 4/3·h² と 2/3·h² が並ぶ、v2 は双対セル h²) と
    壁の電場の取り方だけなので、1〜2% で一致する (差は離散化の違いの大きさ)。
    """
    regions = [
        {"id": "blk", "type": "dielectric", "eps_r": 4.0, "see_gamma": 0.1,
         "polygon": [[0.008, 0], [0.012, 0], [0.012, 0.003], [0.008, 0.003]]},
        {"id": "pin", "type": "conductor", "voltage": 0.0, "see_gamma": 0.05,
         "polygon": [[0.013, 0.006], [0.0145, 0.006], [0.0145, 0.0075], [0.013, 0.0075]]},
    ]
    fluid = {"n_steps": 1500, "avg_steps": 1000}
    v2 = CartesianFluid2dSimulation(_project(regions, fluid=fluid), device="cpu")
    v1 = Fluid2dSimulation(_project(regions, fluid=fluid, mode="structured"))
    v2.run_batch(store_frames=False)
    v1.run_batch(store_frames=False)
    for k in ("n_e_total", "n_i_total", "wall_e", "wall_i", "gen_total"):
        assert v2.history[k][-1] == pytest.approx(v1.history[k][-1], rel=0.03), k

    # v1 構造格子は導体内部の節点を持たないので座標で対応を取る
    def key(xy):
        return [(round(x / 1e-6), round(y / 1e-6)) for x, y in xy]

    pos = {k: i for i, k in enumerate(key(v2.mesh.nodes))}
    m = np.array([pos[k] for k in key(v1.mesh.nodes)])
    act1 = np.zeros(v2.n_nodes, dtype=bool)
    act1[m[v1.active_idx]] = True
    act2 = np.zeros(v2.n_nodes, dtype=bool)
    act2[v2.active_idx] = True
    assert np.array_equal(act1, act2)                  # 輸送節点の集合が同じ
    both = np.nonzero(act2)[0]
    for k, tol in (("n_e", 0.05), ("n_i", 0.05)):
        f1 = np.zeros(v2.n_nodes)
        f1[m] = v1.fields[k]
        a, b = v2.fields[k][both], f1[both]
        assert np.linalg.norm(a - b) / np.linalg.norm(b) < tol, k
    f1 = np.zeros(v2.n_nodes)
    f1[m] = v1.fields["phi"]
    assert np.max(np.abs(v2.fields["phi"][both] - f1[both])) < 2.0      # 振幅 100 V


def test_boltzmann_equilibrium_around_embedded_obstacle():
    """イオン固定 (2D 非一様)・反射壁・電子のみ緩和: 円形の誘電体を置いても n_e ∝ exp(φ/Te)。"""
    length, te0, n_i0 = 0.02, 2.0, 1.0e14
    bnd = [{"edges": [e], "type": "dirichlet", "voltage": 0.0} for e in range(4)]
    obst = {"id": "d", "type": "dielectric", "eps_r": 3.0,
            "shape": {"kind": "circle", "center": [0.0123, 0.0087], "radius": 0.0031}}
    sim = CartesianFluid2dSimulation(
        _project([obst], bnd, size=1e-3, w=length, h=length,
                 fluid={"init_density_m3": n_i0, "init_te_ev": te0, "gas_pressure_pa": 50.0, "n_steps": 1}),
        device="cpu")
    sim.debug_source_enabled = False
    sim.debug_energy_enabled = False
    sim.debug_ions_enabled = False
    sim.debug_reflective_walls = True
    x, y = sim.mesh.nodes[:, 0], sim.mesh.nodes[:, 1]
    act = sim.active_idx
    sim.n_i = np.zeros(sim.n_nodes)
    sim.n_i[act] = n_i0 * (1.0 + 0.5 * np.cos(math.pi * x[act] / length) * np.cos(math.pi * y[act] / length))
    sim.n_e = np.zeros(sim.n_nodes)
    sim.n_e[act] = n_i0
    sim.w = 1.5 * sim.n_e * te0
    total0 = float(np.sum(sim.n_e[act] * sim.node_vol))
    for _ in range(1500):
        sim.step()
    assert float(np.sum(sim.n_e[act] * sim.node_vol)) == pytest.approx(total0, rel=1e-6)  # 反射壁で保存
    ref = act[np.argmin(np.hypot(x[act] - 0.004, y[act] - 0.004))]
    dln = np.log(sim.n_e[act] / sim.n_e[ref])
    dphi = (sim.phi[act] - sim.phi[ref]) / te0
    assert np.max(np.abs(dln - dphi)) < 0.02


def test_particle_balance_with_embedded_solids():
    """(電離生成 − 壁損失) が全量変化と機械精度で一致する (固体の埋め込み境界・小セル併合込み)。"""
    bnd = [{"edges": [e], "type": "dirichlet", "voltage": 0.0, "see_gamma": 0.05} for e in range(4)]
    sim = CartesianFluid2dSimulation(
        _project([PIN, BLOCK], bnd, size=0.5e-3,
                 fluid={"init_density_m3": 1e15, "gas_pressure_pa": 50.0, "dt": 5e-11, "linear_solver": "direct"}),
        device="cpu")
    act = sim.active_idx
    n_e0 = float(np.sum(sim.n_e[act] * sim.node_vol))
    n_i0 = float(np.sum(sim.n_i[act] * sim.node_vol))
    for _ in range(200):
        sim.step()
    n_e1 = float(np.sum(sim.n_e[act] * sim.node_vol))
    n_i1 = float(np.sum(sim.n_i[act] * sim.node_vol))
    assert sim.wall["electron"] > 0.0 and sim.wall["ion"] > 0.0
    assert (n_e1 - n_e0) == pytest.approx(sim.gen_total - sim.wall["electron"], rel=1e-8, abs=1.0)
    assert (n_i1 - n_i0) == pytest.approx(sim.gen_total - sim.wall["ion"], rel=1e-8, abs=1.0)


def test_rz_axisymmetric_smoke_with_axis_dielectric():
    """軸対称 (rz): 軸上に誘電体、右端が RF 電極。有限・非負、対称軸上の節点も健全。"""
    rf = {"amplitude": 80.0, "freq_hz": 13.56e6}
    bnd = [{"edges": [0], "type": "symmetry"},
           {"edges": [1], "type": "dirichlet", "voltage": 0.0, "voltage_rf": rf, "see_gamma": 0.05},
           {"edges": [2], "type": "dirichlet", "voltage": 0.0, "see_gamma": 0.05},
           {"edges": [3], "type": "dirichlet", "voltage": 0.0, "see_gamma": 0.05}]
    reg = [{"id": "d", "type": "dielectric", "eps_r": 4.0,
            "polygon": [[0.004, 0.0], [0.0065, 0.0], [0.0065, 0.0027], [0.004, 0.0027]]}]
    sim = CartesianFluid2dSimulation(
        _project(reg, bnd, coord="rz", size=1e-3, w=0.02, h=0.01,
                 fluid={"init_density_m3": 1e14, "init_te_ev": 2.0, "n_steps": 1500, "avg_steps": 500}),
        device="cpu")
    sim.run_batch()
    for arr in (sim.n_e, sim.n_i, sim.w):
        assert np.all(np.isfinite(arr)) and np.all(arr >= 0.0)
    assert np.all(np.isfinite(sim.phi))
    axis = np.nonzero(np.isclose(sim.mesh.nodes[:, 1], 0.0))[0]
    axis_act = np.intersect1d(axis, sim.active_idx)
    assert axis_act.size > 0 and np.all(sim.n_e[axis_act] > 0.0)
    result = build_fluid2d_result(sim, elapsed_s=1.0)
    assert result["fields"] is not None and len(result["fields"]["phi"]) == sim.n_nodes


def test_explicit_matches_semi_implicit_at_small_dt():
    bnd = [{"edges": [e], "type": "dirichlet", "voltage": 0.0, "see_gamma": 0.05} for e in range(4)]
    proj = _project([PIN], bnd, size=0.5e-3, w=0.02, h=0.01,
                    fluid={"init_density_m3": 1e15, "gas_pressure_pa": 50.0, "dt": 1e-13})
    ex = CartesianFluid2dSimulation(proj, explicit=True, device="cpu")
    im = CartesianFluid2dSimulation(proj.model_copy(deep=True), explicit=False, device="cpu")
    for _ in range(50):
        ex.step()
        im.step()
    for k in ("n_e", "n_i", "w"):
        np.testing.assert_allclose(getattr(ex, k), getattr(im, k), rtol=1e-3, atol=1.0)
    np.testing.assert_allclose(ex.phi, im.phi, rtol=1e-3, atol=1e-5)


def test_stop_between_substeps_rolls_back_and_continues_identically():
    """サブステップの合間の停止要求 (v1 の step を継承): ステップ開始時の状態に戻り、続きは中断なしとビット一致。"""
    proj = _project([PIN, BLOCK], size=0.5e-3, fluid={"init_density_m3": 1e15, "gas_pressure_pa": 50.0, "dt": 1e-9})
    sim = CartesianFluid2dSimulation(proj, device="cpu")
    ref = CartesianFluid2dSimulation(proj.model_copy(deep=True), device="cpu")
    sim.step()
    ref.step()
    keys = ("n_e", "n_i", "w", "phi")
    before = {k: getattr(sim, k).copy() for k in keys}
    n_checks = 0

    def stop_at_third_check():
        nonlocal n_checks
        n_checks += 1
        return n_checks >= 3

    assert sim.step(stop_at_third_check) is None           # dt は安定条件の目安の数倍 → 6 回以上に刻む
    assert n_checks == 3 and sim.step_count == 1
    for k in keys:
        assert np.array_equal(getattr(sim, k), before[k]), k
    sim.step()
    ref.step()
    for k in keys:
        assert np.array_equal(getattr(sim, k), getattr(ref, k)), k
    assert sim.wall == ref.wall and sim.gen_total == ref.gen_total


@pytest.mark.parametrize("coord", ["xy", "rz"])
def test_joule_relaxation_time_matches_uniform_field_analytic(coord):
    """一様プラズマ・一様電場 E では τ_J = (3/2)Te/(μ_e E²) (双対セルの体積と面の重みが整合し、全節点で厳密)。"""
    v0 = 50.0
    bnd = [{"edges": [3], "type": "dirichlet", "voltage": 0.0}, {"edges": [1], "type": "dirichlet", "voltage": v0},
           {"edges": [0, 2], "type": "symmetry"}]
    sim = CartesianFluid2dSimulation(
        _project([], bnd, coord=coord, size=1e-3, fluid={"init_density_m3": 1e15, "gas_pressure_pa": 50.0}),
        device="cpu")
    sim.phi = sim._solve_phi(0.0)                          # 空間電荷 0 → φ は電極間で線形
    te0, mu_e0, *_ = sim._te_and_coeffs(sim.n_e[sim.active_idx], sim.w[sim.active_idx])
    expected = 1.5 * float(te0[0]) / (float(mu_e0[0]) * (v0 / W) ** 2)
    assert sim._joule_relaxation_time(te0, mu_e0) == pytest.approx(expected, rel=1e-9)


def test_rejects_periodic_boundary():
    bnd = [{"edges": [0, 2], "type": "periodic"},
           {"edges": [3], "type": "dirichlet", "voltage": 0.0}, {"edges": [1], "type": "dirichlet", "voltage": 0.0}]
    with pytest.raises(ValueError, match="periodic"):
        CartesianFluid2dSimulation(_project([], bnd), device="cpu")


# ---- Poisson・壁の電場 -----------------------------------------------------------------


def test_poisson_direct_matches_gmg(monkeypatch):
    """LU (既定、小さい格子) と GMG-PCG が同じ φ を返す。Dirichlet の無い特異な問題も GMG で解ける。"""
    proj = _project([PIN, BLOCK], size=0.5e-3)
    lu = CartesianFluid2dSimulation(proj, device="cpu")
    assert lu._lu is not None
    monkeypatch.setattr(gsim, "DIRECT_MAX", 0)
    mg = CartesianFluid2dSimulation(proj.model_copy(deep=True), device="cpu")
    assert mg._lu is None and mg._gmg is not None
    x, y = lu.mesh.nodes[:, 0], lu.mesh.nodes[:, 1]
    for sim in (lu, mg):
        sim.n_i[sim.active_idx] *= 1.0 + 0.3 * np.sin(300 * x[sim.active_idx]) * np.cos(500 * y[sim.active_idx])
    t = 1.3e-8
    p1, p2 = lu._solve_phi(t), mg._solve_phi(t)
    assert np.max(np.abs(p1 - p2)) < 1e-6 * max(1.0, float(np.max(np.abs(p1))))

    neumann = _project([], [])                   # 全周 Neumann (特異) → GMG
    sim = CartesianFluid2dSimulation(neumann, device="cpu")
    assert sim._lu is None
    sim.run_batch(store_frames=False)
    assert np.all(np.isfinite(sim.phi))


def test_wall_field_is_one_sided_difference_at_electrodes():
    """外周の電極では E·n = (φ(隣の節点) − φ(壁)) / h (v1 の「壁に接する要素の E」と同じ量)。"""
    sim = CartesianFluid2dSimulation(_project([PIN]), device="cpu")
    sim.n_i[sim.active_idx] *= 1.5            # 正の空間電荷でシース状の電位を作る
    phi = sim._solve_phi(0.0)
    en = sim._wall_en(phi)
    g, grid = sim.graph, sim.grid
    nx1 = grid.nx + 1
    left = (g.wall_node % nx1 == 0) & (g.wall_group < 0)
    node = g.wall_node[left]
    np.testing.assert_allclose(en[left], (phi[node + 1] - phi[node]) / grid.dx, rtol=1e-12, atol=1e-9)
    assert np.all(en[left] > 0.0)                # 電場は壁 (低電位) を向く
    pin = g.wall_group >= 0
    assert np.all(en[pin] > 0.0)                 # 接地ピンへ向かう電場


# ---- 設定・配線 ------------------------------------------------------------------------


def test_amr_setting_runs_on_base_grid_with_warning():
    amr = {"max_level": 1, "regions": [{"p1": [0.008, 0.003], "p2": [0.012, 0.007], "level": 1}]}
    sim = make_fluid2d_simulation(_project([], amr=amr))
    assert isinstance(sim, CartesianFluid2dSimulation)
    assert sim.n_nodes == (sim.grid.nx + 1) * (sim.grid.ny + 1)
    assert any("mesh.amr" in w for w in sim.warnings)


def test_make_fluid2d_simulation_routes_by_mesh_mode():
    assert type(make_fluid2d_simulation(_project([], mode="structured"))) is Fluid2dSimulation
    assert type(make_fluid2d_simulation(_project([]))) is CartesianFluid2dSimulation


def test_ws_fluid2d_cartesian_sends_display_mesh():
    server._last_simfluid2d = None
    client = TestClient(server.app)
    proj = _project([PIN], size=1e-3, fluid={"n_steps": 20, "frame_every": 10, "avg_steps": 10})
    with client.websocket_connect("/ws/fluid2d") as ws:
        ws.send_text(json.dumps({"cmd": "start", "project": proj.model_dump(mode="json")}))
        started = ws.receive_json()
        assert started["type"] == "started", started
        mesh = started["mesh"]
        n = len(mesh["nodes"])
        assert len(mesh["region_of_triangle"]) == len(mesh["triangles"]) > 0
        frames = 0
        while True:
            msg = ws.receive_json()
            assert msg["type"] != "error", msg.get("detail")
            if msg["type"] == "frame":
                frames += 1
                assert len(msg["phi"]) == len(msg["n_e"]) == n
            elif msg["type"] == "done":
                break
        assert frames >= 1
        res = msg["result"]
        assert len(res["history"]["t"]) == 20 and len(res["fields"]["n_e"]) == n
        assert len(res["fields"]["e_abs"]) == len(mesh["triangles"])
    # /mesh (AMR なし) の表示用メッシュと同じ節点・要素
    m = client.post("/mesh", json=proj.model_dump(mode="json")).json()
    assert m["nodes"] == mesh["nodes"] and m["triangles"] == mesh["triangles"]
    server._last_simfluid2d = None


def test_batch_fluid2d_cartesian_bundle_has_mesh(tmp_path: Path):
    proj = _project([], size=1e-3, fluid={"n_steps": 10, "frame_every": 10, "avg_steps": 5})
    case = tmp_path / "case_f2c.json"
    case.write_text(json.dumps(proj.model_dump(mode="json")), encoding="utf-8")
    assert run_files([str(case)], parallel=1, out_dir=str(tmp_path)) == 0
    obj = json.loads((tmp_path / "case_f2c_results.json").read_text(encoding="utf-8"))
    res = obj["results"]
    assert len(res["mesh"]["nodes"]) == len(res["fluid2d"]["fields"]["phi"])
    assert "region_of_triangle" in res["mesh"]
