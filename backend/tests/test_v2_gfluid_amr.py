"""v2 流体 2D の AMR 版 (es_sim.gfluid.amr・amr_graph) のテスト (prompts/128)。

合成格子 (ぶら下がり節点を含む Delaunay 適合三角形分割の Voronoi 有限体積 + 埋め込み境界) の輸送グラフの
幾何 (気体体積・壁面積・線形場の整合性、細分化域での一様格子との一致)、合成格子の Poisson の精度、
粒子数の保存・Boltzmann 平衡、細かい一様格子との突き合わせ、振り分け・server / batch の配線を確かめる。
GPU は使わない (GPU 版との一致は tests/test_v2_gfluid_gpu.py)。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.interpolate import LinearNDInterpolator

import es_sim.server as server
from es_sim.batch import run_files
from es_sim.fluid2d import build_fluid2d_result
from es_sim.gfluid import CartesianFluid2dSimulation, make_fluid2d_simulation
from es_sim.gfluid.amr import AmrFluid2dSimulation
from es_sim.schema import Project

W, H = 0.02, 0.01
RF = {"amplitude": 100.0, "freq_hz": 13.56e6}
AMR1 = {"max_level": 1, "buffer_cells": 2}

PIN_R = 0.0023
PIN = {"id": "pin", "type": "conductor", "voltage": 0.0, "see_gamma": 0.1,
       "shape": {"kind": "circle", "center": [0.0101, 0.0052], "radius": PIN_R}}
BLOCK = {"id": "blk", "type": "dielectric", "eps_r": 4.0, "see_gamma": 0.2,
         "polygon": [[0.0031, 0.0], [0.0057, 0.0], [0.0057, 0.0033], [0.0031, 0.0033]]}
# 格子に揃った誘電体 (外周の symmetry 辺に接する) と小さい円の導体
ABLK = {"id": "blk", "type": "dielectric", "eps_r": 4.0, "see_gamma": 0.1,
        "polygon": [[0.008, 0.0], [0.012, 0.0], [0.012, 0.003], [0.008, 0.003]]}
APIN = {"id": "pin", "type": "conductor", "voltage": 0.0, "see_gamma": 0.05,
        "shape": {"kind": "circle", "center": [0.0137, 0.0068], "radius": 0.0011}}


def _project(regions=(), boundaries=None, coord="xy", size=0.5e-3, w=W, h=H, fluid=None, amr=AMR1) -> Project:
    if boundaries is None:
        boundaries = [
            {"edges": [3], "type": "dirichlet", "voltage": 0.0, "voltage_rf": RF, "see_gamma": 0.05},
            {"edges": [1], "type": "dirichlet", "voltage": 0.0, "see_gamma": 0.05},
            {"edges": [0, 2], "type": "symmetry"},
        ]
    f = {"init_density_m3": 5e14, "init_te_ev": 3.0, "gas_pressure_pa": 30.0, "n_steps": 20,
         "frame_every": 100000}
    f.update(fluid or {})
    mesh = {"size": size, "mode": "cartesian"}
    if amr is not None:
        mesh["amr"] = amr
    return Project.model_validate({
        "coord": coord,
        "geometry": {"domain": {"polygon": [[0, 0], [w, 0], [w, h], [0, h]]},
                     "boundaries": boundaries, "regions": list(regions)},
        "mesh": mesh,
        "fluid2d": f,
    })


def _divergence_of_linear_field(g, xy: np.ndarray) -> np.ndarray:
    """節点ごとの Σ_j w_ij (x_j − x_i) (閉じた Voronoi 領域なら 0)。"""
    out = np.zeros((xy.shape[0], 2))
    d = xy[g.edge_q] - xy[g.edge_p]
    np.add.at(out, g.edge_p, g.edge_w[:, None] * d)
    np.add.at(out, g.edge_q, -g.edge_w[:, None] * d)
    return out


# ---- 輸送グラフの幾何 ----------------------------------------------------------------


@pytest.mark.parametrize("amr", [AMR1, {"max_level": 2, "buffer_cells": 1}])
def test_graph_volumes_walls_and_consistency(amr):
    """格子に揃っていない円 (導体) と多角形 (誘電体) を細分化: 気体体積・壁面積が解析値に一致し、
    ぶら下がり節点を含む Voronoi 領域が閉じている (線形場の発散が 0)。"""
    sim = AmrFluid2dSimulation(_project([PIN, BLOCK], amr=amr), device="cpu")
    g = sim.graph
    lay = sim._lay
    assert sim.hier.max_level == amr["max_level"]
    assert sim.n_nodes == lay.n_nodes > (sim.grid.nx + 1) * (sim.grid.ny + 1)
    exact = W * H - math.pi * PIN_R**2 - 0.0026 * 0.0033
    assert g.gas_volume == pytest.approx(exact, rel=2e-4)
    assert float(g.node_vol.sum()) == pytest.approx(g.gas_volume, rel=1e-12)
    assert g.dropped_volume == 0.0
    assert float(g.charge_map.sum()) == pytest.approx(g.gas_volume, rel=1e-12)

    pin = g.wall_group >= 0
    blk = g.wall_gamma == 0.2
    side = ~pin & ~blk
    assert float(g.wall_area[pin].sum()) == pytest.approx(2.0 * math.pi * PIN_R, rel=1e-9)
    assert float(g.wall_area[blk].sum()) == pytest.approx(2 * 0.0033 + 0.0026, rel=1e-9)
    assert float(g.wall_area[side].sum()) == pytest.approx(2 * H, rel=1e-12)
    assert np.allclose(np.hypot(g.wall_normal[:, 0], g.wall_normal[:, 1]), 1.0)
    assert np.all(g.edge_w > 0.0)                         # M 行列 (鈍角の三角形なし・斜辺の 0 は除いた)
    assert not any("鈍角" in w for w in sim.warnings)
    assert np.all(g.active[g.wall_node])

    # 固体・外周から離れた節点 (遷移セルのぶら下がり節点を含む) で Σ w_ij (x_j − x_i) = 0
    xy = sim.mesh.nodes
    div = _divergence_of_linear_field(g, xy)
    near = np.abs(np.hypot(xy[:, 0] - 0.0101, xy[:, 1] - 0.0052) - PIN_R) < 1.5e-3
    near |= (xy[:, 0] > 0.0016) & (xy[:, 0] < 0.0072) & (xy[:, 1] < 0.0048)
    near |= (xy[:, 0] < 1e-9) | (xy[:, 0] > W - 1e-9) | (xy[:, 1] < 1e-9) | (xy[:, 1] > H - 1e-9)
    inner = g.active & ~near
    u = lay.op.u_of_node
    hanging = np.zeros(sim.n_nodes, dtype=bool)
    hanging[u >= 0] = lay.op.hanging[u[u >= 0]]
    assert np.sum(inner & hanging) > 10                   # 遷移セルのぶら下がり節点も含む
    scale = float(np.max(g.edge_w * g.edge_len))
    assert np.max(np.abs(div[inner])) < 1e-10 * scale


def _fine_zone(g, h: float) -> np.ndarray:
    """輸送の辺がすべて長さ h の節点 (まわりの葉セルがすべて最も細かいレベル)。"""
    n = g.active.size
    short = np.abs(g.edge_len - h) < 1e-9 * h
    deg = np.bincount(g.edge_p, minlength=n) + np.bincount(g.edge_q, minlength=n)
    deg_h = (np.bincount(g.edge_p[short], minlength=n) + np.bincount(g.edge_q[short], minlength=n))
    return (deg > 0) & (deg == deg_h)


def test_refined_zone_matches_uniform_fine_grid():
    """格子に揃った固体のまわりの細分化域では、節点体積・辺の重み・壁が同じ幅の一様格子版と一致する
    (固体の表面から出る辺の開口も全開: 端点が表面上にある標本線の丸め誤差を固体と数えない)。"""
    h = 0.25e-3
    sq = {"id": "pin", "type": "conductor", "voltage": 0.0, "see_gamma": 0.05,
          "polygon": [[0.0135, 0.006], [0.015, 0.006], [0.015, 0.0075], [0.0135, 0.0075]]}
    amr = AmrFluid2dSimulation(_project([ABLK, sq]), device="cpu")
    fine = CartesianFluid2dSimulation(_project([ABLK, sq], size=h, amr=None), device="cpu")
    ga, gf = amr.graph, fine.graph

    def key(xy):
        return np.rint(xy / (0.5 * h)).astype(np.int64) @ np.array([1, 1 << 20])

    pos = {k: i for i, k in enumerate(key(fine.mesh.nodes))}
    a2f = np.array([pos[k] for k in key(amr.mesh.nodes)])
    assert np.array_equal(ga.active, gf.active[a2f])
    zone = _fine_zone(ga, h) & ga.active
    assert zone.sum() > 400
    np.testing.assert_allclose(ga.node_vol[zone], gf.node_vol[a2f[zone]], rtol=1e-9)

    wf = {}
    for p, q, w in zip(gf.edge_p, gf.edge_q, gf.edge_w):
        wf[(min(p, q), max(p, q))] = w
    n_cmp = 0
    for p, q, w in zip(ga.edge_p, ga.edge_q, ga.edge_w):
        if zone[p] and zone[q]:
            fp, fq = a2f[p], a2f[q]
            assert w == pytest.approx(wf[(min(fp, fq), max(fp, fq))], rel=1e-9), (amr.mesh.nodes[p], amr.mesh.nodes[q])
            n_cmp += 1
    assert n_cmp > 800

    wa = np.bincount(ga.wall_node, weights=ga.wall_area, minlength=amr.n_nodes)
    wfn = np.bincount(gf.wall_node, weights=gf.wall_area, minlength=fine.n_nodes)[a2f]
    solid_nodes = zone & (np.abs(amr.mesh.nodes[:, 0]) > 1e-9) & (np.abs(amr.mesh.nodes[:, 0] - W) > 1e-9)
    np.testing.assert_allclose(wa[solid_nodes], wfn[solid_nodes], rtol=1e-9, atol=1e-15)
    assert float(wa[solid_nodes].sum()) > 0.9 * (2 * 0.003 + 0.004 + 4 * 0.0015)


@pytest.mark.parametrize("coord", ["rz", "rz_x0"])
def test_graph_axisymmetric_volume_and_walls(coord):
    """軸対称の円筒 (軸上に誘電体の円柱) を細分化: 体積 π R² L・壁面積が厳密に合う。"""
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
    sim = AmrFluid2dSimulation(_project(reg, bnd, coord=coord, w=w, h=h), device="cpu")
    g = sim.graph
    vol = math.pi * radius**2 * length - math.pi * rd**2 * ld
    assert g.gas_volume == pytest.approx(vol, rel=2e-4)
    assert float(g.node_vol.sum()) == pytest.approx(g.gas_volume, rel=1e-12)
    wall = 2 * math.pi * radius * length + 2 * math.pi * radius**2 + 2 * math.pi * rd * ld + 2 * math.pi * rd**2
    assert float(g.wall_area.sum()) == pytest.approx(wall, rel=1e-9)


# ---- Poisson (合成格子) ----------------------------------------------------------------


def test_poisson_matches_fine_uniform_grid():
    """同じ空間電荷の Poisson: 合成格子の解が細かい一様格子の解に、粗い一様格子より近い。"""
    projs = {"fine": _project([ABLK, APIN], size=0.25e-3, amr=None), "amr": _project([ABLK, APIN]),
             "coarse": _project([ABLK, APIN], amr=None)}
    sims = {k: (AmrFluid2dSimulation if k == "amr" else CartesianFluid2dSimulation)(p, device="cpu")
            for k, p in projs.items()}
    phis = {}
    for k, s in sims.items():
        x, y = s.mesh.nodes[:, 0], s.mesh.nodes[:, 1]
        ne = np.zeros(s.n_nodes)
        ne[s.active_idx] = 1e14 * (1.0 + 0.5 * np.sin(math.pi * x[s.active_idx] / W) * np.sin(math.pi * y[s.active_idx] / H))
        s.n_e = ne
        s.n_i = 1.5 * ne
        phis[k] = s._solve_phi(2e-8)
    ref = phis["fine"]
    xy = sims["fine"].mesh.nodes
    err = {}
    for k in ("amr", "coarse"):
        f = LinearNDInterpolator(sims[k].mesh.nodes, phis[k])(xy)
        err[k] = float(np.nanmax(np.abs(f - ref)))
    assert err["amr"] < 2e-3 * float(np.max(np.abs(ref)))
    assert err["amr"] < err["coarse"]


# ---- 保存則・平衡 ------------------------------------------------------------------------


def test_particle_balance_with_hanging_nodes():
    """(電離生成 − 壁損失) が全量変化と機械精度で一致する (ぶら下がり節点・固体の併合込み)。"""
    bnd = [{"edges": [e], "type": "dirichlet", "voltage": 0.0, "see_gamma": 0.05} for e in range(4)]
    sim = AmrFluid2dSimulation(
        _project([PIN, BLOCK], bnd,
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


def test_boltzmann_equilibrium_across_refinement_levels():
    """イオン固定 (2D 非一様)・反射壁・電子のみ緩和: 細分化の境目をまたいでも n_e ∝ exp(φ/Te)。"""
    length, te0, n_i0 = 0.02, 2.0, 1.0e14
    bnd = [{"edges": [e], "type": "dirichlet", "voltage": 0.0} for e in range(4)]
    obst = {"id": "d", "type": "dielectric", "eps_r": 3.0,
            "shape": {"kind": "circle", "center": [0.0123, 0.0087], "radius": 0.0031}}
    sim = AmrFluid2dSimulation(
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
    assert float(np.sum(sim.n_e[act] * sim.node_vol)) == pytest.approx(total0, rel=1e-6)
    ref = act[np.argmin(np.hypot(x[act] - 0.004, y[act] - 0.004))]
    dln = np.log(sim.n_e[act] / sim.n_e[ref])
    dphi = (sim.phi[act] - sim.phi[ref]) / te0
    assert np.max(np.abs(dln - dphi)) < 0.02


def test_explicit_matches_semi_implicit_at_small_dt():
    bnd = [{"edges": [e], "type": "dirichlet", "voltage": 0.0, "see_gamma": 0.05} for e in range(4)]
    proj = _project([PIN], bnd, fluid={"init_density_m3": 1e15, "gas_pressure_pa": 50.0, "dt": 1e-13})
    ex = AmrFluid2dSimulation(proj, explicit=True, device="cpu")
    im = AmrFluid2dSimulation(proj.model_copy(deep=True), explicit=False, device="cpu")
    for _ in range(50):
        ex.step()
        im.step()
    for k in ("n_e", "n_i", "w"):
        np.testing.assert_allclose(getattr(ex, k), getattr(im, k), rtol=1e-3, atol=1.0)
    np.testing.assert_allclose(ex.phi, im.phi, rtol=1e-3, atol=1e-5)


# ---- 細かい一様格子との突き合わせ ---------------------------------------------------------


def test_ccp_with_solids_matches_fine_uniform_grid():
    """固体入り CCP: 固体のまわりだけ細かくした AMR が、全体を細かくした一様格子に粗い一様格子より近い。"""
    fluid = {"n_steps": 800, "avg_steps": 400}
    runs = {"amr": AmrFluid2dSimulation(_project([APIN, ABLK], fluid=fluid), device="cpu"),
            "fine": CartesianFluid2dSimulation(_project([APIN, ABLK], size=0.25e-3, fluid=fluid, amr=None), device="cpu"),
            "coarse": CartesianFluid2dSimulation(_project([APIN, ABLK], fluid=fluid, amr=None), device="cpu")}
    for s in runs.values():
        s.run_batch(store_frames=False)
    amr, fine, coarse = runs["amr"], runs["fine"], runs["coarse"]
    assert amr.warnings == []

    def err(s, k):
        return abs(s.history[k][-1] / fine.history[k][-1] - 1.0)

    for k in ("n_e_total", "n_i_total", "wall_e", "gen_total"):
        assert err(amr, k) < 0.01, k
    assert err(amr, "wall_i") < 0.03
    for k in ("n_e_total", "wall_i", "gen_total"):
        assert err(amr, k) < err(coarse, k), k

    # 時間平均の場 (AMR の節点は細かい格子の節点に乗る)
    key = lambda xy: [(round(x / 1e-6), round(y / 1e-6)) for x, y in xy]   # noqa: E731
    pos = {k: i for i, k in enumerate(key(fine.mesh.nodes))}
    m = np.array([pos[k] for k in key(amr.mesh.nodes)])
    act = amr.active_idx
    for k, tol in (("n_e", 0.03), ("n_i", 0.03), ("t_e", 0.03)):
        a, b = amr.fields[k][act], fine.fields[k][m[act]]
        assert np.linalg.norm(a - b) / np.linalg.norm(b) < tol, k


def test_rz_axisymmetric_smoke_with_axis_dielectric():
    rf = {"amplitude": 80.0, "freq_hz": 13.56e6}
    bnd = [{"edges": [0], "type": "symmetry"},
           {"edges": [1], "type": "dirichlet", "voltage": 0.0, "voltage_rf": rf, "see_gamma": 0.05},
           {"edges": [2], "type": "dirichlet", "voltage": 0.0, "see_gamma": 0.05},
           {"edges": [3], "type": "dirichlet", "voltage": 0.0, "see_gamma": 0.05}]
    reg = [{"id": "d", "type": "dielectric", "eps_r": 4.0,
            "polygon": [[0.004, 0.0], [0.0065, 0.0], [0.0065, 0.0027], [0.004, 0.0027]]}]
    sim = AmrFluid2dSimulation(
        _project(reg, bnd, coord="rz", size=1e-3,
                 fluid={"init_density_m3": 1e14, "init_te_ev": 2.0, "n_steps": 600, "avg_steps": 200}),
        device="cpu")
    sim.run_batch()
    for arr in (sim.n_e, sim.n_i, sim.w):
        assert np.all(np.isfinite(arr)) and np.all(arr >= 0.0)
    assert np.all(np.isfinite(sim.phi))
    axis = np.nonzero(np.isclose(sim.mesh.nodes[:, 1], 0.0))[0]
    axis_act = np.intersect1d(axis, sim.active_idx)
    assert axis_act.size > 0 and np.all(sim.n_e[axis_act] > 0.0)
    result = build_fluid2d_result(sim, elapsed_s=1.0)
    assert len(result["fields"]["phi"]) == sim.n_nodes
    assert len(result["fields"]["e_abs"]) == len(sim.mesh.triangles)


# ---- 設定・配線 ------------------------------------------------------------------------


def test_factory_routes_amr_settings(monkeypatch):
    monkeypatch.setenv("ES_SIM_DEVICE", "cpu")
    regions = {"max_level": 1, "regions": [{"p1": [0.008, 0.003], "p2": [0.012, 0.007], "level": 1}]}
    sim = make_fluid2d_simulation(_project([], amr=regions))
    assert type(sim) is AmrFluid2dSimulation
    assert sim.n_nodes > (sim.grid.nx + 1) * (sim.grid.ny + 1)
    assert not any("mesh.amr" in w for w in sim.warnings)
    assert type(make_fluid2d_simulation(_project([PIN]), explicit=True)) is AmrFluid2dSimulation
    # 細分化が起きない設定 (固体も矩形も無い・max_level 0) は一様格子版
    assert type(make_fluid2d_simulation(_project([]))) is CartesianFluid2dSimulation
    assert type(make_fluid2d_simulation(_project([PIN], amr={"max_level": 0}))) is CartesianFluid2dSimulation
    adaptive = make_fluid2d_simulation(_project([PIN], amr={"max_level": 1, "adaptive": True}))
    assert any("adaptive" in w for w in adaptive.warnings)


def test_ws_fluid2d_amr_sends_conforming_mesh(monkeypatch):
    monkeypatch.setenv("ES_SIM_DEVICE", "cpu")
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
        assert len(msg["result"]["fields"]["e_abs"]) == len(mesh["triangles"])
    sim = server._last_simfluid2d
    assert isinstance(sim, AmrFluid2dSimulation) and n == sim.n_nodes
    # 表示用メッシュ: 全節点を使う適合三角形分割 (導体の内部の葉セルは除く)
    tri = np.asarray(mesh["triangles"])
    xy = np.asarray(mesh["nodes"])
    area = 0.5 * np.abs((xy[tri[:, 1], 0] - xy[tri[:, 0], 0]) * (xy[tri[:, 2], 1] - xy[tri[:, 0], 1])
                        - (xy[tri[:, 1], 1] - xy[tri[:, 0], 1]) * (xy[tri[:, 2], 0] - xy[tri[:, 0], 0]))
    assert float(area.sum()) == pytest.approx(W * H - math.pi * PIN_R**2, rel=0.02)
    server._last_simfluid2d = None


def test_batch_fluid2d_amr_bundle_has_mesh(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ES_SIM_DEVICE", "cpu")
    proj = _project([PIN], size=1e-3, fluid={"n_steps": 10, "frame_every": 10, "avg_steps": 5})
    case = tmp_path / "case_f2a.json"
    case.write_text(json.dumps(proj.model_dump(mode="json")), encoding="utf-8")
    assert run_files([str(case)], parallel=1, out_dir=str(tmp_path)) == 0
    obj = json.loads((tmp_path / "case_f2a_results.json").read_text(encoding="utf-8"))
    res = obj["results"]
    assert len(res["mesh"]["nodes"]) == len(res["fluid2d"]["fields"]["phi"])
    assert len(res["mesh"]["triangles"]) == len(res["fluid2d"]["fields"]["e_abs"])
