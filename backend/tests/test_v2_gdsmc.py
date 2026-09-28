"""v2 GPU DSMC (直交格子 + 埋め込み固体、prompts/124) の検証。v1 (tests/test_dsmc.py) と同じ物理条件:

1. 平衡箱: n・T・p を保持し u ≈ 0
2. 自由分子流の流出: 密度 = リザーバの 1/2、u = c̄/2、流入 = 流出
3. 圧力駆動チャネル: 質量収支と単調な圧力勾配 (v1 と一致)
4. 流量指定 (sccm) の流入口: 流入レートが換算値に一致し流出と釣り合う
5. 線分指定の流入口: 上辺の一部だけから流入
6. 軸対称: 密閉円筒の平衡 (径方向に一様)・無衝突円筒の流出
7. 固体 (導体の円・誘電体の多角形) 越しの流れ: 固体内に分子が入らず、質量収支が成り立つ
8. 高温壁との熱交換: 温度が壁温の間に単調に分布
9. 平滑化: 総量保存・正値
10. /dsmc・/ws/dsmc が mesh.mode="cartesian" で GPU 版を使う
GPU (CUDA) が無い環境ではスキップ。
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from es_sim.device import cuda_available
from es_sim.dsmc import AMU, KB, SCCM_TO_PER_S
from es_sim.schema import Project

pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA (CuPy) が使えない環境")

L = 0.02
H = 0.01
M_AR = 39.948 * AMU


def _project(dsmc: dict, mesh: float = 1.5e-3, regions=(), coord: str = "xy", domain=None) -> Project:
    return Project.model_validate({
        "coord": coord,
        "geometry": {"domain": {"polygon": domain or [[0, 0], [L, 0], [L, H], [0, H]]},
                     "regions": list(regions), "boundaries": []},
        "mesh": {"size": mesh, "mode": "cartesian"},
        "dsmc": dsmc,
    })


def _sim(project: Project):
    from es_sim.gdsmc import GpuDsmcSimulation

    return GpuDsmcSimulation(project)


def _wmean(res, sim, arr):
    a = sim.area
    return float(np.sum(arr * res.n * a) / np.sum(res.n * a))


def test_equilibrium_box():
    p0, t0 = 10.0, 300.0
    sim = _sim(_project({"init_pressure_pa": p0, "init_temperature_k": t0, "wall_temperature_k": t0,
                         "n_particles": 30000, "n_steps": 600, "avg_steps": 300, "seed": 1}))
    res = sim.run()
    a = sim.area
    n0 = p0 / (KB * t0)
    assert float(np.sum(res.n * a) / a.sum()) == pytest.approx(n0, rel=0.03)
    assert _wmean(res, sim, res.t) == pytest.approx(t0, rel=0.03)
    assert float(np.sum(res.p * a) / a.sum()) == pytest.approx(p0, rel=0.05)
    u = np.hypot(_wmean(res, sim, res.u[:, 0]), _wmean(res, sim, res.u[:, 1]))
    assert u < 0.02 * math.sqrt(2.0 * KB * t0 / M_AR) + 5.0
    assert res.n_particles == 30000                                    # 密閉箱: 分子は消えない
    assert res.elapsed_s > 0 and res.timing["move"] > 0 and res.timing["collide"] > 0


def test_free_molecular_effusion():
    p0, t0 = 1.0, 300.0
    sim = _sim(_project({
        "gas": {"d_ref_m": 1e-15},
        "boundaries": [{"edges": [3], "type": "inlet", "pressure_pa": p0, "temperature_k": t0},
                       {"edges": [1], "type": "outlet"}, {"edges": [0], "type": "symmetry"},
                       {"edges": [2], "type": "symmetry"}],
        "init_pressure_pa": p0 / 2.0, "init_temperature_k": t0, "n_particles": 40000, "n_steps": 1500,
        "avg_steps": 500, "seed": 2}))
    res = sim.run()
    a = sim.area
    assert float(np.sum(res.n * a) / a.sum()) == pytest.approx(0.5 * p0 / (KB * t0), rel=0.05)
    c_bar = math.sqrt(8.0 * KB * t0 / (math.pi * M_AR))
    assert _wmean(res, sim, res.u[:, 0]) == pytest.approx(0.5 * c_bar, rel=0.05)
    assert res.outflow == pytest.approx(res.inflow, rel=0.05)


def test_pressure_driven_channel_matches_v1():
    from es_sim.dsmc import DsmcSimulation

    t0 = 300.0
    cfg = {"boundaries": [{"edges": [3], "type": "inlet", "pressure_pa": 20.0, "temperature_k": t0},
                          {"edges": [1], "type": "outlet", "pressure_pa": 5.0, "temperature_k": t0}],
           "init_pressure_pa": 12.0, "init_temperature_k": t0, "wall_temperature_k": t0, "n_particles": 40000,
           "n_steps": 2000, "avg_steps": 600, "seed": 3}
    slabs = {}
    for mode in ("cartesian", "structured"):
        p = _project(cfg)
        p.mesh.mode = mode
        sim = _sim(p) if mode == "cartesian" else DsmcSimulation(p)
        res = sim.run()
        assert res.outflow == pytest.approx(res.inflow, rel=0.10)
        cen = sim.mesh.nodes[sim.tris].mean(axis=1)
        a = sim.area
        s = []
        for i in range(4):
            sel = (cen[:, 0] >= i * L / 4) & (cen[:, 0] < (i + 1) * L / 4)
            s.append(float(np.sum(res.p[sel] * a[sel]) / a[sel].sum()))
        slabs[mode] = np.array(s)
        assert all(s[i] > s[i + 1] for i in range(3)) and 5.0 < s[-1] < s[0] < 20.0
        assert _wmean(res, sim, res.u[:, 0]) > 0.0
    assert np.max(np.abs(slabs["cartesian"] / slabs["structured"] - 1.0)) < 0.04


def test_sccm_flow_inlet_mass_balance():
    sccm = 10.0
    sim = _sim(_project({
        "gas": {"d_ref_m": 1e-15},
        "boundaries": [{"edges": [3], "type": "inlet", "flow_sccm": sccm}, {"edges": [1], "type": "outlet"},
                       {"edges": [0], "type": "symmetry"}, {"edges": [2], "type": "symmetry"}],
        "init_pressure_pa": 0.01, "init_temperature_k": 300.0, "n_particles": 30000, "n_steps": 1500,
        "avg_steps": 500, "seed": 7}))
    res = sim.run()
    assert res.inflow / (500 * sim.dt) == pytest.approx(sccm * SCCM_TO_PER_S, rel=0.02)
    assert res.outflow == pytest.approx(res.inflow, rel=0.10)
    assert _wmean(res, sim, res.u[:, 0]) > 0.0


def test_segment_inlet_only_injects_through_the_segment():
    sim = _sim(_project({
        "boundaries": [{"type": "inlet", "p1": [0.0, H], "p2": [L / 4, H], "pressure_pa": 10.0}],
        "init_pressure_pa": 5.0, "n_particles": 5000, "n_steps": 10, "avg_steps": 5, "seed": 6}))
    assert len(sim._res_edges) == 1
    ed = sim._res_edges[0]
    assert ed["a"] == (0.0, H) and ed["b"] == (L / 4, H) and ed["nrm"] == (0.0, -1.0)
    assert ed["rate"] == pytest.approx(10.0 / (KB * 300.0) * math.sqrt(KB * 300.0 / (2 * math.pi * M_AR)) * L / 4,
                                       rel=1e-12)
    n0 = sim.n
    for _ in range(20):
        sim._inject()                   # 流入だけを繰り返す (移動・並べ替えの前の位置を見る)
    assert sim.n > n0
    new = np.stack([sim._p["x"][n0:sim.n].get(), sim._p["y"][n0:sim.n].get()], axis=1)
    v = np.stack([sim._p["vx"][n0:sim.n].get(), sim._p["vy"][n0:sim.n].get()], axis=1)
    assert np.all((new[:, 0] >= 0.0) & (new[:, 0] <= L / 4))       # 上辺の左 1/4 だけ
    assert np.allclose(new[:, 1], H - sim._delta)                  # 上辺のすぐ内側
    assert np.all(v[:, 1] < 0.0)                                   # 内向き (下向き) に入る


def test_rz_equilibrium_cylinder_is_radially_uniform():
    p0, t0, rr = 10.0, 300.0, 0.01
    sim = _sim(_project({"init_pressure_pa": p0, "init_temperature_k": t0, "wall_temperature_k": t0,
                         "n_particles": 30000, "n_steps": 600, "avg_steps": 300, "seed": 8},
                        coord="rz", domain=[[0, 0], [L, 0], [L, rr], [0, rr]]))
    res = sim.run()
    vol = sim.vol
    n0 = p0 / (KB * t0)
    assert float(np.sum(res.n * vol) / vol.sum()) == pytest.approx(n0, rel=0.03)
    assert float(np.sum(res.t * res.n * vol) / np.sum(res.n * vol)) == pytest.approx(t0, rel=0.03)
    r_c = sim.mesh.nodes[sim.tris].mean(axis=1)[:, 1]
    inner = r_c < rr / 2
    n_in = float(np.sum(res.n[inner] * vol[inner]) / vol[inner].sum())
    n_out = float(np.sum(res.n[~inner] * vol[~inner]) / vol[~inner].sum())
    assert n_in == pytest.approx(n_out, rel=0.08)
    assert res.n_particles == 30000


def test_rz_axial_effusion():
    p0, t0, rr = 1.0, 300.0, 0.005
    sim = _sim(_project({
        "gas": {"d_ref_m": 1e-15},
        "boundaries": [{"edges": [3], "type": "inlet", "pressure_pa": p0, "temperature_k": t0},
                       {"edges": [1], "type": "outlet"}, {"edges": [2], "type": "symmetry"}],
        "init_pressure_pa": p0 / 2.0, "init_temperature_k": t0, "n_particles": 40000, "n_steps": 1500,
        "avg_steps": 500, "seed": 9}, mesh=1.2e-3, coord="rz", domain=[[0, 0], [L, 0], [L, rr], [0, rr]]))
    res = sim.run()
    vol = sim.vol
    assert float(np.sum(res.n * vol) / vol.sum()) == pytest.approx(0.5 * p0 / (KB * t0), rel=0.05)
    c_bar = math.sqrt(8.0 * KB * t0 / (math.pi * M_AR))
    assert float(np.sum(res.u[:, 0] * res.n * vol) / np.sum(res.n * vol)) == pytest.approx(0.5 * c_bar, rel=0.05)
    assert res.outflow == pytest.approx(res.inflow, rel=0.07)


def test_flow_around_solids_conserves_mass_and_never_enters_solids():
    t0 = 300.0
    regions = [
        {"id": "pin", "type": "conductor", "voltage": 0.0, "shape": {"kind": "circle", "center": [0.008, 0.005], "radius": 0.002}},
        {"id": "blk", "type": "dielectric", "eps_r": 4.0,
         "polygon": [[0.013, 0.0], [0.015, 0.0], [0.015, 0.006], [0.013, 0.006]]},
    ]
    sim = _sim(_project({
        "boundaries": [{"edges": [3], "type": "inlet", "pressure_pa": 20.0, "temperature_k": t0},
                       {"edges": [1], "type": "outlet", "pressure_pa": 5.0, "temperature_k": t0}],
        "init_pressure_pa": 12.0, "init_temperature_k": t0, "wall_temperature_k": t0, "n_particles": 40000,
        "n_steps": 2000, "avg_steps": 600, "seed": 11}, mesh=0.5e-3, regions=regions))
    res = sim.run()
    assert res.outflow == pytest.approx(res.inflow, rel=0.10)
    x = sim.x
    assert not np.any(sim.model.gas_at(x[:, 0], x[:, 1]) == False)  # noqa: E712
    # 固体のセル (表示要素) にはガスが無い
    cen = sim.mesh.nodes[sim.tris].mean(axis=1)
    inside = (np.hypot(cen[:, 0] - 0.008, cen[:, 1] - 0.005) < 0.0015) | (
        (cen[:, 0] > 0.0133) & (cen[:, 0] < 0.0147) & (cen[:, 1] < 0.0057))
    assert np.all(res.n[inside] == 0.0)


def test_hot_wall_heats_the_gas_monotonically():
    """左壁 600 K・右壁 300 K (上下は鏡面): 温度が壁温の間に単調に分布する。"""
    sim = _sim(_project({
        "boundaries": [{"edges": [3], "type": "wall", "temperature_k": 600.0},
                       {"edges": [1], "type": "wall", "temperature_k": 300.0},
                       {"edges": [0], "type": "symmetry"}, {"edges": [2], "type": "symmetry"}],
        "init_pressure_pa": 20.0, "init_temperature_k": 450.0, "wall_temperature_k": 300.0, "n_particles": 40000,
        "n_steps": 3000, "avg_steps": 1000, "seed": 12}))
    res = sim.run()
    cen = sim.mesh.nodes[sim.tris].mean(axis=1)
    a = sim.area
    t_slab = []
    for i in range(5):
        sel = (cen[:, 0] >= i * L / 5) & (cen[:, 0] < (i + 1) * L / 5)
        t_slab.append(float(np.sum(res.t[sel] * res.n[sel] * a[sel]) / np.sum(res.n[sel] * a[sel])))
    assert all(t_slab[i] > t_slab[i + 1] for i in range(4)), t_slab
    assert 300.0 < t_slab[-1] < t_slab[0] < 600.0
    assert res.n_particles == pytest.approx(sim.s.n_particles, rel=1e-9)


def test_smoothing_conserves_totals_and_stays_nonnegative():
    sim = _sim(_project({"init_pressure_pa": 10.0, "n_particles": 20000, "n_steps": 300, "avg_steps": 200,
                         "seed": 5, "smoothing_passes": 3}))
    res = sim.run()
    raw = [a.get() for a in (sim._acc_cnt, sim._acc_v, sim._acc_v2)]
    sm = sim._smooth_moments(*raw)
    for r, s_ in zip(raw, sm):
        assert s_.sum() == pytest.approx(r.sum(), rel=1e-12)
    assert np.all(sm[0] >= 0.0) and np.all(res.n >= 0.0) and np.all(res.t >= 0.0)


def test_server_dsmc_endpoints_use_gpu_engine_for_cartesian():
    from fastapi.testclient import TestClient

    from es_sim import server as srv

    p = json.loads(_project({"init_pressure_pa": 10.0, "n_particles": 5000, "n_steps": 200, "avg_steps": 100,
                             "seed": 1}).model_dump_json())
    client = TestClient(srv.app)
    r = client.post("/dsmc", json=p)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["n"]) == len(body["mesh"]["triangles"]) and body["n_particles"] == 5000
    with client.websocket_connect("/ws/dsmc") as ws:
        ws.send_text(json.dumps({"cmd": "start", "project": p}))
        msgs = []
        while True:
            m = ws.receive_json()
            msgs.append(m)
            if m["type"] in ("done", "error"):
                break
    assert [m["type"] for m in msgs][0] == "started" and msgs[-1]["type"] == "done", msgs[-1]
    assert any(m["type"] == "progress" for m in msgs)
