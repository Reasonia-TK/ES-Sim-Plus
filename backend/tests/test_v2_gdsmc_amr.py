"""GPU DSMC の AMR 対応 (葉セルを衝突・サンプリングのセルに、平均自由行程での動的再格子化) の
テスト (prompts/127)。CUDA が無ければ skip。"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from es_sim.device import cuda_available

pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA が使えません")

from es_sim.dsmc import KB  # noqa: E402
from es_sim.schema import Project  # noqa: E402

if cuda_available():
    from es_sim.amr.cell_layout import cell_of_points, mfp_tags
    from es_sim.gdsmc import GpuDsmcSimulation


def _project(w, h, amr, dsmc, regions=(), coord="xy", size=0.5e-3) -> Project:
    d = {"boundaries": [], "init_temperature_k": 300.0, "wall_temperature_k": 300.0, "seed": 1}
    d.update(dsmc)
    return Project.model_validate({
        "coord": coord,
        "geometry": {"domain": {"polygon": [[0, 0], [w, 0], [w, h], [0, h]]}, "boundaries": [],
                     "regions": list(regions)},
        "mesh": {"size": size, "mode": "cartesian", "amr": amr},
        "dsmc": d,
    })


def _level_means(sim, res):
    lvl = sim.lay.cell_level[sim._tri_cell]
    a = sim.area
    out = {}
    for lv in range(sim.lay.hier.n_levels):
        m = (lvl == lv) & (a > 0)
        if m.any():
            out[lv] = (float(np.sum(res.n[m] * a[m]) / a[m].sum()), float(np.sum(res.t[m] * a[m]) / a[m].sum()))
    return out


def test_equilibrium_box_uniform_across_levels():
    """密閉箱 + 指定矩形 (L1) + 導体の円の境界 (L2): どのレベルでも密度・温度が一様、導体内に分子が入らない。"""
    p0, t0 = 10.0, 300.0
    circ = {"id": "c", "type": "conductor", "voltage": 0.0,
            "shape": {"kind": "circle", "center": [0.013, 0.005], "radius": 0.002}}
    amr = {"max_level": 2, "buffer_cells": 2, "regions": [{"p1": [0.002, 0.002], "p2": [0.006, 0.008], "level": 1}]}
    sim = GpuDsmcSimulation(_project(0.02, 0.01, amr, {
        "init_pressure_pa": p0, "n_particles": 150000, "n_steps": 500, "avg_steps": 350, "smoothing_passes": 0},
        regions=[circ]))
    assert sim.lay is not None and sim.lay.hier.n_leaf_cells()[2] > 0
    assert sim.dt == pytest.approx(0.25 * sim.lay.h_min / math.sqrt(2.0 * KB * t0 / sim.m))
    res = sim.run()
    n0 = p0 / (KB * t0)
    means = _level_means(sim, res)
    assert set(means) == {0, 1, 2}
    for lv, (n, t) in means.items():
        assert n == pytest.approx(n0, rel=0.02), lv
        assert t == pytest.approx(t0, rel=0.02), lv
    assert res.n_particles == 150000
    x = sim.x
    assert not np.any(np.hypot(x[:, 0] - 0.013, x[:, 1] - 0.005) < 0.002 - 1e-9)
    assert len(res.n) == len(sim.tris) == 2 * int(np.sum(sim.lay.cell_level >= 0)) - 2 * int(
        np.sum(np.hypot(sim.lay.cell_xy[sim.lay.cell_level >= 0, 0] - 0.013,
                        sim.lay.cell_xy[sim.lay.cell_level >= 0, 1] - 0.005) <= 0.002))


def test_channel_with_refined_half_matches_uniform():
    """圧力駆動チャネル (20 → 5 Pa): 左半分だけ L1 の AMR と一様な細かい格子で圧力分布が一致する。"""
    L, H = 0.02, 0.005
    cfg = {"boundaries": [{"edges": [3], "type": "inlet", "pressure_pa": 20.0, "temperature_k": 300.0},
                          {"edges": [1], "type": "outlet", "pressure_pa": 5.0, "temperature_k": 300.0}],
           "init_pressure_pa": 12.0, "n_particles": 100000, "n_steps": 3000, "avg_steps": 1500, "seed": 3,
           "dt": 2.0e-8}
    amr = {"max_level": 1, "refine_boundaries": False, "regions": [{"p1": [0, 0], "p2": [0.01, H], "level": 1}]}
    slabs = {}
    for name, p in (("amr", _project(L, H, amr, cfg)),
                    ("fine", Project.model_validate({**_project(L, H, None, cfg).model_dump(),
                                                     "mesh": {"size": 0.25e-3, "mode": "cartesian"}}))):
        sim = GpuDsmcSimulation(p)
        res = sim.run()
        assert res.outflow == pytest.approx(res.inflow, rel=0.1)
        cen = sim.mesh.nodes[sim.tris].mean(axis=1)
        a = sim.area
        slabs[name] = np.array([float(np.sum(res.p[m] * a[m]) / a[m].sum()) for m in
                                [(cen[:, 0] >= i * L / 4) & (cen[:, 0] < (i + 1) * L / 4) for i in range(4)]])
    np.testing.assert_allclose(slabs["amr"], slabs["fine"], rtol=0.03)


def test_regrid_follows_mean_free_path_and_keeps_molecules(monkeypatch):
    """平均自由行程が短い所だけ細分化し、長くなれば粗視化する (ヒステリシス付き)。分子は数・位置とも保たれる。"""
    amr = {"max_level": 2, "refine_boundaries": False, "dsmc_regrid_every": 50, "dsmc_h_over_mfp": 0.5}
    sim = GpuDsmcSimulation(_project(0.02, 0.01, amr, {
        "init_pressure_pa": 5.0, "n_particles": 50000, "n_steps": 400, "avg_steps": 100}))
    assert sim.lay.hier.max_level == 0 and sim._regrid_every == 50
    x0 = sim.x.copy()
    n0 = sim.n

    def fake_lambda(lam_left, lam_right):
        def f():
            lam = np.where(sim.lay.cell_xy[:, 0] < 0.006, lam_left, lam_right)
            return np.where(sim.cell_vol > 0.0, lam, np.inf)
        return f

    # 左 (x < 6 mm) の λ = 0.3 mm: 格子幅 0.5 mm → L1 (0.25 mm、比 0.83) → L2 (0.125 mm、比 0.42)
    monkeypatch.setattr(sim, "_mean_free_path", fake_lambda(0.3e-3, 10e-3))
    assert sim._regrid()
    hier = sim.lay.hier
    assert hier.max_level == 2
    lvl = sim.lay.cell_level
    xy = sim.lay.cell_xy
    assert np.all(xy[lvl == 2, 0] < 0.0085)          # 左側 (+ proper nesting の緩衝帯) だけが L2
    assert np.any(lvl == 0) and np.all(xy[lvl == 0, 0] > 0.006)
    assert sim.n == n0 and np.array_equal(np.sort(sim.x, axis=0), np.sort(x0, axis=0))
    keys = sim._key[: sim.n].get()
    np.testing.assert_array_equal(keys, cell_of_points(sim.lay, sim.x[:, 0], sim.x[:, 1]))
    assert np.all(sim.cell_vol[keys] > 0.0)
    counts = sim._count[: sim.n_cells].get()
    assert int(counts.sum()) == sim.n
    for _ in range(20):
        sim.step()
    assert sim.n == n0

    # λ = 0.6 mm: L1 の比 0.42 は新しく細分化する閾値 0.5 未満だが、保持の閾値 0.35 (= 0.7 × 0.5) を超える
    # ので L2 を保つ (ヒステリシス)。保持しない判定なら L1 のセルはタグされない
    monkeypatch.setattr(sim, "_mean_free_path", fake_lambda(0.6e-3, 10e-3))
    fresh = mfp_tags(sim.lay, sim._mean_free_path(), sim.lay.hier, 0.5, keep=None)
    assert not fresh[1].any() and fresh[0].any()
    assert not sim._regrid()
    assert sim.lay.hier.max_level == 2
    monkeypatch.setattr(sim, "_mean_free_path", fake_lambda(5e-3, 10e-3))
    assert sim._regrid()
    assert sim.lay.hier.max_level == 0 and sim.n == n0
    assert [r["leaf_cells"] for r in sim.regrid_log] == [hier.n_leaf_cells(), [sim.n_cells]]


def test_regrid_during_run_on_channel():
    """実行中の再格子化 (流れの発達に合わせて変わる) と、平均区間の格子上の結果。"""
    L, H = 0.02, 0.005
    cfg = {"boundaries": [{"edges": [3], "type": "inlet", "pressure_pa": 40.0, "temperature_k": 300.0},
                          {"edges": [1], "type": "outlet", "pressure_pa": 1.0, "temperature_k": 300.0}],
           "init_pressure_pa": 10.0, "n_particles": 60000, "n_steps": 3000, "avg_steps": 1000, "seed": 5,
           "dt": 1.0e-7}
    amr = {"max_level": 2, "refine_boundaries": False, "dsmc_regrid_every": 500, "dsmc_h_over_mfp": 0.5}
    sim = GpuDsmcSimulation(_project(L, H, amr, cfg))
    res = sim.run()
    assert len(sim.regrid_log) >= 2
    assert all(r["step"] < 2000 for r in sim.regrid_log)       # 平均区間 (最後の 1000 ステップ) の前だけ
    assert len(res.n) == len(sim.tris)
    cen = sim.mesh.nodes[sim.tris].mean(axis=1)
    a = sim.area
    p_in = float(np.sum((res.p * a)[cen[:, 0] < L / 4]) / a[cen[:, 0] < L / 4].sum())
    p_out = float(np.sum((res.p * a)[cen[:, 0] > 3 * L / 4]) / a[cen[:, 0] > 3 * L / 4].sum())
    assert p_in > p_out
    lvl = sim.lay.cell_level
    # 圧力の高い入口側ほど細かい (平均自由行程が短い)
    assert np.mean(lvl[(lvl >= 0) & (sim.lay.cell_xy[:, 0] < L / 4)]) >= np.mean(
        lvl[(lvl >= 0) & (sim.lay.cell_xy[:, 0] > 3 * L / 4)])


def test_axisymmetric_cylinder_equilibrium_with_amr():
    """軸対称の密閉円筒 (軸上を L1): 径方向・レベルによらず密度が一様。"""
    p0, t0 = 10.0, 300.0
    amr = {"max_level": 1, "refine_boundaries": False, "regions": [{"p1": [0.0, 0.0], "p2": [0.02, 0.004], "level": 1}]}
    sim = GpuDsmcSimulation(_project(0.02, 0.01, amr, {
        "init_pressure_pa": p0, "n_particles": 150000, "n_steps": 500, "avg_steps": 350, "smoothing_passes": 0},
        coord="rz"))
    res = sim.run()
    n0 = p0 / (KB * t0)
    for lv, (n, t) in _level_means(sim, res).items():
        assert n == pytest.approx(n0, rel=0.03), lv
        assert t == pytest.approx(t0, rel=0.03), lv


def test_dsmc_endpoint_returns_amr_mesh_same_as_mesh_endpoint():
    from fastapi.testclient import TestClient

    import es_sim.server as server

    amr = {"max_level": 1, "regions": [{"p1": [0.004, 0.002], "p2": [0.01, 0.008], "level": 1}]}
    p = _project(0.02, 0.01, amr, {"init_pressure_pa": 5.0, "n_particles": 20000, "n_steps": 60, "avg_steps": 30})
    client = TestClient(server.app)
    body = p.model_dump(mode="json")
    r = client.post("/dsmc", json=body)
    assert r.status_code == 200, r.text
    res = r.json()
    m = client.post("/mesh", json=body).json()
    assert res["mesh"]["triangles"] == m["triangles"] and res["mesh"]["nodes"] == m["nodes"]
    assert len(res["n"]) == len(m["triangles"])
    assert json.dumps(res["t"])  # 直列化できる
