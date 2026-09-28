"""v2 AMR P4c: PIC の動的再格子化 (prompts/123) の検証。

- 移し替えの補助: 点電荷の CIC 堆積が総和を保存、節点値の補間が線形場で厳密
- λ_D のタグ: デバイ長の短い (密度の高い) 所だけ細分化、既存の細分化はヒステリシスで保つ
- GPU PIC: 接地箱の高密度プラズマでバルクだけが細かくなる (シース側は粗いまま)、再格子化の前後で
  粒子・表面電荷が保存され、電子数の推移が静的な計算と一致、フレーム・done の格子と値の大きさが一致
- 回帰: フレームを作る実行でも history の末尾が正しい (非ブロッキングストリームの読み出し待ち)
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from es_sim.amr import AmrHierarchy, AmrSpec, build_pic_layout
from es_sim.amr.pic_layout import debye_kappa, debye_tags, deposit_points, sample_nodes
from es_sim.device import cuda_available
from es_sim.eb.grid import make_grid
from es_sim.geom.model import GeometryModel
from es_sim.schema import Project

needs_cuda = pytest.mark.skipif(not cuda_available(), reason="CUDA (CuPy) が使えない環境")
QE = 1.602176634e-19
EPS0 = 8.8541878128e-12


def _box(amr: dict | None, n_steps: int = 1200, dielectric: bool = False) -> Project:
    d = {
        "geometry": {"domain": {"polygon": [[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01]]},
                     "boundaries": [{"edges": [0, 1, 2, 3], "type": "dirichlet", "voltage": 0.0}]},
        "mesh": {"size": 0.0004, "mode": "cartesian"},
        "pic": {"initial_plasma": {"density": 1e16, "te_ev": 3.0, "ti_ev": 0.0, "ion_mass_amu": 40.0,
                                   "immobile_ions": True, "seed": 2},
                "n_macro": 200000, "n_steps": n_steps, "avg_steps": 300, "frame_every": 100, "phase_bins": 0},
    }
    if dielectric:
        d["geometry"]["regions"] = [{"id": "d", "type": "dielectric", "eps_r": 4.0,
                                     "polygon": [[0.0, 0.0], [0.002, 0.0], [0.002, 0.01], [0.0, 0.01]]}]
    if amr is not None:
        d["mesh"]["amr"] = amr
    return Project.model_validate(d)


_REGRID = {"max_level": 2, "refine_boundaries": False, "blocking_factor": 4,
           "pic_regrid_every": 200, "pic_h_over_debye": 1.5}


def _layout(p: Project, spec: AmrSpec):
    model = GeometryModel(p)
    grid = make_grid(model.domain, float(p.mesh.size))
    return model, grid, build_pic_layout(model, AmrHierarchy(model, grid, spec))


# ---- 移し替えの補助 (CPU) -----------------------------------------------------------------


def test_remap_helpers_conserve_charge_and_reproduce_linear_fields():
    p = _box(None)
    _, _, lay = _layout(p, AmrSpec(max_level=2, refine_boundaries=False, blocking_factor=4,
                                   regions=((0.003, 0.003, 0.006, 0.007, 2),)))
    assert lay.hier.n_levels == 3
    rng = np.random.default_rng(0)
    x, y, q = rng.uniform(0, 0.01, 5000), rng.uniform(0, 0.01, 5000), rng.normal(size=5000)
    dep = deposit_points(lay, x, y, q)
    assert dep.sum() == pytest.approx(q.sum(), abs=1e-9)
    phi = 3.0 * lay.xy[:, 0] - 7.0 * lay.xy[:, 1] + 0.5
    assert np.max(np.abs(sample_nodes(lay, phi, x, y) - (3.0 * x - 7.0 * y + 0.5))) < 1e-12
    assert np.array_equal(lay.xy[lay.node_of_c], lay.op.xy[lay.node_of_c])
    assert np.all(lay.op.fixed_group[lay.node_of_c] < 0)


def test_debye_tags_follow_the_debye_length_with_hysteresis():
    p = _box(None)
    spec = AmrSpec(max_level=2, refine_boundaries=False, blocking_factor=4, pic_regrid_every=100, pic_h_over_debye=1.0)
    model, grid, lay = _layout(p, spec)
    # 中央の 4 mm 角だけ高密度 (λ_D ≈ 0.13 mm < 格子幅 0.4 mm)、周りは低密度 (λ_D ≈ 1.3 mm)
    x, y = lay.xy[:, 0], lay.xy[:, 1]
    dense = (np.abs(x - 0.005) < 0.002) & (np.abs(y - 0.005) < 0.002)
    n = np.where(dense, 1e16, 1e14)
    te = 3.0
    w = n * lay.node_vol_gas
    ke = 1.5 * te * QE * w
    kappa = debye_kappa(lay, w, ke, 1)
    assert kappa[dense].max() == pytest.approx(np.sqrt(QE * 1e16 / (EPS0 * te)), rel=1e-9)
    cand = lay.hier
    for _ in range(3):
        cand = AmrHierarchy(model, grid, spec, extra_tags=debye_tags(lay, kappa, cand, spec.pic_h_over_debye))
    assert cand.n_levels == 3
    for x0, y0, x1, y1 in cand.level_boxes(2):
        assert x0 >= 0.0025 and x1 <= 0.0075 and y0 >= 0.0025 and y1 <= 0.0075
    # ヒステリシス: 密度が少し下がってレベル 1 の 格子幅/λ_D (≈1.35) が閾値 1.6 を下回っても、
    # 閾値 × 0.7 を超える間は既存の細分化 (レベル 2) を保つ
    kappa2 = kappa * 0.9
    lay2 = build_pic_layout(model, cand)
    kappa2_new = sample_nodes(lay, kappa2, lay2.xy[:, 0], lay2.xy[:, 1])
    kept = AmrHierarchy(model, grid, spec, extra_tags=debye_tags(lay2, kappa2_new, cand, 1.6, keep=cand))
    dropped = AmrHierarchy(model, grid, spec, extra_tags=debye_tags(lay2, kappa2_new, cand, 1.6))
    assert kept.n_levels == 3 and kept.n_leaf_cells()[-1] > 0
    assert dropped.n_levels < 3


# ---- GPU PIC -------------------------------------------------------------------------------


@needs_cuda
def test_pic_regrid_refines_the_bulk_and_matches_static_run():
    from es_sim.gpic import GpuPicSimulation

    stat = GpuPicSimulation(_box(None))
    h_stat, _ = stat.run_batch(store_frames=False)
    sim = GpuPicSimulation(_box(_REGRID))
    assert sim.amr is not None and sim.amr.hier.n_levels == 1         # 始めは細分化なし
    hist, frames = sim.run_batch(store_frames=True)
    assert len(sim.regrid_log) >= 1 and sim.mesh_version == len(sim.regrid_log)
    assert all(r["step"] < 901 for r in sim.regrid_log)               # 時間平均区間 (901〜) の前だけ
    hier = sim.amr.hier
    assert hier.n_levels == 3
    # 最細レベルはバルクだけ (壁際のシースは電子密度が低く λ_D が長いので粗いまま)
    for x0, y0, x1, y1 in hier.level_boxes(2):
        assert x0 > 0.0 and x1 < 0.01 and y0 > 0.0 and y1 < 0.01
    # 電子数の推移は静的な計算と同じ (シース形成で 12% 程度失う)
    ne, ne_s = np.asarray(hist["n_e"], dtype=float), np.asarray(h_stat["n_e"], dtype=float)
    assert np.all(np.diff(hist["t"]) > 0) and np.all(ne > 0)
    assert ne[-1] == pytest.approx(ne_s[-1], rel=0.02)
    # フレームの値の大きさは直前に届いた格子と一致
    nodes = tris = None
    for f in frames:
        if "mesh" in f:
            nodes, tris = len(f["mesh"]["nodes"]), len(f["mesh"]["triangles"])
        if nodes is not None:
            assert len(f["phi"]) == nodes and len(f["n_e"]) == tris
    assert len(sim.fields["n_e"]) == len(sim.mesh.nodes)
    assert len(sim.fields["e_abs"]) == len(sim.mesh.triangles)


@needs_cuda
def test_regrid_conserves_particles_and_dielectric_surface_charge():
    from es_sim.gpic import GpuPicSimulation

    sim = GpuPicSimulation(_box(_REGRID, n_steps=150, dielectric=True))
    sim.run_batch(store_frames=False)
    before = {k: sim.get_particles(k) for k in ("electron", "ion")}
    q_before = float(sim._q_surf.get().sum())
    assert q_before != 0.0                                              # 誘電体が帯電している
    for _ in range(2):
        sim._launch_tag_deposit()
    assert sim._regrid()
    after = {k: sim.get_particles(k) for k in ("electron", "ion")}
    for k in before:
        for f in ("x", "y", "vx", "w"):
            assert np.array_equal(before[k][f], after[k][f])
    assert float(sim._q_surf.get().sum()) == pytest.approx(q_before, rel=1e-12)
    sim.pic.n_steps = 50
    hist, _ = sim.run_batch(store_frames=False)
    assert np.all(np.isfinite(hist["phi_max"])) and hist["n_e"][-1] > 0


@needs_cuda
def test_history_is_complete_when_frames_are_made():
    """回帰: フレームを作る実行でも history の末尾が 0 にならない (ストリームの完了待ち)。"""
    from es_sim.gpic import GpuPicSimulation

    sim = GpuPicSimulation(_box(None, n_steps=400))
    hist, frames = sim.run_batch(store_frames=True)
    assert len(frames) >= 2 and frames[0]["step"] < 400        # 途中のフレームで history を読んでいる
    t = np.asarray(hist["t"])
    assert len(t) == 400 and np.all(np.diff(t) > 0)
    assert np.all(np.asarray(hist["n_e"]) > 0)


@needs_cuda
def test_ws_pic_sends_new_mesh_after_regrid():
    from fastapi.testclient import TestClient

    from es_sim.server import app

    p = json.loads(_box(_REGRID, n_steps=600).model_dump_json())
    p["pic"]["frame_every"] = 50
    client = TestClient(app)
    with client.websocket_connect("/ws/pic") as ws:
        ws.send_text(json.dumps({"cmd": "start", "project": p}))
        msgs = []
        while True:
            m = ws.receive_json()
            msgs.append(m)
            if m["type"] in ("done", "error"):
                break
    assert msgs[-1]["type"] == "done", msgs[-1]
    done = msgs[-1]
    assert done["regrids"] and "mesh" in done
    assert len(done["fields"]["n_e"]) == len(done["mesh"]["nodes"])
    nodes = len(msgs[0]["mesh"]["nodes"])
    for m in msgs[1:-1]:
        if "mesh" in m:
            nodes = len(m["mesh"]["nodes"])
        assert len(m["phi"]) == nodes
