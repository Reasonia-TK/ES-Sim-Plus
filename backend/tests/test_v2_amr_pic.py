"""v2 AMR P4b: 合成格子の GPU AMG-PCG・節点電場ステンシル・AMR 上の GPU PIC の検証 (prompts/122)。

- GPU AMG-PCG (Chebyshev 平滑化) が CPU の pyamg と一致、CUDA Graph に取り込んだ warm start の
  固定反復が収束、特異問題 (全周期) も収束
- 節点電場ステンシル: 細分化なしで一様格子の efield_nodes カーネルと一致、細分化ありで線形場が厳密
- AMR PIC: 細分化なしの AMR 経路が一様 PIC と機械精度で一致、GPU の所属セル判定・堆積が
  CPU の参照と一致、プラズマ振動 2f_pe・エネルギー保存 (細分化パッチあり)、一様な熱プラズマが
  パッチの内外で一様なまま (粗細界面の自己力が問題にならない)、CCP (RF・MCC) で細かい一様格子と一致
- 周期境界の一様格子 PIC の回帰: 従属節点の表示密度と初期密度の規格化

GPU (CUDA) が無い環境ではステンシルの線形場テスト以外はスキップする。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from es_sim.amr import AmrHierarchy, AmrSpec, assemble_composite, build_hierarchy, solve_composite
from es_sim.amr.composite import _canon_key
from es_sim.amr.fieldops import build_efield_stencil
from es_sim.amr.locate import locate_fine, to_fine_index
from es_sim.amr.pic_layout import build_pic_layout
from es_sim.device import cuda_available
from es_sim.field import group_voltages
from es_sim.geom.model import GeometryModel
from es_sim.schema import Project

from test_v2_amr import LR, PLATES, _coax, _project

needs_cuda = pytest.mark.skipif(not cuda_available(), reason="CUDA (CuPy) が使えない環境")
EPS0 = 8.8541878128e-12
QE = 1.602176634e-19
ME = 9.1093837015e-31


# ---- GPU AMG-PCG -------------------------------------------------------------------------


@needs_cuda
def test_gpu_amg_pcg_matches_cpu_and_async_warm_start_converges():
    import cupy as cp

    from es_sim.amr.gpu_solver import AmgGpuSolver

    p = _coax(0.1 / 128, amr={"max_level": 2})
    model = GeometryModel(p)
    op = assemble_composite(model, build_hierarchy(p, model=model))
    v = group_voltages(model, None)
    phi_cpu, info_cpu = solve_composite(op, v, device="cpu")
    phi_gpu, info_gpu = solve_composite(op, v, device="cuda")
    assert info_gpu.converged and info_gpu.iterations <= info_cpu.iterations + 5
    assert np.max(np.abs(phi_gpu - phi_cpu)) < 1e-7
    # 固定反復の同期なし求解を CUDA Graph に取り込み、warm start で繰り返す (PIC と同じ使い方)
    vK = np.zeros(op.coup_c.shape[1])
    vK[: v.size] = v
    b = op.q_c + op.coup_c @ vK
    solver = AmgGpuSolver(op.A_c)
    assert not solver.direct and solver.n_levels >= 3
    bd, xd = cp.asarray(b), cp.zeros(op.n_unknowns)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        solver.launch_solve(bd, xd, 3)
        graph = stream.end_capture()
    rel = []
    for _ in range(4):
        graph.launch(stream)
        stream.synchronize()
        rn, bn = solver.monitor()
        rel.append(rn / bn)
    assert rel[-1] < 1e-10 and all(a > b for a, b in zip(rel, rel[1:]))


@needs_cuda
def test_gpu_amg_pcg_singular_periodic_problem():
    import cupy as cp

    from es_sim.amr.gpu_solver import AmgGpuSolver

    p = _project(PLATES, [
        {"id": "p", "type": "charge", "rho": 1e-8, "shape": {"kind": "circle", "center": [0.03, 0.025], "radius": 0.008}},
        {"id": "n", "type": "charge", "rho": -1e-8, "shape": {"kind": "circle", "center": [0.07, 0.025], "radius": 0.008}},
    ], [{"edges": [3, 1], "type": "periodic"}, {"edges": [0, 2], "type": "periodic"}], h=0.0005, amr={"max_level": 2})
    model = GeometryModel(p)
    op = assemble_composite(model, build_hierarchy(p, model=model))
    assert op.singular
    phi_cpu, _ = solve_composite(op, np.zeros(1), device="cpu")
    phi_gpu, info = solve_composite(op, np.zeros(1), device="cuda")
    assert info.converged
    assert np.max(np.abs(phi_gpu - phi_cpu)) < 1e-9 * np.max(np.abs(phi_cpu)) + 1e-12
    solver = AmgGpuSolver(op.A_c, singular=True)
    bd, xd = cp.asarray(op.q_c), cp.zeros(op.n_unknowns)
    for _ in range(6):
        bd[...] = cp.asarray(op.q_c)
        solver.launch_solve(bd, xd, 3)
    rn, bn = solver.monitor()
    assert rn / bn < 1e-9


# ---- 節点電場ステンシル ------------------------------------------------------------------


@needs_cuda
@pytest.mark.parametrize("case", ["coax", "plates_diel_cond", "periodic"])
def test_efield_stencil_equals_uniform_kernel_without_refinement(case):
    import cupy as cp

    from es_sim.device.cuda import get_kernel, grid_2d
    from es_sim.eb.build import MASK_FIXED, build_level
    from es_sim.eb.grid import make_grid

    if case == "coax":
        p = _coax(0.1 / 48)
    elif case == "plates_diel_cond":
        p = _project(PLATES, [
            {"id": "d", "type": "dielectric", "eps_r": 4.0, "polygon": [[0.04, 0.01], [0.06, 0.01], [0.06, 0.04], [0.04, 0.04]]},
            {"id": "c", "type": "conductor", "voltage": 30.0, "shape": {"kind": "circle", "center": [0.021, 0.0243], "radius": 0.0061}},
        ], LR, h=0.1 / 48)
    else:
        p = _project(PLATES, [
            {"id": "c", "type": "conductor", "voltage": 5.0, "shape": {"kind": "circle", "center": [0.05, 0.025], "radius": 0.007}},
        ], [{"edges": [3, 1], "type": "periodic"}, {"edges": [0], "type": "dirichlet", "voltage": 0.0}], h=0.1 / 64)
    model = GeometryModel(p)
    grid = make_grid(model.domain, float(p.mesh.size))
    lop = build_level(model, grid, full=True)
    hier = build_hierarchy(p, spec=AmrSpec(max_level=0), model=model)
    op = assemble_composite(model, hier)
    Gx, Gy = build_efield_stencil(model, hier, op)
    K = Gx.shape[1] - op.keys.size
    vK = np.zeros(K)
    v = group_voltages(model, None)
    vK[: v.size] = v
    nx, ny = grid.nx, grid.ny
    phi_u = np.random.default_rng(0).standard_normal((ny + 1, nx + 1))
    fixed = lop.mask == MASK_FIXED
    phi_u[fixed] = vK[np.maximum(lop.fixed_group, 0)][fixed]
    if model.periodic_x:
        phi_u[:, nx] = phi_u[:, 0]
    kI, kJ = op.keys % (nx + 1), op.keys // (nx + 1)
    ext = np.concatenate([phi_u[kJ, kI], vK])
    ex_amr, ey_amr = Gx @ ext, Gy @ ext
    ex, ey = cp.zeros((ny + 1, nx + 1)), cp.zeros((ny + 1, nx + 1))
    g, bl = grid_2d(nx + 1, ny + 1)
    get_kernel("pic", "efield_nodes")(g, bl, (
        ex, ey, cp.asarray(phi_u), cp.asarray(lop.mask), cp.asarray(lop.cut_theta),
        cp.asarray(np.maximum(lop.cut_group, 0).astype(np.int32)), cp.asarray(vK), np.int32(nx), np.int32(ny),
        np.float64(grid.dx), np.float64(grid.dy), np.int32(model.periodic_x), np.int32(model.periodic_y)))
    scale = float(np.max(np.abs(ex.get())))
    assert np.max(np.abs(ex_amr - ex.get()[kJ, kI])) < 1e-9 * scale
    assert np.max(np.abs(ey_amr - ey.get()[kJ, kI])) < 1e-9 * scale


def test_efield_stencil_is_exact_for_linear_fields_with_refinement():
    p = _project(PLATES, [], LR, h=0.002,
                 amr={"max_level": 2, "regions": [{"p1": [0.03, 0.01], "p2": [0.06, 0.03], "level": 2}]})
    model = GeometryModel(p)
    hier = build_hierarchy(p, model=model)
    op = assemble_composite(model, hier)
    assert op.hanging.sum() > 0
    Gx, Gy = build_efield_stencil(model, hier, op)
    vK = np.zeros(Gx.shape[1] - op.keys.size)
    vK[:2] = [0.0, 100.0]
    for f, ex_true, ey_true in ((1000.0 * op.xy[:, 0], -1000.0, 0.0), (0.0 * op.xy[:, 0], 0.0, 0.0)):
        ext = np.concatenate([f, vK if f.any() else np.zeros_like(vK)])
        assert np.max(np.abs(Gx @ ext - ex_true)) < 1e-8
        assert np.max(np.abs(Gy @ ext - ey_true)) < 1e-8


# ---- AMR PIC -------------------------------------------------------------------------------


def _osc_project(n_steps: int, n_macro: int = 60000, amr: dict | None = None) -> Project:
    d = {
        "geometry": {
            "domain": {"polygon": [[0, 0], [0.01, 0], [0.01, 0.03], [0, 0.03]]},
            "boundaries": [{"edges": [0, 1, 2, 3], "type": "dirichlet", "voltage": 0.0}],
        },
        "mesh": {"size": 8e-4, "mode": "cartesian"},
        "pic": {
            "initial_plasma": {"density": 1e14, "te_ev": 0.0, "ti_ev": 0.0, "ion_mass_amu": 40.0,
                               "immobile_ions": True, "seed": 1},
            "n_macro": n_macro, "dt": None, "n_steps": n_steps, "frame_every": 1000,
        },
    }
    if amr is not None:
        d["mesh"]["amr"] = amr
    return Project.model_validate(d)


_PATCH = {"max_level": 1, "refine_boundaries": False, "blocking_factor": 4,
          "regions": [{"p1": [0.003, 0.01], "p2": [0.007, 0.02], "level": 1}]}


def _perturb(sim, v0: float) -> None:
    p = sim.get_particles("electron")
    sim.set_particles("electron", vx=p["vx"] + v0 * np.sin(2.0 * np.pi * p["x"] / 0.01))


@needs_cuda
def test_amr_pic_without_refinement_equals_uniform_pic():
    """細分化の無い階層を AMR 経路で強制しても、一様格子 PIC と機械精度で一致する。"""
    from es_sim.gpic import GpuPicSimulation

    p = _osc_project(20, n_macro=20000)
    uni = GpuPicSimulation(p)
    orig = GpuPicSimulation.__dict__["_build_amr_layout"]
    GpuPicSimulation._build_amr_layout = staticmethod(
        lambda project, model, grid: build_pic_layout(model, AmrHierarchy(model, grid, AmrSpec(max_level=0))))
    try:
        amr = GpuPicSimulation(p)
    finally:
        GpuPicSimulation._build_amr_layout = orig
    assert amr.amr is not None and uni.amr is None
    for sim in (uni, amr):
        _perturb(sim, 3.0e3)
    for _ in range(20):
        uni.step()
        phi_a = amr.step()
    kI = amr.amr.op.keys % (uni.grid.nx + 1)
    kJ = amr.amr.op.keys // (uni.grid.nx + 1)
    phi_u = uni._phi.get()[kJ, kI]
    assert np.max(np.abs(phi_a.get() - phi_u)) < 1e-12 * np.max(np.abs(phi_u)) + 1e-15
    pu, pa = uni.get_particles("electron"), amr.get_particles("electron")
    assert np.max(np.abs(pa["x"] - pu["x"])) < 1e-12
    assert np.max(np.abs(pa["vx"] - pu["vx"])) < 1e-9 * np.max(np.abs(pu["vx"]))


@needs_cuda
def test_amr_pic_locate_and_deposit_match_cpu_reference():
    from es_sim.gpic import GpuPicSimulation

    sim = GpuPicSimulation(_osc_project(10, n_macro=1000, amr={
        "max_level": 2, "refine_boundaries": False, "blocking_factor": 4,
        "regions": [{"p1": [0.002, 0.008], "p2": [0.006, 0.015], "level": 2}]}))
    lay, hier = sim.amr, sim.amr.hier
    assert hier.n_levels == 3
    rng = np.random.default_rng(1)
    n = 100000
    x, y, w = rng.uniform(0, 0.01, n), rng.uniform(0, 0.03, n), rng.uniform(0.5, 1.5, n)
    sim.set_particles("electron", x=x, y=y, vx=np.zeros(n), vy=np.zeros(n), vz=np.zeros(n), w=w)
    import cupy as cp

    out = cp.zeros(lay.n_nodes)
    el = sim.species["electron"]
    sim._k["deposit"](sim._grid1(el.cap), (256,), (
        el.x, el.y, el.vx, el.vy, el.vz, el.w, sim._cnt, np.int32(0), np.int32(0), np.float64(1.0), out, out,
        np.int32(0), np.int64(lay.n_nodes), np.int32(0), sim._prm, *sim._grid_args, np.int32(sim._px),
        np.int32(sim._py), *sim._gx))
    I, J = to_fine_index(hier, x, y)
    lvl, i, j, wx, wy = locate_fine(hier, I, J)
    s = hier.max_level - lvl
    ref = np.zeros(lay.n_nodes)
    for a, b, wt in ((0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)), (0, 1, (1 - wx) * wy), (1, 1, wx * wy)):
        np.add.at(ref, lay.op.node_index(_canon_key(hier, (i + a) << s, (j + b) << s)), w * wt)
    assert np.max(np.abs(out.get() - ref)) < 1e-9
    assert out.get().sum() == pytest.approx(w.sum(), rel=1e-12)
    # 一様に分布した粒子の密度推定 (堆積 / 双対体積) は界面をまたいで平坦
    dens = ref / lay.node_vol_gas
    interior = (lay.xy[:, 0] > 0.0005) & (lay.xy[:, 0] < 0.0095) & (lay.xy[:, 1] > 0.0005) & (lay.xy[:, 1] < 0.0295)
    hang = np.zeros(lay.n_nodes, dtype=bool)
    unk = np.nonzero(lay.op.fixed_group < 0)[0]
    hang[unk] = lay.op.hanging[lay.op.u_of_node[unk]]
    assert np.mean(dens[hang & interior]) == pytest.approx(np.mean(dens[interior]), rel=0.03)
    # 全体の体積は domain 面積 (軸対称でない 2D は奥行き 1 m あたり)
    assert lay.total_gas_volume == pytest.approx(0.01 * 0.03, rel=1e-12)
    assert lay.cell_vol_gas.sum() == pytest.approx(0.01 * 0.03, rel=1e-12)


@needs_cuda
def test_amr_pic_plasma_oscillation_and_energy_with_refinement_patch():
    from es_sim.gpic import GpuPicSimulation

    sim = GpuPicSimulation(_osc_project(400, amr=_PATCH))
    assert sim.amr is not None and int(sim.amr.op.hanging.sum()) > 0
    _perturb(sim, 3.0e3)
    hist, _ = sim.run_batch(store_frames=False)
    ke = np.asarray(hist["ke_e"])
    s = ke - ke.mean()
    idx = np.nonzero((s[:-1] < 0) & (s[1:] >= 0))[0]
    tz = idx + s[idx] / (s[idx] - s[idx + 1])
    f = (len(tz) - 1) / ((tz[-1] - tz[0]) * sim.dt)
    wpe = math.sqrt(1e14 * QE**2 / (EPS0 * ME))
    assert f == pytest.approx(2.0 * wpe / (2.0 * math.pi), rel=0.05)
    total = ke + np.asarray(hist["ke_i"]) + np.asarray(hist["fe"])
    assert np.max(np.abs(total - total[0])) / total[0] < 0.06


def _thermal_box(patch: bool) -> Project:
    d = {
        "geometry": {"domain": {"polygon": [[0, 0], [0.02, 0], [0.02, 0.02], [0, 0.02]]},
                     "boundaries": [{"edges": [3, 1], "type": "periodic"}, {"edges": [0, 2], "type": "periodic"}]},
        "mesh": {"size": 0.02 / 64, "mode": "cartesian"},
        "pic": {"initial_plasma": {"density": 1e15, "te_ev": 1.0, "ti_ev": 0.0, "ion_mass_amu": 40.0,
                                   "immobile_ions": True, "seed": 3},
                "n_macro": 200000, "n_steps": 2500, "frame_every": 100000, "avg_steps": 2000, "phase_bins": 0},
    }
    if patch:
        d["mesh"]["amr"] = {"max_level": 1, "refine_boundaries": False, "blocking_factor": 4,
                            "regions": [{"p1": [0.0075, 0.0075], "p2": [0.0125, 0.0125], "level": 1}]}
    return Project.model_validate(d)


@needs_cuda
def test_thermal_plasma_stays_uniform_across_refinement_patch():
    """粗細界面の自己力 (合成格子のグリーン関数の非対称) で粒子が細分化領域に集まらないこと。

    同じ初期粒子の一様格子との比較で、パッチ内外の時間平均密度の比が変わらない (差 < 0.5%)。
    自己力による電位は 1 粒子あたり ~0.1·w·e/(2πε0) で、この条件では熱エネルギーより十分小さい。
    """
    from es_sim.gpic import GpuPicSimulation

    contrast = {}
    for patch in (False, True):
        sim = GpuPicSimulation(_thermal_box(patch))
        assert (sim.amr is not None) == patch
        sim.run_batch(store_frames=False)
        ne = np.asarray(sim.fields["n_e"])
        x, y = sim.mesh.nodes[:, 0], sim.mesh.nodes[:, 1]
        inside = (np.abs(x - 0.01) < 0.002) & (np.abs(y - 0.01) < 0.002)
        outside = (np.abs(x - 0.01) > 0.005) | (np.abs(y - 0.01) > 0.005)
        contrast[patch] = ne[inside].mean() / ne[outside].mean()
        assert ne[outside].mean() == pytest.approx(1e15, rel=0.01)       # 規格化 (周期の従属節点を二重計上しない)
    assert abs(contrast[True] - contrast[False]) < 0.005


@needs_cuda
def test_amr_pic_ccp_matches_fine_uniform_grid():
    """CCP ストリップ (RF・MCC): 電極近傍だけ細かい AMR が、全体を細かくした一様格子と一致。"""
    from scipy.interpolate import LinearNDInterpolator

    from es_sim.gpic import GpuPicSimulation
    from es_sim.pic1d_presets import edupic_ar_processes

    e_procs, i_procs = edupic_ar_processes()
    L, H = 0.025, 0.002

    def project(h, amr=None):
        d = {
            "geometry": {
                "domain": {"polygon": [[0, 0], [L, 0], [L, H], [0, H]]},
                "boundaries": [
                    {"edges": [3], "type": "dirichlet", "voltage": 0.0,
                     "voltage_rf": {"amplitude": 150.0, "freq_hz": 13.56e6, "phase_deg": 0.0}},
                    {"edges": [1], "type": "dirichlet", "voltage": 0.0},
                ],
            },
            "mesh": {"size": h, "mode": "cartesian"},
            "pic": {
                "initial_plasma": {"density": 5e14, "te_ev": 2.0, "ti_ev": 0.026, "ion_mass_amu": 39.948, "seed": 1},
                "n_macro": 40000, "dt": 1.0 / (13.56e6 * 400), "n_steps": 4000, "avg_steps": 2000,
                "frame_every": 100000, "phase_bins": 0, "reflect_edges": [0, 2],
                "mcc": {"gas": {"name": "Ar", "pressure_pa": 10.0, "temperature_k": 300.0},
                        "electron_processes": [q.model_dump() for q in e_procs],
                        "ion_processes": [q.model_dump() for q in i_procs], "seed": 3},
            },
        }
        if amr:
            d["mesh"]["amr"] = amr
        return Project.model_validate(d)

    xs = np.linspace(0.0005, L - 0.0005, 49)
    prof = {}
    for name, p in (("fine", project(L / 100)),
                    ("amr", project(L / 50, {"max_level": 1, "refine_boundaries": False, "blocking_factor": 4,
                                             "regions": [{"p1": [0, 0], "p2": [0.005, H], "level": 1},
                                                         {"p1": [0.02, 0], "p2": [L, H], "level": 1}]}))):
        sim = GpuPicSimulation(p)
        assert (sim.amr is not None) == (name == "amr")
        sim.run_batch(store_frames=False)
        prof[name] = {k: LinearNDInterpolator(sim.mesh.nodes, np.asarray(sim.fields[k]))(xs, np.full(xs.size, H / 2))
                      for k in ("n_e", "n_i")}
    r = prof["amr"]["n_e"] / prof["fine"]["n_e"]
    assert np.max(np.abs(r - 1.0)) < 0.06, np.round(r, 3)
    # イオンは遅く時間平均でも統計揺らぎが大きいので 7 点ずつ平均して比べる
    ni = {k: prof[k]["n_i"].reshape(7, 7).mean(axis=1) for k in prof}
    r = ni["amr"] / ni["fine"]
    assert np.max(np.abs(r - 1.0)) < 0.06, np.round(r, 3)


@needs_cuda
def test_uniform_periodic_pic_display_density_at_slave_nodes():
    """回帰: 一様格子の周期境界で、従属節点 (右端・上端) の表示密度が主節点と同じ (0 にならない)。"""
    from es_sim.gpic import GpuPicSimulation

    p = _thermal_box(False)
    p.pic.n_steps = 200
    p.pic.avg_steps = 100
    p.pic.n_macro = 50000
    sim = GpuPicSimulation(p)
    sim.run_batch(store_frames=False)
    ne = np.asarray(sim.fields["n_e"])
    x = sim.mesh.nodes[:, 0]
    right = np.isclose(x, 0.02)
    left = np.isclose(x, 0.0)
    assert np.all(ne[right] > 0.0)
    assert np.allclose(ne[right], ne[left])
