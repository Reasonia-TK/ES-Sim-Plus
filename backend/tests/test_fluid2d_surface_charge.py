"""流体 2D の誘電体表面の帯電と反射壁 (Neumann 外周) のテスト (prompts/129)。

v1 (fluid2d.Fluid2dSimulation、三角形メッシュ)・v2 (gfluid.CartesianFluid2dSimulation、直交格子 + EB)・
その AMR 版 (gfluid.amr.AmrFluid2dSimulation) で確かめる (GPU 版は tests/test_v2_gfluid_gpu.py が CPU 版と
突き合わせる)。

1. 境界条件なし (Neumann) の外周辺は反射壁: symmetry を指定した場合とビット一致する。
2. 表面電荷の Poisson 右辺 (v1 は FEM の 2π 規約、v2 は 2π を落とした EB の右辺): 層状の誘電体を
   挟んだ平行平板で、界面の一様な面電荷 σ の電位が解析解 σ/(ε0/d1 + ε_r ε0/d2) に一致する
   (xy・rz)。
3. 電荷保存: 電極を誘電体で覆い、壁が全て帯電する系で e(N_i − N_e) + Σq_surf が保存する
   (フル物理・SEE 込み)。
4. 浮遊電位: 接地電極と「接地の裏電極を持つ誘電体」で挟んだ対称なプラズマでは、誘電体表面は
   浮遊して電極と同じ 0 V になり、両側とも接地電極の場合と同じ解になる。表面電荷は隣の半分の
   空間電荷と釣り合う (ガウスの法則)。
5. UI の既定サンプル (100×50 mm・誘電体ブロック・右辺 100 V) で電位が暴走しない (DC・上下
   symmetry・RF)。修正前はサブステップ数が際限なく増え、電位が kV まで上がっていた。
6. サブステップの合間の停止要求は表面電荷もステップ開始時へ戻し、続きは中断なしとビット一致する。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from es_sim.fem import EPS0
from es_sim.fluid2d import Fluid2dSimulation
from es_sim.gfluid import CartesianFluid2dSimulation
from es_sim.gfluid.amr import AmrFluid2dSimulation
from es_sim.particles import QE
from es_sim.schema import Project

RF = {"amplitude": 150.0, "freq_hz": 13.56e6}
#: mode="amr" の既定: 固体の境界の近傍を 1 段細分化 (gfluid.amr、prompts/128)
AMR_NEAR_SOLIDS = {"max_level": 1, "buffer_cells": 2}


def _project(w, h, regions=(), boundaries=(), coord="xy", size=0.5e-3, mode="structured", fluid=None,
             amr=None) -> Project:
    """mode: "structured"・"unstructured" (v1)、"cartesian" (v2)、"amr" (v2 の AMR 版、amr で細分化を指定)。"""
    f = {"init_density_m3": 1e15, "init_te_ev": 2.0, "gas_pressure_pa": 50.0, "n_steps": 1,
         "frame_every": 10**9}
    f.update(fluid or {})
    mesh = {"size": size, "mode": "cartesian" if mode == "amr" else mode}
    if mode == "amr":
        mesh["amr"] = amr or AMR_NEAR_SOLIDS
    return Project.model_validate({
        "coord": coord,
        "geometry": {"domain": {"polygon": [[0, 0], [w, 0], [w, h], [0, h]]},
                     "boundaries": list(boundaries), "regions": list(regions)},
        "mesh": mesh,
        "fluid2d": f,
    })


def _make(project: Project):
    if project.mesh.mode == "cartesian":
        cls = AmrFluid2dSimulation if project.mesh.amr is not None else CartesianFluid2dSimulation
        return cls(project, device="cpu")
    return Fluid2dSimulation(project)


def _slab(x0, x1, h, eps_r=4.0, see_gamma=0.0, rid="diel"):
    return {"id": rid, "type": "dielectric", "eps_r": eps_r, "see_gamma": see_gamma,
            "polygon": [[x0, 0.0], [x1, 0.0], [x1, h], [x0, h]]}


def _charge(sim) -> float:
    """プラズマの正味の電荷 e(N_i − N_e) [C] (xy は奥行き 1 m あたり、rz は 2π 込み)。"""
    act = sim.active_idx
    return QE * float(np.sum((sim.n_i[act] - sim.n_e[act]) * sim.node_vol))


def _run(sim, n_steps):
    for _ in range(n_steps):
        sim.step()
        assert np.all(np.isfinite(sim.phi)) and np.all(np.isfinite(sim.n_e))


# ---- 1. Neumann の外周辺は反射壁 --------------------------------------------------------


@pytest.mark.parametrize("mode", ["structured", "unstructured", "cartesian", "amr"])
def test_neumann_edges_reflect_like_symmetry(mode):
    """上下の辺を境界条件なし (Neumann) にした結果が symmetry 指定とビット一致する (壁にならない)。"""
    w, h = 0.01, 0.003
    electrodes = [{"edges": [3], "type": "dirichlet", "voltage": 0.0, "voltage_rf": RF, "see_gamma": 0.05},
                  {"edges": [1], "type": "dirichlet", "voltage": 0.0}]
    fluid = {"init_density_m3": 5e14, "gas_pressure_pa": 30.0}
    amr = {"max_level": 1, "regions": [{"p1": [0.004, 0.0], "p2": [0.006, 0.003], "level": 1}]}
    none = _make(_project(w, h, boundaries=electrodes, mode=mode, fluid=fluid, amr=amr))
    sym = _make(_project(w, h, boundaries=electrodes + [{"edges": [0, 2], "type": "symmetry"}], mode=mode,
                         fluid=fluid, amr=amr))
    assert len(none.wall_n1) == len(sym.wall_n1)
    _run(none, 60)
    _run(sym, 60)
    for k in ("n_e", "n_i", "w", "phi"):
        assert np.array_equal(getattr(none, k), getattr(sym, k)), k
    assert none.wall == sym.wall
    assert none.wall["electron"] > 0.0          # 電極の壁は吸収する


# ---- 2. 表面電荷の Poisson 右辺: 層状誘電体の平行平板 -----------------------------------------


@pytest.mark.parametrize("coord", ["xy", "rz"])
@pytest.mark.parametrize("mode", ["structured", "cartesian"])
def test_surface_charge_potential_matches_layered_capacitor(mode, coord):
    """気体層 d1 と誘電体層 d2 (ε_r) を接地電極で挟み、界面に一様な σ を置く: V = σ/(ε0/d1 + ε_r ε0/d2)。

    rz は軸方向の平行平板 (円板)。表面電荷は 2π 込みのリング電荷で持ち、Poisson では 2π で
    割る規約 — 解析解は r によらないので、2π の扱いを誤ると電位がその倍率だけずれる。
    界面は格子線上にあるので、区分線形の厳密解を離散化誤差なしに再現する。
    """
    d1, d2, h, eps_r, sigma = 0.006, 0.004, 0.002, 4.0, 3.0e-7
    bnd = [{"edges": [3], "type": "dirichlet", "voltage": 0.0}, {"edges": [1], "type": "dirichlet", "voltage": 0.0},
           {"edges": [0, 2], "type": "symmetry"}]
    sim = _make(_project(d1 + d2, h, [_slab(d1, d1 + d2, h, eps_r)], bnd, coord=coord, mode=mode))
    x = sim.mesh.nodes[:, 0]
    # 帯電する壁端は誘電体の表面 (x = d1) の節点だけで、面積の合計は界面の面積
    assert np.allclose(x[sim.surf_node], d1, atol=1e-12)
    area = 2.0 * math.pi * 0.5 * h * h if coord == "rz" else h
    assert float(sim.surf_area.sum()) == pytest.approx(area, rel=1e-12)

    sim.q_surf = sigma * np.bincount(sim.surf_node, weights=sim.surf_area, minlength=sim.n_nodes)
    phi = sim._solve_phi(0.0)                    # 初期状態は準中性 (n_e = n_i) なので空間電荷なし
    v_s = sigma / (EPS0 / d1 + eps_r * EPS0 / d2)
    exact = np.where(x <= d1, v_s * x / d1, v_s * (d1 + d2 - x) / d2)
    np.testing.assert_allclose(phi, exact, rtol=0.0, atol=1e-9 * v_s)


@pytest.mark.parametrize("mode", ["structured", "unstructured", "cartesian", "amr"])
def test_surface_charge_sits_on_the_wall_flux_node(mode):
    """表面電荷は、その壁の流束を評価する輸送節点 (壁コンダクタンスを配分する節点) に置かれる。

    壁の流束条件は壁の密度をその節点の密度で代用するので、浮遊 (Γ_e = Γ_i) にはその節点の
    電位が下がる必要がある。電荷を固体側の節点に置くと、シースを解像しない粗い格子で誘電体の
    内部が大きく負になる (gfluid.geometry のモジュール docstring、既定サンプルの DC で −59 V)。
    格子に揃わない円形・斜辺の誘電体でも成り立つことを確かめる。
    """
    w, h = 0.01, 0.006
    bnd = [{"edges": [3], "type": "dirichlet", "voltage": 0.0}, {"edges": [1], "type": "dirichlet", "voltage": 0.0}]
    regions = [{"id": "c", "type": "dielectric", "eps_r": 3.0,
                "shape": {"kind": "circle", "center": [0.0037, 0.0031], "radius": 0.0017}},
               {"id": "t", "type": "dielectric", "eps_r": 5.0,
                "polygon": [[0.0061, 0.0012], [0.0087, 0.0019], [0.0072, 0.0047]]}]
    sim = _make(_project(w, h, regions, bnd, mode=mode))
    assert sim.surf_elem.size > 0
    end1 = sim.surf_loc == sim.wall_n1[sim.surf_elem]
    end2 = sim.surf_loc == sim.wall_n2[sim.surf_elem]
    assert np.all(end1 | end2)
    np.testing.assert_array_equal(sim.surf_node, sim.active_idx[sim.surf_loc])
    # 誘電体の壁の面積 (両端に配分した境界質量の和) は全て帯電する (電極に接していない)
    diel_area = float(sim.surf_area.sum())
    wall_area = float(np.sum(sim.wall_w1) + np.sum(sim.wall_w2))
    x = sim.mesh.nodes[:, 0]
    electrode_area = wall_area - diel_area
    assert electrode_area == pytest.approx(2 * h, rel=1e-9)            # 残りは左右の電極だけ
    assert np.all((x[sim.surf_node] > 0.0) & (x[sim.surf_node] < w))


# ---- 3. 電荷保存 (電極を誘電体で覆った系) ---------------------------------------------------


@pytest.mark.parametrize("mode", ["structured", "unstructured", "cartesian", "amr"])
def test_charge_conservation_with_dielectric_covered_electrodes(mode):
    """両電極を誘電体で覆う (壁は全て帯電する誘電体表面) と、プラズマから出た電荷は全て表面に残る:
    e(N_i − N_e) + Σq_surf = 0 (初期は準中性)、Σq_surf = e(壁のイオン − 電子)。SEE・RF・電離込み。
    """
    w, h, d = 0.012, 0.002, 0.001
    bnd = [{"edges": [3], "type": "dirichlet", "voltage": 0.0, "voltage_rf": RF},
           {"edges": [1], "type": "dirichlet", "voltage": 0.0}]           # 上下は Neumann (反射)
    regions = [_slab(0.0, d, h, see_gamma=0.1, rid="d1"), _slab(w - d, w, h, see_gamma=0.1, rid="d2")]
    sim = _make(_project(w, h, regions, bnd, mode=mode, fluid={"init_te_ev": 3.0, "linear_solver": "direct"}))
    assert sim.surf_elem.size > 0
    assert len(sim.surf_elem) == len(sim.wall_n1) * (1 if mode in ("cartesian", "amr") else 2)   # 全ての壁端が帯電
    q0 = _charge(sim)
    _run(sim, 300)

    q_plasma = _charge(sim) - q0
    q_surf = float(sim.q_surf.sum())
    moved = QE * sim.wall["ion"]
    assert sim.wall["electron"] > 0.0 and sim.wall["ion"] > 0.0 and q_surf < 0.0
    assert q_surf == pytest.approx(QE * (sim.wall["ion"] - sim.wall["electron"]), rel=1e-12)
    assert abs(q_plasma + q_surf) < 1e-10 * moved
    assert sim.history["surf_q"][-1] == q_surf
    # 表面電荷は誘電体表面 (x = d, w − d) の節点だけに載る
    x = sim.mesh.nodes[:, 0]
    on = np.nonzero(sim.q_surf)[0]
    assert np.all(np.isclose(x[on], d) | np.isclose(x[on], w - d))


# ---- 4. 浮遊電位 --------------------------------------------------------------------------


def _floating_pair(mode, coord, n_steps):
    """(浮遊誘電体の系, 両側接地電極の系) を同条件で n_steps 進めて返す。

    プラズマは x∈[0, L]。浮遊側は x∈[L, L+d] が誘電体で、その裏 (x = L+d) が接地電極。
    電離・エネルギー方程式を止めた (Te 一定) 減衰プラズマで、両壁の粒子の境界条件は同じなので、
    誘電体表面は浮遊して電極と同じ 0 V になる (対称な解。両壁とも正味電流 0)。
    """
    L, d, h = 0.01, 0.002, 0.001
    size = 0.25e-3 if mode == "cartesian" else 0.5e-3
    sides = [{"edges": [3], "type": "dirichlet", "voltage": 0.0}, {"edges": [1], "type": "dirichlet", "voltage": 0.0},
             {"edges": [0, 2], "type": "symmetry"}]
    fluid = {"init_density_m3": 1e15, "init_te_ev": 2.0, "gas_pressure_pa": 50.0}
    flo = _make(_project(L + d, h, [_slab(L, L + d, h)], sides, coord=coord, size=size, mode=mode, fluid=fluid))
    ref = _make(_project(L, h, [], sides, coord=coord, size=size, mode=mode, fluid=fluid))
    for sim in (flo, ref):
        sim.debug_source_enabled = False
        sim.debug_energy_enabled = False
        _run(sim, n_steps)
    return flo, ref, L


def _half_charge(sim, L) -> float:
    """プラズマの右半分 (x > L/2、中央の節点は半分) の正味電荷 [C] — Poisson の右辺と同じ重み。"""
    x = sim.mesh.nodes[sim.active_idx, 0]
    wgt = np.where(np.isclose(x, 0.5 * L), 0.5, (x > 0.5 * L).astype(float))
    act = sim.active_idx
    return QE * float(np.sum(wgt * (sim.n_i[act] - sim.n_e[act]) * sim.node_vol))


def _column_avg(sim, field):
    """輸送節点の x ごとの断面平均 (節点体積重み)。

    v1 の構造格子は P1 集中質量が市松に並ぶので解が y 方向にわずかな構造を持ち、浮遊した表面の
    電位はそれに節点ごとに追従する (電極は一様に 0 V。イオンは冷たい (T_i = 0.026 eV) ので表面に
    沿った ±0.03 V 程度の差でも壁の節点の n_i が数 % 入れ替わる)。比べるのは断面平均。
    """
    act = sim.active_idx
    xs, inv = np.unique(np.round(sim.mesh.nodes[act, 0], 9), return_inverse=True)
    v = sim.node_vol
    return xs, np.bincount(inv, weights=field[act] * v) / np.bincount(inv, weights=v)


@pytest.mark.parametrize("coord", ["xy", "rz"])
@pytest.mark.parametrize("mode", ["structured", "cartesian"])
def test_floating_dielectric_matches_grounded_electrode_by_symmetry(mode, coord):
    """接地電極 | プラズマ | 浮遊誘電体 (裏は接地) は、対称性から両側接地電極の場合と同じ解になる。

    表面電荷の符号・大きさ・置き場所 (rz の 2π を含む) のどれかを誤ると、過渡中に左右が非対称に
    なって一致しない。帯電しなければ誘電体は電子を吸い続け、電位は正へ暴走する。

    v2 (直交格子) は離散的にも対称で、実測 1e-7 で一致する (許容 1e-5)。v1 の構造格子は P1 集中
    質量の市松模様のため浮遊した表面の電位が節点ごとに ±0.03 V ほど揺れ (_column_avg)、断面平均で
    φ 1e-3・n_i 8e-3 の差になる (許容は φ 3e-3・n 1.5e-2)。
    """
    flo, ref, L = _floating_pair(mode, coord, 3000)
    te = 2.0
    tol_phi, tol_n, tol_q = (1e-5, 1e-5, 1e-5) if mode == "cartesian" else (3e-3, 1.5e-2, 5e-3)
    x = flo.mesh.nodes[:, 0]
    assert np.allclose(x[flo.surf_node], L)

    xs_f, phi_f = _column_avg(flo, flo.phi)
    xs_r, phi_r = _column_avg(ref, ref.phi)
    np.testing.assert_allclose(xs_f, xs_r, atol=1e-12)
    phi_max = float(phi_r.max())
    assert phi_max > 3.0 * te                                # プラズマ電位 (壁より数 Te 高い)
    assert abs(phi_f[-1]) < tol_phi * phi_max, phi_f[-1]     # 誘電体表面は電極と同じ 0 V に浮く
    assert np.max(np.abs(phi_f - phi_r)) < tol_phi * phi_max
    for k in ("n_e", "n_i"):
        _, a = _column_avg(flo, getattr(flo, k))
        _, b = _column_avg(ref, getattr(ref, k))
        assert np.max(np.abs(a - b)) < tol_n * np.max(b), k

    # 浮遊した表面は接地電極と同じだけの正味電荷を集める (壁全体の e(Γ_i − Γ_e) の半分)
    q_s = float(flo.q_surf.sum())
    assert q_s < 0.0
    assert q_s == pytest.approx(0.5 * QE * (flo.wall["ion"] - flo.wall["electron"]), rel=tol_q)
    assert q_s == pytest.approx(0.5 * QE * (ref.wall["ion"] - ref.wall["electron"]), rel=tol_q)
    # ガウスの法則: 誘電体の中は電場 0 (表面 = 裏電極 = 0 V)、中央面も電場 0 (対称) なので、
    # 表面電荷は右半分の空間電荷と打ち消し合う
    assert q_s == pytest.approx(-_half_charge(flo, L), rel=min(tol_q, 1e-3))


# ---- 5. UI の既定サンプル -------------------------------------------------------------------


def _default_sample(mode, variant) -> Project:
    """フロントの SAMPLE (App.tsx、examples/parallel_plates.json) + Fluid2dPanel の既定値。"""
    bnd = [{"edges": [3], "type": "dirichlet", "voltage": 0.0},
           {"edges": [1], "type": "dirichlet", "voltage": 100.0}]
    if variant == "rf":
        bnd[1] = {"edges": [1], "type": "dirichlet", "voltage": 0.0,
                  "voltage_rf": {"amplitude": 100.0, "freq_hz": 13.56e6}}
    if variant == "symmetry":
        bnd.append({"edges": [0, 2], "type": "symmetry"})
    diel = {"id": "diel1", "type": "dielectric", "eps_r": 4.0,
            "polygon": [[0.04, 0.01], [0.06, 0.01], [0.06, 0.04], [0.04, 0.04]]}
    return _project(0.1, 0.05, [diel], bnd, size=0.004, mode=mode,
                    fluid={"init_density_m3": 1e15, "init_te_ev": 2.0, "gas_pressure_pa": 50.0})


@pytest.mark.parametrize("variant", ["dc", "symmetry", "rf"])
@pytest.mark.parametrize("mode", ["unstructured", "cartesian"])
def test_default_sample_stays_bounded(mode, variant):
    """既定サンプルで電位が電極電圧 + 数 Te の範囲に留まり、サブステップ数も増えない。

    修正前は誘電体表面 (と Neumann の上下辺) が帯電せずに電子だけを吸い、電位が kV まで
    暴走してサブステップ数が際限なく増えた (1 ステップが終わらない)。上限を超えたら打ち切る。
    DC の下限 −1 V (陰極 0 V) は、表面電荷の置き場所の誤りで誘電体の内部が負に落ち込むのも
    捕まえる (固体側の節点に置くと −59 V まで下がった、test_surface_charge_sits_on_the_wall_flux_node)。
    陰極シースの電子がほぼ空の節点で Joule 加熱の安定条件が縮む問題 (誘電体の有無によらない) は
    τ_J の最小値からその節点を除いて解消済み (0fa8c6c)。
    """
    sim = _make(_default_sample(mode, variant))
    n_steps, max_sub = 1500, 10
    calls = [0]
    step_once = sim._step_once

    def counted(*args, **kwargs):
        calls[0] += 1
        if calls[0] > max_sub * n_steps:
            raise AssertionError(f"サブステップ数が {max_sub}/ステップを超えた (step {sim.step_count + 1}、"
                                 f"φ {sim.phi.min():.4g}〜{sim.phi.max():.4g} V)")
        return step_once(*args, **kwargs)

    sim._step_once = counted
    lo, hi = (-110.0, 130.0) if variant == "rf" else (-1.0, 130.0)   # 電極は 0 V と 100 V (RF は ±100 V)
    for _ in range(n_steps):
        sim.step()
        assert lo < sim.phi.min() and sim.phi.max() < hi, (sim.step_count, sim.phi.min(), sim.phi.max())
    assert np.all(np.isfinite(sim.n_e)) and np.all(np.isfinite(sim.w))
    assert float(sim.q_surf.sum()) < 0.0    # 誘電体ブロックは負に帯電する


# ---- 6. サブステップの合間の停止 ---------------------------------------------------------------


@pytest.mark.parametrize("mode", ["structured", "cartesian", "amr"])
def test_stop_between_substeps_rolls_back_surface_charge(mode):
    """停止要求で状態をステップ開始時に戻すとき、誘電体の表面電荷も戻す (戻さないと e(N_i − N_e) +
    Σq_surf がずれる)。続きは中断しなかった実行とビット一致する。"""
    bnd = [{"edges": [3], "type": "dirichlet", "voltage": 0.0}, {"edges": [1], "type": "dirichlet", "voltage": 0.0}]
    fluid = {"init_density_m3": 1e15, "init_te_ev": 3.0, "dt": 1.0e-9}     # 1 ステップが数サブステップ
    proj = _project(0.01, 0.004, [_slab(0.004, 0.006, 0.002)], bnd, mode=mode, fluid=fluid)
    sim, ref = _make(proj.model_copy(deep=True)), _make(proj.model_copy(deep=True))
    sim.step()
    ref.step()
    keys = ("n_e", "n_i", "w", "phi", "q_surf")
    before = {k: getattr(sim, k).copy() for k in keys}
    assert np.any(before["q_surf"] != 0.0)
    calls = []
    step_once = sim._step_once

    def counting(dt_sub, t, implicit):
        calls.append(t)
        return step_once(dt_sub, t, implicit)

    sim._step_once = counting
    assert sim.step(lambda: len(calls) >= 2) is None
    for k, arr in before.items():
        assert np.array_equal(getattr(sim, k), arr), k
    sim.step()
    ref.step()
    for k in keys:
        assert np.array_equal(getattr(sim, k), getattr(ref, k)), k
    assert sim.history["surf_q"] == ref.history["surf_q"]
