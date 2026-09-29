"""fluid2d.py (2D/軸対称 EAFE/FEM-SG 流体ソルバー) のテスト (prompts/111)。

1. 1D 突き合わせ (最重要): 細長い矩形 (上下 symmetry) の xy メッシュで 2D 流体を
   実行し、断面平均プロファイル (n_e/φ/T_e) が同条件の fluid1d と一致する
   (SG/BC/積分の 2D 実装が 1D と整合することの強い検証)。n_e/φ は全域で rtol
   10%、T_e はシースエッジ (n_e が小さくシース比 (2/3)w/n_e のノイズが増幅
   されやすい領域) を除いた「バルク」領域 (n_e > 0.2·max(n_e)) で rtol 10% と
   する — シースエッジの T_e は分母 n_e が両ソルバーで独立に決まる離散化誤差の
   影響を強く受ける比なので、バルクでの一致 (メッシュ・時間積分の実装が正しい
   ことの本質的な検証) と切り分ける (手計算・対話的スクリプトで確認済み、
   下記コメント参照)。
2. ボルツマン平衡 (2D): イオン固定 (非一様、x・y 両方に依存する 2D プロファイル)・
   反射壁・ソース/エネルギー方程式無効で電子密度を緩和させると n_e ∝ exp(φ/Te)
   に収束する (SG の熱平衡保存性は 1D と同じ導出で 2D の各エッジにも成り立つ
   ため、非構造メッシュ上でも厳密に近い精度で成立する — 実測は machine
   precision 近くまで一致することを対話的スクリプトで確認済み)。
3. 粒子収支: 体積積分 (電離生成 gen_total − 壁損失) が全量変化と機械精度で
   一致する (EAFE の反対称フラックス F_ji=−F_ij + 集中質量による構成的な保存)。
4. rz 軸対称スモーク: 軸対称ドメインで実行し、全量有限・非負・対称軸近傍でも
   発散しないことを確認する。
5. 陽的 vs 半陰: 小 dt (両者とも安定条件を十分下回る) で一致する。
6. validator: periodic 境界を拒否すること。

## 発見した実装バグの記録 (対話的スクリプトでの調査、まずここで確認してから
## 上記テストを書いた、prompts/111 の指示どおり)

初期実装は CCP 的な RF 駆動条件で電子エネルギー方程式が発散する不安定
(局所的に T_e が floor まで落ち込み、隣接ノードで n_e が異常増大するが
全電子数は機械精度で保存される — 空間的な発振不安定) を示した。fluid1d の
同条件は安定だったため 2D 実装固有の不具合と特定し、Joule 加熱項
(_step_once のエッジ電力 p_edge) の符号が反転していた実装バグ (`+flux*dphi`
であるべきところが `-flux*dphi`ではなく `+flux*dphi` になっていた — 正しくは
-Γ_e・E = -flux_e_edge・dphi) を特定・修正した。修正後は同条件で fluid1d と
定性的に一致する滑らかな T_e 上昇・n_e 増加が得られることを確認済み
(/tmp の対話的スクリプトで検証、本ファイルのテストはこの修正後の状態を前提)。
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pydantic
import pytest

from es_sim.fluid2d import FLOOR_N, MAX_SUBSTEPS, Fluid2dSimulation, build_fluid2d_result
from es_sim.fluid1d import Fluid1dSimulation
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

# fluid2d.py は FLOOR_N/FLOOR_W を fluid1d.py から re-export しているだけなので
# import 元は fluid1d でも fluid2d でもよい (fluid2d.FLOOR_N が使えることの確認も兼ねる)


def _project2d(geometry: Geometry, mesh: MeshSettings, fluid2d: Fluid2dSettings, coord: str = "xy") -> Project:
    return Project(coord=coord, geometry=geometry, mesh=mesh, fluid2d=fluid2d)


def _project1d(fluid1d: Fluid1dSettings) -> Project:
    dummy_geo = Geometry(domain=Domain(polygon=[(0, 0), (1, 0), (1, 1), (0, 1)]))
    return Project(geometry=dummy_geo, mesh=MeshSettings(size=0.1), fluid1d=fluid1d)


# ---- 1. 1D 突き合わせ (最重要) ---------------------------------------------------


def test_matches_fluid1d_cross_section_average():
    """細長い矩形 (上下 symmetry) の 2D 流体と同条件の fluid1d の断面平均が一致する。"""
    gap = 0.02
    n_cells = 20
    h = gap / n_cells  # 正方セル (checkerboard 対角の非単調性を避ける、モジュール docstring 参照)
    rf = VoltageRF(amplitude=100.0, freq_hz=13.56e6, phase_deg=0.0)
    n_steps = 6000
    avg_steps = 1500
    n0 = 1.0e14
    p_gas = 20.0

    s1 = Fluid1dSettings(
        gap_m=gap, n_cells=n_cells,
        left=Pic1dElectrode(v_dc=0.0, voltage_rf=rf, see_gamma=0.05),
        right=Pic1dElectrode(v_dc=0.0, see_gamma=0.05),
        init_density_m3=n0, init_te_ev=2.0,
        gas_pressure_pa=p_gas, gas_temperature_k=300.0,
        n_steps=n_steps, frame_every=n_steps, avg_steps=avg_steps,
    )
    sim1 = Fluid1dSimulation(_project1d(s1))
    sim1.run_batch(store_frames=False)
    f1 = sim1.fields
    assert f1 is not None

    geo2 = Geometry(
        domain=Domain(polygon=[(0, 0), (gap, 0), (gap, h), (0, h)]),
        boundaries=[
            BoundaryCondition(edges=[3], type="dirichlet", voltage=0.0, voltage_rf=rf, see_gamma=0.05),
            BoundaryCondition(edges=[1], type="dirichlet", voltage=0.0, see_gamma=0.05),
            BoundaryCondition(edges=[0], type="symmetry"),
            BoundaryCondition(edges=[2], type="symmetry"),
        ],
    )
    s2 = Fluid2dSettings(
        init_density_m3=n0, init_te_ev=2.0,
        gas_pressure_pa=p_gas, gas_temperature_k=300.0,
        n_steps=n_steps, frame_every=n_steps, avg_steps=avg_steps,
    )
    sim2 = Fluid2dSimulation(_project2d(geo2, MeshSettings(size=h, mode="structured"), s2))
    sim2.run_batch(store_frames=False)
    f2 = sim2.fields
    assert f2 is not None

    nodes = sim2.mesh.nodes
    xs_all = nodes[:, 0]
    uniq_x = np.unique(np.round(xs_all, 12))

    def cross_avg(field: np.ndarray) -> np.ndarray:
        return np.array([field[np.isclose(xs_all, x)].mean() for x in uniq_x])

    ne2 = cross_avg(f2["n_e"])
    te2 = cross_avg(f2["t_e"])
    phi2 = cross_avg(f2["phi"])

    ne1 = np.interp(uniq_x, sim1.xg, f1["n_e"])
    te1 = np.interp(uniq_x, sim1.xg, f1["t_e"])
    phi1 = np.interp(uniq_x, sim1.xg, f1["phi"])

    def relerr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        denom = np.maximum(np.abs(a), np.abs(b))
        denom = np.where(denom < 1.0, 1.0, denom)
        return np.abs(a - b) / denom

    ne_err = relerr(ne1, ne2)
    assert ne_err.max() < 0.10, f"n_e 断面平均が fluid1d と rtol 10% を超えて乖離: {ne_err.max():.3f}"

    phi_range = float(phi1.max() - phi1.min())
    assert np.abs(phi1 - phi2).max() < 0.05 * phi_range

    # T_e はバルク (n_e が最大値の 20% 以上) だけで比較する (モジュール docstring の理由)
    bulk = ne1 > 0.2 * ne1.max()
    assert bulk.sum() >= 5
    te_err = relerr(te1[bulk], te2[bulk])
    assert te_err.max() < 0.10, f"T_e (バルク) が fluid1d と rtol 10% を超えて乖離: {te_err.max():.3f}"


# ---- 2. ボルツマン平衡 (2D) -------------------------------------------------------


def test_boltzmann_equilibrium_2d_nonuniform_ion_background():
    """イオン固定 (x・y 両方に依存する非一様プロファイル)・反射壁・電子のみ緩和で
    n_e ∝ exp(φ/Te) に収束する (SG の熱平衡保存性、モジュール docstring 参照)。
    """
    length = 0.02
    size = 0.002
    te0 = 2.0
    n_i0 = 1.0e14
    geo = Geometry(
        domain=Domain(polygon=[(0, 0), (length, 0), (length, length), (0, length)]),
        boundaries=[
            BoundaryCondition(edges=[0], type="dirichlet", voltage=0.0),
            BoundaryCondition(edges=[1], type="dirichlet", voltage=0.0),
            BoundaryCondition(edges=[2], type="dirichlet", voltage=0.0),
            BoundaryCondition(edges=[3], type="dirichlet", voltage=0.0),
        ],
    )
    s = Fluid2dSettings(
        init_density_m3=n_i0, init_te_ev=te0,
        gas_pressure_pa=50.0, gas_temperature_k=300.0, n_steps=1, frame_every=100000,
    )
    sim = Fluid2dSimulation(_project2d(geo, MeshSettings(size=size, mode="unstructured"), s))
    sim.debug_source_enabled = False
    sim.debug_energy_enabled = False
    sim.debug_ions_enabled = False
    sim.debug_reflective_walls = True

    nodes = sim.mesh.nodes
    x, y = nodes[:, 0], nodes[:, 1]
    sim.n_i = n_i0 * (1.0 + 0.5 * np.cos(math.pi * x / length) * np.cos(math.pi * y / length))
    sim.n_e = np.zeros(sim.n_nodes)
    sim.n_e[sim.active_idx] = n_i0
    sim.w = 1.5 * sim.n_e * te0

    for _ in range(1500):
        sim.step()

    active = sim.active_idx
    center = np.array([length / 2.0, length / 2.0])
    ref = active[np.argmin(np.linalg.norm(nodes[active] - center, axis=1))]

    rng = np.random.default_rng(0)
    sample = rng.choice(active, size=min(15, len(active)), replace=False)
    for idx in sample:
        if idx == ref:
            continue
        dln = math.log(sim.n_e[idx] / sim.n_e[ref])
        dphi_over_te = (sim.phi[idx] - sim.phi[ref]) / te0
        assert dln == pytest.approx(dphi_over_te, rel=0.05, abs=1.0e-2)


# ---- 3. 粒子収支 -----------------------------------------------------------------


def test_particle_balance_matches_generation_minus_wall_loss():
    """(電離生成 gen_total − 壁損失累積) が全量変化と機械精度で一致する (フル物理)。

    この保存則はテレスコーピング (F_ji=-F_ij + 集中質量) という「離散化そのものの
    構成的な性質」であり、線形方程式を厳密に (機械精度まで) 解いたときにしか
    machine precision では成立しない (prompts/115: 既定になった iterative
    (BiCGSTAB, rtol=1e-10) は方程式を近似的にしか解かないため、200ステップの
    累積で rel=1e-8 の許容誤差を超えてしまう — これは離散化の破綻ではなく
    ソルバーの近似誤差なので、このテストは「離散化が保存的であること」を厳密に
    検証する目的どおり linear_solver="direct" (spsolve、機械精度解) を明示する)。
    """
    length = 0.02
    size = 0.002
    geo = Geometry(
        domain=Domain(polygon=[(0, 0), (length, 0), (length, length), (0, length)]),
        boundaries=[
            BoundaryCondition(edges=[i], type="dirichlet", voltage=0.0, see_gamma=0.05) for i in range(4)
        ],
    )
    s = Fluid2dSettings(
        init_density_m3=1.0e15, init_te_ev=3.0,
        gas_pressure_pa=50.0, gas_temperature_k=300.0,
        n_steps=1, frame_every=100000, dt=5.0e-11,
        linear_solver="direct",
    )
    sim = Fluid2dSimulation(_project2d(geo, MeshSettings(size=size, mode="unstructured"), s))
    n_e0 = float(np.sum(sim.n_e[sim.active_idx] * sim.node_vol))
    n_i0 = float(np.sum(sim.n_i[sim.active_idx] * sim.node_vol))

    for _ in range(200):
        sim.step()

    n_e1 = float(np.sum(sim.n_e[sim.active_idx] * sim.node_vol))
    n_i1 = float(np.sum(sim.n_i[sim.active_idx] * sim.node_vol))

    assert (n_e1 - n_e0) == pytest.approx(sim.gen_total - sim.wall["electron"], rel=1e-8, abs=1.0)
    assert (n_i1 - n_i0) == pytest.approx(sim.gen_total - sim.wall["ion"], rel=1e-8, abs=1.0)


# ---- 4. rz 軸対称スモーク ----------------------------------------------------------


def test_rz_axisymmetric_smoke():
    """軸対称ドメイン (coord='rz') で実行し、全量有限・非負・対称軸近傍で発散しない。"""
    gap = 0.02   # 軸方向 (x)
    radius = 0.01  # 径方向 (y = r)
    size = 0.002
    rf = VoltageRF(amplitude=80.0, freq_hz=13.56e6, phase_deg=0.0)
    geo = Geometry(
        domain=Domain(polygon=[(0, 0), (gap, 0), (gap, radius), (0, radius)]),
        boundaries=[
            BoundaryCondition(edges=[0], type="symmetry"),  # y=0 (対称軸)
            BoundaryCondition(edges=[1], type="dirichlet", voltage=0.0, voltage_rf=rf, see_gamma=0.05),
            BoundaryCondition(edges=[2], type="dirichlet", voltage=0.0, see_gamma=0.05),
            BoundaryCondition(edges=[3], type="dirichlet", voltage=0.0, see_gamma=0.05),
        ],
    )
    s = Fluid2dSettings(
        init_density_m3=1.0e14, init_te_ev=2.0,
        gas_pressure_pa=30.0, gas_temperature_k=300.0,
        n_steps=2000, frame_every=2000, avg_steps=500,
    )
    sim = Fluid2dSimulation(_project2d(geo, MeshSettings(size=size, mode="unstructured"), s, coord="rz"))
    sim.run_batch()

    assert np.all(np.isfinite(sim.n_e)) and np.all(sim.n_e >= 0.0)
    assert np.all(np.isfinite(sim.n_i)) and np.all(sim.n_i >= 0.0)
    assert np.all(np.isfinite(sim.w)) and np.all(sim.w >= 0.0)
    assert np.all(np.isfinite(sim.phi))

    nodes = sim.mesh.nodes
    axis_idx = np.nonzero(np.isclose(nodes[:, 1], 0.0))[0]
    assert axis_idx.size > 0
    assert np.all(np.isfinite(sim.n_e[axis_idx]))
    assert np.all(sim.n_e[axis_idx] >= 0.0)

    result = build_fluid2d_result(sim, elapsed_s=1.0)
    assert result["fields"] is not None
    assert result["cycle"] is None  # phase_bins 未指定 (既定 0)


# ---- 5. 陽的 vs 半陰 --------------------------------------------------------------


def test_explicit_matches_semi_implicit_at_small_dt():
    """小 dt (両者とも安定条件を十分下回る) で陽的経路と半陰経路の状態が一致する。"""
    length = 0.01
    size = 0.001
    geo = Geometry(
        domain=Domain(polygon=[(0, 0), (length, 0), (length, length), (0, length)]),
        boundaries=[
            BoundaryCondition(edges=[i], type="dirichlet", voltage=0.0, see_gamma=0.05) for i in range(4)
        ],
    )
    s = Fluid2dSettings(
        init_density_m3=1.0e15, init_te_ev=3.0,
        gas_pressure_pa=50.0, gas_temperature_k=300.0,
        n_steps=1, frame_every=100000, dt=1.0e-13,
    )
    geo_mesh = MeshSettings(size=size, mode="unstructured")
    sim_ex = Fluid2dSimulation(_project2d(geo, geo_mesh, s), explicit=True)
    sim_im = Fluid2dSimulation(_project2d(geo, geo_mesh, s), explicit=False)
    for _ in range(50):
        sim_ex.step()
        sim_im.step()

    np.testing.assert_allclose(sim_ex.n_e, sim_im.n_e, rtol=1e-3, atol=1.0)
    np.testing.assert_allclose(sim_ex.n_i, sim_im.n_i, rtol=1e-3, atol=1.0)
    np.testing.assert_allclose(sim_ex.w, sim_im.w, rtol=1e-3, atol=1.0)
    np.testing.assert_allclose(sim_ex.phi, sim_im.phi, rtol=1e-3, atol=1e-5)


# ---- 6. validator ----------------------------------------------------------------


def test_validator_rejects_periodic_boundary():
    """periodic 境界は未対応 (ValueError、モジュール docstring の理由コメント参照)。"""
    length = 0.01
    geo = Geometry(
        domain=Domain(polygon=[(0, 0), (length, 0), (length, length), (0, length)]),
        boundaries=[
            BoundaryCondition(edges=[0, 2], type="periodic"),
            BoundaryCondition(edges=[1], type="dirichlet", voltage=0.0),
            BoundaryCondition(edges=[3], type="dirichlet", voltage=0.0),
        ],
    )
    s = Fluid2dSettings(init_density_m3=1e14, gas_pressure_pa=10.0, n_steps=1)
    with pytest.raises(ValueError, match="periodic"):
        Fluid2dSimulation(_project2d(geo, MeshSettings(size=0.001), s))


def test_validator_requires_fluid2d_settings():
    dummy_geo = Geometry(domain=Domain(polygon=[(0, 0), (1, 0), (1, 1), (0, 1)]))
    project = Project(geometry=dummy_geo, mesh=MeshSettings(size=0.1))
    with pytest.raises(ValueError, match="fluid2d"):
        Fluid2dSimulation(project)


def test_validator_gas_pressure_must_be_positive():
    with pytest.raises(pydantic.ValidationError):
        Fluid2dSettings(init_density_m3=1e15, gas_pressure_pa=0.0)


# ---- 7. 陰的反復ソルバー (numba 並列 Jacobi-BiCGSTAB、prompts/115) -------------------


def _small_solver_project(n_steps: int = 20, dt: float = 5.0e-11, **fluid_kwargs) -> Project:
    """反復ソルバーのテスト共通: 小規模メッシュ・少数ステップのフル物理プロジェクト
    (test_particle_balance_matches_generation_minus_wall_loss と同じ形状・条件)。
    """
    length = 0.01
    size = 0.002
    geo = Geometry(
        domain=Domain(polygon=[(0, 0), (length, 0), (length, length), (0, length)]),
        boundaries=[
            BoundaryCondition(edges=[i], type="dirichlet", voltage=0.0, see_gamma=0.05) for i in range(4)
        ],
    )
    kwargs: dict = dict(
        init_density_m3=1.0e15, init_te_ev=3.0,
        gas_pressure_pa=50.0, gas_temperature_k=300.0,
        n_steps=n_steps, frame_every=n_steps, avg_steps=max(1, n_steps // 4), dt=dt,
    )
    kwargs.update(fluid_kwargs)
    s = Fluid2dSettings(**kwargs)
    return _project2d(geo, MeshSettings(size=size, mode="unstructured"), s)


def test_iterative_matches_direct():
    """既定 (iterative) と "direct" (従来の spsolve) の結果が一致する。

    BiCGSTAB は既定 rtol=1e-10 まで収束させるため、direct (SuperLU の直接解) との
    差は反復法の丸め誤差程度に収まるはずで、ビット一致は要求せず rtol 1e-6 とする
    (BiCGSTAB と SuperLU は演算順序が全く異なるアルゴリズムなのでビット一致は
    そもそも成立しない — これは「行並列 matvec のスレッド数依存性」の話とは別物)。
    """
    n_steps = 30
    sim_it = Fluid2dSimulation(_small_solver_project(n_steps=n_steps, linear_solver="iterative"))
    sim_dr = Fluid2dSimulation(_small_solver_project(n_steps=n_steps, linear_solver="direct"))
    sim_it.run_batch(store_frames=False)
    sim_dr.run_batch(store_frames=False)

    np.testing.assert_allclose(sim_it.n_e, sim_dr.n_e, rtol=1e-6, atol=1.0)
    np.testing.assert_allclose(sim_it.n_i, sim_dr.n_i, rtol=1e-6, atol=1.0)
    np.testing.assert_allclose(sim_it.w, sim_dr.w, rtol=1e-6, atol=1.0)
    np.testing.assert_allclose(sim_it.phi, sim_dr.phi, rtol=1e-6, atol=1e-6)
    assert sim_it.solver_fallback_count == 0
    assert not any("フォールバック" in w for w in sim_it.warnings)


def test_iterative_threads_bit_identical():
    """threads=1 と threads=2 で run_batch 結果が完全にビット同一 (np.array_equal)。

    行並列 matvec (_numba_kernels.csr_matvec_parallel) は行ごとに独立な逐次積和
    (どのスレッドがどの行を処理しても、その行自身の浮動小数点演算順序は不変)
    であり、BiCGSTAB の内積・ノルムは常に numpy 単一スレッドで計算するため、
    反復列全体・最終的な密度/エネルギー/電位はスレッド数に一切依存しないはず
    ── これを allclose ではなく array_equal で厳密に検証する (規約どおり)。
    """
    n_steps = 30
    sim1 = Fluid2dSimulation(_small_solver_project(n_steps=n_steps, linear_solver="iterative", threads=1))
    sim2 = Fluid2dSimulation(_small_solver_project(n_steps=n_steps, linear_solver="iterative", threads=2))
    sim1.run_batch(store_frames=False)
    sim2.run_batch(store_frames=False)

    assert np.array_equal(sim1.n_e, sim2.n_e)
    assert np.array_equal(sim1.n_i, sim2.n_i)
    assert np.array_equal(sim1.w, sim2.w)
    assert np.array_equal(sim1.phi, sim2.phi)


def test_iterative_convergence_counter_and_no_warnings():
    """通常ケースで solver_iters (総反復数) > 0、フォールバック関連 warnings は出ない。"""
    sim = Fluid2dSimulation(_small_solver_project(n_steps=20, linear_solver="iterative"))
    sim.run_batch(store_frames=False)
    assert sim.timing["solver_iters"] > 0
    assert sim.solver_fallback_count == 0
    assert not any("フォールバック" in w for w in sim.warnings)


def test_iterative_falls_back_to_direct_when_not_converged():
    """反復上限をテスト用フックで0に強制すると BiCGSTAB が必ず未収束になり、
    spsolve (direct) へフォールバックする。結果は依然として有限のまま
    (堅牢性: フォールバックが機能していれば発散しない)。
    """
    sim = Fluid2dSimulation(_small_solver_project(n_steps=5, linear_solver="iterative"))
    sim._solver_max_iter = 0  # テスト専用フック (モジュール docstring 参照)
    sim.run_batch(store_frames=False)

    assert sim.solver_fallback_count > 0
    assert any("フォールバック" in w for w in sim.warnings)
    assert np.all(np.isfinite(sim.n_e))
    assert np.all(np.isfinite(sim.n_i))
    assert np.all(np.isfinite(sim.w))


# ---- 8. サブステップ数の上限・サブステップ途中の停止要求 ---------------------------------
#
# 解が暴走すると (UI 既定のサンプル形状で実測: 電位 kV 超・Te 数百 eV 超) 必要なサブステップ数が
# 際限なく増え、step() が終わらず停止ボタンも効かなかった。ここでは dt を安定条件の目安
# (この条件で ~1e-10 s) より大きく取り、同じ経路を決定論的に再現する。

_STATE_KEYS = ("n_e", "n_i", "w", "phi")


def test_substep_limit_raises_value_error_without_changing_state():
    """必要なサブステップ数が上限を超えたら ValueError (発散の兆候)。状態は変えない。"""
    sim = Fluid2dSimulation(_small_solver_project(n_steps=5, dt=1.0e-6))  # 必要数 ~6e3
    before = {k: getattr(sim, k).copy() for k in _STATE_KEYS}
    with pytest.raises(ValueError, match="サブステップ数") as exc:
        sim.run_batch(store_frames=False)
    assert f"上限 {MAX_SUBSTEPS}" in str(exc.value)
    assert sim.step_count == 0 and sim.t == 0.0 and sim.history["step"] == []
    for k, arr in before.items():
        assert np.array_equal(getattr(sim, k), arr), k

    # 上限は _max_substeps (テスト用フック) で決まる: 6〜7 回に刻む条件で上限 5 なら超過
    sim2 = Fluid2dSimulation(_small_solver_project(n_steps=5, dt=1.0e-9))
    sim2._max_substeps = 5
    with pytest.raises(ValueError, match="上限 5 "):
        sim2.step()


def test_stop_request_between_substeps_rolls_back_the_step():
    """サブステップの合間の停止要求で状態をステップ開始時に戻し、None を返す。

    戻した状態から続けた結果は、中断しなかった実行とビット一致する (停止 → 続きからの整合)。
    """
    dt = 1.0e-9  # 安定条件の目安の約 10 倍 → 1 ステップを 6〜7 回のサブステップに刻む
    sim = Fluid2dSimulation(_small_solver_project(n_steps=5, dt=dt))
    ref = Fluid2dSimulation(_small_solver_project(n_steps=5, dt=dt))
    sim.step()
    ref.step()
    before = {k: getattr(sim, k).copy() for k in _STATE_KEYS}
    wall0, gen0 = dict(sim.wall), sim.gen_total

    calls: list[float] = []
    step_once = sim._step_once

    def counting_step_once(dt_sub, t, implicit):
        calls.append(t)
        return step_once(dt_sub, t, implicit)

    sim._step_once = counting_step_once
    checks: list[int] = []

    def should_stop() -> bool:
        checks.append(len(calls))
        return len(calls) >= 2  # 2 サブステップ進んだところで停止要求

    assert sim.step(should_stop) is None
    assert checks == [1, 2]  # 1 回目のサブステップの前には確認しない (run_batch がステップ前に確認する)
    for k, arr in before.items():
        assert np.array_equal(getattr(sim, k), arr), k
    assert sim.wall == wall0 and sim.gen_total == gen0
    assert sim.step_count == 1 and sim.t == dt and len(sim.history["step"]) == 1

    sim.step()
    ref.step()
    for k in _STATE_KEYS:
        assert np.array_equal(getattr(sim, k), getattr(ref, k)), k
    assert sim.wall == ref.wall and sim.gen_total == ref.gen_total


def test_run_batch_stop_request_takes_effect_within_a_step():
    """run_batch の should_stop はステップ間だけでなくサブステップの合間にも効く。

    途中で止めたステップは history・フレームに残さない (step_count も進めない)。
    """
    sim = Fluid2dSimulation(_small_solver_project(n_steps=5, dt=1.0e-9))
    n_checks = 0

    def should_stop() -> bool:
        nonlocal n_checks
        n_checks += 1
        return n_checks >= 3  # 1 回目はステップ前 (run_batch)、2 回目以降がサブステップの合間

    history, _ = sim.run_batch(should_stop=should_stop, store_frames=False)
    assert n_checks == 3
    assert sim.step_count == 0 and history["step"] == []
    assert sim.fields is None


# ---- 9. Joule 加熱の緩和時間 τ_J (サブステップ分割の上限の一つ) ---------------------------


def _uniform_field_tau_ratio(size: float) -> float:
    """一様プラズマ (n_e=n_i)・一様電場 E=V0/L (上下 symmetry の帯) での τ_J / 解析値。

    解析値は τ_J = w / (Joule 加熱密度) = (3/2)n_e Te / (n_e μ_e E²) = (3/2)Te / (μ_e E²)
    (一様密度では SG フラックスが純ドリフト −μ_e n_e E になる)。
    """
    length, height, v0, te = 0.02, 0.004, 50.0, 3.0
    geo = Geometry(
        domain=Domain(polygon=[(0, 0), (length, 0), (length, height), (0, height)]),
        boundaries=[
            BoundaryCondition(edges=[3], type="dirichlet", voltage=0.0),
            BoundaryCondition(edges=[1], type="dirichlet", voltage=v0),
            BoundaryCondition(edges=[0, 2], type="symmetry"),
        ],
    )
    s = Fluid2dSettings(init_density_m3=1.0e15, init_te_ev=te, gas_pressure_pa=50.0, n_steps=1)
    sim = Fluid2dSimulation(_project2d(geo, MeshSettings(size=size, mode="structured"), s))
    sim.phi = sim._solve_phi(0.0)  # 空間電荷 0 なので φ は電極間で線形
    te0, mu_e0, *_ = sim._te_and_coeffs(sim.n_e[sim.active_idx], sim.w[sim.active_idx])
    expected = 1.5 * te / (float(mu_e0[0]) * (v0 / length) ** 2)
    return sim._joule_relaxation_time(te0, mu_e0) / expected


def test_joule_relaxation_time_is_intensive_and_matches_uniform_field():
    """τ_J は節点体積に依らず、一様電場の解析値 (3/2)Te/(μ_e E²) と O(1) で一致する。

    以前は加熱率 (節点へ配分した外延量) を節点体積で割っておらず、τ_J が 1/V_i 倍
    (ここでは 1e6 倍前後) 過大で、メッシュを細かくするほど大きくなっていた (上限が実質無効)。
    最小を取るのは領域の角の節点: 三角形 1 枚だけに属し、P1 の集中体積 (1/6 セル) が
    エッジの重みに対応する制御体積 (1/4 セル) より小さく加熱密度が 1.5 倍になるため、
    構造格子では解析値の 2/3 になる (その節点の w の式にとってはこれが正しい時間スケール)。
    """
    r_coarse = _uniform_field_tau_ratio(0.002)
    r_fine = _uniform_field_tau_ratio(0.0005)  # 節点体積は 1/16
    assert r_fine == pytest.approx(r_coarse, rel=1e-9)
    assert 0.5 < r_coarse <= 1.0 + 1e-9


def test_dc_cathode_sheath_does_not_trip_substep_limit():
    """健全な DC 放電 (100×50 mm、0 V/100 V、上下 symmetry、既定条件) でサブステップが突発的に増えない。

    陰極シースのほぼ空の節点 (w が下限値付近) が τ_J の最小値を決めていたため、節点体積を
    割る修正の後は step 124 前後で 1 ステップ 1 億回超を要求し、サブステップ上限で誤って
    ValueError になっていた (修正前の τ_J でも step 101 で 4,228 回の突発的な刻み)。
    """
    geo = Geometry(
        domain=Domain(polygon=[(0, 0), (0.1, 0), (0.1, 0.05), (0, 0.05)]),
        boundaries=[
            BoundaryCondition(edges=[3], type="dirichlet", voltage=0.0),
            BoundaryCondition(edges=[1], type="dirichlet", voltage=100.0),
            BoundaryCondition(edges=[0, 2], type="symmetry"),
        ],
    )
    s = Fluid2dSettings(init_density_m3=1.0e15, gas_pressure_pa=50.0, n_steps=200, frame_every=200)
    sim = Fluid2dSimulation(_project2d(geo, MeshSettings(size=0.002, mode="unstructured"), s))
    n_sub: list[int] = []
    step_once = sim._step_once

    def counting_step_once(*args, **kwargs):
        n_sub[-1] += 1
        return step_once(*args, **kwargs)

    sim._step_once = counting_step_once
    for _ in range(200):
        n_sub.append(0)
        sim.step()
    assert max(n_sub) <= 5
    assert np.all(np.isfinite(sim.phi)) and float(np.max(sim.phi)) < 110.0
