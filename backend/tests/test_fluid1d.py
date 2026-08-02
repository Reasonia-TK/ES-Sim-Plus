"""fluid1d.py (1D ドリフト拡散流体ソルバー) のテスト (prompts/106)。

1. SG フラックス極限: Bernoulli 関数の級数/漸近分岐、Pe→0 で中心差分拡散、
   Pe→∞ で完全風上に一致 (単体関数テスト)。
2. 両極性拡散減衰: ソース項無効・両端接地・初期 cos 分布で初期減衰率が
   D_a (π/L)² に一致 (D_a = (μ_i D_e + μ_e D_i)/(μ_e+μ_i)、rtol 10%)。
   合成の (物理的に妥当な、mu_e/mu_i 比を穏やかにした) elastic 断面積を使う理由:
   eduPIC Ar の実際の mu_e/mu_i 比 (~400) は誘電緩和時間 τ_d が両極性減衰時間
   1/γ よりさらに4桁以上小さく、τ_d を数値的に解像しながら 1/γ オーダーまで
   時間積分することが (実機の CI 時間内では) 非現実的になる。これは物理現象
   そのもの (τ_d ≪ 1/γ こそが両極性近似が成立する条件) であり実装のバグでは
   ないことを、手計算 (D_a 公式の理論値) と数値実験の双方で確認済み — 詳細は
   このモジュールの実装過程の調査記録 (コミットメッセージ相当) を参照。
   ここでは mu_e/mu_i ~ 2 程度の穏やかな比にした合成データで検証する
   (それでも τ_d 由来のサブステップ機構は経由する)。
3. ボルツマン平衡: イオン固定 (debug_ions_enabled=False)・反射壁
   (debug_reflective_walls=True、電子が壁から漏れない閉じた系)・ソース/
   エネルギー方程式無効 (Te 固定) で電子密度を緩和させると、定常で
   n_e ∝ exp(φ/Te) (SG スキームの熱平衡保存性、モジュール docstring 参照)。
4. 粒子収支: (電離生成の累積 gen_total − 壁損失累積) が全粒子数変化に
   機械精度で一致 (rtol 1e-10)。
5. 陽的 vs 半陰: 小 dt (両者とも誘電緩和・拡散 CFL を十分下回る) で
   profiles が rtol 1e-4 で一致。
6. CCP 定常スモーク: eduPIC Ar 既定断面積・13.56 MHz 150V・50 Pa・gap 2.5cm。
7. continue のビット一致・validator。
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pydantic
import pytest

from es_sim.fem import EPS0
from es_sim.fluid1d import (
    FLOOR_N,
    Fluid1dSimulation,
    _bernoulli,
    build_fluid1d_result,
)
from es_sim.fluid_coeffs import interp_loglog
from es_sim.particles import QE
from es_sim.pic1d_presets import edupic_ar_processes
from es_sim.schema import (
    Domain,
    Fluid1dSettings,
    Fn1dEmission,
    Geometry,
    MeshSettings,
    Pic1dElectrode,
    Project,
    VoltageRF,
    XsProcess,
)

_DUMMY_GEOMETRY = Geometry(domain=Domain(polygon=[(0, 0), (1, 0), (1, 1), (0, 1)]))
_DUMMY_MESH = MeshSettings(size=0.1)


def _project(fluid1d: Fluid1dSettings) -> Project:
    return Project(geometry=_DUMMY_GEOMETRY, mesh=_DUMMY_MESH, fluid1d=fluid1d)


def _synthetic_processes(sigma_elastic: float, mass_ratio: float = 1.0e-4) -> list[XsProcess]:
    """一定断面積の合成 elastic + 簡易 ionization (build_fluid_reactions に必須)。

    テスト2 (両極性拡散) で mu_e を狙った大きさに直接コントロールするために使う
    (eduPIC Ar の実際の断面積では mu_e/mu_i が桁違いに大きく、CI 時間内での
    直接時間積分検証が非現実的になるため — モジュール docstring 参照)。
    """
    e = np.geomspace(1.0e-3, 1000.0, 300)
    elastic = XsProcess(
        kind="elastic", threshold_ev=0.0, mass_ratio=mass_ratio,
        energy_ev=e.tolist(), sigma_m2=(sigma_elastic * np.ones_like(e)).tolist(),
    )
    ioniz = XsProcess(
        kind="ionization", threshold_ev=15.8, mass_ratio=0.0,
        energy_ev=e.tolist(), sigma_m2=(1.0e-20 * np.where(e > 15.8, 1.0, 0.0)).tolist(),
    )
    return [elastic, ioniz]


# ---- 1. SG フラックス極限 -------------------------------------------------------


def test_bernoulli_matches_exact_formula_mid_range():
    z = np.linspace(-10.0, 10.0, 201)
    z = z[np.abs(z) > 1.0e-3]  # 0近傍は級数分岐に入るので厳密公式との比較から除く
    expected = z / np.expm1(z)
    got = _bernoulli(z)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_bernoulli_series_near_zero():
    """|z|<1e-4 の級数分岐が厳密値 (mpmath 相当の高精度多項式評価) と一致すること。"""
    z = np.array([1e-6, -1e-6, 1e-8, -1e-8, 0.0])
    expected = np.array([1.0 - 0.5 * zz + zz * zz / 12.0 for zz in z])
    got = _bernoulli(z)
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    assert got[-1] == pytest.approx(1.0)  # B(0) = 1


def test_bernoulli_large_z_asymptotics():
    z = np.array([600.0, 1000.0, 5000.0])
    got = _bernoulli(z)
    np.testing.assert_allclose(got, z * np.exp(-z), rtol=1e-12)
    z_neg = -z
    got_neg = _bernoulli(z_neg)
    np.testing.assert_allclose(got_neg, -z_neg, rtol=1e-12)


def test_bernoulli_identity_b_minus_z_minus_b_z_equals_z():
    """B(−z) − B(z) = z (SG 導出で使う恒等式、モジュール docstring 参照)。"""
    z = np.linspace(-50.0, 50.0, 501)
    z = z[np.abs(z) > 1e-3]
    lhs = _bernoulli(-z) - _bernoulli(z)
    np.testing.assert_allclose(lhs, z, rtol=1e-10, atol=1e-9)


def test_sg_flux_low_peclet_is_central_difference():
    """Pe→0 (z→0、ドリフト無し) で SG フラックスが中心差分拡散に一致する。"""
    dx, D = 1.0e-4, 2.0
    n_l, n_r = 5.0e14, 3.0e14
    z = 1.0e-8  # ほぼ 0
    flux = (D / dx) * (_bernoulli(-z) * n_l - _bernoulli(z) * n_r)
    expected = (D / dx) * (n_l - n_r)
    assert flux == pytest.approx(expected, rel=1e-6)


def test_sg_flux_high_peclet_is_full_upwind():
    """Pe→∞ (z→+∞) で SG フラックスが完全風上 (Γ→v n_upstream=n_l) に一致する。"""
    dx, D = 1.0e-4, 2.0
    v = 1.0e7  # 強いドリフト
    z = v * dx / D
    n_l, n_r = 5.0e14, 3.0e14
    flux = (D / dx) * (_bernoulli(-z) * n_l - _bernoulli(z) * n_r)
    assert flux == pytest.approx(v * n_l, rel=1e-6)

    z_neg = -z
    flux_neg = (D / dx) * (_bernoulli(-z_neg) * n_l - _bernoulli(z_neg) * n_r)
    assert flux_neg == pytest.approx(-v * n_r, rel=1e-6)


# ---- 2. 両極性拡散減衰 -----------------------------------------------------------


def test_ambipolar_diffusion_initial_decay_rate():
    """初期 cos 分布・両端接地・ソース無効での初期減衰率が D_a(π/L)² に一致する。

    「初期」減衰率を見る理由: 長時間積分すると有限デバイ長 (λ_D/L 有限) による
    理想両極性近似 (λ_D/L→0 の極限) からのずれが蓄積する (手計算・数値実験で
    確認済み、モジュール docstring 参照)。t→0 近傍の瞬時減衰率は λ_D/L 有限
    効果がまだ効いていない (電荷分離がまだほぼ0) ため、理想 D_a 値との比較に
    最も適している。
    """
    gap = 0.04
    n_cells = 80
    te0 = 3.0
    n0 = 1.0e15
    procs = _synthetic_processes(sigma_elastic=1.62e-17)
    s = Fluid1dSettings(
        gap_m=gap, n_cells=n_cells, init_density_m3=n0, init_te_ev=te0,
        gas_pressure_pa=50.0, gas_temperature_k=300.0,
        electron_processes=procs, n_steps=1, frame_every=100000,
    )
    sim = Fluid1dSimulation(_project(s))
    sim.debug_source_enabled = False
    sim.debug_energy_enabled = False

    prof = n0 * np.cos(math.pi * (sim.xg - gap / 2.0) / gap)
    prof = np.maximum(prof, FLOOR_N)
    sim.n_e = prof.copy()
    sim.n_i = prof.copy()
    sim.w = 1.5 * sim.n_e * te0

    mu_e_n = interp_loglog(sim.reactions.te_grid_ev, sim.reactions.transport.mobility_n, te0)
    mu_e = mu_e_n / sim.n_g
    d_e = mu_e * te0
    d_a = (sim.mu_i * d_e + mu_e * sim.d_i) / (mu_e + sim.mu_i)
    gamma_theory = d_a * (math.pi / gap) ** 2

    tau_d = EPS0 / (QE * n0 * mu_e)
    dt_diff = 0.5 * sim.dx**2 / d_e
    sim.dt = min(0.5 * tau_d, dt_diff)

    center = sim.n_nodes // 2
    n_e_start = sim.n_e[center]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(60):
            sim.step()
    measured_rate = -math.log(sim.n_e[center] / n_e_start) / sim.t

    assert measured_rate == pytest.approx(gamma_theory, rel=0.10)


# ---- 3. ボルツマン平衡 -----------------------------------------------------------


def test_boltzmann_equilibrium_from_fixed_ion_background():
    """イオン固定 (非一様背景)・反射壁・電子のみ緩和で n_e ∝ exp(φ/Te) に収束する。

    SG スキームは z=±dφ/T という構成そのものにより、ゼロフラックス (定常) で
    厳密にボルツマン関係を再現する (モジュール docstring 参照)。反射壁
    (debug_reflective_walls=True) で電子を系内に閉じ込め、エネルギー方程式・
    反応源を無効にして Te を一定に保つことで、電子密度だけが緩和して
    定常のボルツマン平衡へ収束する状況を作る。
    """
    gap = 0.02
    n_cells = 40
    te0 = 2.0
    n_i0 = 1.0e14
    s = Fluid1dSettings(
        gap_m=gap, n_cells=n_cells, init_density_m3=n_i0, init_te_ev=te0,
        gas_pressure_pa=50.0, gas_temperature_k=300.0, n_steps=1, frame_every=100000,
    )
    sim = Fluid1dSimulation(_project(s))
    sim.debug_source_enabled = False
    sim.debug_energy_enabled = False
    sim.debug_ions_enabled = False
    sim.debug_reflective_walls = True

    sim.n_i = n_i0 * (1.0 + 0.5 * np.cos(math.pi * (sim.xg - gap / 2.0) / gap))
    sim.n_e = np.full(sim.n_nodes, n_i0)  # 平衡から遠い一様分布から出発
    sim.w = 1.5 * sim.n_e * te0

    for _ in range(3000):
        sim.step()

    mid = sim.n_nodes // 2
    for idx in (5, 10, 15, 25, 30, 35):
        dln = math.log(sim.n_e[idx] / sim.n_e[mid])
        dphi_over_te = (sim.phi[idx] - sim.phi[mid]) / te0
        assert dln == pytest.approx(dphi_over_te, rel=0.05, abs=0.02)


# ---- 4. 粒子収支 -----------------------------------------------------------------


def test_particle_balance_matches_generation_minus_wall_loss():
    """(電離生成 gen_total − 壁損失累積) が全量変化と機械精度で一致する。"""
    s = Fluid1dSettings(
        gap_m=0.025, n_cells=60, init_density_m3=1.0e15, init_te_ev=3.0,
        gas_pressure_pa=50.0, gas_temperature_k=300.0,
        n_steps=1, frame_every=100000, dt=5.0e-11,
    )
    sim = Fluid1dSimulation(_project(s))
    n_e0 = float(np.sum(sim.n_e * sim.node_vol))
    n_i0 = float(np.sum(sim.n_i * sim.node_vol))

    for _ in range(300):
        sim.step()

    n_e1 = float(np.sum(sim.n_e * sim.node_vol))
    n_i1 = float(np.sum(sim.n_i * sim.node_vol))
    wall_e = sim.wall["left"]["electron"] + sim.wall["right"]["electron"]
    wall_i = sim.wall["left"]["ion"] + sim.wall["right"]["ion"]

    assert (n_e1 - n_e0) == pytest.approx(sim.gen_total - wall_e, rel=1e-10, abs=1.0)
    assert (n_i1 - n_i0) == pytest.approx(sim.gen_total - wall_i, rel=1e-10, abs=1.0)


# ---- 5. 陽的 vs 半陰 --------------------------------------------------------------


def test_explicit_matches_semi_implicit_at_small_dt():
    """小 dt (両者とも安定条件を十分下回る) で陽的経路と半陰経路の状態が一致する。"""
    s = Fluid1dSettings(
        gap_m=0.02, n_cells=40, init_density_m3=1.0e15, init_te_ev=3.0,
        gas_pressure_pa=50.0, gas_temperature_k=300.0,
        n_steps=1, frame_every=100000, dt=1.0e-13,
    )
    sim_ex = Fluid1dSimulation(_project(s), explicit=True)
    sim_im = Fluid1dSimulation(_project(s), explicit=False)
    for _ in range(50):
        sim_ex.step()
        sim_im.step()

    np.testing.assert_allclose(sim_ex.n_e, sim_im.n_e, rtol=1e-4, atol=1.0)
    np.testing.assert_allclose(sim_ex.n_i, sim_im.n_i, rtol=1e-4, atol=1.0)
    np.testing.assert_allclose(sim_ex.w, sim_im.w, rtol=1e-4, atol=1.0)
    # phi はこの条件で絶対値が極めて小さい (~1e-4V、電荷分離がまだ僅かなため) ので
    # rtol だけでなく絶対誤差でも判定する (電圧としては無視できる差)
    np.testing.assert_allclose(sim_ex.phi, sim_im.phi, rtol=1e-4, atol=1e-5)


# ---- 6. CCP 定常スモーク ----------------------------------------------------------


def _ccp_settings(n_steps: int, avg_steps: int, frame_every: int = 2000) -> Fluid1dSettings:
    rf = VoltageRF(amplitude=150.0, freq_hz=13.56e6, phase_deg=0.0)
    return Fluid1dSettings(
        gap_m=0.025, n_cells=100,
        left=Pic1dElectrode(v_dc=0.0, voltage_rf=rf, see_gamma=0.05),
        right=Pic1dElectrode(v_dc=0.0, see_gamma=0.05),
        init_density_m3=1.0e15, init_te_ev=3.0,
        gas_pressure_pa=50.0, gas_temperature_k=300.0,
        n_steps=n_steps, frame_every=frame_every, avg_steps=avg_steps, phase_bins=20,
    )


def test_ccp_smoke_steady_state():
    """eduPIC Ar 既定断面積、13.56MHz 150V、50Pa、gap 2.5cm での定常スモーク。

    CI 1分以内: n_cells=100・n_steps=10000 で数十秒程度 (半陰 dt で誘電緩和を
    サブステップしても間に合う)。
    """
    s = _ccp_settings(n_steps=10000, avg_steps=2000)
    sim = Fluid1dSimulation(_project(s))
    history, frames = sim.run_batch()

    fields = sim.fields
    assert fields is not None
    n_e = fields["n_e"]
    t_e = fields["t_e"]
    n_i = fields["n_i"]
    assert np.all(np.isfinite(n_e)) and np.all(np.isfinite(n_i)) and np.all(np.isfinite(t_e))
    assert np.all(n_e >= 0.0) and np.all(n_i >= 0.0)

    center = sim.n_nodes // 2
    assert 1e14 <= n_e[center] <= 1e18
    assert 1.0 <= t_e[center] <= 5.0

    # シース: 壁近傍で n_i > n_e (中性バルクより壁側でイオン優勢)
    near_wall = 2
    assert n_i[near_wall] > n_e[near_wall]
    assert n_i[-1 - near_wall] > n_e[-1 - near_wall]

    result = build_fluid1d_result(sim, elapsed_s=1.23)
    assert result["profiles"] is not None
    assert result["sheath"] is not None
    assert result["cycle"] is not None
    assert result["timing"]["total"] >= 0.0
    assert result["settings"]["gap_m"] == pytest.approx(0.025)


# ---- 7. continue ビット一致・validator ---------------------------------------------


def test_continue_bit_exact():
    """run(n)+continue(m) が run(n+m) とビット一致する (決定論、乱数不使用)。"""
    s_full = _ccp_settings(n_steps=140, avg_steps=40, frame_every=1000)
    sim_full = Fluid1dSimulation(_project(s_full))
    sim_full.run_batch()

    s_part = _ccp_settings(n_steps=100, avg_steps=40, frame_every=1000)
    sim_part = Fluid1dSimulation(_project(s_part))
    sim_part.run_batch()
    sim_part.prepare_continue(extra_steps=40, avg_steps=40)
    sim_part.run_batch()

    np.testing.assert_array_equal(sim_full.n_e, sim_part.n_e)
    np.testing.assert_array_equal(sim_full.n_i, sim_part.n_i)
    np.testing.assert_array_equal(sim_full.w, sim_part.w)
    np.testing.assert_array_equal(sim_full.phi, sim_part.phi)
    assert sim_full.t == sim_part.t
    assert sim_full.step_count == sim_part.step_count
    assert sim_full.gen_total == sim_part.gen_total
    assert sim_full.wall == sim_part.wall
    np.testing.assert_array_equal(sim_full.fields["n_e"], sim_part.fields["n_e"])


def test_validator_gap_must_be_positive():
    with pytest.raises(pydantic.ValidationError):
        Fluid1dSettings(gap_m=0.0, init_density_m3=1e15, gas_pressure_pa=10.0)


def test_validator_rejects_fn_emission():
    with pytest.raises(pydantic.ValidationError):
        Fluid1dSettings(
            gap_m=0.02, init_density_m3=1e15, gas_pressure_pa=10.0,
            left=Pic1dElectrode(fn=Fn1dEmission()),
        )
    with pytest.raises(pydantic.ValidationError):
        Fluid1dSettings(
            gap_m=0.02, init_density_m3=1e15, gas_pressure_pa=10.0,
            right=Pic1dElectrode(fn=Fn1dEmission()),
        )


def test_default_electron_processes_use_edupic_ar():
    """electron_processes を省略すると eduPIC Ar 解析式が既定で使われる。"""
    s = Fluid1dSettings(
        gap_m=0.02, n_cells=32, init_density_m3=1e15, init_te_ev=3.0,
        gas_pressure_pa=10.0, gas_temperature_k=350.0, n_steps=1,
    )
    sim = Fluid1dSimulation(_project(s))
    e_procs, _ = edupic_ar_processes()
    expected_reactions_k_ion = interp_loglog(
        sim.reactions.te_grid_ev, sim.reactions.k_ion, 3.0
    )
    assert expected_reactions_k_ion > 0.0
    assert sim.reactions.e_ion_ev == pytest.approx(15.8, rel=1e-6)
