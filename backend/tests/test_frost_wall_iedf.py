"""修正 Frost イオン移動度 + 壁 IEDF (1D PIC / 1D 流体) のテスト (prompts/116)。

Part 1 (修正 Frost 移動度、fluid1d.py/fluid2d.py 共通):
  1. frost_mobility: E=0 で低電界値と厳密に一致 (ビット一致)、300 Td 相当で
     μ_L/√(1+300/150)=μ_L/√3 に一致 (単体関数)。
  2. Fluid1dSimulation._mu_i_at / Fluid2dSimulation._mu_i_at: E=0 で self.mu_i と
     厳密に一致すること (SG 係数へ配線される直前の値としての確認)。
  3. ion_mobility_model="const" は E に関わらず低電界値のまま (frost 導入前と
     ビット不変であることの直接確認)。
  4. CCP スモーク (fluid1d): frost (既定) でも収束・有限、かつ壁近傍の強電界で
     実際に μ_i が低電界値より低下していること。

Part 2a (1D PIC 壁 IEDF、pic1d.py、粒子ベース):
  1. wall_iedf_bins=0 は無効 (None)。
  2. DC 強バイアス (無衝突・左電極 -100V) で左壁 IEDF の平均エネルギーが
     印加電位差 100 eV に近い値になり (rtol 10%)、重み和が同じ区間の壁吸収
     重みと一致する。

Part 2b (1D 流体 壁 IEDF、fluid1d.py、無衝突シース近似のモデルベース):
  1. フィルタ・ヒストグラムの核となる純関数 (sheath_voltage_filter,
     _weighted_energy_histogram) を合成信号で単体テストし、τ_i≪T_rf (瞬時追従・
     二山) と τ_i≫T_rf (時間平均・単峰) の遷移を分散の定量比較で検証する。
  2. RF なし (DC) の縮退が単峰 (1サンプル) で例外を出さないこと。
  3. CCP スモークで両壁とも有限・重み正であること。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from es_sim.fluid1d import (
    Fluid1dSimulation,
    _weighted_energy_histogram,
    build_fluid1d_result,
    frost_mobility,
    sheath_voltage_filter,
)
from es_sim.fluid2d import Fluid2dSimulation
from es_sim.pic1d import Pic1dSimulation, build_pic1d_result, electrode_voltage
from es_sim.schema import (
    Domain,
    Fluid1dSettings,
    Fluid2dSettings,
    Geometry,
    MeshSettings,
    Pic1dElectrode,
    Pic1dSettings,
    Project,
    VoltageRF,
)

_DUMMY_GEOMETRY = Geometry(domain=Domain(polygon=[(0, 0), (1, 0), (1, 1), (0, 1)]))
_DUMMY_MESH = MeshSettings(size=0.1)


def _project1d(fluid1d: Fluid1dSettings | None = None, pic1d: Pic1dSettings | None = None) -> Project:
    return Project(geometry=_DUMMY_GEOMETRY, mesh=_DUMMY_MESH, fluid1d=fluid1d, pic1d=pic1d)


def _project2d(fluid2d: Fluid2dSettings) -> Project:
    geo = Geometry(domain=Domain(polygon=[(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01)]))
    return Project(geometry=geo, mesh=MeshSettings(size=0.002), fluid2d=fluid2d)


# ==== Part 1: 修正 Frost イオン移動度 ==============================================


def test_frost_mobility_zero_field_matches_low_field_value_bit_exact():
    """E=0 では μ_i(E/N)=μ_L/√(1+0/C)=μ_L (割り算が1.0による恒等演算なのでビット一致)。"""
    mu_l = 0.145
    got = frost_mobility(mu_l, 0.0, n_g=3.22e22, c_td=150.0)
    assert got == mu_l


def test_frost_mobility_300td_matches_formula():
    """一様強電界 (300 Td 相当) で μ_i=μ_L/√(1+300/150)=μ_L/√3 に一致 (単体関数)。"""
    mu_l = 0.2
    n_g = 3.22e22
    c_td = 150.0
    e_abs = 300.0 * n_g * 1.0e-21  # 300 Td → |E| [V/m]
    got = frost_mobility(mu_l, e_abs, n_g, c_td)
    expected = mu_l / math.sqrt(1.0 + 300.0 / 150.0)
    assert got == pytest.approx(expected, rel=1e-12)
    assert got == pytest.approx(mu_l / math.sqrt(3.0), rel=1e-12)


def test_fluid1d_mu_i_at_zero_field_matches_low_field_and_300td_formula():
    s = Fluid1dSettings(
        gap_m=0.02, n_cells=16, init_density_m3=1e15, gas_pressure_pa=50.0,
        gas_temperature_k=300.0, n_steps=1,
    )
    sim = Fluid1dSimulation(_project1d(fluid1d=s))
    assert sim._mu_i_at(0.0) == sim.mu_i

    e_abs = 300.0 * sim.n_g * 1.0e-21
    expected = sim.mu_i / math.sqrt(1.0 + 300.0 / 150.0)
    assert sim._mu_i_at(e_abs) == pytest.approx(expected, rel=1e-12)


def test_fluid1d_const_model_ignores_field_bit_exact():
    """ion_mobility_model='const' は E に関わらず低電界値のまま (frost 導入前とビット不変)。"""
    s = Fluid1dSettings(
        gap_m=0.02, n_cells=16, init_density_m3=1e15, gas_pressure_pa=50.0,
        gas_temperature_k=300.0, n_steps=1, ion_mobility_model="const",
    )
    sim = Fluid1dSimulation(_project1d(fluid1d=s))
    e_abs = 300.0 * sim.n_g * 1.0e-21
    assert sim._mu_i_at(e_abs) == sim.mu_i
    assert sim._mu_i_at(0.0) == sim.mu_i


def test_fluid1d_face_coeffs_frost_equals_const_at_zero_field():
    """E=0 (一様電位) では frost/const いずれのモデルでも SG 係数がビット一致する。"""
    common = dict(
        gap_m=0.02, n_cells=16, init_density_m3=1e15, gas_pressure_pa=50.0,
        gas_temperature_k=300.0, n_steps=1,
    )
    sim_f = Fluid1dSimulation(_project1d(fluid1d=Fluid1dSettings(ion_mobility_model="frost", **common)))
    sim_c = Fluid1dSimulation(_project1d(fluid1d=Fluid1dSettings(ion_mobility_model="const", **common)))
    phi = np.zeros(sim_f.n_nodes)
    te = np.full(sim_f.n_nodes, 3.0)
    mu_e = np.full(sim_f.n_nodes, 1.0)
    a_i_f, b_i_f, *_ = sim_f._face_coeffs(phi, te, mu_e)
    a_i_c, b_i_c, *_ = sim_c._face_coeffs(phi, te, mu_e)
    np.testing.assert_array_equal(a_i_f, a_i_c)
    np.testing.assert_array_equal(b_i_f, b_i_c)


def test_fluid2d_mu_i_at_zero_field_and_300td():
    s = Fluid2dSettings(init_density_m3=1e15, gas_pressure_pa=50.0, n_steps=1)
    sim = Fluid2dSimulation(_project2d(s))
    assert sim._mu_i_at(0.0) == sim.mu_i

    e_abs = 300.0 * sim.n_g * 1.0e-21
    expected = sim.mu_i / math.sqrt(1.0 + 300.0 / 150.0)
    assert sim._mu_i_at(e_abs) == pytest.approx(expected, rel=1e-12)

    s_const = Fluid2dSettings(init_density_m3=1e15, gas_pressure_pa=50.0, n_steps=1, ion_mobility_model="const")
    sim_c = Fluid2dSimulation(_project2d(s_const))
    assert sim_c._mu_i_at(e_abs) == sim_c.mu_i


def test_fluid1d_ccp_smoke_frost_finite_and_mobility_reduced_near_wall():
    """CCP スモーク (frost 既定) が収束・有限であり、壁近傍の強電界で実際に
    μ_i が低電界値より低下していること (Frost 補正が実際に働いている証拠)。
    """
    rf = VoltageRF(amplitude=150.0, freq_hz=13.56e6, phase_deg=0.0)
    s = Fluid1dSettings(
        gap_m=0.025, n_cells=100,
        left=Pic1dElectrode(v_dc=0.0, voltage_rf=rf, see_gamma=0.05),
        right=Pic1dElectrode(v_dc=0.0, see_gamma=0.05),
        init_density_m3=1.0e15, init_te_ev=3.0,
        gas_pressure_pa=50.0, gas_temperature_k=300.0,
        n_steps=10000, frame_every=2000, avg_steps=2000, phase_bins=20,
        ion_mobility_model="frost",
    )
    sim = Fluid1dSimulation(_project1d(fluid1d=s))
    sim.run_batch(store_frames=False)
    fields = sim.fields
    assert fields is not None
    assert np.all(np.isfinite(fields["n_e"])) and np.all(np.isfinite(fields["n_i"]))
    assert np.all(fields["n_e"] >= 0.0) and np.all(fields["n_i"] >= 0.0)

    # シース近傍 (強電界) で frost 補正が実際に働き μ_i が低電界値より小さい
    ex_wall = fields["e"][0]
    mu_wall = sim._mu_i_at(ex_wall)
    assert mu_wall < sim.mu_i


# ==== Part 2a: 1D PIC 壁 IEDF (粒子ベース) ==========================================


def test_pic1d_wall_iedf_disabled_when_bins_zero():
    s = Pic1dSettings(
        gap_m=0.01, n_cells=20, init_density_m3=1e14, n_macro=500, dt=2e-10,
        n_steps=50, frame_every=50, wall_iedf_bins=0,
    )
    sim = Pic1dSimulation(_project1d(pic1d=s))
    sim.run_batch()
    assert sim.wall_iedf is None
    result = build_pic1d_result(sim, 0.0)
    assert result["wall_iedf"] is None


def test_pic1d_wall_iedf_dc_bias_mean_energy_matches_sheath_potential():
    """DC 強バイアス (無衝突・左電極 -100V、右電極 0V) の左壁 IEDF。

    ionization/MCC 無しの単純な系では電子は自由拡散で速やかに失われるため、
    シース構造が確立し (電子がほぼ枯渇した) 十分後の時間窓で平均をとる。
    このとき左壁へ吸収されるイオンは (右寄りのバルク電位 ≈ 0V) から
    (左電極電位 -100V) までをほぼ無衝突で自由落下するので、平均エネルギーは
    印加電位差 100 eV に近い値になる (rtol 10%)。重み和はこの区間中に実際に
    左壁で吸収されたイオンの重み (self.wall["left"]["ion"] の増分) と一致する。
    """
    s = Pic1dSettings(
        gap_m=0.02, n_cells=64, init_density_m3=1.0e14, init_te_ev=2.0, init_ti_ev=0.05,
        n_macro=8000, n_steps=1, frame_every=100000, seed=3,
        left=Pic1dElectrode(v_dc=-100.0), right=Pic1dElectrode(v_dc=0.0),
        wall_iedf_bins=80,
    )
    sim = Pic1dSimulation(_project1d(pic1d=s))
    for _ in range(21000):
        sim.step()

    sim.enable_density_accum(sim.step_count + 1)
    wall_i_before = sim.wall["left"]["ion"]
    for _ in range(1500):
        sim.step()
    wall_i_after = sim.wall["left"]["ion"]

    sim.fields = sim.averaged_fields()
    sim.wall_iedf = sim.wall_iedf_data()
    assert sim.wall_iedf is not None
    left = sim.wall_iedf["left"]

    assert left["total_weight"] == pytest.approx(wall_i_after - wall_i_before, rel=1e-9)
    assert wall_i_after > wall_i_before  # 実際にこの区間でイオンが吸収されている
    assert left["mean_energy_ev"] == pytest.approx(100.0, rel=0.10)

    # electrode_voltage は fluid1d/pic1d 共通の電圧評価関数 (回帰確認を兼ねる)
    assert electrode_voltage(sim.s.left, sim.t) == pytest.approx(-100.0)


# ==== Part 2b: 1D 流体 壁 IEDF (無衝突シース近似、モデルベース) ========================


def test_sheath_voltage_filter_transit_time_bimodal_to_unimodal_transition():
    """合成 V_sh (正弦波) を τ_i の大小で二山→単峰に遷移させる (純関数の単体テスト)。

    τ_i≪T_rf (瞬時追従) では V_eff がほぼ V_sh そのものになりヒストグラムの分散が
    大きい (両端に二山)。τ_i≫T_rf (時間平均への収束) では V_eff がほぼ一定になり
    分散が小さい (単峰)。この分散の定量比較で遷移を検証する。
    """
    bins = 64
    t_rf = 1.0 / 13.56e6
    dt_bin = t_rf / bins
    phase = (np.arange(bins) + 0.5) / bins * 2.0 * np.pi
    v_sh = 100.0 + 80.0 * np.sin(phase)  # 常に正 (20V〜180V の範囲を往復)

    def _variance(tau_i: float) -> float:
        v_eff = sheath_voltage_filter(v_sh, dt_bin, tau_i, n_periods=50)
        hist = _weighted_energy_histogram(v_eff, np.ones(bins), bins=50)
        f = np.asarray(hist["f"])
        e_centers = np.asarray(hist["e_centers"])
        d_e = e_centers[1] - e_centers[0]
        return float(np.sum(f * (e_centers - hist["mean_energy_ev"]) ** 2) * d_e)

    var_instant = _variance(t_rf / 1000.0)  # τ_i≪T_rf: 瞬時追従・二山
    var_averaged = _variance(1000.0 * t_rf)  # τ_i≫T_rf: 時間平均に収束・単峰

    assert var_instant > 20.0 * var_averaged
    assert math.sqrt(var_averaged) < 0.05 * 80.0  # 単峰側: 入力振幅に対して分散が十分小さい
    assert math.sqrt(var_instant) > 0.3 * 80.0    # 二山側: 入力振幅と同程度に分散が大きい


def test_sheath_voltage_filter_small_tau_tracks_input_exactly():
    """τ_i→0 (瞬時追従) では V_eff は V_sh そのものに一致する。"""
    bins = 32
    dt_bin = 1.0e-9
    v_sh = 50.0 + 40.0 * np.sin((np.arange(bins) + 0.5) / bins * 2.0 * np.pi)
    v_eff = sheath_voltage_filter(v_sh, dt_bin, tau_i=dt_bin * 1e-6, n_periods=20)
    np.testing.assert_allclose(v_eff, v_sh, rtol=1e-6)


def test_sheath_voltage_filter_zero_tau_i_returns_input_unchanged():
    v_sh = np.array([10.0, 20.0, 30.0])
    got = sheath_voltage_filter(v_sh, dt_bin=1.0e-9, tau_i=0.0, n_periods=10)
    np.testing.assert_array_equal(got, v_sh)


def test_weighted_energy_histogram_normalization_and_zero_weight_fallback():
    e_ev = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    hist = _weighted_energy_histogram(e_ev, w, bins=20)
    d_e = hist["e_centers"][1] - hist["e_centers"][0]
    assert np.sum(hist["f"]) * d_e == pytest.approx(1.0, rel=1e-9)
    assert hist["total_weight"] == pytest.approx(4.0)

    empty = _weighted_energy_histogram(np.zeros(0), np.zeros(0), bins=10)
    assert empty["total_weight"] == 0.0
    assert np.all(empty["f"] == 0.0)
    assert empty["mean_energy_ev"] == 0.0


def test_fluid1d_wall_iedf_dc_degenerate_is_single_peak_no_exception():
    """RF なし (DC、phase_bins=0) の縮退で単峰 (1サンプル) になり、例外を出さない。"""
    s = Fluid1dSettings(
        gap_m=0.02, n_cells=40, init_density_m3=1e15, init_te_ev=3.0,
        gas_pressure_pa=50.0, gas_temperature_k=300.0,
        left=Pic1dElectrode(v_dc=-50.0), right=Pic1dElectrode(v_dc=0.0),
        n_steps=3000, frame_every=3000, avg_steps=1000, phase_bins=0,
        wall_iedf_bins=50,
    )
    sim = Fluid1dSimulation(_project1d(fluid1d=s))
    sim.run_batch(store_frames=False)
    assert sim.wall_iedf is not None
    left = sim.wall_iedf["left"]
    assert left["model"] == "collisionless_sheath"
    assert left["n_samples"] == 1
    assert left["total_weight"] > 0.0
    assert np.all(np.isfinite(left["f"]))


def test_fluid1d_wall_iedf_bins_zero_disabled():
    s = Fluid1dSettings(
        gap_m=0.02, n_cells=40, init_density_m3=1e15, gas_pressure_pa=50.0,
        n_steps=200, frame_every=200, wall_iedf_bins=0,
    )
    sim = Fluid1dSimulation(_project1d(fluid1d=s))
    sim.run_batch(store_frames=False)
    assert sim.wall_iedf is None
    result = build_fluid1d_result(sim, elapsed_s=0.1)
    assert result["wall_iedf"] is None


def test_fluid1d_wall_iedf_ccp_smoke_finite_and_positive_weight():
    """CCP スモークで両壁とも有限・重み正であること (無衝突シース近似)。"""
    rf = VoltageRF(amplitude=150.0, freq_hz=13.56e6, phase_deg=0.0)
    s = Fluid1dSettings(
        gap_m=0.025, n_cells=100,
        left=Pic1dElectrode(v_dc=0.0, voltage_rf=rf, see_gamma=0.05),
        right=Pic1dElectrode(v_dc=0.0, see_gamma=0.05),
        init_density_m3=1.0e15, init_te_ev=3.0,
        gas_pressure_pa=50.0, gas_temperature_k=300.0,
        n_steps=10000, frame_every=2000, avg_steps=2000, phase_bins=20,
        wall_iedf_bins=60,
    )
    sim = Fluid1dSimulation(_project1d(fluid1d=s))
    sim.run_batch(store_frames=False)
    assert sim.wall_iedf is not None
    for side in ("left", "right"):
        d = sim.wall_iedf[side]
        assert np.all(np.isfinite(d["f"]))
        assert d["total_weight"] > 0.0
        assert d["mean_energy_ev"] > 0.0
        assert d["model"] == "collisionless_sheath"

    result = build_fluid1d_result(sim, elapsed_s=1.0)
    assert result["wall_iedf"] is not None
    assert set(result["wall_iedf"]["left"]) == {
        "e_centers", "f", "mean_energy_ev", "total_weight", "n_samples", "model",
    }
