"""fluid_coeffs.py のテスト (prompts/105)。

1. 一定断面積の解析解: k(Te) = σ0・v̄ と rtol 1e-2 で一致 (積分グリッド分解能の検証)
2. 閾値反応の低温極限: 低 Te で k が exp(-threshold/Te) スケールで小さいこと
3. eduPIC Ar セット: build_fluid_reactions が例外なく構築でき、係数が物理的常識範囲
4. interp_loglog: グリッド点上で厳密一致・範囲外クランプ
"""

import math

import numpy as np
import pytest

from es_sim.fluid_coeffs import (
    TE_GRID_EV,
    ElectronTransport,
    FluidReactions,
    build_fluid_reactions,
    electron_transport,
    interp_loglog,
    rate_coefficient_table,
)
from es_sim.particles import ME, QE
from es_sim.pic1d_presets import edupic_ar_processes
from es_sim.schema import XsProcess


def _constant_sigma_process(sigma0: float, e_min=0.1, e_max=1000.0, n=400) -> XsProcess:
    """σ = σ0 一定 (閾値なし) の合成 XsProcess (elastic 扱い)。"""
    e = np.geomspace(e_min, e_max, n)
    s = np.full(n, sigma0)
    return XsProcess(kind="elastic", label="const sigma", threshold_ev=0.0, mass_ratio=0.0,
                      energy_ev=e.tolist(), sigma_m2=s.tolist())


def _threshold_process(sigma0: float, threshold_ev: float, e_max=1000.0, n=400) -> XsProcess:
    """σ = σ0 (E > threshold)・0 (E <= threshold) の合成 XsProcess (ionization 扱い)。"""
    e = np.unique(np.concatenate([[threshold_ev], np.geomspace(1e-3, e_max, n)]))
    s = np.where(e > threshold_ev, sigma0, 0.0)
    return XsProcess(kind="ionization", label="threshold sigma", threshold_ev=threshold_ev,
                      mass_ratio=0.0, energy_ev=e.tolist(), sigma_m2=s.tolist())


# ---- 1. 一定断面積の解析解 ------------------------------------------------------


def test_constant_sigma_matches_mean_speed():
    """k(Te) = sigma0 * v̄ (v̄ = sqrt(8 e Te / (pi m_e))、Maxwell 分布の平均速さ)。

    合成テーブルは 0.1〜1000 eV に限定されているため、Te が小さすぎると
    Maxwell 分布の低エネルギー側 (< 0.1 eV) の寄与が切り捨てられて誤差が
    大きくなる。物理的に意味のある Te (>=1 eV、この切り捨て寄与は <1%) で検証する。
    """
    sigma0 = 3.0e-19
    proc = _constant_sigma_process(sigma0)
    te_test = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
    k = rate_coefficient_table(proc, te_test)
    v_bar = np.sqrt(8.0 * QE * te_test / (math.pi * ME))
    expected = sigma0 * v_bar
    np.testing.assert_allclose(k, expected, rtol=1e-2)


def test_constant_sigma_full_grid_reasonable():
    """モジュール既定の TE_GRID_EV 全域でも NaN/inf なく妥当な桁になること。"""
    proc = _constant_sigma_process(3.0e-19)
    k = rate_coefficient_table(proc, TE_GRID_EV)
    assert np.all(np.isfinite(k))
    assert np.all(k >= 0.0)


# ---- 2. 閾値反応の低温極限 -------------------------------------------------------


def test_threshold_reaction_low_temperature_limit():
    """閾値 10 eV・Te=1 eV では k が exp(-threshold/Te) スケールで極端に小さい。"""
    proc = _threshold_process(1.0e-20, threshold_ev=10.0)
    k = rate_coefficient_table(proc, np.array([1.0, 10.0]))
    k_1ev, k_10ev = k
    assert k_10ev > 0.0
    assert k_1ev / k_10ev < 1e-3


def test_threshold_reaction_zero_below_grid():
    """閾値未満の Te でも k は非負・有限 (アンダーフローで 0 になるのは正常)。"""
    proc = _threshold_process(1.0e-20, threshold_ev=10.0)
    k = rate_coefficient_table(proc, np.array([0.05, 0.1, 0.5]))
    assert np.all(np.isfinite(k))
    assert np.all(k >= 0.0)
    assert k[0] <= k[-1]  # 低温ほど電離レートは小さいか同等


# ---- 3. eduPIC Ar セット --------------------------------------------------------


def test_edupic_ar_build_fluid_reactions():
    e_procs, _i_procs = edupic_ar_processes()
    reactions = build_fluid_reactions(e_procs)
    assert isinstance(reactions, FluidReactions)
    assert isinstance(reactions.transport, ElectronTransport)
    assert reactions.te_grid_ev.shape == TE_GRID_EV.shape

    k_ion_3ev = float(interp_loglog(reactions.te_grid_ev, reactions.k_ion, 3.0))
    assert 1e-16 <= k_ion_3ev <= 1e-13, f"k_ion(3eV) = {k_ion_3ev:.3e} m^3/s は想定範囲外"

    mobility_n = reactions.transport.mobility_n
    finite = mobility_n[np.isfinite(mobility_n) & (mobility_n > 0.0)]
    assert finite.size > 0
    # 典型的な Te 範囲 (数 eV 近傍) で 1e23〜1e25 1/(m・V・s) のオーダーにあること
    mid = reactions.transport.te_grid_ev
    mask = (mid >= 1.0) & (mid <= 10.0)
    mid_vals = mobility_n[mask]
    assert np.all(mid_vals > 1e22) and np.all(mid_vals < 1e26), (
        f"mobility_n (1-10eV) range = [{mid_vals.min():.3e}, {mid_vals.max():.3e}]"
    )

    assert reactions.e_ion_ev == pytest.approx(15.8, rel=1e-6)
    assert reactions.e_exc_ev == pytest.approx(11.5, rel=1e-6)
    assert np.all(reactions.k_ion >= 0.0)
    assert np.all(reactions.k_exc >= 0.0)
    assert np.all(np.isfinite(reactions.k_ion))
    assert np.all(np.isfinite(reactions.k_exc))


def test_edupic_ar_requires_ionization():
    e_procs, _i_procs = edupic_ar_processes()
    no_ion = [p for p in e_procs if p.kind != "ionization"]
    with pytest.raises(ValueError):
        build_fluid_reactions(no_ion)


def test_edupic_ar_requires_elastic():
    e_procs, _i_procs = edupic_ar_processes()
    no_elastic = [p for p in e_procs if p.kind != "elastic"]
    with pytest.raises(ValueError):
        build_fluid_reactions(no_elastic)


def test_electron_transport_matches_manual_mobility():
    """electron_transport の mobility_n = e/(m_e・nu_m_per_ng) を単体で確認する。"""
    e_procs, _i_procs = edupic_ar_processes()
    elastic = next(p for p in e_procs if p.kind == "elastic")
    transport = electron_transport(elastic, TE_GRID_EV)
    nu = transport.nu_m_per_ng
    expected = np.divide(QE, ME * nu, out=np.zeros_like(nu), where=nu > 0.0)
    np.testing.assert_allclose(transport.mobility_n, expected, rtol=1e-12)


# ---- 4. 補間 (interp_loglog) ----------------------------------------------------


def test_interp_loglog_exact_at_grid_points():
    te_grid = np.geomspace(0.1, 100.0, 20)
    table = 1e-15 * te_grid ** 1.5  # 単調なべき乗則テーブル
    got = interp_loglog(te_grid, table, te_grid)
    np.testing.assert_allclose(got, table, rtol=1e-10)


def test_interp_loglog_scalar_exact_at_grid_point():
    te_grid = np.geomspace(0.1, 100.0, 20)
    table = 1e-15 * te_grid ** 1.5
    got = interp_loglog(te_grid, table, float(te_grid[5]))
    assert isinstance(got, float)
    assert got == pytest.approx(table[5], rel=1e-10)


def test_interp_loglog_powerlaw_between_grid_points():
    """厳密なべき乗則テーブルなら格子点の間でも log-log 補間が厳密に一致する。"""
    te_grid = np.geomspace(0.1, 100.0, 20)
    table = 2.0e-15 * te_grid ** 2.0
    te_mid = np.sqrt(te_grid[:-1] * te_grid[1:])  # 各区間の幾何平均 (log空間で中点)
    got = interp_loglog(te_grid, table, te_mid)
    expected = 2.0e-15 * te_mid ** 2.0
    np.testing.assert_allclose(got, expected, rtol=1e-8)


def test_interp_loglog_clamps_out_of_range():
    te_grid = np.geomspace(0.1, 100.0, 20)
    table = 1e-15 * te_grid ** 1.5
    below = interp_loglog(te_grid, table, 0.001)
    above = interp_loglog(te_grid, table, 1000.0)
    assert below == pytest.approx(table[0], rel=1e-10)
    assert above == pytest.approx(table[-1], rel=1e-10)


def test_interp_loglog_handles_zero_segment_linearly():
    """テーブルに 0 (閾値未満) が含まれる区間は線形補間にフォールバックする。"""
    te_grid = np.array([1.0, 2.0, 3.0, 4.0])
    table = np.array([0.0, 0.0, 5.0, 10.0])
    got = interp_loglog(te_grid, table, 2.5)
    assert got == pytest.approx(2.5, rel=1e-10)  # (0+5)/2 の線形補間


def test_interp_loglog_vector_input_matches_scalar():
    te_grid = np.geomspace(0.1, 100.0, 20)
    table = 1e-15 * te_grid ** 1.5
    query = np.array([0.05, 0.3, 1.7, 50.0, 500.0])
    vec = interp_loglog(te_grid, table, query)
    scalar = np.array([interp_loglog(te_grid, table, float(q)) for q in query])
    np.testing.assert_allclose(vec, scalar, rtol=1e-12)
