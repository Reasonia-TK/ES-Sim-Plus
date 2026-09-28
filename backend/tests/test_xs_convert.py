"""断面積の変換 es_sim.xs.convert のテスト (prompts/120)。

- EFFECTIVE→ELASTIC (LXCat/BOLSIG+ の定義 σ_el = max(σ_eff − Σσ_inel, 0)): 解析的に作った
  区分線形の断面積で差分が評価点の間も含めて厳密に正しいこと、閾値・表の開始点・表の重複点の
  段差、0 クリップの折れ点と警告
- resolve_momentum_transfer (ELASTIC 優先で EFFECTIVE 除外・EFFECTIVE のみなら変換・どちらも無い)
- build_mixture (分率・質量の決定順・検証エラー)、GasMixture.number_density
- to_boltzpmp_mixture → bp.PMSolver(...).solve_dc(100) が収束 (Ar 単体と Ar/N2 混合)、
  `<->` の生成物の扱い
- to_v1_processes (v1 XsProcess) と v1 互換の意図的な挙動変更
- masses (分子量テーブル・質量比)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from es_sim.lxcat import parse_lxcat
from es_sim.xs import (
    CrossSection,
    GasComponent,
    GasMixture,
    build_mixture,
    effective_to_elastic,
    load_lxcat,
    lookup_mass_amu,
    mass_amu_from_ratio,
    mass_ratio_from_amu,
    parse_lxcat_document,
    resolve_momentum_transfer,
    to_boltzpmp_cross_section,
    to_boltzpmp_mixture,
    to_v1_processes,
)
from es_sim.xs.masses import KB

DATA = Path(__file__).parent / "data"
FULL = DATA / "synthetic_lxcat_full.txt"
SYN_E = DATA / "synthetic_electron.txt"
SYN_I = DATA / "synthetic_ion.txt"


def _cs(kind: str, e, s, *, target: str = "Ar", threshold: float = 0.0, **kw) -> CrossSection:
    return CrossSection(
        kind=kind, projectile=kw.pop("projectile", "e"), target=target, threshold_ev=threshold,
        energy_ev=e, sigma_m2=s, **kw,
    )


def _expected_elastic(eff: CrossSection, inel: list[CrossSection], eps: np.ndarray) -> np.ndarray:
    total = eff.sigma(eps)
    for c in inel:
        total = total - c.sigma(eps)
    return np.maximum(total, 0.0)


def _dense_grid(lo: float, hi: float, *specials: float) -> np.ndarray:
    pts = [np.linspace(lo, hi, 20001)]
    for x in specials:
        pts.append(np.array([x, np.nextafter(x, -np.inf), np.nextafter(x, np.inf), x - 1e-7, x + 1e-7]))
    g = np.unique(np.concatenate(pts))
    return g[(g >= lo) & (g <= hi)]


# ---- 1. EFFECTIVE → ELASTIC ------------------------------------------------------------


def _analytic_set():
    eff = _cs("EFFECTIVE", [0.0, 10.0, 100.0], [1.0e-19, 2.0e-19, 2.0e-19], mass_ratio=1.36e-5,
              process="E + Ar -> E + Ar, Effective", line=7)
    # 閾値で 0 → 5e-21 に跳ぶ (表の最初の値が 0 でない) 励起
    exc = _cs("EXCITATION", [5.0, 20.0, 100.0], [5.0e-21, 3.0e-20, 1.0e-20], threshold=5.0,
              product="Ar*")
    # effective の範囲 (100 eV) を越えて続く電離
    ion = _cs("IONIZATION", [15.0, 50.0, 200.0], [0.0, 5.0e-20, 5.0e-20], threshold=15.0,
              product="Ar^+")
    att = _cs("ATTACHMENT", [0.0, 3.0, 4.0, 1000.0], [0.0, 0.0, 1.0e-21, 0.0])
    rot = _cs("ROTATION", [0.02, 1.0, 100.0], [0.0, 2.0e-21, 1.0e-21], threshold=0.02,
              lower_state=(0.0, 1.0), upper_state=(0.02, 5.0), weight_ratio=5.0)
    return eff, [exc, ion, att, rot]


def test_effective_to_elastic_matches_analytic_difference():
    eff, inel = _analytic_set()
    el, warnings = effective_to_elastic(eff, inel)
    assert warnings == []
    assert el.kind == "ELASTIC" and el.threshold_ev == 0.0
    assert el.mass_ratio == 1.36e-5
    assert el.param == {"converted_from": "EFFECTIVE"}
    assert (el.projectile, el.target, el.line, el.process) == ("e", "Ar", 7, eff.process)
    assert el.sigma_mt_m2 is None
    assert el.energy_ev[0] == 0.0 and el.energy_ev[-1] == 100.0  # effective の範囲
    eps = _dense_grid(0.0, 100.0, 0.02, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 50.0)
    np.testing.assert_allclose(
        el.sigma(eps), _expected_elastic(eff, inel, eps), rtol=1e-12, atol=1e-33
    )


def test_effective_to_elastic_threshold_step():
    eff, inel = _analytic_set()
    el, _ = effective_to_elastic(eff, inel)
    # 閾値 5 eV で段差: 同じエネルギーの 2 点 (左極限 → 右極限)
    k = np.flatnonzero(el.energy_ev == 5.0)
    assert len(k) == 2
    left = eff.sigma(5.0) - inel[2].sigma(5.0) - inel[3].sigma(5.0)  # 励起は閾値未満で 0
    assert el.sigma_m2[k[0]] == pytest.approx(left, rel=1e-14)
    assert el.sigma_m2[k[1]] == pytest.approx(left - 5.0e-21, rel=1e-14)
    assert el.sigma(5.0) == pytest.approx(left - 5.0e-21, rel=1e-14)  # 点ちょうどは右連続
    assert el.sigma(5.0 - 1e-9) == pytest.approx(left, rel=1e-9)
    # 閾値 15 eV の電離は表が 0 から始まるので段差なし (重複点を作らない)
    assert np.count_nonzero(el.energy_ev == 15.0) == 1
    # 表の開始点が閾値より上で値が 0 でない場合と、effective 自身の段差
    eff2 = _cs("EFFECTIVE", [0.0, 10.0, 10.0, 100.0], [1e-19, 1e-19, 2e-19, 2e-19])
    exc2 = _cs("EXCITATION", [6.0, 100.0], [1e-20, 1e-20], threshold=5.0)
    el2, w2 = effective_to_elastic(eff2, [exc2])
    assert w2 == []
    assert np.count_nonzero(el2.energy_ev == 6.0) == 2
    assert np.count_nonzero(el2.energy_ev == 10.0) == 2
    eps = _dense_grid(0.0, 100.0, 5.0, 6.0, 10.0)
    np.testing.assert_allclose(el2.sigma(eps), _expected_elastic(eff2, [exc2], eps), rtol=1e-12, atol=1e-33)


def test_effective_to_elastic_clipping_warns_and_inserts_crossing():
    eff = _cs("EFFECTIVE", [0.0, 100.0], [1.0e-20, 1.0e-20], mass_ratio=1.36e-5, line=3)
    exc = _cs("EXCITATION", [10.0, 30.0, 100.0], [0.0, 3.0e-20, 3.0e-20], threshold=10.0)
    el, warnings = effective_to_elastic(eff, [exc])
    assert len(warnings) == 1
    assert warnings[0].startswith("3 行目") and "クリップ" in warnings[0]
    # σ_eff − σ_exc は 10 + 20/3 eV で 0 を横切る: その折れ点が表にある
    crossing = 10.0 + 20.0 / 3.0
    assert np.min(np.abs(el.energy_ev - crossing)) < 1e-12
    eps = _dense_grid(0.0, 100.0, 10.0, crossing, 30.0)
    np.testing.assert_allclose(el.sigma(eps), _expected_elastic(eff, [exc], eps), rtol=1e-12, atol=1e-33)
    assert el.sigma(16.0) == pytest.approx(0.1e-20, rel=1e-12)
    assert el.sigma(17.0) == 0.0 and el.sigma(90.0) == 0.0
    assert np.all(el.sigma_m2 >= 0.0)


def test_effective_to_elastic_small_clip_is_silent():
    eff = _cs("EFFECTIVE", [0.0, 100.0], [1.0e-20, 1.0e-20])
    exc = _cs("EXCITATION", [10.0, 100.0], [1.004e-20, 1.004e-20], threshold=10.0)
    el, warnings = effective_to_elastic(eff, [exc])
    assert warnings == []           # クリップ量 0.4% ≤ 1%
    assert el.sigma(50.0) == 0.0     # 0 でクリップはする


def test_effective_to_elastic_without_inelastic_and_input_checks():
    eff = _cs("EFFECTIVE", [0.0, 1.0, 1.0, 5.0], [1e-20, 2e-20, 3e-20, 3e-20],
              sigma_mt_m2=[1e-20, 1e-20, 1e-20, 1e-20])
    el, warnings = effective_to_elastic(eff, [])
    np.testing.assert_array_equal(el.energy_ev, eff.energy_ev)
    np.testing.assert_array_equal(el.sigma_m2, eff.sigma_m2)
    assert len(warnings) == 1 and "3 列目" in warnings[0]  # EFFECTIVE の運動量移行列は使わない
    with pytest.raises(ValueError, match="EFFECTIVE ではありません"):
        effective_to_elastic(_cs("ELASTIC", [0.0, 1.0], [1e-20, 1e-20]), [])
    with pytest.raises(ValueError, match="差し引けません"):
        effective_to_elastic(eff, [_cs("ELASTIC", [0.0, 1.0], [1e-20, 1e-20])])


# ---- 2. resolve_momentum_transfer ------------------------------------------------------


def test_resolve_momentum_transfer_cases():
    el_ar = _cs("ELASTIC", [0.0, 10.0], [1e-19, 1e-19], mass_ratio=1.36e-5, line=1)
    eff_ar = _cs("EFFECTIVE", [0.0, 10.0], [2e-19, 2e-19], line=2)
    exc_ar = _cs("EXCITATION", [5.0, 10.0], [0.0, 1e-20], threshold=5.0, line=3)
    eff_n2 = _cs("EFFECTIVE", [0.0, 10.0], [2e-19, 2e-19], target="N2", mass_ratio=1.96e-5, line=4)
    exc_n2 = _cs("EXCITATION", [1.0, 10.0], [0.0, 1e-20], target="N2", threshold=1.0, line=5)
    exc_kr = _cs("EXCITATION", [9.9, 20.0], [0.0, 1e-20], target="Kr", threshold=9.9, line=6)
    iso = _cs("ISOTROPIC", [0.0, 10.0], [3e-19, 3e-19], projectile="Ar^+", line=7)
    out, warnings = resolve_momentum_transfer([el_ar, eff_ar, exc_ar, eff_n2, exc_n2, exc_kr, iso])
    # Ar: ELASTIC があるので EFFECTIVE を除外、N2: EFFECTIVE を同じ位置で ELASTIC に変換
    assert [(c.line, c.kind, c.target) for c in out] == [
        (1, "ELASTIC", "Ar"), (3, "EXCITATION", "Ar"), (4, "ELASTIC", "N2"),
        (5, "EXCITATION", "N2"), (6, "EXCITATION", "Kr"), (7, "ISOTROPIC", "Ar"),
    ]
    assert out[0] is el_ar and out[5] is iso  # それ以外はそのまま通す
    converted, _ = effective_to_elastic(eff_n2, [exc_n2])  # 同じ標的の非弾性だけを差し引く
    np.testing.assert_array_equal(out[2].sigma_m2, converted.sigma_m2)
    assert out[2].mass_ratio == 1.96e-5
    assert len(warnings) == 3
    assert warnings[0].startswith("2 行目") and "EFFECTIVE" in warnings[0] and "除外" in warnings[0]
    assert warnings[1].startswith("4 行目") and "変換" in warnings[1]
    assert "Kr" in warnings[2] and "運動量移行断面積" in warnings[2]


def test_resolve_momentum_transfer_duplicate_warning():
    el1 = _cs("ELASTIC", [0.0, 10.0], [1e-19, 1e-19], line=1)
    el2 = _cs("ELASTIC", [0.0, 10.0], [1e-19, 1e-19], line=9)
    out, warnings = resolve_momentum_transfer([el1, el2])
    assert out == [el1, el2]
    assert len(warnings) == 1 and "二重計上" in warnings[0]


# ---- 3. 混合ガス -------------------------------------------------------------------


def test_build_mixture_full_fixture():
    doc = load_lxcat(FULL)
    mix, warnings = build_mixture(doc, {"Ar": 0.7, "N2": 0.3}, 100.0, 300.0)
    assert [c.name for c in mix.components] == ["Ar", "N2"]
    assert [c.fraction for c in mix.components] == [0.7, 0.3]
    ar, n2 = mix.components
    assert [c.kind for c in ar.cross_sections] == ["ELASTIC", "EXCITATION", "IONIZATION"]
    assert [c.kind for c in n2.cross_sections] == [
        "ELASTIC", "ROTATION", "EXCITATION", "EXCITATION", "IONIZATION", "ATTACHMENT",
    ]
    assert n2.cross_sections[0].param["converted_from"] == "EFFECTIVE"
    assert all(c.projectile == "e" for comp in mix.components for c in comp.cross_sections)
    assert mix.number_density == pytest.approx(100.0 / (KB * 300.0), rel=1e-15)
    assert len(warnings) == 2
    assert warnings[0].startswith("108 行目") and "除外" in warnings[0]  # MOMENTUM (Ar)
    assert warnings[1].startswith("123 行目") and "変換" in warnings[1]  # EFFECTIVE (N2)
    mix.validate()


def test_build_mixture_mass_resolution_order():
    doc = load_lxcat(FULL)
    # 1) masses 引数が最優先
    mix, _ = build_mixture(doc, {"Ar": 1.0}, 100.0, 300.0, masses={"Ar": 39.948})
    assert mix.components[0].mass_amu == 39.948
    # 2) 次に質量表 (LXCat の m/M 1.36e-5 は丸められていて 40.34 amu になるため表を優先)
    mix, w = build_mixture(doc, {"Ar": 1.0}, 100.0, 300.0)
    assert mix.components[0].mass_amu == 39.948
    assert any("N2" in x and "使いません" in x for x in w)  # 指定しなかった標的
    # 3) 表に無い標的は m/M から M = m_e/(m/M)、4) どちらも無ければ None + 警告
    text = (
        "ELASTIC\nYy\n 1.36e-5\n-----\n 0 1e-19\n 10 1e-19\n-----\n"
        "ELASTIC\nXx\n-----\n 0 1e-19\n 10 1e-19\n-----\n"
    )
    mix, w = build_mixture(parse_lxcat_document(text), {"Yy": 0.5, "Xx": 0.5}, 100.0, 300.0)
    assert mix.components[0].mass_amu == pytest.approx(mass_amu_from_ratio(1.36e-5), rel=1e-15)
    assert mix.components[0].mass_amu == pytest.approx(40.3368, rel=1e-5)
    assert mix.components[1].mass_amu is None
    assert len(w) == 1 and "Xx" in w[0] and "分子量" in w[0]


@pytest.mark.parametrize(
    ("fractions", "match"),
    [
        ({"Ar": 0.5, "N2": 0.4}, "分率の和"),
        ({"Ar": 1.2, "N2": -0.2}, "非負"),
        ({"O2": 1.0}, "O2"),
        ({}, "fractions が空"),
    ],
)
def test_build_mixture_validation_errors(fractions, match):
    with pytest.raises(ValueError, match=match):
        build_mixture(load_lxcat(FULL), fractions, 100.0, 300.0)


def test_build_mixture_requires_momentum_transfer_and_positive_state():
    doc = parse_lxcat_document("EXCITATION\nAr -> Ar*\n 11.5\n-----\n 11.5 0\n 20 1e-20\n-----\n")
    with pytest.raises(ValueError, match="運動量移行断面積"):
        build_mixture(doc, {"Ar": 1.0}, 100.0, 300.0)
    with pytest.raises(ValueError, match="圧力"):
        build_mixture(load_lxcat(FULL), {"Ar": 1.0}, 0.0, 300.0)
    mix = GasMixture(components=[], pressure_pa=1.0, temperature_k=300.0)
    with pytest.raises(ValueError, match="成分がありません"):
        mix.validate()


# ---- 4. boltzpmp -------------------------------------------------------------------


def _solve_dc_100(bp, mixture):
    solver = bp.PMSolver(mixture, eps_max_eV=60, d_eps_eV=0.2, n_theta=16)
    return solver.solve_dc(100)


def test_boltzpmp_mixture_ar_converges():
    bp = pytest.importorskip("boltzpmp")
    mix, _ = build_mixture(load_lxcat(FULL), {"Ar": 1.0}, 133.0, 300.0)
    notes: list[str] = []
    bmix = to_boltzpmp_mixture(mix, warnings_out=notes)
    assert notes == []
    assert isinstance(bmix, bp.Mixture)
    (gas,) = bmix.gases
    assert (gas.name, gas.fraction) == ("Ar", 1.0)
    assert gas.mass_amu == 39.948  # 質量表を優先 (m/M は弾性のエネルギー損失にそのまま使われる)
    assert [c.kind for c in gas.cross_sections] == ["ELASTIC", "EXCITATION", "IONIZATION"]
    el = gas.cross_sections[0]
    assert el.mt_data is not None and el.mass_ratio == 1.36e-5  # 3 列目 → mt_data
    assert bmix.N == pytest.approx(mix.number_density, rel=1e-12)
    res = _solve_dc_100(bp, bmix)
    assert res.converged
    assert 1.0 < res.mean_energy < 20.0
    assert res.drift_velocity > 0.0 and res.reduced_ionization_frequency > 0.0
    assert "Ar:E + Ar -> E + E + Ar+, Ionization" in res.rate_coefficients


def test_boltzpmp_mixture_ar_n2_converges():
    bp = pytest.importorskip("boltzpmp")
    mix, _ = build_mixture(load_lxcat(FULL), {"Ar": 0.8, "N2": 0.2}, 133.0, 300.0)
    notes: list[str] = []
    bmix = to_boltzpmp_mixture(mix, warnings_out=notes)
    assert [g.name for g in bmix.gases] == ["Ar", "N2"]
    assert [g.fraction for g in bmix.gases] == [0.8, 0.2]
    n2 = bmix.gases[1]
    kinds = [c.kind for c in n2.cross_sections]
    assert kinds == ["ELASTIC", "ROTATION", "EXCITATION", "EXCITATION", "IONIZATION", "ATTACHMENT"]
    rot = n2.cross_sections[1]
    assert rot.lower_state == (0.0, 1.0) and rot.upper_state == (1.488e-3, 5.0)
    assert rot.threshold == pytest.approx(1.488e-3) and rot.weight_ratio is None
    a3 = n2.cross_sections[2]
    # 生成物 N2(A3) は混合ガスの成分に無い → `<->` を外す (二準位系として扱われる)
    assert a3.species == "N2 -> N2(A3)" and a3.weight_ratio == 3.0
    assert len(notes) == 1 and "N2(A3)" in notes[0] and "<->" in notes[0]
    res = _solve_dc_100(bp, bmix)
    assert res.converged
    assert 1.0 < res.mean_energy < 20.0
    assert res.reduced_attachment_frequency > 0.0
    assert "N2:E + N2 -> E + N2, Effective [EFFECTIVE→ELASTIC]" in res.rate_coefficients
    assert "N2:E + N2(J=0) -> E + N2(J=2), Rotation (superelastic)" in res.rate_coefficients


def _two_level_mixture(fraction_star: float | None) -> GasMixture:
    el = _cs("ELASTIC", [0.0, 100.0], [1e-19, 1e-19], mass_ratio=1.36e-5)
    exc = _cs("EXCITATION", [11.55, 20.0, 100.0], [0.0, 1e-20, 1e-20], threshold=11.55,
              product="Ar*", reversible=True, weight_ratio=5.0, process="Ar <-> Ar*")
    comps = [GasComponent("Ar", 1.0 if fraction_star is None else 1.0 - fraction_star, 39.948, [el, exc])]
    if fraction_star is not None:
        el_star = _cs("ELASTIC", [0.0, 100.0], [1e-18, 1e-18], target="Ar*", mass_ratio=1.36e-5)
        comps.append(GasComponent("Ar*", fraction_star, 39.948, [el_star]))
    return GasMixture(components=comps, pressure_pa=133.0, temperature_k=300.0)


def test_boltzpmp_reversible_product_handling():
    bp = pytest.importorskip("boltzpmp")
    # 生成物 Ar* が成分にあれば `<->` を残す (逆過程の標的は Ar* の気体)
    notes: list[str] = []
    bmix = to_boltzpmp_mixture(_two_level_mixture(1e-3), warnings_out=notes)
    assert notes == []
    assert bmix.gases[0].cross_sections[1].species == "Ar <-> Ar*"
    procs = bp.PMSolver(bmix, eps_max_eV=60, d_eps_eV=0.2, n_theta=16).processes()
    assert any(p["gas"] == "Ar*" and p["name"] == "Ar <-> Ar* (superelastic)" for p in procs)
    # 成分に無いまま `<->` を渡すと boltzpmp はエラーにする (外す理由)
    raw = _two_level_mixture(None)
    gas = bp.Gas("Ar", 1.0, [to_boltzpmp_cross_section(c) for c in raw.components[0].cross_sections])
    with pytest.raises(ValueError, match="not a gas of the mixture"):
        bp.PMSolver(bp.Mixture([gas], p_Pa=133.0, T_K=300.0), eps_max_eV=60, d_eps_eV=0.2, n_theta=16)
    # to_boltzpmp_mixture は外して警告し、ソルバーが組める
    notes = []
    bmix = to_boltzpmp_mixture(raw, warnings_out=notes)
    assert bmix.gases[0].cross_sections[1].species == "Ar -> Ar*"
    assert len(notes) == 1
    assert _solve_dc_100(bp, bmix).converged


def test_boltzpmp_mixture_converts_leftover_effective_and_checks_mass():
    pytest.importorskip("boltzpmp")
    eff = _cs("EFFECTIVE", [0.0, 100.0], [2e-19, 2e-19])  # m/M なし
    exc = _cs("EXCITATION", [11.55, 100.0], [0.0, 1e-20], threshold=11.55, product="Ar*")
    iso = _cs("ISOTROPIC", [0.0, 100.0], [3e-19, 3e-19], projectile="Ar^+")
    dup = _cs("EXCITATION", [12.0, 100.0], [0.0, 1e-20], threshold=12.0, product="Ar*")
    mix = GasMixture([GasComponent("Ar", 1.0, 39.948, [eff, exc, dup, iso])], 133.0, 300.0)
    notes: list[str] = []
    bmix = to_boltzpmp_mixture(mix, warnings_out=notes)
    names = [c.name for c in bmix.gases[0].cross_sections]
    kinds = [c.kind for c in bmix.gases[0].cross_sections]
    assert kinds == ["ELASTIC", "EXCITATION", "EXCITATION"]  # EFFECTIVE は変換、イオンは渡さない
    assert names[1] == "Ar -> Ar* excitation" and names[2] == "Ar -> Ar* excitation #2"
    assert any("ISOTROPIC" in n for n in notes) and any("変換" in n for n in notes)
    assert bmix.gases[0].cross_sections[0].mass_ratio is None  # Gas.mass_amu から
    mix.components[0].mass_amu = None
    with pytest.raises(ValueError, match="mass_amu"):
        to_boltzpmp_mixture(mix)


# ---- 5. v1 XsProcess ---------------------------------------------------------------


def test_to_v1_synthetic_electron_intentional_change():
    """ELASTIC と EFFECTIVE を両方持つ Ar: EFFECTIVE を除外 (v1 は二重計上していた)。"""
    procs, warnings = parse_lxcat(SYN_E.read_text(encoding="utf-8"), "electron")
    assert [p.kind for p in procs] == ["elastic", "excitation", "ionization"]
    assert len(warnings) == 2
    assert "EFFECTIVE" in warnings[0] and "ATTACHMENT" in warnings[1]
    assert procs[0].sigma_m2 == [1e-19] * 4 and procs[0].mass_ratio == 1.36e-5


def test_to_v1_full_fixture_electron():
    doc = load_lxcat(FULL)
    procs, warnings = to_v1_processes(doc, "electron")
    assert [(p.kind, p.label) for p in procs] == [
        ("elastic", "E + Ar -> E + Ar, Elastic"),
        ("excitation", "E + Ar -> E + Ar*(11.55eV), Excitation"),
        ("ionization", "E + Ar -> E + E + Ar+, Ionization"),
        ("elastic", "E + N2 -> E + N2, Effective"),
        ("excitation", "E + N2(J=0) -> E + N2(J=2), Rotation"),
        ("excitation", "E + N2 <-> E + N2(A3), Excitation"),
        ("excitation", "E + N2 -> E + N2(v1), Excitation"),
        ("ionization", "E + N2 -> E + E + N2+, Ionization"),
    ]
    by_line = {c.line: c for c in doc.all_cross_sections()}
    assert procs[0].sigma_m2 == by_line[21].sigma_mt_m2.tolist()  # 3 列目 (運動量移行)
    assert procs[1].threshold_ev == 11.55 and procs[1].mass_ratio == 0.0
    assert procs[3].mass_ratio == 1.96e-5
    n2 = [c for c in doc.all_cross_sections() if c.target == "N2" and c.projectile == "e"]
    converted, _ = effective_to_elastic(n2[0], n2[1:])
    assert procs[3].sigma_m2 == converted.sigma_m2.tolist()
    assert procs[4].threshold_ev == 1.488e-3
    assert procs[7].sigma_m2[-1] == pytest.approx(1.0e-20)  # cm² → m²
    assert warnings[0] == doc.warnings[0]  # パース警告が先頭
    joined = "\n".join(warnings)
    for key in ("除外", "変換", "単一ガス", "ROTATION", "<->", "ATTACHMENT", "3 列目"):
        assert key in joined, key
    assert sum("species='electron'" in w for w in warnings) == 2  # イオン 2 ブロック


def test_to_v1_ion_and_filters():
    doc = load_lxcat(FULL)
    procs, warnings = to_v1_processes(doc, "ion")
    assert [(p.kind, p.threshold_ev, p.mass_ratio) for p in procs] == [
        ("isotropic", 0.0, 0.0), ("backscat", 0.0, 0.0),
    ]
    assert procs[0].sigma_m2 == [3e-19, 3e-19, 2e-19]
    assert sum("species='ion'" in w for w in warnings) == 10  # 電子の 10 ブロック
    procs, warnings = parse_lxcat(SYN_I.read_text(encoding="utf-8"), "ion")
    assert [p.kind for p in procs] == ["backscat", "isotropic"] and warnings == []
    with pytest.raises(ValueError, match="species"):
        to_v1_processes(doc, "neutral")  # type: ignore[arg-type]


def test_to_v1_elastic_mass_ratio_fallbacks():
    text = (
        "ELASTIC\nAr\n-----\n 0 1e-19\n 10 1e-19\n-----\n"
        "ELASTIC\nXx\n-----\n 0 1e-19\n 10 1e-19\n-----\n"
    )
    procs, warnings = to_v1_processes(parse_lxcat_document(text), "electron")
    assert procs[0].mass_ratio == mass_ratio_from_amu(39.948)  # 質量表から
    assert procs[1].mass_ratio == 0.0                          # 不明 → 0 (警告)
    assert any("Xx" in w and "m/M = 0" in w for w in warnings)
    assert any("単一ガス" in w for w in warnings)


# ---- 6. masses ---------------------------------------------------------------------


def test_masses_table_and_ratios():
    assert lookup_mass_amu("Ar") == 39.948
    assert lookup_mass_amu("Ar*(11.55eV)") == 39.948
    assert lookup_mass_amu("Ar^+") == 39.948
    assert lookup_mass_amu("N2(A3)") == 28.014
    assert lookup_mass_amu("CO") == 28.010 and lookup_mass_amu("Co") is None
    assert lookup_mass_amu("Xx") is None
    ratio = mass_ratio_from_amu(39.948)
    assert mass_amu_from_ratio(ratio) == pytest.approx(39.948, rel=1e-15)
    with pytest.raises(ValueError):
        mass_amu_from_ratio(1.5)
    with pytest.raises(ValueError):
        mass_ratio_from_amu(0.0)


def test_mass_ratio_matches_boltzpmp():
    pytest.importorskip("boltzpmp")
    from boltzpmp import _core

    for m in (4.002602, 28.014, 39.948, 131.293):
        assert mass_ratio_from_amu(m) == _core.mass_ratio_from_amu(m)
