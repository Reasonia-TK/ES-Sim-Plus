"""LXCat パーサー es_sim.xs のテスト (prompts/120)。

- 合成フィクスチャ synthetic_lxcat_full.txt (実データではない) の全ブロック種の各フィールド
  (kind/projectile/target/product/reversible/閾値/m/M/重み比/準位/単位換算後の値/sigma_mt/
  database/line) と DATABASE の meta
- 拡張 (MOMENTUM・ATTACHMENT の閾値行・PARAM 由来の閾値と m/M・COLUMNS 単位・4 列・
  タイプ行なし形式) と警告
- 構造エラーが行番号付きの ValueError になること
- boltzpmp 0.5.0 の bp.parse_lxcat と、電子の標準ブロックで全フィールドが一致すること
- CRLF・BOM・Latin-1・UTF-16・CR のみの改行
- CrossSection.sigma() の below/above
- to_dict()/from_dict() の往復
- /v2/xs/parse と /lxcat/parse のエンドポイント
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from es_sim.server import app
from es_sim.xs import (
    KINDS,
    CrossSection,
    LxcatDocument,
    load_lxcat,
    mass_amu_from_ratio,
    parse_lxcat_document,
    to_boltzpmp_cross_section,
)

DATA = Path(__file__).parent / "data"
FULL = DATA / "synthetic_lxcat_full.txt"
SYN_E = DATA / "synthetic_electron.txt"

DB_A = "SynthA (synthetic electron database A)"
DB_B = "SynthB (synthetic database B)"


def _full_text() -> str:
    return FULL.read_text(encoding="utf-8")  # 改行は universal newline で LF に揃う


def _by_line(doc: LxcatDocument) -> dict[int, CrossSection]:
    return {cs.line: cs for cs in doc.all_cross_sections()}


# ---- 1. 合成フィクスチャの全フィールド ------------------------------------------------


def test_full_fixture_block_kinds_and_order():
    doc = parse_lxcat_document(_full_text())
    css = doc.all_cross_sections()
    assert [(cs.line, cs.kind) for cs in css] == [
        (21, "ELASTIC"),
        (52, "EXCITATION"),
        (74, "IONIZATION"),
        (108, "EFFECTIVE"),   # MOMENTUM (EFFECTIVE の別名)
        (123, "EFFECTIVE"),
        (147, "ROTATION"),
        (165, "EXCITATION"),
        (183, "EXCITATION"),  # 閾値は PARAM.: のみ
        (203, "IONIZATION"),  # COLUMNS が cm2
        (221, "ATTACHMENT"),
        (237, "ISOTROPIC"),   # タイプ行なし (Phelps 形式)
        (248, "BACKSCAT"),
    ]
    assert {cs.kind for cs in css} == set(KINDS)  # 全 8 種別
    assert doc.targets() == ["Ar", "N2"]
    assert doc.targets("Ar^+") == ["Ar"]
    # 回復可能な問題は PARAM 由来の閾値の 1 件だけ (行番号付き)
    assert len(doc.warnings) == 1
    assert doc.warnings[0].startswith("185 行目") and "PARAM" in doc.warnings[0]


def test_full_fixture_databases_and_meta():
    doc = parse_lxcat_document(_full_text())
    assert [db.name for db in doc.databases] == [DB_A, DB_B]
    assert [len(db.cross_sections) for db in doc.databases] == [3, 9]
    a, b = doc.databases
    assert a.meta["PERMLINK"] == "www.example.invalid/SynthA"
    assert a.meta["CONTACT"] == "ES-Sim developers (synthetic)"
    assert a.meta["HOW TO REFERENCE"] == "not applicable (synthetic data)"
    assert a.meta["COMMENT"] == "Synthetic Ar group comment (database level)."
    # 字下げされた継続行は改行で連結
    assert a.meta["DESCRIPTION"] == (
        "Synthetic Ar set: ELASTIC with a 3-column table (integral + momentum\n"
        "transfer), EXCITATION with a step and a '!' comment, IONIZATION."
    )
    assert b.meta["PERMLINK"] == "www.example.invalid/SynthB"
    assert b.meta["DESCRIPTION"].count("\n") == 2
    for db in doc.databases:
        assert "DATABASE" not in db.meta
        assert all(cs.database == db.name for cs in db.cross_sections)


def test_full_fixture_elastic_three_columns():
    cs = _by_line(parse_lxcat_document(_full_text()))[21]
    assert (cs.kind, cs.projectile, cs.target, cs.product, cs.reversible) == (
        "ELASTIC", "e", "Ar", None, False,
    )
    assert cs.threshold_ev == 0.0
    assert cs.mass_ratio == 1.36e-5
    assert cs.weight_ratio is None and cs.lower_state is None and cs.upper_state is None
    assert cs.energy_ev.dtype == np.float64
    assert cs.energy_ev.tolist() == [
        0.0, 0.01, 0.05, 0.1, 0.23, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0,
        100.0, 200.0, 1000.0,
    ]
    assert cs.sigma_m2[0] == 7.5e-20 and cs.sigma_m2[-1] == 4.0e-20
    assert cs.sigma_mt_m2 is not None
    assert cs.sigma_mt_m2[7] == 4.5e-20 and cs.sigma_mt_m2[-1] == 9.0e-21
    assert cs.process == "E + Ar -> E + Ar, Elastic"
    assert cs.species == "e / Ar"
    assert cs.param == {"m/M": 1.36e-5, "complete set": True}
    # COMMENT: 行 + キーを持たない自由記述行
    assert cs.comment == (
        "synthetic elastic; column 2 = integral, column 3 = momentum transfer.\n"
        "Free text line without a key (goes to the comment)."
    )
    assert cs.meta == {"SOURCE": "synthetic generator"}
    assert cs.updated == "2026-09-28 12:00:00"
    assert cs.columns == "Energy (eV) | Cross section (m2) | Momentum transfer cross section (m2)"
    assert cs.database == DB_A
    assert cs.label == "E + Ar -> E + Ar, Elastic"


def test_full_fixture_excitation_ionization_ar():
    by = _by_line(parse_lxcat_document(_full_text()))
    exc = by[52]
    assert (exc.target, exc.product, exc.reversible) == ("Ar", "Ar*(11.55eV)", False)
    assert exc.threshold_ev == 11.55
    assert exc.weight_ratio is None  # "!" 以降 (コメント) は読まない
    assert exc.param == {"E": 11.55, "complete set": True}
    # 段差 (エネルギー 20 eV の重複): 点ちょうどでは後ろ側の値 (右連続)
    assert np.count_nonzero(np.diff(exc.energy_ev) == 0.0) == 1
    assert exc.sigma(20.0) == 6.0e-21
    assert exc.sigma(20.0 - 1e-9) == pytest.approx(5.5e-21, rel=1e-6)
    assert exc.sigma(11.0) == 0.0
    ion = by[74]
    assert (ion.kind, ion.target, ion.product) == ("IONIZATION", "Ar", "Ar^+")
    assert ion.threshold_ev == 15.76
    assert ion.sigma_mt_m2 is None
    assert ion.mass_ratio is None


def test_full_fixture_momentum_alias():
    cs = _by_line(parse_lxcat_document(_full_text()))[108]
    assert cs.kind == "EFFECTIVE"
    assert cs.param == {"keyword_alias": "MOMENTUM"}
    assert cs.mass_ratio == 1.36e-5
    assert (cs.target, cs.database) == ("Ar", DB_B)


def test_full_fixture_n2_blocks():
    by = _by_line(parse_lxcat_document(_full_text()))
    eff = by[123]
    assert (eff.kind, eff.target, eff.mass_ratio) == ("EFFECTIVE", "N2", 1.96e-5)
    assert len(eff.energy_ev) == 13

    rot = by[147]
    assert rot.kind == "ROTATION" and rot.target == "N2" and rot.product is None
    assert rot.lower_state == (0.0, 1.0)
    assert rot.upper_state == (1.488e-3, 5.0)
    assert rot.threshold_ev == 1.488e-3  # E_up − E_low
    assert rot.weight_ratio == 5.0      # g_up / g_low

    a3 = by[165]
    assert (a3.target, a3.product, a3.reversible) == ("N2", "N2(A3)", True)
    assert a3.threshold_ev == 6.17
    assert a3.weight_ratio == 3.0

    v1 = by[183]
    assert (v1.target, v1.product) == ("N2", "N2(v1)")
    assert v1.threshold_ev == 0.29
    assert v1.param["threshold_from_param"] is True
    assert v1.comment.count("\n") == 1  # COMMENT 2 行を改行で連結

    ion = by[203]  # COLUMNS: ... (cm2) → m² に換算
    assert ion.threshold_ev == 15.6
    assert ion.columns == "Energy (eV) | Cross section (cm2)"
    assert ion.energy_ev.tolist() == [15.6, 20.0, 30.0, 50.0, 100.0, 200.0, 1000.0]
    np.testing.assert_allclose(
        ion.sigma_m2,
        np.array([0.0, 3.0e-17, 1.0e-16, 1.9e-16, 2.5e-16, 2.2e-16, 1.0e-16]) * 1e-4,
        rtol=1e-15,
    )

    att = by[221]
    assert (att.kind, att.target, att.product) == ("ATTACHMENT", "N2", "N2^-")
    assert att.threshold_ev == 0.0 and "threshold_line" not in att.param
    assert att.projectile == "e"


def test_full_fixture_typeless_ion_blocks():
    by = _by_line(parse_lxcat_document(_full_text()))
    iso, back = by[237], by[248]
    for cs, kind in ((iso, "ISOTROPIC"), (back, "BACKSCAT")):
        assert cs.kind == kind
        assert (cs.projectile, cs.target, cs.product) == ("Ar^+", "Ar", None)
        assert cs.species == "Ar^+ / Ar"
        assert cs.threshold_ev == 0.0 and cs.mass_ratio is None
        assert cs.param == {"Mi": 39.948, "Mi/M": 1.0, "complete set": True}
        assert cs.database == DB_B
    assert iso.sigma_m2.tolist() == [3e-19, 3e-19, 2e-19]
    assert back.process == "Ar+ + Ar -> Ar + Ar+, Backscat"


# ---- 2. 拡張と警告 ------------------------------------------------------------------


def _one(text: str) -> tuple[CrossSection, list[str]]:
    doc = parse_lxcat_document(text)
    (cs,) = doc.all_cross_sections()
    return cs, doc.warnings


def test_attachment_threshold_line_extension():
    cs, w = _one("ATTACHMENT\nO2 -> O^-\n 4.4 ! dissociative\n-----\n 4.4 0\n 10 1e-22\n-----\n")
    assert cs.threshold_ev == 4.4
    assert cs.param == {"threshold_line": True}
    assert cs.sigma(4.0) == 0.0 and w == []
    # 3 行目が "KEY: value" (コロンあり) なら閾値行ではない (LXCat 標準)
    cs, _ = _one("ATTACHMENT\nO2\nSPECIES: e / O2\n-----\n 0 0\n 10 1e-22\n-----\n")
    assert cs.threshold_ev == 0.0 and "threshold_line" not in cs.param


def test_parameter_line_extra_tokens():
    cs, _ = _one("IONIZATION\nAr -> Ar^+\n 15.76 abc 3 ! note\n-----\n 15.76 0\n 20 1e-20\n-----\n")
    assert cs.threshold_ev == 15.76
    assert cs.param == {"extra_tokens": "abc 3"}
    cs, _ = _one("EXCITATION\nAr -> Ar*\n 11.5 2.0 7.0 # c\n-----\n 11.5 0\n 20 1e-20\n-----\n")
    assert (cs.threshold_ev, cs.weight_ratio) == (11.5, 2.0)
    assert cs.param == {"extra_tokens": "7.0"}


def test_param_mismatch_warnings_and_mass_ratio_from_param():
    text = (
        "ELASTIC\nAr\n 1.36e-5\nPARAM.: m/M = 1.40e-5\n-----\n 0 1e-20\n 1 1e-20\n-----\n"
        "EXCITATION\nAr -> Ar*\n 11.55\nPARAM.: E = 11.60 eV\n-----\n 11.55 0\n 20 1e-20\n-----\n"
        "ELASTIC\nKr\nPARAM.: m/M = 6.5e-6, complete set\n-----\n 0 1e-20\n 1 1e-20\n-----\n"
        "EXCITATION\nKr -> Kr*\n 9.9\nPARAM.: E = 9900 meV\n-----\n 9.9 0\n 20 1e-20\n-----\n"
    )
    doc = parse_lxcat_document(text)
    el_ar, exc_ar, el_kr, exc_kr = doc.all_cross_sections()
    assert el_ar.mass_ratio == 1.36e-5    # パラメータ行を優先
    assert exc_ar.threshold_ev == 11.55
    assert el_kr.mass_ratio == 6.5e-6     # m/M 行が無いので PARAM の m/M を採用
    assert el_kr.param["mass_ratio_from_param"] is True
    assert exc_kr.param["E"] == pytest.approx(9.9)  # meV → eV 換算で一致 (警告なし)
    assert len(doc.warnings) == 2
    assert doc.warnings[0].startswith("1 行目") and "m/M" in doc.warnings[0]
    assert doc.warnings[1].startswith("9 行目") and "閾値" in doc.warnings[1]


def test_param_parsing_values():
    cs, _ = _one(
        "EXCITATION\nAr -> Ar*\n 11.55\n"
        "PARAM.: E = 11.55 eV, g1/g0 = 3, note = see text, complete set, Q = 1e-20 m2\n"
        "-----\n 11.55 0\n 20 1e-20\n-----\n"
    )
    assert cs.param == {
        "E": 11.55, "g1/g0": 3.0, "note": "see text", "complete set": True, "Q": 1e-20,
    }


def test_species_target_mismatch_warning_and_projectile():
    cs, w = _one("ELASTIC\nAr\n 1.36e-5\nSPECIES: e / Kr\n-----\n 0 1e-20\n 1 1e-20\n-----\n")
    assert cs.target == "Ar"  # 2 行目を優先
    assert len(w) == 1 and w[0].startswith("1 行目") and "Kr" in w[0]
    cs, w = _one("ELASTIC\nAr\n 1.36e-5\n-----\n 0 1e-20\n 1 1e-20\n-----\n")
    assert cs.projectile == "e" and cs.species == "" and w == []  # SPECIES 行なし → 電子


@pytest.mark.parametrize(
    ("columns", "e_factor", "s_factor"),
    [
        ("Energy (eV) | Cross section (m2)", 1.0, 1.0),
        ("Energy (eV) | Cross section (m^2)", 1.0, 1.0),
        ("Energy (meV) | Cross section (cm2)", 1e-3, 1e-4),
        ("Energy (keV) | Cross section (cm^2)", 1e3, 1e-4),
        ("Energy (eV) | Cross section (1e-20 m2)", 1.0, 1e-20),
        ("Energy (eV) | Cross section (10^-20 m2)", 1.0, 1e-20),
        ("Energy (eV) | Cross section (Å2)", 1.0, 1e-20),
        ("Energy (eV) | Cross section (A2)", 1.0, 1e-20),
        ("Energy (eV) | Cross section (1e-16 cm2)", 1.0, 1e-20),
    ],
)
def test_columns_unit_conversion(columns, e_factor, s_factor):
    cs, w = _one(
        f"ELASTIC\nAr\n 1.36e-5\nCOLUMNS: {columns}\n-----\n 0 1.5\n 2 2.5\n 4 3.0\n-----\n"
    )
    assert w == []
    np.testing.assert_allclose(cs.energy_ev, np.array([0.0, 2.0, 4.0]) * e_factor, rtol=1e-15)
    np.testing.assert_allclose(cs.sigma_m2, np.array([1.5, 2.5, 3.0]) * s_factor, rtol=1e-15)
    assert cs.columns == columns


def test_columns_unknown_unit_warns_and_assumes_si():
    cs, w = _one(
        "ELASTIC\nAr\n 1.36e-5\nCOLUMNS: Energy (Ry) | Cross section (barn)\n"
        "-----\n 0 1e-20\n 1 2e-20\n-----\n"
    )
    assert cs.energy_ev.tolist() == [0.0, 1.0] and cs.sigma_m2.tolist() == [1e-20, 2e-20]
    assert len(w) == 2 and all(x.startswith("4 行目") for x in w)
    assert "Ry" in w[0] and "barn" in w[1]


def test_four_column_table_warns_and_keeps_three():
    cs, w = _one("ELASTIC\nAr\n 1e-5\n-----\n 0 1e-20 2e-20 5\n 1 2e-20 3e-20\n-----\n")
    assert cs.sigma_mt_m2 is not None and cs.sigma_mt_m2.tolist() == [2e-20, 3e-20]
    assert len(w) == 1 and w[0].startswith("4 行目") and "4 列" in w[0]


def test_typeless_unknown_process_is_skipped_with_warning():
    doc = parse_lxcat_document(
        "SPECIES: Ar^+ / Ar\nPROCESS: Ar+ + Ar -> Ar+ + Ar, Mystery.\n-----\n 0 1\n 1 x\n-----\n"
        "SPECIES: Ar^+ / Ar\nPROCESS: Ar+ + Ar -> Ar + Ar+, Backscattering.\n-----\n 0 1e-19\n 1 2e-19\n-----\n"
    )
    (cs,) = doc.all_cross_sections()  # 未知の種別のテーブル ("x" を含む) は読み飛ばす
    assert cs.kind == "BACKSCAT" and cs.line == 7
    assert len(doc.warnings) == 1 and doc.warnings[0].startswith("1 行目")
    assert "mystery" in doc.warnings[0]


def test_species_line_before_keyword_block_is_ignored_like_boltzpmp():
    """SPECIES 行の後にテーブルが無いまま標準ブロックが始まっても、そのブロックは失わない。"""
    doc = parse_lxcat_document("SPECIES: e / Ar\nELASTIC\nAr\n 1.36e-5\n-----\n 0 1e-20\n 1 1e-20\n-----\n")
    (cs,) = doc.all_cross_sections()
    assert cs.kind == "ELASTIC" and cs.line == 2 and cs.mass_ratio == 1.36e-5
    assert len(doc.warnings) == 1 and doc.warnings[0].startswith("1 行目")


def test_database_switching_and_default_database():
    block = "ELASTIC\n{t}\n 1e-5\n-----\n 0 1e-20\n 1 1e-20\n-----\n"
    doc = parse_lxcat_document(
        "DATABASE: A\n" + block.format(t="Ar") + "DATABASE: B\n" + block.format(t="N2")
        + "DATABASE: A\n" + block.format(t="Kr")
    )
    assert [db.name for db in doc.databases] == ["A", "B"]  # 同名の DATABASE は同じ DB に戻る
    assert [cs.target for cs in doc.databases[0].cross_sections] == ["Ar", "Kr"]
    assert [cs.target for cs in doc.all_cross_sections()] == ["Ar", "N2", "Kr"]  # 出現順
    doc = parse_lxcat_document("PERMLINK: x\n" + block.format(t="Ar"))
    assert len(doc.databases) == 1 and doc.databases[0].name is None
    assert doc.databases[0].meta == {"PERMLINK": "x"}
    assert doc.all_cross_sections()[0].database is None


def test_keyword_is_case_sensitive_and_number_grammar_matches_rust():
    # 小文字のキーワードはブロックにならない (boltzpmp と同じ) → ブロック無しで ValueError
    with pytest.raises(ValueError, match="ブロックが見つかりません"):
        parse_lxcat_document("elastic\nAr\n 1e-5\n-----\n 0 1e-20\n-----\n")
    # "1_0e-5" (Python の float は読むが Rust の f64 は読まない) は m/M 行として消費しない
    cs, _ = _one("ELASTIC\nAr\n 1_0e-5\n-----\n 0 1e-20\n 1 1e-20\n-----\n")
    assert cs.mass_ratio is None and cs.comment == "1_0e-5"


def test_no_blocks_raises():
    with pytest.raises(ValueError, match="ブロックが見つかりません"):
        parse_lxcat_document("これは LXCat ファイルではありません\n1.0 2.0\n")
    with pytest.raises(ValueError):
        parse_lxcat_document("")


# ---- 3. 構造エラー (行番号付き ValueError) ---------------------------------------------


@pytest.mark.parametrize(
    ("text", "line"),
    [
        # 区切りなし (テーブル開始区切りが無いままファイル末尾)
        ("EXCITATION\nAr -> Ar*\n 11.5\nPROCESS: x\n 11.5 0\n 20 1e-20\n", 1),
        # 終了区切りなし
        ("ELASTIC\nAr\n 1e-5\n-----\n 0 1e-20\n 1 2e-20\n", 4),
        # テーブルより先に次のブロック (boltzpmp は次のブロックのテーブルを誤読する)
        ("EXCITATION\nAr\n 11.5\nIONIZATION\nAr\n 15.7\n-----\n 15.7 0\n-----\n", 1),
        # 表が空
        ("ELASTIC\nAr\n 1e-5\n-----\n\n-----\n", 4),
        # 列数不一致
        ("ELASTIC\nAr\n 1e-5\n-----\n 0 1e-20 1e-20\n 1 2e-20\n-----\n", 6),
        # エネルギー減少
        ("ELASTIC\nAr\n 1e-5\n-----\n 0 1e-20\n 2 2e-20\n 1 2e-20\n-----\n", 7),
        # 負の σ・負の運動量移行・負のエネルギー
        ("ELASTIC\nAr\n 1e-5\n-----\n 0 1e-20\n 1 -2e-20\n-----\n", 6),
        ("ELASTIC\nAr\n 1e-5\n-----\n 0 1e-20 1e-20\n 1 2e-20 -1e-20\n-----\n", 6),
        ("ELASTIC\nAr\n 1e-5\n-----\n -1 1e-20\n 1 2e-20\n-----\n", 5),
        # 数値でない行・有限でない値・1 列の行
        ("ELASTIC\nAr\n 1e-5\n-----\n 0 1e-20\n 1 abc\n-----\n", 6),
        ("ELASTIC\nAr\n 1e-5\n-----\n 0 1e-20\n 1 nan\n-----\n", 6),
        ("ELASTIC\nAr\n 1e-5\n-----\n 0 1e-20\n 1\n-----\n", 6),
        # m/M 範囲外
        ("ELASTIC\nAr\n 1.5\n-----\n 0 1e-20\n-----\n", 3),
        ("EFFECTIVE\nAr\n 0.0\n-----\n 0 1e-20\n-----\n", 3),
        # 閾値欠落 (boltzpmp と同じ 3 行目)・負の閾値・統計重み比が読めない/非正
        ("EXCITATION\nAr -> Ar*\nPROCESS: x\n-----\n 1 0\n-----\n", 3),
        ("IONIZATION\nAr -> Ar^+\n-----\n 16 0\n-----\n", 3),
        ("EXCITATION\nAr -> Ar*\nPARAM.: E = -1 eV\n-----\n 1 0\n 2 1e-20\n-----\n", 3),
        ("EXCITATION\nAr -> Ar*\n -0.5\n-----\n 0 0\n 1 1e-20\n-----\n", 3),
        ("EXCITATION\nAr <-> Ar*\n 11.5 abc\n-----\n 11.5 0\n 20 1e-20\n-----\n", 3),
        ("EXCITATION\nAr <-> Ar*\n 11.5 0\n-----\n 11.5 0\n 20 1e-20\n-----\n", 3),
        # ROTATION: 上下準位の逆転・重みの欠落・非正の重み
        ("ROTATION\nN2\n 0.0015 5\n 0.0 1\n-----\n 0.0015 0\n 1 1e-20\n-----\n", 4),
        ("ROTATION\nN2\n 0.0 1\n 0.0015\n-----\n 0.0015 0\n 1 1e-20\n-----\n", 4),
        ("ROTATION\nN2\n 0.0 0\n 0.0015 5\n-----\n 0.0015 0\n 1 1e-20\n-----\n", 3),
        # 反応式の行なし・PROCESS の無い (値が空を含む) タイプ行なしブロック
        ("ELASTIC\n\n 1e-5\n-----\n 0 1e-20\n-----\n", 1),
        ("header\nSPECIES: Ar^+ / Ar\nCOMMENT: x\n-----\n 0 1e-19\n-----\n", 2),
        ("SPECIES: Ar^+ / Ar\nPROCESS:\n-----\n 0 1e-19\n-----\n", 1),
    ],
)
def test_structural_errors_have_line_numbers(text, line):
    with pytest.raises(ValueError, match=rf"^{line} 行目"):
        parse_lxcat_document(text)


# ---- 4. boltzpmp (bp.parse_lxcat) との一致 -------------------------------------------

# boltzpmp が扱う標準ブロックの細部 (小文字キー・# と ! のコメント・余分なトークン・段差・
# 空行・3 列/4 列・m/M 行の省略・PROCESS の無いブロックの既定名・末尾空白のキーワード行)
_BP_STANDARD = """\
header text is skipped
ELASTIC
Ar
 1.371000e-5
process: E + Ar -> E + Ar, Elastic
comment: first comment
COMMENT: second comment
-----
 0.0 7.5e-20
 1.0 1.0e-20

 10.0 1.5e-19
-----
EFFECTIVE\x20\x20\x20
Ar
 1.4e-5 # trailing comment
-----
 0 2e-19
 100 2e-19
-----
EXCITATION
HF(J=0) -> HF(J=1)
 5.126e-3  3.0   7.0
PROCESS: rot 0-1
-----
 5.126e-3 0.0
 1.0 2.0e-19
 1.0 2.5e-19
 2.0 2.0e-19
-----
EXCITATION
HF <-> HF(v1)
 4.912e-1 ! comment 2.0
-----
 4.912e-1 0.0 0.0
 2.0 1.0e-20 4.0e-21
-----
ROTATION
HF
 0.0 1.0
 5.126e-3 3.0
-----
 5.126e-3 0.0
 1.0 2.0e-19
-----
IONIZATION
HF -> HF^+
 15.9  0.0  extra
-----
 15.9 0
 100 1e-20
-----
ATTACHMENT
HF -> H + F^-
COMMENT: attachment
-----
 2.5 0.0
 3.0 1.0e-23
-----
ELASTIC
Kr
PARAM.: none
-----
 0 1e-20 2e-20 7
 1 2e-20 3e-20 8
-----
"""


def _eval_grid(*css) -> np.ndarray:
    pts = [np.linspace(0.0, 1200.0, 4001), np.geomspace(1e-4, 1e3, 2001)]
    for cs in css:
        pts += [cs.energy_ev, np.nextafter(cs.energy_ev, -np.inf), np.nextafter(cs.energy_ev, np.inf)]
        pts.append(np.array([cs.threshold_ev]))
    return np.unique(np.concatenate(pts))


def _assert_core_equal(ours, theirs, skip=()):
    a = to_boltzpmp_cross_section(ours).to_core()
    b = theirs.to_core()
    assert a.keys() == b.keys()
    for key in a:
        if key not in skip:
            assert a[key] == b[key], (ours.label, key, a[key], b[key])


@pytest.mark.parametrize("variant", ["lf", "bom_crlf"])
def test_matches_boltzpmp_on_standard_blocks(variant):
    bp = pytest.importorskip("boltzpmp")
    assert "\nEFFECTIVE   \n" in _BP_STANDARD  # 末尾空白付きのキーワード行
    text = _BP_STANDARD if variant == "lf" else "﻿" + _BP_STANDARD.replace("\n", "\r\n")
    doc = parse_lxcat_document(text)
    ours = doc.all_cross_sections()
    theirs = bp.parse_lxcat(text)
    assert [c.kind for c in ours] == [c.kind for c in theirs]
    for c, b in zip(ours, theirs):
        # to_boltzpmp_cross_section の結果が bp.parse_lxcat と全フィールド一致
        # (kind・反応式・過程名・閾値・m/M・重み比・準位・(E, σ) 表・mt・コメント)
        _assert_core_equal(c, b)
        assert c.target == b.target and c.product == b.product
        assert c.reversible == ("<->" in b.species)
        eps = _eval_grid(c)
        np.testing.assert_array_equal(c.sigma(eps), b.sigma(eps))  # 評価規約もビット一致
    rot = ours[4]
    assert rot.weight_ratio == 3.0 and theirs[4].weight_ratio is None  # ROTATION は準位から
    # 拡張部分の警告 (4 列の表) 以外は警告なし
    assert len(doc.warnings) == 1 and "4 列" in doc.warnings[0]


@pytest.mark.parametrize("name", ["Ar.txt", "Ar_star.txt"])
def test_matches_boltzpmp_on_its_bundled_data(name):
    """boltzpmp 同梱の検証用データ (インストール済みパッケージから読むだけでリポジトリには
    入れない) でも全フィールドが一致する。"""
    bp = pytest.importorskip("boltzpmp")
    path = Path(bp.__file__).parent / "data" / name
    if not path.exists():
        pytest.skip(f"boltzpmp に {name} が同梱されていません")
    doc = load_lxcat(path)
    ours = doc.all_cross_sections()
    theirs = bp.parse_lxcat(path)
    assert doc.warnings == [] and len(ours) == len(theirs) > 0
    for c, b in zip(ours, theirs):
        _assert_core_equal(c, b)


def _strip_block(text: str, keyword_line: int) -> str:
    """keyword_line (1 始まり) のブロックをテーブルの終了区切りまで取り除く。"""
    lines = text.split("\n")
    i = keyword_line - 1
    dashes = [k for k in range(i, len(lines)) if lines[k].strip().startswith("-----")]
    return "\n".join(lines[:i] + lines[dashes[1] + 1:])


def test_matches_boltzpmp_on_full_fixture():
    """合成フィクスチャ全体: boltzpmp が読まない拡張部分 (MOMENTUM・タイプ行なし) は除き、
    PARAM 由来閾値のブロック (boltzpmp はエラー) は取り除いたテキストで比較する。"""
    bp = pytest.importorskip("boltzpmp")
    text = _full_text()
    doc = parse_lxcat_document(text)
    param_block = [cs for cs in doc.all_cross_sections() if cs.param.get("threshold_from_param")]
    assert [cs.line for cs in param_block] == [183]
    with pytest.raises(ValueError, match="line 185"):  # boltzpmp は閾値行の欠落をエラーにする
        bp.parse_lxcat(text)
    theirs = bp.parse_lxcat(_strip_block(text, 183))

    ours = [
        cs for cs in doc.all_cross_sections()
        if cs.projectile == "e"
        and "keyword_alias" not in cs.param
        and "threshold_from_param" not in cs.param
    ]
    assert [c.kind for c in ours] == [c.kind for c in theirs]
    assert len(ours) == 8
    for c, b in zip(ours, theirs):
        skip: tuple[str, ...] = ()
        if c.columns and "cm2" in c.columns:
            # boltzpmp は COLUMNS の単位を見ない: 表の数値は cm² のまま
            skip = ("sigma",)
            np.testing.assert_allclose(c.sigma_m2, b.data[:, 1] * 1e-4, rtol=1e-15)
            eps = _eval_grid(c)
            np.testing.assert_allclose(c.sigma(eps), b.sigma(eps) * 1e-4, rtol=1e-12)
        else:
            eps = _eval_grid(c)
            np.testing.assert_array_equal(c.sigma(eps), b.sigma(eps))
        if "\n" in c.comment and c.line == 21:
            skip += ("comment",)  # 自由記述行は拡張 (comment に追記)
            assert c.comment.startswith(b.comment + "\n")
        _assert_core_equal(c, b, skip=skip)


# ---- 5. 改行・BOM・文字コード --------------------------------------------------------


def test_crlf_bom_and_cr_variants_give_identical_documents():
    text = _full_text()
    ref = parse_lxcat_document(text).to_dict()
    for variant in (
        text.replace("\n", "\r\n"),
        "﻿" + text,
        "﻿" + text.replace("\n", "\r\n"),
        text.replace("\n", "\r"),  # 旧 Mac (CR のみ)
    ):
        assert parse_lxcat_document(variant).to_dict() == ref


def test_load_lxcat_encodings(tmp_path):
    text = _full_text()
    ref = parse_lxcat_document(text).to_dict()
    p = tmp_path / "utf8sig_crlf.txt"
    p.write_bytes(text.replace("\n", "\r\n").encode("utf-8-sig"))
    assert load_lxcat(p).to_dict() == ref
    p = tmp_path / "utf16.txt"
    p.write_bytes(text.encode("utf-16"))  # BOM 付き
    assert load_lxcat(str(p)).to_dict() == ref
    # UTF-8 として不正なバイト列 (Latin-1 の Å = 0xC5) は Latin-1 で読む
    latin = text.replace("COMMENT: synthetic ionization.", "COMMENT: synthetic ionization (Å).")
    p = tmp_path / "latin1.txt"
    p.write_bytes(latin.encode("latin-1"))
    doc = load_lxcat(p)
    assert _by_line(doc)[74].comment == "synthetic ionization (Å)."


# ---- 6. sigma() の below/above -----------------------------------------------------


def _step_cs() -> CrossSection:
    # 閾値 2 eV が表の内側、3 eV に段差 (2.0 → 4.0)
    return CrossSection(
        kind="EXCITATION", projectile="e", target="Ar", threshold_ev=2.0,
        energy_ev=[1.0, 3.0, 3.0, 5.0], sigma_m2=[1.0, 2.0, 4.0, 6.0],
    )


def test_sigma_below_zero_above_clamp_default():
    cs = _step_cs()
    eps = np.array([0.5, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0])
    expected = np.array([0.0, 0.0, 1.5, 1.75, 4.0, 5.0, 6.0, 6.0])
    np.testing.assert_allclose(cs.sigma(eps), expected, rtol=1e-15)
    assert cs.sigma(3.0 - 1e-12) == pytest.approx(2.0)  # 段差の直前は前側の値
    assert isinstance(cs.sigma(2.5), float)
    grid = np.array([[0.5, 2.5], [3.0, 7.0]])
    assert cs.sigma(grid).shape == (2, 2)


def test_sigma_below_clamp_is_np_interp_and_above_zero():
    cs = _step_cs()
    eps = np.linspace(0.0, 8.0, 161)
    np.testing.assert_array_equal(
        cs.sigma(eps, below="clamp"), np.interp(eps, cs.energy_ev, cs.sigma_m2)
    )
    assert cs.sigma(1.5, below="clamp") == 1.25  # 閾値マスクもしない (v1 MCC 互換)
    assert cs.sigma(0.5, below="clamp") == 1.0
    assert cs.sigma(7.0, above="zero") == 0.0
    assert cs.sigma(5.0, above="zero") == 6.0   # 最後の点ちょうどは表の値
    assert cs.sigma(0.5, below="clamp", above="zero") == 1.0
    with pytest.raises(ValueError):
        cs.sigma(1.0, below="nearest")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        cs.sigma(1.0, above="extrapolate")  # type: ignore[arg-type]


def test_sigma_matches_boltzpmp_evaluation():
    bp = pytest.importorskip("boltzpmp")
    cs = _step_cs()
    b = to_boltzpmp_cross_section(cs)
    assert isinstance(b, bp.CrossSection)
    eps = _eval_grid(cs)
    np.testing.assert_array_equal(cs.sigma(eps), b.sigma(eps))


# ---- 7. to_dict()/from_dict() の往復 ------------------------------------------------


def test_cross_section_dict_round_trip():
    doc = parse_lxcat_document(_full_text())
    for cs in doc.all_cross_sections():
        d = json.loads(json.dumps(cs.to_dict()))  # JSON 化できること
        assert d["label"] == cs.label
        back = CrossSection.from_dict(d)
        assert back.to_dict() == cs.to_dict()
        np.testing.assert_array_equal(back.energy_ev, cs.energy_ev)
        np.testing.assert_array_equal(back.sigma_m2, cs.sigma_m2)
        assert (back.sigma_mt_m2 is None) == (cs.sigma_mt_m2 is None)
        assert back.lower_state == cs.lower_state and back.upper_state == cs.upper_state
        assert back.energy_ev.dtype == np.float64
    rot = _by_line(doc)[147].to_dict()
    assert rot["lower_state"] == [0.0, 1.0] and rot["upper_state"] == [1.488e-3, 5.0]


def test_cross_section_value_equality():
    css = parse_lxcat_document(_full_text()).all_cross_sections()
    copy = CrossSection.from_dict(css[1].to_dict())
    assert copy == css[1] and copy is not css[1]
    assert copy in css            # ndarray の比較で例外にならない
    assert css.index(copy) == 1
    assert copy != css[2]
    copy.sigma_m2[-1] *= 2.0
    assert copy != css[1]
    assert (css[0] == "ELASTIC") is False


def test_document_dict_round_trip():
    doc = parse_lxcat_document(_full_text())
    d = json.loads(json.dumps(doc.to_dict()))
    back = LxcatDocument.from_dict(d)
    assert back.to_dict() == doc.to_dict()
    assert back.warnings == doc.warnings


def test_from_dict_validates_tables():
    d = _by_line(parse_lxcat_document(_full_text()))[52].to_dict()
    d["energy_ev"] = d["energy_ev"][::-1]
    with pytest.raises(ValueError, match="非減少"):
        CrossSection.from_dict(d)
    with pytest.raises(ValueError, match="未知の断面積種別"):
        CrossSection.from_dict({**d, "kind": "VIBRATION"})
    rot = _by_line(parse_lxcat_document(_full_text()))[147].to_dict()
    with pytest.raises(ValueError, match="lower_state"):
        CrossSection.from_dict({**rot, "lower_state": [0.0]})


# ---- 8. エンドポイント ---------------------------------------------------------------


def test_v2_xs_parse_endpoint():
    client = TestClient(app)
    r = client.post("/v2/xs/parse", json={"text": _full_text()})
    assert r.status_code == 200
    body = r.json()
    assert [db["name"] for db in body["databases"]] == [DB_A, DB_B]
    assert [len(db["cross_sections"]) for db in body["databases"]] == [3, 9]
    assert body["databases"][0]["meta"]["PERMLINK"] == "www.example.invalid/SynthA"
    first = body["databases"][0]["cross_sections"][0]
    assert first["kind"] == "ELASTIC" and first["line"] == 21
    assert first["label"] == "E + Ar -> E + Ar, Elastic"
    assert first["param"] == {"m/M": 1.36e-5, "complete set": True}  # bool は bool のまま
    assert first["mass_ratio"] == 1.36e-5 and first["product"] is None
    assert len(first["sigma_mt_m2"]) == 17
    rot = body["databases"][1]["cross_sections"][2]
    assert rot["kind"] == "ROTATION" and rot["upper_state"] == [1.488e-3, 5.0]
    assert body["databases"][1]["cross_sections"][0]["param"] == {"keyword_alias": "MOMENTUM"}

    targets = body["targets"]
    assert [(t["projectile"], t["name"]) for t in targets] == [("e", "Ar"), ("e", "N2"), ("Ar^+", "Ar")]
    ar, n2, ion = targets
    assert ar["counts"] == {"ELASTIC": 1, "EXCITATION": 1, "IONIZATION": 1, "EFFECTIVE": 1}
    assert ar["momentum_transfer"] == "elastic"
    # 分子量は質量表を優先 (LXCat の m/M は丸められていることが多いため。xs.convert.component_mass_amu)
    assert ar["mass_amu"] == pytest.approx(39.948)
    assert n2["counts"] == {
        "EFFECTIVE": 1, "ROTATION": 1, "EXCITATION": 2, "IONIZATION": 1, "ATTACHMENT": 1,
    }
    assert n2["momentum_transfer"] == "effective"
    assert n2["mass_amu"] == pytest.approx(28.014, rel=1e-6)
    assert ion["counts"] == {"ISOTROPIC": 1, "BACKSCAT": 1}
    assert ion["momentum_transfer"] == "none"
    assert ion["mass_amu"] == pytest.approx(39.948)
    assert len(body["warnings"]) == 1 and body["warnings"][0].startswith("185 行目")


def test_v2_xs_parse_endpoint_errors_are_422():
    client = TestClient(app)
    r = client.post("/v2/xs/parse", json={"text": "no blocks here"})
    assert r.status_code == 422
    r = client.post("/v2/xs/parse", json={"text": "ELASTIC\nAr\n 1e-5\n-----\n 0 1e-20\n 1 abc\n-----\n"})
    assert r.status_code == 422
    assert r.json()["detail"].startswith("6 行目")


def test_lxcat_endpoint_uses_new_parser_on_full_fixture():
    """/lxcat/parse (v1 互換): 新パーサー経由で全フィクスチャを読める。"""
    client = TestClient(app)
    r = client.post("/lxcat/parse", json={"text": _full_text(), "species": "electron"})
    assert r.status_code == 200
    body = r.json()
    assert [p["kind"] for p in body["processes"]] == [
        "elastic", "excitation", "ionization",                  # Ar (MOMENTUM は除外)
        "elastic", "excitation", "excitation", "excitation", "ionization",  # N2 (EFFECTIVE→elastic)
    ]
    assert body["processes"][3]["label"] == "E + N2 -> E + N2, Effective"
    assert body["processes"][4]["threshold_ev"] == pytest.approx(1.488e-3)  # ROTATION
    # 3 列の ELASTIC は運動量移行断面積 (3 列目) を v1 の elastic にする (等方散乱の v1 MCC 用)
    ar_el = _by_line(parse_lxcat_document(_full_text()))[21]
    assert body["processes"][0]["sigma_m2"] == ar_el.sigma_mt_m2.tolist()
    assert body["processes"][0]["mass_ratio"] == 1.36e-5
    joined = "\n".join(body["warnings"])
    for key in (
        "EFFECTIVE", "ATTACHMENT", "ROTATION", "<->", "単一ガス", "PARAM", "ISOTROPIC", "BACKSCAT",
        "3 列目",
    ):
        assert key in joined, key

    r = client.post("/lxcat/parse", json={"text": _full_text(), "species": "ion"})
    assert r.status_code == 200
    assert [p["kind"] for p in r.json()["processes"]] == ["isotropic", "backscat"]

    r = client.post("/lxcat/parse", json={"text": "ELASTIC\nAr\n 2.0\n-----\n 0 1\n-----\n", "species": "electron"})
    assert r.status_code == 422
    assert r.json()["detail"].startswith("3 行目")
