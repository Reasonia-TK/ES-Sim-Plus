"""代表的なガスの分子量テーブル [amu] と電子/標的の質量比の換算 (prompts/120)。

LXCat の ELASTIC/EFFECTIVE ブロックは 3 行目に m/M (電子質量/標的質量) を持つが省略も
許される。その場合や v1 XsProcess への変換 (mass_ratio 必須) で標的の質量が要るときの
補完に使う。値は標準原子量 (IUPAC) からの分子量で、断面積の文献値に付随する m/M と
有効数字 3〜4 桁で一致すれば十分な用途 (弾性衝突のエネルギー損失 2m/M) を想定する。

物理定数は CODATA 2018 (particles.ME・dsmc.AMU・boltzpmp.constants と同値) を使う。
boltzpmp の `mass_ratio_from_amu` と同じ式 m/M = m_e / (M·amu) なので、同じ分子量からは
ビット単位で同じ質量比になる。
"""

from __future__ import annotations

import re

ME_KG = 9.1093837015e-31      # 電子質量 [kg]
AMU_KG = 1.66053906660e-27    # 原子質量単位 [kg]
KB = 1.380649e-23             # ボルツマン定数 [J/K]

# 代表的なプラズマプロセス用ガスの分子量 [amu] (標準原子量からの和)
MOLECULAR_MASS_AMU: dict[str, float] = {
    # 希ガス
    "He": 4.002602,
    "Ne": 20.1797,
    "Ar": 39.948,
    "Kr": 83.798,
    "Xe": 131.293,
    # 原子・二原子分子
    "H": 1.008,
    "H2": 2.016,
    "D2": 4.0282,
    "N": 14.007,
    "N2": 28.014,
    "O": 15.999,
    "O2": 31.998,
    "F2": 37.997,
    "Cl2": 70.906,
    "CO": 28.010,
    "NO": 30.006,
    "HF": 20.006,
    "HCl": 36.461,
    "HBr": 80.912,
    # 三原子以上
    "O3": 47.997,
    "H2O": 18.015,
    "CO2": 44.009,
    "N2O": 44.013,
    "NO2": 46.005,
    "NH3": 17.031,
    "CH4": 16.043,
    "C2H2": 26.038,
    "C2H4": 28.054,
    "C2H6": 30.070,
    "C3H8": 44.097,
    "SiH4": 32.117,
    # フッ素・塩素系エッチングガス
    "CF4": 88.004,
    "CHF3": 70.014,
    "C2F6": 138.012,
    "C3F8": 188.020,
    "c-C4F8": 200.031,
    "C4F8": 200.031,
    "NF3": 71.002,
    "SF6": 146.055,
    "BCl3": 117.170,
    # 金属蒸気
    "Hg": 200.592,
    "Na": 22.990,
    "Cs": 132.905,
}

# 状態・電荷の装飾 ("Ar*", "N2(A3)", "Ar^+", "O2(a1Dg)" 等) を剥がすための正規表現
_STATE_SUFFIX = re.compile(r"\([^()]*\)")


def _base_species_name(name: str) -> str:
    """励起状態・電荷の表記を除いた基底の化学種名 ("Ar*(11.55eV)" → "Ar")。"""
    s = _STATE_SUFFIX.sub("", name).strip()
    return s.rstrip("*+-^'").strip()


def lookup_mass_amu(name: str) -> float | None:
    """ガス名から分子量 [amu] を引く (完全一致 → 状態/電荷表記を除いた名前の順)。

    見つからなければ None。大文字小文字は区別する ("CO" と "Co" を混同しないため)。
    """
    key = name.strip()
    if key in MOLECULAR_MASS_AMU:
        return MOLECULAR_MASS_AMU[key]
    base = _base_species_name(key)
    return MOLECULAR_MASS_AMU.get(base)


def mass_ratio_from_amu(mass_amu: float) -> float:
    """標的の分子量 [amu] から電子との質量比 m/M を求める (boltzpmp と同じ式)。"""
    if not mass_amu > 0.0:
        raise ValueError(f"分子量は正の値が必要です: {mass_amu}")
    return ME_KG / (mass_amu * AMU_KG)


def mass_amu_from_ratio(mass_ratio: float) -> float:
    """質量比 m/M から標的の分子量 M [amu] = m_e / (m/M) / amu を求める。"""
    if not 0.0 < mass_ratio < 1.0:
        raise ValueError(f"質量比 m/M は 0 < m/M < 1 が必要です: {mass_ratio}")
    return ME_KG / mass_ratio / AMU_KG
