"""断面積パッケージ (v2 フェーズ P1、prompts/120)。

LXCat 形式の断面積ファイルを boltzpmp のパーサーの上位互換で読み (`lxcat`)、全ソルバー
(v1 MCC・流体・Boltzmann、将来の GPU MCC) が共通に使う断面積モデル (`model`) にする。
EFFECTIVE→ELASTIC 変換・混合ガス・v1 XsProcess / boltzpmp への変換は `convert`、
分子量テーブルは `masses`。POST /v2/xs/parse のモデルは `api` (pydantic) にある。
"""

from __future__ import annotations

from .convert import (
    boltzpmp_name,
    build_mixture,
    component_mass_amu,
    effective_to_elastic,
    resolve_momentum_transfer,
    to_boltzpmp_cross_section,
    to_boltzpmp_mixture,
    to_v1_processes,
)
from .lxcat import (
    KEYWORD_ALIASES,
    KEYWORDS,
    decode_lxcat_bytes,
    load_lxcat,
    parse_lxcat_document,
    parse_param,
    split_reaction,
)
from .masses import (
    MOLECULAR_MASS_AMU,
    lookup_mass_amu,
    mass_amu_from_ratio,
    mass_ratio_from_amu,
)
from .model import (
    ELECTRON_KINDS,
    INELASTIC_KINDS,
    ION_KINDS,
    KINDS,
    MOMENTUM_KINDS,
    CrossSection,
    GasComponent,
    GasMixture,
    Kind,
    LxcatDatabase,
    LxcatDocument,
)

__all__ = [
    "CrossSection",
    "ELECTRON_KINDS",
    "GasComponent",
    "GasMixture",
    "INELASTIC_KINDS",
    "ION_KINDS",
    "KEYWORDS",
    "KEYWORD_ALIASES",
    "KINDS",
    "Kind",
    "LxcatDatabase",
    "LxcatDocument",
    "MOLECULAR_MASS_AMU",
    "MOMENTUM_KINDS",
    "boltzpmp_name",
    "build_mixture",
    "component_mass_amu",
    "decode_lxcat_bytes",
    "effective_to_elastic",
    "load_lxcat",
    "lookup_mass_amu",
    "mass_amu_from_ratio",
    "mass_ratio_from_amu",
    "parse_lxcat_document",
    "parse_param",
    "resolve_momentum_transfer",
    "split_reaction",
    "to_boltzpmp_cross_section",
    "to_boltzpmp_mixture",
    "to_v1_processes",
]
