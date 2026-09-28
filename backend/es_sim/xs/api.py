"""POST /v2/xs/parse の入出力モデル (prompts/120)。

schema.py は v1 のプロジェクト JSON 用なので触らず、v2 断面積 API のモデルはこちらに置く。
cross_sections の各要素は CrossSection.to_dict() と同じ形 (表示用の label を含む)。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .convert import component_mass_amu
from .lxcat import parse_lxcat_document
from .model import MOMENTUM_KINDS, Kind, LxcatDocument


class XsParseRequest(BaseModel):
    """POST /v2/xs/parse のリクエスト (LXCat 形式テキスト)。"""

    text: str


class XsCrossSectionOut(BaseModel):
    """CrossSection.to_dict() の形 (単位は SI: eV, m²)。"""

    kind: Kind
    projectile: str
    target: str
    product: str | None
    reversible: bool
    threshold_ev: float
    mass_ratio: float | None
    weight_ratio: float | None
    lower_state: tuple[float, float] | None
    upper_state: tuple[float, float] | None
    energy_ev: list[float]
    sigma_m2: list[float]
    sigma_mt_m2: list[float] | None
    process: str
    species: str
    param: dict[str, float | str | bool]
    comment: str
    updated: str | None
    columns: str | None
    database: str | None
    line: int
    meta: dict[str, str]
    label: str


class XsDatabaseOut(BaseModel):
    name: str | None
    meta: dict[str, str]
    cross_sections: list[XsCrossSectionOut]


class XsTargetOut(BaseModel):
    """(入射粒子, 標的) ごとの要約。"""

    name: str
    projectile: str
    counts: dict[str, int]  # 種別 → ブロック数 (出現順)
    # 電子の運動量移行断面積の系統 (ELASTIC があれば "elastic"、EFFECTIVE のみなら "effective")
    momentum_transfer: Literal["elastic", "effective", "none"]
    mass_amu: float | None  # 質量表 → m/M (M = m_e/(m/M)) の順 (convert.component_mass_amu)


class XsParseResponse(BaseModel):
    databases: list[XsDatabaseOut]
    targets: list[XsTargetOut]
    warnings: list[str]


def summarize_targets(doc: LxcatDocument) -> list[XsTargetOut]:
    """文書の (入射粒子, 標的) ごとの種別数・運動量移行の系統・分子量 (出現順)。"""
    css = doc.all_cross_sections()
    keys: list[tuple[str, str]] = []
    for c in css:
        if (c.projectile, c.target) not in keys:
            keys.append((c.projectile, c.target))
    out: list[XsTargetOut] = []
    for projectile, target in keys:
        group = [c for c in css if c.projectile == projectile and c.target == target]
        counts: dict[str, int] = {}
        for c in group:
            counts[c.kind] = counts.get(c.kind, 0) + 1
        mt: Literal["elastic", "effective", "none"] = "none"
        if projectile == "e" and "ELASTIC" in counts:
            mt = "elastic"
        elif projectile == "e" and "EFFECTIVE" in counts:
            mt = "effective"
        momentum = [c for c in group if c.kind in MOMENTUM_KINDS]
        out.append(
            XsTargetOut(
                name=target,
                projectile=projectile,
                counts=counts,
                momentum_transfer=mt,
                mass_amu=component_mass_amu(target, momentum),
            )
        )
    return out


def build_parse_response(doc: LxcatDocument) -> XsParseResponse:
    data = doc.to_dict()
    return XsParseResponse(
        databases=data["databases"],
        targets=summarize_targets(doc),
        warnings=list(doc.warnings),
    )


def parse_xs_text(text: str) -> XsParseResponse:
    """LXCat 形式テキストを v2 断面積モデルで読み、API のレスポンスにする (失敗は ValueError)。"""
    return build_parse_response(parse_lxcat_document(text))
