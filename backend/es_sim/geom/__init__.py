"""v2 CAD 幾何 — 形状の解析的な問い合わせと、プロジェクト幾何の解釈 (prompts/119)。"""

from __future__ import annotations

from .model import EPS0, DirichletGroup, DomainRect, GeometryModel, RegionShape, SideBC
from .shapes import CircleShape, PolygonShape, Shape

__all__ = [
    "EPS0",
    "CircleShape",
    "DirichletGroup",
    "DomainRect",
    "GeometryModel",
    "PolygonShape",
    "RegionShape",
    "Shape",
    "SideBC",
]
