"""v2 直交格子 + 埋め込み境界 (EB、カットセル) の離散化 (prompts/119)。"""

from __future__ import annotations

from .build import LevelOperator, build_level
from .grid import CartesianGrid, choose_cells, make_grid

__all__ = ["CartesianGrid", "LevelOperator", "build_level", "choose_cells", "make_grid"]
