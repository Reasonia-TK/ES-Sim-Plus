"""v2 場ソルバー — 直交格子 + 埋め込み境界の節点 Poisson (GMG-PCG、CPU/GPU) (prompts/119)。"""

from __future__ import annotations

from .electrostatic import StaticSolution, fill_fixed, group_voltages, node_field, solve_electrostatic
from .gmg import GMGSolver, SolveInfo

__all__ = [
    "GMGSolver",
    "SolveInfo",
    "StaticSolution",
    "fill_fixed",
    "group_voltages",
    "node_field",
    "solve_electrostatic",
]
