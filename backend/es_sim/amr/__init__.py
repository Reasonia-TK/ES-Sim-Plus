"""v2 AMR — AMReX 型のブロック構造局所細分化と合成格子ソルバー (prompts/121)。"""

from __future__ import annotations

from .composite import CompositeOperator, assemble_composite, energy_and_charges, solve_composite
from .electrostatic import AmrStaticSolution, build_hierarchy, display_mesh, solve_electrostatic_amr
from .hierarchy import AmrHierarchy, AmrSpec, dilate

__all__ = [
    "AmrHierarchy",
    "AmrSpec",
    "AmrStaticSolution",
    "CompositeOperator",
    "assemble_composite",
    "build_hierarchy",
    "dilate",
    "display_mesh",
    "energy_and_charges",
    "solve_composite",
    "solve_electrostatic_amr",
]
