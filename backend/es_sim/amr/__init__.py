"""v2 AMR — AMReX 型のブロック構造局所細分化と合成格子ソルバー (prompts/121, 122)。"""

from __future__ import annotations

from .composite import CompositeOperator, assemble_composite, energy_and_charges, solve_composite
from .electrostatic import AmrStaticSolution, adapt_tags, build_hierarchy, display_mesh, solve_electrostatic_amr
from .fieldops import build_efield_stencil, build_second_difference
from .gpu_solver import AmgGpuSolver
from .hierarchy import AmrHierarchy, AmrSpec, dilate
from .pic_layout import AmrPicLayout, build_pic_layout

__all__ = [
    "AmgGpuSolver",
    "AmrHierarchy",
    "AmrPicLayout",
    "AmrSpec",
    "AmrStaticSolution",
    "CompositeOperator",
    "adapt_tags",
    "assemble_composite",
    "build_efield_stencil",
    "build_hierarchy",
    "build_pic_layout",
    "build_second_difference",
    "dilate",
    "display_mesh",
    "energy_and_charges",
    "solve_composite",
    "solve_electrostatic_amr",
]
