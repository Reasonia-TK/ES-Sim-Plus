"""v2 流体 2D (直交格子 + 埋め込み境界、prompts/125)。

``make_fluid2d_simulation(project)`` が mesh.mode で実装を選ぶ:

- "cartesian": v2 の直交格子版 (CartesianFluid2dSimulation。Poisson は GMG、GPU があれば GPU)
- それ以外: v1 の三角形メッシュ版 (fluid2d.Fluid2dSimulation)
"""

from __future__ import annotations


def make_fluid2d_simulation(project, explicit: bool = False):
    if project.mesh.mode == "cartesian":
        from .simulation import CartesianFluid2dSimulation

        return CartesianFluid2dSimulation(project, explicit=explicit)
    from ..fluid2d import Fluid2dSimulation

    return Fluid2dSimulation(project, explicit=explicit)


def __getattr__(name: str):
    if name == "CartesianFluid2dSimulation":
        from .simulation import CartesianFluid2dSimulation

        return CartesianFluid2dSimulation
    raise AttributeError(name)
