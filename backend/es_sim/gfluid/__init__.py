"""v2 流体 2D (直交格子 + 埋め込み境界、prompts/125・126・128)。

``make_fluid2d_simulation(project)`` が mesh.mode で実装を選ぶ:

- "cartesian": v2 の直交格子版。mesh.amr で実際に細分化が起きるなら合成格子 (AMR) 版
  (amr.AmrFluid2dSimulation)、それ以外は一様格子版 (simulation.CartesianFluid2dSimulation)。
  CUDA が使えて未知数が多い (gpu.GPU_MIN_NODES 節点以上、AMR は葉セル数で見積もる) なら
  輸送も GPU で解く版 (GpuCartesianFluid2dSimulation / GpuAmrFluid2dSimulation)。
  陽的検証経路 (explicit)・linear_solver="direct" は常に CPU 版
- それ以外: v1 の三角形メッシュ版 (fluid2d.Fluid2dSimulation)
"""

from __future__ import annotations


def _fluid_mesh(project):
    """(AMR の階層、細分化が起きなければ None, 流体の未知数の見積もり)。"""
    from ..eb.grid import make_grid
    from ..geom.model import GeometryModel
    from .amr import amr_fluid_hierarchy

    model = GeometryModel(project)
    grid = make_grid(model.domain, float(project.mesh.size))
    hier = amr_fluid_hierarchy(project, model, grid)
    if hier is None:
        return None, grid.n_nodes
    return hier, sum(hier.leaf_cells(lvl)[0].size for lvl in range(hier.n_levels))


def _gpu_ok(n_nodes: int) -> bool:
    from ..device import get_device
    from .gpu import GPU_MIN_NODES

    return get_device().is_gpu and n_nodes >= GPU_MIN_NODES


def use_gpu_fluid(project) -> bool:
    """直交格子の流体を GPU 版で解くか (CUDA が使え、未知数が GPU_MIN_NODES 以上)。"""
    return _gpu_ok(_fluid_mesh(project)[1])


def make_fluid2d_simulation(project, explicit: bool = False):
    if project.mesh.mode == "cartesian":
        s = project.fluid2d
        hier, n_nodes = _fluid_mesh(project)
        gpu = not explicit and s is not None and s.linear_solver != "direct" and _gpu_ok(n_nodes)
        if hier is not None:
            from .amr import AmrFluid2dSimulation, GpuAmrFluid2dSimulation

            return GpuAmrFluid2dSimulation(project) if gpu else AmrFluid2dSimulation(project, explicit=explicit)
        if gpu:
            from .gpu import GpuCartesianFluid2dSimulation

            return GpuCartesianFluid2dSimulation(project)
        from .simulation import CartesianFluid2dSimulation

        return CartesianFluid2dSimulation(project, explicit=explicit)
    from ..fluid2d import Fluid2dSimulation

    return Fluid2dSimulation(project, explicit=explicit)


def __getattr__(name: str):
    if name == "CartesianFluid2dSimulation":
        from .simulation import CartesianFluid2dSimulation

        return CartesianFluid2dSimulation
    if name == "GpuCartesianFluid2dSimulation":
        from .gpu import GpuCartesianFluid2dSimulation

        return GpuCartesianFluid2dSimulation
    if name in ("AmrFluid2dSimulation", "GpuAmrFluid2dSimulation"):
        from . import amr

        return getattr(amr, name)
    raise AttributeError(name)
