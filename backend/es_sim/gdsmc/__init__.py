"""v2 GPU DSMC — 直交格子 + 埋め込み固体の上の定常ガス流れ (prompts/124)。"""

from __future__ import annotations


def make_dsmc_simulation(project):
    """mesh.mode に応じて DSMC を作る (server の共通入口)。

    - "cartesian": v2 GPU DSMC (GpuDsmcSimulation、CUDA 必須)
    - それ以外: v1 DSMC (DsmcSimulation、三角形メッシュ)
    どちらも run / prepare_continue / mesh / dt / x / step_count / timing の同じ外部インターフェースを持つ。
    """
    if project.mesh.mode == "cartesian":
        from .simulation import GpuDsmcSimulation

        return GpuDsmcSimulation(project)
    from ..dsmc import DsmcSimulation

    return DsmcSimulation(project)


def __getattr__(name):
    # GpuDsmcSimulation は CuPy を import するので、使うときだけ読み込む
    if name == "GpuDsmcSimulation":
        from .simulation import GpuDsmcSimulation

        return GpuDsmcSimulation
    raise AttributeError(name)


__all__ = ["GpuDsmcSimulation", "make_dsmc_simulation"]
