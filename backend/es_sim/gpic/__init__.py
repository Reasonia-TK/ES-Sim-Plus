"""v2 GPU PIC-MCC — 直交格子 + 埋め込み境界 (EB) 上の粒子シミュレーション (prompts/119 P3)。"""

from __future__ import annotations


def make_pic_simulation(project, gas_field=None):
    """mesh.mode に応じて PIC シミュレーションを作る (server / batch / sweep の共通入口)。

    - "cartesian": v2 GPU PIC (GpuPicSimulation、CUDA 必須)
    - それ以外: v1 FEM-PIC (PicSimulation)
    どちらも run_batch / fields / cycle / collector_results / eedf_results / prepare_continue 等の
    同じ外部インターフェースを持つ。
    """
    if project.mesh.mode == "cartesian":
        from .simulation import GpuPicSimulation

        return GpuPicSimulation(project, gas_field)
    from ..pic import PicSimulation

    return PicSimulation(project, gas_field)


def __getattr__(name):
    # GpuPicSimulation は CuPy を import するので、使うときだけ読み込む
    if name == "GpuPicSimulation":
        from .simulation import GpuPicSimulation

        return GpuPicSimulation
    raise AttributeError(name)


__all__ = ["GpuPicSimulation", "make_pic_simulation"]
