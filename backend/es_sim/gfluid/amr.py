"""AMR 階層 (合成格子) 上の流体 2D (prompts/128)。

輸送グラフ (gfluid.amr_graph: ぶら下がり節点を含む Delaunay 適合三角形分割の Voronoi 有限体積、固体は
埋め込み境界) と、P4 の合成格子の Poisson (build_pic_layout の PTq / PC) で、直交格子版
(CartesianFluid2dSimulation) と同じ物理・時間積分を進める。

- AmrFluid2dSimulation: CPU。Poisson は合成格子の行列 A_c を、未知数 DIRECT_MAX 以下なら疎行列 LU (前分解)、
  それより大きいか特異なら pyamg の AMG-CG (前サブステップの解から) で解く。
- GpuAmrFluid2dSimulation: GPU (gfluid.gpu の輸送 + GPU の AMG-PCG)。
- 細分化は mesh.amr の静的な指定 (導体・誘電体の境界近傍・指定矩形)。表示用メッシュは適合三角形分割
  (ぶら下がり節点の T 字接続の無い形) で、/ws/fluid2d の started に載せる。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla

from ..amr.hierarchy import AmrHierarchy, AmrSpec
from ..field.electrostatic import group_voltages
from ..gpic.geometry import DisplayMesh
from ..particles import QE
from ..schema import Project
from .amr_graph import build_amr_fluid_graph
from .gpu import GpuCartesianFluid2dSimulation
from .simulation import DIRECT_MAX, POISSON_TOL, CartesianFluid2dSimulation


def amr_fluid_hierarchy(project: Project, model, grid) -> AmrHierarchy | None:
    """mesh.amr で実際に細分化が起きるなら階層、それ以外は None (流体は一様格子版で解く)。"""
    amr = project.mesh.amr
    if amr is None or (amr.max_level <= 0 and not amr.regions):
        return None
    hier = AmrHierarchy(model, grid, AmrSpec.from_settings(amr))
    return hier if hier.max_level > 0 else None


class AmrFluid2dSimulation(CartesianFluid2dSimulation):
    """合成格子 (AMR) 上の 2D/軸対称 ドリフト拡散流体 (CPU)。"""

    def _build_graph(self, project: Project, model, grid):
        hier = amr_fluid_hierarchy(project, model, grid)
        if hier is None:
            raise ValueError("mesh.amr で細分化が起きません (一様格子版 CartesianFluid2dSimulation を使ってください)")
        self.hier = hier
        g = build_amr_fluid_graph(model, hier)
        lay = g.lay
        self._lay = lay
        op = lay.op
        self.warnings.extend(op.warnings)
        if project.mesh.amr.adaptive:
            self.warnings.append("流体 2D は静電場の適応細分化 (adaptive) を使いません (静的な細分化の格子で解きます)")
        self._lu = None
        self._amg = None
        if self._allow_lu:
            n_c = op.A_c.shape[0]
            if n_c and n_c <= DIRECT_MAX and not op.singular:
                self._lu = spla.splu(op.A_c.tocsc(), permc_spec="MMD_AT_PLUS_A")
            elif n_c:
                import pyamg

                self._amg = pyamg.smoothed_aggregation_solver(op.A_c, symmetry="symmetric", max_coarse=500)
        dm = DisplayMesh(nodes=g.nodes, triangles=g.tris, tri_region=g.tri_region,
                         tri_cell=np.zeros(len(g.tris), dtype=np.int64))
        return g, dm

    # ---- Poisson (合成格子) --------------------------------------------------------------

    def _init_poisson(self, project: Project) -> None:
        self._n_groups = len(self.model.groups)
        self._v_now = group_voltages(self.model, 0.0)
        self._x_prev = None
        self._poisson_warned = False
        self.poisson_iters = 0

    def _solve_phi(self, t: float) -> np.ndarray:
        """b_c = PTq [q_all; V] を解き、全節点の電位 φ = PC [x_c; V] を返す。"""
        lay = self._lay
        op = lay.op
        v = group_voltages(self.model, t)
        self._v_now = v
        vK = np.zeros(lay.n_groups)
        vK[: v.size] = v
        q_all = lay.q_static + QE * np.asarray(self.graph.charge_map @ (self.n_i - self.n_e)).ravel()
        b = lay.PTq @ np.concatenate([q_all, vK])
        if op.singular and b.size:
            b = b - b.mean()
        if b.size == 0:
            x = np.zeros(0)
        elif self._lu is not None:
            x = self._lu.solve(b)
        else:
            res: list[float] = []
            x = self._amg.solve(b, x0=self._x_prev, tol=POISSON_TOL, accel="cg", maxiter=200, residuals=res)
            if op.singular:
                x = x - x.mean()
            self.poisson_iters += max(len(res) - 1, 0)
            bn = float(np.linalg.norm(b))
            rel = float(np.linalg.norm(b - op.A_c @ x)) / bn if bn > 0.0 else 0.0
            if rel > 10.0 * POISSON_TOL and not self._poisson_warned:
                self.warnings.append(
                    f"Poisson (AMG-CG) が収束しませんでした (相対残差 {rel:.2e}、このメッセージは初回のみ)"
                )
                self._poisson_warned = True
        self._x_prev = x
        return np.asarray(lay.PC @ np.concatenate([x, vK])).ravel()


class GpuAmrFluid2dSimulation(GpuCartesianFluid2dSimulation, AmrFluid2dSimulation):
    """合成格子 (AMR) 上の流体 2D を GPU で進める (輸送は gfluid.gpu、Poisson は GPU の AMG-PCG)。"""
