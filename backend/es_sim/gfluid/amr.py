"""AMR 階層 (合成格子) 上の流体 2D (prompts/128)。

輸送グラフ (gfluid.amr_graph: ぶら下がり節点を含む Delaunay 適合三角形分割の Voronoi 有限体積、固体は
埋め込み境界) と、P4 の合成格子の Poisson (build_pic_layout の PTq / PC) で、直交格子版
(CartesianFluid2dSimulation) と同じ物理・時間積分を進める。

- AmrFluid2dSimulation: CPU。Poisson は合成格子の行列 A_c を、未知数 DIRECT_MAX 以下なら疎行列 LU (前分解)、
  それより大きいか特異なら pyamg の AMG-CG (前サブステップの解から) で解く。
- GpuAmrFluid2dSimulation: GPU (gfluid.gpu の輸送 + GPU の AMG-PCG)。
- 細分化は mesh.amr の静的な指定 (導体・誘電体の境界近傍・指定矩形)。表示用メッシュは適合三角形分割
  (ぶら下がり節点の T 字接続の無い形) で、/ws/fluid2d の started に載せる。
- 阻止コンデンサ (自己バイアス、prompts/134): 電極の電荷は静電場の AMR (amr.composite.energy_and_charges) と
  同じ「Dirichlet 結合の電束 + ぶら下がり節点の拘束反力 Cᵀr」から電極の固定節点の電荷を引いたもの。全て
  φ・グループの電位・節点の電荷について線形なので、電極ごとの重み (a・b・c) を最初に作り、Q = a·φ + b·V + c·q
  で求める。ψ は合成格子で 1 回解き (x_ψ、φ_ψ = PC [x_ψ; e])、解は x += ΣΔV x_ψ で直す。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla

from ..amr.hierarchy import AmrHierarchy, AmrSpec
from ..circuit import BlockingCircuit
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

    def _poisson_unknown(self) -> np.ndarray:
        """合成格子の節点ごとに Poisson の未知数か (固定 = Dirichlet でない。ぶら下がり節点も含む:
        その電荷は PTq が親の未知数へ配る)。"""
        return self._lay.op.fixed_group < 0

    def _node_fixed_group(self) -> np.ndarray:
        return self._lay.op.fixed_group

    # ---- Poisson (合成格子) --------------------------------------------------------------

    def _init_poisson(self, project: Project) -> None:
        self._n_groups = len(self.model.groups)
        self._v_now = group_voltages(self.model, 0.0)
        self._x_prev = None
        self._poisson_warned = False
        self.poisson_iters = 0

    def _solve_phi(self, t: float) -> np.ndarray:
        """b_c = PTq [q_all; V] を解き、全節点の電位 φ = PC [x_c; V] を返す。

        q_all は一様格子版の右辺 b と同じ 2π を落とした単位: 体積電荷 (charge_map) と誘電体の表面電荷
        (q_surf は 2π 込みで持っているので軸対称では 2π で割る、prompts/129)。
        """
        lay = self._lay
        v = group_voltages(self.model, t)
        q_all = (lay.q_static + QE * np.asarray(self.graph.charge_map @ (self.n_i - self.n_e)).ravel()
                 + (self.q_surf / (2.0 * np.pi) if self.rz else self.q_surf))
        circuit = self.circuit
        if circuit is not None:
            # 阻止コンデンサの電極は前のサブステップの電位で φ* を解く (モジュール docstring)
            v_src = np.array([v[groups[0]] for groups in self._cap_groups])
            v_old = circuit.potentials(v_src)
            v = self._cap_set(v, v_old)
        vK = np.zeros(lay.n_groups)
        vK[: v.size] = v
        x = self._solve_composite(lay.PTq @ np.concatenate([q_all, vK]), self._x_prev)
        phi = np.asarray(lay.PC @ np.concatenate([x, vK])).ravel()
        if circuit is not None:
            q_star = self._cap_factor * (self._cap_a @ phi + self._cap_b @ vK + self._cap_c @ q_all)
            v_new = circuit.solve(q_star, v_old, v_src)
            x = x + (v_new - v_old) @ self._cap_x
            v = self._cap_set(v, v_new)
            vK[: v.size] = v
            phi = np.asarray(lay.PC @ np.concatenate([x, vK])).ravel()
        self._v_now = v
        self._x_prev = x
        return phi

    def _solve_composite(self, b: np.ndarray, x0) -> np.ndarray:
        """合成格子の A_c x = b (LU か AMG-CG。特異なら右辺と解の平均を除く)。"""
        op = self._lay.op
        if op.singular and b.size:
            b = b - b.mean()
        if b.size == 0:
            return np.zeros(0)
        if self._lu is not None:
            return self._lu.solve(b)
        res: list[float] = []
        x = self._amg.solve(b, x0=x0, tol=POISSON_TOL, accel="cg", maxiter=200, residuals=res)
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
        return x

    # ---- 阻止コンデンサ (自己バイアス、prompts/134・モジュール docstring) ----------------------

    def _init_circuit(self) -> None:
        """阻止コンデンサの回路: 電極の電荷の重み (a・b・c)、合成格子の ψ と容量行列、壁の小片の行き先。"""
        elecs = self._capacitor_electrodes()
        if not elecs:
            return
        lay = self._lay
        op = lay.op
        N = lay.n_nodes
        K = lay.n_groups
        m = len(elecs)
        self._cap_groups = [groups for _, groups in elecs]
        group_elec = np.full(K, -1, dtype=np.int64)
        for j, groups in enumerate(self._cap_groups):
            group_elec[groups] = j
        self._cap_factor = 2.0 * np.pi if self.rz else 1.0
        # 未知番号 → 節点番号
        unk = np.nonzero(op.fixed_group < 0)[0]
        node_of_u = np.empty(unk.size, dtype=np.int64)
        node_of_u[op.u_of_node[unk]] = unk
        a = np.zeros((m, N))
        b = np.zeros((m, K))
        c = np.zeros((m, N))
        ej = group_elec[op.coup_grp] if op.coup_grp.size else np.zeros(0, dtype=np.int64)
        for j in range(m):
            # Dirichlet 結合の電束 Σ g (V − φ_u)
            s = ej == j
            np.add.at(a[j], node_of_u[op.coup_u[s]], -op.coup_g[s])
            np.add.at(b[j], op.coup_grp[s], op.coup_g[s])
            # ぶら下がり節点の拘束反力 (Cᵀ r)_j、r = A_full φ_U − coup_full V − q_U
            if op.C.nnz:
                cols = np.zeros(op.C.shape[1])
                cols[self._cap_groups[j]] = 1.0
                w = np.asarray(op.C @ cols).ravel()
                a[j][node_of_u] += op.A_full.T @ w
                b[j] -= op.coup_full.T @ w
                c[j][node_of_u] -= w
            # 電極の固定節点の電荷 (Poisson は見ない) を引く
            fixed = np.nonzero((op.fixed_group >= 0) & (group_elec[np.maximum(op.fixed_group, 0)] == j))[0]
            c[j][fixed] -= 1.0
        self._cap_a, self._cap_b, self._cap_c = a, b, c
        # ψ_j: 合成格子の未知 x_ψ と全節点 φ_ψ (空間電荷 0)。GPU 版は CPU の解法を持たないので、ここだけ
        # 疎行列の直接法 (大きければ AMG-CG) で解く
        if self._lu is None and self._amg is None and op.A_c.shape[0]:
            if op.A_c.shape[0] <= 4 * DIRECT_MAX:
                lu = spla.splu(op.A_c.tocsc(), permc_spec="MMD_AT_PLUS_A")
                solve_psi = lambda b: lu.solve(b)  # noqa: E731
            else:
                import pyamg

                amg = pyamg.smoothed_aggregation_solver(op.A_c, symmetry="symmetric", max_coarse=500)
                solve_psi = lambda b: amg.solve(b, tol=1e-12, accel="cg", maxiter=500)  # noqa: E731
        else:
            solve_psi = lambda b: self._solve_composite(b, None)  # noqa: E731
        xs, phis, units = [], [], []
        for j in range(m):
            vK = np.zeros(K)
            vK[self._cap_groups[j]] = 1.0
            x = solve_psi(lay.PTq @ np.concatenate([np.zeros(N), vK]))
            xs.append(x)
            phis.append(np.asarray(lay.PC @ np.concatenate([x, vK])).ravel())
            units.append(vK)
        self._cap_x = np.array(xs)
        self._cap_psi = np.array(phis)
        c_matrix = np.array([[self._cap_factor * (a[i] @ phis[k] + b[i] @ units[k]) for k in range(m)]
                             for i in range(m)])
        period = 1.0 / self._cycle_freq if self._cycle_freq is not None else None
        self.circuit = BlockingCircuit([spec for spec, _ in elecs], c_matrix, period)
        self._init_cap_ends(group_elec)



class GpuAmrFluid2dSimulation(GpuCartesianFluid2dSimulation, AmrFluid2dSimulation):
    """合成格子 (AMR) 上の流体 2D を GPU で進める (輸送は gfluid.gpu、Poisson は GPU の AMG-PCG)。"""
