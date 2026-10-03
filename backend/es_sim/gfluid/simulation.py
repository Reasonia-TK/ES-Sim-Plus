"""v2 流体 2D — 直交格子 + 埋め込み境界 (EB) 上のドリフト拡散流体 (prompts/125)。

v1 ``fluid2d.Fluid2dSimulation`` と**同じ物理・同じ時間積分・同じ外部インターフェース**
(run_batch / prepare_continue / step / fields / cycle / history / build_fluid2d_result) で、
輸送グラフと Poisson だけを v2 のものに差し替える (mesh.mode="cartesian" のとき)。

- 輸送グラフ: 直交格子の双対セル (気体部分) を制御体積とし、固体を埋め込み境界として扱う
  (gfluid.geometry)。種ごとの SG フラックス・壁の流束条件・SEE・誘電体表面の帯電・Joule 加熱・
  反応・半陰的なサブステップ分割・反復ソルバーは v1 のコードをそのまま使う (節点番号の付け方が
  違うだけ)。
- Poisson: v2 の EB 静電場の離散化 (eb.build) をそのまま使う。格子の節点数が DIRECT_MAX 以下なら
  疎行列の LU を初期化時に 1 回だけ分解して毎ステップ前進後退代入 (v1 と同じ流儀。小さな格子
  では GPU の反復法よりずっと速い)、それより大きい・または特異 (Dirichlet 無し) なら GMG-PCG
  (GPU があれば GPU) を前ステップの解から解き直す。電荷は双対セルの気体体積 × e(n_i − n_e)
  (v1 と同じ集中質量近似) と誘電体の表面電荷 (小片を受け持つ輸送節点に置く、prompts/129)。
- 結果・フレームは表示用メッシュ (セルの 2 三角形分割、導体内は穴) の節点値・要素値で返す
  (/mesh の直交格子の表示用メッシュと同じ節点番号。/ws/fluid2d の started にも載せる)。
  mesh.amr の局所細分化は AMR 版 (gfluid.amr.AmrFluid2dSimulation、prompts/128) が
  _build_graph / Poisson を差し替えて解く (make_fluid2d_simulation が振り分ける)。
- 阻止コンデンサ (自己バイアス、prompts/134): 電極 (導体・Dirichlet の辺。1 つの境界条件の辺はまとめて 1 つ)
  ごとに ψ (その電極 1 V・ほか 0 V・空間電荷 0) を最初に 1 回解き、容量行列もそこから求める。毎サブステップ、
  前の電極の電位で φ* を解き、電極の電荷 (結合の電束 ΣG(V − φ) − 電極の固定節点の電荷) から回路
  (circuit.BlockingCircuit) が新しい電位を決め、φ = φ* + ΣΔV ψ に直す。伝導電流は壁の小片ごとに行き先の
  電極へ足す (導体の表面・外周の Dirichlet の辺・電極に接した誘電体の縁)。AMR 版は合成格子の電荷の式で
  _init_circuit・_solve_phi を差し替える (gfluid.amr)。
- v1 と同じく periodic 境界は未対応。
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..circuit import BlockingCircuit, CapacitorSpec, model_capacitor_electrodes
from ..device import Device, get_device
from ..eb.build import MASK_FIXED, MASK_UNKNOWN, LevelOperator, build_level
from ..eb.grid import make_grid
from ..field.electrostatic import fill_fixed, group_voltages
from ..field.gmg import GMGSolver
from ..fluid2d import Fluid2dSimulation
from ..geom.model import GeometryModel
from ..gpic.geometry import DisplayMesh, ParticleGeometry
from ..particles import QE, _barycentric_coeffs
from ..schema import Project
from .geometry import FluidGraph, build_fluid_graph

#: Poisson (GMG-PCG) の収束判定 (右辺ノルム比)
POISSON_TOL = 1e-10
#: Poisson を疎行列 LU (前分解) で解く格子の節点数の上限。これを超えると GMG-PCG
DIRECT_MAX = 60_000


def _sparse_operator(op: LevelOperator) -> tuple[np.ndarray, sp.csc_matrix]:
    """EB Poisson の演算子 (周期なし) を未知節点だけの疎行列にする。戻り値: (未知節点の平坦番号, A)。"""
    nx1 = op.grid.nx + 1
    mask = op.mask.ravel()
    idx = np.nonzero(mask == MASK_UNKNOWN)[0]
    pos = np.full(mask.size, -1, dtype=np.int64)
    pos[idx] = np.arange(idx.size)
    rows, cols, vals = [pos[idx]], [pos[idx]], [op.diag.ravel()[idx]]
    for c_arr, step in ((op.cx, 1), (op.cy, nx1)):
        jj, ii = np.nonzero(c_arr)
        p = jj * nx1 + ii
        q = p + step
        c = c_arr[jj, ii]
        ok = (pos[p] >= 0) & (pos[q] >= 0)
        rows += [pos[p[ok]], pos[q[ok]]]
        cols += [pos[q[ok]], pos[p[ok]]]
        vals += [-c[ok], -c[ok]]
    a = sp.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(idx.size, idx.size)
    )
    return idx, a


class CartesianFluid2dSimulation(Fluid2dSimulation):
    """直交格子 EB 版の 2D/軸対称 ドリフト拡散流体 (v1 Fluid2dSimulation の派生)。"""

    #: Poisson を CPU の疎行列 LU で解いてよいか (GPU 版は常に GPU の GMG を使う)
    _allow_lu = True

    def __init__(self, project: Project, explicit: bool = False, device: Device | str | None = None):
        self._device = device if isinstance(device, Device) else get_device(device)
        super().__init__(project, explicit=explicit)

    # ---- 幾何 (v1 の _init_geometry を差し替え) -------------------------------------------

    def _init_geometry(self, project: Project) -> None:
        t0 = time.perf_counter()
        model = GeometryModel(project)
        self.model = model
        self.ridx = model.radial_axis()
        self.rz = self.ridx is not None
        grid = make_grid(model.domain, float(project.mesh.size))
        self.grid = grid
        g, dm = self._build_graph(project, model, grid)
        self.mesh = dm
        self.n_nodes = len(dm.nodes)
        self.tris = dm.triangles
        _, self._bc_b, self._bc_c, self._bc_det = _barycentric_coeffs(dm.nodes, dm.triangles)
        self.graph = g
        self.warnings.extend(g.warnings)
        self._adopt_graph(g)
        self.h_min = float(getattr(g, "h_min", 0.0) or min(grid.dx, grid.dy))
        self.setup_s = time.perf_counter() - t0

    def _build_graph(self, project: Project, model: GeometryModel, grid) -> tuple[FluidGraph, DisplayMesh]:
        """輸送グラフと表示用メッシュを組み、Poisson の前処理を用意する (AMR 版が差し替える)。"""
        # Poisson: 中小規模は疎行列 LU (前分解)、大規模・特異 (Dirichlet 無し) は GMG-PCG
        self._lu = None
        self._gmg = None
        op = None
        if self._allow_lu and grid.n_nodes <= DIRECT_MAX and model.groups:
            op = build_level(model, grid, full=True)
            if np.any(op.mask == MASK_FIXED):
                self._lu_idx, a = _sparse_operator(op)
                self._lu = spla.splu(a, permc_spec="MMD_AT_PLUS_A")
            else:
                op = None
        if op is None:
            self._gmg = GMGSolver(model, grid, self._device)
            op = self._gmg.finest
        self._op = op
        self.warnings.extend(op.warnings)
        pg = ParticleGeometry(model, grid)
        return build_fluid_graph(model, grid, pg.cell_state, op.vol_gas), pg.display_mesh()

    def _adopt_graph(self, g: FluidGraph) -> None:
        """輸送グラフを v1 Fluid2dSimulation の属性 (アクティブ節点・辺・壁) に写す。"""
        self.node_vol_full = g.node_vol
        self.active_idx = np.nonzero(g.active)[0]
        self.n_active = len(self.active_idx)
        glob_to_loc = np.full(self.n_nodes, -1, dtype=np.int64)
        glob_to_loc[self.active_idx] = np.arange(self.n_active)
        self._glob_to_loc = glob_to_loc
        self.node_vol = g.node_vol[self.active_idx]

        self.i_idx = glob_to_loc[g.edge_p]
        self.j_idx = glob_to_loc[g.edge_q]
        self.w_ij = g.edge_w
        self._edge_len = g.edge_len
        self._build_sparse_pattern()

        # 壁: 1 つの壁小片は 1 節点に属する (v1 の壁エッジの 2 端点の片側だけを使う形)
        loc = glob_to_loc[g.wall_node]
        self.wall_n1 = loc
        self.wall_n2 = loc
        self.wall_w1 = g.wall_area
        self.wall_w2 = np.zeros_like(g.wall_area)
        self.wall_nout = g.wall_normal
        self.wall_tri = np.zeros(len(loc), dtype=np.int64)  # v1 の要素 E は使わない (_wall_en)
        gamma_node = np.zeros(self.n_nodes)
        np.maximum.at(gamma_node, g.wall_node, g.wall_gamma)
        self.gamma_see = gamma_node[self.active_idx]
        self._wall_group = g.wall_group
        self._wall_is_cond = g.wall_group >= 0

        # 帯電する壁端 (v1 の _set_surface_charge_ends と同じ表): 誘電体表面の小片のうち、受け持つ
        # 輸送節点が Poisson の未知節点のもの (固定節点 = 電極に接した誘電体の縁に落ちる分は電極へ
        # 流れる扱い、v1 と同じ)
        sel = np.nonzero(g.wall_dielectric & (g.wall_area > 0.0) & self._poisson_unknown()[g.wall_node])[0]
        self._set_surface_charge_ends(sel, loc[sel], g.wall_area[sel])

    def _poisson_unknown(self) -> np.ndarray:
        """節点ごとに Poisson の未知数か (表面電荷を置くと電位に効く節点。AMR 版は差し替える)。"""
        return self._op.mask.ravel() == MASK_UNKNOWN

    # ---- Poisson (v1 の splu を v2 の GMG-PCG に差し替え) ----------------------------------

    def _init_poisson(self, project: Project) -> None:
        op = self._op
        self._coupling = op.coupling
        self._q_static = op.q_static.ravel()
        self._n_groups = len(self.model.groups)
        self._v_now = group_voltages(self.model, 0.0)
        self._x_prev = None
        self._poisson_warned = False
        self.poisson_iters = 0

    def _solve_phi(self, t: float) -> np.ndarray:
        """現在の n_e・n_i・q_surf と時刻 t の電極電位で φ を解く (LU なら前進後退代入、GMG なら warm start)。

        右辺は 2π を落とした単位 (eb.build): 体積電荷は charge_map (2π なし)、表面電荷 q_surf は
        2π 込みの物理単位で持っているので軸対称では 2π で割る (v1 と同じ)。
        """
        v = group_voltages(self.model, t)
        q = QE * np.asarray(self.graph.charge_map @ (self.n_i - self.n_e)).ravel()
        b_q = self._q_static + q + (self.q_surf / (2.0 * np.pi) if self.rz else self.q_surf)
        circuit = self.circuit
        if circuit is not None:
            # 阻止コンデンサの電極は前のサブステップの電位で φ* を解く (モジュール docstring)
            v_src = np.array([v[groups[0]] for groups in self._cap_groups])
            v_old = circuit.potentials(v_src)
            v = self._cap_set(v, v_old)
        b = b_q + self._coupling @ v if self._n_groups else b_q
        op = self._op
        dev = self._device
        if self._lu is not None:
            x = np.zeros(self.n_nodes)
            x[self._lu_idx] = self._lu.solve(b[self._lu_idx])
            phi = fill_fixed(op, x.reshape(op.grid.shape), v).ravel()
        else:
            x, info = self._gmg.solve(dev.asarray(b.reshape(op.grid.shape)), x0=self._x_prev, tol=POISSON_TOL)
            self._x_prev = x
            self.poisson_iters += info.iterations
            if not info.converged and not self._poisson_warned:
                self.warnings.append(
                    f"Poisson (GMG-PCG) が収束しませんでした (相対残差 {info.relative_residual:.2e}、"
                    "このメッセージは初回のみ)"
                )
                self._poisson_warned = True
            phi = fill_fixed(op, dev.to_host(x), v).ravel()
        if circuit is not None:
            v_new = circuit.solve(self._cap_charges(phi, v, b_q), v_old, v_src)
            shift = (v_new - v_old) @ self._cap_psi
            v = self._cap_set(v, v_new)
            phi = fill_fixed(op, (phi + shift).reshape(op.grid.shape), v).ravel()
            if self._lu is None:
                self._x_prev = self._x_prev + dev.asarray(shift.reshape(op.grid.shape))
        self._v_now = v
        return phi

    # ---- 阻止コンデンサ (自己バイアス、prompts/134・モジュール docstring) ----------------------

    def _capacitor_electrodes(self) -> list[tuple[CapacitorSpec, list[int]]]:
        """阻止コンデンサを付けた電極と、その Dirichlet グループの番号 (circuit.model_capacitor_electrodes)。"""
        return model_capacitor_electrodes(self.project, self.model)

    def _init_circuit(self) -> None:
        """阻止コンデンサの回路を組む: 電荷の集計の表、ψ と容量行列、壁の小片の行き先。"""
        elecs = self._capacitor_electrodes()
        if not elecs:
            return
        op = self._op
        n_groups = len(self.model.groups)
        m = len(elecs)
        self._cap_groups = [groups for _, groups in elecs]
        group_elec = np.full(n_groups, -1, dtype=np.int64)
        for j, groups in enumerate(self._cap_groups):
            group_elec[groups] = j
        # 電極の電荷 = 結合 (未知節点 P → グループ g) の電束 − 電極の固定節点に置かれた電荷
        coo = op.coupling.tocoo()
        ej = group_elec[coo.col]
        keep = ej >= 0
        self._cap_coo = (coo.row[keep].astype(np.int64), coo.col[keep].astype(np.int64), coo.data[keep], ej[keep])
        fixed = op.fixed_group.ravel()
        fixed_elec = np.where(fixed >= 0, group_elec[np.maximum(fixed, 0)], -1)
        self._cap_fixed_nodes = np.nonzero((op.mask.ravel() == MASK_FIXED) & (fixed_elec >= 0))[0]
        self._cap_fixed_elec = fixed_elec[self._cap_fixed_nodes]
        self._cap_factor = 2.0 * np.pi if self.rz else 1.0
        self._cap_m = m
        # ψ_j (その電極 1 V・ほかの電極 0 V・空間電荷 0) と容量行列 C_jk = ψ_k の解での電極 j の電荷
        units = [self._cap_set(np.zeros(n_groups), np.eye(m)[k]) for k in range(m)]
        self._cap_psi = np.array([self._solve_unit(v) for v in units])
        zero = np.zeros(self.n_nodes)
        c_matrix = np.array([self._cap_charges(self._cap_psi[k], units[k], zero) for k in range(m)]).T
        period = 1.0 / self._cycle_freq if self._cycle_freq is not None else None
        self.circuit = BlockingCircuit([spec for spec, _ in elecs], c_matrix, period)
        self._init_cap_ends(group_elec)

    def _init_cap_ends(self, group_elec: np.ndarray) -> None:
        """伝導電流の行き先: 壁の小片 (1 つの端) ごとの電極 (_wall_sink_groups のグループの電極)。"""
        sink = self._wall_sink_groups()
        elec = np.where(sink >= 0, group_elec[np.maximum(sink, 0)], -1)
        pieces = np.nonzero(elec >= 0)[0]
        self._cap_ends = (pieces, self.wall_n1[pieces], self.wall_w1[pieces], elec[pieces])

    def _cap_set(self, v: np.ndarray, values) -> np.ndarray:
        """グループの電位 v のうち、阻止コンデンサの電極のグループを values (電極ごと) にしたもの。"""
        v = np.array(v, dtype=np.float64)
        for groups, value in zip(self._cap_groups, values):
            v[groups] = value
        return v

    def _solve_unit(self, v: np.ndarray) -> np.ndarray:
        """空間電荷 0・グループの電位 v の解 (全節点。ψ 用)。"""
        op = self._op
        b = self._coupling @ v
        if self._lu is not None:
            x = np.zeros(self.n_nodes)
            x[self._lu_idx] = self._lu.solve(b[self._lu_idx])
            return fill_fixed(op, x.reshape(op.grid.shape), v).ravel()
        dev = self._device
        x, info = self._gmg.solve(dev.asarray(b.reshape(op.grid.shape)), tol=POISSON_TOL)
        if not info.converged:
            self.warnings.append(
                f"阻止コンデンサの ψ の Poisson (GMG-PCG) が収束しませんでした (相対残差 {info.relative_residual:.2e})"
            )
        return fill_fixed(op, dev.to_host(x), v).ravel()

    def _cap_charges(self, phi: np.ndarray, v: np.ndarray, b_q: np.ndarray) -> np.ndarray:
        """阻止コンデンサの電極の表面の電荷 (軸対称は 2π 込み、平面は奥行き 1 m あたり)。

        電極から領域への電束 Σ G_Pk (V_k − φ_P) (静電場の電極の電荷と同じ) から、電極の固定節点に置かれた
        電荷 (b_q: 体積電荷・誘電体の表面電荷・固定の電荷) を引く。Poisson はその電荷を見ない (固定節点) ので、
        電極の表面のすぐ外の層とみなし、残りを導体の電荷とする (circuit.py の docstring)。
        """
        rows, cols, data, ej = self._cap_coo
        flux = np.bincount(ej, weights=data * (v[cols] - phi[rows]), minlength=self._cap_m)
        own = np.bincount(self._cap_fixed_elec, weights=b_q[self._cap_fixed_nodes], minlength=self._cap_m)
        return self._cap_factor * (flux - own)

    def _wall_sink_groups(self) -> np.ndarray:
        """壁の小片ごとに、流れ込む電荷を受け取る Dirichlet グループ (−1 は電極でない: 帯電する誘電体の面)。

        導体の表面は wall_group。外周の Dirichlet の辺の小片 (wall_group = −1 で誘電体でない) は法線 (気体 → 壁)
        の向きから辺を決める。電極に接した誘電体の縁 (受け持つ節点が Poisson の固定節点) は表面電荷にならず
        電極へ流れる扱い (_adopt_graph) なので、その節点のグループ。
        """
        g = self.graph
        sink = np.array(g.wall_group, dtype=np.int64)
        side = (sink < 0) & ~g.wall_dielectric
        if np.any(side):
            nx_, ny_ = g.wall_normal[side, 0], g.wall_normal[side, 1]
            names = np.where(ny_ < -0.5, "bottom", np.where(ny_ > 0.5, "top", np.where(nx_ < -0.5, "left", "right")))
            sink[side] = [self.model.side_group.get(str(s), -1) for s in names]
        fixed = self._node_fixed_group()
        diel = g.wall_dielectric & (fixed[g.wall_node] >= 0)
        sink[diel] = fixed[g.wall_node[diel]]
        return sink

    def _node_fixed_group(self) -> np.ndarray:
        """節点ごとの Poisson の固定のグループ (−1 は未知。AMR 版は合成格子の節点で差し替える)。"""
        return self._op.fixed_group.ravel()

    def _wall_en(self, phi: np.ndarray) -> np.ndarray:
        """壁小片ごとの E·n = (φ(A) − φ(W)) / L (gfluid.geometry のモジュール docstring)。"""
        g = self.graph
        phi_a = np.sum(phi[g.wall_a_idx] * g.wall_a_wt, axis=1)
        phi_w = np.sum(phi[g.wall_w_idx] * g.wall_w_wt, axis=1)
        if self._n_groups and np.any(self._wall_is_cond):
            phi_w = np.where(self._wall_is_cond, self._v_now[np.maximum(self._wall_group, 0)], phi_w)
        return (phi_a - phi_w) / g.wall_len

    # ---- 表示 -------------------------------------------------------------------------

    def mesh_payload(self) -> dict:
        """表示用メッシュ (/ws/fluid2d の started に載せる。/mesh の MeshResult と同じ形)。"""
        return {
            "nodes": self.mesh.nodes.tolist(),
            "triangles": self.mesh.triangles.tolist(),
            "region_of_triangle": self.mesh.tri_region.tolist(),
        }
