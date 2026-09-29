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
- v1 と同じく periodic 境界は未対応。
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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
        b = self._q_static + q + (self.q_surf / (2.0 * np.pi) if self.rz else self.q_surf)
        if self._n_groups:
            b = b + self._coupling @ v
        op = self._op
        self._v_now = v
        if self._lu is not None:
            x = np.zeros(self.n_nodes)
            x[self._lu_idx] = self._lu.solve(b[self._lu_idx])
            return fill_fixed(op, x.reshape(op.grid.shape), v).ravel()
        dev = self._device
        x, info = self._gmg.solve(dev.asarray(b.reshape(op.grid.shape)), x0=self._x_prev, tol=POISSON_TOL)
        self._x_prev = x
        self.poisson_iters += info.iterations
        if not info.converged and not self._poisson_warned:
            self.warnings.append(
                f"Poisson (GMG-PCG) が収束しませんでした (相対残差 {info.relative_residual:.2e}、"
                "このメッセージは初回のみ)"
            )
            self._poisson_warned = True
        return fill_fixed(op, dev.to_host(x), v).ravel()

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
