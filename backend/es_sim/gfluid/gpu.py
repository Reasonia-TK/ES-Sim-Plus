"""v2 流体 2D の GPU 版 — 直交格子 EB の輸送を CUDA で解く (prompts/126)。

CartesianFluid2dSimulation (gfluid.simulation) と同じ輸送グラフ・物理・時間積分 (v1 fluid2d の
半陰的なサブステップ分割) を、状態をデバイスに置いたまま CUDA で進める。

- 係数表 (log-log 補間)・SG フラックス・壁条件・反応・Joule 加熱は kernels/fluid.cu のカーネル。
  陰的行列は辺の接続リストによる行列フリー形式 (対角 + 各辺の a/b) で、行列を組み立てない。
- 種ごとの線形方程式は Jacobi 前処理 BiCGSTAB (v1 _numba_kernels.bicgstab と同じ漸化式)。
  内積は固定ブロック数の 2 段集約で決定的 (実行ごとにビット一致)。1 反復あたり 5 カーネルと
  残差の読み出し 1 回。収束しなければ v1 と同じく spsolve へフォールバックする。
- Poisson は GPU の GMG-PCG (未知数 4096 以下は密な逆行列で厳密に解く、_EbPoisson)。AMR 版
  (gfluid.amr.GpuAmrFluid2dSimulation) は合成格子の AMG-PCG (_AmrPoisson、prompts/128)。
- ホストとの同期は 1 ステップに 1 回の統計 (全量・壁損失・発散検出・次のステップのサブステップ
  制御量) と、BiCGSTAB・GMG の収束判定だけ。サブステップ数の上限 (超えたら ValueError)・サブステップの
  合間の停止要求 (デバイスの状態をその場でステップ開始時へ戻す)・Joule 緩和時間 (体積あたりの加熱率、
  電子がほぼ空の節点を除く) は v1 の step と同じ判定。
- 阻止コンデンサ (自己バイアス、prompts/134): 回路 (circuit.BlockingCircuit) はホストに置き、CPU 版と同じ量を
  デバイスで集約して読む (Poisson のあとに電極の電荷、輸送のあとに伝導電流。1 サブステップに 2 回の小さな読み出し)。
  新しい電極の電位はグループの電位に入れ直し、ψ の重ね合わせ (x += ΣΔV ψ) で解を直す。
- n_e / n_i / w / phi / q_surf (誘電体の表面電荷、prompts/129) はデバイスが正。属性として読むとホストへ
  写し、代入 (または読んだ配列の書き換え) は次のステップの前にデバイスへ戻す。フレーム・時間平均・
  位相分解は v1 と同じ形。

離散化は CPU 版と同じなので結果は丸め誤差の範囲で一致する (総和の順序が違うのでビット一致は
しない)。陽的検証経路 (explicit) と linear_solver="direct" は CPU 版を使う
(make_fluid2d_simulation が振り分ける)。
"""

from __future__ import annotations

import math
import time
import warnings

import numpy as np
import scipy.sparse.linalg as spla

from ..device import Device, get_device
from ..eb.build import MASK_FIXED
from ..fem import EPS0
from ..field.electrostatic import group_voltages
from ..fluid1d import FLOOR_N, FLOOR_W
from ..mcc import KB
from ..particles import QE
from ..schema import Project
from .simulation import POISSON_TOL, CartesianFluid2dSimulation

#: kernels/fluid.cu の FL_NB / FL_NT (決定的な集約のブロック数・スレッド数)
NB = 256
NT = 256
#: BiCGSTAB の収束判定・反復上限 (v1 Fluid2dSimulation と同じ値)
BICG_RTOL = 1.0e-10
BICG_ATOL = 1.0e-300
BICG_MAX_ITER = 200
#: これ以上の格子節点数なら make_fluid2d_simulation が GPU 版を選ぶ (小さい格子は CPU 版が速い)
GPU_MIN_NODES = 1_000
#: Poisson (GMG-PCG) を同期なしで積む反復数の初期値と上限 (前サブステップの解から warm start)
PCG_ITERS_START = 4
PCG_ITERS_MAX = 200

# BiCGSTAB の部分和スロット (fluid.cu の P_*)
_P_RR, _P_BB = 5, 6
_KERNELS = (
    "fl_coeffs", "fl_edges", "fl_nodes", "fl_wall_ci", "fl_wall_diag", "fl_diag", "fl_rhs", "fl_floor",
    "fl_see", "fl_gwe", "fl_joule", "fl_scale2", "fl_energy_off", "fl_wallgen", "fl_poisson_rhs",
    "fl_fixed", "bcg_init", "bcg_p", "bcg_v", "bcg_s", "bcg_t", "bcg_x", "fl_nemax", "fl_stats", "fl_accum",
    "fl_tri_e", "fl_surf", "fl_sum", "fl_cap_charge", "fl_cap_current", "fl_axpy",
)


class _DeviceField:
    """GPU 版の状態変数 (n_e / n_i / w / phi / q_surf) の属性。デバイスが正で、読むとホストへ写す。

    読んだ配列はその場で書き換えられるかもしれないので、読み書きのどちらでも「ホスト側が新しい
    かもしれない」印を付け、次のステップの前にデバイスへ戻す (_GpuFluid.push)。
    """

    def __set_name__(self, owner, name: str) -> None:
        self.name = name
        self.key = "_host_" + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        d = obj.__dict__
        g = d.get("_g")
        if g is not None:
            if self.name not in g.dirty and d.get(self.key + "_ver") != g.version:
                d[self.key] = g.download(self.name)
                d[self.key + "_ver"] = g.version
            g.dirty.add(self.name)
        return d.get(self.key)

    def __set__(self, obj, value) -> None:
        d = obj.__dict__
        d[self.key] = value
        g = d.get("_g")
        if g is not None:
            g.dirty.add(self.name)


class GpuCartesianFluid2dSimulation(CartesianFluid2dSimulation):
    """直交格子 EB 版の流体 2D を GPU で進める (外部インターフェースは v1 Fluid2dSimulation と同じ)。"""

    _allow_lu = False
    n_e = _DeviceField()
    n_i = _DeviceField()
    w = _DeviceField()
    phi = _DeviceField()
    q_surf = _DeviceField()

    def __init__(self, project: Project, device: Device | str | None = None):
        dev = device if isinstance(device, Device) else get_device(device if device is not None else "cuda")
        if not dev.is_gpu:
            raise RuntimeError("GpuCartesianFluid2dSimulation には CUDA デバイスが必要です")
        super().__init__(project, explicit=False, device=dev)
        self._g = _GpuFluid(self)

    # ---- 1 ステップ (v1 Fluid2dSimulation.step と同じサブステップ分割) ---------------------------

    def step(self, should_stop=None):
        """1 ステップ (v1 の step と同じ判定: サブステップ数の上限超過は状態を変えずに ValueError、
        サブステップの合間の停止要求はステップ開始時の状態へ戻して None)。

        戻り値は φ のデバイス配列 (run_batch は None を途中停止と見なし、発散検出は _state_finite)。
        """
        g = self._g
        g.push()
        dt = self.dt
        t = self.t
        tau_d = EPS0 / (QE * max(g.ctl_nemu, 1.0e-300))
        dt_diff_bound = 0.5 * self.h_min**2 / max(g.ctl_de, 1.0e-300)
        n_sub = self._substep_count(
            dt, (("誘電緩和時間", 0.5 * tau_d), ("拡散 CFL", dt_diff_bound), ("Joule 加熱", 0.5 * g.ctl_joule))
        )
        accumulating = self._accum_start is not None and self.step_count + 1 >= self._accum_start
        if accumulating:
            self._ensure_accumulators()
        dt_sub = dt / n_sub
        saved = self._save_step_state() if should_stop is not None and n_sub > 1 else None
        for k in range(n_sub):
            if saved is not None and k > 0 and should_stop():
                self._restore_step_state(saved)
                return None
            g.substep(dt_sub, t + k * dt_sub)
        self.t = t + dt
        self.step_count += 1
        g.end_step(self._phase_bin(t) if (accumulating and self._cycle_phi is not None) else None, accumulating)
        if accumulating:
            self._accum_count += 1
        h = self.history
        h["step"].append(self.step_count)
        h["t"].append(self.t)
        h["n_e_total"].append(g.ne_total)
        h["n_i_total"].append(g.ni_total)
        h["wall_e"].append(self.wall["electron"])
        h["wall_i"].append(self.wall["ion"])
        h["gen_total"].append(self.gen_total)
        h["surf_q"].append(g.qsurf_total)   # 誘電体の表面電荷の合計 (デバイスで集約、属性を読まない)
        return g.phi

    def _state_finite(self, phi) -> bool:
        return self._g.n_bad == 0

    def _save_step_state(self):
        """ステップ開始時のデバイスの状態 (n_e・n_i・w・φ・誘電体の表面電荷と壁損失・生成の積算) の複製。"""
        g = self._g
        with g.stream:
            dev = tuple(a.copy() for a in (g.ne, g.ni, g.w, g.phi, g.acc_wg, g.qsurf))
        return dev, (None if self.circuit is None else self.circuit.snapshot())

    def _restore_step_state(self, saved) -> None:
        """退避した状態へその場で戻す (CUDA Graph が配列のポインタを持つので差し替えない)。

        ホスト側の壁損失・生成の累計・全量・制御量はステップの終わりにしか更新しないので戻す必要が無い。
        阻止コンデンサの回路 (ホスト) は退避した写しへ戻す。
        """
        dev, circuit = saved
        g = self._g
        with g.stream:
            for dst, src in zip((g.ne, g.ni, g.w, g.phi, g.acc_wg, g.qsurf), dev):
                dst[...] = src
            g.version += 1           # 途中のサブステップで読まれたホスト側の写しを無効にする
        if circuit is not None:
            self.circuit.restore(circuit)

    # ---- 時間平均・位相分解 (デバイスで積算し、取り出すときに v1 の形へ) ---------------------------

    def enable_density_accum(self, start_step: int) -> None:
        super().enable_density_accum(start_step)
        self._g.acc = None

    def _ensure_accumulators(self) -> None:
        g = self._g
        if g.acc is not None:
            return
        cp = g.cp
        nb = self._cycle_bins if self._cycle_enabled else 0
        g.acc = {
            "phi": cp.zeros(self.n_nodes), "e": cp.zeros((len(self.tris), 2)), "ne": cp.zeros(self.n_active),
            "ni": cp.zeros(self.n_active), "te": cp.zeros(self.n_active), "ion": cp.zeros(self.n_active),
        }
        if nb:
            g.acc.update({
                "c_phi": cp.zeros((nb, self.n_nodes)), "c_ne": cp.zeros((nb, self.n_active)),
                "c_ni": cp.zeros((nb, self.n_active)), "c_te": cp.zeros((nb, self.n_active)),
            })
            self._cycle_count = np.zeros(nb, dtype=np.int64)
        # v1 の _ensure_accumulators の「確保済み」判定・cycle_data の有無判定に使われる印
        self._accum_phi = np.zeros(0)
        self._cycle_phi = np.zeros(0) if nb else None

    def _pull_accumulators(self) -> None:
        a = self._g.acc
        if a is None:
            return
        self._accum_phi = a["phi"].get()
        self._accum_e = a["e"].get()
        self._accum_ne = a["ne"].get()
        self._accum_ni = a["ni"].get()
        self._accum_te = a["te"].get()
        self._accum_ion = a["ion"].get()
        if "c_phi" in a:
            self._cycle_phi = a["c_phi"].get()
            self._cycle_ne = a["c_ne"].get()
            self._cycle_ni = a["c_ni"].get()
            self._cycle_te = a["c_te"].get()

    def averaged_fields(self):
        self._pull_accumulators()
        return super().averaged_fields()

    def cycle_data(self):
        self._pull_accumulators()
        return super().cycle_data()

    def prepare_continue(self, extra_steps, frame_every=None, avg_steps=None, phase_bins=None) -> None:
        super().prepare_continue(extra_steps, frame_every, avg_steps, phase_bins)
        self._g.acc = None


class _EbPoisson:
    """一様格子の EB Poisson (GMG-PCG。未知数 4096 以下は密な逆行列で厳密に)。"""

    def __init__(self, g: "_GpuFluid", sim) -> None:
        cp = g.cp
        op = sim._op
        self.g = g
        self.op = op
        self.shape = op.grid.shape
        i32 = np.int32
        cpl = op.coupling.tocsr()
        self.g_ptr, self.g_idx, self.g_val = (cp.asarray(cpl.indptr.astype(i32)), cp.asarray(cpl.indices.astype(i32)),
                                              cp.asarray(cpl.data.astype(np.float64)))
        self.qs = cp.asarray(np.ascontiguousarray(op.q_static.ravel(), dtype=np.float64))
        fidx = np.nonzero(op.mask.ravel() == MASK_FIXED)[0]
        fgrp = op.fixed_group.ravel()[fidx].astype(np.int64)
        fgrp = np.where(fgrp >= 0, fgrp, g.n_groups)      # 孤立節点 (グループ無し) は末尾の 0 V
        self.fidx, self.fgrp = cp.asarray(fidx.astype(i32)), cp.asarray(fgrp.astype(i32))
        self.b2 = g.b.reshape(self.shape)
        solver = sim._gmg
        solver.enable_async()
        self.solver = solver
        self.direct = solver.direct
        self.mask = solver.levels[0].mask.ravel()
        self.x2 = cp.zeros(self.shape)

    def solve(self) -> None:
        g = self.g
        sim = g.sim
        g.k["fl_poisson_rhs"](*g._g1(g.nn), (
            self.qs, g.cp_ptr, g.cp_idx, g.cp_val, g.ni, g.ne, self.g_ptr, self.g_idx, self.g_val,
            g.vgrp, self.mask, g.qsurf, np.float64(g.qdiv), np.int32(g.nn), g.b))
        solver = self.solver
        if solver.direct:
            solver.launch_solve(self.b2, self.x2, 1)            # 密な逆行列で厳密に (同期なし)
        elif solver.singular:
            # Dirichlet の無い特異な問題は右辺の整合化・解の正規化がある同期版で解く
            x, info = solver.solve(self.b2, x0=self.x2, tol=POISSON_TOL)
            self.x2[...] = x
            sim.poisson_iters += info.iterations
            g._poisson_check(info.converged, info.relative_residual)
        else:
            g._pcg_loop("pcg", lambda kk: solver.launch_solve(self.b2, self.x2, kk), solver.monitor)
        g.phi[...] = self.x2.ravel()
        if self.fidx.size:
            g.k["fl_fixed"](*g._g1(self.fidx.size), (g.phi, self.fidx, self.fgrp, g.vgrp, np.int32(self.fidx.size)))

    def shift(self, dv) -> None:
        """阻止コンデンサの電極の電位の変化 dv を ψ の重ね合わせで解に足し、固定節点を埋め直す (prompts/134)。

        解のバッファ x2 も直すので、次のサブステップの PCG はこの解から始まる。
        """
        g = self.g
        x = self.x2.ravel()
        for psi, d in zip(g.cap_psi, dv):
            if d != 0.0:
                g.k["fl_axpy"](*g._g1(x.size), (x, psi, np.float64(d), np.int64(x.size)))
        g.phi[...] = x
        if self.fidx.size:
            g.k["fl_fixed"](*g._g1(self.fidx.size), (g.phi, self.fidx, self.fgrp, g.vgrp, np.int32(self.fidx.size)))


class _AmrPoisson:
    """AMR の合成格子の Poisson (gfluid.amr_graph): b_c = PTq [q; V]、GPU の AMG-PCG、φ = PC [x_c; V]。

    疎行列と列ベクトルの積は amr.cu の csr_spmv (cupyx.scipy.sparse は cuSPARSE を使い、配布版は NVRTC だけを
    同梱するため、prompts/133)。
    """

    def __init__(self, g: "_GpuFluid", sim) -> None:
        from ..amr.gpu_solver import AmgGpuSolver, _Csr
        from ..device.cuda import get_kernel

        cp = g.cp
        lay = sim._lay
        op = lay.op
        self.g = g
        self.K = lay.n_groups
        self.PTq = _Csr.of(lay.PTq, cp)
        self.PC = _Csr.of(lay.PC, cp)
        self.k_spmv = get_kernel("amr", "csr_spmv")
        self.qs = cp.asarray(np.ascontiguousarray(lay.q_static, dtype=np.float64))
        n = g.nn
        self.g_ptr = cp.zeros(n + 1, dtype=np.int32)          # 結合は PTq が受け持つ (右辺には足さない)
        self.g_idx = cp.zeros(1, dtype=np.int32)
        self.g_val = cp.zeros(1)
        self.mask = cp.zeros(n, dtype=np.uint8)
        self.ext = cp.zeros(n + self.K)
        n_c = max(op.A_c.shape[0], 1)
        self.bc = cp.zeros(n_c)
        self.xc = cp.zeros(n_c)
        self.xv = cp.zeros(n_c + self.K)
        self.solver = AmgGpuSolver(op.A_c, sim._device, singular=op.singular)
        self.direct = self.solver.direct

    def solve(self) -> None:
        g = self.g
        n = g.nn
        g.k["fl_poisson_rhs"](*g._g1(n), (
            self.qs, g.cp_ptr, g.cp_idx, g.cp_val, g.ni, g.ne, self.g_ptr, self.g_idx, self.g_val,
            g.vgrp, self.mask, g.qsurf, np.float64(g.qdiv), np.int32(n), self.ext[:n]))
        self.ext[n:] = g.vgrp[: self.K]
        self._spmv(self.bc, self.PTq, self.ext)
        solver = self.solver
        if solver.n == 0:
            pass
        elif solver.direct:
            solver.launch_solve(self.bc, self.xc, 1)
        else:
            g._pcg_loop("amg", lambda kk: solver.launch_solve(self.bc, self.xc, kk), solver.monitor)
        self.xv[: self.xc.size] = self.xc
        self.xv[self.xc.size:] = g.vgrp[: self.K]
        self._spmv(g.phi, self.PC, self.xv)

    def _spmv(self, y, m, x) -> None:
        """y = m x (m は _Csr、y の長さは m の行数)。"""
        if m.n:
            self.k_spmv(((m.n + 255) // 256,), (256,), (y, m.ip, m.ix, m.a, x, np.int32(m.n), np.int32(0)))


class _GpuFluid:
    """GpuCartesianFluid2dSimulation のデバイス側の配列・カーネルと 1 サブステップの手順。

    GPU の処理は専用の非ブロッキングストリームに積む。PCG (Poisson) の反復列と BiCGSTAB の 1 反復は
    CUDA Graph に取り込んで 1 回の起動で再生する (カーネル起動の Python 側のコストが律速のため)。
    そのため解のバッファ・係数配列はポインタが変わらない固定の配列を使う。
    """

    def __init__(self, sim: GpuCartesianFluid2dSimulation):
        import cupy as cp

        from ..device.cuda import load_module

        self.cp = cp
        mod = load_module("fluid")
        self.k = {name: mod.get_function(name) for name in _KERNELS}
        self.sim = sim
        self.dirty: set[str] = set()
        self.version = 0
        self.acc: dict | None = None
        self.use_graph = True
        self._graphs: dict = {}

        g = sim.graph
        n = sim.n_active
        nn = sim.n_nodes
        self.n, self.nn = n, nn
        act = sim.active_idx
        f64 = np.float64
        i32 = np.int32

        def dev(a, dtype=f64):
            return cp.asarray(np.ascontiguousarray(a, dtype=dtype))

        # ---- 状態 (アクティブ節点。phi は全節点) ----
        d = sim.__dict__
        self.ne = dev(d["_host_n_e"][act])
        self.ni = dev(d["_host_n_i"][act])
        self.w = dev(d["_host_w"][act])
        self.phi = dev(d["_host_phi"])
        self.qsurf = dev(d["_host_q_surf"])      # 誘電体の表面電荷 (全節点、2π 込み、prompts/129)
        self.vol = dev(sim.node_vol)

        # ---- 辺と接続リスト ----
        ei, ej = sim.i_idx.astype(np.int64), sim.j_idx.astype(np.int64)
        n_edges = len(ei)
        self.n_edges = n_edges
        self.ei, self.ej = dev(ei, i32), dev(ej, i32)
        self.gi, self.gj = dev(act[ei], i32), dev(act[ej], i32)
        self.wij = dev(sim.w_ij)
        self.elen = dev(sim._edge_len)
        rows = np.concatenate([ei, ej])
        codes = np.concatenate([2 * np.arange(n_edges), 2 * np.arange(n_edges) + 1])
        others = np.concatenate([ej, ei])
        order = np.argsort(rows, kind="stable")
        self.rp = dev(np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n))]), i32)
        self.ic = dev(codes[order], i32)
        self.io = dev(others[order], i32)

        # ---- 壁 (節点順に並べ替え) ----
        loc = sim._glob_to_loc[g.wall_node]
        wo = np.argsort(loc, kind="stable")
        self.n_wall = len(wo)
        self.wptr = dev(np.concatenate([[0], np.cumsum(np.bincount(loc, minlength=n))]), i32)
        self.warea = dev(g.wall_area[wo])
        self.asum = dev(np.bincount(loc, weights=g.wall_area, minlength=n))
        self.waidx = dev(g.wall_a_idx[wo], i32)
        self.wawt = dev(g.wall_a_wt[wo])
        self.wwidx = dev(g.wall_w_idx[wo], i32)
        self.wwwt = dev(g.wall_w_wt[wo])
        self.wgrp = dev(g.wall_group[wo], i32)
        self.wlen = dev(g.wall_len[wo])
        self.gamma = dev(sim.gamma_see)
        # 誘電体表面の帯電 (prompts/129): 帯電する小片の印 (節点順) と節点ごとのその面積の合計。
        # 表面電荷は小片を受け持つ輸送節点 (の全節点番号 gidx) に置く (CPU 版 _set_surface_charge_ends と同じ)
        chg = np.zeros(len(loc))
        chg[sim.surf_elem] = 1.0
        self.wchg = dev(chg[wo])
        self.sarea = dev(np.bincount(sim.surf_loc, weights=sim.surf_area, minlength=n))
        self.n_surf = int(sim.surf_elem.size)
        self.gidx = dev(act, i32)
        self.qdiv = 2.0 * math.pi if sim.rz else 1.0
        self.qsum = cp.zeros(NB)
        self.qsurf_total = 0.0

        # ---- Poisson (電荷の写像は共通、解き方は一様格子の EB / AMR の合成格子で別) ----
        cm = g.charge_map[:, act].tocsr()
        self.cp_ptr, self.cp_idx, self.cp_val = dev(cm.indptr, i32), dev(cm.indices, i32), dev(cm.data)
        self.n_groups = len(sim.model.groups)
        self.vgrp = cp.zeros(self.n_groups + 1)
        self.b = cp.zeros(nn)
        self.pcg_k = PCG_ITERS_START
        self.poisson = _AmrPoisson(self, sim) if getattr(sim, "_lay", None) is not None else _EbPoisson(self, sim)
        if sim.circuit is not None:
            self._init_circuit(act, wo)

        # ---- 係数表 ----
        s = sim.s
        if sim.electron_model == "boltzmann":
            bz = sim._boltz
            self.model = 1
            self.grid = dev(bz.eps_grid_ev)
            self.tabs = [dev(bz.mobility_n), dev(bz.k_ion), dev(bz.k_exc), dev(bz.e_ion_ev), dev(bz.e_exc_ev)]
            self.e_ion_c = self.e_exc_c = 0.0
        else:
            r = sim.reactions
            self.model = 0
            self.grid = dev(r.te_grid_ev)
            self.tabs = [dev(r.transport.mobility_n), dev(r.k_ion), dev(r.k_exc), dev(r.transport.nu_m_per_ng),
                         dev(r.k_exc)]
            self.e_ion_c, self.e_exc_c = float(r.e_ion_ev), float(r.e_exc_ev)
        self.ng_tab = int(self.grid.size)
        self.tg_ev = s.gas_temperature_k * KB / QE
        self.v_th_i4 = 0.25 * math.sqrt(8.0 * sim.t_i_ev * QE / (math.pi * sim.m_ion))

        # ---- 作業配列 (ポインタ固定) ----
        for name in ("te", "mue", "kion", "kexc", "num", "eion", "eexc", "sion", "loss", "ce", "wdi", "wde",
                     "dg", "dinv", "rhs", "gwi", "see", "gwe", "wdw", "joule", "zero", "scratch",
                     "ni_new", "ne_new", "w_new", "r", "r0", "p", "ph", "v", "s", "shat", "t"):
            setattr(self, name, cp.zeros(n))
        for name in ("dphi", "ai", "bi", "ae", "be", "a5", "b5"):
            setattr(self, name, cp.zeros(n_edges))
        self.ci = cp.zeros(max(self.n_wall, 1))
        self.part = cp.zeros(7 * NB)
        self.nemax = cp.zeros(NB)
        self.S = cp.zeros(8)
        self.stat = cp.zeros(9 * NB + 2)   # [統計 6 × NB | 壁損失・生成の積算 3 × NB | 係数表範囲外 | 予備]
        self.acc_wg = self.stat[6 * NB:9 * NB]
        self.flag = self.stat[9 * NB:]
        # 種ごとの (解のバッファ, 辺の係数 a, b)
        self.species = {"i": (self.ni_new, self.ai, self.bi), "e": (self.ne_new, self.ae, self.be),
                        "w": (self.w_new, self.a5, self.b5)}

        # ---- 表示用三角形の E (時間平均の e_abs 用) ----
        tris = sim.tris
        self.n_tri = len(tris)
        self.tri = dev(tris, i32)
        self.tb, self.tc, self.tdet = dev(sim._bc_b), dev(sim._bc_c), dev(sim._bc_det)

        self.n_bad = 0
        self.ne_total = self.ni_total = 0.0
        self._te_warned = False
        self.stream = cp.cuda.Stream(non_blocking=True)
        cp.cuda.Device().synchronize()    # 既定ストリームでの転送を終えてから専用ストリームで使う
        with self.stream:
            self.update_control()

    # ---- ホストとの受け渡し ---------------------------------------------------------------

    def download(self, name: str) -> np.ndarray:
        with self.stream:
            if name == "phi":
                return self.phi.get()
            if name == "q_surf":
                return self.qsurf.get()
            out = np.zeros(self.nn)
            out[self.sim.active_idx] = getattr(self, {"n_e": "ne", "n_i": "ni", "w": "w"}[name]).get()
            return out

    def push(self) -> None:
        """ホスト側で代入・書き換えられた状態をデバイスへ戻す (戻したら制御量を計算し直す)。"""
        if not self.dirty:
            return
        d = self.sim.__dict__
        act = self.sim.active_idx
        with self.stream:
            for name in self.dirty:
                host = np.asarray(d["_host_" + name], dtype=np.float64)
                if name == "phi":
                    self.phi.set(np.ascontiguousarray(host))
                elif name == "q_surf":
                    self.qsurf.set(np.ascontiguousarray(host))
                else:
                    getattr(self, {"n_e": "ne", "n_i": "ni", "w": "w"}[name]).set(np.ascontiguousarray(host[act]))
                d["_host_" + name + "_ver"] = self.version
            self.dirty.clear()
            self.update_control()

    # ---- カーネル呼び出し ---------------------------------------------------------------

    @staticmethod
    def _g1(m: int):
        return ((max(1, (int(m) + 255) // 256),), (256,))

    def _graph(self, key, launch) -> None:
        """launch() が積むカーネル列を CUDA Graph に取り込んで (初回) 再生する。専用ストリーム上で呼ぶこと。"""
        if not self.use_graph:
            launch()
            return
        graph = self._graphs.get(key)
        if graph is None:
            self.stream.begin_capture()
            try:
                launch()
            finally:
                graph = self.stream.end_capture()
            self._graphs[key] = graph
        graph.launch(self.stream)

    def _coeffs(self) -> None:
        t = self.tabs
        self.k["fl_coeffs"](*self._g1(self.n), (
            self.ne, self.w, np.int32(self.n), self.grid, np.int32(self.ng_tab), t[0], t[1], t[2], t[3], t[4],
            np.int32(self.model), np.float64(self.sim.n_g), np.float64(self.e_ion_c), np.float64(self.e_exc_c),
            self.te, self.mue, self.kion, self.kexc, self.num, self.eion, self.eexc, self.flag))

    def _edges(self) -> None:
        sim = self.sim
        frost = sim.ion_mobility_model != "const"
        self.k["fl_edges"](*self._g1(self.n_edges), (
            self.phi, self.gi, self.gj, self.ei, self.ej, self.wij, self.elen, self.te, self.mue,
            np.int32(self.n_edges), np.float64(sim.t_i_ev), np.float64(sim.d_i), np.int32(frost),
            np.float64(sim.n_g), np.float64(sim.frost_c_td), self.dphi, self.ai, self.bi, self.ae, self.be))

    def _poisson(self, t: float) -> None:
        sim = self.sim
        vg = np.zeros(self.n_groups + 1)
        if self.n_groups:
            vg[:self.n_groups] = group_voltages(sim.model, t)
        circuit = sim.circuit
        if circuit is not None:
            # 阻止コンデンサの電極は前のサブステップの電位で解く (CPU 版の _solve_phi と同じ手順)
            v_src = np.array([vg[groups[0]] for groups in sim._cap_groups])
            v_old = circuit.potentials(v_src)
            vg[:self.n_groups] = sim._cap_set(vg[:self.n_groups], v_old)
        self.vgrp.set(vg)
        self.poisson.solve()
        if circuit is not None:
            v_new = circuit.solve(self._cap_charges(), v_old, v_src)
            vg = vg.copy()
            vg[:self.n_groups] = sim._cap_set(vg[:self.n_groups], v_new)
            self.vgrp.set(vg)
            self.poisson.shift(v_new - v_old)

    # ---- 阻止コンデンサ (prompts/134): CPU 版 (gfluid.simulation) と同じ量をデバイスで集約する ------------

    def _init_circuit(self, act: np.ndarray, wo: np.ndarray) -> None:
        """電極の電荷と伝導電流の集計の表 (電極ごとに連続に並べ、区間ごとに和を取る: 決定的)。"""
        cp = self.cp
        sim = self.sim
        g = sim.graph
        m = sim.circuit.m

        def offsets(elec):
            return cp.asarray(np.concatenate([[0], np.cumsum(np.bincount(elec, minlength=m))]).astype(np.int64))

        def i64(a):
            return cp.asarray(np.ascontiguousarray(a, dtype=np.int64))

        # 電束 Σ G (V − φ): 結合のうち電極のもの
        rows, cols, data, ej = sim._cap_coo
        o = np.argsort(ej, kind="stable")
        self.cap_rows, self.cap_cols = i64(rows[o]), i64(cols[o])
        self.cap_data = cp.asarray(np.ascontiguousarray(data[o], dtype=np.float64))
        self.cap_flux_off = offsets(ej)
        # 電極の固定節点の電荷: 固定の電荷 (定数) + e·w·(n_i − n_e) (charge_map の行の和) + 表面電荷 / qdiv
        nodes, fe = sim._cap_fixed_nodes, sim._cap_fixed_elec
        self.cap_static = np.bincount(fe, weights=sim._q_static[nodes], minlength=m)
        cm = g.charge_map[:, act].tocsr()
        widx, wval = [], []
        for j in range(m):
            row = np.asarray(cm[nodes[fe == j]].sum(axis=0)).ravel()
            nz = np.nonzero(row)[0]
            widx.append(nz)
            wval.append(row[nz])
        self.cap_widx = i64(np.concatenate(widx))
        self.cap_wval = cp.asarray(np.ascontiguousarray(np.concatenate(wval), dtype=np.float64))
        self.cap_woff = i64(np.concatenate([[0], np.cumsum([a.size for a in widx])]))
        o = np.argsort(fe, kind="stable")
        self.cap_fnodes = i64(nodes[o])
        self.cap_foff = offsets(fe)
        # ψ (全節点、解のバッファに x += ΔV ψ で足す)
        self.cap_psi = [cp.asarray(np.ascontiguousarray(sim._cap_psi[j], dtype=np.float64)) for j in range(m)]
        # 伝導電流: 行き先が電極の壁の小片 (節点順に並べ替えた位置) と、その節点 (アクティブ節点の番号)
        inv = np.empty(len(wo), dtype=np.int64)
        inv[wo] = np.arange(len(wo))
        o = np.argsort(sim._cap_wall_elec, kind="stable")
        self.cap_piece = i64(inv[sim._cap_wall][o])
        self.cap_ploc = i64(sim._glob_to_loc[g.wall_node[sim._cap_wall]][o])
        self.cap_poff = offsets(sim._cap_wall_elec)
        self.cap_part = cp.zeros(3 * NB)          # fl_cap_charge / fl_cap_current の電極ごとの和 (slot × NB + j)

    def _cap_charges(self) -> np.ndarray:
        """今の φ (デバイス) での電極の表面の電荷 (CPU 版の _cap_charges と同じ量)。"""
        sim = self.sim
        m = sim.circuit.m
        self.k["fl_cap_charge"]((m,), (NT,), (
            self.phi, self.vgrp, self.cap_rows, self.cap_cols, self.cap_data, self.cap_flux_off,
            self.ni, self.ne, self.cap_widx, self.cap_wval, self.cap_woff, self.qsurf, self.cap_fnodes,
            self.cap_foff, self.cap_part))
        s = self.cap_part.get().reshape(3, NB)[:, :m]
        own = self.cap_static + QE * s[1] + s[2] / self.qdiv
        return sim._cap_factor * (s[0] - own)

    def _cap_current(self) -> np.ndarray:
        """電極へ流れ込む伝導電流 (壁の小片ごとの e·(Γ_i(1 + γ) − Γ_e) の和、CPU 版の _accumulate_circuit と同じ)。"""
        sim = self.sim
        m = sim.circuit.m
        if sim.debug_reflective_walls or self.cap_piece.size == 0:
            return np.zeros(m)
        self.k["fl_cap_current"]((m,), (NT,), (
            self.warea, self.ci, self.ni_new, self.ne_new, self.gamma, self.ce, self.cap_piece, self.cap_ploc,
            self.cap_poff, self.cap_part))
        return QE * self.cap_part[:m].get()

    def _pcg_loop(self, key, launch, monitor) -> None:
        """前のサブステップの解から pcg_k 反復ずつ (グラフで再生) 積み、残差を 1 回読んで判定する。"""
        sim = self.sim
        k = self.pcg_k
        total = 0
        while True:
            self._graph((key, k), lambda kk=k: launch(kk))
            total += k
            rn, bn = monitor()
            rel = rn / bn if bn > 0.0 else 0.0
            if rel <= POISSON_TOL or total >= PCG_ITERS_MAX:
                break
            k = 2
        sim.poisson_iters += total
        self._poisson_check(rel <= POISSON_TOL, rel)
        if total > self.pcg_k:
            self.pcg_k = min(total, PCG_ITERS_MAX)
        elif rel < 1e-2 * POISSON_TOL and self.pcg_k > 1:
            self.pcg_k -= 1

    def _poisson_check(self, converged: bool, rel: float) -> None:
        sim = self.sim
        if not converged and not sim._poisson_warned:
            sim.warnings.append(
                f"Poisson (GMG-PCG) が収束しませんでした (相対残差 {rel:.2e}、このメッセージは初回のみ)"
            )
            sim._poisson_warned = True

    def _solve(self, sp: str, nold, src, sgn: float, extra, wd, dt: float) -> None:
        """種 sp の (V/dt + wd + K(a, b)) x = V/dt n_old + sgn·src·V + extra を解く (初期値 n_old)。"""
        sim = self.sim
        x, ca, cb = self.species[sp]
        n = np.int32(self.n)
        g1 = self._g1(self.n)
        self.k["fl_diag"](*g1, (self.vol, np.float64(dt), wd, ca, cb, self.rp, self.ic, n, self.dg, self.dinv))
        self.k["fl_rhs"](*g1, (self.vol, np.float64(dt), nold, src, np.float64(sgn), extra, n, self.rhs))
        x[...] = nold
        it, ok = self._bicgstab(sp, x, ca, cb)
        sim.timing["solver_iters"] += it
        if ok:
            return
        # 収束しなければ v1 と同じく直接法 (spsolve) へ (初回のみ warnings に記録)
        if not sim._solver_fallback_warned:
            sim.warnings.append(
                "反復ソルバー (Jacobi-BiCGSTAB) が収束しませんでした。spsolve (直接法) へ"
                "フォールバックしました (このメッセージは初回のみ表示。以降の発生回数は "
                "solver_fallback_count に集計されます)"
            )
            sim._solver_fallback_warned = True
        sim.solver_fallback_count += 1
        m = sim._assemble_active_matrix(ca.get(), cb.get(), wd.get(), dt)
        x.set(np.ascontiguousarray(spla.spsolve(m, self.rhs.get())))

    def _bcg_iter(self, x, ca, cb, first: bool) -> None:
        k = self.k
        n = np.int32(self.n)
        gr = ((NB,), (NT,))
        mat = (self.dg, ca, cb, self.rp, self.ic, self.io)
        k["bcg_p"](*gr, (self.p, self.ph, self.r, self.v, self.dinv, self.S, self.part, n, np.int32(first)))
        k["bcg_v"](*gr, (self.ph, self.r0, *mat, n, self.v, self.part))
        k["bcg_s"](*gr, (self.r, self.v, self.dinv, self.S, self.part, n, self.s, self.shat))
        k["bcg_t"](*gr, (self.shat, self.s, *mat, n, self.t, self.part))
        k["bcg_x"](*gr, (x, self.r, self.s, self.t, self.ph, self.shat, self.r0, self.S, self.part, n))

    def _bicgstab(self, sp: str, x, ca, cb) -> tuple[int, bool]:
        n = np.int32(self.n)
        mat = (self.dg, ca, cb, self.rp, self.ic, self.io)
        self.k["bcg_init"]((NB,), (NT,), (x, self.rhs, *mat, n, self.r, self.r0, self.part))
        pb = self.part[_P_RR * NB:(_P_BB + 1) * NB].get()
        rr, bb = float(pb[:NB].sum()), float(pb[NB:].sum())
        tol = max(BICG_RTOL * math.sqrt(bb), BICG_ATOL)
        if math.sqrt(rr) <= tol:
            return 0, True
        self.S.set(np.array([0.0, 1.0, 1.0, 1.0, tol * tol, 0.0, 0.0, 0.0]))
        for it in range(1, BICG_MAX_ITER + 1):
            first = it == 1
            self._graph(("bcg", sp, first), lambda f=first: self._bcg_iter(x, ca, cb, f))
            rr = float(self.part[_P_RR * NB:(_P_RR + 1) * NB].get().sum())
            if not math.isfinite(rr):
                return it, False
            if math.sqrt(rr) <= tol:
                return it, True
        return BICG_MAX_ITER, False

    # ---- 1 サブステップ (v1 Fluid2dSimulation._step_once と同じ手順) ------------------------------

    def substep(self, dt: float, t: float) -> None:
        with self.stream:
            self._substep(dt, t)

    def _substep(self, dt: float, t: float) -> None:
        sim = self.sim
        k = self.k
        n = np.int32(self.n)
        g1 = self._g1(self.n)
        reflective = np.int32(bool(sim.debug_reflective_walls))
        t0 = time.perf_counter()
        self._poisson(t)
        self._coeffs()
        self._edges()
        k["fl_nodes"](*g1, (
            self.ne, self.te, self.kion, self.kexc, self.num, self.eion, self.eexc, n, np.float64(sim.n_g),
            np.float64(self.tg_ev), np.float64(sim._mass_ratio), np.int32(bool(sim.debug_source_enabled)),
            self.sion, self.loss, self.ce))
        if self.n_wall:
            k["fl_wall_ci"](*self._g1(self.n_wall), (
                self.phi, self.waidx, self.wawt, self.wwidx, self.wwwt, self.wgrp, self.wlen, self.vgrp,
                np.int32(self.n_wall), np.int32(sim.ion_mobility_model != "const"), np.float64(sim.mu_i),
                np.float64(sim.n_g), np.float64(sim.frost_c_td), np.float64(self.v_th_i4), reflective, self.ci))
        k["fl_wall_diag"](*g1, (self.ci, self.warea, self.wptr, self.ce, self.asum, n, reflective, self.wdi,
                                self.wde))
        t1 = time.perf_counter()
        sim.timing["poisson"] += t1 - t0

        # ---- イオン ----
        if sim.debug_ions_enabled:
            self._solve("i", self.ni, self.sion, 1.0, self.zero, self.wdi, dt)
            k["fl_floor"](*g1, (self.ni_new, np.float64(FLOOR_N), n))
        else:
            self.ni_new[...] = self.ni
        k["fl_see"](*g1, (self.wdi, self.ni_new, self.gamma, n, reflective, self.gwi, self.see))

        # ---- 電子 (SEE はいま求めたイオン壁流束を使う) ----
        self._solve("e", self.ne, self.sion, 1.0, self.see, self.wde, dt)
        k["fl_floor"](*g1, (self.ne_new, np.float64(FLOOR_N), n))
        k["fl_gwe"](*g1, (self.wde, self.ne_new, self.see, n, reflective, self.gwe, self.wdw))
        # 誘電体の表面電荷 (壁損失と同じ流束。次のサブステップの Poisson から効く、CPU 版と同じ)
        if self.n_surf and not sim.debug_reflective_walls:
            k["fl_surf"](*g1, (self.ci, self.warea, self.wchg, self.wptr, self.sarea, self.ni_new, self.ne_new,
                               self.gamma, self.ce, self.gidx, n, np.float64(QE * dt), self.qsurf))
        # 阻止コンデンサの電極へ流れ込む伝導電流 (同じ流束、prompts/134)
        if sim.circuit is not None:
            sim.circuit.conduct(self._cap_current(), t, dt)
        t2 = time.perf_counter()
        sim.timing["transport"] += t2 - t1

        # ---- エネルギー (Joule 加熱は新しい n_e のエッジ電力の折半) ----
        if sim.debug_energy_enabled:
            k["fl_joule"](*g1, (self.ae, self.be, self.ne_new, self.dphi, self.ei, self.ej, self.rp, self.ic, n,
                                self.joule))
            k["fl_scale2"](*self._g1(self.n_edges), (self.ae, self.be, np.float64(5.0 / 3.0),
                                                     np.int32(self.n_edges), self.a5, self.b5))
            self._solve("w", self.w, self.loss, -1.0, self.joule, self.wdw, dt)
            k["fl_floor"](*g1, (self.w_new, np.float64(FLOOR_W), n))
        else:
            k["fl_energy_off"](*g1, (self.ne_new, np.float64(1.5 * sim.s.init_te_ev), n, self.w_new))
        t3 = time.perf_counter()
        sim.timing["energy"] += t3 - t2

        k["fl_wallgen"]((NB,), (NT,), (self.gwi, self.gwe, self.sion, self.vol, np.float64(dt), n, self.acc_wg))
        self.ni[...] = self.ni_new
        self.ne[...] = self.ne_new
        self.w[...] = self.w_new
        self.version += 1
        sim.timing["other"] += time.perf_counter() - t3

    # ---- ステップの終わり: 時間平均・統計・次のステップの制御量 (同期 1 回) ---------------------------

    def end_step(self, cycle_bin: int | None, accumulating: bool) -> None:
        t0 = time.perf_counter()
        sim = self.sim
        with self.stream:
            self._coeffs()
            self._edges()
            n = np.int32(self.n)
            g1 = self._g1(self.n)
            self.k["fl_joule"](*g1, (self.ae, self.be, self.ne, self.dphi, self.ei, self.ej, self.rp, self.ic, n,
                                     self.joule))
            if accumulating and self.acc is not None:
                a = self.acc
                gm = self._g1(max(self.nn, self.n))
                src_on = np.int32(bool(sim.debug_source_enabled))
                self.k["fl_accum"](*gm, (self.phi, np.int32(self.nn), self.ne, self.ni, self.te, self.kion, n,
                                         np.float64(sim.n_g), src_on, a["phi"], a["ne"], a["ni"], a["te"],
                                         a["ion"]))
                self.k["fl_tri_e"](*self._g1(self.n_tri), (self.phi, self.tri, self.tb, self.tc, self.tdet,
                                                            np.int32(self.n_tri), a["e"]))
                if cycle_bin is not None and "c_phi" in a:
                    b = int(cycle_bin)
                    self.k["fl_accum"](*gm, (self.phi, np.int32(self.nn), self.ne, self.ni, self.te, self.kion, n,
                                             np.float64(sim.n_g), src_on, a["c_phi"][b], a["c_ne"][b],
                                             a["c_ni"][b], a["c_te"][b], self.scratch))
                    sim._cycle_count[b] += 1
            self._stats()
        sim.timing["other"] += time.perf_counter() - t0

    def update_control(self) -> None:
        """現在の状態からサブステップの制御量・全量を求める (初期化時・ホストから状態を戻したとき)。"""
        self._coeffs()
        self._edges()
        self.k["fl_joule"](*self._g1(self.n), (self.ae, self.be, self.ne, self.dphi, self.ei, self.ej, self.rp,
                                               self.ic, np.int32(self.n), self.joule))
        self._stats()

    def _stats(self) -> None:
        sim = self.sim
        n = np.int32(self.n)
        self.k["fl_nemax"]((NB,), (NT,), (self.ne, n, self.nemax))
        self.k["fl_stats"]((NB,), (NT,), (self.ne, self.ni, self.w, self.vol, self.mue, self.te, self.joule,
                                         n, self.phi, np.int32(self.nn), self.nemax, self.stat))
        self.k["fl_sum"]((NB,), (NT,), (self.qsurf, np.int32(self.nn), self.qsum))
        h = self.stat.get()
        self.qsurf_total = float(self.qsum.get().sum())
        self.ne_total = float(h[0:NB].sum())
        self.ni_total = float(h[NB:2 * NB].sum())
        self.n_bad = int(h[2 * NB:3 * NB].sum())
        self.ctl_nemu = float(h[3 * NB:4 * NB].max())
        self.ctl_de = float(h[4 * NB:5 * NB].max())
        m3 = float(h[5 * NB:6 * NB].min())
        self.ctl_joule = math.inf if m3 >= 1.0e307 else m3
        sim.wall["ion"] = float(h[6 * NB:7 * NB].sum())
        sim.wall["electron"] = float(h[7 * NB:8 * NB].sum())
        sim.gen_total = float(h[8 * NB:9 * NB].sum())
        if h[9 * NB] > 0.0 and not self._te_warned:
            grid = self.grid.get()
            key = "Te" if self.model == 0 else "ε̄=(3/2)Te"
            src = "係数テーブル" if self.model == 0 else "boltzpm テーブル"
            warnings.warn(
                f"fluid2d: {key} が{src}範囲 [{grid[0]:.3g}, {grid[-1]:.3g}] eV の外に出ました "
                "(テーブル引きはクランプして継続します)"
            )
            self._te_warned = True
