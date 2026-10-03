"""v2 GPU PIC-MCC — 直交格子 + 埋め込み境界 (EB) 上の 2d3v / 軸対称 PIC (prompts/119 P3)。

v1 ``pic.PicSimulation`` と**同じ外部インターフェース** (run_batch / fields / cycle /
collector_results / eedf_results / timing / prepare_continue / mesh / dt / warnings) を持ち、
server の /ws/pic からそのまま駆動できる (mesh.mode="cartesian" のとき)。結果は v1 UI が
描ける三角形メッシュ (直交格子のセルを 2 分割、geometry.DisplayMesh) 上の値で返す。

## 高速化の設計 (AMReX / WarpX の GPU 実装と同じ考え方)

- **ホスト同期ゼロの 1 ステップ**: 粒子数・カウンタ・時刻・電極電位 V(t)・適応 ν_max・
  平均化フラグ・位相ビンを全て GPU メモリに置き、カーネルがそれを読む。1 ステップは
  固定のカーネル列になり、**CUDA Graph** として一度記録して毎ステップ 1 回の起動で再生する
  (Windows の WDDM ではホスト同期 1 回あたりの待ちが大きく、同期を残すと小規模問題で
  CPU 版より遅くなるため)。
- 粒子は容量固定の SoA 配列。吸収粒子は w = 0 にして次の圧縮 (COMPACT_EVERY ステップ
  ごと、順序保存) で詰める。電離・二次電子の生成粒子は atomicAdd で末尾に追加する。
- 容量 (下の「粒子の容量」): 配列は軟らかい容量 + 予備。軟らかい容量を越える追加があると次のステップから
  デバイスがステップを止め、ホストが次の検査で配列を広げて止まったステップをやり直す (粒子を失わず、
  1 ステップごとの同期も要らない)。検査のたびに直近の追加の速さから先回りして広げる。
- デバイスの配列への書き込み (確保・拡張・粒子の書き換え・パラメータ) は全てステップと同じストリームに
  積む (既定のストリームに積むと非ブロッキングのストリームのグラフと順序が保証されず、拡張した配列を
  コピーし終わる前にステップが読むことがあった)。
- 吸収粒子の後処理 (衝突点・法線の厳密計算、誘電体の表面電荷、SEE、IEDF コレクタ) も
  GPU 上で行う。
- 場: 未知数 ≤ 4096 の小さな問題は密な逆行列で厳密に、それ以上は GMG-PCG を固定反復
  (前ステップ解から warm start、残差を定期監視して反復数を自動調整)。
- 診断 (history) は GPU 上のリングバッファに書き、まとめてホストへ転送する。

## AMR (prompts/122)

mesh.amr で実際に細分化が起きるなら、AMR 階層の合成格子で同じステップを実行する
(amr.pic_layout の表・行列と、kernels/pic.cu を -DES_AMR でコンパイルした変種)。
粒子の所属セルはブロック表で O(レベル数) に求め (es_locate)、電荷は葉セルの 4 隅へ CIC で堆積、
場は b_c = PTq [q; V] → GPU AMG-PCG (amr.gpu_solver) → φ = PC [x_c; V] → E = G [φ; V]
(節点電場ステンシル、amr.fieldops) → 葉セルの CIC 補間。細分化が無ければ一様格子の経路と
機械精度で一致する。粗細界面には運動量保存型でも自己力が残る (大きさは prompts/122)。

## 物理 (v1 と同じ所・違う所)

同じ: 2d3v リープフロッグ (初期半ステップ後退キック)、一様 B の Boris 回転 (xy のみ)、
null-collision MCC (電子: 弾性/励起/電離、イオン: 等方/電荷交換、ガス質量 = イオン質量、
電離の余剰エネルギー分配 half/random、イオンの lab/com 参照エネルギー、適応 ν_max)、
電極・誘電体の SEE、誘電体の表面電荷、IEDF/IADF コレクタ、EEDF 領域、時間平均・
RF 位相分解、イオンサブサイクリング、quiet start、既定 dt = 0.1/ωpe、
イオン質量 = amu × 陽子質量 (v1 と同じ換算)、軸対称の押し出し (3D 直線移動→子午面へ回転、
角運動量厳密保存、軸で r ≥ 0 のまま。v1 も同じ回転法)。

違う:
- 場: 節点 FV Poisson (EB ゴーストフルイド)。v1 は P1-FEM + 前分解 LU。
- 形状関数: 双一次 (CIC) の堆積と節点電場の双一次補間 (運動量保存型)。v1 は P1 堆積と
  要素一定電場 (エネルギー保存型)。
- 粒子位置の特定: 格子なので O(1) (v1 は隣接要素 walk)。
- 乱数: カウンタ方式 Philox (GPU)。統計的には等価だがビット一致はしない。電荷堆積・
  粒子追加は atomic なので同じシードでも実行ごとに丸め誤差レベルで変わる。
- 電極 SEE・誘電体 SEE の放出方向: どちらも衝突点の法線方向 (v1 は誘電体のみ
  入射と逆方向の近似)。
- 時間平均の電子温度: 双一次重みで堆積した w·v² から評価 (v1 は P1 重み)。

## 阻止コンデンサ (自己バイアス、prompts/134)

回路もデバイスで進め、1 ステップは CUDA Graph のまま: 堆積のあと cap_induced が節点の電荷の誘導電荷
Q_e0 = Σ W·q (Green の相反定理。一様格子は W = −2π ψ̃、AMR は合成格子の電極の電荷の汎関数を随伴で解いた重み) を
電極ごとに決定的な和で求め、cap_solve が V_e = (C + C_b)⁻¹ (Q_N − Q_e0 + C_b V_s) をグループ電位 vg に書く
(右辺を組む前なので Poisson は 1 回、V_e は PCG の反復回数によらない)。境界のカーネルが吸収した粒子と電極から
出た二次電子の電荷を電極ごとに atomicAdd で数え、cap_step が Q_N に足して V_e・電流をデバイスの履歴に書く。
周期の集計 (V_dc・|V1|・I_dc) はホストの circuit.BlockingCircuit が履歴を読むとき (_flush_history) に行う。
AMR の再格子化では W と容量行列を作り直す (Q_N はそのまま)。

## 粒子の容量

種ごとに軟らかい容量 cap と配列の長さ alloc = cap + 予備 (CAP_RESERVE_MIN 個か cap の CAP_RESERVE_FRAC 倍の
大きい方) を持ち、どちらもデバイスのカウンタ (C_SOFT_*・C_LIM_*) に置く。

1. 追加 (電離・二次電子) が cap 以上の位置に入ったら停止を要求する (C_HALT_REQ)。そのステップは最後まで進め、
   次のステップの begin_step から停止する: 状態を変えるカーネル (押し出し・境界・MCC・平均) は何もせず、
   ステップ番号も進まない (場の求解は止まった粒子のまま回る)。
2. ホストは同期する所 (COMPACT_EVERY ごとの圧縮、フレーム、履歴の転送など) で必ずカウンタを読む
   (_capacity_check)。停止していれば、止まっていたステップの分だけホストのステップ数・時刻・平均の回数を
   巻き戻し、配列を広げて停止を解き、同じステップからやり直す。結果は初めから容量が足りていた場合と同じ
   (粒子の並びを変えないよう、圧縮の予定はそのまま)。
3. 圧縮のたびに、直近の追加の速さ (電離・二次電子の数の差分 / ステップ) で CAP_AHEAD 回分の圧縮間隔の追加が
   入るように先回りして広げる (旧来の n > CAP_HIGH·cap の規則も残す)。
4. 予備まで使い切った (1 ステップの追加が予備を越えた) ときだけ粒子が失われ (C_OVERFLOW)、エラーにする。
   カウンタは配列の長さを越えて増え得るので、カーネルは粒子数を配列の長さで切る。

未対応 (指定するとエラー): 粒子注入 (injection)、FN 電界放出、粒子マージ、
DSMC ガス場連成 (use_dsmc_gas)。
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from ..circuit import BlockingCircuit, model_capacitor_electrodes, rf_period
from ..device import Device, get_device
from ..eb.build import MASK_FIXED, MASK_SLAVE
from ..eb.grid import make_grid
from ..field.electrostatic import fill_fixed
from ..field.gmg import GMGSolver
from ..geom.model import EPS0, GeometryModel
from ..mcc import MccModel
from ..particles import ME, MP, QE, _boris_matrix, b_vector
from ..schema import Project
from .geometry import (
    HIT_CONDUCTOR,
    HIT_DIELECTRIC,
    DisplayMesh,
    ParticleGeometry,
    cell_gas_volume,
    triangle_gradients,
)

MAX_FRAME_PARTICLES = 2000
COLLECTOR_MAX_SAMPLES = 50000
#: フレーム生成の最小間隔 [s] (GPU は速いので frame_every ごとだと JSON 化が律速になる)
MIN_FRAME_INTERVAL_S = 0.25
#: 位相分解データ (bins × 節点数) の上限。超えたら位相ビン数を減らす
CYCLE_MAX_VALUES = 4_000_000
#: 吸収粒子の圧縮・容量検査の間隔 [ステップ]
COMPACT_EVERY = 32
#: 初めの軟らかい容量 max(CAP_INIT·n_macro, n_macro + CAP_HEADROOM_MIN)
CAP_INIT = 2.0
CAP_HEADROOM_MIN = 65536
#: 容量の余裕 (圧縮時に n > CAP_HIGH·cap なら cap を CAP_GROW·n + CAP_GROW_MIN に拡張)
CAP_HIGH = 0.75
CAP_GROW = 1.6
CAP_GROW_MIN = 1024
#: 予備の容量 (1 ステップの追加の分): 配列の長さ = cap + max(CAP_RESERVE_MIN, CAP_RESERVE_FRAC·cap)
CAP_RESERVE_MIN = 65536
CAP_RESERVE_FRAC = 0.125
#: 先回りの拡張: 直近の追加の速さで CAP_AHEAD 回分の圧縮間隔の追加が入る余裕を保つ。速さの記憶は
#: CAP_RATE_HALFLIFE ステップで半分に減る
CAP_AHEAD = 4
CAP_RATE_HALFLIFE = 256
#: GPU 上の history リングバッファの行数
HISTORY_ROWS = 2048
#: PCG の反復数の初期値・上限と、残差監視の間隔 [ステップ]
PCG_ITERS_INIT = 3
PCG_ITERS_MAX = 30
PCG_TOL = 1e-8
MONITOR_EVERY = 64
#: 動的再格子化のタグ用に電子の密度・温度を積算する間隔 [ステップ]
TAG_EVERY = 8
#: 阻止コンデンサ (prompts/134): 電極の数の上限 (kernels/pic.cu の CAP_MAX)、誘導電荷の部分和のブロック数、
#: ψ の Poisson の収束判定、合成格子の ψ を疎行列の直接法で解く未知数の上限 (超えたら AMG-CG)
CAP_MAX = 16
CAP_NB = 64
CAP_PSI_TOL = 1e-12
CAP_HOST_DIRECT_MAX = 240_000

_BLOCK = 256
_HIST_COLS = (
    "t", "ke_e", "ke_i", "fe", "n_e", "n_i", "wall_e", "wall_i", "phi_min", "phi_max",
    "coll_e", "ion_events", "see_events", "surf_q",
)
_HISTORY_KEYS = _HIST_COLS + ("fn_i", "fn_events", "merged")
_TIMING_KEYS = ("solve", "gather_push", "walk", "deposit", "mcc", "other", "frame")

# prm (double) の位置 — kernels/pic.cu の P_* と一致させること
P_T, P_STEP, P_DT, P_ACCUM, P_BIN, P_ION_STEP = 0, 1, 2, 3, 4, 5
P_NUMAX_E, P_PCAND_E, P_NUMAX_I, P_PCAND_I = 6, 7, 8, 9
P_ACC_START, P_PERIOD, P_NBINS, P_SUB, P_T0, P_STEP0 = 10, 11, 12, 13, 14, 15
P_HALT = 16
_N_PRM = 24
# cnt (uint64) の位置 — kernels/pic.cu の C_* と一致させること
C_NE, C_NI, C_ION_EV, C_SEE_EV, C_OVERFLOW = 0, 1, 7, 8, 9
C_HALT_REQ, C_HALT, C_HALT_STEP, C_LIM_E, C_LIM_I, C_SOFT_E, C_SOFT_I = 16, 17, 18, 19, 20, 21, 22
_N_CNT = 24


def _reserve(cap: int) -> int:
    """軟らかい容量 cap の上に置く予備の粒子数 (1 ステップの追加の分、モジュール docstring の「粒子の容量」)。"""
    return max(CAP_RESERVE_MIN, int(CAP_RESERVE_FRAC * cap))


class _Species:
    """GPU 上の粒子 (SoA、容量固定)。v[2] は xy なら vz、軸対称なら vθ。数はデバイスの cnt[index]。

    cap は軟らかい容量、alloc は配列の長さ (cap + 予備、モジュール docstring の「粒子の容量」)。
    確保・コピーは呼び出し側がステップのストリームの中で行う。
    """

    FIELDS = ("x", "y", "vx", "vy", "vz", "w")

    def __init__(self, name: str, index: int, q: float, m: float, mobile: bool, cp, cap: int, alloc: int):
        self.name = name
        self.index = index
        self.q = q
        self.m = m
        self.mobile = mobile
        self._cp = cp
        self.cap = 0
        self.alloc = 0
        self.arrays: dict[str, object] = {}
        self._alloc(cap, alloc)

    def _alloc(self, cap: int, alloc: int) -> None:
        cp = self._cp
        old = self.arrays
        self.arrays = {k: cp.zeros(alloc) for k in self.FIELDS}
        if old:
            n = min(self.alloc, alloc)
            for k in self.FIELDS:
                self.arrays[k][:n] = old[k][:n]
        self.cap = cap
        self.alloc = alloc

    def __getattr__(self, item):
        arrays = self.__dict__.get("arrays")
        if arrays is not None and item in arrays:
            return arrays[item]
        raise AttributeError(item)


@dataclass
class _DisplayMeshView:
    nodes: np.ndarray
    triangles: np.ndarray
    tri_region: np.ndarray = field(default_factory=lambda: np.zeros(0))


class GpuPicSimulation:
    """v2 GPU PIC-MCC。コンストラクタは v1 PicSimulation と同じ (project, gas_field)。"""

    def __init__(self, project: Project, gas_field=None, device: Device | str | None = None):
        if project.pic is None:
            raise ValueError("project.pic が指定されていません")
        dev = device if isinstance(device, Device) else get_device(device or "cuda")
        if not dev.is_gpu:
            raise RuntimeError(
                "v2 PIC (mesh.mode='cartesian') は現状 GPU (CUDA) 専用です。"
                "CPU で実行する場合はメッシュを unstructured/structured にしてください (v1 PIC)"
            )
        import cupy as cp

        from ..device.cuda import load_module

        self.cp = cp
        self.device = dev
        self.project = project
        self.pic = pic = project.pic
        self._check_supported(project, gas_field)
        self._use_graph = os.environ.get("ES_SIM_NO_GRAPH") != "1"
        self._stream = cp.cuda.Stream(non_blocking=True)

        t_setup = time.perf_counter()
        self.model = model = GeometryModel(project)
        self.grid = grid = make_grid(model.domain, float(project.mesh.size))
        # mesh.amr で実際に細分化が起きるなら AMR 階層の合成格子で解く (prompts/122)
        self.amr = self._build_amr_layout(project, model, grid)
        with self._stream:
            if self.amr is None:
                self.solver = GMGSolver(model, grid, dev)
                self.solver.enable_async()
                op = self.op = self.solver.finest
            else:
                from ..amr.gpu_solver import AmgGpuSolver

                op = self.op = self.amr.op
                self.solver = AmgGpuSolver(op.A_c, dev, singular=op.singular)
        self.pgeo = ParticleGeometry(model, grid)
        self.ridx = model.radial_axis()
        self.rz = self.ridx is not None
        self._two_pi = 2.0 * math.pi if self.rz else 1.0
        self.warnings: list[str] = list(op.warnings)
        self.effective_threads = 1
        mod = load_module("pic", ("ES_AMR",) if self.amr is not None else ())
        names = ["begin_step", "end_step", "eval_groups", "push", "boundary", "deposit", "accum_phi",
                 "deposit_cell", "vmax2", "numax_lookup", "mcc_electron", "mcc_ion", "history_row",
                 "sum_into", "eedf_hist", "compact_scan", "compact_scan_blocks", "compact_scatter",
                 "cap_induced", "cap_solve", "cap_step"]
        if self.amr is None:
            names += ["rhs_base", "rhs_coupling", "fill_phi", "efield_nodes", "edge_energy"]
        else:
            names += ["amr_rhs", "amr_fill_phi", "amr_efield", "amr_edge_energy", "kick_half"]
        self._k = {name: mod.get_function(name) for name in names}
        self._k_copy = load_module("poisson").get_function("copy_vec")

        g = grid
        self.n_nodes = g.n_nodes if self.amr is None else self.amr.n_nodes
        self._shape = g.shape if self.amr is None else (self.amr.n_nodes,)
        self._inv_dx = 1.0 / g.dx
        self._inv_dy = 1.0 / g.dy
        self._px = int(model.periodic_x)
        self._py = int(model.periodic_y)
        self._grid_args = (np.float64(g.x0), np.float64(g.y0), np.float64(self._inv_dx), np.float64(self._inv_dy),
                           np.int32(g.nx), np.int32(g.ny))

        # ---- dt・プラズマパラメータ (v1 と同じ規約) -------------------------------------
        ip = pic.initial_plasma
        amu = ip.ion_mass_amu if ip is not None else 40.0
        self.m_ion = amu * MP
        wpe = math.sqrt(ip.density * QE**2 / (EPS0 * ME)) if ip is not None else 0.0
        if pic.dt is not None:
            self.dt = float(pic.dt)
        elif wpe > 0.0:
            self.dt = 0.1 / wpe
        else:
            raise ValueError("pic.dt を指定してください (初期密度が無いため自動決定できません)")
        self._stability_warnings(ip, wpe)
        self._sub = max(1, int(pic.ion_subcycle))

        with self._stream:
            self._setup_device_state(ip)
        self._pcg_iters = PCG_ITERS_INIT
        self._graph = None
        self._graph_key = None

        # ---- 診断 ----------------------------------------------------------------------
        self.t = 0.0
        self.step_count = 0
        self.history: dict[str, list[float]] = {k: [] for k in _HISTORY_KEYS}
        self.timing: dict[str, float] = {k: 0.0 for k in _TIMING_KEYS}
        self._hist_pending = 0
        self.fields = None
        self.cycle = None
        self.collector_results = None
        self.collector_result = None
        self.eedf_results = None
        self._last_frame_wall = -math.inf
        self._reset_accumulators()

        # ---- 粒子の容量の検査 (モジュール docstring の「粒子の容量」) -----------------------------
        self._checked_step = -1          # カウンタを読んで停止を解いたステップ (同じステップの再検査を省く)
        self._chk_step = 0               # 前の検査のデバイスのステップ数と電離・二次電子の累計
        self._chk_ev = (0, 0)
        self._cap_rate = {"electron": 0.0, "ion": 0.0}   # 追加の速さ [粒子/ステップ]
        self.capacity_log: list[dict] = []               # 停止からのやり直し (step・止まったステップ数・容量)

        # ---- 動的再格子化 (prompts/123) ------------------------------------------------------
        self.mesh_version = 0            # 再格子化のたびに増える (フレーム・done に新しい格子を添える)
        self._frame_mesh_version = 0
        self.regrid_log: list[dict] = []
        self._regrid_every = self.amr.hier.spec.pic_regrid_every if self.amr is not None else 0
        self._tag_count = 0
        self._tag_steps: list[int] = []  # 前の検査の後にタグを積算したステップ (停止で巻き戻したら数え直す)
        if self._regrid_every:
            with self._stream:
                self._tag_w = self.cp.zeros(self._shape)
                self._tag_ke = self.cp.zeros(self._shape)
        self._self_force_warning()

        # ---- 初期半ステップ後退キック (t=0 の場で v を -dt/2 へ) -----------------------
        with self._stream:
            self._kick_backward_half()
        self._stream.synchronize()
        self.setup_s = time.perf_counter() - t_setup

    # ======================================================================================
    # 設定の検査・補助
    # ======================================================================================

    @staticmethod
    def _check_supported(project: Project, gas_field) -> None:
        pic = project.pic
        unsupported = []
        if pic.injection is not None:
            unsupported.append("粒子注入 (injection)")
        if pic.fn is not None:
            unsupported.append("FN 電界放出 (fn)")
        if pic.merge is not None:
            unsupported.append("粒子マージ (merge)")
        if gas_field is not None or (pic.mcc is not None and pic.mcc.use_dsmc_gas):
            unsupported.append("DSMC ガス場 (use_dsmc_gas)")
        if unsupported:
            raise ValueError(
                "v2 PIC (mesh.mode='cartesian') は次の機能に未対応です: " + "、".join(unsupported)
                + " (v1 PIC を使う場合はメッシュを unstructured/structured にしてください)"
            )

    @staticmethod
    def _build_amr_layout(project: Project, model: GeometryModel, grid):
        """mesh.amr が細分化を起こす (または動的再格子化が有効) なら AMR PIC の表、それ以外は None。"""
        amr = project.mesh.amr
        if amr is None or (amr.max_level <= 0 and not amr.regions):
            return None
        from ..amr import AmrHierarchy, AmrSpec
        from ..amr.pic_layout import build_pic_layout

        spec = AmrSpec.from_settings(amr)
        hier = AmrHierarchy(model, grid, spec)
        if hier.max_level == 0 and spec.pic_regrid_every <= 0:
            return None
        return build_pic_layout(model, hier)

    def _disp(self, a: np.ndarray) -> np.ndarray:
        """節点値 (最後の軸) を表示用メッシュの節点へ。

        AMR は表示節点 → 合成格子の節点。一様格子は同じ並びだが、周期境界の従属節点
        (右端・上端) には堆積が入らない (主節点へ巻き戻す) ので主節点の値を写す。
        """
        if self.amr is not None:
            return a[..., self.amr.disp_idx]
        if not (self._px or self._py):
            return a
        g = self.grid
        idx = np.arange((g.ny + 1) * (g.nx + 1)).reshape(g.ny + 1, g.nx + 1)
        if self._px:
            idx[:, g.nx] = idx[:, 0]
        if self._py:
            idx[g.ny, :] = idx[0, :]
        return a[..., idx.ravel()]

    def _side_kinds(self) -> list[int]:
        """外周の粒子境界 [left, right, bottom, top]: 0 吸収、1 鏡面反射、2 周期。"""
        m = self.model
        reflect = set(self.pic.reflect_edges)
        kinds = {}
        for side, s in m.sides.items():
            if s.kind == "periodic":
                kinds[side] = 2
            elif s.kind == "symmetry" or s.edge in reflect:
                kinds[side] = 1
            else:
                kinds[side] = 0
        if self.ridx == 1 and m.domain.y0 <= m.tol:
            kinds["bottom"] = 1
        elif self.ridx == 0 and m.domain.x0 <= m.tol:
            kinds["left"] = 1
        return [kinds["left"], kinds["right"], kinds["bottom"], kinds["top"]]

    def _find_rf_freq(self) -> float | None:
        for gr in self.model.groups:
            if gr.rf:
                return float(gr.rf[0].freq_hz)
        for gr in self.model.groups:
            if gr.waveform is not None:
                return float(gr.waveform.freq_hz)
        return None

    def _stability_warnings(self, ip, wpe: float) -> None:
        if wpe <= 0.0:
            return
        if wpe * self.dt > 0.3:
            self.warnings.append(f"ωpe·dt = {wpe * self.dt:.3g} > 0.3: 時間刻みが粗すぎます (数値不安定の恐れ)")
        if ip.te_ev > 0.0:
            lam_d = math.sqrt(EPS0 * ip.te_ev * QE / (ip.density * QE**2))
            h = max(self.grid.dx, self.grid.dy)
            if h > 3.0 * lam_d:
                self.warnings.append(
                    f"セルサイズ {h:.3g} m > 3×デバイ長 {lam_d:.3g} m: メッシュがデバイ長を解像していません"
                )

    def _cell_gas_volume(self) -> np.ndarray:
        """セルごとの気体体積 (物理単位、軸対称は 2π 込み) — フレームの要素密度用。"""
        return cell_gas_volume(self.model, self.grid, self.pgeo.cell_state)

    # ======================================================================================
    # デバイス状態の構築
    # ======================================================================================

    def _setup_device_state(self, ip) -> None:
        cp = self.cp
        op = self.op
        model = self.model
        pic = self.pic
        shp = self._shape

        # ---- 場・右辺 ----
        self._q_surf = cp.zeros(shp)
        self._rho = cp.zeros(shp)
        self._phi = cp.zeros(shp)
        self._ex = cp.zeros(shp)
        self._ey = cp.zeros(shp)
        self._gx = ()
        if self.amr is None:
            coo = op.coupling.tocoo()
            self._crow = cp.asarray(coo.row.astype(np.int64))
            self._ccol = cp.asarray(coo.col.astype(np.int64))
            self._cval = cp.asarray(coo.data.astype(np.float64))
            self._nnz = int(coo.nnz)
            self._q_static = cp.asarray(op.q_static)
            self._b = cp.zeros(shp)
            self._x = cp.zeros(shp)
            self._fixed = cp.asarray((op.mask == MASK_FIXED).astype(np.uint8))
            self._fgrp = cp.asarray(np.maximum(op.fixed_group, 0).astype(np.int64))
            self._mask = cp.asarray(op.mask)
            self._cut_theta = cp.asarray(op.cut_theta)
            self._cut_group = cp.asarray(np.maximum(op.cut_group, 0).astype(np.int32))
            self._cx = self.solver.levels[0].cx
            self._cy = self.solver.levels[0].cy
            vol_gas = op.vol_gas
            # 周期の従属節点 (右端列・上端行) は主節点と同じ体積を複製して持つので総和から除く
            nyv = vol_gas.shape[0] - (1 if op.periodic_y else 0)
            nxv = vol_gas.shape[1] - (1 if op.periodic_x else 0)
            self._total_gas_volume = float(self._two_pi * vol_gas[:nyv, :nxv].sum())
            vol_phys = self._two_pi * vol_gas
            self._inv_vol_phys = cp.asarray(np.where(vol_phys > 0.0, 1.0 / np.where(vol_phys > 0, vol_phys, 1.0), 0.0))
            self._vol_gas = vol_gas
        else:
            self._setup_amr_fields(self.amr)

        # ---- Dirichlet グループの電位波形 ----
        groups = model.groups
        n_g = max(len(groups), 1)
        self._n_groups = len(groups)
        dc = np.zeros(n_g)
        rf_off = [0]
        rf_amp, rf_om, rf_ph = [], [], []
        wf_off = [0]
        wf_freq = np.zeros(n_g)
        wf_ph, wf_v = [], []
        for k, gr in enumerate(groups):
            dc[k] = gr.voltage
            for c in gr.rf:
                rf_amp.append(c.amplitude)
                rf_om.append(2.0 * math.pi * c.freq_hz)
                rf_ph.append(math.radians(c.phase_deg))
            rf_off.append(len(rf_amp))
            if gr.waveform is not None:
                wf_ph.extend(gr.waveform.phase)
                wf_v.extend(gr.waveform.v)
                wf_freq[k] = gr.waveform.freq_hz
            wf_off.append(len(wf_ph))
        while len(rf_off) < n_g + 1:
            rf_off.append(rf_off[-1])
            wf_off.append(wf_off[-1])
        f64 = lambda a: cp.asarray(np.asarray(a if len(a) else [0.0], dtype=np.float64))  # noqa: E731
        self._g_dc = cp.asarray(dc)
        self._g_rf_off = cp.asarray(np.asarray(rf_off, dtype=np.int32))
        self._g_rf = (f64(rf_amp), f64(rf_om), f64(rf_ph))
        self._g_wf_off = cp.asarray(np.asarray(wf_off, dtype=np.int32))
        self._g_wf = (cp.asarray(wf_freq), f64(wf_ph), f64(wf_v))
        self._vg = cp.zeros(n_g)

        # ---- 粒子の境界 (固体形状の統一表) ----
        pg = self.pgeo
        s_type, s_kind, s_index, s_off, pxy, circ = [], [], [], [0], [], []
        for s in pg.solids:
            if hasattr(s.shape, "vertices"):
                s_type.append(0)
                pxy.extend(s.shape.vertices.ravel().tolist())
                circ.extend([0.0, 0.0, 0.0])
            else:
                s_type.append(1)
                circ.extend([s.shape.cx, s.shape.cy, s.shape.r])
            s_kind.append(1 if s.kind == HIT_CONDUCTOR else 2)
            s_index.append(s.index)
            s_off.append(len(pxy) // 2)
        i32 = lambda a: cp.asarray(np.asarray(a if len(a) else [0], dtype=np.int32))  # noqa: E731
        self._n_solid = len(pg.solids)
        self._s_type, self._s_kind, self._s_index, self._s_off = i32(s_type), i32(s_kind), i32(s_index), i32(s_off)
        self._s_pxy = f64(pxy)
        self._s_circ = f64(circ)
        self._cell_state = cp.asarray(pg.cell_state)
        self._side_kind = cp.asarray(np.asarray(self._side_kinds(), dtype=np.int32))
        grp_gamma = np.array([gr.see_gamma for gr in groups] + [0.0])
        side_gamma = np.zeros(4)
        for k, side in enumerate(("left", "right", "bottom", "top")):
            gidx = model.side_group.get(side)
            if gidx is not None:
                side_gamma[k] = grp_gamma[gidx]
        self._side_gamma = cp.asarray(side_gamma)
        cg = np.asarray(model.conductor_group, dtype=np.int64)
        self._cond_gamma = f64(grp_gamma[cg] if len(cg) else [])
        self._diel_gamma = f64([o.region.see_gamma if o.type == "dielectric" else 0.0 for o in model.others])
        self._see_on = int(np.any(grp_gamma > 0.0) or np.any(self._diel_gamma.get() > 0.0))
        self._see_speed = math.sqrt(2.0 * pic.see_energy_ev * QE / ME)
        h_min = min(self.grid.dx, self.grid.dy) if self.amr is None else self.amr.h_min
        self._see_delta = 1e-3 * h_min
        self._has_diel = any(o.type == "dielectric" for o in model.others)

        # ---- 表示用メッシュ (v1 UI 互換) ----
        if self.amr is None:
            dm = pg.display_mesh()
            self._cell_vol_gas = self._cell_gas_volume()
            self._n_cells = self.grid.nx * self.grid.ny
            self._dmesh = dm
            self.mesh = _DisplayMeshView(nodes=dm.nodes, triangles=dm.triangles, tri_region=dm.tri_region)
        else:
            self._set_amr_display(self.amr)

        # ---- 一様磁場 (xy のみ) ----
        self._bvec = b_vector(self.project)
        self._R = {}
        if self._bvec is not None:
            if self.rz:
                raise ValueError("軸対称モードでは一様磁場を指定できません")
            for name, q_s, m_s in (("electron", -QE, ME), ("ion", QE, self.m_ion)):
                dt_s = self.dt * self._sub if (name == "ion" and self._sub > 1) else self.dt
                self._R[name] = cp.asarray(_boris_matrix(q_s, m_s, dt_s, self._bvec).ravel())
            wc_e = QE * float(np.linalg.norm(self._bvec)) / ME
            if wc_e * self.dt > 0.3:
                self.warnings.append(f"ωce·dt = {wc_e * self.dt:.3g} > 0.3: 電子サイクロトロン運動を解像できていません")
        self._R_dummy = cp.zeros(9)

        # ---- RF 位相分解 ----
        self._cycle_freq = self._find_rf_freq()
        self._cycle_bins = int(pic.phase_bins)
        if self._cycle_freq is not None and self._cycle_bins * self.n_nodes > CYCLE_MAX_VALUES:
            nb = max(1, CYCLE_MAX_VALUES // self.n_nodes)
            self.warnings.append(
                f"位相分解データが大きすぎるため位相ビン数を {self._cycle_bins} → {nb} に減らしました (節点数 {self.n_nodes})"
            )
            self._cycle_bins = nb
        self._cycle_enabled = self._cycle_freq is not None and self._cycle_bins > 0
        self._cycle_period = 1.0 / self._cycle_freq if self._cycle_enabled else 0.0

        # ---- パラメータ・カウンタ ----
        self._prm = cp.zeros(_N_PRM)
        prm = np.zeros(_N_PRM)
        prm[P_DT] = self.dt
        prm[P_ACC_START] = 1e300
        prm[P_PERIOD] = self._cycle_period if self._cycle_enabled else 0.0
        prm[P_NBINS] = max(self._cycle_bins, 1)
        prm[P_SUB] = self._sub
        self._prm[...] = cp.asarray(prm)
        self._cnt = cp.zeros(_N_CNT, dtype=np.uint64)
        self._dsl = cp.zeros(8)   # 0 ke_e, 1 ke_i, 2 fe, 3 surf_q
        self._hist = cp.zeros((HISTORY_ROWS, len(_HIST_COLS)))

        # ---- 粒子 ----
        n0 = int(pic.n_macro) if ip is not None else 0
        cap = int(max(CAP_INIT * n0, n0 + CAP_HEADROOM_MIN))
        alloc = cap + _reserve(cap)
        self.species = {
            "electron": _Species("electron", 0, -QE, ME, True, cp, cap, alloc),
            "ion": _Species("ion", 1, QE, self.m_ion, not (ip is not None and ip.immobile_ions), cp, cap, alloc),
        }
        self._scan_buffers(alloc)
        self._upload_limits()
        self._w0 = None
        if ip is not None:
            self._load_initial_plasma(ip)

        # ---- MCC ----
        self.mcc = None
        self._mcc_seed = np.uint64(0)
        if pic.mcc is not None:
            self.mcc = MccModel(pic.mcc, self.m_ion)
            self._mcc_seed = np.uint64(int(pic.mcc.seed) & 0xFFFFFFFFFFFFFFFF)
            self._setup_mcc_tables()
            self._check_collision_probability()
        self._see_seed = np.uint64(((pic.mcc.seed if pic.mcc else 0) + 7919) & 0xFFFFFFFFFFFFFFFF)

        # ---- コレクタ・EEDF ----
        self._init_collectors()
        self._init_eedf()

        # ---- 阻止コンデンサ (prompts/134) ----
        self._init_circuit()

    def _setup_amr_fields(self, lay) -> None:
        """AMR の場の表・行列・体積を GPU へ (初期化と再格子化で共通)。"""
        cp = self.cp
        i32 = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.int32))  # noqa: E731

        def dev_csr(m):
            m = m.tocsr()
            m.sort_indices()
            return (i32(m.indptr), i32(m.indices), cp.asarray(m.data.astype(np.float64)))

        self._q_static = cp.asarray(lay.q_static)
        self._b = cp.zeros(max(lay.n_c, 1))
        self._x = cp.zeros(max(lay.n_c, 1))
        self._csr = {name: dev_csr(getattr(lay, name)) for name in ("PTq", "PC", "Gx", "Gy")}
        self._e_lists = (i32(lay.edge_u), i32(lay.edge_v), cp.asarray(lay.edge_g, dtype=np.float64),
                         np.int64(lay.edge_u.size), i32(lay.coup_u), i32(lay.coup_grp),
                         cp.asarray(lay.coup_g, dtype=np.float64), np.int64(lay.coup_u.size))
        self._gx = (cp.asarray(lay.dd), cp.asarray(lay.di), cp.asarray(lay.refined), i32(lay.blockid),
                    i32(lay.tab))
        self._total_gas_volume = float(lay.total_gas_volume)
        vol_phys = lay.node_vol_gas
        self._inv_vol_phys = cp.asarray(np.where(vol_phys > 0.0, 1.0 / np.where(vol_phys > 0, vol_phys, 1.0), 0.0))
        self._vol_gas = vol_phys / self._two_pi

    def _set_amr_display(self, lay) -> None:
        """AMR の表示用メッシュ (v1 UI 互換) と葉セルの気体体積。"""
        dm = DisplayMesh(nodes=lay.disp_nodes, triangles=lay.disp_tris, tri_region=lay.disp_tri_region,
                         tri_cell=lay.tri_cell)
        self._cell_vol_gas = lay.cell_vol_gas
        self._n_cells = lay.n_cells
        self._dmesh = dm
        self.mesh = _DisplayMeshView(nodes=dm.nodes, triangles=dm.triangles, tri_region=dm.tri_region)

    def _scan_buffers(self, alloc: int) -> None:
        """圧縮の作業配列 (ステップのストリームの中で呼ぶ)。"""
        cp = self.cp
        nb = (alloc + 1023) // 1024
        self._scan_off = cp.zeros(nb * 1024, dtype=np.int32)
        self._scan_flag = cp.zeros(nb * 1024, dtype=np.uint8)
        self._scan_bsum = cp.zeros(nb, dtype=np.int64)
        self._tmp = {k: cp.zeros(alloc) for k in _Species.FIELDS}

    def _upload_limits(self) -> None:
        """配列の長さと軟らかい容量をデバイスのカウンタ (C_LIM_*・C_SOFT_*) へ。"""
        el, io = self.species["electron"], self.species["ion"]
        with self._stream:
            self._cnt[C_LIM_E:C_SOFT_I + 1] = self.cp.asarray(
                np.array([el.alloc, io.alloc, el.cap, io.cap], dtype=np.uint64))

    def _load_initial_plasma(self, ip) -> None:
        cp = self.cp
        rng = np.random.default_rng(ip.seed)
        n_macro = int(self.pic.n_macro)
        d = self.model.domain
        xs, ys = [], []
        need = n_macro
        while need > 0:
            m = max(1024, int(need * 1.3) + 64)
            if self.ridx == 1:
                x = rng.uniform(d.x0, d.x1, m)
                y = np.sqrt(rng.uniform(d.y0**2, d.y1**2, m))
            elif self.ridx == 0:
                x = np.sqrt(rng.uniform(d.x0**2, d.x1**2, m))
                y = rng.uniform(d.y0, d.y1, m)
            else:
                x = rng.uniform(d.x0, d.x1, m)
                y = rng.uniform(d.y0, d.y1, m)
            ok = self.model.gas_at(x, y)
            take = int(min(ok.sum(), need))
            xs.append(x[ok][:take])
            ys.append(y[ok][:take])
            need -= take
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        v_gas = self._total_gas_volume
        if v_gas <= 0.0:
            raise ValueError("粒子を装荷できる気体領域がありません")
        w0 = ip.density * v_gas / n_macro
        self._w0 = w0
        for name, m_s, t_ev in (("electron", ME, ip.te_ev), ("ion", self.m_ion, ip.ti_ev)):
            sigma = math.sqrt(t_ev * QE / m_s) if t_ev > 0.0 else 0.0
            v = rng.normal(0.0, sigma, size=(n_macro, 3)) if sigma > 0.0 else np.zeros((n_macro, 3))
            self._write_particles(name, {"x": x, "y": y, "vx": v[:, 0], "vy": v[:, 1], "vz": v[:, 2],
                                         "w": np.full(n_macro, w0)})

    def _write_particles(self, name: str, host: dict[str, np.ndarray]) -> None:
        """ホスト配列で種の粒子を丸ごと置き換える (容量不足なら拡張)。"""
        cp = self.cp
        sp = self.species[name]
        n = len(host["x"])
        if n > CAP_HIGH * sp.cap:
            self._grow(sp, int(CAP_GROW * n) + CAP_GROW_MIN)
        with self._stream:
            for k in _Species.FIELDS:
                a = sp.arrays[k]
                a[:n] = cp.asarray(np.asarray(host[k], dtype=np.float64))
                a[n:] = 0.0
            cnt = self._cnt.get()
            cnt[sp.index] = n
            cnt[14 + sp.index] = cnt[4 + sp.index]   # 吸収カウンタの基準 (生存数 = 格納数 − 基準以後の吸収)
            self._cnt[...] = cp.asarray(cnt)
        self._graph = None

    def _grow(self, sp: _Species, cap: int) -> None:
        """種 sp の軟らかい容量を cap に (配列の長さは予備を足す)。中身はステップのストリームの上でコピーする。"""
        alloc = cap + _reserve(cap)
        with self._stream:
            sp._alloc(cap, alloc)
            alloc_max = max(s.alloc for s in self.species.values())
            if self._scan_off.size < ((alloc_max + 1023) // 1024) * 1024 or self._tmp["x"].size < alloc_max:
                self._scan_buffers(alloc_max)
        self._upload_limits()
        self._graph = None

    def _setup_mcc_tables(self) -> None:
        cp = self.cp
        mc = self.mcc
        kind_code = {"elastic": 0, "excitation": 1, "ionization": 2, "isotropic": 0, "backscat": 1}
        self._me_n = len(mc.e_procs)
        if mc.e_procs:
            e_tab, s_tab, lens = mc._e_xs_pack
            self._me_kind = cp.asarray(np.array([kind_code[p.kind] for p in mc.e_procs], dtype=np.int32))
            self._me_thr = cp.asarray(np.array([p.threshold_ev for p in mc.e_procs], dtype=np.float64))
            self._me_mr = cp.asarray(np.array([p.mass_ratio for p in mc.e_procs], dtype=np.float64))
            self._me_tab_e, self._me_tab_s = cp.asarray(e_tab), cp.asarray(s_tab)
            self._me_len = cp.asarray(lens.astype(np.int32))
            self._me_w = int(e_tab.shape[1])
            grid, pref = mc._nu_e_table
            self._me_pref = cp.asarray(pref)
            self._me_ecap = float(grid[-1])
            self._me_ngrid = len(grid)
        self._mi_n = len(mc.i_procs)
        if mc.i_procs:
            e_tab, s_tab, lens = mc._i_xs_pack
            self._mi_kind = cp.asarray(np.array([kind_code[p.kind] for p in mc.i_procs], dtype=np.int32))
            self._mi_tab_e, self._mi_tab_s = cp.asarray(e_tab), cp.asarray(s_tab)
            self._mi_len = cp.asarray(lens.astype(np.int32))
            self._mi_w = int(e_tab.shape[1])
            grid, pref = mc._nu_i_table
            self._mi_pref = cp.asarray(pref)
            self._mi_ecap = float(grid[-1])
            self._mi_ngrid = len(grid)

    def _check_collision_probability(self) -> None:
        mc = self.mcc
        for label, numax, dt_s in (("電子", mc.numax_e, self.dt), ("イオン", mc.numax_i, self.dt * self._sub)):
            p = 1.0 - math.exp(-numax * dt_s) if numax > 0 else 0.0
            if p > 0.5:
                self.warnings.append(
                    f"{label}の null-collision 候補確率 {p:.2f} > 0.5 (ν_max·dt = {numax * dt_s:.3g}): dt が大きすぎます"
                )

    def _init_collectors(self) -> None:
        cp = self.cp
        tol_default = float(self.project.mesh.size)
        params = []
        for c in self.pic.collectors:
            p1 = np.asarray(c.p1, dtype=np.float64)
            p2 = np.asarray(c.p2, dtype=np.float64)
            seg = p2 - p1
            length = float(np.linalg.norm(seg))
            if length <= 0.0:
                raise ValueError("コレクタ線分の長さが 0 です")
            tan = seg / length
            params.extend([p1[0], p1[1], tan[0], tan[1], -tan[1], tan[0], length,
                           float(c.tol) if c.tol is not None else tol_default])
        self._n_coll = len(self.pic.collectors)
        self._coll = cp.asarray(np.asarray(params if params else [0.0] * 8, dtype=np.float64))
        nc = max(self._n_coll, 1)
        self._rec_e = cp.zeros(nc * COLLECTOR_MAX_SAMPLES)
        self._rec_a = cp.zeros(nc * COLLECTOR_MAX_SAMPLES)
        self._rec_w = cp.zeros(nc * COLLECTOR_MAX_SAMPLES)
        self._rec_n = cp.zeros(nc, dtype=np.uint64)
        self._coll_w = cp.zeros(nc)

    def _init_eedf(self) -> None:
        cp = self.cp
        regs = self.pic.eedf_regions
        self._n_eedf = len(regs)
        self._eedf_bins = max((int(r.bins) for r in regs), default=1)
        self._eedf_auto = [r.e_max_ev is None for r in regs]
        self._eedf_meta = [{"label": r.label, "bins": int(r.bins)} for r in regs]
        reg = []
        for r in regs:
            x0, x1 = sorted((r.p1[0], r.p2[0]))
            y0, y1 = sorted((r.p1[1], r.p2[1]))
            reg.extend([x0, x1, y0, y1, float(r.e_max_ev) if r.e_max_ev is not None else 30.0])
        self._eedf_reg = cp.asarray(np.asarray(reg if reg else [0.0] * 5, dtype=np.float64))
        self._eedf_hist = cp.zeros(max(self._n_eedf, 1) * self._eedf_bins)
        self._eedf_sums = cp.zeros(max(self._n_eedf, 1) * 3)
        self._eedf_samples = 0

    # ======================================================================================
    # カーネル起動 (1 ステップ = 固定のカーネル列)
    # ======================================================================================

    def _grid1(self, n: int) -> tuple[int]:
        return (max(1, (int(n) + _BLOCK - 1) // _BLOCK),)

    def _launch_field(self) -> None:
        """電荷堆積 → 右辺 → 求解 → φ (Dirichlet 値・周期スレーブ込み) → 節点電場・場エネルギー。"""
        from ..device.cuda import grid_2d

        k = self._k
        g = self.grid
        n = self.n_nodes
        k["eval_groups"]((1,), (max(32, self._n_groups),),
                         (self._vg, np.int32(self._n_groups), self._g_dc, self._g_rf_off, *self._g_rf,
                          self._g_wf_off, *self._g_wf, self._prm))
        self._rho.fill(0.0)
        for sp in self.species.values():
            k["deposit"](self._grid1(sp.alloc), (_BLOCK,),
                         (sp.x, sp.y, sp.vx, sp.vy, sp.vz, sp.w, self._cnt, np.int32(sp.index), np.int32(0),
                          np.float64(sp.q / self._two_pi), self._rho, self._rho, np.int32(0), np.int64(n),
                          np.int32(0), self._prm, *self._grid_args, np.int32(self._px), np.int32(self._py),
                          *self._gx))
        if self.circuit is not None:
            self._launch_circuit()
        if self.amr is not None:
            self._launch_field_amr()
            return
        k["rhs_base"](self._grid1(n), (_BLOCK,), (self._b, self._q_static, self._rho, self._q_surf, np.int64(n)))
        if self._nnz:
            k["rhs_coupling"](self._grid1(self._nnz), (_BLOCK,),
                              (self._b, self._crow, self._ccol, self._cval, self._vg, np.int64(self._nnz)))
        self.solver.launch_solve(self._b, self._x, self._pcg_iters)
        gr, bl = grid_2d(g.nx + 1, g.ny + 1)
        k["fill_phi"](gr, bl, (self._phi, self._x, self._fixed, self._fgrp, self._vg, np.int32(g.nx), np.int32(g.ny),
                               np.int32(self._px), np.int32(self._py), self._cnt))
        k["efield_nodes"](gr, bl, (self._ex, self._ey, self._phi, self._mask, self._cut_theta, self._cut_group,
                                   self._vg, np.int32(g.nx), np.int32(g.ny), np.float64(g.dx), np.float64(g.dy),
                                   np.int32(self._px), np.int32(self._py)))
        n_edges = (g.ny + 1) * g.nx + g.ny * (g.nx + 1) + self._nnz
        k["edge_energy"](self._grid1(n_edges), (_BLOCK,),
                         (self._phi, self._cx, self._cy, np.int32(g.nx), np.int32(g.ny), np.int32(self._px),
                          np.int32(self._py), self._crow, self._ccol, self._cval, np.int64(self._nnz), self._vg,
                          self._dsl[2:3]))

    def _launch_field_amr(self) -> None:
        """AMR: b_c = PTq [q; V] → AMG-PCG → φ_all = PC [x_c; V] → E = G [φ; V]・場のエネルギー。"""
        k = self._k
        lay = self.amr
        n, n_c = lay.n_nodes, lay.n_c
        ptq, pc, gx, gy = (self._csr[name] for name in ("PTq", "PC", "Gx", "Gy"))
        if n_c:
            k["amr_rhs"](self._grid1(n_c), (_BLOCK,), (self._b, *ptq, self._q_static, self._rho, self._q_surf,
                                                      self._vg, np.int32(n), np.int32(n_c)))
            self.solver.launch_solve(self._b, self._x, self._pcg_iters)
        k["amr_fill_phi"](self._grid1(n), (_BLOCK,), (self._phi, *pc, self._x, self._vg, np.int32(n_c), np.int32(n),
                                                     self._cnt))
        k["amr_efield"](self._grid1(n), (_BLOCK,), (self._ex, self._ey, *gx, *gy, self._phi, self._vg, np.int32(n)))
        eu, ev, eg, ne, cu, cgrp, cg, nc = self._e_lists
        k["amr_edge_energy"](self._grid1(int(ne) + int(nc)), (_BLOCK,),
                             (self._phi, eu, ev, eg, ne, cu, cgrp, cg, nc, self._vg, self._dsl[2:3]))

    def _launch_push(self, sp: _Species) -> None:
        if not sp.mobile:
            return
        dt_s = self.dt * (self._sub if sp.index == 1 else 1)
        use_b = sp.name in self._R
        self._k["push"](self._grid1(sp.alloc), (_BLOCK,),
                        (sp.x, sp.y, sp.vx, sp.vy, sp.vz, sp.w, self._cnt, np.int32(sp.index), self._ex, self._ey,
                         *self._grid_args, np.float64(sp.q / sp.m), np.float64(dt_s), np.int32(1 if self.rz else 0),
                         np.int32(self.ridx if self.rz else 0), np.int32(1 if use_b else 0),
                         self._R[sp.name] if use_b else self._R_dummy, np.float64(0.5 * sp.m), self._prm,
                         self._dsl[sp.index:sp.index + 1], *self._gx))

    def _launch_boundary(self, sp: _Species) -> None:
        if not sp.mobile:
            return
        d = self.model.domain
        el = self.species["electron"]
        dt_s = self.dt * (self._sub if sp.index == 1 else 1)
        self._k["boundary"](self._grid1(sp.alloc), (_BLOCK,), (
            sp.x, sp.y, sp.vx, sp.vy, sp.vz, sp.w, self._cnt, np.int32(sp.index), np.float64(dt_s),
            np.int32(1 if self.rz else 0), np.int32(self.ridx if self.rz else 0),
            np.float64(d.x0), np.float64(d.y0), np.float64(d.x1), np.float64(d.y1), self._side_kind,
            self._cell_state, np.float64(self._inv_dx), np.float64(self._inv_dy),
            np.int32(self.grid.nx), np.int32(self.grid.ny), np.int32(self._px), np.int32(self._py),
            np.int32(self._n_solid), self._s_type, self._s_kind, self._s_index, self._s_off, self._s_pxy,
            self._s_circ, self._side_gamma, self._cond_gamma, self._diel_gamma,
            self._q_surf, np.float64(sp.q), np.float64(1.0 / self._two_pi),
            np.int32(self._see_on), np.float64(self._see_speed), np.float64(self._see_delta),
            el.x, el.y, el.vx, el.vy, el.vz, el.w,
            np.int32(self._n_coll), self._coll, np.float64(sp.m),
            self._rec_e, self._rec_a, self._rec_w, self._rec_n, self._coll_w, np.int64(COLLECTOR_MAX_SAMPLES),
            self._prm, self._see_seed, self._side_elec, self._cond_elec, self._cap_dq, *self._gx,
        ))

    def _launch_mcc(self) -> None:
        if self.mcc is None:
            return
        k = self._k
        mc = self.mcc
        el, io = self.species["electron"], self.species["ion"]
        g = self.grid
        if self._me_n:
            k["vmax2"](self._grid1(el.alloc), (_BLOCK,), (el.vx, el.vy, el.vz, el.w, self._cnt, np.int32(0), self._prm))
            k["numax_lookup"]((1,), (1,), (self._prm, self._cnt, np.int32(0), np.float64(ME), np.float64(mc.mu),
                                           np.float64(mc.vth_gas), np.int32(0), self._me_pref,
                                           np.int32(self._me_ngrid), np.float64(self._me_ecap), np.float64(self.dt)))
            use_cyc = self._cyc_ion is not None
            k["mcc_electron"](self._grid1(el.alloc), (_BLOCK,), (
                el.x, el.y, el.vx, el.vy, el.vz, el.w,
                io.x, io.y, io.vx, io.vy, io.vz, io.w,
                self._cnt, np.int32(self._me_n), self._me_kind, self._me_thr, self._me_mr, self._me_tab_e,
                self._me_tab_s, self._me_len, np.int32(self._me_w), np.float64(mc.n_gas),
                np.int32(1 if self.pic.mcc.ionization_split == "half" else 0), np.float64(mc.vth_gas),
                self._prm, self._mcc_seed,
                self._acc_ion, self._cyc_ion if use_cyc else self._acc_ion, np.int32(1 if use_cyc else 0),
                np.int64(self.n_nodes), *self._grid_args, np.int32(self._px), np.int32(self._py), *self._gx,
            ))
        if self._mi_n and io.mobile:
            dt_i = self.dt * self._sub
            k["vmax2"](self._grid1(io.alloc), (_BLOCK,), (io.vx, io.vy, io.vz, io.w, self._cnt, np.int32(1), self._prm))
            k["numax_lookup"]((1,), (1,), (self._prm, self._cnt, np.int32(1), np.float64(mc.m_ion),
                                           np.float64(mc.mu), np.float64(mc.vth_gas),
                                           np.int32(1 if mc.ion_energy_frame == "com" else 0), self._mi_pref,
                                           np.int32(self._mi_ngrid), np.float64(self._mi_ecap), np.float64(dt_i)))
            k["mcc_ion"](self._grid1(io.alloc), (_BLOCK,), (
                io.vx, io.vy, io.vz, io.w, self._cnt, np.int32(self._mi_n), self._mi_kind, self._mi_tab_e,
                self._mi_tab_s, self._mi_len, np.int32(self._mi_w), np.float64(mc.n_gas), np.float64(mc.m_ion),
                np.float64(mc.mu), np.float64(mc.vth_gas), np.int32(1 if mc.ion_energy_frame == "com" else 0),
                self._prm, self._mcc_seed,
            ))
        del g

    def _launch_accumulate(self) -> None:
        k = self._k
        n = self.n_nodes
        use_cyc = self._cyc is not None
        k["accum_phi"](self._grid1(n), (_BLOCK,), (self._phi, self._acc_phi, self._cyc["phi"] if use_cyc else self._acc_phi,
                                                  np.int32(1 if use_cyc else 0), np.int64(n), self._prm, self._cyc_count))
        for name, sp in self.species.items():
            key = "n_e" if sp.index == 0 else "n_i"
            out2 = self._cyc[key] if use_cyc else self._acc_n[name]
            k["deposit"](self._grid1(sp.alloc), (_BLOCK,),
                         (sp.x, sp.y, sp.vx, sp.vy, sp.vz, sp.w, self._cnt, np.int32(sp.index), np.int32(0),
                          np.float64(1.0), self._acc_n[name], out2, np.int32(1 if use_cyc else 0), np.int64(n),
                          np.int32(1), self._prm, *self._grid_args, np.int32(self._px), np.int32(self._py),
                          *self._gx))
        el = self.species["electron"]
        out2 = self._cyc["ke"] if use_cyc else self._acc_ke
        k["deposit"](self._grid1(el.alloc), (_BLOCK,),
                     (el.x, el.y, el.vx, el.vy, el.vz, el.w, self._cnt, np.int32(0), np.int32(1),
                      np.float64(0.5 * ME), self._acc_ke, out2, np.int32(1 if use_cyc else 0), np.int64(n),
                      np.int32(1), self._prm, *self._grid_args, np.int32(self._px), np.int32(self._py),
                      *self._gx))
        if self._n_eedf:
            shm = self._n_eedf * (self._eedf_bins + 3) * 8
            k["eedf_hist"](self._grid1(el.alloc), (_BLOCK,),
                           (el.x, el.y, el.vx, el.vy, el.vz, el.w, self._cnt, self._eedf_reg, np.int32(self._n_eedf),
                            np.int32(self._eedf_bins), self._eedf_hist, self._eedf_sums, self._prm), shared_mem=shm)

    def _launch_step(self) -> None:
        """1 ステップ分のカーネル列を現在のストリームに積む (同期なし、CUDA Graph に記録可能)。"""
        k = self._k
        k["begin_step"]((1,), (1,), (self._prm, self._cnt, self._dsl, np.int32(8)))
        self._launch_field()
        el, io = self.species["electron"], self.species["ion"]
        self._launch_push(el)
        self._launch_push(io)
        self._launch_boundary(el)
        self._launch_boundary(io)
        self._launch_mcc()
        self._launch_accumulate()
        if self._has_diel:
            k["sum_into"]((64,), (_BLOCK,), (self._q_surf, np.int64(self.n_nodes), self._dsl[3:4]))
        k["history_row"]((1,), (1,), (self._hist, np.int32(HISTORY_ROWS), np.int32(len(_HIST_COLS)), self._prm,
                                      self._cnt, self._dsl, np.float64(0.5 * self._two_pi), np.float64(self._two_pi)))
        if self.circuit is not None:
            k["cap_step"]((1,), (1,), (self._cap_st, self._cap_hist, np.int32(HISTORY_ROWS), self._prm,
                                       np.int32(self._cap_m)))
        k["end_step"]((1,), (1,), (self._prm,))

    def _run_one_step(self) -> None:
        if self._use_graph:
            key = (tuple(sp.alloc for sp in self.species.values()), self._pcg_iters, id(self._acc_phi))
            if self._graph is None or self._graph_key != key:
                with self._stream:
                    self._stream.begin_capture()
                    try:
                        self._launch_step()
                    finally:
                        self._graph = self._stream.end_capture()
                self._graph_key = key
            self._graph.launch(self._stream)
        else:
            with self._stream:
                self._launch_step()

    def _kick_backward_half(self) -> None:
        """初期の半ステップ後退キック v(-dt/2) = v(0) − (q/m) E(x0) dt/2 (面内 2 成分)。"""
        cp = self.cp
        self._k["begin_step"]((1,), (1,), (self._prm, self._cnt, self._dsl, np.int32(8)))
        self._launch_field()
        g = self.grid
        for sp in self.species.values():
            if not sp.mobile:
                continue
            n = int(self._cnt[sp.index].get())
            if n == 0:
                continue
            dt_s = self.dt * (self._sub if sp.index == 1 else 1)
            if self.amr is not None:
                self._k["kick_half"](self._grid1(sp.alloc), (_BLOCK,),
                                     (sp.x, sp.y, sp.vx, sp.vy, sp.w, self._cnt, np.int32(sp.index), self._ex,
                                      self._ey, np.float64(sp.q / sp.m * 0.5 * dt_s), *self._gx))
                continue
            x, y = sp.x[:n], sp.y[:n]
            fx = (x - g.x0) * self._inv_dx
            fy = (y - g.y0) * self._inv_dy
            i = cp.clip(cp.floor(fx).astype(np.int64), 0, g.nx - 1)
            j = cp.clip(cp.floor(fy).astype(np.int64), 0, g.ny - 1)
            wx = cp.clip(fx - i, 0.0, 1.0)
            wy = cp.clip(fy - j, 0.0, 1.0)
            s = g.nx + 1
            kk = j * s + i
            exf, eyf = self._ex.reshape(-1), self._ey.reshape(-1)

            def interp(f):
                return (1 - wx) * (1 - wy) * f[kk] + wx * (1 - wy) * f[kk + 1] + (1 - wx) * wy * f[kk + s] + wx * wy * f[kk + s + 1]

            qm = sp.q / sp.m
            sp.vx[:n] -= qm * interp(exf) * 0.5 * dt_s
            sp.vy[:n] -= qm * interp(eyf) * 0.5 * dt_s

    # ======================================================================================
    # 阻止コンデンサ (自己バイアス、prompts/134・モジュール docstring)
    # ======================================================================================

    def _init_circuit(self) -> None:
        """電極の表 (外周の辺・導体 → 電極)、誘導電荷の重み W・容量行列、デバイスの回路の状態と履歴。"""
        cp = self.cp
        model = self.model
        self.circuit: BlockingCircuit | None = None
        self._cap_m = 0
        elecs = model_capacitor_electrodes(self.project, model)
        side_elec = np.full(4, -1, dtype=np.int32)
        cond_elec = np.full(max(len(model.conductors), 1), -1, dtype=np.int32)
        if elecs:
            m = len(elecs)
            if m > CAP_MAX:
                raise ValueError(f"阻止コンデンサの電極は {CAP_MAX} 個までです ({m} 個)")
            self._cap_m = m
            self._cap_groups = [groups for _, groups in elecs]
            group_elec = np.full(max(len(model.groups), 1), -1, dtype=np.int64)
            for j, groups in enumerate(self._cap_groups):
                group_elec[groups] = j
            # 境界のカーネルの st 1〜4 (左・右・下・上) と導体の番号 (model.conductors の順) → 電極
            for k, side in enumerate(("left", "right", "bottom", "top")):
                g = model.side_group.get(side)
                if g is not None:
                    side_elec[k] = group_elec[g]
            for k, g in enumerate(model.conductor_group):
                cond_elec[k] = group_elec[g]
            w, c_matrix = self._cap_weights()
            self.circuit = BlockingCircuit([spec for spec, _ in elecs], c_matrix, rf_period(self.project))
            self._cap_upload(w)
            # グループの表: m+1 個の位置 (この配列の中の絶対位置) とグループ番号
            flat: list[int] = []
            off: list[int] = []
            for groups in self._cap_groups:
                off.append(m + 1 + len(flat))
                flat.extend(groups)
            off.append(m + 1 + len(flat))
            self._cap_grp = cp.asarray(np.asarray(off + flat, dtype=np.int32))
            self._cap_st = cp.zeros(5 * m + 1)
            self._cap_part = cp.zeros(m * CAP_NB)
            self._cap_hist = cp.zeros((HISTORY_ROWS, 2 * m))
            self._cap_dq = self._cap_st[3 * m:4 * m]
        else:
            self._cap_st = cp.zeros(1)
            self._cap_dq = self._cap_st  # 書かれない (電極が無いので境界のカーネルは数えない)
        self._side_elec = cp.asarray(side_elec)
        self._cond_elec = cp.asarray(cond_elec)

    def _cap_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """誘導電荷の重み W (電極 × 全節点。Q_e0 = W·q、q は右辺の単位の節点の電荷) と容量行列 C。

        どちらも物理の電荷の単位 (軸対称は 2π 込み、平面は奥行き 1 m あたり)。
        一様格子: W_j = −2π ψ̃_j (ψ_j は電極 j のグループ 1 V・ほか 0 V・空間電荷 0 の解を GMG-PCG で解き、固定節点に
        グループの電位を入れたもの。周期の従属節点は 0)。C_jk は ψ_k の解での電極 j の電荷 (結合の電束、v2 の流体と同じ)。
        AMR: 電極の電荷の汎関数 Q_j = a_j·φ + b_j·V + c_j·q (amr.composite.electrode_charge_functionals) に
        φ = P_node A_c⁻¹ (P_nodeᵀ q + coup_c V) + C_node V を入れ、随伴 y_j = A_c⁻¹ P_nodeᵀ a_j から
        W_j = c_j + P_node y_j、C_jk = y_j·coup_c e_k + a_j·C_node e_k + b_j·e_k。
        """
        factor = self._two_pi
        m = self._cap_m
        if self.amr is None:
            op = self.op
            k_groups = op.coupling.shape[1]
            group_elec = np.full(k_groups, -1, dtype=np.int64)
            for j, groups in enumerate(self._cap_groups):
                group_elec[groups] = j
            coo = op.coupling.tocoo()
            ej = group_elec[coo.col]
            sel = ej >= 0
            slave = (op.mask == MASK_SLAVE).ravel()
            w = np.zeros((m, self.n_nodes))
            c = np.zeros((m, m))
            for k, groups in enumerate(self._cap_groups):
                v = np.zeros(k_groups)
                v[groups] = 1.0
                b = np.asarray(op.coupling @ v).reshape(op.grid.shape)
                x, info = self.solver.solve(self.cp.asarray(b), tol=CAP_PSI_TOL, max_iter=2000)
                if not info.converged:
                    self.warnings.append(
                        f"阻止コンデンサの ψ の Poisson (GMG-PCG) が収束しませんでした (相対残差 {info.relative_residual:.2e})"
                    )
                psi = fill_fixed(op, self.cp.asnumpy(x), v).ravel()
                w[k] = np.where(slave, 0.0, -factor * psi)
                flux = coo.data * (v[coo.col] - psi[coo.row])
                c[:, k] = factor * np.bincount(ej[sel], weights=flux[sel], minlength=m)
            return w, c
        from ..amr.composite import electrode_charge_functionals

        lay = self.amr
        n, n_c = lay.n_nodes, lay.n_c
        a, b, cfun = electrode_charge_functionals(lay.op, self._cap_groups)
        p_node = lay.PC[:, :n_c]
        c_node = lay.PC[:, n_c:]
        coup_c = lay.PTq[:, n:]
        units = []
        for groups in self._cap_groups:
            u = np.zeros(lay.n_groups)
            u[groups] = 1.0
            units.append(u)
        solve = _host_composite_solver(lay.op.A_c) if n_c else None
        w = np.zeros((m, n))
        c = np.zeros((m, m))
        for j in range(m):
            y = solve(np.asarray(p_node.T @ a[j]).ravel()) if n_c else np.zeros(0)
            w[j] = factor * (cfun[j] + np.asarray(p_node @ y).ravel())
            for k in range(m):
                c[j, k] = factor * (y @ np.asarray(coup_c @ units[k]).ravel()
                                    + a[j] @ np.asarray(c_node @ units[k]).ravel() + b[j] @ units[k])
        return w, c

    def _cap_upload(self, w: np.ndarray) -> None:
        """誘導電荷の重みと回路の定数 ((C + C_b)⁻¹・C・C_b・初期バイアス) をデバイスへ。"""
        cp = self.cp
        circ = self.circuit
        ainv = np.linalg.inv(circ.c + np.diag(circ.c_b))
        self._cap_w = cp.asarray(np.ascontiguousarray(w, dtype=np.float64))
        self._cap_par = cp.asarray(np.concatenate([ainv.ravel(), circ.c.ravel(), circ.c_b, circ.bias0]))

    def _launch_circuit(self) -> None:
        """堆積のあと: 誘導電荷 → 新しい電極の電位をグループ電位 vg へ (右辺を組む前、同期なし)。"""
        m = self._cap_m
        self._k["cap_induced"]((CAP_NB, m), (_BLOCK,), (self._cap_w, self._q_static, self._rho, self._q_surf,
                                                       np.int64(self.n_nodes), self._cap_part))
        self._k["cap_solve"]((1,), (1,), (self._cap_st, self._vg, self._cap_par, self._cap_grp, self._cap_part,
                                         np.int32(CAP_NB), np.int32(m)))

    def _flush_circuit(self, idx: np.ndarray, t: np.ndarray) -> None:
        """デバイスの回路の履歴 (V_e・電流) を周期の集計へ、状態 (Q_N・V_e・V_s) をホストの回路へ (同期済みで呼ぶ)。"""
        m = self._cap_m
        rows = self._cap_hist.get()[idx]
        for row, t0 in zip(rows, t):
            self.circuit.record(row[:m], row[m:], float(t0), self.dt)
        st = self._cap_st.get()
        self.circuit.set_state(st[:m], st[m:2 * m], st[2 * m:3 * m])

    # ======================================================================================
    # 動的再格子化 (prompts/123)
    # ======================================================================================

    def _self_force_warning(self) -> None:
        """AMR の粗細界面の自己力 (prompts/122) の目安が冷たい種の熱エネルギーに比べて大きければ警告。

        自己力による電位は 1 粒子あたり ~0.1·λ/(2πε0) (λ = マクロ粒子の線電荷密度 w·e。軸対称は
        リングなので w·e/(2πr))。導体壁の近くでマクロ粒子が受ける自分の鏡像力と同程度の大きさ。
        """
        if self.amr is None or self._w0 is None or (self.amr.hier.max_level == 0 and not self._regrid_every):
            return
        ip = self.pic.initial_plasma
        lam = self._w0 * QE
        if self.rz:
            d = self.model.domain
            r_mid = 0.5 * ((d.y0 + d.y1) if self.ridx == 1 else (d.x0 + d.x1))
            lam /= 2.0 * math.pi * max(r_mid, 1e-30)
        u_self = 0.1 * lam / (2.0 * math.pi * EPS0)
        cold = [(t, name) for t, name, mobile in ((ip.te_ev, "電子", True), (ip.ti_ev, "イオン", not ip.immobile_ions))
                if mobile and t > 0.0]
        for t, name in cold:
            if u_self > 0.2 * t:
                self.warnings.append(
                    f"粗細界面の自己力の目安 {u_self:.3g} V (1 粒子あたり) が{name}温度 {t:.3g} eV に比べて大きい: "
                    "界面付近の密度に偽の構造が出る恐れがあります。粒子数 (n_macro) を増やしてください"
                )
                break

    def _launch_tag_deposit(self) -> None:
        """再格子化のタグ用に電子の重みと運動エネルギーを節点へ積算する (グラフ外、同期なし)。

        容量の停止中は積算しない (gate 2)。その回は巻き戻しで _tag_count から引く (_rewind)。
        """
        el = self.species["electron"]
        n = self.n_nodes
        for out, mode, scale in ((self._tag_w, 0, 1.0), (self._tag_ke, 1, 0.5 * ME)):
            self._k["deposit"](self._grid1(el.alloc), (_BLOCK,),
                               (el.x, el.y, el.vx, el.vy, el.vz, el.w, self._cnt, np.int32(0), np.int32(mode),
                                np.float64(scale), out, out, np.int32(0), np.int64(n), np.int32(2), self._prm,
                                *self._grid_args, np.int32(self._px), np.int32(self._py), *self._gx))
        self._tag_count += 1
        self._tag_steps.append(self.step_count)

    def _regrid(self) -> bool:
        """区間平均の λ_D で AMR 階層を作り直し、場の状態を新しい格子へ移す (ここで同期)。

        粒子は座標のまま、誘電体の表面電荷は旧節点の電荷を新しい格子へ CIC で配り直し (総電荷保存)、
        前ステップ解 (warm start) は旧格子の電位を新しい節点で補間する。時間平均区間の前だけ呼ぶ
        (平均用の節点配列は 0 のまま作り直す)。戻り値: 格子が変わったか。
        """
        from ..amr import AmrHierarchy
        from ..amr.gpu_solver import AmgGpuSolver
        from ..amr.pic_layout import build_pic_layout, debye_kappa, debye_tags, deposit_points, sample_nodes

        def same(h1, h2) -> bool:
            return h1.max_level == h2.max_level and all(np.array_equal(a, b) for a, b in zip(h1.refined, h2.refined))

        cp = self.cp
        lay = self.amr
        spec = lay.hier.spec
        self._stream.synchronize()
        if self._tag_count == 0:
            return False
        kappa = debye_kappa(lay, self._tag_w.get(), self._tag_ke.get(), self._tag_count)
        with self._stream:
            self._tag_w.fill(0.0)
            self._tag_ke.fill(0.0)
        self._tag_count = 0
        self._tag_steps = []
        # 候補の階層で判定し直すことを繰り返し、新しい領域の中も一度で必要なレベルまで細分化する
        cand = lay.hier
        for _ in range(spec.max_level + 1):
            nxt = AmrHierarchy(self.model, self.grid, spec,
                               extra_tags=debye_tags(lay, kappa, cand, spec.pic_h_over_debye, keep=lay.hier))
            if same(nxt, cand):
                break
            cand = nxt
        new_hier = cand
        if same(new_hier, lay.hier):
            return False
        t0 = time.perf_counter()
        new = build_pic_layout(self.model, new_hier)
        phi_new = sample_nodes(lay, self._phi.get(), new.xy[:, 0], new.xy[:, 1])
        q_surf_old = self._q_surf.get()
        q_surf_new = (deposit_points(new, lay.xy[:, 0], lay.xy[:, 1], q_surf_old) if np.any(q_surf_old)
                      else np.zeros(new.n_nodes))
        with self._stream:
            self.solver = AmgGpuSolver(new.op.A_c, self.device, singular=new.op.singular)
            self.amr = new
            self.op = new.op
            self.n_nodes = new.n_nodes
            self._shape = (new.n_nodes,)
            self._setup_amr_fields(new)
            self._q_surf = cp.asarray(q_surf_new)
            self._rho = cp.zeros(self._shape)
            self._phi = cp.asarray(phi_new)
            self._ex = cp.zeros(self._shape)
            self._ey = cp.zeros(self._shape)
            if new.n_c:
                self._x[: new.n_c] = cp.asarray(phi_new[new.node_of_c])
            self._set_amr_display(new)
            self._see_delta = 1e-3 * new.h_min
            # 阻止コンデンサ: 新しい格子の誘導電荷の重みと容量行列 (Q_N はそのまま、prompts/134)
            if self.circuit is not None:
                w, c_matrix = self._cap_weights()
                self.circuit.set_capacitance(c_matrix)
                self._cap_upload(w)
            # 平均用の節点配列 (平均区間の前なので 0 のまま大きさだけ変える)
            self._acc_phi = cp.zeros(self._shape)
            self._acc_n = {"electron": cp.zeros(self._shape), "ion": cp.zeros(self._shape)}
            self._acc_ke = cp.zeros(self._shape)
            self._acc_ion = cp.zeros(self._shape)
            self._tag_w = cp.zeros(self._shape)
            self._tag_ke = cp.zeros(self._shape)
        if self._cycle_enabled and self._cycle_bins * self.n_nodes > CYCLE_MAX_VALUES:
            nb = max(1, CYCLE_MAX_VALUES // self.n_nodes)
            self.warnings.append(f"再格子化で節点数が増えたため位相ビン数を {self._cycle_bins} → {nb} に減らしました")
            self._cycle_bins = nb
            with self._stream:
                self._prm[P_NBINS] = float(nb)
        self._graph = None
        self.mesh_version += 1
        self.regrid_log.append({"step": self.step_count, "n_nodes": new.n_nodes, "levels": new.hier.n_levels,
                                "leaf_cells": new.hier.n_leaf_cells(), "setup_s": time.perf_counter() - t0})
        return True

    # ======================================================================================
    # 粒子の容量 (モジュール docstring の「粒子の容量」)
    # ======================================================================================

    def _capacity_check(self, compact: bool) -> bool:
        """カウンタを読む同期点 (compact なら先に吸収粒子 (w = 0) を順序保存で詰める)。

        予備まで使い切って粒子が失われていたらエラー。デバイスが停止していたら、止まっていたステップの分だけ
        ホストを巻き戻す (停止中の圧縮はカーネルが詰めずに写すだけにするので、粒子の並びは停止しなかった場合と
        同じ)。圧縮のとき・停止の要求があったときは、追加の速さから先回りして配列を広げ、停止を解く。
        戻り値: ステップを巻き戻したか (呼び出し側は巻き戻したステップから続ける)。
        """
        k = self._k
        with self._stream:
            if compact:
                for sp in self.species.values():
                    nb = (sp.alloc + 1023) // 1024
                    k["compact_scan"]((nb,), (1024,), (sp.w, self._cnt, np.int32(sp.index), self._scan_off,
                                                        self._scan_flag, self._scan_bsum, np.int64(sp.alloc)))
                    k["compact_scan_blocks"]((1,), (1,), (self._scan_bsum, np.int64(nb), self._cnt,
                                                          np.int32(sp.index)))
                    t = self._tmp
                    k["compact_scatter"](self._grid1(sp.alloc), (_BLOCK,),
                                         (self._scan_flag, np.int64(sp.alloc), self._scan_off, self._scan_bsum,
                                          sp.x, sp.y, sp.vx, sp.vy, sp.vz, sp.w,
                                          t["x"], t["y"], t["vx"], t["vy"], t["vz"], t["w"]))
                    for f in _Species.FIELDS:
                        self._k_copy(self._grid1(sp.alloc), (_BLOCK,), (sp.arrays[f], t[f], np.int64(sp.alloc)))
            cnt = self._cnt.get()   # ステップのストリームで読む (ここで同期)
        lost = int(cnt[C_OVERFLOW])
        if lost:
            reserve = min(sp.alloc - sp.cap for sp in self.species.values())
            raise RuntimeError(
                f"粒子配列の容量を超えました: 1 ステップの電離・二次電子の生成が予備の容量 ({reserve} 個) を越え、"
                f"{lost} 個が失われました。電離が急増しています — dt や粒子数 (n_macro) を見直してください"
            )
        halted = bool(cnt[C_HALT])
        dev_step = int(cnt[C_HALT_STEP]) if halted else self.step_count
        skipped = self.step_count - dev_step
        if halted:
            self._rewind(skipped)
        self._tag_steps = []
        # 追加の速さ [粒子/ステップ] (電離は電子とイオン、二次電子は電子)
        ev = (int(cnt[C_ION_EV]), int(cnt[C_SEE_EV]))
        steps = dev_step - self._chk_step
        if steps > 0:
            d_ion, d_see = ev[0] - self._chk_ev[0], ev[1] - self._chk_ev[1]
            decay = 0.5 ** (steps / CAP_RATE_HALFLIFE)
            for name, added in (("electron", d_ion + d_see), ("ion", d_ion)):
                self._cap_rate[name] = max(added / steps, self._cap_rate[name] * decay)
            self._chk_step, self._chk_ev = dev_step, ev
        requested = halted or bool(cnt[C_HALT_REQ])
        changed = requested
        if compact or requested:
            for sp in self.species.values():
                n = int(cnt[sp.index])
                ahead = CAP_AHEAD * COMPACT_EVERY * self._cap_rate[sp.name]
                if n > CAP_HIGH * sp.cap or n + ahead > sp.cap:
                    self._grow(sp, int(max(CAP_GROW * n, n + 2.0 * ahead)) + CAP_GROW_MIN)
                    changed = True
        if requested:
            with self._stream:
                self._cnt[C_HALT_REQ:C_HALT + 1] = 0
            self.capacity_log.append({"step": dev_step, "halted_steps": skipped,
                                      "cap": {name: sp.cap for name, sp in self.species.items()}})
        if changed:
            self._stream.synchronize()   # 呼び出し側は戻った後に既定のストリームで読む
        self._checked_step = self.step_count
        return halted

    def _sync_point(self) -> bool:
        """ホストが結果を読む前の同期と容量の検査 (同じステップの 2 回目は同期だけ)。戻り値: 巻き戻したか。"""
        if self._checked_step == self.step_count:
            self._stream.synchronize()
            return False
        return self._capacity_check(compact=False)

    def _rewind(self, skipped: int) -> None:
        """停止していた skipped ステップ (デバイスでは状態もステップ番号も変わっていない) の分だけ、ホストの
        ステップ数・時刻・履歴の未転送の行数・平均の回数・再格子化のタグの回数を戻す。"""
        hi = self.step_count
        lo = hi - skipped
        self.step_count = lo
        self.t -= skipped * self.dt
        self._hist_pending -= skipped
        if self._accum_start is not None:
            acc = max(0, hi - max(lo + 1, self._accum_start) + 1)
            self._accum_count -= acc
            if self._n_eedf:
                self._eedf_samples -= acc
        self._tag_count -= sum(1 for s in self._tag_steps if s > lo)

    # ======================================================================================
    # 時間平均・位相分解・コレクタ・EEDF
    # ======================================================================================

    def _reset_accumulators(self) -> None:
        cp = self.cp
        shp = self._shape
        self._accum_start: int | None = None
        self._accum_count = 0
        with self._stream:
            self._acc_phi = cp.zeros(shp)
            self._acc_n = {"electron": cp.zeros(shp), "ion": cp.zeros(shp)}
            self._acc_ke = cp.zeros(shp)
            self._acc_ion = cp.zeros(shp)
            self._cyc_count = cp.zeros(max(self._cycle_bins, 1), dtype=np.uint64)
            self._rec_n.fill(0)
            self._coll_w.fill(0.0)
            self._eedf_hist.fill(0.0)
            self._eedf_sums.fill(0.0)
        self._cyc = None
        self._cyc_ion = None
        self._cycle_particles = None
        self._snap_t_start = math.inf
        self._eedf_samples = 0
        self._graph = None

    def enable_density_accum(self, start_step: int) -> None:
        cp = self.cp
        self._accum_start = int(start_step)
        self._accum_count = 0
        with self._stream:
            if self._cycle_enabled:
                b = self._cycle_bins
                shp = self._shape
                self._cyc = {key: cp.zeros((b, *shp)) for key in ("phi", "n_e", "n_i", "ke")}
                self._cyc_ion = cp.zeros((b, *shp))
                self._cyc_count = cp.zeros(b, dtype=np.uint64)
            self._prm[P_ACC_START] = float(start_step)
        self._graph = None

    def _phase_bin(self, t: float) -> int:
        ph = (t / self._cycle_period) % 1.0
        return min(int(ph * self._cycle_bins), self._cycle_bins - 1)

    def _prepare_eedf_auto(self) -> None:
        """平均区間の最初のステップで、e_max 自動の EEDF 領域を「領域内の最大エネルギー×1.2」に決める。"""
        if not any(self._eedf_auto):
            return
        cp = self.cp
        el = self.species["electron"]
        with self._stream:
            n = min(int(self._cnt[0].get()), el.alloc)
            x, y = el.x[:n], el.y[:n]
            e = 0.5 * ME * (el.vx[:n] ** 2 + el.vy[:n] ** 2 + el.vz[:n] ** 2) / QE
            alive = el.w[:n] != 0.0
            reg = self._eedf_reg.get()
            for r, auto in enumerate(self._eedf_auto):
                if not auto:
                    continue
                x0, x1, y0, y1 = reg[5 * r: 5 * r + 4]
                sel = alive & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
                em = float(cp.max(cp.where(sel, e, 0.0))) if n else 0.0
                reg[5 * r + 4] = em * 1.2 if em > 0.0 else 30.0
            self._eedf_reg[...] = cp.asarray(reg)

    def _node_density(self, acc, count: int) -> np.ndarray:
        return self._disp((acc * self._inv_vol_phys / max(count, 1)).get().ravel())

    def averaged_fields(self) -> dict | None:
        if self._accum_start is None or self._accum_count == 0:
            return None
        cnt = self._accum_count
        phi = self._disp((self._acc_phi / cnt).get().ravel())
        grad = triangle_gradients(self._dmesh.nodes, self._dmesh.triangles, phi)
        e_abs = np.hypot(grad[:, 0], grad[:, 1])
        n_e = self._node_density(self._acc_n["electron"], cnt)
        n_i = self._node_density(self._acc_n["ion"], cnt)
        w_e = self._disp(self._acc_n["electron"].get().ravel())
        ke = self._disp(self._acc_ke.get().ravel())
        te = np.where(w_e > 0.0, (2.0 / 3.0) * ke / np.where(w_e > 0, w_e, 1.0) / QE, 0.0)
        ion_rate = self._disp((self._acc_ion * self._inv_vol_phys).get().ravel()) / (cnt * self.dt)
        return {"phi": phi, "e_abs": e_abs, "n_e": n_e, "n_i": n_i, "te_ev": te, "ion_rate": ion_rate,
                "avg_steps": cnt}

    def cycle_data(self) -> dict | None:
        if self._cyc is None:
            return None
        cp = self.cp
        counts = self._cyc_count.get().astype(np.float64)
        if counts.sum() == 0:
            return None
        b = self._cycle_bins
        cnt_d = cp.asarray(np.maximum(counts, 1.0)).reshape((b,) + (1,) * len(self._shape))
        inv_v = self._inv_vol_phys[None]
        phi = self._disp((self._cyc["phi"] / cnt_d).get().reshape(b, -1))
        n_e = self._disp((self._cyc["n_e"] * inv_v / cnt_d).get().reshape(b, -1))
        n_i = self._disp((self._cyc["n_i"] * inv_v / cnt_d).get().reshape(b, -1))
        ne_w = self._disp(self._cyc["n_e"].get().reshape(b, -1))
        ke = self._disp(self._cyc["ke"].get().reshape(b, -1))
        te = np.where(ne_w > 0.0, (2.0 / 3.0) * ke / np.where(ne_w > 0, ne_w, 1.0) / QE, 0.0)
        ion_rate = self._disp((self._cyc_ion * inv_v / cnt_d).get().reshape(b, -1)) / self.dt
        grad = triangle_gradients(self._dmesh.nodes, self._dmesh.triangles, phi)
        e_abs = np.hypot(grad[..., 0], grad[..., 1])
        particles = {}
        empty = np.zeros((0, 2))
        for name in ("electron", "ion"):
            snaps = self._cycle_particles[name] if self._cycle_particles is not None else [None] * b
            particles[name] = [s if s is not None else empty for s in snaps]
        return {"bins": b, "period_s": self._cycle_period, "phi": phi, "n_e": n_e, "n_i": n_i,
                "e_abs": e_abs, "te_ev": te, "ion_rate": ion_rate, "particles": particles}

    def _collector_data(self) -> list[dict] | None:
        if not self._n_coll:
            return None
        rec_n = self._rec_n.get().astype(np.int64)
        coll_w = self._coll_w.get()
        e = self._rec_e.get().reshape(-1, COLLECTOR_MAX_SAMPLES)
        a = self._rec_a.get().reshape(-1, COLLECTOR_MAX_SAMPLES)
        w = self._rec_w.get().reshape(-1, COLLECTOR_MAX_SAMPLES)
        out = []
        for c in range(self._n_coll):
            count = int(rec_n[c])
            s = min(count, COLLECTOR_MAX_SAMPLES)
            out.append({"count": count, "total_weight": float(coll_w[c]), "energies_ev": e[c, :s].copy(),
                        "angles_deg": a[c, :s].copy(), "weights": w[c, :s].copy(),
                        "truncated": count > COLLECTOR_MAX_SAMPLES})
        return out

    def _eedf_data(self) -> list[dict] | None:
        if not self._n_eedf:
            return None
        hist = self._eedf_hist.get().reshape(self._n_eedf, self._eedf_bins)
        sums = self._eedf_sums.get().reshape(self._n_eedf, 3)
        reg = self._eedf_reg.get().reshape(self._n_eedf, 5)
        out = []
        for r, meta in enumerate(self._eedf_meta):
            bins = meta["bins"]
            e_max = float(reg[r, 4])
            edges = np.linspace(0.0, e_max, bins + 1)
            centers = 0.5 * (edges[1:] + edges[:-1])
            de = edges[1] - edges[0]
            h = hist[r, :bins]
            tot = float(h.sum())
            f = h / (tot * de) if tot > 0.0 else np.zeros(bins)
            sw, swe, ov = sums[r]
            mean_e = swe / sw if sw > 0.0 else 0.0
            out.append({"label": meta["label"], "e_centers": centers, "f": f, "mean_energy_ev": mean_e,
                        "t_eff_ev": 2.0 / 3.0 * mean_e, "total_weight": float(sw),
                        "overflow_frac": float(ov / sw) if sw > 0.0 else 0.0, "n_samples": self._eedf_samples})
        return out

    def _snapshot_particles(self, t_step: float) -> None:
        b = self._phase_bin(t_step)
        for name in ("electron", "ion"):
            if self._cycle_particles[name][b] is not None:
                continue
            self._cycle_particles[name][b] = self._subsample_positions(self.species[name], 1000)

    def _subsample_positions(self, sp: _Species, limit: int) -> np.ndarray:
        n = min(int(self._cnt[sp.index].get()), sp.alloc)
        if n == 0:
            return np.zeros((0, 2))
        stride = max(1, int(math.ceil(n / limit)))
        alive = sp.w[:n:stride] != 0.0
        return np.stack([sp.x[:n:stride][alive].get(), sp.y[:n:stride][alive].get()], axis=1)

    # ======================================================================================
    # 粒子状態の読み書き (テスト・検証スクリプト用。実装非依存の API)
    # ======================================================================================

    def get_particles(self, name: str) -> dict[str, np.ndarray]:
        """種 name の生存粒子をホスト配列で返す (x, y, vx, vy, vz, w)。v は半ステップ時刻の値。"""
        self._stream.synchronize()
        sp = self.species[name]
        n = min(int(self._cnt[sp.index].get()), sp.alloc)
        w = sp.w[:n].get()
        alive = w != 0.0
        return {f: (w if f == "w" else sp.arrays[f][:n].get())[alive] for f in _Species.FIELDS}

    def set_particles(self, name: str, **arrays) -> None:
        """種 name の粒子状態を置き換える。与えなかった成分は現在値のまま (粒子数を変えるなら全成分)。"""
        cur = self.get_particles(name)
        n = None
        for key, v in arrays.items():
            if key not in cur:
                raise KeyError(key)
            n = len(v) if n is None else n
            if len(v) != n:
                raise ValueError("set_particles: 配列の長さが揃っていません")
        if n is not None and n != len(cur["x"]):
            missing = [key for key in cur if key not in arrays]
            if missing:
                raise ValueError(f"粒子数を変える場合は全成分を与えてください (不足: {missing})")
        merged = {key: np.asarray(arrays.get(key, cur[key]), dtype=np.float64) for key in cur}
        self._write_particles(name, merged)

    # ======================================================================================
    # 1 ステップ・フレーム・実行
    # ======================================================================================

    def _flush_history(self) -> None:
        """GPU のリングバッファに溜まった history 行をホストへ移す (ここで同期)。"""
        # 計算は非ブロッキングのストリームに積んでいるので、読む前に完了を待つ (待たないと
        # 直近のステップの行がまだ書かれておらず 0 や前周の値を読んでしまう)。容量の停止で止まっていた
        # ステップの行は読まない (_sync_point が巻き戻す)
        self._sync_point()
        m = self._hist_pending
        if m == 0:
            return
        rows = self._hist.get()
        start = (self.step_count - m) % HISTORY_ROWS
        idx = (start + np.arange(m)) % HISTORY_ROWS
        block = rows[idx]
        h = self.history
        for c, key in enumerate(_HIST_COLS):
            col = block[:, c]
            if key in ("n_e", "n_i", "wall_e", "wall_i", "coll_e", "ion_events", "see_events"):
                h[key].extend(int(v) for v in col)
            else:
                h[key].extend(float(v) for v in col)
        h["fn_i"].extend([0.0] * m)
        h["fn_events"].extend([0] * m)
        h["merged"].extend([0] * m)
        if self.circuit is not None:
            self._flush_circuit(idx, block[:, 0])
        self._hist_pending = 0

    def _check_solver(self) -> None:
        """PCG の残差を監視して反復数を自動調整する (ここで同期)。"""
        if self.solver.direct:
            return
        self._stream.synchronize()
        rn, bn = self.solver.monitor()
        rel = rn / bn if bn > 0.0 else 0.0
        if rel > PCG_TOL and self._pcg_iters < PCG_ITERS_MAX:
            self._pcg_iters = min(PCG_ITERS_MAX, self._pcg_iters + 2)
        elif rel < 1e-3 * PCG_TOL and self._pcg_iters > 1:
            self._pcg_iters -= 1

    def step(self):
        """1 ステップ進める (テスト・デバッグ用の同期 API)。戻り値は電位 (デバイス配列)。

        毎ステップ容量を検査するので、軟らかい容量を越えたらその場で広げ、ステップは止まらない。
        """
        self._run_one_step()
        self.step_count += 1
        self.t += self.dt
        self._hist_pending += 1
        self._sync_point()
        if self._hist_pending >= HISTORY_ROWS or self.circuit is not None:
            self._flush_history()
        return self._phi

    def _make_frame(self) -> dict:
        cp = self.cp
        self._flush_history()
        particles = {name: self._subsample_positions(sp, MAX_FRAME_PARTICLES).tolist()
                     for name, sp in self.species.items()}
        diag = {key: v[-1] for key, v in self.history.items() if v}
        dens = {}
        for name, sp in self.species.items():
            cells = cp.zeros(self._n_cells)
            self._k["deposit_cell"](self._grid1(sp.alloc), (_BLOCK,),
                                    (sp.x, sp.y, sp.w, self._cnt, np.int32(sp.index), cells, *self._grid_args,
                                     *self._gx))
            c = cells.get()
            vol = self._cell_vol_gas.ravel()
            dens[name] = np.where(vol > 0, c / np.where(vol > 0, vol, 1.0), 0.0)[self._dmesh.tri_cell]
        frame = {
            "type": "frame",
            "step": self.step_count,
            "t": self.t,
            "phi": self._disp(self._phi.get().ravel()).tolist(),
            "n_e": dens["electron"].tolist(),
            "n_i": dens["ion"].tolist(),
            "particles": particles,
            "diag": diag,
            # 阻止コンデンサの電極の今の電位・直近の周期の自己バイアス (prompts/134)。無ければ None
            "circuit": None if self.circuit is None else self.circuit.frame(),
        }
        # 再格子化で表示用メッシュが変わったら、その後の最初のフレームに新しい格子を添える
        # (server は古いフレームを捨てるとき、この mesh を新しいフレームへ引き継ぐ)
        if self.mesh_version != self._frame_mesh_version:
            frame["mesh"] = self.mesh_payload()
            frame["mesh_version"] = self.mesh_version
            self._frame_mesh_version = self.mesh_version
        return frame

    def mesh_payload(self) -> dict:
        """表示用メッシュ (started・再格子化後の frame/done に載せる形)。"""
        return {"nodes": self.mesh.nodes.tolist(), "triangles": self.mesh.triangles.tolist()}

    def run_batch(self, callback=None, should_stop=None, store_frames: bool = True):
        """n_steps 回実行して (診断履歴, フレーム列) を返す (v1 PicSimulation.run_batch と同じ契約)。

        ホストが同期する所では必ず先に容量を検査し (_sync_point・_capacity_check)、デバイスが容量の停止で
        止まっていたら、巻き戻したステップから続ける (モジュール docstring の「粒子の容量」)。
        """
        t_run = time.perf_counter()
        if self._accum_start is None:
            avg = self.pic.avg_steps if self.pic.avg_steps is not None else max(1, self.pic.n_steps // 4)
            avg = min(avg, self.pic.n_steps)
            self.enable_density_accum(self.step_count + self.pic.n_steps - avg + 1)
        if self._cycle_enabled:
            self._cycle_particles = {"electron": [None] * self._cycle_bins, "ion": [None] * self._cycle_bins}
            self._snap_t_start = self.t + self.pic.n_steps * self.dt - self._cycle_period
        frames: list[dict] = []
        end = self.step_count + self.pic.n_steps
        t_frames = 0.0
        stopped = False
        while True:
            while self.step_count < end:
                if should_stop is not None and should_stop():
                    stopped = True
                    break
                if self.step_count + 1 == self._accum_start:
                    if self._sync_point():
                        continue
                    self._prepare_eedf_auto()
                self._run_one_step()
                self.step_count += 1
                self.t += self.dt
                self._hist_pending += 1
                if self._accum_start is not None and self.step_count >= self._accum_start:
                    self._accum_count += 1
                    if self._n_eedf:
                        self._eedf_samples += 1
                last = self.step_count == end
                # 動的再格子化 (prompts/123): 時間平均区間の前だけ。タグ用の電子の密度・温度を
                # TAG_EVERY ステップごとに積算し、regrid_every ステップごとに階層を作り直す
                if (self._regrid_every and self._accum_start is not None
                        and self.step_count + 1 < self._accum_start and not last):
                    if self.step_count % TAG_EVERY == 0:
                        with self._stream:
                            self._launch_tag_deposit()
                    if self.step_count % self._regrid_every == 0:
                        if self._sync_point():
                            continue
                        self._regrid()
                if self.step_count % COMPACT_EVERY == 0 and self._capacity_check(compact=True):
                    continue
                if self.step_count % MONITOR_EVERY == 0:
                    if self._sync_point():
                        continue
                    self._check_solver()
                if self._hist_pending >= HISTORY_ROWS:
                    if self._sync_point():
                        continue
                    self._flush_history()
                if self._cycle_enabled and self.t - self.dt >= self._snap_t_start - 1e-30:
                    if self._sync_point():
                        continue
                    self._snapshot_particles(self.t - self.dt)
                do_frame = (self.step_count % self.pic.frame_every == 0 or last) and (callback is not None or store_frames)
                if do_frame:
                    now = time.perf_counter()
                    if now - self._last_frame_wall >= MIN_FRAME_INTERVAL_S or last:
                        if self._sync_point():
                            continue
                        self._last_frame_wall = now
                        self._check_finite()
                        frame = self._make_frame()
                        if store_frames:
                            frames.append(frame)
                        if callback is not None:
                            callback(frame)
                        t_frames += time.perf_counter() - now
            # 最後のステップの後にも検査する (止まっていたら巻き戻して残りを実行)
            if not self._sync_point() or stopped:
                break
        self._flush_history()
        self._check_finite()
        self.fields = self.averaged_fields()
        self.cycle = self.cycle_data()
        self.collector_results = self._collector_data()
        self.collector_result = self.collector_results[0] if self.collector_results else None
        self.eedf_results = self._eedf_data()
        # GPU では位相ごとの時間を測ると同期で遅くなるため、実行時間を solve 以外の内訳に
        # 按分せず "gather_push" (= GPU の 1 ステップ全体) と "frame" に分けて報告する
        total = time.perf_counter() - t_run
        self.timing["frame"] += t_frames
        self.timing["gather_push"] += total - t_frames
        return self.history, frames

    def _check_finite(self) -> None:
        h = self.history
        if not h["phi_min"]:
            return
        if not (math.isfinite(h["phi_min"][-1]) and math.isfinite(h["phi_max"][-1]) and math.isfinite(h["ke_e"][-1])):
            raise ValueError(
                f"数値発散を検出しました (step {self.step_count}: 電位またはエネルギーが非有限値)。"
                "dt を小さくする、初期密度を下げる、メッシュを細かくする等を検討してください"
            )

    def prepare_continue(self, n_steps: int, frame_every=None, avg_steps=None, phase_bins=None) -> None:
        """完了/停止後の状態から追加実行の準備 (v1 と同じ契約。粒子・表面電荷・時刻は維持)。"""
        self.pic.n_steps = int(n_steps)
        if frame_every is not None:
            self.pic.frame_every = int(frame_every)
        if avg_steps is not None:
            self.pic.avg_steps = int(avg_steps)
        if phase_bins is not None:
            self.pic.phase_bins = int(phase_bins)
            self._cycle_bins = int(phase_bins)
            self._cycle_enabled = self._cycle_freq is not None and self._cycle_bins > 0
            self._cycle_period = 1.0 / self._cycle_freq if self._cycle_enabled else 0.0
            with self._stream:
                self._prm[P_PERIOD] = self._cycle_period if self._cycle_enabled else 0.0
                self._prm[P_NBINS] = float(max(self._cycle_bins, 1))
        self._flush_history()
        self.history = {key: [] for key in self.history}
        if self.circuit is not None:
            self.circuit.reset_history()  # 回路の状態 (コンデンサの電荷) はデバイスにあり、引き継ぐ
        self.timing = {key: 0.0 for key in self.timing}
        self._reset_accumulators()
        with self._stream:
            self._prm[P_ACC_START] = 1e300
        self.fields = None
        self.cycle = None
        self.collector_results = None
        self.collector_result = None
        self.eedf_results = None


def _host_composite_solver(a_c):
    """合成格子の A_c の求解 (阻止コンデンサの随伴の重み用。中小規模は疎行列 LU、大きければ AMG-CG)。"""
    if a_c.shape[0] <= CAP_HOST_DIRECT_MAX:
        import scipy.sparse.linalg as spla

        lu = spla.splu(a_c.tocsc(), permc_spec="MMD_AT_PLUS_A")
        return lu.solve
    import pyamg

    ml = pyamg.smoothed_aggregation_solver(a_c, symmetry="symmetric", max_coarse=500)
    return lambda rhs: ml.solve(rhs, tol=1e-12, accel="cg", maxiter=500)
