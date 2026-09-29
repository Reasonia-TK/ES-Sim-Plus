"""v2 GPU DSMC — 直交格子 + 埋め込み固体の上の定常ガス流れ (prompts/124)。

v1 ``dsmc.DsmcSimulation`` と**同じ外部インターフェース** (run / prepare_continue / mesh / dt / x /
step_count / timing / DsmcResult) を持ち、server の /dsmc・/ws/dsmc からそのまま駆動できる
(mesh.mode="cartesian" のとき)。結果は v1 UI が描ける三角形メッシュ (セルを 2 分割した表示用
メッシュ) の要素ごとの値で返すので、v1 PIC の MCC 背景ガス場 (use_dsmc_gas) にもそのまま渡せる。

## v1 との違い (物理は同じ: VHS 分子・NTC 衝突・拡散/鏡面反射・圧力リザーバ・真空・流量指定)

- セル = 直交格子のセル。粒子の所属セルは O(1) (v1 は三角形メッシュの隣接 walk で、実機で
  実行時間の 97% を占めていた、prompts/88)。
- 固体 (導体・誘電体) は厳密形状 (円・多角形) との交点で拡散反射する (v1 は導体輪郭を
  メッシュ境界の折れ線で、誘電体は「移動を取り消して等方 Maxwell で再抽選」で近似していた)。
- 反射・再放出の後は残り時間を最大 6 レグまで飛行させる (v1 と同じ)。
- NTC の候補対はセルごとの 1 スレッドが順に処理するので、同じ分子が 2 度選ばれても更新後の
  速度を使う (v1 の一括ベクトル化で必要だった重複除去が要らない)。
- 乱数は Philox (カウンタ方式)。統計的には等価だがビット一致はしない。
- 平滑化 (smoothing_passes) は 4 近傍の体積重み対称拡散 (θ = 0.2、総量保存・正値)。
- threads 設定は無視 (GPU)。mesh_scale は格子幅 × mesh_scale の DSMC 専用格子。

## AMR (mesh.amr、prompts/127)

mesh.amr で細分化が起きる (境界近傍・指定矩形) なら、衝突・サンプリングのセルを AMR 階層の葉セルに
する (kernels/dsmc.cu の -DES_AMR 変種が移動の終点の葉セルをブロック表で求める)。時間刻みの既定は
最細レベルの格子幅で決める。結果は葉セルを 2 三角形に分けた表示用メッシュ (/mesh と同じ)。
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..amr.cell_layout import AmrCellLayout, build_cell_layout, cell_of_points, mfp_tags, smooth_cells
from ..amr.hierarchy import AmrHierarchy, AmrSpec
from ..device import Device, get_device
from ..dsmc import AMU, KB, MAX_CALLBACK_PARTICLES, SCCM_TO_PER_S, DsmcResult
from ..eb.grid import make_grid
from ..geom.model import GeometryModel
from ..gpic.geometry import DisplayMesh, ParticleGeometry, cell_gas_volume
from ..schema import DsmcSettings, Project

_B_WALL, _B_SYM, _B_RES, _B_VAC = 0, 1, 2, 3
_BLOCK = 256
_SIDES = ("left", "right", "bottom", "top")
_SIDE_EDGE = {"bottom": 0, "right": 1, "top": 2, "left": 3}   # 矩形 domain の外周エッジ番号
#: 進捗コールバック・walk 診断の間隔 [ステップ] (v1 と同じ)
PROGRESS_EVERY = 100
#: 動的再格子化 (prompts/127) のためにセルの密度・温度を積算する間隔 [ステップ]
TAG_EVERY = 4


def _grid1(n: int) -> tuple[int]:
    return (max(1, (int(n) + _BLOCK - 1) // _BLOCK),)


class GpuDsmcSimulation:
    """v2 GPU DSMC。run() で n_steps 進めて DsmcResult を返す (v1 と同じ契約)。"""

    def __init__(self, project: Project, device: Device | str | None = None):
        if project.dsmc is None:
            raise ValueError("project.dsmc が指定されていません")
        dev = device if isinstance(device, Device) else get_device(device or "cuda")
        if not dev.is_gpu:
            raise RuntimeError(
                "v2 DSMC (mesh.mode='cartesian') は GPU (CUDA) 専用です。"
                "CPU で実行する場合はメッシュを unstructured/structured にしてください (v1 DSMC)"
            )
        import cupy as cp

        from ..device.cuda import load_module

        self.cp = cp
        self.device = dev
        self.project = project
        self.s: DsmcSettings = project.dsmc
        s = self.s
        gas = s.gas
        self.m = gas.mass_amu * AMU
        self.mu = 0.5 * self.m
        self._nthreads = 1

        self.model = model = GeometryModel(project)
        self.ridx = model.radial_axis()
        self.rz = self.ridx is not None
        if model.periodic_x or model.periodic_y:
            raise ValueError("v2 DSMC は周期境界に未対応です")
        self.grid = grid = make_grid(model.domain, float(project.mesh.size) * float(s.mesh_scale))
        self.pgeo = pg = ParticleGeometry(model, grid)
        # AMR (prompts/127): 細分化が起きる (または平均自由行程での動的再格子化が有効) なら葉セルを
        # 計算セルにする
        self.lay: AmrCellLayout | None = None
        self._regrid_every = 0
        self.regrid_log: list[dict] = []
        amr = project.mesh.amr
        if amr is not None and (amr.max_level > 0 or amr.regions):
            spec = AmrSpec.from_settings(amr)
            hier = AmrHierarchy(model, grid, spec)
            dynamic = spec.dsmc_regrid_every > 0 and spec.max_level > 0
            if hier.max_level > 0 or dynamic:
                self.lay = build_cell_layout(model, hier)
                self._regrid_every = spec.dsmc_regrid_every if dynamic else 0
        if self.lay is not None:
            lay = self.lay
            self.n_cells = lay.n_cells
            vol = lay.cell_vol_gas
            dm = self._display_of(lay)
            # 動的再格子化でも時間刻みを変えなくてよいよう、到達しうる最細レベルの格子幅で決める
            top = max(lay.hier.max_level, lay.hier.spec.max_level if self._regrid_every else 0)
            h_cell = min(grid.dx, grid.dy) / (1 << top)
        else:
            self.n_cells = grid.nx * grid.ny
            vol = cell_gas_volume(model, grid, pg.cell_state).ravel()
            dm = pg.display_mesh()
            h_cell = min(grid.dx, grid.dy)
        self.cell_vol = vol
        gas_volume = float(vol.sum())
        if gas_volume <= 0.0:
            raise ValueError("気体の領域がありません")

        # ---- 表示用メッシュ (v1 UI 互換、要素 = セルの 2 分割) ------------------------------
        self._set_display(dm)

        # ---- 境界 (外周の区間表・流入口) ------------------------------------------------
        self._build_boundaries()

        # ---- マクロ重み・時間刻み (v1 と同じ規約) -----------------------------------------
        n_init = s.init_pressure_pa / (KB * s.init_temperature_k)
        self.w = n_init * gas_volume / s.n_particles
        t_hot = max([s.init_temperature_k, s.wall_temperature_k] + [bc.temperature_k for bc in s.boundaries])
        v_mp = math.sqrt(2.0 * KB * t_hot / self.m)
        self.dt = float(s.dt) if s.dt is not None else 0.25 * h_cell / v_mp

        # ---- VHS -----------------------------------------------------------------------
        self._sig_coef = (math.pi * gas.d_ref_m**2 / math.gamma(2.5 - gas.omega)
                          * (2.0 * KB * gas.t_ref_k / self.mu) ** (gas.omega - 0.5))
        self._sig_pow = 1.0 - 2.0 * gas.omega
        cr0 = 2.0 * v_mp
        self._sigcr0 = self._sig_coef * cr0 ** self._sig_pow * cr0
        self._sigcr_max = cp.full(self.n_cells, self._sigcr0)
        self._coll_frac = cp.zeros(self.n_cells)

        # ---- GPU の表・カーネル -----------------------------------------------------------
        mod = load_module("dsmc", ("ES_AMR",) if self.lay is not None else ())
        names = ("dsmc_move", "dsmc_ncand", "dsmc_pairs", "dsmc_sample", "dsmc_inject")
        self._k = {name: mod.get_function(name) for name in names + (("dsmc_relocate",) if self.lay else ())}
        self._set_amr_args()
        self._ncand = cp.zeros(self.n_cells, dtype=np.int64)
        self._cand_start = cp.zeros(self.n_cells + 1, dtype=np.int64)
        solids_type, solids_off, pxy, circ = [], [0], [], []
        for sol in pg.solids:
            if hasattr(sol.shape, "vertices"):
                solids_type.append(0)
                pxy.extend(sol.shape.vertices.ravel().tolist())
                circ.extend([0.0, 0.0, 0.0])
            else:
                solids_type.append(1)
                circ.extend([sol.shape.cx, sol.shape.cy, sol.shape.r])
            solids_off.append(len(pxy) // 2)
        i32 = lambda a: cp.asarray(np.asarray(a if len(a) else [0], dtype=np.int32))  # noqa: E731
        f64 = lambda a: cp.asarray(np.asarray(a if len(a) else [0.0], dtype=np.float64))  # noqa: E731
        self._n_solid = len(pg.solids)
        self._s_type, self._s_off, self._s_pxy, self._s_circ = i32(solids_type), i32(solids_off), f64(pxy), f64(circ)
        self._cell_state = cp.asarray(pg.cell_state)
        self._side_off = cp.asarray(np.asarray(self._side_off_h, dtype=np.int32))
        self._iv = f64(self._iv_h)
        self._vol_d = cp.asarray(vol)
        self._delta = 1e-3 * h_cell
        self._seed = np.uint64(int(s.seed) & 0xFFFFFFFFFFFFFFFF)
        self._cnt = cp.zeros(4, dtype=np.uint64)   # 0: 吸収数 (累計), 1: 衝突数 (累計)

        # ---- 粒子 (SoA、容量固定) と初期充填 (一様 Maxwell) -------------------------------
        n0 = int(round(n_init * gas_volume / self.w))
        self.n = 0
        self._alloc(max(2 * n0, n0 + 65536))
        self._load_initial(n0)

        # ---- サンプリング -----------------------------------------------------------------
        self._acc_cnt = cp.zeros(self.n_cells)
        self._acc_v = cp.zeros(3 * self.n_cells)
        self._acc_v2 = cp.zeros(self.n_cells)
        self._samples = 0
        self._sampling = False
        self.inflow = 0.0
        self.outflow = 0.0
        self.step_count = 0
        self.timing: dict[str, float] = {"inject": 0.0, "move": 0.0, "collide": 0.0, "sample": 0.0, "other": 0.0}
        self.timing.update({"h_mean_m": h_cell, "h_min_m": h_cell, "walk_cells_est": 0.0})
        self._walk_diag_sum = 0.0
        self._walk_diag_n = 0
        self._start = cp.zeros(self.n_cells + 1, dtype=np.int32)
        self._count = cp.zeros(self.n_cells + 1, dtype=np.int32)
        self._alloc_tags()
        self._sort()

    # ---- 表示・AMR の表 -------------------------------------------------------------------

    @staticmethod
    def _display_of(lay: AmrCellLayout) -> DisplayMesh:
        return DisplayMesh(nodes=lay.disp_nodes, triangles=lay.disp_tris, tri_region=lay.disp_tri_region,
                           tri_cell=lay.tri_cell)

    def _set_display(self, dm: DisplayMesh) -> None:
        self.mesh = dm
        self.tris = dm.triangles
        self._tri_cell = dm.tri_cell
        # 要素の重み (v1 テストの面積重み平均と同じ使い方ができるよう、セルの気体体積を 2 等分)
        self.area = self.cell_vol[self._tri_cell] * 0.5
        self.vol = self.area

    def _set_amr_args(self) -> None:
        cp = self.cp
        if self.lay is None:
            self._amr_args = ()
            return
        lay = self.lay
        self._amr_args = (cp.asarray(lay.dd), cp.asarray(lay.di), cp.asarray(lay.refined), cp.asarray(lay.blockid),
                          np.int32(lay.n_cells))

    def _alloc_tags(self) -> None:
        cp = self.cp
        self._tag_cnt = cp.zeros(self.n_cells)
        self._tag_v = cp.zeros(3 * self.n_cells)
        self._tag_v2 = cp.zeros(self.n_cells)
        self._tag_samples = 0

    # ---- 境界 ---------------------------------------------------------------------------

    def _build_boundaries(self) -> None:
        """外周 4 辺の区間表 (c0, c1, 種類, 温度) と流入口 (リザーバ・流量指定) を作る。

        v1 と同じ規約: 指定の無い部分と導体・誘電体の輪郭は壁温の拡散反射壁、後の指定が先の
        指定を上書き、流量指定の流入口は入射分子を拡散反射し指定流量だけを注入、軸対称の
        対称軸 (r=0) は鏡面に強制。線分指定 (p1-p2) は外周の辺上の部分区間 (prompts/55)。
        """
        s = self.s
        d = self.model.domain
        span = {"left": (d.y0, d.y1), "right": (d.y0, d.y1), "bottom": (d.x0, d.x1), "top": (d.x0, d.x1)}
        tol = 1e-6 * max(d.x1 - d.x0, d.y1 - d.y0)
        intervals: dict[str, list] = {sd: [] for sd in _SIDES}
        inflow: list[dict] = []
        for bc in s.boundaries:
            flow_inlet = bc.type == "inlet" and bc.flow_sccm is not None
            if flow_inlet:
                code = _B_WALL
            elif bc.type in ("inlet", "outlet"):
                code = _B_RES if (bc.pressure_pa and bc.pressure_pa > 0.0) else _B_VAC
            else:
                code = {"wall": _B_WALL, "symmetry": _B_SYM}[bc.type]
            pieces = []   # (side, c0, c1)
            for e in bc.edges:
                side = {v: k for k, v in _SIDE_EDGE.items()}[e % 4]
                pieces.append((side, *span[side]))
            if bc.p1 is not None and bc.p2 is not None:
                (ax, ay), (bx, by) = bc.p1, bc.p2
                for side in _SIDES:
                    if side in ("left", "right"):
                        xs = d.x0 if side == "left" else d.x1
                        if abs(ax - xs) <= tol and abs(bx - xs) <= tol:
                            pieces.append((side, max(min(ay, by), d.y0), min(max(ay, by), d.y1)))
                    else:
                        ys = d.y0 if side == "bottom" else d.y1
                        if abs(ay - ys) <= tol and abs(by - ys) <= tol:
                            pieces.append((side, max(min(ax, bx), d.x0), min(max(ax, bx), d.x1)))
            if not pieces:
                continue
            for side, c0, c1 in pieces:
                intervals[side].append((c0, c1, code, bc.temperature_k))
            if flow_inlet or code == _B_RES:
                areas = [self._segment_area(side, c0, c1) for side, c0, c1 in pieces]
                total_a = sum(areas)
                for (side, c0, c1), a in zip(pieces, areas):
                    if flow_inlet:
                        rate = bc.flow_sccm * SCCM_TO_PER_S * a / total_a if total_a > 0 else 0.0
                    else:
                        n_res = bc.pressure_pa / (KB * bc.temperature_k)
                        rate = n_res * math.sqrt(KB * bc.temperature_k / (2.0 * math.pi * self.m)) * a
                    inflow.append(self._inflow_entry(side, c0, c1, rate, bc.temperature_k))
        if self.rz:
            axis = "bottom" if self.ridx == 1 else "left"
            if (self.ridx == 1 and abs(d.y0) <= tol) or (self.ridx == 0 and abs(d.x0) <= tol):
                intervals[axis].append((*span[axis], _B_SYM, s.wall_temperature_k))
        side_off, iv = [0], []
        for side in _SIDES:
            for c0, c1, code, temp in intervals[side]:
                iv.extend([c0, c1, float(code), temp])
            side_off.append(len(iv) // 4)
        self._side_off_h = side_off
        self._iv_h = iv
        self._res_edges = inflow

    def _segment_area(self, side: str, c0: float, c1: float) -> float:
        """外周の区間の面積 (xy: 長さ × 奥行き 1 m、rz: 回転面の面積)。"""
        L = max(c1 - c0, 0.0)
        if not self.rz:
            return L
        d = self.model.domain
        radial_side = side in (("left", "right") if self.ridx == 1 else ("bottom", "top"))
        if radial_side:            # 辺に沿って r が変わる: 円環 π (r1² − r0²)
            return math.pi * (c1 * c1 - c0 * c0)
        r = {"left": d.x0, "right": d.x1, "bottom": d.y0, "top": d.y1}[side]
        return 2.0 * math.pi * r * L

    def _inflow_entry(self, side: str, c0: float, c1: float, rate: float, temp: float) -> dict:
        d = self.model.domain
        if side in ("left", "right"):
            xs = d.x0 if side == "left" else d.x1
            a, b = (xs, c0), (xs, c1)
            nrm = (1.0, 0.0) if side == "left" else (-1.0, 0.0)
        else:
            ys = d.y0 if side == "bottom" else d.y1
            a, b = (c0, ys), (c1, ys)
            nrm = (0.0, 1.0) if side == "bottom" else (0.0, -1.0)
        rweight = bool(self.rz and side in (("left", "right") if self.ridx == 1 else ("bottom", "top")))
        return {"a": a, "b": b, "nrm": nrm, "rate": rate, "temp": temp, "frac": 0.0, "rweight": rweight}

    # ---- 粒子配列 -------------------------------------------------------------------------

    def _alloc(self, cap: int) -> None:
        cp = self.cp
        old = getattr(self, "_p", None)
        self._p = {k: cp.zeros(cap) for k in ("x", "y", "vx", "vy", "vz")}
        self._tmp = {k: cp.zeros(cap) for k in ("x", "y", "vx", "vy", "vz")}
        self._key = cp.zeros(cap, dtype=np.int32)
        self._claim = cp.zeros(cap, dtype=np.uint32)   # 衝突の占有フラグ (ステップ番号のスタンプ)
        if old is not None and self.n:
            for k in old:
                self._p[k][: self.n] = old[k][: self.n]
        self.cap = cap

    def _ensure_capacity(self, n_new: int) -> None:
        if n_new > self.cap:
            self._alloc(int(1.6 * n_new) + 1024)

    def _load_initial(self, n0: int) -> None:
        cp = self.cp
        rng = np.random.default_rng(self.s.seed)
        d = self.model.domain
        xs, ys = [], []
        need = n0
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
        sig0 = math.sqrt(KB * self.s.init_temperature_k / self.m)
        v = rng.normal(0.0, sig0, size=(n0, 3))
        self._ensure_capacity(n0)
        p = self._p
        p["x"][:n0] = cp.asarray(np.concatenate(xs) if xs else np.zeros(0))
        p["y"][:n0] = cp.asarray(np.concatenate(ys) if ys else np.zeros(0))
        for c, k in enumerate(("vx", "vy", "vz")):
            p[k][:n0] = cp.asarray(v[:, c])
        self.n = n0
        # 所属セル (移動前の初期位置)
        self._key[:n0] = cp.asarray(self._cell_of(p["x"][:n0].get(), p["y"][:n0].get()).astype(np.int32))

    def _cell_of(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """点の所属セル (ホスト。一様格子は floor、AMR は葉セル)。"""
        if self.lay is not None:
            return cell_of_points(self.lay, x, y)
        g = self.grid
        i = np.clip(np.floor((x - g.x0) / g.dx).astype(np.int64), 0, g.nx - 1)
        j = np.clip(np.floor((y - g.y0) / g.dy).astype(np.int64), 0, g.ny - 1)
        return j * g.nx + i

    @property
    def x(self) -> np.ndarray:
        """生存粒子の位置 (n, 2) (ホスト配列。v1 の DsmcSimulation.x と同じ形)。"""
        return np.stack([self._p["x"][: self.n].get(), self._p["y"][: self.n].get()], axis=1)

    @property
    def v(self) -> np.ndarray:
        return np.stack([self._p[k][: self.n].get() for k in ("vx", "vy", "vz")], axis=1)

    # ---- 1 ステップ -----------------------------------------------------------------------

    def _inject(self) -> None:
        if not self._res_edges:
            return
        ks = []
        for ed in self._res_edges:
            quota = ed["rate"] * self.dt / self.w + ed["frac"]
            k = int(quota)
            ed["frac"] = quota - k
            ks.append(k)
        total = sum(ks)
        if total == 0:
            return
        self._ensure_capacity(self.n + total)
        p = self._p
        base = self.n
        for sid, (ed, k) in enumerate(zip(self._res_edges, ks)):
            if k == 0:
                continue
            (ax, ay), (bx, by), (nx_, ny_) = ed["a"], ed["b"], ed["nrm"]
            self._k["dsmc_inject"](_grid1(k), (_BLOCK,), (
                p["x"], p["y"], p["vx"], p["vy"], p["vz"], np.int64(base), np.int64(k),
                np.float64(ax), np.float64(ay), np.float64(bx), np.float64(by), np.float64(nx_), np.float64(ny_),
                np.float64(ed["temp"]), np.int32(1 if ed["rweight"] else 0),
                np.int32(self.ridx if self.rz else -1), np.float64(self._delta), np.float64(self.m),
                np.uint64(self.step_count), self._seed, np.uint64(sid)))
            base += k
        if self._sampling:
            self.inflow += total * self.w
        self.n += total

    def _move(self) -> None:
        g = self.grid
        d = self.model.domain
        p = self._p
        if self.n == 0:
            return
        self._k["dsmc_move"](_grid1(self.n), (_BLOCK,), (
            p["x"], p["y"], p["vx"], p["vy"], p["vz"], self._key, np.int64(self.n), np.float64(self.dt),
            np.int32(1 if self.rz else 0), np.int32(self.ridx if self.rz else 0),
            np.float64(d.x0), np.float64(d.y0), np.float64(d.x1), np.float64(d.y1),
            np.float64(1.0 / g.dx), np.float64(1.0 / g.dy), np.int32(g.nx), np.int32(g.ny), self._cell_state,
            self._side_off, self._iv, np.float64(self.s.wall_temperature_k), np.float64(self.m),
            np.int32(self._n_solid), self._s_type, self._s_off, self._s_pxy, self._s_circ,
            np.float64(self._delta), np.uint64(self.step_count), self._seed, self._cnt, *self._amr_args))

    def _sort(self) -> None:
        """セル番号で安定ソートし (吸収された分子は末尾)、生存数・セルごとの開始位置と個数を得る。"""
        cp = self.cp
        n = self.n
        nc = self.n_cells
        if n == 0:
            self._count.fill(0)
            self._start.fill(0)
            return
        keys = self._key[:n]
        order = cp.argsort(keys) if n > 1 else cp.zeros(1, dtype=np.int64)   # 基数ソート (安定)
        counts = cp.bincount(keys, minlength=nc + 1).astype(np.int32)
        self._count[...] = counts[: nc + 1]
        self._start[0] = 0
        self._start[1:] = cp.cumsum(counts[:nc]).astype(np.int32)
        for k in ("x", "y", "vx", "vy", "vz"):
            cp.take(self._p[k][:n], order, out=self._tmp[k][:n])
        self._p, self._tmp = self._tmp, self._p
        removed = int(counts[nc])
        self.n = n - removed
        self._key[: self.n] = keys[order][: self.n]
        if self._sampling:
            self.outflow += removed * self.w

    def _collide(self) -> None:
        """NTC: セルごとの候補数 → 候補対ごとに 1 スレッド (kernels/dsmc.cu の dsmc_pairs)。"""
        if self.n < 2:
            return
        cp = self.cp
        nc = self.n_cells
        p = self._p
        self._k["dsmc_ncand"](_grid1(nc), (_BLOCK,), (
            self._count, self._vol_d, self._sigcr_max, self._coll_frac, self._ncand,
            np.float64(self.w * self.dt), np.int32(nc)))
        self._cand_start[1:] = cp.cumsum(self._ncand)
        total = int(self._cand_start[nc])
        if total == 0:
            return
        stamp = np.uint32((self.step_count + 1) & 0xFFFFFFFF)
        self._k["dsmc_pairs"](_grid1(total), (_BLOCK,), (
            p["vx"], p["vy"], p["vz"], self._start, self._count, self._cand_start, np.int32(nc), np.int64(total),
            self._sigcr_max, self._coll_frac, self._claim, stamp, np.float64(self._sig_coef),
            np.float64(self._sig_pow), np.uint64(self.step_count), self._seed, self._cnt[1:2]))

    def _sample(self) -> None:
        p = self._p
        self._k["dsmc_sample"](_grid1(self.n_cells), (_BLOCK,), (
            p["vx"], p["vy"], p["vz"], self._start, self._count, self._acc_cnt, self._acc_v, self._acc_v2,
            np.int32(self.n_cells)))
        self._samples += 1

    def _sample_tags(self) -> None:
        p = self._p
        self._k["dsmc_sample"](_grid1(self.n_cells), (_BLOCK,), (
            p["vx"], p["vy"], p["vz"], self._start, self._count, self._tag_cnt, self._tag_v, self._tag_v2,
            np.int32(self.n_cells)))
        self._tag_samples += 1

    def _mean_free_path(self) -> np.ndarray:
        """積算した密度・温度からセルごとの平均自由行程 λ = 1 / (√2 n σ(T)) (VHS、分子の居ないセルは inf)。"""
        gas = self.s.gas
        cnt = self._tag_cnt.get()
        v = self._tag_v.get().reshape(-1, 3)
        v2 = self._tag_v2.get()
        vol = self.cell_vol
        ok = (cnt > 0.0) & (vol > 0.0)
        lam = np.full(self.n_cells, np.inf)
        if not np.any(ok) or self._tag_samples == 0:
            return lam
        n = cnt[ok] * self.w / (self._tag_samples * vol[ok])
        u = v[ok] / cnt[ok, None]
        t = np.maximum(self.m / (3.0 * KB) * (v2[ok] / cnt[ok] - np.sum(u * u, axis=1)), 1.0)
        sigma = math.pi * gas.d_ref_m**2 * (gas.t_ref_k / t) ** (gas.omega - 0.5)
        lam[ok] = 1.0 / (math.sqrt(2.0) * n * sigma)
        return lam

    def _regrid(self) -> bool:
        """区間平均の平均自由行程で AMR 階層を作り直し、分子のセル番号を付け直す (ここで同期)。

        分子は座標・速度のまま。セルごとの (σc_r)_max は新しいセルの中心を含む旧セルの値を引き継ぎ、
        NTC の端数は 0 に戻す。平均区間の前だけ呼ぶ (サンプリングの積算は空)。戻り値: 格子が変わったか。
        """
        def same(h1, h2) -> bool:
            return h1.max_level == h2.max_level and all(np.array_equal(a, b) for a, b in zip(h1.refined, h2.refined))

        cp = self.cp
        lay = self.lay
        spec = lay.hier.spec
        lam = self._mean_free_path()
        self._alloc_tags()
        if not np.any(np.isfinite(lam)):
            return False
        cand = lay.hier
        for _ in range(spec.max_level + 1):
            nxt = AmrHierarchy(self.model, self.grid, spec,
                               extra_tags=mfp_tags(lay, lam, cand, spec.dsmc_h_over_mfp, keep=lay.hier))
            if same(nxt, cand):
                break
            cand = nxt
        if same(cand, lay.hier):
            return False
        t0 = time.perf_counter()
        new = build_cell_layout(self.model, cand)
        old_ids = cell_of_points(lay, new.cell_xy[:, 0], new.cell_xy[:, 1])
        sig_new = self._sigcr_max.get()[old_ids]
        sig_new[new.cell_level < 0] = self._sigcr0
        self.lay = new
        self.n_cells = new.n_cells
        self.cell_vol = new.cell_vol_gas
        self._set_display(self._display_of(new))
        self._set_amr_args()
        self._sigcr_max = cp.asarray(sig_new)
        self._coll_frac = cp.zeros(self.n_cells)
        self._ncand = cp.zeros(self.n_cells, dtype=np.int64)
        self._cand_start = cp.zeros(self.n_cells + 1, dtype=np.int64)
        self._vol_d = cp.asarray(self.cell_vol)
        self._acc_cnt = cp.zeros(self.n_cells)
        self._acc_v = cp.zeros(3 * self.n_cells)
        self._acc_v2 = cp.zeros(self.n_cells)
        self._start = cp.zeros(self.n_cells + 1, dtype=np.int32)
        self._count = cp.zeros(self.n_cells + 1, dtype=np.int32)
        self._alloc_tags()
        if self.n:
            p = self._p
            self._k["dsmc_relocate"](_grid1(self.n), (_BLOCK,), (p["x"], p["y"], self._key, np.int64(self.n),
                                                                 *self._amr_args))
        self._sort()
        self.regrid_log.append({"step": self.step_count, "levels": new.hier.n_levels,
                                "leaf_cells": new.hier.n_leaf_cells(), "setup_s": time.perf_counter() - t0})
        return True

    def step(self) -> None:
        t0 = time.perf_counter()
        self._inject()
        t1 = time.perf_counter()
        self._move()
        self._sort()
        t2 = time.perf_counter()
        self._collide()
        self.cp.cuda.get_current_stream().synchronize()
        t3 = time.perf_counter()
        self.timing["inject"] += t1 - t0
        self.timing["move"] += t2 - t1
        self.timing["collide"] += t3 - t2
        self.step_count += 1

    # ---- 結果 -----------------------------------------------------------------------------

    def _smooth_moments(self, acc_cnt: np.ndarray, acc_v: np.ndarray, acc_v2: np.ndarray):
        """4 近傍の体積重み対称拡散 (v1 の _smooth_moments の直交格子版、総量保存・正値)。"""
        passes = int(self.s.smoothing_passes)
        if passes <= 0:
            return acc_cnt, acc_v, acc_v2
        if self.lay is not None:
            vol = self.cell_vol
            safe = np.where(vol > 0.0, vol, 1.0)[:, None]
            q = np.concatenate([acc_cnt[:, None], acc_v.reshape(-1, 3), acc_v2[:, None]], axis=1) / safe
            q = smooth_cells(self.lay, q, passes) * vol[:, None]
            return q[:, 0], q[:, 1:4].reshape(-1), q[:, 4]
        g = self.grid
        vol = self.cell_vol.reshape(g.ny, g.nx)
        gas = vol > 0.0
        safe = np.where(gas, vol, 1.0)
        q = np.concatenate([acc_cnt.reshape(g.ny, g.nx, 1), acc_v.reshape(g.ny, g.nx, 3),
                            acc_v2.reshape(g.ny, g.nx, 1)], axis=2) / safe[..., None]
        q[~gas] = 0.0
        theta = 0.2
        for _ in range(passes):
            dq = np.zeros_like(q)
            for axis in (0, 1):
                a = [slice(None)] * 2
                b = [slice(None)] * 2
                a[axis], b[axis] = slice(0, -1), slice(1, None)
                a, b = tuple(a), tuple(b)
                wij = np.where(gas[a] & gas[b], np.minimum(vol[a], vol[b]), 0.0)[..., None]
                flux = wij * (q[b] - q[a])
                dq[a] += flux
                dq[b] -= flux
            q = q + theta * dq / safe[..., None]
            q[~gas] = 0.0
        q = q * vol[..., None]
        return q[..., 0].ravel(), q[..., 1:4].reshape(-1), q[..., 4].ravel()

    def _thin_positions(self) -> np.ndarray:
        n = self.n
        if n == 0:
            return np.zeros((0, 2))
        idx = self.cp.asarray(np.linspace(0, n - 1, min(n, MAX_CALLBACK_PARTICLES)).astype(np.int64))
        return np.stack([self._p["x"][idx].get(), self._p["y"][idx].get()], axis=1)

    def _accum_walk_diag(self) -> None:
        if self.n == 0:
            return
        p = self._p
        vm = float(self.cp.mean(self.cp.sqrt(p["vx"][: self.n] ** 2 + p["vy"][: self.n] ** 2 + p["vz"][: self.n] ** 2)))
        self._walk_diag_sum += vm * self.dt / self.timing["h_mean_m"]
        self._walk_diag_n += 1

    def prepare_continue(self, n_steps: int, avg_steps: int | None = None) -> None:
        """完了/停止後の状態から追加実行の準備 (v1 と同じ契約: 粒子・乱数・端数は維持)。"""
        self.s.n_steps = int(n_steps)
        if avg_steps is not None:
            self.s.avg_steps = int(avg_steps)
        self._reset_sampling()
        h = self.timing.get("h_mean_m", 0.0)
        self.timing = {k: 0.0 for k in self.timing}
        self.timing["h_mean_m"] = self.timing["h_min_m"] = h
        self._walk_diag_sum = 0.0
        self._walk_diag_n = 0

    def _reset_sampling(self) -> None:
        self._acc_cnt.fill(0.0)
        self._acc_v.fill(0.0)
        self._acc_v2.fill(0.0)
        self._samples = 0
        self.inflow = 0.0
        self.outflow = 0.0

    def run(self, callback=None, should_stop=None) -> DsmcResult:
        """n_steps 進め、最終 avg_steps の時間平均から DsmcResult を作る (v1 と同じ契約)。"""
        t0 = time.perf_counter()
        n_steps = self.s.n_steps
        avg_start = max(0, n_steps - self.s.avg_steps)
        self._sampling = False
        for i in range(n_steps):
            if should_stop is not None and should_stop():
                break
            t_o0 = time.perf_counter()
            if i == avg_start:
                self._reset_sampling()
                self._sampling = True
            t_o1 = time.perf_counter()
            self.step()
            t_s0 = time.perf_counter()
            if i >= avg_start:
                self._sample()
            elif self._regrid_every:
                # 平均区間の前だけ: 密度・温度を積算し、間隔ごとに平均自由行程で格子を作り直す
                if (i + 1) % TAG_EVERY == 0:
                    self._sample_tags()
                if (i + 1) % self._regrid_every == 0 and i + 1 < avg_start:
                    self._regrid()
            t_s1 = time.perf_counter()
            if (i + 1) % PROGRESS_EVERY == 0 or i == n_steps - 1:
                self._accum_walk_diag()
            if callback is not None and (i + 1) % PROGRESS_EVERY == 0:
                callback(i + 1, self.n, self._thin_positions())
            t_c1 = time.perf_counter()
            self.timing["other"] += (t_o1 - t_o0) + (t_c1 - t_s1)
            self.timing["sample"] += t_s1 - t_s0
        self._sampling = False
        if self._samples == 0:
            raise ValueError(
                "平均区間に入る前に停止したため、ガス場の結果がありません "
                "(avg_steps を長くするか、停止せず完走させてください)"
            )
        self.timing["walk_cells_est"] = self._walk_diag_sum / self._walk_diag_n if self._walk_diag_n else 0.0
        acc_cnt, acc_v, acc_v2 = self._smooth_moments(self._acc_cnt.get(), self._acc_v.get(), self._acc_v2.get())
        acc_v = acc_v.reshape(-1, 3)
        vol = self.cell_vol
        cnt = np.maximum(acc_cnt, 1e-300)
        with np.errstate(divide="ignore", invalid="ignore"):
            n_cell = np.where(vol > 0.0, acc_cnt * self.w / (self._samples * np.where(vol > 0, vol, 1.0)), 0.0)
        u3 = acc_v / cnt[:, None]
        v2 = acc_v2 / cnt
        t_cell = np.maximum(self.m / (3.0 * KB) * (v2 - np.sum(u3 * u3, axis=1)), 0.0)
        empty = acc_cnt <= 0.0
        t_cell[empty] = 0.0
        u3[empty] = 0.0
        tc = self._tri_cell
        n_e = n_cell[tc]
        t_e = t_cell[tc]
        return DsmcResult(
            n=n_e, t=t_e, u=u3[tc, :2], p=n_e * KB * t_e, n_particles=self.n, macro_weight=self.w, dt=self.dt,
            inflow=self.inflow, outflow=self.outflow, elapsed_s=time.perf_counter() - t0, timing=dict(self.timing),
        )
