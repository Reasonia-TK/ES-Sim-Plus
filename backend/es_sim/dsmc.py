"""定常ガス流れの DSMC (Direct Simulation Monte Carlo、prompts/54)。

- セル = 既存の三角形メッシュ要素 (meshing / particles のインフラを再利用)
- 分子モデル: VHS (Variable Hard Sphere)。σ_T(c_r) = π d_ref² (c_ref/c_r)^{2ω−1} 型
  (Bird 1994 の標準形、Γ(5/2−ω) 因子込み)
- 衝突: NTC (No-Time-Counter) 法。セルごとに候補対数
  N_cand = ½ N(N−1) W (σc_r)_max Δt / V を抽選し、σ(c_r)c_r/(σc_r)_max で採択
- 境界: 拡散反射壁 (完全適応)・鏡面反射・圧力リザーバ流入/流出・真空排気。
  壁到達粒子は交点に置き直して境界条件に応じた速度を与える (交点は重心座標 L=0 の
  線形補間、粒子は交点からエッジ高さ×1e-3 だけ内側へ配置)。残余移動時間は
  無視する近似 (Δt がセル通過時間より小さければ誤差は小さい)
- 誘電体 (固体) 要素へ入った粒子は移動を取り消し、壁温の等方 Maxwell で
  再抽選する (質量保存を守る簡易処理)
- 定常判定はユーザ指定の n_steps に委ね、最終 avg_steps の時間平均で
  セルごとの n・u・T (・p = n kB T) を得る。結果は MCC の GasField にそのまま渡せる
- 平滑化 (smoothing_passes、prompts/67): 統計ノイズ低減のため、n/T/u/p 導出前の
  生モーメントに隣接セル対称拡散を適用できる (体積重みで総量厳密保存・正値保証)。
  既定 0 (無効、従来と完全一致)
- 平面2D (coord="xy") 専用。奥行き 1 m 換算 (セル体積 = 面積 × 1 m)
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from . import _numba_kernels
from .fem import _radial_index
from .meshing import generate_mesh
from .particles import (
    _adjacency,
    _barycentric_coeffs,
    _pack_coeffs,
    _solid_elements,
    _walk_step,
)
from .schema import DsmcSettings, Project

KB = 1.380649e-23     # ボルツマン定数 [J/K]
AMU = 1.66053906660e-27  # 原子質量単位 [kg]

# 1 sccm = 標準状態 (273.15 K, 101325 Pa) の 1 cm^3/min の分子数流量 [分子/s]
# = (101325/(kB·273.15))·1e-6/60
SCCM_TO_PER_S = 101325.0 / (KB * 273.15) * 1e-6 / 60.0  # ≈ 4.478e17

# 境界タイプの内部コード ((要素, ローカルエッジ) 表で引く)
_B_WALL = 0
_B_SYMMETRY = 1
_B_RESERVOIR = 2  # inlet / outlet (圧力 > 0): 吸収 + リザーバ流入
_B_VACUUM = 3     # outlet (圧力なし): 吸収のみ

# 進捗コールバックへ渡す間引き粒子数の上限 (pic.py の MAX_FRAME_PARTICLES と同じ値。
# ライブ表示用途は同程度の点数で十分なため揃えている)
MAX_CALLBACK_PARTICLES = 2000


@dataclass
class DsmcResult:
    """定常時間平均のガス場 (要素ごと)。"""

    n: np.ndarray        # (M,) 数密度 [m^-3]
    t: np.ndarray        # (M,) 温度 [K]
    u: np.ndarray        # (M, 2) 面内流速 [m/s]
    p: np.ndarray        # (M,) 圧力 [Pa] = n kB T
    n_particles: int     # 最終シミュレーション粒子数
    macro_weight: float  # 実分子数/シミュレーション粒子
    dt: float            # 実際に使った dt [s]
    inflow: float        # 平均区間の流入実分子数 (リザーバ)
    outflow: float       # 平均区間の流出実分子数 (リザーバ + 真空)


class DsmcSimulation:
    """DSMC 本体。run() で n_steps 進めて DsmcResult を返す。"""

    def __init__(self, project: Project):
        if project.dsmc is None:
            raise ValueError("project.dsmc が指定されていません")
        self.project = project
        self.s: DsmcSettings = project.dsmc
        gas = self.s.gas
        self.m = gas.mass_amu * AMU
        self.mu = 0.5 * self.m  # 同種2体の換算質量
        # 軸対称モード (prompts/56)。粒子はリング (周方向一様)、v[:, 2] は vθ。
        # 自由飛行は3D直線の厳密な子午面射影 (r' = √((r+v_r dt)² + (v_θ dt)²)) で扱う
        self.ridx = _radial_index(project.coord)
        self.rz = self.ridx is not None

        # ---- メッシュ (particles.py のインフラを再利用) ----------------------
        self.mesh = generate_mesh(project)
        self.tris = self.mesh.triangles
        self.coeffs = _barycentric_coeffs(self.mesh.nodes, self.tris)
        self._packed = _pack_coeffs(self.coeffs)
        self.adjacency = _adjacency(self.tris)
        det = self.coeffs[3]
        self.area = 0.5 * np.abs(det)
        # セル体積: xy = 面積 (奥行き1m換算)、rz = 2π r̄ A。xy では area と同一
        # オブジェクトなので既存経路は数値不変
        if self.rz:
            self.r_bar = self.mesh.nodes[self.tris][:, :, self.ridx].mean(axis=1)
            self.vol = 2.0 * np.pi * self.r_bar * self.area
        else:
            self.vol = self.area
        self._solid = _solid_elements(project, self.mesh)  # 誘電体 (固体) 要素 or None

        # ---- 粒子チャンク並列 (prompts/65) ------------------------------------
        # walk 探索を粒子チャンクに分けてスレッド並列化する (pic.py の
        # _walk_chunked_submit と同じ方式)。粒子ごとの walk は独立 (読み取り共有のみ・
        # 乱数不使用) な決定的処理なので、チャンク分割しても結果は逐次実行と
        # ビット単位で一致する
        self._nthreads = max(1, int(self.s.threads))
        self._chunk_pool = (
            ThreadPoolExecutor(max_workers=self._nthreads, thread_name_prefix="dsmc-chunk")
            if self._nthreads > 1
            else None
        )
        # numba の walk カーネル (prange) が使うスレッド数を threads に合わせる
        # (numba 無し環境では no-op。numpy フォールバックは上の _chunk_pool で並列化する)
        _numba_kernels.set_num_threads(self._nthreads)
        gas_area = float(
            self.vol.sum() if self._solid is None else self.vol[~self._solid].sum()
        )

        # ---- 境界エッジ表 ((要素, ローカルエッジ) で引く) --------------------
        self._build_boundary_tables()

        # ---- マクロ重み・時間刻み --------------------------------------------
        # マクロ重みは初期充填密度基準 (prompts/57)。以前は最大リザーバ密度基準
        # だったが、小孔から高圧を供給する形状では大半の領域が初期圧近傍に留まり、
        # 実粒子数が n_particles 目標の何十分の一にもなって統計が崩れていた。
        # 初期圧は想定される定常の代表圧に合わせること (定常密度が初期圧より
        # 大きく育つ設定では粒子数が n_particles を超えて増える)
        n_init = self.s.init_pressure_pa / (KB * self.s.init_temperature_k)
        self.w = n_init * gas_area / self.s.n_particles  # 実分子数/シミュレーション粒子

        t_hot = max(
            [self.s.init_temperature_k, self.s.wall_temperature_k]
            + [bc.temperature_k for bc in self.s.boundaries]
        )
        v_mp = math.sqrt(2.0 * KB * t_hot / self.m)  # 最確速さ
        if self.s.dt is not None:
            self.dt = float(self.s.dt)
        else:
            # セル代表寸法 (最小要素の外接半径相当) を最確速さで割った時間の 1/4
            p = self.mesh.nodes[self.tris]
            edges = (
                np.linalg.norm(p[:, 0] - p[:, 1], axis=1)
                + np.linalg.norm(p[:, 1] - p[:, 2], axis=1)
                + np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
            ) / 3.0
            self.dt = 0.25 * float(edges.min()) / v_mp

        # ---- VHS 断面積 -------------------------------------------------------
        # σ_T(c_r) = π d_ref² · (2 kB T_ref / (μ c_r²))^{ω−1/2} / Γ(5/2 − ω)
        self._sig_coef = (
            math.pi * gas.d_ref_m**2
            / math.gamma(2.5 - gas.omega)
            * (2.0 * KB * gas.t_ref_k / self.mu) ** (gas.omega - 0.5)
        )
        self._sig_pow = 1.0 - 2.0 * gas.omega  # c_r 依存: c_r^{1−2ω} (× c_r で σc_r)
        # (σ c_r)_max のセル別初期値: c_r ~ 2 v_mp で評価 (走行中に実測最大へ更新)
        cr0 = 2.0 * v_mp
        self._sigcr_max = np.full(len(self.tris), self._sigma(np.array([cr0]))[0] * cr0)
        self._coll_frac = np.zeros(len(self.tris))  # NTC 候補数の端数持ち越し

        # ---- 初期充填 (一様 Maxwell) -----------------------------------------
        self.rng = np.random.default_rng(self.s.seed)
        n0 = int(round(n_init * gas_area / self.w))
        loadable = (
            np.arange(len(self.tris))
            if self._solid is None
            else np.nonzero(~self._solid)[0]
        )
        p_elem = self.vol[loadable] / self.vol[loadable].sum()
        elem0 = loadable[self.rng.choice(len(loadable), size=n0, p=p_elem)]
        r1 = np.sqrt(self.rng.random(n0))
        r2 = self.rng.random(n0)
        pts = self.mesh.nodes[self.tris[elem0]]
        self.x = (
            (1.0 - r1)[:, None] * pts[:, 0]
            + (r1 * (1.0 - r2))[:, None] * pts[:, 1]
            + (r1 * r2)[:, None] * pts[:, 2]
        )
        sig0 = math.sqrt(KB * self.s.init_temperature_k / self.m)
        self.v = self.rng.normal(0.0, sig0, size=(n0, 3))
        self.elem = elem0.astype(np.int64)

        # ---- サンプリング (最終 avg_steps) -----------------------------------
        self._acc_cnt = np.zeros(len(self.tris))
        self._acc_v = np.zeros((len(self.tris), 3))
        self._acc_v2 = np.zeros(len(self.tris))
        self._samples = 0
        self.inflow = 0.0
        self.outflow = 0.0

        # 通算ステップ数 (continue (prompts/74) をまたいで増え続ける)。run() 内の
        # サンプリング区間判定はこの回の self.s.n_steps/avg_steps から独立に
        # 求まるので物理計算には使わないが、PIC の step_count と同じく「今まで
        # 何ステップ進めたか」を外から参照できるように保持しておく
        self.step_count = 0

    # ---- VHS 断面積 -----------------------------------------------------------

    def _sigma(self, c_r: np.ndarray) -> np.ndarray:
        """VHS 全断面積 σ_T(c_r) = coef · c_r^{1−2ω} [m²] (c_r ≤ 0 は 0)。"""
        out = np.zeros_like(c_r)
        pos = c_r > 0.0
        out[pos] = self._sig_coef * c_r[pos] ** self._sig_pow
        return out

    # ---- 境界表 ---------------------------------------------------------------

    def _build_boundary_tables(self) -> None:
        """境界メッシュエッジ (隣接 = -1) の種別・温度・リザーバ密度と幾何を作る。

        domain 外周エッジは DsmcBoundary の指定 (エッジ中点が指定 domain エッジ上に
        あるか) で分類し、未指定エッジと conductor 輪郭は拡散反射壁とする。
        """
        tris = self.tris
        nodes = self.mesh.nodes
        ts, loc = np.nonzero(self.adjacency == -1)
        n1 = tris[ts, (loc + 1) % 3]
        n2 = tris[ts, (loc + 2) % 3]
        n_opp = tris[ts, loc]
        p1, p2, po = nodes[n1], nodes[n2], nodes[n_opp]
        mid = 0.5 * (p1 + p2)
        t_vec = p2 - p1
        length = np.linalg.norm(t_vec, axis=1)
        perp = np.stack([-t_vec[:, 1], t_vec[:, 0]], axis=1)
        sgn = np.where(np.sum(perp * (po - mid), axis=1) >= 0.0, 1.0, -1.0)
        nrm = perp * (sgn / length)[:, None]  # 内向き単位法線
        height = np.abs(np.sum((po - mid) * nrm, axis=1))

        m = len(tris)
        self._b_type = np.full((m, 3), _B_WALL, dtype=np.int8)
        self._b_temp = np.full((m, 3), self.s.wall_temperature_k)
        self._b_nrm = np.zeros((m, 3, 2))
        self._b_tan = np.zeros((m, 3, 2))
        self._b_delta = np.zeros((m, 3))
        self._b_nrm[ts, loc] = nrm
        self._b_tan[ts, loc] = t_vec / length[:, None]
        self._b_delta[ts, loc] = 1e-3 * height

        # domain 外周エッジ / 線分 (p1-p2、prompts/55) への割り当て
        # (境界メッシュエッジの中点が指定線分上にあるかで判定する)
        poly = np.asarray(self.project.geometry.domain.polygon, dtype=np.float64)
        nv = len(poly)
        scale = float(np.max(np.abs(poly)))
        tol_edge = 1e-8 * (scale if scale > 0.0 else 1.0)
        # 線分指定はユーザ入力座標なので緩めの許容 (グリッドスナップ誤差を吸収)
        tol_seg = 1e-6 * (scale if scale > 0.0 else 1.0)
        type_code = {"wall": _B_WALL, "symmetry": _B_SYMMETRY}

        def _mid_on(q1: np.ndarray, q2: np.ndarray, tol: float) -> np.ndarray:
            seg = q2 - q1
            seg_len = float(np.hypot(seg[0], seg[1]))
            if seg_len <= 0.0:
                return np.zeros(len(ts), dtype=bool)
            d = mid - q1
            dist = np.abs(d[:, 0] * seg[1] - d[:, 1] * seg[0]) / seg_len
            tpar = (d[:, 0] * seg[0] + d[:, 1] * seg[1]) / (seg_len * seg_len)
            return (dist <= tol) & (tpar >= -1e-9) & (tpar <= 1.0 + 1e-9)

        def _edge_entry(i: int, rate: float, temp: float) -> dict:
            """流入エッジの前計算エントリ (rate = 実分子数流量 [1/s])。"""
            return {
                "elem": int(ts[i]),
                "loc": int(loc[i]),
                "p1": p1[i],
                "p2": p2[i],
                "nrm": nrm[i],
                "tan": t_vec[i] / length[i],
                "delta": float(1e-3 * height[i]),
                "rate": rate,
                "temp": temp,
                "frac": 0.0,
            }

        # エッジの「面積」: xy = 長さ (奥行き1m換算)、rz = 輪帯面積 2π r̄_e L_e
        # (流束 [1/m²s] × 面積 = 分子数流量 [1/s] の換算に使う)
        if self.rz:
            a_edge = 2.0 * np.pi * mid[:, self.ridx] * length
        else:
            a_edge = length

        # 流入エッジのリスト (圧力リザーバ・流量指定の両方)
        self._res_edges: list[dict] = []
        for bc in self.s.boundaries:
            flow_inlet = bc.type == "inlet" and bc.flow_sccm is not None
            if flow_inlet:
                # 流量指定: 入射粒子は拡散反射壁として扱い (吸収しない)、
                # 指定流量だけを注入する → 正味流量が指定値に厳密一致する
                code = _B_WALL
            elif bc.type in ("inlet", "outlet"):
                code = (
                    _B_RESERVOIR if (bc.pressure_pa and bc.pressure_pa > 0.0) else _B_VACUUM
                )
            else:
                code = type_code[bc.type]

            on = np.zeros(len(ts), dtype=bool)
            for e in bc.edges:
                on |= _mid_on(poly[e % nv], poly[(e + 1) % nv], tol_edge)
            if bc.p1 is not None and bc.p2 is not None:
                on |= _mid_on(
                    np.asarray(bc.p1, dtype=np.float64),
                    np.asarray(bc.p2, dtype=np.float64),
                    tol_seg,
                )
            if not np.any(on):
                continue
            self._b_type[ts[on], loc[on]] = code
            self._b_temp[ts[on], loc[on]] = bc.temperature_k
            idxs = np.nonzero(on)[0]

            if flow_inlet:
                # 総分子流量 [1/s] をエッジ面積 (xy: 長さ、rz: 輪帯面積) で比例配分
                ndot = bc.flow_sccm * SCCM_TO_PER_S
                total_a = float(a_edge[idxs].sum())
                for i in idxs:
                    self._res_edges.append(
                        _edge_entry(i, ndot * float(a_edge[i]) / total_a, bc.temperature_k)
                    )
            elif code == _B_RESERVOIR:
                # 圧力リザーバ: 平衡流束 Φ = n c̄/4 = n √(kT/2πm) [1/m²s] × エッジ面積
                n_res = bc.pressure_pa / (KB * bc.temperature_k)
                flux = n_res * math.sqrt(KB * bc.temperature_k / (2.0 * math.pi * self.m))
                for i in idxs:
                    self._res_edges.append(
                        _edge_entry(i, flux * float(a_edge[i]), bc.temperature_k)
                    )

        # 軸対称: 対称軸 (r = 0) 上の境界メッシュエッジは物理境界ではないため、
        # ユーザ指定によらず鏡面 (数値的なかすり対策) に強制する。厳密な子午面
        # 射影では r' ≥ 0 なので通常は到達しない
        if self.rz:
            r1 = p1[:, self.ridx]
            r2 = p2[:, self.ridx]
            tol_axis = 1e-12 * (scale if scale > 0.0 else 1.0)
            on_axis = (np.abs(r1) <= tol_axis) & (np.abs(r2) <= tol_axis)
            if np.any(on_axis):
                self._b_type[ts[on_axis], loc[on_axis]] = _B_SYMMETRY

    # ---- 流入 (リザーバ) ------------------------------------------------------

    def _inject(self) -> None:
        """流入エッジ (圧力リザーバ・流量指定) から粒子を注入する。

        エッジごとの実分子数流量 rate [1/s] は前計算済み (リザーバ: 平衡流束
        n c̄/4 × 長さ、流量指定: sccm 換算をエッジ長で比例配分)。
        流入法線速度は流束重み付き Maxwell (v_n = σ√(−2 ln U))、接線・z 成分は
        熱速度の正規分布。端数はエッジごとに持ち越す。
        """
        if not self._res_edges:
            return
        new_x, new_v, new_e = [], [], []
        for ed in self._res_edges:
            sig = math.sqrt(KB * ed["temp"] / self.m)
            quota = ed["rate"] * self.dt / self.w + ed["frac"]
            k = int(quota)
            ed["frac"] = quota - k
            if k == 0:
                continue
            u = self.rng.random(k)
            pos = ed["p1"][None, :] + u[:, None] * (ed["p2"] - ed["p1"])[None, :]
            pos = pos + ed["delta"] * ed["nrm"][None, :]
            vn = sig * np.sqrt(-2.0 * np.log(np.maximum(self.rng.random(k), 1e-300)))
            vt = self.rng.normal(0.0, sig, size=k)
            vz = self.rng.normal(0.0, sig, size=k)
            vel = np.empty((k, 3))
            vel[:, :2] = vn[:, None] * ed["nrm"][None, :] + vt[:, None] * ed["tan"][None, :]
            vel[:, 2] = vz
            new_x.append(pos)
            new_v.append(vel)
            new_e.append(np.full(k, ed["elem"], dtype=np.int64))
            self.inflow += k * self.w if self._samples else 0.0
        if new_x:
            self.x = np.concatenate([self.x, *new_x])
            self.v = np.concatenate([self.v, *new_v])
            self.elem = np.concatenate([self.elem, *new_e])

    # ---- 移動と境界 -----------------------------------------------------------

    def _walk_chunked(self, elem: np.ndarray, x_new: np.ndarray):
        """walk を粒子チャンクに分割してスレッド並列実行する (prompts/65)。

        pic.py の _walk_chunked_submit を参考にした同期版: DSMC は種が1つ
        (walk 対象の粒子集合が常に単一) なので、この関数内で完了を待って結合
        結果を返せばよく、futures を呼び出し側へ持ち回す必要がない。
        粒子ごとの walk は独立 (読み取り共有のみ・書き込みは自分の行のみ) で
        乱数も使わない決定的処理なので、チャンク分割しても結果は逐次実行と
        ビット単位で一致する。粒子数が少ない場合はスレッド起動コストの方が
        高いためその場で逐次計算する (dsmc.py の _walk_step 呼び出しは l_new
        を渡していないため、ここでも常に None のまま _walk_step へ渡す)。

        numba あり (prompts/76) では walk カーネル自体が njit(parallel=True) の
        prange で全コアを使うため、ここでのスレッドチャンク分割は不要
        (二重並列化になるだけ)。1回の呼び出しに委ねる。
        """
        n = len(elem)
        if _numba_kernels.HAVE_NUMBA or n < 4096 or self._chunk_pool is None:
            return _walk_step(self.coeffs, self.adjacency, elem, x_new, packed=self._packed)
        k = self._nthreads
        bounds = [(i * n) // k for i in range(k + 1)]
        elem_new = np.empty(n, dtype=np.int64)
        absorbed = np.empty(n, dtype=bool)
        b_elem = np.empty(n, dtype=np.int64)
        b_loc = np.empty(n, dtype=np.int64)

        def _run(lo: int, hi: int) -> None:
            e, a, be, bl = _walk_step(
                self.coeffs, self.adjacency, elem[lo:hi], x_new[lo:hi], packed=self._packed
            )
            elem_new[lo:hi] = e
            absorbed[lo:hi] = a
            b_elem[lo:hi] = be
            b_loc[lo:hi] = bl

        futures = [
            self._chunk_pool.submit(_run, bounds[i], bounds[i + 1])
            for i in range(k)
            if bounds[i + 1] > bounds[i]
        ]
        for fut in futures:
            fut.result()
        return elem_new, absorbed, b_elem, b_loc

    def _move(self) -> None:
        """自由分子移動 + 境界処理。

        壁/対称/リザーバ/真空のいずれも、交点 (重心座標 L=0 の線形補間) で処理する。
        壁は拡散反射 (壁温 Maxwell 流束)、対称は鏡面反射、リザーバ・真空は吸収。
        再放出/反射した粒子は残余移動時間 (1−frac)·t を完走させる (最大6レグ。
        残余時間を捨てると壁近傍に放出直後の粒子が滞留して温度が数%高く出る)。
        固体要素へ入った粒子は移動を取り消して壁温の等方 Maxwell で再抽選する。
        """
        dt = self.dt
        n = len(self.x)
        if self.rz:
            # 軸対称 (prompts/56): 自由飛行は3D直線の厳密な子午面射影。
            # r' = √((r + v_r dt)² + (v_θ dt)²) は常に ≥ 0 で軸を越えない。
            # 速度は新しい子午面へ回転する (r' ≈ 0 の粒子は回転せずそのまま)。
            # walk・境界判定は (z, r) 平面の直線分近似 (dt が小さければ誤差は二次)
            ridx = self.ridx
            r_mid = self.x[:, ridx] + self.v[:, ridx] * dt
            d_out = self.v[:, 2] * dt
            r_new = np.sqrt(r_mid * r_mid + d_out * d_out)
            x_new = self.x + dt * self.v[:, :2]
            x_new[:, ridx] = r_new
            r_safe = np.maximum(r_new, 1e-300)
            cosw = np.where(r_new > 0.0, r_mid / r_safe, 1.0)
            sinw = np.where(r_new > 0.0, d_out / r_safe, 0.0)
            vr = self.v[:, ridx].copy()
            vt = self.v[:, 2].copy()
            self.v[:, ridx] = vr * cosw + vt * sinw
            self.v[:, 2] = vt * cosw - vr * sinw
        else:
            x_new = self.x + dt * self.v[:, :2]
        elem_new, absorbed, b_elem, b_loc = self._walk_chunked(self.elem, x_new)

        remove = np.zeros(n, dtype=bool)
        t_rem = np.full(n, dt)       # 残余移動時間
        x_from = self.x.copy()       # 現レグの開始位置
        e_from = self.elem.copy()    # 現レグの開始要素
        a_c, b_c, c_c, det_c = self.coeffs
        act = np.nonzero(absorbed)[0]

        for _ in range(6):
            if act.size == 0:
                break
            ea, el = b_elem[act], b_loc[act]
            btype = self._b_type[ea, el]

            # 交点 (境界エッジの重心座標 L=0 を線形補間)
            aa, bb, cc, dd = a_c[ea, el], b_c[ea, el], c_c[ea, el], det_c[ea]
            xp, xn = x_from[act], x_new[act]
            l0 = (aa + bb * xp[:, 0] + cc * xp[:, 1]) / dd
            l1 = (aa + bb * xn[:, 0] + cc * xn[:, 1]) / dd
            denom = np.where(np.abs(l0 - l1) < 1e-300, 1e-300, l0 - l1)
            frac = np.clip(l0 / denom, 0.0, 1.0)
            hit = xp + frac[:, None] * (xn - xp)
            nrm = self._b_nrm[ea, el]
            tan = self._b_tan[ea, el]
            inside = hit + self._b_delta[ea, el][:, None] * nrm
            t_rem[act] = t_rem[act] * (1.0 - frac)

            # 吸収 (リザーバ・真空)
            absorb = (btype == _B_RESERVOIR) | (btype == _B_VACUUM)
            remove[act[absorb]] = True
            if self._samples:
                self.outflow += float(np.count_nonzero(absorb)) * self.w

            # 拡散反射壁: 壁温 Maxwell 流束で内向きに再放出
            wall = btype == _B_WALL
            if np.any(wall):
                k = int(wall.sum())
                w_idx = act[wall]
                sig = np.sqrt(KB * self._b_temp[ea[wall], el[wall]] / self.m)
                vn = sig * np.sqrt(-2.0 * np.log(np.maximum(self.rng.random(k), 1e-300)))
                vt = self.rng.normal(0.0, 1.0, size=k) * sig
                vz = self.rng.normal(0.0, 1.0, size=k) * sig
                self.v[w_idx, 0] = vn * nrm[wall, 0] + vt * tan[wall, 0]
                self.v[w_idx, 1] = vn * nrm[wall, 1] + vt * tan[wall, 1]
                self.v[w_idx, 2] = vz

            # 鏡面反射 (対称境界): 法線速度成分を反転
            sym = btype == _B_SYMMETRY
            if np.any(sym):
                s_idx = act[sym]
                vn_c = self.v[s_idx, 0] * nrm[sym, 0] + self.v[s_idx, 1] * nrm[sym, 1]
                self.v[s_idx, 0] -= 2.0 * np.minimum(vn_c, 0.0) * nrm[sym, 0]
                self.v[s_idx, 1] -= 2.0 * np.minimum(vn_c, 0.0) * nrm[sym, 1]

            # 再放出/反射粒子: 交点内側から残余時間を完走させて再 walk
            cont = wall | sym
            c_idx = act[cont]
            if c_idx.size == 0:
                act = np.zeros(0, dtype=np.int64)
                break
            x_from[c_idx] = inside[cont]
            e_from[c_idx] = ea[cont]
            x_new[c_idx] = inside[cont] + t_rem[c_idx][:, None] * self.v[c_idx, :2]
            e2, a2, be2, bl2 = self._walk_chunked(ea[cont], x_new[c_idx])
            elem_new[c_idx] = e2
            b_elem[c_idx] = be2
            b_loc[c_idx] = bl2
            act = c_idx[a2]

        # レグ上限に達しても未解決の粒子 (稀): 直近の交点内側で停止させる
        if act.size:
            x_new[act] = x_from[act]
            elem_new[act] = e_from[act]

        # 固体 (誘電体) 要素へ入った粒子: 移動を取り消し、壁温の等方 Maxwell で再抽選
        if self._solid is not None:
            in_solid = ~remove & self._solid[elem_new]
            if np.any(in_solid):
                s_idx = np.nonzero(in_solid)[0]
                x_new[s_idx] = self.x[s_idx]
                elem_new[s_idx] = self.elem[s_idx]
                sigw = math.sqrt(KB * self.s.wall_temperature_k / self.m)
                self.v[s_idx] = self.rng.normal(0.0, sigw, size=(len(s_idx), 3))

        keep = ~remove
        self.x = x_new[keep]
        self.v = self.v[keep]
        self.elem = elem_new[keep]

    # ---- NTC 衝突 -------------------------------------------------------------

    def _collide(self) -> None:
        """NTC 法のセル内2体衝突 (VHS、COM 系等方散乱)。

        セルごとの候補対数 N_cand = ½ N(N−1) W (σc_r)_max Δt / V を端数持ち越しで
        抽選し、σ(c_r)c_r/(σc_r)_max で採択する。同一粒子が同ステップ内で複数対に
        選ばれた場合は後の衝突が上書きする近似 (候補数が少なければ影響は僅少)。
        """
        n = len(self.x)
        if n < 2:
            return
        counts = np.bincount(self.elem, minlength=len(self.tris))
        order = np.argsort(self.elem, kind="stable")
        starts = np.zeros(len(self.tris) + 1, dtype=np.int64)
        np.cumsum(counts, out=starts[1:])

        vol = self.vol  # xy: 面積 × 奥行き 1 m、rz: 2π r̄ A
        n_cand_f = (
            0.5 * counts * np.maximum(counts - 1, 0) * self.w
            * self._sigcr_max * self.dt / vol
            + self._coll_frac
        )
        n_cand = n_cand_f.astype(np.int64)
        self._coll_frac = n_cand_f - n_cand
        n_cand = np.where(counts >= 2, n_cand, 0)
        total = int(n_cand.sum())
        if total == 0:
            return

        cell = np.repeat(np.arange(len(self.tris)), n_cand)
        c_cnt = counts[cell]
        r1 = (self.rng.random(total) * c_cnt).astype(np.int64)
        r2 = (self.rng.random(total) * c_cnt).astype(np.int64)
        valid = r1 != r2
        cell, r1, r2 = cell[valid], r1[valid], r2[valid]
        i1 = order[starts[cell] + r1]
        i2 = order[starts[cell] + r2]

        g = self.v[i1] - self.v[i2]
        g_mag = np.sqrt(np.sum(g * g, axis=1))
        sig_cr = self._sigma(g_mag) * g_mag
        # (σc_r)_max の実測更新 (セルごとの最大値)
        np.maximum.at(self._sigcr_max, cell, sig_cr)
        accept = self.rng.random(len(cell)) * self._sigcr_max[cell] < sig_cr
        if not np.any(accept):
            return
        i1, i2 = i1[accept], i2[accept]
        g_mag = g_mag[accept]
        cell_a = cell[accept]

        # 同一粒子を含む複数対の一括更新はエネルギー保存を壊す (古い速度で計算した
        # 相手側更新が上書きされ、分散が湧いて数値加熱する)。各粒子について最初に
        # 現れた対のみ実行し、落とした対は候補として次ステップへ持ち越す
        pos = np.arange(len(i1))
        first = np.full(len(self.x), len(i1), dtype=np.int64)
        np.minimum.at(first, i1, pos)
        np.minimum.at(first, i2, pos)
        keep = (first[i1] == pos) & (first[i2] == pos)
        if not np.all(keep):
            np.add.at(self._coll_frac, cell_a[~keep], 1.0)
            i1, i2 = i1[keep], i2[keep]
            g_mag = g_mag[keep]
        if len(i1) == 0:
            return
        k = len(i1)
        # COM 系で等方散乱 (|g| 保存)
        cos_t = 1.0 - 2.0 * self.rng.random(k)
        sin_t = np.sqrt(np.maximum(1.0 - cos_t * cos_t, 0.0))
        phi = 2.0 * math.pi * self.rng.random(k)
        d_new = np.stack(
            [sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1
        )
        v_com = 0.5 * (self.v[i1] + self.v[i2])
        half_g = 0.5 * g_mag[:, None] * d_new
        self.v[i1] = v_com + half_g
        self.v[i2] = v_com - half_g

    # ---- 結果の平滑化 (隣接セル拡散、prompts/67) -------------------------------

    def _smooth_moments(
        self, acc_cnt: np.ndarray, acc_v: np.ndarray, acc_v2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """統計ノイズ低減のための隣接セル拡散 (n/t/u/p 導出前の生モーメントに適用)。

        導出後の n/T/u/p を別々に平滑化すると p = n kB T の整合が崩れて負温度等の
        人工値が出るため、蓄積モーメント (Σ個数・Σv・Σv²) の段階でまとめて平滑化する。
        セル密度的な量 q_i = acc_i / V_i に対し、1パスで
            q_i' = q_i + θ Σ_{j∈adj(i)} (W_ij/V_i)(q_j − q_i),  W_ij = min(V_i, V_j), θ=0.25
        を全成分 (個数・vx・vy・vz・v²) 一括で行う。W_ij は i, j について対称なので
        ペアごとの交換 (i の増分×V_i と j の増分×V_j) が厳密に相殺し、
        Σ_i V_i q_i (= Σ acc、質量・運動量・エネルギーに対応) は保存される。
        係数 1 − θ·Σ_j W_ij/V_i は W_ij ≤ V_i・隣接最大3枚から ≥ 1 − 0.25×3 = 0.25 > 0
        なので q_i' は非負の入力に対して常に非負 (凸結合、正値性保証)。
        固体 (誘電体) 要素にはガスが存在しないため交換対象から完全に除外する
        (自身が固体の行も、固体を隣接に持つ場合の当該隣接も、両方無効化する。
        固体セルの acc は元々0なので、除外しても保存性は損なわれない)。
        呼び出し側の蓄積配列そのものは書き換えず、コピー (計算結果) を返す。
        """
        passes = self.s.smoothing_passes
        if passes <= 0:
            return acc_cnt, acc_v, acc_v2

        adj = self.adjacency  # (n_elem, 3)、-1 = 境界
        has_nb = adj >= 0
        adj_safe = np.where(has_nb, adj, 0)  # 無効エントリはダミー参照 (後で重み0にする)

        valid = has_nb.copy()
        if self._solid is not None:
            valid &= ~self._solid[:, None]       # 自身が固体の行は交換しない
            valid &= ~self._solid[adj_safe]      # 固体隣接も除外

        vol = self.vol
        w = np.where(valid, np.minimum(vol[:, None], vol[adj_safe]), 0.0)  # (n,3)
        coef = w / vol[:, None]  # θ 倍前の拡散係数 (パス間で不変なので事前計算)
        theta = 0.25

        # 5成分 (個数・vx・vy・vz・v²) をまとめて (n_elem, 5) にしてベクトル化する
        q = np.empty((len(vol), 5))
        q[:, 0] = acc_cnt / vol
        q[:, 1:4] = acc_v / vol[:, None]
        q[:, 4] = acc_v2 / vol

        for _ in range(passes):
            qj = q[adj_safe]  # (n,3,5)
            diff = qj - q[:, None, :]
            delta = theta * np.einsum("nk,nkc->nc", coef, diff)
            q = q + delta

        return q[:, 0] * vol, q[:, 1:4] * vol[:, None], q[:, 4] * vol

    # ---- サンプリングと実行 ---------------------------------------------------

    def _sample(self) -> None:
        cnt = np.bincount(self.elem, minlength=len(self.tris)).astype(np.float64)
        self._acc_cnt += cnt
        for c in range(3):
            self._acc_v[:, c] += np.bincount(
                self.elem, weights=self.v[:, c], minlength=len(self.tris)
            )
        v2 = np.sum(self.v * self.v, axis=1)
        self._acc_v2 += np.bincount(self.elem, weights=v2, minlength=len(self.tris))
        self._samples += 1

    def step(self) -> None:
        self._inject()
        self._move()
        self._collide()

    def _thin_positions(self) -> np.ndarray:
        """粒子位置 (self.x) を最大 MAX_CALLBACK_PARTICLES 点に間引いて返す (ライブ表示用)。

        pic.py の frame 粒子間引きと同じ考え方: 等間隔サンプル (linspace) を使う。
        粒子順序に物理的意味はないが、injection で新粒子が末尾に追加されるため、
        先頭からの単純な stride 抽出だと新しい粒子ばかり削られて偏る。
        self.x は (x,y) または (z,r) の描画平面2次元座標そのもの (rz でも同じ)。
        """
        n = len(self.x)
        if n <= MAX_CALLBACK_PARTICLES:
            return self.x
        idx = np.linspace(0, n - 1, MAX_CALLBACK_PARTICLES).astype(np.int64)
        return self.x[idx]

    def prepare_continue(self, n_steps: int, avg_steps: int | None = None) -> None:
        """完了/停止後の状態から追加実行の準備をする (pic.py の prepare_continue と同じ設計)。

        維持するもの: 粒子状態 (x, v, elem)・rng・_coll_frac (NTC 候補数の端数)・
        _sigcr_max・_res_edges の frac (流入端数)・通算ステップ数 (step_count)。
        これらを一切リセットしないため、「N+M ステップ連続実行」と「N ステップ
        完了後に continue して M ステップ」は粒子状態がビット単位で一致する。
        run() 内の平均区間 (avg_start) はその回の self.s.n_steps/avg_steps だけから
        決まるので、両ケースで平均区間が同じ末尾ステップ範囲を指すように
        n_steps/avg_steps を選べば、平均結果も一致する。

        リセットするもの: サンプリング蓄積 (_acc_cnt/_acc_v/_acc_v2/_samples) と
        inflow/outflow。run() は平均区間に入った瞬間 (i == avg_start) にも同じ
        リセットを行うが、ここで先に明示的にクリアしておかないと、新区間が
        avg_start に到達する前に should_stop() で打ち切られた場合、前回実行分の
        古い蓄積が「平均区間に入れた」と誤判定されて (_samples != 0 のため
        run() 末尾のエラー検出をすり抜けて) 古い結果がそのまま返ってしまう。
        """
        self.s.n_steps = int(n_steps)
        if avg_steps is not None:
            self.s.avg_steps = int(avg_steps)
        self._acc_cnt[:] = 0.0
        self._acc_v[:] = 0.0
        self._acc_v2[:] = 0.0
        self._samples = 0
        self.inflow = 0.0
        self.outflow = 0.0

    def run(self, callback=None, should_stop=None) -> DsmcResult:
        """n_steps 進め、最終 avg_steps の時間平均から DsmcResult を作る。

        callback(step, n_particles, positions) を 100 ステップごとに呼ぶ (進捗表示用)。
        step はこの run() 呼び出し内の区間相対ステップ (1..n_steps)。continue
        (prepare_continue、prompts/74) で複数回 run() を呼んでも進捗表示は毎回
        0 から始まる方が分かりやすく、PIC の継続実行で通算ステップとの整合を
        取るためにフロント側で offset 計算が必要になった経緯 (prompts 直近の fix)
        を DSMC では最初から避けられるため、あえて通算値にしない。
        n_particles は間引き前の実シミュレーション粒子数、positions は間引き後の
        粒子座標 (≤2000点、(k,2) ndarray)。should_stop() が True を返したら中断する
        (WS の stop コマンド用)。平均区間に入る前に中断された場合は結果が無いためエラーを送出する。
        """
        n_steps = self.s.n_steps
        avg_start = max(0, n_steps - self.s.avg_steps)
        for i in range(n_steps):
            if should_stop is not None and should_stop():
                break
            if i == avg_start:
                self._samples = 0  # 以降のステップでサンプリング
                self._acc_cnt[:] = 0.0
                self._acc_v[:] = 0.0
                self._acc_v2[:] = 0.0
                self.inflow = 0.0
                self.outflow = 0.0
            self.step()
            self.step_count += 1
            if i >= avg_start:
                self._sample()
            if callback is not None and (i + 1) % 100 == 0:
                callback(i + 1, len(self.x), self._thin_positions())
        if self._samples == 0:
            raise ValueError(
                "平均区間に入る前に停止したため、ガス場の結果がありません "
                "(avg_steps を長くするか、停止せず完走させてください)"
            )

        # 結果組み立て直前に (蓄積配列そのものは変更せず) 平滑化する。smoothing_passes=0
        # なら acc_* をそのまま返すので既存動作と完全一致する
        acc_cnt, acc_v, acc_v2 = self._smooth_moments(
            self._acc_cnt, self._acc_v, self._acc_v2
        )
        cnt = np.maximum(acc_cnt, 1e-300)
        n_avg = acc_cnt * self.w / (self._samples * self.vol)
        u_mean3 = acc_v / cnt[:, None]
        v2_mean = acc_v2 / cnt
        # T = m/(3k)·(⟨|v|²⟩ − |⟨v⟩|²) (3成分)
        t_avg = np.maximum(
            self.m / (3.0 * KB) * (v2_mean - np.sum(u_mean3 * u_mean3, axis=1)), 0.0
        )
        # サンプルが無いセル (固体・枯渇) は 0 に落とす (平滑化後の値で判定する:
        # 平滑化は隣接から統計を分け合わせる操作なので、平滑化後に実質0のセルのみ除く)
        empty = acc_cnt <= 0.0
        t_avg[empty] = 0.0
        u_mean3[empty] = 0.0
        return DsmcResult(
            n=n_avg,
            t=t_avg,
            u=u_mean3[:, :2],
            p=n_avg * KB * t_avg,
            n_particles=len(self.x),
            macro_weight=self.w,
            dt=self.dt,
            inflow=self.inflow,
            outflow=self.outflow,
        )
