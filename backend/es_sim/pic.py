"""FEM-PIC (フェーズ3)。仕様書 §9 参照。

サイクル (PicSimulation.step):
  1. 電荷堆積: P1 形状関数 (重心座標) の重みで粒子電荷を節点荷重ベクトルへ散布
  2. ポアソン求解: 剛性行列は固定なので splu を初回のみ実行し、右辺のみ更新。
     Dirichlet 値は V(t) = V_dc + A sin(2πft + φ) で毎ステップ更新 (RF 電極対応)
  3. E 補間 (P1 なので所属要素の一定値) → 4. リープフロッグでプッシュ →
  5. walk 更新・境界吸収 (壁到達粒子は除去、種別ごとに集計) → 6. 注入

- 粒子種は常に electron / ion の2種を管理する
- リープフロッグの速度は半整数ステップに置く。装荷時・注入時に初期半ステップ
  後退キック v(-dt/2) = v(0) - (q/m) E(x) dt/2 を適用する
- 診断は毎ステップ記録。運動エネルギーは v(n-1/2)·v(n+1/2) の時刻中心化で評価し、
  場のエネルギー (整数ステップ) と整合させる
- 安定性チェック: 開始時に ωpe·dt とセルサイズ/デバイ長を確認して警告文字列を返す
- ホットループは numpy ベクトル化 (粒子 for ループなし)。walk 探索・隣接配列・
  重心座標係数は particles.py の実装を再利用する
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from .fem import EPS0, _material_arrays, _radial_index, assemble
from .fn import build_fn_surface, distribute_particles, fn_segment_currents
from .mcc import GasField, MccModel
from .meshing import generate_mesh
from .particles import (
    ME,
    MP,
    QE,
    _adjacency,
    _barycentric_coeffs,
    _boris_matrix,
    _build_boundary_tables,
    _init_particles,
    _locate_initial,
    _pack_coeffs,
    _solid_elements,
    _walk_step,
    b_vector,
)
from .schema import PicSettings, Project, VoltageWaveform, rf_components

# フレーム送出時の種ごとの最大粒子数 (間引き)
MAX_FRAME_PARTICLES = 2000

# IEDF/IADF コレクタのサンプル上限 (超過分は count/total_weight のみ計上、prompts/30)
COLLECTOR_MAX_SAMPLES = 50000

# walk 並列実行用の共有ワーカースレッド (遅延生成、プロセスで1本)
_WALK_POOL: ThreadPoolExecutor | None = None

# 軸上 (r=0) のゼロ割ガード (particles.py の trace と同じ値)
_R_TINY = 1e-30

# FN 電界放出の1ステップあたりの放出マクロ電子数の上限 (prompts/53)。
# 強電界 + 小さい macro_weight では I·dt/(e·w) が天文学的な数になり得るため、
# 超過時はマクロ重みを引き上げて粒子数をこの上限に抑える (総電荷は保存)
FN_MAX_MACROS_PER_STEP = 10000


def _walk_pool() -> ThreadPoolExecutor:
    global _WALK_POOL
    if _WALK_POOL is None:
        _WALK_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pic-walk")
    return _WALK_POOL


def _eval_waveform(phase, v, freq_hz: float, t):
    """CSV インポート波形 V_wf(t) の評価 (prompts/73)。単体テストから直接呼べるよう
    モジュール関数として切り出す (PicSimulation の初期化なしで検証できる)。

    phase は [0, 1) に正規化済みの位相 (昇順)、v は対応する電圧。実際の周期は
    freq_hz で決まり、時刻 t の位相は frac(t·freq_hz) (0〜1 を繰り返す)。
    末尾サンプルと先頭サンプルの間 (位相 phase[-1] 〜 1) が非連続にならないよう、
    phase 配列の末尾に仮想的に (phase[0]+1, v[0]) を足してから線形補間する
    (= 1周期ループの折返しを連続にするため)。
    """
    phase = np.asarray(phase, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    frac = np.mod(np.asarray(t, dtype=np.float64) * freq_hz, 1.0)
    phase_ext = np.concatenate([phase, [phase[0] + 1.0]])
    v_ext = np.concatenate([v, [v[0]]])
    return np.interp(frac, phase_ext, v_ext)


@dataclass
class PicSpecies:
    """マクロ粒子種。状態はすべて numpy 配列で保持する。"""

    name: str
    q: float               # 電荷 [C]
    m: float               # 質量 [kg]
    x: np.ndarray          # (n, 2) 位置 [m]
    v: np.ndarray          # (n, 3) 速度 [m/s] (2d3v。E は vx, vy のみに作用。半整数ステップ)
    w: np.ndarray          # (n,) マクロ重み (実粒子数/マクロ粒子)
    elem: np.ndarray       # (n,) 所属要素番号
    mobile: bool = True    # False ならプッシュしない (immobile_ions)
    wall_absorbed: int = 0  # 壁吸収の累計 (マクロ粒子数)
    # (n, 3) 所属要素での重心座標キャッシュ (walk が計算した値の再利用)。
    # None または長さ不一致なら電荷堆積時に再計算する。x を差し替えたら None にすること
    bary: np.ndarray | None = None
    # (n, 3) 所属要素の節点番号キャッシュ (tris[elem] の再利用)。同上のルール
    nidx: np.ndarray | None = None


class PicSimulation:
    """FEM-PIC シミュレーション本体。

    __init__ でメッシュ生成・FEM 行列組み立て・K_ff の LU 前分解 (1回のみ) を行い、
    step() ごとに右辺のみ更新して解く。
    """

    def __init__(self, project: Project, gas_field: GasField | None = None):
        if project.pic is None:
            raise ValueError("project.pic が指定されていません")
        self.project = project
        self.pic: PicSettings = project.pic
        # 非一様背景ガス場 (prompts/54)。DSMC の定常解などを MCC が参照する
        self.gas_field = gas_field
        # 軸対称モード (prompts/47)。rz: 径方向 = y、rz_x0: 径方向 = x。
        # マクロ粒子は「リング」(周方向に一様な実粒子 w 個の輪) として扱い、
        # v[:, 2] は vz の代わりに vθ (周方向速度) を表す
        self.ridx = _radial_index(project.coord)
        self.rz = self.ridx is not None

        # ---- メッシュ・幾何前処理 (particles.py の実装を再利用) --------------
        self.mesh = generate_mesh(project)
        mesh = self.mesh
        self.n_nodes = len(mesh.nodes)
        self.tris = mesh.triangles
        self.coeffs = _barycentric_coeffs(mesh.nodes, self.tris)  # (a, b, c, det)
        self._coeffs_packed = _pack_coeffs(self.coeffs)  # walk 用 (M,10) 詰め込み係数
        self.adjacency = _adjacency(self.tris)
        det = self.coeffs[3]
        self.area = 0.5 * np.abs(det)          # (M,) 要素面積
        self.eps_elem, _ = _material_arrays(project, mesh)  # (M,) 要素ごとの ε
        # 要素「体積」: xy = 面積 (奥行き1m換算)、rz = 2π r̄ A (周方向積分)。
        # xy では self.area と同一オブジェクトなので既存経路はビット単位で不変
        if self.rz:
            self._r_nodes = mesh.nodes[self.tris][:, :, self.ridx]  # (M, 3) 頂点の r
            self.r_bar = self._r_nodes.mean(axis=1)                 # (M,) 要素重心半径
            self.elem_vol = 2.0 * np.pi * self.r_bar * self.area
        else:
            self.elem_vol = self.area

        # 周期境界の正準化写像 (スレーブ→マスター)。電荷堆積・密度積算・求解は
        # 正準節点番号 (tris_dep) を使い、粒子 walk・E 補間は元の tris を使う
        self.canon = mesh.periodic_map
        self.tris_dep = self.canon[self.tris] if self.canon is not None else self.tris

        # 誘電体 (固体) 要素のマスク: 粒子は侵入できず表面で吸収する (prompts/24)。
        # 場の計算は変更しない (εr 付き要素としてメッシュに残る)。無ければ None
        self._solid_elem = _solid_elements(project, mesh)

        # 誘電体 SEE (prompts/38): 固体要素ごとの γ (see_gamma > 0 の dielectric)。
        # 該当領域が無ければ None (従来経路と完全一致)
        self._solid_gamma: np.ndarray | None = None
        if self._solid_elem is not None:
            gam_elem = np.zeros(len(self.tris))
            has_gamma = False
            for i, region in enumerate(project.geometry.regions):
                if region.type == "dielectric" and region.see_gamma > 0.0:
                    gam_elem[mesh.tri_region == i] = region.see_gamma
                    has_gamma = True
            if has_gamma:
                self._solid_gamma = gam_elem

        # 誘電体の表面電荷 (prompts/25): 吸収された粒子の電荷 w·q [C/m] を節点へ
        # P1 射影で蓄積し、毎ステップのポアソン右辺へ恒常的に加える。
        # 実行開始時は 0 で、停止までラン内で持続する
        self.q_surf = np.zeros(self.n_nodes)

        # ---- FEM 行列: K_ff を splu で1回だけ前分解 --------------------------
        k, self.f_static = assemble(project, mesh)  # 静的 charge 領域の右辺を含む
        items = sorted(mesh.dirichlet.items())      # 節点順に固定して決定的に
        self.fixed = np.array([i for i, _ in items], dtype=np.int64)
        self.v_dc = np.array([v for _, v in items], dtype=np.float64)
        # RF 成分 (デュアル周波数対応、prompts/49): (n_fixed, K) 行列。
        # K は全電極の最大成分数 (成分の無い列はゼロパディング = 振幅0で寄与なし)。
        # K=1 (従来の単一周波数) では sum(axis=1) が単一要素の恒等なので数値不変
        rf = mesh.dirichlet_rf
        kmax = max((len(c) for c in rf.values()), default=1)
        kmax = max(kmax, 1)
        n_fixed = len(items)
        self.rf_amp = np.zeros((n_fixed, kmax))
        self.rf_omega = np.zeros((n_fixed, kmax))
        self.rf_phase = np.zeros((n_fixed, kmax))
        for row, (i, _) in enumerate(items):
            for kc, (amp, freq, ph) in enumerate(rf.get(i, ())):
                self.rf_amp[row, kc] = amp
                self.rf_omega[row, kc] = 2.0 * math.pi * freq
                self.rf_phase[row, kc] = math.radians(ph)
        # CSV インポート波形 (prompts/73): 種類数が少ない想定なので、電極ごとに
        # 配列を持つのではなく「波形リスト (self._waveforms) + 節点→波形インデックス
        # (self._wf_index、-1=波形なし)」で持つ。評価時 (_dirichlet_values) は
        # 波形ごとに np.interp を1回だけ呼び、インデックスで各節点へ展開する
        wf_map = mesh.dirichlet_waveform
        self._waveforms: list[VoltageWaveform] = []
        self._wf_index = np.full(n_fixed, -1, dtype=np.int64)
        wf_id_to_idx: dict[int, int] = {}
        for row, (i, _) in enumerate(items):
            wf = wf_map.get(i)
            if wf is None:
                continue
            key = id(wf)
            idx = wf_id_to_idx.get(key)
            if idx is None:
                idx = len(self._waveforms)
                wf_id_to_idx[key] = idx
                self._waveforms.append(wf)
            self._wf_index[row] = idx
        exclude = self.fixed
        if self.canon is not None:
            # 周期スレーブ節点は自由度から除外する (剛性行列の行がマスターへ寄っている)
            slaves = np.nonzero(self.canon != np.arange(self.n_nodes))[0]
            exclude = np.union1d(self.fixed, slaves)
        self.free = np.setdiff1d(np.arange(self.n_nodes), exclude)
        if len(self.free) == 0:
            raise ValueError("自由節点がありません (全節点が Dirichlet)")
        self.k_fd = k[self.free][:, self.fixed].tocsr()
        self.lu = spla.splu(k[self.free][:, self.free].tocsc())

        # ---- dt とプラズマパラメータ -----------------------------------------
        ip = self.pic.initial_plasma
        amu = ip.ion_mass_amu if ip is not None else 40.0
        self.m_ion = amu * MP
        wpe = 0.0
        if ip is not None:
            wpe = math.sqrt(ip.density * QE**2 / (EPS0 * ME))
        if self.pic.dt is not None:
            self.dt = float(self.pic.dt)
        elif wpe > 0.0:
            self.dt = 0.1 / wpe  # 既定: ωpe·dt = 0.1
        else:
            raise ValueError("pic.dt を指定してください (初期密度が無いため自動決定できません)")

        # ---- 安定性チェック (警告のみ、実行は継続) ---------------------------
        self.warnings: list[str] = []
        if wpe > 0.0:
            if wpe * self.dt > 0.3:
                self.warnings.append(
                    f"ωpe·dt = {wpe * self.dt:.3g} > 0.3: 時間刻みが粗すぎます (数値不安定の恐れ)"
                )
            if ip.te_ev > 0.0:
                lam_d = math.sqrt(EPS0 * ip.te_ev * QE / (ip.density * QE**2))
                p = mesh.nodes[self.tris]
                edges = (
                    np.linalg.norm(p[:, 0] - p[:, 1], axis=1)
                    + np.linalg.norm(p[:, 1] - p[:, 2], axis=1)
                    + np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
                ) / 3.0
                h_mean = float(edges.mean())
                if h_mean > 3.0 * lam_d:
                    self.warnings.append(
                        f"平均セルサイズ {h_mean:.3g} m > 3×デバイ長 {lam_d:.3g} m: "
                        "メッシュがデバイ長を解像していません"
                    )

        # ---- RF 1周期の位相分解 (prompts/28) ----------------------------------
        # RF 周波数は boundaries / conductor 領域の voltage_rf から最初に見つかった
        # ものを使う。無ければ (または phase_bins=0 なら) cycle 機能は無効
        self._cycle_freq = self._find_rf_freq()
        self._cycle_bins = int(self.pic.phase_bins)
        self._cycle_enabled = self._cycle_freq is not None and self._cycle_bins > 0
        self._cycle_period = 1.0 / self._cycle_freq if self._cycle_enabled else 0.0
        if self._cycle_enabled:
            # 平均区間が RF 1周期より短いとステップ数 0 の位相ビンが生じるため警告
            avg = (
                self.pic.avg_steps
                if self.pic.avg_steps is not None
                else max(1, self.pic.n_steps // 4)
            )
            avg = min(avg, self.pic.n_steps)
            if avg * self.dt < self._cycle_period:
                self.warnings.append(
                    f"位相分解平均の区間 {avg * self.dt:.3g} s が RF 1周期 "
                    f"{self._cycle_period:.3g} s より短いため、空の位相ビンが生じます"
                )

        # ---- 初期プラズマ装荷 -------------------------------------------------
        # 電子・イオンを同一位置に装荷して初期の厳密な電気的中性を保つ (quiet start)
        self.species: dict[str, PicSpecies] = {}
        if ip is not None:
            rng = np.random.default_rng(ip.seed)
            n_macro = self.pic.n_macro
            # 装荷可能体積 = 誘電体 (固体) 要素を除いた体積
            # (xy: 面積 = 奥行き1m換算、rz: 2πr̄A。elem_vol は xy では area と同一)
            if self._solid_elem is None:
                area_load = float(self.elem_vol.sum())
            else:
                area_load = float(self.elem_vol[~self._solid_elem].sum())
            w0 = ip.density * area_load / n_macro  # マクロ重み (実粒子数/マクロ。rz はリング)
            x0, elem0 = self._sample_uniform(rng, n_macro)
            for name, q, m, t_ev, mobile in (
                ("electron", -QE, ME, ip.te_ev, True),
                ("ion", QE, self.m_ion, ip.ti_ev, not ip.immobile_ions),
            ):
                sigma = math.sqrt(t_ev * QE / m) if t_ev > 0.0 else 0.0
                # 2d3v: Maxwell 速度は3成分で抽選する
                v = (
                    rng.normal(0.0, sigma, size=(n_macro, 3))
                    if sigma > 0.0
                    else np.zeros((n_macro, 3))
                )
                self.species[name] = PicSpecies(
                    name, q, m, x0.copy(), v, np.full(n_macro, w0), elem0.copy(), mobile
                )
        else:
            for name, q, m in (("electron", -QE, ME), ("ion", QE, self.m_ion)):
                self.species[name] = PicSpecies(
                    name, q, m,
                    np.zeros((0, 2)), np.zeros((0, 3)),
                    np.zeros(0), np.zeros(0, dtype=np.int64),
                )

        # ---- 注入の前計算 (エミッタ形状は固定なので所属要素をキャッシュ) ------
        inj = self.pic.injection
        if inj is not None:
            sp = self.species[inj.species]
            pos, vel = _init_particles(inj.emitter, sp.m)
            self._inj_pos = pos
            self._inj_vel_base = vel  # mono の場合の決定的な速度
            self._inj_elem = _locate_initial(self.coeffs, pos)
            # 注入位置・所属要素は固定なので重心座標も前計算できる
            self._inj_bary = self._bary_of(pos, self._inj_elem)
            self._inj_rng = np.random.default_rng(inj.emitter.seed)
            # 毎ステップの実電荷 I·dt を n 個のマクロ粒子へ等分
            self._inj_w = inj.current_a_per_m * self.dt / (QE * inj.emitter.n)

        # ---- FN 電界放出 (prompts/46) ----------------------------------------
        # fn=null なら無効 (従来動作と完全一致)。毎ステップの表面電界から
        # I·dt 分の電子を放出面 (電極表面エッジ列) から注入する
        fn = self.pic.fn
        self._fn_surf = None
        self.fn_events = 0  # 放出したマクロ電子の累計
        if fn is not None:
            self._fn_surf = build_fn_surface(project, self.mesh, self.adjacency, fn)
        if self._fn_surf is not None:
            if fn.macro_weight is not None:
                self._fn_w = float(fn.macro_weight)
            elif ip is not None:
                self._fn_w = w0  # 初期プラズマと同じマクロ重み
            else:
                raise ValueError(
                    "pic.fn.macro_weight を指定してください (初期プラズマが無いため自動決定できません)"
                )
            # セグメントごとの端数マクロ粒子の持ち越し (毎ステップの床関数の余り)
            self._fn_frac = np.zeros(len(self._fn_surf.elem))
            self._fn_rng = np.random.default_rng(fn.seed)
            self._fn_speed = (
                math.sqrt(2.0 * fn.init_energy_ev * QE / ME)
                if fn.init_energy_ev > 0.0
                else 0.0
            )

        # ---- イオンサブサイクリング (prompts/50) ------------------------------
        # イオンを _sub ステップに1回、_sub·dt で押す。休止ステップ中はイオンが
        # 動かないため電荷堆積ベクトルをキャッシュして再利用する (イオン配列が
        # 変わる箇所 — プッシュ・電離生成・イオン注入 — で必ず無効化すること)
        self._sub = max(1, int(self.pic.ion_subcycle))
        self._f_ion_cache: np.ndarray | None = None
        self._last_ke_i = 0.0  # 休止ステップの KE 診断用 (直近プッシュ時の値を流用)

        # ---- 粒子チャンク並列 (prompts/50) ------------------------------------
        # walk 探索を粒子チャンクに分けてスレッド並列化する。粒子ごとの walk は
        # 独立で numpy ufunc は GIL を解放するため実効的。チャンク分割しても
        # 各粒子の演算は不変なので、結果は逐次実行とビット単位で一致する
        self._nthreads = max(1, int(self.pic.threads))
        self._chunk_pool = (
            ThreadPoolExecutor(max_workers=self._nthreads, thread_name_prefix="pic-chunk")
            if self._nthreads > 1
            else None
        )

        # ---- 一様磁場 (prompts/51) --------------------------------------------
        # Boris 回転行列を種ごとに前計算する (イオンサブサイクルで dt が異なるため)。
        # b_field なし (または全成分 0) では None = 従来経路と完全一致。
        # 初期/注入の半ステップ後退キックは E のみの近似 (回転は初回ステップから)
        self._b = b_vector(project)
        self._boris_rt: dict[str, np.ndarray] = {}
        if self._b is not None:
            for name, q_s, m_s in (("electron", -QE, ME), ("ion", QE, self.m_ion)):
                dt_s = self.dt * self._sub if (name == "ion" and self._sub > 1) else self.dt
                self._boris_rt[name] = _boris_matrix(q_s, m_s, dt_s, self._b).T
            wc_e = QE * float(np.linalg.norm(self._b)) / ME
            if wc_e * self.dt > 0.3:
                self.warnings.append(
                    f"ωce·dt = {wc_e * self.dt:.3g} > 0.3: 電子サイクロトロン運動を"
                    "解像できていません (dt を小さくしてください)"
                )

        # ---- MCC (モンテカルロ衝突、prompts/19) ------------------------------
        # mcc=null なら無効 (従来の無衝突動作と完全一致)。
        # gas_field あり (prompts/54) なら要素数の整合を検査して MCC へ渡す
        if gas_field is not None and len(gas_field.n_g) != len(self.tris):
            raise ValueError(
                f"ガス場の要素数 ({len(gas_field.n_g)}) がメッシュの要素数 "
                f"({len(self.tris)}) と一致しません (DSMC を再実行してください)"
            )
        self.mcc = (
            MccModel(self.pic.mcc, self.m_ion, gas_field)
            if self.pic.mcc is not None
            else None
        )
        if self.mcc is not None:
            # 衝突頻度の健全性チェック (prompts/57)。非一様ガス場では ν_max が
            # 最大密度で評価されるため、密度スパイクや高圧領域があると毎ステップ
            # ほぼ全粒子が衝突候補になり、dt が実効的に粗すぎる状態になり得る
            # 代表上限 200 eV (シース加速の典型上限) までの ν_max で評価する
            # (実行時の適応 ν_max と同じテーブル。全域 max だと keV 域の断面積で
            # 過大評価になるため)
            for label, table in (
                ("電子", self.mcc._nu_e_table),
                ("イオン", self.mcc._nu_i_table),
            ):
                numax = MccModel._numax_upto(table, 200.0)
                if numax > 0.0:
                    p_coll = 1.0 - math.exp(-numax * self.dt)
                    if p_coll > 0.5:
                        self.warnings.append(
                            f"{label}の1ステップ衝突候補率が {p_coll:.2f} > 0.5 "
                            "(〜200 eV 域): dt が衝突頻度に対して粗すぎます "
                            "(dt を小さくするか、ガス場の最大密度を確認してください)"
                        )
            if gas_field is not None:
                n_arr = np.asarray(gas_field.n_g, dtype=np.float64)
                pos_n = n_arr[n_arr > 0.0]
                if len(pos_n):
                    med = float(np.median(pos_n))
                    if med > 0.0 and float(n_arr.max()) > 100.0 * med:
                        self.warnings.append(
                            f"ガス場の最大密度 {n_arr.max():.3g} m^-3 が中央値 {med:.3g} の"
                            "100倍を超えています (DSMC の統計スパイクの可能性。"
                            "n_particles や avg_steps を増やして再計算を検討してください)"
                        )

        # ---- SEE (二次電子放出) の境界エッジ属性表 ----------------------------
        mcc_seed = self.pic.mcc.seed if self.pic.mcc is not None else 0
        self._see_rng = np.random.default_rng(mcc_seed + 12345)
        self._see_speed = math.sqrt(2.0 * self.pic.see_energy_ev * QE / ME)
        self._build_see_edges()

        # ---- 鏡面反射 (reflect_edges + symmetry)・周期エッジの前計算 -----------
        self._build_boundary_edges()

        # ---- 節点密度アキュムレータ (enable_density_accum で有効化) -----------
        self._accum_start: int | None = None
        self._accum_count = 0
        self._accum: dict[str, np.ndarray] = {}
        # 時間平均フィールドのアキュムレータ (prompts/26、enable_density_accum で確保)
        self._accum_phi: np.ndarray | None = None
        self._accum_e: np.ndarray | None = None
        self._accum_ke_e: np.ndarray | None = None
        self._accum_ion: np.ndarray | None = None
        # run_batch 完了時の時間平均フィールド (averaged_fields() の結果)
        self.fields: dict | None = None
        # 位相分解アキュムレータ (prompts/28、enable_density_accum で確保)
        self._cycle_phi: np.ndarray | None = None    # (bins, N)
        self._cycle_ne: np.ndarray | None = None     # (bins, N) 節点重み積算
        self._cycle_ni: np.ndarray | None = None
        self._cycle_count: np.ndarray | None = None  # (bins,) ビンごとのステップ数
        # 位相分解の追加フィールド (prompts/52): E ベクトル (要素)・電子運動エネルギー・
        # 電離イベント (いずれも位相ビンごとの積算)
        self._cycle_e: np.ndarray | None = None      # (bins, M, 2)
        self._cycle_ke: np.ndarray | None = None     # (bins, N)
        self._cycle_ion: np.ndarray | None = None    # (bins, N)
        # 最後の1周期の粒子スナップショット (run_batch で確保)
        self._cycle_particles: dict[str, list] | None = None
        self._snap_t_start = math.inf
        # run_batch 完了時の位相分解データ (cycle_data() の結果)
        self.cycle: dict | None = None

        # ---- IEDF/IADF コレクタ (prompts/30、複数対応 prompts/36) -------------
        # 平均区間中に吸収されたイオンのうち、各コレクタ線分から距離 tol 以内かつ
        # 線分区間内で吸収されたものをコレクタごとに記録する。
        # 旧単数形 collector はスキーマ側で collectors へ正規化済み (空なら無効)
        self.collector_results: list[dict] | None = None
        self.collector_result: dict | None = None  # 先頭コレクタのエイリアス (後方互換)
        self._collectors_st: list[dict] = []  # コレクタごとの前計算 + 記録ストレージ
        for col in self.pic.collectors:
            p1 = np.asarray(col.p1, dtype=np.float64)
            p2 = np.asarray(col.p2, dtype=np.float64)
            seg = p2 - p1
            seg_len = float(np.hypot(seg[0], seg[1]))
            if seg_len <= 0.0:
                raise ValueError("collector の p1 と p2 が同一点です")
            tan = seg / seg_len  # 接線 (p1→p2)
            self._collectors_st.append(
                {
                    "p1": p1,
                    "len": seg_len,
                    "tan": tan,
                    "nrm": np.array([-tan[1], tan[0]]),  # 法線
                    "tol": col.tol if col.tol is not None else project.mesh.size,
                    # チャンク追記 (numpy 配列のリスト) で高速に記録する
                    "e": [], "a": [], "w": [],
                    "count": 0,     # 記録したマクロイオン数 (総数)
                    "weight": 0.0,  # 総実イオン数 [1/m]
                    "samples": 0,   # 保存済みサンプル数 (上限はコレクタごとに適用)
                }
            )
        # 節点集中体積 (averaged_density の規格化に使う)。
        # xy: Σ隣接要素面積/3 (従来どおり)。rz: Σ 2π∫L_i r dA = Σ 2π(A/12)(2r_i+r_j+r_k)
        # (線形 r の厳密積分。fem.assemble の右辺と同じ公式)。
        # 周期境界では正準節点番号で積むため、スレーブ節点の値は 0 になる
        if self.rz:
            node_w = (
                2.0 * np.pi * (self.area / 12.0)[:, None]
                * (self._r_nodes + 3.0 * self.r_bar[:, None])
            )
            self._node_area = np.bincount(
                self.tris_dep.ravel(), weights=node_w.ravel(), minlength=self.n_nodes
            )
        else:
            self._node_area = np.bincount(
                self.tris_dep.ravel(),
                weights=np.repeat(self.area / 3.0, 3),
                minlength=self.n_nodes,
            )

        # ---- 診断・時刻 -------------------------------------------------------
        self.t = 0.0
        self.step_count = 0
        # 累計カウンタ (既存フィールドは変更せず追加のみ: フロントの後方互換)
        self.coll_e = 0      # 電子衝突の累計
        self.ion_events = 0  # 電離の累計
        self.see_events = 0  # SEE 発生の累計
        self.history: dict[str, list[float]] = {
            k: []
            for k in (
                "t", "ke_e", "ke_i", "fe", "n_e", "n_i",
                "wall_e", "wall_i", "phi_min", "phi_max",
                "coll_e", "ion_events", "see_events", "surf_q",
                "fn_i", "fn_events",
            )
        }
        self._f_immobile: dict[str, np.ndarray] = {}  # 不動種の堆積キャッシュ

        # ---- 位相別プロファイル計測 (prompts/75) -------------------------------
        # 大規模化に向けた高速化①の土台として、1ステップ内の主要フェーズの
        # 累積時間を常時計測する (ON/OFF切替は無し)。time.perf_counter の呼び出しは
        # 1ステップあたり数回程度なのでオーバーヘッドは無視できる。
        # done メッセージ (server.py) で返し、continue では区間分のみ返すよう
        # prepare_continue でリセットする
        self.timing: dict[str, float] = {
            "solve": 0.0,       # ポアソン求解 (前分解済み LU の rhs 更新+solve)
            "gather_push": 0.0,  # E 補間 + リープフロッグ押し出し
            "walk": 0.0,        # walk 探索・境界吸収・反射・周期・SEE
            "deposit": 0.0,     # 電荷堆積 (節点荷重ベクトルへの散布)
            "mcc": 0.0,         # MCC 衝突 (電子・イオン)
            "other": 0.0,       # 注入・FN放出・時間平均積算・診断記録など
        }

        # ---- 初期半ステップ後退キック (t=0 の場で v を -dt/2 へ) --------------
        # E は vx, vy のみに作用する (vz は不変)
        phi0 = self._solve_phi(self._deposit(), 0.0)
        ex, ey = self._e_field(phi0)
        for sp in self.species.values():
            if sp.mobile and len(sp.x):
                # サブサイクル時のイオンは実効時間刻み _sub·dt で後退させる
                dt_sp = self.dt * self._sub if (sp.name == "ion" and self._sub > 1) else self.dt
                e_at = np.stack([ex[sp.elem], ey[sp.elem]], axis=1)
                if self.rz:
                    # 軸対称: 遠心力項も含めて半ステップ後退させる
                    a0 = (sp.q / sp.m) * e_at
                    r0 = np.maximum(sp.x[:, self.ridx], _R_TINY)
                    a0[:, self.ridx] += sp.v[:, 2] ** 2 / r0
                    sp.v[:, :2] -= 0.5 * dt_sp * a0
                else:
                    sp.v[:, :2] -= 0.5 * dt_sp * (sp.q / sp.m) * e_at

    # ---- 内部処理 -----------------------------------------------------------

    def _find_rf_freq(self) -> float | None:
        """boundaries / conductor 領域の voltage_rf / voltage_waveform から
        位相分解の基本周波数を返す。

        デュアル周波数 (prompts/49) では全成分の最小周波数 (= 基本波。低周波の
        1周期に高周波の複数サイクルが収まる) を使う。CSV 波形 (prompts/73) の
        freq_hz も同じ集合に加える (波形も1周期ループなので基本波として扱える)。
        無ければ None (cycle 無効)。単一周波数のみの従来構成では従来と同じ値になる。
        """
        freqs: list[float] = []
        for bc in self.project.geometry.boundaries:
            freqs.extend(c.freq_hz for c in rf_components(bc.voltage_rf))
            if bc.voltage_waveform is not None:
                freqs.append(bc.voltage_waveform.freq_hz)
        for region in self.project.geometry.regions:
            if region.type == "conductor":
                freqs.extend(c.freq_hz for c in rf_components(region.voltage_rf))
                if region.voltage_waveform is not None:
                    freqs.append(region.voltage_waveform.freq_hz)
        return min(freqs) if freqs else None

    def _phase_bin(self, t: float) -> int:
        """時刻 t の RF 位相ビン番号 bin = floor((t mod T)/T × bins) を返す。"""
        frac = (t % self._cycle_period) / self._cycle_period
        return min(int(frac * self._cycle_bins), self._cycle_bins - 1)

    def _build_see_edges(self) -> None:
        """境界エッジ (隣接 = -1) ごとの SEE 属性表を構築する。

        エッジ両端の節点がともに γ>0 の電極/Dirichlet 辺に属する場合に、
        γ (両端の最小値)・内向き単位法線・境界からわずかに内側へ置く
        オフセット量を (要素, ローカルエッジ) で引ける配列に記録する。
        """
        self._edge_gamma: np.ndarray | None = None
        gm_dict = self.mesh.see_gamma
        if not gm_dict:
            return
        gm = np.zeros(self.n_nodes)
        gm[np.fromiter(gm_dict.keys(), dtype=np.int64)] = np.fromiter(
            gm_dict.values(), dtype=np.float64
        )
        # 境界エッジ: adjacency[t, i] == -1 (頂点 i の対辺)
        ts, loc = np.nonzero(self.adjacency == -1)
        n1 = self.tris[ts, (loc + 1) % 3]
        n2 = self.tris[ts, (loc + 2) % 3]
        n_opp = self.tris[ts, loc]
        g_edge = np.minimum(gm[n1], gm[n2])  # 両端とも γ>0 のときのみ >0
        if not np.any(g_edge > 0.0):
            return
        nodes = self.mesh.nodes
        p1, p2, po = nodes[n1], nodes[n2], nodes[n_opp]
        mid = 0.5 * (p1 + p2)
        t_vec = p2 - p1
        perp = np.stack([-t_vec[:, 1], t_vec[:, 0]], axis=1)
        # 内向き = 対頂点の側
        sgn = np.where(np.sum(perp * (po - mid), axis=1) >= 0.0, 1.0, -1.0)
        nrm = perp * (sgn / np.linalg.norm(perp, axis=1))[:, None]
        h = np.abs(np.sum((po - mid) * nrm, axis=1))  # エッジから対頂点までの高さ

        m = len(self.tris)
        self._edge_gamma = np.zeros((m, 3))
        self._edge_normal = np.zeros((m, 3, 2))
        self._edge_delta = np.zeros((m, 3))
        self._edge_gamma[ts, loc] = g_edge
        self._edge_normal[ts, loc] = nrm
        self._edge_delta[ts, loc] = 1e-3 * h  # 境界からわずかに内側

    def _build_boundary_edges(self) -> None:
        """鏡面反射・周期の境界メッシュエッジ表を構築する。

        鏡面反射エッジは pic.reflect_edges と boundaries の symmetry 指定の和集合。
        periodic 指定の対辺は周期ラップ対象とする。表の構築は particles.py の
        _build_boundary_tables (trace と共通) に委譲する。
        """
        self._edge_reflect: np.ndarray | None = None
        self._edge_periodic: np.ndarray | None = None
        refl = set(self.pic.reflect_edges)
        periodic_pairs: list[tuple[int, int]] = []
        for bc in self.project.geometry.boundaries:
            if bc.type == "symmetry":
                refl.update(bc.edges)
            elif bc.type == "periodic":
                periodic_pairs.append((bc.edges[0], bc.edges[1]))
        tables = _build_boundary_tables(
            self.project.geometry.domain.polygon, self.mesh, self.adjacency,
            sorted(refl), periodic_pairs,
        )
        if tables is None:
            return
        if tables.reflect is not None:
            self._edge_reflect = tables.reflect
            self._refl_normal = tables.refl_normal
            self._refl_point = tables.refl_point  # エッジ上の基準点 (符号付き距離用)
        if tables.periodic is not None:
            self._edge_periodic = tables.periodic
            self._per_shift = tables.shift

    def _apply_reflection(
        self,
        x_new: np.ndarray,
        v_new: np.ndarray,
        elem: np.ndarray,
        absorbed: np.ndarray,
        b_elem: np.ndarray,
        b_loc: np.ndarray,
        l_out: np.ndarray | None = None,
    ) -> None:
        """反射エッジで検出された粒子を鏡面反射する (全引数を in-place 更新)。

        位置は境界線について折り返し、速度は法線成分 (vx, vy のみ) を反転する。
        反射後に所属要素を再探索し、コーナーで別の壁に到達した粒子は次の反復で
        処理する。反射エッジ以外の壁に達した粒子は absorbed のまま残す
        (通常の壁吸収として扱われる)。
        """
        for _ in range(8):
            idx = np.nonzero(absorbed)[0]
            if idx.size == 0:
                return
            ea, el = b_elem[idx], b_loc[idx]
            refl = self._edge_reflect[ea, el]
            if not np.any(refl):
                return
            r_idx = idx[refl]
            ea, el = ea[refl], el[refl]
            nrm = self._refl_normal[ea, el]
            # 符号付き距離 d < 0 = 境界の外側。折り返して境界内へ戻す
            d = np.sum((x_new[r_idx] - self._refl_point[ea, el]) * nrm, axis=1)
            x_new[r_idx] -= 2.0 * np.minimum(d, 0.0)[:, None] * nrm
            vn = np.sum(v_new[r_idx, :2] * nrm, axis=1)
            v_new[r_idx, :2] -= 2.0 * vn[:, None] * nrm
            # 反射後の所属要素を境界要素から再探索 (再度壁に達したら次の反復へ)
            sub_l = None if l_out is None else np.empty((r_idx.size, 3))
            e2, a2, be2, bl2 = _walk_step(
                self.coeffs, self.adjacency, ea, x_new[r_idx], sub_l,
                packed=self._coeffs_packed,
            )
            elem[r_idx] = e2
            absorbed[r_idx] = a2
            b_elem[r_idx] = be2
            b_loc[r_idx] = bl2
            if l_out is not None:
                l_out[r_idx] = sub_l

    def _apply_periodic(
        self,
        x_new: np.ndarray,
        elem: np.ndarray,
        absorbed: np.ndarray,
        b_elem: np.ndarray,
        b_loc: np.ndarray,
        l_out: np.ndarray | None = None,
    ) -> bool:
        """周期エッジで検出された粒子を反対側へラップする (in-place 更新)。

        位置を周期ベクトル分平行移動し (速度不変)、所属要素はラップした
        少数粒子のみ _locate_initial (総当たり) で再特定して walk で確定する。
        壁カウンタには計上しない。ラップ対象を処理した場合 True を返す
        (コーナーで別の壁に達した粒子は呼び出し側の反復で処理する)。
        """
        idx = np.nonzero(absorbed)[0]
        if idx.size == 0:
            return False
        per = self._edge_periodic[b_elem[idx], b_loc[idx]]
        if not np.any(per):
            return False
        p_idx = idx[per]
        ea, el = b_elem[p_idx], b_loc[p_idx]
        x_new[p_idx] += self._per_shift[ea, el]
        elem0 = _locate_initial(self.coeffs, x_new[p_idx])
        sub_l = None if l_out is None else np.empty((p_idx.size, 3))
        e2, a2, be2, bl2 = _walk_step(
            self.coeffs, self.adjacency, elem0, x_new[p_idx], sub_l,
            packed=self._coeffs_packed,
        )
        elem[p_idx] = e2
        absorbed[p_idx] = a2
        b_elem[p_idx] = be2
        b_loc[p_idx] = bl2
        if l_out is not None:
            l_out[p_idx] = sub_l
        return True

    def _accumulate_surface_charge(
        self,
        sp: PicSpecies,
        elem_new: np.ndarray,
        l_new: np.ndarray,
        hit: np.ndarray,
    ) -> None:
        """誘電体に吸収された粒子の電荷 w·q を表面電荷 Q_surf へ加算する。

        吸収位置 (進入した誘電体要素内) の P1 重心座標重みで節点へ散布する
        (点電荷の P1 射影と同じ扱い)。位置は表面の直内側 (高々1ステップの
        移動量) なので重みはほぼ表面節点に集中するが、進入要素の内部節点にも
        僅かに載る近似である。周期境界では正準節点番号 (tris_dep) で積む。
        """
        contrib = (sp.q * sp.w[hit])[:, None] * l_new[hit]
        self.q_surf += np.bincount(
            self.tris_dep[elem_new[hit]].ravel(),
            weights=contrib.ravel(),
            minlength=self.n_nodes,
        )

    def _emit_see_dielectric(
        self,
        sp: PicSpecies,
        v_new: np.ndarray,
        elem_new: np.ndarray,
        l_new: np.ndarray,
        hit: np.ndarray,
    ) -> None:
        """γ>0 の誘電体へ進入して吸収されたイオンから確率 γ で二次電子を放出する。

        固体吸収経路には境界エッジ (法線) が無いため、放出速度は入射イオン速度の
        単位ベクトルの逆向きに see_energy_ev を与える近似とする (面内2成分 + vz)。
        位置はプッシュ前のイオン位置 (直前までいたプラズマ側の要素内)、所属要素も
        プッシュ前のものを流用する (locate 不要)。重みは吸収イオンと同じ。

        表面電荷の収支整合: 電子が表面から放出される分だけ表面は正に帯電するため、
        +e·w をイオン吸収と同じ P1 射影 (進入要素の重心座標) で Q_surf へ加算する。
        電極 γ 由来の既存 SEE (_emit_see) は表面電荷管理外なので変更しない。
        sp.x / sp.elem は更新前 (プッシュ前) の状態であることを前提とする。
        """
        idx = np.nonzero(hit)[0]
        gam = self._solid_gamma[elem_new[idx]]
        cand = gam > 0.0
        if not np.any(cand):
            return
        c_idx = idx[cand]
        accept = self._see_rng.random(len(c_idx)) < gam[cand]
        if not np.any(accept):
            return
        c_idx = c_idx[accept]

        # 放出速度 = 入射イオン速度 (3成分) の逆向き単位ベクトル × SEE 速さ。
        # 速度ゼロの入射 (通常あり得ない) は方向が定義できないためスキップする
        v_in = v_new[c_idx]
        speed = np.sqrt(v_in[:, 0] ** 2 + v_in[:, 1] ** 2 + v_in[:, 2] ** 2)
        ok = speed > 0.0
        c_idx, v_in, speed = c_idx[ok], v_in[ok], speed[ok]
        if c_idx.size == 0:
            return
        v_see = -self._see_speed * v_in / speed[:, None]

        pos = sp.x[c_idx]        # プッシュ前の位置 (プラズマ側)
        elems = sp.elem[c_idx]   # プッシュ前の所属要素
        el = self.species["electron"]
        el.x = np.concatenate([el.x, pos])
        el.v = np.concatenate([el.v, v_see])
        el.w = np.concatenate([el.w, sp.w[c_idx]])
        el.elem = np.concatenate([el.elem, elems])
        self._bary_append(el, self._bary_of(pos, elems))
        self.see_events += len(c_idx)

        # 表面電荷の収支: 放出電子分 +e·w を吸収位置と同じ P1 射影で加算する
        contrib = (QE * sp.w[c_idx])[:, None] * l_new[c_idx]
        self.q_surf += np.bincount(
            self.tris_dep[elem_new[c_idx]].ravel(),
            weights=contrib.ravel(),
            minlength=self.n_nodes,
        )

    def _collect_ions(
        self,
        sp: PicSpecies,
        x_new: np.ndarray,
        v_new: np.ndarray,
        absorbed: np.ndarray,
        b_elem: np.ndarray,
        b_loc: np.ndarray,
        removed: np.ndarray,
    ) -> None:
        """吸収イオンのうちコレクタ線分に達したものを記録する (IEDF/IADF)。

        吸収位置は既存の補間位置を使う: 壁吸収 (walk 検出) は境界エッジ L=0 との
        線形補間、誘電体表面吸収は現在位置 (prompts/24 と同じ扱い)。
        エネルギーは衝突時速度 (3成分) から ½m|v|²/e [eV]。入射角は法線に対する
        面内角 [deg] で、符号は接線 (p1→p2) 方向成分の符号 (範囲 (-90, 90))。
        法線は入射イオンと逆向き成分を持つ側 (表面正面) を基準とするため |v·n| を使う。
        sp.x は更新前 (プッシュ前) の位置であることを前提とする。
        """
        idx = np.nonzero(removed)[0]
        if idx.size == 0:
            return
        pos = np.empty((idx.size, 2))
        wall = absorbed[idx]
        if np.any(wall):
            # 壁吸収: 越えた境界エッジの重心座標 L=0 を x_prev → x_new で線形補間
            w_idx = idx[wall]
            ea, eloc = b_elem[w_idx], b_loc[w_idx]
            a, b, c, det = self.coeffs
            aa, bb, cc, dd = a[ea, eloc], b[ea, eloc], c[ea, eloc], det[ea]
            xp, xn = sp.x[w_idx], x_new[w_idx]
            l0 = (aa + bb * xp[:, 0] + cc * xp[:, 1]) / dd
            l1 = (aa + bb * xn[:, 0] + cc * xn[:, 1]) / dd
            denom = l0 - l1
            denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
            frac = np.clip(l0 / denom, 0.0, 1.0)
            pos[wall] = xp + frac[:, None] * (xn - xp)
        if not np.all(wall):
            # 誘電体表面吸収: 現在位置 (侵入直前〜現在の間で良い)
            pos[~wall] = x_new[idx[~wall]]

        # 各コレクタ線分との距離 (法線方向) と区間 (接線方向の射影) の判定。
        # 1つのイオンが複数コレクタに同時該当する場合は両方に記録する
        for st in self._collectors_st:
            d = pos - st["p1"]
            proj = d[:, 0] * st["tan"][0] + d[:, 1] * st["tan"][1]
            dist = np.abs(d[:, 0] * st["nrm"][0] + d[:, 1] * st["nrm"][1])
            sel = (dist <= st["tol"]) & (proj >= 0.0) & (proj <= st["len"])
            if not np.any(sel):
                continue
            s_idx = idx[sel]
            vel = v_new[s_idx]
            wgt = sp.w[s_idx].astype(np.float64)
            k = len(s_idx)
            st["count"] += k
            st["weight"] += float(wgt.sum())

            # サンプル上限 (コレクタごと) 到達後は count / total_weight のみ加算する
            room = COLLECTOR_MAX_SAMPLES - st["samples"]
            if room <= 0:
                continue
            if k > room:
                vel, wgt, k = vel[:room], wgt[:room], room
            e_ev = 0.5 * sp.m * (vel[:, 0] ** 2 + vel[:, 1] ** 2 + vel[:, 2] ** 2) / QE
            vt = vel[:, 0] * st["tan"][0] + vel[:, 1] * st["tan"][1]
            vn = vel[:, 0] * st["nrm"][0] + vel[:, 1] * st["nrm"][1]
            ang = np.degrees(np.arctan2(vt, np.abs(vn)))
            st["e"].append(e_ev)
            st["a"].append(ang)
            st["w"].append(wgt)
            st["samples"] += k

    def _collector_data(self) -> list[dict] | None:
        """各コレクタの記録を dict のリストにまとめる (collectors と同順)。"""
        if not self._collectors_st:
            return None
        empty = np.zeros(0)
        return [
            {
                "count": st["count"],
                "total_weight": st["weight"],
                "energies_ev": np.concatenate(st["e"]) if st["e"] else empty,
                "angles_deg": np.concatenate(st["a"]) if st["a"] else empty,
                "weights": np.concatenate(st["w"]) if st["w"] else empty,
                "truncated": st["count"] > st["samples"],
            }
            for st in self._collectors_st
        ]

    def _emit_see(
        self,
        sp: PicSpecies,
        x_new: np.ndarray,
        b_elem: np.ndarray,
        b_loc: np.ndarray,
        absorbed: np.ndarray,
    ) -> None:
        """γ>0 の電極エッジに吸収されたイオンから確率 γ で二次電子を放出する。

        位置 = 吸収位置 (境界エッジとの交点をわずかに内側へ)、
        速度 = 内向き法線方向に see_energy_ev、重み = 吸収イオンと同じ。
        sp.x は更新前 (プッシュ前) の位置であることを前提とする。
        """
        idx = np.nonzero(absorbed)[0]
        ea = b_elem[idx]
        eloc = b_loc[idx]
        gam = self._edge_gamma[ea, eloc]
        cand = gam > 0.0
        if not np.any(cand):
            return
        c_idx = idx[cand]
        ea, eloc = ea[cand], eloc[cand]
        accept = self._see_rng.random(len(c_idx)) < gam[cand]
        if not np.any(accept):
            return
        c_idx, ea, eloc = c_idx[accept], ea[accept], eloc[accept]

        # 吸収位置: 越えたエッジの重心座標 L=0 を x_prev → x_new で線形補間
        a, b, c, det = self.coeffs
        aa, bb, cc, dd = a[ea, eloc], b[ea, eloc], c[ea, eloc], det[ea]
        xp, xn = sp.x[c_idx], x_new[c_idx]
        l0 = (aa + bb * xp[:, 0] + cc * xp[:, 1]) / dd
        l1 = (aa + bb * xn[:, 0] + cc * xn[:, 1]) / dd
        denom = l0 - l1
        denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
        frac = np.clip(l0 / denom, 0.0, 1.0)
        x_hit = xp + frac[:, None] * (xn - xp)

        nrm = self._edge_normal[ea, eloc]
        pos = x_hit + self._edge_delta[ea, eloc][:, None] * nrm
        el = self.species["electron"]
        el.x = np.concatenate([el.x, pos])
        # SEE 電子は法線方向のみ (vz = 0)
        v_see = np.column_stack([self._see_speed * nrm, np.zeros(len(nrm))])
        el.v = np.concatenate([el.v, v_see])
        el.w = np.concatenate([el.w, sp.w[c_idx]])
        el.elem = np.concatenate([el.elem, ea])
        self._bary_append(el, self._bary_of(pos, ea))
        self.see_events += len(c_idx)

    def _sample_uniform(self, rng: np.random.Generator, n: int):
        """ドメイン内 (メッシュ要素上 = 電極領域を除く) の一様分布サンプリング。

        要素を体積比例 (xy: 面積、rz: 2πr̄A) で選び、要素内は重心座標の一様分布で
        配置する。所属要素が同時に確定するので walk 初期化が不要。
        誘電体 (固体) 要素は粒子が入れないため装荷対象から除外する。
        rz では要素内の r 依存 (∝r) を r̄ で代表する近似 (要素サイズ ≪ r で厳密に
        一様密度へ収束。軸近傍の要素では僅かに軸寄りが過剰になる)。
        """
        if self._solid_elem is None:
            p_elem = self.elem_vol / self.elem_vol.sum()
            elem = rng.choice(len(self.tris), size=n, p=p_elem).astype(np.int64)
        else:
            loadable = np.nonzero(~self._solid_elem)[0]
            p_elem = self.elem_vol[loadable] / self.elem_vol[loadable].sum()
            elem = loadable[rng.choice(len(loadable), size=n, p=p_elem)].astype(np.int64)
        r1 = np.sqrt(rng.random(n))
        r2 = rng.random(n)
        pts = self.mesh.nodes[self.tris[elem]]  # (n, 3, 2)
        x = (
            (1.0 - r1)[:, None] * pts[:, 0]
            + (r1 * (1.0 - r2))[:, None] * pts[:, 1]
            + (r1 * r2)[:, None] * pts[:, 2]
        )
        return x, elem

    def _bary_of(self, x: np.ndarray, elem: np.ndarray) -> np.ndarray:
        """位置・所属要素から P1 重心座標 (n, 3) を計算する。"""
        a, b, c, det = self.coeffs
        return (a[elem] + b[elem] * x[:, 0:1] + c[elem] * x[:, 1:2]) / det[elem][:, None]

    def _bary_cached(self, sp: PicSpecies) -> np.ndarray:
        """種の重心座標キャッシュを返す (無効なら再計算して保存)。"""
        l = sp.bary
        if l is None or len(l) != len(sp.x):
            l = self._bary_of(sp.x, sp.elem)
            sp.bary = l
        return l

    @staticmethod
    def _bary_append(sp: PicSpecies, l_new: np.ndarray) -> None:
        """粒子追加後にキャッシュへ重心座標行を追記する (不整合なら破棄して遅延再計算)。"""
        if sp.bary is not None and len(sp.bary) + len(l_new) == len(sp.x):
            sp.bary = np.concatenate([sp.bary, l_new])
        else:
            sp.bary = None

    def _nidx_cached(self, sp: PicSpecies) -> np.ndarray:
        """種の所属要素節点番号 tris_dep[elem] を返す (キャッシュが無効なら再計算)。

        密度積算 (ステップ終端) と次ステップの電荷堆積は同じ粒子状態を見るので、
        キャッシュにより gather を1回に減らせる。周期境界では正準化された
        節点番号 (スレーブ→マスター置換済み) を使うため、堆積は自動的に整合する。
        """
        ni = sp.nidx
        if ni is None or len(ni) != len(sp.elem):
            ni = self.tris_dep[sp.elem]
            sp.nidx = ni
        return ni

    def _deposit_species(self, sp: PicSpecies) -> np.ndarray:
        """1種の電荷を P1 形状関数 (重心座標) の重みで節点へ散布する。

        f_i = Σ_p w_p q_p L_i(x_p)。散布は np.add.at と等価だが高速な
        np.bincount で行う。重心座標は walk のキャッシュを再利用する。
        """
        l = self._bary_cached(sp)
        contrib = (sp.q * sp.w)[:, None] * l
        return np.bincount(
            self._nidx_cached(sp).ravel(), weights=contrib.ravel(), minlength=self.n_nodes
        )

    def _deposit(self) -> np.ndarray:
        """全種の電荷堆積 (不動種はキャッシュを再利用)。

        イオンサブサイクリング (prompts/50) 中はイオンが休止ステップで動かない
        ため、堆積ベクトルをキャッシュして再利用する (イオン配列が変わる箇所で
        _f_ion_cache = None に無効化される)。
        """
        f = np.zeros(self.n_nodes)
        for sp in self.species.values():
            if len(sp.x) == 0:
                continue
            if not sp.mobile:
                if sp.name not in self._f_immobile:
                    self._f_immobile[sp.name] = self._deposit_species(sp)
                f += self._f_immobile[sp.name]
            elif sp.name == "ion" and self._sub > 1:
                if self._f_ion_cache is None:
                    self._f_ion_cache = self._deposit_species(sp)
                f += self._f_ion_cache
            else:
                f += self._deposit_species(sp)
        return f

    def _dirichlet_values(self, t: float) -> np.ndarray:
        """時刻 t の Dirichlet 値 V(t) = V_dc + Σ_k A_k sin(ω_k t + φ_k) + V_wf(t)。

        デュアル周波数対応 (prompts/49): rf_* は (n_fixed, K) 行列で、成分和を取る。
        CSV 波形 (prompts/73) は種類数が少ない想定なので、波形ごとに1回だけ
        _eval_waveform を呼び、節点→波形インデックス (self._wf_index) で展開する。
        """
        v = self.v_dc + np.sum(
            self.rf_amp * np.sin(self.rf_omega * t + self.rf_phase), axis=1
        )
        if self._waveforms:
            wf_vals = np.array(
                [_eval_waveform(wf.phase, wf.v, wf.freq_hz, t) for wf in self._waveforms]
            )
            mask = self._wf_index >= 0
            v[mask] += wf_vals[self._wf_index[mask]]
        return v

    def _solve_phi(self, f_dep: np.ndarray, t: float) -> np.ndarray:
        """ポアソン求解。前分解済み LU で右辺のみ更新して解く (再分解しない)。"""
        v = np.zeros(self.n_nodes)
        vd = self._dirichlet_values(t)
        v[self.fixed] = vd
        if self.rz:
            # 軸対称: リング電荷 Q [C] の P1 射影は f_i = Q·L_i/(2π)
            # (fem.assemble は弱形式の 2π を両辺で落とした規約のため。q_surf も同様)
            f_dep = f_dep / (2.0 * np.pi)
        f = self.f_static + f_dep
        if self._solid_elem is not None:
            # 誘電体の蓄積表面電荷を恒常的に加算する (誘電体なしの経路は数値不変)
            f = f + (self.q_surf / (2.0 * np.pi) if self.rz else self.q_surf)
        rhs = f[self.free] - self.k_fd @ vd
        v[self.free] = self.lu.solve(rhs)
        if self.canon is not None:
            v = v[self.canon]     # スレーブ節点へマスター値をコピー (表示互換)
            v[self.fixed] = vd    # Dirichlet 値は厳密に保持
        return v

    def _e_field(self, phi: np.ndarray):
        """要素ごとの E = -∇φ (P1 なので要素内一定)。"""
        _, b, c, det = self.coeffs
        vt = phi[self.tris]  # (M, 3)
        ex = -np.sum(vt * b, axis=1) / det
        ey = -np.sum(vt * c, axis=1) / det
        return ex, ey

    def _injection_velocities(self) -> np.ndarray:
        """注入粒子の速度。maxwell は持続 rng でステップごとに独立にサンプルする。"""
        inj = self.pic.injection
        em = inj.emitter
        sp = self.species[inj.species]
        if em.energy_dist == "maxwell":
            angle = math.radians(em.direction_deg)
            speed = math.sqrt(2.0 * em.energy_ev * QE / sp.m) if em.energy_ev > 0 else 0.0
            drift = speed * np.array([math.cos(angle), math.sin(angle), 0.0])
            sigma = math.sqrt(em.temperature_ev * QE / sp.m)
            # 2d3v: 熱速度成分は3成分で抽選する
            return drift[None, :] + self._inj_rng.normal(0.0, sigma, size=(em.n, 3))
        # mono: フェーズ2 の2成分速度に vz = 0 を付加
        base = self._inj_vel_base
        return np.column_stack([base, np.zeros(len(base))])

    def _inject(self, ex: np.ndarray, ey: np.ndarray) -> None:
        """エミッタ定常注入。初期半ステップ後退キックを適用して追加する。"""
        inj = self.pic.injection
        sp = self.species[inj.species]
        elem = self._inj_elem
        v = self._injection_velocities()
        e_at = np.stack([ex[elem], ey[elem]], axis=1)
        # イオン注入 + サブサイクル時は実効刻み _sub·dt で後退させる
        dt_inj = self.dt * self._sub if (inj.species == "ion" and self._sub > 1) else self.dt
        if self.rz:
            # 軸対称: 遠心力項 (maxwell 注入では vθ ≠ 0) も含めて半ステップ後退
            a0 = (sp.q / sp.m) * e_at
            r0 = np.maximum(self._inj_pos[:, self.ridx], _R_TINY)
            a0[:, self.ridx] += v[:, 2] ** 2 / r0
            v[:, :2] -= 0.5 * dt_inj * a0
        else:
            v[:, :2] -= 0.5 * dt_inj * (sp.q / sp.m) * e_at
        n = len(elem)
        sp.x = np.concatenate([sp.x, self._inj_pos])
        sp.v = np.concatenate([sp.v, v])
        sp.w = np.concatenate([sp.w, np.full(n, self._inj_w)])
        sp.elem = np.concatenate([sp.elem, elem])
        self._bary_append(sp, self._inj_bary)
        if inj.species == "ion":
            self._f_ion_cache = None  # イオンが増えたので堆積キャッシュを無効化

    def _emit_fn(self, exy: np.ndarray) -> float:
        """FN 電界放出 (prompts/46)。現ステップの表面電界から電子を放出する。

        各セグメントの放出実電子数 I·dt/e をマクロ重み w で割り、床関数の余りを
        セグメントごとに持ち越す (時間平均で正確に I を再現する)。放出位置は
        セグメント上の一様乱数、初速は法線方向 init_energy_ev。注入と同様に
        初期半ステップ後退キックを適用する。

        放出マクロ数が FN_MAX_MACROS_PER_STEP を超える場合 (強電界 + 小さい
        macro_weight で I·dt/(e·w) が巨大になるケース、prompts/53) は、マクロ数を
        上限に抑えて実効重み w_eff = total·w/上限 を与える (総電荷は保存される。
        メモリ確保エラー対策)。

        戻り値: このステップの総放出電流 [A/m]。
        """
        surf = self._fn_surf
        fn = self.pic.fn
        # 座標系を渡す (rz では I [A] = J·L·2πr̄。放出実電子数 I·dt/e は自動整合)
        _f_surf, seg_i = fn_segment_currents(surf, exy, fn, self.project.coord)
        i_tot = float(seg_i.sum())
        quota = seg_i * (self.dt / (QE * self._fn_w)) + self._fn_frac
        n_int = np.floor(quota).astype(np.int64)
        self._fn_frac = quota - n_int
        total = int(n_int.sum())
        if total == 0:
            return i_tot
        if total > FN_MAX_MACROS_PER_STEP:
            # マクロ数を上限に抑え、重みを引き上げて電荷を保存する (prompts/53)
            counts = distribute_particles(n_int.astype(np.float64), FN_MAX_MACROS_PER_STEP)
            w_emit = self._fn_w * (total / FN_MAX_MACROS_PER_STEP)
            n_emit = int(counts.sum())  # == FN_MAX_MACROS_PER_STEP
            seg_idx = np.repeat(np.arange(len(seg_i)), counts)
        else:
            w_emit = self._fn_w
            n_emit = total
            seg_idx = np.repeat(np.arange(len(seg_i)), n_int)
        t = self._fn_rng.random(n_emit)
        pos = (
            surf.pa[seg_idx]
            + t[:, None] * (surf.pb[seg_idx] - surf.pa[seg_idx])
            + surf.delta[seg_idx, None] * surf.nrm[seg_idx]
        )
        elem = surf.elem[seg_idx]
        sp = self.species["electron"]
        v = np.zeros((n_emit, 3))
        v[:, :2] = self._fn_speed * surf.nrm[seg_idx]
        # 初期半ステップ後退キック (注入と同じ規約)
        e_at = exy[elem]
        v[:, :2] -= 0.5 * self.dt * (sp.q / sp.m) * e_at
        bary = self._bary_of(pos, elem)
        sp.x = np.concatenate([sp.x, pos])
        sp.v = np.concatenate([sp.v, v])
        sp.w = np.concatenate([sp.w, np.full(n_emit, w_emit)])
        sp.elem = np.concatenate([sp.elem, elem])
        self._bary_append(sp, bary)
        self.fn_events += n_emit
        return i_tot

    def _walk_chunked_submit(
        self, elem: np.ndarray, x_new: np.ndarray, l_new: np.ndarray, futures: list
    ):
        """1種の walk を粒子チャンクに分割してプールへ投入する (prompts/50)。

        各粒子の walk は完全に独立 (読み取り共有のみ・書き込みは自分の行のみ) の
        決定的処理なので、チャンク分割しても結果は逐次実行とビット単位で一致する。
        戻り値は出力先の (elem_new, absorbed, b_elem, b_loc)。呼び出し側が
        futures の完了を待ってから読むこと。粒子数が少ない種はスレッド起動コストの
        方が高いため、その場で逐次計算する (futures には積まない)。
        """
        n = len(elem)
        if n < 4096:
            return _walk_step(
                self.coeffs, self.adjacency, elem, x_new, l_new,
                packed=self._coeffs_packed,
            )
        k = self._nthreads
        bounds = [(i * n) // k for i in range(k + 1)]
        elem_new = np.empty(n, dtype=np.int64)
        absorbed = np.empty(n, dtype=bool)
        b_elem = np.empty(n, dtype=np.int64)
        b_loc = np.empty(n, dtype=np.int64)

        def _run(lo: int, hi: int) -> None:
            sub_l = None if l_new is None else l_new[lo:hi]
            e, a, be, bl = _walk_step(
                self.coeffs, self.adjacency, elem[lo:hi], x_new[lo:hi], sub_l,
                packed=self._coeffs_packed,
            )
            elem_new[lo:hi] = e
            absorbed[lo:hi] = a
            b_elem[lo:hi] = be
            b_loc[lo:hi] = bl

        for i in range(k):
            if bounds[i + 1] > bounds[i]:
                futures.append(self._chunk_pool.submit(_run, bounds[i], bounds[i + 1]))
        return elem_new, absorbed, b_elem, b_loc

    def _run_walks(self, pushed) -> list:
        """種ごとの walk を実行する (2種以上なら並列)。

        walk は乱数を使わず入力のみで決まる決定的処理で、種間で共有する
        書き込み先も無いため、並列化しても結果はビット単位で不変。
        threads > 1 (prompts/50) では全種のチャンクをまとめてプールへ投入し、
        全ワーカーで消化する (種横断で負荷分散する)。
        """
        if self._chunk_pool is not None:
            futures: list = []
            results = [
                self._walk_chunked_submit(sp.elem, x_new, l_new, futures)
                for sp, _v, x_new, l_new in pushed
            ]
            for fut in futures:
                fut.result()
            return results
        if len(pushed) <= 1:
            return [
                _walk_step(
                    self.coeffs, self.adjacency, sp.elem, x_new, l_new,
                    packed=self._coeffs_packed,
                )
                for sp, _v, x_new, l_new in pushed
            ]
        # 先頭の種 (通常は電子 = 最も重い walk) をワーカーへ、残りを主スレッドで
        futures = [
            _walk_pool().submit(
                _walk_step, self.coeffs, self.adjacency, sp.elem, x_new, l_new,
                self._coeffs_packed,
            )
            for sp, _v, x_new, l_new in pushed[:1]
        ]
        rest = [
            _walk_step(
                self.coeffs, self.adjacency, sp.elem, x_new, l_new,
                packed=self._coeffs_packed,
            )
            for sp, _v, x_new, l_new in pushed[1:]
        ]
        return [f.result() for f in futures] + rest

    # ---- 1ステップ -----------------------------------------------------------

    def step(self) -> np.ndarray:
        """PIC 1サイクル。時刻 t_n の場を解き、粒子を t_{n+1} へ進める。

        戻り値: 時刻 t_n の節点電位 φ (フレーム生成に使う)。
        """
        dt = self.dt
        t = self.t
        # このステップが時間平均区間に入るか (step_count はステップ末尾で +1 される)
        accumulating = (
            self._accum_start is not None
            and self.step_count + 1 >= self._accum_start
        )

        # 1. 電荷堆積 → 2. ポアソン求解 (RF 含む V(t) で Dirichlet 更新)
        # 位相別計測 (prompts/75): 堆積とポアソン求解を別フェーズとして計測する
        t_dep0 = time.perf_counter()
        f_dep = self._deposit()
        t_dep1 = time.perf_counter()
        self.timing["deposit"] += t_dep1 - t_dep0
        phi = self._solve_phi(f_dep, t)
        t_solve1 = time.perf_counter()
        self.timing["solve"] += t_solve1 - t_dep1
        # 3. E 補間の準備 (要素ごとの一定値)
        ex, ey = self._e_field(phi)
        # 場のエネルギー (xy: [J/m]、rz: [J]。elem_vol は xy では area と同一)
        fe = float(np.sum(0.5 * self.eps_elem * (ex**2 + ey**2) * self.elem_vol))
        # 要素ごとの E を (M,2) に詰め、粒子への補間 gather を1回で済ませる
        exy = np.stack([ex, ey], axis=1)

        # 4. リープフロッグでプッシュ (種ごとの v_new, x_new を先に全て計算する)
        # イオンサブサイクリング (prompts/50): イオンは _sub ステップに1回だけ
        # 実効刻み _sub·dt で押す (最初のプッシュは step_count = 0)
        push_ions = self._sub == 1 or (self.step_count % self._sub == 0)
        ke: dict[str, float] = {}
        pushed: list[tuple[PicSpecies, np.ndarray, np.ndarray, np.ndarray]] = []
        for sp in self.species.values():
            if len(sp.x) == 0:
                ke[sp.name] = 0.0
                continue
            if not sp.mobile:
                # 不動種: 運動エネルギーのみ評価
                ke[sp.name] = 0.5 * sp.m * float(np.sum(sp.w * np.sum(sp.v**2, axis=1)))
                continue
            if sp.name == "ion" and not push_ions:
                # サブサイクル休止ステップ: イオンは動かさず、KE は直近プッシュ時の
                # 値を流用する (位置・所属要素・堆積キャッシュはそのまま有効)
                ke[sp.name] = self._last_ke_i
                continue
            dt_sp = dt * self._sub if (sp.name == "ion" and self._sub > 1) else dt
            e_at = exy[sp.elem]
            v_new = sp.v.copy()
            ang_l = None
            if self.rz:
                # 軸対称プッシュ (prompts/47): 遠心力項 vθ²/r を現在位置で評価して
                # 径方向加速度に加える (trace と同じ半陰的規約)。v[:, 2] は vθ
                ridx = self.ridx
                a_rz = (sp.q / sp.m) * e_at
                r_cur = np.maximum(sp.x[:, ridx], _R_TINY)
                ang_l = sp.x[:, ridx] * sp.v[:, 2]  # 角運動量 L = r·vθ (保存量)
                a_rz[:, ridx] += sp.v[:, 2] ** 2 / r_cur
                v_new[:, :2] += dt_sp * a_rz
            elif self._b is not None:
                # 一様磁場 (prompts/51): Boris 法 (半キック E → 回転 B → 半キック E)
                half = (sp.q / sp.m) * (0.5 * dt_sp)
                v_new[:, :2] += half * e_at
                v_new = v_new @ self._boris_rt[sp.name]
                v_new[:, :2] += half * e_at
            else:
                # 2d3v: E は vx, vy のみに作用し、vz はそのまま
                v_new[:, :2] += (sp.q / sp.m) * dt_sp * e_at
            # 時刻中心化した運動エネルギー: KE(t_n) ≈ ½ m Σ w v(n-1/2)·v(n+1/2)
            # (v·v_new は列ごとの積和で評価: axis 縮約より高速で結果はビット一致)
            vdot = (
                sp.v[:, 0] * v_new[:, 0] + sp.v[:, 1] * v_new[:, 1]
            ) + sp.v[:, 2] * v_new[:, 2]
            ke[sp.name] = 0.5 * sp.m * float(np.sum(sp.w * vdot))
            if sp.name == "ion":
                self._last_ke_i = ke[sp.name]
            x_new = sp.x + dt_sp * v_new[:, :2]
            if self.rz:
                # 軸交差 (r < 0): 径座標・径速度・vθ (= L の符号) を鏡映してから
                # walk する (軸 r=0 は境界メッシュエッジだが吸収させない)
                cross = x_new[:, self.ridx] < 0.0
                if np.any(cross):
                    x_new[cross, self.ridx] = -x_new[cross, self.ridx]
                    v_new[cross, self.ridx] = -v_new[cross, self.ridx]
                    ang_l[cross] = -ang_l[cross]
                # 角運動量保存: 移動後の位置で vθ = L/r を更新する
                r_new = np.maximum(x_new[:, self.ridx], _R_TINY)
                v_new[:, 2] = np.where(ang_l != 0.0, ang_l / r_new, 0.0)
            pushed.append((sp, v_new, x_new, np.empty((len(x_new), 3))))
        t_push1 = time.perf_counter()
        self.timing["gather_push"] += t_push1 - t_solve1

        # 5. walk 更新 (種ごとに独立・決定的なので、2種のときは並列に実行して
        #    2コアを使う。numpy の大きな ufunc は GIL を解放するため実効的)
        # 位相別計測 (prompts/75): スレッド並列時も submit〜完了待ちの壁時計時間を
        # そのまま計測する (_run_walks 内部で完了待ちまで行うため)
        walk_results = self._run_walks(pushed)
        t_walk1 = time.perf_counter()
        self.timing["walk"] += t_walk1 - t_push1

        # 5.1. 境界吸収・鏡面反射・SEE (種の順序は従来どおり electron → ion)
        for (sp, v_new, x_new, l_new), res_walk in zip(pushed, walk_results):
            elem_new, absorbed, b_elem, b_loc = res_walk
            # 鏡面反射エッジに達した粒子は吸収せず折り返し、周期エッジに達した
            # 粒子は反対側へラップする (いずれも壁カウンタに含めない)
            if (
                self._edge_reflect is not None or self._edge_periodic is not None
            ) and np.any(absorbed):
                for _ in range(8):
                    if self._edge_reflect is not None and np.any(absorbed):
                        self._apply_reflection(
                            x_new, v_new, elem_new, absorbed, b_elem, b_loc, l_new
                        )
                    if self._edge_periodic is None or not self._apply_periodic(
                        x_new, elem_new, absorbed, b_elem, b_loc, l_new
                    ):
                        break
            # 誘電体 (固体) 要素へ入った粒子は表面で吸収する
            # (壁カウンタに計上するが SEE は発生させない、prompts/24)。
            # 吸収電荷は表面電荷 Q_surf へ蓄積して場にフィードバックする (prompts/25)
            if self._solid_elem is not None:
                solid_hit = ~absorbed & self._solid_elem[elem_new]
                if np.any(solid_hit):
                    self._accumulate_surface_charge(sp, elem_new, l_new, solid_hit)
                    # 誘電体 SEE (prompts/38): γ>0 の誘電体へのイオン吸収のみ
                    # (電子の誘電体吸収では発生させない)
                    if sp.name == "ion" and self._solid_gamma is not None:
                        self._emit_see_dielectric(sp, v_new, elem_new, l_new, solid_hit)
                removed = absorbed | solid_hit
            else:
                removed = absorbed
            n_abs = int(removed.sum())
            if n_abs:
                sp.wall_absorbed += n_abs
                # IEDF/IADF コレクタ: 平均区間中に吸収されたイオンを記録
                # (外周・電極輪郭・誘電体表面のすべて。電子・SEE は対象外)
                if sp.name == "ion" and self._collectors_st and accumulating:
                    self._collect_ions(
                        sp, x_new, v_new, absorbed, b_elem, b_loc, removed
                    )
                # SEE: γ>0 電極へのイオン吸収で二次電子を生成 (sp.x は更新前の位置)
                if sp.name == "ion" and self._edge_gamma is not None and np.any(absorbed):
                    self._emit_see(sp, x_new, b_elem, b_loc, absorbed)
                keep = ~removed
                sp.x = x_new[keep]
                sp.v = v_new[keep]
                sp.w = sp.w[keep]
                sp.elem = elem_new[keep]
                sp.bary = l_new[keep]
            else:
                sp.x = x_new
                sp.v = v_new
                sp.elem = elem_new
                sp.bary = l_new
            sp.nidx = None  # 所属要素が変わったので節点番号キャッシュを無効化
            if sp.name == "ion":
                self._f_ion_cache = None  # イオンが動いたので堆積キャッシュを無効化
        # 境界吸収・反射・周期・SEE も「5. walk 更新・境界吸収」の一部として計測する
        t_walk2 = time.perf_counter()
        self.timing["walk"] += t_walk2 - t_walk1

        # 5.5. MCC 衝突 (衝突は位置を変えないので所属要素の更新は不要)
        if self.mcc is not None:
            el = self.species["electron"]
            io = self.species["ion"]
            if len(el.x):
                res = self.mcc.collide_electrons(el.x, el.v, el.w, el.elem, dt)
                self.coll_e += res.n_coll
                self.ion_events += res.n_ionization
                if res.new_x is not None:
                    # 電離: 衝突位置に新電子 + 新イオン (ガス温度 Maxwell) を生成
                    new_l = self._bary_of(res.new_x, res.new_elem)  # 電子・イオン共通
                    if accumulating:
                        # 電離イベント発生位置を P1 重みで散布 (電離レート用)
                        ion_vec = np.bincount(
                            self.tris_dep[res.new_elem].ravel(),
                            weights=(res.new_w[:, None] * new_l).ravel(),
                            minlength=self.n_nodes,
                        )
                        self._accum_ion += ion_vec
                        if self._cycle_ion is not None:
                            # 位相分解の電離レート (prompts/52)。t はステップ開始時刻
                            self._cycle_ion[self._phase_bin(t)] += ion_vec
                    el.x = np.concatenate([el.x, res.new_x])
                    el.v = np.concatenate([el.v, res.new_v_e])
                    el.w = np.concatenate([el.w, res.new_w])
                    el.elem = np.concatenate([el.elem, res.new_elem])
                    self._bary_append(el, new_l)
                    io.x = np.concatenate([io.x, res.new_x.copy()])
                    io.v = np.concatenate([io.v, res.new_v_i])
                    io.w = np.concatenate([io.w, res.new_w.copy()])
                    io.elem = np.concatenate([io.elem, res.new_elem.copy()])
                    self._bary_append(io, new_l)
                    # 不動イオン・サブサイクル中のイオンの堆積キャッシュを無効化
                    self._f_immobile.pop("ion", None)
                    self._f_ion_cache = None
            if io.mobile and len(io.x) and push_ions:
                # サブサイクル時 (prompts/50) はイオン衝突もプッシュステップに
                # 実効刻み _sub·dt でまとめて適用する (衝突は速度のみ変えるため
                # 堆積キャッシュには影響しない)。elem は非一様ガス場の局所参照用
                self.mcc.collide_ions(
                    io.v, dt * self._sub if self._sub > 1 else dt, io.elem
                )
        # MCC 無効時も t_walk2 との差分は測るので常に加算する (self.mcc is None なら ≈0)
        t_mcc1 = time.perf_counter()
        self.timing["mcc"] += t_mcc1 - t_walk2

        # 6. 注入
        if self.pic.injection is not None:
            self._inject(ex, ey)
        # 6.5. FN 電界放出 (prompts/46)。表面電界はこのステップの場 (exy) を使う
        fn_i = self._emit_fn(exy) if self._fn_surf is not None else 0.0

        self.t = t + dt
        self.step_count += 1

        # 節点密度・時間平均フィールド (enable_density_accum 以後、毎ステップ積算)
        if accumulating:
            self._accumulate_fields(phi, ex, ey, t)

        # 診断記録 (毎ステップ)
        el, io = self.species["electron"], self.species["ion"]
        h = self.history
        h["t"].append(t)
        h["ke_e"].append(ke["electron"])
        h["ke_i"].append(ke["ion"])
        h["fe"].append(fe)
        h["n_e"].append(len(el.x))
        h["n_i"].append(len(io.x))
        h["wall_e"].append(el.wall_absorbed)
        h["wall_i"].append(io.wall_absorbed)
        h["phi_min"].append(float(phi.min()))
        h["phi_max"].append(float(phi.max()))
        h["coll_e"].append(self.coll_e)
        h["ion_events"].append(self.ion_events)
        h["see_events"].append(self.see_events)
        # 累計表面電荷 (全誘電体合計 [C/m])。誘電体なしなら常に 0
        h["surf_q"].append(float(self.q_surf.sum()))
        # FN 電界放出: このステップの総放出電流 [A/m] と累計放出マクロ電子数
        h["fn_i"].append(fn_i)
        h["fn_events"].append(self.fn_events)
        # 注入・FN放出・時間平均積算・診断記録などの細かい処理をまとめて計測する
        t_other1 = time.perf_counter()
        self.timing["other"] += t_other1 - t_mcc1
        return phi

    # ---- 節点密度の時間平均 ---------------------------------------------------

    def enable_density_accum(self, start_step: int) -> None:
        """節点密度・時間平均フィールドの積算を有効化する。

        step_count (完了ステップ数、1始まり) が start_step 以上のステップから、
        毎ステップ種ごとに P1 重みで節点へマクロ重みを散布して積算する。
        併せて φ・E ベクトル・電子運動エネルギー・電離イベントも積算する
        (prompts/26。追加コストは平均区間のみに限られる)。
        """
        self._accum_start = int(start_step)
        self._accum_count = 0
        self._accum = {name: np.zeros(self.n_nodes) for name in self.species}
        self._accum_phi = np.zeros(self.n_nodes)
        self._accum_e = np.zeros((len(self.tris), 2))
        self._accum_ke_e = np.zeros(self.n_nodes)
        self._accum_ion = np.zeros(self.n_nodes)
        # 位相分解アキュムレータ (RF あり + phase_bins > 0 のときのみ、prompts/28)
        if self._cycle_enabled:
            nb = self._cycle_bins
            self._cycle_phi = np.zeros((nb, self.n_nodes))
            self._cycle_ne = np.zeros((nb, self.n_nodes))
            self._cycle_ni = np.zeros((nb, self.n_nodes))
            self._cycle_count = np.zeros(nb, dtype=np.int64)
            # 追加フィールド (prompts/52): E ベクトル・電子KE・電離イベント
            self._cycle_e = np.zeros((nb, len(self.tris), 2))
            self._cycle_ke = np.zeros((nb, self.n_nodes))
            self._cycle_ion = np.zeros((nb, self.n_nodes))

    def _accumulate_fields(
        self, phi: np.ndarray, ex: np.ndarray, ey: np.ndarray, t_step: float
    ) -> None:
        """1ステップ分のフィールドと密度を積算する (平均区間のみ呼ばれる)。

        - φ (節点)・E ベクトル (要素) はそのまま加算
        - 電子温度用に Σ w·L_i·(½m|v|²) (3速度成分) を節点へ散布
        - 種ごとの節点重み (Σ w_p L_i) は1回だけ散布し、全体平均 (_accum) と
          位相ビン (RF あり) の両方で共有する。重心座標は walk のキャッシュを再利用
        - t_step はこのステップの開始時刻 (φ の評価時刻。位相ビンの割り当てに使う)
        """
        self._accum_phi += phi
        self._accum_e[:, 0] += ex
        self._accum_e[:, 1] += ey

        # RF 位相ビン (cycle 無効なら b = -1 でスキップ)
        b = -1
        if self._cycle_phi is not None:
            b = self._phase_bin(t_step)
            self._cycle_phi[b] += phi
            self._cycle_count[b] += 1
            # 位相分解の E ベクトル (prompts/52)
            self._cycle_e[b, :, 0] += ex
            self._cycle_e[b, :, 1] += ey

        el = self.species["electron"]
        if len(el.x):
            ke_p = 0.5 * ME * (el.v[:, 0] ** 2 + el.v[:, 1] ** 2 + el.v[:, 2] ** 2)
            contrib = (el.w * ke_p)[:, None] * self._bary_cached(el)
            ke_vec = np.bincount(
                self._nidx_cached(el).ravel(), weights=contrib.ravel(), minlength=self.n_nodes
            )
            self._accum_ke_e += ke_vec
            if b >= 0:
                # 位相分解の電子温度用 KE 散布 (prompts/52)
                self._cycle_ke[b] += ke_vec

        for name, sp in self.species.items():
            if len(sp.x) == 0:
                continue
            contrib = sp.w[:, None] * self._bary_cached(sp)
            vec = np.bincount(
                self._nidx_cached(sp).ravel(), weights=contrib.ravel(), minlength=self.n_nodes
            )
            self._accum[name] += vec
            if b >= 0:
                if name == "electron":
                    self._cycle_ne[b] += vec
                elif name == "ion":
                    self._cycle_ni[b] += vec
        self._accum_count += 1

    def averaged_density(self) -> dict[str, np.ndarray]:
        """種ごとの時間平均節点密度 [m^-3] を返す。

        節点集中面積 (= Σ隣接要素面積/3) で割って数密度に換算する
        (奥行き 1 m 換算)。
        """
        if self._accum_start is None or self._accum_count == 0:
            raise ValueError("enable_density_accum が呼ばれていないか、まだ積算されていません")
        # 周期スレーブ節点は面積 0 (正準番号で積むため)。ゼロ割を避けてから
        # マスター値をコピーする
        na = np.where(self._node_area > 0.0, self._node_area, 1.0)
        result = {}
        for name, acc in self._accum.items():
            dens = acc / (self._accum_count * na)
            if self.canon is not None:
                dens = dens[self.canon]
            result[name] = dens
        return result

    def averaged_fields(self) -> dict | None:
        """時間平均した2Dフィールド一式を返す (WS done / 検証スクリプト用、prompts/26)。

        平均区間で1ステップも積算していなければ None。
          phi:      節点、時間平均電位 [V]
          e_abs:    要素、時間平均 |E| [V/m] (E ベクトルを平均してから絶対値)
          n_e/n_i:  節点、時間平均密度 [m^-3]
          te_ev:    節点、電子温度 [eV] = (2/3)×平均運動エネルギー (3速度成分)。
                    重み和 0 (粒子なし) の節点は 0
          ion_rate: 節点、電離レート [m^-3 s^-1]
                    (電離イベントの P1 散布を平均時間 × 節点集中面積で規格化)
          avg_steps: 実際に平均したステップ数
        """
        if self._accum_start is None or self._accum_count == 0:
            return None
        cnt = self._accum_count
        dens = self.averaged_density()

        phi_avg = self._accum_phi / cnt
        e_avg = self._accum_e / cnt
        e_abs = np.sqrt(e_avg[:, 0] ** 2 + e_avg[:, 1] ** 2)

        # Te[eV] = (2/3)·(エネルギー和 / 重み和)/e。重み和 0 の節点は 0
        w_e = self._accum["electron"]
        te = np.zeros(self.n_nodes)
        pos = w_e > 0.0
        te[pos] = (2.0 / 3.0) * (self._accum_ke_e[pos] / w_e[pos]) / QE

        # 電離レート = 積算重み / (平均時間 × 節点集中面積)。
        # 周期スレーブ節点は面積 0 なのでゼロ割を避けてマスター値をコピーする
        na = np.where(self._node_area > 0.0, self._node_area, 1.0)
        ion_rate = self._accum_ion / (cnt * self.dt * na)
        if self.canon is not None:
            te = te[self.canon]
            ion_rate = ion_rate[self.canon]

        return {
            "phi": phi_avg,
            "e_abs": e_abs,
            "n_e": dens["electron"],
            "n_i": dens["ion"],
            "te_ev": te,
            "ion_rate": ion_rate,
            "avg_steps": cnt,
        }

    # ---- RF 1周期の位相分解 (prompts/28) ---------------------------------------

    def _snapshot_particles(self, t_step: float) -> None:
        """最後の1周期中、各位相ビンに最初に該当したステップの粒子位置を保存する。

        種ごとに最大 1000 点へ間引く (フレーム送出と同じストライド間引き)。
        """
        b = self._phase_bin(t_step)
        for name in ("electron", "ion"):
            if self._cycle_particles[name][b] is not None:
                continue
            sp = self.species[name]
            n = len(sp.x)
            if n > 1000:
                stride = int(math.ceil(n / 1000))
                pts = sp.x[::stride].copy()
            else:
                pts = sp.x.copy()
            self._cycle_particles[name][b] = pts

    def cycle_data(self) -> dict | None:
        """RF 1周期の位相分解データを返す (WS done の cycle、prompts/28)。

        RF 未設定・phase_bins=0・平均区間で未積算なら None。
          bins:      位相ビン数
          period_s:  RF 周期 [s]
          phi:       (bins, N) 位相分解平均の電位 [V]
          n_e / n_i: (bins, N) 同 密度 [m^-3]
          particles: 種ごとの最後の1周期の生スナップショット (bins × ≤1000点)
        ステップ数 0 の位相ビンはゼロのまま返す (平均区間が1周期以上あれば
        通常発生しない。不足時は started に警告済み)。
        """
        if not self._cycle_enabled or self._cycle_phi is None:
            return None
        if int(self._cycle_count.sum()) == 0:
            return None
        # ステップ数 0 のビンはゼロ除算を避ける (積算値も 0 なので結果は 0)
        cnt = np.maximum(self._cycle_count, 1)[:, None].astype(np.float64)
        phi = self._cycle_phi / cnt
        na = np.where(self._node_area > 0.0, self._node_area, 1.0)
        n_e = self._cycle_ne / (cnt * na[None, :])
        n_i = self._cycle_ni / (cnt * na[None, :])
        # 追加フィールド (prompts/52)。規格化は averaged_fields と同じ規約:
        # |E| は E ベクトルをビン平均してから絶対値 (要素値)、Te = (2/3)(KE和/重み和)/e、
        # 電離レート = 積算重み / (ビン内時間 × 節点集中体積)
        e_avg = self._cycle_e / cnt[:, :, None]  # cnt (bins,1) → (bins,1,1)
        e_abs = np.sqrt(e_avg[:, :, 0] ** 2 + e_avg[:, :, 1] ** 2)  # (bins, M)
        te = np.zeros_like(self._cycle_ne)
        pos = self._cycle_ne > 0.0
        te[pos] = (2.0 / 3.0) * (self._cycle_ke[pos] / self._cycle_ne[pos]) / QE
        ion_rate = self._cycle_ion / (cnt * self.dt * na[None, :])
        if self.canon is not None:
            # 周期境界: スレーブ節点へマスター値をコピー (表示互換)
            n_e = n_e[:, self.canon]
            n_i = n_i[:, self.canon]
            te = te[:, self.canon]
            ion_rate = ion_rate[:, self.canon]

        particles: dict[str, list] = {}
        empty = np.zeros((0, 2))
        for name in ("electron", "ion"):
            snaps = (
                self._cycle_particles[name]
                if self._cycle_particles is not None
                else [None] * self._cycle_bins
            )
            particles[name] = [s if s is not None else empty for s in snaps]

        return {
            "bins": self._cycle_bins,
            "period_s": self._cycle_period,
            "phi": phi,
            "n_e": n_e,
            "n_i": n_i,
            "e_abs": e_abs,        # (bins, M) 要素値 (prompts/52)
            "te_ev": te,           # (bins, N)
            "ion_rate": ion_rate,  # (bins, N)
            "particles": particles,
        }

    # ---- フレーム・実行 -------------------------------------------------------

    def prepare_continue(
        self,
        n_steps: int,
        frame_every: int | None = None,
        avg_steps: int | None = None,
        phase_bins: int | None = None,
    ) -> None:
        """完了/停止後の状態から追加実行の準備をする (prompts/32)。

        維持するもの: 粒子状態 (x, v, w, elem)・表面電荷 q_surf・時刻 t・
        step_count・乱数 Generator (SEE/注入/MCC)・注入状態・累計カウンタ。
        乱数と粒子状態を保持するため「N+M ステップ連続実行」と
        「N ステップ → continue で M ステップ」の粒子状態はビット単位で一致する
        (平均系アキュムレータは読み取り専用で物理状態に影響しない)。

        リセットするもの: 診断 history (追加区間分のみ返す。t は通算時刻のまま
        単調増加)・平均/位相/コレクタのアキュムレータ (次の run_batch が追加区間の
        設定で再有効化する)。avg_steps / phase_bins が None なら前回設定を踏襲する。
        警告 (ωpe·dt 等) は再評価せず前回のものを流用する。
        """
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

        # 診断 history は追加区間分のみ (キー構成は不変)
        self.history = {k: [] for k in self.history}
        # 位相別計測 (prompts/75) も追加区間分のみ返すようリセットする
        self.timing = {k: 0.0 for k in self.timing}
        # 平均・位相・コレクタ系のアキュムレータをリセット
        self._accum_start = None
        self._accum_count = 0
        self._accum = {}
        self._accum_phi = None
        self._accum_e = None
        self._accum_ke_e = None
        self._accum_ion = None
        self._cycle_phi = None
        self._cycle_ne = None
        self._cycle_ni = None
        self._cycle_count = None
        self._cycle_e = None
        self._cycle_ke = None
        self._cycle_ion = None
        self._cycle_particles = None
        self._snap_t_start = math.inf
        self.fields = None
        self.cycle = None
        self.collector_results = None
        self.collector_result = None
        for st in self._collectors_st:
            st["e"], st["a"], st["w"] = [], [], []
            st["count"] = 0
            st["weight"] = 0.0
            st["samples"] = 0
        # 不動種の堆積キャッシュは維持して良い (粒子状態が変わらない限り有効)

    def _make_frame(self, phi: np.ndarray) -> dict:
        """WS 送出用フレーム (JSON 化可能な dict)。粒子は種ごと最大2000点に間引く。"""
        particles = {}
        for name, sp in self.species.items():
            n = len(sp.x)
            if n > MAX_FRAME_PARTICLES:
                stride = int(math.ceil(n / MAX_FRAME_PARTICLES))
                pts = sp.x[::stride]
            else:
                pts = sp.x
            particles[name] = pts.tolist()
        diag = {k: v[-1] for k, v in self.history.items()}
        return {
            "type": "frame",
            "step": self.step_count,
            "t": self.t,
            "phi": phi.tolist(),
            "particles": particles,
            "diag": diag,
        }

    def run_batch(self, callback=None, should_stop=None):
        """n_steps 回実行して (診断履歴, フレーム列) を返す (テスト用同期 API)。

        callback(frame) は frame_every ステップごとに呼ばれる。
        should_stop() が True を返したら中断する (WS の stop コマンド用)。
        完了時に時間平均フィールドを self.fields へ格納する (averaged_fields()
        の結果。WS の done 送出と検証スクリプトが利用する)。
        """
        # 時間平均区間を決めて積算を有効化する (avg_steps、None なら最後の 25%)。
        # enable_density_accum が手動で呼ばれていればその設定を尊重する
        if self._accum_start is None:
            avg = (
                self.pic.avg_steps
                if self.pic.avg_steps is not None
                else max(1, self.pic.n_steps // 4)
            )
            avg = min(avg, self.pic.n_steps)
            self.enable_density_accum(self.step_count + self.pic.n_steps - avg + 1)

        # 粒子スナップショット: 実行の最後の1周期を対象にする (prompts/28)
        if self._cycle_enabled:
            self._cycle_particles = {
                "electron": [None] * self._cycle_bins,
                "ion": [None] * self._cycle_bins,
            }
            self._snap_t_start = (
                self.t + self.pic.n_steps * self.dt - self._cycle_period
            )

        frames: list[dict] = []
        for _ in range(self.pic.n_steps):
            if should_stop is not None and should_stop():
                break
            phi = self.step()
            # 数値発散の検出 (prompts/57): NaN/Inf をフレームに載せると JSON として
            # 不正になりフロントの解析が落ちるため、その前に明確なエラーで止める
            if not (
                math.isfinite(self.history["phi_min"][-1])
                and math.isfinite(self.history["phi_max"][-1])
                and math.isfinite(self.history["ke_e"][-1])
            ):
                raise ValueError(
                    f"数値発散を検出しました (step {self.step_count}: 電位またはエネルギーが"
                    "非有限値)。dt を小さくする、初期密度・注入量を下げる、"
                    "メッシュを細かくする (デバイ長解像) 等を検討してください。"
                    "警告一覧 (ωpe·dt、衝突確率など) も参照してください"
                )
            if self._cycle_enabled and self.t - self.dt >= self._snap_t_start - 1e-30:
                # ステップ開始時刻の位相ビンで、粒子位置 (ステップ終端) を保存する
                self._snapshot_particles(self.t - self.dt)
            if self.step_count % self.pic.frame_every == 0:
                frame = self._make_frame(phi)
                frames.append(frame)
                if callback is not None:
                    callback(frame)
        self.fields = self.averaged_fields()
        self.cycle = self.cycle_data()
        self.collector_results = self._collector_data()
        # 旧単数形の後方互換: 先頭コレクタの結果をエイリアスとして残す
        self.collector_result = (
            self.collector_results[0] if self.collector_results else None
        )
        return self.history, frames
