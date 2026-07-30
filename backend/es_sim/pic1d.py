"""1D PIC/MCC (1d3v) — 専用一様格子ソルバー (prompts/91)。

CCP (容量結合プラズマ) の標準ベンチマーク構成 (Turner et al., Phys. Plasmas 20,
013507 (2013) / eduPIC, Donkó et al., PSST 30, 095017 (2021)) と同じ 1d3v
(空間1D・速度3成分) を、2D FEM-PIC (pic.py) とは完全に独立したモジュールとして
実装する。

一様格子 (n_cells 個のセル、n_nodes = n_cells+1 節点) では、2D の walk 探索
(隣接三角形を辿って所属要素を特定する処理) が丸ごと不要になる — セル番号は
floor(x/dx) だけで直接求まるため。このため particles.py/_numba_kernels.py に
あるような numba カーネルもここでは使わない。理由: 1D は節点・セル数が
2D メッシュよりずっと少なく (典型 128〜1000)、電荷堆積・ポアソン求解・電場計算は
いずれも O(N_particles) か O(n_cells) の軽い numpy ベクトル演算 (bincount 等)
に完全に落ちる。walk のような「1粒子ごとに可変回数のループを回す」処理が無い
ため、numba で書き下すべきホットスポット自体が存在しない。

サイクル (Pic1dSimulation.step):
  1. 電荷堆積: CIC (隣接2節点への線形重み) で節点荷重ベクトルへ散布
  2. ポアソン求解: 三重対角行列は不変 (dx が変わらない限り) なので初期化時に
     一度だけ組み立て、右辺 (電荷密度 + 両端 Dirichlet 電圧) だけ毎ステップ
     更新して scipy.linalg.solve_banded で解く
  3. 電場: 節点 E = -dφ/dx (内点中心差分・端点片側差分) → CIC で粒子へ補間 (gather)
  4. リープフロッグで vx のみ更新 (静電1D・磁場なしのため vy, vz は不変。
     MCC 散乱でのみ変化する)
  5. 位置更新 → 境界吸収 (壁ごと・種ごとに重み和 [実粒子数] で記録) → SEE
     (二次電子放出、壁法線内向き半球等方サンプリング)
  6. MCC: 既存 MccModel (mcc.py) をそのまま流用する

MccModel.collide_electrons/collide_ions の x, elem 引数について (mcc.py を
実際に読んで検証した結果):
  - x はどの分岐でも `x[sub].copy()` (電離生成粒子の位置記録) にしか使われない
    ので、(n,) 1D 形状のままで問題なく動作する (2D の (n,2) 前提のコードは無い)。
    よって MccModel 側の変更は一切不要。
  - elem は非一様ガス場が無ければ候補抽選・プロセス選択には使われないが、
    電離時は `new_elem.append(elem[sub].copy())` を **無条件に** 実行する
    (mcc.py の collide_electrons 内、ionization 分岐)。したがって
    elem=None を渡すと電離が起きた瞬間に IndexError になる。
    1D は非一様ガス場を使わないため elem の値そのものは物理には影響しないが、
    形だけでも実配列 (このセルの粒子のセル番号 clip(floor(x/dx),0,n_cells-1)) を
    常に渡す必要がある (collide_ions 側は elem=None を正しく扱えるので None のままでよい)。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from .fem import EPS0
from .fn import fn_current_density
from .mcc import MccModel
from .particles import ME, MP, QE
from .pic import _eval_rf, _eval_waveform
from .schema import Pic1dElectrode, Pic1dSettings, Project, rf_components

# フレーム送出時の電子位相空間サンプルの最大点数 (pic.py の MAX_FRAME_PARTICLES と同じ趣旨)
MAX_FRAME_PARTICLES = 2000

# 診断 history の列名 (どのステップでも必ずこの全キーを持つ)
_HISTORY_KEYS = (
    "step", "t", "n_e", "n_i", "w_e", "w_i",
    "wall_left_e", "wall_left_i", "wall_right_e", "wall_right_i",
    "ion_events", "see_events", "coll_e",
    "fn_left", "fn_right",
)


@dataclass
class Pic1dSpecies:
    """1D マクロ粒子種。状態はすべて numpy 配列で保持する。"""

    name: str
    q: float          # 電荷 [C]
    m: float          # 質量 [kg]
    x: np.ndarray     # (n,) 位置 [m]
    v: np.ndarray     # (n, 3) 速度 [m/s] (1d3v。E は vx のみに作用)
    w: np.ndarray     # (n,) マクロ重み [m^-2] (単位断面積あたりの実粒子数)


class Pic1dSimulation:
    """1D PIC/MCC (1d3v) シミュレーション本体。geometry/mesh とは無関係に動く。"""

    def __init__(self, project: Project):
        if project.pic1d is None:
            raise ValueError("project.pic1d が指定されていません")
        self.project = project
        self.s: Pic1dSettings = project.pic1d
        s = self.s

        # ---- 格子 ---------------------------------------------------------
        self.gap = float(s.gap_m)
        self.n_cells = int(s.n_cells)
        self.n_nodes = self.n_cells + 1
        self.dx = self.gap / self.n_cells
        self.xg = np.linspace(0.0, self.gap, self.n_nodes)
        # 節点「体積」(実際は単位面積あたりの長さ [m])。内点 dx、端点は半分 (CIC の
        # 対称性から自然に決まる規約。密度出力・電離レート正規化にのみ使う。
        # ポアソン方程式自体は Dirichlet 行で上書きされるため端点の値は使われない)
        self.node_vol = np.full(self.n_nodes, self.dx)
        self.node_vol[0] = self.dx * 0.5
        self.node_vol[-1] = self.dx * 0.5

        # ---- ポアソン行列 (三重対角、banded 形式。dx が不変なので初期化時に1回だけ) --
        # -ε0 φ'' = ρ を中心差分で離散化: ε0/dx^2 (2φ_i - φ_{i-1} - φ_{i+1}) = ρ_i。
        # 両端は Dirichlet (恒等行) で上書きする。scipy.linalg.solve_banded の
        # 格納形式は ab[u + i - j, j] = A[i, j] (u = l = 1)
        k = EPS0 / self.dx**2
        diag = np.full(self.n_nodes, 2.0 * k)
        diag[0] = 1.0
        diag[-1] = 1.0
        upper = np.full(self.n_nodes - 1, -k)  # upper[i] = A[i, i+1]
        upper[0] = 0.0  # 行0は Dirichlet (恒等行)
        lower = np.full(self.n_nodes - 1, -k)  # lower[i] = A[i+1, i]
        lower[-1] = 0.0  # 行 n_nodes-1 は Dirichlet (恒等行)
        ab = np.zeros((3, self.n_nodes))
        ab[0, 1:] = upper
        ab[1, :] = diag
        ab[2, :-1] = lower
        self._poisson_ab = ab

        # ---- dt とプラズマパラメータ ---------------------------------------
        self.m_ion = s.ion_mass_amu * MP  # 2D (pic.py) と同じ amu→kg 換算規約
        wpe = math.sqrt(s.init_density_m3 * QE**2 / (EPS0 * ME))
        self.dt = float(s.dt) if s.dt is not None else 0.1 / wpe

        # ---- 安定性チェック (警告のみ、実行は継続。2D と同じ考え方) -----------
        self.warnings: list[str] = []
        if wpe * self.dt > 0.3:
            self.warnings.append(
                f"ωpe·dt = {wpe * self.dt:.3g} > 0.3: 時間刻みが粗すぎます (数値不安定の恐れ)"
            )
        if s.init_te_ev > 0.0:
            lam_d = math.sqrt(EPS0 * s.init_te_ev * QE / (s.init_density_m3 * QE**2))
            if self.dx > 3.0 * lam_d:
                self.warnings.append(
                    f"セル幅 {self.dx:.3g} m > 3×デバイ長 {lam_d:.3g} m: "
                    "格子がデバイ長を解像していません (n_cells を増やしてください)"
                )

        # ---- RF 1周期の位相分解の基本周波数 (prompts/93) -----------------------
        # 優先順位: ①左右電極の voltage_rf 成分 (デュアル周波数含む) の freq_hz の
        # 最小値 (= 基本波。低周波1周期に高周波の複数サイクルが収まるため、2D の
        # _find_rf_freq と同じ考え方で min を使う) → ②voltage_rf が一つも無ければ
        # 従来どおり CSV 波形 (waveforms) の freq_hz の最小値。voltage_rf が本来の
        # RF 駆動の表現であり CSV 波形はその近似 (サンプル補間) or 任意波形用途な
        # ので、両方指定された電極では voltage_rf 側の周波数を優先する
        # (2D の _find_rf_freq は voltage_rf と waveform を区別せず全体最小を取るが、
        # 1D では voltage_rf を優先することを明示的に選ぶ)。どちらも無ければ None
        # (cycle 無効)。
        rf_freqs = [
            c.freq_hz
            for c in (*rf_components(s.left.voltage_rf), *rf_components(s.right.voltage_rf))
        ]
        if rf_freqs:
            self._cycle_freq = min(rf_freqs)
        else:
            wf_freqs = [wf.freq_hz for wf in s.left.waveforms] + [wf.freq_hz for wf in s.right.waveforms]
            self._cycle_freq = min(wf_freqs) if wf_freqs else None
        self._cycle_bins = int(s.phase_bins)
        self._cycle_enabled = self._cycle_freq is not None and self._cycle_bins > 0
        self._cycle_period = 1.0 / self._cycle_freq if self._cycle_enabled else 0.0

        # ---- 初期プラズマ装荷 -----------------------------------------------
        # 電子・イオンを同一位置に装荷して初期の厳密な電気的中性を保つ (pic.py と
        # 同じ quiet start の考え方)
        rng = np.random.default_rng(s.seed)
        n_macro = s.n_macro
        w0 = s.init_density_m3 * self.gap / n_macro  # マクロ重み [m^-2]
        x0 = rng.uniform(0.0, self.gap, size=n_macro)
        sigma_e = math.sqrt(s.init_te_ev * QE / ME)
        sigma_i = math.sqrt(s.init_ti_ev * QE / self.m_ion)
        v_e = rng.normal(0.0, sigma_e, size=(n_macro, 3))
        v_i = rng.normal(0.0, sigma_i, size=(n_macro, 3))
        self.species: dict[str, Pic1dSpecies] = {
            "electron": Pic1dSpecies("electron", -QE, ME, x0.copy(), v_e, np.full(n_macro, w0)),
            "ion": Pic1dSpecies("ion", QE, self.m_ion, x0.copy(), v_i, np.full(n_macro, w0)),
        }

        # ---- SEE (二次電子放出) ----------------------------------------------
        mcc_seed = s.mcc.seed if s.mcc is not None else 0
        self._see_rng = np.random.default_rng(mcc_seed + 12345)
        self._see_speed = math.sqrt(2.0 * s.see_energy_ev * QE / ME)

        # ---- FN 電界放出 (prompts/95、Fowler–Nordheim) -------------------------
        # 左右電極それぞれ fn=None なら放出処理を一切行わない (従来動作と完全ビット
        # 不変)。1D は電極がちょうど1点 (左端 x=0 / 右端 x=gap) なので、2D PIC
        # (pic.py の FnSurface) のような放出面のセグメント分割・位置サンプリング
        # 乱数が不要 — 毎ステップの放出数は端数の決定論的キャリーのみで決まる。
        # また 1D のマクロ重みは [m^-2] (単位断面積あたりの実粒子数) 単位そのもの
        # なので、2D のような奥行き/周方向の面積換算は不要 (J·dt/e がそのまま
        # 放出実電子数密度になる)。
        self.fn_left = s.left.fn
        self.fn_right = s.right.fn
        # 壁からわずかに内側へ置くオフセット (2D の fn.py FnSurface.delta と同じ考え方:
        # 局所メッシュ規模の 1e-3 倍。1D の局所規模はセル幅 dx に対応する)
        self._fn_delta = 1e-3 * self.dx
        self._fn_w: dict[str, float] = {}
        self._fn_speed: dict[str, float] = {}
        self._fn_frac: dict[str, float] = {"left": 0.0, "right": 0.0}  # 端数の決定論的キャリー
        self.fn_events: dict[str, int] = {"left": 0, "right": 0}  # 累計放出マクロ数 (continue でも維持)
        self.fn_total_w: dict[str, float] = {"left": 0.0, "right": 0.0}  # 累計放出重み [m^-2]
        self._fn_accum_j: dict[str, float] = {"left": 0.0, "right": 0.0}  # 平均区間内の J 積算
        self.fn: dict | None = None  # done result 用サマリ (run_batch で確定)
        for side, fn in (("left", self.fn_left), ("right", self.fn_right)):
            if fn is None:
                continue
            self._fn_w[side] = float(fn.macro_weight) if fn.macro_weight is not None else w0
            self._fn_speed[side] = (
                math.sqrt(2.0 * fn.init_energy_ev * QE / ME) if fn.init_energy_ev > 0.0 else 0.0
            )

        # ---- MCC (既存 MccModel を流用、prompts/19) ---------------------------
        # 1D は非一様背景ガス場 (DSMC 連成) 未対応なので gas_field は常に None
        # (schema 側で use_dsmc_gas=True を弾いているため、mcc は常に一様ガスとして使う)
        self.mcc = MccModel(s.mcc, self.m_ion, None) if s.mcc is not None else None

        # ---- EEDF/EEPF 領域 (2D の EedfRegion と同じ規約、prompts/85 の1D版) ---
        self._eedf_st: list[dict] = []
        for reg in s.eedf_regions:
            x0_, x1_ = sorted((reg.x1, reg.x2))
            auto = reg.e_max_ev is None
            self._eedf_st.append(
                {
                    "x0": x0_, "x1": x1_, "bins": reg.bins, "label": reg.label,
                    "auto": auto, "e_max": reg.e_max_ev,
                    "hist": None if auto else np.zeros(reg.bins),
                    "sum_w": 0.0, "sum_we": 0.0, "overflow_w": 0.0,
                }
            )
        self.eedf_results: list[dict] | None = None

        # ---- 診断・時刻 --------------------------------------------------------
        self.t = 0.0
        self.step_count = 0
        # 壁ごと・種ごとの累計吸収数 (重み和 = 実粒子数。continue でも維持する)
        self.wall: dict[str, dict[str, float]] = {
            "left": {"electron": 0.0, "ion": 0.0},
            "right": {"electron": 0.0, "ion": 0.0},
        }
        self.ion_events = 0     # 電離の累計 (マクロイベント数)
        self.see_events = 0.0   # SEE 放出の累計 (重み和 = 実電子数)
        self.coll_e = 0         # 電子 MCC 衝突の累計 (弾性+励起+電離)
        self.history: dict[str, list] = {k: [] for k in _HISTORY_KEYS}
        self.timing: dict[str, float] = {
            "deposit": 0.0,  # 電荷堆積 (CIC)
            "field": 0.0,    # ポアソン求解 + 電場計算
            "push": 0.0,     # リープフロッグ押し出し・境界吸収・SEE
            "mcc": 0.0,      # MCC 衝突
            "other": 0.0,    # 時間平均積算・EEDF 集計・診断記録
        }

        # ---- 時間平均・位相分解アキュムレータ (run_batch/enable_density_accum で確保) ---
        self._accum_start: int | None = None
        self._accum_count = 0
        self._accum_phi: np.ndarray | None = None
        self._accum_e: np.ndarray | None = None
        self._accum_ke_e: np.ndarray | None = None
        self._accum_ion: np.ndarray | None = None
        self._accum_w: dict[str, np.ndarray] = {}
        self._cycle_phi: np.ndarray | None = None
        self._cycle_ne: np.ndarray | None = None
        self._cycle_ni: np.ndarray | None = None
        self._cycle_ion: np.ndarray | None = None
        self._cycle_count: np.ndarray | None = None
        self.fields: dict | None = None
        self.cycle: dict | None = None
        self._run_t0 = 0.0  # run_batch 開始時刻 (フレームの elapsed_s 用)

        # ---- 初期半ステップ後退キック (t=0 の場で v を -dt/2 へ、vx のみ) --------
        f0 = self._deposit()
        phi0 = self._solve_phi(f0, 0.0)
        ex0 = self._e_field(phi0)
        for sp in self.species.values():
            if len(sp.x):
                e_at = self._cic_gather(ex0, sp.x)
                sp.v[:, 0] -= 0.5 * self.dt * (sp.q / sp.m) * e_at

    # ---- CIC (電荷/物理量の堆積・補間) ----------------------------------------

    def _cic_index(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """粒子位置 → (左側節点インデックス, 右側節点への重み frac) の CIC 係数。"""
        idx0 = np.clip(np.floor(x / self.dx).astype(np.int64), 0, self.n_cells - 1)
        frac = np.clip(x / self.dx - idx0, 0.0, 1.0)
        return idx0, frac

    def _cic_deposit(self, x: np.ndarray, values: np.ndarray) -> np.ndarray:
        """粒子ごとの量 (電荷・重み・エネルギー等) を CIC で節点へ散布する。"""
        if len(x) == 0:
            return np.zeros(self.n_nodes)
        idx0, frac = self._cic_index(x)
        out = np.bincount(idx0, weights=values * (1.0 - frac), minlength=self.n_nodes)
        out += np.bincount(idx0 + 1, weights=values * frac, minlength=self.n_nodes)
        return out

    def _cic_gather(self, node_vals: np.ndarray, x: np.ndarray) -> np.ndarray:
        """節点値を CIC で粒子位置へ補間する (電場の gather)。"""
        if len(x) == 0:
            return np.zeros(0)
        idx0, frac = self._cic_index(x)
        return (1.0 - frac) * node_vals[idx0] + frac * node_vals[idx0 + 1]

    def _deposit(self) -> np.ndarray:
        """全種の電荷堆積 (節点荷重ベクトル、単位: [C/m] = 単位断面積あたりの電荷)。"""
        f = np.zeros(self.n_nodes)
        for sp in self.species.values():
            if len(sp.x):
                f += self._cic_deposit(sp.x, sp.q * sp.w)
        return f

    # ---- 場 --------------------------------------------------------------

    def _electrode_voltage(self, elec: Pic1dElectrode, t: float) -> float:
        """電極電圧 V(t) = v_dc + Σ RF sin + Σ waveforms(t) (prompts/93)。

        RF 成分は pic.py の _eval_rf (2D の _dirichlet_values と同じ式・位相規約)、
        CSV 波形は pic.py の _eval_waveform をそのまま流用する。voltage_rf=None
        (rf_components が空リスト) なら _eval_rf は 0.0 を返すため、waveforms のみの
        従来経路とビット不変になる。
        """
        v = elec.v_dc + _eval_rf(rf_components(elec.voltage_rf), t)
        for wf in elec.waveforms:
            v += float(_eval_waveform(wf.phase, wf.v, wf.freq_hz, t))
        return float(v)

    def _solve_phi(self, f_dep: np.ndarray, t: float) -> np.ndarray:
        """ポアソン求解。行列は不変 (__init__ で構築済み)、右辺のみ毎回組み立てて解く。"""
        rhs = np.empty(self.n_nodes)
        rhs[1:-1] = f_dep[1:-1] / self.dx  # 内点の電荷密度 ρ_i = f_dep[i]/dx
        rhs[0] = self._electrode_voltage(self.s.left, t)
        rhs[-1] = self._electrode_voltage(self.s.right, t)
        return solve_banded((1, 1), self._poisson_ab, rhs)

    def _e_field(self, phi: np.ndarray) -> np.ndarray:
        """節点電場 E = -dφ/dx (内点中心差分、端点片側差分)。"""
        ex = np.empty(self.n_nodes)
        ex[1:-1] = -(phi[2:] - phi[:-2]) / (2.0 * self.dx)
        ex[0] = -(phi[1] - phi[0]) / self.dx
        ex[-1] = -(phi[-1] - phi[-2]) / self.dx
        return ex

    # ---- FN 電界放出 (prompts/95、Fowler–Nordheim) -----------------------------

    def _emit_fn(self, ex: np.ndarray) -> tuple[float, float, float, float]:
        """このステップの表面電界から左右電極の FN 電界放出を行う。

        表面電界 F は現ステップの場 ex (端点値) から「電子を真空側へ引き出す
        向き」成分のみを取る (fn.py と同じ規約: 左電極は真空側 n̂=+x なので
        F=-ex[0]、右電極は n̂=-x なので F=+ex[-1]。負なら放出ゼロ)。
        放出実電子数 J·dt/e をマクロ重み w で割った端数を電極ごとに決定論的に
        持ち越す (pic.py の _emit_fn と同じ「時間平均で正確に J を再現する」
        考え方。1D は電極が1点なので位置サンプリングの乱数自体が不要)。
        新粒子は放出法線方向のみの速さ (init_energy_ev 相当) を持ち、
        リープフロッグの半ステップ補正は行わない (1D の SEE/電離生成粒子と同じ
        流儀に揃える。pic.py の FN/注入は半ステップ補正を行うが、1D 側の新粒子
        追加処理はいずれも半ステップ補正をしていないため、それに合わせる)。

        戻り値: (j_left, j_right, w_left, w_right) — このステップの放出電流密度
        [A/m^2] と放出重み [m^-2] (fn 未設定の側は常に 0.0)。
        """
        el = self.species["electron"]
        j = {"left": 0.0, "right": 0.0}
        w_emit = {"left": 0.0, "right": 0.0}
        for side, fn, e_end, sign, wall_x in (
            ("left", self.fn_left, float(ex[0]), 1.0, 0.0),
            ("right", self.fn_right, float(ex[-1]), -1.0, self.gap),
        ):
            if fn is None:
                continue
            f_surf = max(0.0, -sign * e_end)
            j[side] = float(fn_current_density(np.array([f_surf]), fn.phi_ev, fn.beta)[0])
            w_fn = self._fn_w[side]
            quota = j[side] * self.dt / (QE * w_fn) + self._fn_frac[side]
            n_emit = int(math.floor(quota))
            self._fn_frac[side] = quota - n_emit
            if n_emit <= 0:
                continue
            x_new = np.full(n_emit, wall_x + sign * self._fn_delta)
            v_new = np.zeros((n_emit, 3))
            v_new[:, 0] = sign * self._fn_speed[side]
            el.x = np.concatenate([el.x, x_new])
            el.v = np.concatenate([el.v, v_new])
            el.w = np.concatenate([el.w, np.full(n_emit, w_fn)])
            w_emit[side] = n_emit * w_fn
            self.fn_events[side] += n_emit
            self.fn_total_w[side] += w_emit[side]
        return j["left"], j["right"], w_emit["left"], w_emit["right"]

    def fn_summary(self) -> dict | None:
        """done result 用の FN 放出サマリ (prompts/95)。

        j_avg は時間平均プロファイル (averaged_fields) と同じ平均区間
        (_accum_start 以降) の平均放出電流密度 [A/m^2]、total_w は放出開始からの
        累積放出重み [m^-2] (continue をまたいでも維持される)。左右いずれの
        電極も fn 未設定なら None。
        """
        if self.fn_left is None and self.fn_right is None:
            return None
        cnt = self._accum_count if self._accum_count > 0 else 1
        out: dict[str, dict | None] = {}
        for side, fn in (("left", self.fn_left), ("right", self.fn_right)):
            if fn is None:
                out[side] = None
                continue
            out[side] = {
                "j_avg": self._fn_accum_j[side] / cnt,
                "total_w": self.fn_total_w[side],
            }
        return out

    # ---- 位相分解 -----------------------------------------------------------

    def _phase_bin(self, t: float) -> int:
        frac = (t % self._cycle_period) / self._cycle_period
        return min(int(frac * self._cycle_bins), self._cycle_bins - 1)

    # ---- SEE (二次電子放出) ---------------------------------------------------

    def _sample_hemisphere(self, rng: np.random.Generator, n: int, sign: float) -> np.ndarray:
        """壁法線内向き半球等方サンプリング。

        mcc.py の全球等方散乱 (cosχ 一様・方位角一様) と同じ手法で単位球面上の
        方向を一様に抽選し、法線成分 (vx) の符号だけ壁の内向きへ強制する
        (sign=+1: 左壁、内向き=+x。sign=-1: 右壁、内向き=-x)。
        """
        mu = rng.uniform(-1.0, 1.0, size=n)
        phi = rng.uniform(0.0, 2.0 * math.pi, size=n)
        s = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
        dx_dir = np.abs(mu) * sign
        dy_dir = s * np.cos(phi)
        dz_dir = s * np.sin(phi)
        return np.stack([dx_dir, dy_dir, dz_dir], axis=1)

    # ---- EEDF/EEPF 領域 (prompts/85 の1D版) ------------------------------------

    def _accumulate_eedf(self) -> None:
        if not self._eedf_st:
            return
        el = self.species["electron"]
        if len(el.x) == 0:
            return
        x = el.x
        e_all: np.ndarray | None = None
        for st in self._eedf_st:
            mask = (x >= st["x0"]) & (x <= st["x1"])
            if not np.any(mask):
                continue
            if e_all is None:
                v = el.v
                e_all = 0.5 * ME * (v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2) / QE
            e_sel = e_all[mask]
            w_sel = el.w[mask]
            if st["auto"] and st["hist"] is None:
                e_max = float(e_sel.max()) * 1.2 if e_sel.size else 30.0
                st["e_max"] = e_max if e_max > 0.0 else 30.0
                st["hist"] = np.zeros(st["bins"])
            e_max = st["e_max"]
            in_range = e_sel <= e_max
            if np.any(in_range):
                idx = np.minimum(
                    (e_sel[in_range] / e_max * st["bins"]).astype(np.int64), st["bins"] - 1
                )
                st["hist"] += np.bincount(idx, weights=w_sel[in_range], minlength=st["bins"])
                st["sum_w"] += float(w_sel[in_range].sum())
                st["sum_we"] += float((w_sel[in_range] * e_sel[in_range]).sum())
            if not np.all(in_range):
                st["overflow_w"] += float(w_sel[~in_range].sum())

    def _eedf_data(self) -> list[dict] | None:
        if not self._eedf_st:
            return None
        n_samples = self._accum_count
        out = []
        for st in self._eedf_st:
            bins = st["bins"]
            e_max = st["e_max"] if st["e_max"] is not None else 30.0
            hist = st["hist"] if st["hist"] is not None else np.zeros(bins)
            d_e = e_max / bins
            e_centers = (np.arange(bins) + 0.5) * d_e
            sum_w = st["sum_w"]
            if sum_w > 0.0:
                f = hist / (sum_w * d_e)
                mean_e = st["sum_we"] / sum_w
            else:
                f = np.zeros(bins)
                mean_e = 0.0
            denom = sum_w + st["overflow_w"]
            overflow_frac = st["overflow_w"] / denom if denom > 0.0 else 0.0
            out.append(
                {
                    "label": st["label"],
                    "e_centers": e_centers,
                    "f": f,
                    "mean_energy_ev": mean_e,
                    "t_eff_ev": (2.0 / 3.0) * mean_e,
                    "total_weight": sum_w,
                    "overflow_frac": overflow_frac,
                    "n_samples": n_samples,
                }
            )
        return out

    # ---- 時間平均アキュムレータ -------------------------------------------------

    def enable_density_accum(self, start_step: int) -> None:
        self._accum_start = int(start_step)
        self._accum_count = 0
        self._accum_phi = None
        self._accum_e = None
        self._accum_ke_e = None
        self._accum_ion = None
        self._accum_w = {}
        self._cycle_phi = None
        self._cycle_ne = None
        self._cycle_ni = None
        self._cycle_ion = None
        self._cycle_count = None
        self._fn_accum_j = {"left": 0.0, "right": 0.0}  # j_avg (fn_summary) の平均区間積算

    def _ensure_accumulators(self) -> None:
        if self._accum_phi is not None:
            return
        self._accum_phi = np.zeros(self.n_nodes)
        self._accum_e = np.zeros(self.n_nodes)
        self._accum_ke_e = np.zeros(self.n_nodes)
        self._accum_ion = np.zeros(self.n_nodes)
        self._accum_w = {name: np.zeros(self.n_nodes) for name in self.species}
        if self._cycle_enabled:
            nb = self._cycle_bins
            self._cycle_phi = np.zeros((nb, self.n_nodes))
            self._cycle_ne = np.zeros((nb, self.n_nodes))
            self._cycle_ni = np.zeros((nb, self.n_nodes))
            self._cycle_ion = np.zeros((nb, self.n_nodes))
            self._cycle_count = np.zeros(nb, dtype=np.int64)

    def _accumulate_fields(self, phi: np.ndarray, ex: np.ndarray, t_step: float) -> None:
        self._accum_phi += phi
        self._accum_e += ex
        b = -1
        if self._cycle_phi is not None:
            b = self._phase_bin(t_step)
            self._cycle_phi[b] += phi
            self._cycle_count[b] += 1
        el = self.species["electron"]
        if len(el.x):
            ke_p = 0.5 * ME * (el.v[:, 0] ** 2 + el.v[:, 1] ** 2 + el.v[:, 2] ** 2)
            self._accum_ke_e += self._cic_deposit(el.x, el.w * ke_p)
        for name, sp in self.species.items():
            if len(sp.x) == 0:
                continue
            vec = self._cic_deposit(sp.x, sp.w)
            self._accum_w[name] += vec
            if b >= 0:
                if name == "electron":
                    self._cycle_ne[b] += vec
                elif name == "ion":
                    self._cycle_ni[b] += vec
        self._accum_count += 1

    def averaged_fields(self) -> dict | None:
        """時間平均プロファイル一式 (WS done / ResultsBundle 用)。

        phi/e: 節点、時間平均電位 [V] / 電場 [V/m]
        n_e/n_i: 節点、時間平均密度 [m^-3]
        t_e: 節点、電子温度 [eV] = (2/3)×平均運動エネルギー (3速度成分)
        ionization: 節点、電離レート [m^-3 s^-1]
        """
        if self._accum_start is None or self._accum_count == 0:
            return None
        cnt = self._accum_count
        phi_avg = self._accum_phi / cnt
        e_avg = self._accum_e / cnt
        n_e = self._accum_w["electron"] / (cnt * self.node_vol)
        n_i = self._accum_w["ion"] / (cnt * self.node_vol)
        w_e = self._accum_w["electron"]
        te = np.zeros(self.n_nodes)
        pos = w_e > 0.0
        te[pos] = (2.0 / 3.0) * (self._accum_ke_e[pos] / w_e[pos]) / QE
        ion_rate = self._accum_ion / (cnt * self.dt * self.node_vol)
        return {
            "phi": phi_avg, "e": e_avg, "n_e": n_e, "n_i": n_i,
            "t_e": te, "ionization": ion_rate, "avg_steps": cnt,
        }

    def cycle_data(self) -> dict | None:
        """RF 1周期の位相分解データ (bins, freq_hz, phi[][], n_e[][], n_i[][])。"""
        if not self._cycle_enabled or self._cycle_phi is None:
            return None
        if int(self._cycle_count.sum()) == 0:
            return None
        cnt = np.maximum(self._cycle_count, 1)[:, None].astype(np.float64)
        phi = self._cycle_phi / cnt
        n_e = self._cycle_ne / (cnt * self.node_vol[None, :])
        n_i = self._cycle_ni / (cnt * self.node_vol[None, :])
        return {
            "bins": self._cycle_bins, "freq_hz": self._cycle_freq,
            "phi": phi, "n_e": n_e, "n_i": n_i,
        }

    # ---- 1ステップ ------------------------------------------------------------

    def step(self) -> np.ndarray:
        """PIC 1サイクル。時刻 t_n の場を解き、粒子を t_{n+1} へ進める。"""
        t = self.t
        dt = self.dt
        # このステップが時間平均区間に入るか (step_count はステップ末尾で +1 される)
        accumulating = (
            self._accum_start is not None and self.step_count + 1 >= self._accum_start
        )
        if accumulating:
            self._ensure_accumulators()

        # 1. 電荷堆積 → 2. ポアソン求解 (V(t) で両端 Dirichlet 更新)
        t0 = time.perf_counter()
        f_dep = self._deposit()
        t1 = time.perf_counter()
        self.timing["deposit"] += t1 - t0
        phi = self._solve_phi(f_dep, t)
        ex = self._e_field(phi)
        t2 = time.perf_counter()
        self.timing["field"] += t2 - t1

        # 3-5. リープフロッグ (vx のみ) → 位置更新 → 境界吸収・SEE
        el = self.species["electron"]
        io = self.species["ion"]
        see_x: list[np.ndarray] = []
        see_v: list[np.ndarray] = []
        see_w: list[np.ndarray] = []
        for sp in self.species.values():
            if len(sp.x) == 0:
                continue
            e_at = self._cic_gather(ex, sp.x)
            sp.v[:, 0] += (sp.q / sp.m) * e_at * dt
            x_new = sp.x + sp.v[:, 0] * dt
            left_mask = x_new < 0.0
            right_mask = x_new > self.gap
            absorbed = left_mask | right_mask
            if np.any(absorbed):
                self.wall["left"][sp.name] += float(sp.w[left_mask].sum())
                self.wall["right"][sp.name] += float(sp.w[right_mask].sum())
                if sp.name == "ion":
                    for mask, sign, gamma, wall_x in (
                        (left_mask, 1.0, self.s.left.see_gamma, 0.0),
                        (right_mask, -1.0, self.s.right.see_gamma, self.gap),
                    ):
                        if gamma <= 0.0 or not np.any(mask):
                            continue
                        cand_idx = np.nonzero(mask)[0]
                        accept = self._see_rng.random(len(cand_idx)) < gamma
                        if not np.any(accept):
                            continue
                        sel = cand_idx[accept]
                        k = int(len(sel))
                        see_x.append(np.full(k, wall_x))
                        see_v.append(self._see_speed * self._sample_hemisphere(self._see_rng, k, sign))
                        see_w.append(sp.w[sel].copy())
                        self.see_events += float(sp.w[sel].sum())
                keep = ~absorbed
                sp.x = x_new[keep]
                sp.v = sp.v[keep]
                sp.w = sp.w[keep]
            else:
                sp.x = x_new
        if see_x:
            el.x = np.concatenate([el.x, np.concatenate(see_x)])
            el.v = np.concatenate([el.v, np.concatenate(see_v)])
            el.w = np.concatenate([el.w, np.concatenate(see_w)])
        t3 = time.perf_counter()
        self.timing["push"] += t3 - t2

        # 6. MCC (mcc.py 参照。elem は電離時の記録に無条件で使われるため常に実配列を渡す)
        if self.mcc is not None:
            if len(el.x):
                elem_e, _ = self._cic_index(el.x)
                res = self.mcc.collide_electrons(el.x, el.v, el.w, elem_e, dt)
                self.coll_e += res.n_coll
                self.ion_events += res.n_ionization
                if res.new_x is not None:
                    if accumulating:
                        self._accum_ion += self._cic_deposit(res.new_x, res.new_w)
                        if self._cycle_ion is not None:
                            self._cycle_ion[self._phase_bin(t)] += self._cic_deposit(
                                res.new_x, res.new_w
                            )
                    el.x = np.concatenate([el.x, res.new_x])
                    el.v = np.concatenate([el.v, res.new_v_e])
                    el.w = np.concatenate([el.w, res.new_w])
                    io.x = np.concatenate([io.x, res.new_x])
                    io.v = np.concatenate([io.v, res.new_v_i])
                    io.w = np.concatenate([io.w, res.new_w])
            if len(io.x):
                elem_i, _ = self._cic_index(io.x)
                self.mcc.collide_ions(io.v, dt, elem_i)
        t4 = time.perf_counter()
        self.timing["mcc"] += t4 - t3

        # 7. FN 電界放出 (prompts/95)。表面電界はこのステップの場 (ex) を使う。
        # 両電極とも fn=None ならこの呼び出し自体を省略する (従来経路とビット不変)
        if self.fn_left is not None or self.fn_right is not None:
            fn_j_left, fn_j_right, fn_w_left, fn_w_right = self._emit_fn(ex)
        else:
            fn_j_left = fn_j_right = fn_w_left = fn_w_right = 0.0

        self.t = t + dt
        self.step_count += 1

        if accumulating:
            self._accumulate_fields(phi, ex, t)
            self._accumulate_eedf()
            # j_avg (fn_summary) 用: 時間平均プロファイルと同じ平均区間で積算する
            if self.fn_left is not None:
                self._fn_accum_j["left"] += fn_j_left
            if self.fn_right is not None:
                self._fn_accum_j["right"] += fn_j_right

        h = self.history
        h["step"].append(self.step_count)
        h["t"].append(self.t)
        h["n_e"].append(len(el.x))
        h["n_i"].append(len(io.x))
        h["w_e"].append(float(el.w.sum()) if len(el.w) else 0.0)
        h["w_i"].append(float(io.w.sum()) if len(io.w) else 0.0)
        h["wall_left_e"].append(self.wall["left"]["electron"])
        h["wall_left_i"].append(self.wall["left"]["ion"])
        h["wall_right_e"].append(self.wall["right"]["electron"])
        h["wall_right_i"].append(self.wall["right"]["ion"])
        h["ion_events"].append(self.ion_events)
        h["see_events"].append(self.see_events)
        h["coll_e"].append(self.coll_e)
        h["fn_left"].append(fn_w_left)
        h["fn_right"].append(fn_w_right)

        t5 = time.perf_counter()
        self.timing["other"] += t5 - t4
        return phi

    # ---- フレーム・実行 ---------------------------------------------------------

    def _make_frame(self, phi: np.ndarray) -> dict:
        el = self.species["electron"]
        io = self.species["ion"]
        n_e = self._cic_deposit(el.x, el.w) / self.node_vol
        n_i = self._cic_deposit(io.x, io.w) / self.node_vol
        n = len(el.x)
        if n > MAX_FRAME_PARTICLES:
            stride = int(math.ceil(n / MAX_FRAME_PARTICLES))
            sample_x = el.x[::stride]
            sample_vx = el.v[::stride, 0]
        else:
            sample_x = el.x
            sample_vx = el.v[:, 0]
        counts = {k: v[-1] for k, v in self.history.items()}
        return {
            "type": "frame",
            "step": self.step_count,
            "t": self.t,
            "phi": phi.tolist(),
            "n_e": n_e.tolist(),
            "n_i": n_i.tolist(),
            "counts": counts,
            "elapsed_s": time.perf_counter() - self._run_t0,
            "sample": {"x": sample_x.tolist(), "vx": sample_vx.tolist()},
        }

    def run_batch(self, callback=None, should_stop=None, store_frames: bool = True):
        """n_steps 回実行して (history, フレーム列) を返す。2D の run_batch と同じ設計。"""
        if self._accum_start is None:
            avg = self.s.avg_steps if self.s.avg_steps is not None else max(1, self.s.n_steps // 4)
            avg = min(avg, self.s.n_steps)
            self.enable_density_accum(self.step_count + self.s.n_steps - avg + 1)

        frames: list[dict] = []
        n_steps_total = self.s.n_steps
        self._run_t0 = time.perf_counter()
        for _ in range(n_steps_total):
            if should_stop is not None and should_stop():
                break
            phi = self.step()
            if not (
                np.all(np.isfinite(phi))
                and math.isfinite(self.history["w_e"][-1])
                and math.isfinite(self.history["w_i"][-1])
            ):
                raise ValueError(
                    f"数値発散を検出しました (step {self.step_count}: 電位または重みが"
                    "非有限値)。dt を小さくする、初期密度を下げる、n_cells を増やす"
                    "(デバイ長解像) 等を検討してください"
                )
            if self.step_count % self.s.frame_every == 0:
                frame = self._make_frame(phi)
                if store_frames:
                    frames.append(frame)
                if callback is not None:
                    callback(frame)

        self.fields = self.averaged_fields()
        self.cycle = self.cycle_data()
        self.eedf_results = self._eedf_data()
        self.fn = self.fn_summary()
        return self.history, frames

    def prepare_continue(
        self,
        extra_steps: int,
        frame_every: int | None = None,
        avg_steps: int | None = None,
        phase_bins: int | None = None,
    ) -> None:
        """完了/停止後の状態から追加実行の準備をする (2D の prepare_continue と同じ設計)。

        維持するもの: 粒子状態 (x, v, w)・時刻 t・step_count・乱数 Generator
        (SEE/MCC)・累計カウンタ (wall/ion_events/see_events/coll_e/fn_events/fn_total_w
        および端数キャリー _fn_frac — FN のキャリーも維持しないと continue で放出数が
        ずれ、run(n+m) とビット一致しなくなる)。
        リセットするもの: 診断 history (追加区間分のみ)・timing・平均/位相/EEDF/FN j_avg の
        アキュムレータ。これにより run(n) → continue(m) は run(n+m) とビット一致する
        (平均区間の開始ステップは常に「現在の step_count + 今回の n_steps - avg + 1」
        という絶対ステップ番号で決まるため、区間の切り方によらず同じ結果になる)。
        """
        self.s.n_steps = int(extra_steps)
        if frame_every is not None:
            self.s.frame_every = int(frame_every)
        if avg_steps is not None:
            self.s.avg_steps = int(avg_steps)
        if phase_bins is not None:
            self.s.phase_bins = int(phase_bins)
            self._cycle_bins = int(phase_bins)
            self._cycle_enabled = self._cycle_freq is not None and self._cycle_bins > 0
            self._cycle_period = 1.0 / self._cycle_freq if self._cycle_enabled else 0.0

        self.history = {k: [] for k in self.history}
        self.timing = {k: 0.0 for k in self.timing}
        self._accum_start = None
        self._accum_count = 0
        self._accum_phi = None
        self._accum_e = None
        self._accum_ke_e = None
        self._accum_ion = None
        self._accum_w = {}
        self._cycle_phi = None
        self._cycle_ne = None
        self._cycle_ni = None
        self._cycle_ion = None
        self._cycle_count = None
        self.fields = None
        self.cycle = None
        self.eedf_results = None
        self.fn = None
        self._fn_accum_j = {"left": 0.0, "right": 0.0}
        for st in self._eedf_st:
            st["sum_w"] = 0.0
            st["sum_we"] = 0.0
            st["overflow_w"] = 0.0
            st["hist"] = None if st["auto"] else np.zeros(st["bins"])
            if st["auto"]:
                st["e_max"] = None


# ---- 結果バンドル組み立て (server.py / batch.py 共通、prompts/96) -----------------------
#
# /ws/pic1d の done.result と batch/sweep の ResultsBundle.pic1d は同じ形にする必要がある
# (フロントの Pic1dResult 型・applyLoadedProject の復元経路を両方から使い回すため)。
# 循環 import を避けるため (server.py は pic1d.py を import するが逆はしない)、
# ビルダー本体をここに置き server.py / batch.py の双方から import する。


def build_pic1d_result(sim: Pic1dSimulation, elapsed_s: float) -> dict:
    """done.result (= ResultsBundle.pic1d に格納する想定の形) を組み立てる。

    2D の _build_results_bundle (batch.py) と同じ考え方: done で返す内容を
    そのまま保存し、そのまま読み込んで復元できる自己完結な dict にする。
    """
    profiles = None
    if sim.fields is not None:
        f = sim.fields
        profiles = {
            "x": sim.xg.tolist(),
            "phi": f["phi"].tolist(),
            "e": f["e"].tolist(),
            "n_e": f["n_e"].tolist(),
            "n_i": f["n_i"].tolist(),
            "t_e": f["t_e"].tolist(),
            "ionization": f["ionization"].tolist(),
            "avg_steps": f["avg_steps"],
        }
    cycle = None
    if sim.cycle is not None:
        c = sim.cycle
        cycle = {
            "bins": c["bins"],
            "freq_hz": c["freq_hz"],
            "phi": c["phi"].tolist(),
            "n_e": c["n_e"].tolist(),
            "n_i": c["n_i"].tolist(),
        }
    eedf: list[dict] = []
    if sim.eedf_results is not None:
        for r in sim.eedf_results:
            eedf.append(
                {
                    "label": r["label"],
                    "e_centers": r["e_centers"].tolist(),
                    "f": r["f"].tolist(),
                    "mean_energy_ev": r["mean_energy_ev"],
                    "t_eff_ev": r["t_eff_ev"],
                    "total_weight": r["total_weight"],
                    "overflow_frac": r["overflow_frac"],
                    "n_samples": r["n_samples"],
                }
            )
    timing_total = sum(sim.timing.values())
    return {
        "history": sim.history,
        "profiles": profiles,
        "cycle": cycle,
        "eedf": eedf,
        "walls": sim.wall,
        "fn": sim.fn,  # FN 電界放出サマリ (prompts/95)。両電極とも fn 未設定なら None
        "elapsed_s": elapsed_s,
        "timing": {**sim.timing, "total": timing_total},
        "settings": sim.s.model_dump(),
    }
