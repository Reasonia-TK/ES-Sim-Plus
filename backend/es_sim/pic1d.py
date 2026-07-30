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
from .mcc import MccModel
from .particles import ME, MP, QE
from .pic import _eval_waveform
from .schema import Pic1dElectrode, Pic1dSettings, Project

# フレーム送出時の電子位相空間サンプルの最大点数 (pic.py の MAX_FRAME_PARTICLES と同じ趣旨)
MAX_FRAME_PARTICLES = 2000

# 診断 history の列名 (どのステップでも必ずこの全キーを持つ)
_HISTORY_KEYS = (
    "step", "t", "n_e", "n_i", "w_e", "w_i",
    "wall_left_e", "wall_left_i", "wall_right_e", "wall_right_i",
    "ion_events", "see_events", "coll_e",
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

        # ---- RF 1周期の位相分解 (2D の phase_bins と同じ規約) -----------------
        # 電極 (left/right) の waveforms のうち最初に見つかったものの freq_hz を
        # 基本周波数とする (Pic1dElectrode は voltage_rf を持たないため、RF 駆動は
        # 正弦波をサンプルした VoltageWaveform で表現する。pic1d_presets.py 参照)
        freqs = [wf.freq_hz for wf in s.left.waveforms] + [wf.freq_hz for wf in s.right.waveforms]
        self._cycle_freq = min(freqs) if freqs else None
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
        """電極電圧 V(t) = v_dc + Σ waveforms(t) (pic.py の _eval_waveform を流用)。"""
        v = elec.v_dc
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

        self.t = t + dt
        self.step_count += 1

        if accumulating:
            self._accumulate_fields(phi, ex, t)
            self._accumulate_eedf()

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
        (SEE/MCC)・累計カウンタ (wall/ion_events/see_events/coll_e)。
        リセットするもの: 診断 history (追加区間分のみ)・timing・平均/位相/EEDF の
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
        for st in self._eedf_st:
            st["sum_w"] = 0.0
            st["sum_we"] = 0.0
            st["overflow_w"] = 0.0
            st["hist"] = None if st["auto"] else np.zeros(st["bins"])
            if st["auto"]:
                st["e_max"] = None
