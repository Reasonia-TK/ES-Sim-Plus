"""1D ドリフト拡散流体ソルバー (prompts/104〜106)。

pic1d.py (1D PIC/MCC) と同一の電極・格子規約を共有し、直接比較できることを
設計目標にする。粒子を追わず、電子・イオンの数密度 n_e/n_i と電子エネルギー
密度 w = (3/2) n_e T_e、静電ポテンシャル φ を格子節点上の連続場として解く
(ドリフト拡散近似 + 電子エネルギー方程式 + Poisson の自己無撞着系)。

## 変数と格子

節点数 n_nodes = n_cells+1 (pic1d と同じ一様格子 xg)。フラックスは半節点
(セル界面、i+1/2) で評価する。

## Scharfetter–Gummel (SG) フラックスの導出

連続の式のフラックス Γ = v n − D ∂n/∂x (v: ドリフト速度、D: 拡散係数) を、
セル [x_i, x_i+h] 内で v, D が一定という局所近似のもとで厳密に解くと
(ξ=x−x_i、n(0)=n_i、n(h)=n_{i+1}、z=vh/D):

    n(ξ) = n(0)・e^{vξ/D} − (Γ/v)・(e^{vξ/D} − 1)

ξ=h で n(h) が既知という境界条件から Γ について解くと

    Γ = (D/h)・[B(−z)・n_i − B(z)・n_{i+1}],  B(z) = z/(e^z−1)

が得られる (Bernoulli 関数 B、B(z)>0 は恒等式 B(−z)−B(z)=z から従う)。この式は
z→0 (Pe→0) で中心差分の純拡散 (D/h)(n_i−n_{i+1}) に、z→±∞ (Pe→∞) で完全風上
Γ→v・n_upstream に厳密に一致する (test_fluid1d.py のテスト1で直接検証)。

さらに界面の z を「隣接節点の電位差そのもの」から組み立てる (面の電場を平均
するのではなく、E_face・dx = φ_i−φ_{i+1} を直接使う) と、Einstein 関係
D=μT のもとで z = ±(φ_i−φ_{i+1})/T (μ が厳密に約分される) という純粋に
「電位差/温度」の比になる。これが SG スキームの核心的性質 — 格子解像度に
よらず熱平衡 (n ∝ exp(±eφ/T)、ゼロフラックス) を厳密に再現する — の由来であり、
テスト3 (ボルツマン平衡) はこの性質を直接検証する。

電子エネルギーフラックスも同じ SG 型で、Hagelaar & Kroesen (Plasma Sources
Sci. Technol. 9, 249 (2000)) の標準形 μ_ε=(5/3)μ_e、D_ε=(5/3)D_e を使う。
z は電子フラックスの z と同一 (μ_e, D_e が同じ比率 5/3 でスケールされるため
比は不変) なので、電子フラックスの z をそのまま再利用できる (実装上の
簡略化、_face_coeffs 参照)。

## 時間積分 (2段構え — prompts/104 のリスク対策)

1. **陽的経路** (`Fluid1dSimulation(project, explicit=True)`、検証用): 現在値で
   フラックスを評価し前進オイラーで更新、Poisson は毎ステップ解く。安定条件は
   拡散 CFL (dt ≲ dx²/(2D)) と誘電緩和時間 τ_d=ε0/(e n_e μ_e) (dt ≲ τ_d) の
   両方を満たす必要がある (小さい dt でのみ安定、__init__ で目安を警告表示)。
   schema には出さない — 陽的経路はテスト・検証用の内部 API であり、通常の
   実行 (Project.fluid1d 経由) は常に半陰的経路を使う。
2. **半陰的経路** (既定): 種ごとの連続の式・エネルギー方程式を後退オイラーで
   陰化し、三重対角を solve_banded で解く (種ごと独立)。SG 係数 (a, b: 面の
   Bernoulli 係数) はステップ開始時の n から解いた Poisson の φ で固定し
   (①Poisson→E、②その E で SG 係数を組んで電子・イオン連続を陰的に解く、
   ③エネルギー方程式を陰的に解く)、反応源項 (電離・励起・弾性損失) は
   ステップ開始時の Te/n_e で評価する陽的項として扱う (IMEX 分割)。この
   分割の理由: 数値的に最も硬い (stiff な) 項は輸送 (対流拡散、特に電子の
   誘電緩和) であり、それだけを陰化すれば主要な安定性制約は取り除ける。
   反応レートまで陰的にすると n_e に対して非線形な連立になり実装が大きく
   複雑化するが、電離・励起・弾性損失は典型的な運転条件で誘電緩和よりずっと
   緩やかなタイムスケールなので、陽的に扱っても半陰スキーム全体の安定性を
   支配しない。
   - **誘電緩和のサブステップ化**: dt が τ_d=ε0/(e n_e μ_e) の 0.5 倍を超えると、
     「①Poisson→②輸送陰解」という逐次法 (E を n_e^{new} ではなく n_e^{old} 由来で
     固定する近似) は電子密度と電場の結合が強くなりすぎて不安定になりうる。
     対策として、prompts/106 が許容する2つの案のうち **サブステップ分割
     (dt を τ_d/2 以下に内部分割して①②③を繰り返す)** を採用した。
     理由: (a) 種ごと独立な三重対角ソルブという既存の実装をそのまま複数回
     回すだけで実現でき、φ と n_e を連立させるブロック三重対角ソルバー
     (実装・検証コストが大きく、pic1d/fluid_coeffs にも前例がない) を新たに
     作り込む必要がない。(b) CCP 条件 (n_e~1e14〜1e18 m^-3、Ar 数十 Pa) では
     τ_d が dt よりずっと大きいことが多く、サブステップが実際に発動する場面は
     限定的 — 過剰な実装コストに見合わない。(c) 陽的経路との一致テスト
     (小 dt、サブステップ無効) で半陰スキームの正しさを既に検証できるため、
     サブステップ自体は「安定性の保険」であり結果の精度そのものを左右しない。
   - 陽的経路と半陰経路は同じ SG 係数・BC 式を共有するため、小 dt では両者は
     一致するはずで、これをテスト5で直接確認する。

## 境界条件 (Hagelaar & Kroesen (2000) 系)

- イオン: 壁向きドリフト流束 (μ_i n_i E が壁向きのときのみ、逆向きなら0) +
  熱流束 (1/4) n_i v_th,i (v_th = √(8T/(πm))、Maxwell 分布の平均速さ)。
- 電子: 熱流束 (1/4) n_e v_th,e − γ·Γ_i,wall (SEE、Pic1dElectrode.see_gamma。
  Γ_i,wall はイオンの壁流束、γ 分だけ電子を「壁から」注入するので符号は
  電子の壁損失を減らす向き)。電子には明示的なドリフト項を含めない
  (Hagelaar & Kroesen の簡略形。壁近傍の急峻な密度勾配は SG 拡散フラックスが
  既に担っており、二重に数えない)。
- エネルギー: q_wall = (5/3)·T_e,wall·Γ_e,wall。文献には壁を通過する電子1個
  あたりの平均運動エネルギーとして 2T_e を使う流儀もあるが (truncated
  Maxwellian の厳密平均)、本実装は SG エネルギー輸送係数 (μ_ε=(5/3)μ_e) との
  一貫性を優先し (5/3)T_e を採用する (差は同じオーダーで、CCP スモークテストの
  ゆるい許容幅には影響しない)。
- 電位: 両端 Dirichlet。pic1d.electrode_voltage (prompts/93、v_dc + Σ RF +
  Σ CSV波形) を共用する。

## 反応源項

S_ion = k_ion(Te)·n_g·n_e (電子・イオン共通の生成源)。エネルギー損失は
電離損失 + 励起損失 + 弾性損失 = (e_ion·k_ion + e_exc·k_exc)·n_g·n_e +
3·(m/M)·⟨σ_m v⟩·(Te−Tg)·n_g·n_e (⟨σ_m v⟩ は弾性のレート係数、Tg はガス温度
[eV])。係数はいずれも fluid_coeffs.interp_loglog で Te から引く (テーブル
範囲外はクランプ + 実行を通して1回だけ警告)。電子加熱は Joule 加熱
-Γ_e·E (面フラックスを隣接節点で平均した節点中心値を使う。単位は
「電子1個の電荷 e を跨いで e が約分されるため」eV·m^-3·s^-1 に直接一致する
— SI の -eΓ_eE [W/m^3] を eV に変換すると e で割るのでちょうど e が消える)。
"""

from __future__ import annotations

import math
import time
import warnings

import numpy as np
from scipy.linalg import solve_banded

from .fem import EPS0
from .fluid_coeffs import FluidReactions, build_fluid_reactions, interp_loglog
from .mcc import KB
from .particles import ME, MP, QE
from .pic1d import brinkmann_sheath_edge, electrode_voltage
from .pic1d_presets import edupic_ar_processes
from .schema import Fluid1dSettings, Project, rf_components

# 電子密度フロア [m^-3] (Te=(2/3)w/n_e のゼロ除算回避、prompts/106 の指示どおり)。
# これを下回るほど希薄な領域は物理的には「ほぼ真空」であり、Te の値自体に意味は
# なくなるが、指数的な消滅を避けるため通常運転では触れない値に設定してある
FLOOR_N = 1.0e6
# w (電子エネルギー密度) のフロア。FLOOR_N・0.001eV 相当 (n_e がフロアに落ちた
# 極限でも Te が非物理的な値に発散しないための安全弁)
FLOOR_W = 1.5 * FLOOR_N * 1.0e-3

# history の列名 (毎ステップこの全キーを持つ、pic1d の _HISTORY_KEYS と同じ設計)
_HISTORY_KEYS = (
    "step", "t", "n_e_total", "n_i_total",
    "wall_left_e", "wall_left_i", "wall_right_e", "wall_right_i",
    "gen_total",
)


def _bernoulli(z: np.ndarray) -> np.ndarray:
    """Bernoulli 関数 B(z) = z/(e^z−1) (B(0)=1、常に B(z)>0)。SG フラックスの核。

    z→0 近傍は e^z−1 の桁落ちを避けるため級数展開 (1 − z/2 + z²/12 − …)、
    z が大きい領域は expm1 のオーバーフロー (z≳709 で e^z が float64 の範囲を
    超える) を避けるため漸近形 (z→+∞: B(z)=z e^{-z}→0、z→−∞: B(z)=−z) を使う。
    |z|<500 の中間領域は e^500≈1.4e217 で float64 の範囲内に十分収まるため
    expm1 をそのまま使って構わない (prompts/106 の分岐指示どおり)。
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    small = np.abs(z) < 1.0e-4
    big_pos = z > 500.0
    big_neg = z < -500.0
    mid = ~(small | big_pos | big_neg)
    zs = z[small]
    out[small] = 1.0 - 0.5 * zs + zs * zs / 12.0
    out[big_pos] = z[big_pos] * np.exp(-z[big_pos])
    out[big_neg] = -z[big_neg]
    zm = z[mid]
    out[mid] = zm / np.expm1(zm)
    return out


def frost_mobility(mu_l, e_abs, n_g: float, c_td: float):
    """修正 Frost 式によるイオン移動度 μ_i(E/N) (prompts/116、fluid1d.py/fluid2d.py 共通)。

        μ_i(E/N) = μ_L / √(1 + (E/N)/C),   E/N [Td] = |E| / n_g / 1e-21

    μ_L (mu_l) は低電界一定値 (mu_i_ref から決まる)、C (c_td) は移動度が
    μ_L/√2 まで落ちる E/N。既定 C=150 Td は Ar+ in Ar の実測
    (Ellis et al., At. Data Nucl. Data Tables 17, 177 (1976)) に対する粗い
    フィットであり、精密フィットではなく工学近似 (ガス種が変われば C の
    調整が必要 — schema.Fluid1dSettings.frost_c_td のコメント参照)。

    D_i (拡散係数) はこの関数の対象外: 呼び出し側は低電界値 D_i=μ_L・T_i の
    ままにする (Frost 補正を掛けない)。理由: 強電界域はドリフト支配で拡散の
    寄与自体が小さく、そもそも Einstein 関係 D=μT は熱平衡 (弱電界) を前提と
    した関係式で強電界では崩れており、D にも同じ補正を掛ける物理的根拠が
    薄いため (むしろ低電界値のまま残す方が保守的)。
    """
    e_n_td = np.abs(e_abs) / n_g / 1.0e-21
    return mu_l / np.sqrt(1.0 + e_n_td / c_td)


# ---- 壁 IEDF (無衝突シース近似、prompts/116) --------------------------------------
#
# 流体は粒子を持たないため、壁 (電極) への入射イオンエネルギー分布 (IEDF) を
# 「位相分解シース電圧 + イオン走行時間フィルタ」という工学近似で再構成する。
# 仮定・限界: 無衝突シース (電荷交換 (CX) 衝突による低エネルギーテールは表現
# できない)、シースエッジで静止したイオンがそこから壁まで無衝突で加速される
# という単純化 (エッジでの熱エネルギー分・シース内の弾性散乱は無視)。
# フィルタ・ヒストグラムの核となる2つの純関数をここに切り出し、単体テストで
# 二山 (τ_i≪T_rf)/単峰 (τ_i≫T_rf) の遷移を定量検証できるようにする。


def sheath_voltage_filter(
    v_sh: np.ndarray, dt_bin: float, tau_i: float, n_periods: int = 50
) -> np.ndarray:
    """位相分解シース電圧 V_sh (1周期分、bins 点・等時間間隔 dt_bin) を、イオン
    走行時間 τ_i の1次ローパス dV_eff/dt=(V_sh−V_eff)/τ_i で周期定常状態まで
    フィルタし、フィルタ後の V_eff (同じ bins 点、1周期分) を返す。

    各位相ビン区間内で V_sh が一定という近似のもとでこの ODE を厳密に積分すると、
    zero-order-hold の指数フィルタ (無条件安定、dt_bin/τ_i の大小によらず発散しない)

        V_eff[k+1] = V_sh[k] + (V_eff[k] − V_sh[k])・exp(−dt_bin/τ_i)

    という漸化式になる。周期境界条件 (V_eff[0] は前周期の V_eff[bins−1] から続く)
    を満たす周期定常解を、この漸化式を n_periods 周期分 (初期値は V_sh の平均) 繰り返して
    最後の1周期を返すことで近似する — 線形時不変フィルタなので、十分な周期数を
    回せば初期値によらず同じ周期解に収束する。

    τ_i≪T_rf (dt_bin≫τ_i, exp(−dt_bin/τ_i)→0) では V_eff→V_sh
    (瞬時追従、IEDF は二山型) に、τ_i≫T_rf (dt_bin≪τ_i, exp(−dt_bin/τ_i)→1) では
    V_eff がほぼ一定値 (時間平均 V̄_sh、IEDF は単峰型) に収束する — Lieberman &
    Lichtenberg の教科書に載っている transit-time 効果の標準的な振る舞い。
    """
    v_sh = np.asarray(v_sh, dtype=np.float64)
    bins = len(v_sh)
    if bins == 0:
        return v_sh.copy()
    if tau_i <= 0.0 or not np.isfinite(tau_i) or dt_bin <= 0.0:
        return v_sh.copy()  # τ_i=0 相当 (瞬時追従) は V_eff=V_sh と同じ
    ratio = dt_bin / tau_i
    decay = math.exp(-ratio) if ratio < 700.0 else 0.0
    v_eff = np.empty(bins)
    prev = float(np.mean(v_sh))
    for _ in range(n_periods):
        for k in range(bins):
            prev = v_sh[k] + (prev - v_sh[k]) * decay
            v_eff[k] = prev
    return v_eff


def _weighted_energy_histogram(e_ev: np.ndarray, weight: np.ndarray, bins: int) -> dict:
    """重み付きエネルギー値集合 (e_ev, weight) から IEDF ヒストグラムを組み立てる
    (EEDF/壁 IEDF (PIC) と同じ規約: e_max は最大値×1.2 で自動決定、f は
    ∫f dE=1 に正規化する)。

    壁 IEDF (流体) はサンプル数が高々 phase_bins 個 (典型数十) と少ないため、
    PIC/EEDF のようなオンライン (ストリーミング) 蓄積ではなく run_batch 完了後に
    まとめて一度だけ計算する (メモリ上の制約が無いのでこの方が単純)。

    重みの総和が 0 (壁イオン流束が全て0等の退化ケース) なら f・mean_energy_ev を
    すべて0として返す (EEDF の「電子が入らない領域」と同じフォールバック)。
    """
    e_ev = np.asarray(e_ev, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    sum_w = float(np.sum(weight))
    e_max = float(e_ev.max()) * 1.2 if e_ev.size and sum_w > 0.0 else 0.0
    if e_max <= 0.0:
        e_max = 30.0  # 全エネルギー0等の退化ケースのフォールバック (EEDF と同じ既定値)
    d_e = e_max / bins
    e_centers = (np.arange(bins) + 0.5) * d_e
    if sum_w > 0.0:
        idx = np.clip((e_ev / e_max * bins).astype(np.int64), 0, bins - 1)
        hist = np.bincount(idx, weights=weight, minlength=bins)
        f = hist / (sum_w * d_e)
        mean_e = float(np.sum(weight * e_ev) / sum_w)
    else:
        f = np.zeros(bins)
        mean_e = 0.0
    return {"e_centers": e_centers, "f": f, "mean_energy_ev": mean_e, "total_weight": sum_w}


class Fluid1dSimulation:
    """1D ドリフト拡散流体 (SG フラックス + 半陰的時間積分) シミュレーション本体。

    explicit=True で陽的経路 (検証用、schema には出さない内部 API) を選ぶ。
    既定 (explicit=False) は半陰的経路 — Project.fluid1d 経由の通常実行は
    常にこちらを使う。
    """

    def __init__(self, project: Project, explicit: bool = False):
        if project.fluid1d is None:
            raise ValueError("project.fluid1d が指定されていません")
        self.project = project
        self.s: Fluid1dSettings = project.fluid1d
        self.explicit = bool(explicit)
        s = self.s

        # ---- 格子 (pic1d と同じ規約) ---------------------------------------
        self.gap = float(s.gap_m)
        self.n_cells = int(s.n_cells)
        self.n_nodes = self.n_cells + 1
        self.dx = self.gap / self.n_cells
        self.xg = np.linspace(0.0, self.gap, self.n_nodes)
        # 節点「体積」(単位断面積あたりの長さ [m])。境界節点は半分 (有限体積法の
        # 対称な取り方。pic1d.node_vol と同じ考え方)
        self.node_vol = np.full(self.n_nodes, self.dx)
        self.node_vol[0] = self.dx * 0.5
        self.node_vol[-1] = self.dx * 0.5

        # ---- Poisson 行列 (三重対角、dx 不変なので初期化時に1回だけ構築) -------
        k = EPS0 / self.dx**2
        diag = np.full(self.n_nodes, 2.0 * k)
        diag[0] = 1.0
        diag[-1] = 1.0
        upper = np.full(self.n_nodes - 1, -k)
        upper[0] = 0.0
        lower = np.full(self.n_nodes - 1, -k)
        lower[-1] = 0.0
        ab = np.zeros((3, self.n_nodes))
        ab[0, 1:] = upper
        ab[1, :] = diag
        ab[2, :-1] = lower
        self._poisson_ab = ab

        # ---- ガス・イオン輸送係数 (mu_i は低電界値、D_i はスカラー、Te に依存しない
        #      低電界移動度モデル。ガス密度への依存だけスケールする) -----------
        self.n_g = s.gas_pressure_pa / (KB * s.gas_temperature_k)
        self.m_ion = s.ion_mass_amu * MP
        # mu_i: 低電界値 μ_L (frost モデルでもこの値自体は変えない。d_i の算出・
        # SG z の比の基準としてスカラーのまま保持する。frost 補正後の値は
        # _mu_i_at (界面・境界ごと、ベクトル) で都度求める)
        self.mu_i = s.mu_i_ref * (s.n_ref_m3 / self.n_g)
        self.t_i_ev = float(s.t_i_ev)
        self.d_i = self.mu_i * self.t_i_ev  # Frost 補正なし (frost_mobility の docstring 参照)
        self.ion_mobility_model = s.ion_mobility_model
        self.frost_c_td = float(s.frost_c_td)

        # ---- 電子の反応・輸送係数テーブル (fluid_coeffs.py、Phase A) ----------
        electron_processes = s.electron_processes if s.electron_processes else edupic_ar_processes()[0]
        self.reactions: FluidReactions = build_fluid_reactions(electron_processes)
        # 弾性衝突エネルギー損失の質量比 m/M (通常は弾性プロセス1個だけを想定。
        # 複数あれば単純平均でフォールバックする — 弾性が複数あること自体が
        # 稀なケースなので、レート加重平均のような精密な扱いまでは行わない)
        elastic_procs = [p for p in electron_processes if p.kind == "elastic"]
        self._mass_ratio = float(np.mean([p.mass_ratio for p in elastic_procs])) if elastic_procs else 0.0
        self._te_warned = False  # テーブル範囲外の Te 警告は実行を通して1回だけ

        # ---- 状態変数 (節点、一様初期化) --------------------------------------
        self.n_e = np.full(self.n_nodes, float(s.init_density_m3))
        self.n_i = np.full(self.n_nodes, float(s.init_density_m3))
        self.w = 1.5 * self.n_e * float(s.init_te_ev)
        self.phi = np.zeros(self.n_nodes)

        # ---- RF 周波数 (dt 自動決定・位相分解の両方に使う。pic1d と同じ優先順位:
        #      voltage_rf があればそちら優先、なければ CSV 波形) -------------------
        rf_freqs = [
            c.freq_hz for c in (*rf_components(s.left.voltage_rf), *rf_components(s.right.voltage_rf))
        ]
        wf_freqs = [wf.freq_hz for wf in s.left.waveforms] + [wf.freq_hz for wf in s.right.waveforms]
        self._cycle_freq = min(rf_freqs) if rf_freqs else (min(wf_freqs) if wf_freqs else None)
        self._cycle_bins = int(s.phase_bins)
        self._cycle_enabled = self._cycle_freq is not None and self._cycle_bins > 0
        self._cycle_period = 1.0 / self._cycle_freq if self._cycle_enabled else 0.0

        # ---- dt (None なら RF周期/2000 と 1e-10 の小さい方、prompts/106) --------
        if s.dt is not None:
            self.dt = float(s.dt)
        elif self._cycle_freq is not None:
            self.dt = min(1.0 / self._cycle_freq / 2000.0, 1.0e-10)
        else:
            self.dt = 1.0e-10

        # ---- 安定性の目安警告 (陽的経路のみ。半陰は後退オイラー + サブステップで
        #      無条件安定なので警告不要) -------------------------------------------
        self.warnings: list[str] = []
        if self.explicit:
            te0, mu_e0, _k_ion0, _k_exc0, _nu0 = self._te_and_coeffs(self.n_e, self.w)
            d_e0 = float(np.max(mu_e0 * te0))
            dt_diff = 0.5 * self.dx**2 / max(d_e0, self.d_i, 1.0e-300)
            tau_d0 = EPS0 / (QE * max(float(np.max(self.n_e * mu_e0)), 1.0e-300))
            dt_bound = min(dt_diff, tau_d0)
            if self.dt > dt_bound:
                self.warnings.append(
                    f"陽的経路: dt={self.dt:.3g}s が安定条件の目安 {dt_bound:.3g}s "
                    "(拡散CFL・誘電緩和時間の小さい方) を超えています (数値不安定の恐れ)"
                )

        # ---- テスト専用フラグ (既定 True/False。schema には出ない Python API。
        #      解析検証テスト (両極性拡散・ボルツマン平衡) で反応源・エネルギー
        #      方程式・イオン輸送・壁吸収を個別に無効化するために使う) --------------
        self.debug_source_enabled = True    # False: S_ion・反応エネルギー損失を0に
        self.debug_energy_enabled = True    # False: Te を init_te_ev に固定 (エネルギー方程式を解かない)
        self.debug_ions_enabled = True      # False: n_i を更新しない (固定背景)
        self.debug_reflective_walls = False  # True: 壁フラックスを全て0 (閉じた系)

        # ---- 診断・時刻 ---------------------------------------------------------
        self.t = 0.0
        self.step_count = 0
        self.wall: dict[str, dict[str, float]] = {
            "left": {"electron": 0.0, "ion": 0.0},
            "right": {"electron": 0.0, "ion": 0.0},
        }
        self.gen_total = 0.0  # 電離による累計生成数 [m^-2] (電子・イオン共通)
        self.history: dict[str, list] = {k: [] for k in _HISTORY_KEYS}
        self.timing: dict[str, float] = {
            "poisson": 0.0, "transport": 0.0, "energy": 0.0, "other": 0.0,
        }

        # ---- 時間平均・位相分解アキュムレータ (run_batch/enable_density_accum で確保) --
        self._accum_start: int | None = None
        self._accum_count = 0
        self._accum_phi: np.ndarray | None = None
        self._accum_e: np.ndarray | None = None
        self._accum_ne: np.ndarray | None = None
        self._accum_ni: np.ndarray | None = None
        self._accum_te: np.ndarray | None = None
        self._accum_ion: np.ndarray | None = None
        self._cycle_phi: np.ndarray | None = None
        self._cycle_ne: np.ndarray | None = None
        self._cycle_ni: np.ndarray | None = None
        self._cycle_te: np.ndarray | None = None
        self._cycle_count: np.ndarray | None = None
        self.fields: dict | None = None
        self.cycle: dict | None = None
        self.sheath: dict | None = None
        # 壁 IEDF (無衝突シース近似、prompts/116)。wall_iedf_bins=0 で無効
        self._wall_iedf_bins = int(s.wall_iedf_bins)
        self.wall_iedf: dict[str, dict] | None = None
        self._run_t0 = 0.0

    # ---- 係数評価 --------------------------------------------------------------

    def _te_and_coeffs(self, n_e: np.ndarray, w: np.ndarray):
        """節点ごとの Te・電子移動度・反応レート係数をまとめて求める。

        Te = (2/3) w/n_e (n_e はフロア FLOOR_N でゼロ除算回避)。テーブル引き
        (interp_loglog) は Te をテーブル範囲へ内部でクランプするが、Te 自体
        (v_th 等の物理式に使う生値) はクランプしない。テーブル範囲を外れた
        場合は実行を通して1回だけ警告する (prompts/104 の既定方針)。
        戻り値: (te, mu_e, k_ion, k_exc, nu_m_per_ng) — nu_m_per_ng は弾性の
        ⟨σ_m v⟩ (n_g を掛けていない生のレート係数、呼び出し側で n_g を掛ける)。
        """
        n_e_safe = np.maximum(n_e, FLOOR_N)
        te = np.maximum((2.0 / 3.0) * w / n_e_safe, 1.0e-6)
        grid = self.reactions.te_grid_ev
        lo, hi = grid[0], grid[-1]
        if not self._te_warned and (float(te.min()) < lo or float(te.max()) > hi):
            warnings.warn(
                f"fluid1d: Te が係数テーブル範囲 [{lo:.3g}, {hi:.3g}] eV の外に出ました "
                "(テーブル引きはクランプして継続します)"
            )
            self._te_warned = True
        mu_e = interp_loglog(grid, self.reactions.transport.mobility_n, te) / self.n_g
        k_ion = interp_loglog(grid, self.reactions.k_ion, te)
        k_exc = interp_loglog(grid, self.reactions.k_exc, te)
        nu_m_per_ng = interp_loglog(grid, self.reactions.transport.nu_m_per_ng, te)
        return te, np.asarray(mu_e), np.asarray(k_ion), np.asarray(k_exc), np.asarray(nu_m_per_ng)

    def _mu_i_at(self, e_abs):
        """界面/境界の |E| における μ_i (frost: 修正 Frost 式、const: 低電界値のまま)。

        "const" は e_abs に関わらず self.mu_i (スカラー) をそのまま返すので、
        呼び出し側で ×1.0 する形にしておけば frost 導入前と完全にビット一致する
        (frost_mobility 自体を経由しないため、E/N 計算の丸め誤差すら混入しない)。
        """
        if self.ion_mobility_model == "const":
            return self.mu_i
        return frost_mobility(self.mu_i, e_abs, self.n_g, self.frost_c_td)

    def _face_coeffs(self, phi: np.ndarray, te: np.ndarray, mu_e: np.ndarray):
        """種ごとの SG 面係数 (a, b: Γ_face = a・n_left − b・n_right) と、電子の
        z (エネルギーフラックスと共有) を計算する。全て長さ n_cells (界面の数)。

        モジュール docstring の導出のとおり、面の z は隣接節点の電位差 dphi=
        φ_i−φ_{i+1} と面の温度 (イオンはスカラー T_i、電子は隣接ノードの単純
        平均 Te_face) から z=±dphi/T として直接組み立てる。電子の D_face は
        「面の mu_e (平均) と面の Te (平均) から Einstein 関係で直接組み立てる」
        (ノードごとの D_e を単純平均するのではない) — こうすることで Te が
        空間一様なら mu_e_face/D_e_face が厳密に 1/Te_face になり、SG の熱平衡
        保存性がイオンと同様に電子側にも厳密に成り立つ。

        イオンの z (修正 Frost、prompts/116): SG の一般形 z=v_face・dx/D_i で
        v_face=μ_i(|E_face|)・E_face、E_face・dx=dphi (この面の電位差そのもの)。
        D_i は低電界値のまま (frost_mobility の docstring 参照) なので
        z_i = μ_i(|E_face|)・dphi/D_i = (dphi/T_i)・(μ_i(|E_face|)/μ_L) —
        従来の z_i=dphi/T_i に「frost 補正/低電界値」の比を掛けるだけでよい
        (μ_L=self.mu_i が厳密に約分されるのは const 相当の μ_i(E)=μ_L の場合のみで、
        frost では E 依存性が残る分だけ従来の熱平衡保存性からずれる —
        工学近似としての Frost モデル導入に伴う意図的なトレードオフ)。
        "const" は比が恒等的に 1.0 (self.mu_i/self.mu_i) になるため、
        z_i はビット完全に dphi/self.t_i_ev と一致する。
        """
        dphi = phi[:-1] - phi[1:]

        e_face = dphi / self.dx
        mu_i_face = self._mu_i_at(e_face)
        z_i = (dphi / self.t_i_ev) * (mu_i_face / self.mu_i)
        a_i = (self.d_i / self.dx) * _bernoulli(-z_i)
        b_i = (self.d_i / self.dx) * _bernoulli(z_i)

        mu_e_face = 0.5 * (mu_e[:-1] + mu_e[1:])
        te_face = 0.5 * (te[:-1] + te[1:])
        d_e_face = mu_e_face * te_face
        z_e = -dphi / te_face
        a_e = (d_e_face / self.dx) * _bernoulli(-z_e)
        b_e = (d_e_face / self.dx) * _bernoulli(z_e)
        return a_i, b_i, a_e, b_e, z_e, d_e_face

    def _e_field(self, phi: np.ndarray) -> np.ndarray:
        """節点電場 E=−dφ/dx (内点中心差分、端点片側差分。pic1d._e_field と同じ規約)。"""
        ex = np.empty(self.n_nodes)
        ex[1:-1] = -(phi[2:] - phi[:-2]) / (2.0 * self.dx)
        ex[0] = -(phi[1] - phi[0]) / self.dx
        ex[-1] = -(phi[-1] - phi[-2]) / self.dx
        return ex

    def _solve_phi(self, n_e: np.ndarray, n_i: np.ndarray, t: float) -> np.ndarray:
        """Poisson 求解: −ε0 φ'' = e(n_i−n_e)、両端 Dirichlet (electrode_voltage)。"""
        rhs = np.empty(self.n_nodes)
        rhs[1:-1] = QE * (n_i[1:-1] - n_e[1:-1])
        rhs[0] = electrode_voltage(self.s.left, t)
        rhs[-1] = electrode_voltage(self.s.right, t)
        return solve_banded((1, 1), self._poisson_ab, rhs)

    def _wall_side_coeffs(self, side: str, ex_boundary: float, te_boundary: float):
        """壁境界の線形フラックス係数 (Γ_wall = c・n[wall] の形の c) を返す。

        Hagelaar & Kroesen (2000) の標準形 (モジュール docstring 参照)。
        戻り値: (c_i, c_e, v_th_e) — v_th_e はエネルギー BC (q_wall) 計算用。
        """
        v_th_i = math.sqrt(8.0 * self.t_i_ev * QE / (math.pi * self.m_ion))
        v_th_e = math.sqrt(8.0 * te_boundary * QE / (math.pi * ME))
        n_out = -1.0 if side == "left" else 1.0  # 壁の外向き法線 (左=-x, 右=+x)
        # 壁向きドリフト流束の μ_i も SG 係数と同じ修正 Frost 補正を使う (prompts/116)。
        # "const" では self._mu_i_at が self.mu_i をそのまま返すので従来と完全に一致する
        v_drift_i = self._mu_i_at(ex_boundary) * ex_boundary
        c_i = max(n_out * v_drift_i, 0.0) + 0.25 * v_th_i
        c_e = 0.25 * v_th_e
        if self.debug_reflective_walls:
            c_i = c_e = 0.0
        return c_i, c_e, v_th_e

    # ---- 汎用の種輸送ソルバー (連続の式・エネルギー方程式で共用) -------------------

    def _implicit_transport_solve(
        self, n_old, a, b, c_left, c_right, source, extra_left, extra_right, dt
    ) -> np.ndarray:
        """後退オイラー (SG 係数 a/b、壁係数 c_left/c_right を固定した線形移流拡散)。

        extra_left/extra_right は「境界の n に比例しない既知の追加フラックス」
        (電子の SEE 源、エネルギー方程式の q_wall)。壁でのフラックスは
        Γ_wall = c・n[wall] − extra なので、extra は RHS へ「+extra」として
        加わる (壁損失を extra 分だけ減らす = 増加源として働く)。
        """
        n = self.n_nodes
        vol = self.node_vol
        b_left = np.concatenate(([c_left], b))    # 各節点の左フラックス係数 (境界=壁係数)
        a_right = np.concatenate((a, [c_right]))  # 各節点の右フラックス係数 (境界=壁係数)
        diag = vol / dt + b_left + a_right
        upper = -b
        lower = -a
        rhs = vol / dt * n_old + source * vol
        rhs = np.asarray(rhs, dtype=np.float64).copy()
        rhs[0] += extra_left
        rhs[-1] += extra_right
        ab = np.zeros((3, n))
        ab[0, 1:] = upper
        ab[1, :] = diag
        ab[2, :-1] = lower
        return solve_banded((1, 1), ab, rhs)

    def _explicit_transport_update(
        self, n_old, a, b, c_left, c_right, source, extra_left, extra_right, dt
    ):
        """前進オイラー版 (陽的経路)。_implicit_transport_solve と全く同じ係数・
        符号規約を使うので、両者の一致 (テスト5) は係数構築の共有によって保証される。

        戻り値: (n_new, flux_faces, flux_wall_left, flux_wall_right)。
        """
        flux = a * n_old[:-1] - b * n_old[1:]
        flux_wall_left = c_left * n_old[0] - extra_left
        flux_wall_right = c_right * n_old[-1] - extra_right
        net_in = np.empty(self.n_nodes)
        net_in[0] = -flux_wall_left - flux[0]
        net_in[1:-1] = flux[:-1] - flux[1:]
        net_in[-1] = flux[-1] - flux_wall_right
        n_new = n_old + dt * (net_in / self.node_vol + source)
        return n_new, flux, flux_wall_left, flux_wall_right

    # ---- 1ステップ (半陰・陽的 共通の下請け) -------------------------------------

    def _reaction_terms(self, n_e, te, k_ion, k_exc, nu_m_per_ng):
        """反応源項 (S_ion, 損失内訳) をまとめて計算する。debug_source_enabled=False
        なら電離・励起の生成/損失を丸ごと無効化する (両極性拡散テスト用)。
        """
        tg_ev = self.s.gas_temperature_k * KB / QE
        if self.debug_source_enabled:
            s_ion = k_ion * self.n_g * n_e
            loss_ion = self.reactions.e_ion_ev * s_ion
            loss_exc = self.reactions.e_exc_ev * k_exc * self.n_g * n_e
        else:
            s_ion = np.zeros(self.n_nodes)
            loss_ion = np.zeros(self.n_nodes)
            loss_exc = np.zeros(self.n_nodes)
        loss_elastic = 3.0 * self._mass_ratio * nu_m_per_ng * self.n_g * (te - tg_ev) * n_e
        return s_ion, loss_ion + loss_exc + loss_elastic

    def _accumulate_wall_and_gen(self, dt, s_ion, gwl_i, gwr_i, gwl_e, gwr_e):
        self.wall["left"]["ion"] += dt * gwl_i
        self.wall["right"]["ion"] += dt * gwr_i
        self.wall["left"]["electron"] += dt * gwl_e
        self.wall["right"]["electron"] += dt * gwr_e
        self.gen_total += dt * float(np.sum(np.asarray(s_ion) * self.node_vol))

    def _step_once(self, dt: float, t: float, implicit: bool):
        """半陰 (implicit=True) または陽的 (False) の1ステップ本体。

        手順: ①現在の n から Poisson→E・SG係数を組む、②イオン連続を解く
        (electron の SEE 源に使うため先に解く)、③電子連続を解く、④エネルギー
        方程式を解く (debug_energy_enabled=False なら Te を初期値に固定する
        だけで方程式自体は解かない)。半陰・陽的とも全く同じ係数構築・BC 式を
        共有し、線形ソルバー呼び出し ( _implicit_transport_solve /
        _explicit_transport_update ) だけを切り替える。
        """
        t0 = time.perf_counter()
        n_e, n_i, w = self.n_e, self.n_i, self.w
        phi = self._solve_phi(n_e, n_i, t)
        ex = self._e_field(phi)
        te, mu_e, k_ion, k_exc, nu_m_per_ng = self._te_and_coeffs(n_e, w)
        a_i, b_i, a_e, b_e, z_e, d_e_face = self._face_coeffs(phi, te, mu_e)
        s_ion, loss_total = self._reaction_terms(n_e, te, k_ion, k_exc, nu_m_per_ng)
        t1 = time.perf_counter()
        self.timing["poisson"] += t1 - t0

        c_i_left, c_e_left, _ = self._wall_side_coeffs("left", ex[0], te[0])
        c_i_right, c_e_right, _ = self._wall_side_coeffs("right", ex[-1], te[-1])

        solve = self._implicit_transport_solve if implicit else (
            lambda *a_, **kw: self._explicit_transport_update(*a_, **kw)[0]
        )

        # ---- イオン ----
        if self.debug_ions_enabled:
            n_i_new = solve(n_i, a_i, b_i, c_i_left, c_i_right, s_ion, 0.0, 0.0, dt)
            n_i_new = np.maximum(n_i_new, FLOOR_N)
        else:
            n_i_new = n_i.copy()
        gwl_i = 0.0 if self.debug_reflective_walls else c_i_left * n_i_new[0]
        gwr_i = 0.0 if self.debug_reflective_walls else c_i_right * n_i_new[-1]

        # ---- 電子 (SEE はいま求めたイオン壁流束を使う) ----
        see_left = self.s.left.see_gamma * gwl_i
        see_right = self.s.right.see_gamma * gwr_i
        n_e_new = solve(n_e, a_e, b_e, c_e_left, c_e_right, s_ion, see_left, see_right, dt)
        n_e_new = np.maximum(n_e_new, FLOOR_N)
        gwl_e = c_e_left * n_e_new[0] - see_left
        gwr_e = c_e_right * n_e_new[-1] - see_right
        t2 = time.perf_counter()
        self.timing["transport"] += t2 - t1

        # ---- エネルギー ----
        if self.debug_energy_enabled:
            flux_e_faces = a_e * n_e_new[:-1] - b_e * n_e_new[1:]
            faces_all = np.concatenate(([gwl_e], flux_e_faces, [gwr_e]))
            gamma_e_node = 0.5 * (faces_all[:-1] + faces_all[1:])
            joule = -gamma_e_node * ex  # eV/m^3/s (docstring: e が約分され直接一致)
            src_w = joule - loss_total
            a_eps, b_eps = (5.0 / 3.0) * a_e, (5.0 / 3.0) * b_e
            q_wall_left = (5.0 / 3.0) * te[0] * gwl_e
            q_wall_right = (5.0 / 3.0) * te[-1] * gwr_e
            w_new = solve(w, a_eps, b_eps, 0.0, 0.0, src_w, -q_wall_left, -q_wall_right, dt)
            w_new = np.maximum(w_new, FLOOR_W)
        else:
            w_new = 1.5 * n_e_new * self.s.init_te_ev
        t3 = time.perf_counter()
        self.timing["energy"] += t3 - t2

        self._accumulate_wall_and_gen(dt, s_ion, gwl_i, gwr_i, gwl_e, gwr_e)

        self.phi = phi
        self.n_i = n_i_new
        self.n_e = n_e_new
        self.w = w_new
        self.timing["other"] += time.perf_counter() - t3

    def step(self) -> np.ndarray:
        """流体1サイクル。時刻 t の場を解き、状態を t+dt へ進める。"""
        dt = self.dt
        t = self.t
        accumulating = self._accum_start is not None and self.step_count + 1 >= self._accum_start
        if accumulating:
            self._ensure_accumulators()

        if self.explicit:
            self._step_once(dt, t, implicit=False)
        else:
            # 半陰の逐次法 (①Poisson→E を固定 ②その E で輸送を陰的に解く) が
            # 精度良く成り立つための2つの制約から dt_sub の上限を決め、必要なら
            # サブステップ分割する (モジュール docstring の設計判断参照)。
            #   (a) 誘電緩和時間 τ_d=ε0/(e n_e μ_e): dt が τ_d を大きく超えると
            #       電荷不均衡の補正が電子の動きに追いつかず、①②の逐次反復
            #       そのものが発散しうる (prompts/106 が名指しする制約)。
            #   (b) 電子拡散時間 dx²/D_e: (a) を満たしていても、1 ステップの間に
            #       電子が「電場が更新される前に」格子間隔を大きく超えて自由拡散
            #       できてしまうと、両極性結合 (イオンと足並みを揃える制動) が
            #       効く前に電子だけが素通りしてしまい、電荷分離が暴走して発散する
            #       (実測で確認: (a) だけでは τ_d が大きくても発散するケースが
            #       あった)。dx²/D_e は「1ステップで電子が隣接セルより遠くまで
            #       拡散しない」という古典的な拡散 CFL に相当する
            # 両方とも最も厳しい (最小になる) 節点の値で判定し、全域の安定性を保証する
            te0, mu_e0, *_ = self._te_and_coeffs(self.n_e, self.w)
            tau_d = EPS0 / (QE * max(float(np.max(self.n_e * mu_e0)), 1.0e-300))
            d_e_max = max(float(np.max(mu_e0 * te0)), 1.0e-300)
            dt_diff_bound = 0.5 * self.dx**2 / d_e_max
            dt_bound = min(0.5 * tau_d, dt_diff_bound)
            n_sub = max(1, math.ceil(dt / dt_bound))
            dt_sub = dt / n_sub
            for k in range(n_sub):
                self._step_once(dt_sub, t + k * dt_sub, implicit=True)

        self.t = t + dt
        self.step_count += 1

        if accumulating:
            self._accumulate_fields(t)

        h = self.history
        h["step"].append(self.step_count)
        h["t"].append(self.t)
        h["n_e_total"].append(float(np.sum(self.n_e * self.node_vol)))
        h["n_i_total"].append(float(np.sum(self.n_i * self.node_vol)))
        h["wall_left_e"].append(self.wall["left"]["electron"])
        h["wall_left_i"].append(self.wall["left"]["ion"])
        h["wall_right_e"].append(self.wall["right"]["electron"])
        h["wall_right_i"].append(self.wall["right"]["ion"])
        h["gen_total"].append(self.gen_total)
        return self.phi

    # ---- 時間平均・位相分解アキュムレータ (pic1d と同じ設計) -----------------------

    def enable_density_accum(self, start_step: int) -> None:
        self._accum_start = int(start_step)
        self._accum_count = 0
        self._accum_phi = None
        self._accum_e = None
        self._accum_ne = None
        self._accum_ni = None
        self._accum_te = None
        self._accum_ion = None
        self._cycle_phi = None
        self._cycle_ne = None
        self._cycle_ni = None
        self._cycle_te = None
        self._cycle_count = None

    def _ensure_accumulators(self) -> None:
        if self._accum_phi is not None:
            return
        n = self.n_nodes
        self._accum_phi = np.zeros(n)
        self._accum_e = np.zeros(n)
        self._accum_ne = np.zeros(n)
        self._accum_ni = np.zeros(n)
        self._accum_te = np.zeros(n)
        self._accum_ion = np.zeros(n)
        if self._cycle_enabled:
            nb = self._cycle_bins
            self._cycle_phi = np.zeros((nb, n))
            self._cycle_ne = np.zeros((nb, n))
            self._cycle_ni = np.zeros((nb, n))
            self._cycle_te = np.zeros((nb, n))
            self._cycle_count = np.zeros(nb, dtype=np.int64)

    def _phase_bin(self, t: float) -> int:
        frac = (t % self._cycle_period) / self._cycle_period
        return min(int(frac * self._cycle_bins), self._cycle_bins - 1)

    def _accumulate_fields(self, t_step: float) -> None:
        ex = self._e_field(self.phi)
        te, _mu_e, k_ion, _k_exc, _nu = self._te_and_coeffs(self.n_e, self.w)
        s_ion = (k_ion * self.n_g * self.n_e) if self.debug_source_enabled else np.zeros(self.n_nodes)
        self._accum_phi += self.phi
        self._accum_e += ex
        self._accum_ne += self.n_e
        self._accum_ni += self.n_i
        self._accum_te += te
        self._accum_ion += s_ion
        if self._cycle_phi is not None:
            b = self._phase_bin(t_step)
            self._cycle_phi[b] += self.phi
            self._cycle_ne[b] += self.n_e
            self._cycle_ni[b] += self.n_i
            self._cycle_te[b] += te
            self._cycle_count[b] += 1
        self._accum_count += 1

    def averaged_fields(self) -> dict | None:
        if self._accum_start is None or self._accum_count == 0:
            return None
        cnt = self._accum_count
        return {
            "phi": self._accum_phi / cnt,
            "e": self._accum_e / cnt,
            "n_e": self._accum_ne / cnt,
            "n_i": self._accum_ni / cnt,
            "t_e": self._accum_te / cnt,
            "ionization": self._accum_ion / cnt,
            "avg_steps": cnt,
        }

    def cycle_data(self) -> dict | None:
        if not self._cycle_enabled or self._cycle_phi is None:
            return None
        if int(self._cycle_count.sum()) == 0:
            return None
        cnt = np.maximum(self._cycle_count, 1)[:, None].astype(np.float64)
        return {
            "bins": self._cycle_bins,
            "freq_hz": self._cycle_freq,
            "phi": self._cycle_phi / cnt,
            "n_e": self._cycle_ne / cnt,
            "n_i": self._cycle_ni / cnt,
            "t_e": self._cycle_te / cnt,
        }

    # ---- 壁 IEDF (無衝突シース近似、prompts/116) --------------------------------

    def wall_iedf_data(self) -> dict | None:
        """壁 IEDF (無衝突シース近似のモデルベース再構成)。run_batch 完了後
        (fields/cycle/sheath が確定した後) に呼ぶ想定。wall_iedf_bins=0、または
        fields が無ければ None。RF 位相分解が無効 (phase_bins=0 または RF 無し)
        でも DC 縮退 (単峰、1サンプル) として例外を出さずに返す。

        仮定・限界 (モジュールの「壁 IEDF」節も参照): 無衝突シース (CX 衝突による
        低エネルギーテールは表現できない)、シースエッジで静止したイオンがそこ
        から壁まで無衝突で加速されるという単純化。

        手順:
          1. 壁ごとに時間平均シースエッジ位置 s̄ (self.sheath、Brinkmann) を得る
             (根が求まらない退化ケースは gap/2 を代わりに使うフォールバック)。
          2. 位相分解 φ・n_i (self.cycle) から、壁ごとの瞬時シース電圧
             V_sh(φ_rf)=φ(シースエッジ)−φ(壁) (イオンを壁へ加速する符号で正) と
             瞬時イオン壁流束 Γ_i(φ_rf)=c_i(φ_rf)・n_i(壁,φ_rf) を組み立てる
             (c_i は _wall_side_coeffs と全く同じ壁 BC 式)。cycle が無効なら
             fields (時間平均) から1点だけの V_sh・Γ_i を作る (DC 縮退)。
          3. τ_i=3s̄√(m_i/(2e・V̄_sh)) (Lieberman & Lichtenberg の無衝突 Child
             シース走行時間、V̄_sh は V_sh(φ_rf) の周期平均) を求め、
             sheath_voltage_filter で V_sh(φ_rf) を周期定常までフィルタする。
          4. E(φ_rf)=V_eff(φ_rf) [eV] (数値としては V_eff [V] と同じ — 1eV=e×1V
             の定義そのもの) を Γ_i(φ_rf) で重み付けしてヒストグラム化する。
        """
        bins = self._wall_iedf_bins
        if bins <= 0 or self.fields is None:
            return None
        out: dict[str, dict] = {}
        for side, node_idx in (("left", 0), ("right", -1)):
            s_bar = self.sheath.get(f"{side}_s") if self.sheath is not None else None
            if s_bar is None or s_bar <= 0.0:
                s_bar = self.gap / 2.0  # 根が求まらない退化ケースのフォールバック (バルク参照点)
            x_edge = s_bar if side == "left" else max(self.gap - s_bar, 0.0)

            if self.cycle is not None:
                phi_c = self.cycle["phi"]
                n_i_c = self.cycle["n_i"]
                nb = int(self.cycle["bins"])
                v_sh = np.empty(nb)
                gamma_i = np.empty(nb)
                for b in range(nb):
                    phi_b = phi_c[b]
                    if side == "left":
                        ex_boundary = -(phi_b[1] - phi_b[0]) / self.dx
                    else:
                        ex_boundary = -(phi_b[-1] - phi_b[-2]) / self.dx
                    phi_edge = float(np.interp(x_edge, self.xg, phi_b))
                    v_sh[b] = phi_edge - float(phi_b[node_idx])
                    c_i, _c_e, _v_th_e = self._wall_side_coeffs(side, ex_boundary, 1.0)
                    gamma_i[b] = c_i * float(n_i_c[b, node_idx])
                dt_bin = self._cycle_period / nb
            else:
                phi_avg = self.fields["phi"]
                if side == "left":
                    ex_boundary = -(phi_avg[1] - phi_avg[0]) / self.dx
                else:
                    ex_boundary = -(phi_avg[-1] - phi_avg[-2]) / self.dx
                phi_edge = float(np.interp(x_edge, self.xg, phi_avg))
                v_sh = np.array([phi_edge - float(phi_avg[node_idx])])
                c_i, _c_e, _v_th_e = self._wall_side_coeffs(side, ex_boundary, 1.0)
                gamma_i = np.array([c_i * float(self.fields["n_i"][node_idx])])
                dt_bin = 0.0  # DC: 1点のみなのでフィルタ (周期性) は無意味

            v_sh_bar = float(np.mean(v_sh))
            v_sh_bar_safe = max(v_sh_bar, 1.0e-6)  # 負/ゼロは工学近似の対象外 (下限でガード)
            tau_i = 3.0 * s_bar * math.sqrt(self.m_ion / (2.0 * QE * v_sh_bar_safe))
            v_eff = sheath_voltage_filter(v_sh, dt_bin, tau_i)

            e_ev = v_eff  # E[eV] = V_eff[V] (数値そのもの、1eV=e×1V の定義)
            weight = gamma_i * (dt_bin if dt_bin > 0.0 else 1.0)
            hist = _weighted_energy_histogram(e_ev, weight, bins)
            out[side] = {
                "e_centers": hist["e_centers"],
                "f": hist["f"],
                "mean_energy_ev": hist["mean_energy_ev"],
                "total_weight": hist["total_weight"],
                "n_samples": len(v_sh),
                # PIC (粒子ベース) の壁 IEDF と区別するためのモデル種別 (prompts/116)
                "model": "collisionless_sheath",
            }
        return out

    # ---- フレーム・実行 ---------------------------------------------------------

    def _make_frame(self) -> dict:
        te = np.maximum((2.0 / 3.0) * self.w / np.maximum(self.n_e, FLOOR_N), 0.0)
        # counts: 直近ステップの history 行 (pic1d._make_frame の counts と同じ設計。
        # step() が history へ追記済みの状態でここが呼ばれるため v[-1] でよい)
        counts = {k: v[-1] for k, v in self.history.items()}
        return {
            "type": "frame",
            "step": self.step_count,
            "t": self.t,
            "phi": self.phi.tolist(),
            "n_e": self.n_e.tolist(),
            "n_i": self.n_i.tolist(),
            "t_e": te.tolist(),
            "counts": counts,
            "elapsed_s": time.perf_counter() - self._run_t0,
        }

    def run_batch(self, callback=None, should_stop=None, store_frames: bool = True):
        """n_steps 回実行して (history, フレーム列) を返す (pic1d.run_batch と同じ設計)。"""
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
                and np.all(np.isfinite(self.n_e))
                and np.all(np.isfinite(self.n_i))
                and np.all(np.isfinite(self.w))
            ):
                raise ValueError(
                    f"数値発散を検出しました (step {self.step_count}: 電位・密度・"
                    "エネルギーのいずれかが非有限値)。dt を小さくする、n_cells を"
                    "増やす等を検討してください"
                )
            if self.step_count % self.s.frame_every == 0:
                frame = self._make_frame()
                if store_frames:
                    frames.append(frame)
                if callback is not None:
                    callback(frame)

        self.fields = self.averaged_fields()
        self.cycle = self.cycle_data()
        if self.fields is not None:
            self.sheath = _fluid_sheath_pair(self.xg, self.fields["n_e"], self.fields["n_i"], self.gap)
        else:
            self.sheath = None
        self.wall_iedf = self.wall_iedf_data()
        return self.history, frames

    def prepare_continue(
        self,
        extra_steps: int,
        frame_every: int | None = None,
        avg_steps: int | None = None,
        phase_bins: int | None = None,
    ) -> None:
        """完了/停止後の状態から追加実行の準備をする (pic1d.prepare_continue と同じ設計)。

        維持するもの: 状態 (n_e, n_i, w, φ)・時刻 t・step_count・累計カウンタ
        (wall/gen_total)。リセットするもの: 診断 history (追加区間のみ)・timing・
        平均/位相分解アキュムレータ。決定論的 (乱数不使用) なので
        run(n)+continue(m) は run(n+m) とビット一致する。
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
        self._accum_ne = None
        self._accum_ni = None
        self._accum_te = None
        self._accum_ion = None
        self._cycle_phi = None
        self._cycle_ne = None
        self._cycle_ni = None
        self._cycle_te = None
        self._cycle_count = None
        self.fields = None
        self.cycle = None
        self.sheath = None
        self.wall_iedf = None


def _fluid_sheath_pair(x: np.ndarray, n_e: np.ndarray, n_i: np.ndarray, gap: float) -> dict:
    """左右シースエッジのペア (pic1d._sheath_pair と同じ規約、brinkmann_sheath_edge を流用)。"""
    x_b = gap / 2.0
    return {
        "left_s": brinkmann_sheath_edge(x, n_e, n_i, True, x_b),
        "right_s": brinkmann_sheath_edge(x, n_e, n_i, False, x_b),
    }


# ---- 結果バンドル組み立て (server.py / batch.py 共通、Phase C で配線、prompts/106) -----


def build_fluid1d_result(sim: Fluid1dSimulation, elapsed_s: float) -> dict:
    """done.result (= ResultsBundle.fluid1d に格納する想定の形) を組み立てる。

    build_pic1d_result と同じ考え方: done で返す内容をそのまま保存し、そのまま
    読み込んで復元できる自己完結な dict にする。
    """
    profiles = None
    sheath = None
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
        sheath = sim.sheath
    cycle = None
    if sim.cycle is not None:
        c = sim.cycle
        cycle = {
            "bins": c["bins"],
            "freq_hz": c["freq_hz"],
            "phi": c["phi"].tolist(),
            "n_e": c["n_e"].tolist(),
            "n_i": c["n_i"].tolist(),
            "t_e": c["t_e"].tolist(),
        }
    wall_iedf = None
    if sim.wall_iedf is not None:
        wall_iedf = {
            side: {
                "e_centers": d["e_centers"].tolist(),
                "f": d["f"].tolist(),
                "mean_energy_ev": d["mean_energy_ev"],
                "total_weight": d["total_weight"],
                "n_samples": d["n_samples"],
                "model": d["model"],
            }
            for side, d in sim.wall_iedf.items()
        }
    timing_total = sum(sim.timing.values())
    return {
        "history": sim.history,
        "profiles": profiles,
        "sheath": sheath,
        "cycle": cycle,
        # 壁 IEDF (無衝突シース近似、モデルベース。prompts/116)。wall_iedf_bins=0 なら None
        "wall_iedf": wall_iedf,
        "walls": sim.wall,
        "gen_total": sim.gen_total,
        "elapsed_s": elapsed_s,
        "timing": {**sim.timing, "total": timing_total},
        "settings": sim.s.model_dump(),
    }
