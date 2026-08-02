"""1D 流体モデル (prompts/104〜) の係数生成モジュール — Phase A (prompts/105)。

既存の断面積データ (XsProcess: lxcat インポート / pic1d_presets.edupic_ar_processes)
から、1D ドリフト拡散流体ソルバー (prompts/106 で実装、fluid1d.py) が必要とする
輸送・反応係数 (電子の Maxwell 平均レート係数 k(Te)・運動量移行衝突頻度 ν_m(Te)・
移動度 μ_e) をテーブル化する。粒子モデル (PIC/MCC) は衝突を粒子ごとに離散的に
扱うのに対し、流体モデルは電子集団を局所 Maxwell 分布 (温度 Te) と仮定して
係数をあらかじめ Te の関数として畳み込んでおく — これが本モジュールの役割。

## Maxwell 平均レート係数の導出

等方 Maxwell 分布のエネルギー分布関数 (E, Te は eV 単位):

    f(E) = (2/√π) · Te^{-3/2} · √E · exp(−E/Te)      (∫f(E) dE = 1)

レート係数は k(Te) = ∫ σ(E) v(E) f(E) dE、v(E) = √(2 e E / m_e) (e=QE, E は eV
なので eE が Joule のエネルギー) を代入すると:

    k(Te) = √(2eE/m_e) の係数を前に出して整理すると
          = √(8e/(π m_e)) · Te^{-3/2} · ∫ σ(E)·E·exp(−E/Te) dE

(∫σ(E)E·exp(−E/Te)dE の因子分解は
 √(2e/m_e)·(2/√π) = √(8e/(π m_e)) であることによる)。

一定断面積 σ=σ0 (閾値なし) の解析検証: ∫E·exp(−E/Te)dE (0→∞) = Te^2 なので
k = σ0·√(8e/(π m_e))·Te^{1/2} = σ0·v̄ (v̄ = Maxwell 分布の平均速さ)。
test_fluid_coeffs.py の解析解テストはこれを直接検証する。

## 電子輸送係数 (μ_e, D_e) — 標準近似と BOLSIG 2項近似との差

厳密な (BOLSIG+ 型) 2項近似 Boltzmann ソルバーでは、移動度はエネルギー依存の
運動量移行衝突頻度 ν_m(E) を EEDF の微分 dF0/dE で重み付けた積分
    μ_e n_g = −(e / 3m_e) ∫ (E / ν_m(E)) · (dF0/dE) dE
で定義され、EEDF が非 Maxwell 的な場合は ν_m(E) のエネルギー依存性が
陽に効いてくる。本モジュールはその代わりに ν_m(Te) = ⟨σ_m v⟩ (Maxwell 平均の
単一衝突頻度、上記のレート係数と同じ積分) を求め、単純な Drude 型の関係

    μ_e n_g = e / (m_e · ν_m(Te))                    (Einstein: D_e = μ_e·Te)

で移動度を近似する。これは「局所平均エネルギー近似」と呼ばれる流体モデルで
広く使われる第一近似であり、Maxwell 分布に近い低電界・高圧領域では妥当だが、
EEDF が非 Maxwell 的に歪む条件 (低圧・強電界) では BOLSIG+ 型の解と数十%
オーダーで乖離しうる。第1弾 (Phase A) ではこの標準近似を採用し、より精密な
2項近似テーブルへの置き換えは将来の拡張とする。

## 単位系

- エネルギー E, Te: eV (呼び出し側・断面積テーブルと共通)
- 断面積 σ: m^2、レート係数 k: m^3/s、衝突周波数/n_g: m^3/s (ν_m = n_g · この値)
- 移動度は n_g との積 μ_e·n_g [1/(m·V·s)] でテーブル化する (n_g 非依存にして
  ガス圧・密度が変わっても再利用できるようにするため。呼び出し側で n_g で除す)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .particles import ME, QE
from .schema import XsProcess

# ---- 定数 ---------------------------------------------------------------------

#: 既定の電子温度グリッド [eV]。流体ソルバーの Te はこの範囲にクランプして
#: 係数テーブルを引く (0.05〜30 eV は CCP/ICP 放電の典型的な Te 範囲を広くカバー)。
TE_GRID_EV = np.geomspace(0.05, 30.0, 160)

# ⟨σv⟩ 積分の補助グリッド: 被積分関数 σ(E)·E·exp(−E/Te) は E = threshold + Te 付近に
# ピークを持ち exp(−E/Te) のスケール Te で減衰する。断面積表の格子点だけでは
# Te が表の格子間隔に対して極端に小さい/大きい場合に分解能不足になるため、
# [threshold, threshold + _INTEG_TE_SPAN·Te] に密な補助点を追加してから
# 断面積表の格子点と合成し、台形則で積分する。
_INTEG_TE_SPAN = 40.0   # exp(-40) ≈ 4e-18 で積分打ち切りの寄与は無視できる
_INTEG_AUX_N = 400


def _integrate_sigma_e_exp(
    e_table: np.ndarray, s_table: np.ndarray, threshold_ev: float, te_ev: float
) -> float:
    """∫ σ(E)·E·exp(−E/Te) dE を台形則で評価する (表の外は σ=0)。"""
    e_min, e_max = e_table[0], e_table[-1]
    lo = max(threshold_ev, e_min)
    hi = min(threshold_ev + _INTEG_TE_SPAN * te_ev, e_max)
    if hi <= lo:
        return 0.0
    aux = np.linspace(lo, hi, _INTEG_AUX_N)
    in_range = e_table[(e_table >= lo) & (e_table <= hi)]
    e_grid = np.union1d(np.union1d(aux, in_range), [lo, hi])
    # 表の外 (e_grid が表の [e_min, e_max] からわずかに外れる浮動小数誤差を含め)
    # は σ=0 として扱う (left/right=0.0)
    sigma = np.interp(e_grid, e_table, s_table, left=0.0, right=0.0)
    integrand = sigma * e_grid * np.exp(-e_grid / te_ev)
    return float(np.trapezoid(integrand, e_grid))


def rate_coefficient_table(proc: XsProcess, te_grid_ev: np.ndarray = TE_GRID_EV) -> np.ndarray:
    """k(Te) = ⟨σ v⟩ [m^3/s] を Te グリッド上でテーブル化する (Maxwell 平均)。

    等方 Maxwell 分布での平均 (モジュール docstring 参照):
        k(Te) = √(8e/(π m_e)) · Te^{-3/2} · ∫ σ(E)·E·exp(−E/Te) dE
    積分は断面積表の E 範囲で台形則を使う (表の外は σ=0)。閾値反応
    (excitation/ionization) は threshold_ev 未満で σ=0 (既存表がそう作られている
    ことを前提とするが、積分補助グリッドに閾値点そのものを含めることで
    閾値直上の立ち上がりが台形則で鈍らないようにしている)。

    Te ごとに被積分関数の形状が変わる (ピーク位置・減衰スケールが Te に比例) ため
    Te ループで補助グリッドを作り直す (E 方向の積分自体は各 Te でベクトル化)。
    """
    e_table = np.asarray(proc.energy_ev, dtype=np.float64)
    s_table = np.asarray(proc.sigma_m2, dtype=np.float64)
    te_grid_ev = np.asarray(te_grid_ev, dtype=np.float64)
    prefactor = math.sqrt(8.0 * QE / (math.pi * ME))

    k = np.empty_like(te_grid_ev, dtype=np.float64)
    for i, te in enumerate(te_grid_ev):
        integral = _integrate_sigma_e_exp(e_table, s_table, proc.threshold_ev, float(te))
        k[i] = prefactor * te ** -1.5 * integral
    return k


# ---- 電子輸送係数 ---------------------------------------------------------------


@dataclass
class ElectronTransport:
    """電子の運動量移行衝突頻度・移動度テーブル (Te の関数、n_g 非依存で正規化)。"""

    te_grid_ev: np.ndarray
    nu_m_per_ng: np.ndarray   # 運動量移行衝突頻度 / n_g [m^3/s] = ⟨σ_m v⟩ (ν_m = n_g・この値)
    mobility_n: np.ndarray    # μ_e・n_g [1/(m・V・s)] = e / (m_e・⟨σ_m v⟩)


def electron_transport(
    elastic: XsProcess, te_grid_ev: np.ndarray = TE_GRID_EV
) -> ElectronTransport:
    """弾性散乱断面積 (= 運動量移行断面積) から電子輸送係数テーブルを作る。

    eduPIC/Phelps の elastic 断面積は運動量移行断面積 (momentum-transfer
    cross section) として与えられている (mcc.py の弾性散乱は角度分布を等方
    近似しているため、実効的な運動量緩和は全散乱断面積ではなく運動量移行
    断面積で決まる)。したがって ν_m/n_g = ⟨σ_elastic v⟩ をそのままレート係数
    計算 (rate_coefficient_table) に流用できる。

    μ_e・n_g = e / (m_e・⟨σ_m v⟩) は Drude 型の標準近似 (BOLSIG 2項近似との
    差についてはモジュール docstring 参照)。D_e はテーブル化せず、呼び出し側が
    Einstein 関係 D_e = μ_e・Te (Te も呼び出し側で決まる局所量なので) の積として
    その場で計算する。
    """
    te_grid_ev = np.asarray(te_grid_ev, dtype=np.float64)
    nu_m_per_ng = rate_coefficient_table(elastic, te_grid_ev)
    # ν_m/n_g = 0 (断面積が全域0など、通常は起こらないが防御的に) では移動度が
    # 発散するため 0 として扱う (呼び出し側が「係数未定義」を検知できるように
    # 0 のままにしておく。安易に inf/nan を出さない)
    mobility_n = np.divide(
        QE,
        ME * nu_m_per_ng,
        out=np.zeros_like(nu_m_per_ng),
        where=nu_m_per_ng > 0.0,
    )
    return ElectronTransport(
        te_grid_ev=te_grid_ev, nu_m_per_ng=nu_m_per_ng, mobility_n=mobility_n
    )


# ---- 反応セット構築 --------------------------------------------------------------


@dataclass
class FluidReactions:
    """流体ソルバー (fluid1d.py) がそのまま使う反応・輸送係数一式。"""

    te_grid_ev: np.ndarray
    k_ion: np.ndarray          # 電離レート係数の和 [m^3/s]
    k_exc: np.ndarray          # 励起レート係数の和 [m^3/s]
    e_ion_ev: float            # 電離の代表閾値 [eV] (複数プロセスがあれば重み平均)
    e_exc_ev: float            # 励起の代表エネルギー損失 [eV] (同上)
    transport: ElectronTransport


def _weighted_threshold(procs: list[XsProcess], te_grid_ev: np.ndarray) -> float:
    """複数プロセスの threshold_ev を、各プロセスのレート係数の Te 積分量で重み平均する。

    プロセスが1個だけならその threshold_ev をそのまま返す (重み計算は自明に
    その1点へ収束する)。全プロセスの寄与がテーブル上でほぼ0 (異常系) の場合は
    単純平均にフォールバックする。
    """
    if not procs:
        return 0.0
    weights = np.array(
        [np.trapezoid(rate_coefficient_table(p, te_grid_ev), te_grid_ev) for p in procs]
    )
    total = float(weights.sum())
    thresholds = np.array([p.threshold_ev for p in procs])
    if total <= 0.0:
        return float(np.mean(thresholds))
    return float(np.sum(weights * thresholds) / total)


def build_fluid_reactions(
    electron_processes: list[XsProcess], te_grid_ev: np.ndarray = TE_GRID_EV
) -> FluidReactions:
    """electron_processes (MccSettings.electron_processes と同じ形式) から
    流体ソルバー用の反応・輸送係数一式を組み立てる。

    kind="elastic" が必須 (輸送係数の元になる)。kind="ionization" が最低1個
    必要 (電離が無いと放電を維持できないため、電離0個は設定ミスとみなしてエラー
    にする)。kind="excitation" は0個でもよい (k_exc は全て0のテーブルになる)。
    複数の同種プロセスがあれば k は和、threshold_ev は _weighted_threshold で
    重み平均する。
    """
    te_grid_ev = np.asarray(te_grid_ev, dtype=np.float64)
    elastic_procs = [p for p in electron_processes if p.kind == "elastic"]
    ion_procs = [p for p in electron_processes if p.kind == "ionization"]
    exc_procs = [p for p in electron_processes if p.kind == "excitation"]

    if not elastic_procs:
        raise ValueError("build_fluid_reactions: kind='elastic' のプロセスが必要です")
    if not ion_procs:
        raise ValueError("build_fluid_reactions: kind='ionization' のプロセスが最低1個必要です")

    # 弾性が複数あることは通常想定しないが、あれば ⟨σ_m v⟩ を単純に合算する
    # (運動量移行断面積は加法的: 複数の弾性チャンネルがあれば全断面積の和と等価)
    if len(elastic_procs) == 1:
        transport = electron_transport(elastic_procs[0], te_grid_ev)
    else:
        nu_sum = np.zeros_like(te_grid_ev)
        for p in elastic_procs:
            nu_sum += rate_coefficient_table(p, te_grid_ev)
        mobility_n = np.divide(
            QE, ME * nu_sum, out=np.zeros_like(nu_sum), where=nu_sum > 0.0
        )
        transport = ElectronTransport(
            te_grid_ev=te_grid_ev, nu_m_per_ng=nu_sum, mobility_n=mobility_n
        )

    k_ion = np.zeros_like(te_grid_ev)
    for p in ion_procs:
        k_ion += rate_coefficient_table(p, te_grid_ev)
    k_exc = np.zeros_like(te_grid_ev)
    for p in exc_procs:
        k_exc += rate_coefficient_table(p, te_grid_ev)

    e_ion_ev = _weighted_threshold(ion_procs, te_grid_ev)
    e_exc_ev = _weighted_threshold(exc_procs, te_grid_ev) if exc_procs else 0.0

    return FluidReactions(
        te_grid_ev=te_grid_ev,
        k_ion=k_ion,
        k_exc=k_exc,
        e_ion_ev=e_ion_ev,
        e_exc_ev=e_exc_ev,
        transport=transport,
    )


# ---- 補間ヘルパー ---------------------------------------------------------------


def interp_loglog(
    te_grid: np.ndarray, table: np.ndarray, te: np.ndarray | float
) -> np.ndarray | float:
    """k(Te) テーブルの log-log 補間 (ベクトル化)。

    レート係数・移動度は Te に対しべき乗則的 (log-log でほぼ直線) に振る舞う
    ことが多く、log 空間での線形補間 (= 元空間での log-log 補間) の方が線形
    補間より滑らかで自然な近似になる。ただしテーブルに 0 (閾値未満の電離/励起
    レートなど) が含まれる区間は log(0) が特異点になるため、その区間だけ通常の
    線形補間にフォールバックする。範囲外の問い合わせはテーブル端点の値へ
    クランプする (係数テーブルの Te 範囲外は外挿せず定数扱いにする、という
    流体ソルバー側の既定方針に合わせる。prompts/104 の「テーブル範囲外は
    クランプ + warning」に対応 — warning はソルバー側の責務)。
    """
    te_grid = np.asarray(te_grid, dtype=np.float64)
    table = np.asarray(table, dtype=np.float64)
    te_arr = np.asarray(te, dtype=np.float64)
    scalar_input = te_arr.ndim == 0

    te_q = np.atleast_1d(te_arr)
    te_c = np.clip(te_q, te_grid[0], te_grid[-1])
    # 区間インデックス: te_c を含む [te_grid[idx], te_grid[idx+1]] を選ぶ
    idx = np.searchsorted(te_grid, te_c, side="right") - 1
    idx = np.clip(idx, 0, len(te_grid) - 2)
    t0, t1 = te_grid[idx], te_grid[idx + 1]
    y0, y1 = table[idx], table[idx + 1]

    # 線形補間 (フォールバック用) の重みは te 空間の線形分率
    frac_lin = np.where(t1 > t0, (te_c - t0) / (t1 - t0), 0.0)
    linear_interp = y0 + frac_lin * (y1 - y0)

    # log-log 補間は log(te) 空間での線形分率を使う (te_grid は常に正なので
    # log は well-defined。これが「log-log」たる所以で、frac_lin をそのまま
    # 使うと単に log(y) を te 線形で補間するだけになり、べき乗則テーブルで
    # 厳密性が失われる)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_log = np.where(t1 > t0, (np.log(te_c) - np.log(t0)) / (np.log(t1) - np.log(t0)), 0.0)
        both_positive = (y0 > 0.0) & (y1 > 0.0)
        y0_safe = np.where(both_positive, y0, 1.0)
        y1_safe = np.where(both_positive, y1, 1.0)
        log_interp = np.exp(np.log(y0_safe) + frac_log * (np.log(y1_safe) - np.log(y0_safe)))
    # frac_lin==0 (グリッド点そのもの) は log/exp の往復誤差すら避けて厳密に y0 を返す
    result = np.where(frac_lin == 0.0, y0, np.where(both_positive, log_interp, linear_interp))

    if scalar_input:
        return float(result[0])
    return result
