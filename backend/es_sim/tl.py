"""VHF 定在波スタディ — 非線形径方向伝送線路モデル (prompts/101)。

大面積 VHF (very high frequency) CCP (容量結合プラズマ) では、駆動波長 λ0=c/f0
(100 MHz で 3 m) が電極寸法に対して無視できなくなり、電極面内 (径方向) に
定在波が生じて中心と周辺でシース電圧・プラズマ密度が不均一になる
(「定在波効果」)。さらにシースの非線形容量 (電荷-電圧関係が2次) が高調波を
生成し、各高調波はそれぞれ (基本波より短い) 波長で独自の定在波を張る。

本モジュールは、この現象の標準モデルである「非線形径方向伝送線路モデル」
(Lieberman, Lichtenberg, Kaganovich, Selwyn, Plasma Sources Sci. Technol. 11,
283 (2002); Chabert, Lichtenberg, Lieberman, Marakhtanov, Phys. Plasmas 11,
1775 (2004); Chabert, J. Phys. D: Appl. Phys. 40, R63 (2007) レビュー) を
実装する。円板電極 (半径 R、ギャップ l)・中心給電・軸対称の径方向 1D。

## 物理モデル

電極間電圧 V(r,t)・電極を流れる径方向全電流 I(r,t) が、伝送線路 (TL) 方程式

    ∂V/∂r = −L'(r) ∂I/∂t,     L'(r) = μ0 l / (2π r)   [H/m]         (1)
    ∂I/∂r = −2π r J_z(r,t)                                          (2)

に従う。J_z [A/m²] は放電を軸方向に流れる電流密度で、(2) は「電極表面の
局所電荷保存」そのものである — 各半径ノードには上下シース(非線形容量)+
バルク (電子慣性 L_b・衝突抵抗 R_b) の直列シャント回路があり、

    V_s(q) = q² / (2 e n_s ε0)                (行列シースの電荷-電圧関係、sheath_law="matrix")
    dq_a/dt = +J_z,  dq_b/dt = −J_z            (上シース・下シースの面電荷)
    V(r,t) = V_s(q_a) − V_s(q_b) + L_b dJ_z/dt + R_b J_z            (3)

    L_b = m_e d / (e² n_e)  [H·m²],  R_b = m_e ν_m d / (e² n_e)  [Ω·m²]
    d = l − 2 s0  (s0: 平衡シース厚。上下対称なので s_a0=s_b0=s0 とする)

符号規約: V>0 のとき上電極が正 → 上シースが厚くなる (q_a 増加)。対称平衡
(V=0) では q_a=q_b=q0=e n_s s0。q<0 は物理的に不可 (シースが完全に潰れる)
なので 0 にクリップし、崩壊中はその側のシース電圧を 0 とする (整流性)。

## シース則 sheath_law: "child" (既定) / "matrix" (prompts/102)

行列シース (V_s∝q²) を対称放電 (上下直列) で使うと、dq_a/dt=+J_z・dq_b/dt=−J_z
により q_a+q_b=2q0 が厳密に保存されるため

    V_a − V_b = (q_a² − q_b²)/(2 e n_s ε0) = (q_a+q_b)(q_a−q_b)/(2 e n_s ε0)
              = (q0/(e n_s ε0))・(q_a−q_b)          (q_a+q_b=2q0 を代入)

と Δq=(q_a−q_b) に対して**厳密に線形**になってしまう (2次関数どうしの差でも
和が保存されている限り1次関数に潰れるという代数的な事実)。この場合、高調波は
シース崩壊 (q<0 クリップ) の過渡でしか生成されず、しかもクリップは
「q_a+q_b を押し上げる」一方向のラチェットとして働くため (q が0未満に
なる方向にしかクリップは効かない)、系は数周期でクリップが起こらない
(=厳密に線形な) 新しい平衡点に緩和し、高調波は自己消滅する。これは
「定在波+波長短縮による高調波生成」という本モデルの検証目的にとって
致命的である (prompts/101 のテスト2が定常窓ではなく過渡窓を使わざるを
得なかった理由)。

Child-Langmuir 則 (シース内でイオン密度が空間的に減衰する現実的な
プロファイルを反映した電荷-電圧関係) V_s(q)=K・q^{4/3} を使うと、
(q0+Δ)^{4/3}−(q0−Δ)^{4/3} は Δ の奇数次項 (Δ, Δ³, Δ⁵, ...) をすべて含み
(4/3 乗は整数次多項式に潰れない)、対称放電でも定常的に奇数次高調波
(3f0, 5f0, ...) が生成される。偶数次は q_a↔q_b, Δ→−Δ の対称性で厳密に
相殺する (実放電で観測される偶数次高調波は電極間の非対称性に由来する
別現象であり、本モデルの対称配置では意図的にスコープ外とする)。

**K の決め方 (小信号容量整合)**: 平衡点 q0 での微分容量 dV_s/dq|_{q0} が
sheath_law に依らず matrix と一致するように K を選ぶ:

    matrix: dV_s/dq|_{q0} = q0/(e n_s ε0) = s0/ε0   (q0=e n_s s0 を代入)
    child:  dV_s/dq|_{q0} = K・(4/3)・q0^{1/3}
    → K = (3/4)・(s0/ε0)・q0^{-1/3}

こう選ぶことで、線形極限 (小振幅 V0、Δ≪q0) での実効波長 λ_eff は
sheath_law に依らず同一の分散関係 v_p=c√(2s0/l) に従う (小信号では
V_s の高次項は寄与せず微分容量だけで決まるため) — 波長短縮の理論検証
テストが両則で共通に成り立つ。

q<0 のクリップ (シース崩壊の整流) は child でも matrix と同様に適用する
(強駆動時にシースが物理的に潰れるのは電荷-電圧関係の形に依らない現象の
ため)。linear=True (テスト専用) は sheath_law に依らず従来通り線形容量
V_s=q·s0/ε0 のまま (真に恒等的な線形分散関係の検証用)。

## 数値スキーム — 局所結合・半陰的 (per-node implicit) 更新

(1)(2)(3) から I を消去すると (∂/∂r・∂/∂t の可換性を使う)、V だけの2階の
波動方程式が得られる:

    ∂/∂r[ (1/L'(r)) ∂V/∂r ] = 2π r ∂J_z/∂t                          (4)

線形シース (C_a=ε0/s_a0, C_b=ε0/s_b0 の直列合成 C_eff=ε0/(s_a0+s_b0)、
J_z≈C_eff ∂V/∂t) を代入すると (1/r)∂/∂r(r ∂V/∂r) = (1/v_p²) ∂²V/∂t²
(v_p=c√(s_a0+s_b0)/√l) という円柱波動方程式に一致する (r 依存性が
打ち消し合うのは L'(r)∝1/r と単位長さ容量 2πr·C_eff∝r の積が r に依らない
ため — Bessel 関数的な定在波を生む所以)。

(4) の右辺 ∂J_z/∂t を「前ステップの J_z との後退差分」で評価し、シャント
ODE (3) を J_z について解いた式を代入すると、J_z は V の一次式
(J_z = a(r)·V + b(r)、a,b は q_a,q_b,J_z の直前値から決まる既知量) になる
(非線形シースは各ステップ q_a,q_b まわりに線形化し、V_s の局所勾配
∂V_s/∂q を使う — 1ステップだけのニュートン線形化で、次ステップでは新しい
q で再線形化するため誤差は蓄積しない)。これを (4) の離散化 (V を整数
ノード r_i、L'(r) を半ノード r_{i+1/2} で評価する staggered な二階差分—
1D FDTD の空間格子と同型) に代入すると、対角成分だけが J_z の依存分だけ
補正された三重対角行列になり、scipy.linalg.solve_banded で解ける。

**なぜこの (V と J_z を同時に解く) 方式が必要か**: J_z を I の空間差分から
「測定」してから (3) 経由で V を求める、または J_z を V (1ステップ前の値)
から前進的に駆動して I を別に leapfrog するという素朴な明示スキームは、
どちらも L_b・R_b の寄与を介した正のフィードバックで無条件に (dt を
どれだけ小さくしても) 発散することを実装時に数値実験で確認した — J_z の
新しい値が「V の新しい値」自体に依存する項を、既知の値として扱ってしまう
(半歩・1歩遅れさせる) と、その遅れが r 方向の結合を通じて指数的に増幅
される。J_z を V の新しい値の関数として陰的に解き、(4) に代入して一度に
解くことで、この正フィードバックが構造的に断たれ、無条件に安定になる
(電荷 q_a,q_b の更新自体は J_z が確定してからの明示的な時間積分なので、
実質的な演算コストは三重対角ソルバー1回/ステップ (pic1d.py のポアソン
ソルバーと同型) で、節点ループは一切発生しない)。

外側境界 r=R: 開放端 I(R)=0 は、初期条件 (V=I=0) から出発して恒等的に
∂V/∂r(R,t)=0 (ノイマン条件) に帰着する (eq(1) より ∂I/∂t∝-∂V/∂r なので、
∂V/∂r(R,t) の時間微分が恒等的に0 であり、t=0 での値も0 だから)。
内側 r_feed (既定 R/50): r→0 では L'(r) が発散するため、給電点を
ちょうど r=0 に置けない (同軸給電線の実効半径の近似としても自然)。
給電点の V はこの陰的解を使わず駆動電圧 V0·ramp(t)·sin(2π f0 t) で
上書きする (Dirichlet)。

径方向の全電流 I(r,t) (給電電力の評価にのみ使用) は、この (安定な) V の
時系列から eq(1) の局所 leapfrog (最近傍ノードの V 差分のみ使う、標準
FDTD と同型) で診断的に求める — 上記の「V と J_z を同時に解く」主計算とは
独立な後付けの副生成物であり、主計算の安定性には影響しない。

## CFL 条件

線形化位相速度 v_p = c √(s_a0+s_b0) / √l (docstring 冒頭参照)。
dt ≤ 0.5 Δr / v_p (安全率 0.5、自動設定・上書き可)。

## テスト専用オプション

linear=True で V_s(q)=q·s0/ε0 (線形容量、∂V_s/∂q=s0/ε0 で q に依らず一定)
に切り替える。線形極限での波長短縮 λ_eff≈λ0·√((s_a0+s_b0)/l) の検証に使う
(tests/test_tl.py)。
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.linalg import solve_banded

from .fem import EPS0
from .particles import ME, QE
from .schema import Project, TlSettings

# 真空の光速 [m/s] (SI 定義値) と透磁率 [H/m] (2019年 SI 再定義後も実用上厳密な値)。
# fem.py の EPS0 とは独立な定数として持つ (相互に厳密な c²=1/(EPS0*MU0) を要求
# しない — 本モデルの用途では 1e-10 相対程度の差は無関係)。
C0 = 299792458.0
MU0 = 4.0 * math.pi * 1.0e-7

# 給電電圧をなめらかに立ち上げるランプ周期数。t=0 でいきなり振幅 V0 を掛けると
# 立ち上がりの不連続 (実質ステップ入力) が広帯域の過渡応答を生み、それが
# 定常状態の高調波解析 (FFT窓) に残留ノイズとして混入しうるため、
# コサイン窓で滑らかに (値も微分も 0 から) 立ち上げる。n_periods (既定200) に
# 対して 5 周期は十分小さく、定常化の妨げにならない
N_RAMP_PERIODS = 5.0

# spectrum_probe (3点の振幅スペクトル) の上限次数 (40·f0 まで、prompts/101)
SPECTRUM_MAX_HARMONIC = 40


def _ramp(t: float, t_ramp: float) -> float:
    """滑らかな立ち上げ窓 (コサイン窓、値・微分ともに t=0 で 0)。"""
    if t >= t_ramp:
        return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * t / t_ramp))


class TlSimulation:
    """VHF 定在波 (非線形径方向伝送線路) シミュレーション本体。

    run() で n_periods 分 (n_steps_total ステップ) を一括実行し、結果 dict を返す。
    継続実行 (continue) には対応しない (毎回フルの定常化をやり直す設計、
    プロンプトの指示通り)。
    """

    def __init__(self, project: Project, linear: bool = False):
        if project.tl is None:
            raise ValueError("project.tl が指定されていません")
        self.project = project
        self.s: TlSettings = project.tl
        self.linear = linear
        s = self.s

        # ---- 幾何・格子 -----------------------------------------------------
        self.radius = float(s.radius_m)
        self.gap = float(s.gap_m)
        self.sheath0 = float(s.sheath_m)  # s0 (上下対称)
        # r_feed = R/50 (既定): r→0 の L'(r) 特異性回避。同軸給電線の実効半径の
        # 近似としても自然 (docstring 参照)
        self.r_feed = self.radius / 50.0
        self.n_r = int(s.n_r)
        self.r = np.linspace(self.r_feed, self.radius, self.n_r)
        self.dr = float(self.r[1] - self.r[0])
        self.r_half = 0.5 * (self.r[:-1] + self.r[1:])  # I の半ノード (n_r-1 点)
        self.Lp_half = MU0 * self.gap / (2.0 * math.pi * self.r_half)  # L'(r) [H/m]
        self.invL_half = 1.0 / self.Lp_half

        # ---- プラズマ・シース パラメータ ------------------------------------
        self.n_e = float(s.n_e_m3)
        self.n_s = float(s.n_s_ratio) * self.n_e
        self.nu_m = float(s.nu_m_hz)
        d_bulk = self.gap - 2.0 * self.sheath0  # バルク厚 (validator で >0 保証)
        self.Lb = ME * d_bulk / (QE**2 * self.n_e)       # [H・m^2]
        self.Rb = ME * self.nu_m * d_bulk / (QE**2 * self.n_e)  # [Ω・m^2]
        self.q0 = QE * self.n_s * self.sheath0  # 平衡シース面電荷 [C/m^2]
        self.sheath_law = s.sheath_law
        # Child 則の係数 K (V_s=K・q^{4/3})。平衡点 q0 での微分容量を matrix
        # (dV_s/dq=s0/ε0) と一致させる整合条件 K=(3/4)・(s0/ε0)・q0^{-1/3}
        # (docstring の「シース則」節、K の導出参照) — これにより線形極限の
        # 実効波長は sheath_law に依らず同一になる
        self._K_child = 0.75 * (self.sheath0 / EPS0) * self.q0 ** (-1.0 / 3.0)

        # ---- 駆動 ------------------------------------------------------------
        self.f0 = float(s.freq_hz)
        self.v0 = float(s.v0)
        self.t_ramp = N_RAMP_PERIODS / self.f0

        # ---- CFL・dt ----------------------------------------------------------
        # 線形化位相速度 (docstring 参照)。シースを線形容量とみなした保守的な見積り
        v_p_lin = C0 * math.sqrt(2.0 * self.sheath0 / self.gap)
        dt_cfl = 0.5 * self.dr / v_p_lin
        self.warnings: list[str] = []
        if s.dt is not None:
            self.dt = float(s.dt)
            if self.dt > dt_cfl:
                self.warnings.append(
                    f"指定された dt={self.dt:.3g}s は CFL 上限 {dt_cfl:.3g}s を超えています "
                    "(数値不安定の恐れ。dt を小さくすることを推奨します)"
                )
        else:
            self.dt = dt_cfl

        # ---- 総ステップ数・FFT窓・プローブ窓 -----------------------------------
        self.n_steps_total = max(1, round(s.n_periods / self.f0 / self.dt))
        n_fft_steps = max(1, round(s.n_fft_periods / self.f0 / self.dt))
        self.n_fft_steps = min(n_fft_steps, self.n_steps_total)
        self.fft_start = self.n_steps_total - self.n_fft_steps
        n_probe_steps = max(2, round(2.0 / self.f0 / self.dt))
        self.n_probe_steps = min(n_probe_steps, self.n_steps_total)
        self.probe_start = self.n_steps_total - self.n_probe_steps

        # ---- プローブ位置: 中心付近 (r_feed)・R/2・外周 (R) -------------------
        self.idx_center = 0
        self.idx_mid = int(np.argmin(np.abs(self.r - self.radius / 2.0)))
        self.idx_edge = self.n_r - 1

        # ---- 状態 (すべて t=0 で厳密に静止: q_a=q_b=q0, J_z=0, V=0, I=0) -------
        self.V = np.zeros(self.n_r)
        self.I = np.zeros(self.n_r - 1)  # 診断専用 (docstring 参照)
        self.qa = np.full(self.n_r, self.q0)
        self.qb = np.full(self.n_r, self.q0)
        self.Jz = np.zeros(self.n_r)
        self.clipped = False

        # ---- 三重対角行列の非対角成分 (dt・L'(r) は不変なので初期化時に1回だけ) ---
        # eq(4) の左辺 ∂/∂r[(1/L') ∂V/∂r] を二階差分: 内点 i の行は
        # invL_half[i]*(V[i+1]-V[i])/dr^2 - invL_half[i-1]*(V[i]-V[i-1])/dr^2。
        # 対角成分は J_z の V 依存分 (a(r)) を含むため毎ステップ更新する (_step 参照)。
        n = self.n_r
        self._ab_offdiag = np.zeros((3, n))
        self._ab_offdiag[0, 2:] = self.invL_half[1:] / self.dr**2      # upper (row i, col i+1)
        self._ab_offdiag[2, :-2] = self.invL_half[:-1] / self.dr**2    # lower (row i, col i-1)
        self._diag_base = np.zeros(n)
        self._diag_base[1:-1] = -(self.invL_half[:-1] + self.invL_half[1:]) / self.dr**2

        # ---- 高調波アキュムレータ (回転子方式、時系列を保存せず O(1) メモリで
        # 正確な DFT ビンを積算する。窓が f0 の整数周期なのでリークが無い) -------
        n_bins = int(s.n_harm) + 1
        self.n_harm = int(s.n_harm)
        self._rot_v = np.exp(-1j * 2.0 * math.pi * np.arange(n_bins) * self.f0 * self.dt)
        self._phasor_v = np.ones(n_bins, dtype=complex)
        self._acc_v = np.zeros((n_bins, self.n_r), dtype=complex)
        self._acc_j = np.zeros((n_bins, self.n_r), dtype=complex)

        sb = SPECTRUM_MAX_HARMONIC + 1
        self._rot_p = np.exp(-1j * 2.0 * math.pi * np.arange(sb) * self.f0 * self.dt)
        self._phasor_p = np.ones(sb, dtype=complex)
        self._acc_probe = np.zeros((sb, 3), dtype=complex)

        self._power_acc = np.zeros(self.n_r)
        self._feed_power_acc = 0.0
        self._fft_count = 0

        self._probe_t = np.zeros(self.n_probe_steps)
        self._probe_v = np.zeros((self.n_probe_steps, 3))

        self.step_count = 0

    # ---- シースの電荷-電圧関係とその局所勾配 --------------------------------

    def _vs(self, q: np.ndarray) -> np.ndarray:
        """シースの電荷-電圧関係。既定は sheath_law="child" (Child-Langmuir 型)。

        linear=True (テスト専用) では sheath_law に依らず V_s=q·s0/ε0 (線形容量)
        に切り替える — 線形極限の波長短縮 λ_eff≈λ0√((s_a0+s_b0)/l) の検証用
        (tests/test_tl.py)。sheath_law="child"/"matrix" の使い分けと、対称放電で
        matrix が厳密に線形化してしまう理由は docstring の「シース則」節を参照。
        """
        if self.linear:
            return q * self.sheath0 / EPS0
        if self.sheath_law == "child":
            return self._K_child * np.power(q, 4.0 / 3.0)
        return q**2 / (2.0 * QE * self.n_s * EPS0)

    def _dvs_dq(self, q: np.ndarray) -> np.ndarray:
        """∂V_s/∂q (半陰的更新の線形化・solve_banded の対角成分に使う局所勾配)。

        child 則ではこの勾配自体が q^{1/3} で非線形に変化するため、matrix と
        同じく「今ステップの q まわりの1ステップ線形化」がそのまま安定性の
        要になる (docstring の「数値スキーム」節参照 — L_b・R_b 項の陰的解法と
        同じ理由で、この勾配を陽的に (前ステップ値のまま) 扱うと発散する)。
        """
        if self.linear:
            return np.full_like(q, self.sheath0 / EPS0)
        if self.sheath_law == "child":
            return self._K_child * (4.0 / 3.0) * np.power(q, 1.0 / 3.0)
        return q / (QE * self.n_s * EPS0)

    def _v_feed(self, t: float) -> float:
        return _ramp(t, self.t_ramp) * self.v0 * math.sin(2.0 * math.pi * self.f0 * t)

    # ---- 1ステップ ---------------------------------------------------------

    def _step(self, t: float) -> None:
        """時刻 t (=step_count*dt) の状態から t+dt へ1ステップ進める。

        V と J_z を三重対角ソルバーで同時に解く (docstring の「数値スキーム」参照)。
        """
        dt = self.dt
        qa, qb, Jz_old = self.qa, self.qb, self.Jz

        # シャント ODE (eq3) を J_z について解いた式: J_z_new = a(r)*V_new + b(r)。
        # 非線形シースは今ステップの qa,qb まわりに線形化する (1ステップぶんの
        # ニュートン線形化。次ステップは新しい qa,qb で再線形化するため誤差は
        # 蓄積しない — L_b・R_b の寄与を「既知」として扱うと無条件に発散する
        # ことを実装時に確認しているため、この線形化が安定性の要である)
        dvs_a = self._dvs_dq(qa)
        dvs_b = self._dvs_dq(qb)
        denom = self.Lb / dt + self.Rb + dt * (dvs_a + dvs_b)
        a_coef = 1.0 / denom
        dVs_old = self._vs(qa) - self._vs(qb)
        b_coef = (self.Lb / dt * Jz_old - dVs_old) * a_coef

        # eq(4) の三重対角行列: 対角成分だけ J_z の V 依存分 (2π r_i * a_coef_i / dt)
        # を追加する (非対角成分は L'(r) のみで決まり不変、__init__ で構築済み)
        ab = self._ab_offdiag.copy()
        ab[1, 1:-1] = self._diag_base[1:-1] - 2.0 * math.pi * self.r[1:-1] * a_coef[1:-1] / dt
        ab[1, 0] = 1.0   # 給電点: Dirichlet (恒等行)
        ab[1, -1] = 1.0  # 外周: ノイマン (V[-1]-V[-2]=0、下記 lower で -1 を入れる)
        ab[2, -2] = -1.0

        rhs = np.empty(self.n_r)
        rhs[1:-1] = 2.0 * math.pi * self.r[1:-1] * (b_coef[1:-1] - Jz_old[1:-1]) / dt
        rhs[0] = self._v_feed(t + dt)
        rhs[-1] = 0.0

        V_new = solve_banded((1, 1), ab, rhs)
        Jz_new = a_coef * V_new + b_coef

        # 電荷を明示的に時間積分する (J_z は上ですでに確定済みの既知量)。
        # シース崩壊 (q<0) は物理的に不可なので 0 にクリップする (整流性)。
        #
        # このクリップは linear=True (テスト専用の線形容量モード) では行わない —
        # 線形モードは「V_s が q によらず一定勾配を持つ理想化された線形波動方程式」
        # (docstring の式(4)・線形分散関係 v_p=c√((s_a0+s_b0)/l)) を検証するための
        # ものであり、q<0 という物理的クリップ (整流) 自体が非線形動作そのものなので、
        # これを線形モードに混ぜると (a) 波長短縮の理論式と一致しなくなる、
        # (b) 「非線形をオンにしたときだけ高調波が出る」という tests/test_tl.py の
        # 比較試験が成立しなくなる (クリップは linear フラグに関係なく同じ q(t) の
        # 軌道に対して起こるため、線形モードでも同程度のクリップ由来の高調波が
        # 混入し、非線形モードとの差が消えてしまう — 実装時にこれを数値実験で確認した)。
        # 線形モードでは q が一時的に負になっても V_s=q·s0/ε0 の式がそのまま
        # 外挿的に成立する「理想化された線形容量」として扱う (q<0 の物理的意味を
        # 問わない、あくまで分散関係検証用の数学的モデル)。
        qa_new = qa + dt * Jz_new
        qb_new = qb - dt * Jz_new
        if not self.linear:
            if np.any(qa_new < 0.0) or np.any(qb_new < 0.0):
                self.clipped = True
            np.clip(qa_new, 0.0, None, out=qa_new)
            np.clip(qb_new, 0.0, None, out=qb_new)

        # 径方向全電流 I (診断専用、給電電力にのみ使用): eq(1) の局所 leapfrog
        # (最近傍ノードの V 差分のみ使う、docstring 参照)。安定な V の時系列
        # から後付けで求めるだけなので主計算の安定性には影響しない
        v_feed_prev = self.V[0]  # このステップで使った (直前の) 駆動電圧
        self.I = self.I - (dt / (self.Lp_half * self.dr)) * (self.V[1:] - self.V[:-1])
        i_feed = self.I[0]  # 給電点直近の半ノード電流 (I(r_feed) の近似)

        self.V = V_new
        self.qa = qa_new
        self.qb = qb_new
        self.Jz = Jz_new

        abs_n = self.step_count + 1  # このステップ完了後の絶対ステップ数

        # ---- FFT 窓 (定常化後の n_fft_periods) ---------------------------------
        if abs_n > self.fft_start:
            self._acc_v += self._phasor_v[:, None] * V_new[None, :]
            self._acc_j += self._phasor_v[:, None] * Jz_new[None, :]
            self._phasor_v *= self._rot_v

            probe_vals = V_new[[self.idx_center, self.idx_mid, self.idx_edge]]
            self._acc_probe += self._phasor_p[:, None] * probe_vals[None, :]
            self._phasor_p *= self._rot_p

            self._power_acc += self.Rb * Jz_new**2 * dt
            self._feed_power_acc += v_feed_prev * i_feed * dt
            self._fft_count += 1

        # ---- v_probe (最終2周期の生波形プレビュー) ------------------------------
        if abs_n > self.probe_start:
            k = abs_n - self.probe_start - 1
            if 0 <= k < self.n_probe_steps:
                self._probe_t[k] = abs_n * dt
                self._probe_v[k, :] = V_new[[self.idx_center, self.idx_mid, self.idx_edge]]

        self.step_count = abs_n

    # ---- 結果組み立て -------------------------------------------------------

    def _harmonics(self) -> dict:
        n_bins = self.n_harm + 1
        if self._fft_count > 0:
            scale = np.where(np.arange(n_bins) == 0, 1.0 / self._fft_count, 2.0 / self._fft_count)
            v_amp = np.abs(self._acc_v) * scale[:, None]
            j_amp = np.abs(self._acc_j) * scale[:, None]
        else:
            v_amp = np.zeros((n_bins, self.n_r))
            j_amp = np.zeros((n_bins, self.n_r))
        return {
            "n": list(range(n_bins)),
            "v": v_amp.tolist(),
            "j": j_amp.tolist(),
        }

    def _lambda_eff(self, v_amp: np.ndarray) -> dict:
        """基本波 |V_1(r)| の隣接する極小 (節) 間隔 × 2 から λ_eff を推定する。

        位相勾配 (dθ/dr) は定在波では節でπ跳ぶため使えない (プロンプト指摘通り)。
        極小が2つ未満しか見つからなければ null (推定不能)。
        """
        lambda0 = C0 / self.f0
        lambda_eff = None
        if self._fft_count > 0 and self.n_r >= 3:
            v1 = v_amp[1, :]
            is_min = (v1[1:-1] < v1[:-2]) & (v1[1:-1] < v1[2:])
            idxs = np.nonzero(is_min)[0] + 1
            if len(idxs) >= 2:
                spacings = np.diff(self.r[idxs])
                lambda_eff = float(2.0 * np.mean(spacings))
        ratio = lambda_eff / lambda0 if lambda_eff is not None else None
        return {"lambda_m": lambda_eff, "lambda0_m": lambda0, "ratio": ratio}

    def _power(self) -> dict:
        if self._fft_count > 0:
            window_s = self._fft_count * self.dt
            p = self._power_acc / window_s
        else:
            p = np.zeros(self.n_r)
        pmin, pmax = float(p.min()), float(p.max())
        max_over_min = pmax / pmin if pmin > 0.0 else None
        weights = self.r  # 2π·dr は共通因子なので正規化された統計量では相殺する
        mean_p = float(np.average(p, weights=weights))
        std_over_mean = None
        if mean_p > 0.0:
            var_p = float(np.average((p - mean_p) ** 2, weights=weights))
            std_over_mean = math.sqrt(var_p) / mean_p
        return {
            "p": p.tolist(),
            "max_over_min": max_over_min,
            "area_weighted_std_over_mean": std_over_mean,
        }

    def feed_power(self) -> float:
        if self._fft_count == 0:
            return 0.0
        return self._feed_power_acc / (self._fft_count * self.dt)

    def _spectrum_probe(self) -> dict:
        sb = SPECTRUM_MAX_HARMONIC + 1
        if self._fft_count > 0:
            scale = np.where(np.arange(sb) == 0, 1.0 / self._fft_count, 2.0 / self._fft_count)
            amp = np.abs(self._acc_probe) * scale[:, None]
        else:
            amp = np.zeros((sb, 3))
        freq = np.arange(sb) * self.f0
        return {
            "freq_hz": freq.tolist(),
            "center": amp[:, 0].tolist(),
            "mid": amp[:, 1].tolist(),
            "edge": amp[:, 2].tolist(),
        }

    def _v_probe(self) -> dict:
        return {
            "t": self._probe_t.tolist(),
            "center": self._probe_v[:, 0].tolist(),
            "mid": self._probe_v[:, 1].tolist(),
            "edge": self._probe_v[:, 2].tolist(),
        }

    def build_result(self) -> dict:
        harmonics = self._harmonics()
        v_amp = np.asarray(harmonics["v"])
        warnings = list(self.warnings)
        if self.clipped:
            warnings.append(
                "シース崩壊 (面電荷が0にクリップ) が発生しました。強非線形の目安ですが、"
                "モデル上は正常動作です"
            )
        return {
            "r": self.r.tolist(),
            "probe_r": {
                "center": float(self.r[self.idx_center]),
                "mid": float(self.r[self.idx_mid]),
                "edge": float(self.r[self.idx_edge]),
            },
            "harmonics": harmonics,
            "lambda_eff": self._lambda_eff(v_amp),
            "power": self._power(),
            "feed_power": self.feed_power(),
            "v_probe": self._v_probe(),
            "spectrum_probe": self._spectrum_probe(),
            "warnings": warnings,
            "dt": self.dt,
            "n_steps": self.n_steps_total,
            "settings": self.s.model_dump(),
        }

    # ---- 実行 ----------------------------------------------------------------

    def run(self, callback=None, should_stop=None) -> dict:
        """n_steps_total ステップ実行し、結果 dict (done.result 相当) を返す。

        continue には対応しない (毎回フルの定常化からやり直す。プロンプト通り)。
        """
        report_every = max(1, self.n_steps_total // 200)
        t0 = time.perf_counter()
        for step in range(self.n_steps_total):
            if should_stop is not None and should_stop():
                break
            t = step * self.dt
            self._step(t)
            if callback is not None and (
                self.step_count % report_every == 0 or self.step_count == self.n_steps_total
            ):
                callback(self.step_count, self.n_steps_total, time.perf_counter() - t0)
            if not np.all(np.isfinite(self.V)):
                raise ValueError(
                    f"数値発散を検出しました (step {self.step_count}: V が非有限値)。"
                    "dt を小さくする、n_r を増やす等を検討してください"
                )
        return self.build_result()
