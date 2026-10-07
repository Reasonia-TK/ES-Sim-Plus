"""時間発展の収束の判定 (prompts/137)。

PIC・流体の時間発展で、周期平均の量が定常に落ち着いたかを判定する。判定の周期は RF の最低周波数の周期の整数倍
(``rf_periods``。境は絶対時刻の倍数で、阻止コンデンサの周期の集計と同じ)、RF が無ければ ``steps`` ステップ。
エンジンは毎ステップ (PIC は ``stride`` ステップおき) 場と積分量を ``ConvergenceMonitor.sums`` に足し込み、周期の
境を越えたステップのあとに ``close_period`` を呼ぶ。

量 (どれも周期平均):

- 場 φ・n_e: 粗いブロック (2D は約 32×32、1D は 64 個、``BlockMap``) へ体積で平均してから比べる。PIC の粒子の雑音が
  減り、AMR の再格子化で節点が変わっても比べられる。ノルムはブロックの体積で重み付けした L2。
- 積分量: 電子・イオンの総数 N_e・N_i。
- 阻止コンデンサの電極ごとの自己バイアス V_dc (尺度は基本波の振幅 |V1|)。

判定 (量ごと、周期の終わりごと):

- 雑音: 周期ごとの雑音の分散 σ² を三階差分で見積もる: σ² ≈ mean‖x_j − 3x_{j−1} + 3x_{j−2} − x_{j−3}‖² / 20 (白色
  雑音なら不偏。2 次までの変化は消え、時定数 τ 周期でゆっくり減る変化は (1/τ)² 倍まで小さくなるので、流体では
  ほぼ 0。二階差分だと減る変化の曲がりが 1/τ 倍で残り、下の q の誤差と区別できない)。
- 窓 W: 窓の平均どうしの比較の雑音の幅 √(2σ²/W) が閾値の半分以下になる最小の W。``max_window`` で足りなければ
  「揺れが大きく比べられない」(状態 noisy。走り始めの速い変化も三階差分に残るので、流体でも初めの数〜数十周期は
  これになることがある。続くなら閾値を上げるか周期を長くする)。量ごとの W の最大を使う。
- 変化 D: 直近 W 周期の平均 B とその前の W 周期の平均 A の差から、雑音の分を引いた相対変化
  √max(0, ‖B − A‖² − 2σ²/W) / ‖B‖。
- 残りの変化 R: 窓 W' の平均を 3 つ (B'・A'・A'') 並べ、Δ1 = B' − A'、Δ2 = A' − A''、q = ⟨Δ1, Δ2⟩ / ‖Δ2‖² (窓ごとに
  変化が減る割合) とし、等比で減るとみなした残り R = D'·q⁺ / (1 − q⁺) (Aitken の外挿。D' は窓 W' の変化)。q⁺ は
  |q| に雑音による誤差の 2 倍 (SE(q) ≈ √(2σ²/W' / ‖Δ2‖²)·√(1 + q²)) を足した上限で、q⁺ ≥ 1 (減っていると言えない。
  雑音のある遅い傾きはこちら) なら無限大。外挿はこれまでの長さ (周期の数) の HORIZON (10) 倍の先までにとどめ、
  今の速さでそれだけ変わり続けたときの変化を上限にする (観測した長さよりはるか先の外挿は当てにならず、時定数が
  とても長く見える数値的なずれで無意味に大きくなる。時定数がこれまでの長さの 10 倍までの緩和は見逃さない)。
  W' は W から広げて、Δ1 が雑音の幅の 3 倍を越える (有意な) 最初の窓にする (雑音に埋もれた遅い傾きも長い窓で
  見つける)。標本の範囲で有意な変化が無い、または変化が丸め誤差 (1e-10) 以下なら 0。1 周期の変化が小さくても緩和の
  時定数が長い (阻止コンデンサの充電など) ときに、まだ定常でないことを見逃さない。
- 全ての量で D ≤ tol かつ R ≤ tol を ``hold`` 回続けて満たしたら収束 (最初の 1 回の周期・時刻・ステップを残す)。
  判定には 3W 周期と、雑音の見積もりの分 (MIN_SAMPLES 周期) が要る。PIC と雑音のある量 (周期ごとの雑音が閾値の
  1% を越える) は 3 × max_window 周期まで判定しない (短い標本で早すぎる合格をしない)。

PIC の量 (CV-d): 粒子の雑音と周期をまたぐ揺れ (総数などのゆっくりした揺れ) があるので、残りの変化 R は上の窓の
走査ではなく、長い履歴の回帰で見積もる (``_trend``)。

- 長い履歴: 周期平均を等しい長さのビンにまとめて持つ (``BinnedSeries``、最大 TREND_BINS 個。あふれたら隣どうしを
  合わせて長さを 2 倍)。直近の窓 (最大 3 × max_window + 2 周期) では雑音に埋もれる遅い傾きも、長い区間なら見える。
- 傾きの検定: これまでの周期の後ろ半分 (走り始めの速い変化を除く) のビンに直線を当て、傾きを 1 回だけ検定する
  (窓を何通りも試すと雑音を傾きと見誤りやすい)。傾きの分散は二次式を当てた残差 (バッチ平均。減っていく量の曲がりは
  数えない) から見積もり、残差の隣どうしの相関 ρ で (1 + ρ)/(1 − ρ) 倍する (三階差分は周期をまたぐ相関を見ない)。
  しきい値はスカラーなら 3σ 相当の t 分布の点、場 (ブロックのベクトル) はブロックの雑音をまとめた χ² の点 (自由度は
  残差から見た実効的なブロックの数。相関のあるブロックはまとめて数える)。
- 有意な傾きは、区間の前半と後半の傾きの比 q で減り方を見分ける。比も後半の傾きも上限 (3σ) を使い、減っていれば
  時定数 τ = 前半と後半の中心の間隔 / ln(1/q⁺) から R = 終わりの傾き × τ (指数で近づく量なら正確、"decay")。減って
  いると言えなければ、傾きが調べた後ろ半分の長さ (PIC_HORIZON × これまでの長さ) だけ続いたときの変化 ("span")。
  流体の 10 倍の外挿は PIC には使わない (総数などのゆっくりした揺れ (±0.5%、千周期ほど) を傾きとして外挿し、
  定常でも不合格にした。2026-10-07 決定)。
- 分解能 (``res``): 検定で見つけられる最小の傾きで、同じ長さ (後ろ半分) だけ走ったときの変化。閾値を越えるうちは
  合格にしない (状態 "resolving")。収束としたときに見逃しうる傾きは、調べた後ろ半分と同じ長さだけ走っても閾値の分
  ほどしか変わらない。雑音が閾値と同じくらいの量は、分解能が届くまで百〜数百周期かかる。
- 限界: 傾きを見る区間は後ろ半分なので、途中で急に止まった変化は、止まった周期の約 2 倍まで不合格のまま
  (安全側)。後ろ半分で閾値の分も変わらないほど遅い傾き (それより長く続くもの) は、収束としたあとで見えてから
  不合格になる。

結果の無限大・未計算の値は JSON に載せられないので None にする (状態 "fail" で R が None なら無限大)。R の種類
(``r_kind``) も残す: "none" (有意な変化が無い)・"decay" (減っているとして等比で外挿)・"trend" (流体: 減っていると
言えない、または減り方がとても遅い: 今の傾きでこれまでの長さの 10 倍だけ変わり続けたときの値なので大きく出る)・
"span" (PIC: 減っていると言えない傾きが、調べた後ろ半分の長さだけ続いたときの変化)。
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

#: 既定の閾値 (相対): 流体 (決定的) と PIC (粒子の雑音がある)
TOL_FLUID = 1.0e-3
TOL_PIC = 1.0e-2
#: 場を比べる粗いブロックの数の目安
BLOCKS_2D = 1024
BLOCKS_1D = 64
#: 判定を始めるのに要る周期の数 (雑音の見積もりの三階差分が 4 個)
MIN_SAMPLES = 7
#: 変化が有意とみなす雑音の幅の倍数 (窓を広げながら何度も試すので厳しめ)
Z_SIG = 3.0
#: 変化 (相対) がこれ以下なら丸め誤差とみなし、残りの変化を見積もらない
R_FLOOR = 1.0e-10
#: 減る割合 q の上限に足す誤差の倍数
Q_SIG = 2.0
#: 周期ごとの雑音 (相対) がこの割合 × tol 以下なら雑音の無い量とみなす
NOISE_FREE = 1.0e-2
#: 残りの変化の外挿は、これまでの長さ (周期の数) のこの倍の先まで
HORIZON = 10.0
#: 収束で止めるときの平均区間の既定 (avg_steps が無いとき) [判定の周期]
STOP_AVG_PERIODS = 10
#: PIC の場を 1 周期に足し込む回数の目安
PIC_SAMPLES_PER_PERIOD = 64
#: PIC の長い履歴のビンの数の上限 (あふれたら隣どうしを合わせて長さを 2 倍にする、CV-d)
TREND_BINS = 64
#: 傾きを見る区間: これまでの周期の後ろからこの割合 (走り始めの速い変化を除く)
TREND_SPAN = 0.5
#: 傾きの検定に要る、区間の中のビンの数の下限 (二次式の残差の自由度 5)
TREND_MIN_BINS = 8
#: 残差の隣どうしの相関 ρ の上限 (傾きの分散を (1 + ρ) / (1 − ρ) 倍する。強すぎる相関は見積もりが当てにならない)
RHO_MAX = 0.9
#: PIC の分解能と、減っていると言えない傾きを外挿する長さ: これまでの周期の数のこの倍 (= 傾きを見た後ろ半分。
#: 2026-10-07 決定。10 倍にすると、PIC の総数のゆっくりした揺れ (±0.5%、千周期ほど) を傾きとして外挿して不合格に
#: した。これまでと同じ長さで分解能を保証すると、揺れで分解能が閾値まで下がらなかった)
PIC_HORIZON = TREND_SPAN

FIELDS = ("phi", "n_e")
TOTALS = ("N_e", "N_i")
VDC_PREFIX = "V_dc:"


def _to_host(a):
    """CuPy の配列・スカラーならホストへ写す。"""
    return a.get() if hasattr(a, "get") else a


def _finite_or_none(v):
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


class BlockMap:
    """節点 → 粗いブロック (軸平行の矩形) の体積で重み付けした平均。

    coords: (n,) か (n, 2) の節点の座標、volumes: (n,) の節点の体積 (軸対称は 2πr を含む)。座標の範囲を
    n_blocks 個ほどの正方形に近いブロックに分ける。節点の無いブロックは重み 0 で、値も 0。
    """

    def __init__(self, coords, volumes, n_blocks: int):
        c = np.asarray(coords, dtype=np.float64)
        if c.ndim == 1:
            c = c[:, None]
        v = np.asarray(volumes, dtype=np.float64)
        if len(v) != len(c):
            raise ValueError("BlockMap: 座標と体積の数が違います")
        lo = c.min(axis=0)
        hi = c.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        if c.shape[1] == 1:
            counts = [max(1, min(int(n_blocks), len(c)))]
        else:
            nx = max(1, round(math.sqrt(n_blocks * span[0] / span[1])))
            ny = max(1, round(n_blocks / nx))
            counts = [nx, ny]
        idx = np.zeros(len(c), dtype=np.int64)
        stride = 1
        for d, nb in enumerate(counts):
            k = np.clip(np.floor((c[:, d] - lo[d]) / span[d] * nb).astype(np.int64), 0, nb - 1)
            idx += k * stride
            stride *= nb
        self.shape = tuple(counts)
        self.n = stride
        self.index = idx
        self.node_volume = v
        self.weights = np.bincount(idx, weights=v, minlength=self.n)
        self._inv = np.where(self.weights > 0.0, 1.0 / np.where(self.weights > 0.0, self.weights, 1.0), 0.0)

    def project(self, f) -> np.ndarray:
        """節点の値 f のブロックごとの体積平均。"""
        f = np.asarray(f, dtype=np.float64)
        return np.bincount(self.index, weights=self.node_volume * f, minlength=self.n) * self._inv


class PeriodSums:
    """周期の中の足し込み。配列は NumPy でも CuPy でもよい (最初の値を写して、あとはその場で足す)。"""

    def __init__(self) -> None:
        self._sum: dict = {}
        self._n: dict[str, int] = {}

    def add(self, name: str, value) -> None:
        if name not in self._sum:
            self._sum[name] = value.copy() if hasattr(value, "copy") else value
            self._n[name] = 1
            return
        s = self._sum[name]
        if getattr(s, "ndim", 0):
            s += value
        else:
            self._sum[name] = s + value
        self._n[name] += 1

    def means(self) -> dict:
        out: dict = {}
        for name, s in self._sum.items():
            h = _to_host(s)
            n = self._n[name]
            out[name] = np.asarray(h, dtype=np.float64) / n if np.ndim(h) else float(h) / n
        return out

    def clear(self) -> None:
        self._sum.clear()
        self._n.clear()

    def drop(self, names) -> None:
        """名前の足し込みだけ捨てる (格子を作り直して場の配列の大きさが変わったとき)。"""
        for name in names:
            self._sum.pop(name, None)
            self._n.pop(name, None)

    def __len__(self) -> int:
        return len(self._sum)


class BinnedSeries:
    """周期ごとの値を等しい長さのビンにまとめた長い履歴 (PIC の傾きの検定、CV-d)。

    ビンは最大 ``max_bins`` 個で、あふれたら隣どうしを合わせて長さを 2 倍にする (どのビンも同じ長さ、最後のビン
    だけ途中のことがある)。ビンごとに周期の数・周期の番号の和・値の和 (場はブロックの配列) を持つので、走らせた
    長さによらずメモリは一定。
    """

    def __init__(self, max_bins: int = TREND_BINS) -> None:
        self.max_bins = int(max_bins)
        self.size = 1
        self.count: list[int] = []
        self.ksum: list[float] = []
        self.xsum: list = []

    def add(self, k: int, x) -> None:
        if not self.count or self.count[-1] >= self.size:
            if len(self.count) >= self.max_bins:
                self._merge()
            self.count.append(0)
            self.ksum.append(0.0)
            self.xsum.append(np.zeros(np.shape(x)) if np.ndim(x) else 0.0)
        self.count[-1] += 1
        self.ksum[-1] += float(k)
        self.xsum[-1] = self.xsum[-1] + x

    def _merge(self) -> None:
        def pairs(a: list) -> list:
            return [a[i] + a[i + 1] for i in range(0, len(a) - 1, 2)] + ([a[-1]] if len(a) % 2 else [])

        self.count, self.ksum, self.xsum = pairs(self.count), pairs(self.ksum), pairs(self.xsum)
        self.size *= 2

    @property
    def n(self) -> int:
        """足した周期の数。"""
        return sum(self.count)

    def tail(self, frac: float) -> list[int]:
        """後ろから、周期の数が全体の frac 以上になるまでのビンの番号 (古い順)。"""
        need = frac * self.n
        idx: list[int] = []
        cum = 0
        for i in range(len(self.count) - 1, -1, -1):
            idx.append(i)
            cum += self.count[i]
            if cum >= need:
                break
        return idx[::-1]

    def arrays(self, idx: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ビンの周期の数 c、中心の周期の番号 t、平均 y (行がビン。スカラーの量は 1 列)。"""
        c = np.array([self.count[i] for i in idx], dtype=np.float64)
        t = np.array([self.ksum[i] / self.count[i] for i in idx])
        y = np.array([np.atleast_1d(self.xsum[i] / self.count[i]) for i in idx], dtype=np.float64)
        return c, t, y

    def last_period(self) -> float:
        """最後に足した周期の番号 (ビンの周期は続いているとみなす)。"""
        return self.ksum[-1] / self.count[-1] + (self.count[-1] - 1) / 2.0


def _t_point(dof: int, z: float = Z_SIG) -> float:
    """t 分布で、正規分布の z (3σ) と同じ片側の確率になる点 (Cornish–Fisher の展開。自由度 5 で誤差 0.3% 以下)。"""
    v = float(max(dof, 1))
    return (z + (z**3 + z) / (4 * v) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * v**2)
            + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * v**3)
            + (79 * z**9 + 776 * z**7 + 1482 * z**5 - 1920 * z**3 - 945 * z) / (92160 * v**4))


def _fit_slope(c: np.ndarray, t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """周期の数 c で重み付けして直線を当てた傾き (1 周期あたり、列ごと)・Σc(t − t̄)²・重心の周期の番号。"""
    tbar = float(np.dot(c, t) / c.sum())
    dt = t - tbar
    stt = float(np.dot(c, dt * dt))
    ybar = (c @ y) / c.sum()
    b = ((c * dt) @ (y - ybar)) / stt
    return b, stt, tbar


def _chi2_point(p: float, z: float) -> float:
    """χ²_p / p の上側の点 (Wilson–Hilferty の近似。z は正規分布の点、p は実効的な自由度)。"""
    p = max(float(p), 1.0)
    return (1.0 - 2.0 / (9.0 * p) + z * math.sqrt(2.0 / (9.0 * p))) ** 3


def _fit_noise(c: np.ndarray, t: np.ndarray, y: np.ndarray,
               w: np.ndarray | None) -> tuple[float, float, int, float]:
    """二次式を当てた残差から、1 周期あたりの雑音の分散 (場はブロックの体積で重み付けした和)・残差の隣どうしの相関・
    自由度・実効的なブロックの数。ビンの平均の分散は雑音の分散 / c なので、c を掛けて 1 周期あたりにそろえる
    (バッチ平均)。二次式なので、減っていく量の曲がりは雑音に数えない。実効的なブロックの数は残差のグラム行列 G
    から (tr G)² / tr G² (互いに独立なブロックの数。相関のあるブロックはまとめて数える。スカラーは 1)。"""
    u = (t - np.dot(c, t) / c.sum()) / max(1.0, float(np.ptp(t)))
    x = np.stack([np.ones_like(u), u, u * u], axis=1)
    beta = np.linalg.solve(x.T @ (c[:, None] * x), x.T @ (c[:, None] * y))
    r = (y - x @ beta) * np.sqrt(c)[:, None]
    rw = r * w if w is not None else r
    rss = float(np.sum(rw * r))
    dof = len(c) - 3
    if rss <= 0.0 or dof <= 0:
        return 0.0, 0.0, max(dof, 1), 1.0
    rho = float(np.sum(rw[1:] * r[:-1]) / rss)
    g = rw @ r.T
    p_eff = rss * rss / float(np.sum(g * g))
    return rss / dof, rho, dof, p_eff


def circuit_vdc(circuit, t0: float, t1: float) -> list[tuple[str, float, float]]:
    """阻止コンデンサの RF 周期ごとの集計のうち、終わりの時刻が (t0, t1] の周期の V_dc と |V1| の平均 (電極ごと)。"""
    if circuit is None or circuit.period is None:
        return []
    h = circuit.history
    eps = 1.0e-6 * circuit.period
    rows = [j for j, tj in enumerate(h["t"]) if t0 + eps < tj <= t1 + eps]
    if not rows:
        return []
    out = []
    for e, label in enumerate(circuit.labels):
        v_dc = float(np.mean([h["v_dc"][j][e] for j in rows]))
        v1 = float(np.mean([h["v1"][j][e] for j in rows]))
        out.append((label, v_dc, v1))
    return out


class ConvergenceMonitor:
    """収束の判定の状態 (モジュールの docstring)。

    settings: schema.ConvergenceSettings。kind: "fluid" か "pic" (閾値の既定・足し込みの間隔)。rf_period_s: RF の
    最低周波数の周期 (無ければ None で、判定の周期は steps ステップ)。t: 作った時刻 (周期の途中なら最初の周期は
    捨てる)。続きの実行でも同じものを使い続ける (標本と履歴を引き継ぐ)。
    """

    def __init__(self, settings, *, kind: str, rf_period_s: float | None, dt: float, frame_every: int, t: float = 0.0):
        if kind not in ("fluid", "pic"):
            raise ValueError(f"ConvergenceMonitor: kind は fluid か pic です ({kind})")
        self.kind = kind
        self.tol = float(settings.tol) if settings.tol is not None else (TOL_PIC if kind == "pic" else TOL_FLUID)
        self.hold = int(settings.hold)
        self.stop = bool(settings.stop)
        self.max_window = int(settings.max_window)
        self.dt = float(dt)
        if rf_period_s is not None and rf_period_s > 0.0:
            self.basis = "rf"
            self.rf_periods = int(settings.rf_periods)
            self.period_s = self.rf_periods * float(rf_period_s)
        else:
            self.basis = "steps"
            self.rf_periods = None
            steps = settings.steps if settings.steps is not None else 10 * int(frame_every)
            self.period_s = int(steps) * self.dt
        steps_per = self.period_s / self.dt
        self.stride = max(1, int(steps_per // PIC_SAMPLES_PER_PERIOD)) if kind == "pic" else 1
        self.sums = PeriodSums()
        self._eps = 1.0e-9 * self.period_s
        self._k = math.floor(t / self.period_s + 1.0e-9)
        # 周期の途中から始めたら、最初の周期は短いので捨てる
        self._partial = t - self._k * self.period_s > 0.5 * self.dt
        self._samples: deque = deque(maxlen=3 * self.max_window + 2)
        #: PIC の量ごとの長い履歴 (傾きの検定、CV-d)。流体は使わない
        self._bins: dict[str, BinnedSeries] = {}
        self.weights: np.ndarray | None = None
        self.blocks: tuple | None = None
        self._w_prev = 1
        self._passes = 0
        self.converged = False
        self.converged_at: dict | None = None
        self.now_passing = False
        self.stopped = False
        self.stop_now = False
        #: この実行で収束で早めた終わりのステップ (早めていなければ None。ジョブの進捗の総ステップ数に使う)
        self.run_end: int | None = None
        self._converged_at_run_start = False
        self.history: dict = {"t": [], "step": [], "period": [], "window": [], "status": [],
                              "d": {}, "r": {}, "noise": {}, "r_kind": {}, "res": {}}

    # ---- エンジンから -------------------------------------------------------------------------

    def due(self, step_count: int) -> bool:
        """このステップの値を足し込むか (PIC は stride ステップおき、流体は毎ステップ)。"""
        return step_count % self.stride == 0

    def boundary_passed(self, t: float) -> bool:
        """時刻 t (ステップの終わり) が今の周期の終わりの境に届いたか。"""
        return t >= (self._k + 1) * self.period_s - self._eps

    def begin_run(self) -> None:
        """実行 (run_batch) の始まりに呼ぶ。止める設定は、この実行の中で初めて収束したときだけ効く。"""
        self._converged_at_run_start = self.converged
        self.stop_now = False
        self.run_end = None

    def default_avg_steps(self) -> int:
        """収束で止めるときの平均区間の既定 (avg_steps が無いとき) [ステップ]。"""
        return max(1, round(STOP_AVG_PERIODS * self.period_s / self.dt))

    def stop_end(self, step_count: int, end: int, avg_steps: int | None, accum_start: int | None) -> int | None:
        """収束で止めるときの新しい終わりのステップ。今の終わりより早まらなければ None。

        呼ぶのは stop_now のとき。早まるなら呼び出し側が時間平均の区間の始まりを step_count + 1 に移す。時間平均の
        区間がもう始まっていたら (accum_start ≤ step_count) 移さずに今の終わりまで走る (区間に結び付いた積算を
        途中で捨てないため。残りは平均区間より短い)。
        """
        self.stop_now = False
        if accum_start is not None and accum_start <= step_count:
            return None
        avg = int(avg_steps) if avg_steps is not None else self.default_avg_steps()
        new_end = step_count + max(1, avg)
        if new_end >= end:
            return None
        self.stopped = True
        self.run_end = new_end
        return new_end

    def close_period(self, t: float, step: int, *, blocks: BlockMap | None = None, circuit=None,
                     means: dict | None = None) -> None:
        """周期を閉じる: 足し込んだ量の平均を標本にして判定する (エンジンは boundary_passed のあとに呼ぶ)。

        means: エンジンが sums.means() を変換したもの (GPU PIC が重みを密度にし、表示用の節点へ写す)。無ければ
        sums.means() をそのまま使う。
        """
        if means is None:
            means = self.sums.means()
        self.sums.clear()
        k_closed = self._k
        t0 = k_closed * self.period_s
        t1 = (k_closed + 1) * self.period_s
        self._k = max(self._k + 1, math.floor(t / self.period_s + 1.0e-9))
        if self._partial:
            self._partial = False
            return
        sample: dict = {}
        if blocks is not None:
            self.weights = blocks.weights
            self.blocks = blocks.shape
            for name in FIELDS:
                if name in means and np.ndim(means[name]):
                    sample[name] = blocks.project(means[name])
        for name in TOTALS:
            if name in means:
                sample[name] = float(means[name])
        for label, v_dc, v1 in circuit_vdc(circuit, t0, t1):
            sample[VDC_PREFIX + label] = (v_dc, v1)
        if not sample:
            return
        self._samples.append(sample)
        if self.kind == "pic":
            for name, v in sample.items():
                self._bins.setdefault(name, BinnedSeries()).add(k_closed, v[0] if name.startswith(VDC_PREFIX) else v)
        self._judge(t, step, k_closed)

    # ---- 判定 ---------------------------------------------------------------------------------

    def _series(self, name: str) -> list:
        return [s[name] for s in self._samples if name in s]

    def _inner(self, name: str):
        """量の内積 (場はブロックの体積で重み付け)。"""
        if name in FIELDS:
            w = self.weights

            def field_inner(a, b) -> float:
                if w is not None and len(w) == len(a):
                    return float(np.dot(w, a * b))
                return float(np.dot(a, b))

            return field_inner
        return lambda a, b: float(a * b)

    @staticmethod
    def _values(name: str, x: list) -> list:
        """V_dc は (V_dc, |V1|) の組なので V_dc だけにする。"""
        if name.startswith(VDC_PREFIX):
            return [v for v, _ in x]
        return x

    def _scale(self, name: str, x: list, w: int, inner) -> float:
        """相対変化の尺度: 場はノルム、総数は絶対値、V_dc は |V1| (どれも直近 w 周期の平均)。"""
        if name.startswith(VDC_PREFIX):
            return float(np.mean([v1 for _, v1 in x[-w:]]))
        vals = self._values(name, x)
        b = sum(vals[-w:]) / w
        return math.sqrt(max(inner(b, b), 0.0)) if name in FIELDS else abs(float(b))

    @staticmethod
    def _sigma2(vals: list, span: int, inner) -> float | None:
        """三階差分から周期ごとの雑音の分散を見積もる (直近 span 周期)。"""
        n = len(vals)
        m = min(n, span)
        if m < 4:
            return None
        acc = 0.0
        cnt = 0
        for j in range(n - m + 3, n):
            d = vals[j] - 3.0 * vals[j - 1] + 3.0 * vals[j - 2] - vals[j - 3]
            acc += inner(d, d)
            cnt += 1
        return acc / (20.0 * cnt)

    def _window_needed(self, sig2: float | None, scale: float) -> float:
        if sig2 is None or sig2 <= 0.0:
            return 1.0
        if scale <= 0.0:
            return math.inf
        return float(math.ceil(8.0 * sig2 / (self.tol**2 * scale**2)))

    def _judge(self, t: float, step: int, k_closed: int) -> None:
        names = list(self._samples[-1].keys())
        series = {n: self._series(n) for n in names}
        inners = {n: self._inner(n) for n in names}
        n_min = min(len(x) for x in series.values())

        # 窓: 前回の窓の 3 倍の範囲で雑音を見積もり、広がったら広い範囲で見積もり直す
        w_cur = self._w_prev
        need = 1.0
        sig2: dict[str, float | None] = {}
        noise_free = True
        for _ in range(2):
            span = max(MIN_SAMPLES, 3 * w_cur + 2)
            need = 1.0
            noise_free = True
            for n in names:
                vals = self._values(n, series[n])
                sig2[n] = self._sigma2(vals, span, inners[n])
                scale = self._scale(n, series[n], 1, inners[n])
                need = max(need, self._window_needed(sig2[n], scale))
                if (sig2[n] or 0.0) > (NOISE_FREE * self.tol * scale) ** 2:
                    noise_free = False
            if need <= w_cur or need > self.max_window:
                break
            w_cur = int(need)
        window = int(need) if math.isfinite(need) and need <= self.max_window else None
        if window is not None:
            self._w_prev = window

        d_out: dict[str, float | None] = {n: None for n in names}
        r_out: dict[str, float | None] = {n: None for n in names}
        noise_out: dict[str, float | None] = {n: None for n in names}
        kind_out: dict[str, str | None] = {n: None for n in names}
        res_out: dict[str, float | None] = {n: None for n in names}
        pic = self.kind == "pic"
        if n_min < MIN_SAMPLES:
            status = "warming"
        elif window is None:
            status = "noisy"
        elif n_min < 3 * window:
            status = "warming"
        elif (pic or not noise_free) and n_min < max(3 * self.max_window, 2 * TREND_MIN_BINS):
            status = "warming"       # PIC・雑音のある量: 短い標本で早すぎる合格をしないよう 3 × max_window 周期まで
        else:
            fail = resolving = False
            for n in names:
                d, r, eta, kind = self._change(n, series[n], window, sig2[n] or 0.0, inners[n], k_closed + 1,
                                               scan=not pic)
                res = None
                if pic:
                    r, kind, res = self._trend(n, series[n], sig2[n] or 0.0, inners[n], k_closed + 1)
                d_out[n], r_out[n], noise_out[n], kind_out[n], res_out[n] = d, r, eta, kind, res
                if not (d <= self.tol and r <= self.tol):
                    fail = True
                elif res is not None and res > self.tol:
                    resolving = True
            status = "fail" if fail else ("resolving" if resolving else "pass")

        self._passes = self._passes + 1 if status == "pass" else 0
        self.now_passing = status == "pass"
        if not self.converged and self._passes >= self.hold:
            self.converged = True
            self.converged_at = {"t": float(t), "step": int(step), "period": int(k_closed)}
            if self.stop and not self._converged_at_run_start:
                self.stop_now = True

        h = self.history
        n_prev = len(h["t"])
        h["t"].append(float(t))
        h["step"].append(int(step))
        h["period"].append(int(k_closed))
        h["window"].append(window)
        h["status"].append(status)
        for key, vals in (("d", d_out), ("r", r_out), ("noise", noise_out), ("r_kind", kind_out), ("res", res_out)):
            for n in set(h[key]) | set(vals):
                lst = h[key].setdefault(n, [None] * n_prev)
                lst.append(vals.get(n))

    def _change(self, name: str, x: list, w: int, sig2: float, inner,
                elapsed: int, *, scan: bool = True) -> tuple[float, float, float, str]:
        """窓 w の変化 D・残りの変化 R・雑音の幅 η (どれも相対) と R の種類。elapsed: 始めからの周期の数 (外挿の上限)。

        R の種類: "none" (有意な変化が無い・丸め誤差、R = 0)、"decay" (減っているとして等比で外挿)、"trend" (減って
        いると言えない、または減り方がとても遅いので、今の傾きでこれまでの長さの HORIZON 倍だけ変わり続けたとき)。
        scan=False なら R は見積もらない (0、PIC は ``_trend`` で見積もる)。
        """
        vals = self._values(name, x)
        n = len(vals)

        def diffs(wr: int):
            b = sum(vals[-wr:]) / wr
            a = sum(vals[-2 * wr:-wr]) / wr
            a2 = sum(vals[-3 * wr:-2 * wr]) / wr
            return b - a, a - a2

        d1, d2 = diffs(w)
        nv = 2.0 * sig2 / w
        n1 = inner(d1, d1)
        scale = self._scale(name, x, w, inner)
        if scale <= 0.0:
            if n1 <= 0.0:
                return 0.0, 0.0, 0.0, "none"
            return math.inf, math.inf, 0.0, "trend"
        d = math.sqrt(max(0.0, n1 - nv)) / scale
        eta = math.sqrt(max(nv, 0.0)) / scale
        r = 0.0
        kind = "none"
        if not scan:
            return d, r, eta, kind
        # 残りの変化: 変化が有意になる最初の窓 (W から広げる) で、窓ごとの変化の減り方から外挿する
        for wr in range(w, n // 3 + 1):
            if wr != w:
                d1, d2 = diffs(wr)
            nvr = 2.0 * sig2 / wr
            n1 = inner(d1, d1)
            if n1 <= Z_SIG**2 * nvr:
                continue
            dr = math.sqrt(n1 - nvr) / scale
            if dr <= R_FLOOR:
                break
            n2 = inner(d2, d2)
            if n2 <= nvr or n2 <= 0.0:
                r_geo = math.inf
            else:
                q = inner(d1, d2) / n2
                q_up = abs(q) + Q_SIG * math.sqrt(nvr / n2 * (1.0 + q * q))
                r_geo = dr * q_up / (1.0 - q_up) if q_up < 1.0 else math.inf
            # 外挿はこれまでの長さの HORIZON 倍の先まで: 今の速さ (1 周期あたり dr / wr) でそれだけ変わり続けたときの
            # 変化を上限にする (減っていると言えない傾き・時定数がとても長く見える数値的なずれを無限大にしない)
            cap = dr / wr * HORIZON * elapsed
            r, kind = (r_geo, "decay") if r_geo <= cap else (cap, "trend")
            break
        return d, r, eta, kind

    def _trend(self, name: str, x: list, sig2: float, inner, elapsed: int) -> tuple[float, str, float]:
        """PIC の量の残りの変化 R・その種類・分解能 (どれも相対、CV-d)。

        長い履歴の後ろ半分のビンに直線を当て、傾きを 1 回だけ検定する。傾きの分散は二次式を当てた残差から見積もり
        (バッチ平均)、残差の隣どうしの相関 ρ で (1 + ρ)/(1 − ρ) 倍する (周期をまたぐ揺れ)。三階差分の雑音より
        小さくはしない。しきい値はスカラーなら 3σ 相当の t 分布の点の 2 乗、場はブロックの雑音をまとめた χ² の点
        (実効的なブロックの数 p_eff の自由度。ブロックごとの雑音に埋もれる全体の傾きも見える)。有意でなければ
        R = 0 (none)。有意なら区間の前半と後半の傾きの比 q で減り方を見分け、減っていれば時定数 τ = 前半と後半の
        中心の間隔 / ln(1/q⁺) と区間の終わりの傾きから R = 傾き × τ (decay)。比も後半の傾きも上限 (3σ。場は傾きの
        向きの成分なので雑音は 1/p_eff) を使う。減っていると言えなければ、傾きが調べた後ろ半分の長さ (PIC_HORIZON ×
        これまでの長さ) だけ続いたときの変化 (span)。分解能は、検定で見つけられる最小の傾きで同じ長さだけ走ったとき
        の変化。
        """
        bs = self._bins[name]
        scale = self._scale(name, x, 1, inner)
        if scale <= 0.0:
            return 0.0, "none", 0.0
        w = self.weights if name in FIELDS and self.weights is not None else None
        idx = bs.tail(TREND_SPAN)
        if len(idx) < TREND_MIN_BINS:
            return 0.0, "none", math.inf       # 途中から現れた量: 見分けるにはまだ周期が足りない
        c, t, y = bs.arrays(idx)
        if w is not None and len(w) != y.shape[1]:
            w = None

        def ip(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.dot(w, a * b)) if w is not None else float(np.dot(a, b))

        s2, rho, dof, p_eff = _fit_noise(c, t, y, w)
        rho = min(max(rho, 0.0), RHO_MAX)
        var_unit = max(sig2, s2 * (1.0 + rho) / (1.0 - rho))
        b, stt, _ = _fit_slope(c, t, y)
        var_b = var_unit / stt
        z = _t_point(dof)
        if y.shape[1] > 1:
            thr = _chi2_point(p_eff, z)                 # ‖b‖² のしきい値 (var_b の倍数)
            b_min = math.sqrt(max(thr - 1.0, 0.0) * var_b)
            p_dir = p_eff
        else:
            thr = z * z
            b_min = z * math.sqrt(var_b)
            p_dir = 1.0
        span = PIC_HORIZON * elapsed
        res = b_min * span / scale
        nb2 = ip(b, b)
        b_ns = math.sqrt(max(0.0, nb2 - var_b)) / scale
        if nb2 <= thr * var_b or b_ns * HORIZON * elapsed <= R_FLOOR:
            return 0.0, "none", res
        cap = b_ns * HORIZON * elapsed
        # 減り方: 区間を周期の数で半分に分け、前半と後半の傾きの比 (指数で近づくなら e^(−間隔/τ))
        cum = np.cumsum(c)
        j = min(max(int(np.searchsorted(cum, cum[-1] / 2.0)) + 1, 2), len(c) - 2)
        b_old, s_old, t_old = _fit_slope(c[:j], t[:j], y[:j])
        b_new, s_new, t_new = _fit_slope(c[j:], t[j:], y[j:])
        v_old, v_new = var_unit / s_old, var_unit / s_new
        no2 = ip(b_old, b_old)
        r_geo = math.inf
        if no2 > v_old:
            # 雑音で後半の傾きがたまたま小さいと、まっすぐな傾きを「減っている」と見誤るので、比も後半の傾きも
            # 上限 (3σ、傾きの検定と同じ) で外挿する
            q = ip(b_new, b_old) / no2
            q_up = abs(q) + Z_SIG * math.sqrt((v_new + q * q * v_old) / no2 / p_dir)
            if q_up < 1.0:
                tau = (t_new - t_old) / math.log(1.0 / q_up)
                bn = (math.sqrt(max(0.0, ip(b_new, b_new) - v_new)) + Z_SIG * math.sqrt(v_new / p_dir)) / scale
                r_geo = bn * q_up ** ((bs.last_period() - t_new) / (t_new - t_old)) * tau
        if r_geo <= cap:
            return r_geo, "decay", res
        return b_ns * span, "span", res

    # ---- 結果 ---------------------------------------------------------------------------------

    def frame(self) -> dict:
        """フレームに載せる今の状態 (最後に判定した周期)。worst は max(D, R) がいちばん大きい量の値 (無限大は None)、
        res は分解能がいちばん粗い量の分解能 (PIC、CV-d)。"""
        h = self.history
        worst = worst_name = worst_kind = None
        res = res_name = None
        if h["status"] and h["status"][-1] in ("pass", "fail", "resolving"):
            best = -1.0
            for n in h["d"]:
                d = h["d"][n][-1]
                r = h["r"][n][-1]
                if d is None:
                    continue
                v = math.inf if r is None else max(d, r)
                if v > best:
                    best, worst_name, worst_kind = v, n, h["r_kind"][n][-1]
            if worst_name is not None:
                worst = _finite_or_none(best)
            coarse = -1.0
            for n, lst in h["res"].items():
                v = lst[-1]
                if v is not None and v > coarse:
                    coarse, res_name = v, n
            if res_name is not None:
                res = _finite_or_none(coarse)
        return {
            "status": h["status"][-1] if h["status"] else "warming",
            "window": h["window"][-1] if h["window"] else None,
            "worst": worst,
            "worst_name": worst_name,
            "worst_kind": worst_kind,
            "res": res,
            "res_name": res_name,
            "tol": self.tol,
            "checks": len(h["t"]),
            "converged": self.converged,
            "converged_t": None if self.converged_at is None else self.converged_at["t"],
            "now_passing": self.now_passing,
        }

    def result(self) -> dict:
        """結果の ``convergence`` (判定の設定・周期ごとの履歴・収束した時)。"""
        h = self.history
        at = self.converged_at or {}

        def clean(d: dict) -> dict:
            return {n: [_finite_or_none(v) for v in lst] for n, lst in d.items()}

        return {
            "tol": self.tol,
            "basis": self.basis,
            "rf_periods": self.rf_periods,
            "period_s": self.period_s,
            "hold": self.hold,
            "stop": self.stop,
            "max_window": self.max_window,
            "blocks": None if self.blocks is None else list(self.blocks),
            "t": list(h["t"]),
            "step": list(h["step"]),
            "period": list(h["period"]),
            "window": list(h["window"]),
            "status": list(h["status"]),
            "d": clean(h["d"]),
            "r": clean(h["r"]),
            "noise": clean(h["noise"]),
            "r_kind": {n: list(lst) for n, lst in h["r_kind"].items()},
            "res": clean(h["res"]),
            "converged": self.converged,
            "converged_t": at.get("t"),
            "converged_step": at.get("step"),
            "converged_period": at.get("period"),
            "now_passing": self.now_passing,
            "stopped": self.stopped,
        }


def make_monitor(settings, *, kind: str, rf_period_s: float | None, dt: float, frame_every: int,
                 t: float = 0.0) -> ConvergenceMonitor | None:
    """設定 (None・enabled=false なら判定しない) から ConvergenceMonitor を作る。"""
    if settings is None or not settings.enabled:
        return None
    return ConvergenceMonitor(settings, kind=kind, rf_period_s=rf_period_s, dt=dt, frame_every=frame_every, t=t)
