"""阻止コンデンサ (自己バイアス) の回路 (prompts/134)。

電極と電源の間に直列の阻止コンデンサ C_b を置き、電極の電位 V_e (直流分 = 自己バイアス) を放電に合わせて
自己無撞着に決める。Vahedi & DiPeso (J. Comput. Phys. 131, 149 (1997)) の「電位と回路の同時解」を、回路が
コンデンサだけの場合に使う:

- 電極の節点 N (電極の導体 + コンデンサの電極側の板) の電荷 Q_N は、伝導電流 (壁へのイオン・電子の流束、
  二次電子) でだけ変わる: dQ_N/dt = I_cond
- Q_N = Q_e + C_b (V_e − V_s)。Q_e は電極の表面の電荷、V_s は電源の電圧 (voltage + ΣRF + 波形)
- Q_e は V_e について線形: Q_e = Q_e0 + C V_e。C は電極の容量行列 (コンデンサを付けた電極どうし)、Q_e0 は
  空間電荷とほかの電極の電位で決まる
- よって (C + diag C_b) V_e = Q_N − Q_e0 + C_b V_s

エンジンの 1 サブステップ (Poisson は今までどおり 1 回):

1. ``potentials()`` の V_e (前のサブステップの値、最初は V_s + initial_bias_v) で φ* を解く
2. φ* の電極の電荷 Q_e* から ``solve()`` が新しい V_e を返す (Q_e0 = Q_e* − C V_e^old)
3. エンジンが φ = φ* + Σ_k (V_e,k − V_e,k^old) ψ_k に直す。ψ_k は「電極 k が 1 V・ほかの電極が 0 V・空間電荷
   0」の解で、最初に 1 回だけ解く。C も ψ から求める (C_jk = ψ_k の解での電極 j の電荷)
4. 輸送のあと ``conduct()`` で Q_N に伝導電流を積む

電極の電荷 Q_e はエンジンが離散の Gauss の法則と合う形で求める: 電極から領域への電束から、電極の節点
(Dirichlet 節点) に置かれた空間電荷を引く。引かないと、壁の節点の電荷が電極へ流れ込んだときに V_e が跳ぶ
(その電荷は前から壁の位置にあり電場は変わらないので、跳ぶのは誤り)。

最初の ``solve()`` で Q_N = Q_e* + C_b (V_e − V_s) と置くので、V_e(0) = V_s(0) + initial_bias_v から始まる。
C_b → ∞ では V_e → V_s + initial_bias_v (直結に一定のずれを足したもの) になる。定常の直流分は「1 周期の
正味の伝導電流 0」で決まり、C_b ≫ C なら C_b によらない (C_b が決めるのは落ち着くまでの時間と RF の分圧)。

PIC は ``solve_induced()`` を使う (Green の相反定理の形): 全ての電極を 0 V にしたとき空間電荷が電極 k に誘導する
電荷 Q_e0,k = −Σ_i ψ̃_k(i) q_i (q_i は節点の電荷、ψ̃_k は ψ_k の固定節点に「電極 k なら 1、ほかは 0」を入れた
もの) をエンジンが求めれば、V_e = (C + C_b)⁻¹ (Q_N − Q_e0 + C_b V_s) が Poisson を解く前に決まる。Poisson の
行列が対称なら上の Q_e* − C V_e^old と厳密に同じ量で、Poisson は新しい V_e で 1 回解くだけでよく、V_e は
Poisson の解の精度 (GPU の固定回数の反復) によらない。

単位は座標系の電荷の単位: 軸対称は C・F・A、平面 2D は奥行き 1 m あたり、1D は面積あたり。

RF 1 周期 (最低周波数) ごとに、電極の電位の平均 V_dc (自己バイアス)、基本波の振幅 |V1|、正味の伝導電流 I_dc を
集計する。サブステップの間は V_e と I_cond が一定とみなし、区間の積分を周期の境で分けて足す。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: 結果に載せる最後の 1 周期の波形の点の数の上限
WAVE_POINTS = 400


@dataclass(frozen=True)
class CapacitorSpec:
    """阻止コンデンサを付けた電極 1 つ (label は 2D なら導体の region id か "edge{k}"、1D なら "left"/"right")。"""

    label: str
    capacitance: float
    initial_bias_v: float = 0.0


class BlockingCircuit:
    """阻止コンデンサを付けた電極の回路の状態と、RF 1 周期ごとの集計 (モジュールの docstring)。

    c_matrix: 電極の容量行列 C (M×M、エンジンが ψ から求める)。period: 集計の周期 [s] (RF の最低周波数の
    周期。RF が無ければ None で集計しない)。
    """

    def __init__(self, specs: list[CapacitorSpec], c_matrix, period: float | None):
        self.specs = list(specs)
        m = len(self.specs)
        if m == 0:
            raise ValueError("阻止コンデンサの電極がありません")
        self.labels = [s.label for s in self.specs]
        self.c_b = np.array([float(s.capacitance) for s in self.specs])
        self.bias0 = np.array([float(s.initial_bias_v) for s in self.specs])
        self.c = np.asarray(c_matrix, dtype=np.float64).reshape(m, m)
        self._a = self.c + np.diag(self.c_b)
        self.period = float(period) if period is not None and period > 0.0 else None
        self._omega = 2.0 * math.pi / self.period if self.period else 0.0
        # 回路の状態 (最初の solve で決まる)
        self.q_node: np.ndarray | None = None   # 電極の節点の電荷 Q_N
        self.v: np.ndarray | None = None        # 電極の電位 V_e (直近の solve)
        self.v_src: np.ndarray | None = None    # その時刻の電源の電圧 V_s
        # 周期の集計: 今の周期の番号・積分 (∫V dt, ∫V cos ωτ dτ, ∫V sin ωτ dτ, ∫I dt)・区間の長さ・波形
        self._k: int | None = None
        self._acc = np.zeros((4, m))
        self._dur = 0.0
        self._wave_t: list[float] = []
        self._wave_v: list[np.ndarray] = []
        self.last_wave: dict | None = None
        self.history: dict[str, list] = {"t": [], "v_dc": [], "v1": [], "i_dc": []}

    @property
    def m(self) -> int:
        return len(self.specs)

    # ---- 1 サブステップ ---------------------------------------------------------------

    def potentials(self, v_src) -> np.ndarray:
        """Poisson (φ*) に使う電極の電位: 前のサブステップの V_e (最初は V_s + initial_bias_v)。"""
        if self.v is None:
            return np.asarray(v_src, dtype=np.float64) + self.bias0
        return self.v.copy()

    def solve(self, q_star, v_old, v_src) -> np.ndarray:
        """V_e^old (= potentials()) で解いた φ* の電極の電荷 q_star から新しい V_e を求める。"""
        q_star = np.asarray(q_star, dtype=np.float64)
        v_old = np.asarray(v_old, dtype=np.float64)
        v_src = np.asarray(v_src, dtype=np.float64)
        if self.q_node is None:
            self.q_node = q_star + self.c_b * (v_old - v_src)
        rhs = self.q_node - q_star + self.c @ v_old + self.c_b * v_src
        self.v = np.linalg.solve(self._a, rhs)
        self.v_src = v_src.copy()
        return self.v.copy()

    def solve_induced(self, q_induced, v_src) -> np.ndarray:
        """空間電荷が誘導する電極の電荷 Q_e0 (全ての電極が 0 V のときの電極の電荷) から V_e を求める (PIC 用)。

        最初の呼び出しで V_e = V_s + initial_bias_v になるよう Q_N = Q_e0 + C V_e + C_b initial_bias_v と置く。
        """
        q0 = np.asarray(q_induced, dtype=np.float64)
        v_src = np.asarray(v_src, dtype=np.float64)
        if self.q_node is None:
            self.q_node = self.initial_charge(q0, v_src)
        self.v = np.linalg.solve(self._a, self.q_node - q0 + self.c_b * v_src)
        self.v_src = v_src.copy()
        return self.v.copy()

    def initial_charge(self, q_induced, v_src) -> np.ndarray:
        """V_e = V_s + initial_bias_v となる電極の節点の電荷 Q_N (solve_induced の最初・GPU 版の初期値)。"""
        v0 = np.asarray(v_src, dtype=np.float64) + self.bias0
        return np.asarray(q_induced, dtype=np.float64) + self.c @ v0 + self.c_b * self.bias0

    def conduct(self, i_cond, t0: float, dt: float) -> None:
        """サブステップ [t0, t0+dt] の伝導電流 (電極へ流れ込む正の電荷の流量) を Q_N に積み、集計を進める。"""
        i = np.asarray(i_cond, dtype=np.float64)
        self.q_node = self.q_node + i * dt
        if self.period is not None and dt > 0.0:
            self._accumulate(self.v, i, t0, t0 + dt)

    def residual(self, q_e) -> np.ndarray:
        """電荷の恒等式 Q_N − (Q_e + C_b (V_e − V_s)) (検証用。q_e は解き直した φ の電極の電荷)。"""
        return self.q_node - (np.asarray(q_e, dtype=np.float64) + self.c_b * (self.v - self.v_src))

    # ---- 回路をデバイスで進めるエンジン (GPU PIC) ----------------------------------------

    def record(self, v, i, t0: float, dt: float) -> None:
        """周期の集計だけを進める (Q_N はデバイスが積む。conduct の集計の部分)。"""
        if self.period is not None and dt > 0.0:
            self._accumulate(np.asarray(v, dtype=np.float64), np.asarray(i, dtype=np.float64), t0, t0 + dt)

    def set_state(self, q_node, v, v_src) -> None:
        """デバイスの回路の状態 (Q_N・V_e・V_s) を写す (結果・フレーム用)。"""
        self.q_node = np.array(q_node, dtype=np.float64)
        self.v = np.array(v, dtype=np.float64)
        self.v_src = np.array(v_src, dtype=np.float64)

    def set_capacitance(self, c_matrix) -> None:
        """電極の容量行列を差し替える (AMR の再格子化。Q_N はそのまま)。"""
        self.c = np.asarray(c_matrix, dtype=np.float64).reshape(self.m, self.m)
        self._a = self.c + np.diag(self.c_b)

    # ---- 周期の集計 ----------------------------------------------------------------------

    def _accumulate(self, v: np.ndarray, i: np.ndarray, t0: float, t1: float) -> None:
        period = self.period
        om = self._omega
        eps = 1e-9 * period
        t = t0
        while t1 - t > eps:
            k = math.floor(t / period + 1e-9)
            if self._k is None:
                self._k = k
            elif k != self._k:
                # 前の周期が閉じないまま次へ進んだ (区間が飛んだ): 前の周期をそこまでの分で閉じる
                self._close()
                self._k = k
            tb = (k + 1) * period
            te = min(t1, tb)
            tau0 = t - k * period
            tau1 = te - k * period
            w = te - t
            self._acc[0] += v * w
            self._acc[1] += v * ((math.sin(om * tau1) - math.sin(om * tau0)) / om)
            self._acc[2] += v * ((math.cos(om * tau0) - math.cos(om * tau1)) / om)
            self._acc[3] += i * w
            self._dur += w
            self._wave_t.append(tau0)
            self._wave_v.append(np.array(v, dtype=np.float64))
            if te >= tb - eps:
                self._close()
                self._k = k + 1
            t = te

    def _close(self) -> None:
        """今の周期の集計を履歴へ移す (区間が無ければ何もしない)。"""
        if self._dur <= 0.0 or self._k is None:
            return
        period = self.period
        dur = self._dur
        a1 = 2.0 * self._acc[1] / period
        b1 = 2.0 * self._acc[2] / period
        self.history["t"].append((self._k + 1) * period)
        self.history["v_dc"].append((self._acc[0] / dur).tolist())
        self.history["v1"].append(np.hypot(a1, b1).tolist())
        self.history["i_dc"].append((self._acc[3] / dur).tolist())
        n = len(self._wave_t)
        idx = np.unique(np.linspace(0, n - 1, min(n, WAVE_POINTS)).round().astype(np.int64)) if n else []
        wave = np.asarray(self._wave_v)
        self.last_wave = {
            "t": [self._wave_t[j] for j in idx],
            "v": [[float(wave[j, e]) for j in idx] for e in range(self.m)],
        }
        self._acc[:] = 0.0
        self._dur = 0.0
        self._wave_t = []
        self._wave_v = []

    # ---- 続き・途中停止 ----------------------------------------------------------------

    def reset_history(self) -> None:
        """続き実行の前に履歴だけ空にする (回路の状態・今の周期の途中の積分は引き継ぐ)。"""
        self.history = {k: [] for k in self.history}

    def snapshot(self):
        """途中停止でステップ開始時へ戻すための退避 (restore と対)。"""
        return (
            None if self.q_node is None else self.q_node.copy(),
            None if self.v is None else self.v.copy(),
            None if self.v_src is None else self.v_src.copy(),
            self._k, self._acc.copy(), self._dur, list(self._wave_t), list(self._wave_v), self.last_wave,
            {k: len(v) for k, v in self.history.items()},
        )

    def restore(self, saved) -> None:
        q, v, vs, k, acc, dur, wt, wv, last, lens = saved
        self.q_node, self.v, self.v_src = q, v, vs
        self._k, self._acc, self._dur = k, acc, dur
        self._wave_t, self._wave_v, self.last_wave = wt, wv, last
        for key, n in lens.items():
            del self.history[key][n:]

    # ---- 結果 --------------------------------------------------------------------------

    def frame(self) -> list[dict]:
        """フレームに載せる今の値 (電極ごとの V_e と、最後に閉じた周期の V_dc・|V1|)。"""
        out = []
        h = self.history
        for j, s in enumerate(self.specs):
            d: dict = {"label": s.label, "v_e": None if self.v is None else float(self.v[j])}
            if h["t"]:
                d["v_dc"] = h["v_dc"][-1][j]
                d["v1"] = h["v1"][-1][j]
            out.append(d)
        return out

    def result(self, units: dict[str, str]) -> dict:
        """done.result の ``circuit`` (電極ごとの容量・RF 1 周期ごとの集計・最後の 1 周期の波形)。"""
        h = self.history
        electrodes = []
        for j, s in enumerate(self.specs):
            c_self = float(self.c[j, j])
            electrodes.append({
                "label": s.label,
                "capacitance": float(s.capacitance),
                "initial_bias_v": float(s.initial_bias_v),
                "c_self": c_self,
                # 真空の RF の分圧 C_b / (C_b + C) (プラズマがあるとシースの容量の分だけ下がる)
                "rf_division": float(s.capacitance) / (float(s.capacitance) + c_self),
                "v_e": None if self.v is None else float(self.v[j]),
                "q_node": None if self.q_node is None else float(self.q_node[j]),
                "t": list(h["t"]),
                "v_dc": [row[j] for row in h["v_dc"]],
                "v1": [row[j] for row in h["v1"]],
                "i_dc": [row[j] for row in h["i_dc"]],
                "last_period": None if self.last_wave is None else {
                    "t": list(self.last_wave["t"]), "v": list(self.last_wave["v"][j]),
                },
            })
        return {
            "period_s": self.period,
            "units": dict(units),
            "c_matrix": self.c.tolist(),
            "electrodes": electrodes,
        }


# ---- 電極 (直交格子の Dirichlet グループ・三角形メッシュの節点) --------------------------------------


def rf_period(project) -> float | None:
    """集計の周期: 2D の電極の RF 成分と CSV 波形の最低周波数の周期 (流体 2D・v1 PIC の _find_rf_freq と同じ)。"""
    from .schema import rf_components

    freqs: list[float] = []
    geo = project.geometry
    sources = [bc for bc in geo.boundaries] + [r for r in geo.regions if r.type == "conductor"]
    for src in sources:
        freqs.extend(c.freq_hz for c in rf_components(src.voltage_rf))
        if src.voltage_waveform is not None:
            freqs.append(src.voltage_waveform.freq_hz)
    return 1.0 / min(freqs) if freqs else None


def model_capacitor_electrodes(project, model) -> list[tuple[CapacitorSpec, list[int]]]:
    """直交格子 (v2 の流体 2D・PIC) の阻止コンデンサの電極と、その Dirichlet グループの番号 (geom.model)。

    外周の境界条件は辺ごとに別のグループになるので、1 つの境界条件の辺はまとめて 1 つの電極にする (辺どうしは
    つながった 1 つの導体とみなす)。導体の領域は 1 つのグループ。
    """
    out: list[tuple[CapacitorSpec, list[int]]] = []
    for bc in project.geometry.boundaries:
        cap = bc.blocking_capacitor
        if bc.type != "dirichlet" or cap is None:
            continue
        groups: list[int] = []
        for e in bc.edges:
            g = model.side_group.get(model.domain.edge_sides.get(e))
            if g is not None and g not in groups:
                groups.append(g)
        if groups:
            label = "+".join(f"edge{e}" for e in bc.edges)
            out.append((CapacitorSpec(label, cap.capacitance, cap.initial_bias_v), groups))
    for k, c in enumerate(model.conductors):
        cap = c.region.blocking_capacitor
        if cap is not None:
            out.append((CapacitorSpec(c.id, cap.capacitance, cap.initial_bias_v), [model.conductor_group[k]]))
    used: set[int] = set()
    for spec, groups in out:
        if used & set(groups):
            raise ValueError(f"阻止コンデンサの電極 {spec.label} の辺が、ほかの阻止コンデンサの電極と重なっています")
        used |= set(groups)
    return out


def mesh_capacitor_electrodes(project, mesh) -> list[tuple[CapacitorSpec, np.ndarray]]:
    """三角形メッシュ (v1 の流体 2D・PIC) の阻止コンデンサの電極と、その Dirichlet 節点 (mesh.electrode のラベル)。

    1 つの境界条件の辺はまとめて 1 つの電極 (ラベル "edge1+edge3")。外周の円弧を弦に分けたときは元の辺の番号の
    ラベル (Geometry.edge_label。mesh.electrode と同じ) で引く (同じ円弧の弦は 1 つ)。
    """
    geo = project.geometry
    elecs: list[tuple[CapacitorSpec, set[str]]] = []
    for bc in geo.boundaries:
        cap = bc.blocking_capacitor
        if bc.type == "dirichlet" and cap is not None:
            labels = list(dict.fromkeys(geo.edge_label(e) for e in bc.edges))
            elecs.append((CapacitorSpec("+".join(labels), cap.capacitance, cap.initial_bias_v), set(labels)))
    for r in geo.regions:
        cap = r.blocking_capacitor
        if r.type == "conductor" and cap is not None:
            elecs.append((CapacitorSpec(r.id, cap.capacitance, cap.initial_bias_v), {r.id}))
    out = []
    for spec, labels in elecs:
        nodes = np.array(sorted(n for n, label in mesh.electrode.items() if label in labels), dtype=np.int64)
        if nodes.size == 0:
            raise ValueError(f"阻止コンデンサの電極 {spec.label} に Dirichlet 節点がありません")
        out.append((spec, nodes))
    return out
