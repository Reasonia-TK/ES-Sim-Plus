"""モンテカルロ衝突 (MCC, null-collision 法)。prompts/19 参照。

- 前処理: 全プロセステーブルの最大エネルギー (安全率 2 倍) までの共通グリッドで
  ν_tot(E) = Σ_j n_g σ_j(E) v(E) を評価し、ν_max = max_E ν_tot を求める
  (n_g = p/(kB·T_gas))
- 毎ステップ種ごとに: P_coll = 1 − exp(−ν_max·dt) で衝突候補を抽選し、
  候補粒子のエネルギーでの ν_j(E)/ν_max により実プロセス or null を選択する
- 断面積の評価はテーブルの np.interp (範囲外は端点値でクランプ)。
  excitation/ionization はテーブルが閾値から σ=0 で始まるので閾値未満は自然に 0
- 衝突は位置を変えないので所属要素の更新は不要
- 乱数はすべて mcc.seed からのシード付き rng (再現性)
- Numba 利用時は候補抽出・速度/エネルギー評価・断面積補間・プロセス選択・
  散乱方向構築・速度更新を JIT 化。乱数生成は numpy Generator に残して
  フォールバック経路と乱数列・計算結果を完全一致させる
- Numba 無しでは従来の numpy ベクトル化へフォールバック
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import _numba_kernels
from .particles import ME, QE
from .schema import MccSettings, XsProcess

KB = 1.380649e-23  # ボルツマン定数 [J/K]

# ν_max 評価用エネルギーグリッドの点数
_NU_GRID_N = 4096
# グリッド上限のテーブル最大エネルギーに対する安全率 (範囲外クランプ σ でも
# v(E) が伸びる分を吸収する)
_NU_GRID_MARGIN = 2.0


@dataclass
class GasField:
    """非一様背景ガス場 (prompts/54)。要素ごとの値 (DSMC の定常解など)。

    null-collision の ν_max は最大密度 n_max で評価し、粒子ごとの採択時に
    局所密度比 n_g(x)/n_max を掛ける (非一様密度でも null-collision は厳密)。
    t_g / u_g が None の場合は一様温度 (gas.temperature_k)・静止ガスとして扱う。
    """

    n_g: np.ndarray            # (M,) 要素ごとの数密度 [m^-3]
    t_g: np.ndarray | None = None  # (M,) 温度 [K]
    u_g: np.ndarray | None = None  # (M, 2) 面内流速 [m/s]


@dataclass
class _Proc:
    """前処理済みプロセステーブル (numpy 配列)。"""

    kind: str
    threshold_ev: float
    mass_ratio: float
    e: np.ndarray  # (K,) エネルギー [eV] (昇順)
    s: np.ndarray  # (K,) 断面積 [m^2]


@dataclass
class ElectronCollisionResult:
    """collide_electrons の結果。速度は in-place 更新し、生成粒子のみ返す。"""

    n_coll: int = 0              # 実衝突数 (elastic + excitation + ionization)
    n_ionization: int = 0        # 電離数
    new_x: np.ndarray | None = None     # 電離位置 (新電子・新イオン共通)
    new_elem: np.ndarray | None = None  # 所属要素 (入射電子と同じ)
    new_w: np.ndarray | None = None     # マクロ重み (入射電子と同じ)
    new_v_e: np.ndarray | None = None   # 放出電子の速度
    new_v_i: np.ndarray | None = None   # 新イオンの速度 (ガス温度 Maxwell)


class MccModel:
    """null-collision MCC。PicSimulation から毎ステップ呼ばれる。"""

    def __init__(
        self, settings: MccSettings, m_ion: float, gas_field: GasField | None = None
    ):
        gas = settings.gas
        self.n_gas = gas.pressure_pa / (KB * gas.temperature_k)  # 中性ガス数密度 [m^-3]
        self.m_ion = m_ion
        # ガス原子の Maxwell 速度分布の成分ごとの標準偏差 (ガス原子質量 = イオン質量とみなす)
        self.vth_gas = math.sqrt(KB * gas.temperature_k / m_ion)
        self.rng = np.random.default_rng(settings.seed)
        self.ionization_split = settings.ionization_split
        self.ion_energy_frame = settings.ion_energy_frame
        # 換算質量 μ = m_i·m_g/(m_i+m_g) (ガス原子質量 = イオン質量なので μ = m_i/2)
        self.mu = m_ion * m_ion / (m_ion + m_ion)

        # ---- 非一様背景ガス場 (prompts/54) -----------------------------------
        # field あり: ν_max は最大密度で評価し、採択時に局所密度比 _rel[elem] を掛ける。
        # 一定密度場 (rel ≡ 1.0) は一様指定と数値的に完全一致する
        self.field = gas_field
        if gas_field is not None:
            n_arr = np.asarray(gas_field.n_g, dtype=np.float64)
            if np.any(n_arr < 0.0) or not np.all(np.isfinite(n_arr)):
                raise ValueError("gas_field.n_g に負値または非有限値があります")
            n_max = float(n_arr.max()) if len(n_arr) else 0.0
            self._n_ref = n_max if n_max > 0.0 else self.n_gas
            self._rel = n_arr / self._n_ref if self._n_ref > 0.0 else n_arr
            self._vth_elem = (
                np.sqrt(KB * np.asarray(gas_field.t_g, dtype=np.float64) / m_ion)
                if gas_field.t_g is not None
                else None
            )
            self._u_elem = (
                np.asarray(gas_field.u_g, dtype=np.float64)
                if gas_field.u_g is not None
                else None
            )
        else:
            self._n_ref = self.n_gas
            self._rel = None
            self._vth_elem = None
            self._u_elem = None

        self.e_procs = [
            self._conv(p, ("elastic", "excitation", "ionization"))
            for p in settings.electron_processes
        ]
        self.i_procs = [
            self._conv(p, ("isotropic", "backscat")) for p in settings.ion_processes
        ]
        # Numbaのプロセス選択カーネルは可変長断面積表を直接扱えないため、初期化時に
        # 右側ゼロ詰めの2D配列へまとめる。lengthsまで渡すので補間範囲は元表と同じ。
        self._e_xs_pack = self._pack_proc_tables(self.e_procs)
        self._i_xs_pack = self._pack_proc_tables(self.i_procs)
        self._nu_e_table = self._nu_table(self.e_procs, ME)
        # com 系ではテーブルの E は重心系エネルギーなので、相対速度 g = √(2E/μ) で ν を評価
        m_ref = self.mu if self.ion_energy_frame == "com" else m_ion
        self._m_i_ref = m_ref
        self._nu_i_table = self._nu_table(self.i_procs, m_ref)
        # グリッド全域の ν_max (上限値。実行時は粒子の最大エネルギーまでに適応的に絞る)
        self.numax_e = self._numax_upto(self._nu_e_table, math.inf)
        self.numax_i = self._numax_upto(self._nu_i_table, math.inf)

    # ---- 前処理 --------------------------------------------------------------

    @staticmethod
    def _conv(p: XsProcess, allowed: tuple[str, ...]) -> _Proc:
        if p.kind not in allowed:
            raise ValueError(f"このプロセスリストでは kind='{p.kind}' は使えません ({p.label})")
        e = np.asarray(p.energy_ev, dtype=np.float64)
        s = np.asarray(p.sigma_m2, dtype=np.float64)
        if np.any(np.diff(e) < 0.0):
            raise ValueError(f"エネルギー列が昇順ではありません: {p.label}")
        return _Proc(p.kind, p.threshold_ev, p.mass_ratio, e, s)

    @staticmethod
    def _pack_proc_tables(
        procs: list[_Proc],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """可変長断面積表をNumbaへ渡せる連続2D配列にまとめる。"""
        if not procs:
            return None
        lengths = np.asarray([len(p.e) for p in procs], dtype=np.int64)
        width = int(lengths.max())
        e_table = np.zeros((len(procs), width), dtype=np.float64)
        s_table = np.zeros((len(procs), width), dtype=np.float64)
        for j, p in enumerate(procs):
            e_table[j, : len(p.e)] = p.e
            s_table[j, : len(p.s)] = p.s
        return e_table, s_table, lengths

    def _nu_table(self, procs: list[_Proc], m: float):
        """共通エネルギーグリッド上の ν_tot(E) と、その累積最大 (prefix max) を返す。

        累積最大 pref[i] = max(ν_tot(E ≤ grid[i])) を持つことで、
        「その時点の粒子最大エネルギーまでの ν_max」を O(log N) で引ける
        (適応 ν_max、prompts/57)。プロセスが無ければ None。
        """
        if not procs:
            return None
        e_cap = _NU_GRID_MARGIN * max(float(p.e[-1]) for p in procs)
        grid = np.linspace(0.0, e_cap, _NU_GRID_N)
        v = np.sqrt(2.0 * grid * QE / m)
        nu = np.zeros_like(grid)
        for p in procs:
            # 非一様ガス場では最大密度 _n_ref で評価する (一様時は _n_ref = n_gas)
            nu += self._n_ref * np.interp(grid, p.e, p.s) * v
        return grid, np.maximum.accumulate(nu)

    @staticmethod
    def _numax_upto(table, e_max_ev: float) -> float:
        """テーブルの E ≤ e_max_ev における ν_tot の最大値 (適応 ν_max)。

        断面積テーブルが数 keV まで伸びていても、実在する粒子の最大エネルギー
        までに絞ることで候補率の過大評価を避ける (null-collision は
        「候補時の全粒子エネルギーで ν_max ≥ ν(E)」なら厳密なまま)。
        """
        if table is None:
            return 0.0
        grid, pref = table
        if e_max_ev >= grid[-1]:
            return float(pref[-1])
        idx = int(np.searchsorted(grid, e_max_ev, side="right"))
        return float(pref[min(idx, len(pref) - 1)])

    # ---- 共通: 候補抽選とプロセス選択 -----------------------------------------

    def _candidates(self, n: int, numax: float, dt: float) -> np.ndarray:
        """null-collision の衝突候補インデックスを抽選する。"""
        p_coll = 1.0 - math.exp(-numax * dt)
        random_u = self.rng.random(n)
        if _numba_kernels.HAVE_NUMBA:
            return _numba_kernels.mcc_candidates(random_u, p_coll)
        return np.nonzero(random_u < p_coll)[0]

    def _choose_process(
        self,
        e_ev: np.ndarray,
        speed: np.ndarray,
        numax: float,
        procs: list[_Proc],
        rel: np.ndarray | None = None,
    ) -> np.ndarray:
        """候補ごとの参照エネルギー・速さから実プロセスを選択する (-1 = null 衝突)。

        rel: 候補ごとの局所密度比 n_g(x)/n_max (非一様ガス場、prompts/54)。
        None なら一様 (従来と完全一致)。
        """
        # 乱数は実装分岐より前に従来と同じ本数・順序で引く。Numba経路でも
        # Generator状態と後続の散乱乱数列がNumPy経路から変わらない。
        random_u = self.rng.random(len(e_ev))
        if _numba_kernels.HAVE_NUMBA:
            packed = self._e_xs_pack if procs is self.e_procs else self._i_xs_pack
            if packed is not None:
                return _numba_kernels.mcc_choose_process(
                    e_ev,
                    speed,
                    numax,
                    self._n_ref,
                    rel,
                    *packed,
                    random_u,
                )
        nu = np.empty((len(e_ev), len(procs)))
        for j, p in enumerate(procs):
            nu[:, j] = self._n_ref * np.interp(e_ev, p.e, p.s) * speed
        if rel is not None:
            nu *= rel[:, None]
        cum = np.cumsum(nu, axis=1)
        u = random_u * numax
        hit = u < cum[:, -1]
        return np.where(hit, np.argmax(u[:, None] < cum, axis=1), -1)

    def _select(
        self,
        v: np.ndarray,
        m: float,
        numax: float,
        procs: list[_Proc],
        dt: float,
        elem: np.ndarray | None = None,
    ):
        """衝突候補の抽選と実プロセスの選択 (実験室系エネルギー参照)。

        戻り値: (cand, proc_idx, e_ev, speed)。cand は候補粒子のインデックス、
        proc_idx は選ばれたプロセス番号 (-1 = null 衝突)。
        elem は非一様ガス場 (prompts/54) の局所密度参照用 (一様なら不使用)。
        """
        cand = self._candidates(len(v), numax, dt)
        if cand.size == 0:
            return cand, None, None, None
        if _numba_kernels.HAVE_NUMBA:
            packed = self._e_xs_pack if procs is self.e_procs else self._i_xs_pack
            if packed is not None:
                # 候補速度のgather、エネルギー算出、局所密度参照、断面積補間、
                # プロセス選択を1つのJITループへまとめる。選択乱数は従来と同じ
                # タイミング・本数でNumPy Generatorから生成する。
                random_u = self.rng.random(cand.size)
                proc_idx, e_ev, speed = _numba_kernels.mcc_select_velocity(
                    v,
                    cand,
                    m,
                    QE,
                    numax,
                    self._n_ref,
                    self._rel,
                    elem,
                    *packed,
                    random_u,
                )
                return cand, proc_idx, e_ev, speed
        vv = v[cand]
        speed = np.sqrt(np.sum(vv * vv, axis=1))
        e_ev = 0.5 * m * speed * speed / QE
        rel = self._rel[elem[cand]] if (self._rel is not None and elem is not None) else None
        proc_idx = self._choose_process(e_ev, speed, numax, procs, rel)
        return cand, proc_idx, e_ev, speed

    def _iso_dir(self, k: int) -> np.ndarray:
        """3D 等方な単位方向ベクトルを k 個サンプルする (cosχ 一様・方位角一様)。"""
        random_cos = self.rng.random(k)
        random_phi = self.rng.random(k)
        if _numba_kernels.HAVE_NUMBA:
            return _numba_kernels.mcc_iso_dir(random_cos, random_phi)
        cos_t = 1.0 - 2.0 * random_cos
        sin_t = np.sqrt(np.maximum(1.0 - cos_t * cos_t, 0.0))
        phi = random_phi * (2.0 * np.pi)
        return np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)

    # ---- 電子衝突 -------------------------------------------------------------

    def collide_electrons(
        self,
        x: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        elem: np.ndarray,
        dt: float,
    ) -> ElectronCollisionResult:
        """電子の MCC。v を in-place 更新し、電離の生成粒子を結果で返す。"""
        res = ElectronCollisionResult()
        if len(v) == 0 or self.numax_e <= 0.0:
            return res
        if self.field is not None and elem is None:
            raise ValueError("非一様ガス場には所属要素 (elem) が必要です")
        # 適応 ν_max (prompts/57): 現在の粒子最大エネルギーまでに絞る。
        # テーブルが keV 域まで伸びていても実在しないエネルギーで候補率を
        # 過大評価しない (ν_max ≥ 全粒子の ν(E) は保たれるので厳密なまま)
        if _numba_kernels.HAVE_NUMBA:
            max_speed2 = _numba_kernels.mcc_max_speed_squared(v)
        else:
            max_speed2 = float(np.max(np.sum(v * v, axis=1)))
        e_max = 0.5 * ME * max_speed2 / QE
        numax = self._numax_upto(self._nu_e_table, e_max)
        if numax <= 0.0:
            return res
        cand, proc_idx, e_ev, speed = self._select(
            v, ME, numax, self.e_procs, dt, elem
        )
        if cand.size == 0:
            return res

        new_x: list[np.ndarray] = []
        new_elem: list[np.ndarray] = []
        new_w: list[np.ndarray] = []
        new_ve: list[np.ndarray] = []
        new_vi: list[np.ndarray] = []

        for j, p in enumerate(self.e_procs):
            mask = proc_idx == j
            if p.kind in ("excitation", "ionization"):
                mask &= e_ev >= p.threshold_ev  # 閾値未満は null 扱い (通常 σ=0 で選ばれない)
            k = int(mask.sum())
            if k == 0:
                continue
            sub = cand[mask]
            d_new = self._iso_dir(k)  # 3D 等方散乱: cosχ 一様・方位角一様
            if p.kind == "elastic":
                # 散乱角 χ = 旧方向と新方向のなす角。ΔE = 2(m/M)(1−cosχ)E
                if _numba_kernels.HAVE_NUMBA:
                    _numba_kernels.mcc_scatter_electrons(
                        v,
                        sub,
                        e_ev[mask],
                        speed[mask],
                        d_new,
                        0,
                        p.threshold_ev,
                        p.mass_ratio,
                        None,
                        QE,
                        ME,
                    )
                else:
                    d_old = v[sub] / speed[mask][:, None]
                    cos_chi = np.sum(d_old * d_new, axis=1)
                    e_new = np.maximum(
                        e_ev[mask] * (1.0 - 2.0 * p.mass_ratio * (1.0 - cos_chi)),
                        0.0,
                    )
                    v[sub] = np.sqrt(2.0 * e_new * QE / ME)[:, None] * d_new
            elif p.kind == "excitation":
                # E − 閾値 に減速して等方散乱
                if _numba_kernels.HAVE_NUMBA:
                    _numba_kernels.mcc_scatter_electrons(
                        v,
                        sub,
                        e_ev[mask],
                        speed[mask],
                        d_new,
                        1,
                        p.threshold_ev,
                        p.mass_ratio,
                        None,
                        QE,
                        ME,
                    )
                else:
                    e_new = e_ev[mask] - p.threshold_ev
                    v[sub] = np.sqrt(2.0 * e_new * QE / ME)[:, None] * d_new
            else:  # ionization
                # 余剰 E − 閾値 を散乱電子/放出電子に分配 (両者 3D 等方)。
                # "half": 等分 (Turner ベンチマーク互換、既定) / "random": 一様乱数比
                excess = e_ev[mask] - p.threshold_ev
                if self.ionization_split == "half":
                    e_scat = 0.5 * excess
                else:
                    e_scat = self.rng.random(k) * excess
                e_eject = excess - e_scat
                if _numba_kernels.HAVE_NUMBA:
                    _numba_kernels.mcc_scatter_electrons(
                        v,
                        sub,
                        e_ev[mask],
                        speed[mask],
                        d_new,
                        2,
                        p.threshold_ev,
                        p.mass_ratio,
                        e_scat,
                        QE,
                        ME,
                    )
                else:
                    v[sub] = np.sqrt(2.0 * e_scat * QE / ME)[:, None] * d_new
                new_ve.append(np.sqrt(2.0 * e_eject * QE / ME)[:, None] * self._iso_dir(k))
                # 新イオンはガス温度の Maxwell 速度 (3成分)。非一様ガス場では
                # 局所温度・局所流速を使う (一様時は従来と完全一致)
                if self._vth_elem is not None:
                    vi_new = (
                        self.rng.standard_normal((k, 3))
                        * self._vth_elem[elem[sub]][:, None]
                    )
                else:
                    vi_new = self.rng.normal(0.0, self.vth_gas, size=(k, 3))
                if self._u_elem is not None:
                    vi_new[:, :2] += self._u_elem[elem[sub]]
                new_vi.append(vi_new)
                new_x.append(x[sub].copy())
                new_elem.append(elem[sub].copy())
                new_w.append(w[sub].copy())  # マクロ重みは入射電子と同じ
                res.n_ionization += k
            res.n_coll += k

        if new_x:
            res.new_x = np.concatenate(new_x)
            res.new_elem = np.concatenate(new_elem)
            res.new_w = np.concatenate(new_w)
            res.new_v_e = np.concatenate(new_ve)
            res.new_v_i = np.concatenate(new_vi)
        return res

    # ---- イオン衝突 -----------------------------------------------------------

    def collide_ions(self, v: np.ndarray, dt: float, elem: np.ndarray | None = None) -> int:
        """イオンの MCC。v を in-place 更新し、実衝突数を返す。

        断面積テーブルの energy_ev の解釈は ion_energy_frame で切り替える:
        - "lab": 実験室系イオンエネルギー ½ m_i v² (従来動作、ν は実験室速さで評価)
        - "com": 重心系エネルギー E = ½μg² (g = 相対速度、ν = n_g σ(E) g)。
          Turner の He+/He (Phelps) データはこちら
        elem は非一様ガス場 (prompts/54) の局所密度・温度・流速の参照用。
        """
        if len(v) == 0 or self.numax_i <= 0.0:
            return 0
        if self.field is not None and elem is None:
            raise ValueError("非一様ガス場には所属要素 (elem) が必要です")
        # 適応 ν_max (prompts/57): 参照エネルギーの上限を粒子の最大速さから見積もる。
        # com 系はガス原子速度が乗るため 6σ (+流速) のマージンを取る
        if _numba_kernels.HAVE_NUMBA:
            max_speed2 = _numba_kernels.mcc_max_speed_squared(v)
        else:
            max_speed2 = float(np.max(np.sum(v * v, axis=1)))
        v_max = math.sqrt(max_speed2)
        if self.ion_energy_frame == "com":
            vth_m = (
                float(self._vth_elem.max()) if self._vth_elem is not None else self.vth_gas
            )
            u_m = float(np.abs(self._u_elem).max()) if self._u_elem is not None else 0.0
            g_max = v_max + 6.0 * vth_m + u_m
            e_cap = 0.5 * self.mu * g_max * g_max / QE
        else:
            e_cap = 0.5 * self.m_ion * v_max * v_max / QE
        numax = self._numax_upto(self._nu_i_table, e_cap)
        if numax <= 0.0:
            return 0
        cand = self._candidates(len(v), numax, dt)
        if cand.size == 0:
            return 0
        vi = v[cand]
        # 候補ごとにガス原子の Maxwell 速度を先に抽選し、参照エネルギーと衝突の両方に使う。
        # 非一様ガス場では局所温度・局所流速を使う (一様時は従来と完全一致)
        if self._vth_elem is not None:
            vg = (
                self.rng.standard_normal((cand.size, 3))
                * self._vth_elem[elem[cand]][:, None]
            )
        else:
            vg = self.rng.normal(0.0, self.vth_gas, size=(cand.size, 3))
        if self._u_elem is not None:
            vg[:, :2] += self._u_elem[elem[cand]]
        g = vi - vg
        g_mag = np.sqrt(np.sum(g * g, axis=1))
        if self.ion_energy_frame == "com":
            e_ref = 0.5 * self.mu * g_mag * g_mag / QE
            s_ref = g_mag
        else:
            s_ref = np.sqrt(np.sum(vi * vi, axis=1))
            e_ref = 0.5 * self.m_ion * s_ref * s_ref / QE
        rel = self._rel[elem[cand]] if (self._rel is not None and elem is not None) else None
        proc_idx = self._choose_process(e_ref, s_ref, numax, self.i_procs, rel)
        n_coll = 0
        for j, p in enumerate(self.i_procs):
            mask = proc_idx == j
            k = int(mask.sum())
            if k == 0:
                continue
            sub = cand[mask]
            if p.kind == "backscat":
                # 電荷交換: イオン速度をガス原子の速度で置き換える
                if _numba_kernels.HAVE_NUMBA:
                    _numba_kernels.mcc_scatter_ions(
                        v,
                        sub,
                        vi[mask],
                        vg[mask],
                        g_mag[mask],
                        None,
                        True,
                    )
                else:
                    v[sub] = vg[mask]
            else:  # isotropic: 等質量弾性衝突、COM 系 3D 等方散乱 (|g| 保存)
                direction = self._iso_dir(k)
                if _numba_kernels.HAVE_NUMBA:
                    _numba_kernels.mcc_scatter_ions(
                        v,
                        sub,
                        vi[mask],
                        vg[mask],
                        g_mag[mask],
                        direction,
                        False,
                    )
                else:
                    v_com = 0.5 * (vi[mask] + vg[mask])
                    v[sub] = v_com + 0.5 * g_mag[mask][:, None] * direction
            n_coll += k
        return n_coll
