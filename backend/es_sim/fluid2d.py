"""2D/軸対称 ドリフト拡散流体ソルバー (EAFE/FEM-SG、prompts/104〜106, 111)。

fluid1d.py (1D ドリフト拡散流体) の 2D/軸対称拡張。geometry/mesh (非構造 or
構造格子の三角形メッシュ)・境界条件 (Dirichlet 電圧・voltage_rf・CSV 波形・
symmetry・see_gamma・conductor/dielectric 領域) は既存のプロジェクト設定を
そのまま使い、2D PIC (pic.py) と同一条件で直接比較できることを設計目標にする。

## 離散化: EAFE (Edge-Averaged Finite Element = FEM-SG)

半導体デバイスシミュレーションの標準手法 (Xu & Zikatanov 2000 ほか)。P1-FEM の
剛性行列のエッジ重みに Scharfetter–Gummel (SG) を載せた node-centered 有限体積
スキームで、1D の SG (fluid1d.py) を非構造三角形メッシュ上のグラフへ一般化した
ものと見なせる。

- 「一様拡散係数 (=1) の P1 剛性行列」K を fem.assemble_transport_operator() で
  組む (要素幾何・rz の r̄ 重み近似・線形 r の厳密積分は fem.assemble() と全く
  同じルーチンを流用するので、軸対称の重みが自動的に正しくなる)。K のオフ対角
  K_ij を反転した w_ij = −K_ij がエッジ (i, j) の実効重みになる。
  Delaunay 的でないメッシュでは w_ij < 0 になり得る (非単調性の源) — 出現数を
  数えて起動時に warning を出すが、フラックス計算自体はそのまま続行する
  (prompts/111 の指示どおり。負重みを無理にクリップすると保存性が崩れるため)。
- 種 s のエッジフラックス (i→j、電位差 Δφ_ij = φ_i − φ_j):
      z = ±Δφ_ij / T_s  (Einstein 関係 D_s=μ_s T_s のもとで μ_s が厳密に約分される。
                          fluid1d.py の導出と全く同じ、符号は種の電荷で決まる)
      F_ij = w_ij · D_s · [B(−z)·n_i − B(z)·n_j]   (B: Bernoulli 関数、fluid1d._bernoulli を再利用)
  ∂n_i/∂t · V_i = −Σ_j F_ij + S_i·V_i − (壁損失)_i   (V_i: 節点の集中体積)
  F_ij は i↔j に関して反対称 (F_ji = −F_ij) なので、Σ_i (内部エッジ由来の項) は
  厳密に 0 — これが「保存性は EAFE + 集中質量で構成的に成り立つ」という
  prompts/111 の要求の根拠であり、test_fluid2d.py のテスト3が直接検証する。
- 節点の集中体積 V_i は同じ fem.assemble_transport_operator() が返す (pic.py の
  _node_area と同じ「隣接要素の面積/rz体積を節点へ P1 配分」の流儀)。
- 電子エネルギー w = (3/2) n_e T_e も同じ EAFE (係数 5/3、Hagelaar & Kroesen 形。
  fluid1d.py と同じ z を流用できる導出も同一)。
- E ベクトル (要素定数の勾配) は pic.py の流儀 (particles._barycentric_coeffs)
  をそのまま使う。時間平均の |E| (e_abs) は「E ベクトルを平均してから絶対値」
  という pic.py の averaged_fields と同じ規約。

## ドメインと境界条件

- 輸送領域 = 全メッシュから固体要素 (dielectric、pic.py の particles._solid_elements
  と全く同じ定義) を除いた領域。固体要素にしか属さない節点は「非輸送」節点として
  除外し (node_vol=0)、常に 0 のまま出力する (フロント側は mesh 情報から固体
  領域を判定してマスクできる想定)。
- 壁 (吸収) 境界 = Dirichlet 電極エッジ・固体表面・domain 外周の非対称エッジ
  (symmetry でも rz の対称軸でもない外周エッジ)。フラックス BC は fluid1d.py と
  同じ Hagelaar & Kroesen 系だが、2D では「壁エッジ」ごとに P1 境界質量重み
  (xy: L/2、rz: 2π·(L/6)(2r_i+r_j) — assemble() の rhs 公式の1D版) を計算し、
  それを節点へ配分した「壁コンダクタンス」を implicit 行列の対角に加える形で
  一般化する (1D の c_left/c_right をエッジ単位に分解して足し合わせたものに相当)。
- 対称境界 (symmetry BC エッジ・rz の対称軸 r=0) は自然境界 (フラックス0、
  何もしない)。periodic は未対応 (下記 __init__ で ValueError — 理由はそこに
  コメント)。
- Joule 加熱: 2D 非構造メッシュには 1D のような「隣接する2つの面」という自明な
  概念が無いため、エッジ単位の電力 P_ij = F_ij・Δφ_ij (電流×電圧降下、抵抗網の
  発熱と同じ発想) を計算し、両端の節点へ半分ずつ配分する (半導体デバイス
  シミュレーションで自己無撞着な Joule 加熱を計算する標準的な手法)。
  壁エッジには対応する「電位差を持つ相手ノード」が無い (吸収境界条件であり
  抵抗網の枝ではない) ため、壁での Joule 加熱項は含めない (内部エッジのみで
  計算する簡略化。CCP 典型条件ではシース近傍の電子加熱の寄与は相対的に小さく、
  むしろ支配的なのはイオン加速によるスパッタ/衝撃エネルギーだが、本モデルには
  イオンエネルギー方程式が無いためこの点はそもそも扱わない — prompts/111 の
  逸脱として明記)。

## Poisson (電場) の解法

pic.py の PicSimulation が毎ステップやっている「fixed/free 分解 + K_ff の splu
事前分解 + 電圧波形 (v_dc + Σ RF + CSV) の Dirichlet 値評価」と全く同じ構成を
fluid2d.py 内で独立に組み立てる (fem.assemble()・pic._eval_waveform を再利用)。
pic.py 側の関数・メソッドを抽出して共通化する代わりにこの方式を選んだ理由:
pic.py の該当コードは周期境界のスレーブ除外・粒子電荷堆積・誘電体表面電荷
(q_surf) など密結合な事情が絡んでおり、それらを含めて安全に抽出すると
pic.py 側の 280 件超の既存テストに対する回帰リスクが実装コストに見合わない。
prompts/111 が「抽出せず同型再実装でも可」と明示的に許容しているため、
静的な初期化ロジック (K 組み立て・LU 分解・RF/波形テーブル) だけを重複実装する
方針を採った (pic.py は一切変更していないので既存テストは無影響)。
唯一の物理的な差分は Poisson 右辺の電荷ベクトルの作り方: PIC は粒子ごとの
点電荷を P1 補間で厳密に (Galerkin 的に) 節点へ散布するのに対し、fluid2d は
連続場として持つ節点密度 n_e/n_i に対して「集中質量近似」(電荷_i = e·(n_i−n_e)_i
・V_i、V_i は上記の輸送領域節点体積) を使う。これは本モジュールの EAFE/FVM
スキーム全体で採用している質量集中の流儀と一貫しており、フル P1 質量行列を
別途組む複雑さを避けられる。

## 時間積分

fluid1d.py と全く同じ2段構え (陽的検証経路・半陰的既定経路) を踏襲する。
半陰的経路は「①Poisson→E を固定 → ②その E で SG 係数を組んで種ごとに独立な
疎行列を陰的に解く」という逐次法で、誘電緩和時間 τ_d と拡散 CFL (メッシュ最小
エッジ長基準、非構造メッシュなので dx の代わりに輸送領域内の最小辺長 h_min を
使う) の小さい方を超えたら内部でサブステップ分割する (fluid1d.py のコメント
参照。ロジックは同一、格子間隔だけ 1D の dx → 2D の h_min に置き換えている)。

## 陰的線形ソルバー (prompts/111 → 115 で並列反復法へ置換)

種ごとの疎行列 (ion/electron/energy) を毎ステップ (毎サブステップ) 組み直す。
prompts/111 の Phase A では正しさ優先で scipy.sparse.linalg.spsolve (SuperLU の
都度分解) のみを使っていたが、実測プロファイル (2455 節点・200 ステップ) で
transport 65% + energy 28% = 93% がこの spsolve 3本に費やされていることが
判明した。SuperLU の直接分解自体は並列化できないが、行列 M = V_i/dt + K
(時間項 + EAFE 剛性行列) は対角優位 (dt が誘電緩和スケールで小さいほど対角の
時間項が支配的になる) なので、Jacobi (対角) 前処理付き BiCGSTAB が数〜十数
反復で収束する — その主要コスト (matvec 2回/反復) は行並列で完全にスケール
でき、_numba_kernels.csr_matvec_parallel の「行ごとに1スレッドが昇順に積和」
という構成によりスレッド数に依らずビット決定論も保てる (内積・ノルムは
numpy 単一スレッドに残す。理由は _numba_kernels.bicgstab のコメント参照)。

Fluid2dSettings.linear_solver で切り替える: 既定 "iterative" (上記 BiCGSTAB、
x0=前ステップ値で反復数を削減)。収束しなかった場合は自動的に "direct"
(従来の spsolve) へフォールバックする (堅牢性優先、初回のみ warnings に記録)。
"direct" を明示すれば常に spsolve を使う従来経路のまま (比較・検証用に残す)。
Poisson (splu) はプロファイルで 7% に留まり、初期化時の1回の分解を使い回して
いる (陰的輸送のような「毎ステップ組み直し」ではない) ため現状維持とする。
1D (fluid1d.py) は solve_banded (三重対角、O(n)) が既に十分高速なため対象外。
"""

from __future__ import annotations

import math
import os
import time
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import _numba_kernels
from .boltz import BoltzCoeffs, boltz_coeffs_at, boltz_coeffs_from_table
from .fem import EPS0, _radial_index, assemble, assemble_transport_operator
from .fluid1d import FLOOR_N, FLOOR_W, _bernoulli, frost_mobility
from .fluid_coeffs import FluidReactions, build_fluid_reactions, interp_loglog
from .mcc import KB
from .meshing import generate_mesh
from .particles import (
    ME,
    MP,
    QE,
    _adjacency,
    _barycentric_coeffs,
    _build_boundary_tables,
    _solid_elements,
)
from .pic import _auto_thread_cap, _eval_waveform
from .pic1d_presets import edupic_ar_processes
from .schema import Fluid2dSettings, Project, VoltageWaveform, rf_components

# timing dict のうち秒数ではない診断キー (prompts/115、pic.py の WALK_DIAG_KEYS と同じ
# 位置づけ)。build_fluid2d_result の timing.total 集計・フロントの内訳表示 (%計算・合計)
# からはこのキー集合を除外する
FLUID2D_DIAG_KEYS = frozenset({"solver_iters"})


def _effective_fluid2d_threads(requested: int, cpu_count: int | None = None) -> int:
    """Fluid2dSettings.threads の実効値を返す (0=自動)。

    pic.py._auto_thread_cap と同じ式 (max(2, min(16, 論理コア数//2))) をそのまま
    流用する。PIC の _effective_thread_count と異なり「粒子数が閾値未満なら
    強制的に逐次」というゲートは持ち込まない — あれは小規模な粒子配列で
    prange 起動コストが償却できないケースへの対処であり、流体の疎行列反復法
    (matvec を1ステップに数十回呼ぶ) では起動コストの相対的な重みが粒子カーネル
    より小さく、行列サイズに応じた別ゲートを追加する根拠が実測的に無いため
    (prompts/115 のベンチマーク条件でも auto_thread_cap をそのまま使えば十分)。
    """
    if requested > 0:
        return int(requested)
    cores = max(1, int(cpu_count if cpu_count is not None else (os.cpu_count() or 1)))
    return min(_auto_thread_cap(cores), cores)


# history の列名 (毎ステップこの全キーを持つ、fluid1d._HISTORY_KEYS と同じ設計。
# 2D は「壁」がドメイン外周・固体表面など多数になり得るため左右の区別をやめ、
# 全壁合計の電子/イオン損失にまとめる (prompts/111 の history 定義どおり)
_HISTORY_KEYS = ("step", "t", "n_e_total", "n_i_total", "wall_e", "wall_i", "gen_total")


class Fluid2dSimulation:
    """2D/軸対称 ドリフト拡散流体 (EAFE + 半陰的時間積分) シミュレーション本体。

    explicit=True で陽的経路 (検証用、schema には出さない内部 API)。既定
    (explicit=False) は半陰的経路 — Project.fluid2d 経由の通常実行は常に
    こちらを使う (fluid1d.Fluid1dSimulation と同じ設計)。
    """

    def __init__(self, project: Project, explicit: bool = False):
        if project.fluid2d is None:
            raise ValueError("project.fluid2d が指定されていません")
        # periodic 境界は未対応: EAFE のエッジ抽出・壁境界の集計は「節点番号が
        # そのまま物理的な位置に対応する」ことを前提に実装しており、周期スレーブの
        # 正準化 (mesh.periodic_map) をエッジ重み・壁エッジ分類の両方に一貫して
        # 反映する実装コストが (少なくとも第1弾では) 見合わないため、明示的に拒否する
        if any(bc.type == "periodic" for bc in project.geometry.boundaries):
            raise ValueError(
                "fluid2d は periodic 境界に未対応です (EAFE のエッジ抽出・壁エッジ"
                "分類が周期スレーブの正準化に未対応のため、prompts/111 の第1弾では"
                "非対応としています)"
            )
        self.project = project
        self.s: Fluid2dSettings = project.fluid2d
        self.explicit = bool(explicit)
        s = self.s

        self.ridx = _radial_index(project.coord)
        self.rz = self.ridx is not None

        mesh = generate_mesh(project)
        self.mesh = mesh
        self.n_nodes = len(mesh.nodes)
        self.tris = mesh.triangles

        # E ベクトル計算用の重心座標係数 (pic.py と同じ流儀。メッシュ固定なので
        # 初期化時に1回だけ計算する)
        _, self._bc_b, self._bc_c, self._bc_det = _barycentric_coeffs(mesh.nodes, self.tris)

        solid = _solid_elements(project, mesh)
        self.fluid_mask = np.ones(len(self.tris), dtype=bool) if solid is None else ~solid

        # ---- EAFE エッジ重み・節点体積 (fem.py の一様係数ラプラシアン、固体要素を除く) --
        k_uniform, node_vol_full = assemble_transport_operator(
            mesh, project.coord, elem_mask=self.fluid_mask
        )
        self.node_vol_full = node_vol_full
        active = node_vol_full > 0.0
        self.active_idx = np.nonzero(active)[0]
        self.n_active = len(self.active_idx)
        if self.n_active == 0:
            raise ValueError("流体輸送領域が空です (全要素が固体、または要素がありません)")
        glob_to_loc = np.full(self.n_nodes, -1, dtype=np.int64)
        glob_to_loc[self.active_idx] = np.arange(self.n_active)
        self._glob_to_loc = glob_to_loc
        self.node_vol = node_vol_full[self.active_idx]

        k_active = k_uniform[self.active_idx][:, self.active_idx].tocsr()
        k_upper = sp.triu(k_active, k=1).tocoo()
        self.i_idx = k_upper.row.astype(np.int64)
        self.j_idx = k_upper.col.astype(np.int64)
        self.w_ij = -k_upper.data

        self.warnings: list[str] = []
        n_neg = int(np.sum(self.w_ij < 0.0))
        if n_neg > 0:
            self.warnings.append(
                f"EAFE エッジ重み負: {n_neg}/{len(self.w_ij)} 本 (メッシュが局所的に "
                "Delaunay 的でない可能性があります。フラックス計算はそのまま継続しますが、"
                "非単調性 (数値振動) の原因になり得ます — 該当箇所のメッシュを細かくする、"
                "または三角形の内角が鈍角にならないよう再分割することを検討してください)"
            )

        self._build_sparse_pattern()
        self._build_wall_geometry(project, mesh)

        # ---- 陰的反復ソルバー (prompts/115) --------------------------------------
        self.linear_solver = s.linear_solver
        self.effective_threads = _effective_fluid2d_threads(int(s.threads))
        # numba の matvec カーネル (prange) が使うスレッド数を設定に合わせる
        # (numba 無し環境では no-op。pic.py/dsmc.py と同じプロセス共有の set_num_threads
        # 流儀に従う — 実行のたびに設定し直す設計であり、他ソルバーと衝突しない)
        _numba_kernels.set_num_threads(self.effective_threads)
        # BiCGSTAB の収束判定・反復上限。テストでフォールバック経路を強制する際は
        # これらの属性を直接書き換えればよい (モンキーパッチ用の公開インターフェース)
        self._solver_rtol = 1.0e-10
        self._solver_atol = 1.0e-300
        self._solver_max_iter = 200
        self.solver_fallback_count = 0
        self._solver_fallback_warned = False

        # ---- Poisson (fem.assemble を再利用。free/fixed/splu は pic.py と同型の
        #      再実装 — モジュール docstring の設計判断参照) --------------------------
        k_poisson, self.f_static = assemble(project, mesh)
        items = sorted(mesh.dirichlet.items())
        self.fixed = np.array([i for i, _ in items], dtype=np.int64)
        self.v_dc = np.array([v for _, v in items], dtype=np.float64)
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
        self.free = np.setdiff1d(np.arange(self.n_nodes), self.fixed)
        if len(self.free) == 0:
            raise ValueError("自由節点がありません (全節点が Dirichlet)")
        self.k_fd = k_poisson[self.free][:, self.fixed].tocsr()
        self.lu = spla.splu(k_poisson[self.free][:, self.free].tocsc())

        # ---- ガス・イオン輸送係数 (fluid1d.py と同じ規約。mu_i は低電界値 μ_L) -------
        self.n_g = s.gas_pressure_pa / (KB * s.gas_temperature_k)
        self.m_ion = s.ion_mass_amu * MP
        self.mu_i = s.mu_i_ref * (s.n_ref_m3 / self.n_g)
        self.t_i_ev = float(s.t_i_ev)
        self.d_i = self.mu_i * self.t_i_ev  # Frost 補正なし (frost_mobility の docstring 参照)
        self.ion_mobility_model = s.ion_mobility_model
        self.frost_c_td = float(s.frost_c_td)
        # 修正 Frost 用: エッジ (i,j) の Δφ→|E| 換算に使う概算エッジ長 (1D の dx に
        # 相当する量)。EAFE の z=Δφ/T_i 自体はエッジ長を必要としない (w_ij に
        # 幾何因子が既に吸収されている) が、Frost の E/N 評価には |E| を dphi から
        # 復元する必要があるため、この換算専用にノード間ユークリッド距離を使う
        # (mesh は不変なので初期化時に1回だけ計算する)
        p_i = mesh.nodes[self.active_idx[self.i_idx]]
        p_j = mesh.nodes[self.active_idx[self.j_idx]]
        self._edge_len = np.linalg.norm(p_i - p_j, axis=1)

        electron_processes = s.electron_processes if s.electron_processes else edupic_ar_processes()[0]
        self.reactions: FluidReactions = build_fluid_reactions(electron_processes)
        elastic_procs = [p for p in electron_processes if p.kind == "elastic"]
        self._mass_ratio = float(np.mean([p.mass_ratio for p in elastic_procs])) if elastic_procs else 0.0
        self._te_warned = False

        # ---- 電子係数のソース切り替え (fluid1d.py と同じ規約、prompts/117) -----------
        self.electron_model = s.electron_model
        self._boltz: BoltzCoeffs | None = (
            boltz_coeffs_from_table(s.boltz_table) if s.electron_model == "boltzmann" else None
        )

        # ---- 状態変数 (全節点長。非輸送節点は常に 0 のまま — モジュール docstring) ----
        self.n_e = np.zeros(self.n_nodes)
        self.n_i = np.zeros(self.n_nodes)
        self.w = np.zeros(self.n_nodes)
        self.n_e[self.active_idx] = float(s.init_density_m3)
        self.n_i[self.active_idx] = float(s.init_density_m3)
        self.w[self.active_idx] = 1.5 * float(s.init_density_m3) * float(s.init_te_ev)
        self.phi = np.zeros(self.n_nodes)

        # ---- RF 周波数 (dt 自動決定・位相分解の両方に使う。pic.py._find_rf_freq と同じ) --
        self._cycle_freq = self._find_rf_freq()
        self._cycle_bins = int(s.phase_bins)
        self._cycle_enabled = self._cycle_freq is not None and self._cycle_bins > 0
        self._cycle_period = 1.0 / self._cycle_freq if self._cycle_enabled else 0.0

        # ---- dt (None なら RF周期/2000 と 1e-10 の小さい方、fluid1d.py と同じ) --------
        if s.dt is not None:
            self.dt = float(s.dt)
        elif self._cycle_freq is not None:
            self.dt = min(1.0 / self._cycle_freq / 2000.0, 1.0e-10)
        else:
            self.dt = 1.0e-10

        # ---- 安定性の目安警告 (陽的経路のみ) --------------------------------------
        if self.explicit:
            te0, mu_e0, *_ = self._te_and_coeffs(
                self.n_e[self.active_idx], self.w[self.active_idx]
            )
            d_e0 = float(np.max(mu_e0 * te0))
            dt_diff = 0.5 * self.h_min**2 / max(d_e0, self.d_i, 1.0e-300)
            tau_d0 = EPS0 / (QE * max(float(np.max(self.n_e[self.active_idx] * mu_e0)), 1.0e-300))
            dt_bound = min(dt_diff, tau_d0)
            if self.dt > dt_bound:
                self.warnings.append(
                    f"陽的経路: dt={self.dt:.3g}s が安定条件の目安 {dt_bound:.3g}s "
                    "(拡散CFL・誘電緩和時間の小さい方) を超えています (数値不安定の恐れ)"
                )

        # ---- テスト専用フラグ (fluid1d.py と同じ設計) ------------------------------
        self.debug_source_enabled = True
        self.debug_energy_enabled = True
        self.debug_ions_enabled = True
        self.debug_reflective_walls = False

        # ---- 診断・時刻 ------------------------------------------------------------
        self.t = 0.0
        self.step_count = 0
        self.wall: dict[str, float] = {"electron": 0.0, "ion": 0.0}
        self.gen_total = 0.0
        self.history: dict[str, list] = {k: [] for k in _HISTORY_KEYS}
        self.timing: dict[str, float] = {
            "poisson": 0.0, "transport": 0.0, "energy": 0.0, "other": 0.0,
            # 反復ソルバーの総反復数 (診断用、秒数ではないので FLUID2D_DIAG_KEYS で
            # timing.total 集計から除外する。WALK_DIAG_KEYS と同じ位置づけ)
            "solver_iters": 0.0,
        }

        # ---- 時間平均・位相分解アキュムレータ -------------------------------------
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
        self._run_t0 = 0.0

    # ---- 幾何前処理 --------------------------------------------------------------

    def _build_wall_geometry(self, project: Project, mesh) -> None:
        """壁 (吸収境界) エッジの幾何 (法線・境界質量重み・SEE γ) を前計算する。

        壁エッジ = 輸送領域 (固体を除く) の要素境界のうち
          (a) 隣接要素が無い (mesh 外周) かつ symmetry / rz 対称軸ではない、または
          (b) 隣接要素が固体 (dielectric) 要素
        のいずれか (prompts/111 のドメイン定義そのもの)。
        """
        tris = self.tris
        adjacency = _adjacency(tris)  # メッシュ全体で1回だけ (前処理、particles.py 参照)
        self.adjacency = adjacency

        fluid_tri_idx = np.nonzero(self.fluid_mask)[0]
        t_rep = np.repeat(fluid_tri_idx, 3)
        loc_rep = np.tile(np.arange(3), len(fluid_tri_idx))
        neighbor = adjacency[t_rep, loc_rep]
        is_outer = neighbor == -1
        safe_neighbor = np.where(is_outer, 0, neighbor)
        solid_mask = ~self.fluid_mask
        is_solid_nb = (~is_outer) & solid_mask[safe_neighbor]

        # symmetry BC エッジの分類 (particles._build_boundary_tables を再利用。
        # reflect_edges には symmetry のみを渡す — pic.reflect_edges 相当の概念は
        # fluid2d に無いので混ぜない)
        symmetry_edges = sorted(
            {e for bc in project.geometry.boundaries if bc.type == "symmetry" for e in bc.edges}
        )
        is_symmetry = np.zeros(len(t_rep), dtype=bool)
        if symmetry_edges:
            tables = _build_boundary_tables(
                project.geometry.domain.polygon, mesh, adjacency, symmetry_edges, []
            )
            if tables is not None and tables.reflect is not None:
                is_symmetry = tables.reflect[t_rep, loc_rep]

        # rz の対称軸 (r=0): 両端節点の r 座標が 0 とみなせる外周エッジは、symmetry
        # 指定の有無によらず自然境界として扱う (schema._check_rz が r=0 辺への
        # Dirichlet 指定を禁止しているのと対になる規約)
        is_axis = np.zeros(len(t_rep), dtype=bool)
        n1_all = tris[t_rep, (loc_rep + 1) % 3]
        n2_all = tris[t_rep, (loc_rep + 2) % 3]
        if self.rz:
            r_all = mesh.nodes[:, self.ridx]
            scale = float(np.max(np.abs(r_all))) if len(r_all) else 1.0
            tol = 1.0e-9 * (scale if scale > 0.0 else 1.0)
            is_axis = is_outer & (np.abs(r_all[n1_all]) <= tol) & (np.abs(r_all[n2_all]) <= tol)

        is_wall = (is_outer & ~is_symmetry & ~is_axis) | is_solid_nb
        wsel = np.nonzero(is_wall)[0]
        t_wall = t_rep[wsel]
        loc_wall = loc_rep[wsel]
        n1 = n1_all[wsel]
        n2 = n2_all[wsel]
        neighbor_wall = neighbor[wsel]

        nodes = mesh.nodes
        p1, p2 = nodes[n1], nodes[n2]
        po = nodes[tris[t_wall, loc_wall]]
        t_vec = p2 - p1
        seg_len = np.linalg.norm(t_vec, axis=1)
        perp = np.stack([-t_vec[:, 1], t_vec[:, 0]], axis=1)
        mid = 0.5 * (p1 + p2)
        sgn = np.where(np.sum(perp * (po - mid), axis=1) >= 0.0, 1.0, -1.0)
        nrm_in = perp * (sgn / np.where(seg_len > 0.0, seg_len, 1.0))[:, None]  # 内向き法線
        n_out = -nrm_in  # 壁の外向き法線 (流束 BC 用)

        # 境界質量重み (P1 の線積分 ∫N_i ds を厳密に評価。xy: L/2、
        # rz: 2π·(L/6)(2r_i+r_j) — assemble() の rhs 公式 (三角形版) の1D 版。
        # fem.assemble_transport_operator と同じく 2π を最初から掛ける流儀に揃える)
        if self.rz:
            r1 = nodes[n1, self.ridx]
            r2 = nodes[n2, self.ridx]
            w1 = 2.0 * np.pi * (seg_len / 6.0) * (2.0 * r1 + r2)
            w2 = 2.0 * np.pi * (seg_len / 6.0) * (2.0 * r2 + r1)
        else:
            w1 = seg_len * 0.5
            w2 = seg_len * 0.5

        self.wall_tri = t_wall
        self.wall_n1 = self._glob_to_loc[n1]
        self.wall_n2 = self._glob_to_loc[n2]
        self.wall_w1 = w1
        self.wall_w2 = w2
        self.wall_nout = n_out

        # ---- SEE γ (節点単位。mesh.see_gamma (Dirichlet/conductor 由来) と
        #      dielectric 領域の see_gamma (固体表面) の両方を「共有節点は最大値」で
        #      合成する。pic.py の _solid_gamma・_assign_gamma と同じ定義) ------------
        gamma_node = np.zeros(self.n_nodes)
        for node, gamma in mesh.see_gamma.items():
            gamma_node[node] = max(gamma_node[node], gamma)
        solid_gamma_elem = np.zeros(len(tris))
        has_solid_gamma = False
        for i, region in enumerate(project.geometry.regions):
            if region.type == "dielectric" and region.see_gamma > 0.0:
                solid_gamma_elem[mesh.tri_region == i] = region.see_gamma
                has_solid_gamma = True
        if has_solid_gamma:
            solid_edges = np.nonzero(neighbor_wall >= 0)[0]
            if solid_edges.size:
                g_edge = solid_gamma_elem[neighbor_wall[solid_edges]]
                np.maximum.at(gamma_node, n1[solid_edges], g_edge)
                np.maximum.at(gamma_node, n2[solid_edges], g_edge)
        self.gamma_see = gamma_node[self.active_idx]

        # ---- 拡散 CFL 判定用の最小エッジ長 (流体要素のみ、fluid1d.py の dx に相当) ----
        ft = tris[self.fluid_mask]
        p = mesh.nodes[ft]
        d0 = np.linalg.norm(p[:, 1] - p[:, 2], axis=1)
        d1 = np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
        d2 = np.linalg.norm(p[:, 0] - p[:, 1], axis=1)
        self.h_min = float(min(d0.min(), d1.min(), d2.min())) if len(ft) else 1.0

    def _find_rf_freq(self) -> float | None:
        """boundaries / conductor 領域から位相分解の基本周波数を返す (pic.py._find_rf_freq と同じ)。"""
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

    # ---- 係数評価 --------------------------------------------------------------

    def _te_and_coeffs(self, n_e: np.ndarray, w: np.ndarray):
        """アクティブ節点の Te・電子移動度・反応レート係数 (fluid1d._te_and_coeffs と同じ)。

        electron_model の "maxwell"/"boltzmann" 切り替えも fluid1d.py と全く同じ
        (戻り値は7要素: te, mu_e, k_ion, k_exc, nu_m_per_ng, e_ion_ev, e_exc_ev)。
        """
        n_e_safe = np.maximum(n_e, FLOOR_N)
        te = np.maximum((2.0 / 3.0) * w / n_e_safe, 1.0e-6)

        if self.electron_model == "boltzmann":
            mobility_n, k_ion, k_exc, e_ion_ev, e_exc_ev, nu_m_per_ng, _eps_bar, out_of_range = (
                boltz_coeffs_at(self._boltz, te)
            )
            if not self._te_warned and out_of_range:
                lo, hi = self._boltz.eps_grid_ev[0], self._boltz.eps_grid_ev[-1]
                warnings.warn(
                    f"fluid2d: ε̄=(3/2)Te が boltzpm テーブル範囲 [{lo:.3g}, {hi:.3g}] eV の"
                    "外に出ました (テーブル引きはクランプして継続します)"
                )
                self._te_warned = True
            mu_e = mobility_n / self.n_g
            return te, np.asarray(mu_e), k_ion, k_exc, nu_m_per_ng, e_ion_ev, e_exc_ev

        grid = self.reactions.te_grid_ev
        lo, hi = grid[0], grid[-1]
        if not self._te_warned and (float(te.min()) < lo or float(te.max()) > hi):
            warnings.warn(
                f"fluid2d: Te が係数テーブル範囲 [{lo:.3g}, {hi:.3g}] eV の外に出ました "
                "(テーブル引きはクランプして継続します)"
            )
            self._te_warned = True
        mu_e = interp_loglog(grid, self.reactions.transport.mobility_n, te) / self.n_g
        k_ion = interp_loglog(grid, self.reactions.k_ion, te)
        k_exc = interp_loglog(grid, self.reactions.k_exc, te)
        nu_m_per_ng = interp_loglog(grid, self.reactions.transport.nu_m_per_ng, te)
        return (
            te, np.asarray(mu_e), np.asarray(k_ion), np.asarray(k_exc), np.asarray(nu_m_per_ng),
            self.reactions.e_ion_ev, self.reactions.e_exc_ev,
        )

    def _reaction_terms(self, n_e, te, k_ion, k_exc, nu_m_per_ng, e_ion_ev, e_exc_ev):
        """反応源項 (fluid1d._reaction_terms と同じ)。"""
        tg_ev = self.s.gas_temperature_k * KB / QE
        if self.debug_source_enabled:
            s_ion = k_ion * self.n_g * n_e
            loss_ion = e_ion_ev * s_ion
            loss_exc = e_exc_ev * k_exc * self.n_g * n_e
        else:
            s_ion = np.zeros_like(n_e)
            loss_ion = np.zeros_like(n_e)
            loss_exc = np.zeros_like(n_e)
        loss_elastic = 3.0 * self._mass_ratio * nu_m_per_ng * self.n_g * (te - tg_ev) * n_e
        return s_ion, loss_ion + loss_exc + loss_elastic

    def _mu_i_at(self, e_abs):
        """エッジ/壁の |E| における μ_i (fluid1d.Fluid1dSimulation._mu_i_at と同じ設計)。

        "const" は e_abs に関わらず self.mu_i (スカラー) をそのまま返すので、
        呼び出し側で ×1.0 する形にしておけば frost 導入前と完全にビット一致する。
        """
        if self.ion_mobility_model == "const":
            return self.mu_i
        return frost_mobility(self.mu_i, e_abs, self.n_g, self.frost_c_td)

    def _e_field_elements(self, phi: np.ndarray):
        """要素ごとの E = −∇φ (P1 なので要素内一定、pic.py._e_field と同じ規約)。"""
        vt = phi[self.tris]
        ex = -np.sum(vt * self._bc_b, axis=1) / self._bc_det
        ey = -np.sum(vt * self._bc_c, axis=1) / self._bc_det
        return ex, ey

    def _dirichlet_values(self, t: float) -> np.ndarray:
        """時刻 t の Dirichlet 値 (pic.py._dirichlet_values と同じ式)。"""
        v = self.v_dc + np.sum(self.rf_amp * np.sin(self.rf_omega * t + self.rf_phase), axis=1)
        if self._waveforms:
            wf_vals = np.array(
                [_eval_waveform(wf.phase, wf.v, wf.freq_hz, t) for wf in self._waveforms]
            )
            mask = self._wf_index >= 0
            v[mask] += wf_vals[self._wf_index[mask]]
        return v

    def _solve_phi(self, t: float) -> np.ndarray:
        """Poisson 求解 (前分解済み LU で右辺のみ更新)。現在の self.n_e/self.n_i を使う。

        電荷ベクトルは「集中質量近似」(モジュール docstring 参照): charge_i =
        e·(n_i−n_e)_i・V_i (V_i は輸送領域の節点体積、非輸送節点は V_i=0 なので
        n_e/n_i の値によらず寄与しない)。rz は fem.assemble() の「2π を後から
        掛ける」規約に合わせるため、ここで 2π で割ってから足す
        (node_vol_full 自体は 2π 込みの物理単位で持っているため)。
        """
        v = np.zeros(self.n_nodes)
        vd = self._dirichlet_values(t)
        v[self.fixed] = vd
        charge = QE * (self.n_i - self.n_e) * self.node_vol_full
        f_dep = charge / (2.0 * np.pi) if self.rz else charge
        f = self.f_static + f_dep
        rhs = f[self.free] - self.k_fd @ vd
        v[self.free] = self.lu.solve(rhs)
        return v

    # ---- EAFE 輸送ソルバー (種ごとに共用) ----------------------------------------

    def _build_sparse_pattern(self) -> None:
        """疎行列のスパースパターン (グラフ構造) を初期化時に1回だけ確定する。

        性能上の判断 (prompts/111): 種ごと・(サブ)ステップごとに疎行列を
        組み直す必要はあるが、非零パターン (どの (i,j) が非零か) はエッジ構造
        (i_idx, j_idx) だけで決まりステップ間で不変。素朴に毎回
        scipy.sparse.coo_matrix(...).tocsr() を呼ぶと、重複座標の検出・ソート・
        dtype 判定など (プロファイルで実測: 数千ノード規模でも全体時間の大半を
        占めていた) が値が変わらないのに毎回走ってしまう。そこで CSC 形式
        (spsolve/splu がそのまま使える形) の (indices, indptr) と、「素の寄与配列
        (a_ij, -b_ij, b_ij, -a_ij, 対角) の各要素がどの一意な (row,col) スロットへ
        加算されるか」の対応表 (_patt_inv) を1回だけ計算しておき、毎ステップは
        np.bincount 1回で CSC の data 配列を作るだけにする (ソート・重複検出をやり直さない)。
        """
        n = self.n_active
        i_idx, j_idx = self.i_idx, self.j_idx
        diag_idx = np.arange(n)
        rows = np.concatenate([i_idx, i_idx, j_idx, j_idx, diag_idx])
        cols = np.concatenate([i_idx, j_idx, j_idx, i_idx, diag_idx])
        # CSC 用に列優先 (col, row) でユニーク化する。key=col*n+row は col が
        # 主キーになるので、np.unique が返す昇順のユニーク key は自動的に
        # 「列ごとにまとまり、各列内で row 昇順」という CSC の要件を満たす
        # (col < n が保証されているので row/col の桁が衝突しない)
        key = cols.astype(np.int64) * n + rows.astype(np.int64)
        uniq_key, inv = np.unique(key, return_inverse=True)
        uniq_col = uniq_key // n
        uniq_row = uniq_key % n
        indptr = np.searchsorted(uniq_col, np.arange(n + 1))
        self._patt_inv = inv
        self._patt_indptr = indptr.astype(np.int32)
        self._patt_indices = uniq_row.astype(np.int32)
        self._patt_nnz = len(uniq_key)

    def _assemble_active_matrix(self, a_ij, b_ij, wall_diag, dt) -> sp.csc_matrix:
        """後退オイラーの疎行列を組む (EAFE のグラフ構造、モジュール docstring 参照)。

        エッジ (i, j) (i=i_idx, j=j_idx) は F_ij = a_ij·n_i − b_ij·n_j を生み、
        node i の式へ「−F_ij」(M[i,i]+=a_ij, M[i,j]-=b_ij)、node j の式へ
        「+F_ij」= −F_ji (M[j,j]+=b_ij, M[j,i]-=a_ij) として寄与する
        (F_ji=−F_ij、w_ji=w_ij、z_ji=−z_ij から B(±z) の入れ替えで導ける)。
        対角には時間項 V_i/dt と壁コンダクタンスを足す。_build_sparse_pattern で
        前計算した (indices, indptr, inv) を使い、bincount 1回で CSC を直接作る
        (coo 経由のソート・重複検出をステップごとに繰り返さない、性能上の判断)。
        """
        flat_data = np.concatenate([a_ij, -b_ij, b_ij, -a_ij, self.node_vol / dt + wall_diag])
        data = np.bincount(self._patt_inv, weights=flat_data, minlength=self._patt_nnz)
        n = self.n_active
        return sp.csc_matrix((data, self._patt_indices, self._patt_indptr), shape=(n, n))

    def _implicit_transport_solve(self, n_old, a_ij, b_ij, wall_diag, source, extra_rhs, dt):
        """後退オイラー (陰的)。source は体積あたり (×V_i される)、
        extra_rhs は体積化されない既知の追加フラックス (SEE 源・Joule 加熱・
        エネルギー壁損失など、fluid1d._implicit_transport_solve と同じ規約)。

        既定 (linear_solver="iterative") は Jacobi-BiCGSTAB (prompts/115、
        モジュール docstring 参照)、明示指定なら従来の spsolve (direct) を使う。
        """
        m = self._assemble_active_matrix(a_ij, b_ij, wall_diag, dt)
        rhs = self.node_vol / dt * n_old + source * self.node_vol + extra_rhs
        if self.linear_solver == "direct":
            return spla.spsolve(m, rhs)
        return self._iterative_solve(m, rhs, n_old)

    def _iterative_solve(self, m: sp.csc_matrix, rhs: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """Jacobi 前処理付き BiCGSTAB (numba 並列 matvec) で m@x=rhs を解く。

        x0 = n_old (呼び出し元の前ステップ値) をそのまま初期推定に使う: dt が
        誘電緩和スケールで小さく選ばれているため、前ステップからの変化量は
        小さく、ゼロ初期化に比べて反復数を大きく減らせる (モジュール docstring)。
        収束しなければ堅牢性を優先して spsolve (direct) へフォールバックし、
        初回発生時のみ warnings に記録する (以降は solver_fallback_count に
        集計するだけで warnings を汚さない — 反復ごとに毎回出ると実用上ノイズに
        なるため)。
        """
        # spsolve (splu 系) は CSC を前提に組んでいるが、行並列 matvec は行方向の
        # 連続アクセスが要る CSR が必要 (_build_sparse_pattern は spsolve 用に
        # CSC で組んでいるため、反復法用にここで変換する。変換コストは O(nnz) で
        # SuperLU の都度分解よりずっと軽い)
        m_csr = m.tocsr()
        diag = m_csr.diagonal()
        # 対角は必ず非零 (時間項 V_i/dt > 0 が全節点の対角に乗るため)。念のため
        # 0 除算だけガードする (前処理なし=1.0 に落とす、結果の正しさには影響しない)
        diag_inv = np.where(diag != 0.0, 1.0 / diag, 1.0)
        x, n_iter, converged = _numba_kernels.bicgstab(
            m_csr.indptr,
            m_csr.indices,
            m_csr.data,
            rhs,
            x0,
            rtol=self._solver_rtol,
            atol=self._solver_atol,
            max_iter=self._solver_max_iter,
            diag_inv=diag_inv,
        )
        self.timing["solver_iters"] += n_iter
        if converged:
            return x
        if not self._solver_fallback_warned:
            self.warnings.append(
                "反復ソルバー (Jacobi-BiCGSTAB) が収束しませんでした。spsolve (直接法) へ"
                "フォールバックしました (このメッセージは初回のみ表示。以降の発生回数は "
                "solver_fallback_count に集計されます)"
            )
            self._solver_fallback_warned = True
        self.solver_fallback_count += 1
        return spla.spsolve(m, rhs)

    def _explicit_transport_update(self, n_old, a_ij, b_ij, wall_diag, source, extra_rhs, dt):
        """前進オイラー版 (陽的経路)。_implicit_transport_solve と全く同じ係数・
        符号規約を使うため、両者の一致 (テスト5) は係数構築の共有で保証される。
        """
        flux = a_ij * n_old[self.i_idx] - b_ij * n_old[self.j_idx]  # F_ij (i→j)
        net_flux = np.zeros(self.n_active)
        np.add.at(net_flux, self.i_idx, -flux)
        np.add.at(net_flux, self.j_idx, flux)
        wall_loss = wall_diag * n_old - extra_rhs
        return n_old + dt * ((net_flux - wall_loss) / self.node_vol + source)

    # ---- 1ステップ (半陰・陽的 共通の下請け) -------------------------------------

    def _accumulate_wall_and_gen(self, dt, s_ion, gw_i, gw_e) -> None:
        self.wall["ion"] += dt * float(np.sum(gw_i))
        self.wall["electron"] += dt * float(np.sum(gw_e))
        self.gen_total += dt * float(np.sum(np.asarray(s_ion) * self.node_vol))

    def _step_once(self, dt: float, t: float, implicit: bool) -> None:
        """半陰 (implicit=True) または陽的 (False) の1ステップ本体 (fluid1d._step_once と同じ手順)。"""
        t0 = time.perf_counter()
        phi = self._solve_phi(t)
        self.phi = phi
        ex_e, ey_e = self._e_field_elements(phi)

        n_e_a = self.n_e[self.active_idx]
        n_i_a = self.n_i[self.active_idx]
        w_a = self.w[self.active_idx]
        te, mu_e, k_ion, k_exc, nu_m_per_ng, e_ion_ev, e_exc_ev = self._te_and_coeffs(n_e_a, w_a)

        phi_a = phi[self.active_idx]
        dphi = phi_a[self.i_idx] - phi_a[self.j_idx]

        # イオン z (修正 Frost、prompts/116): fluid1d._face_coeffs と全く同じ導出
        # (z_i = (dphi/T_i)・(μ_i(|E_edge|)/μ_L))。E_edge はこのエッジの Δφ を
        # 概算エッジ長 (_edge_len、初期化時に前計算) で割った近似値。"const" は
        # 比が恒等的に 1.0 になりビット完全に従来の z_i=dphi/T_i と一致する
        e_edge = dphi / self._edge_len
        mu_i_edge = self._mu_i_at(e_edge)
        z_i = (dphi / self.t_i_ev) * (mu_i_edge / self.mu_i)
        a_i = self.w_ij * self.d_i * _bernoulli(-z_i)
        b_i = self.w_ij * self.d_i * _bernoulli(z_i)

        # d_e_face = mu_e_face・te_face は electron_model="boltzmann" でも変更不要
        # (fluid1d._face_coeffs の同じ行のコメント参照: w=(3/2)n_e・Te という定義から
        # (2/3)ε̄=Te が厳密に成り立つため、一般化 Einstein D_e=μ・(2/3)ε̄ と自動的に一致する)
        mu_e_face = 0.5 * (mu_e[self.i_idx] + mu_e[self.j_idx])
        te_face = 0.5 * (te[self.i_idx] + te[self.j_idx])
        d_e_face = mu_e_face * te_face
        z_e = -dphi / te_face
        a_e = self.w_ij * d_e_face * _bernoulli(-z_e)
        b_e = self.w_ij * d_e_face * _bernoulli(z_e)

        s_ion, loss_total = self._reaction_terms(n_e_a, te, k_ion, k_exc, nu_m_per_ng, e_ion_ev, e_exc_ev)
        t1 = time.perf_counter()
        self.timing["poisson"] += t1 - t0

        # ---- 壁 BC 係数 (現在の場・Te で評価、fluid1d._wall_side_coeffs の2D 一般化) --
        en_wall = ex_e[self.wall_tri] * self.wall_nout[:, 0] + ey_e[self.wall_tri] * self.wall_nout[:, 1]
        v_th_i = math.sqrt(8.0 * self.t_i_ev * QE / (math.pi * self.m_ion))
        # 壁向きドリフト流束の μ_i も SG 係数と同じ修正 Frost 補正を使う (prompts/116)
        c_i_edge = np.maximum(self._mu_i_at(en_wall) * en_wall, 0.0) + 0.25 * v_th_i
        v_th_e_node = np.sqrt(8.0 * te * QE / (math.pi * ME))
        c_e_n1 = 0.25 * v_th_e_node[self.wall_n1]
        c_e_n2 = 0.25 * v_th_e_node[self.wall_n2]
        if self.debug_reflective_walls:
            c_i_edge = np.zeros_like(c_i_edge)
            c_e_n1 = np.zeros_like(c_e_n1)
            c_e_n2 = np.zeros_like(c_e_n2)
        wall_diag_i = (
            np.bincount(self.wall_n1, weights=self.wall_w1 * c_i_edge, minlength=self.n_active)
            + np.bincount(self.wall_n2, weights=self.wall_w2 * c_i_edge, minlength=self.n_active)
        )
        wall_diag_e = (
            np.bincount(self.wall_n1, weights=self.wall_w1 * c_e_n1, minlength=self.n_active)
            + np.bincount(self.wall_n2, weights=self.wall_w2 * c_e_n2, minlength=self.n_active)
        )

        solve = self._implicit_transport_solve if implicit else self._explicit_transport_update
        zeros = np.zeros(self.n_active)

        # ---- イオン ----
        if self.debug_ions_enabled:
            n_i_new = solve(n_i_a, a_i, b_i, wall_diag_i, s_ion, zeros, dt)
            n_i_new = np.maximum(n_i_new, FLOOR_N)
        else:
            n_i_new = n_i_a.copy()
        gw_i = 0.0 if self.debug_reflective_walls else wall_diag_i * n_i_new

        # ---- 電子 (SEE はいま求めたイオン壁流束を使う) ----
        see_source = self.gamma_see * gw_i
        n_e_new = solve(n_e_a, a_e, b_e, wall_diag_e, s_ion, see_source, dt)
        n_e_new = np.maximum(n_e_new, FLOOR_N)
        gw_e = 0.0 if self.debug_reflective_walls else (wall_diag_e * n_e_new - see_source)
        t2 = time.perf_counter()
        self.timing["transport"] += t2 - t1

        # ---- エネルギー (Joule 加熱はエッジ電力 F_ij・Δφ_ij の折半、モジュール docstring) --
        if self.debug_energy_enabled:
            flux_e_edge = a_e * n_e_new[self.i_idx] - b_e * n_e_new[self.j_idx]
            # Joule 加熱密度は fluid1d と同じ -Γ_e・E (電子は E と逆向きに流れるとき
            # 加熱が正になる符号)。E_face・dx = -(φ_j-φ_i) = dphi (i→j 方向) なので
            # -Γ_e・E_face を「エッジ全体の外延量」に直すと -F_ij・dphi になる
            # (F_ij=flux_e_edge、dphi=φ_i-φ_j)。符号を落として +flux*dphi としていたのは
            # 実装バグ (発熱と冷却が逆転し、電流方向によっては符号付き正フィード
            # バックで発散しうる — 非構造メッシュでの数値不安定の実測調査で発見)。
            p_edge = -flux_e_edge * dphi  # eV/s (extensive)。e は約分されて消える (fluid1d と同じ)
            joule = np.zeros(self.n_active)
            np.add.at(joule, self.i_idx, 0.5 * p_edge)
            np.add.at(joule, self.j_idx, 0.5 * p_edge)
            # 壁のエネルギー損失 q_wall=(5/3)·Te,wall·Γ_e,wall は w の陰解に対して
            # 「陽的な (frozen) 追加項」として引くのではなく、wall_diag と同様に
            # 対角へ加える (実装上の注意、fluid1d からの意図的な変更点)。
            # 理由: Te,wall=(2/3)w/n_e は w に比例するので、q_wall は本質的に
            # 「w に比例する壁損失」= w の緩和項であり、これを陽的に扱うと
            # (壁近傍の集中体積 V_i が小さいメッシュで) 1ステップあたりの
            # 相対損失 q_wall·dt/(V_i·w) が大きくなりやすく、陽的オイラー特有の
            # 振動的発散 (符号が反転しながら振幅が増大する) を起こし得る
            # (このバグを非構造メッシュでの実測で発見し修正した — 経緯は
            # 実装過程の調査記録相当のコメントとしてここに残す)。
            # gw_e (壁への正味電子流束、n_e_new 使用で既知) と n_e_new から
            # 「w に対する壁コンダクタンス」wall_diag_w = (5/3)·gw_e/n_e_new を
            # 作れば q_wall = wall_diag_w·w_new となり、他の壁項と全く同じ
            # 「対角に足す」形の陰的処理に統一できる (電子・イオンの壁項と
            # 同じ安定化の恩恵を受ける)。Joule 加熱項はエッジフラックス (n_e_new、
            # 既知) と Δφ (frozen) だけで決まり w に比例しないため、こちらは
            # 従来どおり陽的な extra_rhs のままでよい。
            wall_diag_w = (5.0 / 3.0) * gw_e / np.maximum(n_e_new, FLOOR_N)
            a_eps, b_eps = (5.0 / 3.0) * a_e, (5.0 / 3.0) * b_e
            w_new = solve(w_a, a_eps, b_eps, wall_diag_w, -loss_total, joule, dt)
            w_new = np.maximum(w_new, FLOOR_W)
        else:
            w_new = 1.5 * n_e_new * self.s.init_te_ev
        t3 = time.perf_counter()
        self.timing["energy"] += t3 - t2

        self._accumulate_wall_and_gen(dt, s_ion, gw_i, gw_e)

        self.n_i[self.active_idx] = n_i_new
        self.n_e[self.active_idx] = n_e_new
        self.w[self.active_idx] = w_new
        self.timing["other"] += time.perf_counter() - t3

    def _joule_relaxation_time(self, te0: np.ndarray, mu_e0: np.ndarray) -> float:
        """Joule 加熱の陽的処理 (extra_rhs) が安定であるための目安時間 τ_J を返す。

        なぜ必要か (非構造メッシュでの実測で判明): Joule 加熱項 (エッジ電力
        F_ij・Δφ_ij の折半) は w に比例しない真の陽的項なので、fluid1d と同じ
        「τ_d (誘電緩和)・拡散 CFL」の二重上限だけでは安定性を保証できない —
        低 Te で電子移動度が上がるガス (Ar の低エネルギー側など) では
        Joule 加熱そのものが正のフィードバック (加熱→移動度増加→電流増加→
        さらに加熱) を持ち得るため、これを陽的に扱う限り別の時間スケール
        τ_J ~ w / (Joule 加熱率) 以下にサブステップを刻む必要がある
        (壁損失の陰的化だけでは解消しない不安定性だったことを実測で確認済み)。
        直近 (前ステップ終了時) の φ・Te・n_e から Joule 加熱率を見積もり、
        加熱側 (正) のみを対象に w/heating の最小値を返す (冷却側は問題にならない)。
        """
        phi_a = self.phi[self.active_idx]
        dphi = phi_a[self.i_idx] - phi_a[self.j_idx]
        mu_e_face = 0.5 * (mu_e0[self.i_idx] + mu_e0[self.j_idx])
        te_face = 0.5 * (te0[self.i_idx] + te0[self.j_idx])
        d_e_face = mu_e_face * te_face
        z_e = -dphi / te_face
        a_e = self.w_ij * d_e_face * _bernoulli(-z_e)
        b_e = self.w_ij * d_e_face * _bernoulli(z_e)
        n_e_a = self.n_e[self.active_idx]
        flux_e_edge = a_e * n_e_a[self.i_idx] - b_e * n_e_a[self.j_idx]
        p_edge = -flux_e_edge * dphi  # _step_once と同じ符号 (-Γ_e・E)
        joule = np.zeros(self.n_active)
        np.add.at(joule, self.i_idx, 0.5 * p_edge)
        np.add.at(joule, self.j_idx, 0.5 * p_edge)
        heating = np.maximum(joule, 0.0)
        w_a = self.w[self.active_idx]
        if not np.any(heating > 0.0):
            return math.inf
        # heating=0 の要素は w_a/heating が桁溢れし RuntimeWarning を出すだけで
        # 結果には使われない (np.where で inf に置き換える) ので、np.errstate で
        # その警告だけ黙らせる (計算結果自体は変えない)
        with np.errstate(divide="ignore", over="ignore"):
            ratio = np.where(heating > 0.0, w_a / np.maximum(heating, 1.0e-300), math.inf)
        return float(np.min(ratio))

    def step(self) -> np.ndarray:
        """流体1サイクル (fluid1d.Fluid1dSimulation.step と同じ設計)。"""
        dt = self.dt
        t = self.t
        accumulating = self._accum_start is not None and self.step_count + 1 >= self._accum_start
        if accumulating:
            self._ensure_accumulators()

        if self.explicit:
            self._step_once(dt, t, implicit=False)
        else:
            # 誘電緩和時間・拡散 CFL・Joule 加熱の三重上限によるサブステップ分割
            # (メッシュ最小エッジ長 h_min を使う以外は fluid1d.step と同一ロジックに、
            # Joule 加熱の安定性条件を追加している。_joule_relaxation_time 参照)
            te0, mu_e0, *_ = self._te_and_coeffs(self.n_e[self.active_idx], self.w[self.active_idx])
            tau_d = EPS0 / (QE * max(float(np.max(self.n_e[self.active_idx] * mu_e0)), 1.0e-300))
            d_e_max = max(float(np.max(mu_e0 * te0)), 1.0e-300)
            dt_diff_bound = 0.5 * self.h_min**2 / d_e_max
            tau_joule = self._joule_relaxation_time(te0, mu_e0)
            dt_bound = min(0.5 * tau_d, dt_diff_bound, 0.5 * tau_joule)
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
        h["n_e_total"].append(float(np.sum(self.n_e[self.active_idx] * self.node_vol)))
        h["n_i_total"].append(float(np.sum(self.n_i[self.active_idx] * self.node_vol)))
        h["wall_e"].append(self.wall["electron"])
        h["wall_i"].append(self.wall["ion"])
        h["gen_total"].append(self.gen_total)
        return self.phi

    # ---- 時間平均・位相分解アキュムレータ (pic.py の averaged_fields/cycle_data と同じキー体系) --

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
        self._accum_phi = np.zeros(self.n_nodes)
        self._accum_e = np.zeros((len(self.tris), 2))
        self._accum_ne = np.zeros(self.n_active)
        self._accum_ni = np.zeros(self.n_active)
        self._accum_te = np.zeros(self.n_active)
        self._accum_ion = np.zeros(self.n_active)
        if self._cycle_enabled:
            nb = self._cycle_bins
            self._cycle_phi = np.zeros((nb, self.n_nodes))
            self._cycle_ne = np.zeros((nb, self.n_active))
            self._cycle_ni = np.zeros((nb, self.n_active))
            self._cycle_te = np.zeros((nb, self.n_active))
            self._cycle_count = np.zeros(nb, dtype=np.int64)

    def _phase_bin(self, t: float) -> int:
        frac = (t % self._cycle_period) / self._cycle_period
        return min(int(frac * self._cycle_bins), self._cycle_bins - 1)

    def _accumulate_fields(self, t_step: float) -> None:
        ex, ey = self._e_field_elements(self.phi)
        n_e_a = self.n_e[self.active_idx]
        n_i_a = self.n_i[self.active_idx]
        w_a = self.w[self.active_idx]
        te, _mu_e, k_ion, *_ = self._te_and_coeffs(n_e_a, w_a)
        s_ion = (k_ion * self.n_g * n_e_a) if self.debug_source_enabled else np.zeros(self.n_active)
        self._accum_phi += self.phi
        self._accum_e[:, 0] += ex
        self._accum_e[:, 1] += ey
        self._accum_ne += n_e_a
        self._accum_ni += n_i_a
        self._accum_te += te
        self._accum_ion += s_ion
        if self._cycle_phi is not None:
            b = self._phase_bin(t_step)
            self._cycle_phi[b] += self.phi
            self._cycle_ne[b] += n_e_a
            self._cycle_ni[b] += n_i_a
            self._cycle_te[b] += te
            self._cycle_count[b] += 1
        self._accum_count += 1

    def averaged_fields(self) -> dict | None:
        """時間平均フィールド (pic.py の averaged_fields と同じキー体系、prompts/111)。

        phi: 節点、e_abs: 要素 (E ベクトルを平均してから絶対値、pic.py と同じ規約)、
        n_e/n_i/t_e/ionization: 節点 (非輸送節点は 0)。
        """
        if self._accum_start is None or self._accum_count == 0:
            return None
        cnt = self._accum_count
        e_avg = self._accum_e / cnt
        e_abs = np.sqrt(e_avg[:, 0] ** 2 + e_avg[:, 1] ** 2)

        n_e_full = np.zeros(self.n_nodes)
        n_i_full = np.zeros(self.n_nodes)
        t_e_full = np.zeros(self.n_nodes)
        ion_full = np.zeros(self.n_nodes)
        n_e_full[self.active_idx] = self._accum_ne / cnt
        n_i_full[self.active_idx] = self._accum_ni / cnt
        t_e_full[self.active_idx] = self._accum_te / cnt
        ion_full[self.active_idx] = self._accum_ion / cnt

        return {
            "phi": self._accum_phi / cnt,
            "e_abs": e_abs,
            "n_e": n_e_full,
            "n_i": n_i_full,
            "t_e": t_e_full,
            "ionization": ion_full,
            "avg_steps": cnt,
        }

    def cycle_data(self) -> dict | None:
        """RF 1周期の位相分解 (pic.py の cycle_data と同じキー体系: phi/n_e/n_i/t_e)。"""
        if not self._cycle_enabled or self._cycle_phi is None:
            return None
        if int(self._cycle_count.sum()) == 0:
            return None
        cnt = np.maximum(self._cycle_count, 1)[:, None].astype(np.float64)
        n_e = np.zeros((self._cycle_bins, self.n_nodes))
        n_i = np.zeros((self._cycle_bins, self.n_nodes))
        t_e = np.zeros((self._cycle_bins, self.n_nodes))
        n_e[:, self.active_idx] = self._cycle_ne / cnt
        n_i[:, self.active_idx] = self._cycle_ni / cnt
        t_e[:, self.active_idx] = self._cycle_te / cnt
        return {
            "bins": self._cycle_bins,
            "freq_hz": self._cycle_freq,
            "phi": self._cycle_phi / cnt,
            "n_e": n_e,
            "n_i": n_i,
            "t_e": t_e,
        }

    # ---- フレーム・実行 ---------------------------------------------------------

    def _make_frame(self) -> dict:
        te_full = np.zeros(self.n_nodes)
        te_full[self.active_idx] = np.maximum(
            (2.0 / 3.0) * self.w[self.active_idx] / np.maximum(self.n_e[self.active_idx], FLOOR_N), 0.0
        )
        counts = {k: v[-1] for k, v in self.history.items()}
        return {
            "type": "frame",
            "step": self.step_count,
            "t": self.t,
            "phi": self.phi.tolist(),
            "n_e": self.n_e.tolist(),
            "n_i": self.n_i.tolist(),
            "t_e": te_full.tolist(),
            "counts": counts,
            "elapsed_s": time.perf_counter() - self._run_t0,
        }

    def run_batch(self, callback=None, should_stop=None, store_frames: bool = True):
        """n_steps 回実行して (history, フレーム列) を返す (fluid1d.run_batch と同じ設計)。"""
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
                    "エネルギーのいずれかが非有限値)。dt を小さくする、メッシュを"
                    "細かくする等を検討してください"
                )
            if self.step_count % self.s.frame_every == 0:
                frame = self._make_frame()
                if store_frames:
                    frames.append(frame)
                if callback is not None:
                    callback(frame)

        self.fields = self.averaged_fields()
        self.cycle = self.cycle_data()
        return self.history, frames

    def prepare_continue(
        self,
        extra_steps: int,
        frame_every: int | None = None,
        avg_steps: int | None = None,
        phase_bins: int | None = None,
    ) -> None:
        """完了/停止後の状態から追加実行の準備をする (fluid1d.prepare_continue と同じ設計)。"""
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


# ---- 結果バンドル組み立て (server.py / batch.py 共通、Phase B (prompts/112-113) で配線) -----


def build_fluid2d_result(sim: Fluid2dSimulation, elapsed_s: float) -> dict:
    """done.result (= ResultsBundle.fluid2d に格納する想定の形) を組み立てる。

    mesh 情報は含めない (既存の MeshResult を別途参照する前提、prompts/111)。
    フィールド配列 (phi/e_abs/n_e/n_i/t_e/ionization) と設定スナップショットのみ。
    """
    fields = None
    if sim.fields is not None:
        f = sim.fields
        fields = {
            "phi": f["phi"].tolist(),
            "e_abs": f["e_abs"].tolist(),
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
            "t_e": c["t_e"].tolist(),
        }
    # solver_iters は秒数ではない診断値 (FLUID2D_DIAG_KEYS) なので合計から除外する
    # (pic.py の WALK_DIAG_KEYS と同じ扱い)
    timing_total = sum(v for k, v in sim.timing.items() if k not in FLUID2D_DIAG_KEYS)
    return {
        "history": sim.history,
        "fields": fields,
        "cycle": cycle,
        "walls": sim.wall,
        "gen_total": sim.gen_total,
        "elapsed_s": elapsed_s,
        "timing": {**sim.timing, "total": timing_total},
        "settings": sim.s.model_dump(),
    }
