# 111: 2D 流体 Phase A — 非構造メッシュ流体ソルバー fluid2d.py (EAFE/FEM-SG)

## 背景 (ユーザー要望)

「(1D 流体を) 2D/2D軸対称に拡張してほしい」
1D 流体 (fluid1d.py、実装済み) を、既存の非構造三角形 FEM メッシュ上の
2D (coord "xy") / 軸対称 ("rz"・"rz_x0") へ拡張する。本プロンプトは
**ソルバーコアとテストのみ** (server/UI は prompts/112-113)。

## 離散化: EAFE (Edge-Averaged Finite Element = FEM-SG)

半導体デバイスシミュレーションの標準手法。P1-FEM の剛性行列の**エッジ重み**に
Scharfetter–Gummel を載せる node-centered 有限体積:

- 剛性行列 K (拡散係数 1 で組んだラプラシアン) のオフ対角 K_ij は
  エッジ (i,j) の「実効断面/距離」重み w_ij = −K_ij に相当する
  (Delaunay 的メッシュで w_ij ≥ 0。負重みエッジは非単調性の源 — 出現数を
  数えて warning に載せる。フラックスはそのまま計算して良い)。
- 種 s のエッジフラックス (i→j、電位差 Δφ = φ_i − φ_j):
    z = s の符号付き移動度 μ_s·Δφ / D_s  (無次元ペクレ数)
    F_ij = w_ij · D_s · [B(−z)·n_i − B(z)·n_j]、B = Bernoulli (fluid1d の実装を流用)
  ∂n_i/∂t · V_i = −Σ_j F_ij + S_i·V_i (V_i = 集中質量の節点体積)
- **既存資産の再利用が鍵**:
  - K は fem.py の assemble (ε_r=1 相当) と同じ要素幾何で組める。rz では
    既存実装が r 重み付き積分をしているはず — 同じ要素ルーチンを使えば
    軸対称の重みが自動で正しくなる (fem.py を読み、流用方法を決める。
    coefficient 抽出用に「一様係数ラプラシアンの K と節点体積」を返す
    ヘルパーを fem.py に追加して良い — 既存 solve の挙動は不変で)。
  - 節点体積 V_i は pic.py の _node_area (rz では r 重み付き) と同じ流儀。
  - **Poisson は pic.py の場ソルブをそのまま再利用**: PicSimulation が毎ステップ
    やっている「fixed/free 分解 + splu 事前分解 + 電圧波形 Dirichlet」を
    流体からも使えるよう、pic.py の該当部分 (K 組み立て・_dirichlet_values・
    solve) を**共通クラス/関数に抽出** (pic の既存テストがビット不変で
    通ることが絶対条件。抽出せず fluid2d 内で同型再実装でも可 — 判断理由を
    コメントに)。
  - 電子エネルギー w = (3/2) n_e T_e も同じ EAFE (係数 5/3) — fluid1d と同じ
    Hagelaar & Kroesen 形。
  - E ベクトル (勾配) は要素定数 → 節点平均、既存 pic.py の流儀。

## ドメインと境界条件

- **輸送領域**: 全メッシュから「固体要素」(conductor / dielectric solid の
  region 要素 — pic.py が粒子を入れない扱いにしている領域) を除く。
  pic.py の固体判定を読んで同じ定義に (無ければ material 配列から判定)。
  固体に接する節点は壁節点。
- **壁 (吸収) 境界**: Dirichlet 電極エッジ・固体表面・domain 外周の非対称
  エッジ。フラックス BC (fluid1d と同じ Hagelaar & Kroesen 系):
  イオン = 壁向きドリフト + (1/4) v_th、電子 = (1/4) v_th − γ·Γ_i (SEE は
  BoundaryCondition.see_gamma / Region.see_gamma を流用)、エネルギー =
  (5/3) Te Γ_e。壁エッジの長さ重み (rz は r 重み) で節点に配分。
- **対称境界**: symmetry BC エッジ・rz の対称軸 (r=0) は自然境界 (フラックス 0
  = 何もしない)。periodic は第1弾では未対応 (ValueError、理由コメント)。
- 電圧: 既存の _dirichlet_values (v_dc + RF + CSV) をそのまま使用。

## スキーマ (backend/es_sim/schema.py)

```py
class Fluid2dSettings(BaseModel):
    """2D/軸対称 プラズマ流体 (prompts/111)。geometry/mesh/境界条件は既存の
    プロジェクト設定 (Dirichlet 電圧・voltage_rf・waveform・see_gamma) を使う。"""
    init_density_m3: float = Field(..., gt=0)
    init_te_ev: float = Field(2.0, gt=0)
    gas_pressure_pa: float = Field(..., gt=0)
    gas_temperature_k: float = Field(300.0, gt=0)
    ion_mass_amu: float = Field(39.948, gt=0)
    mu_i_ref: float = Field(1.45e-1, gt=0)
    n_ref_m3: float = Field(3.22e22, gt=0)
    t_i_ev: float = Field(0.026, gt=0)
    electron_processes: list[XsProcess] = []   # 空 = eduPIC Ar 解析式
    dt: float | None = Field(None, gt=0)
    n_steps: int = Field(20000, gt=0)
    frame_every: int = Field(200, gt=0)
    avg_steps: int | None = Field(None, gt=0)
    phase_bins: int = Field(0, ge=0)
class Project: fluid2d: Fluid2dSettings | None = None
```

## 時間積分 (fluid1d の方針を踏襲)

- 逐次半陰: ①Poisson (splu 再利用で高速) → ②各種の輸送を後退オイラー陰解
  (係数は現在の E で凍結)。陰解の疎行列は毎ステップ組み直し —
  scipy.sparse (CSR) + spsolve or splu。**性能メモ**: 数万節点で
  splu が毎ステップ 3 回は重い。まず正しさ優先で spsolve、その後
  行列構造 (スパースパターン) 固定の再利用や bicgstab+ILU を試して
  1万節点・1万ステップが実用時間 (数分〜十数分) に入るよう調整
  (どこまでやったか・実測 ms/step をレポート)。
- サブステップ: fluid1d と同じ τ_d と拡散 CFL の二重上限 (メッシュ最小
  エッジ長基準)。
- explicit=True 検証経路も fluid1d 同様に。

## 診断・結果

- history {step, t, n_e_total, n_i_total, wall_e, wall_i, gen_total} (体積積分)
- 時間平均: 節点配列 phi/e_abs/n_e/n_i/t_e/ionization (2D PIC の
  averaged_fields と同じキー体系 — フロントの既存フィールド表示に載せるため。
  pic.py の picFields の形を読んで揃える)
- cycle (phase_bins>0 + RF): 2D PIC の cycle_data と同じ形 (phi/n_e/n_i/te)
- timing {poisson, transport, energy, other}、elapsed_s、continue ビット一致
- `build_fluid2d_result(sim, elapsed_s)` (mesh 情報 = 既存 MeshResult 参照で
  フロントが描けるよう、結果にはフィールド配列のみ。設定スナップショット含む)

## テスト (backend/tests/test_fluid2d.py — 新規)

1. **1D 突き合わせ (最重要)**: 細長い矩形 (例 25mm × 1mm、上下 symmetry) の
   xy メッシュで 2D 流体を実行し、断面平均プロファイルが**同条件の fluid1d と
   一致** (定常の n_e/T_e/φ、rtol 10%。SG/BC/積分の 2D 実装が 1D と整合する
   ことの強い検証)。CI 予算内の条件に調整。
2. **ボルツマン平衡 (2D)**: イオン固定・ソース無しで n_e ∝ exp(eφ/Te)
   (非一様メッシュの矩形、rtol 5%)。
3. **粒子収支**: 体積積分 (生成 − 壁損失) = ΔN (rtol 1e-8 程度。EAFE+集中質量
   で保存が成り立つ構成にする)。
4. **rz スモーク**: 軸対称ドメイン (rz) で実行し全量有限・軸で発散しない・
   n_e ≥ 0。
5. **陽的 vs 半陰一致** (小 dt、rtol 1e-3)。
6. validator (periodic でエラー等)。既存テスト全パス (現在 321 + 新規)。

`python -m pytest tests/ -q` 全件パス。frontend は触らない。

## 注意

- コメントは日本語で「なぜ」(EAFE の導出、負重みエッジ、rz 重み、性能判断)。
- git commit はしない。numpy/scipy ベクトル化 (エッジ・節点ループ禁止。
  エッジリストは K の疎構造から一括抽出)。
- pic.py / fem.py の既存挙動はビット不変 (共通化する場合は既存テストで担保)。
- 固体領域・BC の定義は pic.py と同じに (独自定義を発明しない)。
