# 106: 流体モデル Phase B — 1D ドリフト拡散ソルバー fluid1d.py

## 背景

prompts/104 計画の Phase B (最難関)。Phase A (fluid_coeffs.py、実装済み) の係数を
使い、1D ドリフト拡散 + 電子エネルギー方程式 + Poisson の自己無撞着ソルバーを
実装する。スタディ「流体 (1D)」の心臓部。1D PIC (pic1d.py) と同一条件で直接
比較できることが設計目標 — 設定スキーマ・結果形式は意図的に pic1d に揃える。

## スキーマ (backend/es_sim/schema.py)

```py
class Fluid1dSettings(BaseModel):
    """1D プラズマ流体 (ドリフト拡散 + 電子エネルギー、prompts/104-106)。"""
    gap_m: float = Field(..., gt=0)
    n_cells: int = Field(200, ge=16, le=100000)
    left: Pic1dElectrode = Pic1dElectrode()    # 電圧合成・SEE γ を共用 (fn は無視、
    right: Pic1dElectrode = Pic1dElectrode()   # validator で fn 指定時エラー)
    init_density_m3: float = Field(..., gt=0)
    init_te_ev: float = Field(2.0, gt=0)
    gas_pressure_pa: float = Field(..., gt=0)  # 一様背景ガス
    gas_temperature_k: float = Field(300.0, gt=0)
    ion_mass_amu: float = Field(39.948, gt=0)
    # イオン低電界移動度 μ_i·p (圧力規格化) [m^2/(V·s)·Pa]…ではなく分かりやすく:
    # 標準状態でない任意ガス密度へのスケールは μ_i = mu_i_ref · (n_ref/n_g)
    mu_i_ref: float = Field(1.45e-1, gt=0)  # Ar+ in Ar の 1 Torr 換算値を既定に
    n_ref_m3: float = Field(3.22e22, gt=0)  # 上記 μ_i_ref の基準ガス密度 (1 Torr, 300K)
    t_i_ev: float = Field(0.026, gt=0)      # イオン温度 (D_i = μ_i T_i)
    electron_processes: list[XsProcess] = []  # 空なら eduPIC Ar 解析式を既定使用
    dt: float | None = Field(None, gt=0)      # None = RF 周期/2000 か 1e-10 の小さい方
    n_steps: int = Field(20000, gt=0)
    frame_every: int = Field(200, gt=0)
    avg_steps: int | None = Field(None, gt=0)
    phase_bins: int = Field(40, ge=0)
    seed 不要 (決定論)。
class Project: fluid1d: Fluid1dSettings | None = None
```

## ソルバー (backend/es_sim/fluid1d.py — 新規)

### 変数と格子

- 節点 n_nodes = n_cells+1 (pic1d と同じ一様格子 xg)。節点変数: n_e, n_i,
  w = (3/2) n_e T_e [eV·m^-3]、φ。フラックスは半節点 (セル界面)。

### Scharfetter–Gummel フラックス

界面 i+1/2 (dx、ドリフト速度 v = ±μE、拡散 D) の SG フラックス:

    Γ = (D/dx) · [ B(−z)·n_i − B(z)·n_{i+1} ],  z = ±μ E dx / D (符号は種の電荷向き)
    B(z) = z/(e^z − 1) (Bernoulli 関数。|z|→0 で 1 − z/2 + …、オーバーフロー対策の
    分岐実装: |z|<1e-4 は級数、z>500 は漸近形)

電子エネルギーフラックスも SG 型 (係数 5/3: q = (5/3)·SG(μ_e→(5/3)…) —
Hagelaar & Kroesen (2000) の標準形: エネルギー輸送係数 μ_ε=(5/3)μ_e、
D_ε=(5/3)D_e で w に対する SG フラックス)。

### 時間積分 (2段構え — 計画のリスク対策)

1. **陽的経路 (explicit=True、検証用)**: Γ を現在値で評価し前進オイラー、
   Poisson は毎ステップ解く。dt 安定条件 (拡散 CFL・誘電緩和) を docstring に。
2. **半陰的経路 (既定)**:
   - 各種の連続の式を後退オイラーで陰化 (SG 係数は前ステップの E で評価):
     三重対角を solve_banded で解く (種ごと独立)。
   - **Poisson の半陰化** (誘電緩和制約の回避): 電子連続を解く際の E に
     予測子 E* を使う代わりに、簡潔で実績のある逐次法:
     ステップ順 = ①現在の n から Poisson → E、②E で SG 係数を組み
     電子・イオン連続を陰的に解く、③エネルギー方程式を陰的に解く。
     dt が誘電緩和時間 τ_d = ε0/(e n_e μ_e) を超えると逐次法は不安定に
     なり得るため、**dt > 0.5·τ_d の場合は電子連続と Poisson を連立した
     真の半陰スキーム** (n_e^{new} を Poisson の電荷に使う: φ と n_e の
     連立をブロック三重対角…実装が重い場合は「dt を τ_d/2 に内部分割する
     サブステップ」でも可 — どちらを選んだか理由と共にコメント)。
   - 陽的経路と半陰経路の一致テスト (小 dt) を必ず入れる。
3. ソース項: S_ion = k_ion(Te)·n_g·n_e (電子・イオン共通)、エネルギー損失
   = (e_ion·k_ion + e_exc·k_exc + 3(m/M)·⟨σ_m v⟩·(Te−Tg)) ·n_g·n_e。
   Te = (2/3) w/n_e (n_e フロア 1e6 m^-3 でゼロ除算回避)。
   係数は interp_loglog (Te は TE_GRID 範囲へクランプ + 範囲外 warning 1回)。

### 境界条件 (Hagelaar & Kroesen (2000) 系、出典 docstring)

- イオン: 壁向きドリフト流束 (μ_i n_i E が壁向きのときのみ) + (1/4) n_i v_th,i。
- 電子: (1/4) n_e v_th,e − γ·Γ_i (SEE、Pic1dElectrode.see_gamma)。
- エネルギー: q_wall = (5/3)·Te ·Γ_e,wall (2Te 系の係数差は流儀があるので採用値を
  コメント)。
- 電位: 両端 Dirichlet、v_dc + Σ RF + CSV (pic1d の _electrode_voltage を流用
  できる形に共通化 — pic1d 側は挙動不変で)。

### 診断・結果 (pic1d と同じ形に揃える)

- history: {step, t, n_e_total, n_i_total, wall フラックス積算 (左右 e/i)}
- profiles (時間平均): x, phi, e, n_e, n_i, t_e, ionization (S_ion 平均)
- cycle (位相分解、phase_bins>0 + RF): phi/n_e/n_i/t_e — pic1d の cycle 規約
- sheath: Brinkmann (pic1d.brinkmann_sheath_edge / _sheath_pair を import 流用)
- elapsed_s、timing {poisson, transport, energy, other}
- continue: prepare_continue で run(n)+continue(m) == run(n+m) **ビット一致**
  (決定論なので乱数考慮不要)
- `build_fluid1d_result(sim, elapsed_s)` (build_pic1d_result と同じ役割。
  server/batch から使う — Phase C で配線するのでここで定義だけ)

## テスト (backend/tests/test_fluid1d.py — 新規)

1. **SG フラックス極限**: B(z) の級数/漸近分岐、Pe→0 で中心差分拡散、
   Pe→∞ で完全風上に一致 (単体関数テスト)。
2. **両極性拡散減衰**: ソース項無効 (テスト用フラグ)・両端接地・初期 cos 分布で
   減衰率 ≈ D_a (π/L)² (D_a = (μ_i D_e + μ_e D_i)/(μ_e+μ_i)、rtol 10%)。
3. **ボルツマン平衡**: イオン固定・ソース無し・電子のみ平衡化で
   n_e ∝ exp(eφ/Te) (φ 差 vs log 密度比、rtol 5%)。
4. **粒子収支**: (生成 − 壁損失) の積算 = 総粒子数変化 (機械精度 rtol 1e-10)。
5. **陽的 vs 半陰**: 小 dt で両経路の profiles が rtol 1e-4 一致。
6. **CCP 定常スモーク**: eduPIC Ar 既定断面積・13.56 MHz 150V・50 Pa・
   gap 2.5cm で実行し、定常で: 中心 n_e が 1e14〜1e18、Te 中心 1〜5 eV、
   シース (壁近傍 n_i > n_e)、全量有限。CI 1分以内に収める
  (n_cells 100・数万ステップ、半陰 dt で回ることを確認しつつ調整可)。
7. **continue ビット一致**。validator (fn 指定でエラー等)。

`python -m pytest tests/ -q` 全件パス (現在 299 + 新規)。frontend 変更なし。

## 注意

- コメントは日本語で「なぜ」(SG の導出、半陰化の設計判断、BC の出典)。
- git commit はしない。numpy ベクトル化 (節点ループ禁止)。solve_banded。
- pic1d.py の共通化 (電圧評価) は pic1d の既存テストがビット不変で通ること。
- 既存モジュールへの変更は最小 (schema 追加 + pic1d 電圧評価の共通化のみ)。
