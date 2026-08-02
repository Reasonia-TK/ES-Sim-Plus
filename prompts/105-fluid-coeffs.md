# 105: 流体モデル Phase A — 係数生成モジュール fluid_coeffs.py

## 背景

prompts/104 の計画 Phase A。1D ドリフト拡散流体 (prompts/106 で実装) が使う
輸送・反応係数を、既存の断面積データ (XsProcess: lxcat インポート /
pic1d_presets.edupic_ar_processes) から生成する。

## backend/es_sim/fluid_coeffs.py (新規)

### レート係数 (Maxwell 平均)

```py
def rate_coefficient_table(
    proc: XsProcess, te_grid_ev: np.ndarray
) -> np.ndarray:
    """k(Te) = ⟨σ v⟩ [m^3/s]。等方 Maxwell 分布での平均:
    k = √(8e/(π m_e)) · Te^{-3/2} · ∫ σ(E)·E·exp(−E/Te) dE  (E, Te は eV)
    積分は断面積表の E 範囲で台形則 (表の外は σ=0)。指数の急峻さに備えて
    E グリッドは断面積表の点 + Te ごとの指数スケール補助点を併合する。"""
```

- 閾値反応 (excitation/ionization) は threshold_ev 未満で σ=0 (既存表がそうなって
  いることを確認しつつ、閾値点の挿入で立ち上がりを保つ)。
- Te グリッド: `TE_GRID_EV = np.geomspace(0.05, 30.0, 160)` をモジュール定数に。

### 電子輸送係数

```py
@dataclass
class ElectronTransport:
    te_grid_ev: np.ndarray
    nu_m_per_ng: np.ndarray   # 運動量移行衝突頻度 / n_g [m^3/s] = ⟨σ_m v⟩
    mobility_n: np.ndarray    # μ_e·n_g [1/(m·V·s)] = e/(m_e · nu_m_per_ng) ... 単位注意
def electron_transport(elastic: XsProcess, te_grid_ev) -> ElectronTransport
```

- ν_m/n_g = ⟨σ_m v⟩ (elastic の断面積を運動量移行断面積とみなす。eduPIC/Phelps の
  elastic は運動量移行断面積である旨コメント)。
- μ_e n_g = e / (m_e · ⟨σ_m v⟩) — 正確には Maxwell 平均の定義差があるが、
  第1弾はこの標準近似 (BOLSIG 2項近似との差は係数レベル。docstring に明記)。
- D_e = μ_e T_e (Einstein)。D はテーブル化せず呼び出し側で積 (μ·Te) を取る。

### 反応セット構築

```py
@dataclass
class FluidReactions:
    te_grid_ev: np.ndarray
    k_ion: np.ndarray          # 電離 [m^3/s] (複数あれば和)
    k_exc: np.ndarray          # 励起の和
    e_ion_ev: float            # 電離閾値 (加重平均で可)
    e_exc_ev: float            # 励起の平均損失 [eV] (k 重み平均、Te 依存でも定数近似可)
    transport: ElectronTransport
def build_fluid_reactions(electron_processes: list[XsProcess]) -> FluidReactions
```

- elastic 弾性エネルギー損失 (2m/M 項) 用に ⟨σ_m v⟩ を流用 (M は呼び出し側から)。
- 補間ヘルパー `interp_loglog(te_grid, table, te)` (k は log-log 補間が自然。
  0 値は線形フォールバック。ベクトル化)。

## テスト (backend/tests/test_fluid_coeffs.py — 新規)

1. **一定断面積の解析解**: σ = σ0 (0.1〜1000 eV で一定・閾値なし) を合成した
   XsProcess に対し k(Te) = σ0·v̄ (v̄ = √(8eTe/(π m_e))) と rtol 1e-2 で一致
   (積分グリッド分解能の検証)。
2. **閾値反応の低温極限**: 閾値 10 eV・Te = 1 eV で k が exp(−10) スケールの
   小ささ (k(1eV)/k(10eV) < 1e-3)。
3. **eduPIC Ar セット**: edupic_ar_processes() から build_fluid_reactions が
   例外なく構築でき、k_ion(3eV) が 1e-16〜1e-13 m³/s のオーダー、
   μ_e·n_g が 1e23〜1e25 1/(m·V·s) のオーダー (物理的常識範囲)。
4. 補間: log-log 補間がグリッド点上で厳密一致、範囲外クランプ。

`python -m pytest tests/ -q` 全件パス (現在 285 + 新規)。frontend 変更なし。

## 注意

- コメントは日本語で「なぜ」(Maxwell 平均の式、2項近似との差、単位系)。
- git commit はしない。numpy ベクトル化 (Te ループは可、E 積分はベクトル)。
- 既存モジュールへの変更なし (新規ファイル + テストのみ)。
