# 116: 修正 Frost イオン移動度 + 壁 IEDF (1D PIC / 1D 流体)

## 背景 (ユーザー要望)

1. 「修正フロストを導入してください」— 流体モデルのイオン移動度を定数低電界値
   から **E/N 依存の修正 Frost 式**へ (シース強電界での移動度低下を表現)。
2. 「PICも流体モデルも壁のIEDFを出力できるようにしてほしい」— 壁 (電極) への
   入射イオンエネルギー分布。対象は 1D PIC と 1D 流体 (2D PIC は既存コレクタで
   IEDF 取得可。2D 流体は面上のシース定義が要るため今回はスコープ外 —
   docstring にその旨)。

## Part 1: 修正 Frost 移動度 (fluid1d.py / fluid2d.py 共通)

### 式

    μ_i(E/N) = μ_L / √(1 + (E/N) / C)
    μ_L = mu_i_ref · (n_ref_m3 / n_g)   (従来の低電界値)
    E/N [Td] = |E| / n_g / 1e-21        (1 Td = 1e-21 V·m²)

- C (frost_c_td) は移動度が μ_L/√2 に落ちる E/N。既定 **150 Td** — Ar⁺ in Ar の
  実測 (Ellis et al., At. Data Nucl. Data Tables 17, 177 (1976)) に対する
  粗いフィットである旨と、ガス種に応じて調整すべき旨をコメント (精密フィット
  ではなく工学近似であることを隠さない)。
- D_i は **低電界値のまま** D_i = μ_L·T_i (Frost 補正は掛けない。強電界域では
  ドリフト支配で拡散寄与は小さく、Einstein 関係自体が非平衡で崩れるため
  補正の意味が薄い — 理由コメント)。

### schema

```py
# Fluid1dSettings / Fluid2dSettings 共通:
ion_mobility_model: Literal["frost", "const"] = "frost"  # 新規機能につき frost 既定。
frost_c_td: float = Field(150.0, gt=0)                   # const では無視
```

### 実装

- SG/EAFE フラックス係数を組む場所で、界面/要素の |E| (前ステップ場 — 既存の
  遅延評価と同じ) から μ_i を界面ごとの配列にする (これまでスカラーだった
  μ_i がベクトル化される。d_i はスカラーのまま)。
- 壁境界フラックスの μ_i·E も同じ補正値を使う。
- "const" は従来経路とビット不変。
- テスト: (a) E=0 で frost == const (ビット一致)、(b) 一様強電界 (300 Td 相当)
  の片側シース的ケースで μ_i が μ_L/√(1+300/150)=μ_L/√3 に一致 (単体関数)、
  (c) CCP スモークが frost でも収束・有限、(d) 既存テスト (const 明示 or
  許容誤差内) 全パス。

## Part 2: 壁 IEDF

### 2a. 1D PIC (pic1d.py) — 粒子ベース (厳密)

- 平均区間中、壁で吸収されたイオンの **全運動エネルギー E = ½m|v|² [eV]** と
  重みを壁ごと (左/右) に記録し、重み付きヒストグラム化。
- 設定: `Pic1dSettings.wall_iedf_bins: int = Field(100, ge=0, le=1000)`
  (0=無効)。e_max は自動 (既存 EEDF の自動決定・ビン規約を読んで同じ流儀に。
  サンプル保持方式もメモリ配慮を含め EEDF 実装に合わせる)。
- 結果: `result.wall_iedf: {left: {e_centers, f, mean_energy_ev, total_weight,
  n_samples}, right: {...}} | null` (EEDF の結果形に揃える)。
- continue: 平均区間リセット規約に従う。
- テスト: DC 強バイアス (無衝突・左電極 -100V) で左壁 IEDF のピークが
  ≈ 100 eV + 初期エネルギー分の幅 (平均 ≈ シース電位差、rtol 10%)。
  重み和 = 壁吸収重みと一致。

### 2b. 1D 流体 (fluid1d.py) — 無衝突シース近似 (モデルベース)

流体は粒子を持たないため、**位相分解シース電圧 + イオン走行時間フィルタ**で
IEDF を再構成する (工学近似。仮定を docstring に明記: 無衝突シース、
CX 衝突による低エネルギーテール無し):

1. 平均区間で phase_bins の位相分解 φ(x, φ_rf) (cycle) と時間平均シースエッジ
   位置 s̄ (Brinkmann、既存 sheath) を得る。
2. 壁ごとの瞬時シース電圧 V_sh(φ_rf) = φ(シースエッジ節点, φ_rf) − φ(壁, φ_rf)
   (イオンを壁へ加速する符号で正)。
3. イオン走行時間 τ_i = 3·s̄·√(m_i / (2e·V̄_sh)) (Lieberman & Lichtenberg の
   衝突なし Child シース走行時間。V̄_sh = 時間平均シース電圧)。
4. V_sh(t) を1次ローパス dV_eff/dt = (V_sh − V_eff)/τ_i で周期定常までフィルタ
   (τ_i ≪ T_rf → 瞬時追従で二山型、τ_i ≫ T_rf → e·V̄_sh の単峰に収束 —
   IEDF の transit-time 効果の標準的な振る舞い。コメントに)。
5. IEDF: E(φ_rf) = e·V_eff(φ_rf) を、壁イオン流束 Γ_i(φ_rf) (位相分解が無ければ
   時間平均) で重み付けしてヒストグラム化。
- RF なし (DC) の縮退: 単峰 (1ビンに集中) で例外を出さない。
- 設定: `Fluid1dSettings.wall_iedf_bins: int = Field(100, ge=0, le=1000)`
  (0=無効)。phase_bins=0 or RF なしなら V̄_sh の単峰。
- 結果: `result.wall_iedf` (2a と同形 + `model: "collisionless_sheath"` を付与
  — PIC の粒子ベースと区別できるように)。
- テスト: (a) τ_i ≫ T_rf の条件で IEDF が単峰 (分散が小さい)、(b) 合成 V_sh
  (正弦) + 小さな τ_i でヒストグラム両端に二山 (フィルタ・ヒストグラム部を
  純関数に切り出して単体テスト)、(c) CCP スモークで有限・重み正。

### frontend (両モデル)

- types.ts: WallIedf 型、Pic1dResult / Fluid1dResult に wall_iedf、設定に
  wall_iedf_bins。
- 1D PIC ビュー (Plot1dView) と 流体 1D ビュー: 「壁 IEDF」チャート (左右2色、
  既存 EEDF チャートの流儀・CSV 書き出し付き)。流体側はチャート下に
  「無衝突シース近似 (CX 衝突による低エネルギー成分は含みません)」の hint。
- パネル: wall_iedf_bins 入力 (0=無効)。

## 検証

- backend: `cd /home/claude/ES-Sim/backend && python -m pytest tests/ -q` 全件パス
  (現在 347 + 新規)。
- frontend: `cd /home/claude/ES-Sim/frontend && npx tsc --noEmit && npx vite build`。

## 注意

- コメントは日本語で「なぜ」(Frost 式の出典と工学近似である旨、D_i 非補正の理由、
  流体 IEDF の仮定と限界、τ_i の式の出典)。
- git commit はしない。const / wall_iedf_bins=0 の既定外経路は従来とビット不変。
- Math.min/max スプレッド禁止、CommitNumberInput 系。
