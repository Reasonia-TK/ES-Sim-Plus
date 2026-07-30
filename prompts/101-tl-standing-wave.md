# 101: VHF 定在波スタディ — 非線形径方向伝送線路モデル (定在波効果・波長短縮・高調波)

## 背景 (ユーザー要望)

「定在波効果や波長短縮による高調波を検証したい」
VHF 大面積 CCP の定在波効果の標準モデルである**非線形径方向伝送線路モデル**
(Lieberman et al., PSST 11, 283 (2002); Chabert et al., Phys. Plasmas 11, 1775
(2004); Chabert, J. Phys. D 40, R63 (2007) レビュー) を新スタディとして実装する。
円板電極 (半径 R、ギャップ l)・中心給電・軸対称の半径方向 1D。
シースの非線形容量 (電荷-電圧が2次) が高調波を生成し、各高調波は波長短縮した
定在波を張る — これをそのまま数値で再現する。

## 物理モデル (backend/es_sim/tl.py — 新規)

### 伝送線路部 (半径方向、FDTD 風の staggered 格子)

電極間電圧 V(r,t)、電極を流れる径方向全電流 I(r,t):

    ∂V/∂r = −L'(r) ∂I/∂t          L'(r) = μ0 l / (2π r)   [H/m]
    ∂I/∂r = −2π r J_z(r,t)         (電荷保存。J_z は放電を軸方向に流れる電流密度 [A/m²])

- r ∈ [r_feed, R]。V は整数節点、I は半節点の staggered 配置で leapfrog
  (1D FDTD と同型)。
- 給電: r = r_feed (内側境界) で V を強制 V(r_feed, t) = V0 sin(2π f0 t)
  (電圧駆動)。r_feed 既定 = R/50 (r→0 の特異性回避。コメントに理由)。
- 外周 r = R: 開放端 I(R) = 0 (反射 → 定在波が立つ物理的な条件)。

### シャント部 (各節点の軸方向直列チェーン: シース/バルク/シース)

各半径節点で、上下シース (非線形容量) とバルク (電子慣性 L + 衝突抵抗 R) の
直列回路に同一の J_z が流れる:

    状態: q_a (上シースの面電荷 [C/m²])、q_b (下シース)、J_z
    dq_a/dt = +J_z、dq_b/dt = −J_z
    行列シース (一様イオン密度 n_s) の電荷-電圧: V_s(q) = q² / (2 e n_s ε0)
      — この2次非線形が高調波の源。q < 0 は物理的に不可 (シース崩壊) なので
      q は 0 でクリップし、崩壊中はその側のシース電圧 0 (整流性)。
    バルク: 厚さ d = l − s_a0 − s_b0 (s0 = 平衡シース厚)、
      L_b = m_e d / (e² n_e ε0? → 正しくは L_b = m_e d / (e² n_e) [H·m²]、
      R_b = m_e ν_m d / (e² n_e) [Ω·m²] (単位面積あたり)
    電圧則: V(r,t) = V_s(q_a) − V_s(q_b) + L_b dJ_z/dt + R_b J_z

    符号規約は「V > 0 のとき上電極が正 → 上シースが厚くなる (q_a 増)」で一貫させ、
    対称平衡 (V=0) では q_a = q_b = q0 = e n_s s0 (平衡シース厚 s0 = 入力値)。
    初期条件: q_a = q_b = q0、J_z = 0、V = 0 (滑らかに立ち上げるため最初の
    数周期は V0 をランプアップ。理由コメント)。

- シャント ODE は各節点独立 → numpy で全節点一括の半陰的更新
  (L_b·dJ/dt + R_b·J の項は指数積分または後退オイラーで安定化。dt は伝送線路の
  CFL (下記) で決まり十分小さいので精度は足りる。方式と理由をコメント)。

### 数値条件

- 線形化位相速度 v_p = c·√(s_a0 + s_b0) / √(l) → CFL: dt ≤ 0.5·Δr / v_p
  (係数 0.5 の余裕。自動設定、上書き可)。
- 定常化: n_periods (既定 200) 周期回し、最後の n_fft_periods (既定 32、
  f0 の整数周期) を FFT 窓にする。
- 全計算は numpy ベクトル化 (節点ループ禁止)。N=500 節点 × 数十万ステップでも
  数秒〜数十秒で終わる規模。

### 出力

- `harmonics`: n = 0..n_harm (既定 10) の |V_n(r)| [V] (各節点の V(t) を FFT した
  片側振幅)。J_z の高調波 |J_n(r)| も同様に。
- `lambda_eff`: 基本波の実効波長 [m]。推定は |V_1(r)| の理論形 (線形極限で
  Bessel J0(2πr/λ)) への当てはめではなく、**位相勾配から**:
  V_1(r) の複素位相 θ(r) の |dθ/dr| の中央値 → λ = 2π/|dθ/dr|…定在波では位相が
  節で π 跳ぶため、代わりに |V_1(r)| の隣接する節 (極小) 間隔 × 2 を使う
  (節が 1 つも無ければ null)。vacuum 波長 λ0 = c/f0 と短縮率 λ_eff/λ0 も返す。
- `power`: 時間平均吸収電力密度 p(r) = ⟨R_b J_z²⟩ [W/m²] と、その一様性指標
  (max/min、面積重み標準偏差/平均)。
- `v_probe`: 中心付近・R/2・外周の 3 点の V(t) 最終 2 周期 (時系列プレビュー)。
- `spectrum_probe`: 同 3 点の振幅スペクトル (40·f0 まで)。
- warnings: q クリップ (シース崩壊) が起きた場合はその旨 (物理的には正常動作だが
  强非線形の目安になるため)。

## schema (backend/es_sim/schema.py)

```py
class TlSettings(BaseModel):
    """VHF 定在波 (非線形径方向伝送線路、prompts/101)。null なら無効。"""
    radius_m: float = Field(0.15, gt=0)      # 電極半径 R
    gap_m: float = Field(0.04, gt=0)         # ギャップ l
    sheath_m: float = Field(5e-4, gt=0)      # 平衡シース厚 s0 (片側)
    n_e_m3: float = Field(1e16, gt=0)        # バルク電子密度
    n_s_ratio: float = Field(0.4, gt=0, le=1)  # シース端イオン密度比 n_s/n_e (h係数)
    nu_m_hz: float = Field(1e8, ge=0)        # 電子運動量衝突周波数 ν_m
    freq_hz: float = Field(100e6, gt=0)      # 駆動周波数 f0
    v0: float = Field(100.0, gt=0)           # 駆動振幅 [V]
    n_r: int = Field(400, ge=32, le=20000)   # 半径方向節点数
    n_periods: int = Field(200, ge=8)        # 総周期数
    n_fft_periods: int = Field(32, ge=4)     # FFT 窓の周期数 (n_periods より小)
    n_harm: int = Field(10, ge=1, le=40)     # 返す高調波次数
    dt: float | None = Field(None, gt=0)     # None = CFL 自動
class Project: tl: TlSettings | None = None
```

- validator: n_fft_periods < n_periods、sheath_m·2 < gap_m。

## server (backend/es_sim/server.py)

- `/ws/tl`: /ws/dsmc の流儀 (started {n_steps, dt} / progress {step, elapsed_s} /
  done {result} / error、cancel 対応、anyio オフロード)。continue は不要。
- 結果は ResultsBundle にも追加 (`tl` キー。結果付き保存/読込対応)。

## テスト (tests/test_tl.py — 新規)

1. **線形極限の波長短縮**: 非線形をオフにする内部オプション (テスト専用引数
   linear=True: V_s = q·s0/(ε0) の線形容量) で、λ_eff ≈ λ0·√((s_a0+s_b0)/l)
   (rtol 5%。例: f0=100 MHz、l=4 cm、s0=0.4 mm → λ0=3 m、λ_eff≈0.42 m、
   R=0.3 m に節が入る条件を選ぶ)。
2. **高調波生成**: 非線形オンで |V_2|・|V_3| が中心近傍で |V_1| の 0.1% 以上
   (2次非線形なら必ず出る)。linear=True では |V_2|/|V_1| < 1e-6。
3. **エネルギー整合**: ⟨給電電力⟩ ≈ ∫ p(r) 2πr dr (定常で rtol 10%。
   給電電力 = ⟨V(r_feed)·I(r_feed)⟩)。
4. **縮退**: ν_m=0 でも発散しない (数値安定)。validator エラー系。
5. スキーマ: tl 未設定の既存プロジェクト読込不変。

`python -m pytest tests/ -q` 全件パス (現在 273 + 新規)。

## frontend

### スタディ「VHF定在波」

- ProjectTree: スタディに「VHF定在波」(PIC-MCC 1D の下)、結果に「VHF定在波」。
- **TlPanel.tsx** (新規): 設定一式 (radius/gap/sheath は表示単位変換、
  CommitNumberInput 系)。「1D PIC 結果から取込」ボタン: 直近の pic1dResult が
  あれば n_e ← 中央の n_i (バルク代表として n_i(gap/2)、コメントに理由)、
  sheath_m ← (左右 s の平均)、無ければ disabled + hint。ν_m は手入力のまま
  (hint: 「ν_m ≈ n_g·K_m(T_e)。MCC 断面積からの自動推定は未対応」)。
  実行/停止ボタン、進捗はステータスバー統合 (anyRunning・dismissStatusError)。
- **TlPlotView.tsx** (新規、キャンバス領域): Plot1dView の部品/流儀を流用。
  - |V_n(r)| チャート: n=1..n_harm を色分け (表示する次数のトグルまたは上位数本
    + 全表示トグル)。対数軸トグル。
  - λ_eff・短縮率 λ_eff/λ0・λ0 のサマリ行。電力一様性指標も。
  - p(r) (吸収電力密度) チャート。
  - プローブ点の V(t) 波形と振幅スペクトル (3点色分け)。
- 結果付き保存 (ResultsBundle.tl) と applyLoadedProject 復元。
- types.ts: TlSettings / TlResult / WS メッセージ型。

### 検証

- backend: `python -m pytest tests/ -q` 全件パス。
- frontend: `cd frontend && npx tsc --noEmit && npx vite build`。

## 注意

- コメントは日本語で「なぜ」(モデルの式・出典・符号規約・CFL・r_feed の理由)。
- git commit はしない。既存機能への影響ゼロ (Project への optional 追加のみ)。
- Math.min/max スプレッド禁止。単位表示は lengthUnit 追従 (radius/gap/sheath/λ)。
- 波長・周波数など物理量の表示は formatNumber。
