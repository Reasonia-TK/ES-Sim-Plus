# 109: 流体モデル Phase D2 — Fluid1dPanel 本体・表示・PIC 比較・App 統合

## 背景

prompts/104 計画の Phase D2。Phase D1 (雛形: types/fluid1dClient/ProjectTree/
Fluid1dPanel 骨組み、実装済み) を完成させ、App.tsx に統合する。
**1D PIC との比較オーバーレイが本機能の売り**。frontend のみ。

現状: App.tsx に NODE_TITLES の "study-fluid1d"/"result-fluid1d" キー不足の
tsc エラーが1件残っている (D1 が意図的に残した統合ポイント)。

## 作業内容

### 1. Fluid1dPanel の完成 (panels/Fluid1dPanel.tsx)

D1 の placeholder を実装:

- **電極 (左/右)**: Pic1dPanel の ElectrodeEditor を流用 (export されていなければ
  export 化 or 同等実装)。v_dc / RF重畳 (RfComponentsEditor) / CSV 波形 / SEE γ。
  **fn (FN放出) は流体では未対応**なので FN トグルは出さない (Pic1dElectrode 共用
  だが validator でエラーになる旨、共通部品側の表示制御で対応)。
- **断面積 (electron_processes)**: Pic1dPanel の MCC 断面積編集 (ProcessList +
  lxcat インポート) の電子側のみを流用。空 = eduPIC Ar 解析式を使用という
  hint を表示。
- **プリセット**: 「1D PIC の設定を取込」ボタン — 現在の pic1d 設定から
  gap_m / left / right (fn は除去) / init_density_m3 / init_te_ev /
  ion_mass_amu / mcc.gas.pressure_pa → gas_pressure_pa / temperature /
  electron_processes / phase_bins を写す (流体固有パラメータは維持)。
  pic1d 未設定なら disabled。
- **実行設定**: dt (null=自動)、n_steps、frame_every、avg_steps、phase_bins、
  RF サイクル換算ヒント (Pic1dPanel の流儀)。
- 「続きから (+Nステップ)」「停止」— pic1d と同じ。

### 2. 表示 (canvas/Fluid1dPlotView.tsx — 新規)

Plot1dView.tsx の部品 (Pic1dLineChart / cycle player / サマリの流儀) を流用:

- **ライブ**: φ(x)・n_e/n_i(x) (対数トグル)・T_e(x) チャート + 進捗/経過秒。
  RF 波形モニタ (Pic1dRfMonitor 流用、左電極)。
- **結果**: フィールド選択 (φ/E/n_e/n_i/重ね/T_e/電離レート) + 対数トグル、
  シースエッジマーカー (sheath、右側は gap−s 変換 — prompts/99 の規約)、
  位相アニメ (cycle: φ/n_e/n_i/T_e)、history チャート (総粒子数・壁損失)、
  数値サマリ (elapsed_s、timing、中心 n_e/T_e、シースエッジ、壁フラックス)。
- **PIC 比較オーバーレイ (最重要)**: 結果ビューに「1D PIC と比較」Toggle。
  オンのとき、pic1dResult (App から渡す) の profiles を**同じチャートに破線で
  重ね描き** (凡例「流体」実線 / 「PIC」破線、同色系の明暗差)。対象フィールド:
  φ/E/n_e/n_i/T_e/電離レート (PIC 側にあるもの)。x 格子が違っても各系列を
  そのまま描けば良い (同一チャートの2系列)。pic1dResult が無ければ Toggle を
  disabled + hint。
  さらに比較サマリ行: 中心 n_e 比 (流体/PIC)、中心 T_e 差、シースエッジ位置差。

### 3. App.tsx 統合 (最小差分)

pic1d の統合パターンの複製:
- NODE_TITLES に2キー追加 (tsc エラー解消)、selection ルーティング
  (study-fluid1d / result-fluid1d → Fluid1dPanel + Fluid1dPlotView)。
- state: fluid1d 設定 (DEFAULT_FLUID1D 初期値)・fluid1dRunning/Started/Frame/
  Result/Error/ContinueReady、Fluid1dClient 配線 (makePic1dCallbacks の流儀)。
- anyRunning・dismissStatusError・ステータスバー分岐に fluid1d を追加。
- 結果付き保存: buildResults / applyLoadedProject / hasAnyResults に
  fluid1d を追加 (ResultsBundle.fluid1d は D1 で型定義済み)。
- ProjectTree への props (D1 で optional 定義済み) を配線。
- Fluid1dPlotView に pic1dResult を渡す (比較用)。
- スイープ: SweepPanel のプリセットパスに fluid1d 系
  (fluid1d.gap_m / init_density_m3 / gas_pressure_pa / left.v_dc /
  left.voltage_rf.0.amplitude 等、pic1d の流儀) を追加し、
  sweepModuleForPath が "fluid1d." → "fluid1d" を返すよう拡張
  (バックエンドは対応済み)。projectForSweep に fluid1d を含める。

## 検証

`cd /home/claude/ES-Sim/frontend && npx tsc --noEmit && npx vite build`
(エラー0で通ること)。backend は触らない。

## 注意

- コメントは日本語で「なぜ」。git commit はしない。
- Pic1dPanel から部品を流用する際、pic1d 側の挙動を変えない (export 追加のみ可)。
- Math.min/max スプレッド禁止 (arrayMin/arrayMax)、CommitNumberInput 系、
  import type ブロックに割り込まない。App.tsx は最小差分・既存パターン模倣。
- 単位表示は lengthUnit 追従、値は formatNumber。
