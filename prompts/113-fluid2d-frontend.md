# 113: 2D 流体 Phase C — frontend (スタディ「流体 (2D)」)

## 背景

prompts/111-112 (fluid2d コア + /ws/fluid2d、実装済み) のフロントエンド。
2D 流体はジオメトリ・メッシュ・境界条件 (電圧/RF/CSV/SEE) を既存プロジェクト
設定と共有するため、**キャンバスは CadCanvas のまま** (1D のような専用ビューは
作らない)。フィールド表示は既存の汎用機構 (picFieldView / picFrame / 配色・
レンジ・プローブ・シースエッジ等値線) に載せる。

## /ws/fluid2d プロトコル (実装済み — server.py を読んで正確に)

- start/continue/stop。started {n_steps, step_offset, dt, warnings} (mesh 無し —
  既存の /mesh 結果とメッシュ整合)。
- frame {step, t, phi[], n_e[], n_i[], t_e[] (全節点), counts, elapsed_s}
- done {result}: fields {phi, e_abs (要素), n_e, n_i, t_e, ionization,
  avg_steps} | null、cycle {bins, freq_hz, phi/n_e/n_i/t_e (bins×nodes)} | null、
  history、walls {electron, ion}、gen_total、elapsed_s、timing、settings。

## 作業内容 (frontend のみ)

1. **types.ts**: Fluid2dSettings / Fluid2dResult / WS メッセージ型 /
   Project.fluid2d? / ResultsBundle.fluid2d? (backend schema.py・fluid2d.py を
   読んで正確に)。
2. **fluid2dClient.ts** (新規): fluid1dClient の複製 (/ws/fluid2d)。
3. **ProjectTree**: スタディ「流体 (2D)」(study-fluid2d、流体 (1D) の下) と
   結果ノード (result-fluid2d)。
4. **panels/Fluid2dPanel.tsx** (新規): Fluid1dPanel を手本に:
   - hint: 「ジオメトリ・メッシュ・電極電圧 (RF/CSV/SEE) は既存のプロジェクト
     設定 (境界条件) を使います」
   - 初期値 (density/Te)、ガス (圧力/温度/イオン質量/μ_i/T_i)、
     電子断面積 (ProcessList + lxcat、空 = eduPIC Ar)、実行設定
     (dt/n_steps/frame_every/avg_steps/phase_bins + RF サイクル換算は
     プロジェクトの RF 周波数から)、実行/続きから/停止。
   - 「流体 (1D) の設定を取込」ボタン (fluid1d 設定からガス・初期値・断面積を
     写す。無ければ disabled)。
5. **App.tsx 統合** (fluid1d/pic の統合パターンの複製):
   - state 一式 + Fluid2dClient 配線、anyRunning・ステータスバー・
     dismissStatusError。
   - **結果表示**: result-fluid2d 選択時、既存の PIC 結果表示機構
     (picFieldView 相当の「フィールド選択 select + CadCanvas 節点フィールド
     描画」) に流体の fields を載せる。App の picFieldView の組み立てを読み、
     流体用の同等 state (fluid2dFieldView) を追加して CadCanvas へ渡す
     (CadCanvas 側は汎用 nodeBased/log/unit 対応済みのはず — 必要最小の
     プロパティ追加のみ可)。フィールド: 電位 φ / |E| / n_e / n_i / T_e /
     電離レート (+ 対数トグル、密度系)。
   - **ライブ**: 実行中は frame の phi/n_e/n_i/t_e を picFrame と同じ機構で
     表示 (フィールド選択 select)。
   - **位相アニメ**: cycle があれば既存 2D PIC の位相アニメ UI の流儀で
     再生 (可能なら既存部品を流用。困難なら第1弾はスライダのみでも可 —
     判断をレポート)。
   - シースエッジ等値線 (準中性度 α)・プローブ・配色・レンジは汎用機構なので
     流体フィールドでも自動で効くことを確認 (効かなければ最小配線)。
   - 結果付き保存 buildResults/applyLoadedProject/hasAnyResults に fluid2d。
   - スイープ: SweepPanel プリセットに fluid2d.init_density_m3 /
     gas_pressure_pa / (geometry.boundaries 系の電圧パスは既存を流用)、
     sweepModuleForPath "fluid2d."。projectForSweep に fluid2d。
6. **数値サマリ**: 結果パネル (Fluid2dPanel の結果セクション or 既存の流儀) に
   elapsed_s・timing 内訳・壁損失・中心付近の n_e/T_e (メッシュ重心最近傍
   でよい)・gen_total。

## 検証

`cd /home/claude/ES-Sim/frontend && npx tsc --noEmit && npx vite build` エラー0。
backend は触らない。

## 注意

- コメントは日本語で「なぜ」。git commit はしない。App.tsx は最小差分。
- CadCanvas のフィールド表示は既存汎用機構を流用 (新しい描画経路を発明しない)。
- arrayMin/arrayMax、CommitNumberInput 系、import type ブロックに割り込まない。
- 実行前にメッシュ未生成なら 2D PIC と同じ挙動 (自動生成 or ガイダンス —
  既存 PIC 開始フローを読んで揃える)。
