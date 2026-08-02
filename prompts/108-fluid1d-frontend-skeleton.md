# 108: 流体モデル Phase D1 — frontend 雛形 (パターン複製、難易度: 低)

## 背景

prompts/104 計画の Phase D1。**frontend のみ**。pic1d の frontend 配線を手本に、
流体 (1D) スタディの型・WS クライアント・ツリーノード・パネル骨組みを作る。
パネルの中身の作り込みと表示・PIC 比較は Phase D2 (prompts/109) が行うので、
ここでは**ビルドが通る最小の骨組み**まで。App.tsx は触らない (D2 が統合する。
ここで触ると D2 と競合するため)。

## バックエンド仕様 (実装済み、これに合わせる)

- schema: `Fluid1dSettings` (backend/es_sim/schema.py を読んで正確に写す。
  gap_m, n_cells, left/right (Pic1dElectrode 流用・fn 不可), init_density_m3,
  init_te_ev, gas_pressure_pa, gas_temperature_k, ion_mass_amu, mu_i_ref,
  n_ref_m3, t_i_ev, electron_processes, dt, n_steps, frame_every, avg_steps,
  phase_bins)。`Project.fluid1d?: Fluid1dSettings | null`。
- /ws/fluid1d (backend/es_sim/server.py を読んで正確に):
  - start/continue/stop、started {n_steps, step_offset, dt, x, warnings}、
    frame {step, t, phi[], n_e[], n_i[], t_e[], counts, elapsed_s}、
    done {result: Fluid1dResult}、error {detail}
- Fluid1dResult (backend/es_sim/fluid1d.py の build_fluid1d_result を読んで正確に):
  history {step,t,n_e_total,n_i_total,wall_left_e,wall_left_i,wall_right_e,
  wall_right_i,gen_total}、profiles {x,phi,e,n_e,n_i,t_e,ionization,avg_steps}|null、
  sheath {left_s,right_s}|null、cycle {bins,freq_hz,phi,n_e,n_i,t_e}|null、
  walls、gen_total、elapsed_s、timing {poisson,transport,energy,other,total}、
  settings。

## 作業内容 (frontend/src)

1. **types.ts**: `Fluid1dSettings`、`Fluid1dResult` 系一式、WS メッセージ型
   (`Fluid1dStartedMsg` / `Fluid1dFrameMsg` / `Fluid1dDoneMsg` / `Fluid1dErrorMsg` /
   `Fluid1dServerMessage` / `Fluid1dClientCommand`)、`Project.fluid1d?`、
   `ResultsBundle.fluid1d?: Fluid1dResult`。pic1d の型定義の並びに置く。
2. **fluid1dClient.ts** (新規): pic1dClient.ts の忠実な複製 (エンドポイント
   /ws/fluid1d、メッセージ型差し替え)。
3. **ProjectTree.tsx**: スタディ「流体 (1D)」(study-fluid1d) と結果 (result-fluid1d)
   のノード追加 — pic1d の行を手本に props・バッジ含め同構造。
4. **panels/Fluid1dPanel.tsx** (新規・骨組みのみ): Props 型 (Pic1dPanel の Props を
   手本に onChange/canRun/running/onStart/onStop/canContinue/onContinue/started/
   frame/error + lengthUnit)、中身は「形状 (gap/n_cells)」「初期値」「ガス」
   「実行設定」「実行ボタン群」の最小セクション (CommitNumberInput 系で全
   フィールド編集可能に)。電極 (RF/CSV/SEE) 編集・断面積・プリセットは D2 で
   追加するので placeholder コメントを残す。
5. `DEFAULT_FLUID1D` 定数 (App が使う既定値。gap 0.025、density 1e15、50 Pa 等
   バックエンド既定と整合) を Fluid1dPanel.tsx から export。

**App.tsx への統合はしない** (D2 の担当)。未使用 export の TS エラーが出ない
ことを確認 (tsc は未参照ファイルも型検査する)。

## 検証

`cd /home/claude/ES-Sim/frontend && npx tsc --noEmit && npx vite build`
(App 未統合でもビルドが通ること)。backend は触らない。

## 注意

- コメントは日本語で「なぜ」。git commit はしない。
- 型はバックエンドの実装を読んで正確に (推測で書かない)。
- Math.min/max スプレッド禁止、import type ブロックに割り込まない。
