# 74: DSMC の続きから実行 (continue)

## 背景 (ユーザー要望)

「DSMCもPICのように続きから計算を進めることはできますか？」
PIC の continue (prompts/32) と同じ構造を DSMC にも実装する。定常統計が
足りないときに、過渡を再計算せず追加ステップ+平均取り直しができるようにする。

## backend

### es_sim/dsmc.py

- `prepare_continue(n_steps: int, avg_steps: int | None = None) -> None` を追加:
  - 維持: 粒子状態 (x, v, elem)・rng・_coll_frac・_sigcr_max・リザーバ/流入状態・
    通算ステップ数 (新設 `self.step_count` — 現状 run() ローカルの i なら
    インスタンス変数に昇格させる)・inflow/outflow は平均区間にリセットが絡むなら
    現行 run() の扱いに合わせる (実装を読んで一貫させる)。
  - リセット: サンプリング蓄積 (_acc_cnt/_acc_v/_acc_v2/_samples) と
    平均開始ステップの再計算 (追加区間の n_steps/avg_steps 基準)。
  - PIC と同様、「N+M 連続」と「N → continue M」の**粒子状態がビット単位で一致**
    するように、乱数や端数持ち越し (_coll_frac) を一切リセットしない。
- `run()` が通算 step_count を使うよう調整 (progress callback の step は
  **区間内ステップ** (0..n_steps) を送る — DSMC の進捗表示は区間相対のままにして
  PIC で起きた進捗ずれ (prompts 直近の fix) を最初から避ける)。

### es_sim/server.py

- PIC と同様に `_last_dsmc_sim` (実行後のシミュレーションオブジェクト) を保持。
- ws_dsmc に `continue` コマンドを追加: `{cmd:"continue", n_steps, avg_steps}`。
  保持中の sim が無ければ error。prepare_continue → 既存のストリーム実行を流用し、
  started → progress → done を送出 (started に step_offset 等は不要。progress の
  step は区間内)。done の結果は _store_dsmc_result で上書き (PIC 連成にも新しい
  平均が使われる)。
- start 実行時は _last_dsmc_sim を新しい sim で置き換える。
- 同時実行ロックは PIC 側の流儀 (実行中は拒否) に合わせる。

### テスト (backend/tests/test_dsmc.py に追加)

1. **一致性**: 同一シードで (a) n_steps=400, avg=100 を1回 → continue 400/avg=100、
   (b) n_steps=800, avg=100 を1回。(a) の continue 後の粒子数・結果フィールドが
   (b) と**完全一致** (乱数連続性の担保。avg 区間が同じ末尾100ステップになるよう
   条件を選ぶ)。
2. **WS continue**: ws_dsmc で start → done → continue → done がエラーなく流れ、
   2回目の done の結果が有限で、サンプル数が追加区間の avg_steps に対応すること。

`python -m pytest tests/ -q` 全件パス (現在 150)。

## frontend

### dsmcClient.ts

- `continueRun(params: { n_steps: number; avg_steps: number | null })` を追加
  (PicClient.continueRun と同じ流儀。既存接続で continue コマンド送信)。
- コールバック差し替え用 `setCallbacks` が無ければ追加 (PicClient 参照)。

### App.tsx

- state: `gasContinueReady` / `gasProjectChangedSinceRun` (PIC の
  picContinueReady / picProjectChangedSinceRun と同じ役割)。
- **無効化条件の注意**: DSMC 設定は project.dsmc 内にあるため、単純に
  commitProject 全部で無効化すると「ステップ数を増やして続きから」ができない。
  commitProject 内で prev と next を比較し、**dsmc の n_steps / avg_steps /
  threads / smoothing_passes 以外** (geometry / mesh / coord / dsmc.gas /
  dsmc.boundaries / dsmc.init_* / wall_temperature_k / n_particles / dt / seed)
  が変わった場合のみ gasProjectChangedSinceRun = true にする。
  比較ヘルパは JSON.stringify で除外フィールドを落として比較する程度の
  簡潔な実装で良い (コメントに理由を書く)。
- `runDsmcContinue()`: dsmcClientRef.current.continueRun({ n_steps, avg_steps })。
  開始時に gasResult/gasProgress をリセットし進捗表示は新区間で 0 から。
- done で gasContinueReady = true、エラー時 false (PIC と同じ)。

### GasPanel.tsx

- 実行ボタンの隣に「続きから実行」ボタン (canContinue prop)。PIC と同じ
  説明文の hint (「現在のステップ数・平均ステップ数で追加実行します。粒子状態・
  乱数は前回から継続。ジオメトリやガス条件を変更した場合は再実行が必要です」)。
  無効理由の表示も PicPanel の continueDisabledByProjectChange に倣う。

## 検証

- backend: pytest 全件。frontend: `npx tsc --noEmit && npx vite build`。

## 注意

- コメントは日本語で「なぜ」を書く既存スタイル (ビット一致の条件、無効化条件の
  除外リストの理由を書く)。
- git commit はしない。
