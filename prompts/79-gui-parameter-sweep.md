# 79: GUI パラメータスイープ (1パラメータ × 値リスト)

## 背景 (ユーザー要望・設計確認済み)

- GUI からパラメータスイープしたい。
- 対象指定は「プリセット選択」と「任意フィールドのパス指定」の**両方**。
- 結果は**アプリ内で一覧→クリックで既存の結果ページに読込**んで閲覧。
- 実行はバッチ機構 (prompts/78) を流用し、ケースごとに別プロセス並列。

## 全体構造

- backend はパス指定のみの汎用実装 (プリセットは frontend が現在のプロジェクトから
  候補パスを生成する糖衣)。
- 1パラメータ × 値リスト (1D)。2Dグリッドは対象外。

## backend

### es_sim/sweep.py (新規)

- `set_by_path(obj: dict, path: str, value: float) -> None`:
  ドット区切りパス (配列インデックスは数字トークン、例
  `geometry.boundaries.1.voltage`、`pic.n_macro`、`b_field.bz`、
  `geometry.regions.0.voltage_rf.0.amp`) で project dict の数値フィールドを上書き。
  存在しないパス・数値でない終端はエラー文字列を返す/例外。
- ケース生成: ベース project (dict) を deepcopy し、パスへ値を適用 → N ケース。
- 実行: prompts/78 の batch と同じ multiprocessing (spawn) ワーカを流用
  (batch.py の子プロセス実行関数を共通化して import)。並列数 parallel。
- 結果はサーバーの一時ディレクトリ (tempfile.mkdtemp、サーバープロセス生存中のみ)
  に「結果付き保存」JSON として書き、メモリには持たない (ケース数×cycle で
  巨大化するため)。スイープ開始ごとに前回の一時ディレクトリは破棄。

### server.py

- `WS /ws/sweep`:
  - `{cmd:"start", project, param_path, values:[...], parallel}` →
    `{type:"started", n_cases, param_path, values}` →
    ケースごとの進捗 `{type:"progress", case:i, step, n_steps}` (数百ms間引き) と
    `{type:"case_done", case:i, ok, error?}` →
    全部終わったら `{type:"done", summary:[{case,value,ok,error?}]}`。
  - `{cmd:"stop"}` で実行中プロセスを terminate し、完了済みケースは保持。
  - PIC/DSMC の実行ロックとは独立で良いが、スイープの同時複数実行は拒否。
- `GET /sweep/result/{i}` → ケース i の結果付き JSON をそのまま返す
  (未完了/失敗は 404)。
- `use_dsmc_gas` ケースは batch 同様に開始時点でエラーにする。

### テスト (backend/tests/test_sweep.py)

1. set_by_path の単体 (正常系: ネスト/配列/RF成分、異常系: 不在パス・非数値終端)。
2. WS スイープ: 小ケース × 2値 (例: 電極電圧 50/100V) → started/progress/
   case_done×2/done が流れ、GET /sweep/result/0,1 が有効な結果付きJSONを返し、
   スイープ値が各ケースの project に反映されていること。
3. stop: 開始直後に stop してエラーなく閉じること。

`python -m pytest tests/ -q` 全件パス (現在 164)。

## frontend

### ツリー / ページ

- スタディセクションに「スイープ」ノード (study-sweep) を追加 (DSMC の下)。
  状態バッジ: 実行中 k/N / ✓完了 / エラー / 未実行。
- インスペクタページ (App.tsx 直書きでも SweepPanel.tsx 新設でも可、新設推奨):

  1. **対象パラメータ**:
     - select (プリセット): 現在の project から動的生成 —
       Dirichlet 辺ごとの「〜辺 電圧」、RF成分ごとの「〜辺 RF振幅/RF周波数」、
       conductor 領域ごとの「領域〜 電圧」、b_field があれば Bx/By/Bz、
       dsmc があれば「初期圧力」、PIC の「マクロ粒子数」「dt」。
       各候補は {label, path} で持つ。
     - 「カスタムパス」選択肢を選ぶと CommitTextInput でパスを直接入力
       (hint にパス例を表示)。
  2. **値リスト**: モード切替 (リスト / 範囲)。
     - リスト: カンマ区切りテキスト (指数表記可) → parse。
     - 範囲: 開始・終了・点数 (線形等分)。対数等分のトグルもあると良い (任意)。
     - 現在の値 (パスから読んだ値) を参考表示。
  3. **並列数** (既定1) + 「合計スレッド数 = 並列数 × PICスレッド数 ≦ コア数」hint。
  4. **実行 / 中止** ボタン (backend 未接続時 disabled)。
  5. **ケース一覧**: 行 = ケース番号・パラメータ値・状態 (待機/実行中 xx%/✓完了/
     エラー)。完了行をクリック → `GET /sweep/result/i` を fetch し、
     **loadProject と同じ復元処理**でプロジェクト+結果を読み込む
     (App.tsx の loadProject の復元部を `applyLoadedProject(obj)` として関数抽出し、
     ファイル読込とケース読込で共用する)。読込されたことが分かるよう
     行をハイライト。
     注意: ケース読込は commitProject 経由なので Undo で戻れる。

### 通信

- sweepClient.ts (新規、dsmcClient と同じ流儀の WS ラッパ)。
- 型: types.ts に SweepStartCmd / SweepServerMessage 等。

### ステータスバー

- スイープ実行中は「スイープ実行中... k/N ケース完了」を表示
  (優先順位は PIC/DSMC 単発実行と同列でどちらか動いている方)。

## 検証

- backend: pytest 全件。
- frontend: `npx tsc --noEmit && npx vite build`。
- 統合: uvicorn を実起動し、websockets クライアントで sweep start→done→
  result 取得の一連を実確認 (終了後プロセス停止・一時スクリプト削除)。

## 注意

- コメントは日本語で「なぜ」を書く既存スタイル。git commit はしない。
- batch.py との共通化でリグレッションを出さない (batch のテストも全パスのこと)。
