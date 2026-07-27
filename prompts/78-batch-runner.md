# 78: PIC バッチ実行モード (複数プロセス並列)

## 背景 (ユーザー要望)

「複数同時にプロセスを起動してPIC計算を複数同時に回すことはできるかな？」
GUI は1アプリ=1計算 (サーバーがロックで同時 start を拒否) なので、
パラメータスイープ向けに**ヘッドレスのバッチ実行モード**を追加する:
複数のプロジェクト JSON を渡すと別プロセスで並列実行し、それぞれ
**「結果付き保存」形式の JSON** を書き出す。結果は GUI の「読込」でそのまま開ける。

## CLI 仕様 (backend/es_sim/batch.py 新規)

```
python -m es_sim.batch run case1.json case2.json ... [オプション]
  --parallel N     同時実行プロセス数 (既定: 1)
  --out DIR        出力先ディレクトリ (既定: 入力と同じ場所)
  --suffix S       出力ファイル名サフィックス (既定 "_results" → case1_results.json)
```

- 入力: GUI の「保存」形式のプロジェクト JSON (pic 設定を含む。particles.emitter は
  保存時に pic.injection.emitter へ合成済みのはず — 未合成ファイルにも対応するため、
  App.tsx の withInjectionEmitter と同じ合成をロード時に行う)。
- 各ケースを **multiprocessing (spawn) の別プロセス**で実行 (GIL・numba スレッドの
  独立性のため)。プロセス内で PicSimulation を構築し run_batch を最後まで実行。
- 進捗: 親プロセスで「[case1] 1200/8000 (15%)」のような行を数秒おきに表示
  (子→親は Queue か、簡単のため子が直接 print + flush でも良い。行頭にケース名)。
- 完了時: 各ケースの結果を「結果付き保存」JSON として書き出す。トップレベルは
  入力プロジェクト + `results` キー:
  ```
  results = {
    "version": 1,
    "pic": {
      "started": {"type":"started","dt":…,"n_steps":…,"step_offset":0,
                   "warnings":[…],"mesh":{"nodes":…,"triangles":…}},
      "frame": null,
      "history": [PicDiag行の配列],   ← frontend の toDiagArray 後の形 (行ごとの dict)
      "fields": …, "cycle": …, "collectors": […]
    }
  }
  ```
  **形式は frontend/src/types.ts の ResultsBundle / App.tsx の loadProject 復元処理と
  厳密に一致させること** (server.py の done メッセージ組み立てと同じ変換を流用する。
  history は server が送る「列ごとの辞書」ではなく「行ごとの配列」なので注意 —
  frontend の toDiagArray 相当の変換を batch 側で行う)。
- サマリ: 全ケース終了後に 成功/失敗・実行時間・出力パスの表を表示。
  1件でも失敗したら exit code 1。

## 検証ポイント: GUI で読めること

書き出した JSON を frontend の loadProject が受け付ける形式かをテストで担保する:
- results.pic.started に dt/n_steps/mesh (nodes/triangles) が存在
- history が配列で各行に t/ke_e/… のキー
- fields のキー (phi/e_abs/…) が PicFields と一致
実装時に frontend/src/types.ts の ResultsBundle・PicStartedMsg・PicDiag・PicFields を
読み、キー名を写して assert するテストを書く。

## 配布版対応 (run_server.py)

- backend/run_server.py (PyInstaller エントリ) に argparse のサブコマンドを追加:
  - 引数なし / `--port N` → 従来どおりサーバー起動 (完全後方互換。Tauri サイドカーの
    呼び出し方は変えない)
  - `batch <files...> [--parallel N] [--out DIR]` → es_sim.batch を実行
  - **`multiprocessing.freeze_support()` をエントリ先頭で必ず呼ぶ**
    (Windows の PyInstaller exe で multiprocessing spawn が再帰起動するのを防ぐ。
    コメントに理由を書く)。子プロセスの起動は sys.frozen 環境でも動くよう
    multiprocessing の標準 API のみ使う。
- これで配布版ユーザーも `es-sim-backend.exe batch a.json b.json --parallel 2` で
  スイープできる。README にバッチ実行の節を追加 (使い方3行程度+GUIで読める旨)。

## リソースの注意 (README とヘルプ文に記載)

- 合計スレッド数 = parallel × pic.threads がCPUコア数を超えないように、の注意書き。

## テスト (backend/tests/test_batch.py 新規)

1. 小さな2ケース (examples の流用か合成) を --parallel 2 で実行し、
   出力2ファイルが生成され、上記のキー構造チェックに合格すること。
2. 出力 JSON を再度 pydantic の Project として読めること (results キーは
   Project スキーマ外なので、除去してから validate する — loadProject と同じ分離)。
3. 失敗ケース (壊れた JSON) を混ぜたとき exit code 非0 で、正常ケースの出力は出ること。

`python -m pytest tests/ -q` 全件パス (現在 160)。

## 検証

- backend: pytest 全件。frontend 変更は無し (確認のみ)。
- 実際に `python -m es_sim.batch run` を2ケース並列で実行し、壁時計時間が
  逐次実行より短いことを確認して報告。

## 注意

- コメントは日本語で「なぜ」を書く既存スタイル。git commit はしない。
- server.py の既存挙動 (ロック等) は変えない。
