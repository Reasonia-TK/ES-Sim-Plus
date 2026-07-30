# 96: パラメータスイープ / バッチ実行の 1D PIC/MCC 対応

## 背景 (ユーザー要望)

「1Dの方もパラメトリック計算できるのかな？」
GUI パラメータスイープ (prompts/79) とバッチ CLI (prompts/78) のケース実行
(batch.py `_worker`) は 2D PIC 専用 (project.pic 必須)。set_by_path は汎用なので
`pic1d.*` パスは既に通る。ケース実行に 1D 分岐を追加し、GUI からも 1D の
スイープを回せるようにする。

## モジュール判定の仕様 (backend/frontend 共通)

**ケースの project に対しどちらのソルバーを走らせるかは明示指定 + 自動判定**:

- `module: "pic" | "pic1d" | None` を導入。None (未指定) は自動判定:
  `pic1d.` で始まる対象パス → "pic1d"、それ以外 → "pic" (従来互換)。
  自動判定はスイープのリクエスト側 (server の ws_sweep / sweep.run_sweep) で
  解決し、_worker には確定した module を渡す。
- バッチ CLI は `--module {auto,pic,pic1d}` (既定 auto: project に pic があれば
  pic、無ければ pic1d。両方無ければエラー)。

## backend

### batch.py

- `_worker(case_path, out_path, case_name, progress_q, module="pic")` に拡張
  (既定値により既存呼び出し・既存挙動は不変)。
- module="pic1d" のとき: `Pic1dSimulation(project)` を実行 (progress は 2D と同じ
  形式で report)、結果バンドルは `{"pic1d": <done result>}` —
  server.py `_pic1d_result()` と**同一の形** (frontend ResultsBundle.pic1d /
  Pic1dResult と一致)。共通化できるなら `_pic1d_result` 相当のビルダーを
  server.py から batch.py (または pic1d.py) へ移して両方から使う
  (循環 import 回避のため置き場所は pic1d.py 側が無難)。
- use_dsmc_gas は 1D では常に不可 (schema validator 済みなので追加対応不要)。
- 検証系 (pic1d.dt=None の自動 dt など) はそのまま動くこと。

### sweep.py / server.py (ws_sweep)

- run_sweep に module を渡す (リクエスト JSON に optional `module`。
  未指定は上記の自動判定)。started メッセージに解決済み module を含める
  (フロント表示用)。
- GET /sweep/result/{i} は形式不変 (bundle に pic1d キーが入るだけ)。

### テスト (tests/test_sweep.py または test_pic1d.py に追加)

1. `set_by_path` で `pic1d.gap_m` / `pic1d.left.v_dc` が上書きできる (既存機能の
   確認、失敗パスのエラーも)。
2. 1D スイープ 2ケース (小規模: n_cells 16、n_macro 500、n_steps 100、MCC なし、
   値リスト = init_density 2種) を run_sweep で実行し、両ケースの bundle に
   pic1d 結果があり、上書きした値が bundle 内 project に反映されていること。
3. 自動判定: パス "pic1d.n_steps" → module "pic1d"、"pic.n_steps" → "pic"。
4. 既存の 2D スイープテストが不変で通ること。

`python -m pytest tests/ -q` 全件パス (現在 247 + 新規)。

## frontend

### SweepPanel.tsx

- プリセットパス生成に 1D を追加 (project.pic1d があるとき):
  - `pic1d.gap_m` (ギャップ長 [m])
  - `pic1d.init_density_m3` (初期密度)
  - `pic1d.n_macro`
  - `pic1d.mcc.gas.pressure_pa` (mcc があるとき)
  - `pic1d.left.v_dc` / `pic1d.right.v_dc`
  - `pic1d.left.voltage_rf...amplitude` / freq_hz (voltage_rf があるとき。
    単一/リストの正規化: 既存の ensureSweepPath の「不足キーの実体化」の流儀で、
    スイープ開始前に voltage_rf を**リスト形式へ正規化**してから
    `pic1d.left.voltage_rf.0.amplitude` 形式のパスを使う)
  - `pic1d.left.fn.beta` / `pic1d.left.fn.phi_ev` (fn があるとき。右も同様)
  - ラベルは日本語 (例: 「1D: ギャップ長 [m]」「1D: 左電極 RF振幅 [V]」)。
- モジュール表示: 選択パスが `pic1d.` 始まりのとき「対象: PIC-MCC 1D」の hint。
  リクエストに module を明示して送る (自動判定に頼らず、UI 確定値を送る)。
- スイープ実行の前提条件 (pic 設定必須のガードがあれば) を「pic または pic1d が
  あれば可」に緩和。
- ケース結果の閲覧 (ケースクリック → GET /sweep/result/i → applyLoadedProject) は
  既存機構のままで pic1d 結果が復元されるはず — App.tsx の applyLoadedProject が
  results.pic1d を復元することを確認し、必要なら sweep ケース読込経路にも同じ
  復元を通す。ケースを開いたとき result-pic1d ノードが活性になること。

### 検証

`cd /home/claude/ES-Sim/frontend && npx tsc --noEmit && npx vite build`。

## 注意

- コメントは日本語で「なぜ」。git commit はしない。
- 既存 2D スイープ・バッチの挙動はビット不変 (module 既定値で担保)。
- multiprocessing spawn (Windows) 前提の _worker のピクル制約に注意
  (トップレベル関数のまま拡張する)。
