# 86: 計算経過時間のリアルタイム表示 + 結果表示

## 背景 (ユーザー要望)

「計算に要した時間をリアルタイム表示+結果に表示してほしい」

## 仕様

### リアルタイム表示 (frontend)

- 下部ステータスバーの実行中表示に経過時間を追加:
  - PIC: 「PIC-MCC 実行中... 62% (.../...) — 経過 1:23」
  - DSMC: 「DSMC 実行中... 45% — 経過 0:41」
  - スイープ: 「スイープ実行中... 2/5 ケース完了 — 経過 3:05」
  - FEM/トレース (busy): 「静電場/トレース 計算中... — 経過 0:07」
- 実装: 実行開始時刻を state に記録し、実行中のみ 1秒間隔の setInterval で
  再レンダー (経過秒 state を tick)。表示フォーマットは m:ss (1時間超は h:mm:ss)。
  複数同時実行はそれぞれの開始時刻で個別に計算。

### 結果への表示

- **PIC**: server の done メッセージに `elapsed_s` (run_batch の壁時計秒、
  continue は区間分) を追加。PicPanel の「実行時間内訳」セクションの先頭に
  「経過時間 (壁時計)」kv を追加 (内訳 total との差 = フレーム送信等の
  オーバーヘッドも見える)。
- **DSMC**: `DsmcResult` に `elapsed_s: float` フィールドを追加 (run() 内で計測)。
  GasPanel の結果サマリに「計算時間」kv を追加。DsmcResult に持たせるので
  結果付き保存・バッチ・スイープにも自動で載る。
- **FEM Solve / Trace**: フロント側で api 呼び出しの前後の時刻差を計測し
  (backend 変更不要)、solve は ResultSummary に「計算時間」、trace は
  トレース結果サマリに「計算時間」を表示。App state に保持し、結果破棄時に
  一緒にクリア。※ 結果付き保存への同梱は PIC/DSMC のみで良い
  (solve/trace は数秒スケールで再現容易なため。コメントに明記)。
- **ResultsBundle**: pic に `elapsed_s?: number` を追加 (done から保存、復元表示)。

### 型・経路

- types.ts: PicDoneMsg.elapsed_s?、DsmcResult.elapsed_s、ResultsBundle.pic.elapsed_s?。
- batch.py の結果組み立てに pic elapsed_s を含める (server done と整合)。

## テスト

- backend: PIC done の elapsed_s > 0、DSMC result の elapsed_s > 0 を既存の
  WS/実行テストにアサーション追加 (新規ファイル不要、既存テストへの追記で可)。
- `python -m pytest tests/ -q` 全件パス (現在 215+)。

## 検証

- backend: pytest 全件。frontend: `npx tsc --noEmit && npx vite build`。

## 注意

- コメントは日本語で「なぜ」を書く既存スタイル。git commit はしない。
- 1秒 tick の setInterval は実行中のみ張る (アイドル時の無駄な再レンダーを避ける)。
