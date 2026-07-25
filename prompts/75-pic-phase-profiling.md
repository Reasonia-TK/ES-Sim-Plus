# 75: PIC の位相別プロファイル計測 (高速化①)

## 背景

大規模PICへ向けた高速化の第一歩として、1ステップ内の位相別時間を常時計測し、
どこが支配的かをユーザーにも見えるようにする (以降の Numba 化②・粒子マージ③の
効果測定の土台)。

## backend (es_sim/pic.py)

- PicSimulation に累積タイマー dict を追加:
  `self.timing = {"solve": 0.0, "gather_push": 0.0, "walk": 0.0, "deposit": 0.0, "mcc": 0.0, "other": 0.0}`
  (キーは実装のステップ構造に合わせて調整可。ただし solve / walk / deposit / mcc は必須。
   FN/SEE/注入などの細かい処理は "other" にまとめて良い)。
- step 内の各位相を time.perf_counter で挟んで加算する。呼び出しは1ステップ
  あたり十数回程度なのでオーバーヘッドは無視できる (コメントに明記)。
- スレッド並列の walk は submit〜完了待ちの壁時計時間を計測する。
- done メッセージ (server.py の _stream_run 完了時) に
  `timing: {solve: s, walk: s, ..., total: s}` を追加 (continue でも区間分を返す。
  prepare_continue で timing をリセットする)。
- 計測自体の ON/OFF は不要 (常時ON)。

## frontend

- types.ts: PicDoneMsg に `timing?: Record<string, number>` を追加 (optional)。
- App.tsx: done で timing を state に保持 (新規実行/continueで置換)。
- PicPanel (setup ページの診断のあたり、done 後のみ):
  `<h2>PIC: 実行時間内訳</h2>` として位相ごとに `名称 / 秒 / %` の kv 行を
  値の大きい順に表示。total も表示。ラベルは日本語
  (場ソルブ/粒子押し出し/walk探索/電荷デポジット/MCC衝突/その他 等)。

## テスト

- 小ケースの run_batch 後に timing の全キーが ≥0 かつ total ≈ 合計 (±計測誤差) で
  あることのテストを1件追加。
- `python -m pytest tests/ -q` 全件パス (現在 152)。

## 計測レポート (最終報告に含める)

代表ケース (2万マクロ粒子・MCCあり・500ステップ程度の合成ケース) を実行し、
位相別の秒数と割合を報告する (②の効果測定のベースラインとする)。
ベンチスクリプトは backend/benchmarks/pic_bench.py として**リポジトリに残す**
(次フェーズで同条件比較に使うため。実行方法をファイル先頭コメントに書く)。

## 検証

- backend: pytest 全件。frontend: `npx tsc --noEmit && npx vite build`。

## 注意

- コメントは日本語で「なぜ」を書く既存スタイル。git commit はしない。
