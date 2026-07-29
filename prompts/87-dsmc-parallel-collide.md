# 87: DSMC の位相別計測 + NTC 衝突のセル並列化

## 背景 (ユーザー報告)

「DSMCの方はスレッド数を変えてもCPU負荷がかわらないみたい」(Xeon 6442Y ×2)。
現状スレッドが効くのは walk のみで、NTC 衝突 (_collide) とサンプリングは
直列 numpy のまま。以前の計測でも DSMC の律速は衝突判定だった。

## ① 位相別計測 (PIC の prompts/75 と同じ流儀)

- DsmcSimulation に timing dict (inject / move / collide / sample / other) を追加し、
  step() の各位相を perf_counter で計測。
- DsmcResult に `timing: dict[str, float]` を追加 (elapsed_s と並べる)。
  GasPanel の結果サマリに「実行時間内訳」(降順、秒と%) を表示。
  結果付き保存・バッチにも自然に載る (DsmcResult 経由)。

## ② _collide の Numba JIT + セル並列化

現行 _collide (dsmc.py) は全セル一括のベクトル化 numpy。これを:

- **乱数は従来どおり numpy Generator で事前に配列生成**して JIT カーネルへ渡す
  (決定性の維持。候補数の計算・持ち越し (_coll_frac)・(σc_r)_max の実測更新の
  意味を変えないこと)。
- **セルごとの処理を prange で並列化**: 候補対の選定・採択・衝突更新はセル内で
  完結する (i1/i2 は同一セルの粒子) ため、セル単位の並列は書き込み競合しない。
  ただし現行実装の「同一粒子が複数対に選ばれた場合は最初の対のみ実行し、
  落とした対は持ち越し」というエネルギー保存の扱い (dedup) をセル内で
  厳密に再現すること。
- `_sigcr_max` の更新はセルごとに独立 (競合なし)。`_coll_frac` も同様。
- スレッド数は既存の set_num_threads (pic と同じ) を流用。
- numpy フォールバック: numba 無し環境では従来実装をそのまま使う
  (ES_SIM_NO_NUMBA=1 で強制可、既存の分岐流儀に合わせる)。

### 決定性

- 乱数列は従来と同じ順・同じ本数で引く (事前生成) こと。並列化はセルごとの
  書き込み独立性で担保し、**threads=1 と threads=N がビット単位一致**、
  さらに可能なら**従来 numpy 実装ともビット一致** (乱数消費順を変えなければ
  達成できるはず。無理なら理由を報告し、統計同等テストに切替え)。

## ③ _sample の軽量化 (余力があれば)

- bincount 4本 (cnt, vx, vy, v²) を1つの njit ループに融合 (並列は per-thread
  部分和+固定順合算で決定的に)。効果が小さければスキップして報告。

## テスト

1. 既存 DSMC テスト全パス (threads=1 vs 4 の一致テスト・continue 一致テスト含む —
   これが決定性の担保になる)。
2. 従来実装との一致 (可能な場合): 小ケースで新旧 _collide の結果がビット一致。
3. timing のキーと合計の妥当性テスト (PIC の test_timing_phases に倣う)。

`python -m pytest tests/ -q` 全件パス (現在 215)。

## ベンチ (最終報告に含める)

- 20万粒子・衝突が活発な条件 (init_pressure 高め) で threads=1/2 の
  位相別前後比較 (このコンテナは2コア)。②の狙いは衝突位相のスケールなので、
  衝突位相の並列化効果と、直列残り (Amdahl) を報告。

## 注意

- コメントは日本語で「なぜ」を書く既存スタイル。git commit はしない。
- ユーザーの perf 改修と衝突しないよう diff は最小限。
- 平滑化 (smoothing_passes) や続きから実行の挙動を壊さない。
