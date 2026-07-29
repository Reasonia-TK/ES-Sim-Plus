# 88: walk 診断 — 平均横断セル数の推定表示

## 背景 (ユーザー報告)

実機 (Xeon 6442Y×2) で walk 探索が実行時間の 97% を占める。ケース JSON は
共有不可の環境のため、**診断値をアプリ内に表示**してユーザーに読み上げてもらう
方式で原因を切り分ける。

walk コスト ∝ 粒子が1ステップに横切るセル数 ≈ v̄·dt / h (h = 代表セル寸法)。
この推定値と関連量を timing と一緒に表示する。

## 計測 (backend)

### 共通

- 代表セル寸法: h_i = sqrt(2·A_i) (要素ごと)、代表値は **面積重み平均ではなく
  「粒子が実際にいる要素の h の平均」**が理想だが、簡易版として
  h_mean = mean(sqrt(2·A_i)) と h_min = min(sqrt(2·A_i)) の両方を出す
  (ローカル細分化の影響は h_min / h_mean の比で見える)。
- 横断セル数の推定: 数十ステップに1回 (DSMC は progress 間隔、PIC は frame 間隔)、
  mean(|v|)·dt_species / h_mean を計算し、実行全体で平均する。O(N) の numpy 演算
  なので頻度が低ければコストは無視できる。

### DSMC (dsmc.py)

- run() 中に上記を集計し、DsmcResult.timing に追加キーで格納:
  `walk_cells_est` (平均横断セル数)、`h_mean_m`、`h_min_m`。
  (timing は dict[str, float] なのでキー追加だけで schema 変更不要のはず — 確認)

### PIC (pic.py)

- 種ごと (electron / ion) に同様の推定を frame 間隔で集計し、done の timing に
  `walk_cells_est_e` / `walk_cells_est_i`、`h_mean_m`、`h_min_m` を追加。

## 表示 (frontend)

- GasPanel / PicPanel の「実行時間内訳」セクションの末尾に診断行を追加:
  - 「平均横断セル数/ステップ: X.X (電子)」等。**時間の行と混ざらないよう**
    timing dict から `walk_cells_est*` / `h_*` キーは除外して別表示にする
    (既存の内訳表示が壊れないことを必ず確認)。
  - h_min / h_mean 比が 1/5 未満なら「メッシュ寸法の偏りが大きい」旨を薄字で付記。
  - 横断セル数 > 3 なら hint: 「walk コストが支配的な場合、DSMC ではメッシュを
    粗くする (計算精度の目安はセル寸法 < 平均自由行程/3)、PIC では dt を
    見直すことで改善できる可能性があります」
- 表示は既存 kv の流儀。

## テスト

- DSMC / PIC の小ケースで timing に新キーが存在し、値が正 (h は妥当な m オーダー)
  であることのアサーションを既存 timing テストに追記。
- `python -m pytest tests/ -q` 全件パス (現在 217)。

## 検証

- backend: pytest。frontend: `npx tsc --noEmit && npx vite build`。
- 既存の timing 表示 (秒・%) に新キーが混入して % 計算を壊さないことを確認。

## 注意

- コメントは日本語で「なぜ」を書く既存スタイル。git commit はしない。
- 計測は推定で良い (カーネル変更・実カウントはしない。決定性と速度を守る)。
