# 89: DSMC 専用メッシュ (粗化係数) + メッシュプレビュー更新不具合の修正

## 背景 (ユーザー要望)

1. walk が 97% を占める DSMC で、FEM 用の細かいメッシュをそのまま使っているのが
   コスト源。**DSMC 専用に粗いメッシュを使えるようにする** (PIC/FEM メッシュとは分離)。
   walk コスト ∝ 1/セル寸法なので粗化係数分だけ直接効く。
2. 一度計算を実行すると、再メッシュ (Mesh ボタン) してもキャンバスの
   メッシュプレビューが反映されない不具合の修正。

## ① DSMC メッシュ粗化係数

### schema (backend/es_sim/schema.py)

- `DsmcSettings.mesh_scale: float = Field(1.0, ge=1.0, le=20.0)` を追加。
  コメント: DSMC 用メッシュの寸法係数。FEM メッシュ寸法 × この係数で
  DSMC 専用メッシュを生成する。1.0 = 従来どおり FEM と同一メッシュ
  (このとき経路も従来と完全一致させ、結果はビット不変)。

### メッシュ生成 (dsmc 実行経路)

- mesh_scale > 1 のとき、project の mesh.size (と mesh.local_sizes の各 size) を
  係数倍した設定でメッシュを生成し、それを DSMC に使う
  (メッシュ生成関数は既存の meshing を流用。構造格子モードも同様に係数適用)。
- DsmcResult.mesh は従来どおり「DSMC が実際に使ったメッシュ」を返すので、
  結果表示・平滑化・粒子数目安 (result.mesh 参照) は自動で追従するはず (確認)。
- 続きから実行の整合: mesh_scale の変更は「ジオメトリ相当の変更」として
  continue を無効化する (App.tsx の dsmcContinueRelevantKey の除外リストに
  **入れない** = 変更で無効化される。確認のみ)。

### PIC 連成 (use_dsmc_gas) のメッシュ間マッピング

- 現状: GasField (n_g/t_g/u_g) は「PIC と同一メッシュの要素値」前提。
  mesh_scale > 1 だと DSMC メッシュ ≠ PIC メッシュ になるため、
  **PIC 開始時に PIC メッシュの各要素重心を DSMC メッシュ上で点位置特定して
  n/T/u を引き写す** マッピングを実装する。
- 点位置特定は既存機構を最大限流用する: postprocess.py (プロファイルの点サンプル)
  や particles.py の walk に既存の要素特定があるはず。無ければ
  scipy.spatial.cKDTree で DSMC 要素重心の最近傍を初期値に _walk_step で確定、
  ドメイン外/固体は最近傍要素へフォールバック。
- マッピングは PIC start 時に一度だけ (O(N_pic要素) で軽い)。
- server.py の GasField 構築箇所 (_store_dsmc_result / PIC start の受け渡し) を
  読み、DSMC メッシュと結果を保持して PIC 側でマッピングする形に変える。
  **mesh_scale=1.0 (メッシュ同一) のときはマッピングを介さず従来経路のまま**
  (ビット不変)。

### UI (GasPanel)

- 計算設定に「メッシュ粗化係数」CommitNumberInput (1〜20、既定1)。
  hint: 「DSMC 用メッシュの寸法 = FEM メッシュ寸法 × 係数。walk コストは係数分
  軽くなります。精度の目安はセル寸法 < 平均自由行程/3。PIC連成時は要素重心で
  自動マッピングされます」
- 粒子数目安 (セル数) の概算式に係数を反映 (実測メッシュは DSMC result があれば
  そちら優先 — 既にそうなっているか確認)。

### テスト (backend/tests/)

1. mesh_scale=1.0 が従来とビット不変 (既存の決定性テストがそのまま通ること + 明示1件)。
2. mesh_scale=3.0 で要素数が概ね 1/9 に減り、一様平衡箱の n/T が理論値と一致
   (粗くても平衡は保たれる)。
3. マッピング: 非一様 (圧力勾配) ケースを mesh_scale=2 で解き、PIC 側へ渡る
   GasField が DSMC 結果の対応位置の値と一致 (数要素をスポットチェック)。
   PIC スモーク (use_dsmc_gas + mesh_scale>1) が正常完走。
4. sccm 質量収支テストが scaled メッシュでも成立。

## ② メッシュプレビュー更新不具合

- 症状: Solve や PIC 実行後に Mesh ボタンで再メッシュしても、キャンバスの
  メッシュ表示が古いまま。
- 調査: CadCanvas のメッシュ描画がどのソース (meshResult / result.mesh /
  picStarted.mesh) をどの優先順で使っているかを読み、実行結果側のメッシュが
  meshResult より優先されている箇所を特定して、**「最後に生成した meshResult を
  最優先」**に直す (原因が別にあればそれを直す。修正内容を報告に明記)。
- 回帰確認: メッシュ表示トグルの通常動作 (Mesh→表示、Solve後の表示) が壊れないこと。

## 検証

- backend: `python -m pytest tests/ -q` 全件パス (現在 217+)。
- frontend: `npx tsc --noEmit && npx vite build`。
- ベンチ: 小ケースで mesh_scale=1 vs 3 の DSMC 実行時間 (walk位相) 比較を報告。

## 注意

- コメントは日本語で「なぜ」を書く既存スタイル。git commit はしない。
- ユーザーの perf 改修・平滑化・continue の挙動を壊さない (diff 最小限)。
