# 72: DSMC 境界線分の2点クリック配置ツール

## 背景 (ユーザー要望)

「DSMCで境界を指定する際に、コレクタと同じように2点のクリックで定義できるようにしてほしい」

現在 DSMC の線分境界 (流入口など、p1/p2 指定) は GasPanel の数値入力のみ。
コレクタツール (2点クリックで pic.collectors へ追加) と同じ操作感で置けるようにする。

## 仕様

### CadCanvas (frontend/src/canvas/CadCanvas.tsx)

- `Tool` union に `"gasbc"` を追加。
- クリック挙動はコレクタツールと同一 (1点目クリック→プレビュー線→2点目クリックで確定、
  グリッドスナップ有効)。コレクタの実装 (onSetCollector まわり) を参考に
  `onSetGasBoundary: (p1: Point, p2: Point) => void` prop を追加して同じ流れにする。
- **配置済み DSMC 線分境界のオーバーレイ表示**: 新 prop
  `gasBoundaries?: { p1: Point; p2: Point; label: string }[]` を追加し、
  コレクタのオーバーレイ描画を参考に橙系 (#ffb454) の線分+ラベルで常時描画する
  (コレクタは別色なので区別が付く)。

### App.tsx

- ツールバーに「ガス境界」ボタン (tool="gasbc") をコレクタの隣に追加。
- `setGasBoundaryPoints(p1, p2)` ハンドラ:
  - project.dsmc が null なら GasPanel の既定 DSMC 設定 (DEFAULT_DSMC を export して
    流用) をベースに有効化してから追加する (ツールを使った時点で DSMC を使う意図が
    明確なため。コメントに理由を書く)。
  - dsmc.boundaries に線分境界を1件追加する。既定値は **GasPanel の「境界を追加」
    ボタンが作る新規境界と同じ内容** (type 等の既定を GasPanel の実装から流用。
    共通化できるなら定数を export して共有) + p1/p2。
  - commitProject 経由 (project.dsmc は Undo/Redo 対象) で反映し、
    activeNode を "study-gas" に切替える (コレクタ→PIC と同じ誘導)。
- `gasBoundaries` オーバーレイ用の派生値: project.dsmc の線分境界 (p1/p2 を持つもの)
  を `G1, G2, ...` のラベルで CadCanvas へ渡す (エッジ指定境界は対象外)。

### GasPanel.tsx

- 境界条件セクションに「キャンバスの「ガス境界」ツールで2点クリックでも追加できます」
  の hint を1行追加 (発見性のため)。既存の数値編集・追加ボタンはそのまま。

## 検証

- `cd frontend && npx tsc --noEmit && npx vite build`。
- コレクタツールの既存挙動 (ツール切替・プレビュー・確定・上限) が壊れていないこと
  (コードレビューで確認)。

## 注意

- backend には触れない (schema の DsmcBoundary は既に p1/p2 線分に対応済み)。
- コメントは日本語で「なぜ」を書く既存スタイル。git commit はしない。
