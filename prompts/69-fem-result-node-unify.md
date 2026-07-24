# 69: 結果ツリーの静電場ノードを他モジュールと一貫した形に統一

## 背景 (ユーザー要望・スクリーンショット付き)

結果 (ポスト) セクションが現在:
電位分布φ / 電場|E| / ラインプロファイル / 粒子軌道 / PIC結果 / ガス流れ結果

静電場だけ3ノードに分裂しており、他モジュール (1モジュール=1結果ノード) と
不揃い。**「静電場結果」1ノードに統一**する。

## 変更後のツリー (結果セクション)

```
▼ 結果                    [ポスト]
    静電場結果
    粒子軌道
    PIC 結果
    ガス流れ結果
```

## ノード型の変更 (App.tsx / ProjectTree.tsx)

- TreeNode union から `"result-phi" | "result-e" | "result-profile"` を削除し、
  `"result-fem"` を追加。NODE_TITLES は「結果 — 静電場」等の既存の命名規則に合わせる。
- ProjectTree の結果セクション定義を上記4ノードに変更。
- selectNode の `result-phi`/`result-e`/`result-profile` の副作用
  (fieldView 設定 / tool 切替) は削除する (新ページ内の UI が担う)。

## 「静電場結果」インスペクタページ (App.tsx 内、旧 result-phi/result-e ページを置換)

他の結果ページ (PIC結果/ガス流れ結果) と同じ構成に合わせる:

1. `<h2>結果表示</h2>`: select (電位 V / |E|) — 既存の `fieldView` state に直結
   (キャンバスツールバーの「表示」セレクトと同じ state を共有。二重配置で良い)。
   その下に 等電位線 / ベクトル チェックボックス (showIsolines/showVectors、
   これもツールバーと共有)。
2. `<h2>ラインプロファイル</h2>`: 「プロファイル線を引く」ボタン
   (onClick で `setTool("profile")`)。旧 result-profile ページの使い方ヒント文と
   「(まだプロファイル線が指定されていません)」表示をここへ移設。
   profileLine があれば「プロファイル表示中」等の一言 + 「閉じる」ボタン
   (setProfileLine(null)) があると親切 (任意)。
3. `<h2>解析結果</h2>`: ResultSummary (既存コンポーネント)。
4. result が無い場合はページ先頭に
   「静電場FEMが未実行です。スタディ「静電場 FEM」から実行してください。」の hint
   (他モジュールの結果ページと同じ文体)。表示オプション等はそのまま出して良い。

## その他の追従

- 保存/読込やその他のロジックに result-phi 等への参照が無いか grep で確認
  (`grep -rn "result-phi\|result-e\|result-profile" frontend/src`) し、すべて置換。
- ツリーの状態バッジ: 結果ノードには元々バッジは無いはずなので変更不要 (確認のみ)。

## 検証

- `cd frontend && npx tsc --noEmit && npx vite build` が通ること。
- grep で旧ノード名の残存が無いこと。
- 機能欠落チェック: 旧3ページで出来たこと (電位/|E|切替、等電位線、ベクトル、
  プロファイルツール起動、結果サマリ閲覧) がすべて新1ページから可能なこと。

## 注意

- backend には触れない。コメントは日本語で「なぜ」を書く既存スタイル。
- git commit はしない。
