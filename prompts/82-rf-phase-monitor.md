# 82: ライブモニタの RF 位相モニタ (画面下部の波形グラフ)

## 背景 (ユーザー要望)

「ライブモニタにRF波形のどのタイミングなのか分かるように、波形グラフを
画面下部に追加してほしい」

PIC ライブ表示中、現在のフレーム時刻 t が RF 波形のどの位相にあたるかを
一目で分かるようにする。

## 仕様

### コンポーネント (frontend/src/panels/RfPhaseMonitor.tsx 新規)

- props: `{ project, t }` (t = 現在フレームの時刻 [s]、picFrame.t)。
- 表示内容:
  - **基本周期 T = 1/(全時間依存ソースの最小周波数)** の1周期分 [0, T) を横軸に、
    時間依存電圧を持つ電極 (Dirichlet 辺 + conductor 領域、RF成分 or CSV波形あり)
    それぞれの合成 V(t) を色分けした折れ線で描画 (凡例: 辺名/領域id、
    最大4電極程度まで。超えたら先頭4つ+「他N」)。
    ※ 周波数収集・合成式は backend の _find_rf_freq / _dirichlet_values と
    一致させる。合成評価は VoltagePreviewChart (prompts/80) に実装済みなので、
    その評価関数を export して**必ず流用**する (式の二重実装を避ける)。
  - **現在位相マーカー**: x = t mod T の縦線 (アクセント色) + 上部に
    「位相 xx% (t=…)」の小ラベル。フレーム更新のたびに滑らかに動く。
  - 高さ 100px 程度、キャンバス幅いっぱい。ダークテーマ、軸は最小限
    (横軸: 0〜T を ns/µs 自動スケール、縦軸: V 範囲)。
- DC のみ (時間依存ソースなし) のプロジェクトでは null を返す (非表示)。

### 配置 (App.tsx)

- canvas-col 内、CadCanvas の直下 (ProfilePanel と同じ並び。両方出る場合は
  RF位相モニタ → ProfilePanel の順で縦に並んで良い)。
- 表示条件: `showRfMonitor && picFrame != null && (ライブ表示が有効な状態)` —
  実行中はもちろん、done 後にライブフレームが残っている間も表示して良い。
  結果フィールド表示や周期アニメ表示中 (picCycleView/picFieldView が優先されて
  ライブが見えていない時) は非表示にする (現在の描画優先ロジックの派生値を流用)。
- `showRfMonitor` state (既定 true)。トグルは PicPanel の「ライブ表示」select の
  並びに「RF波形モニタ」Toggle として追加。

## 検証

- `cd frontend && npx tsc --noEmit && npx vite build`。
- 合成値の確認は prompts/80 で検証済みの関数を流用するため再検証不要
  (流用していることをコードで確認)。
- マーカー位置の式 (t mod T)/T をコメントに明記。

## 注意

- backend には触れない。コメントは日本語で「なぜ」を書く既存スタイル。
- git commit はしない。
- style.css に必要な CSS を追加 (`.rf-phase-monitor` 等)。
