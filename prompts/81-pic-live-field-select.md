# 81: PIC ライブモニタの表示フィールド切替 (電位/電子密度/イオン密度)

## 背景 (ユーザー要望)

「PIC計算中のライブモニタについて、現状は電位と超粒子を表示してもらっているが、
電位を電子密度やイオン密度に表示を切り替える機能を追加してほしい。
あるいは複数を同時に表示でもよい」

方針: frame メッセージに電子/イオン密度 (要素値) を同梱し、フロント側の
セレクトで**実行中でも即座に**切り替えられるようにする (キャンバスの色マップは
1つなので同時表示ではなく切替式。粒子オーバーレイは従来どおり常時重畳)。

## backend

### es_sim/pic.py

- frame 生成箇所 (run_batch で phi・粒子スナップショットを組む所) に、
  種ごとの**要素密度** n_e / n_i を追加:
  n[elem] = Σ_p w_p / V_elem。V_elem は xy: 面積×奥行1m、rz: 2πr̄A
  (デポジットや診断で同等の体積があれば流用。無ければ要素幾何から一度だけ
  前計算してキャッシュ)。np.bincount(weights=w) で軽量に計算できる
  (フレームは frame_every ステップごとなのでコストは無視できる)。
- ion が複数種になることは現状ない前提 (electron / ion の2種) だが、
  species dict の実装に合わせて electron→n_e、ion→n_i とする。

### es_sim/server.py

- frame メッセージに `"n_e": [...], "n_i": [...]` を追加 (要素値)。
  ペイロードは要素数×2 の追加で、frame_every ごとなので許容
  (コメントに明記)。

### テスト

- 既存の PIC WS テスト (または frame を検証しているテスト) に、frame の
  n_e/n_i が要素数長・全て有限・非負であることのアサーションを追加。
- 密度の妥当性: 一様初期プラズマの1フレーム目で、Σ n_e·V_elem ≈ マクロ粒子の
  総重み (相対誤差 1e-10 以下) のテストを1件。

`python -m pytest tests/ -q` 全件パス (現在 180)。

## frontend

### types.ts

- `PicFrameMsg` に `n_e?: number[]; n_i?: number[]` (optional、旧バックエンド互換)。
- `PicLiveFrame` を拡張: 現行 {mesh, phi, particles} に加えて表示対象の
  値配列を選べる形にする (実装しやすい形で良い。例:
  {mesh, values, nodeBased, unit, log, particles} に一般化し、App 側で
  選択フィールドから組み立てる)。

### App.tsx

- state `picLiveField: "phi" | "n_e" | "n_i"` (既定 "phi") と
  `picLiveLogScale: boolean` (既定 false、密度で有用)。
- picLiveFrame の組み立てを選択に応じて変える:
  - phi: 節点値 (nodeBased=true、単位 V) — 従来
  - n_e/n_i: 要素値 (nodeBased=false、単位 m^-3)、n_e/n_i が無い旧バックエンド
    フレームでは phi にフォールバック
- PicPanel へ props (picLiveField/onPicLiveFieldChange/picLiveLogScale/
  onPicLiveLogScaleChange) を渡す。

### CadCanvas.tsx

- picFrame (ライブ) の描画を一般化: 現在 phi 節点値前提の色マップ描画を、
  nodeBased フラグと log に対応させる (picFieldView 描画と同じ補間/要素塗り
  ロジックがあるはずなので流用)。カラーバーの単位表示も PicLiveFrame の unit から。
- 粒子オーバーレイは従来どおり。

### PicPanel.tsx

- setup ページの実行ボタン群の近く (started 進捗表示のあたり) に:
  「ライブ表示」select (電位 φ / 電子密度 n_e / イオン密度 n_i) +
  「対数スケール」Toggle (密度選択時のみ表示で良い)。
  hint: 「実行中のキャンバス表示を切り替えます (超粒子は常に重畳)」。

## 検証

- backend: pytest 全件。
- frontend: `npx tsc --noEmit && npx vite build`。
- uvicorn 実起動 + websockets で frame に n_e/n_i が載ることを実確認
  (終了後停止・一時スクリプト削除)。

## 注意

- コメントは日本語で「なぜ」を書く既存スタイル。git commit はしない。
- 結果付き保存 (ResultsBundle) の frame にも自然に含まれる (追加対応不要、確認のみ)。
