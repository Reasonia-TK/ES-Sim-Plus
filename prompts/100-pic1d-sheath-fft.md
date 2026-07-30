# 100: 1D シース振動の FFT スペクトル抽出+可視化

## 背景 (ユーザー要望)

「1Dシースの振幅をFFTかけて周波数成分を抽出+可視化する機能を追加してほしい」
prompts/97 の位相分解 s(φ) は基本周波数の1周期に折り畳むため、基本波の高調波
しか表現できない。**時系列 s(t) を平均区間中に毎ステップ評価して FFT** すれば、
デュアル周波数駆動の混変調成分・ビートやシース振動の非高調波成分も抽出できる。

## backend (backend/es_sim/pic1d.py)

### s(t) 時系列の収集

- 平均区間 (avg_steps ウィンドウ) 中の**毎ステップ**、瞬時の節点密度 n_e/n_i
  (そのステップの deposit 済み値。既存の平均アキュムレータへ足しているのと同じ
  瞬時値) から `brinkmann_sheath_edge` で s_left(t)・s_right(t) を評価して蓄積する。
  1回の評価は O(n_nodes) のベクトル演算なので毎ステップでも負荷は無視できる
  (理由コメント)。瞬時密度は統計ノイズを含むが FFT では平均化される。
- 根なし (None) のステップ: 系列には NaN として積む。FFT 前に
  - 先頭/末尾の連続 NaN は切り落とす
  - 内部の孤立 NaN は線形補間で埋める (理由: FFT は等間隔連続系列が前提。
    コメントに書く)
  - 有効点が全体の 50% 未満なら FFT は諦めて null (縮退)。
- 続きから実行: 他の平均アキュムレータと同じ規約 (continue 時に新しい平均区間で
  取り直し)。

### FFT

- 基本周波数 f0 (cycle と同じ決定ロジック: voltage_rf 優先 → CSV 波形) が
  あれば、系列末尾を **f0 の整数周期分にトリム**してから FFT する
  (窓長が周期の整数倍でないと高調波がリークで滲むため。理由コメント)。
  f0 が無ければ全系列をそのまま使う。
- `np.fft.rfft` で片側振幅スペクトル: amp = 2·|X_k|/N (k=0 の DC は 1·|X_0|/N)。
  平均値 (DC) は mean_left/mean_right として別出しし、スペクトルは
  **平均を引いた変動分**に対して計算 (振幅 [m] がそのままシース振動振幅になる)。
- 返す帯域の上限: f0 があれば **40·f0 まで**、無ければ低周波側 2048 ビンまで
  (Nyquist 全帯域を JSON で返すと巨大になるだけで意味がないため。コメント)。
- 出力 (result に追加):

```py
"sheath_fft": {
  "df_hz": float,             # 周波数分解能
  "freq_hz": [...],           # 返却帯域の周波数列
  "amp_left": [...],          # 片側振幅 [m] (変動分)
  "amp_right": [...],
  "mean_left": float | None,  # s の平均 [m] (電極からの距離)
  "mean_right": float | None,
  "n_samples": int,           # FFT に使ったサンプル数 (トリム後)
  "f0_hz": float | None,      # 整数周期トリムに使った基本周波数
} | None
```

- s(t) のプレビュー系列も返す (チャート用。**最大 2048 点に間引き**。
  全ステップを JSON で返すと保存ファイルが肥大するため):

```py
"sheath_ts": { "t": [...], "s_left": [...], "s_right": [...] } | None  # NaN は null
```

- どちらも `build_pic1d_result` に含める (server / batch / sweep 全経路)。

### テスト (tests/test_pic1d.py に追加)

1. **合成系列の FFT 検証**: Pic1dSimulation を通さず、FFT 部分を関数
   `sheath_fft(s: np.ndarray, dt: float, f0: float | None) -> dict | None`
   (モジュールレベル純関数) に切り出し、s(t) = s0 + A1·sin(2πf1 t) + A2·sin(2πf2 t)
   (f1 = f0、f2 = 3f0、窓 = 非整数周期) に対して:
   - mean ≈ s0
   - f1・f2 ビンの振幅 ≈ A1・A2 (rtol 1e-2。整数周期トリムが効いている証拠)
   - 他ビンの振幅が A1 の 1% 未満
2. **NaN 補間**: 内部に数点 NaN を入れても 1. が成立。有効率 50% 未満で None。
3. **実行搭載**: RF 付き CCP スモークで result.sheath_fft が非 null、
   freq_hz の上限 ≈ 40·f0、amp 配列長一致、sheath_ts が ≤2048 点。
4. 既存テスト全パス (result キー集合を検査するテストがあれば追随)。

`python -m pytest tests/ -q` 全件パス (現在 268 + 新規)。

## frontend

### types.ts

- `Pic1dSheathFft`、`Pic1dSheathTs`、`Pic1dResult.sheath_fft?` / `sheath_ts?`。

### Plot1dView.tsx (結果ビュー、シースエッジ関連の並び)

- **s(t) チャート**: sheath_ts があれば時間 [µs] vs s (表示単位)、左右2色
  (シースエッジの既存配色 #ffb454 / #ff7a45)。null は線を切る。
- **スペクトルチャート**: sheath_fft があれば「シース振動スペクトル」:
  横軸 周波数 [MHz]、縦軸 振幅 (表示単位、**対数軸トグル** 既定オン)。
  左右2系列。f0 があれば n·f0 (n=1..) の位置に薄い縦グリッド線。
- **ピーク表**: 振幅上位 5 つの局所ピーク (左右それぞれ。DC 除く) を
  「f [MHz] / f/f0 / 振幅」の小さな表で表示 (f/f0 は f0 があるときのみ)。
- 数値サマリに「シース振動: 左 A@f0 / 右 A@f0」(f0 ビンの振幅) を追加
  (f0 があるときのみ)。

### 検証

`cd /home/claude/ES-Sim/frontend && npx tsc --noEmit && npx vite build`。

## 注意

- コメントは日本語で「なぜ」。git commit はしない。
- sheath_fft/sheath_ts は optional — 旧保存ファイル読込は不変。
- 毎ステップ評価の追加で既存の数値結果 (フィールド・history 等) がビット不変で
  あること (読み取りのみで乱数消費・状態変更をしない)。
- Math.min/max スプレッド禁止 (arrayMin/arrayMax)。対数軸は値 0 を除外して描く。
