# 97: 1D PIC/MCC — Brinkmann 基準のシースエッジ検出+可視化

## 背景 (ユーザー要望)

「1Dと2Dの両方にsheath edgeを検出および可視化する機能を追加したい。brinkmannの式がよいかな」
→ 1D は Brinkmann 基準 (本プロンプト)。2D は prompts/98 (別実装)。

## Brinkmann 基準 (Brinkmann, J. Appl. Phys. 102, 093303 (2007) の step model)

電極表面 (x=0) からバルク参照点 x_b までの密度プロファイルに対し、シースエッジ
位置 s を次で定義する:

    G(s) = ∫₀ˢ n_e dx − ∫ₛ^{x_b} (n_i − n_e) dx = 0

意味: 実際の滑らかな n_e を「s より内側 (電極側) は 0、外側は n_i に等しい」段差
分布に置き換えたとき、総電子量が保存される位置。
dG/ds = n_i(s) ≥ 0 なので G は単調非減少 → 根は一意 (存在すれば)。

## backend (backend/es_sim/pic1d.py)

### 実装

- 関数 `brinkmann_sheath_edge(x, n_e, n_i, from_left: bool, x_b: float) -> float | None`
  (モジュールレベル・純関数。テスト容易性のため):
  - 左電極: 上式そのまま。右電極: x → gap−x に鏡映して同じ式 (電極からの距離で評価)。
  - 台形則の累積積分で G を節点上に評価し、符号が変わる区間を線形補間して根を返す。
  - 根が無い (全域 G<0: プラズマ未形成 / 全域 G>0) 場合は None。
  - x_b は **gap/2** (CCP の両シースを対称に扱う標準的な取り方。コメントに理由)。
- 時間平均: `averaged_fields` の n_e/n_i から左右の s を計算 →
  done result の `profiles` と並ぶ形で `sheath: {left_s, right_s} | null`
  (どちらか None でもフィールドは null でなく個別 null 可)。
- 位相分解: cycle データ (bins × 節点の n_e/n_i) の各ビンで同様に →
  `cycle.sheath: {s_left: (bins,), s_right: (bins,)}` (None は NaN でなく null を
  JSON に入れる。TS 側は (number | null)[])。cycle が無ければ省略。
- 計算は run_batch の結果組み立て時のみ (毎ステップは計算しない。理由コメント)。
- `build_pic1d_result` に含める (server / batch / sweep 全経路に自動で載る)。

### テスト (tests/test_pic1d.py に追加)

1. **段差プロファイルの解析解**: n_i = n0 (一様)、n_e = 0 (x<d) / n0 (x≥d) の
   合成配列 → s = d に一致 (節点補間精度、rtol 1e-9。d は節点上に置かない)。
2. **線形ランプ**: n_i = n0、n_e = n0·x/x_b (0≤x≤x_b) → 手計算根
   G(s) = n0 s²/(2x_b) − [n0(x_b−s) − n0(x_b²−s²)/(2x_b)] = 0
   → s = x_b (√2 − 1)? を検算し (実装ではなく**プロンプト読者の検算ではなく
   テスト内で sympy や手計算値をコメント付きで**)、その値と一致すること。
3. **右電極の鏡映**: 左のケースを反転した配列で right が対称値を返す。
4. **プラズマ無し**: n_e = n_i = 0 → None。
5. **実行結果への搭載**: CCP スモーク (既存の短い Ar 実行を流用) で
   result.sheath.left_s / right_s が 0 < s < gap/2 の有限値、cycle ありなら
   cycle.sheath の配列長が bins。

`python -m pytest tests/ -q` 全件パス (現在 260 + 新規)。

## frontend

### types.ts

- `Pic1dResult.sheath?: { left_s: number | null; right_s: number | null } | null`
- `Pic1dCycle.sheath?: { s_left: (number | null)[]; s_right: (number | null)[] }`

### Plot1dView.tsx

- **時間平均プロファイル表示**: sheath があれば s_left / s_right の位置に縦の
  破線マーカー (#ffb454 系の目立つ色、ラベル「シースエッジ」)。密度表示・電位表示
  どちらのフィールド選択でも表示 (トグル「シースエッジ」で非表示可、既定オン)。
- **位相アニメーション**: cycle.sheath があれば、現在の位相ビンの s_left/s_right
  マーカーを位相スライダ/再生に連動して動かす。
- **s(φ) チャート**: cycle.sheath があれば、位相 (横軸 0..1) vs s_left / s_right
  [表示単位] の小さな折れ線チャートを結果ビューに追加 (2色 + 凡例。null ビンは
  線を切る)。
- **数値サマリ**: 「シースエッジ: 左 s / 右 s」行を追加 (formatNumber、表示単位)。

### 検証

`cd /home/claude/ES-Sim/frontend && npx tsc --noEmit && npx vite build`。

## 注意

- コメントは日本語で「なぜ」(Brinkmann 基準の式と x_b=gap/2 の理由を docstring に)。
- git commit はしない。既存結果 (sheath キー無し保存ファイル) の読込は不変
  (optional フィールド)。
- Math.min/max スプレッド禁止 (arrayMin/arrayMax)。
