# 70: チェックボックスをトグルスイッチに置換 + ラベル隣接レイアウト

## 背景 (ユーザー要望)

1. チェックボックスをアプリの雰囲気 (ダークテーマ) に合うトグルスイッチに変えたい。
2. 現状のレイアウトは「ラベルが左端・チェックボックスが右端」で離れすぎており、
   どの項目のチェックか目が滑って分かりにくい。**ラベルとコントロールを隣接**させる。

## 共通コンポーネント (frontend/src/Toggle.tsx を新規作成)

```tsx
// トグルスイッチ (チェックボックスの見た目置換)。label を渡すとテキスト+スイッチが
// 隣接した1つの <label> になり、テキストクリックでも切り替えられる
export function Toggle({ checked, onChange, label, disabled, title }: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label?: string;
  disabled?: boolean;
  title?: string;
})
```

- 実装: `<label className="toggle"><input type="checkbox" (視覚的に隠す) /><span className="toggle-track"><span className="toggle-knob"/></span>{label && <span className="toggle-label">{label}</span>}</label>`
- input は `position:absolute; opacity:0` で隠すがフォーカス/キーボード操作 (Space) は
  生かす。`:focus-visible` でトラックに outline。

## CSS (style.css、既存のテーマ変数を使う)

```css
.toggle { display: inline-flex; align-items: center; gap: 8px; cursor: pointer; font-size: 12px; }
.toggle input { position: absolute; opacity: 0; width: 0; height: 0; }
.toggle-track {
  width: 32px; height: 18px; border-radius: 9px; background: var(--border);
  position: relative; transition: background 0.15s; flex: none;
}
.toggle-knob {
  position: absolute; top: 2px; left: 2px; width: 14px; height: 14px;
  border-radius: 50%; background: var(--muted); transition: left 0.15s, background 0.15s;
}
.toggle input:checked + .toggle-track { background: var(--accent); }
.toggle input:checked + .toggle-track .toggle-knob { left: 16px; background: #fff; }
.toggle input:focus-visible + .toggle-track { outline: 2px solid var(--accent); outline-offset: 1px; }
.toggle.disabled { opacity: 0.5; cursor: default; }
```

(細部は調整可。トーンは既存の --panel/--border/--accent に馴染ませること)

## 置換対象 (`grep -rn 'type="checkbox"' frontend/src --include="*.tsx"` の全23箇所)

- **App.tsx**: 静電場結果ページの等電位線/ベクトル、その他 inspector 内のチェックすべて。
  キャンバスツールバーの「グリッドスナップ」も Toggle に置換 (label 付き)。
- **PicPanel.tsx**: 有効 (初期プラズマ/注入/MCC)、対数スケール、周期アニメの粒子表示等。
- **FnPanel.tsx**: FN有効。
- **GasPanel.tsx**: DSMC有効、対数スケール、粒子を表示。
- **ParticlePanel.tsx**: 軌道を表示、エミッタを表示。
- **FieldPanel.tsx**: 該当があれば同様に。

### レイアウト規則 (分かりにくさの解消)

- `.field` 行 (「<span class="label">有効</span> + 右端 checkbox」の2カラム) だった箇所は、
  **`<div className="field">` をやめて `<Toggle label="有効" ... />` の1行**にする。
  ラベルテキストとスイッチが 8px ギャップで隣接し、どの項目のトグルか一目で分かる。
  セクション見出し (h2) 直後の「有効」は `<Toggle label="有効">` とする。
- 既に `label.snap` で隣接している箇所 (軌道を表示等) も Toggle に置換して統一する
  (テキストは Toggle の label に渡す)。
- 「対数スケール」等、select の隣にある小さなチェックも同様に Toggle 化。
- 置換後、未使用になった CSS があれば掃除は不要 (触らない)。

## 検証

- `cd frontend && npx tsc --noEmit && npx vite build`。
- `grep -rn 'type="checkbox"' frontend/src --include="*.tsx"` が Toggle.tsx 内の1件のみに
  なること (実装内部の hidden input)。
- 機能は一切変えない (同じ state / ハンドラに接続するだけ)。disabled の挙動も維持。

## 注意

- backend には触れない。コメントは日本語で「なぜ」を書く既存スタイル。
- git commit はしない。
