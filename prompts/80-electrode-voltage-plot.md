# 80: 電極電位 V(t) のプレビュープロット

## 背景 (ユーザー要望)

「電極電位をプロットする機能を追加してほしい」
電極の印加電位は DC + RF成分 (複数可) + CSV波形の合成で、設定が複雑になってきた。
実際に印加される V(t) を時系列プロットで確認できるようにする。
境界 (Dirichlet) の電位は固定値なので V(t) は設定から決定的に計算できる —
**frontend のみで完結** (backend 変更なし)。

## 仕様

### 合成式 (backend の pic.py と厳密に一致させる)

V(t) = V_dc + Σ_k A_k·sin(2π f_k t + φ_k) + V_wf(t)

- RF成分: types.ts の VoltageRf のフィールド (amp/freq/phase の単位・名前) を確認し、
  pic.py の _dirichlet_values の式と一致させる (位相の単位 deg/rad に注意)。
- CSV波形: V_wf(t) = 線形補間(frac(t·freq_hz), phase, v)。周期の折返しは
  末尾に (phase[0]+1, v[0]) を仮想的に足した補間 (backend の _eval_waveform と同じ)。

### コンポーネント (frontend/src/panels/VoltagePreviewChart.tsx 新規)

```ts
function VoltagePreviewChart({ voltage, rf, waveform }: {
  voltage: number;
  rf: VoltageRf[];           // rfComponents() で正規化済み
  waveform: VoltageWaveform | null;
})
```

- 時間窓: 存在する周波数 (RF各成分 + 波形) の**最低周波数の2周期**。
  周波数が1つも無い (DCのみ) 場合はコンポーネント自体を描画しない
  (呼び出し側で分岐)。
- サンプル数 600 点程度で canvas に折れ線描画。既存チャート
  (PicPanel の PicHistoryChart や ProfilePanel) の描画流儀・配色に合わせる:
  ダーク背景、軸+目盛り (時間軸は ns/µs/ms を自動スケール、縦軸 V)、
  ホバーで (t, V) 読み値表示 (ProfilePanel のホバーを参考に。工数が嵩むなら
  ホバーは省略可 — その場合は報告に明記)。
- 高さ ~140px、パネル幅いっぱい。

### 配置 (FieldPanel.tsx)

- **Dirichlet 辺エディタ**: CSV波形セクションの直後に「電位プレビュー」小見出し +
  チャート。RF も波形も無い (DCのみ) 場合は何も出さない。
- **conductor 領域エディタ**: 同様に RF/波形設定の直後。
- どちらも同じ VoltagePreviewChart を共用。

## 検証

- `cd frontend && npx tsc --noEmit && npx vite build`。
- 合成式の一致確認: 代表値 (DC=100, RF 50V@13.56MHz phase 0, 三角波 CSV) で
  数点の V(t) を backend の該当式で手計算 (python ワンライナー可) して
  チャートのサンプル値と一致することを確認し、報告に記載。

## 注意

- backend には触れない。コメントは日本語で「なぜ」を書く既存スタイル。
- git commit はしない。
