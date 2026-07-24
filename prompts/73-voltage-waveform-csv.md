# 73: 電極電位のCSV波形インポート (1周期ループ)

## 背景 (ユーザー要望)

- 電極電位を外部 CSV からインポートして適用したい。CSV 構造は 1列目=時間、2列目=電圧。
- **周波数はユーザーが指定**し、読み込んだ電圧波形を**1周期としてループ**させる。

RF 重畳 (voltage_rf: DC + Σ A·sin) と同格の「任意周期波形」電圧源を追加する。
静的ソルブ (Solve) では RF と同様に無視され (DC のみ)、PIC の毎ステップ
Dirichlet 更新で効く。

## データモデル

CSV の時間列は**波形の形状**を定義するためだけに使い、実際の周期は指定周波数で決まる。
フロントで取り込み時に正規化して保存する:

```
VoltageWaveform { freq_hz: float (>0), phase: list[float], v: list[float] }
```

- phase: CSV の時間 t を [t_min, t_max] → [0, 1) に線形写像した正規化位相
  (昇順ソート済み、範囲 [0, 1)、要素数 ≥ 2)。
- v: 対応する電圧 [V]。
- 評価: V_wf(t) = interp(frac(t·freq_hz), phase, v)。周期の折返しは
  phase 配列の末尾に (phase[0]+1, v[0]) を仮想的に足した線形補間
  (末尾サンプルと先頭サンプルの間を連続に繋ぐ)。

## backend

### es_sim/schema.py

- `VoltageWaveform(BaseModel)`: freq_hz (gt=0) / phase / v。validator で
  len(phase)==len(v)、len≥2、phase 昇順かつ [0,1) を検証。
- `BoundaryCondition` と `Region` (conductor) に
  `voltage_waveform: VoltageWaveform | None = None` を追加
  (voltage_rf と同じ場所・同じ流儀。dirichlet / conductor のみ有効)。

### es_sim/pic.py

- 現在の Dirichlet 時間発展 (1021行付近: V(t) = V_dc + Σ A_k sin(ω_k t + φ_k)) に
  波形項を追加: V(t) += V_wf(t)。
- 実装: PIC 初期化時に Dirichlet 節点ごとの波形参照 (どの VoltageWaveform を使うか)
  を組み立てる。voltage_rf の per-node 配列 (rf_amp/rf_omega/rf_phase) の組み立て箇所と
  同じ経路 (boundaries と conductor 領域の両方) に倣うこと。
  波形は種類数が少ない想定なので「波形リスト + 節点→波形インデックス (-1=なし)」で持ち、
  評価はベクトル化 (np.interp を波形ごとに1回)。
- 位相分解 (cycle) の基本周波数選択 (530行付近 `freqs.extend(...)`) に
  voltage_waveform の freq_hz も加える。
- 収束・警告まわりで RF 振幅を参照している箇所があれば波形の max|v| も考慮
  (確認のみ。過剰な変更は不要)。

### テスト (backend/tests/)

1. 波形評価の単体: 三角波1周期を CSV 相当のサンプル (例: phase [0, 0.25, 0.75], v [0, 1, -1])
   で与え、t = 0, T/4, T/2, 3T/4, T, 1.25T の V_wf が線形補間+ループの期待値と一致。
   (PicSimulation の該当評価関数を直接呼ぶ。内部関数化して単体テスト可能にする)
2. PIC スモーク: 小ケースの Dirichlet 辺に voltage_waveform を設定して数十ステップ
   実行し、エラーなく完走すること (診断値が有限であること)。
3. schema validator: phase 非昇順 / 長さ不一致 / freq 0 が ValidationError。

`python -m pytest tests/ -q` 全件パス (現在 142)。

## frontend

### types.ts

- `VoltageWaveform` 型、BoundaryCondition / Region に voltage_waveform を追加。

### FieldPanel.tsx (境界条件エディタ、RF重畳セクションの下)

- 「CSV波形」小セクション (Dirichlet 辺のみ):
  - 未設定時: 「CSVをインポート」ボタン (hidden file input、accept=".csv,text/csv")
    + 周波数入力は取り込み後に表示で良い。
  - CSV パース: 行分割 → ヘッダ行 (数値2つに解釈できない行) はスキップ →
    区切りはカンマ/タブ/空白に対応 → (t, v) を数値化 → t 昇順ソート →
    t を [0,1) に正規化 (t_max==t_min はエラー表示)。2点未満はエラー表示。
  - 取り込み後: 「N点 読み込み済み」表示、周波数 [Hz] の CommitNumberInput
    (既定 13.56e6? → いいえ、既定は 1e6 等の恣意値を避け、取り込み時に 1/(t_max−t_min)
    を初期値にする — CSV の時間レンジをそのまま1周期と解釈した周波数)、
    「解除」ボタン (voltage_waveform を undefined に)。
  - 適用は setEdgeVoltageWaveform ハンドラ (App.tsx に setEdgeVoltageRf と同様の
    ハンドラを追加、commitProject 経由)。
  - ヒント: 「1列目=時間、2列目=電圧のCSV。波形は指定周波数の1周期としてループ再生
    されます (PIC でのみ有効)」。
- conductor 領域側の UI は今回は追加しない (スキーマ上は対応するが UI は辺のみ。
  コメントに明記)。

### PicPanel.tsx

- RFサイクル換算 (collectRfFrequencies) に voltage_waveform.freq_hz も含める。

## 検証

- backend: pytest 全件。frontend: `npx tsc --noEmit && npx vite build`。

## 注意

- コメントは日本語で「なぜ」を書く既存スタイル (正規化位相・ループ補間の理由を書く)。
- 保存/読込は project JSON に自然に含まれるので追加対応不要 (確認のみ)。
- git commit はしない。
