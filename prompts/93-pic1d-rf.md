# 93: 1D PIC/MCC 電極に RF 重畳 (voltage_rf) を追加

## 背景 (ユーザー要望)

「1DもRF重畳のモードを追加して」
2D の境界条件は `voltage (DC) + voltage_rf (正弦 RF、単一/リストでデュアル周波数) +
voltage_waveform (CSV)` の合成に対応しているが、1D の `Pic1dElectrode` は
`v_dc + waveforms (CSV)` のみで、RF はプリセットのように正弦波をサンプルした
CSV 波形でしか表現できない。2D と同じ **voltage_rf** を 1D 電極にも追加し、
V(t) = v_dc + Σ RF sin + Σ V_wf(t) の合成に揃える。

## backend

### schema.py — Pic1dElectrode

```py
class Pic1dElectrode(BaseModel):
    v_dc: float = 0.0
    # RF 重畳 (prompts/93)。2D の BoundaryCondition.voltage_rf と同じ規約:
    # 単一 VoltageRF / リスト (デュアル周波数など) / None。
    # V_rf(t) = Σ amplitude·sin(2π·freq_hz·t + phase_deg·π/180)
    voltage_rf: VoltageRF | list[VoltageRF] | None = None
    waveforms: list[VoltageWaveform] = []
    see_gamma: ...
```

docstring の「voltage_rf は持たない」の記述を更新。既存 `rf_components()` を流用。

### pic1d.py — 電圧評価

- 電極電圧評価関数に RF 成分を追加: **2D の pic.py の RF 評価式と完全に同じ式**
  (sin の位相規約・度→ラジアン変換を pic.py から読み取って一致させる) を使う。
- `voltage_rf=None` なら従来経路と**ビット不変** (既存テストで担保)。

### 位相分解 (cycle) の基本周波数

- 現状は「最初の CSV 波形の freq_hz」を基本周波数にしているはず。2D の規約
  (pic.py がどう基本周波数を決めているか) を読み、**voltage_rf を優先**して同じ
  優先順位に揃える (例: 左右電極の RF 成分の最初の freq → なければ CSV 波形の freq)。
  決定ロジックはコメントで明記。

### pic1d_presets.py

- edupic_ar / turner_he_case1 の RF 駆動を、正弦波サンプル CSV から
  `voltage_rf: {amplitude, freq_hz, phase_deg}` へ置き換える (本来の表現。
  サンプル補間誤差もなくなる)。プリセットのテストも追随。

### テスト (tests/test_pic1d.py に追加)

1. RF 評価: voltage_rf 指定時の電極電圧が振幅・周波数・位相どおり
   (t=0, T/4 などの点値。2D の式と同値であること)。
2. 等価性: voltage_rf {A, f, phase 0} と、同じ正弦を十分細かく (例 2000 点)
   サンプルした VoltageWaveform とで、短い実行の profiles が近い (rtol 緩め 1e-3、
   補間誤差分)。
3. voltage_rf=None で従来とビット不変 (既存の続きからテスト等が通ることでも担保、
   明示テスト1本追加: waveforms のみの短い実行結果が本変更前後で不変 —
   これは「既存テストが全パス」で代替可)。
4. cycle 基本周波数: voltage_rf のみ指定でも cycle が生成され freq_hz が一致。
5. デュアル周波数: voltage_rf をリスト2成分で指定して実行できる (数値の妥当性は
   有限値チェック程度で可)。

`python -m pytest tests/ -q` 全件パス (現在 239 + 新規)。

## frontend

### types.ts

- `Pic1dElectrode` に `voltage_rf?: VoltageRF | VoltageRF[] | null` を追加
  (2D の既存 VoltageRF 型を流用)。

### Pic1dPanel — 電極 (左/右) セクション

- 「RF 重畳」編集 UI を追加: 2D の FieldPanel の voltage_rf 編集 UI
  (振幅 [V]・周波数 [Hz]・位相 [deg]、成分の追加/削除でデュアル周波数対応) を
  読んで**同じ見た目・文言**で実装 (共通部品化できるならする、無理ならコピー+
  二重管理コメント)。CSV 波形 (WaveformImportEditor) とは併記 (両方指定可、
  hint: 「V(t) = DC + Σ RF + CSV 波形の合成」)。
- RF サイクル換算ヒント・位相ビン推奨の基本周波数も voltage_rf を含めた
  優先順位 (backend と同じ) に更新。

### Plot1dView — RF 波形モニタ

- Pic1dRfMonitor の電圧合成に RF 成分を追加 (backend の評価式と一致させる。
  2D の RfPhaseMonitor / VoltagePreviewChart の evalVoltage 流儀を参照)。
  voltage_rf があれば波形が表示されるようにする (現状 waveforms 空だと非表示の
  判定を「RF か CSV のどちらかがあれば表示」へ)。

### 検証

`cd /home/claude/ES-Sim/frontend && npx tsc --noEmit && npx vite build`。

## 注意

- コメントは日本語で「なぜ」。git commit はしない。
- 2D の式・規約 (pic.py の RF 評価、rf_components、FieldPanel の UI 文言) を必ず
  読んでから実装し、完全に一致させること。
- voltage_rf 未指定の既存プロジェクト・保存ファイルは挙動不変。
