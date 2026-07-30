# 92: 1D PIC/MCC フロントエンド — スタディ「PIC-MCC 1D」

## 背景

prompts/91 でバックエンドに 1D PIC/MCC (1d3v、一様格子) を実装済み
(`/ws/pic1d`、`GET /pic1d/presets`、schema の `Project.pic1d`)。
本プロンプトでフロントエンドを追加する。**設定は左パネルのみ・キャンバス領域には
結果のラインプロット/ライブモニタを表示** (CAD キャンバスでの 1D 編集はしない)。

## バックエンドのプロトコル (実装済み・これに合わせる)

### /ws/pic1d

```ts
// client → server
{ cmd: "start", project: ProjectJSON }
{ cmd: "continue", extra_steps: number, frame_every?: number, avg_steps?: number, phase_bins?: number }
{ cmd: "stop" }

// server → client
{ type: "started", n_steps: number, step_offset: number, dt: number, x: number[], warnings: string[] }
{ type: "frame", step: number, t: number, phi: number[], n_e: number[], n_i: number[],
  counts: Record<string, number>, elapsed_s: number,
  sample: { x: number[], vx: number[] } }   // 電子位相空間 ≤2000点
{ type: "done", result: Pic1dResult }
{ type: "error", detail: string }
```

```ts
interface Pic1dResult {
  history: { step: number[]; t: number[]; n_e: number[]; n_i: number[];
    w_e: number[]; w_i: number[];
    wall_left_e: number[]; wall_left_i: number[];
    wall_right_e: number[]; wall_right_i: number[];
    ion_events: number[]; see_events: number[]; coll_e: number[] };
  profiles: { x: number[]; phi: number[]; e: number[]; n_e: number[]; n_i: number[];
    t_e: number[]; ionization: number[]; avg_steps: number } | null;
  cycle: { bins: number; freq_hz: number;
    phi: number[][]; n_e: number[][]; n_i: number[][] } | null;
  eedf: Array<{ label: string; e_centers: number[]; f: number[]; mean_energy_ev: number;
    t_eff_ev: number; total_weight: number; overflow_frac: number; n_samples: number }>;
  walls: { left: { electron: number; ion: number }; right: { electron: number; ion: number } };
  elapsed_s: number;
  timing: { deposit: number; field: number; push: number; mcc: number; other: number; total: number };
  settings: Pic1dSettings;
}
```

### GET /pic1d/presets

```json
{ "edupic_ar":       { "label": "...", "description": "...", "pic1d": Pic1dSettings },
  "turner_he_case1": { "label": "...", "description": "...", "pic1d": Pic1dSettings, "note": "He 断面積を LXCat からインポート..." } }
```

### schema (types.ts へ写す)

```ts
interface Pic1dElectrode { v_dc?: number; waveforms?: VoltageWaveform[]; see_gamma?: number }
interface Eedf1dRegion { x1: number; x2: number; label?: string; bins?: number; e_max_ev?: number | null }
interface Pic1dSettings {
  gap_m: number; n_cells: number; left: Pic1dElectrode; right: Pic1dElectrode;
  init_density_m3: number; init_te_ev?: number; init_ti_ev?: number; ion_mass_amu?: number;
  n_macro?: number; dt?: number | null; n_steps?: number; frame_every?: number;
  avg_steps?: number | null; phase_bins?: number; mcc?: MccSettings | null;
  see_energy_ev?: number; eedf_regions?: Eedf1dRegion[]; seed?: number;
}
interface Project { ...; pic1d?: Pic1dSettings | null }
```

正確なフィールドは backend/es_sim/schema.py (Pic1dSettings ほか) を読んで確認すること。

## UI 仕様

### ProjectTree

- 「スタディ」に **PIC-MCC 1D** ノードを追加 (既存 PIC-MCC の下)。
- 「結果」に **PIC-MCC 1D** ノードを追加 (結果があるときのみ活性、既存の結果系列の流儀)。
- 選択キー例: "pic1d" / "result-pic1d" (既存の selection の命名を読んで合わせる)。

### 新パネル frontend/src/panels/Pic1dPanel.tsx

スタディ「PIC-MCC 1D」選択時に表示。既存 PicPanel の見た目・部品
(CommitNumberInput/CommitNullableNumberInput、Toggle、hint 文体) を踏襲。

- **プリセット**: 起動時 (パネル初回表示時) に GET /pic1d/presets を取得し、
  「プリセット適用」select + 適用ボタン。適用は commitProject で pic1d を丸ごと置換
  (確認なしで良いが、hint に「現在の 1D 設定を上書きします」)。
  turner_he_case1 の note はプリセット説明に表示。
- **形状**: ギャップ長 (表示単位 mm/µm — units.ts の lengthUnit に追従、内部は m)、
  セル数。
- **初期プラズマ**: 密度 [m^-3]、Te [eV]、Ti [eV]、イオン質量 [amu]、マクロ粒子数、seed。
- **電極 (左/右)**: v_dc、RF/CSV 波形 — 既存の WaveformImportEditor
  (FieldPanel の共有部品) を流用して waveforms を編集。see_gamma。
- **実行設定**: dt (null=自動)、ステップ数、フレーム間隔、平均ステップ (null=最後の25%)、
  位相ビン数 (既存 PicPanel の推奨値ヒントの流儀があれば軽く流用)、SEE エネルギー。
  RF サイクル換算表示 (n_steps·dt が何 RF 周期か。既存 PicPanel の実装を流用)。
- **MCC**: 既存 PicPanel の MCC 設定 UI (ガス圧・温度・断面積インポート・
  ionization_split・ion_energy_frame) を**部品化して共用**するのが理想だが、
  PicPanel の構造次第でコピーでも可 (二重管理のコメントを残す)。use_dsmc_gas の
  トグルは 1D では出さない。
- **EEDF 区間**: x1/x2 (表示単位変換)・label・bins・e_max の行リスト (最大4)。
  2D の EEDF 設定 UI の文体に合わせる。
- **実行**: 「PIC 1D 開始」「続きから (+Nステップ)」「停止」。既存 picClient.ts を
  参考に **pic1dClient.ts** を新規作成 (started/frame/done/error、continue、stop)。
  進捗はステータスバー (App.tsx の既存進捗機構: anyRunning・経過秒表示・
  エラー居座り修正の流儀に統合。pic1dRunning 状態を追加し dismissStatusError にも組み込む)。

### キャンバス領域 (1D ビュー)

CadCanvas に手を入れず、**App.tsx で pic1d スタディ/結果選択時に CadCanvas の代わりに
新コンポーネント `frontend/src/canvas/Plot1dView.tsx` を表示**する (レイアウト・
ツールバーの見た目は崩さない。1D では CAD ツールバーの代わりにプロット用の
簡素なヘッダで良い)。

Plot1dView (SVG 描画。既存の VoltagePreviewChart / EEDF チャートの描画流儀・配色
(ダークテーマ、#59c2ff 系) を踏襲):

- **ライブ (実行中)**: 上段 φ(x) 折れ線 + n_e(x)/n_i(x) (右軸 or 対数トグル)、
  下段 電子位相空間 x–vx 散布 (sample、点は小さく半透明)。
  RF 波形モニタ (既存 RfPhaseMonitor) を下部に (左電極の波形 + 現在時刻 t マーカー。
  波形が空なら非表示)。
- **結果 (done 後 / result-pic1d 選択)**:
  - フィールド選択 select: 電位 φ / 電場 E / n_e / n_i / n_e+n_i 重ね / T_e / 電離レート。
    対数表示 Toggle (密度系のみ)。
  - 位相分解 (cycle があれば): 位相スライダ (0..bins-1) + 再生/停止ボタンで
    φ/n_e/n_i の位相アニメーション (既存 2D の位相アニメ UI があれば流儀を合わせる)。
  - EEDF チャート (eedf があれば): 2D の EEDF チャート部品を流用。CSV 書き出しも
    既存流儀で。
  - 数値サマリ: elapsed_s、timing 内訳 (既存 PicTimingSection の流儀)、
    壁吸収 (左右 e/i)、中央密度 n_i(gap/2) など。
  - history: N_e/N_i (マクロ数) vs ステップの小さな折れ線。
- 軸ラベルは表示単位 (mm/µm) に追従。値表示は formatNumber。

### 保存/読込 (結果付き保存)

- types.ts の ResultsBundle に `pic1d?: Pic1dResult` を追加。
- App.tsx の 結果付き保存 (buildResults) と applyLoadedProject に pic1d 結果の
  保存・復元を追加 (既存 pic/gas の流儀)。復元時は result-pic1d ノードが活性になる。
- 「結果付き保存」ボタンの有効条件に pic1d 結果を追加。

### 実行状態の整合

- pic1d 実行中は他のソルバー実行ボタンと同様に排他 (anyRunning に pic1dRunning を追加)。
- pic1d 設定変更 (commitProject) で pic1d 結果を破棄するか → **破棄しない**
  (既存 PIC の流儀を読んで合わせる。既存が破棄するなら合わせる)。

## 検証

- `cd /home/claude/ES-Sim/frontend && npx tsc --noEmit && npx vite build` が通ること
  (必ず frontend ディレクトリから実行)。
- バックエンドは変更しない (万一プロトコル齟齬を見つけたら報告のみ)。

## 注意

- コメントは日本語で「なぜ」。git commit はしない。
- Math.min/max のスプレッドは使わない → mathUtils の arrayMin/arrayMax を使う
  (節点配列は大きくなり得る)。
- 生の onChange 数値入力は使わない → CommitNumberInput / CommitNullableNumberInput。
- App.tsx は巨大なので編集は最小差分・既存パターンの模倣を最優先。
