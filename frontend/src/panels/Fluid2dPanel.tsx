import { useRef, useState } from "react";
import { api } from "../api";
import { CommitNullableNumberInput, CommitNumberInput, formatNumber } from "../CommitInput";
import { Toggle } from "../Toggle";
import { ProcessList } from "./PicPanel";
import { rfComponents } from "../types";
import type {
  Fluid1dSettings,
  Fluid2dCycle,
  Fluid2dFrameMsg,
  Fluid2dResult,
  Fluid2dSettings,
  Fluid2dStartedMsg,
  MeshResult,
  Point,
  Project,
} from "../types";

/**
 * 2D/軸対称 プラズマ流体 (ドリフト拡散 + 電子エネルギー、EAFE/FEM-SG) スタディの
 * 設定・実行パネル (prompts/111-113)。fluid1d と異なりジオメトリ・メッシュ・境界条件
 * (電極電圧/RF/CSV波形/SEE) は既存のプロジェクト設定をそのまま使う (2D PIC と同一条件で
 * 比較できる設計目標) ため、このパネルではガス・初期値・断面積・実行設定のみを編集する。
 * 結果のフィールド表示 (キャンバス側) は既存の CadCanvas 汎用機構 (picFieldView/picFrame と
 * 同じ配線) に載せるため、App.tsx が state を持ち CadCanvas へ配線する — このパネルは
 * PicPanel と同じ役割分担で「選択状態の入力 UI」のみを担当する。
 */

// 「結果表示」セレクトの選択肢。"live" はライブ (最終フレーム) 表示、それ以外は
// done メッセージの fields から時間平均フィールドをカラーマップで描画する対象を表す
export type Fluid2dResultField = "live" | "phi" | "e_abs" | "n_e" | "n_i" | "t_e" | "ionization";

export const FLUID2D_FIELD_OPTIONS: { value: Fluid2dResultField; label: string }[] = [
  { value: "live", label: "ライブ (最終フレーム)" },
  { value: "phi", label: "電位 φ [V]" },
  { value: "e_abs", label: "|E| [V/m]" },
  { value: "n_e", label: "電子密度 [m^-3]" },
  { value: "n_i", label: "イオン密度 [m^-3]" },
  { value: "t_e", label: "電子温度 [eV]" },
  { value: "ionization", label: "電離レート [m^-3 s^-1]" },
];

// フィールドキーごとの節点/要素の別・単位 (App 側で CadCanvas 用の picFieldView 構築に使う)。
// e_abs のみ要素値 (fluid2d.py の e_abs は要素定数の E ベクトルの絶対値)、他は全節点値
export const FLUID2D_FIELD_META: Record<Exclude<Fluid2dResultField, "live">, { unit: string; nodeBased: boolean }> = {
  phi: { unit: "V", nodeBased: true },
  e_abs: { unit: "V/m", nodeBased: false },
  n_e: { unit: "m^-3", nodeBased: true },
  n_i: { unit: "m^-3", nodeBased: true },
  t_e: { unit: "eV", nodeBased: true },
  ionization: { unit: "m^-3 s^-1", nodeBased: true },
};

// ライブモニタ (実行中のキャンバス表示) で切替可能なフィールド。frame は phi/n_e/n_i/t_e の
// 全節点値を毎回含むため (PIC の PicLiveField と異なり n_e/n_i も節点値、fluid2d.py 参照)
export type Fluid2dLiveField = "phi" | "n_e" | "n_i" | "t_e";

export const FLUID2D_LIVE_FIELD_OPTIONS: { value: Fluid2dLiveField; label: string }[] = [
  { value: "phi", label: "電位 φ [V]" },
  { value: "n_e", label: "電子密度 n_e [m^-3]" },
  { value: "n_i", label: "イオン密度 n_i [m^-3]" },
  { value: "t_e", label: "電子温度 T_e [eV]" },
];

// 位相アニメーションで表示可能なフィールド。cycle は phi/n_e/n_i/t_e のみを持つ (e_abs/ionization は無い)
export type Fluid2dCycleField = "phi" | "n_e" | "n_i" | "t_e";

export const FLUID2D_CYCLE_FIELD_OPTIONS: { value: Fluid2dCycleField; label: string }[] = [
  { value: "phi", label: "電位 [V]" },
  { value: "n_e", label: "電子密度 [m^-3]" },
  { value: "n_i", label: "イオン密度 [m^-3]" },
  { value: "t_e", label: "電子温度 [eV]" },
];

// 実行時間内訳のキー→表示ラベル (fluid2d.py の timing 辞書のキーと一致させる)
const TIMING_PHASE_LABELS: Record<string, string> = {
  poisson: "Poisson (電場)",
  transport: "輸送 (イオン/電子)",
  energy: "電子エネルギー",
  other: "その他",
};

interface Props {
  project: Project;
  fluid2d: Fluid2dSettings;
  onChange: (next: Fluid2dSettings) => void;
  canRun: boolean;
  running: boolean;
  onStart: () => void;
  onStop: () => void;
  // 「続きから実行」ボタン: 直前の実行が done/stop 済みで現在実行中でなく、かつ前回実行以降に
  // ジオメトリが編集されていない場合のみ true (App 側で判定する、2D PIC の picCanContinue と同じ設計)
  canContinue: boolean;
  onContinue: (extraSteps: number) => void;
  // true の場合、canContinue=false の理由が「ジオメトリ編集による食い違い」であることを示す
  continueDisabledByProjectChange: boolean;
  started: Fluid2dStartedMsg | null;
  frame: Fluid2dFrameMsg | null;
  error: string | null;
  // 「流体 (1D) の設定を取込」プリセット用。App が現在の fluid1d state をそのまま渡す
  fluid1d: Fluid1dSettings | null;
  // done で受け取った結果一式 (settings を含み自己完結)。数値サマリ表示に使う
  result: Fluid2dResult | null;
  // 中心付近の n_e/T_e サマリ (メッシュ重心最近傍) 用。フィールド表示に使うメッシュと同じもの
  meshResult: MeshResult | null;

  // 「結果表示」セレクトの現在値と対数スケール (App 側で保持・CadCanvas に反映)
  resultField: Fluid2dResultField;
  onResultFieldChange: (v: Fluid2dResultField) => void;
  logScale: boolean;
  onLogScaleChange: (v: boolean) => void;

  // ライブ表示 (実行中のキャンバス色マップ) の選択フィールドと対数スケール
  liveField: Fluid2dLiveField;
  onLiveFieldChange: (v: Fluid2dLiveField) => void;
  liveLogScale: boolean;
  onLiveLogScaleChange: (v: boolean) => void;

  // done で受信した RF 1周期の位相分解データ (RFなし/phase_bins=0 では null)
  cycle: Fluid2dCycle | null;
  cycleField: Fluid2dCycleField;
  onCycleFieldChange: (v: Fluid2dCycleField) => void;
  cycleLogScale: boolean;
  onCycleLogScaleChange: (v: boolean) => void;
  cyclePlaying: boolean;
  onCyclePlayingChange: (v: boolean) => void;
  cycleBinIndex: number;
  onCycleBinIndexChange: (v: number) => void;
  cycleFps: number;
  onCycleFpsChange: (v: number) => void;

  // 表示モード: "all"=従来通り全表示、"setup"=設定/実行UIのみ、"results"=結果表示のみ
  // (PicPanel と同じ設計。study-fluid2d/result-fluid2d の2インスタンスで使い分ける)
  mode?: "all" | "setup" | "results";
}

// project.geometry.boundaries / regions から RF 周波数を集める (backend fluid2d.py
// Fluid2dSimulation._find_rf_freq と同じ規則: Dirichlet 境界 + conductor 領域の
// voltage_rf/voltage_waveform の両方を対象にする。PicPanel.collectRfFrequencies は
// boundaries のみを見るため、conductor 領域の RF 電極も持つ fluid2d 用に別途用意する)
function collectFluid2dFrequencies(project: Project): number[] {
  const freqs = new Set<number>();
  for (const bc of project.geometry.boundaries) {
    if (bc.type !== "dirichlet") continue;
    for (const rf of rfComponents(bc.voltage_rf)) if (rf.freq_hz > 0) freqs.add(rf.freq_hz);
    if (bc.voltage_waveform && bc.voltage_waveform.freq_hz > 0) freqs.add(bc.voltage_waveform.freq_hz);
  }
  for (const region of project.geometry.regions) {
    if (region.type !== "conductor") continue;
    for (const rf of rfComponents(region.voltage_rf)) if (rf.freq_hz > 0) freqs.add(rf.freq_hz);
    if (region.voltage_waveform && region.voltage_waveform.freq_hz > 0) freqs.add(region.voltage_waveform.freq_hz);
  }
  return Array.from(freqs).sort((a, b) => a - b);
}

// サイクル数の表示整形 (PicPanel.formatCycles と同じ)
function formatCycles(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  return v.toPrecision(3).replace("e+", "e");
}

// 「流体 (1D) の設定を取込」(prompts/113): fluid1d 設定からガス・初期値・断面積のみを写す。
// ジオメトリ・メッシュ・電極電圧は fluid2d が既存のプロジェクト設定を直接使うため対象外
// (fluid1d 側の gap_m/n_cells/left/right は fluid2d に対応するフィールドが無い)。
// n_steps 等の実行設定・μ_i 基準値/T_i は共通フィールドだが流体固有の調整済み値の可能性が
// あるため維持し、ガス・初期値・断面積 (fluid1d でも fluid2d でも意味が同じ物理量) だけを写す
function applyFluid1dPreset(fluid1d: Fluid1dSettings, current: Fluid2dSettings): Fluid2dSettings {
  return {
    ...current,
    init_density_m3: fluid1d.init_density_m3,
    init_te_ev: fluid1d.init_te_ev,
    gas_pressure_pa: fluid1d.gas_pressure_pa,
    gas_temperature_k: fluid1d.gas_temperature_k,
    ion_mass_amu: fluid1d.ion_mass_amu,
    mu_i_ref: fluid1d.mu_i_ref,
    n_ref_m3: fluid1d.n_ref_m3,
    t_i_ev: fluid1d.t_i_ev,
    electron_processes: fluid1d.electron_processes,
  };
}

// ドメイン多角形の頂点平均 (簡易重心)。中心付近の n_e/T_e サマリ用のメッシュ最近傍探索の
// 基準点として使う (「メッシュ重心最近傍でよい」という prompts/113 の指示どおり厳密な
// 面重心ではなく頂点平均で十分とする)
function domainCentroid(project: Project): Point | null {
  const poly = project.geometry.domain.polygon;
  if (!poly || poly.length === 0) return null;
  let sx = 0;
  let sy = 0;
  for (const [x, y] of poly) {
    sx += x;
    sy += y;
  }
  return [sx / poly.length, sy / poly.length];
}

// 指定点に最も近いメッシュ節点のインデックスを返す (総当たり。節点数は数千程度なので十分高速)
function nearestNodeIndex(nodes: Point[], x: number, y: number): number | null {
  if (nodes.length === 0) return null;
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < nodes.length; i++) {
    const dx = nodes[i][0] - x;
    const dy = nodes[i][1] - y;
    const d = dx * dx + dy * dy;
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

// 位相アニメーションプレイヤー (PicPanel.PicCyclePlayer の流儀を踏襲した簡略版)。
// fluid2d の cycle は freq_hz を持つ (PicCycle の period_s とは異なるキー) ため周期を
// 1/freq_hz から求める点、粒子スナップショットが無いため showParticles トグルが無い点が差分。
// PicCyclePlayer 自体は非 export のため (型 (PicCycle 専用) も異なる)、流用ではなく
// 同じ UI/UX を再実装する (prompts/113: 「困難なら第1弾はスライダのみでも可」の判断として、
// 完全な部品流用ではなく同型の専用コンポーネントを用意した)
function Fluid2dCyclePlayer({
  cycle,
  field,
  onFieldChange,
  logScale,
  onLogScaleChange,
  playing,
  onPlayingChange,
  binIndex,
  onBinIndexChange,
  fps,
  onFpsChange,
}: {
  cycle: Fluid2dCycle;
  field: Fluid2dCycleField;
  onFieldChange: (v: Fluid2dCycleField) => void;
  logScale: boolean;
  onLogScaleChange: (v: boolean) => void;
  playing: boolean;
  onPlayingChange: (v: boolean) => void;
  binIndex: number;
  onBinIndexChange: (v: number) => void;
  fps: number;
  onFpsChange: (v: number) => void;
}) {
  const bin = Math.min(binIndex, cycle.bins - 1);
  const periodS = 1 / cycle.freq_hz;
  const phaseDeg = (bin / cycle.bins) * 360;
  const tInBin = (bin / cycle.bins) * periodS;

  return (
    <>
      <h2>流体 (2D): 周期アニメーション</h2>
      <div className="field">
        <span className="label">表示フィールド</span>
        <select value={field} onChange={(e) => onFieldChange(e.target.value as Fluid2dCycleField)}>
          {FLUID2D_CYCLE_FIELD_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
      </div>
      <Toggle label="対数スケール" checked={logScale} onChange={onLogScaleChange} />

      <div className="actions">
        <button className="secondary" onClick={() => onPlayingChange(!playing)}>
          {playing ? "一時停止" : "再生"}
        </button>
        <select value={fps} onChange={(e) => onFpsChange(Number(e.target.value))}>
          <option value={5}>5 fps</option>
          <option value={10}>10 fps</option>
          <option value={20}>20 fps</option>
        </select>
      </div>

      <div className="field">
        <span className="label">位相 (bin {bin + 1}/{cycle.bins})</span>
        <input
          type="range"
          min={0}
          max={cycle.bins - 1}
          step={1}
          value={bin}
          onChange={(e) => {
            onPlayingChange(false); // スライダー操作で明示的に位相を選んだら再生は止める
            onBinIndexChange(Number(e.target.value));
          }}
        />
      </div>
      <p className="hint">
        位相角 {phaseDeg.toFixed(1)}° / ビン内時刻 {(tInBin * 1e9).toFixed(2)} ns
        (周期 {(periodS * 1e9).toFixed(2)} ns、基本周波数 {formatNumber(cycle.freq_hz)} Hz)
      </p>
    </>
  );
}

export default function Fluid2dPanel({
  project,
  fluid2d,
  onChange,
  canRun,
  running,
  onStart,
  onStop,
  canContinue,
  onContinue,
  continueDisabledByProjectChange,
  started,
  frame,
  error,
  fluid1d,
  result,
  meshResult,
  resultField,
  onResultFieldChange,
  logScale,
  onLogScaleChange,
  liveField,
  onLiveFieldChange,
  liveLogScale,
  onLiveLogScaleChange,
  cycle,
  cycleField,
  onCycleFieldChange,
  cycleLogScale,
  onCycleLogScaleChange,
  cyclePlaying,
  onCyclePlayingChange,
  cycleBinIndex,
  onCycleBinIndexChange,
  cycleFps,
  onCycleFpsChange,
  mode = "all",
}: Props) {
  const show = (m: "setup" | "results") => mode === "all" || mode === m;

  // --- 電子断面積の LXCat インポート (電子のみ、Fluid1dPanel の同名処理の複製) ---
  const [lxcatWarnings, setLxcatWarnings] = useState<string[]>([]);
  const [lxcatError, setLxcatError] = useState<string | null>(null);
  const electronFileRef = useRef<HTMLInputElement>(null);
  const importLxcat = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result ?? "");
      api
        .lxcatParse(text, "electron")
        .then((res) => {
          setLxcatError(null);
          setLxcatWarnings(res.warnings);
          onChange({ ...fluid2d, electron_processes: res.processes });
        })
        .catch((e) => setLxcatError(String(e)));
    };
    reader.readAsText(file);
  };

  // --- RFサイクル換算 (Fluid1dPanel と同じ考え方だが、周波数はプロジェクトの境界条件/領域から) ---
  const rfFrequencies = collectFluid2dFrequencies(project);
  const effectiveDt = fluid2d.dt ?? started?.dt ?? null;
  const dtFromStarted = fluid2d.dt == null && started != null;
  const nSteps = fluid2d.n_steps ?? 20000;

  // --- 実行進捗 (続き実行では frame.step が通算のため、区間開始オフセットを引く) ---
  const segOffset = started?.step_offset ?? 0;
  const segStep = Math.max(0, (frame?.step ?? segOffset) - segOffset);
  const progressPct = started && started.n_steps > 0 ? Math.min(100, (segStep / started.n_steps) * 100) : 0;

  // 「続きから」の追加ステップ数。既定は現在の n_steps 設定を踏襲するが、以後は独立に編集できる
  const [extraSteps, setExtraSteps] = useState(nSteps);

  // 中心付近の n_e/T_e サマリ (メッシュ重心最近傍、prompts/113)
  const centroid = domainCentroid(project);
  const centerIdx =
    meshResult && centroid ? nearestNodeIndex(meshResult.nodes, centroid[0], centroid[1]) : null;
  const centerNe = result?.fields && centerIdx != null ? result.fields.n_e[centerIdx] : null;
  const centerTe = result?.fields && centerIdx != null ? result.fields.t_e[centerIdx] : null;

  return (
    <>
      {show("setup") && (
        <>
      <h2>流体 (2D): プリセット</h2>
      <div className="actions">
        <button
          className="secondary"
          onClick={() => fluid1d && onChange(applyFluid1dPreset(fluid1d, fluid2d))}
          disabled={!fluid1d}
          title={
            fluid1d
              ? "現在の流体 (1D) 設定からガス・初期値・断面積を取込みます (確認なし、Undo 対象外)"
              : "流体 (1D) が未設定です"
          }
        >
          流体 (1D) の設定を取込
        </button>
      </div>
      <p className="hint">
        ジオメトリ・メッシュ・電極電圧 (RF/CSV/SEE) は既存のプロジェクト設定 (境界条件) を使います。
        現在の流体 (1D) 設定からガス・初期密度/Te・電子断面積のみを取込みます。
      </p>

      <h2>流体 (2D): 初期値</h2>
      <div className="field">
        <span className="label">初期密度 [m^-3]</span>
        <CommitNumberInput
          value={fluid2d.init_density_m3}
          onCommit={(v) => onChange({ ...fluid2d, init_density_m3: v })}
        />
      </div>
      <div className="field">
        <span className="label">初期 Te [eV]</span>
        <CommitNumberInput
          value={fluid2d.init_te_ev ?? 2.0}
          onCommit={(v) => onChange({ ...fluid2d, init_te_ev: v })}
        />
      </div>

      <h2>流体 (2D): ガス</h2>
      <div className="field">
        <span className="label">圧力 [Pa]</span>
        <CommitNumberInput
          value={fluid2d.gas_pressure_pa}
          onCommit={(v) => onChange({ ...fluid2d, gas_pressure_pa: v })}
        />
      </div>
      <div className="field">
        <span className="label">ガス温度 [K]</span>
        <CommitNumberInput
          value={fluid2d.gas_temperature_k ?? 300.0}
          onCommit={(v) => onChange({ ...fluid2d, gas_temperature_k: v })}
        />
      </div>
      <div className="field">
        <span className="label">イオン質量 [amu]</span>
        <CommitNumberInput
          value={fluid2d.ion_mass_amu ?? 39.948}
          onCommit={(v) => onChange({ ...fluid2d, ion_mass_amu: v })}
        />
      </div>
      <div className="field">
        <span className="label">イオン温度 T_i [eV]</span>
        <CommitNumberInput value={fluid2d.t_i_ev ?? 0.026} onCommit={(v) => onChange({ ...fluid2d, t_i_ev: v })} />
      </div>
      <div className="field">
        <span className="label">μ_i 基準値 [m^2/(V・s)]</span>
        <CommitNumberInput
          value={fluid2d.mu_i_ref ?? 1.45e-1}
          onCommit={(v) => onChange({ ...fluid2d, mu_i_ref: v })}
        />
      </div>
      <div className="field">
        <span className="label">μ_i 基準ガス密度 [m^-3]</span>
        <CommitNumberInput
          value={fluid2d.n_ref_m3 ?? 3.22e22}
          onCommit={(v) => onChange({ ...fluid2d, n_ref_m3: v })}
        />
      </div>

      <h2>流体 (2D): 電子断面積</h2>
      <p className="hint">
        空 (未読込) の場合は eduPIC の Ar 解析式 (弾性・励起・電離) を既定で使用します
        (PicPanel の MCC 断面積編集と同じ LXCat 形式のインポートに対応)。
      </p>
      <div className="actions">
        <button className="secondary" onClick={() => electronFileRef.current?.click()}>
          電子断面積を読込
        </button>
        <input
          ref={electronFileRef}
          type="file"
          accept=".txt,text/plain"
          className="file-input"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) importLxcat(f);
            e.target.value = "";
          }}
        />
      </div>
      {lxcatError && <div className="error">{lxcatError}</div>}
      {lxcatWarnings.length > 0 && (
        <div className="pic-warnings">
          {lxcatWarnings.map((w, i) => (
            <div key={i}>警告: {w}</div>
          ))}
        </div>
      )}
      <p className="hint">電子プロセス ({(fluid2d.electron_processes ?? []).length})</p>
      <ProcessList processes={fluid2d.electron_processes ?? []} />
      <div className="actions">
        <button className="secondary" onClick={() => onChange({ ...fluid2d, electron_processes: [] })}>
          電子プロセスをクリア (eduPIC Ar 解析式に戻す)
        </button>
      </div>

      <h2>流体 (2D): 実行設定</h2>
      <div className="field">
        <span className="label">dt [s] (空欄=自動)</span>
        <CommitNullableNumberInput
          value={fluid2d.dt ?? null}
          placeholder="自動"
          onCommit={(v) => onChange({ ...fluid2d, dt: v })}
        />
      </div>
      <div className="field">
        <span className="label">ステップ数</span>
        <CommitNumberInput
          value={nSteps}
          onCommit={(v) => onChange({ ...fluid2d, n_steps: Math.max(1, Math.round(v)) })}
        />
      </div>
      {rfFrequencies.length > 0 &&
        (effectiveDt === null ? (
          <p className="hint">dt が自動のため、RFサイクル換算は実行開始後に確定します</p>
        ) : (
          rfFrequencies.map((f) => {
            const periodS = 1 / f;
            const stepsPerCycle = periodS / effectiveDt;
            const cycles = nSteps / stepsPerCycle;
            return (
              <p className="hint" key={f}>
                {formatNumber(f)} Hz: {nSteps} ステップ ≈ {formatCycles(cycles)} RFサイクル
                (1サイクル ≈ {stepsPerCycle.toFixed(1)} ステップ)
                {dtFromStarted && " (前回実行の dt で換算)"}
              </p>
            );
          })
        ))}
      <div className="field">
        <span className="label">フレーム間隔</span>
        <CommitNumberInput
          value={fluid2d.frame_every ?? 200}
          onCommit={(v) => onChange({ ...fluid2d, frame_every: Math.max(1, Math.round(v)) })}
        />
      </div>
      <div className="field">
        <span className="label">平均ステップ数 (空欄=最後の25%)</span>
        <CommitNullableNumberInput
          value={fluid2d.avg_steps ?? null}
          placeholder="最後の25%"
          onCommit={(v) => onChange({ ...fluid2d, avg_steps: v == null ? null : Math.max(1, Math.round(v)) })}
        />
      </div>
      <div className="field">
        <span className="label">位相ビン数 (周期アニメ用、0=無効)</span>
        <CommitNumberInput
          value={fluid2d.phase_bins ?? 0}
          onCommit={(v) => onChange({ ...fluid2d, phase_bins: Math.max(0, Math.round(v)) })}
        />
      </div>
      {rfFrequencies.length > 0 && effectiveDt !== null && (fluid2d.phase_bins ?? 0) > 0 && (() => {
        const basePeriod = 1 / rfFrequencies[0];
        const stepsPerCycle = basePeriod / effectiveDt;
        const maxBins = Math.max(1, Math.floor(stepsPerCycle));
        const over = (fluid2d.phase_bins ?? 0) > maxBins;
        return (
          <p className="hint" style={over ? { color: "#e0b050" } : undefined}>
            目安: 基本波1周期 ≈ {stepsPerCycle.toFixed(1)} ステップのため、ビン数は {maxBins} 以下を推奨
            {over && " (現在の設定では空のビンが生じる可能性があります)"}。
            また平均ステップ数は1周期分 ({Math.ceil(stepsPerCycle)}) 以上、推奨は3周期分 ({3 * Math.ceil(stepsPerCycle)}) 以上。
            {dtFromStarted && " (前回実行の dt で換算)"}
          </p>
        );
      })()}

      <h2>流体 (2D): 実行</h2>
      <p className="hint">
        メッシュ未生成でも実行できます (開始時にジオメトリから自動生成し、既存の /mesh 結果と
        同じメッシュを使います)。ジオメトリ・境界条件は「ジオメトリ」配下のノードで編集してください。
      </p>
      <div className="actions">
        <button onClick={onStart} disabled={!canRun || running}>
          {running ? "実行中..." : "流体2D開始"}
        </button>
        <button className="secondary" onClick={onStop} disabled={!running}>
          停止
        </button>
        <button
          className="secondary"
          onClick={() => onContinue(extraSteps)}
          disabled={!canContinue}
          title={
            continueDisabledByProjectChange
              ? "ジオメトリ・プラズマ設定が変更されたため続き実行できません (再度「流体2D開始」してください)"
              : undefined
          }
        >
          続きから (+{extraSteps}ステップ)
        </button>
      </div>
      <div className="field">
        <span className="label">続きから: +ステップ</span>
        <CommitNumberInput value={extraSteps} onCommit={(v) => setExtraSteps(Math.max(1, Math.round(v)))} />
      </div>
      <p className="hint">
        「続きから」は現在保持中の場・時刻から追加実行します
        (フレーム間隔・平均ステップ数・位相ビン数は現在の設定値を使用)。
        ジオメトリ・プラズマ設定の変更は続き実行には反映されません。
        {continueDisabledByProjectChange &&
          " ジオメトリ・プラズマ設定を編集したため、続き実行するには再度「流体2D開始」が必要です。"}
      </p>

      {started && (
        <>
          <div className="pic-progress">
            <div className="pic-progress-bar" style={{ width: `${progressPct}%` }} />
          </div>
          <div className="kv">
            <span>進捗</span>
            <span>{segStep} / {started.n_steps}</span>
          </div>
          {started.warnings.length > 0 && (
            <div className="pic-warnings">
              {started.warnings.map((w, i) => (
                <div key={i}>警告: {w}</div>
              ))}
            </div>
          )}
          {/* ライブモニタの表示フィールド切替 (2D PIC の prompts/81 と同じ考え方)。
              実行中でも即座にキャンバスの色マップを切り替えられる */}
          <div className="field">
            <span className="label">ライブ表示</span>
            <select value={liveField} onChange={(e) => onLiveFieldChange(e.target.value as Fluid2dLiveField)}>
              {FLUID2D_LIVE_FIELD_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
          {liveField !== "phi" && (
            <Toggle label="対数スケール" checked={liveLogScale} onChange={onLiveLogScaleChange} />
          )}
        </>
      )}

      {frame && (
        <>
          <h2>流体 (2D): 診断</h2>
          <div className="kv">
            <span>全域密度 電子/イオン [積算]</span>
            <span>{frame.counts.n_e_total ?? "-"} / {frame.counts.n_i_total ?? "-"}</span>
          </div>
          <div className="kv">
            <span>壁吸収 (電子/イオン、累計)</span>
            <span>{frame.counts.wall_e ?? "-"} / {frame.counts.wall_i ?? "-"}</span>
          </div>
          <div className="kv">
            <span>電離生成 (累計)</span>
            <span>{frame.counts.gen_total ?? "-"}</span>
          </div>
        </>
      )}
      </>
      )}

      {show("results") && result?.fields && (
        <>
          <h2>流体 (2D): 結果フィールド</h2>
          <p className="hint">時間平均ステップ数: {result.fields.avg_steps}</p>
          <div className="field">
            <span className="label">結果表示</span>
            <select
              value={resultField}
              onChange={(e) => onResultFieldChange(e.target.value as Fluid2dResultField)}
            >
              {FLUID2D_FIELD_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
          {resultField !== "live" && (
            <Toggle label="対数スケール" checked={logScale} onChange={onLogScaleChange} />
          )}
        </>
      )}

      {show("results") && cycle && (
        <Fluid2dCyclePlayer
          cycle={cycle}
          field={cycleField}
          onFieldChange={onCycleFieldChange}
          logScale={cycleLogScale}
          onLogScaleChange={onCycleLogScaleChange}
          playing={cyclePlaying}
          onPlayingChange={onCyclePlayingChange}
          binIndex={cycleBinIndex}
          onBinIndexChange={onCycleBinIndexChange}
          fps={cycleFps}
          onFpsChange={onCycleFpsChange}
        />
      )}

      {show("results") && result && (
        <>
          <h2>流体 (2D): 数値サマリ</h2>
          <div className="kv">
            <span>経過時間 (壁時計)</span>
            <span>{result.elapsed_s.toFixed(3)} s</span>
          </div>
          {Object.entries(result.timing)
            .filter(([k]) => k !== "total")
            .sort((a, b) => b[1] - a[1])
            .map(([key, sec]) => (
              <div className="kv" key={key}>
                <span>{TIMING_PHASE_LABELS[key] ?? key}</span>
                <span>
                  {sec.toFixed(3)} s (
                  {result.timing.total > 0 ? ((100 * sec) / result.timing.total).toFixed(1) : "0.0"}%)
                </span>
              </div>
            ))}
          <div className="kv">
            <span>実行時間合計</span>
            <span>{(result.timing.total ?? 0).toFixed(3)} s</span>
          </div>
          <div className="kv">
            <span>壁吸収 (電子/イオン、累計)</span>
            <span>{result.walls.electron.toExponential(3)} / {result.walls.ion.toExponential(3)}</span>
          </div>
          <div className="kv">
            <span>電離生成 (累計)</span>
            <span>{result.gen_total.toExponential(3)}</span>
          </div>
          <div className="kv">
            <span>中心付近 n_e [m^-3] (メッシュ重心最近傍)</span>
            <span>{centerNe != null ? centerNe.toExponential(3) : "-"}</span>
          </div>
          <div className="kv">
            <span>中心付近 T_e [eV] (メッシュ重心最近傍)</span>
            <span>{centerTe != null ? centerTe.toFixed(3) : "-"}</span>
          </div>
        </>
      )}

      {mode === "results" && !result && (
        <p className="hint">流体 (2D) 計算が未実行です。スタディ「流体 (2D)」から実行してください。</p>
      )}

      {error && (
        <>
          <h2>流体 (2D) エラー</h2>
          <div className="error">{error}</div>
        </>
      )}
    </>
  );
}

// project.fluid2d が未設定の場合の初期表示に使う既定値 (App が使う)。fluid1d と同じく
// particles/pic とは独立の state で管理する想定 (geometry/mesh は既存のものをそのまま使う)。
// init_density_m3/gas_pressure_pa は backend に既定値が無い必須項目のため fluid1d の
// DEFAULT_FLUID1D と同じ値に揃える。それ以外は backend/es_sim/schema.py Fluid2dSettings の
// Field 既定値と一致させる (phase_bins は fluid1d と異なり既定0)
export const DEFAULT_FLUID2D: Fluid2dSettings = {
  init_density_m3: 1.0e15,
  init_te_ev: 2.0,
  gas_pressure_pa: 50.0,
  gas_temperature_k: 300.0,
  ion_mass_amu: 39.948,
  mu_i_ref: 1.45e-1,
  n_ref_m3: 3.22e22,
  t_i_ev: 0.026,
  electron_processes: [],
  dt: null,
  n_steps: 20000,
  frame_every: 200,
  avg_steps: null,
  phase_bins: 0,
};
