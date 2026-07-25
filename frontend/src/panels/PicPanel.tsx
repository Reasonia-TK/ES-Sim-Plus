import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { saveTextFile } from "../saveFile";
import { CommitNullableNumberInput, CommitNumberInput, CommitTextInput, formatNumber } from "../CommitInput";
import { Toggle } from "../Toggle";
import FnEmissionSection from "./FnPanel";
import { LENGTH_UNIT_LABEL, mToUnit, unitToM } from "../units";
import type { LengthUnit } from "../units";
import { isAxisymmetric, rfComponents } from "../types";
import type {
  Emitter,
  FnEmission,
  InitialPlasma,
  McSettings,
  PicCollectorResult,
  PicCollectorSettings,
  PicCycle,
  PicDiag,
  PicFields,
  PicFrameMsg,
  PicInjection,
  PicMerge,
  PicSettings,
  PicStartedMsg,
  Project,
  XsProcess,
} from "../types";

// 周期アニメーションで表示可能なフィールド。e_abs のみ要素値、他は節点値
export type CyclePicField = "phi" | "n_e" | "n_i" | "e_abs" | "te_ev" | "ion_rate";

export const CYCLE_FIELD_OPTIONS: { value: CyclePicField; label: string }[] = [
  { value: "phi", label: "電位 [V]" },
  { value: "n_e", label: "電子密度 [m^-3]" },
  { value: "n_i", label: "イオン密度 [m^-3]" },
  { value: "e_abs", label: "|E| [V/m]" },
  { value: "te_ev", label: "電子温度 [eV]" },
  { value: "ion_rate", label: "電離レート [m^-3 s^-1]" },
];

// PIC「結果表示」セレクトの選択肢。"live" はライブ (最終フレーム) 表示、それ以外は
// done メッセージの fields から時間平均フィールドをカラーマップで描画する対象を表す
export type PicResultField = "live" | "phi" | "e_abs" | "n_e" | "n_i" | "te_ev" | "ion_rate";

export const PIC_FIELD_OPTIONS: { value: PicResultField; label: string }[] = [
  { value: "live", label: "ライブ (最終フレーム)" },
  { value: "phi", label: "電位 [V]" },
  { value: "e_abs", label: "|E| [V/m]" },
  { value: "n_e", label: "電子密度 [m^-3]" },
  { value: "n_i", label: "イオン密度 [m^-3]" },
  { value: "te_ev", label: "電子温度 [eV]" },
  { value: "ion_rate", label: "電離レート [m^-3 s^-1]" },
];

// フィールドキーごとの節点/要素の別・単位 (App 側で CadCanvas 用の picFieldView 構築に使う)
export const PIC_FIELD_META: Record<Exclude<PicResultField, "live">, { unit: string; nodeBased: boolean }> = {
  phi: { unit: "V", nodeBased: true },
  e_abs: { unit: "V/m", nodeBased: false },
  n_e: { unit: "m^-3", nodeBased: true },
  n_i: { unit: "m^-3", nodeBased: true },
  te_ev: { unit: "eV", nodeBased: true },
  ion_rate: { unit: "m^-3 s^-1", nodeBased: true },
};

/**
 * PICパネル
 * - 初期プラズマ装荷 / エミッタ定常注入 / マクロ粒子数・積分設定の編集
 * - project.pic として保存/読込対象だが、particles と同様 Undo/Redo 履歴には積まない
 * - 実行制御 (開始/停止・進捗・警告) と診断表示 (数値 + 履歴チャート)
 */

interface Props {
  project: Project;
  // 長さの表示・入力単位 (mm/µm)。project 内部は常に m のまま
  lengthUnit: LengthUnit;
  pic: PicSettings;
  onChange: (next: PicSettings) => void;
  // フェーズ2 (粒子) パネルの現在のエミッタ設定。injection.emitter として共用する
  emitter: Emitter;
  canRun: boolean;
  running: boolean;
  onStart: () => void;
  onStop: () => void;
  // 「続きから実行」ボタン: 直前の実行が done/stop 済みで現在実行中でなく、かつ前回実行以降に
  // ジオメトリが編集されていない場合のみ true (App 側で判定する)
  canContinue: boolean;
  onContinue: () => void;
  // true の場合、canContinue=false の理由が「ジオメトリ編集による食い違い」であることを示す
  // (ヒント文言の出し分けに使う)
  continueDisabledByProjectChange: boolean;
  started: PicStartedMsg | null;
  frame: PicFrameMsg | null;
  history: PicDiag[];
  error: string | null;
  // done メッセージで受け取った時間平均フィールド一式 (未受信 or 未対応バックエンドでは null)
  fields: PicFields | null;
  // done メッセージで受け取った位相別プロファイル計測 (prompts/75)。
  // continue では区間分のみに置き換わる。未受信 or 未対応バックエンドでは null
  timing: Record<string, number> | null;
  // 「結果表示」セレクトの現在値と対数スケールチェックボックスの状態 (App 側で保持・CadCanvas に反映)
  resultField: PicResultField;
  onResultFieldChange: (v: PicResultField) => void;
  logScale: boolean;
  onLogScaleChange: (v: boolean) => void;

  // done で受信した RF 1周期の位相分解データ (RFなし/phase_bins=0 では null)。
  // 新しい実行開始時に App 側で null にリセットされる
  cycle: PicCycle | null;
  cycleField: CyclePicField;
  onCycleFieldChange: (v: CyclePicField) => void;
  cycleLogScale: boolean;
  onCycleLogScaleChange: (v: boolean) => void;
  cyclePlaying: boolean;
  onCyclePlayingChange: (v: boolean) => void;
  cycleBinIndex: number;
  onCycleBinIndexChange: (v: number) => void;
  cycleFps: number;
  onCycleFpsChange: (v: number) => void;
  cycleShowParticles: boolean;
  onCycleShowParticlesChange: (v: boolean) => void;

  // done で受信した IEDF/IADF コレクタの記録結果一覧 (pic.collectors と同順)。
  // 新しい実行開始時に App 側で [] にリセットされる
  collectorResults: PicCollectorResult[];
  // コレクタ一覧で選択中のインデックス (未配置なら null)。一覧行クリック/「表示コレクタ」
  // セレクトのどちらからでも更新でき、キャンバス上の強調表示にも使われる (App 側で共有管理)
  selectedCollectorIndex: number | null;
  onSelectCollector: (index: number | null) => void;
  onUpdateCollector: (index: number, patch: Partial<PicCollectorSettings>) => void;
  onDeleteCollector: (index: number) => void;

  // 表示モード: "all"=従来通り全表示 (既定・後方互換)、"setup"=設定/実行UIのみ、
  // "results"=結果表示のみ (結果ノード用インスペクタページで使う)
  mode?: "all" | "setup" | "results";
}

const DEFAULT_INITIAL_PLASMA: InitialPlasma = {
  density: 1.0e14,
  te_ev: 2.0,
  ti_ev: 0.03,
  ion_mass_amu: 40.0,
  immobile_ions: false,
  seed: 0,
};

// MCC(背景ガス衝突)設定の既定値。有効チェックを一度オフにしても直前の値を復元できるよう保持する
const DEFAULT_MCC: McSettings = {
  gas: { name: "Ar", pressure_pa: 10.0, temperature_k: 300.0 },
  electron_processes: [],
  ion_processes: [],
  seed: 0,
};

// 粒子マージ (prompts/77) の既定値。backend/es_sim/schema.py PicMerge の既定と同じ
const DEFAULT_MERGE: PicMerge = { n_max: 100000, every: 100 };

// プロセスラベルは長いことがあるので一覧表示では短縮する (title 属性でフルテキストを見せる)
function shortLabel(label: string, max = 34): string {
  return label.length > max ? `${label.slice(0, max - 1)}…` : label;
}

// LXCatインポート済みの断面積プロセス一覧 (種別・ラベル(短縮)・閾値・点数)
function ProcessList({ processes }: { processes: XsProcess[] }) {
  if (processes.length === 0) return <div className="muted">(未読込)</div>;
  return (
    <div className="mcc-process-list">
      {processes.map((p, i) => (
        <div className="mcc-process-row" key={i} title={p.label}>
          <span className="tag">{p.kind}</span>
          <span className="mcc-process-label">{shortLabel(p.label)}</span>
          <span>{p.threshold_ev.toFixed(2)} eV</span>
          <span>{p.energy_ev.length} pts</span>
        </div>
      ))}
    </div>
  );
}

function emitterSummary(e: Emitter, lengthUnit: LengthUnit): string {
  const u = LENGTH_UNIT_LABEL[lengthUnit];
  const f = (m: number) => mToUnit(m, lengthUnit).toFixed(1);
  if (e.kind === "point") return `点 (${f(e.p1[0])}, ${f(e.p1[1])} ${u}), n=${e.n}`;
  return `ライン (${f(e.p1[0])}, ${f(e.p1[1])}) - (${f(e.p2[0])}, ${f(e.p2[1])}) ${u}, n=${e.n}`;
}

// project.geometry.boundaries の Dirichlet 境界に設定された RF 周波数を、全境界・全成分から
// 重複排除して昇順に集める (prompts/68: ステップ数のRFサイクル換算に使う)。
// 0Hz や未設定 (voltage_rf なし) は換算の意味がないので除外する。
// CSV波形 (voltage_waveform、prompts/73) も1周期でループする周波数として同様に含める
function collectRfFrequencies(project: Project): number[] {
  const freqs = new Set<number>();
  for (const bc of project.geometry.boundaries) {
    if (bc.type !== "dirichlet") continue;
    for (const rf of rfComponents(bc.voltage_rf)) {
      if (rf.freq_hz > 0) freqs.add(rf.freq_hz);
    }
    if (bc.voltage_waveform && bc.voltage_waveform.freq_hz > 0) freqs.add(bc.voltage_waveform.freq_hz);
  }
  return Array.from(freqs).sort((a, b) => a - b);
}

// サイクル数の表示整形。有効3桁程度 (toPrecision(3)) で丸め、formatNumber と同様に
// "e+" の "+" を除いて見た目を揃える (通常は 0.1〜数百程度の範囲を想定)
function formatCycles(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  return v.toPrecision(3).replace("e+", "e");
}

export default function PicPanel({
  project,
  lengthUnit,
  pic,
  onChange,
  emitter,
  canRun,
  running,
  onStart,
  onStop,
  canContinue,
  onContinue,
  continueDisabledByProjectChange,
  started,
  frame,
  history,
  error,
  fields,
  timing,
  resultField,
  onResultFieldChange,
  logScale,
  onLogScaleChange,
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
  cycleShowParticles,
  onCycleShowParticlesChange,
  collectorResults,
  selectedCollectorIndex,
  onSelectCollector,
  onUpdateCollector,
  onDeleteCollector,
  mode = "all",
}: Props) {
  // mode が "all" のときは従来通り両方表示。それ以外は該当モードのみ表示する
  const show = (m: "setup" | "results") => mode === "all" || mode === m;
  const unitLabel = LENGTH_UNIT_LABEL[lengthUnit];
  // 有効チェックを一度オフにしても、再度オンにしたときに直前の値を復元できるよう保持する
  const plasmaDefaultsRef = useRef<InitialPlasma>(pic.initial_plasma ?? DEFAULT_INITIAL_PLASMA);
  useEffect(() => {
    if (pic.initial_plasma) plasmaDefaultsRef.current = pic.initial_plasma;
  }, [pic.initial_plasma]);

  const injectionDefaultsRef = useRef<{ species: "electron" | "ion"; current_a_per_m: number }>({
    species: pic.injection?.species ?? "electron",
    current_a_per_m: pic.injection?.current_a_per_m ?? 1e-4,
  });
  useEffect(() => {
    if (pic.injection) {
      injectionDefaultsRef.current = {
        species: pic.injection.species,
        current_a_per_m: pic.injection.current_a_per_m,
      };
    }
  }, [pic.injection]);

  const updatePlasma = (patch: Partial<InitialPlasma>) => {
    if (!pic.initial_plasma) return;
    onChange({ ...pic, initial_plasma: { ...pic.initial_plasma, ...patch } });
  };

  const updateInjection = (patch: Partial<PicInjection>) => {
    if (!pic.injection) return;
    onChange({ ...pic, injection: { ...pic.injection, ...patch } });
  };

  // MCC設定。有効チェックを一度オフにしても、再度オンにしたときに直前の値を復元できるよう保持する
  const mccDefaultsRef = useRef<McSettings>(pic.mcc ?? DEFAULT_MCC);
  useEffect(() => {
    if (pic.mcc) mccDefaultsRef.current = pic.mcc;
  }, [pic.mcc]);

  const updateMcc = (patch: Partial<McSettings>) => {
    if (!pic.mcc) return;
    onChange({ ...pic, mcc: { ...pic.mcc, ...patch } });
  };
  const updateGas = (patch: Partial<McSettings["gas"]>) => {
    if (!pic.mcc) return;
    onChange({ ...pic, mcc: { ...pic.mcc, gas: { ...pic.mcc.gas, ...patch } } });
  };

  // 粒子マージ (prompts/77)。有効チェックを一度オフにしても、再度オンにしたときに
  // 直前の値を復元できるよう保持する (他の任意設定と同じ流儀)
  const mergeDefaultsRef = useRef<PicMerge>(pic.merge ?? DEFAULT_MERGE);
  useEffect(() => {
    if (pic.merge) mergeDefaultsRef.current = pic.merge;
  }, [pic.merge]);
  const updateMerge = (patch: Partial<PicMerge>) => {
    if (!pic.merge) return;
    onChange({ ...pic, merge: { ...pic.merge, ...patch } });
  };

  // FN電界放出 (prompts/46)
  const setFn = (next: FnEmission | null) => onChange({ ...pic, fn: next });

  // LXCatインポート (電子/イオン共通)。ファイルテキストを api.lxcatParse に送り、
  // 成功したら該当プロセス列を置換する。失敗時はエラー文言を表示する
  const [lxcatWarnings, setLxcatWarnings] = useState<string[]>([]);
  const [lxcatError, setLxcatError] = useState<string | null>(null);
  const electronFileRef = useRef<HTMLInputElement>(null);
  const ionFileRef = useRef<HTMLInputElement>(null);

  const importLxcat = (species: "electron" | "ion", file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result ?? "");
      api
        .lxcatParse(text, species)
        .then((res) => {
          setLxcatError(null);
          setLxcatWarnings(res.warnings);
          if (species === "electron") updateMcc({ electron_processes: res.processes });
          else updateMcc({ ion_processes: res.processes });
        })
        .catch((e) => setLxcatError(String(e)));
    };
    reader.readAsText(file);
  };

  // 続きから実行では frame.step が通算のため、区間開始オフセット (started.step_offset) を
  // 引いた区間内ステップで進捗率を出す (App のステータスバーと同じ扱い)
  const segOffset = started?.step_offset ?? 0;
  const segStep = Math.max(0, (frame?.step ?? segOffset) - segOffset);
  const progressPct = started && started.n_steps > 0 ? Math.min(100, (segStep / started.n_steps) * 100) : 0;

  // ステップ数のRFサイクル換算 (prompts/68)。RF周波数は境界条件から収集、
  // dt は pic.dt (手動指定) を優先し、未指定なら直近実行の started.dt を使う
  // (どちらもなければ dt自動かつ未実行 = 換算不能)
  const rfFrequencies = useMemo(() => collectRfFrequencies(project), [project]);
  const effectiveDt = pic.dt ?? started?.dt ?? null;
  const dtFromStarted = pic.dt === null && started != null;

  // IEDF/IADF ヒストグラムのビン数 (既定60、変更で再ビニング。コレクタ設定自体ではないので project 保存対象外)
  const [collectorBins, setCollectorBins] = useState(60);

  // コレクタCSV保存の失敗を表示するためのローカルエラー状態
  const [csvError, setCsvError] = useState<string | null>(null);

  const collectors = pic.collectors ?? [];
  // 「表示コレクタ」セレクト/一覧行クリックで選んでいるコレクタの設定・結果
  const selectedCollector = selectedCollectorIndex !== null ? collectors[selectedCollectorIndex] : undefined;
  const selectedResult = selectedCollectorIndex !== null ? collectorResults[selectedCollectorIndex] : undefined;

  // 選択中コレクタの生サンプル (energy_ev, angle_deg, weight) をCSVで保存する (ファイル名にラベルを含める)
  const downloadCollectorCsv = () => {
    if (!selectedResult) return;
    const label = selectedCollector?.label || `C${(selectedCollectorIndex ?? 0) + 1}`;
    const lines = ["energy_ev,angle_deg,weight"];
    for (let i = 0; i < selectedResult.energies_ev.length; i++) {
      lines.push(`${selectedResult.energies_ev[i]},${selectedResult.angles_deg[i]},${selectedResult.weights[i]}`);
    }
    setCsvError(null);
    saveTextFile(`iedf_iadf_${label}.csv`, lines.join("\n"), "CSV", ["csv"]).catch((err) => {
      setCsvError(String(err));
    });
  };

  return (
    <>
      {show("setup") && (
      <>
      <h2>PIC: 初期プラズマ</h2>
      <Toggle
        label="有効"
        checked={pic.initial_plasma !== null}
        onChange={(v) => onChange({ ...pic, initial_plasma: v ? plasmaDefaultsRef.current : null })}
      />
      {pic.initial_plasma && (
        <>
          <div className="field">
            <span className="label">密度 [m^-3]</span>
            <CommitNumberInput value={pic.initial_plasma.density} onCommit={(v) => updatePlasma({ density: v })} />
          </div>
          <div className="field">
            <span className="label">Te [eV]</span>
            <CommitNumberInput value={pic.initial_plasma.te_ev} onCommit={(v) => updatePlasma({ te_ev: v })} />
          </div>
          <div className="field">
            <span className="label">Ti [eV]</span>
            <CommitNumberInput value={pic.initial_plasma.ti_ev} onCommit={(v) => updatePlasma({ ti_ev: v })} />
          </div>
          <div className="field">
            <span className="label">イオン質量 [amu]</span>
            <CommitNumberInput
              value={pic.initial_plasma.ion_mass_amu}
              onCommit={(v) => updatePlasma({ ion_mass_amu: v })}
            />
          </div>
          <Toggle
            label="イオン固定"
            checked={pic.initial_plasma.immobile_ions}
            onChange={(v) => updatePlasma({ immobile_ions: v })}
          />
          <div className="field">
            <span className="label">乱数シード</span>
            <CommitNumberInput
              value={pic.initial_plasma.seed}
              onCommit={(v) => updatePlasma({ seed: Math.round(v) })}
            />
          </div>
        </>
      )}

      <h2>PIC: 注入</h2>
      <Toggle
        label="有効"
        checked={pic.injection !== null}
        onChange={(v) =>
          onChange({
            ...pic,
            injection: v
              ? {
                  emitter,
                  species: injectionDefaultsRef.current.species,
                  current_a_per_m: injectionDefaultsRef.current.current_a_per_m,
                }
              : null,
          })
        }
      />
      {pic.injection && (
        <>
          <p className="hint">
            エミッタはフェーズ2(粒子)パネルの設定を共用します: {emitterSummary(emitter, lengthUnit)}
          </p>
          <div className="field">
            <span className="label">種</span>
            <select
              value={pic.injection.species}
              onChange={(e) => updateInjection({ species: e.target.value as "electron" | "ion" })}
            >
              <option value="electron">電子</option>
              <option value="ion">イオン</option>
            </select>
          </div>
          <div className="field">
            {/* 軸対称ではリングエミッタの全電流 [A]、平面2Dでは奥行き1mあたり [A/m] */}
            <span className="label">{isAxisymmetric(project.coord) ? "電流 [A]" : "電流 [A/m]"}</span>
            <CommitNumberInput
              value={pic.injection.current_a_per_m}
              onCommit={(v) => updateInjection({ current_a_per_m: v })}
            />
          </div>
        </>
      )}

      <h2>PIC: MCC(衝突)</h2>
      <Toggle
        label="有効"
        checked={pic.mcc !== null}
        onChange={(v) => onChange({ ...pic, mcc: v ? mccDefaultsRef.current : null })}
      />
      {pic.mcc && (
        <>
          <div className="field">
            <span className="label">ガス名</span>
            <CommitTextInput value={pic.mcc.gas.name} onCommit={(v) => updateGas({ name: v })} />
          </div>
          <div className="field">
            <span className="label">圧力 [Pa]</span>
            <CommitNumberInput value={pic.mcc.gas.pressure_pa} onCommit={(v) => updateGas({ pressure_pa: v })} />
          </div>
          <div className="field">
            <span className="label">ガス温度 [K]</span>
            <CommitNumberInput
              value={pic.mcc.gas.temperature_k}
              onCommit={(v) => updateGas({ temperature_k: v })}
            />
          </div>

          <div className="actions">
            <button className="secondary" onClick={() => electronFileRef.current?.click()}>
              電子断面積を読込
            </button>
            <button className="secondary" onClick={() => ionFileRef.current?.click()}>
              イオン断面積を読込
            </button>
            <input
              ref={electronFileRef}
              type="file"
              accept=".txt,text/plain"
              className="file-input"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) importLxcat("electron", f);
                e.target.value = "";
              }}
            />
            <input
              ref={ionFileRef}
              type="file"
              accept=".txt,text/plain"
              className="file-input"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) importLxcat("ion", f);
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

          <p className="hint">電子プロセス ({pic.mcc.electron_processes.length})</p>
          <ProcessList processes={pic.mcc.electron_processes} />
          <div className="actions">
            <button className="secondary" onClick={() => updateMcc({ electron_processes: [] })}>
              電子プロセスをクリア
            </button>
          </div>

          <p className="hint">イオンプロセス ({pic.mcc.ion_processes.length})</p>
          <ProcessList processes={pic.mcc.ion_processes} />
          <div className="actions">
            <button className="secondary" onClick={() => updateMcc({ ion_processes: [] })}>
              イオンプロセスをクリア
            </button>
          </div>

          <div className="field">
            <span className="label">乱数シード</span>
            <CommitNumberInput value={pic.mcc.seed} onCommit={(v) => updateMcc({ seed: Math.round(v) })} />
          </div>

          <Toggle
            label="DSMCガス場を使用"
            checked={pic.mcc.use_dsmc_gas ?? false}
            onChange={(v) => updateMcc({ use_dsmc_gas: v })}
          />
          <p className="hint">
            直前に実行したガス流れ (DSMC) の n・T・u を背景ガスとして使う
            (圧力・温度の一様指定は無視される)
          </p>
        </>
      )}
      <div className="field">
        <span className="label">SEE初期エネルギー [eV]</span>
        <CommitNumberInput
          value={pic.see_energy_ev}
          onCommit={(v) => onChange({ ...pic, see_energy_ev: v })}
        />
      </div>

      <FnEmissionSection project={project} fn={pic.fn} onChange={setFn} mode="pic" />

      <h2>PIC: 計算設定</h2>
      <div className="field">
        <span className="label">マクロ粒子数</span>
        <CommitNumberInput
          value={pic.n_macro}
          onCommit={(v) => onChange({ ...pic, n_macro: Math.max(1, Math.round(v)) })}
        />
      </div>
      <div className="field">
        <span className="label">dt [s] (空欄=自動)</span>
        <CommitNullableNumberInput
          value={pic.dt}
          placeholder="自動"
          onCommit={(v) => onChange({ ...pic, dt: v })}
        />
      </div>
      <div className="field">
        <span className="label">ステップ数</span>
        <CommitNumberInput
          value={pic.n_steps}
          onCommit={(v) => onChange({ ...pic, n_steps: Math.max(1, Math.round(v)) })}
        />
      </div>
      {rfFrequencies.length > 0 &&
        (effectiveDt === null ? (
          <p className="hint">dt が自動のため、RFサイクル換算は実行開始後に確定します</p>
        ) : (
          rfFrequencies.map((f) => {
            const periodS = 1 / f;
            const stepsPerCycle = periodS / effectiveDt;
            const cycles = pic.n_steps / stepsPerCycle;
            return (
              <p className="hint" key={f}>
                {formatNumber(f)} Hz: {pic.n_steps} ステップ ≈ {formatCycles(cycles)} RFサイクル
                (1サイクル ≈ {stepsPerCycle.toFixed(1)} ステップ)
                {dtFromStarted && " (前回実行の dt で換算)"}
              </p>
            );
          })
        ))}
      <div className="field">
        <span className="label">フレーム間隔</span>
        <CommitNumberInput
          value={pic.frame_every}
          onCommit={(v) => onChange({ ...pic, frame_every: Math.max(1, Math.round(v)) })}
        />
      </div>
      <div className="field">
        <span className="label">イオンサブサイクル</span>
        <CommitNumberInput
          value={pic.ion_subcycle ?? 1}
          onCommit={(v) => onChange({ ...pic, ion_subcycle: Math.max(1, Math.round(v)) })}
        />
      </div>
      <p className="hint">
        イオンを N ステップに1回 N·dt で押す (1=無効)。5〜10 で高速化、結果は近似
      </p>
      <div className="field">
        <span className="label">スレッド数</span>
        <CommitNumberInput
          value={pic.threads ?? 1}
          onCommit={(v) => onChange({ ...pic, threads: Math.max(1, Math.round(v)) })}
        />
      </div>
      <p className="hint">
        walk 探索の並列スレッド数。結果は1と完全一致。CPUコア数程度まで
      </p>
      <div className="field">
        <span className="label">平均ステップ数 (空欄=最後の25%)</span>
        <input
          type="text"
          value={pic.avg_steps === null || pic.avg_steps === undefined ? "" : String(pic.avg_steps)}
          placeholder="最後の25%"
          onChange={(e) => {
            const raw = e.target.value;
            if (raw.trim() === "") {
              onChange({ ...pic, avg_steps: null });
              return;
            }
            const n = Number(raw);
            if (Number.isFinite(n)) onChange({ ...pic, avg_steps: Math.max(1, Math.round(n)) });
          }}
        />
      </div>
      <div className="field">
        <span className="label">位相ビン数 (周期アニメ用、0=無効)</span>
        <CommitNumberInput
          value={pic.phase_bins ?? 40}
          onCommit={(v) => onChange({ ...pic, phase_bins: Math.max(0, Math.round(v)) })}
        />
      </div>

      <Toggle
        label="粒子マージ"
        checked={!!pic.merge}
        onChange={(v) => onChange({ ...pic, merge: v ? mergeDefaultsRef.current : null })}
      />
      {pic.merge && (
        <>
          <div className="field">
            <span className="label">上限マクロ粒子数 (種ごと)</span>
            <CommitNumberInput
              value={pic.merge.n_max}
              onCommit={(v) => updateMerge({ n_max: Math.max(1000, Math.round(v)) })}
            />
          </div>
          <div className="field">
            <span className="label">チェック間隔 [ステップ]</span>
            <CommitNumberInput
              value={pic.merge.every}
              onCommit={(v) => updateMerge({ every: Math.max(1, Math.round(v)) })}
            />
          </div>
        </>
      )}
      <p className="hint">
        電離で増え続けるマクロ粒子をセル内で保存的に統合します (重み・運動量・エネルギー保存)。
        IEDF等の分布に僅かな統計的影響があります。
      </p>

      </>
      )}

      {/* IEDF/IADF は一覧編集(setup)と結果表示(results)にまたがるため、
          見出しは両モードで出し、中身だけモードで出し分ける */}
      <h2>PIC: IEDF/IADF (コレクタ、最大8個)</h2>
      {show("setup") && (
      <>
      <p className="hint">
        キャンバスの「コレクタ」ツールでウエハ面上に2点クリックして線分を追加します。
        一覧の行をクリックするとキャンバス上で選択中のコレクタを強調表示します。
      </p>
      <div className="collector-list">
        {collectors.length === 0 && <div className="muted">(コレクタなし。キャンバスで配置してください)</div>}
        {collectors.map((c, i) => (
          <div
            key={i}
            className={`collector-row ${selectedCollectorIndex === i ? "selected" : ""}`}
            onClick={() => onSelectCollector(i)}
          >
            <input
              type="text"
              className="collector-label-input"
              value={c.label ?? ""}
              onClick={(e) => e.stopPropagation()}
              onChange={(e) => onUpdateCollector(i, { label: e.target.value })}
            />
            <span
              className="collector-points"
              title={`(${mToUnit(c.p1[0], lengthUnit).toFixed(2)}, ${mToUnit(c.p1[1], lengthUnit).toFixed(2)}) - (${mToUnit(c.p2[0], lengthUnit).toFixed(2)}, ${mToUnit(c.p2[1], lengthUnit).toFixed(2)}) ${unitLabel}`}
            >
              ({mToUnit(c.p1[0], lengthUnit).toFixed(1)},{mToUnit(c.p1[1], lengthUnit).toFixed(1)})–({mToUnit(c.p2[0], lengthUnit).toFixed(1)},{mToUnit(c.p2[1], lengthUnit).toFixed(1)})
            </span>
            <input
              type="text"
              inputMode="decimal"
              className="collector-tol-input"
              placeholder="mesh"
              title={`判定距離 tol [${unitLabel}] (空欄=メッシュサイズ)`}
              value={c.tol == null ? "" : String(mToUnit(c.tol, lengthUnit))}
              onClick={(e) => e.stopPropagation()}
              onChange={(e) => {
                const raw = e.target.value;
                if (raw.trim() === "") {
                  onUpdateCollector(i, { tol: null });
                  return;
                }
                const n = Number(raw);
                if (Number.isFinite(n)) onUpdateCollector(i, { tol: unitToM(n, lengthUnit) });
              }}
            />
            <button
              className="danger collector-delete"
              onClick={(e) => {
                e.stopPropagation();
                onDeleteCollector(i);
              }}
              title="このコレクタを削除"
            >
              ×
            </button>
          </div>
        ))}
      </div>
      </>
      )}

      {show("results") && collectorResults.length > 0 && (
        <>
          <div className="field">
            <span className="label">表示コレクタ</span>
            <select
              value={selectedCollectorIndex ?? 0}
              onChange={(e) => onSelectCollector(Number(e.target.value))}
            >
              {collectors.map((c, i) => (
                <option key={i} value={i}>
                  {c.label || `C${i + 1}`}
                </option>
              ))}
            </select>
          </div>
          {selectedResult ? (
            <>
              <div className="kv">
                <span>記録サンプル数</span>
                <span>{selectedResult.count}</span>
              </div>
              <div className="kv">
                <span>総実イオン数 [1/m]</span>
                <span>{selectedResult.total_weight.toExponential(3)}</span>
              </div>
              {selectedResult.truncated && (
                <div className="pic-warnings">
                  警告: サンプル数上限に達したため、以降は個数のみ集計しています (ヒストグラムは記録分のみ反映)
                </div>
              )}
              <div className="field">
                <span className="label">ビン数</span>
                <CommitNumberInput
                  value={collectorBins}
                  onCommit={(v) => setCollectorBins(Math.max(1, Math.round(v)))}
                />
              </div>
              <p className="hint">IEDF (入射エネルギー分布、横軸 eV)</p>
              <HistogramChart
                values={selectedResult.energies_ev}
                weights={selectedResult.weights}
                bins={collectorBins}
                unit="eV"
                color="#4da3ff"
              />
              <p className="hint">IADF (入射角分布、横軸 deg、-90〜90)</p>
              <HistogramChart
                values={selectedResult.angles_deg}
                weights={selectedResult.weights}
                bins={collectorBins}
                unit="deg"
                color="#ffb84d"
                fixedRange={[-90, 90]}
              />
              <p className="hint">IAEDF (角度×エネルギー、カラー = 確率密度)</p>
              <IaedfChart result={selectedResult} bins={collectorBins} />
              {csvError && <div className="error">{csvError}</div>}
              <div className="actions">
                <button className="secondary" onClick={downloadCollectorCsv}>
                  CSV保存
                </button>
              </div>
            </>
          ) : (
            <div className="muted">(選択中のコレクタの実行結果がありません)</div>
          )}
        </>
      )}

      {show("setup") && (
      <>
      <div className="actions">
        <button onClick={onStart} disabled={!canRun || running}>
          {running ? "実行中..." : "PIC開始"}
        </button>
        <button className="secondary" onClick={onStop} disabled={!running}>
          停止
        </button>
        <button
          className="secondary"
          onClick={onContinue}
          disabled={!canContinue}
          title={
            continueDisabledByProjectChange
              ? "ジオメトリ・プラズマ設定が変更されたため続き実行できません (再度 PIC開始 してください)"
              : undefined
          }
        >
          続きから実行
        </button>
      </div>
      <p className="hint">
        「続きから実行」は現在のステップ数・フレーム間隔・平均ステップ数・位相ビン数で追加実行します。
        ジオメトリ・プラズマ設定の変更は続き実行には反映されません
        (粒子状態・表面電荷・時刻は前回から継続)。
        {continueDisabledByProjectChange &&
          " ジオメトリ・プラズマ設定を編集したため、続き実行するには再度「PIC開始」が必要です。"}
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
        </>
      )}

      {frame && (
        <>
          <h2>PIC: 診断</h2>
          <div className="kv">
            <span>粒子数 (電子/イオン)</span>
            <span>{frame.diag.n_e} / {frame.diag.n_i}</span>
          </div>
          <div className="kv">
            <span>φ min/max</span>
            <span>{frame.diag.phi_min.toFixed(2)} / {frame.diag.phi_max.toFixed(2)} V</span>
          </div>
          <div className="kv">
            <span>壁吸収 (電子/イオン)</span>
            <span>{frame.diag.wall_e} / {frame.diag.wall_i}</span>
          </div>
          <div className="kv">
            <span>衝突/電離/SEE (累計)</span>
            <span>
              {frame.diag.coll_e ?? "-"} / {frame.diag.ion_events ?? "-"} / {frame.diag.see_events ?? "-"}
            </span>
          </div>
          {pic.merge && (
            <div className="kv">
              <span>マージ削減粒子数 (累計)</span>
              <span>{frame.diag.merged ?? "-"}</span>
            </div>
          )}
          <div className="kv">
            <span>{isAxisymmetric(project.coord) ? "誘電体表面電荷 [C]" : "誘電体表面電荷 [C/m]"}</span>
            <span>{frame.diag.surf_q !== undefined ? frame.diag.surf_q.toExponential(3) : "-"}</span>
          </div>
          {frame.diag.fn_i !== undefined && (
            <div className="kv">
              <span>{isAxisymmetric(project.coord) ? "FN放出電流 [A]" : "FN放出電流 [A/m]"}</span>
              <span>{frame.diag.fn_i.toExponential(3)}</span>
            </div>
          )}
          <PicHistoryChart history={history} />
        </>
      )}

      {/* 位相別プロファイル計測 (prompts/75、done 後のみ表示。実行中/未実行は非表示) */}
      {timing && <PicTimingSection timing={timing} />}
      </>
      )}

      {show("results") && fields && (
        <>
          <h2>PIC: 結果フィールド</h2>
          <p className="hint">時間平均ステップ数: {fields.avg_steps}</p>
          <div className="field">
            <span className="label">結果表示</span>
            <select
              value={resultField}
              onChange={(e) => onResultFieldChange(e.target.value as PicResultField)}
            >
              {PIC_FIELD_OPTIONS.map((o) => (
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
        <PicCyclePlayer
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
          showParticles={cycleShowParticles}
          onShowParticlesChange={onCycleShowParticlesChange}
        />
      )}

      {/* results専用ページで未実行の場合のヒント (mode="all" の従来ページでは出さない) */}
      {mode === "results" && !fields && !cycle && (
        <p className="hint">PIC計算が未実行です。スタディ「PIC-MCC」から実行してください。</p>
      )}

      {error && (
        <>
          <h2>PICエラー</h2>
          <div className="error">{error}</div>
        </>
      )}
    </>
  );
}

// 位相別プロファイル計測 (prompts/75) の日本語ラベル。timing のキーと1対1対応させる
// (total はここに含めず、フェーズ一覧の下に別枠で表示する)
const TIMING_PHASE_LABELS: Record<string, string> = {
  solve: "場ソルブ",
  gather_push: "粒子押し出し",
  walk: "walk探索",
  deposit: "電荷デポジット",
  mcc: "MCC衝突",
  other: "その他",
};

// PIC: 実行時間内訳。done メッセージの timing を値の大きい順に「名称 / 秒 / %」で表示する
// (次フェーズ (Numba化・粒子マージ) の効果測定のベースライン確認用、prompts/75)。
// 未知のキー (将来の位相追加) も TIMING_PHASE_LABELS に無ければキー名そのままで表示する
function PicTimingSection({ timing }: { timing: Record<string, number> }) {
  const total = timing.total ?? Object.entries(timing)
    .filter(([k]) => k !== "total")
    .reduce((sum, [, v]) => sum + v, 0);
  const rows = Object.entries(timing)
    .filter(([k]) => k !== "total")
    .sort((a, b) => b[1] - a[1]);
  return (
    <>
      <h2>PIC: 実行時間内訳</h2>
      {rows.map(([key, sec]) => (
        <div className="kv" key={key}>
          <span>{TIMING_PHASE_LABELS[key] ?? key}</span>
          <span>
            {sec.toFixed(3)} s ({total > 0 ? ((100 * sec) / total).toFixed(1) : "0.0"}%)
          </span>
        </div>
      ))}
      <div className="kv">
        <span>合計</span>
        <span>{total.toFixed(3)} s</span>
      </div>
    </>
  );
}

// 小さな履歴チャート: 横軸=時刻、縦軸は 運動E・場E・全E・粒子数 の4系列を
// それぞれ独立に 0〜1 正規化して重ね描きする (canvas 直描き、依存追加なし)
function PicHistoryChart({ history }: { history: PicDiag[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = el.getBoundingClientRect();
    el.width = rect.width * dpr;
    el.height = rect.height * dpr;
    const ctx = el.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);

    // 防御: 想定外の形式 (配列でない等) が渡っても例外で画面全体を落とさない
    if (!Array.isArray(history) || history.length < 2) return;

    const ts = history.map((h) => h.t);
    const ke = history.map((h) => h.ke_e + h.ke_i);
    const fe = history.map((h) => h.fe);
    const total = ke.map((v, i) => v + fe[i]);
    const nParticles = history.map((h) => h.n_e + h.n_i);

    const tMin = ts[0];
    const tMax = ts[ts.length - 1];
    const tRange = tMax - tMin || 1;
    const xOf = (t: number) => ((t - tMin) / tRange) * rect.width;

    const padY = 3;
    const drawSeries = (values: number[], color: string) => {
      const vMin = Math.min(...values);
      const vMax = Math.max(...values);
      const range = vMax - vMin || 1;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      values.forEach((v, i) => {
        const x = xOf(ts[i]);
        const y = rect.height - padY - ((v - vMin) / range) * (rect.height - 2 * padY);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };

    drawSeries(ke, "#4da3ff");
    drawSeries(fe, "#ffb84d");
    drawSeries(total, "#6fd08c");
    drawSeries(nParticles, "#d8dce4");
  }, [history]);

  return (
    <>
      <canvas ref={canvasRef} className="pic-chart" />
      <div className="pic-chart-legend">
        <span><span className="swatch" style={{ background: "#4da3ff" }} />運動E</span>
        <span><span className="swatch" style={{ background: "#ffb84d" }} />場E</span>
        <span><span className="swatch" style={{ background: "#6fd08c" }} />全E</span>
        <span><span className="swatch" style={{ background: "#d8dce4" }} />粒子数</span>
      </div>
    </>
  );
}

interface HistogramChartProps {
  values: number[];   // サンプル値 (energies_ev または angles_deg)
  weights: number[];  // 同数、重み付きカウントに使う (values と同じインデックス対応)
  bins: number;       // ビン数
  unit: string;       // 横軸の単位表示
  color: string;      // バーの色
  // 指定時はこの範囲でビニングする (未指定時は values の min/max を使う)。IADFは常に固定 [-90, 90]
  fixedRange?: [number, number];
}

// IEDF/IADF 用のヒストグラム (重み付きカウント、canvas 直描き)。
// ProfilePanel のグラフ実装を参考に、軸目盛り・ホバーでのビン範囲/カウント読み取りを持たせる
function HistogramChart({ values, weights, bins, unit, color, fixedRange }: HistogramChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const scaleRef = useRef<{ padL: number; binW: number; n: number } | null>(null);

  // 重み付きカウントのビニング (values/weights が変わるか bins/range が変わるたびに再計算)
  const { counts, lo, hi } = useMemo(() => {
    const n = Math.max(1, Math.round(bins));
    let rangeLo: number;
    let rangeHi: number;
    if (fixedRange) {
      [rangeLo, rangeHi] = fixedRange;
    } else if (values.length === 0) {
      rangeLo = 0;
      rangeHi = 1;
    } else {
      rangeLo = Math.min(...values);
      rangeHi = Math.max(...values);
      if (!(rangeHi > rangeLo)) rangeHi = rangeLo + 1;
    }
    const counts = new Array(n).fill(0) as number[];
    const range = rangeHi - rangeLo || 1;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      let idx = Math.floor(((v - rangeLo) / range) * n);
      if (idx < 0) idx = 0;
      if (idx >= n) idx = n - 1;
      counts[idx] += weights[i] ?? 1;
    }
    return { counts, lo: rangeLo, hi: rangeHi };
  }, [values, weights, bins, fixedRange]);

  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = el.getBoundingClientRect();
    el.width = rect.width * dpr;
    el.height = rect.height * dpr;
    const ctx = el.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const padL = 44;
    const padR = 6;
    const padT = 6;
    const padB = 16;
    const plotW = rect.width - padL - padR;
    const plotH = rect.height - padT - padB;
    const n = counts.length;
    const binW = plotW / n;
    scaleRef.current = { padL, binW, n };

    const maxCount = Math.max(...counts, 1e-30);

    // 枠
    ctx.strokeStyle = "#363c48";
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, padT, plotW, plotH);

    // バー
    ctx.fillStyle = color;
    for (let i = 0; i < n; i++) {
      const h = (counts[i] / maxCount) * plotH;
      const x = padL + i * binW;
      const y = padT + plotH - h;
      ctx.fillRect(x + 0.5, y, Math.max(1, binW - 1), h);
    }

    // 横軸目盛り (最小/中央/最大)
    ctx.font = "9px system-ui, sans-serif";
    ctx.fillStyle = "#8a919e";
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText(lo.toFixed(1), padL, padT + plotH + 3);
    ctx.textAlign = "center";
    ctx.fillText(((lo + hi) / 2).toFixed(1), padL + plotW / 2, padT + plotH + 3);
    ctx.textAlign = "right";
    ctx.fillText(hi.toFixed(1), padL + plotW, padT + plotH + 3);

    // 縦軸目盛り (0 / 最大カウント)
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(maxCount.toExponential(1), padL - 4, padT);
    ctx.textBaseline = "bottom";
    ctx.fillText("0", padL - 4, padT + plotH);

    // ホバー中のビンを枠で強調
    if (hoverIdx !== null && hoverIdx >= 0 && hoverIdx < n) {
      ctx.strokeStyle = "rgba(255,255,255,0.6)";
      ctx.lineWidth = 1;
      ctx.strokeRect(padL + hoverIdx * binW, padT, binW, plotH);
    }
  }, [counts, lo, hi, hoverIdx, color]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const el = canvasRef.current;
    const sc = scaleRef.current;
    if (!el || !sc) return;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const idx = Math.floor((x - sc.padL) / sc.binW);
    setHoverIdx(idx >= 0 && idx < sc.n ? idx : null);
  };

  const n = counts.length;
  const range = hi - lo || 1;
  const hoverBinLo = hoverIdx !== null ? lo + (range * hoverIdx) / n : null;
  const hoverBinHi = hoverIdx !== null ? lo + (range * (hoverIdx + 1)) / n : null;

  return (
    <div className="iedf-hist-wrap">
      <canvas
        ref={canvasRef}
        className="iedf-hist"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverIdx(null)}
      />
      <div className="hint iedf-hist-hover">
        {hoverIdx !== null && hoverBinLo !== null && hoverBinHi !== null
          ? `${hoverBinLo.toFixed(2)} 〜 ${hoverBinHi.toFixed(2)} ${unit}: ${counts[hoverIdx].toExponential(3)}`
          : " "}
      </div>
    </div>
  );
}

// 簡易 viridis 風カラーマップ (CadCanvas の colormap と同じ制御点、canvas/CadCanvas.tsx からは
// export されていないためここで独立に定義する)
const IAEDF_COLOR_STOPS: [number, number, number][] = [
  [68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37],
];
function iaedfColormap(t: number): string {
  const x = Math.min(0.9999, Math.max(0, t)) * (IAEDF_COLOR_STOPS.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const [r1, g1, b1] = IAEDF_COLOR_STOPS[i];
  const [r2, g2, b2] = IAEDF_COLOR_STOPS[i + 1];
  return `rgb(${r1 + (r2 - r1) * f},${g1 + (g2 - g1) * f},${b1 + (b2 - b1) * f})`;
}
// 0カウントのビンに使う背景色 (既存ダークテーマの canvas 背景相当)
const IAEDF_BG_COLOR = "#0e1116";

interface IaedfChartProps {
  result: PicCollectorResult;
  bins: number;
}

// IAEDF (角度×エネルギー) 2Dヒートマップ。HistogramChart と同じ canvas 直描きスタイル
// (枠 #363c48, 9px 目盛りフォント, padL≈44) に合わせ、右側に細いカラーバーを添える。
// 横軸=入射角 [-90,90] 固定、縦軸=エネルギー (サンプルのmin/max、下から上へ増加)。
// 「対数」チェックボックスでカラーを log10 スケールに切替できる (0カウントは常に背景色)
function IaedfChart({ result, bins }: IaedfChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoverCell, setHoverCell] = useState<{ ix: number; iy: number } | null>(null);
  const [logScale, setLogScale] = useState(false);
  const scaleRef = useRef<{ padL: number; padT: number; plotW: number; plotH: number; nx: number; ny: number } | null>(
    null
  );

  // 重み付き2Dカウントのビニング (角度は常に[-90,90]固定、エネルギーはサンプルのmin/max)
  const { counts, eLo, eHi, maxCount, nx, ny } = useMemo(() => {
    const n = Math.max(1, Math.round(bins));
    const nx = n;
    const ny = n;
    const aLo = -90;
    const aHi = 90;
    const energies = result.energies_ev;
    let eLo: number;
    let eHi: number;
    if (energies.length === 0) {
      eLo = 0;
      eHi = 1;
    } else {
      eLo = Math.min(...energies);
      eHi = Math.max(...energies);
      if (!(eHi > eLo)) eHi = eLo + 1;
    }
    const counts: number[][] = [];
    for (let j = 0; j < ny; j++) counts.push(new Array(nx).fill(0));
    const aRange = aHi - aLo || 1;
    const eRange = eHi - eLo || 1;
    for (let i = 0; i < energies.length; i++) {
      const a = result.angles_deg[i];
      const e = energies[i];
      let ix = Math.floor(((a - aLo) / aRange) * nx);
      let iy = Math.floor(((e - eLo) / eRange) * ny);
      if (ix < 0) ix = 0;
      if (ix >= nx) ix = nx - 1;
      if (iy < 0) iy = 0;
      if (iy >= ny) iy = ny - 1;
      counts[iy][ix] += result.weights[i] ?? 1;
    }
    let maxCount = 0;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        if (counts[j][i] > maxCount) maxCount = counts[j][i];
      }
    }
    return { counts, eLo, eHi, maxCount, nx, ny };
  }, [result, bins]);

  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = el.getBoundingClientRect();
    el.width = rect.width * dpr;
    el.height = rect.height * dpr;
    const ctx = el.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const padL = 44;
    const padT = 6;
    const padB = 16;
    const barW = 14;
    const barGap = 8;
    const labelW = 52; // カラーバー右のtoExponential(2)ラベル分の余白見積もり
    const padR = barW + barGap + labelW;
    const plotW = rect.width - padL - padR;
    const plotH = rect.height - padT - padB;
    scaleRef.current = { padL, padT, plotW, plotH, nx, ny };

    const cellW = plotW / nx;
    const cellH = plotH / ny;

    // 対数スケール時の正規化に使う「正の最小カウント」
    let minPositive = Infinity;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const v = counts[j][i];
        if (v > 0 && v < minPositive) minPositive = v;
      }
    }
    if (!(minPositive < Infinity)) minPositive = 1e-30;
    const maxCountSafe = Math.max(maxCount, 1e-30);
    const logLo = Math.log10(minPositive);
    const logHi = Math.log10(maxCountSafe);
    const logRange = logHi - logLo || 1;

    // 正規化: 0ビンは負値を返し、呼び出し側で背景色に落とす
    const normalize = (v: number): number => {
      if (v <= 0) return -1;
      if (logScale) return (Math.log10(v) - logLo) / logRange;
      return v / maxCountSafe;
    };

    // ヒートマップ本体 (行iy=0がエネルギー最小=下端になるよう描画)
    for (let j = 0; j < ny; j++) {
      const y = padT + plotH - (j + 1) * cellH;
      for (let i = 0; i < nx; i++) {
        const t = normalize(counts[j][i]);
        ctx.fillStyle = t < 0 ? IAEDF_BG_COLOR : iaedfColormap(t);
        const x = padL + i * cellW;
        // 継ぎ目の隙間を避けるためわずかに重ねて描画する
        ctx.fillRect(x, y, cellW + 0.6, cellH + 0.6);
      }
    }

    // 枠
    ctx.strokeStyle = "#363c48";
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, padT, plotW, plotH);

    // 横軸目盛り (-90/0/90)
    ctx.font = "9px system-ui, sans-serif";
    ctx.fillStyle = "#8a919e";
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText("-90", padL, padT + plotH + 3);
    ctx.textAlign = "center";
    ctx.fillText("0", padL + plotW / 2, padT + plotH + 3);
    ctx.textAlign = "right";
    ctx.fillText("90", padL + plotW, padT + plotH + 3);

    // 縦軸目盛り (min/中央/max エネルギー、下から上へ増加)
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(eHi.toFixed(1), padL - 4, padT);
    ctx.textBaseline = "middle";
    ctx.fillText(((eLo + eHi) / 2).toFixed(1), padL - 4, padT + plotH / 2);
    ctx.textBaseline = "bottom";
    ctx.fillText(eLo.toFixed(1), padL - 4, padT + plotH);

    // カラーバー (右側、縦方向。上=最大値、下=0)
    const barX = padL + plotW + barGap;
    const barY = padT;
    const barH = plotH;
    const steps = 32;
    for (let s = 0; s < steps; s++) {
      const tVal = 1 - s / steps; // このセルが表す色の正規化値 (上ほど大きい)
      ctx.fillStyle = iaedfColormap(tVal);
      const y = barY + (s / steps) * barH;
      ctx.fillRect(barX, y, barW, barH / steps + 0.6);
    }
    ctx.strokeStyle = "#363c48";
    ctx.strokeRect(barX, barY, barW, barH);
    ctx.fillStyle = "#8a919e";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(maxCount.toExponential(2), barX + barW + 4, barY);
    ctx.textBaseline = "bottom";
    ctx.fillText("0", barX + barW + 4, barY + barH);

    // ホバー中のビンを枠で強調
    if (hoverCell) {
      const { ix, iy } = hoverCell;
      if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) {
        ctx.strokeStyle = "rgba(255,255,255,0.6)";
        ctx.lineWidth = 1;
        const x = padL + ix * cellW;
        const y = padT + plotH - (iy + 1) * cellH;
        ctx.strokeRect(x, y, cellW, cellH);
      }
    }
  }, [counts, eLo, eHi, maxCount, nx, ny, hoverCell, logScale]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const el = canvasRef.current;
    const sc = scaleRef.current;
    if (!el || !sc) return;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const ix = Math.floor((x - sc.padL) / (sc.plotW / sc.nx));
    const rowFromTop = Math.floor((y - sc.padT) / (sc.plotH / sc.ny));
    const iy = sc.ny - 1 - rowFromTop;
    if (ix >= 0 && ix < sc.nx && iy >= 0 && iy < sc.ny) setHoverCell({ ix, iy });
    else setHoverCell(null);
  };

  const aRange = 180;
  const eRange = eHi - eLo || 1;
  const hoverAngleLo = hoverCell ? -90 + (aRange * hoverCell.ix) / nx : null;
  const hoverAngleHi = hoverCell ? -90 + (aRange * (hoverCell.ix + 1)) / nx : null;
  const hoverEnergyLo = hoverCell ? eLo + (eRange * hoverCell.iy) / ny : null;
  const hoverEnergyHi = hoverCell ? eLo + (eRange * (hoverCell.iy + 1)) / ny : null;
  const hoverCount = hoverCell ? counts[hoverCell.iy][hoverCell.ix] : null;

  return (
    <div className="iedf-hist-wrap">
      <canvas
        ref={canvasRef}
        className="iaedf-canvas"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverCell(null)}
      />
      <div className="hint iedf-hist-hover">
        {hoverCell && hoverAngleLo !== null && hoverAngleHi !== null && hoverEnergyLo !== null && hoverEnergyHi !== null
          ? `${hoverAngleLo.toFixed(1)} 〜 ${hoverAngleHi.toFixed(1)} deg, ${hoverEnergyLo.toFixed(2)} 〜 ${hoverEnergyHi.toFixed(2)} eV: ${hoverCount!.toExponential(3)}`
          : " "}
      </div>
      <div className="iaedf-log-field">
        <Toggle label="対数スケール" checked={logScale} onChange={setLogScale} />
      </div>
    </div>
  );
}

interface PicCyclePlayerProps {
  cycle: PicCycle;
  field: CyclePicField;
  onFieldChange: (v: CyclePicField) => void;
  logScale: boolean;
  onLogScaleChange: (v: boolean) => void;
  playing: boolean;
  onPlayingChange: (v: boolean) => void;
  binIndex: number;
  onBinIndexChange: (v: number) => void;
  fps: number;
  onFpsChange: (v: number) => void;
  showParticles: boolean;
  onShowParticlesChange: (v: boolean) => void;
}

// PIC: 周期アニメーションプレイヤー。RF 1周期分の位相分解データ (cycle) を
// 再生/一時停止・位相スライダー・再生速度で辿る UI (実際のビン送り・描画は App/CadCanvas 側で行う)。
// このコンポーネントは選択状態の表示・入力のみを担う
function PicCyclePlayer({
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
  showParticles,
  onShowParticlesChange,
}: PicCyclePlayerProps) {
  const bin = Math.min(binIndex, cycle.bins - 1);
  const phaseDeg = (bin / cycle.bins) * 360;
  const tInBin = (bin / cycle.bins) * cycle.period_s;

  return (
    <>
      <h2>PIC: 周期アニメーション</h2>
      <div className="field">
        <span className="label">表示フィールド</span>
        <select value={field} onChange={(e) => onFieldChange(e.target.value as CyclePicField)}>
          {CYCLE_FIELD_OPTIONS.map((o) => (
            <option key={o.value} value={o.value} disabled={cycle[o.value] === undefined}>
              {o.label}
            </option>
          ))}
        </select>
      </div>
      <Toggle label="対数スケール" checked={logScale} onChange={onLogScaleChange} />
      <Toggle label="粒子スナップショット表示" checked={showParticles} onChange={onShowParticlesChange} />

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
        (周期 {(cycle.period_s * 1e9).toFixed(2)} ns)
      </p>
    </>
  );
}
