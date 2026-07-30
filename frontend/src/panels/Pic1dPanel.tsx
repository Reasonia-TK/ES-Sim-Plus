import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { CommitNullableNumberInput, CommitNumberInput, CommitTextInput, formatNumber } from "../CommitInput";
import { Toggle } from "../Toggle";
import { LENGTH_UNIT_LABEL, mToUnit, unitToM } from "../units";
import type { LengthUnit } from "../units";
import { WaveformImportEditor } from "./FieldPanel";
import { ProcessList } from "./PicPanel";
import type {
  Eedf1dRegion,
  McSettings,
  Pic1dElectrode,
  Pic1dFrameMsg,
  Pic1dSettings,
  Pic1dStartedMsg,
} from "../types";

/**
 * 1D PIC/MCC (PIC-MCC 1D) スタディの設定・実行パネル。
 * geometry/mesh とは無関係な専用の一様格子ソルバー (backend/es_sim/pic1d.py) を対象とする。
 * 既存 PicPanel (2D) と見た目・部品 (CommitNumberInput/CommitNullableNumberInput・Toggle・
 * hint 文体) は揃えるが、結果の可視化 (ラインプロット・位相アニメ・EEDFチャート・数値サマリ)
 * は全てキャンバス領域 (canvas/Plot1dView.tsx) 側に置き、このパネルは設定編集と実行制御のみを持つ
 * (1D では「設定は左パネルのみ・キャンバスは結果表示のみ」という役割分担にするため、prompts/91)。
 */

interface Props {
  lengthUnit: LengthUnit;
  pic1d: Pic1dSettings;
  onChange: (next: Pic1dSettings) => void;
  canRun: boolean;
  running: boolean;
  onStart: () => void;
  onStop: () => void;
  // 「続きから」ボタン: 直前の実行が done/stop 済みで現在実行中でない場合のみ true。
  // 1D は geometry/mesh に依存しないため、2D の continueDisabledByProjectChange に相当する
  // 概念はない (pic1d 設定を変えても「続きから」は常に現在保持中のサーバー状態へ継続実行するだけ)
  canContinue: boolean;
  onContinue: (extraSteps: number) => void;
  started: Pic1dStartedMsg | null;
  frame: Pic1dFrameMsg | null;
  error: string | null;
}

// MCC(背景ガス衝突)設定の既定値。有効チェックを一度オフにしても直前の値を復元できるよう保持する
// (PicPanel の DEFAULT_MCC と同じ考え方。1D 独自に ionization_split/ion_energy_frame の
// 既定値を明示しておく — Turner/eduPIC ベンチマークの再現に関わるため)
const DEFAULT_MCC: McSettings = {
  gas: { name: "Ar", pressure_pa: 10.0, temperature_k: 300.0 },
  electron_processes: [],
  ion_processes: [],
  seed: 0,
  ionization_split: "half",
  ion_energy_frame: "lab",
};

const DEFAULT_ELECTRODE: Pic1dElectrode = { v_dc: 0.0, waveforms: [], see_gamma: 0.0 };

type Pic1dPreset = { label: string; description: string; pic1d: Pic1dSettings; note?: string };

// project.pic1d.left/right.waveforms から重複排除・昇順に周波数を集める (RFサイクル換算用。
// PicPanel の collectRfFrequencies と同じ考え方だが、1D は project.geometry の境界条件では
// なく電極 (Pic1dElectrode.waveforms) から集める点が異なる)
function collectPic1dFrequencies(pic1d: Pic1dSettings): number[] {
  const freqs = new Set<number>();
  for (const wf of pic1d.left.waveforms ?? []) if (wf.freq_hz > 0) freqs.add(wf.freq_hz);
  for (const wf of pic1d.right.waveforms ?? []) if (wf.freq_hz > 0) freqs.add(wf.freq_hz);
  return Array.from(freqs).sort((a, b) => a - b);
}

// サイクル数の表示整形 (PicPanel の formatCycles と同じ)
function formatCycles(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  return v.toPrecision(3).replace("e+", "e");
}

// 次の EEDF 領域ラベルを生成する ("E1", "E2", ...)。App.tsx の nextEedfLabel (2D) と同じ流儀
function nextEedf1dLabel(regions: Eedf1dRegion[]): string {
  let maxN = 0;
  for (const r of regions) {
    const m = /^E(\d+)$/.exec(r.label ?? "");
    if (m) maxN = Math.max(maxN, parseInt(m[1], 10));
  }
  return `E${maxN + 1}`;
}

const MAX_EEDF1D_REGIONS = 4;

// 電極 (左/右) の DC電圧・SEE収率・波形リストの編集UI
function ElectrodeEditor({
  title,
  electrode,
  onChange,
}: {
  title: string;
  electrode: Pic1dElectrode;
  onChange: (next: Pic1dElectrode) => void;
}) {
  const waveforms = electrode.waveforms ?? [];
  return (
    <>
      <p className="hint" style={{ fontWeight: 600, color: "#d8dce4" }}>{title}</p>
      <div className="field">
        <span className="label">DC電圧 [V]</span>
        <CommitNumberInput value={electrode.v_dc ?? 0} onCommit={(v) => onChange({ ...electrode, v_dc: v })} />
      </div>
      <div className="field">
        <span className="label">SEE収率 γ</span>
        <CommitNumberInput
          value={electrode.see_gamma ?? 0}
          onCommit={(v) => onChange({ ...electrode, see_gamma: v })}
        />
      </div>
      {/* Pic1dElectrode.waveforms は複数波形の和 (v_dc + Σ waveforms(t)) を表す配列のため、
          既存の WaveformImportEditor (1個編集用、FieldPanel の共有部品) を配列の各要素と
          末尾の「追加スロット」に割り当てて流用する (RF駆動は正弦波を1周期分サンプルした
          CSV波形として表現する、pic1d_presets.py と同じ流儀) */}
      {waveforms.map((wf, i) => (
        <WaveformImportEditor
          key={i}
          waveform={wf}
          onChange={(next) => {
            const list = waveforms.slice();
            if (next === undefined) list.splice(i, 1);
            else list[i] = next;
            onChange({ ...electrode, waveforms: list });
          }}
        />
      ))}
      <WaveformImportEditor
        waveform={undefined}
        onChange={(next) => {
          if (next) onChange({ ...electrode, waveforms: [...waveforms, next] });
        }}
      />
    </>
  );
}

export default function Pic1dPanel({
  lengthUnit,
  pic1d,
  onChange,
  canRun,
  running,
  onStart,
  onStop,
  canContinue,
  onContinue,
  started,
  frame,
  error,
}: Props) {
  const unitLabel = LENGTH_UNIT_LABEL[lengthUnit];

  // --- プリセット (prompts/91): パネル初回表示時に GET /pic1d/presets を取得する ---
  const [presets, setPresets] = useState<Record<string, Pic1dPreset> | null>(null);
  const [presetsError, setPresetsError] = useState<string | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<string>("");
  useEffect(() => {
    api
      .pic1dPresets()
      .then((res) => {
        setPresets(res);
        const keys = Object.keys(res);
        if (keys.length > 0) setSelectedPreset(keys[0]);
      })
      .catch((e) => setPresetsError(String(e)));
  }, []);
  const presetEntry = presets && selectedPreset ? presets[selectedPreset] : null;
  const applyPreset = () => {
    if (!presetEntry) return;
    // 確認なしで丸ごと置換する (仕様上は「上書きします」の hint 表示のみで足りる)
    onChange(presetEntry.pic1d);
  };

  // --- MCC設定。有効チェックを一度オフにしても、再度オンにしたときに直前の値を復元できるよう保持する ---
  const mccDefaultsRef = useRef<McSettings>(pic1d.mcc ?? DEFAULT_MCC);
  useEffect(() => {
    if (pic1d.mcc) mccDefaultsRef.current = pic1d.mcc;
  }, [pic1d.mcc]);
  const updateMcc = (patch: Partial<McSettings>) => {
    if (!pic1d.mcc) return;
    onChange({ ...pic1d, mcc: { ...pic1d.mcc, ...patch } });
  };
  const updateGas = (patch: Partial<McSettings["gas"]>) => {
    if (!pic1d.mcc) return;
    onChange({ ...pic1d, mcc: { ...pic1d.mcc, gas: { ...pic1d.mcc.gas, ...patch } } });
  };

  // LXCatインポート (電子/イオン共通)。PicPanel (2D) の同名処理の複製 (二重管理)。
  // 理想は共通コンポーネント化だが、PicPanel 側がローカル state を抱えたまま関数本体に
  // インラインされており抽出には PicPanel 側の構造変更を要するため、変更範囲を抑えて複製する
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

  // --- EEDF/EEPF 領域 (x1/x2、最大4個) ---
  const eedfRegions = pic1d.eedf_regions ?? [];
  const addEedfRegion = () => {
    if (eedfRegions.length >= MAX_EEDF1D_REGIONS) return;
    const label = nextEedf1dLabel(eedfRegions);
    const next: Eedf1dRegion = { x1: 0, x2: pic1d.gap_m * 0.1, label, bins: 100, e_max_ev: null };
    onChange({ ...pic1d, eedf_regions: [...eedfRegions, next] });
  };
  const updateEedfRegion = (i: number, patch: Partial<Eedf1dRegion>) => {
    onChange({ ...pic1d, eedf_regions: eedfRegions.map((r, idx) => (idx === i ? { ...r, ...patch } : r)) });
  };
  const deleteEedfRegion = (i: number) => {
    onChange({ ...pic1d, eedf_regions: eedfRegions.filter((_, idx) => idx !== i) });
  };

  // --- RFサイクル換算 (PicPanel と同じ考え方、電極波形から周波数を集める点のみ異なる) ---
  const rfFrequencies = collectPic1dFrequencies(pic1d);
  const effectiveDt = pic1d.dt ?? started?.dt ?? null;
  const dtFromStarted = pic1d.dt == null && started != null;
  const nSteps = pic1d.n_steps ?? 2000;

  // --- 実行進捗 (続き実行では frame.step が通算のため、区間開始オフセットを引く) ---
  const segOffset = started?.step_offset ?? 0;
  const segStep = Math.max(0, (frame?.step ?? segOffset) - segOffset);
  const progressPct = started && started.n_steps > 0 ? Math.min(100, (segStep / started.n_steps) * 100) : 0;

  // 「続きから」の追加ステップ数。既定は現在の n_steps 設定を踏襲するが、以後は独立に編集できる
  const [extraSteps, setExtraSteps] = useState(nSteps);

  return (
    <>
      <h2>PIC 1D: プリセット</h2>
      {presetsError && <div className="error">プリセット取得に失敗しました: {presetsError}</div>}
      {presets && (
        <>
          <div className="field">
            <span className="label">プリセット</span>
            <select value={selectedPreset} onChange={(e) => setSelectedPreset(e.target.value)}>
              {Object.entries(presets).map(([key, p]) => (
                <option key={key} value={key}>{p.label}</option>
              ))}
            </select>
          </div>
          {presetEntry && (
            <p className="hint">
              {presetEntry.description}
              {presetEntry.note && <><br />※ {presetEntry.note}</>}
            </p>
          )}
          <div className="actions">
            <button className="secondary" onClick={applyPreset} disabled={!presetEntry}>
              プリセット適用
            </button>
          </div>
          <p className="hint">適用すると現在の 1D 設定を上書きします (確認なし、Undo 対象外)。</p>
        </>
      )}

      <h2>PIC 1D: 形状</h2>
      <div className="field">
        <span className="label">ギャップ長 [{unitLabel}]</span>
        <CommitNumberInput
          value={mToUnit(pic1d.gap_m, lengthUnit)}
          onCommit={(v) => onChange({ ...pic1d, gap_m: Math.max(1e-9, unitToM(v, lengthUnit)) })}
        />
      </div>
      <div className="field">
        <span className="label">セル数</span>
        <CommitNumberInput
          value={pic1d.n_cells}
          onCommit={(v) => onChange({ ...pic1d, n_cells: Math.max(8, Math.round(v)) })}
        />
      </div>

      <h2>PIC 1D: 初期プラズマ</h2>
      <div className="field">
        <span className="label">密度 [m^-3]</span>
        <CommitNumberInput
          value={pic1d.init_density_m3}
          onCommit={(v) => onChange({ ...pic1d, init_density_m3: v })}
        />
      </div>
      <div className="field">
        <span className="label">Te [eV]</span>
        <CommitNumberInput
          value={pic1d.init_te_ev ?? 2.0}
          onCommit={(v) => onChange({ ...pic1d, init_te_ev: v })}
        />
      </div>
      <div className="field">
        <span className="label">Ti [eV]</span>
        <CommitNumberInput
          value={pic1d.init_ti_ev ?? 0.03}
          onCommit={(v) => onChange({ ...pic1d, init_ti_ev: v })}
        />
      </div>
      <div className="field">
        <span className="label">イオン質量 [amu]</span>
        <CommitNumberInput
          value={pic1d.ion_mass_amu ?? 39.948}
          onCommit={(v) => onChange({ ...pic1d, ion_mass_amu: v })}
        />
      </div>
      <div className="field">
        <span className="label">マクロ粒子数</span>
        <CommitNumberInput
          value={pic1d.n_macro ?? 20000}
          onCommit={(v) => onChange({ ...pic1d, n_macro: Math.max(1, Math.round(v)) })}
        />
      </div>
      <div className="field">
        <span className="label">乱数シード</span>
        <CommitNumberInput
          value={pic1d.seed ?? 0}
          onCommit={(v) => onChange({ ...pic1d, seed: Math.round(v) })}
        />
      </div>

      <h2>PIC 1D: 電極</h2>
      <ElectrodeEditor
        title="左電極 (x=0)"
        electrode={pic1d.left ?? DEFAULT_ELECTRODE}
        onChange={(next) => onChange({ ...pic1d, left: next })}
      />
      <ElectrodeEditor
        title="右電極 (x=gap)"
        electrode={pic1d.right ?? DEFAULT_ELECTRODE}
        onChange={(next) => onChange({ ...pic1d, right: next })}
      />

      <h2>PIC 1D: MCC(衝突)</h2>
      <Toggle
        label="有効"
        checked={pic1d.mcc !== null && pic1d.mcc !== undefined}
        onChange={(v) => onChange({ ...pic1d, mcc: v ? mccDefaultsRef.current : null })}
      />
      {pic1d.mcc && (
        <>
          <div className="field">
            <span className="label">ガス名</span>
            <CommitTextInput value={pic1d.mcc.gas.name} onCommit={(v) => updateGas({ name: v })} />
          </div>
          <div className="field">
            <span className="label">圧力 [Pa]</span>
            <CommitNumberInput value={pic1d.mcc.gas.pressure_pa} onCommit={(v) => updateGas({ pressure_pa: v })} />
          </div>
          <div className="field">
            <span className="label">ガス温度 [K]</span>
            <CommitNumberInput
              value={pic1d.mcc.gas.temperature_k}
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

          <p className="hint">電子プロセス ({pic1d.mcc.electron_processes.length})</p>
          <ProcessList processes={pic1d.mcc.electron_processes} />
          <div className="actions">
            <button className="secondary" onClick={() => updateMcc({ electron_processes: [] })}>
              電子プロセスをクリア
            </button>
          </div>

          <p className="hint">イオンプロセス ({pic1d.mcc.ion_processes.length})</p>
          <ProcessList processes={pic1d.mcc.ion_processes} />
          <div className="actions">
            <button className="secondary" onClick={() => updateMcc({ ion_processes: [] })}>
              イオンプロセスをクリア
            </button>
          </div>

          <div className="field">
            <span className="label">乱数シード</span>
            <CommitNumberInput value={pic1d.mcc.seed} onCommit={(v) => updateMcc({ seed: Math.round(v) })} />
          </div>

          <div className="field">
            <span className="label">電離余剰エネルギー分配</span>
            <select
              value={pic1d.mcc.ionization_split ?? "half"}
              onChange={(e) => updateMcc({ ionization_split: e.target.value as "half" | "random" })}
            >
              <option value="half">半分ずつ (half、Turner互換)</option>
              <option value="random">乱数比 (random)</option>
            </select>
          </div>
          <div className="field">
            <span className="label">イオン断面積の参照系</span>
            <select
              value={pic1d.mcc.ion_energy_frame ?? "lab"}
              onChange={(e) => updateMcc({ ion_energy_frame: e.target.value as "com" | "lab" })}
            >
              <option value="lab">実験室系 (lab)</option>
              <option value="com">重心系 (com、Turner He+/He用)</option>
            </select>
          </div>
          <p className="hint">
            1D (pic1d) は DSMC 連成に未対応のため、DSMCガス場使用トグルはここには表示しません。
          </p>
        </>
      )}

      <div className="field">
        <span className="label">SEE初期エネルギー [eV]</span>
        <CommitNumberInput
          value={pic1d.see_energy_ev ?? 2.0}
          onCommit={(v) => onChange({ ...pic1d, see_energy_ev: v })}
        />
      </div>

      <h2>PIC 1D: 計算設定</h2>
      <div className="field">
        <span className="label">dt [s] (空欄=自動)</span>
        <CommitNullableNumberInput
          value={pic1d.dt ?? null}
          placeholder="自動"
          onCommit={(v) => onChange({ ...pic1d, dt: v })}
        />
      </div>
      <div className="field">
        <span className="label">ステップ数</span>
        <CommitNumberInput
          value={nSteps}
          onCommit={(v) => onChange({ ...pic1d, n_steps: Math.max(1, Math.round(v)) })}
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
          value={pic1d.frame_every ?? 20}
          onCommit={(v) => onChange({ ...pic1d, frame_every: Math.max(1, Math.round(v)) })}
        />
      </div>
      <div className="field">
        <span className="label">平均ステップ数 (空欄=最後の25%)</span>
        <CommitNullableNumberInput
          value={pic1d.avg_steps ?? null}
          placeholder="最後の25%"
          onCommit={(v) => onChange({ ...pic1d, avg_steps: v == null ? null : Math.max(1, Math.round(v)) })}
        />
      </div>
      <div className="field">
        <span className="label">位相ビン数 (周期アニメ用、0=無効)</span>
        <CommitNumberInput
          value={pic1d.phase_bins ?? 40}
          onCommit={(v) => onChange({ ...pic1d, phase_bins: Math.max(0, Math.round(v)) })}
        />
      </div>
      {/* 位相ビン数の推奨値ヒント (PicPanel と同じ考え方: 基本波1周期のステップ数を超える
          ビン数は空のビンを生みうる) */}
      {rfFrequencies.length > 0 && effectiveDt !== null && (pic1d.phase_bins ?? 40) > 0 && (() => {
        const basePeriod = 1 / rfFrequencies[0];
        const stepsPerCycle = basePeriod / effectiveDt;
        const maxBins = Math.max(1, Math.floor(stepsPerCycle));
        const over = (pic1d.phase_bins ?? 40) > maxBins;
        return (
          <p className="hint" style={over ? { color: "#e0b050" } : undefined}>
            目安: 基本波1周期 ≈ {stepsPerCycle.toFixed(1)} ステップのため、ビン数は {maxBins} 以下を推奨
            {over && " (現在の設定では空のビンが生じる可能性があります)"}。
            また平均ステップ数は1周期分 ({Math.ceil(stepsPerCycle)}) 以上、推奨は3周期分 ({3 * Math.ceil(stepsPerCycle)}) 以上。
            {dtFromStarted && " (前回実行の dt で換算)"}
          </p>
        );
      })()}

      <h2>PIC 1D: EEDF/EEPF 区間 (最大{MAX_EEDF1D_REGIONS}個)</h2>
      <p className="hint">
        x1/x2 (電極間の位置 [{unitLabel}]) で区間を指定します。e_max (空欄=自動) は最初の集計ステップで
        「区間内電子の最大エネルギー×1.2」に決まります。
      </p>
      <div className="collector-list">
        {eedfRegions.length === 0 && <div className="muted">(区間なし)</div>}
        {eedfRegions.map((r, i) => (
          <div key={i} className="collector-row" style={{ cursor: "default" }}>
            <input
              type="text"
              className="collector-label-input"
              value={r.label ?? ""}
              onChange={(e) => updateEedfRegion(i, { label: e.target.value })}
            />
            <label className="rf-compact-label" title={`x1 [${unitLabel}]`}>
              x1
              <input
                type="text"
                inputMode="decimal"
                className="rf-compact"
                value={String(mToUnit(r.x1, lengthUnit))}
                onChange={(e) => {
                  const n = Number(e.target.value);
                  if (Number.isFinite(n)) updateEedfRegion(i, { x1: unitToM(n, lengthUnit) });
                }}
              />
            </label>
            <label className="rf-compact-label" title={`x2 [${unitLabel}]`}>
              x2
              <input
                type="text"
                inputMode="decimal"
                className="rf-compact"
                value={String(mToUnit(r.x2, lengthUnit))}
                onChange={(e) => {
                  const n = Number(e.target.value);
                  if (Number.isFinite(n)) updateEedfRegion(i, { x2: unitToM(n, lengthUnit) });
                }}
              />
            </label>
            <input
              type="text"
              inputMode="numeric"
              className="collector-tol-input"
              title="ビン数 (10〜1000)"
              value={String(r.bins ?? 100)}
              onChange={(e) => {
                const n = Number(e.target.value);
                if (Number.isFinite(n)) updateEedfRegion(i, { bins: Math.round(n) });
              }}
            />
            <input
              type="text"
              inputMode="decimal"
              className="collector-tol-input"
              placeholder="自動"
              title="e_max [eV] (空欄=自動決定)"
              value={r.e_max_ev == null ? "" : String(r.e_max_ev)}
              onChange={(e) => {
                const raw = e.target.value;
                if (raw.trim() === "") {
                  updateEedfRegion(i, { e_max_ev: null });
                  return;
                }
                const n = Number(raw);
                if (Number.isFinite(n)) updateEedfRegion(i, { e_max_ev: n });
              }}
            />
            <button className="danger collector-delete" onClick={() => deleteEedfRegion(i)} title="この区間を削除">
              ×
            </button>
          </div>
        ))}
      </div>
      <div className="actions">
        <button className="secondary" onClick={addEedfRegion} disabled={eedfRegions.length >= MAX_EEDF1D_REGIONS}>
          + 区間を追加
        </button>
      </div>

      <h2>PIC 1D: 実行</h2>
      <div className="actions">
        <button onClick={onStart} disabled={!canRun || running}>
          {running ? "実行中..." : "PIC 1D 開始"}
        </button>
        <button className="secondary" onClick={onStop} disabled={!running}>
          停止
        </button>
      </div>
      <div className="field">
        <span className="label">続きから: +ステップ</span>
        <CommitNumberInput value={extraSteps} onCommit={(v) => setExtraSteps(Math.max(1, Math.round(v)))} />
      </div>
      <div className="actions">
        <button className="secondary" onClick={() => onContinue(extraSteps)} disabled={!canContinue}>
          続きから (+{extraSteps}ステップ)
        </button>
      </div>
      <p className="hint">
        「続きから」は現在保持中の粒子状態・時刻から追加実行します
        (フレーム間隔・平均ステップ数・位相ビン数は現在の設定値を使用)。
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
          <h2>PIC 1D: 診断</h2>
          <div className="kv">
            <span>マクロ粒子数 (電子/イオン)</span>
            <span>{frame.counts.n_e ?? "-"} / {frame.counts.n_i ?? "-"}</span>
          </div>
          <div className="kv">
            <span>壁吸収 電子 (左/右)</span>
            <span>{frame.counts.wall_left_e ?? "-"} / {frame.counts.wall_right_e ?? "-"}</span>
          </div>
          <div className="kv">
            <span>壁吸収 イオン (左/右)</span>
            <span>{frame.counts.wall_left_i ?? "-"} / {frame.counts.wall_right_i ?? "-"}</span>
          </div>
          <div className="kv">
            <span>衝突/電離/SEE (累計)</span>
            <span>
              {frame.counts.coll_e ?? "-"} / {frame.counts.ion_events ?? "-"} / {frame.counts.see_events ?? "-"}
            </span>
          </div>
        </>
      )}

      {error && (
        <>
          <h2>PIC 1D エラー</h2>
          <div className="error">{error}</div>
        </>
      )}
    </>
  );
}
