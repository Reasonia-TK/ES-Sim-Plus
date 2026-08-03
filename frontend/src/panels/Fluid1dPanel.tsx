import { useRef, useState } from "react";
import { api } from "../api";
import { CommitNullableNumberInput, CommitNumberInput, formatNumber } from "../CommitInput";
import { LENGTH_UNIT_LABEL, mToUnit, unitToM } from "../units";
import type { LengthUnit } from "../units";
import { ElectrodeEditor } from "./Pic1dPanel";
import { ProcessList } from "./PicPanel";
import { rfComponents } from "../types";
import type {
  Fluid1dFrameMsg,
  Fluid1dSettings,
  Fluid1dStartedMsg,
  Pic1dElectrode,
  Pic1dSettings,
} from "../types";

/**
 * 1D プラズマ流体 (ドリフト拡散 + 電子エネルギー) スタディの設定・実行パネル (Phase D2、
 * prompts/104-109)。pic1d (Pic1dPanel) と同じ一様格子・電極規約を共用し、直接比較できることが
 * 設計目標 (fluid1d.py モジュール docstring参照)。電極編集 (ElectrodeEditor)・断面積編集
 * (ProcessList) は Pic1dPanel/PicPanel の部品をそのまま流用する (export 追加のみ、挙動不変)。
 * 結果の可視化・PIC比較オーバーレイは canvas/Fluid1dPlotView.tsx 側の役割 (Pic1dPanel と同じ
 * 「設定パネルは編集と実行制御のみ」の役割分担、prompts/91)。
 */

interface Props {
  lengthUnit: LengthUnit;
  fluid1d: Fluid1dSettings;
  onChange: (next: Fluid1dSettings) => void;
  canRun: boolean;
  running: boolean;
  onStart: () => void;
  onStop: () => void;
  // 「続きから」ボタン: 直前の実行が done/stop 済みで現在実行中でない場合のみ true。
  // pic1d と同じく geometry/mesh に依存しないため continue 可否は project 変更と無関係
  canContinue: boolean;
  onContinue: (extraSteps: number) => void;
  started: Fluid1dStartedMsg | null;
  frame: Fluid1dFrameMsg | null;
  error: string | null;
  // 「1D PIC の設定を取込」プリセット用。App が現在の pic1d state をそのまま渡す
  // (pic1d は project 非依存の独立 state のため常に非 null だが、将来の呼び出し元での
  // 未設定ケースにも安全に対応できるよう null 許容にしてボタンを disabled にする)
  pic1d: Pic1dSettings | null;
}

// 電極の既定値 (Pic1dPanel.DEFAULT_ELECTRODE と同じ形。fluid1d は fn を持たせない
// = backend validator が fn 設定を拒否するため、Fluid1dPanel は fn に一切触れない)
const DEFAULT_ELECTRODE: Pic1dElectrode = { v_dc: 0.0, waveforms: [], see_gamma: 0.0 };

// project.pic1d.left/right.voltage_rf/waveforms から周波数を集める (RFサイクル換算用)。
// Pic1dPanel.collectPic1dFrequencies と全く同じロジック (fluid1d.left/right も
// Pic1dElectrode 型を共用しているため書式を完全に揃えられる)
function collectFluid1dFrequencies(fluid1d: Fluid1dSettings): number[] {
  const rfFreqs = new Set<number>();
  for (const c of rfComponents(fluid1d.left.voltage_rf)) if (c.freq_hz > 0) rfFreqs.add(c.freq_hz);
  for (const c of rfComponents(fluid1d.right.voltage_rf)) if (c.freq_hz > 0) rfFreqs.add(c.freq_hz);
  if (rfFreqs.size > 0) return Array.from(rfFreqs).sort((a, b) => a - b);
  const wfFreqs = new Set<number>();
  for (const wf of fluid1d.left.waveforms ?? []) if (wf.freq_hz > 0) wfFreqs.add(wf.freq_hz);
  for (const wf of fluid1d.right.waveforms ?? []) if (wf.freq_hz > 0) wfFreqs.add(wf.freq_hz);
  return Array.from(wfFreqs).sort((a, b) => a - b);
}

// サイクル数の表示整形 (Pic1dPanel.formatCycles と同じ)
function formatCycles(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  return v.toPrecision(3).replace("e+", "e");
}

// 「1D PIC の設定を取込」(prompts/109): pic1d 設定から流体側と共通の物理パラメータのみを写す。
// - left/right は fn (FN電界放出) を除いて写す (流体は FN 未対応、backend の _check_no_fn が拒否する)
// - mcc が有効なときのみガス圧/温度/電子断面積を写す (mcc 無効なら流体側の現在値をそのまま維持)
// - n_cells・mu_i_ref/n_ref_m3/t_i_ev・dt/n_steps/frame_every/avg_steps 等の流体固有パラメータは維持する
function applyPic1dPreset(pic1d: Pic1dSettings, current: Fluid1dSettings): Fluid1dSettings {
  const stripFn = (e: Pic1dElectrode): Pic1dElectrode => {
    const { fn: _fn, ...rest } = e;
    return rest;
  };
  const next: Fluid1dSettings = {
    ...current,
    gap_m: pic1d.gap_m,
    left: stripFn(pic1d.left),
    right: stripFn(pic1d.right),
    init_density_m3: pic1d.init_density_m3,
    init_te_ev: pic1d.init_te_ev,
    ion_mass_amu: pic1d.ion_mass_amu,
    phase_bins: pic1d.phase_bins,
  };
  if (pic1d.mcc) {
    next.gas_pressure_pa = pic1d.mcc.gas.pressure_pa;
    next.gas_temperature_k = pic1d.mcc.gas.temperature_k;
    next.electron_processes = pic1d.mcc.electron_processes;
  }
  return next;
}

export default function Fluid1dPanel({
  lengthUnit,
  fluid1d,
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
  pic1d,
}: Props) {
  const unitLabel = LENGTH_UNIT_LABEL[lengthUnit];

  // --- 電子断面積の LXCat インポート (電子のみ、PicPanel/Pic1dPanel の同名処理の電子側だけの複製) ---
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
          onChange({ ...fluid1d, electron_processes: res.processes });
        })
        .catch((e) => setLxcatError(String(e)));
    };
    reader.readAsText(file);
  };

  // --- RFサイクル換算 (Pic1dPanel と同じ考え方) ---
  const rfFrequencies = collectFluid1dFrequencies(fluid1d);
  const effectiveDt = fluid1d.dt ?? started?.dt ?? null;
  const dtFromStarted = fluid1d.dt == null && started != null;
  const nSteps = fluid1d.n_steps ?? 20000;

  // --- 実行進捗 (続き実行では frame.step が通算のため、区間開始オフセットを引く。Pic1dPanel と同じ設計) ---
  const segOffset = started?.step_offset ?? 0;
  const segStep = Math.max(0, (frame?.step ?? segOffset) - segOffset);
  const progressPct = started && started.n_steps > 0 ? Math.min(100, (segStep / started.n_steps) * 100) : 0;

  // 「続きから」の追加ステップ数。既定は現在の n_steps 設定を踏襲するが、以後は独立に編集できる
  const [extraSteps, setExtraSteps] = useState(nSteps);

  return (
    <>
      <h2>流体 (1D): プリセット</h2>
      <div className="actions">
        <button
          className="secondary"
          onClick={() => pic1d && onChange(applyPic1dPreset(pic1d, fluid1d))}
          disabled={!pic1d}
          title={
            pic1d
              ? "現在の 1D PIC 設定から共通パラメータを取込みます (確認なし、Undo 対象外)"
              : "1D PIC (pic1d) が未設定です"
          }
        >
          1D PIC の設定を取込
        </button>
      </div>
      <p className="hint">
        現在の 1D PIC (PIC-MCC 1D) 設定からギャップ長・電極 (DC/RF重畳/CSV波形/SEE収率)・初期密度・
        初期Te・イオン質量・ガス圧/温度・電子断面積・位相ビン数を取込みます。FN電界放出は流体側が
        未対応のため取込みません。セル数・μ_i 等の流体固有パラメータは維持されます。
      </p>

      <h2>流体 (1D): 形状</h2>
      <div className="field">
        <span className="label">ギャップ長 [{unitLabel}]</span>
        <CommitNumberInput
          value={mToUnit(fluid1d.gap_m, lengthUnit)}
          onCommit={(v) => onChange({ ...fluid1d, gap_m: Math.max(1e-9, unitToM(v, lengthUnit)) })}
        />
      </div>
      <div className="field">
        <span className="label">セル数</span>
        <CommitNumberInput
          value={fluid1d.n_cells}
          onCommit={(v) => onChange({ ...fluid1d, n_cells: Math.max(16, Math.round(v)) })}
        />
      </div>

      <h2>流体 (1D): 電極</h2>
      {/* Pic1dPanel.ElectrodeEditor を流用 (showFn=false で FN電界放出トグルを隠す。
          流体は backend の _check_no_fn validator が fn 指定を拒否するため入口自体を出さない) */}
      <ElectrodeEditor
        title="左電極 (x=0)"
        electrode={fluid1d.left ?? DEFAULT_ELECTRODE}
        onChange={(next) => onChange({ ...fluid1d, left: next })}
        showFn={false}
      />
      <ElectrodeEditor
        title="右電極 (x=gap)"
        electrode={fluid1d.right ?? DEFAULT_ELECTRODE}
        onChange={(next) => onChange({ ...fluid1d, right: next })}
        showFn={false}
      />

      <h2>流体 (1D): 初期値</h2>
      <div className="field">
        <span className="label">初期密度 [m^-3]</span>
        <CommitNumberInput
          value={fluid1d.init_density_m3}
          onCommit={(v) => onChange({ ...fluid1d, init_density_m3: v })}
        />
      </div>
      <div className="field">
        <span className="label">初期 Te [eV]</span>
        <CommitNumberInput
          value={fluid1d.init_te_ev ?? 2.0}
          onCommit={(v) => onChange({ ...fluid1d, init_te_ev: v })}
        />
      </div>
      <div className="field">
        <span className="label">イオン質量 [amu]</span>
        <CommitNumberInput
          value={fluid1d.ion_mass_amu ?? 39.948}
          onCommit={(v) => onChange({ ...fluid1d, ion_mass_amu: v })}
        />
      </div>
      <div className="field">
        <span className="label">イオン温度 T_i [eV]</span>
        <CommitNumberInput value={fluid1d.t_i_ev ?? 0.026} onCommit={(v) => onChange({ ...fluid1d, t_i_ev: v })} />
      </div>
      <div className="field">
        <span className="label">μ_i 基準値 [m^2/(V・s)]</span>
        <CommitNumberInput
          value={fluid1d.mu_i_ref ?? 1.45e-1}
          onCommit={(v) => onChange({ ...fluid1d, mu_i_ref: v })}
        />
      </div>
      <div className="field">
        <span className="label">イオン移動度モデル</span>
        <select
          value={fluid1d.ion_mobility_model ?? "frost"}
          onChange={(e) => onChange({ ...fluid1d, ion_mobility_model: e.target.value as "frost" | "const" })}
        >
          <option value="frost">修正 Frost 式 (電界強度依存、推奨)</option>
          <option value="const">一定値 (μ_i 基準値のみ、旧来互換)</option>
        </select>
      </div>
      <p className="hint">
        修正 Frost 式 μ_i(E/N) = μ_L/√(1+(E/N)/C) は Ar+/Ar の測定値 (Ellis et al., At. Data
        Nucl. Data Tables 17, 177 (1976)) への大まかな工学的近似であり、厳密なフィットではありません
        (ガス種が異なれば C の再調整が必要)。拡散係数 D_i は低電界値 (D_i=μ_L・T_i) のまま据え置きます
        (強電界域はドリフト支配的で拡散寄与が小さく、Einstein の関係自体も非平衡近似のため)。
      </p>
      {(fluid1d.ion_mobility_model ?? "frost") === "frost" && (
        <div className="field">
          <span className="label">Frost C [Td]</span>
          <CommitNumberInput
            value={fluid1d.frost_c_td ?? 150.0}
            onCommit={(v) => onChange({ ...fluid1d, frost_c_td: Math.max(1e-9, v) })}
          />
        </div>
      )}
      <div className="field">
        <span className="label">μ_i 基準ガス密度 [m^-3]</span>
        <CommitNumberInput
          value={fluid1d.n_ref_m3 ?? 3.22e22}
          onCommit={(v) => onChange({ ...fluid1d, n_ref_m3: v })}
        />
      </div>

      <h2>流体 (1D): ガス</h2>
      <div className="field">
        <span className="label">圧力 [Pa]</span>
        <CommitNumberInput
          value={fluid1d.gas_pressure_pa}
          onCommit={(v) => onChange({ ...fluid1d, gas_pressure_pa: v })}
        />
      </div>
      <div className="field">
        <span className="label">ガス温度 [K]</span>
        <CommitNumberInput
          value={fluid1d.gas_temperature_k ?? 300.0}
          onCommit={(v) => onChange({ ...fluid1d, gas_temperature_k: v })}
        />
      </div>

      <h2>流体 (1D): 電子断面積</h2>
      <p className="hint">
        空 (未読込) の場合は eduPIC の Ar 解析式 (弾性・励起・電離) を既定で使用します
        (Pic1dPanel の MCC 断面積編集と同じ LXCat 形式のインポートに対応)。
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
      <p className="hint">電子プロセス ({(fluid1d.electron_processes ?? []).length})</p>
      <ProcessList processes={fluid1d.electron_processes ?? []} />
      <div className="actions">
        <button className="secondary" onClick={() => onChange({ ...fluid1d, electron_processes: [] })}>
          電子プロセスをクリア (eduPIC Ar 解析式に戻す)
        </button>
      </div>

      <h2>流体 (1D): 実行設定</h2>
      <div className="field">
        <span className="label">dt [s] (空欄=自動)</span>
        <CommitNullableNumberInput
          value={fluid1d.dt ?? null}
          placeholder="自動"
          onCommit={(v) => onChange({ ...fluid1d, dt: v })}
        />
      </div>
      <div className="field">
        <span className="label">ステップ数</span>
        <CommitNumberInput
          value={nSteps}
          onCommit={(v) => onChange({ ...fluid1d, n_steps: Math.max(1, Math.round(v)) })}
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
          value={fluid1d.frame_every ?? 200}
          onCommit={(v) => onChange({ ...fluid1d, frame_every: Math.max(1, Math.round(v)) })}
        />
      </div>
      <div className="field">
        <span className="label">平均ステップ数 (空欄=最後の25%)</span>
        <CommitNullableNumberInput
          value={fluid1d.avg_steps ?? null}
          placeholder="最後の25%"
          onCommit={(v) => onChange({ ...fluid1d, avg_steps: v == null ? null : Math.max(1, Math.round(v)) })}
        />
      </div>
      <div className="field">
        <span className="label">位相ビン数 (周期アニメ用、0=無効)</span>
        <CommitNumberInput
          value={fluid1d.phase_bins ?? 40}
          onCommit={(v) => onChange({ ...fluid1d, phase_bins: Math.max(0, Math.round(v)) })}
        />
      </div>
      <div className="field">
        <span className="label">壁 IEDF ビン数 (0=無効)</span>
        <CommitNumberInput
          value={fluid1d.wall_iedf_bins ?? 100}
          onCommit={(v) => onChange({ ...fluid1d, wall_iedf_bins: Math.max(0, Math.min(1000, Math.round(v))) })}
        />
      </div>
      <p className="hint">
        無衝突シース近似 (位相分解シース電圧 + イオン通過時間フィルタ) により壁入射イオンエネルギー
        分布を再構成します。CX 衝突による低エネルギー成分は含みません (prompts/116)。
      </p>
      {/* 位相ビン数の推奨値ヒント (Pic1dPanel と同じ考え方) */}
      {rfFrequencies.length > 0 && effectiveDt !== null && (fluid1d.phase_bins ?? 40) > 0 && (() => {
        const basePeriod = 1 / rfFrequencies[0];
        const stepsPerCycle = basePeriod / effectiveDt;
        const maxBins = Math.max(1, Math.floor(stepsPerCycle));
        const over = (fluid1d.phase_bins ?? 40) > maxBins;
        return (
          <p className="hint" style={over ? { color: "#e0b050" } : undefined}>
            目安: 基本波1周期 ≈ {stepsPerCycle.toFixed(1)} ステップのため、ビン数は {maxBins} 以下を推奨
            {over && " (現在の設定では空のビンが生じる可能性があります)"}。
            また平均ステップ数は1周期分 ({Math.ceil(stepsPerCycle)}) 以上、推奨は3周期分 ({3 * Math.ceil(stepsPerCycle)}) 以上。
            {dtFromStarted && " (前回実行の dt で換算)"}
          </p>
        );
      })()}

      <h2>流体 (1D): 実行</h2>
      <div className="actions">
        <button onClick={onStart} disabled={!canRun || running}>
          {running ? "実行中..." : "流体 1D 開始"}
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
        「続きから」は現在保持中の場・時刻から追加実行します
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
          <h2>流体 (1D): 診断</h2>
          <div className="kv">
            <span>全域密度 電子/イオン [m^-2]</span>
            <span>{frame.counts.n_e_total ?? "-"} / {frame.counts.n_i_total ?? "-"}</span>
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
            <span>電離生成 (累計)</span>
            <span>{frame.counts.gen_total ?? "-"}</span>
          </div>
        </>
      )}

      {error && (
        <>
          <h2>流体 (1D) エラー</h2>
          <div className="error">{error}</div>
        </>
      )}
    </>
  );
}

// project.fluid1d が未設定の場合の初期表示に使う既定値 (App が使う、prompts/108)。pic1d 同様
// particles/pic とは独立の state で管理する想定 (geometry/mesh には一切依存しない)。
// gap_m/init_density_m3/gas_pressure_pa は backend に既定値が無い必須項目のため、
// backend/tests/test_fluid1d.py の CCP 定常スモークテスト設定 (典型的な Ar CCP 条件) に
// 揃える。それ以外は backend/es_sim/schema.py Fluid1dSettings の Field 既定値と一致させる
export const DEFAULT_FLUID1D: Fluid1dSettings = {
  gap_m: 0.025,
  n_cells: 200,
  // left/right は同じ内容でも別オブジェクトにする (App.tsx の DEFAULT_PIC1D と同じ流儀。
  // 片方だけを編集したときに参照共有で意図せずもう片方まで変わらないようにするため)
  left: { v_dc: 0.0, waveforms: [], see_gamma: 0.0 },
  right: { v_dc: 0.0, waveforms: [], see_gamma: 0.0 },
  init_density_m3: 1.0e15,
  init_te_ev: 2.0,
  gas_pressure_pa: 50.0,
  gas_temperature_k: 300.0,
  ion_mass_amu: 39.948,
  mu_i_ref: 1.45e-1,
  n_ref_m3: 3.22e22,
  t_i_ev: 0.026,
  ion_mobility_model: "frost",
  frost_c_td: 150.0,
  electron_processes: [],
  dt: null,
  n_steps: 20000,
  frame_every: 200,
  avg_steps: null,
  phase_bins: 40,
  wall_iedf_bins: 100,
};
