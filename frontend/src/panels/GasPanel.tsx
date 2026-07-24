import { useEffect, useRef } from "react";
import { CommitNullableNumberInput, CommitNumberInput, CommitTextInput } from "../CommitInput";
import { Toggle } from "../Toggle";
import type { DsmcBoundary, DsmcBoundaryType, DsmcGas, DsmcResult, DsmcSettings, Point, Project } from "../types";
import { LENGTH_UNIT_LABEL, mToUnit, unitToM } from "../units";
import type { LengthUnit } from "../units";

/**
 * ガスパネル (タブ4): DSMC 定常ガス流れ
 * - DSMC 有効チェック (project.dsmc の有無)
 * - ガス種 (VHS分子モデル) 設定
 * - 境界条件リスト (domain エッジ番号 + タイプ + 温度/圧力、追加/削除)
 * - 初期条件・積分設定
 * - 「ガス流れ計算」ボタン (POST /dsmc) と結果サマリ
 * - 結果フィールド (n/T/|u|/p) 選択 + 対数スケール → App 側で PicFieldView 型に変換して CadCanvas へ渡す
 *
 * project.dsmc は Project 本体のフィールドなので、編集は App 側の commitProject 経由で
 * Undo/Redo 履歴に積まれる (ジオメトリ・メッシュ設定と同じ扱い)。
 */

// 結果表示セレクトの選択肢。"u" は要素ごとの流速ベクトルの大きさ |u| を表示する
export type GasResultField = "n" | "t" | "u" | "p";

export const GAS_FIELD_OPTIONS: { value: GasResultField; label: string }[] = [
  { value: "n", label: "数密度 n [m^-3]" },
  { value: "t", label: "温度 T [K]" },
  { value: "u", label: "流速 |u| [m/s]" },
  { value: "p", label: "圧力 p [Pa]" },
];

// フィールドキーごとの単位 (App 側で CadCanvas 用の PicFieldView 構築に使う)。
// DSMC 結果はすべて要素値 (nodeBased=false)
export const GAS_FIELD_META: Record<GasResultField, { unit: string }> = {
  n: { unit: "m^-3" },
  t: { unit: "K" },
  u: { unit: "m/s" },
  p: { unit: "Pa" },
};

// DsmcResult から表示用の要素値配列を取り出す (u のみ大きさへ変換)
export function gasFieldValues(result: DsmcResult, field: GasResultField): number[] {
  if (field === "u") return result.u.map(([ux, uy]) => Math.hypot(ux, uy));
  return result[field];
}

const DEFAULT_GAS: DsmcGas = {
  name: "Ar",
  mass_amu: 39.948,
  d_ref_m: 4.17e-10,
  omega: 0.81,
  t_ref_k: 273.0,
};

// 有効チェックを一度オフにしても、再度オンにしたときに直前の値を復元できるよう保持する既定値
const DEFAULT_DSMC: DsmcSettings = {
  gas: DEFAULT_GAS,
  boundaries: [],
  wall_temperature_k: 300.0,
  init_pressure_pa: 1.0,
  init_temperature_k: 300.0,
  n_particles: 50000,
  dt: null,
  n_steps: 2000,
  avg_steps: 500,
  seed: 0,
  threads: 1,
  smoothing_passes: 0,
};

const DEFAULT_BOUNDARY: DsmcBoundary = {
  edges: [],
  p1: null,
  p2: null,
  type: "wall",
  temperature_k: 300.0,
  pressure_pa: null,
  flow_sccm: null,
};

const BOUNDARY_TYPE_LABELS: Record<DsmcBoundaryType, string> = {
  wall: "壁 (拡散反射)",
  symmetry: "対称",
  inlet: "流入 (圧力リザーバ)",
  outlet: "流出 (圧力/真空)",
};

// "0,2,3" のようなカンマ区切りテキストをエッジ番号配列にパースする (不正な値は無視)
function parseEdgesText(text: string): number[] {
  return text
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s !== "")
    .map((s) => Math.round(Number(s)))
    .filter((n) => Number.isFinite(n) && n >= 0);
}

// 適用範囲の指定方法。p1/p2 が両方指定されていれば線分指定 (edges との併用も可だが、
// UI 上はどちらか一方のモードで編集する)
type RangeMode = "edges" | "segment";

function rangeModeOf(b: DsmcBoundary): RangeMode {
  return b.p1 != null && b.p2 != null ? "segment" : "edges";
}

// inlet の指定方法。flow_sccm が指定されていれば流量、それ以外は圧力
type InletSpecMode = "pressure" | "flow";

function inletSpecModeOf(b: DsmcBoundary): InletSpecMode {
  return b.flow_sccm != null ? "flow" : "pressure";
}

// 実行中の進捗 (WebSocket started/progress メッセージ由来)
export interface GasProgress {
  step: number;
  nSteps: number;
  nParticles: number;
}

interface Props {
  project: Project;
  // 長さの表示・入力単位 (mm/µm)。project 内部は常に m のまま
  lengthUnit: LengthUnit;
  dsmc: DsmcSettings | null;
  onChange: (next: DsmcSettings | null) => void;
  canRun: boolean;
  running: boolean;
  onRun: () => void;
  onStop: () => void;
  progress: GasProgress | null;
  result: DsmcResult | null;
  error: string | null;
  resultField: GasResultField;
  onResultFieldChange: (v: GasResultField) => void;
  logScale: boolean;
  onLogScaleChange: (v: boolean) => void;
  // 「粒子を表示」チェックボックス (実行中のライブ粒子表示のON/OFF、既定 ON、prompts/66)。
  // App 側 state で保持する (ローカルにしないのは setup/results 両ページで同じ値を共有するため)
  showParticles: boolean;
  onShowParticlesChange: (v: boolean) => void;

  // 表示モード: "all"=従来通り全表示 (既定・後方互換)、"setup"=設定/実行UIのみ、
  // "results"=結果表示のみ (結果ノード用インスペクタページで使う)
  mode?: "all" | "setup" | "results";
}

export default function GasPanel({
  project,
  lengthUnit,
  dsmc,
  onChange,
  canRun,
  running,
  onRun,
  onStop,
  progress,
  result,
  error,
  resultField,
  onResultFieldChange,
  logScale,
  onLogScaleChange,
  showParticles,
  onShowParticlesChange,
  mode = "all",
}: Props) {
  // mode が "all" のときは従来通り両方表示。それ以外は該当モードのみ表示する
  const show = (m: "setup" | "results") => mode === "all" || mode === m;
  const unitLabel = LENGTH_UNIT_LABEL[lengthUnit];

  const dsmcDefaultsRef = useRef<DsmcSettings>(dsmc ?? DEFAULT_DSMC);
  useEffect(() => {
    if (dsmc) dsmcDefaultsRef.current = dsmc;
  }, [dsmc]);

  const updateGas = (patch: Partial<DsmcGas>) => {
    if (!dsmc) return;
    onChange({ ...dsmc, gas: { ...dsmc.gas, ...patch } });
  };

  const addBoundary = () => {
    if (!dsmc) return;
    onChange({ ...dsmc, boundaries: [...dsmc.boundaries, { ...DEFAULT_BOUNDARY }] });
  };

  const updateBoundary = (i: number, patch: Partial<DsmcBoundary>) => {
    if (!dsmc) return;
    const next = dsmc.boundaries.slice();
    next[i] = { ...next[i], ...patch };
    onChange({ ...dsmc, boundaries: next });
  };

  const removeBoundary = (i: number) => {
    if (!dsmc) return;
    onChange({ ...dsmc, boundaries: dsmc.boundaries.filter((_, idx) => idx !== i) });
  };

  const domainEdgeCount = project.geometry.domain.polygon.length;

  return (
    <>
      {show("setup") && (
      <>
      <h2>ガス流れ (DSMC)</h2>
      <Toggle
        label="有効"
        checked={dsmc !== null}
        onChange={(v) => onChange(v ? dsmcDefaultsRef.current : null)}
      />
      <p className="hint">
        NTC 法 + VHS 分子モデルによる定常ガス流れ解析。既存の三角形メッシュをセルとして使う
        (平面2Dのみ対応)。結果は PIC の MCC で「DSMCガス場を使用」を有効にすると背景ガスとして使える。
      </p>
      </>
      )}

      {dsmc && (
        <>
          {show("setup") && (
          <>
          <h2>ガス種 (VHS)</h2>
          <div className="field">
            <span className="label">ガス名</span>
            <CommitTextInput value={dsmc.gas.name} onCommit={(v) => updateGas({ name: v })} />
          </div>
          <div className="field">
            <span className="label">分子質量 [amu]</span>
            <CommitNumberInput value={dsmc.gas.mass_amu} onCommit={(v) => updateGas({ mass_amu: v })} />
          </div>
          <div className="field">
            <span className="label">VHS基準直径 [m]</span>
            <CommitNumberInput value={dsmc.gas.d_ref_m} onCommit={(v) => updateGas({ d_ref_m: v })} />
          </div>
          <div className="field">
            <span className="label">粘性温度指数 ω</span>
            <CommitNumberInput value={dsmc.gas.omega} onCommit={(v) => updateGas({ omega: v })} />
          </div>
          <div className="field">
            <span className="label">基準温度 [K]</span>
            <CommitNumberInput value={dsmc.gas.t_ref_k} onCommit={(v) => updateGas({ t_ref_k: v })} />
          </div>

          <h2>境界条件 (domain 外周)</h2>
          <p className="hint">
            domain 外周のエッジ番号 (カンマ区切り、0-indexed、現在 {domainEdgeCount} 辺)、または
            外周上の線分 p1-p2 (部分区間) で適用範囲を指定します。未指定のエッジは壁 (拡散反射) になります。
            線分指定は外周上の線分に載る境界メッシュエッジへ適用されます (電極との隙間など部分区間の指定用)。
          </p>
          <p className="hint">
            流入口 (inlet) は圧力指定に加えて流量指定 [sccm] も選べます。
            1 sccm = 標準状態の 1 cm³/min (奥行き1m換算)。流量指定では入射粒子は壁反射になり、正味流量が指定値に一致します。
          </p>
          <div className="collector-list">
            {dsmc.boundaries.length === 0 && <div className="muted">(未指定。すべて壁境界として扱われます)</div>}
            {dsmc.boundaries.map((b, i) => {
              const rangeMode = rangeModeOf(b);
              const specMode = inletSpecModeOf(b);
              const p1: Point = b.p1 ?? [0, 0];
              const p2: Point = b.p2 ?? [0, 0];
              return (
                <div className="dsmc-boundary-row" key={i}>
                  <div className="dsmc-boundary-row-main">
                    <select
                      className="dsmc-range-select"
                      value={rangeMode}
                      onChange={(e) => {
                        const mode = e.target.value as RangeMode;
                        if (mode === "segment") {
                          updateBoundary(i, { edges: [], p1: b.p1 ?? [0, 0], p2: b.p2 ?? [0, 0] });
                        } else {
                          updateBoundary(i, { p1: null, p2: null });
                        }
                      }}
                    >
                      <option value="edges">エッジ番号</option>
                      <option value="segment">線分 (p1-p2)</option>
                    </select>
                    {rangeMode === "edges" && (
                      <CommitTextInput
                        className="dsmc-edges-input"
                        value={b.edges.join(",")}
                        onCommit={(v) => updateBoundary(i, { edges: parseEdgesText(v) })}
                      />
                    )}
                    <select
                      value={b.type}
                      onChange={(e) => updateBoundary(i, { type: e.target.value as DsmcBoundaryType })}
                    >
                      {(Object.keys(BOUNDARY_TYPE_LABELS) as DsmcBoundaryType[]).map((t) => (
                        <option key={t} value={t}>
                          {BOUNDARY_TYPE_LABELS[t]}
                        </option>
                      ))}
                    </select>
                    <button className="danger collector-delete" onClick={() => removeBoundary(i)} title="この境界を削除">
                      ×
                    </button>
                  </div>
                  {rangeMode === "segment" && (
                    <div className="dsmc-boundary-row-sub">
                      <label className="rf-compact-label" title={`p1 x [${unitLabel}]`}>
                        p1x
                        <CommitNumberInput
                          className="rf-compact"
                          value={mToUnit(p1[0], lengthUnit)}
                          step="0.1"
                          onCommit={(x) => updateBoundary(i, { p1: [unitToM(x, lengthUnit), p1[1]] })}
                        />
                      </label>
                      <label className="rf-compact-label" title={`p1 y [${unitLabel}]`}>
                        p1y
                        <CommitNumberInput
                          className="rf-compact"
                          value={mToUnit(p1[1], lengthUnit)}
                          step="0.1"
                          onCommit={(y) => updateBoundary(i, { p1: [p1[0], unitToM(y, lengthUnit)] })}
                        />
                      </label>
                      <label className="rf-compact-label" title={`p2 x [${unitLabel}]`}>
                        p2x
                        <CommitNumberInput
                          className="rf-compact"
                          value={mToUnit(p2[0], lengthUnit)}
                          step="0.1"
                          onCommit={(x) => updateBoundary(i, { p2: [unitToM(x, lengthUnit), p2[1]] })}
                        />
                      </label>
                      <label className="rf-compact-label" title={`p2 y [${unitLabel}]`}>
                        p2y
                        <CommitNumberInput
                          className="rf-compact"
                          value={mToUnit(p2[1], lengthUnit)}
                          step="0.1"
                          onCommit={(y) => updateBoundary(i, { p2: [p2[0], unitToM(y, lengthUnit)] })}
                        />
                      </label>
                    </div>
                  )}
                  <div className="dsmc-boundary-row-sub">
                    <label className="rf-compact-label" title="温度 [K]">
                      T
                      <CommitNumberInput
                        className="rf-compact"
                        value={b.temperature_k}
                        onCommit={(v) => updateBoundary(i, { temperature_k: v })}
                      />
                    </label>
                    {b.type === "inlet" && (
                      <select
                        className="dsmc-spec-select"
                        value={specMode}
                        onChange={(e) => {
                          const mode = e.target.value as InletSpecMode;
                          if (mode === "flow") {
                            updateBoundary(i, { pressure_pa: null, flow_sccm: b.flow_sccm ?? 1.0 });
                          } else {
                            updateBoundary(i, { flow_sccm: null, pressure_pa: b.pressure_pa ?? 1.0 });
                          }
                        }}
                      >
                        <option value="pressure">圧力 [Pa]</option>
                        <option value="flow">流量 [sccm]</option>
                      </select>
                    )}
                    {(b.type === "outlet" || (b.type === "inlet" && specMode === "pressure")) && (
                      <label className="rf-compact-label" title="圧力 [Pa] (outlet は空欄/0で真空排気)">
                        p
                        <input
                          type="text"
                          inputMode="decimal"
                          className="rf-compact"
                          value={b.pressure_pa == null ? "" : String(b.pressure_pa)}
                          placeholder={b.type === "outlet" ? "真空" : ""}
                          onChange={(e) => {
                            const raw = e.target.value;
                            if (raw.trim() === "") {
                              updateBoundary(i, { pressure_pa: null });
                              return;
                            }
                            const n = Number(raw);
                            if (Number.isFinite(n)) updateBoundary(i, { pressure_pa: n });
                          }}
                        />
                      </label>
                    )}
                    {b.type === "inlet" && specMode === "flow" && (
                      <label className="rf-compact-label" title="流量 [sccm] (1 sccm = 標準状態の1 cm³/min、奥行き1m換算)">
                        Q
                        <input
                          type="text"
                          inputMode="decimal"
                          className="rf-compact"
                          value={b.flow_sccm == null ? "" : String(b.flow_sccm)}
                          onChange={(e) => {
                            const raw = e.target.value;
                            if (raw.trim() === "") {
                              updateBoundary(i, { flow_sccm: null });
                              return;
                            }
                            const n = Number(raw);
                            if (Number.isFinite(n)) updateBoundary(i, { flow_sccm: n });
                          }}
                        />
                      </label>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="actions">
            <button className="secondary" onClick={addBoundary}>
              境界を追加
            </button>
          </div>

          <h2>初期条件・計算設定</h2>
          <div className="field">
            <span className="label">壁温 (未指定エッジ) [K]</span>
            <CommitNumberInput
              value={dsmc.wall_temperature_k}
              onCommit={(v) => onChange({ ...dsmc, wall_temperature_k: v })}
            />
          </div>
          <div className="field">
            <span className="label">初期充填圧 [Pa]</span>
            <CommitNumberInput
              value={dsmc.init_pressure_pa}
              onCommit={(v) => onChange({ ...dsmc, init_pressure_pa: v })}
            />
          </div>
          <div className="field">
            <span className="label">初期温度 [K]</span>
            <CommitNumberInput
              value={dsmc.init_temperature_k}
              onCommit={(v) => onChange({ ...dsmc, init_temperature_k: v })}
            />
          </div>
          <div className="field">
            <span className="label">目標粒子数</span>
            <CommitNumberInput
              value={dsmc.n_particles}
              onCommit={(v) => onChange({ ...dsmc, n_particles: Math.max(1, Math.round(v)) })}
            />
          </div>
          <div className="field">
            <span className="label">dt [s] (空欄=自動)</span>
            <CommitNullableNumberInput
              value={dsmc.dt}
              placeholder="自動"
              onCommit={(v) => onChange({ ...dsmc, dt: v })}
            />
          </div>
          <div className="field">
            <span className="label">ステップ数</span>
            <CommitNumberInput
              value={dsmc.n_steps}
              onCommit={(v) => onChange({ ...dsmc, n_steps: Math.max(1, Math.round(v)) })}
            />
          </div>
          <div className="field">
            <span className="label">平均ステップ数</span>
            <CommitNumberInput
              value={dsmc.avg_steps}
              onCommit={(v) => onChange({ ...dsmc, avg_steps: Math.max(1, Math.round(v)) })}
            />
          </div>
          <div className="field">
            <span className="label">乱数シード</span>
            <CommitNumberInput value={dsmc.seed} onCommit={(v) => onChange({ ...dsmc, seed: Math.round(v) })} />
          </div>
          <div className="field">
            <span className="label">スレッド数</span>
            <CommitNumberInput
              value={dsmc.threads ?? 1}
              onCommit={(v) => onChange({ ...dsmc, threads: Math.max(1, Math.round(v)) })}
            />
          </div>
          <p className="hint">
            walk 探索の並列スレッド数。結果は1と完全一致。CPUコア数程度まで
          </p>
          <div className="field">
            <span className="label">平滑化回数</span>
            <CommitNumberInput
              value={dsmc.smoothing_passes ?? 0}
              onCommit={(v) => onChange({ ...dsmc, smoothing_passes: Math.min(20, Math.max(0, Math.round(v))) })}
            />
          </div>
          <p className="hint">
            隣接セル拡散による統計ノイズの平滑化。0=無効。総量 (質量・運動量・エネルギー) は
            保存され、結果表示と PIC 連成の両方に適用されます。目安 1〜5
          </p>

          <Toggle
            label="粒子を表示"
            checked={showParticles}
            onChange={onShowParticlesChange}
            title="実行中のキャンバスに間引き粒子位置をライブ表示する (PICのライブ粒子表示と同様)"
          />
          <div className="actions">
            <button onClick={onRun} disabled={!canRun || running}>
              {running ? "計算中..." : "ガス流れ計算"}
            </button>
            <button className="secondary" onClick={onStop} disabled={!running}>
              停止
            </button>
          </div>

          {running && progress && (
            <>
              <div className="gas-progress">
                <div
                  className="gas-progress-bar"
                  style={{
                    width: `${progress.nSteps > 0 ? Math.min(100, (progress.step / progress.nSteps) * 100) : 0}%`,
                  }}
                />
              </div>
              <div className="kv">
                <span>進捗</span>
                <span>
                  ステップ {progress.step} / {progress.nSteps} (粒子数 {progress.nParticles})
                </span>
              </div>
            </>
          )}
          </>
          )}

          {error && (
            <>
              <h2>エラー</h2>
              <div className="error">{error}</div>
            </>
          )}

          {show("results") && result && (
            <>
              <h2>結果</h2>
              <div className="kv">
                <span>シミュレーション粒子数</span>
                <span>{result.n_particles}</span>
              </div>
              <div className="kv">
                <span>マクロ重み (実分子数/粒子)</span>
                <span>{result.macro_weight.toExponential(3)}</span>
              </div>
              <div className="kv">
                <span>実際に使った dt [s]</span>
                <span>{result.dt.toExponential(3)}</span>
              </div>
              <div className="kv">
                <span>流入 (平均区間、実分子数)</span>
                <span>{result.inflow.toExponential(3)}</span>
              </div>
              <div className="kv">
                <span>流出 (平均区間、実分子数)</span>
                <span>{result.outflow.toExponential(3)}</span>
              </div>

              <div className="field">
                <span className="label">結果表示</span>
                <select value={resultField} onChange={(e) => onResultFieldChange(e.target.value as GasResultField)}>
                  {GAS_FIELD_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </div>
              <Toggle label="対数スケール" checked={logScale} onChange={onLogScaleChange} />
            </>
          )}
        </>
      )}

      {/* results専用ページで未実行の場合のヒント (mode="all" の従来ページでは出さない)。
          dsmc が無効の場合も含め、結果セクションが表示されないケースをまとめて拾う */}
      {mode === "results" && !(dsmc && result) && (
        <p className="hint">DSMC計算が未実行です。スタディ「DSMC」から実行してください。</p>
      )}
    </>
  );
}
