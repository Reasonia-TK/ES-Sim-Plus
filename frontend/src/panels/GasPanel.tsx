import { useEffect, useRef } from "react";
import { CommitNullableNumberInput, CommitNumberInput, CommitTextInput } from "../CommitInput";
import { Toggle } from "../Toggle";
import type { DsmcBoundary, DsmcBoundaryType, DsmcGas, DsmcResult, DsmcSettings, MeshResult, Point, Project } from "../types";
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

// 有効チェックを一度オフにしても、再度オンにしたときに直前の値を復元できるよう保持する既定値。
// App 側のガス境界配置ツール (キャンバス2点クリック) が project.dsmc==null から有効化する際にも流用する
export const DEFAULT_DSMC: DsmcSettings = {
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
  mesh_scale: 1.0,
};

// 「境界を追加」ボタン・キャンバスの「ガス境界」ツールが追加する新規境界の共通既定値
// (両者で内容を一致させるため App 側からも import して使う)
export const DEFAULT_BOUNDARY: DsmcBoundary = {
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
  // Mesh ボタンで生成済みのメッシュ (無ければ null)。粒子数の目安表示のセル数に使う
  meshResult: MeshResult | null;
  // 長さの表示・入力単位 (mm/µm)。project 内部は常に m のまま
  lengthUnit: LengthUnit;
  dsmc: DsmcSettings | null;
  onChange: (next: DsmcSettings | null) => void;
  canRun: boolean;
  running: boolean;
  onRun: () => void;
  onStop: () => void;
  // 続きから実行 (prompts/74、PIC の canContinue/onContinue と同じ役割)
  canContinue: boolean;
  onContinue: () => void;
  // true の場合、canContinue=false の理由が「ジオメトリ・ガス条件編集による食い違い」であることを示す
  // (PicPanel の continueDisabledByProjectChange 参照)
  continueDisabledByProjectChange: boolean;
  progress: GasProgress | null;
  // 直近実行の実効スレッド数 (started 由来、未実行/旧バックエンドは null)
  lastThreads: number | null;
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
  meshResult,
  lengthUnit,
  dsmc,
  onChange,
  canRun,
  running,
  onRun,
  onStop,
  canContinue,
  onContinue,
  continueDisabledByProjectChange,
  progress,
  lastThreads,
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
        (平面2D・軸対称 r-z の両対応)。結果は PIC の MCC で「DSMCガス場を使用」を有効にすると背景ガスとして使える。
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
          <p className="hint">
            キャンバスの「ガス境界」ツールで2点クリックでも追加できます (発見性のため案内)。
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
                        {/* 確定時コミット (blur/Enter)。素の input のキー毎確定だと
                            "0.5" 入力途中の "0." が数値化→再表示で "0" になり小数点が
                            消える (1未満の値が入力できない) ため */}
                        <CommitNullableNumberInput
                          className="rf-compact"
                          value={b.pressure_pa ?? null}
                          placeholder={b.type === "outlet" ? "真空" : ""}
                          onCommit={(v) => updateBoundary(i, { pressure_pa: v })}
                        />
                      </label>
                    )}
                    {b.type === "inlet" && specMode === "flow" && (
                      <label className="rf-compact-label" title="流量 [sccm] (1 sccm = 標準状態の1 cm³/min、奥行き1m換算)">
                        Q
                        {/* 確定時コミット。0.5 sccm など1未満の流量も指数表記も入力可能にする */}
                        <CommitNullableNumberInput
                          className="rf-compact"
                          value={b.flow_sccm ?? null}
                          onCommit={(v) => updateBoundary(i, { flow_sccm: v })}
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
            <span className="label">メッシュ粗化係数</span>
            <CommitNumberInput
              value={dsmc.mesh_scale ?? 1.0}
              onCommit={(v) => onChange({ ...dsmc, mesh_scale: Math.min(20, Math.max(1, v)) })}
            />
          </div>
          <p className="hint">
            DSMC 用メッシュの寸法 = FEM メッシュ寸法 × 係数。walk コストは係数分軽くなります。
            精度の目安はセル寸法 &lt; 平均自由行程/3。PIC連成時は要素重心で自動マッピングされます。
          </p>
          <div className="field">
            <span className="label">目標粒子数</span>
            <CommitNumberInput
              value={dsmc.n_particles}
              onCommit={(v) => onChange({ ...dsmc, n_particles: Math.max(1, Math.round(v)) })}
            />
          </div>
          {/* 粒子数の目安 (セルあたり20個以上ないと NTC 衝突統計が粗くなる)。
              セル数は DSMC 実行済みならその実測メッシュ (result.mesh、DSMC が実際に使った
              粗化後メッシュ) を最優先し、次に Mesh ボタンで生成済みの meshResult (FEM側、
              mesh_scale 未反映) を使う。どちらも無ければドメイン面積とメッシュサイズ
              (mesh_scale 反映後) からの概算 (正三角形 (√3/4)·size² で割る) を使う */}
          {(() => {
            const nElemsActual = result?.mesh.triangles.length ?? meshResult?.triangles.length ?? null;
            let nElems = nElemsActual;
            let approx = false;
            if (nElems == null) {
              const poly = project.geometry.domain.polygon;
              let area = 0;
              for (let k = 0; k < poly.length; k++) {
                const [x1, y1] = poly[k];
                const [x2, y2] = poly[(k + 1) % poly.length];
                area += x1 * y2 - x2 * y1;
              }
              area = Math.abs(area) / 2;
              const size = project.mesh.size * (dsmc.mesh_scale ?? 1.0);
              if (area > 0 && size > 0) {
                nElems = Math.max(1, Math.round(area / (0.433 * size * size)));
                approx = true;
              }
            }
            if (nElems == null) return null;
            const perCell = dsmc.n_particles / nElems;
            const low = perCell < 20;
            return (
              <p className="hint" style={low ? { color: "#e0b050" } : undefined}>
                目安: メッシュ {approx ? "約" : ""}{nElems.toLocaleString()} セル × 20〜50 個/セル ≈{" "}
                {(nElems * 20).toLocaleString()}〜{(nElems * 50).toLocaleString()} 粒子
                (現在の設定は約 {perCell.toFixed(1)} 個/セル{low ? " — 統計が粗くなる可能性があります" : ""})。
                実際のセル内粒子数は密度分布で偏るため、圧力勾配が強い場合は多めを推奨
              </p>
            );
          })()}
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
            walk・NTC衝突の並列スレッド数
            {lastThreads != null ? ` (前回実行: ${lastThreads}スレッド)` : ""}。
            前回実行の値が設定と一致しない場合、古いバックエンドに接続している可能性があります
          </p>
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
            <button
              className="secondary"
              onClick={onContinue}
              disabled={!canContinue}
              title={
                continueDisabledByProjectChange
                  ? "ジオメトリ・ガス条件が変更されたため続き実行できません (再度 ガス流れ計算 してください)"
                  : undefined
              }
            >
              続きから実行
            </button>
          </div>
          <p className="hint">
            「続きから実行」は現在のステップ数・平均ステップ数で追加実行します。
            粒子状態・乱数は前回から継続します。
            ジオメトリやガス条件を変更した場合は再実行が必要です。
            {continueDisabledByProjectChange &&
              " ジオメトリ・ガス条件を編集したため、続き実行するには再度「ガス流れ計算」が必要です。"}
          </p>

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
              <div className="kv">
                <span>計算時間</span>
                {/* 旧形式の結果付き保存ファイル (elapsed_s 追加前) を読み込んだ場合は
                    値が無いことがあるため、実行時同様に安全側の表示にする (prompts/86) */}
                <span>{result.elapsed_s != null ? `${result.elapsed_s.toFixed(3)} s` : "-"}</span>
              </div>
              {/* 位相別プロファイル計測 (prompts/87)。旧形式の結果付き保存ファイル
                  (timing 追加前) には無いことがあるため、無ければ表示を省略する */}
              {result.timing && <DsmcTimingSection timing={result.timing} />}

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

// 位相別プロファイル計測 (prompts/87) の日本語ラベル。timing のキーと1対1対応させる
// (PicPanel の TIMING_PHASE_LABELS/PicTimingSection と同じ流儀)
const DSMC_TIMING_LABELS: Record<string, string> = {
  inject: "流入 (リザーバ/流量注入)",
  move: "移動 + 境界処理",
  collide: "NTC衝突判定",
  sample: "サンプリング蓄積",
  other: "その他 (区間リセット・進捗コールバック等)",
};

// walk コスト診断 (prompts/88) が timing dict に同居させるキー。秒数ではないため
// 時間の内訳 (%計算・合計) からは除外し、別枠で表示する
// (backend/es_sim/dsmc.py の WALK_DIAG_KEYS と対応させること)
const DSMC_WALK_DIAG_KEYS = new Set(["walk_cells_est", "h_mean_m", "h_min_m"]);

// DSMC: 実行時間内訳。result.timing を値の大きい順に「名称 / 秒 / %」で表示する
// (セル並列化 (prompts/87) 前後の効果測定用)。未知のキーはキー名そのままで表示する。
// walk コスト診断 (prompts/88) のキーは時間ではないためここには含めない
function DsmcTimingSection({ timing }: { timing: Record<string, number> }) {
  const rows = Object.entries(timing)
    .filter(([key]) => !DSMC_WALK_DIAG_KEYS.has(key))
    .sort((a, b) => b[1] - a[1]);
  const total = rows.reduce((sum, [, v]) => sum + v, 0);
  return (
    <>
      {rows.map(([key, sec]) => (
        <div className="kv" key={key}>
          <span>{DSMC_TIMING_LABELS[key] ?? key}</span>
          <span>
            {sec.toFixed(3)} s ({total > 0 ? ((100 * sec) / total).toFixed(1) : "0.0"}%)
          </span>
        </div>
      ))}
      <WalkDiagSection timing={timing} />
    </>
  );
}

// walk コスト診断 (prompts/88): 平均横断セル数の推定・代表セル寸法。
// 時間の行 (DsmcTimingSection の秒・%表示) と混ざらないよう別枠で表示する。
// 旧形式の結果 (診断キー追加前) には無いので、無ければ何も表示しない
function WalkDiagSection({ timing }: { timing: Record<string, number> }) {
  const walkCellsEst = timing.walk_cells_est;
  const hMean = timing.h_mean_m;
  const hMin = timing.h_min_m;
  if (walkCellsEst == null || hMean == null || hMin == null) return null;
  const ratio = hMean > 0 ? hMin / hMean : 1;
  return (
    <>
      <div className="kv">
        <span>平均横断セル数/ステップ</span>
        <span>{walkCellsEst.toFixed(2)}</span>
      </div>
      <div className="kv">
        <span>代表セル寸法 (平均 / 最小) [m]</span>
        <span>
          {hMean.toExponential(3)} / {hMin.toExponential(3)}
        </span>
      </div>
      {ratio < 0.2 && (
        <p className="hint">
          最小/平均セル寸法比が {ratio.toFixed(2)} (&lt; 1/5) と小さく、メッシュ寸法の偏りが大きいです
        </p>
      )}
      {walkCellsEst > 3 && (
        <p className="hint">
          walk コストが支配的な場合、DSMC ではメッシュを粗くする (計算精度の目安はセル寸法 &lt;
          平均自由行程/3)、PIC では dt を見直すことで改善できる可能性があります
        </p>
      )}
    </>
  );
}
