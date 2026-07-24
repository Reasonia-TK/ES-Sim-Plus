import { CommitNullableNumberInput, CommitNumberInput } from "../CommitInput";
import { LENGTH_UNIT_LABEL, mToUnit, unitToM } from "../units";
import type { LengthUnit } from "../units";
import FnEmissionSection from "./FnPanel";
import { isAxisymmetric } from "../types";
import type { Emitter, FnEmission, ParticleSettings, Project, Species, TraceResult } from "../types";

/**
 * 粒子パネル
 * - 粒子種 (電子/陽子/カスタム)・エミッタ (line/point)・積分設定の編集
 * - project.particles として保存/読込対象だが、編集操作自体は Undo/Redo 履歴には積まない
 *   (App 側で geometry とは独立な state として管理している)
 * - FN (Fowler–Nordheim) 電界放出の設定 (有効時はエミッタ・粒子種は無視される)
 * - Trace 実行ボタンと結果サマリの表示
 */

interface Props {
  project: Project;
  // 長さの表示・入力単位 (mm/µm)。project 内部は常に m のまま
  lengthUnit: LengthUnit;
  particles: ParticleSettings;
  onChange: (next: ParticleSettings) => void;
  busy: boolean;
  canRun: boolean;
  onTrace: () => void;
  traceResult: TraceResult | null;
  showTrajectories: boolean;
  onToggleTrajectories: (v: boolean) => void;

  // 表示モード: "all"=従来通り全表示 (既定・後方互換)、"setup"=設定/実行UIのみ、
  // "results"=結果表示のみ (結果ノード用インスペクタページで使う)
  mode?: "all" | "setup" | "results";
  // 結果ページの背景表示 (電位V/|E|/なし)。未指定なら select 自体を出さない (後方互換)
  background?: "v" | "e_abs" | "none";
  onBackgroundChange?: (v: "v" | "e_abs" | "none") => void;
  // エミッタのキャンバス表示 ON/OFF。未指定ならチェックボックスを出さない (後方互換)
  showEmitter?: boolean;
  onToggleEmitter?: (v: boolean) => void;
}

const ELECTRON: Species = { preset: "electron" };
const PROTON: Species = { preset: "proton" };

export default function ParticlePanel({
  project,
  lengthUnit,
  particles,
  onChange,
  busy,
  canRun,
  onTrace,
  traceResult,
  showTrajectories,
  onToggleTrajectories,
  background,
  onBackgroundChange,
  showEmitter,
  onToggleEmitter,
  mode = "all",
}: Props) {
  // mode が "all" のときは従来通り両方表示。それ以外は該当モードのみ表示する
  const show = (m: "setup" | "results") => mode === "all" || mode === m;

  const { species, emitter, fn } = particles;
  const fnEnabled = !!fn;
  const unitLabel = LENGTH_UNIT_LABEL[lengthUnit];

  const setSpecies = (next: Species) => onChange({ ...particles, species: next });
  const setEmitter = (patch: Partial<Emitter>) =>
    onChange({ ...particles, emitter: { ...emitter, ...patch } });
  const setFn = (next: FnEmission | null) => onChange({ ...particles, fn: next });

  // Maxwell 分布かどうか (未指定は mono 扱い)
  const energyDist = emitter.energy_dist ?? "mono";
  const isMaxwell = energyDist === "maxwell";

  let summary: {
    n: number;
    absorbed: number;
    alive: number;
    avgTof: number | null;
    eMin: number | null;
    eMax: number | null;
    angleMean: number | null;
    angleStd: number | null;
    angleMin: number | null;
    angleMax: number | null;
  } | null = null;
  if (traceResult) {
    const n = traceResult.status.length;
    const absorbed = traceResult.status.filter((s) => s === "absorbed").length;
    const tofs = traceResult.tof.filter((t): t is number => t !== null);
    const avgTof = tofs.length ? tofs.reduce((a, b) => a + b, 0) / tofs.length : null;
    const energies = traceResult.final_energy_ev;
    const eMin = energies.length ? Math.min(...energies) : null;
    const eMax = energies.length ? Math.max(...energies) : null;
    // 吸収粒子の入射角統計 (final_angle_deg を status == "absorbed" の粒子で集計)
    const absorbedAngles = traceResult.status
      .map((s, i) => (s === "absorbed" ? traceResult.final_angle_deg[i] : null))
      .filter((a): a is number => a !== null);
    let angleMean: number | null = null;
    let angleStd: number | null = null;
    let angleMin: number | null = null;
    let angleMax: number | null = null;
    if (absorbedAngles.length) {
      angleMean = absorbedAngles.reduce((a, b) => a + b, 0) / absorbedAngles.length;
      const variance =
        absorbedAngles.reduce((a, b) => a + (b - angleMean!) ** 2, 0) / absorbedAngles.length;
      angleStd = Math.sqrt(variance);
      angleMin = Math.min(...absorbedAngles);
      angleMax = Math.max(...absorbedAngles);
    }
    summary = { n, absorbed, alive: n - absorbed, avgTof, eMin, eMax, angleMean, angleStd, angleMin, angleMax };
  }

  return (
    <>
      {show("setup") && (
      <>
      {fnEnabled ? (
        <p className="hint">
          FN電界放出が有効なため、粒子種・エミッタの設定は無効です (下の「FN電界放出」を参照)。
        </p>
      ) : (
        <>
          <h2>粒子種</h2>
          <div className="field">
            <span className="label">種類</span>
            <select
              value={species.preset}
              onChange={(e) => {
                const v = e.target.value;
                if (v === "electron") setSpecies(ELECTRON);
                else if (v === "proton") setSpecies(PROTON);
                else setSpecies({ preset: "custom", q: species.q ?? -1.6e-19, m: species.m ?? 9.1e-31 });
              }}
            >
              <option value="electron">電子</option>
              <option value="proton">陽子</option>
              <option value="custom">カスタム</option>
            </select>
          </div>
          {species.preset === "custom" && (
            <>
              <div className="field">
                <span className="label">q [C]</span>
                <CommitNumberInput
                  value={species.q ?? 0}
                  onCommit={(q) => setSpecies({ preset: "custom", q, m: species.m ?? 9.1e-31 })}
                />
              </div>
              <div className="field">
                <span className="label">m [kg]</span>
                <CommitNumberInput
                  value={species.m ?? 0}
                  onCommit={(m) => setSpecies({ preset: "custom", q: species.q ?? -1.6e-19, m })}
                />
              </div>
            </>
          )}

          <h2>エミッタ</h2>
          <div className="field">
            <span className="label">種別</span>
            <select value={emitter.kind} onChange={(e) => setEmitter({ kind: e.target.value as Emitter["kind"] })}>
              <option value="line">ライン</option>
              <option value="point">点</option>
            </select>
          </div>
          <div className="field">
            <span className="label">p1 x [{unitLabel}]</span>
            <CommitNumberInput
              value={mToUnit(emitter.p1[0], lengthUnit)}
              step="0.1"
              onCommit={(x) => setEmitter({ p1: [unitToM(x, lengthUnit), emitter.p1[1]] })}
            />
          </div>
          <div className="field">
            <span className="label">p1 y [{unitLabel}]</span>
            <CommitNumberInput
              value={mToUnit(emitter.p1[1], lengthUnit)}
              step="0.1"
              onCommit={(y) => setEmitter({ p1: [emitter.p1[0], unitToM(y, lengthUnit)] })}
            />
          </div>
          {emitter.kind === "line" && (
            <>
              <div className="field">
                <span className="label">p2 x [{unitLabel}]</span>
                <CommitNumberInput
                  value={mToUnit(emitter.p2[0], lengthUnit)}
                  step="0.1"
                  onCommit={(x) => setEmitter({ p2: [unitToM(x, lengthUnit), emitter.p2[1]] })}
                />
              </div>
              <div className="field">
                <span className="label">p2 y [{unitLabel}]</span>
                <CommitNumberInput
                  value={mToUnit(emitter.p2[1], lengthUnit)}
                  step="0.1"
                  onCommit={(y) => setEmitter({ p2: [emitter.p2[0], unitToM(y, lengthUnit)] })}
                />
              </div>
            </>
          )}
          <div className="field">
            <span className="label">粒子数 n</span>
            <CommitNumberInput value={emitter.n} onCommit={(n) => setEmitter({ n: Math.max(1, Math.round(n)) })} />
          </div>
          <div className="field">
            <span className="label">初期エネルギー [eV]</span>
            <CommitNumberInput value={emitter.energy_ev} onCommit={(v) => setEmitter({ energy_ev: v })} />
          </div>
          <div className="field">
            <span className="label">射出方向 [deg]</span>
            <CommitNumberInput value={emitter.direction_deg} onCommit={(v) => setEmitter({ direction_deg: v })} />
          </div>
          <div className="field">
            <span className="label">エネルギー分布</span>
            <select
              value={energyDist}
              onChange={(e) => {
                const v = e.target.value as "mono" | "maxwell";
                if (v === "maxwell") {
                  setEmitter({
                    energy_dist: "maxwell",
                    temperature_ev: emitter.temperature_ev ?? 1.0,
                    seed: emitter.seed ?? 0,
                  });
                } else {
                  setEmitter({ energy_dist: "mono" });
                }
              }}
            >
              <option value="mono">単一エネルギー</option>
              <option value="maxwell">Maxwell</option>
            </select>
          </div>
          <div className="field">
            <span className="label">広がり半角 [deg]</span>
            <CommitNumberInput
              value={emitter.spread_deg}
              onCommit={(v) => setEmitter({ spread_deg: v })}
              disabled={isMaxwell}
            />
          </div>
          {isMaxwell && (
            <>
              <p className="hint">Maxwell分布では熱運動が方向広がりを与えます (広がり半角は無視されます)</p>
              <div className="field">
                <span className="label">温度 kT [eV]</span>
                <CommitNumberInput
                  value={emitter.temperature_ev ?? 1.0}
                  onCommit={(v) => setEmitter({ temperature_ev: v })}
                />
              </div>
              <div className="field">
                <span className="label">乱数シード</span>
                <CommitNumberInput
                  value={emitter.seed ?? 0}
                  onCommit={(v) => setEmitter({ seed: Math.round(v) })}
                />
              </div>
            </>
          )}
        </>
      )}

      <FnEmissionSection project={project} fn={fn} onChange={setFn} mode="trace" />

      {/* エミッタのキャンバス表示切替 (FN 有効時もエミッタの緑マーカーが邪魔になるため常に出す) */}
      {showEmitter !== undefined && onToggleEmitter && (
            <label className="snap particle-trace-toggle">
              <input
                type="checkbox"
                checked={showEmitter}
                onChange={(e) => onToggleEmitter(e.target.checked)}
              />
              エミッタを表示
            </label>
          )}

      <h2>積分設定</h2>
      <div className="field">
        <span className="label">dt [s] (空欄=自動)</span>
        <CommitNullableNumberInput
          value={particles.dt}
          placeholder="自動"
          onCommit={(v) => onChange({ ...particles, dt: v })}
        />
      </div>
      <div className="field">
        <span className="label">ステップ数</span>
        <CommitNumberInput
          value={particles.n_steps}
          onCommit={(v) => onChange({ ...particles, n_steps: Math.max(1, Math.round(v)) })}
        />
      </div>
      <div className="field">
        <span className="label">保存間隔</span>
        <CommitNumberInput
          value={particles.save_every}
          onCommit={(v) => onChange({ ...particles, save_every: Math.max(1, Math.round(v)) })}
        />
      </div>

      <div className="actions">
        <button onClick={onTrace} disabled={busy || !canRun}>
          {busy ? "計算中..." : "Trace"}
        </button>
      </div>
      </>
      )}

      {show("results") && (
        <>
          {/* 軌道の背景に静電場ソルブ結果を重ねるかの選択 (このノード選択中のみ有効)。
              軌道だけを見たいときに「なし」で背景の色マップを消せるようにする */}
          {background !== undefined && onBackgroundChange && (
            <div className="field">
              <span className="label">背景表示</span>
              <select
                value={background}
                onChange={(e) => onBackgroundChange(e.target.value as "v" | "e_abs" | "none")}
              >
                <option value="v">電位 V</option>
                <option value="e_abs">|E|</option>
                <option value="none">なし</option>
              </select>
            </div>
          )}
          <label className="snap particle-trace-toggle">
            <input
              type="checkbox"
              checked={showTrajectories}
              onChange={(e) => onToggleTrajectories(e.target.checked)}
            />
            軌道を表示
          </label>
          {showEmitter !== undefined && onToggleEmitter && (
            <label className="snap particle-trace-toggle">
              <input
                type="checkbox"
                checked={showEmitter}
                onChange={(e) => onToggleEmitter(e.target.checked)}
              />
              エミッタを表示
            </label>
          )}

          {summary && (
            <>
              <h2>トレース結果</h2>
              <div className="kv"><span>粒子数</span><span>{summary.n}</span></div>
              <div className="kv"><span>吸収 / 生存</span><span>{summary.absorbed} / {summary.alive}</span></div>
              <div className="kv">
                <span>平均飛行時間</span>
                <span>{summary.avgTof !== null ? summary.avgTof.toExponential(3) : "-"} s</span>
              </div>
              <div className="kv">
                <span>最終エネルギー min/max</span>
                <span>
                  {summary.eMin !== null ? summary.eMin.toExponential(3) : "-"} /{" "}
                  {summary.eMax !== null ? summary.eMax.toExponential(3) : "-"} eV
                </span>
              </div>
              <div className="kv">
                <span>入射角 平均±標準偏差</span>
                <span>
                  {summary.angleMean !== null ? summary.angleMean.toFixed(2) : "-"} ±{" "}
                  {summary.angleStd !== null ? summary.angleStd.toFixed(2) : "-"} deg
                </span>
              </div>
              <div className="kv">
                <span>入射角 min/max</span>
                <span>
                  {summary.angleMin !== null ? summary.angleMin.toFixed(2) : "-"} /{" "}
                  {summary.angleMax !== null ? summary.angleMax.toFixed(2) : "-"} deg
                </span>
              </div>
              {traceResult && traceResult.fn_current != null && (
                <div className="kv">
                  <span>FN総放出電流</span>
                  <span>
                    {traceResult.fn_current.toExponential(3)} {isAxisymmetric(project.coord) ? "A" : "A/m"}
                  </span>
                </div>
              )}
            </>
          )}

          {/* results専用ページで未実行の場合のヒント (mode="all" の従来ページでは出さない) */}
          {mode === "results" && !traceResult && (
            <p className="hint">トレースが未実行です。スタディ「粒子追跡」から実行してください。</p>
          )}
        </>
      )}
    </>
  );
}
