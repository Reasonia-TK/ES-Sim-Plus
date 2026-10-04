// PIC (2D)・流体 2D の結果のグラフ (v1 PicPanel・Fluid2dPanel の結果の部分): RF 波形モニタ (フレームの位相)・
// 履歴 (PIC はエネルギーと粒子数を系列ごとに正規化、流体 2D は全域の密度)・IEDF/IADF/IAEDF (コレクタごと)・
// EEDF/EEPF (領域ごと)・シース端 (評価線ごとの s と s(φ)、α と等値線はビューアと共通)。

import type { TFunction } from "i18next";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Toggle } from "../../forms/SchemaField";
import { viewMeshOf } from "../../graphics/runScene";
import type { ViewMesh } from "../../graphics/scene";
import { useJobs } from "../../jobs/jobsStore";
import type { JobSummary } from "../../jobs/types";
import { useDocument } from "../../model/documentStore";
import type { Point, Project } from "../../model/project";
import { ChartCard } from "../../plots/ChartCard";
import { columns, saveCsv } from "../../plots/csv";
import { Heatmap } from "../../plots/Heatmap";
import { LineChart, type LineSeries } from "../../plots/LineChart";
import { edgeLabel } from "../../tree/treeModel";
import { formatNumber, maxOf } from "../../util/format";
import { pickTimeUnit, rfComponents, waveformFreqs, type VoltageWaveform } from "../../util/waveform";
import { useChartPref, useResultsView, type Field2dKind } from "../resultsView";
import { useFrame } from "../runData";
import { sheathLinesOf, sheathOnLine } from "../sheath";
import type { PicCollectorResult, PicDiag, PicFrame, PicResult } from "../types";
import { EEDF_COLORS, ELECTRON_COLOR, ION_COLOR, useLength } from "./common";
import { EedfCard } from "./Distributions";
import { RfMonitor, type RfElectrode } from "./RfMonitor";
import { attachCircuit, SelfBiasCard, type CircuitSrc } from "./SelfBias";

// ---- RF 波形 (2D の電極) ----

/**
 * 時間で変わる電極 (Dirichlet の境界は辺ごと、導体の領域は領域ごと。v1 RfPhaseMonitor と同じ)。circuit があれば
 * 阻止コンデンサの電極は実際の電極の電位にする (prompts/134)
 */
export function electrodes2d(project: Project, t: TFunction, circuit?: CircuitSrc): RfElectrode[] {
  const out: RfElectrode[] = [];
  const wfOf = (v: unknown) => (v && typeof v === "object" ? [v as VoltageWaveform] : []);
  for (const bc of project.geometry.boundaries) {
    if (bc.type !== "dirichlet") continue;
    const rf = rfComponents(bc.voltage_rf);
    const wf = wfOf(bc.voltage_waveform);
    if (!waveformFreqs(rf, wf).length) continue;
    for (const e of bc.edges) out.push(attachCircuit({ label: edgeLabel(project, e, t), dc: bc.voltage ?? 0, rf, waveforms: wf }, `edge${e}`, circuit, t));
  }
  for (const r of project.geometry.regions) {
    if (r.type !== "conductor") continue;
    const rf = rfComponents(r.voltage_rf);
    const wf = wfOf(r.voltage_waveform);
    if (!waveformFreqs(rf, wf).length) continue;
    out.push(attachCircuit({ label: r.id, dc: r.voltage ?? 0, rf, waveforms: wf }, r.id, circuit, t));
  }
  return out;
}

/** 2D の電極の RF 波形 (時間で変わる電極が無ければ何も出さない) */
export function RfCardBody({ project, time, title, height, circuit }: { project: Project | null; time: number | null; title: string; height: number; circuit?: CircuitSrc }) {
  const { t } = useTranslation();
  const electrodes = useMemo(() => (project ? electrodes2d(project, t, circuit) : []), [project, t, circuit]);
  if (!electrodes.length) return null;
  return <RfMonitor title={title} electrodes={electrodes} t={time} height={height} />;
}

export function RfCard({ project, time, circuit }: { project: Project | null; time: number | null; circuit?: CircuitSrc }) {
  const { t } = useTranslation();
  const has = useMemo(() => (project ? electrodes2d(project, t).length > 0 : false), [project, t]);
  if (!has) return null;
  return (
    <ChartCard title={t("charts.rfTitle")}>
      <RfCardBody project={project} time={time} title="" height={120} circuit={circuit} />
    </ChartCard>
  );
}

// ---- 履歴 ----

function normalize(v: number[]): number[] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const x of v) {
    if (x < lo) lo = x;
    if (x > hi) hi = x;
  }
  const span = hi - lo;
  return v.map((x) => (span > 0 ? (x - lo) / span : 0));
}

/** PIC の履歴: 運動エネルギー・場のエネルギー・全エネルギー・粒子数を系列ごとに最小〜最大で正規化 (v1 と同じ) */
function PicHistoryCard({ rows }: { rows: Partial<PicDiag>[] }) {
  const { t } = useTranslation();
  const chart = useMemo(() => {
    if (rows.length < 2) return null;
    const tt = rows.map((r) => r.t ?? 0);
    const unit = pickTimeUnit(maxOf(tt));
    const ke = rows.map((r) => (r.ke_e ?? 0) + (r.ke_i ?? 0));
    const fe = rows.map((r) => r.fe ?? 0);
    const tot = ke.map((k, i) => k + fe[i]);
    const np = rows.map((r) => (r.n_e ?? 0) + (r.n_i ?? 0));
    const series: LineSeries[] = [
      { label: t("charts.histKe"), values: normalize(ke), color: "#4da3ff" },
      { label: t("charts.histFe"), values: normalize(fe), color: "#ffb84d" },
      { label: t("charts.histTotal"), values: normalize(tot), color: "#6fd08c" },
      { label: t("charts.histParticles"), values: normalize(np), color: "#d8dce4" },
    ];
    return { x: tt.map((v) => v * unit.scale), unit: unit.label, series };
  }, [rows, t]);
  if (!chart) return null;
  return (
    <ChartCard title={t("charts.histPic")}>
      <LineChart x={chart.x} series={chart.series} xLabel={`t [${chart.unit}]`} yLabel={t("charts.normalized")} yRange={[0, 1]} height={150} />
    </ChartCard>
  );
}

/** 流体 2D の履歴: 全域の密度 (電子・イオン) */
function Fluid2dHistoryCard({ t: time, ne, ni }: { t: number[]; ne: number[]; ni: number[] }) {
  const { t } = useTranslation();
  const chart = useMemo(() => {
    if (time.length < 2) return null;
    const unit = pickTimeUnit(maxOf(time));
    return {
      x: time.map((v) => v * unit.scale),
      unit: unit.label,
      series: [
        { label: "n_e_total", values: ne, color: ELECTRON_COLOR },
        { label: "n_i_total", values: ni, color: ION_COLOR },
      ] as LineSeries[],
    };
  }, [time, ne, ni]);
  if (!chart) return null;
  return (
    <ChartCard title={t("charts.histDensity2d")}>
      <LineChart x={chart.x} series={chart.series} xLabel={`t [${chart.unit}]`} height={140} />
    </ChartCard>
  );
}

// ---- IEDF / IADF / IAEDF ----

/** 重み付きのヒストグラム (範囲外は端のビンに入れる、v1 と同じ。正規化しない) */
export function histogram(values: ArrayLike<number>, weights: ArrayLike<number> | undefined, n: number, [lo, hi]: [number, number]): { centers: number[]; counts: number[] } {
  const counts = new Array<number>(n).fill(0);
  const span = hi - lo || 1;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    const k = Math.min(n - 1, Math.max(0, Math.floor(((v - lo) / span) * n)));
    counts[k] += weights?.[i] ?? 1;
  }
  const centers = Array.from({ length: n }, (_, k) => lo + ((k + 0.5) / n) * (hi - lo));
  return { centers, counts };
}

/** 角度 × エネルギーの 2D ヒストグラム (values[iy = エネルギー][ix = 角度]) */
export function histogram2d(ang: ArrayLike<number>, en: ArrayLike<number>, weights: ArrayLike<number> | undefined, n: number, ar: [number, number], er: [number, number]): number[][] {
  const out = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  const as = ar[1] - ar[0] || 1;
  const es = er[1] - er[0] || 1;
  for (let i = 0; i < ang.length; i++) {
    const a = ang[i];
    const e = en[i];
    if (!Number.isFinite(a) || !Number.isFinite(e)) continue;
    const ix = Math.min(n - 1, Math.max(0, Math.floor(((a - ar[0]) / as) * n)));
    const iy = Math.min(n - 1, Math.max(0, Math.floor(((e - er[0]) / es) * n)));
    out[iy][ix] += weights?.[i] ?? 1;
  }
  return out;
}

/** エネルギーの範囲 (空なら [0, 1]、幅 0 なら +1) */
export function energyRange(v: ArrayLike<number>): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < v.length; i++) {
    if (!Number.isFinite(v[i])) continue;
    if (v[i] < lo) lo = v[i];
    if (v[i] > hi) hi = v[i];
  }
  if (!(lo <= hi)) return [0, 1];
  return hi > lo ? [lo, hi] : [lo, lo + 1];
}

const IADF_RANGE: [number, number] = [-90, 90];

function CollectorCard({ collectors, labels }: { collectors: PicCollectorResult[]; labels: string[] }) {
  const { t } = useTranslation();
  const [idx, setIdx] = useChartPref("pic.collector", 0);
  const [bins, setBins] = useChartPref("pic.histBins", 60);
  const [logIaedf, setLogIaedf] = useChartPref("pic.iaedfLog", false);
  const [binsText, setBinsText] = useState<string | null>(null);
  const i = Math.min(Math.max(0, idx), collectors.length - 1);
  const c = collectors[i];
  const name = labels[i] || `C${i + 1}`;
  const er = useMemo(() => energyRange(c.energies_ev), [c]);
  const iedf = useMemo(() => histogram(c.energies_ev, c.weights, bins, er), [c, bins, er]);
  const iadf = useMemo(() => histogram(c.angles_deg, c.weights, bins, IADF_RANGE), [c, bins]);
  const iaedf = useMemo(() => histogram2d(c.angles_deg, c.energies_ev, c.weights, bins, IADF_RANGE, er), [c, bins, er]);
  const commitBins = () => {
    if (binsText === null) return;
    const v = Math.round(Number(binsText));
    if (Number.isFinite(v)) setBins(Math.min(400, Math.max(1, v)));
    setBinsText(null);
  };
  return (
    <ChartCard
      title={t("charts.collectors")}
      tools={
        <>
          <select className="input" value={i} aria-label={t("charts.collector")} onChange={(e) => setIdx(Number(e.target.value))}>
            {collectors.map((_, k) => (
              <option key={k} value={k}>
                {labels[k] || `C${k + 1}`}
              </option>
            ))}
          </select>
          <label className="inline-field">
            {t("charts.bins")}
            <input
              className="input bins-input"
              value={binsText ?? String(bins)}
              inputMode="numeric"
              onChange={(e) => setBinsText(e.target.value)}
              onBlur={commitBins}
              onKeyDown={(e) => e.key === "Enter" && commitBins()}
            />
          </label>
          <button
            type="button"
            className="button small"
            disabled={c.count === 0}
            onClick={() => void saveCsv(`iedf_iadf_${name}.csv`, ["energy_ev", "angle_deg", "weight"], columns(c.energies_ev, c.angles_deg, c.weights))}
          >
            {t("charts.csv")}
          </button>
        </>
      }
    >
      <div className="kv">
        <span>{t("charts.samples")}</span>
        <span className="mono">{c.count}</span>
        <span>{t("charts.totalIons")}</span>
        <span className="mono">{c.total_weight.toExponential(3)}</span>
      </div>
      {c.truncated && <p className="hint hint-warn">{t("charts.truncated")}</p>}
      {c.count === 0 ? (
        <p className="hint">{t("charts.noIons")}</p>
      ) : (
        <>
          <div className="chart-subtitle">{t("charts.iedf")}</div>
          <LineChart x={iedf.centers} series={[{ label: "IEDF", values: iedf.counts, color: "#4da3ff", paths: "bars", fill: "rgba(77, 163, 255, 0.55)" }]} xLabel="E [eV]" yLabel={t("charts.weightedCount")} height={140} />
          <div className="chart-subtitle">{t("charts.iadf")}</div>
          <LineChart x={iadf.centers} series={[{ label: "IADF", values: iadf.counts, color: "#ffb84d", paths: "bars", fill: "rgba(255, 184, 77, 0.55)" }]} xLabel={t("charts.angleDeg")} yLabel={t("charts.weightedCount")} xRange={IADF_RANGE} height={140} />
          <div className="chart-head-row">
            <div className="chart-subtitle">{t("charts.iaedf")}</div>
            <Toggle checked={logIaedf} onChange={setLogIaedf} label={t("charts.log")} />
          </div>
          <Heatmap values={iaedf} xRange={IADF_RANGE} yRange={er} xLabel={t("charts.angleDeg")} yLabel="E [eV]" valueLabel={t("charts.weightedCount")} log={logIaedf} skipZero height={240} />
        </>
      )}
    </ChartCard>
  );
}

// ---- シース端 ----

interface SheathSource {
  mesh: ViewMesh;
  fields: { n_e: number[]; n_i: number[] } | null;
  cycle: { bins: number; n_e: number[][]; n_i: number[][] } | null;
}

function SheathCard({ kind, src }: { kind: Field2dKind; src: SheathSource }) {
  const { t } = useTranslation();
  const len = useLength();
  const doc = useDocument((s) => s.project);
  const lines = sheathLinesOf(doc);
  const alpha = useResultsView((s) => s.sheathAlpha);
  const contour = useResultsView((s) => s.sheathContour);
  const setSheath = useResultsView((s) => s.setSheath);
  const mode = useResultsView((s) => s.display[kind]?.mode);
  const bin = useResultsView((s) => s.bin);
  const cycle = src.cycle;
  const b = cycle ? Math.min(Math.max(0, bin), cycle.bins - 1) : 0;
  const useCycle = mode === "cycle" && cycle;
  const nE = useCycle ? cycle.n_e[b] : src.fields?.n_e;
  const nI = useCycle ? cycle.n_i[b] : src.fields?.n_i;
  const linesKey = JSON.stringify(lines.map((l) => [l.p1, l.p2]));
  const sNow = lines.map((l) => (nE && nI ? sheathOnLine(src.mesh, l.p1 as Point, l.p2 as Point, nE, nI) : null));
  const sPhi = useMemo(() => {
    if (!cycle || !lines.length || cycle.bins < 2) return null;
    const x = Array.from({ length: cycle.bins }, (_, k) => (k / cycle.bins) * 360);
    const series: LineSeries[] = lines.map((l, i) => ({
      label: l.label || `S${i + 1}`,
      color: EEDF_COLORS[i % EEDF_COLORS.length],
      values: cycle.n_e.map((ne, k) => {
        const s = sheathOnLine(src.mesh, l.p1 as Point, l.p2 as Point, ne, cycle.n_i[k]);
        return s === null ? null : len.of(s);
      }),
    }));
    return { x, series };
  }, [cycle, src.mesh, linesKey, len.unit]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <ChartCard title={t("charts.sheath2d")} tools={<Toggle checked={contour} onChange={(v) => setSheath({ contour: v })} label={t("charts.sheathContour")} />} note={t("charts.sheathLinesHint")}>
      <label className="inline-field alpha-row">
        {t("charts.alpha", { a: alpha.toFixed(2) })}
        <input type="range" min={0.05} max={0.95} step={0.05} value={alpha} onChange={(e) => setSheath({ alpha: Number(e.target.value) })} />
      </label>
      {lines.length === 0 ? (
        <p className="hint">{t("charts.noSheathLines")}</p>
      ) : (
        <div className="kv">
          {lines.map((l, i) => (
            <SheathRow key={i} label={l.label || `S${i + 1}`} color={EEDF_COLORS[i % EEDF_COLORS.length]} s={sNow[i]} unit={len.unit} of={len.of} />
          ))}
        </div>
      )}
      {sPhi && (
        <>
          <div className="chart-subtitle">{t("charts.sPhi2d", { unit: len.unit })}</div>
          {sPhi.series.every((s) => s.values.every((v) => v === null)) ? (
            <p className="hint">{t("charts.noSheath")}</p>
          ) : (
            <LineChart x={sPhi.x} series={sPhi.series} xLabel={t("charts.phaseDeg")} yLabel={`s [${len.unit}]`} xRange={[0, 360]} height={140} markers={[{ x: (b / cycle!.bins) * 360, color: "#ff5c5c" }]} />
          )}
        </>
      )}
    </ChartCard>
  );
}

function SheathRow({ label, color, s, unit, of }: { label: string; color: string; s: number | null; unit: string; of: (m: number) => number }) {
  return (
    <>
      <span style={{ color }}>{label}</span>
      <span className="mono">s = {s === null ? "—" : `${formatNumber(of(s))} ${unit}`}</span>
    </>
  );
}

// ---- まとめ ----

export function PicCharts({ job, project, result }: { job: JobSummary; project: Project | null; result: PicResult | null }) {
  const live = useFrame<PicFrame>(job);
  const liveRows = useJobs((s) => s.liveHistory[job.id]);
  const running = job.state === "running" || job.state === "queued";
  const frame = running ? live : (result?.frame ?? live);
  const rows = running || !result ? (liveRows ?? []) : result.history;
  const mesh = result ? viewMeshOf(result.started.mesh) : null;
  const pic = project?.pic as { collectors?: { label?: string }[] } | null | undefined;
  const labels = (pic?.collectors ?? []).map((c) => c?.label ?? "");
  // 阻止コンデンサ (prompts/134): 実行中はフレームの直近の自己バイアス、完了後は結果の最後の 1 周期の電極の電位
  const circuit = useMemo<CircuitSrc>(() => ({ result: running ? null : result?.circuit, frame: frame?.circuit }), [running, result, frame]);
  return (
    <>
      <RfCard project={project} time={frame?.t ?? null} circuit={circuit} />
      {result?.circuit && <SelfBiasCard circuit={result.circuit} project={project} csvPrefix="pic" />}
      <PicHistoryCard rows={rows as Partial<PicDiag>[]} />
      {result && result.collectors.length > 0 && <CollectorCard collectors={result.collectors} labels={labels} />}
      {result && result.eedf.length > 0 && <EedfCard eedf={result.eedf} csvPrefix="" prefKey="pic.eedf" defaultMode="eepf" />}
      {mesh && (result?.fields || result?.cycle) && <SheathCard kind="pic" src={{ mesh, fields: result!.fields, cycle: result!.cycle }} />}
    </>
  );
}

export function Fluid2dCharts({ job, project, result }: { job: JobSummary; project: Project | null; result: import("../types").Fluid2dResult | null }) {
  const liveRows = useJobs((s) => s.liveHistory[job.id]);
  const started = useJobs((s) => s.startedFull[job.id]);
  const frame = useFrame<{ t: number; circuit?: import("../types").CircuitFrame[] | null }>(job);
  const running = job.state === "running" || job.state === "queued";
  const circuit = useMemo<CircuitSrc>(() => ({ result: running ? null : result?.circuit, frame: frame?.circuit }), [running, result, frame]);
  const hist = useMemo(() => {
    if (!running && result?.history?.t) return { t: result.history.t, ne: result.history.n_e_total ?? [], ni: result.history.n_i_total ?? [] };
    const rows = liveRows ?? [];
    return { t: rows.map((r) => r.t ?? 0), ne: rows.map((r) => r.n_e_total ?? 0), ni: rows.map((r) => r.n_i_total ?? 0) };
  }, [running, result, liveRows]);
  const meshSrc = (result?.mesh ?? started?.mesh) as Parameters<typeof viewMeshOf>[0] | undefined;
  const mesh = meshSrc ? viewMeshOf(meshSrc) : null;
  return (
    <>
      <RfCard project={project} time={running ? (frame?.t ?? null) : null} circuit={circuit} />
      {result?.circuit && <SelfBiasCard circuit={result.circuit} project={project} csvPrefix="fluid2d" />}
      <Fluid2dHistoryCard t={hist.t} ne={hist.ne} ni={hist.ni} />
      {mesh && result && (result.fields || result.cycle) && <SheathCard kind="fluid2d" src={{ mesh, fields: result.fields, cycle: result.cycle }} />}
    </>
  );
}
