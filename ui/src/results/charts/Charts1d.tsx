// PIC 1D・流体 1D の結果のグラフ (v1 Plot1dView・Fluid1dPlotView の移植)。
// 実行中はライブ (φ・密度・T_e または位相空間・RF 波形)、終わったら時間平均のプロファイル (ほかの 1D の実行と
// 重ねて比較できる)・位相分解の再生・シース振動 (PIC 1D)・EEDF/EEPF (PIC 1D)・壁 IEDF・履歴。

import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { Toggle } from "../../forms/SchemaField";
import { PlaybackBar } from "../../graphics/PlaybackBar";
import { jobName, useJobs } from "../../jobs/jobsStore";
import type { JobSummary } from "../../jobs/types";
import type { Project } from "../../model/project";
import { ChartCard } from "../../plots/ChartCard";
import { LineChart, type LineSeries } from "../../plots/LineChart";
import { Scatter } from "../../plots/Scatter";
import { formatNumber, formatSi } from "../../util/format";
import { useChartPref, useResultsView } from "../resultsView";
import { useFrame, useRunResult } from "../runData";
import type { Cycle1d, Fluid1dResult, Frame1d, Pic1dResult, Profiles1d, Sheath1d } from "../types";
import {
  ampAt,
  deriveXGrid,
  ELECTRON_COLOR,
  FIELD1D_COLORS,
  HARMONIC_COLOR,
  interpLinear,
  ION_COLOR,
  mergeSeriesX,
  rowsRange,
  SHEATH_COLOR,
  SHEATH_RIGHT_COLOR,
  spectrumPeaks,
  useLength,
} from "./common";
import { EedfCard, WallIedfCard } from "./Distributions";
import { electrode1d, RfMonitor } from "./RfMonitor";

export type Kind1d = "pic1d" | "fluid1d";
type Result1d = Pic1dResult | Fluid1dResult;

const PHASE_SPACE_COLOR = "rgba(89, 194, 255, 0.35)";
const PROFILE_FIELDS = ["phi", "e", "n_e", "n_i", "overlay", "t_e", "ionization"] as const;
type ProfileField = (typeof PROFILE_FIELDS)[number];
const DENSITY: ProfileField[] = ["n_e", "n_i", "overlay"];

/** どの系列にも値が無い (シース端が見つからないときなど) */
function noValues(series: LineSeries[]): boolean {
  return series.every((s) => s.values.every((v) => v === null || !Number.isFinite(v)));
}

function block(project: Project | null, kind: Kind1d): Record<string, unknown> | null {
  const b = project?.[kind];
  return b && typeof b === "object" ? (b as Record<string, unknown>) : null;
}

function gapOf(result: Result1d): number {
  const g = Number(result.settings?.gap_m);
  if (Number.isFinite(g) && g > 0) return g;
  const x = result.profiles?.x;
  return x && x.length ? x[x.length - 1] : 0;
}

// ---- ライブ ----

export function Live1d({ job, kind, project }: { job: JobSummary; kind: Kind1d; project: Project | null }) {
  const { t } = useTranslation();
  const len = useLength();
  const frame = useFrame<Frame1d>(job);
  const started = useJobs((s) => s.startedFull[job.id] ?? s.started[job.id]);
  const [logN, setLogN] = useChartPref(`${kind}.liveLog`, true);
  const b = block(project, kind);
  const xm = useMemo(() => {
    const sx = started?.x;
    if (Array.isArray(sx)) return sx as number[];
    if (b && typeof b.gap_m === "number" && typeof b.n_cells === "number") return deriveXGrid(b.gap_m, b.n_cells);
    return null;
  }, [started, b]);
  const x = useMemo(() => (xm ? xm.map(len.of) : []), [xm, len.unit]); // eslint-disable-line react-hooks/exhaustive-deps
  const phi = useMemo<LineSeries[]>(() => (frame ? [{ label: "φ", values: frame.phi, color: FIELD1D_COLORS.phi[0] }] : []), [frame]);
  const dens = useMemo<LineSeries[]>(
    () =>
      frame
        ? [
            { label: "n_e", values: frame.n_e, color: FIELD1D_COLORS.n_e[0] },
            { label: "n_i", values: frame.n_i, color: FIELD1D_COLORS.n_i[0] },
          ]
        : [],
    [frame],
  );
  const te = useMemo<LineSeries[]>(() => (frame?.t_e ? [{ label: "T_e", values: frame.t_e, color: FIELD1D_COLORS.t_e[0] }] : []), [frame]);
  const phase = useMemo(() => (frame?.sample ? [{ label: "e", x: frame.sample.x.map(len.of), y: frame.sample.vx, color: PHASE_SPACE_COLOR, size: 2 }] : null), [frame, len.unit]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!frame || x.length < 2) return <p className="hint">{t("charts.waitingFrame")}</p>;
  const xl = `x [${len.unit}]`;
  return (
    <>
      <ChartCard title="φ(x) [V]">
        <LineChart x={x} series={phi} xLabel={xl} yLabel="φ [V]" height={130} />
      </ChartCard>
      <ChartCard title="n_e / n_i (x) [m^-3]" tools={<Toggle checked={logN} onChange={setLogN} label={t("charts.log")} />}>
        <LineChart x={x} series={dens} xLabel={xl} yLabel="n [m^-3]" logY={logN} height={130} />
      </ChartCard>
      {te.length > 0 && (
        <ChartCard title="T_e(x) [eV]">
          <LineChart x={x} series={te} xLabel={xl} yLabel="T_e [eV]" height={130} />
        </ChartCard>
      )}
      {phase && (
        <ChartCard title={t("charts.phaseSpace")}>
          <Scatter sets={phase} xLabel={xl} yLabel="v_x [m/s]" xRange={[x[0], x[x.length - 1]]} height={170} />
        </ChartCard>
      )}
      <div className="kv">
        <span>{t("charts.stepTime")}</span>
        <span className="mono">
          {frame.step} / {formatSi(frame.t, "s")}
        </span>
        <span>{t("charts.elapsedFrame")}</span>
        <span className="mono">{frame.elapsed_s.toFixed(3)} s</span>
      </div>
      {b && <RfMonitor title={t("charts.rfLeft")} electrodes={[electrode1d(b.left, t("charts.leftElectrode"))].filter((e) => e !== null)} t={frame.t} />}
    </>
  );
}

// ---- 時間平均のプロファイル (ほかの 1D の実行と比較) ----

function profileSeries(field: ProfileField, p: Profiles1d): { key: keyof Profiles1d; sym: string }[] {
  const one = (key: keyof Profiles1d, sym: string) => [{ key, sym }];
  switch (field) {
    case "phi":
      return one("phi", "φ");
    case "e":
      return one("e", "E");
    case "n_e":
      return one("n_e", "n_e");
    case "n_i":
      return one("n_i", "n_i");
    case "overlay":
      return [
        { key: "n_e", sym: "n_e" },
        { key: "n_i", sym: "n_i" },
      ];
    case "t_e":
      return one("t_e", "T_e");
    case "ionization":
      return p.ionization ? one("ionization", "ν_iz") : [];
  }
}

function sheathMarkers(sheath: Sheath1d | null | undefined, gap: number, of: (m: number) => number): { x: number; color: string }[] {
  const out: { x: number; color: string }[] = [];
  if (sheath?.left_s != null) out.push({ x: of(sheath.left_s), color: SHEATH_COLOR });
  if (sheath?.right_s != null) out.push({ x: of(gap - sheath.right_s), color: SHEATH_COLOR });
  return out;
}

function CompareSummary({ a, b, nameB }: { a: Result1d; b: Result1d; nameB: string }) {
  const { t } = useTranslation();
  const len = useLength();
  const pa = a.profiles!;
  const pb = b.profiles!;
  const ga = gapOf(a);
  const gb = gapOf(b);
  const neA = interpLinear(pa.x, pa.n_e, ga / 2);
  const neB = interpLinear(pb.x, pb.n_e, gb / 2);
  const teA = interpLinear(pa.x, pa.t_e, ga / 2);
  const teB = interpLinear(pb.x, pb.t_e, gb / 2);
  const d = (x: number | null | undefined, y: number | null | undefined) => (x != null && y != null ? `${formatNumber(len.of(x - y))} ${len.unit}` : "—");
  return (
    <div className="kv compare-kv">
      <span>{t("charts.cmpNe", { name: nameB })}</span>
      <span className="mono">{neB > 0 ? (neA / neB).toFixed(3) : "—"}</span>
      <span>{t("charts.cmpTe", { name: nameB })}</span>
      <span className="mono">{Number.isFinite(teA - teB) ? `${(teA - teB).toFixed(3)} eV` : "—"}</span>
      <span>{t("charts.cmpSheath", { name: nameB })}</span>
      <span className="mono">
        {d(a.sheath?.left_s, b.sheath?.left_s)} / {d(a.sheath?.right_s, b.sheath?.right_s)}
      </span>
    </div>
  );
}

function Profile1dCard({ job, kind, result }: { job: JobSummary; kind: Kind1d; result: Result1d }) {
  const { t } = useTranslation();
  const len = useLength();
  const [fieldPref, setField] = useChartPref<string>(`${kind}.profile`, "phi");
  const [log, setLog] = useChartPref(`${kind}.profileLog`, false);
  const [showSheath, setShowSheath] = useChartPref(`${kind}.sheath`, true);
  const [cmpId, setCmpId] = useChartPref<string>(`${kind}.compare`, "");
  const field = (PROFILE_FIELDS as readonly string[]).includes(fieldPref) ? (fieldPref as ProfileField) : "phi";
  const jobs = useJobs((s) => s.jobs);
  const candidates = useMemo(
    () =>
      Object.values(jobs)
        .filter((j) => (j.kind === "pic1d" || j.kind === "fluid1d") && j.id !== job.id && j.has_result)
        .sort((a, b) => a.created - b.created),
    [jobs, job.id],
  );
  const cmpJob = candidates.find((j) => j.id === cmpId);
  const { result: cmp } = useRunResult<Result1d>(cmpJob);
  const p = result.profiles;
  const cp = cmpJob && cmp?.profiles ? cmp.profiles : null;
  const logOn = log && DENSITY.includes(field);
  const nameA = jobName(job);
  const nameB = cmpJob ? jobName(cmpJob) : "";
  const chart = useMemo(() => {
    if (!p) return null;
    const own = profileSeries(field, p);
    const list: { x: number[]; y: (number | null)[]; label: string; color: string; dash?: number[] }[] = own.map((s) => ({
      x: p.x,
      y: p[s.key] as number[],
      label: cp ? `${s.sym} (${nameA})` : s.sym,
      color: FIELD1D_COLORS[s.key as string]?.[0] ?? "#59c2ff",
    }));
    if (cp)
      for (const s of profileSeries(field, cp))
        list.push({ x: cp.x, y: cp[s.key] as number[], label: `${s.sym} (${nameB})`, color: FIELD1D_COLORS[s.key as string]?.[1] ?? "#2f7dbf", dash: [5, 3] });
    const merged = cp ? mergeSeriesX(list) : { x: p.x, ys: list.map((l) => l.y) };
    return {
      x: merged.x.map(len.of),
      series: list.map((l, i) => ({ label: l.label, color: l.color, dash: l.dash, values: merged.ys[i] })) as LineSeries[],
    };
  }, [p, cp, field, nameA, nameB, len.unit]); // eslint-disable-line react-hooks/exhaustive-deps
  const markers = showSheath && p ? sheathMarkers(result.sheath, gapOf(result), len.of) : undefined;
  const fieldLabel = t(`charts.field1d.${field}`);
  return (
    <ChartCard
      title={t("charts.profile1d")}
      tools={
        <>
          <select className="input" value={field} aria-label={t("charts.resultField")} onChange={(e) => setField(e.target.value)}>
            {PROFILE_FIELDS.map((f) => (
              <option key={f} value={f}>
                {t(`charts.field1d.${f}`)}
              </option>
            ))}
          </select>
          {DENSITY.includes(field) && <Toggle checked={log} onChange={setLog} label={t("charts.log")} />}
          {result.sheath && <Toggle checked={showSheath} onChange={setShowSheath} label={t("charts.sheathEdge")} />}
          <select className="input" value={cmpJob ? cmpJob.id : ""} aria-label={t("charts.compareWith")} title={t("charts.compareHint")} onChange={(e) => setCmpId(e.target.value)}>
            <option value="">{t("charts.compareNone")}</option>
            {candidates.map((j) => (
              <option key={j.id} value={j.id}>
                {t("charts.compareOption", { name: jobName(j) })}
              </option>
            ))}
          </select>
        </>
      }
      note={
        p ? (
          <>
            {t("charts.avgSteps", { n: p.avg_steps })}
            {markers && markers.length > 0 ? ` · ${t("charts.sheathMarkerNote")}` : ""}
            {cp ? ` · ${t("charts.compareNote")}` : ""}
          </>
        ) : undefined
      }
    >
      {!p || !chart ? (
        <p className="hint">{t("charts.noProfiles")}</p>
      ) : (
        <LineChart x={chart.x} series={chart.series} xLabel={`x [${len.unit}]`} yLabel={fieldLabel} logY={logOn} height={190} markers={markers} />
      )}
      {p && cp && cmp && <CompareSummary a={result} b={cmp} nameB={nameB} />}
    </ChartCard>
  );
}

// ---- 位相分解 ----

const CYCLE_FIELDS: Record<Kind1d, ("phi" | "n_e" | "n_i" | "t_e")[]> = { pic1d: ["phi", "n_e", "n_i"], fluid1d: ["phi", "n_e", "n_i", "t_e"] };

function Cycle1dCard({ kind, result, cycle }: { kind: Kind1d; result: Result1d; cycle: Cycle1d }) {
  const { t } = useTranslation();
  const len = useLength();
  const [fieldPref, setField] = useChartPref<string>(`${kind}.cycleField`, "phi");
  const [showSheath] = useChartPref(`${kind}.sheath`, true);
  const bin = useResultsView((s) => Math.min(Math.max(0, s.bin), cycle.bins - 1));
  const fields = CYCLE_FIELDS[kind].filter((f) => cycle[f] !== undefined);
  const field = fields.includes(fieldPref as "phi") ? (fieldPref as "phi" | "n_e" | "n_i" | "t_e") : "phi";
  const rows = cycle[field] as number[][] | undefined;
  const gap = gapOf(result);
  const x = useMemo(() => (result.profiles?.x ?? deriveXGrid(gap, Number(result.settings?.n_cells ?? (rows?.[0]?.length ?? 2) - 1))).map(len.of), [result, gap, len.unit]); // eslint-disable-line react-hooks/exhaustive-deps
  const yRange = useMemo(() => rowsRange(rows), [rows]);
  const series = useMemo<LineSeries[]>(() => (rows ? [{ label: field, values: rows[bin] ?? [], color: FIELD1D_COLORS[field][0] }] : []), [rows, bin, field]);
  // PIC 1D はビンごとのシース端、流体 1D は時間平均のシース端
  const markers = !showSheath
    ? undefined
    : cycle.sheath
      ? sheathMarkers({ left_s: cycle.sheath.s_left[bin] ?? null, right_s: cycle.sheath.s_right[bin] ?? null }, gap, len.of)
      : sheathMarkers(result.sheath, gap, len.of);
  const periodS = cycle.freq_hz > 0 ? 1 / cycle.freq_hz : 0;
  const sPhi = useMemo(() => {
    if (!cycle.sheath || cycle.bins < 2) return null;
    const xs = Array.from({ length: cycle.bins }, (_, i) => (i / cycle.bins) * 360);
    return {
      x: xs,
      series: [
        { label: t("charts.leftS"), values: len.ofAll(cycle.sheath.s_left), color: SHEATH_COLOR },
        { label: t("charts.rightS"), values: len.ofAll(cycle.sheath.s_right), color: SHEATH_RIGHT_COLOR },
      ] as LineSeries[],
    };
  }, [cycle, len.unit, t]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <ChartCard
      title={t("charts.phaseAnim")}
      tools={
        <select className="input" value={field} aria-label={t("charts.phaseField")} onChange={(e) => setField(e.target.value)}>
          {fields.map((f) => (
            <option key={f} value={f}>
              {t(`charts.field1d.${f}`)}
            </option>
          ))}
        </select>
      }
      note={periodS > 0 ? t("charts.freqPeriod", { f: formatSi(cycle.freq_hz, "Hz"), T: formatSi(periodS, "s") }) : undefined}
    >
      <LineChart x={x} series={series} xLabel={`x [${len.unit}]`} yLabel={t(`charts.field1d.${field}`)} yRange={yRange ?? undefined} height={170} markers={markers} />
      <PlaybackBar playback={{ bins: cycle.bins, periodS }} />
      {sPhi && (
        <>
          <div className="chart-subtitle">{t("charts.sPhi", { unit: len.unit })}</div>
          {noValues(sPhi.series) ? (
            <p className="hint">{t("charts.noSheath")}</p>
          ) : (
            <LineChart x={sPhi.x} series={sPhi.series} xLabel={t("charts.phaseDeg")} yLabel={`s [${len.unit}]`} xRange={[0, 360]} height={130} markers={[{ x: (bin / cycle.bins) * 360, color: "#ff5c5c" }]} />
          )}
        </>
      )}
    </ChartCard>
  );
}

// ---- シース振動 (PIC 1D) ----

function PeakTable({ side, color, freq, amp, f0 }: { side: string; color: string; freq: number[]; amp: number[]; f0: number | null }) {
  const len = useLength();
  const { t } = useTranslation();
  const peaks = spectrumPeaks(freq, amp, 5);
  if (!peaks.length) return null;
  return (
    <table className="table peak-table">
      <thead>
        <tr>
          <th style={{ color }}>{side}</th>
          <th>f [MHz]</th>
          {f0 != null && <th>f/f0</th>}
          <th>{t("charts.amplitude", { unit: len.unit })}</th>
        </tr>
      </thead>
      <tbody>
        {peaks.map((p, i) => (
          <tr key={i}>
            <td>{i + 1}</td>
            <td className="mono">{(p.f / 1e6).toFixed(3)}</td>
            {f0 != null && <td className="mono">{(p.f / f0).toFixed(2)}</td>}
            <td className="mono">{formatNumber(len.of(p.a))}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function SheathOscCard({ result }: { result: Pic1dResult }) {
  const { t } = useTranslation();
  const len = useLength();
  const [logSpec, setLogSpec] = useChartPref("pic1d.specLog", true);
  const ts = result.sheath_ts;
  const fft = result.sheath_fft;
  const tsChart = useMemo(
    () =>
      ts
        ? {
            x: ts.t.map((v) => v * 1e6),
            series: [
              { label: t("charts.leftS"), values: len.ofAll(ts.s_left), color: SHEATH_COLOR },
              { label: t("charts.rightS"), values: len.ofAll(ts.s_right), color: SHEATH_RIGHT_COLOR },
            ] as LineSeries[],
          }
        : null,
    [ts, len.unit, t], // eslint-disable-line react-hooks/exhaustive-deps
  );
  // 直流 (k = 0) は平均を引いているので 0 に近く、対数の軸を伸ばすだけなので描かない
  const specChart = useMemo(
    () =>
      fft && fft.freq_hz.length > 1
        ? {
            x: fft.freq_hz.slice(1).map((f) => f / 1e6),
            series: [
              { label: t("charts.left"), values: len.ofAll(fft.amp_left.slice(1)), color: SHEATH_COLOR },
              { label: t("charts.right"), values: len.ofAll(fft.amp_right.slice(1)), color: SHEATH_RIGHT_COLOR },
            ] as LineSeries[],
          }
        : null,
    [fft, len.unit, t], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const harmonics = useMemo(() => {
    if (!fft?.f0_hz || !specChart) return undefined;
    const maxF = specChart.x[specChart.x.length - 1];
    const out: { x: number; color: string }[] = [];
    for (let n = 1; (n * fft.f0_hz) / 1e6 <= maxF && n <= 40; n++) out.push({ x: (n * fft.f0_hz) / 1e6, color: HARMONIC_COLOR });
    return out;
  }, [fft, specChart]);
  if (!tsChart && !specChart) return null;
  return (
    <ChartCard title={t("charts.sheathOsc")} tools={specChart && <Toggle checked={logSpec} onChange={setLogSpec} label={t("charts.logSpectrum")} />}>
      {tsChart && (
        <>
          <div className="chart-subtitle">{t("charts.sT")}</div>
          {noValues(tsChart.series) ? (
            <p className="hint">{t("charts.noSheath")}</p>
          ) : (
            <LineChart x={tsChart.x} series={tsChart.series} xLabel="t [µs]" yLabel={`s [${len.unit}]`} height={140} />
          )}
        </>
      )}
      {specChart && fft && (
        <>
          <div className="chart-subtitle">{t("charts.sSpectrum")}</div>
          <LineChart x={specChart.x} series={specChart.series} xLabel="f [MHz]" yLabel={t("charts.amplitude", { unit: len.unit })} logY={logSpec} height={150} markers={harmonics} />
          <div className="peak-tables">
            <PeakTable side={t("charts.left")} color={SHEATH_COLOR} freq={fft.freq_hz} amp={fft.amp_left} f0={fft.f0_hz} />
            <PeakTable side={t("charts.right")} color={SHEATH_RIGHT_COLOR} freq={fft.freq_hz} amp={fft.amp_right} f0={fft.f0_hz} />
          </div>
        </>
      )}
    </ChartCard>
  );
}

// ---- 履歴 ----

function History1d({ kind, result }: { kind: Kind1d; result: Result1d }) {
  const { t } = useTranslation();
  const h = result.history ?? {};
  const x = h.step ?? [];
  const charts = useMemo(() => {
    const sum = (a?: number[], b?: number[]) => (a && b ? a.map((v, i) => v + (b[i] ?? 0)) : (a ?? []));
    if (kind === "pic1d")
      return [
        {
          title: t("charts.histMacro"),
          series: [
            { label: "N_e", values: h.n_e ?? [], color: ELECTRON_COLOR },
            { label: "N_i", values: h.n_i ?? [], color: ION_COLOR },
          ],
        },
      ];
    return [
      {
        title: t("charts.histDensity"),
        series: [
          { label: "n_e_total", values: h.n_e_total ?? [], color: ELECTRON_COLOR },
          { label: "n_i_total", values: h.n_i_total ?? [], color: ION_COLOR },
        ],
      },
      {
        title: t("charts.histWall"),
        series: [
          { label: t("charts.electronLR"), values: sum(h.wall_left_e, h.wall_right_e), color: ELECTRON_COLOR },
          { label: t("charts.ionLR"), values: sum(h.wall_left_i, h.wall_right_i), color: ION_COLOR },
        ],
      },
    ];
  }, [h, kind, t]);
  if (x.length < 2) return null;
  return (
    <>
      {charts.map((c) => (
        <ChartCard key={c.title} title={c.title}>
          <LineChart x={x} series={c.series as LineSeries[]} xLabel={t("charts.step")} height={130} />
        </ChartCard>
      ))}
    </>
  );
}

// ---- まとめ ----

export function Result1d({ job, kind, result }: { job: JobSummary; kind: Kind1d; result: Result1d }) {
  const { t } = useTranslation();
  const pic = kind === "pic1d" ? (result as Pic1dResult) : null;
  return (
    <>
      <Profile1dCard job={job} kind={kind} result={result} />
      {result.cycle && <Cycle1dCard kind={kind} result={result} cycle={result.cycle} />}
      {pic && <SheathOscCard result={pic} />}
      {pic && pic.eedf.length > 0 && <EedfCard eedf={pic.eedf} csvPrefix="pic1d" prefKey="pic1d.eedf" defaultMode="eepf" />}
      {result.wall_iedf && <WallIedfCard iedf={result.wall_iedf} csvPrefix={kind} prefKey={`${kind}.wallIedf`} hint={kind === "fluid1d" ? t("charts.wallIedfFluidHint") : undefined} />}
      <History1d kind={kind} result={result} />
    </>
  );
}

/** 1D のセンターの値など、サマリで使う */
export function center1d(result: Result1d, key: "n_e" | "n_i" | "t_e"): number | null {
  const p = result.profiles;
  if (!p) return null;
  const v = interpLinear(p.x, p[key], gapOf(result) / 2);
  return Number.isFinite(v) ? v : null;
}

export function ampAtF0(result: Pic1dResult, side: "left" | "right"): number | null {
  const fft = result.sheath_fft;
  if (!fft?.f0_hz) return null;
  return ampAt(fft.freq_hz, side === "left" ? fft.amp_left : fft.amp_right, fft.f0_hz);
}
