// 実行のページの数値: 実行中の診断 (最新のフレーム) と結果のサマリ (経過時間・時間の内訳・壁吸収・中心の値・
// シース端など)。種類ごとの中身は v1 の各パネルと同じ (数値の書式は v2 の formatNumber にそろえる)。

import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { useJobs } from "../jobs/jobsStore";
import type { JobSummary } from "../jobs/types";
import { hasArcs, pathHandles } from "../cad/path";
import { coordOf, domainPath, type Project } from "../model/project";
import { formatNumber, formatSi } from "../util/format";
import { ampAtF0, center1d } from "./charts/Charts1d";
import { circuitFrameRows, circuitResultRows } from "./charts/SelfBias";
import { timingRows, useLength } from "./charts/common";
import { useFrame, useRunProject, useRunResult } from "./runData";
import type { BoltzTable, DsmcFrame, DsmcResult, Fluid1dResult, Fluid2dFrame, Fluid2dResult, Frame1d, Pic1dResult, PicDiag, PicFrame, PicResult, TlResult, TraceResult } from "./types";

type Row = [ReactNode, ReactNode];

function Kv({ title, rows, children }: { title: string; rows: (Row | null | false)[]; children?: ReactNode }) {
  const shown = rows.filter((r): r is Row => Boolean(r));
  if (!shown.length && !children) return null;
  return (
    <section className="run-summary">
      <h3>{title}</h3>
      <div className="kv">
        {shown.map(([k, v], i) => (
          <Frag key={i} k={k} v={v} />
        ))}
      </div>
      {children}
    </section>
  );
}

function Frag({ k, v }: { k: ReactNode; v: ReactNode }) {
  return (
    <>
      <span>{k}</span>
      <span className="mono">{v}</span>
    </>
  );
}

const num = (v: number | null | undefined) => (v === null || v === undefined ? "-" : formatNumber(v));
const fix2 = (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? "-" : v.toFixed(2));
const exp3 = (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? "-" : v.toExponential(3));
const pair = (a: ReactNode, b: ReactNode) => (
  <>
    {a} / {b}
  </>
);

/** 時間の内訳 (秒の大きい順、割合)。kind ごとの名前、無い名前はキーのまま */
function useTimingRows(kind: string, timing: Record<string, number> | undefined, skip: string[] = [], sumOnly = false): Row[] {
  const { t, i18n } = useTranslation();
  if (!timing) return [];
  const filtered: Record<string, number> = {};
  for (const [k, v] of Object.entries(timing)) if (!skip.includes(k) && (k !== "total" || !sumOnly)) filtered[k] = v;
  const { rows, total } = timingRows(filtered);
  const label = (k: string) => {
    const key = `summary.timing.${kind}.${k}`;
    return i18n.exists(key) ? (t as unknown as (k: string) => string)(key) : k;
  };
  const out: Row[] = rows.map((r) => [label(r.key), `${r.sec.toFixed(3)} s (${r.pct === null ? "0.0" : r.pct.toFixed(1)}%)`]);
  if (!sumOnly) out.push([t("summary.total"), `${total.toFixed(3)} s`]);
  return out;
}

/** 走査のセル数とセル寸法 (PIC・DSMC の walk の診断) */
function WalkDiag({ est, hMean, hMin }: { est: [number, number] | [number]; hMean: number; hMin: number }) {
  const { t } = useTranslation();
  const ratio = hMean > 0 ? hMin / hMean : 1;
  return (
    <>
      <div className="kv">
        <span>{est.length === 2 ? t("summary.walkCellsEI") : t("summary.walkCells")}</span>
        <span className="mono">{est.map((v) => v.toFixed(2)).join(" / ")}</span>
        <span>{t("summary.cellSize")}</span>
        <span className="mono">
          {hMean.toExponential(3)} / {hMin.toExponential(3)}
        </span>
      </div>
      {ratio < 0.2 && <p className="hint">{t("summary.cellRatioHint", { r: ratio.toFixed(2) })}</p>}
      {Math.max(...est) > 3 && <p className="hint">{t("summary.walkHint")}</p>}
    </>
  );
}

// ---- 1D ----

function Summary1d({ job, kind }: { job: JobSummary; kind: "pic1d" | "fluid1d" }) {
  const { t } = useTranslation();
  const len = useLength();
  const frame = useFrame<Frame1d>(job);
  const { result } = useRunResult<Pic1dResult | Fluid1dResult>(job);
  const timing = useTimingRows(kind, result?.timing);
  const c = frame?.counts ?? {};
  const diag: Row[] =
    kind === "pic1d"
      ? [
          [t("summary.macroCounts"), pair(num(c.n_e), num(c.n_i))],
          [t("summary.wallE"), pair(num(c.wall_left_e), num(c.wall_right_e))],
          [t("summary.wallI"), pair(num(c.wall_left_i), num(c.wall_right_i))],
          [t("summary.collIonSee"), `${num(c.coll_e)} / ${num(c.ion_events)} / ${num(c.see_events)}`],
        ]
      : [
          [t("summary.totalDensity"), pair(num(c.n_e_total), num(c.n_i_total))],
          [t("summary.wallE"), pair(num(c.wall_left_e), num(c.wall_right_e))],
          [t("summary.wallI"), pair(num(c.wall_left_i), num(c.wall_right_i))],
          [t("summary.genTotal"), num(c.gen_total)],
        ];
  const r = result;
  const pic = kind === "pic1d" ? (r as Pic1dResult | null) : null;
  const fl = kind === "fluid1d" ? (r as Fluid1dResult | null) : null;
  const s = (v: number | null | undefined) => (v == null ? "—" : `${formatNumber(len.of(v))} ${len.unit}`);
  const ne = r ? center1d(r, kind === "pic1d" ? "n_i" : "n_e") : null;
  const te = fl ? center1d(fl, "t_e") : null;
  const aL = pic ? ampAtF0(pic, "left") : null;
  const aR = pic ? ampAtF0(pic, "right") : null;
  return (
    <>
      {frame && <Kv title={t("summary.diag")} rows={[...diag, ...(frame.circuit ? circuitFrameRows(frame.circuit, null, t) : [])]} />}
      {r && (
        <Kv
          title={t("summary.title")}
          rows={[
            [t("summary.elapsedWall"), `${r.elapsed_s.toFixed(3)} s`],
            ...timing,
            ...(r.circuit ? circuitResultRows(r.circuit, null, t) : []),
            [t("summary.wallLeft"), pair(num(r.walls.left.electron), num(r.walls.left.ion))],
            [t("summary.wallRight"), pair(num(r.walls.right.electron), num(r.walls.right.ion))],
            fl ? [t("summary.genTotal"), `${formatNumber(fl.gen_total)} m^-2`] : null,
            pic?.fn?.left ? [t("summary.fnLeft"), `${formatNumber(pic.fn.left.j_avg)} A/m^2`] : null,
            pic?.fn?.right ? [t("summary.fnRight"), `${formatNumber(pic.fn.right.j_avg)} A/m^2`] : null,
            ne !== null ? [kind === "pic1d" ? t("summary.centerNi") : t("summary.centerNe"), `${ne.toExponential(3)} m^-3`] : null,
            te !== null ? [t("summary.centerTe"), `${te.toFixed(3)} eV`] : null,
            r.sheath ? [t("summary.sheathLR"), pair(s(r.sheath.left_s), s(r.sheath.right_s))] : null,
            pic?.sheath_fft?.f0_hz ? [t("summary.sheathAmp"), pair(s(aL), s(aR))] : null,
          ]}
        />
      )}
    </>
  );
}

// ---- PIC 2D ----

const WALK_KEYS = ["walk_cells_est_e", "walk_cells_est_i", "h_mean_m", "h_min_m", "walk_cells_est"];

function PicSummary({ job, project }: { job: JobSummary; project: Project | null }) {
  const { t } = useTranslation();
  const live = useFrame<PicFrame>(job);
  const { result } = useRunResult<PicResult>(job);
  const frame = job.state === "running" ? live : (result?.frame ?? live);
  const d: PicDiag | undefined = frame?.diag;
  const rz = project ? coordOf(project) !== "xy" : false;
  const timing = useTimingRows("pic", result?.timing, WALK_KEYS);
  const tm = result?.timing;
  const walk = tm && ["walk_cells_est_e", "walk_cells_est_i", "h_mean_m", "h_min_m"].every((k) => typeof tm[k] === "number");
  const regrids = (result?.regrids ?? []) as { step?: number; n_nodes?: number }[];
  return (
    <>
      {d && (
        <Kv
          title={t("summary.diag")}
          rows={[
            [t("summary.particles"), pair(num(d.n_e), num(d.n_i))],
            [t("summary.phiMinMax"), `${fix2(d.phi_min)} / ${fix2(d.phi_max)} V`],
            [t("summary.walls"), pair(num(d.wall_e), num(d.wall_i))],
            [t("summary.collIonSee"), `${num(d.coll_e)} / ${num(d.ion_events)} / ${num(d.see_events)}`],
            d.merged ? [t("summary.merged"), num(d.merged)] : null,
            [rz ? t("summary.surfQRz") : t("summary.surfQ"), exp3(d.surf_q)],
            d.fn_i ? [rz ? t("summary.fnRz") : t("summary.fn"), exp3(d.fn_i)] : null,
            ...(frame?.circuit ? circuitFrameRows(frame.circuit, project, t) : []),
          ]}
        />
      )}
      {result?.circuit && <Kv title={t("summary.title")} rows={circuitResultRows(result.circuit, project, t)} />}
      {result && (
        <Kv
          title={t("summary.timingTitle")}
          rows={[
            result.elapsed_s != null ? [t("summary.elapsedWall"), `${result.elapsed_s.toFixed(3)} s`] : null,
            result.fields ? [t("summary.avgSteps"), result.fields.avg_steps] : null,
            ...timing,
            regrids.length > 0 ? [t("summary.regrids"), t("summary.regridsValue", { n: regrids.length, nodes: regrids[regrids.length - 1]?.n_nodes ?? "-" })] : null,
          ]}
        >
          {walk && tm && <WalkDiag est={[tm.walk_cells_est_e, tm.walk_cells_est_i]} hMean={tm.h_mean_m} hMin={tm.h_min_m} />}
        </Kv>
      )}
    </>
  );
}

// ---- 流体 2D ----

/** ドメインの頂点 (円弧は弧の中点も) の平均にいちばん近い節点 (v1 と同じ) */
function centerNode(project: Project | null, nodes: [number, number][] | undefined): number | null {
  const path = project ? domainPath(project) : null;
  if (!path?.polygon.length || !nodes?.length) return null;
  const h = pathHandles(path);
  const poly = hasArcs(path) ? [...h.vertices, ...h.midpoints] : h.vertices;
  const cx = poly.reduce((a, p) => a + p[0], 0) / poly.length;
  const cy = poly.reduce((a, p) => a + p[1], 0) / poly.length;
  let best = 0;
  let bd = Infinity;
  nodes.forEach(([x, y], i) => {
    const d = (x - cx) ** 2 + (y - cy) ** 2;
    if (d < bd) {
      bd = d;
      best = i;
    }
  });
  return best;
}

function Fluid2dSummary({ job, project }: { job: JobSummary; project: Project | null }) {
  const { t } = useTranslation();
  const frame = useFrame<Fluid2dFrame>(job);
  const started = useJobs((s) => s.startedFull[job.id] ?? s.started[job.id]);
  const { result } = useRunResult<Fluid2dResult>(job);
  const timing = useTimingRows("fluid2d", result?.timing, ["solver_iters", "total"], true);
  const c = frame?.counts ?? {};
  const mesh = (result?.mesh ?? started?.mesh) as { nodes: [number, number][] } | undefined;
  const idx = result?.fields ? centerNode(project, mesh?.nodes) : null;
  const iters = result?.timing?.solver_iters;
  const iterative = ((result?.settings?.linear_solver as string | undefined) ?? "iterative") === "iterative";
  return (
    <>
      {frame && (
        <Kv
          title={t("summary.diag")}
          rows={[
            [t("summary.totalDensity2d"), pair(num(c.n_e_total), num(c.n_i_total))],
            [t("summary.walls"), pair(num(c.wall_e), num(c.wall_i))],
            [t("summary.genTotal"), num(c.gen_total)],
            [t("summary.surfQTotal"), exp3(c.surf_q)],
            ...(frame.circuit ? circuitFrameRows(frame.circuit, project, t) : []),
          ]}
        />
      )}
      {result && (
        <Kv
          title={t("summary.title")}
          rows={[
            [t("summary.elapsedWall"), `${result.elapsed_s.toFixed(3)} s`],
            ...(result.circuit ? circuitResultRows(result.circuit, project, t) : []),
            result.fields ? [t("summary.avgSteps"), result.fields.avg_steps] : null,
            ...timing,
            [t("summary.total"), `${(result.timing?.total ?? 0).toFixed(3)} s`],
            iterative && typeof iters === "number" ? [t("summary.solverIters"), iters.toLocaleString()] : null,
            [t("summary.walls"), pair(exp3(result.walls.electron), exp3(result.walls.ion))],
            [t("summary.genTotal"), exp3(result.gen_total)],
            idx !== null && result.fields ? [t("summary.centerNe2d"), exp3(result.fields.n_e[idx])] : null,
            idx !== null && result.fields ? [t("summary.centerTe2d"), result.fields.t_e[idx]?.toFixed(3) ?? "-"] : null,
          ]}
        />
      )}
    </>
  );
}

// ---- DSMC ----

function DsmcSummary({ job }: { job: JobSummary }) {
  const { t } = useTranslation();
  const frame = useFrame<DsmcFrame>(job);
  const { result } = useRunResult<DsmcResult>(job);
  const timing = useTimingRows("dsmc", result?.timing, WALK_KEYS, true);
  const tm = result?.timing;
  const walk = tm && ["walk_cells_est", "h_mean_m", "h_min_m"].every((k) => typeof tm[k] === "number");
  return (
    <>
      {job.state === "running" && frame && <Kv title={t("summary.diag")} rows={[[t("summary.dsmcStep"), t("summary.dsmcStepValue", { step: frame.step, n: frame.n_steps, np: frame.n_particles })]]} />}
      {result && (
        <Kv
          title={t("summary.title")}
          rows={[
            [t("summary.simParticles"), num(result.n_particles)],
            [t("summary.macroWeight"), exp3(result.macro_weight)],
            [t("summary.dtUsed"), exp3(result.dt)],
            [t("summary.inflow"), exp3(result.inflow)],
            [t("summary.outflow"), exp3(result.outflow)],
            [t("summary.computeTime"), result.elapsed_s != null ? `${result.elapsed_s.toFixed(3)} s` : "-"],
            ...timing,
          ]}
        >
          {walk && tm && <WalkDiag est={[tm.walk_cells_est]} hMean={tm.h_mean_m} hMin={tm.h_min_m} />}
        </Kv>
      )}
    </>
  );
}

// ---- 粒子軌道 ----

function TraceSummary({ job, project }: { job: JobSummary; project: Project | null }) {
  const { t } = useTranslation();
  const { result: r } = useRunResult<TraceResult>(job);
  if (!r) return null;
  const n = r.status.length;
  const absorbed = r.status.filter((s) => s === "absorbed").length;
  const tofs = r.tof.filter((v): v is number => v !== null);
  const meanTof = tofs.length ? tofs.reduce((a, b) => a + b, 0) / tofs.length : null;
  const es = r.final_energy_ev.filter(Number.isFinite);
  const angles = r.final_angle_deg.filter((_, i) => r.status[i] === "absorbed");
  const mean = angles.length ? angles.reduce((a, b) => a + b, 0) / angles.length : null;
  const std = mean !== null ? Math.sqrt(angles.reduce((a, b) => a + (b - mean) ** 2, 0) / angles.length) : null;
  const rz = project ? coordOf(project) !== "xy" : false;
  return (
    <Kv
      title={t("summary.traceTitle")}
      rows={[
        job.elapsed_s != null ? [t("summary.computeTime"), `${job.elapsed_s.toFixed(3)} s`] : null,
        [t("summary.nParticles"), n],
        [t("summary.absorbedAlive"), pair(absorbed, n - absorbed)],
        [t("summary.meanTof"), meanTof !== null ? `${meanTof.toExponential(3)} s` : "-"],
        [t("summary.finalEnergy"), es.length ? `${Math.min(...es).toExponential(3)} / ${Math.max(...es).toExponential(3)} eV` : "-"],
        [t("summary.angleMeanStd"), mean !== null && std !== null ? `${mean.toFixed(2)} ± ${std.toFixed(2)} deg` : "-"],
        [t("summary.angleMinMax"), angles.length ? `${Math.min(...angles).toFixed(2)} / ${Math.max(...angles).toFixed(2)} deg` : "-"],
        r.fn_current != null ? [t("summary.fnTotal"), `${r.fn_current.toExponential(3)} ${rz ? "A" : "A/m"}`] : null,
      ]}
    />
  );
}

// ---- VHF 定在波 ----

function TlSummary({ job }: { job: JobSummary }) {
  const { t } = useTranslation();
  const len = useLength();
  const started = useJobs((s) => s.started[job.id]);
  const { result: r } = useRunResult<TlResult>(job);
  const dt = typeof started?.dt === "number" ? started.dt : r?.dt;
  const nSteps = typeof started?.n_steps === "number" ? started.n_steps : r?.n_steps;
  return (
    <>
      {dt != null && <Kv title={t("summary.tlRun")} rows={[[t("summary.dtSteps"), `${formatSi(dt, "s")} / ${nSteps ?? "-"}`]]} />}
      {r && (
        <Kv
          title={t("summary.tlTitle")}
          rows={[
            [t("summary.lambdaEff"), r.lambda_eff.lambda_m != null ? `${formatNumber(len.of(r.lambda_eff.lambda_m))} ${len.unit}` : t("summary.lambdaNone")],
            [t("summary.lambda0"), `${formatNumber(len.of(r.lambda_eff.lambda0_m))} ${len.unit}`],
            [t("summary.lambdaRatio"), num(r.lambda_eff.ratio)],
            [t("summary.powerMaxMin"), num(r.power.max_over_min)],
            [t("summary.powerStd"), num(r.power.area_weighted_std_over_mean)],
            [t("summary.feedPower"), `${formatNumber(r.feed_power)} W`],
          ]}
        >
          {r.warnings.map((w, i) => (
            <p key={i} className="hint hint-warn">
              {w}
            </p>
          ))}
        </Kv>
      )}
    </>
  );
}

// ---- Boltzmann 係数の表 ----

function BoltzSummary({ job }: { job: JobSummary }) {
  const { t } = useTranslation();
  const { result } = useRunResult<{ table: BoltzTable }>(job);
  const tb = result?.table;
  if (!tb || !tb.en_td.length) return null;
  const range = (a: number[], f: (v: number) => string) => `${f(Math.min(...a))} 〜 ${f(Math.max(...a))}`;
  return (
    <Kv
      title={t("summary.boltzTitle")}
      rows={[
        [t("summary.boltzPoints"), tb.en_td.length],
        [t("summary.boltzEnergy"), range(tb.mean_energy_ev, (v) => v.toFixed(3))],
        [t("summary.boltzEn"), range(tb.en_td, formatNumber)],
      ]}
    >
      {tb.warnings.map((w, i) => (
        <p key={i} className="hint hint-warn">
          {w}
        </p>
      ))}
    </Kv>
  );
}

export function RunSummary({ job }: { job: JobSummary }) {
  const project = useRunProject(job);
  switch (job.kind) {
    case "pic1d":
    case "fluid1d":
      return <Summary1d job={job} kind={job.kind} />;
    case "pic":
      return <PicSummary job={job} project={project} />;
    case "fluid2d":
      return <Fluid2dSummary job={job} project={project} />;
    case "dsmc":
      return <DsmcSummary job={job} />;
    case "trace":
      return <TraceSummary job={job} project={project} />;
    case "tl":
      return <TlSummary job={job} />;
    case "boltz":
      return <BoltzSummary job={job} />;
    default:
      return null;
  }
}
