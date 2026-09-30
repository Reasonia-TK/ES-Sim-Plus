// 結果のグラフ: ビューア・グラフに出している実行 (activeRun) の種類ごとのグラフ。

import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { StateBadge } from "../jobs/JobRow";
import { jobName, useJobs } from "../jobs/jobsStore";
import type { JobSummary } from "../jobs/types";
import { Live1d, Result1d, type Kind1d } from "./charts/Charts1d";
import { Fluid2dCharts, PicCharts } from "./charts/Charts2d";
import { BoltzCharts, TlCharts } from "./charts/OtherCharts";
import { useActiveJob, useRunProject, useRunResult } from "./runData";
import type { BoltzTable, Fluid1dResult, Fluid2dResult, Pic1dResult, PicResult, TlResult } from "./types";

/** グラフのある種類 */
export const CHART_KINDS = ["pic", "pic1d", "fluid1d", "fluid2d", "tl", "boltz"];

/** 結果の取り込み待ち・エラー・結果なしの表示をまとめる */
function ResultGate<T>({ job, children, live }: { job: JobSummary; children: (result: T) => ReactNode; live?: ReactNode }) {
  const { t } = useTranslation();
  const { result, loading, error } = useRunResult<T>(job);
  const hasFrame = useJobs((s) => Boolean(s.frames[job.id]));
  if (job.state === "running" || job.state === "queued") return <>{live ?? <p className="hint">{t("charts.running")}</p>}</>;
  if (result) return <>{children(result)}</>;
  if (loading) return <p className="hint">{t("charts.loading")}</p>;
  if (error) return <p className="hint hint-error">{error}</p>;
  if (hasFrame && live) return <>{live}</>;
  return <p className="hint">{t("charts.noResult")}</p>;
}

function Charts1dView({ job, kind }: { job: JobSummary; kind: Kind1d }) {
  const project = useRunProject(job);
  return (
    <ResultGate<Pic1dResult | Fluid1dResult> job={job} live={<Live1d job={job} kind={kind} project={project} />}>
      {(r) => <Result1d job={job} kind={kind} result={r} />}
    </ResultGate>
  );
}

/** 2D は実行中もグラフ (RF 波形・ライブの履歴) を出し、終わったら結果のグラフを足す */
function Charts2dView({ job }: { job: JobSummary }) {
  const project = useRunProject(job);
  const running = job.state === "running" || job.state === "queued";
  const { result } = useRunResult<PicResult | Fluid2dResult>(running ? undefined : job);
  return job.kind === "pic" ? (
    <PicCharts job={job} project={project} result={running ? null : (result as PicResult | null)} />
  ) : (
    <Fluid2dCharts job={job} project={project} result={running ? null : (result as Fluid2dResult | null)} />
  );
}

function KindCharts({ job }: { job: JobSummary }) {
  const { t } = useTranslation();
  switch (job.kind) {
    case "pic1d":
    case "fluid1d":
      return <Charts1dView job={job} kind={job.kind} />;
    case "pic":
    case "fluid2d":
      return <Charts2dView job={job} />;
    case "tl":
      return <ResultGate<TlResult> job={job}>{(r) => <TlCharts result={r} />}</ResultGate>;
    case "boltz":
      return <ResultGate<{ table: BoltzTable }> job={job}>{(r) => <BoltzCharts table={r.table} />}</ResultGate>;
    case "sweep":
      return <p className="hint">{t("results.noChartsSweep")}</p>;
    case "dsmc":
    case "trace":
      return <p className="hint">{t("results.noCharts2d")}</p>;
    default:
      return <p className="hint">{t("results.noCharts")}</p>;
  }
}

export function ResultsCharts() {
  const { t } = useTranslation();
  const job = useActiveJob();
  if (!job) return <p className="hint pad">{t("results.chartsEmpty")}</p>;
  return (
    <div className="results-charts">
      <div className="results-charts-head">
        <span>{jobName(job)}</span>
        <StateBadge job={job} />
      </div>
      <KindCharts job={job} />
    </div>
  );
}
