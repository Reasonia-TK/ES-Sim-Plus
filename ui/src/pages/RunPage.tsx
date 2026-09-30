// 実行 (ジョブ) のページ: 状態・時刻・経過時間・実行回数・開始時の情報・警告・エラー・設定、操作 (停止・続き・
// 削除)、結果の保存 (JSON)、その実行の設定を文書に読み込む (元に戻せる)、数値のサマリ。開くとその実行の結果を
// グラフィックス (2D ビュー・グラフ) に出す。

import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { errorText, logError, logInfo } from "../app/messages";
import { saveTextFile } from "../io/fileAccess";
import { jobProject } from "../jobs/api";
import { JobRow, useTicker } from "../jobs/JobRow";
import { fetchResult, jobElapsed, jobName, useJobs } from "../jobs/jobsStore";
import { useDocument } from "../model/documentStore";
import { VIEWER_KINDS } from "../graphics/runScene";
import { normalizeProject, type Project } from "../model/project";
import { CHART_KINDS } from "../results/ResultsCharts";
import { useResultsView, type GraphicsTab } from "../results/resultsView";
import { RunSummary } from "../results/RunSummary";
import { useWatchJob } from "../results/runData";
import { formatElapsed, formatNumber, formatSi } from "../util/format";
import { Hint } from "./widgets/common";
import { SweepCases } from "./widgets/SweepCases";

function when(ts: number | null, lang: string): string {
  if (!ts) return "-";
  return new Intl.DateTimeFormat(lang, { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(ts * 1000);
}

export function RunPage({ id }: { id: string }) {
  const { t, i18n } = useTranslation();
  const job = useJobs((s) => s.jobs[id]);
  const started = useJobs((s) => s.started[id]);
  const input = useJobs((s) => s.inputs[id]);
  const runSince = useJobs((s) => s.runSince[id]);
  useTicker(job?.state === "running");
  useWatchJob(job?.state === "running" ? id : null);
  const exists = Boolean(job);
  useEffect(() => {
    if (exists) useResultsView.getState().setActiveRun(id);
  }, [id, exists]);
  if (!job) return <p className="hint">{t("jobs.gone")}</p>;
  const name = jobName(job);
  const p = job.progress;
  const dt = typeof started?.dt === "number" ? started.dt : null;

  const saveResult = async () => {
    try {
      const res = await fetchResult(job.id);
      if (res === null) return;
      await saveTextFile(`${job.kind}-${job.seq}.json`, JSON.stringify(res), "json", "JSON");
    } catch (e) {
      logError(t("msg.source.jobs"), errorText(e));
    }
  };

  const show = (tab: GraphicsTab) => {
    const rv = useResultsView.getState();
    rv.setActiveRun(job.id);
    rv.setGraphicsTab(tab);
  };

  const loadSettings = async () => {
    try {
      let project: Project | null = input?.project ?? null;
      if (!project) project = (await jobProject(job.id)).project;
      if (!project) return;
      const next = normalizeProject(project).project;
      const label = t("jobs.loadSettingsLabel", { name });
      useDocument.getState().update(label, (d) => {
        const draft = d as unknown as Record<string, unknown>;
        for (const k of Object.keys(draft)) delete draft[k];
        Object.assign(draft, next);
      });
      logInfo(t("msg.source.jobs"), label);
    } catch (e) {
      logError(t("msg.source.jobs"), errorText(e));
    }
  };

  return (
    <>
      <JobRow job={job} openable={false} />
      <div className="kv">
        <span>{t("jobs.kindLabel")}</span>
        <span>{t(`jobs.kind.${job.kind}`)}</span>
        <span>{t("jobs.created")}</span>
        <span>{when(job.created, i18n.language)}</span>
        <span>{t("jobs.startedAt")}</span>
        <span>{when(job.started_at, i18n.language)}</span>
        <span>{t("jobs.finishedAt")}</span>
        <span>{when(job.finished_at, i18n.language)}</span>
        <span>{t("jobs.elapsed")}</span>
        <span className="mono">{formatElapsed(jobElapsed(job, runSince))}</span>
        <span>{t("jobs.runs")}</span>
        <span>{job.runs}</span>
        {p && (
          <>
            <span>{t("jobs.progressLabel")}</span>
            <span className="mono">
              {/* 進捗はフレームごとなので、完了したら最後まで進んだことにする */}
              {job.state === "done" ? p.n_steps : p.step - (p.step_offset ?? 0)} / {p.n_steps}
              {p.step_offset ? ` (${t("jobs.fromStep", { n: p.step_offset })})` : ""}
            </span>
          </>
        )}
        {dt !== null && (
          <>
            <span>dt</span>
            <span className="mono">{formatSi(dt, "s")}</span>
          </>
        )}
        {(typeof started?.effective_threads === "number" || typeof started?.threads === "number") && (
          <>
            <span>{t("jobs.threads")}</span>
            <span>{formatNumber(Number(started.effective_threads ?? started.threads))}</span>
          </>
        )}
      </div>
      {job.warnings.map((w, i) => (
        <Hint key={i} tone="warn">
          {w}
        </Hint>
      ))}
      {Object.keys(job.options).length > 0 && (
        <details>
          <summary>{t("jobs.options")}</summary>
          <pre className="json">{JSON.stringify(job.options, null, 2)}</pre>
        </details>
      )}
      {job.kind === "sweep" && <SweepCases jobId={job.id} />}
      <div className="button-row">
        <button type="button" className="button" disabled={!job.has_result} onClick={() => void saveResult()}>
          {t("jobs.saveResult")}
        </button>
        <button type="button" className="button" onClick={() => void loadSettings()}>
          {t("jobs.loadSettings")}
        </button>
        {VIEWER_KINDS.includes(job.kind) && job.kind !== "trace" && (
          <button type="button" className="button" onClick={() => show("view")}>
            {t("results.showInView")}
          </button>
        )}
        {CHART_KINDS.includes(job.kind) && (
          <button type="button" className="button" onClick={() => show("charts")}>
            {t("results.showInCharts")}
          </button>
        )}
      </div>
      <RunSummary job={job} />
    </>
  );
}
