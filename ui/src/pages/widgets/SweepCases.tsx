// スイープのケースの一覧 (いちばん新しいスイープ): 値・状態 (待ち・実行中のステップ・完了・失敗)・
// 完了したケースを文書に読み込む (元に戻せる、v1 と同じ。結果の読み込みは P6e)。

import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { errorText, logError, logInfo } from "../../app/messages";
import { fetchResult, isTerminal, jobName, sortedJobs, useJobs } from "../../jobs/jobsStore";
import { sweepCase } from "../../jobs/api";
import type { SweepCase } from "../../jobs/types";
import { useDocument } from "../../model/documentStore";
import { normalizeProject } from "../../model/project";
import { formatNumber } from "../../util/format";
import { t } from "../../i18n";

interface SweepSummary {
  summary: { case: number; value: number; ok: boolean; error?: string }[];
  values: number[];
}

export async function loadSweepCase(jobId: string, i: number, label: string): Promise<void> {
  try {
    const raw = await sweepCase(jobId, i);
    const { project } = normalizeProject(raw);
    useDocument.getState().update(label, (d) => {
      const draft = d as unknown as Record<string, unknown>;
      for (const k of Object.keys(draft)) delete draft[k];
      Object.assign(draft, project);
    });
    logInfo(t("msg.source.jobs"), label);
  } catch (e) {
    logError(t("msg.source.jobs"), `${label}: ${errorText(e)}`);
  }
}

export function SweepCases({ jobId }: { jobId?: string }) {
  const { t } = useTranslation();
  const jobs = useJobs((s) => s.jobs);
  const job = jobId ? jobs[jobId] : sortedJobs(jobs, "sweep")[0];
  const cases = useJobs((s) => (job ? s.cases[job.id] : undefined));
  const started = useJobs((s) => (job ? s.started[job.id] : undefined));
  // 取りこぼしたケースの状態 (画面を開き直したなど) は結果の要約から埋める
  useEffect(() => {
    if (!job || !isTerminal(job) || !job.has_result || cases) return;
    void fetchResult<SweepSummary>(job.id).then((r) => {
      if (!r) return;
      const filled: Record<number, SweepCase> = {};
      for (const c of r.summary) filled[c.case] = { value: c.value, ok: c.ok, error: c.error };
      useJobs.setState((s) => ({ cases: { ...s.cases, [job.id]: { ...filled, ...(s.cases[job.id] ?? {}) } } }));
    });
  }, [job, cases]);
  if (!job) return null;
  const values = (started?.values as number[] | undefined) ?? ((job.options.values as number[] | undefined) ?? []);
  return (
    <div className="subsection">
      <div className="subsection-title">
        {t("jobs.sweepCases")} — {jobName(job)}
      </div>
      <table className="table">
        <thead>
          <tr>
            <th>#</th>
            <th>{t("jobs.sweepValue")}</th>
            <th>{t("jobs.sweepStatus")}</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {values.map((v, i) => {
            const c = cases?.[i];
            const status =
              c?.ok === true
                ? t("jobs.state.done")
                : c?.ok === false
                  ? `${t("jobs.state.error")}${c.error ? `: ${c.error}` : ""}`
                  : c?.step !== undefined
                    ? `${t("jobs.state.running")} ${c.step} / ${c.n_steps}`
                    : job.state === "running" || job.state === "queued"
                      ? t("jobs.state.queued")
                      : "-";
            return (
              <tr key={i}>
                <td>{i}</td>
                <td className="mono">{formatNumber(v)}</td>
                <td className={c?.ok === false ? "text-error" : undefined}>{status}</td>
                <td>
                  {c?.ok === true && (
                    <button type="button" className="button small" onClick={() => void loadSweepCase(job.id, i, t("jobs.sweepLoad", { i, v: formatNumber(v) }))}>
                      {t("jobs.sweepLoadButton")}
                    </button>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="hint">{t("jobs.sweepLoadHint")}</p>
    </div>
  );
}
