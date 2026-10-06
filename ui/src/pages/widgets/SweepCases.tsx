// スイープのケースの一覧 (いちばん新しいスイープ): 値・状態 (待ち・実行中のステップ・完了・失敗)・
// 完了したケースを文書に読み込む (元に戻せる、v1 と同じ)・ケースの結果を「読み込んだ実行」にして表示する。

import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { errorText, logError, logInfo } from "../../app/messages";
import { fetchResult, isTerminal, jobName, sortedJobs, useJobs } from "../../jobs/jobsStore";
import { sweepCase } from "../../jobs/api";
import type { SweepCase, SweepConvergence, SweepSelfBias } from "../../jobs/types";
import { useDocument } from "../../model/documentStore";
import { normalizeProject } from "../../model/project";
import { VIEWER_KINDS } from "../../graphics/runScene";
import { importResultsBundle } from "../../results/bundle";
import { useResultsView } from "../../results/resultsView";
import { formatNumber } from "../../util/format";
import { t } from "../../i18n";

interface SweepSummary {
  summary: { case: number; value: number; ok: boolean; error?: string; self_bias?: SweepSelfBias[]; convergence?: SweepConvergence | null }[];
  values: number[];
}

/** ケースの自己バイアス (阻止コンデンサの電極ごとの V_dc、prompts/134) */
function selfBiasText(c: SweepCase | undefined): string {
  if (!c?.self_bias?.length) return "-";
  return c.self_bias.map((s) => (c.self_bias!.length > 1 ? `${s.label}: ${s.v_dc.toFixed(1)} V` : `${s.v_dc.toFixed(1)} V`)).join(", ");
}

/** ケースの収束の判定 (収束した周期、まだなら最後の状態、prompts/137) */
function convergenceText(c: SweepCase | undefined): string {
  const v = c?.convergence;
  if (!v) return "-";
  if (v.converged && v.converged_period !== null) return t("jobs.sweepConvAt", { n: v.converged_period + 1 });
  return v.status === "fail" || v.status === null ? t("jobs.sweepConvNo") : t(`conv.status.${v.status}`);
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

/** ケースの結果を読み込んだ実行にしてビューア・グラフに出す */
export async function showSweepCase(jobId: string, i: number, source: string): Promise<void> {
  try {
    const raw = await sweepCase(jobId, i);
    const { project, results } = normalizeProject(raw);
    const added = importResultsBundle(results, project, source);
    const last = added[added.length - 1];
    if (!last) return;
    const rv = useResultsView.getState();
    rv.setActiveRun(last.id);
    rv.setGraphicsTab(VIEWER_KINDS.includes(last.kind) ? "view" : "charts");
    logInfo(t("msg.source.jobs"), t("jobs.sweepShown", { name: jobName(last) }));
  } catch (e) {
    logError(t("msg.source.jobs"), `${source}: ${errorText(e)}`);
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
      for (const c of r.summary) filled[c.case] = { value: c.value, ok: c.ok, error: c.error, self_bias: c.self_bias, convergence: c.convergence };
      useJobs.setState((s) => ({ cases: { ...s.cases, [job.id]: { ...filled, ...(s.cases[job.id] ?? {}) } } }));
    });
  }, [job, cases]);
  if (!job) return null;
  const values = (started?.values as number[] | undefined) ?? ((job.options.values as number[] | undefined) ?? []);
  const hasBias = Object.values(cases ?? {}).some((c) => (c.self_bias?.length ?? 0) > 0);
  const hasConv = Object.values(cases ?? {}).some((c) => Boolean(c.convergence));
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
            {hasBias && <th>{t("jobs.sweepSelfBias")}</th>}
            {hasConv && <th>{t("jobs.sweepConvergence")}</th>}
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
                {hasBias && <td className="mono">{selfBiasText(c)}</td>}
                {hasConv && <td>{convergenceText(c)}</td>}
                <td className={c?.ok === false ? "text-error" : undefined}>{status}</td>
                <td>
                  {c?.ok === true && (
                    <div className="button-row tight">
                      <button type="button" className="button small" onClick={() => void loadSweepCase(job.id, i, t("jobs.sweepLoad", { i, v: formatNumber(v) }))}>
                        {t("jobs.sweepLoadButton")}
                      </button>
                      <button type="button" className="button small" onClick={() => void showSweepCase(job.id, i, t("jobs.sweepCaseSource", { name: jobName(job), i, v: formatNumber(v) }))}>
                        {t("jobs.sweepShowButton")}
                      </button>
                    </div>
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
