// 下部パネルの「進捗」(実行中・待ちのジョブ) と「ジョブ」(全部、同時に実行する数の設定・終わったものを消す)。

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { askConfirm } from "../app/dialogs";
import { errorText, logError } from "../app/messages";
import { useConnection } from "../backend/connection";
import { CommitText } from "../pages/inputs";
import { setLimits } from "./api";
import { JobRow } from "./JobRow";
import { isTerminal, removeRun, sortedJobs, useJobs } from "./jobsStore";

export function ProgressView() {
  const { t } = useTranslation();
  const jobs = useJobs((s) => s.jobs);
  const active = sortedJobs(jobs)
    .filter((j) => j.state === "running" || j.state === "queued")
    .sort((a, b) => (a.state === b.state ? (a.queue_position ?? 0) - (b.queue_position ?? 0) : a.state === "running" ? -1 : 1));
  return (
    <div className="jobs-view">
      {active.length === 0 ? <div className="muted pad">{t("jobs.progressEmpty")}</div> : active.map((j) => <JobRow key={j.id} job={j} />)}
    </div>
  );
}

export function JobsView() {
  const { t } = useTranslation();
  const jobs = useJobs((s) => s.jobs);
  const limits = useJobs((s) => s.limits);
  const events = useJobs((s) => s.connected);
  const hasJobs = useConnection((s) => s.info?.jobs === true);
  const [, setTick] = useState(0);
  const list = sortedJobs(jobs);
  const finished = list.filter(isTerminal);
  return (
    <div className="jobs-view">
      <div className="messages-toolbar">
        <span className="muted small">{t("jobs.runningCount", { n: list.filter((j) => j.state === "running").length, q: list.filter((j) => j.state === "queued").length })}</span>
        {limits && (
          <label className="jobs-limit small">
            {t("jobs.limitsTotal")}
            <CommitText
              className="input"
              inputMode="decimal"
              value={String(limits.max_running)}
              validate={(v) => (Number.isInteger(Number(v)) && Number(v) >= 1 ? null : t("input.notInteger"))}
              onCommit={(v) =>
                void setLimits({ max_running: Number(v) })
                  .then((l) => useJobs.setState({ limits: l }))
                  .catch((e) => logError(t("msg.source.jobs"), errorText(e)))
                  .finally(() => setTick((n) => n + 1))
              }
              aria-label={t("jobs.limitsTotal")}
            />
          </label>
        )}
        <span className="spacer" />
        <button
          type="button"
          className="button small"
          disabled={finished.length === 0}
          onClick={() =>
            void askConfirm(t("jobs.clearTitle"), t("jobs.clearMessage", { n: finished.length }), { okLabel: t("jobs.clearFinished"), danger: true }).then(
              (ok) => ok && void Promise.all(finished.map((j) => removeRun(j.id))),
            )
          }
        >
          {t("jobs.clearFinished")}
        </button>
      </div>
      {!hasJobs ? (
        <div className="muted pad">{t("jobs.oldBackend")}</div>
      ) : !events ? (
        <div className="muted pad">{t("jobs.eventsReconnecting")}</div>
      ) : list.length === 0 ? (
        <div className="muted pad">{t("jobs.listEmpty")}</div>
      ) : (
        list.map((j) => <JobRow key={j.id} job={j} />)
      )}
    </div>
  );
}
