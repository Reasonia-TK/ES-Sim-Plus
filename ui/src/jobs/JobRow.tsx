// ジョブの 1 行: 名前・状態・進捗 (棒・ステップ・経過時間)・操作 (停止・続き・削除・結果を開く)。
// スタディのページ、下部パネル (進捗・ジョブ)、結果のページで共用する。

import { Popover } from "radix-ui";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { askConfirm } from "../app/dialogs";
import { useConnection } from "../backend/connection";
import { useSelection } from "../model/selection";
import { CommitText } from "../pages/inputs";
import { formatElapsed, formatNumber } from "../util/format";
import { continueRun, jobElapsed, jobName, removeRun, stopRun, useJobs } from "./jobsStore";
import { HAS_PHASE_BINS, type JobState, type JobSummary } from "./types";

const TONE: Record<JobState, string> = {
  queued: "muted",
  running: "run",
  done: "ok",
  stopped: "warn",
  error: "error",
  cancelled: "muted",
};

/** 1 秒ごとに描き直す (実行中の経過時間) */
export function useTicker(active: boolean, ms = 1000): number {
  const [n, setN] = useState(0);
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setN((x) => x + 1), ms);
    return () => clearInterval(id);
  }, [active, ms]);
  return n;
}

export function StateBadge({ job }: { job: JobSummary }) {
  const { t } = useTranslation();
  const events = useJobs((s) => s.connected);
  if (!events && !job.imported && (job.state === "running" || job.state === "queued")) return <span className="badge badge-muted">{t("jobs.unknownState")}</span>;
  let text = t(`jobs.state.${job.state}`);
  if (job.state === "running" && job.stopping) text = t("jobs.stopping");
  else if (job.state === "running" && job.progress?.fraction != null) text += ` ${Math.round(job.progress.fraction * 100)}%`;
  else if (job.state === "queued" && job.queue_position != null) text = t("jobs.queuePos", { n: job.queue_position + 1 });
  return <span className={`badge badge-${TONE[job.state]}`}>{text}</span>;
}

export function ProgressBar({ fraction }: { fraction: number | null | undefined }) {
  const indeterminate = fraction === null || fraction === undefined;
  return (
    <div className={`progress${indeterminate ? " indeterminate" : ""}`} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={indeterminate ? undefined : Math.round(fraction * 100)}>
      <div className="progress-bar" style={indeterminate ? undefined : { width: `${Math.max(0, Math.min(1, fraction)) * 100}%` }} />
    </div>
  );
}

function ContinueButton({ job }: { job: JobSummary }) {
  const { t } = useTranslation();
  const started = useJobs((s) => s.started[job.id]);
  const defSteps = Number(started?.n_steps ?? 0) || 1000;
  const [steps, setSteps] = useState(String(defSteps));
  const [avg, setAvg] = useState("");
  const [bins, setBins] = useState("");
  const [open, setOpen] = useState(false);
  const bad = !(Number.isInteger(Number(steps)) && Number(steps) > 0);
  const go = () => {
    if (bad) return;
    const options: Record<string, unknown> = { steps: Number(steps) };
    if (avg.trim()) options.avg_steps = Number(avg);
    if (bins.trim()) options.phase_bins = Number(bins);
    void continueRun(job.id, options);
    setOpen(false);
  };
  return (
    <Popover.Root
      open={open}
      onOpenChange={(o) => {
        if (o) setSteps(String(defSteps));
        setOpen(o);
      }}
    >
      <Popover.Trigger asChild>
        <button type="button" className="button small">
          {t("jobs.continue")}
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content className="popover-content continue-popover" align="end" sideOffset={4} collisionPadding={8}>
          <label className="display-row">
            <span className="display-label">{t("jobs.steps")}</span>
            <CommitText className="input" inputMode="decimal" value={steps} onCommit={setSteps} aria-label={t("jobs.steps")} />
          </label>
          <label className="display-row">
            <span className="display-label">{t("jobs.avgSteps")}</span>
            <CommitText className="input" inputMode="decimal" value={avg} placeholder={t("jobs.keep")} onCommit={setAvg} aria-label={t("jobs.avgSteps")} />
          </label>
          {HAS_PHASE_BINS.includes(job.kind) && (
            <label className="display-row">
              <span className="display-label">{t("jobs.phaseBins")}</span>
              <CommitText className="input" inputMode="decimal" value={bins} placeholder={t("jobs.keep")} onCommit={setBins} aria-label={t("jobs.phaseBins")} />
            </label>
          )}
          <p className="hint">{t("jobs.continueHint")}</p>
          <div className="button-row tight">
            <button type="button" className="button primary small" disabled={bad} onClick={go}>
              {t("jobs.continue")}
            </button>
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

export function JobRow({ job, showKind = true, openable = true }: { job: JobSummary; showKind?: boolean; openable?: boolean }) {
  const { t } = useTranslation();
  const runSince = useJobs((s) => s.runSince[job.id]);
  // 停止・続き・削除はバックエンドにつながっているときだけ (読み込んだ実行の削除はこの画面だけ)
  const online = useJobs((s) => s.connected) && useConnection.getState().status === "connected";
  const canAct = online || Boolean(job.imported);
  const remove = async () => {
    if (await askConfirm(t("jobs.deleteTitle"), t("jobs.deleteMessage", { name: jobName(job) }), { okLabel: t("jobs.delete"), danger: true })) void removeRun(job.id);
  };
  useTicker(job.state === "running");
  const active = job.state === "running" || job.state === "queued";
  const p = job.progress;
  const select = useSelection((s) => s.select);
  return (
    <div className={`job-row job-${job.state}`}>
      <div className="job-row-head">
        {openable ? (
          <button type="button" className="link job-name" onClick={() => select(`result:${job.id}`)}>
            {showKind ? jobName(job) : `#${job.seq}`}
          </button>
        ) : (
          <span className="job-name">{showKind ? jobName(job) : `#${job.seq}`}</span>
        )}
        <StateBadge job={job} />
        <span className="muted small mono">{formatElapsed(jobElapsed(job, runSince))}</span>
        {job.runs > 1 && <span className="muted small">{t("jobs.runsN", { n: job.runs })}</span>}
        <span className="spacer" />
        {active && (
          <button type="button" className="button small" disabled={job.stopping || !canAct} onClick={() => void stopRun(job.id)}>
            {job.state === "queued" ? t("jobs.cancel") : t("jobs.stop")}
          </button>
        )}
        {job.can_continue && canAct && <ContinueButton job={job} />}
        {!active && (
          <button type="button" className="button small danger" disabled={!canAct} onClick={() => void remove()}>
            {t("jobs.delete")}
          </button>
        )}
      </div>
      {job.state === "running" && (
        <div className="job-row-progress">
          <ProgressBar fraction={p?.fraction} />
          {p && (
            <span className="muted small mono">
              {job.kind === "sweep" ? p.step : p.step - (p.step_offset ?? 0)} / {p.n_steps}
            </span>
          )}
          {job.kind === "boltz" && typeof p?.en_td === "number" && <span className="muted small mono">E/N {formatNumber(p.en_td)} Td</span>}
        </div>
      )}
      {job.state === "error" && job.error && <div className="hint hint-error job-error">{job.error}</div>}
      {job.warnings.length > 0 && (
        <div className="hint hint-warn job-warnings" title={job.warnings.join("\n")}>
          {job.warnings[0]}
          {job.warnings.length > 1 ? ` ${t("jobs.moreWarnings", { n: job.warnings.length - 1 })}` : ""}
        </div>
      )}
    </div>
  );
}
