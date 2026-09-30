// スタディの実行: 「実行」(今の文書の設定でジョブを出す) と、その種類の実行の一覧 (新しい順に 3 件、
// 進捗・停止・続き・削除)。同じスタディを何度でも・同時にでも実行でき、実行ごとに結果が残る。

import { useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { useConnection } from "../../backend/connection";
import { JobRow } from "../../jobs/JobRow";
import { runJob, sortedJobs, useJobs } from "../../jobs/jobsStore";
import type { JobKind, JobSummary } from "../../jobs/types";
import { useDocument } from "../../model/documentStore";
import type { Project } from "../../model/project";
import { Hint } from "./common";

interface RunControlsProps {
  kind: JobKind;
  /** ジョブの設定 (スイープの対象・DSMC の結果の指定など) */
  options?: () => Record<string, unknown>;
  /** 送る文書 (既定は今の文書。スイープは整えた写し) */
  project?: () => Project;
  /** 実行できない理由 (あれば「実行」を押せない) */
  blocked?: string | null;
  /** 実行ボタンの文言 */
  runLabel?: string;
  /** 一覧の上に出すもの */
  extra?: ReactNode;
  /** 一覧に出す件数 */
  shown?: number;
  /** 一覧に出すジョブを絞る (Boltzmann の表は流体 1D/2D ごと) */
  filter?: (job: JobSummary) => boolean;
}

export function RunControls({ kind, options, project, blocked, runLabel, extra, shown = 3, filter }: RunControlsProps) {
  const { t } = useTranslation();
  const connected = useConnection((s) => s.status === "connected");
  const hasJobs = useConnection((s) => s.info?.jobs === true);
  const events = useJobs((s) => s.connected);
  const jobs = useJobs((s) => s.jobs);
  const [all, setAll] = useState(false);
  const [busy, setBusy] = useState(false);
  const list = sortedJobs(jobs, kind).filter((j) => !filter || filter(j));
  const reason = !connected ? t("static.notConnected") : !hasJobs ? t("jobs.oldBackend") : (blocked ?? null);
  const run = async () => {
    setBusy(true);
    await runJob(kind, project?.() ?? useDocument.getState().project, options?.() ?? {});
    setBusy(false);
  };
  return (
    <section className="run-controls" aria-label={t("jobs.runsOf", { kind: t(`jobs.kind.${kind}`) })}>
      <div className="button-row tight">
        <button type="button" className="button primary" disabled={reason !== null || busy} onClick={() => void run()}>
          {runLabel ?? t("jobs.run")}
        </button>
        {connected && hasJobs && !events && <span className="muted small">{t("jobs.eventsReconnecting")}</span>}
      </div>
      {reason && <Hint tone="warn">{reason}</Hint>}
      {extra}
      {list.length > 0 && (
        <div className="job-list">
          {(all ? list : list.slice(0, shown)).map((j) => (
            <JobRow key={j.id} job={j} />
          ))}
          {list.length > shown && (
            <button type="button" className="link small" onClick={() => setAll(!all)}>
              {all ? t("jobs.showLess") : t("jobs.showAll", { n: list.length })}
            </button>
          )}
        </div>
      )}
    </section>
  );
}
