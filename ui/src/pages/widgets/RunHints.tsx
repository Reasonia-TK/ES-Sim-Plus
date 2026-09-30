// 実行設定の目安 (v1 と同じ): RF 1 周期のステップ数と総ステップ数の周期数、位相分解のビン数・平均区間の推奨。
// dt が自動なら、その種類のいちばん新しい実行で決まった dt で見積もる。

import { useTranslation } from "react-i18next";
import { sortedJobs, useJobs } from "../../jobs/jobsStore";
import type { JobKind } from "../../jobs/types";
import { formatSi } from "../../util/format";
import { phaseBinAdvice, stepsPerPeriod } from "../../util/runHints";
import { Hint } from "./common";

/** その種類のいちばん新しい実行で決まった dt (無ければ null) */
export function useLastDt(kind: JobKind): number | null {
  return useJobs((s) => {
    for (const j of sortedJobs(s.jobs, kind)) {
      const dt = s.started[j.id]?.dt;
      if (typeof dt === "number" && dt > 0) return dt;
    }
    return null;
  });
}

export function RfCycleHint({ kind, freqs, dt, nSteps }: { kind: JobKind; freqs: number[]; dt: number | null | undefined; nSteps: number }) {
  const { t } = useTranslation();
  const last = useLastDt(kind);
  if (!freqs.length) return null;
  const eff = dt || last;
  if (!eff) return <Hint>{t("hints2.dtAuto", { f: freqs.map((f) => formatSi(f, "Hz")).join(", ") })}</Hint>;
  const per = freqs.map((f) => t("hints2.stepsPerPeriod", { f: formatSi(f, "Hz"), n: Number(stepsPerPeriod(f, eff)!.toPrecision(4)).toLocaleString() })).join(" · ");
  const cycles = nSteps * eff * freqs[0];
  return (
    <Hint>
      {per} · {t("hints2.cycles", { c: Number(cycles.toPrecision(3)).toLocaleString(), f: formatSi(freqs[0], "Hz") })}
      {!dt && last ? ` ${t("hints2.lastDt", { dt: formatSi(last, "s") })}` : ""}
    </Hint>
  );
}

export function PhaseBinHint({ kind, freqs, dt, bins, avgSteps, nSteps }: { kind: JobKind; freqs: number[]; dt: number | null | undefined; bins: number; avgSteps: number | null | undefined; nSteps: number }) {
  const { t } = useTranslation();
  const last = useLastDt(kind);
  const a = phaseBinAdvice(freqs, dt || last, bins, avgSteps, nSteps);
  if (!a) return null;
  return (
    <>
      {a.binsTooMany && <Hint tone="warn">{t("hints2.binsTooMany", { n: a.steps })}</Hint>}
      {a.avgTooShort && <Hint tone="warn">{t("hints2.avgTooShort", { n: a.steps, rec: a.recommendedAvg })}</Hint>}
      {!a.binsTooMany && !a.avgTooShort && <Hint>{t("hints2.binsOk", { n: a.steps, rec: a.recommendedAvg })}</Hint>}
    </>
  );
}
