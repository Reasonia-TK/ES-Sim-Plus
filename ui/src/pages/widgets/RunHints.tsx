// 実行設定の目安 (v1 と同じ): RF 1 周期のステップ数と、位相分解のビン数・平均区間の推奨。

import { useTranslation } from "react-i18next";
import { formatSi } from "../../util/format";
import { phaseBinAdvice, stepsPerPeriod } from "../../util/runHints";
import { Hint } from "./common";

export function RfCycleHint({ freqs, dt }: { freqs: number[]; dt: number | null | undefined }) {
  const { t } = useTranslation();
  if (!freqs.length) return null;
  if (!dt) return <Hint>{t("hints2.dtAuto", { f: freqs.map((f) => formatSi(f, "Hz")).join(", ") })}</Hint>;
  return (
    <Hint>
      {freqs.map((f) => t("hints2.stepsPerPeriod", { f: formatSi(f, "Hz"), n: Number(stepsPerPeriod(f, dt)!.toPrecision(4)).toLocaleString() })).join(" · ")}
    </Hint>
  );
}

export function PhaseBinHint({ freqs, dt, bins, avgSteps, nSteps }: { freqs: number[]; dt: number | null | undefined; bins: number; avgSteps: number | null | undefined; nSteps: number }) {
  const { t } = useTranslation();
  const a = phaseBinAdvice(freqs, dt, bins, avgSteps, nSteps);
  if (!a) return null;
  return (
    <>
      {a.binsTooMany && <Hint tone="warn">{t("hints2.binsTooMany", { n: a.steps })}</Hint>}
      {a.avgTooShort && <Hint tone="warn">{t("hints2.avgTooShort", { n: a.steps, rec: a.recommendedAvg })}</Hint>}
      {!a.binsTooMany && !a.avgTooShort && <Hint>{t("hints2.binsOk", { n: a.steps, rec: a.recommendedAvg })}</Hint>}
    </>
  );
}
