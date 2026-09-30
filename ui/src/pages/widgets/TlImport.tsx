// VHF 定在波: PIC 1D の実行の結果から n_e とシース厚を取り込む (v1 の「1D PIC 結果から取込」)。
// n_e = ギャップ中央の n_i (時間平均)、sheath_m = 左右のシース端のうち正の値の平均。元に戻せる。

import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { logInfo } from "../../app/messages";
import { jobName, useJobs } from "../../jobs/jobsStore";
import { useDocument } from "../../model/documentStore";
import { interpLinear, useLength } from "../../results/charts/common";
import { useRunResult } from "../../results/runData";
import type { Pic1dResult } from "../../results/types";
import { formatNumber } from "../../util/format";
import { Hint } from "./common";

/** 取り込む値 (求められなければ null) */
export function tlValuesFromPic1d(r: Pic1dResult | null): { n_e: number; sheath: number } | null {
  const p = r?.profiles;
  if (!r || !p) return null;
  const gap = Number(r.settings?.gap_m ?? p.x[p.x.length - 1]);
  const n = interpLinear(p.x, p.n_i, gap / 2);
  const s = [r.sheath?.left_s, r.sheath?.right_s].filter((v): v is number => typeof v === "number" && v > 0);
  if (!(n > 0) || !s.length) return null;
  return { n_e: n, sheath: s.reduce((a, b) => a + b, 0) / s.length };
}

export function TlImport() {
  const { t } = useTranslation();
  const len = useLength();
  const jobs = useJobs((s) => s.jobs);
  const runs = useMemo(() => Object.values(jobs).filter((j) => j.kind === "pic1d" && j.has_result && j.state !== "running").sort((a, b) => b.created - a.created), [jobs]);
  const [pick, setPick] = useState<string>("");
  const job = runs.find((j) => j.id === pick) ?? runs[0];
  const { result } = useRunResult<Pic1dResult>(job);
  const v = tlValuesFromPic1d(result);
  if (!runs.length) return <Hint>{t("tlPage.importNone")}</Hint>;
  const apply = () => {
    if (!v || !job) return;
    const label = t("tlPage.importLabel", { name: jobName(job) });
    useDocument.getState().update(label, (d) => {
      const tl = d.tl as Record<string, unknown> | null | undefined;
      if (!tl) return;
      tl.n_e_m3 = v.n_e;
      tl.sheath_m = v.sheath;
    });
    logInfo(t("msg.source.jobs"), label);
  };
  return (
    <div className="subsection">
      <div className="button-row">
        <select className="input" value={job?.id ?? ""} aria-label={t("tlPage.importFrom")} onChange={(e) => setPick(e.target.value)}>
          {runs.map((j) => (
            <option key={j.id} value={j.id}>
              {jobName(j)}
            </option>
          ))}
        </select>
        <button type="button" className="button small" disabled={!v} onClick={apply}>
          {t("tlPage.importButton")}
        </button>
      </div>
      <Hint>{v ? t("tlPage.importValues", { n: v.n_e.toExponential(3), s: formatNumber(Number(len.of(v.sheath).toPrecision(4))), unit: len.unit }) : t("tlPage.importMissing")}</Hint>
    </div>
  );
}
