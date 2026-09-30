// PIC の背景ガスに使う DSMC の結果 (完了した DSMC の実行から選ぶ、既定は新しいもの)。

import { usePageValue } from "../../app/pageState";
import { useTranslation } from "react-i18next";
import { jobName, sortedJobs, useJobs } from "../../jobs/jobsStore";
import type { JobSummary } from "../../jobs/types";
import { Field } from "../inputs";

export interface DsmcSource {
  available: boolean;
  selected: string | null;
  options: JobSummary[];
  setSelected: (id: string) => void;
}

export function useDsmcSource(): DsmcSource {
  const jobs = useJobs((s) => s.jobs);
  // 停止した実行も場を持っている (v1 も使えた)。読み込んだ実行はバックエンドに無いので除く
  const done = sortedJobs(jobs, "dsmc").filter((j) => (j.state === "done" || j.state === "stopped") && j.has_result && !j.imported);
  const [sel, setSel] = usePageValue<string | null>("pic.dsmcSource", null);
  const selected = sel && done.some((j) => j.id === sel) ? sel : (done[0]?.id ?? null);
  return { available: done.length > 0, selected, options: done, setSelected: setSel };
}

export function DsmcSourceSelect({ source }: { source: DsmcSource }) {
  const { t } = useTranslation();
  if (!source.available) return null;
  return (
    <Field label={t("jobs.dsmcSource")}>
      {(id) => (
        <select id={id} className="input" value={source.selected ?? ""} onChange={(e) => source.setSelected(e.target.value)}>
          {source.options.map((j) => (
            <option key={j.id} value={j.id}>
              {jobName(j)}
            </option>
          ))}
        </select>
      )}
    </Field>
  );
}
