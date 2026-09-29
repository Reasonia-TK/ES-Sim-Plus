// ジョブが終わったときに文書へ書き戻すもの: Boltzmann 係数の表 (v1 と同じく流体の設定の boltz_table へ。
// 電子の係数モデルは切り替えない)。出したときと別の文書を開いていたら書かない。

import { errorText, logError, logInfo, logWarning } from "../app/messages";
import { t } from "../i18n";
import { useDocument } from "../model/documentStore";
import { fetchResult, jobName, onJobDone, useJobs } from "./jobsStore";

let started = false;

export function startJobEffects(): void {
  if (started) return;
  started = true;
  onJobDone((job) => {
    if (job.kind !== "boltz" || job.state !== "done") return;
    const input = useJobs.getState().inputs[job.id];
    const module = job.options.module;
    if (!input || (module !== "fluid1d" && module !== "fluid2d")) return;
    const name = jobName(job);
    if (input.docSerial !== useDocument.getState().docSerial) {
      logWarning(t("msg.source.jobs"), t("jobs.boltzNotApplied", { name }));
      return;
    }
    void fetchResult<{ table: unknown }>(job.id)
      .then((res) => {
        if (!res) return;
        let applied = false;
        useDocument.getState().update(t("jobs.boltzApplyLabel"), (d) => {
          const blk = d[module] as Record<string, unknown> | null | undefined;
          if (blk) {
            blk.boltz_table = res.table;
            applied = true;
          }
        });
        if (applied) logInfo(t("msg.source.jobs"), t("jobs.boltzApplied", { name, module: t(`jobs.kind.${module}`) }));
        else logWarning(t("msg.source.jobs"), t("jobs.boltzNotApplied", { name }));
      })
      .catch((e) => logError(t("msg.source.jobs"), errorText(e)));
  });
}
