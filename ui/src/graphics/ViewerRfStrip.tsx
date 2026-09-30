// 2D ビューの RF 波形モニタ (v1 RfPhaseMonitor): PIC のライブ表示のとき、ビューの下に電極の V(t) と今の位相を出す。
// 表示の設定の「RF 波形モニタ」で出し分ける。

import { useTranslation } from "react-i18next";
import type { JobSummary } from "../jobs/types";
import { RfCardBody } from "../results/charts/Charts2d";
import { useChartPref } from "../results/resultsView";
import { useFrame, useRunProject, useRunResult } from "../results/runData";
import type { PicFrame, PicResult } from "../results/types";

export function ViewerRfStrip({ run }: { run: JobSummary }) {
  const { t } = useTranslation();
  const [on] = useChartPref("viewer.rfMonitor", true);
  const project = useRunProject(run);
  const live = useFrame<PicFrame>(run);
  const running = run.state === "running";
  const { result } = useRunResult<PicResult>(running || !on ? undefined : run);
  const frame = running ? live : (result?.frame ?? live);
  if (!on || !frame) return null;
  return (
    <div className="viewer-rf">
      <RfCardBody project={project} time={frame.t} title={t("charts.rfTitle")} height={84} />
    </div>
  );
}
