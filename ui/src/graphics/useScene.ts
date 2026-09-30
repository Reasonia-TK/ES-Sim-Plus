// ビューアに出すもの: 結果を出している実行があればその Scene (ライブ・時間平均・位相分解)、無ければ静電場。
// 表示の切り替え (量・表示・対数) もここでまとめ、ビューアの道具の帯とプロファイルが同じものを使う。

import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { jobName, useJobs } from "../jobs/jobsStore";
import type { JobSummary } from "../jobs/types";
import type { Resources } from "../i18n/ja";
import { useDocument } from "../model/documentStore";
import { useStatic } from "../results/staticResults";
import { useResultsView, type Field2dKind, type Mode2d } from "../results/resultsView";
import { useActiveJob, useRunResult } from "../results/runData";
import { sheathLinesOf } from "../results/sheath";
import { runScene, VIEWER_KINDS, type RunScene } from "./runScene";
import type { Scene } from "./scene";
import { staticScene } from "./staticScene";
import { useViewer } from "./viewerStore";

export interface SceneControls {
  quantities: { value: string; label: string }[];
  quantity: string | null;
  setQuantity: (q: string) => void;
  modes: { value: Mode2d; label: string }[];
  mode: Mode2d | null;
  setMode: ((m: Mode2d) => void) | null;
  log: boolean;
  setLog: (v: boolean) => void;
}

export interface ActiveScene {
  scene: Scene | RunScene;
  controls: SceneControls;
  /** 結果を出している実行 (静電場なら undefined) */
  run: JobSummary | undefined;
}

type QuantityKey = keyof Resources["results"]["q"];

const LOG_KEY: Record<Mode2d, "logLive" | "logField" | "logCycle"> = { live: "logLive", field: "logField", cycle: "logCycle" };

export function useActiveScene(): ActiveScene {
  const { t } = useTranslation();
  const qLabel = (q: string) => t(`results.q.${q as QuantityKey}`);
  const project = useDocument((s) => s.project);
  const stMesh = useStatic((s) => s.mesh);
  const stSolve = useStatic((s) => s.solve);
  const stLatest = useStatic((s) => s.latest);
  const vq = useViewer((s) => s.quantity);
  const vlog = useViewer((s) => s.log);
  const rv = useResultsView();
  const job = useActiveJob();
  const viewerJob = job && VIEWER_KINDS.includes(job.kind) ? job : undefined;
  const { result } = useRunResult(viewerJob);
  const startedFull = useJobs((s) => (viewerJob ? s.startedFull[viewerJob.id] : undefined));
  const frame = useJobs((s) => (viewerJob ? s.frames[viewerJob.id] : undefined));
  const liveMesh = useJobs((s) => (viewerJob ? s.liveMesh[viewerJob.id] : undefined));

  const background = useMemo(
    () =>
      staticScene({ mesh: stMesh, solve: stSolve, latest: stLatest }, vq, project, {
        meshTitle: (nodes, elements) => t("viewer.meshTitle", { nodes, elements }),
        potential: t("viewer.potential"),
        field: t("viewer.fieldAbs"),
      }),
    [stMesh, stSolve, stLatest, vq, project, t],
  );

  if (viewerJob) {
    const kind = viewerJob.kind as Field2dKind;
    const display = rv.display[kind] ?? rv.display.pic;
    const scene = runScene({
      job: viewerJob,
      result,
      started: startedFull,
      frame,
      liveMesh,
      display,
      bin: rv.bin,
      sheath: { contour: rv.sheathContour, alpha: rv.sheathAlpha },
      sheathLines: sheathLinesOf(project),
      label: qLabel,
      name: jobName(viewerJob),
      modeLabel: (m) => t(`results.mode.${m}`),
      background,
    });
    if (scene) {
      const mode = scene.mode;
      const isTrace = viewerJob.kind === "trace";
      const controls: SceneControls = isTrace
        ? staticControls()
        : {
            quantities: scene.quantities.map((q) => ({ value: q, label: qLabel(q) })),
            quantity: scene.quantity,
            setQuantity: (q) => mode && rv.setDisplay(kind, { [mode]: q }),
            modes: scene.modes.map((m) => ({ value: m, label: t(`results.mode.${m}`) })),
            mode,
            setMode: viewerJob.state === "running" ? null : (m) => rv.setDisplay(kind, { mode: m }),
            log: scene.log ?? false,
            setLog: (v) => mode && rv.setDisplay(kind, { [LOG_KEY[mode]]: v }),
          };
      return { scene, controls, run: viewerJob };
    }
  }
  return { scene: background, controls: staticControls(), run: undefined };

  function staticControls(): SceneControls {
    return {
      quantities: stSolve
        ? [
            { value: "v", label: t("viewer.quantity_v") },
            { value: "e_abs", label: t("viewer.quantity_e_abs") },
          ]
        : [],
      quantity: stSolve ? vq : null,
      setQuantity: (q) => useViewer.getState().setQuantity(q as "v" | "e_abs"),
      modes: [],
      mode: null,
      setMode: null,
      log: vlog,
      setLog: (v) => useViewer.getState().setLog(v),
    };
  }
}
