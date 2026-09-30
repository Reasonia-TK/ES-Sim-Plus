// 設定欄: 選んだノードのページを出す (見出しに「詳細設定」の切り替え)。

import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { AdvancedSwitch } from "../forms/blocks";
import { useDocument } from "../model/documentStore";
import type { Project } from "../model/project";
import { useSelection } from "../model/selection";
import { edgeLabel, type StudyKind } from "../tree/treeModel";
import { BFieldPage, BoundariesPage, DomainPage, EdgePage, ProjectPage, RegionPage, RegionsPage } from "./GeometryPages";
import { MeshPage } from "./MeshPage";
import { RunPage } from "./RunPage";
import { jobName, useJobs } from "../jobs/jobsStore";
import type { JobSummary } from "../jobs/types";
import { FemPage, TracePage } from "./studies/BasicStudies";
import { Fluid1dPage, Fluid2dPage, Pic1dPage } from "./studies/OneDimStudies";
import { DsmcPage, SweepPage, TlPage } from "./studies/OtherStudies";
import { PicPage } from "./studies/PicPage";

function pageTitle(node: string, project: Project, t: TFunction, jobs: Record<string, JobSummary>): string {
  if (node.startsWith("region:")) return `${t("tree.regions")} › ${node.slice(7)}`;
  if (node.startsWith("result:")) {
    const job = jobs[node.slice(7)];
    return `${t("tree.results")} › ${job ? jobName(job) : "-"}`;
  }
  if (node.startsWith("edge:")) return `${t("tree.boundaries")} › ${edgeLabel(project, Number(node.slice(5)), t)}`;
  if (node.startsWith("study:")) return t(`study.${node.slice(6) as StudyKind}`);
  const titles: Record<string, string> = {
    project: t("tree.project"),
    geometry: t("tree.geometry"),
    domain: t("tree.domain"),
    regions: t("tree.regions"),
    boundaries: t("tree.boundaries"),
    mesh: t("tree.mesh"),
    bfield: t("tree.bfield"),
    studies: t("tree.studies"),
    results: t("tree.results"),
  };
  return titles[node] ?? node;
}

const STUDY_PAGES: Record<StudyKind, () => React.ReactNode> = {
  fem: () => <FemPage />,
  trace: () => <TracePage />,
  pic: () => <PicPage />,
  pic1d: () => <Pic1dPage />,
  fluid1d: () => <Fluid1dPage />,
  fluid2d: () => <Fluid2dPage />,
  dsmc: () => <DsmcPage />,
  tl: () => <TlPage />,
  sweep: () => <SweepPage />,
};

function PageBody({ node }: { node: string }) {
  const { t } = useTranslation();
  if (node === "project") return <ProjectPage />;
  if (node === "domain" || node === "geometry") return <DomainPage />;
  if (node === "regions") return <RegionsPage />;
  if (node.startsWith("region:")) return <RegionPage id={node.slice(7)} />;
  if (node === "boundaries") return <BoundariesPage />;
  if (node.startsWith("edge:")) return <EdgePage edge={Number(node.slice(5))} />;
  if (node === "mesh") return <MeshPage />;
  if (node === "bfield") return <BFieldPage />;
  if (node.startsWith("study:")) {
    const page = STUDY_PAGES[node.slice(6) as StudyKind];
    return page ? <>{page()}</> : null;
  }
  if (node === "studies") return <p className="hint">{t("settings.studiesHint")}</p>;
  if (node === "results") return <p className="hint">{t("settings.resultsLater")}</p>;
  if (node.startsWith("result:")) return <RunPage id={node.slice(7)} />;
  return null;
}

export function SettingsPanel() {
  const { t } = useTranslation();
  const node = useSelection((s) => s.activeNode);
  const project = useDocument((s) => s.project);
  const jobs = useJobs((s) => s.jobs);
  return (
    <div className="settings-panel">
      <div className="panel-header">
        <h2 className="panel-title">{pageTitle(node, project, t, jobs)}</h2>
        <span className="spacer" />
        <AdvancedSwitch />
      </div>
      <div className="panel-body">
        <PageBody key={node} node={node} />
      </div>
    </div>
  );
}
