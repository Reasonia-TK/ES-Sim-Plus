// 静電場・粒子軌道のスタディの設定ページ。

import { useTranslation } from "react-i18next";
import { Section } from "../../forms/blocks";
import { SchemaField } from "../../forms/SchemaField";
import { useDocument } from "../../model/documentStore";
import { edgeLabel } from "../../tree/treeModel";
import { formatNumber } from "../../util/format";
import { defaultParticles } from "../../schema/defaults";
import { Hint } from "../widgets/common";
import { FnSection } from "../widgets/FnSection";
import { RunControls } from "../widgets/RunControls";
import { MeshBuild, SolveSummary } from "../widgets/StaticRun";
import { StudyShell } from "./StudyShell";
import { EmitterFields } from "../widgets/EmitterFields";
import type { Project } from "../../model/project";

export function FemPage() {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const electrodes = [
    ...project.geometry.boundaries
      .filter((b) => b.type === "dirichlet")
      .map((b) => `${b.edges.map((e) => edgeLabel(project, e, t)).join(", ")}: ${formatNumber(b.voltage ?? 0)} V`),
    ...project.geometry.regions.filter((r) => r.type === "conductor").map((r) => `${r.id}: ${formatNumber(r.voltage ?? 0)} V`),
  ];
  return (
    <>
      <p className="hint">{t("femPage.description")}</p>
      <div className="subsection-title">{t("femPage.electrodes")}</div>
      {electrodes.length ? (
        <ul className="list">
          {electrodes.map((e) => (
            <li key={e}>{e}</li>
          ))}
        </ul>
      ) : (
        <Hint tone="warn">{t("femPage.noElectrodes")}</Hint>
      )}
      <Hint>{t("femPage.computeHint")}</Hint>
      <MeshBuild />
      <SolveSummary />
    </>
  );
}

/** 任意の粒子で q・m が空なら電子の値 (v1 の既定値と同じ) */
const ELECTRON_Q = -1.602176634e-19;
const ELECTRON_M = 9.1093837015e-31;

function withSpeciesDefaults(p: Project): Project {
  const sp = (p.particles as { species?: { preset?: string; q?: number | null; m?: number | null } } | null | undefined)?.species;
  if (sp?.preset !== "custom" || (sp.q != null && sp.m != null)) return p;
  const q = structuredClone(p);
  const s = (q.particles as { species: { q?: number | null; m?: number | null } }).species;
  s.q ??= ELECTRON_Q;
  s.m ??= ELECTRON_M;
  return q;
}

export function TracePage() {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const ps = project.particles as { fn?: unknown; species?: { preset?: string }; emitter?: { kind?: string; energy_dist?: string } | null } | null | undefined;
  const fnOn = ps?.fn !== null && ps?.fn !== undefined;
  const custom = ps?.species?.preset === "custom";
  const e = ["particles", "emitter"] as const;
  return (
    <StudyShell settingsKey="particles" defaults={() => defaultParticles(project)} description={t("tracePage.description")} run={<RunControls kind="trace" project={() => withSpeciesDefaults(useDocument.getState().project)} />}>
      {fnOn ? (
        <Hint>{t("tracePage.fnReplaces")}</Hint>
      ) : (
        <>
          <Section title={t("tracePage.species")}>
            <SchemaField path={["particles", "species", "preset"]} />
            {custom && (
              <>
                <SchemaField path={["particles", "species", "q"]} placeholder={`${ELECTRON_Q} (${t("tracePage.electronDefault")})`} />
                <SchemaField path={["particles", "species", "m"]} placeholder={`${ELECTRON_M} (${t("tracePage.electronDefault")})`} />
              </>
            )}
          </Section>
          <Section title={t("tracePage.emitter")}>
            <EmitterFields path={[...e]} />
            <Hint>{t("tracePage.emitterCanvasHint")}</Hint>
          </Section>
        </>
      )}
      <FnSection path={["particles", "fn"]} mode="trace" />
      <Section title={t("studyCommon.run")}>
        <SchemaField path={["particles", "dt"]} />
        <SchemaField path={["particles", "n_steps"]} />
        <SchemaField path={["particles", "save_every"]} />
      </Section>
    </StudyShell>
  );
}
