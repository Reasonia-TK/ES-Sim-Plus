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
import { SolveSummary } from "../widgets/StaticRun";
import { StudyShell } from "./StudyShell";

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
      <SchemaField path={["solver", "backend"]} />
      <Hint>{t("femPage.computeHint")}</Hint>
      <SolveSummary />
    </>
  );
}

export function TracePage() {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const ps = project.particles as { fn?: unknown; species?: { preset?: string }; emitter?: { kind?: string; energy_dist?: string } | null } | null | undefined;
  const fnOn = ps?.fn !== null && ps?.fn !== undefined;
  const custom = ps?.species?.preset === "custom";
  const line = (ps?.emitter?.kind ?? "line") === "line";
  const maxwell = ps?.emitter?.energy_dist === "maxwell";
  const e = ["particles", "emitter"] as const;
  return (
    <StudyShell settingsKey="particles" defaults={() => defaultParticles(project)} description={t("tracePage.description")} run={<RunControls kind="trace" />}>
      {fnOn ? (
        <Hint>{t("tracePage.fnReplaces")}</Hint>
      ) : (
        <>
          <Section title={t("tracePage.species")}>
            <SchemaField path={["particles", "species", "preset"]} />
            {custom && (
              <>
                <SchemaField path={["particles", "species", "q"]} />
                <SchemaField path={["particles", "species", "m"]} />
              </>
            )}
          </Section>
          <Section title={t("tracePage.emitter")}>
            <SchemaField path={[...e, "kind"]} />
            <SchemaField path={[...e, "p1"]} />
            {line && <SchemaField path={[...e, "p2"]} />}
            <SchemaField path={[...e, "n"]} />
            <SchemaField path={[...e, "energy_dist"]} />
            <SchemaField path={[...e, "energy_ev"]} />
            <SchemaField path={[...e, "direction_deg"]} />
            <SchemaField path={[...e, "spread_deg"]} disabled={maxwell} />
            {maxwell && (
              <>
                <SchemaField path={[...e, "temperature_ev"]} />
                <SchemaField path={[...e, "seed"]} />
              </>
            )}
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
