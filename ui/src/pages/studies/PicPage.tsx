// PIC-MCC (2D) の設定ページ (v1 PicPanel の設定部分)。

import { useTranslation } from "react-i18next";
import { OptionalBlock, Section } from "../../forms/blocks";
import { SchemaField, Toggle } from "../../forms/SchemaField";
import { setValue } from "../../forms/useField";
import { useDocument } from "../../model/documentStore";
import { polygonBounds, type Point, type Project } from "../../model/project";
import { DEFAULT_INITIAL_PLASMA, DEFAULT_MCC, DEFAULT_MERGE, DEFAULT_PIC, defaultEmitter } from "../../schema/defaults";
import { edgeLabel } from "../../tree/treeModel";
import { projectFreqs } from "../../util/runHints";
import { Hint } from "../widgets/common";
import { FnSection } from "../widgets/FnSection";
import { ListEditor, nextLabel } from "../widgets/ListEditor";
import { ProcessList } from "../widgets/ProcessList";
import { PhaseBinHint, RfCycleHint } from "../widgets/RunHints";
import { DsmcSourceSelect, useDsmcSource } from "../widgets/DsmcSource";
import { RunControls } from "../widgets/RunControls";
import { StudyShell } from "./StudyShell";
import { useSelection } from "../../model/selection";

interface Pic {
  dt?: number | null;
  n_steps?: number;
  avg_steps?: number | null;
  phase_bins?: number;
  reflect_edges?: number[];
  injection?: { emitter?: { kind?: string } } | null;
}

/** MCC (背景ガス衝突) の設定。PIC 2D と PIC 1D で共用 */
export function MccBlock({ base, allowDsmc }: { base: readonly (string | number)[]; allowDsmc: boolean }) {
  const { t } = useTranslation();
  const m = [...base, "mcc"];
  return (
    <OptionalBlock path={m} title={t("picPage.mcc")} defaults={() => structuredClone(DEFAULT_MCC)} hint={t("picPage.mccHint")}>
      <SchemaField path={[...m, "gas", "name"]} />
      <SchemaField path={[...m, "gas", "pressure_pa"]} />
      <SchemaField path={[...m, "gas", "temperature_k"]} />
      <ProcessList path={[...m, "electron_processes"]} species="electron" />
      <ProcessList path={[...m, "ion_processes"]} species="ion" />
      <SchemaField path={[...m, "ionization_split"]} />
      <SchemaField path={[...m, "ion_energy_frame"]} />
      <SchemaField path={[...m, "seed"]} />
      {allowDsmc ? (
        <>
          <SchemaField path={[...m, "use_dsmc_gas"]} />
          <Hint>{t("picPage.dsmcGasHint")}</Hint>
        </>
      ) : (
        <Hint>{t("picPage.noDsmc1d")}</Hint>
      )}
    </OptionalBlock>
  );
}

function segment(p: Project, fy: number): { p1: Point; p2: Point } {
  const b = polygonBounds(p.geometry.domain.polygon);
  const y = b.y0 + fy * (b.y1 - b.y0);
  return { p1: [b.x0 + 0.25 * (b.x1 - b.x0), y], p2: [b.x0 + 0.75 * (b.x1 - b.x0), y] };
}

/** PIC の実行 (背景ガスに DSMC の結果を使うときは、どの実行の結果かを選ぶ) */
function PicRun() {
  const { t } = useTranslation();
  const useGas = useDocument((s) => Boolean((s.project.pic as { mcc?: { use_dsmc_gas?: boolean } | null } | null)?.mcc?.use_dsmc_gas));
  const src = useDsmcSource();
  return (
    <RunControls
      kind="pic"
      blocked={useGas && !src.available ? t("jobs.dsmcNone") : null}
      options={() => (useGas && src.selected ? { dsmc_job: src.selected } : {})}
      extra={useGas ? <DsmcSourceSelect source={src} /> : null}
    />
  );
}

export function PicPage() {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const pic = (project.pic ?? {}) as Pic;
  const freqs = projectFreqs(project, true);
  const P = ["pic"] as const;
  const inj = [...P, "injection"] as const;
  const injLine = (pic.injection?.emitter?.kind ?? "line") === "line";
  const b = polygonBounds(project.geometry.domain.polygon);
  const reflect = pic.reflect_edges ?? [];
  const sel = useSelection((s) => s.selectedPlacement);
  const selectPlacement = useSelection((s) => s.selectPlacement);
  return (
    <StudyShell settingsKey="pic" defaults={() => structuredClone(DEFAULT_PIC)} description={t("picPage.description")} run={<PicRun />}>
      <OptionalBlock path={[...P, "initial_plasma"]} title={t("picPage.initialPlasma")} defaults={() => ({ ...DEFAULT_INITIAL_PLASMA })}>
        <SchemaField path={[...P, "initial_plasma", "density"]} />
        <SchemaField path={[...P, "initial_plasma", "te_ev"]} />
        <SchemaField path={[...P, "initial_plasma", "ti_ev"]} />
        <SchemaField path={[...P, "initial_plasma", "ion_mass_amu"]} />
        <SchemaField path={[...P, "initial_plasma", "immobile_ions"]} />
        <SchemaField path={[...P, "initial_plasma", "seed"]} />
      </OptionalBlock>
      <OptionalBlock
        path={inj}
        title={t("picPage.injection")}
        defaults={() => ({
          emitter: structuredClone((project.particles as { emitter?: unknown } | null)?.emitter ?? defaultEmitter(project)),
          species: "electron",
          current_a_per_m: 1e-4,
        })}
      >
        <SchemaField path={[...inj, "species"]} />
        <SchemaField path={[...inj, "current_a_per_m"]} />
        <SchemaField path={[...inj, "emitter", "kind"]} />
        <SchemaField path={[...inj, "emitter", "p1"]} />
        {injLine && <SchemaField path={[...inj, "emitter", "p2"]} />}
        <SchemaField path={[...inj, "emitter", "energy_ev"]} />
        <SchemaField path={[...inj, "emitter", "direction_deg"]} />
        <SchemaField path={[...inj, "emitter", "spread_deg"]} />
        {(project.particles as { emitter?: unknown } | null)?.emitter ? (
          <button
            type="button"
            className="button small"
            onClick={() => setValue([...inj, "emitter"], structuredClone((project.particles as { emitter: unknown }).emitter), t("picPage.injection"))}
          >
            {t("picPage.copyTraceEmitter")}
          </button>
        ) : null}
      </OptionalBlock>
      <MccBlock base={P} allowDsmc />
      <SchemaField path={[...P, "see_energy_ev"]} />
      <FnSection path={[...P, "fn"]} mode="pic" />
      <Section title={t("studyCommon.run")}>
        <SchemaField path={[...P, "n_macro"]} />
        <SchemaField path={[...P, "dt"]} />
        <RfCycleHint freqs={freqs} dt={pic.dt} />
        <SchemaField path={[...P, "n_steps"]} />
        <SchemaField path={[...P, "frame_every"]} />
        <SchemaField path={[...P, "ion_subcycle"]} />
        <SchemaField path={[...P, "threads"]} />
        <SchemaField path={[...P, "avg_steps"]} placeholder={t("studyCommon.last25")} />
        <SchemaField path={[...P, "phase_bins"]} />
        <PhaseBinHint freqs={freqs} dt={pic.dt} bins={pic.phase_bins ?? 40} avgSteps={pic.avg_steps} nSteps={pic.n_steps ?? 2000} />
      </Section>
      <OptionalBlock path={[...P, "merge"]} title={t("picPage.merge")} defaults={() => ({ ...DEFAULT_MERGE })} hint={t("picPage.mergeHint")}>
        <SchemaField path={[...P, "merge", "n_max"]} />
        <SchemaField path={[...P, "merge", "every"]} />
      </OptionalBlock>
      <Section title={t("picPage.collectors")}>
        <ListEditor<{ label?: string }>
          path={[...P, "collectors"]}
          label={t("picPage.collectors")}
          max={8}
          title={(c, i) => c.label || `C${i + 1}`}
          emptyText={t("picPage.collectorsEmpty")}
          selected={sel?.kind === "collector" ? sel.index : null}
          onSelect={(i) => selectPlacement({ kind: "collector", index: i })}
          create={(items) => ({ ...segment(project, 0.05), tol: null, label: nextLabel("C", items) })}
          render={(_, i) => (
            <>
              <SchemaField path={[...P, "collectors", i, "label"]} />
              <SchemaField path={[...P, "collectors", i, "p1"]} />
              <SchemaField path={[...P, "collectors", i, "p2"]} />
              <SchemaField path={[...P, "collectors", i, "tol"]} placeholder={t("picPage.tolAuto")} />
            </>
          )}
        />
      </Section>
      <Section title={t("picPage.eedfRegions")}>
        <ListEditor<{ label?: string }>
          path={[...P, "eedf_regions"]}
          label={t("picPage.eedfRegions")}
          max={4}
          title={(r, i) => r.label || `E${i + 1}`}
          emptyText={t("picPage.eedfEmpty")}
          selected={sel?.kind === "eedf" ? sel.index : null}
          onSelect={(i) => selectPlacement({ kind: "eedf", index: i })}
          create={(items) => {
            const cx = (b.x0 + b.x1) / 2;
            const cy = (b.y0 + b.y1) / 2;
            const hw = (b.x1 - b.x0) / 10;
            const hh = (b.y1 - b.y0) / 10;
            return { p1: [cx - hw, cy - hh] as Point, p2: [cx + hw, cy + hh] as Point, label: nextLabel("E", items), bins: 100, e_max_ev: null };
          }}
          render={(_, i) => (
            <>
              <SchemaField path={[...P, "eedf_regions", i, "label"]} />
              <SchemaField path={[...P, "eedf_regions", i, "p1"]} />
              <SchemaField path={[...P, "eedf_regions", i, "p2"]} />
              <SchemaField path={[...P, "eedf_regions", i, "bins"]} />
              <SchemaField path={[...P, "eedf_regions", i, "e_max_ev"]} placeholder={t("input.auto")} />
            </>
          )}
        />
      </Section>
      <Section title={t("picPage.sheathLines")} defaultOpen={false}>
        <ListEditor<{ label?: string }>
          path={[...P, "sheath_lines"]}
          label={t("picPage.sheathLines")}
          max={4}
          title={(s, i) => s.label || `S${i + 1}`}
          emptyText={t("picPage.sheathEmpty")}
          selected={sel?.kind === "sheath" ? sel.index : null}
          onSelect={(i) => selectPlacement({ kind: "sheath", index: i })}
          create={(items) => {
            const cy = (b.y0 + b.y1) / 2;
            return { p1: [b.x0, cy] as Point, p2: [b.x0 + 0.5 * (b.x1 - b.x0), cy] as Point, label: nextLabel("S", items) };
          }}
          render={(_, i) => (
            <>
              <SchemaField path={[...P, "sheath_lines", i, "label"]} />
              <SchemaField path={[...P, "sheath_lines", i, "p1"]} />
              <SchemaField path={[...P, "sheath_lines", i, "p2"]} />
            </>
          )}
        />
        <Hint>{t("picPage.sheathHint")}</Hint>
      </Section>
      <Section title={t("picPage.reflectEdges")} defaultOpen={false}>
        {project.geometry.domain.polygon.map((_, i) => (
          <div key={i} className="field field-toggle">
            <Toggle
              checked={reflect.includes(i)}
              onChange={(on) => setValue([...P, "reflect_edges"], on ? [...reflect, i].sort((a, c) => a - c) : reflect.filter((e) => e !== i), t("picPage.reflectEdges"))}
              label={edgeLabel(project, i, t)}
            />
          </div>
        ))}
        <Hint>{t("picPage.reflectHint")}</Hint>
      </Section>
    </StudyShell>
  );
}
