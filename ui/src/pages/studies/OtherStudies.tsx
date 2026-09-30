// DSMC・VHF 定在波・パラメータスイープの設定ページ。

import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { Section } from "../../forms/blocks";
import { SchemaField, Toggle } from "../../forms/SchemaField";
import { setValue } from "../../forms/useField";
import { useDocument } from "../../model/documentStore";
import { edgeCount, type Point } from "../../model/project";
import { buildSweepCandidates, DEFAULT_SWEEP, sweepModuleForPath, sweepValues, valueAtPath, type SweepSettings } from "../../model/sweep";
import { DEFAULT_DSMC_BOUNDARY, defaultDsmc, defaultTl } from "../../schema/defaults";
import { formatNumber } from "../../util/format";
import { dsmcParticlesPerCell } from "../../util/runHints";
import { CommitText, Field } from "../inputs";
import { Hint } from "../widgets/common";
import { ListEditor } from "../widgets/ListEditor";
import { SweepCases } from "../widgets/SweepCases";
import { TlImport } from "../widgets/TlImport";
import { RunControls } from "../widgets/RunControls";
import { StudyShell } from "./StudyShell";
import { useSelection } from "../../model/selection";

interface DsmcBoundary {
  edges: number[];
  p1?: Point | null;
  p2?: Point | null;
  type: "wall" | "symmetry" | "inlet" | "outlet";
  pressure_pa?: number | null;
  flow_sccm?: number | null;
}

function DsmcBoundaryRow({ i, b }: { i: number; b: DsmcBoundary }) {
  const { t } = useTranslation();
  const n = useDocument((s) => edgeCount(s.project));
  const base = ["dsmc", "boundaries", i] as const;
  const segment = b.p1 !== null && b.p1 !== undefined;
  const label = t("dsmcPage.boundaries");
  const set = (patch: Partial<DsmcBoundary>) =>
    useDocument.getState().update(label, (d) => {
      const bc = (d.dsmc as { boundaries: DsmcBoundary[] }).boundaries[i];
      Object.assign(bc, patch);
    });
  const flow = b.type === "inlet" && b.flow_sccm !== null && b.flow_sccm !== undefined;
  return (
    <>
      <Field label={t("dsmcPage.range")}>
        {(id) => (
          <select
            id={id}
            className="input"
            value={segment ? "segment" : "edges"}
            onChange={(e) => (e.target.value === "segment" ? set({ edges: [], p1: [0, 0], p2: [0, 0] }) : set({ p1: null, p2: null }))}
          >
            <option value="edges">{t("dsmcPage.rangeEdges")}</option>
            <option value="segment">{t("dsmcPage.rangeSegment")}</option>
          </select>
        )}
      </Field>
      {segment ? (
        <>
          <SchemaField path={[...base, "p1"]} />
          <SchemaField path={[...base, "p2"]} />
        </>
      ) : (
        <Field label={t("dsmcPage.edges")} hint={t("dsmcPage.edgesHint", { n })}>
          {(id, onError) => (
            <CommitText
              id={id}
              value={b.edges.join(", ")}
              onError={onError}
              validate={(s) => {
                const nums = s.split(",").map((x) => x.trim()).filter(Boolean).map(Number);
                return nums.every((x) => Number.isInteger(x) && x >= 0 && x < n) ? null : t("dsmcPage.edgesInvalid", { n });
              }}
              onCommit={(s) => set({ edges: s.split(",").map((x) => x.trim()).filter(Boolean).map(Number) })}
            />
          )}
        </Field>
      )}
      <SchemaField path={[...base, "type"]} />
      <SchemaField path={[...base, "temperature_k"]} />
      {b.type === "inlet" && (
        <>
          <Field label={t("dsmcPage.inletBy")}>
            {(id) => (
              <select
                id={id}
                className="input"
                value={flow ? "flow" : "pressure"}
                onChange={(e) => (e.target.value === "flow" ? set({ flow_sccm: 1.0, pressure_pa: null }) : set({ flow_sccm: null, pressure_pa: 1.0 }))}
              >
                <option value="pressure">{t("dsmcPage.byPressure")}</option>
                <option value="flow">{t("dsmcPage.byFlow")}</option>
              </select>
            )}
          </Field>
          {flow ? <SchemaField path={[...base, "flow_sccm"]} /> : <SchemaField path={[...base, "pressure_pa"]} />}
        </>
      )}
      {b.type === "outlet" && <SchemaField path={[...base, "pressure_pa"]} placeholder={t("dsmcPage.vacuum")} />}
      {!segment && b.edges.length === 0 && <Hint tone="warn">{t("dsmcPage.needRange")}</Hint>}
    </>
  );
}

export function DsmcPage() {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const d = (project.dsmc ?? {}) as { n_particles?: number; mesh_scale?: number };
  const ppc = dsmcParticlesPerCell(project, d.n_particles ?? 50000, d.mesh_scale ?? 1);
  const sel = useSelection((s) => s.selectedPlacement);
  const P = ["dsmc"] as const;
  return (
    <StudyShell settingsKey="dsmc" defaults={defaultDsmc} description={t("dsmcPage.description")} run={<RunControls kind="dsmc" />}>
      <Section title={t("dsmcPage.gas")}>
        <SchemaField path={[...P, "gas", "name"]} />
        <SchemaField path={[...P, "gas", "mass_amu"]} />
        <SchemaField path={[...P, "gas", "d_ref_m"]} />
        <SchemaField path={[...P, "gas", "omega"]} />
        <SchemaField path={[...P, "gas", "t_ref_k"]} />
      </Section>
      <Section title={t("dsmcPage.boundaries")}>
        <ListEditor<DsmcBoundary>
          path={[...P, "boundaries"]}
          label={t("dsmcPage.boundaries")}
          title={(b, i) => `G${i + 1} · ${t(`dsmcPage.type.${b.type}`)}`}
          emptyText={t("dsmcPage.boundariesEmpty")}
          selected={sel?.kind === "gasbc" ? sel.index : null}
          onSelect={(i) => useSelection.getState().selectPlacement({ kind: "gasbc", index: i })}
          create={() => structuredClone(DEFAULT_DSMC_BOUNDARY) as DsmcBoundary}
          render={(b, i) => <DsmcBoundaryRow i={i} b={b} />}
        />
        <Hint>{t("dsmcPage.boundariesHint")}</Hint>
      </Section>
      <Section title={t("studyCommon.run")}>
        <SchemaField path={[...P, "wall_temperature_k"]} />
        <SchemaField path={[...P, "init_pressure_pa"]} />
        <SchemaField path={[...P, "init_temperature_k"]} />
        <SchemaField path={[...P, "mesh_scale"]} />
        <SchemaField path={[...P, "n_particles"]} />
        <Hint tone={ppc < 20 ? "warn" : undefined}>{t("dsmcPage.perCell", { n: formatNumber(Math.round(ppc)) })}</Hint>
        <SchemaField path={[...P, "dt"]} />
        <SchemaField path={[...P, "n_steps"]} />
        <SchemaField path={[...P, "avg_steps"]} />
        <SchemaField path={[...P, "seed"]} />
        <SchemaField path={[...P, "threads"]} />
        <SchemaField path={[...P, "smoothing_passes"]} />
      </Section>
    </StudyShell>
  );
}

export function TlPage() {
  const { t } = useTranslation();
  const tl = (useDocument((s) => s.project.tl) ?? {}) as { radius_m?: number; gap_m?: number; sheath_m?: number; n_periods?: number; n_fft_periods?: number };
  const P = ["tl"] as const;
  const sheathBad = (tl.sheath_m ?? 0) * 2 >= (tl.gap_m ?? Infinity);
  const fftBad = (tl.n_fft_periods ?? 0) >= (tl.n_periods ?? Infinity);
  return (
    <StudyShell settingsKey="tl" defaults={defaultTl} description={t("tlPage.description")} run={<RunControls kind="tl" />}>
      <Section title={t("tlPage.geometry")}>
        <SchemaField path={[...P, "radius_m"]} />
        <SchemaField path={[...P, "gap_m"]} />
        <SchemaField path={[...P, "sheath_m"]} />
        <Hint tone={sheathBad ? "error" : undefined}>{t("tlPage.geometryHint")}</Hint>
      </Section>
      <Section title={t("tlPage.plasma")}>
        <SchemaField path={[...P, "sheath_law"]} />
        <SchemaField path={[...P, "n_e_m3"]} />
        <SchemaField path={[...P, "n_s_ratio"]} />
        <SchemaField path={[...P, "nu_m_hz"]} />
        <TlImport />
      </Section>
      <Section title={t("tlPage.drive")}>
        <SchemaField path={[...P, "freq_hz"]} />
        <SchemaField path={[...P, "v0"]} />
      </Section>
      <Section title={t("tlPage.numerics")}>
        <SchemaField path={[...P, "n_r"]} />
        <SchemaField path={[...P, "n_periods"]} />
        <SchemaField path={[...P, "n_fft_periods"]} />
        {fftBad && <Hint tone="error">{t("tlPage.fftTooLong")}</Hint>}
        <SchemaField path={[...P, "n_harm"]} />
        <SchemaField path={[...P, "dt"]} placeholder={t("tlPage.dtCfl")} />
      </Section>
    </StudyShell>
  );
}

export function SweepPage() {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const s: SweepSettings = { ...DEFAULT_SWEEP, ...((project.ui as { sweep?: Partial<SweepSettings> } | undefined)?.sweep ?? {}) };
  const candidates = useMemo(() => buildSweepCandidates(project, t), [project, t]);
  const custom = s.param_path !== "" && !candidates.some((c) => c.path === s.param_path);
  const label = t("study.sweep");
  const set = (patch: Partial<SweepSettings>) => setValue(["ui", "sweep"], { ...s, ...patch }, label);
  const values = sweepValues(s);
  const current = s.param_path ? valueAtPath(project, s.param_path) : undefined;
  return (
    <>
      <p className="hint">{t("sweepPage.description")}</p>
      <Field label={t("sweepPage.param")}>
        {(id) => (
          <select id={id} className="input" value={custom ? "__custom" : s.param_path} onChange={(e) => set({ param_path: e.target.value === "__custom" ? "custom." : e.target.value })}>
            <option value="">{t("sweepPage.choose")}</option>
            {candidates.map((c) => (
              <option key={c.path} value={c.path}>
                {c.label}
              </option>
            ))}
            <option value="__custom">{t("sweepPage.customPath")}</option>
          </select>
        )}
      </Field>
      {custom && (
        <Field label={t("sweepPage.path")} hint={t("sweepPage.pathHint")}>
          {(id) => <CommitText id={id} value={s.param_path} onCommit={(v) => set({ param_path: v.trim() })} />}
        </Field>
      )}
      {s.param_path && (
        <Hint tone={current === undefined ? "warn" : undefined}>
          {current === undefined ? t("sweepPage.invalidPath") : t("sweepPage.current", { v: formatNumber(current), module: sweepModuleForPath(s.param_path) })}
        </Hint>
      )}
      <Field label={t("sweepPage.values")}>
        {(id) => (
          <select id={id} className="input" value={s.mode} onChange={(e) => set({ mode: e.target.value as SweepSettings["mode"] })}>
            <option value="list">{t("sweepPage.modeList")}</option>
            <option value="range">{t("sweepPage.modeRange")}</option>
          </select>
        )}
      </Field>
      {s.mode === "list" ? (
        <Field label={t("sweepPage.list")} hint={t("sweepPage.listHint")}>
          {(id) => <CommitText id={id} value={s.list_text} onCommit={(v) => set({ list_text: v })} />}
        </Field>
      ) : (
        <>
          {(["start", "end", "count"] as const).map((k) => (
            <Field key={k} label={t(`sweepPage.${k}`)}>
              {(id, onError) => (
                <CommitText
                  id={id}
                  value={String(s[k])}
                  onError={onError}
                  validate={(v) => (Number.isFinite(Number(v)) && v.trim() !== "" && (k !== "count" || (Number.isInteger(Number(v)) && Number(v) >= 1)) ? null : t("input.notNumber"))}
                  onCommit={(v) => set({ [k]: Number(v) })}
                />
              )}
            </Field>
          ))}
          <div className="field field-toggle">
            <Toggle checked={s.log} onChange={(v) => set({ log: v })} label={t("sweepPage.log")} />
          </div>
        </>
      )}
      <Hint tone={values.length === 0 ? "warn" : undefined}>
        {values.length === 0 ? t("sweepPage.noValues") : t("sweepPage.preview", { n: values.length, v: values.slice(0, 8).map(formatNumber).join(", ") + (values.length > 8 ? ", …" : "") })}
      </Hint>
      <Field label={t("sweepPage.parallel")}>
        {(id, onError) => (
          <CommitText
            id={id}
            value={String(s.parallel)}
            onError={onError}
            validate={(v) => (Number.isInteger(Number(v)) && Number(v) >= 1 ? null : t("input.notInteger"))}
            onCommit={(v) => set({ parallel: Number(v) })}
          />
        )}
      </Field>
      <RunControls
        kind="sweep"
        runLabel={t("sweepPage.runSweep")}
        blocked={!s.param_path || current === undefined ? t("sweepPage.needPath") : values.length === 0 ? t("sweepPage.noValues") : null}
        options={() => ({ param_path: s.param_path, values, parallel: s.parallel, module: sweepModuleForPath(s.param_path) })}
      />
      <SweepCases />
    </>
  );
}
