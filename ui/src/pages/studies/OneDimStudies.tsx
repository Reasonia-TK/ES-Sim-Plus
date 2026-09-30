// PIC-MCC (1D) と流体 (1D・2D) の設定ページ (v1 Pic1dPanel・Fluid1dPanel・Fluid2dPanel の設定部分)。

import { usePageValue } from "../../app/pageState";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { apiGet } from "../../backend/api";
import { useConnection } from "../../backend/connection";
import { Section } from "../../forms/blocks";
import { SchemaField } from "../../forms/SchemaField";
import { useDocument } from "../../model/documentStore";
import { defaultFluid1d, defaultFluid2d, defaultPic1d } from "../../schema/defaults";
import { electrodeFreqs, projectFreqs } from "../../util/runHints";
import type { VoltageWaveform } from "../../util/waveform";
import { BoltzSection } from "../widgets/BoltzSection";
import { Hint } from "../widgets/common";
import { ElectrodeEditor } from "../widgets/ElectrodeEditor";
import { ListEditor, nextLabel } from "../widgets/ListEditor";
import { ProcessList } from "../widgets/ProcessList";
import { PhaseBinHint, RfCycleHint } from "../widgets/RunHints";
import { MccBlock } from "./PicPage";
import { RunControls } from "../widgets/RunControls";
import { StudyShell } from "./StudyShell";
import type { Project } from "../../model/project";

interface Electrode {
  voltage_rf?: unknown;
  waveforms?: VoltageWaveform[];
  fn?: unknown;
}

interface RunBlock {
  dt?: number | null;
  n_steps?: number;
  avg_steps?: number | null;
  phase_bins?: number;
  gap_m?: number;
  left?: Electrode;
  right?: Electrode;
  mcc?: { gas: { pressure_pa: number; temperature_k: number }; electron_processes: unknown[] } | null;
  ion_mobility_model?: string;
  init_density_m3?: number;
  init_te_ev?: number;
  ion_mass_amu?: number;
}

interface Preset {
  label: string;
  description: string;
  note?: string;
  pic1d: Record<string, unknown>;
}

function Presets() {
  const { t } = useTranslation();
  const connected = useConnection((s) => s.status === "connected");
  const [presets, setPresets] = usePageValue<Record<string, Preset> | null>("pic1d.presets", null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [key, setKey] = usePageValue("pic1d.presetKey", "");
  useEffect(() => {
    if (!connected || presets) return;
    let alive = true;
    apiGet<Record<string, Preset>>("/pic1d/presets")
      .then((p) => {
        if (!alive) return;
        setPresets(p);
        if (!key) setKey(Object.keys(p)[0] ?? "");
        setError(null);
      })
      .catch((e) => alive && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected, presets, attempt]);
  // 取れなかったら 3 秒ごとに取り直す
  useEffect(() => {
    if (!error || !connected) return;
    const id = setTimeout(() => setAttempt((n) => n + 1), 3000);
    return () => clearTimeout(id);
  }, [error, connected, attempt]);
  if (!connected) return <Hint>{t("widgets.needBackend")}</Hint>;
  if (error) return <Hint tone="error">{t("pic1dPage.presetsError", { error })} {t("pic1dPage.presetsRetry")}</Hint>;
  if (!presets) return null;
  const p = presets[key];
  return (
    <div className="subsection">
      <div className="field-control">
        <select className="input" value={key} onChange={(e) => setKey(e.target.value)} aria-label={t("pic1dPage.presets")}>
          {Object.entries(presets).map(([k, v]) => (
            <option key={k} value={k}>
              {v.label}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="button small"
          disabled={!p}
          onClick={() => useDocument.getState().update(t("pic1dPage.applyPreset"), (d) => void (d.pic1d = structuredClone(p.pic1d)))}
        >
          {t("pic1dPage.applyPreset")}
        </button>
      </div>
      {p && <Hint>{p.description}</Hint>}
      {p?.note && <Hint tone="warn">{p.note}</Hint>}
    </div>
  );
}

/** 電子の係数モデルが Boltzmann なのに係数の表が無い (backend の validator と同じ条件) */
function boltzMissing(p: Project, key: "fluid1d" | "fluid2d"): boolean {
  const b = p[key] as { electron_model?: string; boltz_table?: unknown } | null | undefined;
  return b?.electron_model === "boltzmann" && !b.boltz_table;
}

export function Pic1dPage() {
  const { t } = useTranslation();
  const blk = (useDocument((s) => s.project.pic1d) ?? {}) as RunBlock;
  const freqs = electrodeFreqs(blk.left, blk.right);
  const P = ["pic1d"] as const;
  return (
    <StudyShell settingsKey="pic1d" defaults={defaultPic1d} description={t("pic1dPage.description")} run={<RunControls kind="pic1d" />}>
      <Section title={t("pic1dPage.presets")}>
        <Presets />
      </Section>
      <Section title={t("studyCommon.geometry1d")}>
        <SchemaField path={[...P, "gap_m"]} />
        <SchemaField path={[...P, "n_cells"]} />
      </Section>
      <Section title={t("picPage.initialPlasma")}>
        <SchemaField path={[...P, "init_density_m3"]} />
        <SchemaField path={[...P, "init_te_ev"]} />
        <SchemaField path={[...P, "init_ti_ev"]} />
        <SchemaField path={[...P, "ion_mass_amu"]} />
        <SchemaField path={[...P, "n_macro"]} />
        <SchemaField path={[...P, "seed"]} />
      </Section>
      <Section title={t("studyCommon.leftElectrode")}>
        <ElectrodeEditor path={[...P, "left"]} fn />
      </Section>
      <Section title={t("studyCommon.rightElectrode")}>
        <ElectrodeEditor path={[...P, "right"]} fn />
      </Section>
      <MccBlock base={P} allowDsmc={false} />
      <Section title={t("studyCommon.run")}>
        <SchemaField path={[...P, "see_energy_ev"]} />
        <SchemaField path={[...P, "dt"]} />
        <RfCycleHint kind="pic1d" freqs={freqs} dt={blk.dt} nSteps={blk.n_steps ?? 2000} />
        <SchemaField path={[...P, "n_steps"]} />
        <SchemaField path={[...P, "frame_every"]} />
        <SchemaField path={[...P, "avg_steps"]} placeholder={t("studyCommon.last25")} />
        <SchemaField path={[...P, "phase_bins"]} />
        <PhaseBinHint kind="pic1d" freqs={freqs} dt={blk.dt} bins={blk.phase_bins ?? 40} avgSteps={blk.avg_steps} nSteps={blk.n_steps ?? 2000} />
        <SchemaField path={[...P, "wall_iedf_bins"]} />
      </Section>
      <Section title={t("pic1dPage.eedfRegions")}>
        <ListEditor<{ label?: string }>
          path={[...P, "eedf_regions"]}
          label={t("pic1dPage.eedfRegions")}
          max={4}
          title={(r, i) => r.label || `E${i + 1}`}
          emptyText={t("picPage.eedfEmpty")}
          create={(items) => ({ x1: 0, x2: 0.1 * (blk.gap_m ?? 0.02), label: nextLabel("E", items), bins: 100, e_max_ev: null })}
          render={(_, i) => (
            <>
              <SchemaField path={[...P, "eedf_regions", i, "label"]} />
              <SchemaField path={[...P, "eedf_regions", i, "x1"]} />
              <SchemaField path={[...P, "eedf_regions", i, "x2"]} />
              <SchemaField path={[...P, "eedf_regions", i, "bins"]} />
              <SchemaField path={[...P, "eedf_regions", i, "e_max_ev"]} placeholder={t("input.auto")} />
            </>
          )}
        />
      </Section>
    </StudyShell>
  );
}

/** 流体 1D・2D で共通の物理 (初期値・イオン・ガス・電子の断面積・係数) */
function FluidPhysics({ base }: { base: "fluid1d" | "fluid2d" }) {
  const { t } = useTranslation();
  const blk = (useDocument((s) => s.project[base]) ?? {}) as RunBlock;
  const P = [base] as const;
  return (
    <>
      <Section title={t("fluidPage.initial")}>
        <SchemaField path={[...P, "init_density_m3"]} />
        <SchemaField path={[...P, "init_te_ev"]} />
        <SchemaField path={[...P, "ion_mass_amu"]} />
        <SchemaField path={[...P, "t_i_ev"]} />
        <SchemaField path={[...P, "mu_i_ref"]} />
        <SchemaField path={[...P, "n_ref_m3"]} />
        <SchemaField path={[...P, "ion_mobility_model"]} />
        {blk.ion_mobility_model !== "const" && <SchemaField path={[...P, "frost_c_td"]} />}
        <Hint>{t("fluidPage.frostHint")}</Hint>
      </Section>
      <Section title={t("fluidPage.gas")}>
        <SchemaField path={[...P, "gas_pressure_pa"]} />
        <SchemaField path={[...P, "gas_temperature_k"]} />
      </Section>
      <Section title={t("fluidPage.xs")}>
        <ProcessList path={[...P, "electron_processes"]} species="electron" emptyHint={t("fluidPage.defaultXs")} />
      </Section>
      <Section title={t("fluidPage.coefficients")}>
        <BoltzSection path={P} />
      </Section>
    </>
  );
}

export function Fluid1dPage() {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const blk = (project.fluid1d ?? {}) as RunBlock;
  const pic1d = project.pic1d as RunBlock | null | undefined;
  const freqs = electrodeFreqs(blk.left, blk.right);
  const P = ["fluid1d"] as const;
  const importPic1d = () =>
    useDocument.getState().update(t("fluidPage.importPic1d"), (d) => {
      const src = d.pic1d as unknown as RunBlock;
      const dst = d.fluid1d as unknown as RunBlock & Record<string, unknown>;
      if (!src || !dst) return;
      // v1 と同じ: ギャップ・電極 (FN は外す)・初期密度・Te・イオン質量・位相ビン、MCC があればガスと電子の断面積
      dst.gap_m = src.gap_m;
      for (const side of ["left", "right"] as const) {
        const el = JSON.parse(JSON.stringify(src[side] ?? {})) as Electrode; // draft は structuredClone できない
        delete el.fn;
        dst[side] = el;
      }
      dst.init_density_m3 = src.init_density_m3;
      dst.init_te_ev = src.init_te_ev;
      dst.ion_mass_amu = src.ion_mass_amu;
      dst.phase_bins = src.phase_bins;
      if (src.mcc) {
        dst.gas_pressure_pa = src.mcc.gas.pressure_pa;
        dst.gas_temperature_k = src.mcc.gas.temperature_k;
        dst.electron_processes = JSON.parse(JSON.stringify(src.mcc.electron_processes));
      }
    });
  return (
    <StudyShell settingsKey="fluid1d" defaults={defaultFluid1d} description={t("fluidPage.description1d")} run={<RunControls kind="fluid1d" blocked={boltzMissing(project, "fluid1d") ? t("widgets.boltzNoTable") : null} />}>
      <div className="button-row tight">
        <button type="button" className="button small" disabled={!pic1d} onClick={importPic1d} title={pic1d ? undefined : t("fluidPage.noPic1d")}>
          {t("fluidPage.importPic1d")}
        </button>
      </div>
      <Section title={t("studyCommon.geometry1d")}>
        <SchemaField path={[...P, "gap_m"]} />
        <SchemaField path={[...P, "n_cells"]} />
      </Section>
      <Section title={t("studyCommon.leftElectrode")}>
        <ElectrodeEditor path={[...P, "left"]} fn={false} />
      </Section>
      <Section title={t("studyCommon.rightElectrode")}>
        <ElectrodeEditor path={[...P, "right"]} fn={false} />
      </Section>
      <FluidPhysics base="fluid1d" />
      <Section title={t("studyCommon.run")}>
        <SchemaField path={[...P, "dt"]} />
        <RfCycleHint kind="fluid1d" freqs={freqs} dt={blk.dt} nSteps={blk.n_steps ?? 20000} />
        <SchemaField path={[...P, "n_steps"]} />
        <SchemaField path={[...P, "frame_every"]} />
        <SchemaField path={[...P, "avg_steps"]} placeholder={t("studyCommon.last25")} />
        <SchemaField path={[...P, "phase_bins"]} />
        <PhaseBinHint kind="fluid1d" freqs={freqs} dt={blk.dt} bins={blk.phase_bins ?? 40} avgSteps={blk.avg_steps} nSteps={blk.n_steps ?? 20000} />
        <SchemaField path={[...P, "wall_iedf_bins"]} />
      </Section>
    </StudyShell>
  );
}

export function Fluid2dPage() {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const blk = (project.fluid2d ?? {}) as RunBlock & { linear_solver?: string };
  const hasF1 = project.fluid1d !== null && project.fluid1d !== undefined;
  const freqs = projectFreqs(project, true);
  const P = ["fluid2d"] as const;
  const importFluid1d = () =>
    useDocument.getState().update(t("fluidPage.importFluid1d"), (d) => {
      const src = d.fluid1d as Record<string, unknown> | null | undefined;
      const dst = d.fluid2d as Record<string, unknown> | null | undefined;
      if (!src || !dst) return;
      // v1 と同じ: ガス・初期値・イオンのパラメータ・電子の断面積 (形状と電極はプロジェクトのもの)
      for (const k of ["init_density_m3", "init_te_ev", "gas_pressure_pa", "gas_temperature_k", "ion_mass_amu", "mu_i_ref", "n_ref_m3", "t_i_ev", "ion_mobility_model", "frost_c_td"]) {
        if (src[k] !== undefined) dst[k] = src[k];
      }
      dst.electron_processes = JSON.parse(JSON.stringify(src.electron_processes ?? []));
    });
  return (
    <StudyShell settingsKey="fluid2d" defaults={defaultFluid2d} description={t("fluidPage.description2d")} run={<RunControls kind="fluid2d" blocked={boltzMissing(project, "fluid2d") ? t("widgets.boltzNoTable") : null} />}>
      <div className="button-row tight">
        <button type="button" className="button small" disabled={!hasF1} onClick={importFluid1d} title={hasF1 ? undefined : t("fluidPage.noFluid1d")}>
          {t("fluidPage.importFluid1d")}
        </button>
      </div>
      <Hint>{t("fluidPage.geometryFromProject")}</Hint>
      <FluidPhysics base="fluid2d" />
      <Section title={t("studyCommon.run")}>
        <SchemaField path={[...P, "linear_solver"]} />
        {blk.linear_solver !== "direct" && <SchemaField path={[...P, "threads"]} />}
        <SchemaField path={[...P, "dt"]} />
        <RfCycleHint kind="fluid2d" freqs={freqs} dt={blk.dt} nSteps={blk.n_steps ?? 20000} />
        <SchemaField path={[...P, "n_steps"]} />
        <SchemaField path={[...P, "frame_every"]} />
        <SchemaField path={[...P, "avg_steps"]} placeholder={t("studyCommon.last25")} />
        <SchemaField path={[...P, "phase_bins"]} />
        <PhaseBinHint kind="fluid2d" freqs={freqs} dt={blk.dt} bins={blk.phase_bins ?? 0} avgSteps={blk.avg_steps} nSteps={blk.n_steps ?? 20000} />
      </Section>
    </StudyShell>
  );
}
