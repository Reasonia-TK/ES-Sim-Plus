// メッシュ: 幅・方式、直交格子の局所細分化 (AMR)、辺の局所メッシュサイズ (非構造のみ)。

import { useTranslation } from "react-i18next";
import { OptionalBlock } from "../forms/blocks";
import { SchemaField, Toggle } from "../forms/SchemaField";
import { setValue } from "../forms/useField";
import { useDocument } from "../model/documentStore";
import { polygonBounds, type Point } from "../model/project";
import { DEFAULT_AMR } from "../schema/defaults";
import { formatNumber, lengthUnitLabel, toDisplayLength } from "../util/format";
import { usePrefs } from "../prefs/prefs";
import { Hint } from "./widgets/common";
import { ListEditor } from "./widgets/ListEditor";
import { MeshBuild } from "./widgets/StaticRun";
import { useSelection } from "../model/selection";

interface Amr {
  max_level?: number;
  adaptive?: boolean;
  pic_regrid_every?: number;
  dsmc_regrid_every?: number;
}

function AmrEditor() {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const amr = (project.mesh.amr ?? null) as Amr | null;
  const base = ["mesh", "amr"] as const;
  const b = polygonBounds(project.geometry.domain.polygon);
  const label = t("mesh.amr");
  return (
    <OptionalBlock path={base} title={label} defaults={() => structuredClone(DEFAULT_AMR)} hint={t("mesh.amrHint")}>
      <SchemaField path={[...base, "max_level"]} />
      <SchemaField path={[...base, "refine_boundaries"]} />
      <SchemaField path={[...base, "buffer_cells"]} />
      <SchemaField path={[...base, "blocking_factor"]} />
      <SchemaField path={[...base, "adaptive"]} />
      {amr?.adaptive && (
        <>
          <SchemaField path={[...base, "adapt_tol"]} />
          <SchemaField path={[...base, "adapt_iters"]} />
        </>
      )}
      <div className="field field-toggle">
        <Toggle checked={(amr?.pic_regrid_every ?? 0) > 0} onChange={(on) => setValue([...base, "pic_regrid_every"], on ? 500 : 0, label)} label={t("mesh.picRegrid")} />
      </div>
      {(amr?.pic_regrid_every ?? 0) > 0 && (
        <>
          <SchemaField path={[...base, "pic_regrid_every"]} />
          <SchemaField path={[...base, "pic_h_over_debye"]} />
        </>
      )}
      <div className="field field-toggle">
        <Toggle checked={(amr?.dsmc_regrid_every ?? 0) > 0} onChange={(on) => setValue([...base, "dsmc_regrid_every"], on ? 500 : 0, label)} label={t("mesh.dsmcRegrid")} />
      </div>
      {(amr?.dsmc_regrid_every ?? 0) > 0 && (
        <>
          <SchemaField path={[...base, "dsmc_regrid_every"]} />
          <SchemaField path={[...base, "dsmc_h_over_mfp"]} />
        </>
      )}
      <div className="subsection-title">{t("mesh.amrRegions")}</div>
      <ListEditor<{ level: number }>
        path={[...base, "regions"]}
        label={t("mesh.amrRegions")}
        title={(r, i) => `R${i + 1} · L${r.level}`}
        emptyText={t("mesh.amrRegionsEmpty")}
        create={() => {
          // 中央に半分の大きさの矩形 (v1 と同じ)
          const cx = (b.x0 + b.x1) / 2;
          const cy = (b.y0 + b.y1) / 2;
          const hw = (b.x1 - b.x0) / 4;
          const hh = (b.y1 - b.y0) / 4;
          return { p1: [cx - hw, cy - hh] as Point, p2: [cx + hw, cy + hh] as Point, level: Math.max(1, amr?.max_level ?? 1) };
        }}
        render={(_, i) => (
          <>
            <SchemaField path={[...base, "regions", i, "p1"]} />
            <SchemaField path={[...base, "regions", i, "p2"]} />
            <SchemaField path={[...base, "regions", i, "level"]} />
          </>
        )}
      />
      <Hint>{t("mesh.amrExplain")}</Hint>
    </OptionalBlock>
  );
}

function EdgeSizes() {
  const { t } = useTranslation();
  const unit = usePrefs((s) => s.lengthUnit);
  const sel = useSelection((s) => s.selectedPlacement);
  const fmt = (p: Point) => `(${formatNumber(toDisplayLength(p[0], unit))}, ${formatNumber(toDisplayLength(p[1], unit))}) ${lengthUnitLabel(unit)}`;
  return (
    <ListEditor<{ p1: Point; p2: Point }>
      path={["mesh", "local_edge_sizes"]}
      label={t("mesh.edgeSizes")}
      title={(e, i) => `M${i + 1} · ${fmt(e.p1)} – ${fmt(e.p2)}`}
      emptyText={t("mesh.edgeSizesEmpty")}
      selected={sel?.kind === "edgeSize" ? sel.index : null}
      onSelect={(i) => useSelection.getState().selectPlacement({ kind: "edgeSize", index: i })}
      render={(_, i) => (
        <>
          <SchemaField path={["mesh", "local_edge_sizes", i, "size"]} />
          <SchemaField path={["mesh", "local_edge_sizes", i, "dist_in"]} />
          <SchemaField path={["mesh", "local_edge_sizes", i, "dist_out"]} />
        </>
      )}
    />
  );
}

export function MeshPage() {
  const { t } = useTranslation();
  const mode = useDocument((s) => s.project.mesh.mode ?? "unstructured");
  return (
    <>
      <SchemaField path={["mesh", "size"]} />
      <SchemaField path={["mesh", "mode"]} />
      <Hint>{t(`mesh.modeHint.${mode}`)}</Hint>
      <MeshBuild />
      {mode === "cartesian" && <AmrEditor />}
      <div className="subsection-title">{t("mesh.edgeSizes")}</div>
      {mode !== "unstructured" && <Hint tone="warn">{t("mesh.edgeSizesIgnored")}</Hint>}
      <EdgeSizes />
    </>
  );
}
