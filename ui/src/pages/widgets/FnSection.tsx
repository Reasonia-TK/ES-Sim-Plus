// Fowler–Nordheim 電界放出 (軌道追跡の particles.fn と PIC の pic.fn)。放出面は Dirichlet の辺と導体の
// 領域から選ぶ (少なくとも 1 つ、v1 FnPanel と同じ)。

import { useTranslation } from "react-i18next";
import { OptionalBlock } from "../../forms/blocks";
import { SchemaField, Toggle } from "../../forms/SchemaField";
import { setValue } from "../../forms/useField";
import { useDocument } from "../../model/documentStore";
import { DEFAULT_FN } from "../../schema/defaults";
import { getIn, type Path } from "../../schema/schema";
import { edgeLabel } from "../../tree/treeModel";
import { formatNumber } from "../../util/format";
import { Hint } from "./common";

interface Fn {
  edges: number[];
  regions: string[];
}

export function FnSection({ path, mode }: { path: Path; mode: "trace" | "pic" }) {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const fn = getIn(project, path) as Fn | null | undefined;
  const label = t("widgets.fnEmission");
  const edges = project.geometry.boundaries.filter((b) => b.type === "dirichlet").flatMap((b) => b.edges.map((e) => ({ e, v: b.voltage ?? 0 })));
  const regions = project.geometry.regions.filter((r) => r.type === "conductor");
  const toggleIn = <K extends keyof Fn>(key: K, item: Fn[K][number], on: boolean) => {
    const cur = (fn?.[key] ?? []) as unknown[];
    setValue([...path, key], on ? [...cur, item] : cur.filter((x) => x !== item), label);
  };
  return (
    <OptionalBlock path={path} title={label} defaults={() => structuredClone(DEFAULT_FN)} hint={t("widgets.fnHint")}>
      <SchemaField path={[...path, "phi_ev"]} />
      <SchemaField path={[...path, "beta"]} />
      {mode === "trace" && <SchemaField path={[...path, "n"]} />}
      <SchemaField path={[...path, "init_energy_ev"]} />
      {mode === "pic" && (
        <>
          <SchemaField path={[...path, "macro_weight"]} placeholder={t("widgets.sameAsPlasma")} />
          <SchemaField path={[...path, "seed"]} />
        </>
      )}
      <div className="subsection-title">{t("widgets.fnSurfaces")}</div>
      {edges.length === 0 && regions.length === 0 && <p className="muted">{t("widgets.fnNoSurfaces")}</p>}
      {edges.map(({ e, v }) => (
        <div key={`e${e}`} className="field field-toggle">
          <Toggle checked={fn?.edges.includes(e) ?? false} onChange={(on) => toggleIn("edges", e, on)} label={`${edgeLabel(project, e, t)} (${formatNumber(v)} V)`} />
        </div>
      ))}
      {regions.map((r) => (
        <div key={`r${r.id}`} className="field field-toggle">
          <Toggle checked={fn?.regions.includes(r.id) ?? false} onChange={(on) => toggleIn("regions", r.id, on)} label={`${r.id} (${formatNumber(r.voltage ?? 0)} V)`} />
        </div>
      ))}
      {fn && fn.edges.length === 0 && fn.regions.length === 0 && <Hint tone="warn">{t("widgets.fnNeedSurface")}</Hint>}
    </OptionalBlock>
  );
}
