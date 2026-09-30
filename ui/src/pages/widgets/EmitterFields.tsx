// エミッタ (線分・点) の設定: 粒子軌道のエミッタと PIC の注入のエミッタで共通 (v1 ParticlePanel と同じ項目)。
// Maxwell 分布では熱運動が方向の広がりを与えるので広がり半角は使わない。

import { useTranslation } from "react-i18next";
import { SchemaField } from "../../forms/SchemaField";
import { useDocument } from "../../model/documentStore";
import { getIn, type Path } from "../../schema/schema";
import { Hint } from "./common";

export function EmitterFields({ path }: { path: Path }) {
  const { t } = useTranslation();
  const em = useDocument((s) => getIn(s.project, path)) as { kind?: string; energy_dist?: string } | null | undefined;
  const line = (em?.kind ?? "line") === "line";
  const maxwell = em?.energy_dist === "maxwell";
  return (
    <>
      <SchemaField path={[...path, "kind"]} />
      <SchemaField path={[...path, "p1"]} />
      {line && <SchemaField path={[...path, "p2"]} />}
      <SchemaField path={[...path, "n"]} />
      <SchemaField path={[...path, "energy_dist"]} />
      <SchemaField path={[...path, "energy_ev"]} />
      <SchemaField path={[...path, "direction_deg"]} />
      <SchemaField path={[...path, "spread_deg"]} disabled={maxwell} />
      {maxwell && (
        <>
          <Hint>{t("tracePage.maxwellSpread")}</Hint>
          <SchemaField path={[...path, "temperature_ev"]} />
          <SchemaField path={[...path, "seed"]} />
        </>
      )}
    </>
  );
}
