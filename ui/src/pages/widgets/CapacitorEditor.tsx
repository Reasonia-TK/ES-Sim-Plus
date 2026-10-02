// 阻止コンデンサ (自己バイアス、prompts/134): 電極 (導体・Dirichlet の辺・流体 1D の電極) と電源の間の直列の
// コンデンサ。容量の単位は座標系の電荷の単位 (平面 2D は奥行き 1 m あたり F/m、軸対称は全周 F、1D は面積あたり
// F/m²。スキーマは平面 2D の F/m で、軸対称では F と読む)。

import { useTranslation } from "react-i18next";
import { OptionalBlock } from "../../forms/blocks";
import { SchemaField } from "../../forms/SchemaField";
import { useDocument } from "../../model/documentStore";
import { coordOf, type Coord } from "../../model/project";
import type { Path } from "../../schema/schema";
import { Hint } from "./common";

/**
 * 有効にしたときの容量: GEC セルの 2D シミュレーションの文献値 5 nF (軸対称) を座標系に合わせたもの。
 * 平面 2D は奥行き 10 cm で 5 nF になる 5×10⁻⁸ F/m、1D は直径 101.6 mm の電極の面積あたり 5 nF の 6×10⁻⁷ F/m²
 */
export function defaultCapacitance(kind: "2d" | "1d", coord: Coord): number {
  if (kind === "1d") return 6e-7;
  return coord === "xy" ? 5e-8 : 5e-9;
}

export function CapacitorEditor({ path, kind }: { path: Path; kind: "2d" | "1d" }) {
  const { t } = useTranslation();
  const coord = useDocument((s) => coordOf(s.project));
  const base: Path = [...path, "blocking_capacitor"];
  return (
    <OptionalBlock
      path={base}
      title={t("capacitor.title")}
      defaults={() => ({ capacitance: defaultCapacitance(kind, coord), initial_bias_v: 0 })}
      hint={t("capacitor.offHint")}
    >
      <SchemaField path={[...base, "capacitance"]} />
      <SchemaField path={[...base, "initial_bias_v"]} />
      <Hint>{t(kind === "1d" ? "capacitor.hint1d" : "capacitor.hint2d")}</Hint>
    </OptionalBlock>
  );
}
