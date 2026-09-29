import { usePrefs } from "../prefs/prefs";
import { lengthUnitLabel } from "../util/format";

/** 長さの表示単位の記号 (mm / µm) */
export function useLengthUnitLabel(): string {
  return lengthUnitLabel(usePrefs((s) => s.lengthUnit));
}
