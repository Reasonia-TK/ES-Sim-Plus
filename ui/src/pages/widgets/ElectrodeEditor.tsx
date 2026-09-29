// 1D の電極 (PIC 1D・流体 1D の左右): 直流電圧・γ・RF・FN (PIC 1D のみ)・複数の CSV 波形。
// V = DC + ΣRF + CSV (v1 Pic1dPanel の ElectrodeEditor と同じ)。

import { useTranslation } from "react-i18next";
import { OptionalBlock } from "../../forms/blocks";
import { SchemaField } from "../../forms/SchemaField";
import { useDocument } from "../../model/documentStore";
import { DEFAULT_FN_1D } from "../../schema/defaults";
import { getIn, type Path } from "../../schema/schema";
import { rfComponents, type VoltageWaveform } from "../../util/waveform";
import { Hint } from "./common";
import { RfEditor } from "./RfEditor";
import { VoltagePreview } from "./VoltagePreview";
import { WaveformList } from "./WaveformEditor";

interface Electrode {
  v_dc?: number;
  voltage_rf?: unknown;
  waveforms?: VoltageWaveform[];
}

export function ElectrodeEditor({ path, fn }: { path: Path; fn: boolean }) {
  const { t } = useTranslation();
  const e = (useDocument((s) => getIn(s.project, path)) as Electrode | undefined) ?? {};
  return (
    <>
      <SchemaField path={[...path, "v_dc"]} />
      <SchemaField path={[...path, "see_gamma"]} />
      <RfEditor path={[...path, "voltage_rf"]} />
      <div className="subsection-title">{t("widgets.waveforms")}</div>
      <WaveformList path={[...path, "waveforms"]} />
      <Hint>{t("widgets.electrodeFormula")}</Hint>
      <VoltagePreview dc={e.v_dc ?? 0} rf={rfComponents(e.voltage_rf)} waveforms={e.waveforms ?? []} />
      {fn && (
        <OptionalBlock path={[...path, "fn"]} title={t("widgets.fnEmission")} defaults={() => ({ ...DEFAULT_FN_1D })}>
          <SchemaField path={[...path, "fn", "phi_ev"]} />
          <SchemaField path={[...path, "fn", "beta"]} />
          <SchemaField path={[...path, "fn", "init_energy_ev"]} />
          <SchemaField path={[...path, "fn", "macro_weight"]} placeholder={t("widgets.sameAsPlasma")} />
        </OptionalBlock>
      )}
    </>
  );
}
