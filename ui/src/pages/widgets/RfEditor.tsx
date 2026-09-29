// RF 成分の編集 (voltage_rf: 単一 / リスト / null)。v1 と同じ: 1 つ目の既定 100 V・13.56 MHz・0°、
// 2 つ目の既定 100 V・2 MHz、最後の成分を消すと RF なし。

import { useTranslation } from "react-i18next";
import { setValue } from "../../forms/useField";
import { Toggle } from "../../forms/SchemaField";
import { useDocument } from "../../model/documentStore";
import { DEFAULT_VOLTAGE_RF, DEFAULT_VOLTAGE_RF_2ND } from "../../schema/defaults";
import { getIn, type Path } from "../../schema/schema";
import { rfComponents, rfValue, type VoltageRf } from "../../util/waveform";
import { DefQuantity, ListItem } from "./common";

export function RfEditor({ path }: { path: Path }) {
  const { t } = useTranslation();
  const raw = useDocument((s) => getIn(s.project, path));
  const comps = rfComponents(raw);
  const label = t("widgets.rf");
  const write = (next: VoltageRf[]) => setValue(path, rfValue(next), label);
  const edit = (i: number, key: keyof VoltageRf, v: number | null) => {
    if (v === null) return;
    write(comps.map((c, j) => (j === i ? { ...c, [key]: v } : c)));
  };
  return (
    <div className="subsection">
      <Toggle checked={comps.length > 0} onChange={(on) => write(on ? [{ ...DEFAULT_VOLTAGE_RF }] : [])} label={t("widgets.rfOn")} />
      {comps.map((c, i) => (
        <ListItem key={i} title={`${t("widgets.rfComponent")} ${i + 1}`} onRemove={() => write(comps.filter((_, j) => j !== i))}>
          <DefQuantity def="VoltageRF" prop="amplitude" value={c.amplitude} onCommit={(v) => edit(i, "amplitude", v)} />
          <DefQuantity def="VoltageRF" prop="freq_hz" value={c.freq_hz} onCommit={(v) => edit(i, "freq_hz", v)} />
          <DefQuantity def="VoltageRF" prop="phase_deg" value={c.phase_deg ?? 0} onCommit={(v) => edit(i, "phase_deg", v)} />
        </ListItem>
      ))}
      {comps.length > 0 && (
        <button type="button" className="button small" onClick={() => write([...comps, { ...DEFAULT_VOLTAGE_RF_2ND }])}>
          {t("widgets.rfAdd")}
        </button>
      )}
    </div>
  );
}
