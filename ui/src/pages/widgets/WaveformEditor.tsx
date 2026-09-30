// CSV 波形の取り込み (voltage_waveform は 1 つ、1D 電極の waveforms は複数)。v1 と同じ規則で読む
// (1 列目 = 時刻、2 列目 = 電圧、区切りは , タブ 空白、数値でない行は飛ばす、周波数 = 1/(t_max − t_min))。

import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { setValue } from "../../forms/useField";
import { useDocument } from "../../model/documentStore";
import { getIn, type Path } from "../../schema/schema";
import { parseWaveformCsv, type VoltageWaveform } from "../../util/waveform";
import { DefQuantity, Hint, ListItem } from "./common";

function useCsvPicker(onLoaded: (wf: VoltageWaveform) => void) {
  const { t } = useTranslation();
  const input = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);
  const picker = (
    <input
      ref={input}
      type="file"
      accept=".csv,.txt,text/csv,text/plain"
      hidden
      onChange={async (e) => {
        const f = e.target.files?.[0];
        e.target.value = "";
        if (!f) return;
        const r = parseWaveformCsv(await f.text());
        if ("error" in r) {
          setError(r.error === "fewRows" ? t("widgets.csvFewRows") : t("widgets.csvZeroSpan"));
          return;
        }
        setError(null);
        onLoaded({ freq_hz: r.freqHz, phase: r.phase, v: r.v });
      }}
    />
  );
  return { picker, open: () => input.current?.click(), error };
}

/** 1 つの CSV 波形 (境界条件の辺・導体の領域) */
export function WaveformEditor({ path }: { path: Path }) {
  const { t } = useTranslation();
  const wf = useDocument((s) => getIn(s.project, path)) as VoltageWaveform | null | undefined;
  const label = t("widgets.waveform");
  const { picker, open, error } = useCsvPicker((w) => setValue(path, w, label));
  return (
    <div className="subsection">
      {picker}
      <div className="button-row tight">
        <button type="button" className="button small" onClick={open}>
          {t("widgets.csvImport")}
        </button>
        {wf && (
          <button type="button" className="button small danger" onClick={() => setValue(path, null, label)}>
            {t("widgets.csvClear")}
          </button>
        )}
      </div>
      {error && <Hint tone="error">{error}</Hint>}
      <Hint>{t("widgets.csvFormatHint")}</Hint>
      {wf && (
        <>
          <p className="muted">{t("widgets.csvPoints", { n: wf.phase.length })}</p>
          <DefQuantity def="VoltageWaveform" prop="freq_hz" value={wf.freq_hz} onCommit={(v) => v !== null && setValue([...path, "freq_hz"], v, label)} />
        </>
      )}
    </div>
  );
}

/** 複数の CSV 波形 (1D の電極) */
export function WaveformList({ path }: { path: Path }) {
  const { t } = useTranslation();
  const list = (useDocument((s) => getIn(s.project, path)) as VoltageWaveform[] | undefined) ?? [];
  const label = t("widgets.waveform");
  const { picker, open, error } = useCsvPicker((w) => setValue(path, [...list, w], label));
  return (
    <div className="subsection">
      {picker}
      {list.map((w, i) => (
        <ListItem key={i} title={`CSV ${i + 1} · ${t("widgets.csvPoints", { n: w.phase.length })}`} onRemove={() => setValue(path, list.filter((_, j) => j !== i), label)}>
          <DefQuantity def="VoltageWaveform" prop="freq_hz" value={w.freq_hz} onCommit={(v) => v !== null && setValue([...path, i, "freq_hz"], v, label)} />
        </ListItem>
      ))}
      <button type="button" className="button small" onClick={open}>
        {t("widgets.csvImport")}
      </button>
      {error && <Hint tone="error">{error}</Hint>}
    </div>
  );
}
