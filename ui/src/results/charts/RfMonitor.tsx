// RF 波形モニタ: 時間で変わる電極の電圧 V(t) を 1 周期 (最低周波数) ぶん描き、今のフレームの位相に縦線を引く
// (v1 RfPhaseMonitor・Pic1dRfMonitor)。電極は実行したときの設定 (実行の入力) から取る。

import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { LineChart } from "../../plots/LineChart";
import { evalVoltage, pickTimeUnit, rfComponents, waveformFreqs, type VoltageRf, type VoltageWaveform } from "../../util/waveform";

export interface RfElectrode {
  label: string;
  dc: number;
  rf: VoltageRf[];
  waveforms: VoltageWaveform[];
}

const COLORS = ["#4da3ff", "#ffb84d", "#6fd08c", "#b070f0"];
const MARKER = "#ff5c5c";
const N = 300;
const MAX_SHOWN = 4;

/** 1D の電極 (pic1d・fluid1d の left/right) */
export function electrode1d(e: unknown, label: string): RfElectrode | null {
  if (!e || typeof e !== "object") return null;
  const o = e as { v_dc?: number; voltage_rf?: unknown; waveforms?: VoltageWaveform[] };
  return { label, dc: o.v_dc ?? 0, rf: rfComponents(o.voltage_rf), waveforms: o.waveforms ?? [] };
}

export function RfMonitor({ title, electrodes, t, height = 110 }: { title: string; electrodes: RfElectrode[]; t: number | null; height?: number }) {
  const { t: tr } = useTranslation();
  const live = electrodes.filter((e) => waveformFreqs(e.rf, e.waveforms).length > 0);
  const freqs = live.flatMap((e) => waveformFreqs(e.rf, e.waveforms));
  const f0 = freqs.length ? Math.min(...freqs) : 0;
  const key = JSON.stringify(live);
  const data = useMemo(() => {
    if (f0 <= 0) return null;
    const period = 1 / f0;
    const unit = pickTimeUnit(period);
    const x = Array.from({ length: N }, (_, i) => ((period * i) / (N - 1)) * unit.scale);
    const shown = live.slice(0, MAX_SHOWN);
    const series = shown.map((e, k) => ({
      label: e.label,
      color: COLORS[k % COLORS.length],
      values: Array.from({ length: N }, (_, i) => evalVoltage(e.dc, e.rf, e.waveforms, (period * i) / (N - 1))),
    }));
    return { period, unit, x, series };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, f0]);
  if (!data) return null;
  const frac = t === null ? null : t / data.period - Math.floor(t / data.period);
  const markers = frac === null ? undefined : [{ x: frac * data.period * data.unit.scale, color: MARKER }];
  const more = live.length - MAX_SHOWN;
  return (
    <div className="rf-monitor">
      <div className="rf-monitor-head">
        <strong>{title}</strong>
        {frac !== null && t !== null && (
          <span className="muted mono">
            {tr("charts.rfPhase", { pct: (frac * 100).toFixed(0) })} (t = {(t * data.unit.scale).toFixed(2)} {data.unit.label})
          </span>
        )}
        {more > 0 && <span className="muted small">{tr("charts.rfMore", { n: more })}</span>}
      </div>
      <LineChart x={data.x} series={data.series} xLabel={`t [${data.unit.label}]`} yLabel="V [V]" height={height} markers={markers} />
    </div>
  );
}
