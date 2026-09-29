// 電極電圧のプレビュー: V(t) = DC + ΣRF + CSV を最低周波数の 2 周期ぶん (600 点) 描く (v1 と同じ)。

import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { LineChart } from "../../plots/LineChart";
import { evalVoltage, pickTimeUnit, waveformFreqs, type VoltageRf, type VoltageWaveform } from "../../util/waveform";

const N = 600;

export function VoltagePreview({ dc, rf, waveforms }: { dc: number; rf: VoltageRf[]; waveforms: VoltageWaveform[] }) {
  const { t } = useTranslation();
  const freqs = waveformFreqs(rf, waveforms);
  const fMin = freqs.length ? Math.min(...freqs) : 0;
  // 配列は描画のたびに作り直されるので、中身が変わったときだけ計算し直す (グラフの作り直しを避ける)
  const key = JSON.stringify([dc, rf, waveforms]);
  const data = useMemo(() => {
    if (fMin <= 0) return null;
    const tMax = 2 / fMin;
    const unit = pickTimeUnit(tMax);
    const x = new Array<number>(N);
    const v = new Array<number>(N);
    for (let i = 0; i < N; i++) {
      const ti = (tMax * i) / (N - 1);
      x[i] = ti * unit.scale;
      v[i] = evalVoltage(dc, rf, waveforms, ti);
    }
    return { x, series: [{ label: "V", values: v, color: "#4da3ff" }], unit: unit.label };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, fMin]);
  if (!data) return null;
  return (
    <div className="subsection">
      <div className="muted small">{t("widgets.voltagePreview")}</div>
      <LineChart x={data.x} series={data.series} xLabel={`t [${data.unit}]`} yLabel="V [V]" height={150} />
    </div>
  );
}
