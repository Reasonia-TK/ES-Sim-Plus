// VHF 定在波と Boltzmann 係数の表のグラフ (v1 TlPlotView・BoltzSection のグラフの部分)。

import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { Toggle } from "../../forms/SchemaField";
import { ChartCard } from "../../plots/ChartCard";
import { columns, saveCsv } from "../../plots/csv";
import { LineChart, type LineSeries } from "../../plots/LineChart";
import { formatNumber } from "../../util/format";
import { pickTimeUnit } from "../../util/waveform";
import { useChartPref } from "../resultsView";
import type { BoltzTable, TlResult } from "../types";
import { EEDF_COLORS, useLength } from "./common";

// ---- VHF 定在波 ----

const PROBE_COLORS = { center: "#7ec8e3", mid: "#f2b880", edge: "#8ee6a9" };

export function TlCharts({ result }: { result: TlResult }) {
  const { t } = useTranslation();
  const len = useLength();
  const [allHarm, setAllHarm] = useChartPref("tl.allHarmonics", false);
  const [logHarm, setLogHarm] = useChartPref("tl.harmLog", false);
  const [logSpec, setLogSpec] = useChartPref("tl.specLog", true);
  const r = useMemo(() => result.r.map(len.of), [result, len.unit]); // eslint-disable-line react-hooks/exhaustive-deps
  const nHarm = Number(result.settings?.n_harm ?? result.harmonics.v.length - 1);
  const shown = allHarm ? nHarm : Math.min(3, nHarm);
  const harm = useMemo<LineSeries[]>(
    () =>
      Array.from({ length: Math.max(0, Math.min(shown, result.harmonics.v.length - 1)) }, (_, k) => ({
        label: `n=${k + 1}`,
        values: result.harmonics.v[k + 1],
        color: EEDF_COLORS[k % EEDF_COLORS.length],
      })),
    [result, shown],
  );
  const probes = (["center", "mid", "edge"] as const).map((k) => ({ key: k, label: t(`charts.tlProbe.${k}`), color: PROBE_COLORS[k] }));
  const vt = useMemo(() => {
    const tt = result.v_probe.t;
    if (tt.length < 2) return null;
    const t0 = tt[0];
    const unit = pickTimeUnit(tt[tt.length - 1] - t0);
    return {
      x: tt.map((v) => (v - t0) * unit.scale),
      unit: unit.label,
      series: probes.map((p) => ({ label: p.label, values: result.v_probe[p.key], color: p.color })) as LineSeries[],
    };
  }, [result, t]); // eslint-disable-line react-hooks/exhaustive-deps
  const f0 = Number(result.settings?.freq_hz ?? 0);
  const spec = useMemo(() => {
    const f = result.spectrum_probe.freq_hz;
    if (f.length < 2 || !(f0 > 0)) return null;
    return {
      x: f.map((v) => v / f0),
      series: probes.map((p) => ({ label: p.label, values: result.spectrum_probe[p.key], color: p.color })) as LineSeries[],
    };
  }, [result, f0, t]); // eslint-disable-line react-hooks/exhaustive-deps
  const rl = `r [${len.unit}]`;
  return (
    <>
      <ChartCard
        title={t("charts.tlHarmonics")}
        tools={
          <>
            <Toggle checked={allHarm} onChange={setAllHarm} label={t("charts.tlAllHarmonics")} />
            <Toggle checked={logHarm} onChange={setLogHarm} label={t("charts.log")} />
          </>
        }
      >
        <LineChart x={r} series={harm} xLabel={rl} yLabel="|V_n| [V]" logY={logHarm} height={180} />
      </ChartCard>
      <ChartCard title={t("charts.tlPower")}>
        <LineChart x={r} series={[{ label: "p(r)", values: result.power.p, color: "#f2b880" }]} xLabel={rl} yLabel="p [W/m^2]" height={150} legend={false} />
      </ChartCard>
      {vt && (
        <ChartCard title={t("charts.tlProbeV")}>
          <LineChart x={vt.x} series={vt.series} xLabel={`t [${vt.unit}]`} yLabel="V [V]" height={150} />
        </ChartCard>
      )}
      {spec && (
        <ChartCard title={t("charts.tlSpectrum")} tools={<Toggle checked={logSpec} onChange={setLogSpec} label={t("charts.log")} />}>
          <LineChart x={spec.x} series={spec.series} xLabel="f / f0" yLabel={t("charts.amplitudeV")} logY={logSpec} height={150} />
        </ChartCard>
      )}
    </>
  );
}

// ---- Boltzmann 係数の表 ----

export function BoltzCharts({ table }: { table: BoltzTable }) {
  const { t } = useTranslation();
  const [idxPref, setIdx] = useChartPref("boltz.eedfIndex", 0);
  const [mode, setMode] = useChartPref<string>("boltz.eedfMode", "eedf");
  const [log, setLog] = useChartPref("boltz.eedfLog", true);
  const n = table.en_td.length;
  const idx = Math.min(Math.max(0, idxPref), n - 1);
  const x = table.mean_energy_ev;
  const swarm1 = useMemo<LineSeries[]>(() => [{ label: "μ_e·N [1/(m·V·s)]", values: table.mobility_n, color: "#7ec8e3" }], [table]);
  const swarm2 = useMemo<LineSeries[]>(
    () => [
      { label: "k_ion [m^3/s]", values: table.k_ion, color: "#c792ea" },
      { label: "k_exc [m^3/s]", values: table.k_exc, color: "#f2b880" },
    ],
    [table],
  );
  const eps = table.eedf_eps_ev;
  const row = table.eedf[idx] ?? [];
  const eepf = useMemo(() => row.map((f, i) => (eps[i] > 0 && f !== null ? f / Math.sqrt(eps[i]) : null)), [row, eps]);
  const m = mode === "eepf" ? "eepf" : "eedf";
  if (n === 0) return null;
  const enLabel = (i: number) => `${formatNumber(table.en_td[i])} Td (ε̄=${table.mean_energy_ev[i].toFixed(3)} eV)`;
  return (
    <>
      <ChartCard title={t("charts.boltzSwarm")} note={t("charts.boltzSwarmHint")}>
        <LineChart x={x} series={swarm1} xLabel="ε̄ [eV]" yLabel="μ_e·N" logX logY height={150} />
        <LineChart x={x} series={swarm2} xLabel="ε̄ [eV]" yLabel="k [m^3/s]" logX logY height={150} />
      </ChartCard>
      <ChartCard
        title={t("charts.boltzEedf")}
        tools={
          <>
            <select className="input" value={idx} aria-label="E/N" onChange={(e) => setIdx(Number(e.target.value))}>
              {table.en_td.map((_, i) => (
                <option key={i} value={i}>
                  {enLabel(i)}
                </option>
              ))}
            </select>
            <select className="input" value={m} aria-label={t("charts.distMode")} onChange={(e) => setMode(e.target.value)}>
              <option value="eedf">EEDF f(ε) [eV^-1]</option>
              <option value="eepf">EEPF f(ε)/√ε [eV^-1.5]</option>
            </select>
            <Toggle checked={log} onChange={setLog} label={t("charts.logY")} />
            <button
              type="button"
              className="button small"
              onClick={() => void saveCsv(`boltz_eedf_eepf_${formatNumber(table.en_td[idx])}Td.csv`, ["E_eV", "f_eedf_ev-1", "f_eepf_ev-1.5"], columns(eps, row, eepf))}
            >
              {t("charts.csv")}
            </button>
          </>
        }
      >
        <LineChart x={eps} series={[{ label: m, values: m === "eepf" ? eepf : row, color: "#7ec8e3" }]} xLabel="ε [eV]" yLabel={m === "eepf" ? "f(ε)/√ε [eV^-1.5]" : "f(ε) [eV^-1]"} logY={log} height={160} legend={false} />
      </ChartCard>
    </>
  );
}
