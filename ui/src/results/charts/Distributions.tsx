// エネルギー分布のグラフ: EEDF/EEPF (領域ごと、PIC・PIC 1D) と壁 IEDF (左右の壁、PIC 1D・流体 1D)。
// v1 の EedfChart・WallIedfSection と同じ中身 (重みの無い系列は描かず凡例に「データなし」)。CSV も v1 と同じ列。

import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { ChartCard } from "../../plots/ChartCard";
import { columns, saveCsv } from "../../plots/csv";
import { LineChart, type LineSeries } from "../../plots/LineChart";
import { Toggle } from "../../forms/SchemaField";
import { useChartPref } from "../resultsView";
import type { EedfResult, WallIedfSide } from "../types";
import { EEDF_COLORS, mergeSeriesX, WALL_COLORS } from "./common";

type Dist = { e_centers: number[]; f: (number | null)[]; total_weight: number };

/** EEPF = f/√E (E > 0) */
export function eepf(d: Dist): (number | null)[] {
  return d.f.map((f, i) => (f !== null && d.e_centers[i] > 0 ? f / Math.sqrt(d.e_centers[i]) : null));
}

function distChart(list: { label: string; color: string; d: Dist }[], mode: "eedf" | "eepf") {
  const drawn = list.filter((s) => s.d.total_weight > 0 && s.d.e_centers.length > 0);
  const merged = mergeSeriesX(drawn.map((s) => ({ x: s.d.e_centers, y: mode === "eepf" ? eepf(s.d) : s.d.f })));
  const series: LineSeries[] = drawn.map((s, i) => ({ label: s.label, color: s.color, values: merged.ys[i], width: 1.6 }));
  return { x: merged.x, series, empty: drawn.length === 0 };
}

export function EedfCard({ eedf, csvPrefix, prefKey, defaultMode }: { eedf: EedfResult[]; csvPrefix: string; prefKey: string; defaultMode: "eedf" | "eepf" }) {
  const { t } = useTranslation();
  const [mode, setMode] = useChartPref<string>(`${prefKey}.mode`, defaultMode);
  const [log, setLog] = useChartPref(`${prefKey}.log`, true);
  const m = mode === "eepf" ? "eepf" : "eedf";
  const named = eedf.map((e, i) => ({ e, name: e.label || `E${i + 1}`, color: EEDF_COLORS[i % EEDF_COLORS.length] }));
  const chart = useMemo(() => distChart(named.map((n) => ({ label: n.name, color: n.color, d: n.e })), m), [eedf, m]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <ChartCard
      title="EEDF / EEPF"
      tools={
        <>
          <select className="input" value={m} aria-label={t("charts.distMode")} onChange={(e) => setMode(e.target.value)}>
            <option value="eedf">{t("charts.eedfMode")}</option>
            <option value="eepf">{t("charts.eepfMode")}</option>
          </select>
          <Toggle checked={log} onChange={setLog} label={t("charts.logY")} />
        </>
      }
    >
      {chart.empty ? (
        <p className="hint">{t("charts.noDistData")}</p>
      ) : (
        <LineChart x={chart.x} series={chart.series} xLabel="E [eV]" yLabel={m === "eepf" ? "f(E)/√E [eV^-1.5]" : "f(E) [eV^-1]"} logY={log} xRange={[0, null]} height={170} legend />
      )}
      <div className="dist-rows">
        {named.map(({ e, name, color }) => (
          <div key={name} className="dist-row">
            <span className="dist-tag" style={{ color }}>
              {name}
            </span>
            {e.total_weight > 0 ? (
              <>
                <span className="mono">T_eff {e.t_eff_ev.toFixed(2)} eV</span>
                <span className="mono">⟨E⟩ {e.mean_energy_ev.toFixed(2)} eV</span>
                <span className="mono">overflow {(e.overflow_frac * 100).toFixed(2)}%</span>
                <button
                  type="button"
                  className="button small"
                  onClick={() => void saveCsv(`${csvPrefix ? `${csvPrefix}_` : ""}eedf_eepf_${name}.csv`, ["E_eV", "f_eedf_ev-1", "f_eepf_ev-1.5"], columns(e.e_centers, e.f, eepf(e)))}
                >
                  {t("charts.csv")}
                </button>
              </>
            ) : (
              <span className="muted">{t("charts.noElectrons")}</span>
            )}
          </div>
        ))}
      </div>
    </ChartCard>
  );
}

export function WallIedfCard({ iedf, csvPrefix, prefKey, hint }: { iedf: { left: WallIedfSide; right: WallIedfSide }; csvPrefix: string; prefKey: string; hint?: string }) {
  const { t } = useTranslation();
  const [log, setLog] = useChartPref(`${prefKey}.log`, true);
  const sides = [
    { key: "left" as const, label: t("charts.leftWall"), color: WALL_COLORS.left, d: iedf.left },
    { key: "right" as const, label: t("charts.rightWall"), color: WALL_COLORS.right, d: iedf.right },
  ];
  const chart = useMemo(() => distChart(sides.map((s) => ({ label: s.label, color: s.color, d: s.d })), "eedf"), [iedf, t]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <ChartCard title={t("charts.wallIedf")} tools={<Toggle checked={log} onChange={setLog} label={t("charts.logY")} />} note={hint}>
      {chart.empty ? (
        <p className="hint">{t("charts.noDistData")}</p>
      ) : (
        <LineChart x={chart.x} series={chart.series} xLabel="E [eV]" yLabel="f(E) [eV^-1]" logY={log} xRange={[0, null]} height={150} legend />
      )}
      <div className="dist-rows">
        {sides.map((s) => (
          <div key={s.key} className="dist-row">
            <span className="dist-tag" style={{ color: s.color }}>
              {s.label}
            </span>
            {s.d.total_weight > 0 ? (
              <>
                <span className="mono">⟨E⟩ {s.d.mean_energy_ev.toFixed(2)} eV</span>
                <span className="mono">
                  {t("charts.weight")} {s.d.total_weight.toExponential(3)}
                </span>
                <button type="button" className="button small" onClick={() => void saveCsv(`${csvPrefix}_wall_iedf_${s.key}.csv`, ["E_eV", "f_ev-1"], columns(s.d.e_centers, s.d.f))}>
                  {t("charts.csv")}
                </button>
              </>
            ) : (
              <span className="muted">{t("charts.noIons")}</span>
            )}
          </div>
        ))}
      </div>
    </ChartCard>
  );
}
