// 阻止コンデンサの自己バイアス (prompts/134): RF 1 周期ごとの V_dc・|V1| の推移と最後の 1 周期の電極の電位の
// グラフ、数値サマリの行、RF 波形モニタに出す実際の電極の電位 (結果は最後の 1 周期の波形、実行中は電源の波形を
// 直近の自己バイアスだけずらしたもの)。

import type { TFunction } from "i18next";
import type { ReactNode } from "react";
import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import type { Project } from "../../model/project";
import { ChartCard } from "../../plots/ChartCard";
import { columns, saveCsv } from "../../plots/csv";
import { LineChart, type LineSeries } from "../../plots/LineChart";
import { edgeLabel } from "../../tree/treeModel";
import { formatNumber, formatSi, maxOf } from "../../util/format";
import { pickTimeUnit } from "../../util/waveform";
import type { CircuitElectrode, CircuitFrame, CircuitResult } from "../types";
import type { RfElectrode } from "./RfMonitor";

const COLORS = ["#ff8a5c", "#4da3ff", "#6fd08c", "#b070f0"];

/** 結果と最新のフレームの回路 (どちらも無いことがある) */
export interface CircuitSrc {
  result?: CircuitResult | null;
  frame?: CircuitFrame[] | null;
}

/** 回路の電極の表示名 (2D の外周の辺 "edge3"・"edge1+edge3" は辺の表示名、1D の left/right は左右の電極) */
export function circuitLabel(label: string, project: Project | null, t: TFunction): string {
  if (label === "left") return t("studyCommon.leftElectrode");
  if (label === "right") return t("studyCommon.rightElectrode");
  const edges = label.split("+").map((s) => /^edge(\d+)$/.exec(s));
  if (project && edges.every((m) => m !== null)) return edges.map((m) => edgeLabel(project, Number(m![1]), t)).join(" + ");
  return label;
}

/** 回路の電極 label が、RF 波形モニタの電極 key (外周の辺 "edge3"・導体の id・"left") を含むか */
function matches(label: string, key: string): boolean {
  return label === key || label.split("+").includes(key);
}

/**
 * RF 波形モニタの電極に、阻止コンデンサの実際の電極の電位を付ける: 結果があれば最後の 1 周期の波形、実行中は
 * 電源の波形を直近の自己バイアスだけずらす (ずれ = V_dc − 電源の直流分)
 */
export function attachCircuit(el: RfElectrode, key: string, src: CircuitSrc | undefined, t: TFunction): RfElectrode {
  if (!src) return el;
  const res = src.result?.electrodes.find((c) => matches(c.label, key));
  if (res?.last_period && res.last_period.t.length > 1) return { ...el, label: t("charts.rfMeasured", { label: el.label }), measured: res.last_period };
  const fr = src.frame?.find((c) => matches(c.label, key));
  if (fr && typeof fr.v_dc === "number") return { ...el, label: t("charts.rfShifted", { label: el.label }), shift: fr.v_dc - el.dc };
  return el;
}

/** 量の単位 (結果の units の "F/m^2" などを表示用に) */
function unitLabel(u: string): string {
  return u.replace("^2", "²");
}

type Row = [ReactNode, ReactNode];

/** 数値サマリの行: 電極ごとの最後の周期の V_dc・|V1|・比・正味の電流と、電極の容量・阻止コンデンサ・RF の分圧 */
export function circuitResultRows(c: CircuitResult, project: Project | null, t: TFunction): Row[] {
  const rows: Row[] = [];
  const capU = unitLabel(c.units.capacitance);
  const curU = unitLabel(c.units.current);
  for (const e of c.electrodes) {
    const label = circuitLabel(e.label, project, t);
    const n = e.v_dc.length;
    const vdc = n ? e.v_dc[n - 1] : null;
    const v1 = n ? e.v1[n - 1] : null;
    rows.push([t("summary.selfBias", { label }), vdc === null ? (e.v_e === null ? "-" : `${formatNumber(e.v_e)} V`) : `${vdc.toFixed(2)} V`]);
    if (v1 !== null) rows.push([t("summary.selfBiasV1", { label }), `${v1.toFixed(2)} V`]);
    if (vdc !== null && v1) rows.push([t("summary.selfBiasRatio", { label }), (vdc / v1).toFixed(3)]);
    if (n) rows.push([t("summary.selfBiasCurrent", { label }), formatSi(e.i_dc[n - 1], curU, 3)]);
    rows.push([
      t("summary.selfBiasCap", { label }),
      `${formatSi(e.c_self, capU, 3)} / ${formatSi(e.capacitance, capU, 3)} (C_b/(C_b+C) = ${(100 * e.rf_division).toFixed(2)}%)`,
    ]);
  }
  return rows;
}

/** 実行中の数値の行: 電極の今の電位と、最後に閉じた周期の自己バイアス */
export function circuitFrameRows(frame: CircuitFrame[], project: Project | null, t: TFunction): Row[] {
  const rows: Row[] = [];
  for (const e of frame) {
    const label = circuitLabel(e.label, project, t);
    rows.push([t("summary.electrodeV", { label }), e.v_e === null ? "-" : `${e.v_e.toFixed(2)} V`]);
    if (typeof e.v_dc === "number") rows.push([t("summary.selfBias", { label }), `${e.v_dc.toFixed(2)} V`]);
  }
  return rows;
}

function biasChart(es: CircuitElectrode[], project: Project | null, t: TFunction) {
  const shown = es.filter((e) => e.t.length > 0);
  if (!shown.length) return null;
  const tt = shown[0].t;
  const unit = pickTimeUnit(maxOf(tt));
  const series: LineSeries[] = [];
  shown.forEach((e, k) => {
    const label = circuitLabel(e.label, project, t);
    const color = COLORS[k % COLORS.length];
    series.push({ label: t("charts.selfBiasVdc", { label }), values: e.v_dc, color });
    series.push({ label: t("charts.selfBiasV1", { label }), values: e.v1, color, dash: [6, 4] });
  });
  return { x: tt.map((v) => v * unit.scale), unit: unit.label, series, shown };
}

function waveChart(es: CircuitElectrode[], project: Project | null, t: TFunction) {
  const shown = es.filter((e) => e.last_period && e.last_period.t.length > 1);
  if (!shown.length) return null;
  // 電極ごとに標本の時刻が同じ (同じ回路のサブステップ) なので最初の電極の時刻を横軸にする
  const tt = shown[0].last_period!.t;
  const unit = pickTimeUnit(maxOf(tt));
  const series: LineSeries[] = shown.map((e, k) => ({
    label: circuitLabel(e.label, project, t),
    values: e.last_period!.t.length === tt.length ? e.last_period!.v : tt.map(() => null),
    color: COLORS[k % COLORS.length],
  }));
  return { x: tt.map((v) => v * unit.scale), unit: unit.label, series };
}

/** 自己バイアスのグラフ: RF 1 周期ごとの V_dc (実線)・|V1| (破線) と、最後の 1 周期の電極の電位 */
export function SelfBiasCard({ circuit, project, csvPrefix }: { circuit: CircuitResult; project: Project | null; csvPrefix: string }) {
  const { t } = useTranslation();
  const bias = useMemo(() => biasChart(circuit.electrodes, project, t), [circuit, project, t]);
  const wave = useMemo(() => waveChart(circuit.electrodes, project, t), [circuit, project, t]);
  if (!bias && !wave) return null;
  const csv = () => {
    if (!bias) return;
    const header = ["t_s", ...bias.shown.flatMap((e) => [`v_dc_${e.label}_V`, `v1_${e.label}_V`, `i_dc_${e.label}`])];
    const cols = bias.shown.flatMap((e) => [e.v_dc, e.v1, e.i_dc]);
    void saveCsv(`${csvPrefix}_self_bias.csv`, header, columns(bias.shown[0].t, ...cols));
  };
  return (
    <ChartCard
      title={t("charts.selfBiasTitle")}
      tools={
        bias && (
          <button type="button" className="button small" onClick={csv}>
            {t("charts.csv")}
          </button>
        )
      }
      note={t("charts.selfBiasNote")}
    >
      {bias && <LineChart x={bias.x} series={bias.series} xLabel={`t [${bias.unit}]`} yLabel="V [V]" height={150} />}
      {wave && <LineChart x={wave.x} series={wave.series} xLabel={`t [${wave.unit}]`} yLabel="V_e [V]" height={120} />}
    </ChartCard>
  );
}
