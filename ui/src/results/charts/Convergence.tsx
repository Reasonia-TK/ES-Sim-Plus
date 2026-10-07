// 収束の判定 (prompts/137): 判定の周期ごとの「変化 D・残りの変化 R」の推移のグラフと、数値サマリの行 (結果・実行中)。
// D は雑音を差し引いた周期平均の相対変化、R は変化の減り方から見積もった残りの変化 (backend の convergence.py)。
// 全ての量で max(D, R) が閾値以下の周期が hold 回続くと収束。PIC は分解能 (見逃しうる遅い傾きで、調べた後ろ半分の
// 長さだけ走ったときの変化、CV-d) も閾値以下になるまで合格にしない。

import type { TFunction } from "i18next";
import type { ReactNode } from "react";
import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import type { Project } from "../../model/project";
import { ChartCard } from "../../plots/ChartCard";
import { columns, saveCsv } from "../../plots/csv";
import { LineChart, type LineSeries } from "../../plots/LineChart";
import { formatSi } from "../../util/format";
import type { ConvergenceFrame, ConvergenceResult } from "../types";
import { circuitLabel } from "./SelfBias";

const COLORS = ["#4da3ff", "#ff8a5c", "#6fd08c", "#b070f0", "#f0c040", "#40c0c0"];
/** グラフで無限大 (残りの変化を見積もれない) を描く高さ (相対 10 = 1000%) */
const INF_PLOT = 10;

type Row = [ReactNode, ReactNode];

/** 量の表示名 */
export function convQuantityLabel(name: string, project: Project | null, t: TFunction): string {
  if (name === "phi" || name === "n_e" || name === "N_e" || name === "N_i") return t(`conv.qty.${name}`);
  if (name.startsWith("V_dc:")) return t("conv.qty.vdc", { label: circuitLabel(name.slice(5), project, t) });
  return name;
}

/** 相対値を % で (小さい値は指数表記)。null は inf (無限大) か "-" */
export function convPct(v: number | null | undefined, inf = false): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return inf ? "∞" : "-";
  if (v === 0) return "0%";
  const p = v * 100;
  // 100% 以上は整数で (toPrecision(2) は 400 を "4.0e+2" と書く)
  if (Math.abs(p) >= 99.5) return `${Math.round(p)}%`;
  return Math.abs(p) >= 0.01 ? `${p.toPrecision(2)}%` : `${p.toExponential(1)}%`;
}

/** 判定 i で max(D, R) がいちばん大きい量 (判定していなければ null)。R が null の量は無限大とみなす */
export function convWorstAt(c: ConvergenceResult, i: number): { name: string; d: number; r: number | null } | null {
  let best: { name: string; d: number; r: number | null } | null = null;
  let bestV = -1;
  for (const name of Object.keys(c.d)) {
    const d = c.d[name]?.[i];
    if (d === null || d === undefined) continue;
    const r = c.r[name]?.[i] ?? null;
    const v = r === null ? Infinity : Math.max(d, r);
    if (v > bestV) {
      bestV = v;
      best = { name, d, r };
    }
  }
  return best;
}

/** 判定 i で分解能がいちばん粗い量 (PIC、CV-d)。分解能が無ければ null */
export function convResAt(c: ConvergenceResult, i: number): { name: string; v: number } | null {
  let best: { name: string; v: number } | null = null;
  for (const [name, lst] of Object.entries(c.res ?? {})) {
    const v = lst[i];
    if (v === null || v === undefined) continue;
    if (!best || v > best.v) best = { name, v };
  }
  return best;
}

function periodText(c: ConvergenceResult, t: TFunction): string {
  const s = formatSi(c.period_s, "s", 3);
  return c.basis === "rf" && c.rf_periods ? t("conv.periodRf", { n: c.rf_periods, t: s }) : s;
}

/** 数値サマリの行 (結果): 状態・収束した周期・止めたか・いちばん遠い量・閾値と判定の周期 */
export function convergenceResultRows(c: ConvergenceResult, project: Project | null, t: TFunction): Row[] {
  const rows: Row[] = [];
  const n = c.status.length;
  const last = n - 1;
  const lastStatus = n ? c.status[last] : "warming";
  if (c.converged && c.converged_period !== null && c.converged_t !== null) {
    rows.push([t("summary.convState"), t("summary.convConverged", { period: c.converged_period + 1, t: formatSi(c.converged_t, "s", 3) })]);
    if (!c.now_passing && n) rows.push([t("summary.convNow"), t(`conv.status.${lastStatus}`)]);
  } else {
    rows.push([t("summary.convState"), t(`conv.status.${lastStatus}`)]);
  }
  if (c.stopped && c.converged_step !== null) rows.push([t("summary.convStopped"), t("summary.convStoppedValue", { step: c.converged_step })]);
  const w = n && (lastStatus === "pass" || lastStatus === "fail" || lastStatus === "resolving") ? convWorstAt(c, last) : null;
  if (w) {
    const kind = c.r_kind?.[w.name]?.[last];
    rows.push([
      t("summary.convWorst"),
      t("summary.convWorstValue", {
        name: convQuantityLabel(w.name, project, t),
        d: convPct(w.d),
        r: convPct(w.r, true),
        kind: kind ? t(`conv.kind.${kind}`) : "-",
      }),
    ]);
  }
  const res = n ? convResAt(c, last) : null;
  if (res) rows.push([t("summary.convRes"), t("summary.convResValue", { name: convQuantityLabel(res.name, project, t), v: convPct(res.v, true) })]);
  rows.push([t("summary.convChecks"), n]);
  rows.push([t("summary.convSettings"), t("summary.convSettingsValue", { tol: convPct(c.tol), period: periodText(c, t), hold: c.hold })]);
  return rows;
}

/** 実行中の数値の行: 状態 (収束したらその時刻)・いちばん遠い量・判定した周期 */
export function convergenceFrameRows(f: ConvergenceFrame, project: Project | null, t: TFunction): Row[] {
  const rows: Row[] = [
    [t("summary.convState"), f.converged && f.converged_t !== null ? t("summary.convConvergedAt", { t: formatSi(f.converged_t, "s", 3) }) : t(`conv.status.${f.status}`)],
  ];
  if (f.worst_name) rows.push([t("summary.convWorst"), t("summary.convWorstLive", { name: convQuantityLabel(f.worst_name, project, t), v: convPct(f.worst, true) })]);
  if (f.res_name) rows.push([t("summary.convRes"), t("summary.convResValue", { name: convQuantityLabel(f.res_name, project, t), v: convPct(f.res, true) })]);
  rows.push([t("summary.convChecks"), f.checks]);
  return rows;
}

function convChart(c: ConvergenceResult, project: Project | null, t: TFunction) {
  const n = c.period.length;
  if (!n) return null;
  const names = Object.keys(c.d);
  // 対数軸: 0 (変化なし) と小さすぎる値は閾値の 1/10⁴ で描く
  const floor = c.tol * 1e-4;
  const series: LineSeries[] = names.map((name, k) => ({
    label: convQuantityLabel(name, project, t),
    values: c.d[name].map((d, i) => {
      if (d === null) return null;
      const r = c.r[name]?.[i] ?? null;
      const v = r === null ? INF_PLOT : Math.max(d, r);
      return Math.min(INF_PLOT, Math.max(floor, v));
    }),
    color: COLORS[k % COLORS.length],
  }));
  series.push({ label: t("charts.convTol"), values: c.period.map(() => c.tol), color: "#999", dash: [6, 4] });
  // PIC の分解能 (いちばん粗い量、CV-d)。閾値を下回るまで合格にしない
  const res = c.period.map((_, i) => convResAt(c, i)?.v ?? null);
  if (res.some((v) => v !== null)) {
    series.push({
      label: t("charts.convRes"),
      values: res.map((v) => (v === null ? null : Math.min(INF_PLOT, Math.max(floor, v)))),
      color: "#999",
      dash: [2, 3],
    });
  }
  return { x: c.period.map((p) => p + 1), series, names, floor };
}

/** 判定のグラフ: 量ごとの max(D, R) (対数) と閾値 (破線)、収束した周期 (縦線) */
export function ConvergenceCard({ convergence: c, project, csvPrefix }: { convergence: ConvergenceResult; project: Project | null; csvPrefix: string }) {
  const { t } = useTranslation();
  const chart = useMemo(() => convChart(c, project, t), [c, project, t]);
  if (!chart) return null;
  const csv = () => {
    const header = ["period", "t_s", "step", "status", "window", ...chart.names.flatMap((n) => [`d_${n}`, `r_${n}`, `noise_${n}`, `r_kind_${n}`, `res_${n}`])];
    const cols = chart.names.flatMap((n) => [c.d[n], c.r[n], c.noise[n], c.r_kind?.[n] ?? [], c.res?.[n] ?? []]);
    void saveCsv(`${csvPrefix}_convergence.csv`, header, columns(chart.x, c.t, c.step, c.status, c.window, ...cols));
  };
  const markers = c.converged && c.converged_period !== null ? [{ x: c.converged_period + 1, color: "#6fd08c" }] : [];
  return (
    <ChartCard
      title={t("charts.convTitle")}
      tools={
        <button type="button" className="button small" onClick={csv}>
          {t("charts.csv")}
        </button>
      }
      note={t("charts.convNote")}
    >
      <LineChart
        x={chart.x}
        series={chart.series}
        xLabel={t("charts.convX")}
        yLabel={t("charts.convY")}
        logY
        yRange={[chart.floor, INF_PLOT]}
        markers={markers}
        height={170}
      />
    </ChartCard>
  );
}
