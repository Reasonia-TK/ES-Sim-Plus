// 折れ線グラフ (uPlot)。系列 (線・階段・棒・点)・対数軸・固定の範囲・右の縦軸 (2 軸)・縦の印 (マーカー)・
// カーソルの値表示。大きさは入れ物に合わせる。データだけ変わったとき (ライブ・再生) は作り直さずに差し替える。

import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { usePrefs } from "../prefs/prefs";
import { tickText } from "./axes";

export type SeriesPaths = "line" | "step" | "bars" | "points";

export interface LineSeries {
  label: string;
  values: (number | null)[];
  color: string;
  dash?: number[];
  width?: number;
  /** 右の縦軸に描く */
  right?: boolean;
  /** 線 (既定)・階段・棒・点 */
  paths?: SeriesPaths;
  /** 塗り (棒・面の色) */
  fill?: string;
}

export interface LineChartProps {
  x: number[];
  series: LineSeries[];
  xLabel?: string;
  yLabel?: string;
  /** 右の縦軸の名前 (right の系列があるとき) */
  yRightLabel?: string;
  logY?: boolean;
  logX?: boolean;
  /** 軸の範囲を固定 (null の端は自動) */
  xRange?: [number | null, number | null];
  yRange?: [number | null, number | null];
  height?: number;
  /** 縦の印 (x の値) */
  markers?: { x: number; color: string }[];
  ariaLabel?: string;
  /** 凡例 (既定は系列が 2 つ以上のとき) */
  legend?: boolean;
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

let canvasOk: boolean | null = null;

/** 2D の canvas が使えるか (テストの jsdom では使えないので描かない) */
function canDraw(): boolean {
  if (canvasOk === null) {
    try {
      canvasOk = document.createElement("canvas").getContext("2d") !== null;
    } catch {
      canvasOk = false;
    }
  }
  return canvasOk;
}

function pathsOf(p: SeriesPaths | undefined): Partial<uPlot.Series> {
  switch (p) {
    case "step":
      return { paths: uPlot.paths.stepped!({ align: 1 }) };
    case "bars":
      return { paths: uPlot.paths.bars!({ size: [0.92, Infinity], gap: 0 }), points: { show: false } };
    case "points":
      return { paths: () => null, points: { show: true, size: 4 } };
    default:
      return {};
  }
}

function rangeOf(fixed: [number | null, number | null] | undefined, log: boolean): uPlot.Scale["range"] | undefined {
  if (!fixed) return undefined;
  const [lo, hi] = fixed;
  if (lo !== null && hi !== null) return [lo, hi];
  return (_u, dmin, dmax) => {
    const auto = uPlot.rangeNum(dmin, dmax, 0.1, true);
    return [lo ?? (log ? dmin : auto[0]), hi ?? (log ? dmax : auto[1])];
  };
}

export function LineChart({ x, series, xLabel, yLabel, yRightLabel, logY, logX, xRange, yRange, height = 180, markers, ariaLabel, legend }: LineChartProps) {
  const wrap = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const theme = usePrefs((s) => s.theme);
  const markersRef = useRef(markers);
  markersRef.current = markers;

  const buildData = (): uPlot.AlignedData =>
    [x, ...series.map((s) => (logY && !s.right ? s.values.map((v) => (v !== null && v <= 0 ? null : v)) : s.values))] as uPlot.AlignedData;
  const dataRef = useRef(buildData);
  dataRef.current = buildData;

  // 形 (系列の名前・色・軸) が変わったときだけ作り直す
  const shape = JSON.stringify([
    series.map((s) => [s.label, s.color, s.dash, s.width, s.right, s.paths, s.fill]),
    xLabel,
    yLabel,
    yRightLabel,
    logY,
    logX,
    xRange,
    yRange,
    height,
    legend,
  ]);

  useEffect(() => {
    const el = wrap.current;
    if (!el || !canDraw()) return;
    const text = cssVar("--text-muted");
    const grid = cssVar("--border");
    const dual = series.some((s) => s.right);
    const left = series.find((s) => !s.right);
    const right = series.find((s) => s.right);
    const axis = (label?: string, color?: string): uPlot.Axis => ({
      label,
      stroke: color ?? text,
      grid: { stroke: grid, width: 1 },
      ticks: { stroke: grid, width: 1 },
      // 大きい・小さい数は指数表記 (3e+08 のような桁区切りの長い数を出さない)
      values: (_u, splits) => splits.map((v) => (v === null || v === undefined ? "" : tickText(v))),
    });
    const markerPlugin: uPlot.Plugin = {
      hooks: {
        draw: [
          (u) => {
            const ctx = u.ctx;
            for (const m of markersRef.current ?? []) {
              const px = u.valToPos(m.x, "x", true);
              ctx.save();
              ctx.strokeStyle = m.color;
              ctx.setLineDash([4, 3]);
              ctx.lineWidth = 1.5 * devicePixelRatio;
              ctx.beginPath();
              ctx.moveTo(px, u.bbox.top);
              ctx.lineTo(px, u.bbox.top + u.bbox.height);
              ctx.stroke();
              ctx.restore();
            }
          },
        ],
      },
    };
    const axes: uPlot.Axis[] = [axis(xLabel), axis(yLabel, dual ? left?.color : undefined)];
    if (dual) axes.push({ ...axis(yRightLabel, right?.color), scale: "y2", side: 1, grid: { show: false } });
    const opts: uPlot.Options = {
      width: Math.max(el.clientWidth, 200),
      height,
      scales: {
        x: { time: false, ...(logX ? { distr: 3 } : {}), range: rangeOf(xRange, Boolean(logX)) },
        y: { ...(logY ? { distr: 3 } : {}), range: rangeOf(yRange, Boolean(logY)) },
        ...(dual ? { y2: {} } : {}),
      },
      axes,
      series: [
        { label: xLabel ?? "x" },
        ...series.map((s) => ({
          label: s.label,
          stroke: s.color,
          width: s.width ?? 1.5,
          dash: s.dash,
          fill: s.fill,
          spanGaps: false,
          scale: s.right ? "y2" : "y",
          ...pathsOf(s.paths),
        })),
      ],
      legend: { show: legend ?? series.length > 1 },
      cursor: { drag: { x: true, y: false } },
      plugins: [markerPlugin],
    };
    let u: uPlot | null = null;
    try {
      u = new uPlot(opts, dataRef.current(), el);
    } catch {
      return; // canvas の無い環境 (テスト) では描かない
    }
    plot.current = u;
    const ro = new ResizeObserver(() => u?.setSize({ width: Math.max(el.clientWidth, 200), height }));
    ro.observe(el);
    return () => {
      ro.disconnect();
      u?.destroy();
      plot.current = null;
    };
  }, [shape, theme]); // eslint-disable-line react-hooks/exhaustive-deps

  // データだけ変わったら差し替える
  useEffect(() => {
    plot.current?.setData(dataRef.current());
  }, [x, series, logY]);

  useEffect(() => {
    plot.current?.redraw(false);
  }, [markers]);

  return <div ref={wrap} className="line-chart" role="img" aria-label={ariaLabel ?? yLabel} />;
}
