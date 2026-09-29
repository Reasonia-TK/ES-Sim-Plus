// 折れ線グラフ (uPlot)。系列・対数軸・右の縦軸 (2 軸)・縦の印 (マーカー)・カーソルの値表示。大きさは入れ物に合わせる。

import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { usePrefs } from "../prefs/prefs";

export interface LineSeries {
  label: string;
  values: (number | null)[];
  color: string;
  dash?: number[];
  width?: number;
  /** 右の縦軸に描く */
  right?: boolean;
}

export interface LineChartProps {
  x: number[];
  series: LineSeries[];
  xLabel?: string;
  yLabel?: string;
  /** 右の縦軸の名前 (right の系列があるとき) */
  yRightLabel?: string;
  logY?: boolean;
  height?: number;
  /** 縦の印 (x の値) */
  markers?: { x: number; color: string }[];
  ariaLabel?: string;
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

export function LineChart({ x, series, xLabel, yLabel, yRightLabel, logY, height = 180, markers, ariaLabel }: LineChartProps) {
  const wrap = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const theme = usePrefs((s) => s.theme);

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
    });
    const markerPlugin: uPlot.Plugin = {
      hooks: {
        draw: [
          (u) => {
            const ctx = u.ctx;
            for (const m of markers ?? []) {
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
      scales: { x: { time: false }, y: logY ? { distr: 3 } : {}, ...(dual ? { y2: {} } : {}) },
      axes,
      series: [
        {},
        ...series.map((s) => ({ label: s.label, stroke: s.color, width: s.width ?? 1.5, dash: s.dash, spanGaps: false, scale: s.right ? "y2" : "y" })),
      ],
      legend: { show: series.length > 1 },
      cursor: { drag: { x: true, y: false } },
      plugins: [markerPlugin],
    };
    const data = [x, ...series.map((s) => s.values.map((v) => (logY && !s.right && v !== null && v <= 0 ? null : v)))] as uPlot.AlignedData;
    let u: uPlot | null = null;
    try {
      u = new uPlot(opts, data, el);
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
  }, [x, series, xLabel, yLabel, yRightLabel, logY, height, markers, theme]);

  return <div ref={wrap} className="line-chart" role="img" aria-label={ariaLabel ?? yLabel} />;
}
