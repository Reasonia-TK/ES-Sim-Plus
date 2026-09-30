// 散布図 (位相空間 x–v など): 点の組ごとの色で小さな四角を打つ。範囲は固定するか全部の点から。

import { useEffect, useRef, useState } from "react";
import { usePrefs } from "../prefs/prefs";
import { chartColors, CHART_FONT, drawAxes, drawGrid, extent, fitCanvas, scaleOf, type Rect } from "./axes";

export interface ScatterSet {
  label: string;
  x: ArrayLike<number>;
  y: ArrayLike<number>;
  color: string;
  size?: number;
}

export interface ScatterProps {
  sets: ScatterSet[];
  xLabel: string;
  yLabel: string;
  xRange?: [number, number];
  yRange?: [number, number];
  height?: number;
  ariaLabel?: string;
}

const MARGIN = { left: 60, right: 12, top: 10, bottom: 40 };

export function Scatter({ sets, xLabel, yLabel, xRange, yRange, height = 220, ariaLabel }: ScatterProps) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const theme = usePrefs((s) => s.theme);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    setWidth(el.clientWidth);
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const cv = canvas.current;
    if (!cv || width <= 0) return;
    const ctx = fitCanvas(cv, width, height);
    if (!ctx) return;
    const c = chartColors();
    const rect: Rect = { x: MARGIN.left, y: MARGIN.top, w: Math.max(10, width - MARGIN.left - MARGIN.right), h: Math.max(10, height - MARGIN.top - MARGIN.bottom) };
    const xr = xRange ?? extent(sets.map((s) => s.x), 0.02);
    const yr = yRange ?? extent(sets.map((s) => s.y), 0.05);
    drawGrid(ctx, rect, xr, yr, c);
    const px = scaleOf(xr, rect.x, rect.x + rect.w);
    const py = scaleOf(yr, rect.y + rect.h, rect.y);
    ctx.save();
    ctx.beginPath();
    ctx.rect(rect.x, rect.y, rect.w, rect.h);
    ctx.clip();
    for (const s of sets) {
      const size = s.size ?? 2;
      const h = size / 2;
      ctx.fillStyle = s.color;
      const n = Math.min(s.x.length, s.y.length);
      for (let i = 0; i < n; i++) {
        const x = s.x[i];
        const y = s.y[i];
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        ctx.fillRect(px(x) - h, py(y) - h, size, size);
      }
    }
    ctx.restore();
    drawAxes(ctx, rect, xr, yr, xLabel, yLabel, c);
    // 凡例
    if (sets.length > 1) {
      ctx.font = CHART_FONT;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      let lx = rect.x + 8;
      for (const s of sets) {
        ctx.fillStyle = s.color;
        ctx.fillRect(lx, rect.y + 8, 8, 8);
        ctx.fillStyle = c.text;
        ctx.fillText(s.label, lx + 12, rect.y + 12);
        lx += 24 + ctx.measureText(s.label).width;
      }
    }
  }, [sets, xLabel, yLabel, xRange, yRange, width, height, theme]);

  return (
    <div ref={wrap} className="scatter" role="img" aria-label={ariaLabel ?? yLabel}>
      <canvas ref={canvas} />
    </div>
  );
}
