// ヒートマップ (2D のヒストグラム・位相 × 位置の図など): 格子の値を配色で塗り、軸・カラーバー・カーソルの値を出す。
// values[iy][ix] (iy = 0 が下)。対数では 0 以下のセルは塗らない。

import { useEffect, useRef, useState } from "react";
import { colormapRgba, type ColormapKey } from "../graphics/colormaps";
import { usePrefs } from "../prefs/prefs";
import { chartColors, CHART_FONT, drawAxes, fitCanvas, niceTicks, scaleOf, tickText, type Rect } from "./axes";

export interface HeatmapProps {
  values: ArrayLike<number>[];
  /** 格子の端 (セルの外側の縁) */
  xRange: [number, number];
  yRange: [number, number];
  xLabel: string;
  yLabel: string;
  valueLabel?: string;
  log?: boolean;
  colormap?: ColormapKey;
  /** 値の範囲を固定 (再生でビンを送っても色を変えない) */
  range?: [number, number];
  /** 0 以下のセルは塗らない (ヒストグラムの空のビン) */
  skipZero?: boolean;
  height?: number;
  ariaLabel?: string;
}

const MARGIN = { left: 60, right: 84, top: 10, bottom: 40 };

/** 塗る範囲 (対数なら log10)。塗れる値が無ければ null */
export function heatRange(values: ArrayLike<number>[], log: boolean, fixed?: [number, number]): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  if (fixed) {
    [lo, hi] = fixed;
  } else {
    for (const row of values)
      for (let i = 0; i < row.length; i++) {
        const v = row[i];
        if (!Number.isFinite(v) || (log && v <= 0)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
  }
  if (log) {
    if (!(hi > 0)) return null;
    lo = Math.log10(lo > 0 ? lo : hi * 1e-6);
    hi = Math.log10(hi);
  }
  if (!(lo <= hi)) return null;
  return [lo, hi];
}

export function Heatmap({ values, xRange, yRange, xLabel, yLabel, valueLabel, log = false, colormap = "viridis", range, skipZero = false, height = 240, ariaLabel }: HeatmapProps) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const theme = usePrefs((s) => s.theme);
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState<{ x: number; y: number; v: number } | null>(null);
  const ny = values.length;
  const nx = ny ? values[0].length : 0;

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    setWidth(el.clientWidth);
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const rect: Rect = { x: MARGIN.left, y: MARGIN.top, w: Math.max(10, width - MARGIN.left - MARGIN.right), h: Math.max(10, height - MARGIN.top - MARGIN.bottom) };
  const vr = heatRange(values, log, range);

  useEffect(() => {
    const cv = canvas.current;
    if (!cv || width <= 0) return;
    const ctx = fitCanvas(cv, width, height);
    if (!ctx) return;
    const c = chartColors();
    const lut = colormapRgba(colormap);
    if (vr && nx > 0 && ny > 0) {
      const img = new ImageData(nx, ny);
      const span = vr[1] - vr[0];
      for (let iy = 0; iy < ny; iy++) {
        const row = values[iy];
        const oy = ny - 1 - iy;
        for (let ix = 0; ix < nx; ix++) {
          const v = row[ix];
          if (!Number.isFinite(v) || ((log || skipZero) && v <= 0)) continue;
          const x = log ? Math.log10(v) : v;
          const t = span > 0 ? Math.min(1, Math.max(0, (x - vr[0]) / span)) : 0.5;
          const k = Math.round(t * 255) * 4;
          const o = (oy * nx + ix) * 4;
          img.data[o] = lut[k];
          img.data[o + 1] = lut[k + 1];
          img.data[o + 2] = lut[k + 2];
          img.data[o + 3] = 255;
        }
      }
      const off = document.createElement("canvas");
      off.width = nx;
      off.height = ny;
      off.getContext("2d")?.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, rect.x, rect.y, rect.w, rect.h);
    }
    drawAxes(ctx, rect, xRange, yRange, xLabel, yLabel, c);
    // カラーバー
    const bx = rect.x + rect.w + 12;
    const bw = 12;
    const grad = ctx.createLinearGradient(0, rect.y + rect.h, 0, rect.y);
    for (let i = 0; i <= 16; i++) {
      const k = Math.round((i / 16) * 255) * 4;
      grad.addColorStop(i / 16, `rgb(${lut[k]},${lut[k + 1]},${lut[k + 2]})`);
    }
    ctx.fillStyle = grad;
    ctx.fillRect(bx, rect.y, bw, rect.h);
    ctx.strokeStyle = c.frame;
    ctx.strokeRect(bx + 0.5, rect.y + 0.5, bw - 1, rect.h - 1);
    if (vr) {
      ctx.font = CHART_FONT;
      ctx.fillStyle = c.text;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      const py = scaleOf(vr, rect.y + rect.h, rect.y);
      const ticks = log ? niceTicks(Math.ceil(vr[0]), Math.floor(vr[1]), 4).filter(Number.isInteger) : niceTicks(vr[0], vr[1], 4);
      for (const tv of ticks.length ? ticks : [vr[0]]) {
        const y = Math.round(py(tv)) + 0.5;
        ctx.fillText(log ? `1e${tv}` : tickText(tv), bx + bw + 4, y);
      }
    }
    if (valueLabel) {
      ctx.save();
      ctx.font = CHART_FONT;
      ctx.fillStyle = c.text;
      ctx.translate(width - 4, rect.y + rect.h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(valueLabel + (log ? " (log)" : ""), 0, 0);
      ctx.restore();
    }
  }, [values, nx, ny, xRange, yRange, xLabel, yLabel, valueLabel, log, skipZero, colormap, vr?.[0], vr?.[1], width, height, theme]); // eslint-disable-line react-hooks/exhaustive-deps

  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const b = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - b.left;
    const py = e.clientY - b.top;
    if (px < rect.x || px > rect.x + rect.w || py < rect.y || py > rect.y + rect.h || nx === 0) {
      setHover(null);
      return;
    }
    const fx = (px - rect.x) / rect.w;
    const fy = 1 - (py - rect.y) / rect.h;
    const ix = Math.min(nx - 1, Math.floor(fx * nx));
    const iy = Math.min(ny - 1, Math.floor(fy * ny));
    const x = xRange[0] + ((ix + 0.5) / nx) * (xRange[1] - xRange[0]);
    const y = yRange[0] + ((iy + 0.5) / ny) * (yRange[1] - yRange[0]);
    setHover({ x, y, v: values[iy][ix] });
  };

  return (
    <div ref={wrap} className="heatmap" role="img" aria-label={ariaLabel ?? valueLabel}>
      <canvas ref={canvas} onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
      <div className="chart-readout mono">{hover ? `${xLabel} ${tickText(hover.x)} · ${yLabel} ${tickText(hover.y)} · ${tickText(hover.v)}` : " "}</div>
    </div>
  );
}
