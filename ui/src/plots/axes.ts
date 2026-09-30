// キャンバスに描くグラフ (ヒートマップ・散布図) の共通部分: きりのよい目盛り・目盛りの数値・枠と格子と軸の名前・
// 高解像度の画面に合わせたキャンバスの大きさ・テーマの色。

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** きりのよい目盛り (1・2・5 × 10^k、count 個ほど) */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  if (min > max) [min, max] = [max, min];
  const raw = (max - min) / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out: number[] = [];
  const first = Math.ceil(min / step - 1e-9);
  for (let i = first; i * step <= max + step * 1e-9; i++) {
    const v = i * step;
    out.push(Math.abs(v) < step * 1e-9 ? 0 : Number(v.toPrecision(12)));
  }
  return out;
}

/** 目盛りの数値 (1e5 以上か 1e-3 未満は指数表記、ほかは有効数字 4 桁) */
export function tickText(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) return v.toExponential(1).replace("e+", "e");
  return String(Number(v.toPrecision(4)));
}

export interface ChartColors {
  text: string;
  grid: string;
  frame: string;
  bg: string;
}

export function chartColors(): ChartColors {
  const cs = getComputedStyle(document.documentElement);
  const v = (name: string, d: string) => cs.getPropertyValue(name).trim() || d;
  return { text: v("--text-muted", "#8f97a4"), grid: v("--border", "#3a3f49"), frame: v("--border", "#3a3f49"), bg: v("--canvas-bg", "#15171b") };
}

export const CHART_FONT = "11px system-ui, -apple-system, 'Segoe UI', sans-serif";

/** キャンバスを入れ物の大きさ (CSS px) × devicePixelRatio にして、CSS px で描けるようにする */
export function fitCanvas(canvas: HTMLCanvasElement, width: number, height: number): CanvasRenderingContext2D | null {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(width * dpr));
  const h = Math.max(1, Math.round(height * dpr));
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  let ctx: CanvasRenderingContext2D | null = null;
  try {
    ctx = canvas.getContext("2d");
  } catch {
    return null;
  }
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return ctx;
}

/** 値 → 画面 (線形) */
export function scaleOf(range: [number, number], from: number, to: number): (v: number) => number {
  const span = range[1] - range[0] || 1;
  return (v) => from + ((v - range[0]) / span) * (to - from);
}

/** 枠・格子・目盛り・軸の名前 */
export function drawAxes(ctx: CanvasRenderingContext2D, rect: Rect, xr: [number, number], yr: [number, number], xLabel: string, yLabel: string, c: ChartColors): void {
  ctx.save();
  ctx.font = CHART_FONT;
  ctx.lineWidth = 1;
  const px = scaleOf(xr, rect.x, rect.x + rect.w);
  const py = scaleOf(yr, rect.y + rect.h, rect.y);
  const xt = niceTicks(xr[0], xr[1], Math.max(2, Math.floor(rect.w / 90)));
  const yt = niceTicks(yr[0], yr[1], Math.max(2, Math.floor(rect.h / 36)));
  ctx.strokeStyle = c.grid;
  ctx.fillStyle = c.text;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const v of xt) {
    const x = Math.round(px(v)) + 0.5;
    if (x < rect.x - 0.5 || x > rect.x + rect.w + 0.5) continue;
    ctx.beginPath();
    ctx.moveTo(x, rect.y + rect.h);
    ctx.lineTo(x, rect.y + rect.h + 4);
    ctx.stroke();
    ctx.fillText(tickText(v), x, rect.y + rect.h + 6);
  }
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const v of yt) {
    const y = Math.round(py(v)) + 0.5;
    if (y < rect.y - 0.5 || y > rect.y + rect.h + 0.5) continue;
    ctx.beginPath();
    ctx.moveTo(rect.x - 4, y);
    ctx.lineTo(rect.x, y);
    ctx.stroke();
    ctx.fillText(tickText(v), rect.x - 6, y);
  }
  ctx.strokeStyle = c.frame;
  ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.w - 1, rect.h - 1);
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(xLabel, rect.x + rect.w / 2, rect.y + rect.h + 34);
  ctx.translate(12, rect.y + rect.h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = "top";
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

/** 目盛りの格子 (データの下に薄く) */
export function drawGrid(ctx: CanvasRenderingContext2D, rect: Rect, xr: [number, number], yr: [number, number], c: ChartColors): void {
  ctx.save();
  ctx.strokeStyle = c.grid;
  ctx.globalAlpha = 0.5;
  ctx.lineWidth = 1;
  const px = scaleOf(xr, rect.x, rect.x + rect.w);
  const py = scaleOf(yr, rect.y + rect.h, rect.y);
  for (const v of niceTicks(xr[0], xr[1], Math.max(2, Math.floor(rect.w / 90)))) {
    const x = Math.round(px(v)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, rect.y);
    ctx.lineTo(x, rect.y + rect.h);
    ctx.stroke();
  }
  for (const v of niceTicks(yr[0], yr[1], Math.max(2, Math.floor(rect.h / 36)))) {
    const y = Math.round(py(v)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(rect.x, y);
    ctx.lineTo(rect.x + rect.w, y);
    ctx.stroke();
  }
  ctx.restore();
}

/** 範囲 (有限の値の最小・最大、同じなら少し広げる) */
export function extent(arrays: ArrayLike<number>[], pad = 0): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const a of arrays)
    for (let i = 0; i < a.length; i++) {
      const v = a[i];
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  if (lo > hi) return [0, 1];
  if (lo === hi) return lo === 0 ? [-1, 1] : [lo - Math.abs(lo) * 0.1, hi + Math.abs(hi) * 0.1];
  const p = (hi - lo) * pad;
  return [lo - p, hi + p];
}
