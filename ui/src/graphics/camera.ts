// 2D ビューの座標変換 (ワールド [m] ↔ 画面 [CSS px]、y は上向き)。拡大・移動・全体表示。

import { tidy, type Bounds, type Point } from "../model/project";

export interface Camera {
  /** 1 m あたりの px */
  scale: number;
  /** ワールドの原点の画面位置 */
  ox: number;
  oy: number;
}

export const ZOOM_STEP = 1.15;

export function toScreen(c: Camera, [x, y]: Point): Point {
  return [c.ox + x * c.scale, c.oy - y * c.scale];
}

export function toWorld(c: Camera, sx: number, sy: number): Point {
  return [(sx - c.ox) / c.scale, (c.oy - sy) / c.scale];
}

/** 範囲を画面の fill 割に収める (v1 は 80%) */
export function fitCamera(b: Bounds, width: number, height: number, fill = 0.8): Camera {
  const bw = Math.max(b.x1 - b.x0, 1e-12);
  const bh = Math.max(b.y1 - b.y0, 1e-12);
  const scale = Math.max(Math.min((width * fill) / bw, (height * fill) / bh), 1e-9);
  return { scale, ox: width / 2 - ((b.x0 + b.x1) / 2) * scale, oy: height / 2 + ((b.y0 + b.y1) / 2) * scale };
}

/** 画面の点 (sx, sy) を中心に拡大・縮小する (その点のワールド座標は動かない) */
export function zoomAt(c: Camera, sx: number, sy: number, factor: number, limits?: { min: number; max: number }): Camera {
  let scale = c.scale * factor;
  if (limits) scale = Math.min(Math.max(scale, limits.min), limits.max);
  const f = scale / c.scale;
  return { scale, ox: sx - (sx - c.ox) * f, oy: sy - (sy - c.oy) * f };
}

export function panBy(c: Camera, dx: number, dy: number): Camera {
  return { ...c, ox: c.ox + dx, oy: c.oy + dy };
}

/** 画面に見えているワールドの範囲 */
export function visibleBounds(c: Camera, width: number, height: number): Bounds {
  const [x0, y1] = toWorld(c, 0, 0);
  const [x1, y0] = toWorld(c, width, height);
  return { x0, y0, x1, y1 };
}

/**
 * グリッドの間隔 [m]: 画面上で minPx 以上になる 10 の累乗 (v1 は 1 mm で下限、v2 は下限なし)。
 * スナップはその 1/10 (v1 と同じ)。
 */
export function gridStep(c: Camera, minPx = 20): number {
  const raw = minPx / c.scale;
  return Math.pow(10, Math.ceil(Math.log10(raw)));
}

export function snapValue(v: number, step: number): number {
  return tidy(Math.round(v / step) * step);
}

export function snapPoint(p: Point, step: number): Point {
  return [snapValue(p[0], step), snapValue(p[1], step)];
}
