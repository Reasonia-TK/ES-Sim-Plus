// CSS の色 (#rgb・#rrggbb・#rrggbbaa・rgb()・rgba()) → WebGL の色 (0〜1)。

import type { Rgba } from "./gl/renderer";

export function parseColor(css: string, alpha?: number): Rgba {
  const s = css.trim();
  let r = 0.5;
  let g = 0.5;
  let b = 0.5;
  let a = 1;
  const hex = /^#([0-9a-f]{3,8})$/i.exec(s);
  if (hex) {
    const h = hex[1];
    const n = (i: number, len: number) => parseInt(h.slice(i, i + len), 16) / (len === 1 ? 15 : 255);
    if (h.length === 3 || h.length === 4) {
      [r, g, b] = [n(0, 1), n(1, 1), n(2, 1)];
      if (h.length === 4) a = n(3, 1);
    } else if (h.length === 6 || h.length === 8) {
      [r, g, b] = [n(0, 2), n(2, 2), n(4, 2)];
      if (h.length === 8) a = n(6, 2);
    }
  } else {
    const m = /^rgba?\(([^)]+)\)$/i.exec(s);
    if (m) {
      const parts = m[1].split(/[\s,/]+/).filter(Boolean).map(Number);
      [r, g, b] = [parts[0] / 255, parts[1] / 255, parts[2] / 255];
      if (parts.length > 3) a = parts[3];
    }
  }
  return [r, g, b, alpha ?? a];
}
