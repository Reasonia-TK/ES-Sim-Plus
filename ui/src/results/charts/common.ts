// 結果のグラフの共通部分: 1D の格子・線形補間・x の違う系列をまとめる・時間の内訳・スペクトルの山・色。

import { usePrefs } from "../../prefs/prefs";
import { lengthUnitLabel, toDisplayLength } from "../../util/format";

/** 1D の節点 (v1 deriveXGrid: N = round(n_cells) + 1、0〜gap を等分) */
export function deriveXGrid(gapM: number, nCells: number): number[] {
  const n = Math.max(1, Math.round(nCells)) + 1;
  return Array.from({ length: n }, (_, i) => (gapM * i) / (n - 1));
}

/** 線形補間 (範囲外は端の値、空なら NaN。v1 interpLinear と同じ) */
export function interpLinear(xp: ArrayLike<number>, fp: ArrayLike<number | null>, x: number): number {
  const n = Math.min(xp.length, fp.length);
  if (n === 0) return NaN;
  const f = (i: number) => fp[i] ?? NaN;
  if (n === 1 || x <= xp[0]) return f(0);
  if (x >= xp[n - 1]) return f(n - 1);
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xp[mid] <= x) lo = mid;
    else hi = mid;
  }
  const span = xp[hi] - xp[lo];
  return span === 0 ? f(lo) : f(lo) + ((x - xp[lo]) / span) * (f(hi) - f(lo));
}

/**
 * x の違う系列を 1 本の x にまとめる (uPlot は x が共通)。ほかの系列の x の点では、その系列の範囲内なら線形補間、
 * 範囲外は null (線は切れる)。null の値はそのまま null (補間しない)。
 */
export function mergeSeriesX(list: { x: ArrayLike<number>; y: ArrayLike<number | null> }[]): { x: number[]; ys: (number | null)[][] } {
  const all = new Set<number>();
  for (const s of list) for (let i = 0; i < s.x.length; i++) if (Number.isFinite(s.x[i])) all.add(s.x[i]);
  const x = [...all].sort((a, b) => a - b);
  const ys = list.map((s) => {
    const n = Math.min(s.x.length, s.y.length);
    const out: (number | null)[] = new Array(x.length).fill(null);
    if (n === 0) return out;
    const x0 = s.x[0];
    const x1 = s.x[n - 1];
    let j = 0;
    for (let k = 0; k < x.length; k++) {
      const xv = x[k];
      if (xv < x0 || xv > x1) continue;
      while (j < n - 2 && s.x[j + 1] <= xv) j++;
      if (s.x[j] === xv) {
        out[k] = s.y[j];
        continue;
      }
      if (j + 1 < n && s.x[j + 1] === xv) {
        out[k] = s.y[j + 1];
        continue;
      }
      const a = s.y[j];
      const b = s.y[j + 1];
      if (a === null || b === null || a === undefined || b === undefined) continue;
      const span = s.x[j + 1] - s.x[j];
      out[k] = span === 0 ? a : a + ((xv - s.x[j]) / span) * (b - a);
    }
    return out;
  });
  return { x, ys };
}

/** 時間の内訳 (total 以外を秒の大きい順、割合 %)。total が無ければ和 */
export function timingRows(timing: Record<string, number> | undefined): { rows: { key: string; sec: number; pct: number | null }[]; total: number } {
  if (!timing) return { rows: [], total: 0 };
  const entries = Object.entries(timing).filter(([k, v]) => k !== "total" && typeof v === "number");
  const total = typeof timing.total === "number" ? timing.total : entries.reduce((a, [, v]) => a + v, 0);
  const rows = entries.map(([key, sec]) => ({ key, sec, pct: total > 0 ? (100 * sec) / total : null })).sort((a, b) => b.sec - a.sec);
  return { rows, total };
}

/** スペクトルの山 (両隣より大きい点、振幅の大きい順に topN。v1 と同じ、端の点は除く) */
export function spectrumPeaks(freq: ArrayLike<number>, amp: ArrayLike<number>, topN = 5): { f: number; a: number }[] {
  const out: { f: number; a: number }[] = [];
  for (let i = 1; i < amp.length - 1; i++) if (amp[i] > amp[i - 1] && amp[i] > amp[i + 1]) out.push({ f: freq[i], a: amp[i] });
  out.sort((p, q) => q.a - p.a);
  return out.slice(0, topN);
}

/** f0 にいちばん近い点の振幅 */
export function ampAt(freq: ArrayLike<number>, amp: ArrayLike<number>, f0: number): number | null {
  let best = -1;
  let bestD = Infinity;
  for (let i = 0; i < freq.length; i++) {
    const d = Math.abs(freq[i] - f0);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best >= 0 ? amp[best] : null;
}

/** 位相分解の全ビンの最小・最大 (再生で軸が動かないよう固定する) */
export function rowsRange(rows: ArrayLike<number | null>[] | undefined, log = false): [number, number] | null {
  if (!rows) return null;
  let lo = Infinity;
  let hi = -Infinity;
  for (const r of rows)
    for (let i = 0; i < r.length; i++) {
      const v = r[i];
      if (v === null || !Number.isFinite(v) || (log && v <= 0)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  if (!(lo <= hi)) return null;
  if (lo === hi) return lo === 0 ? [-1, 1] : [lo - Math.abs(lo) * 0.1, hi + Math.abs(hi) * 0.1];
  if (log) return [lo, hi];
  const pad = (hi - lo) * 0.04;
  return [lo - pad, hi + pad];
}

/** 長さの表示単位 (m → mm / µm) */
export function useLength(): { unit: string; of: (m: number) => number; ofAll: (a: ArrayLike<number | null>) => (number | null)[] } {
  const u = usePrefs((s) => s.lengthUnit);
  const of = (m: number) => toDisplayLength(m, u);
  return { unit: lengthUnitLabel(u), of, ofAll: (a) => Array.from(a, (v) => (v === null ? null : of(v))) };
}

// ---- 色 (v1 と同じ) ----

/** 1D の量 (この実行・比較の実行は濃い色) */
export const FIELD1D_COLORS: Record<string, [string, string]> = {
  phi: ["#59c2ff", "#2f7dbf"],
  e: ["#59c2ff", "#2f7dbf"],
  n_e: ["#4da3ff", "#2a5fa8"],
  n_i: ["#ffb84d", "#c98a2e"],
  t_e: ["#6fd08c", "#3f9463"],
  ionization: ["#c792ea", "#8a5aa8"],
};
export const ELECTRON_COLOR = "#4da3ff";
export const ION_COLOR = "#ffb84d";
export const SHEATH_COLOR = "#ffb454";
export const SHEATH_RIGHT_COLOR = "#ff7a45";
export const EEDF_COLORS = ["#c792ea", "#7ec8e3", "#f2b880", "#8ee6a9"];
export const WALL_COLORS = { left: "#c792ea", right: "#7ec8e3" };
export const HARMONIC_COLOR = "rgba(200, 208, 220, 0.35)";
