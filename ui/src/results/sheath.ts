// シース端 (v1 sheath.ts): 評価線の上の n_e・n_i から Brinkmann の条件でシース端 s を求める。
// 線の上の 200 点のメッシュの中の位置 (三角形と重み) は線とメッシュごとに 1 回だけ探し、ビン・量ごとに使い回す。

import type { Point } from "../model/project";
import { meshIndexOf, type ViewMesh } from "../graphics/scene";

/**
 * Brinkmann のシース端: G(s) = ∫0^s n_i dx − ∫0^xb (n_i − n_e) dx = 0 の根 (台形則の累積和、線形補間)。
 * dist は電極側からの距離、xb = 最後の点。見つからなければ null。v1 brinkmannSheathEdge と同じ。
 */
export function brinkmannSheathEdge(dist: ArrayLike<number>, nE: ArrayLike<number>, nI: ArrayLike<number>): number | null {
  const n = dist.length;
  if (n < 2) return null;
  const xB = dist[n - 1];
  if (xB <= dist[0]) return null;
  const cumE = new Float64Array(n);
  const cumI = new Float64Array(n);
  for (let i = 1; i < n; i++) {
    const dx = dist[i] - dist[i - 1];
    cumE[i] = cumE[i - 1] + 0.5 * (nE[i] + nE[i - 1]) * dx;
    cumI[i] = cumI[i - 1] + 0.5 * (nI[i] + nI[i - 1]) * dx;
  }
  const c = cumI[n - 1] - cumE[n - 1];
  const g = (i: number) => cumI[i] - c;
  if (g(0) > 0 || g(n - 1) <= 0) return null;
  let idx = 0;
  while (idx < n && g(idx) < 0) idx++;
  if (idx <= 0) return dist[0];
  const lo = g(idx - 1);
  const hi = g(idx);
  if (hi === lo) return dist[idx];
  return dist[idx - 1] + (-lo / (hi - lo)) * (dist[idx] - dist[idx - 1]);
}

export interface LineSampler {
  /** 電極側 (p1) からの距離 [m] (メッシュの外の点は除いてある) */
  dist: number[];
  tri: number[];
  w: [number, number, number][];
}

const samplerCache = new WeakMap<object, Map<string, LineSampler>>();

/** 線 p1→p2 の上の n 点のうちメッシュの中にあるもの */
export function lineSampler(mesh: ViewMesh, p1: Point, p2: Point, n = 200): LineSampler {
  let byLine = samplerCache.get(mesh);
  if (!byLine) {
    byLine = new Map();
    samplerCache.set(mesh, byLine);
  }
  const key = `${p1[0]},${p1[1]},${p2[0]},${p2[1]},${n}`;
  const hit = byLine.get(key);
  if (hit) return hit;
  const idx = meshIndexOf(mesh);
  const len = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
  const out: LineSampler = { dist: [], tri: [], w: [] };
  for (let i = 0; i < n; i++) {
    const t = n === 1 ? 0 : i / (n - 1);
    const h = idx.locate(p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t);
    if (!h) continue;
    out.dist.push(len * t);
    out.tri.push(h.tri);
    out.w.push(h.w);
  }
  byLine.set(key, out);
  return out;
}

function sampleNodes(mesh: ViewMesh, s: LineSampler, values: ArrayLike<number>): Float64Array {
  const out = new Float64Array(s.tri.length);
  for (let k = 0; k < s.tri.length; k++) {
    const [a, b, c] = mesh.triangles[s.tri[k]];
    const w = s.w[k];
    out[k] = w[0] * values[a] + w[1] * values[b] + w[2] * values[c];
  }
  return out;
}

/** 評価線のシース端 s [m] (電極側 p1 からの距離)。n_e・n_i は節点の値 */
export function sheathOnLine(mesh: ViewMesh, p1: Point, p2: Point, nE: ArrayLike<number>, nI: ArrayLike<number>): number | null {
  const s = lineSampler(mesh, p1, p2);
  if (s.dist.length < 2) return null;
  return brinkmannSheathEdge(s.dist, sampleNodes(mesh, s, nE), sampleNodes(mesh, s, nI));
}

/** シース端の点 (p1 から s 進んだ所) */
export function sheathPoint(p1: Point, p2: Point, s: number): Point {
  const len = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) || 1;
  return [p1[0] + ((p2[0] - p1[0]) * s) / len, p1[1] + ((p2[1] - p1[1]) * s) / len];
}

export interface SheathLineDef {
  label?: string;
  p1: Point;
  p2: Point;
}

/** 文書の評価線 (pic.sheath_lines、最大 4 本) */
export function sheathLinesOf(project: { pic?: unknown } | null): SheathLineDef[] {
  const pic = project?.pic as { sheath_lines?: SheathLineDef[] } | null | undefined;
  return Array.isArray(pic?.sheath_lines) ? pic.sheath_lines.filter((l) => Array.isArray(l?.p1) && Array.isArray(l?.p2)) : [];
}
