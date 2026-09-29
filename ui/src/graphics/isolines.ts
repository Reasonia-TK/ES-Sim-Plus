// 等値線 (marching triangles、v1 canvas/isolines.ts の移植)。節点の値が等値を跨ぐ辺で線形補間した交点を
// 三角形ごとに 1 本の線分にする。線分は x0, y0, x1, y1 の並び [m] で返し、そのまま WebGL の線分に使う。

import type { TriMesh } from "./meshIndex";

/** 最小・最大を除いて等分した n 本の値 (v1 と同じ、既定 15 本) */
export function isoLevels(min: number, max: number, n = 15): number[] {
  if (!(max > min)) return [];
  return Array.from({ length: n }, (_, i) => min + ((max - min) * (i + 1)) / (n + 1));
}

export function isolineSegments(mesh: TriMesh, values: ArrayLike<number>, levels: number[]): Float64Array {
  const out: number[] = [];
  const { nodes, triangles } = mesh;
  const pts: number[] = [];
  const cross = (a: number, b: number, level: number) => {
    const va = values[a];
    const vb = values[b];
    if (va < level === vb < level) return;
    const t = (level - va) / (vb - va);
    pts.push(nodes[a][0] + (nodes[b][0] - nodes[a][0]) * t, nodes[a][1] + (nodes[b][1] - nodes[a][1]) * t);
  };
  for (const [a, b, c] of triangles) {
    const va = values[a];
    const vb = values[b];
    const vc = values[c];
    if (!(Number.isFinite(va) && Number.isFinite(vb) && Number.isFinite(vc))) continue;
    const lo = Math.min(va, vb, vc);
    const hi = Math.max(va, vb, vc);
    for (const level of levels) {
      if (level < lo || level > hi) continue;
      pts.length = 0;
      cross(a, b, level);
      cross(b, c, level);
      cross(c, a, level);
      if (pts.length === 4) out.push(pts[0], pts[1], pts[2], pts[3]);
    }
  }
  return Float64Array.from(out);
}
