// 2 つの多角形の領域を 1 つに結合する (和集合、v1 App.tsx の mergeRegions と同じ規則: 円・離れた領域・
// 穴ができる結果はエラー、共有辺に残る一直線上の頂点は除く、結合後の値はこの領域のものを使う)。

import polygonClipping, { type Polygon } from "polygon-clipping";
import type { Point } from "./project";

export type MergeError = "circle" | "notAdjacent" | "holes" | "failed";

/** 一直線に並ぶ頂点を除く (閉じるために繰り返された最後の点も除く、しきい値は座標の大きさに合わせる) */
export function removeCollinear(ring: Point[]): Point[] {
  let pts = ring.slice();
  if (pts.length > 1) {
    const [fx, fy] = pts[0];
    const [lx, ly] = pts[pts.length - 1];
    if (fx === lx && fy === ly) pts = pts.slice(0, -1);
  }
  const n = pts.length;
  if (n < 3) return pts;
  let maxAbs = 0;
  for (const [x, y] of pts) maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y));
  const scale = Math.max(maxAbs, 1);
  const eps = 1e-12 * scale * scale;
  const cross = (o: Point, a: Point, b: Point) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const out: Point[] = [];
  for (let i = 0; i < n; i++) {
    if (Math.abs(cross(pts[(i - 1 + n) % n], pts[i], pts[(i + 1) % n])) > eps) out.push(pts[i]);
  }
  return out.length >= 3 ? out : pts;
}

export function unionPolygons(a: Point[], b: Point[]): { polygon: Point[] } | { error: MergeError } {
  let result;
  try {
    result = polygonClipping.union([a] as unknown as Polygon, [b] as unknown as Polygon);
  } catch {
    return { error: "failed" };
  }
  if (result.length !== 1) return { error: "notAdjacent" };
  if (result[0].length > 1) return { error: "holes" };
  return { polygon: removeCollinear(result[0][0] as Point[]) };
}
