// キャンバスの当たり判定 (v1 CadCanvas と同じ規則): 領域は内側か輪郭から数 px 以内、重なっていれば面積の
// 小さい (内側の) 方。選んだ領域の頂点 > 辺の中点 > 本体の順にハンドルを探す。輪郭の円弧 (bulge) も扱う (P7)。

import { closestPoint, pathSegs, pointInPath } from "../cad/geom";
import { bulgesOf, edgeMidpoint, type PathData } from "../cad/path";
import { domainPath, regionArea, regionPath, type CircleShape, type Point, type Project, type Region } from "../model/project";
import { toScreen, type Camera } from "./camera";

/** クリックとみなす移動量 [px] */
export const CLICK_TOLERANCE_PX = 5;
/** 輪郭の当たりの幅 [px] */
export const EDGE_TOLERANCE_PX = 6;
/** ハンドルの当たりの幅 [px] */
export const HANDLE_TOLERANCE_PX = 8;

export function pointInPolygon([x, y]: Point, poly: Point[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export function distToSegment([px, py]: Point, [ax, ay]: Point, [bx, by]: Point): number {
  const dx = bx - ax;
  const dy = by - ay;
  const len2 = dx * dx + dy * dy;
  const t = len2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2)) : 0;
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

/** 輪郭 (円弧を含む) までの距離 */
export function distToPath(pt: Point, path: PathData): number {
  let d = Infinity;
  for (const s of pathSegs(path.polygon, bulgesOf(path))) d = Math.min(d, closestPoint(s, pt).dist);
  return d;
}

export function hitCircle(pt: Point, s: CircleShape, tol: number): boolean {
  return Math.hypot(pt[0] - s.center[0], pt[1] - s.center[1]) <= s.radius + tol;
}

export function hitRegion(pt: Point, r: Region, tol: number): boolean {
  if (r.shape) return hitCircle(pt, r.shape, tol);
  const path = regionPath(r);
  if (path.polygon.length === 0) return false;
  return pointInPath(pt, path.polygon, path.bulges) || distToPath(pt, path) <= tol;
}

/** 点に当たる領域 (重なっていれば面積の小さい方)。tol はワールド座標 [m] */
export function findRegionAt(pt: Point, regions: Region[], tol: number): Region | null {
  let best: Region | null = null;
  let bestArea = Infinity;
  for (const r of regions) {
    if (!hitRegion(pt, r, tol)) continue;
    const a = regionArea(r);
    if (a < bestArea) {
      bestArea = a;
      best = r;
    }
  }
  return best;
}

/** 点から tol [m] 以内で最も近い外周の辺の番号 */
export function findDomainEdgeAt(pt: Point, p: Project, tol: number): number | null {
  const path = domainPath(p);
  let best: number | null = null;
  let bestD = tol;
  pathSegs(path.polygon, bulgesOf(path)).forEach((s, i) => {
    const d = closestPoint(s, pt).dist;
    if (d <= bestD) {
      bestD = d;
      best = i;
    }
  });
  return best;
}

/** 画面の点 (sx, sy) から tol px 以内で最も近い頂点の番号 */
export function findVertexHandle(poly: Point[], c: Camera, sx: number, sy: number, tol = HANDLE_TOLERANCE_PX): number | null {
  let best: number | null = null;
  let bestD = tol;
  poly.forEach((p, i) => {
    const [x, y] = toScreen(c, p);
    const d = Math.hypot(x - sx, y - sy);
    if (d <= bestD) {
      bestD = d;
      best = i;
    }
  });
  return best;
}

/** 辺の中点のハンドル (番号 i は頂点 i → i+1 の辺) */
export function findMidpointHandle(path: PathData, c: Camera, sx: number, sy: number, tol = HANDLE_TOLERANCE_PX): number | null {
  let best: number | null = null;
  let bestD = tol;
  for (let i = 0; i < path.polygon.length; i++) {
    const [x, y] = toScreen(c, edgeMidpoint(path, i));
    const d = Math.hypot(x - sx, y - sy);
    if (d <= bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

/** 円の半径のハンドル (中心の右、角度 0°) */
export function radiusHandlePoint(s: CircleShape): Point {
  return [s.center[0] + s.radius, s.center[1]];
}

export function hitRadiusHandle(s: CircleShape, c: Camera, sx: number, sy: number, tol = HANDLE_TOLERANCE_PX): boolean {
  const [x, y] = toScreen(c, radiusHandlePoint(s));
  return Math.hypot(x - sx, y - sy) <= tol;
}

/** 線分 (配置物) に当たるか。tol はワールド座標 [m] */
export function hitSegment(pt: Point, p1: Point, p2: Point, tol: number): boolean {
  return distToSegment(pt, p1, p2) <= tol;
}

/** 2 点を対角とする軸平行な矩形 (反時計回り、左下から) */
export function rectFromCorners([x0, y0]: Point, [x1, y1]: Point): Point[] {
  const xa = Math.min(x0, x1);
  const xb = Math.max(x0, x1);
  const ya = Math.min(y0, y1);
  const yb = Math.max(y0, y1);
  return [
    [xa, ya],
    [xb, ya],
    [xb, yb],
    [xa, yb],
  ];
}

/** 折れ線の点の後始末: ダブルクリックで重なった最後の点を除く (v1 と同じ) */
export function dedupeTail(pts: Point[], eps = 1e-12): Point[] {
  if (pts.length < 2) return pts;
  const a = pts[pts.length - 1];
  const b = pts[pts.length - 2];
  return Math.hypot(a[0] - b[0], a[1] - b[1]) < eps ? pts.slice(0, -1) : pts;
}
