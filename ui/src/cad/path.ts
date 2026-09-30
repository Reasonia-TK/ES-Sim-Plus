// 閉じた経路 (ドメインの外周・領域の外周と穴): 頂点と辺ごとの bulge (0 は直線、DXF の LWPOLYLINE と同じ)。
// 辺 i は頂点 i → i+1 (最後は頂点 0 へ)。頂点の追加・削除・移動、円弧の設定、向き、相似変換・鏡映。
// bulges は頂点と同じ長さに揃え、全て 0 なら持たない (null)。

import { arcSweep, midpoint, pathArea, pointAt, segFromBulge, type ArcSeg, type Vec } from "./geom";

export interface PathData {
  polygon: Vec[];
  bulges?: number[] | null;
}

/** bulge を 0 とみなす大きさ */
const BULGE_EPS = 1e-12;

/** 辺ごとの bulge (無ければ 0) */
export function bulgesOf(p: PathData): number[] {
  const n = p.polygon.length;
  const b = p.bulges;
  return Array.from({ length: n }, (_, i) => {
    const v = b?.[i] ?? 0;
    return Number.isFinite(v) && Math.abs(v) > BULGE_EPS ? v : 0;
  });
}

/** 保存する形: 全て 0 なら null */
export function packBulges(b: number[]): number[] | null {
  return b.some((v) => Math.abs(v) > BULGE_EPS) ? b.map((v) => (Math.abs(v) > BULGE_EPS ? v : 0)) : null;
}

export function hasArcs(p: PathData): boolean {
  return bulgesOf(p).some((v) => v !== 0);
}

/** 辺 i の中点 (円弧は弧の中点) */
export function edgeMidpoint(p: PathData, i: number): Vec {
  const poly = p.polygon;
  return midpoint(segFromBulge(poly[i], poly[(i + 1) % poly.length], bulgesOf(p)[i]));
}

/** 頂点と辺の中点 (編集のハンドル・ラベルの位置) */
export function pathHandles(p: PathData): { vertices: Vec[]; midpoints: Vec[] } {
  return { vertices: p.polygon, midpoints: p.polygon.map((_, i) => edgeMidpoint(p, i)) };
}

function make(polygon: Vec[], bulges: number[]): PathData {
  const packed = packBulges(bulges);
  return packed ? { polygon, bulges: packed } : { polygon };
}

/** 中心角 θ [rad] (符号付き、正は反時計回り) → bulge */
export function bulgeFromAngle(theta: number): number {
  return Math.tan(theta / 4);
}

export function angleFromBulge(b: number): number {
  return 4 * Math.atan(b);
}

/** 辺 i を u (0〜1) の位置で 2 つに分ける (新しい頂点は i + 1。円弧は同じ円の 2 つの円弧に) */
export function splitEdge(p: PathData, i: number, u = 0.5): PathData {
  const n = p.polygon.length;
  const b = bulgesOf(p);
  const a = p.polygon[i];
  const c = p.polygon[(i + 1) % n];
  const seg = segFromBulge(a, c, b[i]);
  const m = pointAt(seg, u);
  const theta = angleFromBulge(b[i]);
  const b1 = b[i] ? bulgeFromAngle(theta * u) : 0;
  const b2 = b[i] ? bulgeFromAngle(theta * (1 - u)) : 0;
  const polygon = [...p.polygon.slice(0, i + 1), m, ...p.polygon.slice(i + 1)];
  const bulges = [...b.slice(0, i), b1, b2, ...b.slice(i + 1)];
  return make(polygon, bulges);
}

/** 同じ円の同じ向きの円弧か */
function sameCircle(s: ArcSeg, t: ArcSeg): boolean {
  const tol = 1e-9 * Math.max(s.r, t.r);
  return s.ccw === t.ccw && Math.abs(s.r - t.r) <= tol && Math.hypot(s.center[0] - t.center[0], s.center[1] - t.center[1]) <= tol;
}

/**
 * 頂点 i を消す (前後の辺を 1 本に。同じ円の円弧どうしなら 1 つの円弧、それ以外は直線)。
 * 最小の頂点数 (直線だけなら 3、円弧があれば 2) を下回るなら null
 */
export function removeVertex(p: PathData, i: number): PathData | null {
  const n = p.polygon.length;
  const b = bulgesOf(p);
  const prev = (i - 1 + n) % n;
  const s1 = segFromBulge(p.polygon[prev], p.polygon[i], b[prev]);
  const s2 = segFromBulge(p.polygon[i], p.polygon[(i + 1) % n], b[i]);
  let merged = 0;
  if (s1.kind === "arc" && s2.kind === "arc" && sameCircle(s1, s2)) {
    const theta = (s1.ccw ? 1 : -1) * (arcSweep(s1) + arcSweep(s2));
    // 一周に近い円弧は 1 本では表せない
    if (Math.abs(theta) < 2 * Math.PI - 1e-9) merged = bulgeFromAngle(theta);
  }
  const polygon = p.polygon.filter((_, k) => k !== i);
  const bulges = b.filter((_, k) => k !== i);
  // 前の辺 (prev) が 1 本になった辺。i = 0 のときは最後の辺
  const at = i === 0 ? polygon.length - 1 : prev;
  bulges[at] = merged;
  const out = make(polygon, bulges);
  return isValidPath(out) ? out : null;
}

export function moveVertex(p: PathData, i: number, q: Vec): PathData {
  return make(
    p.polygon.map((v, k) => (k === i ? q : v)),
    bulgesOf(p),
  );
}

export function setBulge(p: PathData, i: number, bulge: number): PathData {
  const b = bulgesOf(p);
  b[i] = bulge;
  return make(p.polygon.slice(), b);
}

/** 符号付き面積 (反時計回りが正) */
export function signedArea(p: PathData): number {
  return pathArea(p.polygon, bulgesOf(p));
}

/** 経路として使えるか (直線だけなら 3 点以上、円弧を含めば 2 点以上、面積が 0 でない) */
export function isValidPath(p: PathData): boolean {
  const n = p.polygon.length;
  if (n < 2 || (n < 3 && !hasArcs(p))) return false;
  if (p.polygon.some(([x, y]) => !Number.isFinite(x) || !Number.isFinite(y))) return false;
  return Math.abs(signedArea(p)) > 0;
}

/** 向きを逆に (頂点の順を逆にし、bulge は逆の辺へ移して符号を反転)。辺 k は元の辺 (n - 2 - k) mod n */
export function reversePath(p: PathData): PathData {
  const n = p.polygon.length;
  const b = bulgesOf(p);
  const polygon = p.polygon.slice().reverse();
  const bulges = Array.from({ length: n }, (_, k) => -b[(((n - 2 - k) % n) + n) % n]);
  return make(polygon, bulges);
}

/** 逆向きにしたときの辺の対応 (新しい辺 k → 元の辺) */
export function reversedEdgeIndex(n: number, k: number): number {
  return (((n - 2 - k) % n) + n) % n;
}

/** 反時計回りにそろえる (逆にしたら reversed = true) */
export function ensureCcw(p: PathData): { path: PathData; reversed: boolean } {
  return signedArea(p) < 0 ? { path: reversePath(p), reversed: true } : { path: p, reversed: false };
}

/** アフィン変換 x' = a·x + b·y + e, y' = c·x + d·y + f (相似変換と鏡映だけが円弧を保つ) */
export interface Affine {
  a: number;
  b: number;
  c: number;
  d: number;
  e: number;
  f: number;
}

export const IDENTITY: Affine = { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };

export function applyAffine(m: Affine, [x, y]: Vec): Vec {
  return [m.a * x + m.b * y + m.e, m.c * x + m.d * y + m.f];
}

/** m2 · m1 (先に m1) */
export function composeAffine(m2: Affine, m1: Affine): Affine {
  return {
    a: m2.a * m1.a + m2.b * m1.c,
    b: m2.a * m1.b + m2.b * m1.d,
    c: m2.c * m1.a + m2.d * m1.c,
    d: m2.c * m1.b + m2.d * m1.d,
    e: m2.a * m1.e + m2.b * m1.f + m2.e,
    f: m2.c * m1.e + m2.d * m1.f + m2.f,
  };
}

export function translation(dx: number, dy: number): Affine {
  return { a: 1, b: 0, c: 0, d: 1, e: dx, f: dy };
}

/** 点 c のまわりに角度 θ [rad] 回す */
export function rotation(theta: number, [cx, cy]: Vec = [0, 0]): Affine {
  const co = Math.cos(theta);
  const si = Math.sin(theta);
  return { a: co, b: -si, c: si, d: co, e: cx - co * cx + si * cy, f: cy - si * cx - co * cy };
}

/** 点 c を中心に k 倍 */
export function scaling(k: number, [cx, cy]: Vec = [0, 0]): Affine {
  return { a: k, b: 0, c: 0, d: k, e: cx - k * cx, f: cy - k * cy };
}

/** 点 p と q を通る直線での鏡映 */
export function mirror([px, py]: Vec, [qx, qy]: Vec): Affine {
  const dx = qx - px;
  const dy = qy - py;
  const L2 = dx * dx + dy * dy;
  const ux = dx * dx - dy * dy;
  const uy = 2 * dx * dy;
  const a = ux / L2;
  const b = uy / L2;
  // x' = R (x - p) + p、R = [[a, b], [b, -a]]
  return { a, b, c: b, d: -a, e: px - a * px - b * py, f: py - b * px + a * py };
}

/** 円弧を保つ変換か (相似: 直交行列の定数倍) */
export function isSimilarity(m: Affine, tol = 1e-9): boolean {
  const s1 = m.a * m.a + m.c * m.c;
  const s2 = m.b * m.b + m.d * m.d;
  const dotp = m.a * m.b + m.c * m.d;
  return Math.abs(s1 - s2) <= tol * Math.max(s1, s2) && Math.abs(dotp) <= tol * Math.max(s1, s2);
}

export function determinant(m: Affine): number {
  return m.a * m.d - m.b * m.c;
}

/** 経路を変換する (鏡映なら bulge の符号を反転し、向きを反時計回りに戻すかは呼ぶ側で) */
export function transformPath(p: PathData, m: Affine): PathData {
  const flip = determinant(m) < 0 ? -1 : 1;
  return make(
    p.polygon.map((q) => applyAffine(m, q)),
    bulgesOf(p).map((v) => v * flip),
  );
}

/** 円 (中心・半径) を 2 つの半円の経路に (頂点は右端と左端、反時計回り) */
export function circlePath([cx, cy]: Vec, r: number): PathData {
  return {
    polygon: [
      [cx + r, cy],
      [cx - r, cy],
    ],
    bulges: [1, 1],
  };
}
