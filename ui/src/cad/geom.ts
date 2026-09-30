// 2D CAD の幾何 (P7): 線分と円弧、閉じた経路 (頂点 + 辺ごとの bulge、DXF の LWPOLYLINE と同じ表現)。
// 円弧は bulge = tan(θ/4) (θ は頂点 i → i+1 の中心角、正は反時計回り)。交点・最近点・垂線の足・接点・分割・
// 面積と向き・内外判定。長さは m、角度はラジアン。

export type Vec = [number, number];

export interface LineSeg {
  kind: "line";
  a: Vec;
  b: Vec;
}

/** 円弧: 中心・半径と、a0 から a1 へ ccw (反時計回り) か cw に回る (a0・a1 はラジアン、端点は a・b) */
export interface ArcSeg {
  kind: "arc";
  center: Vec;
  r: number;
  a0: number;
  a1: number;
  ccw: boolean;
  a: Vec;
  b: Vec;
}

export interface CircleGeom {
  kind: "circle";
  center: Vec;
  r: number;
}

export type Seg = LineSeg | ArcSeg;
export type Curve = Seg | CircleGeom;

const TAU = 2 * Math.PI;

const sub = (p: Vec, q: Vec): Vec => [p[0] - q[0], p[1] - q[1]];
const add = (p: Vec, q: Vec): Vec => [p[0] + q[0], p[1] + q[1]];
const mul = (p: Vec, k: number): Vec => [p[0] * k, p[1] * k];
const dot = (p: Vec, q: Vec) => p[0] * q[0] + p[1] * q[1];
const cross = (p: Vec, q: Vec) => p[0] * q[1] - p[1] * q[0];
export const dist = (p: Vec, q: Vec) => Math.hypot(p[0] - q[0], p[1] - q[1]);

/** 角度を [0, 2π) に */
export function normAngle(a: number): number {
  const x = a % TAU;
  return x < 0 ? x + TAU : x;
}

/** 円弧の回る角 (0 < sweep ≤ 2π) */
export function arcSweep(s: ArcSeg): number {
  const d = s.ccw ? normAngle(s.a1 - s.a0) : normAngle(s.a0 - s.a1);
  return d === 0 ? TAU : d;
}

/** 角度 t が円弧の範囲に入るか */
export function angleOnArc(s: ArcSeg, t: number, eps = 1e-12): boolean {
  const sw = arcSweep(s);
  const d = s.ccw ? normAngle(t - s.a0) : normAngle(s.a0 - t);
  return d <= sw + eps || d >= TAU - eps;
}

/** 頂点 p → q の bulge を円弧に (bulge が 0 に近ければ線分) */
export function segFromBulge(p: Vec, q: Vec, bulge = 0): Seg {
  if (Math.abs(bulge) < 1e-12) return { kind: "line", a: p, b: q };
  const theta = 4 * Math.atan(bulge);
  const chord = dist(p, q);
  const r = Math.abs(chord / (2 * Math.sin(theta / 2)));
  // 弦の中点から中心へ: 弦に垂直、距離 r·cos(θ/2) (θ の符号で向き)
  const mid = mul(add(p, q), 0.5);
  const dir = mul(sub(q, p), 1 / chord);
  const nrm: Vec = [-dir[1], dir[0]];
  const h = (chord / 2) / Math.tan(theta / 2);
  const center = add(mid, mul(nrm, h));
  return { kind: "arc", center, r, a0: Math.atan2(p[1] - center[1], p[0] - center[0]), a1: Math.atan2(q[1] - center[1], q[0] - center[0]), ccw: theta > 0, a: p, b: q };
}

/** 円弧 → bulge */
export function bulgeOf(s: Seg): number {
  if (s.kind === "line") return 0;
  const th = arcSweep(s);
  return (s.ccw ? 1 : -1) * Math.tan(th / 4);
}

export function arcPoint(c: Vec, r: number, t: number): Vec {
  return [c[0] + r * Math.cos(t), c[1] + r * Math.sin(t)];
}

/** 線分・円弧上の点 (u は 0〜1) */
export function pointAt(s: Seg, u: number): Vec {
  if (s.kind === "line") return add(s.a, mul(sub(s.b, s.a), u));
  const sw = arcSweep(s) * u;
  return arcPoint(s.center, s.r, s.a0 + (s.ccw ? sw : -sw));
}

export function segLength(s: Seg): number {
  return s.kind === "line" ? dist(s.a, s.b) : s.r * arcSweep(s);
}

/** 中点 (円弧は弧の中点) */
export function midpoint(s: Seg): Vec {
  return pointAt(s, 0.5);
}

/**
 * a から b へ、点 m を通る円弧の bulge (3 点の円弧)。m が弦 a→b の右にあれば反時計回り (正)。
 * 中心角は 2π - 2α (α は m での角 ∠amb)。m が弦の上 (一直線) なら 0 (直線)、m が端点と重なれば 0
 */
export function bulgeThrough(a: Vec, m: Vec, b: Vec): number {
  const u = sub(a, m);
  const v = sub(b, m);
  const lu = Math.hypot(u[0], u[1]);
  const lv = Math.hypot(v[0], v[1]);
  if (lu === 0 || lv === 0 || dist(a, b) === 0) return 0;
  const c = cross(sub(b, a), sub(m, a));
  if (Math.abs(c) <= 1e-12 * dist(a, b) * Math.max(lu, lv)) return 0;
  const alpha = Math.acos(Math.min(1, Math.max(-1, dot(u, v) / (lu * lv))));
  const mag = Math.tan((Math.PI - alpha) / 2);
  return c < 0 ? mag : -mag;
}

/** 線分・円弧上の点 q のパラメータ (0〜1、範囲外は近い端に寄せない) */
export function paramOf(s: Seg, q: Vec): number {
  if (s.kind === "line") {
    const d = sub(s.b, s.a);
    const L2 = dot(d, d);
    return L2 > 0 ? dot(sub(q, s.a), d) / L2 : 0;
  }
  const t = Math.atan2(q[1] - s.center[1], q[0] - s.center[0]);
  const d = s.ccw ? normAngle(t - s.a0) : normAngle(s.a0 - t);
  const sw = arcSweep(s);
  // 円弧の外で始点の側 (一周の残りの半分より始点に近い) は負のパラメータ
  return d > sw + (TAU - sw) / 2 ? (d - TAU) / sw : d / sw;
}

/**
 * 端での向き: from = "a" は a から出ていく向き (a→b に進むとき)、"b" は b から出ていく向き (b→a に戻るとき)。
 * angle は接線の向き [rad]、curvature は左に曲がるとき正 (1/r)、右は負、直線は 0
 */
export function tangentAt(s: Seg, from: "a" | "b"): { angle: number; curvature: number } {
  if (s.kind === "line") {
    const d = from === "a" ? sub(s.b, s.a) : sub(s.a, s.b);
    return { angle: Math.atan2(d[1], d[0]), curvature: 0 };
  }
  // 反時計回りの円弧を正の向きに進むと、接線は半径の向きから +90°、左に曲がる
  const forward = from === "a";
  const ccwMotion = forward ? s.ccw : !s.ccw;
  const t = forward ? s.a0 : s.a1;
  const angle = t + (ccwMotion ? Math.PI / 2 : -Math.PI / 2);
  return { angle: Math.atan2(Math.sin(angle), Math.cos(angle)), curvature: (ccwMotion ? 1 : -1) / s.r };
}

/** u0 から u1 (0〜1) までの部分 (円弧は同じ円の円弧、bulge は中心角の割合から) */
export function subSeg(s: Seg, u0: number, u1: number): Seg {
  const a = u0 === 0 ? s.a : pointAt(s, u0);
  const b = u1 === 1 ? s.b : pointAt(s, u1);
  if (s.kind === "line") return { kind: "line", a, b };
  const theta = (s.ccw ? 1 : -1) * arcSweep(s) * (u1 - u0);
  return segFromBulge(a, b, Math.tan(theta / 4));
}

/** 円弧を弦の最大のずれ maxErr 以内の点列に (端点を含む、最低 minSeg 分割) */
export function discretize(s: Seg, maxErr: number, minSeg = 1): Vec[] {
  if (s.kind === "line") return [s.a, s.b];
  const sw = arcSweep(s);
  // 弦のずれ r(1 - cos(Δ/2)) ≤ maxErr
  const dmax = maxErr > 0 && maxErr < s.r ? 2 * Math.acos(1 - maxErr / s.r) : Math.PI / 8;
  const n = Math.max(minSeg, Math.ceil(sw / dmax));
  return Array.from({ length: n + 1 }, (_, i) => (i === 0 ? s.a : i === n ? s.b : pointAt(s, i / n)));
}

/** 点 p に最も近い曲線上の点と距離 (u は線分・円弧のパラメータ 0〜1) */
export function closestPoint(c: Curve, p: Vec): { point: Vec; dist: number; u: number } {
  if (c.kind === "line") {
    const d = sub(c.b, c.a);
    const L2 = dot(d, d);
    const u = L2 > 0 ? Math.min(1, Math.max(0, dot(sub(p, c.a), d) / L2)) : 0;
    const q = add(c.a, mul(d, u));
    return { point: q, dist: dist(p, q), u };
  }
  const t = Math.atan2(p[1] - c.center[1], p[0] - c.center[0]);
  if (c.kind === "circle" || angleOnArc(c, t)) {
    const q = arcPoint(c.center, c.r, t);
    const u = c.kind === "arc" ? Math.min(1, (c.ccw ? normAngle(t - c.a0) : normAngle(c.a0 - t)) / arcSweep(c)) : normAngle(t) / TAU;
    return { point: q, dist: dist(p, q), u };
  }
  const da = dist(p, c.a);
  const db = dist(p, c.b);
  return da <= db ? { point: c.a, dist: da, u: 0 } : { point: c.b, dist: db, u: 1 };
}

/** 点 p から曲線 (を延ばしたもの) への垂線の足 (線分は直線、円弧・円は中心を通る直線との交点のうち近い方) */
export function perpendicularFoot(c: Curve, p: Vec): Vec | null {
  if (c.kind === "line") {
    const d = sub(c.b, c.a);
    const L2 = dot(d, d);
    if (L2 === 0) return null;
    return add(c.a, mul(d, dot(sub(p, c.a), d) / L2));
  }
  const v = sub(p, c.center);
  const L = Math.hypot(v[0], v[1]);
  if (L === 0) return null;
  return add(c.center, mul(v, c.r / L));
}

/** 点 p から円 (円弧の円) への接点 (p が円の外にあるとき 2 つ) */
export function tangentPoints(c: CircleGeom | ArcSeg, p: Vec): Vec[] {
  const v = sub(p, c.center);
  const d = Math.hypot(v[0], v[1]);
  if (d <= c.r) return [];
  const base = Math.atan2(v[1], v[0]);
  const alpha = Math.acos(c.r / d);
  const pts = [arcPoint(c.center, c.r, base + alpha), arcPoint(c.center, c.r, base - alpha)];
  return c.kind === "arc" ? pts.filter((q) => angleOnArc(c, Math.atan2(q[1] - c.center[1], q[0] - c.center[0]), 1e-9)) : pts;
}

// ---- 交点 ----

function lineLine(a: LineSeg, b: LineSeg, infinite: boolean): Vec[] {
  const r = sub(a.b, a.a);
  const s = sub(b.b, b.a);
  const den = cross(r, s);
  if (Math.abs(den) < 1e-18 * (dot(r, r) + dot(s, s))) return [];
  const qp = sub(b.a, a.a);
  const t = cross(qp, s) / den;
  const u = cross(qp, r) / den;
  const eps = 1e-9;
  if (!infinite && (t < -eps || t > 1 + eps || u < -eps || u > 1 + eps)) return [];
  return [add(a.a, mul(r, t))];
}

/** 直線 (a→b、infinite なら延長) と円の交点 */
function lineCircle(l: LineSeg, center: Vec, r: number, infinite: boolean): Vec[] {
  const d = sub(l.b, l.a);
  const A = dot(d, d);
  if (A === 0) return [];
  // 中心に最も近い直線上の点 (パラメータ tc) と、そこから交点までのパラメータの幅
  const tc = dot(sub(center, l.a), d) / A;
  const h = dist(center, add(l.a, mul(d, tc)));
  if (h > r * (1 + 1e-12)) return [];
  const w = Math.sqrt(Math.max(0, r * r - h * h) / A);
  // 接する (幅がほぼ 0) なら 1 点
  const ts = w * Math.sqrt(A) <= 1e-9 * r ? [tc] : [tc - w, tc + w];
  const eps = 1e-9;
  return ts.filter((t) => infinite || (t >= -eps && t <= 1 + eps)).map((t) => add(l.a, mul(d, t)));
}

function circleCircle(c1: Vec, r1: number, c2: Vec, r2: number): Vec[] {
  const d = dist(c1, c2);
  if (d === 0 || d > r1 + r2 + 1e-12 * (r1 + r2) || d < Math.abs(r1 - r2) - 1e-12 * (r1 + r2)) return [];
  const a = (r1 * r1 - r2 * r2 + d * d) / (2 * d);
  const h2 = r1 * r1 - a * a;
  const h = h2 > 0 ? Math.sqrt(h2) : 0;
  const u = mul(sub(c2, c1), 1 / d);
  const m = add(c1, mul(u, a));
  if (h === 0) return [m];
  const n: Vec = [-u[1], u[0]];
  return [add(m, mul(n, h)), add(m, mul(n, -h))];
}

function onArc(s: ArcSeg | CircleGeom, q: Vec): boolean {
  return s.kind === "circle" || angleOnArc(s, Math.atan2(q[1] - s.center[1], q[0] - s.center[0]), 1e-9);
}

/** 2 つの曲線の交点 (infinite なら線分を直線に延ばす: トリム・延長・スナップの見かけの交点に使う) */
export function intersections(c1: Curve, c2: Curve, infinite = false): Vec[] {
  if (c1.kind === "line" && c2.kind === "line") return lineLine(c1, c2, infinite);
  if (c1.kind === "line") {
    const c = c2 as ArcSeg | CircleGeom;
    return lineCircle(c1, c.center, c.r, infinite).filter((q) => infinite || onArc(c, q));
  }
  if (c2.kind === "line") return intersections(c2, c1, infinite);
  return circleCircle(c1.center, c1.r, c2.center, c2.r).filter((q) => infinite || (onArc(c1, q) && onArc(c2, q)));
}

// ---- 閉じた経路 (頂点 + bulge) ----

/** 辺ごとの bulge (null・省略・0 は直線) */
export type Bulges = readonly (number | null | undefined)[] | null;

/** 経路の辺 (i 番目は頂点 i → i+1、bulges[i] が円弧) */
export function pathSegs(points: Vec[], bulges?: Bulges): Seg[] {
  const n = points.length;
  return points.map((p, i) => segFromBulge(p, points[(i + 1) % n], bulges?.[i] ?? 0));
}

/** 符号付き面積 (反時計回りが正、円弧の弓形も含める) */
export function pathArea(points: Vec[], bulges?: Bulges): number {
  let area = 0;
  const n = points.length;
  for (let i = 0; i < n; i++) {
    const p = points[i];
    const q = points[(i + 1) % n];
    area += cross(p, q) / 2;
    const b = bulges?.[i] ?? 0;
    if (b) {
      // 弓形の面積 = r²(θ - sin θ)/2 (θ は符号付きの中心角)
      const s = segFromBulge(p, q, b) as ArcSeg;
      const th = 4 * Math.atan(b);
      area += (s.r * s.r * (th - Math.sin(th))) / 2;
    }
  }
  return area;
}

/** 点が閉じた経路の中にあるか (円弧は細かく分けた多角形で判定) */
export function pointInPath(p: Vec, points: Vec[], bulges?: Bulges, maxErr = 0): boolean {
  const poly = pathPolygon(points, bulges, maxErr);
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** 経路 → 多角形 (円弧は maxErr 以内に分ける。maxErr = 0 なら経路の大きさの 1e-4) */
export function pathPolygon(points: Vec[], bulges?: Bulges, maxErr = 0): Vec[] {
  if (!bulges?.some((b) => b)) return points;
  let err = maxErr;
  if (err <= 0) {
    let lo = Infinity;
    let hi = -Infinity;
    for (const [x, y] of points) {
      lo = Math.min(lo, x, y);
      hi = Math.max(hi, x, y);
    }
    err = Math.max(1e-12, (hi - lo) * 1e-4);
  }
  const out: Vec[] = [];
  for (const s of pathSegs(points, bulges)) {
    const pts = discretize(s, err);
    out.push(...pts.slice(0, -1));
  }
  return out;
}

/** 経路の外接矩形 (円弧のふくらみも含める) */
export function pathBounds(points: Vec[], bulges?: Bulges): { x0: number; y0: number; x1: number; y1: number } {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  const addPt = ([x, y]: Vec) => {
    x0 = Math.min(x0, x);
    y0 = Math.min(y0, y);
    x1 = Math.max(x1, x);
    y1 = Math.max(y1, y);
  };
  for (const s of pathSegs(points, bulges)) {
    addPt(s.a);
    if (s.kind === "arc") for (const t of [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2]) if (angleOnArc(s, t)) addPt(arcPoint(s.center, s.r, t));
  }
  return { x0, y0, x1, y1 };
}
