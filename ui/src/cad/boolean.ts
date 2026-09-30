// 形のブーリアン (P7e、prompts/132): 和・差・積。形は外周 + 穴 (どちらも円弧を含む閉じた経路)。
//
// 1. 円弧を細かい弦に分けて polygon-clipping で計算する。弦の点は円ごとの角度の格子 (1 周を等分) に置き、同じ円 (中心・
//    半径がほぼ同じ) の円弧は代表の中心・半径で点を作るので、同じ円の上の円弧どうしは同じ弦になる (ほぼ重なるが一致しない
//    点・弦は polygon-clipping が扱えない)。元の頂点にごく近い格子の点は頂点そのものにする
// 2. 結果の頂点を、元の頂点と曲線どうしの交点 (解析的に求めた正確な点) に合わせる (スナップ)
// 3. 結果の辺ごとに、元のどの曲線 (線分・円弧) の上かを調べ、同じ直線・同じ円の続きを 1 本の辺 (円弧は bulge) に戻す
// 4. 弦の違いでできる細い切れ端 (平均の幅が弦のずれの数倍以下) は捨て、戻した形の面積が弦の形と合うかを確かめる
//
// 結果は外周が反時計回り・穴が時計回り。穴が外周やほかの穴に点で接する形はソルバーが使えないので断る ("touching")。

import polygonClipping, { type MultiPolygon, type Polygon, type Ring } from "polygon-clipping";
import { arcPoint, arcSweep, closestPoint, dist, intersections, pathArea, pathBounds, pathSegs, pointInPath, type ArcSeg, type Seg, type Vec } from "./geom";
import { bulgesOf, isValidPath, packBulges, reversePath, signedArea, type PathData } from "./path";

/** 穴のある形 (外周と穴の経路) */
export interface Shape {
  outer: PathData;
  holes: PathData[];
}

export type BoolOp = "union" | "difference" | "intersection";
export type BoolError = "empty" | "touching" | "failed";

const TAU = 2 * Math.PI;
/** 弦のずれ (形全体の大きさに対する割合) */
const CHORD_ERR = 1e-6;
/** スナップと「曲線の上」の許容差 (同) */
const SNAP_TOL = 1e-4;
/** 同じ直線・同じ円とみなす許容差 (同) */
const SAME_TOL = 1e-9;

export function ringsOf(sh: Shape): PathData[] {
  return [sh.outer, ...sh.holes];
}

function extentOf(shapes: Shape[]): number {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const sh of shapes) {
    const b = pathBounds(sh.outer.polygon, sh.outer.bulges);
    x0 = Math.min(x0, b.x0);
    y0 = Math.min(y0, b.y0);
    x1 = Math.max(x1, b.x1);
    y1 = Math.max(y1, b.y1);
  }
  const e = Math.max(x1 - x0, y1 - y0);
  return Number.isFinite(e) && e > 0 ? e : 1;
}

type Box = [number, number, number, number];

function segBox(s: Seg): Box {
  if (s.kind === "line") return [Math.min(s.a[0], s.b[0]), Math.min(s.a[1], s.b[1]), Math.max(s.a[0], s.b[0]), Math.max(s.a[1], s.b[1])];
  return [s.center[0] - s.r, s.center[1] - s.r, s.center[0] + s.r, s.center[1] + s.r];
}

const boxesMeet = (a: Box, b: Box, tol: number) => !(a[0] > b[2] + tol || b[0] > a[2] + tol || a[1] > b[3] + tol || b[1] > a[3] + tol);

// ---- 弦への分割 ----

interface Circle {
  center: Vec;
  r: number;
}

/** 弦への分け方 (同じ円の代表と、頂点へのスナップを形の集まりで共有する) */
class Discretizer {
  private circles: Circle[] = [];
  private vertex: (q: Vec) => Vec | null;

  constructor(
    shapes: Shape[],
    readonly maxErr: number,
    private tight: number,
  ) {
    this.vertex = snapper(
      shapes.flatMap(ringsOf).flatMap((r) => r.polygon),
      tight,
    );
  }

  private circleOf(s: ArcSeg): Circle {
    for (const c of this.circles) if (dist(c.center, s.center) <= this.tight && Math.abs(c.r - s.r) <= this.tight) return c;
    const c = { center: s.center, r: s.r };
    this.circles.push(c);
    return c;
  }

  /** 円弧の点列: 始点と、円ごとの角度の格子 (1 周の n 等分) のうち円弧の内側の点 (終点は含めない) */
  private arcPoints(s: ArcSeg): Vec[] {
    const { center, r } = this.circleOf(s);
    const dmax = this.maxErr < r ? 2 * Math.acos(1 - this.maxErr / r) : Math.PI / 4;
    const n = Math.max(16, Math.ceil(TAU / dmax));
    const step = TAU / n;
    const sw = arcSweep(s);
    const dir = s.ccw ? 1 : -1;
    const a0 = Math.atan2(s.a[1] - center[1], s.a[0] - center[0]);
    const margin = 1e-6 * step;
    const out: Vec[] = [s.a];
    let k = s.ccw ? Math.floor(a0 / step) + 1 : Math.ceil(a0 / step) - 1;
    for (let guard = 0; guard <= n + 2; guard++, k += dir) {
      const d = (k * step - a0) * dir;
      if (d >= sw - margin) break;
      if (d <= margin) continue;
      // 同じ格子の点は同じ浮動小数点の角度から (別の円弧の同じ点と一致させる)
      const q = arcPoint(center, r, (((k % n) + n) % n) * step);
      out.push(this.vertex(q) ?? q);
    }
    return out;
  }

  ring(p: PathData): Vec[] {
    const out: Vec[] = [];
    for (const s of pathSegs(p.polygon, bulgesOf(p))) {
      if (s.kind === "line") out.push(s.a);
      else out.push(...this.arcPoints(s));
    }
    return out;
  }

  polygon(sh: Shape): Polygon {
    return ringsOf(sh).map((r) => this.ring(r) as Ring);
  }
}

function ringArea(pts: readonly Vec[]): number {
  let s = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const q = pts[(i + 1) % pts.length];
    s += p[0] * q[1] - q[0] * p[1];
  }
  return s / 2;
}

function ringLength(pts: readonly Vec[]): number {
  let s = 0;
  for (let i = 0; i < pts.length; i++) s += dist(pts[i], pts[(i + 1) % pts.length]);
  return s;
}

/** 多角形の集まりの面積 (外周 - 穴) */
function multiArea(mp: MultiPolygon): number {
  let s = 0;
  for (const poly of mp) poly.forEach((ring, k) => (s += (k === 0 ? 1 : -1) * Math.abs(ringArea(ring as Vec[]))));
  return s;
}

// ---- スナップと元の曲線 ----

/** 元の頂点と、別の輪の曲線どうしの交点 */
function specialPoints(shapes: Shape[]): Vec[] {
  const rings = shapes.flatMap(ringsOf).map((r) => pathSegs(r.polygon, bulgesOf(r)));
  const boxes = rings.map((r) => r.map(segBox));
  const out: Vec[] = rings.flatMap((r) => r.map((s) => s.a));
  for (let i = 0; i < rings.length; i++) {
    for (let j = i + 1; j < rings.length; j++) {
      for (let a = 0; a < rings[i].length; a++) {
        for (let b = 0; b < rings[j].length; b++) {
          if (boxesMeet(boxes[i][a], boxes[j][b], 0)) out.push(...intersections(rings[i][a], rings[j][b]));
        }
      }
    }
  }
  return out;
}

/** 点を許容差 tol 以内で最も近い特別な点に合わせる関数 (合わせたら同じ配列の要素を返す) */
function snapper(points: Vec[], tol: number): (q: Vec) => Vec | null {
  const sorted = points.slice().sort((p, q) => p[0] - q[0]);
  return (q) => {
    let lo = 0;
    let hi = sorted.length;
    while (lo < hi) {
      const m = (lo + hi) >> 1;
      if (sorted[m][0] < q[0] - tol) lo = m + 1;
      else hi = m;
    }
    let best: Vec | null = null;
    let bd = tol;
    for (let i = lo; i < sorted.length && sorted[i][0] <= q[0] + tol; i++) {
      const d = dist(sorted[i], q);
      if (d <= bd) {
        bd = d;
        best = sorted[i];
      }
    }
    return best;
  };
}

interface CleanRing {
  pts: Vec[];
  /** 元の頂点・交点に合わせた点か */
  special: boolean[];
}

/** 閉じるための最後の点を除き、スナップし、続けて同じ点と行って戻るだけの点を除く */
function cleanRing(ring: Ring, snap: (q: Vec) => Vec | null, tiny: number): CleanRing | null {
  let pts: Vec[] = [];
  let special: boolean[] = [];
  const n = ring.length > 1 && ring[0][0] === ring[ring.length - 1][0] && ring[0][1] === ring[ring.length - 1][1] ? ring.length - 1 : ring.length;
  for (let i = 0; i < n; i++) {
    const q: Vec = [ring[i][0], ring[i][1]];
    const s = snap(q);
    const p = s ?? q;
    if (pts.length && dist(pts[pts.length - 1], p) <= tiny) {
      // 続けて同じ点: 特別な点を残す
      if (s) {
        pts[pts.length - 1] = p;
        special[special.length - 1] = true;
      }
      continue;
    }
    pts.push(p);
    special.push(s !== null);
  }
  let changed = true;
  while (changed && pts.length >= 3) {
    changed = false;
    const m = pts.length;
    for (let i = 0; i < m; i++) {
      const prev = (i - 1 + m) % m;
      const next = (i + 1) % m;
      if (dist(pts[prev], pts[i]) <= tiny) {
        pts.splice(i, 1);
        special.splice(i, 1);
        changed = true;
        break;
      }
      if (dist(pts[prev], pts[next]) <= tiny) {
        // 行って戻るだけ (i と next を除く)
        const drop = new Set([i, next]);
        pts = pts.filter((_, k) => !drop.has(k));
        special = special.filter((_, k) => !drop.has(k));
        changed = true;
        break;
      }
    }
  }
  return pts.length >= 3 ? { pts, special } : null;
}

interface Src {
  seg: Seg;
  box: Box;
}

/**
 * 辺 p → q が乗っている元の曲線 (両端と中点で判定)。円弧は弦 1 本分 (たわみが弦のずれ程度) の辺だけ
 * (長い直線の辺の両端が同じ円の上にあっても円弧とみなさない)。見つからなければ null (直線として扱う)
 */
function sourceOf(p: Vec, q: Vec, srcs: Src[], tol: number, maxErr: number): Seg | null {
  const mid: Vec = [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2];
  const half = dist(p, q) / 2;
  let best: Seg | null = null;
  let bestScore = Infinity;
  for (const { seg: s, box } of srcs) {
    if (mid[0] < box[0] - tol - half || mid[0] > box[2] + tol + half || mid[1] < box[1] - tol - half || mid[1] > box[3] + tol + half) continue;
    if (closestPoint(s, p).dist > tol || closestPoint(s, q).dist > tol) continue;
    let score: number;
    if (s.kind === "line") {
      score = closestPoint(s, mid).dist;
    } else {
      const sag = s.r - Math.sqrt(Math.max(0, s.r * s.r - half * half));
      if (sag > 4 * maxErr) continue;
      score = Math.abs(closestPoint(s, mid).dist - sag);
    }
    if (score <= tol && score < bestScore) {
      bestScore = score;
      best = s;
    }
  }
  return best;
}

interface OutEdge {
  p: Vec;
  q: Vec;
  arc: ArcSeg | null;
}

/** 中心 c のまわりに p から q へ回る符号付きの角 */
function turn(c: Vec, p: Vec, q: Vec): number {
  const ax = p[0] - c[0];
  const ay = p[1] - c[1];
  const bx = q[0] - c[0];
  const by = q[1] - c[1];
  return Math.atan2(ax * by - ay * bx, ax * bx + ay * by);
}

function sameCarrier(e1: OutEdge, e2: OutEdge, tight: number): boolean {
  if (e1.arc && e2.arc) {
    return (
      dist(e1.arc.center, e2.arc.center) <= tight &&
      Math.abs(e1.arc.r - e2.arc.r) <= tight &&
      Math.sign(turn(e1.arc.center, e1.p, e1.q)) === Math.sign(turn(e2.arc.center, e2.p, e2.q))
    );
  }
  if (e1.arc || e2.arc) return false;
  // 直線どうし: e1.q が直線 e1.p → e2.q の上にあり、同じ向きに進む
  const L = dist(e1.p, e2.q);
  if (L === 0) return false;
  const vx = e2.q[0] - e1.p[0];
  const vy = e2.q[1] - e1.p[1];
  const off = Math.abs(vx * (e1.q[1] - e1.p[1]) - vy * (e1.q[0] - e1.p[0])) / L;
  const along = (e1.q[0] - e1.p[0]) * (e2.q[0] - e2.p[0]) + (e1.q[1] - e1.p[1]) * (e2.q[1] - e2.p[1]);
  return off <= tight && along > 0;
}

/** 結果の輪の点列を、元の曲線に沿った経路 (同じ直線・同じ円の続きは 1 本の辺) に戻す */
function rebuildRing(ring: CleanRing, srcs: Src[], tol: number, tight: number, maxErr: number, snap: (q: Vec) => Vec | null): PathData | null {
  const { pts, special } = ring;
  const n = pts.length;
  const edges: OutEdge[] = pts.map((p, i) => {
    const q = pts[(i + 1) % n];
    const s = sourceOf(p, q, srcs, tol, maxErr);
    return { p, q, arc: s?.kind === "arc" ? s : null };
  });
  let start = -1;
  for (let i = 0; i < n; i++) {
    if (!sameCarrier(edges[(i - 1 + n) % n], edges[i], tight)) {
      start = i;
      break;
    }
  }
  if (start < 0) {
    // 全体が 1 つの円: 元の頂点 (無ければ角度 0 の点) とその反対側の 2 つの半円
    const a = edges[0].arc;
    if (!a) return null;
    const total = edges.reduce((s, e) => s + turn(a.center, e.p, e.q), 0);
    const k = special.indexOf(true);
    const p0 = k >= 0 ? pts[k] : arcPoint(a.center, a.r, 0);
    const t0 = Math.atan2(p0[1] - a.center[1], p0[0] - a.center[0]);
    const anti = arcPoint(a.center, a.r, t0 + Math.PI);
    const p1 = snap(anti) ?? anti;
    const b = total >= 0 ? 1 : -1;
    return { polygon: [p0, p1], bulges: [b, b] };
  }
  const runs: OutEdge[][] = [];
  for (let k = 0; k < n; k++) {
    const e = edges[(start + k) % n];
    const last = runs[runs.length - 1];
    if (last && sameCarrier(last[last.length - 1], e, tight)) last.push(e);
    else runs.push([e]);
  }
  const polygon: Vec[] = [];
  const bulges: number[] = [];
  for (const run of runs) {
    const a = run[0].arc;
    if (!a) {
      polygon.push(run[0].p);
      bulges.push(0);
      continue;
    }
    const sweep = run.reduce((s, e) => s + turn(a.center, e.p, e.q), 0);
    if (Math.abs(sweep) >= TAU - 1e-9) return null;
    polygon.push(run[0].p);
    bulges.push(Math.tan(sweep / 4));
  }
  const path: PathData = { polygon, bulges: packBulges(bulges) };
  // 戻した形の面積が弦の形と合うか (スナップで動いた分と弦のずれを許す)
  const err = Math.abs(signedArea(path) - ringArea(pts));
  if (!isValidPath(path) || err > (tol + 4 * maxErr) * ringLength(pts)) return null;
  return path;
}

// ---- 形の検査 ----

/** 経路の辺どうしが、共有する頂点のほかで交わるか接するか */
function crosses(a: Seg[], b: Seg[], same: boolean, tol: number): boolean {
  const boxA = a.map(segBox);
  const boxB = b.map(segBox);
  for (let i = 0; i < a.length; i++) {
    for (let j = same ? i + 1 : 0; j < b.length; j++) {
      if (!boxesMeet(boxA[i], boxB[j], tol)) continue;
      const shared: Vec[] = [];
      if (same) {
        // 同じ輪で隣り合う辺は共有する頂点で交わってよい
        if (j === i + 1) shared.push(a[i].b);
        if (i === 0 && j === a.length - 1) shared.push(a[i].a);
      }
      for (const x of intersections(a[i], b[j])) if (!shared.some((s) => dist(s, x) <= tol)) return true;
    }
  }
  return false;
}

/**
 * 形として使えるか: 各輪が経路として正しく、自分と交わらず、穴は外周の内側にあって外周・ほかの穴と交わらず
 * 接せず、ほかの穴の中にない。使えなければ理由
 */
export function shapeProblem(sh: Shape): "invalid" | "touching" | null {
  const rings = ringsOf(sh);
  if (!rings.every(isValidPath)) return "invalid";
  const tol = SAME_TOL * extentOf([sh]);
  const segs = rings.map((r) => pathSegs(r.polygon, bulgesOf(r)));
  for (const s of segs) if (crosses(s, s, true, tol)) return "touching";
  for (let k = 1; k < rings.length; k++) {
    if (!rings[k].polygon.every((q) => pointInPath(q, sh.outer.polygon, sh.outer.bulges))) return "invalid";
  }
  for (let i = 0; i < rings.length; i++) for (let j = i + 1; j < rings.length; j++) if (crosses(segs[i], segs[j], false, tol)) return "touching";
  for (let i = 1; i < rings.length; i++) {
    for (let j = 1; j < rings.length; j++) {
      if (i !== j && pointInPath(rings[i].polygon[0], rings[j].polygon, rings[j].bulges)) return "invalid";
    }
  }
  return null;
}

/** 形の面積 (外周 - 穴) */
export function shapeArea(sh: Shape): number {
  return Math.abs(signedArea(sh.outer)) - sh.holes.reduce((s, h) => s + Math.abs(signedArea(h)), 0);
}

/** 2 つの形の重なる面積 (円弧は弦に分けて) */
export function overlapArea(a: Shape, b: Shape): number {
  const ext = extentOf([a, b]);
  const disc = new Discretizer([a, b], CHORD_ERR * ext, SAME_TOL * ext);
  try {
    return multiArea(polygonClipping.intersection(disc.polygon(a), disc.polygon(b)));
  } catch {
    return 0;
  }
}

function orient(p: PathData, ccw: boolean): PathData {
  return signedArea(p) > 0 === ccw ? p : reversePath(p);
}

/**
 * ブーリアン: 和 (target ∪ tools)・差 (target - tools)・積 (target ∩ tools)。
 * 結果は polygon-clipping の順の形 (外周は反時計回り、穴は時計回り)
 */
export function booleanShapes(op: BoolOp, target: Shape, tools: Shape[]): { shapes: Shape[] } | { error: BoolError } {
  const all = [target, ...tools];
  const ext = extentOf(all);
  const maxErr = CHORD_ERR * ext;
  const tol = SNAP_TOL * ext;
  const tight = SAME_TOL * ext;
  let out: MultiPolygon;
  try {
    const disc = new Discretizer(all, maxErr, tight);
    const a = disc.polygon(target);
    const bs = tools.map((t) => disc.polygon(t));
    out = op === "union" ? polygonClipping.union(a, ...bs) : op === "difference" ? polygonClipping.difference(a, ...bs) : polygonClipping.intersection(a, ...bs);
  } catch {
    return { error: "failed" };
  }
  const snap = snapper(specialPoints(all), tol);
  const srcs: Src[] = all
    .flatMap(ringsOf)
    .flatMap((r) => pathSegs(r.polygon, bulgesOf(r)))
    .map((seg) => ({ seg, box: segBox(seg) }));
  // 平均の幅 (2 × 面積 / 周長) が弦のずれの数倍以下の輪は、弦の違いでできた切れ端
  const sliver = (r: CleanRing) => (2 * Math.abs(ringArea(r.pts))) / ringLength(r.pts) <= 8 * maxErr;
  const shapes: Shape[] = [];
  for (const poly of out) {
    const rings: PathData[] = [];
    for (let k = 0; k < poly.length; k++) {
      const clean = cleanRing(poly[k], snap, tight);
      if (!clean || sliver(clean)) {
        if (k === 0) break;
        continue;
      }
      const path = rebuildRing(clean, srcs, tol, tight, maxErr, snap);
      if (!path) return { error: "failed" };
      rings.push(path);
    }
    if (rings.length === 0) continue;
    const [outer, ...holes] = rings;
    shapes.push({ outer: orient(outer, true), holes: holes.map((h) => orient(h, false)) });
  }
  if (shapes.length === 0) return { error: "empty" };
  for (const sh of shapes) {
    const problem = shapeProblem(sh);
    if (problem) return { error: problem === "touching" ? "touching" : "failed" };
  }
  return { shapes };
}

/** 形の面積が 0 でない (弦の形で) か */
export function hasArea(sh: Shape): boolean {
  return Math.abs(pathArea(sh.outer.polygon, sh.outer.bulges)) > 0;
}
