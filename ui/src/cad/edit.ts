// 形の編集の幾何 (P7d): フィレット・面取り (直線どうしの角)、オフセット (線分・円弧・円・経路)、トリム (交点で
// 区切った一部を消す)、延長 (端を次の交点まで)。文書を変えない純粋な関数で、使えないときは理由のキーを返す。

import { splitAtIntersections, toleranceOf } from "./arrangement";
import {
  arcSweep,
  bulgeOf,
  closestPoint,
  dist,
  intersections,
  normAngle,
  paramOf,
  pathSegs,
  segFromBulge,
  segLength,
  subSeg,
  type ArcSeg,
  type LineSeg,
  type Seg,
  type Vec,
} from "./geom";

export type EditError = "notStraight" | "collinear" | "tooLarge" | "tooSmall" | "noCutter" | "noTarget" | "selfIntersect";
export type EditResult<T> = { ok: true; value: T } | { ok: false; error: EditError };

/** 点と辺ごとの bulge (開いていれば辺は n - 1 本、model/sketch.ts の EditPath と同じ形) */
export interface PathPts {
  points: Vec[];
  bulges: number[];
  closed: boolean;
}

const sub = (p: Vec, q: Vec): Vec => [p[0] - q[0], p[1] - q[1]];
const add = (p: Vec, q: Vec): Vec => [p[0] + q[0], p[1] + q[1]];
const mul = (p: Vec, k: number): Vec => [p[0] * k, p[1] * k];
const cross = (p: Vec, q: Vec) => p[0] * q[1] - p[1] * q[0];
const unit = (p: Vec): Vec => {
  const L = Math.hypot(p[0], p[1]);
  return [p[0] / L, p[1] / L];
};

function edgeCount(p: PathPts): number {
  return p.closed ? p.points.length : p.points.length - 1;
}

function segOf(p: PathPts, i: number): Seg {
  return segFromBulge(p.points[i], p.points[(i + 1) % p.points.length], p.bulges[i] ?? 0);
}

// ---- フィレット・面取り ----

interface Corner {
  prev: number;
  V: Vec;
  u1: Vec;
  u2: Vec;
  l1: number;
  l2: number;
  /** 内角 (0〜π) */
  phi: number;
  /** 左に曲がる (反時計回り) なら 1 */
  turn: number;
}

/** 頂点 i の角 (前後の辺がどちらも直線のときだけ) */
function cornerAt(p: PathPts, i: number): EditResult<Corner> {
  const n = p.points.length;
  if (!p.closed && (i === 0 || i === n - 1)) return { ok: false, error: "notStraight" };
  const prev = (i - 1 + n) % n;
  if ((p.bulges[prev] ?? 0) !== 0 || (p.bulges[i] ?? 0) !== 0) return { ok: false, error: "notStraight" };
  const V = p.points[i];
  const P = p.points[prev];
  const N = p.points[(i + 1) % n];
  const l1 = dist(P, V);
  const l2 = dist(N, V);
  if (l1 === 0 || l2 === 0) return { ok: false, error: "collinear" };
  const u1 = unit(sub(P, V));
  const u2 = unit(sub(N, V));
  const c = Math.min(1, Math.max(-1, u1[0] * u2[0] + u1[1] * u2[1]));
  const phi = Math.acos(c);
  const tc = cross(sub(V, P), sub(N, V));
  if (phi > Math.PI - 1e-9 || phi < 1e-9 || Math.abs(tc) <= 1e-15 * l1 * l2) return { ok: false, error: "collinear" };
  return { ok: true, value: { prev, V, u1, u2, l1, l2, phi, turn: tc > 0 ? 1 : -1 } };
}

/** 頂点 i を 2 点 T1・T2 で置き換え、T1 → T2 の辺を bulge にする */
function replaceCorner(p: PathPts, i: number, T1: Vec, T2: Vec, bulge: number): PathPts {
  const points = [...p.points.slice(0, i), T1, T2, ...p.points.slice(i + 1)];
  const bulges = [...p.bulges.slice(0, i), bulge, ...p.bulges.slice(i)];
  return { points, bulges: bulges.slice(0, p.closed ? points.length : points.length - 1), closed: p.closed };
}

/**
 * 頂点 i の角を半径 r の円弧で丸める (前後の辺が直線のとき)。接点までの長さ r/tan(φ/2) が辺より長ければ tooLarge。
 * 新しい頂点は i (前の辺の上) と i + 1 (次の辺の上)、その間の辺 i が円弧
 */
export function filletCorner(p: PathPts, i: number, r: number): EditResult<PathPts> {
  if (!(r > 0)) return { ok: false, error: "tooSmall" };
  const c = cornerAt(p, i);
  if (!c.ok) return c;
  const { V, u1, u2, l1, l2, phi, turn } = c.value;
  const t = r / Math.tan(phi / 2);
  if (t >= Math.min(l1, l2) * (1 - 1e-12)) return { ok: false, error: "tooLarge" };
  const T1 = add(V, mul(u1, t));
  const T2 = add(V, mul(u2, t));
  return { ok: true, value: replaceCorner(p, i, T1, T2, turn * Math.tan((Math.PI - phi) / 4)) };
}

/** 頂点 i の角を、両側の辺を d ずつ切り取る直線で面取りする */
export function chamferCorner(p: PathPts, i: number, d: number): EditResult<PathPts> {
  if (!(d > 0)) return { ok: false, error: "tooSmall" };
  const c = cornerAt(p, i);
  if (!c.ok) return c;
  const { V, u1, u2, l1, l2 } = c.value;
  if (d >= Math.min(l1, l2) * (1 - 1e-12)) return { ok: false, error: "tooLarge" };
  return { ok: true, value: replaceCorner(p, i, add(V, mul(u1, d)), add(V, mul(u2, d)), 0) };
}

/**
 * 2 本の直線 (スケッチの線) のフィレット: クリックした側 (pick1・pick2) を残して、交点の近くを半径 r の円弧で
 * つなぐ。r = 0 なら角 (交点) までそろえる。戻り値は短くした 2 本と円弧 (r = 0 なら null)
 */
export function filletLines(l1: LineSeg, pick1: Vec, l2: LineSeg, pick2: Vec, r: number): EditResult<{ line1: LineSeg; line2: LineSeg; arc: { a: Vec; b: Vec; bulge: number } | null }> {
  const X = intersections(l1, l2, true)[0];
  if (!X) return { ok: false, error: "collinear" };
  // 残す側: 交点から pick の向き
  const side = (l: LineSeg, pick: Vec): { dir: Vec; far: Vec } => {
    const d = unit(sub(l.b, l.a));
    const s = (pick[0] - X[0]) * d[0] + (pick[1] - X[1]) * d[1];
    const dir: Vec = s >= 0 ? d : mul(d, -1);
    // 残す側の遠い端
    const ta = (l.a[0] - X[0]) * dir[0] + (l.a[1] - X[1]) * dir[1];
    const tb = (l.b[0] - X[0]) * dir[0] + (l.b[1] - X[1]) * dir[1];
    return { dir, far: ta >= tb ? l.a : l.b };
  };
  const s1 = side(l1, pick1);
  const s2 = side(l2, pick2);
  const c = Math.min(1, Math.max(-1, s1.dir[0] * s2.dir[0] + s1.dir[1] * s2.dir[1]));
  const phi = Math.acos(c);
  if (phi < 1e-9 || phi > Math.PI - 1e-9) return { ok: false, error: "collinear" };
  const t = r > 0 ? r / Math.tan(phi / 2) : 0;
  const len1 = (s1.far[0] - X[0]) * s1.dir[0] + (s1.far[1] - X[1]) * s1.dir[1];
  const len2 = (s2.far[0] - X[0]) * s2.dir[0] + (s2.far[1] - X[1]) * s2.dir[1];
  if (t >= len1 || t >= len2) return { ok: false, error: "tooLarge" };
  const T1 = add(X, mul(s1.dir, t));
  const T2 = add(X, mul(s2.dir, t));
  // 線 1 の遠い端 → T1 → (円弧) → T2 → 線 2 の遠い端 と進む向きで曲がる向きを決める
  const turn = cross(sub(T1, s1.far), sub(s2.far, T2)) > 0 ? 1 : -1;
  return {
    ok: true,
    value: {
      line1: { kind: "line", a: s1.far, b: T1 },
      line2: { kind: "line", a: T2, b: s2.far },
      arc: r > 0 ? { a: T1, b: T2, bulge: turn * Math.tan((Math.PI - phi) / 4) } : null,
    },
  };
}

// ---- オフセット ----

/** 曲線を左 (進む向きの左) へ d だけずらす。円弧の半径が 0 以下になれば null */
export function offsetSeg(s: Seg, d: number): Seg | null {
  if (s.kind === "line") {
    const n: Vec = unit([-(s.b[1] - s.a[1]), s.b[0] - s.a[0]]);
    return { kind: "line", a: add(s.a, mul(n, d)), b: add(s.b, mul(n, d)) };
  }
  // 反時計回りの円弧の左は中心の側
  const r = s.ccw ? s.r - d : s.r + d;
  if (!(r > 0)) return null;
  const a: Vec = [s.center[0] + r * Math.cos(s.a0), s.center[1] + r * Math.sin(s.a0)];
  const b: Vec = [s.center[0] + r * Math.cos(s.a1), s.center[1] + r * Math.sin(s.a1)];
  return { kind: "arc", center: s.center, r, a0: s.a0, a1: s.a1, ccw: s.ccw, a, b };
}

/** 曲線の向きの単位ベクトル (端 a から、または端 b での進む向き) */
function tangentDir(s: Seg, at: "a" | "b"): Vec {
  if (s.kind === "line") return unit(sub(s.b, s.a));
  const t = at === "a" ? s.a0 : s.a1;
  return s.ccw ? [-Math.sin(t), Math.cos(t)] : [Math.sin(t), -Math.cos(t)];
}

/**
 * 経路のオフセット (左へ d、負なら右へ)。外側の角は直線どうしなら延ばして交わらせ (マイター)、円弧が絡めば元の
 * 頂点を中心の円弧でつなぐ。内側の角は交点で切りそろえる。辺が消える・自己交差するほど大きければ tooLarge
 */
export function offsetPath(p: PathPts, d: number): EditResult<PathPts> {
  const m = edgeCount(p);
  if (m < 1 || d === 0) return { ok: false, error: "tooSmall" };
  const segs: Seg[] = [];
  for (let i = 0; i < m; i++) {
    const o = offsetSeg(segOf(p, i), d);
    if (!o) return { ok: false, error: "tooLarge" };
    segs.push(o);
  }
  const outPts: Vec[] = [];
  const outBulges: number[] = [];
  // 各辺の始点と終点 (角の処理で決まる)
  const starts: Vec[] = segs.map((s) => s.a);
  const ends: Vec[] = segs.map((s) => s.b);
  /** 角 k (辺 k-1 と辺 k の間) の丸めの円弧 (外側の角で円弧が絡むとき) */
  const joinArc: (number | null)[] = new Array(m).fill(null);
  const scale = Math.max(...p.points.map(([x, y]) => Math.max(Math.abs(x), Math.abs(y))), Math.abs(d), 1e-12);
  const eps = 1e-10 * scale;
  const corners = p.closed ? m : m - 1;
  for (let c = 0; c < corners; c++) {
    const i0 = p.closed ? (c - 1 + m) % m : c;
    const i1 = p.closed ? c : c + 1;
    const s0 = segs[i0];
    const s1 = segs[i1];
    const V = p.points[i1];
    if (dist(s0.b, s1.a) <= eps) continue; // なめらかにつながる
    const t0 = tangentDir(segOf(p, i0), "b");
    const t1 = tangentDir(segOf(p, i1), "a");
    const turn = cross(t0, t1);
    // 左へずらすと、右に曲がる角 (turn < 0) は離れる (外側)、左に曲がる角は重なる (内側)
    const outside = d > 0 ? turn < 0 : turn > 0;
    if (outside) {
      if (s0.kind === "line" && s1.kind === "line") {
        const X = intersections(s0, s1, true)[0];
        if (!X) return { ok: false, error: "tooLarge" };
        ends[i0] = X;
        starts[i1] = X;
      } else {
        // 元の頂点を中心、半径 |d| の円弧でつなぐ (s0 の終点 → s1 の始点)
        const a0 = Math.atan2(s0.b[1] - V[1], s0.b[0] - V[0]);
        const a1 = Math.atan2(s1.a[1] - V[1], s1.a[0] - V[0]);
        const ccw = turn > 0;
        const sweep = ccw ? normAngle(a1 - a0) : normAngle(a0 - a1);
        joinArc[i1] = (ccw ? 1 : -1) * Math.tan(sweep / 4);
      }
    } else {
      // 内側: 2 本の交点で切りそろえる (近い方)
      const xs = intersections(s0, s1, false).concat(intersections(s0, s1, true));
      if (xs.length === 0) return { ok: false, error: "tooLarge" };
      let X = xs[0];
      for (const q of xs) if (dist(q, V) < dist(X, V)) X = q;
      ends[i0] = X;
      starts[i1] = X;
    }
  }
  // 切りそろえた辺が向きを変えていないか (消えていないか)
  for (let i = 0; i < m; i++) {
    const s = segs[i];
    const from = starts[i];
    const to = ends[i];
    if (s.kind === "line") {
      const dir = sub(s.b, s.a);
      if ((to[0] - from[0]) * dir[0] + (to[1] - from[1]) * dir[1] <= eps * Math.hypot(dir[0], dir[1])) return { ok: false, error: "tooLarge" };
    } else if (dist(from, to) <= eps) return { ok: false, error: "tooLarge" };
  }
  for (let i = 0; i < m; i++) {
    const s = segs[i];
    if (joinArc[i] !== null) {
      // 前の辺の終点 → この辺の始点 の丸め
      const prev = (i - 1 + m) % m;
      outPts.push(ends[prev]);
      outBulges.push(joinArc[i]!);
    }
    outPts.push(starts[i]);
    if (s.kind === "line") outBulges.push(0);
    else {
      // 始点・終点を変えた円弧 (同じ円・同じ向き) の bulge
      const a0 = Math.atan2(starts[i][1] - s.center[1], starts[i][0] - s.center[0]);
      const a1 = Math.atan2(ends[i][1] - s.center[1], ends[i][0] - s.center[0]);
      let sweep = s.ccw ? normAngle(a1 - a0) : normAngle(a0 - a1);
      if (sweep === 0) sweep = arcSweep(s);
      outBulges.push((s.ccw ? 1 : -1) * Math.tan(sweep / 4));
    }
    if (!p.closed && i === m - 1) outPts.push(ends[i]);
  }
  const res: PathPts = { points: outPts, bulges: outBulges.slice(0, p.closed ? outPts.length : outPts.length - 1), closed: p.closed };
  // 自己交差しない (隣り合わない辺どうしが交わらない)
  const rs = pathSegs(res.points, res.closed ? res.bulges : [...res.bulges, 0]).slice(0, edgeCount(res));
  const tol = toleranceOf(rs) * 100;
  for (let i = 0; i < rs.length; i++) {
    for (let j = i + 2; j < rs.length; j++) {
      if (res.closed && i === 0 && j === rs.length - 1) continue;
      for (const q of intersections(rs[i], rs[j])) {
        const shared = [rs[i].a, rs[i].b].some((e) => dist(e, q) <= tol) && [rs[j].a, rs[j].b].some((e) => dist(e, q) <= tol);
        if (!shared) return { ok: false, error: "selfIntersect" };
      }
    }
  }
  return { ok: true, value: res };
}

// ---- トリム・延長 ----

/** トリムの区切り: 目標の曲線を相手との交点で分けた部分 (pieces)、交点から交点までのまとまり (groups)、クリックに
 * 最も近いまとまり (best) */
function trimSplit(target: Seg[], closed: boolean, cutters: Seg[], click: Vec): EditResult<{ pieces: Seg[]; groups: number[][]; best: number }> {
  if (target.length === 0) return { ok: false, error: "noTarget" };
  const tol = toleranceOf([...target, ...cutters]);
  // 目標の各曲線を相手との交点で分ける (目標どうしのつなぎ目は端点なので分からない)
  const pieces = splitAtIntersections([...target, ...cutters], tol)
    .slice(0, target.length)
    .flat();
  const onCutter = (q: Vec) => cutters.some((c) => closestPoint(c, q).dist <= tol * 10);
  // 区切り (交点から交点まで、元の頂点はまたぐ)。閉じた形は最初の交点から回す
  let order = pieces.map((_, k) => k);
  if (closed) {
    const first = pieces.findIndex((pc) => onCutter(pc.a));
    if (first < 0) return { ok: false, error: "noCutter" };
    order = [...order.slice(first), ...order.slice(0, first)];
  }
  const groups: number[][] = [];
  for (const k of order) {
    if (groups.length === 0 || onCutter(pieces[k].a)) groups.push([k]);
    else groups[groups.length - 1].push(k);
  }
  // 開いた形は 1 つでも交点があれば 2 つ以上、閉じた形は 2 つ以上の交点が要る
  if (groups.length < 2) return { ok: false, error: "noCutter" };
  let best = 0;
  let bestD = Infinity;
  groups.forEach((g, gi) => {
    for (const k of g) {
      const d = closestPoint(pieces[k], click).dist;
      if (d < bestD) {
        bestD = d;
        best = gi;
      }
    }
  });
  return { ok: true, value: { pieces, groups, best } };
}

/**
 * トリム: 曲線 (1 つの形の線分・円弧の並び) を、ほかの曲線 (cutters) との交点で区切り、クリックした点に最も近い
 * 区切りを消す。残りを「つながった曲線の並び」ごとに返す (閉じた形は開いた 1 本、開いた形は前後の 2 本まで)
 */
export function trimAt(target: Seg[], closed: boolean, cutters: Seg[], click: Vec): EditResult<Seg[][]> {
  const r = trimSplit(target, closed, cutters, click);
  if (!r.ok) return r;
  const { pieces, groups, best } = r.value;
  // 閉じた形: 消した区切りの次から回ると 1 本。開いた形: 前と後ろ
  const chains = closed
    ? [[...groups.slice(best + 1), ...groups.slice(0, best)].flat()]
    : [groups.slice(0, best).flat(), groups.slice(best + 1).flat()].filter((c) => c.length > 0);
  return { ok: true, value: chains.map((c) => c.map((k) => pieces[k])) };
}

/** トリムで消える区切り (カーソルの下を先に見せる) */
export function trimRemovedAt(target: Seg[], closed: boolean, cutters: Seg[], click: Vec): Seg[] | null {
  const r = trimSplit(target, closed, cutters, click);
  return r.ok ? r.value.groups[r.value.best].map((k) => r.value.pieces[k]) : null;
}

/**
 * 延長: 曲線の端 (end) を、その先で最初に交わる曲線 (boundaries) まで延ばす。直線はまっすぐ、円弧は同じ円に沿って
 * (もう一方の端を越えない)。交わるものが無ければ noCutter
 */
export function extendEnd(s: Seg, end: "a" | "b", boundaries: Seg[]): EditResult<Seg> {
  const tol = toleranceOf([s, ...boundaries]);
  if (s.kind === "line") {
    const from = end === "b" ? s.b : s.a;
    const other = end === "b" ? s.a : s.b;
    const dir = unit(sub(from, other));
    let far = 0;
    for (const b of boundaries) for (const q of [b.a, b.b]) far = Math.max(far, dist(q, from));
    const ray: LineSeg = { kind: "line", a: from, b: add(from, mul(dir, 2 * far + 1)) };
    let best: Vec | null = null;
    let bestT = Infinity;
    for (const b of boundaries) {
      for (const q of intersections(ray, b)) {
        const t = (q[0] - from[0]) * dir[0] + (q[1] - from[1]) * dir[1];
        if (t > tol && t < bestT) {
          bestT = t;
          best = q;
        }
      }
    }
    if (!best) return { ok: false, error: "noCutter" };
    return { ok: true, value: end === "b" ? { kind: "line", a: s.a, b: best } : { kind: "line", a: best, b: s.b } };
  }
  // 円弧: 同じ円との交点のうち、端から円弧の外へ進んで最初のもの
  const circle = { kind: "circle" as const, center: s.center, r: s.r };
  const sweep = arcSweep(s);
  let best: Vec | null = null;
  let bestAng = Infinity;
  for (const b of boundaries) {
    for (const q of intersections(circle, b)) {
      const t = Math.atan2(q[1] - s.center[1], q[0] - s.center[0]);
      // 延ばす向き: 終点 b からは進む向き、始点 a からは戻る向き
      const ang = end === "b" ? (s.ccw ? normAngle(t - s.a1) : normAngle(s.a1 - t)) : s.ccw ? normAngle(s.a0 - t) : normAngle(t - s.a0);
      if (ang > 1e-9 && ang < 2 * Math.PI - sweep - 1e-9 && ang < bestAng) {
        bestAng = ang;
        best = q;
      }
    }
  }
  if (!best) return { ok: false, error: "noCutter" };
  const total = sweep + bestAng;
  const bulge = (s.ccw ? 1 : -1) * Math.tan(total / 4);
  return { ok: true, value: end === "b" ? segFromBulge(s.a, best, bulge) : segFromBulge(best, s.b, bulge) };
}

/** 曲線の並び (つながっている) を、点と bulge の並びに */
export function chainToPath(chain: Seg[]): PathPts {
  const points: Vec[] = chain.map((s) => s.a);
  points.push(chain[chain.length - 1].b);
  return { points, bulges: chain.map((s) => bulgeOf(s)), closed: false };
}

/** 曲線上の点 q の近い方の端 */
export function nearerEnd(s: Seg, q: Vec): "a" | "b" {
  return dist(q, s.a) <= dist(q, s.b) ? "a" : "b";
}

export { paramOf, segLength, subSeg };
export type { ArcSeg };

/** つながった曲線の並びで、同じ直線の続きと、同じ円・同じ向きの続きの円弧を 1 本にまとめる */
export function simplifyChain(chain: Seg[]): Seg[] {
  const out: Seg[] = [];
  for (const s of chain) {
    const last = out[out.length - 1];
    if (last && last.kind === "line" && s.kind === "line") {
      const d1 = sub(last.b, last.a);
      const d2 = sub(s.b, s.a);
      const l1 = Math.hypot(d1[0], d1[1]);
      const l2 = Math.hypot(d2[0], d2[1]);
      if (Math.abs(cross(d1, d2)) <= 1e-12 * l1 * l2 && d1[0] * d2[0] + d1[1] * d2[1] > 0) {
        out[out.length - 1] = { kind: "line", a: last.a, b: s.b };
        continue;
      }
    }
    if (last && last.kind === "arc" && s.kind === "arc") {
      const tol = 1e-9 * Math.max(last.r, s.r);
      const same = last.ccw === s.ccw && Math.abs(last.r - s.r) <= tol && dist(last.center, s.center) <= tol;
      const total = arcSweep(last) + arcSweep(s);
      if (same && total < 2 * Math.PI - 1e-9) {
        out[out.length - 1] = segFromBulge(last.a, s.b, (last.ccw ? 1 : -1) * Math.tan(total / 4));
        continue;
      }
    }
    out.push(s);
  }
  return out;
}
