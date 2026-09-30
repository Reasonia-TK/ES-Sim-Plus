// 平面の配置 (arrangement): 線分・円弧の集まりをお互いの交点で分け、点を囲む最小の閉じた輪郭 (面) を探す
// (P7b「囲まれた所から領域を作る」)。分けた曲線はトリム (P7d) にも使う。
//
// 面のたどり方: 半辺 (向きのある辺) の左に面があるとして、頂点に着いたら「戻る向きのすぐ時計回り」の半辺へ進む
// (最も左に曲がる)。向きは接線の角度で並べ、同じ角度なら曲率 (左に曲がるほど反時計回り側) で並べる。
// 点から斜めの半直線を出して最初に当たる辺から始め、一周して反時計回り (面積が正) なら囲まれた面。
// 端が他とつながっていない辺 (ひげ) は先に取り除く。面の中の島 (穴) は輪郭に含めない (穴は P7e)。

import {
  arcSweep,
  bulgeOf,
  closestPoint,
  intersections,
  paramOf,
  pathArea,
  pointAt,
  pointInPath,
  segFromBulge,
  segLength,
  subSeg,
  tangentAt,
  type ArcSeg,
  type Seg,
  type Vec,
} from "./geom";
import type { PathData } from "./path";

/** 図形の大きさに対する許容差 (最小 1e-12 m) */
export function toleranceOf(segs: Seg[]): number {
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of segs) {
    for (const [x, y] of [s.a, s.b]) {
      lo = Math.min(lo, x, y);
      hi = Math.max(hi, x, y);
    }
  }
  return Number.isFinite(hi - lo) ? Math.max(1e-12, (hi - lo) * 1e-9) : 1e-12;
}

function bbox(s: Seg): [number, number, number, number] {
  if (s.kind === "line") return [Math.min(s.a[0], s.b[0]), Math.min(s.a[1], s.b[1]), Math.max(s.a[0], s.b[0]), Math.max(s.a[1], s.b[1])];
  // 円弧は外接する円の箱で十分 (交差の候補を絞るだけ)
  return [s.center[0] - s.r, s.center[1] - s.r, s.center[0] + s.r, s.center[1] + s.r];
}

const near = (p: Vec, q: Vec, tol: number) => Math.hypot(p[0] - q[0], p[1] - q[1]) <= tol;

/**
 * 各曲線を、ほかの曲線との交点と、ほかの曲線の端点が乗る所 (T 字・重なり) で分ける。
 * 戻り値は入力の順に、その曲線の部分の並び (元の向き)。長さが許容差以下の部分は除く
 */
export function splitAtIntersections(segs: Seg[], tol = toleranceOf(segs)): Seg[][] {
  const cuts: Vec[][] = segs.map(() => []);
  const boxes = segs.map(bbox);
  for (let i = 0; i < segs.length; i++) {
    for (let j = i + 1; j < segs.length; j++) {
      const [ax0, ay0, ax1, ay1] = boxes[i];
      const [bx0, by0, bx1, by1] = boxes[j];
      if (ax0 > bx1 + tol || bx0 > ax1 + tol || ay0 > by1 + tol || by0 > ay1 + tol) continue;
      const si = segs[i];
      const sj = segs[j];
      for (const q of intersections(si, sj)) {
        cuts[i].push(q);
        cuts[j].push(q);
      }
      for (const e of [sj.a, sj.b]) if (closestPoint(si, e).dist <= tol) cuts[i].push(e);
      for (const e of [si.a, si.b]) if (closestPoint(sj, e).dist <= tol) cuts[j].push(e);
    }
  }
  return segs.map((s, i) => {
    const us = [0, 1];
    for (const q of cuts[i]) {
      const u = paramOf(s, q);
      if (u > 0 && u < 1) us.push(u);
    }
    us.sort((a, b) => a - b);
    const out: Seg[] = [];
    let u0 = 0;
    let p0 = s.a;
    for (const u of us.slice(1)) {
      const p = u === 1 ? s.b : pointAt(s, u);
      // 位置が近すぎる分け目は 1 つに (長い線分ではパラメータの差がとても小さい)
      if (near(p, p0, tol) && u !== 1) continue;
      if (u === 1 && near(p, p0, tol) && out.length > 0) {
        // 最後の分け目が終点に近すぎる: 直前の部分を終点まで伸ばす
        const last = out.pop()!;
        out.push(last.kind === "line" ? { kind: "line", a: last.a, b: s.b } : subSeg(s, paramOf(s, last.a), 1));
        break;
      }
      const piece = subSeg(s, u0, u);
      if (segLength(piece) > tol) out.push(piece);
      u0 = u;
      p0 = p;
    }
    return out;
  });
}

/** 近い点を同じ頂点にまとめる表 */
class PointIndex {
  readonly points: Vec[] = [];
  private cells = new Map<string, number[]>();
  constructor(private tol: number) {}
  private key(x: number, y: number) {
    return `${Math.floor(x / (4 * this.tol))},${Math.floor(y / (4 * this.tol))}`;
  }
  id(p: Vec): number {
    const cx = Math.floor(p[0] / (4 * this.tol));
    const cy = Math.floor(p[1] / (4 * this.tol));
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++)
        for (const k of this.cells.get(`${cx + dx},${cy + dy}`) ?? []) if (near(this.points[k], p, this.tol)) return k;
    const k = this.points.length;
    this.points.push(p);
    const key = this.key(p[0], p[1]);
    const list = this.cells.get(key) ?? [];
    list.push(k);
    this.cells.set(key, list);
    return k;
  }
}

interface Edge {
  u: number;
  v: number;
  bulge: number;
  seg: Seg;
  alive: boolean;
}

interface HalfEdge {
  edge: number;
  from: number;
  to: number;
  /** from → to の向きの bulge */
  bulge: number;
  angle: number;
  curvature: number;
}

/** 分けた曲線の平面グラフ (同じ辺の重複を除き、ひげを取り除いたもの) */
export class Arrangement {
  readonly vertices: Vec[];
  readonly edges: Edge[] = [];
  private half: HalfEdge[] = [];
  /** 頂点ごとの出ていく半辺 (反時計回りに並べる) */
  private out: number[][] = [];
  readonly tol: number;

  constructor(segs: Seg[], tol = toleranceOf(segs)) {
    this.tol = tol;
    const index = new PointIndex(tol);
    const seen = new Map<string, number[]>();
    for (const pieces of splitAtIntersections(segs, tol)) {
      for (const s of pieces) {
        const u = index.id(s.a);
        const v = index.id(s.b);
        if (u === v) continue;
        const b = bulgeOf(s);
        // 同じ 2 頂点を結ぶ同じ形の辺は 1 本に (向きが逆なら bulge も逆)
        const key = u < v ? `${u}-${v}` : `${v}-${u}`;
        const nb = u < v ? b : -b;
        const same = (seen.get(key) ?? []).some((k) => Math.abs(nb - (this.edges[k].u < this.edges[k].v ? this.edges[k].bulge : -this.edges[k].bulge)) <= 1e-9);
        if (same) continue;
        seen.set(key, [...(seen.get(key) ?? []), this.edges.length]);
        this.edges.push({ u, v, bulge: b, seg: segFromBulge(index.points[u], index.points[v], b), alive: true });
      }
    }
    this.vertices = index.points;
    this.prune();
    this.build();
  }

  /** 端がつながっていない辺 (次数 1 の頂点) を繰り返し取り除く */
  private prune(): void {
    const deg = new Array(this.vertices.length).fill(0);
    const inc: number[][] = this.vertices.map(() => []);
    this.edges.forEach((e, k) => {
      deg[e.u]++;
      deg[e.v]++;
      inc[e.u].push(k);
      inc[e.v].push(k);
    });
    const queue = deg.map((d, i) => (d === 1 ? i : -1)).filter((i) => i >= 0);
    while (queue.length) {
      const w = queue.pop()!;
      for (const k of inc[w]) {
        const e = this.edges[k];
        if (!e.alive) continue;
        e.alive = false;
        for (const x of [e.u, e.v]) {
          deg[x]--;
          if (deg[x] === 1) queue.push(x);
        }
      }
    }
  }

  private build(): void {
    this.out = this.vertices.map(() => []);
    this.edges.forEach((e, k) => {
      if (!e.alive) return;
      const ta = tangentAt(e.seg, "a");
      const tb = tangentAt(e.seg, "b");
      this.half.push({ edge: k, from: e.u, to: e.v, bulge: e.bulge, angle: ta.angle, curvature: ta.curvature });
      this.half.push({ edge: k, from: e.v, to: e.u, bulge: -e.bulge, angle: tb.angle, curvature: tb.curvature });
      this.out[e.u].push(this.half.length - 2);
      this.out[e.v].push(this.half.length - 1);
    });
    const eps = 1e-9;
    for (const list of this.out) {
      list.sort((i, j) => {
        const a = this.half[i];
        const b = this.half[j];
        if (Math.abs(a.angle - b.angle) > eps) return a.angle - b.angle;
        return a.curvature - b.curvature;
      });
    }
  }

  private twin(h: number): number {
    return h % 2 === 0 ? h + 1 : h - 1;
  }

  /** 半辺 h の次 (h の終点で、戻る向きのすぐ時計回り) */
  private next(h: number): number {
    const list = this.out[this.half[h].to];
    const k = list.indexOf(this.twin(h));
    return list[(k - 1 + list.length) % list.length];
  }

  /** 点 p から半直線を出して最初に当たる半辺 (p がその左にある向き)。当たらなければ null */
  private firstHit(p: Vec): number | null {
    // 軸に平行な図形の頂点をかすめないよう、わずかに斜めの向き
    const dir: Vec = [Math.cos(0.0137), Math.sin(0.0137)];
    let far = 0;
    for (const [x, y] of this.vertices) far = Math.max(far, Math.hypot(x - p[0], y - p[1]));
    const ray: Seg = { kind: "line", a: p, b: [p[0] + dir[0] * (2 * far + 1), p[1] + dir[1] * (2 * far + 1)] };
    let best: { h: number; t: number } | null = null;
    this.edges.forEach((e, k) => {
      if (!e.alive) return;
      for (const q of intersections(ray, e.seg)) {
        const t = (q[0] - p[0]) * dir[0] + (q[1] - p[1]) * dir[1];
        if (t <= this.tol || (best && t >= best.t)) continue;
        // q での u → v の向き
        let tx: number;
        let ty: number;
        if (e.seg.kind === "line") {
          tx = e.seg.b[0] - e.seg.a[0];
          ty = e.seg.b[1] - e.seg.a[1];
        } else {
          const s = e.seg as ArcSeg;
          const ang = Math.atan2(q[1] - s.center[1], q[0] - s.center[0]);
          tx = s.ccw ? -Math.sin(ang) : Math.sin(ang);
          ty = s.ccw ? Math.cos(ang) : -Math.cos(ang);
        }
        const left = tx * (p[1] - q[1]) - ty * (p[0] - q[0]) > 0;
        const h = this.half.findIndex((x) => x.edge === k && (left ? x.from === e.u : x.from === e.v));
        best = { h, t };
      }
    });
    return best ? (best as { h: number; t: number }).h : null;
  }

  /** 点を囲む最小の面の輪郭 (反時計回り、同じ直線・同じ円の続きはまとめる)。囲まれていなければ null */
  faceAt(p: Vec): PathData | null {
    const start = this.firstHit(p);
    if (start === null) return null;
    const loop: number[] = [];
    let h = start;
    for (let guard = 0; guard <= this.half.length; guard++) {
      loop.push(h);
      h = this.next(h);
      if (h === start) break;
    }
    if (h !== start) return null;
    const polygon = loop.map((k) => this.vertices[this.half[k].from]);
    const bulges = loop.map((k) => this.half[k].bulge);
    if (pathArea(polygon, bulges) <= 0 || !pointInPath(p, polygon, bulges)) return null;
    return simplifyPath({ polygon, bulges }, this.tol);
  }
}

/** 同じ直線の続きの頂点と、同じ円・同じ向きの続きの円弧の間の頂点を除く */
export function simplifyPath(path: PathData, tol: number): PathData {
  let polygon = path.polygon.slice();
  let bulges = (path.bulges ?? polygon.map(() => 0)).slice() as number[];
  let changed = true;
  while (changed && polygon.length > 2) {
    changed = false;
    for (let i = 0; i < polygon.length; i++) {
      const m = polygon.length;
      const prev = (i - 1 + m) % m;
      const s1 = segFromBulge(polygon[prev], polygon[i], bulges[prev]);
      const s2 = segFromBulge(polygon[i], polygon[(i + 1) % m], bulges[i]);
      let merged: number | null = null;
      if (s1.kind === "line" && s2.kind === "line") {
        const d1: Vec = [s1.b[0] - s1.a[0], s1.b[1] - s1.a[1]];
        const d2: Vec = [s2.b[0] - s2.a[0], s2.b[1] - s2.a[1]];
        const l1 = Math.hypot(d1[0], d1[1]);
        const l2 = Math.hypot(d2[0], d2[1]);
        const cr = (d1[0] * d2[1] - d1[1] * d2[0]) / (l1 * l2);
        if (Math.abs(cr) * Math.min(l1, l2) <= tol && d1[0] * d2[0] + d1[1] * d2[1] > 0) merged = 0;
      } else if (s1.kind === "arc" && s2.kind === "arc") {
        const sameCircle = s1.ccw === s2.ccw && Math.abs(s1.r - s2.r) <= tol && near(s1.center, s2.center, tol);
        const total = arcSweep(s1) + arcSweep(s2);
        if (sameCircle && total < 2 * Math.PI - 1e-6) merged = (s1.ccw ? 1 : -1) * Math.tan(total / 4);
      }
      if (merged === null || polygon.length <= (merged !== 0 ? 2 : 3)) continue;
      polygon = polygon.filter((_, k) => k !== i);
      bulges = bulges.filter((_, k) => k !== i);
      bulges[i === 0 ? polygon.length - 1 : prev] = merged;
      // 頂点の番号がずれたので最初から見直す
      changed = true;
      break;
    }
  }
  return { polygon, bulges: bulges.some((b) => b !== 0) ? bulges : null };
}
