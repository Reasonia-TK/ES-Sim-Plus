// 形の編集と変換を文書に (P7d):
// - 変換: 選んだ領域・スケッチの移動・回転・ミラー・尺度 (相似変換だけ。コピーも)、配列 (矩形・円周のコピー)
// - 角のフィレット・面取り (領域・ドメイン・スケッチのポリライン)、2 本のスケッチの線のフィレット
// - オフセット (領域 → 新しい領域、スケッチ → 新しいスケッチ)、トリム・延長 (スケッチ)
// ドメインの角の編集は外周の辺の ID を保ち、新しい辺は前後の辺が同じ境界条件のときだけそれを引き継ぐ。

import type { Draft } from "immer";
import { chamferCorner, chainToPath, extendEnd, filletCorner, filletLines, nearerEnd, offsetPath, simplifyChain, trimAt, type EditError, type EditResult, type PathPts } from "../cad/edit";
import { bulgeOf, closestPoint, intersections, pathSegs, segFromBulge, type LineSeg, type Seg, type Vec } from "../cad/geom";
import { applyAffine, bulgesOf, determinant, rotation, scaling, signedArea, transformPath, translation, type Affine } from "../cad/path";
import { applyDomainEdit, type EdgeRemapReport } from "./domainOps";
import { boundaryOfEdge, domainPath, edgeIdsOf, nextEdgeId, regionPath, tidy, tidyPoint, uniqueRegionId, type Point, type Project, type Region } from "./project";
import { setRegionPath } from "./regionOps";
import type { PickRef } from "./selection";
import { addSketch, replaceSketch, sketchOf, sketchSegs, type NewSketch, type SketchEntity } from "./sketch";

type P = Draft<Project>;

/** 相似変換の倍率 (面積の倍率の平方根) */
function scaleOf(m: Affine): number {
  return Math.sqrt(Math.abs(determinant(m)));
}

// ---- 変換 ----

/** スケッチの形を変換する (鏡映は bulge の符号を反転) */
export function transformSketch<E extends NewSketch>(e: E, m: Affine): E {
  const flip = determinant(m) < 0 ? -1 : 1;
  const tp = (q: Point) => tidyPoint(applyAffine(m, q) as Point);
  switch (e.kind) {
    case "line":
      return { ...e, a: tp(e.a), b: tp(e.b) };
    case "arc":
      return { ...e, a: tp(e.a), b: tp(e.b), bulge: e.bulge * flip };
    case "circle":
      return { ...e, center: tp(e.center), r: tidy(e.r * scaleOf(m)) };
    case "polyline":
      return { ...e, points: e.points.map(tp), bulges: e.bulges ? e.bulges.map((b) => b * flip) : e.bulges };
  }
  return e;
}

/** 領域の形を変換した輪郭 (円は円のまま: 中心と半径) */
export function transformRegionShape(r: Region, m: Affine): Pick<Region, "polygon" | "bulges" | "shape"> {
  if (r.shape) return { shape: { kind: "circle", center: tidyPoint(applyAffine(m, r.shape.center) as Point), radius: tidy(r.shape.radius * scaleOf(m)) } };
  const path = transformPath(regionPath(r), m);
  return { polygon: path.polygon.map((q) => tidyPoint(q as Point)), bulges: path.bulges ?? null };
}

function writeRegionShape(r: Region, s: Pick<Region, "polygon" | "bulges" | "shape">): void {
  if (s.shape) r.shape = s.shape;
  else {
    r.polygon = s.polygon ?? [];
    if (s.bulges?.some((b) => b)) r.bulges = s.bulges;
    else delete r.bulges;
  }
}

interface LocalSize {
  region: string;
  size: number;
}

/** 領域のコピー (値と局所メッシュ幅も写す)。新しい ID */
function copyRegion(d: P, src: Region, shape: Pick<Region, "polygon" | "bulges" | "shape">): string {
  const copy = JSON.parse(JSON.stringify(src)) as Region; // Immer の draft は structuredClone できない
  copy.id = uniqueRegionId(d as Project, `${src.id}_`);
  delete copy.polygon;
  delete copy.bulges;
  delete copy.shape;
  writeRegionShape(copy, shape);
  d.geometry.regions.push(copy);
  const ls = d.mesh.local_sizes as LocalSize[] | undefined;
  const size = ls?.find((l) => l.region === src.id)?.size;
  if (ls && size !== undefined) ls.push({ region: copy.id, size });
  return copy.id;
}

/**
 * 選んだものを変換する (copy なら元を残して変換したコピーを作る)。変換後に選ぶもの (コピーならコピー、それ以外は
 * 同じもの) を返す
 */
export function transformItems(d: P, picked: PickRef[], m: Affine, copy: boolean): PickRef[] {
  const out: PickRef[] = [];
  for (const it of picked) {
    if (it.kind === "region") {
      const r = d.geometry.regions.find((x) => x.id === it.id) as Region | undefined;
      if (!r) continue;
      const shape = transformRegionShape(r, m);
      if (copy) out.push({ kind: "region", id: copyRegion(d, r, shape) });
      else {
        writeRegionShape(r, shape);
        out.push(it);
      }
    } else {
      const e = sketchOf(d as Project).find((x) => x.id === it.id);
      if (!e) continue;
      const { id: _id, ...rest } = e;
      void _id;
      const next = transformSketch(rest as NewSketch, m);
      if (copy) {
        const nid = addSketch(d, next);
        if (nid) out.push({ kind: "sketch", id: nid });
      } else {
        replaceSketch(d, it.id, next);
        out.push(it);
      }
    }
  }
  return out;
}

export type ArraySpec =
  /** 矩形: nx 列 × ny 行、間隔 dx・dy (元が左下) */
  | { kind: "rect"; nx: number; ny: number; dx: number; dy: number }
  /** 円周: 中心のまわりに n 個 (元を含む)。angle は全体の角度 [°] (360 なら一周を n 等分) */
  | { kind: "polar"; n: number; center: Point; angle: number };

/** 配列のコピーの変換 (元の位置は除く) */
export function arrayTransforms(spec: ArraySpec): Affine[] {
  const out: Affine[] = [];
  if (spec.kind === "rect") {
    for (let iy = 0; iy < spec.ny; iy++) for (let ix = 0; ix < spec.nx; ix++) if (ix || iy) out.push(translation(ix * spec.dx, iy * spec.dy));
    return out;
  }
  const full = Math.abs(Math.abs(spec.angle) - 360) < 1e-9;
  const step = spec.n > 1 ? (spec.angle * Math.PI) / 180 / (full ? spec.n : spec.n - 1) : 0;
  for (let k = 1; k < spec.n; k++) out.push(rotation(k * step, spec.center));
  return out;
}

/** 配列のコピーを作る。作ったもの */
export function arrayItems(d: P, picked: PickRef[], spec: ArraySpec): PickRef[] {
  const out: PickRef[] = [];
  for (const m of arrayTransforms(spec)) out.push(...transformItems(d, picked, m, true));
  return out;
}

/** 選んだものの外接矩形の中心 (回転・尺度・ミラーの既定の中心) */
export function pickedCenter(p: Project, picked: PickRef[]): Point | null {
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
  for (const it of picked) {
    const segs = itemSegs(p, it);
    for (const s of segs) {
      addPt(s.a);
      addPt(s.b);
      if (s.kind === "arc") for (const q of [closestPoint(s, [s.center[0] + 2 * s.r, s.center[1]]).point, closestPoint(s, [s.center[0] - 2 * s.r, s.center[1]]).point, closestPoint(s, [s.center[0], s.center[1] + 2 * s.r]).point, closestPoint(s, [s.center[0], s.center[1] - 2 * s.r]).point]) addPt(q);
    }
  }
  return Number.isFinite(x0) ? [tidy((x0 + x1) / 2), tidy((y0 + y1) / 2)] : null;
}

/** 選んだものの曲線 */
export function itemSegs(p: Project, it: PickRef): Seg[] {
  if (it.kind === "region") {
    const r = p.geometry.regions.find((x) => x.id === it.id);
    if (!r) return [];
    const rp = regionPath(r);
    return pathSegs(rp.polygon, bulgesOf(rp));
  }
  const e = sketchOf(p).find((x) => x.id === it.id);
  return e ? sketchSegs(e) : [];
}

export { rotation, scaling, translation };

// ---- 角のフィレット・面取り ----

export type CornerTarget = { kind: "region"; id: string } | { kind: "domain" } | { kind: "sketch"; id: string };

/** 角を丸める・面取りする (頂点 i)。ドメインは外周の辺の ID と境界条件を保つ */
export function cornerEdit(d: P, target: CornerTarget, i: number, size: number, mode: "fillet" | "chamfer"): EditResult<EdgeRemapReport | null> {
  const op = (p: PathPts) => (mode === "fillet" ? filletCorner(p, i, size) : chamferCorner(p, i, size));
  if (target.kind === "region") {
    const r = d.geometry.regions.find((x) => x.id === target.id) as Region | undefined;
    if (!r || r.shape) return { ok: false, error: "notStraight" };
    const rp = regionPath(r);
    const res = op({ points: rp.polygon as Vec[], bulges: bulgesOf(rp), closed: true });
    if (!res.ok) return res;
    setRegionPath(d, target.id, { polygon: res.value.points, bulges: res.value.bulges });
    return { ok: true, value: null };
  }
  if (target.kind === "sketch") {
    const e = sketchOf(d as Project).find((x) => x.id === target.id);
    if (!e || e.kind !== "polyline") return { ok: false, error: "notStraight" };
    const n = e.points.length;
    const m = e.closed ? n : n - 1;
    const res = op({ points: e.points, bulges: Array.from({ length: m }, (_, k) => e.bulges?.[k] ?? 0), closed: e.closed });
    if (!res.ok) return res;
    replaceSketch(d, e.id, { kind: "polyline", points: res.value.points as Point[], bulges: e.closed ? res.value.bulges : [...res.value.bulges, 0], closed: e.closed });
    return { ok: true, value: null };
  }
  const p = d as Project;
  const dp = domainPath(p);
  const res = op({ points: dp.polygon as Vec[], bulges: bulgesOf(dp), closed: true });
  if (!res.ok) return res;
  const ids = edgeIdsOf(p);
  const n = ids.length;
  const prev = (i - 1 + n) % n;
  const newId = nextEdgeId(ids);
  // 前後の辺が同じ境界条件なら新しい辺 (円弧・面取りの辺) もその条件に
  const bcPrev = boundaryOfEdge(p, prev);
  const inherit = bcPrev !== null && bcPrev === boundaryOfEdge(p, i);
  return {
    ok: true,
    value: applyDomainEdit(d, {
      path: { polygon: res.value.points, bulges: res.value.bulges },
      ids: [...ids.slice(0, i), newId, ...ids.slice(i)],
      parents: inherit ? { [newId]: ids[prev] } : {},
    }),
  };
}

/** 2 本のスケッチの線のフィレット (クリックした側を残す)。円弧のスケッチを足す (半径 0 なら角をそろえるだけ) */
export function filletSketchLines(d: P, id1: string, pick1: Point, id2: string, pick2: Point, r: number): EditResult<string | null> {
  const list = sketchOf(d as Project);
  const e1 = list.find((x) => x.id === id1);
  const e2 = list.find((x) => x.id === id2);
  if (!e1 || !e2 || e1.kind !== "line" || e2.kind !== "line" || id1 === id2) return { ok: false, error: "notStraight" };
  const l1: LineSeg = { kind: "line", a: e1.a, b: e1.b };
  const l2: LineSeg = { kind: "line", a: e2.a, b: e2.b };
  const res = filletLines(l1, pick1, l2, pick2, r);
  if (!res.ok) return res;
  replaceSketch(d, id1, { kind: "line", a: res.value.line1.a as Point, b: res.value.line1.b as Point });
  replaceSketch(d, id2, { kind: "line", a: res.value.line2.a as Point, b: res.value.line2.b as Point });
  const arc = res.value.arc;
  return { ok: true, value: arc ? addSketch(d, { kind: "arc", a: arc.a as Point, b: arc.b as Point, bulge: arc.bulge }) : null };
}

// ---- オフセット ----

/** 閉じた形の外側か (点が形の外) */
function outsideOf(segs: Seg[], q: Vec): boolean {
  // 半直線と交わる回数 (斜めの向き)
  const dir: Vec = [Math.cos(0.0137), Math.sin(0.0137)];
  let far = 0;
  for (const s of segs) for (const e of [s.a, s.b]) far = Math.max(far, Math.hypot(e[0] - q[0], e[1] - q[1]));
  const ray: Seg = { kind: "line", a: q, b: [q[0] + dir[0] * (2 * far + 1), q[1] + dir[1] * (2 * far + 1)] };
  let n = 0;
  for (const s of segs) {
    for (const x of intersections(ray, s)) if (Math.hypot(x[0] - q[0], x[1] - q[1]) > 0) n++;
  }
  return n % 2 === 0;
}

/** 開いた曲線のどちら側 (左なら +1) に点があるか: いちばん近い曲線の接線と比べる */
function sideOf(segs: Seg[], q: Vec): number {
  let best = segs[0];
  let bestD = Infinity;
  for (const s of segs) {
    const dd = closestPoint(s, q).dist;
    if (dd < bestD) {
      bestD = dd;
      best = s;
    }
  }
  const c = closestPoint(best, q).point;
  let tx: number;
  let ty: number;
  if (best.kind === "line") {
    tx = best.b[0] - best.a[0];
    ty = best.b[1] - best.a[1];
  } else {
    const t = Math.atan2(c[1] - best.center[1], c[0] - best.center[0]);
    tx = best.ccw ? -Math.sin(t) : Math.sin(t);
    ty = best.ccw ? Math.cos(t) : -Math.cos(t);
  }
  return tx * (q[1] - c[1]) - ty * (q[0] - c[0]) >= 0 ? 1 : -1;
}

/**
 * オフセット: 領域 (新しい領域、同じ値) か スケッチ (新しいスケッチ、同じ種類) を、クリックした側 (side) へ dist だけ
 * ずらしたコピーを作る。作ったもの
 */
export function offsetItem(d: P, item: PickRef, dist: number, side: Point): EditResult<PickRef> {
  if (!(dist > 0)) return { ok: false, error: "tooSmall" };
  const p = d as Project;
  if (item.kind === "region") {
    const r = d.geometry.regions.find((x) => x.id === item.id) as Region | undefined;
    if (!r) return { ok: false, error: "noTarget" };
    if (r.shape) {
      const inside = Math.hypot(side[0] - r.shape.center[0], side[1] - r.shape.center[1]) < r.shape.radius;
      const rad = r.shape.radius + (inside ? -dist : dist);
      if (!(rad > 0)) return { ok: false, error: "tooLarge" };
      return { ok: true, value: { kind: "region", id: copyRegion(d, r, { shape: { kind: "circle", center: r.shape.center, radius: tidy(rad) } }) } };
    }
    const rp = regionPath(r);
    const segs = pathSegs(rp.polygon, bulgesOf(rp));
    const ccw = signedArea(rp) > 0 ? 1 : -1;
    // 反時計回りの左は内側
    const inward = !outsideOf(segs, side);
    const res = offsetPath({ points: rp.polygon as Vec[], bulges: bulgesOf(rp), closed: true }, (inward ? dist : -dist) * ccw);
    if (!res.ok) return res;
    return { ok: true, value: { kind: "region", id: copyRegion(d, r, { polygon: res.value.points.map((q) => tidyPoint(q as Point)), bulges: res.value.bulges }) } };
  }
  const e = sketchOf(p).find((x) => x.id === item.id);
  if (!e) return { ok: false, error: "noTarget" };
  if (e.kind === "circle") {
    const inside = Math.hypot(side[0] - e.center[0], side[1] - e.center[1]) < e.r;
    const rad = e.r + (inside ? -dist : dist);
    if (!(rad > 0)) return { ok: false, error: "tooLarge" };
    const id = addSketch(d, { kind: "circle", center: e.center, r: rad });
    return id ? { ok: true, value: { kind: "sketch", id } } : { ok: false, error: "tooLarge" };
  }
  const segs = sketchSegs(e);
  const closed = e.kind === "polyline" && e.closed;
  const pts: PathPts = closed
    ? { points: (e as { points: Point[] }).points, bulges: segs.map((s) => bulgeOf(s)), closed: true }
    : chainToPath(segs);
  let signed: number;
  if (closed) {
    const ccw = signedArea({ polygon: pts.points, bulges: pts.bulges }) > 0 ? 1 : -1;
    signed = (outsideOf(segs, side) ? -dist : dist) * ccw;
  } else signed = sideOf(segs, side) * dist;
  const res = offsetPath(pts, signed);
  if (!res.ok) return res;
  const v = res.value;
  let next: NewSketch;
  if (e.kind === "line" || e.kind === "arc") {
    const b = v.bulges[0] ?? 0;
    next = b ? { kind: "arc", a: v.points[0] as Point, b: v.points[1] as Point, bulge: b } : { kind: "line", a: v.points[0] as Point, b: v.points[1] as Point };
  } else next = { kind: "polyline", points: v.points as Point[], bulges: closed ? v.bulges : [...v.bulges, 0], closed };
  const id = addSketch(d, next);
  return id ? { ok: true, value: { kind: "sketch", id } } : { ok: false, error: "tooLarge" };
}

// ---- トリム・延長 ----

/** つながった曲線の並び → スケッチの形 (1 本なら線分か円弧、それ以外は開いたポリライン) */
function chainToSketch(chain: Seg[]): NewSketch {
  const c = simplifyChain(chain);
  if (c.length === 1) {
    const s = c[0];
    const b = bulgeOf(s);
    return b ? { kind: "arc", a: s.a as Point, b: s.b as Point, bulge: b } : { kind: "line", a: s.a as Point, b: s.b as Point };
  }
  const path = chainToPath(c);
  return { kind: "polyline", points: path.points as Point[], bulges: [...path.bulges, 0], closed: false };
}

/**
 * トリム: スケッチ id の、クリックした点に近い区切り (ほかの曲線 cutters との交点から交点まで) を消す。残りは元の
 * ID (と足した新しいスケッチ)。全て消えれば削除
 */
export function trimSketch(d: P, id: string, click: Point, cutters: Seg[]): EditResult<string[]> {
  const e = sketchOf(d as Project).find((x) => x.id === id);
  if (!e) return { ok: false, error: "noTarget" };
  const closed = e.kind === "circle" || (e.kind === "polyline" && e.closed);
  const res = trimAt(sketchSegs(e), closed, cutters, click);
  if (!res.ok) return res;
  const chains = res.value;
  const list = sketchOf(d as Project);
  if (chains.length === 0) {
    const k = list.findIndex((x) => x.id === id);
    if (k >= 0) list.splice(k, 1);
    return { ok: true, value: [] };
  }
  replaceSketch(d, id, chainToSketch(chains[0]));
  const ids = [id];
  for (const c of chains.slice(1)) {
    const nid = addSketch(d, chainToSketch(c));
    if (nid) ids.push(nid);
  }
  return { ok: true, value: ids };
}

/** 延長: スケッチ id の、クリックに近い端を、その先で最初に交わる曲線 (boundaries) まで延ばす */
export function extendSketch(d: P, id: string, click: Point, boundaries: Seg[]): EditResult<null> {
  const e = sketchOf(d as Project).find((x) => x.id === id);
  if (!e || e.kind === "circle" || (e.kind === "polyline" && e.closed)) return { ok: false, error: "noTarget" };
  const segs = sketchSegs(e);
  if (e.kind === "polyline") {
    // 近い方の端の辺を延ばす
    const first = segs[0];
    const last = segs[segs.length - 1];
    const atStart = Math.hypot(click[0] - first.a[0], click[1] - first.a[1]) <= Math.hypot(click[0] - last.b[0], click[1] - last.b[1]);
    const s = atStart ? first : last;
    const res = extendEnd(s, atStart ? "a" : "b", boundaries);
    if (!res.ok) return res;
    const next = atStart ? [res.value, ...segs.slice(1)] : [...segs.slice(0, -1), res.value];
    const path = chainToPath(next);
    replaceSketch(d, id, { kind: "polyline", points: path.points as Point[], bulges: [...path.bulges, 0], closed: false });
    return { ok: true, value: null };
  }
  const s = segs[0];
  const res = extendEnd(s, nearerEnd(s, click), boundaries);
  if (!res.ok) return res;
  replaceSketch(d, id, chainToSketch([res.value]));
  return { ok: true, value: null };
}

/** スケッチ以外の曲線と、ほかのスケッチの曲線 (トリム・延長の相手)。exclude のスケッチは除く */
export function cutterSegs(p: Project, exclude: string | null): Seg[] {
  const out: Seg[] = [];
  const dp = domainPath(p);
  out.push(...pathSegs(dp.polygon, bulgesOf(dp)));
  for (const r of p.geometry.regions) {
    const rp = regionPath(r);
    out.push(...pathSegs(rp.polygon, bulgesOf(rp)));
  }
  for (const e of sketchOf(p)) if (e.id !== exclude) out.push(...sketchSegs(e));
  return out;
}

export type { EditError, SketchEntity };
export { segFromBulge };
