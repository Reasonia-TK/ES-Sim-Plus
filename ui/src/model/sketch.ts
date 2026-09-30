// スケッチ (P7b、prompts/132): 領域でない線・円弧・円・ポリライン。文書の project.cad.sketch に置き (ソルバーは
// 使わない)、囲まれた所から領域・ドメインを作る元・スナップやトリムの相手になる。
// 開いたポリラインの点の操作 (動かす・辺を分ける・曲げる・消す) もここに置く (閉じた経路は cad/path.ts)。

import type { Draft } from "immer";
import { bulgeOf, pathBounds, segFromBulge, type Seg } from "../cad/geom";
import { bulgesOf, circlePath, packBulges, removeVertex, splitEdge, type PathData } from "../cad/path";
import { tidy, tidyPoint, type Point, type Project } from "./project";

/** レイヤ (P7f、model/layers.ts。無ければ既定のレイヤ) */
interface OnLayer {
  layer?: string | null;
}

export interface SketchLine extends OnLayer {
  id: string;
  kind: "line";
  a: Point;
  b: Point;
}

/** 円弧 (a → b、bulge = tan(θ/4)、正は反時計回り) */
export interface SketchArc extends OnLayer {
  id: string;
  kind: "arc";
  a: Point;
  b: Point;
  bulge: number;
}

export interface SketchCircle extends OnLayer {
  id: string;
  kind: "circle";
  center: Point;
  r: number;
}

/** ポリライン (辺 i は点 i → i+1、閉じていれば最後の点 → 最初の点も辺) */
export interface SketchPolyline extends OnLayer {
  id: string;
  kind: "polyline";
  points: Point[];
  bulges?: number[] | null;
  closed: boolean;
}

export type SketchEntity = SketchLine | SketchArc | SketchCircle | SketchPolyline;
export type SketchKind = SketchEntity["kind"];
/** id を除いた形 (足すとき) */
export type NewSketch = Omit<SketchLine, "id"> | Omit<SketchArc, "id"> | Omit<SketchCircle, "id"> | Omit<SketchPolyline, "id">;

export interface CadData {
  sketch?: SketchEntity[];
  [key: string]: unknown;
}

type P = Draft<Project>;

export function sketchOf(p: Project): SketchEntity[] {
  const cad = p.cad as CadData | null | undefined;
  return Array.isArray(cad?.sketch) ? cad.sketch : [];
}

function sketchList(d: P): SketchEntity[] {
  const cad = ((d as Project).cad ??= {}) as CadData;
  cad.sketch ??= [];
  return cad.sketch;
}

/** 新しい ID (s の後の番号の最大 + 1) */
export function nextSketchId(p: Project): string {
  let max = 0;
  for (const e of sketchOf(p)) {
    const m = /^s(\d+)$/.exec(e.id);
    if (m) max = Math.max(max, Number(m[1]));
  }
  return `s${max + 1}`;
}

/** 座標を 1 pm で丸めた形 */
function tidyEntity(e: NewSketch): NewSketch {
  switch (e.kind) {
    case "line":
      return { ...e, a: tidyPoint(e.a), b: tidyPoint(e.b) };
    case "arc":
      return { ...e, a: tidyPoint(e.a), b: tidyPoint(e.b) };
    case "circle":
      return { ...e, center: tidyPoint(e.center), r: tidy(e.r) };
    case "polyline": {
      const packed = packBulges(e.points.map((_, i) => e.bulges?.[i] ?? 0));
      const out = { ...e, points: e.points.map(tidyPoint) };
      if (packed) out.bulges = packed;
      else delete out.bulges;
      return out;
    }
  }
}

/** 使える形か (長さ・半径が 0 でない、ポリラインは 2 点以上) */
export function isValidSketch(e: NewSketch): boolean {
  switch (e.kind) {
    case "line":
    case "arc":
      return e.a[0] !== e.b[0] || e.a[1] !== e.b[1];
    case "circle":
      return e.r > 0;
    case "polyline":
      return e.points.length >= (e.closed ? 3 : 2) || (e.closed && e.points.length === 2 && (e.bulges ?? []).some((b) => b));
  }
}

/** スケッチに足す (使えない形は足さない)。足した ID */
export function addSketch(d: P, e: NewSketch): string | null {
  if (!isValidSketch(e)) return null;
  const id = nextSketchId(d as Project);
  sketchList(d).push({ ...tidyEntity(e), id } as SketchEntity);
  return id;
}

export function deleteSketch(d: P, ids: string[]): void {
  const cad = (d as Project).cad as CadData | null | undefined;
  if (!cad?.sketch) return;
  const drop = new Set(ids);
  cad.sketch = cad.sketch.filter((e) => !drop.has(e.id));
}

/** 形を置き換える (ID と、新しい形で指定しなければレイヤも保つ) */
export function replaceSketch(d: P, id: string, e: NewSketch): void {
  const list = sketchList(d);
  const k = list.findIndex((x) => x.id === id);
  if (k < 0 || !isValidSketch(e)) return;
  const layer = e.layer === undefined ? list[k].layer : e.layer;
  const next = { ...tidyEntity(e), id } as SketchEntity;
  if (layer !== undefined && layer !== null) next.layer = layer;
  list[k] = next;
}

export function translateSketch(e: SketchEntity, dx: number, dy: number): SketchEntity {
  const mv = ([x, y]: Point): Point => tidyPoint([x + dx, y + dy]);
  switch (e.kind) {
    case "line":
      return { ...e, a: mv(e.a), b: mv(e.b) };
    case "arc":
      return { ...e, a: mv(e.a), b: mv(e.b) };
    case "circle":
      return { ...e, center: mv(e.center) };
    case "polyline":
      return { ...e, points: e.points.map(mv) };
  }
}

export function moveSketch(d: P, ids: string[], dx: number, dy: number): void {
  if (dx === 0 && dy === 0) return;
  const set = new Set(ids);
  const list = sketchList(d);
  list.forEach((e, k) => {
    if (set.has(e.id)) list[k] = translateSketch(e as SketchEntity, dx, dy);
  });
}

/** 曲線 (線分・円弧) に分けたもの。円は半円 2 つ */
export function sketchSegs(e: SketchEntity): Seg[] {
  switch (e.kind) {
    case "line":
      return [{ kind: "line", a: e.a, b: e.b }];
    case "arc":
      return [segFromBulge(e.a, e.b, e.bulge)];
    case "circle": {
      const [c, r] = [e.center, e.r];
      return [segFromBulge([c[0] + r, c[1]], [c[0] - r, c[1]], 1), segFromBulge([c[0] - r, c[1]], [c[0] + r, c[1]], 1)];
    }
    case "polyline": {
      const n = e.points.length;
      const m = e.closed ? n : n - 1;
      return Array.from({ length: Math.max(0, m) }, (_, i) => segFromBulge(e.points[i], e.points[(i + 1) % n], e.bulges?.[i] ?? 0));
    }
  }
}

/** 閉じた形 (円・閉じたポリライン) の経路。開いた形は null */
export function sketchClosedPath(e: SketchEntity): PathData | null {
  if (e.kind === "circle") return circlePath(e.center, e.r);
  if (e.kind === "polyline" && e.closed) return { polygon: e.points, bulges: e.bulges ?? null };
  return null;
}

export function sketchBounds(e: SketchEntity): { x0: number; y0: number; x1: number; y1: number } {
  if (e.kind === "circle") return { x0: e.center[0] - e.r, y0: e.center[1] - e.r, x1: e.center[0] + e.r, y1: e.center[1] + e.r };
  // 各曲線 (円弧はふくらみも) の外接矩形を合わせる
  let b = { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity };
  for (const sg of sketchSegs(e)) {
    const sb = pathBounds([sg.a, sg.b], [bulgeOf(sg), 0]);
    b = { x0: Math.min(b.x0, sb.x0), y0: Math.min(b.y0, sb.y0), x1: Math.max(b.x1, sb.x1), y1: Math.max(b.y1, sb.y1) };
  }
  return b;
}

// ---- 開いた・閉じたポリラインの点の操作 (線分・円弧は 2 点のポリラインとして扱う) ----

/** 点と辺ごとの bulge (開いていれば辺は n - 1 本) */
export interface EditPath {
  points: Point[];
  bulges: number[];
  closed: boolean;
}

export function edgeCountOf(p: EditPath): number {
  return p.closed ? p.points.length : p.points.length - 1;
}

export function editMove(p: EditPath, i: number, q: Point): EditPath {
  return { ...p, points: p.points.map((v, k) => (k === i ? q : v)) };
}

export function editBend(p: EditPath, i: number, bulge: number): EditPath {
  return { ...p, bulges: p.bulges.map((b, k) => (k === i ? bulge : b)) };
}

/** 辺 i を u の位置で分ける (新しい点は i + 1) */
export function editSplit(p: EditPath, i: number, u = 0.5): EditPath {
  if (p.closed) {
    const s = splitEdge({ polygon: p.points, bulges: p.bulges }, i, u);
    return { points: s.polygon as Point[], bulges: bulgesOf(s), closed: true };
  }
  // 開いた経路: 最後の点から最初の点への辺は無い
  const closed = splitEdge({ polygon: p.points, bulges: [...p.bulges, 0] }, i, u);
  const b = bulgesOf(closed);
  return { points: closed.polygon as Point[], bulges: b.slice(0, b.length - 1), closed: false };
}

/** 点 i を消す。閉じた経路は前後の辺を 1 本に、開いた経路の端の点はその辺ごと。使えなくなるなら null */
export function editRemove(p: EditPath, i: number): EditPath | null {
  const n = p.points.length;
  if (p.closed) {
    const r = removeVertex({ polygon: p.points, bulges: p.bulges } as PathData, i);
    return r ? { points: r.polygon as Point[], bulges: bulgesOf(r), closed: true } : null;
  }
  if (n <= 2) return null;
  if (i === 0) return { points: p.points.slice(1), bulges: p.bulges.slice(1), closed: false };
  if (i === n - 1) return { points: p.points.slice(0, -1), bulges: p.bulges.slice(0, -1), closed: false };
  // 途中の点: 前後の辺を直線 1 本に
  const bulges = p.bulges.filter((_, k) => k !== i);
  bulges[i - 1] = 0;
  return { points: p.points.filter((_, k) => k !== i), bulges, closed: false };
}

/** スケッチの形 → 点の操作の形 (円は null) */
export function editPathOf(e: SketchEntity): EditPath | null {
  switch (e.kind) {
    case "line":
      return { points: [e.a, e.b], bulges: [0], closed: false };
    case "arc":
      return { points: [e.a, e.b], bulges: [e.bulge], closed: false };
    case "polyline":
      return { points: e.points, bulges: e.points.map((_, i) => e.bulges?.[i] ?? 0).slice(0, edgeCountOf({ points: e.points, bulges: [], closed: e.closed })), closed: e.closed };
    case "circle":
      return null;
  }
}

/**
 * 点の操作の結果をスケッチの形に戻す: 2 点の開いた経路は線分か円弧、それ以外はポリライン
 * (線分・円弧の形を保ったまま点を足すとポリラインになる)
 */
export function sketchFromEditPath(p: EditPath): NewSketch {
  if (!p.closed && p.points.length === 2) {
    return p.bulges[0] ? { kind: "arc", a: p.points[0], b: p.points[1], bulge: p.bulges[0] } : { kind: "line", a: p.points[0], b: p.points[1] };
  }
  const bulges = p.closed ? p.bulges : [...p.bulges, 0];
  return { kind: "polyline", points: p.points, bulges: packBulges(bulges), closed: p.closed };
}
