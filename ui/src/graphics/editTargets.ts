// キャンバスで形を編集する相手 (P7b): 1 つだけ選んだ領域・スケッチ、または (ドメイン・外周の辺を選んでいて
// ほかを選んでいないとき) ドメイン。頂点と辺の中点のハンドル、確定したときの文書の変更 (ドメインは外周の辺の ID と
// 参照も付け替える)。複数選んだものの移動・削除、範囲選択、スケッチの当たり判定もここに置く。

import type { Draft } from "immer";
import { closestPoint, pathBounds } from "../cad/geom";
import { bulgesOf, moveVertex } from "../cad/path";
import {
  applyDomainEdit,
  moveDomainVertex,
  removeDomainVertex,
  setDomainBulge,
  splitDomainEdge,
  type EdgeRemapReport,
} from "../model/domainOps";
import { domainPath, regionPath, type Point, type Project, type Region } from "../model/project";
import { deleteRegion, moveRegion, setCircleRadius, setRegionPath } from "../model/regionOps";
import type { PickRef } from "../model/selection";
import {
  deleteSketch,
  edgeCountOf,
  editPathOf,
  moveSketch,
  replaceSketch,
  sketchBounds,
  sketchFromEditPath,
  sketchOf,
  sketchSegs,
  type EditPath,
  type SketchEntity,
} from "../model/sketch";

export type EditTarget = { kind: "region"; id: string } | { kind: "domain" } | { kind: "sketch"; id: string };

export interface EditObject {
  target: EditTarget;
  /** 頂点と辺 (円は null) */
  path: EditPath | null;
  /** 円 (円の領域・スケッチの円) */
  circle: { center: Point; r: number } | null;
}

/** どう変えたか (ドメインの辺の ID の系譜に使う) */
export type EditHow =
  | { op: "move"; index: number }
  /** 辺 edge を分けて、新しい頂点 (edge + 1) を動かした */
  | { op: "splitMove"; edge: number }
  | { op: "bend"; index: number }
  | { op: "remove"; index: number };

function regionEditPath(r: Region): EditPath {
  const path = regionPath(r);
  return { points: path.polygon as Point[], bulges: bulgesOf(path), closed: true };
}

export function editObjectOf(p: Project, picked: PickRef[], activeNode: string): EditObject | null {
  if (picked.length === 1) {
    const it = picked[0];
    if (it.kind === "region") {
      const r = p.geometry.regions.find((x) => x.id === it.id);
      if (!r) return null;
      if (r.shape) return { target: it, path: null, circle: { center: r.shape.center, r: r.shape.radius } };
      return { target: it, path: regionEditPath(r), circle: null };
    }
    const e = sketchOf(p).find((x) => x.id === it.id);
    if (!e) return null;
    if (e.kind === "circle") return { target: it, path: null, circle: { center: e.center, r: e.r } };
    return { target: it, path: editPathOf(e), circle: null };
  }
  if (picked.length === 0 && (activeNode === "domain" || activeNode.startsWith("edge:"))) {
    const d = domainPath(p);
    return { target: { kind: "domain" }, path: { points: d.polygon as Point[], bulges: bulgesOf(d), closed: true }, circle: null };
  }
  return null;
}

/**
 * ドラッグなどの結果 (result) を文書に書く。ドメインは how から外周の辺の ID の系譜を作って参照を付け替える
 * (外れた境界条件の数を返す)。領域・スケッチは形を置き換える
 */
export function commitEdit(d: Draft<Project>, target: EditTarget, how: EditHow, result: EditPath): EdgeRemapReport | null {
  if (target.kind === "region") {
    setRegionPath(d, target.id, { polygon: result.points, bulges: result.bulges });
    return null;
  }
  if (target.kind === "sketch") {
    replaceSketch(d, target.id, sketchFromEditPath(result));
    return null;
  }
  const p = d as Project;
  switch (how.op) {
    case "move":
      return applyDomainEdit(d, moveDomainVertex(p, how.index, result.points[how.index]));
    case "bend":
      return applyDomainEdit(d, setDomainBulge(p, how.index, result.bulges[how.index]));
    case "splitMove": {
      const e = splitDomainEdge(p, how.edge);
      return applyDomainEdit(d, { ...e, path: moveVertex(e.path, how.edge + 1, result.points[how.edge + 1]) });
    }
    case "remove": {
      const e = removeDomainVertex(p, how.index);
      return e ? applyDomainEdit(d, e) : null;
    }
  }
}

/** 円の半径を変える (円の領域・スケッチの円) */
export function commitRadius(d: Draft<Project>, target: EditTarget, r: number): void {
  if (!(r > 0)) return;
  if (target.kind === "region") setCircleRadius(d, target.id, r);
  else if (target.kind === "sketch") {
    const e = sketchOf(d as Project).find((x) => x.id === target.id);
    if (e?.kind === "circle") replaceSketch(d, target.id, { kind: "circle", center: e.center, r });
  }
}

/** 辺の数 (ハンドルの辺の中点の数) */
export function handleEdgeCount(path: EditPath): number {
  return edgeCountOf(path);
}

// ---- 複数選んだもの ----

export function moveItems(d: Draft<Project>, picked: PickRef[], dx: number, dy: number): void {
  for (const it of picked) if (it.kind === "region") moveRegion(d, it.id, dx, dy);
  moveSketch(
    d,
    picked.filter((x) => x.kind === "sketch").map((x) => x.id),
    dx,
    dy,
  );
}

export function deleteItems(d: Draft<Project>, picked: PickRef[]): void {
  for (const it of picked) if (it.kind === "region") deleteRegion(d, it.id);
  deleteSketch(
    d,
    picked.filter((x) => x.kind === "sketch").map((x) => x.id),
  );
}

/** 点から tol [m] 以内で最も近いスケッチ */
export function findSketchAt(pt: Point, entities: SketchEntity[], tol: number): SketchEntity | null {
  let best: SketchEntity | null = null;
  let bestD = tol;
  for (const e of entities) {
    for (const s of sketchSegs(e)) {
      const dd = closestPoint(s, pt).dist;
      if (dd <= bestD) {
        bestD = dd;
        best = e;
      }
    }
  }
  return best;
}

/** 範囲 (2 隅) に全体が入る領域とスケッチ */
export function itemsInBox(p: Project, a: Point, b: Point): PickRef[] {
  const x0 = Math.min(a[0], b[0]);
  const x1 = Math.max(a[0], b[0]);
  const y0 = Math.min(a[1], b[1]);
  const y1 = Math.max(a[1], b[1]);
  const inside = (bb: { x0: number; y0: number; x1: number; y1: number }) => bb.x0 >= x0 && bb.x1 <= x1 && bb.y0 >= y0 && bb.y1 <= y1;
  const out: PickRef[] = [];
  for (const r of p.geometry.regions) {
    const bb = r.shape
      ? { x0: r.shape.center[0] - r.shape.radius, y0: r.shape.center[1] - r.shape.radius, x1: r.shape.center[0] + r.shape.radius, y1: r.shape.center[1] + r.shape.radius }
      : regionBounds(r);
    if (inside(bb)) out.push({ kind: "region", id: r.id });
  }
  for (const e of sketchOf(p)) if (inside(sketchBounds(e))) out.push({ kind: "sketch", id: e.id });
  return out;
}

function regionBounds(r: Region): { x0: number; y0: number; x1: number; y1: number } {
  const path = regionPath(r);
  return pathBounds(path.polygon, path.bulges);
}
