// 形の変換 (P7b): スケッチの閉じた形 (円・閉じたポリライン) を領域に、スケッチ・領域の閉じた形をドメインにする。
// 変換した元 (スケッチ・領域) は消す。ドメインは元の辺に重なる辺の ID (と境界条件など) を引き継ぐ。

import type { Draft } from "immer";
import { applyDomainEdit, domainFromPath, type EdgeRemapReport } from "./domainOps";
import { regionPath, type Point, type Project } from "./project";
import { addCircleRegionAt, addPolygonRegion, deleteRegion } from "./regionOps";
import { deleteSketch, sketchClosedPath, sketchOf } from "./sketch";

type P = Draft<Project>;

/** スケッチの閉じた形を領域にする (円は円の領域)。作った領域の ID (閉じていなければ null) */
export function sketchToRegion(d: P, id: string): string | null {
  const e = sketchOf(d as Project).find((x) => x.id === id);
  if (!e) return null;
  let rid: string | null = null;
  if (e.kind === "circle") rid = addCircleRegionAt(d, e.center, e.r);
  else {
    const path = sketchClosedPath(e);
    if (path) rid = addPolygonRegion(d, path.polygon as Point[], path.bulges);
  }
  if (rid) deleteSketch(d, [id]);
  return rid;
}

/** スケッチの閉じた形をドメインにする。閉じていなければ null */
export function sketchToDomain(d: P, id: string): EdgeRemapReport | null {
  const e = sketchOf(d as Project).find((x) => x.id === id);
  const path = e ? sketchClosedPath(e) : null;
  if (!e || !path) return null;
  const report = applyDomainEdit(d, domainFromPath(d as Project, path));
  deleteSketch(d, [id]);
  return report;
}

/** 領域の形をドメインにする (領域は消す) */
export function regionToDomain(d: P, id: string): EdgeRemapReport | null {
  const r = d.geometry.regions.find((x) => x.id === id);
  if (!r) return null;
  const report = applyDomainEdit(d, domainFromPath(d as Project, regionPath(r as Project["geometry"]["regions"][number])));
  deleteRegion(d, id);
  return report;
}
