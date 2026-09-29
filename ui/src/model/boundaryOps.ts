// 外周の辺の境界条件の操作 (v1 と同じ規則): 種類を変えると辺を今の条件から外す (複数の辺で共有していれば
// 分ける)、周期は矩形の向かいの辺と自動で対にする (外すと両方外れる)、Neumann (自然境界) は条件なし。

import type { Draft } from "immer";
import { axisEdge, isRectDomain, type BoundaryCondition, type Project } from "./project";

export type EdgeBcType = "neumann" | "dirichlet" | "symmetry" | "periodic";

export function edgeBcType(p: Project, i: number): EdgeBcType | "axis" {
  if (axisEdge(p) === i) return "axis";
  const bc = p.geometry.boundaries.find((b) => b.edges.includes(i));
  return bc ? bc.type : "neumann";
}

/** 矩形の向かいの辺 (周期境界の相手)。矩形でなければ null */
export function oppositeEdge(p: Project, i: number): number | null {
  return isRectDomain(p) ? (i + 2) % 4 : null;
}

function detach(d: Draft<Project>, i: number): void {
  const bcs = d.geometry.boundaries;
  const out: BoundaryCondition[] = [];
  for (const b of bcs) {
    if (!b.edges.includes(i)) out.push(b as BoundaryCondition);
    else if (b.type === "periodic") continue; // 周期は対ごと外す
    else {
      b.edges = b.edges.filter((e) => e !== i);
      if (b.edges.length) out.push(b as BoundaryCondition);
    }
  }
  d.geometry.boundaries = out;
}

/** 辺 i の境界条件の種類を変える (同じ種類なら何もしない)。周期にできない (矩形でない) ときは false */
export function setEdgeType(d: Draft<Project>, i: number, type: EdgeBcType): boolean {
  const p = d as Project;
  if (axisEdge(p) === i) return false;
  if (edgeBcType(p, i) === type) return true;
  if (type === "periodic") {
    const opp = oppositeEdge(p, i);
    if (opp === null || axisEdge(p) === opp) return false;
    detach(d, i);
    detach(d, opp);
    d.geometry.boundaries.push({ edges: [i, opp], type: "periodic" });
    return true;
  }
  detach(d, i);
  if (type === "dirichlet") d.geometry.boundaries.push({ edges: [i], type: "dirichlet", voltage: 0 });
  else if (type === "symmetry") d.geometry.boundaries.push({ edges: [i], type: "symmetry" });
  return true;
}

/** 辺 i を含む境界条件の番号 (無ければ -1) */
export function boundaryIndexOfEdge(p: Project, i: number): number {
  return p.geometry.boundaries.findIndex((b) => b.edges.includes(i));
}
