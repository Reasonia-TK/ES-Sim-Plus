// 外周の辺の境界条件の操作 (v1 と同じ規則): 種類を変えると辺を今の条件から外す (複数の辺で共有していれば
// 分ける)、周期は平行・同じ長さの辺と対にする (外すと両方外れる)、Neumann (自然境界) は条件なし。
// 対称軸 (軸対称で r = 0 の上の直線の辺) には条件を付けない。

import type { Draft } from "immer";
import { periodicPartners } from "./domainOps";
import { isAxisEdge, type BoundaryCondition, type Project } from "./project";

export type EdgeBcType = "neumann" | "dirichlet" | "symmetry" | "periodic";

export function edgeBcType(p: Project, i: number): EdgeBcType | "axis" {
  if (isAxisEdge(p, i)) return "axis";
  const bc = p.geometry.boundaries.find((b) => b.edges.includes(i));
  return bc ? bc.type : "neumann";
}

/**
 * 周期境界の相手の既定 (平行・同じ長さの辺がちょうど 1 本ならそれ。矩形では向かいの辺)。無いか選べなければ null。
 * 候補が複数あるときは periodicPartners から選ぶ
 */
export function oppositeEdge(p: Project, i: number): number | null {
  const cand = periodicPartners(p, i);
  return cand.length === 1 ? cand[0] : null;
}

/** 周期境界になっている辺 i の相手 (周期でなければ null) */
export function periodicPartnerOf(p: Project, i: number): number | null {
  const bc = p.geometry.boundaries.find((b) => b.type === "periodic" && b.edges.includes(i));
  return bc ? (bc.edges.find((e) => e !== i) ?? null) : null;
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

/**
 * 辺 i の境界条件の種類を変える (同じ種類なら何もしない)。周期は partner (省略時は oppositeEdge) と対にする。
 * 周期にできない (相手が無い) ときは false
 */
export function setEdgeType(d: Draft<Project>, i: number, type: EdgeBcType, partner?: number): boolean {
  const p = d as Project;
  if (isAxisEdge(p, i)) return false;
  if (type === "periodic") {
    const opp = partner ?? oppositeEdge(p, i);
    if (opp === null || !periodicPartners(p, i).includes(opp)) return false;
    if (periodicPartnerOf(p, i) === opp) return true;
    detach(d, i);
    detach(d, opp);
    d.geometry.boundaries.push({ edges: [i, opp], type: "periodic" });
    return true;
  }
  if (edgeBcType(p, i) === type) return true;
  detach(d, i);
  if (type === "dirichlet") d.geometry.boundaries.push({ edges: [i], type: "dirichlet", voltage: 0 });
  else if (type === "symmetry") d.geometry.boundaries.push({ edges: [i], type: "symmetry" });
  return true;
}

/** 辺 i を含む境界条件の番号 (無ければ -1) */
export function boundaryIndexOfEdge(p: Project, i: number): number {
  return p.geometry.boundaries.findIndex((b) => b.edges.includes(i));
}
