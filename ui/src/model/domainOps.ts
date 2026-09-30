// 外周 (ドメイン) の編集 (prompts/132): 経路 (頂点・円弧) と辺の永続 ID を一緒に変え、外周の辺の番号を持つ参照
// (境界条件・FN 放出 2 つ・PIC の反射・DSMC の境界) を ID の系譜で付け替える。辺を分けると両方に同じ条件が付き、
// 消えた辺の条件は外れる。周期境界は相手と対にできなくなったら外し、対称軸 (r = 0) の上の辺からは条件を外す
// (軸対称への切り替えと同じ)。反時計回りでなくなったら向きを戻す (ID も並べ替える)。

import type { Draft } from "immer";
import {
  bulgesOf,
  edgeMidpoint,
  ensureCcw,
  moveVertex,
  packBulges,
  removeVertex,
  reversedEdgeIndex,
  setBulge,
  splitEdge,
  type PathData,
} from "../cad/path";
import { axisEdges, domainPath, edgeIdsOf, nextEdgeId, tidyPoint, type BoundaryCondition, type Point, type Project } from "./project";

export interface DomainEdit {
  path: PathData;
  /** 新しい経路の辺の ID (辺と同じ数) */
  ids: string[];
  /** 新しい辺の ID → 条件を引き継ぐ元の辺の ID (分けた辺の後ろ半分など) */
  parents?: Record<string, string>;
}

export interface EdgeRemapReport {
  /** 対にできなくなって外した周期境界の数 */
  periodicRemoved: number;
  /** 辺が無くなって外した境界条件の数 */
  boundariesRemoved: number;
}

type Obj = Record<string, unknown>;

/** 外周の辺の番号の配列を持つ場所 (境界条件以外の 4 か所)。読み書きの組 */
function otherEdgeLists(p: Project): { get: () => number[]; set: (e: number[]) => void }[] {
  const out: { get: () => number[]; set: (e: number[]) => void }[] = [];
  const add = (obj: Obj | null | undefined, key: string) => {
    if (obj && Array.isArray(obj[key])) out.push({ get: () => obj[key] as number[], set: (e) => (obj[key] = e) });
  };
  const particles = p.particles as Obj | null | undefined;
  add(particles?.fn as Obj | null | undefined, "edges");
  const pic = p.pic as Obj | null | undefined;
  add(pic, "reflect_edges");
  add(pic?.fn as Obj | null | undefined, "edges");
  const dsmc = p.dsmc as Obj | null | undefined;
  if (Array.isArray(dsmc?.boundaries)) for (const b of dsmc.boundaries as Obj[]) add(b, "edges");
  return out;
}

/** 直線の辺の向き (長さ付き)。円弧なら null */
function straightDir(p: Project, i: number): Point | null {
  const poly = p.geometry.domain.polygon;
  const b = bulgesOf(domainPath(p));
  if (b[i] !== 0) return null;
  const a = poly[i];
  const c = poly[(i + 1) % poly.length];
  return [c[0] - a[0], c[1] - a[1]];
}

/** 周期境界の対にできるか (どちらも直線で、平行かつ同じ長さ。backend の _check_periodic_pairs と同じ許容差) */
export function periodicPairOk(p: Project, e1: number, e2: number): boolean {
  if (e1 === e2) return false;
  const d1 = straightDir(p, e1);
  const d2 = straightDir(p, e2);
  if (!d1 || !d2) return false;
  const l1 = Math.hypot(d1[0], d1[1]);
  const l2 = Math.hypot(d2[0], d2[1]);
  if (!(l1 > 0 && l2 > 0)) return false;
  const cross = d1[0] * d2[1] - d1[1] * d2[0];
  return Math.abs(cross) <= 1e-6 * l1 * l2 && Math.abs(l1 - l2) <= 1e-6 * Math.max(l1, l2);
}

/** 辺 i と周期境界の対にできる辺 (対称軸の辺は除く) */
export function periodicPartners(p: Project, i: number): number[] {
  const axis = new Set(axisEdges(p));
  if (axis.has(i)) return [];
  const n = p.geometry.domain.polygon.length;
  const out: number[] = [];
  for (let j = 0; j < n; j++) if (!axis.has(j) && periodicPairOk(p, i, j)) out.push(j);
  return out;
}

const sortUnique = (xs: number[]) => [...new Set(xs)].sort((a, b) => a - b);

/**
 * 外周を変える (経路・辺の ID) と同時に、外周の辺の番号の参照を付け替える。
 * 付け替えは「同じ ID の辺」と「その ID を親に持つ新しい辺」へ。
 */
export function applyDomainEdit(d: Draft<Project>, edit: DomainEdit): EdgeRemapReport {
  const p = d as Project;
  const oldIds = edgeIdsOf(p);
  let path = edit.path;
  let ids = edit.ids.slice();
  const parents = edit.parents ?? {};
  const ccw = ensureCcw(path);
  if (ccw.reversed) {
    const n = ids.length;
    ids = Array.from({ length: n }, (_, k) => ids[reversedEdgeIndex(n, k)]);
    path = ccw.path;
  }
  const targets = new Map<string, number[]>();
  ids.forEach((id, j) => {
    for (const key of [id, parents[id]]) {
      if (!key) continue;
      const list = targets.get(key) ?? [];
      list.push(j);
      targets.set(key, list);
    }
  });
  const map = (edges: number[]) => sortUnique(edges.flatMap((i) => targets.get(oldIds[i]) ?? []));

  const dom = d.geometry.domain;
  dom.polygon = path.polygon.map((q) => tidyPoint(q as Point));
  const packed = packBulges(bulgesOf(path));
  if (packed) dom.bulges = packed;
  else delete dom.bulges;
  dom.edge_ids = ids;

  for (const l of otherEdgeLists(p)) l.set(map(l.get()));
  const report: EdgeRemapReport = { periodicRemoved: 0, boundariesRemoved: 0 };
  const axis = new Set(axisEdges(p));
  const kept: BoundaryCondition[] = [];
  for (const b of d.geometry.boundaries as BoundaryCondition[]) {
    const edges = map(b.edges).filter((e) => !axis.has(e));
    if (b.type === "periodic") {
      if (edges.length === 2 && periodicPairOk(p, edges[0], edges[1])) {
        b.edges = edges;
        kept.push(b);
      } else report.periodicRemoved++;
      continue;
    }
    if (edges.length === 0) {
      report.boundariesRemoved++;
      continue;
    }
    b.edges = edges;
    kept.push(b);
  }
  d.geometry.boundaries = kept;
  return report;
}

// ---- 編集の組み立て (文書は変えない) ----

/** 辺 i を u の位置で分ける (後ろ半分は新しい ID で、元の辺の条件を引き継ぐ) */
export function splitDomainEdge(p: Project, i: number, u = 0.5): DomainEdit {
  const ids = edgeIdsOf(p);
  const id = nextEdgeId(ids);
  return {
    path: splitEdge(domainPath(p), i, u),
    ids: [...ids.slice(0, i + 1), id, ...ids.slice(i + 1)],
    parents: { [id]: ids[i] },
  };
}

/** 頂点 i を消す (前後の辺は前の辺の ID で 1 本に、辺 i の条件は外れる)。消せなければ null */
export function removeDomainVertex(p: Project, i: number): DomainEdit | null {
  const path = removeVertex(domainPath(p), i);
  if (!path) return null;
  return { path, ids: edgeIdsOf(p).filter((_, k) => k !== i) };
}

export function moveDomainVertex(p: Project, i: number, q: Point): DomainEdit {
  return { path: moveVertex(domainPath(p), i, q), ids: edgeIdsOf(p) };
}

export function setDomainBulge(p: Project, i: number, bulge: number): DomainEdit {
  return { path: setBulge(domainPath(p), i, bulge), ids: edgeIdsOf(p) };
}

/** 頂点の数を変えない置き換え (頂点表の座標・矩形の幅と高さ) */
export function reshapeDomain(p: Project, polygon: Point[], bulges?: number[] | null): DomainEdit {
  const cur = domainPath(p);
  const ids = edgeIdsOf(p);
  if (polygon.length !== ids.length) throw new Error("reshapeDomain: 頂点の数が変わっています");
  return { path: { polygon, bulges: bulges === undefined ? cur.bulges : bulges }, ids };
}

/** 辺 i の中点 (円弧は弧の中点) */
export function domainEdgeMidpoint(p: Project, i: number): Point {
  return edgeMidpoint(domainPath(p), i) as Point;
}
