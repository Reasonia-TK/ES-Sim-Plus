import { produce } from "immer";
import { describe, expect, it } from "vitest";
import { setEdgeType, oppositeEdge, periodicPartnerOf } from "../src/model/boundaryOps";
import {
  applyDomainEdit,
  moveDomainVertex,
  periodicPartners,
  removeDomainVertex,
  reshapeDomain,
  setDomainBulge,
  splitDomainEdge,
} from "../src/model/domainOps";
import { axisEdges, edgeIdsOf, edgeIndexOf, ensureEdgeIds, newProject, nextEdgeId, normalizeProject, type Project } from "../src/model/project";
import { findDomainEdgeAt, findRegionAt } from "../src/graphics/hitTest";

function withRefs(): Project {
  const p = newProject();
  p.pic = { reflect_edges: [1], fn: { edges: [1, 3] } };
  p.particles = { fn: { edges: [3] } };
  p.dsmc = { boundaries: [{ edges: [1], type: "wall" }, { edges: [], p1: [0, 0], p2: [0.01, 0] }] };
  return p;
}

describe("edge ids", () => {
  it("are assigned on load and new documents", () => {
    expect(newProject().geometry.domain.edge_ids).toEqual(["e1", "e2", "e3", "e4"]);
    const { project } = normalizeProject({ geometry: { domain: { polygon: [[0, 0], [1, 0], [0, 1]] } }, mesh: { size: 0.1 } });
    expect(project.geometry.domain.edge_ids).toEqual(["e1", "e2", "e3"]);
    // 壊れた ID (数が合わない・重複) は振り直す
    const bad = newProject();
    bad.geometry.domain.edge_ids = ["a", "a", "b", "c"];
    expect(edgeIdsOf(bad)).toEqual(["e1", "e2", "e3", "e4"]);
    ensureEdgeIds(bad);
    expect(bad.geometry.domain.edge_ids).toEqual(["e1", "e2", "e3", "e4"]);
    expect(nextEdgeId(["e1", "e7", "x"])).toBe("e8");
    // 円弧を含めば 2 頂点のドメインも読める
    const disc = normalizeProject({ geometry: { domain: { polygon: [[1, 0], [-1, 0]], bulges: [1, 1] } }, mesh: { size: 0.1 } }).project;
    expect(disc.geometry.domain.edge_ids).toEqual(["e1", "e2"]);
  });
});

describe("domain edits remap every edge reference", () => {
  it("splitting an edge keeps the condition on both halves", () => {
    const p = produce(withRefs(), (d) => void applyDomainEdit(d, splitDomainEdge(d as Project, 1)));
    expect(p.geometry.domain.polygon).toHaveLength(5);
    expect(p.geometry.domain.edge_ids).toEqual(["e1", "e2", "e5", "e3", "e4"]);
    // 右の辺 (100 V) は 1 と 2 に、左の辺 (0 V) は 3 → 4
    expect(p.geometry.boundaries.map((b) => b.edges)).toEqual([[4], [1, 2]]);
    expect((p.pic as { reflect_edges: number[] }).reflect_edges).toEqual([1, 2]);
    expect((p.pic as { fn: { edges: number[] } }).fn.edges).toEqual([1, 2, 4]);
    expect((p.particles as { fn: { edges: number[] } }).fn.edges).toEqual([4]);
    const dsmc = p.dsmc as { boundaries: { edges: number[] }[] };
    expect(dsmc.boundaries[0].edges).toEqual([1, 2]);
    expect(dsmc.boundaries[1].edges).toEqual([]);
    expect(edgeIndexOf(p, "e5")).toBe(2);
  });

  it("removing a vertex drops the conditions of the removed edge", () => {
    let p = produce(withRefs(), (d) => void applyDomainEdit(d, splitDomainEdge(d as Project, 1)));
    // 頂点 2 (右の辺の中点) を消すと e5 (後ろ半分) が消え、e2 だけが残る
    let report = { periodicRemoved: 0, boundariesRemoved: 0 };
    p = produce(p, (d) => void (report = applyDomainEdit(d, removeDomainVertex(d as Project, 2)!)));
    expect(p.geometry.domain.edge_ids).toEqual(["e1", "e2", "e3", "e4"]);
    expect(p.geometry.boundaries.map((b) => b.edges)).toEqual([[3], [1]]);
    expect(report.boundariesRemoved).toBe(0);
    // 頂点 i を消すと辺 i-1 と i が辺 i-1 の ID で 1 本になる。右下の頂点 (1) を消すと右の辺 (e2) が消え、100 V の
    // 境界条件が外れる
    p = produce(p, (d) => void (report = applyDomainEdit(d, removeDomainVertex(d as Project, 1)!)));
    expect(p.geometry.domain.polygon).toHaveLength(3);
    expect(p.geometry.domain.edge_ids).toEqual(["e1", "e3", "e4"]);
    expect(p.geometry.boundaries.map((b) => b.edges)).toEqual([[2]]);
    expect(report.boundariesRemoved).toBe(1);
  });

  it("keeps ids through moves, bulges and reshapes, and restores the orientation", () => {
    const p0 = withRefs();
    let p = produce(p0, (d) => void applyDomainEdit(d, setDomainBulge(d as Project, 2, 0.3)));
    expect(p.geometry.domain.bulges).toEqual([0, 0, 0.3, 0]);
    p = produce(p, (d) => void applyDomainEdit(d, moveDomainVertex(d as Project, 2, [0.12, 0.06])));
    expect(p.geometry.domain.edge_ids).toEqual(["e1", "e2", "e3", "e4"]);
    // 時計回りに並べ替えた形を渡すと反時計回りに戻し、辺の ID と参照も付いていく
    const cw = produce(p0, (d) =>
      void applyDomainEdit(d, { path: { polygon: [[0, 0], [0, 0.05], [0.1, 0.05], [0.1, 0]] }, ids: ["e4", "e3", "e2", "e1"] }),
    );
    const ids = cw.geometry.domain.edge_ids!;
    expect(cw.geometry.domain.polygon[0]).toEqual([0.1, 0]);
    // 左の辺 (0 V) は ID e4 のまま
    const left = ids.indexOf("e4");
    expect(cw.geometry.domain.polygon[left]).toEqual([0, 0.05]);
    expect(cw.geometry.boundaries.find((b) => b.voltage === 0)!.edges).toEqual([left]);
    expect(() => reshapeDomain(p0, [[0, 0], [1, 0], [1, 1]])).toThrow();
  });

  it("drops periodic pairs that can no longer be paired and strips axis edges", () => {
    let p = produce(newProject(), (d) => void setEdgeType(d, 0, "periodic"));
    expect(periodicPartnerOf(p, 0)).toBe(2);
    let report = { periodicRemoved: 0, boundariesRemoved: 0 };
    p = produce(p, (d) => void (report = applyDomainEdit(d, splitDomainEdge(d as Project, 0))));
    expect(report.periodicRemoved).toBe(1);
    expect(p.geometry.boundaries.some((b) => b.type === "periodic")).toBe(false);
    // 軸対称: 軸 (y = 0) の上の辺には条件を付けない
    const rz = produce(newProject(), (d) => {
      d.coord = "rz";
      d.geometry.boundaries.push({ edges: [0], type: "symmetry" });
    });
    const after = produce(rz, (d) => void applyDomainEdit(d, moveDomainVertex(d as Project, 2, [0.1, 0.06])));
    expect(after.geometry.boundaries.some((b) => b.edges.includes(0))).toBe(false);
  });
});

describe("periodic partners and axis edges", () => {
  it("pairs parallel edges of the same length", () => {
    const p = newProject();
    expect(periodicPartners(p, 0)).toEqual([2]);
    expect(oppositeEdge(p, 1)).toBe(3);
    // 六角形: 向かいの辺が 1 本ずつ
    const hex = newProject();
    hex.geometry.domain.polygon = Array.from({ length: 6 }, (_, k) => [Math.cos((k * Math.PI) / 3), Math.sin((k * Math.PI) / 3)] as [number, number]);
    hex.geometry.domain.edge_ids = null;
    hex.geometry.boundaries = [];
    expect(periodicPartners(hex, 0)).toEqual([3]);
    // 円弧の辺は周期にできない
    const arc = produce(p, (d) => void applyDomainEdit(d, setDomainBulge(d as Project, 2, 0.2)));
    expect(periodicPartners(arc, 0)).toEqual([]);
    // 候補が複数なら相手を選ぶ
    const comb = newProject();
    comb.geometry.domain.polygon = [[0, 0], [3, 0], [3, 1], [2, 1], [2, 2], [1, 2], [1, 1], [0, 1]];
    comb.geometry.domain.edge_ids = null;
    comb.geometry.boundaries = [];
    expect(periodicPartners(comb, 1)).toEqual([3, 5, 7]);
    expect(oppositeEdge(comb, 1)).toBeNull();
    const paired = produce(comb, (d) => void setEdgeType(d, 1, "periodic", 5));
    expect(paired.geometry.boundaries).toEqual([{ edges: [1, 5], type: "periodic" }]);
    expect(produce(comb, (d) => void setEdgeType(d, 1, "periodic", 2)).geometry.boundaries).toEqual([]);
  });

  it("finds every straight edge on the axis", () => {
    const p = newProject();
    p.coord = "rz";
    p.geometry.domain.polygon = [[0, 0], [0.05, 0], [0.1, 0], [0.1, 0.05], [0, 0.05]];
    expect(axisEdges(p)).toEqual([0, 1]);
    p.geometry.domain.bulges = [0, -0.2, 0, 0, 0];
    expect(axisEdges(p)).toEqual([0]);
  });
});

describe("hit tests with arcs", () => {
  it("picks domain edges (arcs included) and bulged regions", () => {
    const p = newProject();
    p.geometry.domain.bulges = [0, 0, 0.3, 0];
    // 上の辺 (0.1,0.05)→(0,0.05) は外 (上) へふくらむ
    expect(findDomainEdgeAt([0.05, 0.05 + 0.0149], p, 1e-3)).toBe(2);
    expect(findDomainEdgeAt([0.05, 0.05], p, 1e-3)).toBeNull();
    expect(findDomainEdgeAt([0.1, 0.02], p, 1e-3)).toBe(1);
    const r = { id: "r", type: "conductor" as const, polygon: [[0.03, 0.02], [0.01, 0.02]] as [number, number][], bulges: [1, 1] };
    expect(findRegionAt([0.02, 0.028], [r], 1e-4)?.id).toBe("r");
    expect(findRegionAt([0.02, 0.0315], [r], 1e-4)).toBeNull();
  });
});
