import { describe, expect, it } from "vitest";
import { pathArea, type Vec } from "../src/cad/geom";
import {
  bulgesOf,
  circlePath,
  edgeMidpoint,
  ensureCcw,
  hasArcs,
  isSimilarity,
  isValidPath,
  mirror,
  moveVertex,
  packBulges,
  removeVertex,
  reversePath,
  reversedEdgeIndex,
  rotation,
  scaling,
  setBulge,
  signedArea,
  splitEdge,
  transformPath,
  translation,
  applyAffine,
  composeAffine,
} from "../src/cad/path";

const Q = Math.tan(Math.PI / 8); // 90°
const square: Vec[] = [[0, 0], [2, 0], [2, 2], [0, 2]];

const close = (a: Vec, b: Vec) => {
  expect(a[0]).toBeCloseTo(b[0], 12);
  expect(a[1]).toBeCloseTo(b[1], 12);
};

describe("paths", () => {
  it("packs bulges (all zero → none)", () => {
    expect(packBulges([0, 0, 0])).toBeNull();
    expect(packBulges([0, 1e-14, 0.5])).toEqual([0, 0, 0.5]);
    expect(bulgesOf({ polygon: square })).toEqual([0, 0, 0, 0]);
    expect(hasArcs({ polygon: square, bulges: [0, 0, 0, 0] })).toBe(false);
  });

  it("splits a straight edge at the midpoint and an arc into two arcs of the same circle", () => {
    expect(splitEdge({ polygon: square }, 0).polygon).toEqual([[0, 0], [1, 0], [2, 0], [2, 2], [0, 2]]);
    const disc = circlePath([0, 0], 1);
    const s = splitEdge(disc, 0);
    expect(s.polygon).toHaveLength(3);
    // 半円 (1,0)→(-1,0) の反時計回りは上を通る
    close(s.polygon[1] as Vec, [0, 1]);
    expect(s.bulges![0]).toBeCloseTo(Q, 12);
    expect(s.bulges![1]).toBeCloseTo(Q, 12);
    expect(signedArea(s)).toBeCloseTo(Math.PI, 12);
  });

  it("removes a vertex: arcs of the same circle merge, other pairs become straight", () => {
    const s = splitEdge(circlePath([0, 0], 1), 0);
    const back = removeVertex(s, 1)!;
    expect(back.polygon).toHaveLength(2);
    expect(back.bulges![0]).toBeCloseTo(1, 12);
    expect(signedArea(back)).toBeCloseTo(Math.PI, 12);
    // 直線どうし (1 本の直線に)
    const tri = removeVertex({ polygon: square, bulges: [0, 0.3, 0, 0] }, 0)!;
    expect(tri.polygon).toEqual([[2, 0], [2, 2], [0, 2]]);
    expect(tri.bulges).toEqual([0.3, 0, 0]);
    // 最後の辺と最初の辺をつなぐ (頂点 0)。円弧と直線は直線に
    const merged = removeVertex({ polygon: square, bulges: [0.2, 0, 0, 0.3] }, 0)!;
    expect(hasArcs(merged)).toBe(false);
    // 直線だけで 3 点未満にはしない
    expect(removeVertex({ polygon: [[0, 0], [1, 0], [0, 1]] }, 1)).toBeNull();
  });

  it("reverses paths (bulges move to the reversed edge with the opposite sign)", () => {
    const p = { polygon: square, bulges: [0.5, 0, 0, 0] };
    const r = reversePath(p);
    expect(r.polygon).toEqual([[0, 2], [2, 2], [2, 0], [0, 0]]);
    // 元の辺 0 (0,0)→(2,0) は逆向きの辺 (2,0)→(0,0) = 新しい辺 2
    expect(reversedEdgeIndex(4, 2)).toBe(0);
    expect(r.bulges).toEqual([-0, -0, -0.5, -0].map((x) => (x === 0 ? 0 : x)));
    expect(signedArea(r)).toBeCloseTo(-signedArea(p), 12);
    const { path, reversed } = ensureCcw(r);
    expect(reversed).toBe(true);
    expect(signedArea(path)).toBeCloseTo(signedArea(p), 12);
    expect(ensureCcw(p).reversed).toBe(false);
  });

  it("moves vertices and sets bulges, validates paths", () => {
    const m = moveVertex({ polygon: square, bulges: [0, 0.2, 0, 0] }, 2, [3, 3]);
    expect(m.polygon[2]).toEqual([3, 3]);
    expect(m.bulges).toEqual([0, 0.2, 0, 0]);
    expect(setBulge({ polygon: square }, 1, 0.4).bulges).toEqual([0, 0.4, 0, 0]);
    expect(isValidPath({ polygon: square })).toBe(true);
    expect(isValidPath({ polygon: [[0, 0], [1, 0]] })).toBe(false);
    expect(isValidPath(circlePath([0, 0], 1))).toBe(true);
    expect(isValidPath({ polygon: [[0, 0], [1, 0], [2, 0]] })).toBe(false);
    close(edgeMidpoint(circlePath([0, 0], 1), 0), [0, 1]);
  });

  it("transforms: similarity keeps arcs, mirror flips the bulge sign", () => {
    const p = { polygon: square, bulges: [0.5, 0, 0, 0] };
    const rot = transformPath(p, rotation(Math.PI / 2, [1, 1]));
    expect(rot.bulges![0]).toBe(0.5);
    close(rot.polygon[0] as Vec, [2, 0]);
    const mir = transformPath(p, mirror([1, 0], [1, 1]));
    close(mir.polygon[0] as Vec, [2, 0]);
    expect(mir.bulges![0]).toBe(-0.5);
    // 鏡映で向きが逆になるので面積の符号も逆
    expect(pathArea(mir.polygon, mir.bulges)).toBeCloseTo(-pathArea(p.polygon, p.bulges), 12);
    expect(isSimilarity(scaling(2, [1, 1]))).toBe(true);
    expect(isSimilarity({ a: 2, b: 0, c: 0, d: 1, e: 0, f: 0 })).toBe(false);
    const m = composeAffine(translation(1, 0), scaling(2));
    expect(applyAffine(m, [1, 1])).toEqual([3, 2]);
  });
});
