import { describe, expect, it } from "vitest";
import { booleanShapes, overlapArea, shapeArea, shapeProblem, type Shape } from "../src/cad/boolean";
import { segFromBulge, type ArcSeg, type Vec } from "../src/cad/geom";
import { bulgesOf, circlePath, type PathData } from "../src/cad/path";

const rect = (x0: number, y0: number, x1: number, y1: number): PathData => ({ polygon: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]] });
const shape = (outer: PathData, holes: PathData[] = []): Shape => ({ outer, holes });
const circle = (c: Vec, r: number) => shape(circlePath(c, r));

function ok(r: { shapes: Shape[] } | { error: string }): Shape[] {
  if ("error" in r) throw new Error(r.error);
  return r.shapes;
}

/** 経路の辺 (円弧は中心・半径も) */
function arcsOf(p: PathData): ArcSeg[] {
  const b = bulgesOf(p);
  return p.polygon.map((q, i) => segFromBulge(q, p.polygon[(i + 1) % p.polygon.length], b[i])).filter((s): s is ArcSeg => s.kind === "arc");
}

const sortPts = (pts: Vec[]) => pts.map(([x, y]) => [Math.round(x * 1e9) / 1e9, Math.round(y * 1e9) / 1e9]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);

describe("booleans of straight shapes", () => {
  it("cuts a hole and keeps both rings exact", () => {
    const [r] = ok(booleanShapes("difference", shape(rect(0, 0, 4, 4)), [shape(rect(1, 1, 2, 2))]));
    expect(sortPts(r.outer.polygon)).toEqual(sortPts(rect(0, 0, 4, 4).polygon));
    expect(r.holes).toHaveLength(1);
    expect(sortPts(r.holes[0].polygon)).toEqual(sortPts(rect(1, 1, 2, 2).polygon));
    expect(shapeArea(r)).toBeCloseTo(15, 12);
    expect(shapeProblem(r)).toBeNull();
  });

  it("merges adjacent rectangles into one without the shared vertices", () => {
    const [r] = ok(booleanShapes("union", shape(rect(0, 0, 1, 1)), [shape(rect(1, 0, 2, 1))]));
    expect(sortPts(r.outer.polygon)).toEqual(sortPts(rect(0, 0, 2, 1).polygon));
    expect(r.outer.bulges ?? null).toBeNull();
  });

  it("splits into pieces, reports empty results and touching holes", () => {
    const parts = ok(booleanShapes("difference", shape(rect(0, 0, 3, 1)), [shape(rect(1, -1, 2, 2))]));
    expect(parts).toHaveLength(2);
    expect(parts.map(shapeArea)).toEqual([1, 1]);
    expect(booleanShapes("intersection", shape(rect(0, 0, 1, 1)), [shape(rect(2, 0, 3, 1))])).toEqual({ error: "empty" });
    expect(booleanShapes("difference", shape(rect(0, 0, 1, 1)), [shape(rect(-1, -1, 2, 2))])).toEqual({ error: "empty" });
    const diamond: PathData = { polygon: [[2, 0], [3, 1], [2, 2], [1, 1]] };
    expect(booleanShapes("difference", shape(rect(0, 0, 4, 4)), [shape(diamond)])).toEqual({ error: "touching" });
  });

  it("unites several tools at once", () => {
    const [r] = ok(booleanShapes("union", shape(rect(0, 0, 1, 1)), [shape(rect(1, 0, 2, 1)), shape(rect(0, 1, 2, 2))]));
    expect(sortPts(r.outer.polygon)).toEqual(sortPts(rect(0, 0, 2, 2).polygon));
  });
});

describe("booleans with arcs", () => {
  it("keeps the circle of a bite exactly (vertices at the intersections)", () => {
    // 長方形 [0,4]x[0,2] から中心 (2,2)・半径 1 の円を引く: 上の辺に半円のくぼみ
    const [r] = ok(booleanShapes("difference", shape(rect(0, 0, 4, 2)), [circle([2, 2], 1)]));
    expect(sortPts(r.outer.polygon)).toEqual(sortPts([[0, 0], [4, 0], [4, 2], [3, 2], [1, 2], [0, 2]]));
    const arcs = arcsOf(r.outer);
    expect(arcs).toHaveLength(1);
    expect(arcs[0].center[0]).toBeCloseTo(2, 12);
    expect(arcs[0].center[1]).toBeCloseTo(2, 12);
    expect(arcs[0].r).toBeCloseTo(1, 12);
    expect(arcs[0].ccw).toBe(false);
    expect(shapeArea(r)).toBeCloseTo(8 - Math.PI / 2, 12);
  });

  it("unites and intersects overlapping circles into exact arcs", () => {
    const lens = 2 * Math.acos(0.5) - 0.5 * Math.sqrt(3);
    const [u] = ok(booleanShapes("union", circle([0, 0], 1), [circle([1, 0], 1)]));
    expect(arcsOf(u.outer)).toHaveLength(u.outer.polygon.length);
    expect(u.outer.polygon).toHaveLength(2);
    expect(shapeArea(u)).toBeCloseTo(2 * Math.PI - lens, 12);
    const [x] = ok(booleanShapes("intersection", circle([0, 0], 1), [circle([1, 0], 1)]));
    expect(sortPts(x.outer.polygon)).toEqual(sortPts([[0.5, Math.sqrt(3) / 2], [0.5, -Math.sqrt(3) / 2]]));
    expect(shapeArea(x)).toBeCloseTo(lens, 12);
  });

  it("makes a circular hole with the original vertices", () => {
    const [r] = ok(booleanShapes("difference", shape(rect(-2, -2, 2, 2)), [circle([0.5, 0], 1)]));
    expect(r.holes).toHaveLength(1);
    expect(sortPts(r.holes[0].polygon)).toEqual(sortPts([[1.5, 0], [-0.5, 0]]));
    expect(bulgesOf(r.holes[0])).toEqual([-1, -1]);
    expect(shapeArea(r)).toBeCloseTo(16 - Math.PI, 12);
  });

  it("merges arcs of the same circle from different shapes without slivers", () => {
    // 円の上半分と右半分 (どちらも同じ円の円弧) の和 = 3/4 の円
    const upper: PathData = { polygon: [[1, 0], [-1, 0]], bulges: [1, 0] };
    const right: PathData = { polygon: [[0, -1], [0, 1]], bulges: [1, 0] };
    const [r] = ok(booleanShapes("union", shape(upper), [shape(right)]));
    expect(r.holes).toHaveLength(0);
    expect(shapeArea(r)).toBeCloseTo((3 * Math.PI) / 4, 12);
    // 円弧は 1 本 (中心角 270°) と直線 2 本
    const arcs = arcsOf(r.outer);
    expect(arcs).toHaveLength(1);
    expect(r.outer.polygon).toHaveLength(3);
    // 円から同じ円の 1/4 を引くと 3/4 の円 (切れ端なし)
    const quarter: PathData = { polygon: [[0, 0], [1, 0], [0, 1]], bulges: [0, Math.tan(Math.PI / 8), 0] };
    const parts = ok(booleanShapes("difference", circle([0, 0], 1), [shape(quarter)]));
    expect(parts).toHaveLength(1);
    expect(shapeArea(parts[0])).toBeCloseTo((3 * Math.PI) / 4, 12);
  });

  it("subtracts a ring (a tool with a hole) leaving an island", () => {
    const ring = shape(circlePath([0, 0], 2), [circlePath([0, 0], 1)]);
    const parts = ok(booleanShapes("difference", shape(rect(-3, -3, 3, 3)), [ring]));
    expect(parts).toHaveLength(2);
    const areas = parts.map(shapeArea).sort((a, b) => a - b);
    expect(areas[0]).toBeCloseTo(Math.PI, 12);
    expect(areas[1]).toBeCloseTo(36 - 4 * Math.PI, 12);
    const big = parts.find((s) => s.holes.length === 1)!;
    expect(arcsOf(big.holes[0]).every((a) => Math.abs(a.r - 2) < 1e-12)).toBe(true);
  });

  it("keeps tangent vertices of a rounded corner", () => {
    // 右上の角を半径 0.5 で丸めた正方形 ∪ 左下の小さな正方形
    const t = Math.tan(Math.PI / 8);
    const rounded: PathData = { polygon: [[0, 0], [2, 0], [2, 1.5], [1.5, 2], [0, 2]], bulges: [0, 0, t, 0, 0] };
    const [r] = ok(booleanShapes("union", shape(rounded), [shape(rect(-1, -1, 0.5, 0.5))]));
    expect(r.outer.polygon.some(([x, y]) => x === 2 && y === 1.5)).toBe(true);
    expect(r.outer.polygon.some(([x, y]) => x === 1.5 && y === 2)).toBe(true);
    const arcs = arcsOf(r.outer);
    expect(arcs).toHaveLength(1);
    expect(arcs[0].r).toBeCloseTo(0.5, 12);
    expect(shapeArea(r)).toBeCloseTo(4 - 0.25 * (1 - Math.PI / 4) + 1.5 * 1.5 - 0.25, 12);
  });

  it("measures overlaps for choosing the main piece", () => {
    expect(overlapArea(shape(rect(0, 0, 2, 2)), shape(rect(1, 1, 3, 3)))).toBeCloseTo(1, 12);
    expect(overlapArea(circle([0, 0], 1), shape(rect(0, -2, 2, 2)))).toBeCloseTo(Math.PI / 2, 4);
  });
});

describe("shape checks", () => {
  it("finds holes outside, crossing, touching or nested", () => {
    expect(shapeProblem(shape(rect(0, 0, 4, 4), [rect(1, 1, 2, 2)]))).toBeNull();
    expect(shapeProblem(shape(rect(0, 0, 4, 4), [rect(5, 1, 6, 2)]))).toBe("invalid");
    expect(shapeProblem(shape(rect(0, 0, 4, 4), [rect(1, 1, 2, 2), rect(1.5, 1.5, 3, 3)]))).toBe("touching");
    expect(shapeProblem(shape(rect(0, 0, 4, 4), [rect(1, 1, 3, 3), rect(1.5, 1.5, 2, 2)]))).toBe("invalid");
    expect(shapeProblem(shape(rect(0, 0, 4, 4), [{ polygon: [[2, 0], [3, 1], [1, 1]] }]))).toBe("touching");
    // 8 の字 (自分と交わる外周)
    expect(shapeProblem(shape({ polygon: [[0, 0], [4, 0], [4, 2], [2, -1], [0, 2]] }))).toBe("touching");
    // 円 (2 つの半円) はそのまま使える
    expect(shapeProblem(circle([0, 0], 1))).toBeNull();
  });
});
