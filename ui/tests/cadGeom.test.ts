import { describe, expect, it } from "vitest";
import {
  arcSweep,
  bulgeOf,
  closestPoint,
  discretize,
  intersections,
  midpoint,
  pathArea,
  pathBounds,
  pathPolygon,
  perpendicularFoot,
  pointInPath,
  segFromBulge,
  segLength,
  tangentPoints,
  type ArcSeg,
  type Vec,
} from "../src/cad/geom";

const close = (a: Vec, b: Vec, eps = 1e-12) => {
  expect(a[0]).toBeCloseTo(b[0], -Math.log10(eps));
  expect(a[1]).toBeCloseTo(b[1], -Math.log10(eps));
};

describe("bulge arcs", () => {
  it("builds a quarter arc (bulge tan(π/8)) with the centre on the left for ccw", () => {
    const s = segFromBulge([1, 0], [0, 1], Math.tan(Math.PI / 8)) as ArcSeg;
    expect(s.kind).toBe("arc");
    close(s.center, [0, 0]);
    expect(s.r).toBeCloseTo(1, 12);
    expect(s.ccw).toBe(true);
    expect(arcSweep(s)).toBeCloseTo(Math.PI / 2, 12);
    expect(segLength(s)).toBeCloseTo(Math.PI / 2, 12);
    close(midpoint(s), [Math.SQRT1_2, Math.SQRT1_2]);
    expect(bulgeOf(s)).toBeCloseTo(Math.tan(Math.PI / 8), 12);
  });

  it("handles semicircles, major arcs and clockwise arcs", () => {
    const semi = segFromBulge([0, 0], [2, 0], 1) as ArcSeg;
    close(semi.center, [1, 0]);
    // 反時計回りに 0 → 2 へ回る半円は下側を通る
    close(midpoint(semi), [1, -1]);
    const cw = segFromBulge([0, 0], [2, 0], -1) as ArcSeg;
    close(midpoint(cw), [1, 1]);
    const major = segFromBulge([1, 0], [0, 1], Math.tan((3 * Math.PI) / 8)) as ArcSeg;
    expect(arcSweep(major)).toBeCloseTo((3 * Math.PI) / 2, 12);
    close(major.center, [1, 1]);
    expect(segFromBulge([0, 0], [1, 0], 0).kind).toBe("line");
  });

  it("area of a path counts the circular segments (a disc from two semicircles)", () => {
    expect(pathArea([[1, 0], [-1, 0]], [1, 1])).toBeCloseTo(Math.PI, 12);
    // 正方形の右の辺を外に半円でふくらませる
    const sq: Vec[] = [[0, 0], [2, 0], [2, 2], [0, 2]];
    expect(pathArea(sq)).toBeCloseTo(4, 12);
    expect(pathArea(sq, [0, 1, 0, 0])).toBeCloseTo(4 + Math.PI / 2, 12);
    expect(pathArea(sq, [0, -1, 0, 0])).toBeCloseTo(4 - Math.PI / 2, 12);
  });

  it("bounds include the arc bulge, and point-in-path follows the arcs", () => {
    const b = pathBounds([[1, 0], [-1, 0]], [1, 1]);
    expect(b.x0).toBeCloseTo(-1, 12);
    expect(b.y0).toBeCloseTo(-1, 12);
    expect(b.x1).toBeCloseTo(1, 12);
    expect(b.y1).toBeCloseTo(1, 12);
    const sq: Vec[] = [[0, 0], [2, 0], [2, 2], [0, 2]];
    expect(pointInPath([2.5, 1], sq, [0, 1, 0, 0])).toBe(true);
    expect(pointInPath([2.5, 1], sq)).toBe(false);
    expect(pointInPath([1.8, 1], sq, [0, -1, 0, 0])).toBe(false);
  });

  it("discretizes arcs within the chord error and keeps the end points exact", () => {
    const s = segFromBulge([1, 0], [-1, 0], 1) as ArcSeg;
    const pts = discretize(s, 1e-3);
    expect(pts[0]).toEqual([1, 0]);
    expect(pts[pts.length - 1]).toEqual([-1, 0]);
    for (let i = 0; i + 1 < pts.length; i++) {
      const m: Vec = [(pts[i][0] + pts[i + 1][0]) / 2, (pts[i][1] + pts[i + 1][1]) / 2];
      expect(1 - Math.hypot(m[0], m[1])).toBeLessThanOrEqual(1e-3 + 1e-15);
    }
    expect(pathPolygon([[0, 0], [1, 0], [1, 1]])).toEqual([[0, 0], [1, 0], [1, 1]]);
  });
});

describe("closest points, feet and tangents", () => {
  it("projects onto lines, arcs and circles", () => {
    const l = { kind: "line" as const, a: [0, 0] as Vec, b: [2, 0] as Vec };
    const c = closestPoint(l, [1, 1]);
    close(c.point, [1, 0]);
    expect(c.u).toBeCloseTo(0.5, 12);
    close(closestPoint(l, [3, 1]).point, [2, 0]);
    const arc = segFromBulge([1, 0], [0, 1], Math.tan(Math.PI / 8));
    close(closestPoint(arc, [2, 2]).point, [Math.SQRT1_2, Math.SQRT1_2]);
    // 円弧の外の角度なら近い端点
    close(closestPoint(arc, [-1, -0.1]).point, [0, 1]);
    close(closestPoint({ kind: "circle", center: [0, 0], r: 2 }, [0, -5]).point, [0, -2]);
  });

  it("perpendicular feet extend the line, tangent points come in pairs", () => {
    const l = { kind: "line" as const, a: [0, 0] as Vec, b: [1, 0] as Vec };
    close(perpendicularFoot(l, [5, 3])!, [5, 0]);
    close(perpendicularFoot({ kind: "circle", center: [0, 0], r: 1 }, [3, 0])!, [1, 0]);
    const tp = tangentPoints({ kind: "circle", center: [0, 0], r: 1 }, [2, 0]);
    expect(tp).toHaveLength(2);
    for (const q of tp) {
      // 接点では半径と接線が直交する
      expect(q[0] * (2 - q[0]) + q[1] * (0 - q[1])).toBeCloseTo(0, 12);
    }
    expect(tangentPoints({ kind: "circle", center: [0, 0], r: 1 }, [0.5, 0])).toHaveLength(0);
  });
});

describe("intersections", () => {
  const line = (a: Vec, b: Vec) => ({ kind: "line" as const, a, b });
  it("line × line (segment and infinite)", () => {
    close(intersections(line([0, 0], [2, 2]), line([0, 2], [2, 0]))[0], [1, 1]);
    expect(intersections(line([0, 0], [1, 0]), line([2, -1], [2, 1]))).toHaveLength(0);
    close(intersections(line([0, 0], [1, 0]), line([2, -1], [2, 1]), true)[0], [2, 0]);
    expect(intersections(line([0, 0], [1, 0]), line([0, 1], [1, 1]), true)).toHaveLength(0);
  });

  it("line × circle / arc, including tangency", () => {
    const circle = { kind: "circle" as const, center: [0, 0] as Vec, r: 1 };
    const pts = intersections(line([-2, 0], [2, 0]), circle);
    expect(pts).toHaveLength(2);
    close(pts[0], [-1, 0]);
    close(pts[1], [1, 0]);
    expect(intersections(line([-2, 1], [2, 1]), circle)).toHaveLength(1);
    expect(intersections(line([-2, 1.01], [2, 1.01]), circle)).toHaveLength(0);
    // 上半分の円弧 (1,0) → (-1,0) とは y = 0.5 で 2 点、y = -0.5 では交わらない
    const upper = segFromBulge([1, 0], [-1, 0], 1);
    expect(intersections(line([-2, 0.5], [2, 0.5]), upper)).toHaveLength(2);
    expect(intersections(line([-2, -0.5], [2, -0.5]), upper)).toHaveLength(0);
    expect(intersections(line([-2, -0.5], [2, -0.5]), upper, true)).toHaveLength(2);
  });

  it("circle × circle and arc × arc", () => {
    const a = { kind: "circle" as const, center: [0, 0] as Vec, r: 1 };
    const b = { kind: "circle" as const, center: [1, 0] as Vec, r: 1 };
    const pts = intersections(a, b);
    expect(pts).toHaveLength(2);
    for (const q of pts) expect(q[0]).toBeCloseTo(0.5, 12);
    expect(intersections(a, { kind: "circle", center: [3, 0], r: 1 })).toHaveLength(0);
    expect(intersections(a, { kind: "circle", center: [2, 0], r: 1 })).toHaveLength(1);
    const upper = segFromBulge([1, 0], [-1, 0], 1);
    const arcB = segFromBulge([2, 0], [0, 0], 1);
    const q = intersections(upper, arcB);
    expect(q).toHaveLength(1);
    expect(q[0][1]).toBeGreaterThan(0);
  });
});
