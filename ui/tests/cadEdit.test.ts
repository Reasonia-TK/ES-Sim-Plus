import { describe, expect, it } from "vitest";
import { chamferCorner, extendEnd, filletCorner, filletLines, offsetPath, simplifyChain, trimAt, type PathPts } from "../src/cad/edit";
import { pathArea, segFromBulge, type LineSeg, type Vec } from "../src/cad/geom";

const line = (a: Vec, b: Vec): LineSeg => ({ kind: "line", a, b });
const square: PathPts = { points: [[0, 0], [2, 0], [2, 2], [0, 2]], bulges: [0, 0, 0, 0], closed: true };
const area = (p: PathPts) => pathArea(p.points, p.closed ? p.bulges : [...p.bulges, 0]);
const ok = <T,>(r: { ok: true; value: T } | { ok: false; error: string }): T => {
  if (!r.ok) throw new Error(r.error);
  return r.value;
};

describe("fillet and chamfer", () => {
  it("rounds a square corner with a tangent arc", () => {
    const f = ok(filletCorner(square, 1, 0.5));
    expect(f.points).toHaveLength(5);
    expect(f.points[1]).toEqual([1.5, 0]);
    expect(f.points[2][0]).toBeCloseTo(2, 12);
    expect(f.points[2][1]).toBeCloseTo(0.5, 12);
    expect(f.bulges[1]).toBeCloseTo(Math.tan(Math.PI / 8), 12);
    expect(area(f)).toBeCloseTo(4 - 0.25 * (1 - Math.PI / 4), 12);
    // 凹んだ角 (右に曲がる) は逆向きの円弧
    const L: PathPts = { points: [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], bulges: [0, 0, 0, 0, 0, 0], closed: true };
    const c = ok(filletCorner(L, 3, 0.25));
    expect(c.bulges[3]).toBeLessThan(0);
    expect(area(c)).toBeCloseTo(3 + 0.0625 * (1 - Math.PI / 4), 12);
  });

  it("refuses radii that are too large, arcs and straight vertices", () => {
    expect(filletCorner(square, 1, 2.5)).toEqual({ ok: false, error: "tooLarge" });
    expect(filletCorner({ ...square, bulges: [0.3, 0, 0, 0] }, 1, 0.1)).toEqual({ ok: false, error: "notStraight" });
    const straight: PathPts = { points: [[0, 0], [1, 0], [2, 0], [2, 2]], bulges: [0, 0, 0, 0], closed: true };
    expect(filletCorner(straight, 1, 0.1)).toEqual({ ok: false, error: "collinear" });
    const open: PathPts = { points: [[0, 0], [2, 0], [2, 2]], bulges: [0, 0], closed: false };
    expect(filletCorner(open, 0, 0.1)).toEqual({ ok: false, error: "notStraight" });
    expect(ok(filletCorner(open, 1, 0.5)).points).toHaveLength(4);
  });

  it("chamfers a corner", () => {
    const c = ok(chamferCorner(square, 2, 0.5));
    expect(c.points).toHaveLength(5);
    expect(c.bulges).toEqual([0, 0, 0, 0, 0]);
    expect(area(c)).toBeCloseTo(4 - 0.125, 12);
    expect(chamferCorner(square, 2, 3)).toEqual({ ok: false, error: "tooLarge" });
  });

  it("fillets two crossing sketch lines on the clicked sides", () => {
    const r = ok(filletLines(line([-2, 0], [2, 0]), [1.5, 0.1], line([0, -2], [0, 2]), [0.1, 1.5], 0.5));
    expect(r.line1.a).toEqual([2, 0]);
    expect(r.line1.b[0]).toBeCloseTo(0.5, 12);
    expect(r.line1.b[1]).toBeCloseTo(0, 12);
    expect(r.line2.a[0]).toBeCloseTo(0, 12);
    expect(r.line2.a[1]).toBeCloseTo(0.5, 12);
    expect(r.line2.b).toEqual([0, 2]);
    // 角の側にふくらむ時計回りの円弧 (中心 (0.5, 0.5))
    const arc = segFromBulge(r.arc!.a, r.arc!.b, r.arc!.bulge);
    expect(arc.kind === "arc" && arc.center[0]).toBeCloseTo(0.5, 12);
    expect(r.arc!.bulge).toBeLessThan(0);
    // 半径 0 は角までそろえる
    const sharp = ok(filletLines(line([-2, 0], [2, 0]), [-1, 0], line([0, -2], [0, 2]), [0, -1], 0));
    expect(sharp.arc).toBeNull();
    expect(sharp.line1.b).toEqual([0, 0]);
    expect(filletLines(line([0, 0], [1, 0]), [0.5, 0], line([0, 1], [1, 1]), [0.5, 1], 0.1)).toEqual({ ok: false, error: "collinear" });
  });
});

describe("offset", () => {
  it("offsets a square outwards with mitred corners and inwards", () => {
    const out = ok(offsetPath(square, -0.5));
    expect(area(out)).toBeCloseTo(9, 12);
    expect(out.points).toHaveLength(4);
    const inn = ok(offsetPath(square, 0.5));
    expect(area(inn)).toBeCloseTo(1, 12);
    expect(offsetPath(square, 1.5).ok).toBe(false);
  });

  it("offsets arcs as concentric arcs and rounds outer corners next to arcs", () => {
    const disc: PathPts = { points: [[1, 0], [-1, 0]], bulges: [1, 1], closed: true };
    expect(area(ok(offsetPath(disc, -0.5)))).toBeCloseTo(Math.PI * 2.25, 12);
    expect(area(ok(offsetPath(disc, 0.5)))).toBeCloseTo(Math.PI * 0.25, 12);
    expect(offsetPath(disc, 1.2)).toEqual({ ok: false, error: "tooLarge" });
    // 右の辺を外へふくらませた正方形を外へ: 角 (円弧と直線の間) は丸める
    const bump: PathPts = { points: [[0, 0], [2, 0], [2, 2], [0, 2]], bulges: [0, 0.4, 0, 0], closed: true };
    const o = ok(offsetPath(bump, -0.2));
    expect(area(o)).toBeGreaterThan(area(bump));
    expect(o.points.length).toBeGreaterThan(4);
  });

  it("offsets open polylines to either side", () => {
    const L: PathPts = { points: [[0, 0], [2, 0], [2, 2]], bulges: [0, 0], closed: false };
    const left = ok(offsetPath(L, 0.5));
    expect(left.points.map((q) => q.map((v) => Math.round(v * 1e9) / 1e9))).toEqual([
      [0, 0.5],
      [1.5, 0.5],
      [1.5, 2],
    ]);
    const right = ok(offsetPath(L, -0.5));
    expect(right.points.map((q) => q.map((v) => Math.round(v * 1e9) / 1e9))).toEqual([
      [0, -0.5],
      [2.5, -0.5],
      [2.5, 2],
    ]);
  });
});

describe("trim and extend", () => {
  it("trims the clicked part between cutting edges", () => {
    const cut = [line([1, -1], [1, 1]), line([3, -1], [3, 1])];
    const r = ok(trimAt([line([0, 0], [4, 0])], false, cut, [2, 0.05]));
    expect(r).toHaveLength(2);
    expect(r[0][0].b).toEqual([1, 0]);
    expect(r[1][0].a).toEqual([3, 0]);
    // 端の区切りを消すと 1 本だけ残る
    const end = ok(trimAt([line([0, 0], [4, 0])], false, cut, [3.5, 0]));
    expect(end).toHaveLength(1);
    // 残りは (0,0) → (1,0) → (3,0) のつながった 2 本 (同じ直線の続きは simplifyChain で 1 本に)
    expect(end[0][0].a).toEqual([0, 0]);
    expect(end[0][end[0].length - 1].b).toEqual([3, 0]);
    expect(simplifyChain(end[0])).toEqual([{ kind: "line", a: [0, 0], b: [3, 0] }]);
    expect(trimAt([line([0, 0], [4, 0])], false, [line([0, 5], [4, 5])], [2, 0])).toEqual({ ok: false, error: "noCutter" });
  });

  it("trims a circle into an arc (needs two cutting points)", () => {
    const circle = [segFromBulge([1, 0], [-1, 0], 1), segFromBulge([-1, 0], [1, 0], 1)];
    const r = ok(trimAt(circle, true, [line([-2, 0], [2, 0])], [0, 0.9]));
    expect(r).toHaveLength(1);
    const rest = r[0];
    // 残りは下の半円
    expect(rest.every((s) => s.kind === "arc")).toBe(true);
    expect(Math.min(...rest.map((s) => Math.min(s.a[1], s.b[1])))).toBeCloseTo(0, 12);
    expect(trimAt(circle, true, [line([-2, 1], [2, 1])], [0, 0.9])).toEqual({ ok: false, error: "noCutter" });
  });

  it("extends lines and arcs to the next boundary", () => {
    const r = ok(extendEnd(line([0, 0], [1, 0]), "b", [line([3, -1], [3, 1])]));
    expect(r).toEqual({ kind: "line", a: [0, 0], b: [3, 0] });
    expect(extendEnd(line([0, 0], [1, 0]), "a", [line([3, -1], [3, 1])])).toEqual({ ok: false, error: "noCutter" });
    const quarter = segFromBulge([1, 0], [0, 1], Math.tan(Math.PI / 8));
    const e = ok(extendEnd(quarter, "b", [line([-2, 0], [-0.5, 0])]));
    expect(e.b[0]).toBeCloseTo(-1, 12);
    expect(e.b[1]).toBeCloseTo(0, 12);
    expect(e.kind === "arc" && e.ccw).toBe(true);
  });
});
