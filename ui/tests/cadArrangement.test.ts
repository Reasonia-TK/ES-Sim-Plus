import { describe, expect, it } from "vitest";
import { Arrangement, simplifyPath, splitAtIntersections } from "../src/cad/arrangement";
import { bulgeThrough, paramOf, pathArea, segFromBulge, subSeg, tangentAt, type Seg, type Vec } from "../src/cad/geom";

const line = (a: Vec, b: Vec): Seg => ({ kind: "line", a, b });
const circle = (c: Vec, r: number): Seg[] => [segFromBulge([c[0] + r, c[1]], [c[0] - r, c[1]], 1), segFromBulge([c[0] - r, c[1]], [c[0] + r, c[1]], 1)];
const area = (p: { polygon: Vec[]; bulges?: number[] | null }) => pathArea(p.polygon, p.bulges);

describe("arc helpers", () => {
  it("builds arcs through three points", () => {
    // 下を通る半円は反時計回り (正)、上を通るのは時計回り (負)
    expect(bulgeThrough([0, 0], [1, -1], [2, 0])).toBeCloseTo(1, 12);
    expect(bulgeThrough([0, 0], [1, 1], [2, 0])).toBeCloseTo(-1, 12);
    expect(bulgeThrough([0, 0], [1, 0], [2, 0])).toBe(0);
    const q = Math.tan(Math.PI / 8);
    expect(bulgeThrough([1, 0], [Math.SQRT1_2, Math.SQRT1_2], [0, 1])).toBeCloseTo(q, 12);
  });

  it("parameters, sub-arcs and end tangents", () => {
    const arc = segFromBulge([1, 0], [-1, 0], 1);
    expect(paramOf(arc, [0, 1])).toBeCloseTo(0.5, 12);
    expect(paramOf(line([0, 0], [2, 0]), [0.5, 3])).toBeCloseTo(0.25, 12);
    const half = subSeg(arc, 0, 0.5);
    expect(half.b[0]).toBeCloseTo(0, 12);
    expect(half.b[1]).toBeCloseTo(1, 12);
    const ta = tangentAt(arc, "a");
    expect(ta.angle).toBeCloseTo(Math.PI / 2, 12);
    expect(ta.curvature).toBeCloseTo(1, 12);
    const tb = tangentAt(arc, "b");
    // 終点 (-1, 0) から逆向き (時計回り) に出ると上向き、右に曲がる
    expect(tb.angle).toBeCloseTo(Math.PI / 2, 12);
    expect(tb.curvature).toBeCloseTo(-1, 12);
  });
});

describe("splitting at intersections", () => {
  it("splits crossing lines and T junctions, keeps the original direction", () => {
    const parts = splitAtIntersections([line([0, 0], [2, 0]), line([1, -1], [1, 1]), line([2, 0], [2, 1])]);
    expect(parts[0]).toHaveLength(2);
    expect(parts[1]).toHaveLength(2);
    expect(parts[2]).toHaveLength(1);
    expect(parts[0][0].b).toEqual([1, 0]);
  });
});

describe("faces", () => {
  it("finds a square made of four lines", () => {
    const arr = new Arrangement([line([0, 0], [1, 0]), line([1, 0], [1, 1]), line([1, 1], [0, 1]), line([0, 1], [0, 0])]);
    const f = arr.faceAt([0.5, 0.5])!;
    expect(f.polygon).toHaveLength(4);
    expect(area(f)).toBeCloseTo(1, 12);
    expect(arr.faceAt([2, 2])).toBeNull();
  });

  it("picks the smallest cell of crossing lines (#)", () => {
    const arr = new Arrangement([line([-1, 0], [3, 0]), line([-1, 1], [3, 1]), line([0, -1], [0, 3]), line([1, -1], [1, 3])]);
    const f = arr.faceAt([0.3, 0.6])!;
    expect(f.polygon).toHaveLength(4);
    expect(area(f)).toBeCloseTo(1, 12);
    // 端がつながっていない所 (外側の枝) は面にならない
    expect(arr.faceAt([2, 0.5])).toBeNull();
  });

  it("splits a square with a diagonal and ignores dangling lines", () => {
    const sq = [line([0, 0], [2, 0]), line([2, 0], [2, 2]), line([2, 2], [0, 2]), line([0, 2], [0, 0])];
    const tri = new Arrangement([...sq, line([0, 0], [2, 2])]).faceAt([1.5, 0.5])!;
    expect(tri.polygon).toHaveLength(3);
    expect(area(tri)).toBeCloseTo(2, 12);
    const dangling = new Arrangement([...sq, line([1, 1], [1.5, 1.2])]).faceAt([0.5, 0.5])!;
    expect(dangling.polygon).toHaveLength(4);
    expect(area(dangling)).toBeCloseTo(4, 12);
    // 辺に重なる線 (一部だけ重なる) があっても同じ正方形
    const overlap = new Arrangement([...sq, line([0.5, 0], [1.5, 0])]).faceAt([1, 1])!;
    expect(overlap.polygon).toHaveLength(4);
    expect(area(overlap)).toBeCloseTo(4, 12);
  });

  it("keeps arcs: a disc cut by a chord gives a half disc", () => {
    const arr = new Arrangement([...circle([0, 0], 1), line([-2, 0], [2, 0])]);
    const up = arr.faceAt([0.1, 0.5])!;
    expect(up.polygon).toHaveLength(2);
    expect(area(up)).toBeCloseTo(Math.PI / 2, 12);
    const whole = new Arrangement(circle([0, 0], 1)).faceAt([0, 0.2])!;
    expect(area(whole)).toBeCloseTo(Math.PI, 12);
    // 円と正方形の重なり: 正方形の角の外 (円の外) の部分
    const sq = [line([0, 0], [2, 0]), line([2, 0], [2, 2]), line([2, 2], [0, 2]), line([0, 2], [0, 0])];
    const corner = new Arrangement([...sq, ...circle([0, 0], 1)]).faceAt([1.5, 1.5])!;
    expect(area(corner)).toBeCloseTo(4 - Math.PI / 4, 12);
    const quarter = new Arrangement([...sq, ...circle([0, 0], 1)]).faceAt([0.3, 0.3])!;
    expect(area(quarter)).toBeCloseTo(Math.PI / 4, 12);
  });

  it("uses the outer boundary when the face has an island", () => {
    const sq = [line([0, 0], [4, 0]), line([4, 0], [4, 4]), line([4, 4], [0, 4]), line([0, 4], [0, 0])];
    const f = new Arrangement([...sq, ...circle([2, 2], 1)]).faceAt([0.5, 0.5])!;
    expect(area(f)).toBeCloseTo(16, 12);
  });
});

describe("simplify", () => {
  it("merges collinear lines and arcs of the same circle", () => {
    const s = simplifyPath({ polygon: [[0, 0], [1, 0], [2, 0], [2, 2], [0, 2]], bulges: [0, 0, 0, 0, 0] }, 1e-12);
    expect(s.polygon).toHaveLength(4);
    expect(s.polygon).not.toContainEqual([1, 0]);
    expect(area(s)).toBeCloseTo(4, 12);
    const q = Math.tan(Math.PI / 8);
    const disc = simplifyPath({ polygon: [[1, 0], [0, 1], [-1, 0]], bulges: [q, q, 1] }, 1e-12);
    expect(disc.polygon).toHaveLength(2);
    expect(area(disc)).toBeCloseTo(Math.PI, 12);
  });
});
