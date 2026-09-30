import { describe, expect, it } from "vitest";
import { parseCoordInput } from "../src/cad/coordInput";
import { segFromBulge, type Seg, type Vec } from "../src/cad/geom";
import { buildSnapScene, findSnap, type SnapKind } from "../src/cad/snap";

const line = (a: Vec, b: Vec): Seg => ({ kind: "line", a, b });
const ALL: Partial<Record<SnapKind, boolean>> = { endpoint: true, midpoint: true, center: true, intersection: true, perpendicular: true, tangent: true };
// 1 m = 100 px、半径 10 px
const snap = (scene: ReturnType<typeof buildSnapScene>, p: Vec, ref: Vec | null = null, enabled = ALL, exclude: string | null = null) =>
  findSnap(scene, p, 0.1, 100, enabled, ref, exclude);

describe("object snaps", () => {
  const scene = buildSnapScene([
    { owner: "a", segs: [line([0, 0], [4, 0]), line([2, -2], [2, 2])] },
    { owner: "b", segs: [segFromBulge([6, 0], [8, 0], 1)], circles: [{ center: [10, 5], r: 1 }] },
  ]);

  it("endpoints, midpoints, centres and intersections", () => {
    expect(snap(scene, [0.03, 0.04])).toEqual({ point: [0, 0], kind: "endpoint" });
    // 円弧 (6,0)→(8,0) (反時計回りの半円は下を通る) の中点 (7, -1)
    expect(snap(scene, [7.03, -0.98])).toMatchObject({ kind: "midpoint" });
    expect(snap(scene, [1.05, 0.02])).toBeNull();
    expect(snap(scene, [10.05, 5])).toEqual({ point: [10, 5], kind: "center" });
    // 円弧 (6,0)→(8,0) の中心 (7, 0)
    expect(snap(scene, [7, 0.05])).toMatchObject({ kind: "center" });
    // (2, 0) は横線の中点でもあり交点でもある: 交点を優先
    expect(snap(scene, [2.04, 0.03])).toMatchObject({ point: [2, 0], kind: "intersection" });
    expect(snap(scene, [3, 1])).toBeNull();
    // 種類を切れば合わない
    expect(snap(scene, [0.03, 0.04], null, { midpoint: true })).toBeNull();
  });

  it("perpendicular feet and tangent points from the reference point", () => {
    // (3, 1) から横線への垂線の足 (3, 0)
    const f = snap(scene, [3.02, 0.05], [3, 1]);
    expect(f?.kind).toBe("perpendicular");
    expect(f?.point[0]).toBeCloseTo(3, 12);
    expect(f?.point[1]).toBeCloseTo(0, 12);
    // (10, 8) から円 (中心 (10, 5)、半径 1) への接点
    const t = snap(scene, [10.95, 5.33], [10, 8], { tangent: true });
    expect(t?.kind).toBe("tangent");
    const [x, y] = t!.point;
    // 接点では半径と接線が直交する
    expect((x - 10) * (10 - x) + (y - 5) * (8 - y)).toBeCloseTo(0, 9);
  });

  it("skips the excluded owner (the shape being dragged)", () => {
    expect(snap(scene, [0.03, 0.04], null, ALL, "a")).toBeNull();
    expect(snap(scene, [6.02, 0.03], null, ALL, "a")).toMatchObject({ kind: "endpoint" });
  });
});

describe("coordinate input", () => {
  const units = { lengthUnit: "mm" as const, axisymmetric: false };
  const ctx = { last: [0.01, 0.02] as Vec, direction: [1, 1] as Vec, units };
  const pt = (s: string, c = ctx) => {
    const r = parseCoordInput(s, c);
    if (!r.ok) throw new Error(r.error);
    return r.point;
  };
  it("reads absolute, relative and polar coordinates in display units", () => {
    expect(pt("10, 5")).toEqual([0.01, 0.005]);
    const r = pt("@5, -2");
    expect(r[0]).toBeCloseTo(0.015, 12);
    expect(r[1]).toBeCloseTo(0.018, 12);
    const p = pt("@10<90");
    expect(p[0]).toBeCloseTo(0.01, 12);
    expect(p[1]).toBeCloseTo(0.03, 12);
    const o = pt("20<30°");
    expect(o[0]).toBeCloseTo(0.02 * Math.cos(Math.PI / 6), 12);
    // 単位と式
    expect(pt("1 cm, 2*3")).toEqual([0.01, 0.006]);
    const e = pt("@sqrt(2)*10<45");
    expect(e[0]).toBeCloseTo(0.02, 12);
    expect(e[1]).toBeCloseTo(0.03, 12);
  });

  it("uses the cursor direction for a bare length", () => {
    const d = pt("10");
    expect(d[0]).toBeCloseTo(0.01 + 0.01 * Math.SQRT1_2, 12);
    expect(d[1]).toBeCloseTo(0.02 + 0.01 * Math.SQRT1_2, 12);
    expect(parseCoordInput("10", { ...ctx, last: null })).toEqual({ ok: false, error: "needLast" });
    expect(parseCoordInput("10", { ...ctx, direction: [0, 0] })).toEqual({ ok: false, error: "needDirection" });
    expect(parseCoordInput("@3, 4", { ...ctx, last: null })).toEqual({ ok: false, error: "needLast" });
    expect(parseCoordInput("abc, 1", ctx).ok).toBe(false);
    expect(parseCoordInput("5 kg, 1", ctx)).toEqual({ ok: false, error: "unit" });
    expect(parseCoordInput("", ctx)).toEqual({ ok: false, error: "empty" });
  });
});
