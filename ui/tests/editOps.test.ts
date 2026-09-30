import { produce } from "immer";
import { describe, expect, it } from "vitest";
import { pathArea } from "../src/cad/geom";
import { mirror, rotation, translation } from "../src/cad/path";
import {
  arrayItems,
  arrayTransforms,
  cornerEdit,
  cutterSegs,
  extendSketch,
  filletSketchLines,
  offsetItem,
  pickedCenter,
  transformItems,
  trimSketch,
} from "../src/model/editOps";
import { newProject, regionArea, type Point, type Project } from "../src/model/project";
import { addSketch, sketchOf } from "../src/model/sketch";

function base(): Project {
  return produce(newProject(), (d) => {
    d.geometry.regions.push({ id: "a", type: "conductor", voltage: 5, polygon: [[0.01, 0.01], [0.02, 0.01], [0.02, 0.02], [0.01, 0.02]] });
    d.geometry.regions.push({ id: "c", type: "dielectric", eps_r: 4, shape: { kind: "circle", center: [0.05, 0.02], radius: 0.005 } });
    d.mesh.local_sizes = [{ region: "a", size: 0.001 }];
    addSketch(d, { kind: "arc", a: [0.06, 0.01], b: [0.08, 0.01], bulge: 0.5 });
  });
}

describe("transforms", () => {
  it("moves, copies and mirrors regions and sketch items", () => {
    const picked = [
      { kind: "region" as const, id: "a" },
      { kind: "region" as const, id: "c" },
      { kind: "sketch" as const, id: "s1" },
    ];
    let res: ReturnType<typeof transformItems> = [];
    const moved = produce(base(), (d) => void (res = transformItems(d, picked, translation(0.001, 0), false)));
    expect(res).toEqual(picked);
    expect(moved.geometry.regions[0].polygon![0]).toEqual([0.011, 0.01]);
    expect(moved.geometry.regions[1].shape!.center).toEqual([0.051, 0.02]);
    expect(sketchOf(moved)[0]).toMatchObject({ a: [0.061, 0.01] });
    // コピー: 値と局所メッシュ幅も写す
    const copied = produce(base(), (d) => void (res = transformItems(d, picked.slice(0, 1), translation(0, 0.02), true)));
    expect(res).toEqual([{ kind: "region", id: "a_1" }]);
    expect(copied.geometry.regions[2]).toMatchObject({ id: "a_1", type: "conductor", voltage: 5 });
    expect(copied.mesh.local_sizes).toEqual([
      { region: "a", size: 0.001 },
      { region: "a_1", size: 0.001 },
    ]);
    // 鏡映は円弧の向きを逆に
    const m = produce(base(), (d) => void transformItems(d, [{ kind: "sketch", id: "s1" }], mirror([0.07, 0], [0.07, 1]), false));
    expect(sketchOf(m)[0]).toMatchObject({ a: [0.08, 0.01], b: [0.06, 0.01], bulge: -0.5 });
    // 回転 (90°、中心は選んだものの中心)
    const center = pickedCenter(base(), [{ kind: "region", id: "a" }])!;
    expect(center).toEqual([0.015, 0.015]);
    const r = produce(base(), (d) => void transformItems(d, [{ kind: "region", id: "a" }], rotation(Math.PI / 2, center), false));
    expect(regionArea(r.geometry.regions[0])).toBeCloseTo(1e-4, 12);
  });

  it("builds rectangular and polar arrays", () => {
    expect(arrayTransforms({ kind: "rect", nx: 3, ny: 2, dx: 1, dy: 1 })).toHaveLength(5);
    expect(arrayTransforms({ kind: "polar", n: 4, center: [0, 0], angle: 360 })).toHaveLength(3);
    let made: ReturnType<typeof arrayItems> = [];
    const p = produce(base(), (d) => void (made = arrayItems(d, [{ kind: "region", id: "c" }], { kind: "polar", n: 4, center: [0.05, 0.03], angle: 360 })));
    expect(made).toHaveLength(3);
    const centers = p.geometry.regions.filter((r) => r.shape).map((r) => r.shape!.center.map((v) => Math.round(v * 1e6) / 1e6));
    expect(centers).toEqual([
      [0.05, 0.02],
      [0.06, 0.03],
      [0.05, 0.04],
      [0.04, 0.03],
    ]);
    // 全体の角度が 360° 未満なら両端を含めて等分
    const t = arrayTransforms({ kind: "polar", n: 3, center: [0, 0], angle: 90 });
    expect(t).toHaveLength(2);
  });
});

describe("corner edits", () => {
  it("fillets a region corner and a domain corner (keeping boundary conditions)", () => {
    const p = produce(base(), (d) => {
      const r = cornerEdit(d, { kind: "region", id: "a" }, 1, 0.002, "fillet");
      expect(r.ok).toBe(true);
    });
    expect(p.geometry.regions[0].polygon).toHaveLength(5);
    expect(regionArea(p.geometry.regions[0])).toBeCloseTo(1e-4 - 4e-6 * (1 - Math.PI / 4), 12);
    // ドメインの右下の角 (頂点 1): 下の辺 (条件なし) と右の辺 (100 V) → 新しい辺は条件なし
    const q = produce(base(), (d) => void cornerEdit(d, { kind: "domain" }, 1, 0.005, "chamfer"));
    expect(q.geometry.domain.polygon).toHaveLength(5);
    expect(q.geometry.domain.edge_ids).toEqual(["e1", "e5", "e2", "e3", "e4"]);
    expect(q.geometry.boundaries.map((b) => b.edges)).toEqual([[4], [2]]);
    // 両側が同じ条件なら新しい辺も同じ条件
    const both = produce(base(), (d) => {
      d.geometry.boundaries = [{ edges: [0, 1], type: "dirichlet", voltage: 0 }];
      cornerEdit(d, { kind: "domain" }, 1, 0.005, "fillet");
    });
    expect(both.geometry.boundaries[0].edges).toEqual([0, 1, 2]);
    expect(produce(base(), (d) => void cornerEdit(d, { kind: "domain" }, 1, 0.2, "fillet")).geometry.domain.polygon).toHaveLength(4);
  });

  it("fillets two sketch lines", () => {
    const p0 = produce(newProject(), (d) => {
      addSketch(d, { kind: "line", a: [0, 0], b: [0.02, 0] });
      addSketch(d, { kind: "line", a: [0.01, -0.01], b: [0.01, 0.01] });
    });
    const p = produce(p0, (d) => {
      const r = filletSketchLines(d, "s1", [0.005, 0], "s2", [0.01, 0.005], 0.002);
      expect(r).toEqual({ ok: true, value: "s3" });
    });
    const [l1, l2, arc] = sketchOf(p);
    expect(l1).toMatchObject({ a: [0, 0], b: [0.008, 0] });
    expect(l2).toMatchObject({ a: [0.01, 0.002], b: [0.01, 0.01] });
    expect(arc.kind).toBe("arc");
  });
});

describe("offset, trim and extend", () => {
  it("offsets a region outwards into a new region and a sketch line to the clicked side", () => {
    let made: ReturnType<typeof offsetItem> | null = null;
    const p = produce(base(), (d) => void (made = offsetItem(d, { kind: "region", id: "a" }, 0.001, [0.03, 0.015])));
    expect(made).toEqual({ ok: true, value: { kind: "region", id: "a_1" } });
    expect(regionArea(p.geometry.regions[2])).toBeCloseTo(1.44e-4, 12);
    const inward = produce(base(), (d) => void offsetItem(d, { kind: "region", id: "a" }, 0.001, [0.015, 0.015]));
    expect(regionArea(inward.geometry.regions[2])).toBeCloseTo(0.64e-4, 12);
    const circ = produce(base(), (d) => void offsetItem(d, { kind: "region", id: "c" }, 0.001, [0.06, 0.02]));
    expect(circ.geometry.regions[2].shape!.radius).toBeCloseTo(0.006, 12);
    const q = produce(newProject(), (d) => {
      addSketch(d, { kind: "line", a: [0, 0], b: [0.02, 0] });
      offsetItem(d, { kind: "sketch", id: "s1" }, 0.003, [0.01, -0.01]);
    });
    expect(sketchOf(q)[1]).toMatchObject({ kind: "line", a: [0, -0.003], b: [0.02, -0.003] });
  });

  it("trims a sketch line between two others and extends a line to a boundary", () => {
    const p0 = produce(newProject(), (d) => {
      addSketch(d, { kind: "line", a: [0.01, 0.02], b: [0.05, 0.02] });
      addSketch(d, { kind: "line", a: [0.02, 0.01], b: [0.02, 0.03] });
      addSketch(d, { kind: "line", a: [0.04, 0.01], b: [0.04, 0.03] });
    });
    let ids: string[] = [];
    const p = produce(p0, (d) => {
      const r = trimSketch(d, "s1", [0.03, 0.02], cutterSegs(d as Project, "s1"));
      if (r.ok) ids = r.value;
    });
    expect(ids).toEqual(["s1", "s4"]);
    const byId = Object.fromEntries(sketchOf(p).map((e) => [e.id, e]));
    expect(byId.s1).toMatchObject({ kind: "line", a: [0.01, 0.02], b: [0.02, 0.02] });
    expect(byId.s4).toMatchObject({ kind: "line", a: [0.04, 0.02], b: [0.05, 0.02] });
    // 延長: s2 の上端を上の外周 (y = 0.05) まで
    const q = produce(p0, (d) => void extendSketch(d, "s2", [0.02, 0.029], cutterSegs(d as Project, "s2")));
    expect(sketchOf(q)[1]).toMatchObject({ a: [0.02, 0.01], b: [0.02, 0.05] });
  });

  it("trims a circle into an arc", () => {
    const p = produce(newProject(), (d) => {
      addSketch(d, { kind: "circle", center: [0.05, 0.025], r: 0.01 });
      addSketch(d, { kind: "line", a: [0.03, 0.025], b: [0.07, 0.025] });
      trimSketch(d, "s1", [0.05, 0.034] as Point, cutterSegs(d as Project, "s1"));
    });
    const arc = sketchOf(p)[0];
    expect(arc.kind).toBe("arc");
    if (arc.kind === "arc") expect(Math.abs(pathArea([arc.a, arc.b], [arc.bulge, 0]))).toBeCloseTo((Math.PI * 1e-4) / 2, 10);
  });
});
