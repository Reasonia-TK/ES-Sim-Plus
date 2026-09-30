import { produce } from "immer";
import { describe, expect, it } from "vitest";
import { mirror } from "../src/cad/path";
import { applyRegionBoolean, type BooleanOutcome } from "../src/model/booleanOps";
import { offsetItem, transformItems } from "../src/model/editOps";
import { newProject, regionArea, type Point, type Project, type Region } from "../src/model/project";
import { moveRegion, removeRegionHole, removeRegionVertex, setRegionHole, setRegionPath } from "../src/model/regionOps";
import { meshKey } from "../src/results/staticResults";

const rect = (x0: number, y0: number, x1: number, y1: number): Point[] => [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];

function base(): Project {
  return produce(newProject(), (d) => {
    d.geometry.regions.push({ id: "a", type: "conductor", voltage: 5, polygon: rect(0, 0, 0.03, 0.01) });
    d.geometry.regions.push({ id: "b", type: "dielectric", eps_r: 4, polygon: rect(0.01, -0.01, 0.02, 0.02) });
    d.geometry.regions.push({ id: "c", type: "dielectric", eps_r: 2, shape: { kind: "circle", center: [0.03, 0.005], radius: 0.005 } });
    d.mesh.local_sizes = [{ region: "a", size: 0.001 }];
  });
}

function run(p: Project, op: "union" | "difference" | "intersection", target: string, tools: string[], keep = false): [Project, BooleanOutcome] {
  let out: BooleanOutcome = { ok: false, error: "failed" };
  const next = produce(p, (d) => void (out = applyRegionBoolean(d, op, target, tools, keep)));
  return [next, out];
}

const byId = (p: Project, id: string) => p.geometry.regions.find((r) => r.id === id) as Region;

describe("region booleans", () => {
  it("splits the target, keeps its ID and values and copies them to the extra piece", () => {
    const [p, out] = run(base(), "difference", "a", ["b"]);
    expect(out).toEqual({ ok: true, ids: ["a", "a_1"] });
    expect(p.geometry.regions.map((r) => r.id)).toEqual(["a", "a_1", "c"]);
    const extra = byId(p, "a_1");
    expect(extra.type).toBe("conductor");
    expect(extra.voltage).toBe(5);
    expect(regionArea(byId(p, "a")) + regionArea(extra)).toBeCloseTo(0.0002, 12);
    expect(p.mesh.local_sizes).toEqual([
      { region: "a", size: 0.001 },
      { region: "a_1", size: 0.001 },
    ]);
  });

  it("keeps the tools when asked and cuts holes", () => {
    const withInner = produce(base(), (d) => void d.geometry.regions.push({ id: "in", type: "charge", rho: 1, polygon: rect(0.002, 0.002, 0.004, 0.004) }));
    const [p, out] = run(withInner, "difference", "a", ["in"], true);
    expect(out.ok).toBe(true);
    const a = byId(p, "a");
    expect(a.holes).toHaveLength(1);
    expect(byId(p, "in")).toBeDefined();
    expect(regionArea(a)).toBeCloseTo(0.0003 - 0.000004, 12);
    expect(meshKey(p)).not.toBe(meshKey(withInner));
    // 穴を埋める
    const filled = produce(p, (d) => removeRegionHole(d, "a", 0));
    expect(byId(filled, "a").holes).toBeUndefined();
  });

  it("turns a circle target into arcs and consumes the tool", () => {
    const [p, out] = run(base(), "union", "c", ["a"]);
    expect(out).toEqual({ ok: true, ids: ["c"] });
    const c = byId(p, "c");
    expect(c.shape).toBeUndefined();
    expect(c.bulges?.some((b) => b !== 0)).toBe(true);
    expect(c.eps_r).toBe(2);
    expect(p.geometry.regions.map((r) => r.id)).toEqual(["b", "c"]);
    expect(p.mesh.local_sizes).toEqual([]);
    expect(regionArea(c)).toBeCloseTo(0.0003 + (Math.PI * 0.005 ** 2) / 2, 10);
  });

  it("refuses empty results without changing the document", () => {
    const p = base();
    const far = produce(p, (d) => void d.geometry.regions.push({ id: "far", type: "conductor", voltage: 0, polygon: rect(0.1, 0.1, 0.11, 0.11) }));
    const [after, bad] = run(far, "intersection", "a", ["far"]);
    expect(bad).toEqual({ ok: false, error: "empty" });
    expect(after).toBe(far);
    expect(run(p, "union", "a", ["a"])[1]).toEqual({ ok: false, error: "noTools" });
  });
});

describe("regions with holes", () => {
  const holed = () =>
    produce(newProject(), (d) => {
      d.geometry.regions.push({ id: "h", type: "dielectric", eps_r: 3, polygon: rect(0, 0, 0.04, 0.04), holes: [{ polygon: rect(0.01, 0.01, 0.02, 0.02) }] });
    });

  it("moves, mirrors and copies the holes with the outline", () => {
    const moved = produce(holed(), (d) => moveRegion(d, "h", 0.001, 0));
    expect(byId(moved, "h").holes![0].polygon[0]).toEqual([0.011, 0.01]);
    let res: ReturnType<typeof transformItems> = [];
    const m = produce(holed(), (d) => void (res = transformItems(d, [{ kind: "region", id: "h" }], mirror([0.05, 0], [0.05, 1]), true)));
    expect(res).toEqual([{ kind: "region", id: "h_1" }]);
    const copy = byId(m, "h_1");
    expect(copy.holes![0].polygon.map(([x]) => x).sort()).toEqual([0.08, 0.08, 0.09, 0.09]);
    expect(regionArea(copy)).toBeCloseTo(0.0016 - 0.0001, 12);
  });

  it("refuses outlines and holes that no longer fit", () => {
    const p = holed();
    expect(produce(p, (d) => void setRegionPath(d, "h", { polygon: rect(0, 0, 0.015, 0.04) }))).toBe(p);
    expect(produce(p, (d) => void setRegionHole(d, "h", 0, { polygon: rect(0.01, 0.01, 0.05, 0.02) }))).toBe(p);
    const moved = produce(p, (d) => void setRegionHole(d, "h", 0, { polygon: rect(0.02, 0.02, 0.03, 0.03) }));
    expect(byId(moved, "h").holes![0].polygon[0]).toEqual([0.02, 0.02]);
    // 穴を外周の外に出す頂点の削除はしない
    const tri = produce(p, (d) => void removeRegionVertex(d, "h", 2));
    expect(tri).toBe(p);
  });

  it("offsets every ring: growing shrinks the holes", () => {
    let made: ReturnType<typeof offsetItem> | null = null;
    const p = produce(holed(), (d) => void (made = offsetItem(d, { kind: "region", id: "h" }, 0.002, [0.05, 0.02])));
    expect(made).toEqual({ ok: true, value: { kind: "region", id: "h_1" } });
    const g = byId(p, "h_1");
    expect(g.holes).toHaveLength(1);
    expect(regionArea(g)).toBeCloseTo(0.044 ** 2 - 0.006 ** 2, 12);
    // 太らせて消える穴は無くなる
    const big = produce(holed(), (d) => void offsetItem(d, { kind: "region", id: "h" }, 0.006, [0.05, 0.02]));
    expect(byId(big, "h_1").holes).toBeUndefined();
    // 穴の中をクリックしても太らせる
    const inHole = produce(holed(), (d) => void offsetItem(d, { kind: "region", id: "h" }, 0.002, [0.015, 0.015]));
    expect(byId(inHole, "h_1").holes![0].polygon).toHaveLength(4);
  });
});
