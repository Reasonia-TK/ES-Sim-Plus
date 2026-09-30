import { produce } from "immer";
import { beforeEach, describe, expect, it } from "vitest";
import { pathArea } from "../src/cad/geom";
import { regionToDomain, sketchToDomain, sketchToRegion } from "../src/model/cadActions";
import { useDocument } from "../src/model/documentStore";
import { domainFromPath } from "../src/model/domainOps";
import { newProject, type Point, type Project } from "../src/model/project";
import { useSelection } from "../src/model/selection";
import {
  addSketch,
  deleteSketch,
  editBend,
  editMove,
  editPathOf,
  editRemove,
  editSplit,
  moveSketch,
  replaceSketch,
  sketchBounds,
  sketchClosedPath,
  sketchFromEditPath,
  sketchOf,
  sketchSegs,
} from "../src/model/sketch";
import { deleteItems, editObjectOf, findSketchAt, itemsInBox, moveItems } from "../src/graphics/editTargets";

describe("sketch entities", () => {
  it("adds, moves, replaces and deletes with stable ids", () => {
    let p = newProject();
    let a: string | null = null;
    let b: string | null = null;
    p = produce(p, (d) => {
      a = addSketch(d, { kind: "line", a: [0, 0], b: [0.01, 0] });
      b = addSketch(d, { kind: "circle", center: [0.05, 0.02], r: 0.005 });
    });
    expect([a, b]).toEqual(["s1", "s2"]);
    // 長さ 0 の線・半径 0 の円は足さない
    expect(produce(p, (d) => void addSketch(d, { kind: "line", a: [0, 0], b: [0, 0] })).cad).toEqual(p.cad);
    p = produce(p, (d) => moveSketch(d, ["s1"], 0.001, 0.002));
    expect(sketchOf(p)[0]).toMatchObject({ a: [0.001, 0.002], b: [0.011, 0.002] });
    p = produce(p, (d) => replaceSketch(d, "s2", { kind: "circle", center: [0.05, 0.02], r: 0.007 }));
    expect(sketchOf(p)[1]).toMatchObject({ id: "s2", r: 0.007 });
    p = produce(p, (d) => deleteSketch(d, ["s1"]));
    expect(sketchOf(p).map((e) => e.id)).toEqual(["s2"]);
    p = produce(p, (d) => void addSketch(d, { kind: "arc", a: [0, 0], b: [0.02, 0], bulge: 1 }));
    // 番号は使っている最大の次
    expect(sketchOf(p).map((e) => e.id)).toEqual(["s2", "s3"]);
  });

  it("splits into segments, bounds and closed paths", () => {
    expect(sketchSegs({ id: "c", kind: "circle", center: [0, 0], r: 1 })).toHaveLength(2);
    const open = { id: "p", kind: "polyline" as const, points: [[0, 0], [1, 0], [1, 1]] as Point[], closed: false };
    expect(sketchSegs(open)).toHaveLength(2);
    expect(sketchSegs({ ...open, closed: true })).toHaveLength(3);
    expect(sketchClosedPath(open)).toBeNull();
    expect(sketchClosedPath({ id: "c", kind: "circle", center: [0, 0], r: 1 })!.polygon).toHaveLength(2);
    const b = sketchBounds({ id: "a", kind: "arc", a: [1, 0], b: [-1, 0], bulge: 1 });
    expect(b.y1).toBeCloseTo(1, 12);
    expect(b.y0).toBeCloseTo(0, 12);
  });

  it("edits open and closed point lists (lines become arcs or polylines)", () => {
    const line = editPathOf({ id: "l", kind: "line", a: [0, 0], b: [2, 0] })!;
    expect(sketchFromEditPath(editBend(line, 0, 1))).toMatchObject({ kind: "arc", bulge: 1 });
    const split = editSplit(line, 0);
    expect(split.points).toEqual([[0, 0], [1, 0], [2, 0]]);
    expect(sketchFromEditPath(split)).toMatchObject({ kind: "polyline", closed: false, points: [[0, 0], [1, 0], [2, 0]] });
    // 開いた経路の端の点を消すとその辺ごと、途中の点は前後を 1 本に
    const poly = { points: [[0, 0], [1, 0], [2, 1], [3, 1]] as Point[], bulges: [0, 0.3, 0], closed: false };
    expect(editRemove(poly, 0)!.points).toEqual([[1, 0], [2, 1], [3, 1]]);
    expect(editRemove(poly, 3)!.bulges).toEqual([0, 0.3]);
    expect(editRemove(poly, 2)!.bulges).toEqual([0, 0]);
    expect(editRemove({ points: [[0, 0], [1, 0]], bulges: [0], closed: false }, 0)).toBeNull();
    expect(editMove(poly, 1, [5, 5]).points[1]).toEqual([5, 5]);
  });
});

describe("conversions", () => {
  it("turns closed sketch shapes into regions (circles stay circles)", () => {
    let p = produce(newProject(), (d) => {
      addSketch(d, { kind: "polyline", points: [[0.01, 0.01], [0.03, 0.01], [0.02, 0.03]], closed: true });
      addSketch(d, { kind: "circle", center: [0.07, 0.02], r: 0.004 });
      addSketch(d, { kind: "line", a: [0, 0], b: [0.01, 0.01] });
    });
    let r1: string | null = null;
    let r2: string | null = null;
    let r3: string | null = null;
    p = produce(p, (d) => {
      r1 = sketchToRegion(d, "s1");
      r2 = sketchToRegion(d, "s2");
      r3 = sketchToRegion(d, "s3");
    });
    expect([r1, r2, r3]).toEqual(["region1", "region2", null]);
    expect(p.geometry.regions[1].shape).toMatchObject({ center: [0.07, 0.02], radius: 0.004 });
    expect(sketchOf(p).map((e) => e.id)).toEqual(["s3"]);
  });

  it("makes a domain from a shape and keeps the conditions of overlapping edges", () => {
    // 新しい外周: 元の 100 × 50 mm の矩形の下半分 (左右の辺は元の辺の上、上の辺は新しい)
    const p0 = newProject();
    const half = { polygon: [[0, 0], [0.1, 0], [0.1, 0.025], [0, 0.025]] as Point[] };
    const edit = domainFromPath(p0, half);
    expect(edit.ids).toEqual(["e1", "e2", "e5", "e4"]);
    const p = produce(p0, (d) => {
      addSketch(d, { kind: "polyline", points: half.polygon, closed: true });
      sketchToDomain(d, "s1");
    });
    expect(p.geometry.domain.polygon).toEqual(half.polygon);
    // 左 0 V・右 100 V はそのまま (上の辺は新しい辺で条件なし)
    expect(p.geometry.boundaries.map((b) => [b.edges, b.voltage])).toEqual([
      [[3], 0],
      [[1], 100],
    ]);
    expect(sketchOf(p)).toHaveLength(0);
    // 広げたとき: 同じ直線の上で重なる辺 (左・右・下) は条件を引き継ぐ
    const tall = domainFromPath(p0, { polygon: [[0, 0], [0.1, 0], [0.1, 0.08], [0, 0.08]] });
    expect(tall.ids).toEqual(["e1", "e2", "e5", "e4"]);
    // 端で接するだけの続き (同じ直線の先) は引き継がない
    const beyond = domainFromPath(p0, { polygon: [[0.1, 0], [0.2, 0], [0.2, 0.05], [0.1, 0.05]] });
    expect(beyond.ids).toEqual(["e5", "e6", "e7", "e2"]);
    // 領域の形をドメインに (円: 元の辺と重ならないので境界条件は全て外れる)
    const q = produce(newProject(), (d) => {
      d.geometry.regions.push({ id: "c", type: "conductor", voltage: 0, shape: { kind: "circle", center: [0.05, 0.025], radius: 0.02 } });
      const r = regionToDomain(d, "c");
      expect(r?.boundariesRemoved).toBe(2);
    });
    expect(q.geometry.regions).toHaveLength(0);
    expect(q.geometry.domain.polygon).toHaveLength(2);
    expect(Math.abs(pathArea(q.geometry.domain.polygon, q.geometry.domain.bulges))).toBeCloseTo(Math.PI * 0.02 ** 2, 12);
  });
});

describe("canvas selection helpers", () => {
  beforeEach(() => {
    useDocument.getState().replace(newProject(), null);
    useSelection.setState({ activeNode: "domain", selectedRegion: null, selectedPlacement: null, picked: [] });
  });

  it("picks, toggles and follows the document", () => {
    const sel = useSelection.getState();
    sel.pick([{ kind: "region", id: "a" }]);
    expect(useSelection.getState()).toMatchObject({ selectedRegion: "a", activeNode: "region:a" });
    useSelection.getState().pick([{ kind: "sketch", id: "s1" }], "toggle");
    expect(useSelection.getState().picked).toHaveLength(2);
    expect(useSelection.getState().activeNode).toBe("sketch");
    useSelection.getState().pick([{ kind: "region", id: "a" }], "toggle");
    expect(useSelection.getState()).toMatchObject({ selectedRegion: null, picked: [{ kind: "sketch", id: "s1" }] });
    // 文書に無いものは外れる
    useDocument.getState().update("x", (d) => void addSketch(d, { kind: "line", a: [0, 0], b: [0.01, 0] }));
    useDocument.getState().update("y", (d) => deleteSketch(d, ["s1"]));
    expect(useSelection.getState().picked).toEqual([]);
    // ドメイン・辺を選ぶとキャンバスの選択は外れる
    useSelection.getState().pick([{ kind: "region", id: "a" }]);
    useSelection.getState().select("edge:e1");
    expect(useSelection.getState()).toMatchObject({ picked: [], selectedRegion: null });
  });

  it("edit objects, box selection, moving and deleting several items", () => {
    let p: Project = produce(newProject(), (d) => {
      d.geometry.regions.push({ id: "r", type: "dielectric", polygon: [[0.01, 0.01], [0.02, 0.01], [0.02, 0.02]] });
      addSketch(d, { kind: "line", a: [0.03, 0.01], b: [0.04, 0.02] });
      addSketch(d, { kind: "circle", center: [0.08, 0.03], r: 0.005 });
    });
    expect(editObjectOf(p, [{ kind: "region", id: "r" }], "region:r")?.path?.closed).toBe(true);
    expect(editObjectOf(p, [{ kind: "sketch", id: "s2" }], "sketch")?.circle).toMatchObject({ r: 0.005 });
    expect(editObjectOf(p, [], "edge:e2")?.target).toEqual({ kind: "domain" });
    expect(editObjectOf(p, [{ kind: "region", id: "r" }, { kind: "sketch", id: "s1" }], "sketch")).toBeNull();
    const boxed = itemsInBox(p, [0, 0], [0.05, 0.03]);
    expect(boxed).toEqual([
      { kind: "region", id: "r" },
      { kind: "sketch", id: "s1" },
    ]);
    expect(findSketchAt([0.035, 0.0151], sketchOf(p), 1e-3)?.id).toBe("s1");
    p = produce(p, (d) => moveItems(d, boxed, 0.001, 0));
    expect(p.geometry.regions[0].polygon![0]).toEqual([0.011, 0.01]);
    expect(sketchOf(p)[0]).toMatchObject({ a: [0.031, 0.01] });
    p = produce(p, (d) => deleteItems(d, boxed));
    expect(p.geometry.regions).toHaveLength(0);
    expect(sketchOf(p).map((e) => e.id)).toEqual(["s2"]);
  });
});
