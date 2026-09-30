import { produce } from "immer";
import { describe, expect, it } from "vitest";
import { drawingBounds } from "../src/graphics/editTargets";
import { addLayer, itemLayer, layersOf, setActiveLayer } from "../src/model/layers";
import { applyDxfImport, scaleImport, type DxfApplied, type DxfImport } from "../src/model/dxfImport";
import { newProject, type Project } from "../src/model/project";
import { sketchOf } from "../src/model/sketch";

const result = (): DxfImport => ({
  unit: "mm",
  assumed: true,
  entities: [
    { kind: "polyline", points: [[0.005, 0.005], [0.035, 0.005], [0.035, 0.02], [0.005, 0.02]], bulges: [0, 0, 0, 0], closed: true, layer: "ANODE" },
    { kind: "circle", center: [0.07, 0.025], r: 0.005, layer: "CATHODE" },
    { kind: "line", a: [0, 0.045], b: [0.1, 0.045], layer: "0" },
    { kind: "arc", a: [0.01, 0.03], b: [0.02, 0.03], bulge: 0.5, layer: "0" },
    // 8 の字 (自分と交わる): 領域にはしない
    { kind: "polyline", points: [[0.05, 0.005], [0.06, 0.015], [0.06, 0.005], [0.05, 0.015]], bulges: [0, 0, 0, 0], closed: true, layer: "ANODE" },
  ],
  layers: [
    { name: "0", color: null, visible: true },
    { name: "ANODE", color: "#ff0000", visible: true },
    { name: "CATHODE", color: "#0000ff", visible: false },
  ],
  counts: { polyline: 2, circle: 1, line: 1, arc: 1 },
  skipped: { TEXT: 1 },
});

function run(p: Project, opts: Parameters<typeof applyDxfImport>[2]): [Project, DxfApplied] {
  let out: DxfApplied = { sketch: 0, regions: 0, layers: 0 };
  const next = produce(p, (d) => void (out = applyDxfImport(d, result(), opts)));
  return [next, out];
}

describe("DXF import", () => {
  it("adds sketch items on the DXF layers", () => {
    const [p, out] = run(newProject(), { unit: "mm", layers: true, toRegions: false });
    expect(out).toEqual({ sketch: 5, regions: 0, layers: 2 });
    const layers = layersOf(p);
    expect(layers.map((l) => [l.name, l.color ?? null, l.visible !== false])).toEqual([
      ["0", null, true],
      ["ANODE", "#ff0000", true],
      ["CATHODE", "#0000ff", false],
    ]);
    const s = sketchOf(p);
    expect(s.map((e) => layersOf(p).find((l) => l.id === itemLayer(p, e))!.name)).toEqual(["ANODE", "CATHODE", "0", "0", "ANODE"]);
    expect(s[3]).toMatchObject({ kind: "arc", bulge: 0.5 });
    // 同じ名前のレイヤは作り直さない
    const [again] = [produce(p, (d) => void applyDxfImport(d, result(), { unit: "mm", layers: true, toRegions: false }))];
    expect(layersOf(again)).toHaveLength(3);
  });

  it("makes regions from closed shapes except self-intersecting ones", () => {
    const [p, out] = run(newProject(), { unit: "mm", layers: true, toRegions: true });
    expect(out).toEqual({ sketch: 3, regions: 2, layers: 2 });
    expect(p.geometry.regions.map((r) => [r.type, r.shape ? "circle" : r.polygon!.length])).toEqual([
      ["conductor", 4],
      ["conductor", "circle"],
    ]);
    expect(layersOf(p).find((l) => l.id === itemLayer(p, p.geometry.regions[1]))!.name).toBe("CATHODE");
    expect(sketchOf(p).map((e) => e.kind)).toEqual(["line", "arc", "polyline"]);
  });

  it("rescales when the drawing unit is changed and uses the current layer without DXF layers", () => {
    const scaled = scaleImport(result(), "in");
    expect(scaled.unit).toBe("in");
    const line = scaled.entities[2] as { a: [number, number]; b: [number, number] };
    expect(line.b[0]).toBeCloseTo(0.1 * 25.4, 12);
    expect((scaled.entities[1] as { r: number }).r).toBeCloseTo(0.005 * 25.4, 12);
    expect(scaleImport(result(), "mm")).toEqual(result());
    const base = produce(newProject(), (d) => void setActiveLayer(d, addLayer(d)));
    const [p, out] = run(base, { unit: "mm", layers: false, toRegions: false });
    expect(out.layers).toBe(0);
    expect(sketchOf(p).every((e) => itemLayer(p, e) === "L1")).toBe(true);
    // 既定のレイヤ 0 に入れた形は、今のレイヤが別でも 0 のまま
    const [q] = run(base, { unit: "mm", layers: true, toRegions: false });
    expect(itemLayer(q, sketchOf(q)[2])).toBe("0");
    expect(sketchOf(q)[2].layer).toBe("0");
  });

  it("fits the view to the domain and the visible shapes", () => {
    const p = produce(newProject(), (d) => {
      applyDxfImport(d, { ...result(), entities: [{ kind: "circle", center: [0.3, 0.1], r: 0.01, layer: "0" }] }, { unit: "mm", layers: true, toRegions: false });
    });
    expect(drawingBounds(p)).toEqual({ x0: 0, y0: 0, x1: 0.31, y1: 0.11 });
    expect(drawingBounds(newProject())).toEqual({ x0: 0, y0: 0, x1: 0.1, y1: 0.05 });
  });
});
