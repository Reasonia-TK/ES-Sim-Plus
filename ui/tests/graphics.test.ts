import { produce } from "immer";
import { describe, expect, it } from "vitest";
import { rememberBlock } from "../src/forms/blocks";
import { fitCamera, gridStep, panBy, snapValue, toScreen, toWorld, zoomAt } from "../src/graphics/camera";
import { parseColor } from "../src/graphics/color";
import { COLORMAP_KEYS, colormapRgba, LUT_SIZE, sampleColormap } from "../src/graphics/colormaps";
import { fieldCsv, profileCsv, stamp } from "../src/graphics/exporting";
import { colorbarTicks, colorPosition, fieldStats, formatColorbarValue, resolveRange } from "../src/graphics/fieldScale";
import {
  dedupeTail,
  findMidpointHandle,
  findRegionAt,
  findVertexHandle,
  hitRadiusHandle,
  insertMidpoint,
  pointInPolygon,
  rectFromCorners,
} from "../src/graphics/hitTest";
import { isoLevels, isolineSegments } from "../src/graphics/isolines";
import { MeshIndex, type TriMesh } from "../src/graphics/meshIndex";
import { tickLabel } from "../src/graphics/overlay";
import { amrBoxesOf, emitterOf, placementsOf } from "../src/graphics/projectOverlays";
import { pathsToSegments, sampleField, type ScalarField } from "../src/graphics/scene";
import { staticScene } from "../src/graphics/staticScene";
import type { OverlayKey } from "../src/graphics/viewerStore";
import {
  MAX_COLLECTORS,
  nextLabel,
  placeCollector,
  placeEdgeMeshSize,
  placeEedfRegion,
  placeEmitter,
  placeGasBoundary,
  placeSheathLine,
} from "../src/model/placements";
import { newProject, tidy, type Point, type Project, type Region } from "../src/model/project";
import {
  addCircleRegionAt,
  addPolygonRegion,
  moveRegion,
  removeRegionVertex,
  setCircleRadius,
  setRegionPolygon,
} from "../src/model/regionOps";
import { meshKey, solveKey, toViewMesh, type MeshEntry, type SolveEntry } from "../src/results/staticResults";

/** nx × ny の格子を三角形に分けたメッシュ ([0, w] × [0, h]) */
function gridMesh(nx: number, ny: number, w = 1, h = 1): TriMesh {
  const nodes: [number, number][] = [];
  for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) nodes.push([(i * w) / nx, (j * h) / ny]);
  const triangles: [number, number, number][] = [];
  const id = (i: number, j: number) => j * (nx + 1) + i;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      triangles.push([id(i, j), id(i + 1, j), id(i + 1, j + 1)]);
      triangles.push([id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
    }
  }
  return { nodes, triangles };
}

describe("colormaps", () => {
  it("interpolates v1's anchors into a 256-entry LUT and clamps", () => {
    expect(sampleColormap("viridis", 0)).toEqual([68, 1, 84]);
    expect(sampleColormap("viridis", 1)).toEqual([253, 231, 37]);
    expect(sampleColormap("gray", 0.5).map(Math.round)).toEqual([128, 128, 128]);
    expect(sampleColormap("jet", -1)).toEqual(sampleColormap("jet", 0));
    expect(sampleColormap("jet", 2)).toEqual(sampleColormap("jet", 1));
    expect(sampleColormap("plasma", Number.NaN)).toEqual(sampleColormap("plasma", 0));
    for (const k of COLORMAP_KEYS) {
      const rgba = colormapRgba(k);
      expect(rgba).toHaveLength(4 * LUT_SIZE);
      expect(rgba[4 * (LUT_SIZE - 1) + 3]).toBe(255);
    }
  });
});

describe("camera", () => {
  const b = { x0: 0, y0: 0, x1: 0.1, y1: 0.05 };
  it("fits the bounds at 80% and round-trips screen ↔ world (y up)", () => {
    const c = fitCamera(b, 1000, 500);
    expect(c.scale).toBeCloseTo(8000);
    expect(toScreen(c, [0.05, 0.025])).toEqual([500, 250]);
    const [x, y] = toWorld(c, ...toScreen(c, [0.012, 0.034]));
    expect(x).toBeCloseTo(0.012, 12);
    expect(y).toBeCloseTo(0.034, 12);
    expect(toScreen(c, [0, 0.05])[1]).toBeLessThan(toScreen(c, [0, 0])[1]);
  });
  it("zooms about the cursor and pans in pixels", () => {
    const c = fitCamera(b, 1000, 500);
    const p = toWorld(c, 300, 120);
    const z = zoomAt(c, 300, 120, 1.15 ** 3);
    const q = toWorld(z, 300, 120);
    expect(q[0]).toBeCloseTo(p[0], 12);
    expect(q[1]).toBeCloseTo(p[1], 12);
    expect(zoomAt(c, 0, 0, 1e9, { min: 1, max: 2 * c.scale }).scale).toBe(2 * c.scale);
    expect(toScreen(panBy(c, 10, -5), [0, 0])).toEqual([toScreen(c, [0, 0])[0] + 10, toScreen(c, [0, 0])[1] - 5]);
  });
  it("picks a power-of-ten grid of at least 20 px (no 1 mm floor) and tidy snapping", () => {
    for (const scale of [1, 37, 8000, 3e5, 7e7]) {
      const s = gridStep({ scale, ox: 0, oy: 0 });
      expect(s * scale).toBeGreaterThanOrEqual(20 - 1e-9);
      expect(s * scale).toBeLessThan(200 + 1e-9);
      expect(Math.log10(s)).toBeCloseTo(Math.round(Math.log10(s)), 9);
    }
    expect(gridStep({ scale: 1e8, ox: 0, oy: 0 })).toBeCloseTo(1e-6, 15);
    expect(snapValue(0.30000000000000004, 0.1)).toBe(0.3);
    expect(snapValue(0.034999999999999996, 0.001)).toBe(0.035);
    expect(tidy(0.1 + 0.2)).toBe(0.3);
  });
});

describe("mesh index (probe / hover)", () => {
  it("locates points and interpolates a linear field exactly, like brute force", () => {
    const mesh = gridMesh(17, 9, 0.2, 0.1);
    const idx = new MeshIndex(mesh);
    const f = (x: number, y: number) => 3 * x - 7 * y + 1;
    const values = mesh.nodes.map(([x, y]) => f(x, y));
    let seed = 1;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    for (let k = 0; k < 500; k++) {
      const x = rnd() * 0.2;
      const y = rnd() * 0.1;
      const v = idx.interpolate(values, x, y);
      expect(v).not.toBeNull();
      expect(v!).toBeCloseTo(f(x, y), 10);
    }
    expect(idx.locate(-0.01, 0.05)).toBeNull();
    expect(idx.locate(0.1, 0.2)).toBeNull();
    // 頂点・辺の上も見つかる
    expect(idx.locate(0, 0)).not.toBeNull();
    expect(idx.locate(0.2, 0.1)).not.toBeNull();
  });
  it("samples element fields per triangle and treats holes as outside", () => {
    const mesh = gridMesh(2, 1);
    // 右半分の三角形を抜く (穴)
    const holed: TriMesh = { nodes: mesh.nodes, triangles: mesh.triangles.slice(0, 2) };
    const f: ScalarField = { mesh: holed, values: [5, 6], location: "element", label: "|E|", unit: "V/m" };
    expect(sampleField(f, 0.4, 0.1)).toBe(5);
    expect(sampleField(f, 0.1, 0.4)).toBe(6);
    expect(sampleField(f, 0.8, 0.5)).toBeNull();
  });
});

describe("isolines", () => {
  it("uses 15 levels between (excluding) the extremes and cuts a linear field on the level lines", () => {
    expect(isoLevels(0, 16, 15)).toEqual(Array.from({ length: 15 }, (_, i) => i + 1));
    expect(isoLevels(1, 1)).toEqual([]);
    const mesh = gridMesh(10, 10);
    const values = mesh.nodes.map(([x]) => 100 * x);
    const segs = isolineSegments(mesh, values, [25, 50]);
    expect(segs.length % 4).toBe(0);
    expect(segs.length).toBeGreaterThan(0);
    for (let i = 0; i < segs.length; i += 2) expect([25, 50].some((l) => Math.abs(100 * segs[i] - l) < 1e-9)).toBe(true);
    // 欠けた値の三角形は飛ばす
    const nan = values.map((v, i) => (i === 0 ? Number.NaN : v));
    expect(() => isolineSegments(mesh, nan, [25])).not.toThrow();
  });
});

describe("field range", () => {
  it("ignores NaN, honours a manual range and log-scales with the smallest positive value", () => {
    const s = fieldStats([Number.NaN, -2, 0, 1e-3, 10, Number.POSITIVE_INFINITY]);
    expect(s).toEqual({ min: -2, max: 10, minPositive: 1e-3 });
    const lin = resolveRange(s, false, { min: null, max: 5 });
    expect([lin.lo, lin.hi, lin.labelMin, lin.labelMax, lin.log]).toEqual([-2, 5, -2, 5, false]);
    const log = resolveRange(s, true);
    expect(log.log).toBe(true);
    expect(log.lo).toBeCloseTo(-3);
    expect(log.hi).toBeCloseTo(1);
    expect(colorPosition(1e-1, log)).toBeCloseTo(0.5);
    expect(colorPosition(-5, log)).toBeCloseTo(0);
    // 正の値が無ければ線形に戻す
    expect(resolveRange(fieldStats([-1, 0]), true).log).toBe(false);
    expect(colorbarTicks(log, 5).map((v) => Number(v.toPrecision(6)))).toEqual([1e-3, 1e-2, 1e-1, 1, 10]);
    expect(formatColorbarValue(12345)).toBe("1.2e+4");
    expect(formatColorbarValue(0.5)).toBe("0.5");
    expect(formatColorbarValue(0)).toBe("0");
  });
});

describe("hit testing (v1 rules)", () => {
  const big: Region = { id: "big", type: "dielectric", polygon: [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]] };
  const small: Region = { id: "small", type: "conductor", polygon: [[0.04, 0.04], [0.06, 0.04], [0.06, 0.06], [0.04, 0.06]] };
  const circ: Region = { id: "c", type: "conductor", shape: { kind: "circle", center: [0.2, 0.05], radius: 0.01 } };
  it("prefers the smallest region and accepts clicks near the outline", () => {
    expect(findRegionAt([0.05, 0.05], [big, small, circ], 1e-4)?.id).toBe("small");
    expect(findRegionAt([0.02, 0.02], [big, small, circ], 1e-4)?.id).toBe("big");
    expect(findRegionAt([0.1003, 0.05], [big], 5e-4)?.id).toBe("big");
    expect(findRegionAt([0.1003, 0.05], [big], 1e-4)).toBeNull();
    expect(findRegionAt([0.2105, 0.05], [circ], 1e-3)?.id).toBe("c");
    expect(pointInPolygon([0.05, 0.05], small.polygon!)).toBe(true);
  });
  it("finds vertex, midpoint and radius handles within 8 px", () => {
    const cam = fitCamera({ x0: 0, y0: 0, x1: 0.1, y1: 0.1 }, 500, 500);
    const [vx, vy] = toScreen(cam, [0.06, 0.04]);
    expect(findVertexHandle(small.polygon!, cam, vx + 5, vy)).toBe(1);
    expect(findVertexHandle(small.polygon!, cam, vx + 12, vy)).toBeNull();
    const [mx, my] = toScreen(cam, [0.05, 0.04]);
    expect(findMidpointHandle(small.polygon!, cam, mx, my + 3)).toBe(0);
    const [rx, ry] = toScreen(cam, [0.21, 0.05]);
    expect(hitRadiusHandle(circ.shape!, cam, rx - 4, ry)).toBe(true);
    expect(insertMidpoint(small.polygon!, 0)).toEqual([[0.04, 0.04], [0.05, 0.04], [0.06, 0.04], [0.06, 0.06], [0.04, 0.06]]);
    expect(rectFromCorners([0.3, 0.1], [0.1, 0.2])).toEqual([[0.1, 0.1], [0.3, 0.1], [0.3, 0.2], [0.1, 0.2]]);
    expect(dedupeTail([[0, 0], [1, 1], [1, 1]])).toEqual([[0, 0], [1, 1]]);
  });
});

describe("canvas edits", () => {
  it("draws, moves and edits regions with tidy coordinates", () => {
    let id: string | null = null;
    let p = produce(newProject(), (d) => void (id = addPolygonRegion(d, [[0.01, 0.01], [0.02, 0.01], [0.015, 0.02]])));
    expect(id).toBe("region1");
    expect(produce(p, (d) => void addPolygonRegion(d, [[0, 0], [1, 1]])).geometry.regions).toHaveLength(1);
    p = produce(p, (d) => moveRegion(d, "region1", 0.1 + 0.2 - 0.3 + 0.005, 0.005));
    expect(p.geometry.regions[0].polygon![0]).toEqual([0.015, 0.015]);
    p = produce(p, (d) => setRegionPolygon(d, "region1", [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]]));
    p = produce(p, (d) => void removeRegionVertex(d, "region1", 3));
    expect(p.geometry.regions[0].polygon).toHaveLength(3);
    // 3 点のときは消さない
    p = produce(p, (d) => void removeRegionVertex(d, "region1", 0));
    expect(p.geometry.regions[0].polygon).toHaveLength(3);
    p = produce(p, (d) => void addCircleRegionAt(d, [0.05, 0.02], 0.004));
    expect(p.geometry.regions[1]).toMatchObject({ id: "region2", type: "conductor", voltage: 0, shape: { center: [0.05, 0.02], radius: 0.004 } });
    p = produce(p, (d) => setCircleRadius(d, "region2", 0.006));
    expect(p.geometry.regions[1].shape!.radius).toBe(0.006);
    expect(produce(p, (d) => setCircleRadius(d, "region2", 0)).geometry.regions[1].shape!.radius).toBe(0.006);
    expect(produce(p, (d) => void addCircleRegionAt(d, [0, 0], 0)).geometry.regions).toHaveLength(2);
  });

  it("places objects, enabling the study (remembered value first) and keeping v1's limits", () => {
    let p: Project = newProject();
    let r = { index: null as number | null, enabledStudy: false };
    p = produce(p, (d) => void (r = placeCollector(d, [0, 0.01], [0.1, 0.01])));
    expect(r).toEqual({ index: 0, enabledStudy: true });
    expect((p.pic as { collectors: { label: string; tol: null }[] }).collectors[0]).toMatchObject({ label: "C1", tol: null });
    for (let i = 1; i < MAX_COLLECTORS; i++) p = produce(p, (d) => void placeCollector(d, [0, 0], [0.01, 0]));
    p = produce(p, (d) => void (r = placeCollector(d, [0, 0], [0.01, 0])));
    expect(r.index).toBeNull();
    expect((p.pic as { collectors: unknown[] }).collectors).toHaveLength(MAX_COLLECTORS);
    p = produce(p, (d) => void placeEedfRegion(d, [0, 0], [0.01, 0.01]));
    p = produce(p, (d) => void placeSheathLine(d, [0, 0.02], [0.02, 0.02]));
    const pic = p.pic as { eedf_regions: { label: string; bins: number }[]; sheath_lines: { label: string }[] };
    expect(pic.eedf_regions[0]).toMatchObject({ label: "E1", bins: 100, e_max_ev: null });
    expect(pic.sheath_lines[0]).toMatchObject({ label: "S1", p1: [0, 0.02], p2: [0.02, 0.02] });
    // DSMC は覚えている値で有効にする
    rememberBlock(["dsmc"], { gas: { name: "N2" }, boundaries: [], n_particles: 123 });
    p = produce(p, (d) => void placeGasBoundary(d, [0.1, 0], [0.1, 0.05]));
    const dsmc = p.dsmc as { gas: { name: string }; n_particles: number; boundaries: { type: string; p1: Point }[] };
    expect(dsmc.gas.name).toBe("N2");
    expect(dsmc.n_particles).toBe(123);
    expect(dsmc.boundaries[0]).toMatchObject({ type: "wall", p1: [0.1, 0] });
    p = produce(p, (d) => void placeEdgeMeshSize(d, [0, 0], [0.1, 0]));
    expect(p.mesh.local_edge_sizes).toEqual([{ p1: [0, 0], p2: [0.1, 0], size: 0.001 }]);
    p = produce(p, (d) => void placeEmitter(d, [0.02, 0.01], [0.02, 0.04]));
    const em = (p.particles as { emitter: { p1: Point; p2: Point; n: number } }).emitter;
    expect(em).toMatchObject({ p1: [0.02, 0.01], p2: [0.02, 0.04], n: 50 });
    expect(nextLabel("C", [{ label: "C1" }, { label: "C5" }, { label: "x" }])).toBe("C6");
  });
});

describe("what the viewer shows", () => {
  const mesh = { nodes: [[0, 0], [1, 0], [1, 1], [0, 1]] as Point[], triangles: [[0, 1, 2], [0, 2, 3]] as [number, number, number][], region_of_triangle: [-1, 0] };
  const labels = { meshTitle: (n: number, e: number) => `mesh ${n}/${e}`, potential: "V", field: "|E|" };
  it("shows the newest of mesh preview and solution, marks stale results", () => {
    const p = newProject();
    const meshEntry: MeshEntry = { result: mesh, view: toViewMesh(mesh), key: meshKey(p), elapsedS: 0.1 };
    const result = { mesh, v: [0, 1, 1, 0], e_field: [[-1, 0], [-1, 0]] as [number, number][], v_min: 0, v_max: 1, e_abs_max: 1, energy: 1, charges: [], capacitance: null };
    const solveEntry: SolveEntry = { result, project: p, view: toViewMesh(mesh), eAbs: new Float64Array([1, 1]), key: solveKey(p), elapsedS: 0.2 };
    const onlyMesh = staticScene({ mesh: meshEntry, solve: null, latest: "mesh" }, "v", p, labels);
    expect(onlyMesh).toMatchObject({ title: "mesh 4/2", stale: false, fillRegions: true, field: null });
    const solved = staticScene({ mesh: null, solve: solveEntry, latest: "solve" }, "e_abs", p, labels);
    expect(solved.field?.location).toBe("element");
    expect(solved.iso?.location).toBe("node");
    expect(solved.vectors?.values).toHaveLength(2);
    // メッシュを作り直したらメッシュを出す
    expect(staticScene({ mesh: meshEntry, solve: solveEntry, latest: "mesh" }, "v", p, labels).field).toBeNull();
    const edited = produce(p, (d) => void (d.geometry.boundaries[0].voltage = 5));
    expect(staticScene({ mesh: null, solve: solveEntry, latest: "solve" }, "v", edited, labels).stale).toBe(true);
    // 電圧はメッシュに効かない
    expect(meshKey(edited)).toBe(meshKey(p));
    expect(solveKey(edited)).not.toBe(solveKey(p));
  });

  it("lists placed objects, the emitter and AMR boxes from the document", () => {
    const show = Object.fromEntries(["collectors", "gasBoundaries", "eedf", "edgeSizes", "sheathLines"].map((k) => [k, true])) as Record<OverlayKey, boolean>;
    let p = produce(newProject(), (d) => {
      placeCollector(d, [0, 0.01], [0.1, 0.01]);
      placeGasBoundary(d, [0.1, 0], [0.1, 0.05]);
      (d.dsmc as { boundaries: unknown[] }).boundaries.push({ edges: [1], type: "inlet", p1: null, p2: null });
      placeEmitter(d, [0.02, 0.01], [0.02, 0.04]);
    });
    const pl = placementsOf(p, show);
    expect(pl.map((x) => `${x.kind}:${x.label}`)).toEqual(["collector:C1", "gasbc:G1"]);
    expect(placementsOf(p, { ...show, collectors: false }).map((x) => x.kind)).toEqual(["gasbc"]);
    expect(emitterOf(p)).toMatchObject({ kind: "line", p1: [0.02, 0.01] });
    p = produce(p, (d) => void ((d.particles as { fn: unknown }).fn = { edges: [0] }));
    expect(emitterOf(p)).toBeNull();
    const amr = produce(p, (d) => {
      d.mesh.mode = "cartesian";
      d.mesh.amr = { max_level: 1, regions: [{ p1: [0, 0], p2: [0.01, 0.01], level: 1 }] };
    });
    expect(amrBoxesOf(amr)).toEqual([[[0, 0], [0.01, 0.01]]]);
    expect(amrBoxesOf(produce(amr, (d) => void (d.mesh.mode = "unstructured")))).toEqual([]);
  });
});

describe("export and small helpers", () => {
  it("writes field and profile CSV", () => {
    const m = gridMesh(1, 1);
    const node: ScalarField = { mesh: m, values: [0, 1, 2, Number.NaN], location: "node", label: "V", unit: "V" };
    expect(fieldCsv(node, ["x", "y"]).split("\n").slice(0, 2)).toEqual(["x [m],y [m],V [V]", "0,0,0"]);
    expect(fieldCsv(node, ["x", "y"]).split("\n")[4]).toBe("1,1,");
    const el: ScalarField = { mesh: m, values: [3, 4], location: "element", label: "|E|", unit: "V/m" };
    const lines = fieldCsv(el, ["z", "r"]).trim().split("\n");
    expect(lines[0]).toBe("z [m],r [m],|E| [V/m]");
    expect(lines).toHaveLength(3);
    expect(profileCsv({ s: [0, 1], v: [1, null], e_abs: [null, 2] })).toBe("s,v,e_abs\n0,1,\n1,,2\n");
    expect(stamp(new Date(2026, 8, 30, 7, 5, 9))).toBe("20260930-070509");
  });
  it("formats ruler ticks for the step and parses CSS colours", () => {
    expect(tickLabel(0.012, 0.01, "mm")).toBe("12");
    expect(tickLabel(0.0123, 0.0001, "mm")).toBe("12.3");
    expect(tickLabel(-0.00000001, 0.0001, "mm")).toBe("0.0");
    expect(tickLabel(2.5e-6, 1e-6, "um")).toBe("3");
    expect(parseColor("#ff8000")).toEqual([1, 128 / 255, 0, 1]);
    expect(parseColor("#fff", 0.5)).toEqual([1, 1, 1, 0.5]);
    expect(parseColor("rgba(0, 51, 255, 0.25)")).toEqual([0, 0.2, 1, 0.25]);
    expect(pathsToSegments([[[0, 0], [1, 0], [1, 1]], [[5, 5]]])).toEqual(new Float64Array([0, 0, 1, 0, 1, 0, 1, 1]));
  });
});
