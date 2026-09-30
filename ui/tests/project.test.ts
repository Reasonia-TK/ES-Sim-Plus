import { describe, expect, it } from "vitest";
import { EXAMPLES } from "../src/io/examples";
import {
  axisEdge,
  boundaryOfEdge,
  isRectDomain,
  newProject,
  normalizeProject,
  ProjectFormatError,
  serializeProject,
  uniqueRegionId,
  type Project,
} from "../src/model/project";

describe("normalizeProject", () => {
  it("rejects files that are not projects", () => {
    expect(() => normalizeProject(null)).toThrow(ProjectFormatError);
    expect(() => normalizeProject([])).toThrow(ProjectFormatError);
    expect(() => normalizeProject({ mesh: { size: 1 } })).toThrow(/geometry/);
    expect(() => normalizeProject({ geometry: { domain: { polygon: [[0, 0]] } } })).toThrow(/polygon/);
  });

  it("fills defaults, strips results and migrates the old single collector", () => {
    const raw = {
      geometry: { domain: { polygon: [[0, 0], [0.02, 0], [0.02, 0.01], [0, 0.01]] } },
      pic: { collector: { p1: [0, 0], p2: [1, 1] } },
      results: { version: 1 },
    };
    const { project, results } = normalizeProject(raw);
    expect(results).toEqual({ version: 1 });
    expect("results" in project).toBe(false);
    expect(project.geometry.regions).toEqual([]);
    expect(project.geometry.boundaries).toEqual([]);
    expect(project.mesh.size).toBeCloseTo(0.02 / 25);
    expect(project.pic).toEqual({ collectors: [{ label: "C1", p1: [0, 0], p2: [1, 1] }] });
    // 元のオブジェクトは変えない
    expect((raw.pic as Record<string, unknown>).collector).toBeDefined();
  });

  it("opens every bundled example", () => {
    expect(EXAMPLES.map((e) => e.key)).toEqual(["parallel_plates", "coaxial", "ccp_demo", "egun_rz", "fn_diode"]);
    for (const ex of EXAMPLES) {
      const { project } = normalizeProject(ex.data);
      expect(project.geometry.domain.polygon.length).toBeGreaterThanOrEqual(3);
      expect(JSON.parse(serializeProject(project))).toEqual(project);
    }
  });
});

describe("geometry helpers", () => {
  it("recognises rectangular domains and the symmetry axis edge", () => {
    const p: Project = newProject();
    expect(isRectDomain(p)).toBe(true);
    expect(axisEdge(p)).toBeNull();
    expect(axisEdge({ ...p, coord: "rz" })).toBe(0);
    expect(axisEdge({ ...p, coord: "rz_x0" })).toBe(3);
    const coax = normalizeProject(EXAMPLES.find((e) => e.key === "coaxial")!.data).project;
    expect(isRectDomain(coax)).toBe(false);
    expect(axisEdge({ ...coax, coord: "rz" })).toBeNull();
  });

  it("finds the boundary condition of an edge and makes unique region ids", () => {
    const p = newProject();
    expect(boundaryOfEdge(p, 1)?.voltage).toBe(100);
    expect(boundaryOfEdge(p, 0)).toBeNull();
    p.geometry.regions.push({ id: "region1", type: "conductor", polygon: [[0, 0], [1, 0], [1, 1]] });
    expect(uniqueRegionId(p)).toBe("region2");
  });
});
