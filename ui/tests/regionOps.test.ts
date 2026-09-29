import { produce } from "immer";
import { describe, expect, it } from "vitest";
import { newProject, type Project } from "../src/model/project";
import {
  addCircleRegion,
  addRectRegion,
  deleteRegion,
  duplicateRegion,
  renameRegion,
  setRegionType,
  validateRegionId,
} from "../src/model/regionOps";

function withRegions(): Project {
  let p = newProject();
  p = produce(p, (d) => {
    addRectRegion(d);
    addCircleRegion(d);
    d.mesh.local_sizes = [
      { region: "region1", size: 0.001 },
      { region: "region2", size: 0.002 },
    ];
  });
  return p;
}

describe("region operations", () => {
  it("adds rectangles and circles at the domain centre", () => {
    const p = withRegions();
    const [r1, r2] = p.geometry.regions;
    expect(r1).toMatchObject({ id: "region1", type: "conductor", voltage: 0 });
    expect(r1.polygon).toHaveLength(4);
    expect(r2.shape?.center).toEqual([0.05, 0.025]);
    expect(r2.shape?.radius).toBeCloseTo(0.05 / 8);
  });

  it("keeps mesh.local_sizes consistent on rename and delete", () => {
    let p = produce(withRegions(), (d) => renameRegion(d, "region1", "anode"));
    expect(p.geometry.regions[0].id).toBe("anode");
    expect(p.mesh.local_sizes).toEqual([
      { region: "anode", size: 0.001 },
      { region: "region2", size: 0.002 },
    ]);
    p = produce(p, (d) => deleteRegion(d, "region2"));
    expect(p.geometry.regions.map((r) => r.id)).toEqual(["anode"]);
    expect(p.mesh.local_sizes).toEqual([{ region: "anode", size: 0.001 }]);
  });

  it("duplicates with an offset and a fresh id", () => {
    let copy: string | null = null;
    const p = produce(withRegions(), (d) => {
      copy = duplicateRegion(d, "region2");
    });
    expect(copy).toBe("region2_1");
    const c = p.geometry.regions[2];
    expect(c.shape?.center[0]).toBeCloseTo(0.05 + 0.0025);
    expect(p.geometry.regions[1].shape?.center[0]).toBe(0.05);
  });

  it("changes the type keeping the shape and dropping RF/waveform/γ", () => {
    const p = produce(withRegions(), (d) => {
      Object.assign(d.geometry.regions[0], { voltage_rf: { amplitude: 1, freq_hz: 1 }, see_gamma: 0.1 });
      setRegionType(d, "region1", "dielectric");
    });
    const r = p.geometry.regions[0];
    expect(r.type).toBe("dielectric");
    expect(r.eps_r).toBe(1);
    expect(r.voltage_rf).toBeUndefined();
    expect(r.see_gamma).toBeUndefined();
    expect(r.polygon).toHaveLength(4);
  });

  it("validates ids", () => {
    const p = withRegions();
    expect(validateRegionId(p, "region1", " ")).toBe("settings.idEmpty");
    expect(validateRegionId(p, "region1", "region2")).toBe("settings.idDuplicate");
    expect(validateRegionId(p, "region1", "region1")).toBeNull();
    expect(validateRegionId(p, "region1", "cathode")).toBeNull();
  });
});
