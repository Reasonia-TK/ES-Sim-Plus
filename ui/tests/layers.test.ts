import { produce } from "immer";
import { beforeEach, describe, expect, it } from "vitest";
import { itemsInBox } from "../src/graphics/editTargets";
import { useDocument } from "../src/model/documentStore";
import { transformItems, trimSketch } from "../src/model/editOps";
import {
  activeLayer,
  addLayer,
  deleteLayer,
  isPickable,
  itemLayer,
  layerColor,
  layerCounts,
  layerNameError,
  layersOf,
  pickableRegions,
  setActiveLayer,
  setItemLayer,
  updateLayer,
  visibleRegions,
  visibleSketch,
} from "../src/model/layers";
import { translation } from "../src/cad/path";
import { newProject, type Project } from "../src/model/project";
import { addPolygonRegion, duplicateRegion } from "../src/model/regionOps";
import { useSelection } from "../src/model/selection";
import { addSketch, replaceSketch, sketchOf } from "../src/model/sketch";

function base(): Project {
  return produce(newProject(), (d) => {
    d.geometry.regions.push({ id: "a", type: "conductor", voltage: 0, polygon: [[0.01, 0.01], [0.02, 0.01], [0.02, 0.02], [0.01, 0.02]] });
    addSketch(d, { kind: "line", a: [0, 0.03], b: [0.05, 0.03] });
  });
}

describe("layers", () => {
  beforeEach(() => useDocument.getState().replace(base(), null));

  it("always has the default layer and manages layers", () => {
    let p = base();
    expect(layersOf(p)).toEqual([{ id: "0", name: "0" }]);
    expect(activeLayer(p)).toBe("0");
    let id = "";
    p = produce(p, (d) => {
      id = addLayer(d);
      setActiveLayer(d, id);
      updateLayer(d, id, { name: "電極", color: "#ff0000" });
    });
    expect(id).toBe("L1");
    expect(layersOf(p).map((l) => l.name)).toEqual(["0", "電極"]);
    expect(activeLayer(p)).toBe("L1");
    expect(layerNameError(p, "L2", "電極")).toBe("layers.nameDuplicate");
    expect(layerNameError(p, "L1", " ")).toBe("layers.nameEmpty");
    // 既定のレイヤの名前は変えない
    p = produce(p, (d) => updateLayer(d, "0", { name: "x" }));
    expect(layersOf(p)[0].name).toBe("0");
  });

  it("puts new shapes on the current layer and keeps copies on their layer", () => {
    const st = useDocument.getState();
    st.update("layer", (d) => void setActiveLayer(d, addLayer(d)));
    st.update("draw", (d) => void addPolygonRegion(d, [[0.03, 0.01], [0.04, 0.01], [0.04, 0.02]]));
    st.update("sketch", (d) => void addSketch(d, { kind: "circle", center: [0.05, 0.02], r: 0.005 }));
    let p = useDocument.getState().project;
    expect(itemLayer(p, p.geometry.regions[1])).toBe("L1");
    expect(itemLayer(p, sketchOf(p)[1])).toBe("L1");
    // 既定のレイヤの領域の複製・変換のコピーは既定のレイヤのまま
    st.update("dup", (d) => void duplicateRegion(d, "a"));
    st.update("copy", (d) => void transformItems(d, [{ kind: "sketch", id: "s1" }], translation(0.001, 0), true));
    p = useDocument.getState().project;
    expect(itemLayer(p, p.geometry.regions[2])).toBe("0");
    expect(itemLayer(p, sketchOf(p)[2])).toBe("0");
    expect(layerCounts(p).get("L1")).toBe(2);
  });

  it("keeps the layer when a sketch is replaced or trimmed", () => {
    let p = produce(base(), (d) => {
      const id = addLayer(d);
      setItemLayer(d, { kind: "sketch", id: "s1" }, id);
      replaceSketch(d, "s1", { kind: "line", a: [0, 0.03], b: [0.06, 0.03] });
      addSketch(d, { kind: "line", a: [0.02, 0], b: [0.02, 0.05] });
      addSketch(d, { kind: "line", a: [0.04, 0], b: [0.04, 0.05] });
    });
    expect(itemLayer(p, sketchOf(p)[0])).toBe("L1");
    p = produce(p, (d) => void trimSketch(d, "s1", [0.03, 0.03], [...sketchOf(d as Project).slice(1).map((e) => ({ kind: "line" as const, a: (e as { a: [number, number] }).a, b: (e as { b: [number, number] }).b }))]));
    const pieces = sketchOf(p).filter((e) => e.kind === "line" && e.a[1] === 0.03);
    expect(pieces).toHaveLength(2);
    expect(pieces.every((e) => itemLayer(p, e) === "L1")).toBe(true);
  });

  it("hides and locks layers for drawing and picking", () => {
    let p = produce(base(), (d) => {
      const id = addLayer(d);
      setItemLayer(d, { kind: "region", id: "a" }, id);
      updateLayer(d, id, { color: "#00ff00" });
    });
    expect(layerColor(p, p.geometry.regions[0])).toBe("#00ff00");
    expect(itemsInBox(p, [0, 0], [0.1, 0.1])).toEqual([
      { kind: "region", id: "a" },
      { kind: "sketch", id: "s1" },
    ]);
    const locked = produce(p, (d) => updateLayer(d, "L1", { locked: true }));
    expect(visibleRegions(locked)).toHaveLength(1);
    expect(pickableRegions(locked)).toHaveLength(0);
    expect(isPickable(locked, locked.geometry.regions[0])).toBe(false);
    expect(itemsInBox(locked, [0, 0], [0.1, 0.1])).toEqual([{ kind: "sketch", id: "s1" }]);
    const hidden = produce(p, (d) => updateLayer(d, "0", { visible: false }));
    expect(visibleSketch(hidden)).toHaveLength(0);
    expect(visibleRegions(hidden)).toHaveLength(1);
    // 消したレイヤの形は既定のレイヤへ
    p = produce(p, (d) => deleteLayer(d, "L1"));
    expect(itemLayer(p, p.geometry.regions[0])).toBe("0");
    expect(p.geometry.regions[0].layer).toBeUndefined();
  });

  it("unpicks shapes whose layer becomes locked", () => {
    const st = useDocument.getState();
    st.update("layer", (d) => {
      const id = addLayer(d);
      setItemLayer(d, { kind: "region", id: "a" }, id);
    });
    useSelection.getState().pick([{ kind: "region", id: "a" }, { kind: "sketch", id: "s1" }]);
    st.update("lock", (d) => updateLayer(d, "L1", { locked: true }));
    expect(useSelection.getState().picked).toEqual([{ kind: "sketch", id: "s1" }]);
  });
});
