import { beforeEach, describe, expect, it } from "vitest";
import { HISTORY_LIMIT, isDirty, useDocument } from "../src/model/documentStore";
import { newProject } from "../src/model/project";

const doc = () => useDocument.getState();

beforeEach(() => {
  doc().replace(newProject(), null);
});

describe("document store", () => {
  it("records each edit as one undoable step and tracks the saved state", () => {
    expect(isDirty(doc())).toBe(false);
    doc().update("mesh", (d) => {
      d.mesh.size = 0.002;
    });
    doc().update("coord", (d) => {
      d.coord = "rz";
    });
    expect(doc().project.mesh.size).toBe(0.002);
    expect(isDirty(doc())).toBe(true);
    expect(doc().past.map((e) => e.label)).toEqual(["mesh", "coord"]);

    doc().undo();
    expect(doc().project.coord).toBe("xy");
    expect(doc().project.mesh.size).toBe(0.002);
    doc().undo();
    expect(doc().project.mesh.size).toBe(0.004);
    // 元に戻して保存時と同じ状態になれば未保存の印も消える
    expect(isDirty(doc())).toBe(false);

    doc().redo();
    doc().redo();
    expect(doc().project.coord).toBe("rz");
    expect(isDirty(doc())).toBe(true);
    doc().redo(); // 何もしない
    expect(doc().future).toHaveLength(0);
  });

  it("ignores edits that change nothing and clears redo on a new edit", () => {
    doc().update("noop", (d) => {
      d.mesh.size = d.mesh.size;
    });
    expect(doc().past).toHaveLength(0);
    doc().update("a", (d) => void (d.mesh.size = 1));
    doc().undo();
    expect(doc().future).toHaveLength(1);
    doc().update("b", (d) => void (d.mesh.size = 2));
    expect(doc().future).toHaveLength(0);
  });

  it("keeps the project immutable (structural sharing)", () => {
    const before = doc().project;
    doc().update("mesh", (d) => void (d.mesh.size = 0.001));
    expect(before.mesh.size).toBe(0.004);
    expect(doc().project.geometry).toBe(before.geometry);
  });

  it("limits the history length", () => {
    for (let i = 0; i < HISTORY_LIMIT + 20; i++) doc().update(`e${i}`, (d) => void (d.mesh.size = 1 + i));
    expect(doc().past).toHaveLength(HISTORY_LIMIT);
    expect(doc().past[0].label).toBe("e20");
  });

  it("marks saved, replaces documents and can open a recovered copy as unsaved", () => {
    doc().update("mesh", (d) => void (d.mesh.size = 0.003));
    doc().markSaved({ name: "a.json", path: "C:/a.json" });
    expect(isDirty(doc())).toBe(false);
    expect(doc().file?.name).toBe("a.json");
    const serial = doc().docSerial;
    doc().replace(newProject(), null, { untitledName: "recovered", dirty: true });
    expect(isDirty(doc())).toBe(true);
    expect(doc().past).toHaveLength(0);
    expect(doc().untitledName).toBe("recovered");
    expect(doc().docSerial).toBe(serial + 1);
  });
});
