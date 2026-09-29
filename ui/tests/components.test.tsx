import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { App } from "../src/app/App";
import { DialogHost } from "../src/app/DialogHost";
import { handleShortcut } from "../src/app/shortcuts";
import { isDirty, useDocument } from "../src/model/documentStore";
import { newProject, polygonBounds } from "../src/model/project";
import { useSelection } from "../src/model/selection";
import { SettingsPanel } from "../src/pages/SettingsPanel";
import { ModelTree } from "../src/tree/ModelTree";

function commit(input: HTMLElement, value: string) {
  fireEvent.focus(input);
  fireEvent.change(input, { target: { value } });
  fireEvent.blur(input);
}

beforeEach(() => {
  const p = newProject();
  p.geometry.regions.push({ id: "diel1", type: "dielectric", eps_r: 4, polygon: [[0.04, 0.01], [0.06, 0.01], [0.06, 0.04], [0.04, 0.04]] });
  useDocument.getState().replace(p, null, { untitledName: "test" });
  useSelection.setState({ activeNode: "domain", selectedRegion: null });
});

describe("settings panel", () => {
  it("edits the domain in display units, reports invalid values, and undoes", () => {
    render(<SettingsPanel />);
    const width = screen.getByLabelText("幅");
    expect((width as HTMLInputElement).value).toBe("100");
    commit(width, "120");
    const b = polygonBounds(useDocument.getState().project.geometry.domain.polygon);
    expect(b.x1 - b.x0).toBeCloseTo(0.12);
    expect(isDirty(useDocument.getState())).toBe(true);

    commit(width, "-1");
    expect(screen.getByRole("alert").textContent).toBe("> 0");
    expect((width as HTMLInputElement).value).toBe("120");

    act(() => useDocument.getState().undo());
    expect((screen.getByLabelText("幅") as HTMLInputElement).value).toBe("100");
    expect(isDirty(useDocument.getState())).toBe(false);
  });

  it("removes the axis boundary condition when switching to axisymmetric", () => {
    useDocument.getState().update("bc", (d) => {
      d.geometry.boundaries.push({ edges: [0, 2], type: "symmetry" });
    });
    render(<SettingsPanel />);
    fireEvent.change(screen.getByLabelText("座標系"), { target: { value: "rz" } });
    const bcs = useDocument.getState().project.geometry.boundaries;
    expect(bcs.find((b) => b.type === "symmetry")?.edges).toEqual([2]);
  });
});

describe("model tree", () => {
  it("selects, renames with F2 and deletes with confirmation", async () => {
    render(
      <>
        <ModelTree />
        <DialogHost />
      </>,
    );
    const tree = screen.getByRole("tree");
    fireEvent.click(within(tree).getByText("diel1"));
    expect(useSelection.getState().selectedRegion).toBe("diel1");

    fireEvent.keyDown(tree, { key: "F2" });
    const input = within(tree).getByLabelText("名前を変更");
    commit(input, "plate");
    expect(useDocument.getState().project.geometry.regions[0].id).toBe("plate");
    expect(useSelection.getState().activeNode).toBe("region:plate");

    fireEvent.keyDown(tree, { key: "Delete" });
    const dialog = await screen.findByRole("alertdialog");
    fireEvent.click(within(dialog).getByText("削除"));
    await act(async () => {});
    expect(useDocument.getState().project.geometry.regions).toHaveLength(0);
    expect(useSelection.getState().activeNode).toBe("regions");
  });

  it("moves the selection with the arrow keys", () => {
    render(<ModelTree />);
    const tree = screen.getByRole("tree");
    fireEvent.keyDown(tree, { key: "ArrowDown" });
    expect(useSelection.getState().activeNode).toBe("regions");
    fireEvent.keyDown(tree, { key: "ArrowUp" });
    fireEvent.keyDown(tree, { key: "ArrowLeft" });
    expect(useSelection.getState().activeNode).toBe("geometry");
  });
});

describe("app shell", () => {
  it("renders the menu, tree, settings and graphics, and handles undo shortcuts outside inputs", () => {
    render(<App />);
    expect(screen.getByRole("menubar")).toBeTruthy();
    expect(screen.getByRole("tree")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "ドメイン" })).toBeTruthy();
    useDocument.getState().update("mesh", (d) => void (d.mesh.size = 0.001));
    const handled = handleShortcut(new KeyboardEvent("keydown", { key: "z", ctrlKey: true }));
    expect(handled).toBe(true);
    expect(useDocument.getState().project.mesh.size).toBe(0.004);
    // 入力欄の中では元に戻すを横取りしない
    const input = screen.getByLabelText("幅");
    const ev = new KeyboardEvent("keydown", { key: "z", ctrlKey: true });
    Object.defineProperty(ev, "target", { value: input });
    expect(handleShortcut(ev)).toBe(false);
    expect(document.title).toContain("test");
  });
});
