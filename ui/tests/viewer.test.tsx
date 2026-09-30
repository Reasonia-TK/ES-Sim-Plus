import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useConnection } from "../src/backend/connection";
import { GraphicsPanel } from "../src/graphics/GraphicsPanel";
import { useViewer } from "../src/graphics/viewerStore";
import { useDocument } from "../src/model/documentStore";
import { newProject } from "../src/model/project";
import { useSelection } from "../src/model/selection";
import { SettingsPanel } from "../src/pages/SettingsPanel";
import { useStatic } from "../src/results/staticResults";

beforeEach(() => {
  useDocument.getState().replace(newProject(), null, { untitledName: "test" });
  useViewer.setState({ tool: "select", profile: null, probe: null });
  useSelection.setState({ activeNode: "study:fem", selectedRegion: null, selectedPlacement: null });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("viewer", () => {
  it("switches tools from the toolbar and explains missing WebGL", () => {
    render(<GraphicsPanel />);
    const rect = screen.getByRole("button", { name: "矩形" });
    fireEvent.click(rect);
    expect(useViewer.getState().tool).toBe("rect");
    expect(rect.getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByText(/1 つ目の角をクリック/)).toBeTruthy();
    // 場が無いとプローブは使えない
    expect((screen.getByRole("button", { name: "プローブ" }) as HTMLButtonElement).disabled).toBe(true);
    // jsdom には WebGL が無い
    expect(screen.getByText(/WebGL2 が使えない/)).toBeTruthy();
  });

  it("shows the line profile from /profile with a CSV button", async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ s: [0, 0.05, 0.1], v: [0, 50, 100], e_abs: [1000, 1000, null] }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    render(<GraphicsPanel />);
    act(() => useViewer.getState().setProfile([[0, 0.025], [0.1, 0.025]]));
    expect(screen.getByText("ラインプロファイル")).toBeTruthy();
    await waitFor(() => expect((screen.getByRole("button", { name: "CSV 保存" }) as HTMLButtonElement).disabled).toBe(false));
    const body = JSON.parse((fetchMock.mock.calls[0] as unknown as [string, { body: string }])[1].body);
    expect(body.p1).toEqual([0, 0.025]);
    expect(body.n).toBe(200);
    fireEvent.click(screen.getByRole("button", { name: "閉じる" }));
    expect(useViewer.getState().profile).toBeNull();
  });
});

describe("electrostatics page", () => {
  it("disables compute while disconnected and summarises a result", async () => {
    useConnection.setState({ status: "disconnected" });
    render(<SettingsPanel />);
    const compute = screen.getByRole("button", { name: "計算" }) as HTMLButtonElement;
    expect(compute.disabled).toBe(true);
    // メッシュの作成と計算の両方に出る
    expect(screen.getAllByText("バックエンドに接続していないので計算できません").length).toBeGreaterThan(0);

    const mesh = { nodes: [[0, 0], [0.1, 0], [0.1, 0.05], [0, 0.05]], triangles: [[0, 1, 2], [0, 2, 3]], region_of_triangle: [-1, -1] };
    const result = {
      mesh,
      v: [0, 100, 100, 0],
      e_field: [[-1000, 0], [-1000, 0]],
      v_min: 0,
      v_max: 100,
      e_abs_max: 1000,
      energy: 4.4e-9,
      charges: [{ label: "edge3", voltage: 0, q: -8.85e-10 }],
      capacitance: 8.85e-12,
    };
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(result), { status: 200 })));
    useConnection.setState({ status: "connected" });
    await act(async () => {
      await useStatic.getState().runSolve();
    });
    expect(screen.getByText("0.0 / 100.0 V")).toBeTruthy();
    expect(screen.getByText("1.00e+3 V/m")).toBeTruthy();
    expect(screen.getByText("8.850e-12 F/m")).toBeTruthy();
    // 辺の名前で電荷を出す
    expect(screen.getByText(/左 .*\(0 V\)/)).toBeTruthy();
    // 設定を変えると「変わっています」
    act(() => useDocument.getState().update("v", (d) => void (d.geometry.boundaries[1].voltage = 50)));
    expect(screen.getByText(/計算のあとで設定が変わっています/)).toBeTruthy();
    act(() => useStatic.getState().clear());
  });
});
