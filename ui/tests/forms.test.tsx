import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { OptionalBlock } from "../src/forms/blocks";
import { SchemaField } from "../src/forms/SchemaField";
import { useDocument } from "../src/model/documentStore";
import { newProject } from "../src/model/project";
import { useSelection } from "../src/model/selection";
import { SettingsPanel } from "../src/pages/SettingsPanel";
import { usePrefs } from "../src/prefs/prefs";
import { DEFAULT_MERGE, DEFAULT_PIC, defaultDsmc, defaultFluid1d, defaultFluid2d, defaultParticles, defaultPic1d, defaultTl } from "../src/schema/defaults";

function commit(input: HTMLElement, value: string) {
  fireEvent.focus(input);
  fireEvent.change(input, { target: { value } });
  fireEvent.blur(input);
}

const doc = () => useDocument.getState();

beforeEach(() => {
  doc().replace(newProject(), null, { untitledName: "t" });
  usePrefs.setState({ showAdvanced: false, lengthUnit: "mm" });
});

describe("schema fields", () => {
  it("commits unit-aware input in SI and rejects wrong units and out-of-range values", () => {
    doc().update("init", (d) => void (d.fluid2d = defaultFluid2d()));
    render(<SchemaField path={["fluid2d", "gas_pressure_pa"]} />);
    const input = screen.getByLabelText("ガス圧");
    expect(screen.getByText("Pa")).toBeTruthy();
    commit(input, "100 mTorr");
    expect((doc().project.fluid2d as { gas_pressure_pa: number }).gas_pressure_pa).toBeCloseTo(13.3322368);
    commit(input, "5 kV");
    expect(screen.getByRole("alert").textContent).toContain("Pa");
    commit(input, "0");
    expect(screen.getByRole("alert").textContent).toBe("> 0 Pa");
    expect((doc().project.fluid2d as { gas_pressure_pa: number }).gas_pressure_pa).toBeCloseTo(13.3322368);
  });

  it("shows geometric lengths in the display unit and hides advanced fields", () => {
    doc().update("init", (d) => void (d.pic1d = defaultPic1d()));
    const { rerender } = render(
      <>
        <SchemaField path={["pic1d", "gap_m"]} />
        <SchemaField path={["pic1d", "seed"]} />
      </>,
    );
    expect((screen.getByLabelText("ギャップ") as HTMLInputElement).value).toBe("20");
    expect(screen.queryByLabelText("乱数シード")).toBeNull();
    act(() => usePrefs.setState({ showAdvanced: true, lengthUnit: "um" }));
    rerender(
      <>
        <SchemaField path={["pic1d", "gap_m"]} />
        <SchemaField path={["pic1d", "seed"]} />
      </>,
    );
    expect((screen.getByLabelText("ギャップ") as HTMLInputElement).value).toBe("20000");
    expect(screen.getByLabelText("乱数シード")).toBeTruthy();
  });

  it("clears nullable fields to null (auto)", () => {
    doc().update("init", (d) => void (d.pic = structuredClone(DEFAULT_PIC)));
    doc().update("dt", (d) => void ((d.pic as { dt: number | null }).dt = 1e-10));
    render(<SchemaField path={["pic", "dt"]} />);
    const input = screen.getByLabelText("時間刻み dt") as HTMLInputElement;
    expect(input.value).toBe("1e-10");
    commit(input, "");
    expect((doc().project.pic as { dt: unknown }).dt).toBeNull();
    expect(input.placeholder).toBe("自動");
  });
});

describe("optional blocks ([R])", () => {
  it("remembers the values when disabled and restores them", () => {
    doc().update("init", (d) => void (d.pic = structuredClone(DEFAULT_PIC)));
    render(
      <OptionalBlock path={["pic", "merge"]} title="マージ" defaults={() => ({ ...DEFAULT_MERGE })}>
        <SchemaField path={["pic", "merge", "every"]} />
      </OptionalBlock>,
    );
    const toggle = screen.getByRole("checkbox");
    fireEvent.click(toggle);
    expect((doc().project.pic as { merge: unknown }).merge).toEqual(DEFAULT_MERGE);
    commit(screen.getByLabelText("確認の間隔"), "250");
    fireEvent.click(toggle);
    expect((doc().project.pic as { merge: unknown }).merge).toBeNull();
    fireEvent.click(toggle);
    expect((doc().project.pic as { merge: { every: number } }).merge.every).toBe(250);
    // 有効/無効の切り替えも元に戻せる
    act(() => doc().undo());
    expect((doc().project.pic as { merge: unknown }).merge).toBeNull();
  });
});

describe("settings pages", () => {
  it("renders every page with all studies enabled", () => {
    const p = newProject();
    p.geometry.regions.push({ id: "c1", type: "conductor", voltage: 5, shape: { kind: "circle", center: [0.05, 0.025], radius: 0.005 } });
    p.geometry.regions.push({ id: "d1", type: "dielectric", eps_r: 4, polygon: [[0.01, 0.01], [0.02, 0.01], [0.02, 0.02], [0.01, 0.02]] });
    p.mesh.mode = "cartesian";
    p.mesh.amr = { max_level: 2, regions: [{ p1: [0, 0], p2: [0.01, 0.01], level: 1 }] };
    Object.assign(p, {
      particles: defaultParticles(p),
      pic: { ...structuredClone(DEFAULT_PIC), mcc: { gas: { name: "Ar", pressure_pa: 10, temperature_k: 300 }, electron_processes: [], ion_processes: [] } },
      pic1d: defaultPic1d(),
      fluid1d: defaultFluid1d(),
      fluid2d: defaultFluid2d(),
      dsmc: { ...defaultDsmc(), boundaries: [{ edges: [3], type: "inlet", temperature_k: 300, pressure_pa: 10 }] },
      tl: defaultTl(),
    });
    doc().replace(p, null);
    usePrefs.setState({ showAdvanced: true });
    const nodes = ["project", "domain", "regions", "region:c1", "region:d1", "boundaries", "edge:0", "edge:1", "mesh", "bfield",
      "study:fem", "study:trace", "study:pic", "study:pic1d", "study:fluid1d", "study:fluid2d", "study:dsmc", "study:tl", "study:sweep", "results"];
    for (const n of nodes) {
      act(() => useSelection.setState({ activeNode: n }));
      const { unmount } = render(<SettingsPanel />);
      expect(screen.getByRole("heading")).toBeTruthy();
      unmount();
    }
  });

  it("changes an edge to periodic and pairs it with the opposite edge", () => {
    useSelection.setState({ activeNode: "edge:0" });
    render(<SettingsPanel />);
    fireEvent.change(screen.getByLabelText("種類"), { target: { value: "periodic" } });
    expect(doc().project.geometry.boundaries.find((b) => b.type === "periodic")?.edges).toEqual([0, 2]);
  });

  it("enables a study with defaults and remembers it when disabled", () => {
    useSelection.setState({ activeNode: "study:fluid2d" });
    render(<SettingsPanel />);
    const toggle = within(document.querySelector(".study-toggle") as HTMLElement).getByRole("checkbox");
    fireEvent.click(toggle);
    expect((doc().project.fluid2d as { gas_pressure_pa: number }).gas_pressure_pa).toBe(50);
    commit(screen.getByLabelText("ガス圧"), "20");
    fireEvent.click(toggle);
    expect(doc().project.fluid2d).toBeNull();
    fireEvent.click(toggle);
    expect((doc().project.fluid2d as { gas_pressure_pa: number }).gas_pressure_pa).toBe(20);
  });

  it("adds an RF component with v1's defaults", () => {
    useSelection.setState({ activeNode: "edge:1" });
    render(<SettingsPanel />);
    fireEvent.click(screen.getByLabelText("RF を重ねる"));
    expect(doc().project.geometry.boundaries[1].voltage_rf).toEqual({ amplitude: 100, freq_hz: 13.56e6, phase_deg: 0 });
    fireEvent.click(screen.getByText("RF 成分を追加"));
    expect(doc().project.geometry.boundaries[1].voltage_rf).toHaveLength(2);
    commit(screen.getAllByLabelText("周波数")[1], "4 MHz");
    expect((doc().project.geometry.boundaries[1].voltage_rf as { freq_hz: number }[])[1].freq_hz).toBe(4e6);
  });
});
