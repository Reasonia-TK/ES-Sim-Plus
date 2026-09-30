// P6f (仕上げ) で直したところ: 式の入力・Esc で戻す・スイープの補完・ツリーの状態・選択の片付け・文書を変えたときの
// 出す実行・PIC の注入エミッタ・静電場の |E| の範囲と背景なし・未接続のジョブ・軸対称への切り替え。

import { act, fireEvent, render, screen } from "@testing-library/react";
import { t } from "i18next";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { injectorOf } from "../src/graphics/projectOverlays";
import { staticScene } from "../src/graphics/staticScene";
import { JobRow } from "../src/jobs/JobRow";
import { useJobs } from "../src/jobs/jobsStore";
import type { JobSummary } from "../src/jobs/types";
import { useDocument } from "../src/model/documentStore";
import { placeInjectionEmitter } from "../src/model/placements";
import { newProject, type Project } from "../src/model/project";
import { useSelection } from "../src/model/selection";
import { prepareSweepProject, valueAtPath } from "../src/model/sweep";
import { CoordSelect } from "../src/pages/GeometryPages";
import { CommitText } from "../src/pages/inputs";
import { useResultsView } from "../src/results/resultsView";
import type { SolveEntry } from "../src/results/staticResults";
import { evalExpression, parseQuantity } from "../src/schema/units";
import { buildTree, filterTree } from "../src/tree/treeModel";

const mm = { lengthUnit: "mm" as const, axisymmetric: false };
const doc = () => useDocument.getState();

beforeEach(() => {
  doc().replace(newProject(), null, { untitledName: "t" });
});

describe("expressions in number fields", () => {
  it("evaluates arithmetic with precedence, functions and pi", () => {
    expect(evalExpression("(1+2)*3")).toBe(9);
    expect(evalExpression("2^10")).toBe(1024);
    expect(evalExpression("-2^2")).toBe(-4);
    expect(evalExpression("2^-1")).toBe(0.5);
    expect(evalExpression("sqrt(2)")).toBeCloseTo(Math.SQRT2);
    expect(evalExpression("2*pi")).toBeCloseTo(2 * Math.PI);
    expect(evalExpression("1e-3/2")).toBe(5e-4);
    expect(evalExpression("2*")).toBeNull();
    expect(evalExpression("foo(1)")).toBeNull();
  });

  it("parses an expression followed by a unit", () => {
    expect(parseQuantity("2*6.78 MHz", "Hz", false, mm)).toEqual({ ok: true, value: 13.56e6 });
    expect(parseQuantity("1/4", "1", false, mm)).toEqual({ ok: true, value: 0.25 });
    const r = parseQuantity("10/2 mm", undefined, true, mm);
    expect(r.ok && r.value).toBeCloseTo(0.005);
    // 表示単位 (mm) のまま式だけ
    const g = parseQuantity("2*5", undefined, true, mm);
    expect(g.ok && g.value).toBeCloseTo(0.01);
    expect(parseQuantity("2*3 kg", "Hz", false, mm)).toEqual({ ok: false, error: "unit" });
    // 数だけ・単位付きは今まで通り
    expect(parseQuantity("13.56 MHz", "Hz", false, mm)).toEqual({ ok: true, value: 13.56e6 });
  });

  it("does not commit when editing is cancelled with Esc", () => {
    const onCommit = vi.fn();
    render(<CommitText value="3" onCommit={onCommit} aria-label="x" />);
    const input = screen.getByLabelText("x");
    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: "5" } });
    fireEvent.keyDown(input, { key: "Escape" });
    expect(onCommit).not.toHaveBeenCalled();
    expect((input as HTMLInputElement).value).toBe("3");
    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: "7" } });
    fireEvent.keyDown(input, { key: "Enter" });
    fireEvent.blur(input);
    expect(onCommit).toHaveBeenCalledWith("7");
  });
});

describe("sweep preparation", () => {
  it("fills a missing optional key with 0 and wraps a single 1D RF component (on a copy)", () => {
    const p = newProject();
    p.pic1d = { left: { v_dc: 0, voltage_rf: { amplitude: 100, freq_hz: 13.56e6, phase_deg: 0 } }, right: { v_dc: 0 } } as unknown as Project["pic1d"];
    const q = prepareSweepProject(p, "geometry.boundaries.0.see_gamma");
    expect(valueAtPath(q, "geometry.boundaries.0.see_gamma")).toBe(0);
    expect(valueAtPath(p, "geometry.boundaries.0.see_gamma")).toBeUndefined();
    const r = prepareSweepProject(p, "pic1d.left.voltage_rf.0.amplitude");
    expect(valueAtPath(r, "pic1d.left.voltage_rf.0.amplitude")).toBe(100);
    expect(Array.isArray((p.pic1d as { left: { voltage_rf: unknown } }).left.voltage_rf)).toBe(false);
  });
});

describe("tree", () => {
  it("shows the electrostatics state, sweep case counts and the highest AMR level", () => {
    const p = newProject();
    p.mesh = { ...p.mesh, mode: "cartesian", amr: { max_level: 1, regions: [{ p1: [0, 0], p2: [0.01, 0.01], level: 3 }] } };
    const sweep: JobSummary = {
      id: "s",
      kind: "sweep",
      seq: 1,
      label: null,
      state: "running",
      stopping: false,
      created: 1,
      started_at: 1,
      finished_at: null,
      elapsed_s: 1,
      runs: 1,
      progress: { step: 2, n_steps: 5, step_offset: 0, fraction: 0.4 },
      error: null,
      warnings: [],
      can_continue: false,
      has_result: false,
      queue_position: null,
      options: {},
    };
    const root = buildTree(p, t, "mm", "d", [sweep], { busy: false, solved: true, stale: true });
    const find = (n: typeof root, id: string): typeof root | undefined => (n.id === id ? n : n.children?.map((c) => find(c, id)).find(Boolean));
    expect(find(root, "study:fem")?.badge?.text).toBe("設定が変更");
    expect(find(root, "study:sweep")?.badge?.text).toBe("実行中 2/5");
    expect(find(root, "mesh")?.detail).toContain("AMR L3");
    // 群の名前で探すと子をすべて出す
    const f = filterTree(root, "境界条件")!;
    expect(find(f, "edge:e1")).toBeTruthy();
  });
});

describe("selection and shown run", () => {
  it("drops a region selection that disappears on undo and forgets the shown run on a new document", () => {
    doc().update("add", (d) => void d.geometry.regions.push({ id: "r1", type: "conductor", polygon: [[0, 0], [0.01, 0], [0.01, 0.01]], voltage: 0 }));
    useSelection.getState().select("region:r1");
    act(() => doc().undo());
    expect(useSelection.getState().selectedRegion).toBeNull();
    expect(useSelection.getState().activeNode).toBe("regions");
    useResultsView.getState().setActiveRun("x");
    act(() => doc().replace(newProject(), null, { untitledName: "u" }));
    expect(useResultsView.getState().activeRun).toBeNull();
  });
});

describe("PIC injection emitter", () => {
  it("places the injection emitter on the canvas (creating a default injection) and shows it", () => {
    const p = newProject();
    expect(injectorOf(p)).toBeNull();
    doc().update("place", (d) => void placeInjectionEmitter(d, [0.01, 0.01], [0.01, 0.04]));
    const v = injectorOf(doc().project);
    expect(v?.p1).toEqual([0.01, 0.01]);
    expect(v?.p2).toEqual([0.01, 0.04]);
    const inj = (doc().project.pic as { injection: { species: string } }).injection;
    expect(inj.species).toBe("electron");
  });
});

describe("electrostatics display", () => {
  it("colours |E| from 0 and can hide the background", () => {
    const mesh = { nodes: [[0, 0], [1, 0], [0, 1]] as [number, number][], triangles: [[0, 1, 2]] as [number, number, number][], region_of_triangle: [-1] };
    const solve = {
      result: { mesh, v: [0, 1, 2], e_field: [[3, 4]], v_min: 0, v_max: 2, e_abs_max: 5, energy: 0, charges: [], capacitance: null },
      project: newProject(),
      view: { nodes: mesh.nodes, triangles: mesh.triangles },
      eAbs: Float64Array.from([5]),
      key: "",
      elapsedS: 0,
    } as unknown as SolveEntry;
    const labels = { meshTitle: () => "", potential: "V", field: "E" };
    const e = staticScene({ mesh: null, solve, latest: "solve" }, "e_abs", newProject(), labels);
    expect(e.field?.stats).toMatchObject({ min: 0, max: 5 });
    const none = staticScene({ mesh: null, solve, latest: "solve" }, "none", newProject(), labels);
    expect(none.field).toBeNull();
  });
});

describe("jobs while disconnected", () => {
  it("shows running jobs as disconnected and disables stop", () => {
    useJobs.setState({ connected: false });
    const job: JobSummary = {
      id: "j",
      kind: "pic1d",
      seq: 1,
      label: null,
      state: "running",
      stopping: false,
      created: 1,
      started_at: 1,
      finished_at: null,
      elapsed_s: 1,
      runs: 1,
      progress: null,
      error: null,
      warnings: ["ωpe·dt = 0.4 > 0.3"],
      can_continue: false,
      has_result: false,
      queue_position: null,
      options: {},
    };
    render(<JobRow job={job} />);
    expect(screen.getByText("未接続")).toBeTruthy();
    expect((screen.getByRole("button", { name: "停止" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText(/ωpe·dt/)).toBeTruthy();
  });
});

describe("axisymmetric switch", () => {
  it("drops a periodic pair that includes the axis edge and clears B", () => {
    doc().update("setup", (d) => {
      d.geometry.boundaries = [
        { edges: [1], type: "dirichlet", voltage: 100 },
        { edges: [0, 2], type: "periodic" },
      ];
      d.b_field = { bx: 0.01, by: 0, bz: 0 };
    });
    render(<CoordSelect />);
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "rz" } });
    const p = doc().project;
    expect(p.geometry.boundaries).toEqual([{ edges: [1], type: "dirichlet", voltage: 100 }]);
    expect(p.b_field).toBeNull();
  });
});
