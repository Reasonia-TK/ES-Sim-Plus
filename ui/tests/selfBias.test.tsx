// 阻止コンデンサ (自己バイアス、prompts/134) の UI: 容量の単位、設定欄の既定値、種類の変更・ツリー・スイープの
// 候補、結果の数値・グラフ・RF 波形モニタの実際の電極の電位、スイープのケースの自己バイアス。

import { act, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import i18n from "../src/i18n";
import { useJobs } from "../src/jobs/jobsStore";
import type { JobSummary } from "../src/jobs/types";
import { useDocument } from "../src/model/documentStore";
import { newProject, type Project } from "../src/model/project";
import { setRegionType } from "../src/model/regionOps";
import { buildSweepCandidates } from "../src/model/sweep";
import { defaultCapacitance } from "../src/pages/widgets/CapacitorEditor";
import { electrodes2d } from "../src/results/charts/Charts2d";
import { electrodeVoltage, type RfElectrode } from "../src/results/charts/RfMonitor";
import { attachCircuit, circuitLabel, circuitResultRows } from "../src/results/charts/SelfBias";
import { ResultsCharts } from "../src/results/ResultsCharts";
import { DEFAULT_DISPLAY, useResultsView } from "../src/results/resultsView";
import { RunSummary } from "../src/results/RunSummary";
import type { CircuitResult } from "../src/results/types";
import { displayUnit, parseQuantity, unitFactor, type UnitContext } from "../src/schema/units";
import { edgeSummary } from "../src/tree/treeModel";

const t = i18n.t.bind(i18n) as unknown as Parameters<typeof circuitLabel>[2];
const planar: UnitContext = { lengthUnit: "mm", axisymmetric: false };
const axi: UnitContext = { lengthUnit: "mm", axisymmetric: true };

function job(p: Partial<JobSummary> = {}): JobSummary {
  return {
    id: "r1", kind: "fluid1d", seq: 1, label: null, state: "done", stopping: false, created: 1000, started_at: 1000,
    finished_at: 1001, elapsed_s: 1, runs: 1, progress: null, error: null, warnings: [], can_continue: true,
    has_result: true, queue_position: null, options: {}, ...p,
  };
}

function circuit(label = "left", units = { capacitance: "F/m^2", charge: "C/m^2", current: "A/m^2" }): CircuitResult {
  const T = 1 / 13.56e6;
  return {
    period_s: T,
    units,
    c_matrix: [[3.54e-10]],
    electrodes: [
      {
        label, capacitance: 1e-7, initial_bias_v: 0, c_self: 3.54e-10, rf_division: 0.99647, v_e: -41.2, q_node: -1e-6,
        t: [T, 2 * T, 3 * T], v_dc: [-20, -35.5, -39.66], v1: [86, 85.6, 85.53], i_dc: [-0.5, -0.1, -2.653e-3],
        last_period: { t: [0, T / 4, T / 2, (3 * T) / 4], v: [-39.7, 46, -39.7, -125] },
      },
    ],
  };
}

const initialJobs = useJobs.getState();

beforeEach(() => {
  useJobs.setState({ ...initialJobs, jobs: {}, started: {}, startedFull: {}, frames: {}, liveMesh: {}, liveHistory: {}, cases: {}, results: {}, inputs: {}, runSince: {}, watching: [] }, true);
  useResultsView.setState({ activeRun: null, bin: 0, playing: false, playBins: 0, chartPrefs: {}, display: structuredClone(DEFAULT_DISPLAY) });
  useDocument.getState().replace(newProject(), null, { untitledName: "test" });
});

describe("capacitance units and defaults", () => {
  it("reads F/m per depth in planar mode and F (total) in axisymmetric mode, with SI prefixes", () => {
    expect(displayUnit("F/m", false, planar)).toBe("F/m");
    expect(displayUnit("F/m", false, axi)).toBe("F");
    expect(displayUnit("F/m^2", false, planar)).toBe("F/m²");
    expect(unitFactor("nF/m", "F/m", planar)).toBe(1e-9);
    expect(unitFactor("nF", "F/m", axi)).toBe(1e-9);
    expect(unitFactor("nF", "F/m", planar)).toBeNull();
    expect(unitFactor("µF/m²", "F/m^2", planar)).toBe(1e-6);
    expect(unitFactor("µF/m^2", "F/m^2", planar)).toBe(1e-6);
    const r = parseQuantity("5 nF", "F/m", false, axi);
    expect(r.ok && r.value).toBeCloseTo(5e-9, 20);
    // 既存の A/m → A (軸対称) は変わらない
    expect(displayUnit("A/m", false, axi)).toBe("A");
    expect(unitFactor("mA", "A/m", axi)).toBe(1e-3);
  });

  it("enables the capacitor with the literature value scaled to the coordinate system", () => {
    expect(defaultCapacitance("2d", "rz")).toBe(5e-9);
    expect(defaultCapacitance("2d", "rz_x0")).toBe(5e-9);
    expect(defaultCapacitance("2d", "xy")).toBe(5e-8);
    expect(defaultCapacitance("1d", "xy")).toBe(6e-7);
  });
});

describe("document operations", () => {
  function project(): Project {
    const p = newProject();
    p.geometry.regions.push({ id: "rf", type: "conductor", polygon: [[0, 0], [0.01, 0], [0.01, 0.01]], voltage: 0, blocking_capacitor: { capacitance: 5e-9 } });
    p.geometry.boundaries = [{ edges: [1], type: "dirichlet", voltage: 0, blocking_capacitor: { capacitance: 1e-8 } }];
    p.fluid1d = { gap_m: 0.02, left: { v_dc: 0, voltage_rf: [{ amplitude: 100, freq_hz: 13.56e6 }], blocking_capacitor: { capacitance: 1e-7 } }, right: { v_dc: 0 } };
    return p;
  }

  it("drops the capacitor when a conductor becomes another type and marks capacitor edges in the tree", () => {
    const p = project();
    setRegionType(p as never, "rf", "dielectric");
    expect(p.geometry.regions[0].blocking_capacitor).toBeUndefined();
    expect(edgeSummary(p, 1, t)).toContain("C_b");
  });

  it("offers the fluid 1D capacitor as sweep parameters", () => {
    const paths = buildSweepCandidates(project(), t).map((c) => c.path);
    expect(paths).toContain("fluid1d.left.blocking_capacitor.capacitance");
    expect(paths).toContain("fluid1d.left.blocking_capacitor.initial_bias_v");
    expect(paths).not.toContain("fluid1d.right.blocking_capacitor.capacitance");
  });
});

describe("self-bias results", () => {
  it("names circuit electrodes and draws the measured or shifted electrode potential", () => {
    const p = newProject();
    expect(circuitLabel("left", null, t)).toBe("左の電極");
    expect(circuitLabel("rf", p, t)).toBe("rf");
    expect(circuitLabel("edge0+edge2", p, t)).toContain("+");
    const src: RfElectrode = { label: "L", dc: 10, rf: [{ amplitude: 100, freq_hz: 13.56e6, phase_deg: 0 }], waveforms: [] };
    const measured = attachCircuit(src, "left", { result: circuit() }, t);
    expect(measured.measured?.v.length).toBe(4);
    const T = 1 / 13.56e6;
    expect(electrodeVoltage(measured, T / 8)).toBeCloseTo((-39.7 + 46) / 2, 9);
    const shifted = attachCircuit(src, "left", { frame: [{ label: "left", v_e: -50, v_dc: -40 }] }, t);
    expect(shifted.shift).toBe(-50);
    expect(electrodeVoltage(shifted, T / 4)).toBeCloseTo(10 + 100 - 50, 9);
    expect(attachCircuit(src, "right", { result: circuit() }, t)).toBe(src);
  });

  it("matches capacitor electrodes of a multi-edge boundary to each edge of the RF monitor", () => {
    const p = newProject();
    p.geometry.boundaries = [{ edges: [1, 3], type: "dirichlet", voltage: 0, voltage_rf: { amplitude: 50, freq_hz: 1e7 } }];
    const es = electrodes2d(p, t, { result: circuit("edge1+edge3", { capacitance: "F/m", charge: "C/m", current: "A/m" }) });
    expect(es.length).toBe(2);
    expect(es.every((e) => e.measured !== undefined)).toBe(true);
  });

  it("lists the last cycle's self-bias, |V1| and the capacitances in the summary", () => {
    const rows = circuitResultRows(circuit(), null, t);
    const text = rows.map(([k, v]) => `${String(k)}=${String(v)}`).join("\n");
    expect(text).toContain("自己バイアス V_dc (左の電極)=-39.66 V");
    expect(text).toContain("|V1| (左の電極)=85.53 V");
    expect(text).toContain("V_dc / |V1| (左の電極)=-0.464");
    expect(text).toContain("354 pF/m² / 100 nF/m² (C_b/(C_b+C) = 99.65%)");
    expect(text).toContain("-2.65 mA/m²");
  });

  it("charts the self-bias of a fluid 1D result and summarises it on the run page", () => {
    const fluid1d = {
      history: { step: [1, 2], n_e_total: [1, 1], n_i_total: [1, 1], wall_left_e: [0, 1], wall_left_i: [0, 1], wall_right_e: [0, 1], wall_right_i: [0, 1] },
      profiles: null, sheath: null, cycle: null, wall_iedf: null,
      walls: { left: { electron: 1, ion: 2 }, right: { electron: 3, ion: 4 } },
      gen_total: 1, elapsed_s: 1, timing: { poisson: 1, total: 1 }, settings: { gap_m: 0.02, n_cells: 2 },
      circuit: circuit(),
    };
    useJobs.setState({ jobs: { r1: job() }, results: { r1: { run: 1, data: fluid1d } } });
    act(() => useResultsView.getState().setActiveRun("r1"));
    render(<ResultsCharts />);
    expect(screen.getByText("自己バイアス (RF 1 周期ごと)")).toBeTruthy();
    render(<RunSummary job={job()} />);
    expect(screen.getByText("自己バイアス V_dc (左の電極)")).toBeTruthy();
    expect(screen.getByText("-39.66 V")).toBeTruthy();
  });
});
