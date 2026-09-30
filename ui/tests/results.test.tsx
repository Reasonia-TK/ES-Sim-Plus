import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { PlaybackBar } from "../src/graphics/PlaybackBar";
import { runScene } from "../src/graphics/runScene";
import { staticScene } from "../src/graphics/staticScene";
import { handleEvent, removeRun, useJobs } from "../src/jobs/jobsStore";
import type { JobSummary } from "../src/jobs/types";
import { useDocument } from "../src/model/documentStore";
import { newProject, type Point } from "../src/model/project";
import { extent, niceTicks, tickText } from "../src/plots/axes";
import { columns, csvText } from "../src/plots/csv";
import { buildResultsBundle, importResultsBundle } from "../src/results/bundle";
import { ampAt, deriveXGrid, interpLinear, mergeSeriesX, rowsRange, spectrumPeaks, timingRows } from "../src/results/charts/common";
import { energyRange, histogram, histogram2d } from "../src/results/charts/Charts2d";
import { eepf } from "../src/results/charts/Distributions";
import { ResultsCharts } from "../src/results/ResultsCharts";
import { DEFAULT_DISPLAY, useResultsView } from "../src/results/resultsView";
import { RunSummary } from "../src/results/RunSummary";
import { brinkmannSheathEdge, sheathOnLine } from "../src/results/sheath";
import { tlValuesFromPic1d } from "../src/pages/widgets/TlImport";
import { useStatic } from "../src/results/staticResults";

function job(p: Partial<JobSummary> = {}): JobSummary {
  return {
    id: "r1",
    kind: "pic1d",
    seq: 1,
    label: null,
    state: "done",
    stopping: false,
    created: 1000,
    started_at: 1000,
    finished_at: 1001,
    elapsed_s: 1,
    runs: 1,
    progress: null,
    error: null,
    warnings: [],
    can_continue: true,
    has_result: true,
    queue_position: null,
    options: {},
    ...p,
  };
}

/** 0〜1 × 0〜0.1 の帯 (x 方向に 10 等分、20 個の三角形) */
function stripMesh(): { nodes: Point[]; triangles: [number, number, number][] } {
  const nodes: Point[] = [];
  for (let i = 0; i <= 10; i++) nodes.push([i / 10, 0], [i / 10, 0.1]);
  const triangles: [number, number, number][] = [];
  for (let i = 0; i < 10; i++) {
    const a = 2 * i;
    triangles.push([a, a + 2, a + 3], [a, a + 3, a + 1]);
  }
  return { nodes, triangles };
}

/** n_i = 1、n_e は x < 0.3 で 0、ほかで 1 (シース端は 0.25) */
function stepDensities(nodes: Point[]): { ne: number[]; ni: number[] } {
  return { ne: nodes.map(([x]) => (x < 0.3 - 1e-9 ? 0 : 1)), ni: nodes.map(() => 1) };
}

const initialJobs = useJobs.getState();

beforeEach(() => {
  useJobs.setState({ ...initialJobs, jobs: {}, started: {}, startedFull: {}, frames: {}, liveMesh: {}, liveHistory: {}, cases: {}, results: {}, inputs: {}, runSince: {}, watching: [] }, true);
  useResultsView.setState({ activeRun: null, bin: 0, playing: false, playBins: 0, chartPrefs: {}, display: structuredClone(DEFAULT_DISPLAY) });
  useDocument.getState().replace(newProject(), null, { untitledName: "test" });
  useStatic.getState().clear();
});

describe("result math", () => {
  it("finds the Brinkmann sheath edge on arrays and along a line in a mesh", () => {
    const dist = Array.from({ length: 11 }, (_, i) => i / 10);
    const ni = dist.map(() => 1);
    const ne = dist.map((x) => (x < 0.3 - 1e-9 ? 0 : 1));
    expect(brinkmannSheathEdge(dist, ne, ni)).toBeCloseTo(0.25, 9);
    // プラズマが無い (n_e = 0) と根が無い
    expect(brinkmannSheathEdge(dist, dist.map(() => 0), ni)).toBeNull();
    const m = stripMesh();
    const { ne: ne2, ni: ni2 } = stepDensities(m.nodes);
    const s = sheathOnLine({ nodes: m.nodes, triangles: m.triangles }, [0, 0.05], [1, 0.05], ne2, ni2);
    expect(s).not.toBeNull();
    expect(s!).toBeCloseTo(0.25, 2);
  });

  it("bins weighted histograms like v1 (edges pile into the end bins)", () => {
    expect(histogram([0, 0.5, 1, 7], [1, 2, 3, 4], 2, [0, 1]).counts).toEqual([1, 9]);
    expect(histogram([-1, 0.2], undefined, 2, [0, 1]).counts).toEqual([2, 0]);
    const h2 = histogram2d([-90, 89, 0], [0, 10, 5], [1, 1, 2], 2, [-90, 90], [0, 10]);
    expect(h2).toEqual([
      [1, 0],
      [0, 3],
    ]);
    expect(energyRange([])).toEqual([0, 1]);
    expect(energyRange([5, 5])).toEqual([5, 6]);
    expect(energyRange([3, 1, NaN])).toEqual([1, 3]);
  });

  it("merges series with different x by interpolating inside each series", () => {
    const m = mergeSeriesX([
      { x: [0, 2], y: [0, 2] },
      { x: [1, 3], y: [10, 30] },
    ]);
    expect(m.x).toEqual([0, 1, 2, 3]);
    expect(m.ys[0]).toEqual([0, 1, 2, null]);
    expect(m.ys[1]).toEqual([null, 10, 20, 30]);
  });

  it("orders the timing breakdown and finds spectrum peaks", () => {
    const r = timingRows({ a: 1, b: 3, total: 4 });
    expect(r.rows.map((x) => x.key)).toEqual(["b", "a"]);
    expect(r.rows[0].pct).toBeCloseTo(75);
    expect(r.total).toBe(4);
    expect(timingRows({ a: 1, b: 1 }).total).toBe(2);
    const f = [0, 1, 2, 3, 4, 5, 6];
    const a = [9, 1, 5, 1, 3, 1, 9];
    expect(spectrumPeaks(f, a).map((p) => p.f)).toEqual([2, 4]);
    expect(ampAt(f, a, 3.9)).toBe(3);
  });

  it("builds 1D grids, interpolates and fixes ranges across bins", () => {
    expect(deriveXGrid(0.02, 4)).toEqual([0, 0.005, 0.01, 0.015, 0.02]);
    expect(interpLinear([0, 1, 2], [0, 10, 20], 1.5)).toBe(15);
    expect(interpLinear([0, 1], [3, 4], -1)).toBe(3);
    expect(Number.isNaN(interpLinear([], [], 0))).toBe(true);
    const r = rowsRange([
      [0, 1],
      [2, null],
    ]);
    expect(r![0]).toBeLessThan(0);
    expect(r![1]).toBeGreaterThan(2);
    // 対数では 0 以下を除き、1 つの値なら ±10% に広げる
    const lr = rowsRange([[0, 5]], true)!;
    expect(lr[0]).toBeCloseTo(4.5);
    expect(lr[1]).toBeCloseTo(5.5);
    expect(eepf({ e_centers: [0, 4], f: [1, 2], total_weight: 1 })).toEqual([null, 1]);
  });

  it("formats chart ticks and CSV cells", () => {
    expect(niceTicks(0, 10, 5)).toEqual([0, 2, 4, 6, 8, 10]);
    expect(tickText(3e8)).toBe("3.0e8");
    expect(tickText(12.3456)).toBe("12.35");
    expect(extent([[1, NaN, 3]])).toEqual([1, 3]);
    expect(csvText(["a", "b"], columns([1, 2], [null, NaN]))).toBe("a,b\n1,\n2,\n");
  });
});

describe("run scenes", () => {
  it("offers field and phase-resolved views and marks the sheath edge of evaluation lines", () => {
    const m = stripMesh();
    const { ne, ni } = stepDensities(m.nodes);
    const n = m.nodes.length;
    const result = {
      started: { dt: 1e-11, n_steps: 10, warnings: [], mesh: m },
      frame: null,
      history: [],
      fields: { phi: m.nodes.map(([x]) => x), e_abs: m.triangles.map(() => 1), n_e: ne, n_i: ni, te_ev: new Array(n).fill(1), ion_rate: new Array(n).fill(0), avg_steps: 5 },
      cycle: { bins: 2, period_s: 1e-7, phi: [ne, ni], n_e: [ne, ne], n_i: [ni, ni], particles: { electron: [[], []], ion: [[], []] } },
      collectors: [],
      eedf: [],
    };
    const background = staticScene({ mesh: null, solve: null, latest: null }, "v", newProject(), { meshTitle: () => "", potential: "V", field: "E" });
    const input = {
      job: job({ kind: "pic" }),
      result,
      display: { ...DEFAULT_DISPLAY.pic, mode: "cycle" as const },
      bin: 1,
      sheath: { contour: true, alpha: 0.5 },
      sheathLines: [{ p1: [0, 0.05] as Point, p2: [1, 0.05] as Point }],
      label: (q: string) => q,
      name: "PIC #1",
      modeLabel: (m: string) => m,
      background,
    };
    const s = runScene(input)!;
    expect(s.modes).toEqual(["field", "cycle"]);
    expect(s.mode).toBe("cycle");
    expect(s.playback).toEqual({ bins: 2, periodS: 1e-7 });
    expect(s.field?.values).toBe(ni);
    expect(s.segments.some((g) => g.id === "sheath")).toBe(true);
    const mark = s.points.find((p) => p.id === "sheathS");
    expect(mark).toBeTruthy();
    expect(mark!.xy[0]).toBeCloseTo(0.25, 2);
    // 時間平均では量を選べ、位相分解の再生は無い
    const f = runScene({ ...input, display: { ...DEFAULT_DISPLAY.pic, mode: "field", field: "n_e" } })!;
    expect(f.mode).toBe("field");
    expect(f.quantity).toBe("n_e");
    expect(f.playback).toBeUndefined();
  });
});

describe("jobs store (P6e)", () => {
  it("collects a live history from frames and clears it when a new run starts", () => {
    handleEvent({ type: "job", job: job({ id: "p", kind: "pic", state: "running", runs: 1, has_result: false }) });
    handleEvent({ type: "frame", id: "p", step: 10, t: 1e-9, diag: { t: 1e-9, ke_e: 1, ke_i: 2, fe: 3, n_e: 4, n_i: 5 } });
    handleEvent({ type: "frame", id: "p", step: 20, t: 2e-9, diag: { t: 2e-9, ke_e: 1, ke_i: 2, fe: 3, n_e: 4, n_i: 5 } });
    expect(useJobs.getState().liveHistory.p.map((r) => r.step)).toEqual([10, 20]);
    handleEvent({ type: "job", job: job({ id: "p", kind: "pic", state: "running", runs: 2, has_result: false }) });
    expect(useJobs.getState().liveHistory.p).toBeUndefined();
  });

  it("imports a results file as runs that survive backend events and delete locally", async () => {
    const project = newProject();
    const added = importResultsBundle(
      {
        version: 1,
        pic: { started: { mesh: stripMesh() }, history: { t: [0, 1], ke_e: [1, 2] }, fields: null, cycle: null },
        pic1d: { history: {}, profiles: null, walls: { left: { electron: 0, ion: 0 }, right: { electron: 0, ion: 0 } } },
        solve: null,
      },
      project,
      "saved.json",
    );
    expect(added.map((j) => j.kind)).toEqual(["pic", "pic1d"]);
    const pic = added[0];
    expect(pic.imported).toBe(true);
    expect(pic.label).toContain("saved.json");
    const cached = useJobs.getState().results[pic.id].data as { history: { t: number }[]; collectors: unknown[] };
    expect(cached.history).toEqual([
      { t: 0, ke_e: 1 },
      { t: 1, ke_e: 2 },
    ]);
    expect(cached.collectors).toEqual([]);
    handleEvent({ type: "hello", jobs: [], limits: { max_running: 4, default_per_kind: 2, per_kind: {} }, kinds: [], instance: "x" });
    expect(Object.keys(useJobs.getState().jobs)).toHaveLength(2);
    await removeRun(pic.id);
    expect(useJobs.getState().jobs[pic.id]).toBeUndefined();
  });

  it("saves one result per kind (the shown run first) with the electrostatics result", async () => {
    const r = (id: string, created: number) => job({ id, kind: "tl", created });
    useJobs.setState({
      jobs: { a: r("a", 1), b: r("b", 2), c: job({ id: "c", kind: "pic1d", state: "running", has_result: false }) },
      results: { a: { run: 1, data: { which: "a" } }, b: { run: 1, data: { which: "b" } } },
    });
    let bundle = await buildResultsBundle();
    expect(bundle).toMatchObject({ version: 1, tl: { which: "b" } });
    expect(bundle!.pic1d).toBeUndefined();
    useResultsView.getState().setActiveRun("a");
    bundle = await buildResultsBundle();
    expect(bundle!.tl).toEqual({ which: "a" });
    useJobs.setState({ jobs: {}, results: {} });
    expect(await buildResultsBundle()).toBeNull();
  });
});

describe("result views", () => {
  const pic1d = {
    history: { step: [1, 2], n_e: [10, 9], n_i: [10, 8] },
    profiles: { x: [0, 0.01, 0.02], phi: [0, 5, 0], e: [1, 0, -1], n_e: [0, 1e15, 0], n_i: [1e14, 1e15, 1e14], t_e: [1, 2, 1], ionization: [0, 1, 0], avg_steps: 100 },
    sheath: { left_s: 0.002, right_s: null },
    cycle: { bins: 2, freq_hz: 1e7, phi: [[0, 1, 0], [0, 2, 0]], n_e: [[0, 1, 0], [0, 1, 0]], n_i: [[1, 1, 1], [1, 1, 1]], sheath: { s_left: [0.001, 0.002], s_right: [null, null] } },
    sheath_ts: null,
    sheath_fft: null,
    eedf: [],
    wall_iedf: null,
    walls: { left: { electron: 1, ion: 2 }, right: { electron: 3, ion: 4 } },
    fn: null,
    elapsed_s: 1.5,
    timing: { deposit: 0.5, field: 1, total: 1.5 },
    settings: { gap_m: 0.02, n_cells: 2 },
  };

  it("charts a finished 1D run and offers other 1D runs for comparison", () => {
    useJobs.setState({
      jobs: { r1: job(), r2: job({ id: "r2", kind: "fluid1d", seq: 1 }) },
      results: { r1: { run: 1, data: pic1d } },
    });
    act(() => useResultsView.getState().setActiveRun("r1"));
    render(<ResultsCharts />);
    expect(screen.getByText("時間平均のプロファイル")).toBeTruthy();
    expect(screen.getByText("位相分解")).toBeTruthy();
    expect(screen.getByText("履歴 (マクロ粒子数)")).toBeTruthy();
    expect(screen.getByText("流体 1D #1 と比較")).toBeTruthy();
    expect(screen.getByRole("group", { name: "位相分解の再生" })).toBeTruthy();
  });

  it("summarises a 1D result on the run page", () => {
    useJobs.setState({ jobs: { r1: job() }, results: { r1: { run: 1, data: pic1d } } });
    render(<RunSummary job={job()} />);
    expect(screen.getByText("数値サマリ")).toBeTruthy();
    expect(screen.getByText("ポアソン求解 + 電場")).toBeTruthy();
    expect(screen.getByText("1.000 s (66.7%)")).toBeTruthy();
    expect(screen.getByText(/シース端 \(電極からの距離\)/)).toBeTruthy();
  });

  it("takes n_e and the sheath thickness for the VHF study from a PIC 1D result", () => {
    const v = tlValuesFromPic1d(pic1d as never)!;
    expect(v.n_e).toBe(1e15);
    expect(v.sheath).toBe(0.002);
    expect(tlValuesFromPic1d({ ...pic1d, sheath: { left_s: null, right_s: null } } as never)).toBeNull();
  });

  it("steps the phase bins and pauses when the slider moves", () => {
    useResultsView.setState({ playing: true, bin: 0 });
    render(<PlaybackBar playback={{ bins: 4, periodS: 1e-7 }} />);
    expect(useResultsView.getState().playBins).toBe(4);
    fireEvent.change(screen.getByRole("slider"), { target: { value: "2" } });
    expect(useResultsView.getState().bin).toBe(2);
    expect(useResultsView.getState().playing).toBe(false);
    expect(screen.getByText(/3 \/ 4 · 180.0°/)).toBeTruthy();
  });
});
