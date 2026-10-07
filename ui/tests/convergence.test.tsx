// 時間発展の収束の判定 (prompts/137) の UI: 量の名前と % の表記、数値サマリの行 (結果・実行中)、
// 結果のグラフ・数値サマリ、スタディの設定欄 (流体 1D・2D、PIC 1D・2D)、スイープのケースの列。

import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import i18n from "../src/i18n";
import { handleEvent, useJobs } from "../src/jobs/jobsStore";
import type { JobSummary, SweepConvergence } from "../src/jobs/types";
import { useDocument } from "../src/model/documentStore";
import { newProject } from "../src/model/project";
import { useSelection } from "../src/model/selection";
import { SettingsPanel } from "../src/pages/SettingsPanel";
import { SweepCases } from "../src/pages/widgets/SweepCases";
import { isolatedPoints } from "../src/plots/LineChart";
import { usePrefs } from "../src/prefs/prefs";
import { convergenceFrameRows, convergenceResultRows, convPct, convQuantityLabel, convResAt, convWorstAt } from "../src/results/charts/Convergence";
import { ResultsCharts } from "../src/results/ResultsCharts";
import { DEFAULT_DISPLAY, useResultsView } from "../src/results/resultsView";
import { RunSummary } from "../src/results/RunSummary";
import type { ConvergenceFrame, ConvergenceResult } from "../src/results/types";
import { DEFAULT_PIC, defaultFluid1d, defaultFluid2d, defaultPic1d } from "../src/schema/defaults";

const t = i18n.t.bind(i18n) as unknown as Parameters<typeof convQuantityLabel>[2];
const T = 1 / 13.56e6;

function job(p: Partial<JobSummary> = {}): JobSummary {
  return {
    id: "r1", kind: "fluid1d", seq: 1, label: null, state: "done", stopping: false, created: 1000, started_at: 1000,
    finished_at: 1001, elapsed_s: 1, runs: 1, progress: null, error: null, warnings: [], can_continue: true,
    has_result: true, queue_position: null, options: {}, ...p,
  };
}

const W = [null, null, null, null, null, null];

/** 10 周期を判定して 6〜9 周期目で合格、hold 3 回で 9 周期目 (period 8) に収束した流体 1D の結果 */
function converged(p: Partial<ConvergenceResult> = {}): ConvergenceResult {
  const n = 10;
  return {
    tol: 1e-3, basis: "rf", rf_periods: 1, period_s: T, hold: 3, stop: false, max_window: 20, blocks: [64],
    t: Array.from({ length: n }, (_, i) => (i + 1) * T),
    step: Array.from({ length: n }, (_, i) => (i + 1) * 1000),
    period: Array.from({ length: n }, (_, i) => i),
    window: [...W, 1, 1, 1, 1],
    status: ["warming", "warming", "warming", "warming", "warming", "warming", "pass", "pass", "pass", "pass"],
    d: { phi: [...W, 5e-4, 3e-4, 2e-4, 1e-4], n_e: [...W, 8e-4, 6e-4, 4e-4, 3e-4], N_e: [...W, 1e-4, 1e-4, 5e-5, 5e-5], N_i: [...W, 1e-4, 1e-4, 5e-5, 5e-5] },
    r: { phi: [...W, 4e-4, 2e-4, 1e-4, 5e-5], n_e: [...W, 9e-4, 7e-4, 5e-4, 2e-4], N_e: [...W, 0, 0, 0, 0], N_i: [...W, 0, 0, 0, 0] },
    noise: { phi: [...W, 0, 0, 0, 0], n_e: [...W, 0, 0, 0, 0], N_e: [...W, 0, 0, 0, 0], N_i: [...W, 0, 0, 0, 0] },
    r_kind: { phi: [...W, "decay", "decay", "decay", "decay"], n_e: [...W, "decay", "decay", "decay", "decay"], N_e: [...W, "none", "none", "none", "none"], N_i: [...W, "none", "none", "none", "none"] },
    converged: true, converged_t: 9 * T, converged_step: 9000, converged_period: 8, now_passing: true, stopped: false,
    ...p,
  };
}

/** 自己バイアスがまだ動いていて未収束の PIC の結果 (V_dc の残りは傾きからの見積もり) */
function drifting(): ConvergenceResult {
  const c = converged({ tol: 1e-2, converged: false, converged_t: null, converged_step: null, converged_period: null, now_passing: false });
  c.status = [...c.status.slice(0, 6), "fail", "fail", "fail", "fail"];
  c.d["V_dc:rf"] = [...W, 6e-3, 5e-3, 4e-3, 4e-3];
  c.r["V_dc:rf"] = [...W, null, 0.5, 0.4, 0.35];
  c.noise["V_dc:rf"] = [...W, 1e-3, 1e-3, 1e-3, 1e-3];
  c.r_kind!["V_dc:rf"] = [...W, "trend", "trend", "trend", "trend"];
  return c;
}

/** PIC: 変化も残りも閾値以下だが、遅い傾きを見分けるには周期が足りない (分解能、CV-d) */
function resolving(): ConvergenceResult {
  const c = converged({ tol: 1e-2, converged: false, converged_t: null, converged_step: null, converged_period: null, now_passing: false });
  c.status = [...c.status.slice(0, 6), "resolving", "resolving", "resolving", "resolving"];
  c.res = { phi: [...W, 0.03, 0.025, 0.02, 0.015], n_e: [...W, 0.01, 0.009, 0.008, 0.007], N_e: [...W, 0.002, 0.002, 0.002, 0.002], N_i: [...W, 0.002, 0.002, 0.002, 0.002] };
  return c;
}

const text =(rows: [unknown, unknown][]) => rows.map(([k, v]) => `${String(k)}=${String(v)}`).join("\n");

const initialJobs = useJobs.getState();

beforeEach(() => {
  useJobs.setState({ ...initialJobs, jobs: {}, started: {}, startedFull: {}, frames: {}, liveMesh: {}, liveHistory: {}, cases: {}, results: {}, inputs: {}, runSince: {}, watching: [] }, true);
  useResultsView.setState({ activeRun: null, bin: 0, playing: false, playBins: 0, chartPrefs: {}, display: structuredClone(DEFAULT_DISPLAY) });
  useDocument.getState().replace(newProject(), null, { untitledName: "test" });
  usePrefs.setState({ showAdvanced: false, lengthUnit: "mm" });
});

describe("labels and numbers", () => {
  it("names the judged quantities, including each capacitor electrode's self-bias", () => {
    expect(convQuantityLabel("phi", null, t)).toBe("φ");
    expect(convQuantityLabel("N_e", null, t)).toBe("電子の総数");
    expect(convQuantityLabel("N_i", null, t)).toBe("イオンの総数");
    expect(convQuantityLabel("V_dc:left", null, t)).toBe("V_dc (左の電極)");
    expect(convQuantityLabel("V_dc:rf", newProject(), t)).toBe("V_dc (rf)");
    expect(convQuantityLabel("other", null, t)).toBe("other");
  });

  it("writes relative values in percent, tiny ones in exponent form, and infinity as ∞", () => {
    expect(convPct(1e-3)).toBe("0.10%");
    expect(convPct(0.35)).toBe("35%");
    expect(convPct(4.0)).toBe("400%");
    expect(convPct(0.996)).toBe("100%");
    expect(convPct(3e-4)).toBe("0.030%");
    expect(convPct(1e-6)).toBe("1.0e-4%");
    expect(convPct(0)).toBe("0%");
    expect(convPct(null)).toBe("-");
    expect(convPct(null, true)).toBe("∞");
    expect(convPct(Infinity, true)).toBe("∞");
  });

  it("draws a judged period between unjudged ones as a point (a lone value makes no line)", () => {
    expect(isolatedPoints([null, null, 0.1])).toEqual([2]);
    expect(isolatedPoints([0.1, null, 0.2, 0.3, null, 0.4])).toEqual([0, 5]);
    expect(isolatedPoints([0.1, 0.2])).toBeNull();
    expect(isolatedPoints([null, null])).toBeNull();
  });

  it("finds the quantity farthest from convergence and treats an unknown remaining change as infinite", () => {
    expect(convWorstAt(converged(), 9)).toEqual({ name: "n_e", d: 3e-4, r: 2e-4 });
    expect(convWorstAt(converged(), 0)).toBeNull();
    expect(convWorstAt(drifting(), 6)).toEqual({ name: "V_dc:rf", d: 6e-3, r: null });
  });
});

describe("summary rows", () => {
  it("shows when the run converged, the farthest quantity, and the settings", () => {
    const s = text(convergenceResultRows(converged(), null, t));
    expect(s).toContain("状態=収束 (9 周期目、t = 664 ns)");
    expect(s).toContain("いちばん遠い量=n_e: 変化 0.030%、残り 0.020% (減り方から外挿)");
    expect(s).toContain("判定した周期=10");
    expect(s).toContain("閾値 / 判定の周期 / 連続回数=0.10% / RF 1 周期 (73.7 ns) / 3 回");
    expect(s).not.toContain("今の判定");
    expect(s).not.toContain("収束で止めた");
  });

  it("reports a run stopped at convergence and a converged run that fails the latest check", () => {
    const stopped = text(convergenceResultRows(converged({ stopped: true, stop: true }), null, t));
    expect(stopped).toContain("収束で止めた=9000 ステップで収束、そこから平均区間を取って終了");
    const c = converged({ now_passing: false });
    c.status[9] = "fail";
    expect(text(convergenceResultRows(c, null, t))).toContain("今の判定=未収束");
  });

  it("shows an unconverged run with the drifting self-bias and its trend estimate", () => {
    const s = text(convergenceResultRows(drifting(), newProject(), t));
    expect(s).toContain("状態=未収束");
    expect(s).toContain("いちばん遠い量=V_dc (rf): 変化 0.40%、残り 35% (傾きが続くとしてこれまでの長さの 10 倍)");
    expect(s).toContain("1.0% / RF 1 周期");
  });

  it("shows the coarsest resolution while a PIC run waits to resolve slow drifts", () => {
    const s = text(convergenceResultRows(resolving(), null, t));
    expect(s).toContain("状態=判定中 (遅い傾きを見分けるにはまだ周期が足りない)");
    expect(s).toContain("いちばん遠い量=n_e: 変化 0.030%、残り 0.020% (減り方から外挿)");
    expect(s).toContain("分解能 (見逃しうる変化)=φ: 1.5% (後ろ半分の長さで)");
    expect(convResAt(resolving(), 9)).toEqual({ name: "phi", v: 0.015 });
    expect(convResAt(resolving(), 0)).toBeNull();
    // 流体 (分解能が無い) の結果には出さない
    expect(text(convergenceResultRows(converged(), null, t))).not.toContain("分解能");
    // PIC の減っていると言えない傾きは、調べた後ろ半分の長さだけ続くとして (span)
    const drift = resolving();
    drift.status[9] = "fail";
    drift.r.n_e[9] = 0.04;
    drift.r_kind!.n_e[9] = "span";
    expect(text(convergenceResultRows(drift, null, t))).toContain("n_e: 変化 0.030%、残り 4.0% (傾きが調べた後ろ半分の長さだけ続くとして)");
  });

  it("omits the farthest quantity while warming up and marks results without the kind of estimate", () => {
    const c = converged({ converged: false, converged_t: null, converged_step: null, converged_period: null, now_passing: false });
    c.status = c.status.slice(0, 4);
    const s = text(convergenceResultRows(c, null, t));
    expect(s).toContain("状態=判定中 (周期がまだ足りない)");
    expect(s).not.toContain("いちばん遠い量");
    const old = converged();
    delete old.r_kind;
    expect(text(convergenceResultRows(old, null, t))).toContain("残り 0.020% (-)");
  });

  it("shows the live state, the farthest quantity and the time of convergence while running", () => {
    const f: ConvergenceFrame = { status: "warming", window: null, worst: null, worst_name: null, tol: 0.01, checks: 3, converged: false, converged_t: null, now_passing: false };
    expect(text(convergenceFrameRows(f, null, t))).toBe("状態=判定中 (周期がまだ足りない)\n判定した周期=3");
    const fail = text(convergenceFrameRows({ ...f, status: "fail", window: 2, worst: 5e-3, worst_name: "phi", worst_kind: "decay", checks: 30 }, null, t));
    expect(fail).toContain("状態=未収束");
    expect(fail).toContain("いちばん遠い量=φ: max(変化, 残り) 0.50%");
    expect(text(convergenceFrameRows({ ...f, status: "fail", worst_name: "V_dc:left", checks: 30 }, null, t))).toContain("V_dc (左の電極): max(変化, 残り) ∞");
    const done = text(convergenceFrameRows({ ...f, status: "pass", converged: true, converged_t: 9 * T, now_passing: true, checks: 10 }, null, t));
    expect(done).toContain("状態=収束 (t = 664 ns)");
    const live = text(convergenceFrameRows({ ...f, status: "resolving", worst: 4e-3, worst_name: "n_e", res: 0.02, res_name: "V_dc:left", checks: 80 }, null, t));
    expect(live).toContain("状態=判定中 (遅い傾きを見分けるにはまだ周期が足りない)");
    expect(live).toContain("分解能 (見逃しうる変化)=V_dc (左の電極): 2.0% (後ろ半分の長さで)");
  });
});

describe("results", () => {
  const fluid1d = (convergence: ConvergenceResult | null) => ({
    history: { step: [1, 2], n_e_total: [1, 1], n_i_total: [1, 1], wall_left_e: [0, 1], wall_left_i: [0, 1], wall_right_e: [0, 1], wall_right_i: [0, 1] },
    profiles: null, sheath: null, cycle: null, wall_iedf: null,
    walls: { left: { electron: 1, ion: 2 }, right: { electron: 3, ion: 4 } },
    gen_total: 1, elapsed_s: 1, timing: { poisson: 1, total: 1 }, settings: { gap_m: 0.02, n_cells: 2 },
    convergence,
  });

  it("charts and summarises the judgement of a fluid 1D result", () => {
    useJobs.setState({ jobs: { r1: job() }, results: { r1: { run: 1, data: fluid1d(converged()) } } });
    act(() => useResultsView.getState().setActiveRun("r1"));
    render(<ResultsCharts />);
    expect(screen.getByText("収束の判定 (判定の周期ごと)")).toBeTruthy();
    render(<RunSummary job={job()} />);
    expect(screen.getByText("収束の判定")).toBeTruthy();
    expect(screen.getByText("収束 (9 周期目、t = 664 ns)")).toBeTruthy();
  });

  it("leaves out the chart and the summary for results without a judgement", () => {
    useJobs.setState({ jobs: { r1: job() }, results: { r1: { run: 1, data: fluid1d(null) } } });
    act(() => useResultsView.getState().setActiveRun("r1"));
    render(<ResultsCharts />);
    render(<RunSummary job={job()} />);
    expect(screen.queryByText("収束の判定 (判定の周期ごと)")).toBeNull();
    expect(screen.queryByText("収束の判定")).toBeNull();
  });

  it("shows the live judgement of a running fluid 1D job", () => {
    const running = job({ state: "running", has_result: false });
    const frame = { step: 10, t: 1e-7, phi: [0, 0], n_e: [1, 1], n_i: [1, 1], counts: {}, elapsed_s: 1, convergence: { status: "noisy", window: null, worst: null, tol: 1e-3, checks: 8, converged: false, converged_t: null, now_passing: false } };
    useJobs.setState({ jobs: { r1: running }, frames: { r1: frame } });
    render(<RunSummary job={running} />);
    expect(screen.getByText("揺れが大きく比べられない (走り始めの速い変化か雑音。続くなら閾値か判定の周期を大きく)")).toBeTruthy();
  });

  it("charts and summarises the judgement of a PIC 2D result with the capacitor electrode's name", () => {
    const p = newProject();
    p.geometry.regions.push({ id: "rf", type: "conductor", polygon: [[0, 0], [0.01, 0], [0.01, 0.01]], voltage: 0, voltage_rf: { amplitude: 100, freq_hz: 13.56e6 }, blocking_capacitor: { capacitance: 5e-9 } });
    p.pic = { n_steps: 10 };
    const mesh = { nodes: [[0, 0], [1, 0], [0, 1]] as [number, number][], triangles: [[0, 1, 2]] as [number, number, number][] };
    const pic = {
      started: { dt: 1e-11, n_steps: 10, step_offset: 0, warnings: [], mesh },
      frame: null, history: [], fields: null, cycle: null, collectors: [], eedf: [], elapsed_s: 1,
      convergence: drifting(),
    };
    const pj = job({ kind: "pic" });
    useJobs.setState({ jobs: { r1: pj }, results: { r1: { run: 1, data: pic } }, inputs: { r1: { project: p, docSerial: -1 } } });
    act(() => useResultsView.getState().setActiveRun("r1"));
    render(<ResultsCharts />);
    expect(screen.getByText("収束の判定 (判定の周期ごと)")).toBeTruthy();
    render(<RunSummary job={pj} />);
    expect(screen.getByText("V_dc (rf): 変化 0.40%、残り 35% (傾きが続くとしてこれまでの長さの 10 倍)")).toBeTruthy();
  });
});

describe("settings", () => {
  const studies: [string, string][] = [
    ["study:pic", "自動 (0.01 = 1%)"],
    ["study:pic1d", "自動 (0.01 = 1%)"],
    ["study:fluid1d", "自動 (0.001 = 0.1%)"],
    ["study:fluid2d", "自動 (0.001 = 0.1%)"],
  ];

  it("shows the judgement settings on every time-dependent study with the automatic tolerance", () => {
    const p = newProject();
    Object.assign(p, { pic: structuredClone(DEFAULT_PIC), pic1d: defaultPic1d(), fluid1d: defaultFluid1d(), fluid2d: defaultFluid2d() });
    useDocument.getState().replace(p, null);
    for (const [node, placeholder] of studies) {
      act(() => useSelection.setState({ activeNode: node }));
      const { unmount } = render(<SettingsPanel />);
      expect(screen.getByText("収束の判定")).toBeTruthy();
      expect(screen.getByPlaceholderText(placeholder)).toBeTruthy();
      expect(screen.getByPlaceholderText("自動 (frame_every の 10 倍)")).toBeTruthy();
      expect(screen.queryByLabelText("窓の最大 (周期)")).toBeNull();
      unmount();
    }
  });

  it("writes the tolerance and the stop switch into the study and shows the advanced window limit", () => {
    useDocument.getState().update("init", (d) => void (d.pic = structuredClone(DEFAULT_PIC)));
    usePrefs.setState({ showAdvanced: true });
    useSelection.setState({ activeNode: "study:pic" });
    render(<SettingsPanel />);
    const tol = screen.getByLabelText("閾値");
    fireEvent.focus(tol);
    fireEvent.change(tol, { target: { value: "0.005" } });
    fireEvent.blur(tol);
    fireEvent.click(screen.getByLabelText("収束したら止める"));
    const conv = (useDocument.getState().project.pic as { convergence: Record<string, unknown> }).convergence;
    expect(conv.tol).toBe(0.005);
    expect(conv.stop).toBe(true);
    expect(screen.getByLabelText("窓の最大 (周期)")).toBeTruthy();
  });
});

describe("sweep cases", () => {
  const conv = (p: Partial<SweepConvergence>): SweepConvergence => ({ converged: false, converged_period: null, converged_t: null, status: "fail", checks: 30, stopped: false, ...p });

  it("lists the period each case converged at, or its last state", () => {
    handleEvent({ type: "job", job: job({ id: "s1", kind: "sweep", state: "running", runs: 1, options: { values: [1, 2, 3] } }) });
    handleEvent({ type: "case", id: "s1", case: 0, value: 1, ok: true, convergence: conv({ converged: true, converged_period: 9, converged_t: 10 * T, status: "pass", checks: 12 }) });
    handleEvent({ type: "case", id: "s1", case: 1, value: 2, ok: true, convergence: conv({}) });
    handleEvent({ type: "case", id: "s1", case: 2, value: 3, ok: true, convergence: conv({ status: "warming", checks: 4 }) });
    handleEvent({ type: "job", job: job({ id: "s1", kind: "sweep", state: "running", runs: 1, options: { values: [1, 2, 3, 4] } }) });
    handleEvent({ type: "case", id: "s1", case: 3, value: 4, ok: true, convergence: conv({ status: "resolving", checks: 90 }) });
    render(<SweepCases />);
    expect(screen.getByText("収束")).toBeTruthy();
    expect(screen.getByText("10 周期目")).toBeTruthy();
    expect(screen.getByText("未収束")).toBeTruthy();
    expect(screen.getByText("判定中 (周期がまだ足りない)")).toBeTruthy();
    expect(screen.getByText("判定中 (遅い傾きを見分けるにはまだ周期が足りない)")).toBeTruthy();
  });

  it("has no convergence column for cases without a judgement", () => {
    handleEvent({ type: "job", job: job({ id: "s1", kind: "sweep", state: "running", runs: 1, options: { values: [1] } }) });
    handleEvent({ type: "case", id: "s1", case: 0, value: 1, ok: true, convergence: null });
    render(<SweepCases />);
    expect(screen.queryByText("収束")).toBeNull();
  });
});
