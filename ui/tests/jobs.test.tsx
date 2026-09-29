import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useMessages } from "../src/app/messages";
import { useConnection } from "../src/backend/connection";
import { t } from "../src/i18n";
import { startJobEffects } from "../src/jobs/effects";
import { handleEvent, jobElapsed, jobName, onJobDone, sortedJobs, useJobs } from "../src/jobs/jobsStore";
import type { JobSummary } from "../src/jobs/types";
import { useDocument } from "../src/model/documentStore";
import { newProject } from "../src/model/project";
import { RunControls } from "../src/pages/widgets/RunControls";
import { SweepCases } from "../src/pages/widgets/SweepCases";
import { buildTree } from "../src/tree/treeModel";

function job(p: Partial<JobSummary> = {}): JobSummary {
  return {
    id: "j1",
    kind: "pic1d",
    seq: 1,
    label: null,
    state: "queued",
    stopping: false,
    created: 1000,
    started_at: null,
    finished_at: null,
    elapsed_s: 0,
    runs: 0,
    progress: null,
    error: null,
    warnings: [],
    can_continue: false,
    has_result: false,
    queue_position: 0,
    options: {},
    ...p,
  };
}

const initial = useJobs.getState();

beforeEach(() => {
  useJobs.setState({ ...initial, jobs: {}, started: {}, startedFull: {}, frames: {}, cases: {}, results: {}, inputs: {}, runSince: {}, watching: [] }, true);
  useMessages.getState().clear();
  useDocument.getState().replace(newProject(), null, { untitledName: "test" });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("jobs store", () => {
  it("mirrors hello/job/progress/started/case/removed events", () => {
    handleEvent({ type: "hello", jobs: [job({ id: "a", seq: 1 }), job({ id: "b", seq: 2, state: "running", runs: 1, elapsed_s: 2 })], limits: { max_running: 4, default_per_kind: 2, per_kind: {} }, kinds: [], instance: "x" });
    expect(Object.keys(useJobs.getState().jobs)).toEqual(["a", "b"]);
    // 実行中のジョブの経過時間は届いた時点の値から数える
    const since = useJobs.getState().runSince.b;
    expect(jobElapsed(useJobs.getState().jobs.b, since, since + 3000)).toBeCloseTo(3);
    handleEvent({ type: "progress", id: "b", step: 50, n_steps: 100, step_offset: 0, fraction: 0.5 });
    expect(useJobs.getState().jobs.b.progress?.fraction).toBe(0.5);
    handleEvent({ type: "started", id: "b", run: 1, full: false, n_steps: 100, dt: 1e-10 });
    handleEvent({ type: "started", id: "b", run: 1, full: true, n_steps: 100, mesh: { nodes: [] } });
    expect(useJobs.getState().started.b).toMatchObject({ n_steps: 100, dt: 1e-10 });
    expect(useJobs.getState().startedFull.b).toHaveProperty("mesh");
    handleEvent({ type: "case_progress", id: "b", case: 1, step: 5, n_steps: 10 });
    handleEvent({ type: "case", id: "b", case: 1, value: 2, ok: true });
    expect(useJobs.getState().cases.b[1]).toEqual({ step: 5, n_steps: 10, value: 2, ok: true });
    handleEvent({ type: "removed", id: "b" });
    const s = useJobs.getState();
    expect(s.jobs.b).toBeUndefined();
    expect(s.started.b).toBeUndefined();
    expect(s.cases.b).toBeUndefined();
    expect(s.runSince.b).toBeUndefined();
    expect(sortedJobs(s.jobs).map((j) => j.id)).toEqual(["a"]);
    expect(jobName(job({ kind: "fluid2d", seq: 3 }))).toBe("流体 2D #3");
  });

  it("drops a cached result when a new run starts and notifies on done", () => {
    handleEvent({ type: "job", job: job({ state: "done", runs: 1, has_result: true }) });
    useJobs.setState({ results: { j1: { run: 1, data: { x: 1 } } } });
    handleEvent({ type: "job", job: job({ state: "running", runs: 2 }) });
    expect(useJobs.getState().results.j1).toBeUndefined();
    const seen: string[] = [];
    const off = onJobDone((j) => seen.push(`${j.id}:${j.state}:${j.has_result}`));
    handleEvent({ type: "job", job: job({ state: "done", runs: 2, has_result: true, elapsed_s: 1.5 }) });
    handleEvent({ type: "done", id: "j1", state: "done", elapsed_s: 1.5 });
    off();
    expect(seen).toEqual(["j1:done:true"]);
    expect(useMessages.getState().items.at(-1)?.text).toBe("PIC 1D #1 が完了しました (1.5 s)");
    handleEvent({ type: "error", id: "j1", detail: "発散" });
    expect(useMessages.getState().items.at(-1)).toMatchObject({ level: "error", text: "PIC 1D #1 が失敗しました: 発散" });
  });

  it("puts a generated Boltzmann table into the document that asked for it", async () => {
    startJobEffects();
    useDocument.getState().update("fluid", (d) => void (d.fluid1d = { n_steps: 10, boltz_table: null }));
    const table = { en_td: [1, 2], mean_energy_ev: [1, 2], source_hash: "h" };
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ table }), { status: 200 })));
    useJobs.setState({ inputs: { j1: { project: null, docSerial: useDocument.getState().docSerial } } });
    handleEvent({ type: "job", job: job({ kind: "boltz", state: "done", runs: 1, has_result: true, options: { module: "fluid1d" } }) });
    handleEvent({ type: "done", id: "j1", state: "done", elapsed_s: 1 });
    await waitFor(() => expect((useDocument.getState().project.fluid1d as { boltz_table: unknown }).boltz_table).toEqual(table));
    expect(useDocument.getState().past.at(-1)?.label).toBe(t("jobs.boltzApplyLabel"));
  });
});

describe("tree", () => {
  it("shows run state on study nodes and lists runs under results", () => {
    const jobs = [
      job({ id: "a", kind: "pic", seq: 1, state: "done", created: 1, has_result: true }),
      job({ id: "b", kind: "pic", seq: 2, state: "running", created: 2, progress: { step: 30, n_steps: 100, step_offset: 0, fraction: 0.3 } }),
      job({ id: "c", kind: "dsmc", seq: 1, state: "error", created: 3 }),
    ];
    const root = buildTree(newProject(), t, "mm", "doc", jobs);
    const studies = root.children!.find((n) => n.id === "studies")!.children!;
    expect(studies.find((n) => n.id === "study:pic")!.badge).toEqual({ text: "実行中 30%", tone: "run" });
    expect(studies.find((n) => n.id === "study:dsmc")!.badge).toEqual({ text: "エラー", tone: "error" });
    expect(studies.find((n) => n.id === "study:pic1d")!.badge?.tone).toBe("muted");
    const results = root.children!.find((n) => n.id === "results")!;
    expect(results.children!.map((n) => `${n.id} ${n.label} ${n.badge?.text}`)).toEqual([
      "result:c DSMC #1 エラー",
      "result:b PIC #2 実行中 30%",
      "result:a PIC #1 完了",
    ]);
  });
});

describe("run controls", () => {
  it("submits a job and lists runs with stop / continue / delete", async () => {
    useConnection.setState({ status: "disconnected" });
    const { rerender } = render(<RunControls kind="pic1d" />);
    expect((screen.getByRole("button", { name: "実行" }) as HTMLButtonElement).disabled).toBe(true);
    useConnection.setState({ status: "connected", info: { version: "0.2", gpu: false, numba: true, jobs: true } });
    const calls: [string, string][] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        calls.push([init?.method ?? "GET", url.replace(/^http:\/\/127\.0\.0\.1:\d+/, "")]);
        if (url.endsWith("/v2/jobs")) return new Response(JSON.stringify(job({ id: "n1", seq: 7 })), { status: 201 });
        return new Response(JSON.stringify(job({ id: "n1", seq: 7, state: "running" })), { status: 200 });
      }),
    );
    rerender(<RunControls kind="pic1d" />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "実行" }));
    });
    expect(calls[0]).toEqual(["POST", "/v2/jobs"]);
    expect(await screen.findByText("PIC 1D #7")).toBeTruthy();
    act(() => handleEvent({ type: "job", job: job({ id: "n1", seq: 7, state: "running", runs: 1, progress: { step: 10, n_steps: 40, step_offset: 0, fraction: 0.25 } }) }));
    expect(screen.getByText("実行中 25%")).toBeTruthy();
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "停止" }));
    });
    expect(calls.at(-1)).toEqual(["POST", "/v2/jobs/n1/stop"]);
    act(() => handleEvent({ type: "job", job: job({ id: "n1", seq: 7, state: "stopped", runs: 1, can_continue: true, has_result: true }) }));
    expect(screen.getByRole("button", { name: "続きを実行" })).toBeTruthy();
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "削除" }));
    });
    expect(calls.at(-1)).toEqual(["DELETE", "/v2/jobs/n1"]);
    expect(screen.queryByText("PIC 1D #7")).toBeNull();
  });

  it("loads a finished sweep case into the document (undoable)", async () => {
    handleEvent({ type: "job", job: job({ id: "s1", kind: "sweep", state: "running", runs: 1, options: { values: [1, 2] } }) });
    handleEvent({ type: "case", id: "s1", case: 1, value: 2, ok: true });
    const caseProject = { ...newProject(), mesh: { size: 0.002 }, results: { pic1d: {} } };
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(caseProject), { status: 200 })));
    render(<SweepCases />);
    expect(screen.getByText("完了")).toBeTruthy();
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "読み込む" }));
    });
    await waitFor(() => expect(useDocument.getState().project.mesh.size).toBe(0.002));
    expect(useDocument.getState().project).not.toHaveProperty("results");
    act(() => useDocument.getState().undo());
    expect(useDocument.getState().project.mesh.size).toBe(0.004);
  });
});
