// サンプルの保存された計算結果 (examples/results/<キー>.json.gz、prompts/135): gzip の展開、サンプルを開いたときの
// 読み込み (「読み込んだ実行」として並ぶ)、読む間に別の文書を開いたら捨てること。

import { gzipSync } from "node:zlib";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { openExample } from "../src/io/documents";
import { exampleResultsUrl, loadExampleResults, readGzipJson } from "../src/io/exampleResults";
import { useJobs } from "../src/jobs/jobsStore";
import { useDocument } from "../src/model/documentStore";
import { newProject } from "../src/model/project";
import { DEFAULT_DISPLAY, useResultsView } from "../src/results/resultsView";

const BUNDLE = {
  version: 1,
  meta: { sample: "ccp_demo", generated: "2026-10-03", es_sim: "1.0.0", runs: { pic: { steps: 2000, periods: 4.8 } } },
  pic: {
    started: { dt: 1e-10, n_steps: 2000, step_offset: 0, warnings: [], mesh: { nodes: [[0, 0], [1, 0], [0, 1]], triangles: [[0, 1, 2]] } },
    frame: null, history: [{ t: 0, n_e: 1 }], fields: null, cycle: null, collectors: [], eedf: [], elapsed_s: 2.3,
  },
};

function gz(obj: unknown): Uint8Array<ArrayBuffer> {
  return new Uint8Array(gzipSync(Buffer.from(JSON.stringify(obj))));
}

const initialJobs = useJobs.getState();

beforeEach(() => {
  useJobs.setState({ ...initialJobs, jobs: {}, results: {}, inputs: {} }, true);
  useResultsView.setState({ activeRun: null, bin: 0, playing: false, playBins: 0, chartPrefs: {}, display: structuredClone(DEFAULT_DISPLAY) });
  useDocument.getState().replace(newProject(), null, { untitledName: "test" });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("saved results of examples", () => {
  it("reads gzip and already-decoded JSON (the dev server sends Content-Encoding: gzip)", async () => {
    expect(await readGzipJson(gz(BUNDLE))).toEqual(BUNDLE);
    expect(await readGzipJson(new TextEncoder().encode(JSON.stringify(BUNDLE)))).toEqual(BUNDLE);
  });

  it("has result files for the fluid and PIC examples only", () => {
    expect(exampleResultsUrl("ccp_demo")).toMatch(/ccp_demo\.json\.gz/);
    expect(exampleResultsUrl("gec_cell")).toMatch(/gec_cell\.json\.gz/);
    expect(exampleResultsUrl("parallel_plates")).toBeUndefined();
  });

  it("adds the saved results as imported runs when an example is opened", async () => {
    const fetchMock = vi.fn(async () => new Response(gz(BUNDLE)));
    vi.stubGlobal("fetch", fetchMock);
    await openExample("ccp_demo");
    expect(fetchMock).toHaveBeenCalledWith(exampleResultsUrl("ccp_demo"));
    const jobs = Object.values(useJobs.getState().jobs);
    expect(jobs).toHaveLength(1);
    expect(jobs[0].kind).toBe("pic");
    expect(jobs[0].imported).toBe(true);
    expect(jobs[0].label).toContain("保存された結果・RF 4.8 周期");
    expect(useResultsView.getState().activeRun).toBe(jobs[0].id);
    expect(useJobs.getState().results[jobs[0].id].data).toEqual(BUNDLE.pic);
  });

  it("drops the results when another document was opened while loading", async () => {
    let release: (r: Response) => void = () => {};
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>((resolve) => (release = resolve))));
    const pending = openExample("ccp_demo");
    await vi.waitFor(() => expect(fetch).toHaveBeenCalled());
    useDocument.getState().replace(newProject(), null, { untitledName: "other" });
    release(new Response(gz(BUNDLE)));
    await pending;
    expect(Object.keys(useJobs.getState().jobs)).toHaveLength(0);
  });

  it("reports a failed load without breaking the opened example", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("missing", { status: 404 })));
    await expect(loadExampleResults("/nowhere.json.gz")).rejects.toThrow("HTTP 404");
    await openExample("ccp_demo");
    expect(useDocument.getState().untitledName).toBe("容量結合プラズマ (PIC)");
    expect(Object.keys(useJobs.getState().jobs)).toHaveLength(0);
  });
});
