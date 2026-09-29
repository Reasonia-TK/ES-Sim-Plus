// ジョブのストア: /v2/events の WebSocket で backend の状態をそのまま映す (最初の hello で全体、以後は差分)。
// 実行ごとの結果は必要になったときに取りに行き、実行 (runs) が変わるまで覚えておく。接続はバックエンドへの
// 接続 (/health) に合わせて張り、切れたら張り直す。

import { create } from "zustand";
import { errorText, logError, logInfo, logWarning } from "../app/messages";
import { useConnection } from "../backend/connection";
import { wsUrl } from "../backend/port";
import { t } from "../i18n";
import { useDocument } from "../model/documentStore";
import type { Project } from "../model/project";
import { continueJob, deleteJob, jobResult, stopJob, submitJob } from "./api";
import { TERMINAL_STATES, type JobEvent, type JobKind, type JobLimits, type JobSummary, type SweepCase } from "./types";

interface JobsState {
  /** /v2/events がつながっている */
  connected: boolean;
  jobs: Record<string, JobSummary>;
  limits: JobLimits | null;
  /** 実行の開始時の情報 (dt・ステップ数・警告。重いメッシュは除く) */
  started: Record<string, Record<string, unknown>>;
  /** watch したジョブの開始時の情報 (表示用メッシュ付き) と最新のフレーム (P6e のライブ表示) */
  startedFull: Record<string, Record<string, unknown>>;
  frames: Record<string, Record<string, unknown>>;
  /** スイープのケースごとの状態 */
  cases: Record<string, Record<number, SweepCase>>;
  /** 取ってきた結果 (runs が変わったら取り直す) */
  results: Record<string, { run: number; data: unknown }>;
  /** この画面から出したジョブの入力と、そのときの文書 (別の文書を開いたら結果を書き戻さない) */
  inputs: Record<string, { project: Project | null; docSerial: number }>;
  watching: string[];
  /** 実行中のジョブの経過時間の起点 (performance.now() の ms、経過 = 今 − 起点) */
  runSince: Record<string, number>;
}

const empty: JobsState = {
  connected: false,
  jobs: {},
  limits: null,
  started: {},
  startedFull: {},
  frames: {},
  cases: {},
  results: {},
  inputs: {},
  watching: [],
  runSince: {},
};

export const useJobs = create<JobsState>()(() => ({ ...empty }));

/** 表示名 (PIC #3) */
export function jobName(job: Pick<JobSummary, "kind" | "seq" | "label">): string {
  return `${job.label ?? t(`jobs.kind.${job.kind}`)} #${job.seq}`;
}

/** 経過時間 [s] (実行中は今までの分を足す) */
export function jobElapsed(job: JobSummary, runSince: number | undefined, now = performance.now()): number {
  return job.state === "running" && runSince !== undefined ? Math.max(job.elapsed_s, (now - runSince) / 1000) : job.elapsed_s;
}

export function isTerminal(job: JobSummary): boolean {
  return TERMINAL_STATES.includes(job.state);
}

/** 新しい順 */
export function sortedJobs(jobs: Record<string, JobSummary>, kind?: JobKind): JobSummary[] {
  return Object.values(jobs)
    .filter((j) => kind === undefined || j.kind === kind)
    .sort((a, b) => b.created - a.created);
}

// ---- 完了のときの処理 (Boltzmann の表を文書へ書くなど) ----

type DoneListener = (job: JobSummary) => void;
const doneListeners = new Set<DoneListener>();

export function onJobDone(fn: DoneListener): () => void {
  doneListeners.add(fn);
  return () => doneListeners.delete(fn);
}

// ---- イベント ----

function without<T>(rec: Record<string, T>, id: string): Record<string, T> {
  if (!(id in rec)) return rec;
  const next = { ...rec };
  delete next[id];
  return next;
}

export function handleEvent(ev: JobEvent): void {
  const set = useJobs.setState;
  switch (ev.type) {
    case "hello": {
      const jobs = Object.fromEntries(ev.jobs.map((j) => [j.id, j]));
      const now = performance.now();
      set((s) => ({
        jobs,
        limits: ev.limits,
        runSince: Object.fromEntries(ev.jobs.filter((j) => j.state === "running").map((j) => [j.id, now - j.elapsed_s * 1000])),
        results: Object.fromEntries(Object.entries(s.results).filter(([id]) => id in jobs)),
        started: Object.fromEntries(Object.entries(s.started).filter(([id]) => id in jobs)),
      }));
      break;
    }
    case "job":
      set((s) => {
        const prev = s.jobs[ev.job.id];
        const results = s.results[ev.job.id] && s.results[ev.job.id].run !== ev.job.runs ? without(s.results, ev.job.id) : s.results;
        // 新しい実行 (続き) が始まったらスイープのケースとフレームを消す
        const restarted = prev && prev.runs !== ev.job.runs;
        const running = ev.job.state === "running";
        const startedNow = running && (!prev || prev.state !== "running" || restarted || !(ev.job.id in s.runSince));
        return {
          jobs: { ...s.jobs, [ev.job.id]: ev.job },
          results,
          frames: restarted ? without(s.frames, ev.job.id) : s.frames,
          runSince: startedNow
            ? { ...s.runSince, [ev.job.id]: performance.now() - ev.job.elapsed_s * 1000 }
            : running
              ? s.runSince
              : without(s.runSince, ev.job.id),
        };
      });
      break;
    case "removed":
      set((s) => ({
        jobs: without(s.jobs, ev.id),
        started: without(s.started, ev.id),
        startedFull: without(s.startedFull, ev.id),
        frames: without(s.frames, ev.id),
        cases: without(s.cases, ev.id),
        results: without(s.results, ev.id),
        inputs: without(s.inputs, ev.id),
        runSince: without(s.runSince, ev.id),
        watching: s.watching.filter((w) => w !== ev.id),
      }));
      break;
    case "started": {
      const { type: _type, id, full, ...rest } = ev;
      void _type;
      set((s) => (full ? { startedFull: { ...s.startedFull, [id]: rest } } : { started: { ...s.started, [id]: rest } }));
      break;
    }
    case "progress": {
      const { type: _type, id, ...progress } = ev;
      void _type;
      set((s) => (s.jobs[id] ? { jobs: { ...s.jobs, [id]: { ...s.jobs[id], progress } } } : {}));
      break;
    }
    case "frame": {
      const { type: _type, id, ...frame } = ev;
      void _type;
      set((s) => ({ frames: { ...s.frames, [id]: frame } }));
      break;
    }
    case "case":
    case "case_progress": {
      const { type, id, case: i, ...rest } = ev;
      set((s) => {
        const cur = s.cases[id] ?? {};
        const prev = cur[i] ?? {};
        const next: SweepCase = type === "case" ? { ...prev, ...rest } : { ...prev, step: ev.step, n_steps: ev.n_steps };
        return { cases: { ...s.cases, [id]: { ...cur, [i]: next } } };
      });
      break;
    }
    case "done": {
      const job = useJobs.getState().jobs[ev.id];
      if (job) {
        const name = jobName(job);
        if (ev.state === "stopped") logInfo(t("msg.source.jobs"), t("jobs.msgStopped", { name, s: ev.elapsed_s.toFixed(1) }));
        else logInfo(t("msg.source.jobs"), t("jobs.msgDone", { name, s: ev.elapsed_s.toFixed(1) }));
        const latest = { ...job, state: ev.state };
        for (const fn of doneListeners) fn(latest);
      }
      break;
    }
    case "error": {
      const job = useJobs.getState().jobs[ev.id];
      logError(t("msg.source.jobs"), t("jobs.msgFailed", { name: job ? jobName(job) : ev.id, error: ev.detail }));
      break;
    }
    case "pong":
      break;
  }
}

// ---- 操作 ----

export async function runJob(kind: JobKind, project: Project | null, options: Record<string, unknown> = {}, label?: string): Promise<JobSummary | null> {
  try {
    const job = await submitJob(kind, project, options, label);
    useJobs.setState((s) => ({
      jobs: s.jobs[job.id] ? s.jobs : { ...s.jobs, [job.id]: job },
      inputs: { ...s.inputs, [job.id]: { project, docSerial: useDocument.getState().docSerial } },
    }));
    logInfo(t("msg.source.jobs"), t("jobs.msgSubmitted", { name: jobName(job) }));
    return job;
  } catch (e) {
    logError(t("msg.source.jobs"), t("jobs.msgSubmitFailed", { kind: t(`jobs.kind.${kind}`), error: errorText(e) }));
    return null;
  }
}

export async function stopRun(id: string): Promise<void> {
  try {
    await stopJob(id);
  } catch (e) {
    logError(t("msg.source.jobs"), errorText(e));
  }
}

export async function continueRun(id: string, options: Record<string, unknown>): Promise<void> {
  try {
    await continueJob(id, options);
  } catch (e) {
    logError(t("msg.source.jobs"), errorText(e));
  }
}

export async function removeRun(id: string): Promise<void> {
  try {
    await deleteJob(id);
    handleEvent({ type: "removed", id });
  } catch (e) {
    logError(t("msg.source.jobs"), errorText(e));
  }
}

/** 実行の結果 (同じ実行なら覚えているものを返す) */
export async function fetchResult<T = unknown>(id: string): Promise<T | null> {
  const s = useJobs.getState();
  const job = s.jobs[id];
  if (!job || !job.has_result) return null;
  const cached = s.results[id];
  if (cached && cached.run === job.runs) return cached.data as T;
  const data = await jobResult<T>(id);
  useJobs.setState((st) => ({ results: { ...st.results, [id]: { run: job.runs, data } } }));
  return data;
}

// ---- 接続 ----

let socket: WebSocket | null = null;
let retry: ReturnType<typeof setTimeout> | null = null;

function send(msg: unknown): void {
  if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(msg));
}

/** ライブ表示するジョブ (フレームと表示用メッシュが届く、P6e) */
export function watchJobs(ids: string[]): void {
  useJobs.setState({ watching: ids });
  send({ cmd: "watch", ids });
}

function open(): void {
  if (socket || typeof WebSocket === "undefined") return;
  let sock: WebSocket;
  try {
    sock = new WebSocket(wsUrl("/v2/events"));
  } catch {
    return;
  }
  socket = sock;
  sock.onopen = () => {
    useJobs.setState({ connected: true });
    const w = useJobs.getState().watching;
    if (w.length) send({ cmd: "watch", ids: w });
  };
  sock.onmessage = (e) => {
    try {
      handleEvent(JSON.parse(String(e.data)) as JobEvent);
    } catch (err) {
      logWarning(t("msg.source.jobs"), errorText(err));
    }
  };
  sock.onclose = () => {
    if (socket === sock) socket = null;
    useJobs.setState({ connected: false });
    // バックエンドにはつながっているのに切れたら少し待って張り直す
    if (retry === null && useConnection.getState().status === "connected") {
      retry = setTimeout(() => {
        retry = null;
        if (useConnection.getState().status === "connected" && useConnection.getState().info?.jobs) open();
      }, 2000);
    }
  };
}

function close(): void {
  if (retry !== null) {
    clearTimeout(retry);
    retry = null;
  }
  const s = socket;
  socket = null;
  s?.close();
  useJobs.setState({ connected: false });
}

let started = false;

/** 起動時に 1 回: バックエンドにつながったら (再起動したら張り直して) イベントを受ける */
export function startJobEvents(): void {
  if (started) return;
  started = true;
  const sync = (status: string, jobs: boolean, instanceChanged: boolean) => {
    if (status !== "connected" || !jobs) {
      close();
      return;
    }
    if (instanceChanged) {
      close();
      // 前のバックエンドのジョブは消えている
      useJobs.setState({ ...empty, inputs: {} });
    }
    open();
  };
  const c = useConnection.getState();
  sync(c.status, Boolean(c.info?.jobs), false);
  useConnection.subscribe((s, prev) => {
    const changed = s.status !== prev.status || s.info?.instance !== prev.info?.instance || s.port !== prev.port;
    if (!changed) return;
    const instanceChanged = prev.info?.instance !== undefined && s.info?.instance !== undefined && s.info.instance !== prev.info.instance;
    sync(s.status, Boolean(s.info?.jobs), instanceChanged || s.port !== prev.port);
  });
}
