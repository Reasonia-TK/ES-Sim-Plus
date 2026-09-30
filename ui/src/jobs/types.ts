// /v2/jobs と /v2/events の型 (backend es_sim/jobs と同じ形)。

export const JOB_KINDS = ["pic", "pic1d", "fluid1d", "fluid2d", "dsmc", "tl", "trace", "boltz", "sweep"] as const;
export type JobKind = (typeof JOB_KINDS)[number];

export type JobState = "queued" | "running" | "done" | "stopped" | "error" | "cancelled";

export const TERMINAL_STATES: JobState[] = ["done", "stopped", "error", "cancelled"];

export interface JobProgress {
  step: number;
  n_steps: number;
  step_offset: number;
  /** 0〜1 (ステップ数が分からなければ null) */
  fraction: number | null;
  [key: string]: unknown;
}

export interface JobSummary {
  id: string;
  kind: JobKind;
  /** 種類ごとの通し番号 (PIC #3 の 3) */
  seq: number;
  label: string | null;
  state: JobState;
  stopping: boolean;
  /** UNIX 時刻 [s] */
  created: number;
  started_at: number | null;
  finished_at: number | null;
  elapsed_s: number;
  runs: number;
  progress: JobProgress | null;
  error: string | null;
  warnings: string[];
  can_continue: boolean;
  has_result: boolean;
  queue_position: number | null;
  options: Record<string, unknown>;
  /** 結果付きのファイルやスイープのケースから読み込んだ実行 (この画面だけにある、バックエンドの操作はできない) */
  imported?: boolean;
}

export interface JobLimits {
  max_running: number;
  default_per_kind: number;
  per_kind: Record<string, number>;
}

export interface SweepCase {
  value?: number;
  ok?: boolean;
  error?: string;
  step?: number;
  n_steps?: number;
}

export type JobEvent =
  | { type: "hello"; jobs: JobSummary[]; limits: JobLimits; kinds: string[]; instance: string }
  | { type: "job"; job: JobSummary }
  | { type: "removed"; id: string }
  | ({ type: "started"; id: string; run: number; full: boolean } & Record<string, unknown>)
  | ({ type: "progress"; id: string } & JobProgress)
  | ({ type: "frame"; id: string } & Record<string, unknown>)
  | { type: "done"; id: string; state: JobState; elapsed_s: number }
  | { type: "error"; id: string; detail: string }
  | { type: "case"; id: string; case: number; value: number; ok: boolean; error?: string }
  | { type: "case_progress"; id: string; case: number; step: number; n_steps: number }
  | { type: "pong" };

/** 続けられる種類 (backend の Runner.continuable) */
export const CONTINUABLE: JobKind[] = ["pic", "pic1d", "fluid1d", "fluid2d", "dsmc"];

/** 続きの設定に位相分解のビン数も渡せる種類 */
export const HAS_PHASE_BINS: JobKind[] = ["pic", "pic1d", "fluid1d", "fluid2d"];
