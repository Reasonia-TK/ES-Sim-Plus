// 結果付きの保存・読み込み (v1 の「結果付き保存」と同じ形): {...project, results: {version: 1, solve, mesh, trace,
// pic, gas, pic1d, fluid1d, fluid2d, tl}}。読み込んだ結果は「読み込んだ実行」として結果の一覧に並べ、ビューアと
// グラフで見られる (続き・停止などバックエンドでの操作はできない)。保存は種類ごとに 1 つ (出している実行、
// 無ければいちばん新しいもの) と静電場の結果。スイープのケースの結果 (同じ形) もここから読み込む。

import { t } from "../i18n";
import { fetchResult, useJobs } from "../jobs/jobsStore";
import type { JobKind, JobSummary } from "../jobs/types";
import { useDocument } from "../model/documentStore";
import type { Project } from "../model/project";
import type { MeshResult, SolveResult } from "../backend/staticApi";
import { useResultsView } from "./resultsView";
import { meshKey, solveKey, toViewMesh, useStatic } from "./staticResults";

export const BUNDLE_VERSION = 1;

/** 結果の束のキー → 実行の種類 */
export const BUNDLE_KINDS: { key: string; kind: JobKind }[] = [
  { key: "pic", kind: "pic" },
  { key: "pic1d", kind: "pic1d" },
  { key: "fluid1d", kind: "fluid1d" },
  { key: "fluid2d", kind: "fluid2d" },
  { key: "gas", kind: "dsmc" },
  { key: "tl", kind: "tl" },
  { key: "trace", kind: "trace" },
];

let importSeq = 0;

/** 読み込んだ結果を実行の一覧に足す (結果と入力を覚えておく) */
export function addImportedRun(kind: JobKind, result: unknown, project: Project | null, source: string): JobSummary {
  importSeq++;
  const id = `import-${Date.now().toString(36)}-${importSeq}`;
  const now = Date.now() / 1000;
  const elapsed = (result as { elapsed_s?: unknown } | null)?.elapsed_s;
  const job: JobSummary = {
    id,
    kind,
    seq: importSeq,
    label: t("results.importedLabel", { kind: t(`jobs.kind.${kind}`), source }),
    state: "done",
    stopping: false,
    created: now,
    started_at: null,
    finished_at: now,
    elapsed_s: typeof elapsed === "number" ? elapsed : 0,
    runs: 1,
    progress: null,
    error: null,
    warnings: [],
    can_continue: false,
    has_result: true,
    queue_position: null,
    options: {},
    imported: true,
  };
  useJobs.setState((s) => ({
    jobs: { ...s.jobs, [id]: job },
    results: { ...s.results, [id]: { run: 1, data: result } },
    inputs: { ...s.inputs, [id]: { project, docSerial: useDocument.getState().docSerial } },
  }));
  return job;
}

/** 列の辞書 (キー → 配列) を行の配列にする (PIC の履歴、バッチの保存は列のことがある) */
function rowsOf(h: unknown): Record<string, number>[] {
  if (Array.isArray(h)) return h as Record<string, number>[];
  if (!h || typeof h !== "object") return [];
  const cols = Object.entries(h as Record<string, unknown>).filter(([, v]) => Array.isArray(v)) as [string, number[]][];
  const n = Math.max(0, ...cols.map(([, v]) => v.length));
  return Array.from({ length: n }, (_, i) => Object.fromEntries(cols.map(([k, v]) => [k, v[i]])));
}

/** 束の中の 1 つの結果を v2 の結果の形にそろえる (古い保存に無いキーを補う) */
function normalizeResult(key: string, r: Record<string, unknown>, bundle: Record<string, unknown>): Record<string, unknown> {
  switch (key) {
    case "pic":
      return { ...r, frame: r.frame ?? null, history: rowsOf(r.history), fields: r.fields ?? null, cycle: r.cycle ?? null, collectors: r.collectors ?? [], eedf: r.eedf ?? [] };
    case "pic1d":
      return { ...r, eedf: r.eedf ?? [], sheath: r.sheath ?? null, sheath_ts: r.sheath_ts ?? null, sheath_fft: r.sheath_fft ?? null, wall_iedf: r.wall_iedf ?? null };
    case "fluid2d":
      // v1 は流体 2D のメッシュを静電場のメッシュ (results.mesh) で持っていた
      return { ...r, mesh: r.mesh ?? bundle.mesh ?? undefined };
    default:
      return r;
  }
}

/**
 * 結果の束を読み込む (実行の一覧に足し、静電場の結果は静電場のページへ)。足した実行を返す。
 * source は実行の名前の出どころ (束の項目 (pic・fluid2d など) ごとに変えるときは関数)
 */
export function importResultsBundle(results: unknown, project: Project, source: string | ((key: string) => string)): JobSummary[] {
  if (!results || typeof results !== "object") return [];
  const b = results as Record<string, unknown>;
  const added: JobSummary[] = [];
  for (const { key, kind } of BUNDLE_KINDS) {
    const r = b[key];
    if (!r || typeof r !== "object") continue;
    const src = typeof source === "function" ? source(key) : source;
    added.push(addImportedRun(kind, normalizeResult(key, r as Record<string, unknown>, b), project, src));
  }
  const solve = b.solve as SolveResult | undefined;
  const mesh = b.mesh as MeshResult | undefined;
  if (solve?.mesh && Array.isArray(solve.v)) {
    const eAbs = Float64Array.from(solve.e_field ?? [], ([x, y]) => Math.hypot(x, y));
    useStatic.setState({ solve: { result: solve, project, view: toViewMesh(solve.mesh), eAbs, key: solveKey(project), elapsedS: 0 }, latest: "solve" });
  }
  if (mesh?.nodes && mesh.triangles) {
    useStatic.setState((s) => ({ mesh: { result: mesh, view: toViewMesh(mesh), key: meshKey(project), elapsedS: 0 }, latest: s.solve ? s.latest : "mesh" }));
  }
  return added;
}

/** 保存する結果の束 (種類ごとに出している実行か、いちばん新しい結果)。何も無ければ null */
export async function buildResultsBundle(): Promise<Record<string, unknown> | null> {
  const out: Record<string, unknown> = { version: BUNDLE_VERSION };
  let any = false;
  const st = useStatic.getState();
  if (st.solve) {
    out.solve = st.solve.result;
    any = true;
  }
  if (st.mesh) {
    out.mesh = st.mesh.result;
    any = true;
  }
  const jobs = Object.values(useJobs.getState().jobs);
  const active = useResultsView.getState().activeRun;
  for (const { key, kind } of BUNDLE_KINDS) {
    const cands = jobs.filter((j) => j.kind === kind && j.has_result && j.state !== "running").sort((a, b) => b.created - a.created);
    const pick = cands.find((j) => j.id === active) ?? cands[0];
    if (!pick) continue;
    const data = await fetchResult(pick.id);
    if (data === null || data === undefined) continue;
    out[key] = data;
    any = true;
  }
  return any ? out : null;
}
