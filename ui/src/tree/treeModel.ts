// モデルツリーの節 (プロジェクト文書から組み立てる。React に依存しないのでテストできる)。

import type { TFunction } from "i18next";
import {
  boundaryOfEdge,
  coordOf,
  domainBounds,
  edgeIdsOf,
  isAxisEdge,
  isRectDomain,
  type Project,
} from "../model/project";
import type { NodeId } from "../model/selection";
import { layersOf } from "../model/layers";
import { paramsOf } from "../model/params";
import { sketchOf } from "../model/sketch";
import type { JobKind, JobSummary } from "../jobs/types";
import { formatNumber, lengthUnitLabel, toDisplayLength, type LengthUnit } from "../util/format";

export type BadgeTone = "muted" | "ok" | "warn" | "error" | "run";

export interface TreeNode {
  id: NodeId;
  label: string;
  /** 補足 (値の概要) */
  detail?: string;
  badge?: { text: string; tone: BadgeTone };
  /** 選べない説明だけの行 (「領域なし」など) */
  placeholder?: boolean;
  children?: TreeNode[];
}

export const STUDIES = ["fem", "trace", "pic", "pic1d", "fluid1d", "fluid2d", "dsmc", "tl", "sweep"] as const;
export type StudyKind = (typeof STUDIES)[number];

/** スタディの設定が入っている project のキー (静電場・スイープは専用の設定ブロックが無い) */
export const STUDY_SETTINGS_KEY: Record<StudyKind, keyof Project | null> = {
  fem: null,
  trace: "particles",
  pic: "pic",
  pic1d: "pic1d",
  fluid1d: "fluid1d",
  fluid2d: "fluid2d",
  dsmc: "dsmc",
  tl: "tl",
  sweep: null,
};

/** 辺の表示名 (矩形は座標系ごとの名前、それ以外は「辺 n」) */
export function edgeLabel(p: Project, i: number, t: TFunction): string {
  if (isRectDomain(p)) {
    const labels = t(`edge.${coordOf(p)}`, { returnObjects: true }) as unknown as string[];
    if (Array.isArray(labels) && labels[i]) return labels[i];
  }
  return t("edge.generic", { n: i });
}

/** 辺の境界条件の概要 (v1 のツリーと同じ: 対称軸 / なし / Dirichlet NV [+RF] / 対称 / 周期) */
export function edgeSummary(p: Project, i: number, t: TFunction): string {
  if (isAxisEdge(p, i)) return t("bc.axis");
  const bc = boundaryOfEdge(p, i);
  if (!bc || bc.type === "neumann") return t("bc.none");
  if (bc.type === "dirichlet") {
    const v = typeof bc.voltage === "number" ? bc.voltage : 0;
    let s = `${t("bc.dirichlet")} ${formatNumber(v)} V`;
    if (bc.voltage_rf) s += ` + ${t("bc.rf")}`;
    if (bc.voltage_waveform) s += ` + ${t("bc.waveform")}`;
    return s;
  }
  return t(`bc.${bc.type}`);
}

function domainDetail(p: Project, unit: LengthUnit): string {
  const b = domainBounds(p);
  const u = lengthUnitLabel(unit);
  const w = formatNumber(toDisplayLength(b.x1 - b.x0, unit));
  const h = formatNumber(toDisplayLength(b.y1 - b.y0, unit));
  return `${w} × ${h} ${u}`;
}

function meshDetail(p: Project, unit: LengthUnit): string {
  const size = `${formatNumber(toDisplayLength(p.mesh.size, unit))} ${lengthUnitLabel(unit)}`;
  const mode = p.mesh.mode ?? "unstructured";
  // 細分化の矩形のレベルも含めた最大 (v1 と同じ)
  const amrObj = p.mesh.amr as { max_level?: number; regions?: { level?: number }[] } | null | undefined;
  const level = Math.max(amrObj?.max_level ?? 0, ...(amrObj?.regions ?? []).map((r) => r.level ?? 0));
  const amr = mode === "cartesian" && amrObj && level > 0 ? ` · AMR L${level}` : "";
  return `${size} · ${mode}${amr}`;
}

function bfieldDetail(p: Project): string {
  const b = p.b_field;
  if (!b || (!b.bx && !b.by && !b.bz)) return "0";
  return `(${formatNumber(b.bx ?? 0)}, ${formatNumber(b.by ?? 0)}, ${formatNumber(b.bz ?? 0)}) T`;
}

/** スタディ → ジョブの種類 (静電場はジョブではない) */
export const STUDY_JOB_KIND: Record<StudyKind, JobKind | null> = {
  fem: null,
  trace: "trace",
  pic: "pic",
  pic1d: "pic1d",
  fluid1d: "fluid1d",
  fluid2d: "fluid2d",
  dsmc: "dsmc",
  tl: "tl",
  sweep: "sweep",
};

function runName(j: JobSummary, t: TFunction): string {
  return `${j.label ?? t(`jobs.kind.${j.kind}`)} #${j.seq}`;
}

const STATE_TONE: Record<JobSummary["state"], BadgeTone> = {
  queued: "muted",
  running: "run",
  done: "ok",
  stopped: "warn",
  error: "error",
  cancelled: "muted",
};

function pct(j: JobSummary): string {
  // スイープは終わったケースの数 (v1 と同じ)
  if (j.kind === "sweep" && j.progress) return ` ${j.progress.step}/${j.progress.n_steps}`;
  const f = j.progress?.fraction;
  return f === null || f === undefined ? "" : ` ${Math.round(f * 100)}%`;
}

/** 実行の状態の印 (実行中の進捗、待ち、直近の失敗)。無ければ null */
function runBadge(jobs: JobSummary[], t: TFunction): TreeNode["badge"] | null {
  if (jobs.length === 0) return null;
  const running = jobs.filter((j) => j.state === "running");
  if (running.length) return { text: `${t("jobs.state.running")}${pct(running[0])}${running.length > 1 ? ` +${running.length - 1}` : ""}`, tone: "run" };
  if (jobs.some((j) => j.state === "queued")) return { text: t("jobs.state.queued"), tone: "muted" };
  const latest = jobs[0];
  if (latest.state === "error") return { text: t("jobs.state.error"), tone: "error" };
  if (latest.state === "done" || latest.state === "stopped") return { text: t(`jobs.state.${latest.state}`), tone: STATE_TONE[latest.state] };
  return null;
}

export function studyConfigured(p: Project, kind: StudyKind): boolean {
  const key = STUDY_SETTINGS_KEY[kind];
  return key === null ? true : p[key] !== null && p[key] !== undefined;
}

/** 静電場の状態 (計算中・結果あり・設定が変わった) */
export interface StaticStatus {
  busy: boolean;
  solved: boolean;
  stale: boolean;
}

export function buildTree(p: Project, t: TFunction, unit: LengthUnit, docName: string, jobs: JobSummary[] = [], stat?: StaticStatus): TreeNode {
  const newest = [...jobs].sort((a, b) => b.created - a.created);
  const regions: TreeNode[] = p.geometry.regions.map((r) => ({
    id: `region:${r.id}`,
    label: r.id,
    detail: `${t(`region.${r.type}`)} · ${r.shape ? t("region.circle") : t("region.polygon")}${r.holes?.length ? ` · ${t("region.holes", { n: r.holes.length })}` : ""}`,
  }));
  const edges: TreeNode[] = edgeIdsOf(p).map((id, i) => ({
    id: `edge:${id}`,
    label: edgeLabel(p, i, t),
    detail: edgeSummary(p, i, t),
  }));
  const studies: TreeNode[] = STUDIES.map((kind) => {
    const configured = studyConfigured(p, kind);
    const hasBlock = STUDY_SETTINGS_KEY[kind] !== null;
    const jk = STUDY_JOB_KIND[kind];
    const run = jk
      ? runBadge(newest.filter((j) => j.kind === jk), t)
      : kind === "fem" && stat
        ? stat.busy
          ? { text: t("jobs.state.running"), tone: "run" as const }
          : stat.solved
            ? stat.stale
              ? { text: t("tree.stale"), tone: "warn" as const }
              : { text: t("jobs.state.done"), tone: "ok" as const }
            : { text: t("tree.notRun"), tone: "muted" as const }
        : null;
    return {
      id: `study:${kind}`,
      label: t(`study.${kind}`),
      badge:
        run ??
        (hasBlock
          ? configured
            ? { text: t("tree.configured"), tone: "ok" }
            : { text: t("tree.notConfigured"), tone: "muted" }
          : undefined),
    };
  });
  return {
    id: "project",
    label: docName,
    children: [
      { id: "params", label: t("tree.params"), detail: String(paramsOf(p).vars.length) },
      {
        id: "geometry",
        label: t("tree.geometry"),
        children: [
          { id: "domain", label: t("tree.domain"), detail: domainDetail(p, unit) },
          {
            id: "regions",
            label: t("tree.regions"),
            detail: String(regions.length),
            children: regions.length ? regions : [{ id: "regions.empty", label: t("tree.noRegions"), placeholder: true }],
          },
          { id: "sketch", label: t("tree.sketch"), detail: String(sketchOf(p).length) },
          { id: "layers", label: t("tree.layers"), detail: String(layersOf(p).length) },
          { id: "boundaries", label: t("tree.boundaries"), detail: String(edges.length), children: edges },
          { id: "mesh", label: t("tree.mesh"), detail: meshDetail(p, unit) },
          { id: "bfield", label: t("tree.bfield"), detail: bfieldDetail(p) },
        ],
      },
      { id: "studies", label: t("tree.studies"), children: studies },
      {
        id: "results",
        label: t("tree.results"),
        detail: newest.length ? String(newest.length) : undefined,
        children: newest.length
          ? newest.map((j) => ({
              id: `result:${j.id}`,
              label: runName(j, t),
              badge: { text: `${t(`jobs.state.${j.state}`)}${j.state === "running" ? pct(j) : ""}`, tone: STATE_TONE[j.state] },
            }))
          : [{ id: "results.empty", label: t("tree.noResults"), placeholder: true }],
      },
    ],
  };
}

/** 検索語で絞る (一致した節とその祖先を残す)。空なら null (絞らない) */
export function filterTree(root: TreeNode, query: string): TreeNode | null {
  const q = query.trim().toLowerCase();
  if (!q) return root;
  const walk = (n: TreeNode): TreeNode | null => {
    const hit = !n.placeholder && `${n.label} ${n.detail ?? ""}`.toLowerCase().includes(q);
    // 群の名前が一致したら子をすべて出す (v1 と同じ)
    if (hit && n.children && n.id !== "project") return n;
    const kids = (n.children ?? []).map(walk).filter((c): c is TreeNode => c !== null);
    if (!hit && kids.length === 0) return null;
    return { ...n, children: n.children ? kids : undefined };
  };
  return walk(root);
}

/** 表示中の節の並び (キーボード操作用)。expanded に無い枝の子は含めない */
export function visibleNodes(root: TreeNode, expanded: Set<NodeId>, forceOpen = false): { node: TreeNode; level: number; parent: NodeId | null }[] {
  const out: { node: TreeNode; level: number; parent: NodeId | null }[] = [];
  const walk = (n: TreeNode, level: number, parent: NodeId | null) => {
    out.push({ node: n, level, parent });
    if (n.children && (forceOpen || expanded.has(n.id))) for (const c of n.children) walk(c, level + 1, n.id);
  };
  walk(root, 1, null);
  return out;
}
