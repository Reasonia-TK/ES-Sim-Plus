// 実行の結果 → ビューアの Scene (PIC・流体 2D・DSMC・粒子軌道)。v1 App.tsx の picLiveFrame・picFieldView・
// picCycleView・fluid2d*View・gasFieldView・軌道と同じ中身で、ライブ (最新のフレーム)・時間平均・位相分解
// (全ビンで範囲を固定、粒子のスナップショット) を出す。シース端 (n_e/n_i = α の等値線) も。

import type { JobSummary } from "../jobs/types";
import type { Point } from "../model/project";
import type { Display2d, Mode2d } from "../results/resultsView";
import type { DsmcFrame, DsmcResult, Fluid2dResult, PicFrame, PicResult, TraceResult } from "../results/types";
import { sheathOnLine, sheathPoint, type SheathLineDef } from "../results/sheath";
import { fieldStats, type FieldStats } from "./fieldScale";
import type { Playback } from "./PlaybackBar";
import { EMPTY_SCENE, pathsToSegments, type FieldLocation, type PointSet, type ScalarField, type Scene, type SegmentSet, type ViewMesh } from "./scene";

export interface RunScene extends Scene {
  /** 位相分解を表示している */
  playback?: Playback;
  /** 使える表示 (ライブ・時間平均・位相分解) */
  modes: Mode2d[];
  /** 表示している量の一覧 (今の表示で選べるもの) */
  quantities: string[];
  /** 表示している表示・量 (ほかが無いとき寄せたもの) */
  mode: Mode2d | null;
  quantity: string | null;
  /** 塗るときの対数 (実行の表示の設定) */
  log?: boolean;
}

export interface QuantityMeta {
  location: FieldLocation;
  unit: string;
}

/** 量の単位と節点/要素 (v1 の PIC_FIELD_META・FLUID2D_FIELD_META・GAS_FIELD_META) */
export const QUANTITY_META: Record<string, QuantityMeta> = {
  phi: { location: "node", unit: "V" },
  e_abs: { location: "element", unit: "V/m" },
  n_e: { location: "node", unit: "m^-3" },
  n_i: { location: "node", unit: "m^-3" },
  te_ev: { location: "node", unit: "eV" },
  t_e: { location: "node", unit: "eV" },
  ion_rate: { location: "node", unit: "m^-3 s^-1" },
  ionization: { location: "node", unit: "m^-3 s^-1" },
  n: { location: "element", unit: "m^-3" },
  t: { location: "element", unit: "K" },
  u: { location: "element", unit: "m/s" },
  p: { location: "element", unit: "Pa" },
};

export const PIC_FIELDS = ["phi", "e_abs", "n_e", "n_i", "te_ev", "ion_rate"];
export const PIC_LIVE = ["phi", "n_e", "n_i"];
export const PIC_CYCLE = ["phi", "n_e", "n_i", "e_abs", "te_ev", "ion_rate"];
export const FLUID2D_FIELDS = ["phi", "e_abs", "n_e", "n_i", "t_e", "ionization"];
export const FLUID2D_LIVE = ["phi", "n_e", "n_i", "t_e"];
export const FLUID2D_CYCLE = ["phi", "n_e", "n_i", "t_e"];
export const DSMC_FIELDS = ["n", "t", "u", "p"];

type MeshLike = { nodes: Point[]; triangles: [number, number, number][]; region_of_triangle?: number[] };

// ---- 同じデータには同じオブジェクト (GPU の資源・メモ化を使い回す) ----

const meshCache = new WeakMap<object, ViewMesh>();
/** 結果のメッシュ → ViewMesh (同じメッシュには同じオブジェクト、メッシュの索引もこれで使い回す) */
export function viewMeshOf(m: MeshLike): ViewMesh {
  let v = meshCache.get(m);
  if (!v) {
    v = { nodes: m.nodes, triangles: m.triangles, regionOfTriangle: m.region_of_triangle };
    meshCache.set(m, v);
  }
  return v;
}

const fieldCache = new WeakMap<object, Map<string, ScalarField>>();
function field(mesh: ViewMesh, values: ArrayLike<number>, location: FieldLocation, label: string, unit: string, stats?: FieldStats): ScalarField {
  const key = values as object;
  let byMeta = fieldCache.get(key);
  if (!byMeta) {
    byMeta = new Map();
    fieldCache.set(key, byMeta);
  }
  const k = `${label}|${unit}|${location}|${stats ? "fixed" : ""}`;
  let f = byMeta.get(k);
  if (!f || f.mesh !== mesh) {
    f = { mesh, values, location, label, unit, stats };
    byMeta.set(k, f);
  }
  return f;
}

const rowsStatsCache = new WeakMap<object, FieldStats>();
/** 位相分解の全ビンの最小・最大 (ビンを送っても色が変わらないよう固定する、v1 と同じ) */
export function rowsStats(rows: number[][]): FieldStats {
  let s = rowsStatsCache.get(rows);
  if (!s) {
    let min = Infinity;
    let max = -Infinity;
    let minPositive = Infinity;
    for (const row of rows) {
      const r = fieldStats(row);
      if (r.min < min) min = r.min;
      if (r.max > max) max = r.max;
      if (r.minPositive < minPositive) minPositive = r.minPositive;
    }
    s = min <= max ? { min, max, minPositive } : { min: 0, max: 0, minPositive };
    rowsStatsCache.set(rows, s);
  }
  return s;
}

const magCache = new WeakMap<object, Float64Array>();
function magnitude(u: [number, number][]): Float64Array {
  let m = magCache.get(u);
  if (!m) {
    m = Float64Array.from(u, ([x, y]) => Math.hypot(x, y));
    magCache.set(u, m);
  }
  return m;
}

const pointsCache = new WeakMap<object, Float64Array>();
function flatPoints(pts: Point[]): Float64Array {
  let a = pointsCache.get(pts);
  if (!a) {
    a = new Float64Array(2 * pts.length);
    pts.forEach(([x, y], i) => {
      a![2 * i] = x;
      a![2 * i + 1] = y;
    });
    pointsCache.set(pts, a);
  }
  return a;
}

// ---- シース端 ----

const sheathCache = new WeakMap<object, Map<string, Float64Array>>();

/**
 * 準中性度 n_e/n_i = α の等値線 (v1 sheath.ts の marchingTrianglesContour と同じ)。3 頂点とも n_i が最大の 2% 未満の
 * 三角形 (ほぼ真空で比が暴れる) は除く。n_e・n_i は節点の値。
 */
export function sheathContour(mesh: ViewMesh, nE: ArrayLike<number>, nI: ArrayLike<number>, alpha: number): Float64Array {
  const key = nE as object;
  let byAlpha = sheathCache.get(key);
  if (!byAlpha) {
    byAlpha = new Map();
    sheathCache.set(key, byAlpha);
  }
  const k = `${alpha}|${mesh.triangles.length}`;
  const hit = byAlpha.get(k);
  if (hit) return hit;
  let maxNi = 0;
  for (let i = 0; i < nI.length; i++) if (nI[i] > maxNi) maxNi = nI[i];
  const thr = 0.02 * maxNi;
  const ratio = (i: number) => (nI[i] > 0 ? nE[i] / nI[i] : nE[i] > 0 ? Infinity : 0);
  const out: number[] = [];
  const pts: number[] = [];
  const cross = (a: number, b: number, va: number, vb: number) => {
    if (va < alpha === vb < alpha) return;
    const t = (alpha - va) / (vb - va);
    if (!Number.isFinite(t)) return;
    const [ax, ay] = mesh.nodes[a];
    const [bx, by] = mesh.nodes[b];
    pts.push(ax + (bx - ax) * t, ay + (by - ay) * t);
  };
  for (const [a, b, c] of mesh.triangles) {
    if (nI[a] < thr && nI[b] < thr && nI[c] < thr) continue;
    const va = ratio(a);
    const vb = ratio(b);
    const vc = ratio(c);
    if (![va, vb, vc].every(Number.isFinite)) continue;
    pts.length = 0;
    cross(a, b, va, vb);
    cross(b, c, vb, vc);
    cross(c, a, vc, va);
    if (pts.length === 4) out.push(pts[0], pts[1], pts[2], pts[3]);
  }
  const arr = Float64Array.from(out);
  byAlpha.set(k, arr);
  return arr;
}

// ---- 組み立て ----

export interface RunSceneInput {
  job: JobSummary;
  result: unknown;
  started?: Record<string, unknown>;
  frame?: Record<string, unknown>;
  liveMesh?: unknown;
  display: Display2d;
  bin: number;
  sheath: { contour: boolean; alpha: number };
  /** シース評価線 (文書の pic.sheath_lines、シース端 s の点を出す) */
  sheathLines?: SheathLineDef[];
  /** 量の表示名 */
  label: (quantity: string) => string;
  /** 見出し (PIC #3 など) */
  name: string;
  /** 表示の名前 (ライブ・時間平均・位相分解) */
  modeLabel: (mode: Mode2d) => string;
  /** 粒子軌道の背景 (静電場の結果) */
  background: Scene;
}

const ELECTRON = "#4dd4ff";
const ION = "#ff9d4d";
const SHEATH = "#ffb454";

function pickMode(want: Mode2d, available: Mode2d[]): Mode2d | null {
  if (available.includes(want)) return want;
  return available[0] ?? null;
}

function pick(want: string, list: string[]): string {
  return list.includes(want) ? want : list[0];
}

function base(input: RunSceneInput, mode: Mode2d | null, modes: Mode2d[], quantities: string[], quantity: string | null): RunScene {
  return { ...EMPTY_SCENE, title: input.name, modes, quantities, mode, quantity };
}

function withField(s: RunScene, input: RunSceneInput, mesh: ViewMesh, values: ArrayLike<number> | undefined, quantity: string, log: boolean, location?: FieldLocation, stats?: FieldStats): RunScene {
  if (!values) return s;
  const meta = QUANTITY_META[quantity];
  const f = field(mesh, values, location ?? meta.location, input.label(quantity), meta.unit, stats);
  return { ...s, mesh, field: f, iso: f.location === "node" ? f : null, log, title: `${input.name} · ${input.modeLabel(s.mode!)}` };
}

function addSheath(s: RunScene, input: RunSceneInput, mesh: ViewMesh, nE?: ArrayLike<number>, nI?: ArrayLike<number>): RunScene {
  if (!nE || !nI) return s;
  let out = s;
  if (input.sheath.contour) {
    const seg: SegmentSet = { id: "sheath", segments: sheathContour(mesh, nE, nI, input.sheath.alpha), color: SHEATH, width: 1.5 };
    out = { ...out, segments: [...out.segments, seg] };
  }
  // 評価線のシース端 s (電極側から s の点)
  const pts: number[] = [];
  for (const l of input.sheathLines ?? []) {
    const sv = sheathOnLine(mesh, l.p1, l.p2, nE, nI);
    if (sv !== null) pts.push(...sheathPoint(l.p1, l.p2, sv));
  }
  if (pts.length) out = { ...out, points: [...out.points, { id: "sheathS", xy: Float64Array.from(pts), color: "#ffffff", size: 8 }] };
  return out;
}

function picScene(input: RunSceneInput): RunScene {
  const res = input.result as PicResult | null;
  const running = input.job.state === "running";
  const frame = (running ? input.frame : (res?.frame ?? undefined)) as PicFrame | undefined;
  const modes: Mode2d[] = [];
  if (frame) modes.push("live");
  if (res?.fields) modes.push("field");
  if (res?.cycle) modes.push("cycle");
  const d = input.display;
  const mode = running ? (frame ? "live" : null) : pickMode(d.mode, modes);
  const quantities = mode === "live" ? PIC_LIVE : mode === "field" ? PIC_FIELDS : mode === "cycle" ? PIC_CYCLE.filter((q) => res?.cycle?.[q as keyof typeof res.cycle] !== undefined) : [];
  if (!mode) return base(input, null, modes, [], null);
  if (mode === "live" && frame) {
    const meshSrc = (input.liveMesh ?? frame.mesh ?? (input.started?.mesh as MeshLike | undefined) ?? res?.started.mesh) as MeshLike | undefined;
    const q = pick(d.live, quantities);
    let s = base(input, mode, modes, quantities, q);
    if (meshSrc) {
      const mesh = viewMeshOf(meshSrc);
      const values = q === "phi" ? frame.phi : q === "n_e" ? frame.n_e : frame.n_i;
      // ライブの n_e・n_i は要素の値 (時間平均の場とは違う)
      s = withField(s, input, mesh, values, q, q !== "phi" && d.logLive, q === "phi" ? "node" : "element");
    }
    s = { ...s, points: particleSets(frame.particles.electron, frame.particles.ion) };
    return s;
  }
  const mesh = viewMeshOf(res!.started.mesh);
  if (mode === "field") {
    const q = pick(d.field, quantities);
    const f = res!.fields!;
    const s = withField(base(input, mode, modes, quantities, q), input, mesh, f[q as keyof typeof f] as number[] | undefined, q, d.logField);
    return addSheath(s, input, mesh, f.n_e, f.n_i);
  }
  const c = res!.cycle!;
  const q = pick(d.cycle, quantities);
  const rows = c[q as keyof typeof c] as number[][] | undefined;
  const bin = Math.min(Math.max(0, input.bin), c.bins - 1);
  let s = base(input, mode, modes, quantities, q);
  if (rows) s = withField(s, input, mesh, rows[bin], q, d.logCycle, undefined, rowsStats(rows));
  s.playback = { bins: c.bins, periodS: c.period_s };
  s = { ...s, points: particleSets(c.particles.electron[bin] ?? [], c.particles.ion[bin] ?? []) };
  return addSheath(s, input, mesh, c.n_e[bin], c.n_i[bin]);
}

function particleSets(electron: Point[], ion: Point[]): PointSet[] {
  return [
    { id: "electron", xy: flatPoints(electron), color: ELECTRON, size: 2 },
    { id: "ion", xy: flatPoints(ion), color: ION, size: 2 },
  ];
}

function fluid2dScene(input: RunSceneInput): RunScene {
  const res = input.result as Fluid2dResult | null;
  const running = input.job.state === "running";
  const frame = running ? (input.frame as Record<string, number[]> | undefined) : undefined;
  const meshSrc = (res?.mesh ?? (input.started?.mesh as MeshLike | undefined)) as MeshLike | undefined;
  const modes: Mode2d[] = [];
  if (frame) modes.push("live");
  if (res?.fields) modes.push("field");
  if (res?.cycle) modes.push("cycle");
  const d = input.display;
  const mode = running ? (frame ? "live" : null) : pickMode(d.mode, modes);
  const quantities = mode === "live" ? FLUID2D_LIVE : mode === "field" ? FLUID2D_FIELDS : mode === "cycle" ? FLUID2D_CYCLE : [];
  if (!mode || !meshSrc) return base(input, mode, modes, quantities, null);
  const mesh = viewMeshOf(meshSrc);
  if (mode === "live" && frame) {
    const q = pick(d.live, quantities);
    return withField(base(input, mode, modes, quantities, q), input, mesh, frame[q], q, q !== "phi" && d.logLive);
  }
  if (mode === "field") {
    const q = pick(d.field, quantities);
    const f = res!.fields!;
    const s = withField(base(input, mode, modes, quantities, q), input, mesh, f[q as keyof typeof f] as number[] | undefined, q, d.logField);
    return addSheath(s, input, mesh, f.n_e, f.n_i);
  }
  const c = res!.cycle!;
  const q = pick(d.cycle, quantities);
  const rows = c[q as keyof typeof c] as number[][] | undefined;
  const bin = Math.min(Math.max(0, input.bin), c.bins - 1);
  let s = base(input, mode, modes, quantities, q);
  if (rows) s = withField(s, input, mesh, rows[bin], q, d.logCycle, undefined, rowsStats(rows));
  s.playback = { bins: c.bins, periodS: c.freq_hz > 0 ? 1 / c.freq_hz : 0 };
  return addSheath(s, input, mesh, c.n_e[bin], c.n_i[bin]);
}

function dsmcScene(input: RunSceneInput): RunScene {
  const res = input.result as DsmcResult | null;
  const running = input.job.state === "running";
  const frame = running ? (input.frame as DsmcFrame | undefined) : undefined;
  const modes: Mode2d[] = [];
  if (frame) modes.push("live");
  if (res) modes.push("field");
  const d = input.display;
  const mode = running ? (frame ? "live" : null) : pickMode(d.mode, modes);
  if (mode === "live" && frame) {
    const s = base(input, mode, modes, [], null);
    return { ...s, title: `${input.name} · ${input.modeLabel(mode)}`, points: [{ id: "gas", xy: flatPoints(frame.particles), color: "#c8ccd4", size: 2.4 }] };
  }
  if (mode !== "field" || !res) return base(input, mode, modes, [], null);
  const q = pick(d.field, DSMC_FIELDS);
  const values = q === "u" ? magnitude(res.u) : (res[q as "n" | "t" | "p"] as number[]);
  return withField(base(input, mode, modes, DSMC_FIELDS, q), input, viewMeshOf(res.mesh), values, q, d.logField);
}

const traceCache = new WeakMap<object, { segs: Float64Array; ends: Float64Array }>();

function traceScene(input: RunSceneInput): RunScene {
  const res = input.result as TraceResult | null;
  const s: RunScene = { ...input.background, modes: [], quantities: [], mode: null, quantity: null, title: input.name };
  if (!res) return s;
  let cached = traceCache.get(res);
  if (!cached) {
    const ends: number[] = [];
    res.trajectories.forEach((tr, i) => {
      if (res.status[i] === "absorbed" && tr.length) ends.push(tr[tr.length - 1][0], tr[tr.length - 1][1]);
    });
    cached = { segs: pathsToSegments(res.trajectories), ends: Float64Array.from(ends) };
    traceCache.set(res, cached);
  }
  return {
    ...s,
    segments: [...s.segments, { id: "trajectories", segments: cached.segs, color: "rgba(0,200,255,0.55)", width: 1 }],
    points: [...s.points, { id: "absorbed", xy: cached.ends, color: "rgba(0,200,255,0.95)", size: 4 }],
  };
}

/** 実行 → Scene (ビューアで場を出す種類でなければ null) */
export function runScene(input: RunSceneInput): RunScene | null {
  switch (input.job.kind) {
    case "pic":
      return picScene(input);
    case "fluid2d":
      return fluid2dScene(input);
    case "dsmc":
      return dsmcScene(input);
    case "trace":
      return traceScene(input);
    default:
      return null;
  }
}

/** ビューアで場を出す種類 */
export const VIEWER_KINDS = ["pic", "fluid2d", "dsmc", "trace"];
