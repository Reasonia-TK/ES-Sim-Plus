// プロジェクト文書 (backend の es_sim.schema.Project と同じ JSON) の型と読み書きの補助。
// P6a では木構造・ジオメトリ表示・ファイル操作に要る部分だけを型にし、各ソルバーの設定の中身は
// P6b でスキーマ (/v2/schema) から扱う。長さは常に m で持つ (表示単位への変換は UI の端で)。
// P7 (prompts/132): ドメイン・領域の輪郭は頂点 + 辺ごとの bulge (円弧)、外周の辺には永続 ID (edge_ids)、領域の穴 (holes)。

import type { Shape } from "../cad/boolean";
import { pathArea, pathBounds } from "../cad/geom";
import { bulgesOf, circlePath, hasArcs, type PathData } from "../cad/path";
import { applyParams } from "./params";

export type Point = [number, number];
export type Coord = "xy" | "rz" | "rz_x0";
export type RegionType = "conductor" | "dielectric" | "charge";
export type BcType = "dirichlet" | "neumann" | "symmetry" | "periodic";

export interface CircleShape {
  kind: "circle";
  center: Point;
  radius: number;
}

export interface Domain {
  /** 外周の頂点 (反時計回り)。辺 i は頂点 i → i+1 */
  polygon: Point[];
  /** 辺ごとの円弧 (bulge = tan(θ/4)、0 か省略は直線) */
  bulges?: number[] | null;
  /** 辺の永続 ID (辺と同じ数、重複なし)。境界条件などの辺の番号を付け替えるのに使う */
  edge_ids?: string[] | null;
  [key: string]: unknown;
}

/** 閉じた経路 (領域の穴) */
export interface Loop {
  polygon: Point[];
  bulges?: number[] | null;
}

export interface Region {
  id: string;
  type: RegionType;
  polygon?: Point[] | null;
  bulges?: number[] | null;
  /** 穴 (多角形の領域だけ。穴の中はこの領域ではない) */
  holes?: Loop[] | null;
  /** レイヤ (P7f、model/layers.ts。無ければ既定のレイヤ) */
  layer?: string | null;
  shape?: CircleShape | null;
  voltage?: number | null;
  eps_r?: number;
  rho?: number;
  [key: string]: unknown;
}

export interface BoundaryCondition {
  edges: number[];
  type: BcType;
  voltage?: number | null;
  voltage_rf?: unknown;
  voltage_waveform?: unknown;
  see_gamma?: number;
  [key: string]: unknown;
}

export interface Geometry {
  domain: Domain;
  regions: Region[];
  boundaries: BoundaryCondition[];
  [key: string]: unknown;
}

export interface MeshSettings {
  size: number;
  mode?: "unstructured" | "structured" | "cartesian";
  amr?: { max_level?: number; [key: string]: unknown } | null;
  [key: string]: unknown;
}

/** ソルバーごとの設定キー (null / 無し = 無効) */
export const SOLVER_KEYS = ["particles", "pic", "pic1d", "fluid1d", "fluid2d", "dsmc", "tl"] as const;
export type SolverKey = (typeof SOLVER_KEYS)[number];

export interface Project {
  version?: number;
  unit?: "m" | "mm";
  coord?: Coord;
  geometry: Geometry;
  mesh: MeshSettings;
  solver?: { backend?: string } | null;
  b_field?: { bx?: number; by?: number; bz?: number } | null;
  particles?: Record<string, unknown> | null;
  pic?: Record<string, unknown> | null;
  pic1d?: Record<string, unknown> | null;
  fluid1d?: Record<string, unknown> | null;
  fluid2d?: Record<string, unknown> | null;
  dsmc?: Record<string, unknown> | null;
  tl?: Record<string, unknown> | null;
  [key: string]: unknown;
}

export function coordOf(p: Project): Coord {
  return p.coord ?? "xy";
}

export interface Bounds {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export function polygonBounds(poly: Point[]): Bounds {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const [x, y] of poly) {
    x0 = Math.min(x0, x);
    y0 = Math.min(y0, y);
    x1 = Math.max(x1, x);
    y1 = Math.max(y1, y);
  }
  return { x0, y0, x1, y1 };
}

// ---- 経路 (円弧を含む輪郭) ----

export function domainPath(p: Project): PathData {
  const d = p.geometry.domain;
  return { polygon: d.polygon, bulges: d.bulges ?? null };
}

/** 領域の輪郭 (多角形の領域は頂点 + bulge、円は半円 2 つ) */
export function regionPath(r: Region): PathData {
  if (r.shape) return circlePath(r.shape.center, r.shape.radius);
  return { polygon: r.polygon ?? [], bulges: r.bulges ?? null };
}

/** 領域の穴の経路 */
export function regionHoles(r: Region): PathData[] {
  return r.shape ? [] : (r.holes ?? []).map((h) => ({ polygon: h.polygon, bulges: h.bulges ?? null }));
}

/** 領域の輪郭の輪 (外周と穴) */
export function regionRings(r: Region): PathData[] {
  return [regionPath(r), ...regionHoles(r)];
}

/** 領域の形 (外周と穴) */
export function regionShape(r: Region): Shape {
  return { outer: regionPath(r), holes: regionHoles(r) };
}

/** ドメインの外接矩形 (円弧のふくらみも含める) */
export function domainBounds(p: Project): Bounds {
  const d = p.geometry.domain;
  return d.bulges?.some((b) => b) ? pathBounds(d.polygon, d.bulges) : polygonBounds(d.polygon);
}

export function domainArea(p: Project): number {
  const d = p.geometry.domain;
  return Math.abs(pathArea(d.polygon, d.bulges));
}

/** 軸平行な矩形 (4 頂点、直線だけ、辺の順は 0 = 下・1 = 右・2 = 上・3 = 左) か */
export function isRectDomain(p: Project): boolean {
  const poly = p.geometry.domain.polygon;
  if (poly.length !== 4 || hasArcs(domainPath(p))) return false;
  const b = polygonBounds(poly);
  const expect: Point[] = [
    [b.x0, b.y0],
    [b.x1, b.y0],
    [b.x1, b.y1],
    [b.x0, b.y1],
  ];
  return poly.every(([x, y], i) => x === expect[i][0] && y === expect[i][1]);
}

/** 径方向の座標の番号 (rz: y、rz_x0: x)。xy は null */
function radialIndex(p: Project): 0 | 1 | null {
  const c = coordOf(p);
  return c === "rz" ? 1 : c === "rz_x0" ? 0 : null;
}

/**
 * 軸対称で対称軸 (r = 0) の上にある辺の番号 (両端が r = 0 の直線の辺。v1 の矩形では rz は下辺 0、rz_x0 は左辺 3)。
 * backend の _check_rz と同じ許容差。xy は空
 */
export function axisEdges(p: Project): number[] {
  const ri = radialIndex(p);
  if (ri === null) return [];
  const poly = p.geometry.domain.polygon;
  const b = bulgesOf(domainPath(p));
  const scale = Math.max(1, ...poly.map(([x, y]) => Math.max(Math.abs(x), Math.abs(y))));
  const tol = 1e-12 * scale;
  const out: number[] = [];
  for (let i = 0; i < poly.length; i++) {
    const q = poly[(i + 1) % poly.length];
    if (b[i] === 0 && Math.abs(poly[i][ri]) <= tol && Math.abs(q[ri]) <= tol) out.push(i);
  }
  return out;
}

export function isAxisEdge(p: Project, i: number): boolean {
  return axisEdges(p).includes(i);
}

export function edgeCount(p: Project): number {
  return p.geometry.domain.polygon.length;
}

/** 辺 i を含む境界条件 (無ければ null = 自然境界 (neumann)) */
export function boundaryOfEdge(p: Project, edge: number): BoundaryCondition | null {
  return p.geometry.boundaries.find((b) => b.edges.includes(edge)) ?? null;
}

// ---- 外周の辺の永続 ID ----

/** 既定の ID (e1, e2, …) */
export function defaultEdgeIds(n: number): string[] {
  return Array.from({ length: n }, (_, i) => `e${i + 1}`);
}

function validEdgeIds(ids: unknown, n: number): ids is string[] {
  return Array.isArray(ids) && ids.length === n && ids.every((x) => typeof x === "string" && x !== "") && new Set(ids).size === n;
}

/** 辺の ID (文書に無い・壊れているなら既定の ID。文書は変えない) */
export function edgeIdsOf(p: Project): string[] {
  const d = p.geometry.domain;
  return validEdgeIds(d.edge_ids, d.polygon.length) ? d.edge_ids : defaultEdgeIds(d.polygon.length);
}

/** 文書に辺の ID を入れる (無い・壊れているとき) */
export function ensureEdgeIds(p: Project): void {
  const d = p.geometry.domain;
  if (!validEdgeIds(d.edge_ids, d.polygon.length)) d.edge_ids = defaultEdgeIds(d.polygon.length);
}

/** 新しい辺の ID (e の後の番号の最大 + 1) */
export function nextEdgeId(ids: string[]): string {
  let max = 0;
  for (const id of ids) {
    const m = /^e(\d+)$/.exec(id);
    if (m) max = Math.max(max, Number(m[1]));
  }
  return `e${max + 1}`;
}

/** ID の辺の番号 (無ければ -1) */
export function edgeIndexOf(p: Project, id: string): number {
  return edgeIdsOf(p).indexOf(id);
}

export function edgeIdOf(p: Project, i: number): string {
  return edgeIdsOf(p)[i] ?? `e${i + 1}`;
}

/** 座標の端数 (0.1 + 0.2 の類) を 1 pm で丸める (保存する JSON を読みやすく保つ) */
export function tidy(v: number): number {
  return Math.round(v * 1e12) / 1e12;
}

export function tidyPoint([x, y]: Point): Point {
  return [tidy(x), tidy(y)];
}

export function polygonArea(poly: Point[]): number {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x0, y0] = poly[i];
    const [x1, y1] = poly[(i + 1) % poly.length];
    s += x0 * y1 - x1 * y0;
  }
  return Math.abs(s) / 2;
}

export function regionArea(r: Region): number {
  if (r.shape) return Math.PI * r.shape.radius ** 2;
  return Math.abs(pathArea(r.polygon ?? [], r.bulges)) - regionHoles(r).reduce((s, h) => s + Math.abs(pathArea(h.polygon, h.bulges)), 0);
}

/** 既存と重ならない領域 ID (region1, region2, …) */
export function uniqueRegionId(p: Project, base = "region"): string {
  const used = new Set(p.geometry.regions.map((r) => r.id));
  for (let i = 1; ; i++) {
    const id = `${base}${i}`;
    if (!used.has(id)) return id;
  }
}

export class ProjectFormatError extends Error {}

/**
 * 読み込んだ JSON をプロジェクト文書に整える (v1 の applyLoadedProject と同じ補完)。
 * 戻り値の results は「結果付き保存」の結果部分 (P6e で使う)。
 */
export function normalizeProject(raw: unknown): { project: Project; results: unknown } {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new ProjectFormatError("プロジェクトファイルではありません (JSON のオブジェクトではない)");
  }
  const src = raw as Record<string, unknown>;
  const geom = src.geometry as Record<string, unknown> | undefined;
  if (!geom || typeof geom !== "object") {
    throw new ProjectFormatError("プロジェクトファイルではありません (geometry がありません)");
  }
  const domain = geom.domain as { polygon?: unknown; bulges?: unknown } | undefined;
  // 直線だけなら 3 点、円弧を含めば 2 点から
  const minPts = Array.isArray(domain?.bulges) && domain.bulges.some((b) => typeof b === "number" && b !== 0) ? 2 : 3;
  if (!domain || !Array.isArray(domain.polygon) || domain.polygon.length < minPts) {
    throw new ProjectFormatError("geometry.domain.polygon (3 点以上) がありません");
  }
  const { results, ...rest } = src;
  const project = structuredClone(rest) as Project;
  project.version ??= 1;
  project.unit ??= "m";
  project.geometry.regions ??= [];
  project.geometry.boundaries ??= [];
  ensureEdgeIds(project);
  // パラメータが正: 束縛した欄を計算した値にする (手で書き換えた JSON にも合わせる、P7f)
  applyParams(project);
  const mesh = (project.mesh ?? {}) as MeshSettings;
  if (!(typeof mesh.size === "number" && mesh.size > 0)) {
    const b = domainBounds(project);
    mesh.size = Math.max(b.x1 - b.x0, b.y1 - b.y0) / 25;
  }
  project.mesh = mesh;
  // 旧形式: pic.collector (単一) → pic.collectors (v1 と同じ移行)
  const pic = project.pic as Record<string, unknown> | null | undefined;
  // collectors が空のときも移し、ラベルが無ければ C1 (v1 と同じ)
  if (pic && pic.collector && (!Array.isArray(pic.collectors) || pic.collectors.length === 0)) {
    const c = pic.collector as Record<string, unknown>;
    pic.collectors = [{ label: "C1", ...c }];
    delete pic.collector;
  }
  return { project, results: results ?? null };
}

/** 保存用の JSON (v1 の保存と同じく整形して書く) */
export function serializeProject(p: Project): string {
  return JSON.stringify(p, null, 2) + "\n";
}

/** 新規プロジェクト: 100 × 50 mm、左 0 V・右 100 V、メッシュ 4 mm */
export function newProject(): Project {
  return {
    version: 1,
    unit: "m",
    coord: "xy",
    geometry: {
      domain: {
        polygon: [
          [0, 0],
          [0.1, 0],
          [0.1, 0.05],
          [0, 0.05],
        ],
        edge_ids: defaultEdgeIds(4),
      },
      regions: [],
      boundaries: [
        { edges: [3], type: "dirichlet", voltage: 0 },
        { edges: [1], type: "dirichlet", voltage: 100 },
      ],
    },
    mesh: { size: 0.004 },
  };
}
