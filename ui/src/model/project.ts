// プロジェクト文書 (backend の es_sim.schema.Project と同じ JSON) の型と読み書きの補助。
// P6a では木構造・ジオメトリ表示・ファイル操作に要る部分だけを型にし、各ソルバーの設定の中身は
// P6b でスキーマ (/v2/schema) から扱う。長さは常に m で持つ (表示単位への変換は UI の端で)。

export type Point = [number, number];
export type Coord = "xy" | "rz" | "rz_x0";
export type RegionType = "conductor" | "dielectric" | "charge";
export type BcType = "dirichlet" | "neumann" | "symmetry" | "periodic";

export interface CircleShape {
  kind: "circle";
  center: Point;
  radius: number;
}

export interface Region {
  id: string;
  type: RegionType;
  polygon?: Point[] | null;
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
  domain: { polygon: Point[] };
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

/** 軸平行な矩形 (4 頂点、辺の順は 0 = 下・1 = 右・2 = 上・3 = 左) か */
export function isRectDomain(p: Project): boolean {
  const poly = p.geometry.domain.polygon;
  if (poly.length !== 4) return false;
  const b = polygonBounds(poly);
  const expect: Point[] = [
    [b.x0, b.y0],
    [b.x1, b.y0],
    [b.x1, b.y1],
    [b.x0, b.y1],
  ];
  return poly.every(([x, y], i) => x === expect[i][0] && y === expect[i][1]);
}

/** 軸対称で対称軸になる辺の番号 (rz: 下辺 0、rz_x0: 左辺 3)。xy・矩形でない domain は null */
export function axisEdge(p: Project): number | null {
  if (!isRectDomain(p)) return null;
  const c = coordOf(p);
  return c === "rz" ? 0 : c === "rz_x0" ? 3 : null;
}

export function edgeCount(p: Project): number {
  return p.geometry.domain.polygon.length;
}

/** 辺 i を含む境界条件 (無ければ null = 自然境界 (neumann)) */
export function boundaryOfEdge(p: Project, edge: number): BoundaryCondition | null {
  return p.geometry.boundaries.find((b) => b.edges.includes(edge)) ?? null;
}

export function regionOutline(r: Region, n = 64): Point[] {
  if (r.shape) {
    const { center, radius } = r.shape;
    return Array.from({ length: n }, (_, i) => {
      const a = (i / n) * 2 * Math.PI;
      return [center[0] + radius * Math.cos(a), center[1] + radius * Math.sin(a)] as Point;
    });
  }
  return r.polygon ?? [];
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
  return r.shape ? Math.PI * r.shape.radius ** 2 : polygonArea(r.polygon ?? []);
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
  const domain = geom.domain as { polygon?: unknown } | undefined;
  if (!domain || !Array.isArray(domain.polygon) || domain.polygon.length < 3) {
    throw new ProjectFormatError("geometry.domain.polygon (3 点以上) がありません");
  }
  const { results, ...rest } = src;
  const project = structuredClone(rest) as Project;
  project.version ??= 1;
  project.unit ??= "m";
  project.geometry.regions ??= [];
  project.geometry.boundaries ??= [];
  const mesh = (project.mesh ?? {}) as MeshSettings;
  if (!(typeof mesh.size === "number" && mesh.size > 0)) {
    const b = polygonBounds(project.geometry.domain.polygon);
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
