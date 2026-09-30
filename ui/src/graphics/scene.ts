// ビューアに渡すデータの形 (静電場の結果・メッシュ、P6e で PIC・流体・DSMC の結果も同じ形で渡す)。
// メッシュ・値の配列は同じオブジェクトである限り GPU に送り直さない (参照で見分ける)。

import type { Point } from "../model/project";
import { fieldStats, type FieldStats } from "./fieldScale";
import { MeshIndex, type TriMesh } from "./meshIndex";

export interface ViewMesh extends TriMesh {
  /** 三角形 → 領域の番号 (geometry.regions の添字、-1 は背景) */
  regionOfTriangle?: number[];
}

export type FieldLocation = "node" | "element";

export interface ScalarField {
  mesh: ViewMesh;
  /** 節点の値 (location = node) か三角形の値 (element) */
  values: ArrayLike<number>;
  location: FieldLocation;
  /** カラーバーの見出し (V・|E| など) */
  label: string;
  unit: string;
  /** 分かっていれば (backend が返した最小・最大など)。無ければ値から求める */
  stats?: FieldStats;
}

/** 三角形ごとのベクトル (電場など、向きの矢印を描く) */
export interface VectorField {
  mesh: ViewMesh;
  values: [number, number][];
}

/** 点の集まり (粒子) */
export interface PointSet {
  id: string;
  /** x0, y0, x1, y1, … [m] */
  xy: ArrayLike<number>;
  color: string;
  /** 直径 [px] */
  size: number;
}

/** 線分の集まり (等値線・軌道・シース端など) */
export interface SegmentSet {
  id: string;
  /** x0, y0, x1, y1 を線分ごとに [m] */
  segments: ArrayLike<number>;
  color: string;
  /** 幅 [px] */
  width: number;
}

export interface Scene {
  /** 見出し (左上に出す) */
  title: string;
  /** 設定が結果の計算後に変わっている */
  stale: boolean;
  /** メッシュの線・領域の塗り (メッシュだけの表示) に使う */
  mesh: ViewMesh | null;
  /** 領域の色で三角形を塗る (メッシュだけの表示) */
  fillRegions: boolean;
  field: ScalarField | null;
  /** 等値線を引く節点の量 (静電場は表示している量によらず電位) */
  iso: ScalarField | null;
  vectors: VectorField | null;
  points: PointSet[];
  segments: SegmentSet[];
}

export const EMPTY_SCENE: Scene = {
  title: "",
  stale: false,
  mesh: null,
  fillRegions: false,
  field: null,
  iso: null,
  vectors: null,
  points: [],
  segments: [],
};

/** 折れ線の集まり → 線分の並び (x0, y0, x1, y1, …) */
export function pathsToSegments(paths: Point[][]): Float64Array {
  let n = 0;
  for (const p of paths) n += Math.max(0, p.length - 1);
  const out = new Float64Array(4 * n);
  let k = 0;
  for (const p of paths) {
    for (let i = 1; i < p.length; i++) {
      out[k++] = p[i - 1][0];
      out[k++] = p[i - 1][1];
      out[k++] = p[i][0];
      out[k++] = p[i][1];
    }
  }
  return out;
}

// ---- メッシュ・場から作るもの (同じオブジェクトには 1 回だけ) ----

const indexCache = new WeakMap<TriMesh, MeshIndex>();
const statsCache = new WeakMap<object, FieldStats>();

export function meshIndexOf(mesh: TriMesh): MeshIndex {
  let idx = indexCache.get(mesh);
  if (!idx) {
    idx = new MeshIndex(mesh);
    indexCache.set(mesh, idx);
  }
  return idx;
}

export function statsOf(f: ScalarField): FieldStats {
  if (f.stats) return f.stats;
  const key = f.values as object;
  let s = statsCache.get(key);
  if (!s) {
    s = fieldStats(f.values);
    statsCache.set(key, s);
  }
  return s;
}

/** 点での値: 節点の量は三角形の中で線形補間、三角形の量はその三角形の値。メッシュの外は null */
export function sampleField(f: ScalarField, x: number, y: number): number | null {
  const idx = meshIndexOf(f.mesh);
  const hit = idx.locate(x, y);
  if (!hit) return null;
  if (f.location === "element") return f.values[hit.tri];
  const [a, b, c] = f.mesh.triangles[hit.tri];
  return hit.w[0] * f.values[a] + hit.w[1] * f.values[b] + hit.w[2] * f.values[c];
}

/** 線分の上の値 (端を含む n 点、s は始点からの距離 [m]。メッシュの外は null) */
export function sampleLine(f: ScalarField, a: [number, number], b: [number, number], n = 200): { s: number[]; v: (number | null)[] } {
  const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
  const s: number[] = [];
  const v: (number | null)[] = [];
  for (let i = 0; i < n; i++) {
    const t = n === 1 ? 0 : i / (n - 1);
    s.push(len * t);
    v.push(sampleField(f, a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t));
  }
  return { s, v };
}

/** 点でのベクトル (その点を含む三角形の値)。メッシュの外は null */
export function sampleVector(v: VectorField, x: number, y: number): [number, number] | null {
  const hit = meshIndexOf(v.mesh).locate(x, y);
  return hit ? v.values[hit.tri] : null;
}
