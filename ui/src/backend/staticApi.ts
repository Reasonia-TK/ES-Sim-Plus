// 静電場の REST (同期): メッシュの作成・求解・線上の分布 (v1 と同じエンドポイント)。

import type { Point, Project } from "../model/project";
import { apiPost } from "./api";

export interface MeshResult {
  nodes: Point[];
  triangles: [number, number, number][];
  region_of_triangle: number[];
}

export interface ElectrodeCharge {
  /** "edge0"… (外周の辺) か導体の領域 ID */
  label: string;
  voltage: number;
  /** xy: C/m、軸対称: C */
  q: number;
}

export interface SolveResult {
  mesh: MeshResult;
  v: number[];
  e_field: [number, number][];
  v_min: number;
  v_max: number;
  e_abs_max: number;
  energy: number;
  charges: ElectrodeCharge[];
  capacitance: number | null;
}

export interface ProfileResult {
  s: number[];
  v: (number | null)[];
  e_abs: (number | null)[];
}

export const buildMesh = (project: Project, signal?: AbortSignal) => apiPost<MeshResult>("/mesh", project, { signal });

export const solveStatic = (project: Project, signal?: AbortSignal) => apiPost<SolveResult>("/solve", project, { signal });

export const lineProfile = (project: Project, p1: Point, p2: Point, n = 200, signal?: AbortSignal) =>
  apiPost<ProfileResult>("/profile", { project, p1, p2, n }, { signal });
