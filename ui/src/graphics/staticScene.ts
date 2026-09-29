// 静電場の結果 → ビューアの Scene。メッシュを作り直したらメッシュを、そうでなければ求解の結果 (V か |E|、
// 等値線は表示している量によらず電位、矢印は E) を出す (v1 と同じ優先順位)。

import type { Project } from "../model/project";
import { isStale, meshKey, solveKey, type MeshEntry, type SolveEntry } from "../results/staticResults";
import { EMPTY_SCENE, type Scene, type ScalarField } from "./scene";
import type { StaticQuantity } from "./viewerStore";

export interface StaticSource {
  mesh: MeshEntry | null;
  solve: SolveEntry | null;
  latest: "mesh" | "solve" | null;
}

export interface SceneLabels {
  meshTitle: (nodes: number, elements: number) => string;
  potential: string;
  field: string;
}

const potentialCache = new WeakMap<SolveEntry, ScalarField>();
const fieldCache = new WeakMap<SolveEntry, ScalarField>();

/** 電位 (節点、backend の最小・最大をそのまま使う) */
export function potentialOf(s: SolveEntry): ScalarField {
  let f = potentialCache.get(s);
  if (!f) {
    f = { mesh: s.view, values: s.result.v, location: "node", label: "V", unit: "V" };
    potentialCache.set(s, f);
  }
  return f;
}

/** |E| (三角形ごと) */
export function fieldAbsOf(s: SolveEntry): ScalarField {
  let f = fieldCache.get(s);
  if (!f) {
    f = { mesh: s.view, values: s.eAbs, location: "element", label: "|E|", unit: "V/m" };
    fieldCache.set(s, f);
  }
  return f;
}

export function staticScene(src: StaticSource, quantity: StaticQuantity, project: Project, labels: SceneLabels): Scene {
  const { mesh, solve, latest } = src;
  if (mesh && (latest === "mesh" || !solve)) {
    return {
      ...EMPTY_SCENE,
      title: labels.meshTitle(mesh.result.nodes.length, mesh.result.triangles.length),
      stale: isStale(mesh, meshKey, project),
      mesh: mesh.view,
      fillRegions: true,
    };
  }
  if (solve) {
    const v = potentialOf(solve);
    const field = quantity === "v" ? v : fieldAbsOf(solve);
    return {
      ...EMPTY_SCENE,
      title: quantity === "v" ? labels.potential : labels.field,
      stale: isStale(solve, solveKey, project),
      mesh: solve.view,
      field,
      iso: v,
      vectors: { mesh: solve.view, values: solve.result.e_field },
    };
  }
  return EMPTY_SCENE;
}
