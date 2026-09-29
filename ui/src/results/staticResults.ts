// 静電場の結果 (メッシュの作成・求解) のストア。v1 は設定を変えるたびに結果を消したが、v2 は結果を残して
// 「設定が変わっている」ことを示す (計算に使った設定の要約と今の設定を比べる)。別の文書を開いたら消す。

import { create } from "zustand";
import { errorText, logError, logInfo } from "../app/messages";
import { buildMesh, solveStatic, type MeshResult, type SolveResult } from "../backend/staticApi";
import { t } from "../i18n";
import { useDocument } from "../model/documentStore";
import type { Project } from "../model/project";
import type { ViewMesh } from "../graphics/scene";

/** メッシュに効く設定の要約 */
export function meshKey(p: Project): string {
  return JSON.stringify([
    p.coord ?? "xy",
    p.geometry.domain.polygon,
    p.geometry.regions.map((r) => [r.id, r.type, r.polygon ?? null, r.shape ?? null]),
    p.mesh,
  ]);
}

/** 求解に効く設定の要約 */
export function solveKey(p: Project): string {
  return JSON.stringify([p.coord ?? "xy", p.geometry, p.mesh, p.solver ?? null]);
}

export interface MeshEntry {
  result: MeshResult;
  view: ViewMesh;
  key: string;
  elapsedS: number;
}

export interface SolveEntry {
  result: SolveResult;
  /** 求解に使った設定 (プロファイルを同じ解で取るため) */
  project: Project;
  view: ViewMesh;
  /** 三角形ごとの |E| */
  eAbs: Float64Array;
  key: string;
  elapsedS: number;
}

type Busy = "mesh" | "solve" | null;

interface StaticState {
  mesh: MeshEntry | null;
  solve: SolveEntry | null;
  /** 最後に作ったもの (メッシュを作り直したらメッシュを優先して表示する、v1 と同じ) */
  latest: "mesh" | "solve" | null;
  busy: Busy;
  error: string | null;
  runMesh: () => Promise<void>;
  runSolve: () => Promise<void>;
  cancel: () => void;
  clear: () => void;
}

export function toViewMesh(m: MeshResult): ViewMesh {
  return { nodes: m.nodes, triangles: m.triangles, regionOfTriangle: m.region_of_triangle };
}

function elementAbs(e: [number, number][]): Float64Array {
  const out = new Float64Array(e.length);
  e.forEach(([x, y], i) => (out[i] = Math.hypot(x, y)));
  return out;
}

let controller: AbortController | null = null;

function isAbort(e: unknown): boolean {
  return e instanceof DOMException && e.name === "AbortError";
}

export const useStatic = create<StaticState>()((set, get) => ({
  mesh: null,
  solve: null,
  latest: null,
  busy: null,
  error: null,

  runMesh: async () => {
    if (get().busy) return;
    const project = useDocument.getState().project;
    controller = new AbortController();
    set({ busy: "mesh", error: null });
    const t0 = performance.now();
    try {
      const result = await buildMesh(project, controller.signal);
      const elapsedS = (performance.now() - t0) / 1000;
      set({ mesh: { result, view: toViewMesh(result), key: meshKey(project), elapsedS }, latest: "mesh" });
      logInfo(t("msg.source.static"), t("static.meshDone", { nodes: result.nodes.length, elements: result.triangles.length, s: elapsedS.toFixed(2) }));
    } catch (e) {
      if (isAbort(e)) return;
      const msg = errorText(e);
      set({ error: msg });
      logError(t("msg.source.static"), t("static.meshFailed", { error: msg }));
    } finally {
      controller = null;
      set({ busy: null });
    }
  },

  runSolve: async () => {
    if (get().busy) return;
    const project = useDocument.getState().project;
    controller = new AbortController();
    set({ busy: "solve", error: null });
    const t0 = performance.now();
    try {
      const result = await solveStatic(project, controller.signal);
      const elapsedS = (performance.now() - t0) / 1000;
      set({
        solve: { result, project, view: toViewMesh(result.mesh), eAbs: elementAbs(result.e_field), key: solveKey(project), elapsedS },
        // 求解したらメッシュだけの表示はやめる (v1 と同じ)
        mesh: null,
        latest: "solve",
      });
      logInfo(t("msg.source.static"), t("static.solveDone", { nodes: result.mesh.nodes.length, elements: result.mesh.triangles.length, s: elapsedS.toFixed(2) }));
    } catch (e) {
      if (isAbort(e)) return;
      const msg = errorText(e);
      set({ error: msg });
      logError(t("msg.source.static"), t("static.solveFailed", { error: msg }));
    } finally {
      controller = null;
      set({ busy: null });
    }
  },

  cancel: () => {
    controller?.abort();
    controller = null;
    set({ busy: null });
  },

  clear: () => {
    controller?.abort();
    controller = null;
    set({ mesh: null, solve: null, latest: null, busy: null, error: null });
  },
}));

// 別の文書を開いたら (新規・開く・サンプル・復元) 結果を消す
useDocument.subscribe((s, prev) => {
  if (s.docSerial !== prev.docSerial) useStatic.getState().clear();
});

/** 今の設定と計算に使った設定が違うか */
export function isStale(entry: { key: string } | null, key: (p: Project) => string, project: Project): boolean {
  return entry !== null && entry.key !== key(project);
}
