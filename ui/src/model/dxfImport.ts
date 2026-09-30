// DXF の読み込みの結果を文書に (CAD v2 P7g、prompts/132)。形の解析はバックエンド (es_sim.dxf.read_dxf) が行い、
// 座標は m で返る。ここでは単位の読み替え (ファイルに単位が無く mm とみなしたときなど)、DXF のレイヤを UI のレイヤに
// (同じ名前があればそれ、無ければ作る。レイヤ 0 は既定のレイヤ)、形をスケッチに (閉じた形は領域にもできる) する。

import type { Draft } from "immer";
import { shapeProblem } from "../cad/boolean";
import { activeLayer, addLayer, DEFAULT_LAYER, layersOf, setItemLayer, updateLayer } from "./layers";
import type { Point, Project } from "./project";
import { addCircleRegionAt, addPolygonRegion } from "./regionOps";
import { addSketch, type NewSketch } from "./sketch";

export type DxfEntity =
  | { kind: "line"; a: Point; b: Point; layer: string }
  | { kind: "arc"; a: Point; b: Point; bulge: number; layer: string }
  | { kind: "circle"; center: Point; r: number; layer: string }
  | { kind: "polyline"; points: Point[]; bulges: number[]; closed: boolean; layer: string };

export interface DxfLayer {
  name: string;
  color: string | null;
  visible: boolean;
}

export interface DxfImport {
  /** 読み替えに使った単位 (mm・cm・m・um・in など) */
  unit: string;
  /** ファイルに単位が無く mm とみなした */
  assumed: boolean;
  entities: DxfEntity[];
  layers: DxfLayer[];
  counts: Record<string, number>;
  /** 形でないため飛ばしたものの数 (DXF の種類ごと) */
  skipped: Record<string, number>;
}

export interface DxfImportOptions {
  /** 図の単位 (結果の単位と違えば読み替える) */
  unit: string;
  /** DXF のレイヤを取り込む (しなければ全て今のレイヤ) */
  layers: boolean;
  /** 閉じた形 (閉じたポリライン・円) を領域にする */
  toRegions: boolean;
}

/** 読み込める単位と 1 単位の m */
export const DXF_UNITS: Record<string, number> = { mm: 1e-3, cm: 1e-2, m: 1, um: 1e-6, in: 0.0254, ft: 0.3048 };

/** 単位の読み替え (座標と半径を factor 倍) */
export function scaleImport(imp: DxfImport, unit: string): DxfImport {
  const from = DXF_UNITS[imp.unit];
  const to = DXF_UNITS[unit];
  if (from === undefined || to === undefined || from === to) return imp;
  const k = to / from;
  const p = (q: Point): Point => [q[0] * k, q[1] * k];
  const entities = imp.entities.map((e): DxfEntity => {
    switch (e.kind) {
      case "line":
        return { ...e, a: p(e.a), b: p(e.b) };
      case "arc":
        return { ...e, a: p(e.a), b: p(e.b) };
      case "circle":
        return { ...e, center: p(e.center), r: e.r * k };
      case "polyline":
        return { ...e, points: e.points.map(p) };
    }
  });
  return { ...imp, unit, assumed: false, entities };
}

function toSketch(e: DxfEntity): NewSketch {
  switch (e.kind) {
    case "line":
      return { kind: "line", a: e.a, b: e.b };
    case "arc":
      return { kind: "arc", a: e.a, b: e.b, bulge: e.bulge };
    case "circle":
      return { kind: "circle", center: e.center, r: e.r };
    case "polyline":
      return { kind: "polyline", points: e.points, bulges: e.bulges, closed: e.closed };
  }
}

export interface DxfApplied {
  sketch: number;
  regions: number;
  layers: number;
}

/** 読み込んだ形を文書に足す (1 回の編集で) */
export function applyDxfImport(d: Draft<Project>, raw: DxfImport, opts: DxfImportOptions): DxfApplied {
  const imp = scaleImport(raw, opts.unit);
  const out: DxfApplied = { sketch: 0, regions: 0, layers: 0 };
  const byName = new Map<string, string>();
  if (opts.layers) {
    for (const l of imp.layers) {
      if (l.name === DEFAULT_LAYER) {
        byName.set(l.name, DEFAULT_LAYER);
        continue;
      }
      const same = layersOf(d as Project).find((x) => x.name === l.name);
      if (same) {
        byName.set(l.name, same.id);
        continue;
      }
      const id = addLayer(d);
      updateLayer(d, id, { name: l.name, color: l.color, visible: l.visible });
      byName.set(l.name, id);
      out.layers++;
    }
  }
  const current = activeLayer(d as Project);
  const layerOf = (name: string) => (opts.layers ? (byName.get(name) ?? DEFAULT_LAYER) : current);
  for (const e of imp.entities) {
    const layer = layerOf(e.layer);
    if (opts.toRegions && (e.kind === "circle" || (e.kind === "polyline" && e.closed))) {
      // 自分と交わる閉じた形は領域にできないのでスケッチのまま
      const ok = e.kind === "circle" || shapeProblem({ outer: { polygon: e.points, bulges: e.bulges }, holes: [] }) === null;
      if (ok) {
        const rid = e.kind === "circle" ? addCircleRegionAt(d, e.center, e.r) : addPolygonRegion(d, e.points, e.bulges);
        if (rid) {
          setItemLayer(d, { kind: "region", id: rid }, layer);
          out.regions++;
          continue;
        }
      }
    }
    const id = addSketch(d, toSketch(e));
    if (id) {
      setItemLayer(d, { kind: "sketch", id }, layer);
      out.sketch++;
    }
  }
  return out;
}
