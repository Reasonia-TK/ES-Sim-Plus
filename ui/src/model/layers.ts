// レイヤ (P7f、prompts/132): 領域とスケッチの表示・ロック・色。文書の project.cad.layers (id・名前・色・表示・ロック)
// と project.cad.active_layer (今のレイヤ、DXF の $CLAYER と同じく文書に持つ)。各領域・スケッチは layer (レイヤの id、
// 無ければ既定のレイヤ "0")。
// - 表示しないレイヤの形は描かず、選べず、スナップ・囲まれた所・トリムの相手にもしない (計算には使う)
// - ロックしたレイヤの形は描くが選べない (スナップ・囲まれた所・トリムの相手にはなる)
// - 新しく作った形 (前の文書に無く layer を持たない) は、編集の後で今のレイヤに入る (documentStore.update)。
//   コピーは元のレイヤのまま (keepLayer)
// - 色はスケッチの線の色 (領域は種類の色のまま)

import type { Draft } from "immer";
import type { PickRef } from "./selection";
import { sketchOf, type CadData, type SketchEntity } from "./sketch";
import type { Project, Region } from "./project";

export const DEFAULT_LAYER = "0";

export interface Layer {
  id: string;
  name: string;
  /** 線の色 (#rrggbb、無ければ既定の色) */
  color?: string | null;
  /** 表示するか (無ければ表示) */
  visible?: boolean;
  /** ロック (選べない) */
  locked?: boolean;
}

interface LayerData extends CadData {
  layers?: Layer[];
  active_layer?: string;
}

function cadOf(p: Project): LayerData {
  return (p.cad as LayerData | null | undefined) ?? {};
}

/** レイヤの一覧 (既定のレイヤ "0" はいつも先頭にある) */
export function layersOf(p: Project): Layer[] {
  const list = Array.isArray(cadOf(p).layers) ? (cadOf(p).layers as Layer[]) : [];
  const def = list.find((l) => l.id === DEFAULT_LAYER) ?? { id: DEFAULT_LAYER, name: DEFAULT_LAYER };
  return [def, ...list.filter((l) => l.id !== DEFAULT_LAYER)];
}

export function layerOf(p: Project, id: string | null | undefined): Layer {
  const all = layersOf(p);
  return all.find((l) => l.id === (id ?? DEFAULT_LAYER)) ?? all[0];
}

/** 今のレイヤ (無い id なら既定) */
export function activeLayer(p: Project): string {
  const id = cadOf(p).active_layer;
  return id && layersOf(p).some((l) => l.id === id) ? id : DEFAULT_LAYER;
}

type Layered = { layer?: string | null };

/** 形のレイヤの id (無い・知らないレイヤは既定) */
export function itemLayer(p: Project, item: Layered): string {
  const id = item.layer ?? DEFAULT_LAYER;
  return layersOf(p).some((l) => l.id === id) ? id : DEFAULT_LAYER;
}

export function isVisible(p: Project, item: Layered): boolean {
  return layerOf(p, itemLayer(p, item)).visible !== false;
}

/** 選べる (表示していて、ロックしていない) */
export function isPickable(p: Project, item: Layered): boolean {
  const l = layerOf(p, itemLayer(p, item));
  return l.visible !== false && !l.locked;
}

/** 表示する領域・スケッチ */
export function visibleRegions(p: Project): Region[] {
  return p.geometry.regions.filter((r) => isVisible(p, r));
}

export function visibleSketch(p: Project): SketchEntity[] {
  return sketchOf(p).filter((e) => isVisible(p, e as Layered));
}

/** 選べる領域・スケッチ */
export function pickableRegions(p: Project): Region[] {
  return p.geometry.regions.filter((r) => isPickable(p, r));
}

export function pickableSketch(p: Project): SketchEntity[] {
  return sketchOf(p).filter((e) => isPickable(p, e as Layered));
}

/** スケッチの線の色 (レイヤの色、無ければ null = 既定の色) */
export function layerColor(p: Project, item: Layered): string | null {
  return layerOf(p, itemLayer(p, item)).color ?? null;
}

// ---- 文書の編集 ----

function draftCad(d: Draft<Project>): LayerData {
  const p = d as Project;
  if (!p.cad || typeof p.cad !== "object") (p as Record<string, unknown>).cad = {};
  return p.cad as LayerData;
}

function draftLayers(d: Draft<Project>): Layer[] {
  const cad = draftCad(d);
  if (!Array.isArray(cad.layers)) cad.layers = [];
  if (!cad.layers.some((l) => l.id === DEFAULT_LAYER)) cad.layers.unshift({ id: DEFAULT_LAYER, name: DEFAULT_LAYER });
  return cad.layers;
}

/** 新しいレイヤ (id は L1, L2, …、名前も同じ)。id を返す */
export function addLayer(d: Draft<Project>): string {
  const list = draftLayers(d);
  let k = 1;
  while (list.some((l) => l.id === `L${k}`)) k++;
  const id = `L${k}`;
  list.push({ id, name: id });
  return id;
}

export function updateLayer(d: Draft<Project>, id: string, patch: Partial<Omit<Layer, "id">>): void {
  const l = draftLayers(d).find((x) => x.id === id);
  if (!l) return;
  for (const [k, v] of Object.entries(patch) as [keyof Layer, unknown][]) {
    // 既定のレイヤの名前は変えない (DXF の "0" と同じ)
    if (k === "name" && id === DEFAULT_LAYER) continue;
    if (v === undefined || v === null || v === "" || (k === "visible" && v === true) || (k === "locked" && v === false)) delete (l as unknown as Record<string, unknown>)[k];
    else (l as unknown as Record<string, unknown>)[k] = v;
  }
}

/** レイヤの名前の検査 (エラーの翻訳キー、正しければ null) */
export function layerNameError(p: Project, id: string, name: string): "layers.nameEmpty" | "layers.nameDuplicate" | null {
  const s = name.trim();
  if (!s) return "layers.nameEmpty";
  if (layersOf(p).some((l) => l.id !== id && l.name === s)) return "layers.nameDuplicate";
  return null;
}

export function setActiveLayer(d: Draft<Project>, id: string): void {
  const cad = draftCad(d);
  if (id === DEFAULT_LAYER) delete cad.active_layer;
  else cad.active_layer = id;
}

/** レイヤを消す (既定は消さない)。その形は既定のレイヤへ */
export function deleteLayer(d: Draft<Project>, id: string): void {
  if (id === DEFAULT_LAYER) return;
  const cad = draftCad(d);
  cad.layers = draftLayers(d).filter((l) => l.id !== id);
  if (cad.active_layer === id) delete cad.active_layer;
  for (const r of d.geometry.regions) if (r.layer === id) delete r.layer;
  for (const e of sketchOf(d as Project)) if ((e as Layered).layer === id) delete (e as Layered).layer;
}

/** 形をレイヤへ (既定のレイヤは明示して "0"、今のレイヤに入れ直されないように) */
export function setItemLayer(d: Draft<Project>, item: PickRef, id: string): void {
  const target =
    item.kind === "region" ? (d.geometry.regions.find((r) => r.id === item.id) as Layered | undefined) : (sketchOf(d as Project).find((e) => e.id === item.id) as Layered | undefined);
  if (target) target.layer = id;
}

/** レイヤごとの形の数 */
export function layerCounts(p: Project): Map<string, number> {
  const out = new Map<string, number>();
  const add = (item: Layered) => {
    const id = itemLayer(p, item);
    out.set(id, (out.get(id) ?? 0) + 1);
  };
  p.geometry.regions.forEach(add);
  sketchOf(p).forEach((e) => add(e as Layered));
  return out;
}

/** コピーした形のレイヤ (元に無ければ既定を明示) */
export function keepLayer<T extends Layered>(src: Layered, copy: T): T {
  copy.layer = src.layer ?? DEFAULT_LAYER;
  return copy;
}

/**
 * 編集の後に: 前の文書に無く layer を持たない領域・スケッチ (新しく作った形) を今のレイヤへ。
 * 今のレイヤが既定なら何もしない (layer を持たないまま)
 */
export function assignNewItemsToLayer(d: Draft<Project>, base: Project): void {
  const p = d as Project;
  const active = activeLayer(p);
  if (active === DEFAULT_LAYER) return;
  const regions = new Set(base.geometry.regions.map((r) => r.id));
  for (const r of d.geometry.regions) if (r.layer === undefined && !regions.has(r.id)) r.layer = active;
  const sketch = new Set(sketchOf(base).map((e) => e.id));
  for (const e of sketchOf(p)) if ((e as Layered).layer === undefined && !sketch.has(e.id)) (e as Layered).layer = active;
}
