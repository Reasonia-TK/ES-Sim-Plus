// 領域の追加・削除・名前変更・複製・種類の変更 (v1 と同じ整合性: mesh.local_sizes の参照を保つ、
// 種類を変えても形と対応する値は保ち、RF・CSV 波形・γ は外す)。

import type { Draft } from "immer";
import { polygonBounds, tidy, tidyPoint, uniqueRegionId, type Point, type Project, type Region, type RegionType } from "./project";

type P = Draft<Project>;

interface LocalSize {
  region: string;
  size: number;
}

function localSizes(p: P): LocalSize[] | undefined {
  return p.mesh.local_sizes as LocalSize[] | undefined;
}

/** ドメインの中央に幅・高さの 1/4 の矩形 (導体・0 V、v1 の新規領域と同じ既定値) */
export function addRectRegion(p: P): string {
  const b = polygonBounds(p.geometry.domain.polygon);
  const cx = (b.x0 + b.x1) / 2;
  const cy = (b.y0 + b.y1) / 2;
  const hw = (b.x1 - b.x0) / 8;
  const hh = (b.y1 - b.y0) / 8;
  const id = uniqueRegionId(p as Project);
  p.geometry.regions.push({
    id,
    type: "conductor",
    voltage: 0,
    polygon: [
      [cx - hw, cy - hh],
      [cx + hw, cy - hh],
      [cx + hw, cy + hh],
      [cx - hw, cy + hh],
    ],
  });
  return id;
}

/** ドメインの中央に短辺の 1/8 の半径の円 */
export function addCircleRegion(p: P): string {
  const b = polygonBounds(p.geometry.domain.polygon);
  const id = uniqueRegionId(p as Project);
  p.geometry.regions.push({
    id,
    type: "conductor",
    voltage: 0,
    shape: {
      kind: "circle",
      center: [(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2],
      radius: Math.min(b.x1 - b.x0, b.y1 - b.y0) / 8,
    },
  });
  return id;
}

export function deleteRegion(p: P, id: string): void {
  p.geometry.regions = p.geometry.regions.filter((r) => r.id !== id);
  const ls = localSizes(p);
  if (ls) p.mesh.local_sizes = ls.filter((l) => l.region !== id);
}

export function renameRegion(p: P, from: string, to: string): void {
  const r = p.geometry.regions.find((x) => x.id === from);
  if (!r) return;
  r.id = to;
  for (const l of localSizes(p) ?? []) if (l.region === from) l.region = to;
}

/** 複製 (ドメイン幅の 5% だけ右上へずらす) */
export function duplicateRegion(p: P, id: string): string | null {
  const src = p.geometry.regions.find((x) => x.id === id);
  if (!src) return null;
  const b = polygonBounds(p.geometry.domain.polygon);
  const d = 0.05 * Math.min(b.x1 - b.x0, b.y1 - b.y0);
  const copy = JSON.parse(JSON.stringify(src)) as Region; // Immer の draft は structuredClone できない
  copy.id = uniqueRegionId(p as Project, `${id}_`);
  if (copy.shape) copy.shape.center = [copy.shape.center[0] + d, copy.shape.center[1] + d];
  if (copy.polygon) copy.polygon = copy.polygon.map(([x, y]) => [x + d, y + d]);
  p.geometry.regions.push(copy);
  return copy.id;
}

/** 種類の変更: 形と対応する値は保ち、RF・CSV 波形・γ は外す (v1 と同じ) */
export function setRegionType(p: P, id: string, type: RegionType): void {
  const r = p.geometry.regions.find((x) => x.id === id);
  if (!r || r.type === type) return;
  r.type = type;
  delete r.voltage_rf;
  delete r.voltage_waveform;
  delete r.see_gamma;
  if (type === "conductor" && typeof r.voltage !== "number") r.voltage = 0;
  if (type === "dielectric" && typeof r.eps_r !== "number") r.eps_r = 1;
  if (type === "charge" && typeof r.rho !== "number") r.rho = 0;
}

/** ID の検査 (空・重複はエラーの翻訳キー) */
export function validateRegionId(p: Project, from: string, to: string): "settings.idEmpty" | "settings.idDuplicate" | null {
  const s = to.trim();
  if (!s) return "settings.idEmpty";
  if (s !== from && p.geometry.regions.some((r) => r.id === s)) return "settings.idDuplicate";
  return null;
}

// ---- キャンバスでの作図・編集 (v1 CadCanvas と同じ規則) ----

/** 多角形の領域を足す (導体・0 V、v1 と同じ)。3 点未満は足さない */
export function addPolygonRegion(p: P, polygon: Point[]): string | null {
  if (polygon.length < 3) return null;
  const id = uniqueRegionId(p as Project);
  p.geometry.regions.push({ id, type: "conductor", voltage: 0, polygon: polygon.map(tidyPoint) });
  return id;
}

/** 円の領域を足す (半径 0 は足さない) */
export function addCircleRegionAt(p: P, center: Point, radius: number): string | null {
  if (!(radius > 0)) return null;
  const id = uniqueRegionId(p as Project);
  p.geometry.regions.push({ id, type: "conductor", voltage: 0, shape: { kind: "circle", center: tidyPoint(center), radius: tidy(radius) } });
  return id;
}

/** 平行移動 (多角形は全頂点、円は中心) */
export function moveRegion(p: P, id: string, dx: number, dy: number): void {
  const r = p.geometry.regions.find((x) => x.id === id);
  if (!r || (dx === 0 && dy === 0)) return;
  if (r.shape) r.shape.center = tidyPoint([r.shape.center[0] + dx, r.shape.center[1] + dy]);
  else if (r.polygon) r.polygon = r.polygon.map(([x, y]) => tidyPoint([x + dx, y + dy]));
}

/** 多角形を置き換える (頂点のドラッグ・辺の中点からの追加。3 点未満は無視) */
export function setRegionPolygon(p: P, id: string, polygon: Point[]): void {
  const r = p.geometry.regions.find((x) => x.id === id);
  if (!r || !r.polygon || polygon.length < 3) return;
  r.polygon = polygon.map(tidyPoint);
}

/** 円の半径 (正の値だけ) */
export function setCircleRadius(p: P, id: string, radius: number): void {
  const r = p.geometry.regions.find((x) => x.id === id);
  if (!r?.shape || !(radius > 0)) return;
  r.shape.radius = tidy(radius);
}

/** 頂点を消す (3 点より多いときだけ) */
export function removeRegionVertex(p: P, id: string, index: number): boolean {
  const r = p.geometry.regions.find((x) => x.id === id);
  if (!r?.polygon || r.polygon.length <= 3 || index < 0 || index >= r.polygon.length) return false;
  r.polygon.splice(index, 1);
  return true;
}
