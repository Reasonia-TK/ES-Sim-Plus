// 領域の追加・削除・名前変更・複製・種類の変更 (v1 と同じ整合性: mesh.local_sizes の参照を保つ、
// 種類を変えても形と対応する値は保ち、RF・CSV 波形・γ は外す)。

import type { Draft } from "immer";
import { polygonBounds, uniqueRegionId, type Project, type Region, type RegionType } from "./project";

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
