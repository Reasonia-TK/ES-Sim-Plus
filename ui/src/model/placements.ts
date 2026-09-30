// キャンバスの配置ツール (2 点クリック) で文書に足すもの: エミッタ・コレクタ・ガス境界・EEDF 領域・辺の
// メッシュ幅・シース評価線 (v1 と同じ既定値と上限)。使うスタディが無効なら有効にしてから足す (無効にする前の
// 値を覚えていればそれを、無ければ既定値で。v1 はガス境界だけがそうだった)。

import type { Draft } from "immer";
import { recallBlock } from "../forms/blocks";
import { DEFAULT_DSMC_BOUNDARY, DEFAULT_PIC, defaultDsmc, defaultEmitter, defaultParticles } from "../schema/defaults";
import { tidyPoint, type Point, type Project, type SolverKey } from "./project";

type P = Draft<Project>;
type Obj = Record<string, unknown>;

export const MAX_COLLECTORS = 8;
export const MAX_EEDF_REGIONS = 4;
export const MAX_SHEATH_LINES = 4;

/** 既存のラベル (C1, C2, …) の最大番号 + 1 (v1 と同じく欠番は詰めない) */
export function nextLabel(prefix: string, items: { label?: string }[]): string {
  let max = 0;
  for (const it of items) {
    const m = new RegExp(`^${prefix}(\\d+)$`).exec(it.label ?? "");
    if (m) max = Math.max(max, Number(m[1]));
  }
  return `${prefix}${max + 1}`;
}

/** スタディの設定 (無効なら有効にする)。有効にしたら true */
function ensureStudy(p: P, key: SolverKey, defaults: () => unknown): { obj: Obj; enabled: boolean } {
  const cur = p[key];
  if (cur !== null && cur !== undefined) return { obj: cur as Obj, enabled: false };
  p[key] = (recallBlock([key]) ?? defaults()) as Obj;
  return { obj: p[key] as Obj, enabled: true };
}

function list<T>(obj: Obj, key: string): T[] {
  if (!Array.isArray(obj[key])) obj[key] = [];
  return obj[key] as T[];
}

const copy = (q: Point): Point => tidyPoint(q);

export interface PlaceResult {
  /** 足した要素の番号 (上限で足せなかったら null) */
  index: number | null;
  /** スタディを有効にした */
  enabledStudy: boolean;
}

/** 粒子軌道のエミッタの 2 点 (種類・数などはそのまま) */
export function placeEmitter(p: P, p1: Point, p2: Point): PlaceResult {
  const { obj, enabled } = ensureStudy(p, "particles", () => defaultParticles(p as Project));
  const em = obj.emitter as Obj | null | undefined;
  if (em) {
    em.p1 = copy(p1);
    em.p2 = copy(p2);
  } else {
    obj.emitter = { ...defaultEmitter(p as Project), p1: copy(p1), p2: copy(p2) };
  }
  return { index: 0, enabledStudy: enabled };
}

/** PIC の注入のエミッタの 2 点 (注入が無ければ既定の注入を作る。種類・数などはそのまま) */
export function placeInjectionEmitter(p: P, p1: Point, p2: Point): PlaceResult {
  const { obj, enabled } = ensureStudy(p, "pic", () => structuredClone(DEFAULT_PIC));
  let inj = obj.injection as Obj | null | undefined;
  if (!inj) {
    inj = { emitter: { ...defaultEmitter(p as Project), p1: copy(p1), p2: copy(p2) }, species: "electron", current_a_per_m: 1e-4 };
    obj.injection = inj;
    return { index: 0, enabledStudy: enabled };
  }
  const em = inj.emitter as Obj | null | undefined;
  if (em) {
    em.p1 = copy(p1);
    em.p2 = copy(p2);
  } else {
    inj.emitter = { ...defaultEmitter(p as Project), p1: copy(p1), p2: copy(p2) };
  }
  return { index: 0, enabledStudy: enabled };
}

export function placeCollector(p: P, p1: Point, p2: Point): PlaceResult {
  const { obj, enabled } = ensureStudy(p, "pic", () => structuredClone(DEFAULT_PIC));
  const items = list<{ label?: string }>(obj, "collectors");
  if (items.length >= MAX_COLLECTORS) return { index: null, enabledStudy: enabled };
  items.push({ p1: copy(p1), p2: copy(p2), tol: null, label: nextLabel("C", items) } as { label: string });
  return { index: items.length - 1, enabledStudy: enabled };
}

export function placeEedfRegion(p: P, p1: Point, p2: Point): PlaceResult {
  const { obj, enabled } = ensureStudy(p, "pic", () => structuredClone(DEFAULT_PIC));
  const items = list<{ label?: string }>(obj, "eedf_regions");
  if (items.length >= MAX_EEDF_REGIONS) return { index: null, enabledStudy: enabled };
  items.push({ p1: copy(p1), p2: copy(p2), label: nextLabel("E", items), bins: 100, e_max_ev: null } as { label: string });
  return { index: items.length - 1, enabledStudy: enabled };
}

/** シース評価線 (1 点目 = 電極側、2 点目 = バルク側) */
export function placeSheathLine(p: P, p1: Point, p2: Point): PlaceResult {
  const { obj, enabled } = ensureStudy(p, "pic", () => structuredClone(DEFAULT_PIC));
  const items = list<{ label?: string }>(obj, "sheath_lines");
  if (items.length >= MAX_SHEATH_LINES) return { index: null, enabledStudy: enabled };
  items.push({ p1: copy(p1), p2: copy(p2), label: nextLabel("S", items) } as { label: string });
  return { index: items.length - 1, enabledStudy: enabled };
}

/** DSMC の線分境界 (既定は壁、v1 の「境界を追加」と同じ) */
export function placeGasBoundary(p: P, p1: Point, p2: Point): PlaceResult {
  const { obj, enabled } = ensureStudy(p, "dsmc", defaultDsmc);
  const items = list<Obj>(obj, "boundaries");
  items.push({ ...structuredClone(DEFAULT_DSMC_BOUNDARY), p1: copy(p1), p2: copy(p2) });
  return { index: items.length - 1, enabledStudy: enabled };
}

/** 辺の局所メッシュ幅 (既定はメッシュ幅の 1/4、v1 と同じ) */
export function placeEdgeMeshSize(p: P, p1: Point, p2: Point): PlaceResult {
  const items = list<Obj>(p.mesh as Obj, "local_edge_sizes");
  items.push({ p1: copy(p1), p2: copy(p2), size: p.mesh.size / 4 });
  return { index: items.length - 1, enabledStudy: false };
}
