// 文書からキャンバスに重ねるもの (配置物・エミッタ・AMR の細分化の矩形) を取り出す。

import type { Point, Project } from "../model/project";
import type { EmitterView, Placement } from "./overlay";
import type { OverlayKey } from "./viewerStore";

interface Seg {
  p1?: Point | null;
  p2?: Point | null;
  label?: string;
}

const isPoint = (q: unknown): q is Point => Array.isArray(q) && q.length === 2 && q.every((v) => typeof v === "number");

function segs(items: unknown, kind: Placement["kind"], prefix: string, out: Placement[]): void {
  if (!Array.isArray(items)) return;
  (items as Seg[]).forEach((it, i) => {
    if (!isPoint(it?.p1) || !isPoint(it?.p2)) return;
    out.push({ kind, label: it.label || `${prefix}${i + 1}`, p1: it.p1, p2: it.p2, index: i });
  });
}

/** 表示する配置物 (v1 と同じ順: コレクタ・ガス境界・EEDF 領域・辺のメッシュ幅・シース評価線) */
export function placementsOf(p: Project, show: Record<OverlayKey, boolean>): Placement[] {
  const out: Placement[] = [];
  const pic = p.pic as { collectors?: unknown; eedf_regions?: unknown; sheath_lines?: unknown } | null | undefined;
  const dsmc = p.dsmc as { boundaries?: unknown } | null | undefined;
  if (show.collectors) segs(pic?.collectors, "collector", "C", out);
  if (show.gasBoundaries && Array.isArray(dsmc?.boundaries)) {
    (dsmc.boundaries as Seg[]).forEach((b, i) => {
      if (isPoint(b?.p1) && isPoint(b?.p2)) out.push({ kind: "gasbc", label: `G${i + 1}`, p1: b.p1, p2: b.p2, index: i });
    });
  }
  if (show.eedf) segs(pic?.eedf_regions, "eedf", "E", out);
  if (show.edgeSizes) {
    const items = (p.mesh as { local_edge_sizes?: unknown }).local_edge_sizes;
    if (Array.isArray(items)) {
      (items as Seg[]).forEach((m, i) => {
        if (isPoint(m?.p1) && isPoint(m?.p2)) out.push({ kind: "edgeSize", label: `M${i + 1}`, p1: m.p1, p2: m.p2, index: i });
      });
    }
  }
  if (show.sheathLines) segs(pic?.sheath_lines, "sheath", "S", out);
  return out;
}

/** 粒子軌道のエミッタ (FN 放出のときは出さない) */
export function emitterOf(p: Project): EmitterView | null {
  const ps = p.particles as { fn?: unknown; emitter?: { kind?: string; p1?: unknown; p2?: unknown; direction_deg?: number } | null } | null | undefined;
  const e = ps?.emitter;
  if (!ps || (ps.fn !== null && ps.fn !== undefined) || !e || !isPoint(e.p1)) return null;
  return { kind: e.kind === "point" ? "point" : "line", p1: e.p1, p2: isPoint(e.p2) ? e.p2 : e.p1, directionDeg: e.direction_deg ?? 0 };
}

/** PIC の注入のエミッタ (PIC が有効で注入があるとき。v1 は粒子軌道のエミッタと共通だった) */
export function injectorOf(p: Project): EmitterView | null {
  const inj = (p.pic as { injection?: { emitter?: { kind?: string; p1?: unknown; p2?: unknown; direction_deg?: number } | null } | null } | null | undefined)?.injection;
  const e = inj?.emitter;
  if (!e || !isPoint(e.p1)) return null;
  return { kind: e.kind === "point" ? "point" : "line", p1: e.p1, p2: isPoint(e.p2) ? e.p2 : e.p1, directionDeg: e.direction_deg ?? 0 };
}

/** AMR の細分化の矩形 (直交格子のときだけ) */
export function amrBoxesOf(p: Project): [Point, Point][] {
  if (p.mesh.mode !== "cartesian") return [];
  const regions = (p.mesh.amr as { regions?: unknown } | null | undefined)?.regions;
  if (!Array.isArray(regions)) return [];
  return (regions as Seg[]).filter((r) => isPoint(r?.p1) && isPoint(r?.p2)).map((r) => [r.p1 as Point, r.p2 as Point]);
}
