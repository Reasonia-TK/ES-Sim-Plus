// オブジェクトスナップ (P7c): カーソルの近く (画面で数 px) の端点・中点・中心・交点・垂線の足・接点に合わせる。
// 垂線と接線は「今描いている線の始点 (直前の点)」から。どれにも合わなければグリッド (呼ぶ側)。
// 同じくらい近いときは端点・中心・交点を中点より、中点を垂線・接線より先にする (距離に少し足して比べる)。

import { angleOnArc, dist, intersections, midpoint, perpendicularFoot, segFromBulge, tangentPoints, type ArcSeg, type Seg, type Vec } from "./geom";

export type SnapKind = "endpoint" | "midpoint" | "center" | "intersection" | "perpendicular" | "tangent" | "grid";

/** 設定で切り替えるスナップの種類 (表示の順) */
export const SNAP_KINDS: SnapKind[] = ["endpoint", "midpoint", "center", "intersection", "perpendicular", "tangent", "grid"];

export interface SnapResult {
  point: Vec;
  kind: SnapKind;
}

/** 同じくらい近いときの優先 (画面の px を足して比べる) */
const PENALTY_PX: Record<SnapKind, number> = {
  endpoint: 0,
  center: 0,
  intersection: 0,
  midpoint: 1.5,
  perpendicular: 3,
  tangent: 3,
  grid: 99,
};

interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

function segBox(s: Seg): Box {
  if (s.kind === "line") return { x0: Math.min(s.a[0], s.b[0]), y0: Math.min(s.a[1], s.b[1]), x1: Math.max(s.a[0], s.b[0]), y1: Math.max(s.a[1], s.b[1]) };
  return { x0: s.center[0] - s.r, y0: s.center[1] - s.r, x1: s.center[0] + s.r, y1: s.center[1] + s.r };
}

/** 円 (円の領域・スケッチの円) は中心と、円周の上の垂線の足・接点に使う */
export interface SnapCircle {
  center: Vec;
  r: number;
}

/** 持ち主の付いた点 (ドラッグしている形の自分自身には合わせないため) */
interface OwnedPoint {
  p: Vec;
  owner: string;
}

/** スナップの相手 (文書が変わったときに作り直す) */
export interface SnapScene {
  segs: { seg: Seg; box: Box; owner: string }[];
  endpoints: OwnedPoint[];
  midpoints: OwnedPoint[];
  centers: OwnedPoint[];
}

/** 相手の形 (持ち主ごと) */
export interface SnapSource {
  owner: string;
  segs: Seg[];
  circles?: SnapCircle[];
  /** 端点として足す点 (今描いている点など) */
  points?: Vec[];
}

/**
 * 相手を作る: 線分・円弧 (端点・中点・円弧の中心) と円 (中心、円周は半円 2 つの円弧として交点・垂線・接線にも)
 */
export function buildSnapScene(sources: SnapSource[]): SnapScene {
  const scene: SnapScene = { segs: [], endpoints: [], midpoints: [], centers: [] };
  for (const { owner, segs, circles = [], points = [] } of sources) {
    for (const p of points) scene.endpoints.push({ p, owner });
    for (const s of segs) {
      scene.segs.push({ seg: s, box: segBox(s), owner });
      scene.endpoints.push({ p: s.a, owner }, { p: s.b, owner });
      scene.midpoints.push({ p: midpoint(s), owner });
      if (s.kind === "arc") scene.centers.push({ p: s.center, owner });
    }
    for (const c of circles) {
      scene.centers.push({ p: c.center, owner });
      for (const seg of [
        segFromBulge([c.center[0] + c.r, c.center[1]], [c.center[0] - c.r, c.center[1]], 1),
        segFromBulge([c.center[0] - c.r, c.center[1]], [c.center[0] + c.r, c.center[1]], 1),
      ])
        scene.segs.push({ seg, box: segBox(seg), owner });
    }
  }
  return scene;
}

/**
 * カーソル (ワールド座標) から aperture [m] 以内で最もよいスナップ。scale は 1 m あたりの px (優先の比較に使う)。
 * ref は垂線・接線の元の点 (描いている線の始点、無ければ使わない)。exclude の持ち主の形には合わせない
 */
export function findSnap(
  scene: SnapScene,
  cursor: Vec,
  aperture: number,
  scale: number,
  enabled: Partial<Record<SnapKind, boolean>>,
  ref: Vec | null,
  exclude: string | null = null,
): SnapResult | null {
  let best: SnapResult | null = null;
  let bestScore = Infinity;
  const consider = (p: Vec, kind: SnapKind) => {
    const d = dist(p, cursor);
    if (d > aperture) return;
    const score = d * scale + PENALTY_PX[kind];
    if (score < bestScore) {
      bestScore = score;
      best = { point: p, kind };
    }
  };
  const mine = (o: { owner: string }) => exclude !== null && o.owner === exclude;
  if (enabled.endpoint) for (const o of scene.endpoints) if (!mine(o)) consider(o.p, "endpoint");
  if (enabled.midpoint) for (const o of scene.midpoints) if (!mine(o)) consider(o.p, "midpoint");
  if (enabled.center) for (const o of scene.centers) if (!mine(o)) consider(o.p, "center");
  // カーソルの近くを通る曲線だけで交点・垂線・接線を調べる
  const near = scene.segs.filter(
    (o) => !mine(o) && cursor[0] >= o.box.x0 - aperture && cursor[0] <= o.box.x1 + aperture && cursor[1] >= o.box.y0 - aperture && cursor[1] <= o.box.y1 + aperture,
  );
  if (enabled.intersection) {
    for (let i = 0; i < near.length; i++) for (let j = i + 1; j < near.length; j++) for (const q of intersections(near[i].seg, near[j].seg)) consider(q, "intersection");
  }
  if (ref) {
    if (enabled.perpendicular) {
      for (const { seg } of near) {
        const f = perpendicularFoot(seg, ref);
        if (!f || dist(f, ref) === 0) continue;
        // 足が線分・円弧の上にあるときだけ
        if (seg.kind === "line") {
          const dx = seg.b[0] - seg.a[0];
          const dy = seg.b[1] - seg.a[1];
          const u = ((f[0] - seg.a[0]) * dx + (f[1] - seg.a[1]) * dy) / (dx * dx + dy * dy);
          if (u < -1e-9 || u > 1 + 1e-9) continue;
        } else if (!angleOnArc(seg, Math.atan2(f[1] - seg.center[1], f[0] - seg.center[0]), 1e-9)) continue;
        consider(f, "perpendicular");
      }
    }
    if (enabled.tangent) {
      for (const { seg } of near) if (seg.kind === "arc") for (const q of tangentPoints(seg as ArcSeg, ref)) consider(q, "tangent");
    }
  }
  return best;
}
