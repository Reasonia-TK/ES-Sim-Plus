// キャンバスの道具の操作 (P7b で Viewer から分けた):
// - 選択: クリック (Shift で足す・外す)、何も無い所・選んでいない所からのドラッグで範囲選択、選んだものの上から
//   ドラッグで移動 (複数まとめて)、編集している形 (1 つ選んだ領域・スケッチ、またはドメイン) の頂点・辺の中点・
//   半径のハンドル。辺の中点は頂点を足す (Shift で曲げる。スケッチの線・円弧は逆)。頂点のダブルクリックで削除
// - 作図: 折れ線 (A で円弧の段、L で直線に戻る、最初の点で閉じる)・矩形・円 (描く先は領域かスケッチ)、線 (続けて描く)、
//   3 点の円弧、囲まれた所から領域
// - 配置 (2 点)、プローブ・計測、移動 (中ボタンか Space + ドラッグ)
// - スナップ (P7c): 端点・中点・中心・交点・垂線・接線 (オブジェクト) のあとグリッド。Alt を押している間は合わせない。
//   F3 でスナップ全体のオン・オフ
// - 数値入力 (P7c): 点を置く道具で数字・@ などを打つと座標の欄が開く (cad/coordInput.ts の書き方)
// - 変換 (P7d): 移動 (基点 → 先)・回転 (中心 → 向きか角度)・ミラー (軸の 2 点)・尺度 (基点 → 倍率か基準の点 → 点)。
//   形を選んでいなければクリックで選ぶ (Shift で足す)。コピーを作るかは道具の帯で
// - 編集 (P7d): フィレット・面取り (角をクリック、2 本の線は残す側を順に)、オフセット (形 → 側)、トリム (消す部分)、
//   延長 (端の近く)。半径・長さ・距離は数字を打って変える

import type { TFunction } from "i18next";
import { produce, type Draft } from "immer";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type SetStateAction,
} from "react";
import { logInfo, logWarning } from "../app/messages";
import { Arrangement } from "../cad/arrangement";
import type { Shape } from "../cad/boolean";
import { parseCoordInput } from "../cad/coordInput";
import { trimRemovedAt } from "../cad/edit";
import { buildSnapScene, findSnap, type SnapResult, type SnapSource } from "../cad/snap";
import { bulgeThrough, closestPoint, midpoint, pathSegs, segFromBulge, type Seg } from "../cad/geom";
import { bulgesOf, mirror, rotation, scaling, translation, type Affine } from "../cad/path";
import { useDocument } from "../model/documentStore";
import { cornerEdit, cutterSegs, extendSketch, filletSketchLines, itemSegs, offsetItem, transformItems, trimSketch, type CornerTarget } from "../model/editOps";
import { usePrefs } from "../prefs/prefs";
import type { EdgeRemapReport } from "../model/domainOps";
import {
  placeCollector,
  placeEdgeMeshSize,
  placeEedfRegion,
  placeEmitter,
  placeGasBoundary,
  placeInjectionEmitter,
  placeSheathLine,
  MAX_COLLECTORS,
  MAX_EEDF_REGIONS,
  MAX_SHEATH_LINES,
  type PlaceResult,
} from "../model/placements";
import { domainPath, edgeIdOf, regionRings, type Point, type Project } from "../model/project";
import { addCircleRegionAt, addPolygonRegion, addShapeRegion } from "../model/regionOps";
import { pickableRegions, pickableSketch, visibleRegions, visibleSketch } from "../model/layers";
import { PLACEMENT_NODES, PLACEMENT_PATHS, useSelection, type PickRef, type PlacementKind } from "../model/selection";
import { addSketch, editBend, editMove, editRemove, editSplit, edgeCountOf, sketchOf, sketchSegs, type EditPath } from "../model/sketch";
import { getIn, setIn } from "../schema/schema";
import { evalExpression, parseParamExpr, parseQuantity } from "../schema/units";
import { paramValues } from "../model/params";
import { formatNumber, lengthUnitLabel, toDisplayLength } from "../util/format";
import { gridStep, panBy, snapPoint, snapValue, toScreen, toWorld, type Camera } from "./camera";
import { commitEdit, commitRadius, deleteItems, editObjectOf, findSketchAt, itemsInBox, moveItems, type EditHow, type EditObject, type EditTarget } from "./editTargets";
import {
  CLICK_TOLERANCE_PX,
  dedupeTail,
  EDGE_TOLERANCE_PX,
  findDomainEdgeAt,
  findRegionAt,
  findVertexHandle,
  HANDLE_TOLERANCE_PX,
  hitRadiusHandle,
  hitRegion,
  hitSegment,
  rectFromCorners,
} from "./hitTest";
import type { Drawing, Placement, Preview, ToolPreview } from "./overlay";
import type { Scene } from "./scene";
import { EDIT_TOOLS, TRANSFORM_TOOLS, useViewer, type Tool } from "./viewerStore";

type Drag =
  | { kind: "pan"; lastX: number; lastY: number; x: number; y: number }
  | { kind: "click"; x: number; y: number; shift: boolean }
  /** 範囲選択 (動かさなければクリック) */
  | { kind: "box"; x: number; y: number; start: Point; shift: boolean }
  /** 頂点を動かす・辺を分けて頂点を動かす・辺を曲げる */
  | { kind: "path"; x: number; y: number; target: EditTarget; how: EditHow; path: EditPath }
  | { kind: "radius"; x: number; y: number; target: EditTarget; center: Point }
  | { kind: "move"; x: number; y: number; start: Point };

/** 2 点で決める道具 */
const TWO_POINT: Tool[] = ["rect", "circle", "profile", "emitter", "injector", "collector", "gasbc", "eedfbox", "meshref", "sheathline", "measure"];

/** 曲げたときに直線に戻す弦からの距離 [px] */
const STRAIGHTEN_PX = 4;

/** オブジェクトスナップの届く距離 [px] */
const SNAP_APERTURE_PX = 10;

/** 点を置く道具と、値を打つ道具 (数値入力を受ける) */
const POINT_TOOLS: Tool[] = ["polyline", "line", "arc", "fill", "probe", ...TWO_POINT, ...TRANSFORM_TOOLS, "fillet", "chamfer", "offset"];

/** 数値入力の欄が受けるもの: 点・角度 [°]・倍率・長さ */
export type CoordMode = "point" | "angle" | "factor" | "length";

/** 途中まで選んだもの (オフセットの元・フィレットの 1 本目の線) */
type Pending = { kind: "offset"; item: PickRef } | { kind: "filletLine"; id: string; pick: Point };

/** 形を選ぶだけの道具 (点をスナップしない) */
const NO_SNAP_TOOLS: Tool[] = ["fill", "fillet", "chamfer", "offset", "trim", "extend"];

/** 数値入力を始める文字 */
const COORD_START = /^[0-9.@+\-(]$/;

/** 編集している形の持ち主 (スナップで自分自身に合わせないため) */
function ownerOf(target: EditTarget): string {
  return target.kind === "domain" ? "domain" : `${target.kind}:${target.id}`;
}

/** ポインタを捕まえる (キャンバスの外へ出てもドラッグを続ける。作ったイベントでは捕まえられないので無視) */
function capture(el: HTMLElement, id: number): void {
  try {
    el.setPointerCapture(id);
  } catch {
    // 捕まえられないポインタ
  }
}

/** 画面の点 → 要素内の座標 (CSS px) */
export function localPoint(e: { clientX: number; clientY: number }, el: HTMLElement): [number, number] {
  const r = el.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}

/** 囲まれた所を探す相手: ドメインの辺・領域の輪郭・スケッチ (表示しているレイヤ) */
function allCurves(p: Project): Seg[] {
  const segs: Seg[] = [];
  const d = domainPath(p);
  segs.push(...pathSegs(d.polygon, bulgesOf(d)));
  for (const r of visibleRegions(p)) for (const rp of regionRings(r)) segs.push(...pathSegs(rp.polygon, bulgesOf(rp)));
  for (const e of visibleSketch(p)) segs.push(...sketchSegs(e));
  return segs;
}

/** 辺の中点のハンドル (開いた形は最後の辺が無い) */
function findEditMidpoint(path: EditPath, c: Camera, sx: number, sy: number): number | null {
  let best: number | null = null;
  let bestD = HANDLE_TOLERANCE_PX;
  for (let i = 0; i < edgeCountOf(path); i++) {
    const [x, y] = toScreen(c, midpoint(segFromBulge(path.points[i], path.points[(i + 1) % path.points.length], path.bulges[i] ?? 0)));
    const d = Math.hypot(x - sx, y - sy);
    if (d <= bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

export interface CanvasToolsArgs {
  camera: Camera | null;
  setCamera: Dispatch<SetStateAction<Camera | null>>;
  /** 使う人が拡大・移動した (大きさが変わっても全体表示に戻さない) */
  markUserMoved: () => void;
  project: Project;
  scene: Scene;
  placements: Placement[];
  t: TFunction;
}

export interface CanvasTools {
  onPointerDown: (e: ReactPointerEvent<HTMLCanvasElement>) => void;
  onPointerMove: (e: ReactPointerEvent<HTMLCanvasElement>) => void;
  onPointerUp: (e: ReactPointerEvent<HTMLCanvasElement>) => void;
  onPointerCancel: () => void;
  onPointerLeave: () => void;
  onDoubleClick: (e: ReactMouseEvent<HTMLCanvasElement>) => void;
  onKeyDown: (e: ReactKeyboardEvent<HTMLCanvasElement>) => void;
  cursor: Point | null;
  hover: Point | null;
  panning: boolean;
  drawing: Drawing;
  preview: Preview;
  measure: [Point, Point] | null;
  fillPreview: Shape | null;
  edit: EditObject | null;
  picked: PickRef[];
  /** いま合っているオブジェクトスナップ (印を出す) */
  snapMark: SnapResult | null;
  /** 数値入力の欄 (null は閉じている) */
  coordText: string | null;
  /** 欄が受けるもの (説明の出し分け) */
  coordMode: CoordMode;
  /** 編集の道具の見せる形 (トリムで消える部分・オフセット・延長の結果・フィレットの角) */
  toolPreview: ToolPreview | null;
  coordError: string | null;
  setCoordText: (s: string) => void;
  /** 数値入力を確定する (点を置けたら true) */
  submitCoord: () => boolean;
  closeCoord: () => void;
}

export function useCanvasTools({ camera, setCamera, markUserMoved, project, scene, placements, t }: CanvasToolsArgs): CanvasTools {
  const vs = useViewer();
  const picked = useSelection((s) => s.picked);
  const activeNode = useSelection((s) => s.activeNode);
  const [cursor, setCursor] = useState<Point | null>(null);
  const [hover, setHover] = useState<Point | null>(null);
  const [pts, setPts] = useState<Point[]>([]);
  const [bulges, setBulges] = useState<number[]>([]);
  const [through, setThrough] = useState<Point | null>(null);
  const [arcMode, setArcMode] = useState(false);
  const [preview, setPreview] = useState<Preview>({});
  const [measure, setMeasure] = useState<[Point, Point] | null>(null);
  const [panning, setPanning] = useState(false);
  const [snapMark, setSnapMark] = useState<SnapResult | null>(null);
  const [coordText, setCoordText] = useState<string | null>(null);
  const [coordError, setCoordError] = useState<string | null>(null);
  const [pending, setPending] = useState<Pending | null>(null);
  const lengthUnit = usePrefs((st) => st.lengthUnit);
  const dragRef = useRef<Drag | null>(null);
  const spaceRef = useRef(false);
  const tool = vs.tool;
  const target = vs.drawTarget;
  // 選べるもの (表示していてロックしていないレイヤ、P7f)
  const sketch = useMemo(() => pickableSketch(project), [project]);
  const regions = useMemo(() => pickableRegions(project), [project]);
  const edit = useMemo(() => editObjectOf(project, picked, activeNode), [project, picked, activeNode]);

  // Space (移動)
  useEffect(() => {
    const editable = (el: EventTarget | null) => el instanceof HTMLElement && (["INPUT", "SELECT", "TEXTAREA"].includes(el.tagName) || el.isContentEditable);
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && !editable(e.target)) spaceRef.current = true;
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space") spaceRef.current = false;
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  const resetDrawing = () => {
    setPts([]);
    setBulges([]);
    setThrough(null);
  };
  // ツールを変えたら作図中の点を捨てる (v1 と同じ)。計測はそのまま残さない
  useEffect(() => {
    resetDrawing();
    setArcMode(false);
    setMeasure(null);
    setCoordText(null);
    setSnapMark(null);
    setPending(null);
  }, [tool]);
  // 選択・形が変わったら (元に戻す・削除など) ドラッグをやめる
  useEffect(() => {
    dragRef.current = null;
    setPreview({});
  }, [picked, activeNode, project.geometry, project.cad]);

  const snapStep = camera ? gridStep(camera) / 10 : 0;
  // スナップの相手: ドメインの辺・領域・スケッチ・配置物の線分・今描いている点 (文書か描いている点が変わったら作り直す)
  const snapScene = useMemo(() => {
    const sources: SnapSource[] = [];
    const d = domainPath(project);
    sources.push({ owner: "domain", segs: pathSegs(d.polygon, bulgesOf(d)) });
    for (const r of visibleRegions(project)) {
      if (r.shape) sources.push({ owner: `region:${r.id}`, segs: [], circles: [{ center: r.shape.center, r: r.shape.radius }] });
      else sources.push({ owner: `region:${r.id}`, segs: regionRings(r).flatMap((rp) => pathSegs(rp.polygon, bulgesOf(rp))) });
    }
    for (const e of visibleSketch(project)) {
      if (e.kind === "circle") sources.push({ owner: `sketch:${e.id}`, segs: [], circles: [{ center: e.center, r: e.r }] });
      else sources.push({ owner: `sketch:${e.id}`, segs: sketchSegs(e) });
    }
    for (const pl of placements) {
      const [a, b] = [pl.p1, pl.p2];
      const segs: Seg[] =
        pl.kind === "eedf"
          ? [
              { kind: "line", a, b: [b[0], a[1]] },
              { kind: "line", a: [b[0], a[1]], b },
              { kind: "line", a: b, b: [a[0], b[1]] },
              { kind: "line", a: [a[0], b[1]], b: a },
            ]
          : [{ kind: "line", a, b }];
      sources.push({ owner: "placement", segs });
    }
    sources.push({ owner: "draw", segs: [], points: pts });
    return buildSnapScene(sources);
  }, [project, placements, pts]);

  /** 点を合わせる: オブジェクトスナップ、なければグリッド。Alt・スナップを切っているときはそのまま */
  const snapAt = (raw: Point, opts: { alt?: boolean; exclude?: string | null } = {}): { point: Point; mark: SnapResult | null } => {
    if (!vs.snap || opts.alt || !camera) return { point: raw, mark: null };
    const ref = pts.length ? pts[pts.length - 1] : null;
    const obj = findSnap(snapScene, raw, SNAP_APERTURE_PX / camera.scale, camera.scale, vs.snapKinds, ref, opts.exclude ?? null);
    if (obj) return { point: obj.point as Point, mark: obj };
    if (vs.snapKinds.grid && snapStep > 0) return { point: snapPoint(raw, snapStep), mark: null };
    return { point: raw, mark: null };
  };
  const snap = (p: Point, alt = false): Point => snapAt(p, { alt }).point;

  const isTransform = (TRANSFORM_TOOLS as Tool[]).includes(tool);
  const isEdit = (EDIT_TOOLS as Tool[]).includes(tool);
  const coordMode: CoordMode =
    tool === "fillet" || tool === "chamfer" || tool === "offset" ? "length" : tool === "rotate" && pts.length === 1 ? "angle" : tool === "scale" && pts.length === 1 ? "factor" : "point";

  /** 変換の道具で、カーソルまでの変換 (選んだものの見せる形) */
  const transformAffine = useMemo((): Affine | null => {
    if (!isTransform || !cursor || pts.length === 0 || picked.length === 0) return null;
    const base = pts[0];
    if (tool === "move") return translation(cursor[0] - base[0], cursor[1] - base[1]);
    if (tool === "rotate") return cursor[0] === base[0] && cursor[1] === base[1] ? null : rotation(Math.atan2(cursor[1] - base[1], cursor[0] - base[0]), base);
    if (tool === "mirror") return cursor[0] === base[0] && cursor[1] === base[1] ? null : mirror(base, cursor);
    if (tool === "scale" && pts.length === 2) {
      const r0 = Math.hypot(pts[1][0] - base[0], pts[1][1] - base[1]);
      const r1 = Math.hypot(cursor[0] - base[0], cursor[1] - base[1]);
      return r0 > 0 && r1 > 0 ? scaling(r1 / r0, base) : null;
    }
    return null;
  }, [isTransform, tool, cursor, pts, picked]);

  /** フィレット・面取りする角 (スケッチのポリライン・多角形の領域・ドメインの頂点のうち画面で近いもの) */
  const cornerAt = (q: Point): { target: CornerTarget; index: number; point: Point } | null => {
    if (!camera) return null;
    let best: { target: CornerTarget; index: number; point: Point } | null = null;
    let bestD = HANDLE_TOLERANCE_PX;
    const [sx, sy] = toScreen(camera, q);
    const consider = (target: CornerTarget, points: Point[], closed: boolean) => {
      points.forEach((v, i) => {
        if (!closed && (i === 0 || i === points.length - 1)) return;
        const [x, y] = toScreen(camera, v);
        const dd = Math.hypot(x - sx, y - sy);
        if (dd <= bestD) {
          bestD = dd;
          best = { target, index: i, point: v };
        }
      });
    };
    for (const e of sketch) if (e.kind === "polyline") consider({ kind: "sketch", id: e.id }, e.points, e.closed);
    for (const r of regions) {
      if (!r.polygon) continue;
      consider({ kind: "region", id: r.id }, r.polygon, true);
      (r.holes ?? []).forEach((h, k) => consider({ kind: "region", id: r.id, hole: k }, h.polygon, true));
    }
    consider({ kind: "domain" }, project.geometry.domain.polygon, true);
    return best;
  };

  /** クリックした形 (スケッチ > 領域) */
  const itemAt = (q: Point): PickRef | null => {
    if (!camera) return null;
    const tol = EDGE_TOLERANCE_PX / camera.scale;
    const e = findSketchAt(q, sketch, tol);
    if (e) return { kind: "sketch", id: e.id };
    const r = findRegionAt(q, regions, tol);
    return r ? { kind: "region", id: r.id } : null;
  };

  const toolPreview = useMemo((): ToolPreview | null => {
    if (!isEdit || !camera || !hover) return null;
    const tol = EDGE_TOLERANCE_PX / camera.scale;
    const pendingSegs = pending?.kind === "offset" ? itemSegs(project, pending.item) : pending?.kind === "filletLine" ? itemSegs(project, { kind: "sketch", id: pending.id }) : [];
    if (tool === "trim") {
      const e = findSketchAt(hover, sketch, tol);
      if (!e) return null;
      const closed = e.kind === "circle" || (e.kind === "polyline" && e.closed);
      const removed = trimRemovedAt(sketchSegs(e), closed, cutterSegs(project, e.id), hover);
      return removed ? { segs: removed, tone: "remove", pending: [] } : null;
    }
    if (tool === "extend") {
      const e = findSketchAt(hover, sketch, tol);
      if (!e) return null;
      let ok = false;
      const next = produce(project, (d) => void (ok = extendSketch(d, e.id, hover, cutterSegs(project, e.id)).ok));
      const ne = ok ? sketchOf(next).find((x) => x.id === e.id) : null;
      return ne ? { segs: sketchSegs(ne), tone: "add", pending: [] } : null;
    }
    if (tool === "offset" && pending?.kind === "offset") {
      let made: PickRef | null = null;
      const next = produce(project, (d) => {
        const r = offsetItem(d, pending.item, vs.offsetDistance, hover);
        if (r.ok) made = r.value;
      });
      return { segs: made ? itemSegs(next, made) : [], tone: "add", pending: pendingSegs };
    }
    if (tool === "fillet" || tool === "chamfer") {
      const c = cornerAt(hover);
      return { segs: [], tone: "add", pending: pendingSegs, points: c ? [c.point] : [] };
    }
    return pendingSegs.length ? { segs: [], tone: "add", pending: pendingSegs } : null;
    // cornerAt・itemAt は文書と画面から決まる
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isEdit, tool, hover, pending, project, camera, vs.offsetDistance]);

  // 囲まれた所から領域を作る道具: 平面の配置は文書が変わったときだけ作り直す
  const arrangement = useMemo(() => (tool === "fill" ? new Arrangement(allCurves(project)) : null), [tool, project]);
  const fillPreview = useMemo(() => (arrangement && hover ? arrangement.shapeAt(hover) : null), [arrangement, hover]);

  // ---- 文書を変える ----

  const update = (label: string, recipe: (d: Draft<Project>) => void) => useDocument.getState().update(label, recipe);
  const reportRemap = (r: EdgeRemapReport | null) => {
    const n = r ? r.boundariesRemoved + r.periodicRemoved : 0;
    if (n > 0) logInfo(t("msg.source.app"), t("settings.bcRemoved", { n }));
  };
  const selectSketch = (id: string | null) => {
    if (id) useSelection.getState().pick([{ kind: "sketch", id }]);
  };

  const commitTwoPoint = (p1: Point, p2: Point) => {
    const same = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) === 0;
    if (tool === "rect") {
      if (same || p1[0] === p2[0] || p1[1] === p2[1]) return;
      let id: string | null = null;
      if (target === "sketch") {
        update(t("cad.addSketch"), (d) => void (id = addSketch(d, { kind: "polyline", points: rectFromCorners(p1, p2), closed: true })));
        selectSketch(id);
      } else {
        update(t("cad.addRegion"), (d) => void (id = addPolygonRegion(d, rectFromCorners(p1, p2))));
        if (id) useSelection.getState().selectRegion(id);
      }
    } else if (tool === "circle") {
      const r = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
      if (!(r > 0)) return;
      let id: string | null = null;
      if (target === "sketch") {
        update(t("cad.addSketch"), (d) => void (id = addSketch(d, { kind: "circle", center: p1, r })));
        selectSketch(id);
      } else {
        update(t("cad.addRegion"), (d) => void (id = addCircleRegionAt(d, p1, r)));
        if (id) useSelection.getState().selectRegion(id);
      }
    } else if (tool === "profile") {
      if (!same) vs.setProfile([p1, p2]);
    } else if (tool === "measure") {
      setMeasure([p1, p2]);
    } else {
      if (same) return;
      const place: Record<string, [(d: Parameters<typeof placeEmitter>[0], a: Point, b: Point) => PlaceResult, string, PlacementKind | null, string, number]> = {
        emitter: [placeEmitter, t("cad.placeEmitter"), null, "study:trace", 0],
        injector: [placeInjectionEmitter, t("cad.placeInjection"), null, "study:pic", 0],
        collector: [placeCollector, t("cad.placeCollector"), "collector", "study:pic", MAX_COLLECTORS],
        gasbc: [placeGasBoundary, t("cad.placeGasBoundary"), "gasbc", "study:dsmc", 0],
        eedfbox: [placeEedfRegion, t("cad.placeEedf"), "eedf", "study:pic", MAX_EEDF_REGIONS],
        meshref: [placeEdgeMeshSize, t("cad.placeEdgeSize"), "edgeSize", "mesh", 0],
        sheathline: [placeSheathLine, t("cad.placeSheathLine"), "sheath", "study:pic", MAX_SHEATH_LINES],
      };
      const entry = place[tool];
      if (!entry) return;
      const [fn, label, kind, node, max] = entry;
      let res: PlaceResult | null = null;
      update(label, (d) => void (res = fn(d, p1, p2)));
      const r = res as PlaceResult | null;
      if (r?.index === null) logWarning(t("msg.source.app"), t("cad.listFull", { what: label, max }));
      const sel = useSelection.getState();
      sel.select(node);
      if (kind && r && r.index !== null) sel.selectPlacement({ kind, index: r.index });
    }
  };

  /** 折れ線を確定する (closed なら最後の点 → 最初の点の辺も。その bulge は closing) */
  const finishPolyline = (closed: boolean, closing = 0) => {
    let list = pts;
    let bs = bulges;
    const tail = dedupeTail(list);
    if (tail.length < list.length) {
      list = tail;
      bs = bs.slice(0, tail.length - 1);
    }
    resetDrawing();
    const arcs = bs.some((b) => b) || closing !== 0;
    let id: string | null = null;
    if (target === "region" || tool !== "polyline") {
      if (list.length < (arcs ? 2 : 3)) return;
      update(t("cad.addRegion"), (d) => void (id = addPolygonRegion(d, list, [...bs, closing])));
      if (id) useSelection.getState().selectRegion(id);
      return;
    }
    if (closed ? list.length < (arcs ? 2 : 3) : list.length < 2) return;
    update(t("cad.addSketch"), (d) => void (id = addSketch(d, { kind: "polyline", points: list, bulges: [...bs, closed ? closing : 0], closed })));
    selectSketch(id);
  };

  /** 最初の点に戻った (クリックで閉じる) か */
  const nearFirst = (sx: number, sy: number): boolean => {
    if (!camera || tool !== "polyline" || pts.length < 2) return false;
    const arcs = bulges.some((b) => b) || through !== null;
    if (pts.length < 3 && !arcs) return false;
    const [x, y] = toScreen(camera, pts[0]);
    return Math.hypot(x - sx, y - sy) <= HANDLE_TOLERANCE_PX;
  };

  const deleteSelected = () => {
    const sel = useSelection.getState();
    if (sel.picked.length) {
      const items = sel.picked;
      update(t("cad.deleteItems"), (d) => deleteItems(d, items));
      sel.selectRegion(null);
    } else if (sel.selectedPlacement) {
      const { kind, index } = sel.selectedPlacement;
      const path = PLACEMENT_PATHS[kind];
      update(t("cad.deletePlacement"), (d) => {
        const items = getIn(d, path);
        if (Array.isArray(items)) setIn(d, path, items.filter((_, i) => i !== index));
      });
      sel.selectPlacement(null);
    }
  };

  const nudge = (dx: number, dy: number) => {
    const items = useSelection.getState().picked;
    if (items.length) update(t("cad.moveItems"), (d) => moveItems(d, items, dx, dy));
  };

  /** 選んだものの上か (移動のドラッグを始める) */
  const onPicked = (world: Point, tol: number): boolean =>
    picked.some((it) => {
      if (it.kind === "region") {
        const r = project.geometry.regions.find((x) => x.id === it.id);
        return r ? hitRegion(world, r, tol) : false;
      }
      const e = sketch.find((x) => x.id === it.id);
      return e ? sketchSegs(e).some((s) => closestPoint(s, world).dist <= tol) : false;
    });

  /** 選択の道具のクリック: 配置物 > スケッチ > 外周の辺 > 領域 (Shift は足す・外す) */
  const selectAt = (raw: Point, shift: boolean) => {
    if (!camera) return;
    const tol = EDGE_TOLERANCE_PX / camera.scale;
    const sel = useSelection.getState();
    if (!shift) {
      // 細い配置物を先に (領域の上に描いてある)
      const hitP = [...placements].reverse().find((pl) => (pl.kind === "eedf" ? hitRect(raw, pl.p1, pl.p2, tol) : hitSegment(raw, pl.p1, pl.p2, tol)));
      if (hitP) {
        sel.selectPlacement({ kind: hitP.kind, index: hitP.index });
        sel.select(PLACEMENT_NODES[hitP.kind]);
        return;
      }
    }
    const sk = findSketchAt(raw, sketch, tol);
    if (sk) {
      sel.pick([{ kind: "sketch", id: sk.id }], shift ? "toggle" : "replace");
      return;
    }
    if (!shift) {
      // 外周の辺 (輪郭から数 px 以内) を領域より先に: 境界条件のページを開く
      const edge = findDomainEdgeAt(raw, project, tol);
      if (edge !== null) {
        sel.selectPlacement(null);
        sel.select(`edge:${edgeIdOf(project, edge)}`);
        return;
      }
    }
    const hit = findRegionAt(raw, regions, tol);
    if (hit) {
      sel.pick([{ kind: "region", id: hit.id }], shift ? "toggle" : "replace");
      return;
    }
    if (!shift) {
      sel.selectPlacement(null);
      sel.selectRegion(null);
    }
  };

  // ---- マウス ----

  const onPointerDown = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const el = e.currentTarget;
    el.focus({ preventScroll: true });
    // キャンバスをクリックしたら数値入力は閉じる (クリックした点を置く)
    setCoordText(null);
    setCoordError(null);
    if (!camera) return;
    const [sx, sy] = localPoint(e, el);
    if (e.button === 1 || (e.button === 0 && spaceRef.current)) {
      e.preventDefault();
      capture(el, e.pointerId);
      dragRef.current = { kind: "pan", lastX: e.clientX, lastY: e.clientY, x: sx, y: sy };
      setPanning(true);
      return;
    }
    if (e.button !== 0) return;
    capture(el, e.pointerId);
    dragRef.current = { kind: "click", x: sx, y: sy, shift: e.shiftKey };
    if (tool !== "select") return;
    const world = toWorld(camera, sx, sy);
    const tol = EDGE_TOLERANCE_PX / camera.scale;
    if (edit?.circle) {
      if (hitRadiusHandle({ kind: "circle", center: edit.circle.center, radius: edit.circle.r }, camera, sx, sy)) {
        dragRef.current = { kind: "radius", x: sx, y: sy, target: edit.target, center: edit.circle.center };
        return;
      }
    } else if (edit?.path) {
      const v = findVertexHandle(edit.path.points, camera, sx, sy);
      if (v !== null) {
        dragRef.current = { kind: "path", x: sx, y: sy, target: edit.target, how: { op: "move", index: v }, path: edit.path };
        return;
      }
      const m = findEditMidpoint(edit.path, camera, sx, sy);
      if (m !== null) {
        // 閉じた形 (領域・ドメイン・閉じたポリライン) と開いたポリラインは頂点を足す、スケッチの線・円弧は曲げる。Shift で逆
        const primitive = !edit.path.closed && edit.path.points.length === 2;
        const bend = primitive !== e.shiftKey;
        dragRef.current = bend
          ? { kind: "path", x: sx, y: sy, target: edit.target, how: { op: "bend", index: m }, path: edit.path }
          : { kind: "path", x: sx, y: sy, target: edit.target, how: { op: "splitMove", edge: m }, path: editSplit(edit.path, m) };
        return;
      }
    }
    // 選んだものの上からは移動、それ以外は範囲選択 (動かさなければクリック)
    if (picked.length && onPicked(world, tol)) dragRef.current = { kind: "move", x: sx, y: sy, start: world };
    else dragRef.current = { kind: "box", x: sx, y: sy, start: world, shift: e.shiftKey };
  };

  const onPointerMove = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!camera) return;
    const [sx, sy] = localPoint(e, e.currentTarget);
    const raw = toWorld(camera, sx, sy);
    setHover(raw);
    const d = dragRef.current;
    // ドラッグしている形の自分自身には合わせない
    const exclude = d && (d.kind === "path" || d.kind === "radius") ? ownerOf(d.target) : null;
    const sn = snapAt(raw, { alt: e.altKey || NO_SNAP_TOOLS.includes(tool), exclude });
    setCursor(sn.point);
    setSnapMark(sn.mark);
    if (!d) return;
    if (d.kind === "pan") {
      const dx = e.clientX - d.lastX;
      const dy = e.clientY - d.lastY;
      d.lastX = e.clientX;
      d.lastY = e.clientY;
      markUserMoved();
      setCamera((c) => (c ? panBy(c, dx, dy) : c));
      return;
    }
    const moved = Math.hypot(sx - d.x, sy - d.y) >= CLICK_TOLERANCE_PX;
    if (!moved) return;
    if (d.kind === "path") {
      if (d.how.op === "bend") {
        const i = d.how.index;
        const a = d.path.points[i];
        const b = d.path.points[(i + 1) % d.path.points.length];
        // 弦の近くなら直線に戻す
        const chord = closestPoint({ kind: "line", a, b }, raw).dist * camera.scale;
        setPreview({ edit: editBend(d.path, i, chord <= STRAIGHTEN_PX ? 0 : bulgeThrough(a, raw, b)) });
      } else {
        const idx = d.how.op === "move" ? d.how.index : d.how.op === "splitMove" ? d.how.edge + 1 : 0;
        setPreview({ edit: editMove(d.path, idx, sn.point) });
      }
    } else if (d.kind === "radius") {
      const q = sn.point;
      setPreview({ radius: Math.max(0, Math.hypot(q[0] - d.center[0], q[1] - d.center[1])) });
    } else if (d.kind === "move") {
      let dx = raw[0] - d.start[0];
      let dy = raw[1] - d.start[1];
      if (vs.snap && snapStep > 0) {
        dx = snapValue(dx, snapStep);
        dy = snapValue(dy, snapStep);
      }
      setPreview({ move: [dx, dy] });
    } else if (d.kind === "box") {
      setPreview({ box: [d.start, raw] });
    }
  };

  const onPointerUp = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const d = dragRef.current;
    dragRef.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
    if (!d || !camera) return;
    if (d.kind === "pan") {
      setPanning(false);
      return;
    }
    const [sx, sy] = localPoint(e, e.currentTarget);
    const isClick = Math.hypot(sx - d.x, sy - d.y) < CLICK_TOLERANCE_PX;
    const raw = toWorld(camera, sx, sy);
    const pt = snap(raw, e.altKey);
    if (!isClick) {
      if (d.kind === "path" && preview.edit) {
        const result = preview.edit;
        let report: EdgeRemapReport | null = null;
        update(d.how.op === "bend" ? t("cad.bendEdge") : t("cad.editVertices"), (dr) => void (report = commitEdit(dr, d.target, d.how, result)));
        reportRemap(report);
      } else if (d.kind === "radius" && preview.radius && preview.radius > 0) {
        const r = preview.radius;
        update(t("cad.editRadius"), (dr) => commitRadius(dr, d.target, r));
      } else if (d.kind === "move" && preview.move && (preview.move[0] !== 0 || preview.move[1] !== 0)) {
        const [dx, dy] = preview.move;
        const items = picked;
        update(t("cad.moveItems"), (dr) => moveItems(dr, items, dx, dy));
      } else if (d.kind === "box" && preview.box) {
        const [a, b] = preview.box;
        useSelection.getState().pick(itemsInBox(project, a, b), d.shift ? "toggle" : "replace");
      }
      setPreview({});
      return;
    }
    setPreview({});
    // クリック
    const shift = d.kind === "click" || d.kind === "box" ? d.shift : false;
    if (tool === "select") selectAt(raw, shift);
    else if (isTransform) transformClick(pt, raw, shift);
    else if (isEdit) editClick(raw);
    else placePoint(pt, raw, nearFirst(sx, sy));
  };

  /** 変換を文書に (コピーならコピーを選ぶ) */
  const commitTransform = (m: Affine) => {
    const items = picked;
    let res: PickRef[] = [];
    update(t(`viewer.tool.${tool as "move"}`), (d) => void (res = transformItems(d, items, m, vs.transformCopy)));
    if (res.length) useSelection.getState().pick(res);
    resetDrawing();
  };

  const transformClick = (pt: Point, raw: Point, shift: boolean) => {
    // 選んでいなければ (Shift なら足す・外す) クリックで選ぶ
    if (shift || picked.length === 0) {
      selectAt(raw, shift);
      return;
    }
    if (pts.length === 0) {
      setPts([pt]);
      return;
    }
    const base = pts[0];
    const same = pt[0] === base[0] && pt[1] === base[1];
    if (tool === "move") commitTransform(translation(pt[0] - base[0], pt[1] - base[1]));
    else if (tool === "rotate") {
      if (!same) commitTransform(rotation(Math.atan2(pt[1] - base[1], pt[0] - base[0]), base));
    } else if (tool === "mirror") {
      if (!same) commitTransform(mirror(base, pt));
    } else if (tool === "scale") {
      if (pts.length === 1) {
        if (!same) setPts([base, pt]);
        return;
      }
      const r0 = Math.hypot(pts[1][0] - base[0], pts[1][1] - base[1]);
      const r1 = Math.hypot(pt[0] - base[0], pt[1] - base[1]);
      if (r0 > 0 && r1 > 0) commitTransform(scaling(r1 / r0, base));
    }
  };

  const warnEdit = (err: string) => logWarning(t("msg.source.app"), t(`editError.${err as "noTarget"}`));

  const editClick = (raw: Point) => {
    if (tool === "fillet" || tool === "chamfer") {
      const size = tool === "fillet" ? vs.filletRadius : vs.chamferDistance;
      const c = cornerAt(raw);
      if (c) {
        let res: ReturnType<typeof cornerEdit> | null = null;
        update(t(`viewer.tool.${tool}`), (d) => void (res = cornerEdit(d, c.target, c.index, size, tool)));
        const r = res as ReturnType<typeof cornerEdit> | null;
        if (r && !r.ok) warnEdit(r.error);
        else if (r) reportRemap(r.value);
        setPending(null);
        return;
      }
      const e = tool === "fillet" && camera ? findSketchAt(raw, sketch, EDGE_TOLERANCE_PX / camera.scale) : null;
      if (e?.kind === "line") {
        if (pending?.kind !== "filletLine") {
          setPending({ kind: "filletLine", id: e.id, pick: raw });
          return;
        }
        if (pending.id === e.id) return;
        const first = pending;
        let res: ReturnType<typeof filletSketchLines> | null = null;
        update(t("viewer.tool.fillet"), (d) => void (res = filletSketchLines(d, first.id, first.pick, e.id, raw, vs.filletRadius)));
        const r = res as ReturnType<typeof filletSketchLines> | null;
        if (r && !r.ok) warnEdit(r.error);
        setPending(null);
        return;
      }
      warnEdit("noCorner");
      return;
    }
    if (tool === "offset") {
      if (pending?.kind !== "offset") {
        const item = itemAt(raw);
        if (item) setPending({ kind: "offset", item });
        else warnEdit("noTarget");
        return;
      }
      const src = pending.item;
      let res: ReturnType<typeof offsetItem> | null = null;
      update(t("viewer.tool.offset"), (d) => void (res = offsetItem(d, src, vs.offsetDistance, raw)));
      const r = res as ReturnType<typeof offsetItem> | null;
      if (r && !r.ok) warnEdit(r.error);
      else if (r) useSelection.getState().pick([r.value]);
      setPending(null);
      return;
    }
    const e = camera ? findSketchAt(raw, sketch, EDGE_TOLERANCE_PX / camera.scale) : null;
    if (!e) {
      warnEdit("noTarget");
      return;
    }
    const cut = cutterSegs(project, e.id);
    let res: { ok: boolean; error?: string } | null = null;
    if (tool === "trim") update(t("viewer.tool.trim"), (d) => void (res = trimSketch(d, e.id, raw, cut)));
    else if (tool === "extend") update(t("viewer.tool.extend"), (d) => void (res = extendSketch(d, e.id, raw, cut)));
    const r = res as { ok: boolean; error?: string } | null;
    if (r && !r.ok && r.error) warnEdit(r.error);
  };

  /** 点を置く (クリック・数値入力)。closing は折れ線の最初の点に戻った */
  const placePoint = (pt: Point, raw: Point, closing: boolean) => {
    if (tool === "polyline") {
      if (pts.length === 0) {
        setPts([pt]);
        return;
      }
      const last = pts[pts.length - 1];
      if (closing) {
        finishPolyline(true, arcMode && through ? bulgeThrough(last, through, pts[0]) : 0);
        return;
      }
      if (arcMode) {
        if (!through) {
          setThrough(pt);
          return;
        }
        setPts([...pts, pt]);
        setBulges([...bulges, bulgeThrough(last, through, pt)]);
        setThrough(null);
        return;
      }
      setPts([...pts, pt]);
      setBulges([...bulges, 0]);
    } else if (tool === "line") {
      if (pts.length === 0) {
        setPts([pt]);
        return;
      }
      const a = pts[0];
      if (a[0] === pt[0] && a[1] === pt[1]) return;
      let id: string | null = null;
      update(t("cad.addSketch"), (dr) => void (id = addSketch(dr, { kind: "line", a, b: pt })));
      selectSketch(id);
      // 続けて次の線 (Enter・Esc で終わる)
      setPts([pt]);
    } else if (tool === "arc") {
      if (pts.length === 0) {
        setPts([pt]);
        return;
      }
      const a = pts[0];
      if (!through) {
        if (a[0] !== pt[0] || a[1] !== pt[1]) setThrough(pt);
        return;
      }
      const bu = bulgeThrough(a, through, pt);
      let id: string | null = null;
      if (a[0] !== pt[0] || a[1] !== pt[1]) update(t("cad.addSketch"), (dr) => void (id = addSketch(dr, bu ? { kind: "arc", a, b: pt, bulge: bu } : { kind: "line", a, b: pt })));
      selectSketch(id);
      resetDrawing();
    } else if (tool === "fill") {
      const face = arrangement?.shapeAt(raw) ?? null;
      if (!face) {
        logWarning(t("msg.source.app"), t("cad.noEnclosedArea"));
        return;
      }
      let id: string | null = null;
      update(t("cad.addRegion"), (dr) => void (id = addShapeRegion(dr, face)));
      if (id) useSelection.getState().selectRegion(id);
    } else if (tool === "probe") {
      if (scene.field) vs.setProbe(pt);
    } else if (TWO_POINT.includes(tool)) {
      if (pts.length === 0) {
        setPts([pt]);
        if (tool === "measure") setMeasure(null);
      } else {
        const p1 = pts[0];
        resetDrawing();
        commitTwoPoint(p1, pt);
      }
    }
  };

  const submitCoord = (): boolean => {
    if (coordText === null) return false;
    if (coordMode !== "point") {
      const text = coordText.trim().replace(/(°|deg)$/i, "");
      let v: number | null = null;
      if (coordMode === "length") {
        const r = parseQuantity(coordText, undefined, true, { lengthUnit, axisymmetric: false, vars: paramValues(project).values });
        v = r.ok ? r.value : null;
      } else {
        // 角度・倍率: 式 (パラメータも使える、P7f)
        v = evalExpression(text);
        if (v === null) {
          const r = parseParamExpr(text, { lengthUnit, axisymmetric: false, vars: paramValues(project).values });
          v = r?.ok ? r.value : null;
        }
      }
      if (v === null || !Number.isFinite(v) || (coordMode !== "angle" && !(v > 0))) {
        setCoordError(t("coordInput.error.value"));
        return false;
      }
      setCoordText(null);
      setCoordError(null);
      if (coordMode === "length") vs.setEditSize(tool === "fillet" ? "filletRadius" : tool === "chamfer" ? "chamferDistance" : "offsetDistance", v);
      else if (coordMode === "angle") commitTransform(rotation((v * Math.PI) / 180, pts[0]));
      else commitTransform(scaling(v, pts[0]));
      return true;
    }
    const last = pts.length ? pts[pts.length - 1] : null;
    const direction: Point | null = last && hover ? [hover[0] - last[0], hover[1] - last[1]] : null;
    const r = parseCoordInput(coordText, { last, direction, units: { lengthUnit, axisymmetric: false, vars: paramValues(project).values } });
    if (!r.ok) {
      setCoordError(t(`coordInput.error.${r.error}`));
      return false;
    }
    const pt = r.point as Point;
    // 折れ線の最初の点と同じ座標なら閉じる
    const tol = camera ? 1e-6 / camera.scale : 0;
    const closing = tool === "polyline" && pts.length >= 2 && Math.hypot(pt[0] - pts[0][0], pt[1] - pts[0][1]) <= tol && (pts.length >= 3 || bulges.some((b) => b) || through !== null);
    setCoordText(null);
    setCoordError(null);
    // 変換の道具は基点・先などの点として
    if (isTransform) transformClick(pt, pt, false);
    else placePoint(pt, pt, closing);
    return true;
  };

  const onDoubleClick = (e: ReactMouseEvent<HTMLCanvasElement>) => {
    if (!camera) return;
    const [sx, sy] = localPoint(e, e.currentTarget);
    if (tool === "select") {
      if (!edit?.path) return;
      const v = findVertexHandle(edit.path.points, camera, sx, sy);
      if (v === null) return;
      const next = editRemove(edit.path, v);
      if (!next) return;
      const tg = edit.target;
      let report: EdgeRemapReport | null = null;
      update(t("cad.deleteVertex"), (d) => void (report = commitEdit(d, tg, { op: "remove", index: v }, next)));
      reportRemap(report);
      return;
    }
    // 折れ線: 領域は閉じて、スケッチは開いたまま確定。線は続けて描くのをやめる
    if (tool === "polyline") finishPolyline(target === "region");
    else if (tool === "line") resetDrawing();
  };

  const onKeyDown = (e: ReactKeyboardEvent<HTMLCanvasElement>) => {
    // 点を置く道具で数字などを打つと数値入力の欄を開く
    if (POINT_TOOLS.includes(tool) && COORD_START.test(e.key) && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      setCoordText(e.key);
      setCoordError(null);
      return;
    }
    if (e.key === "F3") {
      e.preventDefault();
      vs.setSnap(!vs.snap);
      return;
    }
    const busy = pts.length > 0 || through !== null || pending !== null || dragRef.current !== null || !!preview.edit || !!preview.move || preview.radius != null || !!preview.box;
    if (e.key === "Escape") {
      e.preventDefault();
      resetDrawing();
      setPending(null);
      setPreview({});
      dragRef.current = null;
      setMeasure(null);
      vs.setProbe(null);
      if (!busy) {
        if (tool !== "select") vs.setTool("select");
        else if (useSelection.getState().picked.length) useSelection.getState().selectRegion(null);
      }
    } else if (e.key === "Enter" && (tool === "polyline" || tool === "line" || tool === "arc")) {
      e.preventDefault();
      if (tool === "polyline") finishPolyline(target === "region");
      else resetDrawing();
    } else if (e.key === "Backspace" && (tool === "polyline" || tool === "line" || tool === "arc") && (pts.length > 0 || through)) {
      e.preventDefault();
      if (through) setThrough(null);
      else {
        setPts((p) => p.slice(0, -1));
        setBulges((b) => b.slice(0, Math.max(0, pts.length - 2)));
      }
    } else if (tool === "polyline" && (e.key === "a" || e.key === "A") && !e.ctrlKey && !e.metaKey) {
      e.preventDefault();
      setArcMode(true);
    } else if (tool === "polyline" && (e.key === "l" || e.key === "L") && !e.ctrlKey && !e.metaKey) {
      e.preventDefault();
      setArcMode(false);
      setThrough(null);
    } else if ((e.key === "Delete" || e.key === "Backspace") && tool === "select") {
      e.preventDefault();
      deleteSelected();
    } else if (tool === "select" && picked.length && camera && e.key.startsWith("Arrow")) {
      e.preventDefault();
      const step = (gridStep(camera) / 10) * (e.shiftKey ? 10 : 1);
      if (e.key === "ArrowUp") nudge(0, step);
      else if (e.key === "ArrowDown") nudge(0, -step);
      else if (e.key === "ArrowLeft") nudge(-step, 0);
      else if (e.key === "ArrowRight") nudge(step, 0);
    } else if ((e.key === "Home" || e.key === "f" || e.key === "F") && !e.ctrlKey && !e.metaKey) {
      e.preventDefault();
      vs.requestFit();
    }
  };

  const closing = camera && cursor && pts.length ? nearFirst(...(toScreen(camera, hover ?? cursor) as [number, number])) : false;

  return {
    onPointerDown,
    onPointerMove,
    onPointerUp,
    onPointerCancel: () => {
      dragRef.current = null;
      setPanning(false);
      setPreview({});
    },
    onPointerLeave: () => {
      setCursor(null);
      setHover(null);
      setSnapMark(null);
    },
    onDoubleClick,
    onKeyDown,
    cursor,
    hover,
    panning,
    drawing: { tool, target, pts, bulges, through, arcMode, cursor, closing, selected: picked.length, pending: pending !== null },
    preview: transformAffine ? { ...preview, affine: transformAffine } : preview,
    measure: measure ?? (tool === "measure" && pts.length === 1 && cursor ? [pts[0], cursor] : null),
    fillPreview,
    edit,
    picked,
    snapMark,
    coordText,
    coordMode,
    toolPreview,
    coordError,
    setCoordText: (text: string) => {
      setCoordText(text);
      setCoordError(null);
    },
    submitCoord,
    closeCoord: () => {
      setCoordText(null);
      setCoordError(null);
    },
  };
}

function hitRect(pt: Point, a: Point, b: Point, tol: number): boolean {
  const x0 = Math.min(a[0], b[0]) - tol;
  const x1 = Math.max(a[0], b[0]) + tol;
  const y0 = Math.min(a[1], b[1]) - tol;
  const y1 = Math.max(a[1], b[1]) + tol;
  const inside = pt[0] >= x0 && pt[0] <= x1 && pt[1] >= y0 && pt[1] <= y1;
  const deep = pt[0] > x0 + 2 * tol && pt[0] < x1 - 2 * tol && pt[1] > y0 + 2 * tol && pt[1] < y1 - 2 * tol;
  // 枠の近くだけ (中は下の領域を選べるように)
  return inside && !deep;
}

/** 作図の道具の説明 (状況ごと) */
/** 編集の道具の半径・長さ (説明に出す) */
export interface EditSizes {
  filletRadius: number;
  chamferDistance: number;
  offsetDistance: number;
}

export function toolHint(tool: Tool, d: Pick<Drawing, "pts" | "through" | "arcMode" | "target" | "selected" | "pending">, t: TFunction, sizes?: EditSizes, unit: "mm" | "um" = "mm"): string {
  const n = d.pts.length;
  const len = (m: number) => `${formatNumber(toDisplayLength(m, unit))} ${lengthUnitLabel(unit)}`;
  if ((TRANSFORM_TOOLS as Tool[]).includes(tool) && d.selected === 0) return t("viewer.hint.transformSelect");
  switch (tool) {
    case "move":
      return n === 0 ? t("viewer.hint.moveBase") : t("viewer.hint.moveTo");
    case "rotate":
      return n === 0 ? t("viewer.hint.rotateCenter") : t("viewer.hint.rotateAngle");
    case "mirror":
      return n === 0 ? t("viewer.hint.mirrorFirst") : t("viewer.hint.mirrorSecond");
    case "scale":
      return n === 0 ? t("viewer.hint.scaleBase") : n === 1 ? t("viewer.hint.scaleFactor") : t("viewer.hint.scaleTo");
    case "fillet":
      return d.pending ? t("viewer.hint.filletSecond") : t("viewer.hint.fillet", { r: sizes ? len(sizes.filletRadius) : "" });
    case "chamfer":
      return t("viewer.hint.chamfer", { d: sizes ? len(sizes.chamferDistance) : "" });
    case "offset":
      return d.pending ? t("viewer.hint.offsetSide") : t("viewer.hint.offset", { d: sizes ? len(sizes.offsetDistance) : "" });
    case "trim":
      return t("viewer.hint.trim");
    case "extend":
      return t("viewer.hint.extend");
    case "select":
      return t("viewer.hint.select");
    case "polyline":
      if (n === 0) return t("viewer.hint.polylineStart");
      if (d.arcMode) return d.through ? t("viewer.hint.polylineArcEnd") : t("viewer.hint.polylineArcThrough");
      return d.target === "sketch" ? t("viewer.hint.polylineNextSketch") : t("viewer.hint.polylineNext");
    case "rect":
      return n === 0 ? t("viewer.hint.rectStart") : t("viewer.hint.rectEnd");
    case "circle":
      return n === 0 ? t("viewer.hint.circleStart") : t("viewer.hint.circleEnd");
    case "line":
      return n === 0 ? t("viewer.hint.lineStart") : t("viewer.hint.sketchLineNext");
    case "arc":
      return n === 0 ? t("viewer.hint.arcStart") : d.through ? t("viewer.hint.arcEnd") : t("viewer.hint.arcThrough");
    case "fill":
      return t("viewer.hint.fill");
    case "sheathline":
      return n === 0 ? t("viewer.hint.sheathStart") : t("viewer.hint.sheathEnd");
    case "probe":
      return t("viewer.hint.probe");
    case "measure":
      return n === 0 ? t("viewer.hint.measureStart") : t("viewer.hint.measureEnd");
    default:
      return n === 0 ? t("viewer.hint.lineStart") : t("viewer.hint.lineEnd");
  }
}
