// キャンバスの道具の操作 (P7b で Viewer から分けた):
// - 選択: クリック (Shift で足す・外す)、何も無い所・選んでいない所からのドラッグで範囲選択、選んだものの上から
//   ドラッグで移動 (複数まとめて)、編集している形 (1 つ選んだ領域・スケッチ、またはドメイン) の頂点・辺の中点・
//   半径のハンドル。辺の中点は頂点を足す (Shift で曲げる。スケッチの線・円弧は逆)。頂点のダブルクリックで削除
// - 作図: 折れ線 (A で円弧の段、L で直線に戻る、最初の点で閉じる)・矩形・円 (描く先は領域かスケッチ)、線 (続けて描く)、
//   3 点の円弧、囲まれた所から領域
// - 配置 (2 点)、プローブ・計測、移動 (中ボタンか Space + ドラッグ)

import type { TFunction } from "i18next";
import type { Draft } from "immer";
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
import { bulgeThrough, closestPoint, midpoint, pathSegs, segFromBulge, type Seg } from "../cad/geom";
import { bulgesOf, type PathData } from "../cad/path";
import { useDocument } from "../model/documentStore";
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
import { domainPath, edgeIdOf, regionPath, type Point, type Project } from "../model/project";
import { addCircleRegionAt, addPolygonRegion } from "../model/regionOps";
import { PLACEMENT_NODES, PLACEMENT_PATHS, useSelection, type PickRef, type PlacementKind } from "../model/selection";
import { addSketch, editBend, editMove, editRemove, editSplit, edgeCountOf, sketchOf, sketchSegs, type EditPath } from "../model/sketch";
import { getIn, setIn } from "../schema/schema";
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
import type { Drawing, Placement, Preview } from "./overlay";
import type { Scene } from "./scene";
import { useViewer, type Tool } from "./viewerStore";

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

/** 囲まれた所を探す相手: ドメインの辺・領域の輪郭・スケッチ */
function allCurves(p: Project): Seg[] {
  const segs: Seg[] = [];
  const d = domainPath(p);
  segs.push(...pathSegs(d.polygon, bulgesOf(d)));
  for (const r of p.geometry.regions) {
    const rp = regionPath(r);
    segs.push(...pathSegs(rp.polygon, bulgesOf(rp)));
  }
  for (const e of sketchOf(p)) segs.push(...sketchSegs(e));
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
  fillPreview: PathData | null;
  edit: EditObject | null;
  picked: PickRef[];
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
  const dragRef = useRef<Drag | null>(null);
  const spaceRef = useRef(false);
  const tool = vs.tool;
  const target = vs.drawTarget;
  const sketch = sketchOf(project);
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
  }, [tool]);
  // 選択・形が変わったら (元に戻す・削除など) ドラッグをやめる
  useEffect(() => {
    dragRef.current = null;
    setPreview({});
  }, [picked, activeNode, project.geometry, project.cad]);

  const snapStep = camera ? gridStep(camera) / 10 : 0;
  const snap = (p: Point): Point => (vs.snap && snapStep > 0 ? snapPoint(p, snapStep) : p);

  // 囲まれた所から領域を作る道具: 平面の配置は文書が変わったときだけ作り直す
  const arrangement = useMemo(() => (tool === "fill" ? new Arrangement(allCurves(project)) : null), [tool, project]);
  const fillPreview = useMemo(() => (arrangement && hover ? arrangement.faceAt(hover) : null), [arrangement, hover]);

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
    const hit = findRegionAt(raw, project.geometry.regions, tol);
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
    setCursor(snap(raw));
    const d = dragRef.current;
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
        setPreview({ edit: editMove(d.path, idx, snap(raw)) });
      }
    } else if (d.kind === "radius") {
      const q = snap(raw);
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
    const pt = snap(raw);
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
    if (tool === "select") {
      selectAt(raw, shift);
    } else if (tool === "polyline") {
      if (pts.length === 0) {
        setPts([pt]);
        return;
      }
      const last = pts[pts.length - 1];
      if (nearFirst(sx, sy)) {
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
      const face = arrangement?.faceAt(raw) ?? null;
      if (!face) {
        logWarning(t("msg.source.app"), t("cad.noEnclosedArea"));
        return;
      }
      let id: string | null = null;
      update(t("cad.addRegion"), (dr) => void (id = addPolygonRegion(dr, face.polygon as Point[], face.bulges)));
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
    const busy = pts.length > 0 || through !== null || dragRef.current !== null || !!preview.edit || !!preview.move || preview.radius != null || !!preview.box;
    if (e.key === "Escape") {
      e.preventDefault();
      resetDrawing();
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
    },
    onDoubleClick,
    onKeyDown,
    cursor,
    hover,
    panning,
    drawing: { tool, target, pts, bulges, through, arcMode, cursor, closing },
    preview,
    measure: measure ?? (tool === "measure" && pts.length === 1 && cursor ? [pts[0], cursor] : null),
    fillPreview,
    edit,
    picked,
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
export function toolHint(tool: Tool, d: Pick<Drawing, "pts" | "through" | "arcMode" | "target">, t: TFunction): string {
  const n = d.pts.length;
  switch (tool) {
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
