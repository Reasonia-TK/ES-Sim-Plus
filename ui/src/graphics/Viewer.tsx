// 2D ビューア: 3 枚の層 (下: 背景・グリッド、中: WebGL の場・メッシュ・等値線・粒子、上: ジオメトリ・配置物・
// 作図・プローブ・カラーバー・ルーラー) と、作図・編集・配置・調べる道具のマウスとキーボードの操作。
// 拡大はホイール (カーソル中心)、移動は中ボタンか Space + ドラッグ (v1 と同じ)。

import type { TFunction } from "i18next";
import type { Draft } from "immer";
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type PointerEvent as ReactPointerEvent } from "react";
import { useTranslation } from "react-i18next";
import { errorText, logError, logWarning } from "../app/messages";
import { saveBinaryFile, saveTextFile } from "../io/fileAccess";
import { useDocument } from "../model/documentStore";
import { PLACEMENT_NODES, PLACEMENT_PATHS, useSelection, type PlacementKind } from "../model/selection";
import {
  placeCollector,
  placeEdgeMeshSize,
  placeEedfRegion,
  placeEmitter,
  placeGasBoundary,
  placeSheathLine,
  MAX_COLLECTORS,
  MAX_EEDF_REGIONS,
  MAX_SHEATH_LINES,
  type PlaceResult,
} from "../model/placements";
import { coordOf, polygonBounds, type Point, type Project } from "../model/project";
import {
  addCircleRegionAt,
  addPolygonRegion,
  deleteRegion,
  moveRegion,
  removeRegionVertex,
  setCircleRadius,
  setRegionPolygon,
} from "../model/regionOps";
import { usePrefs } from "../prefs/prefs";
import { getIn, setIn } from "../schema/schema";
import { formatNumber, lengthUnitLabel, toDisplayLength } from "../util/format";
import { fitCamera, gridStep, panBy, snapPoint, snapValue, toWorld, zoomAt, ZOOM_STEP, type Camera } from "./camera";
import { parseColor } from "./color";
import { composePng, fieldCsv, stamp } from "./exporting";
import { resolveRange } from "./fieldScale";
import { GlRenderer, type MeshDraw, type PointDraw, type Rgba, type SegmentDraw } from "./gl/renderer";
import {
  CLICK_TOLERANCE_PX,
  dedupeTail,
  EDGE_TOLERANCE_PX,
  findMidpointHandle,
  findRegionAt,
  findVertexHandle,
  hitCircle,
  hitRadiusHandle,
  hitRegion,
  hitSegment,
  insertMidpoint,
  rectFromCorners,
} from "./hitTest";
import { isoLevels, isolineSegments } from "./isolines";
import { drawBase, drawOverlay, readColors, rulerSize, type Arrow, type OverlayState, type Placement, type Preview } from "./overlay";
import { amrBoxesOf, emitterOf, placementsOf } from "./projectOverlays";
import { sampleField, sampleVector, statsOf, type Scene } from "./scene";
import { PlaybackBar } from "./PlaybackBar";
import { ViewerRfStrip } from "./ViewerRfStrip";
import type { ActiveScene } from "./useScene";
import { RULER_FONTS, useViewer, type Tool } from "./viewerStore";
import { ViewerToolbar } from "./ViewerToolbar";

type Drag =
  | { kind: "pan"; lastX: number; lastY: number; x: number; y: number }
  | { kind: "click"; x: number; y: number }
  | { kind: "vertex"; x: number; y: number; id: string; index: number; poly: Point[] }
  | { kind: "radius"; x: number; y: number; id: string }
  | { kind: "move"; x: number; y: number; id: string; start: Point };

/** 2 点で決める道具 */
const TWO_POINT: Tool[] = ["rect", "circle", "profile", "emitter", "collector", "gasbc", "eedfbox", "meshref", "sheathline", "measure"];

/** ポインタを捕まえる (キャンバスの外へ出てもドラッグを続ける。作ったイベントでは捕まえられないので無視) */
function capture(el: HTMLElement, id: number): void {
  try {
    el.setPointerCapture(id);
  } catch {
    // 捕まえられないポインタ
  }
}

/** 画面の点 → 要素内の座標 (CSS px) */
function localPoint(e: { clientX: number; clientY: number }, el: HTMLElement): [number, number] {
  const r = el.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}

function axisNames(p: Project): [string, string] {
  const c = coordOf(p);
  return c === "rz" ? ["z", "r"] : c === "rz_x0" ? ["r", "z"] : ["x", "y"];
}

function setCanvasSize(cv: HTMLCanvasElement, w: number, h: number, dpr: number): CanvasRenderingContext2D | null {
  const W = Math.max(1, Math.round(w * dpr));
  const H = Math.max(1, Math.round(h * dpr));
  if (cv.width !== W) cv.width = W;
  if (cv.height !== H) cv.height = H;
  let ctx: CanvasRenderingContext2D | null = null;
  try {
    ctx = cv.getContext("2d");
  } catch {
    ctx = null;
  }
  ctx?.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/** 矢印を置く点 (画面の格子、最大およそ 600 本、v1 と同じ本数) */
function vectorArrows(scene: Scene, cam: Camera, w: number, h: number, top: number): Arrow[] {
  const vf = scene.vectors;
  if (!vf) return [];
  const spacing = Math.max(22, Math.sqrt((w * h) / 600));
  const out: Arrow[] = [];
  for (let sy = top + spacing / 2; sy < h; sy += spacing) {
    for (let sx = top + spacing / 2; sx < w; sx += spacing) {
      const [x, y] = toWorld(cam, sx, sy);
      const e = sampleVector(vf, x, y);
      if (!e) continue;
      const m = Math.hypot(e[0], e[1]);
      if (!(m > 1e-12)) continue;
      out.push({ x: sx, y: sy, dx: e[0] / m, dy: -e[1] / m });
    }
  }
  return out;
}

export function Viewer({ active }: { active: ActiveScene }) {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const docSerial = useDocument((s) => s.docSerial);
  const selectedRegion = useSelection((s) => s.selectedRegion);
  const selectedPlacement = useSelection((s) => s.selectedPlacement);
  const lengthUnit = usePrefs((s) => s.lengthUnit);
  const theme = usePrefs((s) => s.theme);
  const vs = useViewer();
  const { scene, controls, run } = active;

  const wrapRef = useRef<HTMLDivElement>(null);
  const baseRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<HTMLCanvasElement>(null);
  const topRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<GlRenderer | null>(null);
  const [glOk, setGlOk] = useState(true);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [camera, setCamera] = useState<Camera | null>(null);
  const userMovedRef = useRef(false);
  const [cursor, setCursor] = useState<Point | null>(null);
  const [hover, setHover] = useState<Point | null>(null);
  const [pts, setPts] = useState<Point[]>([]);
  const [preview, setPreview] = useState<Preview>({});
  const [measure, setMeasure] = useState<[Point, Point] | null>(null);
  const [panning, setPanning] = useState(false);
  const dragRef = useRef<Drag | null>(null);
  const spaceRef = useRef(false);
  const [redraw, setRedraw] = useState(0);

  // ---- 大きさ・WebGL ----

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    // 最初の大きさはすぐに読む (ResizeObserver は描画の度にしか届かず、ウィンドウが隠れていると届かない)
    const r0 = el.getBoundingClientRect();
    setSize({ w: Math.floor(r0.width), h: Math.floor(r0.height) });
    const ro = new ResizeObserver(([e]) => setSize({ w: Math.floor(e.contentRect.width), h: Math.floor(e.contentRect.height) }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const cv = glRef.current;
    if (!cv) return;
    let r: GlRenderer | null = null;
    try {
      r = GlRenderer.create(cv);
    } catch (e) {
      logWarning(t("msg.source.app"), t("viewer.webglFailed", { error: String(e) }));
      r = null;
    }
    rendererRef.current = r;
    setGlOk(r !== null);
    if (r) r.onRestored = () => setRedraw((n) => n + 1);
    return () => {
      r?.dispose();
      rendererRef.current = null;
    };
    // t は言語が変わっても作り直さない
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- カメラ ----

  const rulers = vs.overlays.rulers;
  const rulerPx = rulers ? rulerSize(RULER_FONTS[vs.rulerFont]) : 0;
  const fit = (w = size.w, h = size.h): Camera | null => {
    if (w <= 0 || h <= 0) return null;
    const b = polygonBounds(project.geometry.domain.polygon);
    const c = fitCamera(b, Math.max(1, w - rulerPx), Math.max(1, h - rulerPx));
    return { ...c, ox: c.ox + rulerPx, oy: c.oy + rulerPx };
  };
  // 初回・別の文書・全体表示の要求・(まだ動かしていなければ) 大きさの変化で合わせる
  const fitKey = `${docSerial}|${vs.fitSerial}`;
  const lastFitKey = useRef("");
  useEffect(() => {
    if (size.w === 0 || size.h === 0) return;
    if (lastFitKey.current !== fitKey || camera === null || !userMovedRef.current) {
      lastFitKey.current = fitKey;
      userMovedRef.current = false;
      setCamera(fit());
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitKey, size.w, size.h, rulerPx]);

  const zoomLimits = useMemo(() => {
    const f = fit(Math.max(size.w, 100), Math.max(size.h, 100));
    const s = f?.scale ?? 1;
    return { min: s / 200, max: s * 1e6 };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.geometry.domain.polygon, size.w, size.h]);

  // ホイール (React の onWheel は passive なので自前で付ける)
  useEffect(() => {
    const el = topRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const [sx, sy] = localPoint(e, el);
      const dy = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaMode === 2 ? e.deltaY * 400 : e.deltaY;
      const factor = Math.min(2, Math.max(0.5, Math.pow(ZOOM_STEP, -dy / 100)));
      userMovedRef.current = true;
      setCamera((c) => (c ? zoomAt(c, sx, sy, factor, zoomLimits) : c));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoomLimits]);

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

  // ---- 表示するもの ----

  const playback = "playback" in scene ? scene.playback : undefined;
  // 塗る量が変わったら手動の範囲は外す (単位が違う)
  const fieldKey = scene.field ? `${scene.field.label}|${scene.field.unit}` : "";
  const lastFieldKey = useRef(fieldKey);
  useEffect(() => {
    if (lastFieldKey.current !== fieldKey && (vs.range.min !== null || vs.range.max !== null)) vs.setRange({ min: null, max: null });
    lastFieldKey.current = fieldKey;
  }, [fieldKey, vs]);
  const log = controls.log;
  const range = useMemo(() => (scene.field ? resolveRange(statsOf(scene.field), log, vs.range) : null), [scene.field, log, vs.range]);
  const isoSegs = useMemo(() => {
    if (!vs.overlays.isolines || !scene.iso) return null;
    const iso = scene.iso;
    let levels: number[];
    if (iso === scene.field && range) {
      // 塗っている量と同じなら表示の範囲で (対数なら対数で等分)
      const lv = isoLevels(range.lo, range.hi, 15);
      levels = range.log ? lv.map((x) => Math.pow(10, x)) : lv;
    } else {
      const s = statsOf(iso);
      levels = isoLevels(s.min, s.max, 15);
    }
    return isolineSegments(iso.mesh, iso.values, levels);
  }, [vs.overlays.isolines, scene.iso, scene.field, range]);
  const placements = useMemo(() => placementsOf(project, vs.overlays), [project, vs.overlays]);
  const emitter = useMemo(() => (vs.overlays.emitter ? emitterOf(project) : null), [project, vs.overlays.emitter]);
  const amrBoxes = useMemo(() => amrBoxesOf(project), [project]);
  const origin = useMemo<[number, number]>(() => {
    const b = polygonBounds(project.geometry.domain.polygon);
    return [(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2];
  }, [project.geometry.domain.polygon]);

  const snapStep = camera ? gridStep(camera) / 10 : 0;
  const snap = (p: Point): Point => (vs.snap && snapStep > 0 ? snapPoint(p, snapStep) : p);

  // ツールを変えたら作図中の点を捨てる (v1 と同じ)。計測はそのまま残さない
  useEffect(() => {
    setPts([]);
    setMeasure(null);
  }, [vs.tool]);
  // 選択が変わったら (元に戻す・削除など) ドラッグをやめる
  useEffect(() => {
    dragRef.current = null;
    setPreview({});
  }, [selectedRegion, project.geometry.regions]);

  // ---- 描く ----

  const axes = axisNames(project);
  const fieldUnit = scene.field?.unit ?? "";
  const probeLines = (pt: Point): string[] => {
    const unit = lengthUnitLabel(lengthUnit);
    const coord = `${axes[0]}: ${toDisplayLength(pt[0], lengthUnit).toFixed(3)} ${unit}, ${axes[1]}: ${toDisplayLength(pt[1], lengthUnit).toFixed(3)} ${unit}`;
    if (!scene.field) return [coord, t("viewer.noField")];
    const v = sampleField(scene.field, pt[0], pt[1]);
    return [coord, v === null ? t("viewer.outside") : `${scene.field.label} = ${formatNumber(v)} ${fieldUnit}`];
  };

  /** 3 枚の層を描く (書き出しでは WebGL の層を描いた直後に読むので同期で呼ぶ) */
  const paint = (): boolean => {
    // 大きさが変わっていたら測り直してから描く (隣の欄の開閉など。ResizeObserver が遅れても合わせる)
    const wrap = wrapRef.current;
    if (wrap) {
      const r = wrap.getBoundingClientRect();
      const w = Math.floor(r.width);
      const h = Math.floor(r.height);
      if (w !== size.w || h !== size.h) {
        setSize({ w, h });
        return false;
      }
    }
    if (!camera || size.w === 0 || size.h === 0) return false;
    {
      const colors = readColors();
      const dpr = window.devicePixelRatio || 1;
      const view = { camera, width: size.w, height: size.h };
      const base = baseRef.current;
      const bctx = base ? setCanvasSize(base, size.w, size.h, dpr) : null;
      if (bctx) drawBase(bctx, view, project, colors, vs.overlays.grid);

      const renderer = rendererRef.current;
      if (renderer) {
        const meshes: MeshDraw[] = [];
        if (scene.field && range) {
          meshes.push({
            mesh: scene.field.mesh,
            mode: "field",
            field: { values: scene.field.values, location: scene.field.location, range, colormap: vs.colormap, alpha: 1 },
            wire: vs.overlays.mesh ? { color: parseColor(colors.wireField), width: 1 } : null,
          });
        } else if (scene.mesh) {
          const regionColors: Rgba[] = project.geometry.regions.map((r) => parseColor(colors[r.type] ?? colors.edge, 0.16));
          meshes.push({ mesh: scene.mesh, mode: scene.fillRegions ? "regions" : "none", regionColors, wire: { color: parseColor(colors.wire), width: 1 } });
        }
        const segments: SegmentDraw[] = [];
        if (isoSegs && isoSegs.length) segments.push({ data: isoSegs, color: parseColor(colors.isoline), width: 1 });
        // 軌道 (と吸収点) は「軌道」、粒子は「粒子」で出し分ける。シース端はいつも
        for (const s of scene.segments) {
          if (s.id === "trajectories" && !vs.overlays.trajectories) continue;
          segments.push({ data: s.segments, color: parseColor(s.color), width: s.width });
        }
        const points: PointDraw[] = scene.points
          .filter((p) => (p.id === "absorbed" ? vs.overlays.trajectories : p.id === "sheathS" ? vs.overlays.sheathLines : vs.overlays.particles))
          .map((p) => ({ data: p.xy, color: parseColor(p.color), size: p.size }));
        renderer.render({ width: size.w, height: size.h, dpr, camera, origin, meshes, segments, points });
      }

      const top = topRef.current;
      const tctx = top ? setCanvasSize(top, size.w, size.h, dpr) : null;
      if (!tctx) return true;
      const probe = vs.tool === "probe" && vs.probe ? { pt: vs.probe, lines: probeLines(vs.probe) } : null;
      const state: OverlayState = {
        view,
        project,
        colors,
        lengthUnit,
        axisNames: axes,
        overlays: vs.overlays,
        fieldShown: scene.field !== null || scene.mesh !== null,
        selectedRegion,
        selectedPlacement: selectedPlacement as OverlayState["selectedPlacement"],
        tool: vs.tool,
        preview,
        drawing: { tool: vs.tool, pts, cursor },
        placements: placements as Placement[],
        emitter,
        amrBoxes,
        profile: vs.profile,
        measure: measure ?? (vs.tool === "measure" && pts.length === 1 && cursor ? [pts[0], cursor] : null),
        arrows: vs.overlays.vectors ? vectorArrows(scene, camera, size.w, size.h, rulerPx) : [],
        colorbar: scene.field && range ? { colormap: vs.colormap, range, label: scene.field.label, unit: scene.field.unit } : null,
        probe,
        cursor,
        rulerFont: RULER_FONTS[vs.rulerFont],
      };
      drawOverlay(tctx, state);
    }
    return true;
  };
  const paintRef = useRef(paint);
  useLayoutEffect(() => {
    paintRef.current = paint;
  });
  // 描き直しは React の反映ごと (イベントごとにまとまる)。requestAnimationFrame はウィンドウが隠れていると
  // 止まるので使わない
  useEffect(() => {
    paintRef.current();
  });

  const exportPng = async () => {
    const layers = [baseRef.current, glRef.current, topRef.current].filter((c): c is HTMLCanvasElement => c !== null);
    if (!paintRef.current() || layers.length === 0) return;
    try {
      const blob = await composePng(layers);
      await saveBinaryFile(`es-sim-view-${stamp()}.png`, blob, "png", "PNG");
    } catch (e) {
      logError(t("msg.source.app"), t("viewer.exportFailed", { error: errorText(e) }));
    }
  };
  const exportCsv = async () => {
    if (!scene.field) return;
    try {
      const name = (controls.quantity ?? scene.field.label).replace(/[^A-Za-z0-9_]+/g, "") || "field";
      await saveTextFile(`${name}-${stamp()}.csv`, fieldCsv(scene.field, axes), "csv", "CSV");
    } catch (e) {
      logError(t("msg.source.app"), t("viewer.exportFailed", { error: errorText(e) }));
    }
  };

  // ---- 文書を変える操作 ----

  const edit = (label: string, recipe: (d: Draft<Project>) => void) => useDocument.getState().update(label, recipe);

  const commitTwoPoint = (tool: Tool, p1: Point, p2: Point) => {
    const same = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) === 0;
    if (tool === "rect") {
      if (same || p1[0] === p2[0] || p1[1] === p2[1]) return;
      let id: string | null = null;
      edit(t("cad.addRegion"), (d) => void (id = addPolygonRegion(d, rectFromCorners(p1, p2))));
      if (id) useSelection.getState().selectRegion(id);
    } else if (tool === "circle") {
      const r = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
      let id: string | null = null;
      if (r > 0) edit(t("cad.addRegion"), (d) => void (id = addCircleRegionAt(d, p1, r)));
      if (id) useSelection.getState().selectRegion(id);
    } else if (tool === "profile") {
      if (!same) vs.setProfile([p1, p2]);
    } else if (tool === "measure") {
      setMeasure([p1, p2]);
    } else {
      if (same) return;
      const place: Record<string, [(d: Parameters<typeof placeEmitter>[0], a: Point, b: Point) => PlaceResult, string, PlacementKind | null, string, number]> = {
        emitter: [placeEmitter, t("cad.placeEmitter"), null, "study:trace", 0],
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
      edit(label, (d) => void (res = fn(d, p1, p2)));
      const r = res as PlaceResult | null;
      if (r?.index === null) logWarning(t("msg.source.app"), t("cad.listFull", { what: label, max }));
      const sel = useSelection.getState();
      sel.select(node);
      if (kind && r && r.index !== null) sel.selectPlacement({ kind, index: r.index });
    }
  };

  const finishPolyline = (list: Point[]) => {
    const poly = dedupeTail(list);
    setPts([]);
    if (poly.length < 3) return;
    let id: string | null = null;
    edit(t("cad.addRegion"), (d) => void (id = addPolygonRegion(d, poly)));
    if (id) useSelection.getState().selectRegion(id);
  };

  const deleteSelected = () => {
    const sel = useSelection.getState();
    if (sel.selectedRegion) {
      const id = sel.selectedRegion;
      edit(t("cad.deleteRegion"), (d) => deleteRegion(d, id));
      sel.selectRegion(null);
    } else if (sel.selectedPlacement) {
      const { kind, index } = sel.selectedPlacement;
      const path = PLACEMENT_PATHS[kind];
      edit(t("cad.deletePlacement"), (d) => {
        const items = getIn(d, path);
        if (Array.isArray(items)) setIn(d, path, items.filter((_, i) => i !== index));
      });
      sel.selectPlacement(null);
    }
  };

  const nudge = (dx: number, dy: number) => {
    const id = useSelection.getState().selectedRegion;
    if (id) edit(t("cad.moveRegion"), (d) => moveRegion(d, id, dx, dy));
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
    dragRef.current = { kind: "click", x: sx, y: sy };
    if (vs.tool !== "select") return;
    const sel = project.geometry.regions.find((r) => r.id === selectedRegion);
    if (!sel) return;
    const world = toWorld(camera, sx, sy);
    const tol = EDGE_TOLERANCE_PX / camera.scale;
    if (sel.shape) {
      if (hitRadiusHandle(sel.shape, camera, sx, sy)) dragRef.current = { kind: "radius", x: sx, y: sy, id: sel.id };
      else if (hitCircle(world, sel.shape, tol)) dragRef.current = { kind: "move", x: sx, y: sy, id: sel.id, start: world };
      return;
    }
    const poly = sel.polygon ?? [];
    const v = findVertexHandle(poly, camera, sx, sy);
    if (v !== null) {
      dragRef.current = { kind: "vertex", x: sx, y: sy, id: sel.id, index: v, poly: poly.map((q) => [q[0], q[1]] as Point) };
      return;
    }
    const m = findMidpointHandle(poly, camera, sx, sy);
    if (m !== null) {
      dragRef.current = { kind: "vertex", x: sx, y: sy, id: sel.id, index: m + 1, poly: insertMidpoint(poly, m) };
      return;
    }
    if (hitRegion(world, sel, tol)) dragRef.current = { kind: "move", x: sx, y: sy, id: sel.id, start: world };
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
      userMovedRef.current = true;
      setCamera((c) => (c ? panBy(c, dx, dy) : c));
      return;
    }
    const moved = Math.hypot(sx - d.x, sy - d.y) >= CLICK_TOLERANCE_PX;
    if (!moved) return;
    if (d.kind === "vertex") {
      const poly = d.poly.slice();
      poly[d.index] = snap(raw);
      setPreview({ polygon: poly });
    } else if (d.kind === "radius") {
      const sel = project.geometry.regions.find((r) => r.id === d.id);
      if (sel?.shape) {
        const q = snap(raw);
        setPreview({ radius: Math.max(0, Math.hypot(q[0] - sel.shape.center[0], q[1] - sel.shape.center[1])) });
      }
    } else if (d.kind === "move") {
      let dx = raw[0] - d.start[0];
      let dy = raw[1] - d.start[1];
      if (vs.snap && snapStep > 0) {
        dx = snapValue(dx, snapStep);
        dy = snapValue(dy, snapStep);
      }
      setPreview({ move: [dx, dy] });
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
      if (d.kind === "vertex" && preview.polygon) {
        const poly = preview.polygon;
        edit(t("cad.editVertices"), (dr) => setRegionPolygon(dr, d.id, poly));
      } else if (d.kind === "radius" && preview.radius && preview.radius > 0) {
        const r = preview.radius;
        edit(t("cad.editRadius"), (dr) => setCircleRadius(dr, d.id, r));
      } else if (d.kind === "move" && preview.move && (preview.move[0] !== 0 || preview.move[1] !== 0)) {
        const [dx, dy] = preview.move;
        edit(t("cad.moveRegion"), (dr) => moveRegion(dr, d.id, dx, dy));
      }
      setPreview({});
      return;
    }
    setPreview({});
    // クリック
    const tool = vs.tool;
    if (tool === "select") {
      const tol = EDGE_TOLERANCE_PX / camera.scale;
      // 細い配置物を先に (領域の上に描いてある)
      const hitP = [...placements].reverse().find((pl) => (pl.kind === "eedf" ? hitRect(raw, pl.p1, pl.p2, tol) : hitSegment(raw, pl.p1, pl.p2, tol)));
      const sel = useSelection.getState();
      if (hitP) {
        sel.selectPlacement({ kind: hitP.kind, index: hitP.index });
        sel.select(PLACEMENT_NODES[hitP.kind]);
        return;
      }
      const hit = findRegionAt(raw, project.geometry.regions, tol);
      sel.selectPlacement(null);
      sel.selectRegion(hit ? hit.id : null);
    } else if (tool === "polyline") {
      setPts((prev) => [...prev, pt]);
    } else if (tool === "probe") {
      if (scene.field) vs.setProbe(pt);
    } else if (TWO_POINT.includes(tool)) {
      if (pts.length === 0) {
        setPts([pt]);
        if (tool === "measure") setMeasure(null);
      } else {
        const p1 = pts[0];
        setPts([]);
        commitTwoPoint(tool, p1, pt);
      }
    }
  };

  const onDoubleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!camera) return;
    const [sx, sy] = localPoint(e, e.currentTarget);
    if (vs.tool === "select") {
      const sel = project.geometry.regions.find((r) => r.id === selectedRegion);
      if (sel?.polygon) {
        const v = findVertexHandle(sel.polygon, camera, sx, sy);
        if (v !== null && sel.polygon.length > 3) edit(t("cad.deleteVertex"), (d) => void removeRegionVertex(d, sel.id, v));
      }
      return;
    }
    if (vs.tool === "polyline") finishPolyline(pts);
  };

  const onKeyDown = (e: ReactKeyboardEvent<HTMLCanvasElement>) => {
    const tool = vs.tool;
    if (e.key === "Escape") {
      e.preventDefault();
      const busy = pts.length > 0 || dragRef.current !== null || preview.polygon || preview.move || preview.radius != null;
      setPts([]);
      setPreview({});
      dragRef.current = null;
      setMeasure(null);
      vs.setProbe(null);
      if (!busy && tool !== "select") vs.setTool("select");
    } else if (e.key === "Enter" && tool === "polyline") {
      e.preventDefault();
      finishPolyline(pts);
    } else if (e.key === "Backspace" && tool === "polyline" && pts.length > 0) {
      e.preventDefault();
      setPts((p) => p.slice(0, -1));
    } else if ((e.key === "Delete" || e.key === "Backspace") && tool === "select") {
      e.preventDefault();
      deleteSelected();
    } else if (tool === "select" && selectedRegion && camera && e.key.startsWith("Arrow")) {
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

  // ---- 表示 ----

  const hint = toolHint(vs.tool, pts.length, t);
  const unit = lengthUnitLabel(lengthUnit);
  const hoverValue = hover && scene.field ? sampleField(scene.field, hover[0], hover[1]) : null;
  const cursorText = cursor
    ? `${axes[0]}: ${toDisplayLength(cursor[0], lengthUnit).toFixed(3)} ${unit}  ${axes[1]}: ${toDisplayLength(cursor[1], lengthUnit).toFixed(3)} ${unit}`
    : "";
  const cls = `viewer-top${panning ? " panning" : vs.tool === "select" ? "" : " crosshair"}`;

  return (
    <div className="viewer" data-scheme={theme}>
      <ViewerToolbar scene={scene} controls={controls} run={run} onExportPng={exportPng} onExportCsv={exportCsv} />
      <div ref={wrapRef} className="viewer-canvas">
        <canvas ref={baseRef} className="viewer-layer" aria-hidden="true" />
        <canvas ref={glRef} className="viewer-layer" aria-hidden="true" />
        <canvas
          ref={topRef}
          className={`viewer-layer ${cls}`}
          tabIndex={0}
          role="application"
          aria-label={t("viewer.canvasLabel")}
          onPointerDown={onPointerDown}
          onMouseDown={(e) => {
            // 中ボタンの自動スクロールを出さない
            if (e.button === 1) e.preventDefault();
          }}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={() => {
            dragRef.current = null;
            setPanning(false);
            setPreview({});
          }}
          onPointerLeave={() => {
            setCursor(null);
            setHover(null);
          }}
          onDoubleClick={onDoubleClick}
          onKeyDown={onKeyDown}
          onContextMenu={(e) => e.preventDefault()}
          data-redraw={redraw}
        />
        {(scene.title || scene.stale) && (
          <div className="viewer-title" style={{ left: rulerPx + 8, top: rulerPx + 6 }}>
            {scene.title && <span>{scene.title}</span>}
            {scene.stale && <span className="badge badge-warn">{t("viewer.stale")}</span>}
          </div>
        )}
        {!glOk && <div className="viewer-nogl">{t("viewer.noWebgl")}</div>}
        {vs.overlays.legend && (
          <div className="viewer-legend" aria-label={t("viewer.legend")}>
            <span>
              <i className="swatch dirichlet" />
              Dirichlet
            </span>
            <span>
              <i className="swatch symmetry" />
              {t("bc.symmetry")}
            </span>
            <span>
              <i className="swatch periodic" />
              {t("bc.periodic")}
            </span>
            {axisEdgeLegend(project, t)}
          </div>
        )}
      </div>
      {playback && <PlaybackBar playback={playback} />}
      {run && run.kind === "pic" && controls.mode === "live" && <ViewerRfStrip run={run} />}
      <div className="viewer-status">
        <span className="viewer-cursor mono">{cursorText}</span>
        {hoverValue !== null && scene.field && (
          <span className="viewer-hover mono">
            {scene.field.label} = {formatNumber(hoverValue)} {scene.field.unit}
          </span>
        )}
        <span className="spacer" />
        <span className="viewer-hint">{hint}</span>
      </div>
    </div>
  );
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

function axisEdgeLegend(p: Project, t: TFunction) {
  if (coordOf(p) === "xy") return null;
  return (
    <span>
      <i className="swatch axis" />
      {t("viewer.axis")}
    </span>
  );
}

export function toolHint(tool: Tool, n: number, t: TFunction): string {
  switch (tool) {
    case "select":
      return t("viewer.hint.select");
    case "polyline":
      return n === 0 ? t("viewer.hint.polylineStart") : t("viewer.hint.polylineNext");
    case "rect":
      return n === 0 ? t("viewer.hint.rectStart") : t("viewer.hint.rectEnd");
    case "circle":
      return n === 0 ? t("viewer.hint.circleStart") : t("viewer.hint.circleEnd");
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
