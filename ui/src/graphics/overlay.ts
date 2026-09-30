// Canvas2D で描くもの。下の層: 背景・ドメインの塗り・グリッド。上の層: 電場の矢印・ジオメトリ (境界条件の
// 色分け・対称軸・領域・スケッチ)・配置物・選択とハンドル・作図中の線・計測・プローブ・カラーバー・ルーラー。
// 場とメッシュは間の WebGL の層が描く。色はテーマの CSS 変数から読む。

import { bulgeOf, bulgeThrough, midpoint, pathBounds, segFromBulge, type ArcSeg, type Seg } from "../cad/geom";
import { bulgesOf, pathHandles, type Affine, type PathData } from "../cad/path";
import type { SnapResult } from "../cad/snap";
import { transformRegionShape, transformSketch } from "../model/editOps";
import { axisEdges, boundaryOfEdge, domainPath, regionPath, type Point, type Project, type Region } from "../model/project";
import type { PickRef } from "../model/selection";
import { edgeCountOf, sketchSegs, type EditPath, type SketchEntity } from "../model/sketch";
import type { EditObject } from "./editTargets";
import { lengthUnitLabel, toDisplayLength, type LengthUnit } from "../util/format";
import { gridStep, toScreen, toWorld, type Camera } from "./camera";
import { colormapCss, type ColormapKey } from "./colormaps";
import { colorbarTicks, formatColorbarValue, type DisplayRange } from "./fieldScale";
import { radiusHandlePoint } from "./hitTest";
import type { DrawTarget, OverlayKey, Tool } from "./viewerStore";

// ---- 色 ----

const COLOR_VARS = {
  bg: "--canvas-bg",
  domain: "--canvas-domain",
  edge: "--canvas-edge",
  grid: "--canvas-grid",
  wire: "--canvas-wire",
  wireField: "--canvas-wire-field",
  isoline: "--canvas-isoline",
  rulerBg: "--canvas-ruler-bg",
  rulerBorder: "--canvas-ruler-border",
  rulerTick: "--canvas-ruler-tick",
  rulerText: "--canvas-ruler-text",
  text: "--text",
  conductor: "--color-conductor",
  dielectric: "--color-dielectric",
  charge: "--color-charge",
  dirichlet: "--color-dirichlet",
  symmetry: "--color-symmetry",
  periodic: "--color-periodic",
  axis: "--color-axis",
  selection: "--color-selection",
  handle: "--color-handle",
  handleMid: "--color-handle-mid",
  draw: "--color-draw",
  sketch: "--color-sketch",
  snap: "--color-snap",
  fillPreview: "--color-fill-preview",
  emitter: "--color-emitter",
  injector: "--color-injector",
  gasbc: "--color-gasbc",
  eedf: "--color-eedf",
  edgeSize: "--color-edgesize",
  sheath: "--color-sheath",
  amr: "--color-amr",
  profile: "--color-profile",
  measure: "--color-measure",
  probe: "--color-probe",
  panel: "--canvas-panel",
  panelBorder: "--canvas-panel-border",
} as const;

export type OverlayColors = Record<keyof typeof COLOR_VARS, string> & { collectors: string[] };

let cached: { theme: string; colors: OverlayColors } | null = null;

/** テーマの色 (テーマが変わるまで覚えておく) */
export function readColors(): OverlayColors {
  const theme = document.documentElement.dataset.theme ?? "";
  if (cached && cached.theme === theme) return cached.colors;
  const cs = getComputedStyle(document.documentElement);
  const get = (v: string, fb = "#888") => cs.getPropertyValue(v).trim() || fb;
  const out = Object.fromEntries(Object.entries(COLOR_VARS).map(([k, v]) => [k, get(v)])) as unknown as OverlayColors;
  out.collectors = Array.from({ length: 8 }, (_, i) => get(`--collector-${i + 1}`, "#ffd400"));
  cached = { theme, colors: out };
  return out;
}

const FONT = "11px system-ui, -apple-system, 'Segoe UI', sans-serif";

// ---- 共通 ----

export interface View {
  camera: Camera;
  width: number;
  height: number;
}

function pathPolygon(ctx: CanvasRenderingContext2D, v: View, poly: Point[]): void {
  ctx.beginPath();
  poly.forEach((q, i) => {
    const [x, y] = toScreen(v.camera, q);
    if (i) ctx.lineTo(x, y);
    else ctx.moveTo(x, y);
  });
  ctx.closePath();
}

/**
 * 頂点 a から b への辺を今の道に足す (円弧は ctx.arc。画面は y が下向きなので角度の符号を反転し、ワールドの
 * 反時計回りは画面の反時計回り (角度の減る向き) になる)。現在の点は a にあるとする
 */
function edgeTo(ctx: CanvasRenderingContext2D, v: View, a: Point, b: Point, bulge: number): void {
  if (!bulge) {
    const [x, y] = toScreen(v.camera, b);
    ctx.lineTo(x, y);
    return;
  }
  const s = segFromBulge(a, b, bulge) as ArcSeg;
  const [cx, cy] = toScreen(v.camera, s.center);
  ctx.arc(cx, cy, s.r * v.camera.scale, -s.a0, -s.a1, s.ccw);
}

/** 閉じた経路 (頂点 + 円弧) */
function tracePath(ctx: CanvasRenderingContext2D, v: View, path: PathData, offset: Point = [0, 0]): void {
  const poly = path.polygon.map(([x, y]) => [x + offset[0], y + offset[1]] as Point);
  const b = bulgesOf(path);
  ctx.beginPath();
  if (poly.length === 0) return;
  const [x0, y0] = toScreen(v.camera, poly[0]);
  ctx.moveTo(x0, y0);
  for (let i = 0; i < poly.length; i++) edgeTo(ctx, v, poly[i], poly[(i + 1) % poly.length], b[i]);
  ctx.closePath();
}

/** 外周の辺 i だけの道 */
function traceDomainEdge(ctx: CanvasRenderingContext2D, v: View, p: Project, i: number): void {
  const poly = p.geometry.domain.polygon;
  const a = poly[i];
  const [x, y] = toScreen(v.camera, a);
  ctx.beginPath();
  ctx.moveTo(x, y);
  edgeTo(ctx, v, a, poly[(i + 1) % poly.length], bulgesOf(domainPath(p))[i]);
}

/** 点の操作の形 (閉じていなければ最後の点で終わる) */
function traceEditPath(ctx: CanvasRenderingContext2D, v: View, p: EditPath, offset: Point = [0, 0]): void {
  const pts = p.points.map(([x, y]) => [x + offset[0], y + offset[1]] as Point);
  ctx.beginPath();
  if (pts.length === 0) return;
  const [x0, y0] = toScreen(v.camera, pts[0]);
  ctx.moveTo(x0, y0);
  for (let i = 0; i < edgeCountOf(p); i++) edgeTo(ctx, v, pts[i], pts[(i + 1) % pts.length], p.bulges[i] ?? 0);
  if (p.closed) ctx.closePath();
}

/** スケッチの形 (円・線分・円弧・ポリライン) */
function traceSketch(ctx: CanvasRenderingContext2D, v: View, e: SketchEntity, offset: Point = [0, 0]): void {
  ctx.beginPath();
  if (e.kind === "circle") {
    const [cx, cy] = toScreen(v.camera, [e.center[0] + offset[0], e.center[1] + offset[1]]);
    ctx.arc(cx, cy, e.r * v.camera.scale, 0, 2 * Math.PI);
    return;
  }
  for (const sg of sketchSegs(e)) {
    const a: Point = [sg.a[0] + offset[0], sg.a[1] + offset[1]];
    const b: Point = [sg.b[0] + offset[0], sg.b[1] + offset[1]];
    const [x, y] = toScreen(v.camera, a);
    ctx.moveTo(x, y);
    edgeTo(ctx, v, a, b, bulgeOf(sg));
  }
}

function pathRegion(ctx: CanvasRenderingContext2D, v: View, r: Region, offset: Point = [0, 0], radius?: number): void {
  if (r.shape) {
    const [cx, cy] = toScreen(v.camera, [r.shape.center[0] + offset[0], r.shape.center[1] + offset[1]]);
    ctx.beginPath();
    ctx.arc(cx, cy, (radius ?? r.shape.radius) * v.camera.scale, 0, 2 * Math.PI);
  } else {
    tracePath(ctx, v, regionPath(r), offset);
  }
}



function segment(ctx: CanvasRenderingContext2D, v: View, a: Point, b: Point): void {
  const [x0, y0] = toScreen(v.camera, a);
  const [x1, y1] = toScreen(v.camera, b);
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
}

function dot(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, fill: string, stroke?: string): void {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

function label(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string, align: CanvasTextAlign = "center", base: CanvasTextBaseline = "bottom"): void {
  ctx.font = FONT;
  ctx.textAlign = align;
  ctx.textBaseline = base;
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(0,0,0,0.45)";
  ctx.lineJoin = "round";
  ctx.strokeText(text, x, y);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function arrowHead(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, head: number): void {
  const ang = Math.atan2(y1 - y0, x1 - x0);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - head * Math.cos(ang - Math.PI / 6), y1 - head * Math.sin(ang - Math.PI / 6));
  ctx.lineTo(x1 - head * Math.cos(ang + Math.PI / 6), y1 - head * Math.sin(ang + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
}

/** ルーラーの帯の幅 [px] (文字の大きさに合わせる、v1 と同じ) */
export function rulerSize(fontPx: number): number {
  return Math.max(24, fontPx * 2.2);
}

/** 目盛りの数値 (間隔に合わせた桁数) */
export function tickLabel(valueM: number, stepM: number, unit: LengthUnit): string {
  const v = toDisplayLength(valueM, unit);
  const step = toDisplayLength(stepM, unit);
  const decimals = Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9)));
  const s = v.toFixed(decimals);
  return s === "-0" || /^-0\.0*$/.test(s) ? s.slice(1) : s;
}

// ---- 下の層 ----

export function drawBase(ctx: CanvasRenderingContext2D, v: View, project: Project, colors: OverlayColors, showGrid: boolean): void {
  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, v.width, v.height);
  tracePath(ctx, v, domainPath(project));
  ctx.fillStyle = colors.domain;
  ctx.fill();
  if (!showGrid) return;
  const step = gridStep(v.camera);
  const [x0w, y1w] = toWorld(v.camera, 0, 0);
  const [x1w, y0w] = toWorld(v.camera, v.width, v.height);
  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let k = Math.floor(x0w / step); k * step <= x1w; k++) {
    const x = Math.round(toScreen(v.camera, [k * step, 0])[0]) + 0.5;
    ctx.moveTo(x, 0);
    ctx.lineTo(x, v.height);
  }
  for (let k = Math.floor(y0w / step); k * step <= y1w; k++) {
    const y = Math.round(toScreen(v.camera, [0, k * step])[1]) + 0.5;
    ctx.moveTo(0, y);
    ctx.lineTo(v.width, y);
  }
  ctx.stroke();
}

// ---- 上の層 ----

export interface Arrow {
  /** 画面の点 [px] と向き (単位ベクトル、画面の y は下向き) */
  x: number;
  y: number;
  dx: number;
  dy: number;
}

export interface Placement {
  kind: "collector" | "gasbc" | "eedf" | "edgeSize" | "sheath";
  label: string;
  p1: Point;
  p2: Point;
  index: number;
  /** シース端の位置 (p1 からの距離 [m]、P6e) */
  s?: number | null;
}

export interface EmitterView {
  kind: "line" | "point";
  p1: Point;
  p2: Point;
  directionDeg: number;
}

export interface Drawing {
  tool: Tool;
  /** 描く先 (折れ線・矩形・円) */
  target: DrawTarget;
  pts: Point[];
  /** 折れ線の辺ごとの bulge (pts[i] → pts[i+1]) */
  bulges: number[];
  /** 円弧を描く途中の通る点 (折れ線の円弧・3 点の円弧) */
  through: Point | null;
  /** 折れ線の円弧の段 */
  arcMode: boolean;
  cursor: Point | null;
  /** 折れ線の最初の点に近い (クリックで閉じる) */
  closing: boolean;
  /** 選んでいる形の数 (変換の道具の説明) */
  selected: number;
  /** 編集の道具で途中まで選んだ (オフセットの元・フィレットの 1 本目) */
  pending: boolean;
}

/** 編集の道具の見せる形: 足す形 (add) か消える部分 (remove)、途中まで選んだ形、角の印 */
export interface ToolPreview {
  segs: Seg[];
  tone: "add" | "remove";
  pending: Seg[];
  points?: Point[];
}

export interface Preview {
  /** 頂点・辺のドラッグ中の形 (領域・ドメイン・スケッチ) */
  edit?: EditPath | null;
  /** 半径のドラッグ中の半径 */
  radius?: number | null;
  /** 移動のドラッグ中のずれ */
  move?: Point | null;
  /** 範囲選択の 2 隅 (ワールド座標) */
  box?: [Point, Point] | null;
  /** 変換の道具で、選んだものをカーソルまで変換した形 */
  affine?: Affine | null;
}

export interface Colorbar {
  colormap: ColormapKey;
  range: DisplayRange;
  label: string;
  unit: string;
}

export interface ProbeBox {
  pt: Point;
  lines: string[];
}

export interface OverlayState {
  view: View;
  project: Project;
  colors: OverlayColors;
  lengthUnit: LengthUnit;
  axisNames: [string, string];
  overlays: Record<OverlayKey, boolean>;
  /** 場かメッシュを描いているか (領域を薄く塗るかどうか) */
  fieldShown: boolean;
  selectedPlacement: { kind: Placement["kind"]; index: number } | null;
  /** 選んだ外周の辺の番号 */
  selectedEdge: number | null;
  /** キャンバスで選んだ領域とスケッチ */
  picked: PickRef[];
  /** ハンドルを出して編集している形 */
  edit: EditObject | null;
  sketch: SketchEntity[];
  /** 囲まれた所から領域を作る道具の、カーソルの下の面 */
  fillPreview: PathData | null;
  /** 編集の道具の見せる形 */
  toolPreview: ToolPreview | null;
  /** いま合っているオブジェクトスナップと、その名前 */
  snapMark: SnapResult | null;
  snapLabel: string | null;
  tool: Tool;
  preview: Preview;
  drawing: Drawing;
  placements: Placement[];
  emitter: EmitterView | null;
  /** PIC の注入のエミッタ */
  injector: EmitterView | null;
  amrBoxes: [Point, Point][];
  profile: [Point, Point] | null;
  measure: [Point, Point] | null;
  arrows: Arrow[];
  colorbar: Colorbar | null;
  probe: ProbeBox | null;
  cursor: Point | null;
  rulerFont: number;
}

function drawArrows(ctx: CanvasRenderingContext2D, arrows: Arrow[]): void {
  const len = 14;
  for (const pass of [0, 1]) {
    ctx.strokeStyle = pass === 0 ? "rgba(0,0,0,0.55)" : "rgba(255,255,255,0.92)";
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = pass === 0 ? 3 : 1.2;
    for (const a of arrows) {
      const x0 = a.x - (a.dx * len) / 2;
      const y0 = a.y - (a.dy * len) / 2;
      const x1 = a.x + (a.dx * len) / 2;
      const y1 = a.y + (a.dy * len) / 2;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
      arrowHead(ctx, x0, y0, x1, y1, pass === 0 ? 5.5 : 4.5);
    }
  }
}

function drawGeometry(ctx: CanvasRenderingContext2D, s: OverlayState): void {
  const { view: v, project: p, colors: c } = s;
  const poly = p.geometry.domain.polygon;
  // ドメインの輪郭と境界条件の辺
  tracePath(ctx, v, domainPath(p));
  ctx.strokeStyle = c.edge;
  ctx.lineWidth = 1;
  ctx.stroke();
  const axis = new Set(axisEdges(p));
  for (let i = 0; i < poly.length; i++) {
    const bc = boundaryOfEdge(p, i);
    let color: string | null = null;
    let width = 1;
    let dash: number[] = [];
    if (axis.has(i)) {
      color = c.axis;
      width = 2;
      dash = [10, 4, 2, 4];
    } else if (bc?.type === "dirichlet") {
      color = c.dirichlet;
      width = 3.5;
    } else if (bc?.type === "symmetry") {
      color = c.symmetry;
      width = 2.5;
      dash = [8, 5];
    } else if (bc?.type === "periodic") {
      color = c.periodic;
      width = 2.5;
      dash = [8, 5];
    }
    if (!color) continue;
    traceDomainEdge(ctx, v, p, i);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  // 選んだ外周の辺 (境界条件の色の上に、選択の色の縁取り)
  if (s.selectedEdge !== null && s.selectedEdge < poly.length) {
    traceDomainEdge(ctx, v, p, s.selectedEdge);
    ctx.strokeStyle = c.selection;
    ctx.lineWidth = 6;
    ctx.globalAlpha = 0.55;
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
    ctx.setLineDash([]);
    for (const q of [poly[s.selectedEdge], poly[(s.selectedEdge + 1) % poly.length]]) {
      const [x, y] = toScreen(v.camera, q);
      dot(ctx, x, y, 3.5, c.selection, "#1b1e24");
    }
  }
  // 領域
  for (const r of p.geometry.regions) {
    const color = c[r.type] ?? c.edge;
    pathRegion(ctx, v, r);
    if (!s.fieldShown) {
      ctx.save();
      ctx.globalAlpha = 0.28;
      ctx.fillStyle = color;
      ctx.fill();
      ctx.restore();
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  // スケッチ (領域でない線・円弧・円・ポリライン)
  const pickedSketch = new Set(s.picked.filter((x) => x.kind === "sketch").map((x) => x.id));
  for (const e of s.sketch) {
    if (pickedSketch.has(e.id)) continue;
    traceSketch(ctx, v, e);
    ctx.strokeStyle = c.sketch;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  // 領域の名前 (小さすぎる領域には出さない)。位置は頂点と辺の中点 (円弧は弧の中点) の平均
  for (const r of p.geometry.regions) {
    let pts: Point[];
    let size: number;
    if (r.shape) {
      pts = [r.shape.center];
      size = 2 * r.shape.radius * v.camera.scale;
    } else {
      const path = regionPath(r);
      if (path.polygon.length === 0) continue;
      const h = pathHandles(path);
      pts = [...h.vertices, ...h.midpoints];
      const b = pathBounds(path.polygon, path.bulges);
      size = Math.min(b.x1 - b.x0, b.y1 - b.y0) * v.camera.scale;
    }
    const cx = pts.reduce((a, q) => a + q[0], 0) / pts.length;
    const cy = pts.reduce((a, q) => a + q[1], 0) / pts.length;
    const [x, y] = toScreen(v.camera, [cx, cy]);
    if (size < 28) continue;
    label(ctx, r.id, x, y, c.text, "center", "middle");
  }
}

function placementColor(s: OverlayState, pl: Placement): string {
  const c = s.colors;
  switch (pl.kind) {
    case "collector":
      return c.collectors[pl.index % c.collectors.length];
    case "gasbc":
      return c.gasbc;
    case "eedf":
      return c.eedf;
    case "edgeSize":
      return c.edgeSize;
    case "sheath":
      return c.sheath;
  }
}

function drawPlacement(ctx: CanvasRenderingContext2D, s: OverlayState, pl: Placement): void {
  const v = s.view;
  const color = placementColor(s, pl);
  const selected = s.selectedPlacement?.kind === pl.kind && s.selectedPlacement.index === pl.index;
  if (pl.kind === "eedf") {
    const [xa, ya] = toScreen(v.camera, [Math.min(pl.p1[0], pl.p2[0]), Math.max(pl.p1[1], pl.p2[1])]);
    const [xb, yb] = toScreen(v.camera, [Math.max(pl.p1[0], pl.p2[0]), Math.min(pl.p1[1], pl.p2[1])]);
    ctx.strokeStyle = color;
    ctx.lineWidth = selected ? 3 : 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(xa, ya, xb - xa, yb - ya);
    if (selected) {
      ctx.strokeStyle = "rgba(255,255,255,0.85)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.strokeRect(xa, ya, xb - xa, yb - ya);
    }
    ctx.setLineDash([]);
    label(ctx, pl.label, xa + 3, ya - 3, color, "left", "bottom");
    return;
  }
  const dashed = pl.kind === "edgeSize" || pl.kind === "sheath";
  segment(ctx, v, pl.p1, pl.p2);
  ctx.strokeStyle = color;
  ctx.lineWidth = selected ? 5 : pl.kind === "collector" || pl.kind === "gasbc" ? 3.5 : 2.5;
  ctx.setLineDash(dashed ? [6, 4] : []);
  ctx.stroke();
  ctx.setLineDash([]);
  if (selected) {
    segment(ctx, v, pl.p1, pl.p2);
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  const r = pl.kind === "collector" || pl.kind === "gasbc" ? 5 : 4;
  for (const q of [pl.p1, pl.p2]) {
    const [x, y] = toScreen(v.camera, q);
    dot(ctx, x, y, r, color, "#1b1e24");
  }
  const [mx, my] = toScreen(v.camera, [(pl.p1[0] + pl.p2[0]) / 2, (pl.p1[1] + pl.p2[1]) / 2]);
  label(ctx, pl.label, mx, my - 6, color);
  // シース端の位置 (P6e で結果から)
  if (pl.kind === "sheath" && pl.s != null) {
    const len = Math.hypot(pl.p2[0] - pl.p1[0], pl.p2[1] - pl.p1[1]);
    if (len > 0) {
      const t = pl.s / len;
      const [x, y] = toScreen(v.camera, [pl.p1[0] + (pl.p2[0] - pl.p1[0]) * t, pl.p1[1] + (pl.p2[1] - pl.p1[1]) * t]);
      ctx.lineWidth = 2;
      dot(ctx, x, y, 5, "#ffffff", color);
    }
  }
}

function drawEmitter(ctx: CanvasRenderingContext2D, s: OverlayState, e: EmitterView, color = s.colors.emitter, tag?: string): void {
  const v = s.view;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  let origin: Point;
  if (e.kind === "line") {
    segment(ctx, v, e.p1, e.p2);
    ctx.stroke();
    origin = [(e.p1[0] + e.p2[0]) / 2, (e.p1[1] + e.p2[1]) / 2];
  } else {
    const [x, y] = toScreen(v.camera, e.p1);
    ctx.beginPath();
    ctx.moveTo(x - 6, y - 6);
    ctx.lineTo(x + 6, y + 6);
    ctx.moveTo(x + 6, y - 6);
    ctx.lineTo(x - 6, y + 6);
    ctx.stroke();
    origin = e.p1;
  }
  // 射出の向き (画面で一定の長さ、画面の y は下向き)
  const a = (e.directionDeg * Math.PI) / 180;
  const [ox, oy] = toScreen(v.camera, origin);
  const ex = ox + Math.cos(a) * 24;
  const ey = oy - Math.sin(a) * 24;
  ctx.beginPath();
  ctx.moveTo(ox, oy);
  ctx.lineTo(ex, ey);
  ctx.stroke();
  arrowHead(ctx, ox, oy, ex, ey, 7);
  if (tag) label(ctx, tag, ox + 8, oy + 10, color, "left", "top");
}

/** 選んだもの (領域は破線の縁取り、スケッチは選択の色) と、編集している形のハンドル・ドラッグ中の形 */
function drawSelection(ctx: CanvasRenderingContext2D, s: OverlayState): void {
  const { view: v, colors: c, project: p } = s;
  const edit = s.edit;
  const editing = (kind: string, id?: string) => edit !== null && edit.target.kind === kind && (id === undefined || ("id" in edit.target && edit.target.id === id));
  for (const it of s.picked) {
    ctx.strokeStyle = c.selection;
    if (it.kind === "region") {
      const r = p.geometry.regions.find((x) => x.id === it.id);
      if (!r) continue;
      ctx.lineWidth = 3;
      ctx.setLineDash([6, 4]);
      if (editing("region", r.id) && s.preview.edit) traceEditPath(ctx, v, s.preview.edit);
      else pathRegion(ctx, v, r, [0, 0], editing("region", r.id) ? (s.preview.radius ?? undefined) : undefined);
      ctx.stroke();
      ctx.setLineDash([]);
    } else {
      const e = s.sketch.find((x) => x.id === it.id);
      if (!e) continue;
      ctx.lineWidth = 2.5;
      if (editing("sketch", e.id) && s.preview.edit) traceEditPath(ctx, v, s.preview.edit);
      else if (editing("sketch", e.id) && e.kind === "circle" && s.preview.radius) traceSketch(ctx, v, { ...e, r: s.preview.radius });
      else traceSketch(ctx, v, e);
      ctx.stroke();
    }
  }
  // ドメインの編集中の形 (ドメインのページを開いているだけで目立たないよう、選択の道具のときだけ)
  if (edit?.target.kind === "domain" && (s.tool === "select" || s.preview.edit)) {
    ctx.strokeStyle = c.selection;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    traceEditPath(ctx, v, s.preview.edit ?? edit.path!);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  // 移動のプレビュー (選んだもの全て)
  const d = s.preview.move;
  if (d) {
    for (const it of s.picked) {
      if (it.kind === "region") {
        const r = p.geometry.regions.find((x) => x.id === it.id);
        if (!r) continue;
        pathRegion(ctx, v, r, d);
        ctx.save();
        ctx.globalAlpha = 0.15;
        ctx.fillStyle = c.selection;
        ctx.fill();
        ctx.restore();
      } else {
        const e = s.sketch.find((x) => x.id === it.id);
        if (!e) continue;
        traceSketch(ctx, v, e, d);
      }
      ctx.strokeStyle = c.selection;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  // 変換の道具: 選んだものを変換した形
  const m = s.preview.affine;
  if (m) {
    for (const it of s.picked) {
      if (it.kind === "region") {
        const r = p.geometry.regions.find((x) => x.id === it.id);
        if (!r) continue;
        pathRegion(ctx, v, { ...r, ...transformRegionShape(r, m) });
        ctx.save();
        ctx.globalAlpha = 0.15;
        ctx.fillStyle = c.selection;
        ctx.fill();
        ctx.restore();
      } else {
        const e = s.sketch.find((x) => x.id === it.id);
        if (!e) continue;
        traceSketch(ctx, v, transformSketch(e, m));
      }
      ctx.strokeStyle = c.selection;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  // ハンドル (選択の道具で、1 つを編集しているとき)
  if (!edit || s.tool !== "select" || d) return;
  if (edit.circle) {
    const rad = s.preview.radius ?? edit.circle.r;
    const [hx, hy] = toScreen(v.camera, radiusHandlePoint({ kind: "circle", center: edit.circle.center, radius: rad }));
    squareHandle(ctx, hx, hy, c.selection);
    return;
  }
  const path = s.preview.edit ?? edit.path;
  if (!path) return;
  for (const q of path.points) {
    const [x, y] = toScreen(v.camera, q);
    squareHandle(ctx, x, y, c.selection);
  }
  for (let i = 0; i < edgeCountOf(path); i++) {
    const [x, y] = toScreen(v.camera, midpoint(segFromBulge(path.points[i], path.points[(i + 1) % path.points.length], path.bulges[i] ?? 0)));
    dot(ctx, x, y, 4, c.handleMid);
  }
}

function squareHandle(ctx: CanvasRenderingContext2D, x: number, y: number, fill: string): void {
  const h = 7;
  ctx.fillStyle = fill;
  ctx.strokeStyle = "#1b1e24";
  ctx.lineWidth = 1;
  ctx.fillRect(x - h / 2, y - h / 2, h, h);
  ctx.strokeRect(x - h / 2, y - h / 2, h, h);
}

/** 作図中の線 (ツールの色の破線)。折れ線は円弧の辺も、3 点の円弧は通る点から、スケッチに描くときはスケッチの色 */
function drawRubberBand(ctx: CanvasRenderingContext2D, s: OverlayState): void {
  const { view: v, colors: c } = s;
  const { tool, pts, cursor, bulges, through, target } = s.drawing;
  if (pts.length === 0 || tool === "measure") return;
  const sketchTool = tool === "line" || tool === "arc" || ((tool === "polyline" || tool === "rect" || tool === "circle") && target === "sketch");
  const color =
    tool === "emitter"
      ? c.emitter
      : tool === "injector"
        ? c.injector
        : tool === "collector"
          ? c.collectors[0]
          : tool === "gasbc"
            ? c.gasbc
            : tool === "eedfbox"
              ? c.eedf
              : tool === "meshref"
                ? c.edgeSize
                : tool === "sheathline"
                  ? c.sheath
                  : sketchTool
                    ? c.sketch
                    : c.draw;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1.5;
  if (tool === "polyline" || tool === "arc" || tool === "line") {
    // 確定した辺 (実線) と、カーソルまでの辺 (破線。通る点があれば円弧)
    ctx.beginPath();
    const [x0, y0] = toScreen(v.camera, pts[0]);
    ctx.moveTo(x0, y0);
    for (let i = 0; i + 1 < pts.length; i++) edgeTo(ctx, v, pts[i], pts[i + 1], bulges[i] ?? 0);
    ctx.stroke();
    const last = pts[pts.length - 1];
    if (cursor) {
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      const [lx, ly] = toScreen(v.camera, last);
      ctx.moveTo(lx, ly);
      edgeTo(ctx, v, last, cursor, through ? bulgeThrough(last, through, cursor) : 0);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    for (const q of pts) {
      const [x, y] = toScreen(v.camera, q);
      dot(ctx, x, y, 3, color);
    }
    if (through) {
      const [x, y] = toScreen(v.camera, through);
      dot(ctx, x, y, 3.5, "transparent", color);
    }
    // 最初の点に戻ると閉じる
    if (s.drawing.closing) {
      const [x, y] = toScreen(v.camera, pts[0]);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, 2 * Math.PI);
      ctx.stroke();
    }
    return;
  }
  ctx.setLineDash([4, 3]);
  if (!cursor) {
    ctx.setLineDash([]);
    return;
  }
  const [x0, y0] = toScreen(v.camera, pts[0]);
  const [x1, y1] = toScreen(v.camera, cursor);
  if (tool === "rect" || tool === "eedfbox") {
    ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
  } else if (tool === "circle") {
    ctx.beginPath();
    ctx.arc(x0, y0, Math.hypot(x1 - x0, y1 - y0), 0, 2 * Math.PI);
    ctx.stroke();
  } else {
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  dot(ctx, x0, y0, 3, color);
}

/** 囲まれた所から領域を作る道具: カーソルの下の面 */
function drawFillPreview(ctx: CanvasRenderingContext2D, s: OverlayState, face: PathData): void {
  tracePath(ctx, s.view, face);
  ctx.save();
  ctx.globalAlpha = 0.22;
  ctx.fillStyle = s.colors.fillPreview;
  ctx.fill();
  ctx.restore();
  ctx.strokeStyle = s.colors.fillPreview;
  ctx.lineWidth = 2;
  ctx.stroke();
}

/** オブジェクトスナップの印 (端点は四角・中点は三角・中心は円・交点は ×・垂線は直角・接線は円と線) と名前 */
function drawSnapMark(ctx: CanvasRenderingContext2D, s: OverlayState, m: SnapResult, text: string | null): void {
  const [x, y] = toScreen(s.view.camera, m.point as Point);
  const r = 6;
  ctx.save();
  ctx.strokeStyle = s.colors.snap;
  ctx.lineWidth = 2;
  ctx.beginPath();
  switch (m.kind) {
    case "endpoint":
      ctx.rect(x - r, y - r, 2 * r, 2 * r);
      break;
    case "midpoint":
      ctx.moveTo(x, y - r);
      ctx.lineTo(x + r, y + r * 0.8);
      ctx.lineTo(x - r, y + r * 0.8);
      ctx.closePath();
      break;
    case "center":
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      break;
    case "intersection":
      ctx.moveTo(x - r, y - r);
      ctx.lineTo(x + r, y + r);
      ctx.moveTo(x + r, y - r);
      ctx.lineTo(x - r, y + r);
      break;
    case "perpendicular":
      ctx.moveTo(x - r, y - r);
      ctx.lineTo(x - r, y + r);
      ctx.lineTo(x + r, y + r);
      ctx.moveTo(x - r, y);
      ctx.lineTo(x, y);
      ctx.lineTo(x, y + r);
      break;
    case "tangent":
      ctx.arc(x, y + 2, r - 1, 0, 2 * Math.PI);
      ctx.moveTo(x - r - 2, y - r + 1);
      ctx.lineTo(x + r + 2, y - r + 1);
      break;
    default:
      break;
  }
  ctx.stroke();
  ctx.restore();
  if (text) label(ctx, text, x + 10, y - 9, s.colors.snap, "left", "bottom");
}

/** 曲線の並びをたどる (つながっていなくてよい) */
function traceSegs(ctx: CanvasRenderingContext2D, v: View, segs: Seg[]): void {
  ctx.beginPath();
  for (const sg of segs) {
    const [x, y] = toScreen(v.camera, sg.a as Point);
    ctx.moveTo(x, y);
    edgeTo(ctx, v, sg.a as Point, sg.b as Point, bulgeOf(sg));
  }
}

/** 編集の道具: 途中まで選んだ形 (選択の色)、足す形 (破線) か消える部分 (赤)、角の印 */
function drawToolPreview(ctx: CanvasRenderingContext2D, s: OverlayState, tp: ToolPreview): void {
  const { view: v, colors: c } = s;
  if (tp.pending.length) {
    traceSegs(ctx, v, tp.pending);
    ctx.strokeStyle = c.selection;
    ctx.lineWidth = 3;
    ctx.stroke();
  }
  if (tp.segs.length) {
    traceSegs(ctx, v, tp.segs);
    if (tp.tone === "remove") {
      ctx.strokeStyle = "#ff5a5a";
      ctx.lineWidth = 4;
      ctx.setLineDash([5, 3]);
    } else {
      ctx.strokeStyle = c.selection;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const q of tp.points ?? []) {
    const [x, y] = toScreen(v.camera, q);
    ctx.strokeStyle = c.selection;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, 2 * Math.PI);
    ctx.stroke();
  }
}

/** 範囲選択の矩形 */
function drawBox(ctx: CanvasRenderingContext2D, s: OverlayState, [a, b]: [Point, Point]): void {
  const [xa, ya] = toScreen(s.view.camera, a);
  const [xb, yb] = toScreen(s.view.camera, b);
  ctx.save();
  ctx.globalAlpha = 0.12;
  ctx.fillStyle = s.colors.selection;
  ctx.fillRect(Math.min(xa, xb), Math.min(ya, yb), Math.abs(xb - xa), Math.abs(yb - ya));
  ctx.restore();
  ctx.strokeStyle = s.colors.selection;
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 3]);
  ctx.strokeRect(Math.min(xa, xb), Math.min(ya, yb), Math.abs(xb - xa), Math.abs(yb - ya));
  ctx.setLineDash([]);
}

function drawProfile(ctx: CanvasRenderingContext2D, s: OverlayState, line: [Point, Point]): void {
  const v = s.view;
  segment(ctx, v, line[0], line[1]);
  ctx.strokeStyle = s.colors.profile;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 4]);
  ctx.stroke();
  ctx.setLineDash([]);
  for (const q of line) {
    const [x, y] = toScreen(v.camera, q);
    dot(ctx, x, y, 4, s.colors.profile);
  }
}

/** 計測: 2 点の線と長さ・角度 */
function drawMeasure(ctx: CanvasRenderingContext2D, s: OverlayState, m: [Point, Point]): void {
  const v = s.view;
  segment(ctx, v, m[0], m[1]);
  ctx.strokeStyle = s.colors.measure;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  const [x0, y0] = toScreen(v.camera, m[0]);
  const [x1, y1] = toScreen(v.camera, m[1]);
  dot(ctx, x0, y0, 3, s.colors.measure);
  dot(ctx, x1, y1, 3, s.colors.measure);
  const dx = m[1][0] - m[0][0];
  const dy = m[1][1] - m[0][1];
  const len = Math.hypot(dx, dy);
  const unit = lengthUnitLabel(s.lengthUnit);
  const text = `${fmtLen(len, s.lengthUnit)} ${unit}  (Δ${s.axisNames[0]} ${fmtLen(dx, s.lengthUnit)}, Δ${s.axisNames[1]} ${fmtLen(dy, s.lengthUnit)}, ${((Math.atan2(dy, dx) * 180) / Math.PI).toFixed(1)}°)`;
  label(ctx, text, (x0 + x1) / 2, (y0 + y1) / 2 - 8, s.colors.measure);
}

function fmtLen(m: number, unit: LengthUnit): string {
  return String(Number(toDisplayLength(m, unit).toPrecision(5)));
}

function drawProbe(ctx: CanvasRenderingContext2D, s: OverlayState, p: ProbeBox): void {
  const { view: v, colors: c } = s;
  const [px, py] = toScreen(v.camera, p.pt);
  ctx.strokeStyle = c.probe;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(px - 7, py);
  ctx.lineTo(px + 7, py);
  ctx.moveTo(px, py - 7);
  ctx.lineTo(px, py + 7);
  ctx.stroke();
  dot(ctx, px, py, 2.5, c.probe);
  ctx.font = FONT;
  const pad = 6;
  const lh = 15;
  const w = Math.max(...p.lines.map((l) => ctx.measureText(l).width)) + 2 * pad;
  const h = lh * p.lines.length + 2 * pad;
  let bx = px + 10;
  let by = py - h - 10;
  if (bx + w > v.width) bx = px - w - 10;
  if (by < 0) by = py + 10;
  ctx.fillStyle = c.panel;
  ctx.strokeStyle = c.panelBorder;
  ctx.lineWidth = 1;
  ctx.fillRect(bx, by, w, h);
  ctx.strokeRect(bx + 0.5, by + 0.5, w - 1, h - 1);
  ctx.fillStyle = c.text;
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  p.lines.forEach((l, i) => ctx.fillText(l, bx + pad, by + pad + i * lh));
}

/** カラーバー (右下、目盛り 5 個と見出し) */
function drawColorbar(ctx: CanvasRenderingContext2D, s: OverlayState, cb: Colorbar): void {
  const { view: v, colors: c } = s;
  const barW = 14;
  const barH = Math.min(160, Math.max(80, v.height * 0.35));
  ctx.font = FONT;
  // 一定の場 (最小 = 最大) は真ん中に 1 つだけ
  const flat = cb.range.hi === cb.range.lo;
  const ticks = flat ? [cb.range.labelMin] : colorbarTicks(cb.range, 5);
  const texts = ticks.map((x, i) => (flat ? x : i === 0 ? cb.range.labelMin : i === ticks.length - 1 ? cb.range.labelMax : x)).map(formatColorbarValue);
  const title = cb.unit ? `${cb.label} [${cb.unit}]` : cb.label;
  const textW = Math.max(...texts.map((t) => ctx.measureText(t).width));
  const x = v.width - 12 - textW - 8 - barW;
  const y = v.height - barH - 24;
  const grad = ctx.createLinearGradient(0, y, 0, y + barH);
  for (let i = 0; i <= 32; i++) grad.addColorStop(i / 32, colormapCss(cb.colormap, 1 - i / 32));
  ctx.fillStyle = grad;
  ctx.fillRect(x, y, barW, barH);
  ctx.strokeStyle = c.panelBorder;
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 0.5, barW - 1, barH - 1);
  texts.forEach((t, i) => {
    const ty = flat ? y + barH / 2 : y + barH - (barH * i) / (texts.length - 1);
    ctx.beginPath();
    ctx.moveTo(x + barW, ty);
    ctx.lineTo(x + barW + 3, ty);
    ctx.strokeStyle = c.text;
    ctx.stroke();
    label(ctx, t, x + barW + 6, ty, c.text, "left", "middle");
  });
  label(ctx, title + (cb.range.log ? " (log)" : ""), x + barW, y - 8, c.text, "right", "bottom");
}

function drawRulers(ctx: CanvasRenderingContext2D, s: OverlayState): void {
  const { view: v, colors: c } = s;
  const size = rulerSize(s.rulerFont);
  const major = gridStep(v.camera);
  const minor = major / 10;
  ctx.fillStyle = c.rulerBg;
  ctx.fillRect(0, 0, v.width, size);
  ctx.fillRect(0, 0, size, v.height);
  ctx.strokeStyle = c.rulerBorder;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, size + 0.5);
  ctx.lineTo(v.width, size + 0.5);
  ctx.moveTo(size + 0.5, 0);
  ctx.lineTo(size + 0.5, v.height);
  ctx.stroke();
  ctx.font = `${s.rulerFont}px system-ui, -apple-system, 'Segoe UI', sans-serif`;
  ctx.fillStyle = c.rulerText;
  ctx.strokeStyle = c.rulerTick;
  // 上: x
  const [x0w] = toWorld(v.camera, size, 0);
  const [x1w] = toWorld(v.camera, v.width, 0);
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.beginPath();
  for (let k = Math.ceil(x0w / minor); k * minor <= x1w; k++) {
    const x = Math.round(toScreen(v.camera, [k * minor, 0])[0]) + 0.5;
    const isMajor = k % 10 === 0;
    ctx.moveTo(x, size - (isMajor ? 9 : 4));
    ctx.lineTo(x, size);
    if (isMajor) ctx.fillText(tickLabel(k * minor, major, s.lengthUnit), x + 2, 1);
  }
  ctx.stroke();
  // 左: y
  const [, y1w] = toWorld(v.camera, 0, size);
  const [, y0w] = toWorld(v.camera, 0, v.height);
  ctx.textBaseline = "middle";
  ctx.beginPath();
  for (let k = Math.ceil(y0w / minor); k * minor <= y1w; k++) {
    const y = Math.round(toScreen(v.camera, [0, k * minor])[1]) + 0.5;
    const isMajor = k % 10 === 0;
    ctx.moveTo(size - (isMajor ? 9 : 4), y);
    ctx.lineTo(size, y);
    if (isMajor && y > size + 8) {
      ctx.save();
      ctx.translate(size - 11, y);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(tickLabel(k * minor, major, s.lengthUnit), 0, 0);
      ctx.restore();
    }
  }
  ctx.stroke();
  // カーソルの位置
  if (s.cursor) {
    const [cx, cy] = toScreen(v.camera, s.cursor);
    ctx.strokeStyle = c.draw;
    ctx.beginPath();
    if (cx >= size) {
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, size);
    }
    if (cy >= size) {
      ctx.moveTo(0, cy);
      ctx.lineTo(size, cy);
    }
    ctx.stroke();
  }
  // 角: 軸の名前と単位
  ctx.fillStyle = c.rulerBg;
  ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = c.rulerBorder;
  ctx.strokeRect(0.5, 0.5, size - 1, size - 1);
  ctx.fillStyle = c.rulerTick;
  ctx.font = `${Math.min(s.rulerFont, 10)}px system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  // 軸対称では軸の名前も (z/r・r/z、v1 と同じ)
  if (s.axisNames[0] !== "x") {
    ctx.fillText(`${s.axisNames[0]}/${s.axisNames[1]}`, size / 2, size / 2 - 5);
    ctx.fillText(lengthUnitLabel(s.lengthUnit), size / 2, size / 2 + 6);
  } else ctx.fillText(lengthUnitLabel(s.lengthUnit), size / 2, size / 2);
}

export function drawOverlay(ctx: CanvasRenderingContext2D, s: OverlayState): void {
  const { view: v } = s;
  ctx.clearRect(0, 0, v.width, v.height);
  ctx.lineCap = "butt";
  ctx.lineJoin = "miter";
  if (s.arrows.length) drawArrows(ctx, s.arrows);
  drawGeometry(ctx, s);
  if (s.overlays.amr && s.amrBoxes.length) {
    ctx.strokeStyle = s.colors.amr;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([2, 3]);
    for (const [a, b] of s.amrBoxes) {
      pathPolygon(ctx, v, [a, [b[0], a[1]], b, [a[0], b[1]]]);
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  for (const pl of s.placements) drawPlacement(ctx, s, pl);
  if (s.emitter) drawEmitter(ctx, s, s.emitter);
  if (s.injector) drawEmitter(ctx, s, s.injector, s.colors.injector, "PIC");
  if (s.profile) drawProfile(ctx, s, s.profile);
  if (s.fillPreview) drawFillPreview(ctx, s, s.fillPreview);
  drawSelection(ctx, s);
  if (s.measure) drawMeasure(ctx, s, s.measure);
  drawRubberBand(ctx, s);
  if (s.preview.box) drawBox(ctx, s, s.preview.box);
  if (s.toolPreview) drawToolPreview(ctx, s, s.toolPreview);
  if (s.snapMark) drawSnapMark(ctx, s, s.snapMark, s.snapLabel);
  if (s.colorbar) drawColorbar(ctx, s, s.colorbar);
  if (s.probe) drawProbe(ctx, s, s.probe);
  if (s.overlays.rulers) drawRulers(ctx, s);
}
