import { useCallback, useEffect, useRef, useState } from "react";
import type {
  BoundaryCondition,
  CircleShape,
  Emitter,
  MeshResult,
  PicLiveFrame,
  Point,
  Project,
  Region,
  SolveResult,
  TraceResult,
} from "../types";
import { computeIsolines } from "./isolines";
import { LENGTH_UNIT_LABEL, mToUnit } from "../units";
import type { LengthUnit } from "../units";
import { arrayMin, arrayMax } from "../mathUtils";

/**
 * CAD キャンバス
 * - パン: 中ボタンドラッグ / Space+左ドラッグ
 * - ズーム: ホイール (カーソル中心)
 * - 表示: グリッド、ルーラー、ジオメトリ、解析結果 (電位/|E| カラーマップ、等電位線、ベクトル、カラーバー)
 * - 作図ツール: 選択 (頂点/中点グリップ編集含む) / ポリライン / 矩形 / 円 / エミッタ (グリッドスナップ対応)
 * - 粒子軌道: traceResult をシアン系の半透明ポリラインで描画 (吸収位置は小さな点)
 */

export type Tool =
  | "select"
  | "polyline"
  | "rect"
  | "circle"
  | "profile"
  | "emitter"
  | "collector"
  | "gasbc"
  | "eedfbox"
  | "meshref";

// カラーマップの対象: 電位 V か |E|
export type FieldView = "v" | "e_abs";

// IEDF/IADF コレクタ線分の描画用ビュー (キャンバス表示に必要な最小限のみ)
export interface PicCollectorView {
  label: string;
  p1: Point;
  p2: Point;
}

// コレクタ表示の固定パレット (黄・シアン・マゼンタ・緑…)。インデックスを長さで循環させる
const COLLECTOR_COLORS = [
  "#ffd400", // 黄
  "#4dd4ff", // シアン
  "#ff4dd4", // マゼンタ
  "#4ddd8c", // 緑
  "#ff9d4d", // オレンジ
  "#b070f0", // 紫
  "#39d3d3", // ティール
  "#ff6666", // 赤
];
function collectorColor(i: number): string {
  return COLLECTOR_COLORS[i % COLLECTOR_COLORS.length];
}

// DSMC 線分境界の表示色 (橙系)。コレクタのパレットとは重ならない固定色にして、
// 見た目でコレクタ (黄・シアン等) とガス境界を区別できるようにする
const GAS_BOUNDARY_COLOR = "#ffb454";

// EEDF/EEPF 集計領域の表示色 (紫系)。コレクタ・ガス境界のパレットと重ならない固定色にする (prompts/85)
const EEDF_REGION_COLOR = "#c792ea";

// 辺 (線分) ローカルメッシュサイズの表示色 (水色系)。他のオーバーレイと重ならない固定色にする (prompts/90)
const EDGE_MESH_COLOR = "#59c2ff";

// 配置済み EEDF/EEPF 集計領域 (軸平行矩形) の表示用ビュー (最大4個、常時オーバーレイ表示対象)
export interface PicEedfRegionView {
  label: string;
  p1: Point;
  p2: Point;
}

// 配置済み DSMC 線分境界のオーバーレイ表示用ビュー (p1/p2 を持つ境界のみが対象、prompts/72)
export interface GasBoundaryView {
  label: string;
  p1: Point;
  p2: Point;
}

// 配置済み辺ローカルメッシュサイズの描画用ビュー (キャンバス表示に必要な最小限のみ、prompts/90)
export interface EdgeMeshSizeView {
  label: string;
  p1: Point;
  p2: Point;
}

// PIC結果フィールド表示 (done後の「結果表示」セレクトでライブ以外を選んだ場合の描画データ)。
// 値配列・節点/要素の別・単位・対数フラグを1つにまとめて渡すことで、描画分岐の散乱を避ける。
// 周期アニメーション再生時もこの型を流用し (App側で優先的に構築して渡す)、
// fixedRange/particles を指定することで「現在ビンの値+固定min/max+粒子」を表せるようにする
export interface PicFieldView {
  mesh: MeshResult;
  values: number[]; // nodeBased なら節点値 (長さ=nodes.length)、そうでなければ要素値 (長さ=triangles.length)
  nodeBased: boolean; // true: 節点値 (要素は3節点平均で塗る)。false: 要素値 (e_abs)
  unit: string; // カラーバーに表示する単位
  log: boolean; // 対数スケール表示 (値≤0は最小正値にクランプ。全て≤0なら線形にフォールバック)
  // 周期アニメーション用: 指定時はカラースケール (min/max/対数クランプ用の最小正値) をこの値に固定する
  // (未指定時は values から都度自動計算する、従来通りの挙動)
  fixedRange?: { min: number; max: number; minPositive: number };
  // 周期アニメーション用: 該当ビンの粒子スナップショットをドットでオーバーレイする (電子/イオン)
  particles?: { electron: Point[]; ion: Point[] };
}

interface Props {
  project: Project;
  // 長さの表示単位 (mm/µm)。project 内部は常に m のまま (カーソル座標・ルーラー目盛りの表示に使う)
  lengthUnit: LengthUnit;
  result: SolveResult | null;
  // Mesh ボタン (解析なし) で生成したメッシュ。result/picFrame がある間は通常そちら側の
  // 表示を優先するが、meshResultIsLatest が true の間 (直近の操作が Mesh 実行だった場合)
  // はこちらを最優先する (prompts/89 ②: 再メッシュしてもプレビューが更新されない不具合の修正)
  meshResult: MeshResult | null;
  // true: meshResult が result/picFrame よりも「最後に生成されたもの」であることを示し、
  // キャンバスのベースレイヤー描画で最優先にする。Solve 完了・PIC 開始で false に戻る
  // (App.tsx の meshPreviewFresh 参照)。未指定時は従来通り false 相当 (result/picFrame 優先)
  meshResultIsLatest?: boolean;
  showMesh: boolean;
  tool: Tool;
  gridSnap: boolean;
  // ルーラー目盛りラベルのフォントサイズ (px)
  rulerFontSize: number;
  selectedRegionId: string | null;
  fieldView: FieldView;
  showIsolines: boolean;
  showVectors: boolean;
  profileLine: [Point, Point] | null;
  // IEDF/IADF コレクタ線分 (複数、常時オーバーレイ表示の対象。最大8個、prompts/37)
  collectors: PicCollectorView[];
  // コレクタ一覧 (PICパネル) で選択中のインデックス。キャンバス上で該当線分を強調表示する
  selectedCollectorIndex: number | null;
  // EEDF/EEPF 集計領域 (複数、常時オーバーレイ表示の対象。最大4個、prompts/85)
  eedfRegions: PicEedfRegionView[];
  // 領域一覧 (PICパネル) で選択中のインデックス。キャンバス上で該当矩形を強調表示する
  selectedEedfIndex: number | null;
  // 粒子エミッタ (常時オーバーレイ表示の対象)。粒子パネル側で必ず既定値を持つため常に非 null
  emitter: Emitter | null; // null なら描画しない (「エミッタを表示」オプションOFF)
  // 粒子軌道トレース結果 (Trace 実行前は null)
  traceResult: TraceResult | null;
  showTrajectories: boolean;
  // PICライブ表示 (started の mesh + 最新 frame)。実行中〜done後の最終フレームまで非null。
  // 存在する間は既存の Solve 結果表示より優先して描画する (picFieldView がある間はそちらを優先)
  picFrame: PicLiveFrame | null;
  // PIC結果フィールド表示 (done後、「結果表示」セレクトでライブ以外を選んだ場合のみ非null)。
  // 存在する間は picFrame / Solve 結果表示より優先して描画し、粒子オーバーレイは出さない
  picFieldView: PicFieldView | null;
  // DSMC ライブ粒子表示 (prompts/66)。実行中 (かつ表示チェックON) のみ非null で、
  // 間引き済み座標をメッシュ/領域の上に点描画する。App 側で実行中〜完了までのみ渡す
  // (完了後は結果フィールド表示に切り替わるため残さない)
  gasParticles?: Point[] | null;
  // 配置済み DSMC 線分境界 (常時オーバーレイ表示の対象、prompts/72)。エッジ指定のみの境界は
  // p1/p2 を持たないため対象外 (App 側で p1/p2 を持つものだけ抽出して渡す)
  gasBoundaries?: GasBoundaryView[];
  // 配置済み辺ローカルメッシュサイズ (常時オーバーレイ表示の対象、prompts/90)
  edgeMeshSizes?: EdgeMeshSizeView[];
  onSelectRegion: (id: string | null) => void;
  onDeleteRegion: (id: string) => void;
  onAddRegion: (geom: Point[] | CircleShape) => void;
  onMoveRegion: (id: string, dx: number, dy: number) => void;
  onEditRegionPolygon: (id: string, polygon: Point[]) => void;
  onEditRegionShape: (id: string, shape: CircleShape) => void;
  onProfileLine: (p1: Point, p2: Point) => void;
  onSetEmitter: (p1: Point, p2: Point) => void;
  onSetCollector: (p1: Point, p2: Point) => void;
  // ガス境界配置ツールの確定通知 (コレクタと同じ2点クリックUX、prompts/72)
  onSetGasBoundary: (p1: Point, p2: Point) => void;
  // EEDF/EEPF 領域配置ツールの確定通知 (矩形ツールと同じ2点クリックUX、prompts/85)。
  // p1/p2 は対角の2点 (順不同、そのまま渡す。矩形化は呼び出し側が行う)
  onSetEedfRegion: (p1: Point, p2: Point) => void;
  // 辺ローカルメッシュサイズ配置ツールの確定通知 (コレクタ・ガス境界と同じ2点クリックUX、prompts/90)
  onSetEdgeMeshSize: (p1: Point, p2: Point) => void;
}

interface View {
  scale: number; // px / m
  ox: number;    // 原点の画面座標 x
  oy: number;    // 原点の画面座標 y
}

// クリック判定の許容移動量 (mousedown→mouseup の画面上移動量, px)
const CLICK_MOVE_TOLERANCE_PX = 5;
// 選択ヒットテストの輪郭からの許容誤差 (画面px, ワールド換算時は /scale)
const EDGE_HIT_TOLERANCE_PX = 6;
// 頂点/中点ハンドルのヒットテスト許容誤差 (画面px)
const HANDLE_HIT_TOLERANCE_PX = 8;

// ルーラーの幅/高さ (画面px)。フォントサイズに応じて少し広げる
function rulerSizeFor(fontSize: number): number {
  return Math.max(24, fontSize * 2.2);
}

// 簡易 viridis カラーマップ
const STOPS: [number, number, number][] = [
  [68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37],
];
function colormap(t: number): string {
  const x = Math.min(0.9999, Math.max(0, t)) * (STOPS.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const [r1, g1, b1] = STOPS[i];
  const [r2, g2, b2] = STOPS[i + 1];
  return `rgb(${r1 + (r2 - r1) * f},${g1 + (g2 - g1) * f},${b1 + (b2 - b1) * f})`;
}

// 境界条件タイプ別の描画スタイル (色・線種・線幅)。凡例 (.bc-legend, style.css) の色と揃える
function bcStyle(type: BoundaryCondition["type"]): { color: string; dash: number[]; width: number } {
  if (type === "dirichlet") return { color: "#ff9d33", dash: [], width: 4 };
  if (type === "symmetry") return { color: "#39d353", dash: [8, 5], width: 2.5 };
  return { color: "#b070f0", dash: [8, 5], width: 2.5 }; // periodic
}

// 画面上でグリッド間隔が20px以上になるよう 1,10,100...mm 相当 (世界座標=m) の
// 10の冪から自動選択する。表示単位 (mm/µm) には依存しない (ラベル表示のみが追従する)
function gridStep(scale: number): number {
  let step = 0.001;
  while (step * scale < 20) step *= 10;
  return step;
}

// 点内包判定 (ray casting)
function pointInPolygon(pt: Point, poly: Point[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    const intersect =
      yi > pt[1] !== yj > pt[1] &&
      pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

// ポリゴンの面積 (符号なし, shoelace 公式)
function polygonArea(poly: Point[]): number {
  let a = 0;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    a += poly[j][0] * poly[i][1] - poly[i][0] * poly[j][1];
  }
  return Math.abs(a) / 2;
}

// 点と線分の最短距離 (ワールド座標系)
function distToSegment(pt: Point, a: Point, b: Point): number {
  const [px, py] = pt;
  const [ax, ay] = a;
  const [bx, by] = b;
  const dx = bx - ax;
  const dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  let t = lenSq > 0 ? ((px - ax) * dx + (py - ay) * dy) / lenSq : 0;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

// 点とポリゴン輪郭 (各エッジ) との最短距離
function distToPolygonEdges(pt: Point, poly: Point[]): number {
  let min = Infinity;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const d = distToSegment(pt, poly[j], poly[i]);
    if (d < min) min = d;
  }
  return min;
}

// 領域のヒットテスト: 内包 または 輪郭から許容誤差 (ワールド換算) 以内なら命中とする
function hitRegion(pt: Point, poly: Point[], edgeTolWorld: number): boolean {
  return pointInPolygon(pt, poly) || distToPolygonEdges(pt, poly) <= edgeTolWorld;
}

// circle (shape) 領域のヒットテスト: 中心距離 ≤ r+許容 (内包 or 円周付近を包含) なら命中とする
function hitCircle(pt: Point, shape: CircleShape, edgeTolWorld: number): boolean {
  const d = Math.hypot(pt[0] - shape.center[0], pt[1] - shape.center[1]);
  return d <= shape.radius + edgeTolWorld;
}

// 領域 (polygon / circle 共通) のヒットテスト
function hitRegionAny(pt: Point, region: Region, edgeTolWorld: number): boolean {
  if (region.shape) return hitCircle(pt, region.shape, edgeTolWorld);
  return hitRegion(pt, region.polygon ?? [], edgeTolWorld);
}

// 領域の面積 (polygon はシューレース公式、circle は πr²)
function regionArea(region: Region): number {
  if (region.shape) return Math.PI * region.shape.radius * region.shape.radius;
  return polygonArea(region.polygon ?? []);
}

// クリック位置に命中する領域を探す。複数命中時は面積が小さい方 (入れ子の内側) を優先する
function findRegionAt(pt: Point, regions: Region[], edgeTolWorld: number): Region | null {
  let best: Region | null = null;
  let bestArea = Infinity;
  for (const r of regions) {
    if (!hitRegionAny(pt, r, edgeTolWorld)) continue;
    const area = regionArea(r);
    if (area < bestArea) {
      bestArea = area;
      best = r;
    }
  }
  return best;
}

// カラーバーの数値ラベル整形: 大きい値/小さい値は指数表記で桁を抑える
function formatColorbarValue(v: number): string {
  if (v === 0) return "0";
  const av = Math.abs(v);
  if (av >= 1e4 || av < 1e-3) return v.toExponential(1);
  return v.toPrecision(4);
}

// 画面座標同士の距離 (px)
function screenDist(sx: number, sy: number, px: number, py: number): number {
  return Math.hypot(sx - px, sy - py);
}

// 頂点ハンドルのヒットテスト。許容誤差内で最も近い頂点のインデックスを返す
function findHandleVertex(
  poly: Point[],
  screenPt: { x: number; y: number },
  view: View,
  tolPx: number,
): number | null {
  let best: number | null = null;
  let bestD = tolPx;
  poly.forEach(([x, y], i) => {
    const d = screenDist(view.ox + x * view.scale, view.oy - y * view.scale, screenPt.x, screenPt.y);
    if (d <= bestD) {
      bestD = d;
      best = i;
    }
  });
  return best;
}

// 中点ハンドルのヒットテスト。エッジインデックス i は poly[i]-poly[(i+1)%n] の中点を指す
function findHandleMidpoint(
  poly: Point[],
  screenPt: { x: number; y: number },
  view: View,
  tolPx: number,
): number | null {
  let best: number | null = null;
  let bestD = tolPx;
  for (let i = 0; i < poly.length; i++) {
    const [ax, ay] = poly[i];
    const [bx, by] = poly[(i + 1) % poly.length];
    const mx = (ax + bx) / 2;
    const my = (ay + by) / 2;
    const d = screenDist(view.ox + mx * view.scale, view.oy - my * view.scale, screenPt.x, screenPt.y);
    if (d <= bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

// 半径ハンドル (円周上、角度0°=中心から右方向の点) のヒットテスト
function hitHandleRadius(
  shape: CircleShape,
  screenPt: { x: number; y: number },
  view: View,
  tolPx: number,
): boolean {
  const hx = view.ox + (shape.center[0] + shape.radius) * view.scale;
  const hy = view.oy - shape.center[1] * view.scale;
  return screenDist(hx, hy, screenPt.x, screenPt.y) <= tolPx;
}

export default function CadCanvas({
  project,
  lengthUnit,
  result,
  meshResult,
  meshResultIsLatest = false,
  showMesh,
  tool,
  gridSnap,
  rulerFontSize,
  selectedRegionId,
  fieldView,
  showIsolines,
  showVectors,
  profileLine,
  collectors,
  selectedCollectorIndex,
  eedfRegions,
  selectedEedfIndex,
  emitter,
  traceResult,
  showTrajectories,
  picFrame,
  picFieldView,
  gasParticles,
  gasBoundaries = [],
  edgeMeshSizes = [],
  onSelectRegion,
  onDeleteRegion,
  onAddRegion,
  onMoveRegion,
  onEditRegionPolygon,
  onEditRegionShape,
  onProfileLine,
  onSetEmitter,
  onSetCollector,
  onSetGasBoundary,
  onSetEedfRegion,
  onSetEdgeMeshSize,
}: Props) {
  // 軸対称 (r-z) モードかどうか。x=z(軸方向)・y=r(径方向) と読み替えて表示する
  const isRz = project.coord === "rz";
  // 軸対称 (r-z、左辺が軸) モードかどうか。x=r(径方向)・y=z(軸方向) と読み替えて表示する
  const isRzX0 = project.coord === "rz_x0";
  const isAxisym = isRz || isRzX0;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [view, setView] = useState<View | null>(null);
  // キャンバス親要素のサイズ変化 (サイドパネル幅変更・ウィンドウリサイズ) で再描画する
  const [resizeTick, setResizeTick] = useState(0);
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setResizeTick((t) => t + 1));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const [cursor, setCursor] = useState<Point | null>(null);
  const [drawPts, setDrawPts] = useState<Point[]>([]);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const didDragRef = useRef(false);
  const spaceRef = useRef(false);
  // mousedown 時の画面座標 (クリック/ドラッグの判定に使う)
  const mouseDownScreenRef = useRef<{ x: number; y: number } | null>(null);
  // 選択領域の移動ドラッグ: 開始時の領域IDとワールド座標を保持
  const moveDragRef = useRef<{ id: string; startWorld: Point } | null>(null);
  // 移動ドラッグ中のプレビュー量 (ワールド座標系、グリッドスナップ適用後)
  const [movePreviewDelta, setMovePreviewDelta] = useState<Point | null>(null);
  // 頂点/中点ハンドルのドラッグ: 対象領域IDと編集中の頂点インデックス
  const vertexDragRef = useRef<{ regionId: string; index: number } | null>(null);
  // 頂点編集ドラッグ中のプレビュー多角形 (ワールド座標、グリッドスナップ適用後)
  const [vertexPreviewPolygon, setVertexPreviewPolygon] = useState<Point[] | null>(null);
  // circle 領域の半径ハンドルのドラッグ: 対象領域ID
  const radiusDragRef = useRef<{ regionId: string } | null>(null);
  // 半径ドラッグ中のプレビュー半径 (ワールド座標系、グリッドスナップ適用後)
  const [shapeRadiusPreview, setShapeRadiusPreview] = useState<number | null>(null);

  const toWorld = useCallback(
    (px: number, py: number, v: View): Point =>
      [(px - v.ox) / v.scale, (v.oy - py) / v.scale],
    [],
  );

  // グリッドスナップ適用後のワールド座標を取得
  const getPoint = useCallback(
    (px: number, py: number, v: View): Point => {
      const w = toWorld(px, py, v);
      if (!gridSnap) return w;
      const step = gridStep(v.scale) / 10;
      return [Math.round(w[0] / step) * step, Math.round(w[1] / step) * step];
    },
    [toWorld, gridSnap],
  );

  // ツール切替時は作図途中の状態を破棄
  useEffect(() => {
    setDrawPts([]);
  }, [tool]);

  // 選択が変わったら (Undo/Redo・削除・再選択など) 進行中のドラッグ状態を破棄する
  useEffect(() => {
    moveDragRef.current = null;
    setMovePreviewDelta(null);
    vertexDragRef.current = null;
    setVertexPreviewPolygon(null);
    radiusDragRef.current = null;
    setShapeRadiusPreview(null);
  }, [selectedRegionId]);

  // 初期表示: domain 全体をフィット
  useEffect(() => {
    const el = canvasRef.current;
    if (!el || view) return;
    const poly = project.geometry.domain.polygon;
    const xs = poly.map((p) => p[0]);
    const ys = poly.map((p) => p[1]);
    const w = arrayMax(xs) - arrayMin(xs);
    const h = arrayMax(ys) - arrayMin(ys);
    const rect = el.getBoundingClientRect();
    const scale = 0.8 * Math.min(rect.width / w, rect.height / h);
    setView({
      scale,
      ox: rect.width / 2 - scale * (arrayMin(xs) + w / 2),
      oy: rect.height / 2 + scale * (arrayMin(ys) + h / 2),
    });
  }, [project, view]);

  // 描画
  useEffect(() => {
    const el = canvasRef.current;
    if (!el || !view) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = el.getBoundingClientRect();
    el.width = rect.width * dpr;
    el.height = rect.height * dpr;
    const ctx = el.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const sx = (x: number) => view.ox + x * view.scale;
    const sy = (y: number) => view.oy - y * view.scale;

    // グリッド (世界座標の10の冪ステップを自動選択。表示単位には依存しない)
    const step = gridStep(view.scale);
    ctx.strokeStyle = "#2a2f38";
    ctx.lineWidth = 1;
    const x0 = Math.floor(toWorld(0, 0, view)[0] / step) * step;
    const y0 = Math.floor(toWorld(0, rect.height, view)[1] / step) * step;
    ctx.beginPath();
    for (let x = x0; sx(x) < rect.width; x += step) {
      ctx.moveTo(sx(x), 0); ctx.lineTo(sx(x), rect.height);
    }
    for (let y = y0; sy(y) > 0; y += step) {
      ctx.moveTo(0, sy(y)); ctx.lineTo(rect.width, sy(y));
    }
    ctx.stroke();

    // カラーバー描画の共通ヘルパー (右下、画面固定・縦グラデーション)。
    // Solve結果 / PIC結果フィールド のどちらの表示でも使い回す
    const drawColorbar = (minVal: number, maxVal: number, unit: string) => {
      const barW = 16;
      const barH = 140;
      const marginRight = 10; // キャンバス右端からの余白
      const labelGap = 8;     // バーとラベルの間隔

      const maxLabel = `${formatColorbarValue(maxVal)} ${unit}`;
      const minLabel = `${formatColorbarValue(minVal)} ${unit}`;

      ctx.font = "11px system-ui, sans-serif";
      // ラベル幅を測り、右端からはみ出さないようバー位置自体を左にずらして確保する
      const labelW = Math.max(ctx.measureText(maxLabel).width, ctx.measureText(minLabel).width);
      const barX = rect.width - marginRight - labelW - labelGap - barW;
      const barY = rect.height - barH - 24;

      const grad = ctx.createLinearGradient(0, barY, 0, barY + barH);
      const steps = 16;
      for (let i = 0; i <= steps; i++) {
        grad.addColorStop(i / steps, colormap(1 - i / steps));
      }
      ctx.fillStyle = grad;
      ctx.fillRect(barX, barY, barW, barH);
      ctx.strokeStyle = "rgba(216,220,228,0.6)";
      ctx.lineWidth = 1;
      ctx.strokeRect(barX, barY, barW, barH);

      ctx.fillStyle = "#d8dce4";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(maxLabel, barX + barW + labelGap, barY);
      ctx.fillText(minLabel, barX + barW + labelGap, barY + barH);
    };

    // 粒子オーバーレイの共通ヘルパー (1〜2px の小さな矩形。多数点でも軽く保つため fillRect を使う)。
    // PICライブ表示 / 周期アニメーションの粒子スナップショット / DSMCライブ粒子表示 (prompts/66) で
    // 共用する。半径 (px) は種別ごとに変えられるよう省略可にする (既定は従来通り0.8px)
    const drawSpecies = (pts: Point[], color: string, radiusPx = 0.8) => {
      ctx.fillStyle = color;
      const d = radiusPx * 2;
      for (const [px0, py0] of pts) {
        const px = sx(px0);
        const py = sy(py0);
        ctx.fillRect(px - radiusPx, py - radiusPx, d, d);
      }
    };

    // PIC結果フィールド表示: done後に「結果表示」セレクトでライブ以外を選んだ場合、
    // 選択したフィールドをカラーマップで描画する (節点値は要素を3節点平均で塗り、
    // 要素値(e_abs)はそのまま塗る)。対数スケール指定時は値≤0を全体の最小正値にクランプしてから
    // log10 する (全て≤0なら線形にフォールバック)。粒子オーバーレイ・カラーバーはここで完結させ、
    // Solve/Mesh/PICライブ側の描画は行わない (以降のブロックで !picFieldView を条件に含める)
    if (picFieldView) {
      const { nodes, triangles } = picFieldView.mesh;
      const { values, nodeBased, log, fixedRange } = picFieldView;

      let rawMin = Infinity;
      let rawMax = -Infinity;
      let minPositive = Infinity;
      for (const v of values) {
        if (v < rawMin) rawMin = v;
        if (v > rawMax) rawMax = v;
        if (v > 0 && v < minPositive) minPositive = v;
      }
      if (!Number.isFinite(rawMin)) { rawMin = 0; rawMax = 0; }
      // 周期アニメーション等、フレーム間で色が暴れないよう固定範囲が指定されていればそちらを使う
      if (fixedRange) {
        rawMin = fixedRange.min;
        rawMax = fixedRange.max;
        minPositive = fixedRange.minPositive;
      }
      const useLog = log && Number.isFinite(minPositive);
      const transformed = useLog
        ? values.map((v) => Math.log10(v > 0 ? v : minPositive))
        : values;
      const tMin = useLog ? Math.log10(minPositive) : rawMin;
      const tMax = useLog ? Math.log10(rawMax) : rawMax;
      const range = tMax - tMin || 1;

      for (let i = 0; i < triangles.length; i++) {
        const [a, b, c] = triangles[i];
        const val = nodeBased ? (transformed[a] + transformed[b] + transformed[c]) / 3 : transformed[i];
        const t = (val - tMin) / range;
        ctx.fillStyle = colormap(t);
        ctx.beginPath();
        ctx.moveTo(sx(nodes[a][0]), sy(nodes[a][1]));
        ctx.lineTo(sx(nodes[b][0]), sy(nodes[b][1]));
        ctx.lineTo(sx(nodes[c][0]), sy(nodes[c][1]));
        ctx.closePath();
        ctx.fill();
      }

      // 周期アニメーションの粒子スナップショット (表示トグルは App 側で particles の有無に反映済み)
      if (picFieldView.particles) {
        drawSpecies(picFieldView.particles.electron, "#4dd4ff"); // 電子: シアン
        drawSpecies(picFieldView.particles.ion, "#ff9d4d");       // イオン: オレンジ
      }

      drawColorbar(rawMin, rawMax, picFieldView.unit);
    }

    // meshResult (Mesh ボタン) が result/picFrame より「後に生成された」場合は、それらより
    // 手前でメッシュのワイヤーフレームを最優先表示する (prompts/89 ②)。result も picFrame も
    // ジオメトリ変更等でしかクリアされず Mesh 実行では消えないため、この優先判定なしだと
    // 「一度計算した後は再メッシュしてもプレビューが更新されない」不具合になる
    const meshWireTakesPriority = !!meshResult && meshResultIsLatest;

    // PICライブ表示: 選択フィールド (電位/電子密度/イオン密度、prompts/81) を
    // picFieldView と同じ節点値/要素値・対数スケール対応の経路で描画し、
    // 粒子を点描画する (電子=シアン、イオン=オレンジ)。実行中〜done後の最終フレームまで
    // Solve/Mesh 側の表示より優先する (ただし meshWireTakesPriority のときは Mesh 側が勝つ)。
    // フレームごとに min/max を再計算する。picFieldView (結果フィールド表示) が
    // 選択されている間はこちらは描画しない
    if (!picFieldView && !meshWireTakesPriority && picFrame) {
      const { nodes, triangles } = picFrame.mesh;
      const { values, nodeBased, log } = picFrame;

      let rawMin = Infinity;
      let rawMax = -Infinity;
      let minPositive = Infinity;
      for (const v of values) {
        if (v < rawMin) rawMin = v;
        if (v > rawMax) rawMax = v;
        if (v > 0 && v < minPositive) minPositive = v;
      }
      if (!Number.isFinite(rawMin)) { rawMin = 0; rawMax = 0; }
      const useLog = log && Number.isFinite(minPositive);
      const transformed = useLog
        ? values.map((v) => Math.log10(v > 0 ? v : minPositive))
        : values;
      const tMin = useLog ? Math.log10(minPositive) : rawMin;
      const tMax = useLog ? Math.log10(rawMax) : rawMax;
      const range = tMax - tMin || 1;

      for (let i = 0; i < triangles.length; i++) {
        const [a, b, c] = triangles[i];
        const val = nodeBased ? (transformed[a] + transformed[b] + transformed[c]) / 3 : transformed[i];
        const t = (val - tMin) / range;
        ctx.fillStyle = colormap(t);
        ctx.beginPath();
        ctx.moveTo(sx(nodes[a][0]), sy(nodes[a][1]));
        ctx.lineTo(sx(nodes[b][0]), sy(nodes[b][1]));
        ctx.lineTo(sx(nodes[c][0]), sy(nodes[c][1]));
        ctx.closePath();
        ctx.fill();
      }

      // 粒子 (共通ヘルパーで描画)
      drawSpecies(picFrame.particles.electron, "#4dd4ff"); // 電子: シアン
      drawSpecies(picFrame.particles.ion, "#ff9d4d");       // イオン: オレンジ

      drawColorbar(rawMin, rawMax, picFrame.unit);
    }

    // Mesh ボタンで生成したメッシュのワイヤーフレーム (解析結果がない状態でも見えるようにする)。
    // 通常は Solve/PIC 側の表示 (カラーマップ) を優先するが、meshWireTakesPriority が
    // true (= meshResult が最後に生成されたもの) の場合はこちらを優先する (prompts/89 ②)。
    // PICライブ/結果フィールド表示中は出さない
    if (!picFieldView && meshResult && (meshWireTakesPriority || (!picFrame && !result))) {
      const { nodes, triangles, region_of_triangle } = meshResult;
      const regionColor = (type: Region["type"]): string =>
        type === "conductor"
          ? "rgba(224,176,80,0.16)"
          : type === "dielectric"
            ? "rgba(80,176,224,0.16)"
            : "rgba(176,112,224,0.16)";
      for (let i = 0; i < triangles.length; i++) {
        const [a, b, c] = triangles[i];
        const regionIdx = region_of_triangle[i];
        const region = regionIdx >= 0 ? project.geometry.regions[regionIdx] : undefined;
        ctx.beginPath();
        ctx.moveTo(sx(nodes[a][0]), sy(nodes[a][1]));
        ctx.lineTo(sx(nodes[b][0]), sy(nodes[b][1]));
        ctx.lineTo(sx(nodes[c][0]), sy(nodes[c][1]));
        ctx.closePath();
        if (region) {
          ctx.fillStyle = regionColor(region.type);
          ctx.fill();
        }
        ctx.strokeStyle = "rgba(216,220,228,0.45)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    // 解析結果: カラーマップ (fieldView に応じて電位 V または要素ごとの |E| を塗る)。
    // PICライブ/結果フィールド表示中、および meshWireTakesPriority (prompts/89 ②) の間は出さない
    if (!picFieldView && !meshWireTakesPriority && !picFrame && result) {
      const { nodes, triangles } = result.mesh;
      const { v, v_min, v_max, e_field, e_abs_max } = result;

      if (fieldView === "v") {
        const range = v_max - v_min || 1;
        for (let i = 0; i < triangles.length; i++) {
          const [a, b, c] = triangles[i];
          const t = ((v[a] + v[b] + v[c]) / 3 - v_min) / range;
          ctx.fillStyle = colormap(t);
          ctx.beginPath();
          ctx.moveTo(sx(nodes[a][0]), sy(nodes[a][1]));
          ctx.lineTo(sx(nodes[b][0]), sy(nodes[b][1]));
          ctx.lineTo(sx(nodes[c][0]), sy(nodes[c][1]));
          ctx.closePath();
          ctx.fill();
          if (showMesh) {
            ctx.strokeStyle = "rgba(0,0,0,0.25)";
            ctx.stroke();
          }
        }
      } else {
        // e_abs: 要素ごとの |E| で塗る (0〜e_abs_max の線形スケール)
        const max = e_abs_max || 1;
        for (let i = 0; i < triangles.length; i++) {
          const [a, b, c] = triangles[i];
          const [ex, ey] = e_field[i];
          const t = Math.hypot(ex, ey) / max;
          ctx.fillStyle = colormap(t);
          ctx.beginPath();
          ctx.moveTo(sx(nodes[a][0]), sy(nodes[a][1]));
          ctx.lineTo(sx(nodes[b][0]), sy(nodes[b][1]));
          ctx.lineTo(sx(nodes[c][0]), sy(nodes[c][1]));
          ctx.closePath();
          ctx.fill();
          if (showMesh) {
            ctx.strokeStyle = "rgba(0,0,0,0.25)";
            ctx.stroke();
          }
        }
      }

      // 等電位線オーバーレイ (fieldView とは独立に表示可)
      if (showIsolines) {
        const isolines = computeIsolines(nodes, triangles, v, 15);
        ctx.strokeStyle = "rgba(0,0,0,0.5)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (const level of isolines) {
          for (const [p0, p1] of level) {
            ctx.moveTo(sx(p0[0]), sy(p0[1]));
            ctx.lineTo(sx(p1[0]), sy(p1[1]));
          }
        }
        ctx.stroke();
      }

      // E ベクトル矢印 (要素重心、向きのみ・長さ一定)
      if (showVectors) {
        const stride = Math.max(1, Math.ceil(triangles.length / 600));
        const arrowLen = 14; // px
        ctx.strokeStyle = "rgba(255,255,255,0.85)";
        ctx.fillStyle = "rgba(255,255,255,0.85)";
        ctx.lineWidth = 1.2;
        for (let i = 0; i < triangles.length; i += stride) {
          const [a, b, c] = triangles[i];
          const cx = (nodes[a][0] + nodes[b][0] + nodes[c][0]) / 3;
          const cy = (nodes[a][1] + nodes[b][1] + nodes[c][1]) / 3;
          const [ex, ey] = e_field[i];
          const mag = Math.hypot(ex, ey);
          if (mag < 1e-12) continue;
          // 画面座標系では y が反転するため向きベクトルも反転して合わせる
          const dx = (ex / mag) * arrowLen;
          const dy = -(ey / mag) * arrowLen;
          const x0 = sx(cx);
          const y0 = sy(cy);
          const x1 = x0 + dx;
          const y1 = y0 + dy;
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.lineTo(x1, y1);
          ctx.stroke();
          // 矢じり
          const ang = Math.atan2(dy, dx);
          const headLen = 4;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(
            x1 - headLen * Math.cos(ang - Math.PI / 6),
            y1 - headLen * Math.sin(ang - Math.PI / 6),
          );
          ctx.lineTo(
            x1 - headLen * Math.cos(ang + Math.PI / 6),
            y1 - headLen * Math.sin(ang + Math.PI / 6),
          );
          ctx.closePath();
          ctx.fill();
        }
      }
    }

    // ジオメトリ輪郭
    const drawPoly = (poly: Point[], color: string) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      poly.forEach(([x, y], i) =>
        i === 0 ? ctx.moveTo(sx(x), sy(y)) : ctx.lineTo(sx(x), sy(y)),
      );
      ctx.closePath();
      ctx.stroke();
    };
    // circle (shape) 領域は真円で描画する
    const drawCircle = (shape: CircleShape, color: string) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(sx(shape.center[0]), sy(shape.center[1]), shape.radius * view.scale, 0, Math.PI * 2);
      ctx.stroke();
    };
    drawPoly(project.geometry.domain.polygon, "#d8dce4");

    // 境界条件オーバーレイ: domain外周の各辺をBCタイプ別の色・線種で強調表示する
    // (なし/Neumannのエッジは上のドメイン輪郭のままで何も重ねない)
    {
      const domainPoly = project.geometry.domain.polygon;
      for (const b of project.geometry.boundaries) {
        const st = bcStyle(b.type);
        ctx.strokeStyle = st.color;
        ctx.lineWidth = st.width;
        ctx.setLineDash(st.dash);
        ctx.beginPath();
        for (const edgeIdx of b.edges) {
          const a = domainPoly[edgeIdx];
          const c = domainPoly[(edgeIdx + 1) % domainPoly.length];
          if (!a || !c) continue;
          ctx.moveTo(sx(a[0]), sy(a[1]));
          ctx.lineTo(sx(c[0]), sy(c[1]));
        }
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    // 軸対称モード: 対称軸 (r=0) を一点鎖線風 (細い破線+シアン系) でオーバーレイ表示する。
    // rz は下辺 (domainPoly[0]-[1])、rz_x0 は左辺 (domainPoly[3]-[0]) が対称軸になる
    if (isAxisym) {
      const domainPoly = project.geometry.domain.polygon;
      const [a, c] = isRzX0 ? [domainPoly[3], domainPoly[0]] : [domainPoly[0], domainPoly[1]];
      if (a && c) {
        ctx.strokeStyle = "#4dd4ff";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([10, 3, 2, 3]); // 一点鎖線風
        ctx.beginPath();
        ctx.moveTo(sx(a[0]), sy(a[1]));
        ctx.lineTo(sx(c[0]), sy(c[1]));
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    for (const r of project.geometry.regions) {
      const color = r.type === "conductor" ? "#e0b050" : r.type === "dielectric" ? "#50b0e0" : "#b070e0";
      if (r.shape) drawCircle(r.shape, color);
      else drawPoly(r.polygon ?? [], color);
    }

    // DSMC ライブ粒子表示 (prompts/66): 実行中の間引き粒子座標を灰白系の小さい点で散布描画する。
    // PICライブ粒子 (drawSpecies) と同じ体裁で見た目を揃える。メッシュ/領域の上・選択ハイライトの
    // 下に描く (領域を隠しすぎず、かつ選択ハンドルより手前に出ない位置)
    if (gasParticles) {
      drawSpecies(gasParticles, "#c8ccd4", 1.2);
    }

    // 選択中領域のハイライト (+ 選択ツール時は頂点/中点ハンドル、circle は半径ハンドル)
    if (selectedRegionId) {
      const sel = project.geometry.regions.find((r) => r.id === selectedRegionId);
      if (sel?.shape) {
        // 半径ドラッグ中はプレビュー半径を、そうでなければ実際の半径を表示する
        const rWorld = shapeRadiusPreview ?? sel.shape.radius;
        const [cx, cy] = sel.shape.center;
        ctx.strokeStyle = "#ffd24d";
        ctx.lineWidth = 3;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.arc(sx(cx), sy(cy), rWorld * view.scale, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        if (tool === "select") {
          // 半径ハンドル (円周上、角度0°): 頂点/中点ハンドルは circle には出さない
          const hs = 7;
          const hx = sx(cx + rWorld);
          const hy = sy(cy);
          ctx.fillStyle = "#ffd24d";
          ctx.strokeStyle = "#1b1e24";
          ctx.lineWidth = 1;
          ctx.fillRect(hx - hs / 2, hy - hs / 2, hs, hs);
          ctx.strokeRect(hx - hs / 2, hy - hs / 2, hs, hs);
        }
      } else if (sel) {
        // 頂点編集ドラッグ中はプレビュー多角形を、そうでなければ実際の多角形を表示する
        const poly = vertexPreviewPolygon ?? sel.polygon ?? [];
        ctx.strokeStyle = "#ffd24d";
        ctx.lineWidth = 3;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        poly.forEach(([x, y], i) =>
          i === 0 ? ctx.moveTo(sx(x), sy(y)) : ctx.lineTo(sx(x), sy(y)),
        );
        ctx.closePath();
        ctx.stroke();
        ctx.setLineDash([]);

        if (tool === "select") {
          // 頂点ハンドル (矩形)
          const hs = 7;
          ctx.fillStyle = "#ffd24d";
          ctx.strokeStyle = "#1b1e24";
          ctx.lineWidth = 1;
          for (const [x, y] of poly) {
            const px = sx(x);
            const py = sy(y);
            ctx.fillRect(px - hs / 2, py - hs / 2, hs, hs);
            ctx.strokeRect(px - hs / 2, py - hs / 2, hs, hs);
          }
          // 中点ハンドル (円)
          ctx.fillStyle = "#4da3ff";
          for (let i = 0; i < poly.length; i++) {
            const [ax, ay] = poly[i];
            const [bx, by] = poly[(i + 1) % poly.length];
            const mx = sx((ax + bx) / 2);
            const my = sy((ay + by) / 2);
            ctx.beginPath();
            ctx.arc(mx, my, 4, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }
    }

    // 図形移動ドラッグ中のプレビュー (半透明の破線輪郭を移動先に表示)
    if (movePreviewDelta && selectedRegionId) {
      const sel = project.geometry.regions.find((r) => r.id === selectedRegionId);
      if (sel) {
        const [dx, dy] = movePreviewDelta;
        ctx.strokeStyle = "#ffd24d";
        ctx.fillStyle = "rgba(255, 210, 77, 0.15)";
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        if (sel.shape) {
          const [cx, cy] = sel.shape.center;
          ctx.arc(sx(cx + dx), sy(cy + dy), sel.shape.radius * view.scale, 0, Math.PI * 2);
        } else {
          (sel.polygon ?? []).forEach(([x, y], i) => {
            const px = sx(x + dx);
            const py = sy(y + dy);
            i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
          });
          ctx.closePath();
        }
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // 作図中のプレビュー (ラバーバンド)
    if (tool === "polyline" && drawPts.length > 0) {
      ctx.strokeStyle = "#4da3ff";
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      drawPts.forEach(([x, y], i) =>
        i === 0 ? ctx.moveTo(sx(x), sy(y)) : ctx.lineTo(sx(x), sy(y)),
      );
      if (cursor) ctx.lineTo(sx(cursor[0]), sy(cursor[1]));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#4da3ff";
      for (const [x, y] of drawPts) {
        ctx.beginPath();
        ctx.arc(sx(x), sy(y), 3, 0, Math.PI * 2);
        ctx.fill();
      }
    } else if (tool === "rect" && drawPts.length === 1 && cursor) {
      const [x0r, y0r] = drawPts[0];
      const [x1r, y1r] = cursor;
      ctx.strokeStyle = "#4da3ff";
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(
        sx(Math.min(x0r, x1r)),
        sy(Math.max(y0r, y1r)),
        Math.abs(x1r - x0r) * view.scale,
        Math.abs(y1r - y0r) * view.scale,
      );
      ctx.setLineDash([]);
    } else if (tool === "circle" && drawPts.length === 1 && cursor) {
      const [cx, cy] = drawPts[0];
      const r = Math.hypot(cursor[0] - cx, cursor[1] - cy);
      ctx.strokeStyle = "#4da3ff";
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.arc(sx(cx), sy(cy), r * view.scale, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    } else if (tool === "profile" && drawPts.length === 1 && cursor) {
      const [x0p, y0p] = drawPts[0];
      const [x1p, y1p] = cursor;
      ctx.strokeStyle = "#4da3ff";
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(sx(x0p), sy(y0p));
      ctx.lineTo(sx(x1p), sy(y1p));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#4da3ff";
      ctx.beginPath();
      ctx.arc(sx(x0p), sy(y0p), 3, 0, Math.PI * 2);
      ctx.fill();
    } else if (tool === "emitter" && drawPts.length === 1 && cursor) {
      // エミッタ配置ツールのラバーバンド (緑系破線)
      const [x0e, y0e] = drawPts[0];
      const [x1e, y1e] = cursor;
      ctx.strokeStyle = "#4ddd8c";
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(sx(x0e), sy(y0e));
      ctx.lineTo(sx(x1e), sy(y1e));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#4ddd8c";
      ctx.beginPath();
      ctx.arc(sx(x0e), sy(y0e), 3, 0, Math.PI * 2);
      ctx.fill();
    } else if (tool === "collector" && drawPts.length === 1 && cursor) {
      // コレクタ配置ツールのラバーバンド (黄系破線、プロファイルと同じ2点クリックUX)
      const [x0c, y0c] = drawPts[0];
      const [x1c, y1c] = cursor;
      ctx.strokeStyle = "#ffd400";
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(sx(x0c), sy(y0c));
      ctx.lineTo(sx(x1c), sy(y1c));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#ffd400";
      ctx.beginPath();
      ctx.arc(sx(x0c), sy(y0c), 3, 0, Math.PI * 2);
      ctx.fill();
    } else if (tool === "gasbc" && drawPts.length === 1 && cursor) {
      // ガス境界配置ツールのラバーバンド (橙系破線、コレクタと同じ2点クリックUX)
      const [x0g, y0g] = drawPts[0];
      const [x1g, y1g] = cursor;
      ctx.strokeStyle = GAS_BOUNDARY_COLOR;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(sx(x0g), sy(y0g));
      ctx.lineTo(sx(x1g), sy(y1g));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = GAS_BOUNDARY_COLOR;
      ctx.beginPath();
      ctx.arc(sx(x0g), sy(y0g), 3, 0, Math.PI * 2);
      ctx.fill();
    } else if (tool === "eedfbox" && drawPts.length === 1 && cursor) {
      // EEDF/EEPF 領域配置ツールのラバーバンド (紫系破線、矩形ツールと同じ2点クリックUX)
      const [x0f, y0f] = drawPts[0];
      const [x1f, y1f] = cursor;
      ctx.strokeStyle = EEDF_REGION_COLOR;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(
        sx(Math.min(x0f, x1f)),
        sy(Math.max(y0f, y1f)),
        Math.abs(x1f - x0f) * view.scale,
        Math.abs(y1f - y0f) * view.scale,
      );
      ctx.setLineDash([]);
    } else if (tool === "meshref" && drawPts.length === 1 && cursor) {
      // 辺ローカルメッシュサイズ配置ツールのラバーバンド (水色系破線、コレクタと同じ2点クリックUX)
      const [x0m, y0m] = drawPts[0];
      const [x1m, y1m] = cursor;
      ctx.strokeStyle = EDGE_MESH_COLOR;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(sx(x0m), sy(y0m));
      ctx.lineTo(sx(x1m), sy(y1m));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = EDGE_MESH_COLOR;
      ctx.beginPath();
      ctx.arc(sx(x0m), sy(y0m), 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // 確定済みプロファイル線のオーバーレイ (白破線 + 端点マーカー)
    if (profileLine) {
      const [[xp0, yp0], [xp1, yp1]] = profileLine;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(sx(xp0), sy(yp0));
      ctx.lineTo(sx(xp1), sy(yp1));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#ffffff";
      for (const [px, py] of profileLine) {
        ctx.beginPath();
        ctx.arc(sx(px), sy(py), 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // 配置済み IEDF/IADF コレクタ線分のオーバーレイ (常時表示、コレクタごとに固定パレットを循環)。
    // 選択中 (一覧行クリック) のコレクタは白破線のハローで強調し、線分中点付近にラベルを描く
    collectors.forEach((c, i) => {
      const color = collectorColor(i);
      const isSelected = selectedCollectorIndex === i;
      const [xc0, yc0] = c.p1;
      const [xc1, yc1] = c.p2;
      ctx.strokeStyle = color;
      ctx.lineWidth = isSelected ? 5 : 3.5;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(sx(xc0), sy(yc0));
      ctx.lineTo(sx(xc1), sy(yc1));
      ctx.stroke();
      if (isSelected) {
        ctx.strokeStyle = "rgba(255,255,255,0.85)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(sx(xc0), sy(yc0));
        ctx.lineTo(sx(xc1), sy(yc1));
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.fillStyle = color;
      ctx.strokeStyle = "#1b1e24";
      ctx.lineWidth = 1;
      for (const [px, py] of [c.p1, c.p2]) {
        ctx.beginPath();
        ctx.arc(sx(px), sy(py), 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      // ラベル (線分中点のやや上に小さく描く)
      const mx = (xc0 + xc1) / 2;
      const my = (yc0 + yc1) / 2;
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillStyle = color;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(c.label, sx(mx), sy(my) - 6);
    });

    // 配置済み DSMC 線分境界のオーバーレイ (常時表示、橙系固定色。コレクタと混同しないよう
    // パレットを共有せず単色にする、prompts/72)
    gasBoundaries.forEach((g) => {
      const [xg0, yg0] = g.p1;
      const [xg1, yg1] = g.p2;
      ctx.strokeStyle = GAS_BOUNDARY_COLOR;
      ctx.lineWidth = 3.5;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(sx(xg0), sy(yg0));
      ctx.lineTo(sx(xg1), sy(yg1));
      ctx.stroke();
      ctx.fillStyle = GAS_BOUNDARY_COLOR;
      ctx.strokeStyle = "#1b1e24";
      ctx.lineWidth = 1;
      for (const [px, py] of [g.p1, g.p2]) {
        ctx.beginPath();
        ctx.arc(sx(px), sy(py), 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      const mx = (xg0 + xg1) / 2;
      const my = (yg0 + yg1) / 2;
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillStyle = GAS_BOUNDARY_COLOR;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(g.label, sx(mx), sy(my) - 6);
    });

    // 配置済み EEDF/EEPF 集計領域のオーバーレイ (常時表示、紫系固定色の破線矩形+ラベル、
    // 選択中は白破線ハローで強調。prompts/85)
    eedfRegions.forEach((r, i) => {
      const isSelected = selectedEedfIndex === i;
      const [xr0, yr0] = r.p1;
      const [xr1, yr1] = r.p2;
      const rx = sx(Math.min(xr0, xr1));
      const ry = sy(Math.max(yr0, yr1));
      const rw = Math.abs(xr1 - xr0) * view.scale;
      const rh = Math.abs(yr1 - yr0) * view.scale;
      ctx.strokeStyle = EEDF_REGION_COLOR;
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(rx, ry, rw, rh);
      if (isSelected) {
        ctx.strokeStyle = "rgba(255,255,255,0.85)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.strokeRect(rx, ry, rw, rh);
      }
      ctx.setLineDash([]);
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillStyle = EEDF_REGION_COLOR;
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(r.label, rx + 3, ry - 3);
    });

    // 配置済み辺ローカルメッシュサイズのオーバーレイ (常時表示、水色系破線+ラベル、prompts/90)
    edgeMeshSizes.forEach((m) => {
      const [xm0, ym0] = m.p1;
      const [xm1, ym1] = m.p2;
      ctx.strokeStyle = EDGE_MESH_COLOR;
      ctx.lineWidth = 2.5;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(sx(xm0), sy(ym0));
      ctx.lineTo(sx(xm1), sy(ym1));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = EDGE_MESH_COLOR;
      ctx.strokeStyle = "#1b1e24";
      ctx.lineWidth = 1;
      for (const [px, py] of [m.p1, m.p2]) {
        ctx.beginPath();
        ctx.arc(sx(px), sy(py), 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      const mx = (xm0 + xm1) / 2;
      const my = (ym0 + ym1) / 2;
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillStyle = EDGE_MESH_COLOR;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(m.label, sx(mx), sy(my) - 6);
    });

    // 粒子軌道 (trace 結果): シアン系半透明ポリライン。粒子数が多くても見えるように線幅は細く保つ
    if (showTrajectories && traceResult) {
      ctx.strokeStyle = "rgba(0,200,255,0.5)";
      ctx.lineWidth = 1;
      for (const traj of traceResult.trajectories) {
        if (traj.length < 2) continue;
        ctx.beginPath();
        traj.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(sx(x), sy(y)) : ctx.lineTo(sx(x), sy(y))));
        ctx.stroke();
      }
      // 吸収された粒子の最終位置 (着地点) に小さな点を描く
      ctx.fillStyle = "rgba(0,200,255,0.9)";
      traceResult.trajectories.forEach((traj, i) => {
        if (traceResult.status[i] !== "absorbed" || traj.length === 0) return;
        const [lx, ly] = traj[traj.length - 1];
        ctx.beginPath();
        ctx.arc(sx(lx), sy(ly), 2, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // エミッタのオーバーレイ: line は緑線分、point は×マーカー。中点/p1 から射出方向へ矢印を描く。
    // null は「エミッタを表示」オプションOFF (非表示)
    if (emitter) {
      ctx.strokeStyle = "#4ddd8c";
      ctx.fillStyle = "#4ddd8c";
      ctx.lineWidth = 2;
      let originX: number;
      let originY: number;
      if (emitter.kind === "line") {
        const [p1x, p1y] = emitter.p1;
        const [p2x, p2y] = emitter.p2;
        ctx.beginPath();
        ctx.moveTo(sx(p1x), sy(p1y));
        ctx.lineTo(sx(p2x), sy(p2y));
        ctx.stroke();
        originX = (p1x + p2x) / 2;
        originY = (p1y + p2y) / 2;
      } else {
        // point エミッタ: ×マーカー
        const [px0, py0] = emitter.p1;
        const hx = sx(px0);
        const hy = sy(py0);
        const hs = 6;
        ctx.beginPath();
        ctx.moveTo(hx - hs, hy - hs);
        ctx.lineTo(hx + hs, hy + hs);
        ctx.moveTo(hx + hs, hy - hs);
        ctx.lineTo(hx - hs, hy + hs);
        ctx.stroke();
        originX = px0;
        originY = py0;
      }
      // 射出方向の矢印 (画面固定長。画面はワールドに対し y が反転している点に注意)
      const angRad = (emitter.direction_deg * Math.PI) / 180;
      const arrowLen = 24;
      const ox = sx(originX);
      const oy = sy(originY);
      const adx = Math.cos(angRad) * arrowLen;
      const ady = -Math.sin(angRad) * arrowLen;
      const ex = ox + adx;
      const ey = oy + ady;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(ex, ey);
      ctx.stroke();
      const ang = Math.atan2(ady, adx);
      const headLen = 7;
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - headLen * Math.cos(ang - Math.PI / 6), ey - headLen * Math.sin(ang - Math.PI / 6));
      ctx.lineTo(ex - headLen * Math.cos(ang + Math.PI / 6), ey - headLen * Math.sin(ang + Math.PI / 6));
      ctx.closePath();
      ctx.fill();
    }

    // カラーバー (右下、画面固定・縦グラデーション)。PICライブ/結果フィールド表示中は
    // Solve 結果のバーは出さない (結果フィールド側のバーは picFieldView ブロックで描画済み)
    if (!picFieldView && !picFrame && result) {
      const unit = fieldView === "v" ? "V" : "V/m";
      const maxVal = fieldView === "v" ? result.v_max : result.e_abs_max;
      const minVal = fieldView === "v" ? result.v_min : 0;
      drawColorbar(minVal, maxVal, unit);
    }

    // ルーラー (常時表示のオーバーレイ。マウス座標系には影響しない)
    {
      const majorStep = step; // グリッドの自動ステップと揃える
      const minorStep = majorStep / 10;
      // ルーラー帯の幅/高さはフォントサイズに応じて広げる
      const RULER_SIZE = rulerSizeFor(rulerFontSize);

      ctx.fillStyle = "#242830";
      ctx.fillRect(0, 0, rect.width, RULER_SIZE);
      ctx.fillRect(0, 0, RULER_SIZE, rect.height);

      ctx.strokeStyle = "#363c48";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, RULER_SIZE + 0.5);
      ctx.lineTo(rect.width, RULER_SIZE + 0.5);
      ctx.moveTo(RULER_SIZE + 0.5, 0);
      ctx.lineTo(RULER_SIZE + 0.5, rect.height);
      ctx.stroke();

      ctx.font = `${rulerFontSize}px system-ui, sans-serif`;

      // 上辺: x 方向の目盛り (主目盛りに表示単位のラベル、1/10 間隔で副目盛り)
      const xStart = Math.floor(toWorld(RULER_SIZE, 0, view)[0] / minorStep) * minorStep;
      ctx.strokeStyle = "#8a919e";
      ctx.fillStyle = "#d8dce4";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.beginPath();
      let ix = Math.round(xStart / minorStep);
      for (let x = xStart; sx(x) <= rect.width + minorStep; x += minorStep, ix++) {
        const px = sx(x);
        if (px < RULER_SIZE) continue;
        const isMajor = ix % 10 === 0;
        const len = isMajor ? 9 : 4;
        ctx.moveTo(px, RULER_SIZE - len);
        ctx.lineTo(px, RULER_SIZE);
        if (isMajor) ctx.fillText(`${Math.round(mToUnit(x, lengthUnit))}`, px + 2, 1);
      }
      ctx.stroke();

      // 左辺: y 方向の目盛り
      const yStart = Math.floor(toWorld(0, rect.height, view)[1] / minorStep) * minorStep;
      ctx.beginPath();
      let iy = Math.round(yStart / minorStep);
      for (let y = yStart; sy(y) >= -minorStep; y += minorStep, iy++) {
        const py = sy(y);
        if (py < RULER_SIZE) continue;
        const isMajor = iy % 10 === 0;
        const len = isMajor ? 9 : 4;
        ctx.moveTo(RULER_SIZE - len, py);
        ctx.lineTo(RULER_SIZE, py);
        if (isMajor && py > RULER_SIZE + 8) {
          ctx.save();
          ctx.textAlign = "left";
          ctx.textBaseline = "middle";
          ctx.fillText(`${Math.round(mToUnit(y, lengthUnit))}`, 1, py);
          ctx.restore();
        }
      }
      ctx.stroke();

      // カーソル位置マーカー
      if (cursor) {
        const cx = sx(cursor[0]);
        const cy = sy(cursor[1]);
        ctx.strokeStyle = "#4da3ff";
        ctx.lineWidth = 1;
        ctx.beginPath();
        if (cx >= RULER_SIZE && cx <= rect.width) {
          ctx.moveTo(cx, 0);
          ctx.lineTo(cx, RULER_SIZE);
        }
        if (cy >= RULER_SIZE && cy <= rect.height) {
          ctx.moveTo(0, cy);
          ctx.lineTo(RULER_SIZE, cy);
        }
        ctx.stroke();
      }

      // 左上コーナー (塗りつぶし矩形)
      ctx.fillStyle = "#242830";
      ctx.fillRect(0, 0, RULER_SIZE, RULER_SIZE);
      ctx.strokeStyle = "#363c48";
      ctx.strokeRect(0.5, 0.5, RULER_SIZE - 1, RULER_SIZE - 1);

      // 軸対称モード: ルーラーの軸ラベルを表記する (上ルーラー=x方向, 左ルーラー=y方向)。
      // rz は x=z・y=r、rz_x0 は x=r・y=z
      if (isAxisym) {
        const xAxisLabel = isRzX0 ? "r" : "z";
        const yAxisLabel = isRzX0 ? "z" : "r";
        ctx.fillStyle = "#8a919e";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(xAxisLabel, RULER_SIZE / 2, RULER_SIZE / 2 - 5);
        ctx.fillText(yAxisLabel, RULER_SIZE / 2, RULER_SIZE / 2 + 7);
      }
    }
  }, [
    resizeTick,
    project,
    result,
    meshResult,
    meshResultIsLatest,
    showMesh,
    view,
    toWorld,
    tool,
    drawPts,
    cursor,
    selectedRegionId,
    fieldView,
    showIsolines,
    showVectors,
    movePreviewDelta,
    vertexPreviewPolygon,
    shapeRadiusPreview,
    profileLine,
    collectors,
    selectedCollectorIndex,
    eedfRegions,
    selectedEedfIndex,
    rulerFontSize,
    emitter,
    traceResult,
    showTrajectories,
    picFrame,
    picFieldView,
    gasParticles,
    gasBoundaries,
    edgeMeshSizes,
    isRz,
    isRzX0,
    isAxisym,
    lengthUnit,
  ]);

  // Space キーの追跡 (入力欄にフォーカス中は無視)
  useEffect(() => {
    const isEditable = (t: EventTarget | null) =>
      t instanceof HTMLElement && ["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName);
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && !isEditable(e.target)) spaceRef.current = true;
    };
    const up = (e: KeyboardEvent) => { if (e.code === "Space") spaceRef.current = false; };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }, []);

  const finishPolyline = (pts: Point[]) => {
    if (pts.length >= 3) onAddRegion(pts);
    setDrawPts([]);
  };

  return (
    <div className="canvas-wrap">
      <canvas
        ref={canvasRef}
        tabIndex={0}
        onWheel={(e) => {
          if (!view) return;
          const rect = e.currentTarget.getBoundingClientRect();
          const px = e.clientX - rect.left;
          const py = e.clientY - rect.top;
          const k = e.deltaY < 0 ? 1.15 : 1 / 1.15;
          setView({
            scale: view.scale * k,
            ox: px - (px - view.ox) * k,
            oy: py - (py - view.oy) * k,
          });
        }}
        onMouseDown={(e) => {
          e.currentTarget.focus();
          if (e.button === 1 || (e.button === 0 && spaceRef.current)) {
            // 既存のパン操作 (中ボタン or Space+左ドラッグ) を優先する
            dragRef.current = { x: e.clientX, y: e.clientY };
            didDragRef.current = false;
            mouseDownScreenRef.current = null; // パン中はクリック判定の対象外
            e.preventDefault();
            return;
          }
          // クリック/ドラッグ判定用に mousedown 時の画面座標を記録
          mouseDownScreenRef.current = { x: e.clientX, y: e.clientY };
          if (e.button !== 0 || !view || tool !== "select") return;

          const rect = e.currentTarget.getBoundingClientRect();
          const screenPt = { x: e.clientX - rect.left, y: e.clientY - rect.top };
          const sel = selectedRegionId
            ? project.geometry.regions.find((r) => r.id === selectedRegionId) ?? null
            : null;
          if (!sel) return;

          if (sel.shape) {
            // circle 領域: 半径ハンドル > 領域移動 (頂点/中点ハンドルは出さない)
            if (hitHandleRadius(sel.shape, screenPt, view, HANDLE_HIT_TOLERANCE_PX)) {
              radiusDragRef.current = { regionId: sel.id };
              setShapeRadiusPreview(sel.shape.radius);
              return;
            }
            const world = toWorld(screenPt.x, screenPt.y, view);
            const tolWorld = EDGE_HIT_TOLERANCE_PX / view.scale;
            if (hitCircle(world, sel.shape, tolWorld)) {
              moveDragRef.current = { id: sel.id, startWorld: world };
            }
            return;
          }

          const poly = sel.polygon ?? [];
          // ヒットテストの優先順位: 頂点ハンドル > 中点ハンドル > 領域移動
          const vIdx = findHandleVertex(poly, screenPt, view, HANDLE_HIT_TOLERANCE_PX);
          if (vIdx !== null) {
            vertexDragRef.current = { regionId: sel.id, index: vIdx };
            setVertexPreviewPolygon(poly.map(([x, y]) => [x, y] as Point));
            return;
          }
          const mIdx = findHandleMidpoint(poly, screenPt, view, HANDLE_HIT_TOLERANCE_PX);
          if (mIdx !== null) {
            const [ax, ay] = poly[mIdx];
            const [bx, by] = poly[(mIdx + 1) % poly.length];
            const mid: Point = [(ax + bx) / 2, (ay + by) / 2];
            const newPoly = [...poly.slice(0, mIdx + 1), mid, ...poly.slice(mIdx + 1)];
            vertexDragRef.current = { regionId: sel.id, index: mIdx + 1 };
            setVertexPreviewPolygon(newPoly);
            return;
          }
          const world = toWorld(screenPt.x, screenPt.y, view);
          const tolWorld = EDGE_HIT_TOLERANCE_PX / view.scale;
          if (hitRegion(world, poly, tolWorld)) {
            // 選択中領域の内部 (または輪郭付近) を掴んだので移動ドラッグを開始
            moveDragRef.current = { id: sel.id, startWorld: world };
          }
        }}
        onMouseMove={(e) => {
          if (!view) return;
          const rect = e.currentTarget.getBoundingClientRect();
          setCursor(getPoint(e.clientX - rect.left, e.clientY - rect.top, view));
          if (dragRef.current) {
            setView({
              ...view,
              ox: view.ox + e.clientX - dragRef.current.x,
              oy: view.oy + e.clientY - dragRef.current.y,
            });
            dragRef.current = { x: e.clientX, y: e.clientY };
            didDragRef.current = true;
          } else if (vertexDragRef.current) {
            const { index } = vertexDragRef.current;
            const pt = getPoint(e.clientX - rect.left, e.clientY - rect.top, view);
            setVertexPreviewPolygon((prev) => {
              if (!prev) return prev;
              const next = prev.slice();
              next[index] = pt;
              return next;
            });
          } else if (moveDragRef.current) {
            const world = toWorld(e.clientX - rect.left, e.clientY - rect.top, view);
            let dx = world[0] - moveDragRef.current.startWorld[0];
            let dy = world[1] - moveDragRef.current.startWorld[1];
            if (gridSnap) {
              const step = gridStep(view.scale) / 10;
              dx = Math.round(dx / step) * step;
              dy = Math.round(dy / step) * step;
            }
            setMovePreviewDelta([dx, dy]);
          } else if (radiusDragRef.current) {
            const sel = project.geometry.regions.find((r) => r.id === radiusDragRef.current!.regionId);
            if (sel?.shape) {
              const pt = getPoint(e.clientX - rect.left, e.clientY - rect.top, view);
              const r = Math.hypot(pt[0] - sel.shape.center[0], pt[1] - sel.shape.center[1]);
              setShapeRadiusPreview(Math.max(0, r));
            }
          }
        }}
        onMouseUp={(e) => {
          // パン (中ボタン / Space+左) が行われていたかどうか
          const wasPanning = dragRef.current !== null;
          dragRef.current = null;

          if (!view) {
            mouseDownScreenRef.current = null;
            moveDragRef.current = null;
            setMovePreviewDelta(null);
            vertexDragRef.current = null;
            setVertexPreviewPolygon(null);
            radiusDragRef.current = null;
            setShapeRadiusPreview(null);
            return;
          }

          const rect = e.currentTarget.getBoundingClientRect();
          const down = mouseDownScreenRef.current;
          const movedPx = down ? screenDist(down.x, down.y, e.clientX, e.clientY) : Infinity;
          const isClick = !wasPanning && movedPx < CLICK_MOVE_TOLERANCE_PX;
          mouseDownScreenRef.current = null;

          if (tool === "select") {
            if (isClick) {
              // mousedown 側で移動/頂点/中点ドラッグの開始判定に食われていても、
              // 画面上の移動量が閾値未満なら必ずクリック (選択処理) として扱う
              moveDragRef.current = null;
              setMovePreviewDelta(null);
              vertexDragRef.current = null;
              setVertexPreviewPolygon(null);
              radiusDragRef.current = null;
              setShapeRadiusPreview(null);
              const world = toWorld(e.clientX - rect.left, e.clientY - rect.top, view);
              const tolWorld = EDGE_HIT_TOLERANCE_PX / view.scale;
              const hit = findRegionAt(world, project.geometry.regions, tolWorld);
              onSelectRegion(hit ? hit.id : null);
            } else if (vertexDragRef.current) {
              const { regionId } = vertexDragRef.current;
              vertexDragRef.current = null;
              if (vertexPreviewPolygon) onEditRegionPolygon(regionId, vertexPreviewPolygon);
              setVertexPreviewPolygon(null);
            } else if (radiusDragRef.current) {
              const { regionId } = radiusDragRef.current;
              radiusDragRef.current = null;
              const sel = project.geometry.regions.find((r) => r.id === regionId);
              if (sel?.shape && shapeRadiusPreview !== null && shapeRadiusPreview > 0) {
                onEditRegionShape(regionId, { ...sel.shape, radius: shapeRadiusPreview });
              }
              setShapeRadiusPreview(null);
            } else if (moveDragRef.current) {
              const { id } = moveDragRef.current;
              moveDragRef.current = null;
              if (
                movePreviewDelta &&
                (Math.abs(movePreviewDelta[0]) > 1e-9 || Math.abs(movePreviewDelta[1]) > 1e-9)
              ) {
                onMoveRegion(id, movePreviewDelta[0], movePreviewDelta[1]);
              }
              setMovePreviewDelta(null);
            }
          }
        }}
        onMouseLeave={() => {
          dragRef.current = null;
          mouseDownScreenRef.current = null;
          setCursor(null);
          if (moveDragRef.current) {
            // キャンバス外に出た場合は移動をキャンセルする (Esc と同様)
            moveDragRef.current = null;
            setMovePreviewDelta(null);
          }
          if (vertexDragRef.current) {
            vertexDragRef.current = null;
            setVertexPreviewPolygon(null);
          }
          if (radiusDragRef.current) {
            radiusDragRef.current = null;
            setShapeRadiusPreview(null);
          }
        }}
        onClick={(e) => {
          if (!view) return;
          if (spaceRef.current || didDragRef.current) {
            didDragRef.current = false;
            return;
          }
          const rect = e.currentTarget.getBoundingClientRect();
          const pt = getPoint(e.clientX - rect.left, e.clientY - rect.top, view);

          // 選択ツールのクリックはヒット感度改善のため onMouseUp 側で処理する
          if (tool === "rect") {
            if (drawPts.length === 0) {
              setDrawPts([pt]);
            } else {
              const [x0, y0] = drawPts[0];
              const [x1, y1] = pt;
              const poly: Point[] = [
                [Math.min(x0, x1), Math.min(y0, y1)],
                [Math.max(x0, x1), Math.min(y0, y1)],
                [Math.max(x0, x1), Math.max(y0, y1)],
                [Math.min(x0, x1), Math.max(y0, y1)],
              ];
              onAddRegion(poly);
              setDrawPts([]);
            }
          } else if (tool === "circle") {
            if (drawPts.length === 0) {
              setDrawPts([pt]);
            } else {
              const [cx, cy] = drawPts[0];
              const r = Math.hypot(pt[0] - cx, pt[1] - cy);
              // 中心・半径 (グリッドスナップ済みの2点から算出) を circle shape として登録する
              if (r > 0) onAddRegion({ kind: "circle", center: [cx, cy], radius: r });
              setDrawPts([]);
            }
          } else if (tool === "polyline") {
            setDrawPts((prev) => [...prev, pt]);
          } else if (tool === "profile") {
            if (drawPts.length === 0) {
              setDrawPts([pt]);
            } else {
              const p1 = drawPts[0];
              onProfileLine(p1, pt);
              setDrawPts([]);
            }
          } else if (tool === "emitter") {
            if (drawPts.length === 0) {
              setDrawPts([pt]);
            } else {
              const p1 = drawPts[0];
              onSetEmitter(p1, pt);
              setDrawPts([]);
            }
          } else if (tool === "collector") {
            if (drawPts.length === 0) {
              setDrawPts([pt]);
            } else {
              const p1 = drawPts[0];
              onSetCollector(p1, pt);
              setDrawPts([]);
            }
          } else if (tool === "gasbc") {
            if (drawPts.length === 0) {
              setDrawPts([pt]);
            } else {
              const p1 = drawPts[0];
              onSetGasBoundary(p1, pt);
              setDrawPts([]);
            }
          } else if (tool === "eedfbox") {
            if (drawPts.length === 0) {
              setDrawPts([pt]);
            } else {
              const p1 = drawPts[0];
              onSetEedfRegion(p1, pt);
              setDrawPts([]);
            }
          } else if (tool === "meshref") {
            if (drawPts.length === 0) {
              setDrawPts([pt]);
            } else {
              const p1 = drawPts[0];
              onSetEdgeMeshSize(p1, pt);
              setDrawPts([]);
            }
          }
        }}
        onDoubleClick={(e) => {
          if (tool === "select" && view && selectedRegionId) {
            const rect = e.currentTarget.getBoundingClientRect();
            const screenPt = { x: e.clientX - rect.left, y: e.clientY - rect.top };
            const sel = project.geometry.regions.find((r) => r.id === selectedRegionId);
            if (sel?.polygon) {
              // 頂点ハンドルのダブルクリックで頂点を削除する (3点未満になる場合は無視)
              // (circle 領域には頂点ハンドルがないため、この操作は polygon 領域のみが対象)
              const vIdx = findHandleVertex(sel.polygon, screenPt, view, HANDLE_HIT_TOLERANCE_PX);
              if (vIdx !== null && sel.polygon.length > 3) {
                onEditRegionPolygon(sel.id, sel.polygon.filter((_, i) => i !== vIdx));
              }
              return;
            }
            if (sel) return;
          }
          if (tool !== "polyline") return;
          let pts = drawPts;
          // ダブルクリックの2回目の click で追加された重複頂点を除去
          if (pts.length >= 2) {
            const [x1, y1] = pts[pts.length - 1];
            const [x2, y2] = pts[pts.length - 2];
            if (Math.hypot(x1 - x2, y1 - y2) < 1e-9) pts = pts.slice(0, -1);
          }
          finishPolyline(pts);
        }}
        onKeyDown={(e) => {
          if (tool === "polyline" && e.key === "Enter") {
            e.preventDefault();
            finishPolyline(drawPts);
          } else if (e.key === "Escape") {
            setDrawPts([]);
            if (moveDragRef.current) {
              // ドラッグ中の移動をキャンセルする (プロジェクトへは反映しない)
              moveDragRef.current = null;
              setMovePreviewDelta(null);
            }
            if (vertexDragRef.current) {
              // ドラッグ中の頂点/中点編集をキャンセルする
              vertexDragRef.current = null;
              setVertexPreviewPolygon(null);
            }
            if (radiusDragRef.current) {
              // ドラッグ中の半径編集をキャンセルする
              radiusDragRef.current = null;
              setShapeRadiusPreview(null);
            }
          } else if (tool === "select" && e.key === "Delete" && selectedRegionId) {
            onDeleteRegion(selectedRegionId);
          } else if (
            tool === "select" &&
            view &&
            selectedRegionId &&
            (e.key === "ArrowUp" || e.key === "ArrowDown" || e.key === "ArrowLeft" || e.key === "ArrowRight")
          ) {
            // 矢印キーでの微動: スナップ幅ぶん移動 (1回の押下 = 1操作 = 1履歴)
            e.preventDefault();
            const step = gridStep(view.scale) / 10;
            let dx = 0;
            let dy = 0;
            if (e.key === "ArrowUp") dy = step;
            else if (e.key === "ArrowDown") dy = -step;
            else if (e.key === "ArrowLeft") dx = -step;
            else dx = step;
            onMoveRegion(selectedRegionId, dx, dy);
          }
        }}
      />
      {cursor && (
        <div className="coords">
          {isRzX0 ? "r" : isRz ? "z" : "x"}: {mToUnit(cursor[0], lengthUnit).toFixed(2)} {LENGTH_UNIT_LABEL[lengthUnit]}&nbsp;&nbsp;
          {isRzX0 ? "z" : isRz ? "r" : "y"}: {mToUnit(cursor[1], lengthUnit).toFixed(2)} {LENGTH_UNIT_LABEL[lengthUnit]}
        </div>
      )}
      {/* 境界条件の凡例 (常時表示。色/線種は bcStyle() と対応させる) */}
      <div className="bc-legend">
        <div className="bc-legend-item">
          <span className="bc-swatch bc-dirichlet" />Dirichlet
        </div>
        <div className="bc-legend-item">
          <span className="bc-swatch bc-symmetry" />対称
        </div>
        <div className="bc-legend-item">
          <span className="bc-swatch bc-periodic" />周期
        </div>
      </div>
    </div>
  );
}
