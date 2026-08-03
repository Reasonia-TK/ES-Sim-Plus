import { useCallback, useEffect, useMemo, useRef, useState } from "react";
// 実行時の実体は default エクスポートのオブジェクト (src/polygon-clipping.d.ts 参照)
import polygonClipping from "polygon-clipping";
import type { MultiPolygon } from "polygon-clipping";
import { api } from "./api";
import { getPort, initPort, setPort } from "./backendPort";
import CadCanvas from "./canvas/CadCanvas";
import type {
  EdgeMeshSizeView,
  FieldView,
  GasBoundaryView,
  PicCollectorView,
  PicEedfRegionView,
  PicFieldView,
  SheathDensitySource,
  SheathLineView,
  Tool,
} from "./canvas/CadCanvas";
import { sheathLineEdge } from "./sheath";
import { COLORMAPS, DEFAULT_COLORMAP } from "./canvas/colormaps";
import type { ColormapKey } from "./canvas/colormaps";
import { CommitNullableNumberInput } from "./CommitInput";
import Plot1dView from "./canvas/Plot1dView";
import ProfilePanel from "./panels/ProfilePanel";
import RfPhaseMonitor from "./panels/RfPhaseMonitor";
import FieldPanel, { EDGE_LABELS_RZ, EDGE_LABELS_RZ_X0, EDGE_LABELS_XY } from "./panels/FieldPanel";
import type { FieldSection } from "./panels/FieldPanel";
import ProjectTree from "./ProjectTree";
import type { TreeNode } from "./ProjectTree";
import ParticlePanel from "./panels/ParticlePanel";
import PicPanel, { PIC_FIELD_META } from "./panels/PicPanel";
import type { CyclePicField, PicLiveField, PicResultField } from "./panels/PicPanel";
import Pic1dPanel from "./panels/Pic1dPanel";
import Fluid1dPanel, { DEFAULT_FLUID1D } from "./panels/Fluid1dPanel";
import Fluid1dPlotView from "./canvas/Fluid1dPlotView";
import Fluid2dPanel, { DEFAULT_FLUID2D, FLUID2D_FIELD_META } from "./panels/Fluid2dPanel";
import type {
  Fluid2dCycleField,
  Fluid2dLiveField,
  Fluid2dResultField,
} from "./panels/Fluid2dPanel";
import TlPanel from "./panels/TlPanel";
import TlPlotView from "./canvas/TlPlotView";
import GasPanel, { DEFAULT_BOUNDARY, DEFAULT_DSMC, GAS_FIELD_META, gasFieldValues } from "./panels/GasPanel";
import type { GasResultField } from "./panels/GasPanel";
import SweepPanel from "./panels/SweepPanel";
import { Toggle } from "./Toggle";
import { PicClient } from "./picClient";
import type { PicClientCallbacks } from "./picClient";
import { Pic1dClient } from "./pic1dClient";
import type { Pic1dClientCallbacks } from "./pic1dClient";
import { Fluid1dClient } from "./fluid1dClient";
import type { Fluid1dClientCallbacks } from "./fluid1dClient";
import { Fluid2dClient } from "./fluid2dClient";
import type { Fluid2dClientCallbacks } from "./fluid2dClient";
import { TlClient } from "./tlClient";
import type { TlClientCallbacks } from "./tlClient";
import { DsmcClient } from "./dsmcClient";
import type { DsmcClientCallbacks } from "./dsmcClient";
import { SweepClient } from "./sweepClient";
import type { SweepClientCallbacks } from "./sweepClient";
import { useHistory } from "./useHistory";
import { saveTextFile } from "./saveFile";
import { isAxisymmetric, toDiagArray } from "./types";
import { LENGTH_UNIT_LABEL } from "./units";
import type { LengthUnit } from "./units";
import type {
  BField,
  BoundaryCondition,
  CircleShape,
  DsmcBoundary,
  DsmcResult,
  EdgeBcType,
  EdgeMeshSize,
  Fluid1dFrameMsg,
  Fluid1dResult,
  Fluid1dSettings,
  Fluid1dStartedMsg,
  Fluid2dCycle,
  Fluid2dFrameMsg,
  Fluid2dResult,
  Fluid2dSettings,
  Fluid2dStartedMsg,
  Health,
  MeshResult,
  ParticleSettings,
  Pic1dElectrode,
  Pic1dFrameMsg,
  Pic1dResult,
  Pic1dSettings,
  Pic1dStartedMsg,
  PicCollectorResult,
  PicCollectorSettings,
  PicCycle,
  PicDiag,
  PicEedfRegionSettings,
  PicEedfResult,
  PicFields,
  PicFrameMsg,
  PicLiveFrame,
  PicSettings,
  PicStartedMsg,
  Point,
  Project,
  Region,
  RegionType,
  ResultsBundle,
  SheathLineSettings,
  SolveResult,
  SweepCaseState,
  SweepStartedMsg,
  TlResult,
  TlSettings,
  TlStartedMsg,
  TraceResult,
  VoltageRf,
  VoltageWaveform,
} from "./types";

// 粒子パネルの既定値 (project.particles が未設定の場合の初期表示に使う)
const DEFAULT_PARTICLES: ParticleSettings = {
  species: { preset: "electron" },
  emitter: {
    kind: "line",
    p1: [0.02, 0.02],
    p2: [0.02, 0.03],
    n: 50,
    energy_ev: 1.0,
    direction_deg: 0,
    spread_deg: 0,
  },
  dt: null,
  n_steps: 2000,
  save_every: 10,
};

// PIC設定の既定値 (project.pic が未設定の場合の初期表示に使う)
const DEFAULT_PIC: PicSettings = {
  initial_plasma: null,
  injection: null,
  n_macro: 20000,
  dt: null,
  n_steps: 2000,
  frame_every: 20,
  mcc: null,
  see_energy_ev: 2.0,
  avg_steps: null,
  phase_bins: 40,
  threads: 0,
};

// pic.injection.emitter は常にフェーズ2 (粒子) パネルの現在のエミッタ設定で上書きしてから
// 保存/送信する (PicPanel 側では編集用の複製を持たず、都度ここで同期する)
function withInjectionEmitter(pic: PicSettings, emitter: ParticleSettings["emitter"]): PicSettings {
  if (!pic.injection) return pic;
  return { ...pic, injection: { ...pic.injection, emitter } };
}

// 1D PIC/MCC 設定の既定値 (project.pic1d が未設定の場合の初期表示に使う、prompts/91)。
// pic (2D) と同様に particles/pic とは独立の state で管理し (Undo/Redo 対象外)、
// geometry/mesh には一切依存しない (pic1d.py は専用の一様格子ソルバー)
const DEFAULT_PIC1D: Pic1dSettings = {
  gap_m: 0.02,
  n_cells: 128,
  left: { v_dc: 0.0, waveforms: [], see_gamma: 0.0 },
  right: { v_dc: 0.0, waveforms: [], see_gamma: 0.0 },
  init_density_m3: 1.0e14,
  init_te_ev: 2.0,
  init_ti_ev: 0.03,
  ion_mass_amu: 39.948,
  n_macro: 20000,
  dt: null,
  n_steps: 2000,
  frame_every: 20,
  avg_steps: null,
  phase_bins: 40,
  mcc: null,
  see_energy_ev: 2.0,
  eedf_regions: [],
  wall_iedf_bins: 100,
  seed: 0,
};

// VHF 定在波 (非線形径方向伝送線路モデル) 設定の既定値 (project 本体には持たず、pic1d と同じく
// 独立 state で管理する。geometry/mesh とは無関係な専用ソルバー、prompts/101)
const DEFAULT_TL: TlSettings = {
  radius_m: 0.15,
  gap_m: 0.04,
  sheath_m: 5e-4,
  n_e_m3: 1e16,
  n_s_ratio: 0.4,
  nu_m_hz: 1e8,
  freq_hz: 100e6,
  v0: 100.0,
  n_r: 400,
  n_periods: 200,
  n_fft_periods: 32,
  n_harm: 10,
  dt: null,
  sheath_law: "child",
};

// コレクタ追加数の上限 (バックエンドの validator と同じ、prompts/36/37)
const MAX_COLLECTORS = 8;

// EEDF/EEPF 領域追加数の上限 (バックエンドの validator と同じ、prompts/85)
const MAX_EEDF_REGIONS = 4;

// シースエッジ評価ライン追加数の上限 (バックエンドの validator と同じ、prompts/98)
const MAX_SHEATH_LINES = 4;

// 長さ表示単位の localStorage キー。プロジェクトファイルには含めない (表示設定のみ)
const LENGTH_UNIT_STORAGE_KEY = "es-sim-length-unit";

// 次のコレクタラベルを生成する ("C1", "C2", ...)。既存ラベルが "C<数字>" 形式の場合のみ
// 番号を拾い、その最大値+1を使う (欠番があっても詰めない。カスタムラベルは無視する)
function nextCollectorLabel(collectors: PicCollectorSettings[]): string {
  let maxN = 0;
  for (const c of collectors) {
    const m = /^C(\d+)$/.exec(c.label ?? "");
    if (m) maxN = Math.max(maxN, parseInt(m[1], 10));
  }
  return `C${maxN + 1}`;
}

// 次の EEDF 領域ラベルを生成する ("E1", "E2", ...)。nextCollectorLabel と同じ流儀
// (欠番があっても詰めない。カスタムラベルは無視する)
function nextEedfLabel(regions: PicEedfRegionSettings[]): string {
  let maxN = 0;
  for (const r of regions) {
    const m = /^E(\d+)$/.exec(r.label ?? "");
    if (m) maxN = Math.max(maxN, parseInt(m[1], 10));
  }
  return `E${maxN + 1}`;
}

// 次のシースエッジ評価ラインラベルを生成する ("S1", "S2", ...)。nextCollectorLabel と
// 同じ流儀 (欠番があっても詰めない。カスタムラベルは無視する)
function nextSheathLabel(lines: SheathLineSettings[]): string {
  let maxN = 0;
  for (const l of lines) {
    const m = /^S(\d+)$/.exec(l.label ?? "");
    if (m) maxN = Math.max(maxN, parseInt(m[1], 10));
  }
  return `S${maxN + 1}`;
}

// 領域マージ (prompts/63) の union 結果から共線頂点を除去する。
// 隣接する矩形同士を union すると、共有辺上に不要な中間点 (角ではない点) が
// 残ることがあるため、外積がほぼ0 (3点が一直線) の頂点を取り除いて素直な多角形に戻す。
// また polygon-clipping はリングの先頭点を末尾に複製して閉じた形で返すため、
// まずその重複する閉じ点を取り除いてから判定する。
function removeCollinearVertices(ring: Point[]): Point[] {
  let pts = ring.slice();
  if (pts.length > 1) {
    const [fx, fy] = pts[0];
    const [lx, ly] = pts[pts.length - 1];
    if (fx === lx && fy === ly) pts = pts.slice(0, -1);
  }
  const n = pts.length;
  if (n < 3) return pts;
  // 座標のスケールに応じた閾値にする (絶対値1e-12だと大きい座標系で共線判定が緩すぎるため)
  let maxAbs = 0;
  for (const [x, y] of pts) maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y));
  const scale = Math.max(maxAbs, 1);
  const eps = 1e-12 * scale * scale;
  const cross = (o: Point, a: Point, b: Point) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const result: Point[] = [];
  for (let i = 0; i < n; i++) {
    const prev = pts[(i - 1 + n) % n];
    const cur = pts[i];
    const next = pts[(i + 1) % n];
    if (Math.abs(cross(prev, cur, next)) > eps) result.push(cur);
  }
  // 万一すべて共線判定される (退化ケース) 場合は元の頂点列を返して安全側に倒す
  return result.length >= 3 ? result : pts;
}

// 旧形式 (単数 collector) を持つ pic 設定を collectors 配列へ正規化する (後方互換、prompts/37)
function normalizeCollectors(pic: PicSettings): PicSettings {
  if (pic.collector && (!pic.collectors || pic.collectors.length === 0)) {
    const label = pic.collector.label && pic.collector.label.trim() !== "" ? pic.collector.label : "C1";
    return { ...pic, collectors: [{ ...pic.collector, label }], collector: undefined };
  }
  return pic;
}

// インスペクタ (中カラム) 上部に表示する現在ノードのタイトル。boundary/regions は
// 選択中の辺/領域に応じて別途組み立てる (下記 inspectorTitle 参照) ため、ここでは既定値のみ持つ
const NODE_TITLES: Record<TreeNode, string> = {
  domain: "ジオメトリ — ドメイン",
  regions: "ジオメトリ — 領域",
  boundary: "ジオメトリ — 境界条件",
  mesh: "ジオメトリ — メッシュ",
  bfield: "ジオメトリ — 磁場",
  "study-fem": "スタディ — 静電場",
  "study-trace": "スタディ — 粒子追跡",
  "study-pic": "スタディ — PIC-MCC",
  "study-pic1d": "スタディ — PIC-MCC 1D",
  "study-fluid1d": "スタディ — 流体1D",
  "study-fluid2d": "スタディ — 流体2D",
  "study-tl": "スタディ — VHF定在波",
  "study-gas": "スタディ — DSMC",
  "study-sweep": "スタディ — パラメータスイープ",
  "result-fem": "結果 — 静電場",
  "result-trace": "結果 — 粒子追跡",
  "result-pic": "結果 — PIC-MCC",
  "result-pic1d": "結果 — PIC-MCC 1D",
  "result-fluid1d": "結果 — 流体1D",
  "result-fluid2d": "結果 — 流体2D",
  "result-tl": "結果 — VHF定在波",
  "result-gas": "結果 — DSMC",
};

// ツールバーの現在ツールをステータスバーに表示するための日本語ラベル
const TOOL_LABELS: Record<Tool, string> = {
  select: "選択",
  polyline: "ポリライン",
  rect: "矩形",
  circle: "円",
  profile: "プロファイル",
  emitter: "エミッタ",
  collector: "コレクタ",
  gasbc: "ガス境界",
  eedfbox: "EEDF領域",
  meshref: "メッシュ細分",
  sheathline: "シース評価線",
  probe: "プローブ",
};

// 実行中の経過時間表示 (ステータスバー、prompts/86) の秒数を m:ss (1時間以上は h:mm:ss) に整形する
function formatElapsed(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const mm = h > 0 ? String(m).padStart(2, "0") : String(m);
  const ss = String(sec).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

// 電極ラベル ("edge0".."edge3" は FieldPanel の EDGE_LABELS_* で辺名に変換、
// conductor の region id はそのまま表示する)
function electrodeDisplayLabel(label: string, coord: Project["coord"]): string {
  const m = /^edge(\d+)$/.exec(label);
  if (!m) return label;
  const edgeLabels =
    coord === "rz" ? EDGE_LABELS_RZ : coord === "rz_x0" ? EDGE_LABELS_RZ_X0 : EDGE_LABELS_XY;
  const i = Number(m[1]);
  return edgeLabels[i] ?? label;
}

// DSMC「続きから実行」の無効化判定用キー (prompts/74)。project.dsmc.n_steps/avg_steps/
// threads/smoothing_passes は prepare_continue が対応しており、変更しても保持中の粒子状態
// (サーバー側) と非互換にならない。それ以外 (ジオメトリ・メッシュ・座標系・ガス種・境界条件・
// 初期条件・目標粒子数・dt・シード等) が変わった場合はサーバー側の保持状態と食い違うため、
// 「続きから実行」を無効化する必要がある。除外フィールドを落として JSON 文字列化するだけの
// 簡潔な比較で十分 (深い等価判定ライブラリは不要)
function dsmcContinueRelevantKey(p: Project): string {
  if (!p.dsmc) return JSON.stringify({ ...p, dsmc: null });
  const { n_steps: _n_steps, avg_steps: _avg_steps, threads: _threads, smoothing_passes: _smoothing_passes, ...rest } =
    p.dsmc;
  return JSON.stringify({ ...p, dsmc: rest });
}

// 「静電場結果」インスペクタページの解析結果サマリ (FieldPanel の solve
// セクションと同内容だが、結果ノード単体でも確認できるようにここでも表示する)
function ResultSummary({
  result,
  coord,
  elapsedS,
}: {
  result: SolveResult | null;
  coord: Project["coord"];
  // Solve 実行の計算時間 [s] (App 側で api 呼び出し前後の時刻差を計測、prompts/86)。
  // backend 変更なしのフロント側計測のため、結果付き保存には同梱しない (再実行が数秒スケールで容易なため)
  elapsedS: number | null;
}) {
  if (!result) {
    return <div className="muted">(まだ解析結果がありません。スタディ「静電場」で Solve を実行してください)</div>;
  }
  const isAxisym = isAxisymmetric(coord);
  const capUnit = isAxisym ? "F" : "F/m";
  const qUnit = isAxisym ? "C" : "C/m";
  return (
    <>
      {elapsedS != null && (
        <div className="kv"><span>計算時間</span><span>{elapsedS.toFixed(3)} s</span></div>
      )}
      <div className="kv"><span>節点数</span><span>{result.mesh.nodes.length}</span></div>
      <div className="kv"><span>要素数</span><span>{result.mesh.triangles.length}</span></div>
      <div className="kv"><span>V min/max</span><span>{result.v_min.toFixed(1)} / {result.v_max.toFixed(1)} V</span></div>
      <div className="kv"><span>|E| max</span><span>{result.e_abs_max.toExponential(2)} V/m</span></div>
      <div className="kv"><span>エネルギー</span><span>{result.energy.toExponential(3)} {isAxisym ? "J" : "J/m"}</span></div>
      <div className="kv">
        <span>静電容量</span>
        <span>
          {result.capacitance != null
            ? `${result.capacitance.toExponential(3)} ${capUnit}`
            : "- (2電極系のみ)"}
        </span>
      </div>
      {result.charges && result.charges.length > 0 && (
        <>
          <div className="muted">電極電荷</div>
          {result.charges.map((c) => (
            <div className="kv" key={c.label}>
              <span>{electrodeDisplayLabel(c.label, coord)} ({c.voltage}V)</span>
              <span>{c.q.toExponential(3)} {qUnit}</span>
            </div>
          ))}
        </>
      )}
    </>
  );
}

// フェーズ0 のサンプル (examples/parallel_plates.json と同内容)。
// これを初期値として、以降は project state を編集していく。
const SAMPLE: Project = {
  version: 1,
  unit: "m",
  geometry: {
    domain: { polygon: [[0, 0], [0.1, 0], [0.1, 0.05], [0, 0.05]] },
    regions: [
      {
        id: "diel1",
        type: "dielectric",
        polygon: [[0.04, 0.01], [0.06, 0.01], [0.06, 0.04], [0.04, 0.04]],
        eps_r: 4.0,
      },
    ],
    boundaries: [
      { edges: [3], type: "dirichlet", voltage: 0.0 },
      { edges: [1], type: "dirichlet", voltage: 100.0 },
    ],
  },
  mesh: { size: 0.004 },
  solver: { backend: "numpy" },
};

export default function App() {
  const [project, setProjectState] = useState<Project>(SAMPLE);
  // project state の最新値を同期的に参照するための ref。
  // イベントハンドラ内で複数回連続して編集操作が呼ばれても
  // (例: 矢印キーの連続入力) 常に最新の状態を土台にできるようにする。
  const projectRef = useRef<Project>(SAMPLE);
  const history = useHistory<Project>();

  const [health, setHealth] = useState<Health | null>(null);
  // バックエンドポート番号 (入力欄の表示用文字列)。確定 (blur/Enter) で setPort() へ反映する
  const [portInput, setPortInput] = useState<string>(String(getPort()));
  // initPort() 完了フラグ。完了前は health チェックを開始しない (起動直後に誤ったポートへ
  // アクセスして「backend 未接続」を一瞬表示するのを避けるため)
  const [portReady, setPortReady] = useState(false);
  const [portError, setPortError] = useState<string | null>(null);
  const [result, setResult] = useState<SolveResult | null>(null);
  // Solve 実行の計算時間 [s] (App 側で api 呼び出し前後の時刻差を計測、backend 変更不要、prompts/86)。
  // result と一緒にリセットする (再現が数秒スケールで容易なため結果付き保存へは同梱しない)
  const [solveElapsedS, setSolveElapsedS] = useState<number | null>(null);
  // Mesh ボタン (解析なしでメッシュ生成のみ) の結果。Solve 結果とは独立に保持する
  const [meshResult, setMeshResult] = useState<MeshResult | null>(null);
  // meshResult が Solve 結果 (result) / PIC ライブ表示 (picFrame) より「後に生成された」か
  // どうか (prompts/89 ②)。result/picFrame はジオメトリ変更等でしかクリアされず Mesh 実行
  // 単体では消えないため、これが無いと「一度計算した後は再メッシュしてもキャンバスの
  // メッシュプレビューが更新されない」不具合になる。Mesh 実行で true、Solve 完了・PIC
  // 開始 (start/continue) で false に戻す (CadCanvas の meshResultIsLatest 参照)
  const [meshPreviewFresh, setMeshPreviewFresh] = useState(false);
  const [showMesh, setShowMesh] = useState(false);
  // キャンバスオーバーレイの表示切替 (ツールバーの表示トグル群)。コレクタ・ガス境界は
  // 配置済みでも常時表示だと混み合うため、個別に消せるようにする (既定は表示)
  const [showCollectors, setShowCollectors] = useState(true);
  const [showGasBoundaries, setShowGasBoundaries] = useState(true);
  const [showEedfRegions, setShowEedfRegions] = useState(true);
  const [showEdgeMeshSizes, setShowEdgeMeshSizes] = useState(true);
  // シースエッジ (準中性度等値線+評価ラインマーカー) の表示トグル。既定オン (prompts/98)
  const [showSheathEdge, setShowSheathEdge] = useState(true);
  // 準中性度の閾値 α (n_e/n_i=α の等値線・評価ラインどちらにも使う)。α スライダの即時
  // 反映のため project へは保存しない表示専用 state (PicPanel の結果セクションで編集)
  const [sheathAlpha, setSheathAlpha] = useState(0.5);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 実行経過時間のリアルタイム表示 (ステータスバー、prompts/86)。各実行開始時刻を ref に
  // 記録しておき、実行中のみ 1秒間隔の setInterval で tick state を更新して再レンダーする
  // (アイドル時の無駄な再レンダーを避けるため、何か実行中のときだけ setInterval を張る)。
  // 複数同時実行 (PIC/DSMC/スイープ/busy) はそれぞれ個別の開始時刻を持つ
  const busyStartTimeRef = useRef<number | null>(null);
  const picStartTimeRef = useRef<number | null>(null);
  const pic1dStartTimeRef = useRef<number | null>(null);
  const fluid1dStartTimeRef = useRef<number | null>(null);
  const fluid2dStartTimeRef = useRef<number | null>(null);
  const tlStartTimeRef = useRef<number | null>(null);
  const gasStartTimeRef = useRef<number | null>(null);
  const sweepStartTimeRef = useRef<number | null>(null);
  const [, setElapsedTick] = useState(0);

  const [tool, setTool] = useState<Tool>("select");
  // 長さ表示単位 (mm/µm)。内部データ (project) は m のままで、表示・入力の解釈のみが変わる。
  // FN電界放出の検証など µm スケールの形状を扱う場合に切替える (ユーザー要望、prompts/62)
  const [lengthUnit, setLengthUnit] = useState<LengthUnit>(() => {
    const saved = localStorage.getItem(LENGTH_UNIT_STORAGE_KEY);
    return saved === "mm" || saved === "um" ? saved : "mm";
  });
  useEffect(() => {
    localStorage.setItem(LENGTH_UNIT_STORAGE_KEY, lengthUnit);
  }, [lengthUnit]);
  const [gridSnap, setGridSnap] = useState(true);
  // ルーラー目盛りラベルのフォントサイズ (px)。プロジェクトファイルには保存しない表示設定
  const [rulerFontSize, setRulerFontSize] = useState(11);
  // サイドパネル幅 (px)。リサイザのドラッグで変更する表示設定 (保存対象外)
  const [sideWidth, setSideWidth] = useState(280);
  const [fieldView, setFieldView] = useState<FieldView>("v");
  const [showIsolines, setShowIsolines] = useState(false);
  const [showVectors, setShowVectors] = useState(false);
  // コンター表示のカラーマップ選択 (キャンバスツールバー「配色」select、prompts/103)。
  // project へは保存しない表示専用 state (Undo/Redo 対象外)
  const [colormapKey, setColormapKey] = useState<ColormapKey>(DEFAULT_COLORMAP);
  // カラーバーの手動レンジ (min/max、null=自動=従来挙動)。project へは保存しない (prompts/103)
  const [colorRange, setColorRange] = useState<{ min: number | null; max: number | null }>({
    min: null,
    max: null,
  });
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [profileLine, setProfileLine] = useState<[Point, Point] | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // プロジェクトツリー (左カラム) で選択中のノード。中カラムのインスペクタは
  // このノードに応じて表示ページを切替える (表示切替のみで各ページはアンマウントしない。
  // PIC の WS 接続やチャート履歴など、各ページの実行状態を保持するため)
  const [activeNode, setActiveNode] = useState<TreeNode>("domain");
  // "boundary" ノード配下で辺の子ノードを選択した場合のみ非null (該当エッジのみ表示するフィルタ)
  const [edgeFilter, setEdgeFilter] = useState<number | null>(null);

  // 粒子設定 (エミッタ・積分パラメータ)。ジオメトリ編集とは独立に管理し、
  // 既存の Undo/Redo 履歴 (history) には積まない。保存/読込 (project.particles) の対象ではある
  const [particles, setParticles] = useState<ParticleSettings>(DEFAULT_PARTICLES);
  const [traceResult, setTraceResult] = useState<TraceResult | null>(null);
  // トレース実行の計算時間 [s] (solveElapsedS と同じ考え方、prompts/86)。traceResult と一緒にリセットする
  const [traceElapsedS, setTraceElapsedS] = useState<number | null>(null);
  const [showTrajectories, setShowTrajectories] = useState(true);
  // エミッタのオーバーレイ (緑の線分/×マーカー+矢印) をキャンバスに描くか。
  // FN 電界放出のみのケース等でエミッタ表示が邪魔なときに消せるようにする
  const [showEmitter, setShowEmitter] = useState(true);
  // 「結果 — 粒子追跡」ページの背景表示 (電位V/|E|/なし)。軌道だけを見たいときに
  // 静電場の色マップを消せるようにする (result-trace ノード選択中のみ効く)
  const [traceBackground, setTraceBackground] = useState<"v" | "e_abs" | "none">("v");

  // PIC設定 (particles と同様、Undo/Redo履歴には積まない。保存/読込 (project.pic) の対象ではある)
  const [pic, setPic] = useState<PicSettings>(DEFAULT_PIC);
  const [picRunning, setPicRunning] = useState(false);
  const [picStarted, setPicStarted] = useState<PicStartedMsg | null>(null);
  const [picFrame, setPicFrame] = useState<PicFrameMsg | null>(null);
  const [picHistory, setPicHistory] = useState<PicDiag[]>([]);
  const [picError, setPicError] = useState<string | null>(null);
  // done メッセージで受け取った時間平均フィールド一式。新規実行開始時にリセットする
  const [picFields, setPicFields] = useState<PicFields | null>(null);
  // done メッセージで受け取った位相別プロファイル計測 (prompts/75)。continue では
  // 区間分のみに置き換わる (累積ではない)。新規実行開始時にリセットする
  const [picTiming, setPicTiming] = useState<Record<string, number> | null>(null);
  // done メッセージで受け取った run_batch の壁時計秒 (prompts/86)。picTiming と同じ流儀
  // (continue では区間分のみに置き換わる。新規実行開始時にリセットする)
  const [picElapsedS, setPicElapsedS] = useState<number | null>(null);
  // 「結果表示」セレクトの選択と対数スケールチェックボックス。新規実行開始時に既定 (ライブ/線形) へ戻す
  const [picResultField, setPicResultField] = useState<PicResultField>("live");
  const [picLogScale, setPicLogScale] = useState(false);
  // ライブモニタの表示フィールド (電位/電子密度/イオン密度) と対数スケール (prompts/81)。
  // 実行中でも即座に切り替えられるよう独立の state として持つ (新規実行開始時のリセットは不要:
  // ユーザーが選んだ表示方法は次の実行にも引き継いで良い)
  const [picLiveField, setPicLiveField] = useState<PicLiveField>("phi");
  const [picLiveLogScale, setPicLiveLogScale] = useState(false);
  // RF位相モニタ (prompts/82) の表示トグル。既定で表示し、ユーザーが不要なら消せるようにする
  const [showRfMonitor, setShowRfMonitor] = useState(true);
  const picClientRef = useRef<PicClient | null>(null);
  // 「続きから実行」ボタンの有効条件その1: 直前の実行が done (または stop) 済みで、
  // 現在実行中でないこと。start/continue 開始時に false、done 受信時に true にする
  const [picContinueReady, setPicContinueReady] = useState(false);
  // 「続きから実行」ボタンの有効条件その2: 前回の PIC 実行以降にジオメトリが編集 (commitProject /
  // Undo / Redo) されていないこと。サーバー側が保持する状態 (メッシュ・境界条件等) と食い違うため、
  // 編集されたら続き実行を無効化する。新しい start を送るとサーバー状態も同期し直されるので false に戻す
  const [picProjectChangedSinceRun, setPicProjectChangedSinceRun] = useState(false);

  // done メッセージで受け取った IEDF/IADF コレクタの記録結果 (pic.collectors と同順)。
  // 新規実行開始時にリセットする
  const [picCollectors, setPicCollectors] = useState<PicCollectorResult[]>([]);
  // コレクタ一覧 (PICパネル) で選択中のインデックス。範囲外になった場合は下の
  // selectedCollectorIndex (派生値) で自動的に補正する
  const [selectedCollectorIndexRaw, setSelectedCollectorIndexRaw] = useState<number | null>(null);

  // done メッセージで受け取った EEDF/EEPF 領域の集計結果 (pic.eedf_regions と同順、prompts/85)。
  // 新規実行開始時にリセットする
  const [picEedf, setPicEedf] = useState<PicEedfResult[]>([]);
  // 領域一覧 (PICパネル) で選択中のインデックス (collector と同じ流儀)
  const [selectedEedfIndexRaw, setSelectedEedfIndexRaw] = useState<number | null>(null);

  // RF 1周期の位相分解データ (done で受信、RFなし/phase_bins=0 では null)。
  // 周期アニメーションプレイヤー (PicPanel) の状態一式も新規実行開始時にリセットする
  const [picCycle, setPicCycle] = useState<PicCycle | null>(null);
  const [cycleField, setCycleField] = useState<CyclePicField>("phi");
  const [cycleLogScale, setCycleLogScale] = useState(false);
  const [cyclePlaying, setCyclePlaying] = useState(false);
  const [cycleBinIndex, setCycleBinIndex] = useState(0);
  const [cycleFps, setCycleFps] = useState(10);
  const [cycleShowParticles, setCycleShowParticles] = useState(true);
  // 周期アニメーションが「結果表示より優先して描画される」のは、ユーザーがアニメを
  // 操作している間だけにする (不具合修正: cycle データが存在するだけで常時優先されると、
  // 結果表示セレクトを切り替えてもキャンバスが切り替わらない)
  const [cycleViewActive, setCycleViewActive] = useState(false);

  // 1D PIC/MCC 設定 (prompts/91)。pic (2D) と同様 Undo/Redo 履歴には積まない独立 state。
  // 2D の pic とは完全に独立した実行状態・結果を持つ (サーバー側のロック・保持スロットも別)
  const [pic1d, setPic1d] = useState<Pic1dSettings>(DEFAULT_PIC1D);
  const [pic1dRunning, setPic1dRunning] = useState(false);
  const [pic1dStarted, setPic1dStarted] = useState<Pic1dStartedMsg | null>(null);
  const [pic1dFrame, setPic1dFrame] = useState<Pic1dFrameMsg | null>(null);
  // done メッセージで受け取った結果一式 (settings を含み自己完結、結果付き保存にそのまま使える)
  const [pic1dResult, setPic1dResult] = useState<Pic1dResult | null>(null);
  const [pic1dError, setPic1dError] = useState<string | null>(null);
  const pic1dClientRef = useRef<Pic1dClient | null>(null);
  // 「続きから」ボタンの有効条件: 直前の実行が done/stop 済みで現在実行中でないこと。
  // pic1d は geometry/mesh に依存しないため、2D の picProjectChangedSinceRun に相当する
  // 「食い違いで無効化する」概念は無い (pic1d 設定を変えても常にサーバー保持状態へ継続実行するだけ)
  const [pic1dContinueReady, setPic1dContinueReady] = useState(false);

  // 1D プラズマ流体 (ドリフト拡散 + 電子エネルギー、prompts/104-109)。pic1d (Pic1dSettings) と
  // 同一条件で比較できるよう設計された独立ソルバーで、実行状態・結果も pic1d とは完全に別
  // (サーバー側のロック・保持スロットも別、fluid1dClient.ts 参照)。project/particles/pic とは
  // 独立の state で管理する (Undo/Redo 対象外、pic1d と同じ設計)
  const [fluid1d, setFluid1d] = useState<Fluid1dSettings>(DEFAULT_FLUID1D);
  const [fluid1dRunning, setFluid1dRunning] = useState(false);
  const [fluid1dStarted, setFluid1dStarted] = useState<Fluid1dStartedMsg | null>(null);
  const [fluid1dFrame, setFluid1dFrame] = useState<Fluid1dFrameMsg | null>(null);
  // done メッセージで受け取った結果一式 (settings を含み自己完結、結果付き保存にそのまま使える)
  const [fluid1dResult, setFluid1dResult] = useState<Fluid1dResult | null>(null);
  const [fluid1dError, setFluid1dError] = useState<string | null>(null);
  const fluid1dClientRef = useRef<Fluid1dClient | null>(null);
  // 「続きから」ボタンの有効条件: 直前の実行が done/stop 済みで現在実行中でないこと。
  // pic1d と同じく geometry/mesh に依存しないため「食い違いで無効化する」概念は無い
  const [fluid1dContinueReady, setFluid1dContinueReady] = useState(false);

  // 2D/軸対称 プラズマ流体 (ドリフト拡散 + 電子エネルギー、EAFE/FEM-SG、prompts/111-113)。
  // fluid1d と異なりジオメトリ・メッシュ・境界条件は既存のプロジェクト設定をそのまま使う
  // (2D PIC と同一条件で比較できる設計目標) が、実行状態・設定自体は他の * d 系と同じく
  // project 本体とは独立の state で持つ (サーバー側のロック・保持スロットも別、fluid2dClient.ts 参照)。
  // キャンバス表示は CadCanvas の既存汎用機構 (picFieldView/picFrame) に載せるため、
  // 2D PIC (pic) と同じ「結果表示/ライブ表示/周期アニメ」の選択状態一式を持つ
  const [fluid2d, setFluid2d] = useState<Fluid2dSettings>(DEFAULT_FLUID2D);
  const [fluid2dRunning, setFluid2dRunning] = useState(false);
  const [fluid2dStarted, setFluid2dStarted] = useState<Fluid2dStartedMsg | null>(null);
  const [fluid2dFrame, setFluid2dFrame] = useState<Fluid2dFrameMsg | null>(null);
  // done メッセージで受け取った結果一式 (settings を含み自己完結。fields/cycle もここから読む —
  // PIC と違い fluid2d の done は結果を1つの result にまとめて返すため、fields/cycle 用に
  // 別の state を重複して持たない、fluid1d/tl と同じ設計)
  const [fluid2dResult, setFluid2dResult] = useState<Fluid2dResult | null>(null);
  const [fluid2dError, setFluid2dError] = useState<string | null>(null);
  const fluid2dClientRef = useRef<Fluid2dClient | null>(null);
  const [fluid2dContinueReady, setFluid2dContinueReady] = useState(false);
  // 「続きから」無効化その2: fluid2d はジオメトリに依存する (2D PIC の picProjectChangedSinceRun
  // と同じ理由) ため、前回実行以降にジオメトリ・境界条件が編集されたら続き実行を無効化する
  const [fluid2dProjectChangedSinceRun, setFluid2dProjectChangedSinceRun] = useState(false);
  // 「結果表示」セレクトの選択と対数スケール (2D PIC の picResultField/picLogScale と同じ役割)
  const [fluid2dResultField, setFluid2dResultField] = useState<Fluid2dResultField>("live");
  const [fluid2dLogScale, setFluid2dLogScale] = useState(false);
  // ライブモニタの表示フィールドと対数スケール (2D PIC の picLiveField/picLiveLogScale と同じ役割)
  const [fluid2dLiveField, setFluid2dLiveField] = useState<Fluid2dLiveField>("phi");
  const [fluid2dLiveLogScale, setFluid2dLiveLogScale] = useState(false);
  // 周期アニメーションプレイヤーの状態一式 (2D PIC の cycleField 等と同じ役割)。データ自体
  // (fluid2dResult.cycle) とは別に、UI の選択状態のみをここで持つ
  const [fluid2dCycleField, setFluid2dCycleField] = useState<Fluid2dCycleField>("phi");
  const [fluid2dCycleLogScale, setFluid2dCycleLogScale] = useState(false);
  const [fluid2dCyclePlaying, setFluid2dCyclePlaying] = useState(false);
  const [fluid2dCycleBinIndex, setFluid2dCycleBinIndex] = useState(0);
  const [fluid2dCycleFps, setFluid2dCycleFps] = useState(10);
  // 周期アニメーションが「結果表示より優先して描画される」のは、ユーザーがアニメを操作している
  // 間だけにする (2D PIC の cycleViewActive と同じ不具合修正の考え方)
  const [fluid2dCycleViewActive, setFluid2dCycleViewActive] = useState(false);

  // VHF 定在波 (非線形径方向伝送線路モデル、prompts/101)。pic1d と同様 geometry/mesh とは
  // 無関係な独立 state (Undo/Redo 対象外)。continue が無いため pic1d の *ContinueReady に
  // 相当する state は不要 (毎回フルの定常化をやり直すだけ、tl.py の docstring 参照)
  const [tl, setTl] = useState<TlSettings>(DEFAULT_TL);
  const [tlRunning, setTlRunning] = useState(false);
  const [tlStarted, setTlStarted] = useState<TlStartedMsg | null>(null);
  // 実行中の進捗 (started/progress メッセージ由来。GasPanel の progress と同じ設計)
  const [tlProgress, setTlProgress] = useState<{ step: number; nSteps: number } | null>(null);
  // done メッセージで受け取った結果一式 (settings を含み自己完結、結果付き保存にそのまま使える)
  const [tlResult, setTlResult] = useState<TlResult | null>(null);
  const [tlError, setTlError] = useState<string | null>(null);
  const tlClientRef = useRef<TlClient | null>(null);

  // ガス流れ (DSMC) 設定は project.dsmc として project state 本体に置く (particles/pic と異なり
  // 独立 state を持たず、ジオメトリ・メッシュ設定と同様 commitProject 経由で Undo/Redo 対象になる)。
  // 実行結果・実行状態は他パネルの result 系 state と同様に App 側で保持する
  const [gasRunning, setGasRunning] = useState(false);
  const [gasError, setGasError] = useState<string | null>(null);
  const [gasResult, setGasResult] = useState<DsmcResult | null>(null);
  // 実行中の進捗 (started/progress メッセージから更新。未実行/完了後は null)
  const [gasProgress, setGasProgress] = useState<{ step: number; nSteps: number; nParticles: number } | null>(null);
  // DSMC の実効スレッド数 (started で受信)。設定が実際に動いているバックエンドへ
  // 届いているかの確認用 (旧サイドカー残留の切り分けにも使える)
  const [gasThreads, setGasThreads] = useState<number | null>(null);
  // 実行中のライブ粒子位置 (progress の間引き座標。実行中のみ非null、完了後は結果フィールド
  // 表示に切り替わるため null に戻す、prompts/66)
  const [gasLiveParticles, setGasLiveParticles] = useState<Point[] | null>(null);
  // 「粒子を表示」チェックボックス (GasPanel、既定 ON)。OFF なら CadCanvas へ null を渡す
  const [gasShowParticles, setGasShowParticles] = useState(true);
  const dsmcClientRef = useRef<DsmcClient | null>(null);
  // 「結果表示」セレクトの選択と対数スケールチェックボックス
  const [gasResultField, setGasResultField] = useState<GasResultField>("n");
  const [gasLogScale, setGasLogScale] = useState(false);
  // 「続きから実行」ボタンの有効条件 (PIC の picContinueReady/picProjectChangedSinceRun と同じ役割)。
  // その1: 直前の実行が done (または stop) 済みで、現在実行中でないこと
  const [gasContinueReady, setGasContinueReady] = useState(false);
  // その2: 前回の DSMC 実行以降に「n_steps/avg_steps/threads/smoothing_passes 以外」が
  // 変わっていないこと。この4つは prepare_continue が対応しており粒子状態と非互換にならない
  // ため、これらだけの変更 (ステップ数を増やして続き実行、等) では無効化しない (commitProject 参照)
  const [gasProjectChangedSinceRun, setGasProjectChangedSinceRun] = useState(false);

  // パラメータスイープ (prompts/79)。対象パラメータ・値リスト・並列数の選択自体は
  // SweepPanel 内のローカル state で管理し (Undo/Redo 対象外)、App 側は WS の実行状態
  // (実行中フラグ・ケースごとの進捗・エラー) だけを持つ (PIC/DSMC の running/result 系と同じ設計)
  const [sweepRunning, setSweepRunning] = useState(false);
  const [sweepStarted, setSweepStarted] = useState<SweepStartedMsg | null>(null);
  const [sweepCases, setSweepCases] = useState<SweepCaseState[]>([]);
  const [sweepError, setSweepError] = useState<string | null>(null);
  // ケース一覧から結果を読み込んだ行のハイライト用 (読込前/新しいスイープ開始時は null)
  const [sweepLoadedCaseIndex, setSweepLoadedCaseIndex] = useState<number | null>(null);
  const sweepClientRef = useRef<SweepClient | null>(null);

  // 実行経過時間のリアルタイム表示 (ステータスバー、prompts/86)。何か実行中の間だけ
  // 1秒間隔で再レンダーする (アイドル時に setInterval を張り続けて無駄な再レンダーを
  // 起こさないようにするため、実行中フラグが1つでも立っているときだけ張る)
  const anyRunning =
    busy || picRunning || pic1dRunning || fluid1dRunning || fluid2dRunning || tlRunning || gasRunning || sweepRunning;
  useEffect(() => {
    if (!anyRunning) return;
    const id = setInterval(() => setElapsedTick((t) => t + 1), 1000);
    return () => clearInterval(id);
  }, [anyRunning]);

  // 周期アニメーション再生ループ: playing 中は fps に応じた間隔でビンを1つずつ順送りし、
  // 最後まで行ったら先頭へループする (setInterval + 関数更新で古いクロージャの影響を避ける)
  useEffect(() => {
    if (!cyclePlaying || !picCycle || picCycle.bins <= 0) return;
    const bins = picCycle.bins;
    const id = setInterval(() => {
      setCycleBinIndex((i) => (i + 1) % bins);
    }, 1000 / cycleFps);
    return () => clearInterval(id);
  }, [cyclePlaying, cycleFps, picCycle]);

  // 流体 (2D) の周期アニメーション再生ループ (上記 PIC のものと同じ設計。データは
  // fluid2dResult.cycle から読む点のみが異なる)
  useEffect(() => {
    const cycle = fluid2dResult?.cycle;
    if (!fluid2dCyclePlaying || !cycle || cycle.bins <= 0) return;
    const bins = cycle.bins;
    const id = setInterval(() => {
      setFluid2dCycleBinIndex((i) => (i + 1) % bins);
    }, 1000 / fluid2dCycleFps);
    return () => clearInterval(id);
  }, [fluid2dCyclePlaying, fluid2dCycleFps, fluid2dResult]);

  // アンマウント時に WebSocket 接続を確実に閉じる
  useEffect(() => {
    return () => {
      picClientRef.current?.close();
      pic1dClientRef.current?.close();
      fluid1dClientRef.current?.close();
      fluid2dClientRef.current?.close();
      tlClientRef.current?.close();
      dsmcClientRef.current?.close();
      sweepClientRef.current?.close();
    };
  }, []);

  // 起動時にポート設定を初期化する (Tauri内なら AppConfig の backend-port.txt、
  // なければ localStorage、それも無ければ既定値 8317)。完了してから health チェックを始める
  useEffect(() => {
    let cancelled = false;
    initPort().then((p) => {
      if (cancelled) return;
      setPortInput(String(p));
      setPortReady(true);
    });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!portReady) return; // initPort() 完了まで待つ
    const check = () => api.health().then(setHealth).catch(() => setHealth(null));
    check();
    const t = setInterval(check, 5000);
    return () => clearInterval(t);
  }, [portReady]);

  // ポート入力欄の確定 (blur / Enter)。不正な値なら現在値に戻すだけで何もしない
  const commitPortInput = () => {
    const n = parseInt(portInput, 10);
    if (!Number.isFinite(n) || n <= 0) {
      setPortInput(String(getPort()));
      return;
    }
    if (n === getPort()) return; // 変更なしなら再チェック不要
    setPortError(null);
    setPort(n)
      .then(() => {
        setHealth(null); // 変更直後は未確定表示にし、health即再チェックの結果で更新する
        return api.health().then(setHealth).catch(() => setHealth(null));
      })
      .catch((e) => setPortError(String(e)));
  };

  // 選択中領域が project から消えていたら選択解除する
  const ensureSelection = useCallback((p: Project, sel: string | null): string | null => {
    if (sel === null) return null;
    return p.geometry.regions.some((r) => r.id === sel) ? sel : null;
  }, []);

  // 編集操作の確定: 直前の状態を履歴へ積み、新しい状態を反映する。
  // 解析結果は state が変わったら破棄する (プロファイルパネルも連動して閉じる)。
  const commitProject = useCallback((next: Project) => {
    const prev = projectRef.current;
    history.push(prev);
    projectRef.current = next;
    setProjectState(next);
    setResult(null);
    setSolveElapsedS(null); // 計算時間表示も結果と一緒に破棄する (prompts/86)
    setMeshResult(null);
    setProfileLine(null);
    setTraceResult(null); // ジオメトリ変更で解析結果とともに trace 結果も破棄する
    setTraceElapsedS(null);
    setGasResult(null); // メッシュが変わりうるため DSMC 結果 (要素値) も破棄する
    setPicProjectChangedSinceRun(true); // PIC続き実行はサーバー状態と食い違うため無効化する
    setFluid2dProjectChangedSinceRun(true); // 流体2D続き実行も同じ理由で無効化する
    // DSMC続き実行の無効化は n_steps/avg_steps/threads/smoothing_passes 以外が変わったときだけ
    // (dsmcContinueRelevantKey 参照)。ステップ数を増やして続き実行、を妨げないための例外
    if (dsmcContinueRelevantKey(prev) !== dsmcContinueRelevantKey(next)) {
      setGasProjectChangedSinceRun(true);
    }
  }, [history]);

  // --- Undo/Redo ---
  const doUndo = useCallback(() => {
    const prev = history.undo(projectRef.current);
    if (prev === null) return;
    projectRef.current = prev;
    setProjectState(prev);
    setResult(null);
    setSolveElapsedS(null);
    setMeshResult(null);
    setProfileLine(null);
    setTraceResult(null);
    setTraceElapsedS(null);
    setGasResult(null);
    setSelectedRegionId((sel) => ensureSelection(prev, sel));
    setPicProjectChangedSinceRun(true); // PIC続き実行はサーバー状態と食い違うため無効化する
    setGasProjectChangedSinceRun(true); // Undo は任意の過去状態へ飛びうるため常に無効化する
    setFluid2dProjectChangedSinceRun(true); // 流体2D続き実行も同様に常に無効化する
  }, [history, ensureSelection]);

  const doRedo = useCallback(() => {
    const next = history.redo(projectRef.current);
    if (next === null) return;
    projectRef.current = next;
    setProjectState(next);
    setResult(null);
    setSolveElapsedS(null);
    setMeshResult(null);
    setProfileLine(null);
    setTraceResult(null);
    setTraceElapsedS(null);
    setGasResult(null);
    setSelectedRegionId((sel) => ensureSelection(next, sel));
    setPicProjectChangedSinceRun(true); // PIC続き実行はサーバー状態と食い違うため無効化する
    setGasProjectChangedSinceRun(true); // Redo も任意の過去状態へ飛びうるため常に無効化する
    setFluid2dProjectChangedSinceRun(true); // 流体2D続き実行も同様に常に無効化する
  }, [history, ensureSelection]);

  // キーボードショートカット: Ctrl+Z (Undo) / Ctrl+Y, Ctrl+Shift+Z (Redo)
  // テキスト入力中は素通しする (native な編集を邪魔しないため)
  useEffect(() => {
    const isEditable = (t: EventTarget | null) =>
      t instanceof HTMLElement && ["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName);
    const onKeyDown = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      if (isEditable(e.target)) return;
      const key = e.key.toLowerCase();
      if (key === "z") {
        e.preventDefault();
        if (e.shiftKey) doRedo();
        else doUndo();
      } else if (key === "y") {
        e.preventDefault();
        doRedo();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [doUndo, doRedo]);

  const runSolve = async () => {
    busyStartTimeRef.current = Date.now(); // ステータスバーの経過時間表示用 (prompts/86)
    setBusy(true);
    setError(null);
    setMeshResult(null); // Solve 実行時は Mesh のみの結果を破棄し、Solve 側の表示を優先する
    setMeshPreviewFresh(false); // Mesh プレビューの優先表示も解除する (prompts/89 ②)
    const t0 = performance.now(); // 結果サマリの「計算時間」表示用。backend 変更不要のフロント側計測 (prompts/86)
    try {
      setResult(await api.solve(project));
      setSolveElapsedS((performance.now() - t0) / 1000);
    } catch (e) {
      setError(String(e));
      setSolveElapsedS(null);
    } finally {
      setBusy(false);
    }
  };

  // メッシュ生成のみ (解析は行わない)
  const runMesh = async () => {
    busyStartTimeRef.current = Date.now(); // ステータスバーの経過時間表示用 (prompts/86)
    setBusy(true);
    setError(null);
    try {
      setMeshResult(await api.mesh(project));
      // 直前の Solve/PIC 結果が残っていても、今生成したメッシュプレビューを
      // キャンバスで最優先表示する (prompts/89 ②)
      setMeshPreviewFresh(true);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  // 粒子軌道トレース実行 (project.particles として送信する)
  const runTrace = async () => {
    busyStartTimeRef.current = Date.now(); // ステータスバーの経過時間表示用 (prompts/86)
    setBusy(true);
    setError(null);
    const t0 = performance.now(); // トレース結果サマリの「計算時間」表示用 (prompts/86)
    try {
      setTraceResult(await api.trace({ ...project, particles }));
      setTraceElapsedS((performance.now() - t0) / 1000);
    } catch (e) {
      setTraceElapsedS(null);
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  // ガス流れ (DSMC) 設定の更新。project.dsmc は Project 本体のフィールドなので commitProject
  // 経由で反映する (ジオメトリ・メッシュ設定と同じ Undo/Redo 対象)
  const setDsmc = (next: Project["dsmc"]) => {
    const p = projectRef.current;
    commitProject({ ...p, dsmc: next });
  };

  // DSMC実行中のコールバック生成 (start/continue で共通化。PIC の makePicCallbacks と同じ考え方だが、
  // DSMC は history 連結のような区別が不要なので isContinue パラメータは無い)
  const makeDsmcCallbacks = (): DsmcClientCallbacks => ({
    onStarted: (msg) => {
      setGasProgress({ step: 0, nSteps: msg.n_steps, nParticles: msg.n_particles });
      setGasThreads(msg.threads ?? null);
    },
    onProgress: (msg) => {
      setGasProgress({ step: msg.step, nSteps: msg.n_steps, nParticles: msg.n_particles });
      // 未対応バックエンド (particles 省略) では null のまま (何も描画しない)
      setGasLiveParticles(msg.particles ?? null);
    },
    onDone: (msg) => {
      setGasResult(msg.result);
      setGasRunning(false);
      setGasLiveParticles(null); // 完了後は結果フィールド表示に切り替わるため残さない
      setGasContinueReady(true); // done (stop 済みも含む) したので続き実行が可能になる
    },
    onError: (detail) => {
      setGasError(detail);
      setGasRunning(false);
      setGasLiveParticles(null);
      setGasContinueReady(false); // エラー後の状態は不定なので続き実行は無効のままにする
    },
    onClose: () => {
      setGasRunning(false);
      setGasLiveParticles(null);
    },
  });

  // DSMC ガス流れ計算実行 (WebSocket。project.dsmc を送信する。project.dsmc が null の場合は呼ばれない想定)
  const runDsmc = () => {
    gasStartTimeRef.current = Date.now(); // ステータスバーの経過時間表示用 (prompts/86)
    setGasError(null);
    setGasResult(null);
    setGasProgress(null);
    setGasLiveParticles(null);
    setGasRunning(true);
    setGasContinueReady(false);
    const client = new DsmcClient(makeDsmcCallbacks());
    dsmcClientRef.current = client;
    // 現在のプロジェクト状態をサーバーへ送るので、続き実行の食い違いフラグをここで解消する
    setGasProjectChangedSinceRun(false);
    client.start(project);
  };

  // DSMC続きから実行: 保持中のシミュレーション状態 (粒子・乱数・NTC端数等) を維持したまま
  // 現在の n_steps/avg_steps で追加実行する。ジオメトリ・ガス条件などプロジェクト側の変更は
  // サーバーへは送らないため反映されない (変更があれば gasProjectChangedSinceRun でボタンを無効化する)
  const runDsmcContinue = () => {
    if (!dsmcClientRef.current || !project.dsmc || gasRunning || !gasContinueReady || gasProjectChangedSinceRun) {
      return;
    }
    gasStartTimeRef.current = Date.now(); // ステータスバーの経過時間表示用 (prompts/86)
    setGasError(null);
    setGasResult(null); // 進捗表示は新区間で0から (前回結果は継続分の done で置き換わる)
    setGasProgress(null);
    setGasLiveParticles(null);
    setGasRunning(true);
    setGasContinueReady(false);
    dsmcClientRef.current.setCallbacks(makeDsmcCallbacks());
    dsmcClientRef.current.continueRun({
      n_steps: project.dsmc.n_steps,
      avg_steps: project.dsmc.avg_steps ?? null,
    });
  };

  // DSMC 計算の中断
  const stopDsmc = () => {
    dsmcClientRef.current?.stop();
  };

  // パラメータスイープに渡すプロジェクト (particles/pic/pic1d/fluid1d は独立 state のためここで
  // 合成する。saveProject/runPicStart と同じ合成方法。b_field/dsmc/pic/pic1d/fluid1d の候補パスや
  // 現在値のプレビューはこのオブジェクトを基準に組み立てるため、常に最新の
  // pic/pic1d/fluid1d/particles を反映させる。fluid1d も pic1d と同じく geometry/mesh に依存しない
  // 独立設定だが、1D スイープ (prompts/96・107) の候補生成・現在値プレビューのために合成しておく)
  const projectForSweep: Project = {
    ...project,
    particles,
    pic: withInjectionEmitter(pic, particles.emitter),
    pic1d,
    fluid1d,
    fluid2d,
  };

  // スイープ実行中のコールバック生成 (PIC/DSMC の makePicCallbacks/makeDsmcCallbacks と同じ考え方)
  const makeSweepCallbacks = (): SweepClientCallbacks => ({
    onStarted: (msg) => {
      setSweepStarted(msg);
      setSweepCases(msg.values.map((v) => ({ value: v, status: "pending" as const })));
    },
    onProgress: (msg) => {
      setSweepCases((prev) => {
        const next = prev.slice();
        const cur = next[msg.case];
        if (cur) next[msg.case] = { ...cur, status: "running", step: msg.step, nSteps: msg.n_steps };
        return next;
      });
    },
    onCaseDone: (msg) => {
      setSweepCases((prev) => {
        const next = prev.slice();
        const cur = next[msg.case];
        if (cur) next[msg.case] = { ...cur, status: msg.ok ? "done" : "error", error: msg.error };
        return next;
      });
    },
    onDone: () => {
      setSweepRunning(false);
    },
    onError: (detail) => {
      setSweepError(detail);
      setSweepRunning(false);
    },
    onClose: () => setSweepRunning(false),
  });

  // スイープ開始: SweepPanel で組み立てたパス・値リスト・並列数を送信する
  // スイープ対象パスの終端キーを送信前に実体化する。see_gamma のような省略可能
  // フィールドはプロジェクト JSON にキー自体が無いことがあり、そのままだと backend の
  // set_by_path (厳格: 不在パスはエラー) が全ケース失敗するため、親が存在する場合に
  // 限り数値 0 で終端キーを補う (既に数値があれば何もしない)
  const ensureSweepPath = (obj: unknown, path: string) => {
    const toks = path.split(".");
    let cur: unknown = obj;
    for (const tok of toks.slice(0, -1)) {
      if (cur == null) return;
      cur = Array.isArray(cur) ? cur[Number(tok)] : (cur as Record<string, unknown>)[tok];
    }
    const last = toks[toks.length - 1];
    if (cur != null && typeof cur === "object" && !Array.isArray(cur)) {
      const rec = cur as Record<string, unknown>;
      if (typeof rec[last] !== "number") rec[last] = 0;
    }
  };

  // pic1d 電極の voltage_rf を単一オブジェクトからリスト形式へ正規化する (1D スイープ、prompts/96)。
  // SweepPanel の 1D RF プリセットは常に "pic1d.<side>.voltage_rf.0.<field>" 形式のパスを使う
  // (2D の候補と違い、単一/リストいずれの現在値でも同じパス表現で扱えるようにするため) ので、
  // 現在値が単一オブジェクトのままだと配列インデックス 0 を解決できず set_by_path が失敗する。
  // ensureSweepPath と同じ「送信直前に足りない形を実体化する」流儀で [obj] へ包み直す
  // (値そのものは変えないので、正規化してもスイープ前の実行結果には影響しない)
  const normalizePic1dVoltageRf = (pic1d: Pic1dSettings): Pic1dSettings => {
    const normSide = (side: Pic1dElectrode): Pic1dElectrode =>
      side.voltage_rf != null && !Array.isArray(side.voltage_rf)
        ? { ...side, voltage_rf: [side.voltage_rf] }
        : side;
    return { ...pic1d, left: normSide(pic1d.left), right: normSide(pic1d.right) };
  };

  // fluid1d 版 (prompts/109)。fluid1d.left/right も Pic1dElectrode 型を共用しているため、
  // normalizePic1dVoltageRf と全く同じロジックで正規化できる
  const normalizeFluid1dVoltageRf = (fluid1d: Fluid1dSettings): Fluid1dSettings => {
    const normSide = (side: Pic1dElectrode): Pic1dElectrode =>
      side.voltage_rf != null && !Array.isArray(side.voltage_rf)
        ? { ...side, voltage_rf: [side.voltage_rf] }
        : side;
    return { ...fluid1d, left: normSide(fluid1d.left), right: normSide(fluid1d.right) };
  };

  const runSweepStart = (
    paramPath: string,
    values: number[],
    parallel: number,
    module: "pic" | "pic1d" | "fluid1d" | "fluid2d",
  ) => {
    sweepStartTimeRef.current = Date.now(); // ステータスバーの経過時間表示用 (prompts/86)
    setSweepError(null);
    setSweepStarted(null);
    setSweepCases([]);
    setSweepLoadedCaseIndex(null);
    setSweepRunning(true);
    const client = new SweepClient(makeSweepCallbacks());
    sweepClientRef.current = client;
    // 深いコピーに対して終端キーを実体化してから送る (元の project state は汚さない)
    const proj = JSON.parse(JSON.stringify(projectForSweep)) as typeof projectForSweep;
    if (proj.pic1d) proj.pic1d = normalizePic1dVoltageRf(proj.pic1d);
    if (proj.fluid1d) proj.fluid1d = normalizeFluid1dVoltageRf(proj.fluid1d);
    ensureSweepPath(proj, paramPath);
    client.start(proj, paramPath, values, parallel, module);
  };

  const runSweepStop = () => {
    sweepClientRef.current?.stop();
  };

  // PIC実行中のコールバック生成 (start/continue で共通化)。isContinue が true のときは
  // done 受信時に history を「置き換え」ではなく既存へ「連結」する (それ以外の挙動は同じ:
  // 新しい実行区間の started/frame でライブ表示は自然に切り替わり、fields/cycle/collector は
  // 新しい done の内容で置き換わる)
  const makePicCallbacks = (isContinue: boolean): PicClientCallbacks => ({
    onStarted: (msg) => {
      setPicStarted(msg);
      setPicFrame(null); // ライブ表示を新しい実行区間の内容に自然に切り替える
    },
    onFrame: (msg) => {
      setPicFrame(msg);
      setPicHistory((h) => [...h, msg.diag]);
    },
    onDone: (msg) => {
      // バックエンドの history は列ごとの辞書形式なので行ごとの PicDiag[] に変換する
      // (形式不一致のまま描画するとチャートが例外を投げて画面全体が落ちるため必ず変換を通す)
      const added = toDiagArray(msg.history);
      setPicHistory((h) => (isContinue ? [...h, ...added] : added));
      setPicFields(msg.fields ?? null);
      setPicTiming(msg.timing ?? null);
      setPicElapsedS(msg.elapsed_s ?? null); // run_batch の壁時計秒 (prompts/86)
      setPicCycle(msg.cycle ?? null);
      // collectors 配列が来ればそちらを使い、旧バックエンドで単数 collector のみの場合は
      // 先頭(1個)として扱う (prompts/37、必須ではないが安全に対応しておく)
      setPicCollectors(msg.collectors ?? (msg.collector ? [msg.collector] : []));
      setPicEedf(msg.eedf ?? []);
      setPicRunning(false);
      setPicContinueReady(true); // done (stop 済みも含む) したので続き実行が可能になる
    },
    onError: (detail) => {
      setPicError(detail);
      setPicRunning(false);
      setPicContinueReady(false); // エラー後の状態は不定なので続き実行は無効のままにする
    },
    onClose: () => setPicRunning(false),
  });

  // PIC開始: WebSocket接続を張り、project.pic (エミッタはフェーズ2の設定と同期) を送信する
  const runPicStart = () => {
    picStartTimeRef.current = Date.now(); // ステータスバーの経過時間表示用 (prompts/86)
    setPicError(null);
    setPicStarted(null);
    setPicFrame(null);
    setPicHistory([]);
    setPicRunning(true);
    setPicContinueReady(false);
    setMeshPreviewFresh(false); // 新しい PIC 実行のライブ表示を優先する (prompts/89 ②)
    // 新しい実行を開始したら結果フィールド表示 (前回 done の残骸) をリセットする
    setPicFields(null);
    setPicTiming(null); // 位相別プロファイル計測 (prompts/75) も前回 done の残骸を消す
    setPicElapsedS(null);
    setPicResultField("live");
    setPicLogScale(false);
    // 周期アニメーションの状態も新規実行開始時にリセットする (前回 done の cycle・再生状態を破棄)
    setPicCycle(null);
    setCycleField("phi");
    setCycleLogScale(false);
    setCyclePlaying(false);
    setCycleBinIndex(0);
    setCycleShowParticles(true);
    setCycleViewActive(false);
    // IEDF/IADF コレクタ結果も新規実行開始時にリセットする (前回 done の残骸を消す)
    setPicCollectors([]);
    // EEDF/EEPF 領域結果も同様にリセットする (prompts/85)
    setPicEedf([]);
    const client = new PicClient(makePicCallbacks(false));
    picClientRef.current = client;
    // 現在のプロジェクト状態をサーバーへ送るので、続き実行の食い違いフラグをここで解消する
    setPicProjectChangedSinceRun(false);
    client.start({ ...project, pic: withInjectionEmitter(pic, particles.emitter) });
  };

  // PIC続きから実行: 保持中のシミュレーション状態 (粒子・表面電荷・時刻・乱数) を維持したまま
  // 現在の計算設定 (n_steps/frame_every/avg_steps/phase_bins) で追加実行する。
  // ジオメトリ・プラズマ設定などプロジェクト側の変更はサーバーへは送らないため反映されない。
  // picHistory はクリアせず、既存の履歴 (フル実行分) の末尾へ追加区間分を連結する
  const runPicContinue = () => {
    if (!picClientRef.current || picRunning || !picContinueReady || picProjectChangedSinceRun) return;
    picStartTimeRef.current = Date.now(); // ステータスバーの経過時間表示用 (prompts/86)
    setPicError(null);
    setPicRunning(true);
    setPicContinueReady(false);
    setMeshPreviewFresh(false); // 続き実行のライブ表示を優先する (prompts/89 ②)
    // 表示状態を新規実行と同様にリセットする。描画優先順位が
    // 「周期アニメ > 結果フィールド > ライブ」のため、前回 done の cycle / 結果フィールド
    // 選択が残っているとライブ表示が隠れ、続き実行中の画面が追従しない (不具合修正)。
    // 新しい fields / cycle / collectors / eedf は追加区間の done で置き換わる
    setPicFields(null);
    setPicTiming(null);
    setPicElapsedS(null);
    setPicResultField("live");
    setPicLogScale(false);
    setPicCycle(null);
    setCyclePlaying(false);
    setCycleBinIndex(0);
    setCycleViewActive(false);
    setPicCollectors([]);
    setPicEedf([]);
    picClientRef.current.setCallbacks(makePicCallbacks(true));
    picClientRef.current.continueRun({
      n_steps: pic.n_steps,
      frame_every: pic.frame_every,
      avg_steps: pic.avg_steps ?? null,
      phase_bins: pic.phase_bins ?? null,
    });
  };

  const runPicStop = () => {
    picClientRef.current?.stop();
  };

  // 1D PIC実行中のコールバック生成 (start/continue で共通化)。2D の makePicCallbacks と同じ設計だが、
  // pic1d は geometry/mesh に依存しないため「続き実行時のジオメトリ食い違い」は無く、
  // done.result がそのまま自己完結した結果一式 (history 込み) を返すので連結処理も不要
  const makePic1dCallbacks = (): Pic1dClientCallbacks => ({
    onStarted: (msg) => {
      setPic1dStarted(msg);
      setPic1dFrame(null); // ライブ表示を新しい実行区間の内容に自然に切り替える
    },
    onFrame: (msg) => {
      setPic1dFrame(msg);
    },
    onDone: (msg) => {
      setPic1dResult(msg.result);
      setPic1dRunning(false);
      setPic1dContinueReady(true); // done (stop 済みも含む) したので続き実行が可能になる
    },
    onError: (detail) => {
      setPic1dError(detail);
      setPic1dRunning(false);
      setPic1dContinueReady(false); // エラー後の状態は不定なので続き実行は無効のままにする
    },
    onClose: () => setPic1dRunning(false),
  });

  // PIC 1D開始: WebSocket接続を張り、project.pic1d を送信する (2D の runPicStart と同じ設計)
  const runPic1dStart = () => {
    pic1dStartTimeRef.current = Date.now(); // ステータスバーの経過時間表示用 (prompts/86 相当)
    setPic1dError(null);
    setPic1dStarted(null);
    setPic1dFrame(null);
    setPic1dRunning(true);
    setPic1dContinueReady(false);
    const client = new Pic1dClient(makePic1dCallbacks());
    pic1dClientRef.current = client;
    client.start({ ...project, pic1d });
  };

  // PIC 1D続きから実行: 保持中のシミュレーション状態 (粒子・表面電荷・時刻・乱数) を維持したまま
  // extraSteps 分だけ追加実行する。フレーム間隔・平均ステップ数・位相ビン数は現在の pic1d 設定を使う
  const runPic1dContinue = (extraSteps: number) => {
    if (!pic1dClientRef.current || pic1dRunning || !pic1dContinueReady) return;
    pic1dStartTimeRef.current = Date.now();
    setPic1dError(null);
    setPic1dRunning(true);
    setPic1dContinueReady(false);
    pic1dClientRef.current.setCallbacks(makePic1dCallbacks());
    pic1dClientRef.current.continueRun({
      extra_steps: extraSteps,
      frame_every: pic1d.frame_every,
      avg_steps: pic1d.avg_steps ?? null,
      phase_bins: pic1d.phase_bins ?? null,
    });
  };

  const runPic1dStop = () => {
    pic1dClientRef.current?.stop();
  };

  // 1D 流体実行中のコールバック生成。makePic1dCallbacks と全く同じ設計 (pic1d と同様
  // geometry/mesh に依存しないため続き実行時のジオメトリ食い違いは無く、done.result が
  // そのまま自己完結した結果一式を返す、prompts/109)
  const makeFluid1dCallbacks = (): Fluid1dClientCallbacks => ({
    onStarted: (msg) => {
      setFluid1dStarted(msg);
      setFluid1dFrame(null); // ライブ表示を新しい実行区間の内容に自然に切り替える
    },
    onFrame: (msg) => {
      setFluid1dFrame(msg);
    },
    onDone: (msg) => {
      setFluid1dResult(msg.result);
      setFluid1dRunning(false);
      setFluid1dContinueReady(true); // done (stop 済みも含む) したので続き実行が可能になる
    },
    onError: (detail) => {
      setFluid1dError(detail);
      setFluid1dRunning(false);
      setFluid1dContinueReady(false); // エラー後の状態は不定なので続き実行は無効のままにする
    },
    onClose: () => setFluid1dRunning(false),
  });

  // 流体1D開始: WebSocket接続を張り、project.fluid1d を送信する (runPic1dStart と同じ設計)
  const runFluid1dStart = () => {
    fluid1dStartTimeRef.current = Date.now(); // ステータスバーの経過時間表示用
    setFluid1dError(null);
    setFluid1dStarted(null);
    setFluid1dFrame(null);
    setFluid1dRunning(true);
    setFluid1dContinueReady(false);
    const client = new Fluid1dClient(makeFluid1dCallbacks());
    fluid1dClientRef.current = client;
    client.start({ ...project, fluid1d });
  };

  // 流体1D続きから実行: 保持中のシミュレーション状態 (場・時刻) を維持したまま
  // extraSteps 分だけ追加実行する。フレーム間隔・平均ステップ数・位相ビン数は現在の fluid1d 設定を使う
  const runFluid1dContinue = (extraSteps: number) => {
    if (!fluid1dClientRef.current || fluid1dRunning || !fluid1dContinueReady) return;
    fluid1dStartTimeRef.current = Date.now();
    setFluid1dError(null);
    setFluid1dRunning(true);
    setFluid1dContinueReady(false);
    fluid1dClientRef.current.setCallbacks(makeFluid1dCallbacks());
    fluid1dClientRef.current.continueRun({
      extra_steps: extraSteps,
      frame_every: fluid1d.frame_every,
      avg_steps: fluid1d.avg_steps ?? null,
      phase_bins: fluid1d.phase_bins ?? null,
    });
  };

  const runFluid1dStop = () => {
    fluid1dClientRef.current?.stop();
  };

  // 流体 (2D) 実行中のコールバック生成 (2D PIC の makePicCallbacks と同じ設計だが、
  // done.result が settings を含み自己完結した1つのバンドルを返す点は fluid1d と同じ)
  const makeFluid2dCallbacks = (): Fluid2dClientCallbacks => ({
    onStarted: (msg) => {
      setFluid2dStarted(msg);
      setFluid2dFrame(null); // ライブ表示を新しい実行区間の内容に自然に切り替える
    },
    onFrame: (msg) => {
      setFluid2dFrame(msg);
    },
    onDone: (msg) => {
      setFluid2dResult(msg.result);
      setFluid2dRunning(false);
      setFluid2dContinueReady(true); // done (stop 済みも含む) したので続き実行が可能になる
    },
    onError: (detail) => {
      setFluid2dError(detail);
      setFluid2dRunning(false);
      setFluid2dContinueReady(false); // エラー後の状態は不定なので続き実行は無効のままにする
    },
    onClose: () => setFluid2dRunning(false),
  });

  // 流体2D開始: /ws/fluid2d の started はメッシュを含まない (server.py のコメント参照) ため、
  // 2D PIC 開始時の「バックエンドが返すメッシュをそのまま使う」挙動に相当するものとして、
  // 開始前にフロント側で POST /mesh を実行し直し、フィールド描画に使う meshResult を
  // 現在の project (fluid2d 実行に使われるものと同じ geometry/mesh) と確実に整合させる
  // (prompts/113 の「実行前にメッシュ未生成なら2D PICと同じ挙動」の指示に対する対応:
  // 2D PIC はバックエンドがメッシュを返すため自動的に整合するが、fluid2d はメッシュを
  // 返さないため、フロント側の自動再生成でこれに揃える)
  const runFluid2dStart = async () => {
    fluid2dStartTimeRef.current = Date.now(); // ステータスバーの経過時間表示用
    setFluid2dError(null);
    setFluid2dStarted(null);
    setFluid2dFrame(null);
    setFluid2dRunning(true);
    setFluid2dContinueReady(false);
    setMeshPreviewFresh(false); // 新しい実行のライブ表示を優先する (2D PIC と同じ、prompts/89 ②)
    // 新しい実行を開始したら結果フィールド表示 (前回 done の残骸) をリセットする
    setFluid2dResult(null);
    setFluid2dResultField("live");
    setFluid2dLogScale(false);
    setFluid2dCycleField("phi");
    setFluid2dCycleLogScale(false);
    setFluid2dCyclePlaying(false);
    setFluid2dCycleBinIndex(0);
    setFluid2dCycleViewActive(false);
    try {
      setMeshResult(await api.mesh(project));
    } catch (e) {
      setFluid2dError(String(e));
      setFluid2dRunning(false);
      return;
    }
    const client = new Fluid2dClient(makeFluid2dCallbacks());
    fluid2dClientRef.current = client;
    // 現在のプロジェクト状態をサーバーへ送るので、続き実行の食い違いフラグをここで解消する
    setFluid2dProjectChangedSinceRun(false);
    client.start({ ...project, fluid2d });
  };

  // 流体2D続きから実行: 保持中のシミュレーション状態 (場・時刻) を維持したまま
  // extraSteps 分だけ追加実行する。フレーム間隔・平均ステップ数・位相ビン数は現在の fluid2d 設定を使う
  // (メッシュは前回 start 時のものをサーバー側がそのまま保持しているため、ここでは再取得しない)
  const runFluid2dContinue = (extraSteps: number) => {
    if (!fluid2dClientRef.current || fluid2dRunning || !fluid2dContinueReady || fluid2dProjectChangedSinceRun) return;
    fluid2dStartTimeRef.current = Date.now();
    setFluid2dError(null);
    setFluid2dRunning(true);
    setFluid2dContinueReady(false);
    setMeshPreviewFresh(false);
    // 表示状態を新規実行と同様にリセットする (2D PIC の runPicContinue と同じ不具合修正の考え方:
    // 前回 done の cycle / 結果フィールド選択が残っているとライブ表示が隠れてしまう)
    setFluid2dResult(null);
    setFluid2dResultField("live");
    setFluid2dLogScale(false);
    setFluid2dCyclePlaying(false);
    setFluid2dCycleBinIndex(0);
    setFluid2dCycleViewActive(false);
    fluid2dClientRef.current.setCallbacks(makeFluid2dCallbacks());
    fluid2dClientRef.current.continueRun({
      extra_steps: extraSteps,
      frame_every: fluid2d.frame_every,
      avg_steps: fluid2d.avg_steps ?? null,
      phase_bins: fluid2d.phase_bins ?? null,
    });
  };

  const runFluid2dStop = () => {
    fluid2dClientRef.current?.stop();
  };

  // VHF 定在波 (prompts/101) 実行中のコールバック生成。continue が無いため
  // makePic1dCallbacks と異なり ContinueReady 系の更新は無い (常に単発の start/stop のみ)
  const makeTlCallbacks = (): TlClientCallbacks => ({
    onStarted: (msg) => {
      setTlStarted(msg);
      setTlProgress(null);
    },
    onProgress: (msg) => {
      setTlProgress({ step: msg.step, nSteps: msg.n_steps });
    },
    onDone: (msg) => {
      setTlResult(msg.result);
      setTlRunning(false);
    },
    onError: (detail) => {
      setTlError(detail);
      setTlRunning(false);
    },
    onClose: () => setTlRunning(false),
  });

  // VHF定在波開始: WebSocket接続を張り、project.tl を送信する (2D/1D の runPicStart と同じ設計)
  const runTlStart = () => {
    tlStartTimeRef.current = Date.now(); // ステータスバーの経過時間表示用 (prompts/86 相当)
    setTlError(null);
    setTlStarted(null);
    setTlProgress(null);
    setTlRunning(true);
    const client = new TlClient(makeTlCallbacks());
    tlClientRef.current = client;
    client.start({ ...project, tl });
  };

  const runTlStop = () => {
    tlClientRef.current?.stop();
  };

  // エミッタ配置ツール (CadCanvas) からの確定通知。kind/n 等はそのまま維持し p1/p2 のみ更新する。
  // 線を確定したら「粒子軌道追跡」インスペクタページに切替え、プロパティがすぐ見えるようにする
  const setEmitterPoints = (p1: Point, p2: Point) => {
    setParticles((prev) => ({ ...prev, emitter: { ...prev.emitter, p1, p2 } }));
    setActiveNode("study-trace");
  };

  // コレクタ配置ツール (CadCanvas) からの確定通知。線分を確定するたびに pic.collectors へ
  // 1件追加する (最大 MAX_COLLECTORS 個、達したら追加しない)。ラベルは "C1","C2",... を自動採番
  // (欠番があっても詰めない)。配置したら PIC インスペクタページへ切替え、追加したコレクタを選択状態にする
  const setCollectorPoints = (p1: Point, p2: Point) => {
    const collectors = pic.collectors ?? [];
    if (collectors.length < MAX_COLLECTORS) {
      const label = nextCollectorLabel(collectors);
      const next = [...collectors, { p1, p2, tol: null, label }];
      setPic({ ...pic, collectors: next });
      setSelectedCollectorIndexRaw(next.length - 1);
    }
    setActiveNode("study-pic");
  };

  // ガス境界配置ツール (CadCanvas) からの確定通知。コレクタと同じ2点クリックUX (prompts/72)。
  // project.dsmc が null な状態はツール自体は表示されるが実行不能なため、2点クリックした時点で
  // 「DSMCを使う」意図が明確 → GasPanel の既定設定 (DEFAULT_DSMC) で有効化してから境界を追加する。
  // project.dsmc は Undo/Redo 対象の Project 本体フィールドなので setDsmc (commitProject) 経由で反映する
  const setGasBoundaryPoints = (p1: Point, p2: Point) => {
    const base = project.dsmc ?? DEFAULT_DSMC;
    // 新規境界の既定値は GasPanel の「境界を追加」ボタンと同一 (DEFAULT_BOUNDARY を共有) + p1/p2
    const boundary: DsmcBoundary = { ...DEFAULT_BOUNDARY, p1, p2 };
    setDsmc({ ...base, boundaries: [...base.boundaries, boundary] });
    setActiveNode("study-gas");
  };

  // 辺ローカルメッシュサイズ配置ツール (CadCanvas) からの確定通知。コレクタ・ガス境界と同じ
  // 2点クリックUX (prompts/90)。既定サイズは mesh.size / 4 (領域ローカルサイズより粗めの
  // 「まず試す」既定値。細かすぎる既定だと初回クリックで計算コストが跳ね上がるのを避ける)。
  // project.mesh は Undo/Redo 対象の Project 本体フィールドなので commitProject 経由で反映する
  const setEdgeMeshSizePoints = (p1: Point, p2: Point) => {
    const p = projectRef.current;
    const entry: EdgeMeshSize = { p1, p2, size: p.mesh.size / 4 };
    const local_edge_sizes = [...(p.mesh.local_edge_sizes ?? []), entry];
    commitProject({ ...p, mesh: { ...p.mesh, local_edge_sizes } });
    setActiveNode("mesh");
  };

  // 辺ローカルメッシュサイズ一覧 (FieldPanel) の1件のサイズを更新する
  const updateEdgeMeshSize = (index: number, size: number) => {
    const p = projectRef.current;
    const entries = p.mesh.local_edge_sizes ?? [];
    if (index < 0 || index >= entries.length || !(size > 0)) return;
    const next = entries.slice();
    next[index] = { ...next[index], size };
    commitProject({ ...p, mesh: { ...p.mesh, local_edge_sizes: next } });
  };

  // 辺ローカルメッシュサイズ一覧の1件を削除する
  const deleteEdgeMeshSize = (index: number) => {
    const p = projectRef.current;
    const entries = p.mesh.local_edge_sizes ?? [];
    if (index < 0 || index >= entries.length) return;
    commitProject({ ...p, mesh: { ...p.mesh, local_edge_sizes: entries.filter((_, i) => i !== index) } });
  };

  // コレクタ一覧 (PICパネル) の1件を更新する (ラベル・tol の編集)
  const updateCollector = (index: number, patch: Partial<PicCollectorSettings>) => {
    const collectors = pic.collectors ?? [];
    if (index < 0 || index >= collectors.length) return;
    const next = collectors.slice();
    next[index] = { ...next[index], ...patch };
    setPic({ ...pic, collectors: next });
  };

  // コレクタ一覧の1件を削除する
  const deleteCollector = (index: number) => {
    const collectors = pic.collectors ?? [];
    if (index < 0 || index >= collectors.length) return;
    setPic({ ...pic, collectors: collectors.filter((_, i) => i !== index) });
    setSelectedCollectorIndexRaw((sel) => (sel === index ? null : sel));
  };

  // EEDF領域配置ツール (CadCanvas) からの確定通知。矩形ツールと同じ2点クリックUXだが、
  // 領域一覧の格納形式はコレクタと同じ「対角の2点そのまま」で持つ (backend は
  // min/max で軸平行矩形として解釈する)。最大 MAX_EEDF_REGIONS 個、達したら追加しない。
  // ラベルは "E1","E2",... を自動採番。配置したら PIC インスペクタページへ切替え、
  // 追加した領域を選択状態にする
  const setEedfRegionPoints = (p1: Point, p2: Point) => {
    const regions = pic.eedf_regions ?? [];
    if (regions.length < MAX_EEDF_REGIONS) {
      const label = nextEedfLabel(regions);
      // bins/e_max_ev はバックエンドの既定 (100 / 自動決定) に合わせて明示しておく
      // (コレクタの tol: null と同じ流儀)
      const next = [...regions, { p1, p2, label, bins: 100, e_max_ev: null }];
      setPic({ ...pic, eedf_regions: next });
      setSelectedEedfIndexRaw(next.length - 1);
    }
    setActiveNode("study-pic");
  };

  // EEDF領域一覧 (PICパネル) の1件を更新する (ラベル・bins・e_max_ev の編集)
  const updateEedfRegion = (index: number, patch: Partial<PicEedfRegionSettings>) => {
    const regions = pic.eedf_regions ?? [];
    if (index < 0 || index >= regions.length) return;
    const next = regions.slice();
    next[index] = { ...next[index], ...patch };
    setPic({ ...pic, eedf_regions: next });
  };

  // EEDF領域一覧の1件を削除する
  const deleteEedfRegion = (index: number) => {
    const regions = pic.eedf_regions ?? [];
    if (index < 0 || index >= regions.length) return;
    setPic({ ...pic, eedf_regions: regions.filter((_, i) => i !== index) });
    setSelectedEedfIndexRaw((sel) => (sel === index ? null : sel));
  };

  // シースエッジ評価ライン配置ツール (CadCanvas) からの確定通知 (prompts/98)。コレクタと
  // 同じ2点クリックUX、最大 MAX_SHEATH_LINES 本、達したら追加しない。ラベルは "S1","S2",...
  // を自動採番。collectors/eedf_regions と同じく pic (project とは独立の state) を
  // setPic で直接更新するだけなので、commitProject を経由せず解析結果 (picFields/picCycle
  // 等) を一切破棄しない (可視化専用の設定であることと自然に一致する)
  const setSheathLinePoints = (p1: Point, p2: Point) => {
    const lines = pic.sheath_lines ?? [];
    if (lines.length < MAX_SHEATH_LINES) {
      const label = nextSheathLabel(lines);
      setPic({ ...pic, sheath_lines: [...lines, { p1, p2, label }] });
    }
    setActiveNode("study-pic");
  };

  // シースエッジ評価ライン一覧 (PICパネル) の1件のラベルを更新する
  const updateSheathLineLabel = (index: number, label: string) => {
    const lines = pic.sheath_lines ?? [];
    if (index < 0 || index >= lines.length) return;
    const next = lines.slice();
    next[index] = { ...next[index], label };
    setPic({ ...pic, sheath_lines: next });
  };

  // シースエッジ評価ライン一覧の1件を削除する
  const deleteSheathLine = (index: number) => {
    const lines = pic.sheath_lines ?? [];
    if (index < 0 || index >= lines.length) return;
    setPic({ ...pic, sheath_lines: lines.filter((_, i) => i !== index) });
  };

  // キャンバス上で領域を選択したら「領域」インスペクタページに切替える (選択解除時は切替しない)
  const selectRegionFromCanvas = (id: string | null) => {
    setSelectedRegionId(id);
    if (id !== null) setActiveNode("regions");
  };

  // プロジェクトツリーでのノード選択。boundary 以外へ移動したら辺フィルタは解除する
  // ("result-fem" 選択時に fieldView/tool を切替える副作用は廃止。統一後は表示切替・
  // プロファイル起動ともにページ内 UI から行うため、ノード選択自体には副作用を持たせない、prompts/69)
  const selectNode = (node: TreeNode) => {
    setActiveNode(node);
    if (node !== "boundary") setEdgeFilter(null);
  };

  // プロジェクトツリーで境界条件の子ノード (辺) をクリックした場合
  const selectEdgeNode = (edgeIndex: number) => {
    setActiveNode("boundary");
    setEdgeFilter(edgeIndex);
  };

  // プロジェクトツリーで領域の子ノードをクリックした場合
  const selectRegionNode = (id: string) => {
    setSelectedRegionId(id);
    setActiveNode("regions");
  };

  // --- domain ---
  const domainW = Math.max(...project.geometry.domain.polygon.map((p) => p[0]));
  const domainH = Math.max(...project.geometry.domain.polygon.map((p) => p[1]));
  // 軸対称 (r-z または r-z_x0) モードかどうか (未指定 = "xy" 扱い)
  const isAxisym = isAxisymmetric(project.coord);

  const setDomainSize = (w: number, h: number) => {
    if (!(w > 0) || !(h > 0)) return;
    const p = projectRef.current;
    commitProject({
      ...p,
      geometry: {
        ...p.geometry,
        domain: { polygon: [[0, 0], [w, 0], [w, h], [0, h]] },
      },
    });
  };

  // --- 境界条件 (4辺: 0=下,1=右,2=上,3=左)。矩形domain前提で対辺は (i+2)%4 ---
  const oppositeEdge = (edgeIndex: number) => (edgeIndex + 2) % 4;

  // 指定エッジを含むBCエントリを取り除く。periodicエントリは2辺セットで消えるため、
  // 対辺も道連れで自然境界(Neumann)に戻る
  const removeEdgeBoundary = (boundaries: BoundaryCondition[], edgeIndex: number): BoundaryCondition[] =>
    boundaries.filter((b) => !b.edges.includes(edgeIndex));

  // --- 座標系 (平面2D / 軸対称 r-z (下辺軸) / 軸対称 r-z (左辺軸)) 切替 ---
  // 座標系ごとに対称軸となるエッジ番号 (rz: 下辺=0、rz_x0: 左辺=3。xy は該当なし)
  const axisEdgeIndex = (coord: "xy" | "rz" | "rz_x0"): number | null => {
    if (coord === "rz") return 0;
    if (coord === "rz_x0") return 3;
    return null;
  };

  // 軸対称モードへ切替える際、対称軸となる辺には Dirichlet 等の指定は禁止されるため、
  // 既存のBC設定があれば道連れで除去する (commitProject 経由なので Undo 対象・解析結果は破棄される)
  const setCoord = (coord: "xy" | "rz" | "rz_x0") => {
    const p = projectRef.current;
    const axisEdge = axisEdgeIndex(coord);
    const boundaries = axisEdge !== null ? removeEdgeBoundary(p.geometry.boundaries, axisEdge) : p.geometry.boundaries;
    commitProject({ ...p, coord, geometry: { ...p.geometry, boundaries } });
  };

  const edgeState = (
    edgeIndex: number,
  ): {
    type: EdgeBcType;
    voltage: number;
    voltageRf?: VoltageRf | VoltageRf[];
    voltageWaveform?: VoltageWaveform;
    seeGamma: number;
  } => {
    const b = project.geometry.boundaries.find((b) => b.edges.includes(edgeIndex));
    if (!b) return { type: "neumann", voltage: 0, seeGamma: 0 };
    if (b.type === "dirichlet") {
      return {
        type: "dirichlet",
        voltage: b.voltage,
        voltageRf: b.voltage_rf,
        voltageWaveform: b.voltage_waveform,
        seeGamma: b.see_gamma ?? 0,
      };
    }
    return { type: b.type, voltage: 0, seeGamma: 0 };
  };

  // 境界条件タイプの一元切替ハンドラ。周期を選ぶと対辺も自動的に周期エントリへまとめ、
  // 対辺が別タイプ(Dirichlet等)で使用中でも単純に上書きする。他タイプへ切替えた場合、
  // 元が周期エントリであれば removeEdgeBoundary により対辺も道連れで解除される
  const setEdgeType = (edgeIndex: number, type: EdgeBcType) => {
    const p = projectRef.current;
    let boundaries = removeEdgeBoundary(p.geometry.boundaries, edgeIndex);
    if (type === "dirichlet") {
      boundaries = [...boundaries, { edges: [edgeIndex], type: "dirichlet" as const, voltage: 0 }];
    } else if (type === "symmetry") {
      boundaries = [...boundaries, { edges: [edgeIndex], type: "symmetry" as const }];
    } else if (type === "periodic") {
      const opposite = oppositeEdge(edgeIndex);
      boundaries = [
        ...removeEdgeBoundary(boundaries, opposite),
        { edges: [edgeIndex, opposite], type: "periodic" as const },
      ];
    }
    commitProject({ ...p, geometry: { ...p.geometry, boundaries } });
  };

  // Dirichlet辺の電圧値のみ更新 (まだDirichletでなければ新規作成する)
  const setEdgeVoltage = (edgeIndex: number, voltage: number) => {
    const p = projectRef.current;
    const cur = p.geometry.boundaries.find((b) => b.edges.includes(edgeIndex));
    const boundaries =
      cur && cur.type === "dirichlet"
        ? p.geometry.boundaries.map((b) => (b === cur ? { ...b, voltage } : b))
        : [...removeEdgeBoundary(p.geometry.boundaries, edgeIndex), { edges: [edgeIndex], type: "dirichlet" as const, voltage }];
    commitProject({ ...p, geometry: { ...p.geometry, boundaries } });
  };

  // 境界条件のRF重畳設定 (対象エッジが Dirichlet でない場合は何もしない)。
  // 複数成分 (デュアル周波数) は配列で渡す
  const setEdgeVoltageRf = (edgeIndex: number, voltage_rf: VoltageRf | VoltageRf[] | undefined) => {
    const p = projectRef.current;
    commitProject({
      ...p,
      geometry: {
        ...p.geometry,
        boundaries: p.geometry.boundaries.map((b) =>
          b.type === "dirichlet" && b.edges.includes(edgeIndex) ? { ...b, voltage_rf } : b,
        ),
      },
    });
  };

  // 境界条件のCSV波形設定 (対象エッジが Dirichlet でない場合は何もしない、prompts/73)。
  // voltage_rf と併用可 (V(t) = voltage + Σ RF + V_wf(t))。undefined で解除
  const setEdgeVoltageWaveform = (edgeIndex: number, voltage_waveform: VoltageWaveform | undefined) => {
    const p = projectRef.current;
    commitProject({
      ...p,
      geometry: {
        ...p.geometry,
        boundaries: p.geometry.boundaries.map((b) =>
          b.type === "dirichlet" && b.edges.includes(edgeIndex) ? { ...b, voltage_waveform } : b,
        ),
      },
    });
  };

  // 境界条件の二次電子放出係数 γ (対象エッジが Dirichlet でない場合は何もしない)
  const setEdgeSeeGamma = (edgeIndex: number, see_gamma: number) => {
    const p = projectRef.current;
    commitProject({
      ...p,
      geometry: {
        ...p.geometry,
        boundaries: p.geometry.boundaries.map((b) =>
          b.type === "dirichlet" && b.edges.includes(edgeIndex) ? { ...b, see_gamma } : b,
        ),
      },
    });
  };

  // --- メッシュ ---
  const setMeshSize = (size: number) => {
    if (!(size > 0)) return;
    const p = projectRef.current;
    commitProject({ ...p, mesh: { ...p.mesh, size } });
  };

  // メッシュモード (非構造 gmsh / 構造格子)。構造格子は矩形 domain のみ対応
  const setMeshMode = (mode: "unstructured" | "structured") => {
    const p = projectRef.current;
    commitProject({ ...p, mesh: { ...p.mesh, mode } });
  };

  // --- 一様磁場 (prompts/51) ---
  // 全成分0なら b_field を undefined にする (バックエンドでは磁場なしと同値)
  const setBField = (patch: Partial<BField>) => {
    const p = projectRef.current;
    const cur = p.b_field ?? { bx: 0, by: 0, bz: 0 };
    const next = { ...cur, ...patch };
    const isZero = next.bx === 0 && next.by === 0 && next.bz === 0;
    commitProject({ ...p, b_field: isZero ? undefined : next });
  };

  // --- 領域 ---
  // ポリゴン (ポリライン/矩形ツール) または circle shape (円ツール) のどちらでも領域を追加できる
  const addRegion = (geom: Point[] | CircleShape) => {
    const p = projectRef.current;
    const ids = new Set(p.geometry.regions.map((r) => r.id));
    let n = p.geometry.regions.length + 1;
    let id = `region${n}`;
    while (ids.has(id)) { n += 1; id = `region${n}`; }
    const region: Region = Array.isArray(geom)
      ? { id, type: "conductor", polygon: geom, voltage: 0 }
      : { id, type: "conductor", shape: geom, voltage: 0 };
    commitProject({ ...p, geometry: { ...p.geometry, regions: [...p.geometry.regions, region] } });
  };

  const updateRegion = (id: string, patch: Partial<Region>) => {
    const p = projectRef.current;
    commitProject({
      ...p,
      geometry: {
        ...p.geometry,
        regions: p.geometry.regions.map((r) => (r.id === id ? { ...r, ...patch } : r)),
      },
    });
  };

  const renameRegion = (oldId: string, newId: string) => {
    if (!newId || newId === oldId) return;
    const p = projectRef.current;
    if (p.geometry.regions.some((r) => r.id === newId)) return; // ID 重複は不可
    commitProject({
      ...p,
      geometry: {
        ...p.geometry,
        regions: p.geometry.regions.map((r) => (r.id === oldId ? { ...r, id: newId } : r)),
      },
      // ローカルメッシュサイズの参照キーも追従させる
      mesh: {
        ...p.mesh,
        local_sizes: (p.mesh.local_sizes ?? []).map((ls) =>
          ls.region === oldId ? { ...ls, region: newId } : ls,
        ),
      },
    });
    setSelectedRegionId(newId);
  };

  // 領域ごとのローカルメッシュサイズ [m]。null で解除 (全体サイズを使用)。非構造メッシュのみ有効
  const setRegionLocalSize = (id: string, size: number | null) => {
    const p = projectRef.current;
    const rest = (p.mesh.local_sizes ?? []).filter((ls) => ls.region !== id);
    const local_sizes = size !== null && size > 0 ? [...rest, { region: id, size }] : rest;
    commitProject({ ...p, mesh: { ...p.mesh, local_sizes } });
  };

  const setRegionType = (id: string, type: RegionType) => {
    const p = projectRef.current;
    commitProject({
      ...p,
      geometry: {
        ...p.geometry,
        regions: p.geometry.regions.map((r) => {
          if (r.id !== id) return r;
          // shape (circle) 領域はそのまま shape を維持し、polygon 領域は polygon を維持する
          const base = r.shape
            ? { id: r.id, type, shape: r.shape }
            : { id: r.id, type, polygon: r.polygon ?? [] };
          if (type === "conductor") return { ...base, voltage: r.voltage ?? 0 };
          if (type === "dielectric") return { ...base, eps_r: r.eps_r ?? 1 };
          return { ...base, rho: r.rho ?? 0 };
        }),
      },
    });
  };

  const deleteRegion = (id: string) => {
    const p = projectRef.current;
    commitProject({
      ...p,
      geometry: { ...p.geometry, regions: p.geometry.regions.filter((r) => r.id !== id) },
      // 削除された領域のローカルメッシュサイズ設定も掃除する
      mesh: {
        ...p.mesh,
        local_sizes: (p.mesh.local_sizes ?? []).filter((ls) => ls.region !== id),
      },
    });
    setSelectedRegionId((sel) => (sel === id ? null : sel));
  };

  // --- 図形の移動 (CadCanvas からのドラッグ確定 / 矢印キー微動) ---
  // polygon 領域は各頂点を、circle (shape) 領域は中心を平行移動する
  const moveRegion = (id: string, dx: number, dy: number) => {
    if (dx === 0 && dy === 0) return;
    const p = projectRef.current;
    if (!p.geometry.regions.some((r) => r.id === id)) return;
    commitProject({
      ...p,
      geometry: {
        ...p.geometry,
        regions: p.geometry.regions.map((r) => {
          if (r.id !== id) return r;
          if (r.shape) {
            return {
              ...r,
              shape: { ...r.shape, center: [r.shape.center[0] + dx, r.shape.center[1] + dy] },
            };
          }
          return { ...r, polygon: (r.polygon ?? []).map(([x, y]) => [x + dx, y + dy] as Point) };
        }),
      },
    });
  };

  // --- 領域の多角形編集 (CadCanvas からの頂点/中点グリップ操作の確定) ---
  const editRegionPolygon = (id: string, polygon: Point[]) => {
    if (polygon.length < 3) return;
    const p = projectRef.current;
    if (!p.geometry.regions.some((r) => r.id === id)) return;
    commitProject({
      ...p,
      geometry: {
        ...p.geometry,
        regions: p.geometry.regions.map((r) => (r.id === id ? { ...r, polygon } : r)),
      },
    });
  };

  // --- circle 領域の shape 編集 (CadCanvas からの半径グリップ操作の確定、サイドパネルの数値入力共通) ---
  const editRegionShape = (id: string, shape: CircleShape) => {
    if (!(shape.radius > 0)) return;
    const p = projectRef.current;
    if (!p.geometry.regions.some((r) => r.id === id)) return;
    commitProject({
      ...p,
      geometry: {
        ...p.geometry,
        regions: p.geometry.regions.map((r) => (r.id === id ? { ...r, shape } : r)),
      },
    });
  };

  // --- 隣接領域のマージ (prompts/63) ---
  // 領域 otherId を targetId へマージする (union)。成功時は target の polygon を
  // union 結果へ差し替え、other を削除する (プロパティは target 側を維持)。
  // 失敗時 (非隣接で1つにならない / 穴ができる) はエラーメッセージ文字列を返す
  const mergeRegions = (targetId: string, otherId: string): string | null => {
    const p = projectRef.current;
    const target = p.geometry.regions.find((r) => r.id === targetId);
    const other = p.geometry.regions.find((r) => r.id === otherId);
    if (!target || !other) return "領域が見つかりません";
    if (!target.polygon || !other.polygon) return "多角形領域のみマージできます";

    let unionResult: MultiPolygon;
    try {
      // polygon-clipping の Polygon 型はリング(穴なし単一外周)の配列。
      // 単一輪郭のみを扱うためそれぞれ1リングの Polygon として渡す
      unionResult = polygonClipping.union([target.polygon], [other.polygon]);
    } catch (err) {
      return `マージに失敗しました: ${String(err)}`;
    }
    if (unionResult.length !== 1) return "領域が隣接していないためマージできません";
    const poly = unionResult[0];
    if (poly.length > 1) return "マージ結果に穴ができるためマージできません";
    const mergedPolygon = removeCollinearVertices(poly[0]);

    commitProject({
      ...p,
      geometry: {
        ...p.geometry,
        regions: p.geometry.regions
          .filter((r) => r.id !== otherId)
          .map((r) => (r.id === targetId ? { ...r, polygon: mergedPolygon } : r)),
      },
      // マージで消える other への参照 (deleteRegion と同様の掃除) を落とす
      mesh: {
        ...p.mesh,
        local_sizes: (p.mesh.local_sizes ?? []).filter((ls) => ls.region !== otherId),
      },
    });
    return null;
  };

  // --- 保存/読込 ---
  // particles / pic / pic1d / fluid1d / tl は history 管理外の別 state のため、保存時にここで
  // project へ合成する
  const saveProject = () => {
    const toSave: Project = {
      ...project,
      particles,
      pic: withInjectionEmitter(pic, particles.emitter),
      pic1d,
      fluid1d,
      fluid2d,
      tl,
    };
    saveTextFile("project.json", JSON.stringify(toSave, null, 2), "JSON", ["json"]).catch((err) => {
      setError(String(err));
    });
  };

  // 結果付き保存: プロジェクトに results を同梱して1ファイルで保存する。
  // 結果 (特に PIC の cycle) は大きくなりうるため、整形なし (compact) で書き出す
  const saveProjectWithResults = () => {
    const toSave: Project = {
      ...project,
      particles,
      pic: withInjectionEmitter(pic, particles.emitter),
      pic1d,
      fluid1d,
      fluid2d,
      tl,
    };
    // pic 結果は picStarted (mesh を含む) が無いと描画できないため、それが無い場合は同梱しない
    const results: ResultsBundle = {
      version: 1,
      solve: result,
      mesh: meshResult,
      trace: traceResult,
      pic: picStarted
        ? {
            started: picStarted,
            frame: picFrame,
            history: picHistory,
            fields: picFields,
            cycle: picCycle,
            collectors: picCollectors,
            eedf: picEedf,
            elapsed_s: picElapsedS ?? undefined, // run_batch の壁時計秒 (prompts/86)
          }
        : null,
      gas: gasResult,
      // pic1d 結果は settings を含み自己完結 (done.result そのもの) なので、そのまま同梱する
      pic1d: pic1dResult,
      // fluid1d 結果も同様に settings を含み自己完結 (done.result そのもの、prompts/104-109)
      fluid1d: fluid1dResult,
      // fluid2d 結果も同様に settings を含み自己完結 (done.result そのもの)。mesh は含まない
      // (フィールド描画には上記 meshResult を流用する、prompts/111-113)
      fluid2d: fluid2dResult,
      // tl 結果も同様に settings を含み自己完結 (done.result そのもの、prompts/101)
      tl: tlResult,
    };
    saveTextFile("project_results.json", JSON.stringify({ ...toSave, results }), "JSON", ["json"]).catch((err) => {
      setError(String(err));
    });
  };

  // 「結果付き保存」ファイル (または通常のプロジェクトファイル) の中身を state へ適用する。
  // ファイル読込 (loadProject) とスイープのケース読込 (loadSweepCase、prompts/79) の両方から
  // 共用する (App.tsx の関数抽出、prompts/79)。不正な形式は例外を投げるので呼び出し側で catch すること
  const applyLoadedProject = useCallback((obj: unknown) => {
    if (!obj || typeof obj !== "object" || !("geometry" in obj)) {
      throw new Error("不正なプロジェクトファイルです (geometry がありません)");
    }
    // 「結果付き保存」ファイルは project 本体に results (ResultsBundle) を同梱している。
    // results はフロント専用フィールドで、以後 solve 等の API へ送る project に紛れ込んではいけない
    // ため、project 部分 (projectOnly) と分離してから従来どおりの補完処理にかける
    const { results, ...projectOnly } = obj as Project & { results?: ResultsBundle };
    // 省略可能フィールドを既定値で補完する。backend の pydantic スキーマは
    // regions / boundaries 等を省略可 (既定 []) としており、手書き・サンプルの
    // JSON では欠けていることがある (欠けたまま state に入れると .find 等で落ちる)
    const raw = projectOnly as Project;
    const loaded: Project = {
      ...raw,
      geometry: {
        domain: raw.geometry.domain ?? { polygon: [] },
        regions: raw.geometry.regions ?? [],
        boundaries: raw.geometry.boundaries ?? [],
      },
      mesh: { ...SAMPLE.mesh, ...(raw.mesh ?? {}) },
    };
    commitProject(loaded);
    // particles / pic は独立管理の state なので、読込んだファイルにあれば反映し、なければ既定値に戻す
    const loadedParticles = raw.particles;
    // FN 専用プロジェクト (fn_diode.json 等) は emitter を省略できる (スキーマ上
    // fn 指定時は emitter 不要) ため、既定値をベースに合成して欠損フィールドを
    // 補完する (emitter が無いまま state に入れると UI が .p1 等の参照で落ちる)
    setParticles(loadedParticles ? { ...DEFAULT_PARTICLES, ...loadedParticles } : DEFAULT_PARTICLES);
    const loadedPic = raw.pic;
    // mcc/see_energy_ev が無い旧形式のファイルでも安全に読み込めるよう、既定値をベースに合成し、
    // 旧形式 (単数 collector) のプロジェクトは collectors 配列へ移行する
    setPic(loadedPic ? normalizeCollectors({ ...DEFAULT_PIC, ...loadedPic }) : DEFAULT_PIC);
    // 1D PIC設定も同様に既定値をベースに合成する (pic1d が無い旧形式ファイルは丸ごと既定値に戻す)
    const loadedPic1d = raw.pic1d;
    setPic1d(loadedPic1d ? { ...DEFAULT_PIC1D, ...loadedPic1d } : DEFAULT_PIC1D);
    // 1D 流体設定も同様 (fluid1d が無い旧形式ファイルは丸ごと既定値に戻す、prompts/104-109)
    const loadedFluid1d = raw.fluid1d;
    setFluid1d(loadedFluid1d ? { ...DEFAULT_FLUID1D, ...loadedFluid1d } : DEFAULT_FLUID1D);
    // 2D 流体設定も同様 (fluid2d が無い旧形式ファイルは丸ごと既定値に戻す、prompts/111-113)
    const loadedFluid2d = raw.fluid2d;
    setFluid2d(loadedFluid2d ? { ...DEFAULT_FLUID2D, ...loadedFluid2d } : DEFAULT_FLUID2D);
    // VHF 定在波設定も同様 (tl が無い旧形式ファイルは丸ごと既定値に戻す、prompts/101)
    const loadedTl = raw.tl;
    setTl(loadedTl ? { ...DEFAULT_TL, ...loadedTl } : DEFAULT_TL);
    setSelectedCollectorIndexRaw(null);
    setSelectedEedfIndexRaw(null);
    setSelectedRegionId(null);
    // 結果付き保存ファイルなら計算結果も復元する (commitProject による結果クリアの後に上書きする)
    if (results) {
      setResult(results.solve ?? null);
      setMeshResult(results.mesh ?? null);
      setTraceResult(results.trace ?? null);
      setGasResult(results.gas ?? null);
      setPicStarted(results.pic?.started ?? null);
      setPicFrame(results.pic?.frame ?? null);
      setPicHistory(results.pic?.history ?? []);
      setPicFields(results.pic?.fields ?? null);
      setPicCycle(results.pic?.cycle ?? null);
      setPicCollectors(results.pic?.collectors ?? []);
      setPicEedf(results.pic?.eedf ?? []);
      setPicElapsedS(results.pic?.elapsed_s ?? null); // run_batch の壁時計秒 (旧形式ファイルには無い、prompts/86)
      // 結果表示セレクトは既定 (ライブ/線形) へ戻す
      setPicResultField("live");
      setPicLogScale(false);
      // 1D PIC の結果 (settings を含み自己完結)。started/frame は保存対象に含めていないため
      // 常に null に戻す (result-pic1d ページはこの pic1dResult の有無だけで結果表示に切り替わる)
      setPic1dStarted(null);
      setPic1dFrame(null);
      setPic1dResult(results.pic1d ?? null);
      setPic1dContinueReady(false); // サーバー側に保持状態が無いため続き実行は無効にする
      // 1D 流体の結果 (settings を含み自己完結)。started/frame は保存対象に含めていないため
      // 常に null に戻す (result-fluid1d ページはこの fluid1dResult の有無だけで結果表示に切り替わる)
      setFluid1dStarted(null);
      setFluid1dFrame(null);
      setFluid1dResult(results.fluid1d ?? null);
      setFluid1dContinueReady(false); // サーバー側に保持状態が無いため続き実行は無効にする
      // 2D 流体の結果 (settings を含み自己完結)。started/frame は保存対象に含めていないため
      // 常に null に戻す (result-fluid2d ページはこの fluid2dResult の有無だけで結果表示に切り替わる)
      setFluid2dStarted(null);
      setFluid2dFrame(null);
      setFluid2dResult(results.fluid2d ?? null);
      setFluid2dContinueReady(false); // サーバー側に保持状態が無いため続き実行は無効にする
      setFluid2dProjectChangedSinceRun(true);
      // 結果表示セレクト・周期アニメの再生系も既定値へ戻す (フィールド選択・データ自体は
      // 復元済みの値を保つ、2D PIC の picCycle 系リセットと同じ考え方)
      setFluid2dResultField("live");
      setFluid2dLogScale(false);
      setFluid2dCyclePlaying(false);
      setFluid2dCycleBinIndex(0);
      setFluid2dCycleViewActive(false);
      // VHF 定在波の結果 (settings を含み自己完結)。started/progress は保存対象に含めていないため
      // 常に null に戻す (result-tl ページはこの tlResult の有無だけで結果表示に切り替わる、prompts/101)
      setTlStarted(null);
      setTlProgress(null);
      setTlResult(results.tl ?? null);
      // サーバーには読込んだ状態が存在しない (このセッションで実行していない) ため、
      // 「続きから実行」は無効にし、次の新規実行を促す
      setPicContinueReady(false);
      setPicProjectChangedSinceRun(true);
      // 周期アニメーションの再生系も既定値へ戻す (フィールド選択・データ自体は復元済みの値を保つ)
      setCyclePlaying(false);
      setCycleBinIndex(0);
      setCycleViewActive(false);
    }
    // 直接読み込んだ場合はスイープのケースとの紐付けが失われるためハイライトを解除する
    // (loadSweepCase が読込直後に自分の index で setSweepLoadedCaseIndex を上書きする)
    setSweepLoadedCaseIndex(null);
  }, [commitProject]);

  const loadProject = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const obj = JSON.parse(String(reader.result));
        applyLoadedProject(obj);
        setError(null);
      } catch (err) {
        setError(String(err));
      }
    };
    reader.readAsText(file);
    e.target.value = ""; // 同じファイルを連続で読み込めるようにする
  };

  // スイープのケース一覧 (SweepPanel) の行クリック: GET /sweep/result/{i} を取得し、
  // applyLoadedProject で通常のファイル読込と同じ復元処理を行う
  // (commitProject 経由なので Undo で元のプロジェクトに戻せる、prompts/79)
  const loadSweepCase = async (index: number) => {
    try {
      const obj = await api.sweepResult(index);
      applyLoadedProject(obj);
      setSweepLoadedCaseIndex(index);
      setError(null);
    } catch (err) {
      setError(String(err));
    }
  };

  const selected = project.geometry.regions.find((r) => r.id === selectedRegionId) ?? null;

  // 「結果付き保存」ボタンの有効条件: FEM/Mesh/トレース/PIC/PIC-MCC 1D/DSMC のいずれかの結果があること
  // (何も無い状態で保存しても project 部分だけの通常保存と同じになってしまうため無効化する)
  const hasAnyResults =
    !!result ||
    !!meshResult ||
    !!traceResult ||
    !!gasResult ||
    !!picStarted ||
    !!pic1dResult ||
    !!fluid1dResult ||
    !!fluid2dResult ||
    !!tlResult;

  // 「続きから実行」ボタンの有効条件: 直前の実行が done/stop 済みで現在実行中でなく、
  // かつ前回実行以降にジオメトリが編集されていないこと (health 未接続時も不可)
  const picCanContinue = !!health && picContinueReady && !picRunning && !picProjectChangedSinceRun;

  // PIC-MCC 1D の「続きから」有効条件。pic1d は geometry/mesh に依存しないため、
  // 2D のような「ジオメトリ食い違いで無効化」の判定は不要 (pic1dContinueReady のコメント参照)
  const pic1dCanContinue = !!health && pic1dContinueReady && !pic1dRunning;

  // 1D 流体の「続きから」有効条件。pic1dCanContinue と同じ考え方
  const fluid1dCanContinue = !!health && fluid1dContinueReady && !fluid1dRunning;

  // 2D 流体の「続きから」有効条件。ジオメトリに依存するため picCanContinue と同じ考え方
  const fluid2dCanContinue =
    !!health && fluid2dContinueReady && !fluid2dRunning && !fluid2dProjectChangedSinceRun;

  // DSMC「続きから実行」ボタンの有効条件 (PIC の picCanContinue と同じ考え方)
  const gasCanContinue = !!health && gasContinueReady && !gasRunning && !gasProjectChangedSinceRun;

  // 配置済み IEDF/IADF コレクタ線分一覧 (CadCanvas への常時オーバーレイ表示用)。
  // label が未設定 (旧データ等) でも表示できるよう "C<n>" のフォールバックを与える
  const collectorsList: PicCollectorView[] = (pic.collectors ?? []).map((c, i) => ({
    p1: c.p1,
    p2: c.p2,
    label: c.label && c.label.trim() !== "" ? c.label : `C${i + 1}`,
  }));

  // 配置済み DSMC 線分境界一覧 (CadCanvas への常時オーバーレイ表示用、prompts/72)。
  // edges のみ指定でp1/p2を持たない境界は線分として描けないため対象外 (ラベルは線分境界のみで連番)
  const gasBoundariesList: GasBoundaryView[] = (project.dsmc?.boundaries ?? [])
    .filter((b) => b.p1 != null && b.p2 != null)
    .map((b, i) => ({ p1: b.p1 as Point, p2: b.p2 as Point, label: `G${i + 1}` }));

  // 配置済み辺ローカルメッシュサイズ一覧 (CadCanvas への常時オーバーレイ表示用、prompts/90)。
  // ラベルは schema に持たせず (backend には保存しない)、コレクタ・ガス境界と同じ流儀で
  // インデックスから "M1","M2",... を振る
  const edgeMeshSizesList: EdgeMeshSizeView[] = (project.mesh.local_edge_sizes ?? []).map((e, i) => ({
    p1: e.p1,
    p2: e.p2,
    label: `M${i + 1}`,
  }));

  // コレクタ一覧の選択インデックス (範囲外・未選択なら先頭を既定選択とする)
  const selectedCollectorIndex: number | null =
    selectedCollectorIndexRaw !== null && selectedCollectorIndexRaw < collectorsList.length
      ? selectedCollectorIndexRaw
      : collectorsList.length > 0
        ? 0
        : null;

  // 配置済み EEDF/EEPF 領域一覧 (CadCanvas への常時オーバーレイ表示用、prompts/85)。
  // label が未設定でも表示できるよう "E<n>" のフォールバックを与える
  const eedfRegionsList: PicEedfRegionView[] = (pic.eedf_regions ?? []).map((r, i) => ({
    p1: r.p1,
    p2: r.p2,
    label: r.label && r.label.trim() !== "" ? r.label : `E${i + 1}`,
  }));

  // 領域一覧の選択インデックス (範囲外・未選択なら先頭を既定選択とする、コレクタと同じ流儀)
  const selectedEedfIndex: number | null =
    selectedEedfIndexRaw !== null && selectedEedfIndexRaw < eedfRegionsList.length
      ? selectedEedfIndexRaw
      : eedfRegionsList.length > 0
        ? 0
        : null;

  // PICライブ描画用ビュー (started の mesh + 最新 frame)。実行中〜done後の最終フレームまで保持する。
  // 表示フィールド切替 (prompts/81): picLiveField が n_e/n_i でも、frame 側にその配列が無い
  // (旧バックエンド) 場合は phi (節点値) にフォールバックする
  const picLiveFrame: PicLiveFrame | null =
    picStarted && picFrame
      ? (() => {
          const density = picLiveField === "n_e" ? picFrame.n_e : picLiveField === "n_i" ? picFrame.n_i : undefined;
          const useDensity = picLiveField !== "phi" && density !== undefined;
          return {
            mesh: picStarted.mesh,
            values: useDensity ? (density as number[]) : picFrame.phi,
            nodeBased: !useDensity, // phi=節点値、n_e/n_i=要素値
            unit: useDensity ? "m^-3" : "V",
            log: useDensity && picLiveLogScale,
            particles: picFrame.particles,
          };
        })()
      : null;

  // PIC結果フィールド表示用ビュー (「結果表示」セレクトでライブ以外を選び、fields がある場合のみ)。
  // CadCanvas へは値配列・節点/要素の別・単位・対数フラグを1つの prop にまとめて渡す
  const picFieldView: PicFieldView | null =
    picResultField !== "live" && picFields && picStarted
      ? {
          mesh: picStarted.mesh,
          values: picFields[picResultField],
          nodeBased: PIC_FIELD_META[picResultField].nodeBased,
          unit: PIC_FIELD_META[picResultField].unit,
          log: picLogScale,
        }
      : null;

  // 周期アニメーション用のカラースケール固定範囲 (全ビンの min/max)。
  // フレーム(ビン)が変わるたびに色が暴れないよう、選択フィールドが変わったときだけ再計算する
  const cycleFixedRange = useMemo(() => {
    if (!picCycle) return null;
    const rows = picCycle[cycleField];
    if (!rows) return null; // 旧バックエンド等でこのフィールドのデータが無い
    let min = Infinity;
    let max = -Infinity;
    let minPositive = Infinity;
    for (const row of rows) {
      for (const v of row) {
        if (v < min) min = v;
        if (v > max) max = v;
        if (v > 0 && v < minPositive) minPositive = v;
      }
    }
    if (!Number.isFinite(min)) { min = 0; max = 0; }
    return { min, max, minPositive };
  }, [picCycle, cycleField]);

  // 周期アニメーション表示用ビュー (done で cycle を受信している間のみ非null)。
  // 現在の位相ビンの値+固定min/max+粒子スナップショットを picFieldView と同形にまとめて渡す
  const picCycleView: PicFieldView | null =
    cycleViewActive && picCycle && picStarted && cycleFixedRange
      ? (() => {
          const rows = picCycle[cycleField];
          if (!rows) return null; // 旧バックエンド等でこのフィールドのデータが無い
          const bin = Math.min(cycleBinIndex, picCycle.bins - 1);
          const meta = PIC_FIELD_META[cycleField];
          return {
            mesh: picStarted.mesh,
            values: rows[bin],
            nodeBased: meta.nodeBased,
            unit: meta.unit,
            log: cycleLogScale,
            fixedRange: cycleFixedRange,
            particles: cycleShowParticles
              ? {
                  electron: picCycle.particles.electron[bin] ?? [],
                  ion: picCycle.particles.ion[bin] ?? [],
                }
              : undefined,
          };
        })()
      : null;

  // 流体 (2D) の done 結果 (fields/cycle)。fluid2dResult が1つのバンドルなので、PIC の
  // ような複数 state への分解はせず、ここで簡潔な別名として取り出すだけにする
  const fluid2dFields = fluid2dResult?.fields ?? null;
  const fluid2dCycle = fluid2dResult?.cycle ?? null;

  // 流体 (2D) ライブ描画用ビュー (picLiveFrame と同じ設計。フレームは phi/n_e/n_i/t_e が
  // 全て全節点値なので nodeBased は常に true、粒子を追わないため particles は常に空にする —
  // CadCanvas 側の picFrame 経路 (drawSpecies に空配列を渡すだけ) をそのまま使うためのダミー)
  const fluid2dLiveFrame: PicLiveFrame | null =
    fluid2dStarted && fluid2dFrame && meshResult
      ? (() => {
          const density =
            fluid2dLiveField === "n_e" ? fluid2dFrame.n_e :
            fluid2dLiveField === "n_i" ? fluid2dFrame.n_i :
            fluid2dLiveField === "t_e" ? fluid2dFrame.t_e : undefined;
          const useDensity = fluid2dLiveField !== "phi" && density !== undefined;
          return {
            mesh: meshResult,
            values: useDensity ? (density as number[]) : fluid2dFrame.phi,
            nodeBased: true,
            unit: useDensity ? FLUID2D_FIELD_META[fluid2dLiveField].unit : "V",
            log: useDensity && fluid2dLiveLogScale,
            particles: { electron: [], ion: [] },
          };
        })()
      : null;

  // 流体 (2D) 結果フィールド表示用ビュー (picFieldView と同じ設計)
  const fluid2dFieldView: PicFieldView | null =
    fluid2dResultField !== "live" && fluid2dFields && meshResult
      ? {
          mesh: meshResult,
          values: fluid2dFields[fluid2dResultField],
          nodeBased: FLUID2D_FIELD_META[fluid2dResultField].nodeBased,
          unit: FLUID2D_FIELD_META[fluid2dResultField].unit,
          log: fluid2dLogScale,
        }
      : null;

  // 流体 (2D) 周期アニメーション用の固定カラースケール (cycleFixedRange と同じ設計)
  const fluid2dCycleFixedRange = useMemo(() => {
    if (!fluid2dCycle) return null;
    const rows = fluid2dCycle[fluid2dCycleField];
    if (!rows) return null;
    let min = Infinity;
    let max = -Infinity;
    let minPositive = Infinity;
    for (const row of rows) {
      for (const v of row) {
        if (v < min) min = v;
        if (v > max) max = v;
        if (v > 0 && v < minPositive) minPositive = v;
      }
    }
    if (!Number.isFinite(min)) { min = 0; max = 0; }
    return { min, max, minPositive };
  }, [fluid2dCycle, fluid2dCycleField]);

  // 流体 (2D) 周期アニメーション表示用ビュー (picCycleView と同じ設計。粒子スナップショットは無い)
  const fluid2dCycleView: PicFieldView | null =
    fluid2dCycleViewActive && fluid2dCycle && meshResult && fluid2dCycleFixedRange
      ? (() => {
          const rows = fluid2dCycle[fluid2dCycleField];
          if (!rows) return null;
          const bin = Math.min(fluid2dCycleBinIndex, fluid2dCycle.bins - 1);
          return {
            mesh: meshResult,
            values: rows[bin],
            nodeBased: true,
            unit: FLUID2D_FIELD_META[fluid2dCycleField].unit,
            log: fluid2dCycleLogScale,
            fixedRange: fluid2dCycleFixedRange,
          };
        })()
      : null;

  // 「結果 — 流体 (2D)」ノード選択中かどうか (onGasNode と同じ役割の判定)
  const onFluid2dNode = activeNode === "study-fluid2d" || activeNode === "result-fluid2d";

  // シースエッジ用の n_e/n_i 密度ソース (prompts/98)。位相アニメ表示中はその現在ビン、
  // そうでなければ時間平均フィールドを使う (picCycleView の density 版という位置づけ)。
  // どちらも無ければ null (等値線・評価ラインの計算そのものをスキップする)。
  // 流体 (2D) ノード選択中は同じ機構を fluid2d 側のデータで解決する (onFluid2dNode で分岐、
  // onGasNode と同じ設計) — シースエッジ等値線が流体フィールドでも自動的に効くようにするための配線
  const sheathSource: SheathDensitySource | null = onFluid2dNode
    ? (fluid2dCycleViewActive && fluid2dCycle && meshResult
        ? (() => {
            const bin = Math.min(fluid2dCycleBinIndex, fluid2dCycle.bins - 1);
            const nE = fluid2dCycle.n_e[bin];
            const nI = fluid2dCycle.n_i[bin];
            if (!nE || !nI) return null;
            return { mesh: meshResult, nE, nI };
          })()
        : fluid2dFields && meshResult
          ? { mesh: meshResult, nE: fluid2dFields.n_e, nI: fluid2dFields.n_i }
          : null)
    : cycleViewActive && picCycle && picStarted
      ? (() => {
          const bin = Math.min(cycleBinIndex, picCycle.bins - 1);
          const nE = picCycle.n_e[bin];
          const nI = picCycle.n_i[bin];
          if (!nE || !nI) return null; // 旧バックエンド等でこのビンのデータが無い
          return { mesh: picStarted.mesh, nE, nI };
        })()
      : picFields && picStarted
        ? { mesh: picStarted.mesh, nE: picFields.n_e, nI: picFields.n_i }
        : null;

  const sheathLinesRaw = pic.sheath_lines ?? [];

  // シースエッジ評価ラインの現在の s (Brinkmann 判定、prompts/98)。ビン別キャッシュ:
  // nE 配列の参照 (picCycle.n_e[bin] はビンごとに固定の配列オブジェクト) を Map の
  // キーにすることで、位相アニメで同じビンへ戻ってきたときは sheathLineEdge の三角形
  // 総当たり探索をやり直さない。sheathSource 自体はビンごとに新規オブジェクトなので
  // useMemo の依存配列比較では「同じビンへの再訪問」を検出できず、ref で自前管理する
  const sheathCacheRef = useRef<{
    mesh: MeshResult | null;
    lines: SheathLineSettings[];
    map: Map<number[], (number | null)[]>;
  }>({ mesh: null, lines: [], map: new Map() });
  let sheathCurrentS: (number | null)[];
  if (!sheathSource || sheathLinesRaw.length === 0) {
    sheathCurrentS = sheathLinesRaw.map(() => null);
  } else {
    const cache = sheathCacheRef.current;
    if (cache.mesh !== sheathSource.mesh || cache.lines !== sheathLinesRaw) {
      cache.mesh = sheathSource.mesh;
      cache.lines = sheathLinesRaw;
      cache.map = new Map();
    }
    const cached = cache.map.get(sheathSource.nE);
    if (cached) {
      sheathCurrentS = cached;
    } else {
      const { mesh, nE, nI } = sheathSource;
      sheathCurrentS = sheathLinesRaw.map((ln) =>
        sheathLineEdge(mesh.nodes, mesh.triangles, nE, nI, ln.p1, ln.p2),
      );
      cache.map.set(nE, sheathCurrentS);
    }
  }

  // シースエッジ評価ラインの位相分解 s(φ) (全ビン、prompts/98)。PicPanel のライン毎
  // s(φ) 折れ線チャート用。picCycle は実行完了 (done受信) 時に1回だけ新しい参照に
  // なるため、同じ実行結果を表示している間はビンを切り替えても再計算しない
  const sheathPhaseCacheRef = useRef<{
    cycle: PicCycle | null;
    mesh: MeshResult | null;
    lines: SheathLineSettings[];
    result: (number | null)[][];
  }>({ cycle: null, mesh: null, lines: [], result: [] });
  let sheathPhaseS: (number | null)[][] = [];
  if (picCycle && picStarted && sheathLinesRaw.length > 0) {
    const cache = sheathPhaseCacheRef.current;
    if (cache.cycle !== picCycle || cache.mesh !== picStarted.mesh || cache.lines !== sheathLinesRaw) {
      const { nodes, triangles } = picStarted.mesh;
      cache.result = sheathLinesRaw.map((ln) => {
        const perBin: (number | null)[] = [];
        for (let b = 0; b < picCycle.bins; b++) {
          const nE = picCycle.n_e[b];
          const nI = picCycle.n_i[b];
          perBin.push(nE && nI ? sheathLineEdge(nodes, triangles, nE, nI, ln.p1, ln.p2) : null);
        }
        return perBin;
      });
      cache.cycle = picCycle;
      cache.mesh = picStarted.mesh;
      cache.lines = sheathLinesRaw;
    }
    sheathPhaseS = cache.result;
  }

  // 配置済みシースエッジ評価ライン一覧 (CadCanvas への常時オーバーレイ表示 + PicPanel
  // 結果セクションの一覧に共用、prompts/98)。label が未設定でも表示できるよう "S<n>" の
  // フォールバックを与える (collectorsList と同じ流儀)
  const sheathLinesList: SheathLineView[] = sheathLinesRaw.map((l, i) => ({
    p1: l.p1,
    p2: l.p2,
    label: l.label && l.label.trim() !== "" ? l.label : `S${i + 1}`,
    s: sheathCurrentS[i] ?? null,
  }));

  // ガス流れ (DSMC) 結果フィールド表示用ビュー。ガス関連ノード (スタディ/結果) を選んでいて
  // PIC 実行中でなく、かつ結果がある場合のみ非null (DSMC の n/t/u/p はすべて要素値なので
  // nodeBased=false 固定)。PicFieldView 型をそのまま流用し、CadCanvas 側の変更は不要にする
  const gasFieldView: PicFieldView | null =
    (activeNode === "study-gas" || activeNode === "result-gas") && !picRunning && gasResult
      ? {
          mesh: gasResult.mesh,
          values: gasFieldValues(gasResult, gasResultField),
          nodeBased: false,
          unit: GAS_FIELD_META[gasResultField].unit,
          log: gasLogScale,
        }
      : null;

  // 描画優先順位: 周期アニメーション > 結果フィールド表示 (PIC/流体2D) > ガス流れ結果表示
  // (> ライブ表示 > Solve結果、CadCanvas側で処理)。
  // CadCanvas へは既存の picFieldView prop をそのまま使い回す (新規propは増やさない)。
  // ガス関連ノード選択中は PIC 側のビュー (周期アニメ・結果フィールド・ライブ) を抑止して
  // ガス結果を優先する (不具合修正: PIC 実行後に「ガス流れ結果」を開いても PIC の結果
  // フィールドが優先チェーンで勝ち続け、DSMC の数密度等が見えなかった)。
  // 流体 (2D) ノード選択中も同じ理由で PIC/ガス側のビューを抑止し、fluid2d 側 (onFluid2dNode、
  // 上記シースエッジ計算のすぐ上で定義済み) を優先する
  const onGasNode = activeNode === "study-gas" || activeNode === "result-gas";
  // PIC-MCC 1D (prompts/91) 選択中は CadCanvas の代わりに Plot1dView を表示する
  // (1D は geometry/mesh と無関係なので CAD キャンバス自体が意味を持たない)
  const isPic1dNode = activeNode === "study-pic1d" || activeNode === "result-pic1d";
  // 1D 流体 (prompts/104-109) も同様に geometry/mesh と無関係な専用ソルバーなので
  // CadCanvas の代わりに Fluid1dPlotView を表示する
  const isFluid1dNode = activeNode === "study-fluid1d" || activeNode === "result-fluid1d";
  // VHF 定在波 (prompts/101) も同様に geometry/mesh と無関係な専用ソルバーなので
  // CadCanvas の代わりに TlPlotView を表示する
  const isTlNode = activeNode === "study-tl" || activeNode === "result-tl";
  // 「結果 — 粒子追跡」ノード選択中は背景表示 (traceBackground) を CadCanvas の
  // result/fieldView に反映する (なし=背景の色マップ・等値線・ベクトルを消す)
  const onTraceResultNode = activeNode === "result-trace";
  const finalPicFieldView = onFluid2dNode
    ? (fluid2dCycleView ?? fluid2dFieldView)
    : onGasNode
      ? gasFieldView
      : (picCycleView ?? picFieldView);

  // コンター配色select・プローブツールの活性化条件 (prompts/103)。CadCanvas へ実際に
  // 渡す result/picFrame と同じ式で「何らかのフィールドが描画されうるか」を近似する
  // (CadCanvas 内部の meshWireTakesPriority — Mesh 実行直後の一時的優先 — までは
  // 追わないが、該当は稀な操作直後のみなので実用上ほぼ一致する。プローブ本体のクリック
  // 判定は CadCanvas 側の activeField が正確に行うため、ここでの近似が多少ずれても
  // 実害は無い=単にボタンの活性/非活性が一瞬ずれるだけ)
  const fieldDisplayActive =
    !!finalPicFieldView ||
    (onFluid2dNode ? fluid2dLiveFrame != null : !onGasNode && picLiveFrame != null) ||
    (!(onTraceResultNode && traceBackground === "none") && result != null);

  // RF位相モニタ (prompts/82) の表示条件。CadCanvas 上でライブフレームが実際に描画されている
  // 間だけ出す: 周期アニメ/結果フィールド表示 (finalPicFieldView) やガス関連ノード選択中
  // (CadCanvas への picFrame prop 自体を onGasNode で止めている) は「ライブが見えていない」
  // 状態なので合わせて非表示にする (描画優先ロジックの派生値をそのまま流用)。
  // 流体 (2D) ノード選択中は PIC の RF位相モニタと無関係なので合わせて非表示にする
  const showRfMonitorPanel =
    showRfMonitor && picFrame != null && !onGasNode && !onFluid2dNode && !finalPicFieldView;

  // --- インスペクタ (中カラム) の表示制御 ---
  // FieldPanel は1インスタンスのみ mount し、選択ノードに応じて sections/edgeFilter を切替える
  const fieldSections: FieldSection[] =
    activeNode === "domain" ? ["domain"]
    : activeNode === "regions" ? ["regions"]
    : activeNode === "boundary" ? ["boundary"]
    : activeNode === "mesh" ? ["mesh"]
    : activeNode === "bfield" ? ["bfield"]
    : activeNode === "study-fem" ? ["solve"]
    : ["domain"]; // 上記以外のノードでは FieldPanel 自体を表示しないため値は使われない
  const showFieldPage = ["domain", "regions", "boundary", "mesh", "bfield", "study-fem"].includes(activeNode);
  // study-* ノードは設定UI込みの従来ページ、result-* ノードは結果専用ページを表示する
  // (パネル自体は mode="setup"/"results" の2インスタンスを常時 mount し、display:none で切替える)
  const showParticleSetupPage = activeNode === "study-trace";
  const showParticleResultsPage = activeNode === "result-trace";
  const showPicSetupPage = activeNode === "study-pic";
  const showPicResultsPage = activeNode === "result-pic";
  // PIC-MCC 1D (prompts/91) は結果の可視化を全てキャンバス領域 (Plot1dView) 側に持たせているため、
  // 設定パネル (Pic1dPanel) には study/result で内容差が無い。2D の setup/results 2インスタンス方式とは
  // 異なり、単一インスタンスを両ノードで共用する (表示は同じまま、選択ノードでタイトルだけ変わる)
  const showPic1dPage = activeNode === "study-pic1d" || activeNode === "result-pic1d";
  // 1D 流体 (prompts/104-109) も pic1d と同じ理由で study/result 共通の単一インスタンス
  const showFluid1dPage = activeNode === "study-fluid1d" || activeNode === "result-fluid1d";
  // 2D 流体 (prompts/111-113) は CadCanvas を使う (2D PIC と同じ設計) ため、
  // PicPanel と同じ setup/results 2インスタンス分割にする (pic1d/fluid1d の単一インスタンスとは異なる)
  const showFluid2dSetupPage = activeNode === "study-fluid2d";
  const showFluid2dResultsPage = activeNode === "result-fluid2d";
  // VHF 定在波 (prompts/101) も pic1d と同じ理由で study/result 共通の単一インスタンス
  const showTlPage = activeNode === "study-tl" || activeNode === "result-tl";
  const showGasSetupPage = activeNode === "study-gas";
  const showGasResultsPage = activeNode === "result-gas";
  const showResultFemPage = activeNode === "result-fem";
  // パラメータスイープ (prompts/79) は設定+実行UI+ケース一覧を1ページにまとめる
  // (結果は既存の結果ページへ読み込んで見るため、result-* ノードは持たない)
  const showSweepPage = activeNode === "study-sweep";

  // インスペクタ上部のタイトル。boundary/regions は選択中の辺/領域名を付け加える
  const edgeLabelsForTitle =
    project.coord === "rz" ? EDGE_LABELS_RZ : project.coord === "rz_x0" ? EDGE_LABELS_RZ_X0 : EDGE_LABELS_XY;
  const inspectorTitle =
    activeNode === "boundary" && edgeFilter !== null
      ? `境界条件 — ${edgeLabelsForTitle[edgeFilter]} (エッジ${edgeFilter})`
      : activeNode === "regions" && selected
        ? `領域 — ${selected.id}`
        : NODE_TITLES[activeNode];

  // --- 下部ステータスバー ---
  // エラーは error → picError → pic1dError → fluid1dError → fluid2dError → tlError → gasError →
  // sweepError の順で最初の非null を優先表示する
  const statusError =
    error ?? picError ?? pic1dError ?? fluid1dError ?? fluid2dError ?? tlError ?? gasError ?? sweepError;
  // ステータスバーのエラーを閉じる (各エラー state を一括クリア)。パネル内の
  // エラー表示は各パネルの error prop 経由で残したいが、実体は同じ state なので
  // ここでは「ステータスバーに居座る」問題の解消を優先して両方消える仕様とする
  const dismissStatusError = () => {
    setError(null);
    setPicError(null);
    setPic1dError(null);
    setFluid1dError(null);
    setFluid2dError(null);
    setTlError(null);
    setGasError(null);
    setSweepError(null);
  };
  // スイープの完了ケース数 (進捗表示用。PIC/DSMC 単発実行と同列の優先度で表示する)
  const sweepCompletedCount = sweepCases.filter((c) => c.status === "done" || c.status === "error").length;
  // 続きから実行では frame.step が通算で進むため、区間開始オフセットを引いて計算する
  // (不具合修正: 分子だけ通算になり 100% 超の表示になっていた)
  const picStepOffset = picStarted?.step_offset ?? 0;
  const picSegStep = Math.max(0, (picFrame?.step ?? picStepOffset) - picStepOffset);
  const picPct = picStarted && picStarted.n_steps > 0
    ? Math.min(100, Math.round((picSegStep / picStarted.n_steps) * 100))
    : 0;
  // PIC-MCC 1D も続き実行で step が通算するため同じ考え方で区間内ステップを求める
  const pic1dStepOffset = pic1dStarted?.step_offset ?? 0;
  const pic1dSegStep = Math.max(0, (pic1dFrame?.step ?? pic1dStepOffset) - pic1dStepOffset);
  const pic1dPct = pic1dStarted && pic1dStarted.n_steps > 0
    ? Math.min(100, Math.round((pic1dSegStep / pic1dStarted.n_steps) * 100))
    : 0;
  // 1D 流体も続き実行で step が通算するため pic1d と同じ考え方で区間内ステップを求める
  const fluid1dStepOffset = fluid1dStarted?.step_offset ?? 0;
  const fluid1dSegStep = Math.max(0, (fluid1dFrame?.step ?? fluid1dStepOffset) - fluid1dStepOffset);
  const fluid1dPct = fluid1dStarted && fluid1dStarted.n_steps > 0
    ? Math.min(100, Math.round((fluid1dSegStep / fluid1dStarted.n_steps) * 100))
    : 0;
  // 2D 流体も続き実行で step が通算するため同じ考え方で区間内ステップを求める
  const fluid2dStepOffset = fluid2dStarted?.step_offset ?? 0;
  const fluid2dSegStep = Math.max(0, (fluid2dFrame?.step ?? fluid2dStepOffset) - fluid2dStepOffset);
  const fluid2dPct = fluid2dStarted && fluid2dStarted.n_steps > 0
    ? Math.min(100, Math.round((fluid2dSegStep / fluid2dStarted.n_steps) * 100))
    : 0;
  const gasPct = gasProgress && gasProgress.nSteps > 0 ? Math.round((gasProgress.step / gasProgress.nSteps) * 100) : 0;
  // VHF 定在波 (prompts/101) は continue が無いため step は常に区間内の値そのもの (オフセット不要)
  const tlPct = tlProgress && tlProgress.nSteps > 0 ? Math.round((tlProgress.step / tlProgress.nSteps) * 100) : 0;

  // 実行経過時間 (ステータスバー、prompts/86)。Date.now() を毎レンダーで直接読むことで
  // elapsedTick (1秒ごとに更新される tick state) が変わるたびに再計算される。実行中でない
  // 場合や開始時刻が未記録の場合は 0 (該当ブランチ自体が表示されないので使われない)
  const busyElapsedSec = busy && busyStartTimeRef.current != null ? (Date.now() - busyStartTimeRef.current) / 1000 : 0;
  const picElapsedSec = picRunning && picStartTimeRef.current != null ? (Date.now() - picStartTimeRef.current) / 1000 : 0;
  const pic1dElapsedSec = pic1dRunning && pic1dStartTimeRef.current != null ? (Date.now() - pic1dStartTimeRef.current) / 1000 : 0;
  const fluid1dElapsedSec = fluid1dRunning && fluid1dStartTimeRef.current != null ? (Date.now() - fluid1dStartTimeRef.current) / 1000 : 0;
  const fluid2dElapsedSec = fluid2dRunning && fluid2dStartTimeRef.current != null ? (Date.now() - fluid2dStartTimeRef.current) / 1000 : 0;
  const tlElapsedSec = tlRunning && tlStartTimeRef.current != null ? (Date.now() - tlStartTimeRef.current) / 1000 : 0;
  const gasElapsedSec = gasRunning && gasStartTimeRef.current != null ? (Date.now() - gasStartTimeRef.current) / 1000 : 0;
  const sweepElapsedSec = sweepRunning && sweepStartTimeRef.current != null ? (Date.now() - sweepStartTimeRef.current) / 1000 : 0;

  return (
    <div className="app">
      <div className="toolbar">
        <h1>ES-Sim</h1>
        <button className="secondary" onClick={saveProject}>保存</button>
        <button
          className="secondary"
          onClick={saveProjectWithResults}
          disabled={!hasAnyResults}
          title={
            hasAnyResults
              ? "プロジェクト設定に加えて計算結果 (FEM/トレース/PIC/DSMC) も1ファイルに保存します。PIC結果を含むとファイルが大きくなることがあります"
              : "保存できる計算結果がありません (Solve/Mesh/粒子トレース/PIC/DSMC のいずれかを実行してください)"
          }
        >
          結果付き保存
        </button>
        <button className="secondary" onClick={() => fileInputRef.current?.click()}>読込</button>
        <input
          ref={fileInputRef}
          type="file"
          accept="application/json"
          className="file-input"
          onChange={loadProject}
        />
        <div className="sep" />
        <button className="secondary" onClick={doUndo} disabled={!history.canUndo} title="Undo (Ctrl+Z)">
          ↶ Undo
        </button>
        <button
          className="secondary"
          onClick={doRedo}
          disabled={!history.canRedo}
          title="Redo (Ctrl+Y / Ctrl+Shift+Z)"
        >
          ↷ Redo
        </button>
        <div className="sep" />
        {/* 座標系: FieldPanel の domain セクションにも同じ select がある (実体は同じ setCoord/
            project.coord を共有する二重配置。値の食い違いは起きない) */}
        <label className="snap" title="平面2D / 軸対称 r-z の切替">
          座標系
          <select value={project.coord ?? "xy"} onChange={(e) => setCoord(e.target.value as "xy" | "rz" | "rz_x0")}>
            <option value="xy">平面 2D</option>
            <option value="rz">軸対称 r-z (下辺が軸)</option>
            <option value="rz_x0">軸対称 r-z (左辺が軸)</option>
          </select>
        </label>
        <label className="snap" title="長さの表示・入力単位を切替えます (project 内部は常に m)">
          単位
          <select value={lengthUnit} onChange={(e) => setLengthUnit(e.target.value as LengthUnit)}>
            <option value="mm">{LENGTH_UNIT_LABEL.mm}</option>
            <option value="um">{LENGTH_UNIT_LABEL.um}</option>
          </select>
        </label>
        <div className="spacer" />
        <label
          className="snap port-field"
          title="開発時は uvicorn の --port をこの値に合わせてください。配布版ではアプリ再起動後にサイドカーへ反映されます"
        >
          ポート
          <input
            type="number"
            className="port-input"
            value={portInput}
            onChange={(e) => setPortInput(e.target.value)}
            onBlur={commitPortInput}
            onKeyDown={(e) => {
              if (e.key === "Enter") e.currentTarget.blur(); // Enterで確定 (blurのcommitPortInputに委ねる)
            }}
          />
        </label>
        {portError && <div className="status ng">ポート設定の保存に失敗: {portError}</div>}
        <div
          className={`status ${health ? (health.numba === false ? "warn" : "ok") : "ng"}`}
          title={
            health?.numba === false
              ? "Numba JIT が無効なため、PIC/DSMC の粒子カーネルは低速な NumPy 経路で動作しています"
              : "開発時は uvicorn の --port をこの値に合わせてください。配布版ではアプリ再起動後にサイドカーへ反映されます"
          }
        >
          {health
            ? `backend v${health.version} ${health.gpu ? "(GPU)" : "(CPU)"}${
                health.numba === false ? " — Numba無効・低速" : health.numba ? " numba" : ""
              }`
            : `backend 未接続 — uvicorn es_sim.server:app --port ${getPort()} を起動してください`}
        </div>
      </div>

      <div className="main">
        {/* 左カラム: プロジェクトツリー (幅固定、リサイズ不要) */}
        <div className="tree-col">
          <ProjectTree
            project={project}
            lengthUnit={lengthUnit}
            activeNode={activeNode}
            onSelectNode={selectNode}
            selectedRegionId={selectedRegionId}
            onSelectRegion={selectRegionNode}
            edgeFilter={activeNode === "boundary" ? edgeFilter : null}
            onSelectEdge={selectEdgeNode}
            edgeState={edgeState}
            busy={busy}
            result={result}
            traceResult={traceResult}
            picRunning={picRunning}
            picStarted={picStarted}
            picFrame={picFrame}
            picError={picError}
            picFields={picFields}
            picHistory={picHistory}
            pic1dRunning={pic1dRunning}
            pic1dStarted={pic1dStarted}
            pic1dFrame={pic1dFrame}
            pic1dError={pic1dError}
            pic1dResult={pic1dResult}
            fluid1dRunning={fluid1dRunning}
            fluid1dStarted={fluid1dStarted}
            fluid1dFrame={fluid1dFrame}
            fluid1dError={fluid1dError}
            fluid1dResult={fluid1dResult}
            fluid2dRunning={fluid2dRunning}
            fluid2dStarted={fluid2dStarted}
            fluid2dFrame={fluid2dFrame}
            fluid2dError={fluid2dError}
            fluid2dResult={fluid2dResult}
            tlRunning={tlRunning}
            tlStarted={tlStarted}
            tlProgressStep={tlProgress?.step ?? null}
            tlError={tlError}
            tlResult={tlResult}
            gasRunning={gasRunning}
            gasProgress={gasProgress}
            gasError={gasError}
            gasResult={gasResult}
            sweepRunning={sweepRunning}
            sweepCompleted={sweepCases.filter((c) => c.status === "done" || c.status === "error").length}
            sweepTotal={sweepStarted?.n_cases ?? 0}
            sweepHasError={!!sweepError || sweepCases.some((c) => c.status === "error")}
          />
        </div>

        {/* 中カラム: インスペクタ (選択ノードの設定/実行UI)。既存の .side CSS を流用する */}
        <div className="side" style={{ width: sideWidth }}>
          <div className="inspector-title">{inspectorTitle}</div>
          <div className="side-tab-content">
            {/* 各インスペクタページは display:none で非表示化するのみでアンマウントしない。
                これにより PIC の WebSocket 接続やチャート履歴、他ページの編集状態が
                ノード切替をまたいで保持される (旧タブ切替と同じ方式) */}
            <div style={{ display: showFieldPage ? "block" : "none" }}>
              <FieldPanel
                project={project}
                lengthUnit={lengthUnit}
                domainW={domainW}
                domainH={domainH}
                setDomainSize={setDomainSize}
                setCoord={setCoord}
                edgeState={edgeState}
                setEdgeType={setEdgeType}
                setEdgeVoltage={setEdgeVoltage}
                setEdgeVoltageRf={setEdgeVoltageRf}
                setEdgeVoltageWaveform={setEdgeVoltageWaveform}
                setEdgeSeeGamma={setEdgeSeeGamma}
                setMeshSize={setMeshSize}
                setMeshMode={setMeshMode}
                setBField={setBField}
                meshResult={meshResult}
                selectedRegionId={selectedRegionId}
                onSelectRegion={selectRegionNode}
                selected={selected}
                renameRegion={renameRegion}
                setRegionType={setRegionType}
                editRegionShape={editRegionShape}
                editRegionPolygon={editRegionPolygon}
                mergeRegions={mergeRegions}
                updateRegion={updateRegion}
                deleteRegion={deleteRegion}
                setRegionLocalSize={setRegionLocalSize}
                updateEdgeMeshSize={updateEdgeMeshSize}
                deleteEdgeMeshSize={deleteEdgeMeshSize}
                result={result}
                sections={fieldSections}
                edgeFilter={activeNode === "boundary" ? edgeFilter : null}
                runMesh={runMesh}
                runSolve={runSolve}
                busy={busy}
                canRun={!!health}
                showMesh={showMesh}
                onToggleShowMesh={() => setShowMesh(!showMesh)}
              />
            </div>

            {/* study-trace (設定+実行UI) と result-trace (結果専用) は同じ props を渡す
                ParticlePanel の2インスタンスで、mode だけを切り替えて表示する */}
            <div style={{ display: showParticleSetupPage ? "block" : "none" }}>
              <ParticlePanel
                project={project}
                lengthUnit={lengthUnit}
                particles={particles}
                onChange={setParticles}
                busy={busy}
                canRun={!!health}
                onTrace={runTrace}
                traceResult={traceResult}
                elapsedS={traceElapsedS}
                showTrajectories={showTrajectories}
                onToggleTrajectories={setShowTrajectories}
                showEmitter={showEmitter}
                onToggleEmitter={setShowEmitter}
                mode="setup"
              />
            </div>
            <div style={{ display: showParticleResultsPage ? "block" : "none" }}>
              <ParticlePanel
                project={project}
                lengthUnit={lengthUnit}
                particles={particles}
                onChange={setParticles}
                busy={busy}
                canRun={!!health}
                onTrace={runTrace}
                traceResult={traceResult}
                elapsedS={traceElapsedS}
                showTrajectories={showTrajectories}
                onToggleTrajectories={setShowTrajectories}
                showEmitter={showEmitter}
                onToggleEmitter={setShowEmitter}
                mode="results"
                background={traceBackground}
                onBackgroundChange={setTraceBackground}
              />
            </div>

            {/* study-pic (設定+実行UI) と result-pic (結果専用) は同じ props を渡す
                PicPanel の2インスタンスで、mode だけを切り替えて表示する */}
            <div style={{ display: showPicSetupPage ? "block" : "none" }}>
              <PicPanel
                project={project}
                lengthUnit={lengthUnit}
                pic={pic}
                onChange={setPic}
                emitter={particles.emitter}
                canRun={!!health}
                running={picRunning}
                onStart={runPicStart}
                onStop={runPicStop}
                canContinue={picCanContinue}
                onContinue={runPicContinue}
                continueDisabledByProjectChange={picProjectChangedSinceRun}
                started={picStarted}
                frame={picFrame}
                history={picHistory}
                error={picError}
                fields={picFields}
                timing={picTiming}
                elapsedS={picElapsedS}
                resultField={picResultField}
                onResultFieldChange={(v) => {
                  // 結果表示の切替時はアニメ優先を解除し、選択したフィールドを表示する
                  setPicResultField(v);
                  setCycleViewActive(false);
                  setCyclePlaying(false);
                }}
                logScale={picLogScale}
                onLogScaleChange={setPicLogScale}
                picLiveField={picLiveField}
                onPicLiveFieldChange={setPicLiveField}
                picLiveLogScale={picLiveLogScale}
                onPicLiveLogScaleChange={setPicLiveLogScale}
                showRfMonitor={showRfMonitor}
                onShowRfMonitorChange={setShowRfMonitor}
                cycle={picCycle}
                cycleField={cycleField}
                onCycleFieldChange={(v) => { setCycleField(v); setCycleViewActive(true); }}
                cycleLogScale={cycleLogScale}
                onCycleLogScaleChange={(v) => { setCycleLogScale(v); setCycleViewActive(true); }}
                cyclePlaying={cyclePlaying}
                onCyclePlayingChange={(v) => { setCyclePlaying(v); if (v) setCycleViewActive(true); }}
                cycleBinIndex={cycleBinIndex}
                onCycleBinIndexChange={(v) => { setCycleBinIndex(v); setCycleViewActive(true); }}
                cycleFps={cycleFps}
                onCycleFpsChange={setCycleFps}
                cycleShowParticles={cycleShowParticles}
                onCycleShowParticlesChange={(v) => { setCycleShowParticles(v); setCycleViewActive(true); }}
                collectorResults={picCollectors}
                selectedCollectorIndex={selectedCollectorIndex}
                onSelectCollector={setSelectedCollectorIndexRaw}
                onUpdateCollector={updateCollector}
                onDeleteCollector={deleteCollector}
                eedfResults={picEedf}
                selectedEedfIndex={selectedEedfIndex}
                onSelectEedfRegion={setSelectedEedfIndexRaw}
                onUpdateEedfRegion={updateEedfRegion}
                onDeleteEedfRegion={deleteEedfRegion}
                sheathLines={sheathLinesList}
                onUpdateSheathLineLabel={updateSheathLineLabel}
                onDeleteSheathLine={deleteSheathLine}
                sheathAlpha={sheathAlpha}
                onSheathAlphaChange={setSheathAlpha}
                sheathPhaseS={sheathPhaseS}
                mode="setup"
              />
            </div>
            <div style={{ display: showPicResultsPage ? "block" : "none" }}>
              <PicPanel
                project={project}
                lengthUnit={lengthUnit}
                pic={pic}
                onChange={setPic}
                emitter={particles.emitter}
                canRun={!!health}
                running={picRunning}
                onStart={runPicStart}
                onStop={runPicStop}
                canContinue={picCanContinue}
                onContinue={runPicContinue}
                continueDisabledByProjectChange={picProjectChangedSinceRun}
                started={picStarted}
                frame={picFrame}
                history={picHistory}
                error={picError}
                fields={picFields}
                timing={picTiming}
                elapsedS={picElapsedS}
                resultField={picResultField}
                onResultFieldChange={(v) => {
                  // 結果表示の切替時はアニメ優先を解除し、選択したフィールドを表示する
                  setPicResultField(v);
                  setCycleViewActive(false);
                  setCyclePlaying(false);
                }}
                logScale={picLogScale}
                onLogScaleChange={setPicLogScale}
                picLiveField={picLiveField}
                onPicLiveFieldChange={setPicLiveField}
                picLiveLogScale={picLiveLogScale}
                onPicLiveLogScaleChange={setPicLiveLogScale}
                showRfMonitor={showRfMonitor}
                onShowRfMonitorChange={setShowRfMonitor}
                cycle={picCycle}
                cycleField={cycleField}
                onCycleFieldChange={(v) => { setCycleField(v); setCycleViewActive(true); }}
                cycleLogScale={cycleLogScale}
                onCycleLogScaleChange={(v) => { setCycleLogScale(v); setCycleViewActive(true); }}
                cyclePlaying={cyclePlaying}
                onCyclePlayingChange={(v) => { setCyclePlaying(v); if (v) setCycleViewActive(true); }}
                cycleBinIndex={cycleBinIndex}
                onCycleBinIndexChange={(v) => { setCycleBinIndex(v); setCycleViewActive(true); }}
                cycleFps={cycleFps}
                onCycleFpsChange={setCycleFps}
                cycleShowParticles={cycleShowParticles}
                onCycleShowParticlesChange={(v) => { setCycleShowParticles(v); setCycleViewActive(true); }}
                collectorResults={picCollectors}
                selectedCollectorIndex={selectedCollectorIndex}
                onSelectCollector={setSelectedCollectorIndexRaw}
                onUpdateCollector={updateCollector}
                onDeleteCollector={deleteCollector}
                eedfResults={picEedf}
                selectedEedfIndex={selectedEedfIndex}
                onSelectEedfRegion={setSelectedEedfIndexRaw}
                onUpdateEedfRegion={updateEedfRegion}
                onDeleteEedfRegion={deleteEedfRegion}
                sheathLines={sheathLinesList}
                onUpdateSheathLineLabel={updateSheathLineLabel}
                onDeleteSheathLine={deleteSheathLine}
                sheathAlpha={sheathAlpha}
                onSheathAlphaChange={setSheathAlpha}
                sheathPhaseS={sheathPhaseS}
                mode="results"
              />
            </div>

            {/* PIC-MCC 1D (prompts/91): 結果の可視化は Plot1dView (キャンバス領域) 側に
                持たせているため、設定パネルは study-pic1d/result-pic1d で共通の単一インスタンス
                (2D PicPanel のような setup/results 2インスタンス分割はしない) */}
            <div style={{ display: showPic1dPage ? "block" : "none" }}>
              <Pic1dPanel
                lengthUnit={lengthUnit}
                pic1d={pic1d}
                onChange={setPic1d}
                canRun={!!health}
                running={pic1dRunning}
                onStart={runPic1dStart}
                onStop={runPic1dStop}
                canContinue={pic1dCanContinue}
                onContinue={runPic1dContinue}
                started={pic1dStarted}
                frame={pic1dFrame}
                error={pic1dError}
              />
            </div>

            {/* 1D プラズマ流体 (prompts/104-109): 結果の可視化は Fluid1dPlotView (キャンバス領域) 側に
                持たせているため、設定パネルは study-fluid1d/result-fluid1d で共通の単一インスタンス
                (pic1d と同じ設計)。pic1d prop は「1D PIC の設定を取込」プリセットボタン用 */}
            <div style={{ display: showFluid1dPage ? "block" : "none" }}>
              <Fluid1dPanel
                lengthUnit={lengthUnit}
                fluid1d={fluid1d}
                onChange={setFluid1d}
                canRun={!!health}
                running={fluid1dRunning}
                onStart={runFluid1dStart}
                onStop={runFluid1dStop}
                canContinue={fluid1dCanContinue}
                onContinue={runFluid1dContinue}
                started={fluid1dStarted}
                frame={fluid1dFrame}
                error={fluid1dError}
                pic1d={pic1d}
              />
            </div>

            {/* 2D/軸対称 流体 (prompts/111-113): CadCanvas を使う (2D PIC と同じ設計) ため、
                study-fluid2d (設定+実行UI) と result-fluid2d (結果専用) は同じ props を渡す
                Fluid2dPanel の2インスタンスで、mode だけを切り替えて表示する (PicPanel と同じ流儀) */}
            <div style={{ display: showFluid2dSetupPage ? "block" : "none" }}>
              <Fluid2dPanel
                project={project}
                fluid2d={fluid2d}
                onChange={setFluid2d}
                canRun={!!health}
                running={fluid2dRunning}
                onStart={runFluid2dStart}
                onStop={runFluid2dStop}
                canContinue={fluid2dCanContinue}
                onContinue={runFluid2dContinue}
                continueDisabledByProjectChange={fluid2dProjectChangedSinceRun}
                started={fluid2dStarted}
                frame={fluid2dFrame}
                error={fluid2dError}
                fluid1d={fluid1d}
                result={fluid2dResult}
                meshResult={meshResult}
                resultField={fluid2dResultField}
                onResultFieldChange={(v) => {
                  // 結果表示の切替時はアニメ優先を解除し、選択したフィールドを表示する (PicPanel と同じ)
                  setFluid2dResultField(v);
                  setFluid2dCycleViewActive(false);
                  setFluid2dCyclePlaying(false);
                }}
                logScale={fluid2dLogScale}
                onLogScaleChange={setFluid2dLogScale}
                liveField={fluid2dLiveField}
                onLiveFieldChange={setFluid2dLiveField}
                liveLogScale={fluid2dLiveLogScale}
                onLiveLogScaleChange={setFluid2dLiveLogScale}
                cycle={fluid2dCycle}
                cycleField={fluid2dCycleField}
                onCycleFieldChange={(v) => { setFluid2dCycleField(v); setFluid2dCycleViewActive(true); }}
                cycleLogScale={fluid2dCycleLogScale}
                onCycleLogScaleChange={(v) => { setFluid2dCycleLogScale(v); setFluid2dCycleViewActive(true); }}
                cyclePlaying={fluid2dCyclePlaying}
                onCyclePlayingChange={(v) => { setFluid2dCyclePlaying(v); if (v) setFluid2dCycleViewActive(true); }}
                cycleBinIndex={fluid2dCycleBinIndex}
                onCycleBinIndexChange={(v) => { setFluid2dCycleBinIndex(v); setFluid2dCycleViewActive(true); }}
                cycleFps={fluid2dCycleFps}
                onCycleFpsChange={setFluid2dCycleFps}
                mode="setup"
              />
            </div>
            <div style={{ display: showFluid2dResultsPage ? "block" : "none" }}>
              <Fluid2dPanel
                project={project}
                fluid2d={fluid2d}
                onChange={setFluid2d}
                canRun={!!health}
                running={fluid2dRunning}
                onStart={runFluid2dStart}
                onStop={runFluid2dStop}
                canContinue={fluid2dCanContinue}
                onContinue={runFluid2dContinue}
                continueDisabledByProjectChange={fluid2dProjectChangedSinceRun}
                started={fluid2dStarted}
                frame={fluid2dFrame}
                error={fluid2dError}
                fluid1d={fluid1d}
                result={fluid2dResult}
                meshResult={meshResult}
                resultField={fluid2dResultField}
                onResultFieldChange={(v) => {
                  setFluid2dResultField(v);
                  setFluid2dCycleViewActive(false);
                  setFluid2dCyclePlaying(false);
                }}
                logScale={fluid2dLogScale}
                onLogScaleChange={setFluid2dLogScale}
                liveField={fluid2dLiveField}
                onLiveFieldChange={setFluid2dLiveField}
                liveLogScale={fluid2dLiveLogScale}
                onLiveLogScaleChange={setFluid2dLiveLogScale}
                cycle={fluid2dCycle}
                cycleField={fluid2dCycleField}
                onCycleFieldChange={(v) => { setFluid2dCycleField(v); setFluid2dCycleViewActive(true); }}
                cycleLogScale={fluid2dCycleLogScale}
                onCycleLogScaleChange={(v) => { setFluid2dCycleLogScale(v); setFluid2dCycleViewActive(true); }}
                cyclePlaying={fluid2dCyclePlaying}
                onCyclePlayingChange={(v) => { setFluid2dCyclePlaying(v); if (v) setFluid2dCycleViewActive(true); }}
                cycleBinIndex={fluid2dCycleBinIndex}
                onCycleBinIndexChange={(v) => { setFluid2dCycleBinIndex(v); setFluid2dCycleViewActive(true); }}
                cycleFps={fluid2dCycleFps}
                onCycleFpsChange={setFluid2dCycleFps}
                mode="results"
              />
            </div>

            {/* VHF定在波 (prompts/101): 結果の可視化は TlPlotView (キャンバス領域) 側に
                持たせているため、設定パネルは study-tl/result-tl で共通の単一インスタンス
                (pic1d と同じ設計) */}
            <div style={{ display: showTlPage ? "block" : "none" }}>
              <TlPanel
                lengthUnit={lengthUnit}
                tl={tl}
                onChange={setTl}
                canRun={!!health}
                running={tlRunning}
                onStart={runTlStart}
                onStop={runTlStop}
                started={tlStarted}
                progress={tlProgress}
                pic1dResult={pic1dResult}
                error={tlError}
              />
            </div>

            {/* study-gas (設定+実行UI) と result-gas (結果専用) は同じ props を渡す
                GasPanel の2インスタンスで、mode だけを切り替えて表示する */}
            <div style={{ display: showGasSetupPage ? "block" : "none" }}>
              <GasPanel
                project={project}
                meshResult={meshResult}
                lengthUnit={lengthUnit}
                dsmc={project.dsmc ?? null}
                onChange={setDsmc}
                canRun={!!health}
                running={gasRunning}
                onRun={runDsmc}
                onStop={stopDsmc}
                canContinue={gasCanContinue}
                onContinue={runDsmcContinue}
                continueDisabledByProjectChange={gasProjectChangedSinceRun}
                progress={gasProgress}
                lastThreads={gasThreads}
                result={gasResult}
                error={gasError}
                resultField={gasResultField}
                onResultFieldChange={setGasResultField}
                logScale={gasLogScale}
                onLogScaleChange={setGasLogScale}
                showParticles={gasShowParticles}
                onShowParticlesChange={setGasShowParticles}
                mode="setup"
              />
            </div>
            <div style={{ display: showGasResultsPage ? "block" : "none" }}>
              <GasPanel
                project={project}
                meshResult={meshResult}
                lengthUnit={lengthUnit}
                dsmc={project.dsmc ?? null}
                onChange={setDsmc}
                canRun={!!health}
                running={gasRunning}
                onRun={runDsmc}
                onStop={stopDsmc}
                canContinue={gasCanContinue}
                onContinue={runDsmcContinue}
                continueDisabledByProjectChange={gasProjectChangedSinceRun}
                progress={gasProgress}
                lastThreads={gasThreads}
                result={gasResult}
                error={gasError}
                resultField={gasResultField}
                onResultFieldChange={setGasResultField}
                logScale={gasLogScale}
                onLogScaleChange={setGasLogScale}
                showParticles={gasShowParticles}
                onShowParticlesChange={setGasShowParticles}
                mode="results"
              />
            </div>

            {/* パラメータスイープ (prompts/79): 設定+実行UI+ケース一覧を1ページにまとめた
                単一インスタンス (setup/results の分割なし。結果は既存の結果ページに読み込んで見る) */}
            <div style={{ display: showSweepPage ? "block" : "none" }}>
              <SweepPanel
                project={projectForSweep}
                canRun={!!health}
                running={sweepRunning}
                onStart={runSweepStart}
                onStop={runSweepStop}
                started={sweepStarted}
                cases={sweepCases}
                error={sweepError}
                loadedCaseIndex={sweepLoadedCaseIndex}
                onLoadCase={loadSweepCase}
              />
            </div>

            {/* 「静電場結果」ページ: 旧・電位分布φ/電場|E|/ラインプロファイルの3ノードを統合
                (他モジュールと同じ「1モジュール=1結果ノード」に揃える、prompts/69)。
                result が無い場合も表示オプション等は操作可能なままにし、先頭にヒントのみ出す */}
            <div style={{ display: showResultFemPage ? "block" : "none" }}>
              {!result && (
                <p className="hint">静電場FEMが未実行です。スタディ「静電場」から実行してください。</p>
              )}
              <h2>結果表示</h2>
              <label className="snap">
                表示
                <select value={fieldView} onChange={(e) => setFieldView(e.target.value as FieldView)}>
                  <option value="v">電位 V</option>
                  <option value="e_abs">|E|</option>
                </select>
              </label>
              <Toggle label="等電位線" checked={showIsolines} onChange={setShowIsolines} />
              <Toggle label="ベクトル" checked={showVectors} onChange={setShowVectors} />

              <h2>ラインプロファイル</h2>
              <div className="hint">
                「プロファイル線を引く」を押してからキャンバス上で2点クリックすると、その間の
                電位/|E| 分布をキャンバス下部に表示します。
              </div>
              <button className="secondary" onClick={() => setTool("profile")}>プロファイル線を引く</button>
              {!profileLine && <div className="muted">(まだプロファイル線が指定されていません)</div>}
              {profileLine && (
                <div className="kv">
                  <span>プロファイル表示中</span>
                  <button className="secondary" onClick={() => setProfileLine(null)}>閉じる</button>
                </div>
              )}

              <h2>解析結果</h2>
              <ResultSummary result={result} coord={project.coord} elapsedS={solveElapsedS} />
            </div>
          </div>
        </div>

        {/* インスペクタ幅のリサイザ (ドラッグで変更、ダブルクリックで既定幅に戻す)。
            ツリー/インスペクタ/キャンバスの順に並び替えたため、ドラッグ方向は旧実装 (side が右端) から
            反転している (右へドラッグすると幅が増える) */}
        <div
          className="side-resizer"
          onMouseDown={(e) => {
            e.preventDefault();
            const startX = e.clientX;
            const startW = sideWidth;
            const onMove = (ev: MouseEvent) => {
              const w = startW + (ev.clientX - startX);
              setSideWidth(Math.min(560, Math.max(220, w)));
            };
            const onUp = () => {
              window.removeEventListener("mousemove", onMove);
              window.removeEventListener("mouseup", onUp);
              document.body.style.cursor = "";
            };
            document.body.style.cursor = "col-resize";
            window.addEventListener("mousemove", onMove);
            window.addEventListener("mouseup", onUp);
          }}
          onDoubleClick={() => setSideWidth(280)}
          title="ドラッグで幅を変更 / ダブルクリックで既定幅"
        />

        {/* 右カラム: キャンバスツールバー + CadCanvas (PIC-MCC 1D 選択時は Plot1dView、1D 流体選択時は
            Fluid1dPlotView に差し替え、prompts/91・104-109)。1D は geometry/mesh と無関係なので
            CAD 編集ツールバー自体を出さない (レイアウト構造 <div className="canvas-col"> ... </div>
            自体は崩さない) */}
        <div className="canvas-col">
          {isPic1dNode ? (
            <Plot1dView
              lengthUnit={lengthUnit}
              pic1d={pic1d}
              running={pic1dRunning}
              started={pic1dStarted}
              frame={pic1dFrame}
              result={pic1dResult}
              error={pic1dError}
            />
          ) : isFluid1dNode ? (
            <Fluid1dPlotView
              lengthUnit={lengthUnit}
              fluid1d={fluid1d}
              running={fluid1dRunning}
              started={fluid1dStarted}
              frame={fluid1dFrame}
              result={fluid1dResult}
              error={fluid1dError}
              pic1dResult={pic1dResult}
            />
          ) : isTlNode ? (
            <TlPlotView
              lengthUnit={lengthUnit}
              running={tlRunning}
              started={tlStarted}
              progress={tlProgress}
              result={tlResult}
              error={tlError}
            />
          ) : (
            <>
          <div className="tool-toolbar">
            <button className={`tool ${tool === "select" ? "active" : ""}`} onClick={() => setTool("select")}>
              選択
            </button>
            <button
              className={`tool ${tool === "polyline" ? "active" : ""}`}
              onClick={() => setTool("polyline")}
              title="domain外にはみ出した部分は解析時にクリップされます"
            >
              ポリライン
            </button>
            <button
              className={`tool ${tool === "rect" ? "active" : ""}`}
              onClick={() => setTool("rect")}
              title="domain外にはみ出した部分は解析時にクリップされます"
            >
              矩形
            </button>
            <button
              className={`tool ${tool === "circle" ? "active" : ""}`}
              onClick={() => setTool("circle")}
              title="domain外にはみ出した部分は解析時にクリップされます"
            >
              円
            </button>
            <button className={`tool ${tool === "profile" ? "active" : ""}`} onClick={() => setTool("profile")}>
              プロファイル
            </button>
            <button className={`tool ${tool === "emitter" ? "active" : ""}`} onClick={() => setTool("emitter")}>
              エミッタ
            </button>
            <button
              className={`tool ${tool === "collector" ? "active" : ""}`}
              onClick={() => setTool("collector")}
              title={
                collectorsList.length >= MAX_COLLECTORS
                  ? `コレクタは最大${MAX_COLLECTORS}個までです`
                  : "2点クリックでコレクタ線分を追加します"
              }
            >
              コレクタ ({collectorsList.length}/{MAX_COLLECTORS})
            </button>
            {tool === "collector" && collectorsList.length >= MAX_COLLECTORS && (
              <span className="snap" style={{ color: "#e0b050" }}>
                コレクタは最大{MAX_COLLECTORS}個に達しました
              </span>
            )}
            <button
              className={`tool ${tool === "gasbc" ? "active" : ""}`}
              onClick={() => setTool("gasbc")}
              title="2点クリックでDSMCの線分境界 (流入口など) を追加します"
            >
              ガス境界
            </button>
            <button
              className={`tool ${tool === "eedfbox" ? "active" : ""}`}
              onClick={() => setTool("eedfbox")}
              title={
                eedfRegionsList.length >= MAX_EEDF_REGIONS
                  ? `EEDF領域は最大${MAX_EEDF_REGIONS}個までです`
                  : "2点クリックでEEDF/EEPF集計領域 (矩形) を追加します"
              }
            >
              EEDF領域 ({eedfRegionsList.length}/{MAX_EEDF_REGIONS})
            </button>
            {tool === "eedfbox" && eedfRegionsList.length >= MAX_EEDF_REGIONS && (
              <span className="snap" style={{ color: "#e0b050" }}>
                EEDF領域は最大{MAX_EEDF_REGIONS}個に達しました
              </span>
            )}
            <button
              className={`tool ${tool === "meshref" ? "active" : ""}`}
              onClick={() => setTool("meshref")}
              title="2点クリックで線分近傍のローカルメッシュ細分化を追加します (非構造メッシュのみ有効)"
            >
              メッシュ細分
            </button>
            <button
              className={`tool ${tool === "sheathline" ? "active" : ""}`}
              onClick={() => setTool("sheathline")}
              title={
                sheathLinesList.length >= MAX_SHEATH_LINES
                  ? `シース評価線は最大${MAX_SHEATH_LINES}本までです`
                  : "2点クリックでシースエッジ評価ライン (Brinkmann判定) を追加します。" +
                    "1点目を電極側、2点目をバルク側に取ってください (積分の参照点はライン終点)"
              }
            >
              シース評価線 ({sheathLinesList.length}/{MAX_SHEATH_LINES})
            </button>
            {tool === "sheathline" && sheathLinesList.length >= MAX_SHEATH_LINES && (
              <span className="snap" style={{ color: "#e0b050" }}>
                シース評価線は最大{MAX_SHEATH_LINES}本に達しました
              </span>
            )}
            <button
              className={`tool ${tool === "probe" ? "active" : ""}`}
              onClick={() => setTool("probe")}
              disabled={!fieldDisplayActive}
              title={
                fieldDisplayActive
                  ? "クリックした位置のフィールド値を読み取ります"
                  : "フィールド表示中のみ使用できます"
              }
            >
              プローブ
            </button>
            <div className="sep" />
            <Toggle label="グリッドスナップ" checked={gridSnap} onChange={setGridSnap} />
            <label className="snap">
              ルーラー文字
              <select
                className="ruler-font-select"
                value={rulerFontSize}
                onChange={(e) => setRulerFontSize(Number(e.target.value))}
              >
                <option value={9}>小</option>
                <option value={11}>中</option>
                <option value={14}>大</option>
              </select>
            </label>
            {/* 表示 (電位V/|E|)・等電位線・ベクトルの切替は「静電場結果」インスペクタページへ
                集約済みのため、ツールバーからは撤去した (二重配置の解消、prompts/69 の続き) */}
            <div className="sep" />
            {/* キャンバスオーバーレイの表示切替群。メッシュ/エミッタは既存 state
                (FieldPanel の solve セクション・ParticlePanel のトグルと共有) */}
            <span className="field-view-label">表示</span>
            <Toggle label="メッシュ" checked={showMesh} onChange={setShowMesh} />
            <Toggle label="エミッタ" checked={showEmitter} onChange={setShowEmitter} />
            <Toggle label="コレクタ" checked={showCollectors} onChange={setShowCollectors} />
            <Toggle label="ガス境界" checked={showGasBoundaries} onChange={setShowGasBoundaries} />
            <Toggle label="EEDF領域" checked={showEedfRegions} onChange={setShowEedfRegions} />
            <Toggle label="メッシュ細分" checked={showEdgeMeshSizes} onChange={setShowEdgeMeshSizes} />
            <Toggle label="シースエッジ" checked={showSheathEdge} onChange={setShowSheathEdge} />
            {/* コンター配色・カラーバー手動レンジ (prompts/103)。フィールド表示中のみ意味を
                持つため、そうでないときは出さない (混み合ったツールバーに常時出す必要がない) */}
            {fieldDisplayActive && (
              <>
                <div className="sep" />
                <label className="snap">
                  配色
                  <select value={colormapKey} onChange={(e) => setColormapKey(e.target.value as ColormapKey)}>
                    {COLORMAPS.map((cm) => (
                      <option key={cm.key} value={cm.key}>{cm.label}</option>
                    ))}
                  </select>
                </label>
                <label className="snap">
                  レンジ
                  <CommitNullableNumberInput
                    className="colorrange-input"
                    value={colorRange.min ?? null}
                    placeholder="自動"
                    onCommit={(v) => setColorRange((r) => ({ ...r, min: v }))}
                  />
                  〜
                  <CommitNullableNumberInput
                    className="colorrange-input"
                    value={colorRange.max ?? null}
                    placeholder="自動"
                    onCommit={(v) => setColorRange((r) => ({ ...r, max: v }))}
                  />
                </label>
                {(colorRange.min !== null || colorRange.max !== null) && (
                  <button className="secondary" onClick={() => setColorRange({ min: null, max: null })}>
                    自動
                  </button>
                )}
              </>
            )}
          </div>

          <CadCanvas
            project={project}
            lengthUnit={lengthUnit}
            result={onTraceResultNode && traceBackground === "none" ? null : result}
            meshResult={meshResult}
            meshResultIsLatest={meshPreviewFresh}
            showMesh={showMesh}
            tool={tool}
            gridSnap={gridSnap}
            rulerFontSize={rulerFontSize}
            selectedRegionId={selectedRegionId}
            fieldView={onTraceResultNode && traceBackground !== "none" ? traceBackground : fieldView}
            showIsolines={showIsolines}
            showVectors={showVectors}
            profileLine={profileLine}
            collectors={showCollectors ? collectorsList : []}
            selectedCollectorIndex={selectedCollectorIndex}
            eedfRegions={showEedfRegions ? eedfRegionsList : []}
            selectedEedfIndex={selectedEedfIndex}
            emitter={showEmitter ? particles.emitter : null}
            traceResult={traceResult}
            showTrajectories={showTrajectories}
            picFrame={onFluid2dNode ? fluid2dLiveFrame : onGasNode ? null : picLiveFrame}
            picFieldView={finalPicFieldView}
            gasParticles={gasRunning && gasShowParticles ? gasLiveParticles : null}
            gasBoundaries={showGasBoundaries ? gasBoundariesList : []}
            edgeMeshSizes={showEdgeMeshSizes ? edgeMeshSizesList : []}
            sheathLines={showSheathEdge ? sheathLinesList : []}
            sheathDensity={showSheathEdge ? sheathSource : null}
            sheathAlpha={sheathAlpha}
            colormapKey={colormapKey}
            colorRange={colorRange}
            onSelectRegion={selectRegionFromCanvas}
            onDeleteRegion={deleteRegion}
            onAddRegion={addRegion}
            onMoveRegion={moveRegion}
            onEditRegionPolygon={editRegionPolygon}
            onEditRegionShape={editRegionShape}
            onProfileLine={(p1, p2) => setProfileLine([p1, p2])}
            onSetEmitter={setEmitterPoints}
            onSetCollector={setCollectorPoints}
            onSetGasBoundary={setGasBoundaryPoints}
            onSetEedfRegion={setEedfRegionPoints}
            onSetEdgeMeshSize={setEdgeMeshSizePoints}
            onSetSheathLine={setSheathLinePoints}
          />
          {showRfMonitorPanel && <RfPhaseMonitor project={project} t={picFrame!.t} />}
          {profileLine && (
            <ProfilePanel
              project={project}
              lengthUnit={lengthUnit}
              p1={profileLine[0]}
              p2={profileLine[1]}
              onClose={() => setProfileLine(null)}
            />
          )}
            </>
          )}
        </div>
      </div>

      {/* 下部ステータスバー: エラー > 静電場/トレース計算中 > PIC実行中 > PIC-MCC 1D実行中 > 流体1D実行中 > 流体2D実行中 > VHF定在波実行中 > DSMC実行中 > 準備完了 の優先順位 */}
      <div className="statusbar">
        {/* 実行中は進捗を最優先 (エラーが残っていても別計算の進捗を隠さない)。
            アイドル時のエラーは×で閉じられる (居座り防止。新規実行開始でも自動クリア) */}
        {statusError && !anyRunning ? (
          <span className="statusbar-error">
            {statusError}
            <button
              type="button"
              className="statusbar-error-close"
              title="エラー表示を閉じる"
              onClick={dismissStatusError}
            >
              ×
            </button>
          </span>
        ) : busy ? (
          <span>静電場/トレース 計算中... — 経過 {formatElapsed(busyElapsedSec)}</span>
        ) : picRunning ? (
          <>
            <span>
              PIC-MCC 実行中... {picPct}% ({picSegStep}/{picStarted?.n_steps ?? 0}) — 経過 {formatElapsed(picElapsedSec)}
            </span>
            <div className="statusbar-progress">
              <div className="statusbar-progress-bar" style={{ width: `${picPct}%` }} />
            </div>
          </>
        ) : pic1dRunning ? (
          <>
            <span>
              PIC-MCC 1D 実行中... {pic1dPct}% ({pic1dSegStep}/{pic1dStarted?.n_steps ?? 0}) — 経過 {formatElapsed(pic1dElapsedSec)}
            </span>
            <div className="statusbar-progress">
              <div className="statusbar-progress-bar" style={{ width: `${pic1dPct}%` }} />
            </div>
          </>
        ) : fluid1dRunning ? (
          <>
            <span>
              流体1D 実行中... {fluid1dPct}% ({fluid1dSegStep}/{fluid1dStarted?.n_steps ?? 0}) — 経過 {formatElapsed(fluid1dElapsedSec)}
            </span>
            <div className="statusbar-progress">
              <div className="statusbar-progress-bar" style={{ width: `${fluid1dPct}%` }} />
            </div>
          </>
        ) : fluid2dRunning ? (
          <>
            <span>
              流体2D 実行中... {fluid2dPct}% ({fluid2dSegStep}/{fluid2dStarted?.n_steps ?? 0}) — 経過 {formatElapsed(fluid2dElapsedSec)}
            </span>
            <div className="statusbar-progress">
              <div className="statusbar-progress-bar" style={{ width: `${fluid2dPct}%` }} />
            </div>
          </>
        ) : tlRunning ? (
          <>
            <span>
              VHF定在波 実行中... {tlPct}% ({tlProgress?.step ?? 0}/{tlStarted?.n_steps ?? 0}) — 経過 {formatElapsed(tlElapsedSec)}
            </span>
            <div className="statusbar-progress">
              <div className="statusbar-progress-bar" style={{ width: `${tlPct}%` }} />
            </div>
          </>
        ) : gasRunning ? (
          <>
            <span>DSMC 実行中... {gasPct}% — 経過 {formatElapsed(gasElapsedSec)}</span>
            <div className="statusbar-progress">
              <div className="statusbar-progress-bar" style={{ width: `${gasPct}%` }} />
            </div>
          </>
        ) : sweepRunning ? (
          <>
            <span>
              スイープ実行中... {sweepCompletedCount}/{sweepStarted?.n_cases ?? 0} ケース完了 — 経過 {formatElapsed(sweepElapsedSec)}
            </span>
            <div className="statusbar-progress">
              <div
                className="statusbar-progress-bar"
                style={{
                  width: `${sweepStarted && sweepStarted.n_cases > 0 ? (100 * sweepCompletedCount) / sweepStarted.n_cases : 0}%`,
                }}
              />
            </div>
          </>
        ) : (
          <span>準備完了 — {TOOL_LABELS[tool]}</span>
        )}
      </div>
    </div>
  );
}
