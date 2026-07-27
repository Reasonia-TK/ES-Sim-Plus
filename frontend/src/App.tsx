import { useCallback, useEffect, useMemo, useRef, useState } from "react";
// 実行時の実体は default エクスポートのオブジェクト (src/polygon-clipping.d.ts 参照)
import polygonClipping from "polygon-clipping";
import type { MultiPolygon } from "polygon-clipping";
import { api } from "./api";
import { getPort, initPort, setPort } from "./backendPort";
import CadCanvas from "./canvas/CadCanvas";
import type { FieldView, GasBoundaryView, PicCollectorView, PicFieldView, Tool } from "./canvas/CadCanvas";
import ProfilePanel from "./panels/ProfilePanel";
import FieldPanel, { EDGE_LABELS_RZ, EDGE_LABELS_RZ_X0, EDGE_LABELS_XY } from "./panels/FieldPanel";
import type { FieldSection } from "./panels/FieldPanel";
import ProjectTree from "./ProjectTree";
import type { TreeNode } from "./ProjectTree";
import ParticlePanel from "./panels/ParticlePanel";
import PicPanel, { PIC_FIELD_META } from "./panels/PicPanel";
import type { CyclePicField, PicResultField } from "./panels/PicPanel";
import GasPanel, { DEFAULT_BOUNDARY, DEFAULT_DSMC, GAS_FIELD_META, gasFieldValues } from "./panels/GasPanel";
import type { GasResultField } from "./panels/GasPanel";
import SweepPanel from "./panels/SweepPanel";
import { Toggle } from "./Toggle";
import { PicClient } from "./picClient";
import type { PicClientCallbacks } from "./picClient";
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
  Health,
  MeshResult,
  ParticleSettings,
  PicCollectorResult,
  PicCollectorSettings,
  PicCycle,
  PicDiag,
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
  SolveResult,
  SweepCaseState,
  SweepStartedMsg,
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
};

// pic.injection.emitter は常にフェーズ2 (粒子) パネルの現在のエミッタ設定で上書きしてから
// 保存/送信する (PicPanel 側では編集用の複製を持たず、都度ここで同期する)
function withInjectionEmitter(pic: PicSettings, emitter: ParticleSettings["emitter"]): PicSettings {
  if (!pic.injection) return pic;
  return { ...pic, injection: { ...pic.injection, emitter } };
}

// コレクタ追加数の上限 (バックエンドの validator と同じ、prompts/36/37)
const MAX_COLLECTORS = 8;

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
  "study-gas": "スタディ — DSMC",
  "study-sweep": "スタディ — パラメータスイープ",
  "result-fem": "結果 — 静電場",
  "result-trace": "結果 — 粒子追跡",
  "result-pic": "結果 — PIC-MCC",
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
};

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
function ResultSummary({ result, coord }: { result: SolveResult | null; coord: Project["coord"] }) {
  if (!result) {
    return <div className="muted">(まだ解析結果がありません。スタディ「静電場」で Solve を実行してください)</div>;
  }
  const isAxisym = isAxisymmetric(coord);
  const capUnit = isAxisym ? "F" : "F/m";
  const qUnit = isAxisym ? "C" : "C/m";
  return (
    <>
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
  // Mesh ボタン (解析なしでメッシュ生成のみ) の結果。Solve 結果とは独立に保持する
  const [meshResult, setMeshResult] = useState<MeshResult | null>(null);
  const [showMesh, setShowMesh] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
  // 「結果表示」セレクトの選択と対数スケールチェックボックス。新規実行開始時に既定 (ライブ/線形) へ戻す
  const [picResultField, setPicResultField] = useState<PicResultField>("live");
  const [picLogScale, setPicLogScale] = useState(false);
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

  // ガス流れ (DSMC) 設定は project.dsmc として project state 本体に置く (particles/pic と異なり
  // 独立 state を持たず、ジオメトリ・メッシュ設定と同様 commitProject 経由で Undo/Redo 対象になる)。
  // 実行結果・実行状態は他パネルの result 系 state と同様に App 側で保持する
  const [gasRunning, setGasRunning] = useState(false);
  const [gasError, setGasError] = useState<string | null>(null);
  const [gasResult, setGasResult] = useState<DsmcResult | null>(null);
  // 実行中の進捗 (started/progress メッセージから更新。未実行/完了後は null)
  const [gasProgress, setGasProgress] = useState<{ step: number; nSteps: number; nParticles: number } | null>(null);
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

  // アンマウント時に WebSocket 接続を確実に閉じる
  useEffect(() => {
    return () => {
      picClientRef.current?.close();
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
    setMeshResult(null);
    setProfileLine(null);
    setTraceResult(null); // ジオメトリ変更で解析結果とともに trace 結果も破棄する
    setGasResult(null); // メッシュが変わりうるため DSMC 結果 (要素値) も破棄する
    setPicProjectChangedSinceRun(true); // PIC続き実行はサーバー状態と食い違うため無効化する
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
    setMeshResult(null);
    setProfileLine(null);
    setTraceResult(null);
    setGasResult(null);
    setSelectedRegionId((sel) => ensureSelection(prev, sel));
    setPicProjectChangedSinceRun(true); // PIC続き実行はサーバー状態と食い違うため無効化する
    setGasProjectChangedSinceRun(true); // Undo は任意の過去状態へ飛びうるため常に無効化する
  }, [history, ensureSelection]);

  const doRedo = useCallback(() => {
    const next = history.redo(projectRef.current);
    if (next === null) return;
    projectRef.current = next;
    setProjectState(next);
    setResult(null);
    setMeshResult(null);
    setProfileLine(null);
    setTraceResult(null);
    setGasResult(null);
    setSelectedRegionId((sel) => ensureSelection(next, sel));
    setPicProjectChangedSinceRun(true); // PIC続き実行はサーバー状態と食い違うため無効化する
    setGasProjectChangedSinceRun(true); // Redo も任意の過去状態へ飛びうるため常に無効化する
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
    setBusy(true);
    setError(null);
    setMeshResult(null); // Solve 実行時は Mesh のみの結果を破棄し、Solve 側の表示を優先する
    try {
      setResult(await api.solve(project));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  // メッシュ生成のみ (解析は行わない)
  const runMesh = async () => {
    setBusy(true);
    setError(null);
    try {
      setMeshResult(await api.mesh(project));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  // 粒子軌道トレース実行 (project.particles として送信する)
  const runTrace = async () => {
    setBusy(true);
    setError(null);
    try {
      setTraceResult(await api.trace({ ...project, particles }));
    } catch (e) {
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

  // パラメータスイープに渡すプロジェクト (particles/pic は独立 state のためここで合成する。
  // saveProject/runPicStart と同じ合成方法。b_field/dsmc/pic の候補パスや現在値のプレビューは
  // このオブジェクトを基準に組み立てるため、常に最新の pic/particles を反映させる)
  const projectForSweep: Project = { ...project, particles, pic: withInjectionEmitter(pic, particles.emitter) };

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
  const runSweepStart = (paramPath: string, values: number[], parallel: number) => {
    setSweepError(null);
    setSweepStarted(null);
    setSweepCases([]);
    setSweepLoadedCaseIndex(null);
    setSweepRunning(true);
    const client = new SweepClient(makeSweepCallbacks());
    sweepClientRef.current = client;
    client.start(projectForSweep, paramPath, values, parallel);
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
      setPicCycle(msg.cycle ?? null);
      // collectors 配列が来ればそちらを使い、旧バックエンドで単数 collector のみの場合は
      // 先頭(1個)として扱う (prompts/37、必須ではないが安全に対応しておく)
      setPicCollectors(msg.collectors ?? (msg.collector ? [msg.collector] : []));
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
    setPicError(null);
    setPicStarted(null);
    setPicFrame(null);
    setPicHistory([]);
    setPicRunning(true);
    setPicContinueReady(false);
    // 新しい実行を開始したら結果フィールド表示 (前回 done の残骸) をリセットする
    setPicFields(null);
    setPicTiming(null); // 位相別プロファイル計測 (prompts/75) も前回 done の残骸を消す
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
    setPicError(null);
    setPicRunning(true);
    setPicContinueReady(false);
    // 表示状態を新規実行と同様にリセットする。描画優先順位が
    // 「周期アニメ > 結果フィールド > ライブ」のため、前回 done の cycle / 結果フィールド
    // 選択が残っているとライブ表示が隠れ、続き実行中の画面が追従しない (不具合修正)。
    // 新しい fields / cycle / collectors は追加区間の done で置き換わる
    setPicFields(null);
    setPicTiming(null);
    setPicResultField("live");
    setPicLogScale(false);
    setPicCycle(null);
    setCyclePlaying(false);
    setCycleBinIndex(0);
    setCycleViewActive(false);
    setPicCollectors([]);
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
  // particles / pic は history 管理外の別 state のため、保存時にここで project へ合成する
  const saveProject = () => {
    const toSave: Project = { ...project, particles, pic: withInjectionEmitter(pic, particles.emitter) };
    saveTextFile("project.json", JSON.stringify(toSave, null, 2), "JSON", ["json"]).catch((err) => {
      setError(String(err));
    });
  };

  // 結果付き保存: プロジェクトに results を同梱して1ファイルで保存する。
  // 結果 (特に PIC の cycle) は大きくなりうるため、整形なし (compact) で書き出す
  const saveProjectWithResults = () => {
    const toSave: Project = { ...project, particles, pic: withInjectionEmitter(pic, particles.emitter) };
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
          }
        : null,
      gas: gasResult,
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
    setSelectedCollectorIndexRaw(null);
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
      // 結果表示セレクトは既定 (ライブ/線形) へ戻す
      setPicResultField("live");
      setPicLogScale(false);
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

  // 「結果付き保存」ボタンの有効条件: FEM/Mesh/トレース/PIC/DSMC のいずれかの結果があること
  // (何も無い状態で保存しても project 部分だけの通常保存と同じになってしまうため無効化する)
  const hasAnyResults = !!result || !!meshResult || !!traceResult || !!gasResult || !!picStarted;

  // 「続きから実行」ボタンの有効条件: 直前の実行が done/stop 済みで現在実行中でなく、
  // かつ前回実行以降にジオメトリが編集されていないこと (health 未接続時も不可)
  const picCanContinue = !!health && picContinueReady && !picRunning && !picProjectChangedSinceRun;

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

  // コレクタ一覧の選択インデックス (範囲外・未選択なら先頭を既定選択とする)
  const selectedCollectorIndex: number | null =
    selectedCollectorIndexRaw !== null && selectedCollectorIndexRaw < collectorsList.length
      ? selectedCollectorIndexRaw
      : collectorsList.length > 0
        ? 0
        : null;

  // PICライブ描画用ビュー (started の mesh + 最新 frame)。実行中〜done後の最終フレームまで保持する
  const picLiveFrame: PicLiveFrame | null =
    picStarted && picFrame
      ? { mesh: picStarted.mesh, phi: picFrame.phi, particles: picFrame.particles }
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

  // 描画優先順位: 周期アニメーション > 結果フィールド表示 (PIC) > ガス流れ結果表示
  // (> ライブ表示 > Solve結果、CadCanvas側で処理)。
  // CadCanvas へは既存の picFieldView prop をそのまま使い回す (新規propは増やさない)。
  // ただしガス関連ノード選択中は PIC 側のビュー (周期アニメ・結果フィールド・ライブ) を
  // 抑止してガス結果を優先する (不具合修正: PIC 実行後に「ガス流れ結果」を開いても
  // PIC の結果フィールドが優先チェーンで勝ち続け、DSMC の数密度等が見えなかった)
  const onGasNode = activeNode === "study-gas" || activeNode === "result-gas";
  // 「結果 — 粒子追跡」ノード選択中は背景表示 (traceBackground) を CadCanvas の
  // result/fieldView に反映する (なし=背景の色マップ・等値線・ベクトルを消す)
  const onTraceResultNode = activeNode === "result-trace";
  const finalPicFieldView = onGasNode ? gasFieldView : (picCycleView ?? picFieldView);

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
  // エラーは error → picError → gasError → sweepError の順で最初の非null を優先表示する
  const statusError = error ?? picError ?? gasError ?? sweepError;
  // スイープの完了ケース数 (進捗表示用。PIC/DSMC 単発実行と同列の優先度で表示する)
  const sweepCompletedCount = sweepCases.filter((c) => c.status === "done" || c.status === "error").length;
  // 続きから実行では frame.step が通算で進むため、区間開始オフセットを引いて計算する
  // (不具合修正: 分子だけ通算になり 100% 超の表示になっていた)
  const picStepOffset = picStarted?.step_offset ?? 0;
  const picSegStep = Math.max(0, (picFrame?.step ?? picStepOffset) - picStepOffset);
  const picPct = picStarted && picStarted.n_steps > 0
    ? Math.min(100, Math.round((picSegStep / picStarted.n_steps) * 100))
    : 0;
  const gasPct = gasProgress && gasProgress.nSteps > 0 ? Math.round((gasProgress.step / gasProgress.nSteps) * 100) : 0;

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
          className={`status ${health ? "ok" : "ng"}`}
          title="開発時は uvicorn の --port をこの値に合わせてください。配布版ではアプリ再起動後にサイドカーへ反映されます"
        >
          {health
            ? `backend v${health.version} ${health.gpu ? "(GPU)" : "(CPU)"}${health.numba ? " numba" : ""}`
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
                resultField={picResultField}
                onResultFieldChange={(v) => {
                  // 結果表示の切替時はアニメ優先を解除し、選択したフィールドを表示する
                  setPicResultField(v);
                  setCycleViewActive(false);
                  setCyclePlaying(false);
                }}
                logScale={picLogScale}
                onLogScaleChange={setPicLogScale}
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
                resultField={picResultField}
                onResultFieldChange={(v) => {
                  // 結果表示の切替時はアニメ優先を解除し、選択したフィールドを表示する
                  setPicResultField(v);
                  setCycleViewActive(false);
                  setCyclePlaying(false);
                }}
                logScale={picLogScale}
                onLogScaleChange={setPicLogScale}
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
                mode="results"
              />
            </div>

            {/* study-gas (設定+実行UI) と result-gas (結果専用) は同じ props を渡す
                GasPanel の2インスタンスで、mode だけを切り替えて表示する */}
            <div style={{ display: showGasSetupPage ? "block" : "none" }}>
              <GasPanel
                project={project}
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
              <ResultSummary result={result} coord={project.coord} />
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

        {/* 右カラム: キャンバスツールバー + CadCanvas */}
        <div className="canvas-col">
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
          </div>

          <CadCanvas
            project={project}
            lengthUnit={lengthUnit}
            result={onTraceResultNode && traceBackground === "none" ? null : result}
            meshResult={meshResult}
            showMesh={showMesh}
            tool={tool}
            gridSnap={gridSnap}
            rulerFontSize={rulerFontSize}
            selectedRegionId={selectedRegionId}
            fieldView={onTraceResultNode && traceBackground !== "none" ? traceBackground : fieldView}
            showIsolines={showIsolines}
            showVectors={showVectors}
            profileLine={profileLine}
            collectors={collectorsList}
            selectedCollectorIndex={selectedCollectorIndex}
            emitter={showEmitter ? particles.emitter : null}
            traceResult={traceResult}
            showTrajectories={showTrajectories}
            picFrame={onGasNode ? null : picLiveFrame}
            picFieldView={finalPicFieldView}
            gasParticles={gasRunning && gasShowParticles ? gasLiveParticles : null}
            gasBoundaries={gasBoundariesList}
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
          />
          {profileLine && (
            <ProfilePanel
              project={project}
              lengthUnit={lengthUnit}
              p1={profileLine[0]}
              p2={profileLine[1]}
              onClose={() => setProfileLine(null)}
            />
          )}
        </div>
      </div>

      {/* 下部ステータスバー: エラー > 静電場/トレース計算中 > PIC実行中 > DSMC実行中 > 準備完了 の優先順位 */}
      <div className="statusbar">
        {statusError ? (
          <span className="statusbar-error">{statusError}</span>
        ) : busy ? (
          <span>静電場/トレース 計算中...</span>
        ) : picRunning ? (
          <>
            <span>
              PIC-MCC 実行中... {picPct}% ({picSegStep}/{picStarted?.n_steps ?? 0})
            </span>
            <div className="statusbar-progress">
              <div className="statusbar-progress-bar" style={{ width: `${picPct}%` }} />
            </div>
          </>
        ) : gasRunning ? (
          <>
            <span>DSMC 実行中... {gasPct}%</span>
            <div className="statusbar-progress">
              <div className="statusbar-progress-bar" style={{ width: `${gasPct}%` }} />
            </div>
          </>
        ) : sweepRunning ? (
          <>
            <span>
              スイープ実行中... {sweepCompletedCount}/{sweepStarted?.n_cases ?? 0} ケース完了
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
