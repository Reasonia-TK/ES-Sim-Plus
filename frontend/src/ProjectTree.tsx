import { useState } from "react";
import { EDGE_LABELS_RZ, EDGE_LABELS_RZ_X0, EDGE_LABELS_XY } from "./panels/FieldPanel";
import { LENGTH_UNIT_LABEL, mToUnit } from "./units";
import type { LengthUnit } from "./units";
import { isAxisymmetric, rfComponents } from "./types";
import type {
  DsmcResult,
  EdgeBcType,
  Fluid1dFrameMsg,
  Fluid1dResult,
  Fluid1dStartedMsg,
  Pic1dFrameMsg,
  Pic1dResult,
  Pic1dStartedMsg,
  PicDiag,
  PicFields,
  PicFrameMsg,
  PicStartedMsg,
  Project,
  SolveResult,
  TlResult,
  TlStartedMsg,
  TraceResult,
  VoltageRf,
} from "./types";

/**
 * 左カラムのプロジェクトツリーが表現するノード。
 * インスペクタ (中カラム) はこの選択に応じて表示ページを切替える (App.tsx 側で管理)。
 * result-trace/result-pic/result-gas は study-trace/study-pic/study-gas と同じ
 * インスペクタページを指す (実装単純化のため、prompts/60 参照)。
 */
export type TreeNode =
  | "domain"
  | "regions"
  | "boundary"
  | "mesh"
  | "bfield"
  | "study-fem"
  | "study-trace"
  | "study-pic"
  | "study-pic1d"
  | "study-fluid1d"
  | "study-tl"
  | "study-gas"
  | "study-sweep"
  | "result-fem"
  | "result-trace"
  | "result-pic"
  | "result-pic1d"
  | "result-fluid1d"
  | "result-tl"
  | "result-gas";

interface Props {
  project: Project;
  // 長さの表示単位 (mm/µm)。project 内部は常に m のまま
  lengthUnit: LengthUnit;
  activeNode: TreeNode;
  onSelectNode: (node: TreeNode) => void;
  selectedRegionId: string | null;
  onSelectRegion: (id: string) => void;
  // 境界条件の子ノード (辺) 選択中のインデックス。boundary 以外では null
  edgeFilter: number | null;
  onSelectEdge: (edgeIndex: number) => void;
  edgeState: (
    edgeIndex: number,
  ) => { type: EdgeBcType; voltage: number; voltageRf?: VoltageRf | VoltageRf[]; seeGamma: number };
  // 静電場FEM/粒子軌道追跡 (Solve/Mesh/Trace は同じ busy フラグを共用している、App.tsx 参照)
  busy: boolean;
  result: SolveResult | null;
  traceResult: TraceResult | null;
  picRunning: boolean;
  picStarted: PicStartedMsg | null;
  picFrame: PicFrameMsg | null;
  picError: string | null;
  picFields: PicFields | null;
  picHistory: PicDiag[];
  // 1D PIC/MCC (prompts/91)。2D の PIC (pic) と完全に独立の実行状態・結果を持つ
  pic1dRunning: boolean;
  pic1dStarted: Pic1dStartedMsg | null;
  pic1dFrame: Pic1dFrameMsg | null;
  pic1dError: string | null;
  pic1dResult: Pic1dResult | null;
  // 1D プラズマ流体 (prompts/104-108)。pic1d と完全に独立の実行状態・結果を持つ。
  // optional にしているのは、App.tsx への統合が Phase D2 (prompts/109) 側の作業であり、
  // 現時点の (未統合の) App.tsx がこれらの props を渡していなくても型検査が通るようにするため
  // (App.tsx は触らない、prompts/108 の制約)
  fluid1dRunning?: boolean;
  fluid1dStarted?: Fluid1dStartedMsg | null;
  fluid1dFrame?: Fluid1dFrameMsg | null;
  fluid1dError?: string | null;
  fluid1dResult?: Fluid1dResult | null;
  // VHF 定在波 (prompts/101)。continue が無いので進捗は started/progress のステップ数のみ
  tlRunning: boolean;
  tlStarted: TlStartedMsg | null;
  tlProgressStep: number | null;
  tlError: string | null;
  tlResult: TlResult | null;
  gasRunning: boolean;
  gasProgress: { step: number; nSteps: number; nParticles: number } | null;
  gasError: string | null;
  gasResult: DsmcResult | null;
  // パラメータスイープ (prompts/79): 実行中フラグ・完了ケース数/総ケース数・エラー有無
  sweepRunning: boolean;
  sweepCompleted: number;
  sweepTotal: number;
  sweepHasError: boolean;
}

type BadgeKind = "busy" | "done" | "error" | "idle";

function StatusBadge({ text, kind }: { text: string; kind: BadgeKind }) {
  return <span className={`tree-status tree-status-${kind}`}>{text}</span>;
}

// パーセンテージ表示 (分母0/未定義時は0%扱い、NaN表示を避ける)
function pct(numer: number, denom: number): number {
  if (!(denom > 0)) return 0;
  return Math.round((numer / denom) * 100);
}

export default function ProjectTree({
  project,
  lengthUnit,
  activeNode,
  onSelectNode,
  selectedRegionId,
  onSelectRegion,
  edgeFilter,
  onSelectEdge,
  edgeState,
  busy,
  result,
  traceResult,
  picRunning,
  picStarted,
  picFrame,
  picError,
  picFields,
  picHistory,
  pic1dRunning,
  pic1dStarted,
  pic1dFrame,
  pic1dError,
  pic1dResult,
  fluid1dRunning = false,
  fluid1dStarted = null,
  fluid1dFrame = null,
  fluid1dError = null,
  fluid1dResult = null,
  tlRunning,
  tlStarted,
  tlProgressStep,
  tlError,
  tlResult,
  gasRunning,
  gasProgress,
  gasError,
  gasResult,
  sweepRunning,
  sweepCompleted,
  sweepTotal,
  sweepHasError,
}: Props) {
  const [search, setSearch] = useState("");
  const [openGeo, setOpenGeo] = useState(true);
  const [openStudy, setOpenStudy] = useState(true);
  const [openResult, setOpenResult] = useState(true);

  const q = search.trim().toLowerCase();
  // ノード名の部分一致フィルタ (空文字なら常に表示 = 全表示)
  const match = (label: string) => q === "" || label.toLowerCase().includes(q);

  const coord = project.coord ?? "xy";
  const isRz = coord === "rz";
  const isRzX0 = coord === "rz_x0";
  const isAxisym = isAxisymmetric(coord);
  const edgeLabels = isRz ? EDGE_LABELS_RZ : isRzX0 ? EDGE_LABELS_RZ_X0 : EDGE_LABELS_XY;
  const axisEdge = isRz ? 0 : isRzX0 ? 3 : null;

  // 境界条件の子ノードの要約テキスト (対称軸の辺は固定表示)
  const edgeSummary = (i: number): string => {
    if (axisEdge === i) return "対称軸";
    const st = edgeState(i);
    if (st.type === "neumann") return "なし";
    if (st.type === "dirichlet") {
      const rf = rfComponents(st.voltageRf).length > 0 ? " +RF" : "";
      return `Dirichlet ${st.voltage}V${rf}`;
    }
    if (st.type === "symmetry") return "対称 (反射)";
    return "周期";
  };

  const femBadge = (): { text: string; kind: BadgeKind } => {
    if (busy) return { text: "実行中", kind: "busy" };
    if (result) return { text: "✓完了", kind: "done" };
    return { text: "未実行", kind: "idle" };
  };
  const traceBadge = (): { text: string; kind: BadgeKind } => {
    if (busy) return { text: "実行中", kind: "busy" };
    if (traceResult) return { text: "✓完了", kind: "done" };
    return { text: "未実行", kind: "idle" };
  };
  const picBadge = (): { text: string; kind: BadgeKind } => {
    if (picRunning) {
      const p = pct(picFrame?.step ?? 0, picStarted?.n_steps ?? 0);
      return { text: `実行中 ${p}%`, kind: "busy" };
    }
    if (picError) return { text: "エラー", kind: "error" };
    if (picFields || picHistory.length > 0) return { text: "✓完了", kind: "done" };
    return { text: "未実行", kind: "idle" };
  };
  const pic1dBadge = (): { text: string; kind: BadgeKind } => {
    if (pic1dRunning) {
      const p = pct(pic1dFrame?.step ?? 0, pic1dStarted?.n_steps ?? 0);
      return { text: `実行中 ${p}%`, kind: "busy" };
    }
    if (pic1dError) return { text: "エラー", kind: "error" };
    if (pic1dResult) return { text: "✓完了", kind: "done" };
    return { text: "未実行", kind: "idle" };
  };
  const fluid1dBadge = (): { text: string; kind: BadgeKind } => {
    if (fluid1dRunning) {
      const p = pct(fluid1dFrame?.step ?? 0, fluid1dStarted?.n_steps ?? 0);
      return { text: `実行中 ${p}%`, kind: "busy" };
    }
    if (fluid1dError) return { text: "エラー", kind: "error" };
    if (fluid1dResult) return { text: "✓完了", kind: "done" };
    return { text: "未実行", kind: "idle" };
  };
  const tlBadge = (): { text: string; kind: BadgeKind } => {
    if (tlRunning) {
      const p = tlStarted ? pct(tlProgressStep ?? 0, tlStarted.n_steps) : 0;
      return { text: `実行中 ${p}%`, kind: "busy" };
    }
    if (tlError) return { text: "エラー", kind: "error" };
    if (tlResult) return { text: "✓完了", kind: "done" };
    return { text: "未実行", kind: "idle" };
  };
  const gasBadge = (): { text: string; kind: BadgeKind } => {
    if (gasRunning) {
      const p = pct(gasProgress?.step ?? 0, gasProgress?.nSteps ?? 0);
      return { text: `実行中 ${p}%`, kind: "busy" };
    }
    if (gasError) return { text: "エラー", kind: "error" };
    if (gasResult) return { text: "✓完了", kind: "done" };
    return { text: "未実行", kind: "idle" };
  };
  const sweepBadge = (): { text: string; kind: BadgeKind } => {
    if (sweepRunning) return { text: `実行中 ${sweepCompleted}/${sweepTotal}`, kind: "busy" };
    if (sweepHasError) return { text: "エラー", kind: "error" };
    if (sweepTotal > 0 && sweepCompleted === sweepTotal) return { text: "✓完了", kind: "done" };
    return { text: "未実行", kind: "idle" };
  };

  const regions = project.geometry.regions;
  const showRegionsGroup = match("領域") || regions.some((r) => match(r.id));
  const showBoundaryGroup = match("境界条件") || edgeLabels.some((label) => match(label));

  return (
    <div className="tree-search-wrap">
      <input
        className="tree-search"
        type="text"
        placeholder="ノードを検索..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />

      <div className="tree-scroll">
        {/* --- ジオメトリ (プリ) --- */}
        <div className="tree-section">
          <div className="tree-section-header" onClick={() => setOpenGeo(!openGeo)}>
            <span className="tree-caret">{openGeo ? "▼" : "▶"}</span>
            <span className="tree-section-title">ジオメトリ</span>
            <span className="tree-phase-badge pre">プリ</span>
          </div>
          {openGeo && (
            <div className="tree-section-body">
              {match("ドメイン") && (
                <div
                  className={`tree-row ${activeNode === "domain" ? "active" : ""}`}
                  onClick={() => onSelectNode("domain")}
                >
                  <span>ドメイン</span>
                </div>
              )}

              {showRegionsGroup && (
                <>
                  <div className="tree-row tree-row-group">領域 ({regions.length})</div>
                  {regions
                    .filter((r) => q === "" || match("領域") || match(r.id))
                    .map((r) => (
                      <div
                        key={r.id}
                        className={`tree-row tree-row-indent2 ${
                          activeNode === "regions" && selectedRegionId === r.id ? "active" : ""
                        }`}
                        onClick={() => onSelectRegion(r.id)}
                      >
                        <span>{r.id}</span>
                        <span className="tree-row-sub">{r.type}</span>
                      </div>
                    ))}
                  {regions.length === 0 && <div className="tree-row tree-row-indent2 muted">(領域なし)</div>}
                </>
              )}

              {showBoundaryGroup && (
                <>
                  <div className="tree-row tree-row-group">境界条件</div>
                  {edgeLabels.map((label, i) => {
                    if (!(q === "" || match("境界条件") || match(label))) return null;
                    return (
                      <div
                        key={i}
                        className={`tree-row tree-row-indent2 ${
                          activeNode === "boundary" && edgeFilter === i ? "active" : ""
                        }`}
                        onClick={() => onSelectEdge(i)}
                      >
                        <span>{label}</span>
                        <span className="tree-row-sub">{edgeSummary(i)}</span>
                      </div>
                    );
                  })}
                </>
              )}

              {match("メッシュ") && (
                <div
                  className={`tree-row ${activeNode === "mesh" ? "active" : ""}`}
                  onClick={() => onSelectNode("mesh")}
                >
                  <span>メッシュ</span>
                  <span className="tree-row-sub">
                    {mToUnit(project.mesh.size, lengthUnit).toFixed(2)}
                    {LENGTH_UNIT_LABEL[lengthUnit]}
                  </span>
                </div>
              )}

              {match("磁場") && (
                <div
                  className={`tree-row ${activeNode === "bfield" ? "active" : ""}`}
                  onClick={() => onSelectNode("bfield")}
                >
                  <span>磁場</span>
                  {project.b_field && (
                    <span className="tree-row-sub">
                      Bx{project.b_field.bx},By{project.b_field.by},Bz{project.b_field.bz}
                    </span>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* --- スタディ (メイン) --- */}
        <div className="tree-section">
          <div className="tree-section-header" onClick={() => setOpenStudy(!openStudy)}>
            <span className="tree-caret">{openStudy ? "▼" : "▶"}</span>
            <span className="tree-section-title">スタディ</span>
            <span className="tree-phase-badge main">メイン</span>
          </div>
          {openStudy && (
            <div className="tree-section-body">
              {match("静電場") && (
                <div
                  className={`tree-row ${activeNode === "study-fem" ? "active" : ""}`}
                  onClick={() => onSelectNode("study-fem")}
                >
                  <span>静電場</span>
                  <StatusBadge {...femBadge()} />
                </div>
              )}
              {match("粒子追跡") && (
                <div
                  className={`tree-row ${activeNode === "study-trace" ? "active" : ""}`}
                  onClick={() => onSelectNode("study-trace")}
                >
                  <span>粒子追跡</span>
                  <StatusBadge {...traceBadge()} />
                </div>
              )}
              {match("PIC-MCC") && (
                <div
                  className={`tree-row ${activeNode === "study-pic" ? "active" : ""}`}
                  onClick={() => onSelectNode("study-pic")}
                >
                  <span>PIC-MCC</span>
                  <StatusBadge {...picBadge()} />
                </div>
              )}
              {match("PIC-MCC 1D") && (
                <div
                  className={`tree-row ${activeNode === "study-pic1d" ? "active" : ""}`}
                  onClick={() => onSelectNode("study-pic1d")}
                >
                  <span>PIC-MCC 1D</span>
                  <StatusBadge {...pic1dBadge()} />
                </div>
              )}
              {match("流体 (1D)") && (
                <div
                  className={`tree-row ${activeNode === "study-fluid1d" ? "active" : ""}`}
                  onClick={() => onSelectNode("study-fluid1d")}
                >
                  <span>流体 (1D)</span>
                  <StatusBadge {...fluid1dBadge()} />
                </div>
              )}
              {match("VHF定在波") && (
                <div
                  className={`tree-row ${activeNode === "study-tl" ? "active" : ""}`}
                  onClick={() => onSelectNode("study-tl")}
                >
                  <span>VHF定在波</span>
                  <StatusBadge {...tlBadge()} />
                </div>
              )}
              {match("DSMC") && (
                <div
                  className={`tree-row ${activeNode === "study-gas" ? "active" : ""}`}
                  onClick={() => onSelectNode("study-gas")}
                >
                  <span>DSMC</span>
                  <StatusBadge {...gasBadge()} />
                </div>
              )}
              {match("スイープ") && (
                <div
                  className={`tree-row ${activeNode === "study-sweep" ? "active" : ""}`}
                  onClick={() => onSelectNode("study-sweep")}
                >
                  <span>パラメータスイープ</span>
                  <StatusBadge {...sweepBadge()} />
                </div>
              )}
            </div>
          )}
        </div>

        {/* --- 結果 (ポスト) --- */}
        <div className="tree-section">
          <div className="tree-section-header" onClick={() => setOpenResult(!openResult)}>
            <span className="tree-caret">{openResult ? "▼" : "▶"}</span>
            <span className="tree-section-title">結果</span>
            <span className="tree-phase-badge post">ポスト</span>
          </div>
          {openResult && (
            <div className="tree-section-body">
              {/* 静電場の結果 (電位分布φ/電場|E|/ラインプロファイル) は他モジュールと同様
                  1モジュール=1結果ノードに揃えるため単一ノードにまとめる (prompts/69) */}
              {match("静電場") && (
                <div
                  className={`tree-row ${activeNode === "result-fem" ? "active" : ""}`}
                  onClick={() => onSelectNode("result-fem")}
                >
                  静電場
                </div>
              )}
              {match("粒子追跡") && (
                <div
                  className={`tree-row ${activeNode === "result-trace" ? "active" : ""}`}
                  onClick={() => onSelectNode("result-trace")}
                >
                  粒子追跡
                </div>
              )}
              {match("PIC-MCC") && (
                <div
                  className={`tree-row ${activeNode === "result-pic" ? "active" : ""}`}
                  onClick={() => onSelectNode("result-pic")}
                >
                  PIC-MCC
                </div>
              )}
              {match("PIC-MCC 1D") && (
                <div
                  className={`tree-row ${activeNode === "result-pic1d" ? "active" : ""}`}
                  onClick={() => onSelectNode("result-pic1d")}
                >
                  PIC-MCC 1D
                </div>
              )}
              {match("流体 (1D)") && (
                <div
                  className={`tree-row ${activeNode === "result-fluid1d" ? "active" : ""}`}
                  onClick={() => onSelectNode("result-fluid1d")}
                >
                  流体 (1D)
                </div>
              )}
              {match("VHF定在波") && (
                <div
                  className={`tree-row ${activeNode === "result-tl" ? "active" : ""}`}
                  onClick={() => onSelectNode("result-tl")}
                >
                  VHF定在波
                </div>
              )}
              {match("DSMC") && (
                <div
                  className={`tree-row ${activeNode === "result-gas" ? "active" : ""}`}
                  onClick={() => onSelectNode("result-gas")}
                >
                  DSMC
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
