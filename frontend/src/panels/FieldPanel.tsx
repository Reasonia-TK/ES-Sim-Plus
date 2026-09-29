import { useEffect, useRef, useState } from "react";
import { CommitNumberInput, CommitTextInput } from "../CommitInput";
import { Toggle } from "../Toggle";
import { LENGTH_UNIT_LABEL, mToUnit, unitToM } from "../units";
import type { LengthUnit } from "../units";
import { isAxisymmetric, rfComponents } from "../types";
import type {
  AmrRegion,
  AmrSettings,
  BField,
  CircleShape,
  EdgeBcType,
  MeshResult,
  Point,
  Project,
  Region,
  RegionType,
  SolveResult,
  VoltageRf,
  VoltageWaveform,
} from "../types";
import VoltagePreviewChart, { voltagePreviewFreqs } from "./VoltagePreviewChart";

/**
 * 静電場パネル (3カラムシェルの各種インスペクタページから共用)
 * - ジオメトリ (domain 幅/高さ)・境界条件 (4辺、RF含む)・メッシュ (サイズ)
 * - 領域一覧 + 選択中領域のプロパティ編集
 * - 解析実行 (Mesh/Solve) + 解析結果サマリ (節点数・V範囲・エネルギー等)・メッシュ結果サマリ
 * 編集操作自体は App 側の commitProject 経由で Undo/Redo 履歴に積まれる
 *
 * 3カラムシェルではプロジェクトツリーの選択ノードごとに `sections` で表示ブロックを
 * 絞り込んで使う (例: domain ノードなら sections={["domain"]})。sections 未指定時は
 * 従来通り全ブロックを表示する (後方互換、単体テスト等での利用を壊さないため)。
 */

export type FieldSection = "domain" | "boundary" | "mesh" | "bfield" | "regions" | "solve";

// RF重畳電圧の既定値 (13.56MHz の CCP を想定)。Pic1dPanel (1D、prompts/93) からも
// 同じ既定値で「RF重畳」トグルONの初期成分として使うため export する
export const DEFAULT_VOLTAGE_RF: VoltageRf = { amplitude: 100.0, freq_hz: 13.56e6, phase_deg: 0.0 };
// 2成分目以降を追加する際の既定値 (デュアル周波数の例として低周波側を想定)
const DEFAULT_VOLTAGE_RF_2ND: VoltageRf = { amplitude: 100.0, freq_hz: 2e6, phase_deg: 0.0 };

// RF重畳電圧 (単一/複数成分) の編集UI。成分ごとに振幅/周波数/位相 + 削除ボタン、
// 末尾に成分追加ボタンを表示する。全成分削除で RF 自体を無効化 (undefined) する。
// Pic1dPanel (1D の電極、prompts/93) からも共用する (export)
export function RfComponentsEditor({
  components,
  onChange,
}: {
  components: VoltageRf[];
  onChange: (next: VoltageRf[] | undefined) => void;
}) {
  return (
    <div className="rf-editor">
      {components.map((c, idx) => (
        <div className="edge-rf-row" key={idx}>
          <span className="rf-comp-label">成分{idx + 1}</span>
          <label className="rf-compact-label" title="振幅 [V]">
            A
            <CommitNumberInput
              className="rf-compact"
              value={c.amplitude}
              onCommit={(v) => onChange(components.map((cc, i) => (i === idx ? { ...cc, amplitude: v } : cc)))}
            />
          </label>
          <label className="rf-compact-label" title="周波数 [Hz]">
            f
            <CommitNumberInput
              className="rf-compact"
              value={c.freq_hz}
              onCommit={(v) => onChange(components.map((cc, i) => (i === idx ? { ...cc, freq_hz: v } : cc)))}
            />
          </label>
          <label className="rf-compact-label" title="位相 [deg]">
            φ
            <CommitNumberInput
              className="rf-compact"
              value={c.phase_deg}
              onCommit={(v) => onChange(components.map((cc, i) => (i === idx ? { ...cc, phase_deg: v } : cc)))}
            />
          </label>
          <button
            type="button"
            className="rf-remove-btn"
            title="この成分を削除"
            onClick={() => {
              const next = components.filter((_, i) => i !== idx);
              onChange(next.length > 0 ? next : undefined);
            }}
          >
            ×
          </button>
        </div>
      ))}
      <button
        type="button"
        className="rf-add-btn"
        onClick={() => onChange([...components, components.length === 0 ? DEFAULT_VOLTAGE_RF : DEFAULT_VOLTAGE_RF_2ND])}
      >
        + RF成分を追加
      </button>
    </div>
  );
}

// CSV波形インポート (prompts/73)。1列目=時間、2列目=電圧の (t, v) 行を取り出し、
// t の [t_min, t_max] を [0, 1) の正規化位相へ線形写像する。
// 末尾行 (t = t_max) は評価時の周期折返し点 ((phase[0]+1, v[0]) を仮想的に補う、
// pic.py 側) と同じ位相 (=1) に重なるため保存対象からは除く
// (含めると phase が [0,1) の範囲制約に違反するため)。
function parseWaveformCsv(text: string): { phase: number[]; v: number[]; freqHz: number } | { error: string } {
  const rows: [number, number][] = [];
  for (const rawLine of text.split(/\r\n|\r|\n/)) {
    const line = rawLine.trim();
    if (!line) continue;
    // 区切りはカンマ/タブ/空白のいずれにも対応する
    const parts = line.split(/[,\t\s]+/).filter((s) => s.length > 0);
    if (parts.length < 2) continue;
    const t = Number(parts[0]);
    const v = Number(parts[1]);
    // 数値2つに解釈できない行 (ヘッダ等) は黙ってスキップする
    if (!Number.isFinite(t) || !Number.isFinite(v)) continue;
    rows.push([t, v]);
  }
  if (rows.length < 2) return { error: "有効な数値行 (時間, 電圧) が2点未満です" };
  rows.sort((a, b) => a[0] - b[0]);
  const tMin = rows[0][0];
  const tMax = rows[rows.length - 1][0];
  if (tMax === tMin) return { error: "時間の範囲が0です (全行が同じ時刻)" };
  const kept = rows.slice(0, -1); // 末尾 (t_max) は折返し点と重複するため除外
  if (kept.length < 2) return { error: "有効な数値行 (時間, 電圧) が2点未満です" };
  return {
    phase: kept.map(([t]) => (t - tMin) / (tMax - tMin)),
    v: kept.map(([, vv]) => vv),
    freqHz: 1 / (tMax - tMin), // CSVの時間レンジをそのまま1周期と解釈した周波数を初期値にする
  };
}

// CSV波形 (voltage_waveform) の編集UI (Dirichlet辺・conductor領域で共用)。未取り込み時は
// インポートボタン、取り込み後は周波数入力+解除ボタンを表示する。RF重畳 (voltage_rf) とは独立に併用できる。
// export しているのは Pic1dPanel (1D PIC の電極波形リスト編集、prompts/91) からも流用するため
export function WaveformImportEditor({
  waveform,
  onChange,
}: {
  waveform: VoltageWaveform | undefined;
  onChange: (next: VoltageWaveform | undefined) => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const parsed = parseWaveformCsv(String(reader.result ?? ""));
      if ("error" in parsed) {
        setError(parsed.error);
        return;
      }
      setError(null);
      onChange({ freq_hz: parsed.freqHz, phase: parsed.phase, v: parsed.v });
    };
    reader.readAsText(file);
  };

  return (
    <div className="rf-editor">
      <span className="rf-comp-label">CSV波形</span>
      {waveform ? (
        <div className="edge-rf-row">
          <span>{waveform.phase.length}点 読み込み済み</span>
          <label className="rf-compact-label" title="周波数 [Hz] (1周期の繰り返し周波数)">
            f
            <CommitNumberInput
              className="rf-compact"
              value={waveform.freq_hz}
              onCommit={(freq_hz) => onChange({ ...waveform, freq_hz })}
            />
          </label>
          <button type="button" className="rf-remove-btn" title="CSV波形を解除" onClick={() => onChange(undefined)}>
            解除
          </button>
        </div>
      ) : (
        <div className="edge-rf-row">
          <button type="button" className="secondary" onClick={() => fileRef.current?.click()}>
            CSVをインポート
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".csv,text/csv"
            className="file-input"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleFile(f);
              e.target.value = "";
            }}
          />
        </div>
      )}
      {error && <div className="error">{error}</div>}
      <div className="hint">
        1列目=時間、2列目=電圧のCSV。波形は指定周波数の1周期としてループ再生されます (PICでのみ有効)。
      </div>
    </div>
  );
}

// domain 多角形の外接矩形 [x0, y0, x1, y1] [m]
function domainBBox(project: Project): [number, number, number, number] {
  const pts = project.geometry.domain.polygon;
  if (pts.length === 0) return [0, 0, 0, 0];
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

// v2 直交格子の局所細分化 (AMR、prompts/121) の編集 UI。レベル l の格子幅は size/2^l。
// 境界近傍 (導体・誘電体) と指定矩形を細分化する。max_level=0 かつ矩形なしは細分化なし
function AmrEditor({
  amr,
  lengthUnit,
  bbox,
  onChange,
}: {
  amr: AmrSettings | null | undefined;
  lengthUnit: LengthUnit;
  bbox: [number, number, number, number]; // domain の外接矩形 [x0, y0, x1, y1] [m]
  onChange: (next: AmrSettings | null) => void;
}) {
  const cur: AmrSettings = amr ?? { max_level: 0 };
  const regions = cur.regions ?? [];
  const unitLabel = LENGTH_UNIT_LABEL[lengthUnit];
  const update = (patch: Partial<AmrSettings>) => onChange({ ...cur, ...patch });
  const setRegion = (index: number, patch: Partial<AmrRegion>) =>
    update({ regions: regions.map((r, i) => (i === index ? { ...r, ...patch } : r)) });
  const addRegion = () => {
    const [x0, y0, x1, y1] = bbox;
    const w = x1 - x0;
    const h = y1 - y0;
    const r: AmrRegion = {
      p1: [x0 + 0.375 * w, y0 + 0.375 * h],
      p2: [x0 + 0.625 * w, y0 + 0.625 * h],
      level: Math.max(1, cur.max_level),
    };
    update({ regions: [...regions, r] });
  };
  const levelOptions = [1, 2, 3, 4];
  const coordInput = (value: number, commit: (v: number) => void, title: string) => (
    <span title={title}>
      <CommitNumberInput
        className="amr-coord"
        value={mToUnit(value, lengthUnit)}
        onCommit={(v) => commit(unitToM(v, lengthUnit))}
      />
    </span>
  );
  return (
    <>
      <div className="subheading">局所細分化 (AMR)</div>
      <div className="field">
        <span className="label">最大レベル</span>
        <select value={cur.max_level} onChange={(e) => update({ max_level: Number(e.target.value) })}>
          <option value={0}>0 (細分化なし)</option>
          {levelOptions.map((l) => (
            <option key={l} value={l}>
              {l} (格子幅 1/{2 ** l})
            </option>
          ))}
        </select>
      </div>
      <div className="field">
        <Toggle
          label="導体・誘電体の境界近傍を細分化"
          checked={cur.refine_boundaries ?? true}
          onChange={(v) => update({ refine_boundaries: v })}
        />
      </div>
      <div className="field">
        <span className="label">緩衝セル数</span>
        <CommitNumberInput
          value={cur.buffer_cells ?? 2}
          onCommit={(v) => {
            if (Number.isInteger(v) && v >= 0 && v <= 16) update({ buffer_cells: v });
          }}
        />
      </div>
      <div className="field">
        <span className="label">ブロック [セル]</span>
        <select
          value={cur.blocking_factor ?? 8}
          onChange={(e) => update({ blocking_factor: Number(e.target.value) })}
        >
          {[4, 8, 16].map((b) => (
            <option key={b} value={b}>
              {b}×{b}
            </option>
          ))}
        </select>
      </div>
      <div className="field">
        <Toggle
          label="解に基づく適応細分化 (静電場)"
          checked={cur.adaptive ?? false}
          onChange={(v) => update({ adaptive: v })}
        />
      </div>
      {(cur.adaptive ?? false) && (
        <>
          <div className="field">
            <span className="label">許容誤差 (電位範囲比)</span>
            <CommitNumberInput
              value={cur.adapt_tol ?? 1e-3}
              onCommit={(v) => {
                if (v > 0 && v <= 0.5) update({ adapt_tol: v });
              }}
            />
          </div>
          <div className="field">
            <span className="label">最大反復</span>
            <CommitNumberInput
              value={cur.adapt_iters ?? 3}
              onCommit={(v) => {
                if (Number.isInteger(v) && v >= 1 && v <= 8) update({ adapt_iters: v });
              }}
            />
          </div>
        </>
      )}
      <div className="field">
        <Toggle
          label="PIC の動的再格子化 (デバイ長)"
          checked={(cur.pic_regrid_every ?? 0) > 0}
          onChange={(v) => update({ pic_regrid_every: v ? 500 : 0 })}
        />
      </div>
      {(cur.pic_regrid_every ?? 0) > 0 && (
        <>
          <div className="field">
            <span className="label">再格子化の間隔 [step]</span>
            <CommitNumberInput
              value={cur.pic_regrid_every ?? 500}
              onCommit={(v) => {
                if (Number.isInteger(v) && v >= 1) update({ pic_regrid_every: v });
              }}
            />
          </div>
          <div className="field">
            <span className="label">格子幅/λ_D の上限</span>
            <CommitNumberInput
              value={cur.pic_h_over_debye ?? 1.0}
              onCommit={(v) => {
                if (v > 0 && v <= 100) update({ pic_h_over_debye: v });
              }}
            />
          </div>
        </>
      )}
      <div className="field">
        <Toggle
          label="DSMC の動的再格子化 (平均自由行程)"
          checked={(cur.dsmc_regrid_every ?? 0) > 0}
          onChange={(v) => update({ dsmc_regrid_every: v ? 500 : 0 })}
        />
      </div>
      {(cur.dsmc_regrid_every ?? 0) > 0 && (
        <>
          <div className="field">
            <span className="label">再格子化の間隔 [step]</span>
            <CommitNumberInput
              value={cur.dsmc_regrid_every ?? 500}
              onCommit={(v) => {
                if (Number.isInteger(v) && v >= 1) update({ dsmc_regrid_every: v });
              }}
            />
          </div>
          <div className="field">
            <span className="label">格子幅/平均自由行程 の上限</span>
            <CommitNumberInput
              value={cur.dsmc_h_over_mfp ?? 0.5}
              onCommit={(v) => {
                if (v > 0 && v <= 100) update({ dsmc_h_over_mfp: v });
              }}
            />
          </div>
        </>
      )}
      <div className="hint">
        境界近傍は最大レベルまで、下の矩形は指定レベルまで格子幅を 1/2 ずつ細かくします
        (隣り合うセルのレベル差は 1 以下)。静電場は代数マルチグリッド (GPU があれば GPU) で、
        PIC も同じ細分化格子の上で解きます。適応細分化は Solve 時に誤差の大きい所を最大レベルまで
        自動で細かくします (静電場のみ。Mesh ボタンの表示は適応前の格子です)。PIC の動的再格子化は
        実行中に電子の密度・温度から求めたデバイ長 λ_D に合わせて格子を作り直します (時間平均区間の
        前だけ。ライブ表示の格子も更新されます)。DSMC は葉セルを衝突・サンプリングのセルにし、動的
        再格子化では区間平均の密度・温度の平均自由行程に合わせます (時間刻みの既定は最細レベルで決まります)。
        流体 2D も同じ細分化格子 (静的な細分化のみ) の上で解きます。
      </div>
      <div className="collector-list">
        {regions.length === 0 && <div className="muted">(細分化矩形なし)</div>}
        {regions.map((r, i) => (
          <div key={i} className="amr-region-row">
            <span className="amr-region-label">{`R${i + 1}`}</span>
            {coordInput(r.p1[0], (v) => setRegion(i, { p1: [v, r.p1[1]] }), `x1 [${unitLabel}]`)}
            {coordInput(r.p1[1], (v) => setRegion(i, { p1: [r.p1[0], v] }), `y1 [${unitLabel}]`)}
            <span className="muted">–</span>
            {coordInput(r.p2[0], (v) => setRegion(i, { p2: [v, r.p2[1]] }), `x2 [${unitLabel}]`)}
            {coordInput(r.p2[1], (v) => setRegion(i, { p2: [r.p2[0], v] }), `y2 [${unitLabel}]`)}
            <select
              value={r.level}
              title="この矩形の細分化レベル"
              onChange={(e) => setRegion(i, { level: Number(e.target.value) })}
            >
              {levelOptions.map((l) => (
                <option key={l} value={l}>
                  L{l}
                </option>
              ))}
            </select>
            <button
              type="button"
              className="danger collector-delete"
              onClick={() => update({ regions: regions.filter((_, k) => k !== i) })}
              title="この細分化矩形を削除"
            >
              ×
            </button>
          </div>
        ))}
      </div>
      <button type="button" className="rf-add-btn" onClick={addRegion}>
        + 細分化矩形を追加
      </button>
      <div className="hint">矩形は対角 2 点 (x1, y1)–(x2, y2) [{unitLabel}] で指定します。</div>
    </>
  );
}

// 矩形 domain の外周エッジ順: 0=下, 1=右, 2=上, 3=左
// (ProjectTree でもエッジ名を揃えて表示するため export する)
export const EDGE_LABELS_XY = ["下 (y=0)", "右 (x=w)", "上 (y=h)", "左 (x=0)"];
// 軸対称 (r-z) モード: x=z (軸方向)・y=r (径方向)。下辺 (y=0) は対称軸 (r=0)
export const EDGE_LABELS_RZ = ["対称軸 (r=0)", "右 (z=L)", "上 (r=R)", "左 (z=0)"];
// 軸対称 (r-z、左辺が軸) モード: x=r (径方向)・y=z (軸方向)。左辺 (x=0) は対称軸 (r=0)
export const EDGE_LABELS_RZ_X0 = ["下 (z=0)", "右 (r=R)", "上 (z=L)", "対称軸 (r=0)"];

interface Props {
  project: Project;
  // 長さの表示・入力単位 (mm/µm)。project 内部は常に m のまま
  lengthUnit: LengthUnit;
  domainW: number;
  domainH: number;
  setDomainSize: (w: number, h: number) => void;
  setCoord: (coord: "xy" | "rz" | "rz_x0") => void;
  edgeState: (edgeIndex: number) => {
    type: EdgeBcType;
    voltage: number;
    voltageRf?: VoltageRf | VoltageRf[];
    voltageWaveform?: VoltageWaveform;
    seeGamma: number;
  };
  setEdgeType: (edgeIndex: number, type: EdgeBcType) => void;
  setEdgeVoltage: (edgeIndex: number, voltage: number) => void;
  setEdgeVoltageRf: (edgeIndex: number, voltage_rf: VoltageRf | VoltageRf[] | undefined) => void;
  setEdgeVoltageWaveform: (edgeIndex: number, voltage_waveform: VoltageWaveform | undefined) => void;
  setEdgeSeeGamma: (edgeIndex: number, see_gamma: number) => void;
  setMeshSize: (size: number) => void;
  setMeshMode: (mode: "unstructured" | "structured" | "cartesian") => void;
  // v2 直交格子の局所細分化 (AMR、prompts/121)。null で解除
  setMeshAmr: (amr: AmrSettings | null) => void;
  setBField: (patch: Partial<BField>) => void;
  meshResult: MeshResult | null;
  selectedRegionId: string | null;
  onSelectRegion: (id: string) => void;
  selected: Region | null;
  renameRegion: (oldId: string, newId: string) => void;
  setRegionType: (id: string, type: RegionType) => void;
  editRegionShape: (id: string, shape: CircleShape) => void;
  editRegionPolygon: (id: string, polygon: Point[]) => void;
  // 隣接する2領域のマージ (成功時 null、失敗時はエラーメッセージ文字列)
  mergeRegions: (targetId: string, otherId: string) => string | null;
  updateRegion: (id: string, patch: Partial<Region>) => void;
  deleteRegion: (id: string) => void;
  // 領域ごとのローカルメッシュサイズ [m]。null で解除 (全体サイズを使用)
  setRegionLocalSize: (id: string, size: number | null) => void;
  // 辺 (線分) ローカルメッシュサイズ一覧の編集 (prompts/90)
  updateEdgeMeshSize: (index: number, size: number) => void;
  deleteEdgeMeshSize: (index: number) => void;
  result: SolveResult | null;
  // 表示するセクションの絞り込み (プロジェクトツリーの選択ノードに対応)。
  // 未指定なら従来通り全セクションを表示する (後方互換)
  sections?: FieldSection[];
  // "boundary" セクションで、指定した辺のみ表示する (辺の子ノード選択時)。未指定/null なら全辺表示
  edgeFilter?: number | null;
  // Mesh/Solve ボタン (study-fem インスペクタページ = "solve" セクションでのみ使用)
  runMesh: () => void;
  runSolve: () => void;
  busy: boolean;
  canRun: boolean;
  showMesh: boolean;
  onToggleShowMesh: () => void;
}

export default function FieldPanel({
  project,
  lengthUnit,
  domainW,
  domainH,
  setDomainSize,
  setCoord,
  edgeState,
  setEdgeType,
  setEdgeVoltage,
  setEdgeVoltageRf,
  setEdgeVoltageWaveform,
  setEdgeSeeGamma,
  setMeshSize,
  setMeshMode,
  setMeshAmr,
  setBField,
  meshResult,
  selectedRegionId,
  onSelectRegion,
  selected,
  renameRegion,
  setRegionType,
  editRegionShape,
  editRegionPolygon,
  mergeRegions,
  updateRegion,
  deleteRegion,
  setRegionLocalSize,
  updateEdgeMeshSize,
  deleteEdgeMeshSize,
  result,
  sections,
  edgeFilter,
  runMesh,
  runSolve,
  busy,
  canRun,
  showMesh,
  onToggleShowMesh,
}: Props) {
  // セクション表示判定 (sections 未指定なら常に表示 = 後方互換)
  const showSection = (name: FieldSection) => !sections || sections.includes(name);
  // 領域マージ (prompts/63) の選択中マージ先IDとエラーメッセージ (ローカル state)
  const [mergeOtherId, setMergeOtherId] = useState<string>("");
  const [mergeError, setMergeError] = useState<string | null>(null);
  // 選択領域が切り替わったら、別領域向けのエラー表示を持ち越さないようクリアする
  useEffect(() => {
    setMergeError(null);
  }, [selected?.id]);
  const coord = project.coord ?? "xy";
  const isRz = coord === "rz";
  const isRzX0 = coord === "rz_x0";
  const isAxisym = isAxisymmetric(coord);
  const edgeLabels = isRz ? EDGE_LABELS_RZ : isRzX0 ? EDGE_LABELS_RZ_X0 : EDGE_LABELS_XY;
  // 座標系ごとの対称軸エッジ番号 (rz: 下辺=0、rz_x0: 左辺=3。xy は該当なし)
  const axisEdge = isRz ? 0 : isRzX0 ? 3 : null;
  // domain 幅/高さのラベル。rz は x=z(軸方向)・y=r(径方向)、rz_x0 は x=r(径方向)・y=z(軸方向)
  const unitLabel = LENGTH_UNIT_LABEL[lengthUnit];
  const widthLabel = isRz ? `長さ z [${unitLabel}]` : isRzX0 ? `半径 r [${unitLabel}]` : `幅 [${unitLabel}]`;
  const heightLabel = isRz ? `半径 r [${unitLabel}]` : isRzX0 ? `長さ z [${unitLabel}]` : `高さ [${unitLabel}]`;
  const bField = project.b_field ?? { bx: 0, by: 0, bz: 0 };

  return (
    <>
      {showSection("domain") && (
        <>
          <h2>ジオメトリ (domain)</h2>
          <div className="field">
            <span className="label">座標系</span>
            <select value={coord} onChange={(e) => setCoord(e.target.value as "xy" | "rz" | "rz_x0")}>
              <option value="xy">平面 2D</option>
              <option value="rz">軸対称 r-z (下辺が軸)</option>
              <option value="rz_x0">軸対称 r-z (左辺が軸)</option>
            </select>
          </div>
          {isAxisym && (
            <div className="hint">
              軸対称モードでは{isRz ? "下辺" : "左辺"}が対称軸になります。PICは未対応です。
            </div>
          )}
          <div className="field">
            <span className="label">{widthLabel}</span>
            <CommitNumberInput
              value={mToUnit(domainW, lengthUnit)}
              step="0.1"
              onCommit={(w) => setDomainSize(unitToM(w, lengthUnit), domainH)}
            />
          </div>
          <div className="field">
            <span className="label">{heightLabel}</span>
            <CommitNumberInput
              value={mToUnit(domainH, lengthUnit)}
              step="0.1"
              onCommit={(h) => setDomainSize(domainW, unitToM(h, lengthUnit))}
            />
          </div>
        </>
      )}

      {showSection("boundary") && (
        <>
          <h2>境界条件</h2>
          {edgeLabels.map((label, i) => {
            // edgeFilter 指定時 (辺の子ノード選択時) はその辺のみ表示する
            if (edgeFilter != null && edgeFilter !== i) return null;
            const st = edgeState(i);
            // 対称軸そのものとなる辺 (自然境界) は切替不可・固定表示とする
            const isAxisEdge = axisEdge === i;
            const rfList = rfComponents(st.voltageRf);
            return (
              <div className="edge-row" key={i}>
                <span className="edge-label">{label}</span>
                <div className="edge-controls">
                  <select
                    value={st.type}
                    disabled={isAxisEdge}
                    onChange={(e) => setEdgeType(i, e.target.value as EdgeBcType)}
                  >
                    <option value="neumann">なし (Neumann)</option>
                    <option value="dirichlet">Dirichlet</option>
                    <option value="symmetry">対称 (粒子反射)</option>
                    <option value="periodic">周期</option>
                  </select>
                  {!isAxisEdge && st.type === "dirichlet" && (
                    <>
                      <CommitNumberInput value={st.voltage} onCommit={(v) => setEdgeVoltage(i, v)} />
                      <div className="rf-check-inline">
                        <Toggle
                          label="RF"
                          checked={rfList.length > 0}
                          onChange={(v) => setEdgeVoltageRf(i, v ? [DEFAULT_VOLTAGE_RF] : undefined)}
                        />
                      </div>
                      <label className="rf-check-inline" title="二次電子放出係数 γ">
                        γ
                        <CommitNumberInput
                          className="rf-compact"
                          value={st.seeGamma}
                          onCommit={(v) => setEdgeSeeGamma(i, v)}
                        />
                      </label>
                    </>
                  )}
                </div>
                {!isAxisEdge && st.type === "dirichlet" && rfList.length > 0 && (
                  <RfComponentsEditor components={rfList} onChange={(next) => setEdgeVoltageRf(i, next)} />
                )}
                {!isAxisEdge && st.type === "dirichlet" && (
                  <WaveformImportEditor
                    waveform={st.voltageWaveform}
                    onChange={(next) => setEdgeVoltageWaveform(i, next)}
                  />
                )}
                {/* 電位プレビュー (prompts/80): RF・波形のどちらも無い (DCのみ) 場合は
                    時間軸が定まらないため何も表示しない */}
                {!isAxisEdge &&
                  st.type === "dirichlet" &&
                  voltagePreviewFreqs(rfList, st.voltageWaveform ?? null).length > 0 && (
                    <>
                      <div className="voltage-preview-heading">電位プレビュー</div>
                      <VoltagePreviewChart
                        voltage={st.voltage}
                        rf={rfList}
                        waveform={st.voltageWaveform ?? null}
                      />
                    </>
                  )}
              </div>
            );
          })}
        </>
      )}

      {showSection("mesh") && (
        <>
          <h2>メッシュ</h2>
          <div className="field">
            <span className="label">サイズ [{unitLabel}]</span>
            <CommitNumberInput
              value={mToUnit(project.mesh.size, lengthUnit)}
              step="0.01"
              onCommit={(v) => setMeshSize(unitToM(v, lengthUnit))}
            />
          </div>
          <div className="field">
            <span className="label">モード</span>
            <select
              value={project.mesh.mode ?? "unstructured"}
              onChange={(e) => setMeshMode(e.target.value as "unstructured" | "structured" | "cartesian")}
            >
              <option value="unstructured">非構造 (三角形)</option>
              <option value="structured">構造格子</option>
              <option value="cartesian">直交格子+埋め込み境界 (v2・GPU)</option>
            </select>
          </div>
          {(project.mesh.mode ?? "unstructured") === "structured" && (
            <div className="hint">
              構造格子は矩形domainのみ対応。等間隔格子を三角形2分割で切り、
              円・斜め境界は要素中心判定による階段近似になります(局所サイズは無効)。
            </div>
          )}
          {project.mesh.mode === "cartesian" && (
            <>
              <div className="hint">
                v2 エンジン (矩形domainのみ)。導体・誘電体の境界は格子と厳密に交差させる埋め込み境界
                (2次精度) で、静電場はマルチグリッド (CPU/GPU)、PIC・DSMC は GPU (CUDA)、流体2D は
                CPU/GPU で高速に解きます。格子はマルチグリッド向けに指定サイズより最大 ~12% 細かく
                なります。表示は各セルを三角形2分割したものです。粒子注入・FN放出・粒子マージ・DSMC連成は
                未対応 (軌道追跡は同じ解像度の構造格子で実行されます)。
              </div>
              <AmrEditor
                amr={project.mesh.amr}
                lengthUnit={lengthUnit}
                bbox={domainBBox(project)}
                onChange={setMeshAmr}
              />
            </>
          )}

          <div className="subheading">辺ローカルサイズ</div>
          <div className="hint">
            キャンバスの「メッシュ細分」ツールで2点クリックでも追加できます。
            線分近傍が指定サイズに細分化されます (非構造メッシュのみ)。
          </div>
          <div className="collector-list">
            {(project.mesh.local_edge_sizes ?? []).length === 0 && (
              <div className="muted">(辺ローカルサイズなし。キャンバスで配置してください)</div>
            )}
            {(project.mesh.local_edge_sizes ?? []).map((e, i) => (
              <div key={i} className="collector-row" style={{ cursor: "default" }}>
                <span className="collector-label-input">{`M${i + 1}`}</span>
                <span
                  className="collector-points"
                  title={`(${mToUnit(e.p1[0], lengthUnit).toFixed(2)}, ${mToUnit(e.p1[1], lengthUnit).toFixed(2)}) - (${mToUnit(e.p2[0], lengthUnit).toFixed(2)}, ${mToUnit(e.p2[1], lengthUnit).toFixed(2)}) ${unitLabel}`}
                >
                  ({mToUnit(e.p1[0], lengthUnit).toFixed(1)},{mToUnit(e.p1[1], lengthUnit).toFixed(1)})–
                  ({mToUnit(e.p2[0], lengthUnit).toFixed(1)},{mToUnit(e.p2[1], lengthUnit).toFixed(1)})
                </span>
                <CommitNumberInput
                  className="collector-tol-input"
                  value={mToUnit(e.size, lengthUnit)}
                  onCommit={(v) => {
                    if (v > 0) updateEdgeMeshSize(i, unitToM(v, lengthUnit));
                  }}
                />
                <button
                  type="button"
                  className="danger collector-delete"
                  onClick={() => deleteEdgeMeshSize(i)}
                  title="この辺ローカルサイズを削除"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
          {(project.mesh.mode ?? "unstructured") !== "unstructured" &&
            (project.mesh.local_edge_sizes ?? []).length > 0 && (
              <div className="hint">構造格子・直交格子モードでは辺ローカルサイズは無視されます。</div>
            )}
        </>
      )}

      {showSection("bfield") && (
        <>
          <h2>一様磁場 [T]</h2>
          <div className="hint">
            粒子追跡・PIC のローレンツ力に適用 (静電場ソルブには影響しない)。軸対称モードでは使用不可
          </div>
          {isAxisym && (
            <div className="hint">軸対称モードでは一様磁場は設定できません (∇·B=0 と矛盾するため)。</div>
          )}
          <div className="field">
            <span className="label">Bx</span>
            <CommitNumberInput
              value={bField.bx}
              disabled={isAxisym}
              onCommit={(v) => setBField({ bx: v })}
            />
          </div>
          <div className="field">
            <span className="label">By</span>
            <CommitNumberInput
              value={bField.by}
              disabled={isAxisym}
              onCommit={(v) => setBField({ by: v })}
            />
          </div>
          <div className="field">
            <span className="label">Bz</span>
            <CommitNumberInput
              value={bField.bz}
              disabled={isAxisym}
              onCommit={(v) => setBField({ bz: v })}
            />
          </div>
        </>
      )}

      {showSection("regions") && (
        <>
          <h2>領域一覧 ({project.geometry.regions.length})</h2>
          <div className="region-list">
            {project.geometry.regions.map((r) => (
              <div
                key={r.id}
                className={`region-item ${selectedRegionId === r.id ? "selected" : ""}`}
                onClick={() => onSelectRegion(r.id)}
              >
                <span>{r.id}</span>
                <span className="tag">{r.type}</span>
              </div>
            ))}
            {project.geometry.regions.length === 0 && (
              <div className="muted">(領域なし。ツールバーで作図してください)</div>
            )}
          </div>

          {selected && (
            <div className="region-edit">
              <label>
                ID
                <CommitTextInput
                  value={selected.id}
                  onCommit={(newId) => renameRegion(selected.id, newId)}
                />
              </label>
              <label>
                種別
                <select
                  value={selected.type}
                  onChange={(e) => setRegionType(selected.id, e.target.value as RegionType)}
                >
                  <option value="conductor">電極 (conductor)</option>
                  <option value="dielectric">誘電体 (dielectric)</option>
                  <option value="charge">空間電荷 (charge)</option>
                </select>
              </label>
              {selected.shape && (
                <>
                  <label>
                    中心 X [{unitLabel}]
                    <CommitNumberInput
                      value={mToUnit(selected.shape.center[0], lengthUnit)}
                      step="0.1"
                      onCommit={(x) =>
                        editRegionShape(selected.id, {
                          ...selected.shape!,
                          center: [unitToM(x, lengthUnit), selected.shape!.center[1]],
                        })
                      }
                    />
                  </label>
                  <label>
                    中心 Y [{unitLabel}]
                    <CommitNumberInput
                      value={mToUnit(selected.shape.center[1], lengthUnit)}
                      step="0.1"
                      onCommit={(y) =>
                        editRegionShape(selected.id, {
                          ...selected.shape!,
                          center: [selected.shape!.center[0], unitToM(y, lengthUnit)],
                        })
                      }
                    />
                  </label>
                  <label>
                    半径 [{unitLabel}]
                    <CommitNumberInput
                      value={mToUnit(selected.shape.radius, lengthUnit)}
                      step="0.1"
                      onCommit={(radius) =>
                        editRegionShape(selected.id, { ...selected.shape!, radius: unitToM(radius, lengthUnit) })
                      }
                    />
                  </label>
                </>
              )}
              {selected.type === "conductor" && (
                <>
                  <label>
                    電位 V [V]
                    <CommitNumberInput
                      value={selected.voltage ?? 0}
                      onCommit={(v) => updateRegion(selected.id, { voltage: v })}
                    />
                  </label>
                  <Toggle
                    label="RF重畳"
                    checked={rfComponents(selected.voltage_rf).length > 0}
                    onChange={(v) =>
                      updateRegion(selected.id, {
                        voltage_rf: v ? [DEFAULT_VOLTAGE_RF] : undefined,
                      })
                    }
                  />
                  {rfComponents(selected.voltage_rf).length > 0 && (
                    <RfComponentsEditor
                      components={rfComponents(selected.voltage_rf)}
                      onChange={(next) => updateRegion(selected.id, { voltage_rf: next })}
                    />
                  )}
                  {/* CSV波形は conductor 領域電極にも適用できる (backend は prompts/73 で
                      領域側も対応済み。辺と同じエディタを共用する) */}
                  <WaveformImportEditor
                    waveform={selected.voltage_waveform ?? undefined}
                    onChange={(next) => updateRegion(selected.id, { voltage_waveform: next })}
                  />
                  {/* 電位プレビュー (prompts/80): RF・波形のどちらも無い (DCのみ) 場合は
                      時間軸が定まらないため何も表示しない */}
                  {voltagePreviewFreqs(rfComponents(selected.voltage_rf), selected.voltage_waveform ?? null).length >
                    0 && (
                    <>
                      <div className="voltage-preview-heading">電位プレビュー</div>
                      <VoltagePreviewChart
                        voltage={selected.voltage ?? 0}
                        rf={rfComponents(selected.voltage_rf)}
                        waveform={selected.voltage_waveform ?? null}
                      />
                    </>
                  )}
                  <label>
                    二次電子放出係数 γ
                    <CommitNumberInput
                      value={selected.see_gamma ?? 0}
                      onCommit={(v) => updateRegion(selected.id, { see_gamma: v })}
                    />
                  </label>
                </>
              )}
              {selected.type === "dielectric" && (
                <>
                  <label>
                    比誘電率 εr
                    <CommitNumberInput
                      value={selected.eps_r ?? 1}
                      onCommit={(v) => updateRegion(selected.id, { eps_r: v })}
                    />
                  </label>
                  <label title="イオン衝突時に確率γで二次電子を放出 (PICのみ。0=無効)">
                    二次電子放出係数 γ
                    <CommitNumberInput
                      value={selected.see_gamma ?? 0}
                      onCommit={(v) => updateRegion(selected.id, { see_gamma: v })}
                    />
                  </label>
                </>
              )}
              {selected.type === "charge" && (
                <label>
                  電荷密度 ρ [C/m³]
                  <CommitNumberInput
                    value={selected.rho ?? 0}
                    onCommit={(v) => updateRegion(selected.id, { rho: v })}
                  />
                </label>
              )}
              {selected.polygon && (
                <>
                  <div className="subheading">頂点座標 [{unitLabel}]</div>
                  <table className="vertex-table">
                    <tbody>
                      {selected.polygon.map((pt, i) => (
                        <tr key={i}>
                          <td className="vertex-index">#{i}</td>
                          <td>
                            <CommitNumberInput
                              className="vertex-input"
                              value={mToUnit(pt[0], lengthUnit)}
                              onCommit={(x) =>
                                editRegionPolygon(
                                  selected.id,
                                  selected.polygon!.map((p, j) => (j === i ? ([unitToM(x, lengthUnit), p[1]] as Point) : p)),
                                )
                              }
                            />
                          </td>
                          <td>
                            <CommitNumberInput
                              className="vertex-input"
                              value={mToUnit(pt[1], lengthUnit)}
                              onCommit={(y) =>
                                editRegionPolygon(
                                  selected.id,
                                  selected.polygon!.map((p, j) => (j === i ? ([p[0], unitToM(y, lengthUnit)] as Point) : p)),
                                )
                              }
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>

                  {(() => {
                    // マージ先候補: polygon を持つ他領域 (自分以外)。無ければマージUI自体を隠す
                    const mergeCandidates = project.geometry.regions.filter(
                      (r) => r.polygon && r.id !== selected.id,
                    );
                    if (mergeCandidates.length === 0) return null;
                    const otherId = mergeCandidates.some((r) => r.id === mergeOtherId)
                      ? mergeOtherId
                      : mergeCandidates[0].id;
                    return (
                      <>
                        <div className="subheading">マージ</div>
                        <div className="merge-row">
                          <select
                            value={otherId}
                            onChange={(e) => {
                              setMergeOtherId(e.target.value);
                              setMergeError(null);
                            }}
                          >
                            {mergeCandidates.map((r) => (
                              <option key={r.id} value={r.id}>
                                {r.id}
                              </option>
                            ))}
                          </select>
                          <button
                            type="button"
                            onClick={() => setMergeError(mergeRegions(selected.id, otherId))}
                          >
                            この領域とマージ
                          </button>
                        </div>
                        <div className="hint">※ プロパティ (種別・電圧等) はこの領域側の設定が使われます</div>
                        {mergeError && <div className="error">{mergeError}</div>}
                      </>
                    );
                  })()}
                </>
              )}
              <label title="この領域とその輪郭のメッシュ特性長。0で解除 (全体サイズを使用)。非構造メッシュのみ有効">
                ローカルメッシュサイズ [{unitLabel}] (0=全体)
                <CommitNumberInput
                  value={mToUnit(
                    project.mesh.local_sizes?.find((ls) => ls.region === selected.id)?.size ?? 0,
                    lengthUnit,
                  )}
                  onCommit={(v) => setRegionLocalSize(selected.id, v > 0 ? unitToM(v, lengthUnit) : null)}
                />
              </label>
              {(project.mesh.mode ?? "unstructured") !== "unstructured" &&
                project.mesh.local_sizes?.some((ls) => ls.region === selected.id) && (
                  <div className="hint">構造格子・直交格子モードではローカルメッシュサイズは無視されます。</div>
                )}
              <button className="danger" onClick={() => deleteRegion(selected.id)}>
                削除
              </button>
            </div>
          )}
        </>
      )}

      {showSection("solve") && (
        <>
          <h2>解析実行</h2>
          <div className="actions">
            <button className="secondary" onClick={runMesh} disabled={busy || !canRun}>
              {busy ? "計算中..." : "Mesh"}
            </button>
            <button onClick={runSolve} disabled={busy || !canRun}>
              {busy ? "計算中..." : "Solve"}
            </button>
            <button className="secondary" onClick={onToggleShowMesh}>
              メッシュ {showMesh ? "非表示" : "表示"}
            </button>
          </div>
          {!canRun && <div className="hint">backend に接続されていません (上部バーのポート設定を確認してください)。</div>}
          {meshResult && (
            <>
              <div className="kv"><span>節点数 (Mesh)</span><span>{meshResult.nodes.length}</span></div>
              <div className="kv"><span>要素数 (Mesh)</span><span>{meshResult.triangles.length}</span></div>
            </>
          )}
          {result ? (
            <>
              <h2>解析結果</h2>
              <div className="kv"><span>節点数</span><span>{result.mesh.nodes.length}</span></div>
              <div className="kv"><span>要素数</span><span>{result.mesh.triangles.length}</span></div>
              <div className="kv"><span>V min/max</span><span>{result.v_min.toFixed(1)} / {result.v_max.toFixed(1)} V</span></div>
              <div className="kv"><span>|E| max</span><span>{result.e_abs_max.toExponential(2)} V/m</span></div>
              <div className="kv"><span>エネルギー</span><span>{result.energy.toExponential(3)} {isAxisym ? "J" : "J/m"}</span></div>
              <div className="kv">
                <span>静電容量</span>
                <span>
                  {result.capacitance != null
                    ? `${result.capacitance.toExponential(3)} ${isAxisym ? "F" : "F/m"}`
                    : "- (2電極系のみ)"}
                </span>
              </div>
              {result.charges && result.charges.length > 0 && (
                <>
                  <div className="muted">電極電荷</div>
                  {result.charges.map((c) => {
                    const m = /^edge(\d+)$/.exec(c.label);
                    const label = m ? edgeLabels[Number(m[1])] ?? c.label : c.label;
                    return (
                      <div className="kv" key={c.label}>
                        <span>{label} ({c.voltage}V)</span>
                        <span>{c.q.toExponential(3)} {isAxisym ? "C" : "C/m"}</span>
                      </div>
                    );
                  })}
                </>
              )}
            </>
          ) : (
            !meshResult && <div className="muted">(まだ Mesh/Solve を実行していません)</div>
          )}
        </>
      )}
    </>
  );
}
