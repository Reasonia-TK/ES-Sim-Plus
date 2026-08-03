import { useEffect, useRef, useState } from "react";
import { Toggle } from "../Toggle";
import { formatNumber } from "../CommitInput";
import { arrayMax, arrayMin } from "../mathUtils";
import { mToUnit } from "../units";
import type { LengthUnit } from "../units";
import { Pic1dLineChart, Pic1dRfMonitor, WallIedfSection, deriveXGrid, interpLinear } from "./Plot1dView";
import type { ChartMarker, LineSeries } from "./Plot1dView";
import type {
  Fluid1dCycle,
  Fluid1dFrameMsg,
  Fluid1dResult,
  Fluid1dSettings,
  Fluid1dStartedMsg,
  Pic1dResult,
} from "../types";

/**
 * 1D プラズマ流体 (ドリフト拡散 + 電子エネルギー) スタディのキャンバス代替ビュー (Phase D2、
 * prompts/104-109)。canvas/Plot1dView.tsx (pic1d 用) と同じ役割分担・見た目
 * (Pic1dLineChart/Pic1dRfMonitor/deriveXGrid/interpLinear を流用、export 追加のみで
 * pic1d 側の挙動は不変) だが、本コンポーネントの売りは「1D PIC との比較オーバーレイ」
 * (結果ビューの「1D PIC と比較」Toggle) にある。流体とPICは格子 (gap_m/n_cells) が
 * 一致するとは限らないため、Pic1dLineChart の「1つの x 配列を全系列で共有する」設計は
 * そのまま使えず、系列ごとに独立した x 配列を持てる Fluid1dCompareChart を新設する。
 */

interface Props {
  lengthUnit: LengthUnit;
  fluid1d: Fluid1dSettings;
  running: boolean;
  started: Fluid1dStartedMsg | null;
  frame: Fluid1dFrameMsg | null;
  result: Fluid1dResult | null;
  error: string | null;
  // PIC 比較オーバーレイ用 (App から渡す直近の 1D PIC 実行結果)。無ければ比較Toggleをdisabledにする
  pic1dResult: Pic1dResult | null;
}

// シースエッジ (Brinkmann 基準) の縦破線マーカー色。Plot1dView.tsx の
// SHEATH_MARKER_COLOR と同じ値 (見た目を揃えるための複製、値のみで依存追加はしない)
const SHEATH_MARKER_COLOR = "#ffb454";

// ---- 結果フィールド選択 ---------------------------------------------------------

type Fluid1dField = "phi" | "e" | "n_e" | "n_i" | "overlay" | "t_e" | "ionization";

const FLUID1D_FIELD_OPTIONS: { value: Fluid1dField; label: string }[] = [
  { value: "phi", label: "電位 φ [V]" },
  { value: "e", label: "電場 E [V/m]" },
  { value: "n_e", label: "電子密度 n_e [m^-3]" },
  { value: "n_i", label: "イオン密度 n_i [m^-3]" },
  { value: "overlay", label: "n_e + n_i 重ね" },
  { value: "t_e", label: "電子温度 T_e [eV]" },
  { value: "ionization", label: "電離レート [m^-3 s^-1]" },
];

function isFluid1dDensityField(f: Fluid1dField): boolean {
  return f === "n_e" || f === "n_i" || f === "overlay";
}

// フィールドごとの色。流体=実線に使う基準色、pic=PIC比較の破線に使う同色系の濃色
// (「流体=実線・PIC=破線・同色系明暗差」の規約、prompts/109)
const FIELD_COLOR: Record<Exclude<Fluid1dField, "overlay">, { fluid: string; pic: string }> = {
  phi: { fluid: "#59c2ff", pic: "#2f7dbf" },
  e: { fluid: "#59c2ff", pic: "#2f7dbf" },
  n_e: { fluid: "#4da3ff", pic: "#2a5fa8" },
  n_i: { fluid: "#ffb84d", pic: "#c98a2e" },
  t_e: { fluid: "#6fd08c", pic: "#3f9463" },
  ionization: { fluid: "#c792ea", pic: "#8a5aa8" },
};

function fluid1dFieldSeries(field: Fluid1dField, profiles: NonNullable<Fluid1dResult["profiles"]>): LineSeries[] {
  switch (field) {
    case "phi":
      return [{ label: "φ (流体)", values: profiles.phi, color: FIELD_COLOR.phi.fluid }];
    case "e":
      return [{ label: "E (流体)", values: profiles.e, color: FIELD_COLOR.e.fluid }];
    case "n_e":
      return [{ label: "n_e (流体)", values: profiles.n_e, color: FIELD_COLOR.n_e.fluid }];
    case "n_i":
      return [{ label: "n_i (流体)", values: profiles.n_i, color: FIELD_COLOR.n_i.fluid }];
    case "overlay":
      return [
        { label: "n_e (流体)", values: profiles.n_e, color: FIELD_COLOR.n_e.fluid },
        { label: "n_i (流体)", values: profiles.n_i, color: FIELD_COLOR.n_i.fluid },
      ];
    case "t_e":
      return [{ label: "T_e (流体)", values: profiles.t_e, color: FIELD_COLOR.t_e.fluid }];
    case "ionization":
      return [{ label: "電離レート (流体)", values: profiles.ionization, color: FIELD_COLOR.ionization.fluid }];
  }
}

// PIC 側 (pic1dResult.profiles) の同じフィールドを比較オーバーレイ用の破線系列として組み立てる。
// Pic1dProfiles / Fluid1dProfiles はフィールド名 (phi/e/n_e/n_i/t_e/ionization) が一致するため
// 同じ switch 構造で書ける (電離レート等の物理量の定義もそれぞれのソルバーで揃えてある)
function pic1dCompareSeries(field: Fluid1dField, profiles: NonNullable<Pic1dResult["profiles"]>): LineSeries[] {
  switch (field) {
    case "phi":
      return [{ label: "φ (PIC)", values: profiles.phi, color: FIELD_COLOR.phi.pic }];
    case "e":
      return [{ label: "E (PIC)", values: profiles.e, color: FIELD_COLOR.e.pic }];
    case "n_e":
      return [{ label: "n_e (PIC)", values: profiles.n_e, color: FIELD_COLOR.n_e.pic }];
    case "n_i":
      return [{ label: "n_i (PIC)", values: profiles.n_i, color: FIELD_COLOR.n_i.pic }];
    case "overlay":
      return [
        { label: "n_e (PIC)", values: profiles.n_e, color: FIELD_COLOR.n_e.pic },
        { label: "n_i (PIC)", values: profiles.n_i, color: FIELD_COLOR.n_i.pic },
      ];
    case "t_e":
      return [{ label: "T_e (PIC)", values: profiles.t_e, color: FIELD_COLOR.t_e.pic }];
    case "ionization":
      return [{ label: "電離レート (PIC)", values: profiles.ionization, color: FIELD_COLOR.ionization.pic }];
  }
}

// フィールドごとの日本語タイミングラベル (fluid1d.py の self.timing キーと1対1対応、total は除く)
const FLUID1D_TIMING_LABELS: Record<string, string> = {
  poisson: "ポアソン求解+電場計算",
  transport: "ドリフト拡散 (n_e/n_i 輸送)",
  energy: "電子エネルギー方程式",
  other: "時間平均・履歴集計・診断",
};

// ---- 比較オーバーレイ用チャート -------------------------------------------------

// 系列ごとに独立した x 配列を持てる折れ線チャート (Plot1dView.Pic1dLineChart は全系列で
// 単一の x 配列を共有する設計のため、格子が異なりうる流体/PIC の重ね描きには使えない)。
// 見た目 (枠線・フォント・パディング・対数軸) は Pic1dLineChart に揃え、dashed 指定の系列は
// 破線で描く (「流体=実線・PIC=破線」の規約)
interface CompareSeries extends LineSeries {
  x: number[];
  dashed?: boolean;
}

function Fluid1dCompareChart({
  series,
  height = 160,
  logY = false,
  markers = [],
}: {
  series: CompareSeries[];
  height?: number;
  logY?: boolean;
  markers?: ChartMarker[];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = el.getBoundingClientRect();
    el.width = rect.width * dpr;
    el.height = rect.height * dpr;
    const ctx = el.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const padL = 52;
    const padR = 8;
    const padT = 6;
    const padB = 16;
    const plotW = rect.width - padL - padR;
    const plotH = rect.height - padT - padB;

    ctx.strokeStyle = "#363c48";
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, padT, plotW, plotH);

    if (series.length === 0) return;

    // x範囲は全系列の x 配列の和集合 (物理的な位置で揃える。流体/PICで gap_m が
    // 微妙に異なっても、同じ横軸スケール上に重ね描きできるようにするため)
    let xMin = Infinity;
    let xMax = -Infinity;
    for (const s of series) {
      if (s.x.length === 0) continue;
      xMin = Math.min(xMin, arrayMin(s.x));
      xMax = Math.max(xMax, arrayMax(s.x));
    }
    if (!(xMin < xMax)) {
      xMin = 0;
      xMax = 1;
    }
    const xRange = xMax - xMin || 1;

    let yMax = -Infinity;
    let yMinRaw = Infinity;
    let yMinPos = Infinity;
    for (const s of series) {
      const mx = arrayMax(s.values);
      const mn = arrayMin(s.values);
      if (mx > yMax) yMax = mx;
      if (mn < yMinRaw) yMinRaw = mn;
      for (const v of s.values) if (v > 0 && v < yMinPos) yMinPos = v;
    }
    if (!(yMax > -Infinity)) yMax = 1;
    if (!(yMinRaw < Infinity)) yMinRaw = 0;
    const useLog = logY && yMinPos < Infinity && yMinPos < yMax;
    const logLo = useLog ? Math.log10(yMinPos) : 0;
    const logHi = useLog ? Math.log10(yMax) : 0;
    const logRange = logHi - logLo || 1;
    const linRange = yMax - yMinRaw || 1;

    const xOf = (v: number) => padL + ((v - xMin) / xRange) * plotW;
    const yOf = (v: number) => {
      if (useLog) {
        if (!(v > 0)) return padT + plotH;
        const t = (Math.log10(v) - logLo) / logRange;
        return padT + plotH - Math.min(1, Math.max(0, t)) * plotH;
      }
      return padT + plotH - ((v - yMinRaw) / linRange) * plotH;
    };

    for (const s of series) {
      ctx.save();
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.4;
      if (s.dashed) ctx.setLineDash([5, 3]);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < s.x.length && i < s.values.length; i++) {
        const v = s.values[i];
        if (useLog && !(v > 0)) {
          started = false;
          continue;
        }
        const px = xOf(s.x[i]);
        const py = yOf(v);
        if (!started) {
          ctx.moveTo(px, py);
          started = true;
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.stroke();
      ctx.restore();
    }

    for (const m of markers) {
      if (!(m.x >= xMin && m.x <= xMax)) continue;
      const px = xOf(m.x);
      ctx.save();
      ctx.strokeStyle = m.color;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(px, padT);
      ctx.lineTo(px, padT + plotH);
      ctx.stroke();
      ctx.restore();
    }

    ctx.font = "9px system-ui, sans-serif";
    ctx.fillStyle = "#8a919e";
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText(xMin.toPrecision(3), padL, padT + plotH + 3);
    ctx.textAlign = "right";
    ctx.fillText(xMax.toPrecision(3), padL + plotW, padT + plotH + 3);
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(yMax.toPrecision(3), padL - 4, padT);
    ctx.textBaseline = "bottom";
    ctx.fillText((useLog ? yMinPos : yMinRaw).toPrecision(3), padL - 4, padT + plotH);
  }, [series, logY, markers]);

  const legendMarkers = Array.from(new Map(markers.map((m) => [`${m.color}|${m.label}`, m])).values());

  return (
    <>
      <canvas ref={canvasRef} className="pic1d-chart" style={{ height }} />
      <div className="pic-chart-legend">
        {series.map((s, i) => (
          <span key={i}>
            <span className="swatch" style={{ background: s.color }} />
            {s.label}
            {s.dashed ? " (破線)" : ""}
          </span>
        ))}
        {legendMarkers.map((m, i) => (
          <span key={`marker-${i}`}>
            <span className="swatch" style={{ background: m.color }} />
            {m.label}
          </span>
        ))}
      </div>
    </>
  );
}

// ---- ライブビュー -------------------------------------------------------------

function Fluid1dLiveView({
  lengthUnit,
  fluid1d,
  started,
  frame,
}: {
  lengthUnit: LengthUnit;
  fluid1d: Fluid1dSettings;
  started: Fluid1dStartedMsg;
  frame: Fluid1dFrameMsg | null;
}) {
  // プラズマ密度は桁が大きく変わりやすいため既定で対数表示にする (Plot1dView と同じ)
  const [densityLog, setDensityLog] = useState(true);
  const xDisp = started.x.map((v) => mToUnit(v, lengthUnit));

  return (
    <>
      <h3>φ(x) [V]</h3>
      {frame ? (
        <Pic1dLineChart x={xDisp} series={[{ label: "φ", values: frame.phi, color: FIELD_COLOR.phi.fluid }]} />
      ) : (
        <p className="hint">フレーム待機中...</p>
      )}

      <div className="pic1d-row-header">
        <h3>n_e / n_i (x) [m^-3]</h3>
        <Toggle label="対数" checked={densityLog} onChange={setDensityLog} />
      </div>
      {frame && (
        <Pic1dLineChart
          x={xDisp}
          series={[
            { label: "n_e", values: frame.n_e, color: FIELD_COLOR.n_e.fluid },
            { label: "n_i", values: frame.n_i, color: FIELD_COLOR.n_i.fluid },
          ]}
          logY={densityLog}
        />
      )}

      <h3>T_e(x) [eV]</h3>
      {frame ? (
        <Pic1dLineChart x={xDisp} series={[{ label: "T_e", values: frame.t_e, color: FIELD_COLOR.t_e.fluid }]} />
      ) : (
        <p className="hint">フレーム待機中...</p>
      )}

      <div className="kv">
        <span>ステップ / 時刻</span>
        <span>
          {frame?.step ?? started.step_offset} / {frame ? `${(frame.t * 1e9).toFixed(2)} ns` : "-"}
        </span>
      </div>
      <div className="kv">
        <span>経過秒 (このフレームまで)</span>
        <span>{frame ? frame.elapsed_s.toFixed(3) : "-"} s</span>
      </div>

      {frame && <Pic1dRfMonitor electrode={fluid1d.left} t={frame.t} />}
    </>
  );
}

// ---- 位相分解アニメーション ------------------------------------------------------

// Plot1dView.Pic1dCyclePlayer と同じ考え方だが、Fluid1dCycle は t_e (電子温度) も
// 位相分解で持つ (流体は粒子サンプルが無いため T_e を常に持てる) ので選択肢に加えている点、
// sheath がフレーム分解では無く時間平均の1本のみ (result.sheath) なので毎ビン共通の
// 静的マーカーとして重ねる点が異なる (呼び出し側 Fluid1dResultView から渡す)
function Fluid1dCyclePlayer({
  lengthUnit,
  cycle,
  gapM,
  nCells,
  sheathMarkers,
}: {
  lengthUnit: LengthUnit;
  cycle: Fluid1dCycle;
  gapM: number;
  nCells: number;
  sheathMarkers: ChartMarker[];
}) {
  const [field, setField] = useState<"phi" | "n_e" | "n_i" | "t_e">("phi");
  const [playing, setPlaying] = useState(false);
  const [bin, setBin] = useState(0);
  const [fps, setFps] = useState(10);
  const xDisp = deriveXGrid(gapM, nCells).map((v) => mToUnit(v, lengthUnit));

  useEffect(() => {
    if (!playing || cycle.bins <= 0) return;
    const bins = cycle.bins;
    const id = setInterval(() => setBin((b) => (b + 1) % bins), 1000 / fps);
    return () => clearInterval(id);
  }, [playing, fps, cycle.bins]);

  const clampedBin = Math.min(bin, cycle.bins - 1);
  const colorOf = (f: "phi" | "n_e" | "n_i" | "t_e") => FIELD_COLOR[f].fluid;

  return (
    <>
      <h3>位相分解アニメーション</h3>
      <div className="field">
        <span className="label">表示フィールド</span>
        <select value={field} onChange={(e) => setField(e.target.value as "phi" | "n_e" | "n_i" | "t_e")}>
          <option value="phi">電位 φ [V]</option>
          <option value="n_e">電子密度 n_e [m^-3]</option>
          <option value="n_i">イオン密度 n_i [m^-3]</option>
          <option value="t_e">電子温度 T_e [eV]</option>
        </select>
      </div>
      <Pic1dLineChart
        x={xDisp}
        series={[{ label: field, values: cycle[field][clampedBin] ?? [], color: colorOf(field) }]}
        markers={sheathMarkers}
      />
      <div className="actions">
        <button className="secondary" onClick={() => setPlaying(!playing)}>
          {playing ? "一時停止" : "再生"}
        </button>
        <select value={fps} onChange={(e) => setFps(Number(e.target.value))}>
          <option value={5}>5 fps</option>
          <option value={10}>10 fps</option>
          <option value={20}>20 fps</option>
        </select>
      </div>
      <div className="field">
        <span className="label">
          位相 (bin {clampedBin + 1}/{cycle.bins})
        </span>
        <input
          type="range"
          min={0}
          max={cycle.bins - 1}
          step={1}
          value={clampedBin}
          onChange={(e) => {
            setPlaying(false);
            setBin(Number(e.target.value));
          }}
        />
      </div>
      <p className="hint">
        周波数 {formatNumber(cycle.freq_hz)} Hz (周期 {(1e9 / cycle.freq_hz).toFixed(2)} ns)
      </p>
    </>
  );
}

// ---- 結果ビュー ---------------------------------------------------------------

interface CompareSummary {
  neRatio: number | null;
  teDiff: number | null;
  leftDiff: number | null;
  rightDiff: number | null;
}

function Fluid1dResultView({
  lengthUnit,
  result,
  pic1dResult,
}: {
  lengthUnit: LengthUnit;
  result: Fluid1dResult;
  pic1dResult: Pic1dResult | null;
}) {
  const [field, setField] = useState<Fluid1dField>("phi");
  const [logScale, setLogScale] = useState(false);
  const [showSheath, setShowSheath] = useState(true);
  // PIC 比較オーバーレイ (本機能の売り、prompts/109)。既定オフ (pic1dResult が無いときは無意味なため)
  const [compare, setCompare] = useState(false);

  const profiles = result.profiles;
  const xDisp = deriveXGrid(result.settings.gap_m, result.settings.n_cells).map((v) => mToUnit(v, lengthUnit));

  const timingEntries = Object.entries(result.timing).filter(([k]) => k !== "total");
  const timingTotal = result.timing.total ?? timingEntries.reduce((sum, [, v]) => sum + v, 0);
  const timingRows = timingEntries.sort((a, b) => b[1] - a[1]);

  const centerX = result.settings.gap_m / 2;
  const centerNe = profiles ? interpLinear(profiles.x, profiles.n_e, centerX) : null;
  const centerTe = profiles ? interpLinear(profiles.x, profiles.t_e, centerX) : null;

  // 時間平均プロファイルのシースエッジ縦マーカー (流体自身の結果)
  const sheath = result.sheath ?? null;
  const sheathMarkers: ChartMarker[] = [];
  if (showSheath && sheath) {
    if (sheath.left_s != null) {
      sheathMarkers.push({ x: mToUnit(sheath.left_s, lengthUnit), color: SHEATH_MARKER_COLOR, label: "シースエッジ" });
    }
    // right_s は pic1d と同じ規約で「右電極からの距離」として返る (brinkmann_sheath_edge の
    // 鏡映座標、prompts/99)。x 座標に戻すには gap (= xDisp の末尾) から引く
    if (sheath.right_s != null) {
      sheathMarkers.push({
        x: xDisp[xDisp.length - 1] - mToUnit(sheath.right_s, lengthUnit),
        color: SHEATH_MARKER_COLOR,
        label: "シースエッジ",
      });
    }
  }

  // PIC 比較オーバーレイ (最重要機能)。pic1dResult.profiles が無ければ Toggle 自体を disabled にする
  const picProfiles = pic1dResult?.profiles ?? null;
  const canCompare = picProfiles != null;
  const xPicDisp =
    picProfiles && pic1dResult
      ? deriveXGrid(pic1dResult.settings.gap_m, pic1dResult.settings.n_cells).map((v) => mToUnit(v, lengthUnit))
      : [];

  let chartSeries: CompareSeries[] = [];
  if (profiles) {
    chartSeries = fluid1dFieldSeries(field, profiles).map((s) => ({ ...s, x: xDisp }));
    if (compare && picProfiles) {
      chartSeries = [
        ...chartSeries,
        ...pic1dCompareSeries(field, picProfiles).map((s) => ({ ...s, x: xPicDisp, dashed: true })),
      ];
    }
  }

  // 比較サマリ: 中心 n_e 比 (流体/PIC)・中心 T_e 差・シースエッジ位置差 (prompts/109)
  let compareSummary: CompareSummary | null = null;
  if (compare && picProfiles && pic1dResult) {
    const picCenterX = pic1dResult.settings.gap_m / 2;
    const picCenterNe = interpLinear(picProfiles.x, picProfiles.n_e, picCenterX);
    const picCenterTe = interpLinear(picProfiles.x, picProfiles.t_e, picCenterX);
    const neRatio = centerNe != null && picCenterNe > 0 ? centerNe / picCenterNe : null;
    const teDiff = centerTe != null ? centerTe - picCenterTe : null;
    const picSheath = pic1dResult.sheath ?? null;
    const leftDiff =
      sheath?.left_s != null && picSheath?.left_s != null ? sheath.left_s - picSheath.left_s : null;
    const rightDiff =
      sheath?.right_s != null && picSheath?.right_s != null ? sheath.right_s - picSheath.right_s : null;
    compareSummary = { neRatio, teDiff, leftDiff, rightDiff };
  }

  return (
    <>
      <div className="field">
        <span className="label">結果表示</span>
        <select value={field} onChange={(e) => setField(e.target.value as Fluid1dField)}>
          {FLUID1D_FIELD_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
      </div>
      {isFluid1dDensityField(field) && <Toggle label="対数スケール" checked={logScale} onChange={setLogScale} />}
      {sheath && <Toggle label="シースエッジ" checked={showSheath} onChange={setShowSheath} />}
      <Toggle
        label="1D PIC と比較"
        checked={compare}
        onChange={setCompare}
        disabled={!canCompare}
        title={
          canCompare
            ? "1D PIC (PIC-MCC 1D) の時間平均プロファイルを破線で重ね描きします"
            : "1D PIC が未実行、またはプロファイルがありません (スタディ「PIC-MCC 1D」を実行してください)"
        }
      />
      {!canCompare && (
        <p className="hint">1D PIC (PIC-MCC 1D) を実行すると、その結果をこの画面に重ねて比較できます。</p>
      )}

      {profiles ? (
        <Fluid1dCompareChart
          series={chartSeries}
          height={170}
          logY={isFluid1dDensityField(field) && logScale}
          markers={sheathMarkers}
        />
      ) : (
        <p className="hint">時間平均プロファイルがありません (ステップ数0で停止した可能性があります)。</p>
      )}
      {compare && canCompare && (
        <p className="hint">
          流体: 実線 / 1D PIC: 破線 (同色系の濃色)。x格子はそれぞれの設定 (gap_m/n_cells) から
          再構成しているため、両者の gap_m が異なる場合は横軸のスケールがずれます。
        </p>
      )}
      {profiles && <p className="hint">時間平均ステップ数: {profiles.avg_steps}</p>}

      {result.cycle && (
        <Fluid1dCyclePlayer
          lengthUnit={lengthUnit}
          cycle={result.cycle}
          gapM={result.settings.gap_m}
          nCells={result.settings.n_cells}
          sheathMarkers={showSheath ? sheathMarkers : []}
        />
      )}

      <WallIedfSection
        wallIedf={result.wall_iedf}
        downloadPrefix="fluid1d"
        hint="無衝突シース近似 (CX 衝突による低エネルギー成分は含みません)"
      />

      <h3>履歴 (全域密度) [m^-2]</h3>
      <Pic1dLineChart
        x={result.history.step}
        series={[
          { label: "n_e_total", values: result.history.n_e_total, color: FIELD_COLOR.n_e.fluid },
          { label: "n_i_total", values: result.history.n_i_total, color: FIELD_COLOR.n_i.fluid },
        ]}
        height={90}
      />

      <h3>履歴 (壁損失、累計) [m^-2]</h3>
      <Pic1dLineChart
        x={result.history.step}
        series={[
          {
            label: "電子 (左+右)",
            values: result.history.wall_left_e.map((v, i) => v + result.history.wall_right_e[i]),
            color: FIELD_COLOR.n_e.fluid,
          },
          {
            label: "イオン (左+右)",
            values: result.history.wall_left_i.map((v, i) => v + result.history.wall_right_i[i]),
            color: FIELD_COLOR.n_i.fluid,
          },
        ]}
        height={90}
      />

      <h3>数値サマリ</h3>
      <div className="kv">
        <span>経過時間 (壁時計)</span>
        <span>{result.elapsed_s.toFixed(3)} s</span>
      </div>
      {timingRows.map(([key, sec]) => (
        <div className="kv" key={key}>
          <span>{FLUID1D_TIMING_LABELS[key] ?? key}</span>
          <span>
            {sec.toFixed(3)} s ({timingTotal > 0 ? ((100 * sec) / timingTotal).toFixed(1) : "0.0"}%)
          </span>
        </div>
      ))}
      <div className="kv">
        <span>合計</span>
        <span>{timingTotal.toFixed(3)} s</span>
      </div>

      <div className="kv">
        <span>壁吸収 左電極 (電子/イオン)</span>
        <span>{result.walls.left.electron} / {result.walls.left.ion}</span>
      </div>
      <div className="kv">
        <span>壁吸収 右電極 (電子/イオン)</span>
        <span>{result.walls.right.electron} / {result.walls.right.ion}</span>
      </div>
      <div className="kv">
        <span>電離生成 (累計)</span>
        <span>{formatNumber(result.gen_total)} m^-2</span>
      </div>
      {centerNe != null && (
        <div className="kv">
          <span>中央密度 n_e(gap/2)</span>
          <span>{centerNe.toExponential(3)} m^-3</span>
        </div>
      )}
      {centerTe != null && (
        <div className="kv">
          <span>中央電子温度 T_e(gap/2)</span>
          <span>{centerTe.toFixed(3)} eV</span>
        </div>
      )}
      {sheath && (
        <div className="kv">
          <span>シースエッジ (電極からの距離): 左 s / 右 s</span>
          <span>
            {sheath.left_s != null ? `${formatNumber(mToUnit(sheath.left_s, lengthUnit))} ${lengthUnit}` : "—"}
            {" / "}
            {sheath.right_s != null ? `${formatNumber(mToUnit(sheath.right_s, lengthUnit))} ${lengthUnit}` : "—"}
          </span>
        </div>
      )}

      {compareSummary && (
        <>
          <h3>1D PIC 比較サマリ</h3>
          <div className="kv">
            <span>中心 n_e 比 (流体/PIC)</span>
            <span>{compareSummary.neRatio != null ? compareSummary.neRatio.toFixed(3) : "—"}</span>
          </div>
          <div className="kv">
            <span>中心 T_e 差 (流体−PIC)</span>
            <span>{compareSummary.teDiff != null ? `${compareSummary.teDiff.toFixed(3)} eV` : "—"}</span>
          </div>
          <div className="kv">
            <span>シースエッジ位置差 (流体−PIC): 左 / 右</span>
            <span>
              {compareSummary.leftDiff != null
                ? `${formatNumber(mToUnit(compareSummary.leftDiff, lengthUnit))} ${lengthUnit}`
                : "—"}
              {" / "}
              {compareSummary.rightDiff != null
                ? `${formatNumber(mToUnit(compareSummary.rightDiff, lengthUnit))} ${lengthUnit}`
                : "—"}
            </span>
          </div>
        </>
      )}
    </>
  );
}

// ---- ルート -------------------------------------------------------------------

export default function Fluid1dPlotView({
  lengthUnit,
  fluid1d,
  running,
  started,
  frame,
  result,
  error,
  pic1dResult,
}: Props) {
  // ライブ表示の条件: 実行中、または「開始はしたがまだ完了結果が無い」区間 (Plot1dView と同じ)
  const showLive = running || (started != null && result == null);
  const statusLabel = showLive ? "ライブ実行中" : result ? "計算結果" : "未実行";

  return (
    <div className="pic1d-view">
      <div className="tool-toolbar pic1d-toolbar">
        <span className="pic1d-toolbar-title">プラズマ流体 1D</span>
        <span className="muted">{statusLabel}</span>
      </div>
      <div className="pic1d-body">
        {error && <div className="error pic1d-error">{error}</div>}
        {showLive && started && (
          <Fluid1dLiveView lengthUnit={lengthUnit} fluid1d={fluid1d} started={started} frame={frame} />
        )}
        {!showLive && result && (
          <Fluid1dResultView lengthUnit={lengthUnit} result={result} pic1dResult={pic1dResult} />
        )}
        {!showLive && !result && !error && (
          <p className="hint">流体 (1D) が未実行です。左パネルの「流体 1D 開始」から実行してください。</p>
        )}
      </div>
    </div>
  );
}
