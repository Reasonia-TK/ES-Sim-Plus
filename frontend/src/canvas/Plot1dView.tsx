import { useEffect, useRef, useState } from "react";
import { EedfChart, EEDF_CHART_COLORS } from "../panels/PicPanel";
import { Toggle } from "../Toggle";
import { formatNumber } from "../CommitInput";
import { saveTextFile } from "../saveFile";
import { arrayMax, arrayMin } from "../mathUtils";
import { mToUnit } from "../units";
import type { LengthUnit } from "../units";
import { rfComponents } from "../types";
import type {
  Pic1dCycle,
  Pic1dElectrode,
  Pic1dFrameMsg,
  Pic1dResult,
  Pic1dSettings,
  Pic1dStartedMsg,
  VoltageWaveform,
} from "../types";

/**
 * 1D PIC/MCC (PIC-MCC 1D) スタディのキャンバス代替ビュー (prompts/91)。
 * CadCanvas (2D ジオメトリ描画) の代わりに、App.tsx が study-pic1d/result-pic1d 選択時に
 * この SVG ではなく canvas 直描きのコンポーネントを表示する。1D には CAD 編集対象の
 * ジオメトリが無いため、ツールバーの代わりに簡素なヘッダのみを持つ。
 *
 * - ライブ (実行中): φ(x)・n_e/n_i(x)・電子位相空間 (x,vx) 散布・左電極RF波形モニタ
 * - 結果 (done 後): フィールド選択によるラインプロット・位相分解アニメ・EEDFチャート・
 *   数値サマリ (経過時間・実行時間内訳・壁吸収・中央密度)・履歴 (マクロ粒子数) チャート
 *
 * 既存 PicPanel の EedfChart はデータ形状 (label/e_centers/f/...) が2D/1Dで同一のため
 * そのまま流用する (export 済み)。RF波形は 2D の RfPhaseMonitor と異なりデータモデルが
 * 異なる (Pic1dElectrode は v_dc + 複数 RF 成分 + 複数波形の和、2D は BC単位の RF成分+
 * 単一CSV波形) ため、見た目 (配色・レイアウト) だけ揃えた専用実装にする
 * (evalPic1dVoltage 参照。RF 成分の評価式は VoltagePreviewChart の evalVoltage / pic.py の
 * _eval_rf と完全に一致させる、prompts/93)。
 */

interface Props {
  lengthUnit: LengthUnit;
  pic1d: Pic1dSettings;
  running: boolean;
  started: Pic1dStartedMsg | null;
  frame: Pic1dFrameMsg | null;
  result: Pic1dResult | null;
  error: string | null;
}

// ---- 汎用ヘルパー ------------------------------------------------------------

// 一様格子の節点座標 [m] を再構成する (pic1d.py の xg = linspace(0, gap, n_cells+1) と同じ)。
// done メッセージの profiles.x をそのまま使ってもよいが、cycle 表示用にも共通して使えるよう
// settings から常に再構成する (profiles が null な退化ケース — 0ステップで停止等 — でも動く)
function deriveXGrid(gapM: number, nCells: number): number[] {
  const n = Math.max(1, Math.round(nCells)) + 1;
  const xs = new Array<number>(n);
  for (let i = 0; i < n; i++) xs[i] = (gapM * i) / (n - 1);
  return xs;
}

// 単調増加な (xp, fp) の線形補間 (範囲外はクランプ)。中央密度の算出・RF波形評価で使う
function interpLinear(xp: number[], fp: number[], x: number): number {
  const n = xp.length;
  if (n === 0) return NaN;
  if (n === 1) return fp[0];
  if (x <= xp[0]) return fp[0];
  if (x >= xp[n - 1]) return fp[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xp[mid] <= x) lo = mid;
    else hi = mid;
  }
  const frac = (x - xp[lo]) / (xp[hi] - xp[lo]);
  return fp[lo] + frac * (fp[hi] - fp[lo]);
}

// V(t) = v_dc + Σ RF sin + Σ waveforms(t) (prompts/93)。RF 成分の評価式は
// VoltagePreviewChart の evalVoltage / pic.py の _eval_rf と完全に同じ
// (Σ amplitude·sin(2π·freq_hz·t + phase_deg·π/180))。CSV 波形の評価式は
// pic.py の _eval_waveform / VoltagePreviewChart の evalWaveform と同じ
// (位相を frac(t·freq_hz) で求め、1周期ループの折返しを補って線形補間する)。
// voltage_rf が未指定 (rfComponents が空配列) なら寄与0で従来 (waveforms のみ) と数値不変
function evalPic1dVoltage(electrode: Pic1dElectrode, t: number): number {
  let v = electrode.v_dc ?? 0;
  for (const c of rfComponents(electrode.voltage_rf)) {
    v += c.amplitude * Math.sin(2 * Math.PI * c.freq_hz * t + (c.phase_deg * Math.PI) / 180);
  }
  for (const wf of electrode.waveforms ?? []) {
    v += evalPic1dWaveform(wf, t);
  }
  return v;
}

function evalPic1dWaveform(wf: VoltageWaveform, t: number): number {
  const raw = t * wf.freq_hz;
  const frac = raw - Math.floor(raw); // JS の % は負数で符号を保つため frac(x) は floor で計算する
  const phaseExt = [...wf.phase, wf.phase[0] + 1];
  const vExt = [...wf.v, wf.v[0]];
  return interpLinear(phaseExt, vExt, frac);
}

// PIC 1D タイムライン内訳の日本語ラベル (pic1d.py の self.timing キーと1対1対応、total は除く)
const PIC1D_TIMING_LABELS: Record<string, string> = {
  deposit: "電荷デポジット (CIC)",
  field: "ポアソン求解+電場計算",
  push: "粒子押し出し・境界・SEE",
  mcc: "MCC衝突",
  other: "時間平均・EEDF集計・診断",
};

// ---- canvas 直描きの汎用ラインチャート ----------------------------------------

interface LineSeries {
  label: string;
  values: number[];
  color: string;
}

// 複数系列を重ね描きする折れ線チャート (PicPanel の PicHistoryChart/EedfChart と同じ
// canvas 直描きスタイル: 枠 #363c48、9px 目盛りフォント、padL≈50)。
// x/系列値はどちらも「表示用に変換済み」の生の number[] を渡す想定 (単位変換は呼び出し側で行う)
function Pic1dLineChart({
  x,
  series,
  height = 110,
  logY = false,
}: {
  x: number[];
  series: LineSeries[];
  height?: number;
  logY?: boolean;
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

    if (x.length < 2 || series.length === 0) return;

    const xMin = arrayMin(x);
    const xMax = arrayMax(x);
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
        if (!(v > 0)) return padT + plotH; // 0以下はプロット外 (下端に落とす)
        const t = (Math.log10(v) - logLo) / logRange;
        return padT + plotH - Math.min(1, Math.max(0, t)) * plotH;
      }
      return padT + plotH - ((v - yMinRaw) / linRange) * plotH;
    };

    for (const s of series) {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < x.length && i < s.values.length; i++) {
        const v = s.values[i];
        if (useLog && !(v > 0)) {
          started = false; // 対数軸では非正値の点で線を切る
          continue;
        }
        const px = xOf(x[i]);
        const py = yOf(v);
        if (!started) {
          ctx.moveTo(px, py);
          started = true;
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.stroke();
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
  }, [x, series, logY]);

  return (
    <>
      <canvas ref={canvasRef} className="pic1d-chart" style={{ height }} />
      <div className="pic-chart-legend">
        {series.map((s, i) => (
          <span key={i}>
            <span className="swatch" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
    </>
  );
}

// 電子位相空間 (x, vx) の散布図。点は小さく半透明にする (仕様通り)
function Pic1dScatterChart({ x, y, height = 140 }: { x: number[]; y: number[]; height?: number }) {
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

    if (x.length === 0) return;

    const xMin = arrayMin(x);
    const xMax = arrayMax(x);
    const yMin = arrayMin(y);
    const yMax = arrayMax(y);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    const xOf = (v: number) => padL + ((v - xMin) / xRange) * plotW;
    const yOf = (v: number) => padT + plotH - ((v - yMin) / yRange) * plotH;

    ctx.fillStyle = "rgba(89, 194, 255, 0.35)"; // #59c2ff 系、半透明の小さい点
    for (let i = 0; i < x.length; i++) {
      ctx.beginPath();
      ctx.arc(xOf(x[i]), yOf(y[i]), 1.1, 0, Math.PI * 2);
      ctx.fill();
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
    ctx.fillText(yMin.toPrecision(3), padL - 4, padT + plotH);
  }, [x, y]);

  return <canvas ref={canvasRef} className="pic1d-chart" style={{ height }} />;
}

// 左電極の RF波形モニタ (仕様通り左電極のみ)。既存 RfPhaseMonitor (2D) と見た目 (配色・
// ヘッダ+キャンバスの2段レイアウト) は揃えるが、データモデルが異なる (Pic1dElectrode は
// project.geometry を経由しない) ため直接の流用はできず、専用実装にする。
// voltage_rf 追加 (prompts/93) 後は RF 成分・CSV 波形のどちらの周波数からも
// プレビュー周期を決められる (evalPic1dVoltage 参照)
function Pic1dRfMonitor({ electrode, t }: { electrode: Pic1dElectrode; t: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const waveforms = electrode.waveforms ?? [];
  // プレビュー窓の周期は「存在する周波数のうち最低」(VoltagePreviewChart の
  // voltagePreviewFreqs と同じ考え方。RF成分とCSV波形をあわせて集める)。
  // voltage_rf があれば (waveforms が空でも) ここで freqs が非空になり波形が表示される
  const freqs = [
    ...rfComponents(electrode.voltage_rf).map((c) => c.freq_hz),
    ...waveforms.map((w) => w.freq_hz),
  ].filter((f) => f > 0);
  const fMin = freqs.length > 0 ? arrayMin(freqs) : 0;
  const period = fMin > 0 ? 1 / fMin : 0;
  const N_SAMPLES = 300;
  const series =
    period > 0
      ? (() => {
          const v = new Array<number>(N_SAMPLES);
          for (let i = 0; i < N_SAMPLES; i++) v[i] = evalPic1dVoltage(electrode, (period * i) / (N_SAMPLES - 1));
          return v;
        })()
      : null;
  const phaseFrac = period > 0 ? t / period - Math.floor(t / period) : 0;

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
    if (!series) return;

    const padL = 38;
    const padR = 8;
    const padT = 6;
    const padB = 16;
    const plotW = rect.width - padL - padR;
    const plotH = rect.height - padT - padB;

    const vMin = arrayMin(series);
    const vMax = arrayMax(series);
    const vRange = vMax - vMin || 1;
    const xOfFrac = (frac: number) => padL + frac * plotW;
    const yOf = (v: number) => padT + plotH - ((v - vMin) / vRange) * plotH;

    ctx.strokeStyle = "#363c48";
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, padT, plotW, plotH);

    ctx.font = "10px system-ui, sans-serif";
    ctx.fillStyle = "#8a919e";
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText("0", padL, padT + plotH + 3);
    ctx.textAlign = "right";
    ctx.fillText(`T=${(period * 1e9).toFixed(1)}ns`, padL + plotW, padT + plotH + 3);
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(vMax.toPrecision(3), padL - 4, padT);
    ctx.textBaseline = "bottom";
    ctx.fillText(vMin.toPrecision(3), padL - 4, padT + plotH);

    ctx.strokeStyle = "#4da3ff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < series.length; i++) {
      const x = xOfFrac(i / (series.length - 1));
      const y = yOf(series[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    const xMarker = xOfFrac(phaseFrac);
    ctx.strokeStyle = "#ff5c5c";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(xMarker, padT);
    ctx.lineTo(xMarker, padT + plotH);
    ctx.stroke();
  }, [series, period, phaseFrac]);

  // RF (voltage_rf) も CSV 波形 (waveforms) も無い (DC のみ) 場合はプレビューする周期が
  // 定まらないため何も表示しない (prompts/93: 「RF か CSV のどちらかがあれば表示」の判定)
  if (period <= 0 || !series) return null;

  return (
    <div className="rf-phase-monitor pic1d-rf-monitor">
      <div className="rf-phase-monitor-header">
        <span className="rf-phase-monitor-title">左電極 RF波形</span>
        <span className="rf-phase-monitor-phase">位相 {(phaseFrac * 100).toFixed(0)}%</span>
      </div>
      <div className="rf-phase-monitor-body">
        <canvas ref={canvasRef} className="rf-phase-monitor-canvas" />
      </div>
    </div>
  );
}

// ---- ライブビュー -------------------------------------------------------------

function Pic1dLiveView({
  lengthUnit,
  pic1d,
  started,
  frame,
}: {
  lengthUnit: LengthUnit;
  pic1d: Pic1dSettings;
  started: Pic1dStartedMsg;
  frame: Pic1dFrameMsg | null;
}) {
  // プラズマ密度は桁が大きく変わりやすいため既定で対数表示にする
  const [densityLog, setDensityLog] = useState(true);
  const xDisp = started.x.map((v) => mToUnit(v, lengthUnit));

  return (
    <>
      <h3>φ(x) [V]</h3>
      {frame ? (
        <Pic1dLineChart x={xDisp} series={[{ label: "φ", values: frame.phi, color: "#59c2ff" }]} />
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
            { label: "n_e", values: frame.n_e, color: "#4da3ff" },
            { label: "n_i", values: frame.n_i, color: "#ffb84d" },
          ]}
          logY={densityLog}
        />
      )}

      <h3>電子位相空間 (x, vx)</h3>
      {frame ? (
        <Pic1dScatterChart x={frame.sample.x.map((v) => mToUnit(v, lengthUnit))} y={frame.sample.vx} />
      ) : (
        <p className="hint">フレーム待機中...</p>
      )}

      <div className="kv">
        <span>ステップ / 時刻</span>
        <span>
          {frame?.step ?? started.step_offset} / {frame ? `${(frame.t * 1e9).toFixed(2)} ns` : "-"}
        </span>
      </div>

      {frame && <Pic1dRfMonitor electrode={pic1d.left} t={frame.t} />}
    </>
  );
}

// ---- 結果ビュー ---------------------------------------------------------------

type Pic1dField = "phi" | "e" | "n_e" | "n_i" | "overlay" | "t_e" | "ionization";

const PIC1D_FIELD_OPTIONS: { value: Pic1dField; label: string }[] = [
  { value: "phi", label: "電位 φ [V]" },
  { value: "e", label: "電場 E [V/m]" },
  { value: "n_e", label: "電子密度 n_e [m^-3]" },
  { value: "n_i", label: "イオン密度 n_i [m^-3]" },
  { value: "overlay", label: "n_e + n_i 重ね" },
  { value: "t_e", label: "電子温度 T_e [eV]" },
  { value: "ionization", label: "電離レート [m^-3 s^-1]" },
];

function isPic1dDensityField(f: Pic1dField): boolean {
  return f === "n_e" || f === "n_i" || f === "overlay";
}

function pic1dFieldSeries(field: Pic1dField, profiles: NonNullable<Pic1dResult["profiles"]>): LineSeries[] {
  switch (field) {
    case "phi":
      return [{ label: "φ", values: profiles.phi, color: "#59c2ff" }];
    case "e":
      return [{ label: "E", values: profiles.e, color: "#59c2ff" }];
    case "n_e":
      return [{ label: "n_e", values: profiles.n_e, color: "#4da3ff" }];
    case "n_i":
      return [{ label: "n_i", values: profiles.n_i, color: "#ffb84d" }];
    case "overlay":
      return [
        { label: "n_e", values: profiles.n_e, color: "#4da3ff" },
        { label: "n_i", values: profiles.n_i, color: "#ffb84d" },
      ];
    case "t_e":
      return [{ label: "T_e", values: profiles.t_e, color: "#6fd08c" }];
    case "ionization":
      return [{ label: "電離レート", values: profiles.ionization, color: "#c792ea" }];
  }
}

function Pic1dCyclePlayer({
  lengthUnit,
  cycle,
  gapM,
  nCells,
}: {
  lengthUnit: LengthUnit;
  cycle: Pic1dCycle;
  gapM: number;
  nCells: number;
}) {
  const [field, setField] = useState<"phi" | "n_e" | "n_i">("phi");
  const [playing, setPlaying] = useState(false);
  const [bin, setBin] = useState(0);
  const [fps, setFps] = useState(10);
  // settings (gap_m/n_cells) から常に再構成する (deriveXGrid 参照。cycle 配列の長さに
  // 依存させると bins=0 等の退化ケースで壊れうるため、信頼できる settings 側を単一の情報源にする)
  const xDisp = deriveXGrid(gapM, nCells).map((v) => mToUnit(v, lengthUnit));

  useEffect(() => {
    if (!playing || cycle.bins <= 0) return;
    const bins = cycle.bins;
    const id = setInterval(() => setBin((b) => (b + 1) % bins), 1000 / fps);
    return () => clearInterval(id);
  }, [playing, fps, cycle.bins]);

  const clampedBin = Math.min(bin, cycle.bins - 1);
  const colorOf = (f: "phi" | "n_e" | "n_i") => (f === "phi" ? "#59c2ff" : f === "n_e" ? "#4da3ff" : "#ffb84d");

  return (
    <>
      <h3>位相分解アニメーション</h3>
      <div className="field">
        <span className="label">表示フィールド</span>
        <select value={field} onChange={(e) => setField(e.target.value as "phi" | "n_e" | "n_i")}>
          <option value="phi">電位 φ [V]</option>
          <option value="n_e">電子密度 n_e [m^-3]</option>
          <option value="n_i">イオン密度 n_i [m^-3]</option>
        </select>
      </div>
      <Pic1dLineChart
        x={xDisp}
        series={[{ label: field, values: cycle[field][clampedBin] ?? [], color: colorOf(field) }]}
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

function Pic1dResultView({ lengthUnit, result }: { lengthUnit: LengthUnit; result: Pic1dResult }) {
  const [field, setField] = useState<Pic1dField>("phi");
  const [logScale, setLogScale] = useState(false);
  const [eedfMode, setEedfMode] = useState<"eedf" | "eepf">("eepf");
  const [eedfLogScale, setEedfLogScale] = useState(true);

  const profiles = result.profiles;
  const xDisp = deriveXGrid(result.settings.gap_m, result.settings.n_cells).map((v) => mToUnit(v, lengthUnit));

  const timingEntries = Object.entries(result.timing).filter(([k]) => k !== "total");
  const timingTotal = result.timing.total ?? timingEntries.reduce((sum, [, v]) => sum + v, 0);
  const timingRows = timingEntries.sort((a, b) => b[1] - a[1]);

  const centerX = result.settings.gap_m / 2;
  const centerNi = profiles ? interpLinear(profiles.x, profiles.n_i, centerX) : null;

  // EEDF/EEPF CSV書き出し (PicPanel の downloadEedfCsv と同じ書式。1D 独自のプレフィックスにする)
  const downloadEedfCsv = (index: number) => {
    const r = result.eedf[index];
    if (!r) return;
    const label = r.label || `E${index + 1}`;
    const lines = ["E_eV,f_eedf_ev-1,f_eepf_ev-1.5"];
    for (let i = 0; i < r.e_centers.length; i++) {
      const e = r.e_centers[i];
      const eepf = e > 0 ? r.f[i] / Math.sqrt(e) : "";
      lines.push(`${e},${r.f[i]},${eepf}`);
    }
    saveTextFile(`pic1d_eedf_eepf_${label}.csv`, lines.join("\n"), "CSV", ["csv"]).catch(() => {
      /* 保存失敗は致命的でないため、ここではエラー表示を省略する (2D同様の簡略化) */
    });
  };

  return (
    <>
      <div className="field">
        <span className="label">結果表示</span>
        <select value={field} onChange={(e) => setField(e.target.value as Pic1dField)}>
          {PIC1D_FIELD_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
      </div>
      {isPic1dDensityField(field) && <Toggle label="対数スケール" checked={logScale} onChange={setLogScale} />}

      {profiles ? (
        <Pic1dLineChart
          x={xDisp}
          series={pic1dFieldSeries(field, profiles)}
          height={160}
          logY={isPic1dDensityField(field) && logScale}
        />
      ) : (
        <p className="hint">時間平均プロファイルがありません (ステップ数0で停止した可能性があります)。</p>
      )}
      {profiles && <p className="hint">時間平均ステップ数: {profiles.avg_steps}</p>}

      {result.cycle && (
        <Pic1dCyclePlayer
          lengthUnit={lengthUnit}
          cycle={result.cycle}
          gapM={result.settings.gap_m}
          nCells={result.settings.n_cells}
        />
      )}

      {result.eedf.length > 0 && (
        <>
          <h3>EEDF/EEPF</h3>
          <div className="field">
            <span className="label">表示</span>
            <select value={eedfMode} onChange={(e) => setEedfMode(e.target.value as "eedf" | "eepf")}>
              <option value="eedf">EEDF f(E) [eV^-1]</option>
              <option value="eepf">EEPF f(E)/√E [eV^-1.5]</option>
            </select>
          </div>
          <Toggle label="縦軸対数スケール" checked={eedfLogScale} onChange={setEedfLogScale} />
          <EedfChart
            regions={result.eedf.map((r) => ({ label: r.label }))}
            results={result.eedf}
            mode={eedfMode}
            logScale={eedfLogScale}
          />
          {result.eedf.map((r, i) => (
            <div key={i} className="collector-row" style={{ cursor: "default" }}>
              <span className="tag" style={{ color: EEDF_CHART_COLORS[i % EEDF_CHART_COLORS.length] }}>
                {r.label || `E${i + 1}`}
              </span>
              {r.total_weight > 0 ? (
                <>
                  <span>T_eff {r.t_eff_ev.toFixed(2)} eV</span>
                  <span>⟨E⟩ {r.mean_energy_ev.toFixed(2)} eV</span>
                  <span>overflow {(r.overflow_frac * 100).toFixed(2)}%</span>
                  <button className="secondary" onClick={() => downloadEedfCsv(i)}>
                    CSV保存
                  </button>
                </>
              ) : (
                <span className="muted">(電子が一度も入りませんでした)</span>
              )}
            </div>
          ))}
        </>
      )}

      <h3>数値サマリ</h3>
      <div className="kv">
        <span>経過時間 (壁時計)</span>
        <span>{result.elapsed_s.toFixed(3)} s</span>
      </div>
      {timingRows.map(([key, sec]) => (
        <div className="kv" key={key}>
          <span>{PIC1D_TIMING_LABELS[key] ?? key}</span>
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
      {result.fn?.left && (
        <div className="kv">
          <span>FN放出 左 J_avg</span>
          <span>{formatNumber(result.fn.left.j_avg)} A/m^2</span>
        </div>
      )}
      {result.fn?.right && (
        <div className="kv">
          <span>FN放出 右 J_avg</span>
          <span>{formatNumber(result.fn.right.j_avg)} A/m^2</span>
        </div>
      )}
      {centerNi != null && (
        <div className="kv">
          <span>中央密度 n_i(gap/2)</span>
          <span>{centerNi.toExponential(3)} m^-3</span>
        </div>
      )}

      <h3>履歴 (マクロ粒子数)</h3>
      <Pic1dLineChart
        x={result.history.step}
        series={[
          { label: "N_e", values: result.history.n_e, color: "#4da3ff" },
          { label: "N_i", values: result.history.n_i, color: "#ffb84d" },
        ]}
        height={90}
      />
    </>
  );
}

// ---- ルート -------------------------------------------------------------------

export default function Plot1dView({ lengthUnit, pic1d, running, started, frame, result, error }: Props) {
  // ライブ表示の条件: 実行中、または「開始はしたがまだ完了結果が無い」区間。
  // done を受け取ると result が入り running は false になるため、そこで自然に結果表示へ切り替わる
  const showLive = running || (started != null && result == null);
  const statusLabel = showLive ? "ライブ実行中" : result ? "計算結果" : "未実行";

  return (
    <div className="pic1d-view">
      <div className="tool-toolbar pic1d-toolbar">
        <span className="pic1d-toolbar-title">PIC-MCC 1D</span>
        <span className="muted">{statusLabel}</span>
      </div>
      <div className="pic1d-body">
        {error && <div className="error pic1d-error">{error}</div>}
        {showLive && started && (
          <Pic1dLiveView lengthUnit={lengthUnit} pic1d={pic1d} started={started} frame={frame} />
        )}
        {!showLive && result && <Pic1dResultView lengthUnit={lengthUnit} result={result} />}
        {!showLive && !result && !error && (
          <p className="hint">PIC 1Dが未実行です。左パネルの「PIC 1D 開始」から実行してください。</p>
        )}
      </div>
    </div>
  );
}
