import { useEffect, useRef, useState } from "react";
import type { VoltageRf, VoltageWaveform } from "../types";

/**
 * 電極電位 V(t) プレビュー (prompts/80)
 * V(t) = DC + Σ RF成分 + CSV波形 の合成波形を、backend (pic.py) と厳密に同じ式で
 * フロントのみで評価して折れ線描画する。境界の Dirichlet 値は設定から決定的に
 * 計算できる (シミュレーション実行なしでプレビュー可能) ため backend には触れない。
 * ProfilePanel と同様に依存追加禁止のため canvas に直描きする
 */

interface Props {
  voltage: number;
  rf: VoltageRf[]; // 呼び出し側で rfComponents() により正規化済みの配列を渡すこと
  waveform: VoltageWaveform | null;
}

const V_COLOR = "#4da3ff";
const PAD_L = 46;
const PAD_R = 8;
const PAD_T = 8;
const PAD_B = 18;
const N_SAMPLES = 600;
const TICKS = 4;

// 存在する周波数 (RF各成分 + CSV波形) を集める。0Hz は周期性を持たないので除外する。
// 空配列 = DC のみでプレビューする時間軸が定まらない (呼び出し側でコンポーネント自体を
// 表示しない判断に使う想定。本コンポーネント内でも null 描画で二重に防御する)
export function voltagePreviewFreqs(rf: VoltageRf[], waveform: VoltageWaveform | null): number[] {
  const freqs = rf.map((c) => c.freq_hz).filter((f) => f > 0);
  if (waveform && waveform.freq_hz > 0) freqs.push(waveform.freq_hz);
  return freqs;
}

// np.interp 相当 (範囲外はクランプ、範囲内は線形補間)。xp は昇順を仮定する
function interpLinear(x: number, xp: number[], fp: number[]): number {
  const n = xp.length;
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

// CSV波形 V_wf(t) の評価。pic.py の _eval_waveform と同じ式:
// 位相 frac(t·freq_hz) (0〜1 の繰り返し) を、末尾に (phase[0]+1, v[0]) を仮想的に
// 足した配列で線形補間する (1周期ループの折返しを連続にするため、backendと厳密一致)
function evalWaveform(wf: VoltageWaveform, t: number): number {
  const raw = t * wf.freq_hz;
  const frac = raw - Math.floor(raw); // JS の % は負数で符号を保つため frac(x) は floor で計算する
  const phaseExt = [...wf.phase, wf.phase[0] + 1];
  const vExt = [...wf.v, wf.v[0]];
  return interpLinear(frac, phaseExt, vExt);
}

// V(t) = V_dc + Σ_k A_k sin(2π f_k t + φ_k) + V_wf(t)
// pic.py の _dirichlet_values と厳密一致 (phase_deg は度数法。backend も
// math.radians で変換してから sin に渡している点に注意)
export function evalVoltage(voltage: number, rf: VoltageRf[], waveform: VoltageWaveform | null, t: number): number {
  let v = voltage;
  for (const c of rf) {
    v += c.amplitude * Math.sin(2 * Math.PI * c.freq_hz * t + (c.phase_deg * Math.PI) / 180);
  }
  if (waveform) v += evalWaveform(waveform, t);
  return v;
}

// 時間軸の自動スケール (ns/µs/ms/s)。窓の最大時刻 tMax [s] の桁から単位を選ぶ
// RfPhaseMonitor (prompts/82) でも同じ基準で時間軸をスケールしたいため export して共用する
export function pickTimeUnit(tMax: number): { label: string; scale: number } {
  if (tMax >= 1) return { label: "s", scale: 1 };
  if (tMax >= 1e-3) return { label: "ms", scale: 1e3 };
  if (tMax >= 1e-6) return { label: "µs", scale: 1e6 };
  return { label: "ns", scale: 1e9 };
}

// 描画に使ったスケールをホバー処理でも再利用するための情報
interface Scale {
  padL: number;
  plotW: number;
  tMaxDisp: number;
  n: number;
}

export default function VoltagePreviewChart({ voltage, rf, waveform }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scaleRef = useRef<Scale | null>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const freqs = voltagePreviewFreqs(rf, waveform);
  const fMin = freqs.length > 0 ? Math.min(...freqs) : 0;
  // 表示窓: 存在する周波数のうち最低周波数の2周期分 (仕様)
  const tMax = fMin > 0 ? 2 / fMin : 0;

  // サンプル列は毎レンダーで計算する (600点程度なので負荷は軽微。ProfilePanel の
  // ように非同期取得を伴わないため useMemo にせずシンプルにしている)
  const samples =
    tMax > 0
      ? (() => {
          const t = new Array<number>(N_SAMPLES);
          const v = new Array<number>(N_SAMPLES);
          for (let i = 0; i < N_SAMPLES; i++) {
            const ti = (tMax * i) / (N_SAMPLES - 1);
            t[i] = ti;
            v[i] = evalVoltage(voltage, rf, waveform, ti);
          }
          return { t, v };
        })()
      : null;

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

    if (!samples) {
      scaleRef.current = null;
      return;
    }

    const plotW = rect.width - PAD_L - PAD_R;
    const plotH = rect.height - PAD_T - PAD_B;

    const { label: unitLabel, scale: unitScale } = pickTimeUnit(tMax);
    const tMaxDisp = tMax * unitScale;
    const tRangeDisp = tMaxDisp || 1;

    const vMin = Math.min(...samples.v);
    const vMax = Math.max(...samples.v);
    const vRange = vMax - vMin || 1;

    scaleRef.current = { padL: PAD_L, plotW, tMaxDisp, n: samples.t.length };

    const xOf = (tDisp: number) => PAD_L + (tDisp / tRangeDisp) * plotW;
    const yOf = (v: number) => PAD_T + plotH - ((v - vMin) / vRange) * plotH;

    // 枠
    ctx.strokeStyle = "#363c48";
    ctx.lineWidth = 1;
    ctx.strokeRect(PAD_L, PAD_T, plotW, plotH);

    // 目盛り線とラベル (横軸: 時間、縦軸: V)
    ctx.font = "10px system-ui, sans-serif";
    for (let i = 0; i <= TICKS; i++) {
      const tDisp = (tRangeDisp * i) / TICKS;
      const x = xOf(tDisp);
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.beginPath();
      ctx.moveTo(x, PAD_T);
      ctx.lineTo(x, PAD_T + plotH);
      ctx.stroke();
      ctx.fillStyle = "#8a919e";
      ctx.textAlign = i === 0 ? "left" : i === TICKS ? "right" : "center";
      ctx.textBaseline = "top";
      ctx.fillText(tDisp.toFixed(2), x, PAD_T + plotH + 4);

      const frac = (TICKS - i) / TICKS;
      const v = vMin + vRange * frac;
      const y = PAD_T + plotH * (i / TICKS);
      ctx.fillStyle = V_COLOR;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(v.toPrecision(3), PAD_L - 6, y);
    }

    // 軸ラベル
    ctx.fillStyle = "#8a919e";
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(`t [${unitLabel}]`, PAD_L + plotW / 2, rect.height - 2);

    // 曲線
    ctx.strokeStyle = V_COLOR;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < samples.t.length; i++) {
      const x = xOf(samples.t[i] * unitScale);
      const y = yOf(samples.v[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // ホバーカーソル (縦線)
    if (hoverX !== null && hoverX >= PAD_L && hoverX <= PAD_L + plotW) {
      ctx.strokeStyle = "rgba(255,255,255,0.4)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hoverX, PAD_T);
      ctx.lineTo(hoverX, PAD_T + plotH);
      ctx.stroke();
    }
  }, [voltage, rf, waveform, tMax, samples, hoverX]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const el = canvasRef.current;
    const sc = scaleRef.current;
    if (!el || !sc || !samples) return;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left;
    setHoverX(x);
    const tRangeDisp = sc.tMaxDisp || 1;
    const tDisp = ((x - sc.padL) / sc.plotW) * tRangeDisp;
    let idx = Math.round((tDisp / tRangeDisp) * (sc.n - 1));
    idx = Math.max(0, Math.min(sc.n - 1, idx));
    setHoverIdx(idx);
  };

  const handleMouseLeave = () => {
    setHoverX(null);
    setHoverIdx(null);
  };

  // DCのみ (周波数なし) はプレビューする時間軸が定まらないため何も描画しない
  // (呼び出し側 FieldPanel でも同条件で分岐しているが、単体利用時の防御として残す)
  if (!samples) return null;

  const { label: unitLabel, scale: unitScale } = pickTimeUnit(tMax);

  return (
    <div className="voltage-preview">
      <div className="voltage-preview-canvas-wrap">
        <canvas
          ref={canvasRef}
          className="voltage-preview-canvas"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        />
        {hoverIdx !== null && (
          <div className="voltage-preview-hover">
            t: {(samples.t[hoverIdx] * unitScale).toFixed(2)} {unitLabel} &nbsp; V: {samples.v[hoverIdx].toFixed(2)} V
          </div>
        )}
      </div>
    </div>
  );
}
