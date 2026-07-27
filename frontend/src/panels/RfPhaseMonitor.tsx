import { useEffect, useRef } from "react";
import type { Project, VoltageRf, VoltageWaveform } from "../types";
import { rfComponents } from "../types";
import { evalVoltage, voltagePreviewFreqs, pickTimeUnit } from "./VoltagePreviewChart";
import { EDGE_LABELS_RZ, EDGE_LABELS_RZ_X0, EDGE_LABELS_XY } from "./FieldPanel";

/**
 * RF位相モニタ (prompts/82)
 * PICライブ表示中、現在フレーム時刻 t が基本周期のどの位相にあたるかを、電極ごとの
 * 合成波形 V(t) の折れ線 + 現在位相の縦線マーカーで示す。合成波形の評価式は
 * VoltagePreviewChart (prompts/80) の evalVoltage をそのまま流用し (backend の
 * _dirichlet_values と厳密一致させる式を二重実装しない)、本ファイルでは
 * 「どの電極を対象にするか」「基本周期 T をどう決めるか」だけを扱う。
 */

interface Props {
  project: Project;
  t: number; // 現在フレームの時刻 [s] (picFrame.t)
}

interface Electrode {
  label: string;
  voltage: number;
  rf: VoltageRf[];
  waveform: VoltageWaveform | null;
}

// 折れ線の色 (最大4電極分)。既存の凡例配色 (pic-chart-legend / bc-legend) と揃えて
// アクセント=青、以降オレンジ/緑/紫の順にする
const LINE_COLORS = ["#4da3ff", "#ffb84d", "#6fd08c", "#b070f0"];
// 現在位相マーカーは上記4色と衝突しない赤系にする
const MARKER_COLOR = "#ff5c5c";
const PAD_L = 38;
const PAD_R = 8;
const PAD_T = 6;
const PAD_B = 16;
const N_SAMPLES = 300;
const MAX_ELECTRODES = 4;

// project から時間依存 (RF成分 or CSV波形あり) の電極一覧を集める。
// 収集範囲は backend pic.py の _find_rf_freq と厳密に一致させる (Dirichlet辺 + conductor領域)。
// ラベルは backend の電極ラベル規約 ("edge0".."edge3" → FieldPanel の EDGE_LABELS_* で辺名化、
// conductor は region.id) に合わせ、他パネルの表記と混乱しないようにする
function collectElectrodes(project: Project): Electrode[] {
  const edgeLabels =
    project.coord === "rz" ? EDGE_LABELS_RZ : project.coord === "rz_x0" ? EDGE_LABELS_RZ_X0 : EDGE_LABELS_XY;
  const out: Electrode[] = [];
  for (const bc of project.geometry.boundaries) {
    if (bc.type !== "dirichlet") continue;
    const rf = rfComponents(bc.voltage_rf);
    const waveform = bc.voltage_waveform ?? null;
    if (voltagePreviewFreqs(rf, waveform).length === 0) continue; // DCのみは対象外
    for (const e of bc.edges) {
      out.push({ label: edgeLabels[e] ?? `辺${e}`, voltage: bc.voltage, rf, waveform });
    }
  }
  for (const region of project.geometry.regions) {
    if (region.type !== "conductor") continue;
    const rf = rfComponents(region.voltage_rf);
    const waveform = region.voltage_waveform ?? null;
    if (voltagePreviewFreqs(rf, waveform).length === 0) continue;
    out.push({ label: region.id, voltage: region.voltage ?? 0, rf, waveform });
  }
  return out;
}

export default function RfPhaseMonitor({ project, t }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const electrodes = collectElectrodes(project);
  // 基本周期 T = 1/(全時間依存ソースの最小周波数)。backend _find_rf_freq (全境界・全conductor
  // 領域の freq_hz の min) と同じ集合から求める (voltagePreviewFreqs が 0Hz を除外するのも同じ)
  const freqs = electrodes.flatMap((e) => voltagePreviewFreqs(e.rf, e.waveform));
  const fMin = freqs.length > 0 ? Math.min(...freqs) : 0;
  const T = fMin > 0 ? 1 / fMin : 0;

  const shown = electrodes.slice(0, MAX_ELECTRODES);
  const overflowCount = electrodes.length - shown.length;

  // 電極ごとの1周期分 [0, T) サンプル列。evalVoltage は VoltagePreviewChart からの流用
  const series =
    T > 0
      ? shown.map((e) => {
          const v = new Array<number>(N_SAMPLES);
          for (let i = 0; i < N_SAMPLES; i++) {
            const ti = (T * i) / (N_SAMPLES - 1);
            v[i] = evalVoltage(e.voltage, e.rf, e.waveform, ti);
          }
          return v;
        })
      : [];

  // 現在位相 (0〜1) = (t mod T)/T。JS の % は負数で符号を保つため、backend の
  // _phase_bin (t % period) と同じ結果になるよう floor ベースの frac で計算する
  // (picFrame.t は常に0以上想定だが、evalWaveform 同様の書式に揃えて安全側にする)
  const phaseFrac = T > 0 ? t / T - Math.floor(t / T) : 0;

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

    if (T <= 0 || series.length === 0) return;

    const plotW = rect.width - PAD_L - PAD_R;
    const plotH = rect.height - PAD_T - PAD_B;

    const { label: unitLabel, scale: unitScale } = pickTimeUnit(T);
    const tMaxDisp = T * unitScale || 1;

    const allV = series.flat();
    const vMin = Math.min(...allV);
    const vMax = Math.max(...allV);
    const vRange = vMax - vMin || 1;

    const xOfFrac = (frac: number) => PAD_L + frac * plotW;
    const yOf = (v: number) => PAD_T + plotH - ((v - vMin) / vRange) * plotH;

    // 枠
    ctx.strokeStyle = "#363c48";
    ctx.lineWidth = 1;
    ctx.strokeRect(PAD_L, PAD_T, plotW, plotH);

    // 軸ラベル (最小限: 横軸は0とT、縦軸はmin/max のみ)
    ctx.font = "10px system-ui, sans-serif";
    ctx.fillStyle = "#8a919e";
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText("0", PAD_L, PAD_T + plotH + 3);
    ctx.textAlign = "right";
    ctx.fillText(`${tMaxDisp.toFixed(2)} ${unitLabel}`, PAD_L + plotW, PAD_T + plotH + 3);

    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(vMax.toPrecision(3), PAD_L - 4, PAD_T);
    ctx.textBaseline = "bottom";
    ctx.fillText(vMin.toPrecision(3), PAD_L - 4, PAD_T + plotH);

    // 電極ごとの合成波形
    series.forEach((v, si) => {
      ctx.strokeStyle = LINE_COLORS[si % LINE_COLORS.length];
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < v.length; i++) {
        const x = xOfFrac(i / (v.length - 1));
        const y = yOf(v[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    // 現在位相マーカー (縦線): x = (t mod T)/T の位置。フレーム更新のたびに再描画されるため
    // ライブ受信間隔なりに滑らかに動く (このコンポーネント自体はアニメーションを持たない)
    const xMarker = xOfFrac(phaseFrac);
    ctx.strokeStyle = MARKER_COLOR;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(xMarker, PAD_T);
    ctx.lineTo(xMarker, PAD_T + plotH);
    ctx.stroke();
  }, [T, series, phaseFrac]);

  // DCのみ (時間依存の電極なし) はプレビューする周期が定まらないため何も描画しない
  if (T <= 0) return null;

  const { label: unitLabel, scale: unitScale } = pickTimeUnit(T);

  return (
    <div className="rf-phase-monitor">
      <div className="rf-phase-monitor-header">
        <span className="rf-phase-monitor-title">RF位相</span>
        <span className="rf-phase-monitor-phase">
          位相 {(phaseFrac * 100).toFixed(0)}% (t={(t * unitScale).toFixed(2)} {unitLabel})
        </span>
        <div className="rf-phase-monitor-legend">
          {shown.map((e, i) => (
            <span key={i}>
              <span className="swatch" style={{ background: LINE_COLORS[i % LINE_COLORS.length] }} />
              {e.label}
            </span>
          ))}
          {overflowCount > 0 && <span>他{overflowCount}</span>}
        </div>
      </div>
      <div className="rf-phase-monitor-body">
        <canvas ref={canvasRef} className="rf-phase-monitor-canvas" />
      </div>
    </div>
  );
}
