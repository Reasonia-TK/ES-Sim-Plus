// 実行設定の目安 (v1 の各パネルが出していたもの): RF 1 周期のステップ数、位相分解のビン数・平均区間の推奨、
// DSMC のセルあたりの粒子数。

import { domainArea, type Project } from "../model/project";
import { rfComponents, waveformFreqs, type VoltageWaveform } from "./waveform";

/** 2D の電極 (Dirichlet の辺、includeRegions なら導体の領域も) の RF・CSV 波形の周波数 (重複なし・昇順) */
export function projectFreqs(p: Project, includeRegions: boolean): number[] {
  const out: number[] = [];
  for (const b of p.geometry.boundaries) {
    if (b.type !== "dirichlet") continue;
    out.push(...waveformFreqs(rfComponents(b.voltage_rf), b.voltage_waveform ? [b.voltage_waveform as VoltageWaveform] : []));
  }
  if (includeRegions) {
    for (const r of p.geometry.regions) {
      if (r.type !== "conductor") continue;
      out.push(...waveformFreqs(rfComponents(r.voltage_rf), r.voltage_waveform ? [r.voltage_waveform as VoltageWaveform] : []));
    }
  }
  return uniqueSorted(out);
}

interface Electrode1d {
  voltage_rf?: unknown;
  waveforms?: VoltageWaveform[];
}

/** 1D の左右電極の周波数 (RF を優先、無ければ CSV 波形。v1 の Pic1dPanel と同じ) */
export function electrodeFreqs(left: Electrode1d | undefined, right: Electrode1d | undefined): number[] {
  const rf = [...rfComponents(left?.voltage_rf), ...rfComponents(right?.voltage_rf)].map((c) => c.freq_hz).filter((f) => f > 0);
  if (rf.length) return uniqueSorted(rf);
  return uniqueSorted([...(left?.waveforms ?? []), ...(right?.waveforms ?? [])].map((w) => w.freq_hz).filter((f) => f > 0));
}

function uniqueSorted(a: number[]): number[] {
  return [...new Set(a)].sort((x, y) => x - y);
}

/** RF 1 周期のステップ数 (dt が無ければ null) */
export function stepsPerPeriod(freq: number, dt: number | null | undefined): number | null {
  if (!dt || dt <= 0 || freq <= 0) return null;
  return 1 / (freq * dt);
}

export interface PhaseBinAdvice {
  /** 基本周期 (最低周波数) の 1 周期のステップ数 (切り捨て) */
  steps: number;
  binsTooMany: boolean;
  avgTooShort: boolean;
  /** 推奨の平均区間 (3 周期) */
  recommendedAvg: number;
}

/** 位相分解の推奨: ビン数は 1 周期のステップ数以下、平均区間は 1 周期以上 (推奨 3 周期) */
export function phaseBinAdvice(freqs: number[], dt: number | null | undefined, bins: number, avgSteps: number | null | undefined, nSteps: number): PhaseBinAdvice | null {
  if (!freqs.length || bins <= 0) return null;
  const spp = stepsPerPeriod(freqs[0], dt);
  if (spp === null) return null;
  const steps = Math.floor(spp);
  const avg = avgSteps ?? Math.max(1, Math.floor(nSteps / 4));
  return { steps, binsTooMany: bins > steps, avgTooShort: avg < steps, recommendedAvg: 3 * steps };
}

/** DSMC のセルあたりの粒子数の目安 (メッシュが無いときは v1 と同じく 面積 / (0.433 (size·scale)²) のセル数で) */
export function dsmcParticlesPerCell(p: Project, nParticles: number, meshScale: number, nCells?: number): number {
  const cells = nCells ?? domainArea(p) / (0.433 * (p.mesh.size * meshScale) ** 2);
  return cells > 0 ? nParticles / cells : Infinity;
}
