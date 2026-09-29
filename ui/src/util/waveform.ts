// 電極電圧の波形: V(t) = DC + Σ RF 成分 + CSV 波形 (backend の pic.py と同じ式、v1 VoltagePreviewChart の移植)。

export interface VoltageRf {
  amplitude: number;
  freq_hz: number;
  phase_deg: number;
}

export interface VoltageWaveform {
  freq_hz: number;
  phase: number[];
  v: number[];
}

/** voltage_rf (単一 / リスト / null) を成分の列にする */
export function rfComponents(rf: unknown): VoltageRf[] {
  if (!rf) return [];
  return (Array.isArray(rf) ? rf : [rf]) as VoltageRf[];
}

/** 成分の列を保存する形に戻す (0 個 = null、1 個 = 単一、2 個以上 = リスト。v1 と同じ) */
export function rfValue(components: VoltageRf[]): VoltageRf | VoltageRf[] | null {
  if (components.length === 0) return null;
  return components.length === 1 ? components[0] : components;
}

/** CSV (1 列目 = 時刻、2 列目 = 電圧) → 正規化位相 [0,1) の波形 (v1 と同じ規則。最後の行は周期の折り返しと重なるので捨てる) */
export function parseWaveformCsv(text: string): { phase: number[]; v: number[]; freqHz: number } | { error: "fewRows" | "zeroSpan" } {
  const rows: [number, number][] = [];
  for (const raw of text.split(/\r\n|\r|\n/)) {
    const line = raw.trim();
    if (!line) continue;
    const parts = line.split(/[,\t\s]+/).filter((s) => s.length > 0);
    if (parts.length < 2) continue;
    const t = Number(parts[0]);
    const v = Number(parts[1]);
    if (!Number.isFinite(t) || !Number.isFinite(v)) continue; // ヘッダなどは飛ばす
    rows.push([t, v]);
  }
  if (rows.length < 2) return { error: "fewRows" };
  rows.sort((a, b) => a[0] - b[0]);
  const tMin = rows[0][0];
  const tMax = rows[rows.length - 1][0];
  if (tMax === tMin) return { error: "zeroSpan" };
  const kept = rows.slice(0, -1);
  if (kept.length < 2) return { error: "fewRows" };
  return {
    phase: kept.map(([t]) => (t - tMin) / (tMax - tMin)),
    v: kept.map(([, v]) => v),
    freqHz: 1 / (tMax - tMin),
  };
}

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
  return fp[lo] + ((x - xp[lo]) / (xp[hi] - xp[lo])) * (fp[hi] - fp[lo]);
}

/** CSV 波形の値 (位相 frac(t·f) を、末尾に (phase[0]+1, v[0]) を足して線形補間。pic.py と同じ) */
export function evalWaveform(wf: VoltageWaveform, t: number): number {
  const raw = t * wf.freq_hz;
  const frac = raw - Math.floor(raw);
  return interpLinear(frac, [...wf.phase, wf.phase[0] + 1], [...wf.v, wf.v[0]]);
}

export function evalVoltage(dc: number, rf: VoltageRf[], waveforms: VoltageWaveform[], t: number): number {
  let v = dc;
  for (const c of rf) v += c.amplitude * Math.sin(2 * Math.PI * c.freq_hz * t + (c.phase_deg * Math.PI) / 180);
  for (const w of waveforms) v += evalWaveform(w, t);
  return v;
}

/** 波形に含まれる周波数 (RF 成分と CSV 波形、0 Hz は除く) */
export function waveformFreqs(rf: VoltageRf[], waveforms: VoltageWaveform[]): number[] {
  return [...rf.map((c) => c.freq_hz), ...waveforms.map((w) => w.freq_hz)].filter((f) => f > 0);
}

/** 時間軸の単位 (ns / µs / ms / s) */
export function pickTimeUnit(tMax: number): { label: string; scale: number } {
  if (tMax >= 1) return { label: "s", scale: 1 };
  if (tMax >= 1e-3) return { label: "ms", scale: 1e3 };
  if (tMax >= 1e-6) return { label: "µs", scale: 1e6 };
  return { label: "ns", scale: 1e9 };
}
