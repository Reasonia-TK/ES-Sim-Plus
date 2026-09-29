// 配色の範囲: 自動 (値の最小・最大) か手動 (min / max の片方だけでも可)、対数表示 (0 以下の値は最小の
// 正の値に寄せる。正の値が無ければ線形に戻す)。v1 CadCanvas の resolveFieldScale と同じ規則。

export interface FieldStats {
  min: number;
  max: number;
  /** 最小の正の値 (無ければ Infinity) */
  minPositive: number;
}

/** NaN・無限大を除いた最小・最大・最小の正の値 (値が無ければ min = max = 0) */
export function fieldStats(values: ArrayLike<number>): FieldStats {
  let min = Infinity;
  let max = -Infinity;
  let minPositive = Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
    if (v > 0 && v < minPositive) minPositive = v;
  }
  if (min > max) return { min: 0, max: 0, minPositive };
  return { min, max, minPositive };
}

export interface ManualRange {
  min: number | null;
  max: number | null;
}

export interface DisplayRange {
  /** 実際に対数で塗るか (対数の指定があっても正の値が無ければ false) */
  log: boolean;
  /** 配色の両端 (対数なら log10 した値) */
  lo: number;
  hi: number;
  /** 対数で 0 以下の値を寄せる先 */
  minPositive: number;
  /** カラーバーに出す両端 (元の単位) */
  labelMin: number;
  labelMax: number;
}

export function resolveRange(stats: FieldStats, log: boolean, manual?: ManualRange): DisplayRange {
  const useLog = log && Number.isFinite(stats.minPositive);
  const labelMin = manual?.min ?? stats.min;
  const labelMax = manual?.max ?? stats.max;
  const tr = (v: number) => (useLog ? Math.log10(v > 0 ? v : stats.minPositive) : v);
  return { log: useLog, lo: tr(labelMin), hi: tr(labelMax), minPositive: stats.minPositive, labelMin, labelMax };
}

/** 値 → 配色の位置 (0〜1、範囲外は端に寄せない。呼び出し側で寄せる) */
export function colorPosition(v: number, r: DisplayRange): number {
  const x = r.log ? Math.log10(v > 0 ? v : r.minPositive) : v;
  const span = r.hi - r.lo;
  return span === 0 ? 0.5 : (x - r.lo) / span;
}

/** カラーバーの目盛り (両端を含む n 個、対数なら 10 の累乗の等比) */
export function colorbarTicks(r: DisplayRange, n = 5): number[] {
  return Array.from({ length: n }, (_, i) => {
    const x = r.lo + ((r.hi - r.lo) * i) / (n - 1);
    return r.log ? Math.pow(10, x) : x;
  });
}

/** カラーバーの数値 (v1 と同じ: 1e4 以上か 1e-3 未満は指数表記、ほかは有効数字 4 桁) */
export function formatColorbarValue(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e4 || a < 1e-3) return v.toExponential(1);
  return String(Number(v.toPrecision(4)));
}
