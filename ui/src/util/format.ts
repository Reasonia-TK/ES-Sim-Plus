// 数値の表示と解析 (v1 CommitInput.formatNumber と同じ表示規則)。

/** 表示用の数値文字列: |v| が 1e5 以上か 1e-3 未満 (0 を除く) なら指数表記、有効数字は約 6 桁。 */
export function formatNumber(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) {
    return v
      .toExponential(5)
      .replace(/\.?0+e/, "e")
      .replace("e+", "e");
  }
  return String(Number(v.toPrecision(6)));
}

/** 数値の解析 (指数表記可)。空欄・不正な値は null。 */
export function parseNumber(text: string): number | null {
  const s = text.trim();
  if (s === "") return null;
  if (!/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(s)) return null;
  const v = Number(s);
  return Number.isFinite(v) ? v : null;
}

export type LengthUnit = "mm" | "um";

const LENGTH_SCALE: Record<LengthUnit, number> = { mm: 1e3, um: 1e6 };

/** m → 表示単位 */
export function toDisplayLength(m: number, unit: LengthUnit): number {
  return m * LENGTH_SCALE[unit];
}

/** 表示単位 → m */
export function fromDisplayLength(v: number, unit: LengthUnit): number {
  return v / LENGTH_SCALE[unit];
}

export function lengthUnitLabel(unit: LengthUnit): string {
  return unit === "um" ? "µm" : "mm";
}

const SI: [number, string][] = [
  [1e12, "T"],
  [1e9, "G"],
  [1e6, "M"],
  [1e3, "k"],
  [1, ""],
  [1e-3, "m"],
  [1e-6, "µ"],
  [1e-9, "n"],
  [1e-12, "p"],
];

/** SI 接頭辞を付けた表示 (13560000 Hz → "13.56 MHz") */
export function formatSi(v: number, unit: string, digits = 4): string {
  if (!Number.isFinite(v)) return `${v} ${unit}`;
  if (v === 0) return `0 ${unit}`;
  const a = Math.abs(v);
  const [scale, prefix] = SI.find(([s]) => a >= s * 0.9999999) ?? SI[SI.length - 1];
  return `${Number((v / scale).toPrecision(digits))} ${prefix}${unit}`;
}

/** 経過時間 m:ss / h:mm:ss (v1 のステータスバーと同じ) */
export function formatElapsed(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}
