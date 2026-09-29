// 単位の表示と、単位付きの入力 ("13.56 MHz"、"5 mm"、"2 ns"、"10 mTorr" など) の解析。
// 保存値は常に SI (スキーマの x-unit)。幾何の長さ (x-geom) は表示単位 (mm / µm) で出す。

import { formatNumber, type LengthUnit } from "../util/format";

/** スキーマの x-unit → 表示 */
const LABELS: Record<string, string> = {
  "1": "",
  deg: "°",
  "m^-3": "m⁻³",
  "m^-2": "m⁻²",
  "m^2": "m²",
  "C/m^3": "C/m³",
  "m^2/(V*s)": "m²/(V·s)",
};

/** SI 接頭辞を付けて入力できる単位 (記号 → そのまま SI の基本単位として扱う) */
const PREFIXABLE = new Set(["m", "s", "Hz", "V", "eV", "Pa", "K", "T", "A/m", "A", "C", "kg", "sccm"]);

const PREFIX: Record<string, number> = {
  T: 1e12,
  G: 1e9,
  M: 1e6,
  k: 1e3,
  c: 1e-2,
  m: 1e-3,
  µ: 1e-6,
  μ: 1e-6,
  u: 1e-6,
  n: 1e-9,
  p: 1e-12,
  f: 1e-15,
};

/** 単位ごとの別名 (記号 → 基本単位での倍率) */
const ALIASES: Record<string, Record<string, number>> = {
  Pa: { Torr: 133.322368, torr: 133.322368, mTorr: 0.133322368, mtorr: 0.133322368, bar: 1e5, mbar: 100 },
  "m^-3": { "cm^-3": 1e6, "cm-3": 1e6, "/cm3": 1e6, "m^-3": 1, "m-3": 1 },
  deg: { "°": 1, deg: 1 },
};

export interface UnitContext {
  lengthUnit: LengthUnit;
  /** 軸対称では「奥行き 1 m あたり」の単位を全体の量に読み替える (A/m → A など、v1 と同じ) */
  axisymmetric: boolean;
}

/** 入出力に使う単位の記号 (幾何の長さは表示単位) */
export function displayUnit(unit: string | undefined, geom: boolean, ctx: UnitContext): string {
  if (geom) return ctx.lengthUnit === "um" ? "µm" : "mm";
  if (!unit) return "";
  if (ctx.axisymmetric) {
    if (unit === "A/m") return "A";
  }
  return LABELS[unit] ?? unit;
}

/** 保存値 (SI) → 表示の数値 */
export function toDisplay(v: number, geom: boolean, ctx: UnitContext): number {
  if (!geom) return v;
  return v * (ctx.lengthUnit === "um" ? 1e6 : 1e3);
}

/** 表示の数値 → 保存値 (SI) */
export function fromDisplay(v: number, geom: boolean, ctx: UnitContext): number {
  if (!geom) return v;
  return v / (ctx.lengthUnit === "um" ? 1e6 : 1e3);
}

export function formatQuantity(v: number | null | undefined, geom: boolean, ctx: UnitContext): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "";
  return formatNumber(toDisplay(v, geom, ctx));
}

const NUMBER_RE = /^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*)$/;

export type ParseResult = { ok: true; value: number | null } | { ok: false; error: "number" | "unit" };

/**
 * 入力を解析して保存値 (SI) にする。単位を省けば表示単位 (幾何は mm/µm、ほかは SI の基本単位)。
 * 単位を書くときは同じ次元の単位だけ受け付ける (例: 周波数の欄に "13.56 MHz"、長さの欄に "0.5 cm")。
 */
export function parseQuantity(text: string, unit: string | undefined, geom: boolean, ctx: UnitContext): ParseResult {
  const s = text.trim();
  if (s === "") return { ok: true, value: null };
  const m = NUMBER_RE.exec(s);
  if (!m) return { ok: false, error: "number" };
  const num = Number(m[1]);
  if (!Number.isFinite(num)) return { ok: false, error: "number" };
  const suffix = m[2].trim();
  if (suffix === "") return { ok: true, value: fromDisplay(num, geom, ctx) };
  const base = geom ? "m" : unit;
  if (!base) return { ok: false, error: "unit" };
  const factor = unitFactor(suffix, base, ctx);
  if (factor === null) return { ok: false, error: "unit" };
  return { ok: true, value: num * factor };
}

/** 単位記号 → 基本単位への倍率 (合わなければ null) */
export function unitFactor(sym: string, base: string, ctx?: UnitContext): number | null {
  const s = sym.replace("μ", "µ");
  const alias = ALIASES[base]?.[s];
  if (alias !== undefined) return alias;
  const label = LABELS[base];
  if (s === base || (label && label !== "" && s === label)) return 1;
  if (ctx?.axisymmetric && base === "A/m" && s === "A") return 1;
  if (PREFIXABLE.has(base) || (ctx?.axisymmetric && base === "A/m")) {
    const b = ctx?.axisymmetric && base === "A/m" ? "A" : base;
    if (s.endsWith(b) && s.length === b.length + 1) {
      const p = PREFIX[s[0]];
      // m (メートル) の c は cm のみ、ほかの単位に c は付けない
      if (p !== undefined && (s[0] !== "c" || b === "m")) return p;
    }
  }
  return null;
}
