// 数値入力 (P7c): 作図中に打ち込んだ座標を点にする。
//   "x, y"        絶対座標
//   "@dx, dy"     直前の点からのずれ
//   "@長さ<角度"  直前の点から (角度は +x から反時計回り [°])
//   "長さ<角度"   原点から
//   "長さ"        直前の点からカーソルの向きへ (円の半径・線の長さを打つとき)
// 長さは単位と式が使える ("10 mm"・"1 cm"・"2*5")。単位を書かなければ表示の単位。角度は式 ("90-30")、° は付けても
// よい。

import { evalExpression, parseQuantity, type UnitContext } from "../schema/units";
import type { Vec } from "./geom";

export type CoordError = "empty" | "number" | "unit" | "needLast" | "needDirection";
export type CoordParse = { ok: true; point: Vec } | { ok: false; error: CoordError };

/** 括弧の外の区切りで 2 つに分ける (無ければ null) */
function splitTop(s: string, sep: string): [string, string] | null {
  let depth = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === "(") depth++;
    else if (c === ")") depth--;
    else if (c === sep && depth === 0) return [s.slice(0, i), s.slice(i + 1)];
  }
  return null;
}

function length(text: string, units: UnitContext): { ok: true; value: number } | { ok: false; error: CoordError } {
  const r = parseQuantity(text, undefined, true, units);
  if (!r.ok) return { ok: false, error: r.error === "unit" ? "unit" : "number" };
  if (r.value === null) return { ok: false, error: "empty" };
  return { ok: true, value: r.value };
}

function angle(text: string): number | null {
  const s = text.trim().replace(/(°|deg)$/i, "").trim();
  if (s === "") return null;
  const v = evalExpression(s);
  return v !== null && Number.isFinite(v) ? v : null;
}

/**
 * 打ち込んだ文字を点に。last は直前の点 (相対・長さだけのとき)、direction はカーソルの向き (長さだけのとき、
 * 長さ 1 でなくてよい)
 */
export function parseCoordInput(text: string, ctx: { last: Vec | null; direction: Vec | null; units: UnitContext }): CoordParse {
  let s = text.trim();
  if (s === "") return { ok: false, error: "empty" };
  let base: Vec = [0, 0];
  let relative = false;
  if (s.startsWith("@")) {
    if (!ctx.last) return { ok: false, error: "needLast" };
    base = ctx.last;
    relative = true;
    s = s.slice(1).trim();
    if (s === "") return { ok: false, error: "empty" };
  }
  const polar = splitTop(s, "<");
  if (polar) {
    const r = length(polar[0], ctx.units);
    if (!r.ok) return r;
    const deg = angle(polar[1]);
    if (deg === null) return { ok: false, error: "number" };
    const th = (deg * Math.PI) / 180;
    return { ok: true, point: [base[0] + r.value * Math.cos(th), base[1] + r.value * Math.sin(th)] };
  }
  const xy = splitTop(s, ",");
  if (xy) {
    const x = length(xy[0], ctx.units);
    if (!x.ok) return x;
    const y = length(xy[1], ctx.units);
    if (!y.ok) return y;
    return { ok: true, point: [base[0] + x.value, base[1] + y.value] };
  }
  if (relative) return { ok: false, error: "number" };
  // 長さだけ: 直前の点からカーソルの向きへ
  const L = length(s, ctx.units);
  if (!L.ok) return L;
  if (!ctx.last) return { ok: false, error: "needLast" };
  const d = ctx.direction;
  const n = d ? Math.hypot(d[0], d[1]) : 0;
  if (!d || !(n > 0)) return { ok: false, error: "needDirection" };
  return { ok: true, point: [ctx.last[0] + (d[0] / n) * L.value, ctx.last[1] + (d[1] / n) * L.value] };
}
