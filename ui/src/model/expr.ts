// パラメータの式 (P7f、prompts/132)。backend/es_sim/params.py と同じ文法で、共通の試験データ
// (backend/tests/data/param_expr_cases.json) で両方を確かめる。値は SI。
//
// - 数と単位: 2 mm・13.56MHz・10 mTorr・1e10 cm^-3 (単位は数かかっこの直後だけ。単位の指数も書ける)
// - 四則演算 + - * /、べき ^ (右結合、単項のマイナスより強い: -2^2 = -4)、かっこ
// - 関数 sqrt exp ln log log10 sin cos tan abs、定数 pi
// - パラメータの名前 (英字か _ で始まり英数字と _。関数・pi・単位の記号は名前にできない)

const PREFIX: Record<string, number> = { T: 1e12, G: 1e9, M: 1e6, k: 1e3, c: 1e-2, m: 1e-3, u: 1e-6, "µ": 1e-6, "μ": 1e-6, n: 1e-9, p: 1e-12, f: 1e-15 };
/** SI 接頭辞を付けられる記号 (g は kg の g) */
const PREFIXABLE: Record<string, number> = { m: 1, s: 1, Hz: 1, V: 1, A: 1, C: 1, K: 1, T: 1, Pa: 1, eV: 1, J: 1, W: 1, N: 1, F: 1, g: 1e-3 };
/** 接頭辞を付けない記号 */
const PLAIN: Record<string, number> = { Torr: 133.322368, torr: 133.322368, mTorr: 0.133322368, mtorr: 0.133322368, bar: 1e5, mbar: 100, sccm: 1 };

/** 単位の記号 (指数なし) の SI への倍率。単位でなければ null。c は m にだけ付く (cm) */
export function unitFactor(sym: string): number | null {
  if (Object.hasOwn(PLAIN, sym)) return PLAIN[sym];
  if (Object.hasOwn(PREFIXABLE, sym)) return PREFIXABLE[sym];
  const chars = Array.from(sym);
  if (chars.length >= 2 && Object.hasOwn(PREFIX, chars[0])) {
    const base = chars.slice(1).join("");
    if (Object.hasOwn(PREFIXABLE, base) && (chars[0] !== "c" || base === "m")) return PREFIX[chars[0]] * PREFIXABLE[base];
  }
  return null;
}

const FUNCS: Record<string, (x: number) => number> = {
  sqrt: Math.sqrt,
  exp: Math.exp,
  ln: Math.log,
  log: Math.log10,
  log10: Math.log10,
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  abs: Math.abs,
};
const CONSTS: Record<string, number> = { pi: Math.PI };
const NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;

export type ExprErrorKind = "syntax" | "unknown" | "value" | "cycle";

export class ExprError extends Error {
  constructor(
    readonly kind: ExprErrorKind,
    message: string,
  ) {
    super(message);
  }
}

/** パラメータ名に使えない理由 (使えれば null) */
export function nameProblem(name: string): string | null {
  if (!NAME_RE.test(name)) return "英字か _ で始まり、英数字と _ だけの名前にしてください";
  if (Object.hasOwn(FUNCS, name) || Object.hasOwn(CONSTS, name)) return `${name} は関数・定数の名前なので使えません`;
  if (unitFactor(name) !== null) return `${name} は単位の記号なので使えません`;
  return null;
}

// ---- 字句と構文 ----

type Tok = { kind: "num" | "name" | "op"; text: string };

const TOKEN_RE = /\s*(?:((?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)|([A-Za-z_µμ][A-Za-z0-9_]*)|([-+*/^()]))/y;

function tokens(src: string): Tok[] {
  const s = src.replace(/×/g, "*").replace(/÷/g, "/").replace(/π/g, "pi");
  const out: Tok[] = [];
  let pos = 0;
  while (pos < s.length) {
    if (s.slice(pos).trim() === "") break;
    TOKEN_RE.lastIndex = pos;
    const m = TOKEN_RE.exec(s);
    if (!m || TOKEN_RE.lastIndex === pos) throw new ExprError("syntax", `式を読めません (${JSON.stringify(s.slice(pos).trim().slice(0, 10))} の所)`);
    out.push(m[1] !== undefined ? { kind: "num", text: m[1] } : m[2] !== undefined ? { kind: "name", text: m[2] } : { kind: "op", text: m[3] });
    pos = TOKEN_RE.lastIndex;
  }
  return out;
}

export type ExprNode =
  | { k: "num"; v: number }
  | { k: "name"; name: string }
  | { k: "neg"; a: ExprNode }
  | { k: "bin"; op: "+" | "-" | "*" | "/"; a: ExprNode; b: ExprNode }
  | { k: "pow"; a: ExprNode; b: ExprNode }
  | { k: "call"; f: string; a: ExprNode };

class Parser {
  private i = 0;
  private toks: Tok[];

  constructor(src: string) {
    this.toks = tokens(src);
  }

  private peek(k = 0): Tok | undefined {
    return this.toks[this.i + k];
  }

  private isOp(ch: string, k = 0): boolean {
    const t = this.peek(k);
    return t !== undefined && t.kind === "op" && t.text === ch;
  }

  parse(): ExprNode {
    if (this.toks.length === 0) throw new ExprError("syntax", "式が空です");
    const v = this.expr();
    if (this.i !== this.toks.length) throw new ExprError("syntax", `式を読めません (${JSON.stringify(this.toks[this.i].text)} の所)`);
    return v;
  }

  private expr(): ExprNode {
    let v = this.term();
    while (this.isOp("+") || this.isOp("-")) {
      const op = this.toks[this.i++].text as "+" | "-";
      v = { k: "bin", op, a: v, b: this.term() };
    }
    return v;
  }

  private term(): ExprNode {
    let v = this.unary();
    while (this.isOp("*") || this.isOp("/")) {
      const op = this.toks[this.i++].text as "*" | "/";
      v = { k: "bin", op, a: v, b: this.unary() };
    }
    return v;
  }

  private unary(): ExprNode {
    if (this.isOp("-") || this.isOp("+")) {
      const neg = this.toks[this.i++].text === "-";
      const v = this.unary();
      return neg ? { k: "neg", a: v } : v;
    }
    return this.power();
  }

  private power(): ExprNode {
    const b = this.postfix();
    if (this.isOp("^")) {
      this.i++;
      return { k: "pow", a: b, b: this.unary() };
    }
    return b;
  }

  /** 数・かっこの直後の単位 (と指数 ^n / ^-n)。単位でなければ null (読み進めない) */
  private unit(): number | null {
    const t = this.peek();
    if (t === undefined || t.kind !== "name") return null;
    const f = unitFactor(t.text);
    if (f === null) return null;
    this.i++;
    if (this.isOp("^")) {
      // 単位の指数: ^整数 か ^-整数 (^( は式のべきなので単位の指数ではない)
      const neg = this.isOp("-", 1);
      const nt = this.peek(neg ? 2 : 1);
      if (nt !== undefined && nt.kind === "num" && /^\d+$/.test(nt.text)) {
        this.i += neg ? 3 : 2;
        const n = Number(nt.text);
        return f ** (neg ? -n : n);
      }
    }
    return f;
  }

  private postfix(): ExprNode {
    const t = this.peek();
    if (t === undefined) throw new ExprError("syntax", "式が途中で終わっています");
    if (t.kind === "num") {
      this.i++;
      const u = this.unit();
      return { k: "num", v: Number(t.text) * (u ?? 1) };
    }
    if (this.isOp("(")) {
      this.i++;
      const v = this.expr();
      if (!this.isOp(")")) throw new ExprError("syntax", "かっこが閉じていません");
      this.i++;
      const u = this.unit();
      return u !== null ? { k: "bin", op: "*", a: v, b: { k: "num", v: u } } : v;
    }
    if (t.kind === "name") {
      this.i++;
      const name = t.text;
      if (Object.hasOwn(FUNCS, name)) {
        if (!this.isOp("(")) throw new ExprError("syntax", `関数 ${name} の後に ( が要ります`);
        this.i++;
        const a = this.expr();
        if (!this.isOp(")")) throw new ExprError("syntax", "かっこが閉じていません");
        this.i++;
        return { k: "call", f: name, a };
      }
      if (Object.hasOwn(CONSTS, name)) return { k: "num", v: CONSTS[name] };
      return { k: "name", name };
    }
    throw new ExprError("syntax", `式を読めません (${JSON.stringify(t.text)} の所)`);
  }
}

/** 式 → 構文木 (読めなければ ExprError("syntax")) */
export function parseExpr(src: string): ExprNode {
  return new Parser(src).parse();
}

function collectNames(n: ExprNode, out: Set<string>): void {
  switch (n.k) {
    case "name":
      out.add(n.name);
      return;
    case "neg":
    case "call":
      collectNames(n.a, out);
      return;
    case "bin":
    case "pow":
      collectNames(n.a, out);
      collectNames(n.b, out);
      return;
  }
}

/** 式に出てくるパラメータの名前 (関数・定数・単位は除く)。読めなければ ExprError */
export function namesIn(src: string): Set<string> {
  const out = new Set<string>();
  collectNames(parseExpr(src), out);
  return out;
}

function finite(v: number, what: string): number {
  if (!Number.isFinite(v)) throw new ExprError("value", `${what} を計算できません`);
  return v;
}

function evalNode(n: ExprNode, values: Readonly<Record<string, number>>): number {
  switch (n.k) {
    case "num":
      return n.v;
    case "name":
      if (!Object.hasOwn(values, n.name)) {
        if (unitFactor(n.name) !== null) throw new ExprError("unknown", `${n.name} は単位の記号です (単位は数かかっこの直後にだけ書けます)`);
        throw new ExprError("unknown", `${n.name} というパラメータはありません`);
      }
      return values[n.name];
    case "neg":
      return -evalNode(n.a, values);
    case "bin": {
      const a = evalNode(n.a, values);
      const b = evalNode(n.b, values);
      if (n.op === "+") return a + b;
      if (n.op === "-") return a - b;
      if (n.op === "*") return a * b;
      if (b === 0) throw new ExprError("value", "0 で割っています");
      return a / b;
    }
    case "pow":
      // Python の ** と同じく、あふれ・複素数になるべきは計算できない
      return finite(evalNode(n.a, values) ** evalNode(n.b, values), "べき乗");
    case "call": {
      const a = evalNode(n.a, values);
      return finite(FUNCS[n.f](a), `${n.f}(${a})`);
    }
  }
}

/** 式の値 (SI)。values はパラメータの値。計算できなければ ExprError */
export function evaluateExpr(src: string, values: Readonly<Record<string, number>> = {}): number {
  const v = evalNode(parseExpr(src), values);
  if (!Number.isFinite(v)) throw new ExprError("value", "値が有限ではありません");
  return v;
}

export interface ParamDef {
  name: string;
  expr: string;
}

/**
 * パラメータを依存の順に計算した値 (名前 → SI)。overrides の名前は式の代わりにその値。
 * 名前の重複・使えない名前・知らない名前・循環・計算できない式は ExprError (メッセージにパラメータ名)
 */
export function evaluateParams(vars: readonly ParamDef[], overrides: Readonly<Record<string, number>> = {}): Record<string, number> {
  const exprs = new Map<string, string>();
  for (const v of vars) {
    const prob = nameProblem(v.name);
    if (prob) throw new ExprError("syntax", `パラメータ名 ${JSON.stringify(v.name)}: ${prob}`);
    if (exprs.has(v.name)) throw new ExprError("syntax", `パラメータ名 ${v.name} が重複しています`);
    exprs.set(v.name, v.expr);
  }
  for (const name of Object.keys(overrides)) if (!exprs.has(name)) throw new ExprError("unknown", `${name} というパラメータはありません`);
  const deps = new Map<string, Set<string>>();
  for (const [name, src] of exprs) {
    if (Object.hasOwn(overrides, name)) {
      deps.set(name, new Set());
      continue;
    }
    let ds: Set<string>;
    try {
      ds = namesIn(src);
    } catch (e) {
      const err = e as ExprError;
      throw new ExprError(err.kind, `パラメータ ${name}: ${err.message}`);
    }
    for (const d of ds) if (!exprs.has(d)) throw new ExprError("unknown", `パラメータ ${name}: ${d} というパラメータはありません`);
    deps.set(name, ds);
  }
  const values: Record<string, number> = {};
  const state = new Map<string, 1 | 2>();
  const visit = (name: string, chain: string[]) => {
    if (state.get(name) === 2) return;
    if (state.get(name) === 1) {
      const cyc = [...chain.slice(chain.indexOf(name)), name];
      throw new ExprError("cycle", `パラメータが循環しています (${cyc.join(" → ")})`);
    }
    state.set(name, 1);
    for (const d of [...deps.get(name)!].sort()) visit(d, [...chain, name]);
    if (Object.hasOwn(overrides, name)) values[name] = overrides[name];
    else {
      try {
        values[name] = evaluateExpr(exprs.get(name)!, values);
      } catch (e) {
        const err = e as ExprError;
        throw new ExprError(err.kind, `パラメータ ${name}: ${err.message}`);
      }
    }
    state.set(name, 2);
  };
  for (const name of exprs.keys()) visit(name, []);
  return values;
}

/** 式の中のパラメータの名前を置き換える (名前の変更。単位・関数の位置の同じ綴りは変えない) */
export function renameInExpr(src: string, from: string, to: string): string {
  let toks: Tok[];
  try {
    toks = tokens(src);
  } catch {
    return src;
  }
  if (!toks.some((t) => t.kind === "name" && t.text === from)) return src;
  // 字句の位置を探しながら置き換える (空白はそのまま。× ÷ π は * / pi になる)
  let out = "";
  let pos = 0;
  const s = src.replace(/×/g, "*").replace(/÷/g, "/").replace(/π/g, "pi");
  for (let k = 0; k < toks.length; k++) {
    const t = toks[k];
    const at = s.indexOf(t.text, pos);
    if (at < 0) return src;
    out += s.slice(pos, at);
    const prev = toks[k - 1];
    const isUnit = prev !== undefined && (prev.kind === "num" || (prev.kind === "op" && prev.text === ")")) && unitFactor(t.text) !== null;
    out += t.kind === "name" && t.text === from && !isUnit ? to : t.text;
    pos = at + t.text.length;
  }
  return out + s.slice(pos);
}
