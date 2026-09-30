// パラメータと式の束縛の文書操作 (P7f、prompts/132)。
//
// - 文書の params.vars (名前・式・計算した値) と params.bindings (欄の場所・式)。欄には計算した数値も入れておき、
//   ソルバーはそれを使う (バックエンドはスイープで式を計算し直す: backend/es_sim/params.py)。
// - 束縛の場所は文書の道筋で、配列の要素は id を持てば {id}、境界条件は含む辺の ID {edge}、ほかは番号 (領域の
//   並べ替え・削除・名前の変更・境界条件の付け替えでずれないように)。
// - 編集のたびに (documentStore.update): パラメータの式が変わったら束縛した欄を計算し直し、変わっていなければ、
//   値が式と合わなくなった束縛 (欄を手で変えた) と、場所が無くなった・途中の配列 (頂点の列など) の長さが変わった
//   束縛を外す。
// - 欄は確定の直前に stageBinding で束縛 (か外すこと) を置き、同じ編集の中で文書に書く (元に戻すと一緒に戻る)。

import { isDraft, type Draft } from "immer";
import { getIn, setIn, type Path, type PathKey } from "../schema/schema";
import { evaluateExpr, evaluateParams, ExprError, namesIn, nameProblem, renameInExpr } from "./expr";
import { edgeIdsOf, type Project } from "./project";

export type PathToken = string | number | { id: string } | { edge: string };

export interface ParamVar {
  name: string;
  expr: string;
  /** 式を計算した値 (SI) */
  value?: number | null;
  description?: string;
}

export interface ParamBinding {
  path: PathToken[];
  expr: string;
}

export interface Params {
  vars: ParamVar[];
  bindings: ParamBinding[];
}

export function paramsOf(p: Project): Params {
  const raw = p.params as Partial<Params> | null | undefined;
  return { vars: raw?.vars ?? [], bindings: raw?.bindings ?? [] };
}

function draftParams(d: Draft<Project>): Params {
  const p = d as Project;
  if (!p.params || typeof p.params !== "object") (p as Record<string, unknown>).params = { vars: [], bindings: [] };
  const raw = p.params as Partial<Params>;
  raw.vars ??= [];
  raw.bindings ??= [];
  return raw as Params;
}

// ---- 値 ----

export interface ParamEval {
  values: Readonly<Record<string, number>>;
  error: string | null;
}

const EMPTY: ParamEval = Object.freeze({ values: Object.freeze({}), error: null });
const evalCache = new WeakMap<object, ParamEval>();

/** パラメータの値 (計算できなければ値は空でエラー文)。同じ params には同じオブジェクトを返す */
export function paramValues(p: Project): ParamEval {
  const raw = p.params as object | null | undefined;
  if (!raw || typeof raw !== "object") return EMPTY;
  // 下書き (Immer の draft) は同じオブジェクトのまま中身が変わるので覚えない
  const cache = !isDraft(raw);
  const hit = cache ? evalCache.get(raw) : undefined;
  if (hit) return hit;
  let out: ParamEval;
  try {
    out = { values: evaluateParams(paramsOf(p).vars), error: null };
  } catch (e) {
    out = { values: {}, error: (e as Error).message };
  }
  if (cache) evalCache.set(raw, out);
  return out;
}

// ---- 束縛の場所 ----

/** 文書の道筋 (番号) → 束縛の場所 (id・辺の ID で選ぶ)。たどれなければ null */
export function addressOf(p: Project, path: Path): PathToken[] | null {
  const out: PathToken[] = [];
  let cur: unknown = p;
  for (let k = 0; k < path.length; k++) {
    const seg = path[k];
    if (cur === null || typeof cur !== "object") return null;
    if (Array.isArray(cur)) {
      if (typeof seg !== "number" || seg < 0 || seg >= cur.length) return null;
      const el = cur[seg] as Record<string, unknown> | null;
      const isBoundaries = k === 2 && path[0] === "geometry" && path[1] === "boundaries";
      if (el && typeof el === "object" && typeof el.id === "string") out.push({ id: el.id });
      else if (isBoundaries && el && Array.isArray(el.edges) && typeof el.edges[0] === "number") out.push({ edge: edgeIdsOf(p)[el.edges[0] as number] ?? `e${(el.edges[0] as number) + 1}` });
      else out.push(seg);
      cur = cur[seg];
    } else {
      out.push(seg);
      cur = (cur as Record<PathKey, unknown>)[seg];
    }
  }
  return out;
}

/** 束縛の場所 → 文書の道筋 (番号)。無ければ null */
export function resolveAddress(p: Project, addr: readonly PathToken[]): PathKey[] | null {
  const out: PathKey[] = [];
  let cur: unknown = p;
  for (const tok of addr) {
    if (cur === null || typeof cur !== "object") return null;
    if (typeof tok === "object") {
      if (!Array.isArray(cur)) return null;
      let k = -1;
      if ("id" in tok) k = cur.findIndex((el) => el && typeof el === "object" && (el as { id?: unknown }).id === tok.id);
      else {
        const e = edgeIdsOf(p).indexOf(tok.edge);
        if (e >= 0) k = cur.findIndex((el) => el && typeof el === "object" && Array.isArray((el as { edges?: unknown }).edges) && ((el as { edges: number[] }).edges).includes(e));
      }
      if (k < 0) return null;
      out.push(k);
      cur = cur[k];
    } else if (typeof tok === "number") {
      if (!Array.isArray(cur) || tok < 0 || tok >= cur.length) return null;
      out.push(tok);
      cur = cur[tok];
    } else {
      if (Array.isArray(cur) || !(tok in (cur as object))) return null;
      out.push(tok);
      cur = (cur as Record<string, unknown>)[tok];
    }
  }
  return out;
}

const sameAddress = (a: readonly PathToken[], b: readonly PathToken[]) => JSON.stringify(a) === JSON.stringify(b);

/** path の欄の束縛の式 (無ければ null) */
export function bindingAt(p: Project, path: Path): string | null {
  const { bindings } = paramsOf(p);
  if (bindings.length === 0) return null;
  const addr = addressOf(p, path);
  if (!addr) return null;
  return bindings.find((b) => sameAddress(b.path, addr))?.expr ?? null;
}

/** 束縛を付ける・外す (expr = null で外す)。付けるときは欄の値も計算した値にする */
export function setBinding(d: Draft<Project>, path: Path, expr: string | null): void {
  const p = d as Project;
  const addr = addressOf(p, path);
  if (!addr) return;
  if (expr === null) {
    const raw = p.params as Partial<Params> | null | undefined;
    if (!raw?.bindings?.some((b) => sameAddress(b.path, addr))) return;
    raw.bindings = raw.bindings.filter((b) => !sameAddress(b.path, addr));
    return;
  }
  const value = evaluateExpr(expr, paramValues(p).values);
  const params = draftParams(d);
  const i = params.bindings.findIndex((b) => sameAddress(b.path, addr));
  if (i >= 0) params.bindings[i].expr = expr;
  else params.bindings.push({ path: addr, expr });
  setIn(d, path, value);
}

// ---- 編集ごとの同期 ----

/** 値が同じとみなすか (座標の 1e-12 m の丸め・相対 1e-9) */
function near(a: number, b: number): boolean {
  return Math.abs(a - b) <= 1e-12 + 1e-9 * Math.max(Math.abs(a), Math.abs(b));
}

const varsKey = (p: Project) => JSON.stringify(paramsOf(p).vars.map((v) => [v.name, v.expr]));

/** 番号で選ぶ途中の配列 (頂点の列など) の長さが変わったか (頂点を足す・消すと番号がずれる) */
function structureChanged(base: Project, cur: Project, addr: readonly PathToken[]): boolean {
  const pb = resolveAddress(base, addr);
  const pc = resolveAddress(cur, addr);
  if (!pb || !pc) return false;
  let b: unknown = base;
  let c: unknown = cur;
  for (let k = 0; k < addr.length; k++) {
    if (typeof addr[k] === "number" && (!Array.isArray(b) || !Array.isArray(c) || b.length !== c.length)) return true;
    b = (b as Record<PathKey, unknown>)[pb[k]];
    c = (c as Record<PathKey, unknown>)[pc[k]];
  }
  return false;
}

let staged: { path: Path; expr: string | null } | null = null;

/** 欄の確定の直前に、束縛を付ける (expr) か外す (null) ことを置く。次の編集で文書に書く (同じ処理の中だけ有効) */
export function stageBinding(path: Path, expr: string | null): void {
  const mine = { path, expr };
  staged = mine;
  queueMicrotask(() => {
    if (staged === mine) staged = null;
  });
}

export function takeStagedBinding(): { path: Path; expr: string | null } | null {
  const s = staged;
  staged = null;
  return s;
}

/** 置いてあった束縛を書く (付けるのは欄の値が式の値と合うときだけ = 欄の変更が文書に入ったとき) */
export function applyStagedBinding(d: Draft<Project>, s: { path: Path; expr: string | null }): void {
  const p = d as Project;
  if (s.expr === null) {
    setBinding(d, s.path, null);
    return;
  }
  let value: number;
  try {
    value = evaluateExpr(s.expr, paramValues(p).values);
  } catch {
    return;
  }
  const now = getIn(p, s.path);
  if (typeof now === "number" && near(now, value)) setBinding(d, s.path, s.expr);
}

/**
 * 編集の後に: パラメータの式が変わったら各パラメータの値と束縛した欄を計算し直す。変わっていなければ、値が式と
 * 合わなくなった束縛 (欄を手で変えた) を外す。場所が無くなった・途中の配列の長さが変わった束縛はいつも外す。
 * パラメータを計算できない (式の誤り) ときは何もしない (パラメータのページがエラーを出す)
 */
export function syncParams(d: Draft<Project>, base: Project): void {
  const p = d as Project;
  const { vars, bindings } = paramsOf(p);
  if (vars.length === 0 && bindings.length === 0) return;
  let values: Record<string, number>;
  try {
    values = evaluateParams(vars);
  } catch {
    return;
  }
  const params = draftParams(d);
  for (const v of params.vars) if (v.value !== values[v.name]) v.value = values[v.name];
  const changed = varsKey(base) !== varsKey(p);
  const baseBinds = paramsOf(base).bindings;
  const keep: ParamBinding[] = [];
  for (const b of params.bindings) {
    const at = resolveAddress(p, b.path);
    if (!at || structureChanged(base, p, b.path)) continue;
    let value: number;
    try {
      value = evaluateExpr(b.expr, values);
    } catch {
      keep.push(b);
      continue;
    }
    const now = getIn(p, at);
    if (typeof now !== "number") continue;
    if (!near(now, value)) {
      const isNew = !baseBinds.some((x) => sameAddress(x.path, b.path));
      if (!changed && !isNew) continue;
      setIn(d, at, value);
    }
    keep.push(b);
  }
  if (keep.length !== params.bindings.length) params.bindings = keep;
}

/**
 * 開いたとき: パラメータを計算して値と束縛した欄を書く (パラメータが正、手で書き換えた JSON にも合わせる)。
 * 場所の無い束縛は外す。計算できなければ何もしない
 */
export function applyParams(p: Project): void {
  const raw = p.params as Partial<Params> | null | undefined;
  if (!raw || typeof raw !== "object") return;
  const vars = raw.vars ?? [];
  let values: Record<string, number>;
  try {
    values = evaluateParams(vars);
  } catch {
    return;
  }
  for (const v of vars) v.value = values[v.name];
  const keep: ParamBinding[] = [];
  for (const b of raw.bindings ?? []) {
    const at = resolveAddress(p, b.path);
    if (!at || typeof getIn(p, at) !== "number") continue;
    try {
      setIn(p, at, evaluateExpr(b.expr, values));
    } catch {
      // 計算できない束縛は残す (パラメータのページで直せる)
    }
    keep.push(b);
  }
  raw.bindings = keep;
}

// ---- パラメータの表 ----

/** 使われていない名前 p1, p2, … */
export function nextParamName(p: Project): string {
  const used = new Set(paramsOf(p).vars.map((v) => v.name));
  for (let i = 1; ; i++) if (!used.has(`p${i}`)) return `p${i}`;
}

export function addParam(d: Draft<Project>, name: string, expr = "0"): void {
  draftParams(d).vars.push({ name, expr });
}

/** 名前の検査 (エラー文、使えれば null) */
export function paramNameError(p: Project, from: string | null, to: string): string | null {
  const prob = nameProblem(to);
  if (prob) return prob;
  if (to !== from && paramsOf(p).vars.some((v) => v.name === to)) return `${to} は使われています`;
  return null;
}

/** 名前を変える (ほかのパラメータ・束縛の式の中の名前も) */
export function renameParam(d: Draft<Project>, from: string, to: string): void {
  const params = draftParams(d);
  for (const v of params.vars) {
    if (v.name === from) v.name = to;
    v.expr = renameInExpr(v.expr, from, to);
  }
  for (const b of params.bindings) b.expr = renameInExpr(b.expr, from, to);
}

/** 式の検査 (ほかのパラメータとあわせて計算できるか。エラー文、正しければ null) */
export function paramExprError(p: Project, name: string, expr: string): string | null {
  const vars = paramsOf(p).vars.map((v) => (v.name === name ? { name: v.name, expr } : v));
  try {
    evaluateParams(vars);
    return null;
  } catch (e) {
    return e instanceof ExprError ? e.message : String(e);
  }
}

export function setParamExpr(d: Draft<Project>, name: string, expr: string): void {
  const v = draftParams(d).vars.find((x) => x.name === name);
  if (v) v.expr = expr;
}

export function setParamDescription(d: Draft<Project>, name: string, text: string): void {
  const v = draftParams(d).vars.find((x) => x.name === name);
  if (!v) return;
  if (text) v.description = text;
  else delete v.description;
}

/** name を使っているほかのパラメータと束縛の数 */
export function paramUsers(p: Project, name: string): { vars: string[]; bindings: number } {
  const uses = (src: string) => {
    try {
      return namesIn(src).has(name);
    } catch {
      return false;
    }
  };
  const { vars, bindings } = paramsOf(p);
  return { vars: vars.filter((v) => v.name !== name && uses(v.expr)).map((v) => v.name), bindings: bindings.filter((b) => uses(b.expr)).length };
}

/** パラメータを消す (使っている束縛も外す。欄の値はそのまま)。ほかのパラメータが使っていれば消さない (false) */
export function deleteParam(d: Draft<Project>, name: string): boolean {
  const p = d as Project;
  if (paramUsers(p, name).vars.length) return false;
  const params = draftParams(d);
  params.vars = params.vars.filter((v) => v.name !== name);
  params.bindings = params.bindings.filter((b) => {
    try {
      return !namesIn(b.expr).has(name);
    } catch {
      return true;
    }
  });
  return true;
}

/** 束縛を外す (場所で) */
export function removeBinding(d: Draft<Project>, addr: readonly PathToken[]): void {
  const raw = (d as Project).params as Partial<Params> | null | undefined;
  if (raw?.bindings) raw.bindings = raw.bindings.filter((b) => !sameAddress(b.path, addr));
}

/** 領域の名前を変えたとき、束縛の場所の {id} も変える */
export function renameRegionInBindings(d: Draft<Project>, from: string, to: string): void {
  const raw = (d as Project).params as Partial<Params> | null | undefined;
  for (const b of raw?.bindings ?? []) {
    if (b.path[0] === "geometry" && b.path[1] === "regions" && typeof b.path[2] === "object" && "id" in b.path[2] && b.path[2].id === from) b.path[2] = { id: to };
  }
}
