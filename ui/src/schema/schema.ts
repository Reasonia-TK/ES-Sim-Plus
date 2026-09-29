// プロジェクトの JSON Schema (backend の pydantic、/v2/schema または同梱の project.schema.json) をたどり、
// 設定フォームの 1 項目に要る情報 (種類・単位・範囲・既定値・詳細設定の印・表示名のキー) を引く。

import { useConnection } from "../backend/connection";
import bundled from "./project.schema.json";

export interface JsonSchema {
  type?: string;
  $ref?: string;
  anyOf?: JsonSchema[];
  properties?: Record<string, JsonSchema>;
  required?: string[];
  items?: JsonSchema;
  prefixItems?: JsonSchema[];
  enum?: unknown[];
  const?: unknown;
  default?: unknown;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  minItems?: number;
  maxItems?: number;
  title?: string;
  description?: string;
  $defs?: Record<string, JsonSchema>;
  "x-unit"?: string;
  "x-geom"?: boolean;
  "x-advanced"?: boolean;
}

export type PathKey = string | number;
export type Path = readonly PathKey[];

export type FieldKind = "number" | "integer" | "boolean" | "string" | "enum" | "point" | "object" | "array" | "union" | "unknown";

export interface FieldInfo {
  /** 値の型 (null を外した後) */
  kind: FieldKind;
  /** null (= 自動・無効) を取れる */
  nullable: boolean;
  /** 値の型の schema (null・$ref を外した後) */
  schema: JsonSchema;
  /** この項目を持つモデル (表示名のキー fields.<model>.<key>) */
  model: string;
  key: string;
  /** 項目がオブジェクトのときの型名 ($defs の名前) */
  defName?: string;
  unit?: string;
  geom: boolean;
  advanced: boolean;
  required: boolean;
  default?: unknown;
  description?: string;
  enumValues?: unknown[];
  min?: number;
  max?: number;
  exclusiveMin: boolean;
  exclusiveMax: boolean;
}

export const BUNDLED_SCHEMA = bundled as unknown as JsonSchema;

/** 使うスキーマ: 接続中のバックエンドのもの、無ければ同梱のもの */
export function projectSchema(): JsonSchema {
  return (useConnection.getState().schema?.project as JsonSchema | undefined) ?? BUNDLED_SCHEMA;
}

export function useProjectSchema(): JsonSchema {
  return useConnection((s) => (s.schema?.project as JsonSchema | undefined) ?? BUNDLED_SCHEMA);
}

function defName(ref: string): string {
  return ref.replace(/^#\/\$defs\//, "");
}

/** $ref を解決し、anyOf の null を外す (制約は枝に、メタデータは外側に付いているので両方を返す) */
export function unwrap(root: JsonSchema, node: JsonSchema): { node: JsonSchema; nullable: boolean; defName?: string } {
  let n = node;
  let nullable = false;
  if (n.anyOf) {
    const nonNull = n.anyOf.filter((a) => a.type !== "null");
    nullable = nonNull.length < n.anyOf.length;
    if (nonNull.length !== 1) return { node: { anyOf: nonNull }, nullable };
    n = nonNull[0];
  }
  if (n.$ref) {
    const name = defName(n.$ref);
    const def = root.$defs?.[name];
    if (!def) return { node: {}, nullable };
    return { node: def, nullable, defName: name };
  }
  return { node: n, nullable };
}

function kindOf(n: JsonSchema, geom: boolean): FieldKind {
  if (n.anyOf) return "union";
  if (n.enum || n.const !== undefined) return "enum";
  if (n.type === "array" && n.prefixItems?.length === 2) return "point";
  if (n.type === "number") return "number";
  if (n.type === "integer") return "integer";
  if (n.type === "boolean") return "boolean";
  if (n.type === "string") return "string";
  if (n.type === "array") return geom ? "array" : "array";
  if (n.type === "object" || n.properties) return "object";
  return "unknown";
}

/** オブジェクトの schema の中の 1 項目 */
export function propInfo(root: JsonSchema, parent: JsonSchema, parentName: string, key: string): FieldInfo | null {
  const prop = parent.properties?.[key];
  if (!prop) return null;
  const { node, nullable, defName: dn } = unwrap(root, prop);
  const geom = prop["x-geom"] === true;
  return {
    kind: kindOf(node, geom),
    nullable,
    schema: node,
    model: parentName,
    key,
    defName: dn,
    unit: prop["x-unit"],
    geom,
    advanced: prop["x-advanced"] === true,
    required: parent.required?.includes(key) ?? false,
    default: prop.default,
    description: prop.description,
    enumValues: node.enum ?? (node.const !== undefined ? [node.const] : undefined),
    min: node.exclusiveMinimum ?? node.minimum,
    max: node.exclusiveMaximum ?? node.maximum,
    exclusiveMin: node.exclusiveMinimum !== undefined,
    exclusiveMax: node.exclusiveMaximum !== undefined,
  };
}

/** オブジェクトの schema (と型名) を path でたどる。配列は添字、オブジェクトは項目名 */
export function objectAt(root: JsonSchema, path: Path): { node: JsonSchema; name: string } | null {
  let node: JsonSchema = root;
  let name = "Project";
  for (let i = 0; i < path.length; i++) {
    const seg = path[i];
    if (typeof seg === "number") {
      if (!node.items) return null;
      const u = unwrap(root, node.items);
      node = u.node;
      if (u.defName) name = u.defName;
    } else {
      const prop = node.properties?.[seg];
      if (!prop) return null;
      const u = unwrap(root, prop);
      if (u.node.anyOf) {
        // voltage_rf のような「単一 | リスト」: 次が添字ならリストの枝、そうでなければオブジェクトの枝
        const branches = u.node.anyOf.map((a) => unwrap(root, a));
        const wantList = typeof path[i + 1] === "number";
        const b = branches.find((x) => (wantList ? x.node.type === "array" : x.defName !== undefined));
        if (!b) return null;
        node = b.node;
        if (b.defName) name = b.defName;
        continue;
      }
      node = u.node;
      if (u.defName) name = u.defName;
    }
  }
  return { node, name };
}

/** path の指す項目の情報 (最後の要素が項目名) */
export function fieldInfo(root: JsonSchema, path: Path): FieldInfo | null {
  if (path.length === 0) return null;
  const last = path[path.length - 1];
  if (typeof last === "number") return null;
  const parent = objectAt(root, path.slice(0, -1));
  if (!parent) return null;
  return propInfo(root, parent.node, parent.name, last);
}

/** オブジェクトの項目名の並び (schema の順) */
export function propertyKeys(root: JsonSchema, path: Path): string[] {
  const obj = objectAt(root, path);
  return obj ? Object.keys(obj.node.properties ?? {}) : [];
}

/** 型の既定値から作ったオブジェクト (必須で既定の無い項目は含まない) */
export function schemaDefaults(root: JsonSchema, def: string): Record<string, unknown> {
  const node = root.$defs?.[def];
  const out: Record<string, unknown> = {};
  for (const [k, p] of Object.entries(node?.properties ?? {})) {
    if (p.default !== undefined) out[k] = structuredClone(p.default);
  }
  return out;
}

// ---- project の値の読み書き (path) --------------------------------------------------

export function getIn(obj: unknown, path: Path): unknown {
  let cur: unknown = obj;
  for (const seg of path) {
    if (cur === null || cur === undefined) return undefined;
    cur = (cur as Record<PathKey, unknown>)[seg];
  }
  return cur;
}

/** Immer の draft の path に値を入れる (途中が無ければオブジェクトを作る)。undefined は項目を消す */
export function setIn(draft: unknown, path: Path, value: unknown): void {
  let cur = draft as Record<PathKey, unknown>;
  for (let i = 0; i < path.length - 1; i++) {
    const seg = path[i];
    if (cur[seg] === null || cur[seg] === undefined) cur[seg] = typeof path[i + 1] === "number" ? [] : {};
    cur = cur[seg] as Record<PathKey, unknown>;
  }
  const last = path[path.length - 1];
  if (value === undefined) {
    if (Array.isArray(cur) && typeof last === "number") cur.splice(last, 1);
    else delete cur[last];
  } else {
    cur[last] = value;
  }
}

export function pathKey(path: Path): string {
  return path.join(".");
}
