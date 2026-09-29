// 設定項目の表示名・選択肢・ヒント (翻訳ファイルの fields / enums / hints)。無ければスキーマの title・値そのもの。

import i18n from "../i18n";
import type { FieldInfo } from "../schema/schema";

type AnyT = (key: string, opts?: Record<string, unknown>) => string;
const tt = (key: string, opts?: Record<string, unknown>) => (i18n.t as unknown as AnyT)(key, opts);

/** モデル固有のキー、無ければ共通 (_common) のキー */
function lookup(ns: string, model: string, rest: string): string | null {
  for (const k of [`${ns}.${model}.${rest}`, `${ns}._common.${rest}`]) if (i18n.exists(k)) return tt(k);
  return null;
}

export function fieldLabel(info: Pick<FieldInfo, "model" | "key">, fallback?: string): string {
  return lookup("fields", info.model, info.key) ?? fallback ?? info.key;
}

export function enumLabel(info: Pick<FieldInfo, "model" | "key">, value: unknown): string {
  return lookup("enums", info.model, `${info.key}.${String(value)}`) ?? String(value);
}

/** ヒント: 翻訳ファイルに無ければ (日本語のときだけ) スキーマの説明文 */
export function fieldHint(info: Pick<FieldInfo, "model" | "key" | "description">): string | undefined {
  const h = lookup("hints", info.model, info.key);
  if (h !== null) return h;
  return i18n.language === "ja" ? info.description : undefined;
}

export function translate(key: string, opts?: Record<string, unknown>): string {
  return tt(key, opts);
}
