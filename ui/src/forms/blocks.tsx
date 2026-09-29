// フォームの入れ物: 見出しつきの区分、有効/無効を切り替えるブロック (v1 の [R]: 無効にしても直前の値を
// 覚えていて、有効に戻すと復元する)、スキーマから残りの項目を並べる AutoFields。

import { useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { useDocument } from "../model/documentStore";
import { usePrefs } from "../prefs/prefs";
import { getIn, objectAt, pathKey, propInfo, setIn, useProjectSchema, type Path } from "../schema/schema";
import { SchemaField, Toggle } from "./SchemaField";

/** 無効にしたブロックの値 (セッションの間だけ覚える) */
const remembered = new Map<string, unknown>();

export function rememberBlock(path: Path, value: unknown): void {
  remembered.set(pathKey(path), structuredClone(value));
}

export function recallBlock(path: Path): unknown {
  const v = remembered.get(pathKey(path));
  return v === undefined ? undefined : structuredClone(v);
}

export function Section({ title, children, actions, defaultOpen = true }: { title: ReactNode; children: ReactNode; actions?: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="form-section">
      <header className="form-section-header">
        <button type="button" className="section-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
          <span className="tree-twisty">{open ? "▾" : "▸"}</span>
          {title}
        </button>
        {actions && <span className="section-actions">{actions}</span>}
      </header>
      {open && <div className="form-section-body">{children}</div>}
    </section>
  );
}

interface OptionalBlockProps {
  /** null で無効になる設定 (例: ["pic", "mcc"]) */
  path: Path;
  title: ReactNode;
  /** 有効にしたときの値 (覚えている値が無いとき) */
  defaults: () => unknown;
  /** 元に戻すメニューに出る名前 */
  label?: string;
  children: ReactNode;
  hint?: ReactNode;
}

/** 有効/無効を切り替えられるブロック (v1 の [R])。無効のときは中身を出さない */
export function OptionalBlock({ path, title, defaults, label, children, hint }: OptionalBlockProps) {
  const value = useDocument((s) => getIn(s.project, path));
  const enabled = value !== null && value !== undefined;
  const name = label ?? (typeof title === "string" ? title : pathKey(path));
  const toggle = (on: boolean) => {
    if (on) {
      const v = recallBlock(path) ?? defaults();
      useDocument.getState().update(name, (d) => setIn(d, path, v));
    } else {
      rememberBlock(path, value);
      useDocument.getState().update(name, (d) => setIn(d, path, null));
    }
  };
  return (
    <section className={`form-section optional${enabled ? " enabled" : ""}`}>
      <header className="form-section-header">
        <Toggle checked={enabled} onChange={toggle} label={<strong>{title}</strong>} />
      </header>
      {hint && !enabled && <p className="hint">{hint}</p>}
      {enabled && <div className="form-section-body">{children}</div>}
    </section>
  );
}

/**
 * path のオブジェクトの項目を、スキーマの順に自動で並べる (exclude に書いたものと、オブジェクト・配列など
 * 専用の部品で扱う項目は除く)。ページに置き忘れた項目も必ず編集できるようにするための受け皿。
 */
export function AutoFields({ path, exclude = [], only }: { path: Path; exclude?: string[]; only?: string[] }) {
  const schema = useProjectSchema();
  const obj = objectAt(schema, path);
  if (!obj) return null;
  const keys = only ?? Object.keys(obj.node.properties ?? {});
  const simple = new Set(["number", "integer", "boolean", "string", "enum", "point"]);
  return (
    <>
      {keys
        .filter((k) => !exclude.includes(k))
        .filter((k) => simple.has(propInfo(schema, obj.node, obj.name, k)?.kind ?? ""))
        .map((k) => (
          <SchemaField key={k} path={[...path, k]} />
        ))}
    </>
  );
}

/** 詳細設定の表示を切り替えるスイッチ (設定欄の見出しに置く) */
export function AdvancedSwitch() {
  const { t } = useTranslation();
  const show = usePrefs((s) => s.showAdvanced);
  const set = usePrefs((s) => s.setShowAdvanced);
  return <Toggle checked={show} onChange={set} label={t("settings.showAdvanced")} />;
}
