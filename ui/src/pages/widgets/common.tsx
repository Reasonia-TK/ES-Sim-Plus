// 部品で共用する小物: 型の項目の情報、単位付きの数値欄、一覧の行。

import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { fieldLabel } from "../../forms/labels";
import { QuantityInput } from "../../forms/SchemaField";
import { useUnitContext } from "../../forms/useField";
import { propInfo, useProjectSchema, type FieldInfo } from "../../schema/schema";
import { displayUnit } from "../../schema/units";
import { Field } from "../inputs";

/** $defs の型の 1 項目の情報 (配列の中の要素など、path でたどりにくいもの用) */
export function useDefField(def: string, key: string): FieldInfo | null {
  const schema = useProjectSchema();
  const node = schema.$defs?.[def];
  return node ? propInfo(schema, node, def, key) : null;
}

/** 型の項目の数値欄 (値と書き込み先は呼び出し側が持つ) */
export function DefQuantity({ def, prop, value, onCommit, label, placeholder }: { def: string; prop: string; value: number | null | undefined; onCommit: (v: number | null) => void; label?: ReactNode; placeholder?: string }) {
  const info = useDefField(def, prop);
  const ctx = useUnitContext();
  const { t } = useTranslation();
  if (!info) return null;
  const unit = displayUnit(info.unit, info.geom, ctx);
  return (
    <Field label={label ?? fieldLabel(info)} unit={unit || undefined}>
      {(id, onError) => (
        <QuantityInput
          id={id}
          info={info}
          ctx={ctx}
          value={value ?? null}
          placeholder={placeholder ?? (info.nullable ? t("input.auto") : undefined)}
          onError={onError}
          onCommit={onCommit}
        />
      )}
    </Field>
  );
}

/** 一覧の 1 行 (見出し・削除ボタン・中身) */
export function ListItem({ title, onRemove, children, selected, onSelect }: { title: ReactNode; onRemove?: () => void; children: ReactNode; selected?: boolean; onSelect?: () => void }) {
  const { t } = useTranslation();
  return (
    <div className={`list-item${selected ? " selected" : ""}`} onClick={onSelect}>
      <div className="list-item-header">
        <span className="list-item-title">{title}</span>
        {onRemove && (
          <button
            type="button"
            className="button small danger"
            onClick={(e) => {
              e.stopPropagation();
              onRemove();
            }}
          >
            {t("tree.delete")}
          </button>
        )}
      </div>
      <div className="list-item-body">{children}</div>
    </div>
  );
}

export function Hint({ children, tone }: { children: ReactNode; tone?: "warn" | "error" }) {
  return <p className={`hint${tone ? ` hint-${tone}` : ""}`}>{children}</p>;
}
