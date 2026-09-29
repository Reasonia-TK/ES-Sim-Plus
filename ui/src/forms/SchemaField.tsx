// スキーマから作る 1 項目の入力欄。種類 (数値・整数・真偽・選択肢・文字列・座標)・単位・範囲・既定値・
// 表示名・ヒント・詳細設定の印はスキーマと翻訳ファイルから決まる。

import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { CommitText, Field } from "../pages/inputs";
import { usePrefs } from "../prefs/prefs";
import type { FieldInfo, Path } from "../schema/schema";
import { displayUnit, formatQuantity, parseQuantity, type UnitContext } from "../schema/units";
import { formatNumber } from "../util/format";
import { enumLabel, fieldHint } from "./labels";
import { useField, useUnitContext } from "./useField";

/** 範囲の検査 (エラー文、正しければ null) */
export function rangeError(v: number, info: Pick<FieldInfo, "min" | "max" | "exclusiveMin" | "exclusiveMax" | "kind">, ctxShow: (x: number) => string, t: (k: string) => string): string | null {
  if (info.kind === "integer" && !Number.isInteger(v)) return t("input.notInteger");
  if (info.min !== undefined && (info.exclusiveMin ? v <= info.min : v < info.min)) return `${info.exclusiveMin ? ">" : "≥"} ${ctxShow(info.min)}`;
  if (info.max !== undefined && (info.exclusiveMax ? v >= info.max : v > info.max)) return `${info.exclusiveMax ? "<" : "≤"} ${ctxShow(info.max)}`;
  return null;
}

interface QuantityProps {
  id?: string;
  info: FieldInfo;
  value: number | null | undefined;
  onCommit: (v: number | null) => void;
  onError?: (msg: string | null) => void;
  placeholder?: string;
  ctx: UnitContext;
  disabled?: boolean;
  "aria-label"?: string;
}

/** 数値の欄 (単位付きの入力可。空欄は null の項目だけ) */
export function QuantityInput({ id, info, value, onCommit, onError, placeholder, ctx, disabled, ...rest }: QuantityProps) {
  const { t } = useTranslation();
  const tf = t as unknown as (k: string) => string;
  const show = (x: number) => `${formatQuantity(x, info.geom, ctx)} ${displayUnit(info.unit, info.geom, ctx)}`.trim();
  return (
    <CommitText
      id={id}
      inputMode="decimal"
      value={formatQuantity(value, info.geom, ctx)}
      placeholder={placeholder}
      disabled={disabled}
      aria-label={rest["aria-label"]}
      onError={onError}
      validate={(s) => {
        const r = parseQuantity(s, info.unit, info.geom, ctx);
        if (!r.ok) return r.error === "unit" ? t("input.badUnit", { unit: displayUnit(info.unit, info.geom, ctx) || "-" }) : t("input.notNumber");
        if (r.value === null) return info.nullable ? null : t("input.required");
        return rangeError(r.value, info, show, tf);
      }}
      onCommit={(s) => {
        const r = parseQuantity(s, info.unit, info.geom, ctx);
        if (r.ok) onCommit(r.value);
      }}
    />
  );
}

export function Toggle({ id, checked, onChange, label, disabled }: { id?: string; checked: boolean; onChange: (v: boolean) => void; label?: ReactNode; disabled?: boolean }) {
  return (
    <label className="toggle">
      <input id={id} type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <span className="toggle-track" aria-hidden="true" />
      {label !== undefined && <span className="toggle-label">{label}</span>}
    </label>
  );
}

export interface SchemaFieldProps {
  path: Path;
  /** 表示名を差し替える (座標系で呼び名が変わる項目など) */
  label?: ReactNode;
  hint?: ReactNode;
  /** null (空欄) のときの説明 ("自動" など) */
  placeholder?: string;
  /** 詳細設定を閉じていても出す */
  always?: boolean;
  disabled?: boolean;
  /** 選択肢を絞る・並べ替える */
  options?: unknown[];
}

export function SchemaField({ path, label, hint, placeholder, always, disabled, options }: SchemaFieldProps) {
  const { t } = useTranslation();
  const ctx = useUnitContext();
  const showAdvanced = usePrefs((s) => s.showAdvanced);
  const f = useField(path);
  const info = f.info;
  if (!info) return <div className="field-error">? {path.join(".")}</div>;
  if (info.advanced && !showAdvanced && !always) return null;
  const unit = info.kind === "number" || info.kind === "integer" || info.kind === "point" ? displayUnit(info.unit, info.geom, ctx) : "";
  const hintText = hint ?? fieldHint(info);
  const title = typeof hintText === "string" ? hintText : undefined;
  const lbl = <span title={title}>{label ?? f.label}</span>;

  if (info.kind === "boolean") {
    return (
      <div className="field field-toggle">
        <Toggle checked={Boolean(f.value ?? info.default)} disabled={disabled} onChange={(v) => f.set(v)} label={lbl} />
      </div>
    );
  }
  if (info.kind === "enum") {
    const values = (options ?? info.enumValues ?? []) as unknown[];
    const cur = f.value ?? info.default;
    return (
      <Field label={lbl}>
        {(id) => (
          <select id={id} className="input" value={String(cur ?? "")} disabled={disabled} onChange={(e) => f.set(values.find((v) => String(v) === e.target.value))}>
            {info.nullable && <option value="">{placeholder ?? t("input.none")}</option>}
            {values.map((v) => (
              <option key={String(v)} value={String(v)}>
                {enumLabel(info, v)}
              </option>
            ))}
          </select>
        )}
      </Field>
    );
  }
  if (info.kind === "string") {
    return (
      <Field label={lbl}>
        {(id, onError) => (
          <CommitText id={id} value={String(f.value ?? info.default ?? "")} placeholder={placeholder} disabled={disabled} onError={onError} onCommit={(v) => f.set(v)} />
        )}
      </Field>
    );
  }
  if (info.kind === "number" || info.kind === "integer") {
    const v = f.value as number | null | undefined;
    return (
      <Field label={lbl} unit={unit || undefined}>
        {(id, onError) => (
          <QuantityInput
            id={id}
            info={info}
            ctx={ctx}
            value={v === undefined ? ((info.default as number | null | undefined) ?? null) : v}
            placeholder={placeholder ?? (info.nullable ? t("input.auto") : undefined)}
            disabled={disabled}
            onError={onError}
            onCommit={(x) => f.set(x)}
          />
        )}
      </Field>
    );
  }
  if (info.kind === "point") {
    const pt = (f.value as [number, number] | null | undefined) ?? null;
    const axis = info.schema.prefixItems ?? [];
    const one: FieldInfo = { ...info, kind: "number", nullable: false, min: undefined, max: undefined, exclusiveMin: false, exclusiveMax: false };
    return (
      <Field label={lbl} unit={unit || undefined}>
        {(id, onError) => (
          <span className="point-input">
            {[0, 1].map((k) => (
              <QuantityInput
                key={k}
                id={k === 0 ? id : undefined}
                info={{ ...one, schema: axis[k] ?? one.schema }}
                ctx={ctx}
                aria-label={k === 0 ? "x" : "y"}
                value={pt ? pt[k] : null}
                onError={onError}
                onCommit={(x) => {
                  if (x === null) return;
                  const next: [number, number] = pt ? [pt[0], pt[1]] : [0, 0];
                  next[k] = x;
                  f.set(next);
                }}
              />
            ))}
          </span>
        )}
      </Field>
    );
  }
  return (
    <div className="field">
      <span className="field-label">{lbl}</span>
      <span className="muted">{formatValue(f.value)}</span>
    </div>
  );
}

function formatValue(v: unknown): string {
  if (typeof v === "number") return formatNumber(v);
  if (v === null || v === undefined) return "-";
  return JSON.stringify(v).slice(0, 80);
}
