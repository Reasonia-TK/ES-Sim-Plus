// 入力欄の部品。数値・文字列とも v1 の CommitInput と同じ流儀: 編集中は下書きを持ち、Enter か
// フォーカスが外れたときに確定 (履歴 1 件)、Esc か不正な値で元に戻す、編集中は外からの変更で上書きしない。
// (単位付き入力・スキーマからの生成は P6b)

import { useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { useDocument } from "../model/documentStore";
import { bindingAt, paramValues, stageBinding } from "../model/params";
import { usePrefs } from "../prefs/prefs";
import type { Path } from "../schema/schema";
import { parseQuantity } from "../schema/units";
import { formatNumber, lengthUnitLabel, parseNumber, toDisplayLength } from "../util/format";

interface FieldProps {
  label: ReactNode;
  unit?: ReactNode;
  hint?: ReactNode;
  /** 入力欄を描く。onError を入力欄に渡すと、確定できなかった理由をこの欄の下に出す */
  children: (id: string, onError: (msg: string | null) => void) => ReactNode;
}

/** ラベル・入力・単位・エラーを 1 行に並べる */
export function Field({ label, unit, hint, children }: FieldProps) {
  const id = useId();
  const [error, setError] = useState<string | null>(null);
  return (
    <div className="field">
      <label className="field-label" htmlFor={id}>
        {label}
      </label>
      <div className="field-control">
        {children(id, setError)}
        {unit !== undefined && <span className="field-unit">{unit}</span>}
      </div>
      {error && (
        <div className="field-error" role="alert">
          {error}
        </div>
      )}
      {hint && <div className="field-hint">{hint}</div>}
    </div>
  );
}

interface CommitTextProps {
  id?: string;
  value: string;
  /** 編集を始めたときに出す文 (束縛した欄の式など。無ければ value) */
  editText?: string;
  title?: string;
  onCommit: (v: string) => void;
  /** 不正ならエラー文、正しければ null */
  validate?: (v: string) => string | null;
  onError?: (msg: string | null) => void;
  className?: string;
  placeholder?: string;
  inputMode?: "text" | "decimal";
  autoFocus?: boolean;
  onCancel?: () => void;
  disabled?: boolean;
  "aria-label"?: string;
}

export function CommitText({ id, value, editText, title, onCommit, validate, onError, className, placeholder, inputMode, autoFocus, onCancel, disabled, ...rest }: CommitTextProps) {
  const [draft, setDraft] = useState(value);
  const [editing, setEditing] = useState(false);
  const ref = useRef<HTMLInputElement>(null);
  // 編集を始めたときの文 (変えなければ確定しない)
  const initial = useRef(value);
  // フォーカスで文を式に切り替えたら全選択し直す (選択が外れて打った文字が後ろに付かないように)
  const reselect = useRef(false);
  useLayoutEffect(() => {
    if (!reselect.current) return;
    reselect.current = false;
    ref.current?.select();
  });
  // Esc で抜けたときは確定しない (blur が先に走るので印を付けておく。v1 は Esc でも確定していた)
  const cancelled = useRef(false);
  useEffect(() => {
    if (!editing) setDraft(value);
  }, [value, editing]);

  const commit = () => {
    setEditing(false);
    if (cancelled.current) {
      cancelled.current = false;
      return;
    }
    const err = validate?.(draft) ?? null;
    onError?.(err);
    if (err !== null) {
      setDraft(value);
      return;
    }
    if (draft !== initial.current) onCommit(draft);
    else setDraft(value);
  };
  return (
    <input
      id={id}
      ref={ref}
      className={className ?? "input"}
      value={draft}
      placeholder={placeholder}
      inputMode={inputMode}
      autoFocus={autoFocus}
      disabled={disabled}
      spellCheck={false}
      title={title}
      aria-label={rest["aria-label"]}
      onFocus={() => {
        cancelled.current = false;
        const start = editText ?? value;
        initial.current = start;
        if (start !== draft) reselect.current = true;
        setDraft(start);
        setEditing(true);
      }}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          ref.current?.blur();
        } else if (e.key === "Escape") {
          cancelled.current = true;
          setDraft(value);
          setEditing(false);
          onError?.(null);
          onCancel?.();
          ref.current?.blur();
        }
      }}
    />
  );
}

interface NumberInputProps {
  id?: string;
  value: number | null;
  onCommit: (v: number | null) => void;
  /** 空欄を null (自動) として受け付ける */
  nullable?: boolean;
  placeholder?: string;
  min?: number;
  max?: number;
  /** 下限・上限を含まない (> min / < max) */
  exclusive?: boolean;
  integer?: boolean;
  onError?: (msg: string | null) => void;
  "aria-label"?: string;
}

/** 数値欄 (指数表記可)。範囲外・不正な値は確定せず元に戻す */
export function NumberInput({ value, onCommit, nullable, min, max, exclusive, integer, onError, ...rest }: NumberInputProps) {
  const { t } = useTranslation();
  const text = value === null ? "" : formatNumber(value);
  return (
    <CommitText
      {...rest}
      inputMode="decimal"
      value={text}
      onError={onError}
      validate={(s) => {
        const v = parseNumber(s);
        if (v === null) return s.trim() === "" && nullable ? null : t("input.notNumber");
        if (integer && !Number.isInteger(v)) return t("input.notInteger");
        if (min !== undefined && (exclusive ? v <= min : v < min)) return `${exclusive ? ">" : "≥"} ${formatNumber(min)}`;
        if (max !== undefined && (exclusive ? v >= max : v > max)) return `${exclusive ? "<" : "≤"} ${formatNumber(max)}`;
        return null;
      }}
      onCommit={(s) => onCommit(parseNumber(s))}
    />
  );
}

type LengthInputProps = Omit<NumberInputProps, "nullable" | "value" | "onCommit"> & {
  value: number;
  onCommit: (m: number) => void;
  /** 式を束縛する文書の場所 (P7f。無ければパラメータは 1 回だけ計算して値にする) */
  bindPath?: Path;
};

/** 長さの欄: 値は m、表示・入力は設定の単位 (mm/µm)。単位を付けても入れられる ("0.5 cm"・"5 µm") */
export function LengthInput({ id, value, onCommit, min, max, exclusive, onError, placeholder, "aria-label": ariaLabel, bindPath }: LengthInputProps) {
  const { t } = useTranslation();
  const unit = usePrefs((s) => s.lengthUnit);
  const vars = useDocument((s) => paramValues(s.project).values);
  const bound = useDocument((s) => (bindPath ? bindingAt(s.project, bindPath) : null));
  const ctx = { lengthUnit: unit, axisymmetric: false, vars };
  const parse = (text: string) => parseQuantity(text, undefined, true, ctx);
  const shown = (m: number) => `${formatNumber(toDisplayLength(m, unit))} ${lengthUnitLabel(unit)}`;
  return (
    <CommitText
      id={id}
      aria-label={ariaLabel}
      placeholder={placeholder}
      inputMode="decimal"
      className={bound ? "input bound" : undefined}
      title={bound ? `= ${bound}` : undefined}
      editText={bound ?? undefined}
      value={formatNumber(toDisplayLength(value, unit))}
      onError={onError}
      validate={(text) => {
        const r = parse(text);
        if (!r.ok) return r.error === "param" ? (r.message ?? t("input.notNumber")) : r.error === "unit" ? t("input.badUnit", { unit: lengthUnitLabel(unit) }) : t("input.notNumber");
        if (r.value === null) return t("input.notNumber");
        if (min !== undefined && (exclusive ? r.value <= min : r.value < min)) return `${exclusive ? ">" : "≥"} ${shown(min)}`;
        if (max !== undefined && (exclusive ? r.value >= max : r.value > max)) return `${exclusive ? "<" : "≤"} ${shown(max)}`;
        return null;
      }}
      onCommit={(text) => {
        const r = parse(text);
        if (!r.ok || r.value === null) return;
        if (bindPath) stageBinding(bindPath, r.expr ?? null);
        onCommit(r.value);
      }}
    />
  );
}

interface SelectProps<T extends string> {
  id?: string;
  value: T;
  options: { value: T; label: string; disabled?: boolean }[];
  onChange: (v: T) => void;
  "aria-label"?: string;
}

export function Select<T extends string>({ id, value, options, onChange, ...rest }: SelectProps<T>) {
  return (
    <select id={id} className="input" value={value} aria-label={rest["aria-label"]} onChange={(e) => onChange(e.target.value as T)}>
      {options.map((o) => (
        <option key={o.value} value={o.value} disabled={o.disabled}>
          {o.label}
        </option>
      ))}
    </select>
  );
}
