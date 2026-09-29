// 入力欄の部品。数値・文字列とも v1 の CommitInput と同じ流儀: 編集中は下書きを持ち、Enter か
// フォーカスが外れたときに確定 (履歴 1 件)、Esc か不正な値で元に戻す、編集中は外からの変更で上書きしない。
// (単位付き入力・スキーマからの生成は P6b)

import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { usePrefs } from "../prefs/prefs";
import { formatNumber, fromDisplayLength, parseNumber, toDisplayLength } from "../util/format";

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
  onCommit: (v: string) => void;
  /** 不正ならエラー文、正しければ null */
  validate?: (v: string) => string | null;
  onError?: (msg: string | null) => void;
  className?: string;
  placeholder?: string;
  inputMode?: "text" | "decimal";
  autoFocus?: boolean;
  onCancel?: () => void;
  "aria-label"?: string;
}

export function CommitText({ id, value, onCommit, validate, onError, className, placeholder, inputMode, autoFocus, onCancel, ...rest }: CommitTextProps) {
  const [draft, setDraft] = useState(value);
  const [editing, setEditing] = useState(false);
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (!editing) setDraft(value);
  }, [value, editing]);

  const commit = () => {
    setEditing(false);
    const err = validate?.(draft) ?? null;
    onError?.(err);
    if (err !== null) {
      setDraft(value);
      return;
    }
    if (draft !== value) onCommit(draft);
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
      spellCheck={false}
      aria-label={rest["aria-label"]}
      onFocus={() => setEditing(true)}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          ref.current?.blur();
        } else if (e.key === "Escape") {
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
};

/** 長さの欄: 値は m、表示・入力は設定の単位 (mm/µm) */
export function LengthInput({ value, onCommit, min, max, ...rest }: LengthInputProps) {
  const unit = usePrefs((s) => s.lengthUnit);
  return (
    <NumberInput
      {...rest}
      value={toDisplayLength(value, unit)}
      min={min === undefined ? undefined : toDisplayLength(min, unit)}
      max={max === undefined ? undefined : toDisplayLength(max, unit)}
      onCommit={(v) => v !== null && onCommit(fromDisplayLength(v, unit))}
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
