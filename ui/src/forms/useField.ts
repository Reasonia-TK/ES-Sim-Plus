// path で指す設定項目の値・スキーマ情報・書き換え (1 回の確定が元に戻す 1 件)。

import { useMemo } from "react";
import { useDocument } from "../model/documentStore";
import { paramValues } from "../model/params";
import { coordOf } from "../model/project";
import { usePrefs } from "../prefs/prefs";
import { fieldInfo, getIn, setIn, useProjectSchema, type FieldInfo, type Path } from "../schema/schema";
import type { UnitContext } from "../schema/units";
import { fieldLabel } from "./labels";

export function useUnitContext(): UnitContext {
  const lengthUnit = usePrefs((s) => s.lengthUnit);
  const axisymmetric = useDocument((s) => coordOf(s.project) !== "xy");
  const vars = useDocument((s) => paramValues(s.project).values);
  return useMemo(() => ({ lengthUnit, axisymmetric, vars }), [lengthUnit, axisymmetric, vars]);
}

/** 文書の path の値を書き換える (label は元に戻すメニューに出る名前) */
export function setValue(path: Path, value: unknown, label: string): void {
  useDocument.getState().update(label, (d) => setIn(d, path, value));
}

export interface FieldHandle<T = unknown> {
  info: FieldInfo | null;
  value: T;
  label: string;
  set: (v: T | null | undefined) => void;
}

export function useField<T = unknown>(path: Path): FieldHandle<T> {
  const schema = useProjectSchema();
  const info = useMemo(() => fieldInfo(schema, path), [schema, path]);
  const value = useDocument((s) => getIn(s.project, path)) as T;
  const label = info ? fieldLabel(info) : String(path[path.length - 1]);
  return { info, value, label, set: (v) => setValue(path, v, label) };
}
