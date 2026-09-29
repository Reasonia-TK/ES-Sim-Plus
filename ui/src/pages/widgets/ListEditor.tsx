// オブジェクトの一覧 (コレクタ・EEDF 領域・シース評価線・DSMC の境界・AMR の矩形など) の編集:
// 追加 (上限あり)・削除・各行の中身は呼び出し側が描く。

import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { setValue } from "../../forms/useField";
import { useDocument } from "../../model/documentStore";
import { getIn, type Path } from "../../schema/schema";
import { Hint, ListItem } from "./common";

interface ListEditorProps<T> {
  path: Path;
  /** 元に戻すメニューの名前 */
  label: string;
  /** 行の見出し */
  title: (item: T, i: number) => ReactNode;
  /** 行の中身 (項目の path は [...path, i, key]) */
  render: (item: T, i: number) => ReactNode;
  /** 追加する要素 (無ければ追加ボタンを出さない) */
  create?: (items: T[]) => T;
  max?: number;
  emptyText?: string;
  addLabel?: string;
  selected?: number | null;
  onSelect?: (i: number) => void;
}

export function ListEditor<T>({ path, label, title, render, create, max, emptyText, addLabel, selected, onSelect }: ListEditorProps<T>) {
  const { t } = useTranslation();
  const items = (useDocument((s) => getIn(s.project, path)) as T[] | undefined) ?? [];
  const full = max !== undefined && items.length >= max;
  return (
    <div className="list-editor">
      {items.length === 0 && emptyText && <p className="muted">{emptyText}</p>}
      {items.map((item, i) => (
        <ListItem
          key={i}
          title={title(item, i)}
          selected={selected === i}
          onSelect={onSelect ? () => onSelect(i) : undefined}
          onRemove={() => setValue(path, items.filter((_, j) => j !== i), label)}
        >
          {render(item, i)}
        </ListItem>
      ))}
      {create && (
        <div className="button-row tight">
          <button type="button" className="button small" disabled={full} onClick={() => setValue(path, [...items, create(items)], label)}>
            {addLabel ?? t("widgets.add")}
          </button>
          {max !== undefined && (
            <span className="muted small">
              {items.length} / {max}
            </span>
          )}
        </div>
      )}
      {full && <Hint tone="warn">{t("widgets.listFull", { max })}</Hint>}
    </div>
  );
}

export { nextLabel } from "../../model/placements";
