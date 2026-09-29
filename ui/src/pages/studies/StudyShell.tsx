// スタディのページの入れ物: スタディを使うかどうか (設定を null にしても直前の値を覚えていて戻せる)。

import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { recallBlock, rememberBlock } from "../../forms/blocks";
import { Toggle } from "../../forms/SchemaField";
import { useDocument } from "../../model/documentStore";
import type { SolverKey } from "../../model/project";
import { setIn } from "../../schema/schema";

export function StudyShell({
  settingsKey,
  defaults,
  description,
  run,
  children,
}: {
  settingsKey: SolverKey;
  defaults: () => unknown;
  description: ReactNode;
  /** 実行の操作 (スタディを使うときだけ出す) */
  run?: ReactNode;
  children: ReactNode;
}) {
  const { t } = useTranslation();
  const value = useDocument((s) => s.project[settingsKey]);
  const enabled = value !== null && value !== undefined;
  const toggle = (on: boolean) => {
    const label = t("study.toggle");
    if (on) {
      const v = recallBlock([settingsKey]) ?? defaults();
      useDocument.getState().update(label, (d) => setIn(d, [settingsKey], v));
    } else {
      rememberBlock([settingsKey], value);
      useDocument.getState().update(label, (d) => setIn(d, [settingsKey], null));
    }
  };
  return (
    <>
      <div className="study-toggle">
        <Toggle checked={enabled} onChange={toggle} label={<strong>{t("study.use")}</strong>} />
      </div>
      <p className="hint">{description}</p>
      {enabled && run}
      {enabled && children}
    </>
  );
}
