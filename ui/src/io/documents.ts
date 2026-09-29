// 文書の操作 (新規・開く・保存・名前を付けて保存・最近使ったファイル・サンプル)。
// 未保存の変更があるときは先に確認する (v1 には確認が無かった)。ファイルを開くと新しい文書になり、
// 履歴は消える (v1 は読込を元に戻せたが、代わりに開く前に保存を促す)。

import { askUnsaved } from "../app/dialogs";
import { errorText, logError, logInfo, logWarning } from "../app/messages";
import { t } from "../i18n";
import { documentName, isDirty, useDocument, type DocFile } from "../model/documentStore";
import { newProject, normalizeProject, serializeProject } from "../model/project";
import { useSelection } from "../model/selection";
import { usePrefs, type RecentFile } from "../prefs/prefs";
import { clearRecovery } from "./autosave";
import { findExample } from "./examples";
import { pickAndRead, pickAndWrite, readRecent, writeInPlace } from "./fileAccess";

const src = () => t("msg.source.file");

function currentName(): string {
  return documentName(useDocument.getState(), t("app.untitled"));
}

function resetSelection(): void {
  useSelection.setState({ activeNode: "domain", selectedRegion: null });
}

/** 未保存の変更があれば保存するか尋ねる。続けてよければ true */
export async function confirmDiscardIfDirty(): Promise<boolean> {
  if (!isDirty(useDocument.getState())) return true;
  const choice = await askUnsaved(currentName());
  if (choice === "cancel") return false;
  if (choice === "save") return saveDocument();
  return true;
}

function openText(text: string, file: DocFile | null, opts?: { untitledName?: string }): void {
  const { project, results } = normalizeProject(JSON.parse(text));
  useDocument.getState().replace(project, file, { untitledName: opts?.untitledName ?? null });
  resetSelection();
  clearRecovery();
  const name = file?.name ?? opts?.untitledName ?? t("app.untitled");
  if (results) logWarning(src(), t("msg.openedWithResults", { name }));
  else logInfo(src(), t("msg.opened", { name }));
}

export async function newDocument(): Promise<void> {
  if (!(await confirmDiscardIfDirty())) return;
  useDocument.getState().replace(newProject(), null);
  resetSelection();
  clearRecovery();
  logInfo(src(), t("msg.newDocument"));
}

export async function openDocument(): Promise<void> {
  if (!(await confirmDiscardIfDirty())) return;
  try {
    const opened = await pickAndRead();
    if (!opened) return;
    openText(opened.text, opened.file);
    usePrefs.getState().addRecent(opened.file);
  } catch (e) {
    logError(src(), t("msg.openFailed", { error: errorText(e) }));
  }
}

export async function openRecentFile(r: RecentFile): Promise<void> {
  if (!(await confirmDiscardIfDirty())) return;
  try {
    const opened = await readRecent(r);
    openText(opened.text, opened.file);
    usePrefs.getState().addRecent(opened.file);
  } catch (e) {
    logError(src(), t("msg.openFailed", { error: errorText(e) }));
    usePrefs.getState().removeRecent(r);
    logInfo(src(), t("msg.removedMissingRecent", { name: r.name }));
  }
}

/** サンプルを開く (保存先の無い新しい文書になる。保存すると名前を尋ねる) */
export async function openExample(key: string): Promise<void> {
  const ex = findExample(key);
  if (!ex) return;
  if (!(await confirmDiscardIfDirty())) return;
  try {
    const name = t(`examples.${key}` as "examples.parallel_plates", { defaultValue: key });
    const { project } = normalizeProject(ex.data);
    useDocument.getState().replace(project, null, { untitledName: name });
    resetSelection();
    clearRecovery();
    logInfo(src(), t("msg.exampleOpened", { name }));
  } catch (e) {
    logError(src(), t("msg.openFailed", { error: errorText(e) }));
  }
}

function suggestedFileName(): string {
  const s = useDocument.getState();
  if (s.file) return s.file.name;
  const base = (s.untitledName ?? "project").replace(/[\\/:*?"<>|]/g, "_");
  return base.endsWith(".json") ? base : `${base}.json`;
}

/** 保存 (保存先が無い・上書きできないときは名前を付けて保存)。保存できたら true */
export async function saveDocument(): Promise<boolean> {
  const { file, project } = useDocument.getState();
  if (!file) return saveDocumentAs();
  try {
    if (!(await writeInPlace(file, serializeProject(project)))) return saveDocumentAs();
    useDocument.getState().markSaved(file);
    clearRecovery();
    logInfo(src(), t("msg.saved", { name: file.name }));
    return true;
  } catch (e) {
    logError(src(), t("msg.saveFailed", { error: errorText(e) }));
    return false;
  }
}

export async function saveDocumentAs(): Promise<boolean> {
  const { project } = useDocument.getState();
  try {
    const file = await pickAndWrite(suggestedFileName(), serializeProject(project));
    if (!file) return false;
    useDocument.getState().markSaved(file);
    clearRecovery();
    if (file.path !== undefined || file.handleId !== undefined) {
      usePrefs.getState().addRecent(file);
      logInfo(src(), t("msg.saved", { name: file.name }));
    } else {
      logInfo(src(), t("msg.downloaded", { name: file.name }));
    }
    return true;
  } catch (e) {
    logError(src(), t("msg.saveFailed", { error: errorText(e) }));
    return false;
  }
}
