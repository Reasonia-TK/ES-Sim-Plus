// 文書の操作 (新規・開く・保存・名前を付けて保存・最近使ったファイル・サンプル)。
// 未保存の変更があるときは先に確認する (v1 には確認が無かった)。ファイルを開くと新しい文書になり、
// 履歴は消える (v1 は読込を元に戻せたが、代わりに開く前に保存を促す)。結果付きのファイルは結果を「読み込んだ実行」
// として並べる。結果付きで保存は種類ごとに 1 つの結果と静電場の結果を書き出す (文書の保存先は変えない)。

import { askUnsaved } from "../app/dialogs";
import { errorText, logError, logInfo, logWarning } from "../app/messages";
import { t } from "../i18n";
import { documentName, isDirty, useDocument, type DocFile } from "../model/documentStore";
import { newProject, normalizeProject, serializeProject, type Project } from "../model/project";
import { useSelection } from "../model/selection";
import { isTauri } from "../util/env";
import { usePrefs, type RecentFile } from "../prefs/prefs";
import { buildResultsBundle, importResultsBundle } from "../results/bundle";
import { useResultsView } from "../results/resultsView";
import { clearRecovery } from "./autosave";
import { findExample } from "./examples";
import { exampleResultsUrl, loadExampleResults, type ExampleResultsMeta } from "./exampleResults";
import { pickAndRead, pickAndWrite, readRecent, writeInPlace } from "./fileAccess";

const src = () => t("msg.source.file");

function currentName(): string {
  return documentName(useDocument.getState(), t("app.untitled"));
}

function resetSelection(): void {
  useSelection.setState({ activeNode: "domain", selectedRegion: null, selectedPlacement: null });
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
  const added = results ? importResultsBundle(results, project, name) : [];
  if (results) logInfo(src(), t("msg.openedWithResults", { name, n: added.length }));
  else logInfo(src(), t("msg.opened", { name }));
  if (added.length) {
    const rv = useResultsView.getState();
    rv.setActiveRun(added[added.length - 1].id);
  }
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

/**
 * サンプルを開く (保存先の無い新しい文書になる。保存すると名前を尋ねる)。保存された計算結果 (examples/results/、
 * prompts/135) があれば続けて読み込み、「読み込んだ実行」として結果の一覧に並べる (結果付きのファイルを開いたときと同じ)
 */
export async function openExample(key: string): Promise<void> {
  const ex = findExample(key);
  if (!ex) return;
  if (!(await confirmDiscardIfDirty())) return;
  let opened: { name: string; project: Project } | null = null;
  try {
    const name = t(`examples.${key}` as "examples.parallel_plates", { defaultValue: key });
    const { project } = normalizeProject(ex.data);
    useDocument.getState().replace(project, null, { untitledName: name });
    resetSelection();
    clearRecovery();
    logInfo(src(), t("msg.exampleOpened", { name }));
    opened = { name, project };
  } catch (e) {
    logError(src(), t("msg.openFailed", { error: errorText(e) }));
  }
  const url = opened ? exampleResultsUrl(key) : undefined;
  if (opened && url) await importExampleResults(url, opened.project, opened.name);
}

/** サンプルの保存された結果を読み込む。読む間に別の文書を開いたら捨てる */
async function importExampleResults(url: string, project: Project, name: string): Promise<void> {
  const serial = useDocument.getState().docSerial;
  logInfo(src(), t("msg.exampleResultsLoading", { name }));
  try {
    const bundle = await loadExampleResults(url);
    if (useDocument.getState().docSerial !== serial) return;
    const meta = bundle.meta as ExampleResultsMeta | undefined;
    const added = importResultsBundle(bundle, project, (key) => {
      const periods = meta?.runs?.[key]?.periods;
      return typeof periods === "number"
        ? t("examples.savedResultsPeriods", { name, periods: periods >= 10 ? Math.round(periods) : Math.round(periods * 10) / 10 })
        : t("examples.savedResults", { name });
    });
    logInfo(src(), t("msg.exampleResultsLoaded", { name, n: added.length, date: meta?.generated ?? "-" }));
    if (added.length) useResultsView.getState().setActiveRun(added[added.length - 1].id);
  } catch (e) {
    logWarning(src(), t("msg.exampleResultsFailed", { name, error: errorText(e) }));
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

/**
 * Tauri: ウィンドウを閉じる前に未保存の確認 (保存・保存しない・キャンセル)。ブラウザは beforeunload (autosave.ts)。
 * 「保存しない」で閉じたら自動保存の控えも消す (次に開いたときに復元を尋ねない)。
 */
export function startCloseGuard(): void {
  if (!isTauri()) return;
  void import("@tauri-apps/api/window")
    .then(({ getCurrentWindow }) => {
      const win = getCurrentWindow();
      return win.onCloseRequested(async (event) => {
        if (!isDirty(useDocument.getState())) return;
        event.preventDefault();
        const choice = await askUnsaved(currentName());
        if (choice === "cancel") return;
        if (choice === "save" && !(await saveDocument())) return;
        if (choice === "discard") clearRecovery();
        await win.destroy();
      });
    })
    .catch(() => {});
}

/** 結果付きで保存 (v1 と同じ形の書き出し。文書の保存先と未保存の印は変えない) */
export async function saveDocumentWithResults(): Promise<boolean> {
  const { project } = useDocument.getState();
  try {
    const results = await buildResultsBundle();
    if (!results) {
      logWarning(src(), t("msg.noResultsToSave"));
      return false;
    }
    const name = suggestedFileName().replace(/.json$/i, "") + "_results.json";
    const file = await pickAndWrite(name, JSON.stringify({ ...project, results }));
    if (!file) return false;
    logInfo(src(), t("msg.savedWithResults", { name: file.name }));
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
