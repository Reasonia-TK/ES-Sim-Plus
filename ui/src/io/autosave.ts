// 自動保存 (復元用): 未保存の変更があれば一定間隔で localStorage に写しを置き、次の起動時に
// 復元するか尋ねる。保存・新規・開くで消す。ページを閉じるときに未保存なら確認する。

import { askRecovery } from "../app/dialogs";
import { logInfo } from "../app/messages";
import { t } from "../i18n";
import { isDirty, useDocument, type DocFile } from "../model/documentStore";
import { normalizeProject, type Project } from "../model/project";

export const RECOVERY_KEY = "es-sim-ui.recovery";
export const AUTOSAVE_INTERVAL_MS = 30_000;

interface Recovery {
  savedAt: number;
  name: string;
  file: DocFile | null;
  untitledName: string | null;
  project: Project;
}

export function writeRecovery(): void {
  const s = useDocument.getState();
  if (!isDirty(s)) return;
  const rec: Recovery = {
    savedAt: Date.now(),
    name: s.file?.name ?? s.untitledName ?? t("app.untitled"),
    file: s.file,
    untitledName: s.untitledName,
    project: s.project,
  };
  try {
    localStorage.setItem(RECOVERY_KEY, JSON.stringify(rec));
  } catch {
    // 容量超過などは無視 (復元できないだけ)
  }
}

export function clearRecovery(): void {
  try {
    localStorage.removeItem(RECOVERY_KEY);
  } catch {
    // localStorage が使えない環境
  }
}

export function readRecovery(): Recovery | null {
  try {
    const text = localStorage.getItem(RECOVERY_KEY);
    if (!text) return null;
    const rec = JSON.parse(text) as Recovery;
    normalizeProject(rec.project);
    return rec;
  } catch {
    return null;
  }
}

/** 起動時: 復元用の写しがあれば尋ねる */
export async function offerRecovery(): Promise<void> {
  const rec = readRecovery();
  if (!rec) return;
  if (await askRecovery(rec.name, rec.savedAt)) {
    const { project } = normalizeProject(rec.project);
    useDocument.getState().replace(project, rec.file, { untitledName: rec.untitledName, dirty: true });
    logInfo(t("msg.source.file"), t("msg.recoveryRestored"));
  } else {
    clearRecovery();
  }
}

let timer: ReturnType<typeof setInterval> | null = null;

export function startAutosave(): void {
  if (timer !== null) return;
  timer = setInterval(() => {
    if (isDirty(useDocument.getState())) writeRecovery();
  }, AUTOSAVE_INTERVAL_MS);
  window.addEventListener("beforeunload", (e) => {
    if (!isDirty(useDocument.getState())) return;
    writeRecovery();
    e.preventDefault();
  });
}
