// プロジェクトファイルの読み書き。3 通りの実装を同じ関数で隠す:
// - Tauri: ネイティブのダイアログとファイルパス (上書き保存・最近使ったファイルはパスで)
// - ブラウザ (File System Access API あり、Chromium 系): ファイルハンドルを IndexedDB に保存して同じことをする
// - ブラウザ (API なし): 開くは <input type=file>、保存はダウンロード (上書き保存はできず毎回ダウンロード)

import type { DocFile } from "../model/documentStore";
import type { RecentFile } from "../prefs/prefs";
import { isTauri } from "../util/env";
import { getHandle, putHandle } from "./handleStore";

export interface OpenedFile {
  text: string;
  file: DocFile;
}

const JSON_TYPES: FilePickerAcceptType[] = [{ description: "ES-Sim project (JSON)", accept: { "application/json": [".json"] } }];
const TAURI_FILTERS = [{ name: "ES-Sim project (JSON)", extensions: ["json"] }];

function baseName(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

function isAbort(e: unknown): boolean {
  return e instanceof DOMException && e.name === "AbortError";
}

function hasFsAccess(): boolean {
  return typeof window !== "undefined" && typeof window.showOpenFilePicker === "function";
}

async function ensurePermission(h: FileSystemFileHandle, mode: "read" | "readwrite"): Promise<boolean> {
  if (!h.queryPermission || !h.requestPermission) return true;
  if ((await h.queryPermission({ mode })) === "granted") return true;
  return (await h.requestPermission({ mode })) === "granted";
}

async function writeHandle(h: FileSystemFileHandle, text: string): Promise<void> {
  const w = await h.createWritable();
  await w.write(text);
  await w.close();
}

function pickWithInput(): Promise<File | null> {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json,application/json";
    input.onchange = () => resolve(input.files?.[0] ?? null);
    input.addEventListener("cancel", () => resolve(null));
    input.click();
  });
}

function download(name: string, text: string, mime = "application/json"): void {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** ファイルを選んで読む (キャンセルは null) */
export async function pickAndRead(): Promise<OpenedFile | null> {
  if (isTauri()) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const path = await open({ multiple: false, directory: false, filters: TAURI_FILTERS });
    if (typeof path !== "string") return null;
    const { readTextFile } = await import("@tauri-apps/plugin-fs");
    return { text: await readTextFile(path), file: { name: baseName(path), path } };
  }
  if (hasFsAccess()) {
    try {
      const [h] = await window.showOpenFilePicker!({ multiple: false, types: JSON_TYPES });
      const text = await (await h.getFile()).text();
      return { text, file: { name: h.name, handleId: await putHandle(h) } };
    } catch (e) {
      if (isAbort(e)) return null;
      throw e;
    }
  }
  const f = await pickWithInput();
  return f ? { text: await f.text(), file: { name: f.name } } : null;
}

/** 最近使ったファイルを読む (消えた・権限が無いときは例外) */
export async function readRecent(r: RecentFile): Promise<OpenedFile> {
  if (r.path !== undefined) {
    const { readTextFile } = await import("@tauri-apps/plugin-fs");
    return { text: await readTextFile(r.path), file: { name: r.name, path: r.path } };
  }
  if (r.handleId !== undefined) {
    const h = await getHandle(r.handleId);
    if (!h) throw new Error(`${r.name}: ファイルの参照が見つかりません`);
    if (!(await ensurePermission(h, "read"))) throw new Error(`${r.name}: 読み取りが許可されませんでした`);
    return { text: await (await h.getFile()).text(), file: { name: h.name, handleId: r.handleId } };
  }
  throw new Error(`${r.name}: 開き直せないファイルです`);
}

/** 保存先に上書きする。上書きできない (ダウンロードで開いた) ときは false */
export async function writeInPlace(file: DocFile, text: string): Promise<boolean> {
  if (file.path !== undefined && isTauri()) {
    const { writeTextFile } = await import("@tauri-apps/plugin-fs");
    await writeTextFile(file.path, text);
    return true;
  }
  if (file.handleId !== undefined) {
    const h = await getHandle(file.handleId);
    if (!h) return false;
    if (!(await ensurePermission(h, "readwrite"))) throw new Error(`${file.name}: 書き込みが許可されませんでした`);
    await writeHandle(h, text);
    return true;
  }
  return false;
}

/** 保存先を選んで書く (キャンセルは null)。API の無いブラウザはダウンロード */
export async function pickAndWrite(suggestedName: string, text: string): Promise<DocFile | null> {
  if (isTauri()) {
    const { save } = await import("@tauri-apps/plugin-dialog");
    const path = await save({ defaultPath: suggestedName, filters: TAURI_FILTERS });
    if (!path) return null;
    const { writeTextFile } = await import("@tauri-apps/plugin-fs");
    await writeTextFile(path, text);
    return { name: baseName(path), path };
  }
  if (typeof window.showSaveFilePicker === "function") {
    try {
      const h = await window.showSaveFilePicker({ suggestedName, types: JSON_TYPES });
      await writeHandle(h, text);
      return { name: h.name, handleId: await putHandle(h) };
    } catch (e) {
      if (isAbort(e)) return null;
      throw e;
    }
  }
  download(suggestedName, text);
  return { name: suggestedName };
}

/** バイナリ (PNG など) を保存する (Tauri は保存先を選ぶ、ブラウザはダウンロード。キャンセルは false) */
export async function saveBinaryFile(suggestedName: string, data: Blob, ext: string, description: string): Promise<boolean> {
  if (isTauri()) {
    const { save } = await import("@tauri-apps/plugin-dialog");
    const path = await save({ defaultPath: suggestedName, filters: [{ name: description, extensions: [ext] }] });
    if (!path) return false;
    const { writeFile } = await import("@tauri-apps/plugin-fs");
    await writeFile(path, new Uint8Array(await data.arrayBuffer()));
    return true;
  }
  const url = URL.createObjectURL(data);
  const a = document.createElement("a");
  a.href = url;
  a.download = suggestedName;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  return true;
}

/** 任意のテキスト (CSV など) を保存する (v1 saveFile.ts と同じ振る舞い、キャンセルは false) */
export async function saveTextFile(suggestedName: string, text: string, ext: string, description: string): Promise<boolean> {
  if (isTauri()) {
    const { save } = await import("@tauri-apps/plugin-dialog");
    const path = await save({ defaultPath: suggestedName, filters: [{ name: description, extensions: [ext] }] });
    if (!path) return false;
    const { writeTextFile } = await import("@tauri-apps/plugin-fs");
    await writeTextFile(path, text);
    return true;
  }
  download(suggestedName, text, ext === "csv" ? "text/csv" : "text/plain");
  return true;
}
