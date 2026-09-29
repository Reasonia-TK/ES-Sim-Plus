// バックエンドのポート番号 (v1 frontend/src/backendPort.ts と同じ規則・同じ保存先)。
// Tauri 内は AppConfig/backend-port.txt だけを読む (サイドカーが起動時に同じファイルを読むので、
// localStorage を優先すると食い違って「未接続」になる — v0.1.0 の事故)。ブラウザは localStorage。

import { isTauri } from "../util/env";

export const DEFAULT_PORT = 8317;
const LS_KEY = "es-sim.backendPort";
const CONFIG_FILE_NAME = "backend-port.txt";

let currentPort = DEFAULT_PORT;

export function parsePort(text: string): number | null {
  const s = text.trim();
  if (!/^\d+$/.test(s)) return null;
  const n = Number(s);
  return n > 0 && n < 65536 ? n : null;
}

export function getPort(): number {
  return currentPort;
}

export async function initPort(): Promise<number> {
  if (isTauri()) {
    try {
      const { readTextFile, exists, BaseDirectory } = await import("@tauri-apps/plugin-fs");
      if (await exists(CONFIG_FILE_NAME, { baseDir: BaseDirectory.AppConfig })) {
        const n = parsePort(await readTextFile(CONFIG_FILE_NAME, { baseDir: BaseDirectory.AppConfig }));
        if (n !== null) return (currentPort = n);
      }
    } catch {
      // 読めなければサイドカーも既定値で起動している
    }
    return (currentPort = DEFAULT_PORT);
  }
  let stored: string | null = null;
  try {
    stored = localStorage.getItem(LS_KEY);
  } catch {
    stored = null;
  }
  currentPort = (stored !== null ? parsePort(stored) : null) ?? DEFAULT_PORT;
  return currentPort;
}

/** ポートを変える (Tauri はサイドカーが次回起動時に読むファイルにも書く。失敗は例外) */
export async function setPort(n: number): Promise<void> {
  currentPort = n;
  try {
    localStorage.setItem(LS_KEY, String(n));
  } catch {
    // localStorage が使えない環境 (プライベートモードなど) でもメモリ上の値で続ける
  }
  if (isTauri()) {
    const { mkdir, writeTextFile, BaseDirectory } = await import("@tauri-apps/plugin-fs");
    const { appConfigDir } = await import("@tauri-apps/api/path");
    await mkdir(await appConfigDir(), { recursive: true });
    await writeTextFile(CONFIG_FILE_NAME, String(n), { baseDir: BaseDirectory.AppConfig });
  }
}

export function baseUrl(): string {
  return `http://127.0.0.1:${currentPort}`;
}

export function wsUrl(path: string): string {
  return `ws://127.0.0.1:${currentPort}${path}`;
}
