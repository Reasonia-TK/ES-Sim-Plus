// グラフの CSV: 見出しと行 (数値はそのまま、空は空欄)。保存は結果の CSV と同じ (Tauri は保存先を選ぶ)。

import { errorText, logError } from "../app/messages";
import { t } from "../i18n";
import { saveTextFile } from "../io/fileAccess";

export type CsvCell = number | string | null | undefined;

export function csvText(header: string[], rows: CsvCell[][]): string {
  const cell = (v: CsvCell) => (v === null || v === undefined || (typeof v === "number" && !Number.isFinite(v)) ? "" : String(v));
  return [header.join(","), ...rows.map((r) => r.map(cell).join(","))].join("\n") + "\n";
}

/** 列 (同じ長さの配列) から行を作る */
export function columns(...cols: ArrayLike<CsvCell>[]): CsvCell[][] {
  const n = Math.max(0, ...cols.map((c) => c.length));
  return Array.from({ length: n }, (_, i) => cols.map((c) => c[i]));
}

export async function saveCsv(name: string, header: string[], rows: CsvCell[][]): Promise<void> {
  try {
    await saveTextFile(name, csvText(header, rows), "csv", "CSV");
  } catch (e) {
    logError(t("msg.source.app"), errorText(e));
  }
}
