// UI の設定 (テーマ・言語・長さの単位・最近使ったファイル)。localStorage に保存する。
// ペインの配置は react-resizable-panels の useDefaultLayout が別のキーで保存する。

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import type { LengthUnit } from "../util/format";

export type Theme = "dark" | "light";
export type Language = "ja" | "en";

/** 最近使ったファイル (Tauri はパス、ブラウザは IndexedDB に保存した FileSystemFileHandle の ID) */
export interface RecentFile {
  name: string;
  path?: string;
  handleId?: string;
  openedAt: number;
}

export const RECENT_LIMIT = 10;

interface PrefsState {
  theme: Theme;
  language: Language;
  lengthUnit: LengthUnit;
  /** 設定欄で「詳細設定」の項目も出す */
  showAdvanced: boolean;
  recent: RecentFile[];
  setTheme: (t: Theme) => void;
  setLanguage: (l: Language) => void;
  setLengthUnit: (u: LengthUnit) => void;
  setShowAdvanced: (v: boolean) => void;
  addRecent: (f: Omit<RecentFile, "openedAt">) => void;
  removeRecent: (f: RecentFile) => void;
  clearRecent: () => void;
}

const sameFile = (a: Pick<RecentFile, "path" | "handleId">, b: Pick<RecentFile, "path" | "handleId">) =>
  (a.path !== undefined && a.path === b.path) || (a.handleId !== undefined && a.handleId === b.handleId);

export const usePrefs = create<PrefsState>()(
  persist(
    (set) => ({
      theme: "dark",
      language: "ja",
      lengthUnit: "mm",
      showAdvanced: false,
      recent: [],
      setTheme: (theme) => set({ theme }),
      setLanguage: (language) => set({ language }),
      setLengthUnit: (lengthUnit) => set({ lengthUnit }),
      setShowAdvanced: (showAdvanced) => set({ showAdvanced }),
      addRecent: (f) =>
        set((s) => ({
          recent: [{ ...f, openedAt: Date.now() }, ...s.recent.filter((r) => !sameFile(r, f))].slice(0, RECENT_LIMIT),
        })),
      removeRecent: (f) => set((s) => ({ recent: s.recent.filter((r) => !sameFile(r, f)) })),
      clearRecent: () => set({ recent: [] }),
    }),
    {
      name: "es-sim-ui.prefs",
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({
        theme: s.theme,
        language: s.language,
        lengthUnit: s.lengthUnit,
        showAdvanced: s.showAdvanced,
        recent: s.recent,
      }),
    },
  ),
);
