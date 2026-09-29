// メッセージ・ログ (下部パネルの「メッセージ」タブとステータスバー)。v1 は警告が各パネル、エラーが
// ステータスバーに散らばっていたので、ここに集める。

import { create } from "zustand";

export type MessageLevel = "info" | "warning" | "error";

export interface Message {
  id: number;
  time: number;
  level: MessageLevel;
  /** 発生元 (ファイル・バックエンド・ソルバー名など) */
  source: string;
  text: string;
}

export const MESSAGE_LIMIT = 500;

interface MessagesState {
  items: Message[];
  push: (level: MessageLevel, source: string, text: string) => void;
  clear: () => void;
}

let seq = 0;

export const useMessages = create<MessagesState>()((set) => ({
  items: [],
  push: (level, source, text) =>
    set((s) => ({
      items: [...s.items.slice(-(MESSAGE_LIMIT - 1)), { id: ++seq, time: Date.now(), level, source, text }],
    })),
  clear: () => set({ items: [] }),
}));

export const logInfo = (source: string, text: string) => useMessages.getState().push("info", source, text);
export const logWarning = (source: string, text: string) => useMessages.getState().push("warning", source, text);
export const logError = (source: string, text: string) => useMessages.getState().push("error", source, text);

export function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
