// 確認ダイアログを Promise で呼ぶための要求の列 (表示は DialogHost)。

import { create } from "zustand";
import type { DxfImport, DxfImportOptions } from "../model/dxfImport";

export type UnsavedChoice = "save" | "discard" | "cancel";

export type DialogRequest =
  | { kind: "unsaved"; name: string; resolve: (c: UnsavedChoice) => void }
  | { kind: "confirm"; title: string; message: string; okLabel?: string; danger?: boolean; resolve: (ok: boolean) => void }
  | { kind: "recovery"; name: string; savedAt: number; resolve: (restore: boolean) => void }
  | { kind: "about"; resolve: () => void }
  | { kind: "port"; resolve: () => void }
  | { kind: "dxfImport"; name: string; result: DxfImport; resolve: (opts: DxfImportOptions | null) => void };

type Pending = DialogRequest extends infer T ? (T extends { resolve: unknown } ? Omit<T, "resolve"> : never) : never;

interface DialogState {
  queue: DialogRequest[];
  close: () => void;
}

export const useDialogs = create<DialogState>()((set) => ({
  queue: [],
  close: () => set((s) => ({ queue: s.queue.slice(1) })),
}));

function request<R>(req: Pending): Promise<R> {
  return new Promise<R>((resolve) => {
    const full = { ...req, resolve } as unknown as DialogRequest;
    useDialogs.setState((s) => ({ queue: [...s.queue, full] }));
  });
}

export const askUnsaved = (name: string) => request<UnsavedChoice>({ kind: "unsaved", name });
export const askConfirm = (title: string, message: string, opts?: { okLabel?: string; danger?: boolean }) =>
  request<boolean>({ kind: "confirm", title, message, ...opts });
export const askRecovery = (name: string, savedAt: number) => request<boolean>({ kind: "recovery", name, savedAt });
export const showAbout = () => request<void>({ kind: "about" });
export const showPortDialog = () => request<void>({ kind: "port" });
/** DXF の読み込みの確認 (単位・レイヤ・閉じた形を領域に)。やめれば null */
export const askDxfImport = (name: string, result: DxfImport) => request<DxfImportOptions | null>({ kind: "dxfImport", name, result });
