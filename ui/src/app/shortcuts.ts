// キーボードショートカット。元に戻す/やり直しは入力欄の中では効かせない (文字の編集の取り消しを優先、v1 と同じ)。

import { useEffect } from "react";
import { newDocument, openDocument, saveDocument, saveDocumentAs } from "../io/documents";
import { useDocument } from "../model/documentStore";

export const IS_MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);
export const MOD = IS_MAC ? "⌘" : "Ctrl";

function inTextField(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el) return false;
  return el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable;
}

export function handleShortcut(e: KeyboardEvent): boolean {
  const mod = IS_MAC ? e.metaKey : e.ctrlKey;
  if (!mod || e.altKey) return false;
  const key = e.key.toLowerCase();
  if (key === "n" && !e.shiftKey) void newDocument();
  else if (key === "o" && !e.shiftKey) void openDocument();
  else if (key === "s" && e.shiftKey) void saveDocumentAs();
  else if (key === "s") void saveDocument();
  else if ((key === "z" && !e.shiftKey) || key === "y" || (key === "z" && e.shiftKey)) {
    if (inTextField(e.target)) return false;
    const doc = useDocument.getState();
    if (key === "z" && !e.shiftKey) doc.undo();
    else doc.redo();
  } else return false;
  e.preventDefault();
  return true;
}

export function useShortcuts(): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => void handleShortcut(e);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}
