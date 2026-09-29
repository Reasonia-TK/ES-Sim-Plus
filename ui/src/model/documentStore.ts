// 開いているプロジェクト文書のストア。編集は Immer のパッチとして履歴に積み、全ての設定を
// 元に戻す/やり直せる (v1 は幾何・メッシュ・DSMC だけだった)。未保存かどうかは「状態 ID」で判定する
// (編集 → 元に戻す で保存時と同じ状態に戻れば未保存の印も消える)。

import { applyPatches, enablePatches, produceWithPatches, type Draft, type Patch } from "immer";
import { create } from "zustand";
import { newProject, type Project } from "./project";

enablePatches();

/** 履歴の上限 (v1 は 100) */
export const HISTORY_LIMIT = 200;

/** 文書の保存先。path は Tauri のファイルパス、handleId はブラウザの FileSystemFileHandle (IndexedDB) */
export interface DocFile {
  name: string;
  path?: string;
  handleId?: string;
}

interface HistoryEntry {
  label: string;
  patches: Patch[];
  inverse: Patch[];
  before: number;
  after: number;
}

interface DocumentState {
  project: Project;
  /** 現在の状態の ID (編集ごとに新しい値、元に戻すと前の値に戻る) */
  stateId: number;
  /** 最後に保存した (または開いた) 状態の ID */
  savedStateId: number;
  past: HistoryEntry[];
  future: HistoryEntry[];
  /** 保存先 (新規・サンプルから作った文書は null) */
  file: DocFile | null;
  /** 保存先が無いときの表示名 (サンプル名など) */
  untitledName: string | null;
  /** 文書を置き換えるたびに増える (ツリーの開閉などを文書ごとに初期化するため) */
  docSerial: number;

  /** 文書を編集する (1 回の呼び出しが履歴の 1 件)。変更が無ければ何もしない */
  update: (label: string, recipe: (draft: Draft<Project>) => void) => void;
  undo: () => void;
  redo: () => void;
  /** 別の文書に置き換える (履歴を消す)。dirty=true なら未保存の状態で開く (自動保存からの復元など) */
  replace: (project: Project, file: DocFile | null, opts?: { untitledName?: string | null; dirty?: boolean }) => void;
  /** 保存した (保存先を更新し、現在の状態を保存済みにする) */
  markSaved: (file: DocFile) => void;
}

let idCounter = 0;
const nextId = () => ++idCounter;

export const useDocument = create<DocumentState>()((set, get) => ({
  project: newProject(),
  stateId: 0,
  savedStateId: 0,
  past: [],
  future: [],
  file: null,
  untitledName: null,
  docSerial: 0,

  update: (label, recipe) => {
    const { project, stateId, past } = get();
    const [next, patches, inverse] = produceWithPatches(project, recipe);
    if (patches.length === 0) return;
    const after = nextId();
    const entry: HistoryEntry = { label, patches, inverse, before: stateId, after };
    const trimmed = past.length >= HISTORY_LIMIT ? past.slice(past.length - HISTORY_LIMIT + 1) : past;
    set({ project: next, stateId: after, past: [...trimmed, entry], future: [] });
  },

  undo: () => {
    const { project, past, future } = get();
    const entry = past[past.length - 1];
    if (!entry) return;
    set({
      project: applyPatches(project, entry.inverse),
      stateId: entry.before,
      past: past.slice(0, -1),
      future: [...future, entry],
    });
  },

  redo: () => {
    const { project, past, future } = get();
    const entry = future[future.length - 1];
    if (!entry) return;
    set({
      project: applyPatches(project, entry.patches),
      stateId: entry.after,
      past: [...past, entry],
      future: future.slice(0, -1),
    });
  },

  replace: (project, file, opts) => {
    const id = nextId();
    set((s) => ({
      docSerial: s.docSerial + 1,
      project,
      stateId: id,
      savedStateId: opts?.dirty ? -1 : id,
      past: [],
      future: [],
      file,
      untitledName: opts?.untitledName ?? null,
    }));
  },

  markSaved: (file) => set((s) => ({ file, savedStateId: s.stateId, untitledName: null })),
}));

export function isDirty(s: Pick<DocumentState, "stateId" | "savedStateId">): boolean {
  return s.stateId !== s.savedStateId;
}

export const useIsDirty = () => useDocument((s) => s.stateId !== s.savedStateId);

/** 表示名 (保存先のファイル名、無ければサンプル名か「無題」) */
export function documentName(s: Pick<DocumentState, "file" | "untitledName">, untitled: string): string {
  return s.file?.name ?? s.untitledName ?? untitled;
}

export const useUndoLabel = () => useDocument((s) => s.past[s.past.length - 1]?.label ?? null);
export const useRedoLabel = () => useDocument((s) => s.future[s.future.length - 1]?.label ?? null);
