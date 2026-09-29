// ペインの配置の保存・初期化 (react-resizable-panels の useDefaultLayout と同じキーを使う)。

import { create } from "zustand";

export const LAYOUT_IDS = ["es-sim-ui.layout.main", "es-sim-ui.layout.columns"] as const;

/** 配置を初期化すると key が変わり、パネル群が作り直される */
export const useLayoutEpoch = create<{ epoch: number }>()(() => ({ epoch: 0 }));

export function resetLayout(): void {
  for (const id of LAYOUT_IDS) {
    try {
      for (const k of Object.keys(localStorage)) if (k.includes(id)) localStorage.removeItem(k);
    } catch {
      // localStorage が使えない
    }
  }
  useLayoutEpoch.setState((s) => ({ epoch: s.epoch + 1 }));
}
