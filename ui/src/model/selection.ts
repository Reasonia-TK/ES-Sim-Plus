// 選択状態 (元に戻す対象外)。ツリー・設定・グラフィックスで共有する。

import { create } from "zustand";

/**
 * ツリーのノード ID:
 * "project" / "geometry" / "domain" / "regions" / "region:<id>" / "boundaries" / "edge:<i>" / "mesh" / "bfield" /
 * "studies" / "study:<kind>" / "results"
 */
export type NodeId = string;

interface SelectionState {
  activeNode: NodeId;
  /** 選択中の領域 (ツリー・キャンバス・設定で共有) */
  selectedRegion: string | null;
  select: (node: NodeId) => void;
  selectRegion: (id: string | null) => void;
}

export const useSelection = create<SelectionState>()((set) => ({
  activeNode: "domain",
  selectedRegion: null,
  select: (node) =>
    set(() => {
      const m = /^region:(.+)$/.exec(node);
      return m ? { activeNode: node, selectedRegion: m[1] } : { activeNode: node };
    }),
  selectRegion: (id) =>
    set((s) => ({
      selectedRegion: id,
      activeNode: id !== null ? `region:${id}` : s.activeNode.startsWith("region:") ? "regions" : s.activeNode,
    })),
}));
