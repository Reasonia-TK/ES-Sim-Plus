// 選択状態 (元に戻す対象外)。ツリー・設定・グラフィックスで共有する。

import { create } from "zustand";
import { useDocument } from "./documentStore";
import { edgeIndexOf } from "./project";

/**
 * ツリーのノード ID:
 * "project" / "geometry" / "domain" / "regions" / "region:<id>" / "boundaries" / "edge:<辺の ID>" / "mesh" / "bfield" /
 * "studies" / "study:<kind>" / "results"
 */
export type NodeId = string;

/** キャンバスに描く配置物 (一覧の何番目か) */
export type PlacementKind = "collector" | "eedf" | "sheath" | "gasbc" | "edgeSize";

export interface PlacementRef {
  kind: PlacementKind;
  index: number;
}

interface SelectionState {
  activeNode: NodeId;
  /** 選択中の領域 (ツリー・キャンバス・設定で共有) */
  selectedRegion: string | null;
  /** 選択中の配置物 (キャンバスと設定の一覧で共有) */
  selectedPlacement: PlacementRef | null;
  select: (node: NodeId) => void;
  selectRegion: (id: string | null) => void;
  selectPlacement: (p: PlacementRef | null) => void;
}

export const useSelection = create<SelectionState>()((set) => ({
  activeNode: "domain",
  selectedRegion: null,
  selectedPlacement: null,
  select: (node) =>
    set(() => {
      const m = /^region:(.+)$/.exec(node);
      return m ? { activeNode: node, selectedRegion: m[1], selectedPlacement: null } : { activeNode: node };
    }),
  selectRegion: (id) =>
    set((s) => ({
      selectedRegion: id,
      selectedPlacement: id !== null ? null : s.selectedPlacement,
      activeNode: id !== null ? `region:${id}` : s.activeNode.startsWith("region:") ? "regions" : s.activeNode,
    })),
  selectPlacement: (p) => set((s) => ({ selectedPlacement: p, selectedRegion: p ? null : s.selectedRegion })),
}));

/** 配置物の一覧の場所 */
export const PLACEMENT_PATHS: Record<PlacementKind, [string, string]> = {
  collector: ["pic", "collectors"],
  eedf: ["pic", "eedf_regions"],
  sheath: ["pic", "sheath_lines"],
  gasbc: ["dsmc", "boundaries"],
  edgeSize: ["mesh", "local_edge_sizes"],
};

/** 配置物の設定があるノード */
export const PLACEMENT_NODES: Record<PlacementKind, NodeId> = {
  collector: "study:pic",
  eedf: "study:pic",
  sheath: "study:pic",
  gasbc: "study:dsmc",
  edgeSize: "mesh",
};

// 元に戻す・やり直しなどで無くなった領域・配置物は選択から外す (v1 の ensureSelection)
useDocument.subscribe((s, prev) => {
  if (s.project === prev.project) return;
  const sel = useSelection.getState();
  const patch: Partial<Pick<SelectionState, "activeNode" | "selectedRegion" | "selectedPlacement">> = {};
  if (sel.selectedRegion !== null && !s.project.geometry.regions.some((r) => r.id === sel.selectedRegion)) {
    patch.selectedRegion = null;
    if (sel.activeNode === `region:${sel.selectedRegion}`) patch.activeNode = "regions";
  }
  // 消えた外周の辺 (頂点を消した・元に戻したなど) は境界条件の一覧へ
  if (sel.activeNode.startsWith("edge:") && edgeIndexOf(s.project, sel.activeNode.slice(5)) < 0) patch.activeNode = "boundaries";
  const pl = sel.selectedPlacement;
  if (pl) {
    const [a, b] = PLACEMENT_PATHS[pl.kind];
    const blk = (s.project as unknown as Record<string, Record<string, unknown> | null | undefined>)[a];
    const list = blk?.[b];
    if (!Array.isArray(list) || pl.index >= list.length) patch.selectedPlacement = null;
  }
  if (Object.keys(patch).length) useSelection.setState(patch);
});
