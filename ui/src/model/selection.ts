// 選択状態 (元に戻す対象外)。ツリー・設定・グラフィックスで共有する。
// キャンバスでは領域とスケッチを複数選べる (picked、P7b)。最後に選んだものが主で、設定のページに出す。

import { create } from "zustand";
import { useDocument } from "./documentStore";
import { edgeIndexOf } from "./project";
import { sketchOf } from "./sketch";

/**
 * ツリーのノード ID:
 * "project" / "geometry" / "domain" / "regions" / "region:<id>" / "sketch" / "boundaries" / "edge:<辺の ID>" / "mesh" /
 * "bfield" / "studies" / "study:<kind>" / "results"
 */
export type NodeId = string;

/** キャンバスに描く配置物 (一覧の何番目か) */
export type PlacementKind = "collector" | "eedf" | "sheath" | "gasbc" | "edgeSize";

export interface PlacementRef {
  kind: PlacementKind;
  index: number;
}

/** キャンバスで選んだもの */
export type PickRef = { kind: "region"; id: string } | { kind: "sketch"; id: string };

const samePick = (a: PickRef, b: PickRef) => a.kind === b.kind && a.id === b.id;

interface SelectionState {
  activeNode: NodeId;
  /** 選択中の領域 (ツリー・キャンバス・設定で共有。複数選んでいれば最後に選んだ領域) */
  selectedRegion: string | null;
  /** 選択中の配置物 (キャンバスと設定の一覧で共有) */
  selectedPlacement: PlacementRef | null;
  /** キャンバスで選んだ領域とスケッチ (複数可、最後が主) */
  picked: PickRef[];
  select: (node: NodeId) => void;
  selectRegion: (id: string | null) => void;
  selectPlacement: (p: PlacementRef | null) => void;
  /** キャンバスで選ぶ (replace は置き換え、toggle は選んでいれば外し、いなければ足す) */
  pick: (items: PickRef[], mode?: "replace" | "toggle") => void;
}

export const useSelection = create<SelectionState>()((set) => ({
  activeNode: "domain",
  selectedRegion: null,
  selectedPlacement: null,
  picked: [],
  select: (node) =>
    set(() => {
      const m = /^region:(.+)$/.exec(node);
      if (m) return { activeNode: node, selectedRegion: m[1], selectedPlacement: null, picked: [{ kind: "region", id: m[1] }] };
      // ドメイン・外周の辺を選んだらキャンバスではドメインを編集する (ほかの選択を外す)
      if (node === "domain" || node.startsWith("edge:")) return { activeNode: node, selectedRegion: null, picked: [] };
      return { activeNode: node };
    }),
  selectRegion: (id) =>
    set((s) => ({
      selectedRegion: id,
      selectedPlacement: id !== null ? null : s.selectedPlacement,
      picked: id !== null ? [{ kind: "region", id }] : [],
      activeNode: id !== null ? `region:${id}` : s.activeNode.startsWith("region:") ? "regions" : s.activeNode,
    })),
  selectPlacement: (p) => set((s) => ({ selectedPlacement: p, selectedRegion: p ? null : s.selectedRegion, picked: p ? [] : s.picked })),
  pick: (items, mode = "replace") =>
    set((s) => {
      let picked: PickRef[];
      if (mode === "toggle") {
        picked = s.picked.slice();
        for (const it of items) {
          const k = picked.findIndex((x) => samePick(x, it));
          if (k >= 0) picked.splice(k, 1);
          else picked.push(it);
        }
      } else picked = items.slice();
      const last = picked[picked.length - 1];
      const regions = picked.filter((x) => x.kind === "region");
      let activeNode = s.activeNode;
      if (last) activeNode = last.kind === "region" ? `region:${last.id}` : "sketch";
      else if (s.activeNode.startsWith("region:")) activeNode = "regions";
      return {
        picked,
        selectedRegion: regions.length ? regions[regions.length - 1].id : null,
        selectedPlacement: picked.length ? null : s.selectedPlacement,
        activeNode,
      };
    }),
}));

/** 選んだスケッチの ID (最後が主) */
export function pickedSketchIds(picked: PickRef[]): string[] {
  return picked.filter((x) => x.kind === "sketch").map((x) => x.id);
}

export function pickedRegionIds(picked: PickRef[]): string[] {
  return picked.filter((x) => x.kind === "region").map((x) => x.id);
}

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

// 元に戻す・やり直しなどで無くなった領域・スケッチ・配置物は選択から外す (v1 の ensureSelection)
useDocument.subscribe((s, prev) => {
  if (s.project === prev.project) return;
  const sel = useSelection.getState();
  const patch: Partial<Pick<SelectionState, "activeNode" | "selectedRegion" | "selectedPlacement" | "picked">> = {};
  if (sel.selectedRegion !== null && !s.project.geometry.regions.some((r) => r.id === sel.selectedRegion)) {
    patch.selectedRegion = null;
    if (sel.activeNode === `region:${sel.selectedRegion}`) patch.activeNode = "regions";
  }
  if (sel.picked.length) {
    const regions = new Set(s.project.geometry.regions.map((r) => r.id));
    const sketch = new Set(sketchOf(s.project).map((e) => e.id));
    const kept = sel.picked.filter((x) => (x.kind === "region" ? regions.has(x.id) : sketch.has(x.id)));
    if (kept.length !== sel.picked.length) patch.picked = kept;
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
