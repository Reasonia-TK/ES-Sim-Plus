// ビューアの状態 (元に戻す対象外): 作図・配置のツール、表示の設定 (配色・範囲・重ね表示)、プロファイル線・
// プローブ。配色と重ね表示の切り替えは使う人の好みとして localStorage に残す。

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { SNAP_KINDS, type SnapKind } from "../cad/snap";
import type { Point } from "../model/project";
import { DEFAULT_COLORMAP, type ColormapKey } from "./colormaps";
import type { ManualRange } from "./fieldScale";

/** 作図の道具 (P7b: 線・3 点の円弧はスケッチに、囲まれた所から領域 (fill))。折れ線・矩形・円は描く先 (drawTarget) に */
export type DrawTool = "select" | "polyline" | "rect" | "circle" | "line" | "arc" | "fill";
/** 折れ線・矩形・円を描く先 (領域かスケッチ) */
export type DrawTarget = "region" | "sketch";
export type PlaceTool = "profile" | "emitter" | "injector" | "collector" | "gasbc" | "eedfbox" | "meshref" | "sheathline";
/** 調べる道具 (文書は変えない) */
export type InspectTool = "probe" | "measure";
export type Tool = DrawTool | PlaceTool | InspectTool;

export const DRAW_TOOLS: DrawTool[] = ["select", "polyline", "rect", "circle", "line", "arc", "fill"];
export const PLACE_TOOLS: PlaceTool[] = ["profile", "emitter", "injector", "collector", "gasbc", "eedfbox", "meshref", "sheathline"];
export const INSPECT_TOOLS: InspectTool[] = ["probe", "measure"];

/** ルーラーの文字の大きさ (v1 と同じ 3 段) */
export const RULER_FONTS = { s: 9, m: 11, l: 14 } as const;
export type RulerFont = keyof typeof RULER_FONTS;

/** 静電場で塗る量 */
/** 静電場の塗る量 (none は背景なし、粒子軌道の背景に) */
export type StaticQuantity = "v" | "e_abs" | "none";

export type OverlayKey =
  | "mesh"
  | "isolines"
  | "vectors"
  | "grid"
  | "rulers"
  | "legend"
  | "emitter"
  | "collectors"
  | "eedf"
  | "sheathLines"
  | "gasBoundaries"
  | "edgeSizes"
  | "amr"
  | "particles"
  | "trajectories";

export const OVERLAY_KEYS: OverlayKey[] = [
  "mesh",
  "isolines",
  "vectors",
  "particles",
  "trajectories",
  "emitter",
  "collectors",
  "eedf",
  "sheathLines",
  "gasBoundaries",
  "edgeSizes",
  "amr",
  "grid",
  "rulers",
  "legend",
];

const DEFAULT_OVERLAYS: Record<OverlayKey, boolean> = {
  mesh: false,
  isolines: false,
  vectors: false,
  grid: true,
  rulers: true,
  legend: true,
  emitter: true,
  collectors: true,
  eedf: true,
  sheathLines: true,
  gasBoundaries: true,
  edgeSizes: true,
  amr: true,
  particles: true,
  trajectories: true,
};

/** スナップの種類の既定 (全てオン) */
const DEFAULT_SNAP_KINDS: Record<SnapKind, boolean> = Object.fromEntries(SNAP_KINDS.map((k) => [k, true])) as Record<SnapKind, boolean>;

interface ViewerState {
  tool: Tool;
  drawTarget: DrawTarget;
  /** スナップ全体のオン・オフ */
  snap: boolean;
  /** スナップの種類ごと (オブジェクトスナップとグリッド) */
  snapKinds: Record<SnapKind, boolean>;
  rulerFont: RulerFont;
  quantity: StaticQuantity;
  colormap: ColormapKey;
  range: ManualRange;
  log: boolean;
  overlays: Record<OverlayKey, boolean>;
  /** プロファイル線 (確定したもの) */
  profile: [Point, Point] | null;
  /** プローブの点 (値は表示のたびに今の場から読む) */
  probe: Point | null;
  /** 全体表示の要求 (増えたらビューアが合わせる) */
  fitSerial: number;
  setTool: (t: Tool) => void;
  setDrawTarget: (t: DrawTarget) => void;
  setSnap: (v: boolean) => void;
  setSnapKind: (k: SnapKind, v: boolean) => void;
  setRulerFont: (f: RulerFont) => void;
  setQuantity: (q: StaticQuantity) => void;
  setColormap: (c: ColormapKey) => void;
  setRange: (r: ManualRange) => void;
  setLog: (v: boolean) => void;
  setOverlay: (k: OverlayKey, v: boolean) => void;
  setProfile: (p: [Point, Point] | null) => void;
  setProbe: (p: Point | null) => void;
  requestFit: () => void;
}

export const useViewer = create<ViewerState>()(
  persist(
    (set) => ({
      tool: "select",
      drawTarget: "region",
      snap: true,
      snapKinds: { ...DEFAULT_SNAP_KINDS },
      rulerFont: "m",
      quantity: "v",
      colormap: DEFAULT_COLORMAP,
      range: { min: null, max: null },
      log: false,
      overlays: { ...DEFAULT_OVERLAYS },
      profile: null,
      probe: null,
      fitSerial: 0,
      // プローブはプローブのツールの間だけ (v1 と同じ)
      setTool: (tool) => set((s) => ({ tool, probe: tool === "probe" ? s.probe : null })),
      setDrawTarget: (drawTarget) => set({ drawTarget }),
      setSnap: (snap) => set({ snap }),
      setSnapKind: (k, v) => set((st) => ({ snapKinds: { ...st.snapKinds, [k]: v } })),
      setRulerFont: (rulerFont) => set({ rulerFont }),
      // 量を変えたら手動の範囲は外す (単位が違う)
      setQuantity: (quantity) => set((s) => (s.quantity === quantity ? {} : { quantity, range: { min: null, max: null }, probe: null })),
      setColormap: (colormap) => set({ colormap }),
      setRange: (range) => set({ range }),
      setLog: (log) => set({ log }),
      setOverlay: (k, v) => set((s) => ({ overlays: { ...s.overlays, [k]: v } })),
      setProfile: (profile) => set({ profile }),
      setProbe: (probe) => set({ probe }),
      requestFit: () => set((s) => ({ fitSerial: s.fitSerial + 1 })),
    }),
    {
      name: "es-sim-ui.viewer",
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({ snap: s.snap, snapKinds: s.snapKinds, drawTarget: s.drawTarget, rulerFont: s.rulerFont, colormap: s.colormap, overlays: s.overlays }),
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<ViewerState>;
        return { ...current, ...p, overlays: { ...DEFAULT_OVERLAYS, ...(p.overlays ?? {}) }, snapKinds: { ...DEFAULT_SNAP_KINDS, ...(p.snapKinds ?? {}) } };
      },
    },
  ),
);
