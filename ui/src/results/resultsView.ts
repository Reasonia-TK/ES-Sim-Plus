// 結果の表示の状態 (元に戻す対象外): ビューアとグラフに出す実行、2D の表示 (ライブ・時間平均・位相分解と量・
// 対数)、位相分解の再生、シース端、グラフごとの選択 (量・対数・比較など)。粒子・軌道はビューアの重ね表示で出し分ける。

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { useDocument } from "../model/documentStore";

/** 2D で場を塗る種類 */
export type Field2dKind = "pic" | "fluid2d" | "dsmc";
export type Mode2d = "live" | "field" | "cycle";
/** グラフィックス欄のタブ: 2D ビューか結果のグラフ */
export type GraphicsTab = "view" | "charts";

export interface Display2d {
  mode: Mode2d;
  live: string;
  field: string;
  cycle: string;
  logLive: boolean;
  logField: boolean;
  logCycle: boolean;
}

export const DEFAULT_DISPLAY: Record<Field2dKind, Display2d> = {
  pic: { mode: "field", live: "phi", field: "phi", cycle: "phi", logLive: false, logField: false, logCycle: false },
  fluid2d: { mode: "field", live: "phi", field: "phi", cycle: "phi", logLive: false, logField: false, logCycle: false },
  dsmc: { mode: "field", live: "n", field: "n", cycle: "n", logLive: false, logField: false, logCycle: false },
};

interface ResultsViewState {
  /** ビューアとグラフに出す実行 (null は静電場) */
  activeRun: string | null;
  /** この画面から出した実行を自動で表示する */
  follow: boolean;
  graphicsTab: GraphicsTab;
  display: Record<Field2dKind, Display2d>;
  /** 位相分解の再生 */
  bin: number;
  playing: boolean;
  /** 再生で送るビンの数 (出している位相分解のもの) */
  playBins: number;
  fps: number;
  /** シース端: 準中性度 n_e/n_i = α の等値線 */
  sheathContour: boolean;
  sheathAlpha: number;
  /** グラフの表示の選択 (キーはグラフごと。量・対数・比較する実行など) */
  chartPrefs: Record<string, string | number | boolean>;
  setActiveRun: (id: string | null) => void;
  setFollow: (v: boolean) => void;
  setGraphicsTab: (tab: GraphicsTab) => void;
  setDisplay: (kind: Field2dKind, patch: Partial<Display2d>) => void;
  setBin: (bin: number) => void;
  setPlaying: (v: boolean) => void;
  setPlayBins: (n: number) => void;
  setFps: (fps: number) => void;
  setSheath: (patch: { contour?: boolean; alpha?: number }) => void;
  setChartPref: (key: string, value: string | number | boolean) => void;
}

export const useResultsView = create<ResultsViewState>()(
  persist(
    (set) => ({
      activeRun: null,
      follow: true,
      graphicsTab: "view",
      display: structuredClone(DEFAULT_DISPLAY),
      bin: 0,
      playing: false,
      playBins: 0,
      fps: 10,
      sheathContour: true,
      sheathAlpha: 0.5,
      chartPrefs: {},
      setActiveRun: (activeRun) => set((s) => (s.activeRun === activeRun ? {} : { activeRun, playing: false, bin: 0 })),
      setFollow: (follow) => set({ follow }),
      setGraphicsTab: (graphicsTab) => set({ graphicsTab }),
      setDisplay: (kind, patch) => set((s) => ({ display: { ...s.display, [kind]: { ...s.display[kind], ...patch } } })),
      setBin: (bin) => set({ bin }),
      setPlaying: (playing) => set({ playing }),
      setPlayBins: (playBins) => set({ playBins }),
      setFps: (fps) => set({ fps }),
      setSheath: (p) => set((s) => ({ sheathContour: p.contour ?? s.sheathContour, sheathAlpha: p.alpha ?? s.sheathAlpha })),
      setChartPref: (key, value) => set((s) => (s.chartPrefs[key] === value ? {} : { chartPrefs: { ...s.chartPrefs, [key]: value } })),
    }),
    {
      name: "es-sim-ui.results-view",
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({
        follow: s.follow,
        display: s.display,
        fps: s.fps,
        sheathContour: s.sheathContour,
        sheathAlpha: s.sheathAlpha,
        chartPrefs: s.chartPrefs,
      }),
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<ResultsViewState>;
        const display = { ...structuredClone(DEFAULT_DISPLAY) };
        for (const k of Object.keys(display) as Field2dKind[]) display[k] = { ...display[k], ...(p.display?.[k] ?? {}) };
        return { ...current, ...p, display };
      },
    },
  ),
);

// 別の文書にしたら (新規・開く・サンプル・復元) 出している実行を外す (前の文書の結果の上に新しい文書の
// ジオメトリを描かない。結果付きのファイルを開いたときは、読み込んだ実行をこのあとで出す)
useDocument.subscribe((s, prev) => {
  if (s.docSerial !== prev.docSerial) useResultsView.getState().setActiveRun(null);
});

/** グラフの表示の選択 (保存する。型が違う値や未設定は既定値) */
export function useChartPref(key: string, def: boolean): [boolean, (v: boolean) => void];
export function useChartPref(key: string, def: number): [number, (v: number) => void];
export function useChartPref<T extends string = string>(key: string, def: T): [T, (v: T) => void];
export function useChartPref<T extends string | number | boolean>(key: string, def: T): [T, (v: T) => void] {
  const v = useResultsView((s) => s.chartPrefs[key]);
  const value = (v === undefined || typeof v !== typeof def ? def : v) as T;
  return [value, (x: T) => useResultsView.getState().setChartPref(key, x)];
}
