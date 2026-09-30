// 位相分解の再生 (v1 の周期アニメーション): 再生/一時停止・位相のスライダ・速さ (5/10/20 fps)。
// ビンを送るのはアプリに 1 つ (usePlaybackDriver)。2D ビューとグラフが同じビンを出す (同じ実行の位相分解)。
// 再生の帯が 1 つも出ていなければ止める。

import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { useResultsView } from "../results/resultsView";
import { formatSi } from "../util/format";

export interface Playback {
  bins: number;
  /** 1 周期 [s] (0 は不明) */
  periodS: number;
}

/** 再生中はビンを送る (アプリに 1 つ) */
export function usePlaybackDriver(): void {
  const playing = useResultsView((s) => s.playing);
  const fps = useResultsView((s) => s.fps);
  const bins = useResultsView((s) => s.playBins);
  useEffect(() => {
    if (!playing || bins <= 1) return;
    const id = setInterval(() => {
      const s = useResultsView.getState();
      s.setBin((Math.min(s.bin, bins - 1) + 1) % bins);
    }, 1000 / Math.max(1, fps));
    return () => clearInterval(id);
  }, [playing, fps, bins]);
}

let shown = 0;

export function PlaybackBar({ playback }: { playback: Playback }) {
  const { t } = useTranslation();
  const bin = useResultsView((s) => Math.min(s.bin, playback.bins - 1));
  const playing = useResultsView((s) => s.playing);
  const fps = useResultsView((s) => s.fps);
  const rv = useResultsView.getState;
  useEffect(() => {
    useResultsView.getState().setPlayBins(playback.bins);
  }, [playback.bins]);
  useEffect(() => {
    shown++;
    return () => {
      shown--;
      if (shown === 0 && useResultsView.getState().playing) useResultsView.getState().setPlaying(false);
    };
  }, []);
  const phase = (bin / playback.bins) * 360;
  const tIn = (bin / playback.bins) * playback.periodS;
  return (
    <div className="playback-bar" role="group" aria-label={t("results.playback")}>
      <button type="button" className="tool-button" aria-label={playing ? t("results.pause") : t("results.play")} title={playing ? t("results.pause") : t("results.play")} onClick={() => rv().setPlaying(!playing)}>
        {playing ? "❚❚" : "▶"}
      </button>
      <input
        type="range"
        min={0}
        max={Math.max(0, playback.bins - 1)}
        step={1}
        value={bin}
        aria-label={t("results.phaseBin")}
        onChange={(e) => {
          rv().setPlaying(false);
          rv().setBin(Number(e.target.value));
        }}
      />
      <span className="mono small">
        {t("results.binOf", { bin: bin + 1, bins: playback.bins })} · {phase.toFixed(1)}°{playback.periodS > 0 ? ` · ${formatSi(tIn, "s", 3)}` : ""}
      </span>
      <select className="input fps-select" value={fps} aria-label={t("results.fps")} onChange={(e) => rv().setFps(Number(e.target.value))}>
        {[5, 10, 20].map((f) => (
          <option key={f} value={f}>
            {f} fps
          </option>
        ))}
      </select>
    </div>
  );
}
