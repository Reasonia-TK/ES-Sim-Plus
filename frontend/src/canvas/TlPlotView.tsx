import { useState } from "react";
import { Pic1dLineChart } from "./Plot1dView";
import { EEDF_CHART_COLORS } from "../panels/PicPanel";
import { Toggle } from "../Toggle";
import { formatNumber } from "../CommitInput";
import { arrayMax } from "../mathUtils";
import { mToUnit } from "../units";
import type { LengthUnit } from "../units";
import type { TlResult, TlStartedMsg } from "../types";

/**
 * VHF 定在波 (非線形径方向伝送線路モデル) スタディのキャンバス代替ビュー (prompts/101)。
 * canvas/Plot1dView.tsx (1D PIC/MCC) と同じ設計: CadCanvas の代わりに study-tl/result-tl
 * 選択時に表示する。1D PIC 同様 geometry と無関係な専用ソルバーのため、CAD ツールバーは持たない。
 * 汎用の折れ線チャート部品 (Pic1dLineChart) は Plot1dView からそのまま流用する。
 *
 * - 実行中: 進捗のみ表示 (tl.py はフレーム単位のライブ更新を送らない設計、started/progress/done のみ)。
 * - 結果 (done 後): |V_n(r)| 高調波チャート・λ_eff/短縮率サマリ・電力一様性・p(r) チャート・
 *   プローブ点 V(t) 波形・振幅スペクトル。
 */

interface Props {
  lengthUnit: LengthUnit;
  running: boolean;
  started: TlStartedMsg | null;
  progress: { step: number; nSteps: number } | null;
  result: TlResult | null;
  error: string | null;
}

const PROBE_COLORS = { center: "#7ec8e3", mid: "#f2b880", edge: "#8ee6a9" };

// |V_n(r)| チャート: 既定は基本波〜3次まで表示し、「全表示」トグルで n_harm まで全部出す
// (n_harm は最大40なので既定は絞る。PicPanel の EEDF 色パレットを高調波次数で使い回す)
function HarmonicChart({
  r,
  vAmp,
  nHarm,
  unitLabel,
  logY,
  showAll,
  onToggleAll,
  onToggleLog,
}: {
  r: number[];
  vAmp: number[][]; // (n_harm+1) × n_r
  nHarm: number;
  unitLabel: string;
  logY: boolean;
  showAll: boolean;
  onToggleAll: (v: boolean) => void;
  onToggleLog: (v: boolean) => void;
}) {
  const maxN = showAll ? nHarm : Math.min(3, nHarm);
  const series = Array.from({ length: maxN }, (_, i) => {
    const n = i + 1;
    return {
      label: `n=${n}`,
      values: vAmp[n] ?? [],
      color: EEDF_CHART_COLORS[i % EEDF_CHART_COLORS.length],
    };
  });
  return (
    <>
      <div className="pic1d-row-header">
        <h3>高調波振幅 |V_n(r)|</h3>
        <div style={{ display: "flex", gap: 12 }}>
          <Toggle label="全次数表示" checked={showAll} onChange={onToggleAll} />
          <Toggle label="対数軸" checked={logY} onChange={onToggleLog} />
        </div>
      </div>
      <Pic1dLineChart x={r} series={series} height={160} logY={logY} />
      <p className="hint">横軸: 半径 r [{unitLabel}] (r_feed〜R)。縦軸: |V_n(r)| [V]。</p>
    </>
  );
}

export default function TlPlotView({ lengthUnit, running, started, progress, result, error }: Props) {
  const [showAllHarm, setShowAllHarm] = useState(false);
  const [logHarm, setLogHarm] = useState(false);
  const [logSpectrum, setLogSpectrum] = useState(true);

  const unitLabel = lengthUnit === "mm" ? "mm" : "µm";
  const statusLabel = running ? "ライブ実行中" : result ? "計算結果" : "未実行";

  const rDisplay = result ? result.r.map((v) => mToUnit(v, lengthUnit)) : [];

  return (
    <div className="pic1d-view">
      <div className="tool-toolbar pic1d-toolbar">
        <span className="pic1d-toolbar-title">VHF定在波</span>
        <span className="muted">{statusLabel}</span>
      </div>
      <div className="pic1d-body">
        {error && <div className="error pic1d-error">{error}</div>}

        {running && (
          <>
            <h3>実行中</h3>
            {started && (
              <div className="kv">
                <span>dt / 総ステップ数</span>
                <span>
                  {started.dt.toExponential(3)}s / {started.n_steps}
                </span>
              </div>
            )}
            {progress && (
              <>
                <div className="pic-progress">
                  <div
                    className="pic-progress-bar"
                    style={{ width: `${progress.nSteps > 0 ? Math.min(100, (progress.step / progress.nSteps) * 100) : 0}%` }}
                  />
                </div>
                <div className="kv">
                  <span>進捗</span>
                  <span>{progress.step} / {progress.nSteps}</span>
                </div>
              </>
            )}
            <p className="hint">
              tl.py はステップ単位のフィールドを逐次送りません (started → progress → done のみ)。
              完了すると下の結果表示に切り替わります。
            </p>
          </>
        )}

        {!running && result && (
          <>
            <h3>波長短縮・電力一様性</h3>
            <div className="kv">
              <span>λ_eff (基本波の実効波長)</span>
              <span>
                {result.lambda_eff.lambda_m !== null
                  ? `${formatNumber(mToUnit(result.lambda_eff.lambda_m, lengthUnit))} ${unitLabel}`
                  : "節が検出できず算出不可"}
              </span>
            </div>
            <div className="kv">
              <span>λ0 (真空波長 c/f0)</span>
              <span>{formatNumber(mToUnit(result.lambda_eff.lambda0_m, lengthUnit))} {unitLabel}</span>
            </div>
            <div className="kv">
              <span>短縮率 λ_eff/λ0</span>
              <span>{result.lambda_eff.ratio !== null ? formatNumber(result.lambda_eff.ratio) : "-"}</span>
            </div>
            <div className="kv">
              <span>吸収電力 max/min</span>
              <span>{result.power.max_over_min !== null ? formatNumber(result.power.max_over_min) : "-"}</span>
            </div>
            <div className="kv">
              <span>吸収電力 面積重み std/mean</span>
              <span>
                {result.power.area_weighted_std_over_mean !== null
                  ? formatNumber(result.power.area_weighted_std_over_mean)
                  : "-"}
              </span>
            </div>
            <div className="kv">
              <span>給電電力 ⟨V(r_feed)・I(r_feed)⟩</span>
              <span>{formatNumber(result.feed_power)} W</span>
            </div>

            {result.warnings.length > 0 && (
              <div className="pic-warnings">
                {result.warnings.map((w, i) => (
                  <div key={i}>警告: {w}</div>
                ))}
              </div>
            )}

            <HarmonicChart
              r={rDisplay}
              vAmp={result.harmonics.v}
              nHarm={result.settings.n_harm}
              unitLabel={unitLabel}
              logY={logHarm}
              showAll={showAllHarm}
              onToggleAll={setShowAllHarm}
              onToggleLog={setLogHarm}
            />

            <h3>吸収電力密度 p(r)</h3>
            <Pic1dLineChart
              x={rDisplay}
              series={[{ label: "p(r)", values: result.power.p, color: "#f2b880" }]}
              height={130}
            />
            <p className="hint">
              横軸: 半径 r [{unitLabel}]。縦軸: 時間平均吸収電力密度 p(r)=⟨R_b・J_z²⟩ [W/m^2]。
            </p>

            <h3>プローブ点 V(t) (最終2周期)</h3>
            <Pic1dLineChart
              x={result.v_probe.t}
              series={[
                { label: "中心付近", values: result.v_probe.center, color: PROBE_COLORS.center },
                { label: "R/2", values: result.v_probe.mid, color: PROBE_COLORS.mid },
                { label: "外周", values: result.v_probe.edge, color: PROBE_COLORS.edge },
              ]}
              height={130}
            />
            <p className="hint">横軸: 時刻 t [s]。縦軸: V(t) [V]。</p>

            <div className="pic1d-row-header">
              <h3>プローブ点 振幅スペクトル</h3>
              <Toggle label="対数軸" checked={logSpectrum} onChange={setLogSpectrum} />
            </div>
            <Pic1dLineChart
              x={result.spectrum_probe.freq_hz.map((f) => f / result.settings.freq_hz)}
              series={[
                { label: "中心付近", values: result.spectrum_probe.center, color: PROBE_COLORS.center },
                { label: "R/2", values: result.spectrum_probe.mid, color: PROBE_COLORS.mid },
                { label: "外周", values: result.spectrum_probe.edge, color: PROBE_COLORS.edge },
              ]}
              height={130}
              logY={logSpectrum}
            />
            <p className="hint">
              横軸: f/f0 (0〜{arrayMax(result.spectrum_probe.freq_hz.map((f) => f / result.settings.freq_hz)).toFixed(0)}
              )。縦軸: 振幅 [V]。
            </p>
          </>
        )}

        {!running && !result && !error && (
          <p className="hint">VHF定在波が未実行です。左パネルの「VHF定在波 開始」から実行してください。</p>
        )}
      </div>
    </div>
  );
}
