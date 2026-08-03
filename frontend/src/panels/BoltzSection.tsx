import { useEffect, useMemo, useRef, useState } from "react";
import { CommitNullableNumberInput, CommitNumberInput, formatNumber } from "../CommitInput";
import { Toggle } from "../Toggle";
import { saveTextFile } from "../saveFile";
import { arrayMax, arrayMin } from "../mathUtils";
import { Pic1dLineChart } from "../canvas/Plot1dView";
import { computeProcessesHash } from "../boltzHash";
import type { BoltzOpts, BoltzProgressMsg, BoltzStartedMsg, BoltzTable, XsProcess } from "../types";

/**
 * boltzpm (Boltzmann ソルバー) による LMEA 係数テーブル生成 UI (prompts/117-118)。
 * Fluid1dPanel・Fluid2dPanel の両方から共通で使う「電子係数 (boltzpm)」セクション
 * (electron_model/boltz_table の編集・生成実行・鮮度警告・スウォームパラメータ/EEDFチャートを
 * 1つの部品にまとめ、1D/2D で二重管理しないようにする、prompts/118 の指示通り)。
 *
 * /ws/boltz の実行状態自体 (WebSocket・進行中フラグ) は App.tsx が一元管理する
 * (backend/es_sim/server.py の _boltz_lock がモジュール横断で単一のため、フロントも
 * 単一の実行状態で足りる)。このコンポーネントは「今この設定に紐づく生成が進行中か」を
 * props 経由で受け取って表示するだけの役割分担 (他パネルの *Client と同じ設計)。
 */

// 掃引パラメータの既定値 (backend/es_sim/boltz.py DEFAULT_BOLTZ_OPTS と同じキー・既定値)。
// project には含めず、生成の都度指定するローカル state としてのみ扱う
export const DEFAULT_BOLTZ_OPTS: BoltzOpts = {
  en_min_td: 0.5,
  en_max_td: 1000.0,
  n_points: 32,
  eps_max_ev: null,
  d_eps_ev: 0.25,
  n_theta: 16,
};

const COLOR_MUN = "#7ec8e3";
const COLOR_ION = "#c792ea";
const COLOR_EXC = "#f2b880";

interface Props {
  title: string; // 見出しの接頭辞 (例: "流体 (1D)")
  electronProcesses: XsProcess[]; // settings.electron_processes ?? null 埋め済み
  electronModel: "maxwell" | "boltzmann";
  boltzTable: BoltzTable | null;
  onElectronModelChange: (v: "maxwell" | "boltzmann") => void;
  // boltz_table=null かつ electron_model="maxwell" に戻す (呼び出し側で両方まとめて更新する)
  onDeleteTable: () => void;
  // 実行制御 (App.tsx が anyRunning から計算して渡す。BoltzClient 参照)。
  // canStart: health OK かつ「他の何か (他ソルバー・他モジュールの boltz 生成含む) が
  // 実行中でない、またはそれが自分自身の実行である」こと
  canStart: boolean;
  running: boolean; // このモジュール向けの生成が今まさに進行中か
  onStart: (opts: BoltzOpts) => void;
  onStop: () => void;
  started: BoltzStartedMsg | null;
  progress: BoltzProgressMsg | null;
  error: string | null;
}

// ε̄ (mean_energy_ev) をキーにしたスウォームパラメータの対数-対数チャート (prompts/118)。
// x・y ともに対数軸のため、0/負値は「その点で線を切る」形で描画対象から除外する
// (対数の定義域外のため。canvas/Plot1dView.tsx の Pic1dLineChart (線形x・対数y) では
// x 軸まで対数にできず、boltzpm の ε̄ が数桁にわたる場合に見づらいため専用に持つ。
// 1D/2D 共通でこのファイル内だけで使うため、二重管理にはならない)
function BoltzLogLogChart({
  x,
  series,
  height = 140,
}: {
  x: number[];
  series: { label: string; values: number[]; color: string }[];
  height?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = el.getBoundingClientRect();
    el.width = rect.width * dpr;
    el.height = rect.height * dpr;
    const ctx = el.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const padL = 56;
    const padR = 8;
    const padT = 6;
    const padB = 16;
    const plotW = rect.width - padL - padR;
    const plotH = rect.height - padT - padB;

    ctx.strokeStyle = "#363c48";
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, padT, plotW, plotH);

    const xPos: number[] = [];
    for (const v of x) if (v > 0) xPos.push(v);
    let yMax = -Infinity;
    let yMinPos = Infinity;
    for (const s of series) {
      for (const v of s.values) {
        if (v > 0) {
          if (v > yMax) yMax = v;
          if (v < yMinPos) yMinPos = v;
        }
      }
    }
    if (xPos.length < 2 || series.length === 0 || !(yMax > -Infinity)) {
      ctx.fillStyle = "#8a919e";
      ctx.font = "11px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("(表示できるデータがありません)", padL + plotW / 2, padT + plotH / 2);
      return;
    }
    if (!(yMinPos < Infinity) || !(yMinPos < yMax)) yMinPos = yMax * 1e-6;

    const xMin = arrayMin(xPos);
    const xMax = arrayMax(xPos);
    const logXLo = Math.log10(xMin);
    const logXHi = Math.log10(xMax);
    const logXRange = logXHi - logXLo || 1;
    const logYLo = Math.log10(yMinPos);
    const logYHi = Math.log10(yMax);
    const logYRange = logYHi - logYLo || 1;

    const xOf = (v: number) => padL + ((Math.log10(v) - logXLo) / logXRange) * plotW;
    const yOf = (v: number) => padT + plotH - ((Math.log10(v) - logYLo) / logYRange) * plotH;

    for (const s of series) {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < x.length && i < s.values.length; i++) {
        const xv = x[i];
        const yv = s.values[i];
        if (!(xv > 0) || !(yv > 0)) {
          started = false; // 対数軸では非正値の点で線を切る
          continue;
        }
        const px = xOf(xv);
        const py = yOf(yv);
        if (!started) {
          ctx.moveTo(px, py);
          started = true;
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.stroke();
    }

    // 軸目盛り (対数、最小・最大のみの簡素な表示。EedfChart/Pic1dLineChart と同じ流儀)
    ctx.fillStyle = "#8a919e";
    ctx.font = "9px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(xMin.toExponential(1), padL, padT + plotH + 3);
    ctx.textAlign = "right";
    ctx.fillText(xMax.toExponential(1), padL + plotW, padT + plotH + 3);
    ctx.textBaseline = "top";
    ctx.fillText(yMax.toExponential(1), padL - 4, padT);
    ctx.textBaseline = "bottom";
    ctx.fillText(yMinPos.toExponential(1), padL - 4, padT + plotH);
  }, [x, series]);

  return <canvas ref={canvasRef} className="pic1d-chart" style={{ height }} />;
}

export default function BoltzSection({
  title,
  electronProcesses,
  electronModel,
  boltzTable,
  onElectronModelChange,
  onDeleteTable,
  canStart,
  running,
  onStart,
  onStop,
  started,
  progress,
  error,
}: Props) {
  // --- 掃引パラメータ (ローカル state。project には含めない使い捨てパラメータ、prompts/118) ---
  const [opts, setOpts] = useState<BoltzOpts>(DEFAULT_BOLTZ_OPTS);

  // --- 鮮度警告: electron_processes の sha256 と boltz_table.source_hash の比較 (prompts/118) ---
  // 非同期 (Web Crypto) なので useEffect + state で扱う。electron_processes の「内容」が
  // 変わったときだけ再計算したいので、参照ではなく JSON 文字列を依存値にする (呼び出し元の
  // Fluid1dPanel/Fluid2dPanel は `settings.electron_processes ?? []` のように毎レンダー新しい
  // 配列参照を渡してくるため、参照比較だと無駄に再計算してしまう)
  const [currentHash, setCurrentHash] = useState<string | null>(null);
  const processesKey = useMemo(() => JSON.stringify(electronProcesses), [electronProcesses]);
  useEffect(() => {
    let cancelled = false;
    if (electronProcesses.length === 0) {
      // 空 (eduPIC Ar 解析式を既定使用) の場合は boltzHash.ts のコメント通りチェック対象外にする
      setCurrentHash(null);
      return;
    }
    computeProcessesHash(electronProcesses)
      .then((h) => {
        if (!cancelled) setCurrentHash(h);
      })
      .catch(() => {
        if (!cancelled) setCurrentHash(null);
      });
    return () => {
      cancelled = true;
    };
    // processesKey が同じ内容の間は再ハッシュしない
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [processesKey]);

  const isStale = boltzTable != null && currentHash != null && currentHash !== boltzTable.source_hash;

  const pct =
    started && started.n_points > 0
      ? Math.min(100, Math.round(((progress?.i ?? 0) / started.n_points) * 100))
      : 0;

  // --- EEDF ビューア (E/N 選択・EEDF/EEPF 切替・対数トグル・CSV書き出し、prompts/118) ---
  const [eedfIndex, setEedfIndex] = useState(0);
  const [eedfMode, setEedfMode] = useState<"eedf" | "eepf">("eedf");
  const [eedfLogScale, setEedfLogScale] = useState(true);
  const [eedfCsvError, setEedfCsvError] = useState<string | null>(null);
  // プロジェクト読込等でテーブルの点数が変わってもインデックス範囲外にならないようクランプする
  const clampedIndex = boltzTable ? Math.min(eedfIndex, Math.max(0, boltzTable.en_td.length - 1)) : 0;

  const eedfSeries = useMemo(() => {
    if (!boltzTable) return null;
    const eps = boltzTable.eedf_eps_ev;
    const raw = boltzTable.eedf[clampedIndex] ?? [];
    if (eedfMode === "eedf") return { x: eps, y: raw };
    // EEPF = f/√ε (ε=0 は定義できないため除外、PicPanel の EedfChart と同じ扱い)
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < eps.length; i++) {
      if (!(eps[i] > 0)) continue;
      xs.push(eps[i]);
      ys.push(raw[i] / Math.sqrt(eps[i]));
    }
    return { x: xs, y: ys };
  }, [boltzTable, clampedIndex, eedfMode]);

  const downloadEedfCsv = () => {
    if (!boltzTable) return;
    const eps = boltzTable.eedf_eps_ev;
    const raw = boltzTable.eedf[clampedIndex] ?? [];
    const enTd = boltzTable.en_td[clampedIndex];
    const lines = ["E_eV,f_eedf_ev-1,f_eepf_ev-1.5"];
    for (let i = 0; i < eps.length; i++) {
      const e = eps[i];
      const eepf = e > 0 ? raw[i] / Math.sqrt(e) : "";
      lines.push(`${e},${raw[i] ?? ""},${eepf}`);
    }
    setEedfCsvError(null);
    saveTextFile(`boltz_eedf_eepf_${formatNumber(enTd)}Td.csv`, lines.join("\n"), "CSV", ["csv"]).catch((err) => {
      setEedfCsvError(String(err));
    });
  };

  // 生成時 opts の表示 (BoltzTable.opts は Record<string, unknown> なので防御的に読む)
  const genOpts = boltzTable?.opts as Partial<BoltzOpts> & { boltzpm_version?: string } | undefined;

  return (
    <>
      <h2>{title}: 電子係数 (boltzpm)</h2>
      <div className="field">
        <span className="label">電子係数モデル</span>
        <select
          value={electronModel}
          onChange={(e) => onElectronModelChange(e.target.value as "maxwell" | "boltzmann")}
        >
          <option value="maxwell">Maxwell 平均 (既定)</option>
          <option value="boltzmann">Boltzmann (boltzpm)</option>
        </select>
      </div>
      {electronModel === "boltzmann" && !boltzTable && (
        <p className="hint" style={{ color: "#e0b050" }}>
          boltzpm 係数テーブルが未生成です。下の「boltzpm で係数生成」で生成するまで、この
          モジュールは実行できません (backend の validator と同じ制約)。
        </p>
      )}
      <p className="hint">
        E/N (換算電界) を掃引して定常 Boltzmann 解を求め、平均電子エネルギー ε̄ をキーにした
        移動度・反応係数テーブルを作ります (既定設定では低 E/N 点が支配的で数分かかります)。
      </p>

      <div className="actions">
        <button onClick={() => onStart(opts)} disabled={!canStart || running}>
          {running ? "生成中..." : "boltzpm で係数生成"}
        </button>
        <button className="secondary" onClick={onStop} disabled={!running}>
          中断
        </button>
      </div>
      {!canStart && !running && (
        <p className="hint">他の計算 (別ソルバー・別モジュールの boltzpm 生成) が実行中のため開始できません。</p>
      )}

      {(started || running) && (
        <>
          <div className="pic-progress">
            <div className="pic-progress-bar" style={{ width: `${pct}%` }} />
          </div>
          <div className="kv">
            <span>進捗</span>
            <span>{progress?.i ?? 0} / {started?.n_points ?? 0}</span>
          </div>
          <div className="kv">
            <span>現在の E/N</span>
            <span>{progress ? `${formatNumber(progress.en_td)} Td` : "-"}</span>
          </div>
          <div className="kv">
            <span>経過</span>
            <span>{progress ? `${progress.elapsed_s.toFixed(1)} s` : "-"}</span>
          </div>
        </>
      )}
      {error && <div className="error">{error}</div>}

      <h2>{title}: boltzpm 生成パラメータ</h2>
      <div className="field">
        <span className="label">E/N 最小 [Td]</span>
        <CommitNumberInput
          value={opts.en_min_td}
          onCommit={(v) => setOpts({ ...opts, en_min_td: Math.max(1e-6, v) })}
        />
      </div>
      <div className="field">
        <span className="label">E/N 最大 [Td]</span>
        <CommitNumberInput
          value={opts.en_max_td}
          onCommit={(v) => setOpts({ ...opts, en_max_td: Math.max(opts.en_min_td, v) })}
        />
      </div>
      <div className="field">
        <span className="label">点数 (掃引数)</span>
        <CommitNumberInput
          value={opts.n_points}
          onCommit={(v) => setOpts({ ...opts, n_points: Math.max(2, Math.round(v)) })}
        />
      </div>
      <div className="field">
        <span className="label">エネルギーグリッド刻み d_eps [eV]</span>
        <CommitNumberInput
          value={opts.d_eps_ev}
          onCommit={(v) => setOpts({ ...opts, d_eps_ev: Math.max(1e-6, v) })}
        />
      </div>
      <div className="field">
        <span className="label">角度分割数 n_theta</span>
        <CommitNumberInput
          value={opts.n_theta}
          onCommit={(v) => setOpts({ ...opts, n_theta: Math.max(2, Math.round(v)) })}
        />
      </div>
      <div className="field">
        <span className="label">最大エネルギー eps_max [eV] (空欄=自動)</span>
        <CommitNullableNumberInput
          value={opts.eps_max_ev}
          placeholder="自動"
          onCommit={(v) => setOpts({ ...opts, eps_max_ev: v })}
        />
      </div>
      <p className="hint">
        eps_max 自動時は「電離/励起の最大閾値×8」と 40 eV の大きい方を使います (backend 既定と同じ)。
      </p>

      {boltzTable && (
        <>
          <h2>{title}: boltzpm テーブル情報</h2>
          {isStale && (
            <div className="error">
              断面積が変更されています。係数を再生成してください
              (現在の電子プロセスと生成時の断面積が一致しません)。
            </div>
          )}
          {electronProcesses.length === 0 && (
            <p className="hint">
              既定 (eduPIC Ar 解析式) 断面積を使用中のため、鮮度チェックは対象外です
              (既定断面積はソースコードを変更しない限り変わりません)。
            </p>
          )}
          <div className="kv">
            <span>点数</span>
            <span>{boltzTable.mean_energy_ev.length}</span>
          </div>
          <div className="kv">
            <span>ε̄ 範囲 [eV]</span>
            <span>
              {arrayMin(boltzTable.mean_energy_ev).toFixed(3)} 〜 {arrayMax(boltzTable.mean_energy_ev).toFixed(3)}
            </span>
          </div>
          <div className="kv">
            <span>E/N 範囲 [Td]</span>
            <span>
              {formatNumber(arrayMin(boltzTable.en_td))} 〜 {formatNumber(arrayMax(boltzTable.en_td))}
            </span>
          </div>
          {genOpts && (
            <p className="hint">
              生成時パラメータ: en=[{formatNumber(Number(genOpts.en_min_td ?? 0))},{" "}
              {formatNumber(Number(genOpts.en_max_td ?? 0))}] Td、n_points={String(genOpts.n_points ?? "-")}、
              d_eps={String(genOpts.d_eps_ev ?? "-")} eV、n_theta={String(genOpts.n_theta ?? "-")}、
              eps_max={String(genOpts.eps_max_ev ?? "-")} eV
              {genOpts.boltzpm_version ? `、boltzpm ${genOpts.boltzpm_version}` : ""}
            </p>
          )}
          {boltzTable.warnings.length > 0 && (
            <div className="pic-warnings">
              {boltzTable.warnings.map((w, i) => (
                <div key={i}>警告: {w}</div>
              ))}
            </div>
          )}
          <div className="actions">
            <button className="secondary" onClick={onDeleteTable}>
              テーブルを削除
            </button>
          </div>

          <h2>{title}: スウォームパラメータ (boltzpm)</h2>
          <p className="hint">横軸 ε̄ (平均電子エネルギー)・縦軸ともに対数表示です。</p>
          <BoltzLogLogChart
            x={boltzTable.mean_energy_ev}
            series={[{ label: "μ_e・N", values: boltzTable.mobility_n, color: COLOR_MUN }]}
          />
          <div className="pic-chart-legend">
            <span>
              <span className="swatch" style={{ background: COLOR_MUN }} />
              μ_e・N [1/(m・V・s)]
            </span>
          </div>
          <BoltzLogLogChart
            x={boltzTable.mean_energy_ev}
            series={[
              { label: "k_ion", values: boltzTable.k_ion, color: COLOR_ION },
              { label: "k_exc", values: boltzTable.k_exc, color: COLOR_EXC },
            ]}
          />
          <div className="pic-chart-legend">
            <span>
              <span className="swatch" style={{ background: COLOR_ION }} />
              k_ion [m^3/s]
            </span>
            <span>
              <span className="swatch" style={{ background: COLOR_EXC }} />
              k_exc [m^3/s]
            </span>
          </div>

          <h2>{title}: EEDF ビューア (boltzpm)</h2>
          <div className="field">
            <span className="label">E/N [Td]</span>
            <select value={clampedIndex} onChange={(e) => setEedfIndex(Number(e.target.value))}>
              {boltzTable.en_td.map((en, i) => (
                <option key={i} value={i}>
                  {formatNumber(en)} Td (ε̄={boltzTable.mean_energy_ev[i].toFixed(3)} eV)
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <span className="label">表示</span>
            <select value={eedfMode} onChange={(e) => setEedfMode(e.target.value as "eedf" | "eepf")}>
              <option value="eedf">EEDF f(ε) [eV^-1]</option>
              <option value="eepf">EEPF f(ε)/√ε [eV^-1.5]</option>
            </select>
          </div>
          <Toggle label="縦軸対数スケール" checked={eedfLogScale} onChange={setEedfLogScale} />
          {eedfSeries && (
            <Pic1dLineChart
              x={eedfSeries.x}
              series={[{ label: eedfMode, values: eedfSeries.y, color: COLOR_MUN }]}
              logY={eedfLogScale}
            />
          )}
          <div className="actions">
            <button className="secondary" onClick={downloadEedfCsv}>
              CSV保存
            </button>
          </div>
          {eedfCsvError && <div className="error">{eedfCsvError}</div>}
        </>
      )}
    </>
  );
}
