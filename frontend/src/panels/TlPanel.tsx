import { CommitNullableNumberInput, CommitNumberInput } from "../CommitInput";
import { LENGTH_UNIT_LABEL, mToUnit, unitToM } from "../units";
import type { LengthUnit } from "../units";
import type { Pic1dResult, TlSettings, TlStartedMsg } from "../types";

/**
 * VHF 定在波 (非線形径方向伝送線路モデル) スタディの設定・実行パネル (prompts/101)。
 * geometry/mesh とは無関係な専用の一様格子ソルバー (backend/es_sim/tl.py) を対象とする。
 * Pic1dPanel と同じ役割分担: 設定編集と実行制御のみを持ち、結果の可視化は
 * キャンバス領域 (canvas/TlPlotView.tsx) 側に置く。
 */

interface Props {
  lengthUnit: LengthUnit;
  tl: TlSettings;
  onChange: (next: TlSettings) => void;
  canRun: boolean;
  running: boolean;
  onStart: () => void;
  onStop: () => void;
  started: TlStartedMsg | null;
  // 実行中の進捗 (WebSocket started/progress メッセージ由来。GasPanel の progress と同じ設計)
  progress: { step: number; nSteps: number } | null;
  // 「1D PIC 結果から取込」ボタン用 (直近の pic1d 実行結果。無ければボタンを disabled にする)
  pic1dResult: Pic1dResult | null;
  error: string | null;
}

// 単調増加な (x, y) の線形補間 (範囲外はクランプ)。「1D PIC 結果から取込」の n_i(gap/2) 評価用
// (canvas/Plot1dView.tsx の interpLinear と同じ考え方の小規模な複製。パネル側とキャンバス側で
// モジュールが分かれているため、この小さなヘルパーだけを共有するための依存追加は避ける)
function interpAt(x: number[], y: number[], xTarget: number): number | null {
  const n = x.length;
  if (n === 0 || y.length !== n) return null;
  if (n === 1) return y[0];
  if (xTarget <= x[0]) return y[0];
  if (xTarget >= x[n - 1]) return y[n - 1];
  for (let i = 0; i < n - 1; i++) {
    if (x[i] <= xTarget && xTarget <= x[i + 1]) {
      const frac = (xTarget - x[i]) / (x[i + 1] - x[i]);
      return y[i] + frac * (y[i + 1] - y[i]);
    }
  }
  return y[n - 1];
}

// pic1dResult から tl の n_e_m3/sheath_m を導出する。取れない場合は null (呼び出し側でボタンを無効化する)。
// - n_e ← n_i(gap/2): バルクの代表値としてギャップ中央のイオン密度を使う (準中性バルクでは
//   n_e≈n_i であり、時間平均プロファイルは電極近傍のシースで n_e<n_i となるため、シースの
//   影響を受けにくい中央値を採用する)
// - sheath_m ← 左右シースエッジ (Brinkmann 判定、prompts/97) の平均。片側しか根が求まらない
//   場合はその値のみを使う (対称モデルなので左右の非対称性は本スタディでは扱わない)
function derivePic1dImport(pic1dResult: Pic1dResult): { n_e_m3: number; sheath_m: number } | null {
  const profiles = pic1dResult.profiles;
  if (!profiles) return null;
  const gap = pic1dResult.settings.gap_m;
  const nE = interpAt(profiles.x, profiles.n_i, gap / 2);
  if (nE === null || !(nE > 0)) return null;

  const left = pic1dResult.sheath?.left_s ?? null;
  const right = pic1dResult.sheath?.right_s ?? null;
  const sheathVals = [left, right].filter((v): v is number => v !== null && v > 0);
  if (sheathVals.length === 0) return null;
  const sheath = sheathVals.reduce((a, b) => a + b, 0) / sheathVals.length;

  return { n_e_m3: nE, sheath_m: sheath };
}

export default function TlPanel({
  lengthUnit,
  tl,
  onChange,
  canRun,
  running,
  onStart,
  onStop,
  started,
  progress,
  pic1dResult,
  error,
}: Props) {
  const unitLabel = LENGTH_UNIT_LABEL[lengthUnit];
  const imported = pic1dResult ? derivePic1dImport(pic1dResult) : null;

  const applyPic1dImport = () => {
    if (!imported) return;
    onChange({ ...tl, n_e_m3: imported.n_e_m3, sheath_m: imported.sheath_m });
  };

  return (
    <>
      <h2>VHF定在波: 形状</h2>
      <div className="field">
        <span className="label">電極半径 R [{unitLabel}]</span>
        <CommitNumberInput
          value={mToUnit(tl.radius_m, lengthUnit)}
          onCommit={(v) => onChange({ ...tl, radius_m: Math.max(1e-6, unitToM(v, lengthUnit)) })}
        />
      </div>
      <div className="field">
        <span className="label">ギャップ l [{unitLabel}]</span>
        <CommitNumberInput
          value={mToUnit(tl.gap_m, lengthUnit)}
          onCommit={(v) => onChange({ ...tl, gap_m: Math.max(1e-9, unitToM(v, lengthUnit)) })}
        />
      </div>
      <div className="field">
        <span className="label">平衡シース厚 s0 (片側) [{unitLabel}]</span>
        <CommitNumberInput
          value={mToUnit(tl.sheath_m, lengthUnit)}
          onCommit={(v) => onChange({ ...tl, sheath_m: Math.max(1e-9, unitToM(v, lengthUnit)) })}
        />
      </div>
      <p className="hint">
        r_feed = R/50 (中心給電の実効半径、r→0 での特異性回避) を内側境界とし、r=R の外周は開放端
        (定在波が立つ条件) とします。sheath_m の2倍は gap_m より小さくしてください。
      </p>

      <h2>VHF定在波: プラズマ・シース</h2>
      <div className="field">
        <span className="label">シース則</span>
        <select
          value={tl.sheath_law ?? "child"}
          onChange={(e) => onChange({ ...tl, sheath_law: e.target.value as "child" | "matrix" })}
        >
          <option value="child">Child 則 (推奨)</option>
          <option value="matrix">行列シース</option>
        </select>
      </div>
      <p className="hint">
        Child 則はシース内イオン密度減衰を模擬し、奇数次高調波が定常的に生成されます。
        行列シースは対称放電では線形化し高調波がほぼ出ません (比較・検証用)。
      </p>
      <div className="field">
        <span className="label">バルク電子密度 n_e [m^-3]</span>
        <CommitNumberInput value={tl.n_e_m3} onCommit={(v) => onChange({ ...tl, n_e_m3: Math.max(1e-6, v) })} />
      </div>
      <div className="field">
        <span className="label">シース端密度比 n_s/n_e</span>
        <CommitNumberInput
          value={tl.n_s_ratio}
          onCommit={(v) => onChange({ ...tl, n_s_ratio: Math.min(1, Math.max(1e-6, v)) })}
        />
      </div>
      <div className="field">
        <span className="label">電子運動量衝突周波数 ν_m [Hz]</span>
        <CommitNumberInput value={tl.nu_m_hz} onCommit={(v) => onChange({ ...tl, nu_m_hz: Math.max(0, v) })} />
      </div>
      <p className="hint">
        ν_m ≈ n_g・K_m(T_e) (n_g: 中性粒子密度、K_m: 運動量移行レート係数)。MCC 断面積からの
        自動推定は未対応のため手入力してください。
      </p>

      <div className="actions">
        <button
          className="secondary"
          onClick={applyPic1dImport}
          disabled={!imported}
          title={
            imported
              ? "1D PIC の時間平均プロファイルから n_e (中央の n_i) と sheath_m (左右シースエッジの平均) を取込みます"
              : "1D PIC が未実行、またはプロファイル/シースエッジ (prompts/97) が求まっていません"
          }
        >
          1D PIC 結果から取込 (n_e・sheath_m)
        </button>
      </div>
      {imported && (
        <p className="hint">
          取込値: n_e = {imported.n_e_m3.toExponential(3)} m^-3、sheath_m ={" "}
          {mToUnit(imported.sheath_m, lengthUnit).toPrecision(4)} {unitLabel}
          (バルク代表として n_i(gap/2)、シース厚は左右の平均。理由は取込ボタンのツールチップ参照)
        </p>
      )}

      <h2>VHF定在波: 駆動</h2>
      <div className="field">
        <span className="label">駆動周波数 f0 [Hz]</span>
        <CommitNumberInput value={tl.freq_hz} onCommit={(v) => onChange({ ...tl, freq_hz: Math.max(1e-6, v) })} />
      </div>
      <div className="field">
        <span className="label">駆動振幅 V0 [V]</span>
        <CommitNumberInput value={tl.v0} onCommit={(v) => onChange({ ...tl, v0: Math.max(1e-9, v) })} />
      </div>

      <h2>VHF定在波: 数値設定</h2>
      <div className="field">
        <span className="label">半径方向節点数</span>
        <CommitNumberInput
          value={tl.n_r}
          onCommit={(v) => onChange({ ...tl, n_r: Math.min(20000, Math.max(32, Math.round(v))) })}
        />
      </div>
      <div className="field">
        <span className="label">総周期数</span>
        <CommitNumberInput
          value={tl.n_periods}
          onCommit={(v) => onChange({ ...tl, n_periods: Math.max(8, Math.round(v)) })}
        />
      </div>
      <div className="field">
        <span className="label">FFT窓 周期数</span>
        <CommitNumberInput
          value={tl.n_fft_periods}
          onCommit={(v) => onChange({ ...tl, n_fft_periods: Math.max(4, Math.round(v)) })}
        />
      </div>
      <div className="field">
        <span className="label">返す高調波次数</span>
        <CommitNumberInput
          value={tl.n_harm}
          onCommit={(v) => onChange({ ...tl, n_harm: Math.min(40, Math.max(1, Math.round(v))) })}
        />
      </div>
      <div className="field">
        <span className="label">dt [s] (空欄=CFLから自動)</span>
        <CommitNullableNumberInput value={tl.dt ?? null} onCommit={(v) => onChange({ ...tl, dt: v })} />
      </div>
      <p className="hint">
        FFT窓の周期数は総周期数より小さくしてください。dt を指定する場合は CFL 上限
        (0.5・Δr/v_p) 以下を推奨します (超えると警告が出ます)。
      </p>

      <h2>VHF定在波: 実行</h2>
      <div className="actions">
        <button onClick={onStart} disabled={!canRun || running}>
          {running ? "実行中..." : "VHF定在波 開始"}
        </button>
        <button className="secondary" onClick={onStop} disabled={!running}>
          停止
        </button>
      </div>
      <p className="hint">
        continue (続きから) には対応していません。毎回、立ち上げランプ→FFT窓の定常化を
        フルに実行し直します。
      </p>

      {started && (
        <>
          <div className="kv">
            <span>dt / 総ステップ数</span>
            <span>
              {started.dt.toExponential(3)}s / {started.n_steps}
            </span>
          </div>
          {running && progress && (
            <>
              <div className="pic-progress">
                <div
                  className="pic-progress-bar"
                  style={{
                    width: `${progress.nSteps > 0 ? Math.min(100, (progress.step / progress.nSteps) * 100) : 0}%`,
                  }}
                />
              </div>
              <div className="kv">
                <span>進捗</span>
                <span>{progress.step} / {progress.nSteps}</span>
              </div>
            </>
          )}
        </>
      )}

      {error && (
        <>
          <h2>VHF定在波 エラー</h2>
          <div className="error">{error}</div>
        </>
      )}
    </>
  );
}
