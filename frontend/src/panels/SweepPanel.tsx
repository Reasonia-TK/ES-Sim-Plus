import { useEffect, useMemo, useState } from "react";
import { CommitNumberInput, CommitTextInput } from "../CommitInput";
import { Toggle } from "../Toggle";
import { EDGE_LABELS_RZ, EDGE_LABELS_RZ_X0, EDGE_LABELS_XY } from "./FieldPanel";
import { rfComponents } from "../types";
import type { Project, SweepCaseState, SweepStartedMsg } from "../types";

/**
 * スイープパネル (スタディ「パラメータスイープ」、prompts/79): 1パラメータ×値リストを
 * ケースごとに別プロセスで並列実行する (prompts/78 のバッチ実行と同じ子プロセス実行方式)。
 *
 * project prop には App 側で particles/pic (エミッタ同期済み) を合成したプロジェクトを
 * 渡してもらう (project 単体では pic/dsmc の候補パスや現在値が正しく見えないため)。
 * 対象パラメータ・値リスト・並列数の選択は Undo/Redo 対象外のローカル状態としてこの
 * コンポーネント内で保持する (DSMC/PIC の設定とは独立)。実行中の状態
 * (started/cases/error) は App 側の WS コールバックが更新するため props で受け取る。
 */

const CUSTOM_PATH = "__custom__";

interface Candidate {
  label: string;
  path: string;
}

// domain 外周エッジの表示ラベルを座標系に応じて選ぶ (ProjectTree.tsx と同じ方針)
function edgeLabelsFor(coord: Project["coord"]): readonly string[] {
  if (coord === "rz") return EDGE_LABELS_RZ;
  if (coord === "rz_x0") return EDGE_LABELS_RZ_X0;
  return EDGE_LABELS_XY;
}

// 現在のプロジェクトから対象パラメータの候補一覧を動的生成する (プリセット)。
// backend はパスの意味を解釈しないため、ここで組み立てるパスは実際の JSON 構造
// (voltage_rf が単一オブジェクトか配列か等) に正確に合わせる必要がある
export function buildSweepCandidates(project: Project): Candidate[] {
  const candidates: Candidate[] = [];
  const edgeLabels = edgeLabelsFor(project.coord);

  project.geometry.boundaries.forEach((b, i) => {
    if (b.type !== "dirichlet") return;
    const edgeLabel = b.edges.length > 0 ? edgeLabels[b.edges[0]] ?? `辺${b.edges[0]}` : `境界${i}`;
    candidates.push({ label: `${edgeLabel} 電圧`, path: `geometry.boundaries.${i}.voltage` });
    // 二次電子放出係数 γ。未設定 (キー不在) でも App 側の runSweepStart が送信前に
    // 現在値 (無ければ 0) で終端キーを実体化するため、常に候補に出して良い
    candidates.push({ label: `${edgeLabel} 二次電子放出γ`, path: `geometry.boundaries.${i}.see_gamma` });
    const rf = b.voltage_rf;
    if (rf) {
      const isArray = Array.isArray(rf);
      const comps = rfComponents(rf);
      comps.forEach((_, j) => {
        const prefix = isArray
          ? `geometry.boundaries.${i}.voltage_rf.${j}`
          : `geometry.boundaries.${i}.voltage_rf`;
        const suffix = comps.length > 1 ? ` (${j + 1})` : "";
        candidates.push({ label: `${edgeLabel} RF振幅${suffix}`, path: `${prefix}.amplitude` });
        candidates.push({ label: `${edgeLabel} RF周波数${suffix}`, path: `${prefix}.freq_hz` });
      });
    }
  });

  project.geometry.regions.forEach((r, i) => {
    if (r.type !== "conductor") return;
    candidates.push({ label: `領域${r.id} 電圧`, path: `geometry.regions.${i}.voltage` });
    candidates.push({ label: `領域${r.id} 二次電子放出γ`, path: `geometry.regions.${i}.see_gamma` });
  });

  if (project.b_field) {
    candidates.push({ label: "磁場 Bx", path: "b_field.bx" });
    candidates.push({ label: "磁場 By", path: "b_field.by" });
    candidates.push({ label: "磁場 Bz", path: "b_field.bz" });
  }

  if (project.dsmc) {
    candidates.push({ label: "DSMC 初期充填圧", path: "dsmc.init_pressure_pa" });
  }

  if (project.pic) {
    candidates.push({ label: "PIC マクロ粒子数", path: "pic.n_macro" });
    candidates.push({ label: "SEE初期エネルギー [eV]", path: "pic.see_energy_ev" });
    // dt が null (自動推定) の間は終端が数値ではないため set_by_path が必ずエラーになる。
    // まず PIC 設定 (スタディ「PIC-MCC」) で明示的な dt を指定してもらう必要がある
    if (project.pic.dt != null) {
      candidates.push({ label: "PIC dt", path: "pic.dt" });
    }
  }

  return candidates;
}

// ドット区切りパスで project から現在値を読む (プレビュー表示用)。数値でなければ undefined
function getByPath(obj: unknown, path: string): number | undefined {
  let cur: unknown = obj;
  for (const tok of path.split(".")) {
    if (cur == null) return undefined;
    if (Array.isArray(cur)) {
      const idx = Number(tok);
      if (!Number.isInteger(idx)) return undefined;
      cur = cur[idx];
    } else if (typeof cur === "object") {
      cur = (cur as Record<string, unknown>)[tok];
    } else {
      return undefined;
    }
  }
  return typeof cur === "number" ? cur : undefined;
}

// "50,100,1.5e2" のようなカンマ区切りテキストを値配列にパースする (不正な項目は無視)
function parseListValues(text: string): number[] {
  return text
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s !== "")
    .map(Number)
    .filter((n) => Number.isFinite(n));
}

// 開始・終了・点数から等分値リストを作る (log=true で対数等分。開始・終了とも >0 が必要)
function computeRangeValues(start: number, end: number, count: number, log: boolean): number[] {
  const n = Math.max(1, Math.round(count));
  if (n === 1) return [start];
  if (log) {
    if (!(start > 0) || !(end > 0)) return [];
    const ls = Math.log(start);
    const le = Math.log(end);
    return Array.from({ length: n }, (_, i) => Math.exp(ls + ((le - ls) * i) / (n - 1)));
  }
  return Array.from({ length: n }, (_, i) => start + ((end - start) * i) / (n - 1));
}

const STATUS_LABELS: Record<SweepCaseState["status"], string> = {
  pending: "待機",
  running: "実行中",
  done: "✓完了",
  error: "エラー",
};

const STATUS_BADGE_KIND: Record<SweepCaseState["status"], string> = {
  pending: "idle",
  running: "busy",
  done: "done",
  error: "error",
};

interface Props {
  // App 側で particles/pic (エミッタ同期済み) を合成済みのプロジェクト
  project: Project;
  canRun: boolean;
  running: boolean;
  onStart: (paramPath: string, values: number[], parallel: number) => void;
  onStop: () => void;
  started: SweepStartedMsg | null;
  cases: SweepCaseState[];
  error: string | null;
  // 直近読込したケース番号 (行のハイライト用。読込前/読込対象がない場合は null)
  loadedCaseIndex: number | null;
  onLoadCase: (index: number) => void;
}

export default function SweepPanel({
  project,
  canRun,
  running,
  onStart,
  onStop,
  started,
  cases,
  error,
  loadedCaseIndex,
  onLoadCase,
}: Props) {
  const candidates = useMemo(() => buildSweepCandidates(project), [project]);
  const [selectedPath, setSelectedPath] = useState<string>(candidates[0]?.path ?? CUSTOM_PATH);
  const [customPath, setCustomPath] = useState("");
  const [valuesMode, setValuesMode] = useState<"list" | "range">("list");
  const [listText, setListText] = useState("");
  const [rangeStart, setRangeStart] = useState(0);
  const [rangeEnd, setRangeEnd] = useState(100);
  const [rangeCount, setRangeCount] = useState(5);
  const [rangeLog, setRangeLog] = useState(false);
  const [parallel, setParallel] = useState(1);

  // 選択中の候補が project の変更で消えた場合 (領域削除など) は先頭候補へフォールバックする
  useEffect(() => {
    if (selectedPath === CUSTOM_PATH) return;
    if (!candidates.some((c) => c.path === selectedPath)) {
      setSelectedPath(candidates[0]?.path ?? CUSTOM_PATH);
    }
  }, [candidates, selectedPath]);

  const paramPath = selectedPath === CUSTOM_PATH ? customPath.trim() : selectedPath;
  const currentValue = paramPath ? getByPath(project, paramPath) : undefined;
  const values =
    valuesMode === "list" ? parseListValues(listText) : computeRangeValues(rangeStart, rangeEnd, rangeCount, rangeLog);

  const picThreads = project.pic?.threads ?? 1;
  const totalThreads = parallel * picThreads;
  const canStart = canRun && !running && paramPath !== "" && values.length > 0;

  return (
    <>
      <h2>パラメータスイープ</h2>
      <p className="hint">
        1つのパラメータを値リストに沿って変化させ、ケースごとに別プロセスで並列実行します
        (prompts/78 のバッチ実行と同じ子プロセス実行方式)。結果はケース一覧の行をクリックすると
        既存の結果ページへそのまま読み込まれます (Undoで元のプロジェクトに戻せます)。
      </p>

      <h3>対象パラメータ</h3>
      <div className="field">
        <span className="label">プリセット</span>
        <select value={selectedPath} onChange={(e) => setSelectedPath(e.target.value)} disabled={running}>
          {candidates.map((c) => (
            <option key={c.path} value={c.path}>
              {c.label}
            </option>
          ))}
          <option value={CUSTOM_PATH}>カスタムパス...</option>
        </select>
      </div>
      {selectedPath === CUSTOM_PATH && (
        <>
          <div className="field">
            <span className="label">パス</span>
            <CommitTextInput value={customPath} onCommit={setCustomPath} />
          </div>
          <p className="hint">
            ドット区切りでプロジェクトJSON内のフィールドを指定します (配列はインデックス番号)。
            例: geometry.boundaries.1.voltage / pic.n_macro / b_field.bz /
            geometry.regions.0.voltage_rf.0.amplitude
          </p>
        </>
      )}
      <div className="kv">
        <span>現在の値</span>
        <span>{currentValue !== undefined ? currentValue : "- (パスが無効か数値ではありません)"}</span>
      </div>

      <h3>値リスト</h3>
      <div className="field">
        <span className="label">指定方法</span>
        <select
          value={valuesMode}
          onChange={(e) => setValuesMode(e.target.value as "list" | "range")}
          disabled={running}
        >
          <option value="list">リスト (カンマ区切り)</option>
          <option value="range">範囲 (開始・終了・点数)</option>
        </select>
      </div>
      {valuesMode === "list" ? (
        <div className="field">
          <span className="label">値 (カンマ区切り、指数表記可)</span>
          <CommitTextInput value={listText} onCommit={setListText} />
        </div>
      ) : (
        <>
          <div className="field">
            <span className="label">開始</span>
            <CommitNumberInput value={rangeStart} onCommit={setRangeStart} disabled={running} />
          </div>
          <div className="field">
            <span className="label">終了</span>
            <CommitNumberInput value={rangeEnd} onCommit={setRangeEnd} disabled={running} />
          </div>
          <div className="field">
            <span className="label">点数</span>
            <CommitNumberInput
              value={rangeCount}
              onCommit={(v) => setRangeCount(Math.max(1, Math.round(v)))}
              disabled={running}
            />
          </div>
          <Toggle
            label="対数等分 (開始・終了とも正値が必要)"
            checked={rangeLog}
            onChange={setRangeLog}
            disabled={running}
          />
        </>
      )}
      <div className="kv">
        <span>ケース数 ({values.length})</span>
        <span>{values.length > 0 ? values.join(", ") : "(値リストが空です)"}</span>
      </div>

      <h3>並列数</h3>
      <div className="field">
        <span className="label">並列プロセス数</span>
        <CommitNumberInput value={parallel} onCommit={(v) => setParallel(Math.max(1, Math.round(v)))} disabled={running} />
      </div>
      <p className="hint">
        合計スレッド数 = 並列数 × PICスレッド数 ({parallel} × {picThreads} = {totalThreads})。
        CPUコア数を大きく超えるとスレッドの奪い合いで逆に遅くなるので注意してください。
      </p>

      <div className="actions">
        <button onClick={() => onStart(paramPath, values, parallel)} disabled={!canStart}>
          {running ? "実行中..." : "スイープ実行"}
        </button>
        <button className="secondary" onClick={onStop} disabled={!running}>
          中止
        </button>
      </div>

      {error && (
        <>
          <h2>エラー</h2>
          <div className="error">{error}</div>
        </>
      )}

      {started && (
        <>
          <h2>ケース一覧 ({started.param_path})</h2>
          <p className="hint">完了した行をクリックすると、そのケースの結果をプロジェクトへ読込みます。</p>
          <div className="collector-list">
            {cases.map((c, i) => (
              <div
                key={i}
                className={`collector-row ${loadedCaseIndex === i ? "selected" : ""}`}
                onClick={() => (c.status === "done" ? onLoadCase(i) : undefined)}
                style={{ cursor: c.status === "done" ? "pointer" : "default" }}
                title={c.status === "error" ? c.error : undefined}
              >
                <span className="collector-points">
                  #{i}: {c.value}
                  {c.status === "running" && c.step != null && c.nSteps ? ` (${Math.round((100 * c.step) / c.nSteps)}%)` : ""}
                </span>
                <span className={`tree-status tree-status-${STATUS_BADGE_KIND[c.status]}`}>{STATUS_LABELS[c.status]}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </>
  );
}
