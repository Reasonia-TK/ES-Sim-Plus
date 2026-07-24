import { useEffect, useRef } from "react";
import { CommitNumberInput } from "../CommitInput";
import { Toggle } from "../Toggle";
import type { DirichletBC, FnEmission, Project } from "../types";

/**
 * FN (Fowler–Nordheim) 電界放出セクション。粒子追跡パネル/PICパネルの両方から共用する。
 * - 有効/無効チェックボックス (無効時は該当 fn を undefined/null にする。値は defaultsRef で保持し、
 *   再度有効にしたときに直前の設定を復元する)
 * - 放出面選択 (domain 外周の dirichlet 辺 / conductor 領域のチェックボックス列挙)
 * - mode="trace": φ・β・粒子数 n・初期エネルギーを表示 (粒子追跡専用)
 * - mode="pic": φ・β・初期エネルギー・マクロ重み・乱数シードを表示 (PIC専用。n は使わない)
 */

export const DEFAULT_FN: FnEmission = {
  edges: [],
  regions: [],
  phi_ev: 4.5,
  beta: 1.0,
  n: 200,
  init_energy_ev: 0.1,
  macro_weight: null,
  seed: 0,
};

interface Props {
  project: Project;
  fn: FnEmission | null | undefined;
  onChange: (next: FnEmission | null) => void;
  mode: "trace" | "pic";
}

export default function FnEmissionSection({ project, fn, onChange, mode }: Props) {
  // 有効チェックを一度オフにしても、再度オンにしたときに直前の値を復元できるよう保持する
  const defaultsRef = useRef<FnEmission>(fn ?? DEFAULT_FN);
  useEffect(() => {
    if (fn) defaultsRef.current = fn;
  }, [fn]);

  const update = (patch: Partial<FnEmission>) => {
    if (!fn) return;
    onChange({ ...fn, ...patch });
  };

  const dirichletBoundaries = project.geometry.boundaries.filter(
    (b): b is DirichletBC => b.type === "dirichlet",
  );
  const conductorRegions = project.geometry.regions.filter((r) => r.type === "conductor");

  const toggleEdge = (edge: number, checked: boolean) => {
    if (!fn) return;
    const edges = checked ? [...fn.edges, edge] : fn.edges.filter((e) => e !== edge);
    update({ edges });
  };
  const toggleRegion = (id: string, checked: boolean) => {
    if (!fn) return;
    const regions = checked ? [...fn.regions, id] : fn.regions.filter((r) => r !== id);
    update({ regions });
  };

  return (
    <>
      <h2>FN電界放出</h2>
      <Toggle label="有効" checked={!!fn} onChange={(v) => onChange(v ? defaultsRef.current : null)} />
      {fn && (
        <>
          <p className="hint">FN 使用時はエミッタ・粒子種は使われません (電子固定)</p>
          <div className="field">
            <span className="label">仕事関数 φ [eV]</span>
            <CommitNumberInput value={fn.phi_ev} onCommit={(v) => update({ phi_ev: v })} />
          </div>
          <div className="field">
            <span className="label">電界増倍係数 β</span>
            <CommitNumberInput value={fn.beta} onCommit={(v) => update({ beta: v })} />
          </div>
          {mode === "trace" && (
            <div className="field">
              <span className="label">放出粒子数 n</span>
              <CommitNumberInput
                value={fn.n}
                onCommit={(v) => update({ n: Math.max(1, Math.round(v)) })}
              />
            </div>
          )}
          <div className="field">
            <span className="label">初期エネルギー [eV]</span>
            <CommitNumberInput value={fn.init_energy_ev} onCommit={(v) => update({ init_energy_ev: v })} />
          </div>
          {mode === "pic" && (
            <>
              <div className="field">
                <span className="label">マクロ重み (空欄=初期プラズマと同じ)</span>
                <input
                  type="text"
                  inputMode="decimal"
                  value={fn.macro_weight == null ? "" : String(fn.macro_weight)}
                  placeholder="初期プラズマと同じ"
                  onChange={(e) => {
                    const raw = e.target.value;
                    if (raw.trim() === "") {
                      update({ macro_weight: null });
                      return;
                    }
                    const n = Number(raw);
                    if (Number.isFinite(n)) update({ macro_weight: n });
                  }}
                />
              </div>
              <div className="field">
                <span className="label">乱数シード</span>
                <CommitNumberInput value={fn.seed} onCommit={(v) => update({ seed: Math.round(v) })} />
              </div>
            </>
          )}

          <p className="hint">放出面 (domain 外周の Dirichlet 辺 / conductor 領域を選択、少なくとも一方が必要)</p>
          <div className="fn-surface-list">
            {dirichletBoundaries.length === 0 && conductorRegions.length === 0 && (
              <div className="muted">(Dirichlet 辺・conductor 領域がありません)</div>
            )}
            {dirichletBoundaries.map((b) =>
              b.edges.map((edge) => (
                <div className="fn-surface-item" key={`edge-${edge}`}>
                  <Toggle
                    label={`エッジ ${edge} (${b.voltage} V)`}
                    checked={fn.edges.includes(edge)}
                    onChange={(v) => toggleEdge(edge, v)}
                  />
                </div>
              )),
            )}
            {conductorRegions.map((r) => (
              <div className="fn-surface-item" key={`region-${r.id}`}>
                <Toggle
                  label={`領域 ${r.id} (${r.voltage ?? 0} V)`}
                  checked={fn.regions.includes(r.id)}
                  onChange={(v) => toggleRegion(r.id, v)}
                />
              </div>
            ))}
          </div>
        </>
      )}
    </>
  );
}
