import { useEffect, useState } from "react";
import { CommitNumberInput, CommitTextInput } from "../CommitInput";
import { LENGTH_UNIT_LABEL, mToUnit, unitToM } from "../units";
import type { LengthUnit } from "../units";
import { isAxisymmetric, rfComponents } from "../types";
import type {
  BField,
  CircleShape,
  EdgeBcType,
  MeshResult,
  Point,
  Project,
  Region,
  RegionType,
  SolveResult,
  VoltageRf,
} from "../types";

/**
 * 静電場パネル (3カラムシェルの各種インスペクタページから共用)
 * - ジオメトリ (domain 幅/高さ)・境界条件 (4辺、RF含む)・メッシュ (サイズ)
 * - 領域一覧 + 選択中領域のプロパティ編集
 * - 解析実行 (Mesh/Solve) + 解析結果サマリ (節点数・V範囲・エネルギー等)・メッシュ結果サマリ
 * 編集操作自体は App 側の commitProject 経由で Undo/Redo 履歴に積まれる
 *
 * 3カラムシェルではプロジェクトツリーの選択ノードごとに `sections` で表示ブロックを
 * 絞り込んで使う (例: domain ノードなら sections={["domain"]})。sections 未指定時は
 * 従来通り全ブロックを表示する (後方互換、単体テスト等での利用を壊さないため)。
 */

export type FieldSection = "domain" | "boundary" | "mesh" | "bfield" | "regions" | "solve";

// RF重畳電圧の既定値 (13.56MHz の CCP を想定)
const DEFAULT_VOLTAGE_RF: VoltageRf = { amplitude: 100.0, freq_hz: 13.56e6, phase_deg: 0.0 };
// 2成分目以降を追加する際の既定値 (デュアル周波数の例として低周波側を想定)
const DEFAULT_VOLTAGE_RF_2ND: VoltageRf = { amplitude: 100.0, freq_hz: 2e6, phase_deg: 0.0 };

// RF重畳電圧 (単一/複数成分) の編集UI。成分ごとに振幅/周波数/位相 + 削除ボタン、
// 末尾に成分追加ボタンを表示する。全成分削除で RF 自体を無効化 (undefined) する
function RfComponentsEditor({
  components,
  onChange,
}: {
  components: VoltageRf[];
  onChange: (next: VoltageRf[] | undefined) => void;
}) {
  return (
    <div className="rf-editor">
      {components.map((c, idx) => (
        <div className="edge-rf-row" key={idx}>
          <span className="rf-comp-label">成分{idx + 1}</span>
          <label className="rf-compact-label" title="振幅 [V]">
            A
            <CommitNumberInput
              className="rf-compact"
              value={c.amplitude}
              onCommit={(v) => onChange(components.map((cc, i) => (i === idx ? { ...cc, amplitude: v } : cc)))}
            />
          </label>
          <label className="rf-compact-label" title="周波数 [Hz]">
            f
            <CommitNumberInput
              className="rf-compact"
              value={c.freq_hz}
              onCommit={(v) => onChange(components.map((cc, i) => (i === idx ? { ...cc, freq_hz: v } : cc)))}
            />
          </label>
          <label className="rf-compact-label" title="位相 [deg]">
            φ
            <CommitNumberInput
              className="rf-compact"
              value={c.phase_deg}
              onCommit={(v) => onChange(components.map((cc, i) => (i === idx ? { ...cc, phase_deg: v } : cc)))}
            />
          </label>
          <button
            type="button"
            className="rf-remove-btn"
            title="この成分を削除"
            onClick={() => {
              const next = components.filter((_, i) => i !== idx);
              onChange(next.length > 0 ? next : undefined);
            }}
          >
            ×
          </button>
        </div>
      ))}
      <button
        type="button"
        className="rf-add-btn"
        onClick={() => onChange([...components, components.length === 0 ? DEFAULT_VOLTAGE_RF : DEFAULT_VOLTAGE_RF_2ND])}
      >
        + RF成分を追加
      </button>
    </div>
  );
}

// 矩形 domain の外周エッジ順: 0=下, 1=右, 2=上, 3=左
// (ProjectTree でもエッジ名を揃えて表示するため export する)
export const EDGE_LABELS_XY = ["下 (y=0)", "右 (x=w)", "上 (y=h)", "左 (x=0)"];
// 軸対称 (r-z) モード: x=z (軸方向)・y=r (径方向)。下辺 (y=0) は対称軸 (r=0)
export const EDGE_LABELS_RZ = ["対称軸 (r=0)", "右 (z=L)", "上 (r=R)", "左 (z=0)"];
// 軸対称 (r-z、左辺が軸) モード: x=r (径方向)・y=z (軸方向)。左辺 (x=0) は対称軸 (r=0)
export const EDGE_LABELS_RZ_X0 = ["下 (z=0)", "右 (r=R)", "上 (z=L)", "対称軸 (r=0)"];

interface Props {
  project: Project;
  // 長さの表示・入力単位 (mm/µm)。project 内部は常に m のまま
  lengthUnit: LengthUnit;
  domainW: number;
  domainH: number;
  setDomainSize: (w: number, h: number) => void;
  setCoord: (coord: "xy" | "rz" | "rz_x0") => void;
  edgeState: (
    edgeIndex: number,
  ) => { type: EdgeBcType; voltage: number; voltageRf?: VoltageRf | VoltageRf[]; seeGamma: number };
  setEdgeType: (edgeIndex: number, type: EdgeBcType) => void;
  setEdgeVoltage: (edgeIndex: number, voltage: number) => void;
  setEdgeVoltageRf: (edgeIndex: number, voltage_rf: VoltageRf | VoltageRf[] | undefined) => void;
  setEdgeSeeGamma: (edgeIndex: number, see_gamma: number) => void;
  setMeshSize: (size: number) => void;
  setMeshMode: (mode: "unstructured" | "structured") => void;
  setBField: (patch: Partial<BField>) => void;
  meshResult: MeshResult | null;
  selectedRegionId: string | null;
  onSelectRegion: (id: string) => void;
  selected: Region | null;
  renameRegion: (oldId: string, newId: string) => void;
  setRegionType: (id: string, type: RegionType) => void;
  editRegionShape: (id: string, shape: CircleShape) => void;
  editRegionPolygon: (id: string, polygon: Point[]) => void;
  // 隣接する2領域のマージ (成功時 null、失敗時はエラーメッセージ文字列)
  mergeRegions: (targetId: string, otherId: string) => string | null;
  updateRegion: (id: string, patch: Partial<Region>) => void;
  deleteRegion: (id: string) => void;
  // 領域ごとのローカルメッシュサイズ [m]。null で解除 (全体サイズを使用)
  setRegionLocalSize: (id: string, size: number | null) => void;
  result: SolveResult | null;
  // 表示するセクションの絞り込み (プロジェクトツリーの選択ノードに対応)。
  // 未指定なら従来通り全セクションを表示する (後方互換)
  sections?: FieldSection[];
  // "boundary" セクションで、指定した辺のみ表示する (辺の子ノード選択時)。未指定/null なら全辺表示
  edgeFilter?: number | null;
  // Mesh/Solve ボタン (study-fem インスペクタページ = "solve" セクションでのみ使用)
  runMesh: () => void;
  runSolve: () => void;
  busy: boolean;
  canRun: boolean;
  showMesh: boolean;
  onToggleShowMesh: () => void;
}

export default function FieldPanel({
  project,
  lengthUnit,
  domainW,
  domainH,
  setDomainSize,
  setCoord,
  edgeState,
  setEdgeType,
  setEdgeVoltage,
  setEdgeVoltageRf,
  setEdgeSeeGamma,
  setMeshSize,
  setMeshMode,
  setBField,
  meshResult,
  selectedRegionId,
  onSelectRegion,
  selected,
  renameRegion,
  setRegionType,
  editRegionShape,
  editRegionPolygon,
  mergeRegions,
  updateRegion,
  deleteRegion,
  setRegionLocalSize,
  result,
  sections,
  edgeFilter,
  runMesh,
  runSolve,
  busy,
  canRun,
  showMesh,
  onToggleShowMesh,
}: Props) {
  // セクション表示判定 (sections 未指定なら常に表示 = 後方互換)
  const showSection = (name: FieldSection) => !sections || sections.includes(name);
  // 領域マージ (prompts/63) の選択中マージ先IDとエラーメッセージ (ローカル state)
  const [mergeOtherId, setMergeOtherId] = useState<string>("");
  const [mergeError, setMergeError] = useState<string | null>(null);
  // 選択領域が切り替わったら、別領域向けのエラー表示を持ち越さないようクリアする
  useEffect(() => {
    setMergeError(null);
  }, [selected?.id]);
  const coord = project.coord ?? "xy";
  const isRz = coord === "rz";
  const isRzX0 = coord === "rz_x0";
  const isAxisym = isAxisymmetric(coord);
  const edgeLabels = isRz ? EDGE_LABELS_RZ : isRzX0 ? EDGE_LABELS_RZ_X0 : EDGE_LABELS_XY;
  // 座標系ごとの対称軸エッジ番号 (rz: 下辺=0、rz_x0: 左辺=3。xy は該当なし)
  const axisEdge = isRz ? 0 : isRzX0 ? 3 : null;
  // domain 幅/高さのラベル。rz は x=z(軸方向)・y=r(径方向)、rz_x0 は x=r(径方向)・y=z(軸方向)
  const unitLabel = LENGTH_UNIT_LABEL[lengthUnit];
  const widthLabel = isRz ? `長さ z [${unitLabel}]` : isRzX0 ? `半径 r [${unitLabel}]` : `幅 [${unitLabel}]`;
  const heightLabel = isRz ? `半径 r [${unitLabel}]` : isRzX0 ? `長さ z [${unitLabel}]` : `高さ [${unitLabel}]`;
  const bField = project.b_field ?? { bx: 0, by: 0, bz: 0 };

  return (
    <>
      {showSection("domain") && (
        <>
          <h2>ジオメトリ (domain)</h2>
          <div className="field">
            <span className="label">座標系</span>
            <select value={coord} onChange={(e) => setCoord(e.target.value as "xy" | "rz" | "rz_x0")}>
              <option value="xy">平面 2D</option>
              <option value="rz">軸対称 r-z (下辺が軸)</option>
              <option value="rz_x0">軸対称 r-z (左辺が軸)</option>
            </select>
          </div>
          {isAxisym && (
            <div className="hint">
              軸対称モードでは{isRz ? "下辺" : "左辺"}が対称軸になります。PICは未対応です。
            </div>
          )}
          <div className="field">
            <span className="label">{widthLabel}</span>
            <CommitNumberInput
              value={mToUnit(domainW, lengthUnit)}
              step="0.1"
              onCommit={(w) => setDomainSize(unitToM(w, lengthUnit), domainH)}
            />
          </div>
          <div className="field">
            <span className="label">{heightLabel}</span>
            <CommitNumberInput
              value={mToUnit(domainH, lengthUnit)}
              step="0.1"
              onCommit={(h) => setDomainSize(domainW, unitToM(h, lengthUnit))}
            />
          </div>
        </>
      )}

      {showSection("boundary") && (
        <>
          <h2>境界条件</h2>
          {edgeLabels.map((label, i) => {
            // edgeFilter 指定時 (辺の子ノード選択時) はその辺のみ表示する
            if (edgeFilter != null && edgeFilter !== i) return null;
            const st = edgeState(i);
            // 対称軸そのものとなる辺 (自然境界) は切替不可・固定表示とする
            const isAxisEdge = axisEdge === i;
            const rfList = rfComponents(st.voltageRf);
            return (
              <div className="edge-row" key={i}>
                <span className="edge-label">{label}</span>
                <div className="edge-controls">
                  <select
                    value={st.type}
                    disabled={isAxisEdge}
                    onChange={(e) => setEdgeType(i, e.target.value as EdgeBcType)}
                  >
                    <option value="neumann">なし (Neumann)</option>
                    <option value="dirichlet">Dirichlet</option>
                    <option value="symmetry">対称 (粒子反射)</option>
                    <option value="periodic">周期</option>
                  </select>
                  {!isAxisEdge && st.type === "dirichlet" && (
                    <>
                      <CommitNumberInput value={st.voltage} onCommit={(v) => setEdgeVoltage(i, v)} />
                      <label className="rf-check-inline">
                        <input
                          type="checkbox"
                          checked={rfList.length > 0}
                          onChange={(e) => setEdgeVoltageRf(i, e.target.checked ? [DEFAULT_VOLTAGE_RF] : undefined)}
                        />
                        RF
                      </label>
                      <label className="rf-check-inline" title="二次電子放出係数 γ">
                        γ
                        <CommitNumberInput
                          className="rf-compact"
                          value={st.seeGamma}
                          onCommit={(v) => setEdgeSeeGamma(i, v)}
                        />
                      </label>
                    </>
                  )}
                </div>
                {!isAxisEdge && st.type === "dirichlet" && rfList.length > 0 && (
                  <RfComponentsEditor components={rfList} onChange={(next) => setEdgeVoltageRf(i, next)} />
                )}
              </div>
            );
          })}
        </>
      )}

      {showSection("mesh") && (
        <>
          <h2>メッシュ</h2>
          <div className="field">
            <span className="label">サイズ [{unitLabel}]</span>
            <CommitNumberInput
              value={mToUnit(project.mesh.size, lengthUnit)}
              step="0.01"
              onCommit={(v) => setMeshSize(unitToM(v, lengthUnit))}
            />
          </div>
          <div className="field">
            <span className="label">モード</span>
            <select
              value={project.mesh.mode ?? "unstructured"}
              onChange={(e) => setMeshMode(e.target.value as "unstructured" | "structured")}
            >
              <option value="unstructured">非構造 (三角形)</option>
              <option value="structured">構造格子</option>
            </select>
          </div>
          {(project.mesh.mode ?? "unstructured") === "structured" && (
            <div className="hint">
              構造格子は矩形domainのみ対応。等間隔格子を三角形2分割で切り、
              円・斜め境界は要素中心判定による階段近似になります(局所サイズは無効)。
            </div>
          )}
        </>
      )}

      {showSection("bfield") && (
        <>
          <h2>一様磁場 [T]</h2>
          <div className="hint">
            粒子追跡・PIC のローレンツ力に適用 (静電場ソルブには影響しない)。軸対称モードでは使用不可
          </div>
          {isAxisym && (
            <div className="hint">軸対称モードでは一様磁場は設定できません (∇·B=0 と矛盾するため)。</div>
          )}
          <div className="field">
            <span className="label">Bx</span>
            <CommitNumberInput
              value={bField.bx}
              disabled={isAxisym}
              onCommit={(v) => setBField({ bx: v })}
            />
          </div>
          <div className="field">
            <span className="label">By</span>
            <CommitNumberInput
              value={bField.by}
              disabled={isAxisym}
              onCommit={(v) => setBField({ by: v })}
            />
          </div>
          <div className="field">
            <span className="label">Bz</span>
            <CommitNumberInput
              value={bField.bz}
              disabled={isAxisym}
              onCommit={(v) => setBField({ bz: v })}
            />
          </div>
        </>
      )}

      {showSection("regions") && (
        <>
          <h2>領域一覧 ({project.geometry.regions.length})</h2>
          <div className="region-list">
            {project.geometry.regions.map((r) => (
              <div
                key={r.id}
                className={`region-item ${selectedRegionId === r.id ? "selected" : ""}`}
                onClick={() => onSelectRegion(r.id)}
              >
                <span>{r.id}</span>
                <span className="tag">{r.type}</span>
              </div>
            ))}
            {project.geometry.regions.length === 0 && (
              <div className="muted">(領域なし。ツールバーで作図してください)</div>
            )}
          </div>

          {selected && (
            <div className="region-edit">
              <label>
                ID
                <CommitTextInput
                  value={selected.id}
                  onCommit={(newId) => renameRegion(selected.id, newId)}
                />
              </label>
              <label>
                種別
                <select
                  value={selected.type}
                  onChange={(e) => setRegionType(selected.id, e.target.value as RegionType)}
                >
                  <option value="conductor">電極 (conductor)</option>
                  <option value="dielectric">誘電体 (dielectric)</option>
                  <option value="charge">空間電荷 (charge)</option>
                </select>
              </label>
              {selected.shape && (
                <>
                  <label>
                    中心 X [{unitLabel}]
                    <CommitNumberInput
                      value={mToUnit(selected.shape.center[0], lengthUnit)}
                      step="0.1"
                      onCommit={(x) =>
                        editRegionShape(selected.id, {
                          ...selected.shape!,
                          center: [unitToM(x, lengthUnit), selected.shape!.center[1]],
                        })
                      }
                    />
                  </label>
                  <label>
                    中心 Y [{unitLabel}]
                    <CommitNumberInput
                      value={mToUnit(selected.shape.center[1], lengthUnit)}
                      step="0.1"
                      onCommit={(y) =>
                        editRegionShape(selected.id, {
                          ...selected.shape!,
                          center: [selected.shape!.center[0], unitToM(y, lengthUnit)],
                        })
                      }
                    />
                  </label>
                  <label>
                    半径 [{unitLabel}]
                    <CommitNumberInput
                      value={mToUnit(selected.shape.radius, lengthUnit)}
                      step="0.1"
                      onCommit={(radius) =>
                        editRegionShape(selected.id, { ...selected.shape!, radius: unitToM(radius, lengthUnit) })
                      }
                    />
                  </label>
                </>
              )}
              {selected.type === "conductor" && (
                <>
                  <label>
                    電位 V [V]
                    <CommitNumberInput
                      value={selected.voltage ?? 0}
                      onCommit={(v) => updateRegion(selected.id, { voltage: v })}
                    />
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={rfComponents(selected.voltage_rf).length > 0}
                      onChange={(e) =>
                        updateRegion(selected.id, {
                          voltage_rf: e.target.checked ? [DEFAULT_VOLTAGE_RF] : undefined,
                        })
                      }
                    />
                    RF重畳
                  </label>
                  {rfComponents(selected.voltage_rf).length > 0 && (
                    <RfComponentsEditor
                      components={rfComponents(selected.voltage_rf)}
                      onChange={(next) => updateRegion(selected.id, { voltage_rf: next })}
                    />
                  )}
                  <label>
                    二次電子放出係数 γ
                    <CommitNumberInput
                      value={selected.see_gamma ?? 0}
                      onCommit={(v) => updateRegion(selected.id, { see_gamma: v })}
                    />
                  </label>
                </>
              )}
              {selected.type === "dielectric" && (
                <>
                  <label>
                    比誘電率 εr
                    <CommitNumberInput
                      value={selected.eps_r ?? 1}
                      onCommit={(v) => updateRegion(selected.id, { eps_r: v })}
                    />
                  </label>
                  <label title="イオン衝突時に確率γで二次電子を放出 (PICのみ。0=無効)">
                    二次電子放出係数 γ
                    <CommitNumberInput
                      value={selected.see_gamma ?? 0}
                      onCommit={(v) => updateRegion(selected.id, { see_gamma: v })}
                    />
                  </label>
                </>
              )}
              {selected.type === "charge" && (
                <label>
                  電荷密度 ρ [C/m³]
                  <CommitNumberInput
                    value={selected.rho ?? 0}
                    onCommit={(v) => updateRegion(selected.id, { rho: v })}
                  />
                </label>
              )}
              {selected.polygon && (
                <>
                  <div className="subheading">頂点座標 [{unitLabel}]</div>
                  <table className="vertex-table">
                    <tbody>
                      {selected.polygon.map((pt, i) => (
                        <tr key={i}>
                          <td className="vertex-index">#{i}</td>
                          <td>
                            <CommitNumberInput
                              className="vertex-input"
                              value={mToUnit(pt[0], lengthUnit)}
                              onCommit={(x) =>
                                editRegionPolygon(
                                  selected.id,
                                  selected.polygon!.map((p, j) => (j === i ? ([unitToM(x, lengthUnit), p[1]] as Point) : p)),
                                )
                              }
                            />
                          </td>
                          <td>
                            <CommitNumberInput
                              className="vertex-input"
                              value={mToUnit(pt[1], lengthUnit)}
                              onCommit={(y) =>
                                editRegionPolygon(
                                  selected.id,
                                  selected.polygon!.map((p, j) => (j === i ? ([p[0], unitToM(y, lengthUnit)] as Point) : p)),
                                )
                              }
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>

                  {(() => {
                    // マージ先候補: polygon を持つ他領域 (自分以外)。無ければマージUI自体を隠す
                    const mergeCandidates = project.geometry.regions.filter(
                      (r) => r.polygon && r.id !== selected.id,
                    );
                    if (mergeCandidates.length === 0) return null;
                    const otherId = mergeCandidates.some((r) => r.id === mergeOtherId)
                      ? mergeOtherId
                      : mergeCandidates[0].id;
                    return (
                      <>
                        <div className="subheading">マージ</div>
                        <div className="merge-row">
                          <select
                            value={otherId}
                            onChange={(e) => {
                              setMergeOtherId(e.target.value);
                              setMergeError(null);
                            }}
                          >
                            {mergeCandidates.map((r) => (
                              <option key={r.id} value={r.id}>
                                {r.id}
                              </option>
                            ))}
                          </select>
                          <button
                            type="button"
                            onClick={() => setMergeError(mergeRegions(selected.id, otherId))}
                          >
                            この領域とマージ
                          </button>
                        </div>
                        <div className="hint">※ プロパティ (種別・電圧等) はこの領域側の設定が使われます</div>
                        {mergeError && <div className="error">{mergeError}</div>}
                      </>
                    );
                  })()}
                </>
              )}
              <label title="この領域とその輪郭のメッシュ特性長。0で解除 (全体サイズを使用)。非構造メッシュのみ有効">
                ローカルメッシュサイズ [{unitLabel}] (0=全体)
                <CommitNumberInput
                  value={mToUnit(
                    project.mesh.local_sizes?.find((ls) => ls.region === selected.id)?.size ?? 0,
                    lengthUnit,
                  )}
                  onCommit={(v) => setRegionLocalSize(selected.id, v > 0 ? unitToM(v, lengthUnit) : null)}
                />
              </label>
              {(project.mesh.mode ?? "unstructured") === "structured" &&
                project.mesh.local_sizes?.some((ls) => ls.region === selected.id) && (
                  <div className="hint">構造格子モードではローカルメッシュサイズは無視されます。</div>
                )}
              <button className="danger" onClick={() => deleteRegion(selected.id)}>
                削除
              </button>
            </div>
          )}
        </>
      )}

      {showSection("solve") && (
        <>
          <h2>解析実行</h2>
          <div className="actions">
            <button className="secondary" onClick={runMesh} disabled={busy || !canRun}>
              {busy ? "計算中..." : "Mesh"}
            </button>
            <button onClick={runSolve} disabled={busy || !canRun}>
              {busy ? "計算中..." : "Solve"}
            </button>
            <button className="secondary" onClick={onToggleShowMesh}>
              メッシュ {showMesh ? "非表示" : "表示"}
            </button>
          </div>
          {!canRun && <div className="hint">backend に接続されていません (上部バーのポート設定を確認してください)。</div>}
          {meshResult && (
            <>
              <div className="kv"><span>節点数 (Mesh)</span><span>{meshResult.nodes.length}</span></div>
              <div className="kv"><span>要素数 (Mesh)</span><span>{meshResult.triangles.length}</span></div>
            </>
          )}
          {result ? (
            <>
              <h2>解析結果</h2>
              <div className="kv"><span>節点数</span><span>{result.mesh.nodes.length}</span></div>
              <div className="kv"><span>要素数</span><span>{result.mesh.triangles.length}</span></div>
              <div className="kv"><span>V min/max</span><span>{result.v_min.toFixed(1)} / {result.v_max.toFixed(1)} V</span></div>
              <div className="kv"><span>|E| max</span><span>{result.e_abs_max.toExponential(2)} V/m</span></div>
              <div className="kv"><span>エネルギー</span><span>{result.energy.toExponential(3)} {isAxisym ? "J" : "J/m"}</span></div>
            </>
          ) : (
            !meshResult && <div className="muted">(まだ Mesh/Solve を実行していません)</div>
          )}
        </>
      )}
    </>
  );
}
