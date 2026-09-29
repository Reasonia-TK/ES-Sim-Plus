// 静電場の実行 (メッシュの作成・求解) と結果の要約 (v1 FieldPanel の「解析実行・解析結果」と同じ項目)。

import { useTranslation } from "react-i18next";
import { useConnection } from "../../backend/connection";
import { Toggle } from "../../forms/SchemaField";
import { useViewer } from "../../graphics/viewerStore";
import { useDocument } from "../../model/documentStore";
import { coordOf } from "../../model/project";
import { isStale, meshKey, solveKey, useStatic } from "../../results/staticResults";
import { edgeLabel } from "../../tree/treeModel";
import { formatNumber } from "../../util/format";
import { Hint } from "./common";

function RunButton({ kind }: { kind: "mesh" | "solve" }) {
  const { t } = useTranslation();
  const connected = useConnection((s) => s.status === "connected");
  const busy = useStatic((s) => s.busy);
  const run = () => void (kind === "mesh" ? useStatic.getState().runMesh() : useStatic.getState().runSolve());
  return (
    <div className="button-row tight">
      <button type="button" className={`button${kind === "solve" ? " primary" : ""}`} disabled={!connected || busy !== null} onClick={run}>
        {busy === kind ? t("static.running") : kind === "mesh" ? t("static.buildMesh") : t("static.compute")}
      </button>
      {busy === kind && (
        <button type="button" className="button" onClick={() => useStatic.getState().cancel()}>
          {t("static.cancel")}
        </button>
      )}
    </div>
  );
}

function NotConnected() {
  const { t } = useTranslation();
  const connected = useConnection((s) => s.status === "connected");
  return connected ? null : <Hint tone="warn">{t("static.notConnected")}</Hint>;
}

/** メッシュのページ: 作成ボタンと節点・要素の数 */
export function MeshBuild() {
  const { t } = useTranslation();
  const mesh = useStatic((s) => s.mesh);
  const solve = useStatic((s) => s.solve);
  const project = useDocument((s) => s.project);
  const m = mesh?.result ?? solve?.result.mesh ?? null;
  const stale = mesh ? isStale(mesh, meshKey, project) : solve ? meshKey(solve.project) !== meshKey(project) : false;
  return (
    <>
      <RunButton kind="mesh" />
      <NotConnected />
      {m && <p className="muted small">{t("static.meshStats", { nodes: m.nodes.length, elements: m.triangles.length })}</p>}
      {m && stale && <Hint tone="warn">{t("static.stale")}</Hint>}
    </>
  );
}

/** 静電場のページ: 計算ボタンと結果の要約 */
export function SolveSummary() {
  const { t } = useTranslation();
  const solve = useStatic((s) => s.solve);
  const project = useDocument((s) => s.project);
  const showMesh = useViewer((s) => s.overlays.mesh);
  const axisym = coordOf(project) !== "xy";
  const r = solve?.result;
  return (
    <>
      <RunButton kind="solve" />
      <NotConnected />
      {r && (
        <div className="button-row tight">
          <button type="button" className="button small" onClick={() => useViewer.getState().setTool("profile")}>
            {t("static.drawProfile")}
          </button>
          <Toggle checked={showMesh} onChange={(v) => useViewer.getState().setOverlay("mesh", v)} label={t("viewer.overlay.mesh")} />
        </div>
      )}
      {!r ? (
        <p className="muted">{t("static.noResult")}</p>
      ) : (
        <>
          <div className="subsection-title">{t("static.result")}</div>
          {isStale(solve, solveKey, project) && <Hint tone="warn">{t("static.stale")}</Hint>}
          <div className="kv">
            <span>{t("static.nodes")}</span>
            <span>{r.mesh.nodes.length}</span>
            <span>{t("static.elements")}</span>
            <span>{r.mesh.triangles.length}</span>
            <span>{t("static.vMinMax")}</span>
            <span>
              {r.v_min.toFixed(1)} / {r.v_max.toFixed(1)} V
            </span>
            <span>{t("static.eMax")}</span>
            <span>{r.e_abs_max.toExponential(2)} V/m</span>
            <span>{t("static.energy")}</span>
            <span>
              {r.energy.toExponential(3)} {axisym ? "J" : "J/m"}
            </span>
            <span>{t("static.capacitance")}</span>
            <span>{r.capacitance != null ? `${r.capacitance.toExponential(3)} ${axisym ? "F" : "F/m"}` : t("static.capacitanceNa")}</span>
            <span>{t("static.elapsed")}</span>
            <span>{formatNumber(Number(solve.elapsedS.toPrecision(3)))} s</span>
          </div>
          {r.charges.length > 0 && (
            <>
              <div className="subsection-title">{t("static.charges")}</div>
              <div className="kv">
                {r.charges.map((c) => {
                  const m = /^edge(\d+)$/.exec(c.label);
                  const name = m ? edgeLabel(solve.project, Number(m[1]), t) : c.label;
                  return [
                    <span key={`${c.label}-n`}>
                      {name} ({formatNumber(c.voltage)} V)
                    </span>,
                    <span key={`${c.label}-q`}>
                      {c.q.toExponential(3)} {axisym ? "C" : "C/m"}
                    </span>,
                  ];
                })}
              </div>
            </>
          )}
        </>
      )}
    </>
  );
}
