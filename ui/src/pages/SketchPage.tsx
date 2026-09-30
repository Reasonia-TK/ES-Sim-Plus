// スケッチのページ (P7b): 領域でない線・円弧・円・ポリラインの一覧 (クリックで選ぶ、Shift で足す)、選んだものの形の
// 編集 (端点・中心・半径・中心角・ポリラインの点と閉じる)、閉じた形を領域・ドメインにする、削除。

import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { angleFromBulge, bulgeFromAngle } from "../cad/path";
import { logInfo } from "../app/messages";
import { sketchToDomain, sketchToRegion } from "../model/cadActions";
import { useDocument } from "../model/documentStore";
import type { EdgeRemapReport } from "../model/domainOps";
import type { Point } from "../model/project";
import { pickedSketchIds, useSelection } from "../model/selection";
import {
  deleteSketch,
  editBend,
  editMove,
  editPathOf,
  editRemove,
  editSplit,
  replaceSketch,
  sketchClosedPath,
  sketchFromEditPath,
  sketchOf,
  type NewSketch,
  type SketchEntity,
} from "../model/sketch";
import { usePrefs } from "../prefs/prefs";
import { formatNumber, lengthUnitLabel, toDisplayLength, type LengthUnit } from "../util/format";
import { Field, LengthInput } from "./inputs";
import { useLengthUnitLabel } from "./useLengthUnitLabel";
import { Hint } from "./widgets/common";
import { PathTable, type PathTableOps } from "./widgets/PathTable";
import { TransformPanel } from "./widgets/TransformPanel";
import { Toggle } from "../forms/SchemaField";
import { CommitText } from "./inputs";
import { parseNumber } from "../util/format";

const useUpdate = () => useDocument((s) => s.update);

function fmt(q: Point, unit: LengthUnit): string {
  return `(${formatNumber(toDisplayLength(q[0], unit))}, ${formatNumber(toDisplayLength(q[1], unit))})`;
}

/** 一覧の 1 行の説明 */
function summary(e: SketchEntity, unit: LengthUnit, t: TFunction): string {
  const u = lengthUnitLabel(unit);
  switch (e.kind) {
    case "line":
      return `${fmt(e.a, unit)} → ${fmt(e.b, unit)} ${u}`;
    case "arc":
      return `${fmt(e.a, unit)} → ${fmt(e.b, unit)} ${u}, ${formatNumber((angleFromBulge(e.bulge) * 180) / Math.PI)}°`;
    case "circle":
      return `${fmt(e.center, unit)}, r = ${formatNumber(toDisplayLength(e.r, unit))} ${u}`;
    case "polyline":
      return t(e.closed ? "sketchPage.pointsClosed" : "sketchPage.points", { n: e.points.length });
  }
}

function PointField({ label, value, onCommit }: { label: string; value: Point; onCommit: (q: Point) => void }) {
  const u = useLengthUnitLabel();
  return (
    <Field label={label} unit={u}>
      {(id, onError) => (
        <div className="point-input">
          <LengthInput id={id} aria-label={`${label} x`} onError={onError} value={value[0]} onCommit={(x) => onCommit([x, value[1]])} />
          <LengthInput aria-label={`${label} y`} onError={onError} value={value[1]} onCommit={(y) => onCommit([value[0], y])} />
        </div>
      )}
    </Field>
  );
}

function SketchEditor({ e }: { e: SketchEntity }) {
  const { t } = useTranslation();
  const update = useUpdate();
  const u = useLengthUnitLabel();
  const set = (next: NewSketch) => update(t("sketchPage.edit"), (d) => replaceSketch(d, e.id, next));
  const report = (r: EdgeRemapReport | null) => {
    const n = r ? r.boundariesRemoved + r.periodicRemoved : 0;
    if (n > 0) logInfo(t("msg.source.app"), t("settings.bcRemoved", { n }));
  };
  const closed = sketchClosedPath(e) !== null;
  const path = editPathOf(e);
  const ops: PathTableOps | null = path
    ? {
        setVertex: (i, q) => set(sketchFromEditPath(editMove(path, i, q))),
        setBulge: (i, b) => set(sketchFromEditPath(editBend(path, i, b))),
        split: (i) => set(sketchFromEditPath(editSplit(path, i))),
        removeVertex: (i) => {
          const next = editRemove(path, i);
          if (next) set(sketchFromEditPath(next));
        },
      }
    : null;
  return (
    <div className="subsection">
      <div className="subsection-title">
        {e.id} · {t(`sketchPage.kind.${e.kind}`)}
      </div>
      {e.kind === "line" && (
        <>
          <PointField label={t("sketchPage.start")} value={e.a} onCommit={(a) => set({ ...e, a })} />
          <PointField label={t("sketchPage.end")} value={e.b} onCommit={(b) => set({ ...e, b })} />
        </>
      )}
      {e.kind === "arc" && (
        <>
          <PointField label={t("sketchPage.start")} value={e.a} onCommit={(a) => set({ ...e, a })} />
          <PointField label={t("sketchPage.end")} value={e.b} onCommit={(b) => set({ ...e, b })} />
          <Field label={t("sketchPage.angle")} unit="°" hint={t("widgets.arcAngleHint")}>
            {(id, onError) => (
              <CommitText
                id={id}
                inputMode="decimal"
                onError={onError}
                value={formatNumber((angleFromBulge(e.bulge) * 180) / Math.PI)}
                validate={(s) => {
                  const v = parseNumber(s);
                  return v !== null && v !== 0 && Math.abs(v) < 360 ? null : t("sketchPage.angleRange");
                }}
                onCommit={(s) => {
                  const v = parseNumber(s);
                  if (v !== null && v !== 0 && Math.abs(v) < 360) set({ ...e, bulge: bulgeFromAngle((v * Math.PI) / 180) });
                }}
              />
            )}
          </Field>
        </>
      )}
      {e.kind === "circle" && (
        <>
          <PointField label={t("sketchPage.center")} value={e.center} onCommit={(center) => set({ ...e, center })} />
          <Field label={t("sketchPage.radius")} unit={u}>
            {(id, onError) => <LengthInput id={id} onError={onError} value={e.r} min={0} exclusive onCommit={(r) => set({ ...e, r })} />}
          </Field>
        </>
      )}
      {e.kind === "polyline" && path && ops && (
        <>
          <div className="display-toggle">
            <Toggle checked={e.closed} onChange={(v) => set({ ...e, closed: v })} label={t("sketchPage.closed")} disabled={!e.closed && e.points.length < 3} />
          </div>
          <PathTable path={{ polygon: e.points, bulges: e.bulges ?? null }} label={t("tree.sketch")} ops={ops} closed={e.closed} />
        </>
      )}
      <div className="button-row">
        <button
          type="button"
          className="button"
          disabled={!closed}
          title={closed ? undefined : t("sketchPage.needClosed")}
          onClick={() => {
            let rid: string | null = null;
            update(t("cad.toRegion"), (d) => void (rid = sketchToRegion(d, e.id)));
            if (rid) useSelection.getState().selectRegion(rid);
          }}
        >
          {t("cad.toRegion")}
        </button>
        <button
          type="button"
          className="button"
          disabled={!closed}
          title={closed ? t("sketchPage.toDomainHint") : t("sketchPage.needClosed")}
          onClick={() => {
            let r: EdgeRemapReport | null = null;
            update(t("cad.toDomain"), (d) => void (r = sketchToDomain(d, e.id)));
            report(r);
            useSelection.getState().select("domain");
          }}
        >
          {t("cad.toDomain")}
        </button>
        <button type="button" className="button danger" onClick={() => update(t("cad.deleteItems"), (d) => deleteSketch(d, [e.id]))}>
          {t("tree.delete")}
        </button>
      </div>
    </div>
  );
}

export function SketchPage() {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const picked = useSelection((s) => s.picked);
  const pick = useSelection((s) => s.pick);
  const unit = usePrefs((s) => s.lengthUnit);
  const update = useUpdate();
  const list = sketchOf(project);
  const ids = pickedSketchIds(picked);
  const cur = list.find((e) => e.id === ids[ids.length - 1]) ?? null;
  return (
    <>
      <p className="muted">{t("sketchPage.count", { n: list.length })}</p>
      {list.length > 0 && (
        <table className="table">
          <tbody>
            {list.map((e) => (
              <tr key={e.id} className={`clickable${ids.includes(e.id) ? " selected" : ""}`} onClick={(ev) => pick([{ kind: "sketch", id: e.id }], ev.shiftKey ? "toggle" : "replace")}>
                <td>{e.id}</td>
                <td>{t(`sketchPage.kind.${e.kind}`)}</td>
                <td className="muted">{summary(e, unit, t)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {ids.length > 1 && (
        <div className="button-row">
          <span className="muted">{t("sketchPage.multi", { n: ids.length })}</span>
          <button type="button" className="button danger" onClick={() => update(t("cad.deleteItems"), (d) => deleteSketch(d, ids))}>
            {t("tree.delete")}
          </button>
        </div>
      )}
      {cur && <SketchEditor key={cur.id} e={cur} />}
      <TransformPanel />
      <Hint>{t("sketchPage.hint")}</Hint>
    </>
  );
}
