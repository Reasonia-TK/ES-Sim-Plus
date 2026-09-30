// 選んだ形の変換を数値で (P7d): 移動 (dx, dy)・回転 (角度)・ミラー (縦・横の軸)・尺度 (倍率)・配列 (矩形: 列 × 行と間隔、
// 円周: 個数と全体の角度)。回転・ミラー・尺度・円周の配列の中心は、選んだものの外接矩形の中心 (変えられる)。
// 移動・回転・ミラー・尺度は「元を残してコピーを作る」(キャンバスの変換の道具と同じ設定) に従う。

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { mirror, rotation, scaling, translation, type Affine } from "../../cad/path";
import { Toggle } from "../../forms/SchemaField";
import { useDocument } from "../../model/documentStore";
import { arrayItems, pickedCenter, transformItems, type ArraySpec } from "../../model/editOps";
import type { Point } from "../../model/project";
import { useSelection, type PickRef } from "../../model/selection";
import { useViewer } from "../../graphics/viewerStore";
import { formatNumber, parseNumber } from "../../util/format";
import { CommitText, Field, LengthInput, Select } from "../inputs";
import { useLengthUnitLabel } from "../useLengthUnitLabel";
import { Hint } from "./common";

function NumberInput({ value, onCommit, label, integer, min }: { value: number; onCommit: (v: number) => void; label: string; integer?: boolean; min?: number }) {
  const { t } = useTranslation();
  const ok = (v: number | null): v is number => v !== null && Number.isFinite(v) && (!integer || Number.isInteger(v)) && (min === undefined || v >= min);
  return (
    <CommitText
      aria-label={label}
      inputMode="decimal"
      value={formatNumber(value)}
      validate={(s) => (ok(parseNumber(s)) ? null : t("input.notNumber"))}
      onCommit={(s) => {
        const v = parseNumber(s);
        if (ok(v)) onCommit(v);
      }}
    />
  );
}

export function TransformPanel() {
  const { t } = useTranslation();
  const picked = useSelection((s) => s.picked);
  const project = useDocument((s) => s.project);
  const copy = useViewer((s) => s.transformCopy);
  const setCopy = useViewer((s) => s.setTransformCopy);
  const u = useLengthUnitLabel();
  const [move, setMove] = useState<Point>([0.001, 0]);
  const [angle, setAngle] = useState(90);
  const [factor, setFactor] = useState(2);
  const [center, setCenter] = useState<Point | null>(null);
  const [arrayKind, setArrayKind] = useState<"rect" | "polar">("rect");
  const [rect, setRect] = useState({ nx: 3, ny: 1, dx: 0.005, dy: 0.005 });
  const [polar, setPolar] = useState({ n: 6, angle: 360 });
  const auto = pickedCenter(project, picked);
  if (picked.length === 0 || !auto) return null;
  const c = center ?? auto;
  const apply = (label: string, m: Affine) => {
    const items: PickRef[] = picked;
    let res: PickRef[] = [];
    useDocument.getState().update(label, (d) => void (res = transformItems(d, items, m, copy)));
    if (res.length) useSelection.getState().pick(res);
  };
  const makeArray = () => {
    const spec: ArraySpec = arrayKind === "rect" ? { kind: "rect", ...rect } : { kind: "polar", n: polar.n, angle: polar.angle, center: c };
    const items: PickRef[] = picked;
    let made: PickRef[] = [];
    useDocument.getState().update(t("transformPanel.array"), (d) => void (made = arrayItems(d, items, spec)));
    if (made.length) useSelection.getState().pick([...items, ...made]);
  };
  return (
    <div className="subsection transform-panel">
      <div className="subsection-title">{t("transformPanel.title", { n: picked.length })}</div>
      <div className="display-toggle">
        <Toggle checked={copy} onChange={setCopy} label={t("viewer.transformCopy")} />
      </div>
      <Field label={t("transformPanel.move")} unit={u}>
        {(id, onError) => (
          <div className="point-input">
            <LengthInput id={id} aria-label={`${t("transformPanel.move")} x`} onError={onError} value={move[0]} onCommit={(x) => setMove([x, move[1]])} />
            <LengthInput aria-label={`${t("transformPanel.move")} y`} onError={onError} value={move[1]} onCommit={(y) => setMove([move[0], y])} />
            <button type="button" className="button small" onClick={() => apply(t("viewer.tool.move"), translation(move[0], move[1]))}>
              {t("transformPanel.apply")}
            </button>
          </div>
        )}
      </Field>
      <Field label={t("transformPanel.center")} unit={u} hint={center ? undefined : t("transformPanel.centerAuto")}>
        {(id, onError) => (
          <div className="point-input">
            <LengthInput id={id} aria-label={`${t("transformPanel.center")} x`} onError={onError} value={c[0]} onCommit={(x) => setCenter([x, c[1]])} />
            <LengthInput aria-label={`${t("transformPanel.center")} y`} onError={onError} value={c[1]} onCommit={(y) => setCenter([c[0], y])} />
            {center && (
              <button type="button" className="button small" title={t("transformPanel.centerAuto")} onClick={() => setCenter(null)}>
                ↺
              </button>
            )}
          </div>
        )}
      </Field>
      <Field label={t("transformPanel.rotate")} unit="°">
        {() => (
          <div className="point-input">
            <NumberInput label={t("transformPanel.rotate")} value={angle} onCommit={setAngle} />
            <button type="button" className="button small" onClick={() => apply(t("viewer.tool.rotate"), rotation((angle * Math.PI) / 180, c))}>
              {t("transformPanel.apply")}
            </button>
          </div>
        )}
      </Field>
      <Field label={t("transformPanel.scale")}>
        {() => (
          <div className="point-input">
            <NumberInput label={t("transformPanel.scale")} value={factor} onCommit={(v) => v > 0 && setFactor(v)} min={1e-12} />
            <button type="button" className="button small" onClick={() => apply(t("viewer.tool.scale"), scaling(factor, c))}>
              {t("transformPanel.apply")}
            </button>
          </div>
        )}
      </Field>
      <div className="button-row">
        <span className="muted">{t("viewer.tool.mirror")}</span>
        <button type="button" className="button small" onClick={() => apply(t("viewer.tool.mirror"), mirror(c, [c[0], c[1] + 1]))}>
          {t("transformPanel.mirrorV")}
        </button>
        <button type="button" className="button small" onClick={() => apply(t("viewer.tool.mirror"), mirror(c, [c[0] + 1, c[1]]))}>
          {t("transformPanel.mirrorH")}
        </button>
      </div>
      <div className="subsection-title">{t("transformPanel.array")}</div>
      <Field label={t("transformPanel.arrayKind")}>
        {(id) => (
          <Select<"rect" | "polar">
            id={id}
            value={arrayKind}
            options={[
              { value: "rect", label: t("transformPanel.arrayRect") },
              { value: "polar", label: t("transformPanel.arrayPolar") },
            ]}
            onChange={setArrayKind}
          />
        )}
      </Field>
      {arrayKind === "rect" ? (
        <>
          <Field label={t("transformPanel.counts")}>
            {() => (
              <div className="point-input">
                <NumberInput label={t("transformPanel.cols")} value={rect.nx} integer min={1} onCommit={(nx) => setRect({ ...rect, nx })} />
                <NumberInput label={t("transformPanel.rows")} value={rect.ny} integer min={1} onCommit={(ny) => setRect({ ...rect, ny })} />
              </div>
            )}
          </Field>
          <Field label={t("transformPanel.spacing")} unit={u}>
            {(id, onError) => (
              <div className="point-input">
                <LengthInput id={id} aria-label={`${t("transformPanel.spacing")} x`} onError={onError} value={rect.dx} onCommit={(dx) => setRect({ ...rect, dx })} />
                <LengthInput aria-label={`${t("transformPanel.spacing")} y`} onError={onError} value={rect.dy} onCommit={(dy) => setRect({ ...rect, dy })} />
              </div>
            )}
          </Field>
        </>
      ) : (
        <>
          <Field label={t("transformPanel.count")}>{() => <NumberInput label={t("transformPanel.count")} value={polar.n} integer min={2} onCommit={(n) => setPolar({ ...polar, n })} />}</Field>
          <Field label={t("transformPanel.totalAngle")} unit="°">
            {() => <NumberInput label={t("transformPanel.totalAngle")} value={polar.angle} onCommit={(a) => setPolar({ ...polar, angle: a })} />}
          </Field>
        </>
      )}
      <div className="button-row">
        <button type="button" className="button" onClick={makeArray}>
          {t("transformPanel.makeArray")}
        </button>
      </div>
      <Hint>{t("transformPanel.hint")}</Hint>
    </div>
  );
}
