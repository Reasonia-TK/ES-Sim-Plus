// 輪郭の頂点の表 (P7): 頂点の座標 (長さは表示単位)、辺ごとの円弧の中心角 [°]、頂点の追加 (辺を中点で分ける。
// 円弧は同じ円の 2 つの円弧に) と削除 (前後の辺を 1 本に)。領域・ドメイン・スケッチのポリライン (開いていれば最後の
// 点から出る辺は無い) で使い、文書の変え方は呼ぶ側が渡す (ドメインは外周の辺の ID と参照の付け替えも一緒に行う)。

import { useTranslation } from "react-i18next";
import { angleFromBulge, bulgeFromAngle, bulgesOf, removeVertex, type PathData } from "../../cad/path";
import { QuantityInput } from "../../forms/SchemaField";
import { useUnitContext } from "../../forms/useField";
import type { Point } from "../../model/project";
import type { FieldInfo } from "../../schema/schema";
import { displayUnit } from "../../schema/units";

const LENGTH: FieldInfo = {
  kind: "number",
  nullable: false,
  schema: { type: "number" },
  model: "Point",
  key: "x",
  unit: "m",
  geom: true,
  advanced: false,
  required: true,
  exclusiveMin: false,
  exclusiveMax: false,
};

/** 中心角 (一周未満、0 は直線) */
const ANGLE: FieldInfo = {
  ...LENGTH,
  key: "angle",
  unit: "deg",
  geom: false,
  min: -360,
  max: 360,
  exclusiveMin: true,
  exclusiveMax: true,
};

export interface PathTableOps {
  setVertex: (i: number, q: Point) => void;
  setBulge: (i: number, bulge: number) => void;
  /** 辺 i を中点で分ける (新しい頂点は i + 1) */
  split: (i: number) => void;
  removeVertex: (i: number) => void;
}

/** 中心角 [°] (表示用に丸める) */
function angleDeg(b: number): number {
  const deg = (angleFromBulge(b) * 180) / Math.PI;
  return Math.round(deg * 1e9) / 1e9;
}

export function PathTable({ path, label, ops, closed = true }: { path: PathData; label: string; ops: PathTableOps; closed?: boolean }) {
  const { t } = useTranslation();
  const ctx = useUnitContext();
  const unit = displayUnit("m", true, ctx);
  const b = bulgesOf(path);
  const pts = path.polygon;
  return (
    <table className="table vertex-table" aria-label={label}>
      <thead>
        <tr>
          <th>#</th>
          <th>x [{unit}]</th>
          <th>y [{unit}]</th>
          <th title={t("widgets.arcAngleHint")}>{t("widgets.arcAngle")}</th>
          <th />
        </tr>
      </thead>
      <tbody>
        {pts.map((p, i) => (
          <tr key={i}>
            <td className="muted">{i + 1}</td>
            {[0, 1].map((k) => (
              <td key={k}>
                <QuantityInput
                  info={LENGTH}
                  ctx={ctx}
                  aria-label={`${k === 0 ? "x" : "y"} ${i + 1}`}
                  value={p[k]}
                  onCommit={(v) => {
                    if (v === null) return;
                    ops.setVertex(i, k === 0 ? [v, p[1]] : [p[0], v]);
                  }}
                />
              </td>
            ))}
            <td>
              {(closed || i < pts.length - 1) && (
              <QuantityInput
                info={ANGLE}
                ctx={ctx}
                aria-label={`${t("widgets.arcAngle")} ${i + 1}`}
                value={angleDeg(b[i])}
                onCommit={(v) => {
                  if (v === null) return;
                  ops.setBulge(i, bulgeFromAngle((v * Math.PI) / 180));
                }}
              />
              )}
            </td>
            <td className="nowrap">
              <button type="button" className="button small" title={t("widgets.vertexInsert")} disabled={!closed && i === pts.length - 1} onClick={() => ops.split(i)}>
                +
              </button>
              <button
                type="button"
                className="button small danger"
                disabled={closed ? removeVertex(path, i) === null : pts.length <= 2}
                title={t("widgets.vertexDelete")}
                onClick={() => ops.removeVertex(i)}
              >
                −
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
