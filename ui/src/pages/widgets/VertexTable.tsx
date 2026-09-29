// 多角形の頂点の表 (長さは表示単位で編集、頂点の追加・削除、3 点未満にはしない)。

import { useTranslation } from "react-i18next";
import { QuantityInput } from "../../forms/SchemaField";
import { setValue, useUnitContext } from "../../forms/useField";
import { useDocument } from "../../model/documentStore";
import type { Point } from "../../model/project";
import { getIn, type FieldInfo, type Path } from "../../schema/schema";
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

export function VertexTable({ path, label }: { path: Path; label: string }) {
  const { t } = useTranslation();
  const ctx = useUnitContext();
  const pts = (useDocument((s) => getIn(s.project, path)) as Point[] | undefined) ?? [];
  const write = (next: Point[]) => setValue(path, next, label);
  const unit = displayUnit("m", true, ctx);
  return (
    <table className="table vertex-table">
      <thead>
        <tr>
          <th>#</th>
          <th>x [{unit}]</th>
          <th>y [{unit}]</th>
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
                    write(pts.map((q, j) => (j === i ? ((k === 0 ? [v, q[1]] : [q[0], v]) as Point) : q)));
                  }}
                />
              </td>
            ))}
            <td className="nowrap">
              <button
                type="button"
                className="button small"
                title={t("widgets.vertexInsert")}
                onClick={() => {
                  const q = pts[(i + 1) % pts.length];
                  const mid: Point = [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2];
                  write([...pts.slice(0, i + 1), mid, ...pts.slice(i + 1)]);
                }}
              >
                +
              </button>
              <button type="button" className="button small danger" disabled={pts.length <= 3} title={t("widgets.vertexDelete")} onClick={() => write(pts.filter((_, j) => j !== i))}>
                −
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
