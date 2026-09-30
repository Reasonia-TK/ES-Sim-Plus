// レイヤのページ (P7f、prompts/132): 今のレイヤ (新しく描く形が入る)・色 (スケッチの線)・表示・ロック・形の数、
// 追加・名前の変更・削除 (形は既定のレイヤへ)、選んだ形を今のレイヤへ移す。領域・スケッチのページのレイヤの選択も。

import { useTranslation } from "react-i18next";
import { useDocument } from "../model/documentStore";
import {
  activeLayer,
  addLayer,
  DEFAULT_LAYER,
  deleteLayer,
  itemLayer,
  layerCounts,
  layerNameError,
  layersOf,
  setActiveLayer,
  setItemLayer,
  updateLayer,
} from "../model/layers";
import type { Project } from "../model/project";
import { useSelection, type PickRef } from "../model/selection";
import { sketchOf } from "../model/sketch";
import { CommitText, Field } from "./inputs";
import { Hint } from "./widgets/common";

const useUpdate = () => useDocument((s) => s.update);

/** 色の欄の既定 (レイヤの色が無いときのスケッチの色、暗い画面の --color-sketch) */
const DEFAULT_COLOR = "#b8c7d6";

export function LayersPage() {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const picked = useSelection((s) => s.picked);
  const update = useUpdate();
  const layers = layersOf(project);
  const active = activeLayer(project);
  const counts = layerCounts(project);
  const activeName = layers.find((l) => l.id === active)?.name ?? active;
  return (
    <>
      <p className="hint">{t("layers.description")}</p>
      <table className="table layers-table" aria-label={t("tree.layers")}>
        <thead>
          <tr>
            <th title={t("layers.active")}>{t("layers.activeShort")}</th>
            <th>{t("layers.color")}</th>
            <th>{t("layers.name")}</th>
            <th>{t("layers.visible")}</th>
            <th>{t("layers.locked")}</th>
            <th>{t("layers.count")}</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {layers.map((l) => (
            <tr key={l.id} className={l.id === active ? "selected" : undefined}>
              <td>
                <input type="radio" name="active-layer" aria-label={`${t("layers.active")} ${l.name}`} checked={l.id === active} onChange={() => update(t("layers.setActive"), (d) => setActiveLayer(d, l.id))} />
              </td>
              <td className="nowrap">
                <input
                  type="color"
                  className="color-swatch"
                  aria-label={`${t("layers.color")} ${l.name}`}
                  value={l.color ?? DEFAULT_COLOR}
                  onChange={(e) => update(t("layers.setColor"), (d) => updateLayer(d, l.id, { color: e.target.value }))}
                />
                {l.color && (
                  <button type="button" className="button small" title={t("layers.resetColor")} aria-label={`${t("layers.resetColor")} ${l.name}`} onClick={() => update(t("layers.setColor"), (d) => updateLayer(d, l.id, { color: null }))}>
                    ×
                  </button>
                )}
              </td>
              <td>
                {l.id === DEFAULT_LAYER ? (
                  <span>{l.name}</span>
                ) : (
                  <CommitText
                    aria-label={`${t("layers.name")} ${l.name}`}
                    value={l.name}
                    validate={(s) => {
                      const k = layerNameError(project, l.id, s);
                      return k ? t(k) : null;
                    }}
                    onCommit={(s) => update(t("layers.rename"), (d) => updateLayer(d, l.id, { name: s.trim() }))}
                  />
                )}
              </td>
              <td>
                <input type="checkbox" aria-label={`${t("layers.visible")} ${l.name}`} checked={l.visible !== false} onChange={(e) => update(t("layers.setVisible"), (d) => updateLayer(d, l.id, { visible: e.target.checked }))} />
              </td>
              <td>
                <input type="checkbox" aria-label={`${t("layers.locked")} ${l.name}`} checked={Boolean(l.locked)} onChange={(e) => update(t("layers.setLocked"), (d) => updateLayer(d, l.id, { locked: e.target.checked }))} />
              </td>
              <td className="muted">{counts.get(l.id) ?? 0}</td>
              <td>
                {l.id !== DEFAULT_LAYER && (
                  <button type="button" className="button small danger" title={t("layers.delete")} aria-label={`${t("layers.delete")} ${l.name}`} onClick={() => update(t("layers.delete"), (d) => deleteLayer(d, l.id))}>
                    −
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="button-row">
        <button
          type="button"
          className="button"
          onClick={() =>
            update(t("layers.add"), (d) => {
              const id = addLayer(d);
              setActiveLayer(d, id);
            })
          }
        >
          {t("layers.add")}
        </button>
        <button
          type="button"
          className="button"
          disabled={picked.length === 0}
          onClick={() =>
            update(t("layers.movePicked", { name: activeName }), (d) => {
              for (const it of picked) setItemLayer(d, it, active);
            })
          }
        >
          {t("layers.movePicked", { name: activeName })}
        </button>
      </div>
      <Hint>{t("layers.hint")}</Hint>
    </>
  );
}

/** 領域・スケッチのページのレイヤの選択 */
export function LayerSelect({ item }: { item: PickRef }) {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const update = useUpdate();
  const target =
    item.kind === "region" ? project.geometry.regions.find((r) => r.id === item.id) : (sketchOf(project as Project).find((e) => e.id === item.id) as { layer?: string | null } | undefined);
  if (!target) return null;
  const cur = itemLayer(project, target);
  return (
    <Field label={t("layers.layer")}>
      {(id) => (
        <select id={id} className="input" value={cur} onChange={(e) => update(t("layers.moveItem"), (d) => setItemLayer(d, item, e.target.value))}>
          {layersOf(project).map((l) => (
            <option key={l.id} value={l.id}>
              {l.name}
              {l.visible === false ? ` (${t("layers.hidden")})` : ""}
              {l.locked ? ` (${t("layers.lockedShort")})` : ""}
            </option>
          ))}
        </select>
      )}
    </Field>
  );
}
