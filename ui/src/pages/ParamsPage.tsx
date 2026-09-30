// パラメータのページ (P7f、prompts/132): 名前付きの式の表 (名前・式・計算した値・説明、追加・削除・名前の変更は
// 式の中の名前も)、式を束縛した欄の一覧 (外す)。数値の欄にパラメータの名前を使った式を書くと、その欄に束縛される。

import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { askConfirm } from "../app/dialogs";
import { logWarning } from "../app/messages";
import { fieldLabel } from "../forms/labels";
import { useUnitContext } from "../forms/useField";
import { useDocument } from "../model/documentStore";
import {
  addParam,
  deleteParam,
  nextParamName,
  paramExprError,
  paramNameError,
  paramsOf,
  paramUsers,
  paramValues,
  removeBinding,
  renameParam,
  resolveAddress,
  setParamDescription,
  setParamExpr,
  type PathToken,
} from "../model/params";
import { edgeIndexOf, type Project } from "../model/project";
import { fieldInfo, getIn, useProjectSchema, type JsonSchema } from "../schema/schema";
import { displayUnit, formatQuantity } from "../schema/units";
import { edgeLabel } from "../tree/treeModel";
import { formatNumber } from "../util/format";
import { CommitText } from "./inputs";
import { Hint } from "./widgets/common";

const useUpdate = () => useDocument((s) => s.update);

const xy = (k: unknown) => (k === 0 ? "x" : k === 1 ? "y" : String(k));

/** 束縛した欄の名前 (領域 anode › 電圧、ドメイン › 頂点 2 x など) */
function bindingLabel(p: Project, addr: readonly PathToken[], schema: JsonSchema, t: TFunction): string {
  const at = resolveAddress(p, addr);
  const parts: string[] = [];
  let rest: readonly PathToken[] = addr;
  const sel = (tok: PathToken | undefined) => (tok !== undefined && typeof tok === "object" ? ("id" in tok ? tok.id : tok.edge) : String(tok));
  if (addr[0] === "geometry" && addr[1] === "regions") {
    parts.push(`${t("tree.regions")} ${sel(addr[2])}`);
    rest = addr.slice(3);
  } else if (addr[0] === "geometry" && addr[1] === "boundaries") {
    const tok = addr[2];
    const e = tok && typeof tok === "object" && "edge" in tok ? edgeIndexOf(p, tok.edge) : -1;
    parts.push(`${t("tree.boundaries")} ${e >= 0 ? edgeLabel(p, e, t) : sel(tok)}`);
    rest = addr.slice(3);
  } else if (addr[0] === "geometry" && addr[1] === "domain") {
    parts.push(t("tree.domain"));
    rest = addr.slice(2);
  } else if (addr[0] === "cad" && addr[1] === "sketch") {
    parts.push(`${t("tree.sketch")} ${sel(addr[2])}`);
    rest = addr.slice(3);
  }
  if (rest[0] === "holes" && typeof rest[1] === "number") {
    parts.push(t("holes.title", { n: rest[1] + 1 }));
    rest = rest.slice(2);
  }
  if ((rest[0] === "polygon" || rest[0] === "points") && typeof rest[1] === "number") {
    parts.push(`${t("paramsPage.vertex", { n: rest[1] + 1 })} ${xy(rest[2])}`);
  } else if (at) {
    const info = typeof at[at.length - 1] === "number" ? fieldInfo(schema, at.slice(0, -1)) : fieldInfo(schema, at);
    const last = rest[rest.length - 1];
    const name = info ? fieldLabel(info) : String(typeof last === "object" ? sel(last) : last);
    parts.push(typeof at[at.length - 1] === "number" ? `${name} ${xy(at[at.length - 1])}` : name);
  } else parts.push(rest.map((x) => (typeof x === "object" ? sel(x) : String(x))).join("."));
  return parts.join(" › ");
}

export function ParamsPage() {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const schema = useProjectSchema();
  const ctx = useUnitContext();
  const update = useUpdate();
  const { vars, bindings } = paramsOf(project);
  const ev = paramValues(project);
  const remove = async (name: string) => {
    const users = paramUsers(project, name);
    if (users.vars.length) {
      logWarning(t("msg.source.app"), t("paramsPage.usedBy", { name, names: users.vars.join(", ") }));
      return;
    }
    if (users.bindings > 0 && !(await askConfirm(t("paramsPage.deleteTitle"), t("paramsPage.deleteMessage", { name, n: users.bindings }), { okLabel: t("tree.delete"), danger: true }))) return;
    update(t("paramsPage.delete"), (d) => void deleteParam(d, name));
  };
  return (
    <>
      <p className="hint">{t("paramsPage.description")}</p>
      {vars.length > 0 && (
        <table className="table params-table" aria-label={t("tree.params")}>
          <thead>
            <tr>
              <th>{t("paramsPage.name")}</th>
              <th>{t("paramsPage.expr")}</th>
              <th>{t("paramsPage.value")}</th>
              <th>{t("paramsPage.note")}</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {vars.map((v) => (
              <tr key={v.name}>
                <td>
                  <CommitText
                    aria-label={`${t("paramsPage.name")} ${v.name}`}
                    value={v.name}
                    validate={(s) => paramNameError(project, v.name, s.trim())}
                    onCommit={(s) => update(t("paramsPage.rename"), (d) => renameParam(d, v.name, s.trim()))}
                  />
                </td>
                <td>
                  <CommitText
                    aria-label={`${t("paramsPage.expr")} ${v.name}`}
                    value={v.expr}
                    validate={(s) => paramExprError(project, v.name, s.trim())}
                    onCommit={(s) => update(t("paramsPage.setExpr", { name: v.name }), (d) => setParamExpr(d, v.name, s.trim()))}
                  />
                </td>
                <td className="muted nowrap">{ev.values[v.name] !== undefined ? formatNumber(ev.values[v.name]) : "-"}</td>
                <td>
                  <CommitText aria-label={`${t("paramsPage.note")} ${v.name}`} value={v.description ?? ""} onCommit={(s) => update(t("paramsPage.note"), (d) => setParamDescription(d, v.name, s.trim()))} />
                </td>
                <td>
                  <button type="button" className="button small danger" title={t("paramsPage.delete")} aria-label={`${t("paramsPage.delete")} ${v.name}`} onClick={() => void remove(v.name)}>
                    −
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {ev.error && <Hint tone="error">{ev.error}</Hint>}
      <div className="button-row">
        <button type="button" className="button" onClick={() => update(t("paramsPage.add"), (d) => addParam(d, nextParamName(d as Project)))}>
          {t("paramsPage.add")}
        </button>
      </div>
      <div className="subsection-title">{t("paramsPage.bindings", { n: bindings.length })}</div>
      {bindings.length === 0 ? (
        <p className="muted">{t("paramsPage.noBindings")}</p>
      ) : (
        <table className="table" aria-label={t("paramsPage.bindings", { n: bindings.length })}>
          <tbody>
            {bindings.map((b) => {
              const at = resolveAddress(project, b.path);
              const v = at ? getIn(project, at) : undefined;
              const info = at && typeof at[at.length - 1] !== "number" ? fieldInfo(schema, at) : null;
              // 頂点の座標 (情報の無い番号の欄) は長さ
              const geom = info ? info.geom : true;
              const unit = displayUnit(info?.unit, geom, ctx);
              const key = JSON.stringify(b.path);
              return (
                <tr key={key}>
                  <td>{bindingLabel(project, b.path, schema, t)}</td>
                  <td className="nowrap">= {b.expr}</td>
                  <td className="muted nowrap">{typeof v === "number" ? `${formatQuantity(v, geom, ctx)} ${unit}`.trim() : "-"}</td>
                  <td className="nowrap">
                    <button type="button" className="button small" onClick={() => update(t("paramsPage.unbind"), (d) => removeBinding(d, b.path))}>
                      {t("paramsPage.unbind")}
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      <Hint>{t("paramsPage.hint")}</Hint>
    </>
  );
}
