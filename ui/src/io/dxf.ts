// DXF の読み込みと書き出し (CAD v2 P7g、prompts/132)。形の解析と DXF の組み立てはバックエンド (ezdxf) が行う。
// 読み込み: ファイルを選ぶ → バックエンドで解析 → 要約と選択肢 (単位・レイヤ・閉じた形を領域に) を確かめる → 文書へ
// (元に戻す 1 件) → 全体表示。書き出し: 文書をバックエンドで DXF にして保存する (単位は長さの表示単位)。

import { askDxfImport } from "../app/dialogs";
import { errorText, logError, logInfo, logWarning } from "../app/messages";
import { apiPost, apiPostText } from "../backend/api";
import { useViewer } from "../graphics/viewerStore";
import { t } from "../i18n";
import { useDocument, documentName } from "../model/documentStore";
import { applyDxfImport, type DxfApplied, type DxfImport } from "../model/dxfImport";
import { usePrefs } from "../prefs/prefs";
import { pickAndReadBinary, saveTextFile } from "./fileAccess";

function toBase64(data: Uint8Array): string {
  let s = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < data.length; i += CHUNK) s += String.fromCharCode(...data.subarray(i, i + CHUNK));
  return btoa(s);
}

const source = () => t("msg.source.app");

/** 形の数の要約 (線 3・円弧 1 …) */
export function countsText(counts: Record<string, number>): string {
  const order = ["line", "arc", "circle", "polyline"] as const;
  return order
    .filter((k) => counts[k])
    .map((k) => `${t(`sketchPage.kind.${k}`)} ${counts[k]}`)
    .join("・");
}

export async function importDxfFile(): Promise<void> {
  let picked: { name: string; data: Uint8Array } | null;
  try {
    picked = await pickAndReadBinary(["dxf"], "DXF");
  } catch (e) {
    logError(source(), errorText(e));
    return;
  }
  if (!picked) return;
  let result: DxfImport;
  try {
    result = await apiPost<DxfImport>("/v2/cad/dxf/import", { data: toBase64(picked.data) });
  } catch (e) {
    logError(source(), t("dxf.readFailed", { name: picked.name, error: errorText(e) }));
    return;
  }
  if (result.entities.length === 0) {
    logWarning(source(), t("dxf.empty", { name: picked.name }));
    return;
  }
  const opts = await askDxfImport(picked.name, result);
  if (!opts) return;
  let applied: DxfApplied = { sketch: 0, regions: 0, layers: 0 };
  useDocument.getState().update(t("dxf.importAction"), (d) => void (applied = applyDxfImport(d, result, opts)));
  const skipped = Object.entries(result.skipped)
    .map(([k, n]) => `${k} ${n}`)
    .join("・");
  logInfo(source(), t("dxf.imported", { name: picked.name, sketch: applied.sketch, regions: applied.regions, layers: applied.layers }) + (skipped ? ` ${t("dxf.skipped", { list: skipped })}` : ""));
  useViewer.getState().requestFit();
}

export async function exportDxfFile(): Promise<void> {
  const unit = usePrefs.getState().lengthUnit === "um" ? "um" : "mm";
  const st = useDocument.getState();
  let text: string;
  try {
    text = await apiPostText("/v2/cad/dxf/export", { project: st.project, unit });
  } catch (e) {
    logError(source(), t("dxf.writeFailed", { error: errorText(e) }));
    return;
  }
  const base = documentName(st, t("app.untitled")).replace(/\.json$/i, "");
  try {
    if (await saveTextFile(`${base}.dxf`, text, "dxf", "DXF")) logInfo(source(), t("dxf.exported", { name: `${base}.dxf`, unit: unit === "um" ? "µm" : "mm" }));
  } catch (e) {
    logError(source(), errorText(e));
  }
}
