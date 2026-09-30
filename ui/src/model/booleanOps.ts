// 領域のブーリアン (P7e、prompts/132): この領域 (target) と相手 (tools) の和・差・積を target に書く。
// - 結果が複数に分かれたら、target と最も重なるものを target (ID・種類・値はそのまま)、ほかは target の値を写した
//   新しい領域 (局所メッシュ幅も写す、target のすぐ後ろ) にする
// - 相手は消す (keepTools なら残す)。円の領域は結果で多角形 (円弧) になる
// - 結果が空・点で接する穴ができる・計算できないときは文書を変えずに理由を返す

import type { Draft } from "immer";
import { booleanShapes, overlapArea, type BoolError, type BoolOp } from "../cad/boolean";
import { keepLayer } from "./layers";
import { regionShape, uniqueRegionId, type Project, type Region } from "./project";
import { deleteRegion, writeShape } from "./regionOps";

interface LocalSize {
  region: string;
  size: number;
}

export type BooleanOutcome = { ok: true; ids: string[] } | { ok: false; error: BoolError | "noTools" };

export function applyRegionBoolean(d: Draft<Project>, op: BoolOp, targetId: string, toolIds: string[], keepTools: boolean): BooleanOutcome {
  const regions = d.geometry.regions as Region[];
  const target = regions.find((r) => r.id === targetId);
  const tools = toolIds.filter((id) => id !== targetId).map((id) => regions.find((r) => r.id === id)).filter((r): r is Region => r !== undefined);
  if (!target || tools.length === 0) return { ok: false, error: "noTools" };
  const base = regionShape(target);
  const res = booleanShapes(op, base, tools.map(regionShape));
  if ("error" in res) return { ok: false, error: res.error };
  const shapes = res.shapes;
  let main = 0;
  if (shapes.length > 1) {
    const over = shapes.map((sh) => overlapArea(sh, base));
    main = over.indexOf(Math.max(...over));
  }
  // 新しい領域は元の target の値の写し (target を書き換える前に取る)
  const template = JSON.parse(JSON.stringify(target)) as Region; // Immer の draft は structuredClone できない
  writeShape(target, shapes[main]);
  const ids = [target.id];
  const ls = d.mesh.local_sizes as LocalSize[] | undefined;
  const size = ls?.find((l) => l.region === target.id)?.size;
  let at = regions.indexOf(target) + 1;
  shapes.forEach((sh, k) => {
    if (k === main) return;
    const copy = keepLayer(template, JSON.parse(JSON.stringify(template)) as Region);
    copy.id = uniqueRegionId(d as Project, `${target.id}_`);
    writeShape(copy, sh);
    regions.splice(at++, 0, copy);
    if (ls && size !== undefined) ls.push({ region: copy.id, size });
    ids.push(copy.id);
  });
  if (!keepTools) for (const t of tools) deleteRegion(d, t.id);
  return { ok: true, ids };
}
