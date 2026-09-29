// 書き出し: 表示している場の値 (CSV、節点の量は節点ごと、三角形の量は重心ごと)、画面の画像 (PNG、3 枚の層を重ねる)、
// プロファイル (v1 の profile.csv と同じ列)。

import type { ProfileResult } from "../backend/staticApi";
import type { ScalarField } from "./scene";

const num = (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? "" : String(v));

/** 場の CSV (座標は m、値は場の単位) */
export function fieldCsv(f: ScalarField, axes: [string, string]): string {
  const head = `${axes[0]} [m],${axes[1]} [m],${f.label} [${f.unit}]`;
  const lines = [head];
  const { nodes, triangles } = f.mesh;
  if (f.location === "node") {
    nodes.forEach(([x, y], i) => lines.push(`${x},${y},${num(f.values[i])}`));
  } else {
    triangles.forEach(([a, b, c], i) => {
      const x = (nodes[a][0] + nodes[b][0] + nodes[c][0]) / 3;
      const y = (nodes[a][1] + nodes[b][1] + nodes[c][1]) / 3;
      lines.push(`${x},${y},${num(f.values[i])}`);
    });
  }
  return lines.join("\n") + "\n";
}

/** プロファイルの CSV (v1 と同じ: s,v,e_abs、領域の外は空欄) */
export function profileCsv(p: ProfileResult): string {
  const lines = ["s,v,e_abs"];
  p.s.forEach((s, i) => lines.push(`${s},${num(p.v[i])},${num(p.e_abs[i])}`));
  return lines.join("\n") + "\n";
}

/** 層を重ねた PNG (層は同じ大きさ、WebGL の層は描いた直後に呼ぶこと) */
export function composePng(layers: HTMLCanvasElement[]): Promise<Blob> {
  const w = Math.max(...layers.map((c) => c.width));
  const h = Math.max(...layers.map((c) => c.height));
  const out = document.createElement("canvas");
  out.width = w;
  out.height = h;
  const ctx = out.getContext("2d");
  if (!ctx) return Promise.reject(new Error("canvas 2d is not available"));
  for (const c of layers) ctx.drawImage(c, 0, 0, w, h);
  return new Promise((resolve, reject) => out.toBlob((b) => (b ? resolve(b) : reject(new Error("PNG encode failed"))), "image/png"));
}

/** ファイル名に使える日時 (20260930-153012) */
export function stamp(d = new Date()): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}
