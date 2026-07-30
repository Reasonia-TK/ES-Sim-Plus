import type { Point } from "./types";
import { edgeCrossing } from "./canvas/isolines";

/**
 * 2D PIC のシースエッジ検出+可視化 (prompts/98)。
 *
 * 1D (pic1d.py) と異なり、2D は「準中性度の等値線 (面全体)」と「評価ラインに沿った
 * Brinkmann 積分判定 (線)」の2通りで可視化する。どちらもフロント側で計算する
 * (backend は SheathLine の座標を永続化するだけ):
 *  - 時間平均 (fields) も位相分解 (cycle) も done メッセージで既にフロントへ届いている
 *    節点配列なので、backend への再問い合わせなしに計算できる
 *  - 準中性度 α スライダをドラッグするたびに即座に等値線を引き直したい (α は表示専用の
 *    パラメータで project へ保存しない)
 *  - 「結果付き保存」ファイルを読み込んだだけ (backend 未起動/切断中) の状態でも
 *    シースエッジ検出・可視化が動作してほしい
 * という3点から、フロント計算がもっとも自然。
 */

// ---- Brinkmann のシースエッジ判定 ------------------------------------------------
//
// backend/es_sim/pic1d.py の brinkmann_sheath_edge と完全に同じ式 (改変しない):
//   G(s) = ∫₀ˢ n_e dx − ∫ₛ^{x_b} (n_i − n_e) dx
//        = ∫₀ˢ n_i dx − C   (C := ∫₀^{x_b} (n_i − n_e) dx は s に依らない定数)
// dG/ds = n_i(s) ≥ 0 なので G は単調非減少 → 根は高々一意。∫₀ˢ n_i dx は
// n_i≥0 の台形則累積和で s について非減少なので、符号変化点を線形走査 (配列長は
// せいぜい評価ラインのサンプル数 = 200 程度) で見つければよい。
//
// pic1d.py 版との違いは1点だけ: pic1d.py はバルク参照点 x_b が格子点上にあるとは
// 限らない (x_b=gap/2 を任意の格子に当てはめる) ため線形補間で挿入する分岐を持つが、
// 2D の評価ラインでは p2 (バルク側の指定点) がそのまま x_b であり、dist の末尾
// (= sampleAlongLine の最終サンプル点) が必ず p2 そのものになるよう呼び出し側が
// 構成する。よって「x_b が格子点上に無い」ケース自体が存在せず、その挿入分岐は不要。

/**
 * dist (p1 からの距離、昇順、末尾がバルク参照点 x_b) と、対応する n_e/n_i から
 * Brinkmann のシースエッジ位置 s (p1 からの距離 [m]) を求める。根が存在しない
 * (プラズマ未形成・n_i がほぼ0 等の退化ケース) 場合は null。
 */
export function brinkmannSheathEdge(dist: number[], nE: number[], nI: number[]): number | null {
  const n = dist.length;
  if (n < 2) return null;
  const xB = dist[n - 1];
  if (xB <= dist[0]) return null; // 退化ケース (積分区間が無い)

  const cumE = new Array<number>(n).fill(0);
  const cumI = new Array<number>(n).fill(0);
  for (let i = 1; i < n; i++) {
    const dx = dist[i] - dist[i - 1];
    cumE[i] = cumE[i - 1] + 0.5 * (nE[i] + nE[i - 1]) * dx;
    cumI[i] = cumI[i - 1] + 0.5 * (nI[i] + nI[i - 1]) * dx;
  }
  const c = cumI[n - 1] - cumE[n - 1]; // ∫0^xb (n_i-n_e) dx (s に依らない定数)
  const g = cumI.map((v) => v - c); // G(s) を各サンプル点で評価した配列 (n_i>=0 なら非減少)

  // g[0] = -C = cumE[n-1]-cumI[n-1]、g[n-1] = cumE[n-1] (常に n_e>=0 なので0以上)。
  // g[n-1]<=0 になるのは実質 n_e が区間全体でほぼ0 (プラズマ未形成) の退化ケースのみ
  if (g[0] > 0.0 || g[n - 1] <= 0.0) return null;

  // searchsorted(g, 0.0) 相当 (g は非減少なので線形走査で十分、pic1d.py のような
  // 大きな格子ではないため二分探索にする必要はない)
  let idx = 0;
  while (idx < n && g[idx] < 0.0) idx++;
  if (idx <= 0) return dist[0];
  const gLo = g[idx - 1];
  const gHi = g[idx];
  if (gHi === gLo) return dist[idx];
  const frac = -gLo / (gHi - gLo);
  return dist[idx - 1] + frac * (dist[idx] - dist[idx - 1]);
}

// ---- 準中性度の等値線 (marching triangles) --------------------------------------

/**
 * 三角形メッシュ上の節点値 values について、指定レベルの等値線分を抽出する。
 * canvas/isolines.ts の computeIsolines (電位の等電位線、複数レベル一括) と違い、
 * こちらは単一レベル + 三角形マスク (ほぼ真空の三角形を除外する等) 対応にした版
 * (prompts/98: 準中性度 n_e/n_i=α の等値線専用)。
 * mask(tri) が false を返す三角形はスキップする (未指定なら全三角形を対象にする)。
 */
export function marchingTrianglesContour(
  nodes: Point[],
  triangles: [number, number, number][],
  values: number[],
  level: number,
  mask?: (tri: [number, number, number]) => boolean,
): Array<[Point, Point]> {
  const result: Array<[Point, Point]> = [];
  for (const tri of triangles) {
    if (mask && !mask(tri)) continue;
    const [a, b, c] = tri;
    const va = values[a];
    const vb = values[b];
    const vc = values[c];
    if (!Number.isFinite(va) || !Number.isFinite(vb) || !Number.isFinite(vc)) continue;
    const triMin = Math.min(va, vb, vc);
    const triMax = Math.max(va, vb, vc);
    if (level < triMin || level > triMax) continue;

    const pa = nodes[a];
    const pb = nodes[b];
    const pc = nodes[c];
    const pts: Point[] = [];
    const eAB = edgeCrossing(pa, va, pb, vb, level);
    if (eAB) pts.push(eAB);
    const eBC = edgeCrossing(pb, vb, pc, vc, level);
    if (eBC) pts.push(eBC);
    const eCA = edgeCrossing(pc, vc, pa, va, level);
    if (eCA) pts.push(eCA);
    if (pts.length === 2) result.push([pts[0], pts[1]]);
  }
  return result;
}

// ---- ライン上のフィールド補間 (Brinkmann 評価ライン用) --------------------------------

// 三角形の重心座標 (l1,l2,l3 はそれぞれ頂点 a,b,c の重み)。標準的な面積比の公式
// (denom = 2×三角形の符号付き面積)。退化三角形 (面積0) は null を返す
function barycentric(
  ax: number, ay: number, bx: number, by: number, cx: number, cy: number, px: number, py: number,
): [number, number, number] | null {
  const denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
  if (denom === 0) return null;
  const l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom;
  const l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom;
  const l3 = 1 - l1 - l2;
  return [l1, l2, l3];
}

// 重心座標がすべて (許容誤差込みで) 非負なら三角形内 (境界含む) とみなす
const BARYCENTRIC_EPS = -1e-9;

/**
 * 点 (x,y) を含む三角形を総当たりで探し、重心座標補間した values の値を返す。
 * 見つからなければ (ドメイン外) NaN。20k要素程度のメッシュに対して評価ライン1本
 * (n=200点) を処理する程度なら総当たりで十分 (pic.py の walk 探索のような隣接
 * 三角形追跡・専用の点位置特定ヘルパは既存に無いため、ここで素朴に実装する)。
 */
function locateAndInterpolate(
  nodes: Point[],
  triangles: [number, number, number][],
  values: number[],
  x: number,
  y: number,
): number {
  for (const tri of triangles) {
    const [a, b, c] = tri;
    const [ax, ay] = nodes[a];
    const [bx, by] = nodes[b];
    const [cx, cy] = nodes[c];
    const bary = barycentric(ax, ay, bx, by, cx, cy, x, y);
    if (!bary) continue;
    const [l1, l2, l3] = bary;
    if (l1 >= BARYCENTRIC_EPS && l2 >= BARYCENTRIC_EPS && l3 >= BARYCENTRIC_EPS) {
      return l1 * values[a] + l2 * values[b] + l3 * values[c];
    }
  }
  return NaN;
}

/**
 * p1→p2 を n 等分した点で values を重心座標補間してサンプルする。dist は p1 から
 * の距離 (末尾が p1-p2 間の全長)。三角形が見つからない (ドメイン外) サンプルは
 * NaN にする (呼び出し側で NaN 区間を除外して Brinkmann 判定に渡すこと)。
 */
export function sampleAlongLine(
  nodes: Point[],
  triangles: [number, number, number][],
  values: number[],
  p1: Point,
  p2: Point,
  n = 200,
): { dist: number[]; values: number[] } {
  const length = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
  const dist = new Array<number>(n);
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const t = n > 1 ? i / (n - 1) : 0;
    const x = p1[0] + (p2[0] - p1[0]) * t;
    const y = p1[1] + (p2[1] - p1[1]) * t;
    dist[i] = length * t;
    out[i] = locateAndInterpolate(nodes, triangles, values, x, y);
  }
  return { dist, values: out };
}

/**
 * 評価ライン (p1=電極側、p2=バルク側=x_b) 上の Brinkmann シースエッジ位置 [m]
 * (p1 からの距離)。n_e/n_i をそれぞれ sampleAlongLine で補間し、どちらかが NaN
 * (ドメイン外) のサンプルは除外してから brinkmannSheathEdge に渡す。
 * ライン全域がドメイン外、n_i が全域ゼロ等の退化ケースでは例外を投げず null を返す。
 */
export function sheathLineEdge(
  nodes: Point[],
  triangles: [number, number, number][],
  nE: number[],
  nI: number[],
  p1: Point,
  p2: Point,
  n = 200,
): number | null {
  const se = sampleAlongLine(nodes, triangles, nE, p1, p2, n);
  const si = sampleAlongLine(nodes, triangles, nI, p1, p2, n);
  const dist: number[] = [];
  const neOut: number[] = [];
  const niOut: number[] = [];
  for (let i = 0; i < se.dist.length; i++) {
    const e = se.values[i];
    const ii = si.values[i];
    if (!Number.isFinite(e) || !Number.isFinite(ii)) continue; // ドメイン外サンプルを除外
    dist.push(se.dist[i]);
    neOut.push(e);
    niOut.push(ii);
  }
  if (dist.length < 2) return null; // ライン全域ドメイン外などの縮退ケース
  return brinkmannSheathEdge(dist, neOut, niOut);
}
