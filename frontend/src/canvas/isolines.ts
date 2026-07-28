import type { Point } from "../types";
import { arrayMin, arrayMax } from "../mathUtils";

/**
 * 等値線 (マーチング・トライアングル)
 *
 * 各三角形・各レベルについて、頂点値がレベルを跨ぐ辺上で線形補間により
 * 交点を求め、線分 (2点) を返す。線分同士の連結・ポリライン化は行わない
 * (呼び出し側は各線分を moveTo/lineTo で個別に描画すればよい)。
 */

// 三角形の1辺上でレベルと交差する点を線形補間で求める
function edgeCrossing(
  pA: Point,
  vA: number,
  pB: Point,
  vB: number,
  level: number,
): Point | null {
  if ((vA < level) === (vB < level)) return null; // 同じ側なら交差なし
  const t = (level - vA) / (vB - vA);
  return [pA[0] + (pB[0] - pA[0]) * t, pA[1] + (pB[1] - pA[1]) * t];
}

/**
 * 節点電位 v から等値線を計算する。
 * @param nodes 節点座標
 * @param triangles 三角形 (節点インデックス3つ組)
 * @param v 節点ごとの値 (電位など)
 * @param levels 等値レベルの数 (デフォルト15、v_min/v_maxを等分)
 * @returns levels 本ごとの線分群 (各線分は2点のPoint配列)
 */
export function computeIsolines(
  nodes: Point[],
  triangles: [number, number, number][],
  v: number[],
  levels = 15,
): Point[][][] {
  if (v.length === 0 || triangles.length === 0) return [];

  const vMin = arrayMin(v);
  const vMax = arrayMax(v);
  const range = vMax - vMin;
  if (range <= 0) return [];

  // 両端 (vMin, vMax) を除いた等分レベルにする (境界上のみの縮退線分を避ける)
  const levelValues: number[] = [];
  for (let i = 1; i <= levels; i++) {
    levelValues.push(vMin + (range * i) / (levels + 1));
  }

  const result: Point[][][] = levelValues.map(() => []);

  for (const [a, b, c] of triangles) {
    const pa = nodes[a];
    const pb = nodes[b];
    const pc = nodes[c];
    const va = v[a];
    const vb = v[b];
    const vc = v[c];
    const triMin = Math.min(va, vb, vc);
    const triMax = Math.max(va, vb, vc);

    for (let li = 0; li < levelValues.length; li++) {
      const level = levelValues[li];
      if (level < triMin || level > triMax) continue;

      const pts: Point[] = [];
      const eAB = edgeCrossing(pa, va, pb, vb, level);
      if (eAB) pts.push(eAB);
      const eBC = edgeCrossing(pb, vb, pc, vc, level);
      if (eBC) pts.push(eBC);
      const eCA = edgeCrossing(pc, vc, pa, va, level);
      if (eCA) pts.push(eCA);

      // 三角形と等値面との交差は通常2点 (頂点をちょうど通る縮退ケースは無視)
      if (pts.length === 2) {
        result[li].push([pts[0], pts[1]]);
      }
    }
  }

  return result;
}
