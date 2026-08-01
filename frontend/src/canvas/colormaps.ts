/**
 * 2D コンター描画用カラーマップ定義集 (prompts/103)。
 *
 * 各マップは少数のアンカー色 (matplotlib / turbo 論文の代表値を8〜16点程度で近似) を
 * 線形補間し、モジュール初期化時に一度だけ 256 段の LUT ([r,g,b] 数値タプル) を構築する。
 * colormap は等電位線・PIC/DSMC フィールド描画などで三角形ごとに毎フレーム呼ばれるため、
 * 呼び出しのたびにアンカー探索や `rgb(r,g,b)` 文字列生成をやり直すと重くなる。
 * 事前計算した LUT を引くだけにすれば O(1) で済み、256 段あれば見た目の帯 (バンディング) も
 * 目立たない。
 */

export type ColormapKey = "viridis" | "plasma" | "inferno" | "turbo" | "jet" | "coolwarm" | "gray";

// 既定のカラーマップ (従来の見た目を踏襲する viridis)
export const DEFAULT_COLORMAP: ColormapKey = "viridis";

// UI (select) 表示用の一覧。並び順がそのまま select の選択肢順になる
export const COLORMAPS: { key: ColormapKey; label: string }[] = [
  { key: "viridis", label: "Viridis" },
  { key: "plasma", label: "Plasma" },
  { key: "inferno", label: "Inferno" },
  { key: "turbo", label: "Turbo" },
  { key: "jet", label: "Jet" },
  { key: "coolwarm", label: "Coolwarm (発散)" },
  { key: "gray", label: "グレースケール" },
];

type Anchor = [number, number, number];

// 各マップのアンカー色列 (0→1 に等間隔で並ぶ前提)。厳密な原典一致は不要 (見た目が崩れない
// 近似で十分) だが、viridis は既存の簡易実装 (5点) から見た目がほぼ変わらないよう
// matplotlib の公表値に近い9点サンプルに置き換える。
const ANCHORS: Record<ColormapKey, Anchor[]> = {
  // matplotlib viridis の代表的な9点サンプル
  viridis: [
    [68, 1, 84],
    [71, 44, 122],
    [59, 81, 139],
    [44, 113, 142],
    [33, 144, 141],
    [39, 173, 129],
    [92, 200, 99],
    [170, 220, 50],
    [253, 231, 37],
  ],
  // matplotlib plasma の代表的な9点サンプル
  plasma: [
    [13, 8, 135],
    [71, 3, 159],
    [115, 1, 168],
    [156, 23, 158],
    [189, 55, 134],
    [216, 87, 107],
    [237, 121, 83],
    [251, 159, 58],
    [240, 249, 33],
  ],
  // matplotlib inferno の代表的な9点サンプル
  inferno: [
    [0, 0, 4],
    [27, 12, 66],
    [75, 12, 107],
    [120, 28, 109],
    [165, 44, 96],
    [207, 68, 70],
    [237, 105, 37],
    [251, 155, 6],
    [252, 255, 164],
  ],
  // Google turbo の簡略版 (jet の知覚均一改良版)。両端が紫→暗赤になる点が jet と異なる
  turbo: [
    [48, 18, 59],
    [63, 74, 180],
    [44, 129, 227],
    [33, 190, 213],
    [33, 213, 173],
    [95, 232, 106],
    [170, 231, 41],
    [233, 215, 40],
    [247, 156, 39],
    [232, 82, 35],
    [122, 4, 3],
  ],
  // 定番 jet (比較用)。turbo と違い両端が濃紺/濃赤
  jet: [
    [0, 0, 131],
    [0, 0, 255],
    [0, 128, 255],
    [0, 255, 255],
    [128, 255, 128],
    [255, 255, 0],
    [255, 128, 0],
    [255, 0, 0],
    [128, 0, 0],
  ],
  // Kenneth Moreland の発散型 cool-warm を近似 (符号のある場、電位など向け)
  coolwarm: [
    [58, 76, 192],
    [98, 130, 234],
    [141, 176, 254],
    [184, 208, 249],
    [221, 221, 221],
    [236, 199, 175],
    [245, 156, 129],
    [222, 96, 77],
    [180, 4, 38],
  ],
  // グレースケール (印刷用)。単純な黒→白の2点で十分
  gray: [
    [0, 0, 0],
    [255, 255, 255],
  ],
};

const LUT_SIZE = 256;

// アンカー色列を LUT_SIZE 段に線形補間して展開する
function buildLut(anchors: Anchor[]): Anchor[] {
  const lut: Anchor[] = new Array(LUT_SIZE);
  const n = anchors.length;
  for (let i = 0; i < LUT_SIZE; i++) {
    const t = i / (LUT_SIZE - 1);
    const x = t * (n - 1);
    const idx = Math.min(n - 2, Math.floor(x));
    const f = x - idx;
    const [r1, g1, b1] = anchors[idx];
    const [r2, g2, b2] = anchors[idx + 1];
    lut[i] = [r1 + (r2 - r1) * f, g1 + (g2 - g1) * f, b1 + (b2 - b1) * f];
  }
  return lut;
}

// モジュール初期化時に一度だけ全マップの LUT を構築する (毎フレーム呼ばれる colormap の
// ホットパスからアンカー補間の計算を追い出すため)
const LUTS: Record<ColormapKey, Anchor[]> = Object.fromEntries(
  (Object.keys(ANCHORS) as ColormapKey[]).map((k) => [k, buildLut(ANCHORS[k])]),
) as Record<ColormapKey, Anchor[]>;

/**
 * t (0〜1 想定、範囲外はクランプ) に対応する色を [r,g,b] (0〜255) で返す。
 * 手動カラーレンジ指定時に range 外の値が来ても、ここでクランプすることで
 * 自然に「端色にクランプ」される (呼び出し側で追加のクランプ処理は不要)。
 */
export function sampleColormap(key: ColormapKey, t: number): [number, number, number] {
  const c = Math.min(1, Math.max(0, t));
  const idx = Math.min(LUT_SIZE - 1, Math.floor(c * (LUT_SIZE - 1) + 0.5));
  return LUTS[key][idx];
}

// canvas の fillStyle/gradient にそのまま渡せる CSS 色文字列版
export function colormapCss(key: ColormapKey, t: number): string {
  const [r, g, b] = sampleColormap(key, t);
  return `rgb(${r},${g},${b})`;
}
