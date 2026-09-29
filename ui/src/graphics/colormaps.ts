// 配色 (v1 canvas/colormaps.ts と同じアンカー色を 256 段に補間)。WebGL の配色テクスチャ (RGBA) にもする。

export type ColormapKey = "viridis" | "plasma" | "inferno" | "turbo" | "jet" | "coolwarm" | "gray";

export const COLORMAP_KEYS: ColormapKey[] = ["viridis", "plasma", "inferno", "turbo", "jet", "coolwarm", "gray"];
export const DEFAULT_COLORMAP: ColormapKey = "viridis";

type Rgb = [number, number, number];

const ANCHORS: Record<ColormapKey, Rgb[]> = {
  viridis: [[68, 1, 84], [71, 44, 122], [59, 81, 139], [44, 113, 142], [33, 144, 141], [39, 173, 129], [92, 200, 99], [170, 220, 50], [253, 231, 37]],
  plasma: [[13, 8, 135], [71, 3, 159], [115, 1, 168], [156, 23, 158], [189, 55, 134], [216, 87, 107], [237, 121, 83], [251, 159, 58], [240, 249, 33]],
  inferno: [[0, 0, 4], [27, 12, 66], [75, 12, 107], [120, 28, 109], [165, 44, 96], [207, 68, 70], [237, 105, 37], [251, 155, 6], [252, 255, 164]],
  turbo: [[48, 18, 59], [63, 74, 180], [44, 129, 227], [33, 190, 213], [33, 213, 173], [95, 232, 106], [170, 231, 41], [233, 215, 40], [247, 156, 39], [232, 82, 35], [122, 4, 3]],
  jet: [[0, 0, 131], [0, 0, 255], [0, 128, 255], [0, 255, 255], [128, 255, 128], [255, 255, 0], [255, 128, 0], [255, 0, 0], [128, 0, 0]],
  coolwarm: [[58, 76, 192], [98, 130, 234], [141, 176, 254], [184, 208, 249], [221, 221, 221], [236, 199, 175], [245, 156, 129], [222, 96, 77], [180, 4, 38]],
  gray: [[0, 0, 0], [255, 255, 255]],
};

export const LUT_SIZE = 256;

function buildLut(anchors: Rgb[]): Rgb[] {
  const n = anchors.length;
  return Array.from({ length: LUT_SIZE }, (_, i) => {
    const x = (i / (LUT_SIZE - 1)) * (n - 1);
    const k = Math.min(n - 2, Math.floor(x));
    const f = x - k;
    const [r1, g1, b1] = anchors[k];
    const [r2, g2, b2] = anchors[k + 1];
    return [r1 + (r2 - r1) * f, g1 + (g2 - g1) * f, b1 + (b2 - b1) * f] as Rgb;
  });
}

const LUTS = Object.fromEntries(COLORMAP_KEYS.map((k) => [k, buildLut(ANCHORS[k])])) as Record<ColormapKey, Rgb[]>;

/** t (0〜1、範囲外は端の色) の色 */
export function sampleColormap(key: ColormapKey, t: number): Rgb {
  const c = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0));
  return LUTS[key][Math.min(LUT_SIZE - 1, Math.floor(c * (LUT_SIZE - 1) + 0.5))];
}

export function colormapCss(key: ColormapKey, t: number): string {
  const [r, g, b] = sampleColormap(key, t);
  return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
}

/** WebGL の配色テクスチャ (LUT_SIZE × 1、RGBA) */
export function colormapRgba(key: ColormapKey): Uint8Array {
  const out = new Uint8Array(LUT_SIZE * 4);
  LUTS[key].forEach(([r, g, b], i) => {
    out[4 * i] = Math.round(r);
    out[4 * i + 1] = Math.round(g);
    out[4 * i + 2] = Math.round(b);
    out[4 * i + 3] = 255;
  });
  return out;
}
