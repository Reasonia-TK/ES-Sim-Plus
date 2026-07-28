// 配列の min/max をループで計算するヘルパ。
//
// Math.min(...arr) / Math.max(...arr) は配列要素を「関数の引数」としてスタックに
// 積むため、要素数が数万〜十数万を超えると RangeError: Maximum call stack size
// exceeded で落ちる (上限は WebView/ブラウザ実装依存)。メッシュ節点値・粒子ごとの
// トレース結果・コレクタ生サンプルなど、大規模計算でサイズが伸びる配列には
// 必ずこちらを使うこと。
//
// 空配列の戻り値は Math.min() / Math.max() と同じ (Infinity / -Infinity) にして
// 既存コードの「空なら fallback」分岐がそのまま使えるようにしている。

export function arrayMin(arr: ArrayLike<number>): number {
  let m = Infinity;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (v < m) m = v;
  }
  return m;
}

export function arrayMax(arr: ArrayLike<number>): number {
  let m = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (v > m) m = v;
  }
  return m;
}
