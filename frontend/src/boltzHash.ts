// electron_processes (断面積プロセス一覧) の sha256(JSON) を計算するユーティリティ
// (prompts/118 の鮮度警告用: 「断面積が変更されたら boltz テーブルを再生成させる」判定に使う)。
//
// backend/es_sim/boltz.py の _hash_processes と bit-exact に一致させる必要がある:
//
//   payload = [p.model_dump() for p in processes]
//   blob = json.dumps(payload, sort_keys=True).encode("utf-8")
//   sha256(blob).hexdigest()
//
// Python の json.dumps は JS の JSON.stringify と以下の点で異なるため、素直に
// JSON.stringify を使うと同じ processes でもハッシュが一致しない。そのため JSON 文字列を
// フィールドごとに手組みする:
//
//  1. 区切り文字が ", " / ": " (スペース入り。JS の既定は "," / ":" でスペース無し)
//  2. sort_keys=True でオブジェクトキーがアルファベット順に並ぶ。XsProcess は
//     energy_ev/kind/label/mass_ratio/sigma_m2/threshold_ev の6キー固定なので、
//     動的ソートはせずこの順で直接組み立てる
//  3. float は Python の repr (=json.dumps の float 表現、shortest round-trip) 形式。
//     整数値でも必ず小数点が付く ("3.0" 等) 一方、JS の Number は int/float を区別しないため
//     素の String(v) では "3" になってしまう。さらに固定小数点⇔指数表記の切替しきい値・
//     指数の桁数も JS の Number.toString()/toExponential() とは異なる
//     (Python: 指数が -4 未満または 16 以上で指数表記、指数部は符号付き最低2桁)
//  4. ensure_ascii=True (json.dumps の既定) により非ASCII文字は \uXXXX にエスケープされる
//     (JSON.stringify は既定でエスケープしない)
//
// electron_processes が空 (未読込 = eduPIC Ar 解析式を既定使用) の場合、backend は実際には
// pic1d_presets.edupic_ar_processes() が生成する断面積テーブルでハッシュを計算する
// (server.py _run_boltz_session 参照)。この解析式 (Phelps & Petrovic の経験式) を
// フロントで bit-exact に再現するには数値計算そのものを移植する必要があり、鮮度警告という
// 補助機能に対して過大なコストになるため、呼び出し側 (BoltzSection.tsx) は
// electron_processes が空のときはハッシュ比較自体を行わない (既定断面積はソースコードを
// 変更しない限り変化しないため、鮮度が崩れる実害はそもそも無い)。

import type { XsProcess } from "./types";

/** Python の float repr (= json.dumps が使う書式) を再現する。 */
export function pyFloatRepr(v: number): string {
  if (Number.isNaN(v)) return "NaN";
  if (v === Infinity) return "Infinity";
  if (v === -Infinity) return "-Infinity";
  if (Object.is(v, -0)) return "-0.0";
  if (v === 0) return "0.0";

  const neg = v < 0;
  const av = Math.abs(v);
  // toExponential() (引数なし) は一意に元の値へ戻る最短の仮数・指数を返す (V8 等も
  // shortest round-trip アルゴリズムを使うため、Python の repr と同じ「最短表現」の
  // 数字列が得られる。差分は下で行う書式変換だけで吸収できる)
  const expStr = av.toExponential();
  const m = expStr.match(/^(\d)(?:\.(\d+))?e([+-]\d+)$/);
  if (!m) return String(v); // 到達しない想定 (NaN/Infinity は上で処理済み) の防御的フォールバック
  const digits = m[1] + (m[2] ?? "");
  const exp = parseInt(m[3], 10); // value = 0.digits[0]digits[1]... * 10^(exp+1) 相当 (先頭1桁基準)

  let out: string;
  if (exp >= -4 && exp < 16) {
    // 固定小数点表記 (Python の repr がこの範囲で指数表記に切り替えないしきい値と同じ)
    if (exp >= 0) {
      out =
        digits.length <= exp + 1
          ? digits + "0".repeat(exp + 1 - digits.length) + ".0"
          : digits.slice(0, exp + 1) + "." + digits.slice(exp + 1);
    } else {
      out = "0." + "0".repeat(-exp - 1) + digits;
    }
  } else {
    // 科学的記法。指数部は符号付き・最低2桁 (Python: "1e-05" 等)
    const mantissa = digits.length > 1 ? digits[0] + "." + digits.slice(1) : digits[0];
    const sign = exp < 0 ? "-" : "+";
    out = `${mantissa}e${sign}${Math.abs(exp).toString().padStart(2, "0")}`;
  }
  return neg ? "-" + out : out;
}

/**
 * JSON.stringify (JS) は非ASCII文字をエスケープしないため、Python の ensure_ascii=True
 * 相当の追加エスケープを UTF-16 コード単位ごとに行う (基本多言語面外の文字もサロゲート
 * ペアのまま2つの \uXXXX になり、Python 側の挙動と一致する)。
 */
function pyJsonString(s: string): string {
  const base = JSON.stringify(s);
  let out = "";
  for (let i = 0; i < base.length; i++) {
    const code = base.charCodeAt(i);
    out += code > 0x7e ? "\\u" + code.toString(16).padStart(4, "0") : base[i];
  }
  return out;
}

function numberArrayJson(arr: number[]): string {
  return "[" + arr.map(pyFloatRepr).join(", ") + "]";
}

// 1プロセス分の JSON (キーはアルファベット順固定: energy_ev/kind/label/mass_ratio/sigma_m2/threshold_ev。
// pydantic の XsProcess.model_dump() を sort_keys=True で dumps した場合と同じ並び)
function xsProcessJson(p: XsProcess): string {
  return (
    "{" +
    `"energy_ev": ${numberArrayJson(p.energy_ev)}, ` +
    `"kind": ${pyJsonString(p.kind)}, ` +
    `"label": ${pyJsonString(p.label)}, ` +
    `"mass_ratio": ${pyFloatRepr(p.mass_ratio)}, ` +
    `"sigma_m2": ${numberArrayJson(p.sigma_m2)}, ` +
    `"threshold_ev": ${pyFloatRepr(p.threshold_ev)}` +
    "}"
  );
}

/** backend/es_sim/boltz.py _hash_processes の payload と bit-exact に一致する JSON 文字列。 */
export function processesHashJson(processes: XsProcess[]): string {
  return "[" + processes.map(xsProcessJson).join(", ") + "]";
}

/** sha256(JSON) の16進文字列 (Web Crypto、非同期)。呼び出し側は useEffect + state で扱うこと。 */
export async function computeProcessesHash(processes: XsProcess[]): Promise<string> {
  const json = processesHashJson(processes);
  const bytes = new TextEncoder().encode(json);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}
