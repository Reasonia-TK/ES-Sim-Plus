// 電子の断面積プロセスの sha256(JSON) (Boltzmann 係数の表が断面積と合っているかの判定)。
// backend/es_sim/boltz.py の _hash_processes と bit-exact に一致させる (v1 frontend/src/boltzHash.ts の移植):
//   payload = [p.model_dump() for p in processes]
//   sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
// Python の json.dumps は区切りに空白が入る・float を repr で書く (整数値でも "3.0")・非 ASCII を \uXXXX に
// するので、JSON.stringify ではなく項目ごとに組み立てる。

export interface XsProcessLike {
  kind: string;
  label: string;
  threshold_ev: number;
  mass_ratio: number;
  energy_ev: number[];
  sigma_m2: number[];
}

/** Python の float の repr (json.dumps の書式) */
export function pyFloatRepr(v: number): string {
  if (Number.isNaN(v)) return "NaN";
  if (v === Infinity) return "Infinity";
  if (v === -Infinity) return "-Infinity";
  if (Object.is(v, -0)) return "-0.0";
  if (v === 0) return "0.0";
  const neg = v < 0;
  const av = Math.abs(v);
  // toExponential() は往復で元に戻る最短の桁 (Python の repr と同じ桁の列)
  const m = av.toExponential().match(/^(\d)(?:\.(\d+))?e([+-]\d+)$/);
  if (!m) return String(v);
  const digits = m[1] + (m[2] ?? "");
  const exp = parseInt(m[3], 10);
  let out: string;
  if (exp >= -4 && exp < 16) {
    if (exp >= 0) {
      out =
        digits.length <= exp + 1
          ? digits + "0".repeat(exp + 1 - digits.length) + ".0"
          : digits.slice(0, exp + 1) + "." + digits.slice(exp + 1);
    } else {
      out = "0." + "0".repeat(-exp - 1) + digits;
    }
  } else {
    const mantissa = digits.length > 1 ? digits[0] + "." + digits.slice(1) : digits[0];
    out = `${mantissa}e${exp < 0 ? "-" : "+"}${Math.abs(exp).toString().padStart(2, "0")}`;
  }
  return neg ? "-" + out : out;
}

/** Python の ensure_ascii=True と同じ文字列の JSON (非 ASCII は UTF-16 単位で \uXXXX) */
function pyJsonString(s: string): string {
  const base = JSON.stringify(s);
  let out = "";
  for (let i = 0; i < base.length; i++) {
    const code = base.charCodeAt(i);
    out += code > 0x7e ? "\\u" + code.toString(16).padStart(4, "0") : base[i];
  }
  return out;
}

const numbers = (a: number[]) => "[" + a.map(pyFloatRepr).join(", ") + "]";

// キーはアルファベット順 (sort_keys=True): energy_ev / kind / label / mass_ratio / sigma_m2 / threshold_ev
function processJson(p: XsProcessLike): string {
  return (
    "{" +
    `"energy_ev": ${numbers(p.energy_ev)}, ` +
    `"kind": ${pyJsonString(p.kind)}, ` +
    `"label": ${pyJsonString(p.label ?? "")}, ` +
    `"mass_ratio": ${pyFloatRepr(p.mass_ratio ?? 0)}, ` +
    `"sigma_m2": ${numbers(p.sigma_m2)}, ` +
    `"threshold_ev": ${pyFloatRepr(p.threshold_ev ?? 0)}` +
    "}"
  );
}

export function processesHashJson(processes: XsProcessLike[]): string {
  return "[" + processes.map(processJson).join(", ") + "]";
}

export async function computeProcessesHash(processes: XsProcessLike[]): Promise<string> {
  const bytes = new TextEncoder().encode(processesHashJson(processes));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}
