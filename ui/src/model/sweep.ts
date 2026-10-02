// パラメータスイープの組み立て (v1 SweepPanel の移植): 対象パラメータの候補、実行するモジュールの判定、
// 値の列 (列挙・範囲・対数)。設定は文書の ui.sweep に置く (実行は P6d)。

import type { TFunction } from "i18next";
import { edgeLabel } from "../tree/treeModel";
import { rfComponents } from "../util/waveform";
import { paramsOf, paramValues } from "./params";
import type { Project } from "./project";

export interface SweepCandidate {
  label: string;
  path: string;
}

export type SweepModule = "pic" | "pic1d" | "fluid1d" | "fluid2d";

export interface SweepSettings {
  param_path: string;
  /** パラメータをスイープするときに実行するモジュール (P7f。ほかのパスはパスから決まる) */
  module?: SweepModule;
  mode: "list" | "range";
  list_text: string;
  start: number;
  end: number;
  count: number;
  log: boolean;
  parallel: number;
}

export const DEFAULT_SWEEP: SweepSettings = { param_path: "", mode: "list", list_text: "", start: 1, end: 10, count: 5, log: false, parallel: 1 };

/** 現在のプロジェクトから候補を作る (backend はパスの意味を解釈しないので、実際の JSON の形に合わせる) */
export function buildSweepCandidates(p: Project, t: TFunction): SweepCandidate[] {
  const out: SweepCandidate[] = [];
  const tt = t as unknown as (k: string, o?: Record<string, unknown>) => string;
  // パラメータ (P7f): 式を付けた欄ごと計算し直す
  for (const v of paramsOf(p).vars) out.push({ label: tt("sweep.cand.param", { name: v.name }), path: `${PARAM_PREFIX}${v.name}` });
  p.geometry.boundaries.forEach((b, i) => {
    if (b.type !== "dirichlet") return;
    const e = b.edges.length ? edgeLabel(p, b.edges[0], t) : `#${i}`;
    out.push({ label: tt("sweep.cand.voltage", { name: e }), path: `geometry.boundaries.${i}.voltage` });
    out.push({ label: tt("sweep.cand.gamma", { name: e }), path: `geometry.boundaries.${i}.see_gamma` });
    const comps = rfComponents(b.voltage_rf);
    comps.forEach((_, j) => {
      const prefix = Array.isArray(b.voltage_rf) ? `geometry.boundaries.${i}.voltage_rf.${j}` : `geometry.boundaries.${i}.voltage_rf`;
      const suffix = comps.length > 1 ? ` (${j + 1})` : "";
      out.push({ label: tt("sweep.cand.rfAmp", { name: e }) + suffix, path: `${prefix}.amplitude` });
      out.push({ label: tt("sweep.cand.rfFreq", { name: e }) + suffix, path: `${prefix}.freq_hz` });
    });
  });
  p.geometry.regions.forEach((r, i) => {
    if (r.type !== "conductor") return;
    out.push({ label: tt("sweep.cand.voltage", { name: r.id }), path: `geometry.regions.${i}.voltage` });
    out.push({ label: tt("sweep.cand.gamma", { name: r.id }), path: `geometry.regions.${i}.see_gamma` });
  });
  if (p.b_field) for (const c of ["bx", "by", "bz"]) out.push({ label: tt("sweep.cand.b", { c: c.toUpperCase() }), path: `b_field.${c}` });
  if (p.dsmc) out.push({ label: tt("sweep.cand.dsmcPressure"), path: "dsmc.init_pressure_pa" });
  const pic = p.pic as Record<string, unknown> | null | undefined;
  if (pic) {
    out.push({ label: tt("sweep.cand.picNMacro"), path: "pic.n_macro" });
    out.push({ label: tt("sweep.cand.picSee"), path: "pic.see_energy_ev" });
    if (pic.dt !== null && pic.dt !== undefined) out.push({ label: tt("sweep.cand.picDt"), path: "pic.dt" });
  }
  const pic1d = p.pic1d as Record<string, unknown> | null | undefined;
  if (pic1d) {
    out.push({ label: tt("sweep.cand.p1Gap"), path: "pic1d.gap_m" });
    out.push({ label: tt("sweep.cand.p1Density"), path: "pic1d.init_density_m3" });
    out.push({ label: tt("sweep.cand.p1NMacro"), path: "pic1d.n_macro" });
    if (pic1d.mcc) out.push({ label: tt("sweep.cand.p1Pressure"), path: "pic1d.mcc.gas.pressure_pa" });
    for (const side of ["left", "right"] as const) {
      const el = (pic1d[side] ?? {}) as Record<string, unknown>;
      const s = tt(`sweep.side.${side}`);
      out.push({ label: tt("sweep.cand.p1Dc", { side: s }), path: `pic1d.${side}.v_dc` });
      if (el.voltage_rf) {
        out.push({ label: tt("sweep.cand.p1RfAmp", { side: s }), path: `pic1d.${side}.voltage_rf.0.amplitude` });
        out.push({ label: tt("sweep.cand.p1RfFreq", { side: s }), path: `pic1d.${side}.voltage_rf.0.freq_hz` });
      }
      if (el.fn) {
        out.push({ label: tt("sweep.cand.p1FnBeta", { side: s }), path: `pic1d.${side}.fn.beta` });
        out.push({ label: tt("sweep.cand.p1FnPhi", { side: s }), path: `pic1d.${side}.fn.phi_ev` });
      }
    }
  }
  const f1 = p.fluid1d as Record<string, unknown> | null | undefined;
  if (f1) {
    out.push({ label: tt("sweep.cand.f1Gap"), path: "fluid1d.gap_m" });
    out.push({ label: tt("sweep.cand.f1Density"), path: "fluid1d.init_density_m3" });
    out.push({ label: tt("sweep.cand.f1Pressure"), path: "fluid1d.gas_pressure_pa" });
    for (const side of ["left", "right"] as const) {
      const el = (f1[side] ?? {}) as Record<string, unknown>;
      const s = tt(`sweep.side.${side}`);
      out.push({ label: tt("sweep.cand.f1Dc", { side: s }), path: `fluid1d.${side}.v_dc` });
      if (el.voltage_rf) {
        out.push({ label: tt("sweep.cand.f1RfAmp", { side: s }), path: `fluid1d.${side}.voltage_rf.0.amplitude` });
        out.push({ label: tt("sweep.cand.f1RfFreq", { side: s }), path: `fluid1d.${side}.voltage_rf.0.freq_hz` });
      }
      // 阻止コンデンサ (prompts/134)。2D の電極のコンデンサはジオメトリのパス (PIC で実行) になるので、
      // 流体 2D でスイープするときはパラメータに束縛して流体 2D を選ぶ
      if (el.blocking_capacitor) {
        out.push({ label: tt("sweep.cand.f1Cap", { side: s }), path: `fluid1d.${side}.blocking_capacitor.capacitance` });
        out.push({ label: tt("sweep.cand.f1Bias", { side: s }), path: `fluid1d.${side}.blocking_capacitor.initial_bias_v` });
      }
    }
  }
  if (p.fluid2d) {
    out.push({ label: tt("sweep.cand.f2Density"), path: "fluid2d.init_density_m3" });
    out.push({ label: tt("sweep.cand.f2Pressure"), path: "fluid2d.gas_pressure_pa" });
  }
  return out;
}

/** パラメータのパス (params.<名前>)。backend の sweep.PARAM_PREFIX と同じ */
export const PARAM_PREFIX = "params.";

export const isParamPath = (path: string) => path.startsWith(PARAM_PREFIX);

/**
 * 実行するモジュール: パラメータのスイープは選んだもの (無ければ式を付けた欄の最初の 1D・流体のブロック、
 * それも無ければ pic)、ほかはパスから
 */
export function sweepModule(s: Pick<SweepSettings, "param_path" | "module">, p: Project): SweepModule {
  if (!isParamPath(s.param_path)) return sweepModuleForPath(s.param_path);
  if (s.module) return s.module;
  for (const b of paramsOf(p).bindings) {
    const head = b.path[0];
    if (head === "pic1d" || head === "fluid1d" || head === "fluid2d") return head;
  }
  return "pic";
}

/** パスから実行するモジュール (backend の resolve_sweep_module と同じ規則。ジオメトリのパスは pic) */
export function sweepModuleForPath(path: string): SweepModule {
  if (path.startsWith("pic1d.")) return "pic1d";
  if (path.startsWith("fluid1d.")) return "fluid1d";
  if (path.startsWith("fluid2d.")) return "fluid2d";
  return "pic";
}

/** "50, 100, 1.5e2" → 値の列 (数値にならない項目は無視) */
export function parseListValues(text: string): number[] {
  return text
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s !== "")
    .map(Number)
    .filter((n) => Number.isFinite(n));
}

/** 開始・終了・点数の等分 (log は対数等分、開始・終了とも正が必要) */
export function rangeValues(start: number, end: number, count: number, log: boolean): number[] {
  const n = Math.max(1, Math.round(count));
  if (n === 1) return [start];
  if (log) {
    if (!(start > 0) || !(end > 0)) return [];
    const ls = Math.log(start);
    const le = Math.log(end);
    return Array.from({ length: n }, (_, i) => Math.exp(ls + ((le - ls) * i) / (n - 1)));
  }
  return Array.from({ length: n }, (_, i) => start + ((end - start) * i) / (n - 1));
}

export function sweepValues(s: SweepSettings): number[] {
  return s.mode === "list" ? parseListValues(s.list_text) : rangeValues(s.start, s.end, s.count, s.log);
}

/**
 * スイープで送る文書 (写し) を整える (v1 の runSweepStart と同じ): PIC 1D・流体 1D の電極の voltage_rf が 1 成分
 * (オブジェクト) なら配列に包み (候補のパスは voltage_rf.0.* の形)、対象のキーが無ければ親があるときに限り 0 で補う
 * (see_gamma のような省略できる項目)。元の文書は変えない。
 */
export function prepareSweepProject(p: Project, path: string): Project {
  const q = structuredClone(p) as unknown as Record<string, unknown>;
  // パラメータは backend が式ごと計算し直す (欄を 0 で補う必要はない)
  if (isParamPath(path)) return q as unknown as Project;
  for (const mod of ["pic1d", "fluid1d"]) {
    const blk = q[mod] as Record<string, Record<string, unknown> | undefined> | null | undefined;
    if (!blk) continue;
    for (const side of ["left", "right"]) {
      const e = blk[side];
      if (e && e.voltage_rf && !Array.isArray(e.voltage_rf)) e.voltage_rf = [e.voltage_rf];
    }
  }
  const toks = path.split(".");
  let cur: unknown = q;
  for (const tok of toks.slice(0, -1)) {
    if (cur === null || cur === undefined) break;
    cur = Array.isArray(cur) ? cur[Number(tok)] : (cur as Record<string, unknown>)[tok];
  }
  const last = toks[toks.length - 1];
  if (path && cur !== null && typeof cur === "object" && !Array.isArray(cur)) {
    const rec = cur as Record<string, unknown>;
    if (typeof rec[last] !== "number") rec[last] = 0;
  }
  return q as unknown as Project;
}

/** ドット区切りのパスの現在値 (数値でなければ undefined)。params.<名前> はパラメータの値 */
export function valueAtPath(obj: unknown, path: string): number | undefined {
  if (isParamPath(path)) return paramValues(obj as Project).values[path.slice(PARAM_PREFIX.length)];
  let cur: unknown = obj;
  for (const tok of path.split(".")) {
    if (cur === null || cur === undefined) return undefined;
    if (Array.isArray(cur)) {
      const idx = Number(tok);
      if (!Number.isInteger(idx)) return undefined;
      cur = cur[idx];
    } else if (typeof cur === "object") cur = (cur as Record<string, unknown>)[tok];
    else return undefined;
  }
  return typeof cur === "number" ? cur : undefined;
}
