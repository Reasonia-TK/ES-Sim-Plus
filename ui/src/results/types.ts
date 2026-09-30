// 実行の結果の形 (backend es_sim/jobs/runners.py と同じ。v1 の「結果付き保存」(ResultsBundle) の各キーと同じ)。

import type { Point } from "../model/project";
import type { MeshResult } from "../backend/staticApi";

export type { MeshResult };

// ---- PIC (2D) ----

export interface PicDiag {
  t: number;
  ke_e: number;
  ke_i: number;
  fe: number;
  n_e: number;
  n_i: number;
  wall_e: number;
  wall_i: number;
  phi_min: number;
  phi_max: number;
  coll_e?: number;
  ion_events?: number;
  see_events?: number;
  surf_q?: number;
  fn_i?: number;
  fn_events?: number;
  merged?: number;
}

export interface PicStarted {
  dt: number;
  n_steps: number;
  effective_threads?: number;
  step_offset?: number;
  warnings: string[];
  mesh: { nodes: Point[]; triangles: [number, number, number][] };
}

export interface PicFrame {
  step: number;
  t: number;
  /** 節点 */
  phi: number[];
  /** 要素 */
  n_e?: number[];
  n_i?: number[];
  particles: { electron: Point[]; ion: Point[] };
  diag: PicDiag;
  mesh?: { nodes: Point[]; triangles: [number, number, number][] };
  mesh_version?: number;
}

/** 時間平均の場 (e_abs は要素、ほかは節点) */
export interface PicFields {
  phi: number[];
  e_abs: number[];
  n_e: number[];
  n_i: number[];
  te_ev: number[];
  ion_rate: number[];
  avg_steps: number;
}

export interface PicCycle {
  bins: number;
  period_s: number;
  phi: number[][];
  n_e: number[][];
  n_i: number[][];
  e_abs?: number[][];
  te_ev?: number[][];
  ion_rate?: number[][];
  particles: { electron: Point[][]; ion: Point[][] };
}

export interface PicCollectorResult {
  count: number;
  total_weight: number;
  energies_ev: number[];
  angles_deg: number[];
  weights: number[];
  truncated: boolean;
}

export interface EedfResult {
  label: string;
  e_centers: number[];
  f: number[];
  mean_energy_ev: number;
  t_eff_ev: number;
  total_weight: number;
  overflow_frac: number;
  n_samples: number;
}

export interface PicResult {
  started: PicStarted;
  frame: PicFrame | null;
  history: PicDiag[];
  fields: PicFields | null;
  cycle: PicCycle | null;
  collectors: PicCollectorResult[];
  eedf: EedfResult[];
  elapsed_s?: number;
  timing?: Record<string, number>;
  regrids?: unknown[];
}

// ---- 流体 2D ----

export interface Fluid2dStarted {
  n_steps: number;
  step_offset: number;
  dt: number;
  warnings: string[];
  effective_threads?: number;
  mesh?: { nodes: Point[]; triangles: [number, number, number][] };
}

export interface Fluid2dFrame {
  step: number;
  t: number;
  phi: number[];
  n_e: number[];
  n_i: number[];
  t_e: number[];
  counts: Record<string, number>;
  elapsed_s: number;
}

export interface Fluid2dFields {
  phi: number[];
  e_abs: number[];
  n_e: number[];
  n_i: number[];
  t_e: number[];
  ionization: number[];
  avg_steps: number;
}

export interface Fluid2dCycle {
  bins: number;
  freq_hz: number;
  phi: number[][];
  n_e: number[][];
  n_i: number[][];
  t_e: number[][];
}

export interface Fluid2dResult {
  history: Record<string, number[]>;
  fields: Fluid2dFields | null;
  cycle: Fluid2dCycle | null;
  walls: { electron: number; ion: number };
  gen_total: number;
  elapsed_s: number;
  timing: Record<string, number>;
  settings: Record<string, unknown>;
  mesh?: { nodes: Point[]; triangles: [number, number, number][] };
}

// ---- DSMC ----

export interface DsmcFrame {
  step: number;
  n_steps: number;
  n_particles: number;
  particles: Point[];
}

export interface DsmcResult {
  mesh: MeshResult;
  n: number[];
  t: number[];
  u: [number, number][];
  p: number[];
  n_particles: number;
  macro_weight: number;
  dt: number;
  inflow: number;
  outflow: number;
  elapsed_s: number;
  timing: Record<string, number>;
}

// ---- 粒子軌道 ----

export interface TraceResult {
  trajectories: Point[][];
  status: ("absorbed" | "alive")[];
  tof: (number | null)[];
  final_energy_ev: number[];
  final_angle_deg: number[];
  dt: number;
  currents?: number[] | null;
  fn_current?: number | null;
}

// ---- PIC 1D・流体 1D ----

export interface Profiles1d {
  x: number[];
  phi: number[];
  e: number[];
  n_e: number[];
  n_i: number[];
  t_e: number[];
  ionization: number[];
  avg_steps: number;
}

export interface Sheath1d {
  left_s: number | null;
  right_s: number | null;
}

export interface Cycle1d {
  bins: number;
  freq_hz: number;
  phi: number[][];
  n_e: number[][];
  n_i: number[][];
  t_e?: number[][];
  sheath?: { s_left: (number | null)[]; s_right: (number | null)[] };
}

export interface SheathFft {
  df_hz: number;
  freq_hz: number[];
  amp_left: number[];
  amp_right: number[];
  mean_left: number | null;
  mean_right: number | null;
  n_samples: number;
  f0_hz: number | null;
}

export interface SheathTs {
  t: number[];
  s_left: (number | null)[];
  s_right: (number | null)[];
}

export interface WallIedfSide {
  e_centers: number[];
  f: number[];
  mean_energy_ev: number;
  total_weight: number;
  n_samples: number;
  model?: string;
}

export interface WallCounts {
  electron: number;
  ion: number;
}

export interface Pic1dResult {
  history: Record<string, number[]>;
  profiles: Profiles1d | null;
  sheath?: Sheath1d | null;
  cycle: Cycle1d | null;
  sheath_fft?: SheathFft | null;
  sheath_ts?: SheathTs | null;
  eedf: EedfResult[];
  wall_iedf?: { left: WallIedfSide; right: WallIedfSide } | null;
  walls: { left: WallCounts; right: WallCounts };
  fn: { left: { j_avg: number; total_w: number } | null; right: { j_avg: number; total_w: number } | null } | null;
  elapsed_s: number;
  timing: Record<string, number>;
  settings: Record<string, unknown>;
}

export interface Fluid1dResult {
  history: Record<string, number[]>;
  profiles: Profiles1d | null;
  sheath: Sheath1d | null;
  cycle: Cycle1d | null;
  wall_iedf?: { left: WallIedfSide; right: WallIedfSide } | null;
  walls: { left: WallCounts; right: WallCounts };
  gen_total: number;
  elapsed_s: number;
  timing: Record<string, number>;
  settings: Record<string, unknown>;
}

export interface Frame1d {
  step: number;
  t: number;
  phi: number[];
  n_e: number[];
  n_i: number[];
  t_e?: number[];
  counts: Record<string, number>;
  elapsed_s: number;
  sample?: { x: number[]; vx: number[] };
}

// ---- VHF 定在波 ----

export interface TlResult {
  r: number[];
  probe_r: { center: number; mid: number; edge: number };
  harmonics: { n: number[]; v: number[][]; j: number[][] };
  lambda_eff: { lambda_m: number | null; lambda0_m: number; ratio: number | null };
  power: { p: number[]; max_over_min: number | null; area_weighted_std_over_mean: number | null };
  feed_power: number;
  v_probe: { t: number[]; center: number[]; mid: number[]; edge: number[] };
  spectrum_probe: { freq_hz: number[]; center: number[]; mid: number[]; edge: number[] };
  warnings: string[];
  dt: number;
  n_steps: number;
  settings: Record<string, unknown>;
}

// ---- Boltzmann 係数 ----

export interface BoltzTable {
  en_td: number[];
  mean_energy_ev: number[];
  mobility_n: number[];
  k_ion: number[];
  k_exc: number[];
  e_ion_ev: number[];
  e_exc_ev: number[];
  eedf_eps_ev: number[];
  eedf: number[][];
  source_hash: string;
  opts: Record<string, unknown>;
  warnings: string[];
}
