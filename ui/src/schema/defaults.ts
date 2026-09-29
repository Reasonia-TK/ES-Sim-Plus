// スタディ・設定ブロックを有効にしたときの初期値 (v1 frontend の DEFAULT_* と同じ値)。
// 必須で既定値の無い項目 (初期密度・ガス圧など) を含むので、スキーマの既定値だけでは作れない。

import { polygonBounds, type Project } from "../model/project";

export const DEFAULT_VOLTAGE_RF = { amplitude: 100.0, freq_hz: 13.56e6, phase_deg: 0.0 };
export const DEFAULT_VOLTAGE_RF_2ND = { amplitude: 100.0, freq_hz: 2e6, phase_deg: 0.0 };

export function defaultEmitter(p: Project) {
  const b = polygonBounds(p.geometry.domain.polygon);
  const w = b.x1 - b.x0;
  const h = b.y1 - b.y0;
  return {
    kind: "line",
    p1: [b.x0 + 0.2 * w, b.y0 + 0.4 * h],
    p2: [b.x0 + 0.2 * w, b.y0 + 0.6 * h],
    n: 50,
    energy_ev: 1.0,
    direction_deg: 0,
    spread_deg: 0,
  };
}

export function defaultParticles(p: Project) {
  return { species: { preset: "electron" }, emitter: defaultEmitter(p), dt: null, n_steps: 2000, save_every: 10 };
}

export const DEFAULT_PIC = {
  initial_plasma: null,
  injection: null,
  n_macro: 20000,
  dt: null,
  n_steps: 2000,
  frame_every: 20,
  mcc: null,
  see_energy_ev: 2.0,
  avg_steps: null,
  phase_bins: 40,
  threads: 0,
};

export const DEFAULT_INITIAL_PLASMA = { density: 1.0e14, te_ev: 2.0, ti_ev: 0.03, ion_mass_amu: 40.0, immobile_ions: false, seed: 0 };

export const DEFAULT_MCC = {
  gas: { name: "Ar", pressure_pa: 10.0, temperature_k: 300.0 },
  electron_processes: [],
  ion_processes: [],
  seed: 0,
  ionization_split: "half",
  ion_energy_frame: "lab",
};

export const DEFAULT_MERGE = { n_max: 100000, every: 100 };

export const DEFAULT_FN = { edges: [], regions: [], phi_ev: 4.5, beta: 1.0, n: 200, init_energy_ev: 0.1, macro_weight: null, seed: 0 };

export const DEFAULT_FN_1D = { phi_ev: 4.5, beta: 1.0, init_energy_ev: 0.1, macro_weight: null };

export const defaultElectrode = () => ({ v_dc: 0.0, waveforms: [], see_gamma: 0.0 });

export const defaultPic1d = () => ({
  gap_m: 0.02,
  n_cells: 128,
  left: defaultElectrode(),
  right: defaultElectrode(),
  init_density_m3: 1.0e14,
  init_te_ev: 2.0,
  init_ti_ev: 0.03,
  ion_mass_amu: 39.948,
  n_macro: 20000,
  dt: null,
  n_steps: 2000,
  frame_every: 20,
  avg_steps: null,
  phase_bins: 40,
  mcc: null,
  see_energy_ev: 2.0,
  eedf_regions: [],
  wall_iedf_bins: 100,
  seed: 0,
});

export const defaultFluid1d = () => ({
  gap_m: 0.025,
  n_cells: 200,
  left: defaultElectrode(),
  right: defaultElectrode(),
  init_density_m3: 1.0e15,
  init_te_ev: 2.0,
  gas_pressure_pa: 50.0,
  gas_temperature_k: 300.0,
  ion_mass_amu: 39.948,
  mu_i_ref: 1.45e-1,
  n_ref_m3: 3.22e22,
  t_i_ev: 0.026,
  ion_mobility_model: "frost",
  frost_c_td: 150.0,
  electron_processes: [],
  dt: null,
  n_steps: 20000,
  frame_every: 200,
  avg_steps: null,
  phase_bins: 40,
  wall_iedf_bins: 100,
});

export const defaultFluid2d = () => ({
  init_density_m3: 1.0e15,
  init_te_ev: 2.0,
  gas_pressure_pa: 50.0,
  gas_temperature_k: 300.0,
  ion_mass_amu: 39.948,
  mu_i_ref: 1.45e-1,
  n_ref_m3: 3.22e22,
  t_i_ev: 0.026,
  ion_mobility_model: "frost",
  frost_c_td: 150.0,
  electron_processes: [],
  dt: null,
  n_steps: 20000,
  frame_every: 200,
  avg_steps: null,
  phase_bins: 0,
  linear_solver: "iterative",
  threads: 0,
});

export const DEFAULT_DSMC_GAS = { name: "Ar", mass_amu: 39.948, d_ref_m: 4.17e-10, omega: 0.81, t_ref_k: 273.0 };

export const defaultDsmc = () => ({
  gas: { ...DEFAULT_DSMC_GAS },
  boundaries: [],
  wall_temperature_k: 300.0,
  init_pressure_pa: 1.0,
  init_temperature_k: 300.0,
  n_particles: 50000,
  dt: null,
  n_steps: 2000,
  avg_steps: 500,
  seed: 0,
  threads: 1,
  smoothing_passes: 0,
  mesh_scale: 1.0,
});

export const DEFAULT_DSMC_BOUNDARY = { edges: [], p1: null, p2: null, type: "wall", temperature_k: 300.0, pressure_pa: null, flow_sccm: null };

export const defaultTl = () => ({
  radius_m: 0.15,
  gap_m: 0.04,
  sheath_m: 5e-4,
  n_e_m3: 1e16,
  n_s_ratio: 0.4,
  nu_m_hz: 1e8,
  freq_hz: 100e6,
  v0: 100.0,
  n_r: 400,
  n_periods: 200,
  n_fft_periods: 32,
  n_harm: 10,
  dt: null,
  sheath_law: "child",
});

/** Boltzmann 係数の生成条件 (v1 BoltzSection の既定値、保存しない) */
export const DEFAULT_BOLTZ_OPTS = { en_min_td: 0.5, en_max_td: 1000.0, n_points: 32, eps_max_ev: null as number | null, d_eps_ev: 0.25, n_theta: 16 };

/** AMR (mesh.amr) を有効にしたとき */
export const DEFAULT_AMR = { max_level: 1, refine_boundaries: true, buffer_cells: 2, blocking_factor: 8, regions: [] };
