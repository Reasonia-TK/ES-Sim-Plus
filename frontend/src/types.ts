// backend/es_sim/schema.py と手動同期 (将来 openapi.json から自動生成に移行)

export type Point = [number, number];

export type RegionType = "conductor" | "dielectric" | "charge";

// 線分近傍のローカルメッシュサイズ (prompts/90)。gmsh の Distance+Threshold フィールドで
// 線分近傍を size まで細分化する (非構造メッシュのみ有効)。dist_in/dist_out は省略時
// backend 側で自動決定 (2·size / 8·size) するため、フロントの UI では出さない
export interface EdgeMeshSize {
  p1: Point;
  p2: Point;
  size: number;
  dist_in?: number | null;
  dist_out?: number | null;
}

// v2 直交格子 (mesh.mode="cartesian") の局所細分化 (AMR、prompts/121)。レベル l の格子幅は
// size/2^l。細分化の単位は blocking_factor × blocking_factor セルのブロックで、細かいレベルは
// 粗いレベルの内側に 1 ブロックの緩衝帯を持って入る (隣接セルのレベル差は高々 1)
export interface AmrRegion {
  p1: Point; // 矩形の対角 2 点 [m]
  p2: Point;
  level: number; // この矩形を少なくともこのレベルまで細分化する (1〜6)
}

export interface AmrSettings {
  max_level: number; // 0 = 細分化なし
  refine_boundaries?: boolean; // 導体・誘電体の境界近傍を max_level まで細分化 (既定 true)
  buffer_cells?: number; // 境界からのセル数 (各レベルのセル単位、既定 2)
  blocking_factor?: number; // ブロックの一辺のセル数 (既定 8)
  regions?: AmrRegion[];
}

// 円領域のパラメトリック形状 (中心+半径)。メッシュ生成時にバックエンド側で多角形化する
export interface CircleShape {
  kind: "circle";
  center: Point;
  radius: number;
}

// RF重畳電圧。V(t) = voltage(直流分) + amplitude * sin(2π f t + phase) (PICのみで使用。/solve は voltage のみ)
export interface VoltageRf {
  amplitude: number;  // [V]
  freq_hz: number;    // [Hz]
  phase_deg: number;  // [deg]
}

// voltage_rf は単一成分または複数成分 (デュアル周波数等) のリストを受け付ける。
// V(t) = voltage + Σ_k amplitude_k * sin(2π freq_hz_k t + phase_deg_k)
// 位相分解アニメーションの基本周波数は全成分の最小周波数になる (バックエンド側処理)。
export function rfComponents(rf: VoltageRf | VoltageRf[] | null | undefined): VoltageRf[] {
  if (!rf) return [];
  return Array.isArray(rf) ? rf : [rf];
}

// CSV インポート波形 (prompts/73)。V(t) = interp(frac(t·freq_hz), phase, v) として
// 指定周波数の1周期でループ再生する (voltage_rf と併用可、PICのみ使用)。
// phase は CSV の時間を [0, 1) に正規化した位相 (昇順、取り込み時にフロントで計算)
export interface VoltageWaveform {
  freq_hz: number;    // [Hz]
  phase: number[];    // 正規化位相 [0, 1) (昇順)
  v: number[];        // 対応する電圧 [V]
}

export interface Region {
  id: string;
  type: RegionType;
  // polygon / shape はどちらか一方のみ指定する
  polygon?: Point[];
  shape?: CircleShape;
  voltage?: number; // conductor
  eps_r?: number;   // dielectric
  rho?: number;     // charge
  voltage_rf?: VoltageRf | VoltageRf[]; // conductor: RF重畳 (未指定なら直流のみ。複数成分でデュアル周波数)
  // conductor: CSV波形 (prompts/73)。スキーマ上のみ対応、UIは境界条件辺のみ (下記 DirichletBC 参照)
  voltage_waveform?: VoltageWaveform;
  see_gamma?: number; // conductor: 二次電子放出係数 γ (未指定/0 で無効)
}

// 表示/ヒットテスト用の輪郭ポリゴンを返す。
// polygon 領域はそのまま、circle (shape) 領域は64分割の近似ポリゴンを返す。
// (実際の描画・ヒットテストは真円で行うべき箇所も多いが、
//  「とりあえず輪郭が欲しい」用途 — 例: 初期表示のフィット計算等 — で安全に使えるヘルパー)
export function regionOutline(region: Region): Point[] {
  if (region.shape) {
    const { center, radius } = region.shape;
    const n = 64;
    return Array.from({ length: n }, (_, i) => {
      const a = (i / n) * Math.PI * 2;
      return [center[0] + radius * Math.cos(a), center[1] + radius * Math.sin(a)] as Point;
    });
  }
  return region.polygon ?? [];
}

// Dirichlet境界: 電圧を固定する (voltage/voltage_rf/see_gamma はこのタイプのみ有効)
export interface DirichletBC {
  edges: number[];
  type: "dirichlet";
  voltage: number;
  voltage_rf?: VoltageRf | VoltageRf[]; // RF重畳 (未指定なら直流のみ。複数成分でデュアル周波数)
  // CSV インポート波形 (prompts/73)。voltage_rf と併用可 (V(t) = voltage + Σ RF + V_wf(t))
  voltage_waveform?: VoltageWaveform;
  see_gamma?: number; // 二次電子放出係数 γ (未指定/0 で無効)
}

// 対称境界 (Neumann + 粒子鏡面反射): 場は自然境界、粒子はこの辺で反射する
export interface SymmetryBC {
  edges: number[];
  type: "symmetry";
}

// 周期境界: edges はちょうど2本 (平行・同長の対辺) を指定する。
// 場は対辺の節点を同一視して解き、粒子は対辺を越えたら反対側へラップする
export interface PeriodicBC {
  edges: number[];
  type: "periodic";
}

export type BoundaryCondition = DirichletBC | SymmetryBC | PeriodicBC;

// 境界条件セレクトの4択 (フロント内部の表示用タイプ。schema上の type と概ね対応するが
// "neumann" はBCエントリが無い状態=自然境界を表す仮想的な値)
export type EdgeBcType = "neumann" | "dirichlet" | "symmetry" | "periodic";

export interface Geometry {
  domain: { polygon: Point[] };
  regions: Region[];
  boundaries: BoundaryCondition[];
}

// 粒子種。custom の場合のみ q [C] / m [kg] を持つ
export interface Species {
  preset: "electron" | "proton" | "custom";
  q?: number; // custom時のみ [C]
  m?: number; // custom時のみ [kg]
}

// 粒子エミッタ。line: p1-p2 線分上に等間隔配置、point: p1 のみ使用 (全粒子同位置)
export interface Emitter {
  kind: "line" | "point";
  p1: Point;
  p2: Point; // point の場合は未使用 (p1 のみ使用)
  n: number;             // 粒子数
  energy_ev: number;     // 初期運動エネルギー [eV] (ドリフトエネルギー。maxwell 時もドリフト成分として有効)
  direction_deg: number; // 射出方向 (x軸から反時計回り、度)
  spread_deg: number;    // 方向の一様分布半角 [度] (等間隔割り振り、乱数不使用。maxwell 時は無視される)
  energy_dist?: "mono" | "maxwell"; // エネルギー分布。未指定は "mono" (従来動作)
  temperature_ev?: number; // maxwell 時の温度 kT [eV] (>0)
  seed?: number;           // maxwell サンプリングの乱数シード (再現性確保)
}

// Fowler–Nordheim (FN) 電界放出源。particles.fn / pic.fn。null/undefined = 無効。
// edges か regions の少なくとも一方が必要 (バックエンドが両方空を 422 で拒否する)
export interface FnEmission {
  edges: number[];        // domain 外周のエッジ番号 (dirichlet 辺から選ぶ)
  regions: string[];      // conductor 領域の id
  phi_ev: number;         // 仕事関数 φ [eV] 既定 4.5
  beta: number;           // 電界増倍係数 β 既定 1.0
  n: number;              // trace 時の放出マクロ粒子総数 既定 200
  init_energy_ev: number; // 放出電子の初期エネルギー [eV] 既定 0.1
  macro_weight?: number | null; // PIC のみ: マクロ重み。null なら初期プラズマと同じ
  seed: number;           // PIC の放出位置乱数シード 既定 0
}

export interface ParticleSettings {
  species: Species;
  emitter: Emitter;
  // FN 電界放出源 (prompts/46)。指定時はエミッタ・粒子種は無視され、電極表面から電子を放出する
  fn?: FnEmission | null;
  dt: number | null; // 秒。null なら自動推定
  n_steps: number;
  save_every: number;
}

// ---- FEM-PIC (フェーズ3、backend/es_sim/schema.py 予定分と手動同期) ----------------

// 初期プラズマ装荷設定。null なら初期装荷なし
export interface InitialPlasma {
  density: number;        // [m^-3] (奥行き1m換算)
  te_ev: number;           // 電子温度 [eV]
  ti_ev: number;           // イオン温度 [eV]
  ion_mass_amu: number;    // イオン質量 [amu] (Ar+ = 40 など)
  immobile_ions: boolean;  // true でイオン固定 (検証用)
  seed: number;            // 乱数シード
}

// エミッタ定常注入。emitter はフェーズ2の ParticleSettings.emitter と同型 (共用する)
export interface PicInjection {
  emitter: Emitter;
  species: "electron" | "ion";
  current_a_per_m: number; // 電流 [A/m] → 毎ステップの実電荷を等分注入
}

// 断面積プロセスの種別。elastic/excitation/ionization は電子用、isotropic/backscat はイオン用
export type XsKind = "elastic" | "excitation" | "ionization" | "isotropic" | "backscat";

// LXCat形式からパース済みの断面積プロセス (プロジェクトJSONにそのまま埋め込む)
export interface XsProcess {
  kind: XsKind;
  label: string;         // PROCESS行等から取得した表示用ラベル
  threshold_ev: number;  // excitation/ionization のみ >0 (elastic/isotropic/backscat は 0)
  mass_ratio: number;    // elastic のみ m/M。無ければ 0
  energy_ev: number[];   // 断面積テーブルのエネルギー軸 [eV] (昇順)
  sigma_m2: number[];    // 断面積テーブル [m^2] (energy_ev と同長)
}

// MCC(モンテカルロ衝突)用の背景ガス設定
export interface McGas {
  name: string;          // 表示用ガス名 (例: "Ar")
  pressure_pa: number;   // 圧力 [Pa]
  temperature_k: number; // ガス温度 [K]
}

// MCC設定。PicSettings.mcc が null なら MCC 無効
export interface McSettings {
  gas: McGas;
  electron_processes: XsProcess[]; // elastic/excitation/ionization
  ion_processes: XsProcess[];      // isotropic/backscat
  seed: number;                    // 乱数シード
  // 電離の余剰エネルギー分配: "half" = 散乱電子と生成電子で等分 (Turner ベンチマーク互換、既定)、
  // "random" = 一様乱数比で分配 (従来動作)。2D PicPanel には UI が無く既定値のみ使うが、
  // 1D (Pic1dPanel) は Turner/eduPIC ベンチマーク再現のため選択UIを持つ (prompts/91)
  ionization_split?: "half" | "random";
  // イオン断面積テーブルの参照エネルギー系: "lab" = 実験室系 (既定)、
  // "com" = 重心系エネルギー (Turner の He+/He データ用)。ionization_split と同様 1D のみ UI あり
  ion_energy_frame?: "com" | "lab";
  // true なら直前に実行した DSMC の定常ガス場 (n・T・u) を背景として使う (prompts/54)。
  // サーバーが保持する DSMC 結果とメッシュが一致している必要がある (未実行/不一致はサーバーがエラー)。
  // 1D (pic1d) は DSMC 連成に未対応 (backend の validator が true を拒否する) ため、
  // Pic1dPanel ではこのフィールドのトグルを出さない (常に未設定=false のまま送る)
  use_dsmc_gas?: boolean;
}

// ---- 1D PIC/MCC (1d3v、backend/es_sim/schema.py Pic1d* と手動同期、prompts/91) --------------
// 2D FEM-PIC (PicSettings/PicSimulation 系) とは完全に独立な専用ソルバー。geometry/mesh とは
// 無関係な一様格子 (n_cells 個のセル) 上で動く。

// 1D 電極の FN (Fowler–Nordheim) 電界放出 (prompts/95)。2D の FnEmission と同じ物理
// (fn.py の fn_current_density) だが、1D は電極がちょうど1点なので放出面/位置サンプリング
// (edges/regions/n/seed) が無い。null/undefined = 放出なし (従来動作と完全ビット不変)
export interface Fn1dEmission {
  phi_ev: number;               // 仕事関数 φ [eV] 既定 4.5
  beta: number;                 // 電界増倍係数 β 既定 1.0
  init_energy_ev: number;       // 放出電子の初期エネルギー [eV] 既定 0.1
  macro_weight?: number | null; // マクロ重み [m^-2]。null なら初期プラズマと同じ (w0)
}

// 1D の左右電極。電圧は v_dc + Σ RF sin + Σ waveforms(t) の合成 (prompts/93)。
// voltage_rf は 2D の Region/BoundaryCondition と同じ VoltageRf 型を流用する
// (単一成分/複数成分リスト/未指定)。CSV 波形 (waveforms) とは併記可
export interface Pic1dElectrode {
  v_dc?: number;
  voltage_rf?: VoltageRf | VoltageRf[]; // RF重畳 (未指定なら直流+CSV波形のみ。複数成分でデュアル周波数)
  waveforms?: VoltageWaveform[];
  see_gamma?: number; // イオン入射あたりのSEE収率 γ
  fn?: Fn1dEmission | null; // FN 電界放出 (prompts/95)。null/undefined なら放出なし
}

// 1D の EEDF/EEPF 集計区間 [x1, x2] (2D の EedfRegion の1D版、prompts/85 と同じ規約)
export interface Eedf1dRegion {
  x1: number;
  x2: number;
  label?: string;
  bins?: number;
  // null/undefined = 平均区間の最初の集計ステップで自動決定 (2D の EedfRegion.e_max_ev と同じ規約)
  e_max_ev?: number | null;
}

export interface Pic1dSettings {
  gap_m: number;           // 電極間ギャップ [m]
  n_cells: number;         // セル数 (節点数 = n_cells+1)
  left: Pic1dElectrode;
  right: Pic1dElectrode;
  init_density_m3: number; // 初期プラズマ密度 (一様、準中性) [m^-3]
  init_te_ev?: number;
  init_ti_ev?: number;
  ion_mass_amu?: number;
  n_macro?: number;        // 種ごとの初期マクロ粒子数
  dt?: number | null;      // 秒。null なら 0.1/ωpe (初期密度から自動)
  n_steps?: number;
  frame_every?: number;
  // 完了時に返す時間平均プロファイルの平均ステップ数。null なら最後の25%
  avg_steps?: number | null;
  // RF 1周期の位相分解ビン数。0=無効、RF (voltage_rf/waveforms) が無ければ無効
  // (基本周波数の優先順位は Pic1dPanel/pic1d.py と同じ: voltage_rf 優先 → waveforms)
  phase_bins?: number;
  mcc?: McSettings | null; // 既存 McSettings をそのまま流用 (null なら MCC 無効)。
  // ただし 1D は DSMC 連成 (mcc.use_dsmc_gas) に未対応 (backend の validator が拒否する)。
  // Pic1dPanel では use_dsmc_gas のトグル自体を出さない
  see_energy_ev?: number;  // SEE 電子の初期エネルギー [eV]
  eedf_regions?: Eedf1dRegion[]; // 最大4個 (backend validator)
  // 壁 IEDF (入射イオンエネルギー分布、prompts/116) のビン数。0=無効。粒子ベース
  // (壁で吸収されたイオンの全運動エネルギーを重み付きヒストグラム化)
  wall_iedf_bins?: number;
  seed?: number;            // 初期装荷の乱数種 (MCC は mcc.seed を使う)
}

// server → client (/ws/pic1d)
export interface Pic1dStartedMsg {
  type: "started";
  n_steps: number;
  step_offset: number; // 続き実行では前回までの累計 (frame.step が通算で進む)
  dt: number;
  x: number[]; // 節点座標 [m] (一様格子、n_cells+1 点)
  warnings: string[];
}

export interface Pic1dFrameMsg {
  type: "frame";
  step: number;
  t: number;
  phi: number[]; // 節点値 [V]
  n_e: number[]; // 節点値 [m^-3]
  n_i: number[]; // 節点値 [m^-3]
  counts: Record<string, number>; // history の各キーの最新値 (step/t/n_e/n_i/wall_*/ion_events 等)
  elapsed_s: number;
  sample: { x: number[]; vx: number[] }; // 電子位相空間 (≤2000点に間引き済み)
}

// done メッセージの history (列ごとの辞書。step ごとの1行データとして扱いたい場合は
// 呼び出し側で zip すること。2D の toDiagArray に相当する変換は不要な規模のため用意しない)
export interface Pic1dHistoryDict {
  step: number[];
  t: number[];
  n_e: number[];       // マクロ粒子数 (実粒子数ではない)
  n_i: number[];
  w_e: number[];       // 実粒子数の総和 (重み和)
  w_i: number[];
  wall_left_e: number[];
  wall_left_i: number[];
  wall_right_e: number[];
  wall_right_i: number[];
  ion_events: number[];
  see_events: number[];
  coll_e: number[];
  fn_left: number[];   // FN 放出重み [m^-2] (このステップ分。prompts/95。fn 未設定なら常に0)
  fn_right: number[];
}

// 完了時の時間平均プロファイル一式 (done メッセージの result.profiles)
export interface Pic1dProfiles {
  x: number[];
  phi: number[];
  e: number[];          // E = -dφ/dx [V/m] (符号付き)
  n_e: number[];
  n_i: number[];
  t_e: number[];
  ionization: number[]; // 電離レート [m^-3 s^-1]
  avg_steps: number;    // 実際に平均したステップ数
}

// 位相分解版シースエッジ (result.cycle.sheath、prompts/97 の Brinkmann 基準)。
// 各配列は bins 長。根が求まらなかったビットは null (NaN ではなく JSON の null)
export interface Pic1dCycleSheath {
  s_left: (number | null)[];
  s_right: (number | null)[];
}

// RF 1周期の位相分解データ (done メッセージの result.cycle、アニメーション用)
export interface Pic1dCycle {
  bins: number;
  freq_hz: number;
  phi: number[][];  // bins × 節点
  n_e: number[][];
  n_i: number[][];
  sheath?: Pic1dCycleSheath;
}

// 時間平均プロファイルのシースエッジ (result.sheath、prompts/97 の Brinkmann 基準)。
// 左右いずれか (または両方) の根が求まらなければ個別に null になる
export interface Pic1dSheath {
  left_s: number | null;
  right_s: number | null;
}

// シース振動スペクトル (result.sheath_fft、prompts/100)。平均区間中の毎ステップ
// s(t) を評価して FFT した片側振幅スペクトル (平均を引いた変動分)。freq_hz/amp_left/
// amp_right は同じ長さの配列。f0_hz が無ければ (RF 未設定/CSV波形も無し) 整数周期
// トリムをしていないので低周波側 2048 ビンで打ち切ったスペクトルになる
export interface Pic1dSheathFft {
  df_hz: number;
  freq_hz: number[];
  amp_left: number[];  // 片側振幅 [m] (変動分)
  amp_right: number[];
  mean_left: number | null;  // s の平均 [m] (電極からの距離)。左右どちらかが縮退すると null
  mean_right: number | null;
  n_samples: number;   // FFT に使ったサンプル数 (トリム後)
  f0_hz: number | null; // 整数周期トリムに使った基本周波数 (cycle と同じ決定ロジック)
}

// s(t) プレビュー系列 (result.sheath_ts、prompts/100)。最大2048点に間引き済み。
// 根が求まらなかったステップは null (線を切る)
export interface Pic1dSheathTs {
  t: number[];       // [s]
  s_left: (number | null)[];
  s_right: (number | null)[];
}

// 指定区間の EEDF/EEPF 集計結果 (done メッセージの result.eedf、prompts/85 の1D版)
export interface Pic1dEedfResult {
  label: string;
  e_centers: number[];
  f: number[];
  mean_energy_ev: number;
  t_eff_ev: number;
  total_weight: number;
  overflow_frac: number;
  n_samples: number;
}

// 壁 (電極) 入射イオンエネルギー分布 (IEDF、prompts/116)。1D PIC (pic1d.py) は
// 壁で吸収されたイオンを直接ヒストグラム化する粒子ベース (厳密)。1D 流体 (fluid1d.py)
// は位相分解シース電圧 + イオン走行時間フィルタで再構成する無衝突シース近似の
// モデルベース (model="collisionless_sheath" が付与され、PIC の結果と区別できる)
export interface WallIedfSide {
  e_centers: number[];
  f: number[];
  mean_energy_ev: number;
  total_weight: number;
  n_samples: number;
  model?: string; // 流体のみ付与 ("collisionless_sheath")。PIC には無い
}

export interface WallIedfResult {
  left: WallIedfSide;
  right: WallIedfSide;
}

export interface Pic1dWallCounts {
  electron: number;
  ion: number;
}

export interface Pic1dWalls {
  left: Pic1dWallCounts;
  right: Pic1dWallCounts;
}

// FN 電界放出の done サマリ (prompts/95)。fn 未設定の電極側は null
export interface Pic1dFnSide {
  j_avg: number;   // 時間平均プロファイルと同じ平均区間の平均放出電流密度 [A/m^2]
  total_w: number; // 放出開始からの累積放出重み [m^-2] (continue をまたいでも維持)
}

export interface Pic1dFnResult {
  left: Pic1dFnSide | null;
  right: Pic1dFnSide | null;
}

// /ws/pic1d の done.result (= ResultsBundle.pic1d に保存する形そのもの)
export interface Pic1dResult {
  history: Pic1dHistoryDict;
  profiles: Pic1dProfiles | null;
  sheath?: Pic1dSheath | null; // シースエッジ (prompts/97)。sheath キー無しの旧保存ファイルとも互換 (optional)
  cycle: Pic1dCycle | null;
  // シース振動スペクトル/プレビュー系列 (prompts/100)。旧保存ファイルとの互換のため optional
  sheath_fft?: Pic1dSheathFft | null;
  sheath_ts?: Pic1dSheathTs | null;
  eedf: Pic1dEedfResult[];
  // 壁 IEDF (prompts/116)。wall_iedf_bins=0 なら null。旧保存ファイルとの互換のため optional
  wall_iedf?: WallIedfResult | null;
  walls: Pic1dWalls;
  fn: Pic1dFnResult | null; // 両電極とも fn 未設定なら null (prompts/95)
  elapsed_s: number;
  timing: Record<string, number>; // deposit/field/push/mcc/other (+ total、server 側で加算)
  settings: Pic1dSettings; // 実行に使った設定 (グリッド再構成に使える)
}

export interface Pic1dDoneMsg {
  type: "done";
  result: Pic1dResult;
}

export interface Pic1dErrorMsg {
  type: "error";
  detail: string;
}

export type Pic1dServerMessage = Pic1dStartedMsg | Pic1dFrameMsg | Pic1dDoneMsg | Pic1dErrorMsg;

// client → server コマンド (/ws/pic1d)。continue の extra_steps は 2D と異なり必須
// (backend が保持中の n_steps をデフォルトに使うが、フロントは常に明示的に送る)
export type Pic1dClientCommand =
  | { cmd: "start"; project: Project }
  | { cmd: "stop" }
  | {
      cmd: "continue";
      extra_steps: number;
      frame_every?: number;
      avg_steps?: number | null;
      phase_bins?: number | null;
    };

// ---- boltzpm (Boltzmann ソルバー) 連携 — LMEA 流体係数テーブル (backend/es_sim/schema.py
// BoltzTable・boltz.py・server.py /ws/boltz と手動同期、prompts/117-118) ---------------------
// E/N を掃引した各点の定常 EEDF から ε̄=⟨ε⟩・μ_e・N・レート係数を求め、ε̄ をキーにテーブル化
// したもの (BOLSIG+ 流の局所平均エネルギー近似、LMEA)。Fluid1dSettings/Fluid2dSettings の
// electron_model="boltzmann" のときに使う。各リストは ε̄ (mean_energy_ev) 昇順に整列済み。

export interface BoltzTable {
  en_td: number[];
  mean_energy_ev: number[];
  mobility_n: number[];   // μ_e・N [1/(m・V・s)]
  k_ion: number[];
  k_exc: number[];
  e_ion_ev: number[];
  e_exc_ev: number[];
  eedf_eps_ev: number[];
  eedf: number[][];       // eedf[i] が eedf_eps_ev グリッド上の EEDF (∫eedf[i]dε≈1、boltz.py 参照)
  source_hash: string;    // sha256(JSON(processes)) — 断面積変更後の再生成判定用 (boltzHash.ts 参照)
  opts: Record<string, unknown>; // 生成に使った opts 一式 (boltzpm_version 込み、backend meta_opts)
  warnings: string[];
}

// run_boltz_sweep の掃引パラメータ (backend/es_sim/boltz.py DEFAULT_BOLTZ_OPTS と同じキー・既定値。
// 既定値自体は panels/BoltzSection.tsx の DEFAULT_BOLTZ_OPTS に持たせる)。project には含めず、
// 生成 UI のローカル状態としてのみ扱う (掃引の都度指定する使い捨てパラメータのため)
export interface BoltzOpts {
  en_min_td: number;
  en_max_td: number;
  n_points: number;
  eps_max_ev: number | null; // null = 自動 (電離/励起の最大閾値×8 と 40eV の大きい方)
  d_eps_ev: number;
  n_theta: number;
}

// server → client (/ws/boltz)
export interface BoltzStartedMsg {
  type: "started";
  n_points: number;
}

export interface BoltzProgressMsg {
  type: "progress";
  i: number;
  n_points: number;
  en_td: number;
  elapsed_s: number;
}

export interface BoltzDoneMsg {
  type: "done";
  table: BoltzTable;
}

export interface BoltzErrorMsg {
  type: "error";
  detail: string;
}

export type BoltzServerMessage = BoltzStartedMsg | BoltzProgressMsg | BoltzDoneMsg | BoltzErrorMsg;

// client → server コマンド (/ws/boltz)。project+module で既存設定 (electron_processes/
// ion_mass_amu/gas_pressure_pa/gas_temperature_k) から掃引条件を取り出させる (server.py
// _run_boltz_session 参照。processes を直接渡す経路もバックエンドにはあるが、フロントは常に
// project 一式を持っているためこちらだけを使う)。continue は無い (1回性の生成のため)
export type BoltzClientCommand =
  | { cmd: "start"; project: Project; module: "fluid1d" | "fluid2d"; opts?: BoltzOpts }
  | { cmd: "stop" };

// ---- 1D プラズマ流体 (ドリフト拡散 + 電子エネルギー、backend/es_sim/schema.py Fluid1dSettings /
// fluid1d.py build_fluid1d_result と手動同期、prompts/104-108) --------------------------------
// pic1d (Pic1dSettings) と同一条件・同一プリセットで直接比較できることが設計目標のため、格子規約
// (n_cells/n_nodes/xg)・電極 (Pic1dElectrode) を意図的に共用する。geometry/mesh とは無関係な
// 専用の一様格子ソルバー (backend/es_sim/fluid1d.py)。粒子を追わず n_e/n_i/T_e/φ を格子節点上の
// 連続場として解くため、pic1d のようなマクロ粒子数・EEDF・FN 電界放出は持たない。

export interface Fluid1dSettings {
  gap_m: number;              // 電極間ギャップ [m]
  n_cells: number;            // セル数 (節点数 = n_cells+1)
  // Pic1dElectrode を流用 (電圧合成・SEE γ 共用)。ただし fn (FN電界放出) は backend の
  // validator (_check_no_fn) が拒否するため、Fluid1dPanel では常に未設定のまま送る
  left: Pic1dElectrode;
  right: Pic1dElectrode;
  init_density_m3: number;    // 初期プラズマ密度 (一様、準中性) [m^-3]
  init_te_ev?: number;
  gas_pressure_pa: number;    // 一様背景ガス圧 [Pa]
  gas_temperature_k?: number;
  ion_mass_amu?: number;
  // イオン低電界移動度 μ_i の基準値・基準ガス密度 (μ_i = mu_i_ref・(n_ref_m3/n_g) で
  // 任意のガス密度へスケールする、fluid1d.py 参照)
  mu_i_ref?: number;
  n_ref_m3?: number;
  t_i_ev?: number;             // イオン温度 (D_i = μ_i・T_i)
  // 修正 Frost イオン移動度 (prompts/116): μ_i(E/N)=μ_L/√(1+(E/N)/C) でシース強電界での
  // 移動度低下を表現する (Ellis et al. 1976 の Ar+ in Ar 実測に対する粗いフィット、工学近似)。
  // "const" は従来の低電界一定値 (frost 導入前とビット不変)。新規機能につき既定は "frost"
  ion_mobility_model?: "frost" | "const";
  frost_c_td?: number;         // 修正 Frost 式の C [Td] (const では無視)。既定 150
  electron_processes?: XsProcess[]; // 空/未指定なら eduPIC Ar 解析式を既定使用
  dt?: number | null;          // 秒。null なら RF周期/2000 と 1e-10 の小さい方
  n_steps?: number;
  frame_every?: number;
  avg_steps?: number | null;   // 完了時に返す時間平均プロファイルの平均ステップ数。null = 最後の25%
  phase_bins?: number;         // RF 1周期の位相分解ビン数 (0=無効)
  // 壁 IEDF (prompts/116) のビン数。0=無効。無衝突シース近似のモデルベース再構成
  // (位相分解シース電圧 + イオン走行時間フィルタ。fluid1d.py 参照)
  wall_iedf_bins?: number;
  // 電子輸送・反応係数のソース (prompts/117-118): "maxwell" (既定) は従来の Maxwell 平均経路
  // (fluid_coeffs.py、ビット不変)。"boltzmann" は boltz_table (boltzpm による LMEA テーブル、
  // 事前に /ws/boltz で生成してフロントが埋め込む) を ε̄=(3/2)Te で参照する。boltz_table が
  // null のまま "boltzmann" を送ると backend の validator が拒否するため、パネル側で
  // 実行前にチェックする (BoltzSection.tsx 参照)
  electron_model?: "maxwell" | "boltzmann";
  boltz_table?: BoltzTable | null;
}

// server → client (/ws/fluid1d)
export interface Fluid1dStartedMsg {
  type: "started";
  n_steps: number;
  step_offset: number; // 続き実行では前回までの累計 (frame.step が通算で進む)
  dt: number;
  x: number[]; // 節点座標 [m] (一様格子、n_cells+1 点)
  warnings: string[];
}

export interface Fluid1dFrameMsg {
  type: "frame";
  step: number;
  t: number;
  phi: number[]; // 節点値 [V]
  n_e: number[]; // 節点値 [m^-3]
  n_i: number[]; // 節点値 [m^-3]
  t_e: number[]; // 節点値 [eV] (流体は粒子サンプルが無いため t_e を毎フレーム含める)
  counts: Record<string, number>; // history の各キーの最新値 (step/t/n_e_total/n_i_total/wall_*/gen_total)
  elapsed_s: number;
}

// done メッセージの history (列ごとの辞書。fluid1d.py の _HISTORY_KEYS と同じキー)
export interface Fluid1dHistoryDict {
  step: number[];
  t: number[];
  n_e_total: number[]; // 全域積算の実密度 (マクロ粒子数ではない、流体は決定論的)
  n_i_total: number[];
  wall_left_e: number[];
  wall_left_i: number[];
  wall_right_e: number[];
  wall_right_i: number[];
  gen_total: number[]; // 電離による累計生成数 [m^-2] (電子・イオン共通)
}

// 完了時の時間平均プロファイル一式 (done メッセージの result.profiles)
export interface Fluid1dProfiles {
  x: number[];
  phi: number[];
  e: number[];          // E = -dφ/dx [V/m] (符号付き)
  n_e: number[];
  n_i: number[];
  t_e: number[];
  ionization: number[]; // 電離レート [m^-3 s^-1]
  avg_steps: number;    // 実際に平均したステップ数
}

// 時間平均プロファイルのシースエッジ (result.sheath、Brinkmann 基準、pic1d.Pic1dSheath と同形)
export interface Fluid1dSheath {
  left_s: number | null;
  right_s: number | null;
}

// RF 1周期の位相分解データ (done メッセージの result.cycle、アニメーション用)
export interface Fluid1dCycle {
  bins: number;
  freq_hz: number;
  phi: number[][]; // bins × 節点
  n_e: number[][];
  n_i: number[][];
  t_e: number[][];
}

export interface Fluid1dWallCounts {
  electron: number;
  ion: number;
}

export interface Fluid1dWalls {
  left: Fluid1dWallCounts;
  right: Fluid1dWallCounts;
}

// /ws/fluid1d の done.result (= ResultsBundle.fluid1d に保存する形そのもの)
export interface Fluid1dResult {
  history: Fluid1dHistoryDict;
  profiles: Fluid1dProfiles | null;
  sheath: Fluid1dSheath | null;
  cycle: Fluid1dCycle | null;
  // 壁 IEDF (無衝突シース近似、prompts/116)。wall_iedf_bins=0 なら null。
  // 旧保存ファイルとの互換のため optional
  wall_iedf?: WallIedfResult | null;
  walls: Fluid1dWalls;
  gen_total: number; // 電離による累計生成数 [m^-2] (history とは別に累計値そのものを持つ)
  elapsed_s: number;
  timing: Record<string, number>; // poisson/transport/energy/other (+ total、server 側で加算)
  settings: Fluid1dSettings; // 実行に使った設定 (グリッド再構成に使える)
}

export interface Fluid1dDoneMsg {
  type: "done";
  result: Fluid1dResult;
}

export interface Fluid1dErrorMsg {
  type: "error";
  detail: string;
}

export type Fluid1dServerMessage =
  | Fluid1dStartedMsg
  | Fluid1dFrameMsg
  | Fluid1dDoneMsg
  | Fluid1dErrorMsg;

// client → server コマンド (/ws/fluid1d)。pic1d と同じく continue の extra_steps は必須
export type Fluid1dClientCommand =
  | { cmd: "start"; project: Project }
  | { cmd: "stop" }
  | {
      cmd: "continue";
      extra_steps: number;
      frame_every?: number;
      avg_steps?: number | null;
      phase_bins?: number | null;
    };

// ---- 2D/軸対称 プラズマ流体 (ドリフト拡散 + 電子エネルギー、EAFE/FEM-SG、
// backend/es_sim/schema.py Fluid2dSettings・fluid2d.py build_fluid2d_result と手動同期、
// prompts/111-113) ---------------------------------------------------------------------
// fluid1d の 2D/軸対称拡張だが、geometry/mesh・境界条件 (Dirichlet 電圧・voltage_rf・
// voltage_waveform・symmetry・see_gamma・conductor/dielectric 領域) は pic1d/fluid1d のような
// 専用設定を持たず、既存のプロジェクト設定 (Project.geometry) をそのまま使う (2D PIC と
// 同一条件で比較できることが設計目標)。このためフロントのキャンバスも CadCanvas のまま
// (1D のような専用 PlotView は作らない) で、フィールド表示は既存の picFieldView/picFrame と
// 同じ汎用機構に載せる (App.tsx 参照)。

export interface Fluid2dSettings {
  init_density_m3: number;    // 初期プラズマ密度 (一様、準中性) [m^-3]
  init_te_ev?: number;
  gas_pressure_pa: number;    // 一様背景ガス圧 [Pa]
  gas_temperature_k?: number;
  ion_mass_amu?: number;
  // イオン低電界移動度 (fluid1d.py と同じ規約: μ_i = mu_i_ref・(n_ref_m3/n_g))
  mu_i_ref?: number;
  n_ref_m3?: number;
  t_i_ev?: number;             // イオン温度 (D_i = μ_i・T_i)
  // 修正 Frost イオン移動度 (fluid1d.py と全く同じ規約・既定値、prompts/116)
  ion_mobility_model?: "frost" | "const";
  frost_c_td?: number;         // 修正 Frost 式の C [Td] (const では無視)。既定 150
  electron_processes?: XsProcess[]; // 空/未指定なら eduPIC Ar 解析式を既定使用
  dt?: number | null;          // 秒。null なら RF周期/2000 と 1e-10 の小さい方
  n_steps?: number;
  frame_every?: number;
  avg_steps?: number | null;   // 完了時に返す時間平均フィールドの平均ステップ数。null = 最後の25%
  phase_bins?: number;         // RF 1周期の位相分解ビン数 (0=無効、既定0)

  // 陰的線形ソルバー (backend/es_sim/fluid2d.py 参照、prompts/115)。
  // "iterative" (既定): numba 並列 Jacobi-BiCGSTAB。収束しなければ自動的に spsolve へ
  // フォールバックする。"direct": 従来の spsolve (SuperLU 都度分解、比較・検証用)
  linear_solver?: "iterative" | "direct";
  // 陰的反復ソルバー (matvec) の並列スレッド数。0=自動選択 (PicSettings.threads と同じ
  // 規約: pic.py の _auto_thread_cap 式)。linear_solver="direct" のときは無効
  threads?: number;
  // 電子輸送・反応係数のソース (fluid1d.py と同じ規約・既定値、prompts/117-118)
  electron_model?: "maxwell" | "boltzmann";
  boltz_table?: BoltzTable | null;
}

// server → client (/ws/fluid2d)。2D PIC (/ws/pic) の started と異なりメッシュを含まない —
// フロントは既に POST /mesh の結果 (同じ project から生成) を持っている前提で、
// 節点番号・座標がそのまま一致する (server.py のコメント参照)
export interface Fluid2dStartedMsg {
  type: "started";
  n_steps: number;
  step_offset: number; // 続き実行では前回までの累計 (frame.step が通算で進む)
  dt: number;
  warnings: string[];
  // 反復ソルバーの実効スレッド数 (PicStartedMsg.effective_threads と同じ趣旨、prompts/115)。
  // linear_solver="direct" のときも値は入るが、spsolve が並列化されないため意味を持たない。
  // optional なのは PicStartedMsg.effective_threads と同じ理由 (古いバックエンド互換)
  effective_threads?: number;
}

// フレームは phi/n_e/n_i/t_e すべて全節点長の配列 (fluid2d.py _make_frame と同じ規約。
// 粒子を追わないため PicFrameMsg のような particles は持たない)
export interface Fluid2dFrameMsg {
  type: "frame";
  step: number;
  t: number;
  phi: number[]; // 節点値 [V]
  n_e: number[]; // 節点値 [m^-3]
  n_i: number[]; // 節点値 [m^-3]
  t_e: number[]; // 節点値 [eV]
  counts: Record<string, number>; // history の各キーの最新値 (step/t/n_e_total/n_i_total/wall_*/gen_total)
  elapsed_s: number;
}

// done メッセージの history (列ごとの辞書。fluid2d.py の _HISTORY_KEYS と同じキー。
// 2D は壁が多数になり得るため左右の区別をせず全壁合計にまとめる、fluid1d との差分)
export interface Fluid2dHistoryDict {
  step: number[];
  t: number[];
  n_e_total: number[]; // 輸送領域全体の積算実密度 (体積重み和)
  n_i_total: number[];
  wall_e: number[]; // 全壁合計の電子吸収 (累計)
  wall_i: number[]; // 全壁合計のイオン吸収 (累計)
  gen_total: number[]; // 電離による累計生成数
}

// 完了時の時間平均フィールド一式 (done メッセージの result.fields)。phi/n_e/n_i/t_e/ionization は
// 全節点長 (非輸送領域の節点は0)、e_abs のみ要素長 (E ベクトルを平均してから絶対値、pic.py と同じ規約)
export interface Fluid2dFields {
  phi: number[];
  e_abs: number[];
  n_e: number[];
  n_i: number[];
  t_e: number[];
  ionization: number[];
  avg_steps: number; // 実際に平均したステップ数
}

// RF 1周期の位相分解データ (done メッセージの result.cycle、アニメーション用)。PicCycle と異なり
// period_s ではなく freq_hz を持ち (fluid2d.py cycle_data と同じキー)、粒子スナップショットは無い
export interface Fluid2dCycle {
  bins: number;
  freq_hz: number;
  phi: number[][]; // bins × 節点
  n_e: number[][];
  n_i: number[][];
  t_e: number[][];
}

// 壁 (吸収境界) の累計吸収数。2D は左右の区別をせず全壁合計 (fluid2d.py Fluid2dSimulation.wall と同じ)
export interface Fluid2dWallCounts {
  electron: number;
  ion: number;
}

// /ws/fluid2d の done.result (= ResultsBundle.fluid2d に保存する形そのもの)
export interface Fluid2dResult {
  history: Fluid2dHistoryDict;
  fields: Fluid2dFields | null;
  cycle: Fluid2dCycle | null;
  walls: Fluid2dWallCounts;
  gen_total: number; // 電離による累計生成数 (history とは別に累計値そのものを持つ)
  elapsed_s: number;
  timing: Record<string, number>; // poisson/transport/energy/other (+ total、server 側で加算)
  settings: Fluid2dSettings; // 実行に使った設定 (グリッド再構成に使える)
}

export interface Fluid2dDoneMsg {
  type: "done";
  result: Fluid2dResult;
}

export interface Fluid2dErrorMsg {
  type: "error";
  detail: string;
}

export type Fluid2dServerMessage =
  | Fluid2dStartedMsg
  | Fluid2dFrameMsg
  | Fluid2dDoneMsg
  | Fluid2dErrorMsg;

// client → server コマンド (/ws/fluid2d)。fluid1d と同じく continue の extra_steps は必須
export type Fluid2dClientCommand =
  | { cmd: "start"; project: Project }
  | { cmd: "stop" }
  | {
      cmd: "continue";
      extra_steps: number;
      frame_every?: number;
      avg_steps?: number | null;
      phase_bins?: number | null;
    };

// ---- DSMC (定常ガス流れ、prompts/54、backend/es_sim/schema.py と手動同期) ----------------

// DSMC のガス分子モデル (VHS: Variable Hard Sphere)。既定は Ar
export interface DsmcGas {
  name: string;
  mass_amu: number;  // 分子質量 [amu]
  d_ref_m: number;    // VHS 基準直径 [m] (t_ref_k にて)
  omega: number;      // 粘性の温度指数 ω (HS=0.5)
  t_ref_k: number;    // 基準温度 [K]
}

// domain 外周エッジの DSMC 境界条件種別。未指定エッジは拡散反射壁 (wall) になる
export type DsmcBoundaryType = "wall" | "symmetry" | "inlet" | "outlet";

export interface DsmcBoundary {
  edges: number[];             // 空可
  p1?: Point | null;           // 線分指定 (domain 外周上、部分区間)。edges と併用可 (和集合)
  p2?: Point | null;
  type: DsmcBoundaryType;
  temperature_k: number;
  pressure_pa?: number | null; // inlet は pressure_pa/flow_sccm のどちらか必須。outlet は省略/0で真空排気
  flow_sccm?: number | null;   // inlet の流量指定 [sccm] (pressure_pa と排他)
}

// 定常ガス流れの DSMC 設定。Project.dsmc が null なら無効
export interface DsmcSettings {
  gas: DsmcGas;
  boundaries: DsmcBoundary[];
  wall_temperature_k: number; // 未指定エッジ・領域輪郭の壁温 [K]
  init_pressure_pa: number;   // 初期充填圧 [Pa]
  init_temperature_k: number;
  n_particles: number;        // 目標シミュレーション粒子数
  dt: number | null;          // 秒。null なら自動
  n_steps: number;
  avg_steps: number;          // 最終 N ステップで時間平均
  seed: number;
  threads: number;            // walk 探索の並列スレッド数 (1=従来経路)
  // 隣接セル拡散による統計ノイズ平滑化の回数 (0=無効、prompts/67)。総量
  // (質量・運動量・エネルギー) は保存され、結果表示と PIC 連成の両方に適用される
  smoothing_passes: number;
  // DSMC 専用メッシュの寸法係数 (1〜20、既定1、prompts/89)。FEM メッシュ寸法 (mesh.size・
  // local_sizes) × この係数で DSMC 専用の粗いメッシュを生成する。walk コストはセル寸法に
  // 反比例するため、粗化係数分だけ直接軽くなる。1.0 = 従来どおり FEM と同一メッシュ
  mesh_scale: number;
}

// POST /dsmc のレスポンス (定常時間平均のガス場)
export interface DsmcResult {
  mesh: MeshResult;
  n: number[];               // 要素ごとの数密度 [m^-3]
  t: number[];                // 要素ごとの温度 [K]
  u: [number, number][];      // 要素ごとの面内流速 [m/s]
  p: number[];                 // 要素ごとの圧力 [Pa]
  n_particles: number;         // 最終シミュレーション粒子数
  macro_weight: number;        // 実分子数/シミュレーション粒子
  dt: number;                  // 実際に使った dt [s]
  inflow: number;              // 平均区間の流入実分子数
  outflow: number;             // 平均区間の流出実分子数
  elapsed_s: number;           // run() の壁時計秒 (continue は区間分のみ、prompts/86)
  timing: Record<string, number>; // 位相別の累積秒 (continue は区間分のみ、prompts/87)
}

// ---- DSMC WebSocket プロトコル (server→client, /ws/dsmc、prompts/58) ------------------------

export interface DsmcStartedMsg {
  type: "started";
  n_steps: number;
  dt: number;
  n_particles: number;
  // 実効スレッド数 (旧バックエンドには無いので optional)。設定が届いているかの確認用
  threads?: number;
}

// 100ステップごとに送られる進捗通知
export interface DsmcProgressMsg {
  type: "progress";
  step: number;
  n_steps: number;
  n_particles: number;
  // ライブ粒子表示用 (prompts/66)。間引き済み座標 (≤2000点)。未対応バックエンドでは省略され得る
  particles?: Point[];
}

export interface DsmcDoneMsg {
  type: "done";
  result: DsmcResult; // REST /dsmc と同形
}

export interface DsmcErrorMsg {
  type: "error";
  detail: string;
}

export type DsmcServerMessage = DsmcStartedMsg | DsmcProgressMsg | DsmcDoneMsg | DsmcErrorMsg;

// client→server コマンド
export type DsmcClientCommand =
  | { cmd: "start"; project: Project }
  | { cmd: "stop" }
  // 保持中の状態から追加実行 (prompts/74)。avg_steps は null なら前回設定を踏襲
  | { cmd: "continue"; n_steps: number; avg_steps?: number | null };

// ---- VHF 定在波 (非線形径方向伝送線路モデル、backend/es_sim/tl.py と手動同期、prompts/101) ----
// 円板電極 (半径 radius_m・ギャップ gap_m)・中心給電・軸対称の径方向 1D。geometry/mesh とは
// 無関係な専用の一様格子ソルバー (pic1d と同じ位置づけ)。

export interface TlSettings {
  radius_m: number;    // 電極半径 R [m]
  gap_m: number;       // ギャップ l [m]
  sheath_m: number;    // 平衡シース厚 s0 (片側、上下対称) [m]
  n_e_m3: number;      // バルク電子密度 [m^-3]
  n_s_ratio: number;   // シース端イオン密度比 n_s/n_e (h係数)
  nu_m_hz: number;     // 電子運動量衝突周波数 ν_m [Hz]
  freq_hz: number;     // 駆動周波数 f0 [Hz]
  v0: number;          // 駆動振幅 [V]
  n_r: number;         // 半径方向節点数
  n_periods: number;   // 総周期数
  n_fft_periods: number; // FFT 窓の周期数 (n_periods より小)
  n_harm: number;      // 返す高調波次数
  dt?: number | null;  // 秒。null なら CFL から自動
  // シースの電荷-電圧関係 (prompts/102)。"child" (既定): Child-Langmuir 型
  // (V_s∝q^{4/3})、対称放電でも奇数次高調波が定常的に生成される。
  // "matrix": 行列シース (V_s∝q^2、従来モデル)。対称放電では厳密に線形化し
  // 高調波はクリップ過渡でしか出ない (比較・線形極限検証用)。
  sheath_law?: "child" | "matrix";
}

// n=0..n_harm の各高調波の振幅 (各節点)。n[0] は DC 成分
export interface TlHarmonics {
  n: number[];
  v: number[][]; // n_harm+1 × n_r  |V_n(r)| [V]
  j: number[][]; // 同  |J_n(r)| [A/m^2]
}

// 基本波の実効波長 (隣接する |V_1(r)| の極小 (節) 間隔 × 2 から推定。節が2つ未満なら null)
export interface TlLambdaEff {
  lambda_m: number | null;
  lambda0_m: number; // 真空波長 c/f0 [m]
  ratio: number | null; // lambda_m/lambda0_m (波長短縮率)
}

// 時間平均吸収電力密度 p(r)=⟨R_b J_z^2⟩ とその一様性指標
export interface TlPower {
  p: number[]; // [W/m^2]
  max_over_min: number | null; // pがすべて0なら null
  area_weighted_std_over_mean: number | null; // 面積(r)重み標準偏差/平均。平均0なら null
}

// プローブ点 (中心・R/2・外周) の最終2周期の V(t) 波形プレビュー
export interface TlVProbe {
  t: number[];
  center: number[];
  mid: number[];
  edge: number[];
}

// プローブ点の振幅スペクトル (40·f0 まで)
export interface TlSpectrumProbe {
  freq_hz: number[];
  center: number[];
  mid: number[];
  edge: number[];
}

// /ws/tl の done.result (= ResultsBundle.tl に保存する形そのもの、settings を含み自己完結)
export interface TlResult {
  r: number[]; // 半径方向節点 [m]
  probe_r: { center: number; mid: number; edge: number };
  harmonics: TlHarmonics;
  lambda_eff: TlLambdaEff;
  power: TlPower;
  feed_power: number; // 時間平均給電電力 ⟨V(r_feed)·I(r_feed)⟩ [W]
  v_probe: TlVProbe;
  spectrum_probe: TlSpectrumProbe;
  warnings: string[]; // シース崩壊 (クリップ) 発生等
  dt: number;
  n_steps: number;
  settings: TlSettings; // 実行に使った設定 (表示単位換算に使える)
}

export interface TlStartedMsg {
  type: "started";
  n_steps: number;
  dt: number;
}

export interface TlProgressMsg {
  type: "progress";
  step: number;
  n_steps: number;
  elapsed_s: number;
}

export interface TlDoneMsg {
  type: "done";
  result: TlResult;
}

export interface TlErrorMsg {
  type: "error";
  detail: string;
}

export type TlServerMessage = TlStartedMsg | TlProgressMsg | TlDoneMsg | TlErrorMsg;

// client→server コマンド (/ws/tl)。continue は無い (tl.py の docstring 参照: 毎回フルの
// 定常化をやり直す設計のため)
export type TlClientCommand = { cmd: "start"; project: Project } | { cmd: "stop" };

// 粒子マージ設定 (高速化③、prompts/77)。電離でマクロ粒子数が増え続けたときに
// 種ごとの上限 n_max を超えたら every ステップごとにセル内保存的マージ (Vranic
// k→2) で削減する。PicSettings.merge が null なら無効 (既定)
export interface PicMerge {
  n_max: number;  // 種ごとの上限マクロ粒子数
  every: number;  // チェック間隔 [ステップ]
}

export interface PicSettings {
  initial_plasma: InitialPlasma | null;
  injection: PicInjection | null;
  n_macro: number;      // 種ごとの初期マクロ粒子数の目安
  dt: number | null;    // 秒。null = 0.1/ωpe (初期密度から自動)
  n_steps: number;
  frame_every: number;  // フレーム送出間隔 (ステップ)
  mcc: McSettings | null;   // null なら MCC(背景ガス衝突) 無効
  see_energy_ev: number;    // SEE(二次電子放出)電子の初期エネルギー [eV]
  // 完了時の時間平均フィールドの平均ステップ数 (最終Nステップ)。null/省略 = 最後の25%
  avg_steps?: number | null;
  // RF 1周期の位相分解データの位相ビン数 (0 で無効)。省略 = 40
  phase_bins?: number;
  // IEDF/IADF コレクタ線分 (旧単数形、後方互換)。null/省略 = 無効
  collector?: PicCollectorSettings | null;
  // 複数コレクタ (最大8個)。バックエンドは旧単数形をこちらへ正規化する
  collectors?: PicCollectorSettings[];
  // EEDF/EEPF 集計領域 (最大4個、prompts/85)。省略/undefined = 無効
  eedf_regions?: PicEedfRegionSettings[];
  // FN 電界放出源 (prompts/46)。毎ステップの表面電界から放出する。null/undefined = 無効
  fn?: FnEmission | null;
  // イオンサブサイクリング (prompts/50)。イオンを N ステップに1回、実効刻み N·dt で押す。
  // 省略/undefined = 1 (無効、従来経路とビット単位で一致)
  ion_subcycle?: number;
  // 粒子チャンク並列のスレッド数 (prompts/50)。0=粒子数・CPU数から自動選択。
  // 結果は threads の値によらずビット単位で一致。省略/undefined = backend既定の自動選択
  threads?: number;
  // 粒子マージ (高速化③、prompts/77)。null/undefined = 無効 (既定)
  merge?: PicMerge | null;
  // シースエッジ評価ライン (最大4本、prompts/98)。可視化専用 (Brinkmann 判定・
  // 準中性度等値線の計算はすべてフロント側で行う。sheath.ts 参照)。省略/undefined = 無効
  sheath_lines?: SheathLineSettings[];
}

// シースエッジ評価ラインの設定 (prompts/98)。p1=電極側、p2=バルク側
// (Brinkmann 積分の参照点 x_b = p2 の位置)
export interface SheathLineSettings {
  p1: [number, number];
  p2: [number, number];
  label?: string; // 表示用ラベル (空なら "S1" 等をフロントが振る)
}

// IEDF/IADF コレクタ線分の設定
export interface PicCollectorSettings {
  p1: [number, number];  // 線分の始点 [m]
  p2: [number, number];  // 線分の終点 [m]
  tol?: number | null;   // 判定距離 [m]。null = mesh.size と同値
  label?: string;        // 表示用ラベル (空なら "C1" 等をフロントが振る)
}

// EEDF/EEPF 集計領域の設定 (軸平行矩形、prompts/85)
export interface PicEedfRegionSettings {
  p1: [number, number];  // 対角の2点 (順不同) [m]
  p2: [number, number];
  label?: string;         // 表示用ラベル (空なら "E1" 等をフロントが振る)
  bins?: number;          // ヒストグラムのビン数 (既定100)
  e_max_ev?: number | null; // null = 平均区間の最初の集計で自動決定
}

// PIC診断 (1ステップ分)
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
  // MCC/SEE 累計カウンタ。undefined = 未対応バックエンド (後方互換のため optional)
  coll_e?: number;      // 電子衝突数 (累計)
  ion_events?: number;  // 電離数 (累計)
  see_events?: number;  // SEE発生数 (累計)
  surf_q?: number;      // 誘電体の累計表面電荷 [C/m] (全誘電体合計)
  // FN 電界放出 (prompts/46)。未対応バックエンドでは undefined
  fn_i?: number;        // そのステップの総放出電流 [A/m]
  fn_events?: number;   // 累計放出マクロ電子数
  // 粒子マージ (prompts/77) で削減した累計マクロ粒子数。未対応バックエンドでは undefined
  merged?: number;
}

// PIC診断履歴 (done メッセージの形式)。バックエンド (pic.py) は列ごとの辞書
// { t: [...], ke_e: [...], ... } で全ステップの履歴を返すため、フロントでは
// toDiagArray() で PicDiag[] (行ごと) に変換して使う
export type PicHistoryDict = { [K in keyof PicDiag]: number[] };

// 列ごとの辞書 → 行ごとの PicDiag[] 変換。形式が想定外でも例外を投げず空配列を返す
export function toDiagArray(h: PicHistoryDict | PicDiag[] | null | undefined): PicDiag[] {
  if (Array.isArray(h)) return h; // 将来サーバーが行形式になっても許容
  if (!h || !Array.isArray(h.t)) return [];
  const n = h.t.length;
  const col = (a: number[] | undefined, i: number) => (a && Number.isFinite(a[i]) ? a[i] : 0);
  // MCC/SEE カウンタは optional (未対応バックエンドでは配列自体が無い) なので、
  // 値が取れない場合は 0 ではなく undefined のままにして「-」表示に委ねる
  const colOpt = (a: number[] | undefined, i: number): number | undefined =>
    a && Number.isFinite(a[i]) ? a[i] : undefined;
  const out: PicDiag[] = new Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = {
      t: col(h.t, i), ke_e: col(h.ke_e, i), ke_i: col(h.ke_i, i), fe: col(h.fe, i),
      n_e: col(h.n_e, i), n_i: col(h.n_i, i), wall_e: col(h.wall_e, i), wall_i: col(h.wall_i, i),
      phi_min: col(h.phi_min, i), phi_max: col(h.phi_max, i),
      coll_e: colOpt(h.coll_e, i), ion_events: colOpt(h.ion_events, i), see_events: colOpt(h.see_events, i),
      surf_q: colOpt(h.surf_q, i),
      fn_i: colOpt(h.fn_i, i), fn_events: colOpt(h.fn_events, i),
      merged: colOpt(h.merged, i),
    };
  }
  return out;
}

// ---- PIC WebSocket プロトコル (server→client, /ws/pic) ------------------------

export interface PicStartedMsg {
  type: "started";
  dt: number;
  n_steps: number;
  // backendが実際に粒子カーネルへ割り当てたスレッド数。旧backendでは省略
  effective_threads?: number;
  // 区間開始時の通算ステップ (continue では前回までの累計)。frame.step が通算で進むため、
  // 進捗率は (step - step_offset)/n_steps で計算する。旧バックエンドには無いので optional
  step_offset?: number;
  warnings: string[];
  mesh: MeshResult;
}

export interface PicFrameMsg {
  type: "frame";
  step: number;
  t: number;
  phi: number[]; // 節点値
  // 種ごとの要素密度 [m^-3] (prompts/81)。ライブモニタの表示切替用。
  // 旧バックエンドには無いので optional (未対応時はフロント側で phi にフォールバックする)
  n_e?: number[];
  n_i?: number[];
  particles: { electron: Point[]; ion: Point[] }; // 種ごと最大2000点に間引き済み
  diag: PicDiag;
}

// PIC完了時の時間平均2Dフィールド一式 (done メッセージの fields)
export interface PicFields {
  phi: number[];       // 節点、時間平均電位 [V]
  e_abs: number[];     // 要素、時間平均 |E| [V/m] (Eベクトルを平均してから絶対値)
  n_e: number[];       // 節点、電子密度 [m^-3]
  n_i: number[];       // 節点、イオン密度 [m^-3]
  te_ev: number[];     // 節点、電子温度 [eV] (粒子なし節点は 0)
  ion_rate: number[];  // 節点、電離レート [m^-3 s^-1]
  avg_steps: number;   // 実際に平均したステップ数
}

// RF 1周期の位相分解データ (done メッセージの cycle、アニメーション用)
export interface PicCycle {
  bins: number;        // 位相ビン数
  period_s: number;    // RF 周期 [s]
  phi: number[][];     // bins × 節点  位相分解平均の電位 [V]
  n_e: number[][];     // bins × 節点  同 電子密度 [m^-3]
  n_i: number[][];     // bins × 節点  同 イオン密度 [m^-3]
  // 以下3つは追加分 (prompts/52)。古いバックエンドでは省略され得る (optional)
  e_abs?: number[][];    // bins × 要素数  同 |E| [V/m] (要素値。phi/n_e/n_i 等の節点値と異なる)
  te_ev?: number[][];    // bins × 節点数  同 電子温度 [eV] (節点値)
  ion_rate?: number[][]; // bins × 節点数  同 電離レート [m^-3 s^-1] (節点値)
  particles: {         // 最後の1周期の生スナップショット (位相ビンごと、≤1000点)
    electron: [number, number][][];
    ion: [number, number][][];
  };
}

// IEDF/IADF コレクタの記録結果 (done メッセージの collector)
export interface PicCollectorResult {
  count: number;         // 記録したマクロイオン数 (総数)
  total_weight: number;  // 総実イオン数 [1/m]
  energies_ev: number[]; // サンプル (上限 50000。超過分は count/total_weight のみ)
  angles_deg: number[];  // 同数。符号付き入射角 [deg] (範囲 (-90, 90))
  weights: number[];     // 同数
  truncated: boolean;
}

// 指定矩形範囲の EEDF/EEPF 集計結果 (done メッセージの eedf、prompts/85)。
// f は EEDF [eV^-1] (∫f dE = 1 に正規化)。EEPF は f/√E で表示側 (PicPanel) が導出する
export interface PicEedfResult {
  label: string;
  e_centers: number[];    // ビン中心のエネルギー [eV]
  f: number[];            // EEDF [eV^-1] (e_centers と同数)
  mean_energy_ev: number; // 平均運動エネルギー [eV]
  t_eff_ev: number;       // T_eff = (2/3)⟨E⟩ [eV]
  total_weight: number;   // ヒストグラムに入った Σw (実電子数)。0 = 電子が一度も入らなかった
  overflow_frac: number;  // e_max_ev を超えた分の重み比率 (0〜1)
  n_samples: number;      // 平均区間の実ステップ数
}

export interface PicDoneMsg {
  type: "done";
  history: PicHistoryDict; // 列ごとの辞書 (toDiagArray で PicDiag[] に変換して使う)
  fields?: PicFields;      // 時間平均フィールド (未対応バックエンドでは省略)
  cycle?: PicCycle;        // RF 1周期の位相分解 (RFなし/phase_bins=0 では省略)
  collector?: PicCollectorResult;    // 旧単数キー (コレクタが1個のときのみ、後方互換)
  collectors?: PicCollectorResult[]; // 複数コレクタの結果 (collectors と同順)
  eedf?: PicEedfResult[];  // 指定矩形範囲の EEDF/EEPF (eedf_regions と同順、prompts/85)
  // 位相別プロファイル計測 (prompts/75)。キーは solve/gather_push/walk/deposit/mcc/other/total [秒]。
  // continue では区間分のみ (未対応バックエンドでは省略、optional)
  timing?: Record<string, number>;
  // run_batch の壁時計秒 (prompts/86)。continue では区間分のみ。未対応バックエンドでは省略
  elapsed_s?: number;
}

export interface PicErrorMsg {
  type: "error";
  detail: string;
}

export type PicServerMessage = PicStartedMsg | PicFrameMsg | PicDoneMsg | PicErrorMsg;

// client→server コマンド
export type PicClientCommand =
  | { cmd: "start"; project: Project }
  | { cmd: "stop" }
  // 保持中の状態から追加実行 (prompts/32)。avg_steps/phase_bins は null なら前回設定を踏襲
  | {
      cmd: "continue";
      n_steps: number;
      frame_every?: number | null;
      avg_steps?: number | null;
      phase_bins?: number | null;
    };

// CadCanvas でのライブ描画用にまとめたビュー (started の mesh + 最新 frame)。
// 表示フィールド切替 (prompts/81) に対応するため、値配列・節点/要素の別・単位・対数フラグを
// 一般化して持つ (picFieldView (PicFieldView型) と同じ形の情報を、実行中のライブ表示にも
// 持たせる形)。phi: 節点値・単位V。n_e/n_i: 要素値・単位 m^-3 (App 側で選択・フォールバックを解決する)
export interface PicLiveFrame {
  mesh: MeshResult;
  values: number[]; // nodeBased なら節点値、そうでなければ要素値
  nodeBased: boolean;
  unit: string;
  log: boolean; // 対数スケール表示 (密度選択時のみ意味を持つ)
  particles: { electron: Point[]; ion: Point[] };
}

// 一様磁場 [T] (prompts/51)。粒子軌道追跡・PIC の Boris 法ローレンツ力にのみ適用され、
// 静電場ソルブ (/solve) には影響しない。軸対称 (rz/rz_x0) では全成分0以外は不可 (∇·B=0 と矛盾するため)
export interface BField {
  bx: number;
  by: number;
  bz: number;
}

export interface Project {
  version: number;
  unit: "m" | "mm";
  // 座標系。"rz" = 軸対称 (x=z, y=r, 対称軸 y=0)、"rz_x0" = 軸対称 (x=r, y=z, 対称軸 x=0)。省略 = "xy"
  coord?: "xy" | "rz" | "rz_x0";
  geometry: Geometry;
  // mode: "structured" は軸平行矩形 domain 専用の等間隔構造格子 (省略 = unstructured)
  mesh: {
    size: number;
    local_sizes?: { region: string; size: number }[];
    // 任意の線分近傍のローカルメッシュサイズ (prompts/90)。local_sizes と同じく structured では無視される
    local_edge_sizes?: EdgeMeshSize[];
    // "cartesian" (prompts/119): v2 エンジン (直交格子 + 埋め込み境界)。静電場は GMG-PCG、PIC は GPU
    mode?: "unstructured" | "structured" | "cartesian";
    // cartesian モードの局所細分化 (prompts/121)。null/undefined = 細分化なし。静電場 (Solve・
    // プロファイル) のみ対応し、PIC は基本格子 (一様) で計算する
    amr?: AmrSettings | null;
  };
  solver?: { backend: "numpy" | "cupy" | "auto" };
  particles?: ParticleSettings;
  pic?: PicSettings;
  // 一様磁場 [T]。null/undefined または全成分0は磁場なしと同値
  b_field?: BField | null;
  // 定常ガス流れの DSMC 設定 (prompts/54)。null/undefined なら無効
  dsmc?: DsmcSettings | null;
  // 1D PIC/MCC (1d3v、prompts/91)。null/undefined なら無効。geometry/mesh とは無関係に動く
  // 専用の一様格子ソルバー (backend/es_sim/pic1d.py)。2D の pic とは完全に独立
  pic1d?: Pic1dSettings | null;
  // 1D プラズマ流体 (ドリフト拡散 + 電子エネルギー、prompts/104-108)。null/undefined なら無効。
  // pic1d と同一条件で比較できるよう設計された専用ソルバー (backend/es_sim/fluid1d.py)
  fluid1d?: Fluid1dSettings | null;
  // 2D/軸対称 プラズマ流体 (ドリフト拡散 + 電子エネルギー、EAFE/FEM-SG、prompts/111-113)。
  // null/undefined なら無効。geometry/mesh・境界条件は既存のプロジェクト設定をそのまま使う
  // (backend/es_sim/fluid2d.py)。pic と同様キャンバスは CadCanvas を使う
  fluid2d?: Fluid2dSettings | null;
  // VHF 定在波 (非線形径方向伝送線路モデル、prompts/101)。null/undefined なら無効。
  // pic1d 同様 geometry/mesh とは無関係な専用ソルバー (backend/es_sim/tl.py)
  tl?: TlSettings | null;
}

// 軸対称モード判定 (rz: 下辺 y=0 が対称軸、rz_x0: 左辺 x=0 が対称軸)。
// エネルギー単位 [J]・電流 [A]・表面電荷 [C] 表示など「rz または rz_x0 で共通に
// 発動する」判定はここに集約する (PIC は prompts/47 で軸対称対応済み)
export function isAxisymmetric(coord: Project["coord"]): boolean {
  return coord === "rz" || coord === "rz_x0";
}

export interface TraceResult {
  trajectories: Point[][];               // 粒子ごと、save_every ステップごと (初期位置含む)
  status: ("absorbed" | "alive")[];      // absorbed = 電極/外周に到達して停止
  tof: (number | null)[];                // absorbed 粒子の飛行時間 [s]
  final_energy_ev: number[];             // 最終運動エネルギー [eV]
  final_angle_deg: number[];             // 最終速度の向き [度] (x軸から反時計回り、-180〜180)。absorbed 粒子では衝突時の入射方向
  dt: number;                            // 実際に使った dt
  // FN 電界放出 (prompts/46、fn 指定時のみ非 null): 粒子ごとの担持電流と総放出電流。
  // 単位は xy: [A/m] (奥行き1m)、rz/rz_x0: [A]
  currents?: number[] | null;
  fn_current?: number | null;
}

export interface MeshResult {
  nodes: Point[];
  triangles: [number, number, number][];
  region_of_triangle: number[];
}

// 電極 (domain 外周の Dirichlet エッジ or conductor 領域) ごとの誘起電荷 (prompts/71)。
// label は "edge0".."edge3" (FieldPanel の EDGE_LABELS_* で辺名に変換する) または
// conductor の region id。q の単位は xy: [C/m] (奥行き1m)、rz/rz_x0: [C]
export interface ElectrodeCharge {
  label: string;
  voltage: number;
  q: number;
}

export interface SolveResult {
  mesh: MeshResult;
  v: number[];
  e_field: Point[];
  v_min: number;
  v_max: number;
  e_abs_max: number;
  energy: number;
  // 後方互換のため optional (旧バックエンドの応答には無い)
  charges?: ElectrodeCharge[];
  // 静電容量。電極電位がちょうど2水準・空間電荷ρ=0の場合のみ非 null。
  // xy: [F/m]、rz/rz_x0: [F] (prompts/71)
  capacitance?: number | null;
}

export interface Health {
  status: string;
  version: string;
  gpu: boolean;
  // 粒子カーネル (walk/デポジット/gather+push) が numba JIT で高速化されているか
  // (prompts/76)。optional: 旧バックエンドとの互換のため省略可能にする
  numba?: boolean;
}

export interface ProfileResult {
  s: number[];               // 弧長 (p1 からの距離) [m]
  v: (number | null)[];      // 電位 [V] (領域外は null)
  e_abs: (number | null)[];  // |E| [V/m] (領域外は null)
}

// ---- パラメータスイープ WebSocket プロトコル (server→client, /ws/sweep、prompts/79) --------
// 1パラメータ×値リストをケースごとに別プロセスで並列実行する。continue には対応しない
// (毎回フルの N ケースを実行し直す)。

export interface SweepStartedMsg {
  type: "started";
  n_cases: number;
  param_path: string;
  values: number[];
  // 解決済みの実行対象 ("pic"=2D FEM-PIC / "pic1d"=1D PIC-MCC / "fluid1d"=1D プラズマ流体 /
  // "fluid2d"=2D/軸対称プラズマ流体)。未指定リクエストでも server 側 (resolve_sweep_module) が
  // 必ず解決して返す (表示用、prompts/96・107・112)
  module: "pic" | "pic1d" | "fluid1d" | "fluid2d";
}

// 数百ms〜数秒間隔でケースごとに届く進捗 (batch.py の間引きに準じる)
export interface SweepProgressMsg {
  type: "progress";
  case: number;
  step: number;
  n_steps: number;
}

export interface SweepCaseDoneMsg {
  type: "case_done";
  case: number;
  ok: boolean;
  error?: string;
}

export interface SweepSummaryEntry {
  case: number;
  value: number;
  ok: boolean;
  error?: string;
}

export interface SweepDoneMsg {
  type: "done";
  summary: SweepSummaryEntry[];
}

export interface SweepErrorMsg {
  type: "error";
  detail: string;
}

export type SweepServerMessage =
  | SweepStartedMsg
  | SweepProgressMsg
  | SweepCaseDoneMsg
  | SweepDoneMsg
  | SweepErrorMsg;

// client→server コマンド。module は自動判定に頼らず UI 確定値を明示送信する (prompts/96・112)
export type SweepClientCommand =
  | {
      cmd: "start";
      project: Project;
      param_path: string;
      values: number[];
      parallel: number;
      module: "pic" | "pic1d" | "fluid1d" | "fluid2d";
    }
  | { cmd: "stop" };

// フロント側でケースごとに保持する表示用状態 (SweepPanel のケース一覧、App が WS
// コールバックで更新する)
export interface SweepCaseState {
  value: number;
  status: "pending" | "running" | "done" | "error";
  step?: number;
  nSteps?: number;
  error?: string;
}

// ---- 計算結果の保存・読込 (prompts/64) ----------------
// 「結果付き保存」でプロジェクト本体に同梱する計算結果一式。
// フロント専用フィールドであり、solve/trace/mesh/pic/dsmc 等の API へ送る project には含めない
// (結果はジオメトリ・設定と整合していないと意味がないため、プロジェクトと同じ1ファイルに保存する方針)
export interface ResultsBundle {
  version: 1;
  solve?: SolveResult | null;
  mesh?: MeshResult | null;      // Mesh ボタン単独実行の結果
  trace?: TraceResult | null;
  pic?: {
    started: PicStartedMsg;       // mesh を含む (フィールド描画に必須なので必須項目)
    frame: PicFrameMsg | null;    // 最終フレーム (ライブ表示・診断の復元用)
    history: PicDiag[];
    fields: PicFields | null;
    cycle: PicCycle | null;
    collectors: PicCollectorResult[];
    eedf: PicEedfResult[];
    // run_batch の壁時計秒 (prompts/86)。旧形式の結果付き保存ファイルには無いため optional
    elapsed_s?: number;
  } | null;
  gas?: DsmcResult | null;
  // 1D PIC/MCC の完了結果一式 (prompts/91)。done.result そのもの (settings を含み自己完結)
  pic1d?: Pic1dResult | null;
  // 1D プラズマ流体の完了結果一式 (prompts/104-108)。done.result そのもの (settings を含み自己完結)
  fluid1d?: Fluid1dResult | null;
  // 2D/軸対称 プラズマ流体の完了結果一式 (prompts/111-113)。done.result そのもの
  // (settings を含み自己完結)。mesh は含まない (既存の mesh 結果を流用する、Fluid2dResult 参照)
  fluid2d?: Fluid2dResult | null;
  // VHF 定在波の完了結果一式 (prompts/101)。done.result そのもの (settings を含み自己完結)
  tl?: TlResult | null;
}
