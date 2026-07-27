// backend/es_sim/schema.py と手動同期 (将来 openapi.json から自動生成に移行)

export type Point = [number, number];

export type RegionType = "conductor" | "dielectric" | "charge";

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
  // true なら直前に実行した DSMC の定常ガス場 (n・T・u) を背景として使う (prompts/54)。
  // サーバーが保持する DSMC 結果とメッシュが一致している必要がある (未実行/不一致はサーバーがエラー)
  use_dsmc_gas?: boolean;
}

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
}

// ---- DSMC WebSocket プロトコル (server→client, /ws/dsmc、prompts/58) ------------------------

export interface DsmcStartedMsg {
  type: "started";
  n_steps: number;
  dt: number;
  n_particles: number;
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
  // FN 電界放出源 (prompts/46)。毎ステップの表面電界から放出する。null/undefined = 無効
  fn?: FnEmission | null;
  // イオンサブサイクリング (prompts/50)。イオンを N ステップに1回、実効刻み N·dt で押す。
  // 省略/undefined = 1 (無効、従来経路とビット単位で一致)
  ion_subcycle?: number;
  // 粒子チャンク並列のスレッド数 (prompts/50)。walk 探索を threads 分割して並列実行する。
  // 結果は threads の値によらずビット単位で一致。省略/undefined = 1 (逐次)
  threads?: number;
  // 粒子マージ (高速化③、prompts/77)。null/undefined = 無効 (既定)
  merge?: PicMerge | null;
}

// IEDF/IADF コレクタ線分の設定
export interface PicCollectorSettings {
  p1: [number, number];  // 線分の始点 [m]
  p2: [number, number];  // 線分の終点 [m]
  tol?: number | null;   // 判定距離 [m]。null = mesh.size と同値
  label?: string;        // 表示用ラベル (空なら "C1" 等をフロントが振る)
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

export interface PicDoneMsg {
  type: "done";
  history: PicHistoryDict; // 列ごとの辞書 (toDiagArray で PicDiag[] に変換して使う)
  fields?: PicFields;      // 時間平均フィールド (未対応バックエンドでは省略)
  cycle?: PicCycle;        // RF 1周期の位相分解 (RFなし/phase_bins=0 では省略)
  collector?: PicCollectorResult;    // 旧単数キー (コレクタが1個のときのみ、後方互換)
  collectors?: PicCollectorResult[]; // 複数コレクタの結果 (collectors と同順)
  // 位相別プロファイル計測 (prompts/75)。キーは solve/gather_push/walk/deposit/mcc/other/total [秒]。
  // continue では区間分のみ (未対応バックエンドでは省略、optional)
  timing?: Record<string, number>;
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

// CadCanvas でのライブ描画用にまとめたビュー (started の mesh + 最新 frame)
export interface PicLiveFrame {
  mesh: MeshResult;
  phi: number[];
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
    mode?: "unstructured" | "structured";
  };
  solver?: { backend: "numpy" | "cupy" | "auto" };
  particles?: ParticleSettings;
  pic?: PicSettings;
  // 一様磁場 [T]。null/undefined または全成分0は磁場なしと同値
  b_field?: BField | null;
  // 定常ガス流れの DSMC 設定 (prompts/54)。null/undefined なら無効
  dsmc?: DsmcSettings | null;
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

// client→server コマンド
export type SweepClientCommand =
  | { cmd: "start"; project: Project; param_path: string; values: number[]; parallel: number }
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
  } | null;
  gas?: DsmcResult | null;
}
