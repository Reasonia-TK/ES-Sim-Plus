# 91: 1D PIC/MCC (1d3v) バックエンド — 専用一様格子ソルバー

## 背景 (ユーザー要望)

「1DのPIC/MCCを追加できるかな？」
CCP (容量結合プラズマ) の標準ベンチマーク構成 (Turner et al., Phys. Plasmas 20,
013507 (2013) / eduPIC, Donkó et al., PSST 30, 095017 (2021)) と同じ 1d3v
(空間1D・速度3成分) PIC/MCC を、**2D FEM とは独立した専用モジュール**として追加する。
一様格子 + 三重対角 Poisson なので walk 探索が不要で、2D よりはるかに高速。
MCC (null-collision、断面積テーブル、Turner 互換オプション) は既存 `MccModel` を流用する。

このプロンプトは**バックエンドのみ** (フロントは prompts/92 で別途)。

## データモデル (backend/es_sim/schema.py)

```py
class Pic1dElectrode(BaseModel):
    """1D の左右電極。電圧は v_dc + Σ waveforms(t) (既存 VoltageWaveform 合成を流用)。"""
    v_dc: float = 0.0
    waveforms: list[VoltageWaveform] = []
    see_gamma: float = Field(0.0, ge=0.0, le=1.0)  # イオン入射あたりのSEE収率

class Eedf1dRegion(BaseModel):
    """1D の EEDF/EEPF 集計区間 [x1, x2] (2D の EedfRegion の 1D 版)。"""
    x1: float
    x2: float
    label: str = ""
    bins: int = Field(100, ge=10, le=1000)
    e_max_ev: float | None = Field(None, gt=0)

class Pic1dSettings(BaseModel):
    """1D PIC/MCC (1d3v)。null なら無効。2D の pic とは独立に実行できる。"""
    gap_m: float = Field(..., gt=0)          # 電極間ギャップ [m]
    n_cells: int = Field(128, ge=8, le=100000)
    left: Pic1dElectrode = Pic1dElectrode()
    right: Pic1dElectrode = Pic1dElectrode()
    init_density_m3: float = Field(..., gt=0)   # 初期プラズマ密度 (一様、準中性)
    init_te_ev: float = Field(2.0, gt=0)
    init_ti_ev: float = Field(0.03, gt=0)
    ion_mass_amu: float = Field(39.948, gt=0)   # He: 4.0026 (Turner は He+ 6.67e-27 kg)
    n_macro: int = Field(20000, gt=0)           # 種ごとの初期マクロ粒子数
    dt: float | None = Field(None, gt=0)        # None なら 0.1/ωpe
    n_steps: int = Field(2000, gt=0)
    frame_every: int = Field(20, gt=0)
    avg_steps: int | None = Field(None, gt=0)   # None なら最後の25% (2D と同じ規約)
    phase_bins: int = Field(40, ge=0)           # RF 位相分解 (0=無効、2D と同じ規約)
    mcc: MccSettings | None = None              # 既存 MccSettings をそのまま流用
    see_energy_ev: float = Field(2.0, ge=0)
    eedf_regions: list[Eedf1dRegion] = []       # 最大4 (validator)
    seed: int = 0                               # 初期装荷の乱数種 (MCC は mcc.seed)

class Project(...):
    pic1d: Pic1dSettings | None = None  # geometry/mesh とは無関係に動く
```

- `use_dsmc_gas=True` は 1D では未対応 → 明示的な ValueError (メッセージ日本語)。
- 単位: gap_m・x1/x2 は常に **メートル** (フロントで mm/µm 変換。2D と同じ)。

## ソルバー (backend/es_sim/pic1d.py — 新規)

`Pic1dSimulation` クラス。設計方針: **numpy 完全ベクトル化** (粒子ループ禁止)。
1D は walk 不要・格子演算が軽いので numba なしで十分速い (理由をコメントに記す)。

### 格子・場

- 節点 `n_nodes = n_cells + 1`、`dx = gap_m / n_cells`、節点座標 `xg = linspace(0, gap, n_nodes)`。
- **電荷堆積**: CIC (線形重み) で節点へ。節点密度への変換は体積 `dx` (端節点は `dx/2`)。
  1D のマクロ重み w は「単位断面積あたりの実粒子数 [m^-2]」:
  `w0 = init_density_m3 * gap_m / n_macro`。
- **Poisson**: -ε0 φ'' = ρ、両端 Dirichlet φ(0)=V_L(t)、φ(gap)=V_R(t)。
  三重対角を `scipy.linalg.solve_banded` で解く (行列は不変なので係数は初期化時に構築、
  右辺のみ毎ステップ更新)。電圧合成は既存の波形評価 (pic.py の `_eval_waveform` /
  `_dirichlet_values` 相当) と**同一の式**になるよう共通化 or 忠実に移植
  (v_dc + Σ [rf sin + csv ループ]。既存実装を読んで一致させること)。
- **電場**: 節点 E = -dφ/dx (内点は中心差分、端は片側差分)。粒子へは CIC で補間 (gather)。

### 粒子

- 電子 + イオンの2種。位置 x (n,)、速度 v (n,3) の 1d3v。
- **リープフロッグ**: 初回に v を半ステップ戻す (2D と同じ規約)。押しは vx のみ
  (静電 1D、磁場なし)。vy・vz は MCC の散乱でのみ変わる。
- **境界**: x<0 / x>gap で吸収。壁ごと・種ごとの吸収数 (重み和) を記録。
  イオン吸収時に `see_gamma` の確率で二次電子を壁から放出
  (エネルギー see_energy_ev、速度方向は壁法線内向き半球等方 — 2D の SEE 実装を読んで
  速度サンプリング規約を合わせる)。
- **MCC**: 既存 `MccModel` を流用。`collide_electrons(x, v, w, elem, dt)` の x は
  電離位置の記録に使われるだけなので 1D の x (n,) をそのまま渡せるか確認し、
  必要なら形状を合わせる。elem は粒子のセル番号 `clip(int(x/dx), 0, n_cells-1)`
  (一様ガスなら None 可の経路も確認)。電離で生成された電子・イオンを追加。
  ionization_split / ion_energy_frame は既存のまま効く (Turner 互換)。

### 診断 (2D の規約に合わせる)

- **history**: 毎ステップ {step, n_e, n_i (マクロ数), w_e, w_i (重み和), wall_left/right 吸収}。
- **時間平均** (avg_steps、なければ最後の25%): 節点配列 phi_avg, e_avg, n_e_avg, n_i_avg,
  T_e_avg (⟨½mv²⟩ の 2/3 [eV]、粒子重み付き節点集計)、電離レート [m^-3 s^-1]
  (平均区間中の電離イベント位置を CIC 集計 ÷ 体積 ÷ 時間)。
- **位相分解** (phase_bins>0 かつ RF 波形あり): 2D の cycle データと同じ考え方で、
  基本周波数 (最初の RF 波形の freq_hz) の1周期を bins 分割し、平均区間中の
  phi/n_e/n_i を位相ビンごとに平均 → `cycle: {bins, freq_hz, phi[][], n_e[][], n_i[][]}`。
- **EEDF**: eedf_regions ごとに平均区間中の毎ステップ、[x1,x2] 内の電子を
  重み付きエネルギーヒストグラムへ (2D の実装規約 — bins/e_max_ev 自動決定・
  T_eff=(2/3)⟨E⟩ — を読んで合わせる)。
- **timing**: {deposit, field, push, mcc, other} [s] を 2D 同様に計測して done で返す。
- elapsed_s。

### 続きから実行

- `prepare_continue(extra_steps)` で状態 (粒子・rng・平均アキュムレータのリセット・
  step_offset) を保って追加実行。**run(n) → continue(m) が run(n+m) とビット一致**
  (2D の test_continue と同じ水準。MccModel の rng は継続使用)。

## プリセット (backend/es_sim/pic1d_presets.py — 新規)

`GET /pic1d/presets` で返す辞書を組み立てる。目的: ユーザーがワンクリックで
ベンチマーク条件を再現できること。

### 1. "edupic_ar" — eduPIC 基準ケース (Ar、解析式断面積で自己完結)

eduPIC (Donkó et al., PSST 30, 095017 (2021), GPL) の解析式断面積
(出典: Phelps & Petrovic, PSST 8, R21 (1999) [電子]; Phelps, J. Appl. Phys. 76, 747
(1994) [イオン]) を energy グリッド上でサンプルして XsProcess テーブル化する関数
`edupic_ar_processes() -> tuple[list[XsProcess], list[XsProcess]]` を実装:

```
σ [m^2]、en [eV]:
弾性:  1e-20*( |6.0/(1+(en/0.1)+(en/0.6)^2)^3.3
              - 1.1*en^1.4/(1+(en/15)^1.2)/sqrt(1+(en/5.5)^2.5+(en/60)^4.1)|
              + 0.05/(1+en/10)^2 + 0.01*en^3/(1+(en/12)^6) )
励起 (閾値 11.5 eV):
       1e-20*( 0.034*(en-11.5)^1.1*(1+(en/15)^2.8)/(1+(en/23)^5.5)
              + 0.023*(en-11.5)/(1+en/80)^1.9 )        (en>11.5、それ以外 0)
電離 (閾値 15.8 eV):
       1e-20*( 970*(en-15.8)/(70+en)^2 + 0.06*(en-15.8)^2*exp(-en/9) )  (en>15.8)
イオン等方:   qiso(E_lab) = 2e-19*E^-0.5/(1+E) + 3e-19*E/(1+E/3)^2
イオン後方:   qback = (qmom - qiso)/2、qmom = 1.15e-18*E^-0.1*(1+0.015/E)^0.6
```

- energy グリッド: 例 `np.geomspace(1e-3, 1e3, 400)` に閾値点を挿入 (閾値直上の
  立ち上がりが補間で鈍らないよう)。0 eV 点も先頭に。
- ion は lab 系 (`ion_energy_frame="lab"`) — Phelps 1994 は実験室系。
- プリセット条件 (eduPIC base case): gap 2.5 cm、Ar 10 Pa・350 K、13.56 MHz
  振幅 250 V (片側 RF・対向接地)、n_cells 400? — eduPIC の README/論文既定値を
  正確に覚えていなければ、**桁が妥当な実用値** (n_cells 256、dt = 1/(400·f)、
  init_density 1e15 m^-3、n_macro 30000、He ではなく Ar mass 39.948) を設定し、
  コメントに「eduPIC 論文の base case 相当 (セル数等はアプリ向けに調整)」と明記。

### 2. "turner_he_case1" — Turner ベンチマーク Case 1 (He、断面積は要インポート)

- gap 6.7 cm、He (ion_mass_amu = 6.67e-27/1.66054e-27 ≈ 4.017)、ガス: n = 9.64e20 m^-3
  → pressure_pa = n·kB·300 ≈ 3.99 Pa、300 K。RF 450 V・13.56 MHz (左電極、右接地)。
  n_cells 128、dt = 1/(400·13.56e6)、init_density 2.56e14 m^-3、init_te 3 eV?
  (論文の初期値を正確に覚えていなければ妥当値でよい。**放電が定常に達すれば初期値へ
  の依存は消える**旨コメント)。ionization_split="half"、ion_energy_frame="com"。
- electron_processes/ion_processes は**空のまま**返し、プリセット辞書に
  `"note": "He 断面積 (Biagi/Phelps) を LXCat からインポートして設定してください"`
  を付ける (断面積の捏造はしない)。

## サーバー (backend/es_sim/server.py)

- `/ws/pic1d` — `/ws/pic` を踏襲 (コードを読んで同じプロトコルに):
  - 受信: {project, cmd:"start"} / {cmd:"continue", extra_steps} / {cmd:"cancel"}
  - started: {type:"started", n_steps, step_offset, dt, x: 節点座標リスト, threads は不要}
  - frame (frame_every ごと): {type:"frame", step, phi[], n_e[], n_i[], counts, 
    elapsed_s, さらに位相モニタ用に t}
    - ライブ用に電子の位相空間サンプル {x[], vx[]} (最大2000粒子、間引き) も含める
  - done: {type:"done", result: {history, profiles{x,phi,e,n_e,n_i,t_e,ionization},
    cycle, eedf, walls, elapsed_s, timing, settings スナップショット}}
  - error: {type:"error", message}
  - 実行は 2D 同様 anyio スレッドオフロード + キャンセル対応 (既存実装の流儀に従う)。
- 続きから: 2D の /ws/pic の continue の仕組み (直近 sim 保持) をそのまま踏襲。
- 保存/読込 (ResultsBundle): 既存 `_build_results_bundle` に `pic1d` 結果を追加
  (done の result をそのまま格納/復元)。

## テスト (backend/tests/test_pic1d.py — 新規)

1. **Poisson 精度**: 粒子なし・V_L=100, V_R=0 → φ が線形 (rtol 1e-12)、E 一様。
2. **リープフロッグ**: 一様 E 中の単電子 (MCC なし) の軌道が解析解と2次精度で一致。
3. **電荷保存**: 堆積後の Σρ·vol = Σq·w (端節点の半体積込み、rtol 1e-12)。
4. **CCP スモーク (eduPIC Ar 解析式断面積)**: gap 2.5 cm、10 Pa、13.56 MHz 250 V、
   n_cells 64、n_macro 4000、dt=1/(400f)、n_steps 2400 (6 RF周期) 程度 (CI で
   1分以内目安)。アサーション (定常前でも成り立つ頑健なもの):
   - 全プロファイル有限
   - 時間平均で両壁近傍のシース: n_e < n_i (壁から数セル)
   - バルク中央で n_i が壁近傍より大きい
   - 電離イベントが発生している (ionization rate > 0 どこかで)
5. **続きから**: run(600) + continue(600) == run(1200) がビット一致
   (history・profiles・粒子状態)。
6. **SEE**: see_gamma=1.0 でイオン吸収数 ≒ SEE 放出数。
7. **validator**: gap_m<=0、eedf_regions 5個、use_dsmc_gas=True で ValueError。
8. **プリセット**: edupic_ar の断面積が閾値未満で 0・閾値超で正、Turner Case 1 の
   数値 (gap, pressure, freq) が上記どおり。

`python -m pytest tests/ -q` **全件パス** (既存 226 + 新規)。

## 注意

- コメントは日本語で「なぜ」を書く既存スタイル。git commit はしない。
- 既存の 2D コード (pic.py / mcc.py / server.py) の規約・命名・メッセージ形式を
  必ず読んでから書くこと。特に波形合成・EEDF・phase_bins・continue は 2D と
  同じ振る舞いに揃える。
- 既存テスト・既存機能に影響を与えない (Project に optional フィールド追加のみ)。
- 検証: `cd backend && python -m pytest tests/ -q` 全件パス。
