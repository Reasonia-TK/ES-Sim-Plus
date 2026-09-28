# 119: ES-Sim v2 再構築計画 (プランのみ、個別実装は 120〜)

## 目的 (ユーザー要望)

「主要ソルバーは十分検証が済んでいるが、ユーザーフレンドリーな UI かつ高速な計算
アルゴリズムにして作り直したい」。追加要素:

1. LXCat 形式の電子衝突断面積ファイルへの本格対応 (参照: https://github.com/Reasonia-TK/boltzpmp)
2. CUDA 対応 + メッシュ機能向上 (参照: https://github.com/AMReX-Codes/amrex)
3. CAD 機能向上

**最重要の設計目標**: v1 の検証済み物理 (解析解テスト群・Turner ケース1 で 2% 以内) を
失わないこと。v1 は常に動く状態に保ち、v2 の各エンジンは v1 と解析解・ベンチマークで
突き合わせてから置き換える (段階的置換 = strangler 方式)。

## 決定事項 (2026-09-28、ユーザーと合意)

| 論点 | 決定 | 主な理由 |
|---|---|---|
| 計算コア | **Python + CuPy (NVRTC で CUDA C++ カーネルを実行時コンパイル) + Numba (CPU フォールバック)** | v1 と同一言語で段階移行・`es_sim` をライブラリとして import 可能に保つ。この PC で動作確認済み |
| メッシュ | **AMReX 型の直交ブロック構造 AMR + 埋め込み境界 (EB, カットセル)** を PIC/DSMC/流体の新基盤に。静電場の精密計算・軌道追跡は v1 の非構造 FEM を維持 | 粒子位置特定 O(1) (walk 不要)、GPU に最適、AMR で適応細分化 |
| UI 基盤 | **Tauri 2 継続**、フロントは React + TypeScript で新規に作り直し | 既存のリリース CI・インストーラを流用 |
| 着手順 | **計算側から**: 基盤 → LXCat + boltzpmp → GPU 静電場 → GPU PIC-MCC → UI/CAD | 新エンジンは既存 UI からも使える形で順次提供 |

### AMReX を直接リンクしない理由 (調査結果)

- AMReX 26.09 は C++20・MSVC 19.44・CUDA ≥ 12.2 が必要。**Windows は公式非サポート**で、
  上流 CI の "CUDA on Windows" はビルドのみ (CUDA 12.6 / sm_80、GPU テストなし)。
  nvcc + MSVC の拡張ラムダ問題 (amrex issue #4867) も未解決。CUDA 13 + sm_120 は上流で未テスト。
- pyAMReX / WarpX は PyPI 未公開、conda-forge の win-64 は CPU 版のみ。pyAMReX は MLMG
  (線形ソルバー) を公開していない。
- よって **設計思想を移植する**: `Box`/`BoxArray`/`MultiFab` (ゴースト付きブロック)・
  `EB2` (カットセル幾何)・`MLMG` (幾何マルチグリッド)・`ParticleContainer` (SoA + セル
  ソート + Redistribute)・WarpX の null-collision MCC。

### 実測した環境制約 (この PC)

- GPU: RTX 5070 Ti (sm_120, 70 SM, 16 GB)、CUDA 13.0 (nvcc)、ドライバ 13.1。
  MSVC Build Tools 2022 17.14 (C:\BuildTools)。CPU: Ryzen 7 7700 (8C/16T)。Python 3.13。
- **Smart App Control (SAC) が有効**: 公開直後の PyPI バイナリはブロックされる
  (numba 0.67.0、cupy-cuda13x 14.1.1/14.2.0 は DLL 読込失敗)。numba 0.66.0 と
  cupy-cuda13x 14.0.1 / 13.6.0、boltzpmp 0.5.0、shapely・ezdxf・pyamg・h5py は動作。
  ローカルでビルドした DLL は読み込める。→ 依存はバージョンを固定し、更新時は SAC 下での
  import を確認する。
- Rust 未導入 (Tauri のデスクトップビルドには rustup が必要。開発はブラウザで可能)。
- 依存先 `Syb04/boltzpm` リポジトリが消滅 (404) → `uv sync` が失敗する。後継の
  **boltzpmp** (PyPI 0.5.0、Rust 製、API 互換) に切り替える。
- ベースライン: `pytest tests/` は boltzpm 依存の 14 件を除き 364 件パス (273 s)。

### 性能の現状と試作値

| 項目 | 値 |
|---|---|
| v1 PIC (numba, 8 スレッド, ~20 万粒子) | 3.5 ms/step (5,600 万 粒子step/s)。MCC 47%・push+walk 40%・deposit 11% |
| v1 PIC (局所細分化メッシュ, prompts/88) | walk が最大 97% |
| GPU 試作: push + CIC deposit (1,000 万粒子, 1024²) | **2.4 ms/step (41 億 粒子step/s)** |
| GPU 試作: GMG Poisson (1025², ゼロ初期値→相対残差 1e-11) | **11.7 ms** (V サイクル 9 回) |

## アーキテクチャ

```
backend/es_sim/            (v1 モジュールはそのまま残し、新サブパッケージを追加)
├── xs/        LXCat 完全パーサー・断面積モデル・混合ガス・EFFECTIVE→ELASTIC 変換
├── device/    CPU/GPU 切替 (ES_SIM_DEVICE=auto|cpu|cuda)、.cu カーネルの NVRTC ロード
├── kernels/   CUDA C++ カーネル (.cu)。Numba 版と同じ数式を保つ
├── geom/      CAD 幾何 (線分・円・円弧) の解析的な交差・内外判定
├── eb/        直交格子 + 埋め込み境界 (交差率 θ・材料分率・Dirichlet 結合)
├── field/     節点 Poisson (可変 ε・EB Dirichlet・RZ・周期) + GMG-PCG
├── amr/       Box/BoxArray/階層・タグ付け・再格子化 (フェーズ P4)
└── gpic/      GPU PIC-MCC (SoA 粒子・CIC・null-collision MCC・EB 境界処理)
ui/                        v2 フロントエンド (Tauri 2 + React 19 + TS、フェーズ P6〜)
```

### 数値手法の要点 (v2 直交格子エンジン)

- **格子**: 節点中心の φ。セル数は `m·2^k` (m ≤ 15) に切り上げて GMG の階層を確保
  (要求メッシュ幅以下になる、最大 ~12% 細かい)。矩形 domain の外周は常に格子線に一致。
- **導体 (EB Dirichlet)**: CAD 形状 (多角形・円) と格子辺の交点を**解析的に**求めた
  交差率 θ で、Gibou (2002) の対称ゴーストフルイド法。2 次精度・行列対称 (SPD)。
- **誘電体**: 節点間の「チューブ」内を幅方向に複数本の線で標本化し、各線に沿って
  直列 (調和平均)・線間で並列 (算術平均) に合成した辺コンダクタンス。層状誘電体は
  界面の向きによらず厳密。
- **RZ**: r 重み付き有限体積 (2π は v1 fem.assemble と同様に落とす)。軸上節点は半セル。
- **ソルバー**: 幾何マルチグリッド (各レベルで幾何から再離散化、赤黒 GS) を前処理とする
  CG。PIC では剛性不変なので係数は一度だけ組み、毎ステップ右辺 (電荷 + Dirichlet 値の
  線形結合 ΣV_k(t)·g_k) だけ更新し、前ステップ解から開始 (warm start)。
- **PIC**: SoA 粒子、CIC (双一次) 堆積 = 節点 FV と整合するエネルギー保存型の組、
  リープフロッグ/Boris/RZ (v1 と同じ遠心力 + 角運動量保存)。EB との交差は CAD の解析形状で
  厳密に判定 (SEE・IEDF コレクタ・誘電体表面電荷用)。MCC は counter-based 乱数 (Philox)
  で GPU 上の null-collision。
- **決定論**: GPU の atomic 加算は和の順序が非決定的 → v1 のビット一致テストは v2 では
  統計的許容差のテストに置き換える (乱数列自体はシードで再現可能)。

## フェーズと完了基準

| # | 内容 | 完了基準 |
|---|---|---|
| P0 | 基盤: 依存整理 (boltzpmp 切替・SAC 対策の版固定・gpu extra を cupy-cuda13x に)、`device/` | `uv sync` が通る。既存テスト全パス (test_boltz 含む) |
| P1 | **LXCat 完全対応** (`xs/`): boltzpmp 準拠 + 拡張 (DATABASE/ガス別グループ化・EFFECTIVE→ELASTIC 変換・ATTACHMENT・ROTATION・`<->` 統計重み・3 列目運動量移行・PARAM./COLUMNS 単位・Phelps イオン形式)。混合ガス。boltzpmp ブリッジ | 合成フィクスチャで全ブロック種の往復テスト。boltzpmp の parse_lxcat と数値一致。`/lxcat/parse` (v1 互換) と `/v2/xs/parse` |
| P2 | **GPU 静電場** (`geom/` `eb/` `field/`): EB Poisson + GMG-PCG (CPU Numba / GPU CuPy) | 平行平板 1e-10、同軸で 2 次収束・容量 1% 以内、誘電体円柱 (解析解)、RZ 同心球、周期境界、v1 FEM との一致、CPU/GPU 一致 |
| P3 | **GPU PIC-MCC** (`gpic/`): 2d3v/RZ、MCC (多成分ガス・付着・異方散乱オプション)、SEE、コレクタ、平均・位相分解 | プラズマ振動 2f_pe・エネルギー保存 (v1 と同じ許容差)、eduPIC Ar で v1 PIC と一致、**Turner ケース1 で基準解 2% 以内**、v1 比の速度 |
| P4 | **AMR**: Box/BoxArray 階層・タグ付け (λ_D/h、∇n、形状近傍)・Berger–Rigoutsos・合成格子 Poisson | 細分化あり/なしで同一解 (許容差内)、界面での保存性 |
| P5 | 流体 2D・DSMC の直交 EB 移植 (GPU) | v1 の各検証テストと同等 |
| P6 | **UI v2 シェル** (`ui/`): モデルビルダーツリー、スキーマ自動生成フォーム (単位・範囲・詳細設定)、ジョブ管理 (並行実行・進捗・停止・続行)、WebGL 表示 (AMR パッチ・EB・粒子)、uPlot グラフ、i18n、ファイル (保存/別名保存/最近使ったファイル) | v1 の全ユーザー機能 (frontend 調査の「失ってはならない機能」一覧) を網羅 |
| P7 | **CAD v2**: 線・円弧・円・ポリライン、辺/面の永続 ID で任意の辺に境界条件、スナップ (端点/中点/中心/交点/垂線/接線/グリッド)、フィレット/面取り/オフセット/トリム、移動/回転/ミラー/尺度/配列、ブーリアン (和/差/積)、数値入力、パラメータ式、レイヤ、DXF 入出力 | 主要操作の E2E テスト |
| P8 | 配布: PyInstaller に CuPy + CUDA ランタイム (NVRTC 等) を同梱、SAC 環境での起動確認 | インストーラから GPU 計算まで動作 |

## 進捗 (2026-09-28 時点)

| # | 状態 | 内容 |
|---|---|---|
| P0 | **完了** | boltzpmp 0.5.0 へ切替 (陰解法の既定収束判定 + 反復上限 5000。test_boltz 17 件 539 s → 4.6 s)、`[tool.uv]` で SAC 対策の版制約、gpu extra を cupy-cuda13x に、`es_sim/device/` (CPU/GPU 切替・NVRTC ローダ・`#include` 展開) |
| P1 | **完了** | `es_sim/xs/` (prompts/120)。boltzpmp の parse_lxcat と全フィールド一致 + 拡張。EFFECTIVE→ELASTIC は区分線形で厳密 (ランダム 3000 例で相対 5e-14)。`/lxcat/parse` は新パーサー経由 (ELASTIC と EFFECTIVE の二重計上を修正)、`/v2/xs/parse` 追加。分子量は質量表 → m/M の順 (LXCat の m/M は丸められているため) |
| P2 | **完了** | `geom/` `eb/` `field/`: EB 節点 FV + GMG-PCG (CPU Numba/NumPy・GPU CuPy)。同軸で 2 次収束 (容量誤差 32² 1.4e-3 → 512² 9.4e-6)、RZ 同心球 2 次収束、非整合誘電体 (直列は丸め誤差で厳密)、周期・全 Neumann、v1 FEM と一致、CPU/GPU 一致。CG 反復は格子によらず 6〜7 回。1024² を GPU 81 ms で求解 |
| P3 | **コア完了** | `gpic/` + `kernels/pic.cu`: ホスト同期ゼロの 1 ステップを CUDA Graph で再生、吸収粒子の後処理 (表面電荷・SEE・コレクタ) も GPU。物理検証 8 件 (2f_pe・エネルギー保存・衝突頻度・電離閾値/等分・RZ 規格化/遠心運動・SEE・v1 と 15% 以内) パス。**v1 比 18 倍 (20 万粒子/種) 〜 42 倍 (100 万粒子/種)** (`benchmarks/v2_bench.py pic`)。未対応: 注入・FN・マージ・DSMC 連成 (エラーで明示)。Turner ケース 1 は断面積データ (git 管理外) の入手後に実施 |
| 統合 | **完了** | `mesh.mode = "cartesian"` で既存 UI から v2 を利用 (`/mesh` `/solve` `/profile` `/ws/pic`・バッチ/スイープ)。結果はセルを三角形分割した表示用メッシュ上で v1 と同じ形。v1 未移植のソルバー (軌道追跡・DSMC・流体 2D) は同解像度の構造格子にフォールバック |
| P4a | **完了** (2026-09-29) | `amr/` (prompts/121): AMReX 型のブロック構造階層 (blocking factor・proper nesting・2:1 バランス)、タグ付け (導体・誘電体の境界近傍・指定矩形)、合成格子の節点 FV (葉セルの半チューブの和 + ぶら下がり節点の拘束 A_c = Pᵀ A P、対称正定値)、pyamg の AMG-CG (CPU)。拘束の重みは細かい側の半チューブの ∫ε dA の比 (軸対称の軸・誘電体界面でも流束が整合。線形補間だと軸で 1 次に落ち、誘電体界面で 0.3% ずれる)。細分化なし/全域細分化は一様格子と一致、界面を含め 2 次収束、ΣQ = −Σq・2W = ΣQV が厳密、細線同軸で一様 512² と同精度を 1/24 の未知数。`/mesh` `/solve` `/profile` と UI (メッシュ欄の AMR 設定・ツリー表示)。PIC は基本格子で計算 (警告) |
| P4b | **完了** (2026-09-29) | prompts/122: 合成格子の GPU AMG-PCG (pyamg の SA 階層 + 対称 Chebyshev 平滑化、同期なし・CUDA Graph 対応)、AMR 上の GPU PIC (ブロック表による O(レベル数) の所属セル判定、合成格子の堆積・電位・節点電場ステンシル、pic.cu の -DES_AMR 変種。L=0 で一様 PIC と機械精度で一致、CCP で細かい一様格子と一致)、静電場の解に基づく適応細分化 (二階差分の誤差指標)。粗細界面の自己力を測定 (PIC ノイズと同程度で熱平衡の密度に影響なし)。一様 PIC の周期境界の既存バグ 2 件と特異問題の直接法の遅さも修正 |
| P4c〜P8 | 未着手 | PIC の動的再格子化 (λ_D・密度に基づく)・自己力補正、流体・DSMC の GPU/AMR 移植、UI v2、CAD v2、配布 |

## 検証方針

- v1 のテスト群は**そのまま残して毎回走らせる** (回帰防止)。v2 の各エンジンは
  (1) 解析解、(2) v1 の同条件計算、(3) 文献ベンチマーク (Turner 2013) の 3 段で検証する。
- 性能は `backend/benchmarks/` に v1/v2 同条件のベンチを置き、数値を記録する。
- GPU の無い CI では CPU (Numba) 経路でテストする。GPU 経路は GPU 付きローカル環境で
  CPU 経路との一致テストを走らせる。

## リスクと対策

| リスク | 対策 |
|---|---|
| 離散化の変更で検証済みの精度を失う | v1 エンジンを残し、同条件比較テストを必須化。置換は検証完了後 |
| SAC で依存の新版がブロックされる | 版固定 + 更新時の import 確認。配布物のコード署名は P8 で検討 |
| 誘電体の非整合界面の精度 (カットセルの限界) | 直列/並列合成の辺コンダクタンス + AMR で界面近傍を細分化 |
| 小規模問題で GPU の起動オーバーヘッドが支配的 | CUDA Graph 化・カーネル融合。1D は専用 GPU 経路 |
| AMR 境界での PIC の自己力・重み管理 | P3 は単一レベルで完成させ、AMR PIC は P4 で慎重に (まず場のみ AMR) |
