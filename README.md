# ES-Sim

2D スケッチ CAD と、静電場・荷電粒子・低温プラズマのシミュレーションをまとめた個人研究用のデスクトップアプリ。

**📦 ダウンロード**: Windows 用インストーラー (`ES-Sim_<版>_x64-setup.exe`) は **[リリースページ](../../releases)**
から。バックエンドと GPU の計算に要るもの (CUDA の NVRTC) を同梱していて、インストーラを実行するだけで使える
(動作条件と注意は下の「インストール (配布版)」)。

平面 (x-y) または軸対称 (r-z) の断面に電極・誘電体・空間電荷をスケッチし、そのまま静電場 (FEM)・荷電粒子の軌道
追跡・PIC-MCC (1D・2D)・プラズマ流体 (1D・2D)・DSMC (ガス流れ)・VHF 定在波の計算へ展開できる。計算エンジンは、
三角形メッシュの v1 (CPU、P1-FEM) と、直交格子 + 埋め込み境界の v2 (局所細分化 AMR、静電場・流体は CPU/GPU、
PIC・DSMC は NVIDIA GPU) の 2 系統。バックエンドは Python (FastAPI のローカルサーバー)、フロントは
Tauri 2 + React + TypeScript (UI v2、WebGL2 のビューアと uPlot のグラフ)。

**主な機能**: CAD スケッチ (DXF の読み書き) / 静電場 / 粒子軌道追跡 / PIC-MCC (1D・2D) / プラズマ流体 (1D・2D) /
DSMC / VHF 定在波 / Boltzmann 係数 / 軸対称 (r-z) / 阻止コンデンサ (自己バイアス) / 時間発展の収束の判定 /
IEDF・IADF・EEDF / RF 位相分解 / 続き実行 / パラメータスイープ・バッチ実行 / GPU (CUDA)・AMR —
**検証**: Turner ベンチマーク (He CCP) で密度プロファイルが基準解と 2% 以内で一致、GEC 基準セルの自己バイアスを
実測と比較 ([docs/VALIDATION.md](docs/VALIDATION.md)、下の「検証について」)。

仕様の全体像は **[docs/SPEC.md](docs/SPEC.md)** を参照。

## 構成

- `backend/` — Python の計算コア (FastAPI のローカルサーバー、gmsh のメッシュ、P1-FEM、粒子軌道・PIC-MCC・流体・
  DSMC、v2 エンジン `es_sim/eb`・`amr`・`gpic`・`gfluid`・`gdsmc` と CUDA カーネル `es_sim/kernels`)。
  `backend/benchmarks/` は速度の計測、`backend/scripts/` はサンプルの保存された結果を作るスクリプト
- `ui/` — UI v2 (React 19 + Vite 8、Tauri 2 のアプリ `ui/src-tauri`。[prompts/130](prompts/130-ui-v2-plan.md))
- `examples/` — サンプルプロジェクト (JSON) と保存された計算結果 `examples/results/` (詳細は下記)
- `scripts/` — 配布ビルド (`build_app.ps1`・`verify_installer.ps1` など、[docs/PACKAGING.md](docs/PACKAGING.md))
- `docs/SPEC.md` — 仕様書
- `docs/DSMC.md` — DSMC(定常ガス流れ)のアルゴリズム解説
- `docs/VALIDATION.md` / `docs/validation_report.html` — 検証記録
- `docs/PACKAGING.md` — 配布ビルドの作り方・確かめ方・トラブルシューティング
- `prompts/` — 開発時のサブエージェント指示書・計画 (下記参照)

## インストール (配布版)

- **動作条件**: Windows 10/11 (64 ビット)。インストーラは今のユーザーに入れる (既定は `%LOCALAPPDATA%\ES-Sim`、
  管理者権限は要らない)。
- **GPU**: NVIDIA の GPU (Turing 以降 = Compute Capability 7.5 以降) と CUDA 13 に対応したドライバ (R580 以降) が
  あれば、v2 エンジン (メッシュの方式「直交格子 + 埋め込み境界」) の静電場・PIC・DSMC・流体 2D が GPU で動く。
  CUDA Toolkit は要らない。条件を満たさなければ CPU で計算し、理由を「ヘルプ › バージョン情報」に出す。
  最初の GPU の計算ではカーネルのコンパイルに数秒かかる (2 回目からは速い)。
- **署名**: インストーラとアプリは署名していない。実行すると SmartScreen の警告が出る (「詳細情報」→「実行」)。
  Smart App Control が有効な PC では起動できない。
- **使用許諾**: ES-Sim は GPL-3.0-or-later (NVIDIA CUDA のライブラリについての追加許可付き、下の「ライセンス」)。同梱のソフトウェアはインストール先の
  `THIRD_PARTY_NOTICES.txt` (gmsh は GPL-2.0 以降、CUDA の NVRTC は NVIDIA の使用許諾、ほかは MIT・BSD・Apache など)。
- アンインストールは Windows の「設定 › アプリ」から。困ったときは [docs/PACKAGING.md](docs/PACKAGING.md) の
  トラブルシューティング。

## 必要環境 (開発)

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python依存関係・仮想環境の管理)
- Node.js 20+
- Rust(stable。[rustup](https://rustup.rs/) で導入。Tauri のビルドに必要)
- (GPUオプション)NVIDIA GPU + CUDA 13.x ドライバ → `uv sync --extra dev --extra gpu`
  (v2 エンジン `mesh.mode: "cartesian"` の GPU 実行に使う。CUDA カーネルは実行時に NVRTC で
  コンパイルするので nvcc・MSVC は不要。CuPy・numba は `pyproject.toml` の `[tool.uv]` で動作確認済みの版に
  制限している (公開直後のバイナリが Windows の Smart App Control で止められたことがあるため))
- (配布ビルド)依存グループ dist (PyInstaller と NVRTC の wheel) → `uv sync --extra gpu --group dist`。
  手順は [docs/PACKAGING.md](docs/PACKAGING.md)

## セットアップと起動

配布ビルド (インストーラ) の作り方と確かめ方は **[docs/PACKAGING.md](docs/PACKAGING.md)** を参照
(`scripts\build_app.ps1` → `scripts\verify_installer.ps1`)。

### 1. バックエンド

```powershell
cd backend
uv sync --extra dev
.venv\Scripts\python -c "from es_sim import _numba_kernels as k; assert k.HAVE_NUMBA, 'Numba JIT が無効です'"
.venv\Scripts\python -m uvicorn es_sim.server:app --port 8317
```

テスト(解析解・ベンチマークとの比較):

```powershell
.venv\Scripts\python -m pytest tests/
.venv\Scripts\python benchmarks\pic_bench.py --require-numba
```

### 2. フロントエンド(別ターミナル)

```powershell
cd ui
npm install
npm run tauri dev   # Tauri のアプリ (UI v2)。ブラウザだけなら npm run dev → http://localhost:1421
```

pull したあとは依存の更新に追従するため `npm install` を実行しておくこと。

アプリが起動したらステータスバー右下に `backend v<版> (GPU)` (GPU を使えなければ `(CPU)`) と表示されれば疎通OK。

### UI v2 のテスト(prompts/130)

```powershell
cd ui
npm run typecheck   # 型検査 (E2E のコードも)
npm test            # Vitest
npm run e2e         # E2E (Playwright。バックエンドと UI を別のポートで立て、PC の Chrome で操作する)
```

v1 の UI (`frontend/`) は P6f で削除した (コミット b5c5cb5 までの履歴に残る)。

## バッチ実行・パラメータスイープ (prompts/78・79・96)

複数条件を一気に流したいときは、アプリの「パラメータスイープ」(数値の欄やパラメータを掃引し、ケースの一覧と
結果を見られる) か、ヘッドレスのバッチモードを使う。バッチモードは複数のプロジェクト JSON (アプリの「保存」形式)
を別プロセスで並列実行し、アプリの「開く」でそのまま開ける「結果付きで保存」形式の JSON を書き出す。

```bash
cd backend
python -m es_sim.batch run case1.json case2.json ... --parallel 2 --out out_dir
# 配布版 (exe) でも同様に使える:
#   es-sim-backend.exe batch case1.json case2.json ... --parallel 2
```

出力は各入力ファイル名に `_results` を付けたJSON(既定。`--suffix` で変更可)。計算の種類は `--module` で
`pic`・`pic1d`・`fluid1d`・`fluid2d` から選べる (既定 `auto` はプロジェクトの設定から決める)。
`pic.threads: 0` は粒子数とCPU数から自動選択される。複数ケースを並列実行する場合は、
合計実効スレッド数(`--parallel` × 各ケースの実効スレッド数)がCPUコア数を超えないよう
`--parallel` または `pic.threads` を調整すること。

## examples/ のサンプルプロジェクト

メニュー「ファイル › サンプル」(または「開く」で `examples/*.json` を読み込む) で各機能をすぐに試せる。

| ファイル | 内容 | 見られるもの |
|---|---|---|
| `parallel_plates.json` | 平行平板+誘電体ブロック(静電場のみ) | Solve → 電位分布・等電位線・誘電体境界での屈折 |
| `coaxial.json` | 同軸円筒コンデンサ | Solve → 対数分布の電位、容量・エネルギーが解析解と一致 |
| `ccp_demo.json` | 平面2D の RF 容量結合プラズマ (CCP)。PIC-MCC 一式のショーケース | PIC実行 → RF電極(左辺、±150V/13.56MHz)・GND電極(右辺、ギャップ20mm)間のシース形成、MCC衝突(電離/励起/弾性)、両電極のコレクタ(C1/C2)によるIEDF/IADF、RF位相分解アニメーション |
| `gec_cell.json` | GEC RF 基準セル (Hargis ら、Rev. Sci. Instrum. 65, 140 (1994)) を軸対称 (r-z、左辺が軸) で簡略化した容量結合プラズマ: 直径 101.6 mm の平行平板電極・間隔 25.4 mm、下が RF (振幅 100 V・13.56 MHz、阻止コンデンサ 5 nF)・上が接地、電極を支える絶縁物 (εr 2.1、幅 4 mm) と接地シールド (幅 4 mm)、接地した容器 (半径 100 mm・高さ 85.4 mm)。Ar 13.3 Pa (0.1 Torr)・300 K、γ = 0.07。流体 2D と PIC (どちらも v2・直交格子 0.5 mm、GPU) | 流体 (2D) の実行 → RF 電極の自己バイアス (定常で約 −81 V)、電極間のプラズマとシース、RF 電極の端の電離の集中と密度の山、容器へ拡がるプラズマ、時間平均と位相分解の電位・密度・電子温度。条件は [plasma-matters.nl の GEC reference cell](https://plasma-matters.nl/applications/capacitively-coupled/gec-reference-cell.php) のモデル (Ar・0.1 Torr・100 V、PLASIMO の流体) に合わせた。既定の 4 万ステップ (20 RF 周期) は RTX 5070 Ti で約 8 分 (GPU が無いと何倍も遅い)。定常 (収束の判定 (0.1%) に合格) まで約 3400 周期かかる (「続き」で延ばす。RTX 5070 Ti で約 19 時間。保存された結果は収束したもの)。阻止コンデンサは GEC セルの 2D シミュレーションの文献値 5 nF (Rauf 2020 など。実機は 0.1 µF で、定常の自己バイアスは同じ。落ち着くまでの時間が約 20 分の 1)。自己バイアスは 0 V から始めると落ち着くまで数百周期かかるので、初期バイアスを流体で落ち着く値の近く (−85 V) に置いた。Hargis らの実測 (13 Pa、電極の電圧の基本波 \|V1\| ≈ 100 V で −66〜−71 V) より 1〜2 割大きい (prompts/134)。PIC (2D) の実行 → 自己バイアス、RF 電極・接地電極に当たるイオンのエネルギー分布 (IEDF)、中心の EEDF、時間平均と位相分解の電位・密度・電子温度。断面積は eduPIC の Ar (Phelps の解析式)、200 万粒子 (半径に比例した重み)・1 RF 周期 800 ステップ。既定の 1.6 万ステップ (20 RF 周期) は RTX 5070 Ti で約 1.5 分。定常 (収束の判定 (1%) に合格) まで 2000〜3000 周期ほど (保存された結果は 2178 周期で合格。RTX 5070 Ti で約 2.3 時間) で、自己バイアス約 −70 V (実測の範囲) と流体より実測に近い。電極間の電子密度は 2.6〜3.5×10¹⁵ m⁻³ で、中心より周辺 (r 35〜40 mm) で高い (Rauf 2020・Hara ら 2022 の GEC セルの PIC、Langmuir プローブの実測と同じ形。prompts/134・136)。簡略化: 寸法のうち絶縁物・シールド・容器は推定 |
| `egun_rz.json` | 軸対称 (r-z、下辺が軸) の簡易電子銃 | 粒子軌道追跡(/trace)→ 接地カソードからエミッタ放出した電子が、集束電極(0V、開口あり)の作る電界で軸へ向けて収束しながら加速され、陽極(+2kV)側へ到達する様子(半径方向位置が終端で縮小することを確認済み) |
| `fn_diode.json` | 平行平板の真空ナノギャップダイオード (10 µm ギャップ、10 kV、β=10) | FN電界放出(/trace)→ 陰極 (エッジ3) の表面電界 1 GV/m×β から Fowler–Nordheim 電流 (~1.1e8 A/m) を計算し、電流比例で放出した電子が陽極へ ~10 keV で到達する |

`ccp_demo.json` の衝突断面積は **実データではなく合成フィクスチャ**
(`backend/tests/data/synthetic_electron.txt` / `synthetic_ion.txt` をパースして埋め込んだもの。
`mcc.gas.name` を `"Ar(合成断面積デモ)"` として明記している)。LXCat 実データは再配布条件のため
同梱していない。**実際の物理計算を行う場合は [LXCat](https://us.lxcat.net/) からダウンロードした
実データを、アプリの LXCat インポート(またはバックエンドの `POST /lxcat/parse`)経由で
読み込むこと。** (`gec_cell.json` の断面積は eduPIC の Ar の解析式から作ったもので、同梱している)

### 保存された計算結果 (examples/results/)

計算に時間のかかる流体・PIC のサンプルは、計算した結果を `examples/results/<サンプル>.json.gz` (「結果付きで保存」と
同じ結果の束を gzip したもの) に同梱している。メニュー「ファイル › サンプル」でサンプルを開くと、この結果が
「読み込んだ実行」として結果の一覧に並び、計算しなくても時間平均・位相分解の場、グラフ、数値サマリを見られる
(GPU の無い PC でも見られる。読み込んだ実行は続きの計算はできない)。「開く」でサンプルの JSON を読んだときは
読み込まない。

| ファイル | 中身 |
|---|---|
| `gec_cell.json.gz` | 流体 2D (収束まで: 3416 周期で収束の判定 (0.1%) に合格し、最後の 2 周期を時間平均して計 3422 周期。V_dc −80.9 V) と PIC (収束まで: 2178 周期で収束の判定 (1%) に合格し、最後の 50 周期を時間平均して計 2230 周期。V_dc −70.4 V)。どちらも収束の判定の結果 (量ごとの変化と残りの変化のグラフ) が載る。作り直しは RTX 5070 Ti で流体 約 19 時間 (GPU を占有したとき。2026-10-07〜08 は別の計算と分け合って約 25 時間)・PIC 約 2.3 時間 |
| `ccp_demo.json.gz` | PIC (v1・CPU、サンプルのまま 2000 ステップ = RF 4.8 周期) |

作り直すとき (サンプルの設定で走らせ、`PLANS` の表の周期数まで「続き」で延ばし、最後の区間を時間平均する。
収束までとした種類は、収束の判定が収束とした続きのあとに平均の区間を走らせて終える (表の周期数は上限)。GPU の
計算は実行ごとにわずかに違うので、作り直すと中身が変わる。エンジンを大きく変えたときに作り直す):

```powershell
cd backend
.venv\Scripts\python scripts\build_sample_results.py [サンプルのキー ...]
```

`--only pic` のように種類を付けると、その種類だけ計算し直し、ほかの種類は今のファイルのまま残す。

## 検証について

解析解・査読付き文献の基準解との比較結果は **[docs/VALIDATION.md](docs/VALIDATION.md)** にまとめている
(HTML版: [docs/validation_report.html](docs/validation_report.html))。FEM・粒子軌道・MCC単体の各検証に加え、
PIC-MCC統合ではTurnerベンチマーク(M. M. Turner et al., *Phys. Plasmas* **20**, 013507 (2013))
ケース1(He CCP)を v1 PIC で再現し、中心密度が基準解と2%以内で一致することを確認している。

- v2 エンジンは、解析解と v1 の同じ条件の計算との一致で確かめている (段階ごとの完了基準は
  [prompts/119](prompts/119-v2-rebuild-plan.md))。
- 阻止コンデンサの自己バイアスは、電気的非対称効果 (1D の流体・PIC で理想値の 1.06〜1.35 倍) と GEC 基準セル
  (収束した流体 −81 V・PIC −70 V。Hargis らの実測は −66〜−71 V) で確かめた ([prompts/134](prompts/134-self-bias-plan.md))。
- 時間発展の収束の判定は、前の周期との比較 (0.1%) では早すぎる合格になる例 (流体 1D の CCP は 306 周期で満たすが
  そのあと電子の総数が 1.5 倍に、GEC の PIC は 272 周期で満たすがそのあと自己バイアスが約 4 V 動く) で、残りの変化を
  見積もって合格させないことを確かめた ([prompts/137](prompts/137-convergence-plan.md))。
- 軸対称の PIC の軸の近くの統計の雑音 (重みが一定だと軸の近くの電離・密度が過大になる) は、半径に比例した重みで
  取り除いた ([prompts/136](prompts/136-rz-radial-weighting-plan.md))。

## 機能一覧

### CAD / ジオメトリ

- 作図: 領域 (円弧付きの折れ線・矩形・円)、スケッチ (線・円弧・円・ポリライン)、囲まれた所から領域 (中の島は穴)
- スナップ (端点・中点・中心・交点・垂線・接線・グリッド)、数値入力 (`x, y`・`@dx, dy`・`@長さ<角度`、単位と式)
- 編集: 頂点・辺の中点のハンドル (辺を円弧に曲げる)、フィレット・面取り・オフセット・トリム・延長、
  移動・回転・ミラー・尺度・配列、ブーリアン (和・差・積、円弧と穴を保つ)
- 外周・領域・穴の円弧 (DXF と同じ bulge)、外周の辺の永続 ID (任意の辺に境界条件、辺を分けても条件が付いてくる)
- パラメータの式 (数値の欄に束縛、スイープの対象)、レイヤ (表示・ロック・色)
- DXF の読み込み・書き出し (ファイル › DXF を読み込む…/DXF に書き出す…。線・円弧・円・ポリライン、楕円・
  スプラインは折れ線に、ブロックは展開。レイヤの取り込み、閉じた形を領域に。単位はファイルの $INSUNITS か指定)
- 材料割当(電極電位、誘電体εr、空間電荷密度)、境界条件(Dirichlet / 対称 / 周期)
- 全ての設定の Undo/Redo、プロジェクトJSON保存/読込(Tauriのネイティブダイアログ経由)
- 詳細は [prompts/132](prompts/132-cad-v2-plan.md) (CAD v2)

### メッシュ・静電場

- gmsh(OCC + boolean fragment)による非構造三角形メッシュ、領域ごとのローカルサイズ指定
- 軸平行矩形 domain 向けの構造格子メッシュ(`mesh.mode: "structured"`、高速生成)
- P1-FEM ポアソンソルバー(直接法 splu)、軸対称(r-z / rz_x0)座標系対応
- 直交格子 + 埋め込み境界の v2 (`mesh.mode: "cartesian"`、幾何マルチグリッド、局所細分化。下の「v2 エンジン」)
- 電位・|E|表示、等電位線、電場ベクトル、ラインプロファイル(2点間 V/|E| + CSV出力)

### 粒子軌道追跡

- 隣接要素walk探索+リープフロッグ、境界吸収・鏡面反射、dt自動推定
- エミッタ(line/point)、電子/陽子/カスタム種、Maxwell分布ソース、着地点・TOF・最終エネルギー・入射角の集計
- 軸対称(r-z / rz_x0)座標系対応(回転法のプッシュ(3D直線移動→子午面へ回転、角運動量厳密保存、軸の近くでも発散しない)、軸を通り抜ける粒子も r ≥ 0 のまま)
- FN電界放出(Murphy-Good式+Forbes近似、電極エッジ/conductor領域表面から電流比例で放出、総電流・粒子担持電流を出力)

### PIC-MCC (2D)

- 電子+イオン2種の自己無撞着PIC(P1電荷堆積、前分解LUポアソン、リープフロッグ、2d3v速度)
- RF電圧重畳(電極・境界ごとの振幅/周波数/位相、2 周波・CSV 波形)、初期プラズマ装荷、エミッタ定常注入
- MCC衝突(null-collision: 電子 弾性/励起/電離、イオン 等方/電荷交換)+ LXCatインポート
- 二次電子放出 γ(電極・誘電体境界ごと)、複数IEDF/IADFコレクタ(最大8個)、EEDF 領域、シース端の評価ライン
- RF位相分解アニメーション、時間平均フィールド、続き実行(保持状態からの追加実行)
- FN電界放出(毎ステップの表面電界からI·dt分の電子を注入、端数持ち越しで時間平均的に厳密)
- 軸対称(rz / rz_x0)対応(リングマクロ粒子、回転法のプッシュ(3D直線移動→子午面へ回転、角運動量厳密保存、軸の近くでも発散しない)、2πr体積規格化、
  リング電荷のP1射影 f=Q·L/(2π)。注入・FN電流は[A]、表面電荷は[C]単位になる。マクロ粒子の重みは半径に比例
  (分割・同じ要素での併合、`pic.radial_weighting`、prompts/136))
- 粒子マージ (セル内の保存的な併合で粒子数を抑える、prompts/77)
- 非一様背景ガス(DSMCの定常解を mcc.use_dsmc_gas で注入。局所密度比のnull-collision、
  局所温度・流速の背景Maxwell)

### PIC-MCC (1D) ([prompts/91](prompts/91-pic1d-backend.md)〜100)

- 1d3v の専用一様格子ソルバー (三重対角の Poisson。2D より桁違いに速い)。Turner ベンチマーク・eduPIC と同じ構成
- プリセット (eduPIC の Ar・Turner の He ケース 1 の条件)、RF 重畳 (2 周波も)・CSV 波形、FN 電界放出
- シース端 (Brinkmann) の位相分解と時系列の FFT、壁の IEDF、電極の阻止コンデンサ (電気的非対称効果)

### プラズマ流体 (1D・2D) ([prompts/104](prompts/104-fluid1d-plan.md)〜116・125〜129)

- ドリフト拡散 + 電子エネルギー方程式 (局所平均エネルギー)、Scharfetter–Gummel の流束、Poisson と自己無撞着
- 1D は PIC 1D と同じ条件・プリセットで比べられる。2D・軸対称は v1 の三角形メッシュ (EAFE) と v2 の直交格子
  (局所細分化・GPU)
- 係数は断面積の Maxwell 平均か Boltzmann ソルバーの表 (下の「Boltzmann 係数」)、イオン移動度は修正 Frost 式、
  壁の IEDF (1D)、誘電体の表面電荷、電極の阻止コンデンサ (自己バイアス)

### Boltzmann 係数 ([prompts/117](prompts/117-boltzpm-backend.md)・118)

- Boltzmann ソルバー boltzpmp (伝播演算子法) で、LXCat の断面積から E/N ごとの EEDF・電子移動度・速度定数を求め、
  平均エネルギーで引く表として流体の係数に使う (局所平均エネルギー近似)

### VHF 定在波 ([prompts/101](prompts/101-tl-standing-wave.md)・102)

- 円板電極の大面積 VHF CCP の非線形の径方向伝送線路モデル (Lieberman ら 2002・Chabert ら 2004)。シースの
  非線形容量が作る高調波と、波長短縮した定在波 (実効波長・吸収電力の一様性・給電電力)

### DSMC(定常ガス流れ)

- NTC法 + VHS分子モデル(Bird標準形)。既存の三角形メッシュをセルとして使用
- 境界: 拡散反射壁(壁温)・鏡面(対称)・圧力リザーバ流入/流出・真空排気
- 定常後の時間平均で要素ごとの n・T・u・p を取得(「ガス」タブでカラーマップ表示)
- 検証: 平衡箱の n/T/p 保持(3%以内)、自由分子流の解析値一致(リザーバの1/2密度・c̄/2流速、
  5%以内)、圧力駆動チャネル流の質量収支(10%以内)

### 共通

- **阻止コンデンサ (自己バイアス)** ([prompts/134](prompts/134-self-bias-plan.md)): 導体・Dirichlet の辺・1D の
  電極に直列の阻止コンデンサを付けると、電極の直流電位 (自己バイアス) が放電に合わせて決まる (Vahedi & DiPeso
  1997 の電位と回路の同時解。Poisson は 1 ステップに 1 回のまま)。流体・PIC の全て (1D、2D の v2 の一様格子・AMR と
  v1 の三角形メッシュ、流体 2D は CPU・GPU) で使え、結果に RF 1 周期ごとの V_dc・電極の電圧の基本波 |V1|・正味の伝導電流と最後の 1 周期の電極の電位が載る
  (実行のページの数値・グラフ・RF 波形モニタ、スイープのケースの一覧)。容量は文献の値 (GEC セルの 2D
  シミュレーションの 5 nF) を既定にした
- **時間発展の収束の判定** ([prompts/137](prompts/137-convergence-plan.md)): PIC と流体 (1D・2D、全てのエンジン) で、
  RF 周期ごとの周期平均 (φ と n_e は粗いブロックの平均、電子・イオンの総数、阻止コンデンサの自己バイアス) が
  落ち着いたかを判定する。前の周期との比較ではなく、雑音を差し引いた変化 D と、変化の減り方から見積もった残りの
  変化 R (減っていると言えなければ、今の傾きでこれまでの長さの 10 倍走ったときの変化) が全ての量で閾値 (流体 0.1%・
  PIC 1%) 以下の周期が 3 回続いたら収束とする。収束した周期・時刻は結果に残り、実行のページの数値・グラフ (量ごとの
  max(D, R) の推移)・スイープのケースの一覧で見られる。「収束したら止める」をオンにすると、そこから平均区間を取って
  止める (n_steps は上限になる)。計算はほとんど増えない (GPU PIC で +1%)。PIC の量は、粒子の雑音と周期をまたぐ
  揺れがあるので、長い履歴 (周期平均のビン) の後ろ半分の回帰で傾きを検定し (場はブロックの雑音をまとめた χ² の
  検定)、減っていると言えない傾きは後ろ半分の長さだけ続くとして見積もる (流体はこれまでの 10 倍)。見逃しうる遅い
  傾き (分解能: 後ろ半分の長さで変わる分) も閾値以下になるまで合格にしない (雑音が閾値と同じくらいの量は 250〜350
  周期ほど)
- **LXCat 完全対応** (`es_sim/xs/`、`POST /v2/xs/parse`): boltzpmp のパーサーの上位互換
  (DATABASE・複数ガス・ROTATION・`<->` 統計重み・3 列目運動量移行・PARAM./COLUMNS 単位・
  Phelps イオン形式)、EFFECTIVE→ELASTIC の厳密変換、混合ガスの boltzpmp 変換
- **実行と結果** (UI v2、[prompts/130](prompts/130-ui-v2-plan.md)): ジョブ (`/v2/jobs`・`WS /v2/events`) で実行する
  (同じソルバーの同時実行・待ち行列・停止・続き)。結果は実行ごとに 2D ビュー (ライブ・時間平均・位相分解の再生) と
  グラフ (1D のプロファイルと比較・IEDF/IADF・EEDF・シース端・RF 波形・自己バイアス・収束の判定など) と数値サマリで見られ、
  結果付きで保存・読み込みできる。サンプルは保存された計算結果付き (上の「保存された計算結果」)

### v2 エンジン(`mesh.mode: "cartesian"`、[prompts/119](prompts/119-v2-rebuild-plan.md))

GUI のメッシュ設定で「直交格子+埋め込み境界 (v2・GPU)」を選ぶと、静電場・PIC・流体 2D・DSMC を新エンジンで解く。

- **埋め込み境界 (EB)**: 導体・誘電体の CAD 形状 (多角形・円) と格子線の交点を解析的に求める
  カットセル離散化 (導体は Gibou の対称ゴーストフルイド法で 2 次精度、誘電体は直列/並列合成)。
  同軸円筒の容量誤差は 512² で 1e-5 以下
- **幾何マルチグリッド前処理 CG** (AMReX の MLMG と同じく各レベルを幾何から再離散化)。
  CPU (Numba) / GPU (CuPy) 両対応、反復数は格子によらず 6〜7 回
- **局所細分化 (AMR、`mesh.amr`、[prompts/121](prompts/121-amr.md)・[122](prompts/122-amr-gpu-pic.md))**:
  AMReX 型のブロック構造階層 (blocking factor・proper nesting・隣接セルのレベル差 ≤ 1)。導体・
  誘電体の境界近傍と指定矩形を 2 倍ずつ細かくし、ぶら下がり節点を拘束で消去した合成格子
  (対称正定値) を代数マルチグリッド CG で解く (GPU: pyamg の階層 + Chebyshev 平滑化を CUDA で)。
  粗細界面を含めて 2 次精度・電荷保存。細線の同軸では一様格子の 1/24 の未知数で同じ精度。
  **PIC も同じ細分化格子の上で GPU 実行**でき (電極近傍だけ細かい CCP が全体を細かくした一様格子と
  一致)、実行中に電子の密度・温度から求めたデバイ長に合わせて格子を作り直せる (`pic_regrid_every`、
  [prompts/123](prompts/123-pic-regrid.md))。静電場は解に基づく適応細分化 (`adaptive`) も使える
- **GPU PIC-MCC**: 1 ステップを CUDA Graph で再生するホスト同期ゼロの実装。v1 (8 スレッド) 比
  18〜42 倍 (`backend/benchmarks/v2_bench.py`)。軸対称・一様 B・MCC・SEE・誘電体表面電荷・
  IEDF コレクタ・EEDF 領域・位相分解・阻止コンデンサに対応 (注入・FN 放出・粒子マージ・DSMC 連成は未対応)。
  軸対称はマクロ粒子の重みを半径に比例させ、重すぎる粒子の分割と軽すぎる粒子の対の併合でセルあたりの粒子数を
  r によらず保つ (`pic.radial_weighting`、既定オン、v1 PIC も同じ、[prompts/136](prompts/136-rz-radial-weighting-plan.md))。
  重みが一定だと軸の近くの粒子が少なく、統計の雑音で軸の近くの電離・密度が過大になる
- **GPU DSMC** ([prompts/124](prompts/124-gpu-dsmc.md)): 所属セル O(1)・固体は厳密形状の拡散反射・
  セル番号の基数ソート + 候補対ごとに並列の NTC 衝突。v1 (8 スレッド) 比 35 倍。AMR の葉セルを
  衝突・サンプリングのセルにでき、実行中に平均自由行程に合わせて格子を作り直せる
  (`dsmc_regrid_every`、[prompts/127](prompts/127-dsmc-amr.md))
- **流体 2D の直交格子版** ([prompts/125](prompts/125-cartesian-fluid.md)): v1 と同じ物理・時間積分で、
  輸送グラフを双対セルの気体部分 (固体は埋め込み境界、小セル併合) から、Poisson を EB の離散化から
  組む。v1 構造格子と 1〜2% で一致。輸送も GPU で解け ([prompts/126](prompts/126-gpu-fluid.md)、
  v2 CPU 比 8〜28 倍)、AMR の合成格子の上でも解ける ([prompts/128](prompts/128-fluid-amr.md)、
  ぶら下がり節点を含む Voronoi 有限体積。固体のまわりだけ細かくして全体を細かくした一様格子と一致)

## prompts/ について

`prompts/` には開発時にサブエージェントへ与えた作業指示書と計画 (Markdown、番号順) を残している。
各機能がどの意図・制約で実装されたかの経緯と、検証の記録を追う一次資料。実装自体の説明は本README・
`docs/SPEC.md`・コード中のdocstringを参照すること。

## 既知の制約

- PIC の軸対称(rz / rz_x0)ではマクロ粒子をリング(周方向一様)として扱う。重みを一様にしたとき
  (`radial_weighting: false`) の v1 の初期装荷は要素内の r を r̄ で代表する近似 (要素サイズ ≪ r で厳密な一様密度へ
  収束。軸近傍要素では僅かに軸寄り)。半径に比例した重み (既定) の装荷は要素内も厳密
- PIC の軸対称の半径に比例した重み (`pic.radial_weighting`) は、v1 PIC では粒子マージ (`pic.merge`) と一緒に
  使えない (merge があると一様の重みのまま。重みの調整が重なるため)
- 構造格子メッシュ(`mesh.mode: "structured"`)は軸平行の矩形 domain のみ対応。局所メッシュサイズ
  (`local_sizes`)は構造格子生成時には無視される
- MCC(衝突)・二次電子放出を無効にした無衝突条件では、壁での電子損失を補う機構がないため
  プラズマ密度が時間とともに減衰していく(定常的な放電を見るには MCC・SEE の併用を推奨)
- 粒子軌道追跡の着地点分布ヒストグラム表示は未実装 (着地点・TOF・最終エネルギー・入射角は集計値で出す。
  IEDF/IADF のヒストグラムは PIC 側のコレクタ機能で実装済み)
- LXCat実データ(`backend/tests/data/Ar*.txt`)は再配布条件のため git 管理外。テストは同梱の
  合成フィクスチャで常時実行され、実データがあれば追加検証される
- GPU (CuPy) は v2 エンジン (`mesh.mode: "cartesian"`) の静電場・PIC・DSMC・流体 2D のみ。v2 の PIC と
  DSMC は現状 GPU 専用 (CPU で回す場合は unstructured/structured の v1 エンジンを使う)。v2 の流体 2D は
  1,000 節点以上なら GPU、それ未満・陽的検証経路・`linear_solver: "direct"` は CPU。v2 は軸平行の矩形 domain のみ対応
- 流体 2D の外周の「なし (Neumann)」の辺は反射壁 (symmetry と同じ。PIC では吸収壁のまま) で、粒子を
  失わせたい外周の辺は電極 (Dirichlet) にする。誘電体の表面は吸収した電荷を蓄積して浮遊電位へ緩和する
  (v1・v2・AMR・GPU 共通、[prompts/129](prompts/129-fluid2d-surface-charge.md))
- v2 の局所細分化 (AMR) は静電場・PIC・DSMC・流体 2D に対応 (軌道追跡は基本格子)。PIC・DSMC の動的
  再格子化は時間平均区間の前だけ、流体 2D は静的な細分化のみ。粗細界面には粒子の自己力が残る (大きさは
  PIC の粒子ノイズや壁際の鏡像力と同程度、[prompts/122](prompts/122-amr-gpu-pic.md))。解に基づく
  適応細分化 (`adaptive`) は静電場のみ
- 2D の電極の阻止コンデンサ (自己バイアス) をスイープするときは、ジオメトリのパスは PIC で実行する。流体 2D で
  スイープするときはパラメータに束縛してモジュールに流体 2D を選ぶ ([prompts/134](prompts/134-self-bias-plan.md))
- バッチ実行(`python -m es_sim.batch`)はプロセスごとに独立しているため、
  `pic.mcc.use_dsmc_gas`(直前のDSMC結果をサーバー保持状態から参照する機能)は未対応
- 配布版 (インストーラ) は Windows だけ・署名なし。GPU は CUDA 13 の対象 (Turing 以降、ドライバ R580 以降) だけで、
  古い GPU・ドライバでは CPU で計算する ([prompts/133](prompts/133-distribution-plan.md))

## ライセンス

ES-Sim は [GNU General Public License version 3](LICENSE) またはそれ以降 (GPL-3.0-or-later) で配布する。
NVIDIA CUDA のライブラリ (配布物に同梱する NVRTC など、NVIDIA の使用許諾のもの) と組み合わせて配ることを許す
追加許可 (GPL v3 第 7 条) を付けている (`LICENSE` の冒頭)。
配布物 (インストーラ) に同梱している第三者のソフトウェアの使用許諾は、インストール先の `THIRD_PARTY_NOTICES.txt`
(作り方は [docs/PACKAGING.md](docs/PACKAGING.md))。

## ロードマップ

済み: v2 再構築 (計算の GPU 化・局所細分化・UI/CAD の作り直し・配布、P0〜P8、[prompts/119](prompts/119-v2-rebuild-plan.md))、
DXF の読み書き (CAD v2 P7g、[prompts/132](prompts/132-cad-v2-plan.md))、阻止コンデンサ (自己バイアス、
[prompts/134](prompts/134-self-bias-plan.md))、軸対称 PIC の半径に比例した重み ([prompts/136](prompts/136-rz-radial-weighting-plan.md))、
時間発展の収束の判定 ([prompts/137](prompts/137-convergence-plan.md))。

これからの候補:

- 配布物のコード署名 (Smart App Control が有効な PC 向け。今は署名なし、[prompts/133](prompts/133-distribution-plan.md))
- Turner ベンチマーク: ケース 1 の v2 GPU PIC での再検証 (He の断面積データの入手後)、ケース 2〜4 の追加検証
- 外部回路の一般化 (浮いた電極、直列抵抗・整合器・電力の指定。今は阻止コンデンサだけ、[prompts/134](prompts/134-self-bias-plan.md))
- v2 GPU PIC の未対応の機能 (粒子注入・FN 放出・粒子マージ・DSMC 連成)
- 粒子軌道追跡の着地点分布ヒストグラム表示
- バイナリ転送 (大規模メッシュ時の JSON 転送の負荷対策)

初期の段階 (0〜4) の完了基準は仕様書 [docs/SPEC.md](docs/SPEC.md) §12、v2 再構築の段階と完了基準は
[prompts/119](prompts/119-v2-rebuild-plan.md) を参照。
