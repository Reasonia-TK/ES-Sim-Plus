# 配布パッケージ化ガイド (Tauri + PyInstaller の onedir)

エンドユーザーが「インストーラを実行するだけ」で使える配布形態のビルド手順 (Windows)。

構成: Python バックエンド (FastAPI/uvicorn) を PyInstaller で**フォルダ (onedir)** にまとめ、Tauri のアプリの
**リソース (`backend/`)** として NSIS のインストーラに同梱する。配布ビルドのアプリはウィンドウ起動時にバックエンドを
`--port 8317` で自動起動し、終了時にプロセスの木ごと止める。GPU (CuPy) は CUDA Toolkit から NVRTC だけを同梱して
動かすので、利用者の PC に CUDA Toolkit は要らない (prompts/133)。

## 関連ファイル

| ファイル | 役割 |
|---|---|
| `backend/run_server.py` | 配布用エントリポイント (uvicorn を 127.0.0.1:8317 で起動、`--port` 対応、`selftest`・`batch`) |
| `backend/es_sim_server.spec` | PyInstaller 仕様 (onedir、gmsh 共有ライブラリ、CuPy とヘッダ、NVRTC だけを `cuda/bin/x64` に) |
| `backend/pyi_rth_es_sim.py` | 実行時フック (CUDA_PATH を同梱の NVRTC に向ける) |
| `backend/es_sim/selftest.py` | 自己テスト (`es-sim-backend selftest`: gmsh・numba・pyamg・ezdxf・boltzpmp・GPU の計算) |
| `scripts/build_backend.ps1` | バックエンドのビルド (onedir) + CUDA の環境変数なしでの自己テスト |
| `scripts/build_app.ps1` | 配布ビルド一式 (バックエンド → 使用許諾 → Tauri → NSIS のインストーラ) |
| `scripts/verify_installer.ps1` | インストーラの確認 (インストール → UI から計算 (GPU、`-NoGpu` は CPU) → 終了 → アンインストール) |
| `scripts/collect_licenses.py` | 第三者のソフトウェアの使用許諾を `THIRD_PARTY_NOTICES.txt` にまとめる (Python・npm・Rust) |
| `.github/workflows/release.yml` | リリースビルド (タグ `v*` の push か手動実行、GitHub のランナーで CPU の確認まで) |
| `scripts/build_backend.sh` | Linux/macOS 用 (検証用) |
| `ui/src-tauri/tauri.conf.json` | アプリの設定。`bundle.active: false` (これだけでビルドすると exe だけ。開発も) |
| `ui/src-tauri/tauri.bundle.json` | 配布ビルドで重ねる設定 (`--config`): NSIS、`backend/dist/es-sim-backend/` をリソースの `backend/` に |
| `ui/src-tauri/src/main.rs` | バックエンドの起動・停止 (配布ビルドのみ、終了時はプロセスの木ごと)、`AppConfig/backend.log` |
| `ui/e2e-installed/`・`ui/playwright.installed.config.ts` | インストールしたアプリに CDP でつなぐ確認 (verify_installer.ps1 が使う) |

## Windows での配布ビルド

前提: uv、Node.js、Rust (stable, MSVC)、Visual Studio Build Tools。GPU の確認には NVIDIA の GPU とドライバ。
Tauri の CLI は初回に NSIS (と WiX) を GitHub の公式リリースから取ってきてキャッシュする。

```powershell
# 1. 準備 (初回)。依存グループ dist = PyInstaller と NVRTC の wheel。開発用も残すなら --extra dev も
cd ES-Sim\backend
uv sync --extra gpu --group dist
cd ..\ui
npm ci
cd ..

# 2. 一式: バックエンド (onedir) → CUDA の環境変数なしで自己テスト → Tauri → インストーラ
powershell -ExecutionPolicy Bypass -File scripts\build_app.ps1 -RequireGpu

# 3. 確かめる: 一時フォルダへインストール → CUDA の環境変数なしで起動 → UI から GPU の計算 → 閉じる → アンインストール
#    (GPU の無い PC は -NoGpu: GPU を隠して CPU で動くこと・GPU を使えない理由の表示・静電場を見る)
powershell -ExecutionPolicy Bypass -File scripts\verify_installer.ps1
```

- 生成物: `ui\src-tauri\target\release\bundle\nsis\ES-Sim_<版>_x64-setup.exe` (約 156 MB、インストール後 約 505 MB)。
  実行ファイル本体は `ui\src-tauri\target\release\es-sim.exe` (隣の `backend\` にバックエンド)。
- バックエンドだけ: `scripts\build_backend.ps1 [-RequireGpu]` → `backend\dist\es-sim-backend\` (約 490 MB)。
  `scripts\build_app.ps1 -SkipBackend` はそれを使ってアプリだけ作り直す。
- `npm run tauri build` を `--config src-tauri/tauri.bundle.json` なしで実行すると exe だけでインストーラは作らない
  (バックエンドの無いインストーラを作らないため)。

## 第三者のソフトウェアの使用許諾

`scripts\collect_licenses.py` (build_app.ps1 が実行) が配布物に入るものの一覧と使用許諾の文面を
`ui\src-tauri\target\licenses\THIRD_PARTY_NOTICES.txt` にまとめ、インストール先のルートに入れる (GitHub Release にも
添付)。対象は CPython と `es-sim[gpu]` と `nvidia-cuda-nvrtc` の実行時の依存 (再帰、PyInstaller・pytest など作る
ときだけのものは除く)、UI の npm の `dependencies` (再帰)、Tauri のアプリの Windows 向けの通常の依存 (cargo
metadata)。gmsh は GPL-2.0 以降 (ソースは https://gmsh.info/)、NVRTC は NVIDIA の使用許諾 (再配布できる部品)。
ES-Sim 自身は GPL-3.0-or-later で、NVIDIA CUDA のライブラリ (同梱の NVRTC など) と組み合わせて配ることを許す追加許可
(GPL v3 第 7 条) を付けている (リポジトリの `LICENSE` の冒頭、インストール先の `LICENSE.txt`)。gmsh (GPL-2.0 以降) の
作者はこの追加許可を出していないので、gmsh と NVRTC を同じ配布物に入れることには曖昧さが残る。

## 署名 (後から足す)

今は署名しない (prompts/133 の判断)。署名するときは「引数のファイル 1 つに署名する」PowerShell のスクリプトを
用意し、環境変数 `ES_SIM_SIGN_SCRIPT` に指定して `scripts\build_app.ps1` を実行する。`build_backend.ps1` が
バックエンドの署名の無い exe・dll・pyd 全部に (Smart App Control の PC で動かすには同梱のもの全部に要る)、Tauri が
アプリの exe とインストーラに (`bundle.windows.signCommand`) 同じスクリプトで署名する。

```powershell
# sign.ps1 (例。証明書・タイムスタンプのサーバは持っているものに合わせる)
param([string]$File)
& signtool.exe sign /fd sha256 /tr http://timestamp.digicert.com /td sha256 /a $File
if ($LASTEXITCODE -ne 0) { throw "署名に失敗しました: $File" }
```

```powershell
$env:ES_SIM_SIGN_SCRIPT = "C:\path\to\sign.ps1"
powershell -ExecutionPolicy Bypass -File scripts\build_app.ps1 -RequireGpu
```

## インストーラ (NSIS)

- 今のユーザーだけにインストール (既定は `%LOCALAPPDATA%\ES-Sim`、管理者権限は要らない)。WebView2 は Windows 11 に
  入っている (無ければインストーラが取ってくる)。
- 黙ってインストール: `ES-Sim_<版>_x64-setup.exe /S [/NS] [/D=<フォルダ>]` (`/NS` はショートカットを作らない、`/D=`
  は最後に・引用符なし)。アンインストール: `<フォルダ>\uninstall.exe /S`。
- MSI は作らない (Tauri の MSI はマシン全体へのインストールで管理者権限が要り、確かめていないため。要るときは
  `tauri.bundle.json` の `targets` に `"msi"` を足す)。
- コード署名はしていない (prompts/133 の判断: 当面は署名しない)。ダウンロードしたインストーラを実行すると
  SmartScreen の警告が出る (「詳細情報」→「実行」)。Smart App Control が有効な PC では起動できない。

## 動作条件 (利用者の PC)

- Windows 10/11 (64 ビット)。
- GPU で計算するには: NVIDIA の GPU (Compute Capability 7.5 以降 = Turing 以降) と、CUDA 13 に対応したドライバ
  (R580 以降)。CUDA Toolkit は要らない。条件を満たさなければ CPU で計算し、理由をバージョン情報 (ヘルプ ›
  バージョン情報) とステータスバーのバックエンドの表示のツールチップに出す。

## GPU (CuPy と NVRTC、prompts/133)

- 同梱する CUDA は NVRTC (`nvrtc64_130_0.dll`・`nvrtc-builtins64_130.dll`、約 95 MB、依存グループ dist の
  `nvidia-cuda-nvrtc` 13.0 の wheel から) と NVIDIA の使用許諾 (`_internal/cuda/License.txt`) だけ。GPU の計算は
  cuBLAS・cuSOLVER・cuSPARSE を使わない (P8a。`tests/test_v2_gpu_nvrtc_only.py` が NVRTC だけの環境で GPU の
  テストを走らせて確かめる)。CUDA のランタイムは NVIDIA のドライバに含まれる。
- 実行時フック (`pyi_rth_es_sim.py`) が `CUDA_PATH` を `_internal/cuda` に向ける。利用者の PC の CUDA Toolkit の
  有無・版によらず同梱の NVRTC を使う。
- 最初の GPU の計算ではカーネルを NVRTC でコンパイルする (全部で約 4 秒)。結果は `%USERPROFILE%\.cupy\kernel_cache`
  に残り、2 回目からは速い。
- 配布物の確認: `<インストール先>\backend\es-sim-backend.exe selftest [--require-gpu]` (結果は 1 行ずつ、
  `--json PATH` で JSON にも)。GPU を使えないときは理由を表示する。

## 開発モードとの違い

- `npm run tauri dev` (デバッグビルド) では**バックエンドを起動しない** (`main.rs` の `cfg!(debug_assertions)` で
  分岐)。従来通り手動でバックエンドを起動する:

  ```bash
  cd backend && uvicorn es_sim.server:app --port 8317
  ```

- 配布ビルドは起動時にリソースの `backend\es-sim-backend.exe` を起動し、アプリ終了時に止める。フロントエンドは
  どちらのモードでも `127.0.0.1:8317` (または設定したポート) に接続する。

## ポート競合時の対処

既定ポートは 8317。別プロセスが使用中だとバックエンドが起動できない (アプリは開くが計算 API に接続できない)。

- 使用中プロセスの確認: `netstat -ano | findstr :8317` → `taskkill /PID <pid> /F`
- 前回のアプリが異常終了して `es-sim-backend.exe` が残っている場合はタスクマネージャーから終了する
- ポートはアプリの「ヘルプ › バックエンドの接続…」(ステータスバーのバックエンドの表示をクリック) で変える。
  AppConfig (`%APPDATA%\com.tk.es-sim`) の `backend-port.txt` に保存され、次に起動したときバックエンドもその番号で起動する

## トラブルシューティング

- **GPU を使わない**: ヘルプ › バージョン情報の「GPU を使えない理由」を見る (ドライバが古い・GPU が対象外など)。
  `<インストール先>\backend\es-sim-backend.exe selftest` でも確かめられる。AppConfig の `backend.log` の
  「ES-Sim backend: GPU あり/なし (...)」の行にも残る
- **バックエンドにつながらない**: AppConfig の `backend.log` を見る (`backend not found` ならインストールが壊れている、
  `spawn error` なら起動の失敗、`stderr:` の行は Python のエラー)
- **gmsh の DLL 不足** (`gmsh-4.xx.dll が見つかりません` / ImportError): `es_sim_server.spec` が pip の gmsh パッケージ
  から共有ライブラリを検出して `_internal` へ同梱する。ビルド環境で `python -c "import gmsh"` が成功することを
  確認してから再ビルドする。Windows では gmsh が依存する MSVC ランタイム (vc_redist x64) がターゲット PC に必要な
  場合がある
- **アンチウイルス誤検知**: PyInstaller 製 exe は誤検知されることがある。spec は誤検知の一因になる UPX 圧縮を
  無効にしている。開発中は該当フォルダを除外設定にする
- **バックエンドが終了しない**: 通常の終了ではプロセスの木ごと止める。アプリを強制終了したときは残ることがあるので、
  タスクマネージャーで `es-sim-backend` を終了する
- **閉じようとすると「未保存の変更」が出る**: 文書に保存していない変更があるとき (保存・保存しない・キャンセル)
- **`Found version mismatched Tauri packages`**: `ui/package.json` の `@tauri-apps/*` と `ui/src-tauri/Cargo.lock` の
  クレートのマイナー版をそろえる (今は api 2.11・plugin-fs 2.5・plugin-dialog 2.7)

## Linux での検証 (バックエンドの凍結の確認)

```bash
cd backend && uv sync --group dist        # または pip install -e . pyinstaller
bash ../scripts/build_backend.sh         # dist/es-sim-backend/ (onedir) と自己テスト
./dist/es-sim-backend/es-sim-backend --port 8317 &   # /health, /solve が応答することを確認
cd ../ui/src-tauri && cargo check
```

## GitHub Actions によるリリースビルド

`v*` タグを push すると GitHub Actions (windows-latest) がインストーラを作り、`THIRD_PARTY_NOTICES.txt` と一緒に
GitHub Release に添付する (`.github/workflows/release.yml`)。手動実行 (workflow_dispatch) では Artifacts にのみ
保存される。手順は `scripts\build_app.ps1` と同じ (uv の venv に gpu と dist、npm ci、Rust) で、最後に
`scripts\verify_installer.ps1 -NoGpu` (インストール・同梱のバックエンドの自己テスト・アプリを起動して UI から
静電場・閉じる・アンインストール) まで行う。GitHub のランナーには GPU が無いので、GPU の確認は GPU のある PC で
`scripts\verify_installer.ps1` を走らせる。タグは `ui/src-tauri/tauri.conf.json` の `version` と同じにする
(違うと止める)。失敗したときは backend.log と Playwright の結果を Artifacts (`es-sim-release-logs`) に残す。
