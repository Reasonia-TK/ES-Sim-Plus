# 配布パッケージ化ガイド (Tauri サイドカー + PyInstaller)

エンドユーザーが「exe 一つで起動」できる配布形態のビルド手順。

構成: Python バックエンド (FastAPI/uvicorn) を PyInstaller で**フォルダ (onedir)** にまとめ、
Tauri のアプリに同梱する。配布ビルドのアプリはウィンドウ起動時にバックエンドを `--port 8317` で
自動起動し、終了時に子プロセスを kill する。GPU (CuPy) は CUDA Toolkit から NVRTC だけを同梱して動かす
(prompts/133)。

> **移行中 (prompts/133)**: P8b でバックエンドを onefile から onedir に変えた。Tauri 側 (下の手順 3) を
> onedir をリソースとして同梱する形に変えるのは P8c。それまで `npm run tauri build` は旧来の
> `externalBin` (単一 exe) を前提にしたまま。

## 関連ファイル

| ファイル | 役割 |
|---|---|
| `backend/run_server.py` | 配布用エントリポイント (uvicorn を 127.0.0.1:8317 で起動、`--port` 対応、`selftest`・`batch`) |
| `backend/es_sim_server.spec` | PyInstaller 仕様 (onedir、gmsh 共有ライブラリ、CuPy とヘッダ、NVRTC だけを `cuda/bin/x64` に) |
| `backend/pyi_rth_es_sim.py` | 実行時フック (CUDA_PATH を同梱の NVRTC に向ける) |
| `backend/es_sim/selftest.py` | 自己テスト (`es-sim-backend selftest`: gmsh・numba・pyamg・ezdxf・boltzpmp・GPU の計算) |
| `scripts/build_backend.ps1` | Windows 用バックエンドビルド (onedir) + CUDA の環境変数なしでの自己テスト |
| `scripts/build_backend.sh` | Linux/macOS 用 (検証用) |
| `ui/src-tauri/binaries/` | サイドカー配置先 (**target triple 付きファイル名**。コミットしない) |
| `ui/src-tauri/tauri.conf.json` | `bundle.externalBin: ["binaries/es-sim-backend"]` |
| `ui/src-tauri/src/main.rs` | サイドカーの spawn / kill (配布ビルドのみ、終了時はプロセスの木ごと) |

## Windows での配布ビルド全手順

前提: uv、Rust (stable, MSVC)、Node.js、Visual Studio Build Tools。GPU の確認には NVIDIA の GPU とドライバ。

```powershell
# 1. バックエンドの venv 準備 (依存グループ dist = PyInstaller と NVRTC の wheel。開発用も残すなら --extra dev も)
cd ES-Sim\backend
uv sync --extra gpu --group dist

# 2. バックエンドを onedir にまとめ、CUDA の環境変数なしで自己テスト (GPU のある PC なら -RequireGpu)
cd ..
powershell -ExecutionPolicy Bypass -File scripts\build_backend.ps1 -RequireGpu
#  → backend\dist\es-sim-backend\ (es-sim-backend.exe と _internal\、約 490 MB) ができる

# 3. フロントエンド (UI v2) + Tauri の配布ビルド
cd ui
npm install          # 初回のみ
npm run tauri build
```

生成物の場所:

- インストーラ: `ui\src-tauri\target\release\bundle\nsis\ES-Sim_<ver>_x64-setup.exe`
  (および `msi\ES-Sim_<ver>_x64_en-US.msi`)
- 実行ファイル本体: `ui\src-tauri\target\release\es-sim.exe` (`npm run tauri build -- --no-bundle` ならインストーラを作らずこれだけ)
  (同ディレクトリに `es-sim-backend-x86_64-pc-windows-msvc.exe` が並置される)

## GPU (CuPy と NVRTC、prompts/133)

- 同梱する CUDA は NVRTC (`nvrtc64_130_0.dll`・`nvrtc-builtins64_130.dll`、約 95 MB、依存グループ dist の
  `nvidia-cuda-nvrtc` 13.0 の wheel から) と NVIDIA の使用許諾 (`_internal/cuda/License.txt`) だけ。GPU の計算は
  cuBLAS・cuSOLVER・cuSPARSE を使わない (P8a。`tests/test_v2_gpu_nvrtc_only.py` が NVRTC だけの環境で GPU の
  テストを走らせて確かめる)。CUDA のランタイムは NVIDIA のドライバに含まれる。
- 実行時フック (`pyi_rth_es_sim.py`) が `CUDA_PATH` を `_internal/cuda` に向ける。利用者の PC の CUDA Toolkit の
  有無・版によらず同梱の NVRTC を使う。
- GPU を使う条件: NVIDIA のドライバが CUDA 13 に対応 (R580 以降)、GPU が Compute Capability 7.5 以降 (Turing
  以降)。満たさなければ CPU で計算し、理由をバージョン情報とステータスバーのツールチップに出す。
- 最初の GPU の計算ではカーネルを NVRTC でコンパイルする (全部で約 4 秒)。結果は `%USERPROFILE%\.cupy\kernel_cache`
  に残り、2 回目からは速い。
- 配布物の確認: `backend\dist\es-sim-backend\es-sim-backend.exe selftest [--require-gpu]` (結果は 1 行ずつ、
  `--json PATH` で JSON にも)。GPU を使えないときは理由を表示する。

## 開発モードとの違い

- `npm run tauri dev` (デバッグビルド) では**サイドカーを起動しない**
  (`main.rs` の `cfg!(debug_assertions)` で分岐)。従来通り手動でバックエンドを起動する:

  ```bash
  cd backend && uvicorn es_sim.server:app --port 8317
  ```

- 配布ビルド (`npm run tauri build` の成果物) は起動時にサイドカーを自動 spawn し、
  アプリ終了時に kill する。フロントエンドはどちらのモードでも
  `127.0.0.1:8317` に接続する。

## ポート競合時の対処

既定ポートは 8317。別プロセスが使用中だとバックエンドが起動できない
(アプリは開くが計算 API に接続できない)。

- 使用中プロセスの確認: `netstat -ano | findstr :8317` → `taskkill /PID <pid> /F`
- 前回のアプリが異常終了して `es-sim-backend-*.exe` が残っている場合は
  タスクマネージャーから終了する
- ポートはアプリの「ヘルプ › バックエンドの接続…」(ステータスバーのバックエンドの表示をクリック) で変える。
  AppConfig の `backend-port.txt` に保存され、次に起動したときサイドカーもその番号で起動する

## トラブルシューティング

- **gmsh の DLL 不足** (`gmsh-4.xx.dll が見つかりません` / ImportError):
  `es_sim_server.spec` が pip の gmsh パッケージから共有ライブラリを検出して
  バンドル直下へ同梱する。ビルド環境で `pip show gmsh` が通ること、
  `python -c "import gmsh"` が成功することを確認してから再ビルドする。
  それでも失敗する場合は spec の検出ログ (SystemExit メッセージ) を確認。
  Windows では gmsh が依存する MSVC ランタイム (vc_redist x64) が
  ターゲット PC に必要な場合がある
- **アンチウイルス誤検知**: PyInstaller 製 exe は誤検知されることがある。
  本 spec は誤検知の一因になる UPX 圧縮を無効にしている。配布時は
  コード署名を推奨。開発中は該当フォルダを除外設定にする
- **起動が遅い**: onefile 形式は初回起動時に一時フォルダへ自己解凍するため
  数秒かかる。恒常的に問題なら spec を onedir 構成へ変更する
  (その場合 externalBin ではなく `bundle.resources` での同梱に変更が必要)
- **`failed to bundle ... externalBin`**: `ui/src-tauri/binaries/` に
  triple 付きバイナリが無い。`scripts/build_backend.*` を先に実行する
- **サイドカーが終了しない**: 通常の終了ではプロセスの木ごと止める (PyInstaller の onefile は展開役と本体の
  2 つのプロセスで動く)。アプリを強制終了したときは残ることがあるので、タスクマネージャーで `es-sim-backend` を終了する
- **`Found version mismatched Tauri packages`**: `ui/package.json` の `@tauri-apps/*` と `ui/src-tauri/Cargo.lock` の
  クレートのマイナー版をそろえる (今は api 2.11・plugin-fs 2.5・plugin-dialog 2.7)

## Linux での検証 (この構成の動作確認)

```bash
cd backend && pip install -e . pyinstaller
bash ../scripts/build_backend.sh     # dist/es-sim-backend → binaries/es-sim-backend-<triple>
./dist/es-sim-backend --port 8317 &  # /health, /solve が応答することを確認
cd ../ui/src-tauri && cargo check
```

## GitHub Actions によるリリースビルド

`v*` タグを push すると GitHub Actions (windows-latest) が上記手順を自動実行し、
生成されたインストーラ (NSIS `.exe` / `.msi`) を GitHub Release に添付する
(`.github/workflows/release.yml`)。手動実行 (workflow_dispatch) では Artifacts にのみ保存される。

```powershell
git tag v0.1.0
git push origin v0.1.0
```
