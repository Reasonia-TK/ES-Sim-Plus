# 133: v2 P8 — 配布 (計画)

計画書: `prompts/119-v2-rebuild-plan.md` (P8: 配布)。今の手順: `docs/PACKAGING.md`。

## 目的

119 の P8: PyInstaller に CuPy + CUDA ランタイム (NVRTC 等) を同梱し、SAC 環境での起動を確かめる。完了基準は
「インストーラから GPU 計算まで動作」。

## 今の形 (調査の結果、2026-10-01)

- **サイドカー**: PyInstaller の onefile (`backend/es_sim_server.spec`) を Tauri の `externalBin` で同梱。P6f で作ると
  **1.17 GB** (展開後 1.76 GB) で、onefile は起動のたびに一時フォルダへ全部を展開する。
- **CuPy が動かない**: `cupy_backends.cuda._softlink` が漏れて import できない (P6f)。ほかにも CuPy のヘッダ
  (`cupy/_core/include`、要素ごとの演算のカーネルを NVRTC でコンパイルするときに使う) と `nvrtc-builtins64_130.dll`
  (NVRTC が実行時に読む) が入っていなかった。
- **大きさの内訳**: CUDA の DLL が 1.33 GB (PyInstaller が CuPy の拡張モジュールのリンク先を開発機の CUDA Toolkit から
  拾った。cuBLASLt 478 MB・cuFFT 284・cuSPARSE 150・cuSOLVER 126・NVRTC 91・nvJitLink 88・cuRAND 59・cuBLAS 50)。
  ほかは llvmlite 120 (numba)・gmsh 89・SciPy 51・CuPy 39・OpenBLAS 40 (重複あり) など。
- **実際に使っている CUDA の DLL** (GPU のテスト 112 件を走らせて、プロセスに読み込まれた DLL を列挙):
  NVRTC (+ builtins)・cuBLAS・cuBLASLt・cuSOLVER・cuSPARSE・nvJitLink。cuFFT・cuRAND・cudart は読まれない (CUDA の
  ランタイムはドライバの `nvcudart_hybrid64.dll`)。cuBLAS 以下を使うのは 4 か所だけ:
  - `field/gmg.py` の `dense_inverse` (最粗レベルと小さな問題の密な逆行列、`cp.linalg.inv` = cuSOLVER)
  - 同じく `_bottom` の同期版 (`inv @ b` = cuBLAS)。非同期版はもう自前のカーネル (`dense_gemv_idx`)
  - `amr/gpu_solver.py` の残差のノルム (`cp.linalg.norm`)
  - `gfluid/gpu.py` の AMR の Poisson (`cupyx.scipy.sparse` の行列ベクトル積 = cuSPARSE)。自前の `csr_spmv` がある
- **NVRTC だけで動くか**: CUDA_PATH を NVRTC の 2 つの DLL だけを置いたフォルダにし、PATH から CUDA を除いて試すと、
  `import cupy`・要素ごとの演算・縮約・RawModule は動いた (cuBLASLt は import のときに有れば読むだけ)。CUDA の
  ヘッダは要らない (CuPy が自分のヘッダを持つ)。
- **圧縮後の大きさ** (LZMA、インストーラの目安): NVRTC 91 → 27 MB、builtins 4.5 → 0.2 MB に対し、cuBLASLt 478 → 337、
  cuSPARSE 150 → 131、cuSOLVER 126 → 97、cuBLAS 50 → 46 MB (GPU のコードがすでに圧縮されていて縮まない)。

## 方針

1. **GPU は NVRTC だけで動かす**: 上の 4 か所を自前のカーネルに置き換え (密な逆行列は対称正定値のブロック
   Gauss-Jordan、ピボットが正でなければ CPU で求め直す)、同梱する CUDA は NVRTC の 2 つの DLL (約 95 MB、圧縮
   27 MB) だけにする。cuBLAS などを同梱する案 (必要な 7 つで 989 MB、圧縮 664 MB) より**インストーラが約 640 MB 小さい**。今後 cuBLAS などを使うと配布版
   だけ壊れるので、NVRTC だけの環境で GPU のテストを走らせるテストを置く。
2. **NVRTC の出所**: NVIDIA の pip の wheel (`nvidia-cuda-nvrtc` 13.0.x、再配布できる) を配布用の依存グループに固定し、
   ローカルでも CI でも同じ版を同梱する。13.0 に合わせるのは開発機の CUDA Toolkit と同じで、ドライバの条件が最も
   ゆるい (R580 以降) ため。
3. **onedir にする**: onefile は起動のたびに数百 MB を展開するので、PyInstaller を onedir にして Tauri の
   `bundle.resources` で同梱し、`main.rs` がリソースのフォルダから起動する。`tauri dev` がバックエンドのビルドを
   要らないよう、リソースは配布ビルド用の設定 (`--config`) にだけ書く。
4. **凍結したアプリでの CuPy**: 実行時フック (PyInstaller の runtime hook) で CUDA_PATH を同梱の NVRTC のフォルダに
   向ける (利用者の CUDA Toolkit の有無・版に左右されない)。CuPy の拡張・ヘッダ・`cuda.pathfinder` を集め、ほかの
   CUDA の DLL は除く。
5. **GPU を使えないときの理由**: CUDA 13 の条件 (ドライバ R580 以降、Compute Capability 7.5 以降 = Turing 以降) と
   NVRTC の有無を `device.cuda_status()` で調べ、使えなければ CPU にして理由を `/health` と UI (バージョン情報・
   ステータスバー) に出す。
6. **確かめ方**: インストーラを一時フォルダに黙って入れ、CUDA の環境変数を消した状態でアプリを起動し、WebView2 に
   外からつないで (CDP) UI から GPU の計算を走らせ、終了でバックエンドが残らないことまで見る (スクリプトにする)。
7. **SAC とコード署名**: SAC は署名が無く評判も無いファイルを止めるので、SAC 環境での起動には署名が要る
   (証明書は利用者の判断)。署名の手順を差し込める形にしておく。

## 段階 (段ごとにコミットの承認をもらう)

| 段 | 内容 |
|---|---|
| P8a | GPU を NVRTC だけで: 4 か所を自前のカーネルに、NVRTC だけの環境で GPU のテストを走らせるテスト、`cuda_status()` の条件と理由、`/health` と UI に理由 |
| P8b | 凍結したバックエンド: 配布用の依存グループ (PyInstaller・NVRTC の wheel)、onedir の spec (CuPy・ヘッダ・NVRTC・実行時フック、ほかの CUDA を除く)、`selftest` サブコマンド、ビルドのスクリプト、大きさ・起動時間・初回の GPU の計算の測定 |
| P8c | アプリとインストーラ: リソースで同梱、`main.rs` の起動、NSIS/MSI、インストール → 起動 → UI から GPU の計算 → 終了 → アンインストールを確かめるスクリプト |
| P8d | CI のリリース (onedir・NVRTC・自己テスト)、文書 (PACKAGING・README の動作条件・第三者のライセンス)、署名の差し込み口 |

## 検証

- P8a: pytest 全体 (CPU/GPU 一致を含む) と、NVRTC だけの環境での GPU のテスト。密な逆行列は numpy と比べる。
- P8b: 凍結したバックエンドを CUDA の環境変数なしで起動し、`selftest` (GPU の静電場・PIC・DSMC・流体) と `/health`。
- P8c: 上の 6 の手順 (インストールしたアプリで GPU の計算まで)。

## P8a の記録 (2026-10-01)

- **置き換え** (GPU の計算で cuBLAS・cuSOLVER・cuSPARSE を使っていた所):
  - 密な逆行列 (`field/gmg.py` の `dense_inverse`、GMG の最粗レベル・小さな問題の直接法・AMR の AMG の最粗レベル):
    `device/linalg.py` の `spd_inverse` (`kernels/linalg.cu`、幅 32 のブロック Gauss-Jordan、ピボット無し)。判定
    (対称か・特異なら零空間が定数か) はホストで、ピボットが正でない・対称でない・有限でないときは警告して CPU で
    求め直す。numpy との差は 3e-15 (相対)、750 元 5 ms・1024 元 8 ms・4096 元 0.22 s。
  - GMG の最粗レベルの同期版 (`inv @ b`): 非同期版と同じ `dense_gemv_idx`。
  - GMG の PCG の内積 (`cupy.vdot` は cuBLAS): 自前の縮約カーネル `device.linalg.vdot` (CPU は `np.vdot` のまま)。
    NVRTC だけの環境のテストで見つかった (調べた時は見落としていた)。
  - 流体 AMR の Poisson の疎行列の積 (`cupyx.scipy.sparse` = cuSPARSE): `amr.cu` の `csr_spmv`。
  - `cupy.linalg.norm` (ベクトル) は CuPy 14 では要素ごとの演算と縮約なのでそのまま。
- **GPU を使う条件** (`device.cuda_status()`): ドライバのメジャー版が CuPy の CUDA と同じ以上 (CUDA 13 = R580 以降)、
  Compute Capability 7.5 以降、小さなカーネルを NVRTC でコンパイルして実行できる。使えないときは理由を返し、
  `/health` の `gpu` はこの結果だけ (v1 の `backend.gpu_available` も同じ条件に)。`describe()` にドライバ・NVRTC・
  Compute Capability を足した。
- **UI**: バージョン情報の計算デバイスに GPU の名前、CPU なら「GPU を使えない理由」の行。ステータスバーの
  バックエンドの表示のツールチップにも。
- **初回のコンパイル** (空のキャッシュ): es_sim の全カーネル 4.1 s (pic 0.9・pic の AMR 版 1.1・dsmc 1.0・fluid 0.7 s
  ほか)、CuPy の最初の要素ごとの演算と縮約 0.4 s。2 回目からは CuPy のディスクキャッシュ (`~/.cupy/kernel_cache`)。
- **テスト**: `tests/test_v2_gpu_nvrtc_only.py` (16): 逆行列 (大きさ 7 通り)・特異な問題の擬似逆行列が CPU 版と一致・
  正定値でない・対称でない行列は CPU へ・`cuda_status` の条件 6 通り (偽の cupy)・NVRTC だけの環境 (CUDA_PATH に
  NVRTC の 2 つの DLL だけ、PATH から CUDA Toolkit を除く) で GPU のテスト 15 件を子の pytest で走らせ、読まれた
  DLL に cuBLAS などが無いこと (Windows + GPU のときだけ、`tests/cuda_dll_guard.py`)。
  同じ環境で GPU 関係のテスト 215 件が通ることも確かめた。pytest 全体 797 passed (2 skipped)、Vitest 208。

## P8b の記録 (2026-10-01)

- **依存**: 依存グループ `dist` (`pyinstaller` 6.22.3・`nvidia-cuda-nvrtc` 13.0.88) を uv で追加。`uv sync --extra gpu
  --group dist` で入る (CI のテストは pip の `.[dev]` なので影響なし)。開発の venv でも CuPy はこの wheel の NVRTC を
  読むようになった (CUDA Toolkit と同じ 13.0)。
- **spec** (`backend/es_sim_server.spec`): onedir (`dist/es-sim-backend/` に exe と `_internal/`)。CuPy・cupy_backends・
  cupyx のサブモジュール (拡張の中の import は静的解析に掛からない) と CuPy のヘッダ・`.data`、NVRTC の 2 つの DLL を
  `_internal/cuda/bin/x64` に (使用許諾は `_internal/cuda/License.txt`)、CUDA Toolkit から拾われるほかの DLL は除く。
  es_sim はソースのまま (numba のキャッシュが `_internal/es_sim/__pycache__` に書かれることを確かめた)。Python は
  UTF-8 モード (標準出力が UTF-8)。実行時フック `pyi_rth_es_sim.py` が CUDA_PATH を `_internal/cuda` に向ける。
- **凍結して分かったこと** (どれも直した):
  - `cupy/__init__` が `cupy.testing` を LazyLoader で `find_spec` するので、テスト用でも同梱が要る (無いと
    `import cupy` が AttributeError)。
  - `cuda.pathfinder` のサブパッケージは `__init__.py` が無く `collect_submodules` が見落とす → ファイルを数える。
  - CuPy がカーネルのコンパイルで CUDA のヘッダを探すとき、cuda.pathfinder が `sys.executable -m
    cuda.pathfinder._dynamic_libs.dynamic_lib_subprocess` を子プロセスで起動する。凍結した exe ではこれが
    argparse のエラーになり CuPy が止まった → `run_server.py` が `-m cuda.pathfinder.*` を受けて実行する
    (`_run_child_module`、1 プロセスで 1 回・0.3 s)。
- **自己テスト** (`es_sim/selftest.py`、`es-sim-backend selftest [--require-gpu] [--no-gpu] [--json PATH]`): 環境・numba
  (JIT)・gmsh と FEM (平行平板の静電容量、解析解と 1e-14)・AMR の静電場 (CPU、pyamg、φ = 1000·x)・DXF (書き出しと
  読み直し)・Boltzmann (boltzpmp)・GPU (名前・ドライバ・NVRTC の場所)・GPU の静電場 (GMG、CPU と 2e-15 V)・GPU の
  AMR (AMG-PCG)・GPU の PIC (プラズマ振動 2·f_pe と 0.1%)・GPU の DSMC (閉じた箱の平衡)・GPU の流体 (CPU と 5e-14)。
- **ビルド** (`scripts/build_backend.ps1 [-RequireGpu] [-SkipTest]`): venv の PyInstaller で onedir を作り、CUDA_PATH
  などを消して PATH から CUDA Toolkit を除いた状態で自己テスト。Windows PowerShell 5.1 で通した (BOM 付き)。
  `scripts/build_backend.sh` (Linux の検証用) も onedir と自己テストに。
- **結果**: 493 MB・2384 ファイル (P6f の onefile は 1.17 GB・展開後 1.76 GB で CuPy が動かなかった)。内訳は
  llvmlite 115・NVRTC 92・gmsh 86・CuPy 58 (ヘッダ 22)・SciPy 50・OpenBLAS 41 MB など。ビルド 45〜55 s。CUDA の環境
  変数なしで起動から `/health` まで 1.5 s (GPU の確認を含む)、自己テストは全部 OK (16〜21 s)、NVRTC は
  `_internal\cuda\bin\x64` から読まれた。
- **テスト**: `tests/test_selftest.py` (3: CPU の確認が通り GPU は指定で飛ばす・GPU の確認・GPU が無いときは
  `--require-gpu` だけ失敗で JSON にも)。NVRTC だけの環境のテストは、NVRTC が CUDA Toolkit のフォルダから
  読まれていないことを見る形に (venv に wheel があるとそちらが読まれるため)。
- Tauri 側 (onedir をリソースとして同梱、`main.rs` の起動) は P8c。それまで `npm run tauri build` は旧来の
  `externalBin` のまま (`docs/PACKAGING.md` に移行中と書いた)。

## P8c の記録 (2026-10-01)

- **Tauri**: サイドカー (`externalBin`、onefile) をやめ、バックエンドの onedir をリソースの `backend/` に同梱する。
  `tauri.conf.json` は `bundle.active: false` (exe だけ。`tauri dev`・`cargo check` はバックエンドのビルドを要らない)、
  配布ビルドは `tauri.bundle.json` を `--config` で重ねる (NSIS・リソース)。バックエンドの無いインストーラは作れない。
  `main.rs` はリソースの `backend\es-sim-backend.exe` を `--port` 付きで起動 (無ければ `backend.log` に記録して
  GUI は開く)、終了時はプロセスの木ごと止める。サイドカー用の権限 (`shell:allow-execute`) と `binaries/` を消した。
- **MSI は作らない**: Tauri の MSI はマシン全体へのインストール (管理者権限) で、ここでは確かめられないため。
  NSIS は今のユーザーだけ (`%LOCALAPPDATA%\ES-Sim`、管理者権限なし)。要るときは `targets` に `"msi"` を足す。
- **スクリプト**: `scripts/build_app.ps1 [-RequireGpu] [-SkipBackend]` (バックエンド → Tauri → インストーラ)、
  `scripts/verify_installer.ps1 [-Installer] [-KeepInstalled]` (一時フォルダへ `/S /NS /D=` でインストール →
  同梱のバックエンドの自己テスト → CUDA の環境変数を消して PATH から CUDA Toolkit を除き、WebView2 のデータを
  一時フォルダにし、CDP のポートを開けてアプリを起動 → `ui/e2e-installed` (Playwright を CDP でつなぐ) →
  ウィンドウを閉じてバックエンドが残らないこと → `uninstall.exe /S`)。
- **UI の確認** (`ui/e2e/gpuFlow.ts`): バージョン情報の計算デバイスが GPU、サンプルの容量結合プラズマを直交格子
  (v2。PIC は GPU だけ) にして 300 ステップ → 完了、実行のページ (実行時間の内訳・300 / 300)。開発用の E2E にも
  `e2e/gpu.spec.ts` (バックエンドが GPU を使えなければ飛ばす)。閉じる前に新規 (保存しない) にする (未保存の確認で
  ウィンドウが閉じなかった)。
- **結果** (`verify_installer.ps1` がすべて通った): インストーラ 156 MB (MSI も作っていたときは 211 MB)、
  インストール 11 s・505 MB、同梱のバックエンドの自己テストは全部 OK (NVRTC はインストール先から)、起動から
  ウィンドウ (CDP) まで 0.8 s・バックエンドにつながるまでさらに約 4 s、UI から GPU の PIC が完了 (300 ステップ
  0.7 s)、閉じるとバックエンドも止まり、アンインストールで消える。**119 の P8 の完了基準「インストーラから GPU
  計算まで動作」をこの PC で確かめた** (CUDA の無い PC は環境変数と PATH から CUDA Toolkit を除いて模した。
  別の PC での確認はしていない)。
