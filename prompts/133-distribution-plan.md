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
  DLL に cuBLAS などが無く NVRTC はそのフォルダから (Windows + GPU のときだけ、`tests/cuda_dll_guard.py`)。
  同じ環境で GPU 関係のテスト 215 件が通ることも確かめた。pytest 全体 797 passed (2 skipped)、Vitest 208。
