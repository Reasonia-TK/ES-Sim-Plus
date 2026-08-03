# 115: 流体モデルの並列計算対応 — 陰解法の numba 並列反復解法化

## 背景 (ユーザー要望)

「流体モデルも並列計算に対応してほしい」
実測プロファイル (2455 節点・200 ステップ): **transport 65% + energy 28% = 93%**
が毎ステップの `spsolve` (SuperLU の都度分解、逐次) に費やされている。
Poisson は初期化時 splu の後退代入のみで 7%。
→ 3本の陰解法を **numba 並列 (prange 行並列 CSR matvec) の BiCGSTAB** に置換
するのが最も効く。SuperLU の直接分解は並列化できないが、反復法の matvec は
行並列で完全にスケールし、しかもビット決定論を保てる。

## 設計

### 1. 反復解法 (backend/es_sim/_numba_kernels.py に追加)

- `csr_matvec_parallel(indptr, indices, data, x, out)`: prange 行並列。
  各行は1スレッドが昇順に積和 → **スレッド数に依らずビット同一** (理由コメント)。
- `bicgstab(indptr, indices, data, b, x0, rtol, atol, max_iter, diag_inv)`:
  Jacobi (対角) 前処理付き BiCGSTAB。
  - 内積・ノルムは **numpy (単一スレッド) で計算** (並列リダクションの
    加算順序非決定を避けるため。matvec だけを numba 並列にする。理由コメント)。
    → 実装は Python 駆動ループ + numba matvec の構成で良い
    (1反復あたり matvec 2回が支配的。Python オーバーヘッドは行列規模で償却)。
  - 収束判定: ||r|| ≤ max(rtol·||b||, atol)。既定 rtol=1e-10, atol=1e-300。
  - 返り値: (x, n_iter, converged)。
- HAVE_NUMBA=False 時は numpy 実装 (scipy の csr @ x で可 — 単一スレッド決定論)。

### 2. fluid2d.py の組み込み

- 対象: `_implicit_transport_solve` の spsolve (n_e / n_i / energy の3本)。
  行列 M = V/dt + K は EAFE の M 行列 + 質量項で対角優位 → Jacobi-BiCGSTAB が
  数〜十数反復で収束する見込み (dt が誘電緩和スケールで小さいため)。
- 初期推定 x0 = n_old (前ステップ値。反復数を大きく減らす)。
- **フォールバック**: 収束しなかったら spsolve へ (堅牢性。発生時 warnings に
  1回だけ記録 + カウンタ)。
- `Fluid2dSettings.linear_solver: Literal["iterative", "direct"] = "iterative"`
  (direct = 従来 spsolve。比較・検証用に残す)。
- `Fluid2dSettings.threads: int = Field(0, ge=0, le=128)` — 0=自動
  (pic.py の `_auto_thread_cap` と同じ max(2, min(16, cores//2))。流用 or 同式)、
  numba set_num_threads (既存の clamp 流儀)。started メッセージに実効 threads を
  含め、timing に "solver_iters" (総反復数) を追加 (診断用。WALK_DIAG 同様
  時間合計から除外)。
- Poisson (splu) は現状維持 (7% のため。docstring に理由)。
- 1D (fluid1d) は solve_banded が十分速いため対象外 (docstring に明記)。

### 3. テスト (tests/test_fluid2d.py 等に追加)

1. **direct vs iterative の一致**: 同一ケース短時間実行で profiles が
   rtol 1e-6 一致 (rtol=1e-10 収束なら十分)。
2. **スレッド決定論**: threads=1 と threads=2 (iterative) で run_batch 結果が
   **ビット同一** (np.array_equal)。
3. **フォールバック**: max_iter=1 に強制した内部条件で spsolve フォールバックが
   動き結果が有限 (モンキーパッチ or テスト用引数)。
4. **収束カウンタ**: 通常ケースで solver_iters > 0、warnings 無し。
5. 既存テスト全パス (fluid2d の既存テストは direct 既定でない点に注意 —
   既定が iterative に変わるので、既存テストが結果値を厳密固定していれば
   rtol 緩和 or direct 明示のどちらか妥当な方で追随。**物理検証テスト
   (PIC 突き合わせ等) は iterative のまま通ること**)。

### 4. ベンチマーク (レポート用、テストには入れない)

- プロファイルに使った条件 (mesh size 8e-4、200 ステップ) で
  direct vs iterative(threads=1) vs iterative(threads=2) の ms/step を実測して
  最終レポートに記載 (サンドボックスは 2-3 コアなので threads スケールは
  参考値。ユーザー環境は 96 スレッド)。

## frontend

- types.ts: Fluid2dSettings に threads / linear_solver。
- 流体 (2D) パネル: スレッド数入力 (0=自動、DSMC の流儀) +
  ソルバー select (反復法 (推奨) / 直接法)。started の実効スレッド表示
  (「前回実行: Nスレッド」の流儀)。
- 検証: `cd frontend && npx tsc --noEmit && npx vite build`。

## 注意

- コメントは日本語で「なぜ」(行並列 matvec の決定論性、内積を numpy に残す理由、
  Jacobi 前処理で足りる理由 = 対角優位、x0 = 前ステップ値)。
- git commit はしない。ES_SIM_NO_NUMBA=1 経路でも全テストが通ること。
- 既存の pic/dsmc の numba スレッド管理 (set_num_threads clamp) と衝突しない
  こと (同一プロセスで共有されるため、実行ごとに set する既存流儀に従う)。
