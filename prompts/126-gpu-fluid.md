# 126: v2 P5c — 流体 2D の輸送を GPU で解く

計画書: `prompts/119-v2-rebuild-plan.md` (P5)。前段: `prompts/125-cartesian-fluid.md` (直交格子 + EB 版)。

## 目的

P5b の直交格子版は輸送 (種ごとの陰的な線形方程式) が CPU (numba の BiCGSTAB) のままで、2 万節点で
30 ms/サブステップかかっていた。状態をデバイスに置いたまま 1 サブステップ全体を GPU で進める。

## 実装 (es_sim/gfluid/gpu.py、kernels/fluid.cu)

- `GpuCartesianFluid2dSimulation(CartesianFluid2dSimulation)`: 輸送グラフ・物理・時間積分 (v1 の半陰的な
  サブステップ分割) は同じで、`step()` だけを GPU の手順に差し替える。
- カーネル: 係数表の log-log 補間 (fluid_coeffs.interp_loglog と同じ規則、Maxwell / boltzpm)、辺の SG 係数
  (Frost 移動度)、反応・エネルギー損失、壁 (E·n・イオン/電子の壁コンダクタンス)、SEE、Joule 加熱、
  Poisson の右辺と固定節点、統計、時間平均。
- 陰的行列は組み立てず、辺の接続リスト (節点ごとに「辺番号·2 + 向き」) の行列フリー形式:
  (M x)_k = 対角_k x_k − Σ (向きで a_e / b_e) x_相手。Jacobi 前処理 BiCGSTAB は v1 (_numba_kernels.bicgstab)
  と同じ漸化式を 5 カーネル (p 更新・v = A p̂・s 更新・t = A ŝ・x/r 更新) に分け、スカラーはデバイスに置く。
- 総和は固定ブロック数 (256 × 256 スレッド) の 2 段集約: 各ブロックが部分和を書き、次のカーネル (全ブロックが
  同じ順で) かホスト (numpy) が足す。原子演算を使わないので実行ごとにビット一致する。
- Poisson: GPU の GMG-PCG。未知数 4096 以下は密な逆行列で厳密に、それより大きいと前サブステップの解から
  k 反復ずつ積んで残差を 1 回読む (k は自動調整)。Dirichlet の無い特異な問題は同期版。
- 同期は BiCGSTAB の各反復の残差の読み出し、PCG の残差 1 回、ステップの終わりの統計 1 回 (全量・壁損失・
  発散検出・次のステップのサブステップ制御量) だけ。
- GPU の処理は専用の非ブロッキングストリームに積み、PCG の反復列と BiCGSTAB の 1 反復は CUDA Graph に
  取り込んで再生する (解のバッファ・係数配列はポインタ固定)。カーネル起動の Python 側のコストが律速だった
  ため (2 万節点: Poisson 7.4 → 1.1 ms、全体 10.5 → 3.8 ms/サブステップ)。
- n_e / n_i / w / phi はデバイスが正。属性として読むとホストへ写し、代入 (または読んだ配列の書き換え) は
  次のステップの前にデバイスへ戻す。フレーム・時間平均・位相分解・続き実行は v1 と同じ形。
- 振り分け (`make_fluid2d_simulation`): CUDA が使えて格子が 1,000 節点以上なら GPU 版 (これ未満は CPU 版と
  同等以下の速さ)。陽的検証経路・linear_solver="direct" は CPU 版。`ES_SIM_DEVICE=cpu` で常に CPU 版。

## 付随して直したこと

- 開発機のマルチスレッド OpenBLAS は LAPACK (inv・solve) の 1 回ごとに行列の大きさによらず ~1.5 s かかる
  (750 元: 1.5 s、1 スレッド 0.03 s、cuSOLVER 0.006 s)。GMG の最粗レベル・密な直接法と AMR の GPU AMG の
  密な逆行列を、GPU で使うときは cuSOLVER で求めるようにした (`field.gmg.dense_inverse(xp=)`)。GPU 版流体・
  GPU PIC の初期化が ~2.4 s 短くなる。
- v1 `Fluid2dSimulation.run_batch` の発散検出を `_state_finite` に切り出した (GPU 版はステップ末の統計を使う)。

## 検証 (tests/test_v2_gfluid_gpu.py、CUDA が無ければ skip)

| 項目 | 結果 |
|---|---|
| 固体入り CCP (RF・SEE・Frost・位相分解、200 ステップ) | 状態・履歴・時間平均・位相分解が CPU 版と 1e-14、BiCGSTAB の総反復数も同じ |
| 未知数 4096 超 (同期なしの GMG-PCG)・移動度一定 | CPU 版 (LU) と 1e-8 以内 |
| 軸対称・Boltzmann 係数モデル | CPU 版と一致 |
| ボルツマン平衡 (イオン固定・反射壁) | n_e ∝ exp(φ/Te)、状態の代入・その場の書き換えがデバイスへ届く |
| 粒子収支 | 全量変化 = 生成 − 壁損失 (反復法の誤差 1e-6 以内) |
| 特異な Poisson・続き実行・フレーム・振り分け・/ws/fluid2d | 期待どおり |

速度 (CCP 20×10 mm、円形導体ピン + 誘電体ブロック、`benchmarks/v2_bench.py fluid2d`):

| 格子 | CPU (v2) | GPU | 比 |
|---|---|---|---|
| 0.5 mm (804 輸送節点) | 1.83 ms/step | 1.84 ms/step | 1.0 |
| 0.1 mm (20,225) | 29.6 ms/サブステップ | 3.5 ms/サブステップ | 8.4 |
| 0.05 mm (80,286) | 120 ms/サブステップ | 4.3 ms/サブステップ | 28 |

## 追記: v1 の暴走対策の反映 (2026-09-29)

v1 fluid2d の暴走対策 (claude/quizzical-rosalind-96246c、4babddc・207896b・0fa8c6c) を取り込み、GPU 版の
`step` も同じ判定にした。

- サブステップ数は v1 の `_substep_count` で決める (上限 MAX_SUBSTEPS = 1000 を超えたら状態を変えずに
  ValueError)。
- サブステップの合間の停止要求: n_e・n_i・w・φ と壁損失・生成の積算をデバイス上で複製しておき、停止
  したらその場で書き戻して None を返す (CUDA Graph が配列のポインタを持つので差し替えない)。続きは
  中断なしの実行とビット一致する。
- Joule 緩和時間 τ_J: 加熱率を節点体積で割り、n_e が最大の 1e-3 以下の節点を除く。最大値は
  `fl_nemax` (ブロックごとの最大) を `fl_stats` の各ブロックが同じ順で集約する (決定的)。
- 検証 (tests/test_v2_gfluid_gpu.py): 健全な DC 放電 (100×50 mm、0/100 V) の 200 ステップでサブステップ数が
  ステップごとに CPU 版と一致 (除外を外すと step 144 で 1 ステップ 9 千万回を要求して ValueError)、上限超過で
  状態が変わらないこと、停止からの続きがビット一致すること、run_batch の停止がステップ途中でも効くこと。

## 追記: 誘電体表面の帯電の反映 (prompts/129、2026-09-29)

- q_surf (誘電体の表面電荷、全節点、2π 込み) はデバイスが正 (`_DeviceField`)。電子を解いた直後に
  `fl_surf` が壁小片の CSR (wptr、節点順) を節点ごとに走り、帯電する小片 (誘電体表面・受け持つ節点が
  Poisson の未知数) の e·dt·area·(c_i n_i (1+γ) − c_e n_e) を足す (原子演算なし、決定的)。
- `fl_poisson_rhs` に q_surf/qdiv (rz は 2π) を足す。_EbPoisson・_AmrPoisson (全 0 のマスク) の両方に効く。
- history の surf_q は `fl_sum` (固定木の部分和、ホストで順に合計) で集約する (stat の配置は変えない)。
- サブステップの合間の停止では q_surf もデバイス上でその場で戻す。
- 検証: CPU 版との突き合わせ (`_assert_same`) に q_surf と surf_q を加えて全ケース 1e-9 で一致、上限超過・
  途中停止のテストも q_surf をビット単位で確かめる。
