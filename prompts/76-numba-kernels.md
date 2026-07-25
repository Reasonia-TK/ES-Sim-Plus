# 76: 粒子カーネルの Numba JIT 化 (高速化②)

## 背景

①の計測 (backend/benchmarks/pic_bench.py、2万粒子・MCCあり) の結果:
walk 54% / gather_push 18.5% / MCC 15% / deposit 10.8% / solve 1.5%。
支配的な粒子カーネルを Numba njit 化して数倍の高速化を狙う。

## 方針

- **numba を optional 依存**にする: import に失敗したら従来の numpy 実装へ
  自動フォールバック (`try: import numba ... except ImportError`)。
  分岐は particles.py / pic.py 内の実装選択で行い、呼び出し側は変えない。
- **決定性の維持**: 乱数は従来どおり numpy Generator で「同じ順・同じ本数」を
  引き、njit カーネルには乱数配列を引数で渡す。並列 (prange) にするのは
  「粒子ごとに自分の行にしか書かない」処理だけにし、リダクション (デポジット) は
  逐次 njit にする。これにより **numpy 実装とビット単位で同一の結果**を保つ
  (等価性テストで担保)。
- njit オプション: `cache=True` (起動毎の再コンパイル回避), `nogil=True`。
  並列カーネルは `parallel=True` + prange。スレッド数は pic.threads を
  `numba.set_num_threads()` へ反映 (threads=1 なら1)。

## 対象カーネル (優先順)

1. **walk (`particles.py` の `_walk_step`)** — 最重要 (54%)。
   粒子ごと独立の探索ループを njit(parallel=True) の prange で書き直す。
   PIC と DSMC の両方から使われているため、両方が自動で速くなる。
   numba 版では PIC 側の ThreadPool チャンク分割 (_walk_chunked_submit) を
   通さず単一呼び出しにする (prange が内部で全コアを使う)。numpy フォールバック時は
   従来のチャンク並列を維持。DSMC 側 (_walk_chunked) も同様に分岐。
2. **電荷デポジット** — np.add.at を njit 逐次ループへ (決定的)。
   DSMC の _sample (bincount 群) は対象外で良い。
3. **gather + Boris push** — 節点値ギャザーと push を1つの njit ループに融合し
   中間配列を削減 (parallel=True、粒子ごと独立)。
4. **MCC** — 乱数配列・断面積テーブルを引数に渡し、候補選定〜散乱の
   粒子ループを njit 化 (時間があれば。効果は 15% 上限なので優先度低。
   工数が嵩むならスキップして報告に明記)。

## 依存・配布

- backend/pyproject.toml に `numba` を依存追加。クラウド環境では
  `pip install numba --break-system-packages` でインストールして検証。
- server の /health レスポンスに `"numba": bool` を追加
  (フロントは触らなくて良い。Health 型に optional で足すだけ)。
- **リリースビルド (Windows/PyInstaller) 対応**:
  - scripts/build_backend.ps1 (または .spec) を確認し、numba が同梱されるよう
    必要なら hiddenimports 等を追加 (pyinstaller-hooks-contrib が numba hook を
    持つので基本は自動のはず。確認してコメントを残す)。
  - server 起動時に numba カーネルを eager import する (パッケージ漏れがあれば
    /health 前に落ちて CI smoke で検出される)。
  - .github/workflows/release.yml の smoke テストに `$r.numba -eq $true` の
    アサーションを追加 (パッケージ漏れの早期検出)。

## テスト

1. 等価性: ランダム入力で numpy 版と numba 版の walk / deposit / gather+push の
   出力が完全一致 (np.array_equal)。フォールバック実装を直接呼んで比較する。
2. 既存スイート全件パス (153)。既存の決定性テスト (DSMC threads 一致等) が
   そのまま通ることが numba 化の決定性の傍証になる。
3. numba 無し環境の動作: monkeypatch 等でフォールバック経路を強制して
   スモーク1件 (可能な範囲で)。

## ベンチマーク (最終報告に含める)

`python backend/benchmarks/pic_bench.py` を numba あり/なし (環境変数等で強制
フォールバックできる仕組みがあると良い: `ES_SIM_NO_NUMBA=1`) で実行し、
位相別の前後比較を表で報告。DSMC も threads=2 で軽く前後比較。
初回 JIT コンパイル時間も報告 (cache=True の2回目以降と分けて)。

## 検証

- backend: pytest 全件。frontend: 変更があれば `npx tsc --noEmit && npx vite build`。

## 注意

- コメントは日本語で「なぜ」を書く既存スタイル (決定性の設計理由を必ず書く)。
- git commit はしない。
- MCC まで手が回らない場合は 1〜3 だけで良い (その旨を報告)。
