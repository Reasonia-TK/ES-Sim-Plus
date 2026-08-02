# 107: 流体モデル Phase C — server / sweep / batch 配線 (パターン複製、難易度: 低)

## 背景

prompts/104 計画の Phase C。Phase B (fluid1d.py、実装済み) の
`Fluid1dSimulation` / `build_fluid1d_result` をアプリに配線する。
**pic1d の既存配線の忠実な写経**が作業の大半 (バックエンドのみ)。

## 作業内容

### 1. /ws/fluid1d (backend/es_sim/server.py)

- /ws/pic1d を手本に: {cmd:"start", project} / {cmd:"continue", extra_steps,
  frame_every?, avg_steps?, phase_bins?} / {cmd:"stop"}。
- started: {type:"started", n_steps, step_offset, dt, x, warnings}
- frame (frame_every ごと): {type:"frame", step, t, phi[], n_e[], n_i[], t_e[],
  counts, elapsed_s} — 粒子サンプルは無い (流体なので)。t_e[] を含める。
- done: {type:"done", result: build_fluid1d_result(...)}
- error: {type:"error", detail}
- 専用ロック・直近 sim 保持 (continue 用)、anyio オフロード、cancel — すべて
  /ws/pic1d と同じ流儀。fluid1d.py 側に足りないフック (frame コールバック等) が
  あれば pic1d と同形で追加 (既存テストは不変)。

### 2. sweep / batch (backend/es_sim/sweep.py, batch.py)

- module に "fluid1d" を追加:
  - `_resolve_module`: pic → pic1d → fluid1d の優先順 (auto)。
  - `resolve_sweep_module`: パス "fluid1d." 始まり → "fluid1d"。
  - `_worker`: module=="fluid1d" で Fluid1dSimulation 実行、バンドル
    {"version":1, "fluid1d": build_fluid1d_result(...)}。
  - CLI --module choices に fluid1d。
- ws_sweep のリクエスト module validate に "fluid1d" を追加、started echo。

### 3. テスト (tests/test_fluid1d_server.py — 新規)

1. /ws/fluid1d start→done のスモーク (小規模設定、TestClient)。
2. continue: WS 経由 run+continue の step_offset・結果整合。
3. sweep: fluid1d 2ケース (パス "fluid1d.init_density_m3") が回り、
   GET /sweep/result/{i} に fluid1d キーと上書き値が入る。
4. batch: --module auto が fluid1d のみの project で fluid1d を選ぶ。
5. 既存テスト全パス (現在 314 + 新規)。

`python -m pytest tests/ -q` 全件パス。frontend は触らない。

## 注意

- コメントは日本語で「なぜ」。git commit はしない。
- 既存の pic/pic1d/dsmc/tl の挙動はビット不変。
- /ws/pic1d・batch の pic1d 分岐・test_sweep.py の pic1d テストを必ず読んで
  同じ構造にする (独自の形式を発明しない)。
