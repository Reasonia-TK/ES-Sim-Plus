# 112: 2D 流体 Phase B — schema/server/sweep/batch 配線 (パターン複製、難易度: 低)

## 背景

prompts/111 (fluid2d.py コア、実装済み) の配線。**fluid1d の配線
(prompts/107 で実装した /ws/fluid1d・sweep/batch module) の忠実な写経**。
バックエンドのみ。

## 作業内容

1. **/ws/fluid2d** (server.py): /ws/fluid1d を手本に。
   - start/continue/stop、専用ロック・直近 sim 保持、anyio オフロード。
   - started: {type:"started", n_steps, step_offset, dt, warnings} —
     x 配列の代わりにメッシュ参照が必要だが、フロントは既存の /mesh 結果を
     持っているので **started に mesh は含めない** (フロント側でメッシュ整合を
     とる。2D PIC /ws/pic の started がどうしているか読んで同じ流儀に)。
   - frame: {type:"frame", step, t, フィールド節点配列 (phi, n_e, n_i, t_e),
     counts, elapsed_s} — 2D PIC のライブ frame の形 (picFrame) を読んで
     フロントの既存ライブ表示機構に載る形にする。
   - done: {type:"done", result: build_fluid2d_result(...)}。
2. **sweep/batch**: module "fluid2d" を fluid1d の並びで追加
   (_resolve_module 優先順は pic → pic1d → fluid1d → fluid2d、
   resolve_sweep_module は "fluid2d." プレフィクス、CLI choices、
   ws_sweep validate/echo)。fluid2d はメッシュ生成が必要 —
   Fluid2dSimulation のコンストラクタが project からメッシュを
   生成する流れを確認し、_worker から同じように呼ぶ。
3. **テスト** (tests/test_fluid2d_server.py — 新規): fluid1d_server の
   テスト構成の写経 (WS start→frame→done スモーク、continue、sweep 2ケース
   "fluid2d.init_density_m3"、batch --module auto)。小規模メッシュで
   CI 予算内。
4. `python -m pytest tests/ -q` 全件パス (現在 329 + 新規)。

## 注意

- コメントは日本語で「なぜ」。git commit はしない。frontend は触らない。
- 既存の pic/pic1d/fluid1d/dsmc/tl 配線はビット不変。
- 独自形式を発明しない — fluid1d と 2D PIC の既存配線を読んで揃える。
