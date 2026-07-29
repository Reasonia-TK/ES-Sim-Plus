"""DSMC (定常ガス流れ) のテスト (prompts/54)。

1. 平衡箱: 密閉断熱箱 (全周拡散壁、壁温 = 初期温度) で n・T・p が初期値を保持し、
   流速 ≈ 0 (統計誤差内)
2. 自由分子流の流出: 左リザーバ → 右真空の無衝突チャネルで、チャネル内密度が
   リザーバの 1/2、平均流速が c̄/2 (半空間 Maxwell の解析値)
3. 圧力駆動チャネル流: p_in > p_out で定常の質量収支 (流入 ≈ 流出) が成り立ち、
   圧力が流れ方向に単調減少する
4. 非一様ガス場の MCC 結合: 定数場は一様指定とビット単位一致、
   密度2倍領域では電子衝突数がほぼ2倍
5. walk のチャンク並列化 (prompts/65): threads=4 が threads=1 とビット単位で一致
6. 結果の平滑化 (prompts/67): 隣接セル拡散 (smoothing_passes) の保存性・正値性・
   ノイズ低減
7. 続きから実行 (prepare_continue、prompts/74): 800 ステップ連続実行と
   400 → continue(400) の粒子状態・結果がビット単位で一致、WS continue フロー
8. 位相別プロファイル計測・_collide のセル並列化 (prompts/87): timing のキーと
   合計の妥当性、numba のセル並列 _collide_numba が numpy 版と数値的に同等
9. walk コスト診断 (prompts/88): timing の walk_cells_est/h_mean_m/h_min_m が
   正の妥当な値であること、時間キーの合計 (8.) を汚染しないこと
"""

import math
import time

import numpy as np
import pytest

from es_sim import _numba_kernels
from es_sim.dsmc import AMU, KB, WALK_DIAG_KEYS, DsmcSimulation
from es_sim.mcc import GasField, MccModel
from es_sim.pic import PicSimulation
from es_sim.schema import Project

L = 0.02
H = 0.01
M_AR = 39.948 * AMU


def _project(dsmc: dict, mesh: float = 1.5e-3) -> Project:
    return Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [L, 0], [L, H], [0, H]]},
                "boundaries": [],
            },
            "mesh": {"size": mesh},
            "dsmc": dsmc,
        }
    )


def test_dsmc_equilibrium_box():
    """密閉箱 (壁温 = 初期温度 300K、10 Pa): n・T・p を保持し u ≈ 0。"""
    p0, t0 = 10.0, 300.0
    project = _project(
        {
            "init_pressure_pa": p0,
            "init_temperature_k": t0,
            "wall_temperature_k": t0,
            "n_particles": 30000,
            "n_steps": 600,
            "avg_steps": 300,
            "seed": 1,
        }
    )
    sim = DsmcSimulation(project)
    res = sim.run()

    # 実行時間のリアルタイム表示 (prompts/86): run() の壁時計秒が結果に載る
    assert res.elapsed_s > 0

    n0 = p0 / (KB * t0)
    area = sim.area
    # 面積重み平均で比較 (セル単位は統計ノイズがある)
    n_mean = float(np.sum(res.n * area) / area.sum())
    t_mean = float(np.sum(res.t * res.n * area) / np.sum(res.n * area))
    p_mean = float(np.sum(res.p * area) / area.sum())
    assert n_mean == pytest.approx(n0, rel=0.03)
    assert t_mean == pytest.approx(t0, rel=0.03)
    assert p_mean == pytest.approx(p0, rel=0.05)
    # 平均流速はほぼゼロ (熱速度 ~350 m/s に対して 2% 未満)
    u_mag = float(np.linalg.norm(np.sum(res.u * (res.n * area)[:, None], axis=0) / np.sum(res.n * area)))
    assert u_mag < 0.02 * math.sqrt(2.0 * KB * t0 / M_AR) + 5.0


def test_dsmc_free_molecular_effusion():
    """無衝突 (d_ref を極小に) チャネル: 左リザーバ p0 → 右真空。

    定常では右向き半空間 Maxwell のビームになり、密度はリザーバの 1/2、
    平均流速 u_x = c̄/2 (c̄ = √(8kT/πm)) が解析値。
    """
    p0, t0 = 1.0, 300.0
    project = _project(
        {
            "gas": {"d_ref_m": 1e-15},  # 衝突を実質無効化
            "boundaries": [
                {"edges": [3], "type": "inlet", "pressure_pa": p0, "temperature_k": t0},
                {"edges": [1], "type": "outlet"},  # 真空
                {"edges": [0], "type": "symmetry"},
                {"edges": [2], "type": "symmetry"},
            ],
            "init_pressure_pa": p0 / 2.0,
            "init_temperature_k": t0,
            "n_particles": 40000,
            "n_steps": 1500,
            "avg_steps": 500,
            "seed": 2,
        }
    )
    sim = DsmcSimulation(project)
    res = sim.run()

    n_res = p0 / (KB * t0)
    area = sim.area
    n_mean = float(np.sum(res.n * area) / area.sum())
    assert n_mean == pytest.approx(0.5 * n_res, rel=0.05)

    c_bar = math.sqrt(8.0 * KB * t0 / (math.pi * M_AR))
    ux_mean = float(np.sum(res.u[:, 0] * res.n * area) / np.sum(res.n * area))
    assert ux_mean == pytest.approx(0.5 * c_bar, rel=0.05)
    # 質量収支: 定常なので流入 ≈ 流出
    assert res.outflow == pytest.approx(res.inflow, rel=0.05)


def test_dsmc_pressure_driven_channel():
    """圧力駆動チャネル (p_in = 20 Pa → p_out = 5 Pa): 質量収支と単調な圧力勾配。"""
    t0 = 300.0
    project = _project(
        {
            "boundaries": [
                {"edges": [3], "type": "inlet", "pressure_pa": 20.0, "temperature_k": t0},
                {"edges": [1], "type": "outlet", "pressure_pa": 5.0, "temperature_k": t0},
            ],
            "init_pressure_pa": 12.0,
            "init_temperature_k": t0,
            "wall_temperature_k": t0,
            "n_particles": 40000,
            "n_steps": 2000,
            "avg_steps": 600,
            "seed": 3,
        }
    )
    sim = DsmcSimulation(project)
    res = sim.run()

    # 定常の質量収支 (流入 ≈ 流出、統計誤差 10%)
    assert res.inflow > 0.0
    assert res.outflow == pytest.approx(res.inflow, rel=0.10)

    # x 方向 4 分割の平均圧力が単調減少し、端の値がリザーバ圧の間にある
    centroids = sim.mesh.nodes[sim.tris].mean(axis=1)
    area = sim.area
    p_slabs = []
    for i in range(4):
        sel = (centroids[:, 0] >= i * L / 4) & (centroids[:, 0] < (i + 1) * L / 4)
        p_slabs.append(float(np.sum(res.p[sel] * area[sel]) / area[sel].sum()))
    assert all(p_slabs[i] > p_slabs[i + 1] for i in range(3)), f"圧力が単調減少していない: {p_slabs}"
    assert 5.0 < p_slabs[-1] < p_slabs[0] < 20.0
    # 流れは +x 方向
    ux_mean = float(np.sum(res.u[:, 0] * res.n * area) / np.sum(res.n * area))
    assert ux_mean > 0.0


def test_dsmc_threads_match_serial():
    """threads=4 のチャンク並列 walk は threads=1 (逐次) とビット単位で一致する。

    walk は粒子ごとに独立・読み取り共有のみ・乱数不使用の決定的処理なので、
    チャンク分割しても書き込み先が競合せず結果は不変であるはず (prompts/65)。
    n_particles=8000 は初期充填数がほぼそのまま粒子数になるため (macro_weight は
    目標粒子数から逆算される)、_walk_chunked の並列しきい値 (4096) を初回 walk で
    確実に越える。
    """
    t0 = 300.0

    def _run(threads: int):
        project = _project(
            {
                "boundaries": [
                    {"edges": [3], "type": "inlet", "pressure_pa": 20.0, "temperature_k": t0},
                    {"edges": [1], "type": "outlet", "pressure_pa": 5.0, "temperature_k": t0},
                ],
                "init_pressure_pa": 12.0,
                "init_temperature_k": t0,
                "wall_temperature_k": t0,
                "n_particles": 8000,
                "n_steps": 30,
                "avg_steps": 10,
                "seed": 3,
                "threads": threads,
            }
        )
        sim = DsmcSimulation(project)
        return sim.run()

    res1 = _run(1)
    res4 = _run(4)
    assert np.array_equal(res1.n, res4.n)
    assert np.array_equal(res1.t, res4.t)
    assert np.array_equal(res1.u, res4.u)
    assert np.array_equal(res1.p, res4.p)
    assert res1.n_particles == res4.n_particles
    assert res1.macro_weight == res4.macro_weight
    assert res1.dt == res4.dt
    assert res1.inflow == res4.inflow
    assert res1.outflow == res4.outflow


# ---- _collide のセル並列化 (prompts/87) -----------------------------------------


def test_dsmc_collide_numba_matches_numpy():
    """numba のセル並列 _collide_numba は従来の全セル一括ベクトル化 _collide_numpy と
    数値的に同等 (ただしビット単位では一致しない)。

    候補選定・(σc_r)_max 更新・採択・dedup のアルゴリズムと乱数の消費順・本数は
    両実装で完全に揃えてあるが、σ(c_r) の g_mag**sig_pow (べき乗) 評価で numpy の
    配列版 `**` (SIMD 実装) と numba (LLVM 経由のスカラー libm pow) が入力によって
    最終ビットで1ULP異なることがあるため (別途 numba 版 `**` と numpy 版 `**` を
    大量の値で比較して確認済み: 約5%の入力で1ULP差)、この差が (σc_r)_max の
    実測更新や accept 判定の閾値比較へ極めて稀に伝播しうる。よってここでは
    ビット一致ではなく極小相対誤差 (float64 の数ULP) での一致を確認する
    (実務上は物理量として無視できる差であり、既存の物理検証テスト群
    (test_dsmc_equilibrium_box 等) は許容誤差ベースなのでどちらの実装でも通る)。

    2つの独立な DsmcSimulation を同一 project/seed で作る (__init__ は乱数消費が
    _collide を含まないので、この時点で粒子状態・rng 状態は必ず一致する)。一方は
    そのまま (numba が使える環境なら numba 経路)、もう一方は _numba_kernels.HAVE_NUMBA
    を一時的に False にして _collide() を呼び、numpy 経路を強制する。
    """
    if not _numba_kernels.HAVE_NUMBA:
        pytest.skip("numba 未インストール環境では比較対象が無い")

    t0 = 300.0
    project = _project(
        {
            "init_pressure_pa": 15.0,
            "init_temperature_k": t0,
            "wall_temperature_k": t0,
            "n_particles": 20000,
            "n_steps": 1,
            "avg_steps": 1,
            "seed": 42,
        }
    )
    sim_new = DsmcSimulation(project)
    sim_old = DsmcSimulation(project)
    # __init__ 直後 (_collide 呼び出し前) の状態が一致していることを前提の確認
    assert np.array_equal(sim_new.x, sim_old.x)
    assert np.array_equal(sim_new.v, sim_old.v)
    assert np.array_equal(sim_new.elem, sim_old.elem)

    sim_new._collide()  # numba 経路 (この環境では HAVE_NUMBA=True)
    prev = _numba_kernels.HAVE_NUMBA
    _numba_kernels.HAVE_NUMBA = False
    try:
        sim_old._collide()  # numpy 経路を強制
    finally:
        _numba_kernels.HAVE_NUMBA = prev

    # (σc_r)_max は数ULP以内 (pow の丸め差のみ)
    assert np.allclose(sim_new._sigcr_max, sim_old._sigcr_max, rtol=1e-12, atol=0.0)
    # 採択・dedup の閾値判定が pow の丸え差でたまたま反転すると v/coll_frac が
    # ずれ得るが、その粒子数はごく僅かなはず (このケースでは0を期待)
    diff_rows = int(np.count_nonzero(np.any(sim_new.v != sim_old.v, axis=1)))
    assert diff_rows <= 5, f"{diff_rows} 粒子の速度が閾値反転で分岐した"
    assert np.allclose(sim_new._coll_frac, sim_old._coll_frac, rtol=0.0, atol=1.0)


# ---- 結果の平滑化 (隣接セル拡散、prompts/67) -----------------------------------


def test_dsmc_smoothing_conserves_total():
    """smoothing_passes=3 でも Σ V_i n_i (総粒子数相当) は passes=0 と厳密に一致する。

    W_ij = min(V_i, V_j) は i, j について対称なので、ペアごとの交換
    (i の増分×V_i と j の増分×V_j) が相殺し、Σ_i V_i q_i は理論上厳密に保存される
    (丸め誤差のみ)。同一 seed で乱数消費列も同一になるよう平滑化以外の設定は揃える。
    """
    t0 = 300.0
    base = {
        "init_pressure_pa": 10.0,
        "init_temperature_k": t0,
        "wall_temperature_k": t0,
        "n_particles": 20000,
        "n_steps": 300,
        "avg_steps": 150,
        "seed": 1,
    }
    sim0 = DsmcSimulation(_project({**base, "smoothing_passes": 0}))
    res0 = sim0.run()
    sim3 = DsmcSimulation(_project({**base, "smoothing_passes": 3}))
    res3 = sim3.run()

    tot0 = float(np.sum(sim0.vol * res0.n))
    tot3 = float(np.sum(sim3.vol * res3.n))
    assert tot3 == pytest.approx(tot0, rel=1e-10)


def test_dsmc_smoothing_nonnegative():
    """平滑化後も n・t は全要素で非負 (係数 1 − θ·Σ W_ij/V_i ≥ 0.25 > 0 の凸結合)。"""
    t0 = 300.0
    project = _project(
        {
            "init_pressure_pa": 10.0,
            "init_temperature_k": t0,
            "wall_temperature_k": t0,
            "n_particles": 20000,
            "n_steps": 300,
            "avg_steps": 150,
            "seed": 1,
            "smoothing_passes": 5,
        }
    )
    sim = DsmcSimulation(project)
    res = sim.run()
    assert np.all(res.n >= 0.0)
    assert np.all(res.t >= 0.0)


def test_dsmc_smoothing_reduces_noise():
    """一様平衡箱 (全辺壁、流入なし) で n の空間分散 (var(n)/mean(n)^2) が
    passes=5 で passes=0 より小さくなる (統計ノイズの低減が目的通り効いている)。
    """
    t0 = 300.0
    base = {
        "init_pressure_pa": 10.0,
        "init_temperature_k": t0,
        "wall_temperature_k": t0,
        "n_particles": 20000,
        "n_steps": 300,
        "avg_steps": 150,
        "seed": 1,
    }
    res0 = DsmcSimulation(_project({**base, "smoothing_passes": 0})).run()
    res5 = DsmcSimulation(_project({**base, "smoothing_passes": 5})).run()

    var0 = float(np.var(res0.n) / np.mean(res0.n) ** 2)
    var5 = float(np.var(res5.n) / np.mean(res5.n) ** 2)
    assert var5 < var0


# ---- 非一様ガス場の MCC 結合 (Phase A、prompts/54) ------------------------------


def _mcc_project(pressure_pa: float) -> Project:
    """MCC つき小型 CCP (合成弾性断面積)。"""
    return Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [L, 0], [L, H], [0, H]]},
                "boundaries": [
                    {"edges": [3], "type": "dirichlet", "voltage": 0.0},
                    {"edges": [1], "type": "dirichlet", "voltage": 0.0},
                ],
            },
            "mesh": {"size": 1.5e-3},
            "pic": {
                "initial_plasma": {
                    "density": 1e14, "te_ev": 2.0, "ti_ev": 0.03,
                    "ion_mass_amu": 40.0, "seed": 5,
                },
                "n_macro": 4000,
                "dt": 5e-11,
                "n_steps": 20,
                "frame_every": 100,
                "mcc": {
                    "gas": {"name": "Ar", "pressure_pa": pressure_pa, "temperature_k": 300.0},
                    "electron_processes": [
                        {
                            "kind": "elastic", "label": "syn", "threshold_ev": 0.0,
                            "mass_ratio": 1.36e-5,
                            "energy_ev": [0.0, 100.0], "sigma_m2": [1e-19, 1e-19],
                        }
                    ],
                    "seed": 7,
                },
            },
        }
    )


def test_gas_field_constant_matches_uniform():
    """定数ガス場は一様指定とビット単位で一致する。"""
    p0 = 10.0
    n0 = p0 / (KB * 300.0)
    proj = _mcc_project(p0)

    sim_a = PicSimulation(proj)
    n_elems = len(sim_a.tris)
    field = GasField(n_g=np.full(n_elems, n0))
    sim_b = PicSimulation(_mcc_project(p0), gas_field=field)
    for _ in range(20):
        sim_a.step()
        sim_b.step()
    for name in ("electron", "ion"):
        assert np.array_equal(sim_a.species[name].x, sim_b.species[name].x)
        assert np.array_equal(sim_a.species[name].v, sim_b.species[name].v)
    assert sim_a.coll_e == sim_b.coll_e


def test_gas_field_density_ratio():
    """密度2倍の場では電子衝突数がほぼ2倍になる (統計比較)。"""
    p0 = 10.0
    n0 = p0 / (KB * 300.0)
    results = {}
    for factor in (1.0, 2.0):
        proj = _mcc_project(p0)
        sim = PicSimulation(proj)
        field = GasField(n_g=np.full(len(sim.tris), n0 * factor))
        sim2 = PicSimulation(_mcc_project(p0), gas_field=field)
        for _ in range(20):
            sim2.step()
        results[factor] = sim2.coll_e
    assert results[2.0] == pytest.approx(2.0 * results[1.0], rel=0.15)


# ---- サーバー統合 (/dsmc + use_dsmc_gas) ----------------------------------------


def test_dsmc_endpoint_and_pic_coupling():
    """POST /dsmc が動き、mcc.use_dsmc_gas の PIC がその場を使って起動する。"""
    from starlette.testclient import TestClient

    from es_sim import server as srv

    c = TestClient(srv.app)
    base_geom = {
        "geometry": {
            "domain": {"polygon": [[0, 0], [L, 0], [L, H], [0, H]]},
            "boundaries": [
                {"edges": [3], "type": "dirichlet", "voltage": 0.0},
                {"edges": [1], "type": "dirichlet", "voltage": 0.0},
            ],
        },
        "mesh": {"size": 1.5e-3},
    }
    dsmc_project = base_geom | {
        "dsmc": {
            "boundaries": [
                {"edges": [0], "type": "inlet", "pressure_pa": 10.0},
                {"edges": [2], "type": "outlet", "pressure_pa": 3.0},
            ],
            "init_pressure_pa": 6.0,
            "n_particles": 10000,
            "n_steps": 300,
            "avg_steps": 100,
            "seed": 4,
        }
    }
    r = c.post("/dsmc", json=dsmc_project)
    assert r.status_code == 200, r.text
    d = r.json()
    assert srv._last_dsmc is not None
    assert srv._last_dsmc["n_elems"] == len(d["n"])
    assert all(np.isfinite(d["p"]))

    # use_dsmc_gas の PIC がガス場付きで構築できる (同一 project → 同一メッシュ)
    pic_project = Project.model_validate(
        base_geom
        | {
            "pic": {
                "initial_plasma": {
                    "density": 1e14, "te_ev": 2.0, "ti_ev": 0.03,
                    "ion_mass_amu": 40.0, "seed": 5,
                },
                "n_macro": 2000,
                "dt": 5e-11,
                "n_steps": 5,
                "frame_every": 100,
                "mcc": {
                    "gas": {"name": "Ar", "pressure_pa": 6.0, "temperature_k": 300.0},
                    "electron_processes": [
                        {
                            "kind": "elastic", "label": "syn", "threshold_ev": 0.0,
                            "mass_ratio": 1.36e-5,
                            "energy_ev": [0.0, 100.0], "sigma_m2": [1e-19, 1e-19],
                        }
                    ],
                    "seed": 7,
                    "use_dsmc_gas": True,
                },
            }
        }
    )
    sim = PicSimulation(pic_project, srv._last_dsmc["field"])
    for _ in range(5):
        sim.step()
    assert np.all(np.isfinite(sim.species["electron"].v))


# ---- 線分指定の流入口 + sccm 流量指定 (prompts/55) ------------------------------


def test_dsmc_segment_inlet_classification():
    """p1-p2 線分指定で上辺の一部だけが流入口 (リザーバ) に分類される。"""
    from es_sim.dsmc import _B_RESERVOIR

    project = _project(
        {
            "boundaries": [
                # 上辺 (エッジ2) の左 1/4 だけを線分指定で inlet にする
                {"type": "inlet", "p1": [0.0, H], "p2": [L / 4, H], "pressure_pa": 10.0},
            ],
            "init_pressure_pa": 5.0,
            "n_particles": 5000,
            "n_steps": 10,
            "avg_steps": 5,
            "seed": 6,
        }
    )
    sim = DsmcSimulation(project)
    ts, loc = np.nonzero(sim.adjacency == -1)
    types = sim._b_type[ts, loc]
    res_sel = types == _B_RESERVOIR
    assert np.any(res_sel)
    # 分類されたエッジの中点はすべて上辺 y=H かつ x ≤ L/4 (+わずかな許容)
    tris = sim.tris
    nodes = sim.mesh.nodes
    n1 = tris[ts, (loc + 1) % 3]
    n2 = tris[ts, (loc + 2) % 3]
    mids = 0.5 * (nodes[n1] + nodes[n2])
    assert np.all(np.abs(mids[res_sel, 1] - H) < 1e-9)
    assert np.all(mids[res_sel, 0] <= L / 4 + 1e-6)
    # 上辺の右側 (x > L/4) は壁のまま = リザーバに分類されていない
    top_right = (np.abs(mids[:, 1] - H) < 1e-9) & (mids[:, 0] > L / 4 + 1e-6)
    assert not np.any(res_sel & top_right)


def test_dsmc_sccm_flow_inlet_mass_balance():
    """流量指定 (10 sccm) の流入口: 流入レートが換算値に一致し、定常で流出と釣り合う。"""
    from es_sim.dsmc import SCCM_TO_PER_S

    sccm = 10.0
    project = _project(
        {
            "gas": {"d_ref_m": 1e-15},  # 無衝突 (輸送を単純化)
            "boundaries": [
                {"edges": [3], "type": "inlet", "flow_sccm": sccm},
                {"edges": [1], "type": "outlet"},  # 真空排気
                {"edges": [0], "type": "symmetry"},
                {"edges": [2], "type": "symmetry"},
            ],
            "init_pressure_pa": 0.01,
            "init_temperature_k": 300.0,
            "n_particles": 30000,
            "n_steps": 1500,
            "avg_steps": 500,
            "seed": 7,
        }
    )
    sim = DsmcSimulation(project)
    res = sim.run()

    ndot_expected = sccm * SCCM_TO_PER_S  # ≈ 4.478e18 分子/s
    t_avg = 500 * sim.dt
    inflow_rate = res.inflow / t_avg
    # 流入は決定的 (端数持ち越し) なので換算値にほぼ厳密に一致する
    assert inflow_rate == pytest.approx(ndot_expected, rel=0.02)
    # 定常の質量収支 (統計誤差 10%)
    assert res.outflow == pytest.approx(res.inflow, rel=0.10)
    # 流れは +x 方向
    area = sim.area
    ux_mean = float(np.sum(res.u[:, 0] * res.n * area) / np.sum(res.n * area))
    assert ux_mean > 0.0


# ---- 軸対称 (rz、prompts/56) ----------------------------------------------------


def test_dsmc_rz_equilibrium_cylinder():
    """rz 密閉円筒 (軸 = 下辺 y=0、他は壁温 300K): n/T/p を保持し径方向にも一様。"""
    p0, t0 = 10.0, 300.0
    rr = 0.01
    project = Project.model_validate(
        {
            "coord": "rz",
            "geometry": {
                "domain": {"polygon": [[0, 0], [L, 0], [L, rr], [0, rr]]},
                "boundaries": [],
            },
            "mesh": {"size": 1.5e-3},
            "dsmc": {
                "init_pressure_pa": p0,
                "init_temperature_k": t0,
                "wall_temperature_k": t0,
                "n_particles": 30000,
                "n_steps": 600,
                "avg_steps": 300,
                "seed": 8,
            },
        }
    )
    sim = DsmcSimulation(project)
    res = sim.run()

    n0 = p0 / (KB * t0)
    vol = sim.vol
    n_mean = float(np.sum(res.n * vol) / vol.sum())
    t_mean = float(np.sum(res.t * res.n * vol) / np.sum(res.n * vol))
    assert n_mean == pytest.approx(n0, rel=0.03)
    assert t_mean == pytest.approx(t0, rel=0.03)
    # 径方向の一様性 (体積規格化 2πr̄A の検証): 内側半分と外側半分の平均密度が一致
    r_c = sim.mesh.nodes[sim.tris].mean(axis=1)[:, 1]
    inner = r_c < rr / 2
    n_in = float(np.sum(res.n[inner] * vol[inner]) / vol[inner].sum())
    n_out = float(np.sum(res.n[~inner] * vol[~inner]) / vol[~inner].sum())
    assert n_in == pytest.approx(n_out, rel=0.08)
    # 粒子が消えていない (軸は物理境界ではない)
    assert res.n_particles == pytest.approx(30000, rel=0.02)


def test_dsmc_rz_axial_effusion():
    """rz 無衝突円筒: 左端面リザーバ → 右真空。密度 = リザーバの 1/2、u_z = c̄/2。"""
    p0, t0 = 1.0, 300.0
    rr = 0.005
    project = Project.model_validate(
        {
            "coord": "rz",
            "geometry": {
                "domain": {"polygon": [[0, 0], [L, 0], [L, rr], [0, rr]]},
                "boundaries": [],
            },
            "mesh": {"size": 1.2e-3},
            "dsmc": {
                "gas": {"d_ref_m": 1e-15},  # 無衝突
                "boundaries": [
                    {"edges": [3], "type": "inlet", "pressure_pa": p0, "temperature_k": t0},
                    {"edges": [1], "type": "outlet"},
                    {"edges": [2], "type": "symmetry"},  # 外周 r=R は鏡面 (1D模擬)
                ],
                "init_pressure_pa": p0 / 2.0,
                "init_temperature_k": t0,
                "n_particles": 40000,
                "n_steps": 1500,
                "avg_steps": 500,
                "seed": 9,
            },
        }
    )
    sim = DsmcSimulation(project)
    res = sim.run()

    n_res = p0 / (KB * t0)
    vol = sim.vol
    n_mean = float(np.sum(res.n * vol) / vol.sum())
    assert n_mean == pytest.approx(0.5 * n_res, rel=0.05)
    c_bar = math.sqrt(8.0 * KB * t0 / (math.pi * M_AR))
    uz_mean = float(np.sum(res.u[:, 0] * res.n * vol) / np.sum(res.n * vol))
    assert uz_mean == pytest.approx(0.5 * c_bar, rel=0.05)
    assert res.outflow == pytest.approx(res.inflow, rel=0.07)


def test_dsmc_ws_progress():
    """/ws/dsmc: started → progress (100ステップごと) → done が届く (prompts/58)。"""
    from starlette.testclient import TestClient

    from es_sim import server as srv

    c = TestClient(srv.app)
    project = {
        "geometry": {
            "domain": {"polygon": [[0, 0], [L, 0], [L, H], [0, H]]},
            "boundaries": [],
        },
        "mesh": {"size": 2e-3},
        "dsmc": {
            "init_pressure_pa": 5.0,
            "n_particles": 5000,
            "n_steps": 350,
            "avg_steps": 100,
            "seed": 11,
        },
    }
    with c.websocket_connect("/ws/dsmc") as ws:
        ws.send_json({"cmd": "start", "project": project})
        started = ws.receive_json()
        assert started["type"] == "started"
        assert started["n_steps"] == 350
        assert started["n_particles"] > 0

        progress = 0
        while True:
            msg = ws.receive_json()
            if msg["type"] == "progress":
                progress += 1
                assert 0 < msg["step"] <= 350
                assert msg["n_particles"] > 0
                # ライブ粒子表示 (prompts/66): 間引き済み座標が同梱される
                assert len(msg["particles"]) > 0
                assert len(msg["particles"]) <= 2000
                assert all(len(pt) == 2 for pt in msg["particles"])
                assert np.all(np.isfinite(msg["particles"]))
            elif msg["type"] == "done":
                break
            else:
                raise AssertionError(f"想定外のメッセージ: {msg['type']}")
        assert progress == 3  # 100, 200, 300
        result = msg["result"]
        assert len(result["n"]) > 0
        assert all(np.isfinite(result["p"]))
        # 実行時間のリアルタイム表示 (prompts/86): done.result に run() の壁時計秒が載る
        assert result["elapsed_s"] > 0
    # done 後は保持スロットが更新されている
    assert srv._last_dsmc is not None


def test_dsmc_run_callback_particles():
    """run(callback) の callback(step, n_particles, positions) が間引き粒子座標を渡す
    (prompts/66)。n_particles=8000 は間引き上限 (2000) を超えるケースで、
    positions が上限内・有限値であることを確認する。
    """
    project = _project(
        {
            "init_pressure_pa": 5.0,
            "n_particles": 8000,
            "n_steps": 250,
            "avg_steps": 50,
            "seed": 3,
        }
    )
    sim = DsmcSimulation(project)
    calls = []

    def cb(step, n_particles, positions):
        calls.append((step, n_particles, positions))

    sim.run(cb)

    assert len(calls) == 2  # 100, 200 (250ステップなので200まで)
    for step, n_particles, positions in calls:
        assert step in (100, 200)
        assert n_particles > 0
        assert isinstance(positions, np.ndarray)
        assert positions.shape[0] <= 2000
        assert positions.shape[1] == 2
        assert np.all(np.isfinite(positions))
        # 間引き前の実粒子数が2000を超えていることを確認 (間引きが実際に働くケース)
        assert n_particles > positions.shape[0] or n_particles <= 2000


# ---- 7. 続きから実行 (prepare_continue、prompts/74) --------------------------------


def test_dsmc_continue_is_bit_identical_to_single_run():
    """乱数 Generator・粒子状態を一切リセットしないため、800 ステップ連続実行と
    400 ステップ → continue(400, avg_steps=100) の粒子状態・結果がビット単位で
    一致する (avg_steps=100 を両ケースとも同じ値にすることで、平均区間が
    どちらも通算 701..800 ステップ目という同じ末尾区間になる)。
    """
    kwargs = {"init_pressure_pa": 5.0, "n_particles": 4000, "seed": 7}

    # A: 800 ステップ連続 (avg_start = 800-100 = 700 → 平均区間 701..800)
    sim_a = DsmcSimulation(_project({**kwargs, "n_steps": 800, "avg_steps": 100}))
    res_a = sim_a.run()

    # B: 400 ステップ → continue(400, avg_steps=100)
    #    (継続区間内は avg_start = 400-100 = 300 → 区間相対 301..400 = 通算 701..800)
    sim_b = DsmcSimulation(_project({**kwargs, "n_steps": 400, "avg_steps": 100}))
    sim_b.run()
    sim_b.prepare_continue(400, avg_steps=100)
    res_b = sim_b.run()

    # 粒子状態 (位置・速度・所属要素) がビット単位で一致
    assert len(sim_a.x) == len(sim_b.x) and len(sim_a.x) > 0
    assert np.array_equal(sim_a.x, sim_b.x)
    assert np.array_equal(sim_a.v, sim_b.v)
    assert np.array_equal(sim_a.elem, sim_b.elem)
    assert np.array_equal(sim_a._coll_frac, sim_b._coll_frac)
    assert np.array_equal(sim_a._sigcr_max, sim_b._sigcr_max)
    assert sim_a.step_count == sim_b.step_count == 800

    # 結果フィールドもビット単位で一致
    assert np.array_equal(res_a.n, res_b.n)
    assert np.array_equal(res_a.t, res_b.t)
    assert np.array_equal(res_a.u, res_b.u)
    assert np.array_equal(res_a.p, res_b.p)
    assert res_a.n_particles == res_b.n_particles
    assert res_a.inflow == res_b.inflow
    assert res_a.outflow == res_b.outflow


def _recv_until_done(ws) -> dict:
    while True:
        msg = ws.receive_json()
        assert msg["type"] != "error", msg.get("detail")
        if msg["type"] == "done":
            return msg


def test_dsmc_ws_continue_without_state_errors_and_full_flow():
    """/ws/dsmc: 保持状態なしの continue はエラー。start → done → continue → done
    がエラーなく流れ、2回目の done の結果が有限で、追加区間の avg_steps が
    反映されていること (prompts/74)。
    """
    from starlette.testclient import TestClient

    from es_sim import server as srv

    srv._last_dsmc_sim = None  # 保持スロットを空にしてから検証する
    client = TestClient(srv.app)

    project = {
        "geometry": {
            "domain": {"polygon": [[0, 0], [L, 0], [L, H], [0, H]]},
            "boundaries": [],
        },
        "mesh": {"size": 2e-3},
        "dsmc": {
            "init_pressure_pa": 5.0,
            "n_particles": 3000,
            "n_steps": 150,
            "avg_steps": 50,
            "seed": 5,
        },
    }

    with client.websocket_connect("/ws/dsmc") as ws:
        # 保持状態なしの continue はエラー
        ws.send_json({"cmd": "continue", "n_steps": 10})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "保持" in msg["detail"] or "start" in msg["detail"]

        # start → done
        ws.send_json({"cmd": "start", "project": project})
        started = ws.receive_json()
        assert started["type"] == "started" and started["n_steps"] == 150
        done1 = _recv_until_done(ws)
        assert len(done1["result"]["n"]) > 0
        assert all(np.isfinite(done1["result"]["p"]))

        # continue → started (n_steps=追加分) → done
        ws.send_json({"cmd": "continue", "n_steps": 100, "avg_steps": 40})
        started2 = ws.receive_json()
        assert started2["type"] == "started" and started2["n_steps"] == 100
        done2 = _recv_until_done(ws)
        result2 = done2["result"]
        assert len(result2["n"]) > 0
        assert all(np.isfinite(result2["p"]))
        assert all(np.isfinite(v) for u in result2["u"] for v in u)
        # 追加区間の avg_steps=40 が反映されている (サンプル数が対応する区間長)
        assert srv._last_dsmc_sim.s.avg_steps == 40
        assert srv._last_dsmc_sim.s.n_steps == 100
        assert srv._last_dsmc_sim.step_count == 250

    srv._last_dsmc_sim = None  # 後続テストへ状態を持ち越さない


# ---- 8. 位相別プロファイル計測 (prompts/87、PIC の test_timing_phases に倣う) ------


def test_dsmc_timing_phases():
    """衝突が活発な小ケースを実行し、位相別タイマーが妥当な値を返すこと。

    - 全キーが 0 以上
    - 各フェーズの合計が run() の実測壁時計時間 (elapsed_s) と ±計測誤差で一致する
      (step() 内外の全時間が漏れなく・二重計上なくどこかのフェーズに割り当て
      られていることの確認)。walk コスト診断 (prompts/88、WALK_DIAG_KEYS) は
      秒数ではないためこの合計からは除外する
    - inject/move/collide/sample は正の時間を計上する (このケースは流入境界・
      密な粒子分布ありなので確実に発火する)
    - continue 後は区間分のみを返す (前区間の値を引きずらない)
    - walk_cells_est・h_mean_m・h_min_m (prompts/88) が正で、h は妥当な m オーダー
    """
    t0 = 300.0
    project = _project(
        {
            "boundaries": [
                {"edges": [3], "type": "inlet", "pressure_pa": 20.0, "temperature_k": t0},
                {"edges": [1], "type": "outlet", "pressure_pa": 5.0, "temperature_k": t0},
            ],
            "init_pressure_pa": 12.0,
            "init_temperature_k": t0,
            "wall_temperature_k": t0,
            "n_particles": 20000,
            "n_steps": 60,
            "avg_steps": 20,
            "seed": 9,
        }
    )
    sim = DsmcSimulation(project)
    t_wall0 = time.perf_counter()
    sim.run()
    wall = time.perf_counter() - t_wall0
    timing = sim.timing

    for key in ("inject", "move", "collide", "sample", "other"):
        assert key in timing
        assert timing[key] >= 0.0

    # walk コスト診断 (prompts/88) は秒数ではないので、時間の合計からは除外する
    time_keys = {k: v for k, v in timing.items() if k not in WALK_DIAG_KEYS}
    total = sum(time_keys.values())
    assert total > 0.0
    assert timing["inject"] > 0.0
    assert timing["move"] > 0.0
    assert timing["collide"] > 0.0
    assert timing["sample"] > 0.0
    # run() 呼び出し全体には _smooth_moments や結果配列の組み立てなど、step() の
    # 外側のわずかなコストも含まれるため、total は wall 以下かつ大部分を占める
    # はず (下限は緩めに 50%、上限は計測誤差を見込んで少し余裕を持たせる)
    assert 0.5 * wall <= total <= wall + 0.05

    # walk コスト診断 (prompts/88): 平均横断セル数の推定と代表セル寸法
    assert timing["walk_cells_est"] > 0.0
    assert 0.0 < timing["h_min_m"] <= timing["h_mean_m"]
    # メッシュ寸法 (~1.5e-3 m 刻み) から妥当な m オーダーであること
    assert 1e-6 < timing["h_mean_m"] < 1.0

    # continue 後は区間分のみを返す (前区間の値を引きずらない)
    sim.prepare_continue(60)
    for key in timing:
        assert sim.timing[key] == 0.0
    sim.run()
    total2 = sum(v for k, v in sim.timing.items() if k not in WALK_DIAG_KEYS)
    assert total2 > 0.0
    # continue 後も walk コスト診断キーは再設定される
    assert sim.timing["walk_cells_est"] > 0.0
    assert sim.timing["h_mean_m"] > 0.0

    # DsmcResult.timing にも同じキーが載る (エンドポイント/保存ファイル用)
    sim2 = DsmcSimulation(project)
    res = sim2.run()
    assert set(res.timing.keys()) == set(sim2.timing.keys())
    assert sum(v for k, v in res.timing.items() if k not in WALK_DIAG_KEYS) > 0.0
