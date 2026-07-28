"""イオンサブサイクリングと粒子チャンク並列のテスト (prompts/50)。

1. 一定電界での厳密性: 一定 E の下でリープフロッグは厳密なので、
   sub=1 と sub=5 のイオン最終位置が一致する
2. threads>1 のビット一致: walk のチャンク並列は各粒子の演算を変えないため、
   threads=1 と threads=4 の結果 (粒子状態・診断履歴) がビット単位で一致する
3. CCP でのサブサイクル近似: sub=4 と sub=1 の結果が物理的に近い (発散しない)
4. イオン堆積キャッシュの整合: キャッシュ利用時と再計算が一致する
"""

import numpy as np

from es_sim.pic import PicSimulation, _effective_thread_count
from es_sim.schema import Project

L = 0.02
H = 0.01


def _capacitor_project(pic_extra: dict) -> Project:
    """左 0V・右 1000V の平行平板 (初期プラズマなし)。"""
    return Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [L, 0], [L, H], [0, H]]},
                "boundaries": [
                    {"edges": [3], "type": "dirichlet", "voltage": 0.0},
                    {"edges": [1], "type": "dirichlet", "voltage": 1000.0},
                ],
            },
            "mesh": {"size": 1.5e-3},
            "pic": {"n_macro": 100, "dt": 1e-10, "n_steps": 10, "frame_every": 100} | pic_extra,
        }
    )


def _place_single_ion(sim: PicSimulation) -> None:
    """中央に静止イオンを1個置く (自己場は電極電位に対して無視できる大きさ)。"""
    from es_sim.particles import _locate_initial

    io = sim.species["ion"]
    io.x = np.array([[0.005, 0.005]])
    io.v = np.zeros((1, 3))
    io.w = np.array([1.0])
    io.elem = _locate_initial(sim.coeffs, io.x)
    io.bary = None
    io.nidx = None
    sim._f_ion_cache = None


def test_subcycle_constant_field_exact():
    """一定電界ではリープフロッグが厳密 → sub=1 と sub=5 の位置が一致する。"""
    positions = {}
    for sub in (1, 5):
        sim = PicSimulation(_capacitor_project({"ion_subcycle": sub}))
        _place_single_ion(sim)
        # 初期半ステップ後退キックを手動で適用 (粒子を後から入れたため)
        phi = sim._solve_phi(sim._deposit(), 0.0)
        ex, ey = sim._e_field(phi)
        io = sim.species["ion"]
        dt_i = sim.dt * sub
        e_at = np.stack([ex[io.elem], ey[io.elem]], axis=1)
        io.v[:, :2] -= 0.5 * dt_i * (io.q / io.m) * e_at
        for _ in range(10):
            sim.step()
        assert len(io.x) == 1
        positions[sub] = io.x[0].copy()
    assert np.allclose(positions[1], positions[5], rtol=1e-9, atol=0.0)
    # イオンは電界 (右が高電位、正イオンは -x 方向の力) で左へ動いている
    assert positions[1][0] < 0.005


def _ccp_project(pic_extra: dict, n_steps: int = 25) -> Project:
    return Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [L, 0], [L, H], [0, H]]},
                "boundaries": [
                    {
                        "edges": [3], "type": "dirichlet", "voltage": 0.0,
                        "voltage_rf": {"amplitude": 100.0, "freq_hz": 13.56e6},
                    },
                    {"edges": [1], "type": "dirichlet", "voltage": 0.0},
                ],
            },
            "mesh": {"size": 1.2e-3},
            "pic": {
                "initial_plasma": {
                    "density": 1e14, "te_ev": 2.0, "ti_ev": 0.03,
                    "ion_mass_amu": 40.0, "seed": 3,
                },
                "n_macro": 6000,
                "dt": 5e-11,
                "n_steps": n_steps,
                "frame_every": 100,
            } | pic_extra,
        }
    )


def test_threads_bit_identical():
    """threads=4 のチャンク並列 walk は threads=1 とビット単位で一致する。"""
    sims = {}
    for th in (1, 4):
        sim = PicSimulation(_ccp_project({"threads": th}))
        for _ in range(25):
            sim.step()
        sims[th] = sim
    a, b = sims[1], sims[4]
    for name in ("electron", "ion"):
        sa, sb = a.species[name], b.species[name]
        assert np.array_equal(sa.x, sb.x), f"{name}.x が不一致"
        assert np.array_equal(sa.v, sb.v), f"{name}.v が不一致"
        assert np.array_equal(sa.w, sb.w)
        assert np.array_equal(sa.elem, sb.elem)
    for key in a.history:
        assert a.history[key] == b.history[key], f"history[{key}] が不一致"


def test_auto_threads_uses_particle_threshold_and_cpu_cap():
    """threads=0 は小規模で逐次、大規模でCPU数と上限2を考慮して選ぶ。"""
    assert _effective_thread_count(0, 20_000, cpu_count=16) == 1
    assert _effective_thread_count(0, 100_000, cpu_count=2) == 2
    # メニーコア機では論理コア数の半分 (上限16) まで自動で使う (prompts/86 相当の修正:
    # 旧上限2は小コア環境の実測によるもので、24コア級では機械を遊ばせていた)
    assert _effective_thread_count(0, 100_000, cpu_count=16) == 8
    assert _effective_thread_count(0, 200_000, cpu_count=48) == 16
    assert _effective_thread_count(0, 100_000, cpu_count=4) == 2
    # 明示指定は自動選択の閾値・上限に関係なく尊重する
    assert _effective_thread_count(8, 20_000, cpu_count=16) == 8


def test_auto_threads_is_exposed_on_simulation():
    sim = PicSimulation(_ccp_project({"threads": 0}))
    assert sim.effective_threads == 1


def test_subcycle_warns_when_estimated_ion_step_crosses_half_cell():
    sim = PicSimulation(_ccp_project({"ion_subcycle": 1000}))
    assert any("推定移動量" in w and "ion_subcycle=1000" in w for w in sim.warnings)


def test_subcycle_collision_warning_uses_effective_ion_dt():
    sim = PicSimulation(
        _ccp_project(
            {
                "ion_subcycle": 20,
                "mcc": {
                    "gas": {
                        "name": "synthetic",
                        "pressure_pa": 100.0,
                        "temperature_k": 300.0,
                    },
                    "electron_processes": [],
                    "ion_processes": [
                        {
                            "kind": "backscat",
                            "label": "synthetic charge exchange",
                            "energy_ev": [0.0, 1000.0],
                            "sigma_m2": [1.0e-18, 1.0e-18],
                        }
                    ],
                },
            }
        )
    )
    assert any(
        "イオンの1回の更新衝突候補率" in w and "ion_subcycle=20" in w
        for w in sim.warnings
    )


def test_subcycle_ccp_sanity():
    """sub=4 の CCP が sub=1 と物理的に近い結果を保つ (粗い近似一致)。"""
    finals = {}
    for sub in (1, 4):
        sim = PicSimulation(_ccp_project({"ion_subcycle": sub}, n_steps=200))
        history, _ = sim.run_batch()
        for key in ("ke_e", "ke_i", "fe"):
            assert np.all(np.isfinite(np.asarray(history[key])))
        finals[sub] = {
            "n_i": len(sim.species["ion"].x),
            "phi_max": float(np.mean(history["phi_max"][-50:])),
        }
    # イオンダイナミクスの離散化の違いのみなので粒子数・電位は近いはず
    assert abs(finals[4]["n_i"] - finals[1]["n_i"]) / finals[1]["n_i"] < 0.05
    assert abs(finals[4]["phi_max"] - finals[1]["phi_max"]) < 0.3 * abs(finals[1]["phi_max"]) + 5.0


def test_subcycle_deposit_cache_consistency():
    """休止ステップ・プッシュステップを跨いだ後もイオン堆積キャッシュが正確。"""
    sim = PicSimulation(_ccp_project({"ion_subcycle": 3}))
    for _ in range(5):  # プッシュ (0, 3) と休止 (1, 2, 4) が混在する
        sim.step()
    f_cached = sim._deposit()
    assert sim._f_ion_cache is not None  # キャッシュ経路を通ったことを確認
    sim._f_ion_cache = None
    f_fresh = sim._deposit()
    assert np.array_equal(f_cached, f_fresh)
