"""粒子マージ (高速化③、prompts/77) のテスト。

1. 保存性: 人工的な1セル多数粒子集合に _merge_species を直接適用し、
   Σw / Σw·v (3成分) / Σw|v|² が相対誤差 1e-12 以下で保存され、粒子数が減ること。
2. 上限制御: 電離が活発な設定で n_max を小さくした PIC 実行を行い、マージ ON では
   粒子数が n_max 近傍に維持され、OFF (既定) では大きく増え続けること。
3. OFF時の不変性: merge=None ではマージ用 rng を生成せず (乱数を一切引かない)、
   diag の "merged" 累計も常に 0 のままであること。
"""

import numpy as np

from es_sim.pic import PicSimulation
from es_sim.schema import Project

DENSITY = 1.0e14  # [m^-3]


def _oscillation_project(n_steps: int) -> Project:
    """test_pic.py と同じ最小構成 (プラズマ振動テストの流用)。"""
    L, H = 0.01, 0.03
    return Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [L, 0], [L, H], [0, H]]},
                "boundaries": [{"edges": [0, 1, 2, 3], "type": "dirichlet", "voltage": 0.0}],
            },
            "mesh": {"size": 8e-4},
            "pic": {
                "initial_plasma": {
                    "density": DENSITY,
                    "te_ev": 0.0,
                    "ti_ev": 0.0,
                    "ion_mass_amu": 40.0,
                    "immobile_ions": True,
                    "seed": 1,
                },
                "n_macro": 500,
                "dt": None,
                "n_steps": n_steps,
                "frame_every": n_steps,
            },
        }
    )


# ---- 1. 保存性 -----------------------------------------------------------------


def test_merge_conservation():
    """1セルに人工的に集めた500粒子をマージし、W・P(3成分)・E が厳密に保存されること。"""
    sim = PicSimulation(_oscillation_project(n_steps=1))
    sim._merge_rng = np.random.default_rng(0)  # merge=None のプロジェクトなので手動で用意
    sp = sim.species["electron"]

    rng = np.random.default_rng(123)
    n0 = 500
    elem0 = 0
    center = sim.mesh.nodes[sim.tris[elem0]].mean(axis=0)
    x = np.broadcast_to(center, (n0, 2)).copy() + rng.normal(scale=1e-7, size=(n0, 2))
    v = rng.normal(scale=1.0e5, size=(n0, 3))
    w = rng.uniform(1.0e6, 2.0e6, size=n0)

    sp.x, sp.v, sp.w = x, v, w
    sp.elem = np.full(n0, elem0, dtype=np.int64)
    sp.bary = None
    sp.nidx = None

    w0 = w.sum()
    p0 = (w[:, None] * v).sum(axis=0)
    e0 = float((w * np.sum(v * v, axis=1)).sum())

    removed = sim._merge_species(sp, n_max=1)  # n_max=1: 可能な限りマージさせる

    assert removed > 0
    assert len(sp.x) < n0
    assert len(sp.x) == len(sp.v) == len(sp.w) == len(sp.elem)

    w1 = sp.w.sum()
    p1 = (sp.w[:, None] * sp.v).sum(axis=0)
    e1 = float((sp.w * np.sum(sp.v * sp.v, axis=1)).sum())

    assert abs(w1 - w0) / w0 < 1e-12
    assert np.all(np.abs(p1 - p0) / np.abs(p0) < 1e-12)
    assert abs(e1 - e0) / e0 < 1e-12
    # マージ後の位置はすべて元のセル内 (重み付き平均は凸結合)
    assert np.all(sp.elem == elem0)


# ---- 2. 上限制御 ----------------------------------------------------------------


def _ionizing_project(n_steps: int, merge: dict | None) -> Project:
    """電離が活発な合成ケース (test_pic.py test_timing_phases のイオン化条件を流用)。

    全外周を reflect_edges (鏡面反射) にして壁吸収による粒子損失を無くし、
    電離によるマクロ粒子数の純増が持続するようにする (吸収ありだと壁損失が
    電離増加を上回り、マージの有無による差が観測しにくくなるため)。
    また、セルあたりの粒子数が十分でないとオクタント分割後の各グループが
    MERGE_MIN_GROUP (3) 未満になりマージが働かないため、粒子数の規模に対して
    mesh.size をやや粗くしてある。
    """
    return Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [0.02, 0], [0.02, 0.01], [0, 0.01]]},
                "boundaries": [
                    {"edges": [3], "voltage": 0.0},
                    {"edges": [1], "voltage": 0.0},
                ],
            },
            "mesh": {"size": 3.0e-3},
            "pic": {
                "initial_plasma": {
                    "density": DENSITY,
                    "te_ev": 8.0,
                    "ti_ev": 0.03,
                    "ion_mass_amu": 40.0,
                    "seed": 3,
                },
                "n_macro": 1500,
                "dt": 5e-10,
                "n_steps": n_steps,
                "frame_every": n_steps,
                "reflect_edges": [0, 1, 2, 3],
                "mcc": {
                    "gas": {"name": "Ar", "pressure_pa": 300.0, "temperature_k": 300.0},
                    "electron_processes": [
                        {
                            "kind": "ionization",
                            "label": "synthetic ionization",
                            "threshold_ev": 5.0,
                            "energy_ev": [5.0, 5.0 + 1e-6, 1000.0],
                            "sigma_m2": [0.0, 5.0e-20, 5.0e-20],
                        }
                    ],
                    "ion_processes": [],
                    "seed": 11,
                },
                "merge": merge,
            },
        }
    )


def test_merge_caps_particle_count():
    """マージ ON では粒子数が n_max 近傍に維持され、OFF では大きく増え続けること。"""
    n_steps = 300
    n_max = 1500
    merge = {"n_max": n_max, "every": 20}

    sim_on = PicSimulation(_ionizing_project(n_steps, merge))
    hist_on, _ = sim_on.run_batch()
    n_e_on = np.asarray(hist_on["n_e"])

    sim_off = PicSimulation(_ionizing_project(n_steps, merge=None))
    hist_off, _ = sim_off.run_batch()
    n_e_off = np.asarray(hist_off["n_e"])

    assert sim_on.merge_removed > 0, "電離が活発なはずなのにマージが一度も発生していません"
    # 上限近傍 (90%目標 ± マージンを見込む) に維持される
    assert n_e_on[-100:].max() <= n_max * 1.3
    # OFF は同じ電離条件で上限の2倍を超えて増え続ける (マージの効果の比較)
    assert n_e_off[-1] > n_max * 2.0
    assert n_e_off[-1] > n_e_on[-1]


# ---- 3. OFF 時の不変性 -----------------------------------------------------------


def test_merge_off_no_rng():
    """merge=None ならマージ用 rng を生成せず、diag の merged 累計は常に 0。"""
    sim = PicSimulation(_oscillation_project(n_steps=10))
    assert sim.pic.merge is None
    assert not hasattr(sim, "_merge_rng")
    history, _ = sim.run_batch()
    assert all(v == 0 for v in history["merged"])
    assert sim.merge_events == 0
    assert sim.merge_removed == 0
