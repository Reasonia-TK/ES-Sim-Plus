"""指定矩形範囲の EEDF/EEPF 取得のテスト (prompts/85)。

1. Maxwellian 検証: 温度 T の初期プラズマ (無衝突・イオン固定・全反射壁で閉じた系、
   quiet start により初期電場はほぼゼロ) で domain 全体を覆う EEDF 領域を置き、
   得られた EEDF が理論 f(E) = 2√(E/π) T^{-3/2} exp(-E/T) と (理論確率質量で重み付けた
   平均相対誤差で) 数%〜1割程度で一致し、t_eff_ev ≈ T (5%以内) であること。
2. 矩形選択: 電子を左右半分に別温度で人工配置し、左半分のみを覆う領域の t_eff が
   左側温度に一致 (右側温度に引っ張られない) こと。
3. 正規化: Σ f_i・ΔE = 1 (数値積分で 1e-6 以内)。
4. 電子が一度も入らない領域は安全に全ゼロ (total_weight=0, f全ゼロ) で返ること。
"""

import numpy as np
import pytest

from es_sim.particles import ME, QE, _locate_initial
from es_sim.pic import PicSimulation
from es_sim.schema import Project

L, H = 0.01, 0.01
DENSITY = 1.0e14  # [m^-3]


def _closed_box_project(te_ev: float, eedf_regions: list[dict], n_macro: int = 20000) -> Project:
    """全反射壁 (reflect_edges) で閉じた無衝突系。quiet start (電子・イオン同一位置装荷)
    によって初期電場はほぼゼロなので、短時間平均なら初期 Maxwellian 分布がほぼ保たれる。
    イオンは immobile にして電子のみの熱運動を見る。
    """
    return Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [L, 0], [L, H], [0, H]]},
                "boundaries": [{"edges": [0, 1, 2, 3], "voltage": 0.0}],
            },
            "mesh": {"size": 2e-3},
            "pic": {
                "initial_plasma": {
                    "density": DENSITY,
                    "te_ev": te_ev,
                    "ti_ev": te_ev,
                    "ion_mass_amu": 40.0,
                    "immobile_ions": True,
                    "seed": 1,
                },
                "n_macro": n_macro,
                "dt": 1e-11,
                "n_steps": 20,
                "frame_every": 20,
                "avg_steps": 20,
                "reflect_edges": [0, 1, 2, 3],
                "eedf_regions": eedf_regions,
            },
        }
    )


def _maxwellian_eedf(e: np.ndarray, t_ev: float) -> np.ndarray:
    return 2.0 * np.sqrt(e / np.pi) * t_ev**-1.5 * np.exp(-e / t_ev)


# ---- 1. Maxwellian 検証 ------------------------------------------------------------


def test_eedf_matches_maxwellian_theory():
    t_ev = 5.0
    project = _closed_box_project(
        t_ev, [{"p1": [0.0, 0.0], "p2": [L, H], "label": "full", "bins": 20}]
    )
    sim = PicSimulation(project)
    sim.run_batch()

    results = sim.eedf_results
    assert results is not None and len(results) == 1
    r = results[0]
    assert r["label"] == "full"
    assert r["total_weight"] > 0.0
    assert r["overflow_frac"] < 0.05
    assert r["n_samples"] == 20

    # t_eff ≈ T (5%以内)
    assert r["t_eff_ev"] == pytest.approx(t_ev, rel=0.05)

    f = np.asarray(r["f"])
    e_centers = np.asarray(r["e_centers"])
    theory = _maxwellian_eedf(e_centers, t_ev)

    # 理論確率質量で重み付けた平均相対誤差 (ヒストグラムのテール側の統計ノイズに
    # 引きずられないよう、質量の大きいところを重視する)
    weighted_rel_err = float(np.sum(theory * np.abs(f - theory)) / np.sum(theory * theory))
    assert weighted_rel_err < 0.10



# ---- 2. 矩形選択: 左右で別温度 --------------------------------------------------------


def test_eedf_region_selects_only_local_electrons():
    t_left, t_right = 2.0, 8.0
    n_half = 10000
    rng = np.random.default_rng(7)

    project = _closed_box_project(
        t_left,  # ダミー (initial_plasma は使わず手動で電子を配置するので値は無関係)
        [
            {"p1": [0.0, 0.0], "p2": [L / 2, H], "label": "left", "bins": 30},
            {"p1": [L / 2, 0.0], "p2": [L, H], "label": "right", "bins": 30},
        ],
        n_macro=1,
    )
    sim = PicSimulation(project)

    # 左右半分に別温度の電子を人工配置する (y は壁際を避けて一様分布)
    xl = rng.uniform(0.001, L / 2 - 0.0005, n_half)
    xr = rng.uniform(L / 2 + 0.0005, L - 0.001, n_half)
    x = np.concatenate([xl, xr])
    y = rng.uniform(0.001, H - 0.001, 2 * n_half)
    pos = np.stack([x, y], axis=1)

    sigma_l = np.sqrt(t_left * QE / ME)
    sigma_r = np.sqrt(t_right * QE / ME)
    v = np.concatenate(
        [rng.normal(0.0, sigma_l, size=(n_half, 3)), rng.normal(0.0, sigma_r, size=(n_half, 3))]
    )

    el = sim.species["electron"]
    el.x = pos
    el.v = v
    el.w = np.full(2 * n_half, 1.0)
    el.elem = _locate_initial(sim.coeffs, pos)
    el.bary = None
    el.nidx = None

    sim.run_batch()
    results = sim.eedf_results
    assert results is not None and len(results) == 2
    left, right = results

    assert left["label"] == "left"
    assert right["label"] == "right"
    # 左領域の t_eff は左側温度に一致し、右側 (高温) に引っ張られない
    assert left["t_eff_ev"] == pytest.approx(t_left, rel=0.15)
    assert right["t_eff_ev"] == pytest.approx(t_right, rel=0.15)
    # 総重み (電子1個の重み=1) は「毎ステップ」加算されるため、n_half × 平均ステップ数
    # (avg_steps=20) に近い値になる (電子はほぼ動かないので毎回同じ集合が数えられる)
    n_samples = left["n_samples"]
    assert left["total_weight"] == pytest.approx(float(n_half * n_samples), rel=0.05)
    assert right["total_weight"] == pytest.approx(float(n_half * n_samples), rel=0.05)


# ---- 3. 正規化: ∫f dE = 1 -----------------------------------------------------------


def test_eedf_normalization_integral_is_one():
    project = _closed_box_project(
        4.0, [{"p1": [0.0, 0.0], "p2": [L, H], "label": "full", "bins": 50, "e_max_ev": 60.0}]
    )
    sim = PicSimulation(project)
    sim.run_batch()

    r = sim.eedf_results[0]
    assert r["overflow_frac"] == 0.0  # e_max_ev に十分な余裕があるので範囲外は出ない
    f = np.asarray(r["f"])
    e_centers = np.asarray(r["e_centers"])
    d_e = e_centers[1] - e_centers[0]
    integral = float(np.sum(f) * d_e)
    assert abs(integral - 1.0) < 1e-6


# ---- 4. 電子が入らない領域は安全に全ゼロ ------------------------------------------------


def test_eedf_region_without_electrons_is_all_zero():
    # 領域を domain の外に置き、電子が絶対に入らないようにする
    project = _closed_box_project(
        3.0,
        [{"p1": [10 * L, 10 * H], "p2": [11 * L, 11 * H], "label": "empty", "bins": 15}],
    )
    sim = PicSimulation(project)
    sim.run_batch()

    r = sim.eedf_results[0]
    assert r["label"] == "empty"
    assert r["total_weight"] == 0.0
    assert r["overflow_frac"] == 0.0
    assert r["mean_energy_ev"] == 0.0
    assert r["t_eff_ev"] == 0.0
    assert r["n_samples"] == 20
    assert np.all(np.asarray(r["f"]) == 0.0)
    assert len(r["e_centers"]) == 15
