"""v1 PIC (三角形メッシュ) の半径に比例した重み (軸対称、prompts/136 RW-c)。

1. 装荷: 重みは c (r + r0)、全電荷は density·V、時間平均の密度が軸の上の節点まで一様
2. 分割・併合: 分割は同じ位置・速度、併合は同じ要素の重み付き平均の位置と一方の速度で、P1 の電荷堆積が変わらない
3. 損失の無い箱の熱的なプラズマ: 種ごとの重みの和が丸め誤差で一定、分割・併合が釣り合い、w/w_t が [1/2, 2] 付近
4. 粒子マージ (merge) があるときは使わない (警告)
5. 軸の近くの電位の揺らぎ (統計の雑音) が一様の重みより十分小さく、軸から離れた所は変わらない
"""

from __future__ import annotations

import numpy as np
import pytest

from es_sim.particles import _locate_initial
from es_sim.pic import PicSimulation
from es_sim.schema import Project


def _elastic_mcc(seed: int) -> dict:
    el = [{"kind": "elastic", "label": "e elastic", "threshold_ev": 0.0, "mass_ratio": 1.36e-5,
           "energy_ev": [0.0, 1000.0], "sigma_m2": [1e-19, 1e-19]}]
    return {"gas": {"name": "Ar", "pressure_pa": 13.33, "temperature_k": 300.0},
            "electron_processes": el, "ion_processes": [], "seed": seed}


def _box(z_len=0.02, r_len=0.01, density=1e15, te=2.0, n_macro=40_000, n_steps=2, radial=True, mcc=None,
         immobile=False, seed=3, mesh=5e-4, **pic) -> Project:
    """軸対称 (下の辺が軸) の箱。外周は反射 (粒子は失われない)、電位は 0 V。"""
    p = {
        "initial_plasma": {"density": density, "te_ev": te, "ti_ev": 0.03, "ion_mass_amu": 40.0, "seed": seed,
                           "immobile_ions": immobile},
        "n_macro": n_macro, "dt": 2e-11, "n_steps": n_steps, "frame_every": 10**9, "avg_steps": n_steps,
        "phase_bins": 0, "reflect_edges": [1, 2, 3], "radial_weighting": radial,
    }
    if mcc is not None:
        p["mcc"] = mcc
    p.update(pic)
    return Project.model_validate({
        "coord": "rz",
        "geometry": {"domain": {"polygon": [[0, 0], [z_len, 0], [z_len, r_len], [0, r_len]]},
                     "boundaries": [{"edges": [1, 2, 3], "type": "dirichlet", "voltage": 0.0}]},
        "mesh": {"size": mesh},
        "pic": p,
    })


def _target(sim, x: np.ndarray) -> np.ndarray:
    return sim._rw_c * (x[:, sim.ridx] + sim._rw_r0)


# ---- 1. 装荷 ------------------------------------------------------------------------------


def test_loading_follows_the_target_weight_and_density_is_uniform_to_the_axis():
    sim = PicSimulation(_box(te=0.0, immobile=True, n_macro=200_000))
    e, i = sim.species["electron"], sim.species["ion"]
    assert sim._rw and sim._rw_r0 == pytest.approx(5e-4)
    np.testing.assert_allclose(e.w, _target(sim, e.x), rtol=1e-12)
    np.testing.assert_array_equal(e.w, i.w)           # quiet start (同じ位置・重み)
    np.testing.assert_array_equal(e.x, i.x)
    assert e.w.sum() == pytest.approx(1e15 * float(sim.elem_vol.sum()), rel=1e-12)
    # 0.5 mm 幅の帯の粒子数は r/(r + r0) に比例 (一様の重みなら r に比例)
    counts, edges = np.histogram(e.x[:, 1], bins=np.arange(0.0, 0.0100001, 5e-4))
    mid = 0.5 * (edges[1:] + edges[:-1])
    ratio = counts / (mid / (mid + sim._rw_r0))
    assert np.max(np.abs(ratio[1:] / ratio[1:].mean() - 1.0)) < 0.05
    sim.run_batch(store_frames=False)
    nodes = sim.mesh.nodes
    ne = np.asarray(sim.fields["n_e"])
    inner = (nodes[:, 0] > 2e-3) & (nodes[:, 0] < 0.018)
    assert ne[inner & (nodes[:, 1] < 1e-9)].mean() == pytest.approx(1e15, rel=0.05)
    assert ne[inner & (nodes[:, 1] > 1e-3) & (nodes[:, 1] < 4e-3)].mean() == pytest.approx(1e15, rel=0.03)


# ---- 2. 分割・併合 --------------------------------------------------------------------


def _set(sim, name: str, x, v, w) -> None:
    sp = sim.species[name]
    sp.x = np.asarray(x, dtype=np.float64)
    sp.v = np.asarray(v, dtype=np.float64)
    sp.w = np.asarray(w, dtype=np.float64)
    sp.elem = _locate_initial(sim.coeffs, sp.x)
    sp.bary = None
    sp.nidx = None


def test_split_and_merge_keep_the_charge_deposit():
    sim = PicSimulation(_box(density=1e3, te=0.0, n_macro=1000, immobile=True))
    sp = sim.species["electron"]
    # 重い 1 個と、同じ要素に軽い 2 個 (同じ位置の近く)、別の場所に軽い 1 個
    x = np.array([[0.010, 0.0020], [0.0153, 0.0041], [0.0153, 0.0042], [0.0050, 0.0080]])
    _set(sim, "electron", x, [[1e3, 2e3, 3e3], [1e3, 0, 0], [-2e3, 0, 0], [5e2, 0, 0]], [1.0, 1.0, 1.0, 1.0])
    assert sp.elem[1] == sp.elem[2] and sp.elem[3] != sp.elem[1]
    wt = _target(sim, sp.x)
    sp.w = wt * np.array([5.3, 0.2, 0.3, 0.2])
    f0 = sim._deposit_species(sp)
    w_tot = float(sp.w.sum())
    assert sim._pop_control_species(sp)
    assert sim._deposit_species(sp) == pytest.approx(f0, rel=1e-12, abs=1e-12 * float(np.abs(f0).max()))
    assert float(sp.w.sum()) == pytest.approx(w_tot, rel=1e-15)
    assert len(sp.w) == 5 + 1 + 1                       # 5 個に分割、2 個が 1 個に、1 個はそのまま
    assert sim.split_added == 4 and sim.merge_removed == 1
    heavy = np.nonzero(np.abs(sp.x[:, 0] - 0.010) < 1e-12)[0]
    assert len(heavy) == 5
    np.testing.assert_allclose(sp.w[heavy], wt[0] * 5.3 / 5, rtol=1e-12)
    assert np.ptp(sp.v[heavy], axis=0).max() == 0.0
    pair = np.nonzero(np.abs(sp.x[:, 0] - 0.0153) < 1e-9)[0]
    assert len(pair) == 1
    k = pair[0]
    assert sp.w[k] == pytest.approx(wt[1] * 0.2 + wt[2] * 0.3, rel=1e-15)
    r_mean = (wt[1] * 0.2 * 0.0041 + wt[2] * 0.3 * 0.0042) / (wt[1] * 0.2 + wt[2] * 0.3)
    assert sp.x[k, 1] == pytest.approx(r_mean, rel=1e-12)
    assert sp.v[k, 0] in (1e3, -2e3)


# ---- 3. 熱的なプラズマ (損失なし) ---------------------------------------------------------


def test_population_control_conserves_weight_and_stays_bounded():
    sim = PicSimulation(_box(n_steps=300, mcc=_elastic_mcc(7)))
    w0 = {name: float(sp.w.sum()) for name, sp in sim.species.items()}
    n0 = len(sim.species["electron"].w)
    sim.run_batch(store_frames=False)
    h = sim.history
    assert h["split"][-1] > 500 and h["merged"][-1] > 500
    for name, sp in sim.species.items():
        assert abs(float(sp.w.sum()) - w0[name]) < 1e-12 * w0[name], name
    e = sim.species["electron"]
    assert 0.9 * n0 < len(e.w) < 1.15 * n0
    assert h["n_e"][-1] == len(e.w)
    lo, hi = np.percentile(e.w / _target(sim, e.x), [1, 99])
    assert lo > 0.35 and hi < 2.05
    assert h["ke_e"][-1] == pytest.approx(h["ke_e"][0], rel=0.03)


# ---- 4. 粒子マージとの関係 ------------------------------------------------------------------


def test_particle_merge_turns_radial_weighting_off():
    sim = PicSimulation(_box(merge={"n_max": 100_000, "every": 10}))
    assert not sim._rw
    assert any("radial_weighting" in w for w in sim.warnings)
    assert np.all(sim.species["electron"].w == sim.species["electron"].w[0])


# ---- 5. 軸の近くの雑音 -------------------------------------------------------------------


def _phi_noise(radial: bool) -> tuple[float, float]:
    """φ の節点ごとの時間の標準偏差: 軸の上と r 10〜20 mm の二乗平均 (z の端を除く)。"""
    sim = PicSimulation(_box(z_len=0.01, r_len=0.03, n_macro=24_000, n_steps=600, radial=radial, immobile=True,
                             mcc=_elastic_mcc(8), seed=1))
    nodes = sim.mesh.nodes
    s1 = np.zeros(len(nodes))
    s2 = np.zeros(len(nodes))
    cnt = 0
    for k in range(600):
        phi = sim.step()
        if k >= 150:
            s1 += phi
            s2 += phi * phi
            cnt += 1
    sd = np.sqrt(np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0.0))
    inner = (nodes[:, 0] > 1e-3) & (nodes[:, 0] < 9e-3)
    axis = float(np.sqrt(np.mean(sd[inner & (nodes[:, 1] < 1e-9)] ** 2)))
    mid = float(np.sqrt(np.mean(sd[inner & (nodes[:, 1] > 0.01) & (nodes[:, 1] < 0.02)] ** 2)))
    return axis, mid


def test_near_axis_potential_noise_is_reduced():
    axis_u, mid_u = _phi_noise(radial=False)
    axis_r, mid_r = _phi_noise(radial=True)
    assert axis_u > 3.0 * mid_u
    assert axis_r < 0.6 * axis_u
    assert mid_r == pytest.approx(mid_u, rel=0.3)
