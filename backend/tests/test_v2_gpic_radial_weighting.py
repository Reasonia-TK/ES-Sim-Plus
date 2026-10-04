"""v2 GPU PIC の半径に比例した重み (軸対称、prompts/136)。

1. 装荷: 重みは c (r + r0)、全電荷は density·V_gas、セルあたりの粒子数が r によらずほぼ一定、時間平均の密度が軸の
   上の節点まで一様 (軸の上の節点の体積を形状関数の体積にした表示も)
2. 分割: 重すぎる粒子は同じ位置・速度の粒子に等分され、重みの和は変わらない
3. 併合: 同じセルの軽い粒子 2 個が 1 個になり (重みは和、性質は一方のもの)、相手のいない粒子はそのまま
4. 損失の無い箱の熱的なプラズマ: 種ごとの重みの和が丸め誤差で一定、分割・併合が釣り合い、w/w_t が [1/2, 2] 付近
5. 軸の近くの電位の揺らぎ (統計の雑音) が一様の重みより十分小さく、軸から離れた所は変わらない

GPU (CUDA) が無い環境では skip する。
"""

from __future__ import annotations

import numpy as np
import pytest

from es_sim.device import cuda_available
from es_sim.schema import Project

pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA (CuPy) が使えない環境")


def _sim(project: Project):
    from es_sim.gpic import GpuPicSimulation

    return GpuPicSimulation(project)


def _elastic_mcc(seed: int) -> dict:
    el = [{"kind": "elastic", "label": "e elastic", "threshold_ev": 0.0, "mass_ratio": 1.36e-5,
           "energy_ev": [0.0, 1000.0], "sigma_m2": [1e-19, 1e-19]}]
    return {"gas": {"name": "Ar", "pressure_pa": 13.33, "temperature_k": 300.0},
            "electron_processes": el, "ion_processes": [], "seed": seed}


def _box(z_len=0.02, r_len=0.01, density=1e15, te=2.0, n_macro=200_000, n_steps=2, radial=True, mcc=None,
         immobile=False, seed=3) -> Project:
    """軸対称 (下の辺が軸) の箱。外周は反射 (粒子は失われない)、電位は 0 V。"""
    pic = {
        "initial_plasma": {"density": density, "te_ev": te, "ti_ev": 0.03, "ion_mass_amu": 40.0, "seed": seed,
                           "immobile_ions": immobile},
        "n_macro": n_macro, "dt": 2e-11, "n_steps": n_steps, "frame_every": 10**9, "avg_steps": n_steps,
        "phase_bins": 0, "reflect_edges": [1, 2, 3], "radial_weighting": radial,
    }
    if mcc is not None:
        pic["mcc"] = mcc
    return Project.model_validate({
        "coord": "rz",
        "geometry": {"domain": {"polygon": [[0, 0], [z_len, 0], [z_len, r_len], [0, r_len]]},
                     "boundaries": [{"edges": [1, 2, 3], "type": "dirichlet", "voltage": 0.0}]},
        "mesh": {"size": 5e-4, "mode": "cartesian"},
        "pic": pic,
    })


def _target(sim, r: np.ndarray) -> np.ndarray:
    return sim._rw_c * (r + sim._rw_r0)


# ---- 1. 装荷 ------------------------------------------------------------------------------


def test_loading_follows_the_target_weight_and_density_is_uniform_to_the_axis():
    sim = _sim(_box(te=0.0, immobile=True))
    e = sim.get_particles("electron")
    i = sim.get_particles("ion")
    assert sim._rw and sim._rw_r0 == pytest.approx(5e-4)
    np.testing.assert_allclose(e["w"], _target(sim, e["y"]), rtol=1e-12)
    np.testing.assert_array_equal(e["w"], i["w"])             # quiet start (同じ位置・重み)
    np.testing.assert_array_equal(e["y"], i["y"])
    assert e["w"].sum() == pytest.approx(1e15 * sim._total_gas_volume, rel=1e-12)
    # セルあたりの粒子数は r/(r + r0) に比例 (一様の重みなら r に比例)
    counts, edges = np.histogram(e["y"], bins=np.arange(0.0, 0.0100001, 5e-4))
    mid = 0.5 * (edges[1:] + edges[:-1])
    expect = mid / (mid + sim._rw_r0)
    ratio = counts / expect
    assert np.max(np.abs(ratio[1:] / ratio[1:].mean() - 1.0)) < 0.05
    assert counts[0] > 0.2 * counts[-1]
    # 時間平均の密度は軸の上の節点まで一様 (表示の体積の補正)
    sim.run_batch(store_frames=False)
    g = sim.grid
    ne = sim.fields["n_e"].reshape(g.ny + 1, g.nx + 1)[:, 4:-4]
    assert ne[0].mean() == pytest.approx(1e15, rel=0.05)
    assert ne[1:9].mean() == pytest.approx(1e15, rel=0.02)


def test_uniform_weights_keep_the_old_loading():
    sim = _sim(_box(te=0.0, immobile=True, radial=False))
    e = sim.get_particles("electron")
    assert not sim._rw
    assert np.all(e["w"] == e["w"][0]) and e["w"][0] == pytest.approx(sim._w0)
    # 一様の重みでも軸の上の節点の時間平均の密度は 4/3 倍にならない
    sim.run_batch(store_frames=False)
    g = sim.grid
    ne = sim.fields["n_e"].reshape(g.ny + 1, g.nx + 1)[:, 4:-4]
    assert ne[0].mean() == pytest.approx(1e15, rel=0.08)


# ---- 2, 3. 分割・併合 --------------------------------------------------------------------


def _quiet_box():
    """場がほぼ無い (密度 1e3) 箱。粒子は手で置く (イオンは目標の重みで遠くに 1 個)。"""
    return _sim(_box(density=1e3, te=0.0, n_macro=1000, n_steps=1, immobile=True))


def test_split_keeps_position_and_velocity_and_the_weight_sum():
    sim = _quiet_box()
    r = 2.0e-3
    w = 5.3 * float(_target(sim, np.array([r]))[0])
    v = (1.0e3, 2.0e3, 3.0e3)
    sim.set_particles("electron", x=[0.01], y=[r], vx=[v[0]], vy=[v[1]], vz=[v[2]], w=[w])
    sim.run_batch(store_frames=False)
    p = sim.get_particles("electron")
    assert len(p["w"]) == 5
    assert p["w"].sum() == pytest.approx(w, rel=1e-15)
    np.testing.assert_allclose(p["w"], w / 5, rtol=1e-12)
    for key in ("x", "y", "vx", "vy", "vz"):
        assert np.ptp(p[key]) == 0.0, key
    assert sim.history["split"][-1] == 4 and sim.history["merged"][-1] == 0
    assert sim.history["n_e"][-1] == 5


def test_merge_pairs_light_particles_in_the_same_cell():
    sim = _quiet_box()
    # 同じセル (z 10.0〜10.5 mm、r 2.0〜2.5 mm) に軽い 2 個、別のセルに軽い 1 個
    x = np.array([0.01010, 0.01040, 0.01510])
    y = np.array([0.00210, 0.00240, 0.00510])
    w = 0.2 * _target(sim, y)
    vx = np.array([1.0e3, -2.0e3, 5.0e2])
    sim.set_particles("electron", x=x, y=y, vx=vx, vy=np.zeros(3), vz=np.zeros(3), w=w)
    sim.run_batch(store_frames=False)
    p = sim.get_particles("electron")
    assert len(p["w"]) == 2
    assert p["w"].sum() == pytest.approx(w.sum(), rel=1e-15)
    alone = np.argmin(np.abs(p["x"] - x[2]))
    assert p["w"][alone] == w[2]
    pair = 1 - alone
    assert p["w"][pair] == pytest.approx(w[0] + w[1], rel=1e-15)
    k = int(np.argmin(np.abs(vx[:2] - p["vx"][pair])))     # 一方の速度をそのまま持つ
    assert p["vx"][pair] == pytest.approx(vx[k], rel=1e-6)
    assert sim.history["merged"][-1] == 1 and sim.history["n_e"][-1] == 2


# ---- 4. 熱的なプラズマ (損失なし) -----------------------------------------------------------


def test_population_control_conserves_weight_and_stays_bounded():
    sim = _sim(_box(n_macro=40_000, n_steps=600, mcc=_elastic_mcc(7)))
    w0 = {name: float(sim.get_particles(name)["w"].sum()) for name in sim.species}
    n0 = len(sim.get_particles("electron")["w"])
    sim.run_batch(store_frames=False)
    h = sim.history
    assert h["split"][-1] > 1000 and h["merged"][-1] > 1000
    for name in sim.species:
        p = sim.get_particles(name)
        assert abs(float(p["w"].sum()) - w0[name]) < 1e-12 * w0[name], name
    e = sim.get_particles("electron")
    assert 0.9 * n0 < len(e["w"]) < 1.15 * n0
    assert h["n_e"][-1] == len(e["w"])                     # 履歴の生きている粒子数は併合を差し引く
    rho = e["w"] / _target(sim, e["y"])
    lo, hi = np.percentile(rho, [1, 99])
    assert lo > 0.45 and hi < 2.05
    # 運動エネルギーは期待値で保存 (併合の揺らぎと統計の加熱だけ)
    assert h["ke_e"][-1] == pytest.approx(h["ke_e"][0], rel=0.03)


# ---- 5. 軸の近くの雑音 -------------------------------------------------------------------


def _phi_noise(radial: bool) -> tuple[float, float]:
    """φ の節点ごとの時間の標準偏差: 軸の上 (r = 0) と r 10〜20 mm の二乗平均 (z の端を除く)。"""
    sim = _sim(_box(z_len=0.01, r_len=0.03, n_macro=24_000, n_steps=1000, radial=radial, immobile=True,
                    mcc=_elastic_mcc(8), seed=1))
    g = sim.grid
    r = g.y0 + g.dy * np.arange(g.ny + 1)
    s1 = np.zeros((g.ny + 1, g.nx + 1))
    s2 = np.zeros_like(s1)
    cnt = 0
    for k in range(1000):
        sim.step()
        if k >= 250:
            phi = sim._phi.get().reshape(g.ny + 1, g.nx + 1)
            s1 += phi
            s2 += phi * phi
            cnt += 1
    sd = np.sqrt(np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0.0))[:, 2:-2]
    axis = float(np.sqrt(np.mean(sd[r < 1e-9] ** 2)))
    mid = float(np.sqrt(np.mean(sd[(r > 0.0099) & (r < 0.0201)] ** 2)))
    return axis, mid


def test_near_axis_potential_noise_is_reduced():
    """一様の重みでは軸の上の電位の揺らぎが r 10〜20 mm の 5 倍余り (粒子が r に比例して少ない)。半径に比例した
    重みでは半分以下になり、軸から離れた所は変わらない。"""
    axis_u, mid_u = _phi_noise(radial=False)
    axis_r, mid_r = _phi_noise(radial=True)
    assert axis_u > 3.5 * mid_u
    assert axis_r < 0.6 * axis_u
    assert mid_r == pytest.approx(mid_u, rel=0.3)
