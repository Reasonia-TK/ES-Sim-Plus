"""v2 GPU PIC-MCC (直交格子 + EB) の物理検証 (prompts/119 P3)。

v1 の test_pic.py / test_mcc.py / test_pic_rz.py と同じ観点を v2 でも確かめる:

1. プラズマ振動: 冷たい電子の微小ドリフトで運動エネルギーが 2·f_pe で振動する
2. エネルギー保存: 同設定 (ωpe·dt = 0.1) で全エネルギーのドリフトが 5% 以内
3. MCC 衝突頻度: 一定断面積・単色電子で衝突数が N(1−e^{−ν_max dt})·ν/ν_max に一致
4. 電離閾値: 閾値未満の電子は電離しない / 閾値以上は余剰エネルギーを等分 (half)
5. 軸対称: quiet start で φ ≈ 0、リング粒子の密度規格化 (2π)、遠心運動 r(t) = √(r0² + (vθ t)²)
6. SEE: γ = 1 の電極では吸収イオン数 = 放出電子数
7. v1 PIC との比較: 同条件の CCP ストリップで時間平均密度が一致 (統計誤差の範囲)

GPU (CUDA) が無い環境では skip する (v2 PIC は現状 GPU 専用)。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from es_sim.device import cuda_available
from es_sim.fem import EPS0
from es_sim.particles import ME, MP, QE
from es_sim.schema import Project

pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA (CuPy) が使えない環境")

KB = 1.380649e-23


def _sim(project: Project):
    from es_sim.gpic import GpuPicSimulation

    return GpuPicSimulation(project)


def _wpe(density: float) -> float:
    return math.sqrt(density * QE**2 / (EPS0 * ME))


def _zero_cross_freq(sig: np.ndarray, dt: float) -> float:
    s = sig - sig.mean()
    idx = np.nonzero((s[:-1] < 0) & (s[1:] >= 0))[0]
    assert len(idx) >= 3, "ゼロクロスが少なすぎます"
    tz = idx + s[idx] / (s[idx] - s[idx + 1])
    return (len(tz) - 1) / ((tz[-1] - tz[0]) * dt)


# ---- 1, 2. プラズマ振動・エネルギー保存 (v1 test_pic.py と同じ設定) -----------------------

DENSITY = 1.0e14
L_BOX = 0.01
H_BOX = 0.03


def _oscillation_project(n_steps: int, n_macro: int = 60000) -> Project:
    return Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [L_BOX, 0], [L_BOX, H_BOX], [0, H_BOX]]},
                "boundaries": [{"edges": [0, 1, 2, 3], "type": "dirichlet", "voltage": 0.0}],
            },
            "mesh": {"size": 8e-4, "mode": "cartesian"},
            "pic": {
                "initial_plasma": {"density": DENSITY, "te_ev": 0.0, "ti_ev": 0.0, "ion_mass_amu": 40.0,
                                   "immobile_ions": True, "seed": 1},
                "n_macro": n_macro,
                "dt": None,
                "n_steps": n_steps,
                "frame_every": 1000,
            },
        }
    )


def _perturb(sim, v0: float) -> None:
    p = sim.get_particles("electron")
    sim.set_particles("electron", vx=p["vx"] + v0 * np.sin(2.0 * np.pi * p["x"] / L_BOX))


def test_plasma_oscillation_frequency():
    sim = _sim(_oscillation_project(400))
    assert abs(sim.dt * _wpe(DENSITY) - 0.1) < 1e-12
    _perturb(sim, 3.0e3)
    hist, _ = sim.run_batch(store_frames=False)
    f = _zero_cross_freq(np.asarray(hist["ke_e"]), sim.dt)
    f_expected = 2.0 * _wpe(DENSITY) / (2.0 * math.pi)
    assert abs(f - f_expected) / f_expected < 0.05


def test_energy_conservation():
    sim = _sim(_oscillation_project(350))
    _perturb(sim, 3.0e3)
    hist, _ = sim.run_batch(store_frames=False)
    total = np.asarray(hist["ke_e"]) + np.asarray(hist["ke_i"]) + np.asarray(hist["fe"])
    assert total[0] > 0.0
    assert np.max(np.abs(total - total[0])) / total[0] < 0.05


def test_amr_settings_select_amr_engine_only_when_refinement_happens():
    """mesh.amr (prompts/122): 細分化が起きる設定なら AMR 階層の PIC、起きなければ一様格子のまま。"""
    d = _oscillation_project(10, n_macro=2000).model_dump()
    d["mesh"]["amr"] = {"max_level": 2}            # 導体・誘電体が無いので境界細分化のタグが無い
    assert _sim(Project.model_validate(d)).amr is None
    d["mesh"]["amr"] = {"max_level": 1, "regions": [{"p1": [0.002, 0.01], "p2": [0.008, 0.02], "level": 1}]}
    sim = _sim(Project.model_validate(d))
    assert sim.amr is not None and sim.amr.hier.n_levels == 2
    assert not any("AMR" in w for w in sim.warnings)


# ---- 3, 4. MCC ------------------------------------------------------------------------


def _mcc_box(electron_processes, n_macro=200000, p_pa=10.0, dt=1e-11, n_steps=50) -> Project:
    """全周鏡面反射の箱 (電場ほぼ 0 の低密度)。"""
    return Project.model_validate(
        {
            "geometry": {"domain": {"polygon": [[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01]]}, "boundaries": []},
            "mesh": {"size": 1e-3, "mode": "cartesian"},
            "pic": {
                "initial_plasma": {"density": 1e3, "te_ev": 0.0, "ti_ev": 0.0, "ion_mass_amu": 40.0,
                                   "immobile_ions": True, "seed": 2},
                "n_macro": n_macro,
                "dt": dt,
                "n_steps": n_steps,
                "frame_every": 10000,
                "reflect_edges": [0, 1, 2, 3],
                "mcc": {"gas": {"name": "Ar", "pressure_pa": p_pa, "temperature_k": 300.0},
                        "electron_processes": electron_processes, "seed": 5},
            },
        }
    )


def _mono_iso(sim, e_ev: float, seed: int = 0) -> None:
    p = sim.get_particles("electron")
    n = len(p["x"])
    rng = np.random.default_rng(seed)
    ct = 1.0 - 2.0 * rng.random(n)
    ph = 2.0 * np.pi * rng.random(n)
    st = np.sqrt(1.0 - ct * ct)
    s = math.sqrt(2.0 * e_ev * QE / ME)
    sim.set_particles("electron", vx=s * st * np.cos(ph), vy=s * st * np.sin(ph), vz=s * ct)


def test_mcc_collision_frequency_constant_sigma():
    sigma0 = 1.0e-19
    procs = [{"kind": "elastic", "label": "const", "mass_ratio": 0.0,
              "energy_ev": [0.0, 1000.0], "sigma_m2": [sigma0, sigma0]}]
    n_steps = 40
    sim = _sim(_mcc_box(procs, n_steps=n_steps))
    _mono_iso(sim, 10.0)
    n = len(sim.get_particles("electron")["x"])
    hist, _ = sim.run_batch(store_frames=False)
    n_g = 10.0 / (KB * 300.0)
    v = math.sqrt(2.0 * 10.0 * QE / ME)
    nu = n_g * sigma0 * v
    expected = n * n_steps * (1.0 - math.exp(-nu * sim.dt))
    assert hist["coll_e"][-1] == pytest.approx(expected, rel=0.03)


def test_mcc_ionization_threshold_and_half_split():
    thr = 15.76
    procs = [{"kind": "ionization", "label": "ion", "threshold_ev": thr,
              "energy_ev": [thr, 1000.0], "sigma_m2": [1e-16, 1e-16]}]
    # 閾値未満: 10 eV の電子は電離しない
    sim = _sim(_mcc_box(procs, n_macro=20000, n_steps=5))
    _mono_iso(sim, 10.0)
    hist, _ = sim.run_batch(store_frames=False)
    assert hist["ion_events"][-1] == 0
    # 閾値以上: 巨大な断面積でほぼ全電子が 1 ステップ目に電離 → 余剰を等分 (17.5 eV × 2)
    sim = _sim(_mcc_box(procs, n_macro=20000, n_steps=1, p_pa=1000.0))
    _mono_iso(sim, 50.0)
    n0 = len(sim.get_particles("electron")["x"])
    hist, _ = sim.run_batch(store_frames=False)
    k = hist["ion_events"][-1]
    assert k > 0.95 * n0
    p = sim.get_particles("electron")
    assert len(p["x"]) == n0 + k
    assert len(sim.get_particles("ion")["x"]) == n0 + k
    e = 0.5 * ME * (p["vx"] ** 2 + p["vy"] ** 2 + p["vz"] ** 2) / QE
    ionized = np.abs(e - 0.5 * (50.0 - thr)) < 1e-6
    assert np.count_nonzero(ionized) == 2 * k


# ---- 5. 軸対称 ------------------------------------------------------------------------


def _rz_box(density=1e15, n_macro=400000, n_steps=2, te=0.0) -> Project:
    return Project.model_validate(
        {
            "coord": "rz",
            "geometry": {
                "domain": {"polygon": [[0, 0], [0.02, 0], [0.02, 0.01], [0, 0.01]]},
                "boundaries": [{"edges": [1, 2, 3], "type": "dirichlet", "voltage": 0.0}],
            },
            "mesh": {"size": 5e-4, "mode": "cartesian"},
            "pic": {
                "initial_plasma": {"density": density, "te_ev": te, "ti_ev": 0.0, "ion_mass_amu": 40.0,
                                   "immobile_ions": True, "seed": 3},
                "n_macro": n_macro,
                "dt": 1e-12,
                "n_steps": n_steps,
                "frame_every": 1000,
                "avg_steps": n_steps,
            },
        }
    )


def test_rz_quiet_start_and_ring_density_normalization():
    sim = _sim(_rz_box())
    hist, _ = sim.run_batch(store_frames=False)
    # quiet start (電子とイオンが同位置) なので電位はほぼ 0
    assert max(abs(hist["phi_min"][0]), abs(hist["phi_max"][0])) < 1e-6
    # リング粒子の密度 (2π 規格化) が初期密度を再現する (内部節点の平均、統計誤差 ~0.3%)
    g = sim.grid
    ne = sim.fields["n_e"].reshape(g.ny + 1, g.nx + 1)
    interior = ne[2:-2, 2:-2]
    assert np.mean(interior) == pytest.approx(1e15, rel=0.02)


def test_rz_centrifugal_free_motion_is_exact():
    """場の無い軸対称のリング: 3D 直線運動なので r(t) = √(r0² + (vθ t)²)、角運動量保存。"""
    sim = _sim(_rz_box(density=1e3, n_macro=1000, n_steps=200))
    n = 50
    r0 = np.linspace(1e-4, 5e-3, n)
    vth = np.full(n, 2e4)
    sim.set_particles("electron", x=np.full(n, 0.01), y=r0, vx=np.zeros(n), vy=np.zeros(n), vz=vth, w=np.full(n, 1e-6))
    sim.set_particles("ion", x=np.full(n, 0.01), y=r0, vx=np.zeros(n), vy=np.zeros(n), vz=np.zeros(n), w=np.full(n, 1e-6))
    sim.run_batch(store_frames=False)
    p = sim.get_particles("electron")
    t = 200 * sim.dt
    r_exact = np.sqrt(r0**2 + (vth * t) ** 2)
    assert np.max(np.abs(p["y"] - r_exact) / r_exact) < 1e-6
    assert np.max(np.abs(p["y"] * p["vz"] - r0 * vth) / (r0 * vth)) < 1e-9


# ---- 6. SEE ---------------------------------------------------------------------------


def test_see_gamma_one_emits_one_electron_per_ion():
    p = Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01]]},
                "boundaries": [{"edges": [3], "type": "dirichlet", "voltage": 0.0, "see_gamma": 1.0}],
            },
            "mesh": {"size": 1e-3, "mode": "cartesian"},
            "pic": {
                "initial_plasma": {"density": 1e3, "te_ev": 0.0, "ti_ev": 0.0, "ion_mass_amu": 40.0, "seed": 4},
                "n_macro": 5000,
                "dt": 1e-9,
                "n_steps": 200,
                "frame_every": 1000,
                "reflect_edges": [0, 1, 2],
            },
        }
    )
    sim = _sim(p)
    ions = sim.get_particles("ion")
    sim.set_particles("ion", vx=np.full(len(ions["x"]), -1e5))   # 全イオンを左 (γ=1 の電極) へ
    hist, _ = sim.run_batch(store_frames=False)
    assert hist["wall_i"][-1] == len(ions["x"])
    assert hist["see_events"][-1] == hist["wall_i"][-1]


# ---- 7. v1 PIC との比較 --------------------------------------------------------------------


def test_matches_v1_pic_on_ccp_strip():
    """同条件の CCP ストリップ (eduPIC Ar 断面積、上下反射) で v1 と時間平均密度が一致。"""
    from es_sim.pic import PicSimulation
    from es_sim.pic1d_presets import edupic_ar_processes

    e_procs, i_procs = edupic_ar_processes()
    L, H = 0.025, 0.002

    def project(mode: str) -> Project:
        return Project.model_validate({
            "geometry": {
                "domain": {"polygon": [[0, 0], [L, 0], [L, H], [0, H]]},
                "boundaries": [
                    {"edges": [3], "type": "dirichlet", "voltage": 0.0,
                     "voltage_rf": {"amplitude": 150.0, "freq_hz": 13.56e6, "phase_deg": 0.0}},
                    {"edges": [1], "type": "dirichlet", "voltage": 0.0},
                ],
            },
            "mesh": {"size": L / 100, "mode": mode},
            "pic": {
                "initial_plasma": {"density": 5e14, "te_ev": 2.0, "ti_ev": 0.026, "ion_mass_amu": 39.948, "seed": 1},
                "n_macro": 40000,
                "dt": 1.0 / (13.56e6 * 400),
                "n_steps": 4000,
                "avg_steps": 2000,
                "frame_every": 100000,
                "phase_bins": 0,
                "reflect_edges": [0, 2],
                "mcc": {"gas": {"name": "Ar", "pressure_pa": 10.0, "temperature_k": 300.0},
                        "electron_processes": [q.model_dump() for q in e_procs],
                        "ion_processes": [q.model_dump() for q in i_procs], "seed": 3},
            },
        })

    v1 = PicSimulation(project("structured"))
    v1.run_batch(store_frames=False)
    v2 = _sim(project("cartesian"))
    v2.run_batch(store_frames=False)

    def x_profile(nodes, values, nbins=10):
        edges = np.linspace(0.0, L, nbins + 1)
        k = np.clip(np.digitize(nodes[:, 0], edges) - 1, 0, nbins - 1)
        return np.bincount(k, weights=values, minlength=nbins) / np.maximum(np.bincount(k, minlength=nbins), 1)

    p1 = x_profile(v1.mesh.nodes, v1.fields["n_e"])
    p2 = x_profile(v2.mesh.nodes, v2.fields["n_e"])
    center = slice(3, 7)
    assert np.mean(p2[center]) == pytest.approx(np.mean(p1[center]), rel=0.15)
    i1 = x_profile(v1.mesh.nodes, v1.fields["n_i"])
    i2 = x_profile(v2.mesh.nodes, v2.fields["n_i"])
    assert np.mean(i2[center]) == pytest.approx(np.mean(i1[center]), rel=0.15)
