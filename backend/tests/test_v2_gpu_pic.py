"""v2 GPU PIC-MCC (直交格子 + EB) の物理検証 (prompts/119 P3)。

v1 の test_pic.py / test_mcc.py / test_pic_rz.py と同じ観点を v2 でも確かめる:

1. プラズマ振動: 冷たい電子の微小ドリフトで運動エネルギーが 2·f_pe で振動する
2. エネルギー保存: 同設定 (ωpe·dt = 0.1) で全エネルギーのドリフトが 5% 以内
3. MCC 衝突頻度: 一定断面積・単色電子で衝突数が N(1−e^{−ν_max dt})·ν/ν_max に一致
4. 電離閾値: 閾値未満の電子は電離しない / 閾値以上は余剰エネルギーを等分 (half)
5. 軸対称: quiet start で φ ≈ 0、リング粒子の密度規格化 (2π)、遠心運動 r(t) = √(r0² + (vθ t)²)
6. SEE: γ = 1 の電極では吸収イオン数 = 放出電子数
7. v1 PIC との比較: 同条件の CCP ストリップで時間平均密度が一致 (統計誤差の範囲)
8. 粒子の容量: 容量をぎりぎりにしても止まってはやり直して粒子を失わない (容量が足りる計算と 1 ステップずつ
   一致)、予備を越えたらエラー、配列の拡張はステップのストリームの順序に乗る

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


# ---- 8. 粒子の容量 (容量の停止とやり直し、拡張とストリームの順序) ------------------------------------


def _tight_capacity(monkeypatch, room: int = 1000, reserve: int = 4096) -> None:
    """容量をぎりぎりにする: 初めの余裕は 256 個、検査のたびに容量を「粒子数 + room」までしか広げない
    (先回りの拡張なし)。予備 (1 ステップの追加の分) は reserve 個。"""
    from es_sim.gpic import simulation as gsim

    for name, value in (("CAP_INIT", 1.0), ("CAP_HEADROOM_MIN", 256), ("CAP_GROW", 1.0), ("CAP_GROW_MIN", room),
                        ("CAP_AHEAD", 0), ("CAP_RESERVE_MIN", reserve), ("CAP_RESERVE_FRAC", 0.0)):
        monkeypatch.setattr(gsim, name, value)


def _see_stream(n_steps: int = 120):
    """γ = 1 の電極 (左) へ全イオン (2 万個) を流す箱: 毎ステップ約 200 個の二次電子を追加する。

    場はほぼ 0、電子は MCC なし、二次電子の乱数はイオンの番号で決まるので、ステップごとの粒子数は
    追加の順序 (atomicAdd) によらず決まる (容量の違う計算どうしを 1 ステップずつ比べられる)。
    """
    p = Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01]]},
                "boundaries": [{"edges": [3], "type": "dirichlet", "voltage": 0.0, "see_gamma": 1.0}],
            },
            "mesh": {"size": 1e-3, "mode": "cartesian"},
            "pic": {
                "initial_plasma": {"density": 1e3, "te_ev": 0.0, "ti_ev": 0.0, "ion_mass_amu": 40.0, "seed": 4},
                "n_macro": 20000,
                "dt": 1e-9,
                "n_steps": n_steps,
                "frame_every": 10000,
                "reflect_edges": [0, 1, 2],
            },
        }
    )
    sim = _sim(p)
    ions = sim.get_particles("ion")
    sim.set_particles("ion", vx=np.full(len(ions["x"]), -1e5))
    return sim


_COUNT_KEYS = ("n_e", "n_i", "wall_e", "wall_i", "see_events")


def test_capacity_halt_reruns_steps_exactly(monkeypatch):
    """容量がぎりぎり: 軟らかい容量を越える追加があると次のステップから止まり、ホストが次の検査で配列を広げて
    同じステップからやり直す。粒子数の推移は容量が足りている計算と 1 ステップずつ一致し、粒子は失われない
    (旧実装は 32 ステップごとの検査の間に容量を越えると「粒子配列の容量を超えました」で止まった)。"""
    ref = _see_stream()
    h_ref, _ = ref.run_batch(store_frames=False)
    assert not ref.capacity_log
    _tight_capacity(monkeypatch)
    sim = _see_stream()
    h, _ = sim.run_batch(store_frames=False)
    assert len([e for e in sim.capacity_log if e["halted_steps"] > 0]) >= 3
    assert sim.step_count == 120 and len(h["t"]) == 120
    np.testing.assert_allclose(np.diff(h["t"]), sim.dt, rtol=1e-9)
    for key in _COUNT_KEYS:
        assert h[key] == h_ref[key], key
    assert h["see_events"][-1] == h["wall_i"][-1] == 20000


def test_capacity_step_api_grows_without_halting(monkeypatch):
    """同期 API の step() は毎ステップ検査するので、容量を越えたらその場で広げ、ステップは止まらない。"""
    ref = _see_stream()
    h_ref, _ = ref.run_batch(store_frames=False)
    _tight_capacity(monkeypatch)
    sim = _see_stream()
    for _ in range(120):
        sim.step()
    sim._flush_history()
    assert sim.capacity_log and all(e["halted_steps"] == 0 for e in sim.capacity_log)
    for key in _COUNT_KEYS:
        assert sim.history[key] == h_ref[key], key


def test_capacity_overflow_beyond_reserve_raises(monkeypatch):
    """1 ステップの追加が予備を越えたときだけ粒子が失われ、次の検査でエラーにする。"""
    _tight_capacity(monkeypatch, reserve=64)
    sim = _see_stream()
    with pytest.raises(RuntimeError, match="予備の容量"):
        sim.run_batch(store_frames=False)


_SPIN = r"""
extern "C" __global__ void spin(const long long cycles)
{
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
}
"""


def test_grow_is_ordered_with_the_step_stream():
    """配列の拡張 (確保・コピー) はステップのストリームに積む: 既定のストリームが詰まっていても、拡張の直後に
    ステップのストリームで読む粒子は元のまま。旧実装は既定のストリームでコピーしており、非ブロッキングの
    ステップのストリームとは順序がなく、Windows (WDDM) では稀にコピー前の配列をステップが読んだ (イオンの
    電荷が抜けた数ステップで電子が加速され、電離の暴走で容量を越えた)。"""
    sim = _sim(_oscillation_project(10, n_macro=20000))
    cp = sim.cp
    el = sim.species["electron"]
    n = int(sim._cnt[0].get())
    # 読み出し先と既定のストリームのメモリプールの空きは先に確保しておく (塞いだ後に cudaMalloc が走ると
    # デバイス全体が同期し、旧実装でも順序が保たれてしまう)
    with sim._stream:
        before = {f: el.arrays[f][:n].get() for f in ("x", "vx", "w")}
        after = {f: cp.empty(n) for f in before}
    spare = cp.empty(64 << 20, dtype=np.uint8)
    del spare
    cp.cuda.Device().synchronize()
    cp.RawKernel(_SPIN, "spin")((1,), (1,), (np.int64(200_000_000),))   # 既定のストリームを ~0.1 s 塞ぐ
    sim._grow(el, 4 * el.cap)
    with sim._stream:
        for f in before:
            cp.copyto(after[f], el.arrays[f][:n])
    cp.cuda.Device().synchronize()
    for f, v in before.items():
        assert np.array_equal(after[f].get(), v), f


@pytest.mark.parametrize("grid", ["uniform", "amr"])
def test_capacity_halts_keep_ionizing_ccp_consistent(monkeypatch, grid):
    """電離が続く CCP ストリップで容量をぎりぎりにしても、止まってはやり直しを繰り返して最後まで進む: 粒子数の
    恒等式 (初期 + 生成 − 壁への吸収) が厳密に成り立ち、時間平均・位相分解・EEDF の回数は平均区間のステップ数の
    まま (止まったステップは数えない)。AMR は動的再格子化のタグの積算も通る。"""
    from es_sim.pic1d_presets import edupic_ar_processes

    e_procs, i_procs = edupic_ar_processes()
    L, H = 0.025, 0.002
    n_steps, avg, n0 = 1600, 400, 20000

    def project() -> Project:
        d = {
            "geometry": {
                "domain": {"polygon": [[0, 0], [L, 0], [L, H], [0, H]]},
                "boundaries": [
                    {"edges": [3], "type": "dirichlet", "voltage": 0.0,
                     "voltage_rf": {"amplitude": 150.0, "freq_hz": 13.56e6, "phase_deg": 0.0}},
                    {"edges": [1], "type": "dirichlet", "voltage": 0.0},
                ],
            },
            "mesh": {"size": L / 100, "mode": "cartesian"},
            "pic": {
                "initial_plasma": {"density": 5e14, "te_ev": 2.0, "ti_ev": 0.026, "ion_mass_amu": 39.948, "seed": 1},
                "n_macro": n0,
                "dt": 1.0 / (13.56e6 * 400),
                "n_steps": n_steps,
                "avg_steps": avg,
                "frame_every": 100000,
                "phase_bins": 8,
                "reflect_edges": [0, 2],
                "eedf_regions": [{"label": "c", "p1": [0.010, 0.0], "p2": [0.015, H], "bins": 50}],
                "mcc": {"gas": {"name": "Ar", "pressure_pa": 10.0, "temperature_k": 300.0},
                        "electron_processes": [q.model_dump() for q in e_procs],
                        "ion_processes": [q.model_dump() for q in i_procs], "seed": 3},
            },
        }
        if grid == "amr":
            d["mesh"]["amr"] = {"max_level": 1, "refine_boundaries": False, "blocking_factor": 4,
                                "regions": [{"p1": [0.010, 0.0], "p2": [0.015, H], "level": 1}],
                                "pic_regrid_every": 100, "pic_h_over_debye": 1.5}
        return Project.model_validate(d)

    ref = _sim(project())
    h_ref, _ = ref.run_batch(store_frames=False)
    _tight_capacity(monkeypatch, room=64)
    sim = _sim(project())
    assert (sim.amr is not None) == (grid == "amr")
    h, _ = sim.run_batch(store_frames=False)
    assert any(e["halted_steps"] > 0 for e in sim.capacity_log)
    assert sim.step_count == n_steps and len(h["t"]) == n_steps
    np.testing.assert_allclose(np.diff(h["t"]), sim.dt, rtol=1e-9)
    assert h["n_e"][-1] == n0 + h["ion_events"][-1] + h["see_events"][-1] - h["wall_e"][-1]
    assert h["n_i"][-1] == n0 + h["ion_events"][-1] - h["wall_i"][-1]
    assert sim.fields["avg_steps"] == avg
    assert int(sim._cyc_count.get().sum()) == avg
    assert sim.eedf_results[0]["n_samples"] == avg
    assert h["n_e"][-1] == pytest.approx(h_ref["n_e"][-1], rel=0.1)
