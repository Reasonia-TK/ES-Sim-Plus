"""MCC 衝突 + LXCat インポート + SEE のテスト (prompts/19)。

- パーサー: 合成フィクスチャ (常時) と実 LXCat ファイル (あれば) の両形式
- 衝突頻度: 一定断面積・電場なし・単色電子で ν = n_g σ v と数%以内で一致
- 電離: 一様電場 + 電離のみで電子数が増加 / 閾値未満では電離ゼロ
- SEE: γ=1 電極へのイオン打ち込みで吸収数 = SEE 電子生成数
- CCP スモーク: MCC 有効でも NaN なく完走
"""

import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from es_sim import _numba_kernels
from es_sim.lxcat import parse_lxcat
from es_sim.mcc import KB, MccModel
from es_sim.particles import ME, MP, QE, _locate_initial
from es_sim.pic import PicSimulation
from es_sim.schema import MccGas, MccSettings, Project, XsProcess
from es_sim.server import app

DATA = Path(__file__).parent / "data"
SYN_E = DATA / "synthetic_electron.txt"
SYN_I = DATA / "synthetic_ion.txt"
REAL_E = DATA / "Ar電子衝突断面積.txt"
REAL_I = DATA / "Arイオン衝突断面積.txt"


# ---- 1. パーサー: 合成フィクスチャ (常時実行) ----------------------------------


def test_parse_synthetic_electron():
    """標準ブロック形式: ELASTIC/EFFECTIVE/EXCITATION/IONIZATION/ATTACHMENT。"""
    procs, warnings = parse_lxcat(SYN_E.read_text(encoding="utf-8"), "electron")
    # EFFECTIVE は elastic として取り込み、ATTACHMENT はスキップ → 4 プロセス
    assert [p.kind for p in procs] == ["elastic", "elastic", "excitation", "ionization"]
    assert len(warnings) == 2
    assert any("EFFECTIVE" in w for w in warnings)
    assert any("ATTACHMENT" in w for w in warnings)

    elastic, effective, excitation, ionization = procs
    assert abs(elastic.mass_ratio - 1.36e-5) < 1e-12
    assert elastic.threshold_ev == 0.0
    assert len(elastic.energy_ev) == 4 and len(elastic.sigma_m2) == 4
    assert abs(excitation.threshold_ev - 11.55) < 1e-9
    assert abs(ionization.threshold_ev - 15.759) < 1e-9
    assert "Elastic" in elastic.label  # PROCESS 行由来のラベル
    for p in procs:
        assert len(p.energy_ev) == len(p.sigma_m2) >= 2
        assert np.all(np.diff(p.energy_ev) >= 0.0)  # エネルギー昇順


def test_parse_synthetic_ion():
    """タイプ行なし形式 (SPECIES:/PROCESS: ブロック): Backscat + Isotropic。"""
    procs, warnings = parse_lxcat(SYN_I.read_text(encoding="utf-8"), "ion")
    assert [p.kind for p in procs] == ["backscat", "isotropic"]
    assert warnings == []
    for p in procs:
        assert p.threshold_ev == 0.0 and p.mass_ratio == 0.0
        assert len(p.energy_ev) == len(p.sigma_m2) == 3
        assert np.all(np.diff(p.energy_ev) >= 0.0)


def test_parse_species_filter():
    """種別フィルタ: 電子ファイルを species='ion' で読むと全て警告スキップ。"""
    procs, warnings = parse_lxcat(SYN_E.read_text(encoding="utf-8"), "ion")
    assert procs == []
    assert len(warnings) >= 3  # elastic×2 + excitation + ionization 分のスキップ警告


def test_parse_invalid_text_raises():
    """ブロックが 1 つも無いテキストは ValueError (エンドポイントでは 422)。"""
    with pytest.raises(ValueError):
        parse_lxcat("これは LXCat ファイルではありません\n1.0 2.0\n", "electron")


def test_lxcat_endpoint():
    """POST /lxcat/parse: 正常系 200、パース失敗 422。"""
    client = TestClient(app)
    r = client.post(
        "/lxcat/parse",
        json={"text": SYN_E.read_text(encoding="utf-8"), "species": "electron"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["processes"]) == 4
    assert len(body["warnings"]) == 2

    r = client.post("/lxcat/parse", json={"text": "no blocks here", "species": "electron"})
    assert r.status_code == 422


# ---- 2. パーサー: 実 LXCat ファイル (あれば実行) --------------------------------


@pytest.mark.skipif(not REAL_E.exists(), reason="実 LXCat 電子ファイルなし (git 管理外)")
def test_parse_real_electron_file():
    """Morgan Ar 電子: elastic + excitation×2 + ionization。"""
    procs, warnings = parse_lxcat(REAL_E.read_text(encoding="utf-8"), "electron")
    assert [p.kind for p in procs] == ["elastic", "excitation", "excitation", "ionization"]
    assert warnings == []
    assert abs(procs[0].mass_ratio - 1.36e-5) < 1e-12
    assert abs(procs[1].threshold_ev - 11.55) < 1e-9
    assert abs(procs[2].threshold_ev - 11.55) < 1e-9
    assert abs(procs[3].threshold_ev - 15.759) < 1e-9
    for p in procs:
        assert len(p.energy_ev) == len(p.sigma_m2) >= 10
        assert np.all(np.diff(p.energy_ev) >= 0.0)


@pytest.mark.skipif(not REAL_I.exists(), reason="実 LXCat イオンファイルなし (git 管理外)")
def test_parse_real_ion_file():
    """Phelps Ar+ + Ar: Backscat + Isotropic (タイプ行なし形式)。"""
    procs, warnings = parse_lxcat(REAL_I.read_text(encoding="utf-8"), "ion")
    assert [p.kind for p in procs] == ["backscat", "isotropic"]
    assert warnings == []
    for p in procs:
        assert len(p.energy_ev) == len(p.sigma_m2) >= 10
        assert np.all(np.diff(p.energy_ev) >= 0.0)


# ---- 3. 衝突頻度: ν = n_g σ v との一致 -----------------------------------------


def test_collision_frequency_constant_sigma():
    """一定断面積 σ・電場なし・単色電子で、測定衝突率が ν = n_g σ v と数%で一致。"""
    sigma0 = 1.0e-19
    e0 = 10.0  # [eV]
    pressure = 10.0
    temperature = 300.0
    proc = XsProcess(
        kind="elastic",
        label="const sigma",
        mass_ratio=0.0,  # エネルギー損失なし → 速さ一定 → ν 一定
        energy_ev=[0.0, 40.0],
        sigma_m2=[sigma0, sigma0],
    )
    settings = MccSettings(
        gas=MccGas(name="Ar", pressure_pa=pressure, temperature_k=temperature),
        electron_processes=[proc],
        ion_processes=[],
        seed=7,
    )
    model = MccModel(settings, m_ion=40.0 * MP)

    n_gas = pressure / (KB * temperature)
    v0 = math.sqrt(2.0 * e0 * QE / ME)
    nu = n_gas * sigma0 * v0
    dt = 0.01 / nu  # ν·dt = 0.01 (1ステップ1衝突近似のバイアスを抑える)

    n = 50000
    rng = np.random.default_rng(3)
    th = rng.random(n) * 2.0 * np.pi
    # 2d3v: 速度は3成分 (vz = 0 の単色電子)
    v = v0 * np.stack([np.cos(th), np.sin(th), np.zeros(n)], axis=1)
    x = np.zeros((n, 2))
    w = np.ones(n)
    elem = np.zeros(n, dtype=np.int64)

    steps = 40
    count = 0
    for _ in range(steps):
        res = model.collide_electrons(x, v, w, elem, dt)
        count += res.n_coll
        # mass_ratio=0 の弾性衝突なので速さは保存される
        assert np.allclose(np.linalg.norm(v, axis=1), v0, rtol=1e-10)

    rate = count / (n * steps * dt)
    assert abs(rate - nu) / nu < 0.05


# ---- 4. 電離 -------------------------------------------------------------------


_IONIZATION_PROC = {
    "kind": "ionization",
    "label": "synthetic ionization",
    "threshold_ev": 10.0,
    "mass_ratio": 0.0,
    "energy_ev": [10.0, 20.0, 1000.0],
    "sigma_m2": [0.0, 2.0e-20, 2.0e-20],
}


def test_ionization_growth_in_uniform_field():
    """一様電場 + 電離のみ: 電場で加速された電子が電離し、電子数が増加する。"""
    project = Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [0.01, 0], [0.01, 0.005], [0, 0.005]]},
                "boundaries": [
                    {"edges": [3], "voltage": 0.0},     # 左辺: 0 V
                    {"edges": [1], "voltage": 300.0},   # 右辺: +300 V (電子を加速)
                ],
            },
            "mesh": {"size": 5e-4},
            "pic": {
                "initial_plasma": {
                    "density": 1e10,  # 空間電荷が無視できる微小密度
                    "te_ev": 0.5,
                    "ti_ev": 0.03,
                    "ion_mass_amu": 40.0,
                    "immobile_ions": True,
                    "seed": 3,
                },
                "n_macro": 2000,
                "dt": 2e-11,
                "n_steps": 150,
                "frame_every": 50,
                "mcc": {
                    "gas": {"name": "Ar", "pressure_pa": 100.0, "temperature_k": 300.0},
                    "electron_processes": [_IONIZATION_PROC],
                    "ion_processes": [],
                    "seed": 5,
                },
            },
        }
    )
    sim = PicSimulation(project)
    n0 = len(sim.species["electron"].x)
    history, _ = sim.run_batch()

    assert history["ion_events"][-1] > 0          # 電離が発生
    assert max(history["n_e"]) > n0               # 電子数が初期値を超えて増加
    assert history["n_i"][-1] > n0                # 新イオンも生成 (イオンは不動で壁吸収なし)
    for key in ("ke_e", "ke_i", "fe", "phi_min", "phi_max"):
        assert np.all(np.isfinite(history[key]))


@pytest.mark.skipif(
    not _numba_kernels.HAVE_NUMBA,
    reason="push+walk再利用バッファはNumba有効時のみ作成される",
)
def test_mcc_buffered_append_matches_concatenate(monkeypatch):
    """MCC電離のバッファ追記が従来concatenateと最終状態・乱数まで一致する。"""
    project = Project.model_validate(
        {
            "coord": "rz",
            "geometry": {
                "domain": {
                    "polygon": [[0, 0], [0.01, 0], [0.01, 0.005], [0, 0.005]]
                },
                "boundaries": [
                    {"edges": [3], "voltage": 0.0},
                    {"edges": [1], "voltage": 300.0},
                ],
            },
            "mesh": {"size": 5e-4},
            "pic": {
                "initial_plasma": {
                    "density": 1e10,
                    "te_ev": 0.5,
                    "ti_ev": 0.03,
                    "ion_mass_amu": 40.0,
                    "immobile_ions": True,
                    "seed": 3,
                },
                "n_macro": 1200,
                "dt": 2e-11,
                "n_steps": 80,
                "frame_every": 1000,
                "threads": 1,
                "mcc": {
                    "gas": {
                        "name": "Ar",
                        "pressure_pa": 100.0,
                        "temperature_k": 300.0,
                    },
                    "electron_processes": [_IONIZATION_PROC],
                    "ion_processes": [],
                    "seed": 5,
                },
            },
        }
    )

    sim_buffered = PicSimulation(project)
    for _ in range(80):
        sim_buffered.step()
    assert sim_buffered.ion_events > 0
    assert any(
        np.shares_memory(buf[1], sim_buffered.species["electron"].x)
        for buf in sim_buffered._push_walk_buffers["electron"]
    )

    def reference_append(
        self, sp, x_new, v_new, w_new, elem_new, bary_new
    ) -> None:
        """変更前と同じく、追加のたびに全配列を連結する参照経路。"""
        n_old = len(sp.x)
        bary_old = sp.bary
        sp.x = np.concatenate([sp.x, x_new])
        sp.v = np.concatenate([sp.v, v_new])
        sp.w = np.concatenate([sp.w, w_new])
        sp.elem = np.concatenate([sp.elem, elem_new])
        if bary_old is not None and len(bary_old) == n_old:
            sp.bary = np.concatenate([bary_old, bary_new])
        else:
            sp.bary = None
        sp.nidx = None

    monkeypatch.setattr(
        PicSimulation, "_append_species_buffered", reference_append
    )
    sim_reference = PicSimulation(project)
    for _ in range(80):
        sim_reference.step()

    assert sim_buffered.coll_e == sim_reference.coll_e
    assert sim_buffered.ion_events == sim_reference.ion_events
    for name in sim_buffered.species:
        actual = sim_buffered.species[name]
        expected = sim_reference.species[name]
        assert actual.wall_absorbed == expected.wall_absorbed
        for attr in ("x", "v", "w", "elem", "bary"):
            assert np.array_equal(getattr(actual, attr), getattr(expected, attr)), (
                name,
                attr,
            )
    for key in sim_buffered.history:
        assert np.array_equal(
            np.asarray(sim_buffered.history[key]),
            np.asarray(sim_reference.history[key]),
        ), key
    assert np.array_equal(
        sim_buffered.mcc.rng.random(16),
        sim_reference.mcc.rng.random(16),
    )


def test_no_ionization_below_threshold():
    """閾値 (10 eV) 未満の単色電子 (5 eV) では電離が一切起きない。"""
    settings = MccSettings(
        gas=MccGas(name="Ar", pressure_pa=100.0, temperature_k=300.0),
        electron_processes=[XsProcess(**_IONIZATION_PROC)],
        ion_processes=[],
        seed=1,
    )
    model = MccModel(settings, m_ion=40.0 * MP)

    n = 10000
    v0 = math.sqrt(2.0 * 5.0 * QE / ME)
    v = np.tile(np.array([v0, 0.0, 0.0]), (n, 1))
    x = np.zeros((n, 2))
    w = np.ones(n)
    elem = np.zeros(n, dtype=np.int64)
    dt = 1e-10

    for _ in range(200):
        res = model.collide_electrons(x, v, w, elem, dt)
        assert res.n_coll == 0
        assert res.n_ionization == 0
    assert np.array_equal(v, np.tile(np.array([v0, 0.0, 0.0]), (n, 1)))  # 速度不変


# ---- 5. SEE --------------------------------------------------------------------


def test_see_gamma_one_absorption_equals_emission():
    """γ=1 の電極 (下辺) へイオンを打ち込み、吸収数 = SEE 電子生成数を確認。"""
    project = Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01]]},
                "boundaries": [
                    {"edges": [0], "voltage": 0.0, "see_gamma": 1.0},  # 下辺: γ=1
                    {"edges": [1, 2, 3], "voltage": 0.0},
                ],
            },
            "mesh": {"size": 8e-4},
            "pic": {
                "dt": 1e-9,
                "n_steps": 80,
                "n_macro": 10,
                "frame_every": 80,
                "see_energy_ev": 2.0,
            },
        }
    )
    sim = PicSimulation(project)

    # イオンを中央高さから真下 (γ=1 の下辺) へ向けて手動装荷する
    n = 200
    x = np.stack([np.linspace(0.0015, 0.0085, n), np.full(n, 0.005)], axis=1)
    io = sim.species["ion"]
    io.x = x
    io.v = np.tile(np.array([0.0, -1.0e5, 0.0]), (n, 1))
    io.w = np.ones(n)
    io.elem = _locate_initial(sim.coeffs, x)

    history, _ = sim.run_batch()

    assert io.wall_absorbed == n                      # 全イオンが下辺で吸収
    assert history["see_events"][-1] == n             # 吸収数 = SEE 電子生成数
    assert max(history["n_e"]) > 0                    # 生成電子が実在した
    assert np.all(np.diff(history["see_events"]) >= 0)  # 累計カウンタは単調非減少


# ---- 6. CCP スモーク (MCC + SEE 有効) -------------------------------------------


def _load_ar_processes():
    """実 LXCat ファイルがあればそれを、なければ合成フィクスチャをパースする。"""
    if REAL_E.exists() and REAL_I.exists():
        pe, _ = parse_lxcat(REAL_E.read_text(encoding="utf-8"), "electron")
        pi, _ = parse_lxcat(REAL_I.read_text(encoding="utf-8"), "ion")
    else:
        pe, _ = parse_lxcat(SYN_E.read_text(encoding="utf-8"), "electron")
        pi, _ = parse_lxcat(SYN_I.read_text(encoding="utf-8"), "ion")
    return pe, pi


def test_ccp_smoke_with_mcc():
    """平行平板 CCP + RF + 初期プラズマ + MCC + SEE で NaN なく完走すること。"""
    pe, pi = _load_ar_processes()
    freq = 13.56e6
    dt = 5e-10
    n_steps = 100
    project = Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [0.02, 0], [0.02, 0.01], [0, 0.01]]},
                "boundaries": [
                    {
                        "edges": [3],  # 左辺: RF 電極
                        "voltage": 0.0,
                        "voltage_rf": {"amplitude": 50.0, "freq_hz": freq, "phase_deg": 0.0},
                        "see_gamma": 0.1,
                    },
                    {"edges": [1], "voltage": 0.0, "see_gamma": 0.1},  # 右辺: GND
                ],
            },
            "mesh": {"size": 1.2e-3},
            "pic": {
                "initial_plasma": {
                    "density": 1.0e14,
                    "te_ev": 2.0,
                    "ti_ev": 0.03,
                    "ion_mass_amu": 40.0,
                    "immobile_ions": False,
                    "seed": 2,
                },
                "n_macro": 4000,
                "dt": dt,
                "n_steps": n_steps,
                "frame_every": 20,
                "mcc": {
                    "gas": {"name": "Ar", "pressure_pa": 10.0, "temperature_k": 300.0},
                    "electron_processes": [p.model_dump() for p in pe],
                    "ion_processes": [p.model_dump() for p in pi],
                    "seed": 4,
                },
                "see_energy_ev": 2.0,
            },
        }
    )
    sim = PicSimulation(project)
    history, frames = sim.run_batch()

    assert len(history["t"]) == n_steps  # 完走
    for key in ("ke_e", "ke_i", "fe", "phi_min", "phi_max"):
        assert np.all(np.isfinite(history[key])), f"{key} に非有限値"
    for frame in frames:
        assert np.all(np.isfinite(frame["phi"]))
    assert history["n_e"][-1] > 0 and history["n_i"][-1] > 0
    # 累計カウンタは単調非減少で、この条件では電子衝突が実際に起きる
    assert np.all(np.diff(history["coll_e"]) >= 0)
    assert np.all(np.diff(history["ion_events"]) >= 0)
    assert np.all(np.diff(history["see_events"]) >= 0)
    assert history["coll_e"][-1] > 0
