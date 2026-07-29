"""FEM-PIC のテスト (仕様書 §9 / prompts/15)。

WebSocket ではなく PicSimulation.run_batch の同期 API で検証する。

1. プラズマ振動: 冷たい電子 + 固定イオンに微小一様ドリフトを与え、
   運動エネルギー振動の周波数が 2×fpe と 10% 以内で一致すること
2. エネルギー保存: 同上設定 (ωpe·dt = 0.1) で全エネルギードリフトが 5% 以内
3. CCP スモーク: 平行平板 (片側 RF ±50V/13.56MHz、対向 GND) + 初期プラズマで
   200 ステップ → NaN なし・粒子数単調非増加・RF 1周期平均の中央電位が両壁より高い
4. 位相別プロファイル計測 (prompts/75): timing の各キーが妥当な値を持ち、
   合計が run_batch の実測壁時計時間と一致すること
5. walk コスト診断 (prompts/88): timing の walk_cells_est_e/i・h_mean_m・h_min_m が
   正の妥当な値であること、時間キーの合計 (4.) を汚染しないこと
"""

import math
import time

import numpy as np

from es_sim.fem import EPS0
from es_sim.particles import ME, QE
from es_sim.pic import WALK_DIAG_KEYS, PicSimulation
from es_sim.schema import Project

DENSITY = 1.0e14  # [m^-3]
L = 0.01          # 矩形ドメインの幅 (x 方向、振動モードの波長) [m]
H = 0.03          # 矩形ドメインの高さ [m]。縦長にして y 壁境界層の影響を減らす


def _oscillation_project(n_steps: int) -> Project:
    """全周接地の縦長矩形ドメイン + 冷たい電子・固定イオンの一様プラズマ。

    吸収壁 (接地電極) の境界層ではモードの電場が曲げられ、粒子の壁吸収による
    減衰・周波数汚染が生じる。縦長 (H = 3L) にして境界層の体積比を下げる。
    """
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
                    "te_ev": 0.0,       # 冷たいプラズマ
                    "ti_ev": 0.0,
                    "ion_mass_amu": 40.0,
                    "immobile_ions": True,
                    "seed": 1,
                },
                "n_macro": 9000,
                "dt": None,             # 0.1/ωpe
                "n_steps": n_steps,
                "frame_every": 200,
            },
        }
    )


def _wpe(density: float) -> float:
    return math.sqrt(density * QE**2 / (EPS0 * ME))


def _zero_cross_freq(sig: np.ndarray, dt: float) -> float:
    """負→正のゼロクロス間隔の平均から周波数を推定する (線形補間つき)。"""
    s = sig - sig.mean()
    idx = np.nonzero((s[:-1] < 0) & (s[1:] >= 0))[0]
    assert len(idx) >= 3, "ゼロクロスが少なすぎます"
    tz = idx + s[idx] / (s[idx] - s[idx + 1])
    return (len(tz) - 1) / ((tz[-1] - tz[0]) * dt)


# ---- 1. プラズマ振動の周波数 --------------------------------------------------


def _perturb_electrons(sim: PicSimulation, v0: float) -> None:
    """電子に微小な正弦モードのドリフト v_x = v0 sin(2πx/L) を与える。

    接地壁の箱では一様変位 (平均成分) は壁の鏡像電荷に遮蔽されて復元場が
    立たないため、平均がゼロで x 壁でも速度が消える k = 2π/L モードを使う。
    このとき ξ̈ = -ωpe²ξ が成り立つ (冷たいプラズマの Langmuir 振動は
    波数によらず ω = ωpe)。
    """
    el = sim.species["electron"]
    el.v[:, 0] += v0 * np.sin(2.0 * np.pi * el.x[:, 0] / L)


def test_plasma_oscillation_frequency():
    """冷たい電子の微小ドリフトによるプラズマ振動。
    運動エネルギーは KE ∝ cos²(ωpe t) なので 2×fpe で振動する。"""
    sim = PicSimulation(_oscillation_project(n_steps=400))
    assert abs(sim.dt * _wpe(DENSITY) - 0.1) < 1e-12  # 既定 dt = 0.1/ωpe

    # 変位振幅 v0/ωpe ≈ 5e-6 m << L の微小摂動
    _perturb_electrons(sim, 3.0e3)
    history, _ = sim.run_batch()

    ke = np.asarray(history["ke_e"])
    f_measured = _zero_cross_freq(ke, sim.dt)

    f_expected = 2.0 * _wpe(DENSITY) / (2.0 * math.pi)  # 2×fpe
    assert abs(f_measured - f_expected) / f_expected < 0.10


# ---- 2. エネルギー保存 --------------------------------------------------------


def test_energy_conservation():
    """(運動 + 場) エネルギーのドリフトが初期全エネルギー比 5% 以内。"""
    sim = PicSimulation(_oscillation_project(n_steps=350))
    _perturb_electrons(sim, 3.0e3)
    history, _ = sim.run_batch()

    total = (
        np.asarray(history["ke_e"])
        + np.asarray(history["ke_i"])
        + np.asarray(history["fe"])
    )
    e0 = total[0]
    assert e0 > 0.0
    assert np.max(np.abs(total - e0)) / e0 < 0.05


# ---- 3. CCP スモーク ----------------------------------------------------------


def test_ccp_smoke():
    """平行平板 CCP (無衝突)。左壁 RF ±50V/13.56MHz、右壁 GND。
    200 ステップで NaN なし・粒子数単調非増加・シース形成の定性確認。"""
    freq = 13.56e6
    dt = 5e-10  # ωpe·dt ≈ 0.28
    n_steps = 200
    project = Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [0.02, 0], [0.02, 0.01], [0, 0.01]]},
                "boundaries": [
                    {
                        "edges": [3],  # 左辺: RF 電極
                        "type": "dirichlet",
                        "voltage": 0.0,
                        "voltage_rf": {"amplitude": 50.0, "freq_hz": freq, "phase_deg": 0.0},
                    },
                    {"edges": [1], "type": "dirichlet", "voltage": 0.0},  # 右辺: GND
                ],
            },
            "mesh": {"size": 1.2e-3},
            "pic": {
                "initial_plasma": {
                    "density": DENSITY,
                    "te_ev": 2.0,
                    "ti_ev": 0.03,
                    "ion_mass_amu": 40.0,
                    "immobile_ions": False,
                    "seed": 2,
                },
                "n_macro": 4000,
                "dt": dt,
                "n_steps": n_steps,
                "frame_every": 1,  # 全ステップの φ を取得して周期平均する
            },
        }
    )
    sim = PicSimulation(project)
    history, frames = sim.run_batch()

    # NaN なし
    for key in ("ke_e", "ke_i", "fe", "phi_min", "phi_max"):
        assert np.all(np.isfinite(history[key])), f"{key} に非有限値"
    assert len(frames) == n_steps

    # 粒子数は単調非増加 (注入なし・壁吸収のみ)
    n_e = np.asarray(history["n_e"])
    n_i = np.asarray(history["n_i"])
    assert np.all(np.diff(n_e) <= 0)
    assert np.all(np.diff(n_i) <= 0)
    assert n_e[-1] > 0 and n_i[-1] > 0

    # RF 1周期平均の中央電位が両壁より高い (シース形成の定性確認)
    period_steps = int(round(1.0 / freq / dt))  # ≈ 147
    phis = np.asarray([f["phi"] for f in frames[-period_steps:]])
    assert np.all(np.isfinite(phis))
    phi_avg = phis.mean(axis=0)

    nodes = sim.mesh.nodes
    center_mask = np.hypot(nodes[:, 0] - 0.01, nodes[:, 1] - 0.005) < 2e-3
    left_mask = nodes[:, 0] < 1e-9
    right_mask = nodes[:, 0] > 0.02 - 1e-9
    phi_center = phi_avg[center_mask].mean()
    phi_left = phi_avg[left_mask].mean()
    phi_right = phi_avg[right_mask].mean()
    assert phi_center > phi_left
    assert phi_center > phi_right


# ---- 4. 位相別プロファイル計測 (prompts/75) -------------------------------------


def test_timing_phases():
    """MCC 電離ありの小ケースを実行し、位相別タイマーが妥当な値を返すこと。

    - 全キーが 0 以上
    - 各フェーズの合計 (= done メッセージの total) が run_batch の実測壁時計時間と
      ±計測誤差で一致する (step() 内の全時間が漏れなく・二重計上なくどこかの
      フェーズに割り当てられていることの確認)。walk コスト診断 (prompts/88、
      WALK_DIAG_KEYS) は秒数ではないためこの合計からは除外する
    - 少なくとも solve/walk/deposit/mcc は正の時間を計上する
      (このケースは MCC 有効なので mcc も確実に発火する)
    - walk_cells_est_e/i・h_mean_m・h_min_m (prompts/88) が正で、h は妥当な m オーダー
    """
    n_steps = 20
    project = Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [0.02, 0], [0.02, 0.01], [0, 0.01]]},
                "boundaries": [
                    {"edges": [3], "voltage": 0.0},
                    {"edges": [1], "voltage": 0.0},
                ],
            },
            "mesh": {"size": 1.2e-3},
            "pic": {
                "initial_plasma": {
                    "density": DENSITY,
                    "te_ev": 5.0,
                    "ti_ev": 0.03,
                    "ion_mass_amu": 40.0,
                    "seed": 3,
                },
                "n_macro": 1500,
                "dt": 5e-10,
                "n_steps": n_steps,
                "frame_every": n_steps,
                "mcc": {
                    "gas": {"name": "Ar", "pressure_pa": 100.0, "temperature_k": 300.0},
                    "electron_processes": [
                        {
                            "kind": "ionization",
                            "label": "synthetic ionization",
                            "threshold_ev": 5.0,
                            "energy_ev": [5.0, 5.0 + 1e-6, 1000.0],
                            "sigma_m2": [0.0, 2.0e-20, 2.0e-20],
                        }
                    ],
                    "ion_processes": [],
                    "seed": 11,
                },
            },
        }
    )
    sim = PicSimulation(project)
    t0 = time.perf_counter()
    sim.run_batch()
    wall = time.perf_counter() - t0
    timing = sim.timing

    for key in ("solve", "gather_push", "walk", "deposit", "mcc", "other"):
        assert key in timing
        assert timing[key] >= 0.0

    # walk コスト診断 (prompts/88) は秒数ではないので、時間の合計からは除外する
    total = sum(v for k, v in timing.items() if k not in WALK_DIAG_KEYS)
    assert total > 0.0
    assert timing["solve"] > 0.0
    assert timing["walk"] > 0.0
    assert timing["deposit"] > 0.0
    assert timing["mcc"] > 0.0
    # run_batch には run_batch 自身のループ管理・フレーム構築・averaged_fields() 等
    # step() の外側のわずかなコストも含まれるため、total は wall 以下かつ大部分を
    # 占めるはず (下限は緩めに 50%、上限は計測誤差を見込んで少し余裕を持たせる)
    assert 0.5 * wall <= total <= wall + 0.05

    # walk コスト診断 (prompts/88): 種ごとの平均横断セル数の推定と代表セル寸法
    assert timing["walk_cells_est_e"] > 0.0
    assert timing["walk_cells_est_i"] > 0.0
    assert 0.0 < timing["h_min_m"] <= timing["h_mean_m"]
    # メッシュ寸法 (1.2e-3 m 刻み) から妥当な m オーダーであること
    assert 1e-6 < timing["h_mean_m"] < 1.0

    # continue 後は区間分のみを返す (前区間の値を引きずらない)
    sim.prepare_continue(n_steps)
    for key in timing:
        assert sim.timing[key] == 0.0
    sim.run_batch()
    total2 = sum(v for k, v in sim.timing.items() if k not in WALK_DIAG_KEYS)
    assert total2 > 0.0
    # continue 後も walk コスト診断キーは再設定される
    assert sim.timing["walk_cells_est_e"] > 0.0
    assert sim.timing["walk_cells_est_i"] > 0.0
    assert sim.timing["h_mean_m"] > 0.0


# ---- 5. ライブモニタの要素密度 n_e/n_i (prompts/81) -----------------------------


def test_frame_density_fields():
    """frame の n_e/n_i (種ごとの要素密度) が要素数長・有限・非負であること。

    さらに一様初期プラズマの1フレーム目 (1ステップ実行後) で、
    Σ n・V_elem (要素密度×要素体積の総和) がその時点のマクロ粒子総重みと
    相対誤差 1e-10 以下で一致すること (n[elem] = Σ_p w_p / V_elem の定義から
    厳密に成り立つはずの恒等式を、実装がその通り計算していることの確認)。
    """
    project = _oscillation_project(n_steps=1)
    project.pic.frame_every = 1
    sim = PicSimulation(project)
    _, frames = sim.run_batch()
    assert len(frames) == 1
    frame = frames[0]

    n_e = np.asarray(frame["n_e"])
    n_i = np.asarray(frame["n_i"])
    assert n_e.shape == (len(sim.tris),)
    assert n_i.shape == (len(sim.tris),)
    assert np.all(np.isfinite(n_e))
    assert np.all(np.isfinite(n_i))
    assert np.all(n_e >= 0.0)
    assert np.all(n_i >= 0.0)

    for name, n_arr in (("electron", n_e), ("ion", n_i)):
        sp = sim.species[name]
        total_w = float(sp.w.sum())
        total_from_density = float(np.sum(n_arr * sim.elem_vol))
        rel_err = abs(total_from_density - total_w) / total_w
        assert rel_err < 1e-10, (
            f"{name}: Σn·V_elem={total_from_density:.6e} vs Σw={total_w:.6e} "
            f"(rel_err={rel_err:.3e})"
        )


def test_run_batch_can_stream_without_retaining_frames():
    """WebSocket経路はcallbackへ送ったフレームを戻り値に蓄積しない。"""
    project = _oscillation_project(n_steps=3)
    project.pic.frame_every = 1
    sim = PicSimulation(project)
    streamed: list[dict] = []

    _history, frames = sim.run_batch(
        callback=streamed.append,
        store_frames=False,
    )

    assert frames == []
    assert [frame["step"] for frame in streamed] == [1, 2, 3]


def test_run_batch_releases_spare_push_walk_buffers():
    """完了後は現在の粒子状態ではないping-pongバッファを保持しない。"""
    sim = PicSimulation(_oscillation_project(n_steps=3))
    sim.run_batch()

    pool = sim._push_walk_buffers.get("electron", [])
    assert len(pool) <= 1
    if pool:
        buf = pool[0]
        el = sim.species["electron"]
        assert np.shares_memory(buf[0], el.v)
        assert np.shares_memory(buf[1], el.x)
        assert np.shares_memory(buf[3], el.elem)
        assert np.shares_memory(buf[7], el.bary)
