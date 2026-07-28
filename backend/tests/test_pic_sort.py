"""粒子のセル順ソート (空間局所性の改善、高速化③、prompts/84) のテスト。

1. 並べ替えの整合性: 人工的な粒子集合に _sort_particles_by_cell を適用し、
   elem が非減少になること、x/v/w/elem が同じ置換で対応していること
   (stable ソートなので argsort(kind="stable") の結果と完全一致するはず)。
2. 不動種 (immobile_ions) はソート対象外で並びが変化しないこと。
3. 実行時の健全性: SORT_EVERY をまたぐステップ数を実行しても run_batch が
   正常に完了し、最終ステップ後の所属要素が非減少であること
   (このテストのフィクスチャは要素数が SORT_MIN_ELEMS 未満なので、ガードを
   跨いで検証できるよう SORT_MIN_ELEMS を一時的に無効化する)。
4. 物理不変性: MCC 電離ありの小ケースで、ソートあり (既定) と
   _sort_particles_by_cell を無効化したソートなし実行の ke_e 最終値が
   数%以内で一致すること (ビット一致は要求しない。統計的に同等。同上の理由で
   SORT_MIN_ELEMS を一時的に無効化する)。
5. メッシュ要素数ガード: 要素数が SORT_MIN_ELEMS 未満のメッシュでは
   run_batch を通しても _sort_particles_by_cell が一度も呼ばれないこと
   (実測に基づく退行回避ガード。粗いメッシュでは per-element データが
   キャッシュに収まり局所性改善が出ず、ソートコストだけが乗って退行するため)。
   閾値を下げればガードが外れて呼ばれることも合わせて確認する。
"""

import numpy as np

from es_sim.pic import PicSimulation, SORT_EVERY, SORT_MIN_ELEMS
from es_sim.schema import Project

DENSITY = 1.0e14  # [m^-3]


def _oscillation_project(n_steps: int, immobile_ions: bool = False) -> Project:
    """test_pic.py と同じ最小構成 (プラズマ振動テストの流用)。

    既定の mesh.size (8e-4) では要素数が SORT_MIN_ELEMS (4000) を大きく下回る
    (粗いメッシュ)。ガード込みの run_batch 経由テストではこの前提を利用する。
    """
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
                    "te_ev": 3.0,
                    "ti_ev": 0.05,
                    "ion_mass_amu": 40.0,
                    "immobile_ions": immobile_ions,
                    "seed": 1,
                },
                "n_macro": 800,
                "dt": None,
                "n_steps": n_steps,
                "frame_every": n_steps,
            },
        }
    )


def _ionizing_project(n_steps: int) -> Project:
    """電離が活発な合成ケース (test_pic_merge.py の _ionizing_project と同構成)。

    MCC が粒子順に乱数を消費するため、セル順ソートあり/なしの統計的同等性
    (ビット単位では一致しないが最終値は近い) の検証に使う。こちらも粗いメッシュ
    (mesh.size=1.2e-3) なので SORT_MIN_ELEMS ガードの対象になる。
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
                "reflect_edges": [0, 1, 2, 3],
                "mcc": {
                    "gas": {"name": "Ar", "pressure_pa": 100.0, "temperature_k": 300.0},
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
            },
        }
    )


# ---- 1. 並べ替えの整合性 --------------------------------------------------------
# (_sort_particles_by_cell を直接呼ぶため、SORT_MIN_ELEMS ガードの影響を受けない)


def test_sort_orders_by_element_and_preserves_correspondence():
    sim = PicSimulation(_oscillation_project(n_steps=1))
    sp = sim.species["electron"]
    rng = np.random.default_rng(0)
    n = 400
    m = len(sim.tris)
    elem = rng.integers(0, m, size=n).astype(np.int64)
    x = rng.normal(size=(n, 2))
    v = rng.normal(size=(n, 3))
    w = rng.uniform(1.0, 2.0, size=n)
    sp.x, sp.v, sp.w, sp.elem = x.copy(), v.copy(), w.copy(), elem.copy()
    sp.bary = None
    sp.nidx = None

    sim._sort_particles_by_cell()

    # elem は非減少 (セル順)
    assert np.all(np.diff(sp.elem) >= 0)
    assert len(sp.x) == len(sp.v) == len(sp.w) == len(sp.elem) == n

    # stable な argsort と完全に同じ置換であること (対応関係の保存を厳密に確認)
    order = np.argsort(elem, kind="stable")
    assert np.array_equal(sp.elem, elem[order])
    assert np.array_equal(sp.x, x[order])
    assert np.array_equal(sp.v, v[order])
    assert np.array_equal(sp.w, w[order])


# ---- 2. 不動種はスキップ ---------------------------------------------------------
# (同上、_sort_particles_by_cell を直接呼ぶ)


def test_sort_skips_immobile_species():
    sim = PicSimulation(_oscillation_project(n_steps=1, immobile_ions=True))
    io = sim.species["ion"]
    assert io.mobile is False

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(io.elem))
    io.elem = io.elem[perm]
    io.x = io.x[perm]
    io.v = io.v[perm]
    io.w = io.w[perm]
    scrambled_elem = io.elem.copy()

    sim._sort_particles_by_cell()

    assert np.array_equal(io.elem, scrambled_elem)  # 不動種は並びが変わらない


# ---- 3. 実行時の健全性 (run_batch 経由、ガードを無効化して検証) ---------------------


def test_sort_runs_across_multiple_intervals_without_error(monkeypatch):
    # このフィクスチャは要素数が SORT_MIN_ELEMS 未満の粗いメッシュなので、
    # run_batch 経由での定期発火を検証するには閾値を一時的に外す必要がある
    # (ガード自体は別テストで独立に検証する)
    monkeypatch.setattr("es_sim.pic.SORT_MIN_ELEMS", 0)
    # ちょうど SORT_EVERY の倍数で終える: 最終ステップがソート直後になるため、
    # 所属要素が非減少であることをそのまま検証できる
    n_steps = 2 * SORT_EVERY
    sim = PicSimulation(_oscillation_project(n_steps))
    history, _ = sim.run_batch()
    assert len(history["t"]) == n_steps
    assert sim.step_count % SORT_EVERY == 0
    el = sim.species["electron"]
    assert np.all(np.diff(el.elem) >= 0)  # 最後のソート直後、非減少のまま


# ---- 4. 物理不変性 (統計的同等性、ガードを無効化して検証) --------------------------


def test_sort_preserves_statistics_with_ionization(monkeypatch):
    """MCC 電離ありの小ケースで、ソートあり (ガード無効化) と無効化した実行の
    ke_e 最終値・電離イベント累計が数%以内で一致すること (ビット一致は要求しない)。"""
    monkeypatch.setattr("es_sim.pic.SORT_MIN_ELEMS", 0)  # 粗いメッシュでも発火させる
    n_steps = 3 * SORT_EVERY

    sim_sort = PicSimulation(_ionizing_project(n_steps))
    hist_sort, _ = sim_sort.run_batch()

    sim_nosort = PicSimulation(_ionizing_project(n_steps))
    sim_nosort._sort_particles_by_cell = lambda: None  # ソートを無効化した対照実行
    hist_nosort, _ = sim_nosort.run_batch()

    ke_sort = np.asarray(hist_sort["ke_e"])[-50:].mean()
    ke_nosort = np.asarray(hist_nosort["ke_e"])[-50:].mean()
    rel = abs(ke_sort - ke_nosort) / abs(ke_nosort)
    assert rel < 0.05, f"ke_e の統計的乖離が大きすぎます (rel={rel:.4f})"

    n_e_sort = hist_sort["n_e"][-1]
    n_e_nosort = hist_nosort["n_e"][-1]
    assert abs(n_e_sort - n_e_nosort) / n_e_nosort < 0.10


# ---- 5. メッシュ要素数ガード (prompts/84 追加ガード) --------------------------------


def test_sort_guard_skips_below_threshold(monkeypatch):
    """要素数が SORT_MIN_ELEMS 未満なら、run_batch を通しても
    _sort_particles_by_cell が一度も呼ばれないこと (粗いメッシュでの退行回避)。"""
    n_steps = 2 * SORT_EVERY
    sim = PicSimulation(_oscillation_project(n_steps))
    assert len(sim.tris) < SORT_MIN_ELEMS  # このフィクスチャの前提を明示しておく

    calls = []
    monkeypatch.setattr(sim, "_sort_particles_by_cell", lambda: calls.append(1))
    sim.run_batch()

    assert calls == []


def test_sort_guard_allows_above_threshold(monkeypatch):
    """閾値を要素数以下まで下げれば、SORT_EVERY ごとに呼ばれること。"""
    n_steps = 2 * SORT_EVERY
    sim = PicSimulation(_oscillation_project(n_steps))
    monkeypatch.setattr("es_sim.pic.SORT_MIN_ELEMS", len(sim.tris))  # ガードを無効化

    calls = []
    monkeypatch.setattr(sim, "_sort_particles_by_cell", lambda: calls.append(1))
    sim.run_batch()

    assert len(calls) == 2  # n_steps=2*SORT_EVERY → ステップ SORT_EVERY, 2*SORT_EVERY で発火
