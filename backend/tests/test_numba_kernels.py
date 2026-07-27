"""numba カーネル (prompts/76) の等価性・フォールバックのテスト。

- walk / deposit / gather+push / MCC候補・選択・散乱について、numpy 実装と numba 実装が
  ランダム入力に対して完全に一致すること (np.array_equal、ビット単位) を確認する。
  実行環境に numba が無ければこのファイルの等価性テストはスキップする
  (フォールバック自体は下の test_numba_fallback_smoke で monkeypatch を使って
  検証する)。
- HAVE_NUMBA を monkeypatch で False にして、numba が無い環境と同じ経路
  (_walk_step_numpy 等) が実際に選ばれることをスモークテストする。
"""

from __future__ import annotations

import numpy as np
import pytest

from es_sim import _numba_kernels as nk
from es_sim import particles as P
from es_sim.mcc import GasField, MccModel
from es_sim.meshing import generate_mesh
from es_sim.schema import MccSettings, Project

requires_numba = pytest.mark.skipif(
    not nk.HAVE_NUMBA, reason="numba がインストールされていない環境ではスキップ"
)


def _demo_mesh():
    proj = Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [0.02, 0], [0.02, 0.01], [0, 0.01]]},
                "boundaries": [
                    {"edges": [3], "type": "dirichlet", "voltage": 0.0},
                    {"edges": [1], "type": "dirichlet", "voltage": 0.0},
                ],
            },
            "mesh": {"size": 8e-4},
        }
    )
    return generate_mesh(proj)


@requires_numba
def test_walk_numpy_numba_equivalence():
    """ランダムな粒子位置・移動で numpy 版と numba 版の walk が完全一致する。

    領域外へ大きく移動する粒子も混ぜて壁吸収 (absorbed) 経路も踏ませる。
    """
    mesh = _demo_mesh()
    coeffs = P._barycentric_coeffs(mesh.nodes, mesh.triangles)
    adjacency = P._adjacency(mesh.triangles)
    packed = P._pack_coeffs(coeffs)

    rng = np.random.default_rng(42)
    n = 20000
    x0 = rng.uniform(-0.005, 0.025, size=(n, 2))
    elem0 = P._locate_initial(coeffs, np.clip(x0, [0.0, 0.0], [0.02, 0.01]))
    x_new = x0 + rng.normal(0.0, 0.003, size=(n, 2))

    l_np = np.empty((n, 3))
    elem_np, abs_np, be_np, bl_np = P._walk_step_numpy(
        coeffs, adjacency, elem0, x_new, l_np, packed=packed
    )
    l_nb = np.empty((n, 3))
    elem_nb, abs_nb, be_nb, bl_nb = nk.walk_step(
        coeffs, adjacency, elem0, x_new, l_nb, packed=packed
    )

    assert np.array_equal(elem_np, elem_nb)
    assert np.array_equal(abs_np, abs_nb)
    assert np.array_equal(be_np, be_nb)
    assert np.array_equal(bl_np, bl_nb)
    # absorbed 行は両実装とも「未定義」の契約なので、非 absorbed 行だけ比較する
    keep = ~abs_np
    assert np.array_equal(l_np[keep], l_nb[keep])
    # 壁吸収・反射双方の経路を実際に踏んでいることを確認 (テストの意味があることの担保)
    assert 0 < abs_np.sum() < n


@requires_numba
def test_deposit_numpy_numba_equivalence():
    """nidx版・所属要素直接版がnp.bincountと完全一致する。"""
    rng = np.random.default_rng(1)
    n_nodes = 500
    n_elements = 900
    n = 30000
    tris = rng.integers(0, n_nodes, size=(n_elements, 3), dtype=np.int64)
    elem = rng.integers(0, n_elements, size=n, dtype=np.int64)
    nidx = tris[elem]
    bary = rng.uniform(-0.2, 1.2, size=(n, 3))
    w = rng.uniform(1e10, 1e14, size=n)
    q = -1.602176634e-19

    contrib = (q * w)[:, None] * bary
    f_np = np.bincount(nidx.ravel(), weights=contrib.ravel(), minlength=n_nodes)
    f_nb = nk.deposit(nidx, bary, q, w, n_nodes)
    f_direct = nk.deposit_from_elements(tris, elem, bary, q, w, n_nodes)
    assert np.array_equal(f_np, f_nb)
    assert np.array_equal(f_np, f_direct)


@requires_numba
def test_gather_push_numpy_numba_equivalence():
    """gather (E補間) + リープフロッグ push の融合カーネルが numpy 経路と完全一致する。"""
    rng = np.random.default_rng(2)
    m_elem = 400
    n = 30000
    exy = rng.normal(0.0, 1e5, size=(m_elem, 2))
    elem = rng.integers(0, m_elem, size=n).astype(np.int64)
    x = rng.uniform(0.0, 0.02, size=(n, 2))
    v = rng.normal(0.0, 1e5, size=(n, 3))
    q = -1.602176634e-19
    m = 9.1093837015e-31
    dt_sp = 3e-11

    e_at = exy[elem]
    v_new = v.copy()
    v_new[:, :2] += (q / m) * dt_sp * e_at
    vdot = (v[:, 0] * v_new[:, 0] + v[:, 1] * v_new[:, 1]) + v[:, 2] * v_new[:, 2]
    x_new = x + dt_sp * v_new[:, :2]

    v_new_nb, x_new_nb, vdot_nb = nk.gather_push(exy, elem, q, m, dt_sp, x, v)
    assert np.array_equal(v_new, v_new_nb)
    assert np.array_equal(x_new, x_new_nb)
    assert np.array_equal(vdot, vdot_nb)

    w = rng.uniform(1e10, 1e14, size=n)
    ke_np = 0.5 * m * float(np.sum(w * vdot))
    ke_nb = 0.5 * m * float(np.sum(w * vdot_nb))
    assert ke_np == ke_nb


@requires_numba
def test_fused_gather_push_walk_matches_separate_kernels():
    """融合カーネルが従来のNumba push→walkとビット単位で一致する。"""
    mesh = _demo_mesh()
    coeffs = P._barycentric_coeffs(mesh.nodes, mesh.triangles)
    adjacency = P._adjacency(mesh.triangles)
    packed = P._pack_coeffs(coeffs)

    rng = np.random.default_rng(2026)
    n = 30_000
    x = rng.uniform([0.0, 0.0], [0.02, 0.01], size=(n, 2))
    elem = P._locate_initial(coeffs, x)
    v = rng.normal(0.0, 2.0e6, size=(n, 3))
    exy = rng.normal(0.0, 1.0e3, size=(len(mesh.triangles), 2))
    q = -P.QE
    m = P.ME
    dt_sp = 2.0e-10

    v_sep, x_sep, vdot_sep = nk.gather_push(exy, elem, q, m, dt_sp, x, v)
    l_sep = np.empty((n, 3))
    e_sep, a_sep, be_sep, bl_sep = nk.walk_step(
        coeffs, adjacency, elem, x_sep, l_sep, packed
    )
    (
        v_fused,
        x_fused,
        vdot_fused,
        e_fused,
        a_fused,
        be_fused,
        bl_fused,
        l_fused,
    ) = nk.gather_push_walk(exy, packed, adjacency, elem, q, m, dt_sp, x, v)
    out = (
        np.empty((n + 17, 3)),
        np.empty((n + 17, 2)),
        np.empty(n + 17),
        np.empty(n + 17, dtype=np.int64),
        np.empty(n + 17, dtype=np.bool_),
        np.empty(n + 17, dtype=np.int64),
        np.empty(n + 17, dtype=np.int64),
        np.empty((n + 17, 3)),
    )
    buffered = nk.gather_push_walk(
        exy, packed, adjacency, elem, q, m, dt_sp, x, v, out=out
    )

    assert np.array_equal(v_sep, v_fused)
    assert np.array_equal(x_sep, x_fused)
    assert np.array_equal(vdot_sep, vdot_fused)
    assert np.array_equal(e_sep, e_fused)
    assert np.array_equal(a_sep, a_fused)
    assert np.array_equal(be_sep, be_fused)
    assert np.array_equal(bl_sep, bl_fused)
    assert np.array_equal(l_sep[~a_sep], l_fused[~a_fused])
    for expected, actual in zip(
        (v_fused, x_fused, vdot_fused, e_fused, a_fused, be_fused, bl_fused),
        buffered[:7],
    ):
        assert np.array_equal(expected, actual)
    assert np.array_equal(l_fused[~a_fused], buffered[7][~buffered[4]])
    assert 0 < int(a_sep.sum()) < n


def _mcc_settings() -> MccSettings:
    return MccSettings.model_validate(
        {
            "gas": {"name": "synthetic", "pressure_pa": 50.0, "temperature_k": 300.0},
            "electron_processes": [
                {
                    "kind": "elastic",
                    "label": "elastic",
                    "mass_ratio": 1.0e-5,
                    "energy_ev": [0.0, 1.0, 20.0, 1000.0],
                    "sigma_m2": [1.0e-19, 2.0e-19, 8.0e-20, 5.0e-20],
                },
                {
                    "kind": "excitation",
                    "label": "excitation",
                    "threshold_ev": 11.5,
                    "energy_ev": [11.5, 20.0, 1000.0],
                    "sigma_m2": [0.0, 4.0e-20, 1.0e-20],
                },
                {
                    "kind": "ionization",
                    "label": "ionization",
                    "threshold_ev": 15.8,
                    "energy_ev": [15.8, 30.0, 1000.0],
                    "sigma_m2": [0.0, 3.0e-20, 2.0e-20],
                },
            ],
            "ion_processes": [
                {
                    "kind": "isotropic",
                    "label": "ion elastic",
                    "energy_ev": [0.0, 1.0, 1000.0],
                    "sigma_m2": [8.0e-19, 6.0e-19, 2.0e-19],
                },
                {
                    "kind": "backscat",
                    "label": "charge exchange",
                    "energy_ev": [0.0, 1.0, 1000.0],
                    "sigma_m2": [5.0e-19, 4.0e-19, 1.0e-19],
                },
            ],
            "seed": 91,
        }
    )


@requires_numba
def test_mcc_numpy_numba_equivalence(monkeypatch):
    """MCC全電子衝突が速度・生成粒子・後続RNG状態まで完全一致する。"""
    rng = np.random.default_rng(123)
    n = 30_000
    x = rng.uniform(0.0, 0.02, size=(n, 2))
    # 閾値上下を十分含む電子エネルギーになる速度分布
    v = rng.normal(0.0, 2.0e6, size=(n, 3))
    w = rng.uniform(1.0e7, 1.0e9, size=n)
    elem = rng.integers(0, 100, size=n, dtype=np.int64)
    gas_field = GasField(
        n_g=np.linspace(2.0e21, 1.2e22, 100),
        t_g=np.linspace(300.0, 600.0, 100),
        u_g=np.column_stack(
            [np.linspace(-20.0, 20.0, 100), np.linspace(10.0, -10.0, 100)]
        ),
    )

    model_nb = MccModel(_mcc_settings(), 40.0 * P.MP, gas_field)
    v_nb = v.copy()
    result_nb = model_nb.collide_electrons(x, v_nb, w, elem, 1.0e-10)
    next_nb = model_nb.rng.random(16)

    monkeypatch.setattr(nk, "HAVE_NUMBA", False)
    model_np = MccModel(_mcc_settings(), 40.0 * P.MP, gas_field)
    v_np = v.copy()
    result_np = model_np.collide_electrons(x, v_np, w, elem, 1.0e-10)
    next_np = model_np.rng.random(16)

    assert np.array_equal(v_nb, v_np)
    assert result_nb.n_coll == result_np.n_coll
    assert result_nb.n_ionization == result_np.n_ionization
    for name in ("new_x", "new_elem", "new_w", "new_v_e", "new_v_i"):
        assert np.array_equal(getattr(result_nb, name), getattr(result_np, name)), name
    assert np.array_equal(next_nb, next_np)


@requires_numba
def test_mcc_ion_numpy_numba_equivalence(monkeypatch):
    """イオン散乱も速度・衝突数・後続RNG状態がNumPy経路と完全一致する。"""
    rng = np.random.default_rng(456)
    n = 30_000
    v = rng.normal(0.0, 1.5e3, size=(n, 3))
    m_ion = 40.0 * P.MP

    model_nb = MccModel(_mcc_settings(), m_ion)
    v_nb = v.copy()
    n_nb = model_nb.collide_ions(v_nb, 1.0e-8)
    next_nb = model_nb.rng.random(16)

    monkeypatch.setattr(nk, "HAVE_NUMBA", False)
    model_np = MccModel(_mcc_settings(), m_ion)
    v_np = v.copy()
    n_np = model_np.collide_ions(v_np, 1.0e-8)
    next_np = model_np.rng.random(16)

    assert n_nb == n_np
    assert n_nb > 0
    assert np.array_equal(v_nb, v_np)
    assert np.array_equal(next_nb, next_np)


def test_numba_fallback_smoke(monkeypatch):
    """HAVE_NUMBA=False を強制すると、_walk_step が numpy 実装へフォールバックすること。

    numba のインストール有無に関わらず、フォールバック分岐そのものが正しく
    動作することを monkeypatch で確認する (仕様書 prompts/76 のテスト項目3)。
    """
    monkeypatch.setattr(nk, "HAVE_NUMBA", False)
    mesh = _demo_mesh()
    coeffs = P._barycentric_coeffs(mesh.nodes, mesh.triangles)
    adjacency = P._adjacency(mesh.triangles)
    packed = P._pack_coeffs(coeffs)

    rng = np.random.default_rng(3)
    n = 200
    x0 = rng.uniform(0.0, 0.02, size=(n, 2))
    elem0 = P._locate_initial(coeffs, x0)
    x_new = x0 + rng.normal(0.0, 0.001, size=(n, 2))

    l_out = np.empty((n, 3))
    elem, absorbed, b_elem, b_loc = P._walk_step(
        coeffs, adjacency, elem0, x_new, l_out, packed=packed
    )
    l_ref = np.empty((n, 3))
    elem_ref, abs_ref, be_ref, bl_ref = P._walk_step_numpy(
        coeffs, adjacency, elem0, x_new, l_ref, packed=packed
    )
    assert np.array_equal(elem, elem_ref)
    assert np.array_equal(absorbed, abs_ref)
