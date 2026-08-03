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
    # 非連続出力は従来の一時バッファ経路へフォールバックする。
    l_strided_storage = np.empty((n, 6))
    l_strided = l_strided_storage[:, ::2]
    nk.walk_step(coeffs, adjacency, elem0, x_new, l_strided, packed=packed)
    assert np.array_equal(l_np[keep], l_strided[keep])
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
@pytest.mark.parametrize(
    "removed",
    [
        np.zeros(31, dtype=np.bool_),
        np.array([(i % 3) == 1 for i in range(31)], dtype=np.bool_),
        np.ones(31, dtype=np.bool_),
    ],
)
def test_compact_particle_state_matches_boolean_indexing(removed):
    """JIT圧縮が粒子順を保ち、従来のブール抽出と完全一致する。"""
    rng = np.random.default_rng(761)
    n = len(removed)
    x = rng.normal(size=(n, 2))
    v = rng.normal(size=(n, 3))
    w = rng.uniform(1e5, 1e9, size=n)
    elem = rng.integers(0, 100, size=n, dtype=np.int64)
    bary = rng.normal(size=(n, 3))
    keep = ~removed
    expected = tuple(arr[keep] for arr in (x, v, w, elem, bary))

    x_out = x.copy()
    v_out = v.copy()
    elem_out = elem.copy()
    bary_out = bary.copy()
    w_out = np.empty(n + 7)
    n_keep = nk.compact_particle_state(
        x_out, v_out, w, elem_out, bary_out, removed, w_out
    )
    actual = (
        x_out[:n_keep],
        v_out[:n_keep],
        w_out[:n_keep],
        elem_out[:n_keep],
        bary_out[:n_keep],
    )

    assert n_keep == int(keep.sum())
    for expected_array, actual_array in zip(expected, actual):
        assert np.array_equal(expected_array, actual_array)


@requires_numba
def test_rz_pic_jit_compaction_matches_numpy_fallback(monkeypatch):
    """軸対称PICのJIT圧縮後状態が従来フォールバックと完全一致する。"""
    project = Project.model_validate(
        {
            "coord": "rz",
            "geometry": {
                "domain": {
                    "polygon": [[0, 0], [0.02, 0], [0.02, 0.01], [0, 0.01]]
                },
                "boundaries": [
                    {
                        "edges": [1, 2, 3],
                        "type": "dirichlet",
                        "voltage": 0.0,
                    }
                ],
            },
            "mesh": {"size": 8e-4},
            "pic": {
                "initial_plasma": {
                    "density": 1e14,
                    "te_ev": 10.0,
                    "ti_ev": 0.1,
                    "ion_mass_amu": 40.0,
                    "seed": 19,
                },
                "n_macro": 1000,
                "dt": 2e-10,
                "n_steps": 20,
                "frame_every": 1000,
                "threads": 1,
            },
        }
    )

    from es_sim.pic import PicSimulation

    sim_jit = PicSimulation(project)
    for _ in range(20):
        sim_jit.step()

    monkeypatch.setattr(nk, "HAVE_NUMBA", False)
    sim_numpy = PicSimulation(project)
    for _ in range(20):
        sim_numpy.step()

    assert sim_jit.species["electron"].wall_absorbed > 0
    for name in sim_jit.species:
        actual = sim_jit.species[name]
        expected = sim_numpy.species[name]
        assert actual.wall_absorbed == expected.wall_absorbed
        for attr in ("x", "v", "w", "elem", "bary"):
            assert np.array_equal(getattr(actual, attr), getattr(expected, attr)), (
                name,
                attr,
            )
    for key in sim_jit.history:
        assert np.array_equal(
            np.asarray(sim_jit.history[key]),
            np.asarray(sim_numpy.history[key]),
        ), key


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
def test_gather_push_boris_matches_numpy():
    """融合Borisカーネルが従来のNumPy演算順と一致する。"""
    rng = np.random.default_rng(511)
    n_elements = 97
    n = 30_000
    exy = rng.normal(0.0, 1.0e3, size=(n_elements, 2))
    elem = rng.integers(0, n_elements, size=n, dtype=np.int64)
    x = rng.uniform([0.0, 0.0], [0.02, 0.01], size=(n, 2))
    v = rng.normal(0.0, 2.0e6, size=(n, 3))
    q = -P.QE
    m = P.ME
    dt_sp = 2.0e-11
    boris_rt = P._boris_matrix(q, m, dt_sp, np.array([0.2, -0.1, 0.3])).T

    e_at = exy[elem]
    v_ref = v.copy()
    half = (q / m) * (0.5 * dt_sp)
    v_ref[:, :2] += half * e_at
    v_ref = v_ref @ boris_rt
    v_ref[:, :2] += half * e_at
    vdot_ref = (
        v[:, 0] * v_ref[:, 0] + v[:, 1] * v_ref[:, 1]
    ) + v[:, 2] * v_ref[:, 2]
    x_ref = x + dt_sp * v_ref[:, :2]

    v_jit, x_jit, vdot_jit = nk.gather_push_boris(
        exy, elem, q, m, dt_sp, boris_rt, x, v
    )
    # NumPy BLASの3x3行列積とJITスカラー積和には最下位ビットの丸め差がある。
    np.testing.assert_allclose(v_ref, v_jit, rtol=5e-12, atol=2e-9)
    np.testing.assert_allclose(x_ref, x_jit, rtol=1e-14, atol=4e-18)
    np.testing.assert_allclose(vdot_ref, vdot_jit, rtol=1e-14, atol=2e-2)
    w = rng.uniform(1e10, 1e14, size=n)
    ke_ref = 0.5 * m * float(np.sum(w * vdot_ref))
    ke_jit = 0.5 * m * float(np.sum(w * vdot_jit))
    assert ke_jit == pytest.approx(ke_ref, rel=2e-15)


@requires_numba
def test_fused_boris_gather_push_walk_matches_separate_kernels():
    """融合Boris+walkが個別JITカーネルとビット単位で一致する。"""
    mesh = _demo_mesh()
    coeffs = P._barycentric_coeffs(mesh.nodes, mesh.triangles)
    adjacency = P._adjacency(mesh.triangles)
    packed = P._pack_coeffs(coeffs)

    rng = np.random.default_rng(512)
    n = 30_000
    x = rng.uniform([0.0, 0.0], [0.02, 0.01], size=(n, 2))
    elem = P._locate_initial(coeffs, x)
    v = rng.normal(0.0, 2.0e6, size=(n, 3))
    exy = rng.normal(0.0, 1.0e3, size=(len(mesh.triangles), 2))
    q = -P.QE
    m = P.ME
    dt_sp = 2.0e-10
    boris_rt = P._boris_matrix(q, m, dt_sp, np.array([0.0, 0.0, 0.01])).T

    v_sep, x_sep, vdot_sep = nk.gather_push_boris(
        exy, elem, q, m, dt_sp, boris_rt, x, v
    )
    l_sep = np.empty((n, 3))
    e_sep, a_sep, be_sep, bl_sep = nk.walk_step(
        coeffs, adjacency, elem, x_sep, l_sep, packed
    )
    fused = nk.gather_push_walk_boris(
        exy, packed, adjacency, elem, q, m, dt_sp, boris_rt, x, v
    )
    v_fused, x_fused, vdot_fused, e_fused, a_fused, be_fused, bl_fused, l_fused = fused
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
    buffered = nk.gather_push_walk_boris(
        exy, packed, adjacency, elem, q, m, dt_sp, boris_rt, x, v, out=out
    )

    assert np.array_equal(v_sep, v_fused)
    assert np.array_equal(x_sep, x_fused)
    assert np.array_equal(vdot_sep, vdot_fused)
    assert np.array_equal(e_sep, e_fused)
    assert np.array_equal(a_sep, a_fused)
    assert np.array_equal(be_sep[a_sep], be_fused[a_fused])
    assert np.array_equal(bl_sep[a_sep], bl_fused[a_fused])
    assert np.array_equal(l_sep[~a_sep], l_fused[~a_fused])
    for expected, actual in zip(
        (v_fused, x_fused, vdot_fused, e_fused, a_fused),
        buffered[:5],
    ):
        assert np.array_equal(expected, actual)
    assert np.array_equal(be_fused[a_fused], buffered[5][buffered[4]])
    assert np.array_equal(bl_fused[a_fused], buffered[6][buffered[4]])
    assert np.array_equal(l_fused[~a_fused], buffered[7][~buffered[4]])
    assert 0 < int(a_sep.sum()) < n


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
    assert np.array_equal(be_sep[a_sep], be_fused[a_fused])
    assert np.array_equal(bl_sep[a_sep], bl_fused[a_fused])
    assert np.array_equal(l_sep[~a_sep], l_fused[~a_fused])
    for expected, actual in zip(
        (v_fused, x_fused, vdot_fused, e_fused, a_fused),
        buffered[:5],
    ):
        assert np.array_equal(expected, actual)
    assert np.array_equal(be_fused[a_fused], buffered[5][buffered[4]])
    assert np.array_equal(bl_fused[a_fused], buffered[6][buffered[4]])
    assert np.array_equal(l_fused[~a_fused], buffered[7][~buffered[4]])
    assert 0 < int(a_sep.sum()) < n


@requires_numba
@pytest.mark.parametrize("ridx", [0, 1])
def test_fused_rz_gather_push_walk_matches_numpy(ridx):
    """軸対称融合カーネルがrz/rz_x0の従来演算順とビット単位で一致する。"""
    mesh = _demo_mesh()
    coeffs = P._barycentric_coeffs(mesh.nodes, mesh.triangles)
    adjacency = P._adjacency(mesh.triangles)
    packed = P._pack_coeffs(coeffs)

    rng = np.random.default_rng(470 + ridx)
    n = 30_000
    x = rng.uniform([0.0, 0.0], [0.02, 0.01], size=(n, 2))
    elem = P._locate_initial(coeffs, x)
    v = rng.normal(0.0, 4.0e5, size=(n, 3))
    v[:, 2] *= 0.125
    exy = rng.normal(0.0, 1.0e3, size=(len(mesh.triangles), 2))
    q = -P.QE
    m = P.ME
    dt_sp = 2.0e-10

    # PicSimulation.step の従来numpy軸対称経路と演算順を揃えた参照値。
    e_at = exy[elem]
    v_ref = v.copy()
    a_rz = (q / m) * e_at
    r_cur = np.maximum(x[:, ridx], 1e-30)
    ang_l = x[:, ridx] * v[:, 2]
    a_rz[:, ridx] += v[:, 2] ** 2 / r_cur
    v_ref[:, :2] += dt_sp * a_rz
    vdot_ref = (
        v[:, 0] * v_ref[:, 0] + v[:, 1] * v_ref[:, 1]
    ) + v[:, 2] * v_ref[:, 2]
    x_ref = x + dt_sp * v_ref[:, :2]
    cross = x_ref[:, ridx] < 0.0
    x_ref[cross, ridx] = -x_ref[cross, ridx]
    v_ref[cross, ridx] = -v_ref[cross, ridx]
    ang_l[cross] = -ang_l[cross]
    r_new = np.maximum(x_ref[:, ridx], 1e-30)
    v_ref[:, 2] = np.where(ang_l != 0.0, ang_l / r_new, 0.0)

    l_ref = np.empty((n, 3))
    e_ref, a_ref, be_ref, bl_ref = nk.walk_step(
        coeffs, adjacency, elem, x_ref, l_ref, packed
    )
    fused = nk.gather_push_walk_rz(
        exy, packed, adjacency, elem, q, m, dt_sp, ridx, x, v
    )
    v_fused, x_fused, vdot_fused, e_fused, a_fused, be_fused, bl_fused, l_fused = fused

    assert np.array_equal(v_ref, v_fused)
    assert np.array_equal(x_ref, x_fused)
    assert np.array_equal(vdot_ref, vdot_fused)
    assert np.array_equal(e_ref, e_fused)
    assert np.array_equal(a_ref, a_fused)
    # 境界要素・局所辺は吸収粒子だけで定義される。
    assert np.array_equal(be_ref[a_ref], be_fused[a_fused])
    assert np.array_equal(bl_ref[a_ref], bl_fused[a_fused])
    assert np.array_equal(l_ref[~a_ref], l_fused[~a_fused])
    assert 0 < int(cross.sum()) < n
    assert 0 < int(a_ref.sum()) < n


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


@requires_numba
def test_mcc_max_speed_squared_matches_numpy():
    """最大速度二乗の1パスJITがNumPyの積和・最大値とビット一致する。"""
    rng = np.random.default_rng(762)
    v = rng.normal(0.0, 2.0e6, size=(30_000, 3))
    expected = float(np.max(np.sum(v * v, axis=1)))
    actual = nk.mcc_max_speed_squared(v)
    assert actual == expected

    v[123, 1] = np.nan
    assert np.isnan(nk.mcc_max_speed_squared(v))


@requires_numba
def test_mcc_group_process_positions_matches_masks():
    """プロセス別JIT選別が閾値付きブールマスクの順序・件数と一致する。"""
    rng = np.random.default_rng(763)
    proc_idx = rng.integers(-1, 4, size=30_000, dtype=np.int64)
    energy = rng.uniform(0.0, 40.0, size=len(proc_idx))
    thresholds = np.array([0.0, 11.5, 15.8, 0.0])
    use_threshold = np.array([False, True, True, False])

    grouped, offsets = nk.mcc_group_process_positions(
        proc_idx, energy, thresholds, use_threshold
    )
    for j in range(len(thresholds)):
        mask = proc_idx == j
        if use_threshold[j]:
            mask &= energy >= thresholds[j]
        expected = np.nonzero(mask)[0]
        actual = grouped[offsets[j] : offsets[j + 1]]
        assert np.array_equal(expected, actual)


@requires_numba
def test_mcc_select_velocity_parallel_matches_serial():
    """大規模候補のprange経路が直列JITとビット単位で一致する。"""
    if nk.numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("Numbaの利用可能スレッドが1本のみ")

    rng = np.random.default_rng(764)
    n = 50_000
    n_cand = 40_000
    v = rng.normal(0.0, 2.0e6, size=(n, 3))
    cand = rng.choice(n, size=n_cand, replace=False).astype(np.int64)
    random_u = rng.random(n_cand)
    e_table = np.array([[0.0, 20.0, 1000.0], [15.8, 30.0, 1000.0]])
    s_table = np.array([[1e-19, 8e-20, 5e-20], [0.0, 3e-20, 2e-20]])
    lengths = np.array([3, 3], dtype=np.int64)
    args = (
        v,
        cand,
        P.ME,
        P.QE,
        4.0e10,
        1.0e22,
        None,
        None,
        e_table,
        s_table,
        lengths,
        random_u,
    )

    previous_threads = nk.numba.get_num_threads()
    try:
        nk.set_num_threads(1)
        serial = nk.mcc_select_velocity(*args)
        nk.set_num_threads(min(4, nk.numba.config.NUMBA_NUM_THREADS))
        parallel = nk.mcc_select_velocity(*args)
    finally:
        nk.set_num_threads(previous_threads)

    for expected, actual in zip(serial, parallel):
        assert np.array_equal(expected, actual)


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


# ---- 陰的流体ソルバー用の並列反復法 (csr_matvec_parallel / bicgstab、prompts/115) ------


def _random_diag_dominant_csr(n: int, seed: int):
    """対角優位なランダム CSR 行列 (fluid2d.py の M=V/dt+K を模した性質) を作る。"""
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    dense = rng.uniform(-1.0, 1.0, size=(n, n))
    dense[np.abs(dense) < 0.6] = 0.0  # 疎にする
    # 対角優位にする (BiCGSTAB がJacobi前処理だけで収束する条件、fluid2d.py と同じ性質)
    np.fill_diagonal(dense, np.abs(dense).sum(axis=1) + rng.uniform(1.0, 2.0, size=n))
    m = sp.csr_matrix(dense)
    return m


@requires_numba
def test_csr_matvec_parallel_matches_scipy():
    """csr_matvec_parallel (numba 行並列) が scipy の csr@x と一致する (数値的に)。"""
    m = _random_diag_dominant_csr(200, seed=1)
    rng = np.random.default_rng(2)
    x = rng.uniform(-1.0, 1.0, size=m.shape[0])
    out = nk.csr_matvec_parallel(m.indptr, m.indices, m.data, x)
    expected = m @ x
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-12)


@requires_numba
def test_csr_matvec_parallel_threads_bit_identical():
    """threads=1 と threads=2 で csr_matvec_parallel の出力がビット同一 (np.array_equal)。

    行 i の内積は担当スレッドが常に昇順 k で逐次積和し、他の行の値を読まないため、
    スレッドへの行の割り当て方 (スケジューリング) が変わっても各行の演算列は
    不変 — allclose ではなく array_equal で厳密に検証する (規約どおり)。
    """
    m = _random_diag_dominant_csr(500, seed=3)
    rng = np.random.default_rng(4)
    x = rng.uniform(-1.0, 1.0, size=m.shape[0])

    previous_threads = nk.numba.get_num_threads()
    try:
        nk.set_num_threads(1)
        out1 = nk.csr_matvec_parallel(m.indptr, m.indices, m.data, x)
        nk.set_num_threads(min(2, nk.numba.config.NUMBA_NUM_THREADS))
        out2 = nk.csr_matvec_parallel(m.indptr, m.indices, m.data, x)
    finally:
        nk.set_num_threads(previous_threads)

    assert np.array_equal(out1, out2)


def test_bicgstab_solves_small_diagonally_dominant_system():
    """既知解を持つ対角優位な系を Jacobi-BiCGSTAB が rtol=1e-10 まで収束させる。"""
    m = _random_diag_dominant_csr(150, seed=5).tocsr()
    rng = np.random.default_rng(6)
    x_true = rng.uniform(-1.0, 1.0, size=m.shape[0])
    b = m @ x_true
    diag_inv = 1.0 / m.diagonal()
    x0 = np.zeros_like(b)

    x, n_iter, converged = nk.bicgstab(
        m.indptr, m.indices, m.data, b, x0, rtol=1e-10, atol=1e-300, max_iter=500, diag_inv=diag_inv,
    )
    assert converged
    assert n_iter > 0
    np.testing.assert_allclose(x, x_true, rtol=1e-6, atol=1e-8)


def test_bicgstab_reports_nonconvergence_when_max_iter_is_zero():
    """max_iter=0 なら (初期残差が許容誤差を超える限り) 必ず未収束を報告する。"""
    m = _random_diag_dominant_csr(50, seed=7).tocsr()
    rng = np.random.default_rng(8)
    x_true = rng.uniform(-1.0, 1.0, size=m.shape[0])
    b = m @ x_true
    x0 = np.zeros_like(b)  # 真の解と異なる初期値なので残差は非零

    x, n_iter, converged = nk.bicgstab(
        m.indptr, m.indices, m.data, b, x0, rtol=1e-10, atol=1e-300, max_iter=0,
    )
    assert not converged
    assert n_iter == 0
    assert np.all(np.isfinite(x))


def test_csr_matvec_and_bicgstab_numpy_fallback_smoke(monkeypatch):
    """HAVE_NUMBA=False でも csr_matvec_parallel/bicgstab が正しく解を返す
    (scipy の csr@x へフォールバックする経路のスモークテスト、prompts/115)。
    """
    monkeypatch.setattr(nk, "HAVE_NUMBA", False)
    m = _random_diag_dominant_csr(80, seed=9).tocsr()
    rng = np.random.default_rng(10)
    x_true = rng.uniform(-1.0, 1.0, size=m.shape[0])
    b = m @ x_true
    diag_inv = 1.0 / m.diagonal()
    x0 = np.zeros_like(b)

    out = nk.csr_matvec_parallel(m.indptr, m.indices, m.data, x_true)
    np.testing.assert_allclose(out, m @ x_true, rtol=1e-10, atol=1e-12)

    x, n_iter, converged = nk.bicgstab(
        m.indptr, m.indices, m.data, b, x0, rtol=1e-10, atol=1e-300, max_iter=500, diag_inv=diag_inv,
    )
    assert converged
    assert n_iter > 0
    np.testing.assert_allclose(x, x_true, rtol=1e-6, atol=1e-8)
