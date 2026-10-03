"""gmsh によるメッシュ生成 (meshing.generate_mesh) を複数のスレッドから同時に呼ぶテスト。

gmsh の API はプロセスに 1 つだけの状態を使い、スレッドセーフではない。UI v2 のジョブ (1 ジョブ = 1 本の
スレッド)・FastAPI の同期エンドポイント・v1 の WebSocket (asyncio.to_thread) は別々のスレッドから
generate_mesh を呼ぶので、meshing._gmsh_session がロックで initialize → finalize を 1 セッションずつ直列にする。
ロックが無いと、この条件 (スレッドを一斉に開始) では数ラウンドのうちに "Unknown OpenCASCADE point" などの
例外や access violation が出て、最後はヒープ破損 (0xC0000374) でプロセスごと落ちた。
"""

import dataclasses
import threading
import time
from functools import partial

import gmsh
import numpy as np
import pytest

from es_sim import meshing
from es_sim.meshing import Mesh, generate_mesh
from es_sim.schema import Project

W, H = 0.1, 0.05  # domain: 幅 W [m], 高さ H [m]


def _plate(mesh: dict, regions=(), boundaries=()) -> Project:
    return Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [W, 0], [W, H], [0, H]]},
                "regions": list(regions),
                "boundaries": list(boundaries),
            },
            "mesh": mesh,
        }
    )


def _projects() -> list[Project]:
    """gmsh の使い方がそれぞれ違う小さなプロジェクト (1 つ数十 ms)。"""
    return [
        # 平行平板 (左右が Dirichlet)
        _plate({"size": 0.002}, boundaries=[{"edges": [3], "voltage": 0.0}, {"edges": [1], "voltage": 100.0}]),
        # 円の電極 (穴) + 局所サイズ付きの誘電体 (fragment・setSize)
        _plate(
            {"size": 0.003, "local_sizes": [{"region": "diel", "size": 0.001}]},
            regions=[
                {"id": "rod", "type": "conductor", "shape": {"type": "circle", "center": [0.05, 0.025], "radius": 0.01}, "voltage": 50.0},
                {"id": "diel", "type": "dielectric", "polygon": [[0.07, 0.01], [0.09, 0.01], [0.09, 0.04], [0.07, 0.04]], "eps_r": 4.0},
            ],
            boundaries=[{"edges": [0, 1, 2, 3], "voltage": 0.0}],
        ),
        # 上下が周期 (setPeriodic・getPeriodicNodes)
        _plate(
            {"size": 0.0025},
            boundaries=[{"edges": [3], "voltage": 0.0}, {"edges": [1], "voltage": 10.0}, {"edges": [0, 2], "type": "periodic"}],
        ),
        # 辺ローカルサイズ (Distance+Threshold フィールド)
        _plate(
            {"size": 0.004, "local_edge_sizes": [{"p1": [0.02, 0.025], "p2": [0.08, 0.025], "size": 0.0008}]},
            boundaries=[{"edges": [3], "voltage": 0.0}, {"edges": [1], "voltage": 1.0}],
        ),
    ]


def _covered_project() -> Project:
    """domain 全体を conductor が覆う → gmsh のセッションの途中 (fragment の後) で ValueError。"""
    return _plate(
        {"size": 0.002},
        regions=[{"id": "all", "type": "conductor", "polygon": [[-0.01, -0.01], [0.11, -0.01], [0.11, 0.06], [-0.01, 0.06]], "voltage": 0.0}],
    )


def _run_at_once(fns, timeout: float = 120.0) -> list[tuple[str, object]]:
    """fns をそれぞれ別のスレッドで一斉に始め、("ok", 戻り値) か ("error", 例外) の列を返す。"""
    barrier = threading.Barrier(len(fns))
    out: list = [None] * len(fns)

    def work(i: int) -> None:
        try:
            barrier.wait(timeout)  # 全スレッドが揃ってから同時に始める (重なりを最大にする)
            out[i] = ("ok", fns[i]())
        except Exception as exc:
            out[i] = ("error", exc)

    # daemon: ロックの解放漏れなどで止まっても、テストが終わるように
    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(len(fns))]
    for t in threads:
        t.start()
    deadline = time.monotonic() + timeout
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    assert not any(t.is_alive() for t in threads), "終わらないスレッドがある (ロックの解放漏れ?)"
    return out


def _assert_same_mesh(got: Mesh, want: Mesh, label: str) -> None:
    """Mesh の全フィールドが単一スレッドの結果と完全に一致すること。"""
    for f in dataclasses.fields(Mesh):
        a, b = getattr(got, f.name), getattr(want, f.name)
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=f"{label}: {f.name}")
        else:
            assert a == b, f"{label}: {f.name}"


def test_generate_mesh_from_threads_matches_single_thread():
    """8 本のスレッド (各プロジェクト 2 本) で一斉にメッシュを作っても、全て単一スレッドの結果と一致すること。"""
    projects = _projects()
    expected = [generate_mesh(p) for p in projects]
    fns = [partial(generate_mesh, projects[i % len(projects)]) for i in range(2 * len(projects))]
    for _ in range(3):  # ロックが無いと数ラウンドのうちに壊れた
        out = _run_at_once(fns)
        for i, (status, mesh) in enumerate(out):
            k = i % len(projects)
            assert status == "ok", f"thread {i} (project {k}): {mesh!r}"
            _assert_same_mesh(mesh, expected[k], f"thread {i} (project {k})")


def test_generate_mesh_error_finalizes_and_releases_lock():
    """gmsh のセッションの途中で失敗しても finalize とロックの解放が行われ、同時に走る他のスレッドや
    その後のメッシュ生成に影響しないこと。"""
    projects = _projects()
    expected = [generate_mesh(p) for p in projects]

    with pytest.raises(ValueError, match="メッシュ化できる面がありません"):
        generate_mesh(_covered_project())
    # ロックを取れる = 解放されている。取っている間は他のセッションが無いので、初期化の状態を見られる
    assert meshing._GMSH_LOCK.acquire(timeout=60.0), "失敗したセッションのロックが解放されていない"
    try:
        assert not gmsh.isInitialized(), "失敗したセッションが finalize されていない"
    finally:
        meshing._GMSH_LOCK.release()

    # 失敗するスレッドと成功するスレッドを一斉に走らせる
    out = _run_at_once([partial(generate_mesh, _covered_project())] + [partial(generate_mesh, p) for p in projects])
    status, exc = out[0]
    assert status == "error" and isinstance(exc, ValueError), exc
    for k, (status, mesh) in enumerate(out[1:]):
        assert status == "ok", f"project {k}: {mesh!r}"
        _assert_same_mesh(mesh, expected[k], f"project {k}")
