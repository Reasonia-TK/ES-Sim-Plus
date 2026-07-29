"""任意の線分近傍のローカルメッシュ細分化 (mesh.local_edge_sizes, prompts/90) のテスト。

gmsh の Distance+Threshold フィールドで線分近傍を細分化する実装 (meshing._add_edge_mesh_fields)
の実効性・空リスト時のビット不変性・既存の領域ローカルサイズとの共存を検証する。
"""

import numpy as np
import pytest
from pydantic import ValidationError

from es_sim.meshing import _add_edge_mesh_fields, generate_mesh
from es_sim.schema import EdgeMeshSize, Project


def _uniform_project(entries, local_sizes=None, size=0.02, w=1.0, h=1.0, regions=()) -> Project:
    mesh: dict = {"size": size, "local_edge_sizes": entries}
    if local_sizes is not None:
        mesh["local_sizes"] = local_sizes
    return Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [w, 0], [w, h], [0, h]]},
                "regions": list(regions),
            },
            "mesh": mesh,
        }
    )


def _tri_h(mesh) -> np.ndarray:
    """三角形ごとの代表寸法 sqrt(2A) (A は面積)。"""
    tri_pts = mesh.nodes[mesh.triangles]  # (M, 3, 2)
    a = tri_pts[:, 1] - tri_pts[:, 0]
    b = tri_pts[:, 2] - tri_pts[:, 0]
    area = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    return np.sqrt(2.0 * area)


def _dist_to_segment(pts: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """点列 pts (K, 2) と線分 q1-q2 の最短距離 (線分の外側も含む一般の点-線分距離)。"""
    seg = q2 - q1
    seg_len2 = float(seg[0] ** 2 + seg[1] ** 2)
    d = pts - q1
    t = np.clip((d[:, 0] * seg[0] + d[:, 1] * seg[1]) / seg_len2, 0.0, 1.0)
    proj = q1[np.newaxis, :] + t[:, np.newaxis] * seg[np.newaxis, :]
    return np.hypot(pts[:, 0] - proj[:, 0], pts[:, 1] - proj[:, 1])


# ---- 1. 細分化の実効性 -------------------------------------------------------------


def test_edge_local_mesh_refines_near_segment():
    size = 0.02
    edge_size = size / 5.0  # 全体特性長の 1/5
    p1, p2 = np.array([0.2, 0.5]), np.array([0.8, 0.5])
    project = _uniform_project(
        [{"p1": p1.tolist(), "p2": p2.tolist(), "size": edge_size}], size=size
    )
    mesh = generate_mesh(project)

    h = _tri_h(mesh)
    centroids = mesh.nodes[mesh.triangles].mean(axis=1)
    dist_in = 2.0 * edge_size  # 既定の dist_in (meshing._add_edge_mesh_fields 参照)
    near = _dist_to_segment(centroids, p1, p2) <= dist_in

    assert near.sum() > 0
    median_near = float(np.median(h[near]))
    median_all = float(np.median(h))
    # 近傍の代表寸法は全体の中央値の 1/2.5 以下に十分小さくなること
    assert median_near <= median_all / 2.5, (median_near, median_all)


# ---- 2. 空リストでビット不変 --------------------------------------------------------


def test_edge_local_mesh_empty_list_is_gmsh_noop():
    """local_edge_sizes が空なら gmsh の Field API を一切呼ばずに即 return する
    (= フィールドが存在しない従来経路と完全に同じコード経路になる、prompts/90)。

    gmsh.initialize していない状態で呼んでも例外にならないこと自体が
    「gmsh 呼び出しが一切発生していない」ことの直接的な証明になる。
    """
    _add_edge_mesh_fields([], 0.01)  # gmsh 未初期化でも例外にならないはず


# ---- 3. 領域ローカルサイズとの共存 ---------------------------------------------------


def test_edge_local_mesh_coexists_with_region_local_size():
    size = 0.02
    region_lc = size / 6.0
    edge_size = size / 5.0
    p1, p2 = np.array([0.5, 0.5]), np.array([0.5, 0.9])
    project = _uniform_project(
        [{"p1": p1.tolist(), "p2": p2.tolist(), "size": edge_size}],
        local_sizes=[{"region": "diel1", "size": region_lc}],
        size=size,
        regions=[
            {
                "id": "diel1",
                "type": "dielectric",
                "polygon": [[0.05, 0.05], [0.15, 0.05], [0.15, 0.15], [0.05, 0.15]],
                "eps_r": 4.0,
            }
        ],
    )
    mesh = generate_mesh(project)

    h = _tri_h(mesh)
    centroids = mesh.nodes[mesh.triangles].mean(axis=1)
    median_all = float(np.median(h))

    # 領域ローカルサイズ側 (diel1 = regions[0] → tri_region == 0) が効いていること
    region_mask = mesh.tri_region == 0
    assert region_mask.sum() > 0
    assert float(np.median(h[region_mask])) <= median_all / 1.8

    # 辺ローカルサイズ側も同時に効いていること
    dist_in = 2.0 * edge_size
    near_edge = _dist_to_segment(centroids, p1, p2) <= dist_in
    assert near_edge.sum() > 0
    assert float(np.median(h[near_edge])) <= median_all / 1.8


# ---- 4. validator: size <= 0 ---------------------------------------------------------


def test_edge_mesh_size_requires_positive_size():
    with pytest.raises(ValidationError):
        EdgeMeshSize(p1=(0.0, 0.0), p2=(1.0, 0.0), size=0.0)
    with pytest.raises(ValidationError):
        EdgeMeshSize(p1=(0.0, 0.0), p2=(1.0, 0.0), size=-1.0)
