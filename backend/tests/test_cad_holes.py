"""領域の穴 (prompts/132 P7e)。

- schema: 穴は polygon の領域だけ、外周の内側、外周・ほかの穴と交わらず接しない、入れ子にしない、円弧は弦に分ける
- v2 の形状: 内包・交点は全ての輪で偶奇、vertices は外周と穴を水平な橋でつないだ 1 本の輪 (鍵穴形) で、
  GPU カーネルと同じ偶奇判定・最初の交点・流体の固体表面がそのまま正しい
- メッシュ (gmsh・構造格子) と v2 直交格子: 導体の穴の中もメッシュ化され、電位は導体と同じ (静電遮蔽)。
  誘電体の穴の中は真空
- 同軸: 外導体を「円の穴のある 1 つの領域」で書いても静電容量が解析解に近い
"""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from es_sim.fem import EPS0, solve
from es_sim.field import solve_electrostatic
from es_sim.geom.model import GeometryModel
from es_sim.geom.shapes import PolygonShape
from es_sim.gfluid.geometry import solid_pieces
from es_sim.meshing import generate_mesh
from es_sim.paths import arc_segment_count
from es_sim.schema import Project


def _sq(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _project(regions, *, size=0.005, mode=None, boundaries=None) -> Project:
    mesh = {"size": size} | ({"mode": mode} if mode else {})
    return Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": _sq(0.0, 0.0, 0.1, 0.1)},
                "regions": regions,
                "boundaries": boundaries if boundaries is not None else [{"edges": [0, 1, 2, 3], "voltage": 0.0}],
            },
            "mesh": mesh,
        }
    )


def _cage(hole=None):
    """1 V の導体の枠: 外周 [0.02, 0.08]²、穴 (既定は [0.04, 0.06]²)。"""
    return {
        "id": "cage",
        "type": "conductor",
        "voltage": 1.0,
        "polygon": _sq(0.02, 0.02, 0.08, 0.08),
        "holes": [hole or {"polygon": _sq(0.04, 0.04, 0.06, 0.06)}],
    }


# ---- schema ----


def _region(**kw):
    return Project.model_validate(
        {"geometry": {"domain": {"polygon": _sq(-1, -1, 5, 5)}, "regions": [{"id": "r", "type": "dielectric", **kw}]}, "mesh": {"size": 0.1}}
    ).geometry.regions[0]


def test_holes_are_validated():
    ok = _region(polygon=_sq(0, 0, 1, 1), holes=[{"polygon": _sq(0.4, 0.4, 0.6, 0.6)}])
    assert len(ok.holes) == 1
    cases = {
        "外周の外": [{"polygon": _sq(1.2, 0.4, 1.4, 0.6)}],
        "交わるか接しています": [{"polygon": _sq(0.2, 0.2, 0.5, 0.5)}, {"polygon": _sq(0.4, 0.4, 0.7, 0.7)}],
        "一方がもう一方の中": [{"polygon": _sq(0.2, 0.2, 0.8, 0.8)}, {"polygon": _sq(0.4, 0.4, 0.6, 0.6)}],
        "3 頂点以上": [{"polygon": [[0.4, 0.4], [0.6, 0.6]]}],
    }
    for msg, holes in cases.items():
        with pytest.raises(ValidationError, match=msg):
            _region(polygon=_sq(0, 0, 1, 1), holes=holes)
    # 凹んだ外周: 穴の頂点はすべて内側だが、辺が切り欠きを横切る
    u_shape = [[0, 0], [3, 0], [3, 3], [2, 3], [2, 1], [1, 1], [1, 3], [0, 3]]
    with pytest.raises(ValidationError, match="外周と交わるか接しています"):
        _region(polygon=u_shape, holes=[{"polygon": _sq(0.5, 1.5, 2.5, 2.0)}])
    # 外周に頂点で接する穴
    with pytest.raises(ValidationError):
        _region(polygon=_sq(0, 0, 1, 1), holes=[{"polygon": [[0.5, 0.0], [0.7, 0.3], [0.3, 0.3]]}])
    # 円の領域には穴を付けられない
    with pytest.raises(ValidationError, match="polygon の領域だけ"):
        _region(shape={"kind": "circle", "center": [0, 0], "radius": 1}, holes=[{"polygon": _sq(0, 0, 0.1, 0.1)}])


def test_hole_arcs_are_flattened_with_the_region_size():
    circle_hole = {"polygon": [[0.06, 0.05], [0.04, 0.05]], "bulges": [1, 1]}  # 半径 0.01 の円
    p = _project([_cage(circle_hole)], size=0.005)
    hole = p.geometry.regions[0].holes[0]
    assert hole.bulges is None
    assert len(hole.polygon) == 2 * arc_segment_count(0.01, math.pi, 0.005)
    r = np.hypot(np.asarray(hole.polygon)[:, 0] - 0.05, np.asarray(hole.polygon)[:, 1] - 0.05)
    assert r == pytest.approx(np.full(len(r), 0.01), rel=1e-12)
    # 局所メッシュ幅があればそれで分ける
    q = Project.model_validate(
        {
            "geometry": {"domain": {"polygon": _sq(0, 0, 0.1, 0.1)}, "regions": [_cage(circle_hole)]},
            "mesh": {"size": 0.005, "local_sizes": [{"region": "cage", "size": 0.0005}]},
        }
    )
    assert len(q.geometry.regions[0].holes[0].polygon) == 2 * arc_segment_count(0.01, math.pi, 0.0005)


# ---- v2 の形状 (鍵穴形の輪) ----


def _parity(v: np.ndarray, x: float, y: float) -> bool:
    """GPU カーネル (es_in_polygon / ds_in_polygon) と同じ半開区間規則の偶奇判定。"""
    odd = False
    n = len(v)
    for a in range(n):
        (ax, ay), (bx, by) = v[a], v[a - 1]
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            odd = not odd
    return odd


def _first_hit(v: np.ndarray, o: np.ndarray, p: np.ndarray):
    """GPU カーネル (es/ds_solid_hit) と同じ、線分 o→p が最初に交わる辺の交点。"""
    d = p - o
    best, hit = 2.0, None
    for a in range(len(v)):
        q0, e = v[a], v[(a + 1) % len(v)] - v[a]
        den = d[0] * e[1] - d[1] * e[0]
        if den == 0.0:
            continue
        w = q0 - o
        t = (w[0] * e[1] - w[1] * e[0]) / den
        u = (w[0] * d[1] - w[1] * d[0]) / den
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0 and t < best:
            best, hit = t, o + t * d
    return hit


def _dist_to_rings(rings, q) -> float:
    best = math.inf
    for r in rings:
        for a, b in zip(r, np.roll(r, -1, axis=0)):
            e = b - a
            t = min(1.0, max(0.0, float(np.dot(q - a, e) / np.dot(e, e))))
            best = min(best, float(np.linalg.norm(q - (a + t * e))))
    return best


def _area(v):
    w = np.roll(v, -1, axis=0)
    return 0.5 * float(np.sum(v[:, 0] * w[:, 1] - w[:, 0] * v[:, 1]))


SCENES = {
    # 同じ高さに並んだ 2 つの四角い穴 (右端の頂点が同点、橋が穴の辺と一直線)
    "side_by_side": (_sq(0, 0, 4, 3), [_sq(1, 1, 2, 2), _sq(2.5, 1, 3.5, 2)]),
    # 左の穴からの水平線が、先につないだ三角の穴の右端 (輪に 2 回出る点) に当たる
    "shared_vertex": (_sq(0, 0, 4, 4), [[[3.5, 3.0], [3.0, 3.5], [2.8, 3.2]], _sq(1, 2.5, 2, 3)]),
    # 外周は時計回り、穴は多角形の円 (反時計回り) と三角
    "cw_outer": (
        _sq(0, 0, 4, 4)[::-1],
        [[[2 + 0.8 * math.cos(t), 2 + 0.8 * math.sin(t)] for t in np.linspace(0, 2 * math.pi, 40, endpoint=False)], [[0.5, 0.5], [1.0, 0.5], [0.5, 1.0]]],
    ),
}


@pytest.mark.parametrize("name", list(SCENES))
def test_polygon_shape_keyhole_matches_the_rings(name):
    outer, holes = SCENES[name]
    s = PolygonShape(np.asarray(outer, dtype=float), tuple(np.asarray(h, dtype=float) for h in holes))
    rings = s.rings
    v = s.vertices
    assert len(rings) == 1 + len(holes)
    # 面積 = 外周 - 穴、外周は反時計回り
    assert _area(v) == pytest.approx(abs(_area(rings[0])) - sum(abs(_area(h)) for h in rings[1:]), rel=1e-12)
    # 輪に無い辺は水平な橋だけ
    for a, b in zip(v, np.roll(v, -1, axis=0)):
        if _dist_to_rings(rings, (a + b) / 2) > 1e-12:
            assert a[1] == b[1]
    rng = np.random.default_rng(1)
    pts = rng.uniform(-0.2, 4.2, size=(400, 2))
    inside = s.contains(pts[:, 0], pts[:, 1])
    for (x, y), c in zip(pts, inside):
        in_outer = PolygonShape(rings[0]).contains(np.array([x]), np.array([y]))[0]
        in_hole = any(PolygonShape(h).contains(np.array([x]), np.array([y]))[0] for h in rings[1:])
        assert c == (in_outer and not in_hole)
        assert _parity(v, x, y) == c
    # 気体側の点から固体の点へ: 最初の交点は輪の上 (橋には当たらない)
    gas, solid = pts[~inside], pts[inside]
    for k in range(60):
        o, p = gas[k % len(gas)], solid[(7 * k) % len(solid)]
        hit = _first_hit(v, o, p)
        assert hit is not None
        assert _dist_to_rings(rings, hit) < 1e-12


def test_polygon_shape_crossings_count_every_ring():
    outer, holes = SCENES["side_by_side"]
    s = PolygonShape(np.asarray(outer, dtype=float), tuple(np.asarray(h, dtype=float) for h in holes))
    k, x = s.crossings_h(np.array([1.5]), np.array([-1.0]), np.array([5.0]))
    assert sorted(x.tolist()) == pytest.approx([0.0, 1.0, 2.0, 2.5, 3.5, 4.0])
    k, y = s.crossings_v(np.array([1.5, 3.0]), np.array([-1.0, -1.0]), np.array([5.0, 5.0]))
    assert sorted(y[k == 0].tolist()) == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert s.bbox == (0.0, 0.0, 4.0, 3.0)


# ---- メッシュと静電場 ----


def _hole_nodes(nodes, lo=0.04, hi=0.06, margin=1e-9):
    return (nodes[:, 0] > lo + margin) & (nodes[:, 0] < hi - margin) & (nodes[:, 1] > lo + margin) & (nodes[:, 1] < hi - margin)


@pytest.mark.parametrize("mode", [None, "structured"])
def test_conductor_with_a_hole_shields_its_inside(mode):
    p = _project([_cage()], mode=mode)
    mesh = generate_mesh(p)
    cent = mesh.nodes[mesh.triangles].mean(axis=1)
    in_hole = _hole_nodes(cent)
    in_outer = (cent[:, 0] > 0.02) & (cent[:, 0] < 0.08) & (cent[:, 1] > 0.02) & (cent[:, 1] < 0.08)
    assert np.count_nonzero(in_hole) > 0                     # 穴の中もメッシュ化される
    assert not np.any(in_outer & ~in_hole)                   # 導体の中身は除かれる
    ring = [n for n, v in mesh.dirichlet.items() if abs(mesh.nodes[n, 0] - 0.04) < 1e-12 and 0.04 <= mesh.nodes[n, 1] <= 0.06]
    assert ring and all(mesh.dirichlet[n] == pytest.approx(1.0) and mesh.electrode[n] == "cage" for n in ring)
    sol = solve(p, mesh)
    inner = _hole_nodes(mesh.nodes)
    assert np.count_nonzero(inner) > 0
    assert sol.v[inner] == pytest.approx(np.ones(np.count_nonzero(inner)), abs=1e-9)


@pytest.mark.parametrize("mode", [None, "structured"])
def test_dielectric_hole_is_vacuum(mode):
    diel = {"id": "d", "type": "dielectric", "eps_r": 4.0, "polygon": _sq(0.02, 0.02, 0.08, 0.08), "holes": [{"polygon": _sq(0.04, 0.04, 0.06, 0.06)}]}
    mesh = generate_mesh(_project([diel], mode=mode))
    cent = mesh.nodes[mesh.triangles].mean(axis=1)
    in_hole = _hole_nodes(cent)
    in_outer = (cent[:, 0] > 0.02) & (cent[:, 0] < 0.08) & (cent[:, 1] > 0.02) & (cent[:, 1] < 0.08)
    assert np.all(mesh.tri_region[in_hole] == -1)
    assert np.all(mesh.tri_region[in_outer & ~in_hole] == 0)


def test_v2_conductor_with_a_hole_shields_its_inside():
    p = _project([_cage({"polygon": [[0.06, 0.05], [0.04, 0.05]], "bulges": [1, 1]})], size=0.002)
    sol = solve_electrostatic(p, device="cpu")
    X, Y = np.meshgrid(sol.grid.xs, sol.grid.ys)
    inner = np.hypot(X - 0.05, Y - 0.05) < 0.008
    assert np.count_nonzero(inner) > 10
    assert sol.phi[inner] == pytest.approx(np.ones(np.count_nonzero(inner)), abs=1e-8)
    outside = (X < 0.015) | (X > 0.085)
    assert np.all(sol.phi[outside] < 1.0)


def test_v2_coax_with_the_outer_conductor_as_a_region_with_a_hole():
    """外導体 = 円 (r = b) の穴のある正方形 1 つ (従来のテストは 4 つの多角形に分けていた)。"""
    a, b, L = 0.01, 0.04, 0.05
    big = 1.2 * L
    p = Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": _sq(-L, -L, L, L)},
                "regions": [
                    {"id": "inner", "type": "conductor", "voltage": 1.0, "shape": {"kind": "circle", "center": [0.0, 0.0], "radius": a}},
                    {"id": "outer", "type": "conductor", "voltage": 0.0, "polygon": _sq(-big, -big, big, big), "holes": [{"polygon": [[b, 0.0], [-b, 0.0]], "bulges": [1, 1]}]},
                ],
            },
            "mesh": {"size": 0.1 / 128},
        }
    )
    sol = solve_electrostatic(p, device="cpu")
    c_exact = 2.0 * math.pi * EPS0 / math.log(b / a)
    assert sol.capacitance == pytest.approx(c_exact, rel=1e-3)


def test_fluid_solid_pieces_follow_both_rings():
    """流体の固体表面の小片は外周と穴の輪の上だけ (橋は両側が固体なので出ない)、法線は固体の内向き。"""
    model = GeometryModel(_project([_cage()]))
    pieces = solid_pieces(model, 0.001, 1e-7)
    mid = np.concatenate([m for m, *_ in pieces])
    length = np.concatenate([a for _, a, *_ in pieces])
    n_in = np.concatenate([n for _, _, n, *_ in pieces])
    assert float(length.sum()) == pytest.approx(4 * 0.06 + 4 * 0.02, rel=1e-12)
    on_hole = (np.abs(mid - 0.05) <= 0.01 + 1e-12).all(axis=1)
    assert float(length[on_hole].sum()) == pytest.approx(4 * 0.02, rel=1e-12)
    # 穴の輪の法線は穴の中心から外向き (固体の側)、外周の輪の法線は中心向き
    radial = mid - 0.05
    assert np.all(np.sum(n_in[on_hole] * radial[on_hole], axis=1) > 0.0)
    assert np.all(np.sum(n_in[~on_hole] * radial[~on_hole], axis=1) < 0.0)
