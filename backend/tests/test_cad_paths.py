"""CAD v2 の円弧 (bulges) と辺の永続 ID (prompts/132)。

- paths.py: 円弧の中心・面積・弦への分割 (円の多角形化と同じ密度)
- schema: bulges・edge_ids の検査、周期境界は直線だけ、円弧の無い文書は何も変えない
- Project の検証で円弧を弦に分け、外周の辺の番号の参照 (境界条件・FN・反射・DSMC) を展開する
- メッシュと静電場: 円弧の外周の電極ラベルは元の辺の番号 (円弧のあとの辺の電荷・静電容量も落とさない)、
  同軸の静電容量が解析解に近い
"""

import math

import pytest
from pydantic import ValidationError

from es_sim.fem import EPS0, solve
from es_sim.meshing import CIRCLE_SEGMENTS_MIN, _circle_polygon, generate_mesh
from es_sim.paths import arc_of, arc_segment_count, flatten_path, path_area
from es_sim.schema import Project

Q = math.tan(math.pi / 8)  # 90° の円弧の bulge


def _rect(w=0.1, h=0.05):
    return [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]]


def _project(geometry: dict, mesh: dict | None = None, **extra) -> Project:
    return Project.model_validate({"geometry": geometry, "mesh": mesh or {"size": 0.01}, **extra})


# ---- paths.py ----


def test_arc_of_quarter_circle():
    (cx, cy), r, a0, theta = arc_of((1.0, 0.0), (0.0, 1.0), Q)
    assert (cx, cy) == pytest.approx((0.0, 0.0), abs=1e-12)
    assert r == pytest.approx(1.0)
    assert a0 == pytest.approx(0.0, abs=1e-12)
    assert theta == pytest.approx(math.pi / 2)
    # 時計回り (負の bulge) は中心が弦の右
    (cx, cy), _, _, theta = arc_of((1.0, 0.0), (0.0, 1.0), -Q)
    assert (cx, cy) == pytest.approx((1.0, 1.0))
    assert theta == pytest.approx(-math.pi / 2)


def test_path_area_counts_the_segments():
    assert path_area([(1.0, 0.0), (-1.0, 0.0)], [1.0, 1.0]) == pytest.approx(math.pi)
    sq = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    assert path_area(sq, None) == pytest.approx(4.0)
    assert path_area(sq, [0.0, 1.0, 0.0, 0.0]) == pytest.approx(4.0 + math.pi / 2)


def test_flatten_keeps_vertices_and_matches_circle_density():
    r, h = 0.02, 0.002
    pts, origin = flatten_path([(r, 0.0), (-r, 0.0)], [1.0, 1.0], h)
    n_full = math.ceil(2 * math.pi * r / h)  # 63 本
    assert len(pts) == 2 * math.ceil(n_full / 2)
    assert pts[0] == (r, 0.0)
    assert origin[0] == 0 and origin[-1] == 1
    assert pts[origin.index(1)] == (-r, 0.0)
    for x, y in pts:
        assert math.hypot(x, y) == pytest.approx(r, rel=1e-12)
    # 偶数本なら円の多角形化と同じ頂点
    pts2, _ = flatten_path([(r, 0.0), (-r, 0.0)], [1.0, 1.0], 2 * math.pi * r / 64)
    circle = _circle_polygon((0.0, 0.0), r, 2 * math.pi * r / 64)
    assert len(pts2) == len(circle) == 64
    for a, b in zip(pts2, circle):
        assert a == pytest.approx(b, abs=1e-15)
    # 小さい円弧も 1 周 24 本の割合 (90° なら 6 本)
    assert arc_segment_count(1e-6, math.pi / 2, 1.0) == CIRCLE_SEGMENTS_MIN // 4
    # 直線だけならそのまま
    pts3, origin3 = flatten_path(_rect(), None, 0.01)
    assert pts3 == [tuple(p) for p in _rect()] and origin3 == [0, 1, 2, 3]


# ---- スキーマの検査 ----


def test_schema_checks_bulges_and_edge_ids():
    geo = {"domain": {"polygon": _rect(), "bulges": [0, 0.5, 0]}}
    with pytest.raises(ValidationError, match="bulges は頂点と同じ数"):
        _project(geo)
    with pytest.raises(ValidationError, match="3 頂点以上"):
        _project({"domain": {"polygon": [[0, 0], [1, 0]]}})
    # 円弧を含めば 2 頂点 (円) でもよい
    p = _project({"domain": {"polygon": [[0.05, 0], [-0.05, 0]], "bulges": [1, 1]}})
    assert len(p.geometry.domain.polygon) > 24
    with pytest.raises(ValidationError, match="edge_ids は辺と同じ数"):
        _project({"domain": {"polygon": _rect(), "edge_ids": ["e1", "e2"]}})
    with pytest.raises(ValidationError, match="重複のない"):
        _project({"domain": {"polygon": _rect(), "edge_ids": ["e1", "e2", "e2", "e4"]}})
    with pytest.raises(ValidationError, match="bulges・holes は polygon の領域だけ"):
        _project(
            {
                "domain": {"polygon": _rect()},
                "regions": [{"id": "c", "type": "conductor", "shape": {"center": [0.05, 0.02], "radius": 0.01}, "bulges": [1]}],
            }
        )


def test_periodic_edges_must_be_straight():
    geo = {
        "domain": {"polygon": _rect(), "bulges": [0.2, 0, 0.2, 0]},
        "boundaries": [{"edges": [0, 2], "type": "periodic"}],
    }
    with pytest.raises(ValidationError, match="円弧"):
        _project(geo)


def test_projects_without_arcs_are_unchanged():
    raw = {
        "geometry": {
            "domain": {"polygon": _rect(), "edge_ids": ["e1", "e2", "e3", "e4"]},
            "regions": [{"id": "c", "type": "conductor", "polygon": [[0.04, 0.02], [0.06, 0.02], [0.05, 0.03]], "voltage": 1.0}],
            "boundaries": [{"edges": [3], "voltage": 0.0}, {"edges": [1], "voltage": 5.0}],
        },
        "mesh": {"size": 0.01},
    }
    p = Project.model_validate(raw)
    assert p.geometry._edge_origin is None
    assert [list(q) for q in p.geometry.domain.polygon] == raw["geometry"]["domain"]["polygon"]
    assert p.geometry.domain.edge_ids == ["e1", "e2", "e3", "e4"]
    assert [b.edges for b in p.geometry.boundaries] == [[3], [1]]
    assert p.geometry.edge_label(3) == "edge3"


# ---- 検証の段階での弦への分割 ----


def test_domain_arc_expands_every_edge_reference():
    # 上の辺 (辺 2) を外へふくらませた矩形。辺 3 以降の番号がずれる
    geo = {
        "domain": {"polygon": _rect(), "bulges": [0, 0, 0.3, 0], "edge_ids": ["a", "b", "c", "d"]},
        "boundaries": [{"edges": [2], "voltage": 10.0}, {"edges": [3], "voltage": 0.0}],
    }
    extra = {
        "pic": {"reflect_edges": [2, 3], "fn": {"edges": [3]}},
        "dsmc": {"init_pressure_pa": 1.0, "boundaries": [{"edges": [2], "type": "wall"}]},
    }
    p = _project(geo, **extra)
    origin = p.geometry._edge_origin
    assert origin is not None
    arc = [k for k, o in enumerate(origin) if o == 2]
    assert len(arc) > 1
    last = len(origin) - 1
    assert origin[last] == 3
    assert p.geometry.boundaries[0].edges == arc
    assert p.geometry.boundaries[1].edges == [last]
    assert p.pic.reflect_edges == arc + [last]
    assert p.pic.fn.edges == [last]
    assert p.dsmc.boundaries[0].edges == arc
    assert p.geometry.domain.bulges is None and p.geometry.domain.edge_ids is None
    assert len(p.geometry.domain.polygon) == len(origin)
    assert {p.geometry.edge_label(k) for k in arc} == {"edge2"}
    assert p.geometry.edge_label(last) == "edge3"
    # 円弧の途中の点は円の上 (上の辺より外側)
    assert max(y for _, y in p.geometry.domain.polygon) > 0.05


def test_out_of_range_edges_are_rejected_when_arcs_expand_the_numbering():
    geo = {"domain": {"polygon": _rect(), "bulges": [0, 0, 0.3, 0]}, "boundaries": [{"edges": [4], "voltage": 1.0}]}
    with pytest.raises(ValidationError, match="範囲外"):
        _project(geo)


def test_region_arcs_use_the_local_mesh_size():
    region = {"id": "r", "type": "dielectric", "polygon": [[0.06, 0.025], [0.04, 0.025]], "bulges": [1, 1], "eps_r": 4.0}
    coarse = _project({"domain": {"polygon": _rect()}, "regions": [region]})
    fine = _project({"domain": {"polygon": _rect()}, "regions": [region]}, {"size": 0.01, "local_sizes": [{"region": "r", "size": 0.0005}]})
    assert len(coarse.geometry.regions[0].polygon) == CIRCLE_SEGMENTS_MIN
    assert len(fine.geometry.regions[0].polygon) == 2 * math.ceil(math.ceil(2 * math.pi * 0.01 / 0.0005) / 2)
    assert coarse.geometry.regions[0].bulges is None


def test_cartesian_amr_uses_the_finest_level():
    region = {"id": "r", "type": "conductor", "polygon": [[0.06, 0.025], [0.04, 0.025]], "bulges": [1, 1], "voltage": 1.0}
    mesh = {"size": 0.0005, "mode": "cartesian", "amr": {"max_level": 2}}
    p = _project({"domain": {"polygon": _rect()}, "regions": [region]}, mesh)
    n_full = math.ceil(2 * math.pi * 0.01 / (0.0005 / 4))
    assert len(p.geometry.regions[0].polygon) == 2 * math.ceil(min(n_full, 720) / 2)


def test_axisymmetric_arc_crossing_the_axis_is_rejected():
    # rz: y = r ≥ 0。下の辺 (軸、+x 向き) の正の bulge は外 (下) へふくらみ r < 0 になる
    geo = {"domain": {"polygon": _rect(), "bulges": [0.3, 0, 0, 0]}}
    with pytest.raises(ValidationError, match="r\\) ≥ 0"):
        _project(geo, coord="rz")
    # 内 (上) へふくらませた円弧は軸の上ではない (両端が r=0 でも Dirichlet を付けられる)
    ok = _project({"domain": {"polygon": _rect(), "bulges": [-0.3, 0, 0, 0]}, "boundaries": [{"edges": [0], "voltage": 1.0}]}, coord="rz")
    assert ok.geometry.boundaries[0].edges


# ---- メッシュと静電場 ----


def test_arc_edge_electrode_label_uses_the_original_edge():
    geo = {
        "domain": {"polygon": _rect(), "bulges": [0, 0, 0.3, 0]},
        "boundaries": [{"edges": [2], "voltage": 10.0}, {"edges": [3], "voltage": 0.0}],
    }
    p = _project(geo, {"size": 0.005})
    mesh = generate_mesh(p)
    labels = set(mesh.electrode.values())
    assert labels == {"edge2", "edge3"}
    sol = solve(p, mesh)
    charges = {label: q for label, _, q in sol.charges}
    assert set(charges) == {"edge2", "edge3"}
    assert charges["edge2"] == pytest.approx(-charges["edge3"], rel=1e-6)


@pytest.mark.parametrize(
    ("bulges", "boundaries", "labels"),
    [
        # 円弧 (辺 2) のあとの辺 3 は弦の番号が 3 からずれる
        ([0, 0, 0.3, 0], [{"edges": [3], "voltage": 1.0}, {"edges": [1], "voltage": 0.0}], ["edge3", "edge1"]),
        # 円弧 (辺 1) のあとの円弧 (辺 2) の電極: 弦の番号に 2 が無く、弦がいくつあっても 1 つの電極
        ([0, 0.3, 0.3, 0], [{"edges": [2], "voltage": 1.0}, {"edges": [0], "voltage": 0.0}], ["edge2", "edge0"]),
    ],
)
def test_electrodes_after_an_arc_keep_charge_and_capacitance(bulges, boundaries, labels):
    """円弧のあとの外周の辺の電極も電荷の一覧に元の辺の番号で出て、静電容量が定義される。"""
    p = _project({"domain": {"polygon": _rect(0.02, 0.01), "bulges": bulges}, "boundaries": boundaries}, {"size": 1e-3})
    # 前提: 弦の番号のまま "edge{e}" にすると元の辺のラベルが揃わない (修正前の fem._label_order はここで電極を落とした)
    assert not set(labels) <= {f"edge{e}" for bc in p.geometry.boundaries for e in bc.edges}
    sol = solve(p, generate_mesh(p))
    assert [(label, v) for label, v, _ in sol.charges] == [(labels[0], 1.0), (labels[1], 0.0)]
    q_ground = sol.charges[1][2]
    assert sol.capacitance is not None
    assert sol.capacitance == pytest.approx(-q_ground / 1.0, rel=1e-6)


def test_coaxial_capacitance_with_arc_domain():
    """外周 = 半径 R の円 (半円 2 つ、0 V)、内導体 = 半径 a の円 (円弧の領域、1 V)。C = 2πε0 / ln(R/a)。"""
    R, a = 0.05, 0.01
    geo = {
        "domain": {"polygon": [[R, 0.0], [-R, 0.0]], "bulges": [1, 1]},
        "regions": [{"id": "inner", "type": "conductor", "polygon": [[a, 0.0], [-a, 0.0]], "bulges": [1, 1], "voltage": 1.0}],
        "boundaries": [{"edges": [0, 1], "voltage": 0.0}],
    }
    p = _project(geo, {"size": 0.002, "local_sizes": [{"region": "inner", "size": 0.0005}]})
    mesh = generate_mesh(p)
    sol = solve(p, mesh)
    charges = {label: q for label, _, q in sol.charges}
    assert set(charges) == {"edge0", "edge1", "inner"}
    c = charges["inner"] / 1.0
    exact = 2 * math.pi * EPS0 / math.log(R / a)
    assert c == pytest.approx(exact, rel=5e-3)
    assert charges["edge0"] + charges["edge1"] == pytest.approx(-charges["inner"], rel=1e-6)
