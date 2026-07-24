"""静電容量・電極電荷の出力テスト (prompts/71)。

電極ごとの誘起電荷は fem.py の残差法 (r = Kφ - f をDirichlet節点で評価) で計算する。
静電容量は電極電位がちょうど2水準・空間電荷 ρ=0 の場合のみ定義される。
"""

import math

import numpy as np
import pytest

from es_sim.fem import EPS0, solve
from es_sim.meshing import generate_mesh
from es_sim.schema import Project

D, H, V1 = 0.1, 0.05, 100.0  # 平行平板: 幅 D [m] (電極間隔)、高さ H [m] (電極の対向長さ)


def _plates_project(eps_r: float | None = None) -> Project:
    """平行平板コンデンサ: domain 矩形 [0,D]×[0,H]、左辺 (edge3) 0V・右辺 (edge1) V1。

    eps_r を指定すると domain 全域を同じ形状の dielectric 領域で覆う (一様誘電体)。
    domain と厳密に同じ輪郭を与えると gmsh の fragment がそのまま1枚の面として
    扱うため、電極間の場は eps_r=1 のときと変わらず線形のまま (P1 で厳密)。
    """
    geometry: dict = {
        "domain": {"polygon": [[0, 0], [D, 0], [D, H], [0, H]]},
        "boundaries": [
            {"edges": [3], "voltage": 0.0},   # 左辺
            {"edges": [1], "voltage": V1},    # 右辺
        ],
    }
    if eps_r is not None:
        geometry["regions"] = [
            {
                "id": "diel",
                "type": "dielectric",
                "polygon": [[0, 0], [D, 0], [D, H], [0, H]],
                "eps_r": eps_r,
            }
        ]
    return Project.model_validate({"geometry": geometry, "mesh": {"size": 0.01}})


def test_parallel_plate_capacitance():
    """真空平行平板: 場が線形 (P1 で厳密) なので C = ε0·H/D に極めて高精度で一致する。"""
    project = _plates_project()
    mesh = generate_mesh(project)
    sol = solve(project, mesh)

    c_exact = EPS0 * H / D
    assert sol.capacitance == pytest.approx(c_exact, rel=1e-10)

    # 電極ごとの電荷: 高電位側 (右辺 V1) と低電位側 (左辺 0V) で符号反転しほぼ相殺する
    charges = {label: (voltage, q) for label, voltage, q in sol.charges}
    assert set(charges) == {"edge1", "edge3"}
    v_hi, q_hi = charges["edge1"]
    v_lo, q_lo = charges["edge3"]
    assert v_hi == V1 and v_lo == 0.0
    assert q_hi > 0.0 > q_lo
    assert abs(q_hi + q_lo) < 1e-8 * abs(q_hi)


def test_dielectric_capacitance_scales_with_eps_r():
    """domain 全域を εr=4 の誘電体で満たすと静電容量がちょうど4倍になる。"""
    project = _plates_project(eps_r=4.0)
    mesh = generate_mesh(project)
    sol = solve(project, mesh)

    c_exact = 4.0 * EPS0 * H / D
    assert sol.capacitance == pytest.approx(c_exact, rel=1e-10)


def test_coaxial_cylinder_capacitance():
    """同軸円筒 (rz_x0): 内導体 (conductor 領域、半径a・長さL) と外周 r=b の Dirichlet。

    C = 2πε0·L / ln(b/a) に相対誤差 3% 以内で一致する
    (メッシュの離散化誤差のみで、フリンジ電界は上下 Neumann により無視できる)。
    """
    a, b, length = 0.01, 0.05, 0.1
    project = Project.model_validate(
        {
            "coord": "rz_x0",
            "geometry": {
                "domain": {"polygon": [[0, 0], [b, 0], [b, length], [0, length]]},
                "regions": [
                    {
                        "id": "inner",
                        "type": "conductor",
                        "polygon": [[0, 0], [a, 0], [a, length], [0, length]],
                        "voltage": V1,
                    }
                ],
                "boundaries": [
                    {"edges": [1], "voltage": 0.0},  # 外周 r=b
                ],
            },
            "mesh": {"size": 0.002},
        }
    )
    mesh = generate_mesh(project)
    sol = solve(project, mesh)

    c_exact = 2.0 * math.pi * EPS0 * length / math.log(b / a)
    assert sol.capacitance == pytest.approx(c_exact, rel=0.03)

    charges = {label: (voltage, q) for label, voltage, q in sol.charges}
    assert set(charges) == {"inner", "edge1"}
    assert charges["inner"][0] == V1
    assert charges["edge1"][0] == 0.0


def test_capacitance_is_none_for_three_levels_or_charge():
    """3水準の電極、または空間電荷 (ρ≠0) がある場合は静電容量が定義されず None になる。"""
    # 3電極 (3電位水準)
    project_3lv = Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [D, 0], [D, H], [0, H]]},
                "boundaries": [
                    {"edges": [3], "voltage": 0.0},
                    {"edges": [0], "voltage": 50.0},
                    {"edges": [1], "voltage": V1},
                ],
            },
            "mesh": {"size": 0.01},
        }
    )
    mesh_3lv = generate_mesh(project_3lv)
    sol_3lv = solve(project_3lv, mesh_3lv)
    assert sol_3lv.capacitance is None
    assert len(sol_3lv.charges) == 3

    # 2電極だが空間電荷 ρ≠0 がある
    project_rho = Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [D, 0], [D, H], [0, H]]},
                "regions": [
                    {
                        "id": "sp",
                        "type": "charge",
                        "polygon": [[0.03, 0.01], [0.07, 0.01], [0.07, 0.04], [0.03, 0.04]],
                        "rho": 1e-6,
                    }
                ],
                "boundaries": [
                    {"edges": [3], "voltage": 0.0},
                    {"edges": [1], "voltage": V1},
                ],
            },
            "mesh": {"size": 0.01},
        }
    )
    mesh_rho = generate_mesh(project_rho)
    sol_rho = solve(project_rho, mesh_rho)
    assert sol_rho.capacitance is None
    assert len(sol_rho.charges) == 2
