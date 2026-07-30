"""2D シースエッジ評価ライン (SheathLine) の schema テスト (prompts/98)。

計算そのものはフロント側 (frontend/src/sheath.ts) で行うため、backend は
永続化 (schema + 最大本数の validator) のみを担う。
"""

import pytest
from pydantic import ValidationError

from es_sim.schema import Project

L = 0.01
HGT = 0.01


def _project(pic_extra: dict) -> Project:
    pic = {
        "dt": 2e-9,
        "n_steps": 10,
        "n_macro": 10,
        "frame_every": 10,
        **pic_extra,
    }
    return Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [L, 0], [L, HGT], [0, HGT]]},
                "boundaries": [
                    {"edges": [3], "voltage": 0.0},
                    {"edges": [1], "voltage": -100.0},
                    {"edges": [0, 2], "type": "symmetry"},
                ],
            },
            "mesh": {"size": 8e-4},
            "pic": pic,
        }
    )


def test_sheath_lines_default_empty():
    """sheath_lines 未指定の既存プロジェクトは空リストで読み込める (後方互換)。"""
    project = _project({})
    assert project.pic.sheath_lines == []


def test_sheath_lines_roundtrip():
    lines = [
        {"p1": [0.0, 0.0], "p2": [L, 0.0], "label": "S1"},
        {"p1": [0.0, HGT], "p2": [L, HGT], "label": ""},
    ]
    project = _project({"sheath_lines": lines})
    assert len(project.pic.sheath_lines) == 2
    assert project.pic.sheath_lines[0].label == "S1"
    assert project.pic.sheath_lines[1].label == ""
    assert project.pic.sheath_lines[0].p1 == (0.0, 0.0)
    assert project.pic.sheath_lines[0].p2 == (L, 0.0)


def test_more_than_four_sheath_lines_rejected():
    lines = [{"p1": [0.0, y], "p2": [L, y]} for y in (0.001, 0.002, 0.003, 0.004, 0.005)]
    with pytest.raises(ValidationError, match="4"):
        _project({"sheath_lines": lines})
    # 4個ちょうどは通る
    project = _project({"sheath_lines": lines[:4]})
    assert len(project.pic.sheath_lines) == 4
