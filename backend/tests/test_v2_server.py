"""v2 エンジンのサーバー統合 (mesh.mode="cartesian"、prompts/119)。

既存 UI (v1 API) からそのまま v2 を使えることを確かめる:
/health の v2 情報、/mesh の表示用三角形メッシュ、/solve の SolveResult (解析解と一致)、
/profile、/ws/pic (GPU があれば started/frame/done の流れ)、バッチの v2 振り分け。
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from es_sim.device import cuda_available
from es_sim.server import app

EPS0 = 8.8541878128e-12


def _plates(mode: str = "cartesian") -> dict:
    return {
        "version": 1,
        "unit": "m",
        "geometry": {
            "domain": {"polygon": [[0, 0], [0.1, 0], [0.1, 0.05], [0, 0.05]]},
            "regions": [{"id": "d", "type": "dielectric", "eps_r": 4.0,
                         "polygon": [[0.04, 0.01], [0.06, 0.01], [0.06, 0.04], [0.04, 0.04]]}],
            "boundaries": [
                {"edges": [3], "type": "dirichlet", "voltage": 0.0},
                {"edges": [1], "type": "dirichlet", "voltage": 100.0},
            ],
        },
        "mesh": {"size": 0.002, "mode": mode},
    }


def test_health_reports_v2_device():
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "v2" in body and "cuda" in body["v2"]
    assert body["gpu"] == bool(body["v2"]["cuda"]) or body["gpu"]


def test_mesh_endpoint_returns_triangulated_grid():
    r = TestClient(app).post("/mesh", json=_plates())
    assert r.status_code == 200
    m = r.json()
    nodes = np.asarray(m["nodes"])
    tris = np.asarray(m["triangles"])
    assert tris.shape[1] == 3 and len(tris) % 2 == 0
    assert len(m["region_of_triangle"]) == len(tris)
    assert nodes[:, 0].min() == pytest.approx(0.0) and nodes[:, 0].max() == pytest.approx(0.1)
    assert 0 in set(m["region_of_triangle"])       # 誘電体セル
    assert -1 in set(m["region_of_triangle"])      # 真空セル


def test_solve_endpoint_cartesian_matches_v1_shape_and_physics():
    client = TestClient(app)
    r2 = client.post("/solve", json=_plates("cartesian"))
    r1 = client.post("/solve", json=_plates("unstructured"))
    assert r2.status_code == 200 and r1.status_code == 200
    b2, b1 = r2.json(), r1.json()
    assert set(b2) == set(b1)                       # v1 と同じ SolveResult の形
    assert len(b2["v"]) == len(b2["mesh"]["nodes"])
    assert len(b2["e_field"]) == len(b2["mesh"]["triangles"])
    assert b2["v_min"] == pytest.approx(0.0, abs=1e-9) and b2["v_max"] == pytest.approx(100.0)
    assert b2["capacitance"] == pytest.approx(b1["capacitance"], rel=5e-3)
    assert b2["energy"] == pytest.approx(b1["energy"], rel=5e-3)
    assert [c["label"] for c in b2["charges"]] == [c["label"] for c in b1["charges"]]


def test_profile_endpoint_cartesian():
    body = {"project": _plates(), "p1": [0.0, 0.005], "p2": [0.1, 0.005], "n": 51}
    r = TestClient(app).post("/profile", json=body)
    assert r.status_code == 200
    p = r.json()
    v = np.asarray(p["v"], dtype=float)
    assert len(v) == 51 and v[0] == pytest.approx(0.0, abs=1e-6) and v[-1] == pytest.approx(100.0, abs=1e-6)
    assert np.all(np.diff(v) > 0)                  # y=5 mm は誘電体の外: 単調増加


def _plates_amr(max_level: int = 2) -> dict:
    p = _plates()
    p["mesh"]["amr"] = {"max_level": max_level}
    return p


def _triangle_areas(mesh: dict) -> np.ndarray:
    P = np.asarray(mesh["nodes"])[np.asarray(mesh["triangles"])]
    return 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                        - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))


def test_mesh_endpoint_amr_refines_near_dielectric():
    """mesh.amr (prompts/121): 誘電体の境界近傍が 2 レベル細かい三角形になる。"""
    client = TestClient(app)
    m0 = client.post("/mesh", json=_plates()).json()
    m2 = client.post("/mesh", json=_plates_amr(2)).json()
    area = _triangle_areas(m2)
    assert area.sum() == pytest.approx(0.1 * 0.05, rel=1e-12)
    assert area.min() == pytest.approx(area.max() / 16, rel=1e-6)     # 格子幅 1/4 → 面積 1/16
    assert len(m2["triangles"]) > len(m0["triangles"])
    assert len(m2["region_of_triangle"]) == len(m2["triangles"])
    assert {0, -1} <= set(m2["region_of_triangle"])


def test_solve_endpoint_amr_is_closer_to_fine_grid():
    client = TestClient(app)
    b = client.post("/solve", json=_plates_amr(2)).json()
    coarse = client.post("/solve", json=_plates()).json()
    fine = _plates()
    fine["mesh"]["size"] = 0.0005
    f = client.post("/solve", json=fine).json()
    assert len(b["v"]) == len(b["mesh"]["nodes"])
    assert len(b["e_field"]) == len(b["mesh"]["triangles"])
    assert b["v_min"] == pytest.approx(0.0, abs=1e-9) and b["v_max"] == pytest.approx(100.0)
    assert [c["label"] for c in b["charges"]] == [c["label"] for c in coarse["charges"]]
    err_amr = abs(b["capacitance"] - f["capacitance"])
    err_coarse = abs(coarse["capacitance"] - f["capacitance"])
    assert err_amr < err_coarse / 2
    assert b["energy"] == pytest.approx(0.5 * b["capacitance"] * 100.0**2, rel=1e-9)


def test_profile_endpoint_amr():
    body = {"project": _plates_amr(2), "p1": [0.0, 0.005], "p2": [0.1, 0.005], "n": 51}
    r = TestClient(app).post("/profile", json=body)
    assert r.status_code == 200
    v = np.asarray(r.json()["v"], dtype=float)
    assert v[0] == pytest.approx(0.0, abs=1e-6) and v[-1] == pytest.approx(100.0, abs=1e-6)
    assert np.all(np.diff(v) > 0)


def test_amr_without_tags_falls_back_to_uniform_grid():
    """細分化の対象 (境界・領域) が無ければ一様格子 (GMG-PCG、GPU 可) で解く。"""
    p = _plates_amr(2)
    p["geometry"]["regions"] = []
    q = _plates()
    q["geometry"]["regions"] = []
    client = TestClient(app)
    assert client.post("/mesh", json=p).json() == client.post("/mesh", json=q).json()


def test_solve_endpoint_cartesian_rejects_non_rectangular_domain():
    p = _plates()
    p["geometry"]["domain"]["polygon"] = [[0, 0], [0.1, 0], [0.05, 0.05]]
    r = TestClient(app).post("/solve", json=p)
    assert r.status_code == 422
    assert "矩形" in r.json()["detail"]


@pytest.mark.skipif(not cuda_available(), reason="CUDA (CuPy) が使えない環境")
def test_ws_pic_cartesian_runs_gpu_engine():
    p = _plates()
    p["geometry"]["regions"] = []
    p["pic"] = {
        "initial_plasma": {"density": 1e14, "te_ev": 1.0, "ti_ev": 0.03, "ion_mass_amu": 40.0, "seed": 1},
        "n_macro": 20000,
        "n_steps": 200,
        "frame_every": 50,
        "phase_bins": 0,
    }
    client = TestClient(app)
    with client.websocket_connect("/ws/pic") as ws:
        ws.send_text(json.dumps({"cmd": "start", "project": p}))
        msgs = []
        while True:
            m = ws.receive_json()
            msgs.append(m)
            if m["type"] in ("done", "error"):
                break
    kinds = [m["type"] for m in msgs]
    assert kinds[0] == "started" and kinds[-1] == "done", msgs[-1]
    started = msgs[0]
    assert len(started["mesh"]["triangles"]) > 0
    done = msgs[-1]
    assert len(done["history"]["t"]) == 200
    assert done["fields"] is not None
    assert len(done["fields"]["n_e"]) == len(started["mesh"]["nodes"])
    assert len(done["fields"]["e_abs"]) == len(started["mesh"]["triangles"])
    assert any(m["type"] == "frame" for m in msgs)
