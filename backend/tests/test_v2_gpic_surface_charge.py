"""v2 GPU PIC: 誘電体の表面電荷は表面の後ろ (誘電体の中の節点) に置く (prompts/138)。

GEC のサンプルを 2300 RF 周期ほど走らせると暴走した原因の回帰。誘電体リングと接地シールドが下面をそろえて
接する所 (格子にそろわない) で、誘電体に当たった電子の電荷を衝突点のセルの 4 隅へ双一次で配ると、大半が表面の
すぐ前の気体の節点 (シールドの下で、シールドの電位に強く結ばれた節点) に載り、電位の極小が表面ではなく気体の中に
できた。イオンがそこに閉じ込められて溜まり、局所の電離が増えて、ゆっくり育って暴走した。

1. 接合部の最後の誘電体セルに電子を当てる: 表面電荷は誘電体の中の節点だけに載り (総量は厳密)、電位の極小は
   誘電体の中にある (一様格子・AMR、平面・軸対称)
2. 誘電体の二次電子 (γ = 1): イオンの電荷と放出した電子の分 (+2e·w) も誘電体の中の節点だけに載る
3. 1D 相当の誘電体の面の電位が面電荷の 1D の解析解と 5% 以内 (電荷を最大 1 セル後ろに置く分の 1 次の誤差、ここでは
   約 3%。旧規則は約 1.5% だったが、平らな面でも電位の極小が気体の節点にあった)
4. 再格子化の移し替え (deposit_points の allowed): 許された隅だけに規格化して配り、総和を保存
5. 格子より薄い誘電体は警告する (GEC のサンプルと、格子にそろった面は警告しない)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from es_sim.device import cuda_available
from es_sim.schema import Project

needs_cuda = pytest.mark.skipif(not cuda_available(), reason="CUDA (CuPy) が使えない環境")
QE = 1.602176634e-19
L = 0.01
Y_FACE = 0.00604    # 誘電体と導体の下面 (格子 0.5 mm の節点 y = 6.0 mm のすぐ上)
X_JOIN = 0.00498    # 誘電体と導体の接合 (節点 x = 5.0 mm は接合のすぐ先、導体の下の気体)
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _junction(coord: str = "xy", amr: dict | None = None, see_gamma: float = 0.0) -> Project:
    """誘電体 (εr 2.1) と接地導体が下面をそろえて接する箱 (GEC の上のリングとシールドの接合と同じ並び)。"""
    d = {
        "coord": coord,
        "geometry": {
            "domain": {"polygon": [[0, 0], [L, 0], [L, L], [0, L]]},
            "regions": [
                {"id": "ring", "type": "dielectric", "eps_r": 2.1, "see_gamma": see_gamma,
                 "polygon": [[0.0021, Y_FACE], [X_JOIN, Y_FACE], [X_JOIN, L], [0.0021, L]]},
                {"id": "shield", "type": "conductor", "voltage": 0.0,
                 "polygon": [[X_JOIN, Y_FACE], [0.0081, Y_FACE], [0.0081, L], [X_JOIN, L]]},
            ],
            "boundaries": [{"edges": [0, 1, 2], "type": "dirichlet", "voltage": 0.0},
                           {"edges": [3], "type": "symmetry" if coord != "xy" else "dirichlet", "voltage": 0.0}],
        },
        "mesh": {"size": 5e-4, "mode": "cartesian"},
        "pic": {"n_macro": 1, "dt": 1e-11, "n_steps": 2, "frame_every": 1000, "radial_weighting": False,
                "see_energy_ev": 2.0},
    }
    if amr is not None:
        d["mesh"]["amr"] = amr
    return Project.model_validate(d)


_AMR = {"max_level": 1, "blocking_factor": 4, "refine_boundaries": True, "buffer_cells": 2}


def _node_xy(sim) -> tuple[np.ndarray, np.ndarray]:
    if sim.amr is not None:
        return sim.amr.xy[:, 0], sim.amr.xy[:, 1]
    g = sim.grid
    X, Y = np.meshgrid(g.xs, g.ys)
    return X.ravel(), Y.ravel()


def _fixed(sim) -> np.ndarray:
    from es_sim.eb.build import MASK_FIXED

    if sim.amr is not None:
        return sim.amr.op.fixed_group >= 0
    return (sim.op.mask == MASK_FIXED).ravel()


def _hit_last_dielectric_cell(sim, species: str, n: int = 400, w: float = 1e6) -> float:
    """誘電体の最後のセル (x 4.5〜4.98 mm) の下面へ、すぐ下から粒子を打ち込む。戻り値: Σw。"""
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0045, X_JOIN, n)
    y = np.full(n, Y_FACE - 1e-5)
    z = np.zeros(n)
    sim.set_particles(species, x=x, y=y, vx=z, vy=np.full(n, 2e6), vz=z, w=np.full(n, w))
    return n * w


@needs_cuda
@pytest.mark.parametrize("coord", ["xy", "rz_x0"])
@pytest.mark.parametrize("grid", ["uniform", "amr"])
def test_surface_charge_sits_behind_the_surface_at_a_junction(coord, grid):
    from es_sim.gpic import GpuPicSimulation

    sim = GpuPicSimulation(_junction(coord, _AMR if grid == "amr" else None))
    assert (sim.amr is not None) == (grid == "amr")
    wsum = _hit_last_dielectric_cell(sim, "electron")
    sim.step()   # 全部が誘電体に当たって表面電荷になる
    sim._flush_history()
    assert sim.history["wall_e"][-1] == 400
    sim.step()   # 粒子の無い状態の電位 (表面電荷だけ)
    q = sim._q_surf.get().ravel()
    phi = sim._phi.get().ravel()
    x, y = _node_xy(sim)
    inside = ~sim.model.gas_at(x, y) & ~_fixed(sim)
    charged = q != 0.0
    assert charged.any() and np.all(inside[charged]), "表面電荷が誘電体の外の節点に載った"
    assert float(q.sum()) * sim._two_pi == pytest.approx(-QE * wsum, rel=1e-12)
    # 電位の極小 (負の表面電荷) は誘電体の中。気体の節点は表面の電荷より高い電位にいる (閉じ込めの極小が無い)
    free = ~_fixed(sim)
    k = int(np.argmin(np.where(free, phi, np.inf)))
    assert inside[k], f"電位の極小が気体の節点 ({x[k]:.5f}, {y[k]:.5f}) にある"
    gas = sim.model.gas_at(x, y) & free
    near = gas & (np.abs(x - X_JOIN) < 0.0015) & (np.abs(y - Y_FACE) < 0.0015)
    assert near.any() and float(np.min(phi[near])) > float(phi[k])


@needs_cuda
@pytest.mark.parametrize("grid", ["uniform", "amr"])
def test_dielectric_see_charge_sits_behind_the_surface(grid):
    from es_sim.gpic import GpuPicSimulation

    sim = GpuPicSimulation(_junction("xy", _AMR if grid == "amr" else None, see_gamma=1.0))
    wsum = _hit_last_dielectric_cell(sim, "ion")
    sim.step()
    sim._flush_history()
    assert sim.history["wall_i"][-1] == 400 and sim.history["see_events"][-1] == 400
    q = sim._q_surf.get().ravel()
    x, y = _node_xy(sim)
    inside = ~sim.model.gas_at(x, y) & ~_fixed(sim)
    assert np.all(inside[q != 0.0])
    # イオンの電荷 +e·w と、放出した電子の分 +e·w
    assert float(q.sum()) == pytest.approx(2.0 * QE * wsum, rel=1e-12)
    e = sim.get_particles("electron")
    assert len(e["x"]) == 400 and np.all(sim.model.gas_at(e["x"], e["y"]))


@needs_cuda
def test_surface_potential_of_a_slab_matches_1d_analytic():
    """接地 (x = 0) | 気体 a | 誘電体 (εr 2.1、x = L の接地まで) の 1D 相当の箱で、面に当てた電子の電位。"""
    from es_sim.geom.model import EPS0
    from es_sim.gpic import GpuPicSimulation

    lx, h, a, eps_r = 0.02, 0.004, 0.00404, 2.1
    sim = GpuPicSimulation(Project.model_validate({
        "geometry": {
            "domain": {"polygon": [[0, 0], [lx, 0], [lx, h], [0, h]]},
            "regions": [{"id": "d", "type": "dielectric", "eps_r": eps_r,
                         "polygon": [[a, 0.0], [lx, 0.0], [lx, h], [a, h]]}],
            "boundaries": [{"edges": [1, 3], "type": "dirichlet", "voltage": 0.0}],
        },
        "mesh": {"size": 5e-4, "mode": "cartesian"},
        "pic": {"n_macro": 1, "dt": 1e-11, "n_steps": 2, "frame_every": 1000, "reflect_edges": [0, 2]},
    }))
    n = 800
    z = np.zeros(n)
    sim.set_particles("electron", x=np.full(n, a - 1e-5), y=(np.arange(n) + 0.5) * h / n, vx=np.full(n, 2e6),
                      vy=z, vz=z, w=np.full(n, 1e6))
    sim.step()
    sim.step()
    g = sim.grid
    phi = sim._phi.get()[g.ny // 2]
    sigma = float(sim._q_surf.get().sum()) / h
    assert sigma == pytest.approx(-QE * 1e6 * n / h, rel=1e-12)
    ra, rb = a / EPS0, (lx - a) / (eps_r * EPS0)
    v_exact = sigma * ra * rb / (ra + rb)
    i = int(np.floor(a / g.dx))
    dg, ds = a - i * g.dx, (i + 1) * g.dx - a
    # 面の電位: 気体の節点 P と誘電体の節点 S の間で電束が続くように内挿
    v_face = (phi[i] / dg + eps_r * phi[i + 1] / ds) / (1.0 / dg + eps_r / ds)
    assert v_face == pytest.approx(v_exact, rel=0.05)
    assert phi[i + 1] < phi[i]   # 極小は誘電体の側


def test_regrid_deposit_restricted_to_allowed_nodes():
    from es_sim.amr import AmrHierarchy, AmrSpec, build_pic_layout
    from es_sim.amr.pic_layout import deposit_points
    from es_sim.eb.grid import make_grid
    from es_sim.geom.model import GeometryModel

    p = _junction("rz_x0")
    model = GeometryModel(p)
    grid = make_grid(model.domain, float(p.mesh.size))
    lay = build_pic_layout(model, AmrHierarchy(model, grid, AmrSpec(max_level=1, blocking_factor=4)))
    allowed = ~model.gas_at(lay.xy[:, 0], lay.xy[:, 1]) & (lay.op.fixed_group < 0)
    rng = np.random.default_rng(5)
    # 下面の上の点 (隅に誘電体の節点がある) と、気体の奥の点 (無い → 4 隅へ)
    xs = np.concatenate([rng.uniform(0.0022, X_JOIN, 50), rng.uniform(0.001, 0.009, 20)])
    ys = np.concatenate([np.full(50, Y_FACE), rng.uniform(0.001, 0.004, 20)])
    qs = rng.uniform(-1.0, 1.0, xs.size)
    out = deposit_points(lay, xs, ys, qs, allowed=allowed)
    assert float(out.sum()) == pytest.approx(float(qs.sum()), rel=1e-12)
    face = deposit_points(lay, xs[:50], ys[:50], qs[:50], allowed=allowed)
    assert np.all(allowed[face != 0.0])
    deep = deposit_points(lay, xs[50:], ys[50:], qs[50:], allowed=allowed)
    np.testing.assert_allclose(deep, deposit_points(lay, xs[50:], ys[50:], qs[50:]), rtol=0, atol=1e-15)


@needs_cuda
def test_thin_dielectric_warns_and_gec_sample_does_not():
    from es_sim.gpic import GpuPicSimulation

    d = json.loads(_junction("xy").model_dump_json())
    # 格子 (0.5 mm) より薄い 0.2 mm の誘電体の膜 (気体の中、節点を含まない)
    d["geometry"]["regions"] = [{"id": "film", "type": "dielectric", "eps_r": 3.0,
                                 "polygon": [[0.002, 0.0031], [0.008, 0.0031], [0.008, 0.0033], [0.002, 0.0033]]}]
    sim = GpuPicSimulation(Project.model_validate(d))
    assert any("誘電体が格子より薄い" in w for w in sim.warnings)
    # 格子にそろった面 (面の上の節点に置ける) は警告しない
    d["geometry"]["regions"] = [{"id": "slab", "type": "dielectric", "eps_r": 3.0,
                                 "polygon": [[0.002, 0.006], [0.008, 0.006], [0.008, 0.01], [0.002, 0.01]]}]
    sim = GpuPicSimulation(Project.model_validate(d))
    assert not any("誘電体が格子より薄い" in w for w in sim.warnings)
    gec = json.loads((EXAMPLES / "gec_cell.json").read_text(encoding="utf-8"))
    gec["pic"]["n_macro"] = 1000
    sim = GpuPicSimulation(Project.model_validate(gec))
    assert not any("誘電体が格子より薄い" in w for w in sim.warnings)
