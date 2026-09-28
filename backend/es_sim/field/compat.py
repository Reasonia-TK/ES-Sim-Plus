"""v2 直交格子エンジンの結果を v1 API の形 (三角形メッシュ上の値) へ変換する (prompts/119)。

既存の v1 UI (frontend/) は三角形メッシュ (nodes/triangles) の節点値・要素値を描く。
v2 の直交格子は各セルを市松の対角線で 2 三角形に分割した「表示用メッシュ」として渡す
(gpic.geometry.ParticleGeometry.display_mesh と同じ。導体内のセルは穴)。
"""

from __future__ import annotations

import numpy as np

from ..eb.grid import make_grid
from ..geom.model import GeometryModel
from ..gpic.geometry import ParticleGeometry, triangle_gradients
from ..schema import ElectrodeCharge, MeshResult, Project, SolveResult
from .electrostatic import StaticSolution, solve_electrostatic


def _display(project: Project):
    model = GeometryModel(project)
    grid = make_grid(model.domain, float(project.mesh.size))
    return model, grid, ParticleGeometry(model, grid).display_mesh()


def cartesian_mesh_result(project: Project) -> MeshResult:
    """/mesh 用: v2 の格子を三角形分割した表示用メッシュ。"""
    _, _, dm = _display(project)
    return MeshResult(
        nodes=[tuple(p) for p in dm.nodes.tolist()],
        triangles=[tuple(t) for t in dm.triangles.tolist()],
        region_of_triangle=dm.tri_region.tolist(),
    )


def cartesian_solve(project: Project, device: str | None = None) -> tuple[SolveResult, StaticSolution]:
    """/solve 用: v2 EB Poisson (GMG-PCG) で解き、v1 の SolveResult 形式で返す。"""
    model, grid, dm = _display(project)
    sol = solve_electrostatic(project, grid=grid, device=device)
    v = sol.phi.ravel()
    grad = triangle_gradients(dm.nodes, dm.triangles, v)
    e_tri = -grad
    e_abs = np.hypot(e_tri[:, 0], e_tri[:, 1])
    res = SolveResult(
        mesh=MeshResult(
            nodes=[tuple(p) for p in dm.nodes.tolist()],
            triangles=[tuple(t) for t in dm.triangles.tolist()],
            region_of_triangle=dm.tri_region.tolist(),
        ),
        v=v.tolist(),
        e_field=[tuple(e) for e in e_tri.tolist()],
        v_min=float(v.min()),
        v_max=float(v.max()),
        e_abs_max=float(e_abs.max()) if e_abs.size else 0.0,
        energy=sol.energy,
        charges=[ElectrodeCharge(label=lab, voltage=vv, q=q) for lab, vv, q in sol.charges],
        capacitance=sol.capacitance,
    )
    del model
    return res, sol


def cartesian_profile(project: Project, p1, p2, n: int, device: str | None = None):
    """/profile 用: 線分 p1→p2 上の V と |E| (節点値の双一次補間)。domain 外・導体内は NaN。"""
    model = GeometryModel(project)
    grid = make_grid(model.domain, float(project.mesh.size))
    sol = solve_electrostatic(project, grid=grid, device=device)
    t = np.linspace(0.0, 1.0, int(n))
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    pts = p1[None, :] + t[:, None] * (p2 - p1)[None, :]
    s = t * float(np.linalg.norm(p2 - p1))
    fx = (pts[:, 0] - grid.x0) / grid.dx
    fy = (pts[:, 1] - grid.y0) / grid.dy
    inside = (fx >= 0) & (fx <= grid.nx) & (fy >= 0) & (fy <= grid.ny)
    i = np.clip(np.floor(fx).astype(np.int64), 0, grid.nx - 1)
    j = np.clip(np.floor(fy).astype(np.int64), 0, grid.ny - 1)
    wx = np.clip(fx - i, 0.0, 1.0)
    wy = np.clip(fy - j, 0.0, 1.0)

    def interp(f):
        return ((1 - wx) * (1 - wy) * f[j, i] + wx * (1 - wy) * f[j, i + 1]
                + (1 - wx) * wy * f[j + 1, i] + wx * wy * f[j + 1, i + 1])

    v = interp(sol.phi)
    e_abs = np.hypot(interp(sol.ex), interp(sol.ey))
    in_cond = model.classify_conductor(pts[:, 0], pts[:, 1], 0.0) >= 0
    bad = ~inside | in_cond
    v[bad] = np.nan
    e_abs[bad] = np.nan
    return s, v, e_abs
