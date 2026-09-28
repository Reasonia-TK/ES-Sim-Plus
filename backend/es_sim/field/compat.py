"""v2 直交格子エンジンの結果を v1 API の形 (三角形メッシュ上の値) へ変換する (prompts/119, 121)。

既存の v1 UI (frontend/) は三角形メッシュ (nodes/triangles) の節点値・要素値を描く。
v2 の直交格子は各セルを市松の対角線で 2 三角形に分割した「表示用メッシュ」として渡す
(gpic.geometry.ParticleGeometry.display_mesh と同じ。導体内のセルは穴)。

mesh.amr の局所細分化 (prompts/121) が有効なら、全レベルの葉セルを同じく 2 三角形に分割した
メッシュで合成格子の解 (es_sim.amr、CPU の AMG-CG) を返す。設定上は有効でも実際に細分化が
起きない (タグが無い) 場合は一様格子 (GMG-PCG、GPU 可) に戻す。
"""

from __future__ import annotations

import numpy as np

from ..amr import AmrHierarchy, AmrStaticSolution, build_hierarchy, display_mesh, solve_electrostatic_amr
from ..eb.grid import make_grid
from ..geom.model import GeometryModel
from ..gpic.geometry import ParticleGeometry, triangle_gradients
from ..schema import ElectrodeCharge, MeshResult, Project, SolveResult
from .electrostatic import StaticSolution, solve_electrostatic


def amr_hierarchy(project: Project, model: GeometryModel | None = None) -> AmrHierarchy | None:
    """mesh.amr が実際に細分化を起こす (または適応細分化が有効な) ならその階層、それ以外は None。

    適応細分化 (adaptive) は求解時に階層を作り直すので、/mesh が返すのは適応前の階層。
    """
    amr = project.mesh.amr
    if amr is None or (amr.max_level <= 0 and not amr.regions):
        return None
    hier = build_hierarchy(project, model=model)
    if hier.max_level > 0 or (amr.adaptive and amr.max_level > 0):
        return hier
    return None


def _mesh_result(nodes: np.ndarray, triangles: np.ndarray, tri_region: np.ndarray) -> MeshResult:
    return MeshResult(
        nodes=[tuple(p) for p in nodes.tolist()],
        triangles=[tuple(t) for t in triangles.tolist()],
        region_of_triangle=tri_region.tolist(),
    )


def cartesian_mesh_result(project: Project) -> MeshResult:
    """/mesh 用: v2 の格子 (AMR なら全レベルの葉セル) を三角形分割した表示用メッシュ。"""
    model = GeometryModel(project)
    hier = amr_hierarchy(project, model)
    if hier is not None:
        nodes, tris, tri_region, _ = display_mesh(model, hier)
        return _mesh_result(nodes, tris, tri_region)
    dm = ParticleGeometry(model, make_grid(model.domain, float(project.mesh.size))).display_mesh()
    return _mesh_result(dm.nodes, dm.triangles, dm.tri_region)


def cartesian_solve(project: Project, device: str | None = None
                    ) -> tuple[SolveResult, StaticSolution | AmrStaticSolution]:
    """/solve 用: v2 EB Poisson (一様格子は GMG-PCG、AMR は合成格子の AMG-CG) を v1 の SolveResult で返す。"""
    model = GeometryModel(project)
    hier = amr_hierarchy(project, model)
    if hier is not None:
        sol = solve_electrostatic_amr(project, hier=hier, device=device)
        nodes, tris, tri_region, v, e_tri = sol.nodes, sol.triangles, sol.tri_region, sol.phi_display, sol.e_tri
    else:
        grid = make_grid(model.domain, float(project.mesh.size))
        dm = ParticleGeometry(model, grid).display_mesh()
        sol = solve_electrostatic(project, grid=grid, device=device)
        nodes, tris, tri_region = dm.nodes, dm.triangles, dm.tri_region
        v = sol.phi.ravel()
        e_tri = -triangle_gradients(nodes, tris, v)
    e_abs = np.hypot(e_tri[:, 0], e_tri[:, 1])
    res = SolveResult(
        mesh=_mesh_result(nodes, tris, tri_region),
        v=v.tolist(),
        e_field=[tuple(e) for e in e_tri.tolist()],
        v_min=float(v.min()),
        v_max=float(v.max()),
        e_abs_max=float(e_abs.max()) if e_abs.size else 0.0,
        energy=sol.energy,
        charges=[ElectrodeCharge(label=lab, voltage=vv, q=q) for lab, vv, q in sol.charges],
        capacitance=sol.capacitance,
    )
    return res, sol


def cartesian_profile(project: Project, p1, p2, n: int, device: str | None = None):
    """/profile 用: 線分 p1→p2 上の V と |E| (セル内の双一次補間)。domain 外・導体内は NaN。"""
    model = GeometryModel(project)
    t = np.linspace(0.0, 1.0, int(n))
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    pts = p1[None, :] + t[:, None] * (p2 - p1)[None, :]
    s = t * float(np.linalg.norm(p2 - p1))
    hier = amr_hierarchy(project, model)
    if hier is not None:
        sol = solve_electrostatic_amr(project, hier=hier, device=device)
        v, ex, ey = sol.sample(pts[:, 0], pts[:, 1])
        e_abs = np.hypot(ex, ey)
        inside = ~np.isnan(v)
    else:
        grid = make_grid(model.domain, float(project.mesh.size))
        sol = solve_electrostatic(project, grid=grid, device=device)
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
