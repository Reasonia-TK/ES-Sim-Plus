"""AMR (合成格子) 静電場ソルバーの高レベル API (prompts/121)。

field.electrostatic.solve_electrostatic (一様格子) の AMR 版。v1 fem.solve と同じ量
(電位・電場・エネルギー・電極電荷・容量) を返し、表示用に全レベルの葉セルを 2 三角形に
分割したメッシュを持つ (ぶら下がり節点は拘束値を持つので粗いセルの三角形と値が連続する)。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..eb.grid import make_grid
from ..field.electrostatic import group_voltages
from ..geom.model import GeometryModel
from ..schema import Project
from .composite import AmrSolveInfo, CompositeOperator, _canon_key, _coords, assemble_composite, energy_and_charges, solve_composite
from .hierarchy import AmrHierarchy, AmrSpec


@dataclass
class AmrStaticSolution:
    hier: AmrHierarchy
    op: CompositeOperator
    phi: np.ndarray               # (N,) 正準節点の電位
    nodes: np.ndarray             # (Nd, 2) 表示用節点 (周期の両側を別節点として持つ)
    phi_display: np.ndarray       # (Nd,)
    triangles: np.ndarray         # (M, 3) 表示用三角形
    tri_region: np.ndarray        # (M,) geometry.regions の番号 (-1 = 真空)
    e_tri: np.ndarray             # (M, 2) 三角形ごとの E = −∇φ
    energy: float
    charges: list[tuple[str, float, float]]
    capacitance: float | None
    info: AmrSolveInfo
    timing: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_unknowns(self) -> int:
        return self.op.n_unknowns

    @property
    def leaf_counts(self) -> list[int]:
        return self.hier.n_leaf_cells()

    def sample(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """任意点の (φ, Ex, Ey): 点を含む葉セル (最も細かいレベル) の双一次補間。domain 外は NaN。"""
        hier = self.hier
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        phi = np.full(x.shape, np.nan)
        ex = np.full(x.shape, np.nan)
        ey = np.full(x.shape, np.nan)
        b = hier.base
        inside = (x >= b.x0) & (x <= b.x1) & (y >= b.y0) & (y <= b.y1)
        todo = inside.copy()
        L = hier.max_level
        for lvl in range(L, -1, -1):
            if not np.any(todo):
                break
            g = hier.level_grid(lvl)
            leaf = hier.leaf_mask(lvl)
            idx = np.nonzero(todo)[0]
            fx = (x[idx] - g.x0) / g.dx
            fy = (y[idx] - g.y0) / g.dy
            i = np.clip(np.floor(fx).astype(np.int64), 0, g.nx - 1)
            j = np.clip(np.floor(fy).astype(np.int64), 0, g.ny - 1)
            ok = leaf[j, i]
            if not np.any(ok):
                continue
            idx, i, j = idx[ok], i[ok], j[ok]
            wx = np.clip(fx[ok] - i, 0.0, 1.0)
            wy = np.clip(fy[ok] - j, 0.0, 1.0)
            s = L - lvl
            corners = []
            for a, bb in ((0, 0), (1, 0), (0, 1), (1, 1)):
                k = _canon_key(hier, (i + a) << s, (j + bb) << s)
                corners.append(self.phi[self.op.node_index(k)])
            f00, f10, f01, f11 = corners
            phi[idx] = (1 - wx) * (1 - wy) * f00 + wx * (1 - wy) * f10 + (1 - wx) * wy * f01 + wx * wy * f11
            ex[idx] = -((1 - wy) * (f10 - f00) + wy * (f11 - f01)) / g.dx
            ey[idx] = -((1 - wx) * (f01 - f00) + wx * (f11 - f10)) / g.dy
            todo[idx] = False
        return phi, ex, ey


def display_mesh(model: GeometryModel, hier: AmrHierarchy):
    """全レベルの葉セルを市松の対角線で 2 三角形に分割した表示用メッシュ (導体内のセルは穴)。

    返り値: (nodes (Nd, 2), triangles (M, 3), tri_region (M,), node_keys (Nd,) 正準節点キー)。
    ぶら下がり節点は細かい側の三角形の頂点になり、粗い側の三角形の辺上に乗る (T 字接続)。
    """
    L = hier.max_level
    tri_keys = []
    cells_xy = []
    for lvl in range(hier.n_levels):
        ci, cj = hier.leaf_cells(lvl)
        if ci.size == 0:
            continue
        g = hier.level_grid(lvl)
        xc = g.x0 + (ci + 0.5) * g.dx
        yc = g.y0 + (cj + 0.5) * g.dy
        keep = model.classify_conductor(xc, yc, 0.0) < 0 if model.conductors else np.ones(ci.size, dtype=bool)
        ci, cj = ci[keep], cj[keep]
        s = L - lvl
        # 巻き戻さない最細添字 (表示では周期の両側を別の節点にする)
        a = np.stack([ci << s, cj << s], axis=1)
        b = np.stack([(ci + 1) << s, cj << s], axis=1)
        c = np.stack([(ci + 1) << s, (cj + 1) << s], axis=1)
        d = np.stack([ci << s, (cj + 1) << s], axis=1)
        even = ((ci + cj) % 2 == 0)[:, None, None]
        t1 = np.where(even, np.stack([a, b, c], axis=1), np.stack([a, b, d], axis=1))
        t2 = np.where(even, np.stack([a, c, d], axis=1), np.stack([b, c, d], axis=1))
        tri = np.empty((2 * ci.size, 3, 2), dtype=np.int64)
        tri[0::2] = t1
        tri[1::2] = t2
        tri_keys.append(tri)
        cells_xy.append(np.repeat(np.stack([xc[keep], yc[keep]], axis=1), 2, axis=0))
    if not tri_keys:
        return (np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64))
    tri = np.concatenate(tri_keys)                    # (M, 3, 2) 最細添字 (I, J)
    nxf = hier.base.nx << L
    raw = tri[..., 1] * (nxf + 1) + tri[..., 0]
    uniq, inv = np.unique(raw.ravel(), return_inverse=True)
    uI, uJ = uniq % (nxf + 1), uniq // (nxf + 1)
    nodes = np.stack(_coords(hier, uI, uJ), axis=1)
    triangles = inv.reshape(-1, 3)
    cent = np.concatenate(cells_xy)
    other = model.classify_other(cent[:, 0], cent[:, 1])
    region_idx = np.array([o.index for o in model.others] + [-1], dtype=np.int64)
    tri_region = np.where(other >= 0, region_idx[np.maximum(other, 0)], -1)
    return nodes, triangles, tri_region, _canon_key(hier, uI, uJ)


def _display_mesh(model: GeometryModel, hier: AmrHierarchy, op: CompositeOperator, phi: np.ndarray):
    nodes, triangles, tri_region, keys = display_mesh(model, hier)
    phi_d = phi[op.node_index(keys)] if keys.size else np.zeros(0)
    return nodes, phi_d, triangles, tri_region


def build_hierarchy(project: Project, *, h: float | None = None, spec: AmrSpec | None = None,
                    model: GeometryModel | None = None) -> AmrHierarchy:
    """project (mesh.size・mesh.amr) または spec から AMR 階層を作る。"""
    model = model if model is not None else GeometryModel(project)
    base = make_grid(model.domain, float(h if h is not None else project.mesh.size))
    return AmrHierarchy(model, base, spec if spec is not None else AmrSpec.from_settings(project.mesh.amr))


def solve_electrostatic_amr(project: Project, *, h: float | None = None, t: float | None = None,
                            tol: float = 1e-10, spec: AmrSpec | None = None,
                            hier: AmrHierarchy | None = None) -> AmrStaticSolution:
    """project.mesh.amr (または spec・作成済みの hier) の階層で静電場を解く (CPU、AMG-CG)。"""
    from ..gpic.geometry import triangle_gradients

    t_start = time.perf_counter()
    if hier is None:
        hier = build_hierarchy(project, h=h, spec=spec)
    model = hier.model
    t_h = time.perf_counter()
    op = assemble_composite(model, hier)
    t_a = time.perf_counter()
    v = group_voltages(model, t)
    phi, info = solve_composite(op, v, tol=tol)
    t_s = time.perf_counter()
    energy, q = energy_and_charges(model, op, phi, v)
    charges = [(g.label, float(g.voltage if t is None else v[k]), float(q[k])) for k, g in enumerate(model.groups)]
    capacitance = None
    if len(charges) >= 2 and not model.has_charge:
        levels = sorted({vv for _, vv, _ in charges})
        if len(levels) == 2:
            lo, hi = levels
            capacitance = sum(qq for _, vv, qq in charges if vv == hi) / (hi - lo)
    nodes, phi_d, tris, tri_region = _display_mesh(model, hier, op, phi)
    e_tri = -triangle_gradients(nodes, tris, phi_d) if len(tris) else np.zeros((0, 2))
    return AmrStaticSolution(
        hier=hier, op=op, phi=phi, nodes=nodes, phi_display=phi_d, triangles=tris, tri_region=tri_region,
        e_tri=e_tri, energy=energy, charges=charges, capacitance=capacitance, info=info,
        timing={"hierarchy_s": t_h - t_start, "assemble_s": t_a - t_h, "solve_s": t_s - t_a,
                "total_s": time.perf_counter() - t_start},
        warnings=list(op.warnings),
    )
