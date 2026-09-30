"""プロジェクト (v1 スキーマ) の幾何を v2 直交格子エンジン向けに解釈する (prompts/119)。

v1 の意味論をそのまま引き継ぐ (meshing._generate_structured / fem._material_arrays と同じ):

- 領域の優先順位: conductor が最優先 (リスト順)、次に dielectric/charge (リスト順)。
  重なった点は先に一致した領域に属する。
- ε: dielectric は ε0·εr、charge 領域と真空は ε0 (charge の eps_r は使わない — v1 と同じ)。
- 導体は Dirichlet (内部は解かない)。外周の dirichlet 辺も Dirichlet で、両方に属する
  節点は導体の指定を優先する (v1 の「電極の指定を優先して上書き」と同じ)。
- 外周の未指定辺は自然境界 (Neumann)、symmetry も場は Neumann、periodic は対辺同一視。

v2 エンジン (埋め込み境界) 固有の約束:

- ε の分類 (``eps_at``) は**導体を無視**して行う。導体内部の ε は解に現れない一方、
  導体の表面に接する誘電体の ε を正しく拾う必要があるため (導体の中心が誘電体に
  覆われていても、導体外の部分は誘電体として扱われる)。
- ρ・気体体積の分類 (``rho_at`` / ``gas_at``) は導体を考慮する (導体内の電荷は存在しない、
  粒子は導体・誘電体に入れない)。
- 現状 domain は軸平行な矩形のみ (v1 UI も矩形 domain のみ作れる)。非矩形 domain は
  CAD v2 (P7) で EB Neumann 境界として対応する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..schema import BoundaryCondition, Project, Region, VoltageRF, VoltageWaveform, rf_components
from .shapes import CircleShape, PolygonShape, Shape

EPS0 = 8.8541878128e-12  # 真空の誘電率 [F/m] (fem.EPS0 と同じ値)

Side = Literal["bottom", "right", "top", "left"]
Coord = Literal["xy", "rz", "rz_x0"]


@dataclass(frozen=True)
class RegionShape:
    """解釈済みの領域 (元の Region と形状)。"""

    index: int          # project.geometry.regions 内の番号
    id: str
    type: str           # "conductor" | "dielectric" | "charge"
    shape: Shape
    eps_r: float        # ε の相対値 (dielectric のみ eps_r、それ以外 1)
    rho: float          # charge のみ rho、それ以外 0
    region: Region


@dataclass(frozen=True)
class DomainRect:
    x0: float
    y0: float
    x1: float
    y1: float
    # domain.polygon の辺番号 → 辺の位置 (v1 の "edge{k}" ラベルと対応させるため保持)
    edge_sides: dict[int, Side]

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def scale(self) -> float:
        return max(abs(self.x0), abs(self.x1), abs(self.y0), abs(self.y1), self.width, self.height, 1e-300)


@dataclass
class SideBC:
    """外周の 1 辺の境界条件。"""

    side: Side
    edge: int                       # domain.polygon の辺番号
    kind: Literal["neumann", "dirichlet", "symmetry", "periodic"] = "neumann"
    bc: BoundaryCondition | None = None

    @property
    def label(self) -> str:
        return f"edge{self.edge}"


@dataclass
class DirichletGroup:
    """同じ電位波形を持つ Dirichlet 節点の集まり (外周の 1 辺、または導体 1 つ)。

    PIC では毎ステップ V(t) = voltage + Σ RF + waveform を評価し、結合係数
    (eb.build の coupling 列) との積を右辺に足す。
    """

    label: str                      # "edge{k}" または導体の region id (v1 の electrode ラベル)
    kind: Literal["edge", "region"]
    voltage: float
    rf: list[VoltageRF] = field(default_factory=list)
    waveform: VoltageWaveform | None = None
    see_gamma: float = 0.0

    def value(self, t: float) -> float:
        """時刻 t の電位 [V] (v1 pic._dirichlet_values と同じ合成式・同じ評価関数)。"""
        from ..pic import _eval_rf, _eval_waveform  # v1 と完全に同じ位相規約・波形の折返し

        v = self.voltage + _eval_rf(self.rf, t)
        if self.waveform is not None:
            wf = self.waveform
            v += float(_eval_waveform(wf.phase, wf.v, wf.freq_hz, t))
        return float(v)


def _region_shape(region: Region) -> Shape:
    if region.shape is not None:
        cx, cy = region.shape.center
        return CircleShape(float(cx), float(cy), float(region.shape.radius))
    holes = tuple(np.asarray(h.polygon, dtype=np.float64) for h in region.holes)
    return PolygonShape(np.asarray(region.polygon, dtype=np.float64), holes)


def _domain_rect(polygon: list[tuple[float, float]]) -> DomainRect:
    poly = np.asarray(polygon, dtype=np.float64)
    scale = float(np.max(np.abs(poly))) or 1.0
    tol = 1e-12 * scale
    if len(poly) != 4:
        raise ValueError("v2 直交格子エンジンには 4 頂点の軸平行矩形 domain が必要です")
    x0, x1 = float(poly[:, 0].min()), float(poly[:, 0].max())
    y0, y1 = float(poly[:, 1].min()), float(poly[:, 1].max())
    if not (x1 - x0 > 0.0 and y1 - y0 > 0.0):
        raise ValueError("domain の幅・高さが 0 です")
    sides: dict[int, Side] = {}
    for k in range(4):
        p, q = poly[k], poly[(k + 1) % 4]
        if abs(p[1] - q[1]) <= tol and abs(p[1] - y0) <= tol:
            sides[k] = "bottom"
        elif abs(p[1] - q[1]) <= tol and abs(p[1] - y1) <= tol:
            sides[k] = "top"
        elif abs(p[0] - q[0]) <= tol and abs(p[0] - x0) <= tol:
            sides[k] = "left"
        elif abs(p[0] - q[0]) <= tol and abs(p[0] - x1) <= tol:
            sides[k] = "right"
        else:
            raise ValueError(
                f"v2 直交格子エンジンには軸平行の矩形 domain が必要です (辺 {k} が軸平行ではありません)"
            )
    if sorted(sides.values()) != sorted(["bottom", "right", "top", "left"]):
        raise ValueError("domain の 4 辺が矩形の上下左右に対応しません")
    return DomainRect(x0, y0, x1, y1, sides)


class GeometryModel:
    """v2 エンジンが参照する幾何・材料・境界条件の解釈結果。"""

    def __init__(self, project: Project):
        geo = project.geometry
        self.project = project
        self.coord: Coord = project.coord
        self.domain = _domain_rect(geo.domain.polygon)
        self.tol = 1e-9 * self.domain.scale  # 点の内包判定の境界許容 (v1 の 1e-12 より緩め: 格子座標の丸め誤差を吸収)

        regions = [
            RegionShape(
                index=i,
                id=r.id,
                type=r.type,
                shape=_region_shape(r),
                eps_r=float(r.eps_r) if r.type == "dielectric" else 1.0,
                rho=float(r.rho) if r.type == "charge" else 0.0,
                region=r,
            )
            for i, r in enumerate(geo.regions)
        ]
        self.conductors = [r for r in regions if r.type == "conductor"]
        self.others = [r for r in regions if r.type != "conductor"]
        for c in self.conductors:
            if c.region.voltage is None:
                raise ValueError(f"conductor '{c.id}' に voltage がありません")

        # ---- 外周の境界条件 -------------------------------------------------------
        self.sides: dict[Side, SideBC] = {
            side: SideBC(side=side, edge=k) for k, side in self.domain.edge_sides.items()
        }
        for bc in geo.boundaries:
            for e in bc.edges:
                if e not in self.domain.edge_sides:
                    raise ValueError(f"境界条件のエッジ番号 {e} が範囲外です (0..3)")
                s = self.sides[self.domain.edge_sides[e]]
                s.kind = bc.type
                s.bc = bc
        self.periodic_x = self.sides["left"].kind == "periodic" and self.sides["right"].kind == "periodic"
        self.periodic_y = self.sides["bottom"].kind == "periodic" and self.sides["top"].kind == "periodic"
        for pair, ok in ((("left", "right"), self.periodic_x), (("bottom", "top"), self.periodic_y)):
            if not ok and any(self.sides[s].kind == "periodic" for s in pair):
                raise ValueError("periodic 境界は対辺 (左右 または 上下) の組で指定してください")

        # ---- Dirichlet グループ (v1 fem._label_order と同じ順: 外周辺 → 導体) ----------
        groups: list[DirichletGroup] = []
        self.side_group: dict[Side, int] = {}
        seen: set[str] = set()
        for bc in geo.boundaries:
            if bc.type != "dirichlet":
                continue
            for e in bc.edges:
                side = self.domain.edge_sides[e]
                label = f"edge{e}"
                if label in seen:
                    continue
                seen.add(label)
                self.side_group[side] = len(groups)
                groups.append(
                    DirichletGroup(
                        label=label, kind="edge", voltage=float(bc.voltage),
                        rf=rf_components(bc.voltage_rf), waveform=bc.voltage_waveform,
                        see_gamma=float(bc.see_gamma),
                    )
                )
        self.conductor_group: list[int] = []
        for c in self.conductors:
            self.conductor_group.append(len(groups))
            r = c.region
            groups.append(
                DirichletGroup(
                    label=c.id, kind="region", voltage=float(r.voltage),
                    rf=rf_components(r.voltage_rf), waveform=r.voltage_waveform,
                    see_gamma=float(r.see_gamma),
                )
            )
        self.groups = groups

        if self.coord != "xy":
            ridx = 1 if self.coord == "rz" else 0
            rmin = self.domain.y0 if ridx == 1 else self.domain.x0
            if rmin < -self.tol:
                raise ValueError("軸対称モードでは domain の r 座標が 0 以上である必要があります")

    # ---- 点の分類 ---------------------------------------------------------------

    def classify_conductor(self, x: np.ndarray, y: np.ndarray, tol: float | None = None) -> np.ndarray:
        """各点が属する導体の番号 (self.conductors 内、優先順位適用済み)。無ければ -1。"""
        t = self.tol if tol is None else tol
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        out = np.full(np.broadcast(x, y).shape, -1, dtype=np.int64)
        for k, c in enumerate(self.conductors):
            free = out == -1
            if not np.any(free):
                break
            hit = c.shape.contains(x, y, t) & free
            out[hit] = k
        return out

    def classify_other(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """導体を無視した誘電体/電荷領域の番号 (self.others 内、優先順位適用済み)。無ければ -1。"""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        out = np.full(np.broadcast(x, y).shape, -1, dtype=np.int64)
        for k, o in enumerate(self.others):
            free = out == -1
            if not np.any(free):
                break
            hit = o.shape.contains(x, y, 0.0) & free
            out[hit] = k
        return out

    def eps_r_table(self) -> np.ndarray:
        """classify_other の結果 (+1 シフト) → εr の表 (index 0 = 真空)。"""
        return np.array([1.0] + [o.eps_r for o in self.others], dtype=np.float64)

    def rho_table(self) -> np.ndarray:
        return np.array([0.0] + [o.rho for o in self.others], dtype=np.float64)

    def eps_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """誘電率 ε [F/m] (導体は無視して分類)。"""
        return EPS0 * self.eps_r_table()[self.classify_other(x, y) + 1]

    def rho_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """電荷密度 ρ [C/m^3] (導体内は 0)。"""
        rho = self.rho_table()[self.classify_other(x, y) + 1]
        if self.conductors:
            rho = np.where(self.classify_conductor(x, y, 0.0) >= 0, 0.0, rho)
        return rho

    def gas_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """粒子が存在できる点か (導体・誘電体の外)。"""
        other = self.classify_other(x, y)
        solid = np.zeros(other.shape, dtype=bool)
        for k, o in enumerate(self.others):
            if o.type == "dielectric":
                solid |= other == k
        if self.conductors:
            solid |= self.classify_conductor(x, y, 0.0) >= 0
        return ~solid

    @property
    def has_charge(self) -> bool:
        return any(o.type == "charge" and o.rho != 0.0 for o in self.others)

    def radial_axis(self) -> int | None:
        """径方向の座標軸 (rz: 1 (y)、rz_x0: 0 (x)、xy: None)。"""
        return {"rz": 1, "rz_x0": 0}.get(self.coord)
