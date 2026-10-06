"""プロジェクト JSON のスキーマ (pydantic)。仕様書 §10 参照。

このモデルがプロジェクトファイルの唯一の正。
UI の TypeScript 型 (ui/src/model/project.ts ほか) はこれと手動同期し、設定フォームは
python -m es_sim.ui_schema が書き出す JSON Schema (ui/src/schema/project.schema.json) から作る。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from .params import ExprError, evaluate_params, names_in
from .paths import check_holes, check_path, flatten_path, has_arcs

Point = tuple[float, float]


def ui(unit: str | None = None, *, geom: bool = False, advanced: bool = False) -> dict:
    """UI v2 の設定フォームへの追記 (JSON Schema の x-unit / x-geom / x-advanced、prompts/130)。

    unit は保存値の単位 (SI、"1" は無次元)。geom=True の長さ・座標はプロジェクトの長さの表示単位
    (mm / µm) で入出力する。advanced=True は「詳細設定」を開いたときだけ出す項目。
    """
    extra: dict = {}
    if unit is not None:
        extra["x-unit"] = unit
    if geom:
        extra["x-geom"] = True
    if advanced:
        extra["x-advanced"] = True
    return extra


GEOM = ui("m", geom=True)


BULGES_DESCRIPTION = (
    "辺ごとの円弧 (辺 i は頂点 i → i+1)。bulge = tan(θ/4)、θ は中心角で正は反時計回り、0 は直線 "
    "(DXF の LWPOLYLINE と同じ)。ソルバーには検証の段階でメッシュ幅に合わせた弦に分けて渡す (paths.py)"
)


class Domain(BaseModel):
    polygon: list[Point] = Field(
        ...,
        min_length=2,
        description="解析領域の外周 (閉じた経路、反時計回り)。直線だけなら 3 頂点以上、円弧を含めば 2 頂点以上",
        json_schema_extra=GEOM,
    )
    bulges: list[float] | None = Field(None, description=BULGES_DESCRIPTION, json_schema_extra=ui("1"))
    # 辺の永続 ID (prompts/132)。UI が辺を分ける・消すときに境界条件などの辺の番号を付け替えるのに使う
    # (ソルバーは使わない)。辺と同じ数・重複なし
    edge_ids: list[str] | None = None

    @model_validator(mode="after")
    def _check(self) -> "Domain":
        check_path(self.polygon, self.bulges, "domain")
        if self.edge_ids is not None:
            n = len(self.polygon)
            if len(self.edge_ids) != n:
                raise ValueError(f"domain.edge_ids は辺と同じ数 ({n}) が必要です (今は {len(self.edge_ids)})")
            if any(not e for e in self.edge_ids) or len(set(self.edge_ids)) != n:
                raise ValueError("domain.edge_ids は空でない重複のない文字列が必要です")
        return self


class CircleShape(BaseModel):
    """円領域のパラメトリック形状。メッシュ生成時に多角形化する (meshing._region_polygon 参照)。"""

    kind: Literal["circle"] = "circle"
    center: Point = Field(..., json_schema_extra=GEOM)
    radius: float = Field(..., gt=0, json_schema_extra=GEOM)


class VoltageRF(BaseModel):
    """RF 電圧成分 (フェーズ3)。V(t) = voltage + Σ_k amplitude_k * sin(2π f_k t + phase_k)。

    voltage_rf フィールドには単一成分 (VoltageRF) と成分リスト (list[VoltageRF]、
    デュアル周波数など) のどちらも指定できる (prompts/49)。
    静電ソルブ (/solve) は従来通り直流分 voltage のみを使い、PIC のみが V(t) を使う。
    """

    amplitude: float = Field(..., json_schema_extra=ui("V"))
    freq_hz: float = Field(..., gt=0, json_schema_extra=ui("Hz"))
    phase_deg: float = Field(0.0, json_schema_extra=ui("deg"))


def rf_components(rf: "VoltageRF | list[VoltageRF] | None") -> "list[VoltageRF]":
    """voltage_rf フィールド (単一 / リスト / None) を成分リストへ正規化する。"""
    if rf is None:
        return []
    if isinstance(rf, VoltageRF):
        return [rf]
    return list(rf)


class VoltageWaveform(BaseModel):
    """CSV からインポートした任意周期波形 (prompts/73)。

    CSV の時間列は波形の「形状」を定義するためだけに使い、実際の周期は
    freq_hz (ユーザー指定) で決まる。取り込み時にフロント側で時間 t を
    [t_min, t_max] → [0, 1) へ線形写像した正規化位相として保存するため、
    ここでは phase を昇順・[0,1) の範囲としてのみ検証する。
    評価は V_wf(t) = interp(frac(t·freq_hz), phase, v) (pic.py 参照)。
    voltage_rf と同じく静電ソルブでは無視され、PIC のみが使う。
    """

    freq_hz: float = Field(..., gt=0, json_schema_extra=ui("Hz"))
    phase: list[float] = Field(..., json_schema_extra=ui("1"))  # 正規化位相 [0, 1) (昇順)
    v: list[float] = Field(..., json_schema_extra=ui("V"))       # 対応する電圧 [V]

    @model_validator(mode="after")
    def _check_phase(self) -> "VoltageWaveform":
        if len(self.phase) != len(self.v):
            raise ValueError("VoltageWaveform: phase と v は同じ長さが必要です")
        if len(self.phase) < 2:
            raise ValueError("VoltageWaveform: phase/v は2点以上必要です")
        if any(p < 0.0 or p >= 1.0 for p in self.phase):
            raise ValueError("VoltageWaveform: phase は [0, 1) の範囲である必要があります")
        if any(a >= b for a, b in zip(self.phase, self.phase[1:])):
            raise ValueError("VoltageWaveform: phase は狭義昇順である必要があります")
        return self


class BlockingCapacitor(BaseModel):
    """電極と電源の間に直列の阻止コンデンサ (自己バイアス、prompts/134)。2D の導体・Dirichlet の辺に付ける。

    電極の直流電位 (自己バイアス) が放電に合わせて決まる (circuit.py)。電源の直流分 (voltage) は定常では
    コンデンサが止める。流体・PIC の全て (v2 の一様格子・AMR、v1 の三角形メッシュ) が使う。静電場の計算
    (/solve) は無視する。容量は座標系の電荷の単位に合わせ、平面 2D は奥行き 1 m あたり [F/m]、軸対称は
    全周 [F] (静電場の電極の電荷 C/m・C と同じ)。
    """

    capacitance: float = Field(
        ..., gt=0, description="容量 (平面 2D は奥行き 1 m あたり、軸対称は全周)", json_schema_extra=ui("F/m")
    )
    initial_bias_v: float = Field(
        0.0, description="コンデンサの初期電圧 (電極の電位 − 電源の電圧)", json_schema_extra=ui("V")
    )


class BlockingCapacitor1d(BaseModel):
    """1D の電極の阻止コンデンサ (BlockingCapacitor と同じ、容量は面積あたり)。流体 1D・PIC 1D が使う。"""

    capacitance: float = Field(..., gt=0, description="面積あたりの容量", json_schema_extra=ui("F/m^2"))
    initial_bias_v: float = Field(
        0.0, description="コンデンサの初期電圧 (電極の電位 − 電源の電圧)", json_schema_extra=ui("V")
    )


class Loop(BaseModel):
    """閉じた経路 (領域の穴、prompts/132)。polygon の辺 i (頂点 i → i+1) が bulges[i] の円弧。"""

    polygon: list[Point] = Field(..., min_length=2, json_schema_extra=GEOM)
    bulges: list[float] | None = Field(None, description=BULGES_DESCRIPTION, json_schema_extra=ui("1"))


class Region(BaseModel):
    id: str
    type: Literal["conductor", "dielectric", "charge"]
    # 輪郭 (閉じた経路)。直線だけなら 3 頂点以上、円弧 (bulges) を含めば 2 頂点以上
    polygon: list[Point] | None = Field(None, min_length=2, json_schema_extra=GEOM)
    bulges: list[float] | None = Field(None, description=BULGES_DESCRIPTION, json_schema_extra=ui("1"))
    # 穴 (P7e のブーリアンの差など)。穴の中はこの領域ではない (下の領域・真空になる)。polygon の領域だけ
    holes: list[Loop] = []
    # UI のレイヤ (P7f、project.cad.layers の id)。ソルバーは使わない
    layer: str | None = Field(None, description="UI のレイヤ (project.cad.layers の id)。ソルバーは使わない")
    shape: CircleShape | None = None
    voltage: float | None = Field(None, json_schema_extra=ui("V"))  # conductor: 電位 [V] (直流分)
    # conductor: RF 成分 (PIC のみ使用)。単一またはリスト (デュアル周波数、prompts/49)
    voltage_rf: VoltageRF | list[VoltageRF] | None = None
    # conductor: CSV インポート波形 (PIC のみ使用、prompts/73)。voltage_rf と併用可
    # (V(t) = voltage + Σ RF + V_wf(t))。UI は境界条件辺のみ対応、スキーマ上のみ対応
    voltage_waveform: VoltageWaveform | None = None
    eps_r: float = Field(1.0, json_schema_extra=ui("1"))  # dielectric: 比誘電率
    rho: float = Field(0.0, json_schema_extra=ui("C/m^3"))  # charge: 電荷密度 [C/m^3]
    see_gamma: float = Field(
        0.0, ge=0,
        description="conductor / dielectric: 二次電子放出係数 γ (0 = 無効、PIC のみ使用)",
        json_schema_extra=ui("1"),
    )
    # conductor: 電源との間の阻止コンデンサ (自己バイアス、prompts/134)。None なら電源に直結
    blocking_capacitor: BlockingCapacitor | None = None

    @model_validator(mode="after")
    def _check_polygon_xor_shape(self) -> "Region":
        if self.blocking_capacitor is not None and self.type != "conductor":
            raise ValueError(f"領域 {self.id}: 阻止コンデンサ (blocking_capacitor) は導体の領域だけに付けられます")
        if (self.polygon is None) == (self.shape is None):
            raise ValueError("Region には polygon か shape のどちらか一方のみを指定してください")
        if self.polygon is not None:
            check_path(self.polygon, self.bulges, f"領域 {self.id}")
            for k, hole in enumerate(self.holes):
                check_path(hole.polygon, hole.bulges, f"領域 {self.id} の穴 {k + 1}")
            check_holes(self.polygon, self.bulges, [(h.polygon, h.bulges) for h in self.holes], f"領域 {self.id}")
        elif self.bulges is not None or self.holes:
            raise ValueError(f"領域 {self.id}: bulges・holes は polygon の領域だけに指定できます")
        return self


class BoundaryCondition(BaseModel):
    """domain 外周のエッジ単位の境界条件。

    edges: 外周ポリゴンのエッジ番号 (i 番目のエッジは頂点 i → i+1)。
    未指定のエッジは自然境界 (Neumann, dV/dn = 0)。

    type (prompts/22、フロントと共通のスキーマ契約):
      - "dirichlet": 固定電位。voltage / voltage_rf / see_gamma は dirichlet のみ有効
      - "symmetry":  対称境界。場は自然境界 (Neumann)、粒子 (trace / PIC) は鏡面反射
      - "periodic":  周期境界。edges にちょうど2本の対辺 (平行・同長) を指定する。
                     場は対辺の節点 DOF を同一視して解き、粒子は反対側へラップする
    """

    edges: list[int]
    type: Literal["dirichlet", "symmetry", "periodic"] = "dirichlet"
    voltage: float = Field(0.0, json_schema_extra=ui("V"))
    # RF 成分 (PIC のみ使用)。単一またはリスト (デュアル周波数、prompts/49)
    voltage_rf: VoltageRF | list[VoltageRF] | None = None
    # CSV インポート波形 (PIC のみ使用、prompts/73)。voltage_rf と併用可
    # (V(t) = voltage + Σ RF + V_wf(t))。dirichlet のみ有効
    voltage_waveform: VoltageWaveform | None = None
    see_gamma: float = Field(
        0.0, ge=0, description="二次電子放出係数 γ (0 = 無効、PIC のみ使用)", json_schema_extra=ui("1")
    )
    # 電源との間の阻止コンデンサ (自己バイアス、prompts/134)。dirichlet のみ。複数の辺なら 1 つの電極として
    # 扱う (辺どうしはつながっている)。None なら電源に直結
    blocking_capacitor: BlockingCapacitor | None = None

    @model_validator(mode="after")
    def _check_periodic_edge_count(self) -> "BoundaryCondition":
        if self.blocking_capacitor is not None and self.type != "dirichlet":
            raise ValueError("阻止コンデンサ (blocking_capacitor) は dirichlet の境界条件だけに付けられます")
        if self.type == "periodic":
            if len(self.edges) != 2 or self.edges[0] == self.edges[1]:
                raise ValueError("periodic 境界には異なるエッジをちょうど2本指定してください")
        return self


class Geometry(BaseModel):
    domain: Domain
    regions: list[Region] = []
    boundaries: list[BoundaryCondition] = []
    # 円弧を弦に分けたあとの辺 k → 元の辺の番号 (Project の検証で分けたときだけ。電荷のラベル "edge{k}" を
    # 元の辺の番号で付けるのに使う、edge_label)
    _edge_origin: list[int] | None = PrivateAttr(default=None)

    def edge_label(self, k: int) -> str:
        """外周の辺 k (弦に分けたあとの番号) の電極ラベル。円弧を分けた辺は元の辺の番号で付ける。"""
        origin = self._edge_origin
        return f"edge{origin[k] if origin is not None and 0 <= k < len(origin) else k}"

    @model_validator(mode="after")
    def _check_periodic_pairs(self) -> "Geometry":
        """periodic 境界の2辺が domain の平行・同長の対辺であることを検査する。"""
        poly = self.domain.polygon
        n = len(poly)
        bulges = self.domain.bulges
        for bc in self.boundaries:
            if bc.type != "periodic":
                continue
            for e in bc.edges:
                if not (0 <= e < n):
                    raise ValueError(f"periodic 境界のエッジ番号 {e} が範囲外です (0..{n - 1})")
                if bulges is not None and abs(bulges[e]) > 1e-12:
                    raise ValueError(f"periodic 境界のエッジ {e} は円弧です (周期境界は直線の辺だけ)")
            e1, e2 = bc.edges
            d1 = (poly[(e1 + 1) % n][0] - poly[e1][0], poly[(e1 + 1) % n][1] - poly[e1][1])
            d2 = (poly[(e2 + 1) % n][0] - poly[e2][0], poly[(e2 + 1) % n][1] - poly[e2][1])
            l1 = (d1[0] ** 2 + d1[1] ** 2) ** 0.5
            l2 = (d2[0] ** 2 + d2[1] ** 2) ** 0.5
            if l1 <= 0.0 or l2 <= 0.0:
                raise ValueError("periodic 境界のエッジが退化しています (長さ 0)")
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(cross) > 1e-6 * l1 * l2 or abs(l1 - l2) > 1e-6 * max(l1, l2):
                raise ValueError(
                    f"periodic 境界のエッジ {e1}, {e2} は平行かつ同じ長さの対辺である必要があります"
                )
        return self


class BField(BaseModel):
    """一様磁場 [T] (prompts/51)。

    粒子軌道追跡・PIC のローレンツ力 (Boris 回転) に適用する。静電場ソルブには
    影響しない。全成分 0 は未指定 (磁場なし) と等価。面内成分 (bx, by) は
    面外速度 vz と結合し、bz は面内のジャイロ運動・E×B ドリフトを生む。
    軸対称モード (rz / rz_x0) は未対応 (一様な径方向磁場は ∇·B=0 と矛盾するため)。
    """

    bx: float = Field(0.0, json_schema_extra=ui("T"))
    by: float = Field(0.0, json_schema_extra=ui("T"))
    bz: float = Field(0.0, json_schema_extra=ui("T"))

    def is_zero(self) -> bool:
        return self.bx == 0.0 and self.by == 0.0 and self.bz == 0.0


class LocalSize(BaseModel):
    region: str
    size: float = Field(..., json_schema_extra=GEOM)


class EdgeMeshSize(BaseModel):
    """線分近傍のローカルメッシュサイズ (prompts/90)。gmsh の Distance+Threshold フィールドで、
    線分から dist_in までは size、dist_out で全体特性長 (mesh.size) へ線形に戻す。

    キャンバスの「メッシュ細分」ツール (2点クリック) で追加する他、任意の電極エッジ・
    ドメイン辺の近傍だけシース解像などの目的で細かくしたい場合に使う。
    """

    p1: Point = Field(..., json_schema_extra=GEOM)
    p2: Point = Field(..., json_schema_extra=GEOM)
    size: float = Field(..., gt=0, json_schema_extra=GEOM)
    # 遷移距離 (None は自動: dist_in = 2·size、dist_out = 8·size。meshing.py 側で解決する)
    dist_in: float | None = Field(None, gt=0, json_schema_extra=ui("m", geom=True, advanced=True))
    dist_out: float | None = Field(None, gt=0, json_schema_extra=ui("m", geom=True, advanced=True))


class AmrRegion(BaseModel):
    """ユーザー指定の細分化領域 (軸平行矩形、対角の 2 点)。level までの細分化を保証する。"""

    p1: Point = Field(..., json_schema_extra=GEOM)
    p2: Point = Field(..., json_schema_extra=GEOM)
    level: int = Field(1, ge=1, le=6)


class AmrSettings(BaseModel):
    """v2 直交格子エンジン (mesh.mode="cartesian") の局所細分化 (prompts/121)。

    レベル l の格子幅は size/2^l。細分化の単位は blocking_factor × blocking_factor セルの
    ブロックで、細かいレベルは粗いレベルの内側に 1 ブロックの緩衝帯を持って入る (proper
    nesting、隣接する葉セルのレベル差は高々 1)。max_level=0 は細分化なし。
    """

    max_level: int = Field(0, ge=0, le=6)
    # 導体・誘電体の境界から buffer_cells セル以内を max_level まで細分化する
    refine_boundaries: bool = True
    buffer_cells: int = Field(2, ge=0, le=16)
    blocking_factor: int = Field(8, ge=1, le=64, json_schema_extra=ui(advanced=True))
    regions: list[AmrRegion] = []
    # 解に基づく適応細分化 (静電場のみ、prompts/122)。求解 → 誤差指標 (節点の二階差分 ≈ h²φ'') →
    # adapt_tol × (電位の範囲) を超える葉セルを 1 段細かく、を最大 adapt_iters 回 (max_level まで)
    adaptive: bool = False
    adapt_tol: float = Field(1e-3, gt=0.0, le=0.5, json_schema_extra=ui("1"))
    adapt_iters: int = Field(3, ge=1, le=8)
    # PIC の動的再格子化 (prompts/123)。pic_regrid_every ステップごとに、その区間で平均した
    # 電子密度・温度のデバイ長 λ_D に対し 格子幅/λ_D > pic_h_over_debye のセルを細かくする
    # (max_level まで。時間平均区間の前だけ。0 = 静的)
    pic_regrid_every: int = Field(0, ge=0)
    pic_h_over_debye: float = Field(1.0, gt=0.0, le=100.0, json_schema_extra=ui("1"))
    # DSMC の動的再格子化 (prompts/127)。dsmc_regrid_every ステップごとに、その区間で平均した密度・温度の
    # 平均自由行程 λ に対し 格子幅/λ > dsmc_h_over_mfp のセルを細かくする (max_level まで。時間平均区間の
    # 前だけ。0 = 静的)
    dsmc_regrid_every: int = Field(0, ge=0)
    dsmc_h_over_mfp: float = Field(0.5, gt=0.0, le=100.0, json_schema_extra=ui("1"))


class MeshSettings(BaseModel):
    size: float = Field(..., gt=0, description="全体特性長 [m]", json_schema_extra=GEOM)
    local_sizes: list[LocalSize] = []
    # 任意の線分近傍のローカルメッシュサイズ (prompts/90)。local_sizes (領域単位) と異なり
    # 幾何に依存しない任意の線分を指定できる。local_sizes と同じく structured では無視される
    local_edge_sizes: list[EdgeMeshSize] = []
    # メッシュ生成モード (prompts/34)。structured は軸平行矩形 domain 専用の
    # 等間隔構造格子 (三角形2分割)。local_sizes / local_edge_sizes は structured では無視される。
    # cartesian (prompts/119) は v2 エンジン: 直交格子 + 埋め込み境界 (EB)。静電場は GMG-PCG
    # (CPU/GPU)、PIC は GPU 版 (es_sim.gpic) で解く。size は要求メッシュ幅 (実際の格子は
    # マルチグリッド向けに最大 ~12% 細かくなる)。local_sizes / local_edge_sizes は無視され、
    # 局所細分化は amr で指定する (prompts/121)
    mode: Literal["unstructured", "structured", "cartesian"] = "unstructured"
    # cartesian モードの局所細分化 (prompts/121)。None = 細分化なし。他のモードでは無視される
    amr: AmrSettings | None = None


# ---- 粒子軌道追跡 (フェーズ2、仕様書 §8) --------------------------------------


class Species(BaseModel):
    """粒子種。electron/proton プリセット、または custom で q・m を直接指定する。"""

    preset: Literal["electron", "proton", "custom"] = "electron"
    q: float | None = Field(None, json_schema_extra=ui("C"))  # custom 時の電荷 [C]
    m: float | None = Field(None, json_schema_extra=ui("kg"))  # custom 時の質量 [kg]

    @model_validator(mode="after")
    def _check_custom_qm(self) -> "Species":
        if self.preset == "custom" and (self.q is None or self.m is None):
            raise ValueError("preset='custom' には q と m の指定が必要です")
        return self


class Emitter(BaseModel):
    """粒子源。

    line: p1-p2 の線分上に n 個を等間隔配置。point: p1 に全粒子を配置 (p2 は無視)。
    direction_deg は x 軸から反時計回りの射出方向 [度]、spread_deg はその一様分布
    半角 [度] (乱数は使わず、n 個に等間隔で振り分ける)。
    """

    kind: Literal["line", "point"] = "line"
    p1: Point = Field(..., json_schema_extra=GEOM)
    p2: Point | None = Field(None, json_schema_extra=GEOM)
    n: int = Field(..., gt=0)
    energy_ev: float = Field(0.0, json_schema_extra=ui("eV"))
    direction_deg: float = Field(0.0, json_schema_extra=ui("deg"))
    spread_deg: float = Field(0.0, json_schema_extra=ui("deg"))
    energy_dist: Literal["mono", "maxwell"] = "mono"  # "mono": 従来動作 / "maxwell": 熱速度成分を付加
    temperature_ev: float = Field(1.0, gt=0, description="maxwell 時の温度 kT [eV]", json_schema_extra=ui("eV"))
    seed: int = Field(0, json_schema_extra=ui(advanced=True))  # maxwell サンプリングの乱数シード (再現性確保)

    @model_validator(mode="after")
    def _check_line_needs_p2(self) -> "Emitter":
        if self.kind == "line" and self.p2 is None:
            raise ValueError("kind='line' には p2 の指定が必要です")
        return self


class FnEmission(BaseModel):
    """Fowler–Nordheim 電界放出源 (prompts/46)。

    電極 (Dirichlet) 表面の指定区間から、表面電界に応じた FN 電流密度
    (Murphy-Good 式 + Forbes 近似、fn.py 参照) で電子を放出する。
    放出面の指定は edges (domain 外周エッジ番号) と regions (conductor 領域 id)
    の少なくとも一方。
    """

    edges: list[int] = []
    regions: list[str] = []
    phi_ev: float = Field(4.5, gt=0, description="仕事関数 φ [eV]", json_schema_extra=ui("eV"))
    beta: float = Field(1.0, gt=0, description="電界増倍係数 β", json_schema_extra=ui("1"))
    n: int = Field(200, gt=0, description="trace 時の放出マクロ粒子総数")
    init_energy_ev: float = Field(0.1, ge=0, description="放出電子の初期エネルギー [eV]", json_schema_extra=ui("eV"))
    # PIC のみ: マクロ重み (実電子数/マクロ粒子)。None なら初期プラズマの重みを使う
    macro_weight: float | None = Field(None, gt=0, json_schema_extra=ui("1"))
    seed: int = Field(0, json_schema_extra=ui(advanced=True))  # PIC の放出位置サンプリング乱数シード

    @model_validator(mode="after")
    def _check_sources(self) -> "FnEmission":
        if not self.edges and not self.regions:
            raise ValueError("fn には edges か regions を少なくとも1つ指定してください")
        return self


class ParticleSettings(BaseModel):
    species: Species = Species()
    # 通常エミッタ。fn (FN 電界放出) 指定時は省略可 (指定されていても無視される)
    emitter: Emitter | None = None
    # FN 電界放出源 (prompts/46)。指定時は emitter の代わりに電極表面から放出する。
    # 放出種は常に電子 (species は無視される)
    fn: FnEmission | None = None
    dt: float | None = Field(None, json_schema_extra=ui("s"))  # 秒。None なら自動推定 (particles.py 参照)
    n_steps: int = Field(5000, gt=0)
    save_every: int = Field(10, gt=0)

    @model_validator(mode="after")
    def _check_source(self) -> "ParticleSettings":
        if self.emitter is None and self.fn is None:
            raise ValueError("particles には emitter か fn のどちらかの指定が必要です")
        return self


# ---- PIC (フェーズ3、仕様書 §9) ----------------------------------------------


class InitialPlasma(BaseModel):
    """初期プラズマの一様装荷。null なら初期装荷なし。"""

    density: float = Field(..., gt=0, description="数密度 [m^-3] (奥行き1m換算)", json_schema_extra=ui("m^-3"))
    te_ev: float = Field(2.0, ge=0, description="電子温度 kTe [eV]", json_schema_extra=ui("eV"))
    ti_ev: float = Field(0.03, ge=0, description="イオン温度 kTi [eV]", json_schema_extra=ui("eV"))
    ion_mass_amu: float = Field(40.0, gt=0, description="イオン質量 [amu] (Ar+ = 40)", json_schema_extra=ui("amu"))
    immobile_ions: bool = Field(False, json_schema_extra=ui(advanced=True))  # true でイオン固定 (検証用)
    seed: int = Field(0, json_schema_extra=ui(advanced=True))


class PicInjection(BaseModel):
    """エミッタからの定常注入。電流を毎ステップの実電荷として等分注入する。

    current_a_per_m の単位: 平面2D (xy) = [A/m] (奥行き1m換算)、
    軸対称 (rz / rz_x0) = [A] (リングエミッタの全電流)。
    """

    emitter: Emitter
    species: Literal["electron", "ion"] = "electron"
    current_a_per_m: float = Field(..., gt=0, json_schema_extra=ui("A/m"))


# ---- MCC 衝突 (prompts/19、フロントと共通のスキーマ契約) ------------------------


class XsProcess(BaseModel):
    """LXCat 由来の衝突断面積プロセス (パース済み、プロジェクト JSON に埋め込む)。"""

    kind: Literal["elastic", "excitation", "ionization", "isotropic", "backscat"]
    label: str = ""                # PROCESS 行等から
    threshold_ev: float = Field(0.0, json_schema_extra=ui("eV"))  # excitation/ionization のみ >0
    mass_ratio: float = Field(0.0, json_schema_extra=ui("1"))     # elastic のみ (m/M)。無ければ 0
    energy_ev: list[float] = Field(..., json_schema_extra=ui("eV"))   # 断面積テーブルのエネルギー [eV] (昇順)
    sigma_m2: list[float] = Field(..., json_schema_extra=ui("m^2"))   # 断面積 [m^2] (energy_ev と同長)

    @model_validator(mode="after")
    def _check_table(self) -> "XsProcess":
        if len(self.energy_ev) != len(self.sigma_m2):
            raise ValueError("energy_ev と sigma_m2 は同じ長さが必要です")
        if len(self.energy_ev) == 0:
            raise ValueError("断面積テーブルが空です")
        return self


class MccGas(BaseModel):
    """背景中性ガスの状態。数密度は n_g = p/(kB·T) で決まる。"""

    name: str = "Ar"
    pressure_pa: float = Field(..., gt=0, description="ガス圧 [Pa]", json_schema_extra=ui("Pa"))
    temperature_k: float = Field(300.0, gt=0, description="ガス温度 [K]", json_schema_extra=ui("K"))


class MccSettings(BaseModel):
    """MCC 設定。null なら MCC 無効 (従来の無衝突動作)。"""

    gas: MccGas
    electron_processes: list[XsProcess] = []  # elastic/excitation/ionization
    ion_processes: list[XsProcess] = []       # isotropic/backscat
    seed: int = Field(0, json_schema_extra=ui(advanced=True))
    # 電離の余剰エネルギー分配: "half" = 散乱電子と生成電子で等分 (Turner ベンチマーク互換)、
    # "random" = 一様乱数比で分配 (従来動作)
    ionization_split: Literal["half", "random"] = "half"
    # イオン断面積テーブルの参照エネルギー系: "lab" = 実験室系イオンエネルギー (従来動作)、
    # "com" = 重心系エネルギー E = ½μg² (μ = m_i·m_g/(m_i+m_g)、Turner の He+/He データ用)
    ion_energy_frame: Literal["com", "lab"] = "lab"
    # true なら直前に実行した DSMC の定常ガス場 (n·T·u) を背景として使う (prompts/54)。
    # サーバーが保持する DSMC 結果とメッシュが一致している必要がある
    use_dsmc_gas: bool = False


class Collector(BaseModel):
    """IEDF/IADF コレクタ線分 (prompts/30)。null なら無効。

    平均区間中にコレクタ線分の近傍 (距離 tol 以内・線分区間内) で吸収された
    イオンのエネルギー・入射角・重みを記録する (ウエハ面の IEDF/IADF 取得用)。
    """

    p1: Point = Field(..., json_schema_extra=GEOM)
    p2: Point = Field(..., json_schema_extra=GEOM)
    tol: float | None = Field(None, gt=0, description="判定距離 [m]。None なら mesh.size と同値", json_schema_extra=GEOM)
    label: str = ""  # 表示用ラベル (空ならフロントが "C1" 等を振る、prompts/36)


class SheathLine(BaseModel):
    """2D シースエッジの Brinkmann 評価ライン (prompts/98)。可視化専用で計算はフロント側
    (時間平均・位相分解の n_e/n_i 節点配列は既に done メッセージでフロントに届いており、
    α スライダの即時反映や保存済み結果ファイルでの再計算にもフロント計算が適切なため)。

    p1 = 電極側、p2 = バルク側 (Brinkmann 積分の参照点 x_b = p2 の位置)。
    """

    p1: Point = Field(..., json_schema_extra=GEOM)
    p2: Point = Field(..., json_schema_extra=GEOM)
    label: str = ""  # 空ならフロントが S1, S2... を振る


class EedfRegion(BaseModel):
    """EEDF/EEPF の集計領域 (軸平行矩形、prompts/85)。キャンバスの2点クリックで指定する。

    時間平均区間中の毎ステップ、矩形内 (min(p1,p2) ≦ x ≦ max(p1,p2)) の電子を
    重み付きエネルギーヒストグラムへ加算し、EEDF (f(E)、∫f dE=1) を得る。
    """

    p1: Point = Field(..., json_schema_extra=GEOM)
    p2: Point = Field(..., json_schema_extra=GEOM)  # 対角の2点 (順不同)
    label: str = ""
    bins: int = Field(100, ge=10, le=1000)
    # None = 平均区間の最初の集計ステップで「矩形内電子の最大エネルギー×1.2」に自動決定
    # (電子がいなければ 30 eV)。以後のステップはこの値で固定し、範囲外はオーバーフロー計数する
    e_max_ev: float | None = Field(None, json_schema_extra=ui("eV"))


class PicMerge(BaseModel):
    """粒子マージ設定 (高速化③、prompts/77)。

    電離でマクロ粒子数が増え続けると計算コストが際限なく上がるため、種ごとの
    マクロ粒子数が n_max を超えたら every ステップごとにセル内保存的マージ
    (Vranic et al. 2015 の k→2 マージ) で削減する。null なら無効 (既定)。
    """

    n_max: int = Field(100000, ge=1000, description="種ごとの上限マクロ粒子数")
    every: int = Field(100, ge=1, description="チェック間隔 [ステップ]")


class ConvergenceSettings(BaseModel):
    """時間発展の収束の判定 (prompts/137、convergence.py)。

    周期平均 (RF の最低周波数の周期の整数倍、RF が無ければ steps ステップ) の φ・n_e (粗いブロックの体積平均、
    L2) と電子・イオンの総数・阻止コンデンサの自己バイアスを比べる。雑音を差し引いた変化 D と、変化の減り方から
    見積もった残りの変化 R がどの量でも閾値以下の周期が hold 回続いたら収束とする。
    """

    enabled: bool = Field(True, description="周期平均の量が落ち着いたか (収束) を判定する")
    tol: float | None = Field(
        None, gt=0, lt=1, description="閾値 (相対)。空なら自動 (流体 0.001・PIC 0.01)", json_schema_extra=ui("1")
    )
    rf_periods: int = Field(1, ge=1, le=10000, description="判定の周期 (RF の最低周波数の周期の何個分)")
    steps: int | None = Field(
        None, ge=1, description="RF も CSV 波形も無いときの判定の周期 [ステップ]。空なら frame_every の 10 倍"
    )
    hold: int = Field(3, ge=1, le=1000, description="何回続けて満たしたら収束とするか")
    stop: bool = Field(
        False, description="収束したら、そこから平均区間 (avg_steps、空なら判定の周期 10 個分) を取って止める"
    )
    max_window: int = Field(
        20, ge=1, le=200, description="比べる窓の最大 [判定の周期] (雑音が大きいときに広げる上限)",
        json_schema_extra=ui(advanced=True),
    )


class PicSettings(BaseModel):
    initial_plasma: InitialPlasma | None = None
    injection: PicInjection | None = None
    n_macro: int = Field(20000, gt=0, description="種ごとの初期マクロ粒子数の目安")
    dt: float | None = Field(None, description="秒。None なら 0.1/ωpe (初期密度から)", json_schema_extra=ui("s"))
    n_steps: int = Field(2000, gt=0)
    frame_every: int = Field(20, gt=0, description="フレーム送出間隔 (ステップ)")
    mcc: MccSettings | None = None  # null なら MCC 無効
    see_energy_ev: float = Field(2.0, ge=0, description="SEE 電子の初期エネルギー [eV]", json_schema_extra=ui("eV"))
    # 完了時に返す時間平均フィールドの平均ステップ数 (最終 N ステップ、prompts/26)。
    # None なら全ステップの最後の 25% を平均する
    avg_steps: int | None = Field(None, gt=0)
    # RF 1周期の位相分解データ (アニメーション用) の位相ビン数 (prompts/28)。
    # 0 で無効。RF (voltage_rf) が未設定の場合も無効
    phase_bins: int = Field(40, ge=0)
    # IEDF/IADF コレクタ線分 (prompts/30)。null なら無効。
    # 旧単数形 (後方互換用)。validator で collectors へ正規化される
    collector: Collector | None = None
    # 複数コレクタ (prompts/36、最大8個)。内部処理はこちらのみを参照する
    collectors: list[Collector] = []

    @model_validator(mode="after")
    def _normalize_collectors(self) -> "PicSettings":
        """旧単数形 collector を collectors へ正規化する (後方互換)。"""
        if self.collector is not None and not self.collectors:
            self.collectors = [self.collector]
            self.collector = None
        if len(self.collectors) > 8:
            raise ValueError("collectors は最大 8 個までです")
        return self
    # EEDF/EEPF 集計領域 (prompts/85、最大4個)。時間平均区間中の毎ステップ、
    # 矩形内の電子を重み付きエネルギーヒストグラムへ加算する
    eedf_regions: list[EedfRegion] = []

    @model_validator(mode="after")
    def _check_eedf_regions(self) -> "PicSettings":
        if len(self.eedf_regions) > 4:
            raise ValueError("eedf_regions は最大 4 個までです")
        return self
    # 鏡面反射する domain 外周エッジ番号のリスト (エッジ i は頂点 i → i+1)。
    # 到達粒子は吸収せず法線速度成分を反転して境界内へ折り返す (壁カウンタに含めない)。
    # 当該エッジは境界条件なし (Neumann) を想定。2D ストリップで 1D 問題を模擬する用途
    reflect_edges: list[int] = Field(default_factory=list)
    # FN 電界放出源 (prompts/46)。毎ステップの表面電界から I·dt 分の電子を放出する。
    # null なら無効 (従来動作と完全一致)
    fn: FnEmission | None = None
    # イオンサブサイクリング (prompts/50): イオンを N ステップに1回、N·dt で押す。
    # 休止ステップ中はイオンの電荷堆積をキャッシュして再利用する。1 = 無効 (従来と完全一致)
    ion_subcycle: int = Field(1, ge=1, json_schema_extra=ui(advanced=True))
    # 粒子処理 (walk 探索) のワーカースレッド数 (prompts/50)。粒子ごとの walk は独立な
    # ため、チャンク並列化しても結果は逐次実行とビット単位で一致する。
    # 0 = 粒子数・CPU数から自動選択、1 = 逐次、2以上 = 明示並列
    threads: int = Field(0, ge=0, le=128, json_schema_extra=ui(advanced=True))
    # 粒子マージ (高速化③、prompts/77)。null = 無効 (既定。マージ関連の処理・乱数消費が
    # 一切発生せず、従来経路と完全一致する)
    merge: PicMerge | None = None
    # 半径に比例したマクロ粒子の重み (prompts/136)。軸対称の PIC (v1・v2) で効く。目標の重みを c·(r + r0) にし、
    # 重すぎる粒子は分割、軽すぎる粒子は同じセル (v1 は同じ要素) の粒子と対で併合する (電荷は厳密に保存)。
    # 一様の重みでは軸の近くの粒子が少なく、統計の雑音で電子が加熱される。v1 は粒子マージ (merge) があると使わない
    radial_weighting: bool = Field(
        True,
        description="軸対称で、マクロ粒子の重みを半径に比例させる (軸の近くの粒子を増やし、統計の雑音による"
        "加熱を抑える)。v1 PIC は粒子マージ (merge) があると使わない",
        json_schema_extra=ui(advanced=True),
    )
    # シースエッジ評価ライン (prompts/98、最大4本)。可視化専用で backend は永続化のみ
    # (Brinkmann 判定・準中性度等値線の計算はフロント側、SheathLine 参照)
    sheath_lines: list[SheathLine] = []
    # 収束の判定 (prompts/137)
    convergence: ConvergenceSettings = ConvergenceSettings()

    @model_validator(mode="after")
    def _check_sheath_lines(self) -> "PicSettings":
        if len(self.sheath_lines) > 4:
            raise ValueError("sheath_lines は最大 4 個までです")
        return self


# ---- 1D PIC/MCC (1d3v、prompts/91) --------------------------------------------
#
# 2D FEM-PIC (上記 PicSettings/PicSimulation) とは完全に独立な専用ソルバー
# (pic1d.py 参照)。一様格子 + 三重対角 Poisson を使う CCP ベンチマーク
# (Turner et al. 2013 / eduPIC) 向けの軽量モジュール。


class Fn1dEmission(BaseModel):
    """1D 電極の FN (Fowler–Nordheim) 電界放出 (prompts/95)。

    2D の FnEmission (fn.py の Murphy-Good式 + Forbes近似 fn_current_density を
    そのまま流用) と同じ物理・パラメータだが、1D は電極がちょうど1点 (左端 x=0
    または右端 x=gap) なので放出面/位置サンプリングという概念がなく、
    edges/regions/n (放出マクロ粒子総数)/seed (位置乱数シード) は不要 —
    毎ステップの放出数は決定論的な端数キャリーのみで決まる (乱数不使用)。
    """

    phi_ev: float = Field(4.5, gt=0, description="仕事関数 φ [eV]", json_schema_extra=ui("eV"))
    beta: float = Field(1.0, gt=0, description="電界増倍係数 β", json_schema_extra=ui("1"))
    init_energy_ev: float = Field(0.1, ge=0, description="放出電子の初期エネルギー [eV]", json_schema_extra=ui("eV"))
    # マクロ重み [m^-2] (= 実電子数/マクロ粒子。1D のマクロ重みの単位そのものなので
    # 2D のような面積換算は不要)。None なら初期プラズマの w0 を使う
    macro_weight: float | None = Field(None, gt=0, json_schema_extra=ui("m^-2"))


class Pic1dElectrode(BaseModel):
    """1D の左右電極。電圧は v_dc + Σ RF sin + Σ waveforms(t) の合成 (prompts/93)。
    2D の Region/BoundaryCondition の voltage_rf と同じ規約 (単一/リスト/None、
    pic.py の _dirichlet_values と同じ式) をそのまま流用する。CSV 波形
    (VoltageWaveform) とは併記可能 (両方指定すれば両方の寄与が加算される。
    pic1d.py の _electrode_voltage / pic.py の _eval_waveform 参照)。
    """

    v_dc: float = Field(0.0, json_schema_extra=ui("V"))
    # RF 重畳 (prompts/93)。2D の BoundaryCondition.voltage_rf と同じ規約:
    # 単一 VoltageRF / リスト (デュアル周波数など) / None。
    # V_rf(t) = Σ amplitude·sin(2π·freq_hz·t + phase_deg·π/180)
    voltage_rf: VoltageRF | list[VoltageRF] | None = None
    waveforms: list[VoltageWaveform] = []
    see_gamma: float = Field(0.0, ge=0.0, le=1.0, description="イオン入射あたりのSEE収率 γ", json_schema_extra=ui("1"))
    # FN 電界放出 (prompts/95)。None なら放出なし (従来動作と完全ビット不変)
    fn: Fn1dEmission | None = None
    # 電源との間の阻止コンデンサ (自己バイアス、prompts/134)。流体 1D・PIC 1D。
    # None なら電源に直結
    blocking_capacitor: BlockingCapacitor1d | None = None


class Eedf1dRegion(BaseModel):
    """1D の EEDF/EEPF 集計区間 [x1, x2] (2D の EedfRegion の 1D 版、prompts/85 と同じ規約)。"""

    x1: float = Field(..., json_schema_extra=GEOM)
    x2: float = Field(..., json_schema_extra=GEOM)
    label: str = ""
    bins: int = Field(100, ge=10, le=1000)
    # None = 平均区間の最初の集計ステップで自動決定 (2D の EedfRegion.e_max_ev と同じ規約)
    e_max_ev: float | None = Field(None, gt=0, json_schema_extra=ui("eV"))


class Pic1dSettings(BaseModel):
    """1D PIC/MCC (1d3v)。null なら無効。2D の pic とは独立に実行できる (pic1d.py 参照)。

    一様格子 (n_cells 個のセル、n_nodes = n_cells+1 節点) + 三重対角 Poisson。
    メッシュ生成が無いため geometry/mesh の設定とは無関係に動作する。
    """

    gap_m: float = Field(..., gt=0, description="電極間ギャップ [m]", json_schema_extra=GEOM)
    n_cells: int = Field(128, ge=8, le=100000)
    left: Pic1dElectrode = Pic1dElectrode()
    right: Pic1dElectrode = Pic1dElectrode()
    init_density_m3: float = Field(
        ..., gt=0, description="初期プラズマ密度 (一様、準中性) [m^-3]", json_schema_extra=ui("m^-3")
    )
    init_te_ev: float = Field(2.0, gt=0, json_schema_extra=ui("eV"))
    init_ti_ev: float = Field(0.03, gt=0, json_schema_extra=ui("eV"))
    ion_mass_amu: float = Field(39.948, gt=0, description="イオン質量 [amu] (He: 4.0026)", json_schema_extra=ui("amu"))
    n_macro: int = Field(20000, gt=0, description="種ごとの初期マクロ粒子数")
    dt: float | None = Field(None, gt=0, description="秒。None なら 0.1/ωpe (初期密度から)", json_schema_extra=ui("s"))
    n_steps: int = Field(2000, gt=0)
    frame_every: int = Field(20, gt=0)
    # 完了時に返す時間平均プロファイルの平均ステップ数。None なら最後の25% (2D と同じ規約)
    avg_steps: int | None = Field(None, gt=0)
    # RF 1周期の位相分解ビン数。0=無効、RF (voltage_rf/waveforms) が無ければ無効
    # (基本周波数の決定優先順位は pic1d.py の Pic1dSimulation._cycle_freq 算出コメント参照)
    phase_bins: int = Field(40, ge=0)
    mcc: MccSettings | None = None  # 既存 MccSettings をそのまま流用 (null なら MCC 無効)
    see_energy_ev: float = Field(2.0, ge=0, description="SEE 電子の初期エネルギー [eV]", json_schema_extra=ui("eV"))
    eedf_regions: list[Eedf1dRegion] = []  # 最大4個 (validator)
    # 壁 IEDF (入射イオンエネルギー分布、prompts/116) のビン数。0=無効。平均区間中に
    # 壁 (左右) で吸収されたイオンの全運動エネルギーを重み付きヒストグラム化する
    # (粒子ベースの厳密な値。e_max は EEDF (eedf_regions) と同じ流儀で自動決定する)
    wall_iedf_bins: int = Field(100, ge=0, le=1000)
    seed: int = Field(0, json_schema_extra=ui(advanced=True))  # 初期装荷の乱数種 (MCC は mcc.seed を使う)
    # 収束の判定 (prompts/137)
    convergence: ConvergenceSettings = ConvergenceSettings()

    @model_validator(mode="after")
    def _check_no_dsmc(self) -> "Pic1dSettings":
        if self.mcc is not None and self.mcc.use_dsmc_gas:
            raise ValueError(
                "1D PIC (pic1d) は DSMC 連成 (mcc.use_dsmc_gas) に未対応です "
                "(1D は専用の一様格子ソルバーで、2D メッシュ/DSMC ガス場を参照できません)"
            )
        return self

    @model_validator(mode="after")
    def _check_eedf_regions(self) -> "Pic1dSettings":
        if len(self.eedf_regions) > 4:
            raise ValueError("eedf_regions は最大 4 個までです")
        return self


# ---- boltzpm (Boltzmann ソルバー) 連携 — LMEA 流体係数テーブル (prompts/117) -----------
#
# boltz.py 参照。E/N を掃引して各点の定常 EEDF から ε̄=⟨ε⟩・μ_e・N・レート係数を
# 求め、ε̄ をキーにテーブル化したもの (BOLSIG+ 流の局所平均エネルギー近似、LMEA)。
# Fluid1dSettings/Fluid2dSettings の electron_model="boltzmann" のときに使う。


class BoltzTable(BaseModel):
    """boltzpm による LMEA 係数テーブル (prompts/117)。ε̄=(3/2)Te をキーに参照する。

    各リストは ε̄ (mean_energy_ev) 昇順に整列済み (boltz.run_boltz_sweep が保証する)。
    eedf は表示用に全点分を保存する (eedf[i] が eedf_eps_ev グリッド上の EEDF、
    ∫eedf[i] dε ≈ 1 — boltzpm の規格化そのまま、boltz.py モジュール docstring 参照)。
    """

    en_td: list[float]
    mean_energy_ev: list[float]
    mobility_n: list[float]   # μ_e・N [1/(m・V・s)]
    k_ion: list[float]
    k_exc: list[float]
    e_ion_ev: list[float]
    e_exc_ev: list[float]
    eedf_eps_ev: list[float]
    eedf: list[list[float]]
    source_hash: str         # sha256(JSON(processes)) — 断面積変更後の再生成判定用
    opts: dict = {}
    warnings: list[str] = []


# ---- 1D プラズマ流体 (ドリフト拡散 + 電子エネルギー、prompts/104-106) -----------------------
#
# pic1d (上記 Pic1dSettings) と同一条件・同一プリセットで直接比較できることが設計目標
# なので、格子規約 (n_cells/n_nodes/xg)・電極 (Pic1dElectrode)・診断の形は意図的に
# pic1d に揃えている (fluid1d.py 参照)。


class Fluid1dSettings(BaseModel):
    """1D プラズマ流体 (ドリフト拡散 + 電子エネルギー、prompts/104-106)。null なら無効。

    Scharfetter-Gummel フラックス + 半陰的時間積分 (fluid1d.py) で n_e・n_i・電子
    エネルギー w・φ を解く。pic1d 同様 geometry/mesh とは無関係な専用の一様格子
    ソルバー。
    """

    gap_m: float = Field(..., gt=0, description="電極間ギャップ [m]", json_schema_extra=GEOM)
    n_cells: int = Field(200, ge=16, le=100000)
    left: Pic1dElectrode = Pic1dElectrode()   # 電圧合成・SEE γ を共用 (fn は未対応)
    right: Pic1dElectrode = Pic1dElectrode()
    init_density_m3: float = Field(
        ..., gt=0, description="初期プラズマ密度 (一様、準中性) [m^-3]", json_schema_extra=ui("m^-3")
    )
    init_te_ev: float = Field(2.0, gt=0, json_schema_extra=ui("eV"))
    gas_pressure_pa: float = Field(..., gt=0, description="一様背景ガス圧 [Pa]", json_schema_extra=ui("Pa"))
    gas_temperature_k: float = Field(300.0, gt=0, json_schema_extra=ui("K"))
    ion_mass_amu: float = Field(39.948, gt=0, description="イオン質量 [amu]", json_schema_extra=ui("amu"))
    # イオン低電界移動度 μ_i の基準値・基準ガス密度 (任意のガス密度へは
    # μ_i = mu_i_ref・(n_ref_m3/n_g) でスケールする、fluid1d.py 参照)。既定は
    # Ar+ in Ar の 1 Torr (133.3 Pa, 300K) 換算実測値
    mu_i_ref: float = Field(
        1.45e-1, gt=0, description="μ_i の基準値 [m^2/(V・s)] (n_ref_m3 にて)", json_schema_extra=ui("m^2/(V*s)")
    )
    n_ref_m3: float = Field(3.22e22, gt=0, description="mu_i_ref の基準ガス密度 [m^-3]", json_schema_extra=ui("m^-3"))
    t_i_ev: float = Field(0.026, gt=0, description="イオン温度 (D_i = μ_i・T_i)", json_schema_extra=ui("eV"))
    # 修正 Frost イオン移動度 (prompts/116): μ_i(E/N) = μ_L/√(1+(E/N)/C) で
    # シース強電界での移動度低下を表現する。μ_L は上の mu_i_ref/n_ref_m3 から決まる
    # 低電界値、E/N [Td] = |E|/n_g/1e-21。C=frost_c_td は μ_L/√2 に落ちる E/N。
    # 既定 150 Td は Ar+ in Ar の実測 (Ellis et al., At. Data Nucl. Data Tables 17,
    # 177 (1976)) に対する粗いフィットであり精密フィットではない工学近似 —
    # ガス種が変わる場合は調整が必要 (fluid1d.py frost_mobility 参照)。
    # "const" は従来 (低電界一定値) の経路で、既存プロジェクト/テストとのビット
    # 不変を保つために残す (新規機能のため既定は "frost")
    ion_mobility_model: Literal["frost", "const"] = "frost"
    frost_c_td: float = Field(150.0, gt=0, description="修正 Frost 式の C [Td] (const では無視)", json_schema_extra=ui("Td"))
    electron_processes: list[XsProcess] = []  # 空なら eduPIC Ar 解析式を既定使用
    dt: float | None = Field(
        None, gt=0, description="秒。None なら RF周期/2000 と 1e-10 の小さい方", json_schema_extra=ui("s")
    )
    n_steps: int = Field(20000, gt=0)
    frame_every: int = Field(200, gt=0)
    avg_steps: int | None = Field(None, gt=0)
    phase_bins: int = Field(40, ge=0)
    # 壁 IEDF (入射イオンエネルギー分布、prompts/116) のビン数。0=無効。
    # 流体は粒子を持たないため、位相分解シース電圧 + イオン走行時間フィルタで
    # 再構成する工学近似 (無衝突シース・CX 衝突なしを仮定、fluid1d.py 参照)
    wall_iedf_bins: int = Field(100, ge=0, le=1000)
    # 収束の判定 (prompts/137)
    convergence: ConvergenceSettings = ConvergenceSettings()
    # seed は不要 (流体は決定論的で乱数を使わない)
    # 電子輸送・反応係数のソース (prompts/117): "maxwell" (既定) は fluid_coeffs.py の
    # Maxwell 平均 (従来経路、ビット不変)。"boltzmann" は boltz_table (boltzpm による
    # LMEA テーブル、事前に /ws/boltz で生成してフロントが埋め込む) を ε̄=(3/2)Te で
    # 参照する (fluid1d.py の _te_and_coeffs 参照)
    electron_model: Literal["maxwell", "boltzmann"] = "maxwell"
    boltz_table: BoltzTable | None = None

    @model_validator(mode="after")
    def _check_no_fn(self) -> "Fluid1dSettings":
        if self.left.fn is not None or self.right.fn is not None:
            raise ValueError(
                "fluid1d は FN 電界放出 (left/right.fn) に未対応です "
                "(Pic1dElectrode を共用していますが fn は無視されないよう明示的に禁止しています)"
            )
        return self

    @model_validator(mode="after")
    def _check_boltz_table(self) -> "Fluid1dSettings":
        if self.electron_model == "boltzmann" and self.boltz_table is None:
            raise ValueError(
                "electron_model='boltzmann' には boltz_table (boltzpm による LMEA テーブル、"
                "/ws/boltz で事前生成) の指定が必要です"
            )
        return self


# ---- 2D/軸対称 プラズマ流体 (EAFE/FEM-SG、prompts/111) -----------------------------
#
# fluid1d の 2D/軸対称拡張。geometry/mesh・境界条件 (Dirichlet 電圧・voltage_rf・
# waveform・symmetry・see_gamma・conductor/dielectric 領域) は既存のプロジェクト設定を
# そのまま使う (pic (2D PIC) と同じ流儀) ため、fluid1d のように専用の
# 左右電極フィールドは持たない。


class Fluid2dSettings(BaseModel):
    """2D/軸対称 プラズマ流体 (ドリフト拡散 + 電子エネルギー、EAFE/FEM-SG、prompts/111)。

    null なら無効。geometry/mesh (非構造 or 構造格子の三角形メッシュ)・境界条件は
    既存のプロジェクト設定 (Project.geometry.boundaries の dirichlet/symmetry、
    Region の conductor/dielectric/voltage/voltage_rf/see_gamma) をそのまま使う —
    2D PIC (pic.py) と同一条件で直接比較できることが設計目標。
    periodic 境界は未対応 (fluid2d.py の Fluid2dSimulation.__init__ で ValueError)。
    """

    init_density_m3: float = Field(
        ..., gt=0, description="初期プラズマ密度 (一様、準中性) [m^-3]", json_schema_extra=ui("m^-3")
    )
    init_te_ev: float = Field(2.0, gt=0, json_schema_extra=ui("eV"))
    gas_pressure_pa: float = Field(..., gt=0, description="一様背景ガス圧 [Pa]", json_schema_extra=ui("Pa"))
    gas_temperature_k: float = Field(300.0, gt=0, json_schema_extra=ui("K"))
    ion_mass_amu: float = Field(39.948, gt=0, description="イオン質量 [amu]", json_schema_extra=ui("amu"))
    # イオン低電界移動度 (fluid1d.py と同じ規約: μ_i = mu_i_ref・(n_ref_m3/n_g))
    mu_i_ref: float = Field(
        1.45e-1, gt=0, description="μ_i の基準値 [m^2/(V・s)] (n_ref_m3 にて)", json_schema_extra=ui("m^2/(V*s)")
    )
    n_ref_m3: float = Field(3.22e22, gt=0, description="mu_i_ref の基準ガス密度 [m^-3]", json_schema_extra=ui("m^-3"))
    t_i_ev: float = Field(0.026, gt=0, description="イオン温度 (D_i = μ_i・T_i)", json_schema_extra=ui("eV"))
    # 修正 Frost イオン移動度 (fluid1d.py と全く同じ規約・既定値・出典。prompts/116)
    ion_mobility_model: Literal["frost", "const"] = "frost"
    frost_c_td: float = Field(150.0, gt=0, description="修正 Frost 式の C [Td] (const では無視)", json_schema_extra=ui("Td"))
    electron_processes: list[XsProcess] = []  # 空なら eduPIC Ar 解析式を既定使用
    dt: float | None = Field(
        None, gt=0, description="秒。None なら RF周期/2000 と 1e-10 の小さい方", json_schema_extra=ui("s")
    )
    n_steps: int = Field(20000, gt=0)
    frame_every: int = Field(200, gt=0)
    avg_steps: int | None = Field(None, gt=0)
    phase_bins: int = Field(0, ge=0)
    # 収束の判定 (prompts/137)
    convergence: ConvergenceSettings = ConvergenceSettings()
    # seed は不要 (流体は決定論的で乱数を使わない)
    # 電子輸送・反応係数のソース (fluid1d.py と同じ規約・既定値、prompts/117)
    electron_model: Literal["maxwell", "boltzmann"] = "maxwell"
    boltz_table: BoltzTable | None = None

    @model_validator(mode="after")
    def _check_boltz_table(self) -> "Fluid2dSettings":
        if self.electron_model == "boltzmann" and self.boltz_table is None:
            raise ValueError(
                "electron_model='boltzmann' には boltz_table (boltzpm による LMEA テーブル、"
                "/ws/boltz で事前生成) の指定が必要です"
            )
        return self

    # ---- 陰的線形ソルバー (prompts/115、毎ステップの spsolve 3本が実測93%を占めていた
    #      プロファイルへの対応) ----------------------------------------------------
    # "iterative" (既定): numba 並列 (行並列 matvec) の Jacobi-BiCGSTAB。反復法の matvec は
    #   SuperLU の直接分解と違って完全に並列化でき、しかも行並列がビット決定論を保てる。
    #   収束しなかった場合は自動的に spsolve (direct) へフォールバックする (堅牢性優先)。
    # "direct": 従来の scipy.sparse.linalg.spsolve (SuperLU の都度分解)。比較・検証用に残す。
    linear_solver: Literal["iterative", "direct"] = "iterative"
    # 陰的反復ソルバーの並列スレッド数 (numba の matvec に使う)。0=自動選択
    # (pic.py の _auto_thread_cap と同じ式: max(2, min(16, 論理コア数//2))。
    # linear_solver="direct" のときは無効 (spsolve は並列化しない、docstring 参照)
    threads: int = Field(0, ge=0, le=128, json_schema_extra=ui(advanced=True))


# ---- VHF 定在波 (非線形径方向伝送線路モデル、prompts/101) -----------------------------


class TlSettings(BaseModel):
    """VHF 定在波スタディ (非線形径方向伝送線路モデル、prompts/101)。null なら無効。

    円板電極 (半径 radius_m、ギャップ gap_m)・中心給電・軸対称の径方向 1D モデル
    (tl.py 参照)。geometry/mesh とは無関係な専用の一様格子ソルバー (pic1d と同じ位置づけ)。
    """

    radius_m: float = Field(0.15, gt=0, json_schema_extra=GEOM)        # 電極半径 R
    gap_m: float = Field(0.04, gt=0, json_schema_extra=GEOM)           # ギャップ l
    sheath_m: float = Field(5e-4, gt=0, json_schema_extra=GEOM)        # 平衡シース厚 s0 (片側、上下対称)
    n_e_m3: float = Field(1e16, gt=0, json_schema_extra=ui("m^-3"))    # バルク電子密度
    n_s_ratio: float = Field(0.4, gt=0, le=1, json_schema_extra=ui("1"))  # シース端イオン密度比 n_s/n_e (h係数)
    nu_m_hz: float = Field(1e8, ge=0, json_schema_extra=ui("Hz"))      # 電子運動量衝突周波数 ν_m
    freq_hz: float = Field(100e6, gt=0, json_schema_extra=ui("Hz"))    # 駆動周波数 f0
    v0: float = Field(100.0, gt=0, json_schema_extra=ui("V"))          # 駆動振幅 [V]
    n_r: int = Field(400, ge=32, le=20000)     # 半径方向節点数
    n_periods: int = Field(200, ge=8)          # 総周期数
    n_fft_periods: int = Field(32, ge=4)       # FFT 窓の周期数 (n_periods より小)
    n_harm: int = Field(10, ge=1, le=40)       # 返す高調波次数
    dt: float | None = Field(None, gt=0, json_schema_extra=ui("s"))  # 秒。None なら CFL から自動
    # シースの電荷-電圧関係 (prompts/102)。"child": Child-Langmuir 型 (V_s∝q^{4/3})、
    # 既定。対称放電でも上下差し引きで奇数次高調波が定常生成される (tl.py 参照)。
    # "matrix": 行列シース (V_s∝q^2、従来モデル)。対称放電では厳密に線形化し
    # 高調波はクリップ過渡でしか出ない (比較・線形極限検証用に残す)。
    sheath_law: Literal["child", "matrix"] = "child"

    @model_validator(mode="after")
    def _check_tl(self) -> "TlSettings":
        if self.n_fft_periods >= self.n_periods:
            raise ValueError("n_fft_periods は n_periods より小さくしてください")
        if self.sheath_m * 2 >= self.gap_m:
            raise ValueError("sheath_m の2倍は gap_m より小さくしてください (バルク厚が正である必要があります)")
        return self


# ---- DSMC (定常ガス流れ、prompts/54) ------------------------------------------


class DsmcGas(BaseModel):
    """DSMC のガス分子モデル (VHS: Variable Hard Sphere)。既定は Ar。"""

    name: str = "Ar"
    mass_amu: float = Field(39.948, gt=0, description="分子質量 [amu]", json_schema_extra=ui("amu"))
    d_ref_m: float = Field(4.17e-10, gt=0, description="VHS 基準直径 [m] (T_ref にて)", json_schema_extra=ui("m"))
    omega: float = Field(0.81, ge=0.5, le=1.0, description="粘性の温度指数 ω (HS=0.5)", json_schema_extra=ui("1"))
    t_ref_k: float = Field(273.0, gt=0, description="基準温度 [K]", json_schema_extra=ui("K"))


class DsmcBoundary(BaseModel):
    """DSMC 境界条件。未指定の境界は拡散反射壁になる。

    適用範囲は edges (domain 外周のエッジ番号) と p1-p2 (外周上の線分、部分区間
    指定。prompts/55) のどちらでも指定できる (両方指定は和集合)。電極と外枠の
    隙間などエッジの一部だけを流入口にしたい場合は線分指定を使う。

    - "wall":     拡散反射 (完全適応、temperature_k で再放出)
    - "symmetry": 鏡面反射
    - "inlet":    圧力リザーバ (pressure_pa: 平衡流入 + 流出吸収) または
                  流量指定 (flow_sccm: 指定流量を注入し、入射粒子は拡散反射壁。
                  正味流量が指定値に厳密一致する。2D なので奥行き 1 m 換算)
    - "outlet":   圧力リザーバまたは真空 (pressure_pa 省略/0 = 真空排気)
    """

    edges: list[int] = []
    p1: Point | None = Field(None, json_schema_extra=GEOM)
    p2: Point | None = Field(None, json_schema_extra=GEOM)
    type: Literal["wall", "symmetry", "inlet", "outlet"] = "wall"
    temperature_k: float = Field(300.0, gt=0, json_schema_extra=ui("K"))
    pressure_pa: float | None = Field(None, ge=0, json_schema_extra=ui("Pa"))
    # inlet の流量指定 [sccm] (標準状態 273.15 K・101325 Pa の cm^3/min)。
    # pressure_pa と排他。1 sccm = 4.478e17 分子/s
    flow_sccm: float | None = Field(None, gt=0, json_schema_extra=ui("sccm"))

    @model_validator(mode="after")
    def _check(self) -> "DsmcBoundary":
        if not self.edges and (self.p1 is None or self.p2 is None):
            raise ValueError("境界の適用範囲を edges か p1/p2 (線分) で指定してください")
        if self.type == "inlet":
            has_p = bool(self.pressure_pa and self.pressure_pa > 0.0)
            has_f = self.flow_sccm is not None
            if has_p == has_f:
                raise ValueError(
                    "inlet には pressure_pa (> 0) か flow_sccm のどちらか一方を指定してください"
                )
        elif self.flow_sccm is not None:
            raise ValueError("flow_sccm は inlet でのみ指定できます")
        return self


class DsmcSettings(BaseModel):
    """定常ガス流れの DSMC 設定 (prompts/54)。null なら無効。

    NTC 法 + VHS 分子モデル。既存の三角形メッシュをセルとして使い、
    定常後の時間平均で要素ごとの n・T・u を得る (MCC の背景ガス場に使える)。
    平面2D (coord="xy") のみ対応。
    """

    gas: DsmcGas = DsmcGas()
    boundaries: list[DsmcBoundary] = []
    # DSMC 用メッシュの寸法係数 (prompts/89)。FEM メッシュ寸法 (mesh.size・local_sizes の
    # 各 size) × この係数で DSMC 専用の粗いメッシュを生成する (PIC/FEM メッシュとは分離)。
    # walk コストはセル寸法 (≈1/h) に反比例するため、粗化係数分だけ直接軽くなる。
    # 1.0 = 従来どおり FEM と同一メッシュ (このとき経路も従来と完全一致し、結果はビット不変)。
    # use_dsmc_gas (PIC 連成) では、mesh_scale>1 で DSMC メッシュ ≠ PIC メッシュになるため、
    # PIC 開始時に要素重心の点位置特定でガス場を PIC メッシュへ引き写す (dsmc.py 側で解決)
    mesh_scale: float = Field(1.0, ge=1.0, le=20.0, json_schema_extra=ui("1", advanced=True))
    wall_temperature_k: float = Field(
        300.0, gt=0, description="未指定エッジ・領域輪郭の壁温 [K]", json_schema_extra=ui("K")
    )
    init_pressure_pa: float = Field(..., gt=0, description="初期充填圧 [Pa]", json_schema_extra=ui("Pa"))
    init_temperature_k: float = Field(300.0, gt=0, json_schema_extra=ui("K"))
    n_particles: int = Field(50000, gt=0, description="目標シミュレーション粒子数")
    dt: float | None = Field(None, description="秒。None なら 0.25·h_min/v_mp から自動", json_schema_extra=ui("s"))
    n_steps: int = Field(2000, gt=0)
    avg_steps: int = Field(500, gt=0, description="最終 N ステップで時間平均")
    seed: int = Field(0, json_schema_extra=ui(advanced=True))
    # 粒子処理 (walk 探索) のワーカースレッド数 (prompts/65)。粒子ごとの walk は独立な
    # ため、チャンク並列化しても結果は逐次実行とビット単位で一致する。1 = 従来経路
    threads: int = Field(1, ge=1, le=128, json_schema_extra=ui(advanced=True))
    # 隣接セル拡散による統計ノイズ平滑化の回数 (0=無効、prompts/67)。導出前の生モーメント
    # (Σ個数・Σv・Σv²) に体積重み対称拡散を適用してから n/T/u/p を導出するため、
    # p = n kB T の整合を保ったまま総量 (質量・運動量・エネルギー) を厳密に保存する
    smoothing_passes: int = Field(0, ge=0, le=20, json_schema_extra=ui(advanced=True))


class Param(BaseModel):
    """名前付きの式 (P7f、params.py の文法。値は SI)。"""

    name: str
    expr: str
    # 単位は式による (SI)。フォームには出さない
    value: float | None = Field(None, description="式を計算した値 (SI)。UI とスイープが書く", json_schema_extra=ui("1"))
    description: str = ""


class ParamBinding(BaseModel):
    """設定の数値の欄に付けた式 (path は文書の中の道筋: キー・番号・{"id": …}・{"edge": 辺の ID})。"""

    path: list[str | int | dict[str, str]] = Field(..., min_length=1)
    expr: str


class Params(BaseModel):
    """パラメータと式の束縛 (P7f、prompts/132)。欄には計算した数値も入れておき、ソルバーはそれを使う"""

    vars: list[Param] = []
    bindings: list[ParamBinding] = []

    @model_validator(mode="after")
    def _check(self) -> "Params":
        try:
            evaluate_params([v.model_dump() for v in self.vars])
        except ExprError as exc:
            raise ValueError(str(exc)) from exc
        known = {v.name for v in self.vars}
        for b in self.bindings:
            where = "束縛 " + ".".join(str(p) for p in b.path)
            try:
                unknown = names_in(b.expr) - known
            except ExprError as exc:
                raise ValueError(f"{where}: {exc}") from exc
            if unknown:
                raise ValueError(f"{where}: {', '.join(sorted(unknown))} というパラメータはありません")
        return self


class Project(BaseModel):
    # 知らないキーは読み込みで捨てる (pydantic の既定)。初版の "solver": {"backend": "numpy" | "cupy" | "auto"}
    # (計算には効いていなかった。GPU を使うかは mesh.mode と es_sim.device で決まる) は消したが、それを含む
    # 古い文書もそのまま読める
    version: int = 1
    unit: Literal["m", "mm"] = "m"
    # 座標系 (prompts/39, 41)。"xy": 平面2D (従来)。
    # "rz":    軸対称 — x = z (軸方向)、y = r (径方向)。対称軸は y=0 (自然境界)
    # "rz_x0": 軸対称 — x = r (径方向)、y = z (軸方向)。対称軸は x=0 (自然境界)
    coord: Literal["xy", "rz", "rz_x0"] = "xy"
    geometry: Geometry
    mesh: MeshSettings
    # 一様磁場 [T] (prompts/51)。null または全成分 0 で磁場なし (従来と完全一致)
    b_field: BField | None = None
    # 定常ガス流れの DSMC 設定 (prompts/54)。null なら無効
    dsmc: DsmcSettings | None = None
    particles: ParticleSettings | None = None
    pic: PicSettings | None = None
    # 1D PIC/MCC (1d3v、prompts/91)。null なら無効。geometry/mesh とは無関係に動く
    # 専用の一様格子ソルバー (pic1d.py)。2D の pic と同時に設定しても互いに独立に扱われる
    pic1d: Pic1dSettings | None = None
    # 1D プラズマ流体 (ドリフト拡散 + 電子エネルギー、prompts/104-106)。null なら無効。
    # pic1d と同一条件で比較できるよう設計された専用ソルバー (fluid1d.py)
    fluid1d: Fluid1dSettings | None = None
    # 2D/軸対称 プラズマ流体 (ドリフト拡散 + 電子エネルギー、EAFE/FEM-SG、prompts/111)。
    # null なら無効。geometry/mesh・境界条件は既存のプロジェクト設定をそのまま使う
    fluid2d: Fluid2dSettings | None = None
    # VHF 定在波 (非線形径方向伝送線路モデル、prompts/101)。null なら無効。
    # pic1d 同様 geometry/mesh とは無関係な専用ソルバー (tl.py)
    tl: TlSettings | None = None
    # UI の CAD の状態 (prompts/132): スケッチ (領域でない線・円弧・円・ポリライン) など。ソルバーは使わないので
    # 中身は検査しない (壊れていても計算は止めない)
    cad: dict | None = Field(None, description="UI の CAD の状態 (スケッチ・レイヤなど)。ソルバーは使わない")
    # パラメータと式の束縛 (P7f)。欄には計算した数値が入っているのでソルバーは使わない (スイープで式を計算し直す)
    params: Params | None = None

    @model_validator(mode="after")
    def _flatten_arcs(self) -> "Project":
        """円弧 (bulges) を弦に分け、ソルバーには直線だけの多角形を渡す (prompts/132)。

        - 分ける幅 h: 領域は局所メッシュ幅 (local_sizes) か全体のメッシュ幅、ドメインは全体のメッシュ幅。
          直交格子 (cartesian) で AMR があれば最も細かいレベルの幅 (2^-L 倍)。密度は円の多角形化と同じ。
        - 外周の辺の番号を持つ参照 (境界条件・FN 放出・PIC の反射・DSMC の境界) は分けた弦の番号に展開し、
          元の辺の番号を Geometry._edge_origin に残す (電荷のラベルは元の番号、Geometry.edge_label)。
        - 円弧の無い文書は何も変えない (今までと完全に同じ)。
        この検証はほかの Project の検証より先に定義してあり (定義順に実行される)、軸対称の r ≥ 0・軸上の
        Dirichlet の検査は分けたあとの形で行う (円弧のふくらみが軸をまたぐのも検出する)。
        """
        geo = self.geometry
        dom = geo.domain
        region_arcs = any(r.polygon is not None and (has_arcs(r.bulges) or any(has_arcs(h.bulges) for h in r.holes)) for r in geo.regions)
        if not has_arcs(dom.bulges) and not region_arcs:
            return self
        scale = 1.0
        amr = self.mesh.amr
        if self.mesh.mode == "cartesian" and amr is not None:
            scale = 0.5 ** max([amr.max_level, *(r.level for r in amr.regions)])
        local = {ls.region: ls.size for ls in self.mesh.local_sizes}
        for r in geo.regions:
            if r.polygon is None:
                continue
            h = local.get(r.id, self.mesh.size) * scale
            if has_arcs(r.bulges):
                r.polygon, _ = flatten_path(r.polygon, r.bulges, h)
                r.bulges = None
            for hole in r.holes:
                if has_arcs(hole.bulges):
                    hole.polygon, _ = flatten_path(hole.polygon, hole.bulges, h)
                    hole.bulges = None
        if not has_arcs(dom.bulges):
            return self
        n = len(dom.polygon)
        pts, origin = flatten_path(dom.polygon, dom.bulges, self.mesh.size * scale)
        expand: dict[int, list[int]] = {}
        for k, o in enumerate(origin):
            expand.setdefault(o, []).append(k)

        def remap(edges: list[int], what: str) -> list[int]:
            out: list[int] = []
            for e in edges:
                if not 0 <= e < n:
                    raise ValueError(f"{what} の辺の番号 {e} が範囲外です (0..{n - 1})")
                out.extend(expand[e])
            return out

        for bc in geo.boundaries:
            bc.edges = remap(bc.edges, "境界条件")
        if self.particles is not None and self.particles.fn is not None:
            self.particles.fn.edges = remap(self.particles.fn.edges, "particles.fn")
        if self.pic is not None:
            self.pic.reflect_edges = remap(self.pic.reflect_edges, "pic.reflect_edges")
            if self.pic.fn is not None:
                self.pic.fn.edges = remap(self.pic.fn.edges, "pic.fn")
        if self.dsmc is not None:
            for b in self.dsmc.boundaries:
                b.edges = remap(b.edges, "dsmc.boundaries")
        dom.polygon = pts
        dom.bulges = None
        # 弦に分けたあとは辺と ID が 1 対 1 でなくなる (ソルバーは使わない)
        dom.edge_ids = None
        geo._edge_origin = origin
        return self

    @model_validator(mode="after")
    def _check_b_field(self) -> "Project":
        if self.coord != "xy" and self.b_field is not None and not self.b_field.is_zero():
            raise ValueError(
                "一様磁場 (b_field) は平面2D (coord='xy') のみ対応です "
                "(軸対称モードでは未対応)"
            )
        return self

    @model_validator(mode="after")
    def _check_rz(self) -> "Project":
        """軸対称モード (rz / rz_x0) の制約検査。

        径方向座標 (rz: y、rz_x0: x) について、
        - domain の全頂点が r ≥ 0 であること
        - r=0 (対称軸) 上の辺への Dirichlet 指定は禁止 (対称軸は自然境界)
        """
        if self.coord == "xy":
            return self
        ridx = 1 if self.coord == "rz" else 0  # 径方向座標インデックス
        axis = "y" if ridx == 1 else "x"
        poly = self.geometry.domain.polygon
        scale = max((max(abs(p[0]), abs(p[1])) for p in poly), default=1.0)
        tol = 1e-12 * (scale if scale > 0.0 else 1.0)
        if any(p[ridx] < -tol for p in poly):
            raise ValueError(
                f"{self.coord} (軸対称) モードでは domain の全頂点が "
                f"{axis} (= r) ≥ 0 である必要があります"
            )
        n = len(poly)
        for bc in self.geometry.boundaries:
            if bc.type != "dirichlet":
                continue
            for e in bc.edges:
                p1, p2 = poly[e % n], poly[(e + 1) % n]
                if abs(p1[ridx]) <= tol and abs(p2[ridx]) <= tol:
                    raise ValueError(
                        f"{self.coord} (軸対称) モードでは対称軸 ({axis} = 0) 上の辺 "
                        f"(エッジ {e}) に Dirichlet を指定できません (対称軸は自然境界です)"
                    )
        return self


# ---- API レスポンス ----------------------------------------------------------


class DsmcResultModel(BaseModel):
    """POST /dsmc のレスポンス (定常時間平均のガス場、prompts/54)。"""

    mesh: "MeshResult"
    n: list[float]                    # 要素ごとの数密度 [m^-3]
    t: list[float]                    # 要素ごとの温度 [K]
    u: list[tuple[float, float]]      # 要素ごとの面内流速 [m/s]
    p: list[float]                    # 要素ごとの圧力 [Pa] = n kB T
    n_particles: int                  # 最終シミュレーション粒子数
    macro_weight: float               # 実分子数/シミュレーション粒子
    dt: float                         # 実際に使った dt [s]
    inflow: float                     # 平均区間の流入実分子数
    outflow: float                    # 平均区間の流出実分子数
    elapsed_s: float                  # run() の壁時計秒 (continue は区間分のみ、prompts/86)
    timing: dict[str, float]          # 位相別の累積秒 (continue は区間分のみ、prompts/87)


class MeshResult(BaseModel):
    nodes: list[Point]                  # 節点座標 [m]
    triangles: list[tuple[int, int, int]]  # 要素 → 節点番号
    region_of_triangle: list[int]       # 要素 → regions のインデックス (-1: 背景=真空)


class ElectrodeCharge(BaseModel):
    """電極 (Dirichlet エッジ or conductor 領域) ごとの誘起電荷 (fem.py 残差法。prompts/71)。"""

    label: str      # "edge0".."edge3" (domain 外周) または conductor の region id
    voltage: float  # 電極電位 [V] (直流分)
    q: float        # 誘起電荷。xy: [C/m] (奥行き1m あたり)、軸対称: [C]


class SolveResult(BaseModel):
    mesh: MeshResult
    v: list[float]                      # 節点電位 [V]
    e_field: list[tuple[float, float]]  # 要素ごとの E = -∇V [V/m]
    v_min: float
    v_max: float
    e_abs_max: float
    energy: float                       # 蓄積エネルギー W = 1/2 ∫ ε|E|^2 dΩ [J/m (奥行き単位)]
    charges: list[ElectrodeCharge] = []  # 電極ごとの誘起電荷 (prompts/71)
    # 静電容量。電極電位がちょうど2水準かつ空間電荷が全域0の場合のみ定義 (それ以外は null)。
    # xy: [F/m]、軸対称: [F] (prompts/71)
    capacitance: float | None = None


class ProfileRequest(BaseModel):
    project: Project
    p1: Point
    p2: Point
    n: int = 200


class ProfileResult(BaseModel):
    s: list[float]                # 弧長 (p1 からの距離) [m]
    v: list[float | None]         # 電位 [V] (領域外は None)
    e_abs: list[float | None]     # |E| [V/m] (領域外は None)


class LxcatParseRequest(BaseModel):
    """POST /lxcat/parse のリクエスト。"""

    text: str
    species: Literal["electron", "ion"]


class LxcatParseResult(BaseModel):
    processes: list[XsProcess]
    warnings: list[str]


class TraceResult(BaseModel):
    trajectories: list[list[Point]]            # 粒子ごと、save_every ステップごと (初期位置含む)
    status: list[Literal["absorbed", "alive"]]
    tof: list[float | None]                    # absorbed 粒子の飛行時間 [s]
    final_energy_ev: list[float]
    final_angle_deg: list[float]                # 最終速度の向き [度] (x軸から反時計回り, -180〜180)
    dt: float                                  # 実際に使った dt [s]
    # FN 電界放出 (prompts/46、fn 指定時のみ非 None):
    # 粒子ごとの担持電流と総放出電流。単位は xy: [A/m] (奥行き1m)、rz/rz_x0: [A]
    currents: list[float] | None = None
    fn_current: float | None = None
