"""プロジェクト JSON のスキーマ (pydantic)。仕様書 §10 参照。

このモデルがプロジェクトファイルの唯一の正。
フロントエンドの TypeScript 型 (src/types.ts) はこれと手動同期する。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Point = tuple[float, float]


class Domain(BaseModel):
    polygon: list[Point] = Field(..., min_length=3, description="解析領域の外周 (閉ポリゴン、反時計回り)")


class CircleShape(BaseModel):
    """円領域のパラメトリック形状。メッシュ生成時に多角形化する (meshing._region_polygon 参照)。"""

    kind: Literal["circle"] = "circle"
    center: Point
    radius: float = Field(..., gt=0)


class VoltageRF(BaseModel):
    """RF 電圧成分 (フェーズ3)。V(t) = voltage + Σ_k amplitude_k * sin(2π f_k t + phase_k)。

    voltage_rf フィールドには単一成分 (VoltageRF) と成分リスト (list[VoltageRF]、
    デュアル周波数など) のどちらも指定できる (prompts/49)。
    静電ソルブ (/solve) は従来通り直流分 voltage のみを使い、PIC のみが V(t) を使う。
    """

    amplitude: float
    freq_hz: float = Field(..., gt=0)
    phase_deg: float = 0.0


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

    freq_hz: float = Field(..., gt=0)
    phase: list[float]  # 正規化位相 [0, 1) (昇順)
    v: list[float]       # 対応する電圧 [V]

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


class Region(BaseModel):
    id: str
    type: Literal["conductor", "dielectric", "charge"]
    polygon: list[Point] | None = Field(None, min_length=3)
    shape: CircleShape | None = None
    voltage: float | None = None  # conductor: 電位 [V] (直流分)
    # conductor: RF 成分 (PIC のみ使用)。単一またはリスト (デュアル周波数、prompts/49)
    voltage_rf: VoltageRF | list[VoltageRF] | None = None
    # conductor: CSV インポート波形 (PIC のみ使用、prompts/73)。voltage_rf と併用可
    # (V(t) = voltage + Σ RF + V_wf(t))。UI は境界条件辺のみ対応、スキーマ上のみ対応
    voltage_waveform: VoltageWaveform | None = None
    eps_r: float = 1.0            # dielectric: 比誘電率
    rho: float = 0.0              # charge: 電荷密度 [C/m^3]
    see_gamma: float = Field(
        0.0, ge=0,
        description="conductor / dielectric: 二次電子放出係数 γ (0 = 無効、PIC のみ使用)",
    )

    @model_validator(mode="after")
    def _check_polygon_xor_shape(self) -> "Region":
        if (self.polygon is None) == (self.shape is None):
            raise ValueError("Region には polygon か shape のどちらか一方のみを指定してください")
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
    voltage: float = 0.0
    # RF 成分 (PIC のみ使用)。単一またはリスト (デュアル周波数、prompts/49)
    voltage_rf: VoltageRF | list[VoltageRF] | None = None
    # CSV インポート波形 (PIC のみ使用、prompts/73)。voltage_rf と併用可
    # (V(t) = voltage + Σ RF + V_wf(t))。dirichlet のみ有効
    voltage_waveform: VoltageWaveform | None = None
    see_gamma: float = Field(0.0, ge=0, description="二次電子放出係数 γ (0 = 無効、PIC のみ使用)")

    @model_validator(mode="after")
    def _check_periodic_edge_count(self) -> "BoundaryCondition":
        if self.type == "periodic":
            if len(self.edges) != 2 or self.edges[0] == self.edges[1]:
                raise ValueError("periodic 境界には異なるエッジをちょうど2本指定してください")
        return self


class Geometry(BaseModel):
    domain: Domain
    regions: list[Region] = []
    boundaries: list[BoundaryCondition] = []

    @model_validator(mode="after")
    def _check_periodic_pairs(self) -> "Geometry":
        """periodic 境界の2辺が domain の平行・同長の対辺であることを検査する。"""
        poly = self.domain.polygon
        n = len(poly)
        for bc in self.boundaries:
            if bc.type != "periodic":
                continue
            for e in bc.edges:
                if not (0 <= e < n):
                    raise ValueError(f"periodic 境界のエッジ番号 {e} が範囲外です (0..{n - 1})")
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

    bx: float = 0.0
    by: float = 0.0
    bz: float = 0.0

    def is_zero(self) -> bool:
        return self.bx == 0.0 and self.by == 0.0 and self.bz == 0.0


class LocalSize(BaseModel):
    region: str
    size: float


class MeshSettings(BaseModel):
    size: float = Field(..., gt=0, description="全体特性長 [m]")
    local_sizes: list[LocalSize] = []
    # メッシュ生成モード (prompts/34)。structured は軸平行矩形 domain 専用の
    # 等間隔構造格子 (三角形2分割)。local_sizes は structured では無視される
    mode: Literal["unstructured", "structured"] = "unstructured"


class SolverSettings(BaseModel):
    backend: Literal["numpy", "cupy", "auto"] = "numpy"


# ---- 粒子軌道追跡 (フェーズ2、仕様書 §8) --------------------------------------


class Species(BaseModel):
    """粒子種。electron/proton プリセット、または custom で q・m を直接指定する。"""

    preset: Literal["electron", "proton", "custom"] = "electron"
    q: float | None = None  # custom 時の電荷 [C]
    m: float | None = None  # custom 時の質量 [kg]

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
    p1: Point
    p2: Point | None = None
    n: int = Field(..., gt=0)
    energy_ev: float = 0.0
    direction_deg: float = 0.0
    spread_deg: float = 0.0
    energy_dist: Literal["mono", "maxwell"] = "mono"  # "mono": 従来動作 / "maxwell": 熱速度成分を付加
    temperature_ev: float = Field(1.0, gt=0, description="maxwell 時の温度 kT [eV]")
    seed: int = 0  # maxwell サンプリングの乱数シード (再現性確保)

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
    phi_ev: float = Field(4.5, gt=0, description="仕事関数 φ [eV]")
    beta: float = Field(1.0, gt=0, description="電界増倍係数 β")
    n: int = Field(200, gt=0, description="trace 時の放出マクロ粒子総数")
    init_energy_ev: float = Field(0.1, ge=0, description="放出電子の初期エネルギー [eV]")
    # PIC のみ: マクロ重み (実電子数/マクロ粒子)。None なら初期プラズマの重みを使う
    macro_weight: float | None = Field(None, gt=0)
    seed: int = 0  # PIC の放出位置サンプリング乱数シード

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
    dt: float | None = None  # 秒。None なら自動推定 (particles.py 参照)
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

    density: float = Field(..., gt=0, description="数密度 [m^-3] (奥行き1m換算)")
    te_ev: float = Field(2.0, ge=0, description="電子温度 kTe [eV]")
    ti_ev: float = Field(0.03, ge=0, description="イオン温度 kTi [eV]")
    ion_mass_amu: float = Field(40.0, gt=0, description="イオン質量 [amu] (Ar+ = 40)")
    immobile_ions: bool = False  # true でイオン固定 (検証用)
    seed: int = 0


class PicInjection(BaseModel):
    """エミッタからの定常注入。電流を毎ステップの実電荷として等分注入する。

    current_a_per_m の単位: 平面2D (xy) = [A/m] (奥行き1m換算)、
    軸対称 (rz / rz_x0) = [A] (リングエミッタの全電流)。
    """

    emitter: Emitter
    species: Literal["electron", "ion"] = "electron"
    current_a_per_m: float = Field(..., gt=0)


# ---- MCC 衝突 (prompts/19、フロントと共通のスキーマ契約) ------------------------


class XsProcess(BaseModel):
    """LXCat 由来の衝突断面積プロセス (パース済み、プロジェクト JSON に埋め込む)。"""

    kind: Literal["elastic", "excitation", "ionization", "isotropic", "backscat"]
    label: str = ""                # PROCESS 行等から
    threshold_ev: float = 0.0      # excitation/ionization のみ >0
    mass_ratio: float = 0.0        # elastic のみ (m/M)。無ければ 0
    energy_ev: list[float]         # 断面積テーブルのエネルギー [eV] (昇順)
    sigma_m2: list[float]          # 断面積 [m^2] (energy_ev と同長)

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
    pressure_pa: float = Field(..., gt=0, description="ガス圧 [Pa]")
    temperature_k: float = Field(300.0, gt=0, description="ガス温度 [K]")


class MccSettings(BaseModel):
    """MCC 設定。null なら MCC 無効 (従来の無衝突動作)。"""

    gas: MccGas
    electron_processes: list[XsProcess] = []  # elastic/excitation/ionization
    ion_processes: list[XsProcess] = []       # isotropic/backscat
    seed: int = 0
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

    p1: Point
    p2: Point
    tol: float | None = Field(None, gt=0, description="判定距離 [m]。None なら mesh.size と同値")
    label: str = ""  # 表示用ラベル (空ならフロントが "C1" 等を振る、prompts/36)


class PicSettings(BaseModel):
    initial_plasma: InitialPlasma | None = None
    injection: PicInjection | None = None
    n_macro: int = Field(20000, gt=0, description="種ごとの初期マクロ粒子数の目安")
    dt: float | None = Field(None, description="秒。None なら 0.1/ωpe (初期密度から)")
    n_steps: int = Field(2000, gt=0)
    frame_every: int = Field(20, gt=0, description="フレーム送出間隔 (ステップ)")
    mcc: MccSettings | None = None  # null なら MCC 無効
    see_energy_ev: float = Field(2.0, ge=0, description="SEE 電子の初期エネルギー [eV]")
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
    # 鏡面反射する domain 外周エッジ番号のリスト (エッジ i は頂点 i → i+1)。
    # 到達粒子は吸収せず法線速度成分を反転して境界内へ折り返す (壁カウンタに含めない)。
    # 当該エッジは境界条件なし (Neumann) を想定。2D ストリップで 1D 問題を模擬する用途
    reflect_edges: list[int] = Field(default_factory=list)
    # FN 電界放出源 (prompts/46)。毎ステップの表面電界から I·dt 分の電子を放出する。
    # null なら無効 (従来動作と完全一致)
    fn: FnEmission | None = None
    # イオンサブサイクリング (prompts/50): イオンを N ステップに1回、N·dt で押す。
    # 休止ステップ中はイオンの電荷堆積をキャッシュして再利用する。1 = 無効 (従来と完全一致)
    ion_subcycle: int = Field(1, ge=1)
    # 粒子処理 (walk 探索) のワーカースレッド数 (prompts/50)。粒子ごとの walk は独立な
    # ため、チャンク並列化しても結果は逐次実行とビット単位で一致する。1 = 従来経路
    threads: int = Field(1, ge=1, le=32)


# ---- DSMC (定常ガス流れ、prompts/54) ------------------------------------------


class DsmcGas(BaseModel):
    """DSMC のガス分子モデル (VHS: Variable Hard Sphere)。既定は Ar。"""

    name: str = "Ar"
    mass_amu: float = Field(39.948, gt=0, description="分子質量 [amu]")
    d_ref_m: float = Field(4.17e-10, gt=0, description="VHS 基準直径 [m] (T_ref にて)")
    omega: float = Field(0.81, ge=0.5, le=1.0, description="粘性の温度指数 ω (HS=0.5)")
    t_ref_k: float = Field(273.0, gt=0, description="基準温度 [K]")


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
    p1: Point | None = None
    p2: Point | None = None
    type: Literal["wall", "symmetry", "inlet", "outlet"] = "wall"
    temperature_k: float = Field(300.0, gt=0)
    pressure_pa: float | None = Field(None, ge=0)
    # inlet の流量指定 [sccm] (標準状態 273.15 K・101325 Pa の cm^3/min)。
    # pressure_pa と排他。1 sccm = 4.478e17 分子/s
    flow_sccm: float | None = Field(None, gt=0)

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
    wall_temperature_k: float = Field(300.0, gt=0, description="未指定エッジ・領域輪郭の壁温 [K]")
    init_pressure_pa: float = Field(..., gt=0, description="初期充填圧 [Pa]")
    init_temperature_k: float = Field(300.0, gt=0)
    n_particles: int = Field(50000, gt=0, description="目標シミュレーション粒子数")
    dt: float | None = Field(None, description="秒。None なら 0.25·h_min/v_mp から自動")
    n_steps: int = Field(2000, gt=0)
    avg_steps: int = Field(500, gt=0, description="最終 N ステップで時間平均")
    seed: int = 0
    # 粒子処理 (walk 探索) のワーカースレッド数 (prompts/65)。粒子ごとの walk は独立な
    # ため、チャンク並列化しても結果は逐次実行とビット単位で一致する。1 = 従来経路
    threads: int = Field(1, ge=1, le=32)
    # 隣接セル拡散による統計ノイズ平滑化の回数 (0=無効、prompts/67)。導出前の生モーメント
    # (Σ個数・Σv・Σv²) に体積重み対称拡散を適用してから n/T/u/p を導出するため、
    # p = n kB T の整合を保ったまま総量 (質量・運動量・エネルギー) を厳密に保存する
    smoothing_passes: int = Field(0, ge=0, le=20)


class Project(BaseModel):
    version: int = 1
    unit: Literal["m", "mm"] = "m"
    # 座標系 (prompts/39, 41)。"xy": 平面2D (従来)。
    # "rz":    軸対称 — x = z (軸方向)、y = r (径方向)。対称軸は y=0 (自然境界)
    # "rz_x0": 軸対称 — x = r (径方向)、y = z (軸方向)。対称軸は x=0 (自然境界)
    coord: Literal["xy", "rz", "rz_x0"] = "xy"
    geometry: Geometry
    mesh: MeshSettings
    solver: SolverSettings = SolverSettings()
    # 一様磁場 [T] (prompts/51)。null または全成分 0 で磁場なし (従来と完全一致)
    b_field: BField | None = None
    # 定常ガス流れの DSMC 設定 (prompts/54)。null なら無効
    dsmc: DsmcSettings | None = None
    particles: ParticleSettings | None = None
    pic: PicSettings | None = None

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
