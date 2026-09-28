"""断面積データモデル (prompts/120)。

LXCat ファイル 1 つ = `LxcatDocument` (DATABASE ごとの `LxcatDatabase` の列 + 警告)、
1 ブロック = `CrossSection`。ソルバーへ渡す混合ガスは `GasMixture` (成分 = `GasComponent`)。
pydantic ではなく dataclass で持ち、JSON 化は `to_dict()`/`from_dict()` で行う
(配列は list、tuple も list にする)。

単位は SI (エネルギー eV・断面積 m²) に統一済み (COLUMNS: の単位換算はパーサーが行う)。
断面積の評価 `CrossSection.sigma()` は boltzpmp と同じ規約 (区分線形、表の最初の点より下と
閾値未満は 0、最後の点より上は最後の値で一定) を既定とし、v1 MCC 互換
(np.interp の端点クランプ) も選べる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import numpy as np

from .masses import KB

Kind = Literal[
    "ELASTIC", "EFFECTIVE", "EXCITATION", "IONIZATION", "ATTACHMENT", "ROTATION",
    "ISOTROPIC", "BACKSCAT",  # 最後の 2 つはイオン-中性 (Phelps 形式)
]

#: 全種別 (Kind の値)
KINDS: tuple[str, ...] = get_args(Kind)
#: 電子衝突の種別 (LXCat/BOLSIG+ のキーワード、boltzpmp が扱う 6 種)
ELECTRON_KINDS: tuple[str, ...] = (
    "ELASTIC", "EFFECTIVE", "EXCITATION", "IONIZATION", "ATTACHMENT", "ROTATION",
)
#: イオン-中性衝突の種別 (タイプ行なし形式)
ION_KINDS: tuple[str, ...] = ("ISOTROPIC", "BACKSCAT")
#: 運動量移行断面積の種別
MOMENTUM_KINDS: tuple[str, ...] = ("ELASTIC", "EFFECTIVE")
#: 非弾性の種別 (EFFECTIVE→ELASTIC 変換で差し引く対象)
INELASTIC_KINDS: tuple[str, ...] = ("EXCITATION", "IONIZATION", "ATTACHMENT", "ROTATION")

ParamValue = float | str | bool


def _as_table(values: Any, label: str) -> np.ndarray:
    arr = np.array(values, dtype=np.float64)  # 常にコピー (呼び出し側の配列と共有しない)
    if arr.ndim != 1:
        raise ValueError(f"{label} は 1 次元配列が必要です (shape={arr.shape})")
    return arr


def _as_state(value: Any, label: str) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        energy, weight = value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} は (エネルギー[eV], 統計重み) の 2 要素が必要です: {value!r}") from exc
    return (float(energy), float(weight))


@dataclass(kw_only=True, eq=False)
class CrossSection:
    """1 つの衝突過程の断面積 (LXCat の 1 ブロック)。

    - projectile: "e" (電子) またはイオン名 ("Ar^+" 等、SPECIES 行の "/" の左)
    - target/product/reversible: 2 行目の反応式 `A`・`A -> B`・`A <-> B` から
      (タイプ行なし形式は SPECIES 行の "/" の右が target)
    - threshold_ev: 閾値 [eV] (ELASTIC/EFFECTIVE/ISOTROPIC/BACKSCAT は 0)
    - mass_ratio: m/M (ELASTIC/EFFECTIVE のみ、0 < m/M < 1。省略時 None)
    - weight_ratio: g_upper/g_lower (EXCITATION の 3 行目 2 番目の数、ROTATION は準位から)
    - lower_state/upper_state: ROTATION の (E [eV], 統計重み g)
    - energy_ev/sigma_m2: 断面積表 (float64、エネルギー非減少で重複は段差、σ ≥ 0、SI)
    - sigma_mt_m2: 表の 3 列目 (運動量移行断面積)。このとき sigma_m2 は積分断面積
    - process/species/comment/updated/columns/meta: ブロックのヘッダ行
      (PROCESS:/SPECIES:/COMMENT:/UPDATED:/COLUMNS:/その他の "KEY: value")
    - param: PARAM.: 行を解釈した辞書 + パーサー/変換の注記
      (keyword_alias・threshold_line・extra_tokens・converted_from 等)
    - database: 所属 DATABASE 名、line: キーワード行 (タイプ行なし形式は SPECIES 行) の
      1 始まり行番号
    """

    kind: Kind
    projectile: str
    target: str
    product: str | None = None
    reversible: bool = False
    threshold_ev: float = 0.0
    mass_ratio: float | None = None
    weight_ratio: float | None = None
    lower_state: tuple[float, float] | None = None
    upper_state: tuple[float, float] | None = None
    energy_ev: np.ndarray
    sigma_m2: np.ndarray
    sigma_mt_m2: np.ndarray | None = None
    process: str = ""
    species: str = ""
    param: dict[str, ParamValue] = field(default_factory=dict)
    comment: str = ""
    updated: str | None = None
    columns: str | None = None
    database: str | None = None
    line: int = 0
    meta: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"未知の断面積種別です: {self.kind!r} (有効: {', '.join(KINDS)})")
        self.energy_ev = _as_table(self.energy_ev, "energy_ev")
        self.sigma_m2 = _as_table(self.sigma_m2, "sigma_m2")
        if self.sigma_mt_m2 is not None:
            self.sigma_mt_m2 = _as_table(self.sigma_mt_m2, "sigma_mt_m2")
        self.threshold_ev = float(self.threshold_ev)
        if self.mass_ratio is not None:
            self.mass_ratio = float(self.mass_ratio)
        if self.weight_ratio is not None:
            self.weight_ratio = float(self.weight_ratio)
        self.lower_state = _as_state(self.lower_state, "lower_state")
        self.upper_state = _as_state(self.upper_state, "upper_state")
        self.reversible = bool(self.reversible)
        self.line = int(self.line)
        self._check_table()

    def _check_table(self) -> None:
        """表の不変条件 (パーサーは行番号付きで先に検査済み。from_dict 等の入口の保険)。"""
        e, s, mt = self.energy_ev, self.sigma_m2, self.sigma_mt_m2
        label = self.label
        if len(e) == 0:
            raise ValueError(f"{label}: 断面積テーブルが空です")
        if len(s) != len(e) or (mt is not None and len(mt) != len(e)):
            raise ValueError(f"{label}: energy_ev と sigma_m2 (sigma_mt_m2) は同じ長さが必要です")
        tables = (e, s) if mt is None else (e, s, mt)
        if not all(np.all(np.isfinite(t)) for t in tables):
            raise ValueError(f"{label}: 断面積テーブルに有限でない値があります")
        if np.any(e < 0.0) or np.any(s < 0.0) or (mt is not None and np.any(mt < 0.0)):
            raise ValueError(f"{label}: エネルギー・断面積は 0 以上が必要です")
        if np.any(np.diff(e) < 0.0):
            raise ValueError(f"{label}: エネルギーは非減少 (昇順、重複は段差) が必要です")

    def __eq__(self, other: object) -> bool:
        """値としての比較 (配列は要素ごと)。

        dataclass 既定の __eq__ はフィールドの tuple 比較で ndarray の真偽値が曖昧になり
        例外になるため自前で持つ (`cs in リスト` 等でも安全)。可変なので hash は持たない。
        """
        if not isinstance(other, CrossSection):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    # ---- 表示・評価 ----------------------------------------------------------------

    @property
    def label(self) -> str:
        """表示用ラベル: PROCESS 行の値、無ければ "KIND target [-> product]"。"""
        if self.process:
            return self.process
        return f"{self.kind} {self.target}" + (f" -> {self.product}" if self.product else "")

    def sigma(
        self,
        eps: Any,
        *,
        below: Literal["zero", "clamp"] = "zero",
        above: Literal["clamp", "zero"] = "clamp",
    ) -> Any:
        """エネルギー eps [eV] での断面積 [m²] (区分線形補間)。

        - below="zero" (既定): 表の最初の点より下と、閾値 > 0 の過程の閾値未満は 0
          (boltzpmp の `CrossSection.sigma` と同じ規約)。
        - below="clamp": np.interp と同じく最初の値で一定、閾値マスクもしない (v1 MCC 互換)。
        - above="clamp" (既定): 最後の値で一定 (boltzpmp・np.interp と同じ)。"zero": 0。

        重複エネルギー (段差) の点ちょうどでは後ろ側の値 (右連続、np.interp と同じ)。
        スカラーを渡せば float、配列なら同じ形の ndarray を返す。
        """
        if below not in ("zero", "clamp"):
            raise ValueError(f"below は 'zero' か 'clamp' です: {below!r}")
        if above not in ("clamp", "zero"):
            raise ValueError(f"above は 'clamp' か 'zero' です: {above!r}")
        x = np.asarray(eps, dtype=np.float64)
        e, s = self.energy_ev, self.sigma_m2
        left = 0.0 if below == "zero" else float(s[0])
        right = float(s[-1]) if above == "clamp" else 0.0
        out = np.interp(x, e, s, left=left, right=right)
        if below == "zero" and self.threshold_ev > 0.0:
            out = np.where(x < self.threshold_ev, 0.0, out)
        if x.ndim == 0:
            return float(out)
        return np.asarray(out, dtype=np.float64)

    # ---- JSON 化 -------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON 化可能な辞書 (配列・tuple は list)。表示用の "label" も含める。"""
        return {
            "kind": self.kind,
            "projectile": self.projectile,
            "target": self.target,
            "product": self.product,
            "reversible": self.reversible,
            "threshold_ev": self.threshold_ev,
            "mass_ratio": self.mass_ratio,
            "weight_ratio": self.weight_ratio,
            "lower_state": None if self.lower_state is None else list(self.lower_state),
            "upper_state": None if self.upper_state is None else list(self.upper_state),
            "energy_ev": self.energy_ev.tolist(),
            "sigma_m2": self.sigma_m2.tolist(),
            "sigma_mt_m2": None if self.sigma_mt_m2 is None else self.sigma_mt_m2.tolist(),
            "process": self.process,
            "species": self.species,
            "param": dict(self.param),
            "comment": self.comment,
            "updated": self.updated,
            "columns": self.columns,
            "database": self.database,
            "line": self.line,
            "meta": dict(self.meta),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrossSection:
        """to_dict() の逆。"label" 等の派生キーは無視する。"""
        return cls(
            kind=data["kind"],
            projectile=data.get("projectile", "e"),
            target=data["target"],
            product=data.get("product"),
            reversible=bool(data.get("reversible", False)),
            threshold_ev=float(data.get("threshold_ev", 0.0)),
            mass_ratio=data.get("mass_ratio"),
            weight_ratio=data.get("weight_ratio"),
            lower_state=data.get("lower_state"),
            upper_state=data.get("upper_state"),
            energy_ev=data["energy_ev"],
            sigma_m2=data["sigma_m2"],
            sigma_mt_m2=data.get("sigma_mt_m2"),
            process=data.get("process", ""),
            species=data.get("species", ""),
            param=dict(data.get("param") or {}),
            comment=data.get("comment", ""),
            updated=data.get("updated"),
            columns=data.get("columns"),
            database=data.get("database"),
            line=int(data.get("line", 0)),
            meta=dict(data.get("meta") or {}),
        )


@dataclass
class LxcatDatabase:
    """DATABASE: 行 1 つ分 (無ければ name=None の 1 つ)。meta はブロック外の "KEY: value"。"""

    name: str | None
    meta: dict[str, str] = field(default_factory=dict)
    cross_sections: list[CrossSection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "meta": dict(self.meta),
            "cross_sections": [cs.to_dict() for cs in self.cross_sections],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LxcatDatabase:
        return cls(
            name=data.get("name"),
            meta=dict(data.get("meta") or {}),
            cross_sections=[CrossSection.from_dict(d) for d in data.get("cross_sections", [])],
        )


@dataclass
class LxcatDocument:
    """LXCat ファイル 1 つのパース結果 (データベースの列 + 回復可能な問題の警告)。"""

    databases: list[LxcatDatabase] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def all_cross_sections(self) -> list[CrossSection]:
        """全ブロックをファイル中の出現順 (行番号順) で返す。"""
        css = [cs for db in self.databases for cs in db.cross_sections]
        return sorted(css, key=lambda cs: cs.line)  # 安定ソート (同じ行番号は DB 順のまま)

    def targets(self, projectile: str | None = "e") -> list[str]:
        """標的名の一覧 (出現順・重複なし)。projectile=None なら全ての入射粒子。"""
        out: list[str] = []
        for cs in self.all_cross_sections():
            if projectile is not None and cs.projectile != projectile:
                continue
            if cs.target not in out:
                out.append(cs.target)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "databases": [db.to_dict() for db in self.databases],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LxcatDocument:
        return cls(
            databases=[LxcatDatabase.from_dict(d) for d in data.get("databases", [])],
            warnings=list(data.get("warnings", [])),
        )


@dataclass
class GasComponent:
    """混合ガスの 1 成分 (標的 1 種)。cross_sections はその標的の電子衝突断面積。"""

    name: str
    fraction: float
    mass_amu: float | None = None
    cross_sections: list[CrossSection] = field(default_factory=list)


@dataclass
class GasMixture:
    """背景ガス (混合) の状態。数密度は n = p/(k_B T)。"""

    components: list[GasComponent]
    pressure_pa: float
    temperature_k: float

    @property
    def number_density(self) -> float:
        """全粒子数密度 [m^-3] = p / (k_B T)。"""
        return self.pressure_pa / (KB * self.temperature_k)

    def validate(self) -> None:
        """整合性を検査し、問題があれば ValueError (全問題を列挙) を送出する。

        - 成分が 1 つ以上・名前が重複しない (boltzpmp も重複を拒否する)
        - 分率が有限・非負で、和が 1 ± 1e-6
        - 各成分に電子の運動量移行断面積 (ELASTIC か EFFECTIVE) がある
        - 圧力・温度が正
        """
        problems: list[str] = []
        if not self.components:
            problems.append("成分がありません")
        names = [c.name for c in self.components]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            problems.append(f"成分名が重複しています: {', '.join(dup)}")
        fractions = [float(c.fraction) for c in self.components]
        bad = [c.name for c in self.components if not (np.isfinite(c.fraction) and c.fraction >= 0.0)]
        if bad:
            problems.append(f"分率は有限の非負値が必要です: {', '.join(bad)}")
        elif self.components and abs(sum(fractions) - 1.0) > 1e-6:
            problems.append(f"分率の和が 1 ではありません (和 = {sum(fractions):.9g})")
        for c in self.components:
            has_mt = any(
                cs.projectile == "e" and cs.kind in MOMENTUM_KINDS for cs in c.cross_sections
            )
            if not has_mt:
                problems.append(
                    f"成分 {c.name} に電子の運動量移行断面積 (ELASTIC/EFFECTIVE) がありません"
                )
        if not (np.isfinite(self.pressure_pa) and self.pressure_pa > 0.0):
            problems.append(f"圧力は正の値が必要です: {self.pressure_pa}")
        if not (np.isfinite(self.temperature_k) and self.temperature_k > 0.0):
            problems.append(f"温度は正の値が必要です: {self.temperature_k}")
        if problems:
            raise ValueError("混合ガスの検証エラー: " + "; ".join(problems))
