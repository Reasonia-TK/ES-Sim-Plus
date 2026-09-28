"""断面積の変換 (prompts/120): EFFECTIVE→ELASTIC・電子断面積セットの整理・混合ガス・
v1 XsProcess・boltzpmp。

## EFFECTIVE→ELASTIC (LXCat/BOLSIG+ の定義)

LXCat の "effective" は全運動量移行断面積 = 弾性の運動量移行断面積 + 全非弾性 (積分)
断面積 (2 項近似の Boltzmann ソルバー向け)。よって同じ標的の非弾性
(EXCITATION/IONIZATION/ATTACHMENT/ROTATION) を差し引いて
σ_el(ε) = max(σ_eff(ε) − Σ σ_inel(ε), 0) とする。σ_inel は boltzpmp と同じ評価規約
(表の最初の点より下と閾値未満は 0、最後の点より上は一定) で、全関数が区分線形なので、
評価点 (effective と全非弾性表のエネルギー・閾値の和集合、effective の範囲内) で左右の極限を
取り、段差 (閾値・表の開始点・表の重複点) は同じエネルギーの 2 点で、0 クリップの折れ点は
交点を追加して表す。結果は評価点間の線形補間で式と厳密に一致する。

なお boltzpmp 0.5.0 は EFFECTIVE を ELASTIC と同じに扱い非弾性を差し引かない
(v1 も同様に elastic として取り込んでいた)。本モジュールの経路 (build_mixture →
to_boltzpmp_mixture) は必ず変換してから渡す。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..schema import XsProcess
from .masses import lookup_mass_amu, mass_amu_from_ratio, mass_ratio_from_amu
from .model import (
    ELECTRON_KINDS,
    INELASTIC_KINDS,
    ION_KINDS,
    MOMENTUM_KINDS,
    CrossSection,
    GasComponent,
    GasMixture,
    LxcatDocument,
)

if TYPE_CHECKING:  # pragma: no cover
    import boltzpmp


def _unique(items: Sequence[str]) -> list[str]:
    out: list[str] = []
    for x in items:
        if x not in out:
            out.append(x)
    return out


def _is_electron(cs: CrossSection) -> bool:
    return cs.projectile == "e" and cs.kind in ELECTRON_KINDS


# ---- EFFECTIVE → ELASTIC ------------------------------------------------------------


def _limits(cs: CrossSection, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sigma(below="zero", above="clamp") の各点での左極限・右極限。

    np.interp は重複点で後ろ側の値 (= 右極限) を返す。左極限は表の点ちょうどでは
    最初の重複点の値 (表の最初の点なら表より下の値 0)、閾値 t > 0 は ε ≤ t で 0。
    """
    ee, ss = cs.energy_ev, cs.sigma_m2
    right = np.interp(e, ee, ss, left=0.0, right=float(ss[-1]))
    lo = np.searchsorted(ee, e, side="left")
    hi = np.searchsorted(ee, e, side="right")
    left = right.copy()
    on_point = lo < hi
    idx = lo[on_point]
    left[on_point] = np.where(idx > 0, ss[idx], 0.0)  # idx < len(ss) (lo < hi ≤ len)
    t = cs.threshold_ev
    if t > 0.0:
        right = np.where(e < t, 0.0, right)
        left = np.where(e <= t, 0.0, left)
    return left, right


def effective_to_elastic(
    effective: CrossSection, inelastic: Sequence[CrossSection]
) -> tuple[CrossSection, list[str]]:
    """EFFECTIVE (全運動量移行) から非弾性を差し引いて ELASTIC (弾性の運動量移行) にする。

    inelastic は差し引く非弾性 (呼び出し側が同じ projectile・target のものを選ぶ。
    resolve_momentum_transfer 参照)。0 でクリップした量が σ_eff の 1% を超える点が
    あれば警告する。戻り値の mass_ratio は effective のもの、param["converted_from"] =
    "EFFECTIVE"。
    """
    if effective.kind != "EFFECTIVE":
        raise ValueError(f"effective_to_elastic: EFFECTIVE ではありません: {effective.kind} ({effective.label})")
    inel = list(inelastic)
    bad = [f"{c.kind} ({c.label})" for c in inel if c.kind not in INELASTIC_KINDS]
    if bad:
        raise ValueError(
            "effective_to_elastic: 非弾性 (EXCITATION/IONIZATION/ATTACHMENT/ROTATION) 以外は"
            f"差し引けません: {', '.join(bad)}"
        )
    warnings: list[str] = []
    where = f"{effective.line} 行目: " if effective.line else ""

    e_eff = effective.energy_ev
    e_lo, e_hi = float(e_eff[0]), float(e_eff[-1])
    parts = [e_eff] + [c.energy_ev for c in inel]
    parts += [np.array([c.threshold_ev]) for c in inel if c.threshold_ev > 0.0]
    grid = np.unique(np.concatenate(parts))
    grid = grid[(grid >= e_lo) & (grid <= e_hi)]

    l_eff, r_eff = _limits(effective, grid)
    l_inel = np.zeros_like(grid)
    r_inel = np.zeros_like(grid)
    for c in inel:
        lft, rgt = _limits(c, grid)
        l_inel += lft
        r_inel += rgt
    l_el = l_eff - l_inel
    r_el = r_eff - r_inel

    # クリップ量の警告 (effective の範囲内の全ての極限値で判定。最初の点の左極限は範囲外)
    raw = np.concatenate([l_el[1:], r_el])
    ref = np.concatenate([l_eff[1:], r_eff])
    at = np.concatenate([grid[1:], grid])
    deficit = np.maximum(-raw, 0.0)
    over = deficit > 0.01 * ref
    if np.any(over):
        ratio = np.where(ref > 0.0, deficit / np.where(ref > 0.0, ref, 1.0), np.inf)
        k = int(np.argmax(np.where(over, ratio, -1.0)))
        how = "σ_eff = 0 の点で" if not math.isfinite(ratio[k]) else f"最大で σ_eff の {ratio[k]:.1%} を"
        warnings.append(
            f"{where}{effective.label} の EFFECTIVE→ELASTIC 変換で σ_eff − Σσ_inel < 0 となり、"
            f"{how} 0 にクリップしました ({at[k]:g} eV 付近)。非弾性断面積の組が EFFECTIVE と"
            "整合していない可能性があります"
        )

    # 表の組み立て: 段差は同じエネルギーの 2 点 (左極限 → 右極限)、0 との交点は折れ点として追加
    energies: list[float] = []
    values: list[float] = []
    for k, g in enumerate(grid):
        if k > 0:
            a, b = r_el[k - 1], l_el[k]  # 区間 (grid[k-1], grid[k]) の両端での値 (区間内は線形)
            if (a > 0.0 > b) or (a < 0.0 < b):
                x = grid[k - 1] + (g - grid[k - 1]) * (a / (a - b))
                if grid[k - 1] < x < g:
                    energies.append(float(x))
                    values.append(0.0)
            lv, rv = max(float(l_el[k]), 0.0), max(float(r_el[k]), 0.0)
            if lv != rv:
                energies.append(float(g))
                values.append(lv)
            energies.append(float(g))
            values.append(rv)
        else:
            energies.append(float(g))
            values.append(max(float(r_el[k]), 0.0))

    if effective.sigma_mt_m2 is not None:
        warnings.append(
            f"{where}{effective.label} の EFFECTIVE の 3 列目 (運動量移行断面積) は変換後の "
            "ELASTIC では使いません (2 列目から変換します)"
        )
    param = dict(effective.param)
    param["converted_from"] = "EFFECTIVE"
    elastic = CrossSection(
        kind="ELASTIC",
        projectile=effective.projectile,
        target=effective.target,
        product=effective.product,
        reversible=False,
        threshold_ev=0.0,
        mass_ratio=effective.mass_ratio,
        energy_ev=np.asarray(energies, dtype=np.float64),
        sigma_m2=np.asarray(values, dtype=np.float64),
        sigma_mt_m2=None,
        process=effective.process,
        species=effective.species,
        param=param,
        comment=effective.comment,
        updated=effective.updated,
        columns=effective.columns,
        database=effective.database,
        line=effective.line,
        meta=dict(effective.meta),
    )
    return elastic, warnings


def resolve_momentum_transfer(
    cross_sections: Sequence[CrossSection],
) -> tuple[list[CrossSection], list[str]]:
    """電子の標的ごとに運動量移行断面積を 1 系統に整理する (出現順は保つ)。

    - ELASTIC があれば EFFECTIVE を除外 (二重計上の防止、警告)
    - EFFECTIVE だけなら同じ標的の非弾性を差し引いて ELASTIC に変換 (警告)
    - どちらも無ければ警告
    電子以外 (イオン-中性等) の断面積はそのまま通す。
    """
    css = list(cross_sections)
    warnings: list[str] = []
    electron = [c for c in css if _is_electron(c)]
    replacement: dict[int, CrossSection | None] = {}
    for target in _unique([c.target for c in electron]):
        group = [c for c in electron if c.target == target]
        elastic = [c for c in group if c.kind == "ELASTIC"]
        effective = [c for c in group if c.kind == "EFFECTIVE"]
        if elastic:
            if len(elastic) > 1:
                lines = ", ".join(str(c.line) for c in elastic)
                warnings.append(
                    f"標的 {target} に ELASTIC が {len(elastic)} 件あります (行 {lines})。"
                    "運動量移行断面積が二重計上されている可能性があります"
                )
            for c in effective:
                replacement[id(c)] = None
                warnings.append(
                    f"{c.line} 行目: 標的 {target} は ELASTIC を持つため EFFECTIVE ({c.label}) を"
                    "除外しました (運動量移行断面積の二重計上を防ぐため)"
                )
        elif effective:
            if len(effective) > 1:
                lines = ", ".join(str(c.line) for c in effective)
                warnings.append(
                    f"標的 {target} に EFFECTIVE が {len(effective)} 件あります (行 {lines})。"
                    "運動量移行断面積が二重計上されている可能性があります"
                )
            inelastic = [c for c in group if c.kind in INELASTIC_KINDS]
            for c in effective:
                converted, w = effective_to_elastic(c, inelastic)
                replacement[id(c)] = converted
                warnings.append(
                    f"{c.line} 行目: 標的 {target} の EFFECTIVE ({c.label}) を ELASTIC に変換しました "
                    f"(σ_el = max(σ_eff − Σσ_inel, 0)、差し引いた非弾性 {len(inelastic)} 件)"
                )
                warnings.extend(w)
        else:
            warnings.append(
                f"標的 {target} に電子の運動量移行断面積 (ELASTIC/EFFECTIVE) がありません"
            )
    out: list[CrossSection] = []
    for c in css:
        if id(c) in replacement:
            new = replacement[id(c)]
            if new is not None:
                out.append(new)
        else:
            out.append(c)
    return out, warnings


# ---- 混合ガス -----------------------------------------------------------------------


def component_mass_amu(
    name: str,
    cross_sections: Sequence[CrossSection],
    masses: Mapping[str, float] | None = None,
) -> float | None:
    """成分の分子量 [amu]: masses 引数 → 質量表 → 運動量移行断面積の m/M (M = m_e/(m/M))。

    LXCat の m/M は丸められていることが多い (Ar の 1.36e-5 は 40.34 amu で標準原子量 39.948
    と ~1% ずれる) ため、名前が質量表にある標的は表の値を優先する (prompts/120 の当初仕様
    「m/M → 表」から変更)。m/M 自体は弾性のエネルギー損失にそのまま使われる。
    """
    if masses is not None and name in masses:
        return float(masses[name])
    table = lookup_mass_amu(name)
    if table is not None:
        return table
    for c in cross_sections:
        if c.kind in MOMENTUM_KINDS and c.mass_ratio is not None:
            return mass_amu_from_ratio(c.mass_ratio)
    return None


def build_mixture(
    doc: LxcatDocument,
    fractions: Mapping[str, float],
    pressure_pa: float,
    temperature_k: float,
    masses: Mapping[str, float] | None = None,
) -> tuple[GasMixture, list[str]]:
    """文書の電子断面積を標的ごとに GasComponent にまとめた混合ガスを作る。

    fractions は {標的名: モル分率}。文書に無い標的を指定すると ValueError、指定されなかった
    標的は警告して使わない。resolve_momentum_transfer を適用してから成分に分け、成分の質量は
    component_mass_amu の順で決める。最後に GasMixture.validate() (失敗は ValueError)。
    """
    if not fractions:
        raise ValueError("fractions が空です ({標的名: モル分率} を指定してください)")
    warnings: list[str] = []
    electron = [c for c in doc.all_cross_sections() if _is_electron(c)]
    doc_targets = _unique([c.target for c in electron])
    unknown = [t for t in fractions if t not in doc_targets]
    if unknown:
        raise ValueError(
            f"fractions の標的 {', '.join(unknown)} の電子衝突断面積がありません "
            f"(文書内の標的: {', '.join(doc_targets) or 'なし'})"
        )
    unused = [t for t in doc_targets if t not in fractions]
    if unused:
        warnings.append(f"fractions に無い標的の断面積は使いません: {', '.join(unused)}")
    resolved, w = resolve_momentum_transfer([c for c in electron if c.target in fractions])
    warnings.extend(w)

    components: list[GasComponent] = []
    for name in doc_targets:
        if name not in fractions:
            continue
        css = [c for c in resolved if c.target == name]
        mass = component_mass_amu(name, css, masses)
        if mass is None:
            warnings.append(
                f"標的 {name} の分子量が決まりません (masses 引数・m/M・質量表のいずれにもありません)"
            )
        components.append(
            GasComponent(name=name, fraction=float(fractions[name]), mass_amu=mass, cross_sections=css)
        )
    mixture = GasMixture(
        components=components, pressure_pa=float(pressure_pa), temperature_k=float(temperature_k)
    )
    mixture.validate()
    return mixture, warnings


# ---- v1 XsProcess -------------------------------------------------------------------


def _v1_process(
    cs: CrossSection,
    kind: str,
    threshold_ev: float,
    mass_ratio: float,
    sigma_m2: np.ndarray | None = None,
) -> XsProcess:
    return XsProcess(
        kind=kind,  # type: ignore[arg-type]
        label=cs.label,
        threshold_ev=threshold_ev,
        mass_ratio=mass_ratio,
        energy_ev=cs.energy_ev.tolist(),
        sigma_m2=(cs.sigma_m2 if sigma_m2 is None else sigma_m2).tolist(),
    )


def _filter_warning(species: str, cs: CrossSection) -> str:
    return (
        f"{cs.line} 行目: species='{species}' では {cs.kind} (入射粒子 '{cs.projectile}') は"
        f"使えないためスキップしました: {cs.label}"
    )


def to_v1_processes(
    doc: LxcatDocument, species: Literal["electron", "ion"]
) -> tuple[list[XsProcess], list[str]]:
    """v1 の XsProcess 一覧に変換する (v1 MCC が扱える種別だけ、出現順)。

    - electron: resolve_momentum_transfer 後、ELASTIC→elastic (m/M が無ければ質量表、
      それも無ければ 0 で警告)・EXCITATION/ROTATION→excitation (逆過程は v1 非対応で警告)・
      IONIZATION→ionization、ATTACHMENT はスキップ (警告)。電子の標的が 2 種以上なら
      v1 が単一ガスとして合算する旨を警告。
    - ion: ISOTROPIC→isotropic・BACKSCAT→backscat (閾値 0・m/M 0)。
    それ以外 (種別フィルタに合わないブロック) は 1 ブロックにつき 1 警告でスキップ。
    警告の先頭には文書のパース警告を含める。

    v1 の XsProcess には 3 列目 (sigma_mt) の欄が無い。非弾性の 3 列目は捨てる (衝突頻度は
    2 列目の積分断面積で正しい)。ELASTIC に 3 列目があるときは、等方散乱の v1 MCC で運動量
    緩和が正しくなる運動量移行断面積 (3 列目 = LXCat の 2 列の ELASTIC と同じ量) を v1 の
    elastic の σ とし、2 列目 (積分断面積) を捨てる (警告)。
    """
    if species not in ("electron", "ion"):
        raise ValueError(f"species は 'electron' か 'ion' を指定してください: '{species}'")
    warnings = list(doc.warnings)
    css = doc.all_cross_sections()
    out: list[XsProcess] = []
    if species == "ion":
        for c in css:
            if c.kind in ION_KINDS and c.projectile != "e":
                out.append(_v1_process(c, c.kind.lower(), 0.0, 0.0))
            else:
                warnings.append(_filter_warning(species, c))
        return out, warnings

    resolved, w = resolve_momentum_transfer(css)
    warnings.extend(w)
    targets = _unique([c.target for c in resolved if _is_electron(c)])
    if len(targets) >= 2:
        warnings.append(
            f"電子の標的が {len(targets)} 種 ({', '.join(targets)}) あります。v1 は単一ガスとして"
            "全過程を合算します (標的ごとの分率・密度は区別されません)"
        )
    for c in resolved:
        if not _is_electron(c):
            warnings.append(_filter_warning(species, c))
        elif c.kind == "ATTACHMENT":
            warnings.append(f"{c.line} 行目: ATTACHMENT は v1 MCC 未対応のためスキップしました: {c.label}")
        elif c.kind == "ELASTIC":
            mass_ratio = c.mass_ratio
            if mass_ratio is None:
                mass = lookup_mass_amu(c.target)
                if mass is not None:
                    mass_ratio = mass_ratio_from_amu(mass)
                else:
                    mass_ratio = 0.0
                    warnings.append(
                        f"{c.line} 行目: ELASTIC ({c.label}) の m/M が無く標的 {c.target} の"
                        "分子量も不明なため m/M = 0 (弾性衝突のエネルギー損失なし) とします"
                    )
            sigma = None
            if c.sigma_mt_m2 is not None:
                # 3 列の表では 2 列目が積分断面積・3 列目が運動量移行断面積。v1 MCC は弾性散乱を
                # 等方とするので、運動量緩和が正しくなるのは運動量移行断面積 (LXCat の 2 列の
                # ELASTIC と同じ量 = v1 が従来受け取っていた量)。積分断面積のほうを捨てる
                sigma = c.sigma_mt_m2
                warnings.append(
                    f"{c.line} 行目: ELASTIC ({c.label}) は 3 列目 (運動量移行断面積) を v1 の elastic "
                    "として使います (v1 MCC は等方散乱のため。2 列目の積分断面積は使いません)"
                )
            out.append(_v1_process(c, "elastic", 0.0, mass_ratio, sigma))
        elif c.kind in ("EXCITATION", "ROTATION"):
            if c.kind == "ROTATION":
                warnings.append(
                    f"{c.line} 行目: ROTATION ({c.label}) は excitation として取り込みました "
                    "(超弾性 (逆過程) は v1 MCC 非対応)"
                )
            elif c.reversible:
                warnings.append(
                    f"{c.line} 行目: EXCITATION ({c.label}) の逆過程 (<->) は v1 MCC 非対応のため"
                    "無視します (励起のみ取り込み)"
                )
            out.append(_v1_process(c, "excitation", c.threshold_ev, 0.0))
        elif c.kind == "IONIZATION":
            out.append(_v1_process(c, "ionization", c.threshold_ev, 0.0))
        else:  # EFFECTIVE は resolve_momentum_transfer で必ず変換・除外済み
            raise AssertionError(f"未処理の種別: {c.kind}")
    return out, warnings


# ---- boltzpmp -----------------------------------------------------------------------


def _reaction_string(cs: CrossSection, reversible: bool) -> str:
    """boltzpmp の species (反応式) を組み立てる (boltzpmp はここから target/product/<-> を得る)。"""
    if cs.product is None:
        return cs.target
    return f"{cs.target} {'<->' if reversible else '->'} {cs.product}"


def boltzpmp_name(cs: CrossSection) -> str:
    """bp.parse_lxcat と同じ過程名 (PROCESS 行の値、無ければ "反応式 種別小文字")。

    boltzpmp の速度係数の辞書キー "気体名:過程名" になる。
    """
    if cs.process:
        return cs.process
    return f"{_reaction_string(cs, cs.reversible)} {cs.kind.lower()}"


def to_boltzpmp_cross_section(
    cs: CrossSection, *, reversible: bool | None = None, name: str | None = None
) -> boltzpmp.CrossSection:
    """CrossSection → bp.CrossSection (boltzpmp の crosssections.py の引数の意味に合わせる)。

    - species: 反応式 "A"・"A -> B"・"A <-> B" (boltzpmp は target/product/逆過程をここから得る)
    - name: 過程名 (既定は boltzpmp_name)、threshold: 閾値 [eV] (ROTATION は準位差で上書きされる)
    - mass_ratio: ELASTIC/EFFECTIVE のみ (None なら bp.Gas.mass_amu から)
    - data: (E [eV], σ [m²]) の (n, 2) 表、mt_data: 3 列目があれば (E, σ_mt)
      (このとき data は積分断面積として衝突頻度に、mt は角度分布の異方性に使われる)
    - weight_ratio: EXCITATION の g_up/g_low (ROTATION は準位の重みを boltzpmp が直接使うので
      bp.parse_lxcat と同じく None)、lower_state/upper_state: ROTATION の (E, g)
    reversible を与えると反応式の `<->` をそれで置き換える。
    """
    import boltzpmp as bp

    if cs.kind not in ELECTRON_KINDS:
        raise ValueError(f"{cs.label}: {cs.kind} は boltzpmp の電子衝突断面積に変換できません")
    rev = cs.reversible if reversible is None else bool(reversible)
    data = np.column_stack([cs.energy_ev, cs.sigma_m2])
    mt = None if cs.sigma_mt_m2 is None else np.column_stack([cs.energy_ev, cs.sigma_mt_m2])
    is_rotation = cs.kind == "ROTATION"
    return bp.CrossSection(
        kind=cs.kind,
        species=_reaction_string(cs, rev),
        name=name if name is not None else boltzpmp_name(cs),
        threshold=float(cs.threshold_ev),
        mass_ratio=cs.mass_ratio if cs.kind in MOMENTUM_KINDS else None,
        data=data,
        comment=cs.comment,
        weight_ratio=cs.weight_ratio if cs.kind == "EXCITATION" else None,
        lower_state=cs.lower_state if is_rotation else None,
        upper_state=cs.upper_state if is_rotation else None,
        mt_data=mt,
    )


def to_boltzpmp_mixture(
    mixture: GasMixture, *, warnings_out: list[str] | None = None
) -> boltzpmp.Mixture:
    """GasMixture → bp.Mixture (各成分を bp.Gas(name, fraction, cross_sections, mass_amu) に)。

    - 検証 (GasMixture.validate) し、成分に EFFECTIVE が残っていれば
      resolve_momentum_transfer で変換してから渡す (boltzpmp は EFFECTIVE から非弾性を
      差し引かないため)。
    - boltzpmp は `<->` の生成物が混合ガスの成分に無いとソルバー構築時にエラーにするので、
      その場合は逆過程の指定を外す (boltzpmp は下準位・上準位の二準位系の占有で超弾性を扱う)。
    - 電子衝突でない断面積は渡さない。同じ成分内で過程名が重複したら " #2" 等を付ける
      (速度係数の辞書キー "気体名:過程名" の衝突を防ぐ)。
    警告は warnings_out (与えた場合) に追記する。
    """
    import boltzpmp as bp

    mixture.validate()
    warnings: list[str] = []
    names = {c.name for c in mixture.components}
    gases = []
    for comp in mixture.components:
        css = [c for c in comp.cross_sections if _is_electron(c)]
        for c in comp.cross_sections:
            if not _is_electron(c):
                warnings.append(
                    f"成分 {comp.name}: 電子衝突でない {c.kind} ({c.label}) は boltzpmp に渡しません"
                )
        if any(c.kind == "EFFECTIVE" for c in css):
            css, w = resolve_momentum_transfer(css)
            warnings.extend(f"成分 {comp.name}: {m}" for m in w)
        used: dict[str, int] = {}
        bcs = []
        for c in css:
            rev = c.reversible
            if c.kind == "EXCITATION" and rev and c.product not in names:
                rev = False
                warnings.append(
                    f"成分 {comp.name}: <-> 過程 ({c.label}) の生成物 '{c.product}' が混合ガスの成分に"
                    "無いため逆過程の指定を外しました (boltzpmp は二準位系の占有で超弾性を扱います)"
                )
            if c.kind in MOMENTUM_KINDS and c.mass_ratio is None and comp.mass_amu is None:
                raise ValueError(
                    f"成分 {comp.name}: {c.kind} ({c.label}) の質量比 m/M が無く、成分の分子量 "
                    "mass_amu もありません (build_mixture の masses で与えてください)"
                )
            name = boltzpmp_name(c)
            if c.param.get("converted_from") == "EFFECTIVE":
                name += " [EFFECTIVE→ELASTIC]"
            used[name] = used.get(name, 0) + 1
            if used[name] > 1:
                name = f"{name} #{used[name]}"
            bcs.append(to_boltzpmp_cross_section(c, reversible=rev, name=name))
        gases.append(
            bp.Gas(
                name=comp.name,
                fraction=float(comp.fraction),
                cross_sections=bcs,
                mass_amu=None if comp.mass_amu is None else float(comp.mass_amu),
            )
        )
    if warnings_out is not None:
        warnings_out.extend(warnings)
    return bp.Mixture(gases, p_Pa=float(mixture.pressure_pa), T_K=float(mixture.temperature_k))
