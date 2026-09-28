"""boltzpmp (https://github.com/Reasonia-TK/boltzpmp) 連携 — LMEA 流体係数生成 (prompts/117, 119)。

boltzpmp は旧 boltzpm (Python 版、リポジトリ消滅) の Rust 移植・後継で、本モジュールが
使う API (CrossSection/Gas/Mixture/PMSolver/solve_dc/SwarmResult) は互換。旧版との
挙動差 (prompts/119 で対応):

- solve_dc の既定が陰的定常解法 (method="implicit") になり、tol はソース反復の残差
  (既定 1e-8)。旧 boltzpm の陽的時間積分向けに絞っていた tol=3e-3/max_steps=1e5 は
  渡さず、boltzpmp の既定を使う (陰解法なので低 E/N でも高速に収束する)。
- 超弾性衝突 (superelastic) とガス熱運動による電子加熱 (gas_heating) が既定で有効。
  BOLSIG+ と同じ物理で、opts から切替可能にした (旧 boltzpm 相当は両方 False)。
- rate_coefficients に "gas:process (superelastic)" キーが増える (本モジュールは
  プロセスごとのキーを明示して引くので影響なし)。

以下の本文中の「boltzpm」は同 API を持つ boltzpmp を指す。

## 背景・LMEA (Local Mean Energy Approximation) とは

fluid_coeffs.py の Maxwell 平均 (Phase A) は電子集団を局所 Maxwell 分布 (温度 Te)
と仮定して係数を Te の関数にテーブル化するが、実際の EEDF (電子エネルギー分布関数)
は Maxwell 分布からずれる (特に電離・励起で高エネルギー側が削られる非 Maxwell 形)。
boltzpm は伝播演算子法で「E/N を固定した定常状態の実際の EEDF」を直接解ける
ため、BOLSIG+ など標準の流体クロージャで広く使われる LMEA を採用する:

  1. E/N を掃引して各点で定常解 (SwarmResult) を求める。
  2. 各点の平均エネルギー ε̄=⟨ε⟩ をキーに、μ_e·N・k_ion・k_exc 等をテーブル化する
     (E/N ではなく ε̄ をキーにするのが LMEA の核心 — 流体ソルバーは局所の電子
     エネルギー密度 w=(3/2)n_e·Te から Te を持ち、ε̄=(3/2)Te でこのテーブルを
     引くことで「局所的に、この平均エネルギーの電子集団が定常状態で持つはずの
     ε̄-E/N 対応にいた」とみなす近似が成り立つ)。
  3. fluid1d.py/fluid2d.py は Maxwell 経路 (Te→テーブル) と全く同じ枠組みで
     ε̄=(3/2)Te→テーブル参照に差し替えるだけで済む (_te_and_coeffs 参照)。

## boltzpm API 早見表 (README/SPEC/ソースで確認済み)

- `bp.CrossSection(kind, species, name, threshold, mass_ratio, data)`:
  kind は "ELASTIC"/"EFFECTIVE"/"EXCITATION"/"IONIZATION"/"ATTACHMENT"。
  mass_ratio は None なら Gas.mass_amu から自動計算 (elastic/effective のみ使う)。
  data は (n,2) の [energy_eV, sigma_m2] 昇順テーブル。
- `bp.Gas(name, fraction, cross_sections, mass_amu)`, `bp.Mixture(gases, p_Pa, T_K)`
  (N を直接与える経路もあるが本モジュールは p_Pa/T_K を使う)。
- `bp.PMSolver(mixture, eps_max_eV, d_eps_eV, n_theta).solve_dc(EN_Td)` →
  `SwarmResult`: `mean_energy` [eV] = ⟨ε⟩、`drift_velocity` [m/s] = W、
  `rate_coefficients` は `"gas名:process名" -> k [m^3/s]` の dict (gas の
  mole fraction は掛かっていない、boltzpm/output.py の compute_swarm 参照)、
  `converged`(bool)。
- `eedf` (= boltzpm の `SwarmResult.eedf`) の規格化は **∫f(ε)dε = 1**
  (boltzpm/output.py: `eedf = n_i / d_eps_eV`、`n_i` は正規化された cell 占有率の
  θ 和で `sum(n_i) = 1` なので `sum(eedf*d_eps_eV) = 1`。README の表記
  `∫f dε = 1` の通りで、EEPF (`f/√ε`) の規格化 `∫F√ε dε=1` と混同しないよう
  ソースで直接確認した — 推測ではない、test_boltz.py で数値検証もする)。

## 依存の扱い

boltzpm は `backend/pyproject.toml` の依存に加えたが (git 依存、release.yml の
`pip install -e .` で解決される)、モジュールレベルで import を try/except して
おく — サンドボックス/CI には常にインストールされる前提だが、万一 boltzpm が
見つからない環境でも `import es_sim.boltz` 自体は落とさず、実際に使おうとした
呼び出し (xsprocess_to_mixture/run_boltz_sweep) でのみ分かりやすいエラーにする
(既存テストが boltzpm 無し環境に落ちないようにする、というプロンプトの指示)。
"""

from __future__ import annotations

import hashlib
import json as _json
import time
from dataclasses import dataclass

import numpy as np

from .fluid_coeffs import interp_loglog
from .particles import ME, QE
from .schema import XsProcess

try:
    import boltzpmp as bp

    _BOLTZPM_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover - このリポジトリの CI/サンドボックスは常にインストール済み
    bp = None  # type: ignore[assignment]
    _BOLTZPM_IMPORT_ERROR = _exc


def boltzpm_available() -> bool:
    """boltzpmp が import 可能かどうか (server.py の /ws/boltz が使う)。"""
    return bp is not None


def _require_boltzpm() -> None:
    if bp is None:
        raise RuntimeError(
            "boltzpmp がインストールされていません "
            f"(pip install boltzpmp。詳細: {_BOLTZPM_IMPORT_ERROR})"
        )


# ---- 断面積の変換 (XsProcess -> bp.Mixture) -------------------------------------

# XsProcess.kind ("elastic"/"excitation"/"ionization"/"isotropic"/"backscat") のうち
# 電子衝突として boltzpm に渡せるのは最初の3つだけ (isotropic/backscat はイオン用)。
# XsProcess スキーマに attachment 相当の kind は存在しないが、将来の拡張やユーザー
# データの取り違えで紛れ込んだ場合に無言で無視せず ValueError にする (プロンプト指示)
_KIND_MAP = {"elastic": "ELASTIC", "excitation": "EXCITATION", "ionization": "IONIZATION"}


def xsprocess_to_mixture(
    processes: list[XsProcess], mass_amu: float, p_pa: float, t_k: float
):
    """XsProcess のリストから boltzpm の Mixture (単一 Gas、fraction=1.0) を作る。

    既存の electron_processes (MccSettings/Fluid1dSettings/Fluid2dSettings と同じ形式、
    fluid_coeffs.py が Maxwell 平均で使っているのと同じ生データ) をそのまま流用する。
    全プロセスを1つの疑似 "gas" にまとめる (fluid_coeffs.build_fluid_reactions が
    種別を区別せず kind だけで集約するのと同じ扱い — 個々の元ガス種を区別する
    情報が electron_processes には元々無いため)。
    """
    _require_boltzpm()
    cross_sections = []
    for i, p in enumerate(processes):
        if p.kind not in _KIND_MAP:
            raise ValueError(
                f"xsprocess_to_mixture: kind={p.kind!r} は boltzpm 変換に未対応です "
                "(elastic/excitation/ionization のみ対応。attachment 相当のプロセスは未使用です)"
            )
        kind = _KIND_MAP[p.kind]
        # elastic の mass_ratio (m/M) が明示的に入っていればそれを使い (通常はこちら —
        # eduPIC プリセット等は既に m/M を計算済み)、0 (未設定の意味で使われる既定値)
        # なら None にして bp.Gas 側の mass_amu フォールバックに委ねる
        mass_ratio = float(p.mass_ratio) if (kind == "ELASTIC" and p.mass_ratio > 0.0) else None
        data = np.column_stack(
            [
                np.asarray(p.energy_ev, dtype=np.float64),
                np.asarray(p.sigma_m2, dtype=np.float64),
            ]
        )
        # name は rate_coefficients の辞書キー ("gas:name") になるため、同名プロセスが
        # 複数あってもキー衝突しないよう常にインデックスを付す
        label = p.label if p.label else p.kind
        cross_sections.append(
            bp.CrossSection(
                kind=kind,
                species="gas",
                name=f"{label} #{i}",
                threshold=float(p.threshold_ev),
                mass_ratio=mass_ratio,
                data=data,
            )
        )
    gas = bp.Gas(name="gas", fraction=1.0, cross_sections=cross_sections, mass_amu=float(mass_amu))
    return bp.Mixture([gas], p_Pa=float(p_pa), T_K=float(t_k))


# ---- E/N 掃引 → LMEA テーブル ---------------------------------------------------

#: run_boltz_sweep の既定オプション (prompts/117)
DEFAULT_BOLTZ_OPTS: dict = {
    "en_min_td": 0.5,
    "en_max_td": 1000.0,
    "n_points": 32,
    "eps_max_ev": None,   # None なら「電離/励起の最大閾値×8」と 40 eV の大きい方
    "d_eps_ev": 0.25,
    "n_theta": 16,
    # boltzpmp の PMSolver 既定 (BOLSIG+ と同じ物理)。旧 boltzpm 相当は両方 False
    "superelastic": True,
    "gas_heating": True,
    # 陰解法の反復上限 (boltzpmp 既定は 2e6)。収束点は通常 15〜40 反復で済むが、粗すぎる
    # メッシュで収束しない点 (例: d_eps 0.5 eV・n_theta 8 の 1 Td) は既定だと ~170 s
    # 空回りするため、5000 反復 (~0.4 s) で見切ってテーブルから除外する (prompts/119)
    "max_steps": 5000,
}


def _resolve_opts(opts: dict | None) -> dict:
    merged = dict(DEFAULT_BOLTZ_OPTS)
    if opts:
        for k in DEFAULT_BOLTZ_OPTS:
            if k in opts and opts[k] is not None:
                merged[k] = opts[k]
    return merged


def _hash_processes(processes: list[XsProcess]) -> str:
    """processes の sha256(JSON) — フロントが「断面積変更後は再生成」を判定するためのキー。"""
    payload = [p.model_dump() for p in processes]
    blob = _json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def run_boltz_sweep(
    processes: list[XsProcess],
    mass_amu: float,
    p_pa: float,
    t_k: float,
    opts: dict | None = None,
    progress_cb=None,
    should_stop=None,
) -> dict:
    """E/N を掃引し ε̄ をキーにした LMEA テーブル (BoltzTable の dict) を作る。

    各 E/N で `solve_dc` の定常解を求め、ε̄=⟨ε⟩ をキーに以下を記録する:

    - `mobility_n` = μ_e·N [1/(m·V·s)]。μN = W/E ÷ N の同値式で、
      E/N [Td]·1e-21 = E/N [V·m^2] (Td の定義そのもの) なので
      W/(EN_Td·1e-21) = W/((E/N)·1e-21) = W/(E/N_SI) = (W/E)/1 = μ・(E/N を
      1/N 倍せず「E/N で割る」ことで N が自動的に約分され) そのまま μ・N になる
      (W/E = μ、E=(E/N)_SI・N なので W/((E/N)_SI・N) = μ/N ではなく
      W/(E/N)_SI = μ・N — N で割らずに「E を (E/N)・N の形で書いて W/E を
      計算する」のではなく「W を (E/N) で直接割る」ことで N が最初から現れず
      μ・N がそのまま得られる、という単位検算)。
    - `k_ion`/`k_exc`: プロセス別 rate_coefficients を kind (ionization/excitation)
      で集約した和。
    - `e_ion_ev`/`e_exc_ev`: 電離/励起が複数プロセスある場合の「この E/N 点での」
      レート係数加重平均閾値 (fluid_coeffs._weighted_threshold は Te 積分全体の
      加重平均だが、ここは掃引の点ごとに加重する — LMEA はテーブルの各行が
      独立した定常状態なので、点ごとの実際の内訳で加重する方が忠実)。
    - `eedf`: 表示用に全収束点の EEDF (`eedf_eps_ev` 共通グリッド) を保存する。

    収束しなかった点 (`SwarmResult.converged is False`) は `warnings` に記録して
    テーブルから除外する (LMEA は「この E/N では定常状態が求まった」ことが前提
    のため、収束していない生値を混ぜるとテーブル自体が意味を持たなくなる)。
    """
    _require_boltzpm()
    resolved = _resolve_opts(opts)

    mixture = xsprocess_to_mixture(processes, mass_amu, p_pa, t_k)
    # xsprocess_to_mixture は単一 Gas に processes と同じ順序で CrossSection を積む
    # ため、mixture.processes() は processes と同じ長さ・同じ順序になる (kind ごとの
    # 集約・閾値加重をキー文字列のパースではなく、この対応そのもので行う)
    gas_cs_pairs = mixture.processes()
    if len(gas_cs_pairs) != len(processes):
        raise AssertionError("xsprocess_to_mixture の CrossSection 数が processes と一致しません")

    thresholds = [p.threshold_ev for p in processes if p.kind in ("excitation", "ionization")]
    default_eps_max = max((max(thresholds) * 8.0) if thresholds else 0.0, 40.0)
    eps_max_ev = float(resolved["eps_max_ev"]) if resolved["eps_max_ev"] is not None else default_eps_max
    # 実際に使った値をメタ情報として残す (None のままだと再現できないため)
    resolved = dict(resolved, eps_max_ev=eps_max_ev)

    d_eps_ev = float(resolved["d_eps_ev"])
    n_theta = int(resolved["n_theta"])
    n_points = int(resolved["n_points"])
    en_grid = np.geomspace(float(resolved["en_min_td"]), float(resolved["en_max_td"]), n_points)

    solver = bp.PMSolver(
        mixture,
        eps_max_eV=eps_max_ev,
        d_eps_eV=d_eps_ev,
        n_theta=n_theta,
        superelastic=bool(resolved["superelastic"]),
        gas_heating=bool(resolved["gas_heating"]),
    )

    ion_keys: list[str] = []
    exc_keys: list[str] = []
    ion_thresh: dict[str, float] = {}
    exc_thresh: dict[str, float] = {}
    for p, (gas, cs) in zip(processes, gas_cs_pairs):
        key = f"{gas.name}:{cs.name}"
        if p.kind == "ionization":
            ion_keys.append(key)
            ion_thresh[key] = float(p.threshold_ev)
        elif p.kind == "excitation":
            exc_keys.append(key)
            exc_thresh[key] = float(p.threshold_ev)

    rows: list[dict] = []
    warnings_out: list[str] = []
    t0 = time.perf_counter()
    for i, en in enumerate(en_grid, start=1):
        if should_stop is not None and should_stop():
            warnings_out.append(f"掃引が中断されました ({i - 1}/{n_points} 点で停止)")
            break

        # boltzpmp の既定 (陰的定常解法 + Anderson 加速、tol=1e-8 のソース反復残差) を
        # そのまま使う。旧 boltzpm の陽的時間積分では低 E/N で収束が遅く tol=3e-3 まで
        # 緩めていたが、陰解法では 1 点あたり ~0.1 s で厳密に収束する (prompts/119)
        result = solver.solve_dc(EN_Td=float(en), max_steps=int(resolved["max_steps"]))

        if not result.converged:
            warnings_out.append(f"E/N={float(en):.4g} Td: 収束しませんでした (この点はテーブルから除外)")
            if progress_cb is not None:
                progress_cb(i, n_points, float(en), time.perf_counter() - t0)
            continue

        en_v_m2 = float(en) * 1.0e-21  # 1 Td = 1e-21 V*m^2 (boltzpm.constants.TOWNSEND と同じ定義)
        mobility_n = float(result.drift_velocity) / en_v_m2

        k_ion = float(sum(result.rate_coefficients[k] for k in ion_keys))
        k_exc = float(sum(result.rate_coefficients[k] for k in exc_keys)) if exc_keys else 0.0

        if ion_keys:
            w_ion = np.array([result.rate_coefficients[k] for k in ion_keys])
            th_ion = np.array([ion_thresh[k] for k in ion_keys])
            tot = float(w_ion.sum())
            e_ion_ev = float(np.sum(w_ion * th_ion) / tot) if tot > 0.0 else float(np.mean(th_ion))
        else:
            e_ion_ev = 0.0
        if exc_keys:
            w_exc = np.array([result.rate_coefficients[k] for k in exc_keys])
            th_exc = np.array([exc_thresh[k] for k in exc_keys])
            tot = float(w_exc.sum())
            e_exc_ev = float(np.sum(w_exc * th_exc) / tot) if tot > 0.0 else float(np.mean(th_exc))
        else:
            e_exc_ev = 0.0

        rows.append(
            {
                "en_td": float(en),
                "mean_energy_ev": float(result.mean_energy),
                "mobility_n": mobility_n,
                "k_ion": k_ion,
                "k_exc": k_exc,
                "e_ion_ev": e_ion_ev,
                "e_exc_ev": e_exc_ev,
                "eedf": [float(x) for x in result.eedf],
            }
        )
        if progress_cb is not None:
            progress_cb(i, n_points, float(en), time.perf_counter() - t0)

    if not rows:
        raise ValueError(
            "run_boltz_sweep: 収束した E/N 点が1つもありませんでした "
            "(opts の en_min_td/en_max_td やメッシュ (d_eps_ev/n_theta) を見直してください)"
        )

    # ε̄ (mean_energy_ev) 昇順に整列する (LMEA はこれをキーにテーブル参照するため)
    order = np.argsort([r["mean_energy_ev"] for r in rows])
    rows = [rows[i] for i in order]
    mean_e_sorted = [r["mean_energy_ev"] for r in rows]
    if any(b <= a for a, b in zip(mean_e_sorted, mean_e_sorted[1:])):
        warnings_out.append(
            "平均電子エネルギー ε̄ が E/N に対して単調増加していません "
            "(LMEA は ε̄ をキーにテーブル参照する前提のため、非単調だと補間・逆引きが不正確になり得ます)"
        )

    meta_opts = dict(resolved)
    # キー名はフロント (BoltzSection.tsx) 互換のため boltzpm_version のまま。値は boltzpmp の版
    meta_opts["boltzpm_version"] = getattr(bp, "__version__", "unknown")
    meta_opts["solver_package"] = "boltzpmp"

    return {
        "en_td": [r["en_td"] for r in rows],
        "mean_energy_ev": mean_e_sorted,
        "mobility_n": [r["mobility_n"] for r in rows],
        "k_ion": [r["k_ion"] for r in rows],
        "k_exc": [r["k_exc"] for r in rows],
        "e_ion_ev": [r["e_ion_ev"] for r in rows],
        "e_exc_ev": [r["e_exc_ev"] for r in rows],
        "eedf_eps_ev": [float(x) for x in solver.mesh.eps_c],
        "eedf": [r["eedf"] for r in rows],
        "source_hash": _hash_processes(processes),
        "opts": meta_opts,
        "warnings": warnings_out,
    }


# ---- fluid1d.py / fluid2d.py が使う ε̄→係数 補間ラッパー -------------------------


@dataclass
class BoltzCoeffs:
    """BoltzTable を ε̄ 昇順に整理した、fluid1d/2d の係数評価専用の軽量ビュー。"""

    eps_grid_ev: np.ndarray
    mobility_n: np.ndarray   # μ_e・N [1/(m・V・s)]
    k_ion: np.ndarray
    k_exc: np.ndarray
    e_ion_ev: np.ndarray
    e_exc_ev: np.ndarray


def boltz_coeffs_from_table(table) -> BoltzCoeffs:
    """BoltzTable (pydantic モデル、または同じキーを持つ dict) から補間用ビューを作る。

    run_boltz_sweep は既に mean_energy_ev 昇順にソートしているはずだが、フロントが
    手を加えた/古い形式の JSON を渡す可能性を考慮し、ここでも防御的にソートし直す。
    """
    if hasattr(table, "model_dump"):
        d = table.model_dump()
    else:
        d = dict(table)
    eps = np.asarray(d["mean_energy_ev"], dtype=np.float64)
    order = np.argsort(eps)
    return BoltzCoeffs(
        eps_grid_ev=eps[order],
        mobility_n=np.asarray(d["mobility_n"], dtype=np.float64)[order],
        k_ion=np.asarray(d["k_ion"], dtype=np.float64)[order],
        k_exc=np.asarray(d["k_exc"], dtype=np.float64)[order],
        e_ion_ev=np.asarray(d["e_ion_ev"], dtype=np.float64)[order],
        e_exc_ev=np.asarray(d["e_exc_ev"], dtype=np.float64)[order],
    )


def boltz_coeffs_at(coeffs: BoltzCoeffs, te_ev: np.ndarray):
    """Te から LMEA 係数を引く (fluid1d._te_and_coeffs / fluid2d._te_and_coeffs が呼ぶ)。

    ε̄=(3/2)Te は近似ではなく、流体側の状態変数の定義 w=(3/2)n_e・Te そのものから
    来る恒等式 — なので Maxwell 経路と同じ「Te を求めてテーブルを引く」呼び出し
    規約のまま、ε̄ をキーにした boltzpm テーブルへ差し替えるだけで済む。

    戻り値: (mu_e_n, k_ion, k_exc, e_ion_ev, e_exc_ev, nu_m_per_ng, eps_bar, out_of_range)
    - mu_e_n: μ_e・N [1/(m・V・s)] (呼び出し側で n_g で割って μ_e にする、Maxwell 側と同じ規約)
    - nu_m_per_ng: 「ν_m_eff = e/(m_e・μ_e)」から逆算した実効運動量移行衝突頻度を
      n_g で割った値 (= Maxwell 側の nu_m_per_ng と同じ単位・同じ使われ方)。
      boltzpm の弾性 rate_coefficients (⟨σ_elastic v⟩) を直接使わない理由: それは
      「今まさに弾性損失の式で使う μ_e」の由来とは別口の平均であり、2つを混在
      させると (二項近似の) 運動量緩和と実際に使っている移動度が整合しなくなる。
      「今の μ_e から一意に逆算した ν_m」を使えば自己無撞着になる、という考え方
      (二項近似で ν_m と μ_e が e/(m・ν_m) の関係で結ばれているのと同じ式を、
      向きを変えて μ_e→ν_m に使っているだけ)。
      単位検算: mobility_n は既に μ・N の形 ([1/(m・V・s)]) なので
      e/(m_e・mobility_n) は [C]/([kg]・[1/(m・V・s)]) = C・m・V・s/kg
      = C・m・(kg・m^2 s^-2 C^-1)・s/kg = m^3/s (Maxwell 側の nu_m_per_ng と同じ単位、
      n_g を掛ければ実際の衝突周波数 [1/s] になる)。
    - out_of_range: ε̄ がテーブル範囲外にクランプされたか (呼び出し側の1回限り警告用)
    """
    grid = coeffs.eps_grid_ev
    eps_bar = 1.5 * np.asarray(te_ev, dtype=np.float64)
    lo, hi = grid[0], grid[-1]
    out_of_range = bool(np.any(eps_bar < lo) or np.any(eps_bar > hi))

    mobility_n = np.asarray(interp_loglog(grid, coeffs.mobility_n, eps_bar), dtype=np.float64)
    k_ion = np.asarray(interp_loglog(grid, coeffs.k_ion, eps_bar), dtype=np.float64)
    k_exc = np.asarray(interp_loglog(grid, coeffs.k_exc, eps_bar), dtype=np.float64)
    e_ion_ev = np.asarray(interp_loglog(grid, coeffs.e_ion_ev, eps_bar), dtype=np.float64)
    e_exc_ev = np.asarray(interp_loglog(grid, coeffs.e_exc_ev, eps_bar), dtype=np.float64)
    nu_m_per_ng = QE / (ME * mobility_n)

    return mobility_n, k_ion, k_exc, e_ion_ev, e_exc_ev, nu_m_per_ng, eps_bar, out_of_range
