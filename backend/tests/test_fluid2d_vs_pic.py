"""fluid2d.py (2D/軸対称 流体) vs pic.py (2D FEM-PIC) の統合検証 (prompts/114)。

test_fluid1d_vs_pic.py (1D 流体 vs 1D PIC) の 2D 版。2D は非構造メッシュの
EAFE/FEM-SG (fluid2d) と粒子ベースの FEM-PIC (pic.py) を同一の小型 CCP 条件で
突き合わせ、桁が合うこと・場が健全であることを担保する。1D 版と同じ理由
(電子平均自由行程がシース厚・拡散長と同程度以上になる低圧領域では流体の
局所平衡近似が非局所加熱を表現できない) に加え、2D は非構造メッシュ・
粒子ノイズの両方が収束を1D よりさらに緩めるため、判定幅は 1D 版よりさらに
広く取る (prompts/114 の指示どおり)。

## 条件の選定経緯 (対話的な事前実行で確認、/tmp のスクリプトで検証)

- 幾何: examples/ccp_demo.json と同じ 20mm×10mm の矩形、左電極 RF (13.56MHz)・
  右電極接地・上下 symmetry (2D だが y 方向には物理的な変化が出ない「実質1D」
  配置 — 2D 実装 (非構造メッシュ・EAFE・2D PIC walk) を経由した突き合わせに
  なることが本質で、1D 突き合わせ (test_fluid1d_vs_pic.py) を単に置き換える
  ものではない)。mesh.size=0.001 (ccp_demo.json と同値) で 488 要素・275 節点
  (「粗めメッシュ数百〜千要素」の指示どおり)。
- eduPIC Ar 解析式断面積 (pic1d_presets.edupic_ar_processes、1D 版と同じ)、
  Ar 50 Pa・300K (1D 版と同じ「流体近似が効きやすい高め圧力」の理由)、
  ion_mass 39.948、SEE γ=0.05 (両電極)、RF 200V・13.56MHz (150〜250V の指示
  範囲内、事前実験で判定が安定した値)。
- PIC: n_macro=8000・dt=1/(400f)・n_steps=10000 (avg 最後の25%)・threads=1・
  seed 固定。2D FEM-PIC は 1D 専用ソルバー (pic1d) と異なり非構造メッシュの
  walk 探索・疎行列 Poisson を毎ステップ行うため要素数を絞る必要があり、
  1D 版のように n_macro=20000・n_steps=12000 まで上げると CI 予算を圧迫する。
  事前実験で n_macro=8000・n_steps=10000 でも中心密度が定常域に達し判定が
  安定することを確認済み。ωpe·dt や衝突頻度の警告 (PicSimulation.warnings)
  は出るが「時間刻みが粗い」という定性的な注意であり、密度・温度が発散せず
  妥当な範囲に収束することを事前実験で確認しているため許容する (1D 版の
  eduPIC 高圧条件でも同様の性質の警告は許容範囲内だった)。
- 流体: init_density_m3=1.5e15・init_te_ev=3.0・n_steps=8000 (avg 最後の25%)。
  同じメッシュ (2D EAFE) で PIC よりずっと軽いはずだが、実測では 1D 版ほど
  PIC との速度差が大きくない (2D 流体は非構造メッシュ上で毎(サブ)ステップ
  spsolve を種ごとに呼ぶため、節点数が少なくても LU 分解のオーバーヘッドが
  無視できない — fluid2d.py モジュール docstring の「陰的線形ソルバー」節
  参照)。そのため 1D 版のような「流体は PIC の1/3以下の時間で終わる」という
  速度担保の判定は課さない (prompts/114 の判定リストにも速度比較は無い —
  4分の合計 CI 予算だけを満たせばよい)。
- 実測 (開発時点のローカル実行、単発の目安): PIC ≈31秒・流体 ≈32秒
  (このテストだけで合計 ≈63秒)。他の2テスト (rz 整合・位相分解) を含めても
  合計 ≈85秒程度で、CI 4分 (240秒) の予算に十分収まる。

## 判定

1. 中心 (gap/2 近傍、±gap/50 の窓平均) n_e: 流体/PIC 比が 0.3〜3.0
   (1D 版の 0.5〜2.0 よりさらに緩い — 2D は非構造メッシュ・粒子ノイズの
   両方の影響で収束が1D より浅いため、prompts/114 の指示どおり)
2. 中心 T_e: 差が ±2.5 eV 以内、かつ両者とも 1〜6 eV
3. 両者の場 (phi/e_abs/n_e/n_i/t_e 系) が全て有限、n_e・n_i ≥ 0
"""

from __future__ import annotations

import time

import numpy as np

from es_sim.fluid2d import Fluid2dSimulation, build_fluid2d_result
from es_sim.pic import PicSimulation
from es_sim.pic1d_presets import edupic_ar_processes
from es_sim.schema import (
    BoundaryCondition,
    Domain,
    Fluid2dSettings,
    Geometry,
    MccGas,
    MccSettings,
    MeshSettings,
    PicSettings,
    Project,
    VoltageRF,
)

# numba (pic.py の walk カーネル) のスレッド数を1に固定する (test_fluid1d_vs_pic.py
# と同じ理由: 結果自体はスレッド数非依存の設計だが、CI 環境間のタイミング変動を
# 抑えるため明示的に固定する。PicSettings.threads=1 とあわせて二重に固定する)
try:
    import numba

    numba.set_num_threads(1)
except ImportError:
    pass

# ---- 両ソルバー共通条件 (モジュール docstring の選定経緯を参照) ------------------------
GAP_M = 0.02
HEIGHT_M = 0.01
MESH_SIZE = 0.001
RF = VoltageRF(amplitude=200.0, freq_hz=13.56e6, phase_deg=0.0)
PRESSURE_PA = 50.0
TEMPERATURE_K = 300.0
ION_MASS_AMU = 39.948
SEE_GAMMA = 0.05


def _ccp_geometry() -> Geometry:
    """examples/ccp_demo.json と同じ 20mm×10mm・左RF/右GND/上下symmetry の矩形。"""
    return Geometry(
        domain=Domain(polygon=[(0, 0), (GAP_M, 0), (GAP_M, HEIGHT_M), (0, HEIGHT_M)]),
        boundaries=[
            BoundaryCondition(edges=[3], type="dirichlet", voltage=0.0, voltage_rf=RF, see_gamma=SEE_GAMMA),
            BoundaryCondition(edges=[1], type="dirichlet", voltage=0.0, see_gamma=SEE_GAMMA),
            BoundaryCondition(edges=[0, 2], type="symmetry"),
        ],
    )


def _fluid_project() -> Project:
    s = Fluid2dSettings(
        init_density_m3=1.5e15, init_te_ev=3.0,
        gas_pressure_pa=PRESSURE_PA, gas_temperature_k=TEMPERATURE_K,
        ion_mass_amu=ION_MASS_AMU,
        n_steps=8000, frame_every=8000, avg_steps=2000, phase_bins=0,
    )
    return Project(coord="xy", geometry=_ccp_geometry(), mesh=MeshSettings(size=MESH_SIZE), fluid2d=s)


def _pic_project() -> Project:
    e_procs, i_procs = edupic_ar_processes()
    s = PicSettings(
        initial_plasma={
            "density": 2.0e15, "te_ev": 3.0, "ti_ev": 0.03,
            "ion_mass_amu": ION_MASS_AMU, "immobile_ions": False, "seed": 0,
        },
        n_macro=8000,
        dt=1.0 / (400.0 * 13.56e6),
        n_steps=10000,
        frame_every=10000,
        avg_steps=2500,
        phase_bins=0,
        threads=1,
        mcc=MccSettings(
            gas=MccGas(name="Ar", pressure_pa=PRESSURE_PA, temperature_k=TEMPERATURE_K),
            electron_processes=e_procs,
            ion_processes=i_procs,
            seed=0,
        ),
    )
    return Project(coord="xy", geometry=_ccp_geometry(), mesh=MeshSettings(size=MESH_SIZE), pic=s)


def _center_avg(x: np.ndarray, arr: np.ndarray, gap: float, half_width: float | None = None) -> float:
    """gap/2 近傍 (既定 ±gap/50) の単純平均 (test_fluid1d_vs_pic.py._center_avg と同じ理由:
    単一節点だと PIC の統計ノイズを直接拾うため、物理的に同じ窓幅で平均する)。"""
    if half_width is None:
        half_width = gap / 50.0
    x = np.asarray(x)
    arr = np.asarray(arr)
    mask = np.abs(x - gap / 2.0) <= half_width
    assert np.any(mask), "中心窓に節点が1つも入らない (half_width が小さすぎる)"
    return float(np.mean(arr[mask]))


def test_fluid_vs_pic2d_ccp_comparison():
    """eduPIC Ar・200V・50Pa・20mm×10mm (上下symmetry) の同一条件で流体2D と
    2D PIC を走らせ、中心 n_e・T_e をモジュール docstring の判定基準で比較する。
    """
    sim_f = Fluid2dSimulation(_fluid_project())
    t0 = time.perf_counter()
    sim_f.run_batch()
    elapsed_f = time.perf_counter() - t0
    result_f = build_fluid2d_result(sim_f, elapsed_s=elapsed_f)

    sim_p = PicSimulation(_pic_project())
    t0 = time.perf_counter()
    sim_p.run_batch()
    elapsed_p = time.perf_counter() - t0

    fields_f = result_f["fields"]
    fields_p = sim_p.fields
    assert fields_f is not None and fields_p is not None

    # ---- 判定3: 有限性・非負性 (以降の比較の前提) ----
    for key in ("phi", "e_abs", "n_e", "n_i", "t_e"):
        arr = np.asarray(fields_f[key])
        assert np.all(np.isfinite(arr)), f"fluid.{key} に非有限値がある"
    assert np.all(np.asarray(fields_f["n_e"]) >= 0.0), "fluid.n_e に負値がある"
    assert np.all(np.asarray(fields_f["n_i"]) >= 0.0), "fluid.n_i に負値がある"
    # pic.py の averaged_fields は t_e ではなく te_ev というキー名 (pic.py 参照)
    for key in ("phi", "e_abs", "n_e", "n_i", "te_ev"):
        arr = np.asarray(fields_p[key])
        assert np.all(np.isfinite(arr)), f"pic.{key} に非有限値がある"
    assert np.all(np.asarray(fields_p["n_e"]) >= 0.0), "pic.n_e に負値がある"
    assert np.all(np.asarray(fields_p["n_i"]) >= 0.0), "pic.n_i に負値がある"

    # ---- 中心窓の抽出 (両ソルバーとも同一の geometry/mesh から生成した同一メッシュ
    # なので、節点の x 座標配列は共有できる) ----
    xs = sim_f.mesh.nodes[:, 0]
    assert np.array_equal(np.round(xs, 12), np.round(sim_p.mesh.nodes[:, 0], 12)), (
        "流体と PIC のメッシュ節点 x 座標が一致しない (同一 geometry/mesh 設定のはず)"
    )

    # ---- 判定1: 中心 n_e ----
    ne_f = _center_avg(xs, fields_f["n_e"], GAP_M)
    ne_p = _center_avg(xs, fields_p["n_e"], GAP_M)
    ratio_ne = ne_f / ne_p
    # 2D は非構造メッシュ (EAFE の負重み・粗い三角形分割) と PIC の統計ノイズの
    # 両方が1D より収束を浅くするため、1D 版 (0.5〜2.0) よりさらに緩い
    # ファクター3の窓で判定する (prompts/114 の指示どおり)
    assert 0.3 <= ratio_ne <= 3.0, (
        f"中心 n_e 比が範囲外: fluid={ne_f:.3e} pic={ne_p:.3e} ratio={ratio_ne:.3f}"
    )

    # ---- 判定2: 中心 T_e ----
    te_f = _center_avg(xs, fields_f["t_e"], GAP_M)
    te_p = _center_avg(xs, fields_p["te_ev"], GAP_M)
    assert abs(te_f - te_p) <= 2.5, f"中心 T_e 差が2.5eVを超えた: fluid={te_f:.3f} pic={te_p:.3f}"
    assert 1.0 <= te_f <= 6.0, f"fluid T_e が想定域外: {te_f:.3f}"
    assert 1.0 <= te_p <= 6.0, f"pic T_e が想定域外: {te_p:.3f}"


# ---- 2. rz 軸対称の整合 -----------------------------------------------------------


def _thin_strip_geometry(gap: float, h: float, rf: VoltageRF) -> Geometry:
    """test_fluid2d.py の test_matches_fluid1d_cross_section_average と全く同じ
    「細長い矩形・上下 symmetry」の geometry。coord だけ xy/rz で切り替えて使う
    (下記 test_rz_matches_xy_thin_strip の理由コメント参照)。
    """
    return Geometry(
        domain=Domain(polygon=[(0, 0), (gap, 0), (gap, h), (0, h)]),
        boundaries=[
            BoundaryCondition(edges=[3], type="dirichlet", voltage=0.0, voltage_rf=rf, see_gamma=0.05),
            BoundaryCondition(edges=[1], type="dirichlet", voltage=0.0, see_gamma=0.05),
            BoundaryCondition(edges=[0], type="symmetry"),
            BoundaryCondition(edges=[2], type="symmetry"),
        ],
    )


def test_rz_matches_xy_thin_strip():
    """rz (軸対称) と xy (平面) で全く同じ geometry/mesh/設定を使い、coord だけを
    切り替えて cross-section 平均プロファイルを比較する (rtol 15%)。

    幾何の選び方 (prompts/114 が要求する「理由コメント」): test_fluid2d.py の
    1D 突き合わせ (test_matches_fluid1d_cross_section_average、fluid1d と rtol
    10% で一致することを既に検証済み) と全く同じ「細長い矩形・上下 symmetry」の
    geometry をそのまま使う。y=0 は rz の対称軸そのもの (edges=[0] が既に
    type="symmetry" — schema._check_rz の r=0 に Dirichlet 禁止という制約とも
    整合)、y=h (edges=[2]) も symmetry (反射壁・流束0) にしてあるため、初期条件
    が y 非依存であれば PDE の解は厳密に y (=r) 非依存であり続ける
    (拡散項の 1/r・∂/∂r(r・∂n/∂r) は ∂n/∂r≡0 なら恒等的に 0 になるため、
    軸対称の計量因子 r があっても y 方向には何も変化を生まない — Poisson・
    ドリフト拡散のどちらも同じ理屈)。つまり厳密解のレベルでは xy と rz は
    完全に同一の1D問題に帰着するはずで、両者のズレは「EAFE の rz r̄ 重み近似・
    壁境界の rz 積分公式 (fluid2d.py モジュール docstring 参照) が実装として
    正しいか」だけを反映する — これは prompts/114 が求める「rz の r 重み
    (体積・エッジ・壁) が正しいことの実効検証」そのものであり、fluid1d との
    再比較よりも rz 実装のバグを直接検出できる分、判定として明快である。
    事前実験 (対話的スクリプト) では n_e/T_e とも rtol 1〜2% 程度の差しか
    出ず、指示の rtol 15% に対して大きな余裕がある (差の残りは r̄ 近似・
    浮動小数点の LU 分解順序などに由来する数値誤差で、バグではない)。
    """
    gap = 0.02
    n_cells = 20
    h = gap / n_cells  # 正方セル (test_fluid2d.py と同じ理由: checkerboard 対角の非単調性回避)
    rf = VoltageRF(amplitude=100.0, freq_hz=13.56e6, phase_deg=0.0)
    n_steps = 6000
    avg_steps = 1500
    n0 = 1.0e14
    p_gas = 20.0

    s = Fluid2dSettings(
        init_density_m3=n0, init_te_ev=2.0,
        gas_pressure_pa=p_gas, gas_temperature_k=300.0,
        n_steps=n_steps, frame_every=n_steps, avg_steps=avg_steps,
    )

    def run(coord: str) -> Fluid2dSimulation:
        geo = _thin_strip_geometry(gap, h, rf)
        proj = Project(coord=coord, geometry=geo, mesh=MeshSettings(size=h, mode="structured"), fluid2d=s)
        sim = Fluid2dSimulation(proj)
        sim.run_batch(store_frames=False)
        return sim

    sim_xy = run("xy")
    sim_rz = run("rz")

    def cross_avg(sim: Fluid2dSimulation, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs_all = sim.mesh.nodes[:, 0]
        uniq_x = np.unique(np.round(xs_all, 12))
        return uniq_x, np.array([field[np.isclose(xs_all, x)].mean() for x in uniq_x])

    ux_xy, ne_xy = cross_avg(sim_xy, sim_xy.fields["n_e"])
    _, te_xy = cross_avg(sim_xy, sim_xy.fields["t_e"])
    ux_rz, ne_rz = cross_avg(sim_rz, sim_rz.fields["n_e"])
    _, te_rz = cross_avg(sim_rz, sim_rz.fields["t_e"])

    # メッシュ生成は coord に依存しない (同じ polygon/mesh.size) ので、x 座標の
    # 一意点集合が完全一致するはず — ずれていれば xy/rz 比較自体が無意味になる
    assert np.array_equal(ux_xy, ux_rz)

    def relerr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        denom = np.maximum(np.abs(a), np.abs(b))
        denom = np.where(denom < 1.0, 1.0, denom)
        return np.abs(a - b) / denom

    ne_err = relerr(ne_xy, ne_rz)
    te_err = relerr(te_xy, te_rz)
    assert ne_err.max() < 0.15, f"n_e cross-section 平均が xy/rz で rtol 15% を超えて乖離: {ne_err.max():.3f}"
    assert te_err.max() < 0.15, f"T_e cross-section 平均が xy/rz で rtol 15% を超えて乖離: {te_err.max():.3f}"

    assert np.all(np.isfinite(sim_rz.n_e)) and np.all(sim_rz.n_e >= 0.0)
    assert np.all(np.isfinite(sim_rz.phi))


# ---- 3. 位相分解の健全性 (fluid2d 側) -----------------------------------------------


def test_phase_bins_produce_finite_cycle_fields():
    """phase_bins > 0 の短い実行で cycle が生成され、各ビンの場が有限・bins 数が正しい。

    test_fluid2d.py には phase_bins=0 (cycle 無効) のケースしか無い
    (test_rz_axisymmetric_smoke の `assert result["cycle"] is None` コメント参照)
    ので、phase_bins > 0 の健全性はここで新規に確認する (prompts/114 の指示どおり
    「省略可」だが未カバーだったため追加)。RF 周期あたりのステップ数が
    ~2000 (dt 既定 = 周期/2000) になるため、avg_steps はその2周期分
    (=n_steps、事前実験で全ビンにステップが入ることを確認済み) を取り、
    位相ビン (bins=8) 全てにサンプルが入ることを直接検証する。
    """
    gap = 0.02
    n_cells = 20
    h = gap / n_cells
    rf = VoltageRF(amplitude=100.0, freq_hz=13.56e6, phase_deg=0.0)
    bins = 8
    n_steps = 4000  # dt 既定 (周期/2000) でおよそ2周期分、事前実験の実測で確認済み

    geo = _thin_strip_geometry(gap, h, rf)
    s = Fluid2dSettings(
        init_density_m3=1.0e14, init_te_ev=2.0,
        gas_pressure_pa=20.0, gas_temperature_k=300.0,
        n_steps=n_steps, frame_every=n_steps, avg_steps=2000, phase_bins=bins,
    )
    sim = Fluid2dSimulation(Project(coord="xy", geometry=geo, mesh=MeshSettings(size=h, mode="structured"), fluid2d=s))
    sim.run_batch(store_frames=False)

    cycle = sim.cycle
    assert cycle is not None
    assert cycle["bins"] == bins
    assert cycle["freq_hz"] == 13.56e6
    n_nodes = sim.n_nodes
    for key in ("phi", "n_e", "n_i", "t_e"):
        arr = np.asarray(cycle[key])
        assert arr.shape == (bins, n_nodes)
        assert np.all(np.isfinite(arr)), f"cycle.{key} に非有限値がある"
    assert np.all(np.asarray(cycle["n_e"]) >= 0.0)
    # 平均区間がちょうど約2周期なので、全ビンに少なくとも1ステップは割り当てられる
    assert np.all(sim._cycle_count > 0), "サンプルの入らない位相ビンがある"

    result = build_fluid2d_result(sim, elapsed_s=1.0)
    assert result["cycle"] is not None
    assert result["cycle"]["bins"] == bins
