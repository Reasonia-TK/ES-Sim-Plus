"""fluid1d.py (1D 流体) vs pic1d.py (1D PIC/MCC) の統合検証 (prompts/110、Phase E)。

流体 (ドリフト拡散 + 電子エネルギー) と 1D PIC/MCC を**同一条件**で走らせ、中心
n_e・T_e・シースエッジが桁で一致し、profiles が全て有限であることを自動テストで
担保する。低圧では電子の非局所加熱 (フルード近似は表現できない — 電子平均自由行程
がシース厚・拡散長と同程度以上になると局所平衡近似が崩れる) により両者が乖離する
ことが物理的に知られているため、判定はファクター2〜3程度まで緩め、「比較が回る
こと・桁が合うこと」の保証を目的とする (精密な数値一致は要求しない)。

## 条件の選定経緯 (対話的な事前実行で確認した内容)

- eduPIC Ar 解析式断面積 (pic1d_presets.edupic_ar_processes、電子: elastic/
  excitation/ionization、イオン: isotropic/backscat)、gap 2.5 cm、13.56 MHz
  250 V (左 RF・右接地)、Ar **50 Pa・300 K** (eduPIC 既定の 10 Pa より高圧側を選び、
  流体近似が効きやすい条件にする)、ion_mass 39.948、SEE γ=0.05 (両電極。
  既存の fluid1d CCP スモークテスト (test_fluid1d.py) と同じ値を流用し、シースが
  物理的に妥当な範囲に収まることを利用する)。
- PIC: n_cells=64・n_macro=20000・dt=1/(400f)・n_steps=12000 (avg は最後の25%)。
  事前実行で n_steps を6000〜8000まで削ると中心密度がまだ緩和途上で判定が
  ぎりぎりになったため、指示が許す上限の12000を採用して安定させた。
- 流体: n_cells=100・n_steps=5000 (avg は最後の25%)。流体は PIC よりずっと軽い
  (粒子を追わず節点場のみを解く) ため、PIC ほど厳密に定常収束させなくても
  判定の緩い許容幅には十分収まることを事前実験で確認済み。CI 時間の余裕は
  判定が重い PIC 側 (時間・統計ノイズの両方) に回し、流体は判定5 (速さの担保)
  を安定して満たす範囲で短く済ませている。
- 初期密度は両ソルバーとも定常想定値に近い値 (流体 1.5e14 m^-3 台では緩和に
  時間がかかりすぎたため 1.5e15、PIC は定常密度がやや高めに落ち着く傾向が
  あったため 2.3e15) を与えて緩和時間を短縮した。
- phase_bins=0 で位相分解アキュムレータを無効化 (この比較には不要な計算コストを
  削り、CI 時間を節約するため)。
- 実測 (開発時点のローカル実行、単発の目安): 流体 ≈14秒・PIC ≈69秒 (合計 ≈83秒)。
  CI 3分 (180秒) の予算に対して十分な余裕があり、環境による多少の速度変動を
  見込んでも収まる。

## 判定 (根拠は各アサーション付近のコメント参照)

1. 中心 (gap/2 近傍、±gap/50 の窓平均) n_e: 流体/PIC 比が 0.5〜2.0
2. 中心 T_e: 差が ±2 eV 以内、かつ両者とも 1〜6 eV
3. シースエッジ (Brinkmann left_s/right_s、prompts/97): 比が 0.3〜3.0
4. 両者の profiles (phi/e/n_e/n_i/t_e) が全て有限、n_e・n_i ≥ 0
5. 流体の elapsed_s が PIC の elapsed_s の 1/3 以下 (速さの担保。CI 環境変動を
   考慮し緩めの基準にした)

## 再現性

PIC は seed 固定 (Pic1dSettings.seed、MccSettings.seed とも 0)。mcc.py が使う
numba 並列カーネル (_numba_kernels、prange) は「粒子ごとに独立な書き込み先を
持ち、乱数は njit の外側で事前生成した配列を渡す」設計になっている (mcc.py・
pic1d.py の docstring 参照) ため、スレッド数を変えても結果は bit-exact に一致する
ことを事前実験 (threads=1 vs 既定値で同一条件を実行し n_e/phi プロファイルが
np.array_equal になることを確認) 済みだが、CI 環境ごとのスレッド数によるタイミング
変動を抑える目的で明示的に threads=1 に固定する。
"""

from __future__ import annotations

import time

import numpy as np

from es_sim.fluid1d import Fluid1dSimulation, build_fluid1d_result
from es_sim.pic1d import Pic1dSimulation, build_pic1d_result
from es_sim.pic1d_presets import edupic_ar_processes
from es_sim.schema import (
    Domain,
    Fluid1dSettings,
    Geometry,
    MccGas,
    MccSettings,
    MeshSettings,
    Pic1dElectrode,
    Pic1dSettings,
    Project,
    VoltageRF,
)

# numba (mcc.py の並列カーネル) が使えるならスレッド数を1に固定する (上記モジュール
# docstring の「再現性」参照。結果自体はスレッド数非依存だが、CI 環境間のタイミング
# 変動 — 判定5 が時間比較のため — を抑えるために固定する)
try:
    import numba

    numba.set_num_threads(1)
except ImportError:
    pass

_DUMMY_GEOMETRY = Geometry(domain=Domain(polygon=[(0, 0), (1, 0), (1, 1), (0, 1)]))
_DUMMY_MESH = MeshSettings(size=0.1)

# ---- 両ソルバー共通条件 (モジュール docstring の選定経緯を参照) ------------------------
GAP_M = 0.025
RF = VoltageRF(amplitude=250.0, freq_hz=13.56e6, phase_deg=0.0)
PRESSURE_PA = 50.0
TEMPERATURE_K = 300.0
ION_MASS_AMU = 39.948
SEE_GAMMA = 0.05


def _project_fluid(s: Fluid1dSettings) -> Project:
    return Project(geometry=_DUMMY_GEOMETRY, mesh=_DUMMY_MESH, fluid1d=s)


def _project_pic(s: Pic1dSettings) -> Project:
    return Project(geometry=_DUMMY_GEOMETRY, mesh=_DUMMY_MESH, pic1d=s)


def _fluid_settings() -> Fluid1dSettings:
    """流体側条件。electron_processes=[] のままなので eduPIC Ar 解析式が既定で使われる
    (fluid1d.Fluid1dSimulation.__init__、electron_processes 未指定時の分岐を参照)。
    """
    return Fluid1dSettings(
        gap_m=GAP_M,
        n_cells=100,
        left=Pic1dElectrode(v_dc=0.0, voltage_rf=RF, see_gamma=SEE_GAMMA),
        right=Pic1dElectrode(v_dc=0.0, see_gamma=SEE_GAMMA),
        init_density_m3=1.5e15,
        init_te_ev=3.0,
        gas_pressure_pa=PRESSURE_PA,
        gas_temperature_k=TEMPERATURE_K,
        ion_mass_amu=ION_MASS_AMU,
        n_steps=5000,
        frame_every=5000,
        avg_steps=1250,
        phase_bins=0,
    )


def _pic_settings() -> Pic1dSettings:
    """PIC 側条件。eduPIC Ar 解析式断面積を電子・イオンとも明示的に渡す。"""
    e_procs, i_procs = edupic_ar_processes()
    return Pic1dSettings(
        gap_m=GAP_M,
        n_cells=64,
        left=Pic1dElectrode(v_dc=0.0, voltage_rf=RF, see_gamma=SEE_GAMMA),
        right=Pic1dElectrode(v_dc=0.0, see_gamma=SEE_GAMMA),
        init_density_m3=2.3e15,
        init_te_ev=3.0,
        init_ti_ev=0.03,
        ion_mass_amu=ION_MASS_AMU,
        n_macro=20000,
        dt=1.0 / (400.0 * 13.56e6),
        n_steps=12000,
        frame_every=12000,
        avg_steps=3000,
        phase_bins=0,
        mcc=MccSettings(
            gas=MccGas(name="Ar", pressure_pa=PRESSURE_PA, temperature_k=TEMPERATURE_K),
            electron_processes=e_procs,
            ion_processes=i_procs,
            seed=0,
        ),
        seed=0,
    )


def _center_avg(x, arr, gap: float, half_width: float | None = None) -> float:
    """gap/2 近傍 (既定 ±gap/50) の単純平均。

    単一節点の値だけを見ると PIC 側の統計ノイズ (有限マクロ粒子数由来) を直接
    拾ってしまうため、両ソルバーで物理的に同じ位置にある小窓を対称に切り出して
    平均する (fluid は n_cells=100・PIC は n_cells=64 なので窓に入る節点数は
    ソルバーごとに異なるが、物理的な窓幅 (gap/25) は共通)。
    """
    if half_width is None:
        half_width = gap / 50.0
    x = np.asarray(x)
    arr = np.asarray(arr)
    mask = np.abs(x - gap / 2.0) <= half_width
    assert np.any(mask), "中心窓に節点が1つも入らない (half_width が小さすぎる)"
    return float(np.mean(arr[mask]))


def test_fluid_vs_pic1d_ccp_comparison():
    """eduPIC Ar・250V・50Pa・gap2.5cm の同一条件で流体と PIC を走らせ、
    中心 n_e・T_e・シースエッジ・実行時間をモジュール docstring の判定基準で比較する。
    """
    # ---- 流体を走らせる (build_fluid1d_result の elapsed_s は server.py と同じ
    # 「run_batch を perf_counter で挟む」計測方法に揃える) ----
    sim_f = Fluid1dSimulation(_project_fluid(_fluid_settings()))
    t0 = time.perf_counter()
    sim_f.run_batch()
    elapsed_f = time.perf_counter() - t0
    result_f = build_fluid1d_result(sim_f, elapsed_s=elapsed_f)

    # ---- PIC を走らせる ----
    sim_p = Pic1dSimulation(_project_pic(_pic_settings()))
    t0 = time.perf_counter()
    sim_p.run_batch()
    elapsed_p = time.perf_counter() - t0
    result_p = build_pic1d_result(sim_p, elapsed_s=elapsed_p)

    profiles_f = result_f["profiles"]
    profiles_p = result_p["profiles"]
    assert profiles_f is not None and profiles_p is not None

    # ---- 判定4: 有限性・非負性 (これが崩れていると以降の比較自体が無意味になる
    # ので最初に確認する) ----
    for name, profiles in (("fluid", profiles_f), ("pic", profiles_p)):
        for key in ("phi", "e", "n_e", "n_i", "t_e"):
            arr = np.asarray(profiles[key])
            assert np.all(np.isfinite(arr)), f"{name}.{key} に非有限値がある"
        assert np.all(np.asarray(profiles["n_e"]) >= 0.0), f"{name}.n_e に負値がある"
        assert np.all(np.asarray(profiles["n_i"]) >= 0.0), f"{name}.n_i に負値がある"

    # ---- 判定1: 中心 n_e ----
    ne_f = _center_avg(profiles_f["x"], profiles_f["n_e"], GAP_M)
    ne_p = _center_avg(profiles_p["x"], profiles_p["n_e"], GAP_M)
    ratio_ne = ne_f / ne_p
    # 50 Pa (流体近似が効きやすい高め圧力) を選んでもなお、PIC の方が非局所加熱・
    # 統計ノイズの影響でやや高めの密度になる傾向が事前実験で確認できた。
    # 桁が合っていることの確認が目的なのでファクター2の広い窓で判定する
    assert 0.5 <= ratio_ne <= 2.0, (
        f"中心 n_e 比が範囲外: fluid={ne_f:.3e} pic={ne_p:.3e} ratio={ratio_ne:.3f}"
    )

    # ---- 判定2: 中心 T_e ----
    te_f = _center_avg(profiles_f["x"], profiles_f["t_e"], GAP_M)
    te_p = _center_avg(profiles_p["x"], profiles_p["t_e"], GAP_M)
    # PIC は電子加熱の非局所性・統計ノイズを含むため流体よりやや高温になりがちだが、
    # 同じ物理機構 (RF シース加熱主体) で決まる量なので絶対差2eV以内という
    # ゆるい基準で「同じオーダーで合っている」ことだけを確認する
    assert abs(te_f - te_p) <= 2.0, f"中心 T_e 差が2eVを超えた: fluid={te_f:.3f} pic={te_p:.3f}"
    assert 1.0 <= te_f <= 6.0, f"fluid T_e が想定域外: {te_f:.3f}"
    assert 1.0 <= te_p <= 6.0, f"pic T_e が想定域外: {te_p:.3f}"

    # ---- 判定3: シースエッジ (Brinkmann、prompts/97) ----
    sheath_f = result_f["sheath"]
    sheath_p = result_p["sheath"]
    assert sheath_f is not None and sheath_p is not None
    for side in ("left_s", "right_s"):
        sf = sheath_f[side]
        sp = sheath_p[side]
        assert sf is not None and sp is not None
        ratio = sf / sp
        # シースは PIC のシース振動・統計ノイズや、流体側の壁境界条件の簡略化
        # (電子に明示的なドリフト項を含めない Hagelaar & Kroesen の簡略形、
        # fluid1d.py 冒頭 docstring 参照) の影響を n_e・T_e より強く受けるため、
        # 判定1・2よりさらに緩い 0.3〜3.0 のファクター3で判定する
        assert 0.3 <= ratio <= 3.0, (
            f"{side} 比が範囲外: fluid={sf:.4g} pic={sp:.4g} ratio={ratio:.3f}"
        )

    # ---- 判定5: 実行時間 (速さの担保) ----
    # 流体は粒子を追わず節点場のみを解くため PIC よりずっと軽い計算のはずで、
    # PIC の1/3以下の時間で終わることを確認する。事前実験での実測比は約0.2
    # だったが、CI 環境のコア数・負荷変動を見込んでさらに緩い 1/3 を基準にした
    assert elapsed_f <= elapsed_p / 3.0, (
        f"流体が想定より遅い (速さの担保に失敗): fluid={elapsed_f:.2f}s "
        f"pic={elapsed_p:.2f}s (fluid <= pic/3 を要求)"
    )
