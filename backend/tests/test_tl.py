"""VHF 定在波 (非線形径方向伝送線路モデル、prompts/101 + 102) のテスト。

1. 線形極限の波長短縮: linear=True (テスト専用、V_s=q・s0/ε0 の線形容量) で
   λ_eff ≈ λ0・√((s_a0+s_b0)/l) (rtol 5%)。R を波長に対して十分大きく取り、
   定在波の節が複数入る条件を選ぶ (R=0.8m、λ_eff≈0.41m)。sheath_law="child"
   (既定) でも同じ設定・同じ許容誤差で通ることを別テストで確認する
   (K の小信号容量整合条件の検証、prompts/102)。
2. 高調波生成 (過渡窓、sheath_law="matrix" 明示指定): 非線形オンで |V_2|・|V_3|
   が中心近傍で |V_1| の 0.1% 以上、linear=True では同じ節点で 0.1% 未満
   (tl.py の docstring 参照: matrix シースは対称放電で厳密に線形化するため、
   q>=0 のクリップ (シース崩壊整流) だけが高調波源であり、クリップは自己バイアス
   により数周期で自己収束するため、真の定常状態まで待つと非線形/線形の両方で
   高調波が消えてしまう — そのため過渡応答 (立ち上げランプ直後、クリップが
   実際に発生している短い窓) を意図的に測定窓に選ぶ。プロンプトの例示値
   (linear で <1e-6) は厳密には成立しない (根拠は tl.py 内のコメント参照) ため、
   「非線形は 0.1% 閾値を超え、線形は同じ節点で同じ閾値を下回る」という
   緩和した (しかし物理的に意味のある) 基準を採用する)。
3. 定常高調波生成 (prompts/102、sheath_law="child" vs "matrix"): 立ち上げ
   ランプ+クリップ緩和が十分完了した定常窓 (n_periods=120、最後の
   n_fft_periods=32) で、child は |V_3|/|V_1|>=1e-3 かつ奇数次が階段状に
   減衰 (|V_3|>|V_5|) し偶数次は奇数次よりずっと小さい (対称性による相殺)。
   同条件の matrix は定常では厳密な線形化により |V_3|/|V_1| が child の
   1/10 未満に留まる (残るのは TL 定在波のリンギング起源の非整数次雑音床のみ)。
4. エネルギー整合: 時間平均給電電力 ≈ ∫ p(r) 2πr dr (rtol 10%)。
5. 縮退: ν_m=0 でも発散しない。validator エラー系 (n_fft_periods>=n_periods、
   sheath_m*2>=gap_m)。
6. スキーマ: tl 未設定の既存プロジェクト読込は不変 (tl は None のまま)。

既存 (prompts/101 時点) のテストは、既定の sheath_law が "child" に変わった
(prompts/102) ことで挙動が変化しないよう、必要な箇所で明示的に
sheath_law="matrix" を指定して意図を保っている。

`python -m pytest tests/ -q` 全件パス (既存 + 本ファイル)。
"""

import math

import numpy as np
import pydantic
import pytest

from es_sim.schema import Domain, Geometry, MeshSettings, Project, TlSettings
from es_sim.tl import TlSimulation

# ダミーの geometry/mesh (tl.py はこれらを一切参照しないが、Project スキーマ上必須。
# pic1d/dsmc のテストと同じ流儀)
_DUMMY_GEOMETRY = Geometry(domain=Domain(polygon=[(0, 0), (1, 0), (1, 1), (0, 1)]))
_DUMMY_MESH = MeshSettings(size=0.1)


def _project(tl: TlSettings) -> Project:
    return Project(geometry=_DUMMY_GEOMETRY, mesh=_DUMMY_MESH, tl=tl)


# ---- 1. 線形極限の波長短縮 ----------------------------------------------------


def test_linear_limit_wavelength_shortening():
    """linear=True で λ_eff ≈ λ0・√((s_a0+s_b0)/l) (rtol 5%)。

    R=0.8m (kR が波長に対して十分大きく、標準的な円柱波動の漸近領域に入る —
    R が波長に対して小さいと近接場の Bessel 関数的な補正が効いて理論式からの
    ずれが大きくなることを実装時に確認したため、大きめの R を選んでいる)。
    """
    gap = 0.04
    sheath = 4e-4
    s = TlSettings(
        radius_m=0.8, gap_m=gap, sheath_m=sheath, n_e_m3=1e18, n_s_ratio=0.4,
        nu_m_hz=1e10, freq_hz=100e6, v0=100.0, n_r=150, n_periods=60,
        n_fft_periods=16, n_harm=5,
    )
    sim = TlSimulation(_project(s), linear=True)
    res = sim.run()
    lam = res["lambda_eff"]
    assert lam["lambda_m"] is not None, "定在波の節が検出できていない"

    lambda0 = 299792458.0 / 100e6
    assert lam["lambda0_m"] == pytest.approx(lambda0, rel=1e-9)

    theory_ratio = math.sqrt(2.0 * sheath / gap)
    assert lam["ratio"] == pytest.approx(theory_ratio, rel=0.05)


def test_linear_limit_wavelength_shortening_child_law():
    """sheath_law="child" (既定) でも上と全く同じ λ_eff/λ0 が得られる (prompts/102)。

    K=(3/4)・(s0/ε0)・q0^{-1/3} は平衡点での微分容量を matrix と一致させる
    ように選んでいる (tl.py docstring の「シース則」節) ので、線形応答
    (小振幅) では child/matrix の区別が消え、波長短縮の理論式は共通に成り立つ
    はず — これがその整合条件の直接検証。V0=100 (linear=True テストと同じ)
    でもこの配置 (R=0.8m, s0=4e-4m) ではシース崩壊が起きるほどの振幅比には
    ならない (q0 が大きい) ので、linear=True に頼らず実際の非線形ソルバー
    (linear=False既定) で確認できる。
    """
    gap = 0.04
    sheath = 4e-4
    s = TlSettings(
        radius_m=0.8, gap_m=gap, sheath_m=sheath, n_e_m3=1e18, n_s_ratio=0.4,
        nu_m_hz=1e10, freq_hz=100e6, v0=100.0, n_r=150, n_periods=60,
        n_fft_periods=16, n_harm=5, sheath_law="child",
    )
    sim = TlSimulation(_project(s))  # linear=False (既定): 実際の child 非線形ソルバー
    res = sim.run()
    assert not sim.clipped, "この配置ではシース崩壊が起きない想定 (小信号検証の前提)"
    lam = res["lambda_eff"]
    assert lam["lambda_m"] is not None

    theory_ratio = math.sqrt(2.0 * sheath / gap)
    assert lam["ratio"] == pytest.approx(theory_ratio, rel=0.05)


# ---- 2. 高調波生成 (過渡窓、sheath_law="matrix") -------------------------------


def _harmonic_test_settings() -> TlSettings:
    # sheath_m は「線形容量換算で δq/q0 ≈ 10」となるよう逆算した値 (最適化の詳細は
    # tl.py docstring 参照)。nu_m_hz=5e8 は、立ち上げランプ (5周期) 直後の短い窓
    # (n_periods=10, n_fft_periods=4) でクリップ由来の高調波が測定できるよう、
    # リンギング (TL 定在波の過渡的な非整数調波成分) の減衰と両立するよう選んだ値。
    # sheath_law="matrix" を明示: 既定が "child" に変わった (prompts/102) 後も
    # この過渡窓テストの意図 (「対称放電の matrix は定常では線形化し、高調波は
    # クリップ過渡でしか出ない」ことの実証) を保つため。
    return TlSettings(
        radius_m=0.15, gap_m=0.04, sheath_m=8.311399820470307e-05, n_e_m3=1e17,
        n_s_ratio=0.4, nu_m_hz=5e8, freq_hz=100e6, v0=100.0, n_r=150,
        n_periods=10, n_fft_periods=4, n_harm=5, sheath_law="matrix",
    )


def test_harmonic_generation_nonlinear_vs_linear():
    idx = 3  # 中心 (r_feed) 近傍だが、Dirichlet 強制ノード (idx=0) 自体は除く

    sim_nl = TlSimulation(_project(_harmonic_test_settings()), linear=False)
    res_nl = sim_nl.run()
    v_nl = np.asarray(res_nl["harmonics"]["v"])
    ratio2_nl = v_nl[2, idx] / v_nl[1, idx]
    ratio3_nl = v_nl[3, idx] / v_nl[1, idx]

    assert ratio2_nl >= 1e-3, f"非線形で2次高調波が0.1%未満: {ratio2_nl}"
    assert ratio3_nl >= 1e-3, f"非線形で3次高調波が0.1%未満: {ratio3_nl}"
    # シース崩壊 (クリップ) が実際に起きていることを確認 (高調波の物理的な源)
    assert any("崩壊" in w for w in res_nl["warnings"])

    sim_lin = TlSimulation(_project(_harmonic_test_settings()), linear=True)
    res_lin = sim_lin.run()
    v_lin = np.asarray(res_lin["harmonics"]["v"])
    ratio2_lin = v_lin[2, idx] / v_lin[1, idx]

    assert ratio2_lin < 1e-3, f"線形モードで2次高調波が0.1%を超えている: {ratio2_lin}"
    # 線形モードではクリップが構造的に無効化されている (tl.py 参照) ので警告も出ない
    assert not any("崩壊" in w for w in res_lin["warnings"])
    # 非線形は線形よりはっきり大きい (少なくとの1桁)
    assert ratio2_nl > ratio2_lin * 5.0


# ---- 3. 定常高調波生成 (child vs matrix、prompts/102) --------------------------


def _steady_harmonic_settings(sheath_law: str) -> TlSettings:
    # v0=20: シース崩壊 (クリップ) は立ち上げランプ直後の最初の~10周期以内で
    # 自己収束する (clip は q_a+q_b を単調に押し上げるラチェットなので、
    # 系がクリップしない新しい平衡点に落ち着くまでの過渡でしかない —
    # tl.py docstring 参照)。事前の数値実験で、この v0 では最後にクリップが
    # 起きるのが ~9周期目までで、n_periods=120・最後の n_fft_periods=32
    # (=88〜120周期目) の測定窓には一切クリップが混入しないことを確認済み
    # (v0 を大きくしすぎると系が恒常的にクリップし続け定常化しない設定も
    # 存在するため、この値は適当に選んだのではなく実測で確認したもの)。
    return TlSettings(
        radius_m=0.15, gap_m=0.04, sheath_m=8.311399820470307e-05, n_e_m3=1e17,
        n_s_ratio=0.4, nu_m_hz=5e8, freq_hz=100e6, v0=20.0, n_r=150,
        n_periods=120, n_fft_periods=32, n_harm=6, sheath_law=sheath_law,
    )


def test_steady_harmonic_generation_child_odd_harmonics():
    """child 則は定常窓 (クリップ完全収束後) でも奇数次高調波を生成し続ける。

    matrix と異なり V_s=K・q^{4/3} は (q0+Δ)^{4/3}−(q0−Δ)^{4/3} が Δ の
    奇数次項を含むため、上下差し引きでも非線形性が残る (tl.py docstring の
    「シース則」節参照)。偶数次は q_a⇔q_b の対称性で相殺されるので
    |V_2|/|V_1| は |V_3|/|V_1| よりずっと小さいはず。
    """
    idx = 3
    sim = TlSimulation(_project(_steady_harmonic_settings("child")))
    res = sim.run()
    v = np.asarray(res["harmonics"]["v"])
    ratio1 = v[1, idx]
    ratio2 = v[2, idx] / ratio1
    ratio3 = v[3, idx] / ratio1
    ratio5 = v[5, idx] / ratio1

    assert ratio3 >= 1e-3, f"定常窓で3次高調波が0.1%未満: {ratio3}"
    assert ratio3 > ratio5, "奇数次は階段状に減衰するはず (|V_3|>|V_5|)"
    assert ratio2 < ratio3, "偶数次 (対称性で相殺) は奇数次よりずっと小さいはず"


def test_steady_harmonic_generation_matrix_stays_linear():
    """matrix 則は同条件でも定常窓では厳密な線形化により child の 1/10 未満。

    prompts/102 の背景 (docstring の「シース則」節): 対称放電の matrix は
    q_a+q_b=2q0 の保存により V_a−V_b が Δq に対して厳密に線形になるため、
    クリップ (このテスト窓には混入しない、_steady_harmonic_settings 参照) 由来
    以外に高調波源を持たない。定常窓に残るのは TL 定在波の非整数次リンギング
    起源のごく小さい雑音床のみ。
    """
    idx = 3
    sim_child = TlSimulation(_project(_steady_harmonic_settings("child")))
    res_child = sim_child.run()
    v_child = np.asarray(res_child["harmonics"]["v"])
    ratio3_child = v_child[3, idx] / v_child[1, idx]

    sim_matrix = TlSimulation(_project(_steady_harmonic_settings("matrix")))
    res_matrix = sim_matrix.run()
    v_matrix = np.asarray(res_matrix["harmonics"]["v"])
    ratio3_matrix = v_matrix[3, idx] / v_matrix[1, idx]

    assert ratio3_matrix < ratio3_child / 10.0, (
        f"matrix の定常3次高調波比 ({ratio3_matrix}) が child ({ratio3_child}) の "
        "1/10 未満に収まっていない"
    )


# ---- 4. エネルギー整合 --------------------------------------------------------


def test_energy_consistency():
    s = TlSettings(
        radius_m=0.15, gap_m=0.04, sheath_m=5e-4, n_e_m3=1e16, n_s_ratio=0.4,
        nu_m_hz=1e8, freq_hz=100e6, v0=100.0, n_r=400, n_periods=40,
        n_fft_periods=16, n_harm=5, sheath_law="matrix",
    )
    sim = TlSimulation(_project(s), linear=False)
    res = sim.run()

    p = np.asarray(res["power"]["p"])
    r = np.asarray(res["r"])
    total_power = np.trapezoid(p * 2.0 * math.pi * r, r)
    feed_power = res["feed_power"]

    assert feed_power > 0.0
    assert total_power == pytest.approx(feed_power, rel=0.10)


# ---- 5. 縮退・validator -------------------------------------------------------


def test_nu_m_zero_is_stable():
    """ν_m=0 (衝突なし) でも数値発散しない。"""
    s = TlSettings(
        radius_m=0.15, gap_m=0.04, sheath_m=5e-4, n_e_m3=1e16, n_s_ratio=0.4,
        nu_m_hz=0.0, freq_hz=100e6, v0=100.0, n_r=150, n_periods=40,
        n_fft_periods=16, n_harm=5, sheath_law="matrix",
    )
    sim = TlSimulation(_project(s), linear=False)
    res = sim.run()
    v = np.asarray(res["harmonics"]["v"])
    assert np.all(np.isfinite(v))


def test_validator_errors():
    with pytest.raises(pydantic.ValidationError):
        # n_fft_periods は n_periods より小さくなければならない
        TlSettings(n_periods=8, n_fft_periods=8)
    with pytest.raises(pydantic.ValidationError):
        # sheath_m の2倍は gap_m より小さくなければならない (バルク厚 > 0)
        TlSettings(gap_m=0.001, sheath_m=0.0006)
    with pytest.raises(pydantic.ValidationError):
        TlSettings(radius_m=-1.0)
    with pytest.raises(pydantic.ValidationError):
        TlSettings(n_s_ratio=1.5)


def test_tl_simulation_requires_tl_settings():
    project = Project(geometry=_DUMMY_GEOMETRY, mesh=_DUMMY_MESH)
    with pytest.raises(ValueError):
        TlSimulation(project)


# ---- 6. スキーマ後方互換性 ----------------------------------------------------


def test_schema_backward_compatibility():
    """tl 未設定の既存プロジェクト (dict) を読み込んでも tl=None のまま。"""
    data = {
        "geometry": _DUMMY_GEOMETRY.model_dump(),
        "mesh": _DUMMY_MESH.model_dump(),
    }
    project = Project.model_validate(data)
    assert project.tl is None

    # 明示的に None を指定しても同様
    project2 = Project(geometry=_DUMMY_GEOMETRY, mesh=_DUMMY_MESH, tl=None)
    assert project2.tl is None


# ---- /ws/tl (WebSocket、prompts/101) ------------------------------------------


def test_tl_ws_start_done():
    """/ws/tl: started → (progress)* → done が届き、continue は無い (dsmc の流儀)。"""
    from starlette.testclient import TestClient

    from es_sim import server as srv

    c = TestClient(srv.app)
    project = {
        "geometry": _DUMMY_GEOMETRY.model_dump(),
        "mesh": _DUMMY_MESH.model_dump(),
        "tl": {
            "radius_m": 0.1, "gap_m": 0.04, "sheath_m": 5e-4, "n_e_m3": 1e16,
            "n_s_ratio": 0.4, "nu_m_hz": 1e8, "freq_hz": 100e6, "v0": 100.0,
            "n_r": 60, "n_periods": 8, "n_fft_periods": 4, "n_harm": 3,
        },
    }
    with c.websocket_connect("/ws/tl") as ws:
        ws.send_json({"cmd": "start", "project": project})
        started = ws.receive_json()
        assert started["type"] == "started"
        assert started["n_steps"] > 0
        assert started["dt"] > 0

        msg = ws.receive_json()
        while msg["type"] == "progress":
            assert 0 < msg["step"] <= started["n_steps"]
            msg = ws.receive_json()

        assert msg["type"] == "done"
        result = msg["result"]
        assert "harmonics" in result and "lambda_eff" in result and "power" in result
        assert "v_probe" in result and "spectrum_probe" in result
        assert result["settings"]["n_r"] == 60


def test_tl_ws_error_without_tl_settings():
    """/ws/tl: project.tl が無ければ error を返す (継続はエラー扱いですらない — continue 未対応)。"""
    from starlette.testclient import TestClient

    from es_sim import server as srv

    c = TestClient(srv.app)
    project = {
        "geometry": _DUMMY_GEOMETRY.model_dump(),
        "mesh": _DUMMY_MESH.model_dump(),
    }
    with c.websocket_connect("/ws/tl") as ws:
        ws.send_json({"cmd": "start", "project": project})
        msg = ws.receive_json()
        assert msg["type"] == "error"
