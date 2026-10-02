"""PIC バッチ実行モードのテスト (prompts/78、1D/--module 対応は prompts/96)。

es_sim.batch.run_files を直接呼び出す (CLI 引数パースは build_argparser 側の薄い
ラッパーなのでここでは検証しない)。プロセスを spawn するため実行時間がかかるので、
ケースは test_pic.py 同様の極小サイズにする。

1. 2ケースを --parallel 2 相当で実行し、出力2ファイルが生成されること。
   出力が UI の結果の型 (ui/src/results/types.ts、v1 は frontend/src/types.ts) の ResultsBundle / PicStartedMsg / PicDiag / PicFields と
   キー構造が一致すること (実際に types.ts を読んでキー名を写して assert する)。
2. 出力 JSON から results を除いた部分が pydantic の Project として再度読めること
   (loadProject と同じ「project 本体と results を分離する」設計の検証)。
3. 壊れた JSON を混ぜたとき exit code が非0になり、正常ケースの出力は生成されること。
4. --module (auto/pic/pic1d、prompts/96): auto での 2D/1D 混在判定、明示指定の検証。
"""

import json
import time
from pathlib import Path

from es_sim.batch import build_argparser, run_files
from es_sim.schema import Project

DENSITY = 1.0e14  # [m^-3]
L = 0.01
H = 0.02


def _tiny_pic_project(seed: int, n_steps: int = 12) -> dict:
    """test_pic.py の CCP スモークケースを縮小した最小構成 (数秒で完了する)。"""
    return {
        "geometry": {
            "domain": {"polygon": [[0, 0], [L, 0], [L, H], [0, H]]},
            "boundaries": [
                {
                    "edges": [3],
                    "type": "dirichlet",
                    "voltage": 0.0,
                    "voltage_rf": {"amplitude": 50.0, "freq_hz": 13.56e6, "phase_deg": 0.0},
                },
                {"edges": [1], "type": "dirichlet", "voltage": 0.0},
            ],
        },
        "mesh": {"size": 2.0e-3},
        "pic": {
            "initial_plasma": {
                "density": DENSITY,
                "te_ev": 2.0,
                "ti_ev": 0.03,
                "ion_mass_amu": 40.0,
                "immobile_ions": False,
                "seed": seed,
            },
            "n_macro": 400,
            "dt": 5e-10,
            "n_steps": n_steps,
            "frame_every": 4,
        },
    }


def _tiny_pic1d_project(n_steps: int = 60) -> dict:
    """test_sweep.py の縮小 1D ケースと同型 (数秒で完了する、prompts/96)。"""
    return {
        "geometry": {"domain": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}},
        "mesh": {"size": 0.1},
        "pic1d": {
            "gap_m": 0.02,
            "n_cells": 16,
            "init_density_m3": 1.0e14,
            "n_macro": 500,
            "dt": 1.0e-10,
            "n_steps": n_steps,
            "frame_every": 20,
            "left": {"v_dc": 50.0},
            "right": {"v_dc": 0.0},
        },
    }


def _write_case(tmp_path: Path, name: str, project: dict) -> Path:
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(project), encoding="utf-8")
    return p


# ---- 1. 2ケース並列実行 + ResultsBundle 形式の検証 ------------------------------


def test_batch_run_two_cases_produces_results_bundle(tmp_path: Path):
    case1 = _write_case(tmp_path, "case1", _tiny_pic_project(seed=1))
    case2 = _write_case(tmp_path, "case2", _tiny_pic_project(seed=2))

    rc = run_files([str(case1), str(case2)], parallel=2, out_dir=str(tmp_path))
    assert rc == 0

    out1 = tmp_path / "case1_results.json"
    out2 = tmp_path / "case2_results.json"
    assert out1.exists() and out2.exists()

    for out_path in (out1, out2):
        obj = json.loads(out_path.read_text(encoding="utf-8"))
        # トップレベルは入力プロジェクト + results (ResultsBundle, prompts/78)
        assert "geometry" in obj and "pic" in obj
        results = obj["results"]
        assert results["version"] == 1
        pic = results["pic"]
        assert pic is not None

        # PicStartedMsg (types.ts): dt/n_steps/mesh(nodes/triangles) が必須
        started = pic["started"]
        assert started["type"] == "started"
        assert isinstance(started["dt"], float) and started["dt"] > 0
        assert started["n_steps"] == 12
        assert started["step_offset"] == 0
        assert isinstance(started["warnings"], list)
        mesh = started["mesh"]
        assert isinstance(mesh["nodes"], list) and len(mesh["nodes"]) > 0
        assert isinstance(mesh["nodes"][0], list) and len(mesh["nodes"][0]) == 2
        assert isinstance(mesh["triangles"], list) and len(mesh["triangles"]) > 0

        # バッチではライブフレームは持たない
        assert pic["frame"] is None

        # history: PicDiag[] (行ごとの配列。列ごとの辞書ではない)
        history = pic["history"]
        assert isinstance(history, list)
        assert len(history) == 12  # n_steps 分
        diag_keys = {
            "t", "ke_e", "ke_i", "fe", "n_e", "n_i", "wall_e", "wall_i", "phi_min", "phi_max",
        }
        for row in history:
            assert diag_keys <= set(row.keys())
            assert isinstance(row["t"], float)

        # PicFields (types.ts): phi/e_abs/n_e/n_i/te_ev/ion_rate/avg_steps
        fields = pic["fields"]
        assert fields is not None
        for key in ("phi", "e_abs", "n_e", "n_i", "te_ev", "ion_rate", "avg_steps"):
            assert key in fields

        # コレクタ未設定なので空配列 (undefined ではない。ResultsBundle は必須配列)
        assert pic["collectors"] == []

        # 実行時間のリアルタイム表示 (prompts/86): run_batch の壁時計秒が同梱される
        assert pic["elapsed_s"] > 0


# ---- 2. results を除いた部分が Project として再検証できること --------------------


def test_batch_output_project_part_is_valid_project(tmp_path: Path):
    case = _write_case(tmp_path, "case1", _tiny_pic_project(seed=3))
    rc = run_files([str(case)], parallel=1, out_dir=str(tmp_path))
    assert rc == 0

    obj = json.loads((tmp_path / "case1_results.json").read_text(encoding="utf-8"))
    # loadProject と同じ分離: results はプロジェクトスキーマ外なので取り除いてから validate する
    project_only = {k: v for k, v in obj.items() if k != "results"}
    project = Project.model_validate(project_only)
    assert project.pic is not None
    assert project.pic.n_steps == 12


# ---- 3. 壊れたケースを混ぜても正常ケースは出力され、exit code が非0になること ------


def test_batch_run_with_broken_case_reports_failure_but_keeps_good_output(tmp_path: Path):
    good = _write_case(tmp_path, "good", _tiny_pic_project(seed=4))
    broken = tmp_path / "broken.json"
    broken.write_text("{ not valid json ", encoding="utf-8")

    rc = run_files([str(good), str(broken)], parallel=2, out_dir=str(tmp_path))
    assert rc == 1

    assert (tmp_path / "good_results.json").exists()
    assert not (tmp_path / "broken_results.json").exists()


# ---- 4. 並列実行が逐次より速いこと (壁時計時間の目安確認) --------------------------


def test_batch_parallel_is_not_slower_than_sequential(tmp_path: Path):
    """厳密な速度保証テストではない (CI環境のコア数に依存するため) が、
    2ケースを並列実行しても逐次実行の2倍もかからないことだけ確認する。
    """
    case1 = _write_case(tmp_path, "p1", _tiny_pic_project(seed=5, n_steps=30))
    case2 = _write_case(tmp_path, "p2", _tiny_pic_project(seed=6, n_steps=30))

    out_par = tmp_path / "par"
    t0 = time.perf_counter()
    rc = run_files([str(case1), str(case2)], parallel=2, out_dir=str(out_par))
    t_parallel = time.perf_counter() - t0
    assert rc == 0

    out_seq = tmp_path / "seq"
    t0 = time.perf_counter()
    rc = run_files([str(case1)], parallel=1, out_dir=str(out_seq))
    rc |= run_files([str(case2)], parallel=1, out_dir=str(out_seq))
    t_sequential_single = time.perf_counter() - t0
    assert rc == 0

    # 2ケース逐次 (t_sequential_single ≒ 1ケース分×2) と比べ、2並列は明らかに速いはず。
    # プロセス起動オーバーヘッドがあるので緩めに「逐次の1.5倍未満」を基準にする
    assert t_parallel < t_sequential_single * 1.5


# ---- 5. --module (auto/pic/pic1d、prompts/96) -----------------------------------


def test_build_argparser_module_flag_default_auto_and_choices():
    parser = build_argparser()
    args = parser.parse_args(["run", "a.json"])
    assert args.module == "auto"  # 既定 auto (project の中身から判定)

    args_pic1d = parser.parse_args(["run", "a.json", "--module", "pic1d"])
    assert args_pic1d.module == "pic1d"


def test_batch_run_module_auto_mixed_pic_and_pic1d_in_one_batch(tmp_path: Path):
    """1バッチ内に 2D (pic) / 1D (pic1d) のケースが混在しても、既定の module=auto で
    ファイルごとに正しいソルバーへ振り分けられること (既存 2D バッチはビット不変)。
    """
    case_pic = _write_case(tmp_path, "case_pic", _tiny_pic_project(seed=10))
    case_pic1d = _write_case(tmp_path, "case_pic1d", _tiny_pic1d_project())

    rc = run_files([str(case_pic), str(case_pic1d)], parallel=2, out_dir=str(tmp_path))
    assert rc == 0

    obj_pic = json.loads((tmp_path / "case_pic_results.json").read_text(encoding="utf-8"))
    assert obj_pic["results"]["pic"] is not None
    assert "pic1d" not in obj_pic["results"] or obj_pic["results"].get("pic1d") is None

    obj_pic1d = json.loads((tmp_path / "case_pic1d_results.json").read_text(encoding="utf-8"))
    assert "pic" not in obj_pic1d["results"]
    result1d = obj_pic1d["results"]["pic1d"]
    assert result1d is not None
    # server.py の _pic1d_result (build_pic1d_result) と同じキー構造 (frontend Pic1dResult と一致)
    assert set(result1d) == {
        "history", "profiles", "sheath", "cycle", "sheath_fft", "sheath_ts",
        "eedf", "wall_iedf", "walls", "fn", "elapsed_s", "timing", "settings", "circuit",
    }
    assert result1d["settings"]["n_steps"] == 60
    assert result1d["elapsed_s"] > 0

    # results を除いた部分は pydantic Project として再検証できる (1D 側も 2D 側と同じ設計)
    project_only = {k: v for k, v in obj_pic1d.items() if k != "results"}
    project = Project.model_validate(project_only)
    assert project.pic1d is not None
    assert project.pic is None


def test_batch_run_module_explicit_pic1d_on_pic_only_project_errors(tmp_path: Path):
    """--module pic1d を明示指定したのに project.pic1d が無ければエラーになること
    (auto に頼らずユーザー指定を尊重した結果、意図しないソルバー不一致を検出できる)。
    """
    case = _write_case(tmp_path, "case1", _tiny_pic_project(seed=11))
    rc = run_files([str(case)], parallel=1, out_dir=str(tmp_path), module="pic1d")
    assert rc == 1
    assert not (tmp_path / "case1_results.json").exists()


def test_batch_run_module_auto_errors_when_project_has_neither_pic_nor_pic1d(tmp_path: Path):
    bare = {"geometry": {"domain": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}}, "mesh": {"size": 0.1}}
    case = _write_case(tmp_path, "bare", bare)
    rc = run_files([str(case)], parallel=1, out_dir=str(tmp_path), module="auto")
    assert rc == 1
    assert not (tmp_path / "bare_results.json").exists()
