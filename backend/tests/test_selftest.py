"""配布版の自己テスト (es_sim.selftest、``es-sim-backend selftest``、prompts/133 P8b) が開発環境でも通る。

凍結したバックエンドでの確認は scripts/build_backend.ps1 が行う (CUDA の環境変数なしで ``selftest --require-gpu``)。
"""

from __future__ import annotations

import json

import pytest

from es_sim import device as device_mod
from es_sim.device import cuda_available
from es_sim.selftest import CHECKS, main, run_checks

needs_cuda = pytest.mark.skipif(not cuda_available(), reason="CUDA (CuPy) が使えない環境")
GPU_CHECKS = {name for name, _, is_gpu in CHECKS if is_gpu}


def test_cpu_checks_pass_and_gpu_checks_are_skipped_on_request():
    res = run_checks(gpu=False)
    assert [r for r in res if r.status == "fail"] == []
    assert {r.name for r in res if r.status == "skip"} == GPU_CHECKS
    assert all(r.detail for r in res)


@needs_cuda
def test_gpu_checks_pass():
    res = run_checks(require_gpu=True, names=sorted(GPU_CHECKS))
    assert [(r.name, r.status, r.detail) for r in res if r.status != "ok"] == []


def test_missing_gpu_is_a_failure_only_when_required(monkeypatch, tmp_path):
    monkeypatch.setattr(device_mod, "cuda_status", lambda: (False, "テスト用: GPU なし"))
    (r,) = run_checks(names=["GPU"])
    assert (r.status, r.detail) == ("skip", "テスト用: GPU なし")
    (r,) = run_checks(require_gpu=True, names=["GPU"])
    assert r.status == "fail"
    out = tmp_path / "selftest.json"
    lines: list[str] = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: lines.append(" ".join(map(str, a))))
    monkeypatch.setattr("es_sim.selftest.CHECKS", [c for c in CHECKS if c[0] in ("環境", "GPU")])
    assert main(["--json", str(out)]) == 0
    assert main(["--require-gpu", "--json", str(out)]) == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["ok"] is False and [r["status"] for r in data["results"]] == ["ok", "fail"]
    assert any("[FAIL] GPU" in line for line in lines)
