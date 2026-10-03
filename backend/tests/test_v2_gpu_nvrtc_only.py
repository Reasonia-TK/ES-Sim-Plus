"""GPU は CUDA Toolkit の NVRTC だけで動く (配布版は NVRTC だけを同梱する、prompts/133 P8a)。

- 密な逆行列 (``device.linalg.spd_inverse``、ブロック Gauss-Jordan) が numpy と一致し、正定値でない・対称で
  ない行列は警告して CPU で求め直す。GMG の特異な問題の擬似逆行列も CPU 版と一致する。
- ``cuda_status()`` の条件 (ドライバ・Compute Capability・カーネルの実行) と理由 (偽の cupy で)。
- NVRTC の 2 つの DLL だけを置いた CUDA_PATH (PATH から CUDA Toolkit を除く) で GPU のテストを走らせ、cuBLAS・
  cuSOLVER・cuSPARSE などが読まれないこと (Windows + GPU の開発機だけ。CI は GPU が無いので飛ばす)。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from es_sim import device as device_mod
from es_sim.device import cuda_available

needs_cuda = pytest.mark.skipif(not cuda_available(), reason="CUDA (CuPy) が使えない環境")

BACKEND = Path(__file__).resolve().parents[1]


def _spd(n: int, seed: int = 0) -> np.ndarray:
    q = np.random.default_rng(seed).standard_normal((n, n))
    return q @ q.T + n * np.eye(n)


def _neumann_laplacian(nx: int, ny: int) -> np.ndarray:
    """nx × ny 格子のグラフラプラシアン (全周 Neumann、零空間 = 定数)。"""
    n = nx * ny
    a = np.zeros((n, n))
    for j in range(ny):
        for i in range(nx):
            p = j * nx + i
            for q in ((p + 1) if i + 1 < nx else None, (p + nx) if j + 1 < ny else None):
                if q is not None:
                    a[p, p] += 1.0
                    a[q, q] += 1.0
                    a[p, q] -= 1.0
                    a[q, p] -= 1.0
    return a


@needs_cuda
@pytest.mark.parametrize("n", [1, 2, 31, 32, 33, 100, 700])
def test_spd_inverse_matches_numpy(n):
    from es_sim.device.linalg import spd_inverse

    a = _spd(n, seed=n)
    inv = spd_inverse(a).get()
    ref = np.linalg.inv(a)
    assert np.max(np.abs(inv - ref)) <= 1e-12 * np.max(np.abs(ref))


@needs_cuda
def test_dense_inverse_of_singular_problem_matches_cpu():
    import cupy as cp

    from es_sim.field.gmg import dense_inverse

    a = _neumann_laplacian(9, 7)
    gpu = dense_inverse(a, True, xp=cp).get()
    cpu = dense_inverse(a, True)
    assert np.max(np.abs(gpu - cpu)) <= 1e-10 * np.max(np.abs(cpu))
    assert np.max(np.abs(a @ gpu @ a - a)) <= 1e-10 * np.max(np.abs(a))


@needs_cuda
def test_spd_inverse_falls_back_to_cpu_when_not_spd():
    from es_sim.device.linalg import spd_inverse

    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.warns(RuntimeWarning, match="正定値でない"):
        inv = spd_inverse(indefinite).get()
    np.testing.assert_allclose(inv, np.linalg.inv(indefinite), rtol=1e-14)
    asym = np.array([[2.0, 1.0], [0.0, 2.0]])
    with pytest.warns(RuntimeWarning, match="対称でない"):
        inv = spd_inverse(asym).get()
    np.testing.assert_allclose(inv, np.linalg.inv(asym), rtol=1e-14)


class _NoDevice(RuntimeError):
    status = 100  # cudaErrorNoDevice (CuPy の CUDARuntimeError と同じ属性)


def _fake_cupy(driver=13010, runtime=13000, count=1, cc=(8, 6), kernel_error=None):
    def arange(n, dtype=None):
        if kernel_error:
            raise kernel_error
        return np.arange(n, dtype=dtype)

    def device_count():
        if count is None:
            raise _NoDevice("cudaErrorNoDevice: no CUDA-capable device is detected")
        return count

    rt = types.SimpleNamespace(
        driverGetVersion=lambda: driver,
        runtimeGetVersion=lambda: runtime,
        getDeviceCount=device_count,
        getDeviceProperties=lambda i: {"name": b"Fake GPU", "major": cc[0], "minor": cc[1]},
    )
    return types.SimpleNamespace(cuda=types.SimpleNamespace(runtime=rt), arange=arange, float64=np.float64)


@pytest.mark.parametrize(
    ("kw", "ok", "pattern"),
    [
        ({}, True, "Fake GPU"),
        ({"driver": 12040}, False, r"ドライバが古い.*CUDA 12\.4 まで.*CUDA 13 に対応したドライバ \(R580 以降\)"),
        ({"driver": 0}, False, "ドライバが見つかりません"),
        ({"count": 0}, False, "NVIDIA の GPU が見つかりません$"),
        ({"count": None}, False, "NVIDIA の GPU が見つかりません$"),
        ({"cc": (6, 1)}, False, r"Compute Capability 6\.1.*対象外"),
        ({"kernel_error": RuntimeError("nvrtc64_130_0.dll not found")}, False, "カーネルを実行できません.*nvrtc"),
    ],
)
def test_cuda_status_conditions(monkeypatch, kw, ok, pattern):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(**kw))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device_mod.cuda_status.cache_clear()
    try:
        got_ok, info = device_mod.cuda_status()
    finally:
        device_mod.cuda_status.cache_clear()
    assert got_ok is ok
    assert re.search(pattern, info), info


def test_cuda_status_mentions_hidden_devices(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(count=None))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    device_mod.cuda_status.cache_clear()
    try:
        ok, info = device_mod.cuda_status()
    finally:
        device_mod.cuda_status.cache_clear()
    assert not ok and info == "NVIDIA の GPU が見つかりません (CUDA_VISIBLE_DEVICES=-1 で隠されています)"


# ---- NVRTC だけの環境 ------------------------------------------------------------------------------

#: NVRTC だけの環境で走らせる GPU のテスト (P8a で置き換えた 4 か所と、PIC・DSMC・流体の基本、流体の阻止コンデンサ)
_GPU_TESTS = [
    "tests/test_v2_gpu_nvrtc_only.py::test_spd_inverse_matches_numpy",
    "tests/test_v2_gpu_nvrtc_only.py::test_dense_inverse_of_singular_problem_matches_cpu",
    "tests/test_v2_eb_poisson.py::test_gpu_matches_cpu",
    "tests/test_v2_amr_pic.py::test_gpu_amg_pcg_matches_cpu_and_async_warm_start_converges",
    "tests/test_v2_amr_pic.py::test_gpu_amg_pcg_singular_periodic_problem",
    "tests/test_v2_gpu_pic.py::test_plasma_oscillation_frequency",
    "tests/test_v2_gdsmc.py::test_equilibrium_box",
    "tests/test_v2_gfluid_gpu.py::test_amr_matches_cpu_direct_poisson",
    "tests/test_v2_gfluid_gpu.py::test_amr_singular_poisson_on_gpu",
    "tests/test_circuit_fluid2d.py::test_gpu_matches_cpu_with_two_capacitors",
    "tests/test_circuit_fluid2d.py::test_gpu_amr_matches_cpu_amr_with_capacitors",
    "tests/test_circuit_gpic.py::test_charge_conservation_gpu[uniform]",
    "tests/test_circuit_gpic.py::test_charge_conservation_gpu[amr]",
]

#: 配布版に入れない CUDA のライブラリ (読まれたら、配布版では GPU の計算が失敗する)
_FORBIDDEN = re.compile(r"^(cublas|cublaslt|cusolver|cusparse|cufft|curand|nvjitlink|cutensor|npp)[^\\/]*\.dll$", re.I)


def _has_cuda_dlls(d: Path) -> bool:
    return any(any(p.glob(pat)) for p in (d, d / "x64") if p.is_dir() for pat in ("nvrtc64_*.dll", "cublas64_*.dll"))


@pytest.mark.skipif(sys.platform != "win32", reason="DLL の列挙は Windows だけ")
@needs_cuda
def test_gpu_solvers_run_with_only_nvrtc(tmp_path):
    from cuda.pathfinder import load_nvidia_dynamic_lib

    nvrtc = Path(load_nvidia_dynamic_lib("nvrtc").abs_path)
    builtins = sorted(nvrtc.parent.glob("nvrtc-builtins64_*.dll"))
    assert builtins, f"nvrtc-builtins が {nvrtc.parent} にありません"
    cuda = tmp_path / "cuda"
    dst = cuda / "bin" / "x64"
    dst.mkdir(parents=True)
    for f in (nvrtc, builtins[-1]):
        shutil.copy2(f, dst / f.name)

    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("CUDA_PATH", "CUDA_HOME"))}
    env["CUDA_PATH"] = str(cuda)
    env["PATH"] = os.pathsep.join(
        p for p in os.environ.get("PATH", "").split(os.pathsep) if p and not _has_cuda_dlls(Path(p))
    )
    env["PYTHONPATH"] = os.pathsep.join([str(BACKEND / "tests"), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    report = tmp_path / "dlls.json"
    env["ES_SIM_DLL_REPORT"] = str(report)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "cuda_dll_guard", "-p", "no:cacheprovider", *_GPU_TESTS],
        cwd=BACKEND, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1200,
    )
    tail = (proc.stdout + proc.stderr)[-4000:]
    assert proc.returncode == 0, tail
    assert re.search(rf"\b{len(_GPU_TESTS) + 6} passed", proc.stdout), tail  # spd_inverse はパラメータ 7 通り
    mods = json.loads(report.read_text(encoding="utf-8"))["modules"]
    names = [Path(m).name for m in mods]
    assert not [n for n in names if _FORBIDDEN.match(n)], mods
    # NVRTC は CUDA Toolkit からではなく、NVRTC だけのフォルダ (CUDA_PATH か、venv にある nvidia-cuda-nvrtc の wheel) から
    loaded_nvrtc = [Path(m) for m in mods if Path(m).name.lower().startswith("nvrtc64_")]
    assert loaded_nvrtc and not any(_has_cuda_dlls(m.parent) and any(m.parent.glob("cublas64_*.dll"))
                                    for m in loaded_nvrtc), mods
