"""配布版の自己テスト (``es-sim-backend selftest``、prompts/133 P8b)。

凍結したバックエンド (PyInstaller) で、同梱が欠けやすいもの (gmsh の DLL・numba と llvmlite・CuPy と NVRTC・
CUDA カーネルのソース・pyamg・ezdxf・boltzpmp) を、実際に使う小さな計算で確かめる。結果は既知の値 (解析解・
CPU の計算) と比べる。GPU の確認は CUDA が使えるときだけ行い、使えなければ理由を付けて飛ばす
(``require_gpu`` なら失敗にする)。開発環境でもそのまま動く (tests/test_selftest.py)。
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


@dataclass
class CheckResult:
    name: str
    status: str  # "ok" | "fail" | "skip"
    detail: str
    seconds: float


class _Skip(Exception):
    """この確認を飛ばす (GPU が無いなど)。"""


def _project(d: dict):
    from .schema import Project

    return Project.model_validate(d)


_PLATES = [[0, 0], [0.1, 0], [0.1, 0.05], [0, 0.05]]
_LR = [{"edges": [3], "type": "dirichlet", "voltage": 0.0}, {"edges": [1], "type": "dirichlet", "voltage": 100.0}]


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


# ---- CPU ---------------------------------------------------------------------------------------


def _environment() -> str:
    import scipy

    from . import __version__

    parts = [f"es_sim {__version__}", f"Python {platform.python_version()}", f"numpy {np.__version__}",
             f"scipy {scipy.__version__}"]
    if getattr(sys, "frozen", False):
        parts.append("凍結 (PyInstaller)")
    return "、".join(parts)


def _numba() -> str:
    import numba

    from . import _numba_kernels

    _check(_numba_kernels.HAVE_NUMBA, "es_sim の numba カーネルを読み込めません (numpy の遅い実装で動きます)")
    f = numba.njit(lambda a: 2.0 * a + 1.0)
    _check(f(1.0) == 3.0, "JIT でコンパイルした関数の結果が違います")
    return f"numba {numba.__version__}: JIT のコンパイルと実行"


def _fem() -> str:
    import gmsh

    from .fem import EPS0, solve
    from .meshing import generate_mesh

    p = _project({"geometry": {"domain": {"polygon": _PLATES},
                               "boundaries": [{"edges": [3], "voltage": 0.0}, {"edges": [1], "voltage": 100.0}]},
                  "mesh": {"size": 0.01}})
    sol = solve(p, generate_mesh(p))
    exact = EPS0 * 0.05 / 0.1
    err = abs(sol.capacitance - exact) / exact
    _check(err < 1e-8, f"平行平板の静電容量が解析解と {err:.1e} ずれています")
    return f"gmsh {getattr(gmsh, '__version__', '?')}: 平行平板の静電容量 (解析解との差 {err:.0e})"


def _amr_project():
    return _project({"geometry": {"domain": {"polygon": _PLATES}, "boundaries": _LR},
                     "mesh": {"size": 0.001, "mode": "cartesian",
                              "amr": {"max_level": 2,
                                      "regions": [{"p1": [0.03, 0.01], "p2": [0.06, 0.03], "level": 2}]}}})


def _amr_linear_error(s) -> float:
    """平行平板 (左 0 V・右 100 V、幅 0.1 m) の電位は φ = 1000·x で、AMR の粗細界面でも厳密。"""
    return float(np.max(np.abs(s.phi - 1000.0 * s.op.xy[:, 0])))


def _amr_cpu() -> str:
    import pyamg

    from .amr import solve_electrostatic_amr

    s = solve_electrostatic_amr(_amr_project(), device="cpu")
    err = _amr_linear_error(s)
    _check(s.info.converged and err < 1e-6, f"平行平板の電位が {err:.1e} V ずれています (収束 {s.info.converged})")
    return f"pyamg {pyamg.__version__}: 未知数 {s.op.n_unknowns}、φ = 1000·x との差 {err:.0e} V"


def _dxf() -> str:
    import ezdxf

    from .dxf import read_dxf, write_dxf

    doc = {"geometry": {"domain": {"polygon": _PLATES},
                        "regions": [{"id": "c", "type": "conductor", "voltage": 1.0,
                                     "shape": {"kind": "circle", "center": [0.05, 0.025], "radius": 0.01}}]}}
    text = write_dxf(doc, "mm")
    back = read_dxf(text.encode("utf-8"))
    kinds = sorted(e["kind"] for e in back["entities"])
    _check(kinds == ["circle", "polyline"], f"書き出した DXF を読み直した形が違います: {kinds}")
    r = next(e["r"] for e in back["entities"] if e["kind"] == "circle")
    _check(abs(r - 0.01) < 1e-12, f"円の半径が違います: {r}")
    return f"ezdxf {ezdxf.__version__}: 書き出しと読み直し"


def _boltz() -> str:
    from .boltz import boltzpm_available, run_boltz_sweep
    from .pic1d_presets import edupic_ar_processes

    _check(boltzpm_available(), "boltzpmp を import できません")
    electrons, _ = edupic_ar_processes()
    t = run_boltz_sweep(electrons, 39.948, 50.0, 300.0,
                        {"en_min_td": 10.0, "en_max_td": 100.0, "n_points": 2, "d_eps_ev": 0.5, "n_theta": 8})
    e = t["mean_energy_ev"]
    _check(len(e) == 2 and all(0.0 < x < 100.0 for x in e), f"平均エネルギーがおかしい: {e}")
    return f"Ar (eduPIC) の 10・100 Td: 平均エネルギー {e[0]:.2f}・{e[1]:.2f} eV"


# ---- GPU ---------------------------------------------------------------------------------------


def _need_gpu() -> None:
    from .device import cuda_status

    ok, info = cuda_status()
    if not ok:
        raise _Skip(info)


def _nvrtc_dir() -> str:
    """読み込んだ NVRTC のフォルダ (配布版は同梱の _internal/cuda/bin/x64)。"""
    import os

    try:
        from cuda.pathfinder import load_nvidia_dynamic_lib

        return os.path.dirname(load_nvidia_dynamic_lib("nvrtc").abs_path)
    except Exception:  # noqa: BLE001 - 表示だけ
        return "?"


def _gpu_device() -> str:
    from .device import describe

    _need_gpu()
    d = describe()
    drv = int(d.get("cuda_driver", 0))
    return (f"{d['cuda_info']} (Compute Capability {d.get('compute_capability', '?')})、ドライバ CUDA "
            f"{drv // 1000}.{(drv % 1000) // 10}、CuPy {d.get('cupy', '?')}、NVRTC {d.get('nvrtc', '?')} ({_nvrtc_dir()})")


def _gpu_static() -> str:
    from .field import solve_electrostatic

    _need_gpu()
    p = _project({"geometry": {"domain": {"polygon": [[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]]},
                               "regions": [{"id": "inner", "type": "conductor", "voltage": 1.0,
                                            "shape": {"kind": "circle", "center": [0.0, 0.0], "radius": 0.01}}],
                               "boundaries": [{"edges": [0, 1, 2, 3], "type": "dirichlet", "voltage": 0.0}]},
                  "mesh": {"size": 0.1 / 128, "mode": "cartesian"}})
    cpu = solve_electrostatic(p, device="cpu", tol=1e-12)
    gpu = solve_electrostatic(p, device="cuda", tol=1e-12)
    _check(gpu.device == "cuda", f"GPU で解いていません ({gpu.device})")
    d = float(np.max(np.abs(gpu.phi - cpu.phi)))
    _check(d < 1e-9 and gpu.info.converged, f"CPU との差が {d:.1e} V あります (収束 {gpu.info.converged})")
    return f"GMG-PCG {gpu.grid.nx}×{gpu.grid.ny}: {gpu.info.iterations} 反復、CPU との差 {d:.0e} V"


def _gpu_amr() -> str:
    from .amr import solve_electrostatic_amr

    _need_gpu()
    s = solve_electrostatic_amr(_amr_project(), device="cuda")
    err = _amr_linear_error(s)
    _check(s.info.converged and err < 1e-6, f"平行平板の電位が {err:.1e} V ずれています (収束 {s.info.converged})")
    return f"AMG-PCG: 未知数 {s.op.n_unknowns}、{s.info.iterations} 反復、φ = 1000·x との差 {err:.0e} V"


def _gpu_pic() -> str:
    """冷たいプラズマの振動 (運動エネルギーは 2·f_pe で振動する、tests/test_v2_gpu_pic.py と同じ設定)。"""
    from .fem import EPS0
    from .gpic import GpuPicSimulation
    from .particles import ME, QE

    _need_gpu()
    n0, lx = 1.0e14, 0.01
    sim = GpuPicSimulation(_project({
        "geometry": {"domain": {"polygon": [[0, 0], [lx, 0], [lx, 0.03], [0, 0.03]]},
                     "boundaries": [{"edges": [0, 1, 2, 3], "type": "dirichlet", "voltage": 0.0}]},
        "mesh": {"size": 8e-4, "mode": "cartesian"},
        "pic": {"initial_plasma": {"density": n0, "te_ev": 0.0, "ti_ev": 0.0, "ion_mass_amu": 40.0,
                                   "immobile_ions": True, "seed": 1},
                "n_macro": 60000, "dt": None, "n_steps": 400, "frame_every": 1000},
    }))
    e = sim.get_particles("electron")
    sim.set_particles("electron", vx=e["vx"] + 3.0e3 * np.sin(2.0 * np.pi * e["x"] / lx))
    hist, _ = sim.run_batch(store_frames=False)
    ke = np.asarray(hist["ke_e"]) - np.mean(hist["ke_e"])
    up = np.nonzero((ke[:-1] < 0) & (ke[1:] >= 0))[0]
    _check(len(up) >= 3, "運動エネルギーが振動していません")
    tz = up + ke[up] / (ke[up] - ke[up + 1])
    f = (len(tz) - 1) / ((tz[-1] - tz[0]) * sim.dt)
    f_exp = 2.0 * math.sqrt(n0 * QE**2 / (EPS0 * ME)) / (2.0 * math.pi)
    rel = abs(f - f_exp) / f_exp
    _check(rel < 0.05, f"振動数が 2·f_pe と {rel:.1%} ずれています")
    return f"プラズマ振動 6 万粒子 400 ステップ: 2·f_pe との差 {rel:.1%}"


def _gpu_dsmc() -> str:
    """閉じた箱の平衡 (密度・温度が初期値のまま)。"""
    from .gdsmc import GpuDsmcSimulation

    _need_gpu()
    p0, t0 = 10.0, 300.0
    sim = GpuDsmcSimulation(_project({
        "geometry": {"domain": {"polygon": [[0, 0], [0.02, 0], [0.02, 0.01], [0, 0.01]]}, "boundaries": []},
        "mesh": {"size": 1.5e-3, "mode": "cartesian"},
        "dsmc": {"init_pressure_pa": p0, "init_temperature_k": t0, "wall_temperature_k": t0,
                 "n_particles": 20000, "n_steps": 300, "avg_steps": 150, "seed": 1},
    }))
    res = sim.run()
    a = sim.area
    n = float(np.sum(res.n * a) / a.sum())
    t = float(np.sum(res.t * res.n * a) / np.sum(res.n * a))
    n_rel = abs(n / (p0 / (1.380649e-23 * t0)) - 1.0)
    t_rel = abs(t / t0 - 1.0)
    _check(n_rel < 0.05 and t_rel < 0.05 and res.n_particles == 20000,
           f"平衡がずれています (密度 {n_rel:.1%}、温度 {t_rel:.1%}、分子 {res.n_particles})")
    return f"閉じた箱の平衡 2 万分子: 密度の差 {n_rel:.1%}、温度の差 {t_rel:.1%}"


def _gpu_fluid() -> str:
    """固体の無い CCP を 20 ステップ (GPU と CPU の結果が一致する)。"""
    from .gfluid import CartesianFluid2dSimulation, GpuCartesianFluid2dSimulation

    _need_gpu()
    p = _project({
        "geometry": {"domain": {"polygon": [[0, 0], [0.02, 0], [0.02, 0.01], [0, 0.01]]},
                     "boundaries": [
                         {"edges": [3], "type": "dirichlet", "voltage": 0.0,
                          "voltage_rf": {"amplitude": 100.0, "freq_hz": 13.56e6}},
                         {"edges": [1], "type": "dirichlet", "voltage": 0.0},
                         {"edges": [0, 2], "type": "symmetry"}]},
        "mesh": {"size": 0.5e-3, "mode": "cartesian"},
        "fluid2d": {"init_density_m3": 5e14, "init_te_ev": 3.0, "gas_pressure_pa": 30.0, "n_steps": 20,
                    "frame_every": 10**9},
    })
    gpu = GpuCartesianFluid2dSimulation(p.model_copy(deep=True))
    gpu.run_batch(store_frames=False)
    cpu = CartesianFluid2dSimulation(p.model_copy(deep=True), device="cpu")
    cpu.run_batch(store_frames=False)
    rel = max(float(np.max(np.abs(np.asarray(getattr(gpu, k)) - np.asarray(getattr(cpu, k))))
                    / max(float(np.max(np.abs(np.asarray(getattr(cpu, k))))), 1e-300))
              for k in ("n_e", "n_i", "phi"))
    _check(np.all(np.isfinite(gpu.n_e)) and rel < 1e-9, f"CPU との差が {rel:.1e} あります")
    return f"CCP 20 ステップ: CPU との差 {rel:.0e} (相対)"


#: (名前, 関数, GPU の確認か)
CHECKS: list[tuple[str, Callable[[], str], bool]] = [
    ("環境", _environment, False),
    ("numba", _numba, False),
    ("gmsh と FEM", _fem, False),
    ("AMR の静電場 (CPU)", _amr_cpu, False),
    ("DXF", _dxf, False),
    ("Boltzmann", _boltz, False),
    ("GPU", _gpu_device, True),
    ("GPU の静電場", _gpu_static, True),
    ("GPU の AMR の静電場", _gpu_amr, True),
    ("GPU の PIC", _gpu_pic, True),
    ("GPU の DSMC", _gpu_dsmc, True),
    ("GPU の流体", _gpu_fluid, True),
]


def run_checks(*, require_gpu: bool = False, gpu: bool = True, echo: Callable[[str], None] | None = None,
               names: list[str] | None = None) -> list[CheckResult]:
    """確認を順に行う。gpu=False なら GPU の確認を飛ばす。echo があれば 1 行ずつ知らせる。"""
    out: list[CheckResult] = []
    for name, fn, is_gpu in CHECKS:
        if names is not None and name not in names:
            continue
        t0 = time.perf_counter()
        try:
            if is_gpu and not gpu:
                raise _Skip("GPU の確認をしない指定")
            r = CheckResult(name, "ok", fn(), 0.0)
        except _Skip as exc:
            r = CheckResult(name, "fail" if require_gpu and is_gpu else "skip", str(exc), 0.0)
        except Exception as exc:  # noqa: BLE001 - 失敗の内容を知らせて次の確認へ
            tb = traceback.format_exception_only(type(exc), exc)[-1].strip()
            r = CheckResult(name, "fail", tb, 0.0)
        r.seconds = time.perf_counter() - t0
        out.append(r)
        if echo is not None:
            echo(f"[{r.status.upper():4s}] {r.name}: {r.detail} ({r.seconds:.1f} s)")
    return out


def main(argv: list[str] | None = None) -> int:
    """``es-sim-backend selftest [--require-gpu] [--no-gpu] [--json PATH]``。失敗が無ければ 0。"""
    import argparse

    ap = argparse.ArgumentParser(prog="es-sim-backend selftest", description="配布版の自己テスト")
    ap.add_argument("--require-gpu", action="store_true", help="GPU を使えなければ失敗にする")
    ap.add_argument("--no-gpu", action="store_true", help="GPU の確認をしない")
    ap.add_argument("--json", default=None, help="結果を JSON で書き出すファイル")
    args = ap.parse_args(argv)

    def echo(line: str) -> None:
        print(line, flush=True)

    t0 = time.perf_counter()
    results = run_checks(require_gpu=args.require_gpu, gpu=not args.no_gpu, echo=echo)
    n_fail = sum(r.status == "fail" for r in results)
    echo(f"{'失敗 ' + str(n_fail) + ' 件' if n_fail else 'すべて通りました'} ({time.perf_counter() - t0:.1f} s)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"ok": n_fail == 0, "results": [asdict(r) for r in results]}, f, ensure_ascii=False, indent=1)
    return 1 if n_fail else 0


__all__ = ["CHECKS", "CheckResult", "main", "run_checks"]
