"""v1 (FEM/CPU) と v2 (直交格子 EB/GPU) の同条件ベンチマーク (prompts/119)。

使い方:
    .venv\\Scripts\\python benchmarks\\v2_bench.py pic --n-macro 200000 --steps 2000
    .venv\\Scripts\\python benchmarks\\v2_bench.py poisson
    .venv\\Scripts\\python benchmarks\\v2_bench.py dsmc
    .venv\\Scripts\\python benchmarks\\v2_bench.py fluid2d --size 1e-4

pic:     eduPIC Ar の CCP ストリップ (上下反射で 1D 相当、RF 250 V / 13.56 MHz / 10 Pa)。
         v1 (構造格子 + numba) と v2 (GPU) を同じ粒子数・ステップ数で回し ms/step を比べる。
         v1 は --v1-steps で短く打ち切れる (ms/step は定常なので外挿でよい)。
poisson: 同軸円筒の静電場を v1 FEM (gmsh + splu) と v2 (EB + GMG-PCG、CPU/GPU) で解き、
         容量誤差と求解時間を比べる。
dsmc:    50×20 mm の圧力駆動チャネル (20 → 5 Pa、円柱障害物) を v1 (構造格子 + numba) と
         v2 GPU DSMC で回す (prompts/124)。
fluid2d: 20×10 mm の CCP (RF 100 V、円形導体ピン + 誘電体ブロック) を v1 (非構造) と
         v2 (直交格子 + EB) で回す (prompts/125)。サブステップ数も表示する。
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from es_sim.schema import Project


def _ccp_project(mode: str, n_macro: int, steps: int, cells: int) -> Project:
    from es_sim.pic1d_presets import edupic_ar_processes

    e_procs, i_procs = edupic_ar_processes()
    L, H = 0.025, 0.002
    return Project.model_validate({
        "geometry": {
            "domain": {"polygon": [[0, 0], [L, 0], [L, H], [0, H]]},
            "boundaries": [
                {"edges": [3], "type": "dirichlet", "voltage": 0.0,
                 "voltage_rf": {"amplitude": 250.0, "freq_hz": 13.56e6, "phase_deg": 0.0}},
                {"edges": [1], "type": "dirichlet", "voltage": 0.0},
            ],
        },
        "mesh": {"size": L / cells, "mode": mode},
        "pic": {
            "initial_plasma": {"density": 1e15, "te_ev": 2.0, "ti_ev": 0.026, "ion_mass_amu": 39.948, "seed": 1},
            "n_macro": n_macro,
            "dt": 1.0 / (13.56e6 * 2000),
            "n_steps": steps,
            "frame_every": 10**9,
            "phase_bins": 0,
            "reflect_edges": [0, 2],
            "mcc": {"gas": {"name": "Ar", "pressure_pa": 10.0, "temperature_k": 300.0},
                    "electron_processes": [p.model_dump() for p in e_procs],
                    "ion_processes": [p.model_dump() for p in i_procs], "seed": 3},
        },
    })


def bench_pic(args) -> None:
    from es_sim.gpic import GpuPicSimulation
    from es_sim.pic import PicSimulation

    print(f"CCP strip: n_macro={args.n_macro}/species, cells={args.cells}, steps v2={args.steps} v1={args.v1_steps}")
    p2 = _ccp_project("cartesian", args.n_macro, args.steps, args.cells)
    t = time.perf_counter()
    v2 = GpuPicSimulation(p2)
    setup2 = time.perf_counter() - t
    t = time.perf_counter()
    v2.run_batch(store_frames=False)
    run2 = time.perf_counter() - t
    print(f"  v2 GPU : setup {setup2:6.2f} s, {run2 / args.steps * 1e3:8.3f} ms/step "
          f"(grid {v2.grid.nx}x{v2.grid.ny}, {'direct' if v2.solver.direct else 'GMG-PCG'})")
    if args.v1_steps > 0:
        p1 = _ccp_project("structured", args.n_macro, args.v1_steps, args.cells)
        t = time.perf_counter()
        v1 = PicSimulation(p1)
        setup1 = time.perf_counter() - t
        t = time.perf_counter()
        v1.run_batch(store_frames=False)
        run1 = time.perf_counter() - t
        ms1 = run1 / args.v1_steps * 1e3
        print(f"  v1 CPU : setup {setup1:6.2f} s, {ms1:8.3f} ms/step (threads {v1.effective_threads})")
        print(f"  speedup: {ms1 / (run2 / args.steps * 1e3):.1f}x")


def _coax(h: float) -> Project:
    a, b, L = 0.01, 0.04, 0.05
    corners = [(1.2 * L, 1.2 * L), (-1.2 * L, 1.2 * L), (-1.2 * L, -1.2 * L), (1.2 * L, -1.2 * L)]
    regions = [{"id": "inner", "type": "conductor", "voltage": 1.0,
                "shape": {"kind": "circle", "center": [0.0, 0.0], "radius": a}}]
    for q in range(4):
        a0 = math.pi / 4 + q * math.pi / 2
        th = np.linspace(a0, a0 + math.pi / 2, 721)
        arc = [(b * math.cos(t), b * math.sin(t)) for t in th]
        regions.append({"id": f"outer{q}", "type": "conductor", "voltage": 0.0,
                        "polygon": arc + [corners[(q + 1) % 4], corners[q]]})
    return Project.model_validate({
        "geometry": {"domain": {"polygon": [[-L, -L], [L, -L], [L, L], [-L, L]]}, "regions": regions},
        "mesh": {"size": h},
    })


def bench_poisson(args) -> None:
    from es_sim.device import cuda_available
    from es_sim.fem import solve as fem_solve
    from es_sim.field import solve_electrostatic
    from es_sim.meshing import generate_mesh

    eps0 = 8.8541878128e-12
    c_exact = 2 * math.pi * eps0 / math.log(4.0)
    print("coax (a=10 mm, b=40 mm): capacitance error and solve time")
    for n in (64, 128, 256, 512, 1024):
        h = 0.1 / n
        p = _coax(h)
        row = [f"  h=L/{n:<5d}"]
        if n <= 256:
            t = time.perf_counter()
            mesh = generate_mesh(p)
            sol = fem_solve(p, mesh)
            dt1 = time.perf_counter() - t
            row.append(f"v1 FEM: {len(mesh.nodes):7d} nodes err {(sol.capacitance - c_exact) / c_exact:+.2e} {dt1:6.2f} s")
        for dev in ("cpu", "cuda"):
            if dev == "cuda" and not cuda_available():
                continue
            s = solve_electrostatic(p, device=dev)
            row.append(f"v2 {dev}: err {(s.capacitance - c_exact) / c_exact:+.2e} solve {s.timing['solve_s']:6.3f} s "
                       f"(setup {s.timing['setup_s']:5.2f} s, it {s.info.iterations})")
        print(" | ".join(row))


def _channel_project(mode: str, n_particles: int, steps: int, size: float) -> Project:
    w, h = 0.05, 0.02
    return Project.model_validate({
        "geometry": {"domain": {"polygon": [[0, 0], [w, 0], [w, h], [0, h]]}, "boundaries": [],
                     "regions": [{"id": "pin", "type": "conductor", "voltage": 0.0,
                                  "shape": {"kind": "circle", "center": [0.02, 0.01], "radius": 0.003}}]},
        "mesh": {"size": size, "mode": mode},
        "dsmc": {"boundaries": [{"edges": [3], "type": "inlet", "pressure_pa": 20.0},
                                {"edges": [1], "type": "outlet", "pressure_pa": 5.0}],
                 "init_pressure_pa": 12.0, "n_particles": n_particles, "n_steps": steps,
                 "avg_steps": steps // 2, "seed": 1, "threads": 8},
    })


def bench_dsmc(args) -> None:
    from es_sim.dsmc import DsmcSimulation
    from es_sim.gdsmc import GpuDsmcSimulation

    print(f"DSMC channel: n_particles={args.n_particles}, size={args.size * 1e3} mm, "
          f"steps v2={args.steps} v1={args.v1_steps}")
    rows = [("v2 GPU", GpuDsmcSimulation, "cartesian", args.steps)]
    if args.v1_steps > 0:
        rows.append(("v1 CPU", DsmcSimulation, "structured", args.v1_steps))
    ms = {}
    for name, cls, mode, steps in rows:
        t = time.perf_counter()
        sim = cls(_channel_project(mode, args.n_particles, steps, args.size))
        setup = time.perf_counter() - t
        res = sim.run()
        ms[name] = res.elapsed_s / steps * 1e3
        print(f"  {name}: setup {setup:6.2f} s, {ms[name]:8.3f} ms/step ({len(sim.tris)} elements)")
    if len(ms) == 2:
        print(f"  speedup: {ms['v1 CPU'] / ms['v2 GPU']:.1f}x")


def _fluid_project(mode: str, steps: int, size: float) -> Project:
    return Project.model_validate({
        "geometry": {"domain": {"polygon": [[0, 0], [0.02, 0], [0.02, 0.01], [0, 0.01]]},
                     "boundaries": [
                         {"edges": [3], "type": "dirichlet", "voltage": 0.0,
                          "voltage_rf": {"amplitude": 100.0, "freq_hz": 13.56e6}, "see_gamma": 0.05},
                         {"edges": [1], "type": "dirichlet", "voltage": 0.0, "see_gamma": 0.05},
                         {"edges": [0, 2], "type": "symmetry"}],
                     "regions": [
                         {"id": "blk", "type": "dielectric", "eps_r": 4.0, "see_gamma": 0.1,
                          "polygon": [[0.008, 0], [0.012, 0], [0.012, 0.003], [0.008, 0.003]]},
                         {"id": "pin", "type": "conductor", "voltage": 0.0, "see_gamma": 0.05,
                          "shape": {"kind": "circle", "center": [0.0137, 0.0068], "radius": 0.0011}}]},
        "mesh": {"size": size, "mode": mode},
        "fluid2d": {"init_density_m3": 5e14, "init_te_ev": 3.0, "gas_pressure_pa": 30.0,
                    "n_steps": steps, "frame_every": 10**9, "avg_steps": 10},
    })


def bench_fluid2d(args) -> None:
    from es_sim.fluid2d import Fluid2dSimulation
    from es_sim.gfluid import CartesianFluid2dSimulation

    print(f"fluid2d CCP: size={args.size * 1e3} mm, steps={args.steps}")
    ms = {}
    for name, cls, mode in (("v2", CartesianFluid2dSimulation, "cartesian"), ("v1", Fluid2dSimulation, "unstructured")):
        t = time.perf_counter()
        sim = cls(_fluid_project(mode, args.steps, args.size))
        setup = time.perf_counter() - t
        n_sub = [0]
        step_once = sim._step_once

        def counted(dt, t_now, implicit, _f=step_once, _n=n_sub):
            _n[0] += 1
            return _f(dt, t_now, implicit)

        sim._step_once = counted
        t = time.perf_counter()
        sim.run_batch(store_frames=False)
        run = time.perf_counter() - t
        ms[name] = run / args.steps * 1e3
        poisson = "LU" if getattr(sim, "_lu", None) is not None else ("GMG" if name == "v2" else "splu")
        print(f"  {name}: {sim.n_active:6d} active nodes, setup {setup:5.2f} s, {ms[name]:8.2f} ms/step "
              f"({n_sub[0] / args.steps:.1f} substeps/step, {run / n_sub[0] * 1e3:.2f} ms/substep, Poisson {poisson})")
    print(f"  speedup: {ms['v1'] / ms['v2']:.1f}x")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("pic")
    pp.add_argument("--n-macro", type=int, default=200000)
    pp.add_argument("--steps", type=int, default=2000)
    pp.add_argument("--v1-steps", type=int, default=300)
    pp.add_argument("--cells", type=int, default=200)
    sub.add_parser("poisson")
    pd = sub.add_parser("dsmc")
    pd.add_argument("--n-particles", type=int, default=400000)
    pd.add_argument("--steps", type=int, default=300)
    pd.add_argument("--v1-steps", type=int, default=60)
    pd.add_argument("--size", type=float, default=0.25e-3)
    pf = sub.add_parser("fluid2d")
    pf.add_argument("--steps", type=int, default=100)
    pf.add_argument("--size", type=float, default=0.5e-3)
    args = ap.parse_args()
    {"pic": bench_pic, "poisson": bench_poisson, "dsmc": bench_dsmc, "fluid2d": bench_fluid2d}[args.cmd](args)


if __name__ == "__main__":
    main()
