"""ジョブの種類ごとの実行の仕方 (prompts/130 P6d)。

v1 の WebSocket (/ws/pic・/ws/pic1d・/ws/fluid1d・/ws/fluid2d・/ws/dsmc・/ws/tl・/ws/boltz・/ws/sweep) と
REST (/trace) と同じソルバー・同じ started・frame・結果の形を使う。結果は v1 の「結果付き保存」
(ResultsBundle) の各キーと同じ形 (PIC は batch._build_results_bundle と同じ)。
"""

from __future__ import annotations

import math
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..batch import _build_results_bundle
from ..boltz import boltzpm_available, run_boltz_sweep, DEFAULT_BOLTZ_OPTS
from ..fem import solve
from ..fluid1d import Fluid1dSimulation, build_fluid1d_result
from ..fluid2d import build_fluid2d_result
from ..gdsmc import make_dsmc_simulation
from ..gfluid import make_fluid2d_simulation
from ..gpic import make_pic_simulation
from ..meshing import generate_mesh
from ..particles import trace
from ..pic import WALK_DIAG_KEYS
from ..pic1d import Pic1dSimulation, build_pic1d_result
from ..pic1d_presets import edupic_ar_processes
from ..schema import Project, XsProcess
from ..sweep import build_sweep_cases, resolve_sweep_module, run_sweep
from ..tl import TlSimulation
from .manager import Emitter


@dataclass
class Hooks:
    """サーバーと共有するもの (v1 の DSMC 結果の保持スロット: PIC の use_dsmc_gas が使う)。"""

    #: DSMC の結果を保持スロット (とジョブごとの表) へ入れる (sim, res, job_id)
    store_dsmc: Callable[[Any, Any, str], None] | None = None
    #: 保持スロットのガス場 (無ければ None)
    last_dsmc_field: Callable[[], Any] | None = None
    #: DSMC の結果 (DsmcResultModel の dict)
    dsmc_result: Callable[[Any, Any], dict] | None = None
    #: ジョブの DSMC のガス場 (dsmc_job を指定したとき)
    job_dsmc_field: Callable[[str], Any] | None = None


def _project(project: dict | None) -> Project:
    if project is None:
        raise ValueError("project を指定してください")
    return Project.model_validate(project)


def _steps(options: dict, default: int) -> int:
    n = int(options.get("steps") or default)
    if n <= 0:
        raise ValueError("steps は正の整数を指定してください")
    return n


class BaseRunner:
    kind = ""
    continuable = False
    keep_state = False

    def validate(self, project: dict | None, options: dict) -> Any:
        return _project(project)

    def build(self, parsed: Any, options: dict, ctx: Any) -> Any:
        raise NotImplementedError

    def started(self, state: Any) -> dict:
        return {}

    def run(self, state: Any, emit: Emitter, should_stop: Callable[[], bool]) -> dict:
        raise NotImplementedError

    def prepare_continue(self, state: Any, options: dict) -> None:
        raise ValueError("この計算は続きを実行できません")

    def release(self, state: Any) -> None:
        pass


# ---- PIC (2D) ------------------------------------------------------------------------


@dataclass
class SimState:
    sim: Any
    step_offset: int = 0
    extra: dict = field(default_factory=dict)


class PicRunner(BaseRunner):
    kind = "pic"
    continuable = True

    def validate(self, project, options):
        p = _project(project)
        if p.pic is None:
            raise ValueError("project.pic が指定されていません")
        return p

    def build(self, p: Project, options: dict, ctx: Hooks) -> SimState:
        gas = None
        if p.pic.mcc is not None and p.pic.mcc.use_dsmc_gas:
            job = options.get("dsmc_job")
            if job and ctx is not None and ctx.job_dsmc_field is not None:
                gas = ctx.job_dsmc_field(str(job))
            elif ctx is not None and ctx.last_dsmc_field is not None:
                gas = ctx.last_dsmc_field()
            if gas is None:
                raise ValueError("DSMC の実行結果がありません (先にガス流れ (DSMC) を実行してください)")
        return SimState(make_pic_simulation(p, gas))

    def started(self, st: SimState) -> dict:
        sim = st.sim
        st.step_offset = sim.step_count
        return {
            "dt": sim.dt,
            "n_steps": sim.pic.n_steps,
            "effective_threads": sim.effective_threads,
            "step_offset": sim.step_count,
            "warnings": sim.warnings,
            "mesh": {"nodes": sim.mesh.nodes.tolist(), "triangles": sim.mesh.triangles.tolist()},
        }

    def run(self, st: SimState, emit: Emitter, should_stop) -> dict:
        sim = st.sim
        t0 = time.perf_counter()
        sim.run_batch(emit.frame, should_stop, False)
        elapsed = time.perf_counter() - t0
        result = _build_results_bundle(sim, st.step_offset, elapsed)["pic"]
        result["frame"] = emit.last_frame
        timing = {k: v for k, v in sim.timing.items() if k not in WALK_DIAG_KEYS}
        result["timing"] = {**sim.timing, "total": sum(timing.values())}
        if getattr(sim, "mesh_version", 0) > 0:
            result["regrids"] = sim.regrid_log
        return result

    def prepare_continue(self, st: SimState, options: dict) -> None:
        sim = st.sim
        sim.prepare_continue(
            _steps(options, sim.pic.n_steps), options.get("frame_every"), options.get("avg_steps"), options.get("phase_bins")
        )


# ---- PIC 1D・流体 1D・流体 2D ------------------------------------------------------------


class _Run1dRunner(BaseRunner):
    """pic1d / fluid1d / fluid2d に共通の run_batch + build_*_result の形。"""

    continuable = True
    settings_key = ""

    def validate(self, project, options):
        p = _project(project)
        if getattr(p, self.settings_key) is None:
            raise ValueError(f"project.{self.settings_key} が指定されていません")
        return p

    def make(self, p: Project) -> Any:
        raise NotImplementedError

    def result(self, sim: Any, elapsed: float) -> dict:
        raise NotImplementedError

    def build(self, p, options, ctx) -> SimState:
        return SimState(self.make(p))

    def started(self, st: SimState) -> dict:
        sim = st.sim
        st.step_offset = sim.step_count
        return {"n_steps": sim.s.n_steps, "step_offset": sim.step_count, "dt": sim.dt, "x": sim.xg.tolist(), "warnings": sim.warnings}

    def run(self, st: SimState, emit: Emitter, should_stop) -> dict:
        t0 = time.perf_counter()
        st.sim.run_batch(emit.frame, should_stop, False)
        return self.result(st.sim, time.perf_counter() - t0)

    def prepare_continue(self, st: SimState, options: dict) -> None:
        sim = st.sim
        sim.prepare_continue(
            _steps(options, sim.s.n_steps), options.get("frame_every"), options.get("avg_steps"), options.get("phase_bins")
        )


class Pic1dRunner(_Run1dRunner):
    kind = "pic1d"
    settings_key = "pic1d"

    def make(self, p):
        return Pic1dSimulation(p)

    def result(self, sim, elapsed):
        return build_pic1d_result(sim, elapsed)


class Fluid1dRunner(_Run1dRunner):
    kind = "fluid1d"
    settings_key = "fluid1d"

    def make(self, p):
        return Fluid1dSimulation(p)

    def result(self, sim, elapsed):
        return build_fluid1d_result(sim, elapsed)


class Fluid2dRunner(_Run1dRunner):
    kind = "fluid2d"
    settings_key = "fluid2d"

    def make(self, p):
        return make_fluid2d_simulation(p)

    @staticmethod
    def _mesh(sim) -> dict | None:
        if hasattr(sim, "mesh_payload"):
            return sim.mesh_payload()
        mesh = getattr(sim, "mesh", None)
        if mesh is None:
            return None
        return {"nodes": mesh.nodes.tolist(), "triangles": mesh.triangles.tolist()}

    def started(self, st):
        sim = st.sim
        st.step_offset = sim.step_count
        out = {
            "n_steps": sim.s.n_steps,
            "step_offset": sim.step_count,
            "dt": sim.dt,
            "warnings": sim.warnings,
            "effective_threads": sim.effective_threads,
        }
        mesh = self._mesh(sim)
        if mesh is not None:
            out["mesh"] = mesh
        return out

    def result(self, sim, elapsed):
        out = build_fluid2d_result(sim, elapsed)
        # v1 の結果はメッシュを持たなかった (/mesh の結果を使った)。実行ごとに自己完結させる
        mesh = self._mesh(sim)
        if mesh is not None:
            out["mesh"] = mesh
        return out


# ---- DSMC ------------------------------------------------------------------------------


class DsmcRunner(BaseRunner):
    kind = "dsmc"
    continuable = True

    def validate(self, project, options):
        p = _project(project)
        if p.dsmc is None:
            raise ValueError("project.dsmc が指定されていません")
        return p

    def build(self, p, options, ctx: Hooks) -> SimState:
        st = SimState(make_dsmc_simulation(p))
        st.extra["hooks"] = ctx
        return st

    def started(self, st: SimState) -> dict:
        sim = st.sim
        return {"n_steps": sim.s.n_steps, "dt": sim.dt, "n_particles": len(sim.x), "threads": sim._nthreads, "step_offset": 0}

    def run(self, st: SimState, emit: Emitter, should_stop) -> dict:
        sim = st.sim
        n_steps = sim.s.n_steps

        def on_progress(step: int, n_particles: int, positions: np.ndarray) -> None:
            emit.frame({"step": step, "n_steps": n_steps, "n_particles": n_particles, "particles": positions.tolist()})

        res = sim.run(on_progress, should_stop)
        hooks: Hooks | None = st.extra.get("hooks")
        if hooks is not None and hooks.store_dsmc is not None:
            hooks.store_dsmc(sim, res, emit.job_id)
        if hooks is not None and hooks.dsmc_result is not None:
            return hooks.dsmc_result(sim, res)
        return {"n": res.n.tolist(), "t": res.t.tolist(), "p": res.p.tolist()}

    def prepare_continue(self, st: SimState, options: dict) -> None:
        sim = st.sim
        avg = options.get("avg_steps")
        sim.prepare_continue(_steps(options, sim.s.n_steps), None if avg is None else int(avg))


# ---- VHF 定在波 ------------------------------------------------------------------------


class TlRunner(BaseRunner):
    kind = "tl"

    def validate(self, project, options):
        p = _project(project)
        if p.tl is None:
            raise ValueError("project.tl が指定されていません")
        return p

    def build(self, p, options, ctx):
        return TlSimulation(p)

    def started(self, sim) -> dict:
        return {"n_steps": sim.n_steps_total, "dt": sim.dt}

    def run(self, sim, emit: Emitter, should_stop) -> dict:
        def on_progress(step: int, n_steps: int, elapsed_s: float) -> None:
            emit.progress(step, n_steps, elapsed_s=elapsed_s)

        return sim.run(on_progress, should_stop)


# ---- 粒子軌道 (v1 の /trace と同じ) ---------------------------------------------------


class TraceRunner(BaseRunner):
    kind = "trace"

    def validate(self, project, options):
        p = _project(project)
        if p.particles is None:
            raise ValueError("project.particles が指定されていません")
        return p

    def build(self, p, options, ctx):
        return p

    def started(self, p) -> dict:
        return {"n_steps": 3}

    def run(self, p: Project, emit: Emitter, should_stop) -> dict:
        emit.progress(0, 3, phase="mesh")
        mesh = generate_mesh(p)
        emit.progress(1, 3, phase="solve")
        sol = solve(p, mesh)
        emit.progress(2, 3, phase="trace")
        r = trace(p, mesh, sol)
        emit.progress(3, 3, phase="done")
        return {
            "trajectories": r.trajectories.tolist(),
            "status": ["absorbed" if a else "alive" for a in r.absorbed.tolist()],
            "tof": [None if math.isnan(t) else float(t) for t in r.tof.tolist()],
            "final_energy_ev": r.final_energy_ev.tolist(),
            "final_angle_deg": r.final_angle_deg.tolist(),
            "dt": r.dt,
            "currents": None if r.currents is None else r.currents.tolist(),
            "fn_current": r.fn_current,
        }


# ---- Boltzmann 係数の表 (v1 の /ws/boltz と同じ) ---------------------------------------


@dataclass
class BoltzInput:
    processes: list
    mass_amu: float
    p_pa: float
    t_k: float
    opts: dict | None


class BoltzRunner(BaseRunner):
    kind = "boltz"

    def validate(self, project, options) -> BoltzInput:
        if not boltzpm_available():
            raise ValueError("boltzpm がインストールされていません")
        opts = options.get("opts")
        if options.get("processes") is not None:
            processes = [XsProcess.model_validate(x) for x in options["processes"]]
            return BoltzInput(processes, float(options["mass_amu"]), float(options["p_pa"]), float(options.get("t_k", 300.0)), opts)
        module = options.get("module")
        if module not in ("fluid1d", "fluid2d"):
            raise ValueError("processes (+mass_amu/p_pa)、または project と module ('fluid1d'/'fluid2d') を指定してください")
        p = _project(project)
        s = p.fluid1d if module == "fluid1d" else p.fluid2d
        if s is None:
            raise ValueError(f"project.{module} が指定されていません")
        processes = s.electron_processes if s.electron_processes else edupic_ar_processes()[0]
        return BoltzInput(processes, float(s.ion_mass_amu), float(s.gas_pressure_pa), float(s.gas_temperature_k), opts)

    def build(self, inp: BoltzInput, options, ctx) -> BoltzInput:
        return inp

    def started(self, inp: BoltzInput) -> dict:
        return {"n_steps": int((inp.opts or {}).get("n_points") or DEFAULT_BOLTZ_OPTS["n_points"])}

    def run(self, inp: BoltzInput, emit: Emitter, should_stop) -> dict:
        def on_progress(i: int, n: int, en_td: float, elapsed_s: float) -> None:
            emit.progress(i, n, en_td=en_td, elapsed_s=elapsed_s)

        table = run_boltz_sweep(inp.processes, inp.mass_amu, inp.p_pa, inp.t_k, inp.opts, on_progress, should_stop)
        return {"table": table}


# ---- パラメータスイープ (v1 の /ws/sweep と同じ) --------------------------------------


@dataclass
class SweepState:
    cases: list[dict]
    param_path: str
    values: list[float]
    parallel: int
    module: str
    tmp_dir: str | None = None


class SweepRunner(BaseRunner):
    kind = "sweep"
    #: ケースの結果ファイル (一時ディレクトリ) をジョブが消えるまで残す
    keep_state = True

    def validate(self, project, options) -> SweepState:
        if not isinstance(project, dict):
            raise ValueError("project を指定してください")
        param_path = options.get("param_path")
        values = options.get("values")
        if not isinstance(param_path, str) or param_path == "":
            raise ValueError("param_path を指定してください")
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError("values (値リスト) を指定してください")
        requested = options.get("module")
        if requested is not None and requested not in ("pic", "pic1d", "fluid1d", "fluid2d"):
            raise ValueError("module は 'pic'・'pic1d'・'fluid1d'・'fluid2d' のいずれかを指定してください")
        try:
            parallel = max(1, int(options.get("parallel", 1)))
            values = [float(v) for v in values]
        except (TypeError, ValueError) as exc:
            raise ValueError("parallel/values の型が不正です") from exc
        module = resolve_sweep_module(param_path, requested)
        if module == "pic":
            mcc = ((project.get("pic") or {}).get("mcc") or {}) if isinstance(project.get("pic"), dict) else {}
            if isinstance(mcc, dict) and mcc.get("use_dsmc_gas"):
                raise ValueError("pic.mcc.use_dsmc_gas はスイープでは未対応です (DSMC結果はプロセス間で共有されないため)")
        cases = build_sweep_cases(project, param_path, values)
        return SweepState(cases, param_path, values, parallel, module)

    def build(self, st: SweepState, options, ctx) -> SweepState:
        st.tmp_dir = tempfile.mkdtemp(prefix="es_sim_job_sweep_")
        return st

    def started(self, st: SweepState) -> dict:
        return {"n_steps": len(st.cases), "n_cases": len(st.cases), "param_path": st.param_path, "values": st.values, "module": st.module}

    def run(self, st: SweepState, emit: Emitter, should_stop) -> dict:
        summary: list[dict] = []
        n = len(st.cases)

        def on_event(ev: dict) -> None:
            if ev["type"] == "progress":
                emit.event({"type": "case_progress", "case": ev["case"], "step": ev["step"], "n_steps": ev["n_steps"]}, ("case", ev["case"]))
            elif ev["type"] == "case_done":
                entry = {"case": ev["case"], "value": st.values[ev["case"]], "ok": ev["ok"]}
                if not ev["ok"]:
                    entry["error"] = ev.get("error")
                if ev.get("self_bias"):
                    entry["self_bias"] = ev["self_bias"]   # [{label, v_dc, v1}] (prompts/134)
                summary.append(entry)
                emit.event({"type": "case", **entry})
                emit.progress(len(summary), n)

        run_sweep(st.cases, parallel=st.parallel, out_dir=st.tmp_dir, on_event=on_event, should_stop=should_stop, module=st.module)
        summary.sort(key=lambda s: s["case"])
        return {"summary": summary, "param_path": st.param_path, "values": st.values, "module": st.module}

    def case_result_path(self, st: SweepState, i: int) -> Path | None:
        if st.tmp_dir is None:
            return None
        path = Path(st.tmp_dir) / f"case_{i}_result.json"
        return path if path.exists() else None

    def release(self, st: SweepState) -> None:
        if st.tmp_dir is not None:
            shutil.rmtree(st.tmp_dir, ignore_errors=True)
            st.tmp_dir = None


def default_runners() -> list[BaseRunner]:
    return [
        PicRunner(),
        Pic1dRunner(),
        Fluid1dRunner(),
        Fluid2dRunner(),
        DsmcRunner(),
        TlRunner(),
        TraceRunner(),
        BoltzRunner(),
        SweepRunner(),
    ]
