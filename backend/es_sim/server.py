"""FastAPI ローカルサーバー。Tauri フロントエンドから 127.0.0.1:8317 で利用する。

起動:  uvicorn es_sim.server:app --port 8317
"""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from . import _numba_kernels  # noqa: F401 (eager import。理由は下のコメント参照)
from .backend import gpu_available
from .boltz import DEFAULT_BOLTZ_OPTS, boltzpm_available, run_boltz_sweep
from .fem import solve
from .lxcat import parse_lxcat
from .meshing import generate_mesh
from .particles import trace
from .pic import WALK_DIAG_KEYS, PicSimulation
from .pic1d import Pic1dSimulation, build_pic1d_result
from .pic1d_presets import edupic_ar_processes
from .pic1d_presets import get_presets as get_pic1d_presets
from .fluid1d import Fluid1dSimulation, build_fluid1d_result
from .fluid2d import Fluid2dSimulation, build_fluid2d_result
from .postprocess import sample_line
from .dsmc import DsmcSimulation
from .tl import TlSimulation
from .mcc import GasField
from .sweep import build_sweep_cases, resolve_sweep_module, run_sweep
from .xs.api import XsParseRequest, XsParseResponse, parse_xs_text
from .device import describe as describe_device
from .field.compat import cartesian_mesh_result, cartesian_profile, cartesian_solve
from .gpic import make_pic_simulation
from .schema import (
    DsmcResultModel,
    ElectrodeCharge,
    LxcatParseRequest,
    LxcatParseResult,
    MeshResult,
    Project,
    ProfileRequest,
    ProfileResult,
    SolveResult,
    TraceResult,
    XsProcess,
)

# _numba_kernels は particles.py などから optional 依存として import されるが、
# 実際に import が走るのは初回 PIC/DSMC/trace 実行時 (遅延)。PyInstaller ビルドで
# numba の同梱が漏れていると、そのときになって初めて (try/except で握りつぶされて)
# numpy フォールバックへ静かに倒れ、パッケージ漏れに気付きにくい。ここで
# モジュールレベルに先に import しておくことで、サーバー起動 (= /health が
# 応答可能になる前) の時点で HAVE_NUMBA が確定し、/health の "numba" フィールドと
# release.yml の smoke テストで漏れを早期検出できる (prompts/76)。

app = FastAPI(title="ES-Sim backend", version=__version__)

app.add_middleware(
    CORSMiddleware,
    # 開発時: http://localhost:1420 (vite) / ブラウザ直接アクセス
    # 配布版: Windows (WebView2) は http://tauri.localhost、macOS/Linux は tauri://localhost
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|tauri\.localhost)(:\d+)?|tauri://localhost",
    allow_methods=["*"],
    allow_headers=["*"],
)


def _mesh_result(mesh) -> MeshResult:
    return MeshResult(
        nodes=[tuple(p) for p in mesh.nodes.tolist()],
        triangles=[tuple(t) for t in mesh.triangles.tolist()],
        region_of_triangle=mesh.tri_region.tolist(),
    )


@app.get("/health")
def health():
    dev = describe_device()
    return {
        "status": "ok",
        "version": __version__,
        # "gpu" は v2 エンジン (mesh.mode="cartesian") が CUDA を使えるか (prompts/119)
        "gpu": bool(dev["cuda"]) or gpu_available(),
        "numba": _numba_kernels.HAVE_NUMBA,
        "v2": dev,
    }


@app.post("/mesh", response_model=MeshResult)
def mesh_endpoint(project: Project) -> MeshResult:
    try:
        if project.mesh.mode == "cartesian":
            # v2: 直交格子を三角形分割した表示用メッシュ (prompts/119)
            return cartesian_mesh_result(project)
        return _mesh_result(generate_mesh(project))
    except Exception as exc:  # gmsh 由来の失敗をフロントへ伝える
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/solve", response_model=SolveResult)
def solve_endpoint(project: Project) -> SolveResult:
    if project.mesh.mode == "cartesian":
        # v2: 直交格子 + 埋め込み境界の GMG-PCG (GPU があれば GPU、prompts/119)
        try:
            res, _ = cartesian_solve(project)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return res
    try:
        mesh = generate_mesh(project)
        sol = solve(project, mesh)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    e_abs = (sol.e_field[:, 0] ** 2 + sol.e_field[:, 1] ** 2) ** 0.5
    return SolveResult(
        mesh=_mesh_result(mesh),
        v=sol.v.tolist(),
        e_field=[tuple(e) for e in sol.e_field.tolist()],
        v_min=float(sol.v.min()),
        v_max=float(sol.v.max()),
        e_abs_max=float(e_abs.max()),
        energy=sol.energy,
        charges=[ElectrodeCharge(label=label, voltage=voltage, q=q)
                 for label, voltage, q in sol.charges],
        capacitance=sol.capacitance,
    )


@app.post("/profile", response_model=ProfileResult)
def profile_endpoint(req: ProfileRequest) -> ProfileResult:
    try:
        if req.project.mesh.mode == "cartesian":
            s, v, e_abs = cartesian_profile(req.project, req.p1, req.p2, req.n)
        else:
            mesh = generate_mesh(req.project)
            sol = solve(req.project, mesh)
            s, v, e_abs = sample_line(mesh, sol, req.p1, req.p2, req.n)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    def _nan_to_none(arr: np.ndarray) -> list[float | None]:
        return [None if np.isnan(x) else float(x) for x in arr]

    return ProfileResult(
        s=s.tolist(),
        v=_nan_to_none(v),
        e_abs=_nan_to_none(e_abs),
    )


@app.post("/trace", response_model=TraceResult)
def trace_endpoint(project: Project) -> TraceResult:
    if project.particles is None:
        raise HTTPException(status_code=422, detail="project.particles が指定されていません")
    try:
        mesh = generate_mesh(project)
        sol = solve(project, mesh)
        result = trace(project, mesh, sol)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    status = ["absorbed" if a else "alive" for a in result.absorbed.tolist()]
    tof = [None if math.isnan(t) else float(t) for t in result.tof.tolist()]
    return TraceResult(
        trajectories=result.trajectories.tolist(),
        status=status,
        tof=tof,
        final_energy_ev=result.final_energy_ev.tolist(),
        final_angle_deg=result.final_angle_deg.tolist(),
        dt=result.dt,
        currents=None if result.currents is None else result.currents.tolist(),
        fn_current=result.fn_current,
    )


# ---- DSMC (定常ガス流れ、prompts/54) ------------------------------------------

# 直近の DSMC 結果 (プロセス内に1つ)。PIC の mcc.use_dsmc_gas が参照する
_last_dsmc: dict | None = None


def _store_dsmc_result(sim: DsmcSimulation, res) -> None:
    """DSMC 結果を PIC (mcc.use_dsmc_gas) 用の保持スロットへ格納する。

    GasField に DSMC が実際に使ったメッシュ (sim.mesh の節点・要素) を同梱する
    (prompts/89)。mesh_scale=1.0 なら PIC 側のメッシュと要素数が一致するため
    使われず (従来のビット不変経路)、mesh_scale>1 で要素数が食い違う場合のみ
    PicSimulation 側がこれを使って要素重心マッピングを行う
    """
    global _last_dsmc
    _last_dsmc = {
        "n_elems": len(res.n),
        "field": GasField(
            n_g=res.n, t_g=res.t, u_g=res.u,
            src_nodes=sim.mesh.nodes, src_triangles=sim.mesh.triangles,
        ),
    }


def _dsmc_result_model(sim: DsmcSimulation, res) -> DsmcResultModel:
    return DsmcResultModel(
        mesh=_mesh_result(sim.mesh),
        n=res.n.tolist(),
        t=res.t.tolist(),
        u=[tuple(x) for x in res.u.tolist()],
        p=res.p.tolist(),
        n_particles=res.n_particles,
        macro_weight=res.macro_weight,
        dt=res.dt,
        inflow=res.inflow,
        outflow=res.outflow,
        elapsed_s=res.elapsed_s,
        timing=res.timing,
    )


@app.post("/dsmc", response_model=DsmcResultModel)
def dsmc_endpoint(project: Project) -> DsmcResultModel:
    """定常ガス流れの DSMC を実行し、要素ごとの n・T・u・p を返す (同期版)。

    結果はサーバー内に保持され、PIC (mcc.use_dsmc_gas=true) の背景ガス場として
    使われる (メッシュの要素数が一致している必要がある)。進捗表示つきの実行は
    /ws/dsmc (prompts/58) を使う。
    """
    if project.dsmc is None:
        raise HTTPException(status_code=422, detail="project.dsmc が指定されていません")
    try:
        sim = DsmcSimulation(project)
        res = sim.run()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _store_dsmc_result(sim, res)
    return _dsmc_result_model(sim, res)


# 完了/停止後も状態を保持するスロット (プロセス内に1つ、PIC の _last_sim と同じ設計、
# prompts/74)。新しい start で置き換え、continue で追加実行する
_last_dsmc_sim: DsmcSimulation | None = None
# start / continue の同時実行を防ぐロック (PIC 側の流儀に合わせる)
_dsmc_lock = asyncio.Lock()


async def _run_dsmc_session(ws: WebSocket, project_dict: dict) -> None:
    """1回の DSMC 実行 (start、WS)。started → progress (100ステップごと) → done を送出する。"""
    global _last_dsmc_sim
    try:
        project = Project.model_validate(project_dict)
        if project.dsmc is None:
            raise ValueError("project.dsmc が指定されていません")
        # メッシュ生成・初期充填も重いのでスレッドで実行
        sim = await asyncio.to_thread(DsmcSimulation, project)
    except Exception as exc:
        await ws.send_json({"type": "error", "detail": str(exc)})
        return

    _last_dsmc_sim = sim  # 新しい start で保持状態を置き換える
    await _stream_dsmc(ws, sim)


async def _continue_dsmc_session(ws: WebSocket, msg: dict) -> None:
    """保持中の状態から追加実行 (continue、WS)。応答は start と同形。"""
    sim = _last_dsmc_sim
    if sim is None:
        await ws.send_json(
            {"type": "error", "detail": "保持中の実行状態がありません (先に start してください)"}
        )
        return
    try:
        n_steps = int(msg.get("n_steps", sim.s.n_steps))
        if n_steps <= 0:
            raise ValueError("n_steps は正の整数を指定してください")
        avg_steps = msg.get("avg_steps")  # null なら前回設定を踏襲
        sim.prepare_continue(n_steps, avg_steps)
    except Exception as exc:
        await ws.send_json({"type": "error", "detail": str(exc)})
        return
    await _stream_dsmc(ws, sim)


async def _stream_dsmc(ws: WebSocket, sim: DsmcSimulation) -> None:
    """sim.run() をワーカースレッドで実行し、started → progress → done を送出する
    (start / continue 共通、PIC の _stream_run と同じ設計)。
    """
    await ws.send_json(
        {
            "type": "started",
            "n_steps": sim.s.n_steps,
            "dt": sim.dt,
            "n_particles": len(sim.x),
            # 実効スレッド数。設定が実際に動いているバックエンドへ届いているかを
            # フロントで確認できるようにする (PIC の effective_threads と同じ趣旨)
            "threads": sim._nthreads,
        }
    )

    loop = asyncio.get_running_loop()
    stop = threading.Event()
    queue: asyncio.Queue = asyncio.Queue()

    def on_progress(step: int, n_particles: int, positions: np.ndarray) -> None:
        # positions は dsmc.py 側で既に間引き済み (≤2000点)。ライブ粒子表示 (prompts/66) 用に
        # PIC の frame メッセージと同じ形式 ([[x,y], ...]) で同梱する
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {
                "type": "progress",
                "step": step,
                "n_steps": sim.s.n_steps,
                "n_particles": n_particles,
                "particles": positions.tolist(),
            },
        )

    run_task = asyncio.create_task(
        asyncio.to_thread(sim.run, on_progress, stop.is_set)
    )

    async def watch_stop() -> None:
        while True:
            try:
                msg = json.loads(await ws.receive_text())
            except (WebSocketDisconnect, RuntimeError):
                stop.set()
                return
            if msg.get("cmd") == "stop":
                stop.set()
                return

    stop_task = asyncio.create_task(watch_stop())
    try:
        while True:
            if run_task.done() and queue.empty():
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            await ws.send_json(item)
        res = await run_task
        _store_dsmc_result(sim, res)
        await ws.send_json(
            {"type": "done", "result": _dsmc_result_model(sim, res).model_dump()}
        )
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass
    finally:
        stop.set()
        stop_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@app.websocket("/ws/dsmc")
async def ws_dsmc(ws: WebSocket) -> None:
    """DSMC 実行の WebSocket (prompts/58)。

    start で新規実行、stop で中断、continue で保持中の状態から追加実行する
    (prompts/74、完了/停止後も状態はサーバー側に保持され、新しい start で置き換わる)。
    """
    await ws.accept()
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            cmd = msg.get("cmd")
            if cmd in ("start", "continue"):
                if _dsmc_lock.locked():
                    # 別接続で実行中の start / continue は拒否する (PIC 側の流儀と同じ)
                    await ws.send_json(
                        {"type": "error", "detail": "別の DSMC 実行が進行中です"}
                    )
                    continue
                async with _dsmc_lock:
                    if cmd == "start":
                        await _run_dsmc_session(ws, msg.get("project") or {})
                    else:
                        await _continue_dsmc_session(ws, msg)
            elif cmd == "stop":
                continue  # 実行中でなければ無視
            else:
                await ws.send_json(
                    {"type": "error", "detail": f"不明なコマンドです: {cmd}"}
                )
    except (WebSocketDisconnect, RuntimeError):
        pass


# ---- VHF 定在波 (非線形径方向伝送線路モデル、prompts/101) --------------------
# continue には対応しない (毎回フルの定常化をやり直す設計。tl.py 参照) ため、
# dsmc/pic1d と違って「保持中の状態」スロットは不要。start / stop のみの
# 単純なロック付きストリーミングで足りる
_tl_lock = asyncio.Lock()


async def _run_tl_session(ws: WebSocket, project_dict: dict) -> None:
    """1回の VHF 定在波実行 (start)。started → progress → done を送出する。"""
    try:
        project = Project.model_validate(project_dict)
        if project.tl is None:
            raise ValueError("project.tl が指定されていません")
        # 格子・三重対角行列の構築も軽くはないのでスレッドで実行する
        sim = await asyncio.to_thread(TlSimulation, project)
    except Exception as exc:
        await ws.send_json({"type": "error", "detail": str(exc)})
        return

    await ws.send_json(
        {"type": "started", "n_steps": sim.n_steps_total, "dt": sim.dt}
    )

    loop = asyncio.get_running_loop()
    stop = threading.Event()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)

    def on_progress(step: int, n_steps: int, elapsed_s: float) -> None:
        def offer_latest() -> None:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(
                {"type": "progress", "step": step, "n_steps": n_steps, "elapsed_s": elapsed_s}
            )

        loop.call_soon_threadsafe(offer_latest)

    run_task = asyncio.create_task(
        asyncio.to_thread(sim.run, on_progress, stop.is_set)
    )

    async def watch_stop() -> None:
        while True:
            try:
                msg = json.loads(await ws.receive_text())
            except (WebSocketDisconnect, RuntimeError):
                stop.set()
                return
            if msg.get("cmd") == "stop":
                stop.set()
                return

    stop_task = asyncio.create_task(watch_stop())
    try:
        while True:
            if run_task.done() and queue.empty():
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            await ws.send_json(item)
        result = await run_task
        await ws.send_json({"type": "done", "result": result})
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass
    finally:
        stop.set()
        stop_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@app.websocket("/ws/tl")
async def ws_tl(ws: WebSocket) -> None:
    """VHF 定在波 (非線形径方向伝送線路モデル) 実行の WebSocket (prompts/101)。

    start で新規実行、stop で中断する。continue は無い (tl.py の docstring 参照:
    毎回フルの定常化 (立ち上げランプ→FFT窓) をやり直す設計のため、途中から
    追加実行するという概念がそもそも成立しない)。
    """
    await ws.accept()
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            cmd = msg.get("cmd")
            if cmd == "start":
                if _tl_lock.locked():
                    await ws.send_json(
                        {"type": "error", "detail": "別の VHF 定在波実行が進行中です"}
                    )
                    continue
                async with _tl_lock:
                    await _run_tl_session(ws, msg.get("project") or {})
            elif cmd == "stop":
                continue  # 実行中でなければ無視
            else:
                await ws.send_json(
                    {"type": "error", "detail": f"不明なコマンドです: {cmd}"}
                )
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.post("/lxcat/parse", response_model=LxcatParseResult)
def lxcat_parse_endpoint(req: LxcatParseRequest) -> LxcatParseResult:
    """LXCat 形式テキストをパースして断面積プロセス一覧を返す (prompts/19)。"""
    try:
        processes, warnings = parse_lxcat(req.text, req.species)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return LxcatParseResult(processes=processes, warnings=warnings)


@app.post("/v2/xs/parse", response_model=XsParseResponse)
def xs_parse_v2_endpoint(req: XsParseRequest) -> XsParseResponse:
    """LXCat 形式テキストを v2 断面積モデル (es_sim.xs) でパースする (prompts/120)。

    DATABASE ごとの全ブロック (CrossSection.to_dict() の形、SI 単位)・(入射粒子, 標的) ごとの
    要約・パース警告を返す。パース失敗は /lxcat/parse と同じく 422。
    """
    try:
        return parse_xs_text(req.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---- PIC WebSocket ストリーミング (フェーズ3、仕様書 §9) ----------------------

# 完了/停止後もシミュレーション状態を保持するスロット (プロセス内に1つ、prompts/32)。
# 新しい start で置き換え、continue で追加実行する
_last_sim: PicSimulation | None = None
# start / continue の同時実行を防ぐロック (実行中の要求は拒否する)
_pic_lock = asyncio.Lock()


async def _run_pic_session(ws: WebSocket, project_dict: dict) -> None:
    """1回の PIC 実行 (start)。完了/停止後も状態を保持スロットに残す。"""
    global _last_sim
    try:
        project = Project.model_validate(project_dict)
        # 非一様背景ガス場 (prompts/54): mcc.use_dsmc_gas なら直近の DSMC 結果を渡す
        gas_field = None
        if (
            project.pic is not None
            and project.pic.mcc is not None
            and project.pic.mcc.use_dsmc_gas
        ):
            if _last_dsmc is None:
                raise ValueError(
                    "DSMC の実行結果がありません (先にガス流れ (DSMC) を実行してください)"
                )
            gas_field = _last_dsmc["field"]
        # メッシュ生成・行列組み立ても重いのでスレッドで実行。
        # mesh.mode="cartesian" は v2 GPU PIC (prompts/119)、それ以外は v1 FEM-PIC
        sim = await asyncio.to_thread(make_pic_simulation, project, gas_field)
    except Exception as exc:
        await ws.send_json({"type": "error", "detail": str(exc)})
        return

    _last_sim = sim  # 新しい start で保持状態を置き換える
    await _stream_run(ws, sim)


async def _continue_pic_session(ws: WebSocket, msg: dict) -> None:
    """保持中の状態から追加実行 (continue)。応答は start と同形。"""
    sim = _last_sim
    if sim is None:
        await ws.send_json(
            {"type": "error", "detail": "保持中の実行状態がありません (先に start してください)"}
        )
        return
    try:
        n_steps = int(msg.get("n_steps", sim.pic.n_steps))
        if n_steps <= 0:
            raise ValueError("n_steps は正の整数を指定してください")
        frame_every = msg.get("frame_every")
        avg_steps = msg.get("avg_steps")       # null なら前回設定を踏襲
        phase_bins = msg.get("phase_bins")     # null なら前回設定を踏襲
        sim.prepare_continue(n_steps, frame_every, avg_steps, phase_bins)
    except Exception as exc:
        await ws.send_json({"type": "error", "detail": str(exc)})
        return
    await _stream_run(ws, sim)


async def _stream_run(ws: WebSocket, sim: PicSimulation) -> None:
    """run_batch をワーカースレッドで実行し、started → frame → done を送出する。"""
    loop = asyncio.get_running_loop()
    stop = threading.Event()
    # ライブ表示は途中フレームを全て再生する用途ではない。計算がJSON送信より速い場合に
    # 古いフレームを溜めるとメモリが増え、画面も過去へ遅延するため最新1件だけ保持する。
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)

    await ws.send_json(
        {
            "type": "started",
            "dt": sim.dt,
            "n_steps": sim.pic.n_steps,
            "effective_threads": sim.effective_threads,
            # 区間開始時の通算ステップ数。frame.step は start からの通算で進むため、
            # continue の進捗率はフロント側で (step - step_offset)/n_steps として計算する
            "step_offset": sim.step_count,
            "warnings": sim.warnings,
            "mesh": {
                "nodes": sim.mesh.nodes.tolist(),
                "triangles": sim.mesh.triangles.tolist(),
            },
        }
    )

    def on_frame(frame: dict) -> None:
        # ワーカースレッドからイベントループへ安全に渡す。満杯なら古い表示フレームを
        # 捨てて最新へ置換する (物理計算・history・完了結果には影響しない)。
        def offer_latest() -> None:
            if queue.full():
                try:
                    old = queue.get_nowait()
                    # v2 PIC の動的再格子化 (prompts/123): 捨てるフレームが新しい格子を運んでいたら
                    # 引き継ぐ (フロントは mesh 付きフレームで表示格子を差し替える)
                    if "mesh" in old and "mesh" not in frame:
                        frame["mesh"] = old["mesh"]
                        frame["mesh_version"] = old.get("mesh_version")
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(frame)

        loop.call_soon_threadsafe(offer_latest)

    # run_batch の壁時計計測 (prompts/86)。continue では sim.timing 同様この回の
    # 区間分のみになる (通算値にはしない)
    t_run0 = time.perf_counter()
    run_task = asyncio.create_task(
        asyncio.to_thread(sim.run_batch, on_frame, stop.is_set, False)
    )

    async def watch_stop() -> None:
        # 実行中の stop コマンド (または切断) を監視する
        while True:
            try:
                msg = json.loads(await ws.receive_text())
            except (WebSocketDisconnect, RuntimeError):
                stop.set()
                return
            if msg.get("cmd") == "stop":
                stop.set()
                return

    stop_task = asyncio.create_task(watch_stop())
    try:
        while True:
            if run_task.done() and queue.empty():
                break
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            await ws.send_json(frame)
        history, _ = await run_task
        done_msg: dict = {
            "type": "done",
            "history": history,
            # run_batch の壁時計秒 (prompts/86)。フレーム送信等のオーバーヘッドは
            # timing.total (位相別プロファイル計測、prompts/75) との差分で見える
            "elapsed_s": time.perf_counter() - t_run0,
        }
        # v2 PIC の動的再格子化 (prompts/123): fields・cycle は最終の格子の上の値なので格子も添える
        if getattr(sim, "mesh_version", 0) > 0:
            done_msg["mesh"] = sim.mesh_payload()
            done_msg["regrids"] = sim.regrid_log
        # 位相別プロファイル計測 (prompts/75)。total は表示用に別途加算しておく
        # (continue では sim.timing が区間分のみを持つので、total もその区間分になる)。
        # walk コスト診断 (prompts/88、WALK_DIAG_KEYS) は秒数ではないため合計から除外する
        timing_secs = {k: v for k, v in sim.timing.items() if k not in WALK_DIAG_KEYS}
        done_msg["timing"] = {**sim.timing, "total": sum(timing_secs.values())}
        # 時間平均フィールド (prompts/26)。平均区間を積算できていれば添付する
        if sim.fields is not None:
            done_msg["fields"] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in sim.fields.items()
            }
        # RF 1周期の位相分解データ (prompts/28)。RF なし・phase_bins=0 なら省略。
        # ペイロード削減のため数値は float32 精度に丸めて JSON 化する
        if sim.cycle is not None:
            def _f32(arr) -> list:
                return np.asarray(arr, dtype=np.float32).tolist()

            c = sim.cycle
            done_msg["cycle"] = {
                "bins": c["bins"],
                "period_s": c["period_s"],
                "phi": _f32(c["phi"]),
                "n_e": _f32(c["n_e"]),
                "n_i": _f32(c["n_i"]),
                # 追加フィールド (prompts/52)。e_abs は要素値、他は節点値
                "e_abs": _f32(c["e_abs"]),
                "te_ev": _f32(c["te_ev"]),
                "ion_rate": _f32(c["ion_rate"]),
                "particles": {
                    name: [_f32(s) for s in snaps]
                    for name, snaps in c["particles"].items()
                },
            }
        # IEDF/IADF コレクタ (prompts/30、複数対応 prompts/36)。有効時のみ添付する
        if sim.collector_results is not None:
            def _cr_json(cr: dict) -> dict:
                return {
                    "count": cr["count"],
                    "total_weight": cr["total_weight"],
                    "energies_ev": cr["energies_ev"].tolist(),
                    "angles_deg": cr["angles_deg"].tolist(),
                    "weights": cr["weights"].tolist(),
                    "truncated": cr["truncated"],
                }

            done_msg["collectors"] = [_cr_json(cr) for cr in sim.collector_results]
            if len(sim.collector_results) == 1:
                # 後方互換: コレクタが1個のときのみ従来の単数キーも出力する
                done_msg["collector"] = done_msg["collectors"][0]
        # EEDF/EEPF 領域 (prompts/85)。有効時のみ添付する
        if sim.eedf_results is not None:
            done_msg["eedf"] = [
                {
                    "label": r["label"],
                    "e_centers": r["e_centers"].tolist(),
                    "f": r["f"].tolist(),
                    "mean_energy_ev": r["mean_energy_ev"],
                    "t_eff_ev": r["t_eff_ev"],
                    "total_weight": r["total_weight"],
                    "overflow_frac": r["overflow_frac"],
                    "n_samples": r["n_samples"],
                }
                for r in sim.eedf_results
            ]
        await ws.send_json(done_msg)
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass
    finally:
        stop.set()
        stop_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@app.websocket("/ws/pic")
async def ws_pic(ws: WebSocket) -> None:
    """PIC 実行の WebSocket。

    start で新規実行、stop で中断、continue で保持中の状態から追加実行する
    (完了/停止後も状態はサーバー側に保持され、新しい start で置き換わる)。
    """
    await ws.accept()
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            cmd = msg.get("cmd")
            if cmd in ("start", "continue"):
                if _pic_lock.locked():
                    # 別接続で実行中の start / continue は拒否する
                    await ws.send_json(
                        {"type": "error", "detail": "別の PIC 実行が進行中です"}
                    )
                    continue
                async with _pic_lock:
                    if cmd == "start":
                        await _run_pic_session(ws, msg.get("project", {}))
                    else:
                        await _continue_pic_session(ws, msg)
            elif cmd == "stop":
                continue  # 実行中でなければ無視
            else:
                await ws.send_json({"type": "error", "detail": f"不明なコマンド: {cmd}"})
    except WebSocketDisconnect:
        pass


# ---- 1D PIC/MCC WebSocket ストリーミング (prompts/91) --------------------------
#
# 2D の /ws/pic (PicSimulation) とは完全に独立したソルバー・状態を使う。
# geometry/mesh 生成が無く軽量なので、start (メッシュ生成に相当する重い処理が
# 無い) もそのまま同期的に構築して問題ないが、既存の流儀 (anyio/asyncio スレッド
# オフロード) に揃えるため 2D と同じ asyncio.to_thread 経由にする。

_last_sim1d: Pic1dSimulation | None = None
_pic1d_lock = asyncio.Lock()


@app.get("/pic1d/presets")
def pic1d_presets_endpoint() -> dict:
    """1D PIC/MCC のベンチマークプリセット一覧 (prompts/91)。

    レスポンス形式: {プリセット名: {"label": str, "description": str,
    "pic1d": <Pic1dSettings と同じ形の dict>, "note"?: str}}。
    "note" は turner_he_case1 のみ (断面積が未設定であることの注意書き)。
    """
    return get_pic1d_presets()


async def _run_pic1d_session(ws: WebSocket, project_dict: dict) -> None:
    """1回の 1D PIC 実行 (start)。完了/停止後も状態を保持スロットに残す。"""
    global _last_sim1d
    try:
        project = Project.model_validate(project_dict)
        sim = await asyncio.to_thread(Pic1dSimulation, project)
    except Exception as exc:
        await ws.send_json({"type": "error", "detail": str(exc)})
        return
    _last_sim1d = sim  # 新しい start で保持状態を置き換える
    await _stream_run_1d(ws, sim)


async def _continue_pic1d_session(ws: WebSocket, msg: dict) -> None:
    """保持中の状態から追加実行 (continue)。応答は start と同形。"""
    sim = _last_sim1d
    if sim is None:
        await ws.send_json(
            {"type": "error", "detail": "保持中の実行状態がありません (先に start してください)"}
        )
        return
    try:
        extra_steps = int(msg.get("extra_steps", sim.s.n_steps))
        if extra_steps <= 0:
            raise ValueError("extra_steps は正の整数を指定してください")
        frame_every = msg.get("frame_every")
        avg_steps = msg.get("avg_steps")
        phase_bins = msg.get("phase_bins")
        sim.prepare_continue(extra_steps, frame_every, avg_steps, phase_bins)
    except Exception as exc:
        await ws.send_json({"type": "error", "detail": str(exc)})
        return
    await _stream_run_1d(ws, sim)


async def _stream_run_1d(ws: WebSocket, sim: Pic1dSimulation) -> None:
    """run_batch をワーカースレッドで実行し、started → frame → done を送出する
    (2D の _stream_run と同じ設計)。
    """
    loop = asyncio.get_running_loop()
    stop = threading.Event()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)

    await ws.send_json(
        {
            "type": "started",
            "n_steps": sim.s.n_steps,
            "step_offset": sim.step_count,
            "dt": sim.dt,
            "x": sim.xg.tolist(),
            "warnings": sim.warnings,
        }
    )

    def on_frame(frame: dict) -> None:
        def offer_latest() -> None:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(frame)

        loop.call_soon_threadsafe(offer_latest)

    t_run0 = time.perf_counter()
    run_task = asyncio.create_task(
        asyncio.to_thread(sim.run_batch, on_frame, stop.is_set, False)
    )

    async def watch_stop() -> None:
        while True:
            try:
                msg = json.loads(await ws.receive_text())
            except (WebSocketDisconnect, RuntimeError):
                stop.set()
                return
            if msg.get("cmd") == "stop":
                stop.set()
                return

    stop_task = asyncio.create_task(watch_stop())
    try:
        while True:
            if run_task.done() and queue.empty():
                break
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            await ws.send_json(frame)
        await run_task
        elapsed_s = time.perf_counter() - t_run0
        await ws.send_json({"type": "done", "result": build_pic1d_result(sim, elapsed_s)})
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass
    finally:
        stop.set()
        stop_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@app.websocket("/ws/pic1d")
async def ws_pic1d(ws: WebSocket) -> None:
    """1D PIC/MCC 実行の WebSocket (prompts/91)。

    start で新規実行、stop で中断、continue (extra_steps 指定) で保持中の状態
    から追加実行する (完了/停止後も状態はサーバー側に保持され、新しい start で置き換わる)。
    2D の /ws/pic とは独立したロック・保持スロットを持つため、両者は同時に実行できる。
    """
    await ws.accept()
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            cmd = msg.get("cmd")
            if cmd in ("start", "continue"):
                if _pic1d_lock.locked():
                    await ws.send_json(
                        {"type": "error", "detail": "別の 1D PIC 実行が進行中です"}
                    )
                    continue
                async with _pic1d_lock:
                    if cmd == "start":
                        await _run_pic1d_session(ws, msg.get("project", {}))
                    else:
                        await _continue_pic1d_session(ws, msg)
            elif cmd == "stop":
                continue  # 実行中でなければ無視
            else:
                await ws.send_json({"type": "error", "detail": f"不明なコマンド: {cmd}"})
    except WebSocketDisconnect:
        pass


# ---- 1D 流体 (ドリフト拡散、prompts/104-107) ------------------------------------
#
# pic1d と同一の電極・格子規約を共有するが粒子を追わない連続場ソルバー。配線は
# /ws/pic1d をそのまま複製する (専用ロック・保持スロット、start/continue/stop、
# anyio オフロード) — 独自の形式は作らない。

_last_simfluid1d: Fluid1dSimulation | None = None
_fluid1d_lock = asyncio.Lock()


async def _run_fluid1d_session(ws: WebSocket, project_dict: dict) -> None:
    """1回の流体 (1D) 実行 (start)。完了/停止後も状態を保持スロットに残す。"""
    global _last_simfluid1d
    try:
        project = Project.model_validate(project_dict)
        sim = await asyncio.to_thread(Fluid1dSimulation, project)
    except Exception as exc:
        await ws.send_json({"type": "error", "detail": str(exc)})
        return
    _last_simfluid1d = sim  # 新しい start で保持状態を置き換える
    await _stream_run_fluid1d(ws, sim)


async def _continue_fluid1d_session(ws: WebSocket, msg: dict) -> None:
    """保持中の状態から追加実行 (continue)。応答は start と同形。"""
    sim = _last_simfluid1d
    if sim is None:
        await ws.send_json(
            {"type": "error", "detail": "保持中の実行状態がありません (先に start してください)"}
        )
        return
    try:
        extra_steps = int(msg.get("extra_steps", sim.s.n_steps))
        if extra_steps <= 0:
            raise ValueError("extra_steps は正の整数を指定してください")
        frame_every = msg.get("frame_every")
        avg_steps = msg.get("avg_steps")
        phase_bins = msg.get("phase_bins")
        sim.prepare_continue(extra_steps, frame_every, avg_steps, phase_bins)
    except Exception as exc:
        await ws.send_json({"type": "error", "detail": str(exc)})
        return
    await _stream_run_fluid1d(ws, sim)


async def _stream_run_fluid1d(ws: WebSocket, sim: Fluid1dSimulation) -> None:
    """run_batch をワーカースレッドで実行し、started → frame → done を送出する
    (1D PIC の _stream_run_1d と同じ設計)。
    """
    loop = asyncio.get_running_loop()
    stop = threading.Event()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)

    await ws.send_json(
        {
            "type": "started",
            "n_steps": sim.s.n_steps,
            "step_offset": sim.step_count,
            "dt": sim.dt,
            "x": sim.xg.tolist(),
            "warnings": sim.warnings,
        }
    )

    def on_frame(frame: dict) -> None:
        def offer_latest() -> None:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(frame)

        loop.call_soon_threadsafe(offer_latest)

    t_run0 = time.perf_counter()
    run_task = asyncio.create_task(
        asyncio.to_thread(sim.run_batch, on_frame, stop.is_set, False)
    )

    async def watch_stop() -> None:
        while True:
            try:
                msg = json.loads(await ws.receive_text())
            except (WebSocketDisconnect, RuntimeError):
                stop.set()
                return
            if msg.get("cmd") == "stop":
                stop.set()
                return

    stop_task = asyncio.create_task(watch_stop())
    try:
        while True:
            if run_task.done() and queue.empty():
                break
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            await ws.send_json(frame)
        await run_task
        elapsed_s = time.perf_counter() - t_run0
        await ws.send_json({"type": "done", "result": build_fluid1d_result(sim, elapsed_s)})
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass
    finally:
        stop.set()
        stop_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@app.websocket("/ws/fluid1d")
async def ws_fluid1d(ws: WebSocket) -> None:
    """流体 (1D ドリフト拡散) 実行の WebSocket (prompts/107)。

    start で新規実行、stop で中断、continue (extra_steps 指定) で保持中の状態
    から追加実行する (完了/停止後も状態はサーバー側に保持され、新しい start で置き換わる)。
    pic/pic1d とは独立したロック・保持スロットを持つため、他ソルバーと同時に実行できる。
    """
    await ws.accept()
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            cmd = msg.get("cmd")
            if cmd in ("start", "continue"):
                if _fluid1d_lock.locked():
                    await ws.send_json(
                        {"type": "error", "detail": "別の流体 (1D) 実行が進行中です"}
                    )
                    continue
                async with _fluid1d_lock:
                    if cmd == "start":
                        await _run_fluid1d_session(ws, msg.get("project", {}))
                    else:
                        await _continue_fluid1d_session(ws, msg)
            elif cmd == "stop":
                continue  # 実行中でなければ無視
            else:
                await ws.send_json({"type": "error", "detail": f"不明なコマンド: {cmd}"})
    except WebSocketDisconnect:
        pass


# ---- 2D/軸対称 流体 (ドリフト拡散、prompts/111-112) ------------------------------------
#
# /ws/fluid1d の配線 (専用ロック・保持スロット、start/continue/stop、anyio オフロード) を
# そのまま複製する。fluid1d と違い geometry からメッシュ・EAFE エッジ重み・Poisson の
# splu 事前分解を組み立てる重い構築 (PicSimulation と同じ規模) になるため、
# コンストラクタも PIC 同様 asyncio.to_thread でオフロードする。
# started にはメッシュを含めない (2D PIC の /ws/pic と異なる点、prompts/112 の指示):
# フロントは既に /mesh のレスポンスでメッシュを保持しており、fluid2d はその既存メッシュと
# 同じ project から生成されるメッシュを使う (Project.geometry/mesh が変わっていなければ
# 節点番号・座標は /mesh のときと完全に一致する) ので、二重送信を避けてペイロードを削減する。

_last_simfluid2d: Fluid2dSimulation | None = None
_fluid2d_lock = asyncio.Lock()


async def _run_fluid2d_session(ws: WebSocket, project_dict: dict) -> None:
    """1回の流体 (2D) 実行 (start)。完了/停止後も状態を保持スロットに残す。"""
    global _last_simfluid2d
    try:
        project = Project.model_validate(project_dict)
        # メッシュ生成・EAFE エッジ重み・Poisson の splu 事前分解も重いので
        # PIC と同様スレッドで実行する
        sim = await asyncio.to_thread(Fluid2dSimulation, project)
    except Exception as exc:
        await ws.send_json({"type": "error", "detail": str(exc)})
        return
    _last_simfluid2d = sim  # 新しい start で保持状態を置き換える
    await _stream_run_fluid2d(ws, sim)


async def _continue_fluid2d_session(ws: WebSocket, msg: dict) -> None:
    """保持中の状態から追加実行 (continue)。応答は start と同形。"""
    sim = _last_simfluid2d
    if sim is None:
        await ws.send_json(
            {"type": "error", "detail": "保持中の実行状態がありません (先に start してください)"}
        )
        return
    try:
        extra_steps = int(msg.get("extra_steps", sim.s.n_steps))
        if extra_steps <= 0:
            raise ValueError("extra_steps は正の整数を指定してください")
        frame_every = msg.get("frame_every")
        avg_steps = msg.get("avg_steps")
        phase_bins = msg.get("phase_bins")
        sim.prepare_continue(extra_steps, frame_every, avg_steps, phase_bins)
    except Exception as exc:
        await ws.send_json({"type": "error", "detail": str(exc)})
        return
    await _stream_run_fluid2d(ws, sim)


async def _stream_run_fluid2d(ws: WebSocket, sim: Fluid2dSimulation) -> None:
    """run_batch をワーカースレッドで実行し、started → frame → done を送出する
    (fluid1d の _stream_run_fluid1d と同じ設計。frame は sim._make_frame() の形
    (phi/n_e/n_i/t_e、いずれも全節点長の配列) をそのまま使う — 2D PIC のライブ
    frame と同じ「節点配列をそのまま送る」流儀に揃えている)。
    """
    loop = asyncio.get_running_loop()
    stop = threading.Event()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)

    await ws.send_json(
        {
            "type": "started",
            "n_steps": sim.s.n_steps,
            "step_offset": sim.step_count,
            "dt": sim.dt,
            "warnings": sim.warnings,
            # 陰的反復ソルバーの実効スレッド数 (PIC の effective_threads と同じ趣旨、
            # prompts/115)。フロントで設定が実際に反映されているかを確認できるようにする
            "effective_threads": sim.effective_threads,
        }
    )

    def on_frame(frame: dict) -> None:
        def offer_latest() -> None:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(frame)

        loop.call_soon_threadsafe(offer_latest)

    t_run0 = time.perf_counter()
    run_task = asyncio.create_task(
        asyncio.to_thread(sim.run_batch, on_frame, stop.is_set, False)
    )

    async def watch_stop() -> None:
        while True:
            try:
                msg = json.loads(await ws.receive_text())
            except (WebSocketDisconnect, RuntimeError):
                stop.set()
                return
            if msg.get("cmd") == "stop":
                stop.set()
                return

    stop_task = asyncio.create_task(watch_stop())
    try:
        while True:
            if run_task.done() and queue.empty():
                break
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            await ws.send_json(frame)
        await run_task
        elapsed_s = time.perf_counter() - t_run0
        await ws.send_json({"type": "done", "result": build_fluid2d_result(sim, elapsed_s)})
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass
    finally:
        stop.set()
        stop_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@app.websocket("/ws/fluid2d")
async def ws_fluid2d(ws: WebSocket) -> None:
    """流体 (2D/軸対称 ドリフト拡散) 実行の WebSocket (prompts/112)。

    start で新規実行、stop で中断、continue (extra_steps 指定) で保持中の状態
    から追加実行する (完了/停止後も状態はサーバー側に保持され、新しい start で置き換わる)。
    pic/pic1d/fluid1d とは独立したロック・保持スロットを持つため、他ソルバーと同時に実行できる。
    """
    await ws.accept()
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            cmd = msg.get("cmd")
            if cmd in ("start", "continue"):
                if _fluid2d_lock.locked():
                    await ws.send_json(
                        {"type": "error", "detail": "別の流体 (2D) 実行が進行中です"}
                    )
                    continue
                async with _fluid2d_lock:
                    if cmd == "start":
                        await _run_fluid2d_session(ws, msg.get("project", {}))
                    else:
                        await _continue_fluid2d_session(ws, msg)
            elif cmd == "stop":
                continue  # 実行中でなければ無視
            else:
                await ws.send_json({"type": "error", "detail": f"不明なコマンド: {cmd}"})
    except WebSocketDisconnect:
        pass


# ---- boltzpm (Boltzmann ソルバー) 連携 — LMEA 流体係数テーブル生成 (prompts/117) ----------
#
# テーブルはフロントが受け取って fluid1d/fluid2d の settings.boltz_table に格納する
# (サーバー側では保持しない、プロンプト指示通り) ため、tl.py の _run_tl_session と同じ
# 最も単純な配線 (専用ロック、start/stop のみ、continue 無し、保持スロット無し) で足りる
_boltz_lock = asyncio.Lock()


async def _run_boltz_session(ws: WebSocket, msg: dict) -> None:
    """1回の boltzpm E/N 掃引 (start)。started → progress×N → done(table) を送出する。

    processes を直接渡すか (mass_amu/p_pa が必須、t_k は既定 300K)、project+module
    ("fluid1d"/"fluid2d") を渡して既存設定 (electron_processes/ion_mass_amu/
    gas_pressure_pa/gas_temperature_k) から取り出すかのどちらかを選べる。
    electron_processes が空の場合は fluid1d.py/fluid2d.py と同じ既定
    (eduPIC Ar 解析式) にフォールバックする。
    """
    if not boltzpm_available():
        await ws.send_json(
            {"type": "error", "detail": "boltzpm がインストールされていません"}
        )
        return

    processes_raw = msg.get("processes")
    opts = msg.get("opts")
    try:
        if processes_raw is not None:
            processes = [XsProcess.model_validate(p) for p in processes_raw]
            mass_amu = float(msg["mass_amu"])
            p_pa = float(msg["p_pa"])
            t_k = float(msg.get("t_k", 300.0))
        else:
            project_dict = msg.get("project")
            module = msg.get("module")
            if not isinstance(project_dict, dict) or module not in ("fluid1d", "fluid2d"):
                await ws.send_json(
                    {
                        "type": "error",
                        "detail": "processes (+mass_amu/p_pa)、または "
                        "project+module ('fluid1d'/'fluid2d') を指定してください",
                    }
                )
                return
            project = Project.model_validate(project_dict)
            settings = project.fluid1d if module == "fluid1d" else project.fluid2d
            if settings is None:
                await ws.send_json({"type": "error", "detail": f"project.{module} が指定されていません"})
                return
            processes = settings.electron_processes if settings.electron_processes else edupic_ar_processes()[0]
            mass_amu = float(settings.ion_mass_amu)
            p_pa = float(settings.gas_pressure_pa)
            t_k = float(settings.gas_temperature_k)
    except Exception as exc:
        await ws.send_json({"type": "error", "detail": str(exc)})
        return

    n_points = int((opts or {}).get("n_points") or DEFAULT_BOLTZ_OPTS["n_points"])
    await ws.send_json({"type": "started", "n_points": n_points})

    loop = asyncio.get_running_loop()
    stop = threading.Event()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)

    def on_progress(i: int, n: int, en_td: float, elapsed_s: float) -> None:
        def offer_latest() -> None:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(
                {"type": "progress", "i": i, "n_points": n, "en_td": en_td, "elapsed_s": elapsed_s}
            )

        loop.call_soon_threadsafe(offer_latest)

    run_task = asyncio.create_task(
        asyncio.to_thread(
            run_boltz_sweep, processes, mass_amu, p_pa, t_k, opts, on_progress, stop.is_set
        )
    )

    async def watch_stop() -> None:
        while True:
            try:
                m = json.loads(await ws.receive_text())
            except (WebSocketDisconnect, RuntimeError):
                stop.set()
                return
            if m.get("cmd") == "stop":
                stop.set()
                return

    stop_task = asyncio.create_task(watch_stop())
    try:
        while True:
            if run_task.done() and queue.empty():
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            await ws.send_json(item)
        table = await run_task
        await ws.send_json({"type": "done", "table": table})
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass
    finally:
        stop.set()
        stop_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@app.websocket("/ws/boltz")
async def ws_boltz(ws: WebSocket) -> None:
    """boltzpm による E/N 掃引 → LMEA 係数テーブル (BoltzTable) 生成の WebSocket (prompts/117)。

    start で新規実行、stop で中断 (中断時点までの収束済み点でテーブルを返す)。
    continue や保持スロットは無い (テーブルはフロントの settings.boltz_table に
    そのまま埋め込むだけの1回性の生成なので、pic1d/fluid1d のような「続きから」
    という概念が無い)。
    """
    await ws.accept()
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            cmd = msg.get("cmd")
            if cmd == "start":
                if _boltz_lock.locked():
                    await ws.send_json({"type": "error", "detail": "別の boltzpm 実行が進行中です"})
                    continue
                async with _boltz_lock:
                    await _run_boltz_session(ws, msg)
            elif cmd == "stop":
                continue  # 実行中でなければ無視 (実行中は watch_stop が処理する)
            else:
                await ws.send_json({"type": "error", "detail": f"不明なコマンド: {cmd}"})
    except (WebSocketDisconnect, RuntimeError):
        pass


# ---- パラメータスイープ (GUIから1パラメータ×値リストを並列実行、prompts/79) --------------

# 直近スイープの結果一時ディレクトリ (サーバープロセス生存中のみ)。ケース数×cycle で
# 結果が巨大になり得るためメモリには持たず、GET /sweep/result/{i} がここから読む。
# 新しい start のたびに前回の一時ディレクトリを破棄する
_sweep_tmp_dir: str | None = None
# スイープの同時複数実行を拒否するロック (PIC/DSMC の実行ロックとは独立)
_sweep_lock = asyncio.Lock()


def _cleanup_sweep_tmp_dir() -> None:
    global _sweep_tmp_dir
    if _sweep_tmp_dir is not None:
        shutil.rmtree(_sweep_tmp_dir, ignore_errors=True)
        _sweep_tmp_dir = None


async def _run_sweep_session(ws: WebSocket, msg: dict) -> None:
    """1回のスイープ実行 (start)。started → progress/case_done×N → done を送出する。"""
    global _sweep_tmp_dir
    project_dict = msg.get("project") or {}
    param_path = msg.get("param_path")
    values = msg.get("values")
    parallel = msg.get("parallel", 1)
    # "pic"|"pic1d"|"fluid1d"|"fluid2d"|None (未指定は param_path から自動判定、prompts/112)
    requested_module = msg.get("module")

    if not isinstance(param_path, str) or param_path == "":
        await ws.send_json({"type": "error", "detail": "param_path を指定してください"})
        return
    if not isinstance(values, list) or len(values) == 0:
        await ws.send_json({"type": "error", "detail": "values (値リスト) を指定してください"})
        return
    if requested_module is not None and requested_module not in ("pic", "pic1d", "fluid1d", "fluid2d"):
        await ws.send_json(
            {
                "type": "error",
                "detail": "module は 'pic'・'pic1d'・'fluid1d'・'fluid2d' のいずれかを指定してください",
            }
        )
        return
    try:
        parallel = max(1, int(parallel))
        values = [float(v) for v in values]
    except (TypeError, ValueError):
        await ws.send_json({"type": "error", "detail": "parallel/values の型が不正です"})
        return

    # module の解決 (prompts/96): UI 確定値を最優先し、未指定時のみ param_path から判定する。
    # _worker には常にこの確定済みの値だけを渡す (auto のまま渡さない)
    module = resolve_sweep_module(param_path, requested_module)

    # use_dsmc_gas はバッチ実行 (_worker) 同様プロセス間で DSMC 結果を共有できないため、
    # ケースを1つも起動せず開始時点でエラーにする (batch.py の _worker と同じ制約)。
    # 1D (pic1d) はこの制約と無関係 (schema の validator が use_dsmc_gas 自体を拒否する) なので
    # module=="pic" のときのみ検査する
    if module == "pic":
        pic = project_dict.get("pic") if isinstance(project_dict, dict) else None
        if isinstance(pic, dict):
            mcc = pic.get("mcc")
            if isinstance(mcc, dict) and mcc.get("use_dsmc_gas"):
                await ws.send_json(
                    {
                        "type": "error",
                        "detail": "pic.mcc.use_dsmc_gas はスイープでは未対応です "
                        "(DSMC結果はプロセス間で共有されないため)",
                    }
                )
                return

    try:
        cases = build_sweep_cases(project_dict, param_path, values)
    except Exception as exc:
        await ws.send_json({"type": "error", "detail": str(exc)})
        return

    _cleanup_sweep_tmp_dir()
    _sweep_tmp_dir = tempfile.mkdtemp(prefix="es_sim_sweep_")
    tmp_dir = _sweep_tmp_dir  # このセッション実行中に他の start で差し替わらないようローカルへ固定

    await ws.send_json(
        {
            "type": "started",
            "n_cases": len(cases),
            "param_path": param_path,
            "values": values,
            "module": module,  # 解決済みの値をフロントへ返す (表示用、prompts/96)
        }
    )

    loop = asyncio.get_running_loop()
    stop = threading.Event()
    queue: asyncio.Queue = asyncio.Queue()

    def on_event(ev: dict) -> None:
        # ワーカースレッドからイベントループへ安全に渡す (PIC/DSMC の on_frame と同じ設計)
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    run_task = asyncio.create_task(
        asyncio.to_thread(
            run_sweep,
            cases,
            parallel=parallel,
            out_dir=tmp_dir,
            on_event=on_event,
            should_stop=stop.is_set,
            module=module,
        )
    )

    async def watch_stop() -> None:
        while True:
            try:
                m = json.loads(await ws.receive_text())
            except (WebSocketDisconnect, RuntimeError):
                stop.set()
                return
            if m.get("cmd") == "stop":
                stop.set()
                return

    stop_task = asyncio.create_task(watch_stop())
    summary: list[dict] = []
    try:
        while True:
            if run_task.done() and queue.empty():
                break
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            if ev["type"] == "case_done":
                entry = {"case": ev["case"], "value": values[ev["case"]], "ok": ev["ok"]}
                if not ev["ok"]:
                    entry["error"] = ev["error"]
                summary.append(entry)
            await ws.send_json(ev)
        await run_task  # 例外があれば (通常は起きない想定だが) ここで送出される
        summary.sort(key=lambda s: s["case"])
        await ws.send_json({"type": "done", "summary": summary})
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass
    finally:
        stop.set()
        stop_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@app.websocket("/ws/sweep")
async def ws_sweep(ws: WebSocket) -> None:
    """パラメータスイープの WebSocket (prompts/79)。

    start で新規スイープを開始する (同時実行は1つのみ、別接続からの start は拒否する)。
    stop は実行中セッション内の watch_stop タスクが処理する (PIC/DSMC と同じ設計)。
    continue には対応しない (スイープは毎回フルの N ケースを実行する)。
    リクエストに optional な module ("pic"/"pic1d"/"fluid1d"/"fluid2d") を指定すると
    どのソルバーで実行するかを明示できる (未指定は param_path の接頭辞で自動判定、
    prompts/96・107・112)。
    """
    await ws.accept()
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            cmd = msg.get("cmd")
            if cmd == "start":
                if _sweep_lock.locked():
                    await ws.send_json({"type": "error", "detail": "別のスイープが実行中です"})
                    continue
                async with _sweep_lock:
                    await _run_sweep_session(ws, msg)
            elif cmd == "stop":
                continue  # 実行中でなければ無視 (実行中は _run_sweep_session 内で処理される)
            else:
                await ws.send_json({"type": "error", "detail": f"不明なコマンド: {cmd}"})
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.get("/sweep/result/{i}")
def sweep_result_endpoint(i: int) -> dict:
    """スイープ結果 (ケース i の結果付きJSON) をそのまま返す。未完了/失敗は404。"""
    if _sweep_tmp_dir is None:
        raise HTTPException(status_code=404, detail="スイープ結果がありません (先にスイープを実行してください)")
    path = Path(_sweep_tmp_dir) / f"case_{i}_result.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"ケース {i} の結果がありません (未完了または失敗)")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
