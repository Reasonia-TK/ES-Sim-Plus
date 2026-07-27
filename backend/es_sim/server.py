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
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from . import _numba_kernels  # noqa: F401 (eager import。理由は下のコメント参照)
from .backend import gpu_available
from .fem import solve
from .lxcat import parse_lxcat
from .meshing import generate_mesh
from .particles import trace
from .pic import PicSimulation
from .postprocess import sample_line
from .dsmc import DsmcSimulation
from .mcc import GasField
from .sweep import build_sweep_cases, run_sweep
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
    return {
        "status": "ok",
        "version": __version__,
        "gpu": gpu_available(),
        "numba": _numba_kernels.HAVE_NUMBA,
    }


@app.post("/mesh", response_model=MeshResult)
def mesh_endpoint(project: Project) -> MeshResult:
    try:
        return _mesh_result(generate_mesh(project))
    except Exception as exc:  # gmsh 由来の失敗をフロントへ伝える
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/solve", response_model=SolveResult)
def solve_endpoint(project: Project) -> SolveResult:
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


def _store_dsmc_result(res) -> None:
    """DSMC 結果を PIC (mcc.use_dsmc_gas) 用の保持スロットへ格納する。"""
    global _last_dsmc
    _last_dsmc = {
        "n_elems": len(res.n),
        "field": GasField(n_g=res.n, t_g=res.t, u_g=res.u),
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
    _store_dsmc_result(res)
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
        _store_dsmc_result(res)
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


@app.post("/lxcat/parse", response_model=LxcatParseResult)
def lxcat_parse_endpoint(req: LxcatParseRequest) -> LxcatParseResult:
    """LXCat 形式テキストをパースして断面積プロセス一覧を返す (prompts/19)。"""
    try:
        processes, warnings = parse_lxcat(req.text, req.species)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return LxcatParseResult(processes=processes, warnings=warnings)


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
        # メッシュ生成・行列組み立ても重いのでスレッドで実行
        sim = await asyncio.to_thread(PicSimulation, project, gas_field)
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
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(frame)

        loop.call_soon_threadsafe(offer_latest)

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
        done_msg: dict = {"type": "done", "history": history}
        # 位相別プロファイル計測 (prompts/75)。total は表示用に別途加算しておく
        # (continue では sim.timing が区間分のみを持つので、total もその区間分になる)
        done_msg["timing"] = {**sim.timing, "total": sum(sim.timing.values())}
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

    if not isinstance(param_path, str) or param_path == "":
        await ws.send_json({"type": "error", "detail": "param_path を指定してください"})
        return
    if not isinstance(values, list) or len(values) == 0:
        await ws.send_json({"type": "error", "detail": "values (値リスト) を指定してください"})
        return
    try:
        parallel = max(1, int(parallel))
        values = [float(v) for v in values]
    except (TypeError, ValueError):
        await ws.send_json({"type": "error", "detail": "parallel/values の型が不正です"})
        return

    # use_dsmc_gas はバッチ実行 (_worker) 同様プロセス間で DSMC 結果を共有できないため、
    # ケースを1つも起動せず開始時点でエラーにする (batch.py の _worker と同じ制約)
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
        {"type": "started", "n_cases": len(cases), "param_path": param_path, "values": values}
    )

    loop = asyncio.get_running_loop()
    stop = threading.Event()
    queue: asyncio.Queue = asyncio.Queue()

    def on_event(ev: dict) -> None:
        # ワーカースレッドからイベントループへ安全に渡す (PIC/DSMC の on_frame と同じ設計)
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    run_task = asyncio.create_task(
        asyncio.to_thread(
            run_sweep, cases, parallel=parallel, out_dir=tmp_dir, on_event=on_event, should_stop=stop.is_set
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
