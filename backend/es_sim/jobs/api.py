"""/v2/jobs と /v2/events (prompts/130 P6d)。

- POST /v2/jobs {kind, project, options, label} → ジョブの要約 (待ち行列に入る)
- GET /v2/jobs → {jobs, limits, kinds}、GET/PUT /v2/jobs/limits
- GET /v2/jobs/{id} (要約 + 入力)、/started (表示用メッシュ付き)、/frame (最新のフレーム)、/result、
  /cases/{i} (スイープのケースの結果付き JSON)
- POST /v2/jobs/{id}/stop・/continue {options}、DELETE /v2/jobs/{id}
- WS /v2/events: 最初に {type: "hello", jobs, limits, kinds, instance}、以後 job・removed・started・progress・
  frame (watch したジョブだけ)・case・case_progress・done・error。受け手からは {cmd: "watch", ids} と {cmd: "ping"}。
"""

from __future__ import annotations

import asyncio
import json
import math
from collections import deque
from typing import Any

from fastapi import APIRouter, HTTPException, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .manager import JobError, JobManager


def _sanitize(obj: Any) -> Any:
    """NaN・無限大を null に (ブラウザの JSON.parse は NaN を読めない)。"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
    except ValueError:
        return json.dumps(_sanitize(obj), allow_nan=False, ensure_ascii=False, separators=(",", ":"))


def _json(obj: Any, status: int = 200) -> Response:
    return Response(content=dumps(obj), media_type="application/json", status_code=status)


class AsyncSubscriber:
    """WebSocket 1 本分の受け手。push はどのスレッドからも呼べ、progress・frame は key ごとに最新だけ残す。"""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.watch: set[str] = set()
        self.closed = False
        self._items: deque = deque()
        self._latest: dict = {}
        self._wake = asyncio.Event()

    def push(self, event: dict, key: tuple | None = None) -> None:
        if self.closed:
            raise RuntimeError("subscriber closed")
        self.loop.call_soon_threadsafe(self._push, event, key)

    def _push(self, event: dict, key: tuple | None) -> None:
        if key is None:
            self._items.append(("e", event))
        else:
            prev = self._latest.get(key)
            if prev is None:
                self._items.append(("k", key))
            elif "mesh" in prev and "mesh" not in event:
                # 置き換えるフレームが新しいメッシュを運んでいたら引き継ぐ (v1 の server と同じ)
                event = {**event, "mesh": prev["mesh"], "mesh_version": prev.get("mesh_version")}
            self._latest[key] = event
        self._wake.set()

    async def next(self) -> dict:
        while not self._items:
            self._wake.clear()
            await self._wake.wait()
        tag, value = self._items.popleft()
        return value if tag == "e" else self._latest.pop(value)


class JobRequest(BaseModel):
    kind: str
    project: dict | None = None
    options: dict = Field(default_factory=dict)
    label: str | None = None


class ContinueRequest(BaseModel):
    options: dict = Field(default_factory=dict)


class LimitsRequest(BaseModel):
    max_running: int | None = None
    per_kind: dict[str, int] | None = None


def make_router(manager: JobManager, instance: str = "") -> APIRouter:
    router = APIRouter(prefix="/v2")

    def job_or_404(job_id: str):
        try:
            return manager.get(job_id)
        except JobError as e:
            raise HTTPException(e.status, e.detail) from e

    @router.get("/jobs")
    def list_jobs() -> Response:
        return _json({"jobs": manager.list(), "limits": manager.limits(), "kinds": manager.kinds})

    @router.post("/jobs", status_code=201)
    def submit(req: JobRequest) -> Response:
        try:
            job = manager.submit(req.kind, req.project, req.options, req.label)
        except JobError as e:
            raise HTTPException(e.status, e.detail) from e
        return _json(manager.summary(job), 201)

    @router.get("/jobs/limits")
    def get_limits() -> dict:
        return manager.limits()

    @router.put("/jobs/limits")
    def put_limits(req: LimitsRequest) -> dict:
        try:
            return manager.set_limits(req.max_running, req.per_kind)
        except JobError as e:
            raise HTTPException(e.status, e.detail) from e

    @router.get("/jobs/{job_id}")
    def get_job(job_id: str) -> Response:
        job = job_or_404(job_id)
        return _json({**manager.summary(job), "project": job.project})

    @router.get("/jobs/{job_id}/started")
    def get_started(job_id: str) -> Response:
        job = job_or_404(job_id)
        if job.started is None:
            raise HTTPException(404, "まだ始まっていません")
        return _json({"run": job.runs, **job.started})

    @router.get("/jobs/{job_id}/frame")
    def get_frame(job_id: str) -> Response:
        job = job_or_404(job_id)
        if job.frame is None:
            raise HTTPException(404, "フレームがありません")
        return _json(job.frame)

    @router.get("/jobs/{job_id}/result")
    def get_result(job_id: str) -> Response:
        job = job_or_404(job_id)
        if job.result is None:
            raise HTTPException(404, "結果がありません (実行中・失敗・取り消し)")
        return _json(job.result)

    @router.get("/jobs/{job_id}/cases/{i}")
    def get_case(job_id: str, i: int) -> Response:
        job = job_or_404(job_id)
        path_of = getattr(manager.runner(job.kind), "case_result_path", None)
        path = path_of(job.state_obj, i) if path_of is not None and job.state_obj is not None else None
        if path is None:
            raise HTTPException(404, f"ケース {i} の結果がありません (未完了または失敗)")
        # ケースの結果ファイルは json.dump のまま (NaN があり得る) なので読み直して null にする
        return _json(json.loads(path.read_text(encoding="utf-8")))

    @router.post("/jobs/{job_id}/stop")
    def stop(job_id: str) -> Response:
        try:
            return _json(manager.stop(job_id))
        except JobError as e:
            raise HTTPException(e.status, e.detail) from e

    @router.post("/jobs/{job_id}/continue")
    def continue_(job_id: str, req: ContinueRequest | None = None) -> Response:
        try:
            return _json(manager.continue_(job_id, (req.options if req else {})))
        except JobError as e:
            raise HTTPException(e.status, e.detail) from e

    @router.delete("/jobs/{job_id}", status_code=204)
    def delete(job_id: str) -> Response:
        try:
            manager.delete(job_id)
        except JobError as e:
            raise HTTPException(e.status, e.detail) from e
        return Response(status_code=204)

    @router.websocket("/events")
    async def events(ws: WebSocket) -> None:
        await ws.accept()
        sub = AsyncSubscriber(asyncio.get_running_loop())
        unsubscribe = manager.subscribe(sub)

        async def reader() -> None:
            while True:
                msg = json.loads(await ws.receive_text())
                cmd = msg.get("cmd")
                if cmd == "watch":
                    manager.watch(sub, [str(i) for i in msg.get("ids") or []])
                elif cmd == "ping":
                    sub.push({"type": "pong"})

        reader_task = asyncio.create_task(reader())
        try:
            await ws.send_text(
                dumps({"type": "hello", "instance": instance, "jobs": manager.list(), "limits": manager.limits(), "kinds": manager.kinds})
            )
            while True:
                getter = asyncio.create_task(sub.next())
                done, _ = await asyncio.wait({getter, reader_task}, return_when=asyncio.FIRST_COMPLETED)
                if reader_task in done:
                    getter.cancel()
                    break
                await ws.send_text(dumps(getter.result()))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            sub.closed = True
            unsubscribe()
            reader_task.cancel()

    return router
