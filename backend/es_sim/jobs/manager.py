"""UI v2 のジョブ管理 (prompts/130 P6d)。

v1 は WebSocket の接続ごとに「ソルバーの種類ごとに 1 つ」の実行と保持スロットを持っていた。ここでは
計算をジョブとして扱い、同じソルバーでも同時に複数実行でき、実行ごとに状態・結果・続きの実行に使う
シミュレーションの状態を持つ。

- 実行: 1 ジョブ = 1 本のスレッド (ソルバーの組み立てと実行を同じスレッドで行う)。同時に走らせる数は
  全体と種類ごとの上限で決め、超えた分は待ち行列 (到着順) に入れる。
- 続き: 完了/停止したジョブの状態から追加で実行する (続けられる種類だけ)。シミュレーションの状態は大きい
  (粒子・GPU メモリ) ので、種類ごとに新しい方から keep_sims_per_kind 件だけ持つ。
- イベント: 状態の変化・started・progress・frame・done・error を購読者へスレッドから安全に渡す。
  progress と frame は送る間隔を空け (既定 0.1 s・0.05 s。速いソルバーは毎秒数百回知らせてくる)、さらに
  購読者ごとに最新の 1 件だけを残す (遅い受け手のところで溜めない)。frame と重い started (メッシュ付き) は、
  そのジョブを見ている (watch) 購読者にだけ送る。
- ソルバー固有の処理は Runner (runners.py) に置き、ここは種類を知らない。
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

log = logging.getLogger(__name__)

#: 終わった状態 (続き・削除ができる)
TERMINAL = frozenset({"done", "stopped", "error", "cancelled"})

#: started の中で購読者全員には送らない重い項目 (watch している購読者と GET /started だけ)
HEAVY_STARTED_KEYS = frozenset({"mesh", "x"})


class JobError(Exception):
    """API の誤り (HTTP の状態コード付き)。"""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class Runner(Protocol):
    """ソルバーの種類ごとの実行の仕方 (runners.py)。"""

    kind: str
    continuable: bool
    #: 続けられなくても終わった後に状態を持つ (スイープのケースの結果ファイルなど)
    keep_state: bool

    def validate(self, project: dict | None, options: dict) -> Any:
        """要求を受けたスレッドで入力を検査して解釈する (誤りは ValueError → 422)。"""

    def build(self, parsed: Any, options: dict, ctx: Any) -> Any:
        """ワーカースレッドでシミュレーションを組み立てる (メッシュ生成など重い処理)。"""

    def started(self, state: Any) -> dict:
        """実行の開始時に送る情報 (dt・ステップ数・表示用メッシュなど)。"""

    def run(self, state: Any, emit: "Emitter", should_stop: Callable[[], bool]) -> dict:
        """ワーカースレッドで実行し、結果 (JSON にできる dict) を返す。"""

    def prepare_continue(self, state: Any, options: dict) -> None:
        """続きの実行の準備 (続けられない種類は ValueError)。"""

    def release(self, state: Any) -> None:
        """状態が要らなくなったとき (一時ファイルの削除など)。"""


class Subscriber(Protocol):
    """イベントの受け手。push はどのスレッドからも呼ばれる。"""

    watch: set[str]

    def push(self, event: dict, key: tuple | None = None) -> None:
        """key があればその key の古いイベントを置き換える (最新だけ残す)。"""


@dataclass
class Job:
    id: str
    kind: str
    seq: int
    label: str | None
    project: dict | None
    options: dict
    parsed: Any
    created: float
    state: str = "queued"
    stopping: bool = False
    started_at: float | None = None
    finished_at: float | None = None
    #: 実行 (開始・続き) の回数
    runs: int = 0
    #: 今の実行を始めた時刻 (perf_counter) と、それまでの実行時間の合計
    run_t0: float | None = None
    elapsed_s: float = 0.0
    progress: dict | None = None
    started: dict | None = None
    frame: dict | None = None
    result: dict | None = None
    error: str | None = None
    state_obj: Any = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    #: 次に行うこと ("start" / "continue" と続きの設定)
    pending: tuple[str, dict] | None = ("start", {})
    #: 待ち行列に入る前の状態 (続きを止めたときに戻す)
    prev_state: str | None = None
    deleted: bool = False
    #: 最後に progress・frame を送った時刻 (perf_counter) と、間引いて送っていないフレームがあるか
    progress_sent: float = 0.0
    frame_sent: float = 0.0
    frame_pending: bool = False

    def summary(self, continuable: bool, queue_position: int | None = None) -> dict:
        elapsed = self.elapsed_s
        if self.state == "running" and self.run_t0 is not None:
            elapsed += time.perf_counter() - self.run_t0
        warnings = (self.started or {}).get("warnings") or []
        return {
            "id": self.id,
            "kind": self.kind,
            "seq": self.seq,
            "label": self.label,
            "state": self.state,
            "stopping": self.stopping,
            "created": self.created,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": elapsed,
            "runs": self.runs,
            "progress": self.progress,
            "error": self.error,
            "warnings": list(warnings),
            "can_continue": continuable and self.state in ("done", "stopped") and self.state_obj is not None,
            "has_result": self.result is not None,
            "queue_position": queue_position,
            "options": self.options,
        }


class Emitter:
    """Runner がワーカースレッドから進捗・フレームを知らせる口。"""

    def __init__(self, manager: "JobManager", job: Job) -> None:
        self._m = manager
        self._job = job
        self.job_id = job.id
        self.last_frame: dict | None = None

    def frame(self, frame: dict) -> None:
        """ライブ表示のフレーム (最新だけ残す)。step があれば進捗も更新する。"""
        self.last_frame = frame
        self._m._on_frame(self._job, frame)

    def progress(self, step: int, n_steps: int, **extra: Any) -> None:
        self._m._on_progress(self._job, step, n_steps, extra)

    def event(self, event: dict, key: tuple | None = None) -> None:
        """種類に固有のイベント (スイープのケースの完了など)。key があれば最新だけ残す。"""
        self._m._publish({**event, "id": self._job.id}, None if key is None else (self._job.id, *key))


class JobManager:
    def __init__(
        self,
        runners: Iterable[Runner],
        *,
        ctx: Any = None,
        max_running: int = 4,
        per_kind: dict[str, int] | None = None,
        default_per_kind: int = 2,
        keep_finished: int = 50,
        keep_sims_per_kind: int = 2,
        progress_interval: float = 0.1,
        frame_interval: float = 0.05,
    ) -> None:
        self._runners: dict[str, Runner] = {r.kind: r for r in runners}
        self.ctx = ctx
        self.max_running = max(1, int(max_running))
        self.per_kind = dict(per_kind or {})
        self.default_per_kind = max(1, int(default_per_kind))
        self.keep_finished = max(1, int(keep_finished))
        self.keep_sims_per_kind = max(0, int(keep_sims_per_kind))
        self.progress_interval = max(0.0, float(progress_interval))
        self.frame_interval = max(0.0, float(frame_interval))
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._queue: deque[str] = deque()
        self._subs: set[Subscriber] = set()
        self._seq: dict[str, itertools.count] = {}

    # ---- 読み出し ----

    @property
    def kinds(self) -> list[str]:
        return list(self._runners)

    def limits(self) -> dict:
        return {
            "max_running": self.max_running,
            "default_per_kind": self.default_per_kind,
            "per_kind": {k: self.kind_limit(k) for k in self._runners},
        }

    def set_limits(self, max_running: int | None = None, per_kind: dict[str, int] | None = None) -> dict:
        with self._lock:
            if max_running is not None:
                self.max_running = max(1, int(max_running))
            for k, v in (per_kind or {}).items():
                if k not in self._runners:
                    raise JobError(400, f"不明な種類です: {k}")
                self.per_kind[k] = max(1, int(v))
        self._schedule()
        return self.limits()

    def runner(self, kind: str) -> Runner:
        return self._runners[kind]

    def kind_limit(self, kind: str) -> int:
        return self.per_kind.get(kind, self.default_per_kind)

    def get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobError(404, f"ジョブ {job_id} がありません")
        return job

    def summary(self, job: Job) -> dict:
        with self._lock:
            pos = self._queue.index(job.id) if job.id in self._queue else None
        return job.summary(self._runners[job.kind].continuable, pos)

    def list(self) -> list[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [self.summary(j) for j in jobs]

    # ---- 操作 ----

    def submit(self, kind: str, project: dict | None, options: dict | None = None, label: str | None = None) -> Job:
        runner = self._runners.get(kind)
        if runner is None:
            raise JobError(400, f"不明な種類です: {kind}")
        options = dict(options or {})
        try:
            parsed = runner.validate(project, options)
        except JobError:
            raise
        except Exception as exc:  # pydantic の検証・設定の不足
            raise JobError(422, str(exc)) from exc
        with self._lock:
            counter = self._seq.setdefault(kind, itertools.count(1))
            job = Job(
                id=uuid.uuid4().hex[:12],
                kind=kind,
                seq=next(counter),
                label=label,
                project=project,
                options=options,
                parsed=parsed,
                created=time.time(),
            )
            self._jobs[job.id] = job
            self._queue.append(job.id)
        self._publish_job(job)
        self._schedule()
        return job

    def stop(self, job_id: str) -> dict:
        with self._lock:
            job = self.get(job_id)
            if job.state == "queued":
                self._queue.remove(job.id)
                # 続きを待っていたなら元の状態へ、まだ一度も走っていなければ取り消し
                job.state = job.prev_state or "cancelled"
                job.pending = None
                if job.state == "cancelled":
                    job.finished_at = time.time()
            elif job.state == "running":
                job.stopping = True
                job.stop_event.set()
            else:
                return self.summary(job)
        self._publish_job(job)
        self._schedule()
        return self.summary(job)

    def continue_(self, job_id: str, options: dict | None = None) -> dict:
        with self._lock:
            job = self.get(job_id)
            runner = self._runners[job.kind]
            if not runner.continuable:
                raise JobError(409, "この計算は続きを実行できません")
            if job.state not in ("done", "stopped") or job.state_obj is None:
                raise JobError(409, "続きを実行できる状態ではありません (完了または停止したジョブで、状態が残っているときだけ)")
            job.prev_state = job.state
            job.pending = ("continue", dict(options or {}))
            job.state = "queued"
            job.stopping = False
            job.error = None
            job.stop_event = threading.Event()
            self._queue.append(job.id)
        self._publish_job(job)
        self._schedule()
        return self.summary(job)

    def delete(self, job_id: str) -> None:
        with self._lock:
            job = self.get(job_id)
            del self._jobs[job.id]
            if job.id in self._queue:
                self._queue.remove(job.id)
            job.deleted = True
            running = job.state == "running"
            if running:
                job.stop_event.set()  # スレッドが終わるときに状態を捨てる
            state_obj, job.state_obj = (None, job.state_obj) if running else (job.state_obj, None)
        if not running:
            self._release(job.kind, state_obj)
        self._publish({"type": "removed", "id": job.id})
        self._schedule()

    def shutdown(self) -> None:
        """サーバーの終了: 待ちを取り消し、走っているものを止める。"""
        with self._lock:
            self._queue.clear()
            for job in self._jobs.values():
                job.stop_event.set()

    # ---- 購読 ----

    def subscribe(self, sub: Subscriber) -> Callable[[], None]:
        with self._lock:
            self._subs.add(sub)

        def unsubscribe() -> None:
            with self._lock:
                self._subs.discard(sub)

        return unsubscribe

    def watch(self, sub: Subscriber, ids: Iterable[str]) -> None:
        """sub がライブ表示するジョブ。新しく見始めたジョブの重い started と最新のフレームをすぐ送る。"""
        ids = set(ids)
        with self._lock:
            new = ids - sub.watch
            sub.watch = ids
            jobs = [self._jobs[i] for i in new if i in self._jobs]
        for job in jobs:
            if job.started is not None:
                sub.push({"type": "started", "id": job.id, "run": job.runs, "full": True, **job.started})
            if job.frame is not None and job.state == "running":
                sub.push({"type": "frame", "id": job.id, **job.frame}, (job.id, "frame"))

    def _publish(self, event: dict, key: tuple | None = None, watchers_only: str | None = None) -> None:
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            if watchers_only is not None and watchers_only not in sub.watch:
                continue
            try:
                sub.push(event, key)
            except Exception:  # 閉じた接続など
                log.debug("dropping subscriber", exc_info=True)
                with self._lock:
                    self._subs.discard(sub)

    def _publish_job(self, job: Job) -> None:
        if job.deleted:
            return
        self._publish({"type": "job", "job": self.summary(job)})

    # ---- 実行 ----

    def _schedule(self) -> None:
        to_start: list[Job] = []
        with self._lock:
            running: dict[str, int] = {}
            for j in self._jobs.values():
                if j.state == "running":
                    running[j.kind] = running.get(j.kind, 0) + 1
            total = sum(running.values())
            for jid in list(self._queue):
                if total >= self.max_running:
                    break
                job = self._jobs[jid]
                if running.get(job.kind, 0) >= self.kind_limit(job.kind):
                    continue
                self._queue.remove(jid)
                job.state = "running"
                job.stopping = False
                job.run_t0 = time.perf_counter()
                if job.started_at is None:
                    job.started_at = time.time()
                job.finished_at = None
                job.runs += 1
                running[job.kind] = running.get(job.kind, 0) + 1
                total += 1
                to_start.append(job)
        for job in to_start:
            self._publish_job(job)
            threading.Thread(target=self._run, args=(job,), name=f"job-{job.kind}-{job.seq}", daemon=True).start()
        if to_start:
            # 待っているジョブの順番が変わった
            with self._lock:
                waiting = [self._jobs[i] for i in self._queue]
            for job in waiting:
                self._publish_job(job)

    def _run(self, job: Job) -> None:
        runner = self._runners[job.kind]
        action, opts = job.pending or ("start", {})
        job.pending = None
        emit = Emitter(self, job)
        state = job.state_obj
        try:
            if action == "start":
                state = runner.build(job.parsed, job.options, self.ctx)
                with self._lock:
                    job.state_obj = state
            else:
                runner.prepare_continue(state, opts)
            started = runner.started(state)
            job.started = started
            job.progress = None
            job.frame = None
            job.progress_sent = job.frame_sent = 0.0
            job.frame_pending = False
            light = {k: v for k, v in started.items() if k not in HEAVY_STARTED_KEYS}
            self._publish({"type": "started", "id": job.id, "run": job.runs, "full": False, **light})
            if any(k in started for k in HEAVY_STARTED_KEYS):
                self._publish({"type": "started", "id": job.id, "run": job.runs, "full": True, **started}, watchers_only=job.id)
            result = runner.run(state, emit, job.stop_event.is_set)
            if job.frame_pending and job.frame is not None:
                # 間引いて送らなかった最後のフレーム
                self._publish({"type": "frame", "id": job.id, **job.frame}, (job.id, "frame"), watchers_only=job.id)
            with self._lock:
                job.result = result
                job.state = "stopped" if job.stop_event.is_set() else "done"
                self._finish(job)
            self._drop_state_if_unneeded(job, runner)
            # 要約 (結果あり・続けられるか) を先に送ってから完了を知らせる (受け手はすぐ結果を取りに来る)
            self._publish_job(job)
            self._publish({"type": "done", "id": job.id, "state": job.state, "elapsed_s": job.elapsed_s})
        except Exception as exc:
            log.warning("job %s (%s) failed", job.id, job.kind, exc_info=True)
            with self._lock:
                job.state = "error"
                job.error = str(exc) or type(exc).__name__
                # 途中で失敗した状態は続きに使えない
                state, job.state_obj = job.state_obj, None
                self._finish(job)
            self._release(job.kind, state)
            self._publish_job(job)
            self._publish({"type": "error", "id": job.id, "detail": job.error})
        finally:
            self._drop_state_if_unneeded(job, runner)
            self._evict()
            self._schedule()

    def _drop_state_if_unneeded(self, job: Job, runner: Runner) -> None:
        """消されたジョブと、続けられず持っておく必要も無い種類の状態を捨てる。"""
        with self._lock:
            drop = None
            if job.deleted or not (runner.continuable or getattr(runner, "keep_state", False)):
                drop, job.state_obj = job.state_obj, None
        self._release(job.kind, drop)

    def _finish(self, job: Job) -> None:
        if job.run_t0 is not None:
            job.elapsed_s += time.perf_counter() - job.run_t0
        job.run_t0 = None
        job.stopping = False
        job.prev_state = None
        job.finished_at = time.time()

    def _on_frame(self, job: Job, frame: dict) -> None:
        job.frame = frame
        now = time.perf_counter()
        if now - job.frame_sent >= self.frame_interval:
            job.frame_sent = now
            job.frame_pending = False
            self._publish({"type": "frame", "id": job.id, **frame}, (job.id, "frame"), watchers_only=job.id)
        else:
            job.frame_pending = True
        step = frame.get("step")
        if isinstance(step, (int, float)):
            n_steps = frame.get("n_steps") or (job.started or {}).get("n_steps") or 0
            self._on_progress(job, int(step), int(n_steps), {})

    def _on_progress(self, job: Job, step: int, n_steps: int, extra: dict) -> None:
        offset = int((job.started or {}).get("step_offset") or 0)
        done = step - offset if step >= offset else step
        frac = min(1.0, max(0.0, done / n_steps)) if n_steps > 0 else None
        job.progress = {"step": step, "n_steps": n_steps, "step_offset": offset, "fraction": frac, **extra}
        now = time.perf_counter()
        # 最後の進捗はジョブの要約 (完了の通知) に載るので、間引いて落としてよい
        if now - job.progress_sent >= self.progress_interval or (n_steps > 0 and done >= n_steps):
            job.progress_sent = now
            self._publish({"type": "progress", "id": job.id, **job.progress}, (job.id, "progress"))

    def _release(self, kind: str, state: Any) -> None:
        if state is None:
            return
        try:
            self._runners[kind].release(state)
        except Exception:
            log.warning("release failed (%s)", kind, exc_info=True)

    def _evict(self) -> None:
        """終わったジョブが多すぎたら古いものを消し、続きの状態は種類ごとに新しい方だけ残す。"""
        removed: list[Job] = []
        dropped: list[tuple[Job, Any]] = []
        with self._lock:
            finished = sorted((j for j in self._jobs.values() if j.state in TERMINAL), key=lambda j: j.finished_at or 0.0)
            while len(finished) > self.keep_finished:
                job = finished.pop(0)
                del self._jobs[job.id]
                job.deleted = True
                removed.append(job)
            by_kind: dict[str, list[Job]] = {}
            for j in finished:
                if j.state_obj is not None and self._runners[j.kind].continuable:
                    by_kind.setdefault(j.kind, []).append(j)
            for jobs in by_kind.values():
                for job in jobs[: max(0, len(jobs) - self.keep_sims_per_kind)]:
                    dropped.append((job, job.state_obj))
                    job.state_obj = None
            for job in removed:
                if job.state_obj is not None:
                    dropped.append((job, job.state_obj))
                    job.state_obj = None
        for job, state in dropped:
            self._release(job.kind, state)
            self._publish_job(job)
        for job in removed:
            self._publish({"type": "removed", "id": job.id})
