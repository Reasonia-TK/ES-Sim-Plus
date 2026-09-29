"""UI v2 のジョブ (/v2/jobs・/v2/events、prompts/130 P6d)。

1. JobManager (種類を知らない部分): 状態の遷移とイベント、待ち行列と同時実行の上限、停止 (実行中・待ち)、続き、
   失敗、削除、古いジョブ・続きの状態の整理、watch したジョブだけへのフレーム、購読者の「最新だけ残す」。
   制御できる偽のソルバー (FakeRunner) で決定的に確かめる。
2. API: 本物のソルバー (縮小ケース) で、同じソルバーの同時実行・続き・停止・イベントの WebSocket・誤りの応答・
   粒子軌道 (REST の /trace と同じ結果)・DSMC → PIC (use_dsmc_gas + dsmc_job)・スイープのケースの結果。
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from fastapi.testclient import TestClient

import es_sim.server as server
from es_sim.jobs import JobError, JobManager
from es_sim.jobs.api import AsyncSubscriber, dumps


# ---- 偽のソルバー --------------------------------------------------------------------


class FakeRunner:
    kind = "fake"
    continuable = True
    keep_state = False

    def __init__(self, kind: str = "fake", continuable: bool = True) -> None:
        self.kind = kind
        self.continuable = continuable
        self.gate = threading.Event()
        self.gate.set()
        self.released: list[dict] = []

    def validate(self, project, options):
        if options.get("invalid"):
            raise ValueError("設定が足りません")
        return dict(options)

    def build(self, parsed, options, ctx):
        if options.get("fail_build"):
            raise RuntimeError("組み立てに失敗")
        return {"steps": 0, "n": int(options.get("n", 5))}

    def started(self, st):
        return {"n_steps": st["n"], "step_offset": st["steps"], "mesh": {"nodes": [[0, 0]]}, "warnings": ["w"]}

    def run(self, st, emit, should_stop):
        for _ in range(st["n"]):
            if should_stop():
                break
            assert self.gate.wait(10)
            st["steps"] += 1
            emit.frame({"step": st["steps"]})
        return {"steps": st["steps"]}

    def prepare_continue(self, st, options):
        st["n"] = int(options.get("steps", st["n"]))

    def release(self, st):
        self.released.append(st)


class ListSub:
    def __init__(self, watch: set[str] | None = None) -> None:
        self.watch = set(watch or ())
        self.events: list[dict] = []
        self._lock = threading.Lock()

    def push(self, event, key=None):
        with self._lock:
            self.events.append(event)

    def of(self, job_id: str, type_: str | None = None) -> list[dict]:
        with self._lock:
            return [
                e
                for e in self.events
                if (e.get("id") == job_id or e.get("job", {}).get("id") == job_id) and (type_ is None or e["type"] == type_)
            ]


def wait_for(pred, timeout: float = 20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = pred()
        if v:
            return v
        time.sleep(0.01)
    raise AssertionError("timeout")


def _state(m: JobManager, job_id: str) -> str:
    return m.summary(m.get(job_id))["state"]


# ---- 1. JobManager ---------------------------------------------------------------------


def test_lifecycle_events_and_summary():
    r = FakeRunner()
    m = JobManager([r])
    sub = ListSub()
    m.subscribe(sub)
    job = m.submit("fake", None, {"n": 4}, label="テスト")
    wait_for(lambda: _state(m, job.id) == "done")
    wait_for(lambda: sub.of(job.id, "job") and sub.of(job.id, "job")[-1]["job"]["state"] == "done")
    states = [e["job"]["state"] for e in sub.of(job.id, "job")]
    assert states[0] == "queued" and "running" in states and states[-1] == "done"
    started = sub.of(job.id, "started")
    # 見ていない購読者には重い項目 (メッシュ) を送らない
    assert len(started) == 1 and "mesh" not in started[0] and started[0]["n_steps"] == 4
    assert not sub.of(job.id, "frame")
    prog = sub.of(job.id, "progress")
    assert prog[-1]["step"] == 4 and prog[-1]["fraction"] == 1.0
    assert sub.of(job.id, "done")[0]["state"] == "done"
    s = m.summary(m.get(job.id))
    assert s["kind"] == "fake" and s["seq"] == 1 and s["label"] == "テスト"
    assert s["runs"] == 1 and s["has_result"] and s["can_continue"] and s["warnings"] == ["w"]
    assert s["elapsed_s"] >= 0 and s["started_at"] is not None and s["finished_at"] is not None
    assert m.get(job.id).result == {"steps": 4}
    assert m.submit("fake", None, {}).seq == 2


def test_queue_and_limits():
    a, b = FakeRunner("a"), FakeRunner("b")
    a.gate.clear()
    m = JobManager([a, b], max_running=2, per_kind={"a": 1})
    j1 = m.submit("a", None, {"n": 2})
    j2 = m.submit("a", None, {"n": 2})
    j3 = m.submit("b", None, {"n": 1})
    wait_for(lambda: _state(m, j1.id) == "running")
    wait_for(lambda: _state(m, j3.id) == "done")
    # 種類 a は 1 本まで: j2 は待つ (全体の上限 2 には達していない)
    assert _state(m, j2.id) == "queued"
    assert m.summary(m.get(j2.id))["queue_position"] == 0
    a.gate.set()
    wait_for(lambda: _state(m, j2.id) == "done")
    assert m.get(j1.id).finished_at <= m.get(j2.id).started_at + 1e-6


def test_stop_running_and_queued():
    r = FakeRunner()
    r.gate.clear()
    m = JobManager([r], max_running=1)
    j1 = m.submit("fake", None, {"n": 100})
    j2 = m.submit("fake", None, {"n": 1})
    wait_for(lambda: _state(m, j1.id) == "running")
    assert m.stop(j2.id)["state"] == "cancelled"
    s = m.stop(j1.id)
    assert s["state"] == "running" and s["stopping"]
    r.gate.set()
    wait_for(lambda: _state(m, j1.id) == "stopped")
    # 止めたジョブにも途中までの結果がある (v1 と同じ)
    assert m.get(j1.id).result["steps"] < 100
    assert _state(m, j2.id) == "cancelled" and m.get(j2.id).runs == 0


def test_continue_keeps_state_and_counts_runs():
    r = FakeRunner()
    m = JobManager([r])
    job = m.submit("fake", None, {"n": 5})
    wait_for(lambda: _state(m, job.id) == "done")
    r.gate.clear()
    s = m.continue_(job.id, {"steps": 3})
    assert s["state"] in ("queued", "running")
    r.gate.set()
    wait_for(lambda: _state(m, job.id) == "done" and m.get(job.id).runs == 2)
    assert m.get(job.id).result == {"steps": 8}
    assert m.get(job.id).started["step_offset"] == 5
    # 続けられない種類・実行中は 409
    other = FakeRunner("once", continuable=False)
    m2 = JobManager([other])
    j = m2.submit("once", None, {"n": 1})
    wait_for(lambda: _state(m2, j.id) == "done")
    assert not m2.summary(m2.get(j.id))["can_continue"]
    with pytest.raises(JobError) as e:
        m2.continue_(j.id, {})
    assert e.value.status == 409
    assert other.released  # 続けられない種類は終わったら状態を捨てる
    r.gate.clear()
    j3 = m.submit("fake", None, {"n": 3})
    wait_for(lambda: _state(m, j3.id) == "running")
    with pytest.raises(JobError):
        m.continue_(j3.id, {})
    # 待ちの続きを止めると元の状態に戻る
    r.gate.set()
    wait_for(lambda: _state(m, j3.id) == "done")


def test_continue_can_be_cancelled_while_queued():
    r = FakeRunner()
    m = JobManager([r], max_running=1)
    done = m.submit("fake", None, {"n": 1})
    wait_for(lambda: _state(m, done.id) == "done")
    r.gate.clear()
    blocker = m.submit("fake", None, {"n": 5})
    wait_for(lambda: _state(m, blocker.id) == "running")
    assert m.continue_(done.id, {"steps": 1})["state"] == "queued"
    assert m.stop(done.id)["state"] == "done"
    r.gate.set()
    wait_for(lambda: _state(m, blocker.id) == "done")
    assert m.get(done.id).runs == 1


def test_failures_and_validation():
    r = FakeRunner()
    m = JobManager([r])
    sub = ListSub()
    m.subscribe(sub)
    job = m.submit("fake", None, {"fail_build": True})
    wait_for(lambda: _state(m, job.id) == "error")
    s = m.summary(m.get(job.id))
    assert s["error"] == "組み立てに失敗" and not s["can_continue"] and not s["has_result"]
    assert sub.of(job.id, "error")[0]["detail"] == "組み立てに失敗"
    with pytest.raises(JobError) as e:
        m.submit("fake", None, {"invalid": True})
    assert e.value.status == 422 and "足りません" in e.value.detail
    with pytest.raises(JobError) as e:
        m.submit("nope", None, {})
    assert e.value.status == 400
    with pytest.raises(JobError) as e:
        m.get("missing")
    assert e.value.status == 404


def test_delete_running_job_releases_state():
    r = FakeRunner()
    r.gate.clear()
    m = JobManager([r])
    sub = ListSub()
    m.subscribe(sub)
    job = m.submit("fake", None, {"n": 3})
    wait_for(lambda: _state(m, job.id) == "running")
    m.delete(job.id)
    assert [e for e in sub.events if e["type"] == "removed"] == [{"type": "removed", "id": job.id}]
    assert all(s["id"] != job.id for s in m.list())
    r.gate.set()
    wait_for(lambda: len(r.released) == 1)
    # 消えたジョブの状態の通知は出さない
    time.sleep(0.05)
    assert all(e["job"]["state"] != "done" for e in sub.of(job.id, "job"))


def test_eviction_of_old_jobs_and_simulation_states():
    r = FakeRunner()
    m = JobManager([r], keep_finished=3, keep_sims_per_kind=1)
    ids = []
    for _ in range(4):
        j = m.submit("fake", None, {"n": 1})
        wait_for(lambda: _state(m, j.id) == "done")
        ids.append(j.id)
    listed = [s["id"] for s in m.list()]
    assert ids[0] not in listed and listed == ids[1:]
    # 続きの状態は新しい 1 件だけ
    assert [m.summary(m.get(i))["can_continue"] for i in ids[1:]] == [False, False, True]
    assert len(r.released) == 3


def test_frames_only_for_watchers_and_late_watch():
    r = FakeRunner()
    r.gate.clear()
    m = JobManager([r])
    plain, watcher = ListSub(), ListSub()
    m.subscribe(plain)
    m.subscribe(watcher)
    job = m.submit("fake", None, {"n": 3})
    wait_for(lambda: _state(m, job.id) == "running")
    wait_for(lambda: m.get(job.id).started is not None)
    m.watch(watcher, [job.id])
    # 見始めたらすぐ重い started が届く
    full = [e for e in watcher.of(job.id, "started") if e["full"]]
    assert full and full[0]["mesh"] == {"nodes": [[0, 0]]}
    r.gate.set()
    wait_for(lambda: _state(m, job.id) == "done")
    assert watcher.of(job.id, "frame") and not plain.of(job.id, "frame")
    assert plain.of(job.id, "progress")


def test_progress_and_frames_are_rate_limited_but_end_complete():
    """速いソルバー (毎秒数百回の知らせ) でも送る数は間引き、最後の進捗とフレームは必ず届く。"""
    r = FakeRunner()
    m = JobManager([r], progress_interval=10.0, frame_interval=10.0)
    plain, watcher = ListSub(), ListSub()
    m.subscribe(plain)
    m.subscribe(watcher)
    r.gate.clear()
    job = m.submit("fake", None, {"n": 200})
    watcher.watch = {job.id}
    r.gate.set()
    wait_for(lambda: _state(m, job.id) == "done")
    prog = plain.of(job.id, "progress")
    assert 1 <= len(prog) <= 3 and prog[-1]["step"] == 200
    frames = watcher.of(job.id, "frame")
    assert 1 <= len(frames) <= 3 and frames[-1]["step"] == 200


def test_async_subscriber_keeps_latest_per_key_in_order():
    async def main():
        sub = AsyncSubscriber(asyncio.get_running_loop())
        sub.push({"type": "job", "n": 1})
        for i in range(5):
            sub.push({"type": "progress", "step": i}, ("j", "progress"))
        sub.push({"type": "done"})
        await asyncio.sleep(0)
        first = [await asyncio.wait_for(sub.next(), 1) for _ in range(3)]
        # まだ送っていない progress は、後から来たものに置き換わる (位置は最初のまま)
        sub.push({"type": "progress", "step": 7}, ("j", "progress"))
        sub.push({"type": "job", "n": 2})
        sub.push({"type": "progress", "step": 8}, ("j", "progress"))
        await asyncio.sleep(0)
        second = [await asyncio.wait_for(sub.next(), 1) for _ in range(2)]
        return first, second

    first, second = asyncio.run(main())
    assert first == [{"type": "job", "n": 1}, {"type": "progress", "step": 4}, {"type": "done"}]
    assert second == [{"type": "progress", "step": 8}, {"type": "job", "n": 2}]


def test_dumps_replaces_nan_with_null():
    assert dumps({"a": [1.0, float("nan"), float("inf")], "b": "x"}) == '{"a":[1.0,null,null],"b":"x"}'


# ---- 2. API (本物のソルバー、縮小ケース) --------------------------------------------------

DUMMY = {"geometry": {"domain": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}}, "mesh": {"size": 0.1}}


def _pic1d(init_density: float = 1.0e14, n_steps: int = 60) -> dict:
    return DUMMY | {
        "pic1d": {
            "gap_m": 0.02,
            "n_cells": 16,
            "init_density_m3": init_density,
            "n_macro": 400,
            "dt": 1.0e-10,
            "n_steps": n_steps,
            "frame_every": 20,
            "left": {"v_dc": 50.0},
            "right": {"v_dc": 0.0},
        }
    }


def _fluid1d(n_steps: int = 40) -> dict:
    return DUMMY | {
        "fluid1d": {
            "gap_m": 0.02,
            "n_cells": 16,
            "init_density_m3": 1.0e14,
            "gas_pressure_pa": 10.0,
            "n_steps": n_steps,
            "frame_every": 10,
            "avg_steps": 10,
            "left": {"v_dc": 50.0},
            "right": {"v_dc": 0.0},
        }
    }


def _wait_job(client: TestClient, job_id: str, states=("done",), timeout: float = 60.0) -> dict:
    def get():
        s = client.get(f"/v2/jobs/{job_id}").json()
        return s if s["state"] in states else None

    return wait_for(get, timeout)


def test_api_runs_the_same_solver_concurrently():
    client = TestClient(server.app)
    ids = []
    for dens in (1.0e14, 3.0e14):
        r = client.post("/v2/jobs", json={"kind": "pic1d", "project": _pic1d(dens)})
        assert r.status_code == 201, r.text
        ids.append(r.json()["id"])
    finals = [_wait_job(client, i) for i in ids]
    assert [f["kind"] for f in finals] == ["pic1d", "pic1d"]
    assert finals[0]["seq"] != finals[1]["seq"]
    res = [client.get(f"/v2/jobs/{i}/result").json() for i in ids]
    assert all("history" in r or "profiles" in r for r in res)
    # 別々の実行: 設定 (初期密度) がそれぞれの結果に残る
    assert res[0]["settings"]["init_density_m3"] == 1.0e14
    assert res[1]["settings"]["init_density_m3"] == 3.0e14
    started = client.get(f"/v2/jobs/{ids[0]}/started").json()
    assert len(started["x"]) == 17 and started["step_offset"] == 0
    listed = client.get("/v2/jobs").json()
    assert {"jobs", "limits", "kinds"} <= set(listed) and "sweep" in listed["kinds"]
    for i in ids:
        assert client.delete(f"/v2/jobs/{i}").status_code == 204


def test_api_continue_and_stop():
    client = TestClient(server.app)
    jid = client.post("/v2/jobs", json={"kind": "fluid1d", "project": _fluid1d(40)}).json()["id"]
    first = _wait_job(client, jid)
    assert first["can_continue"] and first["runs"] == 1
    r = client.post(f"/v2/jobs/{jid}/continue", json={"options": {"steps": 20}})
    assert r.status_code == 200, r.text
    second = wait_for(lambda: (s := client.get(f"/v2/jobs/{jid}").json())["state"] == "done" and s["runs"] == 2 and s)
    assert second["progress"]["step_offset"] == 40 and second["progress"]["step"] == 60
    assert client.get(f"/v2/jobs/{jid}/started").json()["step_offset"] == 40
    # 長い実行を止めると、途中までの結果が残る
    long_id = client.post("/v2/jobs", json={"kind": "pic1d", "project": _pic1d(n_steps=2_000_000)}).json()["id"]
    wait_for(lambda: (client.get(f"/v2/jobs/{long_id}").json().get("progress") or {}).get("step", 0) > 0)
    assert client.post(f"/v2/jobs/{long_id}/stop").json()["stopping"]
    stopped = _wait_job(client, long_id, ("stopped",))
    assert stopped["has_result"] and stopped["progress"]["step"] < 2_000_000
    for i in (jid, long_id):
        client.delete(f"/v2/jobs/{i}")


def test_api_errors():
    client = TestClient(server.app)
    assert client.post("/v2/jobs", json={"kind": "nope", "project": DUMMY}).status_code == 400
    r = client.post("/v2/jobs", json={"kind": "pic1d", "project": DUMMY})
    assert r.status_code == 422 and "pic1d" in r.json()["detail"]
    assert client.get("/v2/jobs/missing").status_code == 404
    assert client.get("/v2/jobs/missing/result").status_code == 404
    assert client.post("/v2/jobs/missing/stop").status_code == 404
    lim = client.get("/v2/jobs/limits").json()
    assert lim["max_running"] >= 1 and lim["per_kind"]["sweep"] >= 1
    assert client.put("/v2/jobs/limits", json={"per_kind": {"nope": 1}}).status_code == 400


def test_api_events_websocket():
    client = TestClient(server.app)
    with client.websocket_connect("/v2/events") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["instance"] == server.SERVER_INSTANCE and "kinds" in hello
        jid = client.post("/v2/jobs", json={"kind": "pic1d", "project": _pic1d(n_steps=200)}).json()["id"]
        ws.send_json({"cmd": "watch", "ids": [jid]})
        seen: list[dict] = []
        while True:
            ev = ws.receive_json()
            if ev.get("id") == jid or ev.get("job", {}).get("id") == jid:
                seen.append(ev)
            if ev["type"] == "done" and ev["id"] == jid:
                break
        types = [e["type"] for e in seen]
        assert "started" in types and "progress" in types and types[-1] == "done"
        # watch したのでフレームも届く
        assert "frame" in types
        # 完了の通知の前に、結果ありの要約が届いている (受け手はすぐ結果を取りに行ける)
        last_job = [e for e in seen if e["type"] == "job"][-1]["job"]
        assert last_job["state"] == "done" and last_job["has_result"]
        ws.send_json({"cmd": "ping"})
        while ws.receive_json()["type"] != "pong":
            pass
    client.delete(f"/v2/jobs/{jid}")


def test_api_trace_matches_rest_endpoint():
    project = {
        "geometry": {
            "domain": {"polygon": [[0, 0], [0.02, 0], [0.02, 0.01], [0, 0.01]]},
            "boundaries": [
                {"edges": [3], "type": "dirichlet", "voltage": 0.0},
                {"edges": [1], "type": "dirichlet", "voltage": 100.0},
            ],
        },
        "mesh": {"size": 0.002},
        "particles": {
            "species": {"preset": "electron"},
            "emitter": {"kind": "line", "p1": [0.002, 0.004], "p2": [0.002, 0.006], "n": 4, "energy_ev": 1.0, "direction_deg": 0.0, "spread_deg": 0.0},
            "dt": None,
            "n_steps": 400,
            "save_every": 20,
        },
    }
    client = TestClient(server.app)
    rest = client.post("/trace", json=project).json()
    jid = client.post("/v2/jobs", json={"kind": "trace", "project": project}).json()["id"]
    s = _wait_job(client, jid)
    assert not s["can_continue"]
    res = client.get(f"/v2/jobs/{jid}/result").json()
    assert res["status"] == rest["status"] and res["trajectories"] == rest["trajectories"]
    assert client.post(f"/v2/jobs/{jid}/continue", json={}).status_code == 409
    client.delete(f"/v2/jobs/{jid}")


def test_api_dsmc_job_feeds_pic_gas():
    L, H = 0.02, 0.01
    base = {
        "geometry": {
            "domain": {"polygon": [[0, 0], [L, 0], [L, H], [0, H]]},
            "boundaries": [
                {"edges": [3], "type": "dirichlet", "voltage": 0.0},
                {"edges": [1], "type": "dirichlet", "voltage": 0.0},
            ],
        },
        "mesh": {"size": 2.0e-3},
    }
    dsmc = base | {
        "dsmc": {
            "boundaries": [{"edges": [0], "type": "inlet", "pressure_pa": 10.0}, {"edges": [2], "type": "outlet", "pressure_pa": 3.0}],
            "init_pressure_pa": 6.0,
            "n_particles": 4000,
            "n_steps": 200,
            "avg_steps": 100,
            "seed": 4,
        }
    }
    client = TestClient(server.app)
    did = client.post("/v2/jobs", json={"kind": "dsmc", "project": dsmc}).json()["id"]
    _wait_job(client, did)
    res = client.get(f"/v2/jobs/{did}/result").json()
    assert len(res["n"]) == server._last_dsmc["n_elems"]
    pic = base | {
        "pic": {
            "initial_plasma": {"density": 1e14, "te_ev": 2.0, "ti_ev": 0.03, "ion_mass_amu": 40.0, "seed": 5},
            "n_macro": 500,
            "dt": 5e-11,
            "n_steps": 4,
            "frame_every": 2,
            "mcc": {
                "gas": {"name": "Ar", "pressure_pa": 6.0, "temperature_k": 300.0},
                "electron_processes": [
                    {"kind": "elastic", "label": "syn", "threshold_ev": 0.0, "mass_ratio": 1.36e-5, "energy_ev": [0.0, 100.0], "sigma_m2": [1e-19, 1e-19]}
                ],
                "seed": 7,
                "use_dsmc_gas": True,
            },
        }
    }
    pid = client.post("/v2/jobs", json={"kind": "pic", "project": pic, "options": {"dsmc_job": did}}).json()["id"]
    s = _wait_job(client, pid, ("done", "error"))
    assert s["state"] == "done", s["error"]
    result = client.get(f"/v2/jobs/{pid}/result").json()
    assert {"started", "history", "fields", "collectors", "eedf", "elapsed_s", "frame", "timing"} <= set(result)
    assert len(result["started"]["mesh"]["triangles"]) == len(res["n"])
    for i in (did, pid):
        client.delete(f"/v2/jobs/{i}")


def test_api_sweep_job_case_results():
    client = TestClient(server.app)
    body = {
        "kind": "sweep",
        "project": _pic1d(n_steps=40),
        "options": {"param_path": "pic1d.init_density_m3", "values": [1.0e14, 2.0e14], "parallel": 2},
    }
    jid = client.post("/v2/jobs", json=body).json()["id"]
    _wait_job(client, jid, timeout=180)
    res = client.get(f"/v2/jobs/{jid}/result").json()
    assert res["module"] == "pic1d" and [c["ok"] for c in res["summary"]] == [True, True]
    case1 = client.get(f"/v2/jobs/{jid}/cases/1").json()
    assert case1["pic1d"]["init_density_m3"] == 2.0e14
    assert client.get(f"/v2/jobs/{jid}/cases/5").status_code == 404
    client.delete(f"/v2/jobs/{jid}")
    assert client.get(f"/v2/jobs/{jid}/cases/0").status_code == 404
