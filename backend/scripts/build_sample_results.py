"""サンプルの保存された計算結果 (examples/results/<キー>.json.gz) を作る (prompts/135)。

サンプル (examples/<キー>.json) の設定のまま、ジョブと同じ組み立て (流体 2D: make_fluid2d_simulation と
Fluid2dRunner.result、PIC: make_pic_simulation と batch._build_results_bundle) で走らせ、PLANS の周期数まで
「続き」で延ばして、結果の束 {version: 1, meta, <種類>: 結果} を gzip で書く。UI はサンプルを開いたときに
これを「読み込んだ実行」として並べる (ui/src/io/exampleResults.ts)。

- 時間平均の場・位相分解は最後の区間のもの (平均の長さは PLANS の周期数、None ならサンプルの avg_steps)。
- 診断の履歴は MAX_HISTORY 点に間引く。阻止コンデンサの自己バイアスの履歴 (RF 1 周期ごと) は続きをまたいで全部残す。
- 数値は有効数字 7 桁 (float32 相当) に丸める。大きな配列は表示に足りる桁まで減らす (DIGITS_BY_PART: 時間平均の場・
  コレクタは 4 桁、位相分解は 3 桁。時刻のように差の小さい値を並べる履歴・回路・メッシュは 7 桁のまま)。
- GPU の計算は実行ごとにわずかに違う (atomic) ので、作り直すと中身は変わる。エンジンを大きく変えたときに作り直す。

使い方 (backend で): .venv/Scripts/python.exe scripts/build_sample_results.py [キー ...]   (省くと PLANS の全て)
  --repack: 計算し直さず、今のファイルを今の桁数で丸め直して書き直す (桁数の方針を変えたとき)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from es_sim import __version__
from es_sim.batch import _build_results_bundle
from es_sim.circuit import rf_period
from es_sim.gfluid import make_fluid2d_simulation
from es_sim.gpic import make_pic_simulation
from es_sim.jobs.runners import Fluid2dRunner
from es_sim.schema import Project

EXAMPLES = ROOT / "examples"
OUT_DIR = EXAMPLES / "results"
#: 診断の履歴の点数の上限
MAX_HISTORY = 2000
#: 有効数字 (既定) と、結果の項目ごとの有効数字 (大きな配列)
DIGITS = 7
DIGITS_BY_PART = {"fields": 4, "collectors": 4, "cycle": 3}
#: サンプルごとの (種類, 合計の RF 周期数, 最後に時間平均する RF 周期数)。合計の None はサンプルの n_steps の
#: まま (延ばさない)、平均の None はサンプルの avg_steps。続きの 1 回の長さはサンプルの n_steps
PLANS: dict[str, list[tuple[str, int | None, int | None]]] = {
    "gec_cell": [("fluid2d", 100, None), ("pic", 600, 50)],
    "ccp_demo": [("pic", None, None)],
}


def _round_floats(obj, digits: int = DIGITS):
    """数値を有効数字 digits 桁に丸める (JSON を小さくする)。"""
    if isinstance(obj, float):
        return float(f"{obj:.{digits}g}") if math.isfinite(obj) else obj
    if isinstance(obj, list):
        return [_round_floats(v, digits) for v in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, digits) for k, v in obj.items()}
    return obj


def _round_bundle(bundle: dict) -> dict:
    """結果の束を丸める (種類ごとの結果は項目ごとに DIGITS_BY_PART の桁、ほかは DIGITS 桁)。"""
    out = {}
    for key, value in bundle.items():
        if key in ("version", "meta") or not isinstance(value, dict):
            out[key] = _round_floats(value)
        else:
            out[key] = {k: _round_floats(v, DIGITS_BY_PART.get(k, DIGITS)) for k, v in value.items()}
    return out


def _write(key: str, bundle: dict, log) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    raw = json.dumps(_round_bundle(bundle), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    path = OUT_DIR / f"{key}.json.gz"
    with gzip.GzipFile(path, "wb", compresslevel=9, mtime=0) as fh:
        fh.write(raw)
    log(f"{key}: {path.name} ({len(raw) / 1e6:.1f} MB → gzip {path.stat().st_size / 1e6:.1f} MB)")
    return path


def _thin(seq: list, n: int) -> list:
    """seq を最後の点を含めて n 点ほどに間引く。"""
    if len(seq) <= n:
        return seq
    stride = math.ceil(len(seq) / n)
    out = seq[::stride]
    if out[-1] is not seq[-1]:
        out.append(seq[-1])
    return out


def _thin_history(hist):
    """診断の履歴を間引く (PIC は行の列、流体は列の辞書)。"""
    if isinstance(hist, list):
        return _thin(hist, MAX_HISTORY)
    if isinstance(hist, dict):
        n = max((len(v) for v in hist.values() if isinstance(v, list)), default=0)
        if n <= MAX_HISTORY:
            return hist
        stride = math.ceil(n / MAX_HISTORY)
        idx = list(range(0, n, stride))
        if idx[-1] != n - 1:
            idx.append(n - 1)
        return {k: ([v[i] for i in idx] if isinstance(v, list) and len(v) == n else v) for k, v in hist.items()}
    return hist


def _steps_per_period(project: Project, dt: float) -> int | None:
    period = rf_period(project)
    return None if period is None else max(1, round(period / dt))


def _chunks(total: int, chunk: int, final: int) -> list[int]:
    """total ステップを chunk ごとに分け、最後を final ステップ (時間平均の区間を含む長さ) にする。"""
    final = min(total, final)
    rest = total - final
    sizes = []
    while rest > 0:
        sizes.append(min(chunk, rest))
        rest -= sizes[-1]
    return sizes + [final]


def _run(kind: str, project: Project, periods: int | None, avg_periods: int | None, log) -> tuple[dict, dict]:
    """1 つの種類を走らせて (結果, 記録) を返す。"""
    if kind == "fluid2d":
        sim = make_fluid2d_simulation(project)
        settings = sim.s
    elif kind == "pic":
        sim = make_pic_simulation(project)
        settings = sim.pic
    else:
        raise ValueError(f"未対応の種類: {kind}")
    chunk = int(settings.n_steps)
    spp = _steps_per_period(project, sim.dt)
    total = chunk if periods is None or spp is None else max(chunk, periods * spp)
    sample_avg = settings.avg_steps
    avg = None if avg_periods is None or spp is None else avg_periods * spp
    sizes = _chunks(total, chunk, max(chunk, avg or 0))
    acc: dict[str, list] = {"t": [], "v_dc": [], "v1": [], "i_dc": []}
    t0 = time.perf_counter()
    done = 0
    step0 = sim.step_count
    for i, n in enumerate(sizes):
        if avg is not None and i == len(sizes) - 1:
            n_avg = avg
        else:
            n_avg = None if sample_avg is None else min(int(sample_avg), n)
        if i:
            if sim.circuit is not None:
                for k, rows in acc.items():
                    rows.extend(sim.circuit.history[k])
            step0 = sim.step_count
            sim.prepare_continue(n, avg_steps=n_avg)
        else:
            settings.n_steps = n
            if n_avg is not None:
                settings.avg_steps = n_avg
        sim.run_batch(store_frames=False)
        done += n
        msg = f"  {kind}: {done}/{total} ステップ ({time.perf_counter() - t0:.0f} 秒)"
        if sim.circuit is not None and sim.circuit.history["v_dc"]:
            msg += f"、V_dc {sim.circuit.history['v_dc'][-1]}"
        log(msg)
    elapsed = time.perf_counter() - t0
    if kind == "fluid2d":
        result = Fluid2dRunner().result(sim, elapsed)
    else:
        result = _build_results_bundle(sim, step0, elapsed)["pic"]
    result["history"] = _thin_history(result.get("history"))
    circuit = result.get("circuit")
    if circuit:
        h = sim.circuit.history
        for j, e in enumerate(circuit["electrodes"]):
            e["t"] = acc["t"] + list(h["t"])
            for k in ("v_dc", "v1", "i_dc"):
                e[k] = [row[j] for row in acc[k] + list(h[k])]
    record = {"engine": _engine(sim), "steps": int(done), "periods": None if spp is None else round(done / spp, 2),
              "elapsed_s": round(elapsed, 1)}
    return result, record


def _engine(sim) -> str:
    """計算したエンジンと装置 (例: "GpuPicSimulation (NVIDIA GeForce RTX 5070 Ti)"、"PicSimulation (CPU)")。"""
    dev = getattr(sim, "device", None) or getattr(sim, "_device", None)
    if dev is not None and getattr(dev, "is_gpu", False):
        import cupy as cp

        where = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    else:
        where = "CPU"
    return f"{type(sim).__name__} ({where})"


def build(key: str, log=print) -> Path:
    plan = PLANS[key]
    data = json.loads((EXAMPLES / f"{key}.json").read_text(encoding="utf-8"))
    Project.model_validate(data)  # 長い計算の前に設定を確かめる (種類ごとに新しく組み立てる)
    bundle: dict = {"version": 1}
    runs: dict[str, dict] = {}
    for kind, periods, avg_periods in plan:
        log(f"{key}: {kind} ({'サンプルのまま' if periods is None else f'{periods} 周期'}"
            f"{'' if avg_periods is None else f'、最後の {avg_periods} 周期を時間平均'})")
        result, record = _run(kind, Project.model_validate(data), periods, avg_periods, log)
        bundle[kind] = result
        runs[kind] = record
    bundle["meta"] = {
        "sample": key,
        "generated": _dt.datetime.now().astimezone().date().isoformat(),
        "es_sim": __version__,
        "runs": runs,
    }
    return _write(key, bundle, log)


def repack(key: str, log=print) -> Path:
    """計算し直さず、今のファイルを今の桁数で丸め直して書き直す。"""
    with gzip.open(OUT_DIR / f"{key}.json.gz", "rb") as fh:
        bundle = json.loads(fh.read().decode("utf-8"))
    return _write(key, bundle, log)


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("keys", nargs="*", help="サンプルのキー (省くと全て)")
    ap.add_argument("--repack", action="store_true", help="計算し直さず、今のファイルを今の桁数で丸め直す")
    args = ap.parse_args(argv)
    for key in args.keys or list(PLANS):
        if key not in PLANS:
            raise SystemExit(f"結果を作る表にないサンプル: {key} ({', '.join(PLANS)})")
        (repack if args.repack else build)(key, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
