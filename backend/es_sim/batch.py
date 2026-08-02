"""PIC バッチ実行モード (prompts/78)。

GUI (/ws/pic) は1アプリ=1計算 (サーバーが _pic_lock で同時 start を拒否する) だが、
パラメータスイープでは複数ケースを一気に流したいことが多い。本モジュールは
ヘッドレスで複数のプロジェクト JSON を **別プロセスで並列** に PIC 実行し、
GUI の「結果付き保存」と同じ形式 (ResultsBundle) の JSON を書き出す CLI を提供する。

使い方:
    python -m es_sim.batch run case1.json case2.json ... --parallel 2 --out out_dir

各ケースは multiprocessing (spawn) の別プロセスで実行する。GIL の影響を受けず、
numba/BLAS のスレッドプールもプロセスごとに独立するため、--parallel を上げても
1ケースあたりの数値結果は逐次実行と完全に一致する (プロセスを跨いだ共有状態が無いため)。

注意: 合計スレッド数は「--parallel × 各ケースの pic.threads」になる。CPU コア数を
大きく超えるとスレッドの奪い合いで逆に遅くなるので、--parallel × threads がコア数
以下になるように調整すること。

1D PIC/MCC (pic1d.py、prompts/91) にも対応する (prompts/96)。1D 流体
(fluid1d.py、prompts/104-107) にも対応する (prompts/107)。--module で
"pic"/"pic1d"/"fluid1d"/"auto" (既定、project の pic → pic1d → fluid1d の
優先順で判定) を選べる。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue as queue_mod
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .pic import PicSimulation
from .pic1d import Pic1dSimulation, build_pic1d_result
from .fluid1d import Fluid1dSimulation, build_fluid1d_result
from .schema import Project

# frontend/src/types.ts の PicDiag と同じキー順 (toDiagArray が組み立てる行の形に合わせる)。
# pic.py の history はこのキー全てを常に持つ (欠損なし) ので単純な列→行変換でよい。
_DIAG_KEYS = (
    "t", "ke_e", "ke_i", "fe", "n_e", "n_i", "wall_e", "wall_i", "phi_min", "phi_max",
    "coll_e", "ion_events", "see_events", "surf_q", "fn_i", "fn_events", "merged",
)

# 進捗通知の最短間隔 [秒]。子プロセス→親への Queue 送信・親側の表示過多を防ぐ間引き
_PROGRESS_INTERVAL_S = 2.0


def _prepare_project_dict(raw: dict) -> dict:
    """入力 JSON をバッチ実行用に前処理する。

    - 「結果付き保存」ファイルが誤って渡された場合に備え、Project スキーマ外の
      results キーを除去する (frontend/App.tsx の loadProject と同じ分離)。
    - pic.injection.emitter を particles.emitter で上書きする
      (App.tsx の withInjectionEmitter と同じ合成。PicPanel は編集用の複製を持たず、
      保存/送信時に都度同期する設計のため、未合成の手書き JSON にもここで対応する)。
    """
    project = {k: v for k, v in raw.items() if k != "results"}
    pic = project.get("pic")
    particles = project.get("particles")
    if (
        isinstance(pic, dict)
        and pic.get("injection")
        and isinstance(particles, dict)
        and particles.get("emitter")
    ):
        project = {
            **project,
            "pic": {**pic, "injection": {**pic["injection"], "emitter": particles["emitter"]}},
        }
    return project


def _build_results_bundle(sim: PicSimulation, step_offset: int, elapsed_s: float) -> dict:
    """ResultsBundle.pic (frontend/src/types.ts) と同じ形の dict を組み立てる。

    server.py の _stream_run が started/done メッセージを組み立てる変換をそのまま流用する。
    history は pic.py が返す「列ごとの辞書」なので、frontend の toDiagArray 相当の
    変換 (列→行) をここで行う (バッチ出力は GUI の picHistory state と同じ「行の配列」形式)。
    elapsed_s は run_batch の壁時計秒 (server.py done メッセージの elapsed_s と同じ計測対象、prompts/86)。
    """
    started = {
        "type": "started",
        "dt": sim.dt,
        "n_steps": sim.pic.n_steps,
        "effective_threads": sim.effective_threads,
        "step_offset": step_offset,
        "warnings": sim.warnings,
        "mesh": {
            "nodes": sim.mesh.nodes.tolist(),
            "triangles": sim.mesh.triangles.tolist(),
        },
    }

    n_rows = len(sim.history.get("t", []))
    history = [{k: sim.history[k][i] for k in _DIAG_KEYS if k in sim.history} for i in range(n_rows)]

    fields = None
    if sim.fields is not None:
        fields = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sim.fields.items()
        }

    cycle = None
    if sim.cycle is not None:
        def _f32(arr: Any) -> list:
            return np.asarray(arr, dtype=np.float32).tolist()

        c = sim.cycle
        cycle = {
            "bins": c["bins"],
            "period_s": c["period_s"],
            "phi": _f32(c["phi"]),
            "n_e": _f32(c["n_e"]),
            "n_i": _f32(c["n_i"]),
            "e_abs": _f32(c["e_abs"]),
            "te_ev": _f32(c["te_ev"]),
            "ion_rate": _f32(c["ion_rate"]),
            "particles": {
                name: [_f32(s) for s in snaps] for name, snaps in c["particles"].items()
            },
        }

    # ResultsBundle.pic.collectors は必須の配列 (undefined 不可) なので未使用時は空配列にする
    collectors: list[dict] = []
    if sim.collector_results is not None:
        for cr in sim.collector_results:
            collectors.append(
                {
                    "count": cr["count"],
                    "total_weight": cr["total_weight"],
                    "energies_ev": cr["energies_ev"].tolist(),
                    "angles_deg": cr["angles_deg"].tolist(),
                    "weights": cr["weights"].tolist(),
                    "truncated": cr["truncated"],
                }
            )

    # ResultsBundle.pic.eedf も同様に必須配列 (prompts/85)。server.py の done 組み立てと同じ変換
    eedf: list[dict] = []
    if sim.eedf_results is not None:
        for r in sim.eedf_results:
            eedf.append(
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
            )

    return {
        "version": 1,
        "pic": {
            "started": started,
            "frame": None,  # ライブフレームは無い (バッチは history/fields/cycle のみ復元すればよい)
            "history": history,
            "fields": fields,
            "cycle": cycle,
            "collectors": collectors,
            "eedf": eedf,
            "elapsed_s": elapsed_s,
        },
    }


def _resolve_module(project: Project, module: str) -> str:
    """module="auto" を project の中身から解決する (バッチ CLI の --module auto、prompts/96・107)。

    "pic"/"pic1d"/"fluid1d" 指定はそのまま返す (呼び出し側の明示指定を優先。GUI
    スイープは常にこちらの経路 — サーバー側で解決済みの値をそのまま渡す)。
    project にどれが設定されているかで判定する優先順位は pic → pic1d → fluid1d
    (複数設定されていても最初に見つかったものを使う。1つも無ければ auto では
    判定できないためエラーにする)。
    """
    if module != "auto":
        return module
    if project.pic is not None:
        return "pic"
    if project.pic1d is not None:
        return "pic1d"
    if project.fluid1d is not None:
        return "fluid1d"
    raise ValueError("project に pic も pic1d も fluid1d もありません (module=auto では判定できません)")


def _worker(
    case_path: str, out_path: str, case_name: str, progress_q: "mp.Queue", module: str = "pic"
) -> None:
    """1ケース分の実行本体。spawn 起動のためモジュールレベルの picklable な関数にする。

    module は "pic" (既定、2D FEM-PIC。既存呼び出し・既存挙動とビット不変)・"pic1d"
    (1D PIC/MCC、prompts/96)・"fluid1d" (1D ドリフト拡散流体、prompts/107)・"auto"
    (project の中身から解決、バッチ CLI の既定) のいずれか。GUI スイープ
    (sweep.py) は常に解決済みの "pic"/"pic1d"/"fluid1d" を渡す (対象パラメータ
    パスから判定するため、project の中身だけでは判定できない場合がある)。

    成否・所要時間は例外にせず progress_q へ dict で通知する (親プロセスは Process の
    exitcode だけでは失敗理由が分からないため)。
    """
    t0 = time.perf_counter()
    try:
        with open(case_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        project_dict = _prepare_project_dict(raw)
        project = Project.model_validate(project_dict)
        resolved = _resolve_module(project, module)

        if resolved == "pic1d":
            if project.pic1d is None:
                raise ValueError(
                    "pic1d 設定がありません (module=pic1d を指定するには project.pic1d が必要です)"
                )
            # use_dsmc_gas は 1D では schema の validator が既に拒否しているため
            # ここでの追加チェックは不要 (Project.model_validate の時点で ValueError になる)
            sim1d = Pic1dSimulation(project)
            n_steps_1d = sim1d.s.n_steps
            last_sent_1d = 0.0

            def on_frame_1d(frame: dict) -> None:
                nonlocal last_sent_1d
                now = time.perf_counter()
                if now - last_sent_1d >= _PROGRESS_INTERVAL_S or frame["step"] >= n_steps_1d:
                    last_sent_1d = now
                    progress_q.put(
                        {"case": case_name, "kind": "progress", "step": frame["step"], "n_steps": n_steps_1d}
                    )

            t_run0 = time.perf_counter()
            sim1d.run_batch(on_frame_1d, lambda: False, False)
            elapsed_s = time.perf_counter() - t_run0
            # server.py の _pic1d_result (現 build_pic1d_result) と同一の形にする
            # (frontend の ResultsBundle.pic1d / Pic1dResult とキー構造を一致させるため)
            bundle = {"version": 1, "pic1d": build_pic1d_result(sim1d, elapsed_s)}
        elif resolved == "pic":
            if project.pic is None:
                raise ValueError("pic 設定がありません (バッチ実行は PIC プロジェクト専用です)")
            if project.pic.mcc is not None and project.pic.mcc.use_dsmc_gas:
                # use_dsmc_gas はサーバーが保持する直前の DSMC 結果を参照する機能だが、
                # バッチはプロセスごとに独立しておりそのような共有状態を持たないため未対応
                raise ValueError(
                    "pic.mcc.use_dsmc_gas はバッチ実行では未対応です "
                    "(DSMC結果はプロセス間で共有されないため)"
                )

            sim = PicSimulation(project, gas_field=None)
            step_offset = sim.step_count  # 新規構築直後なので常に0
            n_steps = sim.pic.n_steps

            last_sent = 0.0

            def on_frame(frame: dict) -> None:
                nonlocal last_sent
                now = time.perf_counter()
                if now - last_sent >= _PROGRESS_INTERVAL_S or frame["step"] >= n_steps:
                    last_sent = now
                    progress_q.put(
                        {"case": case_name, "kind": "progress", "step": frame["step"], "n_steps": n_steps}
                    )

            # 進捗通知にはcallbackだけを使い、巨大なライブフレーム列は結果へ保存しない。
            # run_batch の壁時計計測 (prompts/86)。server.py の done.elapsed_s と同じ計測対象
            # (プロジェクト読込・メッシュ生成等は含まない) にして GUI 実行との比較を可能にする
            t_run0 = time.perf_counter()
            sim.run_batch(on_frame, lambda: False, False)
            elapsed_s = time.perf_counter() - t_run0
            bundle = _build_results_bundle(sim, step_offset, elapsed_s)
        elif resolved == "fluid1d":
            if project.fluid1d is None:
                raise ValueError(
                    "fluid1d 設定がありません (module=fluid1d を指定するには project.fluid1d が必要です)"
                )
            # 流体は粒子もMCCも無い (use_dsmc_gas 相当の制約が存在しない) ので、
            # pic1d 分岐と違って追加チェックは不要
            simf = Fluid1dSimulation(project)
            n_steps_f = simf.s.n_steps
            last_sent_f = 0.0

            def on_frame_f(frame: dict) -> None:
                nonlocal last_sent_f
                now = time.perf_counter()
                if now - last_sent_f >= _PROGRESS_INTERVAL_S or frame["step"] >= n_steps_f:
                    last_sent_f = now
                    progress_q.put(
                        {"case": case_name, "kind": "progress", "step": frame["step"], "n_steps": n_steps_f}
                    )

            t_run0 = time.perf_counter()
            simf.run_batch(on_frame_f, lambda: False, False)
            elapsed_s = time.perf_counter() - t_run0
            # server.py の _stream_run_fluid1d (build_fluid1d_result) と同一の形にする
            bundle = {"version": 1, "fluid1d": build_fluid1d_result(simf, elapsed_s)}
        else:
            raise ValueError(
                f"不明な module です: {module!r} "
                "('pic'/'pic1d'/'fluid1d'/'auto' のいずれかを指定してください)"
            )

        out_obj = {**project_dict, "results": bundle}
        # cycle の粒子スナップショット等でサイズが大きくなり得るため整形なし (compact) で書く
        # (frontend の saveProjectWithResults と同じ方針)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False)

        progress_q.put(
            {
                "case": case_name,
                "kind": "done",
                "elapsed": time.perf_counter() - t0,
                "out_path": out_path,
            }
        )
    except Exception as exc:  # noqa: BLE001 - 子プロセスの例外は文字列化して親へ伝える
        progress_q.put(
            {
                "case": case_name,
                "kind": "error",
                "elapsed": time.perf_counter() - t0,
                "error": str(exc),
            }
        )


def run_files(
    files: list[str],
    *,
    parallel: int = 1,
    out_dir: str | None = None,
    suffix: str = "_results",
    module: str = "auto",
) -> int:
    """複数ケースを (最大 parallel 並列で) 実行し、サマリを表示する。全成功なら0、1件でも失敗があれば1を返す。

    module は全ファイル共通の指定 (CLI の --module、既定 "auto": ファイルごとに
    project の pic → pic1d → fluid1d の優先順で "pic"/"pic1d"/"fluid1d" を判定する。
    1バッチ内に 2D PIC/1D PIC/1D 流体の混在も可)。既存呼び出し (module を指定
    しない = "auto") でも 2D 専用プロジェクトなら常に "pic" に解決されるため、
    既存の 2D バッチとビット不変。
    """
    parallel = max(1, parallel)
    # spawn を明示: fork だと numba/BLAS のスレッドプール等が親プロセスの状態を引きずり、
    # 独立性が壊れる恐れがある (Windows は元々 spawn のみだが、Linux/macOS でも合わせる)
    ctx = mp.get_context("spawn")
    progress_q: mp.Queue = ctx.Queue()

    jobs: list[dict[str, str]] = []
    for f in files:
        in_path = Path(f)
        dst_dir = Path(out_dir) if out_dir else in_path.parent
        dst_dir.mkdir(parents=True, exist_ok=True)
        out_path = dst_dir / f"{in_path.stem}{suffix}.json"
        jobs.append({"name": in_path.stem, "in": str(in_path), "out": str(out_path)})

    pending = list(jobs)
    running: dict[str, Any] = {}
    summary: dict[str, dict] = {}
    t_all0 = time.perf_counter()

    def _launch(job: dict[str, str]) -> None:
        p = ctx.Process(target=_worker, args=(job["in"], job["out"], job["name"], progress_q, module))
        p.start()
        running[job["name"]] = p

    while pending or running:
        while pending and len(running) < parallel:
            _launch(pending.pop(0))

        try:
            msg = progress_q.get(timeout=0.5)
        except queue_mod.Empty:
            msg = None

        if msg is not None:
            if msg["kind"] == "progress":
                pct = 100.0 * msg["step"] / max(1, msg["n_steps"])
                print(f"[{msg['case']}] {msg['step']}/{msg['n_steps']} ({pct:.0f}%)", flush=True)
            elif msg["kind"] == "done":
                summary[msg["case"]] = {"ok": True, "elapsed": msg["elapsed"], "out": msg["out_path"]}
            elif msg["kind"] == "error":
                summary[msg["case"]] = {"ok": False, "elapsed": msg["elapsed"], "error": msg["error"]}

        # 終了したプロセスを回収する。progress_q からの done/error 通知より先に
        # is_alive() が False になることは無いはず (put 後に子プロセスが return するため) だが、
        # クラッシュ (シグナル終了等) で通知が届かないケースに備えて exitcode でフォールバックする
        for name in list(running):
            p = running[name]
            if not p.is_alive():
                p.join()
                running.pop(name)
                if name not in summary:
                    summary[name] = {
                        "ok": False,
                        "elapsed": 0.0,
                        "error": f"子プロセスが結果を報告せずに終了しました (exit code {p.exitcode})",
                    }

    total_elapsed = time.perf_counter() - t_all0

    print("\n=== バッチ実行サマリ ===")
    n_ok = 0
    for job in jobs:
        r = summary.get(job["name"])
        if r is None:
            print(f"[{job['name']}] 不明 (結果を受け取れませんでした)")
            continue
        if r["ok"]:
            n_ok += 1
            print(f"[{job['name']}] OK   {r['elapsed']:.2f}s -> {r['out']}")
        else:
            print(f"[{job['name']}] NG   {r['elapsed']:.2f}s  エラー: {r['error']}")
    n_total = len(jobs)
    print(f"合計: {n_total}件中 {n_ok}件成功 (壁時計時間 {total_elapsed:.2f}s)")

    return 0 if n_ok == n_total else 1


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m es_sim.batch", description="PIC バッチ実行 (複数プロジェクトJSONの並列実行)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser(
        "run", help="複数のプロジェクトJSONをPICで実行し、結果付きJSON (GUIの「結果付き保存」形式) を書き出す"
    )
    run_p.add_argument("files", nargs="+", help="入力プロジェクトJSON (GUIの「保存」形式)")
    run_p.add_argument(
        "--parallel", type=int, default=1,
        help="同時実行プロセス数 (既定: 1)。合計スレッド数 (--parallel × pic.threads) が"
        "CPUコア数を超えないように注意すること",
    )
    run_p.add_argument("--out", default=None, help="出力先ディレクトリ (既定: 各入力ファイルと同じ場所)")
    run_p.add_argument(
        "--suffix", default="_results", help='出力ファイル名サフィックス (既定 "_results" → case1_results.json)'
    )
    run_p.add_argument(
        "--module", choices=["auto", "pic", "pic1d", "fluid1d"], default="auto",
        help="実行するソルバー (既定 auto: ファイルごとに project.pic → pic1d → fluid1d の"
        "優先順で最初に見つかったものを使う。いずれも無ければエラー。prompts/96・107)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.cmd == "run":
        return run_files(
            args.files, parallel=args.parallel, out_dir=args.out, suffix=args.suffix, module=args.module
        )
    return 1  # pragma: no cover - argparse の required=True により通常到達しない


if __name__ == "__main__":
    # spawn 起動の子プロセスがこのスクリプトを再インポートする際に main() が再実行される
    # 事故を防ぐため (Windows の PyInstaller exe に限らず python -m 実行でも作法として)、
    # __main__ ガードの直後で freeze_support() を呼んでおく (非 Windows / 非フリーズ環境では no-op)
    mp.freeze_support()
    sys.exit(main())
