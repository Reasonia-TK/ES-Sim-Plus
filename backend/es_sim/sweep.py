"""GUI パラメータスイープ (1パラメータ × 値リスト、prompts/79)。

対象パラメータの指定はドット区切りパス (例 "geometry.boundaries.1.voltage") のみを
受け付ける汎用実装にしている。GUI 側の「プリセット選択」は現在のプロジェクトから
候補パスを生成する糖衣であり (frontend/src/panels/SweepPanel.tsx)、backend はパスの
意味を一切解釈しない (dict/list を辿って数値を上書きするだけ)。

ケースごとの実行は prompts/78 のバッチ実行 (batch.py) の子プロセス実行関数 (_worker) を
そのまま流用する (別プロセスで実行することで numba/BLAS のスレッドプールがケース間で
独立し、--parallel を上げても各ケースの結果は逐次実行と完全に一致する)。ただし GUI は
標準出力ではなく WebSocket で進捗を受け取りたいため、ケース管理ループ (run_files 相当)
はここで非同期実行向けに再実装し、stop による中断にも対応させている。
"""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import queue as queue_mod
from pathlib import Path
from typing import Any, Callable

from .batch import _worker

# 数値として扱う Python 型。bool は int のサブクラスだが、True/False を数値スイープの
# 対象にすると意味を破壊しかねない (例: immobile_ions) ため明示的に除外する
_NUMERIC_TYPES = (int, float)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, _NUMERIC_TYPES) and not isinstance(value, bool)


def _as_index(token: str, path: str) -> int:
    """パストークンを配列インデックスとして解釈する (負数・非数字はエラー)。"""
    if not token.isdigit():
        raise ValueError(f"パス {path!r}: 配列インデックスが数字ではありません ({token!r})")
    return int(token)


def _step(cur: Any, token: str, path: str) -> Any:
    """パスの中間トークンを1つ辿って次の要素を返す。"""
    if isinstance(cur, list):
        idx = _as_index(token, path)
        if not (0 <= idx < len(cur)):
            raise ValueError(f"パス {path!r}: 配列インデックス {idx} が範囲外です (要素数 {len(cur)})")
        return cur[idx]
    if isinstance(cur, dict):
        if token not in cur:
            raise ValueError(f"パス {path!r}: キー {token!r} が存在しません")
        return cur[token]
    raise ValueError(f"パス {path!r}: 中間ノード {token!r} の手前が dict/list ではありません")


def set_by_path(obj: dict, path: str, value: float) -> None:
    """ドット区切りパスで project dict 内の数値フィールドを value に上書きする (in-place)。

    パスの各トークンは dict のキーまたは配列のインデックス (数字) のいずれか。
    例: "geometry.boundaries.1.voltage", "pic.n_macro", "b_field.bz",
        "geometry.regions.0.voltage_rf.0.amp"。
    存在しないパス・終端が数値でない場合は ValueError を送出する
    (GUI 側はプリセットから生成した既存パスのみを渡す想定だが、カスタムパス入力の
    誤りをここで確実に検出してエラーメッセージとして返せるようにする)。
    """
    tokens = path.split(".")
    if not tokens or any(t == "" for t in tokens):
        raise ValueError(f"不正なパスです: {path!r}")

    cur: Any = obj
    for token in tokens[:-1]:
        cur = _step(cur, token, path)

    last = tokens[-1]
    if isinstance(cur, list):
        idx = _as_index(last, path)
        if not (0 <= idx < len(cur)):
            raise ValueError(f"パス {path!r}: 配列インデックス {idx} が範囲外です (要素数 {len(cur)})")
        if not _is_numeric(cur[idx]):
            raise ValueError(f"パス {path!r}: 終端の値が数値ではありません ({cur[idx]!r})")
        cur[idx] = value
    elif isinstance(cur, dict):
        if last not in cur:
            raise ValueError(f"パス {path!r}: キー {last!r} が存在しません")
        if not _is_numeric(cur[last]):
            raise ValueError(f"パス {path!r}: 終端の値が数値ではありません ({cur[last]!r})")
        cur[last] = value
    else:
        raise ValueError(f"パス {path!r}: 終端の手前が dict/list ではありません")


def resolve_sweep_module(param_path: str, requested: str | None) -> str:
    """スイープ対象の module ("pic"/"pic1d"/"fluid1d") を解決する (prompts/96、fluid1d は prompts/107)。

    requested (GUI で明示指定された値) があればそれを最優先する — 自動判定に頼らず
    UI 確定値をそのまま使うことで、pic/pic1d/fluid1d の設定を同時に持つプロジェクトでも
    ユーザーの意図どおりに実行できる。requested が None (未指定) のときのみ
    param_path の接頭辞で判定する: "pic1d." で始まれば "pic1d"、"fluid1d." で
    始まれば "fluid1d"、それ以外は従来互換で "pic" (2D は module という概念が
    無かったため、既存の GUI/API 利用者に影響が出ないよう既定を "pic" のままにする)。
    """
    if requested in ("pic", "pic1d", "fluid1d"):
        return requested
    if param_path.startswith("pic1d."):
        return "pic1d"
    if param_path.startswith("fluid1d."):
        return "fluid1d"
    return "pic"


def build_sweep_cases(base_project: dict, param_path: str, values: list[float]) -> list[dict]:
    """ベース project (dict) から N ケースを生成する (deepcopy + set_by_path)。

    base_project 自体は変更しない (deepcopy してから上書きするため、呼び出し側が
    同じ dict を使い回しても安全)。
    """
    cases: list[dict] = []
    for v in values:
        case = copy.deepcopy(base_project)
        set_by_path(case, param_path, float(v))
        cases.append(case)
    return cases


def run_sweep(
    cases: list[dict],
    *,
    parallel: int,
    out_dir: str,
    on_event: Callable[[dict], None],
    should_stop: Callable[[], bool],
    module: str = "pic",
) -> None:
    """N ケースを (最大 parallel 並列で) 実行し、進捗を on_event へ通知する。

    module ("pic"/"pic1d"/"fluid1d"、既定 "pic") は全ケース共通で _worker にそのまま渡す
    (prompts/96・107)。呼び出し側 (server.py の ws_sweep) が resolve_sweep_module で
    既に解決済みの値を渡す想定 — ここでは対象パスの意味を解釈しない (set_by_path と
    同じ「backend はパスの意味を解釈しない」方針を module にも適用する)。
    既定値 "pic" により、module を指定しない既存呼び出しは常に従来どおり 2D PIC を
    実行する (ビット不変)。

    batch.run_files と同じケース管理ループ構造だが、GUI 向けに以下2点を変更している:
    - 標準出力の代わりに on_event コールバックで進捗/完了を通知する (server.py が
      WebSocket へ中継する)。
    - should_stop() が True を返したら、以後の新規ケース起動を止め、実行中のプロセスを
      terminate する (完了済みケースの出力ファイルはそのまま残す)。
    子プロセスの実行本体 (_worker) は batch.py と共通 (プロセス分離により numba/BLAS の
    スレッドプールがケース間で独立し、並列数によらず各ケースの数値結果が一致する)。

    各ケースの入力は out_dir/case_{i}.json、出力 (結果付きJSON) は
    out_dir/case_{i}_result.json に書く (server.py の GET /sweep/result/{i} が読む)。
    """
    parallel = max(1, parallel)
    # batch.py と同じ理由 (numba/BLAS のスレッドプール等が親プロセスの状態を引きずらない
    # ようにする) で spawn を明示する
    ctx = mp.get_context("spawn")
    progress_q: mp.Queue = ctx.Queue()

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        in_path = out_dir_p / f"case_{i}.json"
        out_path = out_dir_p / f"case_{i}_result.json"
        with open(in_path, "w", encoding="utf-8") as f:
            json.dump(case, f)
        jobs.append({"index": i, "in": str(in_path), "out": str(out_path)})

    pending = list(jobs)
    running: dict[int, Any] = {}
    # stop で terminate したケースは、子プロセスが exitcode 異常で終了しても
    # 「失敗」として通知しない (意図的な中断であり、ケース自体の失敗ではないため)
    terminated: set[int] = set()
    reported: set[int] = set()

    def _launch(job: dict[str, Any]) -> None:
        p = ctx.Process(target=_worker, args=(job["in"], job["out"], str(job["index"]), progress_q, module))
        p.start()
        running[job["index"]] = p

    while pending or running:
        if should_stop():
            pending.clear()  # まだ起動していないケースは開始しない
            for idx, p in list(running.items()):
                if idx not in terminated:
                    terminated.add(idx)
                    p.terminate()

        while pending and len(running) < parallel:
            _launch(pending.pop(0))

        try:
            msg = progress_q.get(timeout=0.1)
        except queue_mod.Empty:
            msg = None

        if msg is not None:
            idx = int(msg["case"])
            if msg["kind"] == "progress":
                on_event({"type": "progress", "case": idx, "step": msg["step"], "n_steps": msg["n_steps"]})
            elif msg["kind"] == "done":
                reported.add(idx)
                on_event({"type": "case_done", "case": idx, "ok": True})
            elif msg["kind"] == "error":
                reported.add(idx)
                on_event({"type": "case_done", "case": idx, "ok": False, "error": msg["error"]})

        for idx in list(running):
            p = running[idx]
            if not p.is_alive():
                p.join()
                running.pop(idx)
                if idx not in reported and idx not in terminated:
                    # 子プロセスがクラッシュ (シグナル終了等) して progress_q への通知が
                    # 届かなかったフォールバック (batch.run_files と同じ考え方)
                    reported.add(idx)
                    on_event(
                        {
                            "type": "case_done",
                            "case": idx,
                            "ok": False,
                            "error": f"子プロセスが結果を報告せずに終了しました (exit code {p.exitcode})",
                        }
                    )
