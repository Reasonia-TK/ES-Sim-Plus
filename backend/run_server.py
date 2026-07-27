"""配布用バックエンド起動スクリプト (PyInstaller のエントリポイント、prompts/44)。

Tauri のサイドカーとして起動され、uvicorn で FastAPI サーバーを
127.0.0.1 (既定ポート 8317) に立ち上げる。

使い方:
    es-sim-backend [--port 8317] [--host 127.0.0.1]
    # PIC バッチ実行 (prompts/78)。複数プロジェクトJSONを別プロセスで並列実行し、
    # GUIの「結果付き保存」形式で書き出す (配布版ユーザーもスイープに使える)
    es-sim-backend batch case1.json case2.json ... [--parallel N] [--out DIR]
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys

import uvicorn


def _run_server(args: argparse.Namespace) -> None:
    # ワーカー1・リロード無効 (配布バイナリ)。ログは標準出力へ
    uvicorn.run(
        "es_sim.server:app",
        host=args.host,
        port=args.port,
        workers=1,
        reload=False,
        log_level="info",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ES-Sim backend server")
    parser.add_argument("--port", type=int, default=8317, help="待受ポート (既定: 8317)")
    parser.add_argument(
        "--host", default="127.0.0.1", help="待受アドレス (既定: 127.0.0.1)"
    )
    # サブコマンド未指定 (従来通り引数なし/--port のみ) はサーバー起動。Tauri サイドカーの
    # 呼び出し方 (--port N のみ渡す) を変えないよう、サブコマンドは完全に任意にする
    sub = parser.add_subparsers(dest="cmd")

    batch_p = sub.add_parser(
        "batch", help="複数プロジェクトJSONをPICで並列バッチ実行する (prompts/78、es_sim.batch)"
    )
    batch_p.add_argument("files", nargs="+", help="入力プロジェクトJSON (GUIの「保存」形式)")
    batch_p.add_argument(
        "--parallel", type=int, default=1,
        help="同時実行プロセス数 (既定: 1)。合計スレッド数 (--parallel × pic.threads) が"
        "CPUコア数を超えないように注意すること",
    )
    batch_p.add_argument("--out", default=None, help="出力先ディレクトリ (既定: 各入力ファイルと同じ場所)")
    batch_p.add_argument(
        "--suffix", default="_results", help='出力ファイル名サフィックス (既定 "_results")'
    )

    args = parser.parse_args()

    if args.cmd == "batch":
        # PyInstaller の単一 exe に同梱された es_sim をそのまま使う (追加の依存なし)
        from es_sim.batch import run_files

        sys.exit(run_files(args.files, parallel=args.parallel, out_dir=args.out, suffix=args.suffix))

    _run_server(args)


if __name__ == "__main__":
    # Windows の PyInstaller onefile exe で multiprocessing (spawn) を使うと、子プロセスは
    # exe 自身を再実行し、自分が子プロセスであることを示す内部引数を付けて main を呼び直す。
    # freeze_support() を呼んでおかないと、この再実行時に multiprocessing がその内部引数を
    # 認識できず、通常のエントリポイントとして解釈して main() を再度呼んでしまい、
    # 子プロセスがさらに子プロセスを産む無限増殖 (fork爆弾状態) になる。
    # 非 Windows・非フリーズ環境では no-op なので常時呼んでおいて害はない
    multiprocessing.freeze_support()
    main()
