"""CUDA カーネルのロード (NVRTC による実行時コンパイル) と起動ヘルパー (prompts/119)。

カーネルのソースは ``es_sim/kernels/<name>.cu`` に置く (Python 文字列に埋め込まない —
将来 nvcc で事前コンパイルしたり C++ 拡張へ移したりできるよう、素の CUDA C++ として保つ)。
CuPy の RawModule はソース + オプションのハッシュでコンパイル結果をディスクキャッシュ
(~/.cupy/kernel_cache) するため、2 回目以降の起動ではコンパイルは走らない。

このモジュールは CuPy が import できる環境でのみ import すること
(``es_sim.device.cuda_available()`` で確認してから)。
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import cupy as cp

KERNEL_DIR = Path(__file__).resolve().parent.parent / "kernels"

# 全カーネル共通のコンパイルオプション。-std=c++17 は CUDA 13 の NVRTC 既定と同等だが明示する
_OPTIONS: tuple[str, ...] = ("-std=c++17",)


_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([A-Za-z0-9_./-]+)"\s*$', re.MULTILINE)


def _read_source(path: Path, seen: set[str] | None = None) -> str:
    """ソースを読み、``#include "x.cuh"`` (kernels/ 内のローカルヘッダ) を展開する。

    NVRTC にインクルードパスを渡す代わりに文字列として展開する (PyInstaller 同梱時も
    パス解決の心配が無い)。同じヘッダは一度だけ展開する (#pragma once 相当)。
    """
    seen = set() if seen is None else seen
    src = path.read_text(encoding="utf-8")
    if not src.isascii():
        # CuPy は NVRTC に渡す前にソースを OS 既定の文字コード (日本語 Windows では cp932) で
        # 一時ファイルに書くため、非 ASCII 文字 (コメント中の全角記号など) があると失敗する
        raise ValueError(f"{path.name}: CUDA ソースは ASCII のみで書いてください (コメント含む)")

    def repl(m: re.Match) -> str:
        inc = m.group(1)
        if inc in seen:
            return ""
        seen.add(inc)
        return _read_source(KERNEL_DIR / inc, seen)

    return _INCLUDE_RE.sub(repl, src).replace("#pragma once", "")


@lru_cache(maxsize=None)
def load_module(name: str) -> cp.RawModule:
    """``kernels/<name>.cu`` を RawModule としてコンパイル・ロードする (プロセス内キャッシュ)。"""
    src = _read_source(KERNEL_DIR / f"{name}.cu")
    mod = cp.RawModule(code=src, options=_OPTIONS)
    mod.compile()  # 構文エラーをロード時点で顕在化させる
    return mod


def get_kernel(module: str, func: str) -> cp.RawKernel:
    return load_module(module).get_function(func)


def grid_1d(n: int, block: int = 256) -> tuple[tuple[int], tuple[int]]:
    """1 次元起動の (grid, block)。"""
    return ((max(1, (int(n) + block - 1) // block),), (block,))


def grid_2d(nx: int, ny: int, bx: int = 32, by: int = 8) -> tuple[tuple[int, int], tuple[int, int]]:
    """2 次元起動の (grid, block)。x が高速 (連続) 方向。"""
    return (
        (max(1, (int(nx) + bx - 1) // bx), max(1, (int(ny) + by - 1) // by)),
        (bx, by),
    )
