"""GPU の密行列の演算 (cuBLAS・cuSOLVER を使わない、prompts/133 P8a)。

配布版は CUDA Toolkit から NVRTC だけを同梱する (cuBLAS・cuSOLVER・cuSPARSE を同梱すると
インストーラが約 640 MB 大きくなる)。ソルバーが GPU で使う密な逆行列は ``kernels/linalg.cu`` の
ブロック Gauss-Jordan で求める。

このモジュールは CuPy が import できる環境でのみ使うこと (``es_sim.device.cuda_available()``)。
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np

#: kernels/linalg.cu の GJ_B (1 ブロックの幅)
_GJ_B = 32


def spd_inverse(a) -> object:
    """対称正定値行列の逆行列を GPU で求める (戻り値は cupy 配列)。

    ピボット無しのブロック Gauss-Jordan (対称正定値ならピボットは全て正で安定)。行列が対称でない・
    ピボットが正でない・結果が有限でないときは警告して CPU (LAPACK) で求め直す。
    """
    import cupy as cp

    from .cuda import get_kernel

    host = np.ascontiguousarray(a.get() if hasattr(a, "get") else a, dtype=np.float64)
    n = int(host.shape[0])
    if host.ndim != 2 or host.shape[1] != n:
        raise ValueError(f"正方行列ではありません: {host.shape}")
    if n == 0:
        return cp.zeros((0, 0))
    m = cp.array(host)
    scale = float(cp.max(cp.abs(m)))
    reason = None
    if not np.isfinite(scale):
        reason = "行列に有限でない成分がある"
    elif float(cp.max(cp.abs(m - m.T))) > 1e-12 * scale:
        reason = "行列が対称でない"
    else:
        k_inv = get_kernel("linalg", "gj_block_inv")
        k_row = get_kernel("linalg", "gj_row_panel")
        k_col = get_kernel("linalg", "gj_col_copy")
        k_upd = get_kernel("linalg", "gj_update")
        p = cp.zeros(_GJ_B * _GJ_B)
        rp = cp.zeros(_GJ_B * n)
        cb = cp.zeros(n * _GJ_B)
        piv = cp.zeros(n)
        n32 = np.int32(n)
        for j0 in range(0, n, _GJ_B):
            w = min(_GJ_B, n - j0)
            args = (n32, np.int32(j0), np.int32(w))
            k_inv((1,), (_GJ_B, _GJ_B), (m, p, piv) + args)
            k_row(((n + 255) // 256, w), (256,), (m, p, rp) + args)
            k_col(((n * w + 255) // 256,), (256,), (m, cb) + args)
            k_upd(((n + 31) // 32, (n + 7) // 8), (32, 8), (m, p, rp, cb) + args)
        pv = piv.get()
        if not (np.all(np.isfinite(pv)) and np.all(pv > 0.0)):
            reason = f"行列が正定値でない (ピボットの最小 {float(np.min(pv)):.3g})"
        elif not bool(cp.all(cp.isfinite(m))):
            reason = "結果が有限でない"
        else:
            return m
    warnings.warn(f"GPU の逆行列: {reason}ため CPU で求め直しました ({n} 元)", RuntimeWarning, stacklevel=2)
    return cp.asarray(np.linalg.inv(host))


@lru_cache(maxsize=1)
def _vdot_kernel():
    import cupy as cp

    return cp.ReductionKernel("T x, T y", "T z", "x * y", "a + b", "z = a", "0", "es_sim_vdot")


def vdot(a, b):
    """実ベクトル (配列は平らにみなす) の内積を 0 次元の cupy 配列で返す (``cupy.vdot`` は cuBLAS を使うため、
    自前の縮約カーネルで)。"""
    return _vdot_kernel()(a.ravel(), b.ravel())


__all__ = ["spd_inverse", "vdot"]
