"""v2 EB Poisson の CPU カーネル (Numba、無ければ NumPy) — kernels/poisson.cu と同じ数式 (prompts/119)。

配列規約は kernels/poisson.cu の先頭コメント参照 (節点 (ny+1, nx+1)、cx (ny+1, nx)、
cy (ny, nx+1)、mask 0=未知/1=固定/2=周期スレーブ)。

Numba 版は行 (j) 方向に prange で並列化する。赤黒 GS の同色節点は互いに隣接しないため
並列更新しても逐次と同じ結果になる。numba が無い環境 (ES_SIM_NO_NUMBA=1 等) では
配列シフトによる NumPy ベクトル化版に自動で切り替える (遅いが同じ結果)。
"""

from __future__ import annotations

import numpy as np

from .._numba_kernels import HAVE_NUMBA

if HAVE_NUMBA:
    from numba import njit, prange

    @njit(inline="always")
    def _offdiag(x, cx, cy, i, j, nx, ny, px, py):
        s = 0.0
        if i > 0:
            s += cx[j, i - 1] * x[j, i - 1]
        elif px:
            s += cx[j, nx - 1] * x[j, nx - 1]
        if i < nx:
            ie = 0 if (px and i == nx - 1) else i + 1
            s += cx[j, i] * x[j, ie]
        if j > 0:
            s += cy[j - 1, i] * x[j - 1, i]
        elif py:
            s += cy[ny - 1, i] * x[ny - 1, i]
        if j < ny:
            jn = 0 if (py and j == ny - 1) else j + 1
            s += cy[j, i] * x[jn, i]
        return s

    @njit(parallel=True)
    def rbgs(x, b, cx, cy, diag, mask, px, py, color):
        ny = x.shape[0] - 1
        nx = x.shape[1] - 1
        for j in prange(ny + 1):
            i0 = (color + j) & 1
            for i in range(i0, nx + 1, 2):
                if mask[j, i] != 0:
                    continue
                x[j, i] = (b[j, i] + _offdiag(x, cx, cy, i, j, nx, ny, px, py)) / diag[j, i]

    @njit(parallel=True)
    def residual(r, x, b, cx, cy, diag, mask, px, py):
        ny = x.shape[0] - 1
        nx = x.shape[1] - 1
        for j in prange(ny + 1):
            for i in range(nx + 1):
                if mask[j, i] != 0:
                    r[j, i] = 0.0
                else:
                    r[j, i] = b[j, i] - (diag[j, i] * x[j, i] - _offdiag(x, cx, cy, i, j, nx, ny, px, py))

    @njit(parallel=True)
    def apply_op(y, x, cx, cy, diag, mask, px, py):
        ny = x.shape[0] - 1
        nx = x.shape[1] - 1
        for j in prange(ny + 1):
            for i in range(nx + 1):
                if mask[j, i] != 0:
                    y[j, i] = 0.0
                else:
                    y[j, i] = diag[j, i] * x[j, i] - _offdiag(x, cx, cy, i, j, nx, ny, px, py)

    @njit(parallel=True)
    def restrict_fw(bc, rf, mask_c, fx, fy, px, py):
        nyc = bc.shape[0] - 1
        nxc = bc.shape[1] - 1
        nyf = rf.shape[0] - 1
        nxf = rf.shape[1] - 1
        for J in prange(nyc + 1):
            for I in range(nxc + 1):
                if mask_c[J, I] != 0:
                    bc[J, I] = 0.0
                    continue
                i0 = fx * I
                j0 = fy * J
                s = 0.0
                for b in range(-(fy - 1), fy):
                    j = j0 + b
                    if j < 0:
                        if not py:
                            continue
                        j += nyf
                    elif j > nyf:
                        continue
                    elif py and j == nyf:
                        j = 0
                    wb = 1.0 if b == 0 else 0.5
                    for a in range(-(fx - 1), fx):
                        i = i0 + a
                        if i < 0:
                            if not px:
                                continue
                            i += nxf
                        elif i > nxf:
                            continue
                        elif px and i == nxf:
                            i = 0
                        wa = 1.0 if a == 0 else 0.5
                        s += wa * wb * rf[j, i]
                bc[J, I] = s

    @njit(parallel=True)
    def prolong_add(xf, ec, mask_f, fx, fy, px, py):
        nyf = xf.shape[0] - 1
        nxf = xf.shape[1] - 1
        nyc = ec.shape[0] - 1
        nxc = ec.shape[1] - 1
        for j in prange(nyf + 1):
            for i in range(nxf + 1):
                if mask_f[j, i] != 0:
                    continue
                I0 = i // fx
                di = i - I0 * fx
                J0 = j // fy
                dj = j - J0 * fy
                I1 = I0 + di
                J1 = J0 + dj
                if px and I1 == nxc:
                    I1 = 0
                if py and J1 == nyc:
                    J1 = 0
                wx1 = 0.5 * di
                wx0 = 1.0 - wx1
                wy1 = 0.5 * dj
                wy0 = 1.0 - wy1
                xf[j, i] += wy0 * (wx0 * ec[J0, I0] + wx1 * ec[J0, I1]) + wy1 * (
                    wx0 * ec[J1, I0] + wx1 * ec[J1, I1]
                )

else:

    def _offdiag_np(x, cx, cy, px, py):
        ny = x.shape[0] - 1
        nx = x.shape[1] - 1
        s = np.zeros_like(x)
        s[:, 1:] += cx * x[:, :-1]
        s[:, :-1] += cx * x[:, 1:]
        if px:
            s[:, 0] += cx[:, nx - 1] * x[:, nx - 1]
            s[:, nx - 1] += cx[:, nx - 1] * (x[:, 0] - x[:, nx])
        s[1:, :] += cy * x[:-1, :]
        s[:-1, :] += cy * x[1:, :]
        if py:
            s[0, :] += cy[ny - 1, :] * x[ny - 1, :]
            s[ny - 1, :] += cy[ny - 1, :] * (x[0, :] - x[ny, :])
        return s

    def _color_mask(shape, color):
        j, i = np.indices(shape)
        return ((i + j) & 1) == color

    def rbgs(x, b, cx, cy, diag, mask, px, py, color):
        sel = _color_mask(x.shape, color) & (mask == 0)
        x[sel] = ((b + _offdiag_np(x, cx, cy, px, py)) / diag)[sel]

    def residual(r, x, b, cx, cy, diag, mask, px, py):
        r[...] = np.where(mask == 0, b - (diag * x - _offdiag_np(x, cx, cy, px, py)), 0.0)

    def apply_op(y, x, cx, cy, diag, mask, px, py):
        y[...] = np.where(mask == 0, diag * x - _offdiag_np(x, cx, cy, px, py), 0.0)

    def restrict_fw(bc, rf, mask_c, fx, fy, px, py):
        nyc, nxc = bc.shape[0] - 1, bc.shape[1] - 1
        nyf, nxf = rf.shape[0] - 1, rf.shape[1] - 1
        s = np.zeros_like(bc)
        J, I = np.indices(bc.shape)
        for b in range(-(fy - 1), fy):
            j = fy * J + b
            okj = np.ones_like(j, dtype=bool)
            if py:
                j = np.where(j < 0, j + nyf, j)
                j = np.where(j == nyf, 0, j)
            else:
                okj = (j >= 0) & (j <= nyf)
            wb = 1.0 if b == 0 else 0.5
            for a in range(-(fx - 1), fx):
                i = fx * I + a
                oki = np.ones_like(i, dtype=bool)
                if px:
                    i = np.where(i < 0, i + nxf, i)
                    i = np.where(i == nxf, 0, i)
                else:
                    oki = (i >= 0) & (i <= nxf)
                ok = okj & oki
                wa = 1.0 if a == 0 else 0.5
                s[ok] += wa * wb * rf[np.clip(j, 0, nyf)[ok], np.clip(i, 0, nxf)[ok]]
        bc[...] = np.where(mask_c == 0, s, 0.0)

    def prolong_add(xf, ec, mask_f, fx, fy, px, py):
        nyc, nxc = ec.shape[0] - 1, ec.shape[1] - 1
        J, I = np.indices(xf.shape)
        I0 = I // fx
        di = I - I0 * fx
        J0 = J // fy
        dj = J - J0 * fy
        I1 = I0 + di
        J1 = J0 + dj
        if px:
            I1 = np.where(I1 == nxc, 0, I1)
        if py:
            J1 = np.where(J1 == nyc, 0, J1)
        I0 = np.clip(I0, 0, nxc)
        I1 = np.clip(I1, 0, nxc)
        J0 = np.clip(J0, 0, nyc)
        J1 = np.clip(J1, 0, nyc)
        wx1 = 0.5 * di
        wx0 = 1.0 - wx1
        wy1 = 0.5 * dj
        wy0 = 1.0 - wy1
        e = wy0 * (wx0 * ec[J0, I0] + wx1 * ec[J0, I1]) + wy1 * (wx0 * ec[J1, I0] + wx1 * ec[J1, I1])
        xf += np.where(mask_f == 0, e, 0.0)
