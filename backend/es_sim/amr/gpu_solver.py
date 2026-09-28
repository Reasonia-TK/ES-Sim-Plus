"""合成格子 A_c の GPU AMG-PCG (prompts/122)。

- 階層は CPU の pyamg (smoothed aggregation、symmetric) で作り、各レベルの A_l・P_l・R_l = P_lᵀ を
  GPU の CSR (int32 添字) に載せる。最粗レベルは密な逆行列 (特異なら擬似逆行列) の GEMV。
- 平滑化は D⁻¹A の対称 Chebyshev (既定 3 次)。固有値の上限は冪乗法 × 1.1、下限は上限/30。
  前・後平滑化を同じ多項式にするので V サイクルは対称 (CG の前処理に使える)。pyamg 既定の
  Gauss-Seidel は GPU で並列化できず、pyamg の chebyshev は前処理が不定になった (CG が破綻)。
- CG は poisson.cu の同期なしカーネル (dot_acc・pcg_xr・pcg_p) を使う。``launch_solve`` は
  固定反復・warm start・実行中の確保なしで、CUDA Graph にそのまま取り込める (PIC 用)。
  ``solve`` は反復ごとに残差を見る (静電場用)。未知数 ≤ DENSE_DIRECT_MAX は密な逆行列で直接解く。
- Dirichlet の無い問題 (特異、零空間 = 定数) は右辺を平均 0 に射影し、解の平均を 0 にする
  (CPU の solve_composite と同じ正規化)。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ..device import Device, get_device
from .composite import AmrSolveInfo

#: 密な逆行列で直接解く未知数の上限 (逆行列 4096² × 8 B = 134 MB、GMG の DENSE_DIRECT_MAX と同じ)
DENSE_DIRECT_MAX = 4096
CHEB_DEGREE = 3
CHEB_RATIO = 30.0
MAX_COARSE = 500
_B = 256


def _grid(n: int) -> tuple[int]:
    return (max(1, (int(n) + _B - 1) // _B),)


def _lam_max(A: sp.csr_matrix, dinv: np.ndarray, iters: int = 15, seed: int = 0) -> float:
    """D⁻¹A の最大固有値の冪乗法による推定 (CPU)。"""
    x = np.random.default_rng(seed).standard_normal(A.shape[0])
    lam = 1.0
    for _ in range(iters):
        y = dinv * (A @ x)
        ny = float(np.linalg.norm(y))
        if ny == 0.0:
            return 1.0
        lam = ny / float(np.linalg.norm(x))
        x = y / ny
    return lam


def _dense_inverse(A: sp.spmatrix, singular: bool) -> np.ndarray:
    from ..field.gmg import dense_inverse

    return dense_inverse(A.toarray(), singular)


@dataclass
class _Csr:
    n: int
    ip: object
    ix: object
    a: object

    @classmethod
    def of(cls, m: sp.spmatrix, cp) -> "_Csr":
        m = sp.csr_matrix(m)
        m.sort_indices()
        return cls(int(m.shape[0]), cp.asarray(m.indptr.astype(np.int32)), cp.asarray(m.indices.astype(np.int32)),
                   cp.asarray(m.data.astype(np.float64)))


@dataclass
class _Level:
    A: _Csr
    P: _Csr | None      # 延長 (n_l × n_{l+1})
    R: _Csr | None      # 制限 (n_{l+1} × n_l)
    dinv: object
    theta_inv: float
    coefs: list          # Chebyshev の (c1, c2) (ステップ 1..deg-1)
    x: object
    b: object
    r: object
    d: object


class AmgGpuSolver:
    """A_c x = b の GPU AMG-PCG。``A_c`` は scipy の CSR (対称正定値、特異なら零空間は定数)。"""

    def __init__(self, A_c: sp.spmatrix, device: Device | None = None, *, singular: bool = False,
                 degree: int = CHEB_DEGREE, max_coarse: int = MAX_COARSE):
        t0 = time.perf_counter()
        dev = device if device is not None else get_device("cuda")
        if not dev.is_gpu:
            raise RuntimeError("AmgGpuSolver は GPU 専用です")
        import cupy as cp

        from ..device.cuda import get_kernel

        self.cp = cp
        self.device = dev
        A = sp.csr_matrix(A_c)
        self.n = n = int(A.shape[0])
        self.singular = bool(singular)
        self.degree = int(degree)
        self._k = {name: get_kernel("amr", name) for name in (
            "csr_spmv", "csr_resid", "cheb_first", "cheb_next", "vec_axpy", "vec_fill", "vec_sub_mean",
            "vec_sum", "dense_gemv")}
        for name in ("dot_acc", "pcg_xr", "pcg_p", "copy_vec", "copy_scalar"):
            self._k[name] = get_kernel("poisson", name)
        self._s = cp.zeros(8)   # 0: rz, 1: pAp, 2: rz_new, 3: |r|², 4: |b|², 5: 和 (射影用)
        self.direct = n <= DENSE_DIRECT_MAX
        self.levels: list[_Level] = []
        if n == 0:
            self._inv = None
        elif self.direct:
            self._inv = cp.asarray(_dense_inverse(A, self.singular))
        else:
            import pyamg

            ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric", max_coarse=max_coarse)
            mats = [sp.csr_matrix(lv.A) for lv in ml.levels]
            prol = [sp.csr_matrix(lv.P) for lv in ml.levels[:-1]]
            for li, Al in enumerate(mats):
                nl = Al.shape[0]
                last = li == len(mats) - 1
                dinv = 1.0 / Al.diagonal()
                lmax = 1.1 * _lam_max(Al, dinv)
                lmin = lmax / CHEB_RATIO
                theta, delta = 0.5 * (lmax + lmin), 0.5 * (lmax - lmin)
                sigma = theta / delta
                rho = 1.0 / sigma
                coefs = []
                for _ in range(self.degree - 1):
                    rho_new = 1.0 / (2.0 * sigma - rho)
                    coefs.append((rho_new * rho, 2.0 * rho_new / delta))
                    rho = rho_new
                self.levels.append(_Level(
                    A=_Csr.of(Al, cp),
                    P=None if last else _Csr.of(prol[li], cp),
                    R=None if last else _Csr.of(prol[li].T.tocsr(), cp),
                    dinv=cp.asarray(dinv), theta_inv=1.0 / theta, coefs=coefs,
                    x=cp.zeros(nl), b=cp.zeros(nl), r=cp.zeros(nl), d=cp.zeros(nl),
                ))
            self._inv = cp.asarray(_dense_inverse(mats[-1], self.singular))
        self.n_levels = len(self.levels)
        self._A0 = self.levels[0].A if self.levels else (_Csr.of(A, cp) if n else None)
        self._w = {name: cp.zeros(max(n, 1)) for name in ("r", "z", "p", "ap")}
        self.setup_s = time.perf_counter() - t0

    # ---- 基本演算 ------------------------------------------------------------------------

    def _spmv(self, y, m: _Csr, x, mode: int) -> None:
        self._k["csr_spmv"](_grid(m.n), (_B,), (y, m.ip, m.ix, m.a, x, np.int32(m.n), np.int32(mode)))

    def _resid(self, r, b, m: _Csr, x) -> None:
        self._k["csr_resid"](_grid(m.n), (_B,), (r, b, m.ip, m.ix, m.a, x, np.int32(m.n)))

    def _dot(self, a, b, slot: int, n: int) -> None:
        self._s[slot:slot + 1].fill(0.0)
        blocks = min(1024, max(1, (n + 255) // 256))
        self._k["dot_acc"]((blocks,), (256,), (a, b, np.int64(n), self._s[slot:slot + 1]))

    def _project(self, x, n: int) -> None:
        """x −= mean(x) (特異問題の射影)。"""
        self._s[5:6].fill(0.0)
        blocks = min(1024, max(1, (n + 255) // 256))
        self._k["vec_sum"]((blocks,), (256,), (x, np.int32(n), self._s[5:6]))
        self._k["vec_sub_mean"](_grid(n), (_B,), (x, self._s, np.int32(5), np.int32(n)))

    def _gemv(self, inv, b, x, n: int) -> None:
        self._k["dense_gemv"](((n + 127) // 128,), (128,), (inv, b, x, np.int32(n)))

    # ---- V サイクル ----------------------------------------------------------------------

    def _smooth(self, lv: _Level, b, x, zero_init: bool) -> None:
        k = self._k
        n = lv.A.n
        g = _grid(n)
        if zero_init:
            k["copy_vec"](g, (_B,), (lv.r, b, np.int64(n)))
            k["vec_fill"](g, (_B,), (x, np.float64(0.0), np.int32(n)))
        else:
            self._resid(lv.r, b, lv.A, x)
        k["cheb_first"](g, (_B,), (lv.d, lv.r, lv.dinv, np.float64(lv.theta_inv), np.int32(n)))
        for i in range(self.degree):
            k["vec_axpy"](g, (_B,), (x, lv.d, np.float64(1.0), np.int32(n)))
            if i == self.degree - 1:
                break
            self._spmv(lv.r, lv.A, lv.d, 2)
            c1, c2 = lv.coefs[i]
            k["cheb_next"](g, (_B,), (lv.d, lv.r, lv.dinv, np.float64(c1), np.float64(c2), np.int32(n)))

    def _vcycle(self, li: int, b, x) -> None:
        lv = self.levels[li]
        if li == len(self.levels) - 1:
            self._gemv(self._inv, b, x, lv.A.n)
            return
        self._smooth(lv, b, x, True)
        self._resid(lv.r, b, lv.A, x)
        nxt = self.levels[li + 1]
        self._spmv(nxt.b, lv.R, lv.r, 0)
        self._vcycle(li + 1, nxt.b, nxt.x)
        self._spmv(x, lv.P, nxt.x, 1)
        self._smooth(lv, b, x, False)

    def _precond(self, r, z) -> None:
        self._vcycle(0, r, z)
        if self.singular:
            # 前処理 (最粗の擬似逆行列・平滑化) が零空間 (定数) 成分を持ち込むと CG の直交性が崩れて
            # warm start の収束が鈍るので、探索方向を平均 0 の部分空間に保つ
            self._project(z, self.n)

    # ---- 求解 ----------------------------------------------------------------------------

    def launch_solve(self, b, x, n_iter: int) -> None:
        """A x = b を x (warm start、その場で更新) について解くカーネル列を積む (同期・確保なし)。

        密な直接法 (n ≤ DENSE_DIRECT_MAX) では n_iter を無視して厳密に解く。特異問題では b を
        その場で平均 0 に射影する。
        """
        n = self.n
        if n == 0:
            return
        if self.singular:
            self._project(b, n)
        if self.direct:
            self._gemv(self._inv, b, x, n)
            if self.singular:
                self._project(x, n)
            return
        k = self._k
        w = self._w
        g = _grid(n)
        self._resid(w["r"], b, self._A0, x)
        self._precond(w["r"], w["z"])
        k["copy_vec"](g, (_B,), (w["p"], w["z"], np.int64(n)))
        self._dot(w["r"], w["z"], 0, n)
        for _ in range(int(n_iter)):
            self._spmv(w["ap"], self._A0, w["p"], 0)
            self._dot(w["p"], w["ap"], 1, n)
            k["pcg_xr"](g, (_B,), (x, w["r"], w["p"], w["ap"], self._s, np.int32(0), np.int32(1), np.int64(n)))
            self._precond(w["r"], w["z"])
            self._dot(w["r"], w["z"], 2, n)
            k["pcg_p"](g, (_B,), (w["p"], w["z"], self._s, np.int32(2), np.int32(0), np.int64(n)))
            k["copy_scalar"]((1,), (1,), (self._s, np.int32(0), np.int32(2)))
        self._dot(w["r"], w["r"], 3, n)
        self._dot(b, b, 4, n)
        if self.singular:
            self._project(x, n)

    def monitor(self) -> tuple[float, float]:
        """直前の launch_solve の (||r||₂, ||b||₂) (ここでホスト同期)。直接法は (0, 1)。"""
        if self.direct or self.n == 0:
            return 0.0, 1.0
        s = self._s.get()
        return float(np.sqrt(max(s[3], 0.0))), float(np.sqrt(max(s[4], 0.0)))

    def solve(self, b, x0=None, *, tol: float = 1e-10, max_iter: int = 200):
        """A x = b を解く (反復ごとに残差を確認)。b・x0 はホストまたはデバイス配列。

        戻り値: (x (デバイス配列), AmrSolveInfo)。
        """
        cp = self.cp
        t0 = time.perf_counter()
        n = self.n
        b = cp.array(b, dtype=np.float64)
        x = cp.zeros(n) if x0 is None else cp.array(x0, dtype=np.float64)
        if n == 0:
            return x, AmrSolveInfo(0, 0.0, True, 0.0)
        if self.singular:
            b -= b.mean()
        bn = float(cp.linalg.norm(b))
        if self.direct:
            self._gemv(self._inv, b, x, n)
            if self.singular:
                x -= x.mean()
            r = b - self._matvec(x)
            rel = float(cp.linalg.norm(r)) / bn if bn > 0 else 0.0
            return x, AmrSolveInfo(1, rel, rel <= max(tol * 10, 1e-12), time.perf_counter() - t0)
        k = self._k
        w = self._w
        g = _grid(n)
        self._resid(w["r"], b, self._A0, x)
        target = tol * bn
        rn = float(cp.linalg.norm(w["r"]))
        it = 0
        if rn > target:
            self._precond(w["r"], w["z"])
            k["copy_vec"](g, (_B,), (w["p"], w["z"], np.int64(n)))
            self._dot(w["r"], w["z"], 0, n)
            for it in range(1, max_iter + 1):
                self._spmv(w["ap"], self._A0, w["p"], 0)
                self._dot(w["p"], w["ap"], 1, n)
                k["pcg_xr"](g, (_B,), (x, w["r"], w["p"], w["ap"], self._s, np.int32(0), np.int32(1), np.int64(n)))
                rn = float(cp.linalg.norm(w["r"]))
                if rn <= target:
                    break
                self._precond(w["r"], w["z"])
                self._dot(w["r"], w["z"], 2, n)
                k["pcg_p"](g, (_B,), (w["p"], w["z"], self._s, np.int32(2), np.int32(0), np.int64(n)))
                k["copy_scalar"]((1,), (1,), (self._s, np.int32(0), np.int32(2)))
        if self.singular:
            x -= x.mean()
        rel = rn / bn if bn > 0 else 0.0
        return x, AmrSolveInfo(it, rel, rn <= target, time.perf_counter() - t0)

    def _matvec(self, x):
        y = self.cp.zeros(self.n)
        self._spmv(y, self._A0, x, 0)
        return y
