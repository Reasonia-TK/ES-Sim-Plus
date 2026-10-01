"""幾何マルチグリッド (GMG) 前処理付き共役勾配法 — v2 EB Poisson ソルバー (prompts/119)。

AMReX の MLMG に倣い、各レベルの演算子は**幾何から再離散化**する (eb.build.build_level を
粗い格子で呼ぶ)。CAD 形状が解析的なので、粗いレベルでも導体の交差率 θ・誘電体の合成を
厳密に作り直せる (代数的な Galerkin 粗視化より安く、EB の意味を保つ)。

- 平滑化: 赤黒 Gauss-Seidel。前平滑化 (赤→黒) と後平滑化 (黒→赤) を逆順にして
  V サイクル全体を対称にする (CG の前処理は対称正定値である必要がある)。
- 制限: R = Pᵀ (重み 1, 1/2, 1/4 の和。FV の残差は体積積分量なので正規化しない)。
- 延長: 双一次補間。粗視化は各方向 1 または 2 倍 (片方向だけの半粗視化も可)。
- 最粗レベル: 未知数 ≤ 数百の密行列を事前に逆行列化して直接解く (特異なら擬似逆行列)。
- 外側反復: 前処理付き CG。PIC では剛性が不変なので本クラスを一度作り、毎ステップ
  右辺だけ変えて前ステップの解から解き直す (warm start)。
- Dirichlet 節点を持たない問題 (全周 Neumann/周期) は定数の零空間を持つ。右辺を
  整合化 (和 0) し、解は体積重み平均 0 に正規化する。

CPU/GPU どちらでも同じコードで動く (配列は Device.xp、カーネルは backend 経由)。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..device import Device, get_device
from ..eb.build import MASK_FIXED, MASK_UNKNOWN, LevelOperator, build_level
from ..eb.grid import CartesianGrid
from ..geom.model import GeometryModel

#: 最粗レベルの直接法に回す未知数の上限
MAX_BOTTOM_UNKNOWNS = 1024
#: GPU の非同期求解 (PIC) で、問題全体を密な逆行列で直接解く未知数の上限
#: (逆行列 4096² × 8 B = 134 MB。毎ステップ 1 回の行列ベクトル積で厳密解が得られ、
#: 小さな格子で V サイクルのカーネル起動数が律速になるのを避ける)
DENSE_DIRECT_MAX = 4096


@dataclass
class SolveInfo:
    iterations: int
    residual_norm: float
    rhs_norm: float
    converged: bool
    elapsed_s: float

    @property
    def relative_residual(self) -> float:
        return self.residual_norm / self.rhs_norm if self.rhs_norm > 0.0 else 0.0


@dataclass
class _DevLevel:
    nx: int
    ny: int
    px: bool
    py: bool
    cx: object
    cy: object
    diag: object
    mask: object
    x: object
    b: object
    r: object


class _CpuBackend:
    def __init__(self) -> None:
        from . import _cpu_kernels as k

        self.k = k

    def smooth(self, lv: _DevLevel, color: int) -> None:
        self.k.rbgs(lv.x, lv.b, lv.cx, lv.cy, lv.diag, lv.mask, lv.px, lv.py, color)

    def residual(self, lv: _DevLevel) -> None:
        self.k.residual(lv.r, lv.x, lv.b, lv.cx, lv.cy, lv.diag, lv.mask, lv.px, lv.py)

    def apply(self, lv: _DevLevel, x, y) -> None:
        self.k.apply_op(y, x, lv.cx, lv.cy, lv.diag, lv.mask, lv.px, lv.py)

    def restrict(self, lf: _DevLevel, lc: _DevLevel, fx: int, fy: int) -> None:
        self.k.restrict_fw(lc.b, lf.r, lc.mask, fx, fy, lf.px, lf.py)

    def prolong(self, lc: _DevLevel, lf: _DevLevel, fx: int, fy: int) -> None:
        self.k.prolong_add(lf.x, lc.x, lf.mask, fx, fy, lf.px, lf.py)


class _CudaBackend:
    def __init__(self) -> None:
        from ..device.cuda import get_kernel, grid_2d

        self._grid_2d = grid_2d
        self.k_rbgs = get_kernel("poisson", "rbgs")
        self.k_res = get_kernel("poisson", "residual")
        self.k_apply = get_kernel("poisson", "apply_op")
        self.k_restrict = get_kernel("poisson", "restrict_fw")
        self.k_prolong = get_kernel("poisson", "prolong_add")
        self.k_gemv = get_kernel("poisson", "dense_gemv_idx")

    @staticmethod
    def _i(v) -> np.int32:
        return np.int32(int(v))

    def smooth(self, lv: _DevLevel, color: int) -> None:
        g, b = self._grid_2d(lv.nx + 1, lv.ny + 1)
        i = self._i
        self.k_rbgs(g, b, (lv.x, lv.b, lv.cx, lv.cy, lv.diag, lv.mask,
                           i(lv.nx), i(lv.ny), i(lv.px), i(lv.py), i(color)))

    def residual(self, lv: _DevLevel) -> None:
        g, b = self._grid_2d(lv.nx + 1, lv.ny + 1)
        i = self._i
        self.k_res(g, b, (lv.r, lv.x, lv.b, lv.cx, lv.cy, lv.diag, lv.mask,
                          i(lv.nx), i(lv.ny), i(lv.px), i(lv.py)))

    def apply(self, lv: _DevLevel, x, y) -> None:
        g, b = self._grid_2d(lv.nx + 1, lv.ny + 1)
        i = self._i
        self.k_apply(g, b, (y, x, lv.cx, lv.cy, lv.diag, lv.mask,
                            i(lv.nx), i(lv.ny), i(lv.px), i(lv.py)))

    def restrict(self, lf: _DevLevel, lc: _DevLevel, fx: int, fy: int) -> None:
        g, b = self._grid_2d(lc.nx + 1, lc.ny + 1)
        i = self._i
        self.k_restrict(g, b, (lc.b, lf.r, lc.mask, i(lc.nx), i(lc.ny), i(lf.nx), i(lf.ny),
                               i(fx), i(fy), i(lf.px), i(lf.py)))

    def prolong(self, lc: _DevLevel, lf: _DevLevel, fx: int, fy: int) -> None:
        g, b = self._grid_2d(lf.nx + 1, lf.ny + 1)
        i = self._i
        self.k_prolong(g, b, (lf.x, lc.x, lf.mask, i(lf.nx), i(lf.ny), i(lc.nx), i(lc.ny),
                              i(fx), i(fy), i(lf.px), i(lf.py)))


def dense_inverse(a, singular: bool, xp=np):
    """対称な密行列の逆行列。特異 (零空間 = 定数、全周期/Neumann) なら擬似逆行列。

    擬似逆行列は SVD (4096 元で ~20 s) の代わりに、零空間 u = 1/√n を持ち上げた正則行列の逆から
    A⁺ = (A + α u uᵀ)⁻¹ − u uᵀ/α で求める (A u = 0 の対称半正定値行列で厳密)。零空間が定数で
    ない場合 (想定外) は SVD にフォールバックする。

    xp: 配列モジュール (numpy / cupy)。GPU で使う逆行列は GPU で求める — 開発機の
    マルチスレッド OpenBLAS は LAPACK の呼び出しに行列の大きさによらず ~1.5 s かかるため
    (750 元の逆行列: OpenBLAS 1.5 s、1 スレッド 0.03 s、GPU 0.005 s)。GPU は cuSOLVER ではなく自前の
    ブロック Gauss-Jordan (``device.linalg.spd_inverse``、配布版は NVRTC だけを同梱するため、prompts/133)。
    """
    if xp is not np:
        return _dense_inverse_gpu(np.asarray(a.get() if hasattr(a, "get") else a, dtype=np.float64), singular, xp)
    a = xp.asarray(a, dtype=xp.float64)
    if not singular:
        return xp.linalg.inv(a)
    n = a.shape[0]
    if n == 0:
        return xp.zeros((0, 0))
    u = xp.full(n, 1.0 / np.sqrt(n))
    scale = float(xp.max(xp.abs(xp.diag(a)))) or 1.0
    if float(xp.max(xp.abs(a @ u))) > 1e-9 * scale:
        return xp.linalg.pinv(a)
    uu = xp.outer(u, u)
    return xp.linalg.inv(a + scale * uu) - uu / scale


def _dense_inverse_gpu(a: np.ndarray, singular: bool, cp):
    """dense_inverse の GPU 版 (a はホストの配列、戻り値は cupy 配列)。判定はホストで、逆行列は GPU で。"""
    from ..device.linalg import spd_inverse

    if not singular:
        return spd_inverse(a)
    n = a.shape[0]
    if n == 0:
        return cp.zeros((0, 0))
    u = np.full(n, 1.0 / np.sqrt(n))
    scale = float(np.max(np.abs(np.diag(a)))) or 1.0
    if float(np.max(np.abs(a @ u))) > 1e-9 * scale:
        return cp.asarray(np.linalg.pinv(a))  # 零空間が定数でない (想定外) ときだけ CPU の SVD
    uu = np.outer(u, u)
    return spd_inverse(a + scale * uu) - cp.asarray(uu / scale)


def _dense_operator(op: LevelOperator) -> tuple[np.ndarray, np.ndarray]:
    """小さなレベルの演算子を未知節点だけの密行列にする。戻り値: (未知節点の平坦番号, A)。"""
    g = op.grid
    nx, ny = g.nx, g.ny
    mask = op.mask.ravel()
    idx = np.nonzero(mask == MASK_UNKNOWN)[0]
    pos = np.full(mask.size, -1, dtype=np.int64)
    pos[idx] = np.arange(idx.size)
    a = np.zeros((idx.size, idx.size))
    a[np.arange(idx.size), np.arange(idx.size)] = op.diag.ravel()[idx]

    def canon(i, j):
        if op.periodic_x and i == nx:
            i = 0
        if op.periodic_y and j == ny:
            j = 0
        return j * (nx + 1) + i

    for j in range(ny + 1):
        for i in range(nx):
            c = op.cx[j, i]
            if c != 0.0:
                p, q = pos[canon(i, j)], pos[canon(i + 1, j)]
                if p >= 0 and q >= 0:
                    a[p, q] -= c
                    a[q, p] -= c
    for j in range(ny):
        for i in range(nx + 1):
            c = op.cy[j, i]
            if c != 0.0:
                p, q = pos[canon(i, j)], pos[canon(i, j + 1)]
                if p >= 0 and q >= 0:
                    a[p, q] -= c
                    a[q, p] -= c
    return idx, a


@dataclass
class GMGSolver:
    """EB Poisson の GMG-PCG ソルバー。``model``・``grid`` から全レベルを組む。"""

    model: GeometryModel
    grid: CartesianGrid
    device: Device = field(default_factory=get_device)
    n_pre: int = 2
    n_post: int = 2

    def __post_init__(self) -> None:
        t0 = time.perf_counter()
        ops = [build_level(self.model, self.grid, full=True)]
        factors: list[tuple[int, int]] = []
        g = self.grid
        while ops[-1].n_unknowns > MAX_BOTTOM_UNKNOWNS:
            fx, fy = g.coarsening()
            if (fx, fy) == (1, 1):
                break
            g = g.coarsened(fx, fy)
            ops.append(build_level(self.model, g, full=False))
            factors.append((fx, fy))
        self.ops = ops
        self.factors = factors
        self.finest = ops[0]
        self.singular = not bool(np.any(ops[0].mask == MASK_FIXED))

        dev = self.device
        self.xp = dev.xp
        self.backend = _CudaBackend() if dev.is_gpu else _CpuBackend()
        self.levels: list[_DevLevel] = []
        for op in ops:
            shp = op.grid.shape
            self.levels.append(
                _DevLevel(
                    nx=op.grid.nx, ny=op.grid.ny, px=op.periodic_x, py=op.periodic_y,
                    cx=dev.asarray(op.cx, dtype=np.float64),
                    cy=dev.asarray(op.cy, dtype=np.float64),
                    diag=dev.asarray(op.diag, dtype=np.float64),
                    mask=dev.asarray(op.mask, dtype=np.uint8),
                    x=dev.zeros(shp), b=dev.zeros(shp), r=dev.zeros(shp),
                )
            )
        # 最粗レベルの直接法
        idx, a = _dense_operator(ops[-1])
        self._bottom_idx = dev.asarray(idx, dtype=np.int64)
        if idx.size:
            inv = dense_inverse(a, self.singular, xp=dev.xp)
        else:
            inv = np.zeros((0, 0))
        self._bottom_inv = dev.asarray(inv, dtype=np.float64)
        # 最細レベルの未知マスクと体積 (特異問題の正規化用)
        self._unknown = dev.asarray(ops[0].mask == MASK_UNKNOWN)
        self._n_unknown = int(ops[0].n_unknowns)
        vol = ops[0].vol if ops[0].vol is not None else np.ones(ops[0].grid.shape)
        self._vol = dev.asarray(np.where(ops[0].mask == MASK_UNKNOWN, vol, 0.0))
        self.setup_s = time.perf_counter() - t0

    # ---- 演算子・前処理 ------------------------------------------------------------

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def apply(self, x, out=None):
        """y = A x (未知節点のみ、他は 0)。"""
        y = self.xp.empty_like(x) if out is None else out
        self.backend.apply(self.levels[0], x, y)
        return y

    def _bottom(self, lv: _DevLevel) -> None:
        if self._bottom_idx.size == 0:
            return
        if self.device.is_gpu:
            # 同期・確保なしの融合カーネル (CUDA Graph に取り込めるように。cuBLAS も使わない、prompts/133)
            m = int(self._bottom_idx.size)
            self.backend.k_gemv(
                ((m + 127) // 128,), (128,),
                (self._bottom_inv, self._bottom_idx, lv.b, lv.x, np.int32(m)),
            )
            return
        xv = lv.x.reshape(-1)
        bv = lv.b.reshape(-1)
        xv[self._bottom_idx] = self._bottom_inv @ bv[self._bottom_idx]

    # ---- GPU 非同期求解 (PIC 用: ホスト同期なしでカーネルを積むだけ) ----------------------

    def enable_async(self) -> None:
        """GPU で launch_solve を使う準備 (作業ベクトル・スカラー枠・直接法の逆行列)。

        launch_solve はホストとの同期 (内積の値の読み出し等) を一切行わず、決まった反復数の
        PCG をカーネル列としてキューに積む。CUDA Graph にそのまま取り込める
        (実行中に配列確保をしない)。収束の監視は monitor() で必要なときだけ行う。
        """
        if not self.device.is_gpu:
            raise RuntimeError("enable_async は GPU 専用です")
        if getattr(self, "_async", None) is not None:
            return
        from ..device.cuda import get_kernel

        cp = self.xp
        shp = self.grid.shape
        k = {
            name: get_kernel("poisson", name)
            for name in ("dot_acc", "pcg_xr", "pcg_p", "copy_vec", "copy_scalar", "dense_gemv_idx")
        }
        k["gemv"] = k["dense_gemv_idx"]
        self._async = k
        self._w = {name: cp.zeros(shp) for name in ("r", "z", "p", "ap")}
        self._s = cp.zeros(8)  # 0: rz, 1: pAp, 2: rz_new, 3: |r|², 4: |b|²
        self.direct = self.finest.n_unknowns <= DENSE_DIRECT_MAX
        if self.direct:
            idx, a = _dense_operator(self.finest)
            self._direct_idx = cp.asarray(idx.astype(np.int64))
            self._direct_inv = dense_inverse(a, self.singular, xp=cp)

    def _launch_dot(self, a, b, slot: int) -> None:
        n = a.size
        self._s[slot:slot + 1].fill(0.0)
        blocks = min(1024, max(1, (n + 255) // 256))
        self._async["dot_acc"]((blocks,), (256,), (a, b, np.int64(n), self._s[slot:slot + 1]))

    def _launch_precond(self, r, z) -> None:
        lv = self.levels[0]
        n = r.size
        g = ((n + 255) // 256,)
        self._async["copy_vec"](g, (256,), (lv.b, r, np.int64(n)))
        lv.x.fill(0.0)
        self._vcycle(0)
        self._async["copy_vec"](g, (256,), (z, lv.x, np.int64(n)))

    def launch_solve(self, b, x, n_iter: int) -> None:
        """A x = b を x (warm start、その場で更新) について解くカーネル列を積む (同期なし)。

        小さな問題 (未知数 ≤ DENSE_DIRECT_MAX) は密な逆行列で厳密に解く (n_iter は無視)。
        それ以外は GMG 前処理付き CG を n_iter 反復。b・x の固定節点の値は無視・不変。
        """
        k = self._async
        if self.direct:
            m = int(self._direct_idx.size)
            k["gemv"](((m + 127) // 128,), (128,), (self._direct_inv, self._direct_idx, b, x, np.int32(m)))
            return
        w = self._w
        n = b.size
        g = ((n + 255) // 256,)
        lv0 = self.levels[0]
        self.backend.k_res(*self.backend._grid_2d(lv0.nx + 1, lv0.ny + 1),
                           (w["r"], x, b, lv0.cx, lv0.cy, lv0.diag, lv0.mask,
                            np.int32(lv0.nx), np.int32(lv0.ny), np.int32(lv0.px), np.int32(lv0.py)))
        self._launch_precond(w["r"], w["z"])
        k["copy_vec"](g, (256,), (w["p"], w["z"], np.int64(n)))
        self._launch_dot(w["r"], w["z"], 0)
        for _ in range(int(n_iter)):
            self.backend.apply(lv0, w["p"], w["ap"])
            self._launch_dot(w["p"], w["ap"], 1)
            k["pcg_xr"](g, (256,), (x, w["r"], w["p"], w["ap"], self._s, np.int32(0), np.int32(1), np.int64(n)))
            self._launch_precond(w["r"], w["z"])
            self._launch_dot(w["r"], w["z"], 2)
            k["pcg_p"](g, (256,), (w["p"], w["z"], self._s, np.int32(2), np.int32(0), np.int64(n)))
            k["copy_scalar"]((1,), (1,), (self._s, np.int32(0), np.int32(2)))
        self._launch_dot(w["r"], w["r"], 3)
        self._launch_dot(b, b, 4)

    def monitor(self) -> tuple[float, float]:
        """直前の launch_solve の (||r||₂, ||b||₂) を返す (ここでホスト同期する)。

        直接法では常に (0, ||b||) 相当とみなして (0.0, 1.0) を返す。
        """
        if self.direct:
            return 0.0, 1.0
        s = self._s.get()
        return float(np.sqrt(max(s[3], 0.0))), float(np.sqrt(max(s[4], 0.0)))

    def _vcycle(self, lvl: int) -> None:
        lv = self.levels[lvl]
        if lvl == len(self.levels) - 1:
            self._bottom(lv)
            return
        be = self.backend
        for _ in range(self.n_pre):
            be.smooth(lv, 0)
            be.smooth(lv, 1)
        be.residual(lv)
        lc = self.levels[lvl + 1]
        fx, fy = self.factors[lvl]
        be.restrict(lv, lc, fx, fy)
        lc.x.fill(0.0)
        self._vcycle(lvl + 1)
        be.prolong(lc, lv, fx, fy)
        for _ in range(self.n_post):
            be.smooth(lv, 1)
            be.smooth(lv, 0)

    def precondition(self, r):
        """z = M⁻¹ r (V サイクル 1 回、初期値 0)。"""
        lv = self.levels[0]
        lv.b[...] = r
        lv.x.fill(0.0)
        self._vcycle(0)
        return lv.x.copy()

    # ---- 求解 --------------------------------------------------------------------

    def solve(self, b, x0=None, *, tol: float = 1e-10, atol: float = 0.0, max_iter: int = 200):
        """A x = b を解く。b・x0 は (ny+1, nx+1) のデバイス配列 (固定節点の値は無視)。

        戻り値: (x, SolveInfo)。x の固定節点・周期スレーブは 0 (呼び出し側で埋める)。
        収束判定は ||r||₂ ≤ max(tol·||b||₂, atol)。
        """
        xp = self.xp
        if xp is np:
            vdot = np.vdot
        else:  # cupy.vdot は cuBLAS を使う (配布版は NVRTC だけを同梱する、prompts/133)
            from ..device.linalg import vdot
        t0 = time.perf_counter()
        unknown = self._unknown
        b = xp.where(unknown, b, 0.0)
        if self.singular and self._n_unknown:
            b = xp.where(unknown, b - xp.sum(b) / self._n_unknown, 0.0)
        x = xp.zeros_like(b) if x0 is None else xp.where(unknown, x0, 0.0)
        ax = self.apply(x)
        r = b - ax
        bnorm = float(xp.sqrt(vdot(b, b)))
        target = max(tol * bnorm, atol)
        rnorm = float(xp.sqrt(vdot(r, r)))
        it = 0
        if rnorm > target:
            z = self.precondition(r)
            p = z.copy()
            rz = float(vdot(r, z))
            ap = xp.empty_like(b)
            for it in range(1, max_iter + 1):
                self.apply(p, ap)
                pap = float(vdot(p, ap))
                if pap <= 0.0:
                    break
                alpha = rz / pap
                x += alpha * p
                r -= alpha * ap
                rnorm = float(xp.sqrt(vdot(r, r)))
                if rnorm <= target:
                    break
                z = self.precondition(r)
                rz_new = float(vdot(r, z))
                beta = rz_new / rz
                rz = rz_new
                p *= beta
                p += z
        if self.singular:
            vt = float(xp.sum(self._vol))
            if vt > 0.0:
                x = xp.where(unknown, x - float(xp.sum(x * self._vol)) / vt, 0.0)
        info = SolveInfo(
            iterations=it,
            residual_norm=rnorm,
            rhs_norm=bnorm,
            converged=rnorm <= target,
            elapsed_s=time.perf_counter() - t0,
        )
        return x, info
