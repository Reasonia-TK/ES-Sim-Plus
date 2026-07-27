"""Optional Numba JIT カーネル (prompts/76、粒子カーネルの高速化②)。

粒子ループ (walk・電荷デポジット・gather+push・MCCプロセス選択) を Numba の njit で書き直し、
numpy ベクトル化実装より高速化する。numba は **optional 依存**: import に
失敗する環境 (未インストール) や環境変数 ES_SIM_NO_NUMBA=1 (計測・テスト用の
強制フォールバック) では HAVE_NUMBA=False のままとなり、呼び出し側
(particles.py / pic.py / dsmc.py) は自動的に従来の numpy 実装へフォールバックする
(呼び出し側の分岐は「if HAVE_NUMBA」だけで、アルゴリズムの二重管理を避けるため
 numpy 実装は各モジュールの既存コードをそのまま残す)。

決定性の設計方針 (仕様書 prompts/76):
  - walk は「粒子ごとに自分の行にしか書かない」独立処理なので prange
    (parallel=True) で並列化しても、各粒子内の演算を numpy 版と同じ順序で
    行えば結果はビット単位で一致する (浮動小数点の加算・除算は結合則を
    満たさないため、演算「順序」の一致が本質。並列実行の「どの粒子から先に
    処理するか」は各粒子が独立である限り結果に影響しない)。
  - 電荷デポジットは複数粒子が同じ節点へ加算する「リダクション」なので、
    prange で並列化すると加算順序がスレッド数・スケジューリングに依存して
    しまい、numpy 版 (bincount。内部的には粒子順の逐次加算と等価) と
    一致しなくなる。そのため deposit カーネルは並列化せず逐次 njit のみで
    実装する。
  - gather+push (基本経路: 軸対称/一様磁場なし) も「粒子ごとに自分の行にしか
    書かない」処理なので prange で並列化してよい。ただし運動エネルギーの
    総和は「リダクション」なので、カーネル内では合成せず粒子ごとの vdot
    (速度の内積) 配列を返すだけにとどめ、総和は呼び出し側で numpy の
    np.sum に委ねる (numpy 版と同じ pairwise 和アルゴリズムになるため
    ビット単位一致を保てる)。
  - 軸対称 (rz)・一様磁場 (B) ありの push は分岐が多く、対応する njit
    カーネルは用意していない (numpy 実装のみ)。プロファイル (prompts/75) の
    支配的ケースが xy・無磁場だったため優先度を下げた (詳細は
    prompts/76 の報告を参照)。
"""

from __future__ import annotations

import os

import numpy as np

HAVE_NUMBA = False
try:
    if os.environ.get("ES_SIM_NO_NUMBA") == "1":
        # 計測・テスト用の強制フォールバック (numba 未インストール環境の動作確認、
        # および numba あり/なしのベンチマーク前後比較に使う)
        raise ImportError("ES_SIM_NO_NUMBA=1")
    import numba
    from numba import njit, prange

    HAVE_NUMBA = True
except ImportError:
    numba = None  # type: ignore[assignment]

# particles._walk_step と同じ定数 (循環 import を避けるためここでも定義する。
# 値がずれると walk の挙動が食い違うため、変更時は両方合わせること)
_MAX_WALK_ITERS = 64
_TOL = 1e-9


def set_num_threads(n: int) -> None:
    """numba のスレッド数を pic.threads / dsmc.threads に合わせる。

    numba 無し環境では no-op (numpy フォールバック側は従来どおり
    ThreadPoolExecutor のワーカー数で並列度を制御する)。
    pic.threads / dsmc.threads はコア数を超える値をユーザーが指定できてしまう
    (numpy フォールバックの ThreadPoolExecutor は超過分もそのまま使える) のに対し、
    numba.set_num_threads は物理的な起動時スレッド数 (NUMBA_NUM_THREADS) を
    超える値を渡すと例外になるため、ここで上限にクランプする。
    """
    if HAVE_NUMBA:
        n_max = numba.config.NUMBA_NUM_THREADS
        numba.set_num_threads(max(1, min(int(n), n_max)))


if HAVE_NUMBA:

    @njit(cache=True, nogil=True, parallel=True)
    def _walk_kernel(
        packed, adjacency, elem0, xs, ys, tol, max_iters,
        out_elem, out_absorbed, out_b_elem, out_b_loc, out_l,
    ):
        n = xs.shape[0]
        for p in prange(n):
            e = elem0[p]
            xp = xs[p]
            yp = ys[p]
            absorbed = False
            b_elem = 0
            b_loc = 0
            converged = False
            l0 = 0.0
            l1 = 0.0
            l2 = 0.0
            for _ in range(max_iters):
                g9 = packed[e, 9]
                l0 = (packed[e, 0] + packed[e, 3] * xp) + packed[e, 6] * yp
                l0 = l0 / g9
                l1 = (packed[e, 1] + packed[e, 4] * xp) + packed[e, 7] * yp
                l1 = l1 / g9
                l2 = (packed[e, 2] + packed[e, 5] * xp) + packed[e, 8] * yp
                l2 = l2 / g9
                if l0 >= -tol and l1 >= -tol and l2 >= -tol:
                    converged = True
                    break
                mi = 0
                mv = l0
                if l1 < mv:
                    mi = 1
                    mv = l1
                if l2 < mv:
                    mi = 2
                    mv = l2
                nb = adjacency[e, mi]
                if nb == -1:
                    absorbed = True
                    b_elem = e
                    b_loc = mi
                    break
                e = nb
            if not converged and not absorbed:
                # 反復上限に達した (実運用ではまず起きない): numpy 版の
                # for-else フォールバックと同じく、最終要素で L を再評価する
                g9 = packed[e, 9]
                l0 = (packed[e, 0] + packed[e, 3] * xp) + packed[e, 6] * yp
                l0 = l0 / g9
                l1 = (packed[e, 1] + packed[e, 4] * xp) + packed[e, 7] * yp
                l1 = l1 / g9
                l2 = (packed[e, 2] + packed[e, 5] * xp) + packed[e, 8] * yp
                l2 = l2 / g9
            out_elem[p] = e
            out_absorbed[p] = absorbed
            out_b_elem[p] = b_elem
            out_b_loc[p] = b_loc
            out_l[p, 0] = l0
            out_l[p, 1] = l1
            out_l[p, 2] = l2

    @njit(cache=True, nogil=True)
    def _deposit_kernel(nidx, bary, q, w, n_nodes):
        f = np.zeros(n_nodes)
        n = nidx.shape[0]
        for p in range(n):
            c = q * w[p]
            f[nidx[p, 0]] += c * bary[p, 0]
            f[nidx[p, 1]] += c * bary[p, 1]
            f[nidx[p, 2]] += c * bary[p, 2]
        return f

    @njit(cache=True, nogil=True, parallel=True)
    def _gather_push_kernel(exy, elem, qm_dt, dt_sp, x, v, out_vnew, out_xnew, out_vdot):
        n = elem.shape[0]
        for p in prange(n):
            e = elem[p]
            ax = qm_dt * exy[e, 0]
            ay = qm_dt * exy[e, 1]
            v0 = v[p, 0]
            v1 = v[p, 1]
            v2 = v[p, 2]
            vn0 = v0 + ax
            vn1 = v1 + ay
            vn2 = v2
            out_vnew[p, 0] = vn0
            out_vnew[p, 1] = vn1
            out_vnew[p, 2] = vn2
            out_xnew[p, 0] = x[p, 0] + dt_sp * vn0
            out_xnew[p, 1] = x[p, 1] + dt_sp * vn1
            out_vdot[p] = (v0 * vn0 + v1 * vn1) + v2 * vn2

    @njit(cache=True, nogil=True)
    def _interp_packed_table(x, xs, ys, n):
        """np.interp と同じ端点クランプ・右側探索で1点を線形補間する。"""
        if x < xs[0]:
            return ys[0]
        if x >= xs[n - 1]:
            return ys[n - 1]
        # searchsorted(..., side="right") 相当。重複xがある場合も右側を選ぶ。
        lo = 0
        hi = n
        while lo < hi:
            mid = (lo + hi) // 2
            if x < xs[mid]:
                hi = mid
            else:
                lo = mid + 1
        right = lo
        left = right - 1
        dx = xs[right] - xs[left]
        if dx == 0.0:
            return ys[right]
        return ys[left] + (x - xs[left]) * (ys[right] - ys[left]) / dx

    @njit(cache=True, nogil=True)
    def _mcc_choose_process_kernel(
        e_ev, speed, numax, n_ref, rel, use_rel, e_table, s_table, lengths, random_u
    ):
        """候補ごとの断面積補間と累積頻度選択を一時2D配列なしで行う。"""
        n = e_ev.shape[0]
        n_procs = lengths.shape[0]
        out = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            target = random_u[i] * numax
            cumulative = 0.0
            density_scale = rel[i] if use_rel else 1.0
            for j in range(n_procs):
                sigma = _interp_packed_table(
                    e_ev[i], e_table[j], s_table[j], lengths[j]
                )
                # numpy経路の (n_ref * sigma * speed) * rel と演算順を揃える。
                nu_j = (n_ref * sigma * speed[i]) * density_scale
                cumulative += nu_j
                if target < cumulative:
                    out[i] = j
                    break
        return out


def walk_step(coeffs, adjacency, elem0, x_new, l_out=None, packed=None):
    """particles._walk_step_numpy の numba 版。戻り値・意味は完全に同じ。

    l_out 引数は numpy 版と同じ「absorbed 粒子の行は未定義のまま (書き込まない)」
    という契約を守る (呼び出し側が absorbed 行を読まない前提のコードに
    合わせるため。書き込んでしまうと numpy 版との等価性テストで
    未初期化領域の違いにより np.array_equal が偽陰性になり得る)。
    """
    if packed is None:
        from .particles import _pack_coeffs

        packed = _pack_coeffs(coeffs)
    n = len(x_new)
    elem = np.empty(n, dtype=np.int64)
    absorbed = np.empty(n, dtype=np.bool_)
    b_elem = np.empty(n, dtype=np.int64)
    b_loc = np.empty(n, dtype=np.int64)
    l_buf = np.empty((n, 3), dtype=np.float64)
    _walk_kernel(
        packed,
        adjacency,
        elem0,
        np.ascontiguousarray(x_new[:, 0]),
        np.ascontiguousarray(x_new[:, 1]),
        _TOL,
        _MAX_WALK_ITERS,
        elem,
        absorbed,
        b_elem,
        b_loc,
        l_buf,
    )
    if l_out is not None:
        keep = ~absorbed
        l_out[keep] = l_buf[keep]
    return elem, absorbed, b_elem, b_loc


def deposit(nidx: np.ndarray, bary: np.ndarray, q: float, w: np.ndarray, n_nodes: int) -> np.ndarray:
    """電荷堆積 (P1 重み散布) の numba 版。pic._deposit_species の bincount 経路と等価。"""
    return _deposit_kernel(nidx, bary, float(q), w, int(n_nodes))


def gather_push(exy: np.ndarray, elem: np.ndarray, q: float, m: float, dt_sp: float,
                 x: np.ndarray, v: np.ndarray):
    """基本経路 (軸対称・一様磁場なし) の gather (E補間) + リープフロッグ push を融合する。

    戻り値: (v_new, x_new, vdot)。vdot は sp.v・v_new の内積 (時刻中心化運動
    エネルギー用)。総和 (リダクション) は呼び出し側で np.sum(sp.w * vdot) と
    して行う (numpy 版と同じ和のとり方にして決定性・等価性を保つため)。
    """
    n = len(x)
    v_new = np.empty((n, 3), dtype=np.float64)
    x_new = np.empty((n, 2), dtype=np.float64)
    vdot = np.empty(n, dtype=np.float64)
    qm_dt = (q / m) * dt_sp
    _gather_push_kernel(
        np.ascontiguousarray(exy), np.ascontiguousarray(elem), qm_dt, dt_sp,
        np.ascontiguousarray(x), np.ascontiguousarray(v), v_new, x_new, vdot,
    )
    return v_new, x_new, vdot


def mcc_choose_process(
    e_ev: np.ndarray,
    speed: np.ndarray,
    numax: float,
    n_ref: float,
    rel: np.ndarray | None,
    e_table: np.ndarray,
    s_table: np.ndarray,
    lengths: np.ndarray,
    random_u: np.ndarray,
) -> np.ndarray:
    """MCC候補の実プロセス番号を返す (-1=null)。

    乱数は呼び出し側のnumpy Generatorで従来と同じ順に生成して渡す。ここでは
    候補×プロセスのnu/cumsum行列を作らず、候補ごとの小さな逐次ループで選ぶ。
    """
    rel_arr = np.empty(0, dtype=np.float64) if rel is None else np.ascontiguousarray(rel)
    return _mcc_choose_process_kernel(
        np.ascontiguousarray(e_ev),
        np.ascontiguousarray(speed),
        float(numax),
        float(n_ref),
        rel_arr,
        rel is not None,
        np.ascontiguousarray(e_table),
        np.ascontiguousarray(s_table),
        np.ascontiguousarray(lengths),
        np.ascontiguousarray(random_u),
    )
