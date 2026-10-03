"""Optional Numba JIT カーネル (prompts/76、粒子カーネルの高速化②)。

粒子ループ (walk・電荷デポジット・gather+push・MCC候補/選択/散乱) を Numba の njit で書き直し、
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
  - 軸対称 (rz / rz_x0) は回転法の push (E の蹴り → 局所座標の3D直線移動 →
    子午面への回転、角運動量厳密保存) と walk を融合する。一様磁場 (B) ありの
    push は分岐が多いため numpy 実装を維持する。
"""

from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp

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
# MCC候補選択のprange起動コストを回収できる候補数。2万粒子級では直列版が
# 速く、10万粒子級では並列版が有効になるため、その間を保守的な閾値とする。
_MCC_PARALLEL_MIN_CANDIDATES = 32_768


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
        packed, adjacency, elem0, x_new, tol, max_iters,
        out_elem, out_absorbed, out_b_elem, out_b_loc, out_l,
    ):
        n = x_new.shape[0]
        for p in prange(n):
            e = elem0[p]
            xp = x_new[p, 0]
            yp = x_new[p, 1]
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

    @njit(cache=True, nogil=True)
    def _deposit_elements_kernel(tris, elem, bary, q, w, n_nodes):
        """所属要素から節点番号を直接引き、nidx中間配列なしで堆積する。"""
        f = np.zeros(n_nodes)
        n = elem.shape[0]
        for p in range(n):
            e = elem[p]
            c = q * w[p]
            f[tris[e, 0]] += c * bary[p, 0]
            f[tris[e, 1]] += c * bary[p, 1]
            f[tris[e, 2]] += c * bary[p, 2]
        return f

    @njit(cache=True, nogil=True)
    def _compact_particle_state_kernel(
        x, v, w, elem, bary, removed, out_w,
    ):
        """吸収されなかった粒子を順序を保って各配列の先頭へ詰める。"""
        write = 0
        for read in range(removed.shape[0]):
            if removed[read]:
                continue
            if write != read:
                x[write, 0] = x[read, 0]
                x[write, 1] = x[read, 1]
                v[write, 0] = v[read, 0]
                v[write, 1] = v[read, 1]
                v[write, 2] = v[read, 2]
                elem[write] = elem[read]
                bary[write, 0] = bary[read, 0]
                bary[write, 1] = bary[read, 1]
                bary[write, 2] = bary[read, 2]
            # wは更新前状態として境界処理まで保持するため、別バッファへ書く。
            out_w[write] = w[read]
            write += 1
        return write

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

    @njit(cache=True, nogil=True, parallel=True)
    def _gather_push_boris_kernel(
        exy, elem, half, dt_sp, boris_rt, x, v, out_vnew, out_xnew, out_vdot,
    ):
        """一様磁場のgather・半キック・Boris回転・driftを融合する。"""
        n = elem.shape[0]
        for p in prange(n):
            e = elem[p]
            kick0 = half * exy[e, 0]
            kick1 = half * exy[e, 1]
            v0 = v[p, 0]
            v1 = v[p, 1]
            v2 = v[p, 2]
            vm0 = v0 + kick0
            vm1 = v1 + kick1
            vm2 = v2

            # pic.py の行ベクトル v_minus @ boris_rt と同じ積和順。
            vr0 = (vm0 * boris_rt[0, 0] + vm1 * boris_rt[1, 0])
            vr0 = vr0 + vm2 * boris_rt[2, 0]
            vr1 = (vm0 * boris_rt[0, 1] + vm1 * boris_rt[1, 1])
            vr1 = vr1 + vm2 * boris_rt[2, 1]
            vr2 = (vm0 * boris_rt[0, 2] + vm1 * boris_rt[1, 2])
            vr2 = vr2 + vm2 * boris_rt[2, 2]

            vn0 = vr0 + kick0
            vn1 = vr1 + kick1
            vn2 = vr2
            out_vnew[p, 0] = vn0
            out_vnew[p, 1] = vn1
            out_vnew[p, 2] = vn2
            out_xnew[p, 0] = x[p, 0] + dt_sp * vn0
            out_xnew[p, 1] = x[p, 1] + dt_sp * vn1
            out_vdot[p] = (v0 * vn0 + v1 * vn1) + v2 * vn2

    @njit(cache=True, nogil=True, parallel=True)
    def _gather_push_walk_boris_kernel(
        exy,
        packed,
        adjacency,
        elem0,
        half,
        dt_sp,
        boris_rt,
        x,
        v,
        tol,
        max_iters,
        out_vnew,
        out_xnew,
        out_vdot,
        out_elem,
        out_absorbed,
        out_b_elem,
        out_b_loc,
        out_l,
    ):
        """一様磁場のBoris pushとwalkを1粒子ループへ融合する。"""
        n = elem0.shape[0]
        for p in prange(n):
            e = elem0[p]
            kick0 = half * exy[e, 0]
            kick1 = half * exy[e, 1]
            v0 = v[p, 0]
            v1 = v[p, 1]
            v2 = v[p, 2]
            vm0 = v0 + kick0
            vm1 = v1 + kick1
            vm2 = v2

            vr0 = (vm0 * boris_rt[0, 0] + vm1 * boris_rt[1, 0])
            vr0 = vr0 + vm2 * boris_rt[2, 0]
            vr1 = (vm0 * boris_rt[0, 1] + vm1 * boris_rt[1, 1])
            vr1 = vr1 + vm2 * boris_rt[2, 1]
            vr2 = (vm0 * boris_rt[0, 2] + vm1 * boris_rt[1, 2])
            vr2 = vr2 + vm2 * boris_rt[2, 2]

            vn0 = vr0 + kick0
            vn1 = vr1 + kick1
            vn2 = vr2
            xp = x[p, 0] + dt_sp * vn0
            yp = x[p, 1] + dt_sp * vn1
            out_vnew[p, 0] = vn0
            out_vnew[p, 1] = vn1
            out_vnew[p, 2] = vn2
            out_xnew[p, 0] = xp
            out_xnew[p, 1] = yp
            out_vdot[p] = (v0 * vn0 + v1 * vn1) + v2 * vn2

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
                nb = adjacency[e, mi]
                if nb == -1:
                    absorbed = True
                    b_elem = e
                    b_loc = mi
                    break
                e = nb
            if not converged and not absorbed:
                # walk単体カーネルと同じ反復上限フォールバック。
                g9 = packed[e, 9]
                l0 = (packed[e, 0] + packed[e, 3] * xp) + packed[e, 6] * yp
                l0 = l0 / g9
                l1 = (packed[e, 1] + packed[e, 4] * xp) + packed[e, 7] * yp
                l1 = l1 / g9
                l2 = (packed[e, 2] + packed[e, 5] * xp) + packed[e, 8] * yp
                l2 = l2 / g9

            out_elem[p] = e
            out_absorbed[p] = absorbed
            if absorbed:
                out_b_elem[p] = b_elem
                out_b_loc[p] = b_loc
            else:
                out_l[p, 0] = l0
                out_l[p, 1] = l1
                out_l[p, 2] = l2

    @njit(cache=True, nogil=True, parallel=True)
    def _gather_push_walk_kernel(
        exy,
        packed,
        adjacency,
        elem0,
        qm_dt,
        dt_sp,
        x,
        v,
        tol,
        max_iters,
        out_vnew,
        out_xnew,
        out_vdot,
        out_elem,
        out_absorbed,
        out_b_elem,
        out_b_loc,
        out_l,
    ):
        """基本xy経路のpushとwalkを1粒子ループへ融合する。"""
        n = elem0.shape[0]
        for p in prange(n):
            e = elem0[p]
            ax = qm_dt * exy[e, 0]
            ay = qm_dt * exy[e, 1]
            v0 = v[p, 0]
            v1 = v[p, 1]
            v2 = v[p, 2]
            vn0 = v0 + ax
            vn1 = v1 + ay
            vn2 = v2
            xp = x[p, 0] + dt_sp * vn0
            yp = x[p, 1] + dt_sp * vn1

            out_vnew[p, 0] = vn0
            out_vnew[p, 1] = vn1
            out_vnew[p, 2] = vn2
            out_xnew[p, 0] = xp
            out_xnew[p, 1] = yp
            out_vdot[p] = (v0 * vn0 + v1 * vn1) + v2 * vn2

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
                nb = adjacency[e, mi]
                if nb == -1:
                    absorbed = True
                    b_elem = e
                    b_loc = mi
                    break
                e = nb
            if not converged and not absorbed:
                # walk単体カーネルと同じ反復上限フォールバック。
                g9 = packed[e, 9]
                l0 = (packed[e, 0] + packed[e, 3] * xp) + packed[e, 6] * yp
                l0 = l0 / g9
                l1 = (packed[e, 1] + packed[e, 4] * xp) + packed[e, 7] * yp
                l1 = l1 / g9
                l2 = (packed[e, 2] + packed[e, 5] * xp) + packed[e, 8] * yp
                l2 = l2 / g9

            out_elem[p] = e
            out_absorbed[p] = absorbed
            # 境界情報は absorbed=True の行だけで定義される。
            if absorbed:
                out_b_elem[p] = b_elem
                out_b_loc[p] = b_loc
            else:
                out_l[p, 0] = l0
                out_l[p, 1] = l1
                out_l[p, 2] = l2

    @njit(inline="always")
    def _gather_push_walk_rz_one(
        exy,
        packed,
        adjacency,
        elem0,
        qm,
        dt_sp,
        ridx,
        x,
        v,
        tol,
        max_iters,
        out_vnew,
        out_xnew,
        out_vdot,
        out_elem,
        out_absorbed,
        out_b_elem,
        out_b_loc,
        out_l,
        p,
    ):
        """軸対称push (回転法)・walkを1粒子分処理する。"""
        e = elem0[p]
        v0 = v[p, 0]
        v1 = v[p, 1]
        v2 = v[p, 2]
        x0 = x[p, 0]
        x1 = x[p, 1]

        # numpy経路と同じく q/m·E を作ってから dt を乗じる (E だけの蹴り)。
        a0 = qm * exy[e, 0]
        a1 = qm * exy[e, 1]
        vn0 = v0 + dt_sp * a0
        vn1 = v1 + dt_sp * a1

        # 時刻中心化KEは、回転より前の (蹴った直後の) 速度で評価する。
        out_vdot[p] = (v0 * vn0 + v1 * vn1) + v2 * v2
        xp = x0 + dt_sp * vn0
        yp = x1 + dt_sp * vn1

        # 回転法: 局所座標の3D直線移動 (径 xr、周 yt) の後の半径へ移し、(vr, vθ) を
        # 同じ角だけ回す。r ≥ 0 のままなので軸は吸収境界にならず、別の鏡映は要らない。
        if ridx == 0:
            xr = xp
            vr = vn0
        else:
            xr = yp
            vr = vn1
        yt = dt_sp * v2
        r_new = np.sqrt(xr * xr + yt * yt)
        cos_a = 1.0
        sin_a = 0.0
        if r_new > 0.0:
            cos_a = xr / r_new
            sin_a = yt / r_new
        vr_rot = cos_a * vr + sin_a * v2
        vn2 = -sin_a * vr + cos_a * v2
        if ridx == 0:
            xp = r_new
            vn0 = vr_rot
        else:
            yp = r_new
            vn1 = vr_rot

        out_vnew[p, 0] = vn0
        out_vnew[p, 1] = vn1
        out_vnew[p, 2] = vn2
        out_xnew[p, 0] = xp
        out_xnew[p, 1] = yp

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
            nb = adjacency[e, mi]
            if nb == -1:
                absorbed = True
                b_elem = e
                b_loc = mi
                break
            e = nb
        if not converged and not absorbed:
            # walk単体カーネルと同じ反復上限フォールバック。
            g9 = packed[e, 9]
            l0 = (packed[e, 0] + packed[e, 3] * xp) + packed[e, 6] * yp
            l0 = l0 / g9
            l1 = (packed[e, 1] + packed[e, 4] * xp) + packed[e, 7] * yp
            l1 = l1 / g9
            l2 = (packed[e, 2] + packed[e, 5] * xp) + packed[e, 8] * yp
            l2 = l2 / g9

        out_elem[p] = e
        out_absorbed[p] = absorbed
        # 境界情報は absorbed=True の行だけで定義される。通常粒子への
        # int64 2本の書き込みを省き、融合カーネルのメモリ帯域を抑える。
        if absorbed:
            out_b_elem[p] = b_elem
            out_b_loc[p] = b_loc
        else:
            out_l[p, 0] = l0
            out_l[p, 1] = l1
            out_l[p, 2] = l2

    @njit(cache=True, nogil=True, parallel=True)
    def _gather_push_walk_rz_x_kernel(
        exy, packed, adjacency, elem0, qm, dt_sp, x, v, tol,
        max_iters, out_vnew, out_xnew, out_vdot, out_elem, out_absorbed,
        out_b_elem, out_b_loc, out_l,
    ):
        """ridx=0 (rz_x0) を定数化した軸対称融合カーネル。"""
        for p in prange(elem0.shape[0]):
            _gather_push_walk_rz_one(
                exy, packed, adjacency, elem0, qm, dt_sp, 0, x, v,
                tol, max_iters, out_vnew, out_xnew, out_vdot, out_elem,
                out_absorbed, out_b_elem, out_b_loc, out_l, p,
            )

    @njit(cache=True, nogil=True, parallel=True)
    def _gather_push_walk_rz_y_kernel(
        exy, packed, adjacency, elem0, qm, dt_sp, x, v, tol,
        max_iters, out_vnew, out_xnew, out_vdot, out_elem, out_absorbed,
        out_b_elem, out_b_loc, out_l,
    ):
        """ridx=1 (rz) を定数化した軸対称融合カーネル。"""
        for p in prange(elem0.shape[0]):
            _gather_push_walk_rz_one(
                exy, packed, adjacency, elem0, qm, dt_sp, 1, x, v,
                tol, max_iters, out_vnew, out_xnew, out_vdot, out_elem,
                out_absorbed, out_b_elem, out_b_loc, out_l, p,
            )

    @njit(cache=True, nogil=True)
    def _interp_packed_table(x, xs, ys, n):
        """np.interp と同じ端点クランプ・右側探索で1点を線形補間する。

        prompts/83①でlog10(E)等間隔グリッド+O(1)参照を試したが、本リポジトリの
        断面積テーブル規模 (数十〜百点程度) では二分探索のほうが実測で速かった
        ため不採用・撤去し、この二分探索実装に戻した。
        """
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
            density_scale = rel[i] if use_rel else 1.0
            # 分母はセル別 ν_max,c = numax·density_scale (非一様ガス場、
            # prompts/83②)。一様ガスは density_scale=1.0 でno-op (ビット一致)。
            target = random_u[i] * numax * density_scale
            cumulative = 0.0
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

    @njit(inline="always")
    def _mcc_select_velocity_one(
        v,
        cand,
        energy_scale,
        charge_ev,
        numax,
        n_ref,
        rel_elem,
        elem,
        use_rel,
        e_table,
        s_table,
        lengths,
        random_u,
        out_proc,
        out_energy,
        out_speed,
        k,
    ):
        """候補1粒子の速度・エネルギー評価とプロセス選択を行う。"""
        n_procs = lengths.shape[0]
        i = cand[k]
        v0 = v[i, 0]
        v1 = v[i, 1]
        v2 = v[i, 2]
        speed2 = (v0 * v0 + v1 * v1) + v2 * v2
        speed = np.sqrt(speed2)
        e_ev = (energy_scale * speed * speed) / charge_ev
        out_speed[k] = speed
        out_energy[k] = e_ev

        density_scale = rel_elem[elem[i]] if use_rel else 1.0
        # 分母はセル別 ν_max,c = numax·density_scale (prompts/83②、_candidates
        # のセル別抽選と揃える。一様ガスは density_scale=1.0 でビット一致)。
        target = random_u[k] * numax * density_scale
        cumulative = 0.0
        selected = -1
        for j in range(n_procs):
            sigma = _interp_packed_table(
                e_ev, e_table[j], s_table[j], lengths[j]
            )
            cumulative += (n_ref * sigma * speed) * density_scale
            if target < cumulative:
                selected = j
                break
        out_proc[k] = selected

    @njit(cache=True, nogil=True)
    def _mcc_select_velocity_kernel(
        v,
        cand,
        energy_scale,
        charge_ev,
        numax,
        n_ref,
        rel_elem,
        elem,
        use_rel,
        e_table,
        s_table,
        lengths,
        random_u,
        out_proc,
        out_energy,
        out_speed,
    ):
        """小～中規模候補をスレッド起動なしの1ループで選択する。"""
        n = cand.shape[0]
        for k in range(n):
            _mcc_select_velocity_one(
                v,
                cand,
                energy_scale,
                charge_ev,
                numax,
                n_ref,
                rel_elem,
                elem,
                use_rel,
                e_table,
                s_table,
                lengths,
                random_u,
                out_proc,
                out_energy,
                out_speed,
                k,
            )

    @njit(cache=True, nogil=True, parallel=True)
    def _mcc_select_velocity_parallel_kernel(
        v,
        cand,
        energy_scale,
        charge_ev,
        numax,
        n_ref,
        rel_elem,
        elem,
        use_rel,
        e_table,
        s_table,
        lengths,
        random_u,
        out_proc,
        out_energy,
        out_speed,
    ):
        """大規模候補を粒子ごとに独立なprangeで選択する。"""
        for k in prange(cand.shape[0]):
            _mcc_select_velocity_one(
                v,
                cand,
                energy_scale,
                charge_ev,
                numax,
                n_ref,
                rel_elem,
                elem,
                use_rel,
                e_table,
                s_table,
                lengths,
                random_u,
                out_proc,
                out_energy,
                out_speed,
                k,
            )

    @njit(cache=True, nogil=True)
    def _mcc_group_process_positions_kernel(
        proc_idx, energy, thresholds, use_threshold,
    ):
        """実衝突候補位置をプロセス別・元順序の連続領域へまとめる。"""
        n_procs = thresholds.shape[0]
        counts = np.zeros(n_procs, dtype=np.int64)
        for k in range(proc_idx.shape[0]):
            j = proc_idx[k]
            if j < 0:
                continue
            if use_threshold[j] and energy[k] < thresholds[j]:
                continue
            counts[j] += 1

        offsets = np.empty(n_procs + 1, dtype=np.int64)
        offsets[0] = 0
        for j in range(n_procs):
            offsets[j + 1] = offsets[j] + counts[j]
        positions = np.empty(offsets[n_procs], dtype=np.int64)
        cursor = offsets[:-1].copy()
        for k in range(proc_idx.shape[0]):
            j = proc_idx[k]
            if j < 0:
                continue
            if use_threshold[j] and energy[k] < thresholds[j]:
                continue
            positions[cursor[j]] = k
            cursor[j] += 1
        return positions, offsets

    @njit(cache=True, nogil=True)
    def _mcc_scatter_electrons_kernel(
        v,
        cand,
        selected,
        e_ev,
        speed,
        direction,
        kind,
        threshold_ev,
        mass_ratio,
        scatter_energy,
        charge_ev,
        electron_mass,
    ):
        """選択済み電子衝突の速度更新をin-placeで行う。"""
        n = selected.shape[0]
        for k in range(n):
            pos = selected[k]
            i = cand[pos]
            d0 = direction[k, 0]
            d1 = direction[k, 1]
            d2 = direction[k, 2]
            if kind == 0:  # elastic
                cos_chi = (
                    (v[i, 0] / speed[pos]) * d0
                    + (v[i, 1] / speed[pos]) * d1
                ) + (v[i, 2] / speed[pos]) * d2
                e_new = e_ev[pos] * (
                    1.0 - 2.0 * mass_ratio * (1.0 - cos_chi)
                )
                if e_new < 0.0:
                    e_new = 0.0
            elif kind == 1:  # excitation
                e_new = e_ev[pos] - threshold_ev
            else:  # ionization
                e_new = scatter_energy[k]
            speed_new = np.sqrt(2.0 * e_new * charge_ev / electron_mass)
            v[i, 0] = speed_new * d0
            v[i, 1] = speed_new * d1
            v[i, 2] = speed_new * d2

    @njit(cache=True, nogil=True)
    def _mcc_scatter_ions_kernel(
        v, cand, selected, vi, vg, g_mag, direction, backscat
    ):
        """選択済みイオン衝突の速度更新をin-placeで行う。"""
        n = selected.shape[0]
        for k in range(n):
            pos = selected[k]
            i = cand[pos]
            if backscat:
                v[i, 0] = vg[pos, 0]
                v[i, 1] = vg[pos, 1]
                v[i, 2] = vg[pos, 2]
            else:
                half_g = 0.5 * g_mag[pos]
                v[i, 0] = (
                    0.5 * (vi[pos, 0] + vg[pos, 0])
                    + half_g * direction[k, 0]
                )
                v[i, 1] = (
                    0.5 * (vi[pos, 1] + vg[pos, 1])
                    + half_g * direction[k, 1]
                )
                v[i, 2] = (
                    0.5 * (vi[pos, 2] + vg[pos, 2])
                    + half_g * direction[k, 2]
                )

    @njit(cache=True, nogil=True)
    def _mcc_iso_dir_kernel(random_cos, random_phi, out):
        """NumPy版と同じ演算順で3D等方方向を構築する。"""
        n = random_cos.shape[0]
        for i in range(n):
            cos_t = 1.0 - 2.0 * random_cos[i]
            sin2 = 1.0 - cos_t * cos_t
            if sin2 < 0.0:
                sin2 = 0.0
            sin_t = np.sqrt(sin2)
            phi = random_phi[i] * (2.0 * np.pi)
            out[i, 0] = sin_t * np.cos(phi)
            out[i, 1] = sin_t * np.sin(phi)
            out[i, 2] = cos_t

    @njit(cache=True, nogil=True)
    def _mcc_candidates_kernel(random_u, probability):
        """候補マスクを作らず、採択インデックスを連続配列へ詰める。"""
        count = 0
        for i in range(random_u.shape[0]):
            if random_u[i] < probability:
                count += 1
        out = np.empty(count, dtype=np.int64)
        pos = 0
        for i in range(random_u.shape[0]):
            if random_u[i] < probability:
                out[pos] = i
                pos += 1
        return out

    @njit(cache=True, nogil=True)
    def _mcc_candidates_cellwise_kernel(random_u, p_coll_elem, elem):
        """要素別候補確率 (prompts/83②、非一様ガス場) で候補を選ぶ。

        p_coll_elem は呼び出し側で1回だけ計算済み (numpyのexpをここで
        再評価するとnumpy/numba等価性が崩れうるため、gather+比較のみ行う)。
        """
        n = random_u.shape[0]
        count = 0
        for i in range(n):
            if random_u[i] < p_coll_elem[elem[i]]:
                count += 1
        out = np.empty(count, dtype=np.int64)
        pos = 0
        for i in range(n):
            if random_u[i] < p_coll_elem[elem[i]]:
                out[pos] = i
                pos += 1
        return out

    @njit(cache=True, nogil=True)
    def _mcc_max_speed_squared_kernel(v):
        """3速度成分の二乗和の最大値を一時配列なしで求める。"""
        maximum = 0.0
        for i in range(v.shape[0]):
            speed2 = (
                v[i, 0] * v[i, 0] + v[i, 1] * v[i, 1]
            ) + v[i, 2] * v[i, 2]
            # np.maxと同じく、入力にNaNがあればNaNを返す。
            if np.isnan(speed2):
                return speed2
            if speed2 > maximum:
                maximum = speed2
        return maximum

    @njit(cache=True, nogil=True)
    def _cell_sort_order_kernel(elem, n_cells):
        """粒子のセル (elem) 順への安定な置換インデックスを計数ソートで求める
        (高速化③、prompts/84)。

        elem の値域が [0, n_cells) と既知なので、比較ソート (np.argsort、
        O(N log N)) ではなく計数ソート (O(N + n_cells)) で求められる。2パス構成:
          1. 各セルの粒子数を数え、累積和でセルごとの書き込み開始位置を得る
          2. 元の順序で走査しながら開始位置へ書き込み、書き込み位置を1つ進める
             (同じセル内の粒子は元の相対順序のまま = 安定ソートと同じ結果)
        大粒子数 (N ≫ n_cells) で argsort より大幅に速く、ソートを50ステップ
        ごとに償却してもコストをほぼ無視できる水準にできる。
        """
        n = elem.shape[0]
        counts = np.zeros(n_cells, dtype=np.int64)
        for i in range(n):
            counts[elem[i]] += 1
        pos = np.empty(n_cells, dtype=np.int64)
        acc = 0
        for c in range(n_cells):
            pos[c] = acc
            acc += counts[c]
        order = np.empty(n, dtype=np.int64)
        for i in range(n):
            e = elem[i]
            order[pos[e]] = i
            pos[e] += 1
        return order

    @njit(cache=True, nogil=True, parallel=True)
    def _dsmc_collide_kernel(
        v, order, starts, cand_starts, valid_starts,
        r1, r2, accept_rand, sig_coef, sig_pow,
        sigcr_max, coll_frac,
        i1_out, i2_out, gmag_out, sigcr_out, keep_out, first,
    ):
        """DSMC NTC 衝突のセル並列本体 (dsmc_collide のdocstring参照、prompts/87)。

        セル c ごとに:
          1. 候補 (r1[j], r2[j]) を走査し、自己対 (r1==r2) を捨てながら相対速度
             g_mag・断面積×g_mag (sig_cr) を計算して i1_out/gmag_out 等 (このセルの
             valid_starts[c]..valid_starts[c+1] 区間) に書き、このセル内の最大値
             local_max も同時に求める (dsmc.py 旧実装の np.maximum.at 相当)。
          2. sigcr_max[c] をこのセルのバッチ内最大値で更新してから (旧実装が
             全候補の maximum.at を accept 判定より先に行うのと同じ順序)、
             accept_rand との比較で採択候補を集める。
          3. 採択候補のうち同一粒子が複数回現れた場合は「最初に現れた対」だけを
             残す (旧実装の np.minimum.at による dedup と同じ)。first[] は
             粒子ごとの「このセル内での最初の採択位置」を持つ配列で、通し番号は
             このセル内だけの 0 始まりでよい (ある粒子の候補は自セルにしか
             現れないため、定数オフセット差を除いて旧実装の全体通し番号と
             大小関係が一致する)。
        """
        n_tris = cand_starts.shape[0] - 1
        for c in prange(n_tris):
            n_cand_c = cand_starts[c + 1] - cand_starts[c]
            if n_cand_c == 0:
                continue
            cs = starts[c]
            cbase = cand_starts[c]
            vbase = valid_starts[c]
            vj = 0
            local_max = 0.0
            for j in range(n_cand_c):
                idx = cbase + j
                r1j = r1[idx]
                r2j = r2[idx]
                if r1j == r2j:
                    continue
                i1g = order[cs + r1j]
                i2g = order[cs + r2j]
                gx = v[i1g, 0] - v[i2g, 0]
                gy = v[i1g, 1] - v[i2g, 1]
                gz = v[i1g, 2] - v[i2g, 2]
                g2 = (gx * gx + gy * gy) + gz * gz
                g_mag = np.sqrt(g2)
                # dsmc.py._sigma と同じ (c_r>0 のみ非0、係数×べき乗×g_mag の順)
                if g_mag > 0.0:
                    sig_cr = (sig_coef * g_mag ** sig_pow) * g_mag
                else:
                    sig_cr = 0.0
                p = vbase + vj
                i1_out[p] = i1g
                i2_out[p] = i2g
                gmag_out[p] = g_mag
                sigcr_out[p] = sig_cr
                if sig_cr > local_max:
                    local_max = sig_cr
                vj += 1

            new_max = sigcr_max[c]
            if local_max > new_max:
                new_max = local_max
            sigcr_max[c] = new_max
            if vj == 0:
                continue

            # このセルの採択候補だけを詰めるローカル作業配列 (最大 vj 件、
            # numba の prange 内 np.empty はスレッドごとに独立なバッファになる)
            acc_i1 = np.empty(vj, dtype=np.int64)
            acc_i2 = np.empty(vj, dtype=np.int64)
            acc_p = np.empty(vj, dtype=np.int64)
            acc_n = 0
            for jj in range(vj):
                p = vbase + jj
                if accept_rand[p] * new_max >= sigcr_out[p]:
                    continue
                i1a = i1_out[p]
                i2a = i2_out[p]
                acc_i1[acc_n] = i1a
                acc_i2[acc_n] = i2a
                acc_p[acc_n] = p
                if acc_n < first[i1a]:
                    first[i1a] = acc_n
                if acc_n < first[i2a]:
                    first[i2a] = acc_n
                acc_n += 1

            for kk in range(acc_n):
                i1a = acc_i1[kk]
                i2a = acc_i2[kk]
                if first[i1a] == kk and first[i2a] == kk:
                    keep_out[acc_p[kk]] = True
                else:
                    # 複数対に選ばれて落ちた分はこのセルの次ステップ候補数へ持ち越す
                    coll_frac[c] += 1.0

    @njit(cache=True, nogil=True, parallel=True)
    def _csr_matvec_kernel(indptr, indices, data, x, out):
        """CSR y = A@x の行並列 matvec (陰的流体ソルバーの反復法、prompts/115)。

        行 i の内積 Σ_k data[k]・x[indices[k]] (k は indptr[i]..indptr[i+1) の
        範囲) は、その行を担当する1スレッドが常に昇順 k で逐次積和する。
        行 i の計算は他の行の値を一切読まず、書き込み先 out[i] も行ごとに
        排他 (他スレッドと衝突しない) なので、「どの行をどのスレッドが
        いつ処理するか」というスケジューリングの違いは結果に影響しない —
        各行の浮動小数点加算の順序そのものがスレッド数に依らず不変なので、
        threads=1 と threads=2 で out 全体がビット単位で一致する
        (_walk_kernel と同じ「行/粒子ごとに独立」という決定論の根拠)。
        """
        n = indptr.shape[0] - 1
        for i in prange(n):
            s = indptr[i]
            e = indptr[i + 1]
            acc = 0.0
            for k in range(s, e):
                acc += data[k] * x[indices[k]]
            out[i] = acc


def cell_sort_order(elem: np.ndarray, n_cells: int) -> np.ndarray:
    """粒子のセル順ソート (prompts/84) 用の安定な置換インデックスを返す。

    numba があれば計数ソート (O(N + n_cells))、無ければ np.argsort(kind="stable")
    (O(N log N)) にフォールバックする。どちらも同一セル内の相対順序を保つ
    安定ソートであり、返す置換 (同一値どうしの相対順) は完全に一致する。
    """
    if HAVE_NUMBA:
        return _cell_sort_order_kernel(elem, n_cells)
    return np.argsort(elem, kind="stable")


def walk_step(coeffs, adjacency, elem0, x_new, l_out=None, packed=None):
    """particles._walk_step_numpy の numba 版。戻り値・意味は完全に同じ。

    l_out の absorbed 粒子の行は numpy 版と同じく未定義。連続float64配列は
    中間バッファと非吸収行の再コピーを避けるため直接書き込む。その他の配列は
    従来どおり一時バッファから非吸収行だけをコピーする。
    """
    if packed is None:
        from .particles import _pack_coeffs

        packed = _pack_coeffs(coeffs)
    n = len(x_new)
    elem = np.empty(n, dtype=np.int64)
    absorbed = np.empty(n, dtype=np.bool_)
    b_elem = np.empty(n, dtype=np.int64)
    b_loc = np.empty(n, dtype=np.int64)
    direct_l_out = (
        l_out is not None
        and l_out.shape == (n, 3)
        and l_out.dtype == np.float64
        and l_out.flags.c_contiguous
        and l_out.flags.writeable
        and not np.shares_memory(l_out, x_new)
    )
    l_buf = l_out if direct_l_out else np.empty((n, 3), dtype=np.float64)
    _walk_kernel(
        packed,
        adjacency,
        elem0,
        np.ascontiguousarray(x_new),
        _TOL,
        _MAX_WALK_ITERS,
        elem,
        absorbed,
        b_elem,
        b_loc,
        l_buf,
    )
    if l_out is not None and not direct_l_out:
        keep = ~absorbed
        l_out[keep] = l_buf[keep]
    return elem, absorbed, b_elem, b_loc


def deposit(nidx: np.ndarray, bary: np.ndarray, q: float, w: np.ndarray, n_nodes: int) -> np.ndarray:
    """電荷堆積 (P1 重み散布) の numba 版。pic._deposit_species の bincount 経路と等価。"""
    return _deposit_kernel(nidx, bary, float(q), w, int(n_nodes))


def deposit_from_elements(
    tris: np.ndarray,
    elem: np.ndarray,
    bary: np.ndarray,
    q: float,
    w: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """所属要素から直接P1電荷を散布し、tris[elem]の一時配列を作らない。"""
    return _deposit_elements_kernel(
        np.ascontiguousarray(tris),
        np.ascontiguousarray(elem),
        np.ascontiguousarray(bary),
        float(q),
        np.ascontiguousarray(w),
        int(n_nodes),
    )


def compact_particle_state(
    x: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    elem: np.ndarray,
    bary: np.ndarray,
    removed: np.ndarray,
    out_w: np.ndarray,
) -> int:
    """非removed粒子を安定順序で先頭へ詰め、残存粒子数を返す。

    x/v/elem/bary はpush+walk出力バッファ内でin-place圧縮する。wは境界処理が
    更新前状態を参照し終えるまで変更できないため、KE計算後のvdotバッファを
    out_wとして再利用する。
    """
    n = len(removed)
    if any(len(buf) < n for buf in (x, v, w, elem, bary, out_w)):
        raise ValueError("compact_particle_state のバッファ容量が不足しています")
    return int(
        _compact_particle_state_kernel(
            x,
            v,
            w,
            elem,
            bary,
            removed,
            out_w,
        )
    )


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


def gather_push_boris(
    exy: np.ndarray,
    elem: np.ndarray,
    q: float,
    m: float,
    dt_sp: float,
    boris_rt: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
):
    """一様磁場xy経路のgather・Boris push・driftを融合する。"""
    n = len(x)
    v_new = np.empty((n, 3), dtype=np.float64)
    x_new = np.empty((n, 2), dtype=np.float64)
    vdot = np.empty(n, dtype=np.float64)
    half = (q / m) * (0.5 * dt_sp)
    _gather_push_boris_kernel(
        np.ascontiguousarray(exy),
        np.ascontiguousarray(elem),
        half,
        dt_sp,
        np.ascontiguousarray(boris_rt),
        np.ascontiguousarray(x),
        np.ascontiguousarray(v),
        v_new,
        x_new,
        vdot,
    )
    return v_new, x_new, vdot


def gather_push_walk_boris(
    exy: np.ndarray,
    packed: np.ndarray,
    adjacency: np.ndarray,
    elem: np.ndarray,
    q: float,
    m: float,
    dt_sp: float,
    boris_rt: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    out: tuple[np.ndarray, ...] | None = None,
):
    """一様磁場xy経路のBoris pushとwalkを1回で処理する。"""
    n = len(x)
    if out is None:
        v_new = np.empty((n, 3), dtype=np.float64)
        x_new = np.empty((n, 2), dtype=np.float64)
        vdot = np.empty(n, dtype=np.float64)
        elem_new = np.empty(n, dtype=np.int64)
        absorbed = np.empty(n, dtype=np.bool_)
        b_elem = np.empty(n, dtype=np.int64)
        b_loc = np.empty(n, dtype=np.int64)
        bary = np.empty((n, 3), dtype=np.float64)
    else:
        if len(out) != 8 or any(len(buf) < n for buf in out):
            raise ValueError(
                "gather_push_walk_boris の出力バッファ容量が不足しています"
            )
        v_new = out[0][:n]
        x_new = out[1][:n]
        vdot = out[2][:n]
        elem_new = out[3][:n]
        absorbed = out[4][:n]
        b_elem = out[5][:n]
        b_loc = out[6][:n]
        bary = out[7][:n]
    half = (q / m) * (0.5 * dt_sp)
    _gather_push_walk_boris_kernel(
        np.ascontiguousarray(exy),
        np.ascontiguousarray(packed),
        np.ascontiguousarray(adjacency),
        np.ascontiguousarray(elem),
        half,
        dt_sp,
        np.ascontiguousarray(boris_rt),
        np.ascontiguousarray(x),
        np.ascontiguousarray(v),
        _TOL,
        _MAX_WALK_ITERS,
        v_new,
        x_new,
        vdot,
        elem_new,
        absorbed,
        b_elem,
        b_loc,
        bary,
    )
    return v_new, x_new, vdot, elem_new, absorbed, b_elem, b_loc, bary


def gather_push_walk(
    exy: np.ndarray,
    packed: np.ndarray,
    adjacency: np.ndarray,
    elem: np.ndarray,
    q: float,
    m: float,
    dt_sp: float,
    x: np.ndarray,
    v: np.ndarray,
    out: tuple[np.ndarray, ...] | None = None,
):
    """基本xy経路のgather+push+walkを1回のNumba呼び出しで処理する。

    中間のx_newを別カーネルへ渡す処理とwalk用の重複バッファをなくす。返却値は
    gather_push と walk_step の結果を連結した
    (v_new, x_new, vdot, elem, absorbed, b_elem, b_loc, bary)。境界要素番号と
    局所辺番号は absorbed=True の行だけで定義される。
    """
    n = len(x)
    if out is None:
        v_new = np.empty((n, 3), dtype=np.float64)
        x_new = np.empty((n, 2), dtype=np.float64)
        vdot = np.empty(n, dtype=np.float64)
        elem_new = np.empty(n, dtype=np.int64)
        absorbed = np.empty(n, dtype=np.bool_)
        b_elem = np.empty(n, dtype=np.int64)
        b_loc = np.empty(n, dtype=np.int64)
        bary = np.empty((n, 3), dtype=np.float64)
    else:
        if len(out) != 8 or any(len(buf) < n for buf in out):
            raise ValueError("gather_push_walk の出力バッファ容量が不足しています")
        v_new = out[0][:n]
        x_new = out[1][:n]
        vdot = out[2][:n]
        elem_new = out[3][:n]
        absorbed = out[4][:n]
        b_elem = out[5][:n]
        b_loc = out[6][:n]
        bary = out[7][:n]
    _gather_push_walk_kernel(
        np.ascontiguousarray(exy),
        np.ascontiguousarray(packed),
        np.ascontiguousarray(adjacency),
        np.ascontiguousarray(elem),
        (q / m) * dt_sp,
        dt_sp,
        np.ascontiguousarray(x),
        np.ascontiguousarray(v),
        _TOL,
        _MAX_WALK_ITERS,
        v_new,
        x_new,
        vdot,
        elem_new,
        absorbed,
        b_elem,
        b_loc,
        bary,
    )
    return v_new, x_new, vdot, elem_new, absorbed, b_elem, b_loc, bary


def gather_push_walk_rz(
    exy: np.ndarray,
    packed: np.ndarray,
    adjacency: np.ndarray,
    elem: np.ndarray,
    q: float,
    m: float,
    dt_sp: float,
    ridx: int,
    x: np.ndarray,
    v: np.ndarray,
    out: tuple[np.ndarray, ...] | None = None,
):
    """軸対称gather+push (回転法)+walkを1回で処理する。

    push は E だけの蹴り → 局所座標の3D直線移動 → 子午面への回転 (pic.py の numpy 経路と
    同じ演算順)。ridx=1 は rz (x=z, y=r)、ridx=0 は rz_x0 (x=r, y=z)。返却値と
    再利用バッファの契約は gather_push_walk と同じ。ただし境界要素番号と
    局所辺番号は absorbed=True の行だけで定義される。
    """
    if ridx not in (0, 1):
        raise ValueError("gather_push_walk_rz の ridx は 0 または 1 が必要です")
    n = len(x)
    if out is None:
        v_new = np.empty((n, 3), dtype=np.float64)
        x_new = np.empty((n, 2), dtype=np.float64)
        vdot = np.empty(n, dtype=np.float64)
        elem_new = np.empty(n, dtype=np.int64)
        absorbed = np.empty(n, dtype=np.bool_)
        b_elem = np.empty(n, dtype=np.int64)
        b_loc = np.empty(n, dtype=np.int64)
        bary = np.empty((n, 3), dtype=np.float64)
    else:
        if len(out) != 8 or any(len(buf) < n for buf in out):
            raise ValueError("gather_push_walk_rz の出力バッファ容量が不足しています")
        v_new = out[0][:n]
        x_new = out[1][:n]
        vdot = out[2][:n]
        elem_new = out[3][:n]
        absorbed = out[4][:n]
        b_elem = out[5][:n]
        b_loc = out[6][:n]
        bary = out[7][:n]
    kernel = (
        _gather_push_walk_rz_x_kernel
        if ridx == 0
        else _gather_push_walk_rz_y_kernel
    )
    kernel(
        np.ascontiguousarray(exy),
        np.ascontiguousarray(packed),
        np.ascontiguousarray(adjacency),
        np.ascontiguousarray(elem),
        q / m,
        dt_sp,
        np.ascontiguousarray(x),
        np.ascontiguousarray(v),
        _TOL,
        _MAX_WALK_ITERS,
        v_new,
        x_new,
        vdot,
        elem_new,
        absorbed,
        b_elem,
        b_loc,
        bary,
    )
    return v_new, x_new, vdot, elem_new, absorbed, b_elem, b_loc, bary


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


def mcc_select_velocity(
    v: np.ndarray,
    cand: np.ndarray,
    m: float,
    charge_ev: float,
    numax: float,
    n_ref: float,
    rel_elem: np.ndarray | None,
    elem: np.ndarray | None,
    e_table: np.ndarray,
    s_table: np.ndarray,
    lengths: np.ndarray,
    random_u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """候補粒子の速度からプロセス・エネルギー・速さを一括算出する。"""
    n = len(cand)
    proc = np.empty(n, dtype=np.int64)
    energy = np.empty(n, dtype=np.float64)
    speed = np.empty(n, dtype=np.float64)
    use_rel = rel_elem is not None and elem is not None
    rel_arr = np.empty(0, dtype=np.float64) if rel_elem is None else rel_elem
    elem_arr = np.empty(0, dtype=np.int64) if elem is None else elem
    kernel = (
        _mcc_select_velocity_parallel_kernel
        if n >= _MCC_PARALLEL_MIN_CANDIDATES and numba.get_num_threads() > 1
        else _mcc_select_velocity_kernel
    )
    kernel(
        np.ascontiguousarray(v),
        np.ascontiguousarray(cand),
        0.5 * float(m),
        float(charge_ev),
        float(numax),
        float(n_ref),
        np.ascontiguousarray(rel_arr),
        np.ascontiguousarray(elem_arr),
        use_rel,
        np.ascontiguousarray(e_table),
        np.ascontiguousarray(s_table),
        np.ascontiguousarray(lengths),
        np.ascontiguousarray(random_u),
        proc,
        energy,
        speed,
    )
    return proc, energy, speed


def mcc_candidates_cellwise(
    random_u: np.ndarray, p_coll_elem: np.ndarray, elem: np.ndarray
) -> np.ndarray:
    """非一様ガス場でのnull-collision候補インデックスを返す (prompts/83②)。

    p_coll_elem は要素ごとの候補確率 (呼び出し側でnumpyにより1回だけ計算済み)。
    ここでは候補マスクを作らず、gather + 比較だけを行う。
    """
    return _mcc_candidates_cellwise_kernel(
        np.ascontiguousarray(random_u),
        np.ascontiguousarray(p_coll_elem),
        np.ascontiguousarray(elem),
    )


def mcc_scatter_electrons(
    v: np.ndarray,
    cand: np.ndarray,
    selected: np.ndarray,
    e_ev: np.ndarray,
    speed: np.ndarray,
    direction: np.ndarray,
    kind: int,
    threshold_ev: float,
    mass_ratio: float,
    scatter_energy: np.ndarray | None,
    charge_ev: float,
    electron_mass: float,
) -> None:
    """選択・乱数生成後の電子散乱計算をJITでin-place更新する。"""
    scatter_arr = (
        np.empty(0, dtype=np.float64)
        if scatter_energy is None
        else np.ascontiguousarray(scatter_energy)
    )
    _mcc_scatter_electrons_kernel(
        v,
        np.ascontiguousarray(cand),
        np.ascontiguousarray(selected),
        np.ascontiguousarray(e_ev),
        np.ascontiguousarray(speed),
        np.ascontiguousarray(direction),
        int(kind),
        float(threshold_ev),
        float(mass_ratio),
        scatter_arr,
        float(charge_ev),
        float(electron_mass),
    )


def mcc_scatter_ions(
    v: np.ndarray,
    cand: np.ndarray,
    selected: np.ndarray,
    vi: np.ndarray,
    vg: np.ndarray,
    g_mag: np.ndarray,
    direction: np.ndarray | None,
    backscat: bool,
) -> None:
    """選択・乱数生成後のイオン散乱計算をJITでin-place更新する。"""
    direction_arr = (
        np.empty((0, 3), dtype=np.float64)
        if direction is None
        else np.ascontiguousarray(direction)
    )
    _mcc_scatter_ions_kernel(
        v,
        np.ascontiguousarray(cand),
        np.ascontiguousarray(selected),
        np.ascontiguousarray(vi),
        np.ascontiguousarray(vg),
        np.ascontiguousarray(g_mag),
        direction_arr,
        bool(backscat),
    )


def mcc_iso_dir(random_cos: np.ndarray, random_phi: np.ndarray) -> np.ndarray:
    """一様乱数2列から3D等方単位方向を構築する。"""
    out = np.empty((len(random_cos), 3), dtype=np.float64)
    _mcc_iso_dir_kernel(
        np.ascontiguousarray(random_cos),
        np.ascontiguousarray(random_phi),
        out,
    )
    return out


def mcc_group_process_positions(
    proc_idx: np.ndarray,
    energy: np.ndarray,
    thresholds: np.ndarray,
    use_threshold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """実衝突候補の位置をプロセス別に安定順序でまとめる。"""
    return _mcc_group_process_positions_kernel(
        np.ascontiguousarray(proc_idx),
        np.ascontiguousarray(energy),
        np.ascontiguousarray(thresholds),
        np.ascontiguousarray(use_threshold),
    )


def mcc_candidates(random_u: np.ndarray, probability: float) -> np.ndarray:
    """null-collision候補インデックスを一時bool配列なしで返す。"""
    return _mcc_candidates_kernel(
        np.ascontiguousarray(random_u),
        float(probability),
    )


def mcc_max_speed_squared(v: np.ndarray) -> float:
    """粒子群の max(vx²+vy²+vz²) を一時配列なしで返す。"""
    return float(_mcc_max_speed_squared_kernel(np.ascontiguousarray(v)))


def dsmc_collide(
    v: np.ndarray,
    order: np.ndarray,
    starts: np.ndarray,
    cand_starts: np.ndarray,
    valid_starts: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    accept_rand: np.ndarray,
    sig_coef: float,
    sig_pow: float,
    sigcr_max: np.ndarray,
    coll_frac: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """DSMC NTC 衝突の候補選定・(σc_r)_max 更新・採択・dedup をセル並列で行う (prompts/87)。

    dsmc.py._collide_numba から呼ぶ。乱数 (r1・r2・accept_rand) は呼び出し側が
    旧実装 (dsmc.py._collide_numpy) と同じ順・同じ本数だけ numpy Generator で
    事前生成済み (このカーネルは乱数を一切生成しない)。sigcr_max・coll_frac は
    このセルの分だけをその場で更新する in-place 引数 (呼び出し側の配列そのもの
    が書き換わる)。

    戻り値 (i1, i2, keep, g_mag) はいずれも「有効候補」(r1≠r2 を満たすもの) の
    数 (= valid_starts[-1]) だけの長さを持つ。呼び出し側は keep でマスクした
    (i1, i2, g_mag) から散乱後速度を計算する (この最終段は候補数が少なく
    重くないため、旧実装のまま numpy で行う)。

    セル間の書き込み非衝突性: order/starts によりセルごとの粒子は互いに素な
    添字集合を持つため、i1/i2 (常に自セルの粒子を指す) は他セルの処理と
    絶対に重ならない。sigcr_max[c]・coll_frac[c] もセル c 自身の要素しか
    更新しないため、prange によるセル並列でも書き込み競合は起きない。
    """
    n_particles = len(v)
    total_valid = int(valid_starts[-1])
    i1_out = np.empty(total_valid, dtype=np.int64)
    i2_out = np.empty(total_valid, dtype=np.int64)
    gmag_out = np.empty(total_valid, dtype=np.float64)
    sigcr_out = np.empty(total_valid, dtype=np.float64)
    keep_out = np.zeros(total_valid, dtype=np.bool_)
    # dedup の「未採択」sentinel: 任意セルの採択数はどう転んでも total_valid を
    # 超えないので、total_valid+1 なら全セル共通で安全に「まだ採択なし」を表せる
    first = np.full(n_particles, total_valid + 1, dtype=np.int64)
    _dsmc_collide_kernel(
        np.ascontiguousarray(v),
        np.ascontiguousarray(order),
        np.ascontiguousarray(starts),
        np.ascontiguousarray(cand_starts),
        np.ascontiguousarray(valid_starts),
        np.ascontiguousarray(r1),
        np.ascontiguousarray(r2),
        np.ascontiguousarray(accept_rand),
        float(sig_coef),
        float(sig_pow),
        sigcr_max,
        coll_frac,
        i1_out,
        i2_out,
        gmag_out,
        sigcr_out,
        keep_out,
        first,
    )
    return i1_out, i2_out, keep_out, gmag_out


# ---- 陰的流体ソルバー用の並列反復法 (Jacobi-BiCGSTAB、prompts/115) ------------------
#
# fluid2d.py の毎ステップ spsolve (SuperLU の都度分解、逐次) をここに置き換える。
# SuperLU の直接分解自体は並列化できないが、反復法の主要コストである matvec は
# 行並列で完全にスケールし、かつ上記 _csr_matvec_kernel の設計によりビット決定論も
# 保てる (SuperLU 分解のような複雑な依存関係が無いため)。


def csr_matvec_parallel(indptr, indices, data, x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """CSR 疎行列ベクトル積 y = A@x (bicgstab から呼ばれる主要コスト)。

    HAVE_NUMBA=True: 上の _csr_matvec_kernel (行並列 prange、スレッド数に依らず
    ビット同一)。HAVE_NUMBA=False: scipy の csr_matrix.dot は単一スレッドの
    逐次実装 (BLAS 的な内部並列化はしない) なので、そのまま使えば決定論を
    保ったまま numpy 実装として使える (モジュール docstring の設計方針どおり)。
    """
    n = indptr.shape[0] - 1
    if out is None:
        out = np.empty(n, dtype=np.float64)
    if HAVE_NUMBA:
        _csr_matvec_kernel(indptr, indices, data, x, out)
    else:
        m = sp.csr_matrix((data, indices, indptr), shape=(n, len(x)))
        out[:] = m.dot(x)
    return out


def bicgstab(
    indptr,
    indices,
    data,
    b: np.ndarray,
    x0: np.ndarray,
    rtol: float = 1e-10,
    atol: float = 1e-300,
    max_iter: int = 200,
    diag_inv: np.ndarray | None = None,
) -> tuple[np.ndarray, int, bool]:
    """Jacobi (対角) 前処理付き BiCGSTAB で A x = b を解く (prompts/115)。

    なぜ Jacobi 前処理で足りるか: fluid2d.py の陰的行列 M = V_i/dt + K
    (時間項 + EAFE 剛性行列) は対角優位 — 時間項 V_i/dt が対角にしか乗らず、
    かつ dt が誘電緩和スケール (τ_d) で小さく選ばれるほど対角の相対的な
    優位性が強まる。対角優位な系では対角逆数によるスケーリングだけで
    有効条件数が大きく改善し、フル ILU 分解のような追加コストなしで
    数〜十数反復まで収束数を落とせる。

    なぜ内積・ノルムを numpy (単一スレッド) に残すか: BiCGSTAB のスカラー
    係数 (rho・alpha・omega 等) は「ベクトル全体を1つの数へ畳み込む」
    真のリダクションであり、これを並列化すると加算順序がスレッド数・
    スケジューリングに依存してしまう (浮動小数点の加算は結合則を満たさない
    ため、順序が変われば最終ビットも変わり得る)。一方 csr_matvec_parallel は
    「行ごとに独立」という構造上ビット決定論を保てるので、支配的コストである
    matvec だけを並列化し、内積は逐次 (numpy の pairwise 和、常に同じ
    アルゴリズム) のまま残すことで、性能と threads=1/2 のビット一致を両立する。

    x0 は呼び出し側 (fluid2d.py) が前ステップの密度/エネルギー値を渡す設計。
    対角優位な系は前ステップからの変化が小さく、ゼロ初期化よりずっと
    反復数を減らせる (前ステップ値を初期推定に使う定石)。

    収束判定: ||r|| ≤ max(rtol・||b||, atol)。diag_inv=None は前処理なし
    (単位行列、テスト用) として扱う。

    戻り値: (x, n_iter, converged)。収束しなかった場合、呼び出し側
    (fluid2d.py) が spsolve へフォールバックする設計 (堅牢性優先)。
    """
    n = len(b)
    x = np.array(x0, dtype=np.float64, copy=True)
    if diag_inv is None:
        diag_inv = np.ones(n, dtype=np.float64)
    bnorm = float(np.linalg.norm(b))
    tol = max(rtol * bnorm, atol)

    r = b - csr_matvec_parallel(indptr, indices, data, x)
    if float(np.linalg.norm(r)) <= tol:
        return x, 0, True

    r0 = r.copy()  # 固定シャドウ残差 (BiCGSTAB の規約通り、以後変更しない)
    rho_old = 1.0
    alpha = 1.0
    omega = 1.0
    v = np.zeros(n, dtype=np.float64)
    p = np.zeros(n, dtype=np.float64)

    for it in range(1, max_iter + 1):
        rho_new = float(np.dot(r0, r))
        if rho_new == 0.0:
            # シャドウ残差と直交してしまった breakdown。再起動 (r0 の取り直し) は
            # 実装せず、呼び出し側の spsolve フォールバックに委ねる (堅牢性優先、
            # モジュールの設計方針どおり複雑な再起動ロジックは持ち込まない)
            return x, it - 1, False
        if it == 1:
            p = r.copy()
        else:
            beta = (rho_new / rho_old) * (alpha / omega)
            p = r + beta * (p - omega * v)
        p_hat = diag_inv * p
        v = csr_matvec_parallel(indptr, indices, data, p_hat)
        denom = float(np.dot(r0, v))
        if denom == 0.0:
            return x, it, False
        alpha = rho_new / denom
        s = r - alpha * v
        if float(np.linalg.norm(s)) <= tol:
            x = x + alpha * p_hat
            return x, it, True
        s_hat = diag_inv * s
        t = csr_matvec_parallel(indptr, indices, data, s_hat)
        tt = float(np.dot(t, t))
        if tt == 0.0:
            x = x + alpha * p_hat
            return x, it, False
        omega = float(np.dot(t, s) / tt)
        x = x + alpha * p_hat + omega * s_hat
        r = s - omega * t
        if float(np.linalg.norm(r)) <= tol:
            return x, it, True
        rho_old = rho_new

    return x, max_iter, False
