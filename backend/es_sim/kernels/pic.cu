// ES-Sim v2 GPU PIC-MCC kernels on the node-centred Cartesian EB grid (prompts/119).
// ASCII only. Physics follows the verified v1 implementation (es_sim/pic.py,
// es_sim/mcc.py); differences are documented in es_sim/gpic/simulation.py.
//
// Design (sync-free, CUDA-graph friendly):
//  * Particle arrays have a fixed capacity; the live count n lives in device
//    memory (cnt[species]) and kernels read it, so no host round trip is
//    needed per step. Dead particles are marked with w = 0 and removed by the
//    periodic compaction kernels.
//  * Every per-step quantity that changes with time (t, step, V(t), nu_max,
//    accumulation flags, phase bin) is read from device memory ("prm").
//  * New particles (ionization products, secondary electrons) are appended
//    with atomicAdd on cnt[]; overflow of the capacity is flagged.
//
// Grid: nodes (ny+1) x (nx+1), row-major, x fastest. Cell (i,j) spans
// [x0+i*dx, x0+(i+1)*dx] x [y0+j*dy, ...]. Node fields are gathered with
// bilinear (CIC) weights and charge is deposited with the same weights.

#include "common.cuh"

// ---- layout of the device parameter block "prm" (double[]) ------------------
#define P_T          0   // time at the start of the step
#define P_STEP       1   // step index (as double, exact up to 2^53)
#define P_DT         2
#define P_ACCUM      3   // 1 if the step is inside the averaging window
#define P_BIN        4   // RF phase bin of this step (-1 if cycle data is off)
#define P_ION_STEP   5   // 1 if ions are pushed this step (sub-cycling)
#define P_NUMAX_E    6
#define P_PCAND_E    7
#define P_NUMAX_I    8
#define P_PCAND_I    9
#define P_ACC_START  10  // first accumulating step
#define P_PERIOD     11  // RF period (0 = no cycle data)
#define P_NBINS      12
#define P_SUB        13  // ion sub-cycle factor
#define P_T0         14  // time origin (t = t0 + step*dt)
#define P_STEP0      15  // step index origin

// ---- layout of the integer counters "cnt" (unsigned long long[]) -------------
#define C_NE       0   // live electrons (incl. not yet compacted dead ones)
#define C_NI       1
#define C_SNAP_E   2   // snapshot of C_NE taken before appends
#define C_SNAP_I   3
#define C_WALL_E   4
#define C_WALL_I   5
#define C_COLL_E   6
#define C_ION_EV   7
#define C_SEE_EV   8
#define C_OVERFLOW 9
#define C_VMAX_E   10  // max v^2 bits (atomicMax on non-negative double bits)
#define C_VMAX_I   11
#define C_PHIMIN   12  // order-preserving bits of min(phi)
#define C_PHIMAX   13

__device__ __forceinline__ unsigned long long es_ord(double v)
{
    const unsigned long long b = (unsigned long long)__double_as_longlong(v);
    return (b & 0x8000000000000000ull) ? ~b : (b | 0x8000000000000000ull);
}

// ---------------------------------------------------------------------------
// Grid variants. This source is compiled twice: for the uniform Cartesian grid
// (default) and, with -DES_AMR, for the block-structured AMR composite grid
// (prompts/122). The AMR variant appends five table pointers to every kernel
// that maps positions to nodes (ES_GRID_PARAMS) and finds the leaf cell with
// es_locate(); node arrays are then indexed by composite node number.
//   amr_dd  : x0, y0, 1/dx0, 1/dy0 (level 0)
//   amr_di  : L, bf, nx0, ny0, then per level l: offset_l, nbx_l
//   amr_ref : per level, per block (row-major): 1 if the block is refined
//   amr_bid : per level, per block: block number (-1 outside the level region)
//   amr_tab : per block, (bf+1)^2 composite node numbers of its grid points
// ---------------------------------------------------------------------------
#ifdef ES_AMR
#define ES_GRID_PARAMS , const double* __restrict__ amr_dd, const long long* __restrict__ amr_di, \
    const unsigned char* __restrict__ amr_ref, const int* __restrict__ amr_bid, const int* __restrict__ amr_tab
#define ES_GRID_ARGS , amr_dd, amr_di, amr_ref, amr_bid, amr_tab

// Leaf cell of (px, py): its 4 corner nodes (00, 10, 01, 11), bilinear weights
// and a cell number (block * bf^2 + local). Coordinates are doubled exactly at
// each level, so the child index is always consistent with the parent cell.
__device__ __forceinline__ void es_locate(double px, double py, int* nd, double* wx, double* wy,
                                          long long* cell ES_GRID_PARAMS)
{
    const int L = (int)amr_di[0], bf = (int)amr_di[1];
    int nx = (int)amr_di[2], ny = (int)amr_di[3];
    double fx = (px - amr_dd[0]) * amr_dd[2];
    double fy = (py - amr_dd[1]) * amr_dd[3];
    int lvl = 0;
    for (;;) {
        int i = (int)floor(fx), j = (int)floor(fy);
        i = i < 0 ? 0 : (i > nx - 1 ? nx - 1 : i);
        j = j < 0 ? 0 : (j > ny - 1 ? ny - 1 : j);
        const long long off = amr_di[4 + 2 * lvl];
        const int nbx = (int)amr_di[5 + 2 * lvl];
        const int bi = i / bf, bj = j / bf;
        const long long bk = off + (long long)bj * nbx + bi;
        if (lvl < L && amr_ref[bk]) {
            ++lvl;
            fx *= 2.0;
            fy *= 2.0;
            nx *= 2;
            ny *= 2;
            continue;
        }
        const long long b = (long long)amr_bid[bk];
        const int a = i - bi * bf, c = j - bj * bf;
        const int s = bf + 1;
        const int* t = amr_tab + b * s * s;
        nd[0] = t[c * s + a];
        nd[1] = t[c * s + a + 1];
        nd[2] = t[(c + 1) * s + a];
        nd[3] = t[(c + 1) * s + a + 1];
        const double ux = fx - i, uy = fy - j;
        *wx = ux < 0.0 ? 0.0 : (ux > 1.0 ? 1.0 : ux);
        *wy = uy < 0.0 ? 0.0 : (uy > 1.0 ? 1.0 : uy);
        *cell = b * bf * bf + c * bf + a;
        return;
    }
}
#else
#define ES_GRID_PARAMS
#define ES_GRID_ARGS
#endif

// ---------------------------------------------------------------------------
// Per-step bookkeeping: time, flags, phase bin, zeroed diagnostics.
// ---------------------------------------------------------------------------
extern "C" __global__ void begin_step(double* __restrict__ prm, unsigned long long* __restrict__ cnt,
                                      double* __restrict__ dsl, const int n_dsl)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const double step = prm[P_STEP];
    const double t = prm[P_T0] + (step - prm[P_STEP0]) * prm[P_DT];
    prm[P_T] = t;
    prm[P_ACCUM] = (step + 1.0 >= prm[P_ACC_START]) ? 1.0 : 0.0;
    double bin = -1.0;
    if (prm[P_PERIOD] > 0.0) {
        double ph = t / prm[P_PERIOD];
        ph -= floor(ph);
        int b = (int)(ph * prm[P_NBINS]);
        const int nb = (int)prm[P_NBINS];
        if (b > nb - 1) b = nb - 1;
        if (b < 0) b = 0;
        bin = (double)b;
    }
    prm[P_BIN] = bin;
    const long long sub = (long long)prm[P_SUB];
    const long long si = (long long)step;
    prm[P_ION_STEP] = (sub <= 1 || (si % sub) == 0) ? 1.0 : 0.0;
    cnt[C_SNAP_E] = cnt[C_NE];
    cnt[C_SNAP_I] = cnt[C_NI];
    cnt[C_VMAX_E] = 0ull;
    cnt[C_VMAX_I] = 0ull;
    cnt[C_PHIMIN] = 0xFFFFFFFFFFFFFFFFull;
    cnt[C_PHIMAX] = 0ull;
    // diagnostic slots: 0 ke_e, 1 ke_i (kept on non-ion steps), 2 fe
    dsl[0] = 0.0;
    if (prm[P_ION_STEP] != 0.0) dsl[1] = 0.0;
    dsl[2] = 0.0;
    for (int k = 3; k < n_dsl; ++k) dsl[k] = 0.0;
}

extern "C" __global__ void end_step(double* __restrict__ prm)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) prm[P_STEP] += 1.0;
}

// ---------------------------------------------------------------------------
// Dirichlet group potentials V_k(t) = DC + sum_m A sin(w t + phi) + waveform(t)
// (same formulas as es_sim.pic._eval_rf / _eval_waveform).
// ---------------------------------------------------------------------------
extern "C" __global__ void eval_groups(double* __restrict__ vg, const int n_groups,
                                       const double* __restrict__ dc,
                                       const int* __restrict__ rf_off, const double* __restrict__ rf_amp,
                                       const double* __restrict__ rf_omega, const double* __restrict__ rf_phase,
                                       const int* __restrict__ wf_off, const double* __restrict__ wf_freq,
                                       const double* __restrict__ wf_phase, const double* __restrict__ wf_v,
                                       const double* __restrict__ prm)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_groups) return;
    const double t = prm[P_T];
    double v = dc[k];
    for (int m = rf_off[k]; m < rf_off[k + 1]; ++m) v += rf_amp[m] * sin(rf_omega[m] * t + rf_phase[m]);
    const int w0 = wf_off[k], w1 = wf_off[k + 1];
    if (w1 > w0) {
        double fr = t * wf_freq[k];
        fr -= floor(fr);
        // phase_ext = phase + [phase[0] + 1], v_ext = v + [v[0]]; np.interp semantics
        const int len = w1 - w0;
        const double* ph = wf_phase + w0;
        const double* vv = wf_v + w0;
        double val;
        if (fr <= ph[0]) val = vv[0];
        else {
            val = vv[0];
            bool found = false;
            for (int q = 0; q < len - 1; ++q) {
                if (fr <= ph[q + 1]) {
                    const double d = ph[q + 1] - ph[q];
                    val = d > 0.0 ? vv[q] + (vv[q + 1] - vv[q]) * (fr - ph[q]) / d : vv[q + 1];
                    found = true;
                    break;
                }
            }
            if (!found) {
                const double pe = ph[0] + 1.0, pl = ph[len - 1];
                const double d = pe - pl;
                val = d > 0.0 ? vv[len - 1] + (vv[0] - vv[len - 1]) * (fr - pl) / d : vv[0];
            }
        }
        v += val;
    }
    vg[k] = v;
}

// b = q_static + rho + q_surf  (elementwise)
extern "C" __global__ void rhs_base(double* __restrict__ b, const double* __restrict__ qs,
                                    const double* __restrict__ rho, const double* __restrict__ qsurf,
                                    const long long n)
{
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) b[k] = qs[k] + rho[k] + qsurf[k];
}

// b[row] += val * V[col] for the Dirichlet couplings (COO)
extern "C" __global__ void rhs_coupling(double* __restrict__ b, const long long* __restrict__ row,
                                        const long long* __restrict__ col, const double* __restrict__ val,
                                        const double* __restrict__ vg, const long long nnz)
{
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nnz) atomicAdd(b + row[k], val[k] * vg[col[k]]);
}

// phi = fixed ? V[group] : x ; periodic slaves copy their masters; also min/max.
extern "C" __global__ void fill_phi(double* __restrict__ phi, const double* __restrict__ x,
                                    const unsigned char* __restrict__ fixed, const long long* __restrict__ fgrp,
                                    const double* __restrict__ vg, const int nx, const int ny,
                                    const int px, const int py, unsigned long long* __restrict__ cnt)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > nx || j > ny) return;
    int im = (px && i == nx) ? 0 : i;
    int jm = (py && j == ny) ? 0 : j;
    const int km = jm * (nx + 1) + im;
    const double v = fixed[km] ? vg[fgrp[km]] : x[km];
    phi[j * (nx + 1) + i] = v;
    const unsigned long long o = es_ord(v);
    atomicMin(cnt + C_PHIMIN, o);
    atomicMax(cnt + C_PHIMAX, o);
}

// ---------------------------------------------------------------------------
// Node electric field E = -grad(phi) (see es_sim/field/electrostatic.py).
// ---------------------------------------------------------------------------
__device__ __forceinline__ double es_deriv(const double* phi, const unsigned char* mask,
                                           const double* th, const int* gr, const double* vg,
                                           int i, int j, int nx, int ny, int axis, int per,
                                           double h, long long plane)
{
    const int s = nx + 1;
    const int k = j * s + i;
    const int n_ax = axis == 0 ? nx : ny;
    const int c_ax = axis == 0 ? i : j;
    const int dm = axis == 0 ? 0 : 2;
    const int dp = axis == 0 ? 1 : 3;
    const double f0 = phi[k];
    int kp = -1, km = -1;
    if (c_ax < n_ax || per) {
        int cp = c_ax + 1;
        if (per && cp >= n_ax) cp -= n_ax;
        kp = axis == 0 ? j * s + cp : cp * s + i;
    }
    if (c_ax > 0 || per) {
        int cm = c_ax - 1;
        if (cm < 0) cm += n_ax;
        km = axis == 0 ? j * s + cm : cm * s + i;
    }
    if (mask[k] == 0) {
        double fp = 0.0, fm = 0.0, hp = h, hm = h;
        bool has_p = false, has_m = false;
        if (th[dp * plane + k] > 0.0) { fp = vg[gr[dp * plane + k]]; hp = th[dp * plane + k] * h; has_p = true; }
        else if (kp >= 0) { fp = phi[kp]; has_p = true; }
        if (th[dm * plane + k] > 0.0) { fm = vg[gr[dm * plane + k]]; hm = th[dm * plane + k] * h; has_m = true; }
        else if (km >= 0) { fm = phi[km]; has_m = true; }
        if (has_p && has_m)
            return (hm * hm * (fp - f0) + hp * hp * (f0 - fm)) / (hp * hm * (hp + hm));
        return 0.0;
    }
    double sum = 0.0;
    int c = 0;
    if (kp >= 0 && mask[kp] == 0) {
        const double t = th[dm * plane + kp];
        sum += (phi[kp] - f0) / ((t > 0.0 ? t : 1.0) * h);
        ++c;
    }
    if (km >= 0 && mask[km] == 0) {
        const double t = th[dp * plane + km];
        sum += (f0 - phi[km]) / ((t > 0.0 ? t : 1.0) * h);
        ++c;
    }
    return c ? sum / c : 0.0;
}

extern "C" __global__ void efield_nodes(double* __restrict__ ex, double* __restrict__ ey,
                                        const double* __restrict__ phi,
                                        const unsigned char* __restrict__ mask,
                                        const double* __restrict__ cut_theta,
                                        const int* __restrict__ cut_group,
                                        const double* __restrict__ vgroup,
                                        const int nx, const int ny, const double dx,
                                        const double dy, const int px, const int py)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > nx || j > ny) return;
    const long long plane = (long long)(nx + 1) * (ny + 1);
    const int im = (px && i == nx) ? 0 : i;
    const int jm = (py && j == ny) ? 0 : j;
    const int k = j * (nx + 1) + i;
    ex[k] = -es_deriv(phi, mask, cut_theta, cut_group, vgroup, im, jm, nx, ny, 0, px, dx, plane);
    ey[k] = -es_deriv(phi, mask, cut_theta, cut_group, vgroup, im, jm, nx, ny, 1, py, dy, plane);
}

// ---------------------------------------------------------------------------
// Discrete field energy: sum over unknown-unknown edges + Dirichlet couplings,
// accumulated into out[0] (the 1/2 and 2*pi factors are applied on the host).
// ---------------------------------------------------------------------------
extern "C" __global__ void edge_energy(const double* __restrict__ phi,
                                       const double* __restrict__ cx, const double* __restrict__ cy,
                                       const int nx, const int ny, const int px, const int py,
                                       const long long* __restrict__ crow, const long long* __restrict__ ccol,
                                       const double* __restrict__ cval, const long long nnz,
                                       const double* __restrict__ vg, double* __restrict__ out)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long n_cx = (long long)(ny + 1) * nx;
    const long long n_cy = (long long)ny * (nx + 1);
    const int s = nx + 1;
    double e = 0.0;
    if (t < n_cx) {
        const int j = (int)(t / nx), i = (int)(t % nx);
        const double c = cx[t];
        if (c != 0.0) {
            const int i1 = (px && i + 1 == nx) ? 0 : i + 1;
            const double d = phi[j * s + i] - phi[j * s + i1];
            e = c * d * d;
        }
    } else if (t < n_cx + n_cy) {
        const long long u = t - n_cx;
        const int j = (int)(u / s), i = (int)(u % s);
        const double c = cy[u];
        if (c != 0.0) {
            const int j1 = (py && j + 1 == ny) ? 0 : j + 1;
            const double d = phi[j * s + i] - phi[j1 * s + i];
            e = c * d * d;
        }
    } else if (t < n_cx + n_cy + nnz) {
        const long long u = t - n_cx - n_cy;
        const double d = phi[crow[u]] - vg[ccol[u]];
        e = cval[u] * d * d;
    }
    es_block_sum_atomic(e, out);
}

// ---------------------------------------------------------------------------
// Push. Planar 2d3v leapfrog (optional uniform-B Boris with precomputed R) or
// axisymmetric ring push (rotation method). species: 0 electron, 1 ion. Ions
// are pushed only on ion steps (sub-cycling). ke accumulates 0.5*m*w*v_old.v_new.
// Dead particles (w == 0) are skipped.
// ---------------------------------------------------------------------------
extern "C" __global__ void push(double* __restrict__ x, double* __restrict__ y,
                                double* __restrict__ vx, double* __restrict__ vy,
                                double* __restrict__ vz, const double* __restrict__ w,
                                const unsigned long long* __restrict__ cnt, const int species,
                                const double* __restrict__ ex, const double* __restrict__ ey,
                                const double x0, const double y0, const double inv_dx,
                                const double inv_dy, const int nx, const int ny,
                                const double qm, const double dt_s, const int rz, const int ridx,
                                const int use_b, const double* __restrict__ R, const double half_m,
                                const double* __restrict__ prm, double* __restrict__ ke_out ES_GRID_PARAMS)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long n = (long long)cnt[species == 0 ? C_NE : C_NI];
    const bool active = (species == 0) || (prm[P_ION_STEP] != 0.0);
    double ke = 0.0;
    if (active && p < n && w[p] != 0.0) {
        double wx, wy;
#ifdef ES_AMR
        int nd[4];
        long long cell;
        es_locate(x[p], y[p], nd, &wx, &wy, &cell ES_GRID_ARGS);
        const double w00 = (1.0 - wx) * (1.0 - wy), w10 = wx * (1.0 - wy);
        const double w01 = (1.0 - wx) * wy, w11 = wx * wy;
        const double Ex = w00 * ex[nd[0]] + w10 * ex[nd[1]] + w01 * ex[nd[2]] + w11 * ex[nd[3]];
        const double Ey = w00 * ey[nd[0]] + w10 * ey[nd[1]] + w01 * ey[nd[2]] + w11 * ey[nd[3]];
#else
        int i, j;
        es_cell(x[p], y[p], x0, y0, inv_dx, inv_dy, nx, ny, &i, &j, &wx, &wy);
        const int s = nx + 1;
        const int k = j * s + i;
        const double w00 = (1.0 - wx) * (1.0 - wy), w10 = wx * (1.0 - wy);
        const double w01 = (1.0 - wx) * wy, w11 = wx * wy;
        const double Ex = w00 * ex[k] + w10 * ex[k + 1] + w01 * ex[k + s] + w11 * ex[k + s + 1];
        const double Ey = w00 * ey[k] + w10 * ey[k + 1] + w01 * ey[k + s] + w11 * ey[k + s + 1];
#endif
        const double qmdt = qm * dt_s;
        const double ux = vx[p], uy = vy[p], uz = vz[p];
        if (!rz) {
            double ax, ay, az;
            if (use_b) {
                const double hx = 0.5 * qmdt * Ex, hy = 0.5 * qmdt * Ey;
                const double mx = ux + hx, my = uy + hy, mz = uz;
                ax = R[0] * mx + R[1] * my + R[2] * mz + hx;
                ay = R[3] * mx + R[4] * my + R[5] * mz + hy;
                az = R[6] * mx + R[7] * my + R[8] * mz;
            } else {
                ax = ux + qmdt * Ex;
                ay = uy + qmdt * Ey;
                az = uz;
            }
            ke = half_m * w[p] * (ux * ax + uy * ay + uz * az);
            vx[p] = ax;
            vy[p] = ay;
            vz[p] = az;
            x[p] += ax * dt_s;
            y[p] += ay * dt_s;
        } else {
            const double ax = ux + qmdt * Ex, ay = uy + qmdt * Ey;
            ke = half_m * w[p] * (ux * ax + uy * ay + uz * uz);
            double r, vr, zc, vzc;
            if (ridx == 1) { r = y[p]; vr = ay; zc = x[p]; vzc = ax; }
            else           { r = x[p]; vr = ax; zc = y[p]; vzc = ay; }
            const double xr = r + vr * dt_s;
            const double yp = uz * dt_s;
            const double rn = sqrt(xr * xr + yp * yp);
            double c = 1.0, sn = 0.0;
            if (rn > 0.0) { c = xr / rn; sn = yp / rn; }
            const double vr_n = c * vr + sn * uz;
            const double vt_n = -sn * vr + c * uz;
            zc += vzc * dt_s;
            if (ridx == 1) { x[p] = zc; y[p] = rn; vx[p] = vzc; vy[p] = vr_n; }
            else           { x[p] = rn; y[p] = zc; vx[p] = vr_n; vy[p] = vzc; }
            vz[p] = vt_n;
        }
    }
    es_block_sum_atomic(ke, ke_out);
}

// ---------------------------------------------------------------------------
// Boundary handling after the push, done entirely on the device:
//   domain sides (0 absorb, 1 reflect, 2 periodic) and embedded solids.
// For every absorbed particle the exact hit point and the gas-side normal are
// computed, then (a) dielectric surface charge is deposited, (b) IEDF/IADF
// collector records are appended (ions, averaging window), (c) secondary
// electrons are emitted with probability gamma (ions), (d) the particle is
// marked dead (w = 0) and the wall counter is incremented.
// Solids: type 0 = polygon (vertices [off[s], off[s+1]) in pxy), 1 = circle
// (cx, cy, r in circ[3*s]); kind 1 = conductor, 2 = dielectric; index = number
// within its kind (for the gamma tables).
// Collector parameters (8 doubles each): p1x p1y tx ty nx ny len tol.
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool es_in_polygon(double px, double py, const double* v, int nv)
{
    bool odd = false;
    for (int a = 0, b = nv - 1; a < nv; b = a++) {
        const double ax = v[2 * a], ay = v[2 * a + 1];
        const double bx = v[2 * b], by = v[2 * b + 1];
        if ((ay > py) != (by > py)) {
            const double xc = ax + (py - ay) * (bx - ax) / (by - ay);
            if (px < xc) odd = !odd;
        }
    }
    return odd;
}

__device__ __forceinline__ void es_dep_point(double* __restrict__ out, double px, double py, double q,
                                             double x0, double y0, double inv_dx, double inv_dy,
                                             int nx, int ny, int pxp, int pyp ES_GRID_PARAMS)
{
    double wx, wy;
#ifdef ES_AMR
    int nd[4];
    long long cell;
    es_locate(px, py, nd, &wx, &wy, &cell ES_GRID_ARGS);
    atomicAdd(out + nd[0], q * (1.0 - wx) * (1.0 - wy));
    atomicAdd(out + nd[1], q * wx * (1.0 - wy));
    atomicAdd(out + nd[2], q * (1.0 - wx) * wy);
    atomicAdd(out + nd[3], q * wx * wy);
#else
    int i, j;
    es_cell(px, py, x0, y0, inv_dx, inv_dy, nx, ny, &i, &j, &wx, &wy);
    const int s = nx + 1;
    int i1 = i + 1, j1 = j + 1;
    if (pxp && i1 == nx) i1 = 0;
    if (pyp && j1 == ny) j1 = 0;
    atomicAdd(out + j * s + i, q * (1.0 - wx) * (1.0 - wy));
    atomicAdd(out + j * s + i1, q * wx * (1.0 - wy));
    atomicAdd(out + j1 * s + i, q * (1.0 - wx) * wy);
    atomicAdd(out + j1 * s + i1, q * wx * wy);
#endif
}

extern "C" __global__ void boundary(
    double* __restrict__ x, double* __restrict__ y, double* __restrict__ vx, double* __restrict__ vy,
    double* __restrict__ vz, double* __restrict__ w, unsigned long long* __restrict__ cnt,
    const int species, const double dt_s, const int rz, const int ridx,
    const double X0, const double Y0, const double X1, const double Y1, const int* __restrict__ side_kind,
    const unsigned char* __restrict__ cell_state, const double inv_dx, const double inv_dy,
    const int nx, const int ny, const int pxp, const int pyp,
    const int n_solid, const int* __restrict__ s_type, const int* __restrict__ s_kind,
    const int* __restrict__ s_index, const int* __restrict__ s_off, const double* __restrict__ pxy,
    const double* __restrict__ circ,
    const double* __restrict__ side_gamma, const double* __restrict__ cond_gamma,
    const double* __restrict__ diel_gamma,
    double* __restrict__ qsurf, const double qw_scale, const double two_pi_inv,
    const int see_on, const double see_speed, const double see_delta,
    double* __restrict__ ex_, double* __restrict__ ey_, double* __restrict__ evx, double* __restrict__ evy,
    double* __restrict__ evz, double* __restrict__ ew, const long long cap_e,
    const int n_coll, const double* __restrict__ coll, const double mass,
    double* __restrict__ rec_e, double* __restrict__ rec_a, double* __restrict__ rec_w,
    unsigned long long* __restrict__ rec_n, double* __restrict__ coll_w, const long long rec_cap,
    const double* __restrict__ prm, const unsigned long long seed,
    const int* __restrict__ side_elec, const int* __restrict__ cond_elec, double* __restrict__ cap_dq ES_GRID_PARAMS)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long n = (long long)cnt[species == 0 ? C_NE : C_NI];
    if (p >= n) return;
    if (species == 1 && prm[P_ION_STEP] == 0.0) return;
    const double wp = w[p];
    if (wp == 0.0) return;
    double px = x[p], py = y[p];
    int st = 0;
    const double lx = X1 - X0, ly = Y1 - Y0;
    if (px < X0) {
        const int k = side_kind[0];
        if (k == 2) px += lx;
        else if (k == 1) { px = 2.0 * X0 - px; vx[p] = -vx[p]; }
        else st = 1;
    } else if (px >= X1) {
        const int k = side_kind[1];
        if (k == 2) px -= lx;
        else if (k == 1) { px = 2.0 * X1 - px; vx[p] = -vx[p]; }
        else st = 2;
    }
    if (st == 0) {
        if (py < Y0) {
            const int k = side_kind[2];
            if (k == 2) py += ly;
            else if (k == 1) { py = 2.0 * Y0 - py; vy[p] = -vy[p]; }
            else st = 3;
        } else if (py >= Y1) {
            const int k = side_kind[3];
            if (k == 2) py -= ly;
            else if (k == 1) { py = 2.0 * Y1 - py; vy[p] = -vy[p]; }
            else st = 4;
        }
    }
    if (st == 0 && (px < X0 || px >= X1 || py < Y0 || py >= Y1)) st = 1;
    if (st == 0) {
        int i = (int)floor((px - X0) * inv_dx);
        int j = (int)floor((py - Y0) * inv_dy);
        i = i < 0 ? 0 : (i > nx - 1 ? nx - 1 : i);
        j = j < 0 ? 0 : (j > ny - 1 ? ny - 1 : j);
        const unsigned char cs = cell_state[j * nx + i];
        if (cs == 1) st = 5;
        else if (cs == 2) {
            for (int s = 0; s < n_solid && st == 0; ++s) {
                if (s_type[s] == 1) {
                    const double dx = px - circ[3 * s], dy = py - circ[3 * s + 1], r = circ[3 * s + 2];
                    if (dx * dx + dy * dy <= r * r) st = 5;
                } else if (es_in_polygon(px, py, pxy + 2 * s_off[s], s_off[s + 1] - s_off[s])) {
                    st = 5;
                }
            }
        }
    }
    if (st == 0) {
        x[p] = px;
        y[p] = py;
        return;
    }

    // ---- absorbed: reconstruct the pre-push position (straight line in 3D) ----
    const double ux = vx[p], uy = vy[p], uz = vz[p];
    double ox, oy;
    if (!rz) {
        ox = px - ux * dt_s;
        oy = py - uy * dt_s;
    } else {
        double r, vr, zc, vzc;
        if (ridx == 1) { r = py; vr = uy; zc = px; vzc = ux; }
        else           { r = px; vr = ux; zc = py; vzc = uy; }
        const double xr = r - vr * dt_s, yp = -uz * dt_s;
        const double r0 = sqrt(xr * xr + yp * yp);
        const double z0 = zc - vzc * dt_s;
        if (ridx == 1) { ox = z0; oy = r0; } else { ox = r0; oy = z0; }
    }
    double hx = px, hy = py, nxn = 0.0, nyn = 0.0, gam = 0.0;
    int diel = -1;
    int elec = -1;  // blocking-capacitor electrode that receives the charge (prompts/134), -1 = none
    if (st <= 4) {
        double f;
        if (st == 1 || st == 2) {
            const double xb = st == 1 ? X0 : X1;
            const double den = px - ox;
            f = den != 0.0 ? (xb - ox) / den : 1.0;
            nxn = st == 1 ? 1.0 : -1.0;
        } else {
            const double yb = st == 3 ? Y0 : Y1;
            const double den = py - oy;
            f = den != 0.0 ? (yb - oy) / den : 1.0;
            nyn = st == 3 ? 1.0 : -1.0;
        }
        f = fmin(fmax(f, 0.0), 1.0);
        hx = ox + f * (px - ox);
        hy = oy + f * (py - oy);
        gam = side_gamma[st - 1];
        elec = side_elec[st - 1];
    } else {
        const double dxs = px - ox, dys = py - oy;
        double best = 2.0;
        int bkind = -1, bidx = -1;
        for (int s = 0; s < n_solid; ++s) {
            if (s_type[s] == 1) {
                const double cxs = circ[3 * s], cys = circ[3 * s + 1], r = circ[3 * s + 2];
                const double fx = ox - cxs, fy = oy - cys;
                const double a = dxs * dxs + dys * dys;
                const double b = 2.0 * (fx * dxs + fy * dys);
                const double c = fx * fx + fy * fy - r * r;
                const double disc = b * b - 4.0 * a * c;
                if (a > 0.0 && disc >= 0.0) {
                    const double t0 = (-b - sqrt(disc)) / (2.0 * a);
                    if (t0 >= 0.0 && t0 <= 1.0 && t0 < best) {
                        best = t0;
                        const double hxs = ox + t0 * dxs - cxs, hys = oy + t0 * dys - cys;
                        const double nr = sqrt(hxs * hxs + hys * hys);
                        nxn = nr > 0.0 ? hxs / nr : 0.0;
                        nyn = nr > 0.0 ? hys / nr : 0.0;
                        bkind = s_kind[s];
                        bidx = s_index[s];
                    }
                }
            } else {
                const double* v = pxy + 2 * s_off[s];
                const int nv = s_off[s + 1] - s_off[s];
                for (int a = 0; a < nv; ++a) {
                    const int b2 = (a + 1 == nv) ? 0 : a + 1;
                    const double qx = v[2 * a], qy = v[2 * a + 1];
                    const double ex2 = v[2 * b2] - qx, ey2 = v[2 * b2 + 1] - qy;
                    const double den = dxs * ey2 - dys * ex2;
                    if (den == 0.0) continue;
                    const double wx2 = qx - ox, wy2 = qy - oy;
                    const double t = (wx2 * ey2 - wy2 * ex2) / den;
                    const double u = (wx2 * dys - wy2 * dxs) / den;
                    if (t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 && t < best) {
                        best = t;
                        const double el = sqrt(ex2 * ex2 + ey2 * ey2);
                        nxn = el > 0.0 ? ey2 / el : 0.0;
                        nyn = el > 0.0 ? -ex2 / el : 0.0;
                        bkind = s_kind[s];
                        bidx = s_index[s];
                    }
                }
            }
        }
        if (bkind < 0) {
            best = 1.0;
            const double dl = sqrt(dxs * dxs + dys * dys);
            nxn = dl > 0.0 ? -dxs / dl : 0.0;
            nyn = dl > 0.0 ? -dys / dl : 0.0;
        }
        if (nxn * dxs + nyn * dys > 0.0) { nxn = -nxn; nyn = -nyn; }
        hx = ox + best * dxs;
        hy = oy + best * dys;
        if (bkind == 1) { gam = cond_gamma[bidx]; elec = cond_elec[bidx]; }
        else if (bkind == 2) { gam = diel_gamma[bidx]; diel = bidx; }
    }

    // (a) dielectric surface charge (RHS units: /2pi in RZ), or the charge of a blocking-capacitor electrode
    if (diel >= 0)
        es_dep_point(qsurf, hx, hy, qw_scale * wp * two_pi_inv, X0, Y0, inv_dx, inv_dy, nx, ny, pxp, pyp
                     ES_GRID_ARGS);
    if (elec >= 0) atomicAdd(cap_dq + elec, qw_scale * wp);

    // (b) IEDF/IADF collectors (ions, averaging window)
    if (species == 1 && n_coll > 0 && prm[P_ACCUM] != 0.0) {
        for (int c = 0; c < n_coll; ++c) {
            const double* cc = coll + 8 * c;
            const double ddx = hx - cc[0], ddy = hy - cc[1];
            const double proj = ddx * cc[2] + ddy * cc[3];
            const double dist = fabs(ddx * cc[4] + ddy * cc[5]);
            if (dist <= cc[7] && proj >= 0.0 && proj <= cc[6]) {
                atomicAdd(coll_w + c, wp);
                const unsigned long long slot = atomicAdd(rec_n + c, 1ull);
                if ((long long)slot < rec_cap) {
                    const long long o = (long long)c * rec_cap + (long long)slot;
                    rec_e[o] = 0.5 * mass * (ux * ux + uy * uy + uz * uz) / 1.602176634e-19;
                    const double vt = ux * cc[2] + uy * cc[3];
                    const double vn = ux * cc[4] + uy * cc[5];
                    rec_a[o] = atan2(vt, fabs(vn)) * 57.29577951308232;
                    rec_w[o] = wp;
                }
            }
        }
    }

    // (c) secondary electrons (ions only)
    if (species == 1 && see_on && gam > 0.0) {
        EsRng rng;
        rng.init((unsigned long long)p, (unsigned long long)prm[P_STEP], 4u, seed);
        if (rng.uniform() < gam) {
            const unsigned long long k = atomicAdd(cnt + C_NE, 1ull);
            if ((long long)k < cap_e) {
                ex_[k] = hx + see_delta * nxn;
                ey_[k] = hy + see_delta * nyn;
                evx[k] = see_speed * nxn;
                evy[k] = see_speed * nyn;
                evz[k] = 0.0;
                ew[k] = wp;
                atomicAdd(cnt + C_SEE_EV, 1ull);
                if (diel >= 0)  // the surface loses an electron: +e*w
                    es_dep_point(qsurf, hx, hy, 1.602176634e-19 * wp * two_pi_inv, X0, Y0, inv_dx, inv_dy,
                                 nx, ny, pxp, pyp ES_GRID_ARGS);
                if (elec >= 0) atomicAdd(cap_dq + elec, 1.602176634e-19 * wp);  // the electrode too
            } else {
                atomicAdd(cnt + C_OVERFLOW, 1ull);
            }
        }
    }

    // (d) remove
    x[p] = hx;
    y[p] = hy;
    w[p] = 0.0;
    atomicAdd(cnt + (species == 0 ? C_WALL_E : C_WALL_I), 1ull);
}

// ---------------------------------------------------------------------------
// CIC deposit of particles (dead ones have w = 0 and contribute nothing).
// mode 0: value = w; mode 1: value = w * 0.5*m*v^2 (kinetic energy).
// out2 (optional, stride = N nodes) receives the same deposit at bin*N when
// bin >= 0 (phase-resolved accumulation). gate: 0 = always, 1 = only when the
// step is accumulating, 2 = only on ion steps.
// ---------------------------------------------------------------------------
extern "C" __global__ void deposit(const double* __restrict__ x, const double* __restrict__ y,
                                   const double* __restrict__ vx, const double* __restrict__ vy,
                                   const double* __restrict__ vz, const double* __restrict__ w,
                                   const unsigned long long* __restrict__ cnt, const int species,
                                   const int mode, const double scale, double* __restrict__ out,
                                   double* __restrict__ out2, const int use_out2, const long long n_nodes,
                                   const int gate, const double* __restrict__ prm,
                                   const double x0, const double y0, const double inv_dx,
                                   const double inv_dy, const int nx, const int ny, const int px, const int py
                                   ES_GRID_PARAMS)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gate == 1 && prm[P_ACCUM] == 0.0) return;
    const long long n = (long long)cnt[species == 0 ? C_NE : C_NI];
    if (p >= n) return;
    const double wp = w[p];
    if (wp == 0.0) return;
    double v = wp;
    if (mode == 1) v *= (vx[p] * vx[p] + vy[p] * vy[p] + vz[p] * vz[p]);
    v *= scale;
    es_dep_point(out, x[p], y[p], v, x0, y0, inv_dx, inv_dy, nx, ny, px, py ES_GRID_ARGS);
    if (use_out2) {
        const int bin = (int)prm[P_BIN];
        if (bin >= 0)
            es_dep_point(out2 + (long long)bin * n_nodes, x[p], y[p], v, x0, y0, inv_dx, inv_dy, nx, ny, px, py
                         ES_GRID_ARGS);
    }
}

// Accumulate phi into the running sums (and the phase bin) when accumulating.
extern "C" __global__ void accum_phi(const double* __restrict__ phi, double* __restrict__ acc,
                                     double* __restrict__ cyc, const int use_cyc, const long long n,
                                     const double* __restrict__ prm, unsigned long long* __restrict__ cyc_count)
{
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (prm[P_ACCUM] == 0.0) return;
    const int bin = (int)prm[P_BIN];
    if (k == 0 && use_cyc && bin >= 0) atomicAdd(cyc_count + bin, 1ull);
    if (k >= n) return;
    acc[k] += phi[k];
    if (use_cyc && bin >= 0) cyc[(long long)bin * n + k] += phi[k];
}

// Nearest-cell histogram (per-cell weight sums) for the frame element densities.
extern "C" __global__ void deposit_cell(const double* __restrict__ x, const double* __restrict__ y,
                                        const double* __restrict__ w, const unsigned long long* __restrict__ cnt,
                                        const int species, double* __restrict__ out, const double x0,
                                        const double y0, const double inv_dx, const double inv_dy,
                                        const int nx, const int ny ES_GRID_PARAMS)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long n = (long long)cnt[species == 0 ? C_NE : C_NI];
    if (p >= n || w[p] == 0.0) return;
    double wx, wy;
#ifdef ES_AMR
    int nd[4];
    long long cell;
    es_locate(x[p], y[p], nd, &wx, &wy, &cell ES_GRID_ARGS);
    atomicAdd(out + cell, w[p]);
#else
    int i, j;
    es_cell(x[p], y[p], x0, y0, inv_dx, inv_dy, nx, ny, &i, &j, &wx, &wy);
    atomicAdd(out + j * nx + i, w[p]);
#endif
}

// ---------------------------------------------------------------------------
// Adaptive nu_max (v1 MccModel._numax_upto): max v^2 of live particles, then a
// lookup in the prefix-max table on the uniform energy grid [0, e_cap].
// ---------------------------------------------------------------------------
extern "C" __global__ void vmax2(const double* __restrict__ vx, const double* __restrict__ vy,
                                 const double* __restrict__ vz, const double* __restrict__ w,
                                 unsigned long long* __restrict__ cnt, const int species,
                                 const double* __restrict__ prm)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (species == 1 && prm[P_ION_STEP] == 0.0) return;
    const long long n = (long long)cnt[species == 0 ? C_NE : C_NI];
    double v = 0.0;
    if (p < n && w[p] != 0.0) v = vx[p] * vx[p] + vy[p] * vy[p] + vz[p] * vz[p];
    // warp max then one atomic per warp
    for (int o = 16; o > 0; o >>= 1) v = fmax(v, __shfl_down_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0 && v > 0.0)
        atomicMax(cnt + (species == 0 ? C_VMAX_E : C_VMAX_I), (unsigned long long)__double_as_longlong(v));
}

// kind 0 (electrons): e = 0.5 m v^2 / qe. kind 1 (ions): lab: 0.5 m_i v^2/qe,
// com: g = v + 6 vth, e = 0.5 mu g^2 / qe (v1 collide_ions margin).
extern "C" __global__ void numax_lookup(double* __restrict__ prm, const unsigned long long* __restrict__ cnt,
                                        const int species, const double m, const double mu,
                                        const double vth_gas, const int com,
                                        const double* __restrict__ pref, const int n_grid, const double e_cap,
                                        const double dt_s)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (species == 1 && prm[P_ION_STEP] == 0.0) return;
    const double v2 = __longlong_as_double((long long)cnt[species == 0 ? C_VMAX_E : C_VMAX_I]);
    double e;
    if (species == 0 || !com) e = 0.5 * m * v2 / 1.602176634e-19;
    else {
        const double g = sqrt(v2) + 6.0 * vth_gas;
        e = 0.5 * mu * g * g / 1.602176634e-19;
    }
    double numax;
    if (e >= e_cap) numax = pref[n_grid - 1];
    else {
        const double de = e_cap / (double)(n_grid - 1);
        long long idx = (long long)floor(e / de) + 1;  // searchsorted(side="right") on linspace
        if (idx > n_grid - 1) idx = n_grid - 1;
        numax = pref[idx];
    }
    const double pc = numax > 0.0 ? 1.0 - exp(-numax * dt_s) : 0.0;
    if (species == 0) { prm[P_NUMAX_E] = numax; prm[P_PCAND_E] = pc; }
    else { prm[P_NUMAX_I] = numax; prm[P_PCAND_I] = pc; }
}

// ---------------------------------------------------------------------------
// Electron MCC with in-kernel ionization products. Process kinds: 0 elastic,
// 1 excitation, 2 ionization (tables right-zero-padded: tab_e, tab_s, tab_len).
// Only particles present before the step's appends (C_SNAP_E) are processed.
// Same physics as es_sim.mcc.MccModel.collide_electrons.
// ---------------------------------------------------------------------------
extern "C" __global__ void mcc_electron(
    double* __restrict__ ex_, double* __restrict__ ey_, double* __restrict__ vx, double* __restrict__ vy,
    double* __restrict__ vz, double* __restrict__ ew, const long long cap_e,
    double* __restrict__ ix, double* __restrict__ iy, double* __restrict__ ivx, double* __restrict__ ivy,
    double* __restrict__ ivz, double* __restrict__ iw, const long long cap_i,
    unsigned long long* __restrict__ cnt, const int n_proc, const int* __restrict__ kind,
    const double* __restrict__ thr, const double* __restrict__ mratio, const double* __restrict__ tab_e,
    const double* __restrict__ tab_s, const int* __restrict__ tab_len, const int tab_w,
    const double n_gas, const int split_half, const double vth_gas, const double* __restrict__ prm,
    const unsigned long long seed,
    double* __restrict__ acc_ion, double* __restrict__ cyc_ion, const int use_cyc, const long long n_nodes,
    const double x0, const double y0, const double inv_dx, const double inv_dy,
    const int nx, const int ny, const int pxp, const int pyp ES_GRID_PARAMS)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long n = (long long)cnt[C_SNAP_E];
    if (p >= n) return;
    const double wp = ew[p];
    if (wp == 0.0) return;
    const double numax = prm[P_NUMAX_E];
    if (numax <= 0.0) return;
    EsRng rng;
    rng.init((unsigned long long)p, (unsigned long long)prm[P_STEP], 1u, seed);
    if (rng.uniform() >= prm[P_PCAND_E]) return;
    const double me = 9.1093837015e-31, qe = 1.602176634e-19;
    const double ux = vx[p], uy = vy[p], uz = vz[p];
    const double v2 = ux * ux + uy * uy + uz * uz;
    const double speed = sqrt(v2);
    const double E = 0.5 * me * v2 / qe;
    const double target = rng.uniform() * numax;
    double cum = 0.0;
    int chosen = -1;
    for (int q = 0; q < n_proc; ++q) {
        double sig = es_interp(tab_e + (long long)q * tab_w, tab_s + (long long)q * tab_w, tab_len[q], E);
        if (kind[q] != 0 && E < thr[q]) sig = 0.0;
        cum += n_gas * sig * speed;
        if (target < cum) { chosen = q; break; }
    }
    if (chosen < 0) return;
    atomicAdd(cnt + C_COLL_E, 1ull);
    const int kd = kind[chosen];
    double dx, dy, dz;
    if (kd != 2) {
        rng.iso_dir(&dx, &dy, &dz);
        double e_new;
        if (kd == 0) {
            const double cos_chi = speed > 0.0 ? (ux * dx + uy * dy + uz * dz) / speed : 0.0;
            e_new = fmax(E * (1.0 - 2.0 * mratio[chosen] * (1.0 - cos_chi)), 0.0);
        } else {
            e_new = E - thr[chosen];
        }
        const double s_new = sqrt(2.0 * e_new * qe / me);
        vx[p] = s_new * dx;
        vy[p] = s_new * dy;
        vz[p] = s_new * dz;
        return;
    }
    // ionization: split the excess energy, append the ejected electron and the new ion
    const double excess = fmax(E - thr[chosen], 0.0);
    const double e_scat = split_half ? 0.5 * excess : rng.uniform() * excess;
    const double e_ej = excess - e_scat;
    rng.iso_dir(&dx, &dy, &dz);
    const double s1 = sqrt(2.0 * e_scat * qe / me);
    vx[p] = s1 * dx;
    vy[p] = s1 * dy;
    vz[p] = s1 * dz;
    rng.iso_dir(&dx, &dy, &dz);
    const double s2 = sqrt(2.0 * e_ej * qe / me);
    double g0, g1, g2, g3;
    rng.normal2(&g0, &g1);
    rng.normal2(&g2, &g3);
    const double xp = ex_[p], yp = ey_[p];
    const unsigned long long ke = atomicAdd(cnt + C_NE, 1ull);
    const unsigned long long ki = atomicAdd(cnt + C_NI, 1ull);
    if ((long long)ke < cap_e && (long long)ki < cap_i) {
        ex_[ke] = xp; ey_[ke] = yp; vx[ke] = s2 * dx; vy[ke] = s2 * dy; vz[ke] = s2 * dz; ew[ke] = wp;
        ix[ki] = xp; iy[ki] = yp; ivx[ki] = vth_gas * g0; ivy[ki] = vth_gas * g1; ivz[ki] = vth_gas * g2; iw[ki] = wp;
        atomicAdd(cnt + C_ION_EV, 1ull);
        if (prm[P_ACCUM] != 0.0) {
            es_dep_point(acc_ion, xp, yp, wp, x0, y0, inv_dx, inv_dy, nx, ny, pxp, pyp ES_GRID_ARGS);
            const int bin = (int)prm[P_BIN];
            if (use_cyc && bin >= 0)
                es_dep_point(cyc_ion + (long long)bin * n_nodes, xp, yp, wp, x0, y0, inv_dx, inv_dy, nx, ny, pxp, pyp
                             ES_GRID_ARGS);
        }
    } else {
        // capacity exceeded: keep counts consistent by marking the slots as dead
        if ((long long)ke < cap_e) ew[ke] = 0.0;
        if ((long long)ki < cap_i) iw[ki] = 0.0;
        atomicAdd(cnt + C_OVERFLOW, 1ull);
    }
}

// ---------------------------------------------------------------------------
// Ion MCC (0 isotropic in the CM frame, 1 backscatter/charge exchange).
// Same physics as es_sim.mcc.MccModel.collide_ions (gas mass = ion mass).
// ---------------------------------------------------------------------------
extern "C" __global__ void mcc_ion(double* __restrict__ vx, double* __restrict__ vy,
                                   double* __restrict__ vz, const double* __restrict__ w,
                                   const unsigned long long* __restrict__ cnt, const int n_proc,
                                   const int* __restrict__ kind, const double* __restrict__ tab_e,
                                   const double* __restrict__ tab_s, const int* __restrict__ tab_len,
                                   const int tab_w, const double n_gas, const double m_i,
                                   const double mu, const double vth_gas, const int com,
                                   const double* __restrict__ prm, const unsigned long long seed)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (prm[P_ION_STEP] == 0.0) return;
    const long long n = (long long)cnt[C_SNAP_I];
    if (p >= n || w[p] == 0.0) return;
    const double numax = prm[P_NUMAX_I];
    if (numax <= 0.0) return;
    EsRng rng;
    rng.init((unsigned long long)p, (unsigned long long)prm[P_STEP], 3u, seed);
    if (rng.uniform() >= prm[P_PCAND_I]) return;
    const double qe = 1.602176634e-19;
    double g0, g1, g2, g3;
    rng.normal2(&g0, &g1);
    rng.normal2(&g2, &g3);
    const double vgx = vth_gas * g0, vgy = vth_gas * g1, vgz = vth_gas * g2;
    const double ux = vx[p], uy = vy[p], uz = vz[p];
    const double gx = ux - vgx, gy = uy - vgy, gz = uz - vgz;
    const double gmag = sqrt(gx * gx + gy * gy + gz * gz);
    double s_ref, e_ref;
    if (com) { s_ref = gmag; e_ref = 0.5 * mu * gmag * gmag / qe; }
    else { s_ref = sqrt(ux * ux + uy * uy + uz * uz); e_ref = 0.5 * m_i * s_ref * s_ref / qe; }
    const double target = rng.uniform() * numax;
    double cum = 0.0;
    int chosen = -1;
    for (int q = 0; q < n_proc; ++q) {
        const double sig = es_interp(tab_e + (long long)q * tab_w, tab_s + (long long)q * tab_w, tab_len[q], e_ref);
        cum += n_gas * sig * s_ref;
        if (target < cum) { chosen = q; break; }
    }
    if (chosen < 0) return;
    if (kind[chosen] == 1) {
        vx[p] = vgx;
        vy[p] = vgy;
        vz[p] = vgz;
    } else {
        double dx, dy, dz;
        rng.iso_dir(&dx, &dy, &dz);
        vx[p] = 0.5 * (ux + vgx) + 0.5 * gmag * dx;
        vy[p] = 0.5 * (uy + vgy) + 0.5 * gmag * dy;
        vz[p] = 0.5 * (uz + vgz) + 0.5 * gmag * dz;
    }
}

// ---------------------------------------------------------------------------
// History row: one row per step in a device ring buffer (flushed by the host).
// Columns: t ke_e ke_i fe n_e n_i wall_e wall_i phi_min phi_max coll_e
//          ion_events see_events surf_q
// ---------------------------------------------------------------------------
__device__ __forceinline__ double es_unord(unsigned long long o)
{
    const unsigned long long b = (o & 0x8000000000000000ull) ? (o & 0x7FFFFFFFFFFFFFFFull) : ~o;
    return __longlong_as_double((long long)b);
}

extern "C" __global__ void history_row(double* __restrict__ hist, const int n_rows, const int n_cols,
                                       const double* __restrict__ prm, const unsigned long long* __restrict__ cnt,
                                       const double* __restrict__ dsl, const double fe_scale,
                                       const double surf_scale)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long step = (long long)prm[P_STEP];
    double* r = hist + (step % n_rows) * n_cols;
    r[0] = prm[P_T];
    r[1] = dsl[0];
    r[2] = dsl[1];
    r[3] = fe_scale * dsl[2];
    // live counts = stored - absorbed since the last compaction (cnt[14], cnt[15])
    r[4] = (double)(cnt[C_NE] - (cnt[C_WALL_E] - cnt[14]));
    r[5] = (double)(cnt[C_NI] - (cnt[C_WALL_I] - cnt[15]));
    r[6] = (double)cnt[C_WALL_E];
    r[7] = (double)cnt[C_WALL_I];
    r[8] = es_unord(cnt[C_PHIMIN]);
    r[9] = es_unord(cnt[C_PHIMAX]);
    r[10] = (double)cnt[C_COLL_E];
    r[11] = (double)cnt[C_ION_EV];
    r[12] = (double)cnt[C_SEE_EV];
    r[13] = surf_scale * dsl[3];
}

// sum of an array into dsl[slot] (grid-stride, block reduce)
extern "C" __global__ void sum_into(const double* __restrict__ a, const long long n, double* __restrict__ out)
{
    double v = 0.0;
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < n; k += (long long)gridDim.x * blockDim.x)
        v += a[k];
    es_block_sum_atomic(v, out);
}

// ---------------------------------------------------------------------------
// Blocking capacitors (self-bias, prompts/134). The circuit state lives on the
// device so that the step stays one CUDA graph without host round trips:
//   st[0:m] Q_N (node charge), st[m:2m] V_e, st[2m:3m] V_s (source),
//   st[3m:4m] dQ of this step, st[4m:5m] induced charge Q_e0, st[5m] = 1 once Q_N is set.
// Q_e0,j = sum_i W[j,i] (q_static + rho + q_surf)_i is the charge that the node charges
// induce on electrode j with all electrodes at 0 V (Green reciprocity: W = -2pi psi~_j on
// the uniform grid, the adjoint charge functional on the AMR composite grid), so
// V_e = (C + C_b)^-1 (Q_N - Q_e0 + C_b V_s) is known before the right-hand side is
// built and Poisson is solved once with it (exact even with fixed PCG iterations).
// ---------------------------------------------------------------------------
#define CAP_MAX 16

// Partial sums of the induced charge: grid (nb, m) blocks of 256 threads, part[j * nb + b]
// (summed in a fixed order by cap_solve, so the result is deterministic).
extern "C" __global__ void cap_induced(const double* __restrict__ W, const double* __restrict__ qs,
                                       const double* __restrict__ rho, const double* __restrict__ qsurf,
                                       const long long n, double* __restrict__ part)
{
    __shared__ double sh[256];
    const int j = blockIdx.y;
    const double* w = W + (long long)j * n;
    double s = 0.0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x)
        s += w[i] * (qs[i] + rho[i] + qsurf[i]);
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) part[j * gridDim.x + blockIdx.x] = sh[0];
}

// New electrode potentials, written into the group potentials vg (after eval_groups).
// par: (C + C_b)^-1 (m*m), C (m*m), C_b (m), initial bias (m).
// grp: m+1 offsets (absolute indices into grp) followed by the group numbers of each electrode.
extern "C" __global__ void cap_solve(double* __restrict__ st, double* __restrict__ vg,
                                     const double* __restrict__ par, const int* __restrict__ grp,
                                     const double* __restrict__ part, const int nb, const int m)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const double* ainv = par;
    const double* c = par + m * m;
    const double* cb = par + 2 * m * m;
    const double* bias = cb + m;
    double* q = st;
    double* v = st + m;
    double* vs = st + 2 * m;
    double* qind = st + 4 * m;
    for (int j = 0; j < m; ++j) {
        double s = 0.0;
        for (int b = 0; b < nb; ++b) s += part[j * nb + b];
        qind[j] = s;
        vs[j] = vg[grp[grp[j]]];
    }
    if (st[5 * m] == 0.0) {  // first solve: V_e = V_s + bias (circuit.BlockingCircuit.initial_charge)
        for (int j = 0; j < m; ++j) {
            double s = qind[j] + cb[j] * bias[j];
            for (int k = 0; k < m; ++k) s += c[j * m + k] * (vs[k] + bias[k]);
            q[j] = s;
        }
        st[5 * m] = 1.0;
    }
    double r[CAP_MAX];
    for (int j = 0; j < m; ++j) r[j] = q[j] - qind[j] + cb[j] * vs[j];
    for (int j = 0; j < m; ++j) {
        double s = 0.0;
        for (int k = 0; k < m; ++k) s += ainv[j * m + k] * r[k];
        v[j] = s;
        for (int g = grp[j]; g < grp[j + 1]; ++g) vg[grp[g]] = s;
    }
}

// End of the step: Q_N += dQ (absorbed and emitted charge counted by the boundary kernel),
// history row (V_e, dQ/dt) in the device ring buffer, dQ = 0.
extern "C" __global__ void cap_step(double* __restrict__ st, double* __restrict__ hist, const int n_rows,
                                    const double* __restrict__ prm, const int m)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long step = (long long)prm[P_STEP];
    double* row = hist + (step % n_rows) * 2 * m;
    const double dt = prm[P_DT];
    for (int j = 0; j < m; ++j) {
        const double dq = st[3 * m + j];
        st[j] += dq;
        row[j] = st[m + j];
        row[m + j] = dq / dt;
        st[3 * m + j] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// EEDF regions (axis-aligned boxes): weighted energy histograms of electrons,
// accumulated in shared memory per block. reg: per region (x0 x1 y0 y1 e_max);
// sums: per region (sum_w, sum_we, overflow_w). bins <= 1000 (schema limit).
// ---------------------------------------------------------------------------
extern "C" __global__ void eedf_hist(const double* __restrict__ x, const double* __restrict__ y,
                                     const double* __restrict__ vx, const double* __restrict__ vy,
                                     const double* __restrict__ vz, const double* __restrict__ w,
                                     const unsigned long long* __restrict__ cnt,
                                     const double* __restrict__ reg, const int n_reg, const int bins,
                                     double* __restrict__ hist, double* __restrict__ sums,
                                     const double* __restrict__ prm)
{
    extern __shared__ double sh[];  // n_reg * (bins + 3)
    if (prm[P_ACCUM] == 0.0) return;
    const int stride = bins + 3;
    for (int k = threadIdx.x; k < n_reg * stride; k += blockDim.x) sh[k] = 0.0;
    __syncthreads();
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long n = (long long)cnt[C_NE];
    if (p < n && w[p] != 0.0) {
        const double wp = w[p];
        const double e = 0.5 * 9.1093837015e-31 * (vx[p] * vx[p] + vy[p] * vy[p] + vz[p] * vz[p]) / 1.602176634e-19;
        for (int r = 0; r < n_reg; ++r) {
            const double* g = reg + 5 * r;
            if (x[p] >= g[0] && x[p] <= g[1] && y[p] >= g[2] && y[p] <= g[3]) {
                double* s = sh + r * stride;
                atomicAdd(s + bins, wp);
                atomicAdd(s + bins + 1, wp * e);
                if (e < g[4]) {
                    int b = (int)(e / g[4] * bins);
                    if (b > bins - 1) b = bins - 1;
                    atomicAdd(s + b, wp);
                } else {
                    atomicAdd(s + bins + 2, wp);
                }
            }
        }
    }
    __syncthreads();
    for (int k = threadIdx.x; k < n_reg * stride; k += blockDim.x) {
        const double v = sh[k];
        if (v != 0.0) {
            const int r = k / stride, c = k % stride;
            if (c < bins) atomicAdd(hist + r * bins + c, v);
            else atomicAdd(sums + 3 * r + (c - bins), v);
        }
    }
}

// ===========================================================================
// AMR composite-grid field kernels (prompts/122), compiled only with ES_AMR.
// Matrices are CSR (int32 indices) over an extended column space: columns
// below the base size refer to a node/unknown vector, the remaining columns to
// the Dirichlet group potentials vg.
// ===========================================================================
#ifdef ES_AMR
// b[r] = sum_k a[k] * (col < N ? qs + rho + qsurf : vg[col - N])   (b_c = P^T q + Coup_c V)
extern "C" __global__ void amr_rhs(double* __restrict__ b, const int* __restrict__ ip, const int* __restrict__ ix,
                                   const double* __restrict__ a, const double* __restrict__ qs,
                                   const double* __restrict__ rho, const double* __restrict__ qsurf,
                                   const double* __restrict__ vg, const int n_nodes, const int n_rows)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    double s = 0.0;
    for (int k = ip[r]; k < ip[r + 1]; ++k) {
        const int c = ix[k];
        s += a[k] * (c < n_nodes ? qs[c] + rho[c] + qsurf[c] : vg[c - n_nodes]);
    }
    b[r] = s;
}

// phi[r] = sum_k a[k] * (col < nC ? xc[col] : vg[col - nC])   (phi_all = P_node x_c + C_node V); min/max
extern "C" __global__ void amr_fill_phi(double* __restrict__ phi, const int* __restrict__ ip,
                                        const int* __restrict__ ix, const double* __restrict__ a,
                                        const double* __restrict__ xc, const double* __restrict__ vg,
                                        const int n_c, const int n_nodes, unsigned long long* __restrict__ cnt)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_nodes) return;
    double s = 0.0;
    for (int k = ip[r]; k < ip[r + 1]; ++k) {
        const int c = ix[k];
        s += a[k] * (c < n_c ? xc[c] : vg[c - n_c]);
    }
    phi[r] = s;
    const unsigned long long o = es_ord(s);
    atomicMin(cnt + C_PHIMIN, o);
    atomicMax(cnt + C_PHIMAX, o);
}

// Node field: ex = Gx [phi; vg], ey = Gy [phi; vg]  (es_sim/amr/fieldops.py)
extern "C" __global__ void amr_efield(double* __restrict__ ex, double* __restrict__ ey,
                                      const int* __restrict__ gxp, const int* __restrict__ gxi,
                                      const double* __restrict__ gxa, const int* __restrict__ gyp,
                                      const int* __restrict__ gyi, const double* __restrict__ gya,
                                      const double* __restrict__ phi, const double* __restrict__ vg,
                                      const int n_nodes)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_nodes) return;
    double sx = 0.0, sy = 0.0;
    for (int k = gxp[r]; k < gxp[r + 1]; ++k) {
        const int c = gxi[k];
        sx += gxa[k] * (c < n_nodes ? phi[c] : vg[c - n_nodes]);
    }
    for (int k = gyp[r]; k < gyp[r + 1]; ++k) {
        const int c = gyi[k];
        sy += gya[k] * (c < n_nodes ? phi[c] : vg[c - n_nodes]);
    }
    ex[r] = sx;
    ey[r] = sy;
}

// Discrete field energy over the composite edge list and Dirichlet couplings (x 1/2, 2 pi on the host)
extern "C" __global__ void amr_edge_energy(const double* __restrict__ phi, const int* __restrict__ eu,
                                           const int* __restrict__ ev, const double* __restrict__ eg,
                                           const long long ne, const int* __restrict__ cu,
                                           const int* __restrict__ cgrp, const double* __restrict__ cg,
                                           const long long nc, const double* __restrict__ vg,
                                           double* __restrict__ out)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    double e = 0.0;
    if (t < ne) {
        const double d = phi[eu[t]] - phi[ev[t]];
        e = eg[t] * d * d;
    } else if (t < ne + nc) {
        const long long u = t - ne;
        const double d = phi[cu[u]] - vg[cgrp[u]];
        e = cg[u] * d * d;
    }
    es_block_sum_atomic(e, out);
}

// Initial backward half kick of the in-plane velocity: v -= (q/m) E(x) dt/2
extern "C" __global__ void kick_half(const double* __restrict__ x, const double* __restrict__ y,
                                     double* __restrict__ vx, double* __restrict__ vy,
                                     const double* __restrict__ w, const unsigned long long* __restrict__ cnt,
                                     const int species, const double* __restrict__ ex,
                                     const double* __restrict__ ey, const double qm_half_dt ES_GRID_PARAMS)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long n = (long long)cnt[species == 0 ? C_NE : C_NI];
    if (p >= n || w[p] == 0.0) return;
    int nd[4];
    long long cell;
    double wx, wy;
    es_locate(x[p], y[p], nd, &wx, &wy, &cell ES_GRID_ARGS);
    const double w00 = (1.0 - wx) * (1.0 - wy), w10 = wx * (1.0 - wy);
    const double w01 = (1.0 - wx) * wy, w11 = wx * wy;
    vx[p] -= qm_half_dt * (w00 * ex[nd[0]] + w10 * ex[nd[1]] + w01 * ex[nd[2]] + w11 * ex[nd[3]]);
    vy[p] -= qm_half_dt * (w00 * ey[nd[0]] + w10 * ey[nd[1]] + w01 * ey[nd[2]] + w11 * ey[nd[3]]);
}
#endif

// ---------------------------------------------------------------------------
// Order-preserving compaction of live particles (w > 0), in three kernels:
// block-local exclusive scan of the live flags, a single-block scan of the
// block totals, and the scatter into the destination arrays.
// ---------------------------------------------------------------------------
#define ES_SCAN_BLOCK 1024

extern "C" __global__ void compact_scan(const double* __restrict__ w, const unsigned long long* __restrict__ cnt,
                                        const int species, int* __restrict__ local_off,
                                        unsigned char* __restrict__ flag, long long* __restrict__ block_sum,
                                        const long long cap)
{
    __shared__ int sh[ES_SCAN_BLOCK];
    const long long base = (long long)blockIdx.x * ES_SCAN_BLOCK;
    const long long p = base + threadIdx.x;
    const long long n = (long long)cnt[species == 0 ? C_NE : C_NI];
    const int f = (p < n && p < cap && w[p] != 0.0) ? 1 : 0;
    flag[p] = (unsigned char)f;
    sh[threadIdx.x] = f;
    __syncthreads();
    // Hillis-Steele inclusive scan
    for (int o = 1; o < ES_SCAN_BLOCK; o <<= 1) {
        const int v = threadIdx.x >= o ? sh[threadIdx.x - o] : 0;
        __syncthreads();
        sh[threadIdx.x] += v;
        __syncthreads();
    }
    local_off[p] = sh[threadIdx.x] - f;
    if (threadIdx.x == ES_SCAN_BLOCK - 1) block_sum[blockIdx.x] = sh[threadIdx.x];
}

extern "C" __global__ void compact_scan_blocks(long long* __restrict__ block_sum, const long long nb,
                                               unsigned long long* __restrict__ cnt, const int species)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    long long acc = 0;
    for (long long b = 0; b < nb; ++b) {
        const long long v = block_sum[b];
        block_sum[b] = acc;
        acc += v;
    }
    cnt[species == 0 ? C_NE : C_NI] = (unsigned long long)acc;
    // wall counter at this compaction (live count = stored - absorbed since then)
    cnt[species == 0 ? 14 : 15] = cnt[species == 0 ? C_WALL_E : C_WALL_I];
}

extern "C" __global__ void compact_scatter(const unsigned char* __restrict__ flag, const long long cap,
                                           const int* __restrict__ local_off,
                                           const long long* __restrict__ block_sum,
                                           const double* __restrict__ s0, const double* __restrict__ s1,
                                           const double* __restrict__ s2, const double* __restrict__ s3,
                                           const double* __restrict__ s4, const double* __restrict__ s5,
                                           double* __restrict__ d0, double* __restrict__ d1,
                                           double* __restrict__ d2, double* __restrict__ d3,
                                           double* __restrict__ d4, double* __restrict__ d5)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= cap || !flag[p]) return;
    const long long dst = block_sum[p / ES_SCAN_BLOCK] + local_off[p];
    d0[dst] = s0[p];
    d1[dst] = s1[p];
    d2[dst] = s2[p];
    d3[dst] = s3[p];
    d4[dst] = s4[p];
    d5[dst] = s5[p];
}
