// fluid.cu -- v2 fluid 2D on the Cartesian EB graph: GPU kernels (prompts/126).
//
// Same discretization as es_sim/fluid2d.py (EAFE / Scharfetter-Gummel edge fluxes, backward Euler
// per species, Jacobi-preconditioned BiCGSTAB) on the transport graph of es_sim/gfluid/geometry.py.
//
// Matrix-free form of the implicit matrix M (one row per active node k):
//   (M x)_k = dg[k] x[k] - sum_m coef_m x[other_m],   m over the edges incident to k,
// where an edge e = (i, j) with flux F = a_e n_i - b_e n_j enters row i with -b_e (column j) and
// row j with -a_e (column i). The incidence entry code is 2 e + role (role 0: k is i, 1: k is j).
//
// Determinism: every kernel that reduces is launched with exactly FL_NB blocks of FL_NT threads and
// walks its range with the same grid-stride loop; each block writes one partial per quantity
// (part[slot * FL_NB + block]) and partials are summed in a fixed tree (by the next kernel, or on the
// host). No atomics are used for floating-point sums.

#define FL_NB 256
#define FL_NT 256
#define FL_QE 1.602176634e-19
#define FL_ME 9.1093837015e-31
#define FL_PI 3.141592653589793
#define FL_FLOOR_N 1.0e6

// partial slots of the BiCGSTAB work buffer
#define P_RHO 0
#define P_R0V 1
#define P_SS 2
#define P_TS 3
#define P_TT 4
#define P_RR 5
#define P_BB 6
// scalar slots
#define S_RHO 0
#define S_RHO_OLD 1
#define S_ALPHA 2
#define S_OMEGA 3
#define S_TOL2 4

#define FL_LOOP(k, n) for (long long k = (long long)blockIdx.x * FL_NT + threadIdx.x; k < (n); \
                           k += (long long)FL_NB * FL_NT)

// Bernoulli function B(z) = z / (e^z - 1) with the same branches as fluid1d._bernoulli
__device__ __forceinline__ double fl_bern(double z)
{
    if (fabs(z) < 1.0e-4) return 1.0 - 0.5 * z + z * z / 12.0;
    if (z > 500.0) return z * exp(-z);
    if (z < -500.0) return -z;
    return z / expm1(z);
}

// block reduction (fixed tree) of v; thread 0 stores the block value to part[slot * FL_NB + block].
// mode 0: sum, 1: max, 2: min
__device__ void fl_store(double v, double* sh, double* part, int slot, int mode)
{
    int t = threadIdx.x;
    sh[t] = v;
    __syncthreads();
    for (int w = FL_NT / 2; w > 0; w >>= 1) {
        if (t < w) {
            double o = sh[t + w];
            sh[t] = mode == 0 ? sh[t] + o : (mode == 1 ? fmax(sh[t], o) : fmin(sh[t], o));
        }
        __syncthreads();
    }
    if (t == 0) part[slot * FL_NB + blockIdx.x] = sh[0];
    __syncthreads();
}

// sum of the FL_NB partials of a slot (every block computes the same value in the same order)
__device__ double fl_total(const double* part, int slot, double* sh)
{
    int t = threadIdx.x;
    sh[t] = part[slot * FL_NB + t];
    __syncthreads();
    for (int w = FL_NT / 2; w > 0; w >>= 1) {
        if (t < w) sh[t] += sh[t + w];
        __syncthreads();
    }
    double r = sh[0];
    __syncthreads();
    return r;
}

// one row of the implicit matrix times x
__device__ __forceinline__ double fl_row(int k, const double* x, const double* dg, const double* ca,
                                         const double* cb, const int* rp, const int* ic, const int* io)
{
    double y = dg[k] * x[k];
    for (int m = rp[k]; m < rp[k + 1]; ++m) {
        int c = ic[m];
        int e = c >> 1;
        y -= ((c & 1) ? ca[e] : cb[e]) * x[io[m]];
    }
    return y;
}

// ---- coefficient tables (log-log interpolation, same rules as fluid_coeffs.interp_loglog) ----

struct FlLL {
    int idx;
    double flin;
    double flog;
};

__device__ FlLL fl_ll_setup(const double* g, int ng, double x)
{
    FlLL s;
    double xc = fmin(fmax(x, g[0]), g[ng - 1]);
    int lo = 0, hi = ng;  // first index with g[i] > xc  (searchsorted side="right")
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (g[mid] <= xc) lo = mid + 1; else hi = mid;
    }
    int idx = lo - 1;
    if (idx > ng - 2) idx = ng - 2;
    if (idx < 0) idx = 0;
    double t0 = g[idx], t1 = g[idx + 1];
    s.idx = idx;
    s.flin = (t1 > t0) ? (xc - t0) / (t1 - t0) : 0.0;
    s.flog = (t1 > t0) ? (log(xc) - log(t0)) / (log(t1) - log(t0)) : 0.0;
    return s;
}

__device__ __forceinline__ double fl_ll_eval(const double* tab, FlLL s)
{
    double y0 = tab[s.idx], y1 = tab[s.idx + 1];
    if (s.flin == 0.0) return y0;
    if (y0 > 0.0 && y1 > 0.0) return exp(log(y0) + s.flog * (log(y1) - log(y0)));
    return y0 + s.flin * (y1 - y0);
}

// Te and the electron coefficients of every active node.
// model 0 (Maxwell): key Te, tables mobility_n, k_ion, k_exc, nu_m/n_g; e_ion/e_exc are constants.
// model 1 (boltzpm): key 1.5 Te, tables mobility_n, k_ion, k_exc, e_ion, e_exc; nu_m = e / (m mu N).
// flag[0] = 1 when some key was clamped to the table range.
extern "C" __global__ void fl_coeffs(const double* ne, const double* w, int n, const double* grid, int ng,
                                     const double* t0, const double* t1, const double* t2, const double* t3,
                                     const double* t4, int model, double n_g, double e_ion_c, double e_exc_c,
                                     double* te, double* mue, double* kion, double* kexc, double* num,
                                     double* eion, double* eexc, double* flag)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    double tk = fmax((2.0 / 3.0) * w[k] / fmax(ne[k], FL_FLOOR_N), 1.0e-6);
    double key = model == 0 ? tk : 1.5 * tk;
    if (key < grid[0] || key > grid[ng - 1]) flag[0] = 1.0;
    FlLL s = fl_ll_setup(grid, ng, key);
    double mob = fl_ll_eval(t0, s);
    te[k] = tk;
    mue[k] = mob / n_g;
    kion[k] = fl_ll_eval(t1, s);
    kexc[k] = fl_ll_eval(t2, s);
    if (model == 0) {
        num[k] = fl_ll_eval(t3, s);
        eion[k] = e_ion_c;
        eexc[k] = e_exc_c;
    } else {
        num[k] = FL_QE / (FL_ME * mob);
        eion[k] = fl_ll_eval(t3, s);
        eexc[k] = fl_ll_eval(t4, s);
    }
}

// Scharfetter-Gummel coefficients of every edge (ions: z = dphi/T_i * mu_i(E)/mu_L, electrons: face
// averages of mu_e and Te, z = -dphi/Te_face). dphi = phi_i - phi_j.
extern "C" __global__ void fl_edges(const double* phi, const int* gi, const int* gj, const int* li,
                                    const int* lj, const double* wij, const double* elen, const double* te,
                                    const double* mue, int ne_, double t_i, double d_i, int frost, double n_g,
                                    double c_td, double* dphi, double* ai, double* bi, double* ae, double* be)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= ne_) return;
    double dp = phi[gi[e]] - phi[gj[e]];
    double ratio = 1.0;
    if (frost) {
        double en_td = fabs(dp / elen[e]) / n_g / 1.0e-21;
        ratio = 1.0 / sqrt(1.0 + en_td / c_td);
    }
    double zi = (dp / t_i) * ratio;
    double w = wij[e];
    ai[e] = w * d_i * fl_bern(-zi);
    bi[e] = w * d_i * fl_bern(zi);
    int a = li[e], b = lj[e];
    double muf = 0.5 * (mue[a] + mue[b]);
    double tf = 0.5 * (te[a] + te[b]);
    double de = muf * tf;
    double ze = -dp / tf;
    ae[e] = w * de * fl_bern(-ze);
    be[e] = w * de * fl_bern(ze);
    dphi[e] = dp;
}

// reaction source and energy loss per active node, and the electron wall coefficient v_th,e / 4
extern "C" __global__ void fl_nodes(const double* ne, const double* te, const double* kion, const double* kexc,
                                    const double* num, const double* eion, const double* eexc, int n, double n_g,
                                    double tg, double mass_ratio, int src_on, double* sion, double* loss,
                                    double* ce)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    double nk = ne[k];
    double s = 0.0, l_ion = 0.0, l_exc = 0.0;
    if (src_on) {
        s = kion[k] * n_g * nk;
        l_ion = eion[k] * s;
        l_exc = eexc[k] * kexc[k] * n_g * nk;
    }
    double l_el = 3.0 * mass_ratio * num[k] * n_g * (te[k] - tg) * nk;
    sion[k] = s;
    loss[k] = l_ion + l_exc + l_el;
    ce[k] = 0.25 * sqrt(8.0 * te[k] * FL_QE / (FL_PI * FL_ME));
}

// ion wall coefficient of every wall piece (pieces sorted by node): E.n = (phi(A) - phi(W)) / L with
// bilinear phi, phi(W) = electrode voltage on conductors; c_i = max(mu_i(E) E.n, 0) + v_th,i / 4
extern "C" __global__ void fl_wall_ci(const double* phi, const int* aidx, const double* awt, const int* widx,
                                      const double* wwt, const int* grp, const double* wlen, const double* vgrp,
                                      int nw, int frost, double mu_i, double n_g, double c_td, double vthi4,
                                      int reflective, double* ci)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nw) return;
    if (reflective) { ci[c] = 0.0; return; }
    double pa = 0.0, pw = 0.0;
    for (int q = 0; q < 4; ++q) {
        pa += awt[4 * c + q] * phi[aidx[4 * c + q]];
        pw += wwt[4 * c + q] * phi[widx[4 * c + q]];
    }
    if (grp[c] >= 0) pw = vgrp[grp[c]];
    double en = (pa - pw) / wlen[c];
    double mu = mu_i;
    if (frost) mu = mu_i / sqrt(1.0 + (fabs(en) / n_g / 1.0e-21) / c_td);
    ci[c] = fmax(mu * en, 0.0) + vthi4;
}

// wall conductances per active node (pieces of node k are [wptr[k], wptr[k + 1]))
extern "C" __global__ void fl_wall_diag(const double* ci, const double* area, const int* wptr, const double* ce,
                                        const double* asum, int n, int reflective, double* wdi, double* wde)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    double s = 0.0;
    for (int c = wptr[k]; c < wptr[k + 1]; ++c) s += area[c] * ci[c];
    wdi[k] = s;
    wde[k] = reflective ? 0.0 : ce[k] * asum[k];
}

// diagonal of the implicit matrix (V/dt + wall + outgoing edge coefficients) and its inverse
extern "C" __global__ void fl_diag(const double* vol, double dt, const double* wd, const double* ca,
                                   const double* cb, const int* rp, const int* ic, int n, double* dg,
                                   double* dinv)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    double d = vol[k] / dt + wd[k];
    for (int m = rp[k]; m < rp[k + 1]; ++m) {
        int c = ic[m];
        int e = c >> 1;
        d += (c & 1) ? cb[e] : ca[e];
    }
    dg[k] = d;
    dinv[k] = d != 0.0 ? 1.0 / d : 1.0;
}

// right-hand side V/dt n_old + sgn * src V + extra
extern "C" __global__ void fl_rhs(const double* vol, double dt, const double* nold, const double* src, double sgn,
                                  const double* extra, int n, double* rhs)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    rhs[k] = vol[k] / dt * nold[k] + (sgn * src[k]) * vol[k] + extra[k];
}

extern "C" __global__ void fl_floor(double* x, double lo, int n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) x[k] = fmax(x[k], lo);
}

// ion wall flux per node and the secondary electron source
extern "C" __global__ void fl_see(const double* wdi, const double* ni, const double* gamma, int n, int reflective,
                                  double* gwi, double* see)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    double g = reflective ? 0.0 : wdi[k] * ni[k];
    gwi[k] = g;
    see[k] = gamma[k] * g;
}

// net electron wall flux and the energy wall conductance (5/3) gw_e / n_e
extern "C" __global__ void fl_gwe(const double* wde, const double* ne, const double* see, int n, int reflective,
                                  double* gwe, double* wdw)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    double g = reflective ? 0.0 : wde[k] * ne[k] - see[k];
    gwe[k] = g;
    wdw[k] = (5.0 / 3.0) * g / fmax(ne[k], FL_FLOOR_N);
}

// Joule heating per node: half of the edge power -F_e dphi_e to each end
extern "C" __global__ void fl_joule(const double* ae, const double* be, const double* ne, const double* dphi,
                                    const int* ei, const int* ej, const int* rp, const int* ic, int n,
                                    double* joule)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    double s = 0.0;
    for (int m = rp[k]; m < rp[k + 1]; ++m) {
        int e = ic[m] >> 1;
        double f = ae[e] * ne[ei[e]] - be[e] * ne[ej[e]];
        s += 0.5 * (-f * dphi[e]);
    }
    joule[k] = s;
}

extern "C" __global__ void fl_scale2(const double* a, const double* b, double f, int n, double* fa, double* fb)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    fa[e] = f * a[e];
    fb[e] = f * b[e];
}

extern "C" __global__ void fl_energy_off(const double* ne, double f, int n, double* w)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) w[k] = f * ne[k];
}

// running wall losses and generation per block: acc[slot * FL_NB + block] += dt * block sum
// (slots 0: ions, 1: electrons, 2: ionization V). Launch with FL_NB x FL_NT.
extern "C" __global__ void fl_wallgen(const double* gwi, const double* gwe, const double* sion, const double* vol,
                                      double dt, int n, double* acc)
{
    __shared__ double sh[FL_NT];
    double a = 0.0, b = 0.0, c = 0.0;
    FL_LOOP(k, n)
    {
        a += gwi[k];
        b += gwe[k];
        c += sion[k] * vol[k];
    }
    double v3[3] = {a, b, c};
    for (int s = 0; s < 3; ++s) {
        int t = threadIdx.x;
        sh[t] = v3[s];
        __syncthreads();
        for (int w = FL_NT / 2; w > 0; w >>= 1) {
            if (t < w) sh[t] += sh[t + w];
            __syncthreads();
        }
        if (t == 0) acc[s * FL_NB + blockIdx.x] += dt * sh[0];
        __syncthreads();
    }
}

// ---- Poisson ---------------------------------------------------------------------------------------

// b = q_static + e * (charge map @ (n_i - n_e)) + coupling @ V  (EB Poisson rhs, 2 pi-free);
// 0 on nodes that are not unknowns (mask != 0), as GMGSolver.solve does
extern "C" __global__ void fl_poisson_rhs(const double* qs, const int* cp, const int* cc, const double* cv,
                                          const double* ni, const double* ne, const int* gp, const int* gc,
                                          const double* gv, const double* vgrp, const unsigned char* mask,
                                          int nn, double* b)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nn) return;
    if (mask[k] != 0) { b[k] = 0.0; return; }
    double q = 0.0;
    for (int m = cp[k]; m < cp[k + 1]; ++m) q += cv[m] * (ni[cc[m]] - ne[cc[m]]);
    double r = qs[k] + FL_QE * q;
    double c = 0.0;
    for (int m = gp[k]; m < gp[k + 1]; ++m) c += gv[m] * vgrp[gc[m]];
    b[k] = r + c;
}

extern "C" __global__ void fl_fixed(double* phi, const int* fidx, const int* fgrp, const double* vgrp, int nf)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nf) phi[fidx[k]] = vgrp[fgrp[k]];
}

// ---- Jacobi-preconditioned BiCGSTAB (same recurrence as _numba_kernels.bicgstab) --------------------
// All five kernels are launched with FL_NB x FL_NT.

#define FL_MAT const double *dg, const double *ca, const double *cb, const int *rp, const int *ic, const int *io

// r = b - A x, r0 = r; partials: rr, bb, rho = (r0, r)
extern "C" __global__ void bcg_init(const double* x, const double* b, FL_MAT, int n, double* r, double* r0,
                                    double* part)
{
    __shared__ double sh[FL_NT];
    double rr = 0.0, bb = 0.0;
    FL_LOOP(k, n)
    {
        double rk = b[k] - fl_row((int)k, x, dg, ca, cb, rp, ic, io);
        r[k] = rk;
        r0[k] = rk;
        rr += rk * rk;
        bb += b[k] * b[k];
    }
    fl_store(rr, sh, part, P_RR, 0);
    fl_store(bb, sh, part, P_BB, 0);
    fl_store(rr, sh, part, P_RHO, 0);
}

// rho = (r0, r); p = r (first) or r + beta (p - omega v); ph = D^-1 p.  Stores S[rho].
extern "C" __global__ void bcg_p(double* p, double* ph, const double* r, const double* v, const double* dinv,
                                 double* S, const double* part, int n, int first)
{
    __shared__ double sh[FL_NT];
    double rho = fl_total(part, P_RHO, sh);
    double beta = 0.0, om = S[S_OMEGA];
    if (!first) {
        double ro = S[S_RHO_OLD];
        beta = (ro != 0.0 && om != 0.0) ? (rho / ro) * (S[S_ALPHA] / om) : 0.0;
    }
    FL_LOOP(k, n)
    {
        double pk = first ? r[k] : r[k] + beta * (p[k] - om * v[k]);
        p[k] = pk;
        ph[k] = dinv[k] * pk;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) S[S_RHO] = rho;
}

// v = A ph; partial (r0, v)
extern "C" __global__ void bcg_v(const double* ph, const double* r0, FL_MAT, int n, double* v, double* part)
{
    __shared__ double sh[FL_NT];
    double d = 0.0;
    FL_LOOP(k, n)
    {
        double vk = fl_row((int)k, ph, dg, ca, cb, rp, ic, io);
        v[k] = vk;
        d += r0[k] * vk;
    }
    fl_store(d, sh, part, P_R0V, 0);
}

// alpha = rho / (r0, v); s = r - alpha v; sh = D^-1 s; partial (s, s).  Stores S[alpha].
extern "C" __global__ void bcg_s(const double* r, const double* v, const double* dinv, double* S, double* part,
                                 int n, double* s, double* shat)
{
    __shared__ double sh[FL_NT];
    double r0v = fl_total(part, P_R0V, sh);
    double alpha = r0v != 0.0 ? S[S_RHO] / r0v : 0.0;
    double ss = 0.0;
    FL_LOOP(k, n)
    {
        double sk = r[k] - alpha * v[k];
        s[k] = sk;
        shat[k] = dinv[k] * sk;
        ss += sk * sk;
    }
    fl_store(ss, sh, part, P_SS, 0);
    if (blockIdx.x == 0 && threadIdx.x == 0) S[S_ALPHA] = alpha;
}

// t = A sh; partials (t, s), (t, t)
extern "C" __global__ void bcg_t(const double* shat, const double* s, FL_MAT, int n, double* t, double* part)
{
    __shared__ double sh[FL_NT];
    double ts = 0.0, tt = 0.0;
    FL_LOOP(k, n)
    {
        double tk = fl_row((int)k, shat, dg, ca, cb, rp, ic, io);
        t[k] = tk;
        ts += tk * s[k];
        tt += tk * tk;
    }
    fl_store(ts, sh, part, P_TS, 0);
    fl_store(tt, sh, part, P_TT, 0);
}

// omega = (t, s) / (t, t) (0 when |s| already meets the tolerance); x += alpha ph + omega sh;
// r = s - omega t; partials (r, r) and the next rho = (r0, r).  Stores S[omega], S[rho_old] = S[rho].
extern "C" __global__ void bcg_x(double* x, double* r, const double* s, const double* t, const double* ph,
                                 const double* shat, const double* r0, double* S, double* part, int n)
{
    __shared__ double sh[FL_NT];
    double ts = fl_total(part, P_TS, sh);
    double tt = fl_total(part, P_TT, sh);
    double ss = fl_total(part, P_SS, sh);
    double alpha = S[S_ALPHA];
    double om = (ss <= S[S_TOL2]) ? 0.0 : (tt > 0.0 ? ts / tt : 0.0);
    double rr = 0.0, rho = 0.0;
    FL_LOOP(k, n)
    {
        x[k] += alpha * ph[k] + om * shat[k];
        double rk = s[k] - om * t[k];
        r[k] = rk;
        rr += rk * rk;
        rho += r0[k] * rk;
    }
    fl_store(rr, sh, part, P_RR, 0);
    fl_store(rho, sh, part, P_RHO, 0);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        S[S_OMEGA] = om;
        S[S_RHO_OLD] = S[S_RHO];
    }
}

// ---- per-step statistics, sub-step control and time averages --------------------------------------

// partial slots of out (FL_NB each): 0 sum n_e V, 1 sum n_i V, 2 non-finite count, 3 max n_e mu_e,
// 4 max mu_e Te, 5 min w / Joule heating (heating > 0 only). Launch with FL_NB x FL_NT.
extern "C" __global__ void fl_stats(const double* ne, const double* ni, const double* w, const double* vol,
                                    const double* mue, const double* te, const double* joule, int n,
                                    const double* phi, int nn, double* out)
{
    __shared__ double sh[FL_NT];
    double se = 0.0, si = 0.0, bad = 0.0, m1 = 0.0, m2 = 0.0, m3 = 1.0e308;
    FL_LOOP(k, n)
    {
        double a = ne[k], b = ni[k], c = w[k];
        se += a * vol[k];
        si += b * vol[k];
        if (!isfinite(a) || !isfinite(b) || !isfinite(c)) bad += 1.0;
        m1 = fmax(m1, a * mue[k]);
        m2 = fmax(m2, mue[k] * te[k]);
        double h = joule[k];
        if (h > 0.0) m3 = fmin(m3, c / fmax(h, 1.0e-300));
    }
    FL_LOOP(k, nn)
    {
        if (!isfinite(phi[k])) bad += 1.0;
    }
    fl_store(se, sh, out, 0, 0);
    fl_store(si, sh, out, 1, 0);
    fl_store(bad, sh, out, 2, 0);
    fl_store(m1, sh, out, 3, 1);
    fl_store(m2, sh, out, 4, 1);
    fl_store(m3, sh, out, 5, 2);
}

// time averages of node quantities (phi over all nn nodes, the others over the n active nodes)
extern "C" __global__ void fl_accum(const double* phi, int nn, const double* ne, const double* ni,
                                    const double* te, const double* kion, int n, double n_g, int src_on,
                                    double* aphi, double* ane, double* ani, double* ate, double* aion)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nn) aphi[k] += phi[k];
    if (k < n) {
        ane[k] += ne[k];
        ani[k] += ni[k];
        ate[k] += te[k];
        aion[k] += src_on ? kion[k] * n_g * ne[k] : 0.0;
    }
}

// E = -grad phi of every display triangle, added to acc (ex, ey)
extern "C" __global__ void fl_tri_e(const double* phi, const int* tri, const double* bb, const double* cc,
                                    const double* det, int nt, double* acc)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt) return;
    double sx = 0.0, sy = 0.0;
    for (int q = 0; q < 3; ++q) {
        double p = phi[tri[3 * t + q]];
        sx += p * bb[3 * t + q];
        sy += p * cc[3 * t + q];
    }
    acc[2 * t] += -sx / det[t];
    acc[2 * t + 1] += -sy / det[t];
}
