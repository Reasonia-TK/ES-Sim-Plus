// v2 EB Poisson: matrix-free 5-point node-centred finite-volume operator (prompts/119).
//
// The same formulas are implemented for the CPU in es_sim/field/_cpu_kernels.py -
// keep both in sync.
//
// Layout (row-major, x fastest):
//   node arrays   (ny+1) x (nx+1), k = j*(nx+1) + i
//   cx            (ny+1) x nx      cx[j*nx + i]      = conductance of edge (i,j)-(i+1,j)
//   cy            ny x (nx+1)      cy[j*(nx+1) + i]  = conductance of edge (i,j)-(i,j+1)
//   mask          0 = unknown, 1 = fixed (Dirichlet), 2 = periodic slave
// Periodic in x (px): node nx is a slave of node 0, the wrap edge is cx[j*nx + nx-1].
// Periodic in y (py): row ny is a slave of row 0, the wrap edge is cy[(ny-1)*(nx+1) + i].
// Conductances couple unknown nodes only (Dirichlet couplings live in diag / rhs).

__device__ __forceinline__ double offdiag_sum(const double* __restrict__ x,
                                              const double* __restrict__ cx,
                                              const double* __restrict__ cy,
                                              int i, int j, int nx, int ny, int px, int py)
{
    const int sx = nx + 1;
    double s = 0.0;
    if (i > 0) s += cx[j * nx + i - 1] * x[j * sx + i - 1];
    else if (px) s += cx[j * nx + nx - 1] * x[j * sx + nx - 1];
    if (i < nx) {
        const int ie = (px && i == nx - 1) ? 0 : i + 1;
        s += cx[j * nx + i] * x[j * sx + ie];
    }
    if (j > 0) s += cy[(j - 1) * sx + i] * x[(j - 1) * sx + i];
    else if (py) s += cy[(ny - 1) * sx + i] * x[(ny - 1) * sx + i];
    if (j < ny) {
        const int jn = (py && j == ny - 1) ? 0 : j + 1;
        s += cy[j * sx + i] * x[jn * sx + i];
    }
    return s;
}

// One red-black Gauss-Seidel colour sweep: nodes with (i + j) & 1 == color.
extern "C" __global__ void rbgs(double* __restrict__ x, const double* __restrict__ b,
                                const double* __restrict__ cx, const double* __restrict__ cy,
                                const double* __restrict__ diag, const unsigned char* __restrict__ mask,
                                const int nx, const int ny, const int px, const int py, const int color)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > nx || j > ny) return;
    if (((i + j) & 1) != color) return;
    const int k = j * (nx + 1) + i;
    if (mask[k] != 0) return;
    x[k] = (b[k] + offdiag_sum(x, cx, cy, i, j, nx, ny, px, py)) / diag[k];
}

// r = b - A x on unknown nodes, 0 elsewhere.
extern "C" __global__ void residual(double* __restrict__ r, const double* __restrict__ x,
                                    const double* __restrict__ b,
                                    const double* __restrict__ cx, const double* __restrict__ cy,
                                    const double* __restrict__ diag, const unsigned char* __restrict__ mask,
                                    const int nx, const int ny, const int px, const int py)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > nx || j > ny) return;
    const int k = j * (nx + 1) + i;
    if (mask[k] != 0) { r[k] = 0.0; return; }
    r[k] = b[k] - (diag[k] * x[k] - offdiag_sum(x, cx, cy, i, j, nx, ny, px, py));
}

// y = A x on unknown nodes, 0 elsewhere.
extern "C" __global__ void apply_op(double* __restrict__ y, const double* __restrict__ x,
                                    const double* __restrict__ cx, const double* __restrict__ cy,
                                    const double* __restrict__ diag, const unsigned char* __restrict__ mask,
                                    const int nx, const int ny, const int px, const int py)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > nx || j > ny) return;
    const int k = j * (nx + 1) + i;
    if (mask[k] != 0) { y[k] = 0.0; return; }
    y[k] = diag[k] * x[k] - offdiag_sum(x, cx, cy, i, j, nx, ny, px, py);
}

// Restriction R = P^T (full weighting without scaling: FV residuals are integrated quantities).
// Coarsening factors fx, fy are 1 or 2 (semi-coarsening supported).
extern "C" __global__ void restrict_fw(double* __restrict__ bc, const double* __restrict__ rf,
                                       const unsigned char* __restrict__ mask_c,
                                       const int nxc, const int nyc, const int nxf, const int nyf,
                                       const int fx, const int fy, const int px, const int py)
{
    const int I = blockIdx.x * blockDim.x + threadIdx.x;
    const int J = blockIdx.y * blockDim.y + threadIdx.y;
    if (I > nxc || J > nyc) return;
    const int kc = J * (nxc + 1) + I;
    if (mask_c[kc] != 0) { bc[kc] = 0.0; return; }
    const int i0 = fx * I, j0 = fy * J;
    double s = 0.0;
    for (int b = -(fy - 1); b <= fy - 1; ++b) {
        int j = j0 + b;
        if (j < 0) { if (!py) continue; j += nyf; }
        else if (j > nyf) continue;
        else if (py && j == nyf) j = 0;
        const double wb = (b == 0) ? 1.0 : 0.5;
        for (int a = -(fx - 1); a <= fx - 1; ++a) {
            int i = i0 + a;
            if (i < 0) { if (!px) continue; i += nxf; }
            else if (i > nxf) continue;
            else if (px && i == nxf) i = 0;
            const double wa = (a == 0) ? 1.0 : 0.5;
            s += wa * wb * rf[j * (nxf + 1) + i];
        }
    }
    bc[kc] = s;
}

// ---------------------------------------------------------------------------
// Sync-free PCG helpers (all scalars live in device memory so a whole solve
// can be enqueued - or captured in a CUDA graph - without host round trips).
// ---------------------------------------------------------------------------

__device__ __forceinline__ double es_warp_sum_p(double v)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// out[0] += sum(a * b). Every thread must reach the reduction.
extern "C" __global__ void dot_acc(const double* __restrict__ a, const double* __restrict__ b,
                                   const long long n, double* __restrict__ out)
{
    __shared__ double sh[32];
    double v = 0.0;
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < n;
         k += (long long)gridDim.x * blockDim.x)
        v += a[k] * b[k];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = es_warp_sum_p(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < (int)((blockDim.x + 31) >> 5)) ? sh[lane] : 0.0;
        v = es_warp_sum_p(v);
        if (lane == 0) atomicAdd(out, v);
    }
}

// x += alpha p ; r -= alpha Ap  with alpha = s[i_num] / s[i_den] (0 if the denominator is 0)
extern "C" __global__ void pcg_xr(double* __restrict__ x, double* __restrict__ r,
                                  const double* __restrict__ p, const double* __restrict__ ap,
                                  const double* __restrict__ s, const int i_num, const int i_den,
                                  const long long n)
{
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const double den = s[i_den];
    const double alpha = den != 0.0 ? s[i_num] / den : 0.0;
    x[k] += alpha * p[k];
    r[k] -= alpha * ap[k];
}

// p = z + beta p  with beta = s[i_num] / s[i_den] (0 if the denominator is 0)
extern "C" __global__ void pcg_p(double* __restrict__ p, const double* __restrict__ z,
                                 const double* __restrict__ s, const int i_num, const int i_den,
                                 const long long n)
{
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const double den = s[i_den];
    const double beta = den != 0.0 ? s[i_num] / den : 0.0;
    p[k] = z[k] + beta * p[k];
}

// dst = src (element copy)
extern "C" __global__ void copy_vec(double* __restrict__ dst, const double* __restrict__ src,
                                    const long long n)
{
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) dst[k] = src[k];
}

// s[dst] = s[src] (single scalar copy, one thread)
extern "C" __global__ void copy_scalar(double* __restrict__ s, const int dst, const int src)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) s[dst] = s[src];
}

// Dense direct solve on a subset of nodes: x[idx[r]] = sum_c A[r, c] * b[idx[c]].
// Used for the coarsest multigrid level and for whole small problems.
extern "C" __global__ void dense_gemv_idx(const double* __restrict__ a, const long long* __restrict__ idx,
                                          const double* __restrict__ b, double* __restrict__ x,
                                          const int m)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    const double* ar = a + (long long)row * m;
    double s = 0.0;
    for (int c = 0; c < m; ++c) s += ar[c] * b[idx[c]];
    x[idx[row]] = s;
}

// Bilinear prolongation of the coarse correction, added to unknown fine nodes.
extern "C" __global__ void prolong_add(double* __restrict__ xf, const double* __restrict__ ec,
                                       const unsigned char* __restrict__ mask_f,
                                       const int nxf, const int nyf, const int nxc, const int nyc,
                                       const int fx, const int fy, const int px, const int py)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > nxf || j > nyf) return;
    const int kf = j * (nxf + 1) + i;
    if (mask_f[kf] != 0) return;
    const int I0 = i / fx, di = i - I0 * fx;
    const int J0 = j / fy, dj = j - J0 * fy;
    int I1 = I0 + di, J1 = J0 + dj;
    if (px && I1 == nxc) I1 = 0;
    if (py && J1 == nyc) J1 = 0;
    const double wx1 = 0.5 * di, wx0 = 1.0 - wx1;
    const double wy1 = 0.5 * dj, wy0 = 1.0 - wy1;
    const int sc = nxc + 1;
    const double e = wy0 * (wx0 * ec[J0 * sc + I0] + wx1 * ec[J0 * sc + I1])
                   + wy1 * (wx0 * ec[J1 * sc + I0] + wx1 * ec[J1 * sc + I1]);
    xf[kf] += e;
}
