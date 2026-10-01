// Dense linear algebra without cuBLAS / cuSOLVER (prompts/133 P8a).
//
// The packaged app ships only NVRTC from the CUDA toolkit, so the few dense operations the solvers
// need are written here. Everything is float64, row-major, n x n.
//
// In-place blocked Gauss-Jordan inversion of a symmetric positive definite matrix, without pivoting
// (the pivots of an SPD matrix are the positive diagonals of its Schur complements). The pivots are
// written to piv[] so that the host can check them and fall back to LAPACK when they are not positive.
// One block step J = [j0, j0 + w), w <= GJ_B:
//   P = inv(A[J, J])         (gj_block_inv, one thread block)
//   R = P A[J, :]            (gj_row_panel)
//   C = A[:, J]              (gj_col_copy, the old column panel)
//   A[i, c] -= C[i, :] R[:, c]    for i, c not in J
//   A[i, J]  = -C[i, :] P         for i not in J
//   A[J, c]  = R[:, c]            for c not in J
//   A[J, J]  = P                  (gj_update, every element written once from the buffers)

#define GJ_B 32

extern "C" __global__ void gj_block_inv(const double* __restrict__ a, double* __restrict__ p,
                                        double* __restrict__ piv, const int n, const int j0, const int w)
{
    __shared__ double s[GJ_B][GJ_B + 1];
    const int c = threadIdx.x;
    const int r = threadIdx.y;
    const bool inside = (r < w && c < w);
    if (inside) s[r][c] = a[(long long)(j0 + r) * n + (j0 + c)];
    for (int k = 0; k < w; ++k) {
        __syncthreads();
        const double pk = s[k][k];
        double v = 0.0;
        if (inside) {
            const double rk = s[r][k];
            const double kc = s[k][c];
            if (r == k && c == k) v = 1.0 / pk;
            else if (r == k) v = kc / pk;
            else if (c == k) v = -rk / pk;
            else v = s[r][c] - rk * kc / pk;
        }
        if (r == 0 && c == 0) piv[j0 + k] = pk;
        __syncthreads();
        if (inside) s[r][c] = v;
    }
    __syncthreads();
    if (inside) p[r * GJ_B + c] = s[r][c];
}

// R[k, c] = sum_t P[k, t] A[j0 + t, c]   (k < w, every column c; grid.y = w)
extern "C" __global__ void gj_row_panel(const double* __restrict__ a, const double* __restrict__ p,
                                        double* __restrict__ rp, const int n, const int j0, const int w)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y;
    if (c >= n || k >= w) return;
    double s = 0.0;
    for (int t = 0; t < w; ++t) s += p[k * GJ_B + t] * a[(long long)(j0 + t) * n + c];
    rp[(long long)k * n + c] = s;
}

// C[i, t] = A[i, j0 + t]   (every row i, t < w)
extern "C" __global__ void gj_col_copy(const double* __restrict__ a, double* __restrict__ cb,
                                       const int n, const int j0, const int w)
{
    const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= (long long)n * w) return;
    const int i = (int)(q / w);
    const int t = (int)(q - (long long)i * w);
    cb[(long long)i * GJ_B + t] = a[(long long)i * n + (j0 + t)];
}

extern "C" __global__ void gj_update(double* __restrict__ a, const double* __restrict__ p,
                                     const double* __restrict__ rp, const double* __restrict__ cb,
                                     const int n, const int j0, const int w)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= n || i >= n) return;
    const bool ri = (i >= j0 && i < j0 + w);
    const bool ci = (c >= j0 && c < j0 + w);
    const long long at = (long long)i * n + c;
    if (ri) {
        a[at] = ci ? p[(i - j0) * GJ_B + (c - j0)] : rp[(long long)(i - j0) * n + c];
        return;
    }
    const double* crow = cb + (long long)i * GJ_B;
    double s = 0.0;
    if (ci) {
        for (int t = 0; t < w; ++t) s += crow[t] * p[t * GJ_B + (c - j0)];
        a[at] = -s;
    } else {
        for (int t = 0; t < w; ++t) s += crow[t] * rp[(long long)t * n + c];
        a[at] -= s;
    }
}
