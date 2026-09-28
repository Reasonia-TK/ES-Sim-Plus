// ES-Sim v2 AMR composite-grid kernels (prompts/122). ASCII only.
//
// Sparse matrices are CSR with int32 indices (ip = row pointers, ix = column
// indices, a = values). Every kernel is allocation free and reads its scalars
// from arguments or device memory, so whole solves can be captured in a CUDA
// graph (see es_sim/amr/gpu_solver.py).

// y = A x (mode 0), y += A x (mode 1), y -= A x (mode 2). One thread per row.
extern "C" __global__ void csr_spmv(double* __restrict__ y, const int* __restrict__ ip,
                                    const int* __restrict__ ix, const double* __restrict__ a,
                                    const double* __restrict__ x, const int n, const int mode)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    double s = 0.0;
    for (int k = ip[row]; k < ip[row + 1]; ++k) s += a[k] * x[ix[k]];
    if (mode == 0) y[row] = s;
    else if (mode == 1) y[row] += s;
    else y[row] -= s;
}

// r = b - A x
extern "C" __global__ void csr_resid(double* __restrict__ r, const double* __restrict__ b,
                                     const int* __restrict__ ip, const int* __restrict__ ix,
                                     const double* __restrict__ a, const double* __restrict__ x, const int n)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    double s = 0.0;
    for (int k = ip[row]; k < ip[row + 1]; ++k) s += a[k] * x[ix[k]];
    r[row] = b[row] - s;
}

// Chebyshev smoother steps (symmetric polynomial in D^-1 A):
//   first: d = s * dinv * r            next: d = c1 * d + c2 * dinv * r
extern "C" __global__ void cheb_first(double* __restrict__ d, const double* __restrict__ r,
                                      const double* __restrict__ dinv, const double s, const int n)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) d[k] = s * dinv[k] * r[k];
}

extern "C" __global__ void cheb_next(double* __restrict__ d, const double* __restrict__ r,
                                     const double* __restrict__ dinv, const double c1, const double c2,
                                     const int n)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) d[k] = c1 * d[k] + c2 * dinv[k] * r[k];
}

// y += alpha x
extern "C" __global__ void vec_axpy(double* __restrict__ y, const double* __restrict__ x, const double alpha,
                                    const int n)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) y[k] += alpha * x[k];
}

// x = v
extern "C" __global__ void vec_fill(double* __restrict__ x, const double v, const int n)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) x[k] = v;
}

// x -= s[slot] / n   (projection onto mean zero; s[slot] holds sum(x))
extern "C" __global__ void vec_sub_mean(double* __restrict__ x, const double* __restrict__ s, const int slot,
                                        const int n)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) x[k] -= s[slot] / (double)n;
}

// out[0] += sum(x). Grid-stride; every thread reaches the reduction.
__device__ __forceinline__ double es_warp_sum_a(double v)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

extern "C" __global__ void vec_sum(const double* __restrict__ x, const int n, double* __restrict__ out)
{
    __shared__ double sh[32];
    double v = 0.0;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x) v += x[k];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = es_warp_sum_a(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < (int)((blockDim.x + 31) >> 5)) ? sh[lane] : 0.0;
        v = es_warp_sum_a(v);
        if (lane == 0) atomicAdd(out, v);
    }
}

// x = M b for a dense row-major m x m matrix (coarsest level / small problems)
extern "C" __global__ void dense_gemv(const double* __restrict__ m, const double* __restrict__ b,
                                      double* __restrict__ x, const int n)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    const double* mr = m + (long long)row * n;
    double s = 0.0;
    for (int c = 0; c < n; ++c) s += mr[c] * b[c];
    x[row] = s;
}
