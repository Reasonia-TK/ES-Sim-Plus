// Common device helpers for ES-Sim v2 CUDA kernels (prompts/119).
// Keep this file ASCII-only (CuPy writes NVRTC sources with the OS code page).
#pragma once

// ---------------------------------------------------------------------------
// Philox4x32-10 counter-based RNG (Salmon et al., "Parallel random numbers:
// as easy as 1, 2, 3", SC'11 / Random123). Stateless: the random stream of a
// particle is a pure function of (seed, particle index, step, stream, sub),
// so kernels need no RNG state in memory and results do not depend on the
// thread schedule. es_sim/gpic/rng.py implements the same algorithm in NumPy.
// ---------------------------------------------------------------------------

__device__ __forceinline__ unsigned int es_mulhilo32(unsigned int a, unsigned int b, unsigned int* hi)
{
    const unsigned long long p = (unsigned long long)a * (unsigned long long)b;
    *hi = (unsigned int)(p >> 32);
    return (unsigned int)p;
}

__device__ __forceinline__ void es_philox4x32_10(unsigned int c[4], unsigned int k0, unsigned int k1)
{
    const unsigned int M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
    const unsigned int W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        unsigned int hi0, hi1;
        const unsigned int lo0 = es_mulhilo32(M0, c[0], &hi0);
        const unsigned int lo1 = es_mulhilo32(M1, c[2], &hi1);
        const unsigned int n0 = hi1 ^ c[1] ^ k0;
        const unsigned int n2 = hi0 ^ c[3] ^ k1;
        c[0] = n0;
        c[1] = lo1;
        c[2] = n2;
        c[3] = lo0;
        k0 += W0;
        k1 += W1;
    }
}

// Uniform double in the open interval (0, 1) from 53 random bits.
__device__ __forceinline__ double es_u01(unsigned int a, unsigned int b)
{
    const unsigned long long x = ((unsigned long long)(a >> 5) << 26) | (unsigned long long)(b >> 6);
    return ((double)x + 0.5) * 1.1102230246251565e-16;  // 2^-53
}

// Counter layout: c0,c1 = particle index (64 bit), c2 = step low 32 bits,
// c3 = step bits 32..39 | stream << 8 | sub << 16.  Each Philox call yields
// two doubles; "sub" advances per call so one (index, step, stream) owns a
// sequence of up to 131072 doubles.
struct EsRng {
    unsigned int c0, c1, c2, c3base;
    unsigned int k0, k1;
    unsigned int sub;
    unsigned int buf[4];
    int avail;

    __device__ __forceinline__ void init(unsigned long long index, unsigned long long step,
                                         unsigned int stream, unsigned long long seed)
    {
        c0 = (unsigned int)index;
        c1 = (unsigned int)(index >> 32);
        c2 = (unsigned int)step;
        c3base = ((unsigned int)(step >> 32) & 0xFFu) | ((stream & 0xFFu) << 8);
        k0 = (unsigned int)seed;
        k1 = (unsigned int)(seed >> 32);
        sub = 0;
        avail = 0;
    }

    __device__ __forceinline__ double uniform()
    {
        if (avail == 0) {
            unsigned int c[4] = {c0, c1, c2, c3base | ((sub & 0xFFFFu) << 16)};
            es_philox4x32_10(c, k0, k1);
            buf[0] = c[0];
            buf[1] = c[1];
            buf[2] = c[2];
            buf[3] = c[3];
            ++sub;
            avail = 2;
        }
        const int o = (2 - avail) * 2;
        --avail;
        return es_u01(buf[o], buf[o + 1]);
    }

    // Two independent standard normals (Box-Muller).
    __device__ __forceinline__ void normal2(double* a, double* b)
    {
        const double u1 = uniform();
        const double u2 = uniform();
        const double r = sqrt(-2.0 * log(u1));
        double s, c;
        sincospi(2.0 * u2, &s, &c);
        *a = r * c;
        *b = r * s;
    }

    // Isotropic unit vector (cos theta uniform, azimuth uniform) - same
    // distribution as es_sim.mcc.MccModel._iso_dir.
    __device__ __forceinline__ void iso_dir(double* dx, double* dy, double* dz)
    {
        const double ct = 1.0 - 2.0 * uniform();
        const double st = sqrt(fmax(1.0 - ct * ct, 0.0));
        double s, c;
        sincospi(2.0 * uniform(), &s, &c);
        *dx = st * c;
        *dy = st * s;
        *dz = ct;
    }
};

// ---------------------------------------------------------------------------
// Block reduction: every thread of the block must call this (no early return
// before it). One atomicAdd per block.
// ---------------------------------------------------------------------------

__device__ __forceinline__ double es_warp_sum(double v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ void es_block_sum_atomic(double v, double* out)
{
    __shared__ double sh[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    v = es_warp_sum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    if (wid == 0) {
        v = (lane < nw) ? sh[lane] : 0.0;
        v = es_warp_sum(v);
        if (lane == 0) atomicAdd(out, v);
    }
}

// ---------------------------------------------------------------------------
// Node-centred grid helpers: cell index and bilinear (CIC) weights.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void es_cell(double x, double y, double x0, double y0,
                                        double inv_dx, double inv_dy, int nx, int ny,
                                        int* i, int* j, double* wx, double* wy)
{
    const double fx = (x - x0) * inv_dx;
    const double fy = (y - y0) * inv_dy;
    int ii = (int)floor(fx);
    int jj = (int)floor(fy);
    ii = ii < 0 ? 0 : (ii > nx - 1 ? nx - 1 : ii);
    jj = jj < 0 ? 0 : (jj > ny - 1 ? ny - 1 : jj);
    *i = ii;
    *j = jj;
    *wx = fmin(fmax(fx - ii, 0.0), 1.0);
    *wy = fmin(fmax(fy - jj, 0.0), 1.0);
}

// Linear interpolation in a sorted table with clamping at both ends
// (identical semantics to numpy.interp, as used by the v1 MCC).
__device__ __forceinline__ double es_interp(const double* e, const double* s, int len, double E)
{
    if (len <= 0) return 0.0;
    if (E <= e[0]) return s[0];
    if (E >= e[len - 1]) return s[len - 1];
    int lo = 0, hi = len - 1;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (e[mid] <= E) lo = mid;
        else hi = mid;
    }
    const double de = e[hi] - e[lo];
    if (de <= 0.0) return s[hi];
    return s[lo] + (s[hi] - s[lo]) * (E - e[lo]) / de;
}
