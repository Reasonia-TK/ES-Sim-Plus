// ES-Sim v2 GPU DSMC kernels on the Cartesian grid with embedded solids (prompts/124).
// ASCII only. Physics follows the verified v1 implementation (es_sim/dsmc.py):
// VHS molecules, NTC collisions per cell, diffuse / specular walls, pressure
// reservoirs, vacuum outlets, flow-rate inlets, axisymmetric rings (rz).
//
// Particles are SoA (x, y, vx, vy, vz). After every move the particles are sorted
// by cell (dead ones last) by the host (CuPy argsort + gather), so the collision
// and sampling kernels see the particles of cell c at [start[c], start[c]+count[c]).

#include "common.cuh"

#define B_WALL 0   // diffuse, fully accommodated at the boundary temperature
#define B_SYM  1   // specular
#define B_RES  2   // absorbed (pressure reservoir; inflow is injected separately)
#define B_VAC  3   // absorbed (vacuum)

#define KB_SI 1.380649e-23
#define MAX_LEGS 6

// Boundary intervals of the 4 domain sides (0 left, 1 right, 2 bottom, 3 top):
// side_off[s]..side_off[s+1] index into iv (4 doubles each: c0, c1, type, temperature),
// c = y for left/right, x for bottom/top. Unlisted parts are diffuse walls at t_wall.
__device__ __forceinline__ void ds_side_type(const int* __restrict__ side_off, const double* __restrict__ iv,
                                             int side, double c, double t_wall, int* type, double* temp)
{
    *type = B_WALL;
    *temp = t_wall;
    for (int k = side_off[side]; k < side_off[side + 1]; ++k) {
        const double* q = iv + 4 * k;
        if (c >= q[0] && c <= q[1]) {
            *type = (int)q[2];
            *temp = q[3];
        }
    }
}

__device__ __forceinline__ bool ds_in_polygon(double px, double py, const double* v, int nv)
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

// First crossing of the segment p0 -> p1 with a solid (entering it). Returns the
// fraction t in [0,1] (2 if none) and the unit normal pointing back into the gas.
__device__ double ds_solid_hit(double ox, double oy, double px, double py, int n_solid,
                               const int* __restrict__ s_type, const int* __restrict__ s_off,
                               const double* __restrict__ pxy, const double* __restrict__ circ,
                               double* nx_out, double* ny_out)
{
    const double dxs = px - ox, dys = py - oy;
    double best = 2.0, nxn = 0.0, nyn = 0.0;
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
                    const double hx = ox + t0 * dxs - cxs, hy = oy + t0 * dys - cys;
                    const double nr = sqrt(hx * hx + hy * hy);
                    nxn = nr > 0.0 ? hx / nr : 0.0;
                    nyn = nr > 0.0 ? hy / nr : 0.0;
                }
            }
        } else {
            const double* v = pxy + 2 * s_off[s];
            const int nv = s_off[s + 1] - s_off[s];
            for (int a = 0; a < nv; ++a) {
                const int b2 = (a + 1 == nv) ? 0 : a + 1;
                const double qx = v[2 * a], qy = v[2 * a + 1];
                const double ex = v[2 * b2] - qx, ey = v[2 * b2 + 1] - qy;
                const double den = dxs * ey - dys * ex;
                if (den == 0.0) continue;
                const double wx = qx - ox, wy = qy - oy;
                const double t = (wx * ey - wy * ex) / den;
                const double u = (wx * dys - wy * dxs) / den;
                if (t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 && t < best) {
                    best = t;
                    const double el = sqrt(ex * ex + ey * ey);
                    nxn = el > 0.0 ? ey / el : 0.0;
                    nyn = el > 0.0 ? -ex / el : 0.0;
                }
            }
        }
    }
    if (best <= 1.0 && nxn * dxs + nyn * dys > 0.0) { nxn = -nxn; nyn = -nyn; }
    *nx_out = nxn;
    *ny_out = nyn;
    return best;
}

__device__ __forceinline__ bool ds_inside_solid(double px, double py, int n_solid, const int* __restrict__ s_type,
                                                const int* __restrict__ s_off, const double* __restrict__ pxy,
                                                const double* __restrict__ circ)
{
    for (int s = 0; s < n_solid; ++s) {
        if (s_type[s] == 1) {
            const double dx = px - circ[3 * s], dy = py - circ[3 * s + 1], r = circ[3 * s + 2];
            if (dx * dx + dy * dy <= r * r) return true;
        } else if (ds_in_polygon(px, py, pxy + 2 * s_off[s], s_off[s + 1] - s_off[s])) {
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ unsigned char ds_cell_state(double px, double py, double X0, double Y0,
                                                       double inv_dx, double inv_dy, int nx, int ny,
                                                       const unsigned char* __restrict__ cell_state)
{
    int i = (int)floor((px - X0) * inv_dx);
    int j = (int)floor((py - Y0) * inv_dy);
    i = i < 0 ? 0 : (i > nx - 1 ? nx - 1 : i);
    j = j < 0 ? 0 : (j > ny - 1 ? ny - 1 : j);
    return cell_state[j * nx + i];
}

// ---------------------------------------------------------------------------
// Free flight + boundaries (multi-leg: a reflected / re-emitted molecule completes
// the remaining time, up to MAX_LEGS legs, as in v1). rz: exact 3D straight flight
// projected on the meridian plane (r' = sqrt((r+vr t)^2 + (vt t)^2), velocity
// rotated into the new plane); crossings use the straight (z, r) chord.
// Writes key[p] = cell index (n_cells for removed molecules). cnt[0] += removed.
// ---------------------------------------------------------------------------
extern "C" __global__ void dsmc_move(
    double* __restrict__ x, double* __restrict__ y, double* __restrict__ vx, double* __restrict__ vy,
    double* __restrict__ vz, int* __restrict__ key, const long long n, const double dt,
    const int rz, const int ridx,
    const double X0, const double Y0, const double X1, const double Y1,
    const double inv_dx, const double inv_dy, const int nx, const int ny,
    const unsigned char* __restrict__ cell_state,
    const int* __restrict__ side_off, const double* __restrict__ iv, const double t_wall, const double mass,
    const int n_solid, const int* __restrict__ s_type, const int* __restrict__ s_off,
    const double* __restrict__ pxy, const double* __restrict__ circ,
    const double delta, const unsigned long long step, const unsigned long long seed,
    unsigned long long* __restrict__ cnt)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    double ox = x[p], oy = y[p];
    double ux = vx[p], uy = vy[p], uz = vz[p];
    double t_rem = dt;
    EsRng rng;
    rng.init((unsigned long long)p, step, 8u, seed);
    const int n_cells = nx * ny;
    bool removed = false;
    for (int leg = 0; leg < MAX_LEGS && t_rem > 0.0; ++leg) {
        // end point of the free flight over t_rem (and the rotated velocity for rz)
        double px, py, nvx = ux, nvy = uy, nvz = uz;
        if (!rz) {
            px = ox + ux * t_rem;
            py = oy + uy * t_rem;
        } else {
            double r, vr, zc, vzc;
            if (ridx == 1) { r = oy; vr = uy; zc = ox; vzc = ux; } else { r = ox; vr = ux; zc = oy; vzc = uy; }
            const double rm = r + vr * t_rem, dout = uz * t_rem;
            const double rn = sqrt(rm * rm + dout * dout);
            double c = 1.0, sn = 0.0;
            if (rn > 0.0) { c = rm / rn; sn = dout / rn; }
            const double vr_n = vr * c + uz * sn, vt_n = uz * c - vr * sn;
            const double zn = zc + vzc * t_rem;
            if (ridx == 1) { px = zn; py = rn; nvx = vzc; nvy = vr_n; } else { px = rn; py = zn; nvx = vr_n; nvy = vzc; }
            nvz = vt_n;
        }
        // earliest crossing: domain sides, then solids
        double best = 2.0, hnx = 0.0, hny = 0.0;
        int btype = -1;
        double btemp = t_wall;
        const double dxs = px - ox, dys = py - oy;
        if (px < X0 && dxs != 0.0) {
            const double f = (X0 - ox) / dxs;
            if (f < best) { best = f; hnx = 1.0; hny = 0.0; ds_side_type(side_off, iv, 0, oy + f * dys, t_wall, &btype, &btemp); }
        }
        if (px > X1 && dxs != 0.0) {
            const double f = (X1 - ox) / dxs;
            if (f < best) { best = f; hnx = -1.0; hny = 0.0; ds_side_type(side_off, iv, 1, oy + f * dys, t_wall, &btype, &btemp); }
        }
        if (py < Y0 && dys != 0.0) {
            const double f = (Y0 - oy) / dys;
            if (f < best) { best = f; hnx = 0.0; hny = 1.0; ds_side_type(side_off, iv, 2, ox + f * dxs, t_wall, &btype, &btemp); }
        }
        if (py > Y1 && dys != 0.0) {
            const double f = (Y1 - oy) / dys;
            if (f < best) { best = f; hnx = 0.0; hny = -1.0; ds_side_type(side_off, iv, 3, ox + f * dxs, t_wall, &btype, &btemp); }
        }
        if (n_solid > 0) {
            const double pxc = fmin(fmax(px, X0), X1), pyc = fmin(fmax(py, Y0), Y1);
            const unsigned char s0 = ds_cell_state(ox, oy, X0, Y0, inv_dx, inv_dy, nx, ny, cell_state);
            const unsigned char s1 = ds_cell_state(pxc, pyc, X0, Y0, inv_dx, inv_dy, nx, ny, cell_state);
            if (s0 != 0 || s1 != 0) {
                double snx, sny;
                const double fs = ds_solid_hit(ox, oy, px, py, n_solid, s_type, s_off, pxy, circ, &snx, &sny);
                if (fs < best) { best = fs; hnx = snx; hny = sny; btype = B_WALL; btemp = t_wall; }
            }
        }
        if (best > 1.0) {
            ox = px; oy = py; ux = nvx; uy = nvy; uz = nvz;
            t_rem = 0.0;
            break;
        }
        best = fmin(fmax(best, 0.0), 1.0);
        const double hx = ox + best * dxs, hy = oy + best * dys;
        t_rem *= (1.0 - best);
        // velocity at the boundary (rz: the rotated one, as in v1)
        ux = nvx; uy = nvy; uz = nvz;
        if (btype == B_RES || btype == B_VAC) {
            removed = true;
            ox = hx; oy = hy;
            break;
        }
        if (btype == B_WALL) {
            const double sig = sqrt(KB_SI * btemp / mass);
            const double vn = sig * sqrt(-2.0 * log(fmax(rng.uniform(), 1e-300)));
            double g0, g1;
            rng.normal2(&g0, &g1);
            const double tx = -hny, ty = hnx;  // tangent
            ux = vn * hnx + sig * g0 * tx;
            uy = vn * hny + sig * g0 * ty;
            uz = sig * g1;
        } else {  // specular
            const double vn = ux * hnx + uy * hny;
            if (vn < 0.0) { ux -= 2.0 * vn * hnx; uy -= 2.0 * vn * hny; }
        }
        ox = hx + delta * hnx;
        oy = hy + delta * hny;
    }
    // keep inside the domain (numerical safety) and never inside a solid
    ox = fmin(fmax(ox, X0), X1);
    oy = fmin(fmax(oy, Y0), Y1);
    x[p] = ox; y[p] = oy; vx[p] = ux; vy[p] = uy; vz[p] = uz;
    if (removed) {
        key[p] = n_cells;
        atomicAdd(cnt, 1ull);
        return;
    }
    int i = (int)floor((ox - X0) * inv_dx);
    int j = (int)floor((oy - Y0) * inv_dy);
    i = i < 0 ? 0 : (i > nx - 1 ? nx - 1 : i);
    j = j < 0 ? 0 : (j > ny - 1 ? ny - 1 : j);
    key[p] = j * nx + i;
}

// ---------------------------------------------------------------------------
// NTC collisions in parallel over candidate pairs (two kernels):
//  dsmc_ncand : per cell, N_cand = 1/2 N (N-1) W (sigma c_r)_max dt / V + carried
//               fraction (integer part to ncand[c], fraction carried).
//  dsmc_pairs : one thread per candidate (cell found by binary search in the
//               exclusive scan of ncand). The pair (i, j) is claimed with atomicMax
//               on a per-molecule step stamp, so a molecule collides at most once
//               per step (as the de-duplication of v1); a candidate that cannot claim
//               both molecules is carried to the next step (frac += 1, as v1).
//               (sigma c_r)_max is raised with atomicMax on the double bits
//               (non-negative doubles order like their bit patterns); a candidate
//               above the current maximum is accepted.
// ---------------------------------------------------------------------------
extern "C" __global__ void dsmc_ncand(const int* __restrict__ count, const double* __restrict__ vol,
                                      const double* __restrict__ sigcr_max, double* __restrict__ frac,
                                      long long* __restrict__ ncand, const double w_dt, const int n_cells)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    const int N = count[c];
    const double V = vol[c];
    if (N < 2 || V <= 0.0) { ncand[c] = 0; return; }
    const double nf = 0.5 * (double)N * (double)(N - 1) * w_dt * sigcr_max[c] / V + frac[c];
    const long long nc = (long long)nf;
    frac[c] = nf - (double)nc;
    ncand[c] = nc;
}

extern "C" __global__ void dsmc_pairs(
    double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
    const int* __restrict__ start, const int* __restrict__ count,
    const long long* __restrict__ cand_start, const int n_cells, const long long total,
    double* __restrict__ sigcr_max, double* __restrict__ frac, unsigned int* __restrict__ claim,
    const unsigned int stamp, const double sig_coef, const double sig_pow,
    const unsigned long long step, const unsigned long long seed, unsigned long long* __restrict__ ncoll)
{
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= total) return;
    int lo = 0, hi = n_cells;          // cand_start[lo] <= k < cand_start[hi]
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (cand_start[mid] <= k) lo = mid;
        else hi = mid;
    }
    const int c = lo;
    const int N = count[c];
    EsRng rng;
    rng.init((unsigned long long)k, step, 9u, seed);
    int a = (int)(rng.uniform() * N);
    int b = (int)(rng.uniform() * N);
    if (a >= N) a = N - 1;
    if (b >= N) b = N - 1;
    if (a == b) return;
    const int i = start[c] + a, j = start[c] + b;
    const double gx = vx[i] - vx[j], gy = vy[i] - vy[j], gz = vz[i] - vz[j];
    const double g = sqrt(gx * gx + gy * gy + gz * gz);
    const double scr = g > 0.0 ? sig_coef * pow(g, sig_pow) * g : 0.0;
    const double smax = sigcr_max[c];
    if (scr > smax)
        atomicMax((unsigned long long*)(sigcr_max + c), (unsigned long long)__double_as_longlong(scr));
    if (rng.uniform() * fmax(smax, scr) >= scr) return;
    // claim both molecules for this step
    const unsigned int oi = atomicMax(claim + i, stamp);
    if (oi >= stamp) { atomicAdd(frac + c, 1.0); return; }
    const unsigned int oj = atomicMax(claim + j, stamp);
    if (oj >= stamp) { claim[i] = oi; atomicAdd(frac + c, 1.0); return; }
    const double cos_t = 1.0 - 2.0 * rng.uniform();
    const double sin_t = sqrt(fmax(1.0 - cos_t * cos_t, 0.0));
    const double ph = 6.283185307179586 * rng.uniform();
    const double dxn = sin_t * cos(ph), dyn = sin_t * sin(ph), dzn = cos_t;
    const double cx = 0.5 * (vx[i] + vx[j]), cy = 0.5 * (vy[i] + vy[j]), cz = 0.5 * (vz[i] + vz[j]);
    const double hg = 0.5 * g;
    vx[i] = cx + hg * dxn; vy[i] = cy + hg * dyn; vz[i] = cz + hg * dzn;
    vx[j] = cx - hg * dxn; vy[j] = cy - hg * dyn; vz[j] = cz - hg * dzn;
    atomicAdd(ncoll, 1ull);
}

// ---------------------------------------------------------------------------
// (Reference) NTC collisions with one thread per cell, sequential pairs. Kept for
// validation of dsmc_pairs; not used by the simulation (poor occupancy when cells
// are few and crowded).
// ---------------------------------------------------------------------------
extern "C" __global__ void dsmc_collide(
    double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
    const int* __restrict__ start, const int* __restrict__ count, const double* __restrict__ vol,
    double* __restrict__ sigcr_max, double* __restrict__ frac, const double w_dt,
    const double sig_coef, const double sig_pow, const int n_cells,
    const unsigned long long step, const unsigned long long seed, unsigned long long* __restrict__ ncoll)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    const int N = count[c];
    const double V = vol[c];
    if (N < 2 || V <= 0.0) return;
    double smax = sigcr_max[c];
    const double nf = 0.5 * (double)N * (double)(N - 1) * w_dt * smax / V + frac[c];
    const long long nc = (long long)nf;
    frac[c] = nf - (double)nc;
    EsRng rng;
    rng.init((unsigned long long)c, step, 9u, seed);
    const int s = start[c];
    unsigned long long acc = 0;
    for (long long k = 0; k < nc; ++k) {
        int a = (int)(rng.uniform() * N);
        int b = (int)(rng.uniform() * N);
        if (a >= N) a = N - 1;
        if (b >= N) b = N - 1;
        if (a == b) continue;
        const int i = s + a, j = s + b;
        const double gx = vx[i] - vx[j], gy = vy[i] - vy[j], gz = vz[i] - vz[j];
        const double g = sqrt(gx * gx + gy * gy + gz * gz);
        const double scr = g > 0.0 ? sig_coef * pow(g, sig_pow) * g : 0.0;
        if (scr > smax) smax = scr;
        if (rng.uniform() * smax >= scr) continue;
        const double cos_t = 1.0 - 2.0 * rng.uniform();
        const double sin_t = sqrt(fmax(1.0 - cos_t * cos_t, 0.0));
        const double ph = 6.283185307179586 * rng.uniform();
        const double dxn = sin_t * cos(ph), dyn = sin_t * sin(ph), dzn = cos_t;
        const double cx = 0.5 * (vx[i] + vx[j]), cy = 0.5 * (vy[i] + vy[j]), cz = 0.5 * (vz[i] + vz[j]);
        const double hg = 0.5 * g;
        vx[i] = cx + hg * dxn; vy[i] = cy + hg * dyn; vz[i] = cz + hg * dzn;
        vx[j] = cx - hg * dxn; vy[j] = cy - hg * dyn; vz[j] = cz - hg * dzn;
        ++acc;
    }
    sigcr_max[c] = smax;
    if (acc) atomicAdd(ncoll, acc);
}

// Per-cell moments of the sorted particles: count, sum v (3), sum |v|^2.
extern "C" __global__ void dsmc_sample(const double* __restrict__ vx, const double* __restrict__ vy,
                                       const double* __restrict__ vz, const int* __restrict__ start,
                                       const int* __restrict__ count, double* __restrict__ acc_cnt,
                                       double* __restrict__ acc_v, double* __restrict__ acc_v2, const int n_cells)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    const int s = start[c], N = count[c];
    double sx = 0.0, sy = 0.0, sz = 0.0, s2 = 0.0;
    for (int k = s; k < s + N; ++k) {
        sx += vx[k]; sy += vy[k]; sz += vz[k];
        s2 += vx[k] * vx[k] + vy[k] * vy[k] + vz[k] * vz[k];
    }
    acc_cnt[c] += (double)N;
    acc_v[3 * c] += sx;
    acc_v[3 * c + 1] += sy;
    acc_v[3 * c + 2] += sz;
    acc_v2[c] += s2;
}

// ---------------------------------------------------------------------------
// Injection through one boundary segment (reservoir / flow inlet): k molecules at
// [base, base+k). Position uniform along the segment (rz radial sides: density ~ r),
// offset inside by delta; velocity = flux-weighted half Maxwellian at temperature T.
// seg: ax, ay (start), bx, by (end), nx, ny (inward normal), temp, rweight (0/1),
// ridx (radial coordinate index, -1 for xy).
// ---------------------------------------------------------------------------
extern "C" __global__ void dsmc_inject(double* __restrict__ x, double* __restrict__ y, double* __restrict__ vx,
                                       double* __restrict__ vy, double* __restrict__ vz, const long long base,
                                       const long long k, const double ax, const double ay, const double bx,
                                       const double by, const double nxn, const double nyn, const double temp,
                                       const int rweight, const int ridx, const double delta, const double mass,
                                       const unsigned long long step, const unsigned long long seed,
                                       const unsigned long long stream_id)
{
    const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= k) return;
    EsRng rng;
    rng.init((unsigned long long)q, step, (unsigned int)(10u + (stream_id & 0xEFu)), seed);
    double u = rng.uniform();
    if (rweight) {
        // radial side: sample r with density ~ r between the end points
        const double r0 = ridx == 1 ? ay : ax, r1 = ridx == 1 ? by : bx;
        const double r = sqrt(r0 * r0 + u * (r1 * r1 - r0 * r0));
        u = (r1 != r0) ? (r - r0) / (r1 - r0) : u;
    }
    const long long p = base + q;
    x[p] = ax + u * (bx - ax) + delta * nxn;
    y[p] = ay + u * (by - ay) + delta * nyn;
    const double sig = sqrt(KB_SI * temp / mass);
    const double vn = sig * sqrt(-2.0 * log(fmax(rng.uniform(), 1e-300)));
    double g0, g1;
    rng.normal2(&g0, &g1);
    const double tx = -nyn, ty = nxn;
    vx[p] = vn * nxn + sig * g0 * tx;
    vy[p] = vn * nyn + sig * g0 * ty;
    vz[p] = sig * g1;
}
