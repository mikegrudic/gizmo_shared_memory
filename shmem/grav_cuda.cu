// Brute-force O(N^2) gravity on the GPU, reproducing EXACTLY the pair force the tree walk
// applies -- not Newtonian point masses, and not a simplified softening. Its only purpose is to
// answer one question the tree cannot answer about itself: how much of the momentum this engine
// injects at the gravity kick is the tree's opening approximation, and how much is the force law.
// Swap it in for compute_gravity and the tree's contribution is identically zero.
//
// Structure follows pytreegrav's CUDA brute force: one thread per target, sources staged through
// shared memory in tiles, and the precision discipline made explicit rather than left to the
// array dtypes.
//
// PRECISION. Coordinates arrive and are stored in DOUBLE -- the engine's own -- and the pair
// SEPARATION is differenced in double before anything else happens. Everything downstream (r,
// the spline, the zeta terms, the accumulator) is float by default. That split is the point:
// narrowing the coordinates themselves would difference two nearby ~1e-1 positions in float and
// lose most of the significant digits of a ~1e-5 separation, whereas differencing wide and
// narrowing the small result costs one f64 subtract per pair and keeps every digit that matters.
// On a 1:32-FP64 device the rest in float is worth roughly an order of magnitude.
//
// SHMEM_CUDA_FP64=1 runs the whole chain in double instead. Keep it: a float sum over 1.3e5
// sources carries ~sqrt(N)*eps ~ 4e-5 of its own, which is the same size as the momentum errors
// this was built to measure -- so the diagnostic use wants f64 even though the fast path does not.
//
// The pair rule mirrors tree.cc's pair_force_over_r; see the comment there for why each branch is
// what it is. Kept in sync by construction: both are short and both are read together.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int TILE = 256;   // sources staged per iteration; one block == TILE threads

// ---- spline softening, identical to tree.h's spline_force_over_r -------------------------
// Templated on the arithmetic type: every literal is cast, so a float instantiation stays float
// end to end. Numba needs the same care for the same reason (pytreegrav's _numerics) -- an
// un-cast literal silently promotes the whole chain, which is invisible in the source and costs
// an order of magnitude on a device with 1:32 FP64.
template <typename R>
__device__ __forceinline__ R spline_force_over_r(R r, R h) {
    const R h_inv = R(1) / h;
    const R h_inv3 = h_inv * h_inv * h_inv;
    const R u = r * h_inv;
    if (u >= R(1)) return R(1) / (r * r * r);
    if (u < R(0.5)) return h_inv3 * (R(10.666666666666667) + u*u*(R(32)*u - R(38.4)));
    return h_inv3 * (R(21.333333333333332) - R(48)*u + R(38.4)*u*u - R(10.666666666666667)*u*u*u
                     - R(0.06666666666666667)/(u*u*u));
}

// ---- cubic-spline dW/dr in 3D, identical to hydro.h's kernel_dwdr(dim=3) -----------------
template <typename R>
__device__ __forceinline__ R kernel_dwdr(R r, R h) {
    const R q = r / h;
    if (q >= R(1)) return R(0);
    const R norm = R(8) / (R(M_PI) * h * h * h);
    R fp;
    if (q < R(0.5)) fp = R(-12)*q + R(18)*q*q;
    else { const R u = R(1) - q; fp = R(-6)*u*u; }
    return norm * fp / h;
}

// One target per thread. Sources are tiled through shared memory so each is read from global
// memory once per block rather than once per thread. R is the ARITHMETIC type; positions stay
// double regardless, and only the separation is narrowed (see the header comment).
template <typename R>
__global__ void accel_bruteforce_kernel(
        const double* __restrict__ tx, const double* __restrict__ ty, const double* __restrict__ tz,
        const double* __restrict__ teps, const double* __restrict__ tzeta,
        const double* __restrict__ tmass, const char* __restrict__ tgas,
        const double* __restrict__ sx, const double* __restrict__ sy, const double* __restrict__ sz,
        const double* __restrict__ sm, const double* __restrict__ seps,
        const double* __restrict__ szeta, const char* __restrict__ sgas,
        const int* __restrict__ tindex,             // source index of each target, to skip self
        int n_target, int n_source, double box,
        double* __restrict__ ax, double* __restrict__ ay, double* __restrict__ az) {

    __shared__ double shx[TILE], shy[TILE], shz[TILE];   // positions stay wide
    __shared__ R      shm[TILE], sheps[TILE], shzeta[TILE];
    __shared__ char   shgas[TILE];

    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const bool live = (t < n_target);

    const double xi = live ? tx[t] : 0.0;    // wide: the separation is differenced before narrowing
    const double yi = live ? ty[t] : 0.0;
    const double zi = live ? tz[t] : 0.0;
    const R epsi = live ? R(teps[t]) : R(1);
    const R zti  = live ? R(tzeta[t]): R(0);
    const R mi   = live ? R(tmass[t]): R(1);
    const bool gasi = live ? (tgas[t] != 0) : false;
    const int  self = live ? tindex[t] : -1;

    R axi = R(0), ayi = R(0), azi = R(0);

    for (int base = 0; base < n_source; base += TILE) {
        const int k = base + threadIdx.x;
        if (k < n_source) {
            shx[threadIdx.x] = sx[k]; shy[threadIdx.x] = sy[k]; shz[threadIdx.x] = sz[k];
            shm[threadIdx.x] = R(sm[k]); sheps[threadIdx.x] = R(seps[k]);
            shzeta[threadIdx.x] = R(szeta[k]); shgas[threadIdx.x] = sgas[k];
        }
        __syncthreads();

        const int n_tile = min(TILE, n_source - base);
        if (live) {
            for (int q = 0; q < n_tile; ++q) {
                if (base + q == self) continue;               // no self-interaction
                // DIFFERENCE WIDE, then narrow: two ~1e-1 coordinates differenced in float would
                // keep only a few digits of a ~1e-5 separation.
                double dxd = shx[q] - xi, dyd = shy[q] - yi, dzd = shz[q] - zi;
                if (box > 0) {
                    const double hb = 0.5 * box;
                    if (dxd >  hb) dxd -= box; else if (dxd < -hb) dxd += box;
                    if (dyd >  hb) dyd -= box; else if (dyd < -hb) dyd += box;
                    if (dzd >  hb) dzd -= box; else if (dzd < -hb) dzd += box;
                }
                const R dx = R(dxd), dy = R(dyd), dz = R(dzd);
                const R r2 = dx*dx + dy*dy + dz*dz;
                if (r2 <= R(0)) continue;
                const R r = sqrt(r2);
                const R mj = shm[q], epsj = sheps[q];

                R fac;
                if (!(gasi && shgas[q] != 0)) {
                    // any pair involving a non-gas particle: kernel of the LARGER softening,
                    // no averaging and no zeta
                    R e = epsi > epsj ? epsi : epsj;
                    if (!(e > R(0))) e = R(1e-30);
                    fac = mj * spline_force_over_r<R>(r, e);
                } else {
                    const R et = epsi > R(0) ? epsi : R(1e-30);
                    const R es = epsj > R(0) ? epsj : R(1e-30);
                    fac = R(0.5) * mj * (spline_force_over_r<R>(r, et)
                                       + spline_force_over_r<R>(r, es));
                    // Price & Monaghan zeta terms, each inside its own kernel support, divided by
                    // the TARGET mass so the pair stays antisymmetric (m_t a_t = -m_s a_s)
                    if (mi > R(0)) {
                        const R zs = shzeta[q];
                        if (zti != R(0) && r < et) fac -= (zti / mi) * kernel_dwdr<R>(r, et) / r;
                        if (zs  != R(0) && r < es) fac -= (zs  / mi) * kernel_dwdr<R>(r, es) / r;
                    }
                }
                axi += fac * dx; ayi += fac * dy; azi += fac * dz;
            }
        }
        __syncthreads();
    }

    if (live) { ax[t] = (double)axi; ay[t] = (double)ayi; az[t] = (double)azi; }
}

// Device-side scratch, kept across calls so a run does not re-allocate every step.
struct Buffers {
    double *sx = nullptr, *sy = nullptr, *sz = nullptr, *sm = nullptr, *seps = nullptr, *szeta = nullptr;
    char   *sgas = nullptr;
    double *tx = nullptr, *ty = nullptr, *tz = nullptr, *teps = nullptr, *tzeta = nullptr, *tmass = nullptr;
    char   *tgas = nullptr;
    int    *tidx = nullptr;
    double *ax = nullptr, *ay = nullptr, *az = nullptr;
    size_t n_source_cap = 0, n_target_cap = 0;
};
Buffers g_buf;

template <typename T>
void ensure(T*& p, size_t& cap, size_t want) {
    if (cap >= want) return;
    if (p) cudaFree(p);
    cudaMalloc(&p, want * sizeof(T));
}

void grow_sources(size_t n) {
    if (g_buf.n_source_cap >= n) return;
    size_t c = g_buf.n_source_cap;
    ensure(g_buf.sx, c, n); c = g_buf.n_source_cap; ensure(g_buf.sy, c, n);
    c = g_buf.n_source_cap; ensure(g_buf.sz, c, n); c = g_buf.n_source_cap; ensure(g_buf.sm, c, n);
    c = g_buf.n_source_cap; ensure(g_buf.seps, c, n); c = g_buf.n_source_cap; ensure(g_buf.szeta, c, n);
    c = g_buf.n_source_cap; ensure(g_buf.sgas, c, n);
    g_buf.n_source_cap = n;
}

void grow_targets(size_t n) {
    if (g_buf.n_target_cap >= n) return;
    size_t c;
    c = g_buf.n_target_cap; ensure(g_buf.tx, c, n);   c = g_buf.n_target_cap; ensure(g_buf.ty, c, n);
    c = g_buf.n_target_cap; ensure(g_buf.tz, c, n);   c = g_buf.n_target_cap; ensure(g_buf.teps, c, n);
    c = g_buf.n_target_cap; ensure(g_buf.tzeta, c, n);c = g_buf.n_target_cap; ensure(g_buf.tmass, c, n);
    c = g_buf.n_target_cap; ensure(g_buf.tgas, c, n); c = g_buf.n_target_cap; ensure(g_buf.tidx, c, n);
    c = g_buf.n_target_cap; ensure(g_buf.ax, c, n);   c = g_buf.n_target_cap; ensure(g_buf.ay, c, n);
    c = g_buf.n_target_cap; ensure(g_buf.az, c, n);
    g_buf.n_target_cap = n;
}

} // namespace

// C entry point called from mfm.cc. All arrays are host-side; the accelerations come back
// WITHOUT the factor of G, matching what accel_grouped returns before the caller scales it.
extern "C" void shmem_cuda_accel_bruteforce(
        int n_source, const double* sx, const double* sy, const double* sz,
        const double* sm, const double* seps, const double* szeta, const char* sgas,
        int n_target, const int* tidx,
        double box, double* ax_out, double* ay_out, double* az_out) {
    if (n_target <= 0 || n_source <= 0) return;

    grow_sources((size_t)n_source);
    grow_targets((size_t)n_target);

    cudaMemcpy(g_buf.sx, sx, n_source*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_buf.sy, sy, n_source*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_buf.sz, sz, n_source*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_buf.sm, sm, n_source*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_buf.seps, seps, n_source*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_buf.szeta, szeta, n_source*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_buf.sgas, sgas, n_source*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(g_buf.tidx, tidx, n_target*sizeof(int), cudaMemcpyHostToDevice);

    // Targets are a subset of the sources, so gather their per-particle data on the device by
    // index rather than shipping a second copy: one small kernel, no extra host traffic.
    static double* h_scratch = nullptr; static size_t scratch_cap = 0;
    if (scratch_cap < (size_t)n_target) {
        free(h_scratch); h_scratch = (double*)malloc(n_target*sizeof(double));
        scratch_cap = (size_t)n_target;
    }
    static char* h_gas = nullptr; static size_t gas_cap = 0;
    if (gas_cap < (size_t)n_target) { free(h_gas); h_gas = (char*)malloc(n_target); gas_cap = (size_t)n_target; }

    #define GATHER(src, dst)                                                        \
        for (int i = 0; i < n_target; ++i) h_scratch[i] = src[tidx[i]];             \
        cudaMemcpy(dst, h_scratch, n_target*sizeof(double), cudaMemcpyHostToDevice);
    GATHER(sx, g_buf.tx) GATHER(sy, g_buf.ty) GATHER(sz, g_buf.tz)
    GATHER(seps, g_buf.teps) GATHER(szeta, g_buf.tzeta) GATHER(sm, g_buf.tmass)
    #undef GATHER
    for (int i = 0; i < n_target; ++i) h_gas[i] = sgas[tidx[i]];
    cudaMemcpy(g_buf.tgas, h_gas, n_target, cudaMemcpyHostToDevice);

    static const bool use_fp64 = getenv("SHMEM_CUDA_FP64") != nullptr;
    const int blocks = (n_target + TILE - 1) / TILE;
    if (use_fp64)
      accel_bruteforce_kernel<double><<<blocks, TILE>>>(
        g_buf.tx, g_buf.ty, g_buf.tz, g_buf.teps, g_buf.tzeta, g_buf.tmass, g_buf.tgas,
        g_buf.sx, g_buf.sy, g_buf.sz, g_buf.sm, g_buf.seps, g_buf.szeta, g_buf.sgas,
        g_buf.tidx, n_target, n_source, box, g_buf.ax, g_buf.ay, g_buf.az);
    else
      accel_bruteforce_kernel<float><<<blocks, TILE>>>(
        g_buf.tx, g_buf.ty, g_buf.tz, g_buf.teps, g_buf.tzeta, g_buf.tmass, g_buf.tgas,
        g_buf.sx, g_buf.sy, g_buf.sz, g_buf.sm, g_buf.seps, g_buf.szeta, g_buf.sgas,
        g_buf.tidx, n_target, n_source, box, g_buf.ax, g_buf.ay, g_buf.az);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "shmem_cuda_accel_bruteforce: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    cudaMemcpy(ax_out, g_buf.ax, n_target*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ay_out, g_buf.ay, n_target*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(az_out, g_buf.az, n_target*sizeof(double), cudaMemcpyDeviceToHost);
}
