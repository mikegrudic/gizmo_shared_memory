// Where does the density solve actually spend its time?
//
// Against the reference at matched conditions the ratio is the anomaly: GIZMO's dens+grad costs
// 0.32 of its own gravity on the first all-active step, ours costs 1.15 of ours. That ratio is
// internal to each run, so it survives any difference in machine, thread count, MPI-vs-OpenMP or
// step structure -- something here is disproportionately expensive on its own terms.
//
// Decomposes it on the real 3.5e6-cell state:
//   cold      full solve from the global mean-spacing guess -- what step 0 does
//   warm      full solve seeded with the converged h -- what every later step does
//   traverse  ngb_search alone at the converged h, results discarded -- the traversal floor
//   gather    traversal plus the distance test, no kernel or moment accumulation
// warm - traverse is the arithmetic; traverse is the tree walk. Whichever dominates is the target.
#include "hydro.h"
#include <chrono>
#include <cstdio>
#include <omp.h>
using namespace shmem;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a) {
    return std::chrono::duration<double, std::milli>(clk::now() - a).count();
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "m50_state.bin";
    const double des_ngb = (argc > 2) ? atof(argv[2]) : 32.0;
    const double ngb_tol = (argc > 3) ? atof(argv[3]) : 0.05;
    FILE* fh = fopen(path, "rb");
    if (!fh) { printf("cannot open %s\n", path); return 1; }
    uint64_t N;
    if (fread(&N, 8, 1, fh) != 1) { printf("short read\n"); return 1; }
    Particles P;
    P.x.resize(N); P.y.resize(N); P.z.resize(N); P.m.resize(N); P.soft.resize(N);
    std::vector<double> vx(N), vy(N), vz(N);
    for (auto* a : {&P.x, &P.y, &P.z, &P.m, &P.soft, &vx, &vy, &vz})
        if (fread(a->data(), 8, N, fh) != N) { printf("short read\n"); return 1; }
    fclose(fh);

    Tree T = build(P);
    std::vector<uint32_t> targets(T.orderbuf.begin(), T.orderbuf.end());
    printf("N=%llu  %zu nodes  %d threads  DesNumNgb=%.0f tol=%.3g\n",
           (unsigned long long)N, T.nnodes(), omp_get_max_threads(), des_ngb, ngb_tol);

    auto t0 = clk::now();
    DensityResult cold = density(T, P, targets, des_ngb, ngb_tol, {}, 0.0, 3, nullptr, nullptr);
    const double t_cold = ms(t0);

    t0 = clk::now();
    DensityResult warm = density(T, P, targets, des_ngb, ngb_tol, cold.h, 0.0, 3, nullptr, nullptr);
    const double t_warm = ms(t0);

    // Traversal floor: one ngb_search per target at the already-converged h. Same tree, same
    // radii, same order -- everything the warm solve does except the kernel arithmetic and the
    // Newton bookkeeping.
    t0 = clk::now();
    long long kept_total = 0;
    #pragma omp parallel reduction(+:kept_total)
    {
        std::vector<uint32_t> ngb;
        #pragma omp for schedule(dynamic, 256)
        for (size_t k = 0; k < targets.size(); ++k) {
            ngb.clear();
            ngb_search(T, P, P.pos(targets[k]), cold.h[k], ngb, 0.0, nullptr);
            kept_total += (long long)ngb.size();
        }
    }
    const double t_trav = ms(t0);

    // Traversal plus the distance test that the density loop would do, but no accumulation.
    t0 = clk::now();
    double sink = 0;
    #pragma omp parallel reduction(+:sink)
    {
        std::vector<uint32_t> ngb;
        #pragma omp for schedule(dynamic, 256)
        for (size_t k = 0; k < targets.size(); ++k) {
            ngb.clear();
            const uint32_t i = targets[k];
            const Vec3d pi = P.pos(i);
            ngb_search(T, P, pi, cold.h[k], ngb, 0.0, nullptr);
            for (uint32_t j : ngb) sink += (P.pos(j) - pi).norm_sq();
        }
    }
    const double t_gather = ms(t0);

    // Checksums: the new prune is tighter but still one-sided, so it must find the SAME neighbour
    // lists in the SAME order -- rejecting a node that held nothing changes neither. h and rho are
    // therefore expected bit-identical across the change, and anything else is a dropped neighbour.
    double sh = 0, sr = 0;
    for (size_t k = 0; k < cold.h.size(); ++k) { sh += cold.h[k]; sr += cold.rho[k]; }
    printf("  checksum sum_h=%.17g  sum_rho=%.17g\n", sh, sr);

    double miters = 0; for (int it : cold.iters) miters += it;
    double witers = 0; for (int it : warm.iters) witers += it;
    printf("\n  %-10s %10s %10s   %s\n", "phase", "ms", "vs trav", "what it measures");
    printf("  %-10s %10.0f %10.2f   full solve, mean iters=%.2f\n",
           "cold", t_cold, t_cold/t_trav, miters/cold.iters.size());
    printf("  %-10s %10.0f %10.2f   full solve, mean iters=%.2f\n",
           "warm", t_warm, t_warm/t_trav, witers/warm.iters.size());
    printf("  %-10s %10.0f %10.2f   ngb_search only, %.1f kept/target\n",
           "traverse", t_trav, 1.0, (double)kept_total/targets.size());
    printf("  %-10s %10.0f %10.2f   + distance test, no accumulation\n",
           "gather", t_gather, t_gather/t_trav);
    printf("\n  arithmetic (warm - gather) = %.0f ms;  traversal = %.0f ms (%.0f%% of warm)\n",
           t_warm - t_gather, t_trav, 100.0*t_trav/t_warm);
    (void)sink;
    return 0;
}
