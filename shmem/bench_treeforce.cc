// ONE tree-force evaluation on the real 3.5e6-cell BBB03 state, split by what the walk is asked
// to produce: acceleration alone, acceleration+tidal, acceleration+tidal+jerk. The three share a
// single tree and a single all-active target list, so the difference between them is purely the
// cost of the extra accumulators in the leaf pair loop.
//
// The engine itself is a poor instrument for this: a full run buries the walk under density, the
// gradient pass and the flux loop, and small tests (evrard, 2.7e4 cells) fit in cache and cannot
// show a memory effect at all.
//
// Input is the flat dump of snapshot_000: N, then x,y,z,m,h,vx,vy,vz as contiguous doubles.
#include "tree.h"
#include <chrono>
#include <cstdio>
using namespace shmem;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a) {
    return std::chrono::duration<double, std::milli>(clk::now() - a).count();
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "m50_state.bin";
    const int reps = (argc > 2) ? atoi(argv[2]) : 3;
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

    const double* vel[3] = {vx.data(), vy.data(), vz.data()};
    auto t0 = clk::now();
    Tree T = build(P, nullptr, vel, /*want_vcom=*/true);
    const double tbuild = ms(t0);

    // All-active, in Morton order -- the first tree-force of a run, and the order the engine
    // hands the walk when every particle is active.
    std::vector<uint32_t> targets(T.orderbuf.begin(), T.orderbuf.end());

    printf("N=%llu  tree %zu nodes, build %.0f ms, %d threads, batch 8\n",
           (unsigned long long)N, T.nnodes(), tbuild, omp_get_max_threads());
    printf("\n  %-22s %10s %12s\n", "walk produces", "best ms", "vs accel");

    std::vector<double> ax(N), ay(N), az(N);
    std::vector<SymTensor3d> tt(N);
    std::vector<Vec3d> jk(N);
    double base = 0;
    for (int mode = 0; mode < 3; ++mode) {
        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            auto t = clk::now();
            accel_grouped(T, P, targets, 0.5, 1.0, 8, ax, ay, az,
                          (mode >= 1) ? &tt : nullptr, nullptr, nullptr,
                          (mode >= 2) ? &jk : nullptr, (mode >= 2) ? vel : nullptr);
            best = std::min(best, ms(t));
        }
        if (mode == 0) base = best;
        const char* name[3] = {"accel", "accel+tidal", "accel+tidal+jerk"};
        // Checksums, so a change that is meant to be pure layout can be shown to be pure layout.
        // Printed at full precision: these must match to the last bit across such a change.
        double sa = 0, st = 0, sj = 0;
        for (uint64_t i = 0; i < N; ++i) sa += ax[i] + ay[i] + az[i];
        if (mode >= 1) for (uint64_t i = 0; i < N; ++i) st += tt[i][0][0] + tt[i][1][1] + tt[i][2][2];
        if (mode >= 2) for (uint64_t i = 0; i < N; ++i) sj += jk[i][0] + jk[i][1] + jk[i][2];
        printf("  %-22s %10.0f %11.2fx   sum_a=%.17g sum_t=%.17g sum_j=%.17g\n",
               name[mode], best, best / base, sa, st, sj);
    }
    return 0;
}
