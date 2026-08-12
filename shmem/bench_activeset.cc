// Why does a step with 10,000 active particles cost more gravity than a step with 3,500,000?
//
// Measured on the M50 run: gravity per step is INVERTED against the active count across the middle
// of the range -- 2488 ms at 0.1-1% active against 2168 ms with everything active. The suspicion is
// the fixed-size batching. accel_grouped decides node opening once per batch of consecutive targets
// using the group's bounding box, so when every particle is active a batch of 8 is spatially tight
// and the criterion accepts aggressively. Take 10k of 3.5e6 and consecutive ACTIVE particles are
// ~350 apart in Morton order, so the group's box spans a large volume, the conservative reduction
// (nearest corner, max softening, min aold) opens nearly everything, and each batch walks most of
// the tree.
//
// This sweeps active-set size against batch size, in the engine's production mode
// (accel+tidal+jerk). SCATTERED is a random subset in Morton order, which is what a deep timebin
// looks like to the walk. COMPACT is a contiguous Morton range of the same size -- the control that
// separates "small" from "scattered". If the mechanism is right, batch=1 beats batch=8 when
// scattered and loses when dense, and the fix is choosing the batch from the active fraction.
#include "tree.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <cmath>
using namespace shmem;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a) {
    return std::chrono::duration<double, std::milli>(clk::now() - a).count();
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "m50_state.bin";
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
    Tree T = build(P, nullptr, vel, /*want_vcom=*/true);
    printf("N=%llu  tree %zu nodes, %d threads, accel+tidal+jerk, theta 0.5\n",
           (unsigned long long)N, T.nnodes(), omp_get_max_threads());

    std::vector<double> ax(N), ay(N), az(N);
    std::vector<SymTensor3d> tt(N);
    std::vector<Vec3d> jk(N);
    // The small sizes matter most: 261 of the M50 run's first 400 steps are under 0.1% active,
    // and the only prior datum below 3500 is an nact=8 measurement in a comment.
    const size_t sizes[] = {32, 128, 512, 3500, 35000, 350000, (size_t)N};
    const int batches[] = {1, 2, 4, 8, 0};   // 0 = node-aligned grouping

    printf("\n  %-9s %-9s %9s %9s %9s %9s %9s %8s %8s\n",
           "nactive", "layout", "b=1", "b=2", "b=4", "b=8", "node", "extent", "best");
    for (size_t nact : sizes) {
        for (int compact = 0; compact < 2; ++compact) {
            // At full size the two layouts are the same set; run it once.
            if (nact == (size_t)N && compact) continue;
            std::vector<uint32_t> targets;
            if (compact) {
                // A REAL TREE NODE's particle range, not an arbitrary Morton window. Morton
                // contiguity only implies spatial compactness for ranges aligned to power-of-8
                // blocks; an arbitrary window at an arbitrary offset can straddle top-level
                // boundaries and span much of the box, which is what the first version of this
                // control did. Pick the node whose particle count is closest to nact.
                size_t bestn = 0; double bestd = 1e300;
                for (size_t nd = 0; nd < T.nnodes(); ++nd) {
                    const double c = (double)(T.phi[nd] - T.plo[nd]);
                    if (c <= 0) continue;
                    const double d = fabs(c - (double)nact);
                    if (d < bestd) { bestd = d; bestn = nd; }
                }
                targets.assign(T.orderbuf.begin() + T.plo[bestn],
                               T.orderbuf.begin() + T.phi[bestn]);
            } else {
                // Random subset, then sorted by Morton rank -- exactly how the engine hands a
                // scattered active set to the walk.
                std::mt19937_64 rng(12345);
                std::vector<uint32_t> pick(N);
                for (uint64_t i = 0; i < N; ++i) pick[i] = (uint32_t)i;
                for (size_t i = 0; i < nact; ++i)
                    std::swap(pick[i], pick[i + rng() % (N - i)]);
                targets.assign(pick.begin(), pick.begin() + nact);
                std::sort(targets.begin(), targets.end(),
                          [&](uint32_t a, uint32_t b) { return T.rank[a] < T.rank[b]; });
            }
            // Characterise the set rather than trusting its label: the mean per-group bounding
            // box relative to the whole box is what actually drives the over-opening.
            double ex0=1e300,ey0=1e300,ez0=1e300,ex1=-1e300,ey1=-1e300,ez1=-1e300;
            for (uint32_t t : targets) {
                ex0=std::min(ex0,P.x[t]); ex1=std::max(ex1,P.x[t]);
                ey0=std::min(ey0,P.y[t]); ey1=std::max(ey1,P.y[t]);
                ez0=std::min(ez0,P.z[t]); ez1=std::max(ez1,P.z[t]);
            }
            const double extent = std::max(std::max(ex1-ex0, ey1-ey0), ez1-ez0);
            double t_of[5]; double best = 1e30;
            for (int bi = 0; bi < 5; ++bi) {
                auto t = clk::now();
                accel_grouped(T, P, targets, 0.5, 1.0, batches[bi], ax, ay, az,
                              &tt, nullptr, nullptr, &jk, vel);
                t_of[bi] = ms(t);
                best = std::min(best, t_of[bi]);
            }
            const char* bn[5] = {"b=1","b=2","b=4","b=8","node"};
            int bw = 0; for (int i=1;i<5;++i) if (t_of[i] < t_of[bw]) bw = i;
            printf("  %-9zu %-9s %9.1f %9.1f %9.1f %9.1f %9.1f %8.3f %8s\n", targets.size(),
                   (nact == (size_t)N) ? "all" : (compact ? "compact" : "scattered"),
                   t_of[0], t_of[1], t_of[2], t_of[3], t_of[4], extent, bn[bw]);
            (void)best;
            fflush(stdout);
        }
    }
    return 0;
}
