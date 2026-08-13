// Controlled A/B: does the node memory layout actually matter for the tree walk?
//
// The first measurement of the SoA walk (387 us/particle) was taken on a workstation whose 15-minute
// load average was 3.80, so it cannot be trusted in absolute terms. This benchmark is built to be
// robust to that:
//
//   * BOTH layouts run in ONE process, INTERLEAVED rep by rep, so background load hits them equally.
//   * The figure of merit is the MINIMUM over reps, not the mean. A contended rep is slow; a clean
//     rep is not. The minimum is the closest estimate of the uncontended cost, and contention can
//     only ever push it up, never down.
//   * Results are also reported per node visit, since the visit count is identical between layouts
//     by construction -- the two walks make exactly the same decisions.

#include "tree.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>

using namespace shmem;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a) {
    return std::chrono::duration<double, std::milli>(clk::now() - a).count();
}

static Particles make_plummer(size_t n, unsigned seed) {
    Particles P; P.x.resize(n); P.y.resize(n); P.z.resize(n);
    P.m.assign(n, 1.0/n); P.soft.assign(n, 1e-3);
    std::mt19937_64 rng(seed); std::uniform_real_distribution<double> U(0,1);
    for (size_t i = 0; i < n; ++i) {
        double m = U(rng)*0.999, r = 1.0/std::sqrt(std::pow(m,-2.0/3.0)-1.0);
        double ct = 2*U(rng)-1, st = std::sqrt(1-ct*ct), ph = 2*M_PI*U(rng);
        P.x[i]=r*st*std::cos(ph); P.y[i]=r*st*std::sin(ph); P.z[i]=r*ct;
    }
    return P;
}

int main(int argc, char** argv) {
    size_t N    = (argc > 1) ? (size_t)atof(argv[1]) : 3'500'000;
    size_t nact = (argc > 2) ? (size_t)atof(argv[2]) : 620;
    int    reps = (argc > 3) ? atoi(argv[3]) : 15;
    int    nt   = omp_get_max_threads();

    Particles P = make_plummer(N, 7);
    Tree T = build(P);
    std::vector<uint32_t> tg(nact);
    std::mt19937_64 rng(3);
    for (size_t i = 0; i < nact; ++i) tg[i] = (uint32_t)(rng() % N);

    printf("  N=%zu  nactive=%zu  threads=%d  reps=%d  nodes=%zu\n", N, nact, nt, reps, T.nnodes());
    printf("  node data: SoA spans %.1f MB over 11 arrays; packed spans %.1f MB in one\n",
           T.nnodes()*(7*8.0+4*4.0)/1e6, T.nnodes()*64.0/1e6);
    printf("  reporting MIN over reps (robust to background load), interleaved AoS/SoA\n\n");

    std::vector<double> ax, ay, az;
    accel(T, P, tg, 0.5, 1.0, ax, ay, az);          // warm both paths
    accel_soa(T, P, tg, 0.5, 1.0, ax, ay, az);

    double best_aos = 1e30, best_soa = 1e30;
    std::vector<double> all_aos, all_soa;
    for (int r = 0; r < reps; ++r) {
        auto t0 = clk::now(); accel(T, P, tg, 0.5, 1.0, ax, ay, az);     double a = ms(t0);
        t0 = clk::now();      accel_soa(T, P, tg, 0.5, 1.0, ax, ay, az); double s = ms(t0);
        all_aos.push_back(a); all_soa.push_back(s);
        best_aos = std::min(best_aos, a); best_soa = std::min(best_soa, s);
    }
    std::sort(all_aos.begin(), all_aos.end());
    std::sort(all_soa.begin(), all_soa.end());

    printf("  %-10s %10s %10s %10s %12s\n", "layout", "min ms", "median", "max", "us/particle");
    printf("  %-10s %10.3f %10.3f %10.3f %12.2f\n", "packed",  best_aos,
           all_aos[reps/2], all_aos.back(), best_aos*1000/nact);
    printf("  %-10s %10.3f %10.3f %10.3f %12.2f\n", "SoA",     best_soa,
           all_soa[reps/2], all_soa.back(), best_soa*1000/nact);
    printf("\n  packed / SoA (min)    = %.3f\n", best_aos / best_soa);
    printf("  spread max/min: packed %.2f, SoA %.2f   (>1.5 means the machine was busy)\n",
           all_aos.back()/best_aos, all_soa.back()/best_soa);
    return 0;
}
