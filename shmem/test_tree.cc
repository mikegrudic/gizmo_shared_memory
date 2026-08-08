// Validate and benchmark the shared-memory tree.
//
// Correctness first, and against BRUTE FORCE rather than against GIZMO: a direct O(N*M) sum has no
// shared failure mode with the tree, so agreement is real evidence. Comparing two tree codes would
// hide an opening-criterion bug that both share.
//
// Then the benchmark that matters: accel() over a SMALL active list against a LARGE particle set,
// because that is the post-sink regime (~620 active of 3.5e6) where the MPI code spends 8.85 ms/step
// and ~85% of the machine sits in __sched_yield.

#include "tree.h"
#include <chrono>
#include <cstdio>
#include <random>

using namespace shmem;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a) {
    return std::chrono::duration<double, std::milli>(clk::now() - a).count();
}

// Plummer sphere: centrally concentrated, so the tree is genuinely unbalanced -- a uniform cube
// would flatter the opening criterion and hide errors in `delta`.
static Particles make_plummer(size_t n, double a, unsigned seed) {
    Particles P; P.x.resize(n); P.y.resize(n); P.z.resize(n); P.m.assign(n, 1.0/n); P.soft.assign(n, 1e-3*a);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (size_t i = 0; i < n; ++i) {
        double m = U(rng) * 0.999;
        double r = a / std::sqrt(std::pow(m, -2.0/3.0) - 1.0);
        double ct = 2*U(rng) - 1, st = std::sqrt(1 - ct*ct), ph = 2*M_PI*U(rng);
        P.x[i] = r*st*std::cos(ph); P.y[i] = r*st*std::sin(ph); P.z[i] = r*ct;
    }
    return P;
}

int main(int argc, char** argv) {
    const double G = 1.0;

    // ---------- correctness ----------
    printf("  === force accuracy vs direct summation (Plummer, N=20000) ===\n");
    printf("  %8s %14s %14s %12s\n", "theta", "median relerr", "99th pct", "max");
    {
        size_t N = 20000;
        Particles P = make_plummer(N, 1.0, 42);
        std::vector<uint32_t> tg(2000);
        for (size_t i = 0; i < tg.size(); ++i) tg[i] = (uint32_t)(i * (N / tg.size()));
        std::vector<double> bx, by, bz;
        accel_brute(P, tg, G, bx, by, bz);

        for (double theta : {0.8, 0.5, 0.35, 0.2}) {
            Tree T = build(P);
            std::vector<double> ax, ay, az;
            accel(T, P, tg, theta, G, ax, ay, az);
            std::vector<double> err(tg.size());
            for (size_t i = 0; i < tg.size(); ++i) {
                double dx = ax[i]-bx[i], dy = ay[i]-by[i], dz = az[i]-bz[i];
                double b = std::sqrt(bx[i]*bx[i] + by[i]*by[i] + bz[i]*bz[i]);
                err[i] = b > 0 ? std::sqrt(dx*dx+dy*dy+dz*dz)/b : 0.0;
            }
            std::sort(err.begin(), err.end());
            printf("  %8.2f %14.3e %14.3e %12.3e\n", theta,
                   err[err.size()/2], err[(size_t)(0.99*err.size())], err.back());
        }
    }

    // ---------- scaling ----------
    size_t N = (argc > 1) ? (size_t)atof(argv[1]) : 3'500'000;
    printf("\n  === build + walk, N=%zu (the BBB03 fiducial cell count) ===\n", N);
    Particles P = make_plummer(N, 1.0, 7);
    int tmax = omp_get_max_threads();

    printf("\n  %-6s %12s\n", "thr", "build ms");
    for (int nt = 1; nt <= tmax; nt *= 2) {
        omp_set_num_threads(nt);
        auto t0 = clk::now(); Tree T = build(P); double b = ms(t0);
        printf("  %-6d %12.1f%s\n", nt, b, nt == 1 ? "   <- build is still serial (std::sort + recursion)" : "");
    }

    omp_set_num_threads(tmax);
    Tree T = build(P);
    printf("  tree: %zu nodes for %zu particles\n", T.nnodes(), N);

    printf("\n  %-10s %-6s %12s %14s   vs MPI 8.85 ms/step\n", "nactive", "thr", "walk ms", "us/particle");
    for (size_t nact : {(size_t)620, (size_t)6200, (size_t)62000}) {
        std::vector<uint32_t> tg(nact);
        std::mt19937_64 rng(3);
        for (size_t i = 0; i < nact; ++i) tg[i] = (uint32_t)(rng() % N);
        for (int nt = 1; nt <= tmax; nt *= 2) {
            omp_set_num_threads(nt);
            std::vector<double> ax, ay, az;
            accel(T, P, tg, 0.5, G, ax, ay, az);                 // warm
            auto t0 = clk::now();
            int reps = nact < 5000 ? 20 : 3;
            for (int r = 0; r < reps; ++r) accel(T, P, tg, 0.5, G, ax, ay, az);
            double w = ms(t0) / reps;
            printf("  %-10zu %-6d %12.4f %14.3f   %s\n", nact, nt, w, w*1000/nact,
                   (nact==620 && nt==tmax) ? "<- the post-sink regime" : "");
        }
        printf("\n");
    }
    return 0;
}
