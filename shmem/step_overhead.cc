// Measure the FLOOR on per-step overhead for a shared-memory work-distribution layer.
//
// Why this exists: profiling the MPI code showed 85% of the post-sink step is synchronisation, and
// Stage 0 showed GIZMO's existing OpenMP is 7.4x WORSE than MPI at the same core count. So the
// rewrite stands or falls on the work-distribution layer, and its binding constraint is not
// throughput but overhead: with ~620 active elements of 3.5e6, 96 threads get 6.5 items each, and
// fork/join can easily cost more than the work it distributes.
//
// The question this answers, before any GIZMO code is touched:
//     what does one synchronised step cost when there is (almost) nothing to do?
//
// Targets to beat, from the measured MPI run:
//     8.85 ms/step   current, 96 MPI ranks
//     0.68 ms/step   the aggregate-compute bound (65.6 core-ms perfectly parallelised)
//
// Two strategies are compared, because the difference between them IS the design decision:
//   FORK   `#pragma omp parallel for` per phase -- a new parallel region every step, which is what
//          GIZMO does today.
//   POOL   one persistent `#pragma omp parallel` outside the step loop; steps are delimited by
//          barriers and work is claimed from an atomic counter. No fork/join in the step loop.
//   ADAPT  width chosen from the work available: threads = clamp(nactive/ITEMS_PER_THREAD, 1, max).
//          The 32-thread rows of the first run reversed at ~40 items/thread, so distributing 620
//          items over 96 threads is expected to cost more than it saves. This measures that rather
//          than assuming it.
//
// The synthetic kernel mimics a tree walk's MEMORY behaviour rather than its arithmetic: W
// dependent random accesses into an array far larger than L3, so each is a likely cache miss. That
// is the property that makes GIZMO's walk expensive; reproducing the flops would not.
//
//   g++ -O2 -fopenmp -march=native -o step_overhead step_overhead.cc
//   ./step_overhead [nthreads_max]

#include <omp.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

static const size_t NPART = 3'500'000;   // the BBB03 fiducial cell count
static const int    WALK  = 400;         // memory touches per active element ~ one tree walk
static const int    ITEMS_PER_THREAD = 40; // width cap for ADAPT; ~where the first sweep reversed         // memory touches per active element ~ one tree walk

// Dependent chain of random accesses: each index derives from the previous value, so the loads
// cannot be prefetched or reordered away. Returns a value so the work cannot be optimised out.
static inline double fake_walk(const double* data, size_t n, uint64_t seed) {
    uint64_t h = seed * 0x9E3779B97F4A7C15ull + 1;
    double acc = 0.0;
    for (int k = 0; k < WALK; ++k) {
        h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 27;
        size_t i = (size_t)(h % n);
        acc += data[i];
        h += (uint64_t)acc;              // dependency: next address needs this load
    }
    return acc;
}

int main(int argc, char** argv) {
    int tmax = (argc > 1) ? atoi(argv[1]) : omp_get_max_threads();

    std::vector<double> data(NPART);
    std::mt19937_64 rng(12345);
    for (size_t i = 0; i < NPART; ++i) data[i] = (double)(rng() % 1000) * 1e-3;

    const size_t actives[] = {0, 620, 6200, 62000};
    // Bound the work per measurement: cost is nact*WALK*nsteps, which at the top of the range would
    // otherwise run for minutes. The overhead floor (nact=0) needs many steps to resolve; the loaded
    // cases need only enough to average out jitter.
    auto steps_for = [](size_t nact) -> int {
        if (nact == 0) return 2000;
        long long n = 20'000'000LL / (long long)(nact * WALK);
        return (int)(n < 5 ? 5 : (n > 200 ? 200 : n));
    };

    printf("  %zu particles, %d memory touches per active element; steps auto-scaled per case\n",
           NPART, WALK);
    printf("  targets: MPI today 8.85 ms/step   aggregate-compute bound 0.68 ms/step\n\n");
    printf("  %-6s %-8s %8s %10s %10s %10s %6s %10s\n", "thr", "nactive", "steps", "FORK ms", "POOL ms", "ADAPT ms", "eff", "best/FORK");

    for (size_t ai = 0; ai < sizeof(actives)/sizeof(actives[0]); ++ai) {
        size_t nact = actives[ai];
        for (int nt = 1; nt <= tmax; nt *= 2) {
            omp_set_num_threads(nt);
            volatile double sink = 0.0;

            // ---- FORK: a fresh parallel region every step ----
            const int nsteps = steps_for(nact);
            for (int s = 0; s < 5; ++s) {           // warm up
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < nact; ++i) sink += fake_walk(data.data(), NPART, i + s);
            }
            auto t0 = clk::now();
            for (int s = 0; s < nsteps; ++s) {
                double red = 0.0;
                #pragma omp parallel for schedule(static) reduction(+:red)
                for (size_t i = 0; i < nact; ++i) red += fake_walk(data.data(), NPART, i + s);
                sink += red;
            }
            double fork_ms = ms_since(t0) / nsteps;

            // ---- POOL: one region for all steps, barrier-delimited ----
            std::atomic<size_t> cursor{0};
            double pool_total = 0.0;
            t0 = clk::now();
            #pragma omp parallel
            {
                double mine = 0.0;
                for (int s = 0; s < nsteps; ++s) {
                    #pragma omp single
                    cursor.store(0, std::memory_order_relaxed);
                    #pragma omp barrier
                    for (;;) {                       // claim work in small batches
                        size_t i = cursor.fetch_add(16, std::memory_order_relaxed);
                        if (i >= nact) break;
                        size_t hi = i + 16 < nact ? i + 16 : nact;
                        for (; i < hi; ++i) mine += fake_walk(data.data(), NPART, i + s);
                    }
                    #pragma omp barrier              // end of step: all threads synchronised
                }
                #pragma omp atomic
                pool_total += mine;
            }
            double pool_ms = ms_since(t0) / nsteps;
            sink += pool_total;

            // ---- ADAPT: fork per step, but only as wide as the work justifies ----
            int eff = (int)(nact / ITEMS_PER_THREAD);
            if (eff < 1) eff = 1;
            if (eff > nt) eff = nt;
            for (int s = 0; s < 5; ++s) {
                #pragma omp parallel for schedule(static) num_threads(eff)
                for (size_t i = 0; i < nact; ++i) sink += fake_walk(data.data(), NPART, i + s);
            }
            t0 = clk::now();
            for (int s = 0; s < nsteps; ++s) {
                double red = 0.0;
                #pragma omp parallel for schedule(static) num_threads(eff) reduction(+:red)
                for (size_t i = 0; i < nact; ++i) red += fake_walk(data.data(), NPART, i + s);
                sink += red;
            }
            double adapt_ms = ms_since(t0) / nsteps;

            double best = pool_ms < adapt_ms ? pool_ms : adapt_ms;
            printf("  %-6d %-8zu %8d %10.4f %10.4f %10.4f %6d %10.2f%s\n", nt, nact, nsteps,
                   fork_ms, pool_ms, adapt_ms, eff, fork_ms > 0 ? best / fork_ms : 0.0,
                   (nact == 0 ? "   <- pure overhead floor" : ""));
            if (sink == 12345.678) printf("");      // keep `sink` live
        }
        printf("\n");
    }
    return 0;
}
