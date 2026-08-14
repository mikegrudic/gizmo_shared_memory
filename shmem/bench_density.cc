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
//
// LEAF-GROUPED GATHER: NEGATIVE RESULT (see the report below). Gravity's node-aligned grouping
// wins up to 245x by avoiding bbox blowup when a fixed batch straddles scattered targets --
// ngb_search never batches in the first place, so it has no such catastrophe to fix. What
// leaf-grouping could still buy for density is sharing the TRAVERSAL when a leaf's targets want
// overlapping candidates. Measured (CLUSTER=<k> env var, k = 100/1000/20000/all): the sum/union
// redundancy ratio is 1.76-1.90x EVERYWHERE -- all-active, sparse, or tightly clustered around a
// forming sink alike. That stability says it is a GEOMETRIC CONSTANT (leaf volume is order the
// search sphere's, since LEAF_MAX=16 vs DesNumNgb=32), not a function of active-set size or
// clustering, so it will not improve by chasing a more favourable state. Even a zero-overhead
// shared gather caps out around (1 - 1/1.9) * (traversal's share of warm) -- roughly 20-25% of
// the density solve -- and a real implementation's union/filter overhead would erase a good
// fraction of that. Not built; CACHE_NEIGHBORS (real, measured 31% of the whole hydro phase, no
// implementation risk) was the better trade. See project-grouped-ngb-negative-result memory for
// the related batching finding this extends.
#include "hydro.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <omp.h>
#include <vector>
using namespace shmem;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a) {
    return std::chrono::duration<double, std::milli>(clk::now() - a).count();
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "m50_state.bin";
    const double des_ngb = (argc > 2) ? atof(argv[2]) : 32.0;
    const double ngb_tol = (argc > 3) ? atof(argv[3]) : 0.05;
    // CLUSTER=<k>: restrict targets to the k particles closest to the densest point, emulating
    // the SPARSE, SPATIALLY-CLUSTERED active set a forming sink actually produces (a deep
    // timebin's actives are the collapsing clump, not a random k of the box) -- the all-active
    // default below instead measures the diffuse, near-uniform case, which is a different
    // regime for how much a leaf-grouped gather could share.
    const long long cluster_k = (getenv("CLUSTER")) ? atoll(getenv("CLUSTER")) : 0;
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

    if (cluster_k > 0 && (uint64_t)cluster_k < N) {
        // Densest point: one quick density solve on everyone (needed for the comparison anyway
        // via `cold` below is too late -- do a cheap standalone probe here), then keep the
        // k closest particles to whichever target has the smallest h (smallest h <=> most
        // crowded neighbourhood <=> where a sink would form).
        DensityResult probe = density(T, P, targets, des_ngb, ngb_tol, {}, 0.0, 3, nullptr, nullptr);
        size_t densest = 0;
        for (size_t k = 1; k < probe.h.size(); ++k) if (probe.h[k] < probe.h[densest]) densest = k;
        const Vec3d centre = P.pos(targets[densest]);
        std::vector<std::pair<double,uint32_t>> by_dist(N);
        for (uint64_t i = 0; i < N; ++i)
            by_dist[i] = {(P.pos(i) - centre).norm_sq(), (uint32_t)i};
        std::partial_sort(by_dist.begin(), by_dist.begin() + cluster_k, by_dist.end());
        targets.resize(cluster_k);
        for (long long k = 0; k < cluster_k; ++k) targets[k] = by_dist[k].second;
        // Re-Morton-sort the subset so the leaf-grouping below sees the same order the real
        // active-set path (accel_grouped's targets) would produce.
        std::sort(targets.begin(), targets.end(),
                  [&](uint32_t a, uint32_t b) { return T.rank[a] < T.rank[b]; });
        printf("CLUSTER mode: %lld targets nearest the densest point (h=%.3g)\n",
               cluster_k, probe.h[densest]);
    }

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

    // LEAF-GROUPED CANDIDATE GATHER -- is there anything to share?
    //
    // Gravity's node-aligned grouping (accel_grouped) wins because the ACCEPT/OPEN decision at a
    // node is nearly the same for every target in a leaf, so one traversal serves the whole group.
    // A density search has no such shared decision: each target's h differs, and the final
    // ANSWER is a per-particle distance test, not a per-node approximation. What a shared gather
    // could still buy is the TRAVERSAL -- the descent from the root through the group's common
    // ancestors -- if the leaf's members mostly end up wanting the same candidate PARTICLES.
    // Measure that ceiling directly: with the CONVERGED h (no iteration, so this isolates the
    // geometry from the solver), compare each leaf's SUM of per-target candidate-set sizes
    // (examined, not kept -- examined is what a shared gather would need to cover) against the
    // UNION of those sets. redundancy = sum/union; 1.0 means no sharing is possible (every
    // target's candidates are disjoint), N means a shared gather could in principle replace N
    // searches with one.
    if (T.leaf_of.size() == N) {
        // Bucket targets by leaf using the SAME grouping accel_grouped relies on: targets are
        // already in Morton order (orderbuf), so a leaf's members are contiguous.
        std::vector<size_t> gstart(1, 0);
        for (size_t i = 1; i < targets.size(); ++i)
            if (T.leaf_of[targets[i]] != T.leaf_of[targets[i-1]]) gstart.push_back(i);
        gstart.push_back(targets.size());
        const size_t n_groups = gstart.size() - 1;

        long long sum_examined = 0, union_examined = 0, groups_ge2 = 0, singletons = 0;
        double t_shared_floor_ms = 0;
        auto tg0 = clk::now();
        #pragma omp parallel reduction(+:sum_examined,union_examined,groups_ge2,singletons)
        {
            std::vector<uint32_t> ngb;
            std::vector<uint32_t> group_union;
            #pragma omp for schedule(dynamic, 64)
            for (size_t g = 0; g < n_groups; ++g) {
                const size_t lo = gstart[g], hi = gstart[g+1];
                if (hi - lo < 2) { singletons++; continue; }
                groups_ge2++;
                group_union.clear();
                for (size_t k = lo; k < hi; ++k) {
                    ngb.clear();
                    ngb_search(T, P, P.pos(targets[k]), cold.h[k], ngb, 0.0, nullptr);
                    sum_examined += (long long)ngb.size();
                    group_union.insert(group_union.end(), ngb.begin(), ngb.end());
                }
                std::sort(group_union.begin(), group_union.end());
                union_examined += (long long)(std::unique(group_union.begin(), group_union.end())
                                              - group_union.begin());
            }
        }
        t_shared_floor_ms = ms(tg0);
        printf("\n  LEAF-GROUPED GATHER CEILING (converged h, no iteration):\n");
        printf("    %zu leaf groups, %lld singleton (no group to share), %lld with >=2 targets\n",
               n_groups, singletons, groups_ge2);
        if (union_examined > 0)
            printf("    sum(examined)=%lld  union(examined)=%lld  redundancy=%.2fx"
                   "  (upper bound on traversal-sharing speedup)\n",
                   sum_examined, union_examined, (double)sum_examined/union_examined);
        printf("    wall time to compute this (per-target search, no sharing) = %.0f ms"
               " -- cf. traverse %.0f ms\n", t_shared_floor_ms, t_trav);
    } else {
        printf("\n  (leaf_of unavailable or size mismatch -- skipping overlap measurement)\n");
    }
    return 0;
}
