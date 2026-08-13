// How much bigger is our neighbour-search prune volume than the reference's?
//
// GIZMO prunes against the node BOX: geometric centre, half side, then a sphere of
// (0.5 + CUBE_EDGEFACTOR_1) * len -- the cube's circumsphere, 0.866 * len.
// We prune against a sphere centred on the CENTRE OF MASS with radius size + delta (tree.cc:405),
// where size is the full side length and delta = |COM - geometric centre|.
//
// Two over-estimates stack. Using the full side rather than 0.866 of it is a fixed 1.155x in
// radius. Adding delta is the problem-dependent part: delta is ~0 when a node's mass fills it
// uniformly and grows as the mass clusters into a corner, which is why this costs nothing on sedov
// and a great deal on a turbulent cloud.
//
// Reports the ratio unweighted and weighted by node occupancy, since the nodes a search actually
// visits are not a uniform sample.
#include "tree.h"
#include <algorithm>
#include <cstdio>
#include <cmath>
using namespace shmem;

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

    Tree T = build(P);
    const double GIZMO_R = 0.8660254037844386;   // sqrt(3)/2, the cube's circumsphere over its side

    printf("N=%llu  %zu nodes\n\n", (unsigned long long)N, T.nnodes());
    printf("  %-10s %10s %10s %10s %10s %10s\n",
           "weighting", "mean d/len", "med d/len", "p90 d/len", "vol ratio", "nodes");

    for (int weighted = 0; weighted < 2; ++weighted) {
        std::vector<double> ratios;      // (len + delta) / (0.866 len), the RADIUS ratio
        std::vector<double> dovl;
        double wsum = 0, rsum = 0, dsum = 0;
        for (size_t nd = 0; nd < T.nnodes(); ++nd) {
            const double len = T.size[nd];
            if (len <= 0) continue;
            const int cnt = T.phi[nd] - T.plo[nd];
            if (cnt <= 0) continue;
            const double w = weighted ? (double)cnt : 1.0;
            const double d_over_len = T.delta[nd] / len;
            const double rratio = (len + T.delta[nd]) / (GIZMO_R * len);
            wsum += w; rsum += w * rratio; dsum += w * d_over_len;
            ratios.push_back(rratio);
            dovl.push_back(d_over_len);
        }
        std::sort(dovl.begin(), dovl.end());
        std::sort(ratios.begin(), ratios.end());
        const double meanr = rsum / wsum;
        printf("  %-10s %10.3f %10.3f %10.3f %10.2f %10zu\n",
               weighted ? "by count" : "uniform",
               dsum / wsum,
               dovl[dovl.size()/2],
               dovl[(size_t)(dovl.size()*0.9)],
               meanr * meanr * meanr,          // volume goes as the cube of the radius
               ratios.size());
    }

    // The fixed part alone, for reference: what we would gain from the 0.866 factor with delta=0.
    printf("\n  delta=0 alone (full side vs circumsphere): radius %.3fx, volume %.2fx\n",
           1.0 / GIZMO_R, 1.0 / (GIZMO_R * GIZMO_R * GIZMO_R));

    // OCCUPANCY. An octree splits by geometry, not by count: in a clustered field the mass sits in
    // one or two octants per level, so the descent is deep and most leaves come out far below
    // LEAF_MAX. That inflates both the node count and the number of leaves a fixed-radius search
    // must open to gather DesNumNgb, and it is exactly the difference between a uniform test
    // problem and a turbulent cloud. The uniform control below is the same N, same build.
    auto occupancy = [](const Tree& t, const char* tag, uint64_t n) {
        size_t leaves = 0, empty = 0; long long inleaf = 0; std::vector<int> occ;
        for (size_t nd = 0; nd < t.nnodes(); ++nd) {
            if (t.first[nd] >= 0) continue;               // interior
            const int c = t.phi[nd] - t.plo[nd];
            ++leaves; if (c <= 0) { ++empty; continue; }
            inleaf += c; occ.push_back(c);
        }
        std::sort(occ.begin(), occ.end());
        printf("  %-9s nodes=%-9zu leaves=%-9zu empty=%-8zu mean/leaf=%5.2f  med=%3d  "
               "nodes/particle=%.3f\n",
               tag, t.nnodes(), leaves, empty, occ.empty()?0.0:(double)inleaf/occ.size(),
               occ.empty()?0:occ[occ.size()/2], (double)t.nnodes()/(double)n);
    };
    printf("\n  TREE OCCUPANCY (LEAF_MAX=%d)\n", LEAF_MAX);
    occupancy(T, "clustered", N);

    Particles U;                                    // uniform control, same N, same box
    U.x.resize(N); U.y.resize(N); U.z.resize(N); U.m.resize(N, 1.0); U.soft.resize(N, 0.0);
    double lo0=1e300, hi0=-1e300;
    for (uint64_t i = 0; i < N; ++i) { lo0 = std::min(lo0, P.x[i]); hi0 = std::max(hi0, P.x[i]); }
    uint64_t st = 88172645463325252ull;
    auto urand = [&]() { st ^= st<<13; st ^= st>>7; st ^= st<<17; return (double)(st>>11)/9007199254740992.0; };
    for (uint64_t i = 0; i < N; ++i) {
        U.x[i] = lo0 + (hi0-lo0)*urand(); U.y[i] = lo0 + (hi0-lo0)*urand();
        U.z[i] = lo0 + (hi0-lo0)*urand();
    }
    Tree TU = build(U);
    occupancy(TU, "uniform", N);
    return 0;
}
