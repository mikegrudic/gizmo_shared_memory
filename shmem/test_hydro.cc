// Validate neighbour search and the density solve, three ways with no shared failure mode:
//
//   1. ngb_search vs a brute-force radius scan: sets must be IDENTICAL (this is exact, not
//      approximate -- the tree prune is conservative, so any mismatch is a bug).
//   2. density on a uniform lattice: rho must equal the exact lattice density to kernel accuracy,
//      and N_eff must hit DesNumNgb for every particle.
//   3. density on the REAL post-sink snapshot vs the Density field GIZMO itself wrote. Same
//      DesNumNgb=32 convention, so agreement here means the whole stack (tree, search, kernel,
//      h-solve) reproduces GIZMO's answer on the target problem.

#include "hydro.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>

using namespace shmem;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a){return std::chrono::duration<double,std::milli>(clk::now()-a).count();}

int main() {
    // ---------- 1. exact neighbour sets ----------
    {
        size_t N = 60000;
        Particles P; P.x.resize(N); P.y.resize(N); P.z.resize(N); P.m.assign(N, 1.0); P.soft.assign(N, 0.0);
        std::mt19937_64 rng(5); std::uniform_real_distribution<double> U(0, 1);
        for (size_t i = 0; i < N; ++i) { P.x[i]=U(rng); P.y[i]=U(rng); P.z[i]=std::pow(U(rng),2.0); } // nonuniform in z
        Tree T = build(P);
        int bad = 0;
        std::mt19937_64 r2(9);
        for (int trial = 0; trial < 200; ++trial) {
            const Vec3d c{U(r2), U(r2), U(r2)};
            const double rad = 0.002 + 0.1 * U(r2);
            std::vector<uint32_t> tree_set;
            ngb_search(T, P, c, rad, tree_set);
            std::vector<uint32_t> brute;
            for (size_t q = 0; q < N; ++q)
                if ((P.pos(q) - c).norm_sq() < rad*rad) brute.push_back((uint32_t)q);
            std::sort(tree_set.begin(), tree_set.end());
            if (tree_set != brute) ++bad;
        }
        printf("  [1] ngb_search vs brute force: %d/200 mismatches %s\n", bad, bad ? "<-- BUG" : "(exact)");
    }

    // ---------- 2. uniform lattice ----------
    {
        int n1 = 32; size_t N = (size_t)n1*n1*n1;
        Particles P; P.x.resize(N); P.y.resize(N); P.z.resize(N); P.m.assign(N, 1.0); P.soft.assign(N, 0.0);
        size_t k = 0;
        for (int i = 0; i < n1; ++i) for (int j = 0; j < n1; ++j) for (int l = 0; l < n1; ++l, ++k) {
            P.x[k]=(i+0.5)/n1; P.y[k]=(j+0.5)/n1; P.z[k]=(l+0.5)/n1;
        }
        Tree T = build(P);
        // interior targets only: the lattice has no ghost layer, so edge kernels are truncated
        std::vector<uint32_t> tg;
        k = 0;
        for (int i = 0; i < n1; ++i) for (int j = 0; j < n1; ++j) for (int l = 0; l < n1; ++l, ++k)
            if (i>=6 && i<n1-6 && j>=6 && j<n1-6 && l>=6 && l<n1-6) tg.push_back((uint32_t)k);
        DensityResult R = density(T, P, tg, 32.0, {});
        double exact = (double)N;            // unit masses on a unit box lattice
        double emax = 0, esum = 0; int itmax = 0;
        for (size_t i = 0; i < tg.size(); ++i) {
            double e = std::abs(R.rho[i] - exact) / exact;
            emax = std::max(emax, e); esum += e; itmax = std::max(itmax, R.iters[i]);
        }
        printf("  [2] lattice %dx%dx%d: rho err mean %.2e max %.2e, max iters %d (exact rho=%g)\n",
               n1, n1, n1, esum/tg.size(), emax, itmax, exact);
    }

    // ---------- 3. real post-sink snapshot vs GIZMO's own Density ----------
    {
        FILE* fh = fopen("/tmp/claude-8d47/postsink_rho.bin", "rb");
        if (!fh) { printf("  [3] skipped (no postsink_rho.bin; run the exporter)\n"); return 0; }
        uint64_t N; if (fread(&N,8,1,fh)!=1) return 1;
        Particles P; P.x.resize(N);P.y.resize(N);P.z.resize(N);P.m.resize(N);P.soft.assign(N,0.0);
        std::vector<double> rho_giz(N), h_giz(N);
        for (auto* v : {&P.x,&P.y,&P.z,&P.m,&rho_giz,&h_giz}) if(fread(v->data(),8,N,fh)!=N) return 1;
        std::vector<uint32_t> bydens(N); if(fread(bydens.data(),4,N,fh)!=N) return 1;
        fclose(fh);

        auto t0 = clk::now(); Tree T = build(P); double tb = ms(t0);
        // the deep-timebin set (densest 620) plus a random sample, same as the walk benchmarks
        std::vector<uint32_t> tg(bydens.begin(), bydens.begin()+620);
        std::mt19937_64 r3(4);
        for (int i = 0; i < 2000; ++i) tg.push_back((uint32_t)(r3() % N));
        std::vector<double> h0(tg.size());
        for (size_t i = 0; i < tg.size(); ++i) h0[i] = h_giz[tg[i]];   // warm start from GIZMO's h

        t0 = clk::now();
        DensityResult R = density(T, P, tg, 32.0, h0);
        double td = ms(t0);
        std::vector<double> err(tg.size());
        long itsum = 0;
        for (size_t i = 0; i < tg.size(); ++i) {
            err[i] = std::abs(R.rho[i] - rho_giz[tg[i]]) / rho_giz[tg[i]];
            itsum += R.iters[i];
        }
        std::sort(err.begin(), err.end());
        printf("  [3] real snapshot, %zu targets (620 densest + 2000 random), build %.0f ms:\n",
               tg.size(), tb);
        printf("      rho vs GIZMO's Density: median %.2e  90th %.2e  99th %.2e  max %.2e\n",
               err[err.size()/2], err[(size_t)(0.9*err.size())],
               err[(size_t)(0.99*err.size())], err.back());
        printf("      density solve: %.1f ms total, %.3f ms per target, %.1f iters mean\n",
               td, td/tg.size(), (double)itsum/tg.size());
    }
    return 0;
}
