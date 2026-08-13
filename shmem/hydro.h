// Shared-memory hydro, stage 1: neighbour search and the density/smoothing-length solve.
//
// This is the half of MFM that touches the tree. The Riemann/flux stage sits on top of the
// neighbour machinery built here, so this file is what soundwave/shocktube/square exercise first.
//
// Conventions follow GIZMO so results can be compared field-by-field:
//   * cubic spline kernel, support radius h (GIZMO's KernelRadius/Hsml is the FULL support radius,
//     not the Gaussian-equivalent scale length)
//   * the smoothing length solves  N_eff(h) = (4pi/3) h^3 sum_j W(r_ij, h) = DesNumNgb  (=32),
//     the same "effective neighbour number" condition as GIZMO's density.cc, iterated per particle
//     with bisection-guarded Newton on the same neighbour set.
//
// Threading: plain `#pragma omp parallel for` over the active list (measured optimal in
// step_overhead.cc). The h-iteration is PER PARTICLE and independent, so unlike GIZMO's
// rank-synchronised iterate-all-then-exchange loop there is no global convergence round: each
// particle bisects privately until done. That removes one of the per-step collective patterns the
// MPI profile showed.

#pragma once
#include "tree.h"

namespace shmem {

// ---- cubic spline (Monaghan & Lattanzio 1985), support radius h, dimension-aware ----
// GIZMO's tier-1 suite tests are 1D and 2D (BOX_SPATIAL_DIMENSION), so the kernel, the
// effective-neighbour ball volume and the E-matrix rank all carry `dim`. Norms for support
// radius h: 1D 4/3/h, 2D 40/(7 pi h^2), 3D 8/(pi h^3).
static inline double kernel_norm(double h, int dim) {
    if (dim == 1) return (4.0/3.0) / h;
    if (dim == 2) return 40.0 / (7.0*M_PI * h * h);
    return 8.0 / (M_PI * h * h * h);
}
static inline double ball_vol(double h, int dim) {       // volume of the unit-d ball of radius h
    if (dim == 1) return 2.0 * h;
    if (dim == 2) return M_PI * h * h;
    return 4.0*M_PI/3.0 * h*h*h;
}
static inline double kernel_w(double r, double h, int dim = 3) {
    double q = r / h;
    if (q >= 1.0) return 0.0;
    const double norm = kernel_norm(h, dim);
    if (q < 0.5) return norm * (1.0 - 6.0 * q * q + 6.0 * q * q * q);
    double u = 1.0 - q;
    return norm * 2.0 * u * u * u;
}
static inline double kernel_dwdr(double r, double h, int dim = 3) {   // dW/dr at fixed h
    double q = r / h;
    if (q >= 1.0) return 0.0;
    const double norm = kernel_norm(h, dim);
    double fp;
    if (q < 0.5) fp = -12.0*q + 18.0*q*q;
    else { double u = 1.0 - q; fp = -6.0*u*u; }
    return norm * fp / h;
}
static inline double kernel_dwdh(double r, double h, int dim = 3) {   // dW/dh at fixed r
    double q = r / h;
    if (q >= 1.0) return 0.0;
    const double norm = kernel_norm(h, dim);
    double f, fp;
    if (q < 0.5) { f = 1.0 - 6.0*q*q + 6.0*q*q*q; fp = -12.0*q + 18.0*q*q; }
    else { double u = 1.0 - q; f = 2.0*u*u*u; fp = -6.0*u*u; }
    return -((double)dim / h) * norm * f - (q / h) * norm * fp;
}

// All particles within `radius` of `centre`. APPENDS indices to `found` (does not clear it).
// Prune: every particle in a node lies within (node.s) of the node COM by construction of
// s = size + delta, plus vmax * t_since_build for motion since the tree was built, so a node can be
// skipped when dist(COM, centre) > radius + tree.open_radius(node).
// SHMEM_NGB_COUNT diagnostic totals; see the definition in hydro.cc.
void ngb_counters(long long& calls, long long& nodes, long long& examined, long long& kept,
                  long long& pad_nodes, bool reset);
void hiter_counters(long long out[8], bool reset);

void ngb_search(const Tree& tree, const Particles& particles, const Vec3d& centre,
                double radius, std::vector<uint32_t>& found, double box = 0.0,
                const LazyDrift* lazy = nullptr);

struct DensityResult {
    std::vector<double> h;        // converged support radius
    std::vector<double> rho;      // sum m_j W(r_ij, h_i)
    std::vector<int>    nngb;     // true neighbours inside h (diagnostic)
    std::vector<int>    iters;    // solver iterations (diagnostic)
};

// SHARED NEIGHBOUR SEARCH. The h solve's LAST iteration is a traversal at exactly the converged
// h -- which is the same search every downstream consumer then wants to repeat. Handing that
// iteration's neighbour list out here lets the whole step run on ONE traversal per target
// instead of one per phase.
//
// Flat CSR: target t owns flat[start[t] .. start[t+1]). Indexed by POSITION IN THE TARGET LIST,
// so with a small active fraction the arrays stay proportional to the work rather than the box.
struct NeighborCache {
    std::vector<uint32_t> flat;
    std::vector<size_t>   start;   // size ntargets+1
    bool valid = false;

    void clear() { flat.clear(); start.clear(); valid = false; }
};

// Solve N_eff(h_i) = des_ngb for every target and return h and rho. `h_start` is the per-target
// initial guess; pass an empty vector to derive one from the mean interparticle spacing.
// When `cache` is non-null it is filled with each target's neighbour list at the CONVERGED h.
DensityResult density(const Tree& tree, const Particles& particles,
                      const std::vector<uint32_t>& targets, double des_ngb, double ngb_tol,
                      const std::vector<double>& h_start, double box = 0.0, int n_dims = 3,
                      NeighborCache* cache = nullptr, const LazyDrift* lazy = nullptr);

}  // namespace shmem
