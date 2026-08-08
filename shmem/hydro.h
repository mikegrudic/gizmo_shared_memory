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

// ---- cubic spline (Monaghan & Lattanzio 1985), 3D, support radius h ----
static inline double kernel_w(double r, double h) {
    double q = r / h;
    if (q >= 1.0) return 0.0;
    const double norm = 8.0 / (M_PI * h * h * h);
    if (q < 0.5) return norm * (1.0 - 6.0 * q * q + 6.0 * q * q * q);
    double u = 1.0 - q;
    return norm * 2.0 * u * u * u;
}
static inline double kernel_dwdh(double r, double h) {   // dW/dh at fixed r
    double q = r / h;
    if (q >= 1.0) return 0.0;
    const double norm = 8.0 / (M_PI * h * h * h);
    // W = norm * f(q); dW/dh = -(3/h) W - (q/h) norm f'(q)
    double f, fp;
    if (q < 0.5) { f = 1.0 - 6.0*q*q + 6.0*q*q*q; fp = -12.0*q + 18.0*q*q; }
    else { double u = 1.0 - q; f = 2.0*u*u*u; fp = -6.0*u*u; }
    return -(3.0 / h) * norm * f - (q / h) * norm * fp;
}

// All particles within radius `rad` of (x,y,z). Appends indices to `out` (not cleared).
// Prune: every particle in a node lies within (node.s) of the node COM by construction of
// s = size + delta, so a node can be skipped when dist(COM, centre) > rad + s.
void ngb_search(const Tree& T, const Particles& P, double x, double y, double z,
                double rad, std::vector<uint32_t>& out, double box = 0.0);

struct DensityResult {
    std::vector<double> h;        // converged support radius
    std::vector<double> rho;      // sum m_j W(r_ij, h_i)
    std::vector<int>    nngb;     // true neighbours inside h (diagnostic)
    std::vector<int>    iters;    // solver iterations (diagnostic)
};

// Solve N_eff(h_i) = des_ngb for every target and return h and rho.
// h0 is the initial guess (per target; pass empty to derive from the mean interparticle spacing).
DensityResult density(const Tree& T, const Particles& P, const std::vector<uint32_t>& targets,
                      double des_ngb, const std::vector<double>& h0, double box = 0.0);

}  // namespace shmem
