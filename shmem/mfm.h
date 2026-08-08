// MFM (meshless finite mass) hydro on the shared-memory tree. Hopkins (2015) discretisation:
//
//   n_i   = sum_j W(r_ij, h_i)                          number density; V_i = 1/n_i, rho_i = m_i V_i^-1
//   E_i   = sum_j (x_j-x_i)(x_j-x_i)^T W_ij(h_i)        moments matrix; B_i = E_i^-1
//   grad f|_i = B_i sum_j (f_j - f_i)(x_j - x_i) W_ij(h_i)     exact for linear fields
//   A_ij  = V_i B_i (x_j-x_i) psi_j(x_i) + V_j B_j (x_j-x_i) psi_i(x_j),  psi_j(x_i)=W_ij(h_i)/n_i
//           effective face; antisymmetric (A_ji = -A_ij), so pairwise conservation is exact.
//
// Riemann: HLLC solved in the frame of the face, taking the face velocity equal to the contact
// speed S*. Then the mass flux VANISHES identically -- that choice is what makes the scheme
// Lagrangian and mass-per-particle constant -- and the remaining fluxes in the lab frame are
//   dP/dt = -|A| P* nhat        dE/dt = -|A| P* (v_face . nhat)
//
// Reconstruction: linear, with a pairwise minmod limit at each face (no per-particle scalar
// limiter yet). Time integration: global-timestep MUSCL-Hancock -- primitives predicted a half
// step with the Lagrangian derivatives (Drho/Dt = -rho div v, Dv/Dt = -grad P/rho,
// Du/Dt = -(P/rho) div v), then one flux evaluation. Second order in smooth flow.
//
// Deliberately NOT here yet: individual timesteps, self-gravity coupling, wakeups. The tier-1
// tests (soundwave, shocktube, square) validate exactly this much.

#pragma once
#include "hydro.h"

namespace shmem {

struct Sim {
    Particles P;                       // positions + masses (+ soft, unused here)
    std::vector<double> vx, vy, vz;    // velocities
    std::vector<double> u;             // specific internal energy
    double gamma = 5.0 / 3.0;
    double box   = 0.0;                // >0: periodic cube [0, box)^3
    double des_ngb = 32.0;
    double cfl     = 0.25;

    // derived per step
    std::vector<double> h, ninv, rho, press;

    size_t size() const { return P.size(); }
};

// One MUSCL-Hancock step at global dt; returns the dt actually taken (min of CFL and dt_max).
double mfm_step(Sim& S, double dt_max);

// Diagnostics used by the tests.
struct Conserved { double mass, px, py, pz, E; };
Conserved totals(const Sim& S);

// Face-closure check: max_i |sum_j A_ij| / max|A| over a sample of particles. Discrete surface
// integral of a closed volume; large values mean broken faces, not just inaccuracy.
double face_closure(Sim& S, int nsample);

}  // namespace shmem
