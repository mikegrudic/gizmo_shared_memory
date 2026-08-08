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
    Particles P;                       // positions + masses + gravitational softening
    std::vector<double> vx, vy, vz;    // velocities
    std::vector<double> u;             // specific internal energy
    double gamma = 5.0 / 3.0;
    int    dim   = 3;                  // 1/2/3; matches BOX_SPATIAL_DIMENSION in the suite configs
    double box   = 0.0;                // >0: periodic cube [0, box)^3
    double des_ngb = 32.0;
    double cfl     = 0.25;

    // ---- self-gravity (off unless gravity_on; SELFGRAVITY_OFF in the suite configs) ----
    bool   gravity_on = false;
    double G          = 1.0;           // GravityConstantInternal
    double theta      = 0.5;           // Barnes-Hut opening angle
    double soft_min   = 0.0;           // floor on the gas softening (SofteningGas)
    bool   adaptive_soft = true;       // ADAPTIVE_GRAVSOFT_FORGAS: soften on h, not a fixed length
    double eta_grav   = 0.025;         // accuracy of the dt = sqrt(2 eta eps / |a|) criterion
    std::vector<Vec3d> a_grav;         // acceleration at the CURRENT positions; see mfm_step
    double pending_half_kick = 0.0;    // dt/2 owed from the previous step's closing kick

    // derived per step
    std::vector<double> h, ninv, rho, press;

    size_t size() const { return P.size(); }
};

// One MUSCL-Hancock step at global dt; returns the dt actually taken (min of CFL and dt_max).
//
// With gravity_on this is a leapfrog KDK, arranged so that only ONE tree walk per step is needed.
// The trick is that a step's closing half-kick and the next step's opening half-kick both use the
// acceleration at the SAME instant -- the shared sync point -- so they can be applied together from
// one evaluation. `pending_half_kick` carries the dt/2 owed by the previous step; a fresh Sim starts
// at 0, which makes the very first step a correct half-step opening.
double mfm_step(Sim& sim, double dt_max);

// Diagnostics used by the tests.
struct Conserved { double mass, px, py, pz, E; };
Conserved totals(const Sim& sim);

// Face-closure check: max_i |sum_j A_ij| / max|A| over a sample of particles. Discrete surface
// integral of a closed volume; large values mean broken faces, not just inaccuracy.
double face_closure(Sim& sim, int nsample);

// Gradient-exactness check. The matrix gradient B_i sum_j (f_j-f_i)(x_j-x_i) W_ij reproduces any
// LINEAR field exactly, for ANY neighbour configuration whose E_i is invertible -- it is the
// defining property of the discretisation, not an accuracy statement. So it is the sharpest
// available test of the E/B algebra: a transposed or mis-indexed tensor element still looks
// plausible in a convergence study but destroys exactness here immediately.
// Overwrites the sim's primitive fields with known linear profiles. Non-periodic only (a linear
// field is discontinuous across a periodic wrap).
// Returns max over particles and fields of |grad_computed - grad_exact| / |grad_exact|.
double linear_gradient_error(Sim& sim);

}  // namespace shmem
