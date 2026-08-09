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
// Reconstruction: linear, limited ONCE PER PARTICLE against its whole neighbourhood (Hopkins 2015
// appendix B), with only a range clamp at each face. Time integration: MUSCL-Hancock -- primitives
// predicted a half step with the Lagrangian derivatives (Drho/Dt = -rho div v, Dv/Dt = -grad P/rho,
// Du/Dt = -(P/rho) div v), then one flux evaluation. Second order in smooth flow.
//
// Also here: self-gravity as a leapfrog KDK sharing one tree walk per step, and hierarchical
// individual timesteps with Saitoh-Makino wakeups.
//
// Deliberately NOT here yet: sink particles, MHD, cooling, a barotropic EOS.

#pragma once
#include <array>
#include <utility>
#include "hydro.h"

namespace shmem {

// The five primitive fields, in the one order used by gradients, reconstruction and the Riemann
// state vectors. Named rather than bare 0..4: the flux loop indexes these arrays a dozen times and
// a transposed velocity component is otherwise invisible.
enum PrimitiveField : int { FIELD_DENSITY = 0, FIELD_VX, FIELD_VY, FIELD_VZ, FIELD_PRESSURE,
                            NUM_FIELDS };
using PrimitiveState = std::array<double, NUM_FIELDS>;

// Per-particle scratch for the step. PERSISTENT across steps, not a local: under individual
// timesteps an inactive particle is still a neighbour of active ones, and the flux needs its
// B matrix, gradients and predicted state. Those keep the values from the particle's own last
// update, which is exactly the standard approximation.
struct Work {
    std::vector<Mat3d> moments_inv;                            // E^-1, one per particle
    std::array<std::vector<Vec3d>, NUM_FIELDS> gradient;       // gradient of each primitive field
    std::array<std::vector<double>, NUM_FIELDS> predicted;     // half-step predicted primitives
    std::vector<double> signal_speed;                          // Monaghan signal speed, per particle
    std::vector<double> div_vel;                               // velocity divergence at last update

    void resize(size_t n) {
        moments_inv.resize(n);
        for (auto& g : gradient)  g.resize(n);
        for (auto& p : predicted) p.resize(n);
        signal_speed.resize(n);
        div_vel.resize(n, 0.0);
    }
};

struct Sim {
    Particles P;                       // positions + masses + gravitational softening
    std::vector<double> vx, vy, vz;    // velocities
    std::vector<double> u;             // specific internal energy

    // ---- particle types ----
    // Gas-first layout: indices [0, n_gas) are gas (GIZMO type 0), everything after is
    // collisionless. SIZE_MAX (the default) means "all gas", so engine-internal users that never
    // set it keep the old behaviour unchanged. Hydro passes run over the gas prefix of the active
    // list and hydro neighbour sums skip non-gas particles; gravity, kicks and drifts cover
    // everyone.
    size_t n_gas = (size_t)-1;
    // Fixed softenings per GIZMO type, stored as the KERNEL EXTENT -- 2.8x the Plummer-equivalent
    // value in the params file, the same convention as GIZMO's ForceSoftening table and the same
    // length the spline kernels here take as h. Entry 0 is the constant-softening gas value; with
    // adaptive_soft the gas ignores it (soft_min floors the kernel radius instead).
    std::array<double, 6> soft_fixed{};
    std::vector<uint32_t> active_gas;  // scratch: gas prefix of `active` when types are mixed
    double gamma = 5.0 / 3.0;
    int    dim   = 3;                  // 1/2/3; matches BOX_SPATIAL_DIMENSION in the suite configs

    double box   = 0.0;                // >0: periodic cube [0, box)^3
    double des_ngb = 32.0;
    double cfl     = 0.25;

    // ---- self-gravity (off unless gravity_on; SELFGRAVITY_OFF in the suite configs) ----
    bool   gravity_on = false;
    double G          = 1.0;           // GravityConstantInternal
    double theta      = 0.5;           // geometric opening angle (ErrTolTheta); bootstrap only
                                       // once the relative criterion below is active
    double err_tol_force_acc = 0.0;    // ErrTolForceAcc; 0 = geometric opening only
    double soft_min   = 0.0;           // floor on the gas softening, as a kernel extent
                                       // (2.8 x Softening_Type0/SofteningGas)
    bool   adaptive_soft = true;       // ADAPTIVE_GRAVSOFT_FORGAS: soften on h, not a fixed length
    bool   output_potential = false;   // OUTPUT_POTENTIAL: write phi so energy/momentum checks work
    std::vector<double> phi;           // gravitational potential, filled only when writing output
    double eta_grav   = 0.025;         // ErrTolIntAccuracy; enters dt_grav and dt_tidal below
    bool   tidal_criterion = false;    // TIDAL_TIMESTEP_CRITERION: dt from the tidal tensor
    std::vector<SymTensor3d> tidal;    // d2phi/dxdx per particle (no G factor), from the walk
    std::vector<Vec3d> a_grav;         // acceleration at the CURRENT positions; see mfm_step
    // Half-kick owed by each particle from the close of ITS OWN previous step. Must be per
    // particle: with a spread of timebins the closing half-kick a particle owes is half of its own
    // last step, which has nothing to do with the system step. A single shared scalar silently
    // under-kicks every particle on a longer bin -- 25% low for a bin-0 particle alongside bin-1
    // neighbours -- so gravity comes out systematically weak exactly where the bin spread is
    // widest, which in a collapse is the core.
    std::vector<double> pending_half_kick;

    // ---- individual (hierarchical) timesteps ----
    // A particle on bin b steps dt_base / 2^b. Time is tracked as an INTEGER count of ticks, where
    // one tick = dt_base / 2^MAX_BINS, so every particle's sync points are exactly a subset of the
    // finer bins' and there is no drift from repeated floating-point addition. dt_base is chosen by
    // the caller to divide the snapshot interval exactly (see set_time_base), which is what lets the
    // run land on snapshot times and TimeMax to the bit.
    // Off by default: with every particle on bin 0 this reduces exactly to the global scheme.
    static constexpr int MAX_BINS = 30;
    bool      individual_timesteps = false;
    double    dt_base    = 0.0;        // step of bin 0; 0 until set_time_base()
    long long clock_ticks = 0;         // current time, in ticks of dt_base / 2^MAX_BINS
    int       bin_limit  = 3;          // Saitoh-Makino: a particle may not sit more than this many
                                       // bins above an ACTIVE neighbour, or a shock outruns it
    std::vector<int>      bin;         // current timebin per particle
    std::vector<uint32_t> active;      // indices due at this sync point
    // wall time attributed to each bin, for the cpu-frac column of the timebin dump: a sync is
    // charged to its LONGEST-dt active bin, since that is what made the step expensive
    std::vector<double>    bin_cpu_sum;
    std::vector<long long> bin_cpu_n;
    long long sync_point = 0;

    long long ticks_in_bin(int b) const { return 1LL << (MAX_BINS - b); }
    double    dt_of_bin(int b)    const { return dt_base / (double)(1LL << b); }
    // The integer clock is the ONLY source of truth for time: a step whose length is not a whole
    // number of ticks desynchronises the hierarchy permanently, because every particle's sync
    // points are defined by clock alignment. Once nothing is aligned the active set is empty on
    // every sync, and an empty active set is silent -- no kicks are applied, and the drift pass
    // still moves everyone, so the whole system sails on ballistically with gravity switched off.
    // (That is exactly how plummer/tidal lost its last snapshot interval: one 5e-15 residual step
    // at a snapshot boundary rounded to 0 ticks, was clamped up to 1, and desynchronised the run.)
    static constexpr long long TICKS_PER_BASE = 1LL << MAX_BINS;
    double    time_now()          const { return dt_base * (double)clock_ticks / (double)TICKS_PER_BASE; }
    double    time_of_ticks(long long k) const { return dt_base * (double)k / (double)TICKS_PER_BASE; }
    long long ticks_of_time(double t) const {
        return (long long)std::llround(t / dt_base * (double)TICKS_PER_BASE);
    }
    // Largest whole number of ticks not exceeding dt. Returns 0 when dt is below one tick, which
    // the caller must treat as "no step to take" rather than rounding up to 1.
    long long ticks_floor(double dt) const {
        const double k = std::floor(dt / dt_base * (double)TICKS_PER_BASE);
        if (!(k > 0)) return 0;                                   // also catches NaN
        return (k >= (double)TICKS_PER_BASE * 4.0) ? TICKS_PER_BASE * 4 : (long long)k;
    }
    bool      is_active(size_t i) const {
        return !individual_timesteps || (clock_ticks % ticks_in_bin(bin[i])) == 0;
    }

    // derived per step; under individual timesteps only the ACTIVE entries are refreshed and the
    // rest keep their values from each particle's own last update
    std::vector<double> h, ninv, rho, press;
    // grad-h ("Omega") factor 1/(1 + h/(NDIMS n) dn/dh), needed by the entropic-EOS face
    // correction (GIZMO's DrkernNgbFactor) and the zeta terms
    std::vector<double> omega;
    Work work;

    // ---- persistent tree ----
    // Rebuilding costs O(N log N) whatever the active fraction, so with individual timesteps it
    // dominates completely: at ~100 active out of 2e6 the rebuild was measured at ~130 ms against
    // ~3 ms of actual physics. The tree is therefore kept and REUSED, with Tree::pad inflated by
    // how far particles have drifted so neighbour searches stay exact, and rebuilt only when that
    // pad has grown enough to make the search inefficient.
    Tree   tree;
    bool   tree_valid = false;
    double drift_since_build = 0.0;    // upper bound on any particle's displacement since build
    double tree_rebuild_pad_frac = 0.25;  // rebuild once pad exceeds this fraction of a typical h
    long long tree_builds = 0;         // diagnostic

    // scratch reused across steps, so a sync does not allocate and zero several N-sized arrays
    std::vector<double> dt_of, dmom_x, dmom_y, dmom_z, denergy;
    std::vector<std::pair<uint32_t,int>> wake_requests;   // (particle, bin it must drop to)
    // particles that received flux this sync (actives + their neighbours), duplicates allowed
    std::vector<uint32_t> touched;
    std::vector<std::vector<uint32_t>> active_chunks;     // per-thread, merged into `active`
    std::vector<uint32_t> halo;        // actives + their neighbours; geometry refreshed for these

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

// Choose dt_base for the individual-timestep hierarchy: the largest step not exceeding
// max_step that divides `interval` (the snapshot spacing) a whole number of times. Every
// snapshot boundary is then an exact integer number of ticks, so a run lands on its output
// times and on TimeMax exactly, however the bins are distributed.
void set_time_base(Sim& sim, double interval, double max_step);

// Populate h, volume, density and pressure for every particle without taking a step, so the t=0
// snapshot carries the same fields the scheme itself uses.
//
// Worth having as a function rather than a few lines in the driver: MFM's density is m_i/V_i with
// V_i from the partition of unity, which on a uniform lattice stays SHARP across a contact because
// the mass carries the contrast. The SPH-style sum_j m_j W is a different quantity -- it smooths
// mass over the kernel -- and using it produces a visibly wrong t=0 state at any contact
// discontinuity (in test/square it made the exactly-uniform initial pressure vary by 84%, in a
// square-shaped ring on the interface).
void compute_initial_state(Sim& sim);

// Fill sim.phi with the gravitational potential at the current positions, for OUTPUT_POTENTIAL.
// Only called when writing a snapshot -- the run itself never needs it, but without it neither the
// energy budget nor the momentum-drift normalisation can be evaluated at all: both scale by
// v_grav = sqrt(|W|/M), and a cold start falls back to v_rms(0) ~ 0, which inflates the reported
// drift by orders of magnitude even when momentum is conserved to 1e-5.
void compute_potential(Sim& sim);

// Dump the timebin hierarchy in GIZMO's format (core/run.cc), so output from the two engines can be
// read side by side. NOTE the bin convention is INVERTED relative to GIZMO's: here bin 0 is the
// LONGEST step (dt_base) and deeper bins are shorter, whereas GIZMO numbers upward with dt. Rows are
// still printed longest-step first, so the table reads the same way; `dt` is printed explicitly so
// there is nothing to infer from the index.
//   X          this bin is active at this sync point
//   <          the longest-step bin that is active, i.e. what sets the system step
//   cumulative particles in this bin and every shorter one -- the count actually being integrated
//              at that level and below
void print_timebins(const Sim& sim, double systemstep, double time);

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
