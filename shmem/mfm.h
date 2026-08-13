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
    std::vector<double> condition_number;                      // E-matrix conditioning, per cell
    std::vector<double> face_closure;                          // dimensionless face-closure leak

    void resize(size_t n) {
        moments_inv.resize(n);
        for (auto& g : gradient)  g.resize(n);
        for (auto& p : predicted) p.resize(n);
        signal_speed.resize(n);
        div_vel.resize(n, 0.0);
        condition_number.resize(n, 1.0);
        face_closure.resize(n, 0.0);
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

    // ---- shared neighbour search (SHMEM_CACHE_NEIGHBORS) ----
    // The h solve's converged iteration is a traversal at exactly the h every later phase then
    // searches with, and nothing moves in between (the drift is at the end of the step; the kicks
    // touch only velocities). So density() hands that list out and the volume/zeta sums, the
    // gradients and the fluxes all read it: ONE traversal per active particle per step instead of
    // one per phase.
    //
    // Compile-time rather than runtime because it is a memory-for-time trade with no universally
    // right answer (~4 bytes per neighbour per active particle, plus a transient second copy while
    // the per-thread buffers merge). Off by default.
    NeighborCache ngb_cache;
    // ---- lazy drift (GIZMO's core/predict.cc drift_particle, guarded on P[i].Ti_current) ----
    // An inactive particle's velocity is CONSTANT between its own activations -- the flux loop
    // refuses to touch inactive particles and the gravity kick covers actives only -- so drifting
    // it once over a long interval is EXACTLY equivalent to drifting it every sync, and the O(N)
    // sweep is pure waste in the regime that matters: a few hundred cells on persistent short
    // timesteps, where it is most of the step.
    //
    // Each particle records the tick its position is current at. The step drifts the ACTIVE set;
    // everything else is caught up at the moment a search or a walk first reaches it, which is
    // exactly where GIZMO does it. Everyone is synced before a tree build and before output.
    std::vector<long long> last_drift;
    bool sparse_drift = true;           // SHMEM_DENSE_DRIFT=1 restores the full sweep, for A/B
    // The hook the step hands to every tree walk. Held here rather than threaded through
    // solve_h_and_volumes / gradients / fluxes / compute_gravity as a parameter, which would be
    // five layers of plumbing for one pointer. Null while the dense sweep is in use, and the walks
    // then compile to exactly the code they had before lazy drift existed.
    LazyDrift lazy_drift;
    bool lazy_drift_on = false;
    const LazyDrift* lazy() const { return lazy_drift_on ? &lazy_drift : nullptr; }
    std::vector<uint32_t> grav_targets; // scratch: active list in tree Morton order (see
                                        // compute_gravity -- batch walks need spatial coherence)
    double gamma = 5.0 / 3.0;
    int    dim   = 3;                  // 1/2/3; matches BOX_SPATIAL_DIMENSION in the suite configs

    // ---- equation of state ----
    // One mechanism, three laws. IDEAL is the ordinary P = (gamma-1) rho u with u evolved by the
    // energy equation. The other two make PRESSURE A FUNCTION OF DENSITY ALONE and overwrite u
    // from it at every evaluation: they stand in for radiative cooling, so shock heating is not
    // retained and the energy equation's answer is deliberately discarded.
    enum class EosLaw { IDEAL, ENFORCE_ADIABAT, BAROTROPIC };
    EosLaw eos_law     = EosLaw::IDEAL;
    double eos_adiabat = 0.0;          // EOS_ENFORCE_ADIABAT=A: P = A rho^gamma, code units
    int    baro_variant = 0;           // EOS_GMC_BAROTROPIC=N: 0 = MI2000 piecewise, 1..4 = BBB03
    bool   baro_soundspeed = false;    // EOS_GMC_BAROTROPIC_SOUNDSPEED: sound speed from the
                                       // barotrope's own dlnP/dlnrho rather than from gamma
    double nh_per_code_density = 0.0;  // code rho -> n_H [cm^-3]
    double code_press_per_cgs  = 0.0;  // cgs pressure -> code pressure
    // Per-particle sound speed. A barotrope's dP/drho differs from gamma P/rho off the adiabatic
    // branch, and GIZMO feeds THAT into the Riemann wavespeeds (EOS_GMC_BAROTROPIC implies
    // EOS_GENERAL), so it is carried explicitly rather than re-derived from a single gamma at
    // each use site.
    std::vector<double> csnd;
    bool eos_is_ideal() const { return eos_law == EosLaw::IDEAL; }

    // ---- sink particles (SINGLE_STAR_SINK_FORMATION) ----
    // Formation is DETERMINISTIC: GIZMO multiplies the surviving rate by 1e20, so a cell that
    // passes every criterion converts on the spot. The criteria are all veto-style, exactly as in
    // galaxy_sf/sfr_eff.cc -- see project-shmem-sink-plan for the bit-by-bit mapping.
    bool   sink_formation = false;     // SINGLE_STAR_SINK_FORMATION present in the config
    double mass_to_solar = 1.0;        // code mass -> Msun (UnitMass_in_g / 1.989e33)
    double vel_to_kms = 1.0;           // code velocity -> km/s (UnitVelocity_in_cm_per_s / 1e5)
    // code length -> AU, for the 0.1 AU Larson-core floor on sink formation (sfr_eff.cc:348)
    double length_to_au = 0.0;
    // Radius inside which SINK-SINK gravity is summed exactly rather than through a multipole
    // (SINGLE_STAR_DIRECT_GRAVITY_RADIUS, precompiler_logic.h:378 -- 1000 AU, on by default in the
    // STARFORGE bundle). In CODE length units; 0 disables. A binary integrated through a multipole
    // drifts in energy, which is precisely what the collisional tests measure.
    double sink_direct_radius = 0.0;
    // COOLING || EOS_GMC_BAROTROPIC -- the reference's guard on that floor (sfr_eff.cc:347).
    bool opacity_limit_physics = false;
    double crit_phys_density = 0.0;    // PhysDensThresh, code units (CritPhysDensity / n_H per rho)
    double max_sfr_timescale = 0.0;    // MaxSfrTimescale, code units
    // Rolling time average of 1/(1+alpha_vir), for the &2048 time-averaged virial criterion. A
    // single instant of low alpha_vir in a turbulent flow is noise; the average is what GIZMO
    // actually thresholds on.
    std::vector<double> alpha_vir_smoothed;
    long long sinks_formed = 0;        // diagnostic
    long long cells_accreted = 0;      // diagnostic
    double mass_initial = 0.0;         // total mass at t=0, for the accretion bookkeeping check
    // Bate-style FIXED accretion radius, set once when the sink forms and carried per particle
    // (SINK_GRAVCAPTURE_FIXEDSINKRADIUS; galaxy_sf/sfr_eff.cc:602-608). Zero for gas.
    std::vector<double> sink_radius;
    // Formation time and mass-at-formation, recorded at conversion for the snapshot's
    // StellarFormationTime / Sink_InitialMass datasets (file_io/io.cc:3860,4040). Zero for gas.
    // (In the legacy global-timestep A/B mode time_now() is 0, so tform reads 0 there.)
    std::vector<double> sink_tform;
    std::vector<double> sink_m0;
    // SINK_ALPHADISK_ACCRETION (a STARFORGE default): swallowed gas lands in an unresolved-disk
    // reservoir and drains into the star over t_acc, rather than becoming stellar mass on contact.
    // P.m stays the DYNAMICAL mass throughout -- reservoir included -- so gravity is untouched;
    // the split exists to give Sink_Mdot a smooth value, which is what dt_accr steps on. For
    // SINK_GRAVCAPTURE_FIXEDSINKRADIUS the drain time reduces to a constant set at formation,
    //   t_acc = sqrt(reff^3/(G*M_form)) = G*M_form/cs_min^3,  reff = G*M_form/cs_min^2
    // (sinks/sink.cc:395-399) -- i.e. mdot = (cs_min^3/G) * (reservoir/M_form), the Shu isothermal
    // rate scaled by how full the disk is. cs_min is a hard floor of 0.2 km/s, not the local
    // sound speed. Verified against the reference's own Sink_Mdot to 1-2% at every snapshot.
    std::vector<double> sink_reservoir;
    double sink_mass_split = 0.0;      // MaxMassForParticleSplit = 3.01 * max initial gas mass
                                       // (core/init.cc:965); caps the mass scale in dt_accr
    // SINGLE_STAR_TIMESTEPPING: per-particle minimum approach / freefall time to the sink
    // population (gravity/forcetree.cc:2509-2510), refreshed for active particles each sync.
    // 1e300 = "no sink seen"; feeds the two-body sink criterion and the gas approach cap.
    // Velocity at each particle's last node-kick accounting, so the momentum DELTA fed to
    // Tree::kick_node is the change since the tree last saw it (GIZMO passes dp directly from
    // do_the_kick; here the kicks are applied in bulk, so the delta is reconstructed).
    std::vector<Vec3d> vel_at_last_kick;
    std::vector<double> min_sink_tapp;
    std::vector<double> min_sink_tff;
    // Sink-gas dt ceiling (wakeup + freefall + Courant caps, core/timestep.cc:1002-1026),
    // refreshed per active sink from a gas-neighbour scan. 1e300 = no gas nearby.
    std::vector<double> sink_dt_gas_cap;
    // HERMITE_INTEGRATION (core/kicks.cc:104-177): 4th-order predict-evaluate-correct for the
    // types in hermite_mask (STARFORGE default: 32 = sinks). herm_* is the snapshot of the
    // TRUE dynamical state at the particle's last sync; herm_dt the step it opens. A snapshot
    // is invalidated by accretion (GIZMO's AccretedThisTimestep fallback) and rebuilt at the
    // next sync, so KDK always remains valid underneath.
    int hermite_mask = 0;
    std::vector<uint8_t>   herm_valid;
    // The TICK the snapshot was taken at, not the step length that was expected to follow.
    // GIZMO derives the Hermite interval from integer times (kicks.cc:134-138,
    // tstart = Ti_begstep, tend = tstart + ti_step); storing a dt instead is wrong whenever the
    // step actually taken differs from the one assigned -- which happens on every snapshot
    // boundary (dt_of is min(bin step, dt_cap)) and on every Saitoh-Makino wake-up.
    std::vector<long long> herm_tick;
    std::vector<Vec3d>     herm_pos, herm_vel, herm_acc, herm_jerk;
    // Particle IDs live HERE rather than beside the driver: sink formation swaps particles and
    // accretion deletes them, so anything parallel to the particle arrays has to be permuted with
    // them or the snapshot silently mislabels every particle after the first event.
    std::vector<long long> id;

    double box   = 0.0;                // >0: periodic cube [0, box)^3
    double des_ngb = 32.0;
    // MaxNumNgbDeviation from the params file: how close the kernel-weighted neighbour number
    // must get to des_ngb before the h solve stops. This was hardcoded at 1e-4*des_ngb, i.e.
    // 16x tighter than shu1977 actually asks for -- which matters in a tight knot of particles,
    // where the exactly-converged kernel is much narrower than the tolerated one and gives a
    // correspondingly sharper density peak.
    double ngb_tol = 0.05;
    double cfl     = 0.25;

    // ---- self-gravity (off unless gravity_on; SELFGRAVITY_OFF in the suite configs) ----
    bool   gravity_on = false;
    double G          = 1.0;           // GravityConstantInternal
    double theta      = 0.5;           // geometric opening angle (ErrTolTheta). Bootstrap only
                                       // unless hybrid_opening: see below.
    // GRAVITY_HYBRID_OPENING_CRIT, which GRAVITY_ACCURATE_FEWBODY_INTEGRATION enables and the
    // STARFORGE defaults enable in turn (precompiler_logic.h:373, 567). WITHOUT it the reference
    // zeroes ErrTolTheta once the first walk has established aold (gravtree.cc:489, with
    // TypeOfOpeningCriterion hardcoded to 1 at begrun.cc:2740), leaving the relative criterion
    // alone; WITH it Barnes-Hut stays live and the two run as a union. Getting this wrong costs
    // real time -- the union opens strictly more nodes than either test by itself.
    bool hybrid_opening = false;
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
    // SWALLOW SPLIT. The reference swallows between its closing and opening half-kicks
    // (run.cc:167-169), which is mid-step here, and removing a particle there would shift every
    // index the rest of the step still holds. So the merge happens at that point and the removal
    // is deferred to the end of the step -- the reference does the same thing, zeroing the
    // swallowed cell's mass in place and tidying up later. doomed_mask keeps the marked cells out
    // of the conserved update in between.
    std::vector<uint32_t> doomed_cells;
    std::vector<char>     doomed_mask;
    Vec3d acc_audit_baseline{0, 0, 0};
    // Hydro flux RATES from each particle's own last evaluation: the momentum rate as an
    // acceleration, and the internal-energy rate per unit mass (the total-energy rate minus the
    // kinetic part at the flux-time velocity -- GIZMO's HydroAccel and DtInternalEnergy). Applied
    // by the half-kicks exactly like a_grav, and read by the drift's velocity prediction.
    std::vector<Vec3d> a_hydro;
    std::vector<double> du_dt;
    // False until the first force evaluation. The timestep is decided from the PREVIOUS step's
    // forces (GIZMO decides dt at the top of the step, before any walk), so the first step needs
    // a bootstrap evaluation at t=0 -- the reference's init does the same before its main loop.
    bool forces_valid = false;
    // SHMEM_SINK_PINNED scratch: where each sink formed, and whether that has been recorded.
    std::vector<double> sink_pin_x, sink_pin_y, sink_pin_z;
    std::vector<char>   sink_pinned;
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
    // MFM+GALSF replaces the Monaghan signal velocity with the CONTACT WAVE speed from the
    // Riemann solve, vsig = 2*S_M + max(0, dv_face) (hydro/hydro_core_meshless.h:253), floored at
    // the cell's own sound speed (hydro_evaluate.h:82). That feeds the Courant step and comes out
    // roughly TWICE cs_i+cs_j, so it HALVES dt and costs ~1.3x the steps over a shu1977 run. Set
    // from the config's GALSF, matching the reference; see gizmo_main.cc.
    bool      contact_wave_vsig = false;
    double    wakeup_fac = 4.1;        // Saitoh-Makino: demote an inactive neighbour once a pair's
                                       // signal speed exceeds this multiple of the one it last
                                       // recorded for itself (GIZMO WAKEUP, declarations/constants.h;
                                       // 4.1 allows two bins within a kernel, 2.1 only one)
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
    double tree_rebuild_pad_frac = 0.25;  // rebuild once pad exceeds this fraction of a typical h
    long long tree_builds = 0;         // diagnostic; also seeds the randomised root offset
    // RANDOMIZE_GRAVTREE: redraw the tree's root offset on every build so the walk's force errors
    // are decorrelated between steps rather than repeating. A fixed grid's errors are the same
    // every step and integrate into a secular drift; redrawn ones average out.
    bool      randomize_gravtree = false;
    // Guards the active-set-only softening update: a full sweep is redone whenever either of these
    // changes, which is how sink formation (which reorders the arrays) forces one.
    // Representative h of the WHOLE distribution as of the last tree build. The rebuild trigger
    // asks "has drift degraded THIS tree", which is a property of every particle the tree covers --
    // not of whichever few are active. Sampling h from one active particle instead made the
    // threshold collapse to a core cell's h on deep-bin steps and rebuilt 3.5e6 nodes almost every
    // step: measured ~130 ms of tree per step at nact=21 on the bate cloud.
    double    tree_typical_h = 0.0;
    size_t    soft_valid_n = (size_t)-1;
    size_t    soft_valid_ngas = (size_t)-1;
    // ADAPTIVE_TREEFORCE_UPDATE=f: a GAS cell keeps its cached tree force, advanced with the jerk,
    // until it has aged past f * its tidal time (gravtree.cc:939-951, sfr_eff dt_tidal). 0 = off.
    // Non-gas and Hermite-integrated types always take a fresh force: the predictor-corrector
    // sub-stepping is incompatible with a cached one (gravtree.cc:941).
    double    atu_frac = 0.0;
    std::vector<double> time_since_treeforce;   // per particle, code time since its last real walk
    // mutable because desired_dt() is a const query and this is a CACHE it fills in passing: the
    // tidal dt is computed there and nowhere else, and recomputing it in the caller would mean
    // duplicating the tensor norm and the self-gravity floor.
    mutable std::vector<double> tdyn_for_treeforce;   // per particle, tidal dt setting the cadence
    std::vector<Vec3d>  a_grav_jerk;            // per particle, d(a)/dt from that walk

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
// SHMEM_ENERGY_LOG: total energy with the pending half kick undone, so KE and PE are at
// the same instant. Snapshot energies are NOT usable for this -- see the definition.
void energy_log_step(Sim& sim, double t);

// Bring every particle's position current at the engine clock. The step drifts only the active set
// and catches the rest up on touch, so anything that reads positions in BULK -- writing a snapshot,
// building a tree, any external diagnostic -- must call this first. GIZMO's equivalent sync points
// are domain/domain.cc:238,433 and gravity/potential.cc:68.
void sync_all_positions(Sim& sim);

// Dump the timebin hierarchy in GIZMO's format (core/run.cc), so output from the two engines can be
// read side by side. NOTE the bin convention is INVERTED relative to GIZMO's: here bin 0 is the
// LONGEST step (dt_base) and deeper bins are shorter, whereas GIZMO numbers upward with dt. Rows are
// still printed longest-step first, so the table reads the same way; `dt` is printed explicitly so
// there is nothing to infer from the index.
//   X          this bin is active at this sync point
//   <          the longest-step bin that is active, i.e. what sets the system step
//   cumulative particles in this bin and every shorter one -- the count actually being integrated
//              at that level and below
void hermite_report();
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
