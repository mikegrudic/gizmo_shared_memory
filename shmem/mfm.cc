#include "mfm.h"
#include <array>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <atomic>

#ifdef SHMEM_CUDA
extern "C" void shmem_cuda_accel_bruteforce(
    int n_source, const double* sx, const double* sy, const double* sz,
    const double* sm, const double* seps, const double* szeta, const char* sgas,
    int n_target, const int* tidx, double box,
    double* ax_out, double* ay_out, double* az_out);
#endif

namespace shmem {

// Neighbour-list access for the three hydro phases. With SHMEM_CACHE_NEIGHBORS the list is built
// once per step (in solve_h_and_volumes) and the later phases read it back; without it, each
// phase searches the tree as before. Both paths hand the caller the same `neighbours` vector, so
// the loop bodies are identical and there is only one copy of the physics.
//
// `k` is the caller's index into the active list, which is what the cache is keyed on.
static inline void get_neighbours(const Sim& sim, const Tree& tree, size_t k, const Vec3d& pos_i,
                                  double radius, std::vector<uint32_t>& neighbours) {
#ifdef SHMEM_CACHE_NEIGHBORS
    if (sim.ngb_cache.valid && k + 1 < sim.ngb_cache.start.size()) {
        const size_t lo = sim.ngb_cache.start[k], hi = sim.ngb_cache.start[k + 1];
        neighbours.assign(sim.ngb_cache.flat.begin() + lo, sim.ngb_cache.flat.begin() + hi);
        return;
    }
#else
    (void)k;
#endif
    neighbours.clear();
    ngb_search(tree, sim.P, pos_i, radius, neighbours, sim.box, sim.lazy());
}

// invert_moments now lives in vec.h: the h solve in hydro.cc needs it too, for the
// face-closure correction.

// ---------------------------------------------------------------------------------------------
// EQUATION OF STATE. Sets pressure and sound speed for one particle, and -- for the laws where
// pressure is a function of density alone -- writes the internal energy back from it.
//
// Ported from GIZMO's eos/eos.cc (with the user's barotropic variants). Constants are in cgs and
// take n_H, so density is converted out to cgs and the pressure converted back.
//
// The u write-back must use the THERMODYNAMIC index gamma, never the barotrope's local
// dlnP/dlnrho: u = P/(rho (gamma_eff-1)) diverges as gamma_eff -> 1 on the isothermal branch.
// gamma_eff exists only to set the sound speed, and only when EOS_GMC_BAROTROPIC_SOUNDSPEED asks
// for it -- otherwise the isothermal branch would reach the Riemann solver at sqrt(gamma) c_s0
// instead of c_s0.
static inline void eos_apply(Sim& sim, size_t i) {
    const double rho = sim.rho[i];
    if (rho <= 0) { sim.press[i] = 0; if (!sim.csnd.empty()) sim.csnd[i] = 0; return; }
    double press, gamma_eff = sim.gamma, gamma_index = sim.gamma;

    switch (sim.eos_law) {
    case Sim::EosLaw::IDEAL:
        press = (sim.gamma - 1.0) * rho * sim.u[i];
        break;
    case Sim::EosLaw::ENFORCE_ADIABAT:
        press = sim.eos_adiabat * std::pow(rho, sim.gamma);
        break;
    case Sim::EosLaw::BAROTROPIC: {
        const double nH = rho * sim.nh_per_code_density;
        double p_cgs;
        if (sim.baro_variant == 0) {
            // Masunaga & Inutsuka 2000 / Federrath+ 2014 piecewise form
            const double g = (nH < 2.30181e16) ? 1.4 : (5.0 / 3.0);
            if      (nH < 1.49468e8)  { p_cgs = 6.60677e-16 * nH;                gamma_eff = 1.0; }
            else if (nH < 2.30181e11) { p_cgs = 1.00585e-16 * std::pow(nH, 1.1); gamma_eff = 1.1; }
            else if (nH < 2.30181e16) { p_cgs = 3.92567e-20 * std::pow(nH, g);   gamma_eff = g;   }
            else if (nH < 2.30181e21) { p_cgs = 3.1783e-15  * std::pow(nH, 1.1); gamma_eff = 1.1; }
            else                      { p_cgs = 2.49841e-27 * std::pow(nH, g);   gamma_eff = g;   }
            gamma_index = g;
        } else {
            // Bate, Bonnell & Bromm 2003 family: isothermal below n_crit, adiabatic above.
            // 1/2 join at the critical density, 3/4 join smoothly (Hopkins' EOS_MHD_CORE form);
            // odd variants use gamma = 7/5, even 5/3.
            const double nH_crit = 6.0e10, p_iso = 6.60677e-16 * nH;
            const double g = (sim.baro_variant == 1 || sim.baro_variant == 3) ? 1.4 : (5.0 / 3.0);
            gamma_index = g;
            if (sim.baro_variant <= 2) {
                if (nH < nH_crit) { p_cgs = p_iso; gamma_eff = 1.0; }
                else { p_cgs = 6.60677e-16 * nH_crit * std::pow(nH / nH_crit, g); gamma_eff = g; }
            } else {
                // smooth: P = c_s0^2 rho sqrt(1 + (rho/rho_crit)^(2(gamma-1))), which exceeds the
                // piecewise form by at most sqrt(2) in P (9% in c_s), at rho_crit
                const double x = std::pow(nH / nH_crit, 2.0 * (g - 1.0));
                p_cgs = p_iso * std::sqrt(1.0 + x);
                gamma_eff = 1.0 + (g - 1.0) * x / (1.0 + x);   // runs 1 -> gamma, (1+g)/2 at crit
            }
        }
        press = p_cgs * sim.code_press_per_cgs;
        break;
    }
    default:
        press = (sim.gamma - 1.0) * rho * sim.u[i];
        break;
    }

    sim.press[i] = press;
    if (sim.eos_law != Sim::EosLaw::IDEAL)
        sim.u[i] = press / (rho * (gamma_index - 1.0));
    if (sim.eos_law != Sim::EosLaw::IDEAL && !sim.csnd.empty()) {
        const double g_cs = (sim.eos_law == Sim::EosLaw::BAROTROPIC && sim.baro_soundspeed)
                          ? gamma_eff : sim.gamma;
        sim.csnd[i] = std::sqrt(g_cs * press / rho);
    }
}

// Sound speed for particle i. The IDEAL branch recomputes from press/rho rather than reading a
// stored value, and that is deliberate: this is called per neighbour inside the gradient loop,
// which already has press[j] and rho[j] in cache, whereas csnd[j] would be a THIRD scattered
// array touched per neighbour. Reading it cost 28% of the whole sedov run (55.1 -> 70.8 s) --
// an arithmetic sqrt is far cheaper than the cache miss that avoids it. csnd is therefore only
// allocated and consulted when a density-driven EOS actually makes it differ from gamma P/rho.
static inline double sound_speed(const Sim& sim, size_t i) {
    if (sim.eos_is_ideal()) return std::sqrt(sim.gamma * sim.press[i] / sim.rho[i]);
    return sim.csnd[i];
}

[[nodiscard]] static constexpr double minmod(double a, double b) noexcept {
    return (a*b <= 0) ? 0.0 : (std::abs(a) < std::abs(b) ? a : b);
}

// Per-particle gradient limiter -- a literal port of GIZMO's local_slopelimiter
// (hydro/gradients.cc), decided once per particle against the whole neighbourhood.
//
// The constants matter more than the structure. GIZMO limits the extrapolation over a reach of
// A_LIMITER * h_lim with A_LIMITER = 0.25 -- an earlier draft here used 0.5, which permits only
// HALF the slope for the same neighbour excursions. That clips gradients at shocks, a clipped
// gradient smears the shock over more kernel lengths, and in a collapse problem the smeared
// accretion shock PRE-HEATS the inflow: evrard's core entropy came out 3-11% high and its central
// density 12-15% low against the reference run with every gravity detail already matched.
//
// `stol` allows overshoot beyond the smaller excursion (capped at the larger); GIZMO uses 0 with
// gravity on and 0.1 for pure hydro. Positivity preservation (density, pressure) additionally
// caps the slope so the value stays positive over the farthest neighbour distance d_max.
// GIZMO does NOT hold this fixed: at gradients.cc:1005 it TIGHTENS the limiter where the
// E-matrix is poorly conditioned --
//     a_limiter = 0.25; if(cn > 100) a_limiter = min(0.5, 0.25 + 0.25*(cn-100)/100)
// -- a larger a_limiter permitting a SMALLER slope. Holding it at 0.25 everywhere means
// trusting reconstructed gradients in exactly the cells where the matrix says they are least
// trustworthy, which in a collapsing core is where it matters.
static constexpr double A_LIMITER = 0.25;
// Condition number above which the matrix-based gradients are not trusted
// (declarations/constants.h:79 -- 1e3 for everything except MHD-without-cooling and EOS_ELASTIC).
static constexpr double CONDITION_NUMBER_DANGER = 1.0e3;

static inline double a_limiter_for(double condition_number) {
    if (condition_number <= 100.0) return A_LIMITER;
    return std::min(0.5, 0.25 + 0.25 * (condition_number - 100.0) / 100.0);
}
static inline void limit_slope(Vec3d& gradient, double largest_rise, double largest_drop,
                               double h_lim, double stol, bool pos_preserve,
                               double d_max, double val_cen, double a_limiter = A_LIMITER) {
    const double slope = gradient.norm();
    if (slope <= 0) return;
    double abs_max = std::abs(largest_rise), abs_min = std::abs(largest_drop);
    if (abs_max < abs_min) std::swap(abs_max, abs_min);
    const double allowed = std::min(abs_min + stol * abs_max, abs_max);
    double factor = allowed / (a_limiter * h_lim * slope);
    if (pos_preserve && d_max > 0) {
        const double val_min_ngb = val_cen + largest_drop;      // actual minimum neighbour value
        const double fmin = std::min(val_cen, std::max(0.0,
                              std::min(0.5 * (val_cen + val_min_ngb), val_cen - allowed)));
        factor = std::min(((val_cen - fmin) / d_max) / slope, factor);
    }
    if (factor < 1.0) gradient *= factor;
}

// Pairwise face limiter -- a literal port of GIZMO's reconstruct_face_states (hydro/reimann.h).
// The face value may OVERSHOOT the pair range by FAC_MINMAX of the jump but must stay within
// FAC_MEDDEV of the jump around the midpoint; a value about to cross zero is reinterpreted as a
// logarithmic extrapolation instead, which preserves the sign. Strictly looser than a hard clamp
// into [min,max] on the outer edge and tighter around the midpoint -- the combination keeps
// shocks SHARP (GIZMO's comments: 1.0 unstable, 0.75 creeps, 0.5 works).
static constexpr double FAC_MINMAX = 0.5, FAC_MEDDEV = 0.375;
static inline void limit_face_pair(double Q_i, double Q_j, double& face_i, double& face_j) {
    if (Q_i == Q_j) { face_i = face_j = Q_i; return; }
    const double Qmed = 0.5 * (Q_i + Q_j);
    const double Qmax = std::max(Q_i, Q_j), Qmin = std::min(Q_i, Q_j);
    double fac = FAC_MINMAX * (Qmax - Qmin);
    double Qmax_eff = Qmax + fac, Qmin_eff = Qmin - fac;
    if (Qmax < 0 && Qmax_eff > 0) Qmax_eff = Qmax * Qmax / (Qmax - (Qmax_eff - Qmax));
    if (Qmin > 0 && Qmin_eff < 0) Qmin_eff = Qmin * Qmin / (Qmin + (Qmin - Qmin_eff));
    fac = FAC_MEDDEV * (Qmax - Qmin);
    const double Qmed_max = std::min(Qmed + fac, Qmax_eff);
    const double Qmed_min = std::max(Qmed - fac, Qmin_eff);
    if (Q_i < Q_j) {
        face_i = std::min(std::max(face_i, Qmin_eff), Qmed_max);
        face_j = std::min(std::max(face_j, Qmed_min), Qmax_eff);
    } else {
        face_i = std::min(std::max(face_i, Qmed_min), Qmax_eff);
        face_j = std::min(std::max(face_j, Qmin_eff), Qmed_max);
    }
}

// Contact-wave speed and pressure from the Riemann fan -- the only two quantities the Lagrangian
// MFM flux needs, since the mass flux vanishes by construction.
struct ContactState { double speed, pressure; };

// HLLC for an ideal gas, states already rotated so the given velocities are normal to the face.
// Wave-speed estimates: Davis.
// gamma_left/gamma_right are the LOCAL effective adiabatic indices, so a barotrope reaches the
// wavespeeds at its own dP/drho rather than at the thermodynamic gamma -- GIZMO's EOS_GENERAL
// pathway. For an ideal gas both are simply gamma and this reduces to the usual estimate.
[[nodiscard]] static ContactState solve_hllc_contact(
        double density_left,  double vnorm_left,  double pressure_left,
        double density_right, double vnorm_right, double pressure_right,
        double gamma_left, double gamma_right) {
    const double csound_left  = std::sqrt(gamma_left  * pressure_left  / density_left);
    const double csound_right = std::sqrt(gamma_right * pressure_right / density_right);
    const double wave_left  = std::min(vnorm_left - csound_left, vnorm_right - csound_right);
    const double wave_right = std::max(vnorm_left + csound_left, vnorm_right + csound_right);
    const double numerator = pressure_right - pressure_left
                           + density_left  * vnorm_left  * (wave_left  - vnorm_left)
                           - density_right * vnorm_right * (wave_right - vnorm_right);
    const double denominator = density_left  * (wave_left  - vnorm_left)
                             - density_right * (wave_right - vnorm_right);
    double contact_speed = (std::abs(denominator) > 1e-300)
                         ? numerator / denominator : 0.5*(vnorm_left + vnorm_right);
    // CLAMP into the wave fan, as the reference does (reimann.h:572). The denominator
    // rho_L(S_L-v_L) - rho_R(S_R-v_R) is bounded away from zero for well-separated states, but not
    // for near-degenerate ones, and an unclamped ratio then returns a contact speed orders of
    // magnitude outside the fan -- which is unphysical for the face flux and catastrophic if the
    // signal velocity is derived from it (measured 1.2e5 against a sound speed of 0.2).
    contact_speed = std::min(std::max(contact_speed, wave_left), wave_right);
    double contact_pressure = pressure_left
                            + density_left * (wave_left - vnorm_left) * (contact_speed - vnorm_left);
    // vacuum-adjacent guard; the tier-1 tests never reach it
    if (contact_pressure < 0) contact_pressure = 0.5*(pressure_left + pressure_right);
    return {contact_speed, contact_pressure};
}

// Refresh h, volume, density and pressure for the ACTIVE particles only. Inactive ones keep the
// values from their own last update, which is what makes an individual-timestep step cheap.
static void solve_h_and_volumes(Sim& sim, const Tree& tree,
                                const std::vector<uint32_t>& active) {
    const size_t n_part = sim.size();
    sim.h.resize(n_part); sim.ninv.resize(n_part);
    sim.rho.resize(n_part); sim.press.resize(n_part);
    sim.omega.resize(n_part, 1.0);
    if (!sim.eos_is_ideal()) sim.csnd.resize(n_part, 0.0);   // unused, and unallocated, for ideal gas

    std::vector<double> h_guess(active.size());
    const bool have_guess = !sim.h.empty();
    for (size_t k = 0; k < active.size(); ++k) h_guess[k] = have_guess ? sim.h[active[k]] : 0.0;
    if (!have_guess || h_guess.empty() || h_guess[0] <= 0) h_guess.clear();

#ifdef SHMEM_CACHE_NEIGHBORS
    NeighborCache* const ngb_cache = &sim.ngb_cache;
#else
    NeighborCache* const ngb_cache = nullptr;
#endif
    const DensityResult solved =
        density(tree, sim.P, active, sim.des_ngb, sim.ngb_tol, h_guess, sim.box, sim.dim, ngb_cache,
                sim.lazy());
    for (size_t k = 0; k < active.size(); ++k) sim.h[active[k]] = solved.h[k];

    // SHMEM_NGB_DIAG: how many tree traversals the h solve costs per target. Each Newton
    // iteration is a full traversal, so this is the multiplier on the density phase and it
    // decides whether grouping the solve is worth more than grouping the single-pass consumers.
    if (getenv("SHMEM_NGB_COUNT") && !active.empty()) {
        // Candidates EXAMINED against neighbours KEPT. The kept count should be ~DesNumNgb per
        // traversal; a large ratio means the search is opening nodes that hold nothing it wants,
        // which is a prune problem (the per-node vmax pad, or tree quality) rather than a cost
        // inherent to clustering.
        long long calls, nodes, examined, kept, pad_nodes;
        ngb_counters(calls, nodes, examined, kept, pad_nodes, /*reset=*/true);
        static int shown_c = 0;
        if (calls > 0 && shown_c < 6) {
            ++shown_c;
            fprintf(stderr, "[ngb-count] nact=%zu  calls=%lld  nodes/call=%.1f  examined/call=%.1f"
                            "  kept/call=%.1f  examined/kept=%.1f  pad-only nodes=%.1f/call "
                            "(%.1f%% of visits)\n",
                    active.size(), calls, (double)nodes/calls, (double)examined/calls,
                    (double)kept/calls, kept ? (double)examined/kept : 0.0,
                    (double)pad_nodes/calls, nodes ? 100.0*pad_nodes/nodes : 0.0);
        long long hh[8]; hiter_counters(hh, true);
        long long tot = 0; for (int q = 0; q < 8; ++q) tot += hh[q];
        if (tot > 0)
            fprintf(stderr, "[hiter] 1:%.1f%% 2:%.1f%% 3:%.1f%% 4:%.1f%% 5:%.1f%% 6:%.1f%% "
                    "7:%.1f%% 8+:%.1f%% of %lld solves\n",
                    100.0*hh[0]/tot, 100.0*hh[1]/tot, 100.0*hh[2]/tot, 100.0*hh[3]/tot,
                    100.0*hh[4]/tot, 100.0*hh[5]/tot, 100.0*hh[6]/tot, 100.0*hh[7]/tot, tot);
        }
    }
    if (getenv("SHMEM_NGB_DIAG") && !active.empty()) {
        long long sum = 0; int worst = 0;
        std::vector<int> hist(8, 0);
        for (size_t k = 0; k < active.size(); ++k) {
            const int it = solved.iters[k];
            sum += it; worst = std::max(worst, it);
            hist[std::min(it, 7)]++;
        }
        static int shown = 0;
        if (shown < 6) {
            ++shown;
            fprintf(stderr, "[ngb-diag] nact=%zu  mean iters=%.2f  max=%d  hist(0..7+)=",
                    active.size(), (double)sum / active.size(), worst);
            for (int c = 0; c < 8; ++c) fprintf(stderr, " %d", hist[c]);
            fprintf(stderr, "\n");
        }
    }

    // Adaptive-softening correction coefficients (GIZMO's AGS_zeta, gravity/ags_rkern.cc), rebuilt
    // here because this loop already owns exactly the neighbour set they integrate over. Refreshed
    // for ACTIVE particles only; inactive ones keep the value from their own last update, same as
    // every other AGS quantity.
    // SHMEM_NO_ZETA drops the Price & Monaghan correction entirely, to test whether the value
    // computed here is actually helping: the gating and the application form are verified
    // against forcetree.cc:2110-2124, but the magnitude is only checkable by measurement.
    static const bool no_zeta = getenv("SHMEM_NO_ZETA") != nullptr;
    const bool want_zeta = sim.gravity_on && sim.adaptive_soft && sim.dim == 3 && !no_zeta;
    if (want_zeta && sim.P.zeta.size() != n_part) sim.P.zeta.assign(n_part, 0.0);

    #pragma omp parallel
    {
        std::vector<uint32_t> neighbours;
        #pragma omp for schedule(dynamic, 64)
        for (size_t k = 0; k < active.size(); ++k) {
            const uint32_t i = active[k];
            const Vec3d pos_i = sim.P.pos(i);
            get_neighbours(sim, tree, k, pos_i, sim.h[i], neighbours);
            double weight_sum = 0, dn_dh = 0, dphi_dh_sum = 0;
            for (uint32_t j : neighbours) {
                if (j >= sim.n_gas) continue;   // hydro sums are over GAS neighbours only
                const Vec3d offset = min_image(sim.P.pos(j) - pos_i, sim.box);
                const double r = offset.norm();
                weight_sum += kernel_w(r, sim.h[i], sim.dim);
                // both sums INCLUDE the self term (r = 0), as GIZMO's do
                dn_dh += kernel_dwdh(r, sim.h[i], sim.dim);
                if (want_zeta) dphi_dh_sum += sim.P.m[j] * grav_dphi_dh(r, sim.h[i]);
            }
            sim.ninv[i]  = 1.0 / weight_sum;             // V_i: MFM volume from partition of unity
            sim.rho[i]   = sim.P.m[i] * weight_sum;      // rho_i = m_i / V_i
            eos_apply(sim, i);
            {
                // grad-h factor, GIZMO's DrkernNgbFactor: guards against pathological dn/dh as
                // GIZMO does (the -0.9 test), else 1
                const double omega_pre = sim.h[i] / (sim.dim * weight_sum) * dn_dh;
                sim.omega[i] = (omega_pre > -0.9) ? 1.0 / (1.0 + omega_pre) : 1.0;
            }

            if (want_zeta) {
                // zeta_i = m_i^2 * Omega_i^-1 * [ 0.5 * (sum m_j dphi/dh) * h / (NDIMS m_i n_i) ],
                // with Omega the usual grad-h factor 1 + (h / NDIMS n) dn/dh, guarded as GIZMO
                // guards it. Zero when the softening floor is binding: then eps is CONSTANT and
                // there is no dPhi/dh term to correct for.
                double zeta = 0.0;
                if (sim.h[i] > sim.soft_min && weight_sum > 0) {
                    zeta = 0.5 * sim.P.m[i] * sim.omega[i] * dphi_dh_sum * sim.h[i]
                           / (3.0 * weight_sum);
                }
                sim.P.zeta[i] = zeta;
            }
        }
    }

}

static void gradients(Sim& sim, const Tree& tree, const std::vector<uint32_t>& active) {
    Work& work = sim.work;
    work.resize(sim.size());
    #pragma omp parallel
    {
        std::vector<uint32_t> neighbours;
        #pragma omp for schedule(dynamic, 64)
        for (size_t k = 0; k < active.size(); ++k) {
            const uint32_t i = active[k];
            const Vec3d pos_i = sim.P.pos(i);
            // PREDICTED velocity, not stored: the stored one is leapfrog bookkeeping, half a
            // step behind the state this pass is evaluating (see set_predicted_states). The
            // reference's gradient loop reads VelPred for the same reason.
            const Vec3d vel_i{work.predicted[FIELD_VX][i], work.predicted[FIELD_VY][i],
                              work.predicted[FIELD_VZ][i]};
            get_neighbours(sim, tree, k, pos_i, sim.h[i], neighbours);

            SymTensor3d moments{0,0,0,0,0,0};     // E_i = sum_j (dx ox dx) W_ij
            Vec3d face_w{0, 0, 0};                // sum_j dx W_ij -- first moment, for face closure
            // Signal speed, Monaghan (1997): c_i + c_j minus the APPROACH speed along the pair
            // axis. Built only from RELATIVE velocities, so a uniform boost of the whole domain
            // leaves it unchanged -- the square test advects at |v|~1300 and a lab-frame |v| here
            // would shrink dt by ~1700x for a flow that is trivially Galilean-equivalent to rest.
            const double csound_i = sound_speed(sim, i);
            double signal_speed = 2.0 * csound_i;   // floor: the i==j / no-neighbour case
            for (uint32_t j : neighbours) {
                if (j >= sim.n_gas) continue;   // hydro sums are over GAS neighbours only
                const Vec3d offset = min_image(sim.P.pos(j) - pos_i, sim.box);
                const double separation = offset.norm();
                const double weight = kernel_w(separation, sim.h[i], sim.dim);
                moments += outer_product(offset) * weight;
                face_w += offset * weight;
                if (separation > 0) {
                    // SIGN: GIZMO forms vdotr2 = (x_i - x_j).(v_i - v_j) and boosts vsig when it is
                    // NEGATIVE. `offset` here is x_j - x_i, the opposite sense, so the pair is
                    // CLOSING when approach_speed is positive and the boost is applied then.
                    // Getting this backwards costs the velocity term exactly where it matters --
                    // converging flow, i.e. every shock front and every collapse -- and leaves
                    // those cells on a sound-crossing step they cannot resolve.
                    const Vec3d rel_vel = vel_i - Vec3d{work.predicted[FIELD_VX][j],
                                                        work.predicted[FIELD_VY][j],
                                                        work.predicted[FIELD_VZ][j]};
                    const double approach_speed = dot(rel_vel, offset) / separation;
                    const double csound_j = sound_speed(sim, j);
                    signal_speed = std::max(signal_speed,
                                            csound_i + csound_j + std::max(0.0, approach_speed));
                }
            }
            work.signal_speed[i] = signal_speed;

            Mat3d& moments_inv = work.moments_inv[i];
            if (!invert_moments(moments, moments_inv, sim.dim)) {
                if (work.face_closure.size() == sim.size()) work.face_closure[i] = 0.0;
                // degenerate neighbour geometry: fall back to zero gradient (first order but
                // safe). Must be written explicitly -- these arrays persist between steps now,
                // so "leave it alone" would silently reuse a stale gradient forever.
                for (auto& field_gradient : work.gradient) field_gradient[i] = Vec3d{};
                continue;
            }

            // FACE CLOSURE (density.cc:546-573). The effective faces around a cell should sum to
            // zero; the one-sided estimator 2 V_i B . (sum_j dx W) is what is left over, and
            // normalising it by the cell's cross-section gives a dimensionless leak. GIZMO widens
            // the kernel where this exceeds 0.35 -- see the ncorr pass in solve_h_and_volumes.
            if (work.face_closure.size() == sim.size()) {
                const double vol_i = sim.ninv[i];                       // 1/sum W = V_i
                // dx_i is NOT V^(1/dim): density.cc:545 computes it as V^(1/dim) and then
                // immediately OVERWRITES it with sqrt(V_i * trace(E)) -- the root-mean-square
                // neighbour distance, weighted by the kernel. Using the geometric size instead
                // underestimates the closure error by a factor of ~2 here.
                const double dx_i = std::sqrt(vol_i * moments.trace());
                const Vec3d leak  = moments_inv.matvec(face_w) * (2.0 * vol_i);
                double sum_abs = 0.0;
                for (int c = 0; c < 3; ++c) sum_abs += std::abs(leak[c]) / sim.dim;
                const double denom = 2.0 * sim.dim * std::pow(dx_i, sim.dim - 1);
                work.face_closure[i] = (denom > 0) ? sum_abs / denom : 0.0;
            }

            // gradients of the PREDICTED fields on both sides, like every other hydro input
            const PrimitiveState field_i{work.predicted[FIELD_DENSITY][i],
                                         work.predicted[FIELD_VX][i],
                                         work.predicted[FIELD_VY][i],
                                         work.predicted[FIELD_VZ][i],
                                         work.predicted[FIELD_PRESSURE][i]};
            std::array<Vec3d, NUM_FIELDS> weighted_diff_sum{};
            // widest rise and fall seen across the neighbour set, for the slope limiter below
            PrimitiveState largest_rise{}, largest_drop{};
            double max_ngb_distance = 0.0;
            for (uint32_t j : neighbours) {
                if (j >= sim.n_gas) continue;   // hydro sums are over GAS neighbours only
                const Vec3d offset = min_image(sim.P.pos(j) - pos_i, sim.box);
                const double weight = kernel_w(offset.norm(), sim.h[i], sim.dim);
                max_ngb_distance = std::max(max_ngb_distance, offset.norm());
                const PrimitiveState field_j{work.predicted[FIELD_DENSITY][j],
                                             work.predicted[FIELD_VX][j],
                                             work.predicted[FIELD_VY][j],
                                             work.predicted[FIELD_VZ][j],
                                             work.predicted[FIELD_PRESSURE][j]};
                for (int f = 0; f < NUM_FIELDS; ++f) {
                    const double diff = field_j[f] - field_i[f];
                    weighted_diff_sum[f] += offset * (diff * weight);
                    largest_rise[f] = std::max(largest_rise[f], diff);
                    largest_drop[f] = std::min(largest_drop[f], diff);
                }
            }
            // GIZMO's per-field limiter settings (hydro/gradients.cc): reach h_lim is the larger
            // of the kernel radius and the farthest neighbour; overshoot tolerance 0.1 for pure
            // hydro, 0 with gravity on; density and pressure positivity-preserved.
            const double h_lim = std::max(sim.h[i], max_ngb_distance);
            const double stol = sim.gravity_on ? 0.0 : 0.1;
            // sqrt(||E|| ||E^-1||)/NUMDIMS, GIZMO's matrix_invert_ndims (system/system.cc:195).
            // ~1 for a well-conditioned neighbour geometry.
            double frob_e = 0.0, frob_inv = 0.0;
            for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) {
                frob_e   += moments[a][b] * moments[a][b];
                frob_inv += moments_inv[a][b] * moments_inv[a][b];
            }
            const double cond_num =
                std::max(std::sqrt(frob_e * frob_inv) / sim.dim, 1.0);
            work.condition_number[i] = cond_num;
            const double a_lim = a_limiter_for(cond_num);
            for (int f = 0; f < NUM_FIELDS; ++f) {
                Vec3d gradient = moments_inv.matvec(weighted_diff_sum[f]);
                const bool pos_preserve = (f == FIELD_DENSITY || f == FIELD_PRESSURE);
                limit_slope(gradient, largest_rise[f], largest_drop[f], h_lim,
                            (f == FIELD_DENSITY) ? 0.0 : stol, pos_preserve,
                            max_ngb_distance, field_i[f], a_lim);
                work.gradient[f][i] = gradient;
            }
            // kept for the drift-time prediction of INACTIVE particles
            work.div_vel[i] = work.gradient[FIELD_VX][i][0]
                            + work.gradient[FIELD_VY][i][1]
                            + work.gradient[FIELD_VZ][i][2];
        }
    }
}

// Publish each active particle's PREDICTED primitives -- its best estimate of the true state at
// the current time -- as its face states. The STORED velocity and internal energy are leapfrog
// bookkeeping: after the fused kick they sit half a step ahead of the state they were evaluated
// at, and the half owed back (pending_half_kick) completes only at the next kick. The gradient
// and flux passes must never see that half-stale state: the reference reads VelPred and
// InternalEnergyPred everywhere in its hydro loops, and predict.cc advances those to the current
// time with the SAME rates the kick will use -- which is exactly stored + rate * owed. Publishing
// the stored state instead feeds an O(dt) input into an otherwise second-order scheme, and
// measurably drops the soundwave's convergence below second order.
// There is no half-step time extrapolation beyond this: the reference's face reconstruction is
// spatial only (hydro_core_meshless.h), and time centring comes from splitting the flux rate
// across the two half-kicks.
static void set_predicted_states(Sim& sim, const std::vector<uint32_t>& active) {
    Work& work = sim.work;
    work.resize(sim.size());
    const size_t n = sim.size();
    const bool have_pend = sim.pending_half_kick.size() == n;
    const bool have_grav = sim.gravity_on && sim.a_grav.size() == n;
    const bool have_hyd = sim.a_hydro.size() == n;
    const bool have_du = sim.du_dt.size() == n;
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < active.size(); ++k) {
        const uint32_t i = active[k];
        const double owed = have_pend ? sim.pending_half_kick[i] : 0.0;
        Vec3d a = have_grav ? sim.a_grav[i] : Vec3d{0, 0, 0};
        if (have_hyd) a += sim.a_hydro[i];
        work.predicted[FIELD_VX][i] = sim.vx[i] + a[0] * owed;
        work.predicted[FIELD_VY][i] = sim.vy[i] + a[1] * owed;
        work.predicted[FIELD_VZ][i] = sim.vz[i] + a[2] * owed;
        // density is genuinely current -- the h/volume solve just ran at the drifted positions
        work.predicted[FIELD_DENSITY][i] = sim.rho[i];
        // pressure follows the predicted internal energy under an ideal EOS; the density-driven
        // laws depend on rho alone, and the stored pressure already carries the fresh rho
        double press = sim.press[i];
        if (sim.eos_law == Sim::EosLaw::IDEAL && have_du) {
            const double u_pred = std::max(sim.u[i] + sim.du_dt[i] * owed, 1e-30);
            press = (sim.gamma - 1.0) * sim.rho[i] * u_pred;
        }
        work.predicted[FIELD_PRESSURE][i] = press;
    }
}

// Self-gravity at the current positions, into sim.a_grav.
//
// Softening: with adaptive_soft the gas softens on its OWN kernel radius, which is what
// ADAPTIVE_GRAVSOFT_FORGAS means and what the evrard config asks for. That keeps the gravitational
// and hydrodynamic resolution the same everywhere, so a collapsing region does not end up with
// pressure resolved on a scale the gravity has smoothed away (or the reverse). soft_min is the
// floor from SofteningGas.
// Softening is a property of every particle, active or not: the tree's mass distribution is
// sourced by ALL of them, so this loop stays global even when the forces do not. Gas softens on
// its own kernel radius under adaptive_soft (floored by soft_min) or sits at the fixed gas value;
// collisionless types take their fixed kernel-extent softening from soft_fixed.
// ACTIVE-SET ONLY once the table is valid. soft[i] is a pure function of h[i] for gas and of the
// fixed table for everything else, and h only ever changes for ACTIVE gas -- the h solve runs on
// the active set. So an inactive particle's softening is already correct, and rewriting all of it
// every step is pure O(N) waste. Measured on the 3.5e6-cell bate cloud: this pass sat inside a
// gravity phase that cost 80 ms on a step moving 21 particles.
//
// A FULL pass is still required whenever the layout could have moved under us -- first call, a
// changed particle count, or a changed gas/non-gas split, which is what sink formation does when it
// converts a cell and re-sorts the arrays. Those are exactly the cases where an index no longer
// means what it did last step.
static void update_softenings(Sim& sim, const std::vector<uint32_t>* active = nullptr) {
    const size_t n_part = sim.size();
    sim.P.soft.resize(n_part);
    auto set_one = [&sim](size_t i) {
        if (i < sim.n_gas)
            sim.P.soft[i] = sim.adaptive_soft ? std::max(sim.h[i], sim.soft_min) : sim.soft_min;
        else
            sim.P.soft[i] = sim.soft_fixed[sim.P.type.empty() ? 1 : sim.P.type[i]];
    };
    const bool full = !active || sim.soft_valid_n != n_part || sim.soft_valid_ngas != sim.n_gas;
    if (full) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_part; ++i) set_one(i);
        sim.soft_valid_n = n_part; sim.soft_valid_ngas = sim.n_gas;
    } else {
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < active->size(); ++k) set_one((*active)[k]);
    }
}

// ADAPTIVE_TREEFORCE_UPDATE: does this particle need a real tree walk this step, or can it keep
// its cached acceleration advanced by the jerk? Mirrors needs_new_treeforce (gravtree.cc:939-951).
//
// Only GAS is ever lazy. A Hermite-integrated type must always be fresh -- its predictor-corrector
// sub-stepping is incompatible with a cached force (gravtree.cc:941) -- and the reference restricts
// the lazy path to type 0 regardless (:943), because everything else is a sink whose orbit is the
// thing being resolved.
static inline bool atu_needs_fresh(const Sim& sim, size_t i) {
    if (sim.atu_frac <= 0) return true;
    if (i >= sim.n_gas) return true;                       // non-gas: always fresh
    if (!sim.P.type.empty() && (sim.hermite_mask & (1 << sim.P.type[i]))) return true;
    // Nothing cached yet -- first walk, or a particle created since the last one.
    if (sim.a_grav.size() != sim.size() || sim.a_grav_jerk.size() != sim.size()) return true;
    if (!(sim.tdyn_for_treeforce[i] > 0)) return true;
    return sim.time_since_treeforce[i] >= sim.atu_frac * sim.tdyn_for_treeforce[i];
}

static void compute_gravity(Sim& sim, const Tree& tree, const std::vector<uint32_t>& active) {
    const size_t n_part = sim.size();
    update_softenings(sim, &active);

    // BATCH SPATIAL COHERENCE. accel_grouped walks once per batch of 8 targets and opens the
    // UNION of what the batch needs, so a batch only pays off when its 8 targets are spatially
    // close. The active list is INDEX-ordered, which is spatially coherent only when the IC
    // happened to be written in a space-filling order -- lattices are, but plummer's IC is
    // radius-sorted, so consecutive indices are 8 same-shell targets scattered across the whole
    // sphere and the union walk opened most of the tree per batch: measured 5x slower per target
    // than pytreegrav's reference walk at the same theta on the same machine. Re-emitting the
    // actives in the tree's own Morton order makes every batch compact for ANY IC ordering, at
    // one O(N) pass -- the same cost class as the active-set scan that runs every sync anyway.
    // Three ways to produce the same Morton-ordered active list; which is cheapest depends only on
    // how many are active. The scan is O(N) whatever the answer's size, which on a deep hierarchy
    // means a step moving 8 particles paid the same 0.92 ms as one moving all 128k -- 28% of a
    // small step, and it does not parallelise.
    std::vector<uint32_t>& targets = sim.grav_targets;
    const bool have_rank = tree.rank.size() == n_part;
    if (active.size() >= n_part) {
        // everything active: the Morton order IS orderbuf, no filtering needed
        targets.assign(tree.orderbuf.begin(), tree.orderbuf.end());
    } else if (have_rank && active.size() < n_part / 8) {
        // small active set: sort it by tree rank, O(k log k). Only below the crossover -- sorting
        // the whole set would be O(N log N), strictly worse than the O(N) scan it replaces.
        targets.assign(active.begin(), active.end());
        std::sort(targets.begin(), targets.end(),
                  [&rk = tree.rank](uint32_t a, uint32_t b) { return rk[a] < rk[b]; });
    } else {
        targets.clear();
        targets.reserve(active.size());
        for (size_t r = 0; r < n_part; ++r) {
            const uint32_t i = tree.orderbuf[r];
            if (sim.is_active(i)) targets.push_back(i);
        }
    }

    // ATU: split the Morton-ordered active set into those needing a real walk and those keeping a
    // jerk-advanced cached force. Partitioned IN PLACE so the walk's target list stays contiguous
    // and stays in Morton order -- the batch walk's entire advantage rests on that ordering.
    std::vector<uint32_t> atu_skipped;
    if (sim.atu_frac > 0) {
        size_t keep = 0;
        for (size_t k = 0; k < targets.size(); ++k) {
            const uint32_t i = targets[k];
            if (atu_needs_fresh(sim, i)) targets[keep++] = i;
            else                         atu_skipped.push_back(i);
        }
        targets.resize(keep);
    }

    std::vector<double> ax, ay, az;
    // grouped walk: one traversal per batch of 8, measured 1.71x over the per-target walk.
    // The tidal tensor rides along in the same walk when the tidal timestep criterion wants it.
    std::vector<SymTensor3d>* tidal_out = nullptr;
    std::vector<SymTensor3d> tidal_active;
    if (sim.tidal_criterion) { tidal_out = &tidal_active; }
    // Relative opening criterion needs |a| from each target's PREVIOUS force evaluation
    // (GIZMO's OldAcc, in the same no-G units the walk accumulates). Zero -- including the whole
    // first step -- falls back to the geometric test inside open_node.
    std::vector<double> aold_active;
    const double* aold_ptr = nullptr;
    if (sim.err_tol_force_acc > 0 && sim.a_grav.size() == n_part) {
        aold_active.resize(targets.size());
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < targets.size(); ++k)
            aold_active[k] = sim.err_tol_force_acc * sim.a_grav[targets[k]].norm() / sim.G;
        aold_ptr = aold_active.data();
    }
    // SHMEM_CUDA_GRAVITY swaps the tree walk for an O(N^2) GPU direct sum that applies the SAME
    // pair force (grav_cuda.cu). The tree's opening approximation is then identically absent, so
    // whatever momentum the gravity kick still injects is the force LAW and the timestep
    // structure, not the walk. Diagnostic: it is O(N_active * N) and only affordable because the
    // active set is small for most of a collapse.
    // NODE-ALIGNED GROUPING (0), not a fixed batch. Fixed windows of 8 are catastrophic once the
    // active set is sparse: 32 random targets out of 3.5e6 span the whole cloud, so all four
    // windows open essentially the entire tree. Measured on the M50 state, accel+tidal+jerk:
    //
    //     nactive  layout       b=1     b=8    node     <- ms, 16 threads
    //     32       scattered    2.7   563.4     2.3
    //     512      scattered   18.3  1034.9    18.0
    //     35000    scattered  427.5  1160.1   426.7
    //     24776    compact    147.7    97.4   105.8
    //     3500000  all      23178.7 15008.2 15845.7
    //
    // Grouping on the leaf is within ~10% of the best fixed batch everywhere (worst case 5.6% on
    // the all-active step, where b=8 wins) and up to 245x better where the run actually spends its
    // steps. The old comment here recorded that shrinking the batch was a loss, measured at nact=8
    // on a DENSE clump -- true, and the reason the answer is grouping rather than a smaller batch:
    // it keeps the amortisation when targets really are close and drops it when they are not,
    // with nothing to tune.
    static const int grav_batch = getenv("SHMEM_GRAV_BATCH")
                                ? atoi(getenv("SHMEM_GRAV_BATCH")) : 0;   // diagnostic override
    // The jerk is what makes a skipped force usable: the cached acceleration is advanced as
    // a += j*dt rather than merely reused stale. It costs extra work in every walk that DOES run,
    // which is why the reference gates the whole scheme behind a flag.
    std::vector<Vec3d> jerk_active;
    std::vector<Vec3d>* jerk_out = (sim.atu_frac > 0) ? &jerk_active : nullptr;
    const double* vel3[3] = {sim.vx.data(), sim.vy.data(), sim.vz.data()};
    accel_grouped(tree, sim.P, targets, sim.theta, sim.G, grav_batch, ax, ay, az, tidal_out,
                  aold_ptr, sim.lazy(), jerk_out, (sim.atu_frac > 0 ? vel3 : nullptr),
                  sim.sink_direct_radius);

    // SHMEM_CUDA_GRAVITY replaces the walk's ACCELERATION with an O(N^2) GPU direct sum applying
    // the same pair force (grav_cuda.cu), leaving the tidal tensor to the walk -- that feeds the
    // timestep criterion, and recomputing it here would change the run for a reason unrelated to
    // the force. With the opening approximation identically absent, whatever momentum the gravity
    // kick still injects is the force LAW and the timestep structure, not the tree.
#ifdef SHMEM_CUDA
    static const bool cuda_grav = getenv("SHMEM_CUDA_GRAVITY") != nullptr;
    if (cuda_grav) {
        static std::vector<double> zeta_all, eps_all;
        static std::vector<char> gas_all;
        static std::vector<int> tidx;
        zeta_all.assign(n_part, 0.0);
        if (sim.P.zeta.size() == n_part) zeta_all = sim.P.zeta;
        eps_all.assign(n_part, 0.0);
        for (size_t i = 0; i < n_part; ++i) eps_all[i] = sim.P.soft.empty() ? 0.0 : sim.P.soft[i];
        gas_all.assign(n_part, 0);
        for (size_t i = 0; i < n_part; ++i) gas_all[i] = sim.P.is_gas(i) ? 1 : 0;
        tidx.assign(targets.size(), 0);
        for (size_t k = 0; k < targets.size(); ++k) tidx[k] = (int)targets[k];
        shmem_cuda_accel_bruteforce((int)n_part, sim.P.x.data(), sim.P.y.data(), sim.P.z.data(),
                                    sim.P.m.data(), eps_all.data(), zeta_all.data(), gas_all.data(),
                                    (int)targets.size(), tidx.data(), sim.box,
                                    ax.data(), ay.data(), az.data());
        for (size_t k = 0; k < targets.size(); ++k) { ax[k] *= sim.G; ay[k] *= sim.G; az[k] *= sim.G; }
    }
#endif
    if (sim.tidal_criterion) {
        sim.tidal.resize(n_part);
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < targets.size(); ++k) sim.tidal[targets[k]] = tidal_active[k];
    }

    // SHMEM_EXACT_GRAVITY: replace the walk with the exact pairwise sum. Unlike the radial
    // projection below this removes NO physics -- it is the same force law, summed without
    // approximation -- and it is torque-free to round-off because pair_force_over_r is symmetric,
    // so Newton's third law holds exactly. That makes it the controlled way to ask whether the
    // tree's spurious torque drives the fragmentation. Affordable only because the cost is
    // O(N_active * N) and N_active is small away from full syncs; needs SHMEM_DENSE_DRIFT so every
    // source position is current, since this does not use the lazy-drift hook the walk does.
    static const bool exact_grav = getenv("SHMEM_EXACT_GRAVITY") != nullptr;
    if (exact_grav) accel_brute(sim.P, targets, sim.G, ax, ay, az);

    sim.a_grav.resize(n_part);
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < targets.size(); ++k)
        sim.a_grav[targets[k]] = Vec3d{ax[k], ay[k], az[k]};
    if (sim.atu_frac > 0) {
        sim.a_grav_jerk.resize(n_part);
        sim.time_since_treeforce.resize(n_part, 0.0);
        // Fresh: cache the jerk and reset the age to this step's dt, so the counter measures how
        // old the cached force will be by the end of the step (gravtree.cc:508).
        const bool have_jerk = jerk_active.size() == targets.size();
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < targets.size(); ++k) {
            const uint32_t i = targets[k];
            if (have_jerk) sim.a_grav_jerk[i] = jerk_active[k];
            sim.time_since_treeforce[i] = sim.time_of_ticks(sim.ticks_in_bin(sim.bin[i]));
        }
        // Skipped: advance the cached acceleration with its jerk and age it (gravtree.cc:504-505).
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < atu_skipped.size(); ++k) {
            const uint32_t i = atu_skipped[k];
            const double dt = sim.time_of_ticks(sim.ticks_in_bin(sim.bin[i]));
            sim.a_grav[i] += sim.a_grav_jerk[i] * dt;
            sim.time_since_treeforce[i] += dt;
        }
    }

    // SHMEM_RADIAL_GRAVITY: keep only the component of gravity along the line to the collapse
    // centre, discarding the transverse part. A central field exerts no torque about that point BY
    // CONSTRUCTION, so this cannot conserve angular momentum better by accident -- it removes the
    // spurious torque and a great deal of real physics with it. Purely a causal test, never a fix.
    if (getenv("SHMEM_RADIAL_GRAVITY") && n_part > 0) {
        // ONLY once a sink exists. Radialising before that guts local self-gravity everywhere -- a
        // clump's pull on its own gas is almost entirely tangential in the global frame, so nothing
        // can become self-bound, the centre accumulates a pressure-supported blob and the first
        // sink forms ~9x late (measured). Picking the heaviest particle at t=0 is equally wrong:
        // every gas cell has the same mass, so the choice lands on an arbitrary one.
        size_t heavy = n_part;
        for (size_t i = sim.n_gas; i < n_part; ++i)
            if (!sim.P.type.empty() && sim.P.type[i] == 5 &&
                (heavy == n_part || sim.P.m[i] > sim.P.m[heavy])) heavy = i;
        if (heavy == n_part) return;                 // no sink yet: leave gravity alone
        // WHICH centre matters more than it looks. A force aimed at the sink exerts no torque about
        // the sink instantaneously, but the sink ACCELERATES, so that frame is non-inertial. Worse,
        // the sink does not sit at the centre of the density cusp -- measured 0.80 r_sink away here
        // against the reference's 0.26 -- and a force aimed at a point offset by d still torques
        // about the true mass centre, with a tangential fraction ~d/r: about 20% at r = 4 r_sink,
        // precisely where the spin-up is measured. So a null result from aiming at the sink means
        // nothing. "cusp" aims at the density-weighted centroid, the centre that actually has to be
        // torque-free; "box" aims at a fixed inertial point, additionally removing the sink's own
        // acceleration.
        const char* mode = getenv("SHMEM_RADIAL_GRAVITY");
        Vec3d centre = sim.P.pos(heavy);
        if (mode && strcmp(mode, "box") == 0) {
            centre = Vec3d{0.0, 0.0, 0.0};
        } else if (mode && strcmp(mode, "cusp") == 0) {
            const double r_cut = 30.0 * (sim.sink_radius.size() == n_part && sim.sink_radius[heavy] > 0
                                         ? sim.sink_radius[heavy] : 1.0248e-05);
            Vec3d num{0, 0, 0}; double den = 0;
            for (size_t i = 0; i < sim.n_gas; ++i) {
                const Vec3d d = min_image(sim.P.pos(i) - centre, sim.box);
                if (d.norm() > r_cut) continue;
                const double w = sim.rho[i] * sim.P.m[i];      // density-weighted, as the diagnostic
                num += sim.P.pos(i) * w; den += w;
            }
            if (den > 0) centre = num / den;
        }
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < targets.size(); ++k) {
            const size_t i = targets[k];
            if (i == heavy) continue;
            const Vec3d d = min_image(sim.P.pos(i) - centre, sim.box);
            const double r = d.norm();
            if (!(r > 0)) continue;
            const Vec3d rhat = d / r;
            sim.a_grav[i] = rhat * dot(sim.a_grav[i], rhat);
        }
    }
}

// Flux exchange over unique pairs. Pair discovery from the SMALLER kernel side would miss
// asymmetric pairs, so: i owns the pair when (i < j) and r < max(h_i, h_j); every pair is then
// found exactly once because both sides search with max(h_i, h_j) coverage via the union below.
static void fluxes(Sim& sim, const Tree& tree, const std::vector<uint32_t>& active,
                   std::vector<double>& dmom_x, std::vector<double>& dmom_y,
                   std::vector<double>& dmom_z, std::vector<double>& denergy) {
    const Work& work = sim.work;
    const size_t n_part = sim.size();
    // NOT cleared here. The accumulators are left zeroed by the rate-conversion pass that
    // consumes them (see evaluate_forces), so the zeroing rides along in a loop over the same
    // actives instead of costing a separate sweep of the arrays.
    if (dmom_x.size() != n_part) {
        dmom_x.assign(n_part, 0); dmom_y.assign(n_part, 0);
        dmom_z.assign(n_part, 0); denergy.assign(n_part, 0);
    }
    sim.wake_requests.clear();
    // Staged, not written in place: the wakeup test reads its NEIGHBOUR's recorded signal speed,
    // so overwriting the live array mid-loop would have it compare against values already updated
    // this step by other threads. The reference has the same separation -- it accumulates into a
    // per-target `out` and merges afterwards.
    static std::vector<double> vsig_new;
    const bool contact_vsig = sim.contact_wave_vsig;
    if (contact_vsig) vsig_new.assign(n_part, 0.0);
    #pragma omp parallel
    {
        std::vector<uint32_t> neighbours;
        std::vector<std::pair<uint32_t,int>> local_wakes;
        // Bins a woken cell is demoted BY: the smallest n with 2^n >= WAKEUP, so its step is at
        // least WAKEUP times shorter than the waker's (core/timestep.cc:1332). 3, for WAKEUP=4.1.
        int wake_offset = 0;
        while ((1 << wake_offset) < sim.wakeup_fac) ++wake_offset;
        #pragma omp for schedule(dynamic, 64) nowait
        for (size_t k = 0; k < active.size(); ++k) {
            const uint32_t i = active[k];
            // search with h_i; pairs where h_j > r >= h_i are found from j's side (j also loops)
            const Vec3d pos_i = sim.P.pos(i);
            get_neighbours(sim, tree, k, pos_i, sim.h[i], neighbours);
            // MFM+GALSF signal velocity, accumulated HERE rather than in the density pass, because
            // the reference derives it from the Riemann solve: vsig = 2*S_M + max(0, dv_face)
            // (hydro_core_meshless.h:253), replacing the Monaghan cs_i+cs_j estimate. Only the
            // OWNER i is updated, exactly as the reference does -- it writes j's copy only under
            // j_is_active_for_fluxes, which is never set -- so this needs no atomic.
            //
            // SEEDED with i's own sound speed, not zero (hydro_evaluate.h:82,
            // out.MaxSignalVel = kernel.sound_i). This is the floor, and the pair loop below only
            // ever raises it: 2*S_M carries no lower bound of its own, so a cell in near
            // equilibrium sees S_M ~ 0 on every pair and would otherwise end the step with vsig 0
            // and an unbounded Courant limit.
            double vsig_i = sound_speed(sim, i);
            for (uint32_t j : neighbours) {
                if (j == i) continue;
                if (j >= sim.n_gas) continue;   // fluxes are exchanged between gas pairs only
                const Vec3d offset = min_image(sim.P.pos(j) - pos_i, sim.box);
                const double separation = offset.norm();
                if (separation <= 0) continue;
                // OWNERSHIP. Because the loop now accumulates rates rather than time-integrated
                // amounts, the only thing to avoid is counting a pair twice in the SAME sync --
                // there is no cadence to match. If j is inactive it contributes nothing this sync
                // and takes this pair up from its own side later, so i simply does it. If both are
                // active, the lower index owns it (provided it can see the pair; an owner outside
                // whose kernel the pair falls cannot process it, so the other side keeps it).
                if (sim.is_active(j) && j < i && separation < sim.h[j]) continue;
                const double weight_i = kernel_w(separation, sim.h[i], sim.dim);
                const double weight_j = kernel_w(separation, sim.h[j], sim.dim);

                // effective face A_ij (points i -> j): wt_i B_i dx W_i(r) + wt_j B_j dx W_j(r).
                //
                // CENTRED WEIGHTS (compute_finitevol_faces.h:15-18). The Lanson & Vila form takes
                // wt = V on each side, which -- as the reference's own comment says -- assumes
                // negligible variation in h between neighbours. In a collapse that is false
                // everywhere, and the volume gradient then TILTS the face: the pair force stays
                // antisymmetric (momentum is safe) but its direction is biased, which is a torque.
                // Once the volumes differ by more than 1.25 per dimension GIZMO switches both
                // sides to one centred weight, recovering a face that does not lean on the side
                // with the larger cell.
                const double V_i = sim.ninv[i], V_j = sim.ninv[j];
                double wt_i = V_i, wt_j = V_j;
                const double vmin = std::min(V_i, V_j);
                if (vmin > 0 && (std::abs(V_i - V_j) / vmin) / sim.dim > 1.25) {
                    const double den = V_i * weight_i + V_j * weight_j;
                    if (den > 0) wt_i = wt_j = V_i * V_j * (weight_i + weight_j) / den;
                }
                Vec3d face = work.moments_inv[i].matvec(offset) * (wt_i * weight_i)
                           + work.moments_inv[j].matvec(offset) * (wt_j * weight_j);
                // DEGENERATE GEOMETRY (compute_finitevol_faces.h:56-60). For a positive-definite
                // E both terms of A.dx are positive, so a negative projection means the moment
                // matrix has degenerated numerically; the same applies when the condition numbers
                // blow up. GIZMO then falls back to a face along the SEPARATION, which is exactly
                // central and therefore contributes no torque at all for that pair.
                const bool cond_ok = work.condition_number.size() == sim.size();
                const double cn_i = cond_ok ? work.condition_number[i] : 0.0;
                const double cn_j = cond_ok ? work.condition_number[j] : 0.0;
                if (dot(face, offset) < 0 ||
                    cn_i*cn_i + cn_j*cn_j > 1.0e12 + CONDITION_NUMBER_DANGER*CONDITION_NUMBER_DANGER) {
                    const double dwk_i = kernel_dwdr(separation, sim.h[i], sim.dim);
                    const double dwk_j = kernel_dwdr(separation, sim.h[j], sim.dim);
                    face = offset * (-(wt_i*V_i*dwk_i + wt_j*V_j*dwk_j) / separation);
                }
                const double face_area = face.norm();
                if (face_area <= 0) continue;
                const Vec3d normal = face / face_area;

                // linear reconstruction to the face midpoint, pairwise minmod-limited per field
                const Vec3d to_midpoint = offset * 0.5;
                // gradient extrapolation to the face midpoint (sign: +1 from i, -1 from j)
                auto extrapolate = [&](int field, size_t side, double sign)->double {
                    return sign * dot(work.gradient[field][side], to_midpoint);
                };
                // Pairwise face limiter, ported from GIZMO's reconstruct_face_states: overshoot
                // beyond the pair range allowed up to half the jump, deviation from the midpoint
                // capped at 0.375 of it, signs preserved. See limit_face_pair for why this beats
                // a hard clamp into [min,max] at shocks.
                PrimitiveState left{}, right{};
                for (int f = 0; f < NUM_FIELDS; ++f) {
                    const auto& predicted = work.predicted[f];
                    left[f]  = predicted[i] + extrapolate(f, i, +1.0);
                    right[f] = predicted[j] + extrapolate(f, j, -1.0);
                    limit_face_pair(predicted[i], predicted[j], left[f], right[f]);
                }
                if (left[FIELD_DENSITY]  <= 0 || right[FIELD_DENSITY]  <= 0 ||
                    left[FIELD_PRESSURE] <= 0 || right[FIELD_PRESSURE] <= 0) {  // limiter emergency
                    left[FIELD_DENSITY]   = work.predicted[FIELD_DENSITY][i];
                    left[FIELD_PRESSURE]  = work.predicted[FIELD_PRESSURE][i];
                    right[FIELD_DENSITY]  = work.predicted[FIELD_DENSITY][j];
                    right[FIELD_PRESSURE] = work.predicted[FIELD_PRESSURE][j];
                }

                // face frame: mean velocity; rotate the states onto the normal
                const Vec3d face_vel{
                    0.5*(work.predicted[FIELD_VX][i] + work.predicted[FIELD_VX][j]),
                    0.5*(work.predicted[FIELD_VY][i] + work.predicted[FIELD_VY][j]),
                    0.5*(work.predicted[FIELD_VZ][i] + work.predicted[FIELD_VZ][j])};
                const double vnorm_left = dot(
                    Vec3d{left[FIELD_VX], left[FIELD_VY], left[FIELD_VZ]} - face_vel, normal);
                const double vnorm_right = dot(
                    Vec3d{right[FIELD_VX], right[FIELD_VY], right[FIELD_VZ]} - face_vel, normal);
                // Effective index per side, cs^2 rho / P, so the reconstructed face state keeps
                // the local barotropic stiffness instead of being forced back onto gamma.
                const double geff_i = sim.eos_is_ideal() ? sim.gamma
                    : sound_speed(sim, i)*sound_speed(sim, i) * sim.rho[i] / sim.press[i];
                const double geff_j = sim.eos_is_ideal() ? sim.gamma
                    : sound_speed(sim, j)*sound_speed(sim, j) * sim.rho[j] / sim.press[j];
                const auto [contact_speed, contact_pressure] = solve_hllc_contact(
                    left[FIELD_DENSITY],  vnorm_left,  left[FIELD_PRESSURE],
                    right[FIELD_DENSITY], vnorm_right, right[FIELD_PRESSURE], geff_i, geff_j);

                // THE pair signal speed for this step: the contact-wave form when enabled, the
                // Monaghan estimate otherwise. Computed ONCE and used for both the per-particle
                // accumulator and the wakeup test below -- the two must be the SAME quantity. When
                // they were not (Monaghan in the wakeup, contact-wave in the stored value) the
                // ratio in the wakeup test ran ~40 against a threshold of 4.1, so every inactive
                // neighbour was demoted three bins every step and the box ran away to bin 21 with
                // every dt criterion still reading large.
                double vsig_pair;
                if (contact_vsig) {
                    const double fv_i = dot(Vec3d{work.predicted[FIELD_VX][i],
                                                  work.predicted[FIELD_VY][i],
                                                  work.predicted[FIELD_VZ][i]}, normal);
                    const double fv_j = dot(Vec3d{work.predicted[FIELD_VX][j],
                                                  work.predicted[FIELD_VY][j],
                                                  work.predicted[FIELD_VZ][j]}, normal);
                    vsig_pair = 2.0*contact_speed + std::max(0.0, fv_j - fv_i);
                    vsig_i = std::max(vsig_i, vsig_pair);
                } else {
                    const Vec3d rel = Vec3d{sim.vx[i], sim.vy[i], sim.vz[i]}
                                    - Vec3d{sim.vx[j], sim.vy[j], sim.vz[j]};
                    vsig_pair = sound_speed(sim, i) + sound_speed(sim, j)
                              + std::max(0.0, dot(rel, offset) / separation);
                }

                // Lagrangian flux: zero mass flux; P* acts across the face, which moves at
                // v_frame + S* nhat in the lab. Momentum goes from i to j along +nhat.
                //
                // ENTROPIC-EOS FACE CORRECTION (GIZMO hydro_core_meshless.h, MFM + ideal gas,
                // default-on). When the contact wave is slow against the local sound speed the
                // pair is not a shock, and the Riemann energy flux P* A S* is pure NUMERICAL
                // dissipation; GIZMO replaces the energy exchange with the adiabatic
                // (grad-h-corrected) PdV form in that regime, which is what keeps smooth
                // compressive flow -- a collapse infall -- from generating spurious entropy.
                // The one-sided dtoi/dtoj checks refuse the swap when it would push heat the
                // wrong way between unequal-entropy sides. Constants per GIZMO: eps_big 0.6 /
                // eps_small 1e-2 with gravity, 0.5 / 1e-3 without.
                //
                // These are RATES -- dP/dt and dE/dt -- deliberately NOT multiplied by any dt here.
                // Each particle integrates its own accumulated rate over its OWN timestep at kick
                // time, which is what removes the need to know how long a given PAIR has been
                // integrated. Multiplying by min(dt_i,dt_j) in this loop instead only stays correct
                // while both bins are constant over the step, and the wakeup limiter exists
                // precisely to change them mid-step; the accounting then silently applies flux over
                // the wrong interval.
                const double face_speed_lab = dot(face_vel, normal) + contact_speed;
                const Vec3d momentum_flux = normal * (contact_pressure * face_area);
                double energy_flux = contact_pressure * face_speed_lab * face_area;
                {
                    const double eps_big   = sim.gravity_on ? 0.6  : 0.5;
                    const double eps_small = sim.gravity_on ? 1e-2 : 1e-3;
                    const double cs_i = sound_speed(sim, i);
                    const double cs_j = sound_speed(sim, j);
                    const double sm_over_c = std::abs(contact_speed) / std::min(cs_i, cs_j);
                    // Ideal gas only: under a density-driven EOS the internal energy is reset from
                    // P(rho) every evaluation, so swapping in an adiabatic energy flux changes
                    // nothing that survives the next eos_apply.
                    if (sim.eos_is_ideal() && sm_over_c < eps_big) {
                        // vdotr from the same predicted velocities the reconstruction used;
                        // dv.dp is orientation-free
                        const Vec3d dv{work.predicted[FIELD_VX][i] - work.predicted[FIELD_VX][j],
                                       work.predicted[FIELD_VY][i] - work.predicted[FIELD_VY][j],
                                       work.predicted[FIELD_VZ][i] - work.predicted[FIELD_VZ][j]};
                        const double vdotr_phys = -dot(dv, offset) / separation;
                        const double pdv_fac = contact_pressure * vdotr_phys;
                        const double pdv_i = kernel_dwdr(separation, sim.h[i], sim.dim)
                                             * sim.ninv[i]*sim.ninv[i] * sim.omega[i] * pdv_fac;
                        const double pdv_j = kernel_dwdr(separation, sim.h[j], sim.dim)
                                             * sim.ninv[j]*sim.ninv[j] * sim.omega[j] * pdv_fac;
                        const double adiabatic_flux =
                            contact_pressure * face_area * dot(face_vel, normal)
                            - 0.5 * (pdv_i - pdv_j);
                        bool use_entropic = true;
                        if (sm_over_c > eps_small) {
                            // one-sided sanity: never let the swap move heat against the entropy
                            // gradient (mapped from GIZMO's dtoi/dtoj tests into this sign
                            // convention, where energy_flux flows i -> j along +nhat)
                            const double pa = contact_pressure * face_area;
                            if (sim.press[i]/sim.rho[i] != sim.press[j]/sim.rho[j]) {
                                if (sim.press[i]/sim.rho[i] > sim.press[j]/sim.rho[j]) {
                                    const double vj_n = work.predicted[FIELD_VX][j]*normal[0]
                                                      + work.predicted[FIELD_VY][j]*normal[1]
                                                      + work.predicted[FIELD_VZ][j]*normal[2];
                                    const double dtoj = energy_flux - pa * vj_n;
                                    if (dtoj > 0) use_entropic = false;
                                    else if (dtoj < 0 && dtoj > adiabatic_flux - pa * vj_n)
                                        use_entropic = false;
                                } else {
                                    const double vi_n = work.predicted[FIELD_VX][i]*normal[0]
                                                      + work.predicted[FIELD_VY][i]*normal[1]
                                                      + work.predicted[FIELD_VZ][i]*normal[2];
                                    const double dtoi = -energy_flux + pa * vi_n;
                                    if (dtoi > 0) use_entropic = false;
                                    else if (dtoi < 0 && dtoi > -adiabatic_flux + pa * vi_n)
                                        use_entropic = false;
                                }
                            }
                        }
                        if (use_entropic) energy_flux = adiabatic_flux;
                    }
                }
                #pragma omp atomic
                dmom_x[i] -= momentum_flux[0];
                #pragma omp atomic
                dmom_y[i] -= momentum_flux[1];
                #pragma omp atomic
                dmom_z[i] -= momentum_flux[2];
                #pragma omp atomic
                denergy[i] -= energy_flux;
                // j receives its half ONLY if it is active. An inactive particle must not be
                // modified at all: it is midway through its own step, and a rate handed to it now
                // would be integrated over an interval that does not correspond to when the flux
                // was computed. It picks this pair up itself, from its own side, at its own sync.
                // The cost is that conservation is exact only between binmates and approximate
                // across a bin boundary -- the trade GIZMO makes, and the reason it needs no
                // per-pair time accounting anywhere.
                if (sim.is_active(j)) {
                    #pragma omp atomic
                    dmom_x[j] += momentum_flux[0];
                    #pragma omp atomic
                    dmom_y[j] += momentum_flux[1];
                    #pragma omp atomic
                    dmom_z[j] += momentum_flux[2];
                    #pragma omp atomic
                    denergy[j] += energy_flux;
                } else if (sim.individual_timesteps) {
                    // Saitoh-Makino wakeup (GIZMO hydro_evaluate.h): demote an INACTIVE neighbour
                    // when THIS pair's signal speed outruns the one j last recorded for itself. j
                    // sized its step from its own vsig, so a pair vsig far above it means a
                    // disturbance is arriving faster than that step can resolve. A bin-GAP test
                    // cannot see this -- pairs straddle many bins in quiescent flow, and share a
                    // bin across a strong shock -- which is why the criterion is on velocities.
                    //
                    // Recorded HERE rather than in a pass of its own: the neighbours are already in
                    // hand, so a separate sweep would repeat this whole tree walk for nothing.
                    // Deferred rather than applied, because writing sim.bin now would race the
                    // threads still reading it.
                    // vsig_pair, computed once above, is whichever definition this run stores in
                    // signal_speed -- comparing unlike quantities here is what produced the bin-21
                    // runaway.
                    const double vsig = vsig_pair;
                    // GIZMO stores the request as waker_bin+1 (0 meaning "none") and decodes it to
                    // bin = waker_bin - offset (core/timestep.cc:1348-1351), so the woken cell
                    // lands on a step WAKEUP times SHORTER than the waker's -- not merely closer
                    // to it. Bins count oppositely here, hence bin[i] + offset. The application
                    // side only ever raises a bin, which is GIZMO's "don't increase the timestep".
                    if (vsig > sim.wakeup_fac * work.signal_speed[j])
                        local_wakes.emplace_back(j, sim.bin[i] + wake_offset);
                }
            }
            if (contact_vsig) vsig_new[i] = vsig_i;
        }
        if (!local_wakes.empty()) {
            #pragma omp critical
            sim.wake_requests.insert(sim.wake_requests.end(),
                                     local_wakes.begin(), local_wakes.end());
        }
    }
    // Merge the staged signal speeds now that every thread is done reading the old ones. Every
    // active cell is written, since the seed alone guarantees a positive value; inactive cells keep
    // theirs, as in the reference, where only the active pass resets MaxSignalVel.
    if (contact_vsig) {
        std::vector<double>& sig = const_cast<Work&>(work).signal_speed;
        if (sig.size() == n_part) {
            #pragma omp parallel for schedule(static)
            for (size_t k = 0; k < active.size(); ++k) {
                const uint32_t i = active[k];
                if (vsig_new[i] > 0) sig[i] = vsig_new[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// LAZY DRIFT. GIZMO's core/predict.cc drift_particle(i, time1), guarded on P[i].Ti_current.
//
// Advance one particle's position from the tick it is current at to `target`. EXACT for an
// inactive particle however long the gap: its velocity does not change between its own
// activations, and the density prediction is an exponential in div_vel * elapsed, which composes
// (exp(a)exp(b) = exp(a+b)) -- so one long catch-up equals the many short drifts it replaces.
//
// Idempotent, and re-checks staleness on entry: that early return is what makes a duplicate call
// from a racing thread a no-op, exactly as GIZMO relies on (predict.cc:109).
static inline void drift_particle_to(Sim& sim, size_t i, long long target) {
    const long long from = sim.last_drift[i];
    if (from >= target) return;
    const double dt = sim.time_of_ticks(target - from);
    Vec3d pos_new = sim.P.pos(i) + Vec3d{sim.vx[i], sim.vy[i], sim.vz[i]} * dt;
    if (sim.box > 0) pos_new = fold_into_box(pos_new, sim.box);
    sim.P.x[i] = pos_new[0]; sim.P.y[i] = pos_new[1]; sim.P.z[i] = pos_new[2];

    // Drift-time prediction for INACTIVE particles: between its own updates a particle's density
    // evolves as its neighbourhood converges or expands, rho_dot = -rho div v. Without this a
    // long-binned particle in a steadily converging flow carries a systematically LOW density until
    // it next activates -- in noh the cold supersonic inflow sat 25% under the analytic pre-shock
    // profile with the velocities EXACT, because only the density estimate was stale. The kernel
    // radius follows with the opposite sign (h ~ n^{-1/dim}) and pressure tracks rho at fixed u.
    // Clamped at +-0.3 as GIZMO clamps it; cheap 2nd-order exp for the tiny arguments this sees.
    if (sim.individual_timesteps && !sim.is_active(i)) {
        // PREDICTED VELOCITY -- GIZMO's VelPred (predict.cc:189), which it advances on every drift
        // for every particle. The STORED velocity is a KDK quantity and correctly stays put between
        // an inactive particle's own kicks, but the hydro reads the PREDICTED one, on both sides of
        // every face. Left un-advanced, a coarse-bin neighbour enters its active neighbours'
        // Riemann problems with a velocity stale by up to its whole step -- under gravity that is
        // not noise but a*dt pointing steadily one way, biasing the face frame and hence P*.
        static const bool no_velpred = getenv("SHMEM_NO_VELPRED") != nullptr;  // A/B switch
        if (!no_velpred && i < sim.n_gas && sim.a_grav.size() > i && sim.a_hydro.size() > i) {
            const Vec3d dv = (sim.a_grav[i] + sim.a_hydro[i]) * dt;
            sim.work.predicted[FIELD_VX][i] += dv[0];
            sim.work.predicted[FIELD_VY][i] += dv[1];
            sim.work.predicted[FIELD_VZ][i] += dv[2];
        }
        double divv_fac = sim.work.div_vel[i] * dt;
        if (divv_fac >  0.3) divv_fac =  0.3;
        if (divv_fac < -0.3) divv_fac = -0.3;
        if (divv_fac != 0.0) {
            const double x = -divv_fac;
            const double f = (std::abs(x) < 0.05) ? 1.0 + x*(1.0 + 0.5*x) : std::exp(x);
            sim.rho[i]  *= f;
            sim.ninv[i] /= f;
            eos_apply(sim, i);              // P(rho) directly, not a scaling of the old pressure
            sim.h[i] *= (std::abs(divv_fac) < 0.15)
                        ? 1.0 + divv_fac/sim.dim + 0.5*(divv_fac/sim.dim)*(divv_fac/sim.dim)
                        : std::exp(divv_fac / sim.dim);
            sim.work.predicted[FIELD_DENSITY][i]  = sim.rho[i];
            sim.work.predicted[FIELD_PRESSURE][i] = sim.press[i];
        }
    }
    // Published LAST, so a thread that observes this particle as current has necessarily also
    // observed the writes above (the same ordering GIZMO relies on, predict.cc / forcetree.cc).
    sim.last_drift[i] = target;
}

// The in-walk callback. One global lock, entered ONLY for a genuinely stale particle -- GIZMO's
// `#pragma omp critical(_partdriftngb_)` in system/ngb_codeblock_after_condition_threaded.h. On a
// step where everything is already current it is never taken at all, which is why this scales with
// actual staleness rather than with the size of the box.
static void drift_catch_up(void* ctx, uint32_t j) {
    Sim& sim = *static_cast<Sim*>(ctx);
    #pragma omp critical(shmem_lazy_drift)
    { drift_particle_to(sim, j, sim.clock_ticks); }
}

// The hook handed to the neighbour search and the gravity walk. Null (no hook, no per-neighbour
// check at all) when nothing can be stale.
static LazyDrift lazy_drift_hook(Sim& sim) {
    LazyDrift lazy;
    // Size it HERE, before the pointer is taken: the walks hold `last` for the whole step, so a
    // reallocation while they are running would leave them reading freed memory.
    if (sim.last_drift.size() != sim.size()) sim.last_drift.resize(sim.size(), sim.clock_ticks);
    lazy.last     = sim.last_drift.data();
    lazy.target   = sim.clock_ticks;
    lazy.catch_up = &drift_catch_up;
    lazy.ctx      = &sim;
    return lazy;
}

// Bring EVERY particle current. Used where positions are read in bulk: before a tree build (the
// build reads live coordinates and derives the node bounds from them) and before writing output.
static void drift_all_to(Sim& sim, long long target) {
    if (sim.last_drift.size() != sim.size()) sim.last_drift.assign(sim.size(), target);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < sim.size(); ++i) drift_particle_to(sim, i, target);
}

void sync_all_positions(Sim& sim) {
    if (sim.last_drift.empty()) return;          // nothing has run yet; positions are the ICs
    drift_all_to(sim, sim.clock_ticks);
}

// Rebuild the tree at the current positions, with the per-node velocity bound initialised from the
// current velocities. Everything must be current first.
static void rebuild_tree(Sim& sim) {
    sync_all_positions(sim);
    const double* vel[3] = {sim.vx.data(), sim.vy.data(), sim.vz.data()};
    // Per-node centre-of-mass velocities only when a Hermite jerk will ask for them.
    sim.tree = build(sim.P, nullptr, vel, sim.hermite_mask != 0,
                     sim.randomize_gravtree ? sim.tree_builds : -1);
    sim.tree.t_since_build = 0.0;
    // One O(N) pass per REBUILD (not per step), dwarfed by the O(N log N) build it follows. Mean
    // over gas rather than a median: the point is a scale for "has anything moved appreciably",
    // and sorting 3.5e6 values to refine that would cost more than it is worth.
    {
        const size_t ng = std::min(sim.n_gas, sim.h.size());
        double sum_h = 0.0; size_t cnt = 0;
        #pragma omp parallel for schedule(static) reduction(+:sum_h,cnt)
        for (size_t i = 0; i < ng; ++i) if (sim.h[i] > 0) { sum_h += sim.h[i]; ++cnt; }
        if (cnt > 0) sim.tree_typical_h = sum_h / (double)cnt;
        else if (!sim.P.soft.empty()) sim.tree_typical_h = sim.P.soft[0];
    }
    // vcom was just built from these velocities and the node dp accumulators are zero, so this
    // is the baseline every later kick is measured against.
    if (sim.hermite_mask != 0) {
        const size_t n = sim.size();
        sim.vel_at_last_kick.resize(n);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i)
            sim.vel_at_last_kick[i] = Vec3d{sim.vx[i], sim.vy[i], sim.vz[i]};
    }
    sim.tree_valid = true;
    ++sim.tree_builds;
}

// Negative definite? Sylvester's criterion on the leading principal minors -- for a symmetric 3x3
// this is exact and needs no eigensolver (GIZMO calls gsl_eigen_symm here only because it already
// links GSL).
static inline bool sym3_negative_definite(const SymTensor3d& T) {
    const double m1 = T[0][0];
    const double m2 = T[0][0]*T[1][1] - T[0][1]*T[0][1];
    const double m3 = T[0][0]*(T[1][1]*T[2][2] - T[1][2]*T[1][2])
                    - T[0][1]*(T[0][1]*T[2][2] - T[1][2]*T[0][2])
                    + T[0][2]*(T[0][1]*T[1][2] - T[1][1]*T[0][2]);
    return (m1 < 0) && (m2 > 0) && (m3 < 0);
}

// ---------------------------------------------------------------------------------------------
// SINK FORMATION. A literal port of GIZMO's SINGLE_STAR_SINK_FORMATION criteria
// (galaxy_sf/sfr_eff.cc), which are all VETOES on a rate that is then multiplied by 1e20 -- so a
// cell that survives every test converts deterministically, on the spot.
//
// Everything the criteria need is already computed by the step: the velocity gradient (divergence
// and Frobenius norm), the tidal tensor from the gravity walk, the EOS sound speed, and the
// neighbour list for the local-maximum test.

// Swap two particles across EVERY per-particle array. Conversion has to move a cell out of the
// gas prefix that Sim::n_gas marks, and the prefix must stay contiguous because the hydro passes
// take it as a range.
static void swap_particles(Sim& sim, size_t a, size_t b) {
    if (a == b) return;
    auto sw = [&](auto& v) { if (v.size() > std::max(a,b)) std::swap(v[a], v[b]); };
    sw(sim.P.x); sw(sim.P.y); sw(sim.P.z); sw(sim.P.m); sw(sim.P.soft); sw(sim.P.zeta);
    sw(sim.P.type);
    sw(sim.vx); sw(sim.vy); sw(sim.vz); sw(sim.u);
    sw(sim.h); sw(sim.ninv); sw(sim.rho); sw(sim.press); sw(sim.omega); sw(sim.csnd);
    sw(sim.phi); sw(sim.a_grav); sw(sim.a_hydro); sw(sim.du_dt); sw(sim.tidal);
    sw(sim.pending_half_kick);
    sw(sim.sink_pin_x); sw(sim.sink_pin_y); sw(sim.sink_pin_z); sw(sim.sink_pinned);
    sw(sim.bin); sw(sim.dt_of); sw(sim.alpha_vir_smoothed); sw(sim.last_drift);
    sw(sim.sink_radius); sw(sim.sink_tform); sw(sim.sink_m0); sw(sim.sink_reservoir); sw(sim.id);
    sw(sim.min_sink_tapp); sw(sim.min_sink_tff); sw(sim.sink_dt_gas_cap); sw(sim.vel_at_last_kick);
    sw(sim.herm_valid); sw(sim.herm_tick); sw(sim.herm_pos); sw(sim.herm_vel);
    sw(sim.herm_acc); sw(sim.herm_jerk);
    sw(sim.dmom_x); sw(sim.dmom_y); sw(sim.dmom_z); sw(sim.denergy);
    sw(sim.work.moments_inv); sw(sim.work.signal_speed); sw(sim.work.div_vel);
    sw(sim.work.condition_number); sw(sim.work.face_closure);
    for (auto& g : sim.work.gradient)  sw(g);
    for (auto& p : sim.work.predicted) sw(p);
}

// Shrink every per-particle array by one. The doomed particle must already sit in the LAST slot.
static void pop_particle(Sim& sim) {
    auto pop = [&](auto& v) { if (!v.empty()) v.pop_back(); };
    pop(sim.P.x); pop(sim.P.y); pop(sim.P.z); pop(sim.P.m); pop(sim.P.soft); pop(sim.P.zeta);
    pop(sim.P.type);
    pop(sim.vx); pop(sim.vy); pop(sim.vz); pop(sim.u);
    pop(sim.h); pop(sim.ninv); pop(sim.rho); pop(sim.press); pop(sim.omega); pop(sim.csnd);
    pop(sim.phi); pop(sim.a_grav); pop(sim.a_hydro); pop(sim.du_dt); pop(sim.tidal);
    pop(sim.pending_half_kick);
    pop(sim.sink_pin_x); pop(sim.sink_pin_y); pop(sim.sink_pin_z); pop(sim.sink_pinned);
    pop(sim.bin); pop(sim.dt_of); pop(sim.alpha_vir_smoothed); pop(sim.last_drift);
    pop(sim.sink_radius); pop(sim.sink_tform); pop(sim.sink_m0); pop(sim.sink_reservoir); pop(sim.id);
    pop(sim.min_sink_tapp); pop(sim.min_sink_tff); pop(sim.sink_dt_gas_cap); pop(sim.vel_at_last_kick);
    pop(sim.herm_valid); pop(sim.herm_tick); pop(sim.herm_pos); pop(sim.herm_vel);
    pop(sim.herm_acc); pop(sim.herm_jerk);
    pop(sim.dmom_x); pop(sim.dmom_y); pop(sim.dmom_z); pop(sim.denergy);
    pop(sim.work.moments_inv); pop(sim.work.signal_speed); pop(sim.work.div_vel);
    pop(sim.work.condition_number); pop(sim.work.face_closure);
    for (auto& g : sim.work.gradient)  pop(g);
    for (auto& q : sim.work.predicted) pop(q);
}

// Delete one GAS particle while keeping the gas-first layout contiguous. Two swaps: the doomed
// cell goes to the end of the gas prefix, then to the very end of the array -- which slides the
// LAST SINK down into the slot the gas just vacated, so gas stays [0, n_gas-1) and the sinks stay
// contiguous immediately after it.
static void remove_gas_particle(Sim& sim, size_t j) {
    const size_t last_gas = sim.n_gas - 1, last = sim.size() - 1;
    swap_particles(sim, j, last_gas);
    if (last_gas != last) swap_particles(sim, last_gas, last);
    pop_particle(sim);
    sim.n_gas = last_gas;
}

// ---------------------------------------------------------------------------------------------
// SINK ACCRETION -- gravitational capture, Bate-style fixed sink radius.
// SINGLE_STAR_ACCRETION=12 (declarations/precompiler_logic.h:375) selects SINK_GRAVCAPTURE_GAS +
// SINK_GRAVCAPTURE_FIXEDSINKRADIUS, so the criteria are sinks/sink_feed.cc:254-300 with
// sinks/sink.cc:107 sink_check_boundedness:
//   * inside the fixed sink radius                                    (sink.cc:144)
//   * bound, counting the gas internal energy: (vrel^2+cs^2)/vesc^2 < 1
//   * Bate (1995) angular momentum: L^2 < G (M+m) r_sink              (sink_feed.cc:267-269)
//   * resolution: the cell must be smaller than the sink              (sink.cc:127)
// vesc carries the enclosed gas as an isothermal-sphere interior (sink.cc:97), which is what makes
// this a Shu-type capture rather than a two-body one.
//
// Serial: accretion events are few per step, they mutate the particle arrays, and running them in
// index order makes the outcome independent of thread scheduling.

// Synchronised total momentum, for the SHMEM_MOMAUDIT probes. See the lambda in mfm_step.
static Vec3d audit_total_p(const Sim& sim) {
    const size_t n = sim.size();
    const bool have = sim.pending_half_kick.size() == n && sim.a_grav.size() == n;
    const bool have_hyd = have && sim.a_hydro.size() == n;
    double px = 0, py = 0, pz = 0;
    for (size_t i = 0; i < n; ++i) {
        const double owed = have ? sim.pending_half_kick[i] : 0.0;
        Vec3d a = have ? sim.a_grav[i] : Vec3d{0, 0, 0};
        if (have_hyd && i < sim.n_gas) a += sim.a_hydro[i];   // the kick owes the hydro rate too
        px += sim.P.m[i] * (sim.vx[i] - a[0]*owed);
        py += sim.P.m[i] * (sim.vy[i] - a[1]*owed);
        pz += sim.P.m[i] * (sim.vz[i] - a[2]*owed);
    }
    return Vec3d{px, py, pz};
}
// SHMEM_ENERGY_LOG: total energy at a SYNCHRONISED state.
//
// A snapshot cannot be used for this. write_snapshot_at back-drifts POSITIONS to the requested
// output time, but the stored velocity is half-kicked into the NEXT step (pending_half_kick), so a
// snapshot pairs x(t) with v(t + dt/2) and its energy carries an O(dt * a * v) error. On the
// e=0.9 binary that artifact is ~5e-4 relative -- larger than the integration error it was being
// used to measure, and it scales with dt, so it masquerades as an error floor when two schemes are
// compared at different step sizes.
//
// Undoing the owed half kick exactly as audit_total_p does puts KE and PE at the same instant.
void energy_log_step(Sim& sim, double t) {
    static const bool on = getenv("SHMEM_ENERGY_LOG") != nullptr;
    if (!on) return;
    const size_t n = sim.size();
    const bool have = sim.pending_half_kick.size() == n && sim.a_grav.size() == n;
    const bool have_hyd = have && sim.a_hydro.size() == n;
    compute_potential(sim);
    double ke = 0, pe = 0;
    for (size_t i = 0; i < n; ++i) {
        const double owed = have ? sim.pending_half_kick[i] : 0.0;
        Vec3d a = have ? sim.a_grav[i] : Vec3d{0, 0, 0};
        if (have_hyd && i < sim.n_gas) a += sim.a_hydro[i];
        const double ux = sim.vx[i] - a[0] * owed;
        const double uy = sim.vy[i] - a[1] * owed;
        const double uz = sim.vz[i] - a[2] * owed;
        ke += 0.5 * sim.P.m[i] * (ux*ux + uy*uy + uz*uz);
        pe += 0.5 * sim.P.m[i] * sim.phi[i];   // 0.5 so each pair is counted once
    }
    fprintf(stderr, "[energy] t=%.12g KE=%.17g PE=%.17g E=%.17g\n", t, ke, pe, ke + pe);
}

static void audit_step(const Sim& sim, Vec3d& prev, const char* what) {
    static const bool on = getenv("SHMEM_MOMAUDIT") != nullptr;
    if (!on) return;
    const Vec3d p = audit_total_p(sim);
    const Vec3d d = p - prev;
    if (d.norm() > 0) fprintf(stderr, "[mom] %-14s %14.6e %14.6e %14.6e\n", what, d[0], d[1], d[2]);
    prev = p;
}

// SINK VELOCITY PROBE (SHMEM_SINKV). A sink can be swept up to the speed of the flow around it
// while the gas centre of mass stays put, so the total-momentum audit above cannot see it at all:
// attribute the SINK's own velocity change to each operation instead. Tracks the heaviest sink;
// serial at every call site, hence the plain static.
static void sinkv_probe(const Sim& sim, const char* what) {
    static const bool on = getenv("SHMEM_SINKV") != nullptr;
    if (!on || sim.n_gas >= sim.size()) return;
    size_t k = sim.n_gas;
    for (size_t i = sim.n_gas; i < sim.size(); ++i) if (sim.P.m[i] > sim.P.m[k]) k = i;
    static Vec3d prev{0, 0, 0};
    const Vec3d v{sim.vx[k], sim.vy[k], sim.vz[k]};
    const Vec3d d = v - prev;
    if (d.norm() > 0) {
        // The sink's bin against the deepest GAS bin: a sink integrating on a longer step than the
        // gas it is embedded in gets its orbit resolved worse than the flow driving it.
        int gas_deep = 0;
        if (!sim.bin.empty())
            for (size_t i = 0; i < sim.n_gas && i < sim.bin.size(); ++i)
                if (sim.bin[i] > gas_deep) gas_deep = sim.bin[i];
        fprintf(stderr, "[sinkv] %.8e %-12s m=%.6e |v|=%.6e |dv|=%.6e bin=%d gasdeep=%d\n",
                sim.time_now(), what, sim.P.m[k], v.norm(), d.norm(),
                sim.bin.empty() ? -1 : sim.bin[k], gas_deep);
    }
    prev = v;
}

// DIRECT-SUM CHECK on the sink's own gravity (SHMEM_SINKACC). The sink's trajectory is the thing
// going wrong and the tree force is the hardest input to bound by inspection, so compare it with an
// O(N) sum over every particle. Exact for the sink: pairs involving a non-gas particle take the
// plain max-softening spline, with no kernel averaging and no zeta, so there is nothing the walk
// does here that this does not.
static void sink_accel_check(Sim& sim) {
    static const bool on = getenv("SHMEM_SINKACC") != nullptr;
    if (!on || sim.n_gas >= sim.size() || sim.a_grav.size() != sim.size()) return;
    static long long calls = 0;
    if ((calls++ % 500) != 0) return;
    size_t k = sim.n_gas;
    for (size_t i = sim.n_gas; i < sim.size(); ++i) if (sim.P.m[i] > sim.P.m[k]) k = i;
    // Only when the sink is ACTIVE. a_grav is refreshed for active targets only -- an inactive
    // particle deliberately keeps its last acceleration for the KDK kick -- so on any other step
    // this would compare a stale walk against a current direct sum and indict the tree for it.
    if (!sim.is_active(k)) return;
    // Then bring every position current. Under lazy drift a particle the walk only ever saw
    // inside a node still carries its old coordinates, so a direct sum over live coordinates
    // would be comparing the walk against a half-stale truth. Diagnostic build only.
    drift_all_to(sim, sim.clock_ticks);
    const Vec3d pos_k = sim.P.pos(k);
    const double eps_k = sim.P.soft.empty() ? 0.0 : sim.P.soft[k];
    double ax = 0, ay = 0, az = 0;
    #pragma omp parallel for schedule(static) reduction(+:ax,ay,az)
    for (size_t j = 0; j < sim.size(); ++j) {
        if (j == k) continue;
        const Vec3d d = min_image(sim.P.pos(j) - pos_k, sim.box);
        const double r = d.norm();
        if (!(r > 0)) continue;
        const double eps = std::max(std::max(eps_k, sim.P.soft.empty() ? 0.0 : sim.P.soft[j]),
                                    1e-300);
        const double f = sim.P.m[j] * spline_force_over_r(r, eps);
        ax += f*d[0]; ay += f*d[1]; az += f*d[2];
    }
    const Vec3d direct{ax*sim.G, ay*sim.G, az*sim.G};
    const Vec3d walk = sim.a_grav[k];
    fprintf(stderr, "[sinkacc] %.8e |a_tree|=%.6e |a_direct|=%.6e |da|=%.6e rel=%.3e\n",
            sim.time_now(), walk.norm(), direct.norm(), (walk-direct).norm(),
            (walk-direct).norm() / std::max(direct.norm(), 1e-300));
}

// Drain rate of a sink's unresolved disk (sinks/sink.cc:395-399, 465-474). Under
// SINK_GRAVCAPTURE_FIXEDSINKRADIUS the timescale collapses to a constant fixed at formation --
// see Sim::sink_reservoir -- floored at three steps so no single step can empty the disk.
static double sink_mdot(const Sim& sim, size_t i, double dt) {
    if (sim.sink_reservoir.size() != sim.size() || sim.sink_reservoir[i] <= 0) return 0.0;
    if (sim.sink_m0.size() != sim.size() || sim.sink_m0[i] <= 0) return 0.0;
    const double cs_min = 0.2 / std::max(sim.vel_to_kms, 1e-300);   // 0.2 km/s, in code velocity
    double t_acc = sim.G * sim.sink_m0[i] / (cs_min * cs_min * cs_min);
    if (dt > 0) t_acc = std::max(t_acc, 3.0 * dt);
    return t_acc > 0 ? sim.sink_reservoir[i] / t_acc : 0.0;
}

static void sink_accretion_scan(Sim& sim) {
    if (!sim.sink_formation || sim.n_gas >= sim.size()) return;
    static const bool diag = getenv("SHMEM_SINK_DIAG") != nullptr;
    sinkv_probe(sim, "pre-accrete");

    // Drain each ACTIVE sink's disk into stellar mass, before this step's swallows refill it --
    // the order the reference uses (set_sink_new_mass runs in the sink pass, ahead of the swallow
    // loop). P.m does not move: the reservoir only tracks how much of it is not yet stellar, so
    // this changes no dynamics, only the Mdot that dt_accr reads. The 3-step floor inside
    // sink_mdot bounds the drain at a third of the disk, so it cannot go negative.
    if (sim.sink_reservoir.size() == sim.size() && sim.dt_of.size() == sim.size())
        for (size_t s = sim.n_gas; s < sim.size(); ++s) {
            if (!sim.P.type.empty() && sim.P.type[s] != 5) continue;
            if (!sim.is_active(s) || sim.dt_of[s] <= 0) continue;
            const double drained = sink_mdot(sim, s, sim.dt_of[s]) * sim.dt_of[s];
            sim.sink_reservoir[s] = std::max(0.0, sim.sink_reservoir[s] - drained);
        }

    // TWO PHASES, and the split is not stylistic. Deleting a gas cell slides the last sink into
    // the slot it vacated, so any sink index held across a deletion is stale -- the first version
    // of this kept eating with a stale index and corrupted the sink's mass. So: scan and decide
    // first, touching nothing, then apply every deletion afterwards.
    Vec3d p_acc = audit_total_p(sim);          // baseline BEFORE any merging
    std::vector<uint32_t> doomed;              // gas indices to remove, ascending
    std::vector<char> claimed(sim.n_gas, 0);   // a cell may only be swallowed once
    std::vector<uint32_t> ngb;
    // Cells that reach the physical tests already inside the sink radius, and which test turns
    // them away. Gas that keeps failing here is gas piling up on the sink -- the pileup that
    // then goes self-gravitating and spawns a spurious second sink.
    long long acc_inside = 0, acc_rej_res = 0, acc_rej_unbound = 0, acc_rej_angmom = 0;

    for (size_t sph = sim.n_gas; sph < sim.size(); ++sph) {
        // ACTIVE SINKS ONLY -- the reference accretes from calculate_non_standard_physics, which
        // runs on the active set, so a sink's mass changes only at its own step boundaries.
        // Accreting every sync instead (as this did) splits the operators differently from the
        // kick: the sink's owed pending_half_kick was computed against its mass BEFORE the
        // swallows and is then applied to the grown mass, so the momentum it receives no longer
        // matches the reaction the gas was already given. The 4.1x wake-up cap is what bounds
        // how long gas can wait to be eaten.
        if (sim.individual_timesteps && !sim.is_active(sph)) continue;
        drift_particle_to(sim, sph, sim.clock_ticks);
        const double r_sink = sim.sink_radius.empty() ? 0.0 : sim.sink_radius[sph];
        if (!(r_sink > 0)) continue;
        const Vec3d pos_s = sim.P.pos(sph);
        ngb.clear();
        ngb_search(sim.tree, sim.P, pos_s, r_sink, ngb, sim.box, sim.lazy());
        for (uint32_t j : ngb) {
            if (j >= sim.n_gas || claimed[j]) continue;          // gas only, once only
            const Vec3d dx = min_image(sim.P.pos(j) - pos_s, sim.box);
            const double r = dx.norm();
            if (!(r > 0) || r > r_sink) continue;                // inside the fixed sink radius
            if (diag) ++acc_inside;                              // reached the physical tests
            // the cell must be smaller than the sink it falls into (sink.cc:127). The reference's
            // Get_Particle_Size() is 1.61199*KernelRadius/NumNgb, where NumNgb has already been
            // replaced by its cube root at the end of the density loop (density.cc:1037) to save
            // repeated cbrt calls -- so it is exactly V^(1/3), which is what this computes.
            if (std::pow(sim.ninv[j], 1.0/sim.dim) > r_sink * 1.396263) {
                if (diag) ++acc_rej_res;
                continue;
            }

            const Vec3d dv{sim.vx[j] - sim.vx[sph], sim.vy[j] - sim.vy[sph],
                           sim.vz[j] - sim.vz[sph]};
            const double vrel_sq = dv.norm_sq();
            // internal energy enters as an effective speed; gamma ~ 1 is the isothermal hack,
            // where GIZMO uses 3P/rho rather than 2u (sink.cc:116)
            const double cs_sq = (std::abs(sim.gamma - 1.0) < 0.1)
                               ? 3.0 * sim.press[j] / sim.rho[j] : 2.0 * sim.u[j];
            // Escape speed, sink_vesc (sink.cc:85-103). Two pieces that are easy to get wrong:
            //   * m_eff carries an isothermal-sphere gas interior, 4 pi r^3 rho -- the Shu-type
            //     self-gravity of the enclosed gas. It is gated on SINGLE_STAR_SINK_DYNAMICS
            //     (sink.cc:92), NOT on COOLING, so it is always on for a STARFORGE run.
            //   * the potential is SPLINE-SOFTENED on the sink's softening (sink.cc:103), so
            //     inside that radius vesc is below Keplerian. Using a bare 1/r there overstates
            //     vesc and swallows cells the reference does not.
            const double m_eff = sim.P.m[sph] + sim.P.m[j]
                               + 4.0*M_PI * r*r*r * sim.rho[j];
            const double soft_s = std::max(sim.P.soft[sph], 1e-300);
            double vesc_sq = 2.0 * sim.G * m_eff * std::abs(spline_potential(r, soft_s));
            // The extra opacity-limited boost is the separately COOLING-gated one at
            // sink.cc:128: re-estimate vesc from the enclosed gas alone when the cell sits at
            // the bottom of a quasi-hydrostatic Larson core.
            if (sim.opacity_limit_physics && sim.nh_per_code_density > 0) {
                const double nH = sim.rho[j] * sim.nh_per_code_density;
                if (nH > 1e13 && cs_sq > 0.01 * vrel_sq) {
                    const double m_gas = 4.0*M_PI * r*r*r * sim.rho[j];
                    vesc_sq = std::max(2.0 * sim.G * m_gas / r, vesc_sq);
                }
            }
            if (!(vesc_sq > 0)) continue;
            if ((vrel_sq + cs_sq) / vesc_sq >= 1.0) {            // unbound
                if (diag) ++acc_rej_unbound;
                continue;
            }
            // Bate (1995): angular momentum small enough to actually reach the sink
            const double rv = dot(dx, dv);
            const double spec_mom_sq = r*r*vrel_sq - rv*rv;
            if (spec_mom_sq >= sim.G * (sim.P.m[sph] + sim.P.m[j]) * r_sink) {
                if (diag) ++acc_rej_angmom;
                continue;
            }

            // SWALLOW. Mass, momentum AND centre of mass. Safe to apply now -- this mutates only
            // the SINK, and no index moves until the removal phase.
            //
            // MERGE THE STORED VELOCITIES DIRECTLY, as the reference does: sink.cc:746 is
            //     Vel = (Vel*m_new + sum_j m_j (Vel_j - Vel)) / m_new
            // which is the plain mass-weighted average of P[].Vel on both sides, with no kick-
            // phase correction anywhere. There is nothing to correct: a stored KDK velocity is a
            // well-defined quantity at any point in the step, and this engine's stored velocity is
            // its exact analogue.
            //
            // An earlier version undid each side's outstanding half-kick, averaged, and re-applied
            // the sink's. Expanding that leaves
            //     v = [mass-weighted average] + (m_j/m_new) (a_s owed_s - a_j owed_j)
            // -- a residual that does not vanish and is systematically signed in a collapse, where
            // a points inward and the two particles sit on different bins. It was compensating for
            // a debt that costs nothing: the cell's unpaid half-kick dies with the cell in the
            // reference too, and the sink's is a velocity increment, so the sink's mass growth
            // does not change what it is worth. SHMEM_SYNC_MERGE restores it for A/B.
            //
            // Under SHMEM_SWALLOW_AT_SYNC this needs no correction of any kind: the swallow runs
            // between the two half-kicks, so an active particle's stored velocity already IS the
            // sync-point value, which is exactly the state the reference merges from.
            static const bool sync_merge = getenv("SHMEM_SYNC_MERGE") != nullptr;
            const bool have_debt = sync_merge && sim.pending_half_kick.size() == sim.size() &&
                                   sim.a_grav.size() == sim.size();
            const double owed_s = have_debt ? sim.pending_half_kick[sph] : 0.0;
            const double owed_j = have_debt ? sim.pending_half_kick[j]   : 0.0;
            const Vec3d kick_s = have_debt ? sim.a_grav[sph] * owed_s : Vec3d{0,0,0};
            const Vec3d vs_sync = Vec3d{sim.vx[sph], sim.vy[sph], sim.vz[sph]} - kick_s;
            const Vec3d vj_sync = Vec3d{sim.vx[j], sim.vy[j], sim.vz[j]}
                                - (have_debt ? sim.a_grav[j] * owed_j : Vec3d{0,0,0});
            const double m_new = sim.P.m[sph] + sim.P.m[j];
            const Vec3d v_sync = (vs_sync * sim.P.m[sph] + vj_sync * sim.P.m[j]) / m_new;
            const Vec3d v_store = v_sync + kick_s;
            // CENTRE OF MASS. sink.cc:763 moves the sink onto the mass-weighted centre of itself
            // and what it just ate, exactly parallel to the velocity update above:
            //     Pos = (Pos*m_new + sum_j m_j (x_j - Pos)) / m_new
            // (SINK_FOLLOW_ACCRETED_COM, on whenever SINK_SWALLOWGAS is; the alternative branch is
            // SINK_REPOSITION_ON_POTMIN, which is FIRE_BHS-only and off here.) Without it a sink
            // keeps whatever position gravity gave the single progenitor cell and never recentres
            // on the mass it has absorbed -- so it drifts off the centroid of its own accreted
            // material, which in a collapse is the density peak it is supposed to be sitting in.
            // Written as an offset from the sink so it is correct across a periodic boundary.
            const Vec3d com_shift = dx * (sim.P.m[j] / m_new);
            sim.vx[sph] = v_store[0]; sim.vy[sph] = v_store[1]; sim.vz[sph] = v_store[2];
            {
                Vec3d pos_new = sim.P.pos(sph) + com_shift;
                if (sim.box > 0) pos_new = fold_into_box(pos_new, sim.box);
                sim.P.x[sph] = pos_new[0]; sim.P.y[sph] = pos_new[1]; sim.P.z[sph] = pos_new[2];
            }
            sim.P.m[sph] = m_new;
            // Swallowed gas enters the unresolved disk, not the star. P.m already carries it --
            // this only records how much of that total has yet to drain, which is what sets Mdot.
            if (sim.sink_reservoir.size() == sim.size()) sim.sink_reservoir[sph] += sim.P.m[j];
            // The mass/velocity jump invalidates any Hermite snapshot: the sink falls back to
            // KDK for one step and re-enters at its next sync (GIZMO's AccretedThisTimestep).
            if (sph < sim.herm_valid.size()) sim.herm_valid[sph] = 0;
            claimed[j] = 1; doomed.push_back(j); ++sim.cells_accreted;
            if (diag)
                fprintf(stderr, "[sink-eat] sink=%zu ate cell=%u r/r_sink=%.3g vrel/vesc=%.3g "
                        "M=%.6g\n", sph, j, r/r_sink,
                        std::sqrt((vrel_sq+cs_sq)/vesc_sq), m_new);
        }
    }
    if (diag && acc_inside > 0) {
        static long long acalls = 0;
        if ((acalls++ % 200) == 0 || (acc_inside > (long long)doomed.size() * 4 + 8))
            fprintf(stderr, "[sink-acc] inside_r_sink=%lld eaten=%zu | rejected res=%lld "
                    "unbound=%lld angmom=%lld\n", acc_inside, doomed.size(), acc_rej_res,
                    acc_rej_unbound, acc_rej_angmom);
    }
    // Hand the marked cells to sink_accretion_remove. Under the split ordering that runs at the
    // end of the step; otherwise immediately below.
    sim.acc_audit_baseline = p_acc;
    sim.doomed_cells = doomed;
    if (!doomed.empty()) {
        sim.doomed_mask.assign(sim.n_gas, 0);
        for (uint32_t j : doomed) if (j < sim.doomed_mask.size()) sim.doomed_mask[j] = 1;
    } else {
        sim.doomed_mask.clear();
    }
    audit_step(sim, p_acc, "acc:merge");
    sinkv_probe(sim, "accrete");
}

// Apply the removals decided by the scan above. Separated so the merge can sit at the reference's
// swallow point (between the half-kicks) while the index churn waits until nothing else in the
// step depends on the old indices.
static void sink_accretion_remove(Sim& sim) {
    static const bool diag = getenv("SHMEM_SINK_DIAG") != nullptr;
    std::vector<uint32_t>& doomed = sim.doomed_cells;
    sim.doomed_mask.clear();
    if (doomed.empty()) return;
    Vec3d p_acc = sim.acc_audit_baseline;
    // Accretion is the one operation that moves mass BETWEEN particle types, so it is the one
    // that can silently break the books. Check the total against t=0 every time it fires.
    // DESCENDING: removing index j swaps in the particle at n_gas-1, which is always >= j, so a
    // still-pending (smaller) index is never the one moved into place.
    std::sort(doomed.begin(), doomed.end(), std::greater<uint32_t>());
    for (uint32_t j : doomed) remove_gas_particle(sim, j);
    doomed.clear();
    audit_step(sim, p_acc, "acc:removal");
    // Every index the tree and the neighbour cache hold is now wrong.
    sim.ngb_cache.clear(); sim.tree_valid = false;
    if (sim.mass_initial > 0) {
        double total = 0.0;
        for (size_t i = 0; i < sim.size(); ++i) total += sim.P.m[i];
        const double err = std::abs(total - sim.mass_initial) / sim.mass_initial;
        if (err > 1e-12 || diag)
            fprintf(stderr, "[sink-mass] total=%.15g initial=%.15g rel_err=%.3g  "
                    "gas=%zu sinks=%zu accreted=%lld\n", total, sim.mass_initial, err,
                    sim.n_gas, sim.size() - sim.n_gas, sim.cells_accreted);
    }
}

// SINGLE_STAR_TIMESTEPPING: per-particle minimum approach and freefall times to the SINK
// population, the inputs to the two-body timestep criterion (gravity/forcetree.cc:1685-1699
// leaf branch, :2012-2022 node branch; results stored at :2509-2510). GIZMO folds this into
// the gravity walk with per-node sink summaries because its sink count can be large; here the
// sinks are few (one in shu1977, hundreds in plummer_binaries), so a direct minimum over all
// of them is cheaper than threading state through the walk AND exact where the node branch
// approximates. Only ACTIVE particles are refreshed -- same cadence as GIZMO, which updates
// P.Min_Sink_* when the particle does a gravity walk. Values persist for inactive particles.
static void sink_timestep_pass(Sim& sim, const std::vector<uint32_t>& active) {
    const size_t n_part = sim.size();
    if (sim.n_gas >= n_part) return;             // no non-gas particles at all
    // the sinks: type-5 members of the non-gas suffix (a halo-only sim has none)
    std::vector<uint32_t> sinks;
    for (size_t j = sim.n_gas; j < n_part; ++j)
        if (sim.P.type.empty() || sim.P.type[j] == 5) sinks.push_back((uint32_t)j);
    if (sinks.empty()) return;
    if (sim.min_sink_tapp.size() != n_part) sim.min_sink_tapp.assign(n_part, 1e300);
    if (sim.min_sink_tff.size()  != n_part) sim.min_sink_tff.assign(n_part, 1e300);
    // Distances must be measured at NOW: an inactive sink can be carrying a stale position
    // under lazy drift. Few sinks, so serial catch-up is free.
    if (sim.sparse_drift)
        for (uint32_t j : sinks) drift_particle_to(sim, j, sim.clock_ticks);
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < active.size(); ++k) {
        const uint32_t i = active[k];
        double best_ta2 = 1e300, best_tff4 = 1e300;
        const Vec3d pos_i = sim.P.pos(i);
        const Vec3d vel_i{sim.vx[i], sim.vy[i], sim.vz[i]};
        for (uint32_t j : sinks) {
            if (j == i) continue;
            const Vec3d dx = min_image(sim.P.pos(j) - pos_i, sim.box);
            // softened separation: the larger of the two kernel-extent softenings, converted
            // to its Plummer equivalent (KERNEL_FAC_FROM_FORCESOFT_TO_PLUMMER = 1/2.8), added
            // in quadrature -- forcetree.cc:1686-1688
            const double eps = std::max(sim.P.soft[j], sim.P.soft[i]) / 2.8;
            const double r2s = dx.norm_sq() + eps * eps;
            const Vec3d dv = Vec3d{sim.vx[j], sim.vy[j], sim.vz[j]} - vel_i;
            const double ta2 = r2s / (dv.norm_sq() + 1e-300);
            const double mt = sim.P.m[j] + sim.P.m[i];
            const double tff4 = r2s * r2s * r2s / (mt * mt);
            if (ta2  < best_ta2)  best_ta2  = ta2;
            if (tff4 < best_tff4) best_tff4 = tff4;
        }
        // t_approach = r_soft/|dv|; t_ff = sqrt(r_soft^3 / (G Mtot)) -- forcetree.cc:2509-2510.
        // With no OTHER sink the sentinel must survive intact: sqrt(1e300) is 1e150, which slips
        // under the "< 1e299" guard in desired_dt and feeds the two-body term sentinel arithmetic.
        // It only ever produced a harmlessly huge dt, but the guard was not doing its job.
        const bool have_partner = (best_ta2 < 1e299);
        sim.min_sink_tapp[i] = have_partner ? std::sqrt(best_ta2) : 1e300;
        sim.min_sink_tff[i]  = have_partner ? std::sqrt(std::sqrt(best_tff4) / sim.G) : 1e300;
    }

    // SINK-GAS coupling (core/timestep.cc:1002-1026). A sink parked in dense collapsing gas must
    // not sit bins above the cells it is about to swallow. The reference collects these in the
    // sink's density loop (hydro/density.cc:404: min gas TimeBin, nearest gas distance;
    // sinks/sink_environment.cc:212: kernel-mean relative gas velocity); here an expanding
    // neighbour search around each ACTIVE sink plays that role. Three caps, combined into one
    // per-sink dt ceiling used by desired_dt:
    //   * wakeup:  dt <= 1.01 * 4.1 * dt(shortest-step gas neighbour)
    //   * freefall: dt <= 1.01 * sqrt(2 eta eps^3 / (G M_sink))
    //   * Courant:  dt <= 1.01 * CourantFac * L_sink / v_sig(surrounding gas)
    if (sim.n_gas > 0 && sim.n_gas < n_part) {
        if (sim.sink_dt_gas_cap.size() != n_part) sim.sink_dt_gas_cap.assign(n_part, 1e300);
        static thread_local std::vector<uint32_t> ngb;
        for (uint32_t s : sinks) {
            if (!sim.is_active(s)) continue;
            const Vec3d pos_s = sim.P.pos(s);
            // expanding search: start at the sink's own scales, double until enough gas found
            double radius = std::max({sim.P.soft[s],
                                      sim.sink_radius.empty() ? 0.0 : sim.sink_radius[s],
                                      1e-30});
            size_t n_gas_found = 0;
            for (int tries = 0; tries < 40; ++tries) {
                ngb.clear();
                ngb_search(sim.tree, sim.P, pos_s, radius, ngb, sim.box, sim.lazy());
                n_gas_found = 0;
                for (uint32_t j : ngb) if (j < sim.n_gas) ++n_gas_found;
                if (n_gas_found >= 32) break;
                radius *= 2.0;
            }
            if (n_gas_found == 0) { sim.sink_dt_gas_cap[s] = 1e300; continue; }
            int deepest_bin = 0;
            double dr_nearest = 1e300, m_sum = 0.0, cs2_sum = 0.0, w_sum = 0.0;
            Vec3d mv_rel{0, 0, 0};
            const Vec3d vel_s{sim.vx[s], sim.vy[s], sim.vz[s]};
            for (uint32_t j : ngb) {
                if (j >= sim.n_gas) continue;
                const double r = min_image(sim.P.pos(j) - pos_s, sim.box).norm();
                dr_nearest = std::min(dr_nearest, r);
                deepest_bin = std::max(deepest_bin, sim.bin[j]);
                const double m = sim.P.m[j];
                m_sum += m;
                mv_rel += (Vec3d{sim.vx[j], sim.vy[j], sim.vz[j]} - vel_s) * m;
                const double cs = sound_speed(sim, j);
                cs2_sum += m * cs * cs;
                w_sum += kernel_w(r, radius, sim.dim);   // KERNEL-WEIGHTED, not a raw count
            }
            // 4.1x the shortest-step gas neighbour (core/timestep.cc:1002). SHMEM_SINK_WAKE_FAC
            // overrides it: a sink on a longer bin than the gas around it is kicked by that gas
            // on the gas's cadence but kicks back on its own, so the pair's momentum exchange
            // does not balance across the bin boundary. Driving the factor to 1 tests whether
            // that asymmetry is what injects momentum once a sink exists.
            static const double wake_fac = getenv("SHMEM_SINK_WAKE_FAC")
                                         ? atof(getenv("SHMEM_SINK_WAKE_FAC")) : 4.1;
            const double dt_wake = wake_fac * sim.dt_of_bin(deepest_bin);
            // eps = max(kernel-core softening, nearest gas dr, sink radius, cell size).
            //
            // L_sink is the same estimator the gas uses, V^(1/dim) with V = 1/sum_j W -- the
            // KERNEL-WEIGHTED local volume per particle, which is what the reference's
            // Get_Particle_Size() reduces to. It previously divided the hard-sphere search radius
            // by the cube root of the RAW count inside it. Those agree in uniform density, but
            // around a sink they do not: the kernel-weighted sum is dominated by the dense inner
            // neighbours, so the raw-count form overestimates the spacing exactly where the
            // density is peaked, and dt_ff (~eps^3/2) and dt_cour (~L) come out too permissive.
            const double L_sink = (w_sum > 0) ? std::pow(1.0 / w_sum, 1.0 / sim.dim) : radius;
            double eps = std::max(0.5 * sim.P.soft[s], dr_nearest);
            if (!sim.sink_radius.empty()) eps = std::max(eps, sim.sink_radius[s]);
            eps = std::max(eps, L_sink);
            const double dt_ff = std::sqrt(2.0 * sim.eta_grav * eps * eps * eps
                                           / (sim.G * sim.P.m[s] + 1e-300));
            const double vsig = std::sqrt((mv_rel / m_sum).norm_sq() + cs2_sum / m_sum);
            const double dt_cour = sim.cfl * L_sink / (vsig + 1e-300);
            sim.sink_dt_gas_cap[s] =
                1.01 * std::min({dt_wake, dt_ff, dt_cour});
        }
    }
}

// ---- HERMITE_INTEGRATION: 4th-order predict-evaluate-correct for sinks ----------------------

static inline bool hermite_type_ok(const Sim& sim, size_t i) {
    return sim.hermite_mask != 0 && !sim.P.type.empty() &&
           (sim.hermite_mask & (1 << sim.P.type[i]));
}

// eligible_for_hermite (core/kicks.cc:104): a freshly-formed sink integrates with plain KDK
// for its first couple of steps while its neighbourhood settles. The AccretedThisTimestep
// fallback of the reference is mechanical here: a swallow invalidates the snapshot, which
// forces one KDK step before Hermite re-entry.
static inline bool hermite_eligible(const Sim& sim, size_t i, double dt) {
    if (!hermite_type_ok(sim, i)) return false;
    if (i < sim.sink_tform.size() && sim.sink_tform[i] > 0 &&
        sim.sink_tform[i] >= sim.time_now() - 2.0 * dt) return false;
    return true;
}

// acc and jerk on a test state (x, v), direct-summed over every other particle with the same
// softened pair kernel as the tree walk (max-softening rule -- Hermite targets are sinks,
// never a gas-gas pair). The jerk term is forcetree.cc:2266,
//     jerk += g1 * dv - (dv . dr) * g2 * dr,
// with g1/g2 the first/second kernel derivatives. The reference computes this inside the
// gravity walk, approximating distant sources by node centre-of-mass velocities; direct
// summation is exact where that approximates, and the sink counts here (one to a few hundred)
// make it affordable. Sources are lazily caught up so distances are measured at NOW; their
// stored leapfrog velocities stand in for GIZMO's predicted velocities -- the difference is
// half a kick, and it enters only the jerk's own error term.
// Acceleration AND jerk for a set of targets, from one tree traversal -- the reference's
// COMPUTE_JERK_IN_GRAVTREE (forcetree.cc:2266), where the jerk rides along in the walk that is
// already computing the force and reuses its kernel factors. The targets must already hold the
// state to evaluate at (positions in P, velocities in sim.vx/vy/vz), exactly as
// do_hermite_prediction writes its prediction into P before the HermiteOnlyFlag=2 walk.
//
// This replaced a direct summation over every particle. That version was correct and simpler,
// but it cost O(N_target * N_total) per sync against the walk's O(N_target log N) -- ~10x on
// 512 particles and unusable for a run with real gas counts, since every sink would sum over
// every cell twice per step.
static void hermite_eval_group(Sim& sim, const std::vector<uint32_t>& targets,
                               std::vector<double>& ax, std::vector<double>& ay,
                               std::vector<double>& az, std::vector<Vec3d>& jerk) {
    const double* vel_arrays[3] = {sim.vx.data(), sim.vy.data(), sim.vz.data()};
    const LazyDrift ld = lazy_drift_hook(sim);
    accel_grouped(sim.tree, sim.P, targets, sim.theta, sim.G, 0, ax, ay, az,
                  nullptr, nullptr, sim.lazy_drift_on ? &ld : nullptr, &jerk, vel_arrays);
}

// Runs immediately after the gravity kick loop. KDK ran for EVERYONE, exactly as in GIZMO,
// where do_hermite_prediction/correction (run.cc:173-177) OVERWRITE the kick results for
// eligible particles -- robustness on eligibility loss (a swallow, a fresh sink) comes free
// because the KDK trajectory is always there underneath. One difference in bookkeeping: after
// our kick loop a particle's stored velocity is half-kicked INTO its next step
// (pending_half_kick). The true velocity at the sync is recovered by undoing that half-kick
// with the same acceleration the kick used, and the half-kick is re-applied to whatever
// velocity Hermite settles on -- so the sink can drop back to KDK at any sync with its
// leapfrog state intact, and the redo cancels exactly at the next sync's undo.
// SHMEM_HERMITE_DIAG counters: how often the elapsed Hermite interval differs from the step
// that was assigned when the snapshot was taken. Reported once at exit.
static long long hermite_h_mismatch = 0, hermite_h_match = 0;
static double hermite_h_mismatch_max = 0.0;
static const bool hermite_diag = getenv("SHMEM_HERMITE_DIAG") != nullptr;

void hermite_report() {
    if (!hermite_diag) return;
    const long long tot = hermite_h_match + hermite_h_mismatch;
    if (tot > 0)
        fprintf(stderr, "[hermite-h] steps=%lld  interval != assigned dt in %lld (%.3f%%), "
                "max rel deviation %.4g\n", tot, hermite_h_mismatch,
                100.0 * (double)hermite_h_mismatch / (double)tot, hermite_h_mismatch_max);
}


// GIZMO's HermiteOnlyFlag=1 pass plus the mode==0 snapshot, together (run.cc:150-153,
// kicks.cc:369-372). Runs BEFORE the kick, so the velocity here is v(t_start): the reference
// applies its two half kicks separately, while we fuse them and defer one, so the owed half kick
// is added back to recover v0.
//
// Saves the four "Old" quantities the corrector needs -- OldPos, OldVel, Hermite_OldAcc, OldJerk --
// as ONE consistent snapshot of the start-of-step state, then the KDK step proceeds provisionally
// over it, exactly as the methods paper describes ("saving the initial state of the timestep").
static void hermite_snapshot(Sim& sim, const std::vector<uint32_t>& active,
                             const std::vector<double>& dt_of) {
    if (sim.hermite_mask == 0 || sim.P.type.empty()) return;
    const size_t n_part = sim.size();
    if (sim.herm_valid.size() != n_part) {
        sim.herm_valid.assign(n_part, 0);
        sim.herm_tick.assign(n_part, 0);
        sim.herm_pos.assign(n_part, Vec3d{0, 0, 0});
        sim.herm_vel.assign(n_part, Vec3d{0, 0, 0});
        sim.herm_acc.assign(n_part, Vec3d{0, 0, 0});
        sim.herm_jerk.assign(n_part, Vec3d{0, 0, 0});
    }
    if (sim.pending_half_kick.size() != n_part || sim.a_grav.size() != n_part) return;

    std::vector<uint32_t> tg;
    for (size_t k = 0; k < active.size(); ++k) {
        const uint32_t i = active[k];
        if (i >= sim.n_gas && hermite_eligible(sim, i, dt_of[i])) tg.push_back(i);
    }
    if (tg.empty()) return;

    // v0 = stored + the owed half kick. Write it in so the walk's JERK is evaluated at v0 -- the
    // jerk depends on velocity, the acceleration does not, and this is the whole reason the pass
    // has to exist rather than reusing the step's main walk.
    std::vector<Vec3d> saved(tg.size());
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < tg.size(); ++k) {
        const uint32_t i = tg[k];
        saved[k] = Vec3d{sim.vx[i], sim.vy[i], sim.vz[i]};
        const double owed = sim.pending_half_kick[i];
        sim.vx[i] = saved[k][0] + sim.a_grav[i][0] * owed;
        sim.vy[i] = saved[k][1] + sim.a_grav[i][1] * owed;
        sim.vz[i] = saved[k][2] + sim.a_grav[i][2] * owed;
    }
    std::vector<double> ax, ay, az; std::vector<Vec3d> jk;
    hermite_eval_group(sim, tg, ax, ay, az, jk);
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < tg.size(); ++k) {
        const uint32_t i = tg[k];
        sim.herm_pos[i]  = sim.P.pos(i);
        sim.herm_vel[i]  = Vec3d{sim.vx[i], sim.vy[i], sim.vz[i]};   // v0
        sim.herm_acc[i]  = Vec3d{ax[k], ay[k], az[k]};
        sim.herm_jerk[i] = jk[k];
        sim.herm_tick[i] = sim.clock_ticks;
        sim.herm_valid[i] = 1;
        sim.vx[i] = saved[k][0]; sim.vy[i] = saved[k][1]; sim.vz[i] = saved[k][2];
    }
}

static void hermite_pass(Sim& sim, const std::vector<uint32_t>& active,
                         const std::vector<double>& dt_of) {
    if (sim.hermite_mask == 0 || sim.P.type.empty()) return;
    const size_t n_part = sim.size();
    if (sim.herm_valid.size() != n_part) {
        sim.herm_valid.assign(n_part, 0);
        sim.herm_tick.assign(n_part, 0);
        sim.herm_pos.assign(n_part, Vec3d{0, 0, 0});
        sim.herm_vel.assign(n_part, Vec3d{0, 0, 0});
        sim.herm_acc.assign(n_part, Vec3d{0, 0, 0});
        sim.herm_jerk.assign(n_part, Vec3d{0, 0, 0});
    }
    std::vector<uint32_t> targets;
    for (size_t k = 0; k < active.size(); ++k) {
        const uint32_t i = active[k];
        if (i >= sim.n_gas && hermite_type_ok(sim, i)) targets.push_back(i);
    }
    if (targets.empty()) return;

    // PHASED, exactly as the reference orders it (run.cc:173-177): predict EVERYONE, then
    // evaluate, then correct -- with a barrier between each. A fused per-target loop races on
    // binary partners: one thread evaluates its sink's jerk while the other is mid-overwrite of
    // the partner's position and velocity, and the pair sees an inconsistent mixture of pre-
    // and post-correction states. The phasing also means every evaluation sees its partner at
    // the PREDICTED state, which is what the corrector's error analysis assumes.
    const size_t nt = targets.size();
    std::vector<Vec3d> vel_true(nt);
    std::vector<uint8_t> stepping(nt, 0), eligible(nt, 0);

    // Phase A: undo the forward half-kick; write PREDICTED pos/vel for the stepping targets so
    // the walk sees every pair member at the same moment (GIZMO's do_hermite_prediction).
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < nt; ++k) {
        const uint32_t i = targets[k];
        const double dt_new = dt_of[i];
        // Runs AFTER the drift, as the reference does (run.cc:237 is past
        // find_next_sync_point_and_drift at :156). x0/v0/a0/j0 come from hermite_snapshot, taken
        // before the kick, so there is no half kick to undo here.
        vel_true[k] = Vec3d{sim.vx[i], sim.vy[i], sim.vz[i]};
        eligible[k] = hermite_eligible(sim, i, dt_new) ? 1 : 0;
        // The interval is what the CLOCK says elapsed since the snapshot, never a stored dt.
        const double h_elapsed = sim.time_of_ticks(sim.clock_ticks - sim.herm_tick[i]);
        // SHMEM_HERMITE_DIAG quantifies how often the elapsed interval differs from the step
        // that was ASSIGNED when the snapshot was taken -- i.e. how badly the earlier
        // store-a-dt version was wrong. Zero mismatches would mean that bug was inert here.
        if (hermite_diag && sim.herm_valid[i] && h_elapsed > 0) {
            const double assigned = dt_of[i];
            const double rel = std::abs(h_elapsed - assigned) / h_elapsed;
            if (rel > 1e-9) {
                #pragma omp atomic
                ++hermite_h_mismatch;
                #pragma omp critical(shmem_hermite_diag)
                if (rel > hermite_h_mismatch_max) hermite_h_mismatch_max = rel;
            } else {
                #pragma omp atomic
                ++hermite_h_match;
            }
        }
        if (sim.herm_valid[i] && eligible[k] && h_elapsed > 0) {
            stepping[k] = 1;
            // The interval since the snapshot, from the CLOCK. This pass runs after the drift has
            // advanced clock_ticks (GIZMO's find_next_sync_point_and_drift precedes run.cc:237),
            // so clock - herm_tick is exactly the step this particle just completed -- including
            // the case where its own bin is coarser than the system step, which dt_of cannot
            // represent once the caps differ between the sync it started at and this one.
            const double h = h_elapsed;
            const Vec3d x0 = sim.herm_pos[i], v0 = sim.herm_vel[i];
            const Vec3d a0 = sim.herm_acc[i], j0 = sim.herm_jerk[i];
            // predictor (kicks.cc:147-148)
            Vec3d xp = x0 + (v0 + (a0 + j0 * (h / 3.0)) * (h / 2.0)) * h;
            const Vec3d vp = v0 + (a0 + j0 * (h / 2.0)) * h;
            if (sim.box > 0) xp = fold_into_box(xp, sim.box);
            sim.P.x[i] = xp[0]; sim.P.y[i] = xp[1]; sim.P.z[i] = xp[2];
            sim.vx[i] = vp[0];  sim.vy[i] = vp[1];  sim.vz[i] = vp[2];
            sim.last_drift[i] = sim.clock_ticks;
        } else {
            // not stepping, but the walk below is over ALL targets: leave it at its true state
            sim.vx[i] = vel_true[k][0]; sim.vy[i] = vel_true[k][1]; sim.vz[i] = vel_true[k][2];
        }
    }
    // Phase B: ONE walk over the whole target set at the predicted states -- the reference's
    // HermiteOnlyFlag=2 gravity_tree() call (run.cc:174-176).
    std::vector<uint32_t> stepping_targets;
    for (size_t k = 0; k < nt; ++k) if (stepping[k]) stepping_targets.push_back(targets[k]);
    std::vector<double> ax, ay, az;
    std::vector<Vec3d> jk;
    if (!stepping_targets.empty())
        hermite_eval_group(sim, stepping_targets, ax, ay, az, jk);
    // Phase C: correct (kicks.cc:166-167).
    {
        size_t s = 0;
        for (size_t k = 0; k < nt; ++k) {
            if (!stepping[k]) continue;
            const uint32_t i = targets[k];
            // same interval the predictor used
            const double h = sim.time_of_ticks(sim.clock_ticks - sim.herm_tick[i]);
            const Vec3d v0 = sim.herm_vel[i], x0 = sim.herm_pos[i];
            const Vec3d a0 = sim.herm_acc[i], j0 = sim.herm_jerk[i];
            const Vec3d a1{ax[s], ay[s], az[s]}, j1 = jk[s];
            ++s;
            vel_true[k] = v0 + (a0 + a1) * (h * 0.5) + (j0 - j1) * (h * h / 12.0);
            Vec3d pos_c = x0 + (vel_true[k] + v0) * (h * 0.5) + (a0 - a1) * (h * h / 12.0);
            if (sim.box > 0) pos_c = fold_into_box(pos_c, sim.box);
            sim.P.x[i] = pos_c[0]; sim.P.y[i] = pos_c[1]; sim.P.z[i] = pos_c[2];
            sim.vx[i] = vel_true[k][0]; sim.vy[i] = vel_true[k][1]; sim.vz[i] = vel_true[k][2];
        }
    }
    // NO Phase D walk. The snapshot for the next interval is taken by hermite_snapshot at the
    // START of that step -- the reference's HermiteOnlyFlag=1 pass -- which is after the drift
    // has moved everything else, so a0/j0 see the configuration the interval actually begins in.
    for (size_t k = 0; k < nt; ++k) if (!eligible[k]) sim.herm_valid[targets[k]] = 0;

    // Phase E: hand the leapfrog state back (pending_half_kick is already 0.5*dt_new): stored
    // velocity is half-kicked into the next step, for the drift prediction and any KDK fallback.
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < nt; ++k) {
        const uint32_t i = targets[k];
        if (!stepping[k]) continue;
        // The corrector already produced the END-of-step velocity, so nothing is owed: zero the
        // pending half kick rather than adding one that the next step would apply again with a
        // different acceleration. Leaving it doubled measured a uniform 1.57x in the energy error.
        sim.vx[i] = vel_true[k][0]; sim.vy[i] = vel_true[k][1]; sim.vz[i] = vel_true[k][2];
        sim.pending_half_kick[i] = 0.0;
        // SHMEM_HERMITE_NOPENDING: probe for a double-applied deferred half kick. Phase E adds
        // a_N*0.5*dt_N explicitly, and pending_half_kick makes the NEXT step's kick loop add it
        // again with a_{N+1}; Phase A then removes only 0.5*dt_{N+1}*a_{N+1}, leaving
        // 0.5*dt_N*(a_N + a_{N+1}) on v0. Zeroing the pending for a sink that actually took a
        // Hermite step removes one of the two.
        static const bool no_pending = getenv("SHMEM_HERMITE_NOPENDING") != nullptr;
        if (no_pending && stepping[k]) sim.pending_half_kick[i] = 0.0;
    }
}

// Test every active gas cell and convert those that pass. Serial: formation is rare (shu1977
// forms exactly one), and the conversion reorders the arrays, so it must not race the step.
static void sink_formation_pass(Sim& sim, const std::vector<uint32_t>& active_gas,
                                const std::vector<double>& dt_of) {
    if (!sim.sink_formation || sim.crit_phys_density <= 0) return;
    const size_t n_part = sim.size();
    if (sim.alpha_vir_smoothed.size() != n_part) sim.alpha_vir_smoothed.assign(n_part, 0.0);
    if (sim.P.type.size() != n_part) sim.P.type.assign(n_part, 0);

    // SHMEM_SINK_DIAG=1 counts which veto stopped each candidate. Without it a run that forms no
    // sink gives no signal at all about WHY -- and the criteria are a chain of eight, so guessing
    // is exactly the wrong move.
    static const bool diag = getenv("SHMEM_SINK_DIAG") != nullptr;
    enum Veto { V_DENS=0, V_TSFR, V_DIVV, V_VIRIAL, V_JEANS, V_TIDAL, V_DENSMAX, V_NEARSINK,
                V_SINKTIME, V_PASS, V_NUM };
    long long veto[V_NUM] = {0};
    double rho_max_seen = 0.0, alpha_min_seen = 1e300;
    // criterion-16 inputs actually seen this sync: the smallest sink infall time among cells
    // that reached the test, and the tsfr it was compared against. 1e300 here means the
    // min_sink_* arrays are not carrying live values -- which is invisible from the veto count.
    double tsink_min_seen = 1e300, tsfr_at_min = 0.0;
    long long crit16_unavailable = 0;
    // Independent of whether any cell reaches criterion 16: is sink_timestep_pass actually
    // filling these arrays for gas? A 1e300 here means the criterion is dead on arrival.
    double tff_min_all = 1e300;
    if (diag && sim.min_sink_tff.size() == n_part)
        for (uint32_t g : active_gas) tff_min_all = std::min(tff_min_all, sim.min_sink_tff[g]);

    std::vector<uint32_t> candidates;
    for (size_t k = 0; k < active_gas.size(); ++k) {
        const uint32_t i = active_gas[k];
        if (i >= sim.n_gas) continue;
        const double rho = sim.rho[i];
        if (diag) rho_max_seen = std::max(rho_max_seen, rho);

        // (0) density threshold. Also sets tsfr, the reference timescale the other criteria use.
        if (!(rho > sim.crit_phys_density)) {
            // sfr_eff.cc:201 resets the rolling virial when a cell falls below threshold, so it
            // must re-accumulate from scratch. SHMEM_NO_VIRIAL_RESET suppresses that -- a
            // DIAGNOSTIC ONLY, to test whether the reset is what delays formation here.
            static const bool no_reset = getenv("SHMEM_NO_VIRIAL_RESET") != nullptr;
            if (!no_reset) sim.alpha_vir_smoothed[i] = 0.0;
            ++veto[V_DENS]; continue;
        }
        const double tsfr = std::sqrt(sim.crit_phys_density / rho) * sim.max_sfr_timescale;
        if (!(tsfr > 0)) { ++veto[V_TSFR]; continue; }

        // velocity-gradient terms shared by the virial and convergent-flow criteria
        const Vec3d& gx = sim.work.gradient[FIELD_VX][i];
        const Vec3d& gy = sim.work.gradient[FIELD_VY][i];
        const Vec3d& gz = sim.work.gradient[FIELD_VZ][i];
        const double divv = gx[0] + gy[1] + gz[2];
        double dv2abs = gx.norm_sq() + gy.norm_sq() + gz.norm_sq();

        // (1)+(2048) virial parameter, time-averaged. dv2abs drops the divergence part when the
        // flow is collapsing -- otherwise near-free-fall inflow counts against its own collapse --
        // and gains the thermal support term. k_cs carries the single-star pi factor.
        //
        // ORDER MATTERS HERE, and it is GIZMO's order, not a tidier one. The virial block runs at
        // sfr_eff.cc:244-282 and the convergent-flow veto only at :290 -- BEFORE this, the rolling
        // average was skipped on any step with divv >= 0. The criteria look like order-independent
        // vetoes, but this one carries STATE: AlphaVirial_SF_TimeSmoothed must be advanced on
        // EVERY step the cell is above the density threshold. The pressure-supported core at the
        // resolution limit oscillates, so divv is positive most steps; skipping the update froze
        // the average near its initial 0 and left alpha_vir = 1/avg - 1 permanently enormous, and
        // no sink could ever form no matter how long the run went.
        const double particle_size = std::pow(sim.ninv[i], 1.0 / sim.dim);
        double v_fast = sound_speed(sim, i);
        if (sim.eos_law != Sim::EosLaw::IDEAL && sim.nh_per_code_density > 0) {
            // Opacity-limit relief (sfr_eff.cc:241): once an optically-thick first core forms, its
            // real sound speed rises and the virial criterion never lets a sink form, so the
            // thermal support is capped at 0.2 km/s above n_H = 1e13. The reference writes that as
            // 0.2/UNIT_VEL_IN_KMS -- it is 0.2 KM/S, not 0.2 code units, and the two coincide only
            // because these runs happen to use km/s. A STARFORGE m/s run would cap 1000x too low.
            if (rho * sim.nh_per_code_density > 1e13 && sim.vel_to_kms > 0)
                v_fast = std::min(v_fast, 0.2 / sim.vel_to_kms);
        }
        const double k_cs = M_PI * v_fast / std::max(particle_size, 1e-300);
        // Only INFLOW is excused from counting against the virial criterion (sfr_eff.cc:254
        // guards on divv < 0): near free-fall the inflow speed itself must not bias the cell
        // against recognizing its own collapse, but outflow is genuine support.
        if (divv < 0) dv2abs -= divv * divv / 3.0;
        dv2abs += 2.0 * k_cs * k_cs;
        double alpha_vir = dv2abs / (8.0 * M_PI * sim.G * rho);
        {
            const double alpha_0 = 1.0 / (1.0 + alpha_vir);
            const double dtau = std::exp(-std::min(std::max(8.0 * dt_of[i] / tsfr, 0.0), 20.0));
            double& avg = sim.alpha_vir_smoothed[i];
            avg = std::min(std::max(avg * dtau + alpha_0 * (1.0 - dtau), 1e-10), 1.0);
            alpha_vir = 1.0 / avg - 1.0;
        }
        if (diag) alpha_min_seen = std::min(alpha_min_seen, alpha_vir);
        if (alpha_vir > 1.0) { ++veto[V_VIRIAL]; continue; }

        // (2) convergent flow. GIZMO sfr_eff.cc:290 -- after the virial block, for the reason
        // spelled out above. The SINGLE_STAR path is simply "diverging flow, no SF".
        if (divv >= 0) { ++veto[V_DIVV]; continue; }

        // (64) Jeans mass, in solar masses, against the single-star threshold
        if (sim.nh_per_code_density > 0) {
            const double nH = rho * sim.nh_per_code_density;
            const double MJ = 2.0 * std::pow(v_fast / 0.2, 3.0) / std::sqrt(nH / 760.0);
            const double m_solar = sim.P.m[i] * sim.mass_to_solar;
            const double MJ_crit = std::min(1e4, std::max(1e-3, 100.0 * m_solar));
            if (MJ > MJ_crit) { ++veto[V_JEANS]; continue; }
        }

        // (32) Hill/tidal: the cell must dominate its own Hill sphere, i.e. the tidal tensor is
        // negative definite once its own self-term is restored (the walk omits self-self).
        if (sim.tidal_criterion && sim.tidal.size() == n_part) {
            SymTensor3d T = sim.tidal[i];
            const double h_i = std::max(sim.P.soft[i], 1e-300);
            const double fac_self = -sim.P.m[i] * (2.8 / (h_i * h_i)) / h_i;
            T[0][0] += fac_self; T[1][1] += fac_self; T[2][2] += fac_self;
            const double trace = T[0][0] + T[1][1] + T[2][2];
            if (trace >= 0) { ++veto[V_TIDAL]; continue; }                 // a positive trace forces a positive eigenvalue
            if (!sym3_negative_definite(T)) { ++veto[V_TIDAL]; continue; }
        }

        // (4) local density maximum: no denser gas neighbour inside the kernel
        {
            bool denser_neighbour = false;
            std::vector<uint32_t> ngb;
            get_neighbours(sim, sim.tree, k, sim.P.pos(i), sim.h[i], ngb);
            for (uint32_t j : ngb) {
                if (j == i || j >= sim.n_gas) continue;
                if (sim.rho[j] > rho) { denser_neighbour = true; break; }
            }
            if (denser_neighbour) { ++veto[V_DENSMAX]; continue; }
        }

        // (8) no existing sink close enough to have claimed this gas already
        {
            bool near_sink = false;
            for (size_t sph = sim.n_gas; sph < n_part; ++sph) {
                // a sink on a long bin can be carrying a stale position, and this loop reads it
                // directly rather than through a search; serial here, so no lock is needed
                drift_particle_to(sim, sph, sim.clock_ticks);
                const double d = min_image(sim.P.pos(sph) - sim.P.pos(i), sim.box).norm();
                if (d < sim.h[i] || d < std::max(sim.P.soft[sph], 0.0)) { near_sink = true; break; }
            }
            if (near_sink) { ++veto[V_NEARSINK]; continue; }
            // ... and the opacity-limit floor: closer to a sink than the size of a Larson core
            // means the core belongs to that protostar. Guarded exactly as the reference guards
            // it -- #if (defined(COOLING) || defined(EOS_GMC_BAROTROPIC)) at sfr_eff.cc:347.
            // NOT implied by SINGLE_STAR_STARFORGE_DEFAULTS: COOLING is defined in the HYBRID
            // block (precompiler_logic.h:316), not the STARFORGE one (359-453), so an
            // EOS_ENFORCE_ADIABAT run like shu1977 must not apply this at all.
            if (sim.opacity_limit_physics && sim.length_to_au > 0) {
                double d_min = 1e300;
                for (size_t sph = sim.n_gas; sph < n_part; ++sph)
                    d_min = std::min(d_min,
                                     min_image(sim.P.pos(sph) - sim.P.pos(i), sim.box).norm());
                if (d_min * sim.length_to_au < 0.1) { ++veto[V_NEARSINK]; continue; }
            }
        }

        // (16) the cell must collapse on its OWN faster than it falls into the nearest sink
        // (sfr_eff.cc:352-354). Without this, gas sitting inside an existing sink's accretion
        // radius -- which is being eaten, just not this instant -- can pass every other test and
        // spawn a second sink on top of the first. shu1977 formed three that way once the sink
        // took Hermite-length steps and stopped clearing its neighbourhood every sync.
        if (!sim.min_sink_tapp.empty() && sim.min_sink_tapp.size() == n_part) {
            const double t_sink = std::min(sim.min_sink_tapp[i], sim.min_sink_tff[i]);
            if (diag && t_sink < tsink_min_seen) { tsink_min_seen = t_sink; tsfr_at_min = tsfr; }
            if (t_sink < tsfr) { ++veto[V_SINKTIME]; continue; }
        } else if (diag) {
            ++crit16_unavailable;
        }

        ++veto[V_PASS];
        candidates.push_back(i);
    }
    if (diag && !active_gas.empty() && sim.work.condition_number.size() == n_part) {
        // how ill-conditioned are the cells that matter? >100 is where GIZMO starts tightening
        // the slope limiter; if nothing reaches it, that correction is inert here.
        double cn_max = 0, cn_max_dense = 0; long long n_over100 = 0;
        for (uint32_t g : active_gas) {
            const double c = sim.work.condition_number[g];
            cn_max = std::max(cn_max, c);
            if (c > 100.0) ++n_over100;
            if (sim.rho[g] > 0.1 * sim.crit_phys_density) cn_max_dense = std::max(cn_max_dense, c);
        }
        static long long ccalls = 0;
        if ((ccalls++ % 200) == 0)
            fprintf(stderr, "[cond] nact=%zu cn_max=%.4g cn_max(dense)=%.4g n(cn>100)=%lld\n",
                    active_gas.size(), cn_max, cn_max_dense, n_over100);
    }
    if (diag && !active_gas.empty() && sim.work.condition_number.size() == n_part) {
        // Is the E-matrix conditioning bad enough to matter? GIZMO starts tightening the slope
        // limiter above 100 (gradients.cc:1005). If nothing here reaches that, the correction is
        // inert for this problem and cannot explain anything.
        double cn_max = 0, cn_max_dense = 0; long long n_over100 = 0;
        for (uint32_t g : active_gas) {
            const double c = sim.work.condition_number[g];
            cn_max = std::max(cn_max, c);
            if (c > 100.0) ++n_over100;
            if (sim.rho[g] > 0.1 * sim.crit_phys_density)
                cn_max_dense = std::max(cn_max_dense, c);
        }
        static long long ccalls = 0;
        if (ccalls < 25 || (ccalls % 500) == 0)
            fprintf(stderr, "[cond] step %-6lld nact=%-6zu cn_max=%9.4g cn_max(dense)=%9.4g "
                    "n(cn>100)=%lld\n", ccalls, active_gas.size(), cn_max, cn_max_dense,
                    n_over100);
        ++ccalls;
    }
    if (diag && !active_gas.empty()) {
        static long long calls = 0;
        if ((calls++ % 200) == 0 || veto[V_PASS])
            fprintf(stderr, "[sink-diag] nact_gas=%zu rho_max=%.4g (thresh %.4g, ratio %.3g) "
                    "alpha_min=%.4g | dens=%lld tsfr=%lld divv=%lld virial=%lld jeans=%lld "
                    "tidal=%lld densmax=%lld nearsink=%lld sinktime=%lld PASS=%lld "
                    "| t_sink_min=%.4g vs tsfr=%.4g unavail=%lld tff_min_all=%.4g\n",
                    active_gas.size(), rho_max_seen, sim.crit_phys_density,
                    rho_max_seen/sim.crit_phys_density,
                    (alpha_min_seen>1e299 ? -1.0 : alpha_min_seen),
                    veto[V_DENS], veto[V_TSFR], veto[V_DIVV], veto[V_VIRIAL], veto[V_JEANS],
                    veto[V_TIDAL], veto[V_DENSMAX], veto[V_NEARSINK], veto[V_SINKTIME],
                    veto[V_PASS], tsink_min_seen, tsfr_at_min, crit16_unavailable,
                    tff_min_all);
    }

    // Convert. Descending order so that swapping with the shrinking gas prefix cannot disturb a
    // candidate that has not been handled yet.
    std::sort(candidates.begin(), candidates.end(), std::greater<uint32_t>());
    for (uint32_t i : candidates) {
        if (i >= sim.n_gas) continue;
        const size_t last_gas = sim.n_gas - 1;
        swap_particles(sim, i, last_gas);
        sim.n_gas = last_gas;
        sim.P.type[last_gas] = 5;
        sim.u[last_gas] = 0.0;
        sim.P.soft[last_gas] = sim.soft_fixed[5] > 0 ? sim.soft_fixed[5] : sim.h[last_gas];
        // Bate-style FIXED accretion radius, set once, here (galaxy_sf/sfr_eff.cc:602-608): the
        // volume-equivalent radius at the density where the cell length equals half a Jeans
        // length, floored at the progenitor's kernel radius. cs is the 0.2 km/s isothermal value,
        // raised as n^(1/5) once the gas is opacity-limited.
        if (sim.sink_radius.size() != sim.size()) sim.sink_radius.resize(sim.size(), 0.0);
        double cs_sink = 0.2 / std::max(sim.vel_to_kms, 1e-300);
        // cs ~ n^(1/5) once opacity-limited -- but ONLY with COOLING or a GMC barotrope
        // (sfr_eff.cc:604). An EOS_ENFORCE_ADIABAT run has neither, and applying it there
        // shrinks the Jeans term for no reason.
        if (sim.opacity_limit_physics && sim.nh_per_code_density > 0) {
            const double nH = sim.rho[last_gas] * sim.nh_per_code_density;
            if (nH > 1e10) cs_sink *= std::pow(nH / 1e10, 0.2);
        }
        // The FLOOR is the sink's own force-softening kernel radius -- ForceSoftening_KernelRadius
        // of the (already type-5) particle, sfr_eff.cc:608 -- NOT the progenitor cell's SPH
        // smoothing length. Flooring on h made the accretion radius 1.78x the reference's on
        // shu1977 (1.83e-5 vs 1.02e-5), which opens an annulus between the near-sink formation
        // veto (max(h_gas, ForceSoftening[5])) and the accretion radius: gas there is nominally
        // the sink's to eat, is exempt from the veto, and piles up.
        const double soft_floor = sim.P.soft[last_gas] > 0 ? sim.P.soft[last_gas]
                                                           : sim.h[last_gas];
        sim.sink_radius[last_gas] = std::max(0.79 * sim.P.m[last_gas] * sim.G / (cs_sink*cs_sink),
                                             soft_floor);
        printf("shmem-GIZMO: sink radius = %.6g (jeans term %.6g, softening floor %.6g)\n",
               sim.sink_radius[last_gas],
               0.79 * sim.P.m[last_gas] * sim.G / (cs_sink*cs_sink), soft_floor);
        if (sim.sink_tform.size() != sim.size()) sim.sink_tform.resize(sim.size(), 0.0);
        if (sim.sink_m0.size()    != sim.size()) sim.sink_m0.resize(sim.size(), 0.0);
        if (sim.sink_reservoir.size() != sim.size()) sim.sink_reservoir.resize(sim.size(), 0.0);
        sim.sink_tform[last_gas] = sim.time_now();
        sim.sink_m0[last_gas]    = sim.P.m[last_gas];
        sim.sink_reservoir[last_gas] = 0.0;   // the disk starts empty; only swallows fill it
        // A brand-new sink starts on the DEEPEST occupied bin (core/timestep.cc:1023-1027):
        // its first accretion happens immediately and must be resolved. Deepening is always a
        // legal bin move, so this needs no alignment check.
        if (sim.individual_timesteps) {
            int deepest = 0;
            for (size_t q = 0; q < sim.size(); ++q) deepest = std::max(deepest, (int)sim.bin[q]);
            sim.bin[last_gas] = std::max((int)sim.bin[last_gas], deepest);
        }
        ++sim.sinks_formed;
        printf("shmem-GIZMO: sink formed from cell %zu (m=%.6g, rho=%.6g); %lld total\n",
               last_gas, sim.P.m[last_gas], sim.rho[last_gas], sim.sinks_formed);
        // Why did the sink-proximity criteria let this through? Print the quantities they test,
        // so a spurious second sink is diagnosable from the run log rather than by re-deriving
        // it from a snapshot afterwards.
        if (sim.sinks_formed > 1) {
            double d_near = 1e300; size_t which = 0;
            for (size_t s2 = sim.n_gas; s2 < sim.size(); ++s2) {
                if (s2 == last_gas) continue;
                const double d = min_image(sim.P.pos(s2) - sim.P.pos(last_gas), sim.box).norm();
                if (d < d_near) { d_near = d; which = s2; }
            }
            const double tapp = (sim.min_sink_tapp.size() == sim.size())
                              ? sim.min_sink_tapp[last_gas] : -1.0;
            const double tff  = (sim.min_sink_tff.size() == sim.size())
                              ? sim.min_sink_tff[last_gas]  : -1.0;
            fprintf(stderr, "[sink-why] d_nearest=%.4g (veto radius max(h=%.4g, soft=%.4g)) "
                    "r_sink_of_neighbour=%.4g | min_sink_tapp=%.4g min_sink_tff=%.4g "
                    "arrays_sized=%d\n",
                    d_near, sim.h[last_gas], sim.P.soft[which],
                    sim.sink_radius.empty() ? -1.0 : sim.sink_radius[which], tapp, tff,
                    (int)(sim.min_sink_tapp.size() == sim.size()));
        }
        fflush(stdout);
    }
    // Any conversion PERMUTES the particle arrays, so everything keyed by particle index is stale:
    // the neighbour cache (keyed by position in the active list) and the tree, whose orderbuf and
    // leaf_of still name the pre-swap indices. Both must go.
    if (!candidates.empty()) { sim.ngb_cache.clear(); sim.tree_valid = false; }
}

// SHMEM_ACCTEST: score the gravity walk against an exact O(N^2) sum on whatever was loaded as ICs
// (a GIZMO snapshot is a valid IC, so this runs on any output). It reports the per-particle force
// error and the SPURIOUS TORQUE separately, because they are different quantities and a walk can
// be better on one while worse on the other: torque comes from the ANISOTROPY of the error, not
// its size, so errors that are large but central cancel in sum(r x m a) while small correlated
// ones do not.
//
// The batch=1 arm is the point of the test. accel_grouped decides node opening once per BATCH,
// against the batch's bounding box, so every particle in a batch shares the same opened set and
// inherits the same error -- correlated at the batch scale rather than independent. Independent
// errors cancel in the torque sum; correlated ones survive it. If batch=1 injects less torque at
// equal or worse |da|, the grouping is what generates the spin-up and the fix is a smaller batch,
// not a tighter theta.
static void gravity_ground_truth(Sim& sim) {
    if (!getenv("SHMEM_ACCTEST")) return;
    const size_t n = sim.size();
    std::vector<uint32_t> targets(n);
    for (size_t i = 0; i < n; ++i) targets[i] = (uint32_t)i;

    // one origin for every solver, so the torques are directly comparable
    double cx = 0, cy = 0, cz = 0, mtot = 0;
    for (size_t i = 0; i < n; ++i) {
        cx += sim.P.m[i]*sim.P.x[i]; cy += sim.P.m[i]*sim.P.y[i]; cz += sim.P.m[i]*sim.P.z[i];
        mtot += sim.P.m[i];
    }
    if (mtot > 0) { cx /= mtot; cy /= mtot; cz /= mtot; }

    std::vector<double> bx, by, bz;
    printf("shmem-GIZMO: [acctest] exact O(N^2) reference over %zu particles...\n", n); fflush(stdout);
    const auto t0 = std::chrono::steady_clock::now();
    accel_brute(sim.P, targets, sim.G, bx, by, bz);
    printf("shmem-GIZMO: [acctest]   done in %.1f s\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count());

    Vec3d Ldir{0,0,0};
    auto torque = [&](const std::vector<double>& ax, const std::vector<double>& ay,
                      const std::vector<double>& az) {
        double Lx = 0, Ly = 0, Lz = 0;
        for (size_t i = 0; i < n; ++i) {
            const double dx = sim.P.x[i]-cx, dy = sim.P.y[i]-cy, dz = sim.P.z[i]-cz, m = sim.P.m[i];
            Lx += m*(dy*az[i] - dz*ay[i]);
            Ly += m*(dz*ax[i] - dx*az[i]);
            Lz += m*(dx*ay[i] - dy*ax[i]);
        }
        const double L = std::sqrt(Lx*Lx + Ly*Ly + Lz*Lz);
        Ldir = (L > 0) ? Vec3d{Lx/L, Ly/L, Lz/L} : Vec3d{0,0,0};   // DIRECTION: a grid-imprinted
        return L;                                                   // torque rotates with the root
    };
    // The relative criterion needs each target's PREVIOUS |a| (GIZMO's OldAcc, in the walk's no-G
    // units). Passing nothing leaves it zero, which opens every node and silently turns the walk
    // into the very brute force it is being compared against -- the first run of this test scored
    // 1e-14, i.e. round-off, identically at every batch size. The exact accelerations are the best
    // possible stand-in for a converged previous step.
    std::vector<double> aold(n);
    for (size_t i = 0; i < n; ++i)
        aold[i] = sim.err_tol_force_acc * std::sqrt(bx[i]*bx[i]+by[i]*by[i]+bz[i]*bz[i]) / sim.G;

    printf("shmem-GIZMO: [acctest] theta=%.3f errtolforceacc=%.4g  |torque| exact = %.6e\n",
           sim.theta, sim.err_tol_force_acc, torque(bx,by,bz));

    for (int batch : {8, 4, 1}) {
        std::vector<double> ax, ay, az;
        accel_grouped(sim.tree, sim.P, targets, sim.theta, sim.G, batch, ax, ay, az,
                      nullptr, aold.data());
        std::vector<double> rel; rel.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const double bmag = std::sqrt(bx[i]*bx[i] + by[i]*by[i] + bz[i]*bz[i]);
            if (!(bmag > 0)) continue;
            const double dx = ax[i]-bx[i], dy = ay[i]-by[i], dz = az[i]-bz[i];
            rel.push_back(std::sqrt(dx*dx + dy*dy + dz*dz) / bmag);
        }
        std::sort(rel.begin(), rel.end());
        const auto q = [&](double f) { return rel.empty() ? 0.0 : rel[(size_t)(f*(rel.size()-1))]; };
        const double Lmag = torque(ax, ay, az);
        printf("shmem-GIZMO: [acctest] batch=%-2d  |da|/|a| med=%.3e p95=%.3e max=%.3e   "
               "|torque|=%.6e dir=(%+.3f,%+.3f,%+.3f)\n", batch, q(0.5), q(0.95),
               rel.empty() ? 0.0 : rel.back(), Lmag, Ldir[0], Ldir[1], Ldir[2]);
        fflush(stdout);
        // SHMEM_ACCDUMP: per-particle exact and tree accelerations, keyed by ID, so the REFERENCE's
        // own walk can be scored on the same positions against the same exact sum (its [gacc]
        // probe emits the matching fields). Only the default batch is dumped.
        if (getenv("SHMEM_ACCDUMP")) {
            for (size_t i = 0; i < n; ++i)
                fprintf(stderr, "[sacc] k=%d ID=%llu bx=%.12e by=%.12e bz=%.12e "
                        "tx=%.12e ty=%.12e tz=%.12e\n", batch,
                        (unsigned long long)(sim.id.size() > i ? sim.id[i] : (long long)i),
                        bx[i], by[i], bz[i], ax[i], ay[i], az[i]);
        }
    }
}

void compute_initial_state(Sim& sim) {
    const size_t n_part = sim.size();
    sim.last_drift.assign(n_part, sim.clock_ticks);
    if (sim.atu_frac > 0) {                      // ADAPTIVE_TREEFORCE_UPDATE per-particle state
        sim.time_since_treeforce.assign(n_part, 0.0);
        sim.tdyn_for_treeforce.assign(n_part, 0.0);   // 0 => no cached force yet, forces a walk
        sim.a_grav_jerk.assign(n_part, Vec3d{0,0,0});
    }
    sim.mass_initial = 0.0;
    for (size_t i = 0; i < n_part; ++i) sim.mass_initial += sim.P.m[i];
    const size_t n_gas = std::min(sim.n_gas, n_part);
    // MaxMassForParticleSplit, from the heaviest cell present at startup (core/init.cc:965). Used
    // only to cap the mass scale in dt_accr; splitting/merging itself is not implemented.
    {
        double m_max = 0.0;
        for (size_t i = 0; i < n_gas; ++i) m_max = std::max(m_max, sim.P.m[i]);
        sim.sink_mass_split = 3.01 * m_max;
    }
    std::vector<uint32_t> gas_list(n_gas);
    for (size_t i = 0; i < n_gas; ++i) gas_list[i] = (uint32_t)i;
    rebuild_tree(sim);                       // provisional: neighbour search for the h solve
    solve_h_and_volumes(sim, sim.tree, gas_list);
    update_softenings(sim);
    // Rebuild so the node max-softenings (which gate the softening force-open in the walk) see
    // the real per-particle softenings rather than the zeros the driver loaded with -- the t=0
    // potential is computed from this tree.
    rebuild_tree(sim);
    // Retire the geometric criterion now that the walks above have given every particle an aold,
    // unless the config asked for the hybrid. This is gravtree.cc:489 -- the reference zeroes
    // ErrTolTheta at the END of the first gravity_tree(), which accel.cc:49 schedules at
    // Ti_Current==0 for exactly this purpose. Leaving it live outside STARFORGE configs opens the
    // union of both criteria, which is more nodes than the reference visits and measurably slower.
    if (!sim.hybrid_opening) sim.theta = 0.0;
    gravity_ground_truth(sim);   // SHMEM_ACCTEST only; a no-op otherwise
}

void compute_potential(Sim& sim) {
    const size_t n_part = sim.size();
    sim.phi.assign(n_part, 0.0);
    if (!sim.gravity_on) return;
    sync_all_positions(sim);        // GIZMO gravity/potential.cc:68 -- everyone, before the walk
    if (!sim.tree_valid || sim.tree.nnodes() == 0) rebuild_tree(sim);
    std::vector<uint32_t> all_particles(n_part);
    for (size_t i = 0; i < n_part; ++i) all_particles[i] = (uint32_t)i;
    potential(sim.tree, sim.P, all_particles, sim.theta, sim.G, sim.phi);
}

void set_time_base(Sim& sim, double interval, double max_step) {
    // n = how many equal pieces the snapshot interval must be cut into for each to fit inside
    // max_step. Using interval/n rather than max_step itself is what makes the boundary exact.
    const double n = std::ceil(interval / max_step - 1e-12);
    sim.dt_base = interval / std::max(1.0, n);
    sim.clock_ticks = 0;
}

// Per-criterion timestep values, filled by desired_dt for diagnostics.
struct DtParts {
    double cfl = 1e300, accel = 1e300, tidal = 1e300, selfgrav = 1e300, sink2body = 1e300,
           sinkgas = 1e300, accr = 1e300;
};
static double desired_dt(const Sim& sim, size_t i, DtParts* parts = nullptr);

void print_timebins(const Sim& sim, double systemstep, double time) {
    if (sim.bin.empty()) return;
    const int n_bins = Sim::MAX_BINS + 1;
    std::vector<long long> count(n_bins, 0), count_nc(n_bins, 0);   // cells / non-cells (sinks)
    for (size_t i = 0; i < sim.size(); ++i)
        (i < sim.n_gas ? count : count_nc)[sim.bin[i]]++;

    // cumulative[b] = particles in bin b and in every SHORTER-step bin (higher index here)
    std::vector<long long> cumulative(n_bins, 0);
    long long running = 0;
    for (int b = n_bins - 1; b >= 0; --b) {
        running += count[b] + count_nc[b];
        cumulative[b] = running;
    }

    int longest_active = -1;
    for (int b = 0; b < n_bins; ++b)
        if (count[b] + count_nc[b] > 0 && (sim.clock_ticks % sim.ticks_in_bin(b)) == 0)
            { longest_active = b; break; }

    // cpu-frac: weight each bin's average cost by how OFTEN it recurs. Halving dt doubles the
    // number of syncs, so a cheap deep bin can still dominate the run; that is the whole point of
    // the column, and a raw average would hide it.
    std::vector<double> avg(n_bins, 0.0), frac(n_bins, 0.0);
    for (int b = 0; b < n_bins; ++b)
        if (b < (int)sim.bin_cpu_n.size() && sim.bin_cpu_n[b] > 0)
            avg[b] = sim.bin_cpu_sum[b] / (double)sim.bin_cpu_n[b];
    double frac_sum = 0.0;
    double weight = 1.0;
    for (int b = 0; b < n_bins; ++b) {
        if (count[b] + count_nc[b] == 0) continue;
        frac[b] = weight * avg[b];
        frac_sum += frac[b];
        weight *= 2.0;                     // each deeper bin is visited twice as often
    }
    if (frac_sum > 0) for (int b = 0; b < n_bins; ++b) frac[b] /= frac_sum;

    printf("\nSync-Point %lld, Time: %.16g, Systemstep: %g\n", sim.sync_point, time, systemstep);
    printf("Occupied timebins:  non-cells       cells       dt                cumulative A    avg-time  cpu-frac\n");
    long long total_active = 0, total_active_nc = 0;
    for (int b = 0; b < n_bins; ++b) {
        if (count[b] + count_nc[b] == 0) continue;
        const bool active = (sim.clock_ticks % sim.ticks_in_bin(b)) == 0;
        printf(" %c  bin=%2d       %10lld  %10lld   %16.12f      %10lld %c  %10.2f    %5.1f%%\n",
               active ? 'X' : ' ', b, count_nc[b], count[b], sim.dt_of_bin(b), cumulative[b],
               (b == longest_active) ? '<' : ' ', avg[b] * 1e3, 100.0 * frac[b]);
        if (active) { total_active += count[b]; total_active_nc += count_nc[b]; }
    }
    printf("               ------------------------\n");
    printf("Total active:    %10lld  %10lld    Sum: %10lld\n\n",
           total_active_nc, total_active, total_active + total_active_nc);

    // SHMEM_BIN_DIAG: show WHY the deepest bin is deep. dt = cfl*h/vsig, so a particle lands there
    // either because its h collapsed or because its signal speed blew up, and the two point at
    // completely different causes -- clustering versus spurious heating. Printing the median of the
    // shallowest occupied bin alongside gives the scale to judge against.
    if (getenv("SHMEM_BIN_DIAG")) {
        int deepest = -1, shallowest = -1;
        for (int b = n_bins - 1; b >= 0; --b) if (count[b] > 0) { deepest = b; break; }
        for (int b = 0; b < n_bins; ++b)      if (count[b] > 0) { shallowest = b; break; }
        if (deepest > shallowest && !sim.h.empty()) {
            printf("  [bin-diag] deepest bin %d vs shallowest %d:\n", deepest, shallowest);
            int shown = 0;
            for (size_t i = 0; i < sim.size() && shown < 4; ++i) {
                if (sim.bin[i] != deepest) continue;
                ++shown;
                const double cs = sound_speed(sim, i);
                DtParts dp;
                desired_dt(sim, i, &dp);
                printf("      deep  i=%7zu h=%.4e rho=%.4e P=%.4e cs=%.3f vsig=%.3f "
                       "h/vsig=%.3e x=(%.3f,%.3f)\n", i, sim.h[i], sim.rho[i], sim.press[i], cs,
                       sim.work.signal_speed[i], sim.h[i]/(sim.work.signal_speed[i]+1e-300),
                       sim.P.x[i], sim.P.y[i]);
                printf("            dt: cfl=%.3e accel=%.3e tidal=%.3e selfgrav=%.3e "
                       "sink=%.3e soft=%.3e |a|=%.3e\n", dp.cfl, dp.accel, dp.tidal,
                       dp.selfgrav, dp.sink2body, sim.P.soft[i], sim.a_grav[i].norm());
            }
            shown = 0;
            for (size_t i = 0; i < sim.size() && shown < 2; ++i) {
                if (sim.bin[i] != shallowest) continue;
                ++shown;
                const double cs = sound_speed(sim, i);
                printf("      shal  i=%7zu h=%.4e rho=%.4e P=%.4e cs=%.3f vsig=%.3f "
                       "h/vsig=%.3e x=(%.3f,%.3f)\n", i, sim.h[i], sim.rho[i], sim.press[i], cs,
                       sim.work.signal_speed[i], sim.h[i]/(sim.work.signal_speed[i]+1e-300),
                       sim.P.x[i], sim.P.y[i]);
            }
        }
    }
    fflush(stdout);
}

// Desired step for one particle, from the same criteria the global scheme uses.
// `parts`, when non-null, receives the per-criterion values for diagnostics.
static double desired_dt(const Sim& sim, size_t i, DtParts* parts) {
    // CFL exactly as GIZMO's (core/timestep.cc): CourantFac * L_particle / (0.5 * MaxSignalVel),
    // where L_particle is the EFFECTIVE CELL SIZE (4pi/3)^(1/3) h / Neff^(1/3) -- which is
    // algebraically just V_i^(1/dim), and V_i = ninv is already solved. Using the full kernel
    // radius h instead is ~5% off in 3D at Neff~40 but 25%+ too permissive in 2D.
    // Gas only: a collisionless particle has no signal speed and is limited by the gravity
    // criteria below (plus dt_base / MaxSizeTimestep, exactly as in GIZMO).
    const bool gas = (i < sim.n_gas);
    double dt = gas ? 2.0 * sim.cfl * std::pow(sim.ninv[i], 1.0 / sim.dim)
                      / (sim.work.signal_speed[i] + 1e-300)
                    : 1e300;
    if (parts) parts->cfl = dt;
    if (sim.gravity_on) {
        // WHICH acceleration this criterion sees (core/timestep.cc:348-379). It starts as the
        // GRAVITATIONAL acceleration and picks up the hydro one for gas -- except under
        // TIDAL_TIMESTEP_CRITERION, where the gravitational part is zeroed outright, because the
        // tidal tensor is already supplying the gravity timestep and counting it twice pins every
        // cell to its softening-over-gravity time. Using a_grav here regardless made this the
        // binding criterion for 98.5% of cells where the reference is limited by the tidal term.
        Vec3d acc_for_dt = sim.tidal_criterion ? Vec3d{0, 0, 0} : sim.a_grav[i];
        if (gas && sim.a_hydro.size() == sim.size()) acc_for_dt += sim.a_hydro[i];
        const double accel_mag = acc_for_dt.norm();
        if (accel_mag > 0) {
            // sqrt(2 eta (KERNEL_CORE_SIZE * eps) / |a|) with KERNEL_CORE_SIZE = 1/2 for the
            // cubic spline: GIZMO measures the softening scale by the kernel CORE, not the full
            // support radius. Omitting the 1/2 made this criterion sqrt(2) too permissive.
            const double dt_accel = std::sqrt(sim.eta_grav * sim.P.soft[i] / accel_mag);
            if (parts) parts->accel = dt_accel;
            dt = std::min(dt, dt_accel);
        }
        if (sim.tidal_criterion && sim.tidal.size() == sim.size()) {
            // dt = 0.5 sqrt(eta / sqrt(||G T||_F^2 / 6)): recovers sqrt(eta) * t_dyn in a
            // Keplerian potential. Gas additionally floors at its own self-gravity timescale.
            const double tnorm = sim.G * sim.tidal[i].frobenius_norm();
            if (tnorm > 0) {
                double dt_tidal = 0.5 * std::sqrt(sim.eta_grav / (tnorm / std::sqrt(6.0)));
                if (parts) parts->tidal = dt_tidal;
                if (gas) {                   // gas additionally floors at its self-gravity time
                    const double dt_sg = std::sqrt(sim.eta_grav / (sim.G * sim.rho[i] + 1e-300));
                    if (parts) parts->selfgrav = dt_sg;
                    dt_tidal = std::min(dt_tidal, dt_sg);
                }
                // ADAPTIVE_TREEFORCE_UPDATE reads this as the cadence for refreshing the cached
                // tree force (timestep.cc:443, tdyn_step_for_treeforce = dt_tidal). Stored after
                // the self-gravity floor, exactly where the reference stores it.
                if (sim.atu_frac > 0 && sim.tdyn_for_treeforce.size() == sim.size())
                    sim.tdyn_for_treeforce[i] = dt_tidal;
                dt = std::min(dt, dt_tidal);
                // SHMEM_ATU_PROBE: what ADAPTIVE_TREEFORCE_UPDATE would be worth here, measured
                // before building it. The reference refreshes a gas cell's tree force once it has
                // advanced 0.0625 of its tidal time, so the fraction of walks it would SKIP is
                // 1 - dt/(0.0625*dt_tidal). Skipping is only worth having if that is large, because
                // the price is computing a jerk on every walk that does happen -- and the tidal
                // tensor, a comparable term, already doubles this walk's cost.
                static const bool atu_probe = getenv("SHMEM_ATU_PROBE") != nullptr;
                if (atu_probe && gas) {
                    static std::atomic<long long> n_gas_seen{0}, n_would_skip{0};
                    static std::atomic<long long> ratio_milli{0};
                    const double thresh = 0.0625 * dt_tidal;
                    const double ratio = (thresh > 0) ? dt / thresh : 1.0;
                    n_gas_seen.fetch_add(1, std::memory_order_relaxed);
                    ratio_milli.fetch_add((long long)(1000.0 * std::min(ratio, 10.0)),
                                          std::memory_order_relaxed);
                    if (ratio < 1.0) n_would_skip.fetch_add(1, std::memory_order_relaxed);
                    const long long seen = n_gas_seen.load(std::memory_order_relaxed);
                    if ((seen % 2000000) == 0)
                        fprintf(stderr, "[atu] gas dt-evals=%lld  below-threshold=%.1f%%  "
                                "mean dt/(0.0625*t_tidal)=%.3f  => est. walks skipped %.1f%%\n",
                                seen, 100.0*n_would_skip.load()/seen,
                                1e-3*ratio_milli.load()/seen,
                                100.0*std::max(0.0, 1.0 - 1e-3*ratio_milli.load()/seen));
                }
            }
        }
        // SINGLE_STAR_TIMESTEPPING (core/timestep.cc:451-486). For a SINK, the two-body
        // criterion: the harmonic mean of the approach and freefall times to the nearest-in-
        // time other sink, so binaries advance in lock-step through pericentre. For GAS, the
        // FB_TIMESTEPLIMIT approach-time cap (timestep.cc:483; the reference test builds
        // define SINGLE_STAR_FB_TIMESTEPLIMIT via the STARFORGE feedback bundle).
        if (sim.min_sink_tapp.size() == sim.size() && sim.min_sink_tapp[i] < 1e299) {
            if (!gas && !sim.P.type.empty() && sim.P.type[i] == 5) {
                double dt_2body = std::sqrt(2.0 * sim.eta_grav) * 0.3
                    / (1.0 / sim.min_sink_tapp[i] + 1.0 / sim.min_sink_tff[i]);
                // Hermite tolerates a longer 2-body step (timestep.cc:456): the 0.3 safety
                // factor is a leapfrog need, not a Hermite one.
                if (hermite_eligible(sim, i, dt)) dt_2body /= 0.3;
                if (parts) parts->sink2body = dt_2body;
                dt = std::min(dt, dt_2body);
            }
            // NOT applied to gas: the 0.5*CourantFac*Min_Sink_Approach_Time cap on cells
            // (timestep.cc:483) sits inside SINGLE_STAR_FB_TIMESTEPLIMIT, which requires one of
            // the feedback modules (JETS/WINDS/SNE/RAD/RT). The plain STARFORGE defaults block
            // only TESTS for those, never defines them, so a run like shu1977 -- STARFORGE
            // defaults plus an EOS and nothing else -- does not have it. Applying it anyway
            // shortens every cell's step near a sink for no reason the reference shares.
            // (sink-gas caps for the sink ITSELF are applied below, outside this
            // other-sinks-exist guard: a LONE sink still needs its gas coupling.)
        }
        // Hermite earns a longer step overall -- timestep.cc:477, "gives 10^-6 energy error
        // per orbit for a 0.9 eccentricity binary". Applied before the sink-gas ceiling, as
        // in the reference's ordering.
        if (!gas && !sim.P.type.empty() && sim.P.type[i] == 5 &&
            hermite_eligible(sim, i, dt)) dt *= 1.4;
        // SHMEM_SINK_DT_FAC scales a SINK's step by a constant, to test directly whether the
        // sink's timestep is what drives the momentum injection.
        static const double sink_dt_fac = getenv("SHMEM_SINK_DT_FAC")
                                        ? atof(getenv("SHMEM_SINK_DT_FAC")) : 1.0;
        // dt_accr (core/timestep.cc:989): resolve the star's GROWTH, capping the step at the time
        // to add a tenth of the smaller of its own mass and the split mass. Uses the previous
        // step's dt for the drain-time floor, as the reference does -- the quantity is otherwise
        // self-referential. In the reference this binds ~40% of the time over the window where
        // the spurious second sink appears, and is what steps its sink ~2x finer than the
        // gas-neighbour cap alone would.
        if (!gas && !sim.P.type.empty() && sim.P.type[i] == 5 &&
            sim.sink_reservoir.size() == sim.size()) {
            const double dt_prev = (sim.dt_of.size() == sim.size()) ? sim.dt_of[i] : 0.0;
            const double mdot = sink_mdot(sim, i, dt_prev);
            const double m_star = sim.P.m[i] - sim.sink_reservoir[i];
            if (mdot > 0 && m_star > 0) {
                const double m_scale = sim.sink_mass_split > 0
                                     ? std::min(m_star, sim.sink_mass_split) : m_star;
                const double dt_accr = 0.1 * m_scale / mdot;
                if (parts) parts->accr = dt_accr;
                dt = std::min(dt, dt_accr);
            }
        }
        // Sink-gas ceiling (wakeup/freefall/Courant vs the surrounding gas). Deliberately
        // OUTSIDE the min_sink_tapp guard: a lone sink -- shu1977 -- sees no other sink and
        // skips the block above, but must still not sit bins above the gas it is swallowing.
        if (!gas && !sim.P.type.empty() && sim.P.type[i] == 5 &&
            sim.sink_dt_gas_cap.size() == sim.size()) {
            if (parts) parts->sinkgas = sim.sink_dt_gas_cap[i];
            dt = std::min(dt, sim.sink_dt_gas_cap[i]);
        }
        if (!gas && !sim.P.type.empty() && sim.P.type[i] == 5 && sink_dt_fac != 1.0)
            dt *= sink_dt_fac;
    }
    return dt;
}

// Which criterion actually sets the SINK's step (SHMEM_SINKDT). The 4.1x gas-neighbour ceiling is
// meant to be a backstop; if it is the binding term then the sink is not being stepped by its own
// dynamics at all, and its orbit is resolved only as well as the gas happens to require.
static void sink_dt_report(const Sim& sim) {
    static const bool on = getenv("SHMEM_SINKDT") != nullptr;
    if (!on || sim.n_gas >= sim.size()) return;
    static long long calls = 0;
    if ((calls++ % 400) != 0) return;
    size_t k = sim.n_gas;
    for (size_t i = sim.n_gas; i < sim.size(); ++i) if (sim.P.m[i] > sim.P.m[k]) k = i;
    DtParts dp;
    const double dt = desired_dt(sim, k, &dp);
    const char* who = "none"; double best = 1e300;
    auto pick = [&](double v, const char* n) { if (v > 0 && v < best) { best = v; who = n; } };
    pick(dp.accel, "accel"); pick(dp.tidal, "tidal");
    pick(dp.sink2body, "2body"); pick(dp.sinkgas, "gas-cap"); pick(dp.accr, "accr");
    fprintf(stderr, "[sinkdt] %.8e dt=%.4e BIND=%-8s accel=%.4e tidal=%.4e 2body=%.4e "
            "gascap=%.4e accr=%.4e\n", sim.time_now(), dt, who, dp.accel, dp.tidal,
            dp.sink2body, dp.sinkgas, dp.accr);
}

// Deepest bin whose step does not exceed dt_want. bin 0 is dt_base.
static int bin_for_dt(const Sim& sim, double dt_want) {
    if (dt_want >= sim.dt_base) return 0;
    int b = (int)std::ceil(std::log2(sim.dt_base / dt_want) - 1e-12);
    return std::min(std::max(b, 0), Sim::MAX_BINS);
}

// A particle may only MOVE TO A DEEPER bin (shorter step) away from its own sync point: coarsening
// there would skip time it has already been scheduled through. Deepening is always safe.
static void assign_bins(Sim& sim, const std::vector<uint32_t>& active) {
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < active.size(); ++k) {
        const uint32_t i = active[k];
        const int want = bin_for_dt(sim, desired_dt(sim, i));
        const int current = sim.bin[i];
        int b;
        if (want >= current) {
            b = want;                     // shortening the step is always safe
        } else {
            // Coarsening is only legal onto a bin whose longer step is aligned with the clock, so
            // walk UP one bin at a time from where the particle already is, stopping at the first
            // misalignment. Crucially it starts from `current`, which is aligned by construction
            // (the particle is active), so failing to coarsen leaves it where it is.
            //
            // Searching from `want` DOWNWARD instead -- deepening until something aligns -- looks
            // equivalent and is not: it is a ratchet. Once any particle reaches a deep bin the
            // clock only advances in that bin's ticks, so a particle wanting a shallow bin finds
            // it unaligned on most syncs and gets pushed deep purely for alignment. The whole box
            // then migrates to the deepest bin and can never climb back, and the run does 20x the
            // syncs it needs while every physical quantity looks perfectly healthy.
            b = current;
            while (b > want && (sim.clock_ticks & (sim.ticks_in_bin(b - 1) - 1)) == 0) --b;
        }
        sim.bin[i] = std::min(std::max(b, 0), Sim::MAX_BINS);
    }
    // SHMEM_DTDUMP: every criterion and the bin actually assigned, per active gas cell, to compare
    // the DISTRIBUTIONS against the reference's [gdt] dump at a matched time. Serial so the lines
    // stay whole. NOTE when comparing: `tidal` here is the pure tidal value, whereas the
    // reference's dt_tidal already has the self-gravity floor folded in -- match it against
    // min(tidal, selfgrav). One line per active cell per sync, so redirect this.
    if (getenv("SHMEM_DTDUMP")) {
        for (size_t k = 0; k < active.size(); ++k) {
            const uint32_t i = active[k];
            if (i >= sim.n_gas) continue;
            DtParts dp;
            const double dt = desired_dt(sim, i, &dp);
            fprintf(stderr, "[sdt] %.8e ID=%llu dt=%.6e courant=%.6e accel=%.6e tidal=%.6e "
                    "selfgrav=%.6e bin=%d h=%.6e rho=%.6e\n",
                    sim.time_now(),
                    (unsigned long long)(sim.id.size() > i ? sim.id[i] : (long long)i),
                    dt, dp.cfl, dp.accel, dp.tidal, dp.selfgrav, sim.bin[i], sim.h[i], sim.rho[i]);
        }
    }
}

// Saitoh & Makino (2009) timestep limiter. A particle on a long step can be overrun by a shock
// before it ever wakes; the flux loop demotes it once an arriving pair's signal speed outruns the
// one it recorded for itself. Without this, a strong blast propagates into stale, long-binned
// material and the solution is wrong rather than merely inaccurate.

// Apply the timestep-limiter wake requests the flux loop staged (Saitoh & Makino demotions).
// MUST run before anything reorders the particle arrays: the requests hold INDICES, and sink
// formation and accretion swap particles between slots -- a request applied after a reorder
// lands on whoever moved into the slot, and near a new sink that mis-delivery seeds a wake
// cascade that ratchets the neighbourhood toward MAX_BINS in generations of wake_offset.
// Staging the writes outside the flux loop itself is still required: sim.bin is read by every
// thread while that loop runs. Deepening away from a particle's own sync point is always legal
// (see assign_bins).
static void apply_wake_requests(Sim& sim) {
    if (!sim.individual_timesteps) return;   // global scheme: fluxes() clears the list itself
    for (const auto& [j, floor_bin] : sim.wake_requests)
        if (sim.bin[j] < floor_bin) sim.bin[j] = std::min(floor_bin, Sim::MAX_BINS);
    sim.wake_requests.clear();
}

// Fill sim.active (and sim.active_gas when the layout is mixed) with the particles due at the
// current clock tick -- GIZMO's make_list_of_active_particles. Called at BOTH ends of mfm_step:
// the set whose steps BEGIN at a sync point and the set whose steps END at the next one differ
// whenever more than one timebin is occupied; conflating them puts every force evaluation at
// the wrong end of the step.
static void gather_active(Sim& sim) {
    const size_t n_part = sim.size();
    // Global scheme: everyone, every step. Individual: whoever the integer clock says is due.
    if (sim.individual_timesteps && sim.bin.size() != n_part) sim.bin.assign(n_part, 0);
    sim.active.clear();
    if (sim.individual_timesteps) {
        // Parallel, and with a MASK rather than a modulo. This loop is O(N) every sync however few
        // particles are active, so on a large run it is one of the few things standing between the
        // engine and linear scaling: serial, it costs an integer division per particle per sync and
        // does not get faster with more cores at all. ticks_in_bin is a power of two, so
        // "clock % ticks == 0" is exactly "clock & (ticks-1) == 0".
        const int nthreads = omp_get_max_threads();
        std::vector<std::vector<uint32_t>>& chunks = sim.active_chunks;
        chunks.resize(nthreads);
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            std::vector<uint32_t>& mine = chunks[tid];
            mine.clear();
            #pragma omp for schedule(static) nowait
            for (size_t i = 0; i < n_part; ++i)
                if ((sim.clock_ticks & (sim.ticks_in_bin(sim.bin[i]) - 1)) == 0)
                    mine.push_back((uint32_t)i);
        }
        size_t total = 0;
        for (const auto& c : chunks) total += c.size();
        sim.active.reserve(total);
        // static schedule hands out ascending ranges, so concatenating in thread order keeps the
        // active list sorted -- which the index tiebreak in the flux ownership rule relies on
        for (const auto& c : chunks) sim.active.insert(sim.active.end(), c.begin(), c.end());
    } else {
        sim.active.resize(n_part);
        for (size_t i = 0; i < n_part; ++i) sim.active[i] = (uint32_t)i;
    }
    // Hydro passes take only the GAS particles; with the gas-first layout that is the ascending
    // prefix of the (sorted) active list. All-gas sims skip the copy entirely (the caller aliases
    // sim.active instead).
    if (sim.n_gas < n_part)
        sim.active_gas.assign(sim.active.begin(),
                              std::lower_bound(sim.active.begin(), sim.active.end(),
                                               (uint32_t)sim.n_gas));
}

// One full force evaluation at the CURRENT positions and clock for the given active set: tree
// maintenance, density/volumes, gradients, gravity with the sink timestep quantities, face
// states, and the hydro flux RATES. This is the reference's post-drift force block: it runs at
// the TAIL of every step -- so forces are evaluated at the drifted positions, for the particles
// whose steps end there -- and once before the first step as the bootstrap (GIZMO's init computes
// forces at t=0 before the main loop). Everything it stores (signal speeds, a_grav, a_hydro,
// du_dt, sink approach times) is consumed by the NEXT step's timestep decision and opening kick.
struct EvalTimers { double tree = 0, dens = 0, grad = 0, grav = 0, flux = 0; };
static void evaluate_forces(Sim& sim, const std::vector<uint32_t>& active,
                            const std::vector<uint32_t>& active_gas, EvalTimers* et) {
    const size_t n_part = sim.size();
    auto mark = std::chrono::steady_clock::now();
    auto lap = [&]() {
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - mark).count();
        mark = now;
        return ms;
    };

    // Reuse the tree across syncs. Particles drift every sync, but only by a small fraction of a
    // kernel radius, so the SHAPE of the tree stays good for many steps; Tree::pad keeps the
    // neighbour prune conservative in the meantime, which is what makes reuse exact rather than
    // approximate. Scale for "has drift degraded this tree": the mean h over ALL gas, recorded
    // when the tree was built -- taking it from one ACTIVE particle instead ties the threshold to
    // whoever is awake, and on a deep-bin step that is a core cell whose h is orders of magnitude
    // below the box.
    double typical_h = sim.tree_typical_h;
    if (typical_h <= 0.0 && !sim.h.empty())
        typical_h = sim.h[active.empty() ? 0 : active[0]];
    // collisionless particles have no kernel radius; their softening is the resolution scale the
    // rebuild pad should track instead
    if (typical_h <= 0.0 && !sim.P.soft.empty())
        typical_h = sim.P.soft[active.empty() ? 0 : active[0]];
    // The root node's vmax bounds every particle's speed, so root_vmax * t_since_build is an upper
    // bound on how far anything can have moved since the build -- the same quantity the per-node
    // prune uses, taken globally. Rebuild once that reaches a noticeable fraction of a typical h,
    // because past that the inflated prune starts opening nodes it does not need.
    const double max_drift = (sim.tree_valid && sim.tree.nnodes() > 0)
                           ? (double)sim.tree.vmax[sim.tree.root] * sim.tree.t_since_build : 0.0;
    const bool must_rebuild = !sim.tree_valid || sim.tree.nnodes() == 0 ||
                              typical_h <= 0.0 ||
                              max_drift > sim.tree_rebuild_pad_frac * typical_h;
    if (must_rebuild) rebuild_tree(sim);
    // Make the ACTIVE set current before anything reads a position -- GIZMO's core/run.cc:588,
    // "drift the active timebins at each sync". Parallel over a list with no duplicates, so no
    // lock. Everyone else is caught up on touch, by the hook below.
    sim.lazy_drift = lazy_drift_hook(sim);
    sim.lazy_drift_on = sim.sparse_drift;
    if (sim.sparse_drift) {
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < active.size(); ++k)
            drift_particle_to(sim, active[k], sim.clock_ticks);
    }
    const Tree& tree = sim.tree;
    if (et) et->tree += lap();

    // Actives only. A halo refresh is NOT needed: with rate accumulation an inactive particle is
    // never written to, so the h, V and B it carries are exactly the self-consistent set from its
    // own last sync.
    solve_h_and_volumes(sim, tree, active_gas);
    if (et) et->dens += lap();

    // The predicted states must be current BEFORE the gradient pass: gradients, like every
    // other hydro input, are taken of the predicted fields.
    set_predicted_states(sim, active_gas);

    // Gradients: their neighbour loop is where the signal speed for the NEXT step's CFL
    // criterion comes from.
    gradients(sim, tree, active_gas);
    if (et) et->grad += lap();

    if (sim.gravity_on) compute_gravity(sim, tree, active);
    sink_accel_check(sim);
    sink_dt_report(sim);
    // Refresh the sink approach/freefall minima -- same placement as GIZMO, where these ride
    // along in the gravity walk that precedes the next get_timestep.
    if (sim.gravity_on) sink_timestep_pass(sim, active);
    if (et) et->grav += lap();

    // Hydro flux RATES for the actives. SHMEM_NO_HYDRO skips the exchange entirely, leaving pure
    // gravitational free-fall (diagnostic; separates tree asymmetry from hydro asymmetry).
    if (sim.a_hydro.size() != n_part) sim.a_hydro.assign(n_part, Vec3d{0, 0, 0});
    if (sim.du_dt.size() != n_part) sim.du_dt.assign(n_part, 0.0);
    static const bool no_hydro = getenv("SHMEM_NO_HYDRO") != nullptr;
    if (no_hydro) {
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < active_gas.size(); ++k) {
            const uint32_t i = active_gas[k];
            sim.a_hydro[i] = Vec3d{0, 0, 0};
            sim.du_dt[i] = 0.0;
        }
    } else {
        fluxes(sim, tree, active_gas, sim.dmom_x, sim.dmom_y, sim.dmom_z, sim.denergy);
        // Rates -> per-particle records, GIZMO's hydro_final_operations_and_cleanup: the momentum
        // rate becomes an acceleration and the internal-energy rate is the total-energy rate minus
        // the kinetic part at the flux-time velocity. Nothing is integrated here -- the half-kicks
        // apply these, half closing the step that ends at this sync and half opening the next.
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < active_gas.size(); ++k) {
            const uint32_t i = active_gas[k];
            const double mass = sim.P.m[i];
            const Vec3d rate{sim.dmom_x[i], sim.dmom_y[i], sim.dmom_z[i]};
            sim.a_hydro[i] = rate / mass;
            // the kinetic part at the PREDICTED velocity, the same state the fluxes were
            // computed at (hydro_toplevel.cc:734, vel_phys = VelPred)
            sim.du_dt[i] = (sim.denergy[i]
                            - (sim.work.predicted[FIELD_VX][i]*rate[0]
                             + sim.work.predicted[FIELD_VY][i]*rate[1]
                             + sim.work.predicted[FIELD_VZ][i]*rate[2])) / mass;
            sim.dmom_x[i] = 0; sim.dmom_y[i] = 0; sim.dmom_z[i] = 0; sim.denergy[i] = 0;
        }
    }
    if (et) et->flux += lap();
}

double mfm_step(Sim& sim, double dt_max) {
    const size_t n_part = sim.size();
    Work& work = sim.work;
    const auto step_start = std::chrono::steady_clock::now();

    if (sim.a_hydro.size() != n_part) sim.a_hydro.resize(n_part, Vec3d{0, 0, 0});
    if (sim.du_dt.size() != n_part) sim.du_dt.resize(n_part, 0.0);

    // ---- BEGIN-OF-STEP: the particles whose steps end AND begin at this sync point ----
    gather_active(sim);
    const std::vector<uint32_t>& active = sim.active;
    const std::vector<uint32_t>& active_gas =
        (sim.n_gas < n_part) ? sim.active_gas : sim.active;

    const bool profile = getenv("SHMEM_PROFILE") != nullptr;
    EvalTimers et;

    // Positions of the actives must be current before the Hermite snapshot reads them; under
    // sparse drift everyone else is caught up on touch, via the hook.
    sim.lazy_drift = lazy_drift_hook(sim);
    sim.lazy_drift_on = sim.sparse_drift;
    if (sim.sparse_drift) {
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < active.size(); ++k)
            drift_particle_to(sim, active[k], sim.clock_ticks);
    }

    // BOOTSTRAP. The timestep below is decided from the forces of the PREVIOUS evaluation
    // ("decide timestep based upon state from last timestep"), which the first step does not
    // have: evaluate once at t=0, exactly as the reference's init does before its main loop.
    if (!sim.forces_valid) {
        evaluate_forces(sim, active, active_gas, &et);
        apply_wake_requests(sim);
        sim.forces_valid = true;
    }

    auto mark = std::chrono::steady_clock::now();
    auto lap = [&]() {
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - mark).count();
        mark = now;
        return ms;
    };

    // ---- timestep, from the state the last step left behind ----
    std::vector<double>& dt_of = sim.dt_of;      // reused: allocating N doubles per sync is not free
    dt_of.resize(n_part);
    double dt;                                   // the interval this call advances the system by
    long long step_ticks = 0;                    // and the same interval in whole clock ticks
    // Quantise the caller's cap DOWN to whole ticks, so every dt in this step is an exact tick
    // count and the clock can never be advanced by a fraction of one (see Sim::ticks_floor).
    const double dt_cap = sim.individual_timesteps && sim.dt_base > 0
                        ? sim.time_of_ticks(sim.ticks_floor(dt_max)) : dt_max;
    if (sim.individual_timesteps) {
        assign_bins(sim, active);
        // An EMPTY active set has exactly one legitimate cause: accretion. Removing a particle
        // can empty the bins that alone were aligned with the current clock (the swallowed cells
        // near a sink are precisely the deepest-bin ones), leaving a tick no survivor syncs on.
        // GIZMO never faces this because its next sync point is derived from the particles'
        // own step-ends (core/run.cc find_timesteps / timebin bookkeeping); do the equivalent
        // here and FAST-FORWARD the clock to the earliest tick any survivor is aligned to,
        // integrating nothing in between -- there is nothing scheduled in between. Without
        // removals an empty set still means the hierarchy has desynchronised, and that case
        // stays fatal: left alone it silently turns gravity off for the rest of the run.
        if (active.empty()) {
            if (sim.cells_accreted > 0 && n_part > 0) {
                long long next_tick = LLONG_MAX;
                #pragma omp parallel for schedule(static) reduction(min:next_tick)
                for (size_t i = 0; i < n_part; ++i) {
                    const long long ticks = sim.ticks_in_bin(sim.bin[i]);
                    const long long t_i = (sim.clock_ticks / ticks + 1) * ticks;
                    next_tick = std::min(next_tick, t_i);
                }
                // Never jump past the caller's boundary (snapshot time): close only the interval
                // asked for and let the driver call again.
                const long long cap = sim.ticks_floor(dt_max);
                const long long jump = std::min(next_tick - sim.clock_ticks, cap);
                if (jump >= 1) {
                    const double dt_jump = sim.time_of_ticks(jump);
                    fprintf(stderr, "shmem: accretion emptied the active bins at clock %lld; "
                                    "fast-forwarding %lld ticks (%g) to the next scheduled "
                                    "sync.\n", sim.clock_ticks, jump, dt_jump);
                    sim.clock_ticks += jump;
                    // The pruning pads must still cover the drift the skipped interval implies:
                    // survivors WILL be drifted across it when next touched.
                    sim.tree.t_since_build += dt_jump;
                    ++sim.sync_point;
                    return dt_jump;
                }
            }
            fprintf(stderr, "shmem: FATAL -- empty active set at clock %lld (dt_base=%g). The "
                            "timestep hierarchy has desynchronised; every particle would drift "
                            "with no gravity from here on.\n", sim.clock_ticks, sim.dt_base);
            std::abort();
        }
        int deepest_active = 0;
        #pragma omp parallel for schedule(static) reduction(max:deepest_active)
        for (size_t k = 0; k < active.size(); ++k)
            deepest_active = std::max(deepest_active, sim.bin[active[k]]);
        // dt_cap can be shorter than the bin-0 step near a snapshot boundary; cap everyone.
        // also O(N) every sync, so also parallel -- and dt_of_bin divides, which is not free
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_part; ++i) dt_of[i] = std::min(sim.dt_of_bin(sim.bin[i]), dt_cap);
        dt = std::min(sim.dt_of_bin(deepest_active), dt_cap);
        // dt is now a whole number of ticks by construction: dt_of_bin(b) is 2^(MAX_BINS-b) ticks
        // and dt_cap was floored to ticks above.
        step_ticks = sim.ticks_of_time(dt);
        dt = sim.time_of_ticks(step_ticks);
        // A zero-tick step would advance neither the clock nor the time, so the driver's loop
        // could never terminate. It means the caller asked to close an interval shorter than one
        // tick, which the tick-exact driver never does -- so treat it as a bug, not a short step.
        if (step_ticks < 1) {
            fprintf(stderr, "shmem: FATAL -- zero-length step requested (dt_max=%g, one tick=%g). "
                            "The caller is trying to close a sub-tick interval.\n",
                    dt_max, sim.time_of_ticks(1));
            std::abort();
        }
    } else {
        // CFL from the Galilean-invariant signal speed (see gradients()), plus the gravity
        // criterion; one dt shared by every particle.
        dt = dt_max;
        #pragma omp parallel for reduction(min:dt) schedule(static)
        for (size_t i = 0; i < n_part; ++i) dt = std::min(dt, desired_dt(sim, i));
        std::fill(dt_of.begin(), dt_of.end(), dt);
    }

    if (getenv("SHMEM_DT_DIAG")) {
        static bool reported = false;
        if (!reported) {
            reported = true;
            size_t limiter = 0; double smallest_dt = 1e300;
            double h_min = 1e300, h_max = 0, rho_min = 1e300, rho_max = 0;
            double press_min = 1e300, press_max = -1e300;
            for (size_t i = 0; i < n_part; ++i) {
                const double dt_i = sim.cfl * sim.h[i] / (work.signal_speed[i] + 1e-300);
                if (dt_i < smallest_dt) { smallest_dt = dt_i; limiter = i; }
                h_min = std::min(h_min, sim.h[i]);         h_max = std::max(h_max, sim.h[i]);
                rho_min = std::min(rho_min, sim.rho[i]);   rho_max = std::max(rho_max, sim.rho[i]);
                press_min = std::min(press_min, sim.press[i]);
                press_max = std::max(press_max, sim.press[i]);
            }
            const Vec3d vel_limiter{sim.vx[limiter], sim.vy[limiter], sim.vz[limiter]};
            fprintf(stderr, "[dt-diag] n=%zu dim=%d h:[%.4g,%.4g] rho:[%.4g,%.4g] P:[%.4g,%.4g]\n",
                    n_part, sim.dim, h_min, h_max, rho_min, rho_max, press_min, press_max);
            fprintf(stderr, "[dt-diag] limiter i=%zu dt=%.4g h=%.4g rho=%.4g P=%.4g cs=%.4g "
                    "vsig=%.4g |v_lab|=%.4g x=(%.4g,%.4g,%.4g)\n",
                    limiter, smallest_dt, sim.h[limiter], sim.rho[limiter], sim.press[limiter],
                    std::sqrt(sim.gamma*sim.press[limiter]/sim.rho[limiter]),
                    work.signal_speed[limiter], vel_limiter.norm(),
                    sim.P.x[limiter], sim.P.y[limiter], sim.P.z[limiter]);
        }
    }

    // MOMENTUM AUDIT (SHMEM_MOMAUDIT). Total momentum can only be changed by an operation that
    // is not pairwise antisymmetric, so rather than guess which one, measure Sum(m v) either
    // side of each and attribute the change. Reported in units of M*cs so it is comparable with
    // the drift plots. Costs one O(N) reduction per probe; diagnostic only.
    static const bool momaudit = getenv("SHMEM_MOMAUDIT") != nullptr;
    Vec3d p_prev{0, 0, 0};
    // SYNCHRONISED momentum. A stored velocity is half-kicked into the particle's own next
    // step, so Sum(m v_stored) mixes kick phases and is NOT the physical momentum -- undo each
    // particle's outstanding half-kick first. Without this the audit flags any operation that
    // legitimately completes an impulse (e.g. finishing an accreted cell's kick) as if it were
    // injecting momentum.
    auto total_p = [&sim](bool sync) {
        // sim.size() -- NOT the n_part captured at the top of the step: accretion shrinks the
        // arrays mid-step, and reading the old length walks off the end.
        const size_t n_part = sim.size();
        const bool have = sync && sim.pending_half_kick.size() == n_part
                               && sim.a_grav.size() == n_part;
        const bool have_hyd = have && sim.a_hydro.size() == n_part;
        double px = 0, py = 0, pz = 0;
        #pragma omp parallel for schedule(static) reduction(+:px,py,pz)
        for (size_t i = 0; i < n_part; ++i) {
            const double owed = have ? sim.pending_half_kick[i] : 0.0;
            Vec3d a = have ? sim.a_grav[i] : Vec3d{0, 0, 0};
            if (have_hyd && i < sim.n_gas) a += sim.a_hydro[i];   // the kick owes this rate too
            px += sim.P.m[i] * (sim.vx[i] - a[0]*owed);
            py += sim.P.m[i] * (sim.vy[i] - a[1]*owed);
            pz += sim.P.m[i] * (sim.vz[i] - a[2]*owed);
        }
        return Vec3d{px, py, pz};
    };
    // Every line carries the simulation time: the mix of contributors is not constant over a
    // collapse, so a total summed over the whole run hides which phase produced it.
    // ANGULAR momentum, about the domain centre of mass. Linear momentum is conserved by face
    // antisymmetry and by pairwise gravity; ANGULAR momentum is not automatic for either -- MFM's
    // force acts along the face normal, which is not parallel to the separation, and a tree node's
    // centre of force sits wherever its mass happens to be rather than on the line of centres. So
    // this is measured separately, at the same phase boundaries.
    auto total_L = [&sim]() {
        const size_t n = sim.size();
        double cx = 0, cy = 0, cz = 0, mt = 0;
        #pragma omp parallel for schedule(static) reduction(+:cx,cy,cz,mt)
        for (size_t i = 0; i < n; ++i) {
            cx += sim.P.m[i]*sim.P.x[i]; cy += sim.P.m[i]*sim.P.y[i]; cz += sim.P.m[i]*sim.P.z[i];
            mt += sim.P.m[i];
        }
        if (mt > 0) { cx /= mt; cy /= mt; cz /= mt; }
        double lx = 0, ly = 0, lz = 0;
        #pragma omp parallel for schedule(static) reduction(+:lx,ly,lz)
        for (size_t i = 0; i < n; ++i) {
            const double dx = sim.P.x[i]-cx, dy = sim.P.y[i]-cy, dz = sim.P.z[i]-cz;
            const double m = sim.P.m[i];
            lx += m*(dy*sim.vz[i] - dz*sim.vy[i]);
            ly += m*(dz*sim.vx[i] - dx*sim.vz[i]);
            lz += m*(dx*sim.vy[i] - dy*sim.vx[i]);
        }
        return Vec3d{lx, ly, lz};
    };
    Vec3d L_prev = momaudit ? total_L() : Vec3d{0,0,0};
    auto probeL = [&](const char* what) {
        if (!momaudit) return;
        const Vec3d L = total_L();
        const Vec3d d = L - L_prev;
        fprintf(stderr, "[amom] %.8e %-14s dL=%14.6e L=%14.6e\n",
                sim.time_now(), what, d.norm(), L.norm());
        L_prev = L;
    };

    Vec3d p_prev_raw{0, 0, 0};
    auto probe = [&](const char* what) {
        if (!momaudit) return;
        const Vec3d p = total_p(true);
        const Vec3d d = p - p_prev;
        if (d.norm() > 0)
            fprintf(stderr, "[mom] %.8e %-14s %14.6e %14.6e %14.6e\n",
                    sim.time_now(), what, d[0], d[1], d[2]);
        p_prev = p;
        // RAW sum, in the SAME columns the reference's momentum_audit() prints, so the two logs
        // diff directly: t, phase, |dP|, |P|, M. The reference has no notion of an outstanding
        // half-kick, so only the raw quantity is common to both.
        const Vec3d pr = total_p(false);
        const Vec3d dr = pr - p_prev_raw;
        double mtot = 0;
        for (size_t i = 0; i < sim.size(); ++i) mtot += sim.P.m[i];
        fprintf(stderr, "[smom] %.8e %-14s %14.6e %14.6e %14.6e\n",
                sim.time_now(), what, dr.norm(), pr.norm(), mtot);
        p_prev_raw = pr;
    };
    if (momaudit) {
        p_prev = total_p(true);
        p_prev_raw = total_p(false);   // else the step's FIRST probe reports |P|, not a delta
        // RAW alongside SYNCHRONISED. A snapshot can only ever report the raw sum, so the gap
        // between the two is the part of any snapshot-derived momentum drift that is kick-phase
        // mixing rather than a real loss -- worth knowing before chasing the latter.
        const Vec3d raw = total_p(false);
        int bmin = 1 << 20, bmax = -1;
        for (size_t i = 0; i < sim.bin.size(); ++i) {
            if (sim.bin[i] < bmin) bmin = sim.bin[i];
            if (sim.bin[i] > bmax) bmax = sim.bin[i];
        }
        fprintf(stderr, "[momtot] %.8e sync %14.6e %14.6e %14.6e raw %14.6e %14.6e %14.6e "
                "nact %zu bins %d-%d\n", sim.time_now(),
                p_prev[0], p_prev[1], p_prev[2], raw[0], raw[1], raw[2],
                active.size(), bmin, bmax);
    }

    // KICK. a_grav and the hydro rates were evaluated at this sync point -- the tail of the
    // previous step, at exactly these positions -- which is BOTH the end of the previous step and
    // the start of this one. So the half-kick the previous step still owes and this step's opening
    // half-kick use the same rates and are applied together: each evaluation serves kick #2 of one
    // step and kick #1 of the next, which is what keeps the scheme a proper leapfrog on one
    // evaluation per step. The hydro rate rides in the SAME kick, exactly as the reference applies
    // it (kicks.cc: HydroAccel and DtInternalEnergy times dt_hydrokick) -- the internal energy
    // moves LINEARLY by its rate, not by an exact conserved-quantity solve; the reference
    // linearises the same way, and the time-centring comes from the two rate evaluations at the
    // interval's ends, not from any half-step face prediction.
    // SHMEM_SWALLOW_AT_SYNC splits the two halves apart and swallows between them, which is where
    // the reference does it: do_second_halfstep_kick then calculate_non_standard_physics
    // (run.cc:167-169). At that instant every active particle is exactly AT the sync point,
    // whereas the fused kick leaves them half a step ahead.
    static const bool swallow_at_sync = getenv("SHMEM_SWALLOW_AT_SYNC") != nullptr;
    {
        sim.pending_half_kick.resize(n_part, 0.0);
        // Start-of-step snapshot + the Hermite-only gravity pass, BEFORE the kick (run.cc:150-153).
        if (sim.gravity_on) hermite_snapshot(sim, active, dt_of);
        const bool have_grav = sim.gravity_on && sim.a_grav.size() == n_part;
        const bool have_pred = work.predicted[FIELD_VX].size() == n_part;
        auto rate_of = [&](uint32_t i) {
            Vec3d a = have_grav ? sim.a_grav[i] : Vec3d{0, 0, 0};
            if (i < sim.n_gas) a += sim.a_hydro[i];
            return a;
        };
        // close_dt finishes the particle's PREVIOUS step; open_dt starts its next one. In
        // between, the state is the particle's TRUE synchronised state -- and that is what the
        // predicted fields anchor to (the reference's kicks reset VelPred/InternalEnergyPred
        // there, and the drift advances them by the rates until the next anchor). Anchoring the
        // fully-kicked state instead puts the anchor half a step ahead of the trajectory.
        auto kick_one = [&](uint32_t i, double close_dt, double open_dt, const Vec3d& a_tot) {
            // the reference's floor (kicks.cc:340), per half-kick as it applies it: a kick may
            // at most halve the internal energy
            auto kick_u = [&](double kdt) {
                const double dEnt = sim.u[i] + sim.du_dt[i] * kdt;
                sim.u[i] = (dEnt < 0.5 * sim.u[i]) ? 0.5 * sim.u[i] : dEnt;
            };
            sim.vx[i] += a_tot[0] * close_dt;
            sim.vy[i] += a_tot[1] * close_dt;
            sim.vz[i] += a_tot[2] * close_dt;
            double u_true = 0.0;
            if (i < sim.n_gas) {
                kick_u(close_dt);
                u_true = sim.u[i];
                if (have_pred) {
                    work.predicted[FIELD_VX][i] = sim.vx[i];
                    work.predicted[FIELD_VY][i] = sim.vy[i];
                    work.predicted[FIELD_VZ][i] = sim.vz[i];
                }
            }
            sim.vx[i] += a_tot[0] * open_dt;
            sim.vy[i] += a_tot[1] * open_dt;
            sim.vz[i] += a_tot[2] * open_dt;
            if (i < sim.n_gas) {
                kick_u(open_dt);
                // Pressure follows u immediately; under a density-driven EOS this RESETS u from
                // P(rho) instead -- the energy equation's answer is discarded on purpose, which
                // is what makes the law stand in for cooling.
                eos_apply(sim, i);
                if (have_pred)
                    work.predicted[FIELD_PRESSURE][i] =
                        (sim.eos_law == Sim::EosLaw::IDEAL)
                            ? (sim.gamma - 1.0) * sim.rho[i] * std::max(u_true, 1e-30)
                            : sim.press[i];
            }
        };
        // Split the kick's own momentum injection by type. sum_active m a dt is not zero for a
        // SUBSET of particles even in an exactly antisymmetric force field -- the rest collect
        // their share later -- so this is bookkeeping, not error, but its SIZE says which
        // population is driving it.
        double gx = 0, gy = 0, gz = 0, sx = 0, sy = 0, sz = 0;
        #pragma omp parallel for schedule(static) reduction(+:gx,gy,gz,sx,sy,sz)
        for (size_t k = 0; k < active.size(); ++k) {
            const uint32_t i = active[k];
            const Vec3d a_tot = rate_of(i);
            const double close_dt = sim.pending_half_kick[i];
            const double open_dt = swallow_at_sync ? 0.0 : 0.5 * dt_of[i];
            const double kick_dt = close_dt + open_dt;
            kick_one(i, close_dt, open_dt, a_tot);
            if (momaudit) {
                const double dpx = sim.P.m[i] * a_tot[0] * kick_dt;
                const double dpy = sim.P.m[i] * a_tot[1] * kick_dt;
                const double dpz = sim.P.m[i] * a_tot[2] * kick_dt;
                if (i < sim.n_gas) { gx += dpx; gy += dpy; gz += dpz; }
                else               { sx += dpx; sy += dpy; sz += dpz; }
            }
            // what THIS particle will owe when it next becomes active -- half of its OWN step
            sim.pending_half_kick[i] = swallow_at_sync ? 0.0 : 0.5 * dt_of[i];
        }
        if (momaudit)
            fprintf(stderr, "[kicksplit] %.8e gas %14.6e sink %14.6e nact %zu\n",
                    sim.time_now(), Vec3d{gx, gy, gz}.norm(), Vec3d{sx, sy, sz}.norm(),
                    active.size());
        // --- the reference's swallow point: velocities are AT the sync point here ---
        if (swallow_at_sync) {
            sink_accretion_scan(sim);
            // and now open the new step
            #pragma omp parallel for schedule(static)
            for (size_t k = 0; k < active.size(); ++k) {
                const uint32_t i = active[k];
                kick_one(i, 0.0, 0.5 * dt_of[i], rate_of(i));
                sim.pending_half_kick[i] = 0.5 * dt_of[i];
            }
        }
        probe("kick");
        probeL("kick");
        sinkv_probe(sim, "kick");
    }

    // Feed this step's velocity changes into the per-node bounds -- GIZMO's force_kick_node
    // (forcetree_update.cc:78), placed where GIZMO places it: in the kick, which is now the ONLY
    // point a gas particle's velocity changes. (Hermite-corrected sinks get the same feed after
    // the corrector, below.) The climb stops at the first ancestor that already covers the speed,
    // so in steady state this is one relaxed load per active particle.
    if (sim.tree_valid && !sim.tree.leaf_of.empty()) {
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < active.size(); ++k) {
            const uint32_t i = active[k];
            const int leaf = sim.tree.leaf_of[i];
            if (leaf < 0) continue;
            const Vec3d v_now{sim.vx[i], sim.vy[i], sim.vz[i]};
            sim.tree.raise_vmax(leaf, (float)v_now.norm());
            // ... and the node MOMENTUM, the other half of force_kick_node: node_vel() reads
            // vcom + dp/mass, so the jerk sees where a node's mass is actually going rather
            // than where it was going at build time.
            if (!sim.tree.dp_x.empty() && sim.vel_at_last_kick.size() == n_part)
                sim.tree.kick_node(leaf, (v_now - sim.vel_at_last_kick[i]) * sim.P.m[i]);
            if (sim.vel_at_last_kick.size() == n_part) sim.vel_at_last_kick[i] = v_now;
        }
    }

    const double t_bins = profile ? lap() : 0.0;

    // Why this sync happened and how big it was -- captured NOW, before the end-of-step gather
    // below overwrites sim.active (which `active` references).
    const size_t n_active_top = active.size();
    int longest_active_bin = 0;
    if (sim.individual_timesteps && !active.empty()) {
        longest_active_bin = Sim::MAX_BINS;
        for (uint32_t i : active) longest_active_bin = std::min(longest_active_bin, sim.bin[i]);
    }

    // ---- DRIFT to the next sync point, then advance the clock ----
    // Every kicked particle moves with its half-kicked velocity -- the leapfrog's time-centred
    // drift, with no correction terms. A long-binned particle is drifted in several sub-steps
    // rather than one long one; its velocity is constant between its own kicks, so the sub-steps
    // sum to the same displacement. A wrapping particle needs no special handling: ngb_search
    // prunes on the MIN-IMAGE distance to a node's centre of mass, so a leaf sitting at x ~ box
    // is min-image-adjacent to a query at x ~ 0 and still gets opened.
    const long long drift_target = sim.clock_ticks + step_ticks;
    if (sim.sparse_drift) {
        // Only the ACTIVE particles. Everyone else keeps a stale position and a last_drift tick,
        // and is caught up exactly when something first looks at it -- which is what makes this
        // cost scale with the work rather than with the size of the box. Parallel over a duplicate
        // free list, so no lock is needed here.
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < active.size(); ++k)
            drift_particle_to(sim, active[k], drift_target);
    } else {
        drift_all_to(sim, drift_target);
    }
    // Motion since the build is bounded per node by Tree::vmax * this elapsed time; incremented
    // WITH the drift so the evaluation's rebuild check below already covers it.
    sim.tree.t_since_build += dt;
    // The clock advances BEFORE the end-of-step evaluation (GIZMO's find_next_sync_point_and_
    // drift precedes the force computation): the lazy catch-ups, the is_active test inside the
    // flux ownership rule, and the Hermite elapsed interval must all see the NEW time. No
    // rounding and no minimum: step_ticks was derived FROM the tick grid above, so the clock and
    // the dt every particle was integrated over agree exactly.
    if (sim.individual_timesteps) sim.clock_ticks += step_ticks;
    const double t_drift = profile ? lap() : 0.0;

    // ---- END-OF-STEP: forces at the drifted positions, for the steps that END here ----
    // A DIFFERENT set from the one kicked above whenever more than one bin is occupied. These
    // particles' rates close the steps they just finished (via the pending half-kick, applied at
    // their next activation) and feed the next timestep decision -- "density + gradients + hydro
    // force" between the drift and the closing kick, as the reference orders it.
    gather_active(sim);
    const std::vector<uint32_t>& active2 = sim.active;
    const std::vector<uint32_t>& active2_gas =
        (sim.n_gas < sim.size()) ? sim.active_gas : sim.active;
    evaluate_forces(sim, active2, active2_gas, &et);
    apply_wake_requests(sim);   // while the indices are still the flux loop's -- see the helper

    // ---- sinks: after the closing rates are in place, before the Hermite pass ----
    // GIZMO runs calculate_non_standard_physics after do_second_halfstep_kick and before the
    // Hermite prediction (run.cc:230).
    // SHMEM_SINK_PINNED nails every sink to the position it formed at and holds its velocity at
    // zero. Deliberately unphysical -- it breaks momentum conservation by construction -- and the
    // point is exactly that: if a spurious second sink still forms with the first one immovable,
    // then the fragmentation is something the GAS does.
    static const bool pin_sinks = getenv("SHMEM_SINK_PINNED") != nullptr;
    if (pin_sinks && sim.n_gas < n_part) {
        if (sim.sink_pin_x.size() != n_part) {
            sim.sink_pin_x.resize(n_part, 0.0); sim.sink_pin_y.resize(n_part, 0.0);
            sim.sink_pin_z.resize(n_part, 0.0); sim.sink_pinned.resize(n_part, 0);
        }
        for (size_t i = sim.n_gas; i < n_part; ++i) {
            if (!sim.sink_pinned[i]) {   // remember where it formed, once
                sim.sink_pin_x[i] = sim.P.x[i]; sim.sink_pin_y[i] = sim.P.y[i];
                sim.sink_pin_z[i] = sim.P.z[i]; sim.sink_pinned[i] = 1;
            }
            sim.P.x[i] = sim.sink_pin_x[i]; sim.P.y[i] = sim.sink_pin_y[i];
            sim.P.z[i] = sim.sink_pin_z[i];
            sim.vx[i] = 0.0; sim.vy[i] = 0.0; sim.vz[i] = 0.0;
            if (i < sim.pending_half_kick.size()) sim.pending_half_kick[i] = 0.0;
            if (i < sim.herm_valid.size()) sim.herm_valid[i] = 0;
        }
    }

    // Sink formation, once the fluxes for this sync are in. Serial and after the flux pass
    // because it reorders the particle arrays.
    probe("hydro-flux");
    probeL("hydro-flux");
    sink_formation_pass(sim, active2_gas, dt_of);
    probe("sink-form");
    probeL("sink-form");
    sinkv_probe(sim, "sink-form");
    // The scan already ran up at the swallow point under the split ordering; only the deferred
    // removals are left. Formation stays here either way: it reorders the arrays too, but fires
    // once or twice in a whole run, so moving it buys nothing.
    if (!swallow_at_sync) sink_accretion_scan(sim);
    sink_accretion_remove(sim);
    if (pin_sinks && sim.n_gas < sim.size() && sim.sink_pinned.size() >= sim.size())
        for (size_t i = sim.n_gas; i < sim.size(); ++i) {
            sim.P.x[i] = sim.sink_pin_x[i]; sim.P.y[i] = sim.sink_pin_y[i];
            sim.P.z[i] = sim.sink_pin_z[i];
            sim.vx[i] = 0.0; sim.vy[i] = 0.0; sim.vz[i] = 0.0;
        }
    probe("sink-accrete");
    probeL("sink-accrete");

    {
        // HERMITE PREDICT / EVALUATE / CORRECT for the steps that END here -- run.cc:237-241.
        // The corrector's output is the FINAL end-of-step state; nothing moves these particles
        // afterwards.
        hermite_pass(sim, active2, dt_of);
    }
    // Hermite-corrected sinks changed velocity after the kick's node feed: top up the bounds.
    if (sim.tree_valid && !sim.tree.leaf_of.empty() && sim.n_gas < sim.size()) {
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < active2.size(); ++k) {
            const uint32_t i = active2[k];
            if (i < sim.n_gas) continue;
            const int leaf = sim.tree.leaf_of[i];
            if (leaf < 0) continue;
            const Vec3d v_now{sim.vx[i], sim.vy[i], sim.vz[i]};
            sim.tree.raise_vmax(leaf, (float)v_now.norm());
            if (!sim.tree.dp_x.empty() && sim.vel_at_last_kick.size() == sim.size())
                sim.tree.kick_node(leaf, (v_now - sim.vel_at_last_kick[i]) * sim.P.m[i]);
            if (sim.vel_at_last_kick.size() == sim.size()) sim.vel_at_last_kick[i] = v_now;
        }
    }

    if (profile) {
        // Cumulative totals as well as the per-step lines: the per-step view is dominated by the
        // opening all-active step, but a run's cost is dominated by the many cheap deep-bin steps
        // after it. Only the totals answer "what would fixing this phase actually buy".
        static double c_tree=0, c_dens=0, c_grad=0, c_grav=0, c_bins=0, c_flux=0, c_drift=0;
        static long long c_steps = 0;
        c_tree+=et.tree; c_dens+=et.dens; c_grad+=et.grad; c_grav+=et.grav;
        c_bins+=t_bins; c_flux+=et.flux; c_drift+=t_drift; ++c_steps;
        const double tot = c_tree+c_dens+c_grad+c_grav+c_bins+c_flux+c_drift;
        // BUCKETED BY ACTIVE FRACTION. A single mean does not say where the time goes, because
        // gravity is strongly super-linear per target: measured 0.57 us/target with the whole set
        // active against ~500 us/target at 600 active. A scattered active set walks nearly as much
        // tree per target as a full one with none of the sharing, so the run's cost can sit in the
        // middle buckets while both ends look fine. Accumulated EVERY step -- sampled per-step
        // lines cannot answer this, and the two sampled views in this file disagreed by 15x when
        // they were all we had.
        static const int NBUCK = 5;
        static const char* bname[NBUCK] = {"<0.1%", "0.1-1%", "1-10%", "10-50%", ">50%"};
        static double b_grav[NBUCK]={0}, b_tot[NBUCK]={0};
        static long long b_n[NBUCK]={0};
        {
            // The evaluation phases belong to the set gathered AFTER the drift, not the one this
            // call closed -- bucket them by the EVALUATED set's fraction, and only the closing
            // work (kick/dt bookkeeping, drift) by the top set's. Keying everything on the top
            // set shifted every evaluation into the adjacent bucket on a multi-bin hierarchy.
            const double afrac = n_part ? (double)n_active_top/(double)n_part : 0.0;
            const double afrac2 = sim.size() ? (double)active2.size()/(double)sim.size() : 0.0;
            auto bucket_of = [](double f) {
                return f < 1e-3 ? 0 : f < 1e-2 ? 1 : f < 0.1 ? 2 : f < 0.5 ? 3 : 4;
            };
            const int b = bucket_of(afrac), b2 = bucket_of(afrac2);
            b_n[b]++;
            b_grav[b2] += et.grav;
            b_tot[b2] += et.tree+et.dens+et.grad+et.grav+et.flux;
            b_tot[b] += t_bins+t_drift;
        }
        if (getenv("SHMEM_PROFILE_TOTALS") && tot > 0 && (c_steps % 100) == 0) {
            // tree_builds is what makes the `tree` column readable: a large number there is either
            // many cheap checks or a few expensive REBUILDS, and only the count distinguishes them.
            // Inferring the rate from the timing alone is how the rebuild cost got misattributed.
            fprintf(stderr, "[prof-total] %lld steps  %lld rebuilds  tree=%.1f dens=%.1f grad=%.1f "
                            "grav=%.1f bins=%.1f flux=%.1f DRIFT=%.1f s  (drift %.1f%% of profiled"
                            " time)\n",
                    c_steps, sim.tree_builds, c_tree*1e-3, c_dens*1e-3, c_grad*1e-3, c_grav*1e-3,
                    c_bins*1e-3, c_flux*1e-3, c_drift*1e-3, 100.0*c_drift/tot);
            for (int b = 0; b < NBUCK; ++b) {
                if (!b_n[b]) continue;
                fprintf(stderr, "[prof-bucket] active %-7s %6lld steps  grav %7.1f s (%4.1f%% of "
                                "run, %6.0f ms/step)  all phases %7.1f s (%4.1f%%)\n",
                        bname[b], b_n[b], b_grav[b]*1e-3, 100.0*b_grav[b]/tot,
                        b_grav[b]/(double)b_n[b], b_tot[b]*1e-3, 100.0*b_tot[b]/tot);
            }
        }
        // The FIRST 8 steps, not a sample of the run -- they are the least representative steps
        // there are, taken before the timestep hierarchy has developed. Use [prof-bucket] for
        // anything about where a run's time actually goes.
        static int shown = 0;
        if (shown < 8) {
            ++shown;
            fprintf(stderr, "[prof] nact=%zu/%zu  tree=%.1f dens=%.1f grad=%.1f grav=%.1f "
                            "bins=%.1f flux=%.1f drift=%.1f  (ms)\n",
                    n_active_top, n_part, et.tree, et.dens, et.grad, et.grav,
                    t_bins, et.flux, t_drift);
        }
    }

    // Charge the cpu ledger. Since the rotation, one call does TWO sets' work: it closes the
    // step of the set gathered at the top and EVALUATES the set gathered after the drift, and
    // those differ on every tick of a multi-bin hierarchy. Charging the whole call to the first
    // set's bin dressed the near-empty deep-bin rows in the shallow bins' evaluation costs (a
    // 1-cell sync appeared to cost more than a full-box step). Charge the evaluation to the
    // EVALUATED set's bin and everything else to the closing set's bin.
    {
        sim.bin_cpu_sum.resize(Sim::MAX_BINS + 1, 0.0);
        sim.bin_cpu_n.resize(Sim::MAX_BINS + 1, 0);
        int longest_eval_bin = 0;
        if (sim.individual_timesteps && !active2.empty()) {
            longest_eval_bin = Sim::MAX_BINS;
            for (uint32_t i : active2)
                longest_eval_bin = std::min(longest_eval_bin, (int)sim.bin[i]);
        }
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
        const double t_eval = 1e-3 * (et.tree + et.dens + et.grad + et.grav + et.flux);
        const double t_rest = std::max(elapsed - t_eval, 0.0);
        sim.bin_cpu_sum[longest_eval_bin] += t_eval;
        sim.bin_cpu_sum[longest_active_bin] += t_rest;
        sim.bin_cpu_n[longest_active_bin]   += 1;
    }
    ++sim.sync_point;
    return dt;
}

Conserved totals(const Sim& sim) {
    Conserved total{0,0,0,0,0};
    for (size_t i = 0; i < sim.size(); ++i) {
        const double mass = sim.P.m[i];
        const Vec3d vel{sim.vx[i], sim.vy[i], sim.vz[i]};
        total.mass += mass;
        total.px += mass*vel[0]; total.py += mass*vel[1]; total.pz += mass*vel[2];
        total.E += mass * (sim.u[i] + 0.5*vel.norm_sq());
    }
    return total;
}

double linear_gradient_error(Sim& sim) {
    Tree tree = build(sim.P);
    std::vector<uint32_t> all_particles(sim.size());
    for (size_t i = 0; i < all_particles.size(); ++i) all_particles[i] = (uint32_t)i;
    solve_h_and_volumes(sim, tree, all_particles);

    // One distinct gradient per field, zeroed in the dead dimensions so 1D/2D stay in-plane.
    std::array<Vec3d, NUM_FIELDS> exact_gradient{
        Vec3d{0.30, -0.70, 0.11}, Vec3d{1.30, 0.20, -0.50}, Vec3d{-0.40, 0.90, 0.25},
        Vec3d{0.60, -0.15, 0.80}, Vec3d{-0.20, 0.35, 1.10}};
    for (auto& gradient : exact_gradient)
        for (int d = sim.dim; d < 3; ++d) gradient[d] = 0.0;
    // constant offsets, large enough to keep density and pressure positive across the box
    const PrimitiveState field_offset{10.0, 1.0, -2.0, 0.5, 20.0};

    for (size_t i = 0; i < sim.size(); ++i) {
        const Vec3d pos = sim.P.pos(i);
        sim.rho[i]   = field_offset[FIELD_DENSITY]  + dot(exact_gradient[FIELD_DENSITY],  pos);
        sim.vx[i]    = field_offset[FIELD_VX]       + dot(exact_gradient[FIELD_VX],       pos);
        sim.vy[i]    = field_offset[FIELD_VY]       + dot(exact_gradient[FIELD_VY],       pos);
        sim.vz[i]    = field_offset[FIELD_VZ]       + dot(exact_gradient[FIELD_VZ],       pos);
        sim.press[i] = field_offset[FIELD_PRESSURE] + dot(exact_gradient[FIELD_PRESSURE], pos);
    }

    set_predicted_states(sim, all_particles);   // gradients read the predicted fields
    gradients(sim, tree, all_particles);
    const Work& work = sim.work;

    // Score INTERIOR particles only. This is deliberately non-periodic (a linear field cannot be
    // continuous across a wrap), so particles near the edge see neighbours on one side only. There
    // the field genuinely only rises, or only falls, and the slope limiter correctly clips it --
    // legitimate behaviour that has nothing to do with the E/B algebra under test. Interior means
    // a full kernel radius clear of the bounding box in every live dimension.
    Vec3d box_lo{ 1e300,  1e300,  1e300}, box_hi{-1e300, -1e300, -1e300};
    for (size_t i = 0; i < sim.size(); ++i) {
        const Vec3d pos = sim.P.pos(i);
        for (int d = 0; d < 3; ++d) {
            box_lo[d] = std::min(box_lo[d], pos[d]);
            box_hi[d] = std::max(box_hi[d], pos[d]);
        }
    }

    double worst_error = 0;
    size_t n_scored = 0;
    for (size_t i = 0; i < sim.size(); ++i) {
        // A particle whose moments matrix was singular gets the zero-gradient fallback by design;
        // that is a separate (geometric) condition, so do not score it as an algebra failure.
        if (work.moments_inv[i].frobenius_norm_sq() == 0.0) continue;
        const Vec3d pos = sim.P.pos(i);
        bool interior = true;
        for (int d = 0; d < sim.dim; ++d)
            if (pos[d] - box_lo[d] < sim.h[i] || box_hi[d] - pos[d] < sim.h[i]) interior = false;
        if (!interior) continue;
        ++n_scored;
        for (int f = 0; f < NUM_FIELDS; ++f) {
            const double scale = exact_gradient[f].norm();
            if (scale <= 0) continue;
            worst_error = std::max(worst_error,
                                   (work.gradient[f][i] - exact_gradient[f]).norm() / scale);
        }
    }
    // An empty interior would make the test vacuously pass; say so instead.
    return (n_scored > 0) ? worst_error : -1.0;
}

double face_closure(Sim& sim, int nsample) {
    Tree tree = build(sim.P);
    std::vector<uint32_t> all_particles(sim.size());
    for (size_t i = 0; i < all_particles.size(); ++i) all_particles[i] = (uint32_t)i;
    solve_h_and_volumes(sim, tree, all_particles);
    set_predicted_states(sim, all_particles);   // gradients read the predicted fields
    gradients(sim, tree, all_particles);
    const Work& work = sim.work;

    double worst_residual = 0, largest_face = 0;
    std::vector<uint32_t> neighbours;
    for (int s = 0; s < nsample; ++s) {
        const size_t i = (size_t)((uint64_t)s * 2654435761u % sim.size());
        // union coverage: an h_i search catches pairs from i's side only; for closure include the
        // symmetric contribution by searching the largest kernel radius in the system
        double search_radius = sim.h[i];
        for (size_t j = 0; j < sim.size(); ++j) search_radius = std::max(search_radius, sim.h[j]);
        neighbours.clear();
        const Vec3d pos_i = sim.P.pos(i);
        ngb_search(tree, sim.P, pos_i, search_radius, neighbours, sim.box, sim.lazy());

        Vec3d face_sum{0, 0, 0};
        for (uint32_t j : neighbours) {
            if (j == i) continue;
            const Vec3d offset = min_image(sim.P.pos(j) - pos_i, sim.box);
            const double separation = offset.norm();
            const double weight_i = kernel_w(separation, sim.h[i], sim.dim);
            const double weight_j = kernel_w(separation, sim.h[j], sim.dim);
            if (weight_i <= 0 && weight_j <= 0) continue;
            const Vec3d face = work.moments_inv[i].matvec(offset) * (sim.ninv[i] * weight_i)
                             + work.moments_inv[j].matvec(offset) * (sim.ninv[j] * weight_j);
            face_sum += face;
            largest_face = std::max(largest_face, face.norm());
        }
        worst_residual = std::max(worst_residual, face_sum.norm());
    }
    return (largest_face > 0) ? worst_residual / largest_face : 0.0;
}

}  // namespace shmem
