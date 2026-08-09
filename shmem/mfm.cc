#include "mfm.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>

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

// Rank-d inverse of the symmetric moments matrix E, returning false when the neighbour geometry is
// degenerate; callers fall back to zero gradients there -- first order but safe, as GIZMO does.
// In 1D/2D the dead rows/columns of E are identically zero (every particle offset vanishes there),
// so the full 3x3 inverse does not exist: invert the live block and leave the rest zero, which
// makes gradients and faces exactly in-plane.
static bool invert_moments(const SymTensor3d& moments, Mat3d& inverse, int n_dims) {
    inverse = Mat3d{};                               // zeroed: also the failure/fallback state
    if (n_dims == 1) {
        if (std::abs(moments[0][0]) < 1e-300) return false;
        inverse[0][0] = 1.0 / moments[0][0];
        return true;
    }
    if (n_dims == 2) {
        const double det = moments[0][0]*moments[1][1] - moments[0][1]*moments[0][1];
        if (std::abs(det) < 1e-300) return false;
        inverse[0][0] =  moments[1][1]/det; inverse[0][1] = -moments[0][1]/det;
        inverse[1][0] = -moments[0][1]/det; inverse[1][1] =  moments[0][0]/det;
        return true;
    }
    // 3D: hand the live 3x3 to the shared, unit-tested Mat3 inverse rather than a second copy.
    const Mat3d full{{ {moments[0][0], moments[0][1], moments[0][2]},
                       {moments[1][0], moments[1][1], moments[1][2]},
                       {moments[2][0], moments[2][1], moments[2][2]} }};
    return full.invert(inverse) != 0.0;              // invert() zeroes and returns 0 if singular
}

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
static constexpr double A_LIMITER = 0.25;
static inline void limit_slope(Vec3d& gradient, double largest_rise, double largest_drop,
                               double h_lim, double stol, bool pos_preserve,
                               double d_max, double val_cen) {
    const double slope = gradient.norm();
    if (slope <= 0) return;
    double abs_max = std::abs(largest_rise), abs_min = std::abs(largest_drop);
    if (abs_max < abs_min) std::swap(abs_max, abs_min);
    const double allowed = std::min(abs_min + stol * abs_max, abs_max);
    double factor = allowed / (A_LIMITER * h_lim * slope);
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
    const double contact_speed = (std::abs(denominator) > 1e-300)
                               ? numerator / denominator : 0.5*(vnorm_left + vnorm_right);
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
        density(tree, sim.P, active, sim.des_ngb, h_guess, sim.box, sim.dim, ngb_cache,
                sim.lazy());
    for (size_t k = 0; k < active.size(); ++k) sim.h[active[k]] = solved.h[k];

    // SHMEM_NGB_DIAG: how many tree traversals the h solve costs per target. Each Newton
    // iteration is a full traversal, so this is the multiplier on the density phase and it
    // decides whether grouping the solve is worth more than grouping the single-pass consumers.
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
    const bool want_zeta = sim.gravity_on && sim.adaptive_soft && sim.dim == 3;
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
            const Vec3d vel_i{sim.vx[i], sim.vy[i], sim.vz[i]};
            get_neighbours(sim, tree, k, pos_i, sim.h[i], neighbours);

            SymTensor3d moments{0,0,0,0,0,0};     // E_i = sum_j (dx ox dx) W_ij
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
                if (separation > 0) {
                    const Vec3d rel_vel = vel_i - Vec3d{sim.vx[j], sim.vy[j], sim.vz[j]};
                    const double approach_speed = dot(rel_vel, offset) / separation;
                    const double csound_j = sound_speed(sim, j);
                    signal_speed = std::max(signal_speed,
                                            csound_i + csound_j - std::min(0.0, approach_speed));
                }
            }
            work.signal_speed[i] = signal_speed;

            Mat3d& moments_inv = work.moments_inv[i];
            if (!invert_moments(moments, moments_inv, sim.dim)) {
                // degenerate neighbour geometry: fall back to zero gradient (first order but
                // safe). Must be written explicitly -- these arrays persist between steps now,
                // so "leave it alone" would silently reuse a stale gradient forever.
                for (auto& field_gradient : work.gradient) field_gradient[i] = Vec3d{};
                continue;
            }

            const PrimitiveState field_i{sim.rho[i], sim.vx[i], sim.vy[i], sim.vz[i], sim.press[i]};
            std::array<Vec3d, NUM_FIELDS> weighted_diff_sum{};
            // widest rise and fall seen across the neighbour set, for the slope limiter below
            PrimitiveState largest_rise{}, largest_drop{};
            double max_ngb_distance = 0.0;
            for (uint32_t j : neighbours) {
                if (j >= sim.n_gas) continue;   // hydro sums are over GAS neighbours only
                const Vec3d offset = min_image(sim.P.pos(j) - pos_i, sim.box);
                const double weight = kernel_w(offset.norm(), sim.h[i], sim.dim);
                max_ngb_distance = std::max(max_ngb_distance, offset.norm());
                const PrimitiveState field_j{sim.rho[j], sim.vx[j], sim.vy[j], sim.vz[j],
                                             sim.press[j]};
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
            for (int f = 0; f < NUM_FIELDS; ++f) {
                Vec3d gradient = moments_inv.matvec(weighted_diff_sum[f]);
                const bool pos_preserve = (f == FIELD_DENSITY || f == FIELD_PRESSURE);
                limit_slope(gradient, largest_rise[f], largest_drop[f], h_lim,
                            (f == FIELD_DENSITY) ? 0.0 : stol, pos_preserve,
                            max_ngb_distance, field_i[f]);
                work.gradient[f][i] = gradient;
            }
        }
    }
}

// Lagrangian half-step prediction of primitives (MUSCL-Hancock predictor).
static void predict_half(Sim& sim, const std::vector<uint32_t>& active,
                         const std::vector<double>& dt_of) {
    Work& work = sim.work;
    work.resize(sim.size());
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < active.size(); ++k) {
        const uint32_t i = active[k];
        const double dt = dt_of[i];
        const double div_vel = work.gradient[FIELD_VX][i][0]
                             + work.gradient[FIELD_VY][i][1]
                             + work.gradient[FIELD_VZ][i][2];
        work.div_vel[i] = div_vel;   // kept for the drift-time prediction of INACTIVE particles
        const double inv_density = 1.0 / sim.rho[i], half_dt = 0.5 * dt;
        const Vec3d& grad_pressure = work.gradient[FIELD_PRESSURE][i];
        work.predicted[FIELD_DENSITY][i] = std::max(sim.rho[i] * (1.0 - half_dt*div_vel), 1e-30);
        work.predicted[FIELD_VX][i] = sim.vx[i] - half_dt * inv_density * grad_pressure[0];
        work.predicted[FIELD_VY][i] = sim.vy[i] - half_dt * inv_density * grad_pressure[1];
        work.predicted[FIELD_VZ][i] = sim.vz[i] - half_dt * inv_density * grad_pressure[2];
        work.predicted[FIELD_PRESSURE][i] =
            std::max(sim.press[i] * (1.0 - sim.gamma*half_dt*div_vel), 1e-30);
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
static void update_softenings(Sim& sim) {
    const size_t n_part = sim.size();
    sim.P.soft.resize(n_part);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_part; ++i) {
        if (i < sim.n_gas)
            sim.P.soft[i] = sim.adaptive_soft ? std::max(sim.h[i], sim.soft_min) : sim.soft_min;
        else
            sim.P.soft[i] = sim.soft_fixed[sim.P.type.empty() ? 1 : sim.P.type[i]];
    }
}

static void compute_gravity(Sim& sim, const Tree& tree, const std::vector<uint32_t>& active) {
    const size_t n_part = sim.size();
    update_softenings(sim);

    // BATCH SPATIAL COHERENCE. accel_grouped walks once per batch of 8 targets and opens the
    // UNION of what the batch needs, so a batch only pays off when its 8 targets are spatially
    // close. The active list is INDEX-ordered, which is spatially coherent only when the IC
    // happened to be written in a space-filling order -- lattices are, but plummer's IC is
    // radius-sorted, so consecutive indices are 8 same-shell targets scattered across the whole
    // sphere and the union walk opened most of the tree per batch: measured 5x slower per target
    // than pytreegrav's reference walk at the same theta on the same machine. Re-emitting the
    // actives in the tree's own Morton order makes every batch compact for ANY IC ordering, at
    // one O(N) pass -- the same cost class as the active-set scan that runs every sync anyway.
    std::vector<uint32_t>& targets = sim.grav_targets;
    targets.clear();
    targets.reserve(active.size());
    for (size_t r = 0; r < n_part; ++r) {
        const uint32_t i = tree.orderbuf[r];
        if (sim.is_active(i)) targets.push_back(i);
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
    accel_grouped(tree, sim.P, targets, sim.theta, sim.G, 8, ax, ay, az, tidal_out, aold_ptr,
                  sim.lazy());
    if (sim.tidal_criterion) {
        sim.tidal.resize(n_part);
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < targets.size(); ++k) sim.tidal[targets[k]] = tidal_active[k];
    }

    sim.a_grav.resize(n_part);
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < targets.size(); ++k)
        sim.a_grav[targets[k]] = Vec3d{ax[k], ay[k], az[k]};
}

// Flux exchange over unique pairs. Pair discovery from the SMALLER kernel side would miss
// asymmetric pairs, so: i owns the pair when (i < j) and r < max(h_i, h_j); every pair is then
// found exactly once because both sides search with max(h_i, h_j) coverage via the union below.
static void fluxes(Sim& sim, const Tree& tree, const std::vector<uint32_t>& active,
                   const std::vector<double>& dt_of,
                   std::vector<double>& dmom_x, std::vector<double>& dmom_y,
                   std::vector<double>& dmom_z, std::vector<double>& denergy) {
    const Work& work = sim.work;
    const size_t n_part = sim.size();
    // NOT cleared here. The accumulators are left zeroed by the update pass that consumes them
    // (see mfm_step), which already touches every particle for the drift -- so the zeroing rides
    // along in a pass that has to happen anyway instead of costing a separate sweep of the arrays.
    if (dmom_x.size() != n_part) {
        dmom_x.assign(n_part, 0); dmom_y.assign(n_part, 0);
        dmom_z.assign(n_part, 0); denergy.assign(n_part, 0);
    }
    sim.wake_requests.clear();
    #pragma omp parallel
    {
        std::vector<uint32_t> neighbours;
        std::vector<std::pair<uint32_t,int>> local_wakes;
        #pragma omp for schedule(dynamic, 64) nowait
        for (size_t k = 0; k < active.size(); ++k) {
            const uint32_t i = active[k];
            // search with h_i; pairs where h_j > r >= h_i are found from j's side (j also loops)
            const Vec3d pos_i = sim.P.pos(i);
            get_neighbours(sim, tree, k, pos_i, sim.h[i], neighbours);
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

                // effective face A_ij (points i -> j): V_i B_i dx W_i(r) + V_j B_j dx W_j(r)
                const Vec3d face = work.moments_inv[i].matvec(offset) * (sim.ninv[i] * weight_i)
                                 + work.moments_inv[j].matvec(offset) * (sim.ninv[j] * weight_j);
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
                } else if (sim.individual_timesteps &&
                           sim.bin[j] < sim.bin[i] - sim.bin_limit) {
                    // Saitoh-Makino wakeup, recorded HERE rather than in a pass of its own. The
                    // neighbours are already in hand, so a separate sweep would just repeat this
                    // whole tree walk for nothing -- and GIZMO likewise raises it from inside its
                    // hydro neighbour loop. It is a deferred request: applying it immediately
                    // would mutate bins while other threads are still reading them.
                    local_wakes.emplace_back(j, sim.bin[i] - sim.bin_limit);
                }
            }
        }
        if (!local_wakes.empty()) {
            #pragma omp critical
            sim.wake_requests.insert(sim.wake_requests.end(),
                                     local_wakes.begin(), local_wakes.end());
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
    sim.tree = build(sim.P, nullptr, vel);
    sim.tree.t_since_build = 0.0;
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
    sw(sim.phi); sw(sim.a_grav); sw(sim.tidal); sw(sim.pending_half_kick);
    sw(sim.bin); sw(sim.dt_of); sw(sim.alpha_vir_smoothed); sw(sim.last_drift);
    sw(sim.sink_radius); sw(sim.id);
    sw(sim.dmom_x); sw(sim.dmom_y); sw(sim.dmom_z); sw(sim.denergy);
    sw(sim.work.moments_inv); sw(sim.work.signal_speed); sw(sim.work.div_vel);
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
    pop(sim.phi); pop(sim.a_grav); pop(sim.tidal); pop(sim.pending_half_kick);
    pop(sim.bin); pop(sim.dt_of); pop(sim.alpha_vir_smoothed); pop(sim.last_drift);
    pop(sim.sink_radius); pop(sim.id);
    pop(sim.dmom_x); pop(sim.dmom_y); pop(sim.dmom_z); pop(sim.denergy);
    pop(sim.work.moments_inv); pop(sim.work.signal_speed); pop(sim.work.div_vel);
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
static void sink_accretion_pass(Sim& sim) {
    if (!sim.sink_formation || sim.n_gas >= sim.size()) return;
    static const bool diag = getenv("SHMEM_SINK_DIAG") != nullptr;

    // TWO PHASES, and the split is not stylistic. Deleting a gas cell slides the last sink into
    // the slot it vacated, so any sink index held across a deletion is stale -- the first version
    // of this kept eating with a stale index and corrupted the sink's mass. So: scan and decide
    // first, touching nothing, then apply every deletion afterwards.
    std::vector<uint32_t> doomed;              // gas indices to remove, ascending
    std::vector<char> claimed(sim.n_gas, 0);   // a cell may only be swallowed once
    std::vector<uint32_t> ngb;

    for (size_t sph = sim.n_gas; sph < sim.size(); ++sph) {
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
            // the cell must be smaller than the sink it falls into (sink.cc:127)
            if (std::pow(sim.ninv[j], 1.0/sim.dim) > r_sink * 1.396263) continue;

            const Vec3d dv{sim.vx[j] - sim.vx[sph], sim.vy[j] - sim.vy[sph],
                           sim.vz[j] - sim.vz[sph]};
            const double vrel_sq = dv.norm_sq();
            // internal energy enters as an effective speed; gamma ~ 1 is the isothermal hack,
            // where GIZMO uses 3P/rho rather than 2u (sink.cc:116)
            const double cs_sq = (std::abs(sim.gamma - 1.0) < 0.1)
                               ? 3.0 * sim.press[j] / sim.rho[j] : 2.0 * sim.u[j];
            // enclosed mass: the two bodies plus an isothermal-sphere gas interior (sink.cc:97)
            const double m_eff = sim.P.m[sph] + sim.P.m[j] + 4.0*M_PI * r*r*r * sim.rho[j];
            const double vesc_sq = 2.0 * sim.G * m_eff / r;
            if (!(vesc_sq > 0)) continue;
            if ((vrel_sq + cs_sq) / vesc_sq >= 1.0) continue;                    // unbound
            // Bate (1995): angular momentum small enough to actually reach the sink
            const double rv = dot(dx, dv);
            const double spec_mom_sq = r*r*vrel_sq - rv*rv;
            if (spec_mom_sq >= sim.G * (sim.P.m[sph] + sim.P.m[j]) * r_sink) continue;

            // SWALLOW. Mass and momentum conserved exactly; the sink keeps its position (GIZMO
            // does not recentre single-star sinks) and absorbs the pair's momentum. Safe to apply
            // now -- this mutates only the SINK, and no index moves until the removal phase.
            const double m_new = sim.P.m[sph] + sim.P.m[j];
            sim.vx[sph] = (sim.P.m[sph]*sim.vx[sph] + sim.P.m[j]*sim.vx[j]) / m_new;
            sim.vy[sph] = (sim.P.m[sph]*sim.vy[sph] + sim.P.m[j]*sim.vy[j]) / m_new;
            sim.vz[sph] = (sim.P.m[sph]*sim.vz[sph] + sim.P.m[j]*sim.vz[j]) / m_new;
            sim.P.m[sph] = m_new;
            claimed[j] = 1; doomed.push_back(j); ++sim.cells_accreted;
            if (diag)
                fprintf(stderr, "[sink-eat] sink=%zu ate cell=%u r/r_sink=%.3g vrel/vesc=%.3g "
                        "M=%.6g\n", sph, j, r/r_sink,
                        std::sqrt((vrel_sq+cs_sq)/vesc_sq), m_new);
        }
    }
    if (doomed.empty()) return;
    // Accretion is the one operation that moves mass BETWEEN particle types, so it is the one
    // that can silently break the books. Check the total against t=0 every time it fires.
    // DESCENDING: removing index j swaps in the particle at n_gas-1, which is always >= j, so a
    // still-pending (smaller) index is never the one moved into place.
    std::sort(doomed.begin(), doomed.end(), std::greater<uint32_t>());
    for (uint32_t j : doomed) remove_gas_particle(sim, j);
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
                V_PASS, V_NUM };
    long long veto[V_NUM] = {0};
    double rho_max_seen = 0.0, alpha_min_seen = 1e300;

    std::vector<uint32_t> candidates;
    for (size_t k = 0; k < active_gas.size(); ++k) {
        const uint32_t i = active_gas[k];
        if (i >= sim.n_gas) continue;
        const double rho = sim.rho[i];
        if (diag) rho_max_seen = std::max(rho_max_seen, rho);

        // (0) density threshold. Also sets tsfr, the reference timescale the other criteria use.
        if (!(rho > sim.crit_phys_density)) { sim.alpha_vir_smoothed[i] = 0.0; ++veto[V_DENS]; continue; }
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
            // opacity-limit relief: without it a run that resolves the first core bogs down and
            // can never form the sink at all
            if (rho * sim.nh_per_code_density > 1e13) v_fast = std::min(v_fast, 0.2);
        }
        const double k_cs = M_PI * v_fast / std::max(particle_size, 1e-300);
        dv2abs -= divv * divv / 3.0;
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
        }

        ++veto[V_PASS];
        candidates.push_back(i);
    }
    if (diag && !active_gas.empty()) {
        static long long calls = 0;
        if ((calls++ % 200) == 0 || veto[V_PASS])
            fprintf(stderr, "[sink-diag] nact_gas=%zu rho_max=%.4g (thresh %.4g, ratio %.3g) "
                    "alpha_min=%.4g | dens=%lld tsfr=%lld divv=%lld virial=%lld jeans=%lld "
                    "tidal=%lld densmax=%lld nearsink=%lld PASS=%lld\n",
                    active_gas.size(), rho_max_seen, sim.crit_phys_density,
                    rho_max_seen/sim.crit_phys_density,
                    (alpha_min_seen>1e299 ? -1.0 : alpha_min_seen),
                    veto[V_DENS], veto[V_TSFR], veto[V_DIVV], veto[V_VIRIAL], veto[V_JEANS],
                    veto[V_TIDAL], veto[V_DENSMAX], veto[V_NEARSINK], veto[V_PASS]);
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
        if (sim.nh_per_code_density > 0) {
            const double nH = sim.rho[last_gas] * sim.nh_per_code_density;
            if (nH > 1e10) cs_sink *= std::pow(nH / 1e10, 0.2);
        }
        sim.sink_radius[last_gas] = std::max(0.79 * sim.P.m[last_gas] * sim.G / (cs_sink*cs_sink),
                                             sim.h[last_gas]);
        printf("shmem-GIZMO: sink radius = %.6g (h was %.6g)\n",
               sim.sink_radius[last_gas], sim.h[last_gas]);
        ++sim.sinks_formed;
        printf("shmem-GIZMO: sink formed from cell %zu (m=%.6g, rho=%.6g); %lld total\n",
               last_gas, sim.P.m[last_gas], sim.rho[last_gas], sim.sinks_formed);
        fflush(stdout);
    }
    // Any conversion PERMUTES the particle arrays, so everything keyed by particle index is stale:
    // the neighbour cache (keyed by position in the active list) and the tree, whose orderbuf and
    // leaf_of still name the pre-swap indices. Both must go.
    if (!candidates.empty()) { sim.ngb_cache.clear(); sim.tree_valid = false; }
}

void compute_initial_state(Sim& sim) {
    const size_t n_part = sim.size();
    sim.last_drift.assign(n_part, sim.clock_ticks);
    sim.mass_initial = 0.0;
    for (size_t i = 0; i < n_part; ++i) sim.mass_initial += sim.P.m[i];
    const size_t n_gas = std::min(sim.n_gas, n_part);
    std::vector<uint32_t> gas_list(n_gas);
    for (size_t i = 0; i < n_gas; ++i) gas_list[i] = (uint32_t)i;
    rebuild_tree(sim);                       // provisional: neighbour search for the h solve
    solve_h_and_volumes(sim, sim.tree, gas_list);
    update_softenings(sim);
    // Rebuild so the node max-softenings (which gate the softening force-open in the walk) see
    // the real per-particle softenings rather than the zeros the driver loaded with -- the t=0
    // potential is computed from this tree.
    rebuild_tree(sim);
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

void print_timebins(const Sim& sim, double systemstep, double time) {
    if (sim.bin.empty()) return;
    const int n_bins = Sim::MAX_BINS + 1;
    std::vector<long long> count(n_bins, 0);
    for (size_t i = 0; i < sim.size(); ++i) count[sim.bin[i]]++;

    // cumulative[b] = particles in bin b and in every SHORTER-step bin (higher index here)
    std::vector<long long> cumulative(n_bins, 0);
    long long running = 0;
    for (int b = n_bins - 1; b >= 0; --b) { running += count[b]; cumulative[b] = running; }

    int longest_active = -1;
    for (int b = 0; b < n_bins; ++b)
        if (count[b] > 0 && (sim.clock_ticks % sim.ticks_in_bin(b)) == 0) { longest_active = b; break; }

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
        if (count[b] == 0) continue;
        frac[b] = weight * avg[b];
        frac_sum += frac[b];
        weight *= 2.0;                     // each deeper bin is visited twice as often
    }
    if (frac_sum > 0) for (int b = 0; b < n_bins; ++b) frac[b] /= frac_sum;

    printf("\nSync-Point %lld, Time: %.16g, Systemstep: %g\n", sim.sync_point, time, systemstep);
    printf("Occupied timebins:  non-cells       cells       dt                cumulative A    avg-time  cpu-frac\n");
    long long total_active = 0;
    for (int b = 0; b < n_bins; ++b) {
        if (count[b] == 0) continue;
        const bool active = (sim.clock_ticks % sim.ticks_in_bin(b)) == 0;
        printf(" %c  bin=%2d       %10lld  %10lld   %16.12f      %10lld %c  %10.2f    %5.1f%%\n",
               active ? 'X' : ' ', b, 0LL, count[b], sim.dt_of_bin(b), cumulative[b],
               (b == longest_active) ? '<' : ' ', avg[b] * 1e3, 100.0 * frac[b]);
        if (active) total_active += count[b];
    }
    printf("               ------------------------\n");
    printf("Total active:    %10lld  %10lld    Sum: %10lld\n\n", 0LL, total_active, total_active);

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
                printf("      deep  i=%7zu h=%.4e rho=%.4e P=%.4e cs=%.3f vsig=%.3f "
                       "h/vsig=%.3e x=(%.3f,%.3f)\n", i, sim.h[i], sim.rho[i], sim.press[i], cs,
                       sim.work.signal_speed[i], sim.h[i]/(sim.work.signal_speed[i]+1e-300),
                       sim.P.x[i], sim.P.y[i]);
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
static double desired_dt(const Sim& sim, size_t i) {
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
    if (sim.gravity_on) {
        const double accel_mag = sim.a_grav[i].norm();
        if (accel_mag > 0) {
            // sqrt(2 eta (KERNEL_CORE_SIZE * eps) / |a|) with KERNEL_CORE_SIZE = 1/2 for the
            // cubic spline: GIZMO measures the softening scale by the kernel CORE, not the full
            // support radius. Omitting the 1/2 made this criterion sqrt(2) too permissive.
            dt = std::min(dt, std::sqrt(sim.eta_grav * sim.P.soft[i] / accel_mag));
        }
        if (sim.tidal_criterion && sim.tidal.size() == sim.size()) {
            // dt = 0.5 sqrt(eta / sqrt(||G T||_F^2 / 6)): recovers sqrt(eta) * t_dyn in a
            // Keplerian potential. Gas additionally floors at its own self-gravity timescale.
            const double tnorm = sim.G * sim.tidal[i].frobenius_norm();
            if (tnorm > 0) {
                double dt_tidal = 0.5 * std::sqrt(sim.eta_grav / (tnorm / std::sqrt(6.0)));
                if (gas)                     // gas additionally floors at its self-gravity time
                    dt_tidal = std::min(dt_tidal,
                                        std::sqrt(sim.eta_grav / (sim.G * sim.rho[i] + 1e-300)));
                dt = std::min(dt, dt_tidal);
            }
        }
    }
    return dt;
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
}

// Saitoh & Makino (2009) timestep limiter. A particle sitting many bins above an active neighbour
// can be overrun by a shock before it ever wakes; force it down to within bin_limit of the
// neighbour. Without this, a strong blast propagates into stale, long-binned material and the
// solution is wrong rather than merely inaccurate.

double mfm_step(Sim& sim, double dt_max) {
    const size_t n_part = sim.size();
    Work& work = sim.work;
    const auto step_start = std::chrono::steady_clock::now();

    // ---- active set ----
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
    const std::vector<uint32_t>& active = sim.active;

    // Hydro passes take only the GAS particles; with the gas-first layout that is the ascending
    // prefix of the (sorted) active list. All-gas sims (n_gas = SIZE_MAX) skip the copy entirely.
    const std::vector<uint32_t>* hydro_actives = &active;
    if (sim.n_gas < n_part) {
        sim.active_gas.assign(active.begin(),
                              std::lower_bound(active.begin(), active.end(),
                                               (uint32_t)sim.n_gas));
        hydro_actives = &sim.active_gas;
    }
    const std::vector<uint32_t>& active_gas = *hydro_actives;

    const bool profile = getenv("SHMEM_PROFILE") != nullptr;
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
    // approximate. Rebuild once the accumulated drift is a noticeable fraction of a typical h,
    // because past that the padded prune starts opening nodes it does not need.
    double typical_h = 0.0;
    if (!sim.h.empty()) {
        // cheap stand-in for the median: h of one active particle, which tracks the resolution of
        // the region actually doing work
        typical_h = sim.h[active.empty() ? 0 : active[0]];
    }
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
    // lock; a particle that was also active last step is already current and this is a no-op.
    // Everyone else is caught up on touch, by the hook below.
    if (sim.sparse_drift) {
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < active.size(); ++k)
            drift_particle_to(sim, active[k], sim.clock_ticks);
    }
    sim.lazy_drift = lazy_drift_hook(sim);
    sim.lazy_drift_on = sim.sparse_drift;
    const Tree& tree = sim.tree;
    const double t_tree = profile ? lap() : 0.0;

    // Actives only. A halo refresh is NOT needed: with rate accumulation an inactive particle is
    // never written to, so the h, V and B it carries are exactly the self-consistent set from its
    // own last sync. Refreshing its neighbours as well would cost a whole extra neighbour-search
    // pass per sync -- and this loop runs the same three passes as the global scheme, so any extra
    // one shows up directly as the individual-timestep scheme being SLOWER than global on a
    // uniform problem where it should merely match it.
    solve_h_and_volumes(sim, tree, active_gas);
    const double t_dens = profile ? lap() : 0.0;

    // Gradients first: they need no dt, and their neighbour loop is where the signal speed
    // comes from.
    gradients(sim, tree, active_gas);
    const double t_grad = profile ? lap() : 0.0;

    // Gravity at the CURRENT positions, one walk per step. Done before dt so the acceleration
    // can constrain it.
    if (sim.gravity_on) compute_gravity(sim, tree, active);
    const double t_grav = profile ? lap() : 0.0;

    // ---- timestep ----
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
        // Wake requests raised by the PREVIOUS sync's flux loop are applied here, before this
        // sync's dt is chosen, so a particle a shock is about to reach has already been moved to a
        // short enough bin. Deferring them out of the flux loop keeps that loop free of writes to
        // sim.bin while other threads are reading it.
        for (const auto& [j, floor_bin] : sim.wake_requests)
            if (sim.bin[j] < floor_bin) sim.bin[j] = std::min(floor_bin, Sim::MAX_BINS);
        sim.wake_requests.clear();
        // An EMPTY active set is not a quiet sync -- it is impossible in a consistent hierarchy
        // (bin 0 aligns whenever anything does), so it means the clock has desynchronised. Left
        // alone it is silent and catastrophic: no particle is kicked, yet the drift pass below
        // still advances everyone, so the run continues with gravity effectively switched off.
        // Fail loudly instead of producing plausible-looking ballistic output.
        if (active.empty()) {
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

    // GRAVITY KICK. a_grav is the acceleration at this sync point, which is BOTH the end of the
    // previous step and the start of this one -- so the half-kick the previous step still owes and
    // this step's opening half-kick use the same acceleration and are applied together. That is
    // what keeps the scheme a proper leapfrog on one tree walk per step rather than two.
    if (sim.gravity_on) {
        sim.pending_half_kick.resize(n_part, 0.0);
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < active.size(); ++k) {
            const uint32_t i = active[k];
            const double kick_dt = sim.pending_half_kick[i] + 0.5 * dt_of[i];
            sim.vx[i] += sim.a_grav[i][0] * kick_dt;
            sim.vy[i] += sim.a_grav[i][1] * kick_dt;
            sim.vz[i] += sim.a_grav[i][2] * kick_dt;
            // what THIS particle will owe when it next becomes active -- half of its OWN step
            sim.pending_half_kick[i] = 0.5 * dt_of[i];
        }
        // The predicted primitives carry velocity, so re-predict after the kick rather than
        // before it; the gradients themselves are unaffected (gravity is smooth on the kernel
        // scale and adds no jump across a face).
    }

    const double t_bins = profile ? lap() : 0.0;

    predict_half(sim, active_gas, dt_of);

    std::vector<double>& dmom_x = sim.dmom_x;  std::vector<double>& dmom_y = sim.dmom_y;
    std::vector<double>& dmom_z = sim.dmom_z;  std::vector<double>& denergy = sim.denergy;
    fluxes(sim, tree, active_gas, dt_of, dmom_x, dmom_y, dmom_z, denergy);
    const double t_flux = profile ? lap() : 0.0;

    // Conserved update. This runs over ALL particles, not just the active ones: an inactive
    // neighbour of an active particle still receives its half of the pair's momentum and energy,
    // and dropping that would break conservation exactly where the timebins meet.
    // Pass 1, over the ACTIVE particles only -- which is now the complete set of particles the
    // flux loop can have touched, since inactive ones are deliberately left alone. Each integrates
    // its OWN accumulated rate over its OWN timestep, which is what makes the scheme independent
    // of any per-pair history.
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < active_gas.size(); ++k) {
        const uint32_t i = active_gas[k];
        const double dt_i = dt_of[i];
        const double mass = sim.P.m[i];
        const Vec3d vel_old{sim.vx[i], sim.vy[i], sim.vz[i]};
        // rate * own dt -- the accumulators hold dP/dt and dE/dt, not amounts
        const Vec3d momentum = vel_old * mass + Vec3d{dmom_x[i], dmom_y[i], dmom_z[i]} * dt_i;
        const double energy = mass * (sim.u[i] + 0.5*vel_old.norm_sq()) + denergy[i] * dt_i;
        dmom_x[i] = 0; dmom_y[i] = 0; dmom_z[i] = 0; denergy[i] = 0;

        const Vec3d vel_new = momentum / mass;
        sim.vx[i] = vel_new[0]; sim.vy[i] = vel_new[1]; sim.vz[i] = vel_new[2];
        sim.u[i] = std::max(energy/mass - 0.5*vel_new.norm_sq(), 1e-30);
        // Pressure follows u immediately. Only ACTIVE particles reach here now, so this is simply
        // keeping a particle's own state self-consistent within its own update. Under a
        // density-driven EOS this instead RESETS u from P(rho) -- the energy equation's answer is
        // discarded on purpose, which is what makes the law stand in for cooling.
        eos_apply(sim, i);

        // TIME-CENTRING correction. Pass 2 below drifts everything with the POST-flux velocity,
        // which alone is backward Euler on position -- it produced a clean systematic phase lag in
        // the soundwave and capped convergence at ~order 1. Pre-subtracting half the velocity
        // change makes the net displacement (v_old + v_new)/2 * dt exactly, while keeping the
        // full-N pass free of any flux data.
        const Vec3d correction = (vel_new - vel_old) * (0.5 * dt);
        sim.P.x[i] -= correction[0]; sim.P.y[i] -= correction[1]; sim.P.z[i] -= correction[2];
    }

    // Sink formation, once the cells' own updates for this step are complete. Serial and after
    // the flux pass because it reorders the particle arrays.
    sink_formation_pass(sim, active_gas, dt_of);
    sink_accretion_pass(sim);

    // Pass 2: drift EVERY particle over the system interval dt. A long-binned particle is drifted
    // in several sub-steps rather than one long one; its velocity is constant between its own
    // kicks, so the sub-steps sum to the same displacement, while any momentum it picked up as an
    // inactive neighbour takes effect immediately.
    // A wrapping particle needs no special handling: ngb_search prunes on the MIN-IMAGE distance to
    // a node's centre of mass, so a leaf sitting at x ~ box is min-image-adjacent to a query at
    // x ~ 0 and still gets opened, and the leaf test then uses the particle's live folded position.
    // Only the drift magnitude has to be tracked, and a wrap does not change that.
    // Compare SQUARED displacements and take the one square root at the end: this loop is over
    // every particle every sync, and a per-particle sqrt bought nothing but a rank ordering that
    // squaring already preserves.
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
    // Feed this step's velocity changes into the per-node bound -- GIZMO's force_kick_node
    // (forcetree_update.cc:78), climbing the parent chain with a max. Only ACTIVE particles can
    // have changed velocity (the gravity kick and the conserved update both cover actives only),
    // and the climb stops at the first ancestor that already covers the speed, so in steady state
    // this is one relaxed load per active particle.
    if (sim.tree_valid && !sim.tree.leaf_of.empty()) {
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < active.size(); ++k) {
            const uint32_t i = active[k];
            const int leaf = sim.tree.leaf_of[i];
            if (leaf >= 0)
                sim.tree.raise_vmax(
                    leaf, (float)Vec3d{sim.vx[i], sim.vy[i], sim.vz[i]}.norm());
        }
    }
    // Motion since the build is bounded per node by Tree::vmax * this elapsed time; see the prune
    // in ngb_search. A wrapping particle needs no special handling: the prune is on the MIN-IMAGE
    // distance to a node's centre of mass, so a leaf at x ~ box is min-image-adjacent to a query
    // at x ~ 0 and still gets opened, and the leaf test then reads the live folded position.
    sim.tree.t_since_build += dt;

    if (profile) {
        const double t_drift = lap();
        // Cumulative totals as well as the per-step lines: the per-step view is dominated by the
        // opening all-active step, but a run's cost is dominated by the many cheap deep-bin steps
        // after it, where the O(N) drift is most of the work. Only the totals answer "what would
        // fixing this phase actually buy".
        static double c_tree=0, c_dens=0, c_grad=0, c_grav=0, c_bins=0, c_flux=0, c_drift=0;
        static long long c_steps = 0;
        c_tree+=t_tree; c_dens+=t_dens; c_grad+=t_grad; c_grav+=t_grav;
        c_bins+=t_bins; c_flux+=t_flux; c_drift+=t_drift; ++c_steps;
        const double tot = c_tree+c_dens+c_grad+c_grav+c_bins+c_flux+c_drift;
        if (getenv("SHMEM_PROFILE_TOTALS") && tot > 0 && (c_steps % 100) == 0) {
            fprintf(stderr, "[prof-total] %lld steps  tree=%.1f dens=%.1f grad=%.1f grav=%.1f "
                            "bins=%.1f flux=%.1f DRIFT=%.1f s  (drift %.1f%% of profiled time)\n",
                    c_steps, c_tree*1e-3, c_dens*1e-3, c_grad*1e-3, c_grav*1e-3,
                    c_bins*1e-3, c_flux*1e-3, c_drift*1e-3, 100.0*c_drift/tot);
        }
        static int shown = 0;
        if (shown < 8) {
            ++shown;
            fprintf(stderr, "[prof] nact=%zu/%zu  tree=%.1f dens=%.1f grad=%.1f grav=%.1f "
                            "bins=%.1f flux=%.1f drift=%.1f  (ms)\n",
                    active.size(), n_part, t_tree, t_dens, t_grad, t_grav,
                    t_bins, t_flux, t_drift);
        }
    }

    // Charge this sync to its LONGEST-step active bin: that bin is why the step had to do as much
    // work as it did, and it is the key the cpu-frac column is normalised against.
    {
        sim.bin_cpu_sum.resize(Sim::MAX_BINS + 1, 0.0);
        sim.bin_cpu_n.resize(Sim::MAX_BINS + 1, 0);
        int longest_active = 0;
        if (sim.individual_timesteps && !active.empty()) {
            longest_active = Sim::MAX_BINS;
            for (uint32_t i : active) longest_active = std::min(longest_active, sim.bin[i]);
        }
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
        sim.bin_cpu_sum[longest_active] += elapsed;
        sim.bin_cpu_n[longest_active]   += 1;
    }
    ++sim.sync_point;

    // Advance the integer clock by exactly the ticks this step covered. No rounding and no
    // minimum: step_ticks was derived FROM the tick grid above, so the clock and the dt every
    // particle was integrated over agree exactly, and the hierarchy stays aligned indefinitely.
    if (sim.individual_timesteps) sim.clock_ticks += step_ticks;
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
