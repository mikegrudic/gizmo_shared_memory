#include "mfm.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace shmem {

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
[[nodiscard]] static ContactState solve_hllc_contact(
        double density_left,  double vnorm_left,  double pressure_left,
        double density_right, double vnorm_right, double pressure_right, double gamma) {
    const double csound_left  = std::sqrt(gamma * pressure_left  / density_left);
    const double csound_right = std::sqrt(gamma * pressure_right / density_right);
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

    std::vector<double> h_guess(active.size());
    const bool have_guess = !sim.h.empty();
    for (size_t k = 0; k < active.size(); ++k) h_guess[k] = have_guess ? sim.h[active[k]] : 0.0;
    if (!have_guess || h_guess.empty() || h_guess[0] <= 0) h_guess.clear();

    const DensityResult solved =
        density(tree, sim.P, active, sim.des_ngb, h_guess, sim.box, sim.dim);
    for (size_t k = 0; k < active.size(); ++k) sim.h[active[k]] = solved.h[k];

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
            neighbours.clear();
            ngb_search(tree, sim.P, pos_i, sim.h[i], neighbours, sim.box);
            double weight_sum = 0, dn_dh = 0, dphi_dh_sum = 0;
            for (uint32_t j : neighbours) {
                const Vec3d offset = min_image(sim.P.pos(j) - pos_i, sim.box);
                const double r = offset.norm();
                weight_sum += kernel_w(r, sim.h[i], sim.dim);
                // both sums INCLUDE the self term (r = 0), as GIZMO's do
                dn_dh += kernel_dwdh(r, sim.h[i], sim.dim);
                if (want_zeta) dphi_dh_sum += sim.P.m[j] * grav_dphi_dh(r, sim.h[i]);
            }
            sim.ninv[i]  = 1.0 / weight_sum;             // V_i: MFM volume from partition of unity
            sim.rho[i]   = sim.P.m[i] * weight_sum;      // rho_i = m_i / V_i
            sim.press[i] = (sim.gamma - 1.0) * sim.rho[i] * sim.u[i];
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
            neighbours.clear();
            ngb_search(tree, sim.P, pos_i, sim.h[i], neighbours, sim.box);

            SymTensor3d moments{0,0,0,0,0,0};     // E_i = sum_j (dx ox dx) W_ij
            // Signal speed, Monaghan (1997): c_i + c_j minus the APPROACH speed along the pair
            // axis. Built only from RELATIVE velocities, so a uniform boost of the whole domain
            // leaves it unchanged -- the square test advects at |v|~1300 and a lab-frame |v| here
            // would shrink dt by ~1700x for a flow that is trivially Galilean-equivalent to rest.
            const double csound_i = std::sqrt(sim.gamma * sim.press[i] / sim.rho[i]);
            double signal_speed = 2.0 * csound_i;   // floor: the i==j / no-neighbour case
            for (uint32_t j : neighbours) {
                const Vec3d offset = min_image(sim.P.pos(j) - pos_i, sim.box);
                const double separation = offset.norm();
                const double weight = kernel_w(separation, sim.h[i], sim.dim);
                moments += outer_product(offset) * weight;
                if (separation > 0) {
                    const Vec3d rel_vel = vel_i - Vec3d{sim.vx[j], sim.vy[j], sim.vz[j]};
                    const double approach_speed = dot(rel_vel, offset) / separation;
                    const double csound_j = std::sqrt(sim.gamma * sim.press[j] / sim.rho[j]);
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
static void compute_gravity(Sim& sim, const Tree& tree, const std::vector<uint32_t>& active) {
    const size_t n_part = sim.size();
    sim.P.soft.resize(n_part);
    // Softening is a property of every particle, active or not: the tree's mass distribution is
    // sourced by ALL of them, so this loop stays global even though the forces below do not.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_part; ++i)
        sim.P.soft[i] = sim.adaptive_soft ? std::max(sim.h[i], sim.soft_min) : sim.soft_min;

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
        aold_active.resize(active.size());
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < active.size(); ++k)
            aold_active[k] = sim.err_tol_force_acc * sim.a_grav[active[k]].norm() / sim.G;
        aold_ptr = aold_active.data();
    }
    accel_grouped(tree, sim.P, active, sim.theta, sim.G, 8, ax, ay, az, tidal_out, aold_ptr);
    if (sim.tidal_criterion) {
        sim.tidal.resize(n_part);
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < active.size(); ++k) sim.tidal[active[k]] = tidal_active[k];
    }

    sim.a_grav.resize(n_part);
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < active.size(); ++k)
        sim.a_grav[active[k]] = Vec3d{ax[k], ay[k], az[k]};
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
            neighbours.clear();
            ngb_search(tree, sim.P, pos_i, sim.h[i], neighbours, sim.box);
            for (uint32_t j : neighbours) {
                if (j == i) continue;
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
                const auto [contact_speed, contact_pressure] = solve_hllc_contact(
                    left[FIELD_DENSITY],  vnorm_left,  left[FIELD_PRESSURE],
                    right[FIELD_DENSITY], vnorm_right, right[FIELD_PRESSURE], sim.gamma);

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
                    const double cs_i = std::sqrt(sim.gamma * sim.press[i] / sim.rho[i]);
                    const double cs_j = std::sqrt(sim.gamma * sim.press[j] / sim.rho[j]);
                    const double sm_over_c = std::abs(contact_speed) / std::min(cs_i, cs_j);
                    if (sm_over_c < eps_big) {
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

void compute_initial_state(Sim& sim) {
    const size_t n_part = sim.size();
    std::vector<uint32_t> all_particles(n_part);
    for (size_t i = 0; i < n_part; ++i) all_particles[i] = (uint32_t)i;
    sim.tree = build(sim.P);
    sim.tree_valid = true;
    sim.drift_since_build = 0.0;
    solve_h_and_volumes(sim, sim.tree, all_particles);
}

void compute_potential(Sim& sim) {
    const size_t n_part = sim.size();
    sim.phi.assign(n_part, 0.0);
    if (!sim.gravity_on) return;
    if (!sim.tree_valid || sim.tree.nnodes() == 0) {
        sim.tree = build(sim.P);
        sim.tree_valid = true;
        sim.drift_since_build = 0.0;
    }
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
                const double cs = std::sqrt(sim.gamma * sim.press[i] / sim.rho[i]);
                printf("      deep  i=%7zu h=%.4e rho=%.4e P=%.4e cs=%.3f vsig=%.3f "
                       "h/vsig=%.3e x=(%.3f,%.3f)\n", i, sim.h[i], sim.rho[i], sim.press[i], cs,
                       sim.work.signal_speed[i], sim.h[i]/(sim.work.signal_speed[i]+1e-300),
                       sim.P.x[i], sim.P.y[i]);
            }
            shown = 0;
            for (size_t i = 0; i < sim.size() && shown < 2; ++i) {
                if (sim.bin[i] != shallowest) continue;
                ++shown;
                const double cs = std::sqrt(sim.gamma * sim.press[i] / sim.rho[i]);
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
    double dt = 2.0 * sim.cfl * std::pow(sim.ninv[i], 1.0 / sim.dim)
                / (sim.work.signal_speed[i] + 1e-300);
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
    const bool must_rebuild = !sim.tree_valid || sim.tree.nnodes() == 0 ||
                              typical_h <= 0.0 ||
                              sim.drift_since_build > sim.tree_rebuild_pad_frac * typical_h;
    if (must_rebuild) {
        sim.tree = build(sim.P);
        sim.tree_valid = true;
        sim.drift_since_build = 0.0;
        ++sim.tree_builds;
    }
    sim.tree.pad = sim.drift_since_build;
    const Tree& tree = sim.tree;
    const double t_tree = profile ? lap() : 0.0;

    // Actives only. A halo refresh is NOT needed: with rate accumulation an inactive particle is
    // never written to, so the h, V and B it carries are exactly the self-consistent set from its
    // own last sync. Refreshing its neighbours as well would cost a whole extra neighbour-search
    // pass per sync -- and this loop runs the same three passes as the global scheme, so any extra
    // one shows up directly as the individual-timestep scheme being SLOWER than global on a
    // uniform problem where it should merely match it.
    solve_h_and_volumes(sim, tree, active);
    const double t_dens = profile ? lap() : 0.0;

    // Gradients first: they need no dt, and their neighbour loop is where the signal speed
    // comes from.
    gradients(sim, tree, active);
    const double t_grad = profile ? lap() : 0.0;

    // Gravity at the CURRENT positions, one walk per step. Done before dt so the acceleration
    // can constrain it.
    if (sim.gravity_on) compute_gravity(sim, tree, active);
    const double t_grav = profile ? lap() : 0.0;

    // ---- timestep ----
    std::vector<double>& dt_of = sim.dt_of;      // reused: allocating N doubles per sync is not free
    dt_of.resize(n_part);
    double dt;                                   // the interval this call advances the system by
    if (sim.individual_timesteps) {
        assign_bins(sim, active);
        // Wake requests raised by the PREVIOUS sync's flux loop are applied here, before this
        // sync's dt is chosen, so a particle a shock is about to reach has already been moved to a
        // short enough bin. Deferring them out of the flux loop keeps that loop free of writes to
        // sim.bin while other threads are reading it.
        for (const auto& [j, floor_bin] : sim.wake_requests)
            if (sim.bin[j] < floor_bin) sim.bin[j] = std::min(floor_bin, Sim::MAX_BINS);
        sim.wake_requests.clear();
        // dt_max can be shorter than the bin-0 step near a snapshot boundary; cap everyone.
        int deepest_active = 0;
        #pragma omp parallel for schedule(static) reduction(max:deepest_active)
        for (size_t k = 0; k < active.size(); ++k)
            deepest_active = std::max(deepest_active, sim.bin[active[k]]);
        // also O(N) every sync, so also parallel -- and dt_of_bin divides, which is not free
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_part; ++i) dt_of[i] = std::min(sim.dt_of_bin(sim.bin[i]), dt_max);
        dt = std::min(sim.dt_of_bin(deepest_active), dt_max);
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

    predict_half(sim, active, dt_of);

    std::vector<double>& dmom_x = sim.dmom_x;  std::vector<double>& dmom_y = sim.dmom_y;
    std::vector<double>& dmom_z = sim.dmom_z;  std::vector<double>& denergy = sim.denergy;
    fluxes(sim, tree, active, dt_of, dmom_x, dmom_y, dmom_z, denergy);
    const double t_flux = profile ? lap() : 0.0;

    // Conserved update. This runs over ALL particles, not just the active ones: an inactive
    // neighbour of an active particle still receives its half of the pair's momentum and energy,
    // and dropping that would break conservation exactly where the timebins meet.
    // Pass 1, over the ACTIVE particles only -- which is now the complete set of particles the
    // flux loop can have touched, since inactive ones are deliberately left alone. Each integrates
    // its OWN accumulated rate over its OWN timestep, which is what makes the scheme independent
    // of any per-pair history.
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < active.size(); ++k) {
        const uint32_t i = active[k];
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
        // keeping a particle's own state self-consistent within its own update.
        sim.press[i] = (sim.gamma - 1.0) * sim.rho[i] * sim.u[i];

        // TIME-CENTRING correction. Pass 2 below drifts everything with the POST-flux velocity,
        // which alone is backward Euler on position -- it produced a clean systematic phase lag in
        // the soundwave and capped convergence at ~order 1. Pre-subtracting half the velocity
        // change makes the net displacement (v_old + v_new)/2 * dt exactly, while keeping the
        // full-N pass free of any flux data.
        const Vec3d correction = (vel_new - vel_old) * (0.5 * dt);
        sim.P.x[i] -= correction[0]; sim.P.y[i] -= correction[1]; sim.P.z[i] -= correction[2];
    }

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
    double max_shift_sq = 0.0;
    #pragma omp parallel for schedule(static) reduction(max:max_shift_sq)
    for (size_t i = 0; i < n_part; ++i) {
        const Vec3d shift = Vec3d{sim.vx[i], sim.vy[i], sim.vz[i]} * dt;
        max_shift_sq = std::max(max_shift_sq, shift.norm_sq());
        Vec3d pos_new = sim.P.pos(i) + shift;
        if (sim.box > 0) pos_new = fold_into_box(pos_new, sim.box);
        sim.P.x[i] = pos_new[0]; sim.P.y[i] = pos_new[1]; sim.P.z[i] = pos_new[2];

        // Drift-time prediction for INACTIVE particles (GIZMO's core/predict.cc): between its own
        // updates a particle's density evolves continuously as its neighbourhood converges or
        // expands, rho_dot = -rho div v. Without this, a long-binned particle in a steadily
        // converging flow carries a systematically LOW density until it next activates -- in noh
        // the cold supersonic inflow sat 25% under the analytic pre-shock profile with the
        // velocities EXACT, because only the density estimate was stale. The kernel radius
        // follows with the opposite sign (h ~ n^{-1/dim}) and pressure tracks rho at fixed u.
        // Clamped at +-0.3 per drift as GIZMO clamps it; cheap 2nd-order exp for the tiny
        // arguments this almost always sees.
        if (sim.individual_timesteps && !sim.is_active(i)) {
            double divv_fac = work.div_vel[i] * dt;
            if (divv_fac >  0.3) divv_fac =  0.3;
            if (divv_fac < -0.3) divv_fac = -0.3;
            if (divv_fac != 0.0) {
                const double x = -divv_fac;
                const double f = (std::abs(x) < 0.05) ? 1.0 + x*(1.0 + 0.5*x) : std::exp(x);
                sim.rho[i]  *= f;
                sim.ninv[i] /= f;
                sim.press[i] = (sim.gamma - 1.0) * sim.rho[i] * sim.u[i];
                sim.h[i]    *= (std::abs(divv_fac) < 0.15)
                               ? 1.0 + divv_fac/sim.dim + 0.5*(divv_fac/sim.dim)*(divv_fac/sim.dim)
                               : std::exp(divv_fac / sim.dim);
                work.predicted[FIELD_DENSITY][i]  = sim.rho[i];
                work.predicted[FIELD_PRESSURE][i] = sim.press[i];
            }
        }
    }
    // Accumulated displacement is what Tree::pad must cover for the reused tree to stay exact.
    sim.drift_since_build += std::sqrt(max_shift_sq);

    if (profile) {
        const double t_drift = lap();
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

    if (sim.individual_timesteps) {
        // advance the integer clock by the interval just taken
        long long ticks = (long long)std::llround(dt / sim.dt_base * (double)(1LL << Sim::MAX_BINS));
        if (ticks < 1) ticks = 1;
        sim.clock_ticks += ticks;
    }
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
        ngb_search(tree, sim.P, pos_i, search_radius, neighbours, sim.box);

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
