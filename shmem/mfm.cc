#include "mfm.h"
#include <array>
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

// The five primitive fields, in the one order used by gradients, reconstruction and the Riemann
// state vectors. Named rather than bare 0..4: the flux loop indexes these arrays a dozen times and
// a transposed velocity component is otherwise invisible.
enum PrimitiveField : int { FIELD_DENSITY = 0, FIELD_VX, FIELD_VY, FIELD_VZ, FIELD_PRESSURE,
                            NUM_FIELDS };
using PrimitiveState = std::array<double, NUM_FIELDS>;

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

// per-step scratch shared by the phases
struct Work {
    std::vector<Mat3d> moments_inv;                            // E^-1, one per particle
    std::array<std::vector<Vec3d>, NUM_FIELDS> gradient;       // gradient of each primitive field
    std::array<std::vector<double>, NUM_FIELDS> predicted;     // half-step predicted primitives
    std::vector<double> signal_speed;                          // Monaghan signal speed, per particle
};

static void solve_h_and_volumes(Sim& sim, const Tree& tree) {
    const size_t n_part = sim.size();
    std::vector<uint32_t> all_particles(n_part);
    for (size_t i = 0; i < n_part; ++i) all_particles[i] = (uint32_t)i;
    std::vector<double> h_guess = sim.h;        // warm start from last step if present
    if (h_guess.size() != n_part) h_guess.clear();
    const DensityResult solved =
        density(tree, sim.P, all_particles, sim.des_ngb, h_guess, sim.box, sim.dim);
    sim.h = solved.h;
    sim.ninv.assign(n_part, 0.0); sim.rho.assign(n_part, 0.0); sim.press.assign(n_part, 0.0);
    #pragma omp parallel
    {
        std::vector<uint32_t> neighbours;
        #pragma omp for schedule(dynamic, 64)
        for (size_t i = 0; i < n_part; ++i) {
            const Vec3d pos_i = sim.P.pos(i);
            neighbours.clear();
            ngb_search(tree, sim.P, pos_i, sim.h[i], neighbours, sim.box);
            double weight_sum = 0;
            for (uint32_t j : neighbours) {
                const Vec3d offset = min_image(sim.P.pos(j) - pos_i, sim.box);
                weight_sum += kernel_w(offset.norm(), sim.h[i], sim.dim);
            }
            sim.ninv[i]  = 1.0 / weight_sum;             // V_i: MFM volume from partition of unity
            sim.rho[i]   = sim.P.m[i] * weight_sum;      // rho_i = m_i / V_i
            sim.press[i] = (sim.gamma - 1.0) * sim.rho[i] * sim.u[i];
        }
    }
}

static void gradients(Sim& sim, const Tree& tree, Work& work) {
    const size_t n_part = sim.size();
    work.moments_inv.assign(n_part, Mat3d{});
    for (auto& field_gradient : work.gradient) field_gradient.assign(n_part, Vec3d{});
    work.signal_speed.assign(n_part, 0.0);
    #pragma omp parallel
    {
        std::vector<uint32_t> neighbours;
        #pragma omp for schedule(dynamic, 64)
        for (size_t i = 0; i < n_part; ++i) {
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
            // already zeroed on failure: that is the zero-gradient fallback
            if (!invert_moments(moments, moments_inv, sim.dim)) continue;

            const PrimitiveState field_i{sim.rho[i], sim.vx[i], sim.vy[i], sim.vz[i], sim.press[i]};
            std::array<Vec3d, NUM_FIELDS> weighted_diff_sum{};
            for (uint32_t j : neighbours) {
                const Vec3d offset = min_image(sim.P.pos(j) - pos_i, sim.box);
                const double weight = kernel_w(offset.norm(), sim.h[i], sim.dim);
                const PrimitiveState field_j{sim.rho[j], sim.vx[j], sim.vy[j], sim.vz[j],
                                             sim.press[j]};
                for (int f = 0; f < NUM_FIELDS; ++f)
                    weighted_diff_sum[f] += offset * ((field_j[f] - field_i[f]) * weight);
            }
            for (int f = 0; f < NUM_FIELDS; ++f)
                work.gradient[f][i] = moments_inv.matvec(weighted_diff_sum[f]);
        }
    }
}

// Lagrangian half-step prediction of primitives (MUSCL-Hancock predictor).
static void predict_half(Sim& sim, Work& work, double dt) {
    const size_t n_part = sim.size();
    for (auto& field : work.predicted) field.resize(n_part);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_part; ++i) {
        const double div_vel = work.gradient[FIELD_VX][i][0]
                             + work.gradient[FIELD_VY][i][1]
                             + work.gradient[FIELD_VZ][i][2];
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

// Flux exchange over unique pairs. Pair discovery from the SMALLER kernel side would miss
// asymmetric pairs, so: i owns the pair when (i < j) and r < max(h_i, h_j); every pair is then
// found exactly once because both sides search with max(h_i, h_j) coverage via the union below.
static void fluxes(Sim& sim, const Tree& tree, Work& work, double dt,
                   std::vector<double>& dmom_x, std::vector<double>& dmom_y,
                   std::vector<double>& dmom_z, std::vector<double>& denergy) {
    const size_t n_part = sim.size();
    dmom_x.assign(n_part, 0); dmom_y.assign(n_part, 0);
    dmom_z.assign(n_part, 0); denergy.assign(n_part, 0);
    #pragma omp parallel
    {
        std::vector<uint32_t> neighbours;
        #pragma omp for schedule(dynamic, 64)
        for (size_t i = 0; i < n_part; ++i) {
            // search with h_i; pairs where h_j > r >= h_i are found from j's side (j also loops)
            const Vec3d pos_i = sim.P.pos(i);
            neighbours.clear();
            ngb_search(tree, sim.P, pos_i, sim.h[i], neighbours, sim.box);
            for (uint32_t j : neighbours) {
                if (j == i) continue;
                const Vec3d offset = min_image(sim.P.pos(j) - pos_i, sim.box);
                const double separation = offset.norm();
                if (separation <= 0) continue;
                // Each pair must be processed EXACTLY once, including asymmetric ones (r < h_i but
                // r >= h_j, so only i sees j). Ownership: the lower index owns the pair IF it can
                // see it; otherwise the higher index picks it up. A plain j<=i skip drops the
                // asymmetric pairs that only the larger-h side can see.
                if (j < i && separation < sim.h[j]) continue;
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
                PrimitiveState left{}, right{};
                for (int f = 0; f < NUM_FIELDS; ++f) {
                    const auto& predicted = work.predicted[f];
                    const double jump = predicted[j] - predicted[i];
                    left[f]  = predicted[i] + minmod(extrapolate(f, i, +1.0),  0.5*jump);
                    right[f] = predicted[j] + minmod(extrapolate(f, j, -1.0), -0.5*jump);
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
                const double face_speed_lab = dot(face_vel, normal) + contact_speed;
                const Vec3d momentum_flux = normal * (contact_pressure * face_area * dt);
                const double energy_flux = contact_pressure * face_speed_lab * face_area * dt;
                #pragma omp atomic
                dmom_x[i] -= momentum_flux[0];
                #pragma omp atomic
                dmom_y[i] -= momentum_flux[1];
                #pragma omp atomic
                dmom_z[i] -= momentum_flux[2];
                #pragma omp atomic
                denergy[i] -= energy_flux;
                #pragma omp atomic
                dmom_x[j] += momentum_flux[0];
                #pragma omp atomic
                dmom_y[j] += momentum_flux[1];
                #pragma omp atomic
                dmom_z[j] += momentum_flux[2];
                #pragma omp atomic
                denergy[j] += energy_flux;
            }
        }
    }
}

double mfm_step(Sim& sim, double dt_max) {
    const size_t n_part = sim.size();
    Tree tree = build(sim.P);
    solve_h_and_volumes(sim, tree);

    // Gradients first: they need no dt, and their neighbour loop is where the signal speed
    // comes from.
    Work work;
    gradients(sim, tree, work);

    // CFL from the Galilean-invariant signal speed (see gradients()).
    double dt = dt_max;
    #pragma omp parallel for reduction(min:dt) schedule(static)
    for (size_t i = 0; i < n_part; ++i) {
        dt = std::min(dt, sim.cfl * sim.h[i] / (work.signal_speed[i] + 1e-300));
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

    predict_half(sim, work, dt);

    std::vector<double> dmom_x, dmom_y, dmom_z, denergy;
    fluxes(sim, tree, work, dt, dmom_x, dmom_y, dmom_z, denergy);

    // conserved update + Lagrangian drift
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_part; ++i) {
        const double mass = sim.P.m[i];
        const Vec3d vel_old{sim.vx[i], sim.vy[i], sim.vz[i]};
        const Vec3d momentum = vel_old * mass + Vec3d{dmom_x[i], dmom_y[i], dmom_z[i]};
        const double energy = mass * (sim.u[i] + 0.5*vel_old.norm_sq()) + denergy[i];

        const Vec3d vel_new = momentum / mass;
        sim.vx[i] = vel_new[0]; sim.vy[i] = vel_new[1]; sim.vz[i] = vel_new[2];
        sim.u[i] = std::max(energy/mass - 0.5*vel_new.norm_sq(), 1e-30);

        // TIME-CENTRED drift: x += (v_old + v_new)/2 dt. Drifting with the post-kick velocity is
        // backward Euler on position -- it produced a clean systematic PHASE LAG in the soundwave
        // (visible in plots/soundwave.png as the whole wave trailing the exact curve) and capped
        // convergence at ~order 1. The average recovers the second-order leapfrog phase behaviour.
        Vec3d pos_new = sim.P.pos(i) + (vel_old + vel_new) * (0.5 * dt);
        if (sim.box > 0) pos_new = fold_into_box(pos_new, sim.box);
        sim.P.x[i] = pos_new[0]; sim.P.y[i] = pos_new[1]; sim.P.z[i] = pos_new[2];
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
    solve_h_and_volumes(sim, tree);

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

    Work work;
    gradients(sim, tree, work);

    double worst_error = 0;
    for (size_t i = 0; i < sim.size(); ++i) {
        // A particle whose moments matrix was singular gets the zero-gradient fallback by design;
        // that is a separate (geometric) condition, so do not score it as an algebra failure.
        if (work.moments_inv[i].frobenius_norm_sq() == 0.0) continue;
        for (int f = 0; f < NUM_FIELDS; ++f) {
            const double scale = exact_gradient[f].norm();
            if (scale <= 0) continue;
            worst_error = std::max(worst_error,
                                   (work.gradient[f][i] - exact_gradient[f]).norm() / scale);
        }
    }
    return worst_error;
}

double face_closure(Sim& sim, int nsample) {
    Tree tree = build(sim.P);
    solve_h_and_volumes(sim, tree);
    Work work;
    gradients(sim, tree, work);

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
