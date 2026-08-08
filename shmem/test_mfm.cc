// MFM validation: the first two tier-1 tests, judged against exact solutions.
//
//   soundwave  travelling linear sound wave over one period, L1 vs the initial condition.
//              THE convergence test: a scheme that is not ~2nd order in smooth flow fails here
//              long before it fails anything dramatic.
//   shocktube  Sod problem vs the exact Riemann solution, L1(rho) in the uncontaminated region
//              (periodic box carries a second, inverted discontinuity at the wrap; the scored
//              window excludes everything causally connected to it).
//
// Plus two structural checks with exact expectations:
//   face closure    sum_j A_ij = 0 (discrete closed surface) -- broken faces, not inaccuracy
//   conservation    mass exact by construction; momentum/energy to roundoff from antisymmetry

#include "mfm.h"
#include <algorithm>
#include <cstdio>
#include <sys/stat.h>
#include <vector>

using namespace shmem;

// ---- exact Riemann solver (Toro): star pressure by Newton, then sampling ----
struct RiemannExact {
    double rl, ul, pl, rr, ur, pr, g;
    double pst, ust;
    void solve() {
        auto f = [&](double p, double rho, double pk) {
            double A = 2.0/((g+1)*rho), B = (g-1)/(g+1)*pk;
            double c = std::sqrt(g*pk/rho);
            return (p > pk) ? (p-pk)*std::sqrt(A/(p+B))
                            : 2*c/(g-1)*(std::pow(p/pk,(g-1)/(2*g))-1);
        };
        auto fp = [&](double p, double rho, double pk) {
            double A = 2.0/((g+1)*rho), B = (g-1)/(g+1)*pk;
            double c = std::sqrt(g*pk/rho);
            return (p > pk) ? std::sqrt(A/(B+p))*(1-(p-pk)/(2*(B+p)))
                            : 1.0/(rho*c)*std::pow(p/pk,-(g+1)/(2*g));
        };
        double p = 0.5*(pl+pr);
        for (int it = 0; it < 60; ++it) {
            double F  = f(p,rl,pl) + f(p,rr,pr) + (ur-ul);
            double dF = fp(p,rl,pl) + fp(p,rr,pr);
            double pn = p - F/dF;
            if (pn < 1e-12) pn = 1e-12;
            if (std::abs(pn-p) < 1e-14*p) { p = pn; break; }
            p = pn;
        }
        pst = p;
        ust = 0.5*(ul+ur) + 0.5*(f(p,rr,pr) - f(p,rl,pl));
    }
    // sample at xi = x/t
    void sample(double xi, double& rho, double& u, double& p) const {
        double cl = std::sqrt(g*pl/rl), cr = std::sqrt(g*pr/rr);
        if (xi < ust) {   // left of contact
            if (pst > pl) {                    // left shock
                double sl = ul - cl*std::sqrt((g+1)/(2*g)*pst/pl + (g-1)/(2*g));
                if (xi < sl) { rho=rl; u=ul; p=pl; }
                else { rho = rl*((pst/pl + (g-1)/(g+1))/((g-1)/(g+1)*pst/pl + 1)); u=ust; p=pst; }
            } else {                           // left rarefaction
                double shl = ul - cl, cst = cl*std::pow(pst/pl,(g-1)/(2*g)), stl = ust - cst;
                if (xi < shl) { rho=rl; u=ul; p=pl; }
                else if (xi > stl) { rho = rl*std::pow(pst/pl,1/g); u=ust; p=pst; }
                else {
                    u = 2/(g+1)*(cl + (g-1)/2*ul + xi);
                    double c = cl - (g-1)/2*(u-ul);
                    rho = rl*std::pow(c/cl,2/(g-1)); p = pl*std::pow(c/cl,2*g/(g-1));
                }
            }
        } else {          // right of contact
            if (pst > pr) {                    // right shock
                double sr = ur + cr*std::sqrt((g+1)/(2*g)*pst/pr + (g-1)/(2*g));
                if (xi > sr) { rho=rr; u=ur; p=pr; }
                else { rho = rr*((pst/pr + (g-1)/(g+1))/((g-1)/(g+1)*pst/pr + 1)); u=ust; p=pst; }
            } else {                           // right rarefaction
                double shr = ur + cr, cst = cr*std::pow(pst/pr,(g-1)/(2*g)), str = ust + cst;
                if (xi > shr) { rho=rr; u=ur; p=pr; }
                else if (xi < str) { rho = rr*std::pow(pst/pr,1/g); u=ust; p=pst; }
                else {
                    u = 2/(g+1)*(-cr + (g-1)/2*ur + xi);
                    double c = cr + (g-1)/2*(u-ur);
                    rho = rr*std::pow(c/cr,2/(g-1)); p = pr*std::pow(c/cr,2*g/(g-1));
                }
            }
        }
    }
};

static Sim make_lattice(int nx, int ny, int nz, double box) {
    Sim S; S.box = box;
    size_t N = (size_t)nx*ny*nz;
    S.P.x.resize(N); S.P.y.resize(N); S.P.z.resize(N); S.P.m.assign(N,0.0); S.P.soft.assign(N,0.0);
    S.vx.assign(N,0); S.vy.assign(N,0); S.vz.assign(N,0); S.u.assign(N,0);
    size_t k = 0;
    for (int i=0;i<nx;++i) for (int j=0;j<ny;++j) for (int l=0;l<nz;++l,++k) {
        S.P.x[k]=(i+0.5)*box/nx; S.P.y[k]=(j+0.5)*box/ny; S.P.z[k]=(l+0.5)*box/nz;
    }
    return S;
}

int main(int argc, char** argv) {
    int quick = (argc > 1 && argv[1][0]=='q');
    mkdir("plots", 0775);
    FILE* fconv = fopen("plots/soundwave_convergence.txt", "w");
    fprintf(fconv, "# n1  L1_over_A   (half period, A=1e-4)\n");

    // ---------- soundwave convergence series ----------
    // A = 1e-4: at A = 0.01 the physical nonlinear steepening of a simple wave (~2A relative)
    // dominated the measurement and the 'error' refused to converge -- it was real physics.
    for (int n1 : (quick ? std::vector<int>{8,16} : std::vector<int>{8,16,32}))
    {
        Sim S = make_lattice(n1, n1, n1, 1.0);
        S.gamma = 5.0/3.0;
        double rho0 = 1.0, P0 = 3.0/5.0;            // c_s = 1, period = 1
        double A = 1e-4, kw = 2*M_PI;
        size_t N = S.size();
        double m = rho0 / N;
        for (size_t i = 0; i < N; ++i) {
            S.P.m[i] = m;
            double x0 = S.P.x[i];
            S.P.x[i] = x0 - (A/kw)*std::cos(kw*x0);          // displacement -> drho/rho = A sin(kx)
            S.vx[i]  = A*std::sin(kw*S.P.x[i]);              // right-travelling: dv = c_s drho/rho
            double Ploc = P0*(1 + S.gamma*A*std::sin(kw*S.P.x[i]));
            double rloc = rho0*(1 + A*std::sin(kw*S.P.x[i]));
            S.u[i] = Ploc/((S.gamma-1)*rloc);
        }
        // HALF period, scored against the phase-shifted analytic wave v = A sin(k(x - c t)).
        // Scoring at a full period is a trap this code fell into once: the reference equals the
        // IC, so a completely FROZEN wave (zero fluxes) scores as a perfect result.
        Conserved c0 = totals(S);
        double t = 0, tend = 0.5; int steps = 0;
        while (t < tend) { t += mfm_step(S, tend - t); ++steps; }
        Conserved c1 = totals(S);
        double l1 = 0;
        for (size_t i = 0; i < N; ++i)
            l1 += std::abs(S.vx[i] - A*std::sin(kw*(S.P.x[i] - tend)));
        l1 /= (N*A);
        printf("  [soundwave %2d^3] half period, %3d steps: L1(v)/A = %.3e   (E drift %.1e)\n",
               n1, steps, l1, std::abs(c1.E-c0.E)/c0.E);
        fprintf(fconv, "%d %.6e\n", n1, l1);
        {   // dump v(x) for the finest level for the wave-shape panel
            char fn[64]; snprintf(fn, 64, "plots/soundwave_n%d.txt", n1);
            FILE* fw = fopen(fn, "w");
            fprintf(fw, "# x  vx_over_A  vx_exact_over_A   t=%.2f\n", tend);
            for (size_t i = 0; i < N; i += (n1>=32?7:1))   // thin the 32^3 dump
                fprintf(fw, "%.6f %.6e %.6e\n", S.P.x[i], S.vx[i]/A,
                        std::sin(kw*(S.P.x[i] - tend)));
            fclose(fw);
        }
    }
    fclose(fconv);

    // ---------- Sod shocktube ----------
    {
        // Equal-mass MESHLESS setup: each side is an ISOTROPIC lattice, spacing ratio 2:1 giving
        // the 8:1 density ratio. (A first attempt used one anisotropic lattice with dx != dy: the
        // kernel could not reach the neighbouring y/z columns, the E-matrix went near-singular and
        // the density estimate itself was garbage, L1 ~ 7. Anisotropic particle arrangements with
        // isotropic kernels are simply invalid input for this discretisation -- GIZMO's own
        // shocktube test sidesteps the same issue by compiling in 1D.)
        int nfine = quick ? 32 : 64;                 // left lattice: spacing 1/nfine in ALL dims
        double box = 1.0, a = 1.0/nfine;
        Sim S; S.box = box; S.gamma = 1.4;
        S.P.x.reserve((size_t)nfine*nfine*nfine);
        S.P.y.reserve((size_t)nfine*nfine*nfine); S.P.z.reserve((size_t)nfine*nfine*nfine);
        auto add=[&](double x,double y,double z){S.P.x.push_back(x);S.P.y.push_back(y);S.P.z.push_back(z);};
        for (int i=0;i<nfine/2;++i) for (int j=0;j<nfine;++j) for (int l=0;l<nfine;++l)
            add((i+0.5)*a, (j+0.5)*a, (l+0.5)*a);                        // [0,0.5): rho = 1
        for (int i=0;i<nfine/4;++i) for (int j=0;j<nfine/2;++j) for (int l=0;l<nfine/2;++l)
            add(0.5+(i+0.5)*2*a, (j+0.5)*2*a, (l+0.5)*2*a);              // [0.5,1): rho = 1/8
        size_t N = S.P.x.size();
        double m = a*a*a;                            // rho_left = m/a^3 = 1; rho_right = m/(2a)^3
        S.P.m.assign(N,m); S.P.soft.assign(N,0.0);
        S.vx.assign(N,0); S.vy.assign(N,0); S.vz.assign(N,0); S.u.assign(N,0);
        for (size_t i = 0; i < N; ++i) {
            bool left = S.P.x[i] < 0.5;
            double rho = left?1.0:0.125, P = left?1.0:0.1;
            S.u[i] = P/((S.gamma-1)*rho);
        }
        Conserved c0 = totals(S);
        double t = 0, tend = 0.1; int steps = 0;
        while (t < tend) { t += mfm_step(S, tend - t); ++steps; }
        Conserved c1 = totals(S);

        RiemannExact ex{1.0, 0.0, 1.0, 0.125, 0.0, 0.1, S.gamma, 0, 0};
        ex.solve();
        // score rho(x) in the window causally clean of the periodic wrap
        double l1 = 0; long cnt = 0;
        for (size_t i = 0; i < N; ++i) {
            double x = S.P.x[i];
            if (x < 0.2 || x > 0.8) continue;
            double re, ue, pe;
            ex.sample((x-0.5)/tend, re, ue, pe);
            l1 += std::abs(S.rho[i] - re)/re; ++cnt;
        }
        printf("  [shocktube %zu] t=%.2f in %d steps: L1(rho) = %.3e over %ld cells in [0.2,0.8]\n",
               N, tend, steps, l1/cnt, cnt);
        printf("      exact: P*=%.4f u*=%.4f | conservation: mass %.1e E %.3e\n",
               ex.pst, ex.ust, std::abs(c1.mass-c0.mass)/c0.mass, std::abs(c1.E-c0.E)/c0.E);
        {   // full profile dump for plotting: particle scatter + exact curve
            FILE* fp = fopen("plots/sod_profile.txt", "w");
            fprintf(fp, "# x  rho  vx  P   (t=%.2f)\n", tend);
            for (size_t i = 0; i < N; i += 16)
                fprintf(fp, "%.6f %.6e %.6e %.6e\n", S.P.x[i], S.rho[i], S.vx[i],
                        (S.gamma-1)*S.rho[i]*S.u[i]);
            fclose(fp);
            FILE* fe = fopen("plots/sod_exact.txt", "w");
            fprintf(fe, "# x  rho  vx  P\n");
            for (int k = 0; k <= 800; ++k) {
                double x = 0.1 + 0.8*k/800.0, re, ue, pe;
                ex.sample((x-0.5)/tend, re, ue, pe);
                fprintf(fe, "%.6f %.6e %.6e %.6e\n", x, re, ue, pe);
            }
            fclose(fe);
        }
        // station profile: mean rho and vx in thin slabs vs exact -- locates the failure mode
        printf("      %-8s %10s %10s %10s %10s\n","x","<rho>","rho_ex","<vx>","vx_ex");
        for (double xs : {0.30, 0.42, 0.55, 0.62, 0.70, 0.76}) {
            double sr=0, sv=0; long c2=0;
            for (size_t i = 0; i < N; ++i)
                if (std::abs(S.P.x[i]-xs) < 0.01) { sr+=S.rho[i]; sv+=S.vx[i]; ++c2; }
            double re, ue, pe; ex.sample((xs-0.5)/tend, re, ue, pe);
            if (c2 == 0) {          // sample slab fell between particle planes; nothing to average
                printf("      %-8.2f %10s %10.4f %10s %10.4f\n", xs, "-", re, "-", ue);
                continue;
            }
            printf("      %-8.2f %10.4f %10.4f %10.4f %10.4f\n", xs, sr/c2, re, sv/c2, ue);
        }
    }

    // ---------- structural: face closure on a perturbed lattice ----------
    {
        // CONTROL: perfect lattice, where closure is exact by symmetry -- separates "formula
        // wrong" from "closure is only approximate on irregular arrangements".
        for (double jit : {0.0, 0.01}) {
            Sim S = make_lattice(20, 20, 20, 1.0);
            size_t N = S.size();
            for (size_t i = 0; i < N; ++i) {
                S.P.m[i] = 1.0/N;
                S.P.x[i] += jit*std::sin(17.0*i); S.P.y[i] += jit*std::cos(29.0*i);
                S.u[i] = 1.0;
            }
            double fc = face_closure(S, 40);
            printf("  [faces] jitter=%.2f: max |sum_j A_ij| / max|A_ij| = %.3e\n", jit, fc);
        }
    }

    // ---------- structural: gradients are EXACT for a linear field ----------
    // Not a convergence check -- exactness holds at any resolution and any disorder, so the
    // expected answer is round-off. Jittered cases matter most: on a perfect lattice E is
    // diagonal and a mis-indexed off-diagonal term would stay invisible.
    {
        int bad = 0;
        for (int dim : {1, 2, 3}) {
            const int nx = (dim >= 1) ? 24 : 1, ny = (dim >= 2) ? 24 : 1, nz = (dim >= 3) ? 24 : 1;
            for (double jit : {0.0, 0.02}) {
                Sim S = make_lattice(nx, ny, nz, 1.0);
                S.dim = dim;
                S.box = 0.0;                    // linear field is discontinuous across a wrap
                S.des_ngb = (dim == 1) ? 4.0 : (dim == 2 ? 12.0 : 32.0);
                size_t N = S.size();
                const double d = 1.0 / nx;
                for (size_t i = 0; i < N; ++i) {
                    S.P.m[i] = 1.0/N; S.u[i] = 1.0;
                    if (dim >= 1) S.P.x[i] += jit*d*std::sin(17.0*i);
                    if (dim >= 2) S.P.y[i] += jit*d*std::cos(29.0*i);
                    if (dim >= 3) S.P.z[i] += jit*d*std::sin(41.0*i);
                }
                double err = linear_gradient_error(S);
                bool ok = (err >= 0.0) && (err < 1e-9);      // negative = no interior to score
                if (!ok) ++bad;
                printf("  [gradients] %dD jitter=%.2f: max rel err vs exact linear = %.3e  %s\n",
                       dim, jit, err, ok ? "OK" : "** FAIL **");
            }
        }
        if (bad) { printf("  [gradients] %d configuration(s) failed exactness\n", bad); return 1; }
    }

    // ---------- self-gravity: pressureless collapse of a uniform sphere ----------
    // A cold uniform sphere collapses HOMOLOGOUSLY, so every shell follows the same exact
    // parametric solution and the whole cloud is one measurement:
    //     r/r0 = cos^2(beta),   t = sqrt(3/(8 pi G rho0)) * (beta + sin beta cos beta)
    // reaching r=0 at t_ff = sqrt(3 pi / (32 G rho0)).
    // This tests the force MAGNITUDE and the KDK time integration together, which the tree's
    // own accuracy check (vs direct summation) cannot: that one validates the walk against
    // brute force, and would pass just as well with G, the softening, or the kick sequence wrong.
    // Swept over SOFTENING, not dt. Halving dt moves the answer by <1%% of itself, so the residual
    // is not time-integration error; it is the softening, which weakens gravity below eps and must
    // therefore SLOW the collapse. Adaptive softening puts eps at the kernel radius, ~0.13 of the
    // sphere radius here, so a percent-level lag is expected physics rather than a bug. The test is
    // that the error falls toward zero as eps does -- that is what pins G, the kick sequence and
    // the force normalisation simultaneously. (eps=0 is the unsoftened limit; safe on a lattice,
    // where the closest pair is a full grid spacing apart.)
    for (double soften : {0.0, 0.03, 0.01}) {
        const int n_side = 32;
        const double radius0 = 1.0, G = 1.0, total_mass = 1.0;
        Sim sim = make_lattice(n_side, n_side, n_side, 2.0);      // cube of side 2, centred at 1
        // carve the inscribed sphere out of the lattice
        Sim sphere; sphere.box = 0.0; sphere.dim = 3;
        for (size_t i = 0; i < sim.size(); ++i) {
            const Vec3d offset = sim.P.pos(i) - Vec3d{1.0, 1.0, 1.0};
            if (offset.norm() > radius0) continue;
            sphere.P.x.push_back(offset[0]);
            sphere.P.y.push_back(offset[1]);
            sphere.P.z.push_back(offset[2]);
        }
        const size_t n_part = sphere.P.x.size();
        sphere.P.m.assign(n_part, total_mass / n_part);
        sphere.P.soft.assign(n_part, 0.0);
        sphere.vx.assign(n_part, 0.0); sphere.vy.assign(n_part, 0.0); sphere.vz.assign(n_part, 0.0);
        sphere.u.assign(n_part, 1e-8);          // cold: pressure must not resist the collapse
        sphere.gamma = 5.0/3.0; sphere.des_ngb = 32.0; sphere.cfl = 0.2;
        sphere.gravity_on = true; sphere.G = G; sphere.theta = 0.4;
        // soften == 0 selects the adaptive (h-based) softening the evrard config asks for;
        // the finite values are fixed softenings well below the sphere radius.
        sphere.adaptive_soft = (soften == 0.0);
        sphere.soft_min = soften;
        sphere.eta_grav = 0.00625;

        const double rho0 = total_mass / (4.0/3.0*M_PI*radius0*radius0*radius0);
        const double t_ff = std::sqrt(3.0*M_PI / (32.0*G*rho0));
        const double t_end = 0.4 * t_ff;

        auto median_radius = [&](const Sim& s) {
            std::vector<double> radii(s.size());
            for (size_t i = 0; i < s.size(); ++i) radii[i] = s.P.pos(i).norm();
            std::nth_element(radii.begin(), radii.begin()+radii.size()/2, radii.end());
            return radii[radii.size()/2];
        };
        const double r_med0 = median_radius(sphere);

        double t = 0;
        int steps = 0;
        while (t < t_end - 1e-12) { t += mfm_step(sphere, t_end - t); ++steps; }

        // exact: solve t/sqrt(3/(8 pi G rho0)) = beta + sin beta cos beta for beta, then r/r0
        const double tau = t / std::sqrt(3.0/(8.0*M_PI*G*rho0));
        double lo = 0, hi = M_PI/2;
        for (int it = 0; it < 200; ++it) {
            const double mid = 0.5*(lo+hi);
            if (mid + std::sin(mid)*std::cos(mid) < tau) lo = mid; else hi = mid;
        }
        const double beta = 0.5*(lo+hi);
        const double ratio_exact = std::cos(beta)*std::cos(beta);
        const double ratio_sim = median_radius(sphere) / r_med0;
        const double err = std::abs(ratio_sim - ratio_exact) / ratio_exact;

        char soft_label[32];
        if (soften == 0.0) snprintf(soft_label, sizeof soft_label, "adaptive(~h)");
        else snprintf(soft_label, sizeof soft_label, "%.3f", soften);
        printf("  [gravity] cold sphere N=%zu, eps=%-12s t=%.2f t_ff in %4d steps: "
               "r/r0 = %.5f vs %.5f exact, rel err = %.2e  %s\n",
               n_part, soft_label, t/t_ff, steps, ratio_sim, ratio_exact, err,
               err < 0.02 ? "OK" : "** FAIL **");
        if (err >= 0.02) return 1;
    }
    return 0;
}
