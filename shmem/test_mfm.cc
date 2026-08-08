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
    }

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
        // station profile: mean rho and vx in thin slabs vs exact -- locates the failure mode
        printf("      %-8s %10s %10s %10s %10s\n","x","<rho>","rho_ex","<vx>","vx_ex");
        for (double xs : {0.30, 0.42, 0.55, 0.62, 0.70, 0.76}) {
            double sr=0, sv=0; long c2=0;
            for (size_t i = 0; i < N; ++i)
                if (std::abs(S.P.x[i]-xs) < 0.01) { sr+=S.rho[i]; sv+=S.vx[i]; ++c2; }
            double re, ue, pe; ex.sample((xs-0.5)/tend, re, ue, pe);
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
    return 0;
}
