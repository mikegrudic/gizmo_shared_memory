#include "mfm.h"
#include <cstdio>

namespace shmem {

// 3x3 symmetric inverse; returns false if the condition is hopeless (degenerate neighbour
// geometry). Callers fall back to zero gradients there -- first-order but safe, GIZMO does the same.
static bool inv3(const double E[6], double B[9]) {
    // E packed: xx, xy, xz, yy, yz, zz
    double a=E[0], b=E[1], c=E[2], d=E[3], e=E[4], f=E[5];
    double det = a*(d*f - e*e) - b*(b*f - e*c) + c*(b*e - d*c);
    if (std::abs(det) < 1e-300) return false;
    double inv = 1.0 / det;
    B[0]=(d*f-e*e)*inv; B[1]=(c*e-b*f)*inv; B[2]=(b*e-c*d)*inv;
    B[3]=B[1];          B[4]=(a*f-c*c)*inv; B[5]=(b*c-a*e)*inv;
    B[6]=B[2];          B[7]=B[5];          B[8]=(a*d-b*b)*inv;
    return true;
}

static inline double minmod(double a, double b) {
    return (a*b <= 0) ? 0.0 : (std::abs(a) < std::abs(b) ? a : b);
}

// HLLC for ideal gas, 1D states normal to the face. Returns contact speed S* and pressure P*.
// Wave speed estimates: Davis. Inputs are the face-frame normal velocities.
static void hllc_star(double rhoL, double uL, double pL, double rhoR, double uR, double pR,
                      double gamma, double& Sstar, double& Pstar) {
    double cL = std::sqrt(gamma * pL / rhoL), cR = std::sqrt(gamma * pR / rhoR);
    double SL = std::min(uL - cL, uR - cR);
    double SR = std::max(uL + cL, uR + cR);
    double num = pR - pL + rhoL*uL*(SL - uL) - rhoR*uR*(SR - uR);
    double den = rhoL*(SL - uL) - rhoR*(SR - uR);
    Sstar = (std::abs(den) > 1e-300) ? num / den : 0.5*(uL + uR);
    Pstar = pL + rhoL*(SL - uL)*(Sstar - uL);
    if (Pstar < 0) Pstar = 0.5*(pL + pR);       // vacuum-adjacent guard; tests never hit it
}

// per-step scratch shared by the phases
struct Work {
    std::vector<double> B;                     // 9 per particle
    std::vector<double> gr[5];                 // gradients of rho, vx, vy, vz, P (3 each)
    std::vector<double> rho2, vx2, vy2, vz2, p2;  // half-step predicted primitives
};

static void solve_h_and_volumes(Sim& S, const Tree& T) {
    const size_t n = S.size();
    std::vector<uint32_t> all(n);
    for (size_t i = 0; i < n; ++i) all[i] = (uint32_t)i;
    std::vector<double> h0 = S.h;              // warm start from last step if present
    if (h0.size() != n) h0.clear();
    DensityResult R = density(T, S.P, all, S.des_ngb, h0, S.box);
    S.h = R.h;
    S.ninv.assign(n, 0.0); S.rho.assign(n, 0.0); S.press.assign(n, 0.0);
    #pragma omp parallel
    {
        std::vector<uint32_t> ngb;
        #pragma omp for schedule(dynamic, 64)
        for (size_t i = 0; i < n; ++i) {
            ngb.clear();
            ngb_search(T, S.P, S.P.x[i], S.P.y[i], S.P.z[i], S.h[i], ngb, S.box);
            double wsum = 0;
            for (uint32_t q : ngb) {
                double dx=wrap(S.P.x[q]-S.P.x[i],S.box), dy=wrap(S.P.y[q]-S.P.y[i],S.box), dz=wrap(S.P.z[q]-S.P.z[i],S.box);
                wsum += kernel_w(std::sqrt(dx*dx+dy*dy+dz*dz), S.h[i]);
            }
            S.ninv[i] = 1.0 / wsum;                       // V_i: MFM volume from partition of unity
            S.rho[i]  = S.P.m[i] * wsum;                  // rho_i = m_i / V_i
            S.press[i] = (S.gamma - 1.0) * S.rho[i] * S.u[i];
        }
    }
}

static void gradients(Sim& S, const Tree& T, Work& W) {
    const size_t n = S.size();
    W.B.assign(9*n, 0.0);
    for (auto& g : W.gr) g.assign(3*n, 0.0);
    #pragma omp parallel
    {
        std::vector<uint32_t> ngb;
        #pragma omp for schedule(dynamic, 64)
        for (size_t i = 0; i < n; ++i) {
            ngb.clear();
            ngb_search(T, S.P, S.P.x[i], S.P.y[i], S.P.z[i], S.h[i], ngb, S.box);
            double E[6] = {0,0,0,0,0,0};
            for (uint32_t q : ngb) {
                double dx=wrap(S.P.x[q]-S.P.x[i],S.box), dy=wrap(S.P.y[q]-S.P.y[i],S.box), dz=wrap(S.P.z[q]-S.P.z[i],S.box);
                double w = kernel_w(std::sqrt(dx*dx+dy*dy+dz*dz), S.h[i]);
                E[0]+=dx*dx*w; E[1]+=dx*dy*w; E[2]+=dx*dz*w; E[3]+=dy*dy*w; E[4]+=dy*dz*w; E[5]+=dz*dz*w;
            }
            double* B = &W.B[9*i];
            if (!inv3(E, B)) { for (int k=0;k<9;++k) B[k]=0; continue; }   // zero gradients fallback
            const double f0[5] = {S.rho[i], S.vx[i], S.vy[i], S.vz[i], S.press[i]};
            double acc[5][3] = {{0}};
            for (uint32_t q : ngb) {
                double dx=wrap(S.P.x[q]-S.P.x[i],S.box), dy=wrap(S.P.y[q]-S.P.y[i],S.box), dz=wrap(S.P.z[q]-S.P.z[i],S.box);
                double w = kernel_w(std::sqrt(dx*dx+dy*dy+dz*dz), S.h[i]);
                const double fj[5] = {S.rho[q], S.vx[q], S.vy[q], S.vz[q], S.press[q]};
                for (int f = 0; f < 5; ++f) {
                    double df = (fj[f] - f0[f]) * w;
                    acc[f][0] += df*dx; acc[f][1] += df*dy; acc[f][2] += df*dz;
                }
            }
            for (int f = 0; f < 5; ++f) {
                W.gr[f][3*i+0] = B[0]*acc[f][0] + B[1]*acc[f][1] + B[2]*acc[f][2];
                W.gr[f][3*i+1] = B[3]*acc[f][0] + B[4]*acc[f][1] + B[5]*acc[f][2];
                W.gr[f][3*i+2] = B[6]*acc[f][0] + B[7]*acc[f][1] + B[8]*acc[f][2];
            }
        }
    }
}

// Lagrangian half-step prediction of primitives (MUSCL-Hancock predictor).
static void predict_half(Sim& S, Work& W, double dt) {
    const size_t n = S.size();
    W.rho2.resize(n); W.vx2.resize(n); W.vy2.resize(n); W.vz2.resize(n); W.p2.resize(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        double divv = W.gr[1][3*i+0] + W.gr[2][3*i+1] + W.gr[3][3*i+2];
        double irho = 1.0 / S.rho[i], hdt = 0.5 * dt;
        W.rho2[i] = std::max(S.rho[i] * (1.0 - hdt*divv), 1e-30);
        W.vx2[i]  = S.vx[i] - hdt * irho * W.gr[4][3*i+0];
        W.vy2[i]  = S.vy[i] - hdt * irho * W.gr[4][3*i+1];
        W.vz2[i]  = S.vz[i] - hdt * irho * W.gr[4][3*i+2];
        W.p2[i]   = std::max(S.press[i] * (1.0 - S.gamma*hdt*divv), 1e-30);
    }
}

// Flux exchange over unique pairs. Pair discovery from the SMALLER kernel side would miss
// asymmetric pairs, so: i owns the pair when (i < q) and r < max(h_i, h_q); every pair is then
// found exactly once because both sides search with max(h_i, h_q) coverage via the union below.
static void fluxes(Sim& S, const Tree& T, Work& W, double dt,
                   std::vector<double>& dpx, std::vector<double>& dpy,
                   std::vector<double>& dpz, std::vector<double>& dE) {
    const size_t n = S.size();
    dpx.assign(n,0); dpy.assign(n,0); dpz.assign(n,0); dE.assign(n,0);
    #pragma omp parallel
    {
        std::vector<uint32_t> ngb;
        #pragma omp for schedule(dynamic, 64)
        for (size_t i = 0; i < n; ++i) {
            // search with h_i; pairs where h_q > r >= h_i are found from q's side (q also loops)
            ngb.clear();
            ngb_search(T, S.P, S.P.x[i], S.P.y[i], S.P.z[i], S.h[i], ngb, S.box);
            for (uint32_t q : ngb) {
                if (q == i) continue;
                double dx=wrap(S.P.x[q]-S.P.x[i],S.box), dy=wrap(S.P.y[q]-S.P.y[i],S.box), dz=wrap(S.P.z[q]-S.P.z[i],S.box);
                double r = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (r <= 0) continue;
                // Each pair must be processed EXACTLY once, including asymmetric ones (r < h_i but
                // r >= h_q, so only i sees q). Ownership: the lower index owns the pair IF it can
                // see it; otherwise the higher index picks it up. A plain q<=i skip drops the
                // asymmetric pairs that only the larger-h side can see.
                if (q < i && r < S.h[q]) continue;
                double wi = kernel_w(r, S.h[i]);
                double wq = kernel_w(r, S.h[q]);
                // face vector A_ij (points i -> q)
                const double *Bi = &W.B[9*i], *Bq = &W.B[9*q];
                double Vi = S.ninv[i], Vq = S.ninv[q];
                double ax = Vi*(Bi[0]*dx + Bi[1]*dy + Bi[2]*dz)*wi + Vq*(Bq[0]*dx + Bq[1]*dy + Bq[2]*dz)*wq;
                double ay = Vi*(Bi[3]*dx + Bi[4]*dy + Bi[5]*dz)*wi + Vq*(Bq[3]*dx + Bq[4]*dy + Bq[5]*dz)*wq;
                double az = Vi*(Bi[6]*dx + Bi[7]*dy + Bi[8]*dz)*wi + Vq*(Bq[6]*dx + Bq[7]*dy + Bq[8]*dz)*wq;
                double Amag = std::sqrt(ax*ax + ay*ay + az*az);
                if (Amag <= 0) continue;
                double nx = ax/Amag, ny = ay/Amag, nz = az/Amag;

                // linear reconstruction to the face midpoint, pairwise minmod-limited per field
                double half[3] = {0.5*dx, 0.5*dy, 0.5*dz};
                // gradient extrapolation from side a to the face midpoint (sgn: +1 from i, -1 from q)
                auto recon = [&](int g, size_t a, double sgn)->double {
                    return sgn*(W.gr[g][3*a+0]*half[0] + W.gr[g][3*a+1]*half[1] + W.gr[g][3*a+2]*half[2]);
                };
                // fields: 0 rho, 1 vx, 2 vy, 3 vz, 4 P -- reconstruct from the half-step primitives
                const std::vector<double>* F2[5] = {&W.rho2, &W.vx2, &W.vy2, &W.vz2, &W.p2};
                double L[5], R[5];
                for (int f = 0; f < 5; ++f) {
                    const std::vector<double>& v = *F2[f];
                    double dfi = recon(f, i, +1.0);
                    double dfq = recon(f, q, -1.0);
                    double jump = v[q] - v[i];
                    dfi = minmod(dfi, 0.5*jump);
                    dfq = minmod(dfq, -0.5*jump);
                    L[f] = v[i] + dfi;
                    R[f] = v[q] + dfq;
                }
                if (L[0] <= 0 || R[0] <= 0 || L[4] <= 0 || R[4] <= 0) {   // limiter emergency
                    L[0]=W.rho2[i]; L[4]=W.p2[i]; R[0]=W.rho2[q]; R[4]=W.p2[q];
                }

                // face frame: mean velocity; rotate to the normal
                double vfx = 0.5*(W.vx2[i]+W.vx2[q]), vfy = 0.5*(W.vy2[i]+W.vy2[q]), vfz = 0.5*(W.vz2[i]+W.vz2[q]);
                double uL = (L[1]-vfx)*nx + (L[2]-vfy)*ny + (L[3]-vfz)*nz;
                double uR = (R[1]-vfx)*nx + (R[2]-vfy)*ny + (R[3]-vfz)*nz;
                double Sstar, Pstar;
                hllc_star(L[0], uL, L[4], R[0], uR, R[4], S.gamma, Sstar, Pstar);

                // Lagrangian flux: zero mass flux; pressure P* acts across A, face moves at
                // v_face = v_frame + S* nhat (lab). Momentum from i to q along +nhat.
                double vfn = vfx*nx + vfy*ny + vfz*nz + Sstar;      // lab-frame normal face speed
                double fP  = Pstar * Amag * dt;
                double fE  = Pstar * vfn * Amag * dt;
                #pragma omp atomic
                dpx[i] -= fP * nx;
                #pragma omp atomic
                dpy[i] -= fP * ny;
                #pragma omp atomic
                dpz[i] -= fP * nz;
                #pragma omp atomic
                dE[i]  -= fE;
                #pragma omp atomic
                dpx[q] += fP * nx;
                #pragma omp atomic
                dpy[q] += fP * ny;
                #pragma omp atomic
                dpz[q] += fP * nz;
                #pragma omp atomic
                dE[q]  += fE;
            }
        }
    }
}

double mfm_step(Sim& S, double dt_max) {
    const size_t n = S.size();
    Tree T = build(S.P);
    solve_h_and_volumes(S, T);

    // CFL from signal speed
    double dt = dt_max;
    #pragma omp parallel for reduction(min:dt) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        double c = std::sqrt(S.gamma * S.press[i] / S.rho[i]);
        double v = std::sqrt(S.vx[i]*S.vx[i] + S.vy[i]*S.vy[i] + S.vz[i]*S.vz[i]);
        dt = std::min(dt, S.cfl * S.h[i] / (c + v + 1e-300));
    }

    Work W;
    gradients(S, T, W);
    predict_half(S, W, dt);

    std::vector<double> dpx, dpy, dpz, dE;
    fluxes(S, T, W, dt, dpx, dpy, dpz, dE);

    // conserved update + Lagrangian drift
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        double m = S.P.m[i];
        double px = m*S.vx[i] + dpx[i], py = m*S.vy[i] + dpy[i], pz = m*S.vz[i] + dpz[i];
        double E  = m*(S.u[i] + 0.5*(S.vx[i]*S.vx[i]+S.vy[i]*S.vy[i]+S.vz[i]*S.vz[i])) + dE[i];
        S.vx[i] = px/m; S.vy[i] = py/m; S.vz[i] = pz/m;
        S.u[i]  = std::max(E/m - 0.5*(S.vx[i]*S.vx[i]+S.vy[i]*S.vy[i]+S.vz[i]*S.vz[i]), 1e-30);
        S.P.x[i] += S.vx[i]*dt; S.P.y[i] += S.vy[i]*dt; S.P.z[i] += S.vz[i]*dt;
        if (S.box > 0) {
            auto pw=[&](double& c){ if (c >= S.box) c -= S.box; else if (c < 0) c += S.box; };
            pw(S.P.x[i]); pw(S.P.y[i]); pw(S.P.z[i]);
        }
    }
    return dt;
}

Conserved totals(const Sim& S) {
    Conserved c{0,0,0,0,0};
    for (size_t i = 0; i < S.size(); ++i) {
        double m = S.P.m[i];
        c.mass += m; c.px += m*S.vx[i]; c.py += m*S.vy[i]; c.pz += m*S.vz[i];
        c.E += m*(S.u[i] + 0.5*(S.vx[i]*S.vx[i]+S.vy[i]*S.vy[i]+S.vz[i]*S.vz[i]));
    }
    return c;
}

double face_closure(Sim& S, int nsample) {
    Tree T = build(S.P);
    solve_h_and_volumes(S, T);
    Work W;
    gradients(S, T, W);
    double worst = 0, amax = 0;
    std::vector<uint32_t> ngb;
    for (int s = 0; s < nsample; ++s) {
        size_t i = (size_t)((uint64_t)s * 2654435761u % S.size());
        ngb.clear();
        // union coverage: h_i search catches pairs from i's side only; for closure include the
        // symmetric contribution by searching the larger radius neighbourhood
        double hmax = S.h[i]; for (size_t q=0;q<S.size();++q) hmax = std::max(hmax, S.h[q]);
        ngb_search(T, S.P, S.P.x[i], S.P.y[i], S.P.z[i], hmax, ngb, S.box);
        double sx=0, sy=0, sz=0;
        for (uint32_t q : ngb) {
            if (q == i) continue;
            double dx=wrap(S.P.x[q]-S.P.x[i],S.box), dy=wrap(S.P.y[q]-S.P.y[i],S.box), dz=wrap(S.P.z[q]-S.P.z[i],S.box);
            double r = std::sqrt(dx*dx+dy*dy+dz*dz);
            double wi = kernel_w(r, S.h[i]), wq = kernel_w(r, S.h[q]);
            if (wi <= 0 && wq <= 0) continue;
            const double *Bi = &W.B[9*i], *Bq = &W.B[9*q];
            double Vi = S.ninv[i], Vq = S.ninv[q];
            double ax = Vi*(Bi[0]*dx+Bi[1]*dy+Bi[2]*dz)*wi + Vq*(Bq[0]*dx+Bq[1]*dy+Bq[2]*dz)*wq;
            double ay = Vi*(Bi[3]*dx+Bi[4]*dy+Bi[5]*dz)*wi + Vq*(Bq[3]*dx+Bq[4]*dy+Bq[5]*dz)*wq;
            double az = Vi*(Bi[6]*dx+Bi[7]*dy+Bi[8]*dz)*wi + Vq*(Bq[6]*dx+Bq[7]*dy+Bq[8]*dz)*wq;
            sx += ax; sy += ay; sz += az;
            amax = std::max(amax, std::sqrt(ax*ax+ay*ay+az*az));
        }
        worst = std::max(worst, std::sqrt(sx*sx+sy*sy+sz*sz));
    }
    return (amax > 0) ? worst / amax : 0.0;
}

}  // namespace shmem
