#include "hydro.h"

namespace shmem {

void ngb_search(const Tree& T, const Particles& P, double x, double y, double z,
                double rad, std::vector<uint32_t>& out, double box) {
    const WNode* __restrict W = T.wn.data();
    const double rad2_pad = rad;             // node prune uses rad + s, particle test uses rad
    int node = T.root;
    while (node >= 0) {
        const WNode& w = W[node];
        double dx = wrap(w.cx - x, box), dy = wrap(w.cy - y, box), dz = wrap(w.cz - z, box);
        double r2 = dx*dx + dy*dy + dz*dz;
        double keep = rad2_pad + w.s;        // conservative: all node particles are within s of COM
        if (r2 > keep * keep) { node = w.next; continue; }
        if (w.first < 0) {
            for (int i = w.plo; i < w.phi; ++i) {
                uint32_t q = T.orderbuf[i];
                double qx = wrap(P.x[q]-x, box), qy = wrap(P.y[q]-y, box), qz = wrap(P.z[q]-z, box);
                if (qx*qx + qy*qy + qz*qz < rad*rad) out.push_back(q);
            }
            node = w.next;
        } else {
            node = w.first;
        }
    }
}

DensityResult density(const Tree& T, const Particles& P, const std::vector<uint32_t>& targets,
                      double des_ngb, const std::vector<double>& h0, double box) {
    const size_t nt = targets.size();
    DensityResult R;
    R.h.assign(nt, 0.0); R.rho.assign(nt, 0.0); R.nngb.assign(nt, 0); R.iters.assign(nt, 0);

    // default guess: des_ngb particles at the global mean density
    double hguess = 0.0;
    if (h0.empty()) {
        double lo[3] = {1e300,1e300,1e300}, hi[3] = {-1e300,-1e300,-1e300};
        for (size_t i = 0; i < P.size(); ++i) {
            lo[0]=std::min(lo[0],P.x[i]); hi[0]=std::max(hi[0],P.x[i]);
            lo[1]=std::min(lo[1],P.y[i]); hi[1]=std::max(hi[1],P.y[i]);
            lo[2]=std::min(lo[2],P.z[i]); hi[2]=std::max(hi[2],P.z[i]);
        }
        double vol = (hi[0]-lo[0])*(hi[1]-lo[1])*(hi[2]-lo[2]);
        hguess = std::cbrt(3.0*des_ngb*vol / (4.0*M_PI*P.size()));
    }

    #pragma omp parallel
    {
        std::vector<uint32_t> ngb;           // per-thread scratch, reused across targets
        #pragma omp for schedule(dynamic, 16)
        for (size_t t = 0; t < nt; ++t) {
            const uint32_t p = targets[t];
            const double px = P.x[p], py = P.y[p], pz = P.z[p];
            double h = h0.empty() ? hguess : h0[t];
            double hlo = 0.0, hhi = 0.0;     // bisection bracket, grown as we learn
            int it = 0;
            for (; it < 100; ++it) {
                ngb.clear();
                ngb_search(T, P, px, py, pz, h, ngb, box);
                double wsum = 0.0, dwdh = 0.0;
                for (uint32_t q : ngb) {
                    double dx = wrap(P.x[q]-px, box), dy = wrap(P.y[q]-py, box), dz = wrap(P.z[q]-pz, box);
                    double r = std::sqrt(dx*dx + dy*dy + dz*dz);
                    wsum += kernel_w(r, h);
                    dwdh += kernel_dwdh(r, h);
                }
                const double pref = 4.0 * M_PI / 3.0;
                double neff = pref * h*h*h * wsum;
                double f = neff - des_ngb;
                if (std::abs(f) < 1e-4 * des_ngb) break;
                if (f > 0) hhi = h; else hlo = h;

                // Newton on N_eff(h), guarded by the bracket. dN/dh = pref*(3h^2 wsum + h^3 dwdh);
                // it is positive away from pathological configurations, but the guard makes any
                // bad step safe: fall back to bisection / geometric growth.
                double dndh = pref * (3.0*h*h*wsum + h*h*h*dwdh);
                double hnew = (dndh > 0) ? h - f / dndh : 0.0;
                if (hnew <= hlo || (hhi > 0 && hnew >= hhi) || hnew <= 0) {
                    if (hhi > 0) hnew = (hlo > 0) ? std::sqrt(hlo * hhi) : 0.5 * hhi;
                    else hnew = h * ((f > 0) ? 0.7 : 1.6);
                }
                h = hnew;
            }
            // final density on the converged h
            ngb.clear();
            ngb_search(T, P, px, py, pz, h, ngb, box);
            double rho = 0.0; int inside = 0;
            for (uint32_t q : ngb) {
                double dx = wrap(P.x[q]-px, box), dy = wrap(P.y[q]-py, box), dz = wrap(P.z[q]-pz, box);
                double r = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (r < h) ++inside;
                rho += P.m[q] * kernel_w(r, h);
            }
            R.h[t] = h; R.rho[t] = rho; R.nngb[t] = inside; R.iters[t] = it;
        }
    }
    return R;
}

}  // namespace shmem
