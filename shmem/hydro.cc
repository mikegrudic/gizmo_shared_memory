#include "hydro.h"

namespace shmem {

void ngb_search(const Tree& tree, const Particles& particles, const Vec3d& centre,
                double radius, std::vector<uint32_t>& found, double box) {
    const WNode* __restrict nodes = tree.wn.data();
    const double radius_sq = radius * radius;
    int node_id = tree.root;
    while (node_id >= 0) {
        const WNode& node = nodes[node_id];
        const Vec3d to_com = min_image(Vec3d{node.cx, node.cy, node.cz} - centre, box);
        // conservative: every particle in the node lies within node.s of its centre of mass, plus
        // tree.pad for however far particles have drifted since the tree was built
        const double keep_within = radius + node.s + tree.pad;
        if (to_com.norm_sq() > keep_within * keep_within) { node_id = node.next; continue; }
        if (node.first < 0) {
            for (int slot = node.plo; slot < node.phi; ++slot) {
                const uint32_t j = tree.orderbuf[slot];
                if (min_image(particles.pos(j) - centre, box).norm_sq() < radius_sq)
                    found.push_back(j);
            }
            node_id = node.next;
        } else {
            node_id = node.first;
        }
    }
}

DensityResult density(const Tree& tree, const Particles& particles,
                      const std::vector<uint32_t>& targets, double des_ngb,
                      const std::vector<double>& h_start, double box, int n_dims) {
    const size_t n_targets = targets.size();
    DensityResult result;
    result.h.assign(n_targets, 0.0);    result.rho.assign(n_targets, 0.0);
    result.nngb.assign(n_targets, 0);   result.iters.assign(n_targets, 0);

    // default guess: des_ngb particles at the global mean density
    double h_default = 0.0;
    if (h_start.empty()) {
        double lo[3] = {1e300,1e300,1e300}, hi[3] = {-1e300,-1e300,-1e300};
        for (size_t i = 0; i < particles.size(); ++i) {
            lo[0]=std::min(lo[0],particles.x[i]); hi[0]=std::max(hi[0],particles.x[i]);
            lo[1]=std::min(lo[1],particles.y[i]); hi[1]=std::max(hi[1],particles.y[i]);
            lo[2]=std::min(lo[2],particles.z[i]); hi[2]=std::max(hi[2],particles.z[i]);
        }
        const double bbox_volume = (hi[0]-lo[0])*(hi[1]-lo[1])*(hi[2]-lo[2]);
        // per-dimension guess: des_ngb particles at mean number density (bbox_volume is the 3D
        // volume; degenerate dims collapse it, so use the 1D/2D extents there instead)
        h_default = std::cbrt(3.0*des_ngb*bbox_volume / (4.0*M_PI*particles.size()));
        if (n_dims == 1)
            h_default = 0.5*des_ngb*(hi[0]-lo[0])/particles.size();
        if (n_dims == 2)
            h_default = std::sqrt(des_ngb*(hi[0]-lo[0])*(hi[1]-lo[1])/(M_PI*particles.size()));
    }

    #pragma omp parallel
    {
        std::vector<uint32_t> neighbours;    // per-thread scratch, reused across targets
        #pragma omp for schedule(dynamic, 16)
        for (size_t t = 0; t < n_targets; ++t) {
            const Vec3d pos_target = particles.pos(targets[t]);
            double h = h_start.empty() ? h_default : h_start[t];
            double h_lo = 0.0, h_hi = 0.0;   // bisection bracket, grown as we learn
            int iter = 0;
            for (; iter < 100; ++iter) {
                neighbours.clear();
                ngb_search(tree, particles, pos_target, h, neighbours, box);
                double weight_sum = 0.0, dweight_dh = 0.0;
                for (uint32_t j : neighbours) {
                    const double r = min_image(particles.pos(j) - pos_target, box).norm();
                    weight_sum += kernel_w(r, h, n_dims);
                    dweight_dh += kernel_dwdh(r, h, n_dims);
                }
                const double n_eff = ball_vol(h, n_dims) * weight_sum;
                const double residual = n_eff - des_ngb;
                if (std::abs(residual) < 1e-4 * des_ngb) break;
                if (residual > 0) h_hi = h; else h_lo = h;

                // Newton on N_eff(h), guarded by the bracket. dN/dh is positive away from
                // pathological configurations, but the guard makes any bad step safe: fall back
                // to bisection / geometric growth.
                const double dneff_dh =
                    ball_vol(h, n_dims) * ((double)n_dims/h*weight_sum + dweight_dh);
                double h_next = (dneff_dh > 0) ? h - residual / dneff_dh : 0.0;
                if (h_next <= h_lo || (h_hi > 0 && h_next >= h_hi) || h_next <= 0) {
                    if (h_hi > 0) h_next = (h_lo > 0) ? std::sqrt(h_lo * h_hi) : 0.5 * h_hi;
                    else h_next = h * ((residual > 0) ? 0.7 : 1.6);
                }
                h = h_next;
            }
            // final density on the converged h
            neighbours.clear();
            ngb_search(tree, particles, pos_target, h, neighbours, box);
            double rho = 0.0; int n_inside = 0;
            for (uint32_t j : neighbours) {
                const double r = min_image(particles.pos(j) - pos_target, box).norm();
                if (r < h) ++n_inside;
                rho += particles.m[j] * kernel_w(r, h, n_dims);
            }
            result.h[t] = h; result.rho[t] = rho;
            result.nngb[t] = n_inside; result.iters[t] = iter;
        }
    }
    return result;
}

}  // namespace shmem
