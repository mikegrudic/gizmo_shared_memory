#include "hydro.h"
#include <omp.h>

namespace shmem {

void ngb_search(const Tree& tree, const Particles& particles, const Vec3d& centre,
                double radius, std::vector<uint32_t>& found, double box,
                const LazyDrift* lazy) {
    const WNode* __restrict nodes = tree.wn.data();
    const float* __restrict node_vmax = tree.vmax.data();
    const double elapsed = tree.t_since_build;
    const double radius_sq = radius * radius;
    int node_id = tree.root;
    while (node_id >= 0) {
        const WNode& node = nodes[node_id];
        const Vec3d to_com = min_image(Vec3d{node.cx, node.cy, node.cz} - centre, box);
        // Conservative: every particle in the node lies within node.s of the node's centre of mass
        // as it was AT BUILD TIME, and none can have moved further than this node's own vmax in the
        // time since. Per node rather than a single global pad -- one fast particle must not
        // inflate the prune for the whole box (see Tree::vmax).
        const double keep_within = radius + node.s + (double)node_vmax[node_id] * elapsed;
        if (to_com.norm_sq() > keep_within * keep_within) { node_id = node.next; continue; }
        if (node.first < 0) {
            // Two spellings of the same loop so the common (nothing stale) path keeps exactly the
            // instructions it had before lazy drift existed.
            if (lazy) {
                for (int slot = node.plo; slot < node.phi; ++slot) {
                    const uint32_t j = tree.orderbuf[slot];
                    // catch up BEFORE the distance test, as GIZMO does: testing a stale position
                    // would let a true neighbour fall outside the radius and be dropped
                    if (lazy->last[j] < lazy->target) lazy->catch_up(lazy->ctx, j);
                    if (min_image(particles.pos(j) - centre, box).norm_sq() < radius_sq)
                        found.push_back(j);
                }
            } else {
                for (int slot = node.plo; slot < node.phi; ++slot) {
                    const uint32_t j = tree.orderbuf[slot];
                    if (min_image(particles.pos(j) - centre, box).norm_sq() < radius_sq)
                        found.push_back(j);
                }
            }
            node_id = node.next;
        } else {
            node_id = node.first;
        }
    }
}

DensityResult density(const Tree& tree, const Particles& particles,
                      const std::vector<uint32_t>& targets, double des_ngb, double ngb_tol,
                      const std::vector<double>& h_start, double box, int n_dims,
                      NeighborCache* cache, const LazyDrift* lazy) {
    const size_t n_targets = targets.size();
    DensityResult result;
    result.h.assign(n_targets, 0.0);    result.rho.assign(n_targets, 0.0);
    result.nngb.assign(n_targets, 0);   result.iters.assign(n_targets, 0);

    // Shared-neighbour-search bookkeeping: each thread appends the converged lists of the targets
    // it handled to a private buffer and notes where each landed; a prefix sum then places them.
    // Counting first and re-traversing to fill would put back the traversal this exists to remove.
    const int cache_threads = cache ? omp_get_max_threads() : 0;
    std::vector<std::vector<uint32_t>> local_flat(cache_threads);
    std::vector<std::vector<std::pair<size_t, size_t>>> local_index(cache_threads);
    if (cache) { cache->clear(); cache->start.assign(n_targets + 1, 0); }

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
        const int tid = cache ? omp_get_thread_num() : 0;
        #pragma omp for schedule(dynamic, 16)
        for (size_t t = 0; t < n_targets; ++t) {
            const Vec3d pos_target = particles.pos(targets[t]);
            double h = h_start.empty() ? h_default : h_start[t];
            double h_lo = 0.0, h_hi = 0.0;   // bisection bracket, grown as we learn
            int iter = 0;
            // Density and neighbour count are accumulated INSIDE the solve rather than in a
            // separate pass afterwards. On the iteration that converges, the neighbour set and
            // every kernel weight already correspond to the accepted h, so a final traversal
            // would recompute exactly what this loop just computed -- and a traversal per target
            // per step is not free: with gravity off (sedov) these searches are the entire cost.
            double rho = 0.0;
            int n_inside = 0;
            bool converged = false;
            for (; iter < 100; ++iter) {
                neighbours.clear();
                ngb_search(tree, particles, pos_target, h, neighbours, box, lazy);
                double weight_sum = 0.0, dweight_dh = 0.0;
                rho = 0.0; n_inside = 0;
                for (uint32_t j : neighbours) {
                    if (!particles.is_gas(j)) continue;   // gas h counts GAS neighbours only
                    const double r = min_image(particles.pos(j) - pos_target, box).norm();
                    weight_sum += kernel_w(r, h, n_dims);
                    dweight_dh += kernel_dwdh(r, h, n_dims);
                    rho += particles.m[j] * kernel_w(r, h, n_dims);
                    if (r < h) ++n_inside;
                }
                const double n_eff = ball_vol(h, n_dims) * weight_sum;
                const double residual = n_eff - des_ngb;
                if (std::abs(residual) < ngb_tol) { converged = true; break; }
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
            // Only a solve that ran out of iterations needs a final pass: there h was updated
            // after the last evaluation, so the accumulated rho belongs to the previous h.
            if (!converged) {
                neighbours.clear();
                ngb_search(tree, particles, pos_target, h, neighbours, box, lazy);
                rho = 0.0; n_inside = 0;
                for (uint32_t j : neighbours) {
                    if (!particles.is_gas(j)) continue;   // gas density counts GAS neighbours only
                    const double r = min_image(particles.pos(j) - pos_target, box).norm();
                    if (r < h) ++n_inside;
                    rho += particles.m[j] * kernel_w(r, h, n_dims);
                }
            }
            if (cache) {
                local_index[tid].emplace_back(t, local_flat[tid].size());
                local_flat[tid].insert(local_flat[tid].end(), neighbours.begin(), neighbours.end());
                cache->start[t + 1] = neighbours.size();
            }
            result.h[t] = h; result.rho[t] = rho;
            result.nngb[t] = n_inside; result.iters[t] = iter;
        }
    }

    if (cache) {
        for (size_t t = 0; t < n_targets; ++t) cache->start[t + 1] += cache->start[t];
        cache->flat.resize(cache->start.back());
        #pragma omp parallel for schedule(static)
        for (int th = 0; th < cache_threads; ++th) {
            const std::vector<uint32_t>& my_flat = local_flat[th];
            const std::vector<std::pair<size_t, size_t>>& my_index = local_index[th];
            for (size_t e = 0; e < my_index.size(); ++e) {
                const size_t t = my_index[e].first, lo = my_index[e].second;
                const size_t hi = (e + 1 < my_index.size()) ? my_index[e + 1].second
                                                            : my_flat.size();
                std::copy(my_flat.begin() + lo, my_flat.begin() + hi,
                          cache->flat.begin() + cache->start[t]);
            }
        }
        cache->valid = true;
    }
    return result;
}

}  // namespace shmem
