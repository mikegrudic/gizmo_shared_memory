#include "hydro.h"
#include <omp.h>
#include <atomic>

namespace shmem {

// SHMEM_NGB_COUNT: nodes visited and particles EXAMINED per target, against the ~DesNumNgb
// actually kept. The neighbour analogue of SHMEM_COUNT_WALK, and here for the same reason: a
// traversal that examines hundreds of candidates to keep 32 is doing the wrong work, and no
// per-phase timer can tell you that. Accumulated locally and flushed with one atomic per call, so
// a counting run is still fast enough to be worth running.
std::atomic<long long> g_ngb_nodes{0}, g_ngb_examined{0}, g_ngb_kept{0}, g_ngb_calls{0};
// Nodes visited ONLY because of the per-node vmax*elapsed pad. Measured 0.0% of visits on the
// M50 state at every step type, so a stale tree is NOT what the search is paying for -- kept
// because that is a property of the problem's velocities, not a guarantee.
std::atomic<long long> g_ngb_pad_nodes{0};
// Iteration-count histogram for the h solve, same gate: slot i = targets that took i+1
// iterations (last slot = 8 or more, which includes the non-converged 100-iteration escape).
// The MEAN multiplier hides a fat tail; this is the tail check.
std::atomic<long long> g_hiter_hist[8] = {};
void hiter_counters(long long out[8], bool reset) {
    for (int i = 0; i < 8; ++i) { out[i] = g_hiter_hist[i].load(); if (reset) g_hiter_hist[i] = 0; }
}
static bool ngb_counting() {
    static const bool v = (getenv("SHMEM_NGB_COUNT") != nullptr);
    return v;
}
void ngb_counters(long long& calls, long long& nodes, long long& examined, long long& kept,
                  long long& pad_nodes, bool reset) {
    calls = g_ngb_calls.load(); nodes = g_ngb_nodes.load();
    examined = g_ngb_examined.load(); kept = g_ngb_kept.load();
    pad_nodes = g_ngb_pad_nodes.load();
    if (reset) { g_ngb_calls = 0; g_ngb_nodes = 0; g_ngb_examined = 0; g_ngb_kept = 0;
                 g_ngb_pad_nodes = 0; }
}

void ngb_search(const Tree& tree, const Particles& particles, const Vec3d& centre,
                double radius, std::vector<uint32_t>& found, double box,
                const LazyDrift* lazy) {
    // Prunes against the node's BOX, as the reference does (system/ngb_codeblock_checknode.h):
    // three per-axis comparisons with early exit, then the circumsphere test only for survivors.
    // Was a single sphere centred on the node's CENTRE OF MASS with radius size+delta, which admits
    // ~2x the volume (bench_nodegeom) and computes a full 3D norm for every node visited. The
    // traversal is 75% of the density solve at ~12 ns per node visit, so both the count and the
    // bytes per visit are worth this.
    const SNode* __restrict nodes = tree.sn.data();
    const float* __restrict node_vmax = tree.vmax.data();
    const double elapsed = tree.t_since_build;
    const double radius_sq = radius * radius;
    const bool counting = ngb_counting();
    long long c_nodes = 0, c_examined = 0;
    long long c_pad = 0;   // nodes admitted only by the staleness pad; see below
    const size_t found0 = found.size();
    int node_id = tree.root;
    while (node_id >= 0) {
        const SNode& node = nodes[node_id];
        if (counting) ++c_nodes;
        // Half the side, plus this node's own vmax bound on how far its particles can have moved
        // since the build. Per node rather than a global pad -- one fast particle must not inflate
        // the prune for the whole box (see Tree::vmax). Skipped entirely on a fresh tree.
        double dist = radius + (double)node.half;
        if (elapsed > 0) dist += (double)node_vmax[node_id] * elapsed;
        const double dx = min_image((double)node.cx - centre[0], box);
        if (dx > dist || -dx > dist) { node_id = node.next; continue; }
        const double dy = min_image((double)node.cy - centre[1], box);
        if (dy > dist || -dy > dist) { node_id = node.next; continue; }
        const double dz = min_image((double)node.cz - centre[2], box);
        if (dz > dist || -dz > dist) { node_id = node.next; continue; }
        // Only now the sphere: the box's circumsphere is sqrt(3)/2 * len = 1.732 * half, i.e.
        // dist + (sqrt(3)-1) * half on top of the per-axis bound (CUBE_EDGEFACTOR_1 * len).
        dist += 0.7320508075688772 * (double)node.half;
        if (dx*dx + dy*dy + dz*dz > dist*dist) { node_id = node.next; continue; }
        if (counting && elapsed > 0) {
            const double nopad = radius + 1.7320508075688772 * (double)node.half;
            if (dx*dx + dy*dy + dz*dz > nopad * nopad) ++c_pad;
        }
        if (node.first < 0) {
            // Two spellings of the same loop so the common (nothing stale) path keeps exactly the
            // instructions it had before lazy drift existed.
            if (lazy) {
                if (counting) c_examined += node.phi - node.plo;
                for (int slot = node.plo; slot < node.phi; ++slot) {
                    const uint32_t j = tree.orderbuf[slot];
                    // catch up BEFORE the distance test, as GIZMO does: testing a stale position
                    // would let a true neighbour fall outside the radius and be dropped
                    if (lazy->last[j] < lazy->target) lazy->catch_up(lazy->ctx, j);
                    if (min_image(particles.pos(j) - centre, box).norm_sq() < radius_sq)
                        found.push_back(j);
                }
            } else {
                if (counting) c_examined += node.phi - node.plo;
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
    if (counting) {
        g_ngb_calls += 1; g_ngb_nodes += c_nodes; g_ngb_examined += c_examined;
        g_ngb_kept += (long long)(found.size() - found0); g_ngb_pad_nodes += c_pad;
    }
}

DensityResult density(const Tree& tree, const Particles& particles,
                      const std::vector<uint32_t>& targets, double des_ngb, double ngb_tol,
                      const std::vector<double>& h_start, double box, int n_dims,
                      NeighborCache* cache, const LazyDrift* lazy) {
    static const bool no_face_corr = getenv("SHMEM_NO_FACECORR") != nullptr;  // A/B switch
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
                SymTensor3d moments{0,0,0,0,0,0};   // E_i, for the face-closure correction below
                Vec3d face_w{0, 0, 0};              // sum_j dx W_ij
                for (uint32_t j : neighbours) {
                    if (!particles.is_gas(j)) continue;   // gas h counts GAS neighbours only
                    const Vec3d dx = min_image(particles.pos(j) - pos_target, box);
                    const double r = dx.norm();
                    const double w = kernel_w(r, h, n_dims);
                    weight_sum += w;
                    dweight_dh += kernel_dwdh(r, h, n_dims);
                    rho += particles.m[j] * w;
                    moments += outer_product(dx) * w;
                    face_w += dx * w;
                    if (r < h) ++n_inside;
                }
                const double n_eff = ball_vol(h, n_dims) * weight_sum;

                // FACE-CLOSURE CORRECTION (density.cc:565-598). The effective faces around a cell
                // should sum to zero; what is left of the one-sided estimator 2 V B . (sum_j dx W),
                // divided by the kernel-weighted rms neighbour distance, is a dimensionless leak.
                // Past 0.35 the reference widens the kernel in proportion, up to 2x, and solves for
                // THAT many neighbours instead of DesNumNgb. It fires on well under 1% of cells --
                // the density peak and the free surface -- but those are exactly the cells that set
                // rho_max, and (since gas gravitational softening is h) the short-range gravity
                // there. Without it a collapsing core is resolved with a kernel ~1.3x too narrow.
                double des_eff = des_ngb, tol_eff = ngb_tol;
                if (!no_face_corr && weight_sum > 0) {
                    Mat3d B;
                    if (invert_moments(moments, B, n_dims)) {
                        const double vol = 1.0 / weight_sum;
                        const double dx_i = std::sqrt(vol * moments.trace());
                        const Vec3d leak = B.matvec(face_w) * (2.0 * vol);
                        double sum_abs = 0.0;
                        for (int c = 0; c < 3; ++c) sum_abs += std::abs(leak[c]) / n_dims;
                        const double denom = 2.0 * n_dims * std::pow(dx_i, n_dims - 1);
                        if (denom > 0) {
                            const double fce = sum_abs / denom;
                            const double ncorr = std::min(std::max(fce / 0.35, 1.0), 2.0);
                            des_eff = des_ngb * ncorr; tol_eff = ngb_tol * ncorr;
                        }
                    }
                }
                const double residual = n_eff - des_eff;
                if (std::abs(residual) < tol_eff) { converged = true; break; }
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
            if (ngb_counting()) {
                const int used = converged ? iter + 1 : 100;
                g_hiter_hist[std::min(used, 8) - 1].fetch_add(1, std::memory_order_relaxed);
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
