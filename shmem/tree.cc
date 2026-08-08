#include "tree.h"
#include <chrono>
#include <cstdio>
#include <parallel/algorithm>

static double now_ms(){using c=std::chrono::steady_clock;
    return std::chrono::duration<double,std::milli>(c::now().time_since_epoch()).count();}

namespace shmem {

// Softened point-mass acceleration, Plummer-equivalent. Kept trivial for now: the validation target
// is the TREE (opening criterion, traversal, moments), not the force kernel, and a spline kernel can
// be dropped in later without touching the traversal.
static inline void kick(double dx, double dy, double dz, double m, double eps2,
                        double& ax, double& ay, double& az) {
    double r2 = dx*dx + dy*dy + dz*dz + eps2;
    double inv = 1.0 / (r2 * std::sqrt(r2));      // 1/r^3
    ax += m * dx * inv; ay += m * dy * inv; az += m * dz * inv;
}

int build_node(Tree& T, const Particles& P, const std::vector<uint32_t>& order,
               const std::vector<uint64_t>& key, int lo, int hi, int level,
               double cxi, double cyi, double czi, double sz) {
    // Lock-free allocation: arrays are pre-sized (worst case ~2N/LEAF_MAX interior+leaf nodes,
    // bounded by 2N), so claiming a node is one atomic increment. The earlier critical section
    // around ten push_backs serialised every task on one lock and made the parallel build 3x SLOWER
    // than the serial one.
    int me;
    #pragma omp atomic capture
    me = T.nalloc++;
    T.size[me] = sz; T.first[me] = -1; T.next[me] = -1; T.plo[me] = lo; T.phi[me] = hi;

    bool leaf = (hi - lo <= LEAF_MAX) || (level >= MAX_LEVEL);
    int kids[8]; int nk = 0;
    if (!leaf) {
        // Children are contiguous in Morton order: the 3 bits at this level select the octant, so a
        // single scan splits [lo,hi) into up to 8 ranges. No insertion, no rebalancing.
        int shift = 3 * (MAX_LEVEL - level - 1);
        int bounds[9]; bounds[0] = lo;
        int at = lo;
        for (int oct = 0; oct < 8; ++oct) {
            while (at < hi && (int)((key[at] >> shift) & 7ull) == oct) ++at;
            bounds[oct + 1] = at;
        }
        double h = sz * 0.5, q = sz * 0.25;
        // Octant ranges are disjoint, so subtrees can build as OpenMP tasks. Only worthwhile for
        // large ranges: below the cutoff, task overhead exceeds the work. Node ALLOCATION still
        // serialises in the critical section; the arena is shared. kids[] order must stay
        // deterministic, so each task writes its own slot.
        const bool spawn = (hi - lo > 20000);
        int slot[8];
        for (int oct = 0; oct < 8; ++oct) {
            int a = bounds[oct], b = bounds[oct + 1];
            if (b <= a) continue;
            slot[nk] = oct; kids[nk] = -1; ++nk;
        }
        for (int i = 0; i < nk; ++i) {
            int oct = slot[i];
            int a = bounds[oct], b = bounds[oct + 1];
            double ox = cxi + ((oct & 1) ? q : -q);
            double oy = cyi + ((oct & 2) ? q : -q);
            double oz = czi + ((oct & 4) ? q : -q);
            if (spawn) {
                #pragma omp task shared(T, P, order, key, kids) firstprivate(a, b, level, ox, oy, oz, h, i)
                kids[i] = build_node(T, P, order, key, a, b, level + 1, ox, oy, oz, h);
            } else {
                kids[i] = build_node(T, P, order, key, a, b, level + 1, ox, oy, oz, h);
            }
        }
        if (spawn) {
            #pragma omp taskwait
        }
        if (nk == 0) { leaf = true; }
        else {
            T.first[me] = kids[0];
            for (int i = 0; i + 1 < nk; ++i) T.next[kids[i]] = kids[i + 1];
            T.next[kids[nk - 1]] = -1;               // patched to the uncle in setup_walk
        }
    }

    // Moments. Leaves scan their particles; interior nodes COMBINE their children's already-final
    // moments, so the total moment work is O(N) instead of O(N * depth) -- the old version re-scanned
    // the full particle range at every level, which at depth ~15 was most of the recursion cost.
    double M = 0, sx = 0, sy = 0, sz_ = 0, smax = 0;
    if (T.first[me] < 0 || nk == 0) {
        for (int i = lo; i < hi; ++i) {
            uint32_t p = order[i];
            double m = P.m[p];
            M += m; sx += m * P.x[p]; sy += m * P.y[p]; sz_ += m * P.z[p];
            if (!P.soft.empty() && P.soft[p] > smax) smax = P.soft[p];
        }
    } else {
        for (int i = 0; i < nk; ++i) {
            int k = kids[i];
            double m = T.mass[k];
            M += m; sx += m * T.cx[k]; sy += m * T.cy[k]; sz_ += m * T.cz[k];
            if (T.soft[k] > smax) smax = T.soft[k];
        }
    }
    if (M > 0) { sx /= M; sy /= M; sz_ /= M; }
    T.mass[me] = M; T.cx[me] = sx; T.cy[me] = sy; T.cz[me] = sz_; T.soft[me] = smax;
    double dx = sx - cxi, dy = sy - cyi, dz = sz_ - czi;
    T.delta[me] = std::sqrt(dx*dx + dy*dy + dz*dz);
    return me;
}

void setup_walk(Tree& T, int node, int next_sibling) {
    T.next[node] = next_sibling;
    int c = T.first[node];
    while (c >= 0) {
        int sib = T.next[c];
        setup_walk(T, c, sib >= 0 ? sib : next_sibling);   // last child falls through to the uncle
        c = sib;
    }
}

Tree build(const Particles& P, BuildTimes* bt) {
    double t_a = now_ms(), t_start = t_a;
    const size_t n = P.size();
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    #pragma omp parallel for reduction(min:lo[:3]) reduction(max:hi[:3]) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        lo[0] = std::min(lo[0], P.x[i]); hi[0] = std::max(hi[0], P.x[i]);
        lo[1] = std::min(lo[1], P.y[i]); hi[1] = std::max(hi[1], P.y[i]);
        lo[2] = std::min(lo[2], P.z[i]); hi[2] = std::max(hi[2], P.z[i]);
    }
    if(bt) { bt->bbox = now_ms()-t_a; } t_a = now_ms();
    double cx = 0.5*(lo[0]+hi[0]), cy = 0.5*(lo[1]+hi[1]), cz = 0.5*(lo[2]+hi[2]);
    double side = std::max(hi[0]-lo[0], std::max(hi[1]-lo[1], hi[2]-lo[2])) * 1.0000001;
    if (side <= 0) side = 1.0;

    const double scale = ((1u << MAX_LEVEL) - 1) / side;
    std::vector<uint64_t> key(n);
    std::vector<uint32_t> order(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        uint32_t a = (uint32_t)((P.x[i] - (cx - 0.5*side)) * scale);
        uint32_t b = (uint32_t)((P.y[i] - (cy - 0.5*side)) * scale);
        uint32_t c = (uint32_t)((P.z[i] - (cz - 0.5*side)) * scale);
        key[i] = morton(a, b, c);
        order[i] = (uint32_t)i;
    }
    if(bt) { bt->keys = now_ms()-t_a; } t_a = now_ms();
    __gnu_parallel::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) { return key[a] < key[b]; });
    if(bt) { bt->sort = now_ms()-t_a; } t_a = now_ms();
    std::vector<uint64_t> skey(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) skey[i] = key[order[i]];

    Tree T;
    {   // pre-size: <= one node per LEAF_MAX particles at the bottom + interior ~ 8/7 of that; 2x
        // headroom on top because Morton splits can be uneven. Trimmed to nalloc after the build.
        size_t cap = (2 * n) / LEAF_MAX * 3 + 1024;
        T.cx.resize(cap); T.cy.resize(cap); T.cz.resize(cap);
        T.mass.resize(cap); T.size.resize(cap); T.delta.resize(cap); T.soft.resize(cap);
        T.first.resize(cap); T.next.resize(cap); T.plo.resize(cap); T.phi.resize(cap);
    }
    #pragma omp parallel
    #pragma omp single
    T.root = build_node(T, P, order, skey, 0, (int)n, 0, cx, cy, cz, side);
    {   // trim to what was allocated
        size_t nn = (size_t)T.nalloc;
        T.cx.resize(nn); T.cy.resize(nn); T.cz.resize(nn); T.mass.resize(nn); T.size.resize(nn);
        T.delta.resize(nn); T.soft.resize(nn); T.first.resize(nn); T.next.resize(nn);
        T.plo.resize(nn); T.phi.resize(nn);
    }
    if(bt) { bt->recurse = now_ms()-t_a; } t_a = now_ms();
    setup_walk(T, T.root, -1);
    // Leaf ranges index the Morton-sorted order, so the tree owns it: a leaf's particles are
    // orderbuf[plo..phi), contiguous by construction.
    T.orderbuf.swap(order);

    if(bt) { bt->links = now_ms()-t_a; } t_a = now_ms();
    // Pack the traversal copy: one 64-byte line per node instead of 9 scattered arrays.
    T.wn.resize(T.nnodes());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < T.nnodes(); ++i) {
        WNode& w = T.wn[i];
        w.cx = T.cx[i]; w.cy = T.cy[i]; w.cz = T.cz[i];
        w.s  = T.size[i] + T.delta[i];
        w.mass = T.mass[i]; w.soft = (float)T.soft[i];
        w.first = T.first[i]; w.next = T.next[i];
        w.plo = T.plo[i]; w.phi = T.phi[i];
    }
    if(bt) { bt->pack = now_ms()-t_a; bt->total = now_ms()-t_start; }
    return T;
}

void accel(const Tree& T, const Particles& P, const std::vector<uint32_t>& targets,
           double theta, double G, std::vector<double>& ax, std::vector<double>& ay,
           std::vector<double>& az) {
    const size_t nt = targets.size();
    ax.assign(nt, 0.0); ay.assign(nt, 0.0); az.assign(nt, 0.0);
    const double theta2 = theta * theta;

    // Plain parallel-for over the ACTIVE list. step_overhead showed this beats a persistent pool
    // (4x) and beats capping the width (2.5x) in exactly this regime.
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t t = 0; t < nt; ++t) {
        const uint32_t p = targets[t];
        const double px = P.x[p], py = P.y[p], pz = P.z[p];
        const double eps = P.soft.empty() ? 0.0 : P.soft[p];
        double sx = 0, sy = 0, sz = 0;

        const WNode* __restrict W = T.wn.data();
        int node = T.root;
        while (node >= 0) {
            const WNode& w = W[node];                     // ONE cache line per visit
            double dx = w.cx - px, dy = w.cy - py, dz = w.cz - pz;
            double r2 = dx*dx + dy*dy + dz*dz;
            if (w.first < 0 || w.s * w.s < theta2 * r2) {
                if (w.first < 0) {                        // leaf: direct sum over its particles
                    for (int i = w.plo; i < w.phi; ++i) {
                        uint32_t q = T.orderbuf[i];
                        if (q == p) continue;
                        double e2 = std::max(eps, P.soft.empty() ? 0.0 : P.soft[q]);
                        kick(P.x[q]-px, P.y[q]-py, P.z[q]-pz, P.m[q], e2*e2, sx, sy, sz);
                    }
                } else {                                   // far enough: use the monopole
                    double e2 = std::max(eps, (double)w.soft);
                    kick(dx, dy, dz, w.mass, e2*e2, sx, sy, sz);
                }
                node = w.next;
            } else {
                node = w.first;                            // too close: descend
            }
        }
        ax[t] = G * sx; ay[t] = G * sy; az[t] = G * sz;
    }
}

void accel_soa(const Tree& T, const Particles& P, const std::vector<uint32_t>& targets,
           double theta, double G, std::vector<double>& ax, std::vector<double>& ay,
           std::vector<double>& az) {
    const size_t nt = targets.size();
    ax.assign(nt, 0.0); ay.assign(nt, 0.0); az.assign(nt, 0.0);
    const double theta2 = theta * theta;

    // Plain parallel-for over the ACTIVE list. step_overhead showed this beats a persistent pool
    // (4x) and beats capping the width (2.5x) in exactly this regime.
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t t = 0; t < nt; ++t) {
        const uint32_t p = targets[t];
        const double px = P.x[p], py = P.y[p], pz = P.z[p];
        const double eps = P.soft.empty() ? 0.0 : P.soft[p];
        double sx = 0, sy = 0, sz = 0;

        int node = T.root;
        while (node >= 0) {
            double dx = T.cx[node] - px, dy = T.cy[node] - py, dz = T.cz[node] - pz;
            double r2 = dx*dx + dy*dy + dz*dz;
            double s = T.size[node] + T.delta[node];
            if (T.first[node] < 0 || s * s < theta2 * r2) {
                if (T.first[node] < 0) {
                    for (int i = T.plo[node]; i < T.phi[node]; ++i) {
                        uint32_t q = T.orderbuf[i];
                        if (q == p) continue;
                        double e2 = std::max(eps, P.soft.empty() ? 0.0 : P.soft[q]);
                        kick(P.x[q]-px, P.y[q]-py, P.z[q]-pz, P.m[q], e2*e2, sx, sy, sz);
                    }
                } else {
                    double e2 = std::max(eps, T.soft[node]);
                    kick(dx, dy, dz, T.mass[node], e2*e2, sx, sy, sz);
                }
                node = T.next[node];
            } else {
                node = T.first[node];
            }
        }
        ax[t] = G * sx; ay[t] = G * sy; az[t] = G * sz;
    }
}

void accel_grouped(const Tree& T, const Particles& P, const std::vector<uint32_t>& targets,
                   double theta, double G, int batch, std::vector<double>& ax,
                   std::vector<double>& ay, std::vector<double>& az) {
    const size_t nt = targets.size();
    ax.assign(nt, 0.0); ay.assign(nt, 0.0); az.assign(nt, 0.0);
    const double theta2 = theta * theta;
    const int nb = (int)((nt + batch - 1) / batch);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int b = 0; b < nb; ++b) {
        size_t lo = (size_t)b * batch, hi = std::min(lo + batch, nt);
        // batch bounding box
        double bx0=1e300,by0=1e300,bz0=1e300,bx1=-1e300,by1=-1e300,bz1=-1e300, emax=0;
        for (size_t i = lo; i < hi; ++i) {
            uint32_t p = targets[i];
            bx0=std::min(bx0,P.x[p]); bx1=std::max(bx1,P.x[p]);
            by0=std::min(by0,P.y[p]); by1=std::max(by1,P.y[p]);
            bz0=std::min(bz0,P.z[p]); bz1=std::max(bz1,P.z[p]);
            if (!P.soft.empty()) emax = std::max(emax, P.soft[p]);
        }
        // Gather the batch into contiguous local buffers. Without this the inner loop gathers
        // P.x[targets[i]] for every node, replacing one node fetch per target with `batch` scattered
        // particle fetches -- measured 3-10x SLOWER than the per-target walk. Contiguous buffers keep
        // the batch in L1 and let the inner loop vectorise, which is the whole point of grouping.
        const int nb_ = (int)(hi - lo);
        double tx[512], ty[512], tz[512], te[512], oax[512], oay[512], oaz[512];
        for (int i = 0; i < nb_; ++i) {
            uint32_t p = targets[lo + i];
            tx[i]=P.x[p]; ty[i]=P.y[p]; tz[i]=P.z[p];
            te[i]=P.soft.empty()?0.0:P.soft[p];
            oax[i]=0; oay[i]=0; oaz[i]=0;
        }
        const WNode* __restrict W = T.wn.data();
        int node = T.root;
        while (node >= 0) {
            const WNode& w = W[node];
            double dx = std::max(0.0, std::max(bx0 - w.cx, w.cx - bx1));
            double dy = std::max(0.0, std::max(by0 - w.cy, w.cy - by1));
            double dz = std::max(0.0, std::max(bz0 - w.cz, w.cz - bz1));
            double rmin2 = dx*dx + dy*dy + dz*dz;
            if (w.first < 0 || w.s * w.s < theta2 * rmin2) {
                if (w.first < 0) {
                    for (int k = w.plo; k < w.phi; ++k) {
                        uint32_t q = T.orderbuf[k];
                        double qx=P.x[q], qy=P.y[q], qz=P.z[q], qm=P.m[q];
                        double qs = P.soft.empty()?0.0:P.soft[q];
                        for (int i = 0; i < nb_; ++i) {
                            if (targets[lo+i] == q) continue;
                            double e = std::max(te[i], qs);
                            kick(qx-tx[i], qy-ty[i], qz-tz[i], qm, e*e, oax[i], oay[i], oaz[i]);
                        }
                    }
                } else {
                    for (int i = 0; i < nb_; ++i) {
                        double e = std::max(te[i], (double)w.soft);
                        kick(w.cx-tx[i], w.cy-ty[i], w.cz-tz[i], w.mass, e*e, oax[i], oay[i], oaz[i]);
                    }
                }
                node = w.next;
            } else {
                node = w.first;
            }
        }
        for (int i = 0; i < nb_; ++i) { ax[lo+i]=G*oax[i]; ay[lo+i]=G*oay[i]; az[lo+i]=G*oaz[i]; }

    }
}

void accel_brute(const Particles& P, const std::vector<uint32_t>& targets, double G,
                 std::vector<double>& ax, std::vector<double>& ay, std::vector<double>& az) {
    const size_t nt = targets.size(), n = P.size();
    ax.assign(nt, 0.0); ay.assign(nt, 0.0); az.assign(nt, 0.0);
    #pragma omp parallel for schedule(static)
    for (size_t t = 0; t < nt; ++t) {
        uint32_t p = targets[t];
        double px = P.x[p], py = P.y[p], pz = P.z[p];
        double eps = P.soft.empty() ? 0.0 : P.soft[p];
        double sx = 0, sy = 0, sz = 0;
        for (size_t q = 0; q < n; ++q) {
            if (q == p) continue;
            double e2 = std::max(eps, P.soft.empty() ? 0.0 : P.soft[q]);
            kick(P.x[q]-px, P.y[q]-py, P.z[q]-pz, P.m[q], e2*e2, sx, sy, sz);
        }
        ax[t] = G*sx; ay[t] = G*sy; az[t] = G*sz;
    }
}

}  // namespace shmem
