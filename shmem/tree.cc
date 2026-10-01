#include "tree.h"
#include "hydro.h"    // kernel_dwdr, for the adaptive-softening zeta correction
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cfloat>
#include <parallel/algorithm>

static double now_ms(){using c=std::chrono::steady_clock;
    return std::chrono::duration<double,std::milli>(c::now().time_since_epoch()).count();}

namespace shmem {

// spline_force_over_r and spline_potential live in tree.h, beside grav_tidal_factor.

// SHMEM_TREE_NODELTA, read once. See the use site in the node-packing loop.
static bool nodelta_opening() {
    static const bool v = (getenv("SHMEM_TREE_NODELTA") != nullptr);
    return v;
}

static inline void kick(const Vec3d& offset, double mass, double softening, Vec3d& accel) {
    const double r = offset.norm();
    if (r <= 0) return;
    accel += offset * (mass * spline_force_over_r(r, std::max(softening, 1e-300)));
}

// Scalar force factor for a PARTICLE-PARTICLE pair, target <- source, as GIZMO's forcetree.cc
// computes it under ADAPTIVE_GRAVSOFT_FORGAS. Every pair uses the kernel of the LARGER softening:
// the averaged kernel (ADAPTIVE_GRAVSOFT_SYMMETRIZE_FORCE_BY_AVERAGING) is compiled only under
// ADAPTIVE_GRAVSOFT_FORALL (precompiler_logic.h:124). Averaging lets the smaller softening through
// inside the kernel, i.e. stronger gravity below the hydro resolution.
//
// Gas-gas pairs add the zeta correction (forcetree.cc:2194-2214): the target's term uses the pair
// softening max(et,es), the source's term its own softening, each gated on r inside that kernel,
// both divided by the TARGET's mass. Nodes get max-softening and no zeta.
//
// `base_out`, when non-null, receives the zeta-free force factor the tidal tensor and jerk need.
static inline double pair_force_over_r(double r, double mass_source,
                                       double eps_target, double eps_source,
                                       double zeta_target, double zeta_source,
                                       double mass_target, bool gas_gas,
                                       double* base_out = nullptr) {
    const double e = std::max(std::max(eps_target, eps_source), 1e-300);
    const double base = mass_source * spline_force_over_r(r, e);
    if (base_out) *base_out = base;
    if (!gas_gas || mass_target <= 0) return base;
    double fac = base;
    const double es = std::max(eps_source, 1e-300);
    if (zeta_target != 0.0 && r < e)  fac -= (zeta_target / mass_target) * kernel_dwdr(r, e, 3) / r;
    if (zeta_source != 0.0 && r < es) fac -= (zeta_source / mass_target) * kernel_dwdr(r, es, 3) / r;
    return fac;
}

// Component form, for the grouped walk ONLY. That loop keeps its batch in SoA scratch buffers, so
// it wants components rather than a Vec3d it would immediately take apart. Note this is NOT a
// vectorisation argument -- see the gather comment in the walk; the pair loop does not vectorise.
static inline void kick(double offset_x, double offset_y, double offset_z,
                        double mass, double softening,
                        double& accel_x, double& accel_y, double& accel_z) {
    const double r_sq = offset_x*offset_x + offset_y*offset_y + offset_z*offset_z;
    if (r_sq <= 0) return;
    const double force_over_r =
        mass * spline_force_over_r(std::sqrt(r_sq), std::max(softening, 1e-300));
    accel_x += offset_x * force_over_r;
    accel_y += offset_y * force_over_r;
    accel_z += offset_z * force_over_r;
}

int build_node(Tree& T, const Particles& P, const std::vector<uint32_t>& order,
               const std::vector<uint64_t>& key, int lo, int hi, int level,
               double cxi, double cyi, double czi, double sz, const double* const* vel) {
    // Lock-free allocation: arrays are pre-sized (worst case ~2N/LEAF_MAX interior+leaf nodes,
    // bounded by 2N), so claiming a node is one atomic increment. Do NOT replace this with a
    // critical section around push_backs on the node arrays -- that serialises every task on one
    // lock and measured 3x SLOWER than building the tree serially.
    int me;
    #pragma omp atomic capture
    me = T.nalloc++;
    // REFUSE rather than overrun. The arena is pre-sized from a heuristic tuned on roughly uniform
    // fills, and a heuristic can be wrong by a lot: a cloud occupying a thousandth of its box (the
    // MakeCloud convention -- R=0.1875 inside box=1.875) must descend several levels before it
    // reaches any mass, and turbulent clumping adds more, so it needs far more nodes per particle.
    // Writing past the end silently corrupts the heap, which surfaces as an unrelated free() abort
    // later. nalloc keeps counting past the end on purpose: build() reads it to size the retry.
    if ((size_t)me >= T.size.size()) return -1;
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
                kids[i] = build_node(T, P, order, key, a, b, level + 1, ox, oy, oz, h, vel);
            } else {
                kids[i] = build_node(T, P, order, key, a, b, level + 1, ox, oy, oz, h, vel);
            }
        }
        if (spawn) {
            #pragma omp taskwait
        }
        // Drop any child the arena refused. Everything below is index arithmetic on kids[], and a
        // -1 would be written straight into T.parent[]. This attempt's tree is already garbage --
        // build() is going to throw it away and retry -- but it must not corrupt memory on the way
        // out.
        { int keep = 0;
          for (int i = 0; i < nk; ++i) if (kids[i] >= 0) kids[keep++] = kids[i];
          nk = keep; }
        if (nk == 0) { leaf = true; }
        else {
            for (int i = 0; i < nk; ++i) T.parent[kids[i]] = me;
            T.first[me] = kids[0];
            for (int i = 0; i + 1 < nk; ++i) T.next[kids[i]] = kids[i + 1];
            T.next[kids[nk - 1]] = -1;               // patched to the uncle in setup_walk
        }
    }

    // Moments. Leaves scan their particles; interior nodes COMBINE their children's already-final
    // moments, so the total moment work is O(N) instead of O(N * depth) -- the old version re-scanned
    // the full particle range at every level, which at depth ~15 was most of the recursion cost.
    // The per-node velocity bound (Tree::vmax) accumulates the same way, in the same sweep.
    // The per-node centre-of-mass VELOCITY (Tree::vcom) rides in the same sweep. It is GIZMO's
    // Extnodes[].vs, and it is what a node interaction contributes to the JERK
    // (forcetree.cc:1945, dv = Extnodes[no].vs - vel); leaves use the particle's own velocity
    // (forcetree.cc:1641). Only built when velocities are supplied.
    double M = 0, sx = 0, sy = 0, sz_ = 0, smax = 0, vmax = 0;
    double pvx = 0, pvy = 0, pvz = 0;               // mass-weighted momentum, for vcom
    uint32_t nsink = 0;                             // sinks below this node, for direct summation
    if (T.first[me] < 0 || nk == 0) {
        for (int i = lo; i < hi; ++i) {
            uint32_t p = order[i];
            double m = P.m[p];
            M += m; sx += m * P.x[p]; sy += m * P.y[p]; sz_ += m * P.z[p];
            if (!P.soft.empty() && P.soft[p] > smax) smax = P.soft[p];
            if (vel) {
                const double v = std::sqrt(vel[0][p]*vel[0][p] + vel[1][p]*vel[1][p] +
                                           vel[2][p]*vel[2][p]);
                if (v > vmax) vmax = v;
                pvx += m * vel[0][p]; pvy += m * vel[1][p]; pvz += m * vel[2][p];
            }
            if (!P.type.empty() && P.type[p] == 5) ++nsink;
            T.leaf_of[p] = me;
        }
    } else {
        for (int i = 0; i < nk; ++i) {
            int k = kids[i];
            double m = T.mass[k];
            M += m; sx += m * T.cx[k]; sy += m * T.cy[k]; sz_ += m * T.cz[k];
            if (T.soft[k] > smax) smax = T.soft[k];
            if (T.vmax[k] > vmax) vmax = T.vmax[k];
            nsink += T.nsink.empty() ? 0u : T.nsink[k];
            if (vel && !T.vcom_x.empty()) {
                pvx += m * T.vcom_x[k]; pvy += m * T.vcom_y[k]; pvz += m * T.vcom_z[k];
            }
        }
    }
    if (M > 0) { sx /= M; sy /= M; sz_ /= M; }
    T.mass[me] = M; T.cx[me] = sx; T.cy[me] = sy; T.cz[me] = sz_; T.soft[me] = smax;
    T.vmax[me] = (float)vmax;
    if (!T.nsink.empty()) T.nsink[me] = nsink;
    if (vel && !T.vcom_x.empty() && M > 0) {
        T.vcom_x[me] = pvx / M; T.vcom_y[me] = pvy / M; T.vcom_z[me] = pvz / M;
    }
    double dx = sx - cxi, dy = sy - cyi, dz = sz_ - czi;
    T.delta[me] = std::sqrt(dx*dx + dy*dy + dz*dz);
    // The GEOMETRIC centre, kept for the neighbour search's box test (see SNode). Free here --
    // it is what this node was constructed around.
    if (!T.gcx.empty()) { T.gcx[me] = cxi; T.gcy[me] = cyi; T.gcz[me] = czi; }
    return me;
}

// Atomic max on a NON-NEGATIVE float, done on its bit pattern: for x >= 0 the IEEE-754 encoding is
// monotonic in the value, so an unsigned integer compare is exactly a float compare. Saves needing
// a lock or a double-width CAS on the hot kick path. (GIZMO's atomic_max_double, same idea.)
static inline void atomic_max_nonneg(float* slot, float value) {
    uint32_t want; std::memcpy(&want, &value, sizeof want);
    uint32_t* bits = reinterpret_cast<uint32_t*>(slot);
    uint32_t seen = __atomic_load_n(bits, __ATOMIC_RELAXED);
    while (seen < want &&
           !__atomic_compare_exchange_n(bits, &seen, want, true,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) { /* seen reloaded */ }
}

void Tree::kick_node(int leaf_node, const Vec3d& dp, double tau) {
    if (dp_x.empty()) return;                    // no vcom built, nothing to keep current
    for (int no = leaf_node; no >= 0; no = parent[no]) {
        #pragma omp atomic
        dp_x[no] += dp[0];
        #pragma omp atomic
        dp_y[no] += dp[1];
        #pragma omp atomic
        dp_z[no] += dp[2];
        #pragma omp atomic
        dq_x[no] += dp[0] * tau;
        #pragma omp atomic
        dq_y[no] += dp[1] * tau;
        #pragma omp atomic
        dq_z[no] += dp[2] * tau;
    }
}

Vec3d Tree::node_com(int no, double tau) const {
    const double M = node_mass(no);
    if (dp_x.empty() || M <= 0) return Vec3d{cx[no], cy[no], cz[no]};
    const double m0 = mass[no];
    return Vec3d{(m0 * cx[no] + (m0 * vcom_x[no] + dp_x[no]) * tau - dq_x[no]) / M,
                 (m0 * cy[no] + (m0 * vcom_y[no] + dp_y[no]) * tau - dq_y[no]) / M,
                 (m0 * cz[no] + (m0 * vcom_z[no] + dp_z[no]) * tau - dq_z[no]) / M};
}

// Serial: called from the accretion and Hermite hand-back loops, one particle at a time.
void Tree::mass_event(int leaf_node, double dm, const Vec3d& x, const Vec3d& v, double tau,
                      double box) {
    if (dp_x.empty() || leaf_node < 0 || dm == 0.0) return;
    for (int no = leaf_node; no >= 0; no = parent[no]) {
        const Vec3d c = node_com(no, tau);
        const Vec3d xr = c + min_image(x - c, box);
        dmass[no] += dm;
        dp_x[no] += dm * v[0]; dp_y[no] += dm * v[1]; dp_z[no] += dm * v[2];
        dq_x[no] -= dm * (xr[0] - v[0] * tau);
        dq_y[no] -= dm * (xr[1] - v[1] * tau);
        dq_z[no] -= dm * (xr[2] - v[2] * tau);
    }
}

void Tree::shift_node(int leaf_node, double m, const Vec3d& dx) {
    if (dp_x.empty() || leaf_node < 0) return;
    for (int no = leaf_node; no >= 0; no = parent[no]) {
        dq_x[no] -= m * dx[0]; dq_y[no] -= m * dx[1]; dq_z[no] -= m * dx[2];
    }
}

void Tree::refresh_walk_nodes(double tau) {
    const bool nodelta = nodelta_opening();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nnodes(); ++i) {
        WNode& w = wn[i];
        const Vec3d c = node_com((int)i, tau);
        w.cx = c[0]; w.cy = c[1]; w.cz = c[2];
        w.mass = node_mass((int)i);
        // every particle can have moved vmax*tau and the COM with them, so a bound on the
        // COM-to-particle distance grows by 2*vmax*tau (forcetree_update.cc: len += 2*vmax*dt)
        const double grow = 2.0 * ((double)vmax[i] * tau + (double)xpad[i]);
        w.s   = size[i] + (nodelta ? 0.0 : delta[i]) + grow;
        w.len = (float)(size[i] + grow);
    }
}

void Tree::raise_vmax(int leaf_node, float speed) {
    uint32_t want; std::memcpy(&want, &speed, sizeof want);
    for (int no = leaf_node; no >= 0; no = parent[no]) {
        // A node's vmax is >= every descendant's, so the first ancestor that already covers this
        // speed guarantees all the ones above it do too. Almost every kick stops here.
        const uint32_t* bits = reinterpret_cast<const uint32_t*>(&vmax[no]);
        if (__atomic_load_n(bits, __ATOMIC_RELAXED) >= want) return;
        atomic_max_nonneg(&vmax[no], speed);
    }
}

// Rounded UP to float so the prune stays one-sided.
static inline float h_as_bound(double h) {
    float f = (float)h;
    return ((double)f < h) ? std::nextafter(f, 1e30f) : f;
}

void Tree::raise_hmax(int leaf_node, double h) {
    if (hmax.empty() || leaf_node < 0) return;
    const float hf = h_as_bound(h);
    uint32_t want; std::memcpy(&want, &hf, sizeof want);
    for (int no = leaf_node; no >= 0; no = parent[no]) {
        const uint32_t* bits = reinterpret_cast<const uint32_t*>(&hmax[no]);
        if (__atomic_load_n(bits, __ATOMIC_RELAXED) >= want) return;
        atomic_max_nonneg(&hmax[no], hf);
    }
}

static inline void atomic_max_double(double* slot, double value) {
    double seen; __atomic_load(slot, &seen, __ATOMIC_RELAXED);
    while (seen < value &&
           !__atomic_compare_exchange(slot, &seen, &value, true,
                                      __ATOMIC_RELAXED, __ATOMIC_RELAXED)) { /* seen reloaded */ }
}

void Tree::raise_soft(int leaf_node, double soft) {
    if (leaf_node < 0 || wn.empty()) return;
    const float sf = h_as_bound(soft);
    for (int no = leaf_node; no >= 0; no = parent[no]) {
        float cur; __atomic_load(&wn[no].soft, &cur, __ATOMIC_RELAXED);
        if (cur >= sf) return;
        atomic_max_nonneg(&wn[no].soft, sf);
        atomic_max_double(&this->soft[no], soft);
    }
}

// Serial callers (accretion, the Hermite hand-back), so plain stores suffice.
void Tree::raise_pad(int leaf_node, double d) {
    if (leaf_node < 0 || xpad.empty() || !(d > 0)) return;
    const float leaf_pad = h_as_bound((double)xpad[leaf_node] + d);
    for (int no = leaf_node; no >= 0; no = parent[no]) {
        if (xpad[no] >= leaf_pad) return;
        xpad[no] = leaf_pad;
    }
}

void Tree::init_hmax(const std::vector<double>& h, size_t n_gas) {
    hmax.assign(nnodes(), 0.0f);
    const size_t n = std::min(std::min(n_gas, h.size()), leaf_of.size());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i)
        if (h[i] > 0) raise_hmax(leaf_of[i], h[i]);
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

Tree build(const Particles& P, BuildTimes* bt, const double* const* vel, bool want_vcom,
           long long randomize_seed) {
    if (!vel) want_vcom = false;                  // no velocities, no centre-of-mass velocity
    double t_a = now_ms(), t_start = t_a;
    const size_t n = P.size();
    // Massless particles are tombstones (swallowed cells awaiting compaction): left out of the tree
    // entirely, with leaf_of = -1, so no walk or search can reach them.
    std::vector<uint32_t> order;
    order.reserve(n);
    for (size_t i = 0; i < n; ++i) if (P.m[i] > 0) order.push_back((uint32_t)i);
    const size_t n_live = order.size();
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    #pragma omp parallel for reduction(min:lo[:3]) reduction(max:hi[:3]) schedule(static)
    for (size_t r = 0; r < n_live; ++r) {
        const size_t i = order[r];
        lo[0] = std::min(lo[0], P.x[i]); hi[0] = std::max(hi[0], P.x[i]);
        lo[1] = std::min(lo[1], P.y[i]); hi[1] = std::max(hi[1], P.y[i]);
        lo[2] = std::min(lo[2], P.z[i]); hi[2] = std::max(hi[2], P.z[i]);
    }
    if(bt) { bt->bbox = now_ms()-t_a; } t_a = now_ms();
    double cx = 0.5*(lo[0]+hi[0]), cy = 0.5*(lo[1]+hi[1]), cz = 0.5*(lo[2]+hi[2]);
    double side = std::max(hi[0]-lo[0], std::max(hi[1]-lo[1], hi[2]-lo[2])) * 1.0000001;
    if (side <= 0) side = 1.0;
    // RANDOMIZE_GRAVTREE (domain.cc:2722-2731). Offset the root centre by up to half a side per
    // axis, then DOUBLE the side so the displaced box still covers every particle. That moves every
    // node wall in the tree, and a Barnes-Hut force error depends on where the walls fall relative
    // to the mass -- so redrawing per build decorrelates the error between steps. Uncorrelated
    // errors average out; a fixed grid's repeat and integrate into a secular drift.
    //
    // splitmix64 on the seed rather than rand(): reproducible, no global state, and no lock in what
    // may be called from a parallel region. The reference reseeds from the step number likewise.
    if (randomize_seed >= 0) {
        uint64_t s = (uint64_t)randomize_seed * 0x9E3779B97F4A7C15ull;
        auto next = [&s]() {
            uint64_t z = (s += 0x9E3779B97F4A7C15ull);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            return (double)((z ^ (z >> 31)) >> 11) * (1.0 / 9007199254740992.0);  // [0,1)
        };
        cx += side * (next() - 0.5);
        cy += side * (next() - 0.5);
        cz += side * (next() - 0.5);
        side *= 2.0;
    }
    // SHMEM_TREE_SHIFT="fx,fy,fz": displace the root box by these fractions of a side, which moves
    // every node boundary in the tree with it. The force error of a Barnes-Hut walk is a function
    // of WHERE the cell walls fall relative to the mass, so a grid-imprinted error rotates with
    // this while a physical one does not -- that is the test. It is also the mechanism behind
    // GIZMO's RANDOMIZE_GRAVTREE: re-drawing the offset decorrelates successive steps' errors so
    // they average out instead of accumulating into a secular drift.
    if (const char* s = getenv("SHMEM_TREE_SHIFT")) {
        double fx = 0, fy = 0, fz = 0;
        if (sscanf(s, "%lf,%lf,%lf", &fx, &fy, &fz) == 3) {
            cx += fx * side; cy += fy * side; cz += fz * side;
            // the box has to still cover every particle after the shift
            side *= 1.0 + 2.0 * (std::fabs(fx) + std::fabs(fy) + std::fabs(fz));
        }
    }

    const double scale = ((1u << MAX_LEVEL) - 1) / side;
    std::vector<uint64_t> key(n, 0);
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < n_live; ++r) {
        const size_t i = order[r];
        uint32_t a = (uint32_t)((P.x[i] - (cx - 0.5*side)) * scale);
        uint32_t b = (uint32_t)((P.y[i] - (cy - 0.5*side)) * scale);
        uint32_t c = (uint32_t)((P.z[i] - (cz - 0.5*side)) * scale);
        key[i] = morton(a, b, c);
    }
    if(bt) { bt->keys = now_ms()-t_a; } t_a = now_ms();
    __gnu_parallel::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) { return key[a] < key[b]; });
    if(bt) { bt->sort = now_ms()-t_a; } t_a = now_ms();
    std::vector<uint64_t> skey(n_live);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_live; ++i) skey[i] = key[order[i]];

    Tree T;
    // Pre-size: <= one node per LEAF_MAX particles at the bottom + interior ~ 8/7 of that; 2x
    // headroom on top because Morton splits can be uneven. Trimmed to nalloc after the build.
    //
    // That estimate assumes the particles roughly fill their box. When they do not it is far too
    // small -- a MakeCloud cloud fills a thousandth of its box volume and needs several levels of
    // nearly-empty nodes before it reaches any mass -- so the build can want more nodes than the
    // arena holds. It used to run off the end and corrupt the heap. Now the allocator refuses past
    // the end and we simply build again with a bigger arena. nalloc is only a LOWER bound on what
    // was needed (refused nodes never recursed), hence doubling rather than sizing exactly; two
    // attempts is the most this has ever taken.
    size_t cap = (2 * n_live) / LEAF_MAX * 3 + 1024;
    for (;;) {
        T.cx.assign(cap, 0.0); T.cy.assign(cap, 0.0); T.cz.assign(cap, 0.0);
        T.mass.assign(cap, 0.0); T.size.assign(cap, 0.0); T.delta.assign(cap, 0.0);
        T.gcx.assign(cap, 0.0); T.gcy.assign(cap, 0.0); T.gcz.assign(cap, 0.0);
        T.soft.assign(cap, 0.0);
        T.first.assign(cap, -1); T.next.assign(cap, -1);
        T.plo.assign(cap, 0); T.phi.assign(cap, 0);
        T.parent.assign(cap, -1); T.vmax.assign(cap, 0.0f); T.xpad.assign(cap, 0.0f);
        // Only when there are sinks to find; gas-only runs never allocate or touch this.
        if (!P.type.empty()) T.nsink.assign(cap, 0u);
        if (want_vcom) {
            T.vcom_x.assign(cap, 0.0); T.vcom_y.assign(cap, 0.0); T.vcom_z.assign(cap, 0.0);
            T.dp_x.assign(cap, 0.0);   T.dp_y.assign(cap, 0.0);   T.dp_z.assign(cap, 0.0);
            T.dq_x.assign(cap, 0.0);   T.dq_y.assign(cap, 0.0);   T.dq_z.assign(cap, 0.0);
            T.dmass.assign(cap, 0.0);
        }
        T.leaf_of.assign(n, -1);
        T.nalloc = 0;
        #pragma omp parallel
        #pragma omp single
        T.root = build_node(T, P, order, skey, 0, (int)n_live, 0, cx, cy, cz, side, vel);
        if ((size_t)T.nalloc <= cap) break;
        cap = (size_t)T.nalloc * 2 + 1024;
    }
    {   // trim to what was allocated
        size_t nn = (size_t)T.nalloc;
        T.cx.resize(nn); T.cy.resize(nn); T.cz.resize(nn); T.mass.resize(nn); T.size.resize(nn);
        T.delta.resize(nn); T.soft.resize(nn); T.first.resize(nn); T.next.resize(nn);
        T.gcx.resize(nn); T.gcy.resize(nn); T.gcz.resize(nn);
        T.plo.resize(nn); T.phi.resize(nn); T.parent.resize(nn); T.vmax.resize(nn); T.xpad.resize(nn);
        if (!T.nsink.empty()) T.nsink.resize(nn);
        if (!T.vcom_x.empty()) {
            T.vcom_x.resize(nn); T.vcom_y.resize(nn); T.vcom_z.resize(nn);
            T.dp_x.resize(nn);   T.dp_y.resize(nn);   T.dp_z.resize(nn);
            T.dq_x.resize(nn);   T.dq_y.resize(nn);   T.dq_z.resize(nn);   T.dmass.resize(nn);
        }
    }
    if(bt) { bt->recurse = now_ms()-t_a; } t_a = now_ms();
    setup_walk(T, T.root, -1);
    // Leaf ranges index the Morton-sorted order, so the tree owns it: a leaf's particles are
    // orderbuf[plo..phi), contiguous by construction.
    T.orderbuf.swap(order);
    // Inverse permutation, so a small active set can be put in tree order by sorting rather than
    // by scanning all N. One O(N) pass on a build that is already O(N log N).
    T.rank.assign(n, UINT32_MAX);
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < n_live; ++r) T.rank[T.orderbuf[r]] = (uint32_t)r;

    if(bt) { bt->links = now_ms()-t_a; } t_a = now_ms();
    // Pack the traversal copies. WNode (64 B) serves the gravity walk; SNode (32 B) serves the
    // neighbour search, which needs the node's box rather than its centre of mass.
    //
    // SAFETY EPSILON on the search node's half-side. Its centre is float, so it is off by up to
    // ~1 ulp of the COORDINATE's magnitude -- which for a compact system far from the origin can
    // exceed any fraction of the root size. Inflate `half` by 1e-6 of the root size plus 4 float ulps
    // of the centre's largest coordinate. The prune must only ever err towards opening --
    // rejecting a node that holds a true neighbour is a silent wrong answer, not a slow one.
    const double snode_eps = 1e-6 * (T.nnodes() ? T.size[T.root] : 1.0);
    T.sn.resize(T.nnodes());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < T.nnodes(); ++i) {
        SNode& sq = T.sn[i];
        sq.cx = (float)T.gcx[i]; sq.cy = (float)T.gcy[i]; sq.cz = (float)T.gcz[i];
        const double cmax = std::max(std::fabs(T.gcx[i]), std::max(std::fabs(T.gcy[i]), std::fabs(T.gcz[i])));
        sq.half  = (float)(0.5 * T.size[i] + snode_eps + 4.0 * FLT_EPSILON * cmax);
        sq.first = T.first[i]; sq.next = T.next[i];
        sq.plo = T.plo[i]; sq.phi = T.phi[i];
    }
    T.wn.resize(T.nnodes());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < T.nnodes(); ++i) {
        WNode& w = T.wn[i];
        w.cx = T.cx[i]; w.cy = T.cy[i]; w.cz = T.cz[i];
        // SHMEM_TREE_NODELTA: drop the COM-offset term from the opening radius, leaving the raw
        // side length -- which is what the reference's Barnes-Hut test uses (forcetree.cc:1889,
        // `nop->len * nop->len > r2 * theta^2`). `delta` depends on WHERE the mass sits inside a
        // node, and in a density gradient that offset is systematically oriented rather than
        // random, so it biases which nodes open in a direction correlated with the gradient. That
        // is a candidate for the coherent, axis-aligned torque this tree shows and GIZMO's does
        // not; less conservative, so expect a larger |da| if it is doing real work.
        w.s  = T.size[i] + (nodelta_opening() ? 0.0 : T.delta[i]);
        w.mass = T.mass[i]; w.soft = (float)T.soft[i]; w.len = (float)T.size[i];
        w.first = T.first[i]; w.next = T.next[i];
        w.plo = T.plo[i]; w.phi = T.phi[i];
    }
    if(bt) { bt->pack = now_ms()-t_a; bt->total = now_ms()-t_start; }
    return T;
}

void accel(const Tree& tree, const Particles& particles, const std::vector<uint32_t>& targets,
           double theta, double G, std::vector<double>& ax, std::vector<double>& ay,
           std::vector<double>& az, const double* aold) {
    const size_t n_targets = targets.size();
    ax.assign(n_targets, 0.0); ay.assign(n_targets, 0.0); az.assign(n_targets, 0.0);
    const double theta_sq = theta * theta;

    // Plain parallel-for over the ACTIVE list. step_overhead showed this beats a persistent pool
    // (4x) and beats capping the width (2.5x) in exactly this regime.
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t t = 0; t < n_targets; ++t) {
        const uint32_t target = targets[t];
        const Vec3d pos_target = particles.pos(target);
        const double soft_target = particles.soft.empty() ? 0.0 : particles.soft[target];
        const double mass_target = particles.m[target];
        const double zeta_target = particles.zeta.empty() ? 0.0 : particles.zeta[target];
        const bool gas_target = particles.is_gas(target);
        const double aold_t = aold ? aold[t] : 0.0;
        Vec3d accel{0, 0, 0};

        const WNode* __restrict nodes = tree.wn.data();
        int node_id = tree.root;
        while (node_id >= 0) {
            const WNode& node = nodes[node_id];           // ONE cache line per visit
            const Vec3d to_com = Vec3d{node.cx, node.cy, node.cz} - pos_target;
            const double r_sq = to_com.norm_sq();
            if (node.first < 0 ||
                !open_node(r_sq, node.len, node.s, node.mass, node.soft, soft_target,
                           aold_t, theta_sq,
                           std::abs(to_com[0]), std::abs(to_com[1]), std::abs(to_com[2]))) {
                if (node.first < 0) {                     // leaf: direct sum over its particles
                    for (int slot = node.plo; slot < node.phi; ++slot) {
                        const uint32_t j = tree.orderbuf[slot];
                        if (j == target) continue;
                        const Vec3d offset = particles.pos(j) - pos_target;
                        const double r = offset.norm();
                        if (r <= 0) continue;
                        accel += offset * pair_force_over_r(
                            r, particles.m[j], soft_target,
                            particles.soft.empty() ? 0.0 : particles.soft[j],
                            zeta_target, particles.zeta.empty() ? 0.0 : particles.zeta[j],
                            mass_target, gas_target && particles.is_gas(j));
                    }
                } else {                                  // far enough: use the monopole
                    const double soft = std::max(soft_target, (double)node.soft);
                    kick(to_com, node.mass, soft, accel);
                }
                node_id = node.next;
            } else {
                node_id = node.first;                     // too close: descend
            }
        }
        ax[t] = G * accel[0]; ay[t] = G * accel[1]; az[t] = G * accel[2];
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
        const Vec3d xp = P.pos(p);
        const double eps = P.soft.empty() ? 0.0 : P.soft[p];
        Vec3d acc{0, 0, 0};

        int node = T.root;
        while (node >= 0) {
            const Vec3d d = Vec3d{T.cx[node], T.cy[node], T.cz[node]} - xp;
            const double r2 = d.norm_sq();
            const double s = T.size[node] + T.delta[node];
            if (T.first[node] < 0 || s * s < theta2 * r2) {
                if (T.first[node] < 0) {
                    for (int i = T.plo[node]; i < T.phi[node]; ++i) {
                        uint32_t q = T.orderbuf[i];
                        if (q == p) continue;
                        double e2 = std::max(eps, P.soft.empty() ? 0.0 : P.soft[q]);
                        kick(P.pos(q) - xp, P.m[q], e2, acc);
                    }
                } else {
                    double e2 = std::max(eps, T.soft[node]);
                    kick(d, T.mass[node], e2, acc);
                }
                node = T.next[node];
            } else {
                node = T.first[node];
            }
        }
        ax[t] = G * acc[0]; ay[t] = G * acc[1]; az[t] = G * acc[2];
    }
}

void accel_grouped(const Tree& T, const Particles& P, const std::vector<uint32_t>& targets,
                   double theta, double G, int batch, std::vector<double>& ax,
                   std::vector<double>& ay, std::vector<double>& az,
                   std::vector<SymTensor3d>* tidal, const double* aold, const LazyDrift* lazy,
                   std::vector<Vec3d>* jerk, const double* const* vel,
                   double sink_direct_radius) {
    const size_t nt = targets.size();
    // Only live when there is a radius, the tree counted sinks, and types exist to identify them.
    const bool sink_direct = (sink_direct_radius > 0.0) && (T.nsink.size() == T.nnodes())
                             && !P.type.empty();
    ax.assign(nt, 0.0); ay.assign(nt, 0.0); az.assign(nt, 0.0);
    const bool want_tidal = (tidal != nullptr);
    // A jerk needs source velocities: the tree's per-node vcom for multipoles and `vel` for
    // leaves. Without either the request is silently a no-op rather than a wrong answer.
    const bool want_jerk = (jerk != nullptr) && (vel != nullptr) && !T.vcom_x.empty();
    if (want_tidal) tidal->assign(nt, SymTensor3d{0,0,0,0,0,0});
    if (jerk) jerk->assign(nt, Vec3d{0,0,0});
    const double theta2 = theta * theta;
    // NODE-ALIGNED GROUPING (batch <= 0) instead of fixed-size windows. Fixed windows chunk the
    // Morton-sorted target list blindly, so a window can straddle a high-level Morton boundary and
    // get a bounding box far larger than any member needs -- every member then pays for the
    // over-opening. That is rare when everything is active and near-universal when the active set
    // is sparse, since consecutive ACTIVE targets are then far apart in Morton order.
    //
    // Grouping on the leaf instead bounds a group's extent by the leaf's extent, by construction.
    // It also self-adapts: a dense active set fills leaves and gives groups of ~LEAF_MAX, a sparse
    // one leaves a handful per leaf and gives small groups, with no threshold to tune.
    //
    // Targets are in Morton order and a leaf owns a contiguous orderbuf range, so a leaf's active
    // targets are already contiguous here -- the grouping is one linear scan, no sort.
    std::vector<size_t> gstart;
    const bool node_grouped = (batch <= 0) && !T.leaf_of.empty();
    if (node_grouped) {
        gstart.reserve(nt / 4 + 2);
        gstart.push_back(0);
        for (size_t i = 1; i < nt; ++i) {
            const int li = T.leaf_of[targets[i]], lp = T.leaf_of[targets[i-1]];
            // A target with no leaf gets its own group rather than joining a neighbour's box.
            if (li != lp || li < 0) gstart.push_back(i);
        }
        gstart.push_back(nt);
    }
    const int nb = node_grouped ? (int)(gstart.size() - 1)
                                : (int)((nt + std::max(batch,1) - 1) / std::max(batch,1));

    // SHMEM_COUNT_WALK: node visits and pair interactions per target. Counts, not times, so the
    // frequency scaling that makes wall-clock comparisons here unreliable does not touch them.
    static const bool count_walk = (getenv("SHMEM_COUNT_WALK") != nullptr);
    long long n_nodes_visited = 0, n_multipole = 0, n_pairs = 0;

    #pragma omp parallel for schedule(dynamic, 1) \
            reduction(+:n_nodes_visited,n_multipole,n_pairs)
    for (int b = 0; b < nb; ++b) {
        size_t lo, hi;
        if (node_grouped) { lo = gstart[b]; hi = gstart[b+1]; }
        else { lo = (size_t)b * batch; hi = std::min(lo + (size_t)batch, nt); }
        // batch bounding box
        double bx0=1e300,by0=1e300,bz0=1e300,bx1=-1e300,by1=-1e300,bz1=-1e300, emax=0;
        for (size_t i = lo; i < hi; ++i) {
            uint32_t p = targets[i];
            bx0=std::min(bx0,P.x[p]); bx1=std::max(bx1,P.x[p]);
            by0=std::min(by0,P.y[p]); by1=std::max(by1,P.y[p]);
            bz0=std::min(bz0,P.z[p]); bz1=std::max(bz1,P.z[p]);
            if (!P.soft.empty()) emax = std::max(emax, P.soft[p]);
        }
        // Conservative like every other batch reduction: if ANY member is a sink, the whole batch
        // uses the sink-direct criterion. Over-opening for the batch's non-sinks costs a little
        // work; under-opening would silently approximate a collisional pair.
        bool batch_has_sink = false;
        if (sink_direct)
            for (size_t i = lo; i < hi && !batch_has_sink; ++i)
                batch_has_sink = (P.type[targets[i]] == 5);
        // Batch opening uses the CONSERVATIVE reduction of every per-member test: nearest bbox
        // point for r, the largest softening, and the SMALLEST aold (an aold of 0 anywhere in the
        // batch forces the geometric test, which opens at least as much).
        double aold_min = aold ? 1e300 : 0.0;
        if (aold) for (size_t i = lo; i < hi; ++i) aold_min = std::min(aold_min, aold[i]);
        // Gather the batch into contiguous local buffers. Without this the inner loop gathers
        // P.x[targets[i]] for every node, replacing one node fetch per target with `batch` scattered
        // particle fetches -- measured 3-10x SLOWER than the per-target walk.
        //
        // Do NOT defend this layout on vectorisation: the pair loop below does not vectorise and
        // will not. -fopt-info-vec reports nothing for it under gcc 8.5 -O2 (our build) or gcc 13.3
        // -O3 -- only the tree build and the trivial write-back at the end of this function. The
        // continue, the sqrt and the type-dependent branches in pair_force_over_r see to that. What
        // grouping actually buys is one traversal and one opening decision per batch instead of per
        // target, plus gathering each target's data once per batch rather than per interaction.
        const int nb_ = (int)(hi - lo);
        const bool have_zeta = !P.zeta.empty();
        // 520, NOT 512, AND THE DIFFERENCE IS LARGE. double[512] is exactly 4096 bytes, which is
        // the L1 way size on both targets (32 KB, 8-way, 64 B lines -> 64 sets). Stack arrays sit
        // back to back, so at that stride element [i] of every one of these maps to the SAME cache
        // set -- the batch is ~8 wide, so each array is one line and ~19 lines fight over 8 ways.
        // Padding by one line spreads them one set apart. Measured on the 3.5e6-cell BBB03 state
        // (bench_treeforce, 16 threads), forces bit-identical:
        //     accel              6145 -> 6019 ms
        //     accel+tidal       14920 -> 10517 ms
        //     accel+tidal+jerk  25322 -> 16663 ms
        // Do not "tidy" this back to a power of two.
        #define TBUF 520
        double tx[TBUF], ty[TBUF], tz[TBUF], te[TBUF], oax[TBUF], oay[TBUF], oaz[TBUF];
        double tmass[TBUF], tzeta[TBUF]; bool tgas[TBUF];
        double ott[6][TBUF];
        double tvx[TBUF], tvy[TBUF], tvz[TBUF], ojx[TBUF], ojy[TBUF], ojz[TBUF];
        if (want_tidal) for (int c = 0; c < 6; ++c) for (int i = 0; i < (int)(hi-lo); ++i) ott[c][i] = 0;
        for (int i = 0; i < nb_; ++i) {
            uint32_t p = targets[lo + i];
            tx[i]=P.x[p]; ty[i]=P.y[p]; tz[i]=P.z[p];
            te[i]=P.soft.empty()?0.0:P.soft[p];
            tmass[i]=P.m[p]; tzeta[i]=have_zeta?P.zeta[p]:0.0; tgas[i]=P.is_gas(p);
            oax[i]=0; oay[i]=0; oaz[i]=0;
            if (want_jerk) {
                tvx[i]=vel[0][p]; tvy[i]=vel[1][p]; tvz[i]=vel[2][p];
                ojx[i]=0; ojy[i]=0; ojz[i]=0;
            }
        }
        const WNode* __restrict W = T.wn.data();
        int node = T.root;
        while (node >= 0) {
            const WNode& w = W[node];
            if (count_walk) ++n_nodes_visited;
            double dx = std::max(0.0, std::max(bx0 - w.cx, w.cx - bx1));
            double dy = std::max(0.0, std::max(by0 - w.cy, w.cy - by1));
            double dz = std::max(0.0, std::max(bz0 - w.cz, w.cz - bz1));
            double rmin2 = dx*dx + dy*dy + dz*dz;
            // A LEAF IS A NODE. Testing `w.first < 0` first would short-circuit the criterion and
            // direct-sum every leaf reached, up to LEAF_MAX particles, even where the multipole
            // was acceptable: measured 4444.9 direct pairs per target against pytreegrav's 695.7
            // on the same ICs and theta, which is the whole of a 2.6x interaction-count gap. Ask
            // the criterion first; expand only the leaves it actually wants opened.
            //
            // Safe because the softening-overlap arm of the criterion opens any node whose
            // particles could reach the target's kernel. Beyond that overlap the spline is
            // Newtonian, so the pair kernel equals the single evaluation the multipole
            // applies, and the zeta corrections (r < h only) are
            // identically zero. Accepting a leaf therefore drops nothing that was contributing.
            // SHMEM_NO_LEAF_ACCEPT restores the pre-0f0ae23d behaviour: leaves are always opened
            // and direct-summed rather than being offered to the criterion. Diagnostic only --
            // it is here to attribute the plummer_binaries Lagrange drift, since that change was
            // measured at +17% median force error and nothing protects a pc-scale collisional
            // N-body test from it (SINGLE_STAR_DIRECT_GRAVITY_RADIUS is 1000 AU = 0.005 pc, three
            // orders below this system's r_10).
            static const bool no_leaf_accept = (getenv("SHMEM_NO_LEAF_ACCEPT") != nullptr);
            bool accept = !open_node(rmin2, w.len, w.s, w.mass, w.soft, emax, aold_min,
                                     theta2, dx, dy, dz);
            if (no_leaf_accept && w.first < 0) accept = false;
            // SINK-SINK DIRECT SUMMATION (forcetree.cc:1975). A sink looking at a node that holds
            // sinks within sink_direct_radius opens it, so star-star pairs are always summed
            // exactly. Applied after the general criterion because it only ever ADDS opening.
            if (accept && batch_has_sink && T.nsink[node] > 0) {
                const double reach = sink_direct_radius + 0.6 * w.len;
                if (rmin2 < reach * reach) accept = false;
            }
            if (accept || w.first < 0) {
                if (!accept) {
                    for (int k = w.plo; k < w.phi; ++k) {
                        uint32_t q = T.orderbuf[k];
                        // Lazy drift, exactly as in the neighbour search and in GIZMO's
                        // forcetree.cc:1678: a source particle is caught up the moment the walk
                        // reaches it, before its position is read.
                        if (lazy && lazy->last[q] < lazy->target) lazy->catch_up(lazy->ctx, q);
                        double qx=P.x[q], qy=P.y[q], qz=P.z[q], qm=P.m[q];
                        double qs = P.soft.empty()?0.0:P.soft[q];
                        double qzeta = have_zeta?P.zeta[q]:0.0;
                        const bool qgas = P.is_gas(q);
                        // Hoisted with the rest of the source's data: these depend only on q, and
                        // were being re-gathered once per target in the batch. Worth about 1% of
                        // the walk, not the memory effect it looks like -- the velocities are hot
                        // by the time the batch reaches them. The jerk's real cost is accumulator
                        // pressure, same as the tidal tensor's; see the buffer padding above.
                        double qvx=0, qvy=0, qvz=0;
                        if (want_jerk) { qvx=vel[0][q]; qvy=vel[1][q]; qvz=vel[2][q]; }
                        for (int i = 0; i < nb_; ++i) {
                            if (targets[lo+i] == q) continue;
                            const double dx_=qx-tx[i], dy_=qy-ty[i], dz_=qz-tz[i];
                            const double r2 = dx_*dx_ + dy_*dy_ + dz_*dz_;
                            if (r2 <= 0) continue;
                            if (count_walk) ++n_pairs;
                            const double r = std::sqrt(r2);
                            double g1 = 0.0;
                            const double fac = pair_force_over_r(
                                r, qm, te[i], qs, tzeta[i], qzeta,
                                tmass[i], tgas[i] && qgas,
                                (want_tidal || want_jerk) ? &g1 : nullptr);
                            oax[i] += dx_*fac; oay[i] += dy_*fac; oaz[i] += dz_*fac;
                            if (want_tidal || want_jerk) {
                                // g1 (the zeta-free force factor) comes back from the force call
                                // above rather than being recomputed -- the same max-softening
                                // spline_force_over_r. Only the mode-2
                                // factor is new here. Shared by the tidal tensor and the jerk,
                                // which is why the reference accumulates the jerk in this loop
                                // rather than in a pass of its own.
                                const double g2 =
                                    qm*grav_tidal_factor(r, std::max(std::max(te[i],qs),1e-300));
                                if (want_tidal) {
                                    ott[0][i] += -g1 + dx_*dx_*g2;   // xx
                                    ott[1][i] += -g1 + dy_*dy_*g2;   // yy
                                    ott[2][i] += -g1 + dz_*dz_*g2;   // zz
                                    ott[3][i] += dx_*dy_*g2;         // xy
                                    ott[4][i] += dy_*dz_*g2;         // yz
                                    ott[5][i] += dx_*dz_*g2;         // xz
                                }
                                if (want_jerk) {
                                    // leaf source: its own velocity (forcetree.cc:1641)
                                    const double dvx=qvx-tvx[i], dvy=qvy-tvy[i],
                                                 dvz=qvz-tvz[i];
                                    const double vdotr = dvx*dx_ + dvy*dy_ + dvz*dz_;
                                    ojx[i] += g1*dvx - vdotr*g2*dx_;
                                    ojy[i] += g1*dvy - vdotr*g2*dy_;
                                    ojz[i] += g1*dvz - vdotr*g2*dz_;
                                }
                            }
                        }
                    }
                } else {
                    if (count_walk) n_multipole += nb_;
                    for (int i = 0; i < nb_; ++i) {
                        double e = std::max(te[i], (double)w.soft);
                        kick(w.cx-tx[i], w.cy-ty[i], w.cz-tz[i], w.mass, e, oax[i], oay[i], oaz[i]);
                        if (want_tidal || want_jerk) {
                            const double dx_=w.cx-tx[i], dy_=w.cy-ty[i], dz_=w.cz-tz[i];
                            const double r2 = dx_*dx_+dy_*dy_+dz_*dz_;
                            if (r2 > 0) {
                                const double r = std::sqrt(r2), es = std::max(e,1e-300);
                                const double g1 = w.mass*spline_force_over_r(r,es);
                                const double g2 = w.mass*grav_tidal_factor(r,es);
                                if (want_tidal) {
                                    ott[0][i] += -g1 + dx_*dx_*g2;
                                    ott[1][i] += -g1 + dy_*dy_*g2;
                                    ott[2][i] += -g1 + dz_*dz_*g2;
                                    ott[3][i] += dx_*dy_*g2;
                                    ott[4][i] += dy_*dz_*g2;
                                    ott[5][i] += dx_*dz_*g2;
                                }
                                if (want_jerk) {
                                    // node source: its centre-of-mass velocity, GIZMO's
                                    // Extnodes[].vs (forcetree.cc:1945)
                                    const Vec3d nv = T.node_vel(node);
                                    const double dvx=nv[0]-tvx[i], dvy=nv[1]-tvy[i],
                                                 dvz=nv[2]-tvz[i];
                                    const double vdotr = dvx*dx_ + dvy*dy_ + dvz*dz_;
                                    ojx[i] += g1*dvx - vdotr*g2*dx_;
                                    ojy[i] += g1*dvy - vdotr*g2*dy_;
                                    ojz[i] += g1*dvz - vdotr*g2*dz_;
                                }
                            }
                        }
                    }
                }
                node = w.next;
            } else {
                node = w.first;
            }
        }
        for (int i = 0; i < nb_; ++i) { ax[lo+i]=G*oax[i]; ay[lo+i]=G*oay[i]; az[lo+i]=G*oaz[i]; }
        if (want_jerk) for (int i = 0; i < nb_; ++i)
            (*jerk)[lo+i] = Vec3d{G*ojx[i], G*ojy[i], G*ojz[i]};
        if (want_tidal) for (int i = 0; i < nb_; ++i) {
            SymTensor3d& tt = (*tidal)[lo+i];
            tt[0][0]=ott[0][i]; tt[1][1]=ott[1][i]; tt[2][2]=ott[2][i];
            tt[0][1]=ott[3][i]; tt[1][2]=ott[4][i]; tt[0][2]=ott[5][i];
        }

    }
    if (count_walk && nt > 0) {
        fprintf(stderr, "[walk] targets=%zu batch=%d  nodes_visited=%lld (%.1f/target)  "
                "multipole=%lld (%.1f/target)  pairs=%lld (%.1f/target)  interactions=%.1f/target\n",
                nt, batch, n_nodes_visited, (double)n_nodes_visited/nt,
                n_multipole, (double)n_multipole/nt, n_pairs, (double)n_pairs/nt,
                (double)(n_multipole + n_pairs)/nt);
    }
}

void potential(const Tree& tree, const Particles& particles, const std::vector<uint32_t>& targets,
               double theta, double G, std::vector<double>& phi, const double* aold) {
    const size_t n_targets = targets.size();
    phi.assign(n_targets, 0.0);
    const double theta_sq = theta * theta;
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t t = 0; t < n_targets; ++t) {
        const uint32_t target = targets[t];
        const Vec3d pos_target = particles.pos(target);
        const double soft_target = particles.soft.empty() ? 0.0 : particles.soft[target];
        const double aold_t = aold ? aold[t] : 0.0;
        double sum = 0;
        const WNode* __restrict nodes = tree.wn.data();
        int node_id = tree.root;
        while (node_id >= 0) {
            const WNode& node = nodes[node_id];
            const Vec3d to_com = Vec3d{node.cx, node.cy, node.cz} - pos_target;
            const double r_sq = to_com.norm_sq();
            if (node.first < 0 ||
                !open_node(r_sq, node.len, node.s, node.mass, node.soft, soft_target,
                           aold_t, theta_sq,
                           std::abs(to_com[0]), std::abs(to_com[1]), std::abs(to_com[2]))) {
                if (node.first < 0) {
                    for (int slot = node.plo; slot < node.phi; ++slot) {
                        const uint32_t j = tree.orderbuf[slot];
                        if (j == target) continue;
                        const double r = (particles.pos(j) - pos_target).norm();
                        if (r <= 0) continue;
                        // kernel of the larger softening, same as the pair force
                        const double soft_j = particles.soft.empty() ? 0.0 : particles.soft[j];
                        sum += particles.m[j] *
                               spline_potential(r, std::max(std::max(soft_target, soft_j), 1e-300));
                    }
                } else {
                    const double soft = std::max(soft_target, (double)node.soft);
                    const double r = std::sqrt(r_sq);
                    if (r > 0) sum += node.mass * spline_potential(r, std::max(soft, 1e-300));
                }
                node_id = node.next;
            } else {
                node_id = node.first;
            }
        }
        phi[t] = G * sum;
    }
}

void accel_brute(const Particles& P, const std::vector<uint32_t>& targets, double G,
                 std::vector<double>& ax, std::vector<double>& ay, std::vector<double>& az) {
    const size_t nt = targets.size(), n = P.size();
    ax.assign(nt, 0.0); ay.assign(nt, 0.0); az.assign(nt, 0.0);
    #pragma omp parallel for schedule(static)
    for (size_t t = 0; t < nt; ++t) {
        uint32_t p = targets[t];
        const Vec3d xp = P.pos(p);
        double eps = P.soft.empty() ? 0.0 : P.soft[p];
        const double mass_p = P.m[p];
        const double zeta_p = P.zeta.empty() ? 0.0 : P.zeta[p];
        const bool gas_p = P.is_gas(p);
        Vec3d acc{0, 0, 0};
        for (size_t q = 0; q < n; ++q) {
            if (q == p) continue;
            const Vec3d offset = P.pos(q) - xp;
            const double r = offset.norm();
            if (r <= 0) continue;
            acc += offset * pair_force_over_r(
                r, P.m[q], eps, P.soft.empty() ? 0.0 : P.soft[q],
                zeta_p, P.zeta.empty() ? 0.0 : P.zeta[q],
                mass_p, gas_p && P.is_gas(q));
        }
        ax[t] = G*acc[0]; ay[t] = G*acc[1]; az[t] = G*acc[2];
    }
}

long long audit_tree(const Tree& T, const Particles& P, const double* h, size_t n_gas,
                     const double* vx, const double* vy, const double* vz, double box,
                     const double* dt_behind, char* msg, size_t msglen) {
    long long bad = 0;
    auto fail = [&](const char* fmt, auto... args) {
        if (bad++ == 0 && msg) snprintf(msg, msglen, fmt, args...);
    };
    const size_t N = P.size(), nn = T.nnodes();
    if (T.parent.size() != nn || T.leaf_of.size() != N || T.wn.size() != nn || T.sn.size() != nn) {
        fail("array sizes: nodes %lld, particles %lld", (long long)nn, (long long)N);
        return bad;
    }

    // 1. threading and topology, without trusting parent[]
    std::vector<uint8_t> seen_node(nn, 0), seen_part(N, 0);
    std::vector<int> stack{T.root};
    if (T.parent[T.root] != -1) fail("root %d has parent %d", T.root, T.parent[T.root]);
    size_t steps = 0;
    while (!stack.empty()) {
        const int X = stack.back(); stack.pop_back();
        if (X < 0 || (size_t)X >= nn) { fail("node index %d out of range %zu", X, nn); continue; }
        if (seen_node[X]++) { fail("node %d reached twice", X); continue; }
        if (T.first[X] < 0) {
            for (int slot = T.plo[X]; slot < T.phi[X]; ++slot) {
                const uint32_t p = T.orderbuf[slot];
                if (p >= N) { fail("leaf %d holds index %u", X, p); continue; }
                if (seen_part[p]++) fail("particle %u reached twice, leaf %d", p, X);
                if (T.leaf_of[p] != X) fail("leaf_of[%u]=%d, but threading reaches it in leaf %d", p, T.leaf_of[p], X);
            }
            continue;
        }
        for (int c = T.first[X]; c != T.next[X]; c = T.next[c]) {
            if (c < 0 || (size_t)c >= nn || ++steps > 2 * nn) {
                fail("child chain of node %d never reaches next[X]=%d", X, T.next[X]);
                break;
            }
            if (T.parent[c] != X) fail("parent[%d]=%d, threading says %d", c, T.parent[c], X);
            stack.push_back(c);
        }
    }
    // only tombstones (massless) may be absent from the tree
    for (size_t p = 0; p < N; ++p)
        if (!seen_part[p] && P.m[p] > 0) { fail("particle %zu unreachable from the root", p); break; }
    if (bad) return bad;   // coverage checks below climb parent[], which is only now trusted

    // 2. node mass and centre of mass, re-derived bottom-up from the particles (children are
    // allocated after their parent) against the incremental moments. Positions are taken at the
    // tree's current time (dt_behind undoes lazy drift) and unwrapped against the tracked COM.
    const double tau = T.t_since_build;
    std::vector<double> msum(nn, 0.0);
    std::vector<Vec3d> mdx(nn, Vec3d{0, 0, 0});
    std::vector<Vec3d> com(nn);
    for (size_t no = 0; no < nn; ++no) com[no] = T.node_com((int)no, tau);
    for (size_t p = 0; p < N; ++p) {
        if (T.leaf_of[p] < 0 || P.m[p] <= 0) continue;
        Vec3d x = P.pos(p);
        if (dt_behind && vx) x += Vec3d{vx[p], vy[p], vz[p]} * dt_behind[p];
        for (int no = T.leaf_of[p]; no >= 0; no = T.parent[no]) {
            msum[no] += P.m[p];
            mdx[no] += min_image(x - com[no], box) * P.m[p];
        }
    }
    for (size_t no = 0; no < nn; ++no) {
        const double M = T.node_mass((int)no);
        const double tol = 1e-10 * std::max(msum[no], std::abs(M)) + 1e-300;
        if (std::abs(msum[no] - M) > tol) {
            fail("node %zu mass %.17g, particles sum to %.17g", no, M, msum[no]);
            break;
        }
        if (msum[no] > 0 && !T.dp_x.empty()) {
            const double off = (mdx[no] / msum[no]).norm();
            const double reach = T.size[no] + 2.0 * ((double)T.vmax[no] * tau + (double)T.xpad[no]);
            if (off > 1e-6 * reach) {
                fail("node %zu tracked COM is %.9g off the particles' (node reach %.9g)", no, off, reach);
                break;
            }
        }
    }

    // 3. every ancestor covers every particle
    const float* hm = T.hmax.empty() ? nullptr : T.hmax.data();
    for (size_t p = 0; p < N && bad < 1000; ++p) {
        if (T.leaf_of[p] < 0 || P.m[p] <= 0) continue;   // tombstones are parked, not covered
        const double sp = P.soft.empty() ? 0.0 : P.soft[p];
        const bool gas = p < n_gas && h;
        const double v = vx ? std::sqrt(vx[p]*vx[p] + vy[p]*vy[p] + vz[p]*vz[p]) : 0.0;
        for (int no = T.leaf_of[p]; no >= 0; no = T.parent[no]) {
            if (gas && hm && (double)hm[no] < h[p])
                fail("hmax[%d]=%.9g below h=%.9g of particle %zu", no, (double)hm[no], h[p], p);
            if ((double)T.wn[no].soft * (1.0 + 1e-6) < sp)
                fail("node %d soft %.9g below soft=%.9g of particle %zu", no, (double)T.wn[no].soft, sp, p);
            if (vx && (double)T.vmax[no] * (1.0 + 1e-6) < v)
                fail("vmax[%d]=%.9g below |v|=%.9g of particle %zu", no, (double)T.vmax[no], v, p);
            const SNode& sn = T.sn[no];
            const double reach = (double)sn.half + (double)T.vmax[no] * T.t_since_build
                               + (double)T.xpad[no] + 1e-6 * (double)sn.half;
            const double d[3] = {min_image(P.x[p] - (double)sn.cx, box),
                                 min_image(P.y[p] - (double)sn.cy, box),
                                 min_image(P.z[p] - (double)sn.cz, box)};
            for (int a = 0; a < 3; ++a)
                if (std::abs(d[a]) > reach) {
                    fail("particle %zu outside node %d box: offset %.9g > reach %.9g", p, no, std::abs(d[a]), reach);
                    break;
                }
        }
    }
    return bad;
}

}  // namespace shmem
