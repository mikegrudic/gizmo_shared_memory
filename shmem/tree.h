// Shared-memory Barnes-Hut octree: flat SoA storage, iterative traversal, OpenMP throughout.
//
// Design follows pytreegrav rather than GIZMO's forcetree.cc, for one structural reason: GIZMO's
// tree is organised around domain decomposition (top-level nodes, pseudo-particles, per-rank
// subtrees) and none of that exists here. What remains is the part that matters:
//
//   * FLAT SoA NODE ARRAYS. Nodes are indices into parallel typed arrays, not pointer-linked
//     structs. Cache-friendlier on the walk, and read-only during it, so every thread can walk the
//     whole tree with no ownership, no export/import and no locking. That property is what lets the
//     entire MPI exchange layer disappear rather than be reimplemented.
//
//   * next[] / first[] ITERATIVE TRAVERSAL. pytreegrav's NextBranch/FirstSubnode. The walk is a
//     flat loop over two index arrays: no recursion, no per-thread stack, and any thread can start
//     anywhere in the tree.
//
//   * MORTON-ORDERED BUILD. Particles are sorted by Z-order key once; every node is then a
//     contiguous range of that sorted array, so children are found by scanning key prefixes rather
//     than by insertion. Build parallelises as recursive tasks over disjoint ranges.
//
// Threading follows the step_overhead measurement, not intuition: plain `#pragma omp parallel for`,
// no persistent pool (libgomp already keeps the team alive; explicit barriers were 4x worse) and no
// adaptive width cap (capping at nactive/40 was 2.5x worse than using every thread).

#pragma once
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "vec.h"                   // Vec3/Mat3/SymmetricTensor2 + the periodic wrap helpers

namespace shmem {

struct Particles {                 // SoA: the walk reads x/y/z/m for many particles at once
    std::vector<double> x, y, z, m, soft;
    // Adaptive-softening force-correction coefficients (Price & Monaghan 2007; GIZMO's AGS_zeta).
    // Empty means "no corrections": the walks only read it when non-empty, so gravity-only users
    // of the tree never pay for it.
    std::vector<double> zeta;
    // Particle type, GIZMO numbering (0 = gas; sinks/stars/DM > 0). Empty means "all gas".
    // The gravity walks need it because the PAIR RULE depends on the types (see
    // pair_force_over_r): kernel-averaging and the zeta terms apply to gas-gas pairs only.
    std::vector<uint8_t> type;
    size_t size() const { return m.size(); }

    Vec3d pos(size_t i) const { return Vec3d{x[i], y[i], z[i]}; }
    bool is_gas(size_t i) const { return type.empty() || type[i] == 0; }
};

// Second-derivative (tidal) factor of the cubic-spline softening kernel -- GIZMO's
// kernel_gravity(mode=2). The pair's contribution to the tidal tensor is
//   T_kl += -g1 * delta_kl + g2 * dp_k dp_l,
// with g1 the force-over-r factor and g2 this one; beyond the softening it is the Newtonian
// 3/r^5, so a point mass gives the exact -m/r^3 delta + 3m rr/r^5.
static inline double grav_tidal_factor(double r, double h) {
    const double h_inv = 1.0 / h;
    const double u = r * h_inv;
    if (u >= 1.0) return 3.0 / (r * r * r * r * r);
    const double h_inv5 = h_inv * h_inv * h_inv * h_inv * h_inv;
    double wk;
    if (u < 0.5) wk = 76.8 - 96.0*u;
    else         wk = -0.2/(u*u*u*u*u) + 48.0/u - 76.8 + 32.0*u;
    return wk * h_inv5;
}

// Node-opening decision, following GIZMO's forcetree.cc with the RELATIVE (acceleration)
// criterion. `aold` is ErrTolForceAcc * |a_prev|/G for the target (0 when no previous force
// exists, e.g. the first step) and `theta_sq` the geometric fallback used in that bootstrap case.
// Tested in GIZMO's order:
//   1. softening overlap: open when the node could contain anything within either side's
//      softening -- this is also what guarantees kernel-overlapping pairs are always resolved
//      down to actual particles, so the zeta corrections never need a node form;
//   2. relative criterion  M L^2 > r^4 aold  (geometric  L_open^2 > theta^2 r^2  when aold = 0);
//   3. inside-node: per-axis |dx| < 0.6 L.
// `l_open` should be the conservative opening radius (size + COM offset) used by the geometric
// test; `len` the raw side length.
static inline bool open_node(double r_sq, double len, double l_open, double node_mass,
                             double node_maxsoft, double soft_target,
                             double aold, double theta_sq,
                             double adx, double ady, double adz) {
    const double grow_t = soft_target + 0.6 * len, grow_n = node_maxsoft + 0.6 * len;
    if (r_sq < grow_t * grow_t || r_sq < grow_n * grow_n) return true;
    if (aold > 0.0) {
        if (node_mass * len * len > r_sq * r_sq * aold) return true;
    } else {
        if (l_open * l_open >= theta_sq * r_sq) return true;
    }
    return adx < 0.6 * len && ady < 0.6 * len && adz < 0.6 * len;
}

// d(phi)/dh of the cubic-spline softening kernel at fixed r -- GIZMO's kernel_gravity(mode=0).
// This is what the zeta terms integrate over the neighbourhood: with ADAPTIVE softening the
// potential depends on h, h depends on the particle arrangement, and energy conservation requires
// the force to pick up the corresponding dPhi/dh * dh/dr terms. Zero for r >= h, since there the
// potential is exactly -1/r and has no h dependence at all.
static inline double grav_dphi_dh(double r, double h) {
    const double h_inv = 1.0 / h;
    const double u = r * h_inv;
    if (u >= 1.0) return 0.0;
    double wk;
    if (u < 0.5) wk = 2.8 + 16.0*u*u*(-1.0 + 3.0*u*u*(1.0 - 0.8*u));
    else         wk = 3.2 + 32.0*u*u*(-1.0 + u*(2.0 - 1.5*u + 0.4*u*u));
    return wk * h_inv * h_inv;
}

// ---- Morton (Z-order) key: interleave the low 21 bits of each coordinate ----
static inline uint64_t spread3(uint64_t v) {
    v &= 0x1FFFFFull;
    v = (v | v << 32) & 0x1F00000000FFFFull;
    v = (v | v << 16) & 0x1F0000FF0000FFull;
    v = (v | v << 8)  & 0x100F00F00F00F00Full;
    v = (v | v << 4)  & 0x10C30C30C30C30C3ull;
    v = (v | v << 2)  & 0x1249249249249249ull;
    return v;
}
static inline uint64_t morton(uint32_t a, uint32_t b, uint32_t c) {
    return spread3(a) | (spread3(b) << 1) | (spread3(c) << 2);
}

// LAZY DRIFT HOOK, following GIZMO's system/ngb_codeblock_after_condition_{unthreaded,threaded}.h.
// A particle is brought up to date the MOMENT a search reaches it -- before the distance test, not
// after one passes -- so a stale position can never hide a true neighbour. The slack that makes
// this sound lives in the node extent (Tree::vmax, below), never in the query radius.
//
// `last` is the per-particle tick array; the callback fires only for genuinely stale particles, so
// a step where everything is already current pays one load and one compare per neighbour and never
// calls out at all. That cost model is the whole point: it is proportional to actual staleness.
struct LazyDrift {
    const long long* last = nullptr;              // per-particle "position is current at" tick
    long long        target = 0;                  // tick to bring particles up to
    void           (*catch_up)(void*, uint32_t) = nullptr;
    void*            ctx = nullptr;
};

// Packed node for TRAVERSAL. SoA is right for bulk particle loops but wrong for a tree walk, which
// needs every field of ONE node: with 11 separate arrays each visit touched ~9 cache lines and cost
// ~108 ns (measured), i.e. DRAM latency per node. Packed into exactly one 64-byte line the walk
// touches one line per visit. `s` is (size + delta) precomputed, since the walk never needs them
// apart.
struct alignas(64) WNode {
    double cx, cy, cz;   // 24  centre of mass
    double s;            //  8  size + delta: the conservative opening radius
    double mass;         //  8
    float  soft;         //  4  max softening in the node
    float  len;          //  4  raw side length, for the relative opening + inside-node tests
    int    first;        //  4  first child, or -1 for a leaf
    int    next;         //  4  where to go when not opened
    int    plo, phi;     //  8  leaf particle range
};                       // = 64 bytes exactly (the len field fills what used to be padding)

struct Tree {
    // node arrays, indexed by node id; leaves store a particle range instead of children
    std::vector<double> cx, cy, cz;     // centre of mass
    std::vector<double> mass;           // total mass
    std::vector<double> size;           // side length
    std::vector<double> delta;          // |COM - geometric centre|, for the opening criterion
    std::vector<double> soft;           // max softening in the node
    std::vector<int>    first;          // first child node, or -1 for a leaf
    std::vector<int>    next;           // next node to visit when this one is NOT opened
    std::vector<int>    plo, phi;       // particle range [plo, phi) for leaves
    std::vector<uint32_t> orderbuf;     // Morton-sorted particle indices; leaf ranges index this
    std::vector<WNode>  wn;             // packed traversal copy, built once after the tree
    int root = 0;
    int nalloc = 0;                     // bump allocator cursor for lock-free node claiming

    // ---- per-node velocity bound, for reusing a tree whose particles have moved ----
    // GIZMO's Extnodes[].vmax (gravity/forcetree.cc:827, forcetree_update.cc:78-141), kept in a
    // side array for the same reason GIZMO keeps Extnodes separate from Nodes: the traversal node
    // is exactly one cache line and must not grow.
    //
    // A node's particles all lie within `s` of its centre of mass AT BUILD TIME, and none of them
    // can have moved further than vmax * t_since_build since, so
    //     effective opening radius = s + vmax * t_since_build
    // still cannot reject a node holding a true neighbour. This is GIZMO's `len += 2*vmax*dt`
    // (forcetree_update.cc:375) written for a radius instead of a side length.
    //
    // It has to be PER NODE, not a global bound: a single fast particle would otherwise inflate the
    // prune for the whole box. And the timestep criterion does NOT keep fast particles out of long
    // bins -- v_sig is the Galilean-invariant SIGNAL speed (sound speed plus relative approach), so
    // a cold fast-advecting region has a small v_sig, a long dt and a huge lab-frame displacement.
    // Localising the bound is the entire point.
    std::vector<int>   parent;          // parent node index, -1 at the root; for the kick climb
    std::vector<int>   leaf_of;         // per PARTICLE, the leaf holding it -- GIZMO's Father[]
    std::vector<float> vmax;            // max |v| over the node's particles, since the build
    double t_since_build = 0.0;         // engine time elapsed since this tree was built

    size_t nnodes() const { return mass.size(); }   // valid after build() trims to nalloc

    // Opening radius of a node, inflated for motion since the build. Cheap enough for the
    // neighbour-search inner loop: one float load from a compact array plus an FMA.
    double open_radius(int node_id) const {
        return wn[node_id].s + (double)vmax[node_id] * t_since_build;
    }

    // Record that particle `i` (in leaf `leaf_node`) now moves at |v| = speed, propagating the
    // bound to every ancestor. GIZMO's force_kick_node: climb the parent chain taking a max. The
    // climb STOPS as soon as an ancestor already covers the speed -- a node's vmax is by
    // construction >= all its children's, so once one covers it they all do. In steady state that
    // breaks at the first test, which is what keeps this affordable on an all-active step.
    void raise_vmax(int leaf_node, float speed);
};

static const int LEAF_MAX = 16;        // particles per leaf; below this, direct summation is cheaper
static const int MAX_LEVEL = 20;       // Morton keys carry 21 bits per axis

// Build a subtree over the Morton-sorted range [lo, hi). Returns the new node's index.
// `order` maps sorted position -> particle index; `key` is the sorted key array.
int build_node(Tree& T, const Particles& P, const std::vector<uint32_t>& order,
               const std::vector<uint64_t>& key, int lo, int hi, int level,
               double cxi, double cyi, double czi, double sz,
               const double* const* vel = nullptr);

// Assign next[] so a walk that does not open a node can jump straight past its subtree.
void setup_walk(Tree& T, int node, int next_sibling);

// Stage timings, so build cost is attributed rather than guessed.
struct BuildTimes { double bbox=0, keys=0, sort=0, recurse=0, links=0, pack=0, total=0; };

// Build the whole tree. Parallel key computation and walk-link setup; the recursive build is
// task-parallel over disjoint ranges.
// `vel`, when non-null, points at three per-particle velocity arrays (vx, vy, vz) and switches on
// the per-node vmax bound above. Passed rather than stored on Particles because the engine keeps
// velocities in its own arrays; gravity-only users of the tree pass nothing and pay nothing.
Tree build(const Particles& P, BuildTimes* bt = nullptr, const double* const* vel = nullptr);

// Accelerations for the listed targets, Barnes-Hut with opening angle theta.
// `targets` is the ACTIVE list -- the whole point is that it is usually tiny compared to P.
// `aold`, when non-null, points at ErrTolForceAcc * |a_prev|/G per TARGET (parallel to
// `targets`), switching the walk from the geometric to GIZMO's relative opening criterion;
// entries of 0 fall back to the geometric test (the first-force bootstrap).
void accel(const Tree& T, const Particles& P, const std::vector<uint32_t>& targets,
           double theta, double G, std::vector<double>& ax, std::vector<double>& ay,
           std::vector<double>& az, const double* aold = nullptr);

// Same walk, but reading the SoA node arrays instead of the packed WNode. Kept ONLY as a controlled
// comparison: run against accel() in the same process, interleaved, so background load hits both
// equally and the layout difference is what remains.
void accel_soa(const Tree& T, const Particles& P, const std::vector<uint32_t>& targets,
               double theta, double G, std::vector<double>& ax, std::vector<double>& ay,
               std::vector<double>& az);

// GROUPED walk: one traversal serves a whole batch of targets. Each node is fetched and tested
// once, then applied to every member of the batch, so the latency-bound node fetches amortise by the
// batch size instead of being repeated per target. This is pytreegrav's grouped_treewalk; simply
// reordering targets does NOT do it, since that still performs one independent traversal each.
//
// The opening test uses the batch's bounding box: a node is accepted only if it is far enough from
// the NEAREST point of the box, so the result is at least as accurate as the per-target walk.
// `batch` trades amortisation (large) against over-opening (small batches keep the box tight). The
// optimum is ~8 and is NOT problem-specific: measured 1.71x at B=8 on the real post-sink active set
// and 1.68x at B=32 on a 10x larger one (flat between), falling to 0.36x by B=128 as the box
// dilutes. pytreegrav independently arrived at ~8. Treat it as a default, not a tunable.
// `tidal`, when non-null, receives the tidal tensor (second derivatives of the potential, WITHOUT
// the G factor -- same convention as ax/ay/az) per target, accumulated in the same walk. Built
// from the BASE pair force factor, without the zeta corrections, as GIZMO does.
void accel_grouped(const Tree& T, const Particles& P, const std::vector<uint32_t>& targets,
                   double theta, double G, int batch /* = 8 */, std::vector<double>& ax,
                   std::vector<double>& ay, std::vector<double>& az,
                   std::vector<SymTensor3d>* tidal = nullptr, const double* aold = nullptr,
                   const LazyDrift* lazy = nullptr);

// Gravitational potential at each target, spline-softened to match accel(). Separate from the
// force walk because it is only wanted for diagnostics -- but it is the diagnostic that matters
// for a self-gravitating run, since kinetic + internal alone is not a conserved quantity and can
// look perfectly steady while the integrator quietly mis-applies gravity.
void potential(const Tree& T, const Particles& P, const std::vector<uint32_t>& targets,
               double theta, double G, std::vector<double>& phi, const double* aold = nullptr);

// Direct O(N*M) summation, for validating the tree.
void accel_brute(const Particles& P, const std::vector<uint32_t>& targets, double G,
                 std::vector<double>& ax, std::vector<double>& ay, std::vector<double>& az);

}  // namespace shmem
