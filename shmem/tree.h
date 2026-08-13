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

// (1/r) d(phi)/dr for cubic-spline softening -- GIZMO's kernel_gravity(mode=1), mesh/kernel.h.
// Multiply by the offset vector to get the acceleration. Exactly Newtonian for r >= h, which
// Plummer softening never is (0.35 of Newtonian at r=eps, 0.72 at 2eps, 0.86 at 3eps): with
// adaptive softening that under-counts gravity across a whole neighbourhood rather than only
// below the resolution limit, and in Evrard it suppressed the central density ~2x. Sits beside
// grav_tidal_factor because the two are always used together -- force, tidal tensor and jerk
// all come from the same pair of kernel derivatives.
static inline double spline_force_over_r(double r, double h) {
    const double h_inv = 1.0 / h;
    const double h_inv3 = h_inv * h_inv * h_inv;
    const double u = r * h_inv;
    if (u >= 1.0) return 1.0 / (r * r * r);          // Newtonian outside the softening
    if (u < 0.5) return h_inv3 * (10.666666666666667 + u*u*(32.0*u - 38.4));
    return h_inv3 * (21.333333333333332 - 48.0*u + 38.4*u*u - 10.666666666666667*u*u*u
                     - 0.06666666666666667/(u*u*u));
}

// phi(r) for the same cubic spline -- GIZMO's kernel_gravity(mode=-1). Exactly -1/r beyond h, so
// the potential and the force come from one consistent kernel. Used by the potential walk and by
// the sink escape speed (sink_vesc, sinks/sink.cc:103), which is softened on the same kernel.
static inline double spline_potential(double r, double h) {
    const double h_inv = 1.0 / h;
    const double u = r * h_inv;
    if (u >= 1.0) return -1.0 / r;
    if (u < 0.5)
        return h_inv * (-2.8 + u*u*(5.333333333333333 + u*u*(6.4*u - 9.6)));
    return h_inv * (-3.2 + 0.06666666666666667/u
                    + u*u*(10.666666666666667 + u*(-16.0 + u*(9.6 - 2.1333333333333333*u))));
}

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
    // Barnes-Hut, live only where the config keeps ErrTolTheta nonzero. Under
    // GRAVITY_HYBRID_OPENING_CRIT that is always, and the two tests below form a union; otherwise
    // the reference retires this one after the first walk (gravtree.cc:489) and theta_sq is 0 by
    // then. The union opens strictly more nodes, so it is not free -- see Sim::hybrid_opening.
    if (theta_sq > 0.0 && l_open * l_open >= theta_sq * r_sq) return true;
    // Relative criterion. Its own rationale, from the reference: aold can be dominated by one
    // close companion while distant nodes still need resolving, and the monopole error left in a
    // node that should have been opened is a TORQUE, since its centre of force is wherever its
    // mass sits rather than on the line of centres. Measured on shu1977 with global timesteps,
    // angular momentum per gravity kick is 8.5e-12 for this test alone, 1.5e-12 for Barnes-Hut
    // alone, ~1e-22 for an exact O(N^2) sum -- but note the union did NOT cure the spurious
    // spin-up it was reached for.
    //
    // aold == 0 means no previous force for this target. The reference SKIPS this criterion
    // entirely on that first walk (forcetree.cc:1912, `!(Ti_Current==0 && RestartFlag!=1)`) and
    // bootstraps on Barnes-Hut alone, retiring ErrTolTheta only afterwards (gravtree.cc:491).
    // Letting aold == 0 fall through here instead collapses the right-hand side to zero, opens
    // every node and turns the walk into an O(N^2) direct sum: measured 48.6 s for one 128k step
    // against 1.8 s with this guard, and 76% of a shu1977 run to its first snapshot.
    if (aold > 0) {
        if (node_mass * len * len > r_sq * r_sq * aold) return true;
    } else if (theta_sq <= 0.0) {
        // No criterion left to decide with -- no previous acceleration AND no geometric test.
        // Open, which is the accurate-but-slow path. This is the case for a particle created
        // mid-run (a fresh sink has no a_prev) once theta has been retired, and it must not be
        // turned into "open nothing", which would hand that particle a badly wrong force.
        return true;
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

// The NEIGHBOUR SEARCH's own node, 32 bytes -- half of WNode, and it holds different geometry.
//
// The search wants the node's BOX (geometric centre, half side), which is what GIZMO prunes
// against (system/ngb_codeblock_checknode.h). The gravity walk wants the CENTRE OF MASS, because
// that is where the multipole sits. Sharing one node forced the search to test a sphere centred on
// the COM with radius size+delta, which admits ~2x the volume the reference's box test does
// (bench_nodegeom on the 3.5e6-cell state: 1.54x from using the full side where the circumsphere
// is 0.866*len, another ~1.3x from delta). Splitting them fixes the geometry and halves the bytes
// touched per visit -- and the traversal is 75% of the density solve, at ~12 ns per node visit
// against an 80 MB node array, so the bytes matter as much as the count.
//
// Positions are float. The box is O(1) and h is O(1e-3), so a float centre is accurate to ~2e-7
// absolute; `half` carries a conservative epsilon (see the pack loop) that swamps it by an order of
// magnitude, keeping the prune strictly one-sided. It can never drop a true neighbour.
struct SNode {
    float cx, cy, cz;    // 12  GEOMETRIC centre of the node, NOT the centre of mass
    float half;          //  4  half the side length, plus the float-safety epsilon
    int   first;         //  4  first child, or -1 for a leaf
    int   next;          //  4  where to go when the node is rejected
    int   plo, phi;      //  8  leaf particle range
};                       // = 32 bytes

struct Tree {
    // node arrays, indexed by node id; leaves store a particle range instead of children
    std::vector<double> cx, cy, cz;     // centre of mass
    std::vector<double> mass;           // total mass
    std::vector<double> size;           // side length
    std::vector<double> delta;          // |COM - geometric centre|, for the opening criterion
    std::vector<double> gcx, gcy, gcz;  // geometric centre; build-time only, packed into sn
    std::vector<double> soft;           // max softening in the node
    std::vector<int>    first;          // first child node, or -1 for a leaf
    std::vector<int>    next;           // next node to visit when this one is NOT opened
    std::vector<int>    plo, phi;       // particle range [plo, phi) for leaves
    std::vector<uint32_t> orderbuf;     // Morton-sorted particle indices; leaf ranges index this
    // Inverse of orderbuf: rank[i] is where particle i sits in Morton order. Lets a caller put a
    // SMALL active set into tree order by sorting it (O(k log k)) instead of scanning all N to
    // filter orderbuf -- that scan is O(N) whatever the active set size, which on a deep timestep
    // hierarchy is the dominant per-step cost. Built with orderbuf, so the two never disagree.
    std::vector<uint32_t> rank;
    std::vector<WNode>  wn;             // packed traversal copy, built once after the tree
    std::vector<SNode>  sn;             // packed neighbour-search copy; see SNode
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
    // Sinks contained in the node, for the direct-summation criterion (see open_node). A separate
    // array rather than a WNode field: WNode is exactly one cache line, and this is only read when
    // the target itself is a sink, which is a vanishing fraction of walks.
    std::vector<uint32_t> nsink;
    // Centre-of-mass velocity per node -- GIZMO's Extnodes[].vs. Only populated when the build
    // is given velocities AND a jerk is wanted (Hermite); empty otherwise, so runs that never
    // ask for a jerk pay neither the memory nor the sweep.
    std::vector<double> vcom_x, vcom_y, vcom_z;
    // Momentum accumulated into each node since the build -- GIZMO's Extnodes[].dp, maintained
    // by force_kick_node (forcetree_update.cc:96-100) in the same parent climb that raises vmax.
    // Without it vcom is only right at build time and every subsequent kick makes it staler; the
    // node velocity a walk should use is vcom + dp/mass.
    std::vector<double> dp_x, dp_y, dp_z;
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
    // Accumulate a particle's momentum change into every ancestor node (GIZMO's force_kick_node).
    // No-op unless the build produced vcom, so runs that never ask for a jerk pay nothing.
    void kick_node(int leaf_node, const Vec3d& dp);
    // Node centre-of-mass velocity, corrected for kicks since the build.
    Vec3d node_vel(int node_id) const {
        if (dp_x.empty() || mass[node_id] <= 0)
            return Vec3d{vcom_x[node_id], vcom_y[node_id], vcom_z[node_id]};
        const double inv_m = 1.0 / mass[node_id];
        return Vec3d{vcom_x[node_id] + dp_x[node_id] * inv_m,
                     vcom_y[node_id] + dp_y[node_id] * inv_m,
                     vcom_z[node_id] + dp_z[node_id] * inv_m};
    }
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
// `randomize_seed`, when >= 0, turns on RANDOMIZE_GRAVTREE: the root box is displaced by a random
// offset of up to half a side per axis and then DOUBLED so it still covers everything (GIZMO's
// domain.cc:2722-2731). Every node wall in the tree moves with it, so the walk's force errors --
// which are a function of where the cell walls fall relative to the mass -- are redrawn on each
// build instead of repeating. Uncorrelated errors average out over a run; a fixed grid's errors
// accumulate into a secular drift. Pass the step number so successive builds differ.
Tree build(const Particles& P, BuildTimes* bt = nullptr, const double* const* vel = nullptr,
           bool want_vcom = false, long long randomize_seed = -1);

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
// `jerk`, when non-null, receives da/dt per target in the SAME walk -- GIZMO's
// COMPUTE_JERK_IN_GRAVTREE (forcetree.cc:2266):
//     jerk += g1 * dv - (dv . dr) * g2 * dr
// with g1/g2 the same first/second kernel-derivative factors the tidal tensor already needs, so
// the extra cost is a few FLOPs per interaction rather than a second traversal. dv is the source
// velocity minus the target's: the particle's own for a leaf (forcetree.cc:1641), the node's
// centre-of-mass velocity for a multipole (forcetree.cc:1945, Extnodes[].vs -- Tree::vcom here).
// Requires a tree built with want_vcom. `vel` is the engine's live {vx,vy,vz}; it supplies both
// the sources' and the targets' velocities, so a Hermite predictor step simply writes its
// predicted state into the particle arrays before calling (as the reference's
// do_hermite_prediction does) and the walk picks it up.
void accel_grouped(const Tree& T, const Particles& P, const std::vector<uint32_t>& targets,
                   double theta, double G, int batch /* = 8 */, std::vector<double>& ax,
                   std::vector<double>& ay, std::vector<double>& az,
                   std::vector<SymTensor3d>* tidal = nullptr, const double* aold = nullptr,
                   const LazyDrift* lazy = nullptr,
                   std::vector<Vec3d>* jerk = nullptr,
                   const double* const* vel = nullptr,
                   // SINK-SINK DIRECT SUMMATION (forcetree.cc:1973-1979). When a SINK is the target
                   // and a node holds sinks within this radius, open it and sum the pairs exactly
                   // rather than accepting a multipole. Collisional pairs are what a multipole
                   // approximates worst, and a binary integrated through one drifts in energy.
                   // In code length units; 0 disables. The reference's default is 1000 AU.
                   double sink_direct_radius = 0.0);

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
