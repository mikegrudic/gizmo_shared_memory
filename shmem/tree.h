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

namespace shmem {

struct Particles {                 // SoA: the walk reads x/y/z/m for many particles at once
    std::vector<double> x, y, z, m, soft;
    size_t size() const { return m.size(); }
};

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
    int    first;        //  4  first child, or -1 for a leaf
    int    next;         //  4  where to go when not opened
    int    plo, phi;     //  8  leaf particle range
};                       // = 64 bytes exactly

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

    size_t nnodes() const { return mass.size(); }   // valid after build() trims to nalloc
};

static const int LEAF_MAX = 16;        // particles per leaf; below this, direct summation is cheaper
static const int MAX_LEVEL = 20;       // Morton keys carry 21 bits per axis

// Build a subtree over the Morton-sorted range [lo, hi). Returns the new node's index.
// `order` maps sorted position -> particle index; `key` is the sorted key array.
int build_node(Tree& T, const Particles& P, const std::vector<uint32_t>& order,
               const std::vector<uint64_t>& key, int lo, int hi, int level,
               double cxi, double cyi, double czi, double sz);

// Assign next[] so a walk that does not open a node can jump straight past its subtree.
void setup_walk(Tree& T, int node, int next_sibling);

// Stage timings, so build cost is attributed rather than guessed.
struct BuildTimes { double bbox=0, keys=0, sort=0, recurse=0, links=0, pack=0, total=0; };

// Build the whole tree. Parallel key computation and walk-link setup; the recursive build is
// task-parallel over disjoint ranges.
Tree build(const Particles& P, BuildTimes* bt = nullptr);

// Accelerations for the listed targets, Barnes-Hut with opening angle theta.
// `targets` is the ACTIVE list -- the whole point is that it is usually tiny compared to P.
void accel(const Tree& T, const Particles& P, const std::vector<uint32_t>& targets,
           double theta, double G, std::vector<double>& ax, std::vector<double>& ay,
           std::vector<double>& az);

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
void accel_grouped(const Tree& T, const Particles& P, const std::vector<uint32_t>& targets,
                   double theta, double G, int batch /* = 8 */, std::vector<double>& ax,
                   std::vector<double>& ay, std::vector<double>& az);

// Direct O(N*M) summation, for validating the tree.
void accel_brute(const Particles& P, const std::vector<uint32_t>& targets, double G,
                 std::vector<double>& ax, std::vector<double>& ay, std::vector<double>& az);

}  // namespace shmem
