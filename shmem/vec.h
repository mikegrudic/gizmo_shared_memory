// Bridge from the shared-memory engine's SoA field storage to the repo's own math types.
//
// The bulk per-particle fields stay SoA (x[], y[], z[], vx[], ...): that layout is what makes the
// tree build and neighbour search fast, and it was measured, not assumed. But every piece of
// GEOMETRY and TENSOR ALGEBRA derived from those fields -- pair offsets, the moments matrix E, its
// inverse B, the 5 field gradients, the effective face A_ij -- is expressed in Vec3/Mat3/
// SymmetricTensor2 from math_types/, so the algebra reads as algebra and the inverse/outer-product
// code is the repo's shared, unit-tested implementation rather than a second hand-rolled copy.

#pragma once
#include <cmath>
#include "../math_types/vec3.h"
#include "../math_types/mat3.h"
#include "../math_types/symmetric_tensor2.h"

namespace shmem {

using Vec3d      = Vec3<double>;
using Mat3d      = Mat3<double>;
using SymTensor3d = SymmetricTensor2<double>;   // rank-2 symmetric tensor in 3 dimensions

// Minimum-image SEPARATION for periodic boxes, folded into [-box/2, box/2); box <= 0 means
// non-periodic. Valid while the query radius + node size stays below box/2, which holds for
// kernel-scale searches.
static inline double min_image(double separation, double box) {
    if (box > 0) {
        if (separation >  0.5*box) separation -= box;
        else if (separation < -0.5*box) separation += box;
    }
    return separation;
}
static inline Vec3d min_image(Vec3d separation, double box) {
    return Vec3d{min_image(separation[0], box),
                 min_image(separation[1], box),
                 min_image(separation[2], box)};
}

// Fold a POSITION back into the canonical cell [0, box). Distinct from min_image() above, which
// folds a separation. Uses fmod rather than one add/subtract because a drift can cross more than
// one box length in a single step: the square test advects at |v| ~ 1300 with dt ~ 1e-3, i.e. 1.3
// box lengths per step, and a lone `coord -= box` would leave the coordinate outside the box.
static inline double fold_into_box(double coord, double box) {
    if (box <= 0) return coord;
    // Fast paths first. This runs for EVERY particle EVERY step, and fmod is a libm call an order
    // of magnitude dearer than a compare -- with millions of particles it was the single largest
    // cost in the drift. Almost always the coordinate is already inside, or one box out; the fmod
    // is kept only for the genuine multi-box jump (a fast advection test can cross more than one
    // box length per step, which is why a lone subtract is not enough).
    if (coord >= 0.0 && coord < box) return coord;
    if (coord >= box && coord < 2.0*box) return coord - box;
    if (coord < 0.0 && coord >= -box)   return coord + box;
    coord = std::fmod(coord, box);
    return (coord < 0) ? coord + box : coord;
}
static inline Vec3d fold_into_box(Vec3d pos, double box) {
    return Vec3d{fold_into_box(pos[0], box),
                 fold_into_box(pos[1], box),
                 fold_into_box(pos[2], box)};
}

}  // namespace shmem
