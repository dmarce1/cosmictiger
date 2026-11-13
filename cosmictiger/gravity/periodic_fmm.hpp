/* CosmicTiger - float periodic global FMM; BSL-1.0. */
#pragma once

#include <cmath>

#include <cosmictiger/gravity/fmm_acceptance.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree/ewald_green.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree/ewald_m2l.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp>
#include <cosmictiger/gravity/tree.hpp>

namespace detail {

// Derived unit-box image distances. For a minimum-image displacement d,
// R1²=d.d and R2²=R1²+1-2 max(|d_x|,|d_y|,|d_z|).
// The correction excludes the nearest image selected by the expansion centers.
// Its opening criterion uses the distance to the second-nearest image. Pairs
// crossing an image-selection plane resolve the Newtonian term at these same
// centers, so the primary-image choice cancels from the complete periodic field.
// Symbol directory: R1,R2 are the first/second image distances; d is separation.
CUDA_EXPORT inline bool periodicFmmAccepted(Vector3<float> const& displacement, float sinkRadius,
    float sourceRadius, float inverseTheta, float /* smoothingLength */,
    FlopCount* operations = nullptr) {
    float const radius = sinkRadius + sourceRadius;
    float const largest =
        fmaxf(fabsf(displacement[0]), fmaxf(fabsf(displacement[1]), fabsf(displacement[2])));
    float const nearestSquared = displacement.dot(displacement);
    float const secondSquared = nearestSquared + 1.0f - 2.0f * largest;
    float const opening = radius * inverseTheta;
    bool const separated = opening * opening < secondSquared;
    if (operations != nullptr) {
        // Includes three absolute values and two maxima, each one math result.
        *operations = 16;
    }
    return separated;
}

// Derived cell-cell criteria: R1 > (rs + rt)/theta and R1 > rs + rt + h.
// Symbol directory: R1 is center separation; rs,rt are enclosing radii;
// theta is the opening angle; h is the pair-softening support.
CUDA_EXPORT inline bool periodicFmmDirectAccepted(Vector3<float> const& displacement,
    float sinkRadius, float sourceRadius, float inverseTheta, float smoothingLength,
    FlopCount* operations = nullptr) {
    float const radius = sinkRadius + sourceRadius;
    float const opening = radius * inverseTheta;
    float const support = radius + smoothingLength;
    float const distanceSquared = displacement.dot(displacement);
    bool const separated = opening * opening < distanceSquared;
    bool const supportSeparated = support * support < distanceSquared;
    if (operations != nullptr) {
        *operations = 10;
    }
    return separated && supportSeparated;
}

// The generated Ewald contraction retains the complete quadratic tensor in the
// local expansion; higher derivatives and moments remain trace-free. When both
// terms are pending, their common center-selected image cancels algebraically.
CUDA_EXPORT inline FlopCount periodicFmmM2L(
    cosmictiger::gravity::cartesian::Multipole<float> const& source,
    Vector3<float> const& displacement, bool doPotential,
    cosmictiger::gravity::cartesian::Expansion<float>& local, bool includeNewtonian = false) {
    namespace cartesian = cosmictiger::gravity::cartesian;
    static_assert(cartesian::kQuadrupoleTrace,
        "Periodic gravity requires Cartesian quadrupole trace storage");
    cartesian::Green<float> green;
    cartesian::evaluateEwaldGreen(displacement, green);
    float const oldPotential = local[0];
    cartesian::M2Lewald(source, green, local);
    FlopCount flops = cartesian::kEwaldGreenFlops + cartesian::kEwaldM2LFlops;
    if (includeNewtonian) {
        cartesian::M2L(source, fmmScaleDisplacement(displacement), local);
        flops += 3 + cartesian::kM2LFlops<fmmLocalDegree(kFmmOrder)>;
    }
    if (!doPotential) {
        local[0] = oldPotential;
    }
    return flops;
}

#ifndef __CUDACC__
// checks holds pairs with both primary and periodic terms pending. Only pairs
// whose excluded image is uniform may hand their primary term to directChecks;
// all image-crossing pairs are consumed here by the coupled cell-cell kernels.
inline FlopCount cpuPeriodicFmm(ExpansionBuffer& local, TreeNode const& sink,
    Vector<TreeId>& checks, Vector<TreeId>& directChecks, float theta, float smoothingLength,
    bool doPotential, long long& directCellCount) {
    FlopCount flops = 1;
    float const inverseTheta = fmmInverseOpeningAngle(theta);
    Vector<TreeId> deferred;
    while (!checks.empty()) {
        TreeId const id = checks.back();
        checks.pop_back();
        auto const& source = *treeGetNode(id);
        if (source.isLocal() && source.numberOfParticles() == 0) {
            continue;
        }
        Vector3<float> displacement;
        for (int dim = 0; dim < kDimensions; ++dim) {
            displacement[dim] = distance(sink.pos[dim], source.pos[dim]);
        }
        FlopCount acceptanceFlops;
        bool const separated = periodicFmmAccepted(displacement, sink.radius, source.radius,
            inverseTheta, smoothingLength, &acceptanceFlops);
        flops += 3 + acceptanceFlops;
        bool singleImage = false;
        bool includeNewtonian = false;
        if (separated) {
            singleImage = fmmSinglePeriodicImage(displacement, sink.radius, source.radius);
            flops += 7;
            if (!singleImage) {
                includeNewtonian = periodicFmmDirectAccepted(displacement, sink.radius,
                    source.radius, inverseTheta, smoothingLength, &acceptanceFlops);
                flops += acceptanceFlops;
            }
        }
        if (singleImage || includeNewtonian) {
            flops += periodicFmmM2L(fmmMultiPos(source.mpos).multi, displacement, doPotential,
                local.expansion(), includeNewtonian);
            if (singleImage) {
                directChecks.push_back(id);
            } else {
                ++directCellCount;
            }
        } else if (!source.leaf && (sink.leaf || source.radius >= sink.radius)) {
            checks.push_back(source.children[kLeftChild]);
            checks.push_back(source.children[kRightChild]);
        } else {
            // The mandatory Ewald tree subdivision guarantees that every leaf
            // pair passes either the uniform-image or coupled acceptance path.
            ALWAYS_ASSERT(!sink.leaf);
            deferred.push_back(id);
        }
    }
    checks = std::move(deferred);
    return flops;
}
#endif
}    // namespace detail
