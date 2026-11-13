/* CosmicTiger - float periodic global FMM; BSL-1.0. */
#pragma once

#include <cmath>
#include <numbers>

#include <cosmictiger/gravity/fmm_acceptance.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree/ewald_green.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree/ewald_m2l.hpp>
#include <cosmictiger/gravity/tree.hpp>
#ifndef __CUDACC__
#include <cosmictiger/gravity/source_access.hpp>
#endif

namespace detail {

// True when every pair in these enclosing spheres selects the center image.
CUDA_EXPORT inline bool periodicFmmSingleImage(
    Vector3<float> const& displacement, float sinkRadius, float sourceRadius) {
    return fmaxf(fabsf(displacement[0]), fmaxf(fabsf(displacement[1]), fabsf(displacement[2]))) +
        (sinkRadius + sourceRadius) <
        .5f;
}

// Derived unit-box image distances. For a minimum-image displacement d,
// R1²=d.d and R2²=R1²+1-2 max(|d_x|,|d_y|,|d_z|).
// The correction excludes the nearest singularity. Both parts of that split
// must use a single periodic image throughout an accepted expansion. Open cells
// crossing an image-selection plane and evaluate unresolved pairs directly.
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
    bool const sameImage = largest + radius < 0.5f;
    bool const separated = opening * opening < secondSquared;
    if (operations != nullptr) {
        // Includes three absolute values and two maxima, each one math result.
        *operations = 17;
    }
    return sameImage && separated;
}

// Derived constant-Laplacian contraction, in tree coordinates y=kappa*r:
// G = H + K dz²/2, K=-4pi/kappa³; Qzz=M_STF,zz+Tr(Q)/3.
// L0 += K Qzz/2; Lz -= K Mz; curvature += K M0.
// H uses the stored xx/yy derivatives and reconstructs zz harmonically.
// Symbol directory: G is the Ewald correction, H its local harmonic proxy,
// K its Laplacian, M the source STF moments, Q its raw second moment.
CUDA_EXPORT inline FlopCount periodicFmmM2L(
    cosmictiger::gravity::cartesian::Multipole<float> const& source, float radialTrace,
    Vector3<float> const& displacement, bool doPotential,
    cosmictiger::gravity::cartesian::Expansion<float>& local, float& curvature) {
    namespace cartesian = cosmictiger::gravity::cartesian;
    cartesian::Green<float> green;
    cartesian::evaluateEwaldGreen(displacement, green);
    float const oldPotential = local[0];
    cartesian::M2L(source, green, local);
    constexpr float kCurvature = -4.0f * std::numbers::pi_v<float> * kScaleFactorInverse3;
    if constexpr (kFmmOrder >= 4) {
        float const secondZ = -source(2, 0, 0) - source(0, 2, 0) + radialTrace / 3.0f;
        local[0] += .5f * kCurvature * secondZ;
    }
    local(0, 0, 1) -= kCurvature * source(0, 0, 1);
    curvature += kCurvature * source[0];
    if (!doPotential) {
        local[0] = oldPotential;
    }
    return cartesian::kEwaldGreenFlops + cartesian::kEwaldM2LFlops + 4 + (kFmmOrder >= 4 ? 6 : 0);
}

CUDA_EXPORT inline FlopCount periodicFmmPoint(Vector3<float> const& displacement, float mass,
    bool doPotential, float& potential, float& gx, float& gy, float& gz) {
    namespace cartesian = cosmictiger::gravity::cartesian;
    cartesian::Green<float> green;
    cartesian::evaluateEwaldGreen(displacement, green);
    if (doPotential) {
        potential += mass * kScaleFactor * green[0];
    }
    float const scale = mass * kScaleFactor2;
    gx -= scale * green(1, 0, 0);
    gy -= scale * green(0, 1, 0);
    gz -= scale * green(0, 0, 1);
    return cartesian::kEwaldGreenFlops + 7 + (doPotential ? 3 : 0);
}

#ifndef __CUDACC__
inline FlopCount cpuPeriodicFmm(ExpansionBuffer& local, TreeNode const& sink,
    Vector<TreeId>& checks, ForceVectors& forces, float theta, float smoothingLength,
    bool doPotential) {
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
        bool const accepted = periodicFmmAccepted(displacement, sink.radius, source.radius,
            inverseTheta, smoothingLength, &acceptanceFlops);
        flops += 3 + acceptanceFlops;
        if (accepted) {
            flops += periodicFmmM2L(fmmMultiPos(source.mpos).multi, source.radialTrace,
                displacement, doPotential, local.expansion(), local.curvature);
        } else if (!source.leaf && (sink.leaf || source.radius >= sink.radius)) {
            checks.push_back(source.children[kLeftChild]);
            checks.push_back(source.children[kRightChild]);
        } else if (!sink.leaf) {
            deferred.push_back(id);
        } else {
            // Rare unresolved leaf pair: evaluate the regular correction,
            // including self images. No singular 1/r is evaluated here.
            auto const count = source.numberOfParticles();
            Vector<Position> x(count), y(count), z(count);
            Vector<float> masses(count);
            gravityReadSourcePositions(
                source, x.data(), y.data(), z.data(), nullptr, masses.data(), 0);
            Position const* positions[kDimensions] = {x.data(), y.data(), z.data()};
            for (ParticleInteger i = sink.particleRange.first; i < sink.particleRange.second; ++i) {
                auto const target = i - sink.particleRange.first;
                for (ParticleInteger j = 0; j < count; ++j) {
                    for (int dim = 0; dim < kDimensions; ++dim) {
                        displacement[dim] = distance(particlesPos(dim, i), positions[dim][j]);
                    }
                    flops += 3 +
                        periodicFmmPoint(displacement, masses[j], doPotential, forces.phi[target],
                            forces.gx[target], forces.gy[target], forces.gz[target]);
                }
            }
        }
    }
    checks = std::move(deferred);
    return flops;
}
#endif
}    // namespace detail
