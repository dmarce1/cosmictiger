/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <numbers>

#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/flops.hpp>
#include <cosmictiger/sph/pair_kick.hpp>
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/tree.hpp>
#include <cosmictiger/sph/volume.hpp>

struct SphRunData {
    // Staged source arrays, including the sources that are also local sinks.
    Position* sourcePositionX;
    Position* sourcePositionY;
    Position* sourcePositionZ;
    float* sourceVelocityX;
    float* sourceVelocityY;
    float* sourceVelocityZ;
    float* sourceInternalEnergy;
    float* sourceDensity;
    float* sourceBalsaraFactor;
    float* sourceBulkViscosityCoefficient;
    float* sourceAccelerationX;
    float* sourceAccelerationY;
    float* sourceAccelerationZ;
    float* sourceVelocityDivergence;
    float* sourceSupportRadius;
    // Variational operator: zeta_i of cosmictiger/sph/volume.hpp from this step's density
    // pass.
    float* sourceGradHCorrection;
    // Frozen before sink writes; required by prehydro and rung limiting.
    char* sourceRungs;
    // Optional conservative midpoint output. Source identities and double
    // increments live only for the frozen pass and its reverse MPI exchange.
    SphPairSourceId const* sourceIds = nullptr;
    SphPairUpdate* sourcePairUpdates = nullptr;

    // Local sink arrays indexed by the SPH particle index.
    // Optional total dE/dtau diagnostic; it is not integrated by production kicks.
    float* sinkInternalEnergyRate = nullptr;
    float* sinkEntropyRate = nullptr;    // dA/dtau from irreversible heating only.
    float* sinkDensity;
    // Optional d(log rho_comoving)/d(tau), including the adaptive-support
    // correction. Requires the source velocity arrays even with viscosity off.
    float* sinkLogDensityRate = nullptr;
    float* sinkGradHCorrection;    // Rebuilt with the density on every preparation pass.
    float* sinkBalsaraFactor;
    Float16Storage* sinkBulkViscosityCoefficient;
    SphCullenDehnenState const* sinkCullenDehnenStates;
    // Event-local density results; absent when this pass will not update history.
    SphViscosityDiagnostics* sinkViscosityDiagnostics = nullptr;
    char* sinkSemiactive;
    char* sinkSmoothingConverged;
    float* sinkSupportRadius;
    SphAcceleration* sinkAccelerations;
    float* sinkGravityAccelerationX;
    float* sinkGravityAccelerationY;
    float* sinkGravityAccelerationZ;
    ParticleInteger* sinkParticleIndices;
    char* sinkParticleRungs;    // Indexed through sinkParticleIndices into the full particle array.
    // Optional: a phase-1 hydro pass writes its rung proposals here instead of into
    // sinkParticleRungs, so the closing rung stays in place until the driver commits them.
    char* sinkProposedRungs = nullptr;
    // Optional globally closed rung candidates, indexed by full particle index.
    char* sinkRungCandidates = nullptr;

    int numberOfTargetNodes;
    SphTreeNode* treeNodes;
    int* targetNodeIndices;
    int* neighborNodeIndices;
    float targetNeighborCount;    // Weighted neighbor target for the support-radius solve.
    float gasMass;
    float adiabaticIndex;    // Ideal-gas index shared by all gas particles.
    bool selfGravityEnabled;
    float gravityEffectiveRadius;        // Twice the gravity kernel standard deviation.
    float accelerationTimestepFactor;    // Coefficient of the acceleration timestep bound.
    // Optional external central potential, used by the orbiting-disc problem.
    float centralAcceleration;     // Zero disables this force.
    float centralSupportRadius;    // Support radius of the selected gravity kernel.
};

// Preserve the force-source arithmetic order for both the unsplit kernels and
// the post-gravity completion of a phase-zero pair pass.
CUDA_EXPORT inline Vector3<float> sphCompleteAcceleration(SphRunParameters const& parameters,
    Vector3<Position> const& position, Vector3<float> const& velocity, Vector3<float> acceleration,
    Vector3<float> const& gravityAcceleration, float centralAcceleration,
    float centralSupportRadius, sph_flops::Counter& flops) {
#if COSMICTIGER_WITH_GRAVITY
    acceleration += gravityAcceleration;
    flops += 3;
#endif
    acceleration[kYDimension] += parameters.uniformAccelerationY;
    acceleration[kZDimension] += parameters.midplaneAccelerationZ(position[kZDimension]);
    flops += 2;
    flops.doublePrecision += sph_flops::positionConstructionFlops();
    if (centralAcceleration != 0.0f) {
        Vector3<float> const separation(distance(position[kXDimension], Position(0.5)),
            distance(position[kYDimension], Position(0.5)),
            distance(position[kZDimension], Position(0.5)));
        float const radius = separation.norm();
        float const inverseSupportRadius = 1.0f / centralSupportRadius;
        float const normalizedRadius = radius * inverseSupportRadius;
        float const force = centralAcceleration *
            radial_kernel::forceFactor(selectedGravityKernel(), normalizedRadius) *
            inverseSupportRadius * inverseSupportRadius * inverseSupportRadius;
        acceleration -= force * separation;
        flops += 21 + sph_flops::forceKernelOperations(selectedGravityKernel(), normalizedRadius);
        flops.doublePrecision += 3 * sph_flops::positionConstructionFlops();
    }
    acceleration -= parameters.velocityDampingRate * velocity;
    flops += 6;
    return acceleration;
}

inline constexpr int kMaxNeighborRungDifference = 1;
inline constexpr float kSphSupportRadiusTolerance = float(1.0e-4);

// Target of the support solves: ΣW = Σ_b w(q_ab) = 3 N_target/(4π).
CUDA_EXPORT inline float sphTargetKernelSum(float targetNeighborCount) {
    return targetNeighborCount * (3.0f / (4.0f * std::numbers::pi_v<float>) );
}

struct SphGradHEvaluation {
    sph_volume::GradHCorrection<float> correction;
    sph_volume::SupportConstraint constraint;
    float relativeResidual;    // (ΣW − target)/target at the stored support radius
};

// The variational density pass's correction for one sink, from its unweighted sums
// numberSum = Σ_b w(q_ab) and numberMoment = Σ_b q_ab w'(q_ab) at the converged support.
// Every gas particle has the mass gasMass and the volume weight x = gasMass.
CUDA_EXPORT inline SphGradHEvaluation sphGradHEvaluation(float supportRadius,
    float minimumSupportRadius, float maximumSupportRadius, float targetNeighborCount,
    float gasMass, float numberSum, float numberMoment) {
    float const target = sphTargetKernelSum(targetNeighborCount);
    SphGradHEvaluation result;
    result.relativeResidual = (numberSum - target) / target;
    result.constraint = sph_volume::classifySupport(supportRadius, minimumSupportRadius,
        maximumSupportRadius, result.relativeResidual, kSphSupportRadiusTolerance);
    result.correction = sph_volume::gradHCorrection(
        sph_volume::equalWeightSupportSums(numberSum, numberMoment, gasMass), result.constraint);
    return result;
}

// A source can reach any sink in this box through a periodic image. Expanding
// the unwrapped interval before testing avoids inconsistent endpoint wrapping.
CUDA_EXPORT inline bool sphSupportIntersects(Fixed32Range const& sinkBox,
    Vector3<Position> const& sourcePosition, float sourceSupport,
    sph_flops::Counter* flops = nullptr) {
    Fixed32Range expanded = sinkBox;
    for (int dim = 0; dim < kDimensions; dim++) {
        expanded.begin[dim] -= RangeFixed(sourceSupport);
        expanded.end[dim] += RangeFixed(sourceSupport);
    }
    if (flops) {
        // Each float support needs one single-precision fixed-point scale.
        flops->singlePrecision += 2 * kDimensions;
    }
    return expanded.contains(sourcePosition, flops ? &flops->doublePrecision : nullptr);
}
