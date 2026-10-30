/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <numbers>

#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/sph_particles.hpp>
#include <cosmictiger/sph/sph_tree.hpp>
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
    // Optional local-composition EOS. Null retains the configured ideal gas.
    float* sourcePressureEnergyRatio;
    float* sourceAdiabaticIndex;
    float* sourceHydroCoupling;    // SH03 winds: zero excludes both directions of hydro forces.
    float* sourceDensity;
    float* sourceBalsaraFactor;
    float* sourceViscosityAlpha;
    float* sourceAccelerationX;
    float* sourceAccelerationY;
    float* sourceAccelerationZ;
    float* sourceVelocityDivergence;
    float* sourceSupportRadius;
    // Variational operator: zeta_i of cosmictiger/sph/volume.hpp from this step's density pass.
    float* sourceGradHCorrection;
    char* sourceRungs;

    // Local sink arrays indexed by the SPH particle index.
    float* sinkInternalEnergyRate;
    float* sinkDensity;
    float* sinkGradHCorrection;    // Rebuilt with the density on every preparation pass.
    float* sinkBalsaraFactor;
    SphViscosityState* sinkViscosityStates;
    SphCullenDehnenState* sinkCullenDehnenStates;
    char* sinkSemiactive;
    char* sinkSmoothingConverged;
    SphThermodynamicState* sinkThermodynamicStates;
    SphAcceleration* sinkAccelerations;
    float* sinkGravityAccelerationX;
    float* sinkGravityAccelerationY;
    float* sinkGravityAccelerationZ;
    ParticleInteger* sinkParticleIndices;
    char* sinkParticleRungs;    // Indexed through sinkParticleIndices into the full particle array.

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
    Array<Position, kDimensions> const& sourcePosition, float sourceSupport,
    FlopCounter<>* flops = nullptr) {
    Fixed32Range expanded = sinkBox;
    for (int dim = 0; dim < kDimensions; dim++) {
        expanded.begin[dim] -= RangeFixed(sourceSupport);
        expanded.end[dim] += RangeFixed(sourceSupport);
    }
    if (flops) {
        *flops += 2 * kDimensions;    // Two support-to-fixed scaling products per dimension.
    }
    return expanded.contains(sourcePosition, flops);
}
