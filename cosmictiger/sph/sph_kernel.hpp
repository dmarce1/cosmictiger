/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/sph_particles.hpp>
#include <cosmictiger/sph/sph_tree.hpp>

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
    float* sourceDensity;
    float* sourceBalsaraFactor;
    float* sourceSupportRadius;
    char* sourceRungs;

    // Local sink arrays indexed by the SPH particle index.
    float* sinkInternalEnergyRate;
    float* sinkDensity;
    float* sinkBalsaraFactor;
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

#define MAX_NEIGHBOR_RUNG_DIFFERENCE 1
#define SPH_SUPPORT_RADIUS_TOLERANCE float(1.0e-4)

// A source can reach any sink in this box through a periodic image. Expanding
// the unwrapped interval before testing avoids inconsistent endpoint wrapping.
CUDA_EXPORT inline bool sphSupportIntersects(Fixed32Range const& sinkBox,
    Array<Position, NDIM> const& sourcePosition, float sourceSupport,
    FlopCounter<>* flops = nullptr) {
    Fixed32Range expanded = sinkBox;
    for (int dim = 0; dim < NDIM; dim++) {
        expanded.begin[dim] -= RangeFixed(sourceSupport);
        expanded.end[dim] += RangeFixed(sourceSupport);
    }
    if (flops) {
        *flops += 2 * NDIM;    // Two support-to-fixed scaling products per dimension.
    }
    return expanded.contains(sourcePosition, flops);
}
