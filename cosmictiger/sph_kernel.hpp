/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#pragma once

#include <cosmictiger/kernel.hpp>
#include <cosmictiger/sph.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/sph_tree.hpp>

struct SphRunData {
    // Staged source arrays, including the sources that are also local sinks.
    Fixed32* sourcePositionX;
    Fixed32* sourcePositionY;
    Fixed32* sourcePositionZ;
    float* sourceVelocityX;
    float* sourceVelocityY;
    float* sourceVelocityZ;
    float* sourceInternalEnergy;
    float* sourceDensity;
    float* sourceSupportRadius;
    char* sourceRungs;

    // Local sink arrays indexed by the SPH particle index.
    float* sinkInternalEnergyRate;
    float* sinkDensity;
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
CUDA_EXPORT inline bool sphSupportIntersects(
    Fixed32Range const& sinkBox, Array<Fixed32, NDIM> const& sourcePosition, float sourceSupport) {
    Fixed32Range expanded = sinkBox;
    for (int dim = 0; dim < NDIM; dim++) {
        expanded.begin[dim] -= RangeFixed(sourceSupport);
        expanded.end[dim] += RangeFixed(sourceSupport);
    }
    return expanded.contains(sourcePosition);
}
