/*
 CosmicTiger - A cosmological N-Body code
 Typed standalone MFM prehydro parameters, staged arrays, and completion state.
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cstdint>
#include <limits>

#include <cosmictiger/mfm/geometry.hpp>
#include <cosmictiger/sph/particle_flags.hpp>
#include <cosmictiger/sph/tree.hpp>

namespace mfm {

template <typename Real>
struct PrehydroParameters {
    Real minimumSmoothingLength = Real(1e-6);      // Minimum full support, in box units.
    Real maximumSmoothingLength = Real(0.5);       // Maximum full support, at most half a box.
    Real smoothingLengthTolerance = Real(1e-4);    // Relative kernel-sum convergence tolerance.
    Real maximumMomentCondition = Real(1e4);       // Maximum dimensionless infinity-norm condition.
    int minimumRung = 0;               // Lowest timestep rung active in this pass, inclusive.
    int maximumRung = kMaxRung - 1;    // Highest timestep rung active in this pass, inclusive.
    bool initializeDensity = false;    // Publish density during the initial support solve.

    CUDA_EXPORT bool isActiveRung(int rung) const {
        return rung >= minimumRung && rung <= maximumRung;
    }
};

template <typename Real>
struct PrehydroResult {
    bool needsAnotherPass = false;    // At least one target needs a larger staged neighbor box.
    int geometryFailures = 0;         // Number of evaluated targets whose inverse was rejected.
    Real minimumSmoothingLength = std::numeric_limits<Real>::max();    // Smallest converged h.
    Real maximumSmoothingLength = Real(0);    // Largest converged h; zero for an empty pass.

    CUDA_EXPORT PrehydroResult& operator+=(PrehydroResult const& other) {
        needsAnotherPass = needsAnotherPass || other.needsAnotherPass;
        geometryFailures += other.geometryFailures;
        minimumSmoothingLength = math::min(minimumSmoothingLength, other.minimumSmoothingLength);
        maximumSmoothingLength = math::max(maximumSmoothingLength, other.maximumSmoothingLength);
        return *this;
    }
};

template <typename Real, int dimensions>
struct RunData {
    static_assert(dimensions >= 1 && dimensions <= kDimensions);
    Position const* sourcePositionX = nullptr;      // Frozen periodic source x coordinate.
    Position const* sourcePositionY = nullptr;      // Frozen periodic source y coordinate.
    Position const* sourcePositionZ = nullptr;      // Frozen periodic source z coordinate.
    Real const* sourceSmoothingLength = nullptr;    // Frozen source full support in box units.
    char const* sourceRungs = nullptr;              // Frozen source rung for semiactive selection.
    Real* sinkDensity = nullptr;    // Local gas mass density, in code mass/box-volume units.
    Real* sinkSmoothingLength = nullptr;    // Local full support, updated by the bounded solve.
    math::SymmetricMatrix<Real, dimensions>* sinkInverseMoment = nullptr;    // Persistent B=E⁻¹.
    uint8_t* sinkGeometryValid = nullptr;     // One only when the corresponding B is usable.
    SphParticleFlags* sinkFlags = nullptr;    // Existing convergence and semiactivity flags.
    ParticleInteger* sinkParticleIndices = nullptr;    // Local gas-to-full-particle index map.
    char* sinkParticleRungs = nullptr;         // Full-particle rung array, indexed through the map.
    SphTreeNode const* treeNodes = nullptr;    // Immutable staged source and target nodes.
    int const* targetNodeIndices = nullptr;    // Target-node entries in treeNodes.
    int const* neighborNodeIndices = nullptr;    // Flattened staged neighbor-node entries.
    int numberOfTargetNodes = 0;                 // Number of entries in targetNodeIndices.
    Real gasMass = Real(0);                // Equal mass of every gas particle, in code mass units.
    Real targetNeighborCount = Real(0);    // Weighted neighbor target of the support solve.

    CUDA_EXPORT math::Vector<Position, dimensions> sourcePosition(ParticleInteger index) const {
        Position const* const components[kDimensions] = {
            sourcePositionX, sourcePositionY, sourcePositionZ};
        math::Vector<Position, dimensions> result;
        for (int dim = 0; dim < dimensions; ++dim) {
            result[dim] = components[dim][index];
        }
        return result;
    }

    CUDA_EXPORT Vector3<Position> sourcePaddedPosition(ParticleInteger index) const {
        return Vector3<Position>(
            sourcePositionX[index], sourcePositionY[index], sourcePositionZ[index]);
    }
};

}    // namespace mfm
