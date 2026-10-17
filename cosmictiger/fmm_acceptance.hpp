/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#pragma once

#include <cmath>

#include <cosmictiger/cuda.hpp>

enum class FmmInteraction { CellCell, ParticleCell, CellParticle, ParticleParticle };

constexpr int kMinimumHybridParticleCount = 38;

template <class T>
struct FmmAcceptance {
    T cellCell;
    T particleCell;
    T cellParticle;
    T supportSeparated;
};

namespace detail {

CUDA_EXPORT inline float fmmMaximum(float left, float right) {
    return std::fmax(left, right);
}

CUDA_EXPORT inline double fmmMaximum(double left, double right) {
    return std::fmax(left, right);
}

template <class T>
CUDA_EXPORT inline T fmmMaximum(T const& left, T const& right) {
    return max(left, right);
}

CUDA_EXPORT inline float fmmMultiplyAdd(float first, float second, float third) {
    return std::fma(first, second, third);
}

CUDA_EXPORT inline double fmmMultiplyAdd(double first, double second, double third) {
    return std::fma(first, second, third);
}

template <class T>
CUDA_EXPORT inline T fmmMultiplyAdd(T const& first, T const& second, T const& third) {
    return fmaf(first, second, third);
}

}    // namespace detail

CUDA_EXPORT inline float fmmInverseOpeningAngle(float theta) {
#ifdef __CUDA_ARCH__
    return __fdiv_rn(1.0f, theta);
#else
    return 1.0f / theta;
#endif
}

// Either cell can contain the larger compact support. Retain the kick's fixed
// support as a floor for callers that override the value used to build the tree.
template <class T>
CUDA_EXPORT inline T fmmPairSupportRadius(
    T const& fixedSupportRadius, T const& sinkSupportRadius, T const& sourceSupportRadius) {
    return detail::fmmMaximum(
        fixedSupportRadius, detail::fmmMaximum(sinkSupportRadius, sourceSupportRadius));
}

// Inputs describe enclosing cells and periodic minimum distances. The support
// bound must cover the actual pair-softening law for every represented pair.
// All returned fields are numeric zero/one masks, including SIMD instantiations.
template <class T>
CUDA_EXPORT inline FmmAcceptance<T> computeFmmAcceptance(T const& centerDistanceSquared,
    T const& sinkRadius, T const& sourceRadius, T const& sinkBoxToSourceCenterSquared,
    T const& sourceBoxToSinkCenterSquared, T const& pairSupportRadius, T const& inverseTheta) {
    T const radiusSum = sinkRadius + sourceRadius;
    T const supportDistance = radiusSum + pairSupportRadius;
    T const centerSupportSeparated = T(centerDistanceSquared > supportDistance * supportDistance);
    T const sourceSupportDistance = sourceRadius + pairSupportRadius;
    T const sinkSupportDistance = sinkRadius + pairSupportRadius;
    T const sourceSupportSeparated =
        T(sinkBoxToSourceCenterSquared > sourceSupportDistance * sourceSupportDistance);
    T const sinkSupportSeparated =
        T(sourceBoxToSinkCenterSquared > sinkSupportDistance * sinkSupportDistance);
    T const supportSeparated = detail::fmmMaximum(
        centerSupportSeparated, detail::fmmMaximum(sourceSupportSeparated, sinkSupportSeparated));

    T const cellCellDistance = radiusSum * inverseTheta;
    T const particleCellDistance = detail::fmmMultiplyAdd(inverseTheta, sourceRadius, sinkRadius);
    T const cellParticleDistance = detail::fmmMultiplyAdd(inverseTheta, sinkRadius, sourceRadius);
    T const sourceOpeningRadius = sourceRadius * inverseTheta;
    T const sinkOpeningRadius = sinkRadius * inverseTheta;
    T const particleCellGeometry =
        detail::fmmMaximum(T(centerDistanceSquared > particleCellDistance * particleCellDistance),
            T(sinkBoxToSourceCenterSquared > sourceOpeningRadius * sourceOpeningRadius));
    T const cellParticleGeometry =
        detail::fmmMaximum(T(centerDistanceSquared > cellParticleDistance * cellParticleDistance),
            T(sourceBoxToSinkCenterSquared > sinkOpeningRadius * sinkOpeningRadius));

    return {T(centerDistanceSquared > cellCellDistance * cellCellDistance) * centerSupportSeparated,
        particleCellGeometry * supportSeparated, cellParticleGeometry * supportSeparated,
        supportSeparated};
}

// Select reciprocal PC/CP interactions using the smaller explicit-particle side.
// Equal counts expand the smaller-radius cell; equal radii fall back to PP.
// Integer counts preserve ties even beyond the exact-integer range of float.
template <class Count>
CUDA_EXPORT inline FmmInteraction selectFmmInteraction(bool cellCellAccepted,
    bool particleCellAccepted, bool cellParticleAccepted, bool bothLeaves, Count sinkParticleCount,
    Count sourceParticleCount, float sinkRadius, float sourceRadius) {
    if (cellCellAccepted) {
        return FmmInteraction::CellCell;
    }
    if (!bothLeaves) {
        return FmmInteraction::ParticleParticle;
    }
    bool const particleCell =
        particleCellAccepted && sinkParticleCount >= kMinimumHybridParticleCount;
    bool const cellParticle =
        cellParticleAccepted && sourceParticleCount >= kMinimumHybridParticleCount;
    if (particleCell && cellParticle) {
        if (sinkParticleCount < sourceParticleCount) {
            return FmmInteraction::ParticleCell;
        }
        if (sourceParticleCount < sinkParticleCount) {
            return FmmInteraction::CellParticle;
        }
        if (sourceRadius < sinkRadius) {
            return FmmInteraction::ParticleCell;
        }
        if (sinkRadius < sourceRadius) {
            return FmmInteraction::CellParticle;
        }
        return FmmInteraction::ParticleParticle;
    }
    if (particleCell) {
        return FmmInteraction::ParticleCell;
    }
    if (cellParticle) {
        return FmmInteraction::CellParticle;
    }
    return FmmInteraction::ParticleParticle;
}
