/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#pragma once

#include <cmath>

#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/math/simd_math.hpp>

enum class FmmInteraction { CellCell, ParticleCell, CellParticle, ParticleParticle };

template <class T>
struct FmmAcceptance {
    using Mask = decltype(T{} < T{});

    Mask cellCell;
    Mask particleCell;
    Mask cellParticle;
    Mask supportSeparated;
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
    return simdMax(left, right);
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
// Returned fields are booleans for scalars and native masks for SIMD packs.
template <class T>
CUDA_EXPORT inline FmmAcceptance<T> computeFmmAcceptance(T const& centerDistanceSquared,
    T const& sinkRadius, T const& sourceRadius, T const& sinkBoxToSourceCenterSquared,
    T const& sourceBoxToSinkCenterSquared, T const& pairSupportRadius, T const& inverseTheta) {
    T const radiusSum = sinkRadius + sourceRadius;
    T const supportDistance = radiusSum + pairSupportRadius;
    auto const centerSupportSeparated = centerDistanceSquared > supportDistance * supportDistance;
    T const sourceSupportDistance = sourceRadius + pairSupportRadius;
    T const sinkSupportDistance = sinkRadius + pairSupportRadius;
    auto const sourceSupportSeparated =
        sinkBoxToSourceCenterSquared > sourceSupportDistance * sourceSupportDistance;
    auto const sinkSupportSeparated =
        sourceBoxToSinkCenterSquared > sinkSupportDistance * sinkSupportDistance;
    auto const supportSeparated =
        centerSupportSeparated || sourceSupportSeparated || sinkSupportSeparated;

    T const cellCellDistance = radiusSum * inverseTheta;
    T const particleCellDistance = simdFma(inverseTheta, sourceRadius, sinkRadius);
    T const cellParticleDistance = simdFma(inverseTheta, sinkRadius, sourceRadius);
    T const sourceOpeningRadius = sourceRadius * inverseTheta;
    T const sinkOpeningRadius = sinkRadius * inverseTheta;
    auto const particleCellGeometry =
        (centerDistanceSquared > particleCellDistance * particleCellDistance) ||
        (sinkBoxToSourceCenterSquared > sourceOpeningRadius * sourceOpeningRadius);
    auto const cellParticleGeometry =
        (centerDistanceSquared > cellParticleDistance * cellParticleDistance) ||
        (sourceBoxToSinkCenterSquared > sinkOpeningRadius * sinkOpeningRadius);

    return {(centerDistanceSquared > cellCellDistance * cellCellDistance) && centerSupportSeparated,
        particleCellGeometry && supportSeparated, cellParticleGeometry && supportSeparated,
        supportSeparated};
}

// Only the explicit-particle side must be a leaf. For a leaf pair, the walk
// takes the cheapest of the valid hybrids and direct summation: direct
// summation costs sinkCount * sourceCount * particlePairFlops, a hybrid costs
// hybridFlopsPerParticle for every particle of the leaf that is evaluated
// explicitly. Equal costs keep the smaller-radius cell as the cloud, and equal
// radii fall back to PP, so the decision depends only on the unordered pair
// and mirrors between the two sides. Zero costs leave the radius rule alone;
// a leaf against an internal cell is never costed, since its alternative is
// opening the cell, not direct summation.
CUDA_EXPORT inline FmmInteraction selectFmmInteraction(bool cellCellAccepted,
    bool particleCellAccepted, bool cellParticleAccepted, bool sinkLeaf, bool sourceLeaf,
    float sinkRadius, float sourceRadius, int sinkCount = 0, int sourceCount = 0,
    float hybridFlopsPerParticle = 0.0f, float particlePairFlops = 0.0f) {
    if (cellCellAccepted) {
        return FmmInteraction::CellCell;
    }
    bool particleCell = particleCellAccepted && sinkLeaf;
    bool cellParticle = cellParticleAccepted && sourceLeaf;
    if (sinkLeaf && sourceLeaf && hybridFlopsPerParticle > 0.0f && particlePairFlops > 0.0f) {
        float const pairFlops = float(sinkCount) * float(sourceCount) * particlePairFlops;
        particleCell = particleCell && float(sinkCount) * hybridFlopsPerParticle <= pairFlops;
        cellParticle = cellParticle && float(sourceCount) * hybridFlopsPerParticle <= pairFlops;
        if (particleCell && cellParticle && sinkCount != sourceCount) {
            return sinkCount < sourceCount ? FmmInteraction::ParticleCell :
                                             FmmInteraction::CellParticle;
        }
    }
    if (particleCell && cellParticle) {
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
