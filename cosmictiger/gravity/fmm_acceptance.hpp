/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cmath>
#include <type_traits>

#include <cosmictiger/core/flops.hpp>
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

CUDA_EXPORT inline float fmmInverseOpeningAngle(float theta) {
#ifdef __CUDA_ARCH__
    return __fdiv_rn(1.0f, theta);
#else
    return 1.0f / theta;
#endif
}

// Inputs describe enclosing cells and periodic minimum distances. Every particle
// carries the same fixed comoving gravity support h_G (gravity.smoothingLength),
// which bounds the pair-softening law for every represented pair.
// Returned fields are booleans for scalars and native masks for SIMD packs.
template <class T>
CUDA_EXPORT inline FmmAcceptance<T> computeFmmAcceptance(T const& centerDistanceSquared,
    T const& sinkRadius, T const& sourceRadius, T const& sinkBoxToSourceCenterSquared,
    T const& sourceBoxToSinkCenterSquared, T const& smoothingLength, T const& inverseTheta,
    FlopCount* operationCount = nullptr) {
    T const radiusSum = sinkRadius + sourceRadius;
    T const supportDistance = radiusSum + smoothingLength;
    auto const centerSupportSeparated = centerDistanceSquared > supportDistance * supportDistance;
    T const sourceSupportDistance = sourceRadius + smoothingLength;
    T const sinkSupportDistance = sinkRadius + smoothingLength;
    auto const sourceSupportSeparated =
        sinkBoxToSourceCenterSquared > sourceSupportDistance * sourceSupportDistance;
    auto const sinkSupportSeparated =
        sourceBoxToSinkCenterSquared > sinkSupportDistance * sinkSupportDistance;
    auto const supportSeparated =
        centerSupportSeparated || sourceSupportSeparated || sinkSupportSeparated;

    T const cellCellDistance = radiusSum * inverseTheta;
    T const particleCellDistance = math::fma(inverseTheta, sourceRadius, sinkRadius);
    T const cellParticleDistance = math::fma(inverseTheta, sinkRadius, sourceRadius);
    T const sourceOpeningRadius = sourceRadius * inverseTheta;
    T const sinkOpeningRadius = sinkRadius * inverseTheta;
    auto const particleCellNear =
        centerDistanceSquared > particleCellDistance * particleCellDistance;
    auto const cellParticleNear =
        centerDistanceSquared > cellParticleDistance * cellParticleDistance;
    auto const particleCellGeometry = particleCellNear ||
        (sinkBoxToSourceCenterSquared > sourceOpeningRadius * sourceOpeningRadius);
    auto const cellParticleGeometry =
        cellParticleNear || (sourceBoxToSinkCenterSquared > sinkOpeningRadius * sinkOpeningRadius);
    if (operationCount != nullptr) {
        if constexpr (std::is_arithmetic_v<T>) {
            *operationCount = 17 + int(!particleCellNear) + int(!cellParticleNear);
        } else {
            *operationCount = 19 * T::size();
        }
    }

    FmmAcceptance<T> acceptance;
    acceptance.cellCell =
        (centerDistanceSquared > cellCellDistance * cellCellDistance) && centerSupportSeparated;
    acceptance.particleCell = particleCellGeometry && supportSeparated;
    acceptance.cellParticle = cellParticleGeometry && supportSeparated;
    acceptance.supportSeparated = supportSeparated;
    return acceptance;
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
// A caller may also provide an M2L cost to use direct summation for cheaper
// accepted leaf pairs. Zero keeps unconditional cell-cell precedence.
CUDA_EXPORT inline FmmInteraction selectFmmInteraction(bool cellCellAccepted,
    bool particleCellAccepted, bool cellParticleAccepted, bool sinkLeaf, bool sourceLeaf,
    float sinkRadius, float sourceRadius, int sinkCount = 0, int sourceCount = 0,
    float hybridFlopsPerParticle = 0.0f, float particlePairFlops = 0.0f,
    float cellCellFlops = 0.0f) {
    if (cellCellAccepted) {
        if (sinkLeaf && sourceLeaf && cellCellFlops > 0.0f && particlePairFlops > 0.0f &&
            float(sinkCount) * float(sourceCount) * particlePairFlops < cellCellFlops) {
            return FmmInteraction::ParticleParticle;
        }
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
