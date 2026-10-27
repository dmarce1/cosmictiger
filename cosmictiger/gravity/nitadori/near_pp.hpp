/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/math/simd_math.hpp>

struct PmmmNearParticle {
    std::array<double, 3> offset{};
    double mass = 0;
    int rank = 0;
    int64_t index = 0;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & offset & mass & rank & index;
    }
};

struct PmmmNearSourceRange {
    size_t begin = 0;
    size_t end = 0;
};

// Coordinates are in cell units about the same sink-cell center, with the
// source-cell periodic image already selected. No nearest-image wrapping is
// performed here. Fields are overwritten as {potential, ax, ay, az}, in unit-box
// coordinates without G. Only identical primary particle IDs are excluded.
void pmmmNearField(std::vector<PmmmNearParticle> const& sinks,
    std::vector<PmmmNearParticle> const& sources, double cellWidth, bool doPotential,
    std::vector<std::array<double, 4>>& fields);

// Each sink uses its own half-open source range. Batching retains each sink's
// existing source order and selected periodic image.
void pmmmNearFieldBatch(std::vector<PmmmNearParticle> const& sinks,
    std::vector<PmmmNearParticle> const& sources,
    std::vector<PmmmNearSourceRange> const& sourceRanges, double cellWidth, bool doPotential,
    std::vector<std::array<double, 4>>& fields);

namespace pmmm::detail {

// Conservative launch-amortization thresholds, not measured device optima.
// Apply these to the whole batch, including all cells and excluding empty sinks.
inline constexpr size_t kPmmmNearCudaMinimumSinks = 128;
inline constexpr size_t kPmmmNearCudaMinimumPairs = 65536;

inline constexpr bool shouldUsePmmmNearCuda(size_t usefulSinkCount, size_t candidatePairs) {
    return usefulSinkCount >= kPmmmNearCudaMinimumSinks &&
        candidatePairs >= kPmmmNearCudaMinimumPairs;
}

CUDA_EXPORT inline FlopCount nearBernsteinFlops(int degree) {
    return FlopCount(3) * degree * (degree + 1) / 2;
}

// Scalar radial evaluation executes only the selected span. Each softened
// force/potential evaluation also subtracts the scalar double span width.
template <class Kernel, class Real>
CUDA_EXPORT inline FlopCount nearPairFlops(Kernel const& kernel, Real const& q, bool doPotential) {
    FlopCount forceFlops = 3;
    FlopCount potentialFlops = 1;
    for (int piece = 0; piece < kernel.pieceCount; ++piece) {
        if (q < Real(kernel.changePoints[piece + 1])) {
            if (piece == 0) {
                forceFlops = 4 + nearBernsteinFlops(kernel.degree);
                potentialFlops = 7 + nearBernsteinFlops(kernel.degree);
            } else {
                forceFlops = 8 + nearBernsteinFlops(kernel.degree + 3);
                potentialFlops = 8 + nearBernsteinFlops(kernel.degree + 3) +
                    nearBernsteinFlops(kernel.degree + 2);
            }
            break;
        }
    }
    // Position/radius: 13; mass/force scaling: 2; acceleration updates: 6.
    return 21 + forceFlops + (doPotential ? 3 + potentialFlops : 0);
}

// ADL resolves forceFactor/potential to the selected radial_kernel::KernelTable.
// Real also permits independent arithmetic-counting tests of this exact path.
template <bool countFlops = kCountFlops, class Kernel, class Real>
CUDA_EXPORT inline FlopCount accumulateNearPair(PmmmNearParticle const& sink,
    PmmmNearParticle const& source, Real cellWidth, Real inverseSupportRadius,
    Real inverseSupportRadiusCubed, bool doPotential, Kernel const& kernel,
    std::array<Real, 4>& field) {
    if (source.rank == sink.rank && source.index == sink.index) {
        return 0;
    }
    Real const dx = (Real(sink.offset[0]) - Real(source.offset[0])) * cellWidth;
    Real const dy = (Real(sink.offset[1]) - Real(source.offset[1])) * cellWidth;
    Real const dz = (Real(sink.offset[2]) - Real(source.offset[2])) * cellWidth;
    Real const q = simdSqrt(dx * dx + dy * dy + dz * dz) * inverseSupportRadius;
    Real const factor = Real(source.mass) * inverseSupportRadiusCubed * forceFactor(kernel, q);
    field[1] -= factor * dx;
    field[2] -= factor * dy;
    field[3] -= factor * dz;
    if (doPotential) {
        field[0] -= Real(source.mass) * inverseSupportRadius * potential(kernel, q);
    }
    if constexpr (countFlops) {
        return nearPairFlops(kernel, q, doPotential);
    }
    return 0;
}

#ifdef USE_CUDA
void pmmmNearFieldCuda(std::vector<PmmmNearParticle> const& sinks,
    std::vector<PmmmNearParticle> const& sources,
    std::vector<PmmmNearSourceRange> const& sourceRanges, double cellWidth, double supportRadius,
    bool doPotential, std::vector<std::array<double, 4>>& fields);

// Call only after all PMMM batches have completed, before CUDA device teardown.
void clearPmmmNearCudaWorkspaces();
#endif

}    // namespace pmmm::detail
