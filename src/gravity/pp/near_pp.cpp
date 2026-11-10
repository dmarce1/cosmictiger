/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <limits>
#include <stdexcept>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/pp_near_pp.hpp>
#include <cosmictiger/math/kernel.hpp>

/// Evaluate every supplied source against each sink using the common batched near-field
/// interface.
void pmmmNearField(std::vector<PmmmNearParticle> const& sinks,
    std::vector<PmmmNearParticle> const& sources, double cellWidth, bool doPotential,
    std::vector<std::array<double, 4>>& fields) {
    std::vector<PmmmNearSourceRange> const sourceRanges(sinks.size(), {0, sources.size()});
    pmmmNearFieldBatch(sinks, sources, sourceRanges, cellWidth, doPotential, fields);
}

/// Validate per-sink source ranges and evaluate softened direct near forces on the selected
/// backend.
void pmmmNearFieldBatch(std::vector<PmmmNearParticle> const& sinks,
    std::vector<PmmmNearParticle> const& sources,
    std::vector<PmmmNearSourceRange> const& sourceRanges, double cellWidth, bool doPotential,
    std::vector<std::array<double, 4>>& fields) {
    double const smoothingLength = getOptions().gravity.smoothingLength;
    if (!(cellWidth > 0 && cellWidth <= 1 && smoothingLength > 0 && smoothingLength <= 1)) {
        throw std::invalid_argument("PMMM near-field cell width/support must be in (0,1]");
    }
    if (sourceRanges.size() != sinks.size()) {
        throw std::invalid_argument("PMMM near-field batch requires one source range per sink");
    }
    size_t candidatePairs = 0;
    [[maybe_unused]] size_t usefulSinkCount = 0;
    for (auto const& range : sourceRanges) {
        if (range.begin > range.end || range.end > sources.size()) {
            throw std::invalid_argument("PMMM near-field source range is outside the source array");
        }
        size_t const count = range.end - range.begin;
        usefulSinkCount += count != 0;
        candidatePairs += std::min(count, std::numeric_limits<size_t>::max() - candidatePairs);
    }
    fields.assign(sinks.size(), {});
    if (candidatePairs == 0) {
        return;
    }
#ifdef USE_CUDA
    if (getOptions().gravity.cuda &&
        pmmm::detail::shouldUsePmmmNearCuda(usefulSinkCount, candidatePairs)) {
        pmmm::detail::pmmmNearFieldCuda(
            sinks, sources, sourceRanges, cellWidth, smoothingLength, doPotential, fields);
        return;
    }
#endif
    double const inverseSmoothingLength = 1 / smoothingLength;
    double const inverseSmoothingLengthCubed =
        inverseSmoothingLength * inverseSmoothingLength * inverseSmoothingLength;
    auto const& kernel = selectedGravityKernel();
    FlopCounter<> flops = 3;
    for (size_t sink = 0; sink < sinks.size(); ++sink) {
        auto const& range = sourceRanges[sink];
        for (size_t source = range.begin; source < range.end; ++source) {
            flops += pmmm::detail::accumulateNearPair(sinks[sink], sources[source], cellWidth,
                inverseSmoothingLength, inverseSmoothingLengthCubed, doPotential, kernel,
                fields[sink]);
        }
    }
    addCpuFlops(flops, FlopPrecision::Double);
}
