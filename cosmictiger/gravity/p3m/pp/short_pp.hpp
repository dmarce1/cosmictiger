/* CosmicTiger - P3M direct short-range pairs; BSL-1.0. */
#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include <cosmictiger/gravity/p3m/short_pair.hpp>

namespace cosmictiger::gravity::p3m::pp {

// Each sink uses its own half-open source range. Positions are box units with
// the periodic image already selected. Fields are overwritten as
// {potential, ax, ay, az} without G, and pairCurvature with each sink's direct-pair
// coefficient (short_pair.hpp). Identical primary particles are excluded.
// Returns true when the batch ran on the CUDA device.
bool shortFieldBatch(std::vector<ShortParticle> const& sinks,
    std::vector<ShortParticle> const& sources, std::vector<ShortSourceRange> const& sourceRanges,
    ShortPairParameters const& parameters, bool doPotential,
    std::vector<std::array<double, 4>>& fields, std::vector<double>& pairCurvature);

// Conservative launch-amortization thresholds, as for PMMM's near field.
inline constexpr size_t kShortCudaMinimumSinks = 128;
inline constexpr size_t kShortCudaMinimumPairs = 65536;

inline constexpr bool shouldUseShortCuda(size_t usefulSinkCount, size_t candidatePairs) {
    return usefulSinkCount >= kShortCudaMinimumSinks && candidatePairs >= kShortCudaMinimumPairs;
}

#ifdef USE_CUDA
void shortFieldCuda(std::vector<ShortParticle> const& sinks,
    std::vector<ShortParticle> const& sources, std::vector<ShortSourceRange> const& sourceRanges,
    ShortPairParameters const& parameters, bool doPotential,
    std::vector<std::array<double, 4>>& fields, std::vector<double>& pairCurvature);
// Call only after all batches have completed, before CUDA device teardown.
void clearShortCudaWorkspaces();
#endif

}    // namespace cosmictiger::gravity::p3m::pp
