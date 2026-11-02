/* CosmicTiger - active-suffix gas migration planning; BSL-1.0. */

#pragma once

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <vector>

namespace detail {

template <typename Index>
struct SphMigrationSuffix {
    Index begin;
    Index end;
    std::vector<Index> gasIndices;
};

// Query only the selected main-particle suffix. Negative category indices
// represent collisionless particles. Removing these gas slots in descending
// order permits swap-with-last compaction without invalidating a later removal.
template <typename Index, typename GasIndex>
SphMigrationSuffix<Index> planSphMigrationSuffix(
    Index begin, Index end, Index particleCount, Index gasCount, GasIndex&& gasIndex) {
    if (begin < 0 || begin > end || end != particleCount || gasCount < 0) {
        throw std::invalid_argument("Invalid hierarchical SPH migration range");
    }
    SphMigrationSuffix<Index> result{begin, end, {}};
    for (Index index = begin; index < end; ++index) {
        Index const slot = gasIndex(index);
        if (slot >= gasCount) {
            throw std::invalid_argument("Invalid gas slot in hierarchical SPH migration");
        }
        if (slot >= 0) {
            result.gasIndices.push_back(slot);
        }
    }
    std::sort(result.gasIndices.begin(), result.gasIndices.end(), std::greater<Index>());
    if (std::adjacent_find(result.gasIndices.begin(), result.gasIndices.end()) !=
        result.gasIndices.end()) {
        throw std::invalid_argument("Duplicate gas slot in hierarchical SPH migration");
    }
    return result;
}

}    // namespace detail
