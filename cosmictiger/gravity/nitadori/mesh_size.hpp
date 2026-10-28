/* CosmicTiger - hierarchical PMMM mesh selection; BSL-1.0. */
#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pmmm::detail {

// The base mesh represents all owned particles. Select the smallest allowed
// side whose volume is at least baseSize^3 * activeCount / totalCount.
inline int hierarchicalMeshSize(
    int baseSize, int nearWidth, uint64_t activeCount, uint64_t totalCount) {
    if (nearWidth < 1 || nearWidth > 4 || baseSize < 2 * nearWidth + 1 ||
        activeCount > totalCount) {
        throw std::invalid_argument("Invalid hierarchical PMMM mesh or particle counts");
    }
    auto const cube = [](int side) {
        auto const value = static_cast<unsigned __int128>(side);
        return value * value * value;
    };
    auto const baseCells = cube(baseSize);
    if (baseCells > std::numeric_limits<uint64_t>::max()) {
        throw std::invalid_argument("Hierarchical PMMM mesh exceeds cell indexing limits");
    }
    int lower = 2 * nearWidth + 1;
    int upper = baseSize;
    if (activeCount == 0) {
        return lower;
    }
    // Exact integer comparisons avoid rounding an exact cube ratio upward or
    // losing a one-particle increment in large global populations.
    auto const target = baseCells * activeCount;
    while (lower < upper) {
        int const middle = lower + (upper - lower) / 2;
        if (cube(middle) * totalCount < target) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    return lower;
}

}    // namespace pmmm::detail
