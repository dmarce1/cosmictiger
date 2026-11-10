/* CosmicTiger - hierarchical PMMM mesh selection; BSL-1.0. */
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pmmm::detail {

// Locality counts still use a binary domain tree; mesh sides are unrestricted.
inline constexpr bool isPowerOfTwo(uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

inline constexpr int nextPowerOfTwo(int value) {
    if (value <= 1) {
        return 1;
    }
    if (value > (1 << 30)) {
        throw std::overflow_error("No power-of-two PMMM mesh length fits in an int");
    }
    int result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

// Preferred coarse side. Smaller full meshes use each periodic neighbor once.
inline constexpr int minimumMeshSize(int nearWidth) {
    if (nearWidth < 1 || nearWidth > 4) {
        throw std::invalid_argument("PMMM near width must be between 1 and 4");
    }
    return nextPowerOfTwo(2 * nearWidth + 1);
}

// The coarsest mesh any hierarchical stage may use, and the granularity of
// particle-domain cuts. Zero requests the default: eight times coarser per
// axis than the full mesh, raised to the preferred near-field side and then
// to a divisor of the full mesh. Divisibility keeps domain cuts on stage planes.
inline int resolvedCoarseMeshSize(int meshSize, int coarseMeshSize, int nearWidth) {
    if (coarseMeshSize > 0) {
        return coarseMeshSize;
    }
    if (meshSize <= 0) {
        return 0;
    }
    int const minimum = std::min(meshSize, std::max(meshSize / 8, minimumMeshSize(nearWidth)));
    int result = meshSize;
    for (int factor = 1; factor <= meshSize / factor; ++factor) {
        if (meshSize % factor == 0) {
            for (int side : {factor, meshSize / factor}) {
                if (side >= minimum) {
                    result = std::min(result, side);
                }
            }
        }
    }
    return result;
}

// Particle domains bisect the box along x, y, z in turn, once per halving of
// the locality count. Each coarse cell must survive every cut along its axis,
// so the coarse side must be at least 2^ceil(log2(localities) / 3).
inline int minimumCoarseMeshSizeForLocalities(int localities) {
    if (localities < 1 || !isPowerOfTwo(uint64_t(localities))) {
        throw std::invalid_argument("PMMM requires a power-of-two locality count");
    }
    int levels = 0;
    while ((1 << levels) < localities) {
        ++levels;
    }
    return 1 << ((levels + 2) / 3);
}

// Select the smallest divisor of the full side that is a multiple of the
// coarse side and whose volume meets the selected particle fraction.
inline int hierarchicalMeshSize(
    int baseSize, int coarseSize, uint64_t activeCount, uint64_t totalCount) {
    if (baseSize < 1 || coarseSize < 1 || coarseSize > baseSize || baseSize % coarseSize != 0 ||
        activeCount > totalCount) {
        throw std::invalid_argument("Invalid hierarchical PMMM mesh or particle counts");
    }
    auto const cube = [](int side) {
        auto const value = static_cast<unsigned __int128>(side);
        return value * value * value;
    };
    if (cube(baseSize) > std::numeric_limits<uint64_t>::max()) {
        throw std::invalid_argument("Hierarchical PMMM mesh exceeds cell indexing limits");
    }
    if (activeCount == 0) {
        return coarseSize;
    }
    // Exact integer comparisons avoid rounding an exact cube ratio upward or
    // losing a one-particle increment in large global populations.
    auto const target = cube(baseSize) * activeCount;
    int result = baseSize;
    int const ratio = baseSize / coarseSize;
    for (int factor = 1; factor <= ratio / factor; ++factor) {
        if (ratio % factor == 0) {
            for (int multiplier : {factor, ratio / factor}) {
                int const side = coarseSize * multiplier;
                if (cube(side) * totalCount >= target) {
                    result = std::min(result, side);
                }
            }
        }
    }
    return result;
}

// Unique sink-minus-source offsets, with the same positive half-box tie as
// signedIndex and FmmGeometry. Tiny meshes must not count periodic cells twice.
inline int nearOffsetBegin(int meshSize, int nearWidth) {
    return -std::min(nearWidth, (meshSize - 1) / 2);
}

inline int nearOffsetEnd(int meshSize, int nearWidth) {
    return std::min(nearWidth, meshSize / 2);
}

}    // namespace pmmm::detail
