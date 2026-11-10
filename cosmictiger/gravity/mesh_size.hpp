/* CosmicTiger - hierarchical PMMM mesh selection; BSL-1.0. */
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pmmm::detail {

// Every PMMM mesh is a cube with a power-of-two side. Power-of-two sides make
// each cell boundary an exact fixed-point plane shared by every coarser mesh,
// so particle domains cut on the coarse mesh never split a cell of any finer one.
inline constexpr bool isPowerOfTwo(uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

/// Round a positive mesh length up to a power of two, returning one for small inputs and
/// rejecting int overflow.
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

// The smallest valid side for a near width: periodic near cells must be
// distinct, so the side is at least 2*w+1 and therefore the next power of two.
inline constexpr int minimumMeshSize(int nearWidth) {
    if (nearWidth < 1 || nearWidth > 4) {
        throw std::invalid_argument("PMMM near width must be between 1 and 4");
    }
    return nextPowerOfTwo(2 * nearWidth + 1);
}

// The coarsest mesh any hierarchical stage may use, and the granularity of
// particle-domain cuts. Zero requests the default: eight times coarser per
// axis than the full mesh, never below the minimum side for the near width,
// and never above the full mesh. A zero or invalid full mesh resolves to zero.
inline int resolvedCoarseMeshSize(int meshSize, int coarseMeshSize, int nearWidth) {
    if (coarseMeshSize > 0) {
        return coarseMeshSize;
    }
    if (meshSize <= 0 || !isPowerOfTwo(meshSize)) {
        return 0;
    }
    return std::min(meshSize, std::max(meshSize / 8, minimumMeshSize(nearWidth)));
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

// Select the smallest power-of-two side, between the coarse and full meshes,
// whose volume is at least the full mesh volume times active / total.
inline int hierarchicalMeshSize(
    int baseSize, int coarseSize, uint64_t activeCount, uint64_t totalCount) {
    if (baseSize < 1 || coarseSize < 1 || coarseSize > baseSize || !isPowerOfTwo(baseSize) ||
        !isPowerOfTwo(coarseSize) || activeCount > totalCount) {
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
    int side = coarseSize;
    while (side < baseSize && cube(side) * totalCount < target) {
        side <<= 1;
    }
    return side;
}

}    // namespace pmmm::detail
