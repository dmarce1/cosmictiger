/* CosmicTiger - PMMM mesh geometry; BSL-1.0. */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pmmm::detail {

// Locality counts still use a binary domain tree; mesh sides are unrestricted.
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

// Derived from the cell-cell criterion in computeFmmAcceptance
// (cosmictiger/gravity/fmm_acceptance.hpp), using filled cubic cells:
// rₛ = rₜ = √3 ℓ/2, d > (rₛ+rₜ)/θ; near iff i²+j²+k² ≤ 3/θ².
// Symbol directory: rₛ, rₜ → enclosing radii; ℓ → cell side;
// d → center separation; θ → opening angle; i, j, k → integer cell offsets.
// Code names: d2 → d²/ℓ², squared separation in cell units.
// Cell-side factors cancel. Equality remains near, as in the strict FMM test.
inline bool nearCellOffset(int i, int j, int k, double θ) {
    double const d2 = double(i) * i + double(j) * j + double(k) * k;
    return d2 * (θ * θ) <= 3.0;
}

// Bound the enumeration by the periodic mesh before converting to int, so
// arbitrarily small positive angles never overflow or duplicate source cells.
inline int nearStencilWidth(int meshSize, double openingAngle) {
    if (!(openingAngle > 0 && openingAngle <= 1)) {
        throw std::invalid_argument("PMMM opening angle must be in (0, 1]");
    }
    int width = int(std::min(double(meshSize / 2), std::ceil(std::sqrt(3.0) / openingAngle)));
    while (width > 0 && !nearCellOffset(width, 0, 0, openingAngle)) {
        --width;
    }
    return width;
}

// Minimum possible particle separation in any primary cell pair left to the
// unsoftened mesh field. Search only the first octant and the first excluded z
// at each (x,y); more distant offsets cannot reduce the box-to-box distance.
// Infinity means every periodic source cell is near. This also handles tiny
// meshes and half-box ties without an artificial lower bound on theta.
inline double nearStencilSupport(int meshSize, double openingAngle) {
    int const half = meshSize / 2;
    if (nearCellOffset(half, half, half, openingAngle)) {
        return std::numeric_limits<double>::infinity();
    }
    int const end = std::min(half, nearStencilWidth(meshSize, openingAngle) + 1);
    double minimumSquared = std::numeric_limits<double>::infinity();
    for (int x = 0; x <= end; ++x) {
        for (int y = 0; y <= end; ++y) {
            if (nearCellOffset(x, y, end, openingAngle)) {
                continue;
            }
            int lower = 0;
            int upper = end;
            while (lower < upper) {
                int const middle = lower + (upper - lower) / 2;
                if (nearCellOffset(x, y, middle, openingAngle)) {
                    lower = middle + 1;
                } else {
                    upper = middle;
                }
            }
            double const dx = std::max(x - 1, 0);
            double const dy = std::max(y - 1, 0);
            double const dz = std::max(lower - 1, 0);
            minimumSquared = std::min(minimumSquared, dx * dx + dy * dy + dz * dz);
        }
    }
    return std::sqrt(minimumSquared) / meshSize;
}

// Particle domains bisect the box along x, y, z in turn, once per halving of
// the locality count. Each mesh cell must survive every cut along its axis,
// so the mesh side must be at least 2^ceil(log2(localities) / 3).
inline int minimumMeshSizeForLocalities(int localities) {
    if (localities < 1 || !isPowerOfTwo(uint64_t(localities))) {
        throw std::invalid_argument("PMMM requires a power-of-two locality count");
    }
    int levels = 0;
    while ((1 << levels) < localities) {
        ++levels;
    }
    return 1 << ((levels + 2) / 3);
}

// Unique sink-minus-source offsets, with the same positive half-box tie as
// signedIndex and FmmGeometry. Tiny meshes must not count periodic cells twice.
inline int nearOffsetBegin(int meshSize, int nearWidth) {
    return -std::min(nearWidth, (meshSize - 1) / 2);
}

/// Return the inclusive upper periodic-neighbor offset, keeping the positive half-box tie.
inline int nearOffsetEnd(int meshSize, int nearWidth) {
    return std::min(nearWidth, meshSize / 2);
}

}    // namespace pmmm::detail
