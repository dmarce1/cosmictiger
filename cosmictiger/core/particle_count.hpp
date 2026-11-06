#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include <cosmictiger/core/defs.hpp>

namespace particle_count {

// Equal-mass Sod lattices retain the canonical density ratio of eight.
constexpr size_t kSodLatticeMultiplicity =
    kSpatialDimensions == 1 ? 18 : (kSpatialDimensions == 2 ? 36 : 72);

// Saturating integer lattice measure; avoids floating-point rounding at boundaries.
inline size_t spatialPower(size_t side) {
    size_t count = 1;
    for (int dim = 0; dim < kSpatialDimensions; ++dim) {
        if (side != 0 && count > std::numeric_limits<size_t>::max() / side) {
            return std::numeric_limits<size_t>::max();
        }
        count *= side;
    }
    return count;
}

struct Lattice {
    int side;
    size_t count;
};

// Derived geometry: N = m nᴰ. Find the smallest whole-cell lattice with
// N >= requested, optionally restricting n to multiples of a glass tile side.
// All final products are checked in integers, including at lattice boundaries.
inline Lattice roundLattice(size_t requested, size_t multiplicity = 1, int tileSide = 1) {
    if (requested == 0) {
        return {0, 0};
    }
    if (multiplicity == 0 || tileSide <= 0) {
        throw std::invalid_argument("Invalid particle lattice geometry");
    }
    size_t const cells = requested / multiplicity + (requested % multiplicity != 0);
    size_t lower = 1;
    size_t upper = std::numeric_limits<int>::max();
    while (lower < upper) {
        size_t const middle = lower + (upper - lower) / 2;
        if (spatialPower(middle) >= cells) {
            upper = middle;
        } else {
            lower = middle + 1;
        }
    }
    size_t const side = (lower / tileSide + (lower % tileSide != 0)) * tileSide;
    size_t const cellCount = spatialPower(side);
    // spatialPower saturates on overflow; the sentinel is never a valid grid.
    if (side > size_t(std::numeric_limits<int>::max()) ||
        cellCount == std::numeric_limits<size_t>::max() ||
        cellCount > std::numeric_limits<size_t>::max() / multiplicity || cellCount < cells) {
        throw std::overflow_error("Rounded particle count exceeds the supported range");
    }
    return {int(side), multiplicity * cellCount};
}

struct KelvinHelmholtzLayout {
    int side;
    size_t count;
    size_t lowCount;
    size_t highCount;
};

// Derived common-cell geometry, with n cells across each half-box:
// N_low = s_low 2ᴰ⁻¹ nᴰ, N_high = 2 N_low, N = 3 s_low 2ᴰ⁻¹ nᴰ.
// s_low -> lowSitesPerCell; n -> side. Round upward using whole shared cells;
// neither fluid changes cell size to approximate the density ratio.
inline KelvinHelmholtzLayout roundKelvinHelmholtz(size_t requested, size_t lowSitesPerCell) {
    constexpr size_t kHalfBoxFactor = size_t(1) << (kSpatialDimensions - 1);
    if (lowSitesPerCell == 0 || lowSitesPerCell > (size_t(1) << kSpatialDimensions)) {
        throw std::invalid_argument("Invalid Kelvin-Helmholtz cell population");
    }
    auto const lattice = roundLattice(requested, 3 * lowSitesPerCell * kHalfBoxFactor);
    if (lattice.side > std::numeric_limits<int>::max() / 2) {
        throw std::overflow_error("Kelvin-Helmholtz cell count exceeds the supported range");
    }
    return {lattice.side, lattice.count, lattice.count / 3, 2 * (lattice.count / 3)};
}

}    // namespace particle_count
