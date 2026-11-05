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
    if (side > size_t(std::numeric_limits<int>::max()) ||
        spatialPower(side) > std::numeric_limits<size_t>::max() / multiplicity ||
        spatialPower(side) < cells) {
        throw std::overflow_error("Rounded particle count exceeds the supported range");
    }
    return {int(side), multiplicity * spatialPower(side)};
}

struct Layers {
    int lightSide;
    int heavySide;
    size_t count;
};

// Kelvin–Helmholtz uses two interleaved lattices in each half-box:
// N = 2ᴰ (n_lightᴰ + n_heavyᴰ), n_light = round((1/2)^(1/D) n_heavy).
inline Layers roundKelvinHelmholtz(size_t requested) {
    auto const layout = [](int heavySide) {
        size_t const lightSide =
            std::max(1L, std::lround(std::pow(0.5, 1.0 / kSpatialDimensions) * heavySide));
        size_t const heavy = heavySide;
        size_t const maximum = std::numeric_limits<size_t>::max();
        if (spatialPower(heavy) > maximum / (1 << kSpatialDimensions)) {
            return Layers{int(lightSide), heavySide, maximum};
        }
        size_t const lightCount = (1 << kSpatialDimensions) * spatialPower(lightSide);
        size_t const heavyCount = (1 << kSpatialDimensions) * spatialPower(heavy);
        if (heavyCount > std::numeric_limits<size_t>::max() - lightCount) {
            return Layers{int(lightSide), heavySide, maximum};
        }
        return Layers{int(lightSide), heavySide, lightCount + heavyCount};
    };
    // This upper bound is checked before multiplying the half-box populations.
    int const upperSide = roundLattice(requested, 2 << kSpatialDimensions).side;
    int lower = 1;
    int upper = std::max(1, upperSide * 2);
    while (lower < upper) {
        int const middle = lower + (upper - lower) / 2;
        if (layout(middle).count >= requested) {
            upper = middle;
        } else {
            lower = middle + 1;
        }
    }
    auto const result = layout(lower);
    if (result.count == std::numeric_limits<size_t>::max()) {
        throw std::overflow_error("Rounded layered particle count exceeds the supported range");
    }
    return result;
}

}    // namespace particle_count
