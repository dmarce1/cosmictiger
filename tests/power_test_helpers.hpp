/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef COSMICTIGER_TESTS_POWER_TEST_HELPERS_HPP_
#define COSMICTIGER_TESTS_POWER_TEST_HELPERS_HPP_

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/fft.hpp>
#include <cosmictiger/power_cpu.hpp>

namespace detail {

inline void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline void requireClose(double actual, double expected, double tolerance, char const* message) {
    uint64_t const exponent = std::bit_cast<uint64_t>(actual) & 0x7ff0000000000000ULL;
    if (exponent == 0x7ff0000000000000ULL ||
        std::abs(actual - expected) > tolerance * std::max(1.0, std::abs(expected))) {
        throw std::runtime_error(std::string(message) + ": got " + std::to_string(actual) +
            ", expected " + std::to_string(expected));
    }
}

struct PowerFixture {
    std::array<std::vector<Fixed32>, NDIM> positions;
    std::vector<float> masses;

    explicit PowerFixture(size_t count)
      : masses(count) {
        for (auto& position : positions) {
            position.resize(count);
        }
        uint32_t random = 1729;
        uint32_t const boundaryPositions[] = {0, 1, 127, 128, UINT32_MAX, UINT32_MAX - 128,
            0x20000000 - 128, 0x20000000, 0x20000000 + 128, 0x60000000 - 128, 0x60000000,
            0x60000000 + 128, 0x80000000, 0xe0000000};
        for (size_t index = 0; index < count; index++) {
            for (size_t dimension = 0; dimension < NDIM; dimension++) {
                random = 1664525 * random + 1013904223;
                uint32_t const raw = index < std::size(boundaryPositions) ?
                    boundaryPositions[(index + dimension * 3) % std::size(boundaryPositions)] :
                    random;
                positions[dimension][index].setInteger(raw);
            }
            masses[index] = index % 3 == 0 ? 0.375f : (index % 3 == 1 ? 1.625f : 0.0f);
        }
    }
};

// Independent centered cardinal B-spline of degree five, expressed as
// truncated powers instead of the production piecewise polynomial.
inline long double referenceWeight(long double distance) {
    if (std::abs(distance) >= 3) {
        return 0;
    }
    int const coefficients[] = {1, -6, 15, -20, 15, -6, 1};
    long double weight = 0;
    for (int knot = 0; knot <= 6; knot++) {
        long double const offset = std::max(0.0L, distance + 3 - knot);
        weight += coefficients[knot] * offset * offset * offset * offset * offset;
    }
    return weight / 120;
}

inline std::vector<double> referenceDensity(
    PowerFixture const& fixture, int fold, int gridSize, bool weighted) {
    std::vector<double> result(size_t(gridSize) * gridSize * gridSize, 0.0);
    for (size_t index = 0; index < fixture.masses.size(); index++) {
        std::array<std::vector<long double>, NDIM> weights;
        for (int dimension = 0; dimension < NDIM; dimension++) {
            weights[dimension].resize(gridSize, 0.0L);
            // Match the specified single-precision folding, including raw
            // Fixed32 values just below one that round to 1.0f before folding.
            float const coordinate = float(fixture.positions[dimension][index].raw()) * 0x1p-32f;
            float const unfolded = float(fold) * coordinate;
            long double const folded = unfolded - std::floor(static_cast<long double>(unfolded));
            long double const gridCoordinate = folded * gridSize;
            for (int cell = 0; cell < gridSize; cell++) {
                for (int image = -1; image <= 1; image++) {
                    weights[dimension][cell] +=
                        referenceWeight(gridCoordinate - cell + image * gridSize);
                }
            }
        }
        double const mass = weighted ? fixture.masses[index] : 1.0;
        for (int x = 0; x < gridSize; x++) {
            for (int y = 0; y < gridSize; y++) {
                for (int z = 0; z < gridSize; z++) {
                    size_t const cell = (size_t(x) * gridSize + y) * gridSize + z;
                    result[cell] += mass * weights[XDIM][x] * weights[YDIM][y] * weights[ZDIM][z];
                }
            }
        }
    }
    return result;
}

template <class AssignDensity>
std::vector<double> assignTiledDensity(int gridSize, int tileSize, AssignDensity const& assign) {
    std::vector<double> result(size_t(gridSize) * gridSize * gridSize, 0.0);
    for (int x = 0; x < gridSize; x += tileSize) {
        for (int y = 0; y < gridSize; y += tileSize) {
            for (int z = 0; z < gridSize; z += tileSize) {
                Range<int64_t> box;
                int const start[NDIM] = {x, y, z};
                for (int dimension = 0; dimension < NDIM; dimension++) {
                    box.begin[dimension] = start[dimension] + CLOUD_MIN;
                    box.end[dimension] =
                        std::min(start[dimension] + tileSize, gridSize) + CLOUD_MAX;
                }
                constexpr float kGuard = 123456.0f;
                std::vector<float> density(box.volume() + 2, 0.0f);
                density.front() = density.back() = kGuard;
                assign(box, density.data() + 1);
                require(density.front() == kGuard && density.back() == kGuard,
                    "density assignment preserves guards around the padded tile");
                // Emulate the periodic folding performed by fft3dAccumulateReal.
                for (int64_t i = box.begin[XDIM]; i < box.end[XDIM]; i++) {
                    for (int64_t j = box.begin[YDIM]; j < box.end[YDIM]; j++) {
                        for (int64_t k = box.begin[ZDIM]; k < box.end[ZDIM]; k++) {
                            int64_t const wrappedX = (i % gridSize + gridSize) % gridSize;
                            int64_t const wrappedY = (j % gridSize + gridSize) % gridSize;
                            int64_t const wrappedZ = (k % gridSize + gridSize) % gridSize;
                            size_t const cell =
                                (wrappedX * gridSize + wrappedY) * gridSize + wrappedZ;
                            result[cell] += density[box.index(i, j, k) + 1];
                        }
                    }
                }
            }
        }
    }
    return result;
}

inline void checkDensity(
    std::vector<double> const& actual, std::vector<double> const& reference, double expectedMass) {
    require(actual.size() == reference.size(), "density grids have matching extents");
    for (size_t cell = 0; cell < actual.size(); cell++) {
        requireClose(actual[cell], reference[cell], 3.0e-5,
            "density agrees with independent periodic quintic reference");
    }
    double const totalMass = std::accumulate(actual.begin(), actual.end(), 0.0);
    requireClose(totalMass, expectedMass, 2.0e-5,
        "tiled periodic density assignment conserves total particle mass");
}

inline double totalMass(PowerFixture const& fixture, bool weighted) {
    return weighted ? std::accumulate(fixture.masses.begin(), fixture.masses.end(), 0.0) :
                      double(fixture.masses.size());
}

}    // namespace detail

#endif
