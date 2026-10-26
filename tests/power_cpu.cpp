/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 SPDX-License-Identifier: BSL-1.0
 */

#include <cstdio>

#include <gtest/gtest.h>

#include "power_test_helpers.hpp"

void cosmictigerAssert(char const* condition, bool value, char const* file, int line) {
    if (!value) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + condition);
    }
}

namespace detail {

void testEmpty() {
    Range<int64_t> const box(0);
    accumulateDensityCpuKernel(nullptr, nullptr, nullptr, nullptr, 0, 1, 8, box, nullptr);
    PowerFixture const fixture(0);
    auto const density = assignTiledDensity(8, 3, [](Range<int64_t> const& tile, float* output) {
        accumulateDensityCpuKernel(nullptr, nullptr, nullptr, nullptr, 0, 1, 8, tile, output);
    });
    checkDensity(density, referenceDensity(fixture, 1, 8, false), 0);
}

void testFoldedDensity() {
    constexpr int kGridSize = 8;
    PowerFixture const fixture(79);
    for (bool weighted : {false, true}) {
        for (int fold : {1, 2, 3, 8}) {
            auto const reference = referenceDensity(fixture, fold, kGridSize, weighted);
            for (int tileSize : {1, 3, kGridSize}) {
                auto const actual = assignTiledDensity(
                    kGridSize, tileSize, [&](Range<int64_t> const& box, float* density) {
                        accumulateDensityCpuKernel(fixture.positions[kXDimension].data(),
                            fixture.positions[kYDimension].data(),
                            fixture.positions[kZDimension].data(),
                            weighted ? fixture.masses.data() : nullptr, fixture.masses.size(), fold,
                            kGridSize, box, density);
                    });
                checkDensity(actual, reference, totalMass(fixture, weighted));
            }
        }
    }
}

void testUniformLattice() {
    constexpr int kGridSize = 8;
    PowerFixture fixture(kGridSize * kGridSize * kGridSize);
    for (int x = 0; x < kGridSize; x++) {
        for (int y = 0; y < kGridSize; y++) {
            for (int z = 0; z < kGridSize; z++) {
                size_t const index = (x * kGridSize + y) * kGridSize + z;
                fixture.positions[kXDimension][index] = double(x) / kGridSize;
                fixture.positions[kYDimension][index] = double(y) / kGridSize;
                fixture.positions[kZDimension][index] = double(z) / kGridSize;
            }
        }
    }
    auto const density =
        assignTiledDensity(kGridSize, 3, [&](Range<int64_t> const& box, float* output) {
            accumulateDensityCpuKernel(fixture.positions[kXDimension].data(),
                fixture.positions[kYDimension].data(), fixture.positions[kZDimension].data(),
                nullptr, fixture.masses.size(), 1, kGridSize, box, output);
        });
    for (double value : density) {
        requireClose(value, 1.0, 2.0e-5, "unit lattice produces uniform density");
    }
}

}    // namespace detail

TEST(PowerCpu, Empty) {
    detail::testEmpty();
}

TEST(PowerCpu, FoldedDensity) {
    detail::testFoldedDensity();
}

TEST(PowerCpu, UniformLattice) {
    detail::testUniformLattice();
}
