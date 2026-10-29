/* CosmicTiger - sampled-gravity correction table checks; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/gravity/direct_reference.hpp>
#include <cosmictiger/gravity/reference_table.hpp>

namespace detail {

using ReferencePosition = std::array<double, 3>;

constexpr double kCorrectionTolerance = 1.0e-11;

void checkTableCorrection(ReferencePosition const& position) {
    auto const& table = gravityReferenceCorrectionTable();
    auto const actual = gravityReferenceTableCorrection(position, table.data());
    auto const expected = gravityReferenceCorrection(position, gravityReferenceTables());
    for (int dim = 0; dim < 3; ++dim) {
        EXPECT_NEAR(actual[dim], expected[dim], kCorrectionTolerance)
            << "position " << position[0] << ' ' << position[1] << ' ' << position[2]
            << ", component " << dim;
    }
}

auto const tableTestKernel = radial_kernel::makeKernelTable({radial_kernel::Family::Ferrers, 2});

}    // namespace detail

TEST(GravityReferenceTable, CompactStorageAndHaloNodes) {
    auto const& table = detail::gravityReferenceCorrectionTable();
    ASSERT_EQ(table.size(), detail::kGravityReferenceTableSize);
    ASSERT_EQ(table.size(), 54740);
    size_t next = 0;
    for (int i = 0; i <= 67; ++i) {
        for (int j = 0; j <= i; ++j) {
            for (int k = 0; k <= j; ++k) {
                size_t const index = detail::gravityReferenceTableIndex(i, j, k);
                ASSERT_EQ(index, next++);
                for (double const component : table[index]) {
                    ASSERT_TRUE(std::isfinite(component));
                }
                if ((i % 11 == 0 && j % 7 == 0 && k % 5 == 0) ||
                    (i >= 64 && (j == i || j == 0 || j == 63) && (k == j || k == 0))) {
                    detail::ReferencePosition const position{i / 128.0, j / 128.0, k / 128.0};
                    auto const expected = detail::gravityReferenceCorrection(
                        position, detail::gravityReferenceTables());
                    for (int dim = 0; dim < 3; ++dim) {
                        EXPECT_NEAR(table[index][dim], expected[dim], 3e-14);
                    }
                }
            }
        }
    }
    EXPECT_EQ(next, table.size());
}

TEST(GravityReferenceTable, InteriorGridAndDeterministicRandomPoints) {
    for (double const x : {-.5, -.31, -.13, 0.0, .17, .37, .5}) {
        for (double const y : {-.5, -.31, -.13, 0.0, .17, .37, .5}) {
            for (double const z : {-.5, -.31, -.13, 0.0, .17, .37, .5}) {
                detail::checkTableCorrection({x, y, z});
            }
        }
    }
    std::mt19937_64 generator(0x4c36a9d17ULL);
    std::uniform_real_distribution<double> coordinate(-.5, .5);
    for (int sample = 0; sample < 1024; ++sample) {
        detail::checkTableCorrection(
            {coordinate(generator), coordinate(generator), coordinate(generator)});
    }
}

TEST(GravityReferenceTable, StencilTransitionsAxesAndSortingTies) {
    for (int knot = 0; knot <= 64; ++knot) {
        double const center = knot / 128.0;
        for (double const offset : {-1e-12, 0.0, 1e-12}) {
            double const x = std::clamp(center + offset, 0.0, .5);
            detail::checkTableCorrection({x, .237, -.419});
            detail::checkTableCorrection({x, x, .173});
            detail::checkTableCorrection({x, -x, x});
            detail::checkTableCorrection({x, 0, 0});
        }
    }
    for (double const distance : {0.0, 1e-15, 1e-12, 1e-9, 1e-6}) {
        detail::checkTableCorrection({distance, -distance, distance});
        auto const& table = detail::gravityReferenceCorrectionTable();
        auto const correction =
            detail::gravityReferenceTableCorrection({distance, -distance, distance}, table.data());
        for (int dim = 0; dim < 3; ++dim) {
            // Absolute accuracy is meaningful at a vanishing force; a uniform
            // relative-error requirement is not defined at these symmetries.
            double const expected = (dim == 1 ? -1 : 1) * 4 * std::numbers::pi * distance / 3;
            EXPECT_NEAR(correction[dim], expected, detail::kCorrectionTolerance);
        }
    }
}

TEST(GravityReferenceTable, CubicSymmetriesAndExactNodes) {
    auto const& table = detail::gravityReferenceCorrectionTable();
    for (detail::ReferencePosition const position : {detail::ReferencePosition{.13, .29, .47},
             {.21, .21, .37}, {.5, .19, .031}, {0, .37, .43}}) {
        auto const base = detail::gravityReferenceTableCorrection(position, table.data());
        std::array<int, 3> permutation{0, 1, 2};
        do {
            for (int signs = 0; signs < 8; ++signs) {
                detail::ReferencePosition transformed;
                for (int dim = 0; dim < 3; ++dim) {
                    transformed[dim] = ((signs >> dim) & 1 ? -1 : 1) * position[permutation[dim]];
                }
                auto const actual =
                    detail::gravityReferenceTableCorrection(transformed, table.data());
                for (int dim = 0; dim < 3; ++dim) {
                    double const expected = ((signs >> dim) & 1 ? -1 : 1) * base[permutation[dim]];
                    EXPECT_NEAR(actual[dim], expected, 3e-15);
                }
            }
        } while (std::next_permutation(permutation.begin(), permutation.end()));
    }
    for (int i = 0; i <= 64; i += 4) {
        for (int j = 0; j <= i; j += 4) {
            for (int k = 0; k <= j; k += 4) {
                detail::ReferencePosition const position{i / 128.0, j / 128.0, k / 128.0};
                auto const actual = detail::gravityReferenceTableCorrection(position, table.data());
                auto const& stored = table[detail::gravityReferenceTableIndex(i, j, k)];
                for (int dim = 0; dim < 3; ++dim) {
                    EXPECT_NEAR(actual[dim], stored[dim], 3e-14);
                }
            }
        }
    }
}

TEST(GravityReferenceTable, SmallRadiusLimitAndThreshold) {
    auto const& table = detail::gravityReferenceCorrectionTable();
    constexpr double kOriginSlope = 4 * std::numbers::pi / 3;
    for (double const scale : {1e-20, 1e-18, 1e-15, 1e-12, 1e-9}) {
        for (detail::ReferencePosition const direction :
            {detail::ReferencePosition{1, 0, 0}, {1, -2, 3}, {-3, 1, -2}}) {
            detail::ReferencePosition position;
            detail::ReferencePosition opposite;
            for (int dim = 0; dim < 3; ++dim) {
                position[dim] = scale * direction[dim];
                opposite[dim] = -position[dim];
            }
            auto const actual = detail::gravityReferenceTableCorrection(position, table.data());
            auto const reversed = detail::gravityReferenceTableCorrection(opposite, table.data());
            for (int dim = 0; dim < 3; ++dim) {
                double const expected = kOriginSlope * position[dim];
                // An absolute tolerance alone would miss loss of a fixed64
                // displacement smaller than a unit-grid interpolation ulp.
                EXPECT_NEAR(actual[dim], expected,
                    4 * std::numeric_limits<double>::epsilon() * std::abs(expected));
                EXPECT_DOUBLE_EQ(reversed[dim], -actual[dim]);
            }
        }
    }
    for (double const factor : {1 - 1e-9, 1.0, 1 + 1e-9}) {
        for (detail::ReferencePosition const direction :
            {detail::ReferencePosition{1, 0, 0}, {2.0 / 3, -2.0 / 3, 1.0 / 3}}) {
            detail::ReferencePosition position;
            detail::ReferencePosition opposite;
            for (int dim = 0; dim < 3; ++dim) {
                position[dim] = 1e-6 * factor * direction[dim];
                opposite[dim] = -position[dim];
            }
            auto const actual = detail::gravityReferenceTableCorrection(position, table.data());
            auto const expected =
                detail::gravityReferenceCorrection(position, detail::gravityReferenceTables());
            auto const reversed = detail::gravityReferenceTableCorrection(opposite, table.data());
            for (int dim = 0; dim < 3; ++dim) {
                EXPECT_NEAR(actual[dim], expected[dim], 1e-14);
                EXPECT_DOUBLE_EQ(reversed[dim], -actual[dim]);
            }
        }
    }
}

TEST(GravityReferenceTable, SharedSofteningAndPeriodicFaces) {
    auto const& table = detail::gravityReferenceCorrectionTable();
    for (double const support : {.01, .2, .8, 1.5}) {
        for (detail::ReferencePosition const position : {detail::ReferencePosition{0, 0, 0},
                 {1e-12, -2e-12, 3e-12}, {.13, -.17, .23}, {-.5, .19, -.31}, {.5, .5, -.5}}) {
            auto const actual = detail::tabulatedGravityReference(
                position, detail::tableTestKernel, support, table.data());
            auto const expected =
                detail::directGravityReference(position, detail::tableTestKernel, support);
            for (int dim = 0; dim < 3; ++dim) {
                EXPECT_NEAR(actual[dim], expected[dim], detail::kCorrectionTolerance);
            }
        }
    }
    for (double const x : {-.5, .5}) {
        auto const unsoftened = detail::tabulatedGravityReference(
            {x, .19, -.31}, detail::tableTestKernel, .01, table.data());
        EXPECT_DOUBLE_EQ(unsoftened[0], 0);
        auto const softened = detail::tabulatedGravityReference(
            {x, .19, -.31}, detail::tableTestKernel, .8, table.data());
        EXPECT_GT(std::abs(softened[0]), .01);
    }
}

TEST(GravityReferenceTable, MassWeightedCancellationUsesAbsoluteError) {
    auto const& table = detail::gravityReferenceCorrectionTable();
    std::array<detail::GravityReferenceSum, 3> actual;
    std::array<detail::GravityReferenceSum, 3> expected;
    double totalMass = 0;
    for (int pair = 0; pair < 12; ++pair) {
        detail::ReferencePosition const position{
            .013 + .037 * pair, -.193 + .023 * pair, .431 - .029 * pair};
        for (int sign : {-1, 1}) {
            double const mass = (.07 + .013 * pair) * (pair == 0 && sign == 1 ? 1 + 1e-10 : 1);
            detail::ReferencePosition const separation{
                sign * position[0], sign * position[1], sign * position[2]};
            auto const interpolated = detail::tabulatedGravityReference(
                separation, detail::tableTestKernel, .2, table.data());
            auto const exact =
                detail::directGravityReference(separation, detail::tableTestKernel, .2);
            for (int dim = 0; dim < 3; ++dim) {
                actual[dim].add(mass * interpolated[dim]);
                expected[dim].add(mass * exact[dim]);
            }
            totalMass += mass;
        }
    }
    for (int dim = 0; dim < 3; ++dim) {
        EXPECT_NEAR(
            actual[dim].value(), expected[dim].value(), totalMass * detail::kCorrectionTolerance);
        EXPECT_LT(std::abs(expected[dim].value()), 1e-8);
    }
}
