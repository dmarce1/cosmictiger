/* CosmicTiger - independent sampled-gravity reference checks; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <set>

#include <gtest/gtest.h>

#include <cosmictiger/gravity/direct_reference.hpp>

namespace detail {

// A separate Ewald splitting and larger cubes check the converged periodic
// field, without copying the production reference's central correction split.
std::array<double, 3> alternateEwaldReference(
    std::array<double, 3> const& separation, long double alpha = 2.7L, int cutoff = 5) {
    constexpr long double kPi = std::numbers::pi_v<long double>;
    long double const normalization = 2 * alpha / std::sqrt(kPi);
    std::array<long double, 3> result{};
    for (int x = -cutoff; x <= cutoff; ++x) {
        for (int y = -cutoff; y <= cutoff; ++y) {
            for (int z = -cutoff; z <= cutoff; ++z) {
                std::array<long double, 3> const image{static_cast<long double>(separation[0]) + x,
                    static_cast<long double>(separation[1]) + y,
                    static_cast<long double>(separation[2]) + z};
                long double const squared =
                    image[0] * image[0] + image[1] * image[1] + image[2] * image[2];
                long double const radius = std::sqrt(squared);
                long double const radial = (std::erfc(alpha * radius) / radius +
                                               normalization * std::exp(-alpha * alpha * squared)) /
                    squared;
                for (int dim = 0; dim < 3; ++dim) {
                    result[dim] -= radial * image[dim];
                }
                int const waveSquared = x * x + y * y + z * z;
                if (waveSquared != 0) {
                    std::array<long double, 3> const wave{2 * kPi * x, 2 * kPi * y, 2 * kPi * z};
                    long double const amplitude =
                        std::exp(-kPi * kPi * waveSquared / (alpha * alpha)) / (kPi * waveSquared) *
                        std::sin(wave[0] * separation[0] + wave[1] * separation[1] +
                            wave[2] * separation[2]);
                    for (int dim = 0; dim < 3; ++dim) {
                        result[dim] -= amplitude * wave[dim];
                    }
                }
            }
        }
    }
    return {double(result[0]), double(result[1]), double(result[2])};
}

auto const referenceKernel = radial_kernel::makeKernelTable({radial_kernel::Family::Ferrers, 2});

}    // namespace detail

TEST(GravityDirectReference, AnalyticIsolatedAndSoftenedPairs) {
    auto const far =
        detail::directGravityReference({0.3, 0.4, 0}, detail::referenceKernel, .1, false);
    EXPECT_NEAR(far[0], -2.4, 1e-14);
    EXPECT_NEAR(far[1], -3.2, 1e-14);
    EXPECT_DOUBLE_EQ(far[2], 0);
    // Ferrers degree 2 encloses q^3(5-3q^2)/2 of its source mass.
    auto const close =
        detail::directGravityReference({0.03, 0.04, 0}, detail::referenceKernel, .1, false);
    EXPECT_NEAR(close[0], -63.75, 1e-12);
    EXPECT_NEAR(close[1], -85.0, 1e-12);
    EXPECT_DOUBLE_EQ(close[2], 0);
}

TEST(GravityDirectReference, IndependentPeriodicSplitting) {
    for (std::array<double, 3> const separation :
        {std::array<double, 3>{.11, -.23, .34}, {.499, -.497, .31}, {-.5, .125, -.25}}) {
        auto const actual =
            detail::directGravityReference(separation, detail::referenceKernel, .01);
        auto const expected = detail::alternateEwaldReference(separation);
        for (int dim = 0; dim < 3; ++dim) {
            EXPECT_NEAR(actual[dim], expected[dim], 3e-12);
        }
    }
}

TEST(GravityDirectReference, DistributedDisplacementsAndCutoffBoundaries) {
    auto const compare = [](std::array<double, 3> const& separation) {
        auto const actual =
            detail::directGravityReference(separation, detail::referenceKernel, .001);
        auto const expected = detail::alternateEwaldReference(separation);
        auto const fullCube = detail::alternateEwaldReference(separation, 2.0L, 6);
        for (int dim = 0; dim < 3; ++dim) {
            double const tolerance = 64 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, std::abs(expected[dim]));
            EXPECT_NEAR(actual[dim], expected[dim], tolerance);
            EXPECT_NEAR(actual[dim], fullCube[dim], tolerance);
        }
    };
    for (double const x : {-.49, -.17, .31}) {
        for (double const y : {-.49, -.17, .31}) {
            for (double const z : {-.49, -.17, .31}) {
                compare({x, y, z});
            }
        }
    }
    // The image shifted by +4 crosses the actual real radius at x=-0.4.
    // Also cover wrapped faces, edges, corners and exact symmetry zeros.
    for (std::array<double, 3> const separation : {std::array<double, 3>{-.400000000001, 0, 0},
             {-.4, 0, 0}, {-.399999999999, 0, 0}, {-.5, 0, 0}, {.5, .5, 0}, {-.5, .5, -.5},
             {.499999999999, -.499999999999, .499999999999}, {.002, -.003, .004},
             {67.0 / 128, 0, 0}, {67.0 / 128, -67.0 / 128, 67.0 / 128}}) {
        compare(separation);
    }
}

TEST(GravityDirectReference, CompactTablesAndOmittedForceTail) {
    auto const& tables = detail::gravityReferenceTables();
    std::set<std::array<int, 3>> actualImages;
    std::set<std::array<int, 3>> actualModes;
    for (auto const& image : tables.images) {
        EXPECT_TRUE(actualImages.insert({int(image[0]), int(image[1]), int(image[2])}).second);
    }
    for (auto const& wave : tables.waves) {
        std::array<int, 3> mode;
        for (int dim = 0; dim < 3; ++dim) {
            mode[dim] = std::lround(wave.wave[dim] / (2 * std::numbers::pi));
        }
        EXPECT_TRUE(actualModes.insert(mode).second);
        for (int dim = 0; dim < 3; ++dim) {
            mode[dim] = -mode[dim];
        }
        EXPECT_TRUE(actualModes.insert(mode).second);
    }

    constexpr long double kPi = std::numbers::pi_v<long double>;
    constexpr long double kRadius = 3.6L;
    long double realEnvelope = 0;
    long double fourierEnvelope = 0;
    size_t expectedImages = 0;
    size_t expectedModes = 0;
    // Bound each omitted real force by its magnitude at the smallest possible
    // omitted distance over the whole wrapped box; triangle-bound all modes.
    // Cube +/-10 contains the significant tail. Outside it, the max-norm shell
    // count is 24k^2+2 and r >= k-1/2, making the remaining real tail < 1e-188
    // and reciprocal tail < 1e-126 at alpha=2.
    for (int x = -10; x <= 10; ++x) {
        for (int y = -10; y <= 10; ++y) {
            for (int z = -10; z <= 10; ++z) {
                int const squared = x * x + y * y + z * z;
                if (squared == 0) {
                    continue;
                }
                long double nearestSquared = 0;
                long double nearestHaloSquared = 0;
                long double furthestSquared = 0;
                for (int const coordinate : {x, y, z}) {
                    long double const nearest = std::max(std::abs(coordinate) - .5L, 0.0L);
                    long double const nearestHalo =
                        std::max(std::abs(coordinate) - 67.0L / 128, 0.0L);
                    long double const furthest = std::abs(coordinate) + .5L;
                    nearestSquared += nearest * nearest;
                    nearestHaloSquared += nearestHalo * nearestHalo;
                    furthestSquared += furthest * furthest;
                }
                bool const candidate = nearestSquared < kRadius * kRadius;
                EXPECT_EQ(actualImages.contains({x, y, z}), candidate);
                EXPECT_EQ(actualImages.contains({x, y, z}), nearestHaloSquared < kRadius * kRadius);
                expectedImages += candidate;
                bool const mode = squared <= 16;
                EXPECT_EQ(actualModes.contains({x, y, z}), mode);
                expectedModes += mode;
                if (furthestSquared >= kRadius * kRadius) {
                    long double const radius = std::max(kRadius, std::sqrt(nearestSquared));
                    realEnvelope += std::erfc(2 * radius) / (radius * radius) +
                        4 / std::sqrt(kPi) * std::exp(-4 * radius * radius) / radius;
                }
                if (!mode) {
                    fourierEnvelope += 2 * std::exp(-kPi * kPi * squared / 4) /
                        std::sqrt(static_cast<long double>(squared));
                }
            }
        }
    }
    EXPECT_EQ(expectedImages, 364);
    EXPECT_EQ(expectedModes, 256);
    EXPECT_LT(realEnvelope, 5.2e-21L);
    EXPECT_LT(fourierEnvelope, 1.51e-17L);
}

TEST(GravityDirectReference, CoincidentAndTinySeparations) {
    auto const coincident = detail::directGravityReference({0, 0, 0}, detail::referenceKernel, .1);
    for (auto const component : coincident) {
        EXPECT_DOUBLE_EQ(component, 0);
    }
    for (double const distance : {1e-6, 1e-9, 1e-12}) {
        auto const periodic =
            detail::directGravityReference({distance, 0, 0}, detail::referenceKernel, .1);
        auto const isolated =
            detail::directGravityReference({distance, 0, 0}, detail::referenceKernel, .1, false);
        // Cubic symmetry and Poisson's equation fix the leading correction.
        EXPECT_NEAR(periodic[0] - isolated[0], 4 * std::numbers::pi * distance / 3, 3e-15);
        EXPECT_NEAR(periodic[1], 0, 3e-15);
        EXPECT_NEAR(periodic[2], 0, 3e-15);
    }
}

TEST(GravityDirectReference, SmoothCorrectionAndCubicSymmetries) {
    auto const& tables = detail::gravityReferenceTables();
    auto const zero = detail::gravityReferenceCorrection({0, 0, 0}, tables);
    for (double const component : zero) {
        EXPECT_DOUBLE_EQ(component, 0);
    }
    for (double const distance : {1e-6, 1e-9, 1e-12, 1e-15}) {
        auto const correction =
            detail::gravityReferenceCorrection({distance, -distance, distance}, tables);
        for (int dim = 0; dim < 3; ++dim) {
            double const expected = (dim == 1 ? -1 : 1) * 4 * std::numbers::pi * distance / 3;
            EXPECT_NEAR(correction[dim], expected, 2e-15);
        }
    }
    std::array<double, 3> const displacement{.13, .29, .47};
    auto const correction = detail::gravityReferenceCorrection(displacement, tables);
    std::array<int, 3> permutation{0, 1, 2};
    do {
        for (int signs = 0; signs < 8; ++signs) {
            std::array<double, 3> transformed;
            for (int dim = 0; dim < 3; ++dim) {
                transformed[dim] = ((signs >> dim) & 1 ? -1 : 1) * displacement[permutation[dim]];
            }
            auto const actual = detail::gravityReferenceCorrection(transformed, tables);
            for (int dim = 0; dim < 3; ++dim) {
                double const expected =
                    ((signs >> dim) & 1 ? -1 : 1) * correction[permutation[dim]];
                EXPECT_NEAR(actual[dim], expected, 3e-15);
            }
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
}

TEST(GravityDirectReference, MixedMassPeriodicImageReciprocity) {
    std::array<double, 3> const first{.99, .2, .4};
    std::array<double, 3> const second{.01, .3, .6};
    std::array<double, 3> separation;
    std::array<double, 3> opposite;
    for (int dim = 0; dim < 3; ++dim) {
        separation[dim] = first[dim] - second[dim];
        separation[dim] -= std::nearbyint(separation[dim]);
        opposite[dim] = -separation[dim];
    }
    auto const firstForce = detail::directGravityReference(separation, detail::referenceKernel, .3);
    auto const secondForce = detail::directGravityReference(opposite, detail::referenceKernel, .3);
    auto const softened =
        detail::directGravityReference(separation, detail::referenceKernel, .3, false);
    auto const newtonian =
        detail::directGravityReference(separation, detail::referenceKernel, .01, false);
    auto const periodicNewtonian = detail::alternateEwaldReference(separation);
    constexpr double kFirstMass = .3;
    constexpr double kSecondMass = 1.7;
    for (int dim = 0; dim < 3; ++dim) {
        EXPECT_NEAR(kFirstMass * (kSecondMass * firstForce[dim]) +
                kSecondMass * (kFirstMass * secondForce[dim]),
            0, 3e-13);
        EXPECT_NEAR(
            firstForce[dim], periodicNewtonian[dim] + softened[dim] - newtonian[dim], 3e-12);
    }
}
