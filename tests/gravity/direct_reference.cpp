/* CosmicTiger - independent sampled-gravity reference checks; BSL-1.0. */

#include <array>
#include <cmath>
#include <numbers>

#include <gtest/gtest.h>

#include <cosmictiger/gravity/direct_reference.hpp>

namespace detail {

// A separate Ewald splitting and larger cubes check the converged periodic
// field, without copying the production reference's central correction split.
std::array<double, 3> alternateEwaldReference(std::array<double, 3> const& separation) {
    constexpr double kAlpha = 2.7;
    constexpr double kPi = std::numbers::pi;
    double const normalization = 2 * kAlpha / std::sqrt(kPi);
    std::array<double, 3> result{};
    for (int x = -5; x <= 5; ++x) {
        for (int y = -5; y <= 5; ++y) {
            for (int z = -5; z <= 5; ++z) {
                std::array<double, 3> const image{
                    separation[0] + x, separation[1] + y, separation[2] + z};
                double const squared =
                    image[0] * image[0] + image[1] * image[1] + image[2] * image[2];
                double const radius = std::sqrt(squared);
                double const radial = (std::erfc(kAlpha * radius) / radius +
                                          normalization * std::exp(-kAlpha * kAlpha * squared)) /
                    squared;
                for (int dim = 0; dim < 3; ++dim) {
                    result[dim] -= radial * image[dim];
                }
                int const waveSquared = x * x + y * y + z * z;
                if (waveSquared != 0) {
                    std::array<double, 3> const wave{2 * kPi * x, 2 * kPi * y, 2 * kPi * z};
                    double const amplitude =
                        std::exp(-kPi * kPi * waveSquared / (kAlpha * kAlpha)) /
                        (kPi * waveSquared) *
                        std::sin(wave[0] * separation[0] + wave[1] * separation[1] +
                            wave[2] * separation[2]);
                    for (int dim = 0; dim < 3; ++dim) {
                        result[dim] -= amplitude * wave[dim];
                    }
                }
            }
        }
    }
    return result;
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
