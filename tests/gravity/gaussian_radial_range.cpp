/* CosmicTiger - screened radial dimensional-range regression; BSL-1.0. */

#include <cmath>
#include <numbers>

#include <gtest/gtest.h>

#include <cosmictiger/gravity/kernels/gaussian_screening.hpp>

TEST(GaussianRadialRange, FiniteShortDerivativeDespiteOverflowingInversePower) {
    using cosmictiger::gravity::evaluateGaussianRadial;
    using cosmictiger::gravity::GaussianComponent;
    using cosmictiger::gravity::GaussianScreening;
    // Volatile inputs retain the runtime arithmetic in both strict and
    // fast-math builds. exp(-64) is normal in float, isolating dimensional
    // scaling from the separate limitations of a subnormal Gaussian tail.
    float volatile runtimeRadius = 1.0e-10f;
    float volatile runtimeAlpha = 8.0e10f;
    float const radius = runtimeRadius;
    float const alpha = runtimeAlpha;
    float const radiusSquared = radius * radius;
    auto const radial = evaluateGaussianRadial<GaussianComponent::Short, 3>(
        radiusSquared, GaussianScreening<float>(alpha));
    float const actual = radial.coefficient<3, 0>();

    // Closed-form r^3 (r^-1 d/dr)^3 [-erfc(alpha*r)/r], independently
    // evaluated in long double. Forming r^-4 in float would overflow even
    // though this complete coefficient is approximately 2.47e17.
    long double const oracleRadius = std::sqrt(static_cast<long double>(radiusSquared));
    long double const z = static_cast<long double>(alpha) * oracleRadius;
    long double const zSquared = z * z;
    long double const exponentialTerm =
        2 * z / std::sqrt(std::numbers::pi_v<long double>) * std::exp(-zSquared);
    long double const expected =
        (15 * std::erfc(z) + (15 + 10 * zSquared + 4 * zSquared * zSquared) * exponentialTerm) /
        (oracleRadius * oracleRadius * oracleRadius * oracleRadius);
    EXPECT_TRUE(std::isfinite(actual));
    EXPECT_GT(actual, 0.0f);
    EXPECT_NEAR(static_cast<double>(actual), static_cast<double>(expected),
        5.0e-5 * static_cast<double>(expected));
}
