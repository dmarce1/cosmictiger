#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

#include <cosmictiger/scenarios/sedov.hpp>

TEST(Sedov, CacheRespectsExtentAndDimensionality) {
    double density, velocity, pressure;
    sedov::solution(0.005, 0.05, 0.25, density, velocity, pressure, 3);
    // A larger table at the same time must permit sampling beyond the extent
    // of the first cached table. Undisturbed gas has rho=1, v=P=0.
    sedov::solution(0.005, 0.75, 1.0, density, velocity, pressure, 3);
    EXPECT_NEAR(density, 1.0, 1.0e-12);
    EXPECT_NEAR(velocity, 0.0, 1.0e-12);
    EXPECT_NEAR(pressure, 0.0, 1.0e-12);
    sedov::solution(0.005, 0.05, 1.0, density, velocity, pressure, 3);
    double const sphericalPressure = pressure;
    sedov::solution(0.005, 0.05, 1.0, density, velocity, pressure, 1);
    EXPECT_GT(std::abs(pressure - sphericalPressure), 1.0e-3);
}

TEST(Sedov, CacheRespectsGasGamma) {
    double density, velocity, pressure;
    sedov::solution(0.01, 0.05, 1.0, density, velocity, pressure, 3, 5.0 / 3.0);
    double const monatomicPressure = pressure;
    sedov::solution(0.01, 0.05, 1.0, density, velocity, pressure, 3, 7.0 / 5.0);
    EXPECT_TRUE(std::isfinite(density));
    EXPECT_TRUE(std::isfinite(velocity));
    EXPECT_TRUE(std::isfinite(pressure));
    EXPECT_GT(std::abs(pressure - monatomicPressure), 1.0e-3);
}

TEST(Sedov, RejectsInvalidInterpolationParameters) {
    double density, velocity, pressure;
    EXPECT_THROW(
        sedov::solution(0.0, 0.05, 1.0, density, velocity, pressure, 3), std::invalid_argument);
    EXPECT_THROW(
        sedov::solution(0.01, -0.05, 1.0, density, velocity, pressure, 3), std::invalid_argument);
    EXPECT_THROW(
        sedov::solution(0.01, 1.05, 1.0, density, velocity, pressure, 3), std::invalid_argument);
    EXPECT_THROW(
        sedov::solution(0.01, 0.05, 0.0, density, velocity, pressure, 3), std::invalid_argument);
    EXPECT_THROW(
        sedov::solution(0.01, 0.05, 1.0, density, velocity, pressure, 4), std::invalid_argument);
    EXPECT_THROW(sedov::solution(0.01, 0.05, 1.0, density, velocity, pressure, 3, 1.0),
        std::invalid_argument);
}
