#include <cmath>

#include <gtest/gtest.h>

#include <cosmictiger/scenarios/gresho.hpp>

TEST(Gresho, CenterAndExterior) {
    auto const center = scenarios::greshoState(0.0);
    EXPECT_DOUBLE_EQ(center.pressure, 5.0);
    EXPECT_DOUBLE_EQ(center.azimuthalVelocity, 0.0);
    auto const exterior = scenarios::greshoState(0.7);
    EXPECT_DOUBLE_EQ(exterior.pressure, 3.0 + 4.0 * std::log(2.0));
    EXPECT_DOUBLE_EQ(exterior.azimuthalVelocity, 0.0);
}

TEST(Gresho, ContinuousAtInterfaces) {
    for (double const radius : {0.2, 0.4}) {
        auto const left = scenarios::greshoState(radius - 1.e-10);
        auto const right = scenarios::greshoState(radius + 1.e-10);
        EXPECT_NEAR(left.pressure, right.pressure, 2.e-9);
        EXPECT_NEAR(left.azimuthalVelocity, right.azimuthalVelocity, 2.e-9);
    }
    EXPECT_DOUBLE_EQ(scenarios::greshoState(0.2).azimuthalVelocity, 1.0);
}

TEST(Gresho, PressureBalancesCentrifugalAcceleration) {
    constexpr double kDelta = 1.e-6;
    for (double const radius : {0.05, 0.15, 0.25, 0.35, 0.5}) {
        auto const state = scenarios::greshoState(radius);
        double const gradient = (scenarios::greshoState(radius + kDelta).pressure -
                                    scenarios::greshoState(radius - kDelta).pressure) /
            (2.0 * kDelta);
        EXPECT_NEAR(gradient, state.azimuthalVelocity * state.azimuthalVelocity / radius, 1.e-8);
    }
}

TEST(Gresho, PeriodicClosePackedLatticeHasTwelveNearestNeighbors) {
    int constexpr kSide = 4;
    auto const origin = scenarios::greshoLatticePosition(0, 0, 0, 0, kSide, true);
    int neighbors = 0;
    for (int x = 0; x < kSide; ++x) {
        for (int y = 0; y < kSide; ++y) {
            for (int z = 0; z < kSide; ++z) {
                for (int phase = 0; phase < 4; ++phase) {
                    auto const point =
                        scenarios::greshoLatticePosition(x, y, z, phase, kSide, true);
                    double distance2 = 0.0;
                    for (int dim = 0; dim < 3; ++dim) {
                        double const delta = point[dim] - origin[dim];
                        distance2 += (delta - std::round(delta)) * (delta - std::round(delta));
                    }
                    if (distance2 > 1.e-12) {
                        EXPECT_GE(distance2, 0.5 / (kSide * kSide) - 1.e-12);
                        if (std::abs(distance2 - 0.5 / (kSide * kSide)) < 1.e-12) {
                            ++neighbors;
                        }
                    }
                }
            }
        }
    }
    EXPECT_EQ(neighbors, 12);
}

TEST(Gresho, CubicLatticeUsesCellCenters) {
    auto const position = scenarios::greshoLatticePosition(1, 2, 3, 0, 4, false);
    EXPECT_DOUBLE_EQ(position[0], 1.5 / 4.0);
    EXPECT_DOUBLE_EQ(position[1], 2.5 / 4.0);
    EXPECT_DOUBLE_EQ(position[2], 3.5 / 4.0);
}
