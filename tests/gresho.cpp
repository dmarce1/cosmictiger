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
