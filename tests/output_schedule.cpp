#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include <cosmictiger/gravity/outer_interval.hpp>

TEST(OutputSchedule, RoundsUpToTheNearestIntegerSubdivision) {
    EXPECT_DOUBLE_EQ(pmSelectOuterInterval(0.26, 100.0, 0.0, 1.0, 2.0), 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(pmSelectOuterInterval(100.0, 0.26, 0.0, 1.0, 2.0), 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(pmSelectOuterInterval(0.2, 100.0, 0.0, 1.0, 2.0), 0.2);
    EXPECT_DOUBLE_EQ(pmSelectOuterInterval(100.0, 100.0, 0.4, 1.0, 2.0), 0.6);
    EXPECT_NEAR(pmSelectOuterInterval(0.21, 100.0, 0.35, 1.0, 2.0), 0.65 / 3.0, 1.e-16);
}

TEST(OutputSchedule, IntegerQuotientRoundoffDoesNotDropAStep) {
    EXPECT_DOUBLE_EQ(pmSelectOuterInterval(std::nextafter(0.1, 1.0), 10.0, 0.0, 1.0, 2.0), 0.1);
    EXPECT_DOUBLE_EQ(pmSelectOuterInterval(0.1, 10.0, 0.7, 1.0, 2.0), (1.0 - 0.7) / 3.0);
}

TEST(OutputSchedule, ReachesEveryUniformOutputPhaseUnderChangingBounds) {
    double time = 0.0;
    int selections = 0;
    for (int output = 1; output <= 10; ++output) {
        double const endpoint = output == 10 ? 1.0 : output * 0.1;
        while (time < endpoint) {
            ASSERT_LT(++selections, 200);
            double const candidate = selections % 3 == 0 ? 0.047 : 0.017;
            double const interval = pmSelectOuterInterval(candidate, 10.0, time, 0.1, 1.0);
            double const ratio = (endpoint - time) / interval;
            EXPECT_NEAR(ratio, std::round(ratio), 1.e-12);
            EXPECT_GE(interval + 1.e-15, std::min(candidate, endpoint - time));
            time += interval;
            ASSERT_LE(time, endpoint);
        }
        EXPECT_DOUBLE_EQ(time, endpoint);
    }
}

TEST(OutputSchedule, EndTimeCapsTheLastOutputInterval) {
    EXPECT_DOUBLE_EQ(pmSelectOuterInterval(10.0, 10.0, 0.9, 1.0, 0.95), 0.95 - 0.9);
    EXPECT_DOUBLE_EQ(pmSelectOuterInterval(0.1, 0.1, 1.0, 1.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(pmSelectOuterInterval(0.1, 0.1, std::nextafter(1.0, 0.0), 1.0, 1.0), 0.0);
}

TEST(OutputSchedule, RejectsIntervalsThatCannotAdvanceTheClock) {
    EXPECT_THROW(pmSelectOuterInterval(0.1, 0.1, 0.0, 0.0, 1.0), std::runtime_error);
    EXPECT_THROW(pmSelectOuterInterval(0.0, 0.1, 0.0, 1.0, 1.0), std::runtime_error);
    EXPECT_THROW(
        pmSelectOuterInterval(std::numeric_limits<double>::denorm_min(), 0.1, 0.0, 1.0, 1.0),
        std::runtime_error);
}
