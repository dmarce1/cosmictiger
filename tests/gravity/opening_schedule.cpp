#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include <cosmictiger/gravity/opening_schedule.hpp>

TEST(OpeningSchedule, ParsesUnsortedPairsAndWhitespace) {
    auto const schedule =
        gravity::parseOpeningSchedule(" 20:0.51, -0.5 : 0.7 , +49:4.1e-1,2:0.61 ");
    ASSERT_EQ(schedule.size(), 4);
    EXPECT_DOUBLE_EQ(schedule[0].redshift, -0.5);
    EXPECT_DOUBLE_EQ(schedule[0].theta, 0.7);
    EXPECT_DOUBLE_EQ(schedule[3].redshift, 49.0);
    EXPECT_DOUBLE_EQ(schedule[3].theta, 0.41);
}

TEST(OpeningSchedule, RejectsMalformedOrUnrepresentableValues) {
    for (std::string const text : {"49", "49:", ":0.4", "49:0.4:", "49:0.4,", ",49:0.4",
             "49:0.4,,2:0.6", "49:0.4garbage", "49:0.4 2:0.6", "nan:0.4", "inf:0.4", "2:nan",
             "2:inf", "2:-inf", "-1:0.4", "-2:0.4", "2:0", "2:-0.5", "2:1e40", "2:1e-45", "2:1e999",
             "2:0.4,2.0:0.5", "+-0.5:0.4", "2:+-0.4", "++2:0.4", "2:++0.4"}) {
        EXPECT_THROW(gravity::parseOpeningSchedule(text), std::invalid_argument) << text;
    }
}

TEST(OpeningSchedule, DoesNotImposeAnUnrequestedUpperAngleLimit) {
    auto const schedule = gravity::parseOpeningSchedule("2:1.5");
    ASSERT_EQ(schedule.size(), 1);
    EXPECT_DOUBLE_EQ(schedule[0].theta, 1.5);
}

TEST(OpeningSchedule, EmptySchedulePreservesConstantTheta) {
    auto const schedule = gravity::parseOpeningSchedule(" \t ");
    EXPECT_TRUE(schedule.empty());
    EXPECT_DOUBLE_EQ(gravity::openingScheduleInitialTheta(schedule, 49.0, 0.8), 0.8);
    EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 49.0, 0.0, 0.8), 0.8);
    EXPECT_TRUE(gravity::parseOpeningSchedule("none").empty());
    EXPECT_TRUE(gravity::parseOpeningSchedule(" \t none \n ").empty());
}

TEST(OpeningSchedule, StartsAtNearestRedshiftWithHigherTie) {
    auto const schedule = gravity::parseOpeningSchedule("10:0.5,20:0.4,0:0.6");
    EXPECT_DOUBLE_EQ(gravity::openingScheduleInitialTheta(schedule, 49.0, 0.8), 0.4);
    EXPECT_DOUBLE_EQ(gravity::openingScheduleInitialTheta(schedule, 12.0, 0.8), 0.5);
    EXPECT_DOUBLE_EQ(gravity::openingScheduleInitialTheta(schedule, 15.0, 0.8), 0.4);
    EXPECT_DOUBLE_EQ(gravity::openingScheduleInitialTheta(schedule, -0.5, 0.8), 0.6);
}

TEST(OpeningSchedule, ChangesAtExactThresholdAndRestoresOnlyAboveIt) {
    auto const schedule = gravity::parseOpeningSchedule("49:0.41,20:0.51,2:0.61");
    EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 49.0, 25.0, 0.8), 0.41);
    EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 49.0, 20.0, 0.8), 0.51);
    EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 49.0, 19.0, 0.8), 0.51);
    EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 49.0, 2.0, 0.8), 0.61);
    EXPECT_DOUBLE_EQ(
        gravity::openingScheduleTheta(schedule, 49.0, std::nextafter(2.0, 3.0), 0.8), 0.51);
    EXPECT_DOUBLE_EQ(
        gravity::openingScheduleTheta(schedule, 49.0, std::nextafter(20.0, 21.0), 0.8), 0.41);
}

TEST(OpeningSchedule, ReplaysExpansionIntervalsInReverse) {
    auto const schedule = gravity::parseOpeningSchedule("49:0.41,20:0.51,2:0.61");
    double const redshifts[] = {49.0, 25.0, 20.0, 2.0, 0.0, 2.0, 3.0, 20.0, 21.0, 49.0, 60.0};
    double const expected[] = {0.41, 0.41, 0.51, 0.61, 0.61, 0.61, 0.51, 0.51, 0.41, 0.41, 0.41};
    for (size_t index = 0; index < sizeof(redshifts) / sizeof(redshifts[0]); ++index) {
        EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 49.0, redshifts[index], 0.8),
            expected[index]);
    }
}

TEST(OpeningSchedule, PreservesNearestInitialThetaDuringReverseReplay) {
    auto const schedule = gravity::parseOpeningSchedule("49:0.41,20:0.51,2:0.61");
    EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 25.0, 25.0, 0.8), 0.51);
    EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 25.0, 1.0, 0.8), 0.61);
    EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 25.0, 21.0, 0.8), 0.51);
    EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 25.0, 60.0, 0.8), 0.51);
    EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 1.0, 0.0, 0.8), 0.61);
    EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 1.0, 60.0, 0.8), 0.61);
}

TEST(OpeningSchedule, HandlesMultipleThresholdLeapsAndTurnaroundWithinOneStep) {
    auto const schedule = gravity::parseOpeningSchedule("49:0.41,20:0.51,2:0.61");
    EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 60.0, 1.0, 0.8), 0.61);
    EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 60.0, 50.0, 0.8), 0.41);
    EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 60.0, 20.0, 0.8), 0.51);
    // Returning to the same redshift restores its original angle even if a
    // single step crossed several thresholds and turned around in between.
    EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 49.0, 25.0, 0.61), 0.41);
}

TEST(OpeningSchedule, RestartReconstructsOriginalIntervalsOnBothBranches) {
    auto const schedule = gravity::parseOpeningSchedule("0.1:0.4,-0.25:0.5,-0.4:0.6");
    EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 0.0, 0.0, 0.8), 0.4);
    EXPECT_DOUBLE_EQ(gravity::openingScheduleInitialTheta(schedule, -0.35, 0.8), 0.6);
    EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 0.0, -0.35, 0.8), 0.5);
    EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 0.0, -0.5, 0.8), 0.6);
    EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 0.0, -0.25, 0.8), 0.5);
    EXPECT_DOUBLE_EQ(gravity::openingScheduleTheta(schedule, 0.0, -0.2, 0.8), 0.4);
}
