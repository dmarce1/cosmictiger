#include <limits>

#include <gtest/gtest.h>

#include <cosmictiger/core/terminal.hpp>

TEST(Terminal, DurationRoundsAcrossMinuteAndHourBoundaries) {
    EXPECT_EQ(terminal::formatDuration(0.0), "00:00::00.00");
    EXPECT_EQ(terminal::formatDuration(0.004), "00:00::00.00");
    EXPECT_EQ(terminal::formatDuration(0.006), "00:00::00.01");
    EXPECT_EQ(terminal::formatDuration(59.994), "00:00::59.99");
    EXPECT_EQ(terminal::formatDuration(59.996), "00:01::00.00");
    EXPECT_EQ(terminal::formatDuration(3599.996), "01:00::00.00");
}

TEST(Terminal, DurationPreservesHoursAcrossDays) {
    EXPECT_EQ(terminal::formatDuration(3661.25), "01:01::01.25");
    EXPECT_EQ(terminal::formatDuration(86400.0), "24:00::00.00");
    EXPECT_EQ(terminal::formatDuration(360000.0), "100:00::00.00");
}

TEST(Terminal, InvalidDurationsRemainPrintable) {
    EXPECT_EQ(terminal::formatDuration(-1.0), "00:00::00.00");
    EXPECT_EQ(terminal::formatDuration(std::numeric_limits<double>::infinity()), "--:--::--.--");
    EXPECT_EQ(terminal::formatDuration(std::numeric_limits<double>::quiet_NaN()), "--:--::--.--");
    EXPECT_EQ(terminal::formatDuration(std::numeric_limits<double>::max()), "--:--::--.--");
}
