#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <cosmictiger/cosmology/evolution.hpp>

TEST(CosmologyEvolution, ContractionAndExpansionHaveTheSamePositiveBound) {
    for (double const rate : {0.0, 1.0e-12, 0.1, 1.0, 100.0}) {
        double const expanding = cosmology::expansionTimestep(0.7, rate, -0.3);
        double const contracting = cosmology::expansionTimestep(0.7, -rate, -0.3);
        EXPECT_GT(contracting, 0.0);
        EXPECT_TRUE(std::isfinite(contracting));
        EXPECT_DOUBLE_EQ(contracting, expanding);
        EXPECT_TRUE(std::isfinite(std::log2(1.0 / contracting)));
        double const quadraticChange =
            rate * expanding + 0.5 * (rate * rate + 0.7 * 0.3) * expanding * expanding;
        EXPECT_NEAR(quadraticChange, 0.01, 1.0e-15);
    }
}

TEST(CosmologyEvolution, TurnaroundUsesAccelerationAndStaticBackgroundIsUnrestricted) {
    EXPECT_NEAR(cosmology::expansionTimestep(2.0, 0.0, -0.25), 0.2, 1.0e-15);
    EXPECT_DOUBLE_EQ(cosmology::expansionTimestep(1.0, 0.0, 0.0), 1.0e30);
}

TEST(CosmologyEvolution, EndpointRoundoffDoesNotAddAnotherCoarseStep) {
    EXPECT_TRUE(cosmology::endTimeReached(1.0, 1.0));
    EXPECT_TRUE(cosmology::endTimeReached(std::nextafter(1.0, 0.0), 1.0));
    EXPECT_FALSE(cosmology::endTimeReached(0.999, 1.0));
    EXPECT_TRUE(cosmology::endTimeReached(0.0, 0.0));
    EXPECT_FALSE(cosmology::endTimeReached(0.0, 1.0e-20));
}
