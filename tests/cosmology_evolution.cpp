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

TEST(CosmologyEvolution, FreeStreamingPreservesCosmicEnergyThroughExpansionAndContraction) {
    // For fixed canonical momenta, K = Q/a^2. A constant comoving potential
    // contributes W = C/a and therefore must not affect the expansion work.
    constexpr double kCanonicalKinetic = 1.25;
    constexpr double kComovingPotential = -3.0;
    double const initialScale = 0.02;
    double scale = initialScale;
    double work = -(kCanonicalKinetic / initialScale + kComovingPotential);
    for (double const nextScale : {0.021, 0.1, 0.5, 1.0, 1.5, 1.5, 0.7, 0.02}) {
        work += cosmology::kineticExpansionWork(scale, nextScale, kCanonicalKinetic);
        double const kinetic = kCanonicalKinetic / (nextScale * nextScale);
        double const potential = kComovingPotential / nextScale;
        EXPECT_NEAR(
            cosmology::normalizedEnergyError(nextScale, kinetic, potential, 0.0, 0.0, work, 0.0),
            0.0, 2.0e-14);
        scale = nextScale;
    }
}

TEST(CosmologyEvolution, ExpansionWorkIsIndependentOfDriftSubdivision) {
    constexpr double kCanonicalKinetic = 0.75;
    double const whole = cosmology::kineticExpansionWork(0.02, 1.0, kCanonicalKinetic);
    double scale = 0.02;
    double pieces = 0.0;
    for (double const nextScale : {0.025, 0.04, 0.11, 0.3, 0.8, 1.0}) {
        pieces += cosmology::kineticExpansionWork(scale, nextScale, kCanonicalKinetic);
        scale = nextScale;
    }
    EXPECT_NEAR(pieces, whole, 2.0e-14);
    EXPECT_NEAR(whole, kCanonicalKinetic / 0.02 - kCanonicalKinetic, 2.0e-14);
    EXPECT_DOUBLE_EQ(cosmology::kineticExpansionWork(1.0, 1.0, kCanonicalKinetic), 0.0);
}

TEST(CosmologyEvolution, EnergyNormalizationUsesConsistentlyScaledTerms) {
    // The same scaled energy budget has residual -1/4 and absolute sum 27/4
    // at every epoch, including nonzero thermal and source terms.
    for (double const scale : {0.02, 0.125, 1.0, 2.0}) {
        EXPECT_NEAR(cosmology::normalizedEnergyError(
                        scale, 2.0 / scale, -3.0 / scale, 0.5 / scale, -0.5, 0.75, 0.0),
            -1.0 / 27.0, 1.0e-15);
        EXPECT_NEAR(cosmology::normalizedEnergyError(
                        scale, 2.0 / scale, -3.0 / scale, 0.5 / scale, -0.5, 0.75, -0.25),
            0.0, 1.0e-15);
    }
    EXPECT_DOUBLE_EQ(cosmology::normalizedEnergyError(0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0);
}
