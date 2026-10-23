#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include <cosmictiger/analysis/lightcone_entry.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/math/simulation_units.hpp>

void cosmictigerAssert(char const* expression, bool condition, char const* file, int line) {
    if (!condition) {
        std::fprintf(stderr, "Assertion failed: %s at %s:%d\n", expression, file, line);
        throw std::runtime_error("Regression check failed; see diagnostic above");
    }
}

namespace detail {

Options backgroundOptions;
constexpr double kPi = 3.141592653589793238462643383279502884;

class LightconeEpoch : public ::testing::Test {
protected:
    void SetUp() override {
        backgroundOptions = Options{};
        backgroundOptions.run.scenario = "cosmology";
        backgroundOptions.units.timeSeconds = 1.0 / units::kHubble100.value();
        backgroundOptions.cosmology.hubble = 1.0;
        backgroundOptions.cosmology.omegaM = 2.0;
        backgroundOptions.cosmology.omegaK = -1.0;
        backgroundOptions.cosmology.startRedshift = 0.0;
    }
};

}    // namespace detail

Options const& getOptions() {
    return detail::backgroundOptions;
}

using detail::LightconeEpoch;

TEST_F(LightconeEpoch, SamplesBothBranchesAtTheEmissionEpoch) {
    // Closed dust: a(eta)=1-cos(eta), with eta=pi/2 at the run start.
    // The observer is at eta=3*pi/2, on the contracting branch.
    double const observationTime = detail::kPi;
    auto const background = cosmology::simulationBackground(observationTime);
    for (double const emissionTime : {0.0, 0.3, detail::kPi / 2.0, 2.1, observationTime}) {
        double const expectedScale = 1.0 + std::sin(emissionTime);
        double const radius = observationTime - emissionTime;
        EXPECT_NEAR(
            lightconeEmissionScale(*background, observationTime, radius), expectedScale, 2.0e-10);
        auto const epoch = background->sample(emissionTime);
        EXPECT_NEAR(epoch.state.conformalDerivative, std::cos(emissionTime), 2.0e-10);
        EXPECT_NEAR(epoch.physicalTime, emissionTime + 1.0 - std::cos(emissionTime), 2.0e-10);
        EXPECT_NEAR(epoch.driftIntegral,
            1.0 - 1.0 / std::tan(detail::kPi / 4.0 + 0.5 * emissionTime), 2.0e-10);
        double const storedVelocity = 0.16;
        EXPECT_NEAR(storedVelocity / lightconeEmissionScale(*background, observationTime, radius),
            storedVelocity / expectedScale, 2.0e-11);
    }
    EXPECT_GT(background->sample(1.0).state.physicalDerivative, 0.0);
    EXPECT_LT(background->sample(2.0).state.physicalDerivative, 0.0);
}

TEST_F(LightconeEpoch, TurnaroundIsCurvedAndIndependentOfQueryOrder) {
    auto const background = cosmology::simulationBackground(detail::kPi);
    double const turnaroundTime = detail::kPi / 2.0;
    double const atTurn = lightconeEmissionScale(*background, detail::kPi, turnaroundTime);
    EXPECT_NEAR(atTurn, 2.0, 2.0e-10);
    // A zero local derivative must not make the whole emission history static.
    for (double const offset : {0.6, -0.6, 0.2, -0.2, 0.0}) {
        double const emissionTime = turnaroundTime + offset;
        double const scale =
            lightconeEmissionScale(*background, detail::kPi, detail::kPi - emissionTime);
        EXPECT_NEAR(scale, 1.0 + std::cos(offset), 2.0e-10);
        if (offset != 0.0) {
            EXPECT_LT(scale, atTurn);
        }
    }
    // Restarted output asks the same absolute times, including times earlier
    // than the current driver epoch. Dense lookup is independent of past calls.
    auto const restarted = cosmology::simulationBackground(detail::kPi);
    EXPECT_EQ(restarted.get(), background.get());
    EXPECT_NEAR(restarted->sample(0.3).state.scaleFactor, 1.0 + std::sin(0.3), 2.0e-10);
}

TEST_F(LightconeEpoch, CacheTracksCosmologyAndPreservesExistingReaders) {
    auto const oldBackground = cosmology::simulationBackground(0.8);
    auto const oldState = oldBackground->sample(0.8);
    detail::backgroundOptions.cosmology.hubble = 0.5;
    auto const slowerBackground = cosmology::simulationBackground(0.8);
    EXPECT_NE(oldBackground.get(), slowerBackground.get());
    EXPECT_NEAR(slowerBackground->sample(0.8).state.scaleFactor, 1.0 + std::sin(0.4), 2.0e-10);
    EXPECT_EQ(oldBackground->sample(0.8).state.scaleFactor, oldState.state.scaleFactor);

    auto const extended = cosmology::simulationBackground(1.0);
    EXPECT_NE(extended.get(), slowerBackground.get());
    EXPECT_NEAR(extended->sample(1.0).state.scaleFactor, 1.0 + std::sin(0.5), 2.0e-10);

    detail::backgroundOptions.cosmology.startRedshift = 1.0;
    auto const earlierStart = cosmology::simulationBackground(1.0);
    EXPECT_NE(earlierStart.get(), extended.get());
    EXPECT_EQ(earlierStart->sample(0.0).state.scaleFactor, 0.5);
}

TEST_F(LightconeEpoch, StaticScenariosDoNotEvolveOrRequireCosmologicalDensities) {
    detail::backgroundOptions = Options{};
    detail::backgroundOptions.run.scenario = "glass";
    auto const background = cosmology::simulationBackground(1.0);
    for (double const emissionTime : {0.0, 0.1, 0.9, 1.0}) {
        auto const epoch = background->sample(emissionTime);
        EXPECT_EQ(lightconeEmissionScale(*background, 1.0, 1.0 - emissionTime), 1.0);
        EXPECT_EQ(epoch.state.physicalDerivative, 0.0);
        EXPECT_EQ(epoch.state.conformalDerivative, 0.0);
        EXPECT_EQ(epoch.physicalTime, emissionTime);
        EXPECT_EQ(epoch.driftIntegral, emissionTime);
    }
    cosmology::SimulationBackgroundTable const fixed(0.8, 1.0);
    EXPECT_EQ(lightconeEmissionScale(fixed, 1.0, 0.9), 0.8);
    EXPECT_NEAR(fixed.sample(0.5).physicalTime, 0.4, 1.0e-15);
    EXPECT_NEAR(fixed.sample(0.5).driftIntegral, 0.625, 1.0e-15);
}

TEST_F(LightconeEpoch, BoundsRejectInvalidQueriesAndClampBeforeTheRunStart) {
    auto const background = cosmology::simulationBackground(1.0);
    EXPECT_EQ(lightconeEmissionScale(*background, 1.0, 1.2), 1.0);
    EXPECT_EQ(background->sample(std::nextafter(1.0, 2.0)).state.scaleFactor,
        background->sample(1.0).state.scaleFactor);
    EXPECT_EQ(background->sample(std::nextafter(0.0, -1.0)).state.scaleFactor, 1.0);
    EXPECT_THROW(background->sample(-0.1), std::out_of_range);
    EXPECT_THROW(background->sample(1.1), std::out_of_range);
    EXPECT_THROW(lightconeEmissionScale(*background, 1.0, -0.1), std::invalid_argument);
    EXPECT_THROW(background->sample(std::numeric_limits<double>::quiet_NaN()), std::out_of_range);
}
