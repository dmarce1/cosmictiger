/* CosmicTiger - cosmology solver and diagnostic FLOP regression; BSL-1.0. */

#include <cmath>
#include <initializer_list>

#include <gtest/gtest.h>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/math/simulation_units.hpp>

namespace detail {

Options cosmologyOptions;
FlopCount submittedCosmologyFlops = 0;
unsigned cosmologySubmissions = 0;

void resetCosmologyCount() {
    submittedCosmologyFlops = 0;
    cosmologySubmissions = 0;
}

class CosmologyFlops : public ::testing::Test {
protected:
    void SetUp() override {
        cosmologyOptions = Options{};
        // Einstein-de Sitter has an analytic a(tau), independent of the
        // production integration stages and the FLOP-accounting formulas.
        cosmologyOptions.units.timeSeconds = 1.0 / units::kHubble100.value();
        cosmologyOptions.cosmology.hubble = 0.7;
        cosmologyOptions.cosmology.omegaM = 1.0;
        cosmologyOptions.cosmology.omegaR = 0.0;
        cosmologyOptions.cosmology.omegaK = 0.0;
        cosmologyOptions.cosmology.omegaLambda = 0.0;
        resetCosmologyCount();
    }

    static double hubble() {
        return units::normalizedHubble100(cosmologyOptions) * cosmologyOptions.cosmology.hubble;
    }
};

}    // namespace detail

// Link the real cosmology implementation; replace only its options service
// and counter sink, so this regression does not require an HPX runtime.
Options const& getOptions() {
    return detail::cosmologyOptions;
}

void addCpuFlops(FlopCount count) {
    detail::submittedCosmologyFlops += count;
    ++detail::cosmologySubmissions;
}

using detail::CosmologyFlops;

TEST_F(CosmologyFlops, DerivativesIncludeUnitNormalizationExactlyOnce) {
    for (double const scale : {0.125, 0.5, 1.0}) {
        detail::resetCosmologyCount();
        EXPECT_NEAR(cosmosDadt(scale), hubble() / std::sqrt(scale), 1.e-14);
        EXPECT_EQ(detail::submittedCosmologyFlops, 22);
        EXPECT_EQ(detail::cosmologySubmissions, 1);

        detail::resetCosmologyCount();
        double const conformalDerivative = cosmosDadtau(scale);
        EXPECT_NEAR(conformalDerivative, hubble() * std::sqrt(scale), 1.e-14);
        EXPECT_EQ(detail::submittedCosmologyFlops, 22);
        EXPECT_EQ(detail::cosmologySubmissions, 1);

        detail::resetCosmologyCount();
        EXPECT_EQ(cosmosDadtau(scale, false), conformalDerivative);
        EXPECT_EQ(detail::submittedCosmologyFlops, 0);
        EXPECT_EQ(detail::cosmologySubmissions, 0);
    }
}

TEST_F(CosmologyFlops, EvolutionAndInverseScaleBatchTheirActualStages) {
    constexpr double kInitialScale = 0.16;
    constexpr double kTimestep = 0.02;
    double const rootScale = std::sqrt(kInitialScale);
    double const finalRoot = rootScale + 0.5 * hubble() * kTimestep;
    double const expectedScale = finalRoot * finalRoot;
    double scale = kInitialScale;
    double derivative = hubble() / rootScale;
    double secondDerivative = 0.0;

    cosmosUpdate(secondDerivative, derivative, scale, kTimestep);
    EXPECT_NEAR(scale, expectedScale, 1.e-11);
    EXPECT_NEAR(derivative, hubble() / finalRoot, 1.e-10);
    EXPECT_GT(detail::submittedCosmologyFlops, 100);
    EXPECT_EQ(detail::cosmologySubmissions, 2);
    EXPECT_NEAR(secondDerivative, -0.5 * hubble() * hubble() / (scale * scale), 1.e-12);

    detail::resetCosmologyCount();
    scale = kInitialScale;
    derivative = hubble() / rootScale;
    double const inverseScale = cosmosAinv(derivative, scale, kTimestep);
    EXPECT_NEAR(scale, expectedScale, 1.e-11);
    EXPECT_NEAR(derivative, hubble() / finalRoot, 1.e-10);
    // Exact interval average of 1/a for a(tau) = (sqrt(a0) + H0*tau/2)^2.
    EXPECT_NEAR(inverseScale, 1.0 / (rootScale * finalRoot), 1.e-7);
    EXPECT_GT(detail::submittedCosmologyFlops, 100);
    EXPECT_EQ(detail::cosmologySubmissions, 2);
}

TEST_F(CosmologyFlops, DiagnosticAgeAndHaloConversionsSubmitNothing) {
    constexpr double kInitialScale = 0.16;
    constexpr double kFinalScale = 0.25;
    double const age = cosmosTime(kInitialScale, kFinalScale);
    double const expectedAge =
        2.0 / (3.0 * hubble()) * (std::pow(kFinalScale, 1.5) - std::pow(kInitialScale, 1.5));
    EXPECT_NEAR(age, expectedAge, 1.e-11);

    constexpr double kConformalInterval = 0.02;
    double const scale = cosmosTauToScale(kInitialScale, kConformalInterval);
    double const expectedRoot = std::sqrt(kInitialScale) + 0.5 * hubble() * kConformalInterval;
    EXPECT_NEAR(scale, expectedRoot * expectedRoot, 1.e-7);
    EXPECT_EQ(detail::submittedCosmologyFlops, 0);
    EXPECT_EQ(detail::cosmologySubmissions, 0);
}

TEST_F(CosmologyFlops, ConformalIntegralSeparatesSolverAndDiagnosticUse) {
    constexpr double kInitialScale = 0.16;
    constexpr double kFinalScale = 0.25;
    double const interval = cosmosConformalTime(kInitialScale, kFinalScale);
    double const expected = 2.0 / hubble() * (std::sqrt(kFinalScale) - std::sqrt(kInitialScale));
    EXPECT_NEAR(interval, expected, 1.e-9);
    EXPECT_EQ(detail::cosmologySubmissions, 1);
    EXPECT_GT(detail::submittedCosmologyFlops, 100);

    detail::resetCosmologyCount();
    EXPECT_EQ(cosmosConformalTime(kInitialScale, kFinalScale, false), interval);
    EXPECT_EQ(detail::submittedCosmologyFlops, 0);
    EXPECT_EQ(detail::cosmologySubmissions, 0);
}

TEST_F(CosmologyFlops, WrappersCarryContractionAndIntegratePhysicalTimeAndDrift) {
    detail::cosmologyOptions.cosmology.omegaM = 2.0;
    detail::cosmologyOptions.cosmology.omegaK = -1.0;
    double const pi = std::acos(-1.0);
    double scale = 1.0;
    double derivative = hubble();
    auto const result = cosmosAdvance(derivative, scale, pi / hubble());
    EXPECT_NEAR(scale, 1.0, 1.e-10);
    EXPECT_NEAR(derivative, -hubble(), 1.e-10);
    EXPECT_NEAR(result.physicalTime, (pi + 2.0) / hubble(), 1.e-10);
    EXPECT_NEAR(result.driftIntegral, 2.0 / hubble(), 1.e-10);
    EXPECT_NEAR(cosmosTauToScale(1.0, pi / hubble()), 1.0, 1.e-10);
    EXPECT_NEAR(cosmosConformalTime(1.0, 2.0), 0.5 * pi / hubble(), 1.e-11);
    EXPECT_NEAR(cosmosTime(1.0, 2.0), (0.5 * pi + 1.0) / hubble(), 1.e-11);
    EXPECT_NEAR(cosmosConformalTime(2.0, 1.0), -0.5 * pi / hubble(), 1.e-11);
}

TEST_F(CosmologyFlops, ZeroIntervalAverageAndReversedTimeIntegralRemainFinite) {
    double scale = 0.25;
    double derivative = 2.0 * hubble();
    EXPECT_DOUBLE_EQ(cosmosAinv(derivative, scale, 0.0), 4.0);
    EXPECT_DOUBLE_EQ(scale, 0.25);
    EXPECT_DOUBLE_EQ(derivative, 2.0 * hubble());
    EXPECT_NEAR(cosmosConformalTime(0.25, 0.16), -0.2 / hubble(), 1.e-12);
    EXPECT_NEAR(cosmosTime(0.25, 0.16),
        2.0 / (3.0 * hubble()) * (std::pow(0.16, 1.5) - std::pow(0.25, 1.5)), 1.e-12);
}
