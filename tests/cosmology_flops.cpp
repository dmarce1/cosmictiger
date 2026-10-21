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
        EXPECT_EQ(detail::submittedCosmologyFlops, 17);
        EXPECT_EQ(detail::cosmologySubmissions, 1);

        detail::resetCosmologyCount();
        double const conformalDerivative = cosmosDadtau(scale);
        EXPECT_NEAR(conformalDerivative, hubble() * std::sqrt(scale), 1.e-14);
        EXPECT_EQ(detail::submittedCosmologyFlops, 18);
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
    EXPECT_EQ(detail::submittedCosmologyFlops, 9729);
    EXPECT_EQ(detail::cosmologySubmissions, 1);

    detail::resetCosmologyCount();
    scale = kInitialScale;
    derivative = hubble() / rootScale;
    double const inverseScale = cosmosAinv(derivative, scale, kTimestep);
    EXPECT_NEAR(scale, expectedScale, 1.e-11);
    EXPECT_NEAR(derivative, hubble() / finalRoot, 1.e-10);
    // Exact interval average of 1/a for a(tau) = (sqrt(a0) + H0*tau/2)^2.
    EXPECT_NEAR(inverseScale, 1.0 / (rootScale * finalRoot), 1.e-7);
    EXPECT_EQ(detail::submittedCosmologyFlops, 10369);
    EXPECT_EQ(detail::cosmologySubmissions, 1);
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
    unsigned const refinements = detail::cosmologySubmissions;
    ASSERT_GT(refinements, 1);
    ASSERT_LT(refinements, 32);
    // Refinements visit 2,4,...,2^r trapezoids. Each contributes 29 operations;
    // each refinement has 25 setup/termination operations.
    EXPECT_EQ(detail::submittedCosmologyFlops,
        25 * FlopCount(refinements) + 29 * ((FlopCount(1) << (refinements + 1)) - 2));

    detail::resetCosmologyCount();
    EXPECT_EQ(cosmosConformalTime(kInitialScale, kFinalScale, false), interval);
    EXPECT_EQ(detail::submittedCosmologyFlops, 0);
    EXPECT_EQ(detail::cosmologySubmissions, 0);
}
