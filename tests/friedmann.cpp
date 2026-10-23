/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include <cosmictiger/cosmology/friedmann.hpp>

namespace detail {

using cosmology::BackgroundCosmologyParameters;
using cosmology::CosmologicalTime;
using cosmology::ExpansionBranch;
using cosmology::FriedmannCosmology;
using cosmology::FriedmannOptions;
using cosmology::FriedmannState;

BackgroundCosmologyParameters parameters(
    double radiation, double matter, double darkEnergy, double curvature, double hubble = 1.0) {
    BackgroundCosmologyParameters result;
    result.hubbleConstant = hubble;
    result.omegaRadiation = radiation;
    result.omegaMatter = matter;
    result.omegaDarkEnergy = darkEnergy;
    result.omegaCurvature = curvature;
    return result;
}

void expectState(FriedmannState const& actual, double scaleFactor, double physicalDerivative,
    double conformalDerivative, double tolerance = 1e-10) {
    EXPECT_NEAR(actual.scaleFactor, scaleFactor, tolerance * (1.0 + std::abs(scaleFactor)));
    EXPECT_NEAR(actual.physicalDerivative, physicalDerivative,
        tolerance * (1.0 + std::abs(physicalDerivative)));
    EXPECT_NEAR(actual.conformalDerivative, conformalDerivative,
        tolerance * (1.0 + std::abs(conformalDerivative)));
}

void expectSameState(
    FriedmannState const& actual, FriedmannState const& expected, double tolerance = 1e-10) {
    expectState(actual, expected.scaleFactor, expected.physicalDerivative,
        expected.conformalDerivative, tolerance);
}

TEST(Friedmann, DefaultsAndPresentEpochNormalization) {
    FriedmannCosmology const cosmology;
    EXPECT_DOUBLE_EQ(cosmology.parameters().hubbleConstant, 1.0);
    EXPECT_DOUBLE_EQ(cosmology.parameters().omegaRadiation, 0.0);
    EXPECT_DOUBLE_EQ(cosmology.parameters().omegaMatter, 0.3);
    EXPECT_DOUBLE_EQ(cosmology.parameters().omegaDarkEnergy, 0.7);
    EXPECT_DOUBLE_EQ(cosmology.parameters().omegaCurvature, 0.0);
    EXPECT_DOUBLE_EQ(cosmology.options().relativeTolerance, 1e-12);
    EXPECT_DOUBLE_EQ(cosmology.options().absoluteTolerance, 1e-14);
    EXPECT_EQ(cosmology.options().maxSteps, size_t(100000));
    expectState(cosmology.state(1.0), 1.0, 1.0, 1.0, 1e-14);
    expectState(cosmology.state(1.0, ExpansionBranch::contracting), 1.0, -1.0, -1.0, 1e-14);
}

TEST(Friedmann, PureRadiationMatchesBothAnalyticClocksAndBranches) {
    constexpr double kHubble = 1.7;
    constexpr double kInitialScaleFactor = 0.8;
    constexpr double kElapsed = 0.08;

    FriedmannCosmology const cosmology(parameters(1.0, 0.0, 0.0, 0.0, kHubble));
    for (auto const branch : {ExpansionBranch::expanding, ExpansionBranch::contracting}) {
        double const sign = branch == ExpansionBranch::expanding ? 1.0 : -1.0;
        double const physicalScaleFactor =
            std::sqrt(kInitialScaleFactor * kInitialScaleFactor + 2.0 * sign * kHubble * kElapsed);
        expectState(
            cosmology.evolve(kInitialScaleFactor, kElapsed, CosmologicalTime::physical, branch),
            physicalScaleFactor, sign * kHubble / physicalScaleFactor, sign * kHubble);
        double const conformalScaleFactor = kInitialScaleFactor + sign * kHubble * kElapsed;
        expectState(
            cosmology.evolve(kInitialScaleFactor, kElapsed, CosmologicalTime::conformal, branch),
            conformalScaleFactor, sign * kHubble / conformalScaleFactor, sign * kHubble);
    }
}

TEST(Friedmann, PureMatterMatchesBothAnalyticClocksAndBranches) {
    constexpr double kHubble = 1.7;
    constexpr double kInitialScaleFactor = 0.8;
    constexpr double kElapsed = 0.08;

    FriedmannCosmology const cosmology(parameters(0.0, 1.0, 0.0, 0.0, kHubble));
    for (auto const branch : {ExpansionBranch::expanding, ExpansionBranch::contracting}) {
        double const sign = branch == ExpansionBranch::expanding ? 1.0 : -1.0;
        double const physicalScaleFactor = std::pow(
            std::pow(kInitialScaleFactor, 1.5) + 1.5 * sign * kHubble * kElapsed, 2.0 / 3.0);
        expectState(
            cosmology.evolve(kInitialScaleFactor, kElapsed, CosmologicalTime::physical, branch),
            physicalScaleFactor, sign * kHubble / std::sqrt(physicalScaleFactor),
            sign * kHubble * std::sqrt(physicalScaleFactor));
        double const conformalRoot =
            std::sqrt(kInitialScaleFactor) + 0.5 * sign * kHubble * kElapsed;
        double const conformalScaleFactor = conformalRoot * conformalRoot;
        expectState(
            cosmology.evolve(kInitialScaleFactor, kElapsed, CosmologicalTime::conformal, branch),
            conformalScaleFactor, sign * kHubble / conformalRoot, sign * kHubble * conformalRoot);
    }
}

TEST(Friedmann, PureCurvatureMatchesBothAnalyticClocksAndBranches) {
    constexpr double kHubble = 1.7;
    constexpr double kInitialScaleFactor = 0.8;
    constexpr double kElapsed = 0.08;

    FriedmannCosmology const cosmology(parameters(0.0, 0.0, 0.0, 1.0, kHubble));
    for (auto const branch : {ExpansionBranch::expanding, ExpansionBranch::contracting}) {
        double const sign = branch == ExpansionBranch::expanding ? 1.0 : -1.0;
        double const physicalScaleFactor = kInitialScaleFactor + sign * kHubble * kElapsed;
        expectState(
            cosmology.evolve(kInitialScaleFactor, kElapsed, CosmologicalTime::physical, branch),
            physicalScaleFactor, sign * kHubble, sign * kHubble * physicalScaleFactor);
        double const conformalScaleFactor =
            kInitialScaleFactor * std::exp(sign * kHubble * kElapsed);
        expectState(
            cosmology.evolve(kInitialScaleFactor, kElapsed, CosmologicalTime::conformal, branch),
            conformalScaleFactor, sign * kHubble, sign * kHubble * conformalScaleFactor);
    }
}

TEST(Friedmann, PureDarkEnergyMatchesBothAnalyticClocksAndBranches) {
    constexpr double kHubble = 1.7;
    constexpr double kInitialScaleFactor = 0.8;
    constexpr double kElapsed = 0.08;

    FriedmannCosmology const cosmology(parameters(0.0, 0.0, 1.0, 0.0, kHubble));
    for (auto const branch : {ExpansionBranch::expanding, ExpansionBranch::contracting}) {
        double const sign = branch == ExpansionBranch::expanding ? 1.0 : -1.0;
        double const physicalScaleFactor =
            kInitialScaleFactor * std::exp(sign * kHubble * kElapsed);
        expectState(
            cosmology.evolve(kInitialScaleFactor, kElapsed, CosmologicalTime::physical, branch),
            physicalScaleFactor, sign * kHubble * physicalScaleFactor,
            sign * kHubble * physicalScaleFactor * physicalScaleFactor);
        double const conformalScaleFactor =
            kInitialScaleFactor / (1.0 - sign * kHubble * kInitialScaleFactor * kElapsed);
        expectState(
            cosmology.evolve(kInitialScaleFactor, kElapsed, CosmologicalTime::conformal, branch),
            conformalScaleFactor, sign * kHubble * conformalScaleFactor,
            sign * kHubble * conformalScaleFactor * conformalScaleFactor);
    }
}

TEST(Friedmann, MatterAndRadiationMatchExactConformalSolutionInBothDirections) {
    constexpr double kRadiation = 0.4;
    constexpr double kMatter = 0.6;
    constexpr double kHubble = 2.1;
    constexpr double kInitialScaleFactor = 0.3;

    FriedmannCosmology const cosmology(parameters(kRadiation, kMatter, 0.0, 0.0, kHubble));
    for (double const elapsed : {-0.1, 0.4}) {
        double const scaledElapsed = kHubble * elapsed;
        double const initialVelocity = std::sqrt(kRadiation + kMatter * kInitialScaleFactor);
        double const expectedScaleFactor = kInitialScaleFactor + initialVelocity * scaledElapsed +
            0.25 * kMatter * scaledElapsed * scaledElapsed;
        double const expectedConformalDerivative =
            kHubble * (initialVelocity + 0.5 * kMatter * scaledElapsed);
        expectState(cosmology.evolve(kInitialScaleFactor, elapsed, CosmologicalTime::conformal),
            expectedScaleFactor, expectedConformalDerivative / expectedScaleFactor,
            expectedConformalDerivative);
    }
}

TEST(Friedmann, FlatMatterAndDarkEnergyMatchExactPhysicalSolution) {
    constexpr double kMatter = 0.3;
    constexpr double kDarkEnergy = 0.7;
    constexpr double kHubble = 0.8;
    constexpr double kInitialScaleFactor = 0.2;

    FriedmannCosmology const cosmology(parameters(0.0, kMatter, kDarkEnergy, 0.0, kHubble));
    double const initialPhase =
        std::asinh(std::sqrt(kDarkEnergy / kMatter) * std::pow(kInitialScaleFactor, 1.5));
    for (double const elapsed : {-0.05, 1.4}) {
        double const phase = initialPhase + 1.5 * kHubble * std::sqrt(kDarkEnergy) * elapsed;
        double const expectedScaleFactor =
            std::pow(std::sqrt(kMatter / kDarkEnergy) * std::sinh(phase), 2.0 / 3.0);
        double const expectedPhysicalDerivative = kHubble *
            std::sqrt(kMatter / expectedScaleFactor +
                kDarkEnergy * expectedScaleFactor * expectedScaleFactor);
        expectState(cosmology.evolve(kInitialScaleFactor, elapsed), expectedScaleFactor,
            expectedPhysicalDerivative, expectedScaleFactor * expectedPhysicalDerivative);
    }
}

TEST(Friedmann, ClosedMatterPassesThroughTurnaroundAndIntoContractionWithBothClocks) {
    constexpr double kHubble = 1.3;
    constexpr double kPi = 3.141592653589793238462643383279502884;

    FriedmannCosmology const cosmology(parameters(0.0, 2.0, 0.0, -1.0, kHubble));
    // The cycloid is a = 1 - cos(theta), H0 t = theta - sin(theta), H0 eta = theta.
    // Starting theta = pi/2 gives a = 1 and an expanding derivative H0.
    auto const physicalTurnaround = cosmology.evolve(1.0, (0.5 * kPi + 1.0) / kHubble);
    auto const conformalTurnaround =
        cosmology.evolve(1.0, 0.5 * kPi / kHubble, CosmologicalTime::conformal);
    expectState(physicalTurnaround, 2.0, 0.0, 0.0);
    expectState(conformalTurnaround, 2.0, 0.0, 0.0);
    expectState(cosmology.evolve(1.0, (kPi + 2.0) / kHubble), 1.0, -kHubble, -kHubble);
    expectState(
        cosmology.evolve(1.0, kPi / kHubble, CosmologicalTime::conformal), 1.0, -kHubble, -kHubble);
    expectState(cosmology.advance(physicalTurnaround, (0.5 * kPi + 1.0) / kHubble), 1.0, -kHubble,
        -kHubble);
    expectState(
        cosmology.advance(conformalTurnaround, 0.5 * kPi / kHubble, CosmologicalTime::conformal),
        1.0, -kHubble, -kHubble);
    expectState(cosmology.evolve(2.0, (0.5 * kPi + 1.0) / kHubble), 1.0, -kHubble, -kHubble);
}

TEST(Friedmann, NegativeDarkEnergyAllowsRecollapse) {
    constexpr double kHubble = 0.7;

    FriedmannCosmology const cosmology(parameters(0.0, 2.0, -1.0, 0.0, kHubble));
    // a^(3/2) = sqrt(2) sin(theta), theta = theta0 + (3/2) H0 t.
    // theta0 = pi/4 and theta = 3pi/4 have a = 1 on opposite branches.
    expectState(cosmology.evolve(1.0, 3.141592653589793238462643383279502884 / (3.0 * kHubble)),
        1.0, -kHubble, -kHubble);
}

TEST(Friedmann, AllFourDensitiesPreserveTheConstraintAndDerivativeClockRelation) {
    auto const modelParameters = parameters(0.08, 0.41, 0.57, -0.06, 2.4);
    FriedmannCosmology const cosmology(modelParameters);
    for (auto const clock : {CosmologicalTime::physical, CosmologicalTime::conformal}) {
        for (auto const branch : {ExpansionBranch::expanding, ExpansionBranch::contracting}) {
            auto const initial = cosmology.state(0.8, branch);
            auto const final = cosmology.advance(initial, 0.07, clock);
            double const a = final.scaleFactor;
            double const expectedSquaredVelocity = modelParameters.hubbleConstant *
                modelParameters.hubbleConstant *
                (modelParameters.omegaRadiation / (a * a) + modelParameters.omegaMatter / a +
                    modelParameters.omegaCurvature + modelParameters.omegaDarkEnergy * a * a);
            EXPECT_NEAR(final.physicalDerivative * final.physicalDerivative,
                expectedSquaredVelocity, 2e-10 * expectedSquaredVelocity);
            EXPECT_NEAR(final.conformalDerivative, a * final.physicalDerivative,
                1e-14 * (1.0 + std::abs(final.conformalDerivative)));
            EXPECT_EQ(final.physicalDerivative > 0.0, branch == ExpansionBranch::expanding);
            expectSameState(cosmology.state(a, branch), final);
        }
    }
}

TEST(Friedmann, HubbleConstantOnlyRescalesTheTimeAndDerivatives) {
    auto const firstParameters = parameters(0.02, 0.35, 0.7, -0.07, 0.6);
    auto secondParameters = firstParameters;
    secondParameters.hubbleConstant *= 5.0;
    FriedmannCosmology const first(firstParameters);
    FriedmannCosmology const second(secondParameters);
    for (auto const clock : {CosmologicalTime::physical, CosmologicalTime::conformal}) {
        auto const firstState = first.evolve(0.4, 0.3, clock);
        auto const secondState = second.evolve(0.4, 0.3 / 5.0, clock);
        expectState(secondState, firstState.scaleFactor, 5.0 * firstState.physicalDerivative,
            5.0 * firstState.conformalDerivative);
    }
}

TEST(Friedmann, SplitStepsAndTimeReversalRetainTheFullState) {
    FriedmannCosmology const cosmology(parameters(0.05, 0.33, 0.64, -0.02, 1.2));
    for (auto const clock : {CosmologicalTime::physical, CosmologicalTime::conformal}) {
        for (auto const branch : {ExpansionBranch::expanding, ExpansionBranch::contracting}) {
            auto const initial = cosmology.state(0.9, branch);
            auto const first = cosmology.advance(initial, 0.04, clock);
            auto const split = cosmology.advance(first, 0.06, clock);
            auto const whole = cosmology.advance(initial, 0.1, clock);
            expectSameState(split, whole);
            expectSameState(cosmology.advance(whole, -0.1, clock), initial);
            auto const unchanged = cosmology.advance(initial, 0.0, clock);
            EXPECT_DOUBLE_EQ(unchanged.scaleFactor, initial.scaleFactor);
            EXPECT_DOUBLE_EQ(unchanged.physicalDerivative, initial.physicalDerivative);
            EXPECT_DOUBLE_EQ(unchanged.conformalDerivative, initial.conformalDerivative);
        }
    }
}

TEST(Friedmann, TightAbsoluteToleranceResolvesSmallRadiationScaleFactors) {
    constexpr double kInitialScaleFactor = 1e-7;
    constexpr double kElapsed = 0.25 * kInitialScaleFactor * kInitialScaleFactor;

    FriedmannOptions options;
    options.absoluteTolerance = 1e-24;
    FriedmannCosmology const cosmology(parameters(1.0, 0.0, 0.0, 0.0), options);
    double const expectedScaleFactor = std::sqrt(1.5) * kInitialScaleFactor;
    auto const result = cosmology.evolve(kInitialScaleFactor, kElapsed);
    EXPECT_NEAR(result.scaleFactor / expectedScaleFactor, 1.0, 1e-10);
    EXPECT_NEAR(result.physicalDerivative * expectedScaleFactor, 1.0, 1e-10);
    EXPECT_NEAR(result.conformalDerivative, 1.0, 1e-10);
}

TEST(Friedmann, DefaultTolerancesRetainRelativeAccuracyAtSmallScaleFactors) {
    constexpr double kInitialScaleFactor = 1e-12;

    FriedmannCosmology const radiation(parameters(1.0, 0.0, 0.0, 0.0));
    double const expectedScaleFactor = std::sqrt(1.5) * kInitialScaleFactor;
    auto const result =
        radiation.evolve(kInitialScaleFactor, 0.25 * kInitialScaleFactor * kInitialScaleFactor);
    EXPECT_NEAR(result.scaleFactor / expectedScaleFactor, 1.0, 1e-10);
    EXPECT_NEAR(result.physicalDerivative * expectedScaleFactor, 1.0, 1e-10);
}

TEST(Friedmann, ReportsUnrepresentableDerivativesWithoutReturningAFalseStaticState) {
    FriedmannCosmology const matter(parameters(0.0, 1.0, 0.0, 0.0, 1e-300));
    EXPECT_THROW(matter.state(1e-200), std::overflow_error);
}

TEST(Friedmann, RejectsInvalidParametersAndOptions) {
    auto modelParameters = parameters(0.0, 0.3, 0.7, 0.0);
    for (double const hubble : {0.0, -1.0}) {
        modelParameters.hubbleConstant = hubble;
        EXPECT_THROW((FriedmannCosmology(modelParameters)), std::domain_error);
    }
    modelParameters = parameters(-0.1, 0.3, 0.8, 0.0);
    EXPECT_THROW((FriedmannCosmology(modelParameters)), std::domain_error);
    modelParameters = parameters(0.1, -0.3, 1.2, 0.0);
    EXPECT_THROW((FriedmannCosmology(modelParameters)), std::domain_error);
    modelParameters = parameters(0.1, 0.3, 0.7, 0.0);
    EXPECT_THROW((FriedmannCosmology(modelParameters)), std::domain_error);

    for (double const invalidTolerance : {-1.0, 1.0}) {
        FriedmannOptions options;
        options.relativeTolerance = invalidTolerance;
        EXPECT_THROW(
            (FriedmannCosmology(BackgroundCosmologyParameters{}, options)), std::domain_error);
        options = FriedmannOptions{};
        options.absoluteTolerance = invalidTolerance;
        EXPECT_THROW(
            (FriedmannCosmology(BackgroundCosmologyParameters{}, options)), std::domain_error);
    }
    FriedmannOptions options;
    options.relativeTolerance = 0.0;
    options.absoluteTolerance = 0.0;
    EXPECT_THROW((FriedmannCosmology(BackgroundCosmologyParameters{}, options)), std::domain_error);
    options = FriedmannOptions{};
    options.maxSteps = 0;
    EXPECT_THROW((FriedmannCosmology(BackgroundCosmologyParameters{}, options)), std::domain_error);
}

TEST(Friedmann, RejectsInvalidScaleFactorsCoordinatesAndInconsistentStates) {
    FriedmannCosmology const cosmology;
    for (double const scaleFactor : {0.0, -1.0}) {
        EXPECT_THROW(cosmology.state(scaleFactor), std::domain_error);
        EXPECT_THROW(cosmology.evolve(scaleFactor, 0.1), std::domain_error);
    }
    EXPECT_THROW(cosmology.state(1.0, static_cast<ExpansionBranch>(-1)), std::domain_error);
    EXPECT_THROW(cosmology.evolve(1.0, 0.1, static_cast<CosmologicalTime>(-1)), std::domain_error);
    auto invalidState = cosmology.state(0.5);
    invalidState.scaleFactor = 0.0;
    EXPECT_THROW(cosmology.advance(invalidState, 0.1), std::domain_error);
    invalidState = cosmology.state(0.5);
    invalidState.physicalDerivative *= 2.0;
    EXPECT_THROW(cosmology.advance(invalidState, 0.1), std::domain_error);
    invalidState = cosmology.state(0.5);
    invalidState.conformalDerivative *= -1.0;
    EXPECT_THROW(cosmology.advance(invalidState, 0.1), std::domain_error);
}

TEST(Friedmann, ReportsSingularTimeBoundariesAndStepExhaustion) {
    FriedmannCosmology const closed(parameters(0.0, 2.0, 0.0, -1.0));
    EXPECT_ANY_THROW(closed.state(2.1));
    FriedmannCosmology const radiation(parameters(1.0, 0.0, 0.0, 0.0));
    EXPECT_ANY_THROW(radiation.evolve(1.0, -0.6));
    EXPECT_ANY_THROW(
        radiation.evolve(1.0, 0.6, CosmologicalTime::physical, ExpansionBranch::contracting));
    EXPECT_ANY_THROW(
        radiation.evolve(1.0, 1.1, CosmologicalTime::conformal, ExpansionBranch::contracting));
    FriedmannCosmology const darkEnergy(parameters(0.0, 0.0, 1.0, 0.0));
    EXPECT_ANY_THROW(darkEnergy.evolve(1.0, 1.1, CosmologicalTime::conformal));
    FriedmannOptions options;
    options.maxSteps = 1;
    FriedmannCosmology const limited(parameters(0.0, 1.0, 0.0, 0.0), options);
    EXPECT_ANY_THROW(limited.evolve(1.0, 100.0));
}

}    // namespace detail

namespace detail {

TEST(Friedmann, ExactQuadraturesForEachComponentAndReverseIntervals) {
    double const initial = 0.3;
    double const final = 0.8;
    double const hubble = 1.7;
    for (size_t component = 0; component < 4; component++) {
        std::array<double, 4> densities{};
        densities[component] = 1.0;
        FriedmannCosmology const model(
            parameters(densities[0], densities[1], densities[2], densities[3], hubble));
        double physical;
        double conformal;
        double drift;
        if (component == 0) {
            physical = 0.5 * (final * final - initial * initial) / hubble;
            conformal = (final - initial) / hubble;
            drift = std::log(final / initial) / hubble;
        } else if (component == 1) {
            physical = 2.0 / (3.0 * hubble) * (std::pow(final, 1.5) - std::pow(initial, 1.5));
            conformal = 2.0 / hubble * (std::sqrt(final) - std::sqrt(initial));
            drift = 2.0 / hubble * (1.0 / std::sqrt(initial) - 1.0 / std::sqrt(final));
        } else if (component == 2) {
            physical = std::log(final / initial) / hubble;
            conformal = (1.0 / initial - 1.0 / final) / hubble;
            drift = 0.5 / hubble * (1.0 / (initial * initial) - 1.0 / (final * final));
        } else {
            physical = (final - initial) / hubble;
            conformal = std::log(final / initial) / hubble;
            drift = (1.0 / initial - 1.0 / final) / hubble;
        }
        auto const interval = model.intervalsBetweenScales(initial, final);
        EXPECT_NEAR(interval.physicalTime, physical, 1.e-12);
        EXPECT_NEAR(interval.conformalTime, conformal, 1.e-12);
        EXPECT_NEAR(interval.driftIntegral, drift, 1.e-12);
        auto const reverse = model.intervalsBetweenScales(final, initial);
        EXPECT_DOUBLE_EQ(reverse.physicalTime, -interval.physicalTime);
        EXPECT_DOUBLE_EQ(reverse.conformalTime, -interval.conformalTime);
        EXPECT_DOUBLE_EQ(reverse.driftIntegral, -interval.driftIntegral);
        for (auto const clock : {CosmologicalTime::physical, CosmologicalTime::conformal}) {
            auto const evolution = model.advanceWithIntegrals(model.state(initial),
                clock == CosmologicalTime::physical ? physical : conformal, clock);
            EXPECT_NEAR(evolution.state.scaleFactor, final, 1.e-11);
            EXPECT_NEAR(evolution.physicalTime, physical, 1.e-11);
            EXPECT_NEAR(evolution.conformalTime, conformal, 1.e-11);
            EXPECT_NEAR(evolution.driftIntegral, drift, 1.e-10);
        }
    }
}

TEST(Friedmann, TurnaroundQuadraturesDenseSamplesAndSignedEvolution) {
    double const pi = std::acos(-1.0);
    FriedmannCosmology const closed(parameters(0.0, 2.0, 0.0, -1.0));
    auto const turnaround = closed.intervalsBetweenScales(1.0, 2.0);
    EXPECT_NEAR(turnaround.conformalTime, 0.5 * pi, 1.e-12);
    EXPECT_NEAR(turnaround.physicalTime, 0.5 * pi + 1.0, 1.e-12);
    EXPECT_NEAR(turnaround.driftIntegral, 1.0, 1.e-12);
    EXPECT_ANY_THROW(closed.intervalsBetweenScales(1.0, 2.1));
    auto const trajectory = closed.trajectory(closed.state(1.0), pi);
    EXPECT_GT(trajectory.final().flops, 0);
    for (int i = 0; i <= 100; i++) {
        double const time = pi * i / 100.0;
        auto const sample = trajectory.sample(time);
        double const phase = 0.5 * pi + time;
        double const expectedScale = 1.0 - std::cos(phase);
        expectState(
            sample.state, expectedScale, std::sin(phase) / expectedScale, std::sin(phase), 1.e-10);
        EXPECT_DOUBLE_EQ(sample.conformalTime, time);
        EXPECT_NEAR(sample.physicalTime, time - std::sin(phase) + 1.0, 2.e-10);
        EXPECT_NEAR(sample.driftIntegral, 1.0 - 1.0 / std::tan(phase / 2.0), 2.e-10);
    }
    EXPECT_ANY_THROW(trajectory.sample(-0.1));
    EXPECT_ANY_THROW(trajectory.sample(pi + 0.1));
    auto const backwards = closed.trajectory(closed.state(1.0, ExpansionBranch::contracting), -pi);
    auto const middle = backwards.sample(-0.5 * pi);
    expectState(middle.state, 2.0, 0.0, 0.0);
    EXPECT_NEAR(backwards.final().physicalTime, -pi - 2.0, 2.e-10);
    EXPECT_NEAR(backwards.final().driftIntegral, -2.0, 2.e-10);
}

TEST(Friedmann, ScaleQuadratureRejectsInfiniteTimeRootsAndForbiddenInterior) {
    // Closure normalization of P=(a-2)^2*a*(a+4)/5 has a double root a=2.
    FriedmannCosmology const doubleRoot(parameters(0.0, 16.0 / 5.0, 1.0 / 5.0, -12.0 / 5.0));
    EXPECT_ANY_THROW(doubleRoot.intervalsBetweenScales(1.0, 2.0));
    EXPECT_ANY_THROW(doubleRoot.intervalsBetweenScales(1.0, 3.0));
    FriedmannCosmology const forbidden(parameters(0.0, 3.0, 0.1, -2.1));
    EXPECT_ANY_THROW(forbidden.intervalsBetweenScales(1.0, 5.0));
}

TEST(Friedmann, AdaptiveWorkAndAccuracyIncreaseWithRequestedPrecision) {
    auto const modelParameters = parameters(0.03, 0.37, 0.7, -0.1);
    FriedmannOptions looseOptions;
    looseOptions.relativeTolerance = 1.e-6;
    looseOptions.absoluteTolerance = 1.e-8;
    FriedmannCosmology const loose(modelParameters, looseOptions);
    FriedmannCosmology const tight(modelParameters);
    auto const looseResult = loose.advanceWithIntegrals(loose.state(0.02), 1.0);
    auto const tightResult = tight.advanceWithIntegrals(tight.state(0.02), 1.0);
    EXPECT_GT(tightResult.flops, looseResult.flops);
    EXPECT_NEAR(looseResult.state.scaleFactor, tightResult.state.scaleFactor, 1.e-5);
    auto const reverse = tight.advanceWithIntegrals(tightResult.state, -1.0);
    EXPECT_NEAR(reverse.state.scaleFactor, 0.02, 1.e-11);
    EXPECT_NEAR(reverse.physicalTime, -tightResult.physicalTime, 1.e-11);
    EXPECT_NEAR(reverse.driftIntegral, -tightResult.driftIntegral, 1.e-9);
}

TEST(Friedmann, NonfiniteInputsAreRejectedInReleaseBuilds) {
    FriedmannCosmology const model;
    EXPECT_THROW(model.state(std::numeric_limits<double>::quiet_NaN()), std::domain_error);
    EXPECT_THROW(model.advance(model.state(1.0), std::numeric_limits<double>::infinity()),
        std::domain_error);
    auto modelParameters = parameters(0.0, 0.3, 0.7, 0.0);
    modelParameters.omegaDarkEnergy = std::numeric_limits<double>::infinity();
    EXPECT_THROW(FriedmannCosmology{modelParameters}, std::domain_error);
}

}    // namespace detail

namespace detail {

TEST(Friedmann, RadiationMatterIntervalsSpanTwelveOrdersWithoutCancellation) {
    constexpr double kRadiation = 9.2e-5;
    constexpr double kMatter = 1.0 - kRadiation;
    constexpr double kInitial = 1.e-12;
    constexpr double kFinal = 1.0;
    FriedmannCosmology const model(parameters(kRadiation, kMatter, 0.0, 0.0));
    double const initialRoot = std::sqrt(kRadiation + kMatter * kInitial);
    double const finalRoot = std::sqrt(kRadiation + kMatter * kFinal);
    double const radiationRoot = std::sqrt(kRadiation);
    double const conformal = 2.0 / kMatter * (finalRoot - initialRoot);
    double const physical = 2.0 / (3.0 * kMatter * kMatter) *
        (finalRoot * finalRoot * finalRoot - initialRoot * initialRoot * initialRoot -
            3.0 * kRadiation * (finalRoot - initialRoot));
    double const drift =
        (std::log(kFinal / kInitial) +
            2.0 * std::log((initialRoot + radiationRoot) / (finalRoot + radiationRoot))) /
        radiationRoot;
    auto const interval = model.intervalsBetweenScales(kInitial, kFinal);
    EXPECT_NEAR(interval.physicalTime, physical, 1.e-12);
    EXPECT_NEAR(interval.conformalTime, conformal, 1.e-12);
    EXPECT_NEAR(interval.driftIntegral, drift, 1.e-9);
}

TEST(Friedmann, RealisticLambdaAgeIntervalsComposeAcrossRadiationEra) {
    constexpr double kRadiation = 9.2093902908970532e-5;
    constexpr double kMatter = 0.31346582253960142;
    FriedmannCosmology const model(
        parameters(kRadiation, kMatter, 1.0 - kMatter - kRadiation, 0.0));
    auto const whole = model.intervalsBetweenScales(1.e-12, 1.0);
    auto const early = model.intervalsBetweenScales(1.e-12, 0.02);
    auto const late = model.intervalsBetweenScales(0.02, 1.0);
    EXPECT_NEAR(whole.physicalTime, early.physicalTime + late.physicalTime, 1.e-12);
    EXPECT_NEAR(whole.conformalTime, early.conformalTime + late.conformalTime, 1.e-12);
    EXPECT_NEAR(whole.driftIntegral, early.driftIntegral + late.driftIntegral, 1.e-9);
}

}    // namespace detail

namespace detail {

TEST(Friedmann, RadiationAndMatterRecollapseMatchesAnalyticNearCrunchAndSplitSteps) {
    constexpr double kRadiation = 9.2093902908970532e-5;
    constexpr double kInitial = 0.02;
    constexpr double kFinal = 1.e-6;
    double const curvatureMagnitude = 1.0 + kRadiation;
    double const center = 1.0 / curvatureMagnitude;
    double const radius = std::sqrt(center * center + kRadiation / curvatureMagnitude);
    auto const phase = [&](double scale) {
        return std::atan2(
            std::sqrt((kRadiation + 2.0 * scale - curvatureMagnitude * scale * scale) /
                curvatureMagnitude),
            center - scale);
    };
    double const initialPhase = phase(kInitial);
    double const finalPhase = 2.0 * std::acos(-1.0) - phase(kFinal);
    double const elapsed = (finalPhase - initialPhase) / std::sqrt(curvatureMagnitude);
    double const physical = (center * (finalPhase - initialPhase) -
                                radius * (std::sin(finalPhase) - std::sin(initialPhase))) /
        std::sqrt(curvatureMagnitude);
    FriedmannCosmology const model(parameters(kRadiation, 2.0, 0.0, -curvatureMagnitude));
    auto const whole = model.advanceWithIntegrals(model.state(kInitial), elapsed);
    EXPECT_NEAR(whole.state.scaleFactor, kFinal, 2.e-12);
    EXPECT_LT(whole.state.physicalDerivative, 0.0);
    EXPECT_NEAR(whole.physicalTime, physical, 2.e-11);
    auto split = model.state(kInitial);
    double splitPhysical = 0.0;
    double previous = 0.0;
    for (size_t step = 1; step <= 128; step++) {
        double const position = elapsed * double(step) / 128.0;
        auto const result = model.advanceWithIntegrals(split, position - previous);
        split = result.state;
        splitPhysical += result.physicalTime;
        previous = position;
    }
    EXPECT_NEAR(split.scaleFactor, kFinal, 2.e-12);
    EXPECT_LT(split.physicalDerivative, 0.0);
    EXPECT_NEAR(splitPhysical, physical, 2.e-11);
    auto const history = model.trajectory(model.state(kInitial), elapsed);
    auto const beforeEnd = history.sample(std::nextafter(elapsed, 0.0));
    EXPECT_NEAR(
        beforeEnd.state.conformalDerivative, history.final().state.conformalDerivative, 1.e-11);
    EXPECT_NEAR(history.final().state.scaleFactor, kFinal, 2.e-12);
}

TEST(Friedmann, NegativeLambdaNearCrunchRetainsConstraintAndAnalyticPhysicalClock) {
    constexpr double kFinal = 1.e-6;
    FriedmannCosmology const model(parameters(0.0, 2.0, -1.0, 0.0));
    double const maximum = std::cbrt(2.0);
    double const elapsed = model.intervalsBetweenScales(1.0, maximum).conformalTime +
        model.intervalsBetweenScales(kFinal, maximum).conformalTime;
    auto const result = model.advanceWithIntegrals(model.state(1.0), elapsed);
    double const expectedPhysical =
        (std::acos(-1.0) - std::asin(std::pow(kFinal, 1.5) / std::sqrt(2.0)) -
            std::acos(-1.0) / 4.0) /
        1.5;
    EXPECT_NEAR(result.state.scaleFactor, kFinal, 2.e-11);
    EXPECT_LT(result.state.physicalDerivative, 0.0);
    EXPECT_NEAR(result.physicalTime, expectedPhysical, 2.e-11);
    double const a = result.state.scaleFactor;
    double const expectedConstraint = 2.0 * a - a * a * a * a;
    EXPECT_NEAR(result.state.conformalDerivative * result.state.conformalDerivative,
        expectedConstraint, 1.e-10 * expectedConstraint);
}

}    // namespace detail

namespace detail {

TEST(Friedmann, BothCurvatureSignsWithLambdaReachTheFutureOnTheExpandingBranch) {
    constexpr double kRadiation = 9.2093902908970532e-5;
    constexpr double kMatter = 0.31346582253960142;
    for (double const curvature : {-0.1, 0.1}) {
        double const lambda = 1.0 - kMatter - kRadiation - curvature;
        FriedmannCosmology const model(parameters(kRadiation, kMatter, lambda, curvature));
        auto const interval = model.intervalsBetweenScales(0.02, 10.0);
        for (auto const clock : {CosmologicalTime::physical, CosmologicalTime::conformal}) {
            auto const result = model.advanceWithIntegrals(model.state(0.02),
                clock == CosmologicalTime::physical ? interval.physicalTime :
                                                      interval.conformalTime,
                clock);
            EXPECT_NEAR(result.state.scaleFactor, 10.0, 2.e-9);
            double const expectedAdot =
                std::sqrt(kRadiation / 100.0 + kMatter / 10.0 + curvature + lambda * 100.0);
            EXPECT_NEAR(result.state.physicalDerivative, expectedAdot, 2.e-9);
            EXPECT_GT(result.state.physicalDerivative, 0.0);
        }
    }
}

TEST(Friedmann, CriticalRadiationMatterExpansionSlowsTowardZeroPhysicalDerivative) {
    constexpr double kRadiation = 9.2093902908970532e-5;
    constexpr double kMatter = 1.0 - kRadiation;
    FriedmannCosmology const model(parameters(kRadiation, kMatter, 0.0, 0.0));
    double previousDerivative = model.state(1.0).physicalDerivative;
    for (double const finalScale : {10.0, 100.0, 1.e6}) {
        // With k=Lambda=0, a(eta) is exactly quadratic even when radiation is
        // present. Physical adot falls as sqrt(Omega_m/a), whereas da/deta grows.
        double const elapsed = 2.0 / kMatter * (std::sqrt(kRadiation + kMatter * finalScale) - 1.0);
        auto const state = model.evolve(1.0, elapsed, CosmologicalTime::conformal);
        double const expected = std::sqrt(kRadiation + kMatter * finalScale) / finalScale;
        EXPECT_NEAR(state.scaleFactor / finalScale, 1.0, 2.e-11);
        EXPECT_NEAR(state.physicalDerivative / expected, 1.0, 2.e-11);
        EXPECT_GT(state.physicalDerivative, 0.0);
        EXPECT_LT(state.physicalDerivative, previousDerivative);
        previousDerivative = state.physicalDerivative;
    }
    EXPECT_LT(previousDerivative, 0.001001);
    constexpr double kFinalScale = 1.e6;
    double const elapsed = 2.0 / kMatter * (std::sqrt(kRadiation + kMatter * kFinalScale) - 1.0);
    auto const history = model.trajectory(model.state(1.0), elapsed);
    auto split = model.state(1.0);
    double previousTime = 0.0;
    for (size_t step = 1; step <= 64; step++) {
        double const position = elapsed * double(step) / 64.0;
        double const exactScale = 1.0 + position + 0.25 * kMatter * position * position;
        double const exactDerivative = (1.0 + 0.5 * kMatter * position) / exactScale;
        auto const dense = history.sample(position).state;
        split = model.advance(split, position - previousTime, CosmologicalTime::conformal);
        EXPECT_NEAR(dense.scaleFactor / exactScale, 1.0, 2.e-11);
        EXPECT_NEAR(dense.physicalDerivative / exactDerivative, 1.0, 2.e-11);
        EXPECT_NEAR(split.scaleFactor / exactScale, 1.0, 2.e-11);
        EXPECT_NEAR(split.physicalDerivative / exactDerivative, 1.0, 2.e-11);
        previousTime = position;
    }
}

TEST(Friedmann, SubcriticalWithoutLambdaApproachesNonzeroPhysicalExpansionSpeed) {
    constexpr double kRadiation = 9.2093902908970532e-5;
    constexpr double kMatter = 0.31346582253960142;
    constexpr double kCurvature = 1.0 - kMatter - kRadiation;
    FriedmannCosmology const model(parameters(kRadiation, kMatter, 0.0, kCurvature));
    double const frequency = std::sqrt(kCurvature);
    double const center = kMatter / (2.0 * kCurvature);
    double previousDerivative = 1.0;
    for (double const elapsed : {2.0, 10.0, 20.0}) {
        double const phase = frequency * elapsed;
        double const scale =
            (1.0 + center) * std::cosh(phase) + std::sinh(phase) / frequency - center;
        double const conformalDerivative =
            frequency * (1.0 + center) * std::sinh(phase) + std::cosh(phase);
        auto const state = model.evolve(1.0, elapsed, CosmologicalTime::conformal);
        EXPECT_NEAR(state.scaleFactor / scale, 1.0, 3.e-11);
        EXPECT_NEAR(state.physicalDerivative, conformalDerivative / scale, 3.e-11);
        EXPECT_GT(state.physicalDerivative, frequency);
        EXPECT_LT(state.physicalDerivative, previousDerivative);
        previousDerivative = state.physicalDerivative;
    }
    EXPECT_NEAR(previousDerivative, frequency, 2.e-8);
}

TEST(Friedmann, LightconeObservationMustPrecedeTheFutureConformalBoundary) {
    // Rockstar's catalogue-unit fixture uses about 100 for H0 in code units.
    // A radius of 0.02 would place the observer beyond the finite future
    // conformal horizon, even though its emission event at a=0.5 is valid.
    constexpr double kHubble = 100.0;
    FriedmannCosmology const model(parameters(0.0, 0.3, 0.7, 0.0, kHubble));
    double const emission = model.intervalsBetweenScales(0.1, 0.5).conformalTime;
    auto const initial = model.state(0.1);
    auto const history = model.trajectory(initial, emission + 0.002);
    EXPECT_NEAR(history.sample(emission).state.scaleFactor, 0.5, 2.e-10);
    EXPECT_GT(history.final().state.scaleFactor, 0.5);
    EXPECT_THROW(model.trajectory(initial, emission + 0.02), std::domain_error);
}

}    // namespace detail
