#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <cosmictiger/stars/star_formation_kernel.hpp>

namespace {
namespace sh = springel_and_hernquist;
namespace sf = star_formation_backend;

void relativeNear(double actual, double expected, double tolerance = 2.0e-12) {
    ASSERT_TRUE(std::isfinite(actual));
    EXPECT_LE(std::abs(actual - expected), tolerance * std::max(std::abs(expected), 1.e-100));
}
sf::Parameters parameters() {
    sf::Parameters result;
    result.model = {sh::defaultParameters(), 1.e-24, 0.0};
    result.cooling = gadget4::makeParameters(1.0, false, 0.0);
    result.densityScale = result.energyScale = result.scaleFactor = result.timeSeconds = 1.0;
    result.gasMass = 2.0;
    result.explicitTimestep = 1.e11;
    result.seed = 42;
    return result;
}
sf::Input input(sf::Parameters const& p, double factor = 10.0) {
    sf::Input result;
    result.state.id = 1;
    result.density = float(factor * p.model.thresholdDensity);
    result.rung = 3;
    result.electronFraction = 1.0f;
    double electrons = result.electronFraction;
    sh::Equilibrium phase;
    EXPECT_TRUE(sh::evaluateEquilibrium(p.model, p.cooling, result.density, electrons, phase));
    result.energy = float(0.5 * phase.effectiveEnergy);
    return result;
}

TEST(SpringelAndHernquist, OfficialCalibrationUsesApproximateHotEnergyAndIonizedWarmGas) {
    auto const p = sh::defaultParameters();
    double const coefficient = 1.e25;
    sh::Model model;
    ASSERT_TRUE(sh::calibrate(p, coefficient, model));
    double const hot = p.supernovaEnergy / p.evaporationFactor;
    double const x = (hot - p.warmEnergy) / (hot - p.coldEnergy);
    double const expected = x / std::pow(1.0 - x, 2) *
        (p.supernovaMassFraction * p.supernovaEnergy -
            (1.0 - p.supernovaMassFraction) * p.coldEnergy) /
        (p.starFormationTime * coefficient);
    relativeNear(model.thresholdDensity, expected);
    double const mu = 4.0 / (8.0 - 5.0 * (1.0 - 0.76));
    relativeNear(p.warmEnergy, 1.5 / mu * gadget4::kBoltzmann / gadget4::kProtonMass * 1.e4);
    EXPECT_NE(hot, sh::hotEnergy(model, model.thresholdDensity));
    sh::Model doubled;
    ASSERT_TRUE(sh::calibrate(p, 2.0 * coefficient, doubled));
    relativeNear(doubled.thresholdDensity, 0.5 * model.thresholdDensity);
    EXPECT_FALSE(sh::calibrate(p, 0.0, doubled));
}

TEST(SpringelAndHernquist, EquilibriumMatchesOfficialAlgebraAndClampOrder) {
    auto const p = parameters();
    for (double factor : {1.0, 10.0, 1000.0, 1.e6}) {
        double const rho = factor * p.model.thresholdDensity;
        double const time = p.model.parameters.starFormationTime / std::sqrt(factor);
        double const evaporation = p.model.parameters.evaporationFactor * std::pow(factor, -0.8);
        double const hot = p.model.parameters.supernovaEnergy / (1.0 + evaporation) +
            p.model.parameters.coldEnergy;
        double electrons = 1.0;
        double coolingTime;
        ASSERT_EQ(gadget4::coolingTime(hot, rho, p.cooling, electrons, coolingTime),
            gadget4::Status::Success);
        double const y = time / coolingTime * hot / sh::feedbackEnergy(p.model.parameters);
        double const x = 1.0 + 1.0 / (2.0 * y) - std::sqrt(1.0 / y + 1.0 / (4.0 * y * y));
        sh::Equilibrium phase;
        ASSERT_TRUE(sh::equilibrium(p.model, rho, coolingTime, phase));
        relativeNear(phase.coldFraction, x);
        relativeNear(phase.hotEnergy, hot);
        relativeNear(phase.effectiveEnergy, hot * (1.0 - x) + p.model.parameters.coldEnergy * x);
        ASSERT_TRUE(sh::clampSourceTime(p.model.parameters, 10.0 * time, phase));
        EXPECT_EQ(phase.coldFraction, x);
        relativeNear(phase.starFormationTime, 10.0 * time);
        relativeNear(phase.specificStarFormationRate,
            (1.0 - p.model.parameters.supernovaMassFraction) * x / (10.0 * time));
    }
}

TEST(SpringelAndHernquist, DenseRelaxationKeepsThermalHistoryAndHotElectronGuess) {
    auto const p = parameters();
    auto const old = input(p);
    double electrons = old.electronFraction;
    sh::Equilibrium phase;
    ASSERT_TRUE(sh::evaluateEquilibrium(p.model, p.cooling, old.density, electrons, phase));
    ASSERT_TRUE(sh::clampSourceTime(p.model.parameters, p.explicitTimestep, phase));
    double const energy = phase.effectiveEnergy +
        (double(old.energy) - phase.effectiveEnergy) *
            std::exp(-p.explicitTimestep / phase.relaxationTime);
    auto const result = sf::advance(old, p);
    ASSERT_EQ(result.status, 0);
    EXPECT_FALSE(result.convert);
    EXPECT_EQ(result.energy, float(energy));
    EXPECT_NE(result.energy, float(phase.effectiveEnergy));
    EXPECT_EQ(result.electronFraction, float(electrons));
    relativeNear(result.state.specificStarFormationRate, phase.specificStarFormationRate);
    relativeNear(result.value, -p.gasMass * (double(result.energy) - old.energy));
}

TEST(SpringelAndHernquist, DenseAboveTargetCoolsWithFloorAndTemperatureGate) {
    auto p = parameters();
    auto old = input(p);
    double hotElectrons = old.electronFraction;
    sh::Equilibrium phase;
    ASSERT_TRUE(sh::evaluateEquilibrium(p.model, p.cooling, old.density, hotElectrons, phase));
    old.energy = float(10.0 * phase.effectiveEnergy);
    for (double interval : {1.e6, 1.e18}) {
        p.explicitTimestep = interval;
        double electrons = hotElectrons;
        double cooled;
        ASSERT_EQ(
            gadget4::doCooling(old.energy, old.density, interval, p.cooling, electrons, cooled),
            gadget4::Status::Success);
        double const expected = std::max(cooled, phase.effectiveEnergy);
        auto const result = sf::advance(old, p);
        ASSERT_EQ(result.status, 0);
        EXPECT_EQ(result.energy, float(expected));
        EXPECT_EQ(result.electronFraction, float(hotElectrons));
        if (expected > 1.01 * phase.effectiveEnergy) {
            EXPECT_EQ(result.state.specificStarFormationRate, 0.0);
        } else {
            EXPECT_GT(result.state.specificStarFormationRate, 0.0);
        }
    }
}

TEST(SpringelAndHernquist, SavedRateIsSampledBeforeCoolingAndInitialRateIsZero) {
    auto p = parameters();
    auto old = input(p);
    p.explicitTimestep = 1.e18;
    auto const first = sf::advance(old, p);
    ASSERT_EQ(first.status, 0);
    EXPECT_FALSE(first.convert);
    EXPECT_GT(first.state.specificStarFormationRate, 0.0);
    old.state.specificStarFormationRate = 1.0;
    // A converted particle is not passed through cooling, even if its unused
    // gas thermochemistry is invalid: this distinguishes event-first ordering.
    old.electronFraction = -1.0f;
    old.density = -1.0f;
    auto const second = sf::advance(old, p);
    ASSERT_EQ(second.status, 0);
    EXPECT_TRUE(second.convert);
    EXPECT_EQ(second.energy, 0.0f);
    EXPECT_EQ(second.state.specificStarFormationRate, 0.0);
    relativeNear(second.value, p.gasMass * old.energy);
}

TEST(SpringelAndHernquist, PhysicalAndComovingDensityEligibilityAreIndependent) {
    auto p = parameters();
    auto old = input(p);
    p.cooling.comoving = true;
    p.model.overdensityThreshold = 2.0 * old.density;
    auto blocked = sf::diagnostics(old, p);
    ASSERT_EQ(blocked.status, 0);
    EXPECT_FALSE(blocked.starForming);
    p.cooling.comoving = false;
    auto eligible = sf::diagnostics(old, p);
    ASSERT_EQ(eligible.status, 0);
    EXPECT_TRUE(eligible.starForming);
    old.density = float(0.5 * p.model.thresholdDensity);
    auto diffuse = sf::advance(old, p);
    ASSERT_EQ(diffuse.status, 0);
    EXPECT_EQ(diffuse.state.specificStarFormationRate, 0.0);
}

TEST(SpringelAndHernquist, HydroUsesActualEnergyAndConstantGamma) {
    auto const p = parameters();
    auto old = input(p);
    auto const result = sf::evaluateEos(old, p);
    ASSERT_EQ(result.status, 0);
    EXPECT_EQ(result.pressureRatio, float(gadget4::kGammaMinusOne));
    EXPECT_EQ(result.signalIndex, float(gadget4::kGamma));
    old.energy *= 2.0f;
    auto const hotter = sf::evaluateEos(old, p);
    EXPECT_EQ(hotter.pressureRatio, result.pressureRatio);
    EXPECT_EQ(hotter.signalIndex, result.signalIndex);
}

TEST(SpringelAndHernquist, InvalidSourceLeavesInputUntouched) {
    auto const p = parameters();
    auto old = input(p);
    old.state.id = 0;
    auto const before = old;
    EXPECT_NE(sf::advance(old, p).status, 0);
    EXPECT_EQ(old.state, before.state);
    EXPECT_EQ(old.energy, before.energy);
    EXPECT_EQ(old.electronFraction, before.electronFraction);
}

TEST(SpringelAndHernquist, WholeParticleProbability) {
    sh::Equilibrium phase;
    phase.specificStarFormationRate = 0.25;
    relativeNear(
        sh::starFormationProbability(phase.specificStarFormationRate, 4.0), 1.0 - std::exp(-1.0));
    EXPECT_EQ(sh::starFormationProbability(phase.specificStarFormationRate, 0.0), 0.0);
    EXPECT_EQ(sh::starFormationProbability(phase.specificStarFormationRate, 1.e30), 1.0);
    EXPECT_TRUE(std::isnan(sh::starFormationProbability(phase.specificStarFormationRate, -1.0)));
}
}    // namespace
