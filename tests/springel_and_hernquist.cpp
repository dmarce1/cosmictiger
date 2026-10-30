#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <cosmictiger/chemistry/primordial_rates.hpp>
#include <cosmictiger/sph/springel_and_hernquist/cooling.hpp>

namespace {
namespace sh = springel_and_hernquist;

void relativeNear(double actual, double expected, double tolerance = 2.0e-12) {
    ASSERT_TRUE(std::isfinite(actual));
    EXPECT_LE(std::abs(actual - expected), tolerance * std::abs(expected));
}

TEST(SpringelAndHernquist, ThresholdContinuityAndCoolingBalance) {
    sh::AtomicCooling const cooling;
    sh::Parameters const parameters = sh::defaultParameters(cooling);
    sh::Model model;
    ASSERT_TRUE(sh::makeModel(parameters, cooling, model));
    EXPECT_GT(model.thresholdDensity, 0.0);
    sh::Equilibrium threshold;
    ASSERT_TRUE(sh::evaluateEquilibrium(model, cooling, model.thresholdDensity, threshold));
    relativeNear(threshold.effectiveEnergy, parameters.warmEnergy);
    relativeNear(threshold.pressure,
        (parameters.gamma - 1.0) * model.thresholdDensity * parameters.warmEnergy);
    relativeNear(threshold.hotEnergy,
        parameters.coldEnergy + parameters.supernovaEnergy / (1.0 + parameters.evaporationFactor));
    for (double overdensity : {1.0, 1.01, 10.0, 1.0e3, 1.0e6, 1.0e9}) {
        double const density = overdensity * model.thresholdDensity;
        sh::Equilibrium state;
        sh::AtomicEquilibrium hot;
        ASSERT_TRUE(sh::evaluateEquilibrium(model, cooling, density, state));
        ASSERT_TRUE(cooling.evaluateAtEnergy(state.hotEnergy, hot));
        double const radiativeLoss =
            std::pow(state.hotFraction * density, 2) * hot.coolingCoefficient;
        double const replenishment =
            state.coldFraction * density / state.starFormationTime * sh::feedbackEnergy(parameters);
        relativeNear(radiativeLoss, replenishment);
        relativeNear(state.coldFraction + state.hotFraction, 1.0);
        relativeNear(
            state.starFormationTime, parameters.starFormationTime / std::sqrt(overdensity));
        relativeNear(
            state.evaporationFactor, parameters.evaporationFactor * std::pow(overdensity, -0.8));
        relativeNear(state.specificStarFormationRate,
            (1.0 - parameters.beta) * state.coldFraction / state.starFormationTime);
        EXPECT_GT(state.relaxationTime, 0.0);
    }
    EXPECT_FALSE(sh::evaluateEquilibrium(model, cooling, 0.99 * model.thresholdDensity, threshold));
}

TEST(SpringelAndHernquist, ThresholdCalibrationRespondsToPhysics) {
    sh::Parameters parameters = sh::defaultParameters();
    sh::Model baseline;
    double const coefficient = 1.0e25;
    ASSERT_TRUE(sh::calibrate(parameters, coefficient, baseline));
    sh::Model changed;
    ASSERT_TRUE(sh::calibrate(parameters, 2.0 * coefficient, changed));
    relativeNear(changed.thresholdDensity, 0.5 * baseline.thresholdDensity);
    parameters.starFormationTime *= 4.0;
    ASSERT_TRUE(sh::calibrate(parameters, coefficient, changed));
    relativeNear(changed.thresholdDensity, 0.25 * baseline.thresholdDensity);
    parameters.warmEnergy = parameters.supernovaEnergy;
    EXPECT_FALSE(sh::calibrate(parameters, coefficient, changed));
    EXPECT_FALSE(sh::calibrate(sh::defaultParameters(), 0.0, changed));
    EXPECT_FALSE(sh::calibrate(sh::defaultParameters(), -coefficient, changed));
    EXPECT_FALSE(
        sh::calibrate(sh::defaultParameters(), std::numeric_limits<double>::quiet_NaN(), changed));
}

TEST(SpringelAndHernquist, PhaseFractionsRetainExtremeLimits) {
    EXPECT_EQ(sh::coldFraction(0.0), 0.0);
    EXPECT_EQ(sh::hotFraction(0.0), 1.0);
    for (double coolingRatio :
        {1.0e-300, 1.0e-30, 1.0e-9, 0.1, 1.0, 10.0, 1.0e9, 1.0e30, 1.0e300}) {
        double const cold = sh::coldFraction(coolingRatio);
        double const hot = sh::hotFraction(coolingRatio);
        EXPECT_GE(cold, 0.0);
        EXPECT_LE(cold, 1.0);
        EXPECT_GT(hot, 0.0);
        EXPECT_LE(hot, 1.0);
        relativeNear(cold + hot, 1.0);
        // Quadratic cloud balance independently checks the rationalized root.
        relativeNear(cold, coolingRatio * hot * hot);
    }
    relativeNear(sh::coldFraction(1.0e-20), 1.0e-20);
    relativeNear(sh::hotFraction(1.0e100), 1.0e-50);
    EXPECT_TRUE(std::isnan(sh::coldFraction(-1.0)));
    EXPECT_TRUE(std::isnan(sh::hotFraction(std::numeric_limits<double>::infinity())));
}

TEST(SpringelAndHernquist, ConsumptionProbabilityAndEnrichment) {
    sh::Equilibrium state;
    state.specificStarFormationRate = 0.25;
    EXPECT_EQ(sh::starFormationProbability(state, 0.0), 0.0);
    relativeNear(sh::starFormationProbability(state, 1.0e-20), 2.5e-21);
    relativeNear(sh::starFormationProbability(state, 4.0), 1.0 - std::exp(-1.0));
    EXPECT_EQ(sh::starFormationProbability(state, 1.0e30), 1.0);
    // Survival composes across arbitrary source partitions at fixed density.
    double const halfSurvival = 1.0 - sh::starFormationProbability(state, 2.0);
    relativeNear(halfSurvival * halfSurvival, 1.0 - sh::starFormationProbability(state, 4.0));
    double const yield = 0.02;
    double const timestep = 4.0;
    double const remainingFraction = std::exp(-state.specificStarFormationRate * timestep);
    relativeNear(sh::enrichmentIncrement(state, timestep, yield),
        -yield * std::log(remainingFraction));    // Closed-box Z=y*ln(M0/Mgas).
    EXPECT_TRUE(std::isnan(sh::starFormationProbability(state, -1.0)));
    EXPECT_TRUE(std::isnan(sh::enrichmentIncrement(state, timestep, -yield)));
}

TEST(SpringelAndHernquist, WindMassAndEnergyBudget) {
    sh::Parameters const parameters = sh::defaultParameters();
    sh::Equilibrium state;
    state.specificStarFormationRate = 1.0e-16;
    double const massLoading = 2.0;
    double const energyFraction = 0.25;
    double const speed = sh::windSpeed(parameters, massLoading, energyFraction);
    double const supernovaEnergyPerStellarMass =
        parameters.beta * parameters.supernovaEnergy / (1.0 - parameters.beta);
    relativeNear(0.5 * massLoading * speed * speed, energyFraction * supernovaEnergyPerStellarMass);
    double const time = 1.0e12;
    relativeNear(1.0 - sh::windProbability(state, time, massLoading),
        std::pow(1.0 - sh::starFormationProbability(state, time), massLoading));
    EXPECT_EQ(sh::windProbability(state, time, 0.0), 0.0);
    EXPECT_EQ(sh::windSpeed(parameters, massLoading, 0.0), 0.0);
    EXPECT_TRUE(std::isnan(sh::windSpeed(parameters, 0.0, energyFraction)));
    // Published SH03 does not constrain chi<=1 or subtract it thermally.
    relativeNear(sh::windSpeed(parameters, massLoading, 4.0 * energyFraction), 2.0 * speed);
}

TEST(SpringelAndHernquist, FiniteStepSamplingConvergesToYieldAndMassLoading) {
    sh::Equilibrium state;
    state.specificStarFormationRate = 1.0;
    double const yield = 0.02;
    double const massLoading = 2.0;
    double previousYieldError = 1.0;
    double previousWindError = 1.0;
    for (int refinement = 0; refinement != 9; ++refinement) {
        double const interval = std::ldexp(0.05 / (1.0 + massLoading), -refinement);
        double const stellarMass = sh::starFormationProbability(state, interval);
        // Ng=1 creates all metals before the star/wind draw. Winds are sampled
        // only if the gas survived the stellar draw: both are finite-step biases.
        double const metalMass = sh::enrichmentIncrement(state, interval, yield);
        double const windMass =
            (1.0 - stellarMass) * sh::windProbability(state, interval, massLoading);
        double const yieldError = metalMass / (yield * stellarMass) - 1.0;
        double const windError = 1.0 - windMass / (massLoading * stellarMass);
        EXPECT_GT(yieldError, 0.0);
        EXPECT_GT(windError, 0.0);
        EXPECT_LT(yieldError, 0.51 * previousYieldError);
        EXPECT_LT(windError, 0.51 * previousWindError);
        previousYieldError = yieldError;
        previousWindError = windError;
        if (refinement == 8) {
            // Independent small-interval expansions quantify first-order convergence.
            relativeNear(yieldError / interval, 0.5, 2.0e-5);
            relativeNear(windError / interval, 0.5 * (1.0 + massLoading), 1.0e-4);
        }
    }
}

TEST(SpringelAndHernquist, AtomicEquilibriumChargeAndFluxBalance) {
    sh::AtomicCooling const cooling;
    double const heliumRatio =
        (1.0 - cooling.hydrogenMassFraction) / (4.0 * cooling.hydrogenMassFraction);
    for (double temperature : {10.0, 1000.0, 1.0e4, 2.0e4, 1.0e5, 1.0e6, 1.0e8}) {
        sh::AtomicEquilibrium state;
        ASSERT_TRUE(cooling.evaluateAtTemperature(temperature, state));
        relativeNear(state.hydrogenNeutral + state.hydrogenIonized, 1.0);
        relativeNear(state.heliumNeutral + state.heliumIonized + state.heliumDoubleIonized, 1.0);
        relativeNear(state.electronsPerHydrogen,
            state.hydrogenIonized +
                heliumRatio * (state.heliumIonized + 2.0 * state.heliumDoubleIonized));
        primordial::RateCoefficients rates;
        ASSERT_EQ(primordial::evaluateRateCoefficients(
                      static_cast<float>(temperature), primordial::Network::Atomic6, rates),
            primordial::Status::Success);
        relativeNear(
            state.hydrogenNeutral * rates.collision[0], state.hydrogenIonized * rates.collision[1]);
        relativeNear(
            state.heliumNeutral * rates.collision[2], state.heliumIonized * rates.collision[3]);
        relativeNear(state.heliumIonized * rates.collision[4],
            state.heliumDoubleIonized * rates.collision[5]);
        EXPECT_GE(state.coolingCoefficient, 0.0);
        sh::AtomicEquilibrium recovered;
        ASSERT_TRUE(cooling.evaluateAtEnergy(state.specificEnergy, recovered));
        relativeNear(recovered.temperature, temperature, 3.0e-6);
        relativeNear(recovered.specificEnergy, state.specificEnergy, 3.0e-6);
    }
}

TEST(SpringelAndHernquist, AtomicCoolingNormalization) {
    // Independent double-precision H-only reference for Cen ionization,
    // Hui-Gnedin recombination, excitation and free-free at T=1e5 K.
    sh::AtomicCooling cooling;
    cooling.hydrogenMassFraction = 1.0;
    double const temperature = 1.0e5;
    double const root = std::sqrt(temperature);
    double const suppression = 1.0 + std::sqrt(temperature / 1.0e5);
    double const ionization = 5.85e-11 * root * std::exp(-157809.1 / temperature) / suppression;
    double const lambda = 315614.0 / temperature;
    double const recombination = 1.269e-13 * std::pow(lambda, 1.503) /
        std::pow(1.0 + std::pow(lambda / 0.522, 0.470), 1.923);
    double const ionized = ionization / (ionization + recombination);
    double const neutral = recombination / (ionization + recombination);
    double const recombinationCooling = 1.778e-29 * temperature * std::pow(lambda, 1.965) /
        std::pow(1.0 + std::pow(lambda / 0.541, 0.502), 2.697);
    double const excitation = 7.5e-19 * std::exp(-118348.0 / temperature) / suppression;
    double const gaunt = 1.1 + 0.34 * std::exp(-std::pow(5.5 - std::log10(temperature), 2) / 3.0);
    double const freeFree = 1.42e-27 * root * gaunt;
    double const loss = ionized *
        (neutral * (13.5984 * sh::kElectronVolt * ionization + excitation) +
            ionized * (recombinationCooling + freeFree));
    sh::AtomicEquilibrium state;
    ASSERT_TRUE(cooling.evaluateAtTemperature(temperature, state));
    relativeNear(state.coolingCoefficient, loss / std::pow(sh::kProtonMass, 2), 4.0e-6);
}

TEST(SpringelAndHernquist, ImplicitCoolingAndSupportedDomain) {
    sh::AtomicCooling const cooling;
    sh::Parameters const parameters = sh::defaultParameters(cooling);
    sh::AtomicEquilibrium initial;
    ASSERT_TRUE(cooling.evaluateAtTemperature(2.0e5, initial));
    double const density = 1.0e-24;
    for (double timestep : {1.0e7, 1.0e10, 1.0e13, 1.0e15}) {
        double cooled = -1.0;
        ASSERT_TRUE(
            cooling.cool(initial.specificEnergy, density, timestep, parameters.warmEnergy, cooled));
        EXPECT_GE(cooled, parameters.warmEnergy);
        EXPECT_LE(cooled, initial.specificEnergy);
        if (cooled > parameters.warmEnergy) {
            sh::AtomicEquilibrium final;
            ASSERT_TRUE(cooling.evaluateAtEnergy(cooled, final));
            double const residual =
                cooled - initial.specificEnergy + density * timestep * final.coolingCoefficient;
            EXPECT_LT(std::abs(residual), 5.0e-6 * initial.specificEnergy);
        }
    }
    double cooled = -1.0;
    ASSERT_TRUE(
        cooling.cool(initial.specificEnergy, density, 1.0e30, parameters.warmEnergy, cooled));
    EXPECT_EQ(cooled, parameters.warmEnergy);
    ASSERT_TRUE(cooling.cool(initial.specificEnergy, density, 0.0, parameters.warmEnergy, cooled));
    EXPECT_EQ(cooled, initial.specificEnergy);
    ASSERT_TRUE(
        cooling.cool(parameters.coldEnergy, density, 1.0e12, parameters.warmEnergy, cooled));
    EXPECT_EQ(cooled, parameters.coldEnergy);
    double const adiabaticallyCold = 0.01 * cooling.minimumEnergy();
    ASSERT_TRUE(cooling.cool(adiabaticallyCold, density, 1.0e12, parameters.warmEnergy, cooled));
    EXPECT_EQ(cooled, adiabaticallyCold);
    EXPECT_FALSE(cooling.cool(0.0, density, 1.0e12, parameters.warmEnergy, cooled));
    EXPECT_FALSE(cooling.evaluateAtTemperature(9.99, initial));
    EXPECT_FALSE(cooling.evaluateAtTemperature(1.001e8, initial));
    EXPECT_FALSE(cooling.evaluateAtEnergy(0.99 * cooling.minimumEnergy(), initial));
    EXPECT_FALSE(cooling.evaluateAtEnergy(1.01 * cooling.maximumEnergy(), initial));
    cooled = -1.0;
    EXPECT_FALSE(cooling.cool(
        1.01 * cooling.maximumEnergy(), density, 1.0e12, parameters.warmEnergy, cooled));
    EXPECT_EQ(cooled, -1.0);
    EXPECT_FALSE(
        cooling.cool(parameters.warmEnergy, -density, 1.0e12, parameters.warmEnergy, cooled));
}
}    // namespace
