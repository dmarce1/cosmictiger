#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/chemistry/gadget4_cooling.hpp>

namespace {
namespace g4 = gadget4;

double energyAtTemperature(double temperature, double electrons) {
    double const molecularWeight =
        (1.0 + 4.0 * g4::kHeliumNumberRatio) / (1.0 + g4::kHeliumNumberRatio + electrons);
    return g4::kBoltzmann * temperature / (g4::kGammaMinusOne * g4::kProtonMass * molecularWeight);
}

TEST(Gadget4Cooling, NeutralAndFullyIonizedLimits) {
    auto const parameters = g4::makeParameters(1.0, false, 0.0);
    double electrons = 0.7;
    g4::GasState gas;
    ASSERT_EQ(g4::equilibriumAtTemperature(5.0, 1.0e-24, parameters, electrons, gas),
        g4::Status::Success);
    EXPECT_EQ(electrons, 0.0);
    EXPECT_EQ(gas.hydrogenNeutral, 1.0);
    EXPECT_EQ(gas.heliumNeutral, g4::kHeliumNumberRatio);
    EXPECT_EQ(gas.hydrogenIonized, 0.0);
    ASSERT_EQ(g4::equilibriumAtTemperature(1.0e10, 1.0e-24, parameters, electrons, gas),
        g4::Status::Success);
    EXPECT_EQ(gas.hydrogenNeutral, 0.0);
    EXPECT_EQ(gas.heliumNeutral, 0.0);
    EXPECT_EQ(gas.hydrogenIonized, 1.0);
    EXPECT_EQ(gas.heliumDoubleIonized, g4::kHeliumNumberRatio);
    EXPECT_EQ(electrons, 1.0 + 2.0 * g4::kHeliumNumberRatio);
}

TEST(Gadget4Cooling, EquilibriumConservesNucleiAndChargeWithoutDensityGate) {
    auto const parameters = g4::makeParameters(1.0, false, 0.0);
    for (double density : {1.0e-30, 1.0e-24, 1.0e-16}) {
        for (double temperature : {100.0, 1.0e4, 3.0e4, 1.0e5, 1.0e6, 1.0e8}) {
            double electrons = 0.1;
            g4::GasState gas;
            ASSERT_EQ(
                g4::equilibriumAtTemperature(temperature, density, parameters, electrons, gas),
                g4::Status::Success);
            EXPECT_NEAR(gas.hydrogenNeutral + gas.hydrogenIonized, 1.0, 2.0e-15);
            EXPECT_NEAR(gas.heliumNeutral + gas.heliumIonized + gas.heliumDoubleIonized,
                g4::kHeliumNumberRatio, 2.0e-15);
            EXPECT_EQ(
                electrons, gas.hydrogenIonized + gas.heliumIonized + 2.0 * gas.heliumDoubleIonized);
            double reconstructed = 0.0;
            ASSERT_EQ(g4::temperatureFromEnergy(energyAtTemperature(temperature, electrons),
                          density, parameters, electrons, reconstructed),
                g4::Status::Success);
            EXPECT_NEAR(reconstructed, temperature, 2.0e-3 * temperature);
        }
    }
}

TEST(Gadget4Cooling, HighTemperatureFreeFreeAndComptonContinuation) {
    auto parameters = g4::makeParameters(0.1, false, 0.0);
    double const temperature = 1.0e10;
    double const density = 1.0e-29;
    double const ionizedElectrons = 1.0 + 2.0 * g4::kHeliumNumberRatio;
    double const energy = energyAtTemperature(temperature, ionizedElectrons);
    double electrons = ionizedElectrons;
    double atomicRate;
    ASSERT_EQ(
        g4::coolingRate(energy, density, parameters, electrons, atomicRate), g4::Status::Success);
    // Analytic fully ionized limit above the tabulation; no interpolation or
    // ion-equilibrium iteration contributes to this reference expression.
    double const freeFree = 1.42e-27 * std::sqrt(temperature) * (1.1 + 0.34 * std::exp(-6.75)) *
        (1.0 + 4.0 * g4::kHeliumNumberRatio) * ionizedElectrons;
    EXPECT_NEAR(atomicRate, -freeFree, 2.0e-13 * freeFree);
    parameters.comoving = true;
    parameters.redshift = 9.0;
    double cosmologicalRate;
    ASSERT_EQ(g4::coolingRate(energy, density, parameters, electrons, cosmologicalRate),
        g4::Status::Success);
    double const hydrogenDensity = g4::kHydrogenMassFraction * density / g4::kProtonMass;
    double const compton =
        5.65e-36 * ionizedElectrons * (temperature - 27.3) * 1.0e4 / hydrogenDensity;
    EXPECT_NEAR(cosmologicalRate - atomicRate, -compton, 2.0e-13 * compton);
}

TEST(Gadget4Cooling, PinnedStiffCoolingTrajectory) {
    auto const parameters = g4::makeParameters(1.0, false, 0.0);
    double const density = 1.0e-24;
    double electrons = 1.0;
    g4::GasState gas;
    ASSERT_EQ(g4::equilibriumAtTemperature(1.0e6, density, parameters, electrons, gas),
        g4::Status::Success);
    double const initialEnergy = energyAtTemperature(1.0e6, electrons);
    double coolingTime;
    ASSERT_EQ(g4::coolingTime(initialEnergy, density, parameters, electrons, coolingTime),
        g4::Status::Success);
    ASSERT_GT(coolingTime, 0.0);
    double const interval = 0.5 * coolingTime;
    double finalEnergy = 0.0;
    ASSERT_EQ(g4::doCooling(initialEnergy, density, interval, parameters, electrons, finalEnergy),
        g4::Status::Success);
    EXPECT_LT(finalEnergy, initialEnergy);
    EXPECT_GT(finalEnergy, 0.0);
    // Original GADGET-4 at the pinned commit gives these values. Its 1e-6
    // energy-bracket tolerance is not a relative thermal-residual tolerance:
    // this stiff trajectory reaches the atomic transition, where the separate
    // 1e-3 temperature iteration makes the rate depend on its incoming Ne guess.
    // The original's freshly evaluated residual is -6.3681666168625e10 erg/g.
    constexpr double kReferenceEnergy = 1561313572958.4404;
    constexpr double kReferenceElectrons = 0.12106752964117247;
    EXPECT_NEAR(finalEnergy, kReferenceEnergy, 4.0e-12 * kReferenceEnergy);
    EXPECT_NEAR(electrons, kReferenceElectrons, 4.0e-12 * kReferenceElectrons);
}

TEST(Gadget4Cooling, ImplicitCoolingMeetsEnergyBracketAccuracy) {
    auto const parameters = g4::makeParameters(1.0, false, 0.0);
    double const density = 1.0e-24;
    double const initialTemperature = 1.0e10;
    double electrons = 1.0 + 2.0 * g4::kHeliumNumberRatio;
    double const initialEnergy = energyAtTemperature(initialTemperature, electrons);
    double const temperaturePerEnergy = initialTemperature / initialEnergy;
    double const hydrogenDensity = g4::kHydrogenMassFraction * density / g4::kProtonMass;
    double const rateFactor = hydrogenDensity * hydrogenDensity / density;
    auto loss = [&](double temperature) {
        double const delta = 5.5 - std::log10(temperature);
        double const gaunt = 1.1 + 0.34 * std::exp(-delta * delta / 3.0);
        return rateFactor * 1.42e-27 * std::sqrt(temperature) * gaunt *
            (1.0 + 4.0 * g4::kHeliumNumberRatio) * (1.0 + 2.0 * g4::kHeliumNumberRatio);
    };
    double const interval = 0.5 * initialEnergy / loss(initialTemperature);
    double finalEnergy;
    ASSERT_EQ(g4::doCooling(initialEnergy, density, interval, parameters, electrons, finalEnergy),
        g4::Status::Success);
    ASSERT_LT(finalEnergy, initialEnergy);
    constexpr double kRelativeBracketTolerance = 1.0e-6;
    double const energyTolerance = kRelativeBracketTolerance * finalEnergy;
    double const lowerEnergy = finalEnergy - energyTolerance;
    ASSERT_GT(lowerEnergy * temperaturePerEnergy, 1.0e9);
    // Here composition is exactly fully ionized, with analytic free-free loss
    // L(u)=C*sqrt(u)*g(log10(T)); no abundance/temperature iteration error enters.
    // GADGET-4 brackets u to 1e-6*u. Above 1e9 K, g decreases and L increases,
    // hence 0 < R'(u)=1+dt*L'(u) <= 1+dt*L(lower)/(2*lower).
    // The mean-value theorem turns the specified bracket tolerance into this
    // residual bound without assuming a separate relative-residual tolerance.
    double const slopeBound =
        1.0 + interval * loss(lowerEnergy * temperaturePerEnergy) / (2.0 * lowerEnergy);
    double const residual =
        finalEnergy - initialEnergy + interval * loss(finalEnergy * temperaturePerEnergy);
    EXPECT_LE(std::abs(residual), energyTolerance * slopeBound);
}

TEST(Gadget4Cooling, CmbHeatingRemainsWithoutExternalUv) {
    auto const parameters = g4::makeParameters(1.0e-4, true, 0.0);
    double electrons = 1.0;
    double time = -1.0;
    ASSERT_EQ(g4::coolingTime(1.0e12, 1.0e-30, parameters, electrons, time), g4::Status::Success);
    EXPECT_EQ(time, 0.0);
    double finalEnergy;
    ASSERT_EQ(g4::doCooling(1.0e12, 1.0e-30, 1.0e6, parameters, electrons, finalEnergy),
        g4::Status::Success);
    EXPECT_GT(finalEnergy, 1.0e12);
}

TEST(Gadget4Cooling, FloorAndFailedApplicationAreTransactional) {
    auto const parameters = g4::makeParameters(1.0, false, 2.0e10);
    g4::Input input{{0.3f}, 1.0e10, 1.0e-24, 0.0};
    auto result = g4::advance(input, parameters);
    ASSERT_EQ(result.status, g4::Status::Success);
    EXPECT_EQ(result.specificEnergy, parameters.minimumSpecificEnergy);
    input.density = -1.0;
    result = g4::advance(input, parameters);
    EXPECT_EQ(result.status, g4::Status::InvalidInput);
    EXPECT_EQ(result.state, input.state);
    EXPECT_EQ(result.specificEnergy, input.specificEnergy);
    double electrons = 0.3;
    double output = 123.0;
    EXPECT_EQ(g4::doCooling(std::numeric_limits<double>::quiet_NaN(), 1.0e-24, 1.0, parameters,
                  electrons, output),
        g4::Status::InvalidInput);
    EXPECT_EQ(electrons, 0.3);
    EXPECT_EQ(output, 123.0);
}

TEST(Gadget4Cooling, PinnedOfficialReferenceTrajectories) {
    // Captured by running the original GADGET-4 cooling.cc methods at commit
    // 6fb393b5e25907f2ff06211f67d341c5e40e90d1 with unit conversions set to one.
    // Each independent lookup starts from Ne=0.8. Mode 0: no Compton;
    // mode 2: Compton exchange at z=19. External UV is absent in every case.
    struct Reference {
        int mode;
        double density, energy, temperature, rate, coolingTime, interval, finalEnergy,
            finalElectrons;
    };
    Reference const references[]{{0, 1.0e-30, 507645996.33516669, 4.9953160200425692, 0.0, 0.0,
                                     1.0e10, 507645996.33516669, 0.0},
        {0, 1.0e-24, 1016440141577.0153, 9983.198285404018, -4.8244017279076195e-27,
            1020485068007431.5, 306145520402229.44, 958392919501.96899, 0.00044665019183118629},
        {0, 1.0e-16, 5956949465154.6387, 30017.351436279576, -7.0300587530901795e-23,
            4104.2449449175756, 1231.2734834752725, 2709664021668.5283, 0.60584556895712771},
        {0, 1.0e-24, 2.1048736433409349e18, 10007871743.583698, -2.381575576814954e-22,
            42808496281724592.0, 1.0e16, 1.6673681095160735e18, 1.1578947368421053},
        {2, 1.0e-30, 10152919926.703333, 99.834079608850999, 0.0, 0.0, 1.0e10, 10152919926.703333,
            0.0},
        {2, 1.0e-16, 210486849342088.5, 1000787.1575833943, -5.6924935296877454e-24,
            1790979.9853346664, 537293.99560039991, 145709308149682.75, 1.1578839833918182}};
    auto near = [](double actual, double expected) {
        EXPECT_NEAR(actual, expected, 4.0e-12 * std::abs(expected));
    };
    for (auto const& reference : references) {
        auto const parameters =
            g4::makeParameters(reference.mode == 2 ? 0.05 : 1.0, reference.mode != 0, 0.0);
        double electrons = 0.8;
        double actual;
        ASSERT_EQ(g4::temperatureFromEnergy(
                      reference.energy, reference.density, parameters, electrons, actual),
            g4::Status::Success);
        near(actual, reference.temperature);
        electrons = 0.8;
        ASSERT_EQ(
            g4::coolingRate(reference.energy, reference.density, parameters, electrons, actual),
            g4::Status::Success);
        near(actual, reference.rate);
        electrons = 0.8;
        ASSERT_EQ(
            g4::coolingTime(reference.energy, reference.density, parameters, electrons, actual),
            g4::Status::Success);
        near(actual, reference.coolingTime);
        electrons = 0.8;
        ASSERT_EQ(g4::doCooling(reference.energy, reference.density, reference.interval, parameters,
                      electrons, actual),
            g4::Status::Success);
        near(actual, reference.finalEnergy);
        near(electrons, reference.finalElectrons);
    }
}

TEST(Gadget4Cooling, BatchAndCudaUseTheSameSource) {
    auto const parameters = g4::makeParameters(1.0, false, 0.0);
    std::vector<g4::Input> const inputs{{{0.0f}, 1.0e12, 1.0e-28, 1.0e10},
        {{1.0f}, 1.0e14, 1.0e-24, 1.0e11}, {{1.1f}, 1.0e16, 1.0e-16, 1.0e3}};
    std::vector<g4::Result> cpu(inputs.size()), gpu(inputs.size());
    ASSERT_TRUE(g4::integrateBatchCpu(inputs.data(), cpu.data(), inputs.size(), parameters, 2));
    for (size_t index = 0; index < inputs.size(); ++index) {
        auto const direct = g4::advance(inputs[index], parameters);
        ASSERT_EQ(cpu[index].status, g4::Status::Success);
        EXPECT_EQ(cpu[index].specificEnergy, direct.specificEnergy);
        EXPECT_EQ(cpu[index].state, direct.state);
    }
    std::string error;
    bool const dispatched =
        g4::integrateBatchCuda(inputs.data(), gpu.data(), inputs.size(), parameters, &error);
    if (!dispatched && gpu.front().status == g4::Status::BackendUnavailable) {
        GTEST_SKIP() << error;
    }
    ASSERT_TRUE(dispatched) << error;
    for (size_t index = 0; index < inputs.size(); ++index) {
        ASSERT_EQ(gpu[index].status, cpu[index].status);
        EXPECT_NEAR(gpu[index].specificEnergy, cpu[index].specificEnergy,
            5.0e-6 * cpu[index].specificEnergy);
        EXPECT_NEAR(gpu[index].state.electronFraction, cpu[index].state.electronFraction, 5.0e-6);
    }
}
}    // namespace
