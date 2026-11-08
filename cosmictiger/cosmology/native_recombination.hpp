/* CosmicTiger - homogeneous primordial recombination; BSL-1.0. */
#pragma once

#include <memory>

#include <cosmictiger/chemistry/recombination.hpp>
#include <cosmictiger/cosmology/zero_order.hpp>

inline primordial::RecombinationParameters recombinationParameters(
    CosmicParameters const& parameters) {
    primordial::RecombinationParameters recombination;
    recombination.hubble = parameters.hubble;
    recombination.omegaB = parameters.omegaB;
    recombination.omegaC = parameters.omegaC;
    recombination.omegaRadiation = parameters.omegaGam + parameters.omegaNu;
    recombination.heliumMassFraction = parameters.y;
    recombination.cmbTemperature = parameters.theta * constants::kTcmb;
    return recombination;
}

// This is the homogeneous background used by the linear perturbation solver.
// It does not initialize, evolve, or change any SPH particle chemistry.
inline void createNativeZeroOrderUniverse(ZeroOrderUniverse* universe, double maximumScaleFactor,
    CosmicParameters const& parameters, primordial::RecombinationOptions options = {}) {
    options.maxScaleFactor = maximumScaleFactor;
    auto const history = std::make_shared<primordial::RecombinationHistory>(
        recombinationParameters(parameters), options);
    createZeroOrderUniverse(universe,
        [history](double scaleFactor) { return history->electronFraction(scaleFactor); },
        maximumScaleFactor, parameters,
        {[history](double scaleFactor) { return history->gasTemperature(scaleFactor); },
            [history](double scaleFactor) { return history->hydrogenNumberDensity(scaleFactor); },
            [history](double scaleFactor) {
                constexpr double kElectronVoltErg = 1.602176634e-12;
                return history->pressurePerHydrogen(scaleFactor) * kElectronVoltErg;
            }});
}
