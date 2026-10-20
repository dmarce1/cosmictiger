/* CosmicTiger - homogeneous primordial recombination; GPL-2.0-or-later. */
#pragma once

#include <memory>

#include <cosmictiger/chemistry/recombination.hpp>
#include <cosmictiger/cosmology/zero_order.hpp>

inline primordial::RecombinationParameters recombinationParameters(
    CosmicParameters const& parameters) {
    return {parameters.hubble, parameters.omegaB, parameters.omegaC,
        parameters.omegaGam + parameters.omegaNu, parameters.y,
        parameters.theta * constants::kTcmb};
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
