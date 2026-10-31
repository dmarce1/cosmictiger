/* CosmicTiger - atomic CIE cooling for the SH03 model; BSL-1.0. */
#pragma once

#include <bit>
#include <cmath>
#include <cstdint>

#include <cosmictiger/chemistry/primordial_rates.hpp>
#include <cosmictiger/sph/springel_and_hernquist/cooling.hpp>

// One implementation serves the stable host ABI and inline device calls.
#ifdef __CUDA_ARCH__
#define COSMICTIGER_ATOMIC_COOLING_DEFINITION CUDA_EXPORT inline
#else
#define COSMICTIGER_ATOMIC_COOLING_DEFINITION
#endif

namespace springel_and_hernquist {

namespace detail {
    CUDA_EXPORT inline bool coolingFinite(double value) {
        // Integer exponent inspection preserves validation under fast-math.
#ifdef __CUDA_ARCH__
        uint64_t const bits = static_cast<uint64_t>(__double_as_longlong(value));
#else
        uint64_t const bits = std::bit_cast<uint64_t>(value);
#endif
        return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
    }
}    // namespace detail

COSMICTIGER_ATOMIC_COOLING_DEFINITION bool AtomicCooling::evaluateAtTemperature(
    double temperature, AtomicEquilibrium& result) const {
    if (!detail::coolingFinite(hydrogenMassFraction) || hydrogenMassFraction <= 0.0 ||
        hydrogenMassFraction > 1.0 || !detail::coolingFinite(temperature) ||
        temperature < kMinimumTemperature || temperature > kMaximumTemperature) {
        return false;
    }
    primordial::RateCoefficientsT<double> rates;
    if (primordial::evaluateRateCoefficients(temperature, primordial::Network::Atomic6, rates) !=
        primordial::Status::Success) {
        return false;
    }

    // Collision-only equilibrium: adjacent ionization/recombination fluxes
    // balance. Electron density cancels. Products retain neutral He without
    // subtracting two near-unity ionized fractions.
    double const hydrogenIonization = rates.collision[0];
    double const hydrogenRecombination = rates.collision[1];
    double const heliumIonization = rates.collision[2];
    double const heliumRecombination = rates.collision[3];
    double const heliumPlusIonization = rates.collision[4];
    double const heliumPlusRecombination = rates.collision[5];
    double const hydrogenSum = hydrogenIonization + hydrogenRecombination;
    double const heliumNeutralWeight = heliumRecombination * heliumPlusRecombination;
    double const heliumIonizedWeight = heliumIonization * heliumPlusRecombination;
    double const heliumDoubleWeight = heliumIonization * heliumPlusIonization;
    double const heliumSum = heliumNeutralWeight + heliumIonizedWeight + heliumDoubleWeight;
    if (!(hydrogenSum > 0.0) || !(heliumSum > 0.0)) {
        return false;
    }
    AtomicEquilibrium state;
    state.temperature = temperature;
    state.hydrogenNeutral = hydrogenRecombination / hydrogenSum;
    state.hydrogenIonized = hydrogenIonization / hydrogenSum;
    state.heliumNeutral = heliumNeutralWeight / heliumSum;
    state.heliumIonized = heliumIonizedWeight / heliumSum;
    state.heliumDoubleIonized = heliumDoubleWeight / heliumSum;
    double const heliumRatio = (1.0 - hydrogenMassFraction) / (4.0 * hydrogenMassFraction);
    state.electronsPerHydrogen = state.hydrogenIonized +
        heliumRatio * (state.heliumIonized + 2.0 * state.heliumDoubleIonized);
    double const particlesPerHydrogen = 1.0 + heliumRatio + state.electronsPerHydrogen;
    state.meanMolecularWeight = 1.0 / (hydrogenMassFraction * particlesPerHydrogen);
    state.specificEnergy =
        1.5 * kBoltzmann * temperature / (kProtonMass * state.meanMolecularWeight);

    primordial::CompositionT<double> composition;
    composition.hydrogen = state.hydrogenNeutral;
    composition.hydrogenPlus = state.hydrogenIonized;
    composition.helium = heliumRatio * state.heliumNeutral;
    composition.heliumPlus = heliumRatio * state.heliumIonized;
    composition.heliumDoublePlus = heliumRatio * state.heliumDoubleIonized;
    composition.electrons = state.electronsPerHydrogen;
    composition.particles = particlesPerHydrogen;
    primordial::EnvironmentT<double> environment;
    environment.network = primordial::Network::Atomic6;
    environment.nHydrogen = 1.0;
    environment.heliumToHydrogenNumberRatio = heliumRatio;
    environment.cmbTemperature = 0.0;
    environment.radiation = {};
    double flux[primordial::kReactionCount]{};
    primordial::reactionFluxes(composition, environment, rates, flux);
    primordial::ThermalRatesT<double> thermal;
    if (primordial::evaluateThermalRates(composition, environment, temperature, rates, flux,
            thermal) != primordial::Status::Success) {
        return false;
    }
    // At n_H=1 cm^-3, the thermal result is eV/H/s. Multiplying by n_H
    // and eV->erg then dividing by rho^2 yields this density-independent
    // coefficient. CIE has only two-body losses in this adapter.
    double const hydrogenPerMass = hydrogenMassFraction / kProtonMass;
    state.coolingCoefficient =
        -thermal.netHeating() * kElectronVolt * hydrogenPerMass * hydrogenPerMass;
    if (!detail::coolingFinite(state.specificEnergy) ||
        !detail::coolingFinite(state.coolingCoefficient) || state.coolingCoefficient < 0.0) {
        return false;
    }
    result = state;
    return true;
}

COSMICTIGER_ATOMIC_COOLING_DEFINITION bool AtomicCooling::evaluateAtEnergy(
    double specificEnergy, AtomicEquilibrium& result) const {
    AtomicEquilibrium lower;
    AtomicEquilibrium upper;
    if (!detail::coolingFinite(specificEnergy) ||
        !evaluateAtTemperature(kMinimumTemperature, lower) ||
        !evaluateAtTemperature(kMaximumTemperature, upper) ||
        specificEnergy < lower.specificEnergy || specificEnergy > upper.specificEnergy) {
        return false;
    }
    if (specificEnergy == lower.specificEnergy) {
        result = lower;
        return true;
    }
    if (specificEnergy == upper.specificEnergy) {
        result = upper;
        return true;
    }
    // Logarithmic bisection is bounded, and resolves the rapidly changing
    // equilibrium molecular weight without assuming a fixed fully ionized mu.
    for (int iteration = 0; iteration != 80; ++iteration) {
        AtomicEquilibrium state;
        double const temperature = ::sqrt(lower.temperature * upper.temperature);
        if (!evaluateAtTemperature(temperature, state)) {
            return false;
        }
        if (::fabs(state.specificEnergy - specificEnergy) <= 2.0e-8 * specificEnergy ||
            upper.temperature - lower.temperature <= 1.0e-12 * temperature) {
            if (::fabs(state.specificEnergy - specificEnergy) > 2.0e-6 * specificEnergy) {
                return false;
            }
            result = state;
            return true;
        }
        if (state.specificEnergy < specificEnergy) {
            lower = state;
        } else {
            upper = state;
        }
    }
    return false;
}

COSMICTIGER_ATOMIC_COOLING_DEFINITION double AtomicCooling::minimumEnergy() const {
    AtomicEquilibrium state;
    return evaluateAtTemperature(kMinimumTemperature, state) ? state.specificEnergy :
                                                               detail::invalid();
}

COSMICTIGER_ATOMIC_COOLING_DEFINITION double AtomicCooling::maximumEnergy() const {
    AtomicEquilibrium state;
    return evaluateAtTemperature(kMaximumTemperature, state) ? state.specificEnergy :
                                                               detail::invalid();
}

COSMICTIGER_ATOMIC_COOLING_DEFINITION bool AtomicCooling::cool(double specificEnergy,
    double density, double timestep, double floorEnergy, double& cooledEnergy) const {
    double const minimumSupportedEnergy = minimumEnergy();
    if (!detail::coolingFinite(specificEnergy) || specificEnergy <= 0.0 ||
        !detail::coolingFinite(density) || density < 0.0 || !detail::coolingFinite(timestep) ||
        timestep < 0.0 || !detail::coolingFinite(floorEnergy) ||
        !detail::coolingFinite(minimumSupportedEnergy) || floorEnergy < minimumSupportedEnergy) {
        return false;
    }
    // Adiabatic expansion may put ordinary gas below the rate table's lower
    // temperature. Below the imposed cooling floor no rate is needed: retain
    // its positive thermal energy rather than rejecting it or heating it.
    if (specificEnergy <= floorEnergy) {
        cooledEnergy = specificEnergy;
        return true;
    }
    AtomicEquilibrium upper;
    if (!evaluateAtEnergy(specificEnergy, upper)) {
        return false;
    }
    if (density == 0.0 || timestep == 0.0) {
        cooledEnergy = specificEnergy;
        return true;
    }
    AtomicEquilibrium lower;
    if (!evaluateAtEnergy(floorEnergy, lower)) {
        return false;
    }
    double const exposure = density * timestep;
    if (!detail::coolingFinite(exposure)) {
        return false;
    }
    if (floorEnergy - specificEnergy + exposure * lower.coolingCoefficient >= 0.0) {
        cooledEnergy = floorEnergy;
        return true;
    }
    // The radiative term is evaluated only here, once per source solve. Its
    // removal must not be repeated by the non-equilibrium chemistry operator.
    for (int iteration = 0; iteration != 100; ++iteration) {
        AtomicEquilibrium state;
        double const temperature = ::sqrt(lower.temperature * upper.temperature);
        if (!evaluateAtTemperature(temperature, state)) {
            return false;
        }
        double const residual =
            state.specificEnergy - specificEnergy + exposure * state.coolingCoefficient;
        if (::fabs(residual) <= 2.0e-8 * specificEnergy ||
            upper.temperature - lower.temperature <= 1.0e-12 * temperature) {
            if (!detail::coolingFinite(residual) || ::fabs(residual) > 3.0e-6 * specificEnergy) {
                return false;
            }
            double const limitedEnergy =
                state.specificEnergy < specificEnergy ? state.specificEnergy : specificEnergy;
            cooledEnergy = floorEnergy < limitedEnergy ? limitedEnergy : floorEnergy;
            return true;
        }
        if (residual < 0.0) {
            lower = state;
        } else {
            upper = state;
        }
    }
    return false;
}

COSMICTIGER_ATOMIC_COOLING_DEFINITION Parameters defaultParameters(AtomicCooling const& cooling) {
    Parameters result;
    if (!detail::coolingFinite(cooling.hydrogenMassFraction) ||
        cooling.hydrogenMassFraction <= 0.0 || cooling.hydrogenMassFraction > 1.0) {
        result.coldEnergy = detail::invalid();
        return result;
    }
    double const hydrogen = cooling.hydrogenMassFraction;
    double const neutralWeight = 1.0 / (hydrogen + 0.25 * (1.0 - hydrogen));
    double const ionizedWeight = 1.0 / (2.0 * hydrogen + 0.75 * (1.0 - hydrogen));
    result.coldEnergy = 1.5 * kBoltzmann * 1.0e3 / (kProtonMass * neutralWeight);
    result.warmEnergy = 1.5 * kBoltzmann * 1.0e4 / (kProtonMass * neutralWeight);
    result.supernovaEnergy = 1.5 * kBoltzmann * 1.0e8 / (kProtonMass * ionizedWeight);
    return result;
}

COSMICTIGER_ATOMIC_COOLING_DEFINITION bool makeModel(
    Parameters const& parameters, AtomicCooling const& cooling, Model& result) {
    if (!validParameters(parameters) || ::fabs(parameters.gamma - 5.0 / 3.0) > 1.0e-12) {
        return false;    // AtomicCooling is an ideal monatomic thermal closure.
    }
    double const thresholdHotEnergy =
        parameters.coldEnergy + parameters.supernovaEnergy / (1.0 + parameters.evaporationFactor);
    AtomicEquilibrium state;
    return cooling.evaluateAtEnergy(thresholdHotEnergy, state) &&
        calibrate(parameters, state.coolingCoefficient, result);
}

COSMICTIGER_ATOMIC_COOLING_DEFINITION bool evaluateEquilibrium(
    Model const& model, AtomicCooling const& cooling, double density, Equilibrium& result) {
    if (!validParameters(model.parameters) || !detail::coolingFinite(model.thresholdDensity) ||
        model.thresholdDensity <= 0.0 || !detail::coolingFinite(density) ||
        density < model.thresholdDensity) {
        return false;
    }
    AtomicEquilibrium state;
    return cooling.evaluateAtEnergy(hotEnergy(model, density), state) &&
        equilibrium(model, density, state.coolingCoefficient, result);
}

}    // namespace springel_and_hernquist

#undef COSMICTIGER_ATOMIC_COOLING_DEFINITION
