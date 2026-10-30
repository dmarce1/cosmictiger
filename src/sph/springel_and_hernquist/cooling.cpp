/* CosmicTiger - atomic CIE cooling for the SH03 model; BSL-1.0. */

#include <algorithm>
#include <cmath>
#include <limits>

#include <cosmictiger/chemistry/primordial_rates.hpp>
#include <cosmictiger/sph/springel_and_hernquist/cooling.hpp>

namespace springel_and_hernquist {

bool AtomicCooling::evaluateAtTemperature(double temperature, AtomicEquilibrium& result) const {
    if (!std::isfinite(hydrogenMassFraction) || hydrogenMassFraction <= 0.0 ||
        hydrogenMassFraction > 1.0 || !std::isfinite(temperature) ||
        temperature < kMinimumTemperature || temperature > kMaximumTemperature) {
        return false;
    }
    primordial::RateCoefficients rates;
    if (primordial::evaluateRateCoefficients(static_cast<float>(temperature),
            primordial::Network::Atomic6, rates) != primordial::Status::Success) {
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

    primordial::Composition composition;
    composition.hydrogen = FloatFloat(state.hydrogenNeutral);
    composition.hydrogenPlus = FloatFloat(state.hydrogenIonized);
    composition.helium = FloatFloat(heliumRatio * state.heliumNeutral);
    composition.heliumPlus = FloatFloat(heliumRatio * state.heliumIonized);
    composition.heliumDoublePlus = FloatFloat(heliumRatio * state.heliumDoubleIonized);
    composition.electrons = FloatFloat(state.electronsPerHydrogen);
    composition.particles = FloatFloat(particlesPerHydrogen);
    primordial::Environment environment;
    environment.network = primordial::Network::Atomic6;
    environment.nHydrogen = 1.0f;
    environment.heliumToHydrogenNumberRatio = static_cast<float>(heliumRatio);
    environment.cmbTemperature = 0.0f;
    environment.radiation = {};
    FloatFloat flux[primordial::kReactionCount]{};
    primordial::reactionFluxes(composition, environment, rates, flux);
    primordial::ThermalRates thermal;
    if (primordial::evaluateThermalRates(composition, environment, static_cast<float>(temperature),
            rates, flux, thermal) != primordial::Status::Success) {
        return false;
    }
    // At n_H=1 cm^-3, the thermal result is eV/H/s. Multiplying by n_H
    // and eV->erg then dividing by rho^2 yields this density-independent
    // coefficient. CIE has only two-body losses in this adapter.
    double const hydrogenPerMass = hydrogenMassFraction / kProtonMass;
    state.coolingCoefficient =
        -thermal.netHeating().toDouble() * kElectronVolt * hydrogenPerMass * hydrogenPerMass;
    if (!std::isfinite(state.specificEnergy) || !std::isfinite(state.coolingCoefficient) ||
        state.coolingCoefficient < 0.0) {
        return false;
    }
    result = state;
    return true;
}

bool AtomicCooling::evaluateAtEnergy(double specificEnergy, AtomicEquilibrium& result) const {
    AtomicEquilibrium lower;
    AtomicEquilibrium upper;
    if (!std::isfinite(specificEnergy) || !evaluateAtTemperature(kMinimumTemperature, lower) ||
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
        double const temperature = std::sqrt(lower.temperature * upper.temperature);
        if (!evaluateAtTemperature(temperature, state)) {
            return false;
        }
        if (std::abs(state.specificEnergy - specificEnergy) <= 2.0e-8 * specificEnergy ||
            upper.temperature - lower.temperature <= 1.0e-12 * temperature) {
            // The underlying atomic fits are float-valued; their small jumps
            // bound inversion accuracy even though the bracket uses doubles.
            if (std::abs(state.specificEnergy - specificEnergy) > 2.0e-6 * specificEnergy) {
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

double AtomicCooling::minimumEnergy() const {
    AtomicEquilibrium state;
    return evaluateAtTemperature(kMinimumTemperature, state) ? state.specificEnergy :
                                                               detail::invalid();
}

double AtomicCooling::maximumEnergy() const {
    AtomicEquilibrium state;
    return evaluateAtTemperature(kMaximumTemperature, state) ? state.specificEnergy :
                                                               detail::invalid();
}

bool AtomicCooling::cool(double specificEnergy, double density, double timestep, double floorEnergy,
    double& cooledEnergy) const {
    double const minimumSupportedEnergy = minimumEnergy();
    if (!std::isfinite(specificEnergy) || specificEnergy <= 0.0 || !std::isfinite(density) ||
        density < 0.0 || !std::isfinite(timestep) || timestep < 0.0 ||
        !std::isfinite(floorEnergy) || !std::isfinite(minimumSupportedEnergy) ||
        floorEnergy < minimumSupportedEnergy) {
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
    if (!std::isfinite(exposure)) {
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
        double const temperature = std::sqrt(lower.temperature * upper.temperature);
        if (!evaluateAtTemperature(temperature, state)) {
            return false;
        }
        double const residual =
            state.specificEnergy - specificEnergy + exposure * state.coolingCoefficient;
        if (std::abs(residual) <= 2.0e-8 * specificEnergy ||
            upper.temperature - lower.temperature <= 1.0e-12 * temperature) {
            if (!std::isfinite(residual) || std::abs(residual) > 3.0e-6 * specificEnergy) {
                return false;
            }
            cooledEnergy = std::max(floorEnergy, std::min(specificEnergy, state.specificEnergy));
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

Parameters defaultParameters(AtomicCooling const& cooling) {
    Parameters result;
    if (!std::isfinite(cooling.hydrogenMassFraction) || cooling.hydrogenMassFraction <= 0.0 ||
        cooling.hydrogenMassFraction > 1.0) {
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

bool makeModel(Parameters const& parameters, AtomicCooling const& cooling, Model& result) {
    if (!validParameters(parameters) || std::abs(parameters.gamma - 5.0 / 3.0) > 1.0e-12) {
        return false;    // AtomicCooling is an ideal monatomic thermal closure.
    }
    double const thresholdHotEnergy =
        parameters.coldEnergy + parameters.supernovaEnergy / (1.0 + parameters.evaporationFactor);
    AtomicEquilibrium state;
    return cooling.evaluateAtEnergy(thresholdHotEnergy, state) &&
        calibrate(parameters, state.coolingCoefficient, result);
}

bool evaluateEquilibrium(
    Model const& model, AtomicCooling const& cooling, double density, Equilibrium& result) {
    if (!validParameters(model.parameters) || !std::isfinite(model.thresholdDensity) ||
        model.thresholdDensity <= 0.0 || !std::isfinite(density) ||
        density < model.thresholdDensity) {
        return false;
    }
    AtomicEquilibrium state;
    return cooling.evaluateAtEnergy(hotEnergy(model, density), state) &&
        equilibrium(model, density, state.coolingCoefficient, result);
}

}    // namespace springel_and_hernquist
