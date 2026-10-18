/* CosmicTiger - primordial thermochemistry; GPL-2.0-or-later. */
#pragma once

#include <cmath>

#include <cosmictiger/primordial_types.hpp>

namespace primordial {
namespace detail {
    struct MolecularModes {
        float energyKelvin{}, heatCapacity{};
    };

    // Rigid rotor, separately normalized even/odd ladders, frozen ortho:para=3:1.
    // E = k T^2 d(log Z)/dT, cv = dE/dT; never approximate E as cv(T)*T.
    // Boley et al. 2007, eqs. (1)-(5), including the normalized ortho ground state:
    // https://arxiv.org/abs/astro-ph/0701146 . Vibrations use the harmonic closure.
    PRIMORDIAL_HD inline MolecularModes molecularModes(
        float temperature, float rotation, float vibration) {
        FloatFloat energy(0.0f), capacity(0.0f);
        for (int parity = 0; parity != 2; ++parity) {
            FloatFloat partition(0.0f), first(0.0f), second(0.0f);
            for (int j = parity; j <= 64; j += 2) {
                float const level = rotation * static_cast<float>(j * (j + 1) - 2 * parity);
                float const weight = static_cast<float>(2 * j + 1) * ::expf(-level / temperature);
                partition += FloatFloat(weight);
                first += FloatFloat(weight) * FloatFloat(level);
                second += FloatFloat(weight) * FloatFloat(level) * FloatFloat(level);
            }
            FloatFloat const mean = first / partition;
            FloatFloat const mixing(parity == 0 ? 0.25f : 0.75f);
            energy += mixing * mean;
            capacity += mixing * (second / partition - mean * mean) /
                (FloatFloat(temperature) * FloatFloat(temperature));
        }
        float const ratio = vibration / temperature;
        float const occupation = ::expf(-ratio);
        float const denominator = -::expm1f(-ratio);
        energy += FloatFloat(vibration * occupation / denominator);
        capacity += FloatFloat(ratio * ratio * occupation / (denominator * denominator));
        return {energy.toFloat(), capacity.toFloat()};
    }
}    // namespace detail

// The finite molecular domain is the intersection of the selected cooling fits.
// Atomic fits cover a broader domain; neither branch extrapolates silently.
PRIMORDIAL_HD inline float minimumTemperature(Network network) {
    return network == Network::Molecular9 ? 100.0f : 10.0f;
}
PRIMORDIAL_HD inline float maximumTemperature(Network network) {
    return network == Network::Molecular9 ? 6000.0f : 1.0e8f;
}

PRIMORDIAL_HD inline Status thermalEnergyAtTemperature(WorkingState const& state,
    Environment const& environment, float temperature, FloatFloat& energy, float& heatCapacity) {
    Status const status = validateComposition(state, environment);
    if (status != Status::Success) {
        return status;
    }
    if (!finite(temperature) || temperature <= 0.0f) {
        return Status::InvalidInput;
    }
    if (temperature < minimumTemperature(environment.network) ||
        temperature > maximumTemperature(environment.network)) {
        return Status::UnsupportedState;
    }
    Composition const c = composition(state, environment);
    FloatFloat const translation = c.particles * FloatFloat(1.5f * kBoltzmannEv);
    energy = translation * FloatFloat(temperature);
    FloatFloat capacity = translation;
    if (environment.network == Network::Molecular9) {
        // H2: Boley2007 theta_r=85.4 K, theta_v=5987 K.
        // H2+: B0=Be-alpha_e/2, fundamental omega_e-2omega_exe from NIST
        // CCCBDB experimental diatomics (release22), hc/k=1.438776877 K cm.
        // https://cccbdb.nist.gov/expdiatomicsx.asp
        detail::MolecularModes const neutral = detail::molecularModes(temperature, 85.4f, 5987.0f);
        detail::MolecularModes const ion = detail::molecularModes(
            temperature, (30.21f - 0.8425f) * 1.438776877f, (2321.7f - 132.4f) * 1.438776877f);
        energy += FloatFloat(kBoltzmannEv) *
            (c.molecularHydrogen * FloatFloat(neutral.energyKelvin) +
                c.molecularHydrogenPlus * FloatFloat(ion.energyKelvin));
        capacity += FloatFloat(kBoltzmannEv) *
            (c.molecularHydrogen * FloatFloat(neutral.heatCapacity) +
                c.molecularHydrogenPlus * FloatFloat(ion.heatCapacity));
    }
    heatCapacity = capacity.toFloat();
    if (!energy.isFinite() || !finite(heatCapacity) || heatCapacity <= 0.0f) {
        return Status::InvalidInput;
    }
    return Status::Success;
}
PRIMORDIAL_HD inline Status thermalEnergyAtTemperature(
    State const& state, Environment const& environment, float temperature, float& energy) {
    FloatFloat result;
    float capacity;
    Status const status =
        thermalEnergyAtTemperature(workingState(state), environment, temperature, result, capacity);
    if (status == Status::Success) {
        energy = result.toFloat();
    }
    return status;
}

PRIMORDIAL_HD inline Status evaluateEos(
    WorkingState const& state, Environment const& environment, Eos& result) {
    Status const status = validateComposition(state, environment);
    if (status != Status::Success) {
        return status;
    }
    if (!state.thermalEnergy.isFinite() || state.thermalEnergy <= FloatFloat(0.0f)) {
        return Status::InvalidInput;
    }
    Composition const c = composition(state, environment);
    float lower = minimumTemperature(environment.network);
    float upper = maximumTemperature(environment.network);
    FloatFloat lowerEnergy, upperEnergy;
    float capacity;
    thermalEnergyAtTemperature(state, environment, lower, lowerEnergy, capacity);
    thermalEnergyAtTemperature(state, environment, upper, upperEnergy, capacity);
    // Allow only the storage-rounding uncertainty at domain endpoints; this is
    // not a thermal floor and does not modify the state or accepted energy.
    FloatFloat const roundoff = FloatFloat(4.0f * 1.1920928955078125e-7f) * state.thermalEnergy;
    if (state.thermalEnergy < lowerEnergy - roundoff ||
        state.thermalEnergy > upperEnergy + roundoff) {
        return Status::UnsupportedState;
    }
    float temperature =
        (state.thermalEnergy / (c.particles * FloatFloat(1.5f * kBoltzmannEv))).toFloat();
    temperature = ::fmaxf(lower, ::fminf(upper, temperature));
    for (int iteration = 0; iteration != 40; ++iteration) {
        FloatFloat energy;
        thermalEnergyAtTemperature(state, environment, temperature, energy, capacity);
        FloatFloat const residual = energy - state.thermalEnergy;
        if (::fabsf(residual.toFloat()) <= 3.0e-7f * state.thermalEnergy.toFloat()) {
            break;
        }
        if (residual > FloatFloat(0.0f)) {
            upper = temperature;
        } else {
            lower = temperature;
        }
        float const proposed = temperature - (residual / FloatFloat(capacity)).toFloat();
        float const next = proposed > lower && proposed < upper ? proposed : 0.5f * (lower + upper);
        if (next == temperature) {
            break;
        }
        temperature = next;
    }
    FloatFloat energy;
    thermalEnergyAtTemperature(state, environment, temperature, energy, capacity);
    if (::fabsf((energy - state.thermalEnergy).toFloat()) >
        8.0e-7f * state.thermalEnergy.toFloat()) {
        return Status::NonConvergence;
    }
    result.temperature = temperature;
    result.heatCapacity = capacity;
    result.pressurePerHydrogen = (c.particles * FloatFloat(kBoltzmannEv * temperature)).toFloat();
    result.adiabaticIndex = 1.0f + result.pressurePerHydrogen / (temperature * capacity);
    result.meanMolecularWeight =
        ((FloatFloat(1.0f) +
             FloatFloat(4.0f) * FloatFloat(environment.heliumToHydrogenNumberRatio)) /
            c.particles)
            .toFloat();
    result.pressureEnergyRatio =
        (FloatFloat(result.pressurePerHydrogen) / state.thermalEnergy).toFloat();
    return Status::Success;
}
PRIMORDIAL_HD inline Status evaluateEos(
    State const& state, Environment const& environment, Eos& result) {
    return evaluateEos(workingState(state), environment, result);
}
PRIMORDIAL_HD inline Status validateState(
    WorkingState const& state, Environment const& environment) {
    Eos result;
    return evaluateEos(state, environment, result);
}
PRIMORDIAL_HD inline Status validateState(State const& state, Environment const& environment) {
    for (int i = 0; i < kAbundanceCount; ++i) {
        if (::fabsf(state.abundances[i].raw()) > 0.5f) {
            return Status::InvalidInput;
        }
    }
    return validateState(workingState(state), environment);
}
}    // namespace primordial
