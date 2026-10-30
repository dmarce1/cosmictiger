/* CosmicTiger - primordial thermochemistry; BSL-1.0. */
#pragma once

#include <cmath>

#include <cosmictiger/chemistry/primordial_types.hpp>
#include <cosmictiger/cuda/cuda_export.hpp>

namespace primordial {
namespace detail {
    struct MolecularModes {
        float energyKelvin{}, heatCapacity{};
    };

    // Above this temperature the classical rigid-rotor asymptote replaces the
    // ladder sums: its O(x^4) truncation (59x^4/14175, about 2e-11 at
    // x = theta/T = 8.5e-3 for H2) is negligible there, while the 129-level
    // sums used between 6000 K and 1e4 K are still converged for both
    // molecules (the H2+ ladder needs the longer sum above 6000 K). The 65-level
    // sums below 6000 K are untouched, so every previously supported state
    // is unchanged bit for bit. See docs/molecular9_hot.md.
    inline constexpr float kRotorSeriesTemperature = 1.0e4f;

    // Rigid rotor, separately normalized even/odd ladders, frozen ortho:para=3:1.
    // E = k T^2 d(log Z)/dT, cv = dE/dT; never approximate E as cv(T)*T.
    // Boley et al. 2007, eqs. (1)-(5), including the normalized ortho ground state:
    // https://arxiv.org/abs/astro-ph/0701146 . Vibrations use the harmonic closure.
    CUDA_EXPORT inline MolecularModes molecularModes(
        float temperature, float rotation, float vibration) {
        FloatFloat energy(0.0f), capacity(0.0f);
        if (temperature > kRotorSeriesTemperature) {
            // Mulholland's asymptotic rotational partition function,
            // Z = (T/theta)[1 + x/3 + x^2/15 + 4x^3/315 + O(x^4)], x = theta/T,
            // gives per ladder E/k = T - theta/3 - theta^2/(45T) - 8theta^3/(945T^2)
            // and cv/k = 1 + theta^2/(45T^2) + 16theta^3/(945T^3). The even and
            // odd ladders each hold half of Z to within exp(-pi^2/(4x)) here
            // (x <= 8.6e-3), so both have this mean energy; the odd ladder's
            // normalized ground state subtracts 2theta. Truncation error is
            // O(x^4) ~ 5e-9, below the binary32 EOS inversion tolerance.
            FloatFloat const x = FloatFloat(rotation) / FloatFloat(temperature);
            FloatFloat const xSquared = x * x;
            FloatFloat const mean = FloatFloat(temperature) *
                (((FloatFloat(1.0f) - x / FloatFloat(3.0f)) - xSquared / FloatFloat(45.0f)) -
                    FloatFloat(8.0f / 945.0f) * xSquared * x);
            energy += mean - FloatFloat(1.5f) * FloatFloat(rotation);
            capacity += (FloatFloat(1.0f) + xSquared / FloatFloat(45.0f)) +
                FloatFloat(16.0f / 945.0f) * xSquared * x;
        }
        // 65 levels converge every state up to 6000 K; the H2+ ladder needs the
        // longer sum to stay converged up to the series switch.
        int const highestLevel = temperature > 6000.0f ? 128 : 64;
        for (int parity = 0; parity != 2 && temperature <= kRotorSeriesTemperature; ++parity) {
            FloatFloat partition(0.0f), first(0.0f), second(0.0f);
            for (int j = parity; j <= highestLevel; j += 2) {
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

// The cold molecular cooling branches use published 10 K data; see
// primordial_rates.hpp. This is a supported-domain boundary, not a thermal floor.
CUDA_EXPORT inline float minimumTemperature(Network) {
    return 10.0f;
}
// Both networks share the atomic ceiling. Molecular9 keeps all nine species
// through heating, dissociation and recombination; its molecular fits above
// their published ranges follow the documented continuations in
// primordial_rates.hpp and docs/molecular9_hot.md.
CUDA_EXPORT inline float maximumTemperature(Network) {
    return 1.0e8f;
}

CUDA_EXPORT inline Status thermalEnergyAtTemperature(WorkingState const& state,
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
CUDA_EXPORT inline Status thermalEnergyAtTemperature(
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

CUDA_EXPORT inline Status evaluateEos(
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
    FloatFloat const roundoff = FloatFloat(4.0f * 1.1920928955078125e-7f) * state.thermalEnergy;
    if (environment.network == Network::Molecular9) {
        FloatFloat warmEnergy;
        thermalEnergyAtTemperature(state, environment, 100.0f, warmEnergy, capacity);
        // Keep the established inversion bracket and endpoint rounding for every
        // previously supported state; only colder states need the new bracket.
        if (state.thermalEnergy >= warmEnergy - roundoff) {
            lower = 100.0f;
            lowerEnergy = warmEnergy;
        } else {
            thermalEnergyAtTemperature(state, environment, lower, lowerEnergy, capacity);
        }
    } else {
        thermalEnergyAtTemperature(state, environment, lower, lowerEnergy, capacity);
    }
    thermalEnergyAtTemperature(state, environment, upper, upperEnergy, capacity);
    // Allow only the storage-rounding uncertainty at domain endpoints; this is
    // not a thermal floor and does not modify the state or accepted energy.
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
CUDA_EXPORT inline Status evaluateEos(
    State const& state, Environment const& environment, Eos& result) {
    return evaluateEos(workingState(state), environment, result);
}
CUDA_EXPORT inline Status validateState(WorkingState const& state, Environment const& environment) {
    Eos result;
    return evaluateEos(state, environment, result);
}
CUDA_EXPORT inline Status validateState(State const& state, Environment const& environment) {
    for (int i = 0; i < kAbundanceCount; ++i) {
        if (::fabsf(state.abundances[i].raw()) > 0.5f) {
            return Status::InvalidInput;
        }
    }
    return validateState(workingState(state), environment);
}
}    // namespace primordial
