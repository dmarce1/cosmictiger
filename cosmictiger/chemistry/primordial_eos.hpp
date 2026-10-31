/* CosmicTiger - primordial thermochemistry; BSL-1.0. */
#pragma once

#include <cmath>

#include <cosmictiger/chemistry/primordial_types.hpp>
#include <cosmictiger/cuda/cuda_export.hpp>

namespace primordial {
namespace detail {
    template <class Real>
    struct MolecularModesT {
        Real energyKelvin{}, heatCapacity{};
    };

    // Above this temperature the classical rigid-rotor asymptote replaces the
    // ladder sums: its O(x^4) truncation (59x^4/14175, about 2e-11 at
    // x = theta/T = 8.5e-3 for H2) is negligible there, while the 129-level
    // sums used between 6000 K and 1e4 K are still converged for both
    // molecules (the H2+ ladder needs the longer sum above 6000 K). The 65-level
    // sums below 6000 K retain the same physical closure. See docs/molecular9_hot.md.
    template <class Real>
    inline constexpr Real kRotorSeriesTemperature = Real(1.0e4);

    // Rigid rotor, separately normalized even/odd ladders, frozen ortho:para=3:1.
    // E = k T^2 d(log Z)/dT, cv = dE/dT; never approximate E as cv(T)*T.
    // Boley et al. 2007, eqs. (1)-(5), including the normalized ortho ground state:
    // https://arxiv.org/abs/astro-ph/0701146 . Vibrations use the harmonic closure.
    template <class Real>
    CUDA_EXPORT inline MolecularModesT<Real> molecularModes(
        Real temperature, Real rotation, Real vibration) {
        Real energy(Real(0.0)), capacity(Real(0.0));
        if (temperature > kRotorSeriesTemperature<Real>) {
            // Mulholland's asymptotic rotational partition function,
            // Z = (T/theta)[1 + x/3 + x^2/15 + 4x^3/315 + O(x^4)], x = theta/T,
            // gives per ladder E/k = T - theta/3 - theta^2/(45T) - 8theta^3/(945T^2)
            // and cv/k = 1 + theta^2/(45T^2) + 16theta^3/(945T^3). The even and
            // odd ladders each hold half of Z to within exp(-pi^2/(4x)) here
            // (x <= 8.6e-3), so both have this mean energy; the odd ladder's
            // normalized ground state subtracts 2theta. Truncation error is
            // O(x^4) ~ 5e-9, below the binary32 EOS inversion tolerance.
            Real const x = Real(rotation) / Real(temperature);
            Real const xSquared = x * x;
            Real const mean = Real(temperature) *
                (((Real(1.0) - x / Real(3.0)) - xSquared / Real(45.0)) -
                    Real(Real(8.0) / Real(945.0)) * xSquared * x);
            energy += mean - Real(1.5) * Real(rotation);
            capacity +=
                (Real(1.0) + xSquared / Real(45.0)) + Real(Real(16.0) / Real(945.0)) * xSquared * x;
        }
        // 65 levels converge every state up to 6000 K; the H2+ ladder needs the
        // longer sum to stay converged up to the series switch.
        int const highestLevel = temperature > Real(6000.0) ? 128 : 64;
        for (int parity = 0; parity != 2 && temperature <= kRotorSeriesTemperature<Real>;
            ++parity) {
            Real partition(Real(0.0)), first(Real(0.0)), second(Real(0.0));
            for (int j = parity; j <= highestLevel; j += 2) {
                Real const level = rotation * static_cast<Real>(j * (j + 1) - 2 * parity);
                Real const weight = static_cast<Real>(2 * j + 1) * std::exp(-level / temperature);
                partition += Real(weight);
                first += Real(weight) * Real(level);
                second += Real(weight) * Real(level) * Real(level);
            }
            Real const mean = first / partition;
            Real const mixing(parity == 0 ? Real(0.25) : Real(0.75));
            energy += mixing * mean;
            capacity += mixing * (second / partition - mean * mean) /
                (Real(temperature) * Real(temperature));
        }
        Real const ratio = vibration / temperature;
        Real const occupation = std::exp(-ratio);
        Real const denominator = -std::expm1(-ratio);
        energy += Real(vibration * occupation / denominator);
        capacity += Real(ratio * ratio * occupation / (denominator * denominator));
        return {energy, capacity};
    }
}    // namespace detail

// The cold molecular cooling branches use published 10 K data; see
// primordial_rates.hpp. This is a supported-domain boundary, not a thermal floor.
template <class Real = float>
CUDA_EXPORT inline Real minimumTemperature(Network) {
    return Real(10.0);
}
// Both networks share the atomic ceiling. Molecular9 keeps all nine species
// through heating, dissociation and recombination; its molecular fits above
// their published ranges follow the documented continuations in
// primordial_rates.hpp and docs/molecular9_hot.md.
template <class Real = float>
CUDA_EXPORT inline Real maximumTemperature(Network) {
    return Real(1.0e8);
}

template <class Real>
CUDA_EXPORT inline Status thermalEnergyAtTemperature(StateT<Real> const& state,
    EnvironmentT<Real> const& environment, Real temperature, Real& energy, Real& heatCapacity) {
    Status const status = validateComposition(state, environment);
    if (status != Status::Success) {
        return status;
    }
    if (!finite(temperature) || temperature <= Real(0.0)) {
        return Status::InvalidInput;
    }
    if (temperature < minimumTemperature<Real>(environment.network) ||
        temperature > maximumTemperature<Real>(environment.network)) {
        return Status::UnsupportedState;
    }
    CompositionT<Real> const c = composition(state, environment);
    Real const translation = c.particles * Real(Real(1.5) * kBoltzmannEvT<Real>);
    energy = translation * Real(temperature);
    Real capacity = translation;
    if (environment.network == Network::Molecular9) {
        // H2: Boley2007 theta_r=85.4 K, theta_v=5987 K.
        // H2+: B0=Be-alpha_e/2, fundamental omega_e-2omega_exe from NIST
        // CCCBDB experimental diatomics (release22), hc/k=1.438776877 K cm.
        // https://cccbdb.nist.gov/expdiatomicsx.asp
        detail::MolecularModesT<Real> const neutral =
            detail::molecularModes(temperature, Real(85.4), Real(5987.0));
        detail::MolecularModesT<Real> const ion =
            detail::molecularModes(temperature, (Real(30.21) - Real(0.8425)) * Real(1.438776877),
                (Real(2321.7) - Real(132.4)) * Real(1.438776877));
        energy += Real(kBoltzmannEvT<Real>) *
            (c.molecularHydrogen * Real(neutral.energyKelvin) +
                c.molecularHydrogenPlus * Real(ion.energyKelvin));
        capacity += Real(kBoltzmannEvT<Real>) *
            (c.molecularHydrogen * Real(neutral.heatCapacity) +
                c.molecularHydrogenPlus * Real(ion.heatCapacity));
    }
    heatCapacity = capacity;
    if (!finite(energy) || !finite(heatCapacity) || heatCapacity <= Real(0.0)) {
        return Status::InvalidInput;
    }
    return Status::Success;
}
template <class Real>
CUDA_EXPORT inline Status thermalEnergyAtTemperature(StateT<Real> const& state,
    EnvironmentT<Real> const& environment, Real temperature, Real& energy) {
    Real result;
    Real capacity;
    Status const status =
        thermalEnergyAtTemperature(state, environment, temperature, result, capacity);
    if (status == Status::Success) {
        energy = result;
    }
    return status;
}

template <class Real>
CUDA_EXPORT inline Status evaluateEos(
    StateT<Real> const& state, EnvironmentT<Real> const& environment, EosT<Real>& result) {
    Status const status = validateComposition(state, environment);
    if (status != Status::Success) {
        return status;
    }
    if (!finite(state.thermalEnergy) || state.thermalEnergy <= Real(0.0)) {
        return Status::InvalidInput;
    }
    CompositionT<Real> const c = composition(state, environment);
    Real lower = minimumTemperature<Real>(environment.network);
    Real upper = maximumTemperature<Real>(environment.network);
    Real lowerEnergy, upperEnergy;
    Real capacity;
    Real const roundoff = Real(Real(4) * kScalarEpsilon<Real>) * state.thermalEnergy;
    if (environment.network == Network::Molecular9) {
        Real warmEnergy;
        thermalEnergyAtTemperature(state, environment, Real(100.0), warmEnergy, capacity);
        // Keep the established inversion bracket and endpoint rounding for every
        // previously supported state; only colder states need the new bracket.
        if (state.thermalEnergy >= warmEnergy - roundoff) {
            lower = Real(100.0);
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
    Real temperature =
        (state.thermalEnergy / (c.particles * Real(Real(1.5) * kBoltzmannEvT<Real>)));
    temperature = std::fmax(lower, std::fmin(upper, temperature));
    for (int iteration = 0; iteration != 80; ++iteration) {
        Real energy;
        thermalEnergyAtTemperature(state, environment, temperature, energy, capacity);
        Real const residual = energy - state.thermalEnergy;
        if (std::fabs(residual) <= Real(3) * kScalarEpsilon<Real> * state.thermalEnergy) {
            break;
        }
        if (residual > Real(0.0)) {
            upper = temperature;
        } else {
            lower = temperature;
        }
        Real const proposed = temperature - (residual / Real(capacity));
        Real const next =
            proposed > lower && proposed < upper ? proposed : Real(0.5) * (lower + upper);
        if (next == temperature) {
            break;
        }
        temperature = next;
    }
    Real energy;
    thermalEnergyAtTemperature(state, environment, temperature, energy, capacity);
    if (std::fabs((energy - state.thermalEnergy)) >
        Real(8) * kScalarEpsilon<Real> * state.thermalEnergy) {
        return Status::NonConvergence;
    }
    result.temperature = temperature;
    result.heatCapacity = capacity;
    result.pressurePerHydrogen = (c.particles * Real(kBoltzmannEvT<Real> * temperature));
    result.adiabaticIndex = Real(1.0) + result.pressurePerHydrogen / (temperature * capacity);
    result.meanMolecularWeight =
        ((Real(1.0) + Real(4.0) * Real(environment.heliumToHydrogenNumberRatio)) / c.particles);
    result.pressureEnergyRatio = (Real(result.pressurePerHydrogen) / state.thermalEnergy);
    return Status::Success;
}

template <class Real>
CUDA_EXPORT inline Status validateState(
    StateT<Real> const& state, EnvironmentT<Real> const& environment) {
    EosT<Real> result;
    return evaluateEos(state, environment, result);
}

}    // namespace primordial
