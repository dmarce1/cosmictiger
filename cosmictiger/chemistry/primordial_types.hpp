/* CosmicTiger - primordial thermochemistry; BSL-1.0. */
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cosmictiger/cuda/cuda_export.hpp>

namespace primordial {
inline constexpr int kAbundanceCount = 9;
inline constexpr int kInitialFractionCount = 6;
template <class Real>
inline constexpr Real kDefaultHeliumToHydrogenT = Real(0.24) / (Real(4) * Real(0.76));
template <class Real>
inline constexpr Real kBoltzmannEvT = Real(8.617333262145e-5);    // eV K^-1.
template <class Real>
inline constexpr Real kElectronVoltT = Real(1.602176634e-12);    // erg.
inline constexpr float kBoltzmannEv = kBoltzmannEvT<float>;
inline constexpr float kElectronVolt = kElectronVoltT<float>;
inline constexpr unsigned kPhysicsVersion = 3;
template <class Real>
inline constexpr Real kScalarEpsilon = std::numeric_limits<Real>::epsilon();

enum class Network { Atomic6, Molecular9 };
enum class Status {
    Success,
    InvalidInput,
    UnsupportedState,
    NonConvergence,
    StepLimit,
    BackendUnavailable,
    BackendFailure
};
enum class Species {
    Hydrogen,
    HydrogenPlus,
    Helium,
    HeliumPlus,
    HeliumDoublePlus,
    Electron,
    HydrogenMinus,
    MolecularHydrogen,
    MolecularHydrogenPlus
};

template <class Real>
struct StateT {
    // Direct particle counts per H nucleus, in Species order. Neutral species
    // and electrons remain independent stored values, as in Anninos et al.
    Real abundances[kAbundanceCount]{Real(1), Real(0), kDefaultHeliumToHydrogenT<Real>};
    Real thermalEnergy{};    // eV per H nucleus; excludes chemical binding energy.
};
template <class Real>
struct DerivativeT {
    Real abundances[kAbundanceCount]{};    // particles per H nucleus per second.
    Real thermalEnergy{};                  // eV per H nucleus per second.
};
template <class Real>
struct RadiationT {
    // s^-1 per absorber, optically thin or externally attenuated.
    Real hydrogenIonization{};
    Real heliumIonization{};
    Real heliumPlusIonization{};
    Real hydrogenMinusDetachment{};
    Real molecularHydrogenDissociation{};        // H2 -> 2H, including LW.
    Real molecularHydrogenIonization{};          // H2 -> H2+ + e.
    Real molecularHydrogenPlusDissociation{};    // H2+ -> H + H+.
    Real molecularHydrogenPlusIonization{};      // H2+ -> 2H+ + e.
    // Net thermal energy deposited, eV per absorber per second, excluding
    // thresholds already consumed by the photon. No assumed UV spectrum.
    Real heatingPerHydrogen{};
    Real heatingPerHelium{};
    Real heatingPerHeliumPlus{};
    Real heatingPerHydrogenMinus{};
    Real heatingPerMolecularHydrogen{};
    Real heatingPerMolecularHydrogenPlus{};
};
template <class Real>
struct EnvironmentT {
    Real nHydrogen{};    // proper H nuclei cm^-3, fixed during a source substep.
    Real heliumToHydrogenNumberRatio{kDefaultHeliumToHydrogenT<Real>};
    Real cmbTemperature{Real(2.7255)};    // K, supplied at the current cosmological epoch.
    Network network{Network::Atomic6};
    RadiationT<Real> radiation{};
};
template <class Real>
struct CompositionT {
    // All counts here are particles per H nucleus, including He species.
    Real hydrogen{}, hydrogenPlus{}, helium{}, heliumPlus{}, heliumDoublePlus{};
    Real hydrogenMinus{}, molecularHydrogen{}, molecularHydrogenPlus{};
    Real electrons{}, particles{};
};
template <class Real>
struct EosT {
    Real temperature{};
    Real pressurePerHydrogen{};    // P/nH, eV per H nucleus.
    Real heatCapacity{};           // de/dT, eV per H nucleus per K at fixed composition.
    Real adiabaticIndex{};
    Real meanMolecularWeight{};    // particle mass / mH, with mHe=4mH.
    Real pressureEnergyRatio{};    // P/(rho*u).
};
using State = StateT<float>;
using Derivative = DerivativeT<float>;
using Radiation = RadiationT<float>;
using Environment = EnvironmentT<float>;
using Composition = CompositionT<float>;
using Eos = EosT<float>;
// Inspect IEEE exponent bits because fast-math builds may fold isfinite away.
template <class Real>
CUDA_EXPORT inline bool finite(Real value) {
    if constexpr (std::is_same_v<Real, float>) {
        union Bits {
            float value;
            uint32_t bits;
        } data{value};
        return (data.bits & 0x7f800000U) != 0x7f800000U;
    } else if constexpr (std::is_same_v<Real, double>) {
        union Bits {
            double value;
            uint64_t bits;
        } data{value};
        return (data.bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
    } else {
        using std::isfinite;
        return isfinite(value);
    }
}
template <class Real>
CUDA_EXPORT inline CompositionT<Real> composition(
    StateT<Real> const& state, EnvironmentT<Real> const&) {
    auto const* x = state.abundances;
    CompositionT<Real> c;
    c.hydrogen = x[0];
    c.hydrogenPlus = x[1];
    c.helium = x[2];
    c.heliumPlus = x[3];
    c.heliumDoublePlus = x[4];
    c.electrons = x[5];
    c.hydrogenMinus = x[6];
    c.molecularHydrogen = x[7];
    c.molecularHydrogenPlus = x[8];
    for (int species = 0; species != kAbundanceCount; ++species) {
        c.particles += x[species];
    }
    return c;
}

template <class Real>
CUDA_EXPORT inline Status validateComposition(
    StateT<Real> const& state, EnvironmentT<Real> const& environment) {
    if (!finite(environment.nHydrogen) || environment.nHydrogen < Real(0) ||
        !finite(environment.heliumToHydrogenNumberRatio) ||
        environment.heliumToHydrogenNumberRatio < Real(0) || !finite(environment.cmbTemperature) ||
        environment.cmbTemperature < Real(0) ||
        (environment.network != Network::Atomic6 && environment.network != Network::Molecular9)) {
        return Status::InvalidInput;
    }
    if (environment.nHydrogen > Real(1.e4)) {
        return Status::UnsupportedState;
    }
    for (int species = 0; species != kAbundanceCount; ++species) {
        Real const value = state.abundances[species];
        if (!finite(value) || value < Real(0) ||
            (environment.network == Network::Atomic6 && species >= 6 && value != Real(0))) {
            return Status::InvalidInput;
        }
    }
    CompositionT<Real> const c = composition(state, environment);
    Real const tolerance = Real(32) * kScalarEpsilon<Real>;
    Real const hydrogen = c.hydrogen + c.hydrogenPlus + c.hydrogenMinus +
        Real(2) * (c.molecularHydrogen + c.molecularHydrogenPlus);
    Real const helium = c.helium + c.heliumPlus + c.heliumDoublePlus;
    Real const positiveCharge =
        c.hydrogenPlus + c.heliumPlus + Real(2) * c.heliumDoublePlus + c.molecularHydrogenPlus;
    Real const negativeCharge = c.electrons + c.hydrogenMinus;
    if (!finite(c.particles) || c.particles <= Real(0) ||
        std::fabs(hydrogen - Real(1)) > tolerance * std::fmax(Real(1), hydrogen) ||
        std::fabs(helium - environment.heliumToHydrogenNumberRatio) >
            tolerance * std::fmax(helium, environment.heliumToHydrogenNumberRatio) ||
        std::fabs(positiveCharge - negativeCharge) >
            tolerance * std::fmax(positiveCharge, negativeCharge)) {
        return Status::InvalidInput;
    }
    return Status::Success;
}

// Initial-condition input retains the six convenient fractional nuclei:
// H+/H, He+/He, He++/He, H-/H, 2H2/H, 2H2+/H. This one-time conversion does
// not select an evolving variable basis or encode a complementary fraction.
template <class Real>
CUDA_EXPORT inline Status initializeState(StateT<Real>& state,
    EnvironmentT<Real> const& environment, Real const (&fractions)[kInitialFractionCount]) {
    for (Real fraction : fractions) {
        if (!finite(fraction) || fraction < Real(0) || fraction > Real(1)) {
            return Status::InvalidInput;
        }
    }
    Real* x = state.abundances;
    x[0] = Real(1) - ((fractions[0] + fractions[3]) + (fractions[4] + fractions[5]));
    x[1] = fractions[0];
    x[2] = environment.heliumToHydrogenNumberRatio * (Real(1) - (fractions[1] + fractions[2]));
    x[3] = environment.heliumToHydrogenNumberRatio * fractions[1];
    x[4] = environment.heliumToHydrogenNumberRatio * fractions[2];
    x[6] = fractions[3];
    x[7] = Real(0.5) * fractions[4];
    x[8] = Real(0.5) * fractions[5];
    x[5] = (x[1] - x[6]) + x[3] + Real(2) * x[4] + x[8];
    return validateComposition(state, environment);
}

template <class Real>
CUDA_EXPORT inline Status initializeState(
    StateT<Real>& state, EnvironmentT<Real> const& environment) {
    Real const fractions[kInitialFractionCount]{};
    return initializeState(state, environment, fractions);
}
}    // namespace primordial
