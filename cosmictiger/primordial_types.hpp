/* CosmicTiger - primordial thermochemistry; GPL-2.0-or-later. */
#pragma once

#include <cmath>
#include <cstdint>

#include <cosmictiger/float_float.hpp>

#ifdef __CUDACC__
#define PRIMORDIAL_HD __host__ __device__
#else
#define PRIMORDIAL_HD
#endif

namespace primordial {
inline constexpr int kAbundanceCount = 6;
inline constexpr float kBoltzmannEv = 8.617333262145e-5f;    // eV K^-1.
inline constexpr float kElectronVolt = 1.602176634e-12f;     // erg.
inline constexpr unsigned kPhysicsVersion = 2;

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
    HydrogenPlus,
    HeliumPlus,
    HeliumDoublePlus,
    HydrogenMinus,
    MolecularHydrogen,
    MolecularHydrogenPlus
};

// s=c for c<=1/2; s=c-1 otherwise. The sign bit, including -0, selects
// the branch. Never reconstruct a minority species by subtracting decoded
// float c from one: the negative branch already stores that minority.
class SignedAbundance {
    float encoded_{};

public:
    PRIMORDIAL_HD SignedAbundance() = default;
    PRIMORDIAL_HD static SignedAbundance fromRaw(float encoded) {
        SignedAbundance result;
        result.encoded_ = encoded;
        return result;
    }
    PRIMORDIAL_HD static SignedAbundance fromMinority(float minority, bool upper) {
        return fromRaw(upper ? ::detail::floatFloatNegate(minority) : minority);
    }
    PRIMORDIAL_HD static SignedAbundance fromFraction(float fraction) {
        return fraction > 0.5f ? fromMinority(1.0f - fraction, true) : fromRaw(fraction);
    }
    PRIMORDIAL_HD float raw() const {
        return encoded_;
    }
    PRIMORDIAL_HD void setRaw(float encoded) {
        encoded_ = encoded;
    }
    PRIMORDIAL_HD bool upper() const {
        return (::detail::floatFloatBits(encoded_) & 0x80000000U) != 0;
    }
    PRIMORDIAL_HD FloatFloat value() const {
        return upper() ? FloatFloat(1.0f) + FloatFloat(encoded_) : FloatFloat(encoded_);
    }
    PRIMORDIAL_HD FloatFloat complement() const {
        return upper() ? FloatFloat(::detail::floatFloatNegate(encoded_)) :
                         FloatFloat(1.0f) - FloatFloat(encoded_);
    }
    PRIMORDIAL_HD float fraction() const {
        return value().toFloat();
    }
    PRIMORDIAL_HD float complementFraction() const {
        return complement().toFloat();
    }
};

struct State {
    // Fractional nuclei: Hp/H, Hep/He, Hepp/He, Hminus/H, 2H2/H, 2H2p/H.
    SignedAbundance abundances[kAbundanceCount]{};
    float thermalEnergy{};    // eV per H nucleus; excludes chemical binding energy.
};
struct WorkingState {
    FloatFloat abundances[kAbundanceCount]{};    // decoded fractional nuclei.
    FloatFloat thermalEnergy{};                  // eV per H nucleus.
};
struct Derivative {
    FloatFloat abundances[kAbundanceCount]{};    // fractional nuclei per second.
    FloatFloat thermalEnergy{};                  // eV per H nucleus per second.
};
struct Radiation {
    // s^-1 per absorber, optically thin or externally attenuated.
    float hydrogenIonization{};
    float heliumIonization{};
    float heliumPlusIonization{};
    float hydrogenMinusDetachment{};
    float molecularHydrogenDissociation{};        // H2 -> 2H, including LW.
    float molecularHydrogenIonization{};          // H2 -> H2+ + e.
    float molecularHydrogenPlusDissociation{};    // H2+ -> H + H+.
    float molecularHydrogenPlusIonization{};      // H2+ -> 2H+ + e.
    // Net thermal energy deposited, eV per absorber per second, excluding
    // thresholds already consumed by the photon. No assumed UV spectrum.
    float heatingPerHydrogen{};
    float heatingPerHelium{};
    float heatingPerHeliumPlus{};
    float heatingPerHydrogenMinus{};
    float heatingPerMolecularHydrogen{};
    float heatingPerMolecularHydrogenPlus{};
};
struct Environment {
    float nHydrogen{};    // proper H nuclei cm^-3, fixed during a source substep.
    float heliumToHydrogenNumberRatio{0.24f / (4.0f * 0.76f)};
    float cmbTemperature{2.7255f};    // K, supplied at the current cosmological epoch.
    Network network{Network::Atomic6};
    Radiation radiation{};
};
struct Composition {
    // All counts here are particles per H nucleus, including He species.
    FloatFloat hydrogen{}, hydrogenPlus{}, helium{}, heliumPlus{}, heliumDoublePlus{};
    FloatFloat hydrogenMinus{}, molecularHydrogen{}, molecularHydrogenPlus{};
    FloatFloat electrons{}, particles{};
};
struct Eos {
    float temperature{};
    float pressurePerHydrogen{};    // P/nH, eV per H nucleus.
    float heatCapacity{};           // de/dT, eV per H nucleus per K at fixed composition.
    float adiabaticIndex{};
    float meanMolecularWeight{};    // particle mass / mH, with mHe=4mH.
    float pressureEnergyRatio{};    // P/(rho*u).
};
PRIMORDIAL_HD inline bool finite(float value) {
    return ::detail::floatFloatFinite(value);
}
PRIMORDIAL_HD inline WorkingState workingState(State const& state) {
    WorkingState result;
    for (int i = 0; i < kAbundanceCount; ++i) {
        result.abundances[i] = state.abundances[i].value();
    }
    result.thermalEnergy = FloatFloat(state.thermalEnergy);
    return result;
}
PRIMORDIAL_HD inline Status composeState(WorkingState const& state, State& result) {
    for (int i = 0; i < kAbundanceCount; ++i) {
        FloatFloat const value = state.abundances[i];
        if (!value.isFinite() || value < FloatFloat(0.0f) || value > FloatFloat(1.0f)) {
            return Status::InvalidInput;
        }
        bool const upper = value > FloatFloat(0.5f);
        float const minority = (upper ? FloatFloat(1.0f) - value : value).toFloat();
        result.abundances[i] = SignedAbundance::fromMinority(minority, upper);
    }
    if (!state.thermalEnergy.isFinite() || state.thermalEnergy <= FloatFloat(0.0f)) {
        return Status::InvalidInput;
    }
    result.thermalEnergy = state.thermalEnergy.toFloat();
    return Status::Success;
}
PRIMORDIAL_HD inline Composition composition(
    WorkingState const& state, Environment const& environment) {
    auto const* x = state.abundances;
    FloatFloat const helium(environment.heliumToHydrogenNumberRatio);
    Composition c;
    c.hydrogenPlus = x[0];
    c.hydrogenMinus = x[3];
    c.molecularHydrogen = x[4] * FloatFloat(0.5f);
    c.molecularHydrogenPlus = x[5] * FloatFloat(0.5f);
    c.hydrogen = ((FloatFloat(1.0f) - x[0]) - x[3]) - (x[4] + x[5]);
    c.heliumPlus = helium * x[1];
    c.heliumDoublePlus = helium * x[2];
    c.helium = helium * ((FloatFloat(1.0f) - x[1]) - x[2]);
    c.electrons = (c.hydrogenPlus - c.hydrogenMinus) + c.molecularHydrogenPlus + c.heliumPlus +
        FloatFloat(2.0f) * c.heliumDoublePlus;
    c.particles =
        FloatFloat(1.0f) + helium - c.molecularHydrogen - c.molecularHydrogenPlus + c.electrons;
    return c;
}
PRIMORDIAL_HD inline Composition composition(State const& state, Environment const& environment) {
    return composition(workingState(state), environment);
}
PRIMORDIAL_HD inline Status validateComposition(
    WorkingState const& state, Environment const& environment) {
    if (!finite(environment.nHydrogen) || environment.nHydrogen < 0.0f ||
        !finite(environment.heliumToHydrogenNumberRatio) ||
        environment.heliumToHydrogenNumberRatio < 0.0f || !finite(environment.cmbTemperature) ||
        environment.cmbTemperature < 0.0f ||
        (environment.network != Network::Atomic6 && environment.network != Network::Molecular9)) {
        return Status::InvalidInput;
    }
    if (environment.nHydrogen > 1.0e4f) {
        return Status::UnsupportedState;
    }
    for (int i = 0; i < kAbundanceCount; ++i) {
        FloatFloat const value = state.abundances[i];
        if (!value.isFinite() || value < FloatFloat(0.0f) || value > FloatFloat(1.0f) ||
            (environment.network == Network::Atomic6 && i >= 3 && value != FloatFloat(0.0f))) {
            return Status::InvalidInput;
        }
    }
    if (state.abundances[1] + state.abundances[2] > FloatFloat(1.0f)) {
        return Status::InvalidInput;
    }
    Composition const c = composition(state, environment);
    if (c.hydrogen < FloatFloat(0.0f) || c.electrons < FloatFloat(0.0f) ||
        !c.particles.isFinite() || c.particles <= FloatFloat(0.0f)) {
        return Status::InvalidInput;
    }
    return Status::Success;
}
PRIMORDIAL_HD inline Status validateComposition(
    State const& state, Environment const& environment) {
    return validateComposition(workingState(state), environment);
}
}    // namespace primordial
