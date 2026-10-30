/* CosmicTiger - Springel and Hernquist (2003) multiphase model; BSL-1.0. */
#pragma once

#include <cmath>
#include <limits>

#include <cosmictiger/cuda/cuda_export.hpp>

// Equation numbers refer to the published paper, MNRAS 339, 289 (2003):
// https://doi.org/10.1046/j.1365-8711.2003.06206.x . Densities are proper
// g/cm^3, times seconds, specific energies erg/g, pressures erg/cm^3.
namespace springel_and_hernquist {

struct Parameters {
    double beta{0.1};
    double evaporationFactor{1000.0};
    double starFormationTime{2.1e9 * 31557600.0};
    double coldEnergy{};
    double supernovaEnergy{};
    double warmEnergy{};
    double gamma{5.0 / 3.0};
};

struct Model {
    Parameters parameters{};
    double thresholdDensity{};
};

struct Equilibrium {
    double starFormationTime{};
    double evaporationFactor{};
    double hotEnergy{};
    double coldFraction{};
    // Retain the small hot fraction separately when coldFraction rounds to 1.
    double hotFraction{};
    double effectiveEnergy{};
    double pressure{};
    double relaxationTime{};
    double specificStarFormationRate{};    // Long-lived stellar mass / gas mass / s.
};

namespace detail {
    CUDA_EXPORT inline bool finite(double value) {
        return value == value && ::fabs(value) <= std::numeric_limits<double>::max();
    }
    CUDA_EXPORT inline double invalid() {
        return std::numeric_limits<double>::quiet_NaN();
    }
}    // namespace detail

CUDA_EXPORT inline bool validParameters(Parameters const& parameters) {
    return detail::finite(parameters.beta) && parameters.beta > 0.0 && parameters.beta < 1.0 &&
        detail::finite(parameters.evaporationFactor) && parameters.evaporationFactor >= 0.0 &&
        detail::finite(parameters.starFormationTime) && parameters.starFormationTime > 0.0 &&
        detail::finite(parameters.coldEnergy) && parameters.coldEnergy >= 0.0 &&
        detail::finite(parameters.supernovaEnergy) && parameters.supernovaEnergy > 0.0 &&
        detail::finite(parameters.warmEnergy) && parameters.warmEnergy > parameters.coldEnergy &&
        detail::finite(parameters.gamma) && parameters.gamma > 1.0;
}

// Formula transcriptions use the paper's symbols; public interfaces retain
// descriptive names. The equilibrium closure assumes density-squared cooling.
// NOLINTBEGIN(readability-identifier-naming)

CUDA_EXPORT inline double feedbackEnergy(Parameters const& parameters) {
    // Equations (14), (16): net energy per unit mass consumed from cold clouds.
    double const β = parameters.beta;
    double const u_SN = parameters.supernovaEnergy;
    double const u_c = parameters.coldEnergy;
    return β * u_SN - (1.0 - β) * u_c;
}

CUDA_EXPORT inline double hotEnergy(Model const& model, double density) {
    // Equations (20), (11).
    double const ρ = density;
    double const ρ_th = model.thresholdDensity;
    double const A = model.parameters.evaporationFactor * ::pow(ρ / ρ_th, -0.8);
    return model.parameters.coldEnergy + model.parameters.supernovaEnergy / (1.0 + A);
}

CUDA_EXPORT inline bool calibrate(
    Parameters const& parameters, double coolingCoefficient, Model& result) {
    // Equation (23) with exact equation (11), rather than its u_SN/A0
    // approximation. coolingCoefficient = Lambda_net(rho,u_h0)/rho^2,
    // in erg cm^3 g^-2 s^-1, and must be independent of density.
    if (!validParameters(parameters) || !detail::finite(coolingCoefficient) ||
        coolingCoefficient <= 0.0) {
        return false;
    }
    double const u_c = parameters.coldEnergy;
    double const u_h0 = u_c + parameters.supernovaEnergy / (1.0 + parameters.evaporationFactor);
    double const u_4 = parameters.warmEnergy;
    double const Q = feedbackEnergy(parameters);
    if (!(u_h0 > u_4) || !(Q > 0.0)) {
        return false;
    }
    double const x_th = (u_h0 - u_4) / (u_h0 - u_c);
    double const h_th = (u_4 - u_c) / (u_h0 - u_c);
    double const ρ_th =
        (x_th / h_th) / h_th * (Q / parameters.starFormationTime / coolingCoefficient);
    if (!detail::finite(ρ_th) || ρ_th <= 0.0) {
        return false;
    }
    result = {parameters, ρ_th};
    return true;
}

CUDA_EXPORT inline double coldFraction(double coolingRatio) {
    // Equation (18), rationalized to avoid cancellation as y -> 0.
    double const y = coolingRatio;
    if (!detail::finite(y) || y < 0.0) {
        return detail::invalid();
    }
    if (y <= 1.0) {
        return (2.0 * y) / (1.0 + 2.0 * y + ::sqrt(1.0 + 4.0 * y));
    }
    double const r = 1.0 / ::sqrt(y);
    return 1.0 - r / (::sqrt(1.0 + 0.25 / y) + 0.5 * r);
}

CUDA_EXPORT inline double hotFraction(double coolingRatio) {
    // Complement of equation (18), without subtracting a rounded x from 1.
    double const y = coolingRatio;
    if (!detail::finite(y) || y < 0.0) {
        return detail::invalid();
    }
    if (y <= 1.0) {
        return 2.0 / (1.0 + ::sqrt(1.0 + 4.0 * y));
    }
    double const r = 1.0 / ::sqrt(y);
    return r / (::sqrt(1.0 + 0.25 / y) + 0.5 * r);
}

CUDA_EXPORT inline bool equilibrium(
    Model const& model, double density, double coolingCoefficient, Equilibrium& result) {
    if (!validParameters(model.parameters) || !detail::finite(model.thresholdDensity) ||
        model.thresholdDensity <= 0.0 || !detail::finite(density) ||
        density < model.thresholdDensity || !detail::finite(coolingCoefficient) ||
        coolingCoefficient <= 0.0) {
        return false;
    }
    Parameters const& parameters = model.parameters;
    double const ρ = density;
    double const ρ_th = model.thresholdDensity;
    double const β = parameters.beta;
    double const A = parameters.evaporationFactor * ::pow(ρ / ρ_th, -0.8);                // (20)
    double const t_star = parameters.starFormationTime * ::sqrt(ρ_th / ρ);                // (21)
    double const u_h = parameters.coldEnergy + parameters.supernovaEnergy / (1.0 + A);    // (11)
    double const Q = feedbackEnergy(parameters);
    double const y = (t_star * ρ) * (coolingCoefficient / Q);    // (16)
    if (!(Q > 0.0) || !detail::finite(y) || y <= 0.0) {
        return false;
    }
    double const x = coldFraction(y);
    double const h = hotFraction(y);
    double const u_eff = h * u_h + x * parameters.coldEnergy;     // (19)
    double const P_eff = (parameters.gamma - 1.0) * ρ * u_eff;    // (19)
    double const τ_h = t_star * h / (x * β * (1.0 + A));          // (12)
    double const s = (1.0 - β) * x / t_star;                      // (1)
    if (!detail::finite(u_eff) || !detail::finite(P_eff) || !detail::finite(τ_h) ||
        !detail::finite(s) || s <= 0.0) {
        return false;
    }
    result = {t_star, A, u_h, x, h, u_eff, P_eff, τ_h, s};
    return true;
}

CUDA_EXPORT inline double starFormationProbability(Equilibrium const& state, double timestep) {
    // Equation (39), N_g=1: an event converts the entire remaining particle.
    double const Δt = timestep;
    double const s = state.specificStarFormationRate;
    if (!detail::finite(Δt) || Δt < 0.0 || !detail::finite(s) || s < 0.0) {
        return detail::invalid();
    }
    return -::expm1(-s * Δt);
}

CUDA_EXPORT inline double enrichmentIncrement(
    Equilibrium const& state, double timestep, double yield) {
    // Equation (40): yield is metal mass per mass locked in long-lived stars.
    double const Δt = timestep;
    double const y_star = yield;
    if (!detail::finite(Δt) || Δt < 0.0 || !detail::finite(y_star) || y_star < 0.0 ||
        !detail::finite(state.specificStarFormationRate) || state.specificStarFormationRate < 0.0) {
        return detail::invalid();
    }
    return y_star * state.specificStarFormationRate * Δt;
}

CUDA_EXPORT inline double windProbability(
    Equilibrium const& state, double timestep, double massLoading) {
    // Equation (41).
    double const η = massLoading;
    double const Δt = timestep;
    if (!detail::finite(η) || η < 0.0 || !detail::finite(Δt) || Δt < 0.0 ||
        !detail::finite(state.specificStarFormationRate) || state.specificStarFormationRate < 0.0) {
        return detail::invalid();
    }
    return -::expm1(-η * state.specificStarFormationRate * Δt);
}

CUDA_EXPORT inline double windSpeed(
    Parameters const& parameters, double massLoading, double energyFraction) {
    // Equation (28), cm/s. Published section 4.1 adds this kinetic energy to
    // the unchanged thermal channel: total injection is (1+chi)*epsilon_SN.
    double const η = massLoading;
    double const χ = energyFraction;
    double const β = parameters.beta;
    double const u_SN = parameters.supernovaEnergy;
    if (!validParameters(parameters) || !detail::finite(η) || η <= 0.0 || !detail::finite(χ) ||
        χ < 0.0) {
        return detail::invalid();
    }
    return ::sqrt(2.0 * β * χ * u_SN / (η * (1.0 - β)));
}

// NOLINTEND(readability-identifier-naming)
}    // namespace springel_and_hernquist
