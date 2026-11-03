/* CosmicTiger - GADGET-4 effective multiphase source equations; BSL-1.0. */
#pragma once

#include <cmath>
#include <limits>

#include <cosmictiger/cuda/cuda_export.hpp>

// Independent implementation of the published GADGET-4 source prescription.
// Reference: vrs/gadget4, commit 6fb393b5e25907f2ff06211f67d341c5e40e90d1,
// src/cooling_sfr/sfr_eos.cc (90-158, 182-214, 449-465).
// CGS boundary: proper g/cm^3, seconds, erg/g and erg/cm^3.
namespace springel_and_hernquist {

struct Parameters {
    double supernovaMassFraction{0.1};
    double evaporationFactor{1000.0};
    double starFormationTime{2.1e9 * 3.15576e7};
    double coldEnergy{};
    double supernovaEnergy{};
    double warmEnergy{};
    double adiabaticIndex{5.0 / 3.0};
};
struct Model {
    Parameters parameters{};
    double thresholdDensity{};
    double overdensityThreshold{};    // Comoving CGS baryonic mass density.
};
struct Equilibrium {
    double starFormationTime{};    // Unclamped; the source clamps AFTER computing x.
    double evaporationFactor{};
    double hotEnergy{};
    double coldFraction{};
    double hotFraction{};
    double effectiveEnergy{};
    double pressure{};
    double relaxationTime{};
    double specificStarFormationRate{};
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
    return detail::finite(parameters.supernovaMassFraction) &&
        parameters.supernovaMassFraction > 0.0 && parameters.supernovaMassFraction < 1.0 &&
        detail::finite(parameters.evaporationFactor) && parameters.evaporationFactor >= 0.0 &&
        detail::finite(parameters.starFormationTime) && parameters.starFormationTime > 0.0 &&
        detail::finite(parameters.coldEnergy) && parameters.coldEnergy > 0.0 &&
        detail::finite(parameters.supernovaEnergy) && parameters.supernovaEnergy > 0.0 &&
        detail::finite(parameters.warmEnergy) && parameters.warmEnergy > parameters.coldEnergy &&
        parameters.adiabaticIndex == 5.0 / 3.0;
}
// clang-format off
/*
 * Springel & Hernquist (2003), MNRAS 339, 289, https://arxiv.org/abs/astro-ph/0206393;
 * GADGET-4 source specialization pinned at the top of this file.
 *   ε_SN = β u_SN − (1−β)u_c. (Denominator in equation 16.)
 * Symbol directory:
 *   β → parameters.supernovaMassFraction (supernova mass fraction); u_SN → parameters.supernovaEnergy; u_c → parameters.coldEnergy
 *   ε_SN → return value.
 */
// clang-format on
CUDA_EXPORT inline double feedbackEnergy(Parameters const& parameters) {
    return parameters.supernovaMassFraction * parameters.supernovaEnergy -
        (1.0 - parameters.supernovaMassFraction) * parameters.coldEnergy;
}
// clang-format off
/*
 * Springel & Hernquist (2003), MNRAS 339, 289, https://arxiv.org/abs/astro-ph/0206393;
 * GADGET-4 source specialization pinned at the top of this file.
 *   A = A₀(ρ/ρ_th)^(−4/5), u_h = u_SN/(1+A) + u_c. (20), (11)
 * Symbol directory:
 *   ρ → density; ρ_th → model.thresholdDensity; A₀ → model.parameters.evaporationFactor
 *   A → evaporationFactor; u_SN,u_c → model.parameters.supernovaEnergy,coldEnergy; u_h → return value.
 */
// clang-format on
CUDA_EXPORT inline double hotEnergy(Model const& model, double density) {
    double const evaporationFactor =
        model.parameters.evaporationFactor * ::pow(density / model.thresholdDensity, -0.8);
    return model.parameters.supernovaEnergy / (1.0 + evaporationFactor) +
        model.parameters.coldEnergy;
}
// clang-format off
/*
 * Springel & Hernquist (2003), MNRAS 339, 289, https://arxiv.org/abs/astro-ph/0206393;
 * GADGET-4 source specialization pinned at the top of this file.
 *   ρ_th = [x_th/(1−x_th)²] [βu_SN−(1−β)u_c]/[t⋆⁰ Λ]. (23)
 * GADGET-4 calibration uses u_h = u_SN/A₀, x_th = (u_h−u₄)/(u_h−u_c).
 * Symbol directory:
 *   u_h → hotPhaseEnergy; x_th → coldPhaseFraction; Λ → coolingCoefficient; t⋆⁰ → parameters.starFormationTime
 *   u₄ → parameters.warmEnergy; ρ_th → thresholdDensity; βu_SN−(1−β)u_c → feedbackSpecificEnergy.
 */
// clang-format on
CUDA_EXPORT inline bool calibrate(
    Parameters const& parameters, double coolingCoefficient, Model& result) {
    if (!validParameters(parameters) || !(parameters.evaporationFactor > 0.0) ||
        !detail::finite(coolingCoefficient) || !(coolingCoefficient > 0.0)) {
        return false;
    }
    // GADGET-4 init_clouds deliberately uses uSN/A0, not the runtime hot-phase energy.
    double const hotPhaseEnergy = parameters.supernovaEnergy / parameters.evaporationFactor;
    double const coldPhaseFraction =
        (hotPhaseEnergy - parameters.warmEnergy) / (hotPhaseEnergy - parameters.coldEnergy);
    double const feedbackSpecificEnergy = feedbackEnergy(parameters);
    if (!(coldPhaseFraction > 0.0 && coldPhaseFraction < 1.0) || !(feedbackSpecificEnergy > 0.0)) {
        return false;
    }
    double const thresholdDensity = coldPhaseFraction / ::pow(1.0 - coldPhaseFraction, 2) *
        feedbackSpecificEnergy / (parameters.starFormationTime * coolingCoefficient);
    if (!(thresholdDensity > 0.0) || !detail::finite(thresholdDensity)) {
        return false;
    }
    result = {parameters, thresholdDensity, 0.0};
    return true;
}
// clang-format off
/*
 * Springel & Hernquist (2003), MNRAS 339, 289, https://arxiv.org/abs/astro-ph/0206393;
 * GADGET-4 source specialization pinned at the top of this file.
 *   x = 1 + 1/(2y) − √[1/y + 1/(4y²)]. (18)
 * Symbol directory:
 *   y → coolingToFeedbackRatio; x → return value.
 */
// clang-format on
CUDA_EXPORT inline double coldFraction(double coolingToFeedbackRatio) {
    if (!(coolingToFeedbackRatio > 0.0)) {
        return detail::invalid();
    }
    if (!detail::finite(coolingToFeedbackRatio)) {
        return 1.0;    // Positive infinite y has the same root as GADGET-4 arithmetic.
    }
    // Preserve the arithmetic of the official source for reference parity.
    return 1.0 + 1.0 / (2.0 * coolingToFeedbackRatio) -
        ::sqrt(1.0 / coolingToFeedbackRatio +
            1.0 / (4.0 * coolingToFeedbackRatio * coolingToFeedbackRatio));
}
// clang-format off
/*
 * Springel & Hernquist (2003), MNRAS 339, 289, https://arxiv.org/abs/astro-ph/0206393;
 * GADGET-4 source specialization pinned at the top of this file.
 *   t⋆ = t⋆⁰(ρ/ρ_th)^(−1/2), A = A₀(ρ/ρ_th)^(−4/5),           (21), (20)
 * u_h = u_SN/(1+A)+u_c, τ_h = t⋆(1−x)/[β(1+A)x],               (11), (12)
 * y = (t⋆/t_cool)u_h/[βu_SN−(1−β)u_c],                          (16)
 * u_eff = (1−x)u_h+xu_c, P_eff = (γ−1)ρu_eff.                   (19)
 * Derived specific stellar growth rate: ṁ⋆/m = (1−β)x/t⋆.
 * Symbol directory:
 *   t⋆ → starFormationTime; A → evaporationFactor; u_h → hotPhaseEnergy; x → coldPhaseFraction; 1−x → hotFraction
 *   t_cool → coolingTime; u_eff → effectiveInternalEnergy; P_eff → pressure; τ_h → relaxationTime
 *   ṁ⋆/m → specificStarFormationRate; ρ → density; β → parameters.supernovaMassFraction; γ → parameters.adiabaticIndex
 *   A₀,t⋆⁰,u_c,u_SN → parameters.evaporationFactor,parameters.starFormationTime,parameters.coldEnergy,parameters.supernovaEnergy.
 */
// clang-format on
CUDA_EXPORT inline bool equilibrium(
    Model const& model, double density, double coolingTime, Equilibrium& result) {
    auto const& parameters = model.parameters;
    if (!validParameters(parameters) || !(model.thresholdDensity > 0.0) ||
        !detail::finite(model.thresholdDensity) || !detail::finite(density) ||
        density < model.thresholdDensity || !detail::finite(coolingTime) || coolingTime < 0.0) {
        return false;
    }
    double const starFormationTime =
        ::sqrt(model.thresholdDensity / density) * parameters.starFormationTime;
    double const evaporationFactor =
        ::pow(density / model.thresholdDensity, -0.8) * parameters.evaporationFactor;
    double const hotPhaseEnergy =
        parameters.supernovaEnergy / (1.0 + evaporationFactor) + parameters.coldEnergy;
    double const feedbackSpecificEnergy = feedbackEnergy(parameters);
    if (!(feedbackSpecificEnergy > 0.0)) {
        return false;
    }
    // GADGET-4 returns tcool=0 for net heating. Its y->infinity limit is x=1.
    double const coldPhaseFraction = coolingTime == 0.0 ?
        1.0 :
        coldFraction(starFormationTime / coolingTime * hotPhaseEnergy / feedbackSpecificEnergy);
    double const hotFraction = 1.0 - coldPhaseFraction;
    double const effectiveInternalEnergy =
        hotPhaseEnergy * hotFraction + parameters.coldEnergy * coldPhaseFraction;
    double const pressure = (parameters.adiabaticIndex - 1.0) * density * effectiveInternalEnergy;
    double const relaxationTime = starFormationTime * hotFraction / coldPhaseFraction /
        (parameters.supernovaMassFraction * (1.0 + evaporationFactor));
    double const specificStarFormationRate =
        (1.0 - parameters.supernovaMassFraction) * coldPhaseFraction / starFormationTime;
    if (!(coldPhaseFraction >= 0.0 && coldPhaseFraction <= 1.0) ||
        !detail::finite(effectiveInternalEnergy) || !detail::finite(pressure) ||
        !(relaxationTime >= 0.0) || !detail::finite(specificStarFormationRate)) {
        return false;
    }
    result = {starFormationTime, evaporationFactor, hotPhaseEnergy, coldPhaseFraction, hotFraction,
        effectiveInternalEnergy, pressure, relaxationTime, specificStarFormationRate};
    return true;
}
// clang-format off
/*
 * Springel & Hernquist (2003), MNRAS 339, 289, https://arxiv.org/abs/astro-ph/0206393;
 * GADGET-4 source specialization pinned at the top of this file.
 *   t⋆ ← max(t⋆, Δt), τ_h = t⋆(1−x)/[β(1+A)x], ṁ⋆/m = (1−β)x/t⋆.
 * The clamp is the GADGET-4 source integration policy; relaxation is SH03 (12).
 * Symbol directory:
 *   Δt → interval; t⋆ → state.starFormationTime; x → state.coldFraction
 *   1−x → state.hotFraction; A → state.evaporationFactor; β → parameters.supernovaMassFraction
 *   τ_h → state.relaxationTime; ṁ⋆/m → state.specificStarFormationRate.
 */
// clang-format on
CUDA_EXPORT inline bool clampSourceTime(
    Parameters const& parameters, double interval, Equilibrium& state) {
    if (!detail::finite(interval) || interval < 0.0) {
        return false;
    }
    if (state.starFormationTime < interval) {
        state.starFormationTime = interval;
    }
    state.relaxationTime = state.starFormationTime * state.hotFraction / state.coldFraction /
        (parameters.supernovaMassFraction * (1.0 + state.evaporationFactor));
    state.specificStarFormationRate =
        (1.0 - parameters.supernovaMassFraction) * state.coldFraction / state.starFormationTime;
    return state.relaxationTime >= 0.0 && detail::finite(state.specificStarFormationRate);
}
// clang-format off
/*
 * Springel & Hernquist (2003), MNRAS 339, 289, https://arxiv.org/abs/astro-ph/0206393;
 * GADGET-4 source specialization pinned at the top of this file.
 *   p⋆ = 1 − exp[−(ṁ⋆/m) Δt].
 * Derived Poisson conversion probability for conversion of the whole gas particle.
 * Symbol directory:
 *   ṁ⋆/m → specificRate; Δt → interval; p⋆ → return value.
 */
// clang-format on
CUDA_EXPORT inline double starFormationProbability(double specificRate, double interval) {
    if (!detail::finite(interval) || interval < 0.0 || !detail::finite(specificRate) ||
        specificRate < 0.0) {
        return detail::invalid();
    }
    return 1.0 - ::exp(-specificRate * interval);
}
}    // namespace springel_and_hernquist
