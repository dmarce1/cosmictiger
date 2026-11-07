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
 *   β → supernova mass fraction; u_SN → supernova energy; u_c → cold energy
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
 *   ρ → density; ρ_th → threshold density; A₀ → base evaporation factor
 *   A → evaporation factor; u_SN,u_c → supernova energy, cold energy; u_h → return value.
 */
// clang-format on
CUDA_EXPORT inline double hotEnergy(Model const& model, double ρ) {
    double const A = model.parameters.evaporationFactor * ::pow(ρ / model.thresholdDensity, -0.8);
    return model.parameters.supernovaEnergy / (1.0 + A) + model.parameters.coldEnergy;
}
// clang-format off
/*
 * Springel & Hernquist (2003), MNRAS 339, 289, https://arxiv.org/abs/astro-ph/0206393;
 * GADGET-4 source specialization pinned at the top of this file.
 *   ρ_th = [x_th/(1−x_th)²] [βu_SN−(1−β)u_c]/[t⋆⁰ Λ]. (23)
 * GADGET-4 calibration uses u_h = u_SN/A₀, x_th = (u_h−u₄)/(u_h−u_c).
 * Symbol directory:
 *   u_h → hot phase energy; x_th → cold phase fraction; Λ → cooling coefficient; t⋆⁰ → base star formation time
 *   u₄ → warm energy; ρ_th → threshold density; βu_SN−(1−β)u_c → feedback specific energy.
 *   Code names: ε_SN = βu_SN−(1−β)u_c.
 */
// clang-format on
CUDA_EXPORT inline bool calibrate(Parameters const& parameters, double Λ, Model& result) {
    if (!validParameters(parameters) || !(parameters.evaporationFactor > 0.0) ||
        !detail::finite(Λ) || !(Λ > 0.0)) {
        return false;
    }
    // GADGET-4 init_clouds deliberately uses uSN/A0, not the runtime hot-phase energy.
    double const u_h = parameters.supernovaEnergy / parameters.evaporationFactor;
    double const x_th = (u_h - parameters.warmEnergy) / (u_h - parameters.coldEnergy);
    double const ε_SN = feedbackEnergy(parameters);
    if (!(x_th > 0.0 && x_th < 1.0) || !(ε_SN > 0.0)) {
        return false;
    }
    double const ρ_th = x_th / ::pow(1.0 - x_th, 2) * ε_SN / (parameters.starFormationTime * Λ);
    if (!(ρ_th > 0.0) || !detail::finite(ρ_th)) {
        return false;
    }
    result = {parameters, ρ_th, 0.0};
    return true;
}
// clang-format off
/*
 * Springel & Hernquist (2003), MNRAS 339, 289, https://arxiv.org/abs/astro-ph/0206393;
 * GADGET-4 source specialization pinned at the top of this file.
 *   x = 1 + 1/(2y) − √[1/y + 1/(4y²)]. (18)
 * Symbol directory:
 *   y → cooling to feedback ratio; x → return value.
 */
// clang-format on
CUDA_EXPORT inline double coldFraction(double y) {
    if (!(y > 0.0)) {
        return detail::invalid();
    }
    if (!detail::finite(y)) {
        return 1.0;    // Positive infinite y has the same root as GADGET-4 arithmetic.
    }
    // Preserve the arithmetic of the official source for reference parity.
    return 1.0 + 1.0 / (2.0 * y) - ::sqrt(1.0 / y + 1.0 / (4.0 * y * y));
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
 *   t⋆ → star formation time; A → evaporation factor; u_h → hot phase energy; x → cold phase fraction; 1−x → hot fraction
 *   t_cool → cooling time; u_eff → effective internal energy; P_eff → pressure; τ_h → relaxation time
 *   ṁ⋆/m → specific star formation rate; ρ → density; β → supernova mass fraction; γ → adiabatic index
 *   A₀,t⋆⁰,u_c,u_SN → base evaporation factor, base star formation time, cold energy, supernova energy.
 *   Code names: t_star = t⋆; one_minus_x = 1−x; ṁ_star_over_m = ṁ⋆/m.
 */
// clang-format on
CUDA_EXPORT inline bool equilibrium(
    Model const& model, double ρ, double t_cool, Equilibrium& result) {
    auto const& parameters = model.parameters;
    if (!validParameters(parameters) || !(model.thresholdDensity > 0.0) ||
        !detail::finite(model.thresholdDensity) || !detail::finite(ρ) ||
        ρ < model.thresholdDensity || !detail::finite(t_cool) || t_cool < 0.0) {
        return false;
    }
    double const t_star = ::sqrt(model.thresholdDensity / ρ) * parameters.starFormationTime;
    double const A = ::pow(ρ / model.thresholdDensity, -0.8) * parameters.evaporationFactor;
    double const u_h = parameters.supernovaEnergy / (1.0 + A) + parameters.coldEnergy;
    double const feedbackSpecificEnergy = feedbackEnergy(parameters);
    if (!(feedbackSpecificEnergy > 0.0)) {
        return false;
    }
    // GADGET-4 returns tcool=0 for net heating. Its y->infinity limit is x=1.
    double const x =
        t_cool == 0.0 ? 1.0 : coldFraction(t_star / t_cool * u_h / feedbackSpecificEnergy);
    double const one_minus_x = 1.0 - x;
    double const u_eff = u_h * one_minus_x + parameters.coldEnergy * x;
    double const P_eff = (parameters.adiabaticIndex - 1.0) * ρ * u_eff;
    double const τ_h = t_star * one_minus_x / x / (parameters.supernovaMassFraction * (1.0 + A));
    double const ṁ_star_over_m = (1.0 - parameters.supernovaMassFraction) * x / t_star;
    if (!(x >= 0.0 && x <= 1.0) || !detail::finite(u_eff) || !detail::finite(P_eff) ||
        !(τ_h >= 0.0) || !detail::finite(ṁ_star_over_m)) {
        return false;
    }
    result = {t_star, A, u_h, x, one_minus_x, u_eff, P_eff, τ_h, ṁ_star_over_m};
    return true;
}
// clang-format off
/*
 * Springel & Hernquist (2003), MNRAS 339, 289, https://arxiv.org/abs/astro-ph/0206393;
 * GADGET-4 source specialization pinned at the top of this file.
 *   t⋆ ← max(t⋆, Δt), τ_h = t⋆(1−x)/[β(1+A)x], ṁ⋆/m = (1−β)x/t⋆.
 * The clamp is the GADGET-4 source integration policy; relaxation is SH03 (12).
 * Symbol directory:
 *   Δt → interval; t⋆ → star formation time; x → cold fraction
 *   1−x → hot fraction; A → evaporation factor; β → supernova mass fraction
 *   τ_h → relaxation time; ṁ⋆/m → specific star formation rate.
 */
// clang-format on
CUDA_EXPORT inline bool clampSourceTime(
    Parameters const& parameters, double Δt, Equilibrium& state) {
    if (!detail::finite(Δt) || Δt < 0.0) {
        return false;
    }
    if (state.starFormationTime < Δt) {
        state.starFormationTime = Δt;
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
 *   ṁ⋆/m → specific rate; Δt → interval; p⋆ → return value.
 *   Code names: ṁ_star_over_m = ṁ⋆/m.
 */
// clang-format on
CUDA_EXPORT inline double starFormationProbability(double ṁ_star_over_m, double Δt) {
    if (!detail::finite(Δt) || Δt < 0.0 || !detail::finite(ṁ_star_over_m) || ṁ_star_over_m < 0.0) {
        return detail::invalid();
    }
    return 1.0 - ::exp(-ṁ_star_over_m * Δt);
}
}    // namespace springel_and_hernquist
