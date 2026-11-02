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
    double beta{0.1};
    double evaporationFactor{1000.0};
    double starFormationTime{2.1e9 * 3.15576e7};
    double coldEnergy{};
    double supernovaEnergy{};
    double warmEnergy{};
    double gamma{5.0 / 3.0};
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
CUDA_EXPORT inline bool validParameters(Parameters const& p) {
    return detail::finite(p.beta) && p.beta > 0.0 && p.beta < 1.0 &&
        detail::finite(p.evaporationFactor) && p.evaporationFactor >= 0.0 &&
        detail::finite(p.starFormationTime) && p.starFormationTime > 0.0 &&
        detail::finite(p.coldEnergy) && p.coldEnergy > 0.0 && detail::finite(p.supernovaEnergy) &&
        p.supernovaEnergy > 0.0 && detail::finite(p.warmEnergy) && p.warmEnergy > p.coldEnergy &&
        p.gamma == 5.0 / 3.0;
}
CUDA_EXPORT inline double feedbackEnergy(Parameters const& p) {
    return p.beta * p.supernovaEnergy - (1.0 - p.beta) * p.coldEnergy;
}
CUDA_EXPORT inline double hotEnergy(Model const& model, double density) {
    double const evaporation =
        model.parameters.evaporationFactor * ::pow(density / model.thresholdDensity, -0.8);
    return model.parameters.supernovaEnergy / (1.0 + evaporation) + model.parameters.coldEnergy;
}
CUDA_EXPORT inline bool calibrate(Parameters const& p, double coefficient, Model& result) {
    if (!validParameters(p) || !(p.evaporationFactor > 0.0) || !detail::finite(coefficient) ||
        !(coefficient > 0.0)) {
        return false;
    }
    // GADGET-4 init_clouds deliberately uses uSN/A0, not the runtime hot energy.
    double const hot = p.supernovaEnergy / p.evaporationFactor;
    double const cold = (hot - p.warmEnergy) / (hot - p.coldEnergy);
    double const feedback = feedbackEnergy(p);
    if (!(cold > 0.0 && cold < 1.0) || !(feedback > 0.0)) {
        return false;
    }
    double const threshold =
        cold / ::pow(1.0 - cold, 2) * feedback / (p.starFormationTime * coefficient);
    if (!(threshold > 0.0) || !detail::finite(threshold)) {
        return false;
    }
    result = {p, threshold, 0.0};
    return true;
}
CUDA_EXPORT inline double coldFraction(double ratio) {
    if (!(ratio > 0.0)) {
        return detail::invalid();
    }
    if (!detail::finite(ratio)) {
        return 1.0;    // Positive infinite y has the same root as GADGET-4 arithmetic.
    }
    // Preserve the arithmetic of the official source for reference parity.
    return 1.0 + 1.0 / (2.0 * ratio) - ::sqrt(1.0 / ratio + 1.0 / (4.0 * ratio * ratio));
}
CUDA_EXPORT inline bool equilibrium(
    Model const& model, double density, double coolingTime, Equilibrium& result) {
    auto const& p = model.parameters;
    if (!validParameters(p) || !(model.thresholdDensity > 0.0) ||
        !detail::finite(model.thresholdDensity) || !detail::finite(density) ||
        density < model.thresholdDensity || !detail::finite(coolingTime) || coolingTime < 0.0) {
        return false;
    }
    double const time = ::sqrt(model.thresholdDensity / density) * p.starFormationTime;
    double const evaporation = ::pow(density / model.thresholdDensity, -0.8) * p.evaporationFactor;
    double const hot = p.supernovaEnergy / (1.0 + evaporation) + p.coldEnergy;
    double const feedback = feedbackEnergy(p);
    if (!(feedback > 0.0)) {
        return false;
    }
    // GADGET-4 returns tcool=0 for net heating. Its y->infinity limit is x=1.
    double const cold =
        coolingTime == 0.0 ? 1.0 : coldFraction(time / coolingTime * hot / feedback);
    double const hotFraction = 1.0 - cold;
    double const effective = hot * hotFraction + p.coldEnergy * cold;
    double const pressure = (p.gamma - 1.0) * density * effective;
    double const relaxation = time * hotFraction / cold / (p.beta * (1.0 + evaporation));
    double const rate = (1.0 - p.beta) * cold / time;
    if (!(cold >= 0.0 && cold <= 1.0) || !detail::finite(effective) || !detail::finite(pressure) ||
        !(relaxation >= 0.0) || !detail::finite(rate)) {
        return false;
    }
    result = {time, evaporation, hot, cold, hotFraction, effective, pressure, relaxation, rate};
    return true;
}
CUDA_EXPORT inline bool clampSourceTime(Parameters const& p, double interval, Equilibrium& state) {
    if (!detail::finite(interval) || interval < 0.0) {
        return false;
    }
    if (state.starFormationTime < interval) {
        state.starFormationTime = interval;
    }
    state.relaxationTime = state.starFormationTime * state.hotFraction / state.coldFraction /
        (p.beta * (1.0 + state.evaporationFactor));
    state.specificStarFormationRate = (1.0 - p.beta) * state.coldFraction / state.starFormationTime;
    return state.relaxationTime >= 0.0 && detail::finite(state.specificStarFormationRate);
}
CUDA_EXPORT inline double starFormationProbability(double specificRate, double interval) {
    if (!detail::finite(interval) || interval < 0.0 || !detail::finite(specificRate) ||
        specificRate < 0.0) {
        return detail::invalid();
    }
    return 1.0 - ::exp(-specificRate * interval);
}
}    // namespace springel_and_hernquist
