#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cosmology/background.hpp>
#include <cosmictiger/math/simulation_units.hpp>

namespace cosmology {

namespace detail {

    void validateBackgroundInterval(double finalConformalTime) {
        if (!std::isfinite(finalConformalTime) || finalConformalTime < 0.0) {
            throw std::invalid_argument(
                "Background conformal interval must be finite and nonnegative");
        }
    }

}    // namespace detail

SimulationBackgroundTable::SimulationBackgroundTable(
    FriedmannCosmology const& model, FriedmannState initialState, double finalConformalTime)
  : finalConformalTime_(finalConformalTime) {
    detail::validateBackgroundInterval(finalConformalTime);
    trajectory_.emplace(
        model.trajectory(initialState, finalConformalTime, CosmologicalTime::conformal));
}

SimulationBackgroundTable::SimulationBackgroundTable(double fixedScale, double finalConformalTime)
  : fixedScale_(fixedScale)
  , finalConformalTime_(finalConformalTime) {
    detail::validateBackgroundInterval(finalConformalTime);
    if (!std::isfinite(fixedScale) || fixedScale <= 0.0) {
        throw std::invalid_argument("Fixed background scale factor must be finite and positive");
    }
}

FriedmannAdvance SimulationBackgroundTable::sample(double conformalTime) const {
    double const tolerance = 32.0 * std::numeric_limits<double>::epsilon() * finalConformalTime_;
    if (!std::isfinite(conformalTime) || conformalTime < -tolerance ||
        conformalTime > finalConformalTime_ + tolerance) {
        throw std::out_of_range("Requested time lies outside the background history");
    }
    // Time subtraction at the observation epoch can land a few ulps outside
    // the interval. Preserve exact endpoint states without extrapolating.
    conformalTime = std::clamp(conformalTime, 0.0, finalConformalTime_);
    if (trajectory_) {
        return trajectory_->sample(conformalTime);
    }
    FriedmannAdvance result{};
    result.state.scaleFactor = fixedScale_;
    result.physicalTime = fixedScale_ * conformalTime;
    result.conformalTime = conformalTime;
    result.driftIntegral = conformalTime / fixedScale_;
    return result;
}

double SimulationBackgroundTable::finalConformalTime() const {
    return finalConformalTime_;
}

std::shared_ptr<SimulationBackgroundTable const> simulationBackground(double finalConformalTime) {
    detail::validateBackgroundInterval(finalConformalTime);
    auto const& options = getOptions();
    bool const fixed = !options.run.scenario.empty() && options.run.scenario != "cosmology";
    double const initialScale = fixed ? 1.0 : 1.0 / (1.0 + options.cosmology.startRedshift);
    BackgroundCosmologyParameters const parameters{
        units::normalizedHubble100(options) * options.cosmology.hubble, options.cosmology.omegaR,
        options.cosmology.omegaM, options.cosmology.omegaLambda, options.cosmology.omegaK};
    std::array<double, 8> const key{parameters.hubbleConstant, parameters.omegaRadiation,
        parameters.omegaMatter, parameters.omegaDarkEnergy, parameters.omegaCurvature, initialScale,
        finalConformalTime, fixed ? 1.0 : 0.0};
    static std::mutex mutex;
    static std::array<double, 8> cachedKey{};
    static std::shared_ptr<SimulationBackgroundTable const> cached;
    std::lock_guard<std::mutex> lock(mutex);
    if (!cached || key != cachedKey) {
        std::shared_ptr<SimulationBackgroundTable const> next;
        if (fixed) {
            next = std::make_shared<SimulationBackgroundTable>(initialScale, finalConformalTime);
        } else {
            FriedmannCosmology const model(parameters);
            next = std::make_shared<SimulationBackgroundTable>(
                model, model.state(initialScale, ExpansionBranch::expanding), finalConformalTime);
        }
        cachedKey = key;
        cached = std::move(next);
    }
    return cached;
}

}    // namespace cosmology
