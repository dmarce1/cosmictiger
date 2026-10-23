/* CosmicTiger - cosmological evolution controls; BSL-1.0. */

#include <cmath>
#include <stdexcept>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/cosmology/evolution.hpp>

namespace cosmology {

double configuredDuration(double initialScale) {
    auto const& options = getOptions();
    if (options.run.endTime > 0.0) {
        return options.run.endTime;
    }
    double const finalScale = 1.0 / (1.0 + options.cosmology.endRedshift);
    if (!std::isfinite(initialScale) || !(initialScale > 0.0) || !std::isfinite(finalScale) ||
        !(finalScale > initialScale)) {
        throw std::invalid_argument("cosmology.endRedshift must be below startRedshift when "
                                    "run.endTime=0; use a positive run.endTime to evolve "
                                    "through turnaround and contraction");
    }
    return cosmosConformalTime(initialScale, finalScale);
}

}    // namespace cosmology
