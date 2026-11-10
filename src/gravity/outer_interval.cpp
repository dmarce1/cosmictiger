/* CosmicTiger - outer long-force interval of the PMMM outer interval; BSL-1.0. */

#include <algorithm>
#include <cmath>
#include <limits>

#include <cosmictiger/core/stack_trace.hpp>
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/cosmology/evolution.hpp>
#include <cosmictiger/gravity/outer_interval.hpp>

/// Clamp the proposed mesh interval to the end time and next coarse output boundary without
/// tiny residual steps.
double pmSelectOuterInterval(double accelerationTimestep, double expansionTimestep, double time,
    double coarseInterval, double endTime) {
    if (!(coarseInterval > 0.0) || !std::isfinite(coarseInterval)) {
        THROW_ERROR("The coarse output interval must be finite and positive\n");
    }
    double const remaining = endTime - time;
    if (cosmology::endTimeReached(time, endTime) || !(remaining > 0.0)) {
        return 0.0;
    }
    // Outputs and analysis stay on the nominal coarse clock: an interval never
    // crosses the next coarse boundary, and lands exactly on it when it would.
    double const position = time / coarseInterval;
    double const tolerance =
        32.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, position);
    double const nextBoundary = (std::floor(position + tolerance) + 1.0) * coarseInterval;
    double interval = std::min({accelerationTimestep, expansionTimestep, coarseInterval});
    interval = std::min(interval, remaining);
    double const toBoundary = nextBoundary - time;
    if (toBoundary > 0.0 && toBoundary < remaining) {
        interval = std::min(interval, toBoundary);
    }
    if (!(interval > 0.0) || !std::isfinite(interval) || time + interval == time) {
        THROW_ERROR("The selected outer PM interval cannot advance the evolution clock\n");
    }
    // Absorb a sliver that would otherwise leave a tiny final interval.
    double const slack = 1.0e-9 * coarseInterval;
    if (toBoundary > interval && toBoundary - interval < slack) {
        interval = toBoundary;
    } else if (remaining > interval && remaining - interval < slack) {
        interval = remaining;
    }
    return interval;
}
