/* CosmicTiger - outer long-force interval of the PMMM outer interval; BSL-1.0. */

#include <algorithm>
#include <cmath>
#include <limits>

#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/cosmology/evolution.hpp>
#include <cosmictiger/gravity/outer_interval.hpp>

/// Round the proposed long step upward to an equal subdivision of the time
/// remaining until the next output phase (or the end of the run).
double pmSelectOuterInterval(double accelerationTimestep, double expansionTimestep, double time,
    double outputInterval, double endTime) {
    if (!(outputInterval > 0.0) || !std::isfinite(outputInterval)) {
        THROW_ERROR("The output interval must be finite and positive\n");
    }
    double const remaining = endTime - time;
    if (cosmology::endTimeReached(time, endTime) || !(remaining > 0.0)) {
        return 0.0;
    }
    double const position = time / outputInterval;
    double const tolerance =
        32.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, position);
    double const nextBoundary =
        std::min((std::floor(position + tolerance) + 1.0) * outputInterval, endTime);
    double const toBoundary = nextBoundary - time;
    double const candidate = std::min({accelerationTimestep, expansionTimestep, toBoundary});
    if (!(candidate > 0.0) || !std::isfinite(candidate) || !(toBoundary > 0.0)) {
        THROW_ERROR("The selected outer PM interval must be finite and positive\n");
    }
    // H = R / max(1, floor(R / h)): the smallest equal subdivision >= h.
    // R is time remaining to the next output, h is the raw long-step proposal.
    // Roundoff in an exact-integer quotient must not discard a whole step.
    double const quotient = toBoundary / candidate;
    double const integral = std::round(quotient);
    double const quotientTolerance =
        8.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, quotient);
    double const steps = std::max(
        1.0, std::floor(std::abs(quotient - integral) <= quotientTolerance ? integral : quotient));
    double const interval = toBoundary / steps;
    if (!(interval > 0.0) || !std::isfinite(interval) || time + interval == time) {
        THROW_ERROR("The selected outer PM interval cannot advance the evolution clock\n");
    }
    return interval;
}
