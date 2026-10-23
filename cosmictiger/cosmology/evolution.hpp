/* CosmicTiger - cosmological evolution controls; BSL-1.0. */

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace cosmology {

// Bound the fractional change in a over a conformal step, including the
// acceleration at a turnaround. adot and acceleration use physical time.
inline double expansionTimestep(
    double scale, double adot, double acceleration, double fraction = 0.01) {
    double const rate = std::abs(adot);
    double const curvature = adot * adot + scale * std::abs(acceleration);
    double const denominator = rate + std::sqrt(rate * rate + 2.0 * fraction * curvature);
    return denominator > 0.0 ? std::min(2.0 * fraction / denominator, 1.0e30) : 1.0e30;
}

inline bool endTimeReached(double time, double finalTime) {
    double const tolerance = 32.0 * std::numeric_limits<double>::epsilon() *
        std::max(std::abs(time), std::abs(finalTime));
    return time >= finalTime - tolerance;
}

// Explicit durations are elapsed conformal code time. Otherwise select the
// first, expanding occurrence of endRedshift.
double configuredDuration(double initialScale);

}    // namespace cosmology
