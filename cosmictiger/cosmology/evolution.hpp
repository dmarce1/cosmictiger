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

// Canonical momenta p = a v_pec stay constant during a cosmological drift.
// For driftKinetic = sum(m p^2 / 2), integrate K da exactly, including
// contraction. The difference form avoids subtracting nearby inverse scales.
inline double kineticExpansionWork(double initialScale, double finalScale, double driftKinetic) {
    return (finalScale - initialScale) * driftKinetic / (initialScale * finalScale);
}

// All terms in the Layzer-Irvine residual and its normalization use the same
// scale: a(K + W + U) + sourceWork + expansionWork - referenceEnergy.
inline double normalizedEnergyError(double scale, double kinetic, double potential, double thermal,
    double sourceWork, double expansionWork, double referenceEnergy) {
    double const residual =
        scale * (kinetic + potential + thermal) + sourceWork + expansionWork - referenceEnergy;
    double const norm =
        std::max(std::abs(scale) * (std::abs(kinetic) + std::abs(potential) + std::abs(thermal)) +
                std::abs(sourceWork) + std::abs(expansionWork),
            1.0e-30);
    return residual / norm;
}

// Explicit durations are elapsed conformal code time. Otherwise select the
// first, expanding occurrence of endRedshift.
double configuredDuration(double initialScale);

}    // namespace cosmology
