/* CosmicTiger - conservative fixed-mass gravity impulses; BSL-1.0. */

#pragma once

#include <cmath>

#include <cosmictiger/cuda/cuda_export.hpp>

namespace hopkins {

// Apply one signed HOLD kick to an already-closed MFM conserved anchor.
// The caller supplies finite state/impulse values and a positive finite mass.
// Its pair transfers must be settled first: gravity acts on the resulting
// momentum, and this helper must not also be used as a continuous hydro source.
// The kinetic-energy change leaves thermal energy unchanged to roundoff.
template <class Vector>
CUDA_EXPORT inline void applyGravityImpulse(
    double mass, Vector& momentum, double& totalEnergy, Vector const& deltaVelocity) noexcept {
    for (int dimension = 0; dimension < 3; ++dimension) {
        double const previous = momentum[dimension];
        double const next = ::fma(mass, deltaVelocity[dimension], previous);
        // Use the actual representable momentum change. An impulse too small
        // to change momentum must not inject energy through an unrounded dv.
        double const averageVelocity = 0.5 * (previous / mass) + 0.5 * (next / mass);
        totalEnergy = ::fma(next - previous, averageVelocity, totalEnergy);
        momentum[dimension] = next;
    }
}

}    // namespace hopkins
