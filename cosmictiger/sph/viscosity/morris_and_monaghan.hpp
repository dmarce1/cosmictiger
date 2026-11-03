/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/simd_math.hpp>

namespace morris_and_monaghan {

// clang-format off
/*
 * Morris & Monaghan (1997), JCP 136, 41–50, doi:10.1006/jcph.1997.5690,
 * equations (5), (6), (13):
 *   dα_a/dt = −(α_a − α*)/τ_a + S_a,
 *   τ_a = h_a/(C₁ c_a),  S_a = max(−∇·v_a, 0).
 * Derived backward-Euler update with frozen S_a and τ_a:
 *   α_aⁿ⁺¹ = α* + (α_aⁿ − α* + Δt S_a)/(1 + Δt/τ_a).
 * Symbol directory:
 *   α_aⁿ → bulkViscosityCoefficient; α* → minimumBulkViscosityCoefficient
 *   τ_a⁻¹ → inverseDecayTime; ∇·v_a → divergence; Δt → timeStep
 *   S_a → source; Δt/τ_a → decayStep; α_aⁿ⁺¹ → updated (return value).
 * All rates use the integration time unit. The paper uses α* = 0.1 and C₁ = 0.2.
 * The shifted form preserves the floor; a zero interval returns the input exactly.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real evolvedBulkViscosityCoefficient(Real bulkViscosityCoefficient,
    Real minimumBulkViscosityCoefficient, Real inverseDecayTime, Real divergence, Real timeStep) {
    Real const source = simdMax(-divergence, Real(0));
    Real const decayStep = timeStep * inverseDecayTime;
    // Shift by the floor so its preservation does not depend on cancellation.
    Real const updated = minimumBulkViscosityCoefficient +
        (bulkViscosityCoefficient - minimumBulkViscosityCoefficient + timeStep * source) /
            (Real(1) + decayStep);
    return simdSelect(timeStep == Real(0), bulkViscosityCoefficient, updated);
}

}    // namespace morris_and_monaghan
