/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/sph/viscosity/price_etal.hpp>

namespace price_etal {

// clang-format off
/*
 * Price et al. (2018), PASA 35, e031, doi:10.1017/pasa.2018.25,
 * https://arxiv.org/abs/1702.03930, equations (49)–(50):
 *   dα_a/dt = −(α_a − α_loc,a)/τ_a,
 *   α_aⁿ⁺¹ = (α_aⁿ + α_loc,a Δt/τ_a)/(1 + Δt/τ_a).
 * An increase to α_loc,a is immediate, as specified before equation (49).
 * Symbol directory:
 *   α_aⁿ → bulkViscosityCoefficient; α_loc,a → targetBulkViscosityCoefficient
 *   Δt → timeStep; τ_a⁻¹ → inverseDecayTime; Δt/τ_a → decayStep
 *   α_aⁿ⁺¹ → return value; localBulkViscosityTarget is the target alias.
 */
// clang-format on
float evolvedBulkViscosityCoefficient(float bulkViscosityCoefficient,
    float targetBulkViscosityCoefficient, float inverseDecayTime, float timeStep) {
    float const localBulkViscosityTarget = targetBulkViscosityCoefficient;
    float const decayStep = timeStep * inverseDecayTime;
    return localBulkViscosityTarget > bulkViscosityCoefficient ?
        localBulkViscosityTarget :
        (bulkViscosityCoefficient + localBulkViscosityTarget * decayStep) / (1.0f + decayStep);
}

}    // namespace price_etal
