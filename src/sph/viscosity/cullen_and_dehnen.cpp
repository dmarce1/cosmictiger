/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cmath>

#include <cosmictiger/sph/viscosity/cullen_and_dehnen.hpp>

namespace cullen_and_dehnen {

// clang-format off
/*
 * Cullen & Dehnen (2010), equation (A15), https://arxiv.org/abs/1006.1524:
 *   α_iⁿ⁺¹ = α_loc,i                                      if α_iⁿ < α_loc,i,
 *   α_iⁿ⁺¹ = α_loc,i + (α_iⁿ − α_loc,i) exp(−Δt/τ_i)       otherwise.
 * Symbol directory:
 *   α_iⁿ → bulkViscosityCoefficient/currentBulkViscosityCoefficient
 *   α_loc,i → targetBulkViscosityCoefficient/localBulkViscosityTarget
 *   Δt → timeStep/elapsedTime; τ_i⁻¹ → inverseDecayTime; α_iⁿ⁺¹ → return value.
 */
// clang-format on
float evolvedBulkViscosityCoefficient(float bulkViscosityCoefficient,
    float targetBulkViscosityCoefficient, float inverseDecayTime, float timeStep) {
    float const currentBulkViscosityCoefficient = bulkViscosityCoefficient;
    float const localBulkViscosityTarget = targetBulkViscosityCoefficient;
    float const elapsedTime = timeStep;
    return currentBulkViscosityCoefficient < localBulkViscosityTarget ? localBulkViscosityTarget :
                                                                        localBulkViscosityTarget +
            (currentBulkViscosityCoefficient - localBulkViscosityTarget) *
                std::exp(-elapsedTime * inverseDecayTime);
}

}    // namespace cullen_and_dehnen
