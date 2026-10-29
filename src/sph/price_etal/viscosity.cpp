/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/sph/price_etal/viscosity.hpp>

namespace price_etal {

// NOLINTBEGIN(readability-identifier-naming)
// Price et al. (2018): "If α_loc,a > α_a we set α_a = α_loc,a, otherwise we evolve α_a
// according to" equation (49), integrated with the backward Euler step (50),
//
//   α^{n+1} = (α^n + α_loc Δt/τ) / (1 + Δt/τ),
//
// "which ensures that the decay is stable regardless of the timestep".
float evolvedAlpha(float alpha, float targetAlpha, float inverseDecayTime, float timeStep) {
    float const α_a = alpha;
    float const α_loc_a = targetAlpha;
    float const Δt_over_τ_a = timeStep * inverseDecayTime;
    return α_loc_a > α_a ? α_loc_a : (α_a + α_loc_a * Δt_over_τ_a) / (1.0f + Δt_over_τ_a);
}
// NOLINTEND(readability-identifier-naming)

}    // namespace price_etal
