/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>

#include <cosmictiger/sph/morris_and_monaghan/viscosity.hpp>

namespace morris_and_monaghan {

// Morris & Monaghan (1997), equation (5), advanced with the hydro half-kicks.
// Save the pre-predictor value so the lower bound cannot bias correction.
// NOLINTBEGIN(readability-identifier-naming)
void applyUpdate(SphViscosityState& state, float minimumAlpha, float halfTimeStep, int phase) {
    float const α_star = minimumAlpha;
    float const δτ = halfTimeStep;
    if (phase == 0) {
        state.predictedAlpha = state.alpha;
        state.alpha = std::max(α_star, state.alpha + state.rate * δτ);
    } else if (phase == 1) {
        state.alpha = std::max(α_star, state.predictedAlpha + state.rate * δτ);
    } else if (phase == 2) {
        state.alpha = std::max(α_star, state.alpha + state.rate * δτ);
    }
}
// NOLINTEND(readability-identifier-naming)

}    // namespace morris_and_monaghan
