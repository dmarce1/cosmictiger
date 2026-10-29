/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cmath>

#include <cosmictiger/sph/cullen_and_dehnen/viscosity.hpp>

namespace cullen_and_dehnen {

// NOLINTBEGIN(readability-identifier-naming)
// Cullen & Dehnen (2010), equation (A15): immediate increases, exponential decay.
float evolvedAlpha(float alpha, float targetAlpha, float inverseDecayTime, float timeStep) {
    float const α_i = alpha;
    float const α_loc_i = targetAlpha;
    float const Δt = timeStep;
    return α_i < α_loc_i ? α_loc_i : α_loc_i + (α_i - α_loc_i) * std::exp(-Δt * inverseDecayTime);
}
// NOLINTEND(readability-identifier-naming)

}    // namespace cullen_and_dehnen
