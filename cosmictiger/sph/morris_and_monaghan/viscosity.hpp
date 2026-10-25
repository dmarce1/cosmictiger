/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/math/simd_math.hpp>
#include <cosmictiger/sph/viscosity_state.hpp>

#ifdef __CUDACC__
#define COSMICTIGER_MORRIS_AND_MONAGHAN_EXPORT __host__ __device__
#else
#define COSMICTIGER_MORRIS_AND_MONAGHAN_EXPORT
#endif

namespace morris_and_monaghan {

// NOLINTBEGIN(readability-identifier-naming)

// Morris & Monaghan (1997), "A Switch to Reduce SPH Viscosity", JCP 136,
// 41-50, doi:10.1006/jcph.1997.5690, equations (5), (6), and (13):
//   dα_a/dt = −(α_a − α*)/τ_a + S_a,
//   τ_a = h_a/(C₁ c_a),  S_a = max(−div v_a, 0).
// Their calculations use α* = 0.1 and C₁ = 0.2; there is no upper α cap.
// Both inverseDecayTime and divergence must refer to the integration time unit.
template <typename Real>
COSMICTIGER_MORRIS_AND_MONAGHAN_EXPORT inline Real viscosityAlphaRate(
    Real alpha, Real alphaMinimum, Real inverseDecayTime, Real divergence) {
    Real const α_a = alpha;
    Real const α_star = alphaMinimum;
    Real const one_over_τ_a = inverseDecayTime;
    Real const S_a = simdMax(-divergence, Real(0));
    return -(α_a - α_star) * one_over_τ_a + S_a;
}

void applyUpdate(SphViscosityState& state, float minimumAlpha, float halfTimeStep, int phase);

// NOLINTEND(readability-identifier-naming)

}    // namespace morris_and_monaghan

#undef COSMICTIGER_MORRIS_AND_MONAGHAN_EXPORT
