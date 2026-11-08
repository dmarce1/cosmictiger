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
 *   dα_a/dt = −(α𞁑 − α*)/τ𞁑 + S𞁑,
 *   τ𞁑 = h𞁑/(C₁ c𞁑),  S𞁑 = max(−∇·v𞁑, 0).
 * Derived backward-Euler update with frozen S𞁑 and τ𞁑:
 *   α𞁑ⁿ⁺¹ = α* + (α𞁑ⁿ − α* + Δt S𞁑)/(1 + Δt/τ𞁑).
 * Symbol directory:
 *   α𞁑ⁿ → bulk viscosity coefficient; α* → minimum bulk viscosity coefficient
 *   τ𞁑⁻¹ → inverse decay time; ∇·v𞁑 → divergence; Δt → time step
 *   S𞁑 → source; Δt/τ𞁑 → decay step; α𞁑ⁿ⁺¹ → updated (return value).
 *   Code names: αⁿ𞁑 = α𞁑ⁿ; αₛₜₐᵣ = α*; iτ𞁑 = τ𞁑⁻¹; div_v_a = ∇·v𞁑
 *     Δt_over_τ𞁑 = Δt/τ𞁑; αⁿ_plus_1𞁑 = α𞁑ⁿ⁺¹.
 * All rates use the integration time unit. Morris & Monaghan (1997) use α* = 0.1 and C₁ = 0.2.
 * The shifted form preserves the floor; a zero interval returns the input exactly.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real evolvedBulkViscosityCoefficient(
    Real αⁿ𞁑, Real αₛₜₐᵣ, Real iτ𞁑, Real div_v_a, Real Δt) {
    Real const S𞁑 = simdMax(-div_v_a, Real(0));
    Real const Δt_over_τ𞁑 = Δt * iτ𞁑;
    // Shift by the floor so its preservation does not depend on cancellation.
    Real const αⁿ_plus_1𞁑 = αₛₜₐᵣ + (αⁿ𞁑 - αₛₜₐᵣ + Δt * S𞁑) / (Real(1) + Δt_over_τ𞁑);
    return simdSelect(Δt == Real(0), αⁿ𞁑, αⁿ_plus_1𞁑);
}

}    // namespace morris_and_monaghan
