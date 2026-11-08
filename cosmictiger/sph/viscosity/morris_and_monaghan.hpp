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
 * Morris & Monaghan (1997), JCP 136, 41–50, doi:10.1006/jcph.1997.5690, equations (5), (6),
 * (13), with the source limiter of Rosswog, Davies, Thielemann & Piran (2000), A&A 360, 171:
 *   dα𞁑/dt = −(α𞁑 − αₘᵢₙ)/τ𞁑 + (αₘₐₓ − α𞁑) S𞁑,
 *   τ𞁑 = h_M𞁑/(C c𞁑),  S𞁑 = max(−∇·v𞁑, 0).
 * Derived backward-Euler update with frozen S𞁑 and τ𞁑:
 *   α𞁑ⁿ⁺¹ = αₘᵢₙ + (α𞁑ⁿ − αₘᵢₙ + Δt S𞁑 (αₘₐₓ − αₘᵢₙ))/(1 + Δt/τ𞁑 + Δt S𞁑).
 * Symbol directory:
 *   α𞁑ⁿ → bulk viscosity coefficient; αₘᵢₙ → minimum bulk viscosity coefficient
 *   αₘₐₓ → maximum bulk viscosity coefficient; τ𞁑⁻¹ → inverse decay time
 *   ∇·v𞁑 → divergence; Δt → time step; S𞁑 → source; Δt/τ𞁑 → decay step
 *   h_M𞁑 → Monaghan smoothing length, the h of Morris & Monaghan (1997); C → decay parameter
 *   α𞁑ⁿ⁺¹ → updated (return value).
 *   Code names: αⁿ𞁑 = α𞁑ⁿ; iτ𞁑 = τ𞁑⁻¹; div_v_a = ∇·v𞁑; Δt_over_τ𞁑 = Δt/τ𞁑
 *     αⁿ_plus_1𞁑 = α𞁑ⁿ⁺¹.
 * All rates use the integration time unit. The update is a positive weighting of α𞁑ⁿ, αₘᵢₙ
 * and αₘₐₓ, so [αₘᵢₙ, αₘₐₓ] is invariant for every Δt; αₘᵢₙ = αₘₐₓ holds α fixed. The shifted
 * form preserves the floor; a zero interval returns the input exactly.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real evolvedBulkViscosityCoefficient(
    Real αⁿ𞁑, Real αₘᵢₙ, Real αₘₐₓ, Real iτ𞁑, Real div_v_a, Real Δt) {
    Real const S𞁑 = simdMax(-div_v_a, Real(0));
    Real const Δt_over_τ𞁑 = Δt * iτ𞁑;
    // Shift by the floor so its preservation does not depend on cancellation.
    Real const αⁿ_plus_1𞁑 =
        αₘᵢₙ + (αⁿ𞁑 - αₘᵢₙ + Δt * S𞁑 * (αₘₐₓ - αₘᵢₙ)) / (Real(1) + Δt_over_τ𞁑 + Δt * S𞁑);
    return simdSelect(Δt == Real(0), αⁿ𞁑, αⁿ_plus_1𞁑);
}

}    // namespace morris_and_monaghan
