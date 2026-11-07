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
 *   α_aⁿ → bulk viscosity coefficient; α* → minimum bulk viscosity coefficient
 *   τ_a⁻¹ → inverse decay time; ∇·v_a → divergence; Δt → time step
 *   S_a → source; Δt/τ_a → decay step; α_aⁿ⁺¹ → updated (return value).
 *   Code names: α_star = α*; τ_a_inv = τ_a⁻¹; div_v_a = ∇·v_a; Δt_over_τ_a = Δt/τ_a;
 *     α_aⁿ_plus_1 = α_aⁿ⁺¹.
 * All rates use the integration time unit. The paper uses α* = 0.1 and C₁ = 0.2.
 * The shifted form preserves the floor; a zero interval returns the input exactly.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real evolvedBulkViscosityCoefficient(
    Real α_aⁿ, Real α_star, Real τ_a_inv, Real div_v_a, Real Δt) {
    Real const S_a = simdMax(-div_v_a, Real(0));
    Real const Δt_over_τ_a = Δt * τ_a_inv;
    // Shift by the floor so its preservation does not depend on cancellation.
    Real const α_aⁿ_plus_1 = α_star + (α_aⁿ - α_star + Δt * S_a) / (Real(1) + Δt_over_τ_a);
    return simdSelect(Δt == Real(0), α_aⁿ, α_aⁿ_plus_1);
}

}    // namespace morris_and_monaghan
