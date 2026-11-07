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
 *   α_aⁿ → bulk viscosity coefficient; α_loc,a → target bulk viscosity coefficient
 *   Δt → time step; τ_a⁻¹ → inverse decay time; Δt/τ_a → decay step
 *   α_aⁿ⁺¹ → return value.
 *   Code names: α_loc_a = α_loc,a; τ_a_inv = τ_a⁻¹; Δt_over_τ_a = Δt/τ_a; α_aⁿ_plus_1 = α_aⁿ⁺¹.
 */
// clang-format on
float evolvedBulkViscosityCoefficient(float α_aⁿ, float α_loc_a, float τ_a_inv, float Δt) {
    float const Δt_over_τ_a = Δt * τ_a_inv;
    return α_loc_a > α_aⁿ ? α_loc_a : (α_aⁿ + α_loc_a * Δt_over_τ_a) / (1.0f + Δt_over_τ_a);
}

}    // namespace price_etal
