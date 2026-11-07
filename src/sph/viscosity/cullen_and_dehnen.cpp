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
 *   α_iⁿ → bulk viscosity coefficient
 *   α_loc,i → target bulk viscosity coefficient
 *   Δt → time step; τ_i⁻¹ → inverse decay time; α_iⁿ⁺¹ → return value.
 *   Code names: α_loc_i = α_loc,i; τ_i_inv = τ_i⁻¹.
 */
// clang-format on
float evolvedBulkViscosityCoefficient(float α_iⁿ, float α_loc_i, float τ_i_inv, float Δt) {
    return α_iⁿ < α_loc_i ? α_loc_i : α_loc_i + (α_iⁿ - α_loc_i) * std::exp(-Δt * τ_i_inv);
}

}    // namespace cullen_and_dehnen
