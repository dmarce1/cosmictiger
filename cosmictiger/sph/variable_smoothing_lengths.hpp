/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

/*
 * Variational SPH pressure force and compatible thermal work, after
 *
 *   P. F. Hopkins, "A general class of Lagrangian smoothed particle hydrodynamics
 *   methods and implications for fluid mixing problems", MNRAS 428 (2013), 2840-2856,
 *   doi:10.1093/mnras/sts210, https://arxiv.org/abs/1206.5006,
 *   equations 12-13 with the support-constraint weight x~ⱼ = 1; and
 *   D. J. Price, "Smoothed particle hydrodynamics and magnetohydrodynamics",
 *   J. Comput. Phys. 231 (2012), 759-794, doi:10.1016/j.jcp.2010.12.011,
 *   https://arxiv.org/abs/1012.1885, equations 30 and 35 for the density formulation.
 *
 * Particle i has mass mᵢ, fixed volume weight xᵢ, weighted density yᵢ and volume
 * Vᵢ = xᵢ/yᵢ. cosmictiger/sph/volume.hpp evaluates yᵢ and the grad-H correction
 * ζᵢ for the support constraint Hᵢ^D nᵢ = constant. Its spatial derivative is
 *
 *   dyᵢ = Σⱼ (xⱼ - ζᵢ) gᵢ . drᵢⱼ,   gᵢ = grad_i Wᵢⱼ(Hᵢ).
 *
 * The adiabatic first law d(mᵢ uᵢ) = -Pᵢ dVᵢ = (Pᵢ xᵢ/yᵢ^2) dyᵢ and
 * L = Σᵢ mᵢ (vᵢ^2/2 - uᵢ) give the directed coefficient and inviscid rates
 *
 *   Qᵢⱼ = (Pᵢ xᵢ/yᵢ^2)(xⱼ - ζᵢ),
 *   mᵢ dvᵢ/dt = -Σⱼ [Qᵢⱼ gᵢ + Qⱼᵢ gⱼ],
 *   mᵢ duᵢ/dt =  Σⱼ Qᵢⱼ vᵢⱼ . gᵢ,
 *
 * where rᵢⱼ = rᵢ-rⱼ, vᵢⱼ = vᵢ-vⱼ and gⱼ = grad_i Wᵢⱼ(Hⱼ).
 * Hopkins' fᵢⱼ = 1 - ζᵢ/xⱼ is absorbed into Qᵢⱼ. For equal masses with
 * xᵢ = mᵢ, yᵢ = ρᵢ and xⱼ-ζᵢ = mᵢ/Ωᵢ, these become Price (2012),
 * equations 30 and 35. A held support uses ζᵢ = 0.
 *
 * The same directed coefficient supplies force and work. Callers divide by mᵢ
 * and apply the cosmological scale factors derived in docs/variational_sph.md;
 * artificial viscosity is supplied separately by the Monaghan operator.
 */
#pragma once

#include <cosmictiger/cuda/cuda_export.hpp>

namespace variational {

// Pᵢ xᵢ/yᵢ^2 = -Pᵢ dVᵢ/dyᵢ from Vᵢ = xᵢ/yᵢ and the adiabatic first law.
// clang-format off
/*
 * Derived form of Hopkins (2013), equations (12)–(13), and Price (2012), (30), (35).
 * Full citations and derivation above; Hopkins (2013) https://arxiv.org/abs/1206.5006,
 * Price (2012) https://arxiv.org/abs/1012.1885.
 *   Pᵢ xᵢ/yᵢ² = −Pᵢ ∂Vᵢ/∂yᵢ.
 * Symbol directory:
 *   Pᵢ → pressure; xᵢ → weight; yᵢ → weighted density value; expression → return value.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real pressureVolumeFactor(Real Pᵢ, Real xᵢ, Real yᵢ) {
    return Pᵢ * xᵢ / (yᵢ * yᵢ);
}

// Hopkins (2013), equations 12-13: Qᵢⱼ = (Pᵢ xᵢ/yᵢ^2)(xⱼ - ζᵢ).
// clang-format off
/*
 * Derived form of Hopkins (2013), equations (12)–(13), and Price (2012), (30), (35).
 * Full citations and derivation above; Hopkins (2013) https://arxiv.org/abs/1206.5006,
 * Price (2012) https://arxiv.org/abs/1012.1885.
 *   Qᵢⱼ = (Pᵢ xᵢ/yᵢ²)(xⱼ − ζᵢ).
 * Symbol directory:
 *   Pᵢ xᵢ/yᵢ² → sink pressure volume factor; xⱼ → neighbor volume weight; ζᵢ → sink grad-H correction; Qᵢⱼ → return value.
 *   Code names: Pᵢxᵢ_over_y2ᵢ = Pᵢ xᵢ/yᵢ².
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real directedCoefficient(Real Pᵢxᵢ_over_y2ᵢ, Real xⱼ, Real ζᵢ) {
    return Pᵢxᵢ_over_y2ᵢ * (xⱼ - ζᵢ);
}

// Hopkins (2013), equation 12, or Price (2012), equation 30 in the equal-mass limit.
// With gᵢ = Fᵢ rᵢⱼ and gⱼ = Fⱼ rᵢⱼ, mᵢ dvᵢ/dt gains -(this) rᵢⱼ from pair ij.
// clang-format off
/*
 * Derived form of Hopkins (2013), equations (12)–(13), and Price (2012), (30), (35).
 * Full citations and derivation above; Hopkins (2013) https://arxiv.org/abs/1206.5006,
 * Price (2012) https://arxiv.org/abs/1012.1885.
 *   mᵢ dvᵢ/dt|_j = −(Qᵢⱼ Fᵢ + Qⱼᵢ Fⱼ) rᵢⱼ.
 * Symbol directory:
 *   Qᵢⱼ, Qⱼᵢ → ij coefficient, ji coefficient; Fᵢ, Fⱼ → i gradient factor, j gradient factor
 *   positive parenthesis → return value.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real pairForceFactor(Real Qᵢⱼ, Real Qⱼᵢ, Real Fᵢ, Real Fⱼ) {
    return Qᵢⱼ * Fᵢ + Qⱼᵢ * Fⱼ;
}

// The adiabatic first law above; Price (2012), equation 35 in the equal-mass limit.
// mᵢ duᵢ/dt gains Qᵢⱼ vᵢⱼ . gᵢ = Qᵢⱼ Fᵢ (vᵢⱼ . rᵢⱼ) from pair ij.
// clang-format off
/*
 * Derived form of Hopkins (2013), equations (12)–(13), and Price (2012), (30), (35).
 * Full citations and derivation above; Hopkins (2013) https://arxiv.org/abs/1206.5006,
 * Price (2012) https://arxiv.org/abs/1012.1885.
 *   mᵢ duᵢ/dt|_j = Qᵢⱼ Fᵢ (vᵢⱼ · rᵢⱼ).
 * Symbol directory:
 *   Qᵢⱼ → ij coefficient; Fᵢ → i gradient factor; vᵢⱼ · rᵢⱼ → velocity dot separation
 *   pair work → return value.
 *   Code names: vᵢⱼ_dot_rᵢⱼ = vᵢⱼ · rᵢⱼ.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real pairWork(Real Qᵢⱼ, Real Fᵢ, Real vᵢⱼ_dot_rᵢⱼ) {
    return Qᵢⱼ * Fᵢ * vᵢⱼ_dot_rᵢⱼ;
}

}    // namespace variational
