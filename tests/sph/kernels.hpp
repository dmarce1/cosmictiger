/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

/*
 * Classical SPH formulae from
 *
 *   J. J. Monaghan, "Smoothed Particle Hydrodynamics",
 *   Annual Review of Astronomy and Astrophysics 30 (1992), 543-574.
 *   doi:10.1146/annurev.aa.30.090192.002551, ADS bibcode 1992ARA&A..30..543M,
 *   https://articles.adsabs.harvard.edu/pdf/1992ARA%26A..30..543M
 *
 * Equation numbers and pages below refer to that review. Comments retain its
 * symbols: particle a is the one being updated and particle b a neighbor; m mass,
 * ρ density, P pressure, u specific internal energy, c sound speed, h_M smoothing
 * length (h in the paper), r𞁑𞁓 = r𞁑 − r𞁓, v𞁑𞁓 = v𞁑 − v𞁓, W𞁑𞁓 = W(r𞁑𞁓, h_M) the kernel,
 * W̃𞁑𞁓 the symmetric kernel of Monaghan (1992), p. 554, Π𞁑𞁓 the artificial viscosity and μ𞁑𞁓 its
 * velocity measure, with the constants α, β and η. Equation code uses
 * these symbols; each equation block describes them. The stored smoothing length
 * is the compact support radius h, which equals 2h_M for the cubic spline of
 * Monaghan (1992), section 7.
 */
#pragma once

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/simd_math.hpp>

namespace sph_kernels {

constexpr double kPi = 3.141592653589793238462643383279502884;

/*
 * Monaghan (1992), section 7, pp. 554-555 (unnumbered): the cubic spline kernel
 *
 *   W(r, h_M) = (σ/h_M³) · { 1 − (3/2) q² + (3/4) q³   for 0 ≤ q ≤ 1,
 *                             (1/4) (2 − q)³            for 1 ≤ q ≤ 2,
 *                             0                         otherwise },      q = r/h_M,
 *
 * with σ = 1/π in three dimensions. This returns h³ W as a function of r/h for the
 * stored support radius h = 2h_M, so the Monaghan (1992) q is twice the argument.
 */
// clang-format off
/*
 * Symbol directory for the Monaghan (1992), section 7, formula above:
 *   r/h → normalized radius; q = r/h_M = 2r/h → smoothing normalized radius
 *   σ = 1/π → kernel normalization; h³/h_M³ → support volume scale
 *   2 − q → distance to support edge; dq/d(r/h) = 2 → normalized radius scale
 *   Code names: r_over_h = r/h; h3_over_h3_M = h³/h_M³; two_minus_q = 2 − q
 *     dq_d_r_over_h = dq/d(r/h) = 2.
 * (the last constant is used only in kernelDerivative).
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real kernelValue(Real r_over_h) {
    constexpr Real σ = Real(1.0 / kPi);
    constexpr Real h3_over_h3_M = Real(8);    // (h/h_M)³ with h = 2h_M
    Real const q = Real(2) * r_over_h;
    if (q >= Real(2)) {
        return Real(0);
    }
    if (q <= Real(1)) {
        return σ * h3_over_h3_M * (Real(1) - Real(1.5) * q * q + Real(0.75) * q * q * q);
    }
    Real const two_minus_q = Real(2) - q;
    return σ * h3_over_h3_M * Real(0.25) * two_minus_q * two_minus_q * two_minus_q;
}

/*
 * Derivative of the kernel above with respect to r/h:
 *
 *   d(h³W)/d(r/h) = 2 (σ h³/h_M³) · { −3q + (9/4) q²       for 0 ≤ q ≤ 1,
 *                                       −(3/4) (2 − q)²       for 1 ≤ q ≤ 2,
 *                                       0                     otherwise },
 *
 * where the factor 2 is dq/d(r/h) for h = 2h_M.
 */
// clang-format off
/*
 * Symbol directory for the Monaghan (1992), section 7, formula above:
 *   r/h → normalized radius; q = r/h_M = 2r/h → smoothing normalized radius
 *   σ = 1/π → kernel normalization; h³/h_M³ → support volume scale
 *   2 − q → distance to support edge; dq/d(r/h) = 2 → normalized radius scale
 *   Code names: r_over_h = r/h; h3_over_h3_M = h³/h_M³; two_minus_q = 2 − q
 *     dq_d_r_over_h = dq/d(r/h) = 2.
 * (the last constant is used only in kernelDerivative).
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real kernelDerivative(Real r_over_h) {
    constexpr Real σ = Real(1.0 / kPi);
    constexpr Real h3_over_h3_M = Real(8);
    constexpr Real dq_d_r_over_h = Real(2);
    Real const q = Real(2) * r_over_h;
    if (q >= Real(2)) {
        return Real(0);
    }
    if (q <= Real(1)) {
        return dq_d_r_over_h * σ * h3_over_h3_M * q * (Real(2.25) * q - Real(3));
    }
    Real const two_minus_q = Real(2) - q;
    return -dq_d_r_over_h * σ * h3_over_h3_M * Real(0.75) * two_minus_q * two_minus_q;
}

}    // namespace sph_kernels
