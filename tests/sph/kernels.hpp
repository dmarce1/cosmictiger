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
 * ρ density, P pressure, u specific internal energy, c sound speed, h smoothing
 * length, r_ab = r_a − r_b, v_ab = v_a − v_b, W_ab = W(r_ab, h) the kernel,
 * W̃_ab the symmetric kernel of p. 554, Π_ab the artificial viscosity and μ_ab its
 * velocity measure, with the constants α, β and η. All code identifiers use
 * descriptive names; each equation block maps symbols to those names. The stored smoothing length
 * is the compact support radius H, which equals 2h for the paper's cubic spline of section 7.
 */
#pragma once

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/simd_math.hpp>

namespace sph_kernels {

constexpr double kPi = 3.141592653589793238462643383279502884;

/*
 * Section 7, pp. 554-555 (unnumbered): the cubic spline kernel
 *
 *   W(r, h) = (σ/h³) · { 1 − (3/2) q² + (3/4) q³   for 0 ≤ q ≤ 1,
 *                         (1/4) (2 − q)³            for 1 ≤ q ≤ 2,
 *                         0                         otherwise },      q = r/h,
 *
 * with σ = 1/π in three dimensions. This returns H³ W as a function of r/H for the
 * stored support radius H = 2h, so the paper's q is twice the argument.
 */
// clang-format off
/*
 * Symbol directory for the Monaghan (1992), section 7, formula above:
 *   r/H → normalizedRadius; q = r/h = 2r/H → smoothingNormalizedRadius
 *   σ = 1/π → kKernelNormalization; H³/h³ → kSupportVolumeScale
 *   2 − q → distanceToSupportEdge; dq/d(r/H) = 2 → kNormalizedRadiusScale
 * (the last constant is used only in kernelDerivative).
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real kernelValue(Real normalizedRadius) {
    constexpr Real kKernelNormalization = Real(1.0 / kPi);
    constexpr Real kSupportVolumeScale = Real(8);    // (H/h)³ with H = 2h
    Real const smoothingNormalizedRadius = Real(2) * normalizedRadius;
    if (smoothingNormalizedRadius >= Real(2)) {
        return Real(0);
    }
    if (smoothingNormalizedRadius <= Real(1)) {
        return kKernelNormalization * kSupportVolumeScale *
            (Real(1) - Real(1.5) * smoothingNormalizedRadius * smoothingNormalizedRadius +
                Real(0.75) * smoothingNormalizedRadius * smoothingNormalizedRadius *
                    smoothingNormalizedRadius);
    }
    Real const distanceToSupportEdge = Real(2) - smoothingNormalizedRadius;
    return kKernelNormalization * kSupportVolumeScale * Real(0.25) * distanceToSupportEdge *
        distanceToSupportEdge * distanceToSupportEdge;
}

/*
 * Derivative of the kernel above with respect to r/H:
 *
 *   d(H³W)/d(r/H) = 2 (σ H³/h³) · { −3q + (9/4) q²       for 0 ≤ q ≤ 1,
 *                                     −(3/4) (2 − q)²       for 1 ≤ q ≤ 2,
 *                                     0                     otherwise },
 *
 * where the factor 2 is dq/d(r/H) for H = 2h.
 */
// clang-format off
/*
 * Symbol directory for the Monaghan (1992), section 7, formula above:
 *   r/H → normalizedRadius; q = r/h = 2r/H → smoothingNormalizedRadius
 *   σ = 1/π → kKernelNormalization; H³/h³ → kSupportVolumeScale
 *   2 − q → distanceToSupportEdge; dq/d(r/H) = 2 → kNormalizedRadiusScale
 * (the last constant is used only in kernelDerivative).
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real kernelDerivative(Real normalizedRadius) {
    constexpr Real kKernelNormalization = Real(1.0 / kPi);
    constexpr Real kSupportVolumeScale = Real(8);
    constexpr Real kNormalizedRadiusScale = Real(2);
    Real const smoothingNormalizedRadius = Real(2) * normalizedRadius;
    if (smoothingNormalizedRadius >= Real(2)) {
        return Real(0);
    }
    if (smoothingNormalizedRadius <= Real(1)) {
        return kNormalizedRadiusScale * kKernelNormalization * kSupportVolumeScale *
            smoothingNormalizedRadius * (Real(2.25) * smoothingNormalizedRadius - Real(3));
    }
    Real const distanceToSupportEdge = Real(2) - smoothingNormalizedRadius;
    return -kNormalizedRadiusScale * kKernelNormalization * kSupportVolumeScale * Real(0.75) *
        distanceToSupportEdge * distanceToSupportEdge;
}

}    // namespace sph_kernels
