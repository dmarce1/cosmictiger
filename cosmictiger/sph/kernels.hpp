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
 * Equation numbers and pages below refer to that review. Function bodies use its
 * symbols: particle a is the one being updated and particle b a neighbor; m mass,
 * ρ density, P pressure, u specific internal energy, c sound speed, h smoothing
 * length, r_ab = r_a − r_b, v_ab = v_a − v_b, W_ab = W(r_ab, h) the kernel,
 * W̃_ab the symmetric kernel of p. 554, Π_ab the artificial viscosity and μ_ab its
 * velocity measure, with the constants α, β and η. Function parameters keep
 * descriptive names. The stored smoothing length is the compact support radius H,
 * which equals 2h for the paper's cubic spline of section 7.
 */
#pragma once

#include <cosmictiger/math/simd_math.hpp>

#ifdef __CUDACC__
#define COSMICTIGER_SPH_KERNELS_EXPORT __host__ __device__
#else
#define COSMICTIGER_SPH_KERNELS_EXPORT
#endif

namespace sph_kernels {

// NOLINTBEGIN(readability-identifier-naming)

constexpr double kPi = 3.141592653589793238462643383279502884;

// Section 7, pp. 554-555 (unnumbered): the cubic spline kernel
//
//   W(r, h) = (σ/h³) · { 1 − (3/2) q² + (3/4) q³   for 0 ≤ q ≤ 1,
//                         (1/4) (2 − q)³            for 1 ≤ q ≤ 2,
//                         0                         otherwise },      q = r/h,
//
// with σ = 1/π in three dimensions. This returns H³ W as a function of r/H for the
// stored support radius H = 2h, so the paper's q is twice the argument.
template <typename Real>
COSMICTIGER_SPH_KERNELS_EXPORT inline Real kernelValue(Real normalizedRadius) {
    constexpr Real σ = Real(1.0 / kPi);
    constexpr Real H3_over_h3 = Real(8);    // (H/h)³ with H = 2h
    Real const q = Real(2) * normalizedRadius;
    if (q >= Real(2)) {
        return Real(0);
    }
    if (q <= Real(1)) {
        return σ * H3_over_h3 * (Real(1) - Real(1.5) * q * q + Real(0.75) * q * q * q);
    }
    Real const two_minus_q = Real(2) - q;
    return σ * H3_over_h3 * Real(0.25) * two_minus_q * two_minus_q * two_minus_q;
}

// Derivative of the kernel above with respect to r/H:
//
//   d(H³W)/d(r/H) = 2 (σ H³/h³) · { −3q + (9/4) q²       for 0 ≤ q ≤ 1,
//                                     −(3/4) (2 − q)²       for 1 ≤ q ≤ 2,
//                                     0                     otherwise },
//
// where the factor 2 is dq/d(r/H) for H = 2h.
template <typename Real>
COSMICTIGER_SPH_KERNELS_EXPORT inline Real kernelDerivative(Real normalizedRadius) {
    constexpr Real σ = Real(1.0 / kPi);
    constexpr Real H3_over_h3 = Real(8);
    constexpr Real dq_dnormalizedRadius = Real(2);
    Real const q = Real(2) * normalizedRadius;
    if (q >= Real(2)) {
        return Real(0);
    }
    if (q <= Real(1)) {
        return dq_dnormalizedRadius * σ * H3_over_h3 * q * (Real(2.25) * q - Real(3));
    }
    Real const two_minus_q = Real(2) - q;
    return -dq_dnormalizedRadius * σ * H3_over_h3 * Real(0.75) * two_minus_q * two_minus_q;
}

// NOLINTEND(readability-identifier-naming)

}    // namespace sph_kernels

#undef COSMICTIGER_SPH_KERNELS_EXPORT
