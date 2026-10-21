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
#define COSMICTIGER_MONAGHAN_EXPORT __host__ __device__
#else
#define COSMICTIGER_MONAGHAN_EXPORT
#endif

namespace monaghan_sph {

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
COSMICTIGER_MONAGHAN_EXPORT inline Real kernelValue(Real normalizedRadius) {
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
COSMICTIGER_MONAGHAN_EXPORT inline Real kernelDerivative(Real normalizedRadius) {
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

// Equation 4.2, p. 550, with the regularization of p. 551:
//
//   μ_ab = h v_ab·r_ab / (r_ab² + η²),      η² = 0.01 h²,
//
// and μ_ab = 0 when v_ab·r_ab ≥ 0, the vanishing branch of Π_ab on p. 550. For
// unequal resolution lengths the pair uses h_ab = ½(h_a + h_b), the arithmetic mean
// that p. 554 offers. viscosityEpsilon is η²/h². The two scale factors convert the
// stored support radii to the paper's h: supportRadiusPerEffectiveRadius maps H to
// the common effective radius of docs/kernels.md, and viscosityWidthScale maps that
// radius to h. Their defaults give h = H/2, the cubic-spline convention, for the
// standalone references in tests/monaghan_sph.cpp.
template <typename Real>
COSMICTIGER_MONAGHAN_EXPORT inline Real viscosityMu(Real radiusSquared, Real velocityDotRadius,
    Real supportI, Real supportJ, Real viscosityEpsilon,
    Real supportRadiusPerEffectiveRadius = Real(2), Real viscosityWidthScale = Real(1)) {
    Real const r_ab2 = radiusSquared;    // |r_ab|²
    Real const v_ab_dot_r_ab = velocityDotRadius;
    Real const H_a = supportI;
    Real const H_b = supportJ;
    Real const h_ab = simdBroadcast<Real>(0.5) * (H_a + H_b) / supportRadiusPerEffectiveRadius *
        viscosityWidthScale;
    Real const η2 = viscosityEpsilon * h_ab * h_ab;    // η²
    Real const μ_ab = h_ab * v_ab_dot_r_ab / (r_ab2 + η2);
    return simdSelect(v_ab_dot_r_ab < Real(0), μ_ab, Real(0));
}

// Artificial viscosity Π_ab, p. 550 (unnumbered, following equation 4.1):
//
//   Π_ab = (−α c̄_ab μ_ab + β μ_ab²) / ρ̄_ab    for v_ab·r_ab < 0,
//   Π_ab = 0                                    for v_ab·r_ab > 0,
//
// with c̄_ab = ½(c_a + c_b) and ρ̄_ab = ½(ρ_a + ρ_b). The paper recommends α = 1 and
// β = 2 (p. 551). Since μ_ab already vanishes for separating pairs, the branch is
// selected on μ_ab < 0.
template <typename Real>
COSMICTIGER_MONAGHAN_EXPORT inline Real artificialViscosity(
    Real rhoI, Real rhoJ, Real soundI, Real soundJ, Real mu, Real alpha, Real beta) {
    Real const ρ_a = rhoI;
    Real const ρ_b = rhoJ;
    Real const c_a = soundI;
    Real const c_b = soundJ;
    Real const μ_ab = mu;
    Real const α = alpha;
    Real const β = beta;
    Real const c̄_ab = simdBroadcast<Real>(0.5) * (c_a + c_b);
    Real const ρ̄_ab = simdBroadcast<Real>(0.5) * (ρ_a + ρ_b);
    Real const Π_ab = (-α * c̄_ab * μ_ab + β * μ_ab * μ_ab) / ρ̄_ab;
    return simdSelect(μ_ab < Real(0), Π_ab, Real(0));
}

// Balsara (1995) shear correction, as in the supplied equation (72):
// f_a = |div v_a| / (|div v_a| + |curl v_a|). The density normalization common
// to both SPH derivative sums cancels. Define the indeterminate static case as zero;
// select a safe denominator before division so SIMD lanes never evaluate 0/0.
template <typename Real>
COSMICTIGER_MONAGHAN_EXPORT inline Real balsaraFactor(Real divergence, Real curlMagnitude) {
    Real const compression = simdMax(divergence, -divergence);
    Real const denominator = compression + curlMagnitude;
    return compression / simdSelect(denominator > Real(0), denominator, Real(1));
}

// The symmetric pressure term P_b/ρ_b² + P_a/ρ_a² shared by the momentum equation 3.3,
// its viscous form 4.1, and the variable-resolution pair equations 6.4-6.5. The same
// factor in the force and the thermal work conserves momentum and energy (p. 549).
template <typename Real>
COSMICTIGER_MONAGHAN_EXPORT inline Real pressureCoefficient(
    Real rhoI, Real rhoJ, Real pressureI, Real pressureJ) {
    Real const ρ_a = rhoI;
    Real const ρ_b = rhoJ;
    Real const P_a = pressureI;
    Real const P_b = pressureJ;
    return P_a / (ρ_a * ρ_a) + P_b / (ρ_b * ρ_b);
}

// Equation 6.5, p. 554 (equation 3.17 with Π_ab and the symmetric kernel):
//
//   du_a/dt = ½ Σ_b m_b (P_b/ρ_b² + P_a/ρ_a² + Π_ab) v_ab·∇_a W̃_ab.
//
// This returns one pair's term without m_b. The velocity factor is split so that the
// pressure work uses the peculiar relative velocity while the viscous heating uses the
// physical one, which lets the caller integrate homogeneous cosmological pressure work
// analytically (docs/monaghan_sph.md); with equal arguments it is the paper's form.
template <typename Real>
COSMICTIGER_MONAGHAN_EXPORT inline Real pairEnergyRate(Real pressurePair, Real viscosity,
    Real peculiarVelocityDotGradient, Real physicalVelocityDotGradient) {
    Real const P_over_ρ2_ab = pressurePair;    // P_b/ρ_b² + P_a/ρ_a²
    Real const Π_ab = viscosity;
    Real const v_ab_dot_gradW̃_ab = peculiarVelocityDotGradient;
    Real const v_ab_dot_gradW̃_ab_physical = physicalVelocityDotGradient;
    return simdBroadcast<Real>(0.5) *
        (P_over_ρ2_ab * v_ab_dot_gradW̃_ab + Π_ab * v_ab_dot_gradW̃_ab_physical);
}

// NOLINTEND(readability-identifier-naming)

}    // namespace monaghan_sph

#undef COSMICTIGER_MONAGHAN_EXPORT
