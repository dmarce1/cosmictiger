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

namespace monaghan {

// NOLINTBEGIN(readability-identifier-naming)

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
// standalone references in tests/sph/monaghan/equations.cpp.
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

// NOLINTEND(readability-identifier-naming)

}    // namespace monaghan

#undef COSMICTIGER_MONAGHAN_EXPORT
