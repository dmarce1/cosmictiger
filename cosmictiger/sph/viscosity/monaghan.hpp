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
 * velocity measure, with the constants α, β and η. Equation code uses
 * these symbols; each equation block describes them. The stored smoothing length
 * is the compact support radius H, which equals 2h for the paper's cubic spline of section 7.
 */
#pragma once

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/simd_math.hpp>

namespace monaghan {

// Dimension-dependent cubic calibration (H/2); in 3D sqrt(5/6). Converts the common effective
// kernel radius (2 sigma) to the historical cubic-spline smoothing length used for viscosity.
inline constexpr double kViscosityWidthScale =
    (kSpatialDimensions == 1 ? 0.8660254037844386 :
                               (kSpatialDimensions == 2 ? 0.8890008890013335 : 0.9128709291752769));

/*
 * Equation 4.2, p. 550, with the regularization of p. 551:
 *
 *   μ_ab = h v_ab·r_ab / (r_ab² + η²),      η² = 0.01 h²,
 *
 * and μ_ab = 0 when v_ab·r_ab ≥ 0, the vanishing branch of Π_ab on p. 550. For
 * unequal resolution lengths the pair uses h̄_ab = ½(h_a + h_b), the arithmetic mean that
 * p. 554 offers. ε is η²/h². The
 * two scale factors convert the stored support radii to the paper's h:
 * smoothingLengthPerEffectiveRadius maps H to the common effective radius of docs/kernels.md, and
 * viscosityWidthScale maps that radius to h. Their defaults give h = H/2, the cubic-spline
 * convention, for the standalone references in tests/sph/monaghan/equations.cpp.
 */
// clang-format off
/*
 * Monaghan (1992), equation (4.2), and the unnumbered pair means on p. 554
 * (doi:10.1146/annurev.aa.30.090192.002551; full citation above):
 *   h̄_ab = ½(h_a + h_b),  η² = ε h̄_ab²,
 *   μ_ab = h̄_ab (v_ab · r_ab)/(r_ab² + η²) for v_ab · r_ab < 0; otherwise 0.
 * Symbol directory:
 *   r_ab² → radius squared; v_ab · r_ab → velocity dot radius
 *   H_a, H_b → smoothing length, source smoothing length (stored compact supports)
 *   h̄_ab → pair smoothing length; ε → viscosity epsilon
 *   η² → regularization length squared; μ_ab → viscosity velocity (return value)
 *   h_a = H_a / smoothingLengthPerEffectiveRadius × viscosityWidthScale.
 *   Code names: r_ab2 = r_ab²; v_ab_dot_r_ab = v_ab · r_ab; η2 = η².
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real viscosityMu(Real radiusSquared, Real velocityDotRadius,
    Real smoothingLength, Real sourceSmoothingLength, Real ε,
    Real smoothingLengthPerEffectiveRadius = Real(2), Real viscosityWidthScale = Real(1)) {
    Real const r_ab2 = radiusSquared;    // |r_ab|²
    Real const v_ab_dot_r_ab = velocityDotRadius;
    Real const H_a = smoothingLength;
    Real const H_b = sourceSmoothingLength;
    Real const h̄_ab = simdBroadcast<Real>(0.5) * (H_a + H_b) / smoothingLengthPerEffectiveRadius *
        viscosityWidthScale;
    Real const η2 = ε * h̄_ab * h̄_ab;    // η²
    Real const μ_ab = h̄_ab * v_ab_dot_r_ab / (r_ab2 + η2);
    return simdSelect(v_ab_dot_r_ab < Real(0), μ_ab, Real(0));
}

/*
 * Artificial viscosity Π_ab, p. 550 (unnumbered, following equation 4.1):
 *
 *   Π_ab = (−α c̄_ab μ_ab + β μ_ab²) / ρ̄_ab    for v_ab·r_ab < 0,
 *   Π_ab = 0                                    for v_ab·r_ab > 0,
 *
 * with c̄_ab = ½(c_a + c_b) and ρ̄_ab = ½(ρ_a + ρ_b). The paper
 * recommends α = 1 and β = 2 (p. 551). Since μ_ab already vanishes for separating pairs, the branch
 * is selected on μ_ab < 0.
 */
// clang-format off
/*
 * Monaghan (1992), p. 550, unnumbered formula after equation (4.1):
 *   Π_ab = (−α c̄_ab μ_ab + β μ_ab²)/ρ̄_ab when μ_ab < 0; otherwise 0.
 * Symbol directory:
 *   α → bulk viscosity coefficient; β → von Neumann–Richtmyer coefficient
 *   ρ_a, ρ_b → sink density, source density; ρ̄_ab → mean density
 *   c_a, c_b → sound speed, neighbor sound speed; c̄_ab → mean sound speed
 *   μ_ab → compression velocity; Π_ab → pair artificial viscosity (return value).
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real artificialViscosity(Real sinkDensity, Real sourceDensity, Real c_a,
    Real c_b, Real compressionVelocity, Real α, Real β) {
    Real const ρ_a = sinkDensity;
    Real const ρ_b = sourceDensity;
    Real const μ_ab = compressionVelocity;
    Real const c̄_ab = simdBroadcast<Real>(0.5) * (c_a + c_b);
    Real const ρ̄_ab = simdBroadcast<Real>(0.5) * (ρ_a + ρ_b);
    Real const Π_ab = (-α * c̄_ab * μ_ab + β * μ_ab * μ_ab) / ρ̄_ab;
    return simdSelect(μ_ab < Real(0), Π_ab, Real(0));
}

}    // namespace monaghan
