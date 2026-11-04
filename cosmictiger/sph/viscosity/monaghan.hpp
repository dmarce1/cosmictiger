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

namespace monaghan {

// sqrt(5/6): converts the common effective kernel radius (2 sigma) to the
// historical cubic-spline smoothing length used for viscosity.
inline constexpr double kViscosityWidthScale = 0.9128709291752769;

/*
 * Equation 4.2, p. 550, with the regularization of p. 551:
 *
 *   μ_ab = h v_ab·r_ab / (r_ab² + η²),      η² = 0.01 h²,
 *
 * and μ_ab = 0 when v_ab·r_ab ≥ 0, the vanishing branch of Π_ab on p. 550. For
 * unequal resolution lengths the pair uses pairSmoothingLength = ½(smoothingLength +
 * neighborSmoothingLength), the arithmetic mean that p. 554 offers. viscosityEpsilon is η²/h². The
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
 *   r_ab² → radiusSquared; v_ab · r_ab → velocityDotRadius
 *   H_a, H_b → smoothingLength, sourceSmoothingLength (stored compact supports)
 *   h̄_ab → pairSmoothingLength; ε → viscosityEpsilon
 *   η² → regularizationLengthSquared; μ_ab → viscosityVelocity (return value)
 *   h_a = H_a / smoothingLengthPerEffectiveRadius × viscosityWidthScale.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real viscosityMu(Real radiusSquared, Real velocityDotRadius,
    Real smoothingLength, Real sourceSmoothingLength, Real viscosityEpsilon,
    Real smoothingLengthPerEffectiveRadius = Real(2), Real viscosityWidthScale = Real(1)) {
    Real const separationSquared = radiusSquared;    // |r_ab|²
    Real const velocityDotSeparation = velocityDotRadius;
    Real const sinkKernelSmoothingLength = smoothingLength;
    Real const neighborSmoothingLength = sourceSmoothingLength;
    Real const pairSmoothingLength = simdBroadcast<Real>(0.5) *
        (sinkKernelSmoothingLength + neighborSmoothingLength) / smoothingLengthPerEffectiveRadius *
        viscosityWidthScale;
    Real const regularizationLengthSquared =
        viscosityEpsilon * pairSmoothingLength * pairSmoothingLength;    // η²
    Real const viscosityVelocity = pairSmoothingLength * velocityDotSeparation /
        (separationSquared + regularizationLengthSquared);
    return simdSelect(velocityDotSeparation < Real(0), viscosityVelocity, Real(0));
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
 *   α → bulkViscosityCoefficient; β → vonNeumannRichtmyerCoefficient
 *   ρ_a, ρ_b → sinkDensity, sourceDensity; ρ̄_ab → meanDensity
 *   c_a, c_b → soundSpeed, neighborSoundSpeed; c̄_ab → meanSoundSpeed
 *   μ_ab → compressionVelocity; Π_ab → pairArtificialViscosity (return value).
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real artificialViscosity(Real sinkDensity, Real sourceDensity, Real soundSpeed,
    Real neighborSoundSpeed, Real compressionVelocity, Real bulkViscosityCoefficient,
    Real vonNeumannRichtmyerCoefficient) {
    Real const density = sinkDensity;
    Real const neighborDensity = sourceDensity;
    Real const viscosityVelocity = compressionVelocity;
    Real const meanSoundSpeed = simdBroadcast<Real>(0.5) * (soundSpeed + neighborSoundSpeed);
    Real const meanDensity = simdBroadcast<Real>(0.5) * (density + neighborDensity);
    Real const pairArtificialViscosity =
        (-bulkViscosityCoefficient * meanSoundSpeed * viscosityVelocity +
            vonNeumannRichtmyerCoefficient * viscosityVelocity * viscosityVelocity) /
        meanDensity;
    return simdSelect(viscosityVelocity < Real(0), pairArtificialViscosity, Real(0));
}

}    // namespace monaghan
