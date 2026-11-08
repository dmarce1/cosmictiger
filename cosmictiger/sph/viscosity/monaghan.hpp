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
 * Equation numbers and pages below refer to Monaghan (1992). Comments retain its
 * symbols: particle a is the one being updated and particle b a neighbor; m mass,
 * ρ density, P pressure, u specific internal energy, c sound speed, h_M smoothing
 * length (h in the paper), r𞁑𞁓 = r𞁑 − r𞁓, v𞁑𞁓 = v𞁑 − v𞁓, W𞁑𞁓 = W(r𞁑𞁓, h_M) the kernel,
 * W̃𞁑𞁓 the symmetric kernel of Monaghan (1992), p. 554, Π𞁑𞁓 the artificial viscosity and μ𞁑𞁓 its
 * velocity measure, with the constants α, β and η. Equation code uses
 * these symbols; each equation block describes them. The stored smoothing length
 * h is the full compact support of the kernel and equals 2h_M for the cubic spline of
 * Monaghan (1992), section 7.
 */
#pragma once

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/simd_math.hpp>

namespace monaghan {

// Dimension-dependent cubic calibration (h/2); in 3D sqrt(5/6). Converts the common effective
// kernel radius (2 sigma) to the historical cubic-spline smoothing length used for viscosity.
inline constexpr double kViscosityWidthScale =
    (kSpatialDimensions == 1 ? 0.8660254037844386 :
                               (kSpatialDimensions == 2 ? 0.8890008890013335 : 0.9128709291752769));

// clang-format off
/*
 * Derived conversion from the stored smoothing length h to the Monaghan (1992) smoothing
 * length h_M: h̃ = h/C_H is the common effective radius of docs/kernels.md and
 * h_M = k_M h̃ with the calibration k_M = kViscosityWidthScale, so
 *   h_M/h = k_M/C_H,
 * one factor per kernel launch (1/2 for the cubic spline of Monaghan (1992), section 7).
 * Symbol directory:
 *   h → stored smoothing length; h̃ → common effective radius
 *   h_M → Monaghan (1992) smoothing length
 *   C_H → smoothing length per effective radius; k_M → viscosity width scale
 *   h_M/h → Monaghan to stored smoothing-length ratio (return value).
 *   Code names: smoothingLengthPerEffectiveRadius = C_H; kViscosityWidthScale = k_M.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT constexpr Real lengthPerSupport(Real smoothingLengthPerEffectiveRadius) {
    return Real(kViscosityWidthScale) / smoothingLengthPerEffectiveRadius;
}

// clang-format off
/*
 * Monaghan (1992), p. 554, unnumbered pair mean of the resolution lengths, written with the
 * stored supports h𞁑, h𞁓 and the launch factor h_M/h of lengthPerSupport:
 *   h̄_M𞁑𞁓 = ½(h_M𞁑 + h_M𞁓) = ½(h𞁑 + h𞁓) h_M/h.
 * Each pair forms it once and passes it to viscosityMu and sph_conduction::exchangeRate.
 * Symbol directory:
 *   h𞁑, h𞁓 → smoothing length, neighbor smoothing length
 *   h_M/h → Monaghan to stored smoothing-length ratio
 *   h̄_M𞁑𞁓 → pair smoothing length (return value).
 *   Code names: h_M_over_h = h_M/h.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real pairLength(Real h𞁑, Real h𞁓, Real h_M_over_h) {
    return simdBroadcast<Real>(0.5) * (h𞁑 + h𞁓) * h_M_over_h;
}

/*
 * Equation 4.2 of Monaghan (1992), p. 550, with the regularization of p. 551:
 *
 *   μ𞁑𞁓 = h_M v𞁑𞁓·r𞁑𞁓 / (r𞁑𞁓² + η²),      η² = 0.01 h_M²,
 *
 * and μ𞁑𞁓 = 0 when v𞁑𞁓·r𞁑𞁓 ≥ 0, the vanishing branch of Π𞁑𞁓 on p. 550 of Monaghan (1992). For
 * unequal resolution lengths the pair uses h̄_M𞁑𞁓 = ½(h_M𞁑 + h_M𞁓), the arithmetic mean that
 * Monaghan (1992), p. 554, offers; callers form it with pairLength. ε is η²/h_M².
 */
// clang-format off
/*
 * Monaghan (1992), equation (4.2), and the unnumbered pair means on p. 554
 * (doi:10.1146/annurev.aa.30.090192.002551; full citation above):
 *   η² = ε h̄_M𞁑𞁓²,
 *   μ𞁑𞁓 = h̄_M𞁑𞁓 (v𞁑𞁓 · r𞁑𞁓)/(r𞁑𞁓² + η²) for v𞁑𞁓 · r𞁑𞁓 < 0; otherwise 0.
 * Symbol directory:
 *   r𞁑𞁓² → radius squared; v𞁑𞁓 · r𞁑𞁓 → velocity dot radius
 *   h̄_M𞁑𞁓 → pair smoothing length (see pairLength); ε → viscosity epsilon
 *   η² → regularization length squared; μ𞁑𞁓 → viscosity velocity (return value).
 *   Code names: r2𞁑𞁓 = r𞁑𞁓²; v𞁑𞁓_dot_r𞁑𞁓 = v𞁑𞁓 · r𞁑𞁓; η2 = η².
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real viscosityMu(Real r2𞁑𞁓, Real v𞁑𞁓_dot_r𞁑𞁓, Real h̄_M𞁑𞁓, Real ε) {
    Real const η2 = ε * h̄_M𞁑𞁓 * h̄_M𞁑𞁓;    // η²
    Real const μ𞁑𞁓 = h̄_M𞁑𞁓 * v𞁑𞁓_dot_r𞁑𞁓 / (r2𞁑𞁓 + η2);
    return simdSelect(v𞁑𞁓_dot_r𞁑𞁓 < Real(0), μ𞁑𞁓, Real(0));
}

/*
 * Artificial viscosity Π𞁑𞁓 of Monaghan (1992), p. 550 (unnumbered, following equation 4.1):
 *
 *   Π𞁑𞁓 = (−α c̄𞁑𞁓 μ𞁑𞁓 + β μ𞁑𞁓²) / ρ̄𞁑𞁓    for v𞁑𞁓·r𞁑𞁓 < 0,
 *   Π𞁑𞁓 = 0                                    for v𞁑𞁓·r𞁑𞁓 > 0,
 *
 * with c̄𞁑𞁓 = ½(c𞁑 + c𞁓) and ρ̄𞁑𞁓 = ½(ρ𞁑 + ρ𞁓). Monaghan (1992)
 * recommends α = 1 and β = 2 (p. 551). Since μ𞁑𞁓 already vanishes for separating pairs, the branch
 * is selected on μ𞁑𞁓 < 0.
 */
// clang-format off
/*
 * Monaghan (1992), p. 550, unnumbered formula after equation (4.1):
 *   Π𞁑𞁓 = (−α c̄𞁑𞁓 μ𞁑𞁓 + β μ𞁑𞁓²)/ρ̄𞁑𞁓 when μ𞁑𞁓 < 0; otherwise 0.
 * Symbol directory:
 *   α → bulk viscosity coefficient; β → von Neumann–Richtmyer coefficient
 *   ρ𞁑, ρ𞁓 → sink density, source density; ρ̄𞁑𞁓 → mean density
 *   c𞁑, c𞁓 → sound speed, neighbor sound speed; c̄𞁑𞁓 → mean sound speed
 *   μ𞁑𞁓 → compression velocity; Π𞁑𞁓 → pair artificial viscosity (return value).
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real artificialViscosity(
    Real ρ𞁑, Real ρ𞁓, Real c𞁑, Real c𞁓, Real μ𞁑𞁓, Real α, Real β) {
    Real const c̄𞁑𞁓 = simdBroadcast<Real>(0.5) * (c𞁑 + c𞁓);
    Real const ρ̄𞁑𞁓 = simdBroadcast<Real>(0.5) * (ρ𞁑 + ρ𞁓);
    Real const Π𞁑𞁓 = (-α * c̄𞁑𞁓 * μ𞁑𞁓 + β * μ𞁑𞁓 * μ𞁑𞁓) / ρ̄𞁑𞁓;
    return simdSelect(μ𞁑𞁓 < Real(0), Π𞁑𞁓, Real(0));
}

}    // namespace monaghan
