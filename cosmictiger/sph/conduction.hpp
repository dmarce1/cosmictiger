/* CosmicTiger - artificial SPH thermal conduction; BSL-1.0. */
#pragma once

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/simd_math.hpp>
#include <cosmictiger/sph/viscosity/monaghan.hpp>

namespace sph_conduction {

// clang-format off
/*
 * Monaghan (1992), ARA&A 30, 543–574, equations (4.4)–(4.8), p. 552:
 * https://articles.adsabs.harvard.edu/pdf/1992ARA%26A..30..543M
 *   q_a + q_b = g h̄_ab (c̄_ab + 4|μ_ab|),  η² = ε h̄_ab²,
 *   C_ab = −m_b (q_a+q_b) [r_ab · ∇_a W̃_ab]/[ρ̄_ab(r_ab²+η²)] ≥ 0,
 *   (du_a/dt)_cond = Σ_b C_ab (u_b−u_a).
 * With ∇_a W̃_ab directed down the kernel, this uses the dissipative sign
 * required by (4.4) and the Gaussian example (4.6). The printed leading minus
 * in (4.5) with (u_a−u_b) is inconsistent with that gradient convention.
 * Pair means and W̃_ab follow p. 554. This is artificial diffusion, including
 * the sound-speed term for nonapproaching pairs; no viscosity/shear switch is applied.
 *
 * Derived comoving conversion: x = r_phys/a, ρ_c = a³ρ_phys, E = a²u_phys,
 * c_E = a c_phys, μ_E = a μ_phys, dt = a dτ. The same expression using comoving
 * lengths, densities and E velocities gives dE_a/dτ = Σ_b (C_ab/a)(E_b−E_a).
 * The return value below is C_ab/a. H is the stored compact support; h̄ uses
 * the same calibrated effective-radius conversion as Monaghan viscosity.
 * Symbol directory:
 *   g → conductionCoefficient; m_b/a → scaledGasMass; ρ_a,ρ_b → density,neighborDensity;
 *   c_E,a,c_E,b → soundSpeed,neighborSoundSpeed; H_a,H_b → smoothingLength,neighborSmoothingLength;
 *   r_ab² → separationSquared; (a v_phys,ab)·x_ab → physicalVelocityDotSeparation;
 *   ∇_a W̃_ab = F_ab x_ab → symmetricGradientFactor; ε → viscosityEpsilon;
 *   h̄_ab → pairSmoothingLength; η²+r_ab² → regularizedSeparationSquared;
 *   μ_E,ab → compressionVelocity; q_a+q_b → pairDiffusivity; C_ab/a → return value.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real exchangeRate(Real density, Real neighborDensity, Real soundSpeed,
    Real neighborSoundSpeed, Real smoothingLength, Real neighborSmoothingLength,
    Real separationSquared, Real physicalVelocityDotSeparation, Real symmetricGradientFactor,
    Real conductionCoefficient, Real viscosityEpsilon, Real smoothingLengthPerEffectiveRadius,
    Real scaledGasMass) {
    Real const pairSmoothingLength = simdBroadcast<Real>(0.5) *
        (smoothingLength + neighborSmoothingLength) / smoothingLengthPerEffectiveRadius *
        simdBroadcast<Real>(monaghan::kViscosityWidthScale);
    Real const regularizedSeparationSquared =
        separationSquared + viscosityEpsilon * pairSmoothingLength * pairSmoothingLength;
    Real const compressionVelocity =
        simdMin(pairSmoothingLength * physicalVelocityDotSeparation / regularizedSeparationSquared,
            Real(0));
    Real const meanSoundSpeed = simdBroadcast<Real>(0.5) * (soundSpeed + neighborSoundSpeed);
    Real const meanDensity = simdBroadcast<Real>(0.5) * (density + neighborDensity);
    Real const pairDiffusivity = conductionCoefficient * pairSmoothingLength *
        (meanSoundSpeed - Real(4) * compressionVelocity);
    return -scaledGasMass * pairDiffusivity * separationSquared * symmetricGradientFactor /
        (meanDensity * regularizedSeparationSquared);
}

}    // namespace sph_conduction
