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
 *   g → conduction coefficient; m_b/a → scaled gas mass; ρ_a,ρ_b → density, neighbor density;
 *   c_E,a,c_E,b → sound speed, neighbor sound speed; H_a,H_b → smoothing length, neighbor smoothing length;
 *   r_ab² → separation squared; (a v_phys,ab)·x_ab → physical velocity dot separation;
 *   ∇_a W̃_ab = F_ab x_ab → symmetric gradient factor; ε → viscosity epsilon;
 *   h̄_ab → pair smoothing length; η²+r_ab² → regularized separation squared;
 *   μ_E,ab → compression velocity; q_a+q_b → pair diffusivity; C_ab/a → return value.
 *   Code names: m_b_over_a = m_b/a; c_E_a = c_E,a; c_E_b = c_E,b; r_ab2 = r_ab²;
 *     a_v_phys_ab_dot_x_ab = (a v_phys,ab)·x_ab; η2_plus_r_ab2 = η²+r_ab²; μ_E_ab = μ_E,ab;
 *     q_a_plus_q_b = q_a+q_b.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real exchangeRate(Real ρ_a, Real ρ_b, Real c_E_a, Real c_E_b, Real H_a, Real H_b,
    Real r_ab2, Real a_v_phys_ab_dot_x_ab, Real F_ab, Real g, Real ε,
    Real smoothingLengthPerEffectiveRadius, Real m_b_over_a) {
    Real const h̄_ab = simdBroadcast<Real>(0.5) * (H_a + H_b) / smoothingLengthPerEffectiveRadius *
        simdBroadcast<Real>(monaghan::kViscosityWidthScale);
    Real const η2_plus_r_ab2 = r_ab2 + ε * h̄_ab * h̄_ab;
    Real const μ_E_ab = simdMin(h̄_ab * a_v_phys_ab_dot_x_ab / η2_plus_r_ab2, Real(0));
    Real const meanSoundSpeed = simdBroadcast<Real>(0.5) * (c_E_a + c_E_b);
    Real const meanDensity = simdBroadcast<Real>(0.5) * (ρ_a + ρ_b);
    Real const q_a_plus_q_b = g * h̄_ab * (meanSoundSpeed - Real(4) * μ_E_ab);
    return -m_b_over_a * q_a_plus_q_b * r_ab2 * F_ab / (meanDensity * η2_plus_r_ab2);
}

}    // namespace sph_conduction
