/* CosmicTiger - artificial SPH thermal conduction; BSL-1.0. */
#pragma once

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/simd_math.hpp>

namespace sph_conduction {

// clang-format off
/*
 * Monaghan (1992), ARA&A 30, 543–574, equations (4.4)–(4.8), p. 552:
 * https://articles.adsabs.harvard.edu/pdf/1992ARA%26A..30..543M
 *   q𞁑 + q𞁓 = g h̄_M𞁑𞁓 (c̄𞁑𞁓 + 4|μ𞁑𞁓|),  η² = ε h̄_M𞁑𞁓²,
 *   C𞁑𞁓 = −m𞁓 (q𞁑+q𞁓) [r𞁑𞁓 · ∇𞁑 W̃𞁑𞁓]/[ρ̄𞁑𞁓(r𞁑𞁓²+η²)] ≥ 0,
 *   (du𞁑/dt)𞁞ₒₙ = Σ𞁓 C𞁑𞁓 (u𞁓−u𞁑).
 * With ∇𞁑 W̃𞁑𞁓 directed down the kernel, this uses the dissipative sign
 * required by Monaghan (1992), equation (4.4), and the Gaussian example (4.6). The
 * printed leading minus in Monaghan (1992), equation (4.5), with (u𞁑−u𞁓) is
 * inconsistent with that gradient convention. Pair means and W̃𞁑𞁓 follow
 * Monaghan (1992), p. 554. This is artificial diffusion, including
 * the sound-speed term for nonapproaching pairs; no viscosity/shear switch is applied.
 *
 * Derived comoving conversion: x = rₚₕ𞁟ₛ/a, ρ𞁞 = a³ρₚₕ𞁟ₛ, E = a²uₚₕ𞁟ₛ,
 * c_E = a cₚₕ𞁟ₛ, μ_E = a μₚₕ𞁟ₛ, dt = a dτ. The same expression using comoving
 * lengths, densities and E velocities gives dE𞁑/dτ = Σ𞁓 (C𞁑𞁓/a)(E𞁓−E𞁑).
 * The return value below is C𞁑𞁓/a. The caller passes the pair length h̄_M𞁑𞁓 of
 * monaghan::pairLength, shared with the Monaghan viscosity of the same pair.
 * Symbol directory:
 *   g → conduction coefficient; m𞁓/a → scaled gas mass; ρ𞁑,ρ𞁓 → density, neighbor density;
 *   c_E𞁑,c_E𞁓 → sound speed, neighbor sound speed;
 *   r𞁑𞁓² → separation squared; (a vₚₕ𞁟ₛ𞁑𞁓)·x𞁑𞁓 → physical velocity dot separation;
 *   ∇𞁑 W̃𞁑𞁓 = F𞁑𞁓 x𞁑𞁓 → symmetric gradient factor; ε → viscosity epsilon;
 *   h̄_M𞁑𞁓 → pair smoothing length (monaghan::pairLength); η²+r𞁑𞁓² → regularized separation squared;
 *   μ_E𞁑𞁓 → compression velocity; q𞁑+q𞁓 → pair diffusivity; C𞁑𞁓/a → return value;
 *   c̄𞁑𞁓, ρ̄𞁑𞁓 → mean sound speed, mean density.
 *   Code names: m𞁓_over_a = m𞁓/a; r2𞁑𞁓 = r𞁑𞁓²
 *     a_vₚₕ𞁟ₛ𞁑𞁓_dot_x𞁑𞁓 = (a vₚₕ𞁟ₛ𞁑𞁓)·x𞁑𞁓; η2_plus_r2𞁑𞁓 = η²+r𞁑𞁓²
 *     q𞁑_plus_q𞁓 = q𞁑+q𞁓.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real exchangeRate(Real ρ𞁑, Real ρ𞁓, Real c_E𞁑, Real c_E𞁓, Real h̄_M𞁑𞁓, Real r2𞁑𞁓,
    Real a_vₚₕ𞁟ₛ𞁑𞁓_dot_x𞁑𞁓, Real F𞁑𞁓, Real g, Real ε, Real m𞁓_over_a) {
    Real const η2_plus_r2𞁑𞁓 = r2𞁑𞁓 + ε * h̄_M𞁑𞁓 * h̄_M𞁑𞁓;
    Real const μ_E𞁑𞁓 = math::min(h̄_M𞁑𞁓 * a_vₚₕ𞁟ₛ𞁑𞁓_dot_x𞁑𞁓 / η2_plus_r2𞁑𞁓, Real(0));
    Real const c̄𞁑𞁓 = math::broadcast<Real>(0.5) * (c_E𞁑 + c_E𞁓);
    Real const ρ̄𞁑𞁓 = math::broadcast<Real>(0.5) * (ρ𞁑 + ρ𞁓);
    Real const q𞁑_plus_q𞁓 = g * h̄_M𞁑𞁓 * (c̄𞁑𞁓 - Real(4) * μ_E𞁑𞁓);
    return -m𞁓_over_a * q𞁑_plus_q𞁓 * r2𞁑𞁓 * F𞁑𞁓 / (ρ̄𞁑𞁓 * η2_plus_r2𞁑𞁓);
}

}    // namespace sph_conduction
