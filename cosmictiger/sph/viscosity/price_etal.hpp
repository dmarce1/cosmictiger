/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/sph/viscosity/cullen_and_dehnen.hpp>

/*
 * D. J. Price et al., "Phantom: A smoothed particle hydrodynamics and
 * magnetohydrodynamics code for astrophysics", PASA 35 (2018), e031,
 * doi:10.1017/pasa.2018.25, https://arxiv.org/abs/1702.03930, section 2.2.9.
 * Their variation on the Cullen & Dehnen (2010) switch: the shock indicator of
 * equation (46) is limited by the Balsara-type ratio (47) instead of the sign-weighted
 * tr(S S^T) form, the target (48) is 10 h² A/c_s² capped at α_max = 1, the decay time
 * uses the sound speed with σ_decay = 0.1, and the decay step (50) is backward Euler.
 * The quadratic coefficient β^AV of equation (41) stays constant, independent of α.
 * The paper's h is the smoothing length, with support R_kern h = 2h for the cubic
 * spline (section 2.1.6); the caller supplies the calibrated viscosity length h_M of
 * docs/monaghan_sph.md, which equals H/2 for the cubic spline.
 */
namespace price_etal {

/*
 * Equation (50) with the instantaneous increase that precedes equation (49).
 */
float evolvedBulkViscosityCoefficient(float α_aⁿ, float α_loc_a, float τ_a_inv, float Δt);

/*
 * Price et al. (2018), equations (40)-(41): directed viscous pressure q^a_ab.
 * v_ab_dot_r̂_ab is v_ab . rhat_ab in the same velocity units as c_s_a.
 * Each side uses its own alpha, sound speed and density; beta is independent of alpha.
 * Clamp before multiplication so separating and padded SIMD lanes give exactly zero.
 */
// clang-format off
/*
 * Price et al. (2018), equations (40)–(41), citation above:
 *   q_abᵃ = −½ ρ_a v_sig,a (v_ab · r̂_ab),
 *   v_sig,a = α_aᴬⱽ c_s,a + βᴬⱽ |v_ab · r̂_ab|,  for v_ab · r̂_ab < 0;
 *   q_abᵃ = 0 otherwise.
 * Symbol directory:
 *   ρ_a → density; c_s,a → sound speed; v_ab · r̂_ab → radial velocity
 *   min(v_ab · r̂_ab, 0) → approaching radial velocity (clipped radial velocity in this helper)
 *   α_aᴬⱽ → bulk viscosity coefficient; βᴬⱽ → von Neumann–Richtmyer coefficient
 *   q_abᵃ → return value.
 *   Code names: c_s_a = c_s,a; v_ab_dot_r̂_ab = v_ab · r̂_ab;
 *     min_v_ab_dot_r̂_ab_0 = min(v_ab · r̂_ab, 0).
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real artificialPressure(
    Real ρ_a, Real c_s_a, Real v_ab_dot_r̂_ab, Real α_aᴬⱽ, Real βᴬⱽ) {
    Real const min_v_ab_dot_r̂_ab_0 = simdMin(v_ab_dot_r̂_ab, Real(0));
    return -simdBroadcast<Real>(0.5) * ρ_a * (α_aᴬⱽ * c_s_a - βᴬⱽ * min_v_ab_dot_r̂_ab_0) *
        min_v_ab_dot_r̂_ab_0;
}

/*
 * Equation (39): m_b q^a_ab / (rho_a^2 Omega_a). Equal gas masses and the
 * number-density support constraint give m_b/Omega_a = m_b - ζ_a.
 * The same directed coefficient supplies the local shock heating of equation (42).
 */
// clang-format off
/*
 * Price et al. (2018), equations (39), (42):
 *   C_ab = m_b q_abᵃ/(ρ_a² Ω_a),
 *   (dv_a/dt)_AV = −Σ_b [C_ab ∇_a W_ab(h_a) + C_ba ∇_a W_ab(h_b)],
 *   (du_a/dt)_AV = Σ_b C_ab v_ab · ∇_a W_ab(h_a).
 * Derived equal-mass support relation: m_b/Ω_a = m_b − ζ_a.
 * Symbol directory:
 *   q_abᵃ → viscous pressure; ρ_a → density; m_b → gas mass
 *   ζ_a → grad-H correction; C_ab → return value.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real directedViscosityCoefficient(Real q_abᵃ, Real ρ_a, Real m_b, Real ζ_a) {
    return q_abᵃ * (m_b - ζ_a) / (ρ_a * ρ_a);
}

/*
 * Equations (46)-(48) and (51)-(54), evaluated from the Cullen & Dehnen appendix-B sums
 * accumulated by the density pass (the paper's equations (51)-(52) are that gradient
 * estimate). The nonproduction acceleration-gradient reference follows
 * cullen_and_dehnen::diagnostics: with
 * stored velocity w = a v_pec, the reconstructed velocity gradient is V' = a² ∇_r v_phys
 * and the acceleration gradient G = ∇_x F, so
 *
 *   a⁴ [D(div v_phys)/Dt - 3 dH/dt] = a tr(G) + 3 (a da/dt)² − tr(V'²),
 *
 * subtracting the homogeneous background in our derived cosmological extension.
 * Production instead forms V' = ∇_x w = a² ∇_r v_pec directly and differences its
 * trace before unit conversion, matching GADGET-4. The ratio h² A/c_s² is invariant when
 * h is comoving and c_s is a stored (comoving) velocity. ξ of equation (47) uses |∇·v|² and |∇×v|²
 * from the same V', whose common a² scaling cancels. inverseDecayTime is the conformal-time rate
 * σ_decay c_s/(a h) from the text following equation (45). The counter follows the
 * repository's source-operation convention and may be FlopCounter<>.
 */
// clang-format off
/*
 * Price et al. (2018), equations (46)–(48), (53)–(54), citation above:
 *   A_a = ξ_a max[−D(∇·v_a)/Dt, 0],                                      (46)
 *   ξ_a = |∇·v_a|²/(|∇·v_a|² + |∇×v_a|²),                               (47)
 *   α_loc,a = min(10h_a² A_a/c_s,a², α_max).                               (48)
 * Derived cosmological acceleration-gradient reference subtracts the background derivative:
 *   a⁴ [D(∇·v_phys)/Dt − 3Ḣ] = a tr(G) + 3(aȧ)² − tr(V′²).
 * Symbol directory:
 *   a → scale factor; aȧ → expansion rate; h_a → smoothing length
 *   a c_s,a → sound speed; α_max → maximum bulk viscosity coefficient
 *   V′ = a² ∇_r v_phys → velocity gradient; G = ∇_x F → acceleration gradient
 *   tr(V′) → velocity divergence; tr(G) → acceleration divergence
 *   tr(V′²) → gradient square trace; |∇×v′|² → curl squared
 *   a⁴ [D(∇·v_phys)/Dt − 3Ḣ] → divergence derivative; ξ_a → shear limiter
 *   a⁴ A_a → shock indicator; α_loc,a → target bulk viscosity coefficient
 *   σ_decay → decay coefficient
 *   τ_conformal⁻¹ = σ_decay c_s,stored/(a h_a) → inverse decay time.
 *   Code names: a_c_s_a = a c_s,a; Vʹ = V′; tr_Vʹ = tr(V′); abs_curl_vʹ2 = |∇×v′|²;
 *     a4_D_div_v_phys_Dt_minus_3Ḣ = a⁴ [D(∇·v_phys)/Dt − 3Ḣ]; a4_A_a = a⁴ A_a; α_loc_a = α_loc,a.
 * Production history uses peculiar gradients and their time differences instead,
 * without introducing a homogeneous-background term into the target or limiter.
 */
// clang-format on
template <typename Real, typename Counter>
CUDA_EXPORT inline cullen_and_dehnen::Result<Real> diagnosticsFromGradient(Real const (&Vʹ)[3][3],
    Real scaleFactor, Real a4_D_div_v_phys_Dt_minus_3Ḣ, Real h_a, Real a_c_s_a, Real α_max,
    Real decayCoefficient, Counter& floatingPointOperations) {
    Real const a = scaleFactor;
    Real const σ_decay = decayCoefficient;
    cullen_and_dehnen::Result<Real> result{};
    if (!(a > Real(0)) || !(h_a > Real(0))) {
        return result;
    }
    result.inverseDecayTime = σ_decay * a_c_s_a / (a * h_a);
    Real const tr_Vʹ = Vʹ[0][0] + Vʹ[1][1] + Vʹ[2][2];
    Real const curlX = Vʹ[2][1] - Vʹ[1][2];
    Real const curlY = Vʹ[0][2] - Vʹ[2][0];
    Real const curlZ = Vʹ[1][0] - Vʹ[0][1];
    Real const abs_curl_vʹ2 = curlX * curlX + curlY * curlY + curlZ * curlZ;
    floatingPointOperations += 10;    // Velocity trace and squared curl.
    result.divergence = tr_Vʹ / (a * a);
    /*
     * Equation (53) in the stored variables, then equations (47), (46), and (48).
     */
    Real const divergenceSquared = tr_Vʹ * tr_Vʹ;
    Real const limiterDenominator = divergenceSquared + abs_curl_vʹ2;
    Real const ξ_a =
        limiterDenominator > Real(0) ? divergenceSquared / limiterDenominator : Real(0);
    Real const a4_A_a =
        ξ_a * (a4_D_div_v_phys_Dt_minus_3Ḣ < Real(0) ? -a4_D_div_v_phys_Dt_minus_3Ḣ : Real(0));
    /*
     * Equation (48)'s dimensionless prefactor; the paper gives no further justification.
     */
    constexpr Real kShockPrefactor = Real(10);
    Real const scaledSmoothingLengthSquaredShockIndicator = kShockPrefactor * h_a * h_a * a4_A_a;

    Real const soundSpeedSquared = a_c_s_a * a_c_s_a;
    Real localBulkViscosityTarget = Real(0);
    if (scaledSmoothingLengthSquaredShockIndicator > Real(0)) {
        localBulkViscosityTarget =
            scaledSmoothingLengthSquaredShockIndicator < α_max * soundSpeedSquared ?
            scaledSmoothingLengthSquaredShockIndicator / soundSpeedSquared :
            α_max;
    }
    result.targetBulkViscosityCoefficient = localBulkViscosityTarget;
    floatingPointOperations += 14 + (limiterDenominator > Real(0) ? 1 : 0) +
        (scaledSmoothingLengthSquaredShockIndicator > Real(0) ? 2 : 0);
    return result;
}

template <typename Real, typename Counter>
CUDA_EXPORT inline cullen_and_dehnen::Result<Real> diagnostics(
    cullen_and_dehnen::GradientSums<Real> const& sums, Real a, Real aȧ, Real h_a, Real a_c_s_a,
    Real α_max, Real σ_decay, Counter& floatingPointOperations) {
    Real Vʹ[3][3];
    Real G[3][3];
    cullen_and_dehnen::reconstructGradients(sums, Vʹ, G, floatingPointOperations);
    Real a4_D_div_v_phys_Dt_minus_3Ḣ = Real(3) * aȧ * aȧ;
    for (int i = 0; i < 3; ++i) {
        a4_D_div_v_phys_Dt_minus_3Ḣ += a * G[i][i];
        for (int j = 0; j < 3; ++j) {
            a4_D_div_v_phys_Dt_minus_3Ḣ -= Vʹ[i][j] * Vʹ[j][i];
        }
    }
    floatingPointOperations += 26;
    return diagnosticsFromGradient(
        Vʹ, a, a4_D_div_v_phys_Dt_minus_3Ḣ, h_a, a_c_s_a, α_max, σ_decay, floatingPointOperations);
}

// Production reconstructs peculiar gradients directly, so homogeneous expansion
// never enters the Appendix-B3 history. No separate subtraction of 3H is needed.
// The target and limiter formulas are unchanged; the acceleration-gradient path
// above remains a nonproduction reference.
template <typename Real, typename Counter>
CUDA_EXPORT inline cullen_and_dehnen::Result<Real> diagnosticsFromHistory(
    cullen_and_dehnen::VelocityGradientSums<Real> const& sums, Real a, Float16Storage const& state,
    double sampleInterval, double previousSampleTimestep, Real h_a, Real a_c_s_a, Real α_max,
    Real σ_decay, Counter& floatingPointOperations) {
    Real Vʹ[3][3];
    int const rank =
        cullen_and_dehnen::reconstructVelocityGradient(sums, Vʹ, floatingPointOperations);
    Real const divergence = Vʹ[0][0] + Vʹ[1][1] + Vʹ[2][2];
    floatingPointOperations += 2;
    Real const a4_D_div_v_phys_Dt_minus_3Ḣ = rank == 0 ?
        Real(0) :
        cullen_and_dehnen::divergenceTimeDerivative(
            divergence, a, state, sampleInterval, previousSampleTimestep, floatingPointOperations);
    auto result = diagnosticsFromGradient(
        Vʹ, a, a4_D_div_v_phys_Dt_minus_3Ḣ, h_a, a_c_s_a, α_max, σ_decay, floatingPointOperations);
    result.divergence = divergence;
    return result;
}

template <typename Real>
CUDA_EXPORT inline cullen_and_dehnen::Result<Real> diagnostics(
    cullen_and_dehnen::GradientSums<Real> const& sums, Real a, Real aȧ, Real h_a, Real a_c_s_a,
    Real α_max, Real σ_decay) {
    unsigned long long floatingPointOperations = 0;
    return diagnostics(sums, a, aȧ, h_a, a_c_s_a, α_max, σ_decay, floatingPointOperations);
}

}    // namespace price_etal
