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
float evolvedBulkViscosityCoefficient(float bulkViscosityCoefficient,
    float targetBulkViscosityCoefficient, float inverseDecayTime, float timeStep);

/*
 * Price et al. (2018), equations (40)-(41): directed viscous pressure q^a_ab.
 * radialVelocity is v_ab . rhat_ab in the same velocity units as soundSpeed.
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
 *   ρ_a → density; c_s,a → soundSpeed; v_ab · r̂_ab → radialVelocity
 *   min(v_ab · r̂_ab, 0) → approachingRadialVelocity (clipped radial velocity in this helper)
 *   α_aᴬⱽ → bulkViscosityCoefficient; βᴬⱽ → vonNeumannRichtmyerCoefficient
 *   q_abᵃ → return value.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real artificialPressure(Real density, Real soundSpeed, Real radialVelocity,
    Real bulkViscosityCoefficient, Real vonNeumannRichtmyerCoefficient) {
    Real const approachingRadialVelocity = simdMin(radialVelocity, Real(0));
    return -simdBroadcast<Real>(0.5) * density *
        (bulkViscosityCoefficient * soundSpeed -
            vonNeumannRichtmyerCoefficient * approachingRadialVelocity) *
        approachingRadialVelocity;
}

/*
 * Equation (39): m_b q^a_ab / (rho_a^2 Omega_a). Equal gas masses and the
 * number-density support constraint give m_b/Omega_a = gasMass - gradHCorrection.
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
 *   q_abᵃ → viscousPressure; ρ_a → density; m_b → gasMass
 *   ζ_a → gradHCorrection; C_ab → return value.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real directedViscosityCoefficient(
    Real viscousPressure, Real density, Real gasMass, Real gradHCorrection) {
    return viscousPressure * (gasMass - gradHCorrection) / (density * density);
}

/*
 * Equations (46)-(48) and (51)-(54), evaluated from the Cullen & Dehnen appendix-B sums
 * accumulated by the density pass (the paper's equations (51)-(52) are that gradient
 * estimate). Cosmological variables follow cullen_and_dehnen::diagnostics: with
 * stored velocity w = a v_pec, the reconstructed velocity gradient is V' = a² ∇_r v_phys
 * and the acceleration gradient G = ∇_x F, so
 *
 *   a⁴ [D(div v_phys)/Dt - 3 dH/dt] = a tr(G) + 3 (a da/dt)² − tr(V'²),
 *
 * subtracting the homogeneous background from equation (53). The ratio h² A/c_s² is invariant when
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
 * Derived cosmological form used here subtracts the homogeneous divergence derivative:
 *   a⁴ [D(∇·v_phys)/Dt − 3Ḣ] = a tr(G) + 3(aȧ)² − tr(V′²).
 * Symbol directory:
 *   a → scaleFactor/localScaleFactor; aȧ → expansionRate; h_a → smoothingLength
 *   a c_s,a → soundSpeed; α_max → maximumBulkViscosityCoefficient
 *   V′ = a² ∇_r v_phys → velocityGradient; G = ∇_x F → accelerationGradient
 *   tr(V′) → velocityDivergence; tr(G) → accelerationDivergence
 *   tr(V′²) → gradientSquareTrace; |∇×v′|² → curlSquared
 *   a⁴ [D(∇·v_phys)/Dt − 3Ḣ] → divergenceDerivative; ξ_a → shearLimiter
 *   a⁴ A_a → shockIndicator; α_loc,a → result.targetBulkViscosityCoefficient
 *   σ_decay → decayCoefficient/localDecayCoefficient
 *   τ_conformal⁻¹ = σ_decay c_s,stored/(a h_a) → result.inverseDecayTime.
 * The stored implementation retains the explicit background subtraction described above.
 */
// clang-format on
template <typename Real, typename Counter>
CUDA_EXPORT inline cullen_and_dehnen::Result<Real> diagnosticsFromGradient(
    Real const (&velocityGradient)[3][3], Real scaleFactor, Real divergenceDerivative,
    Real smoothingLength, Real soundSpeed, Real maximumBulkViscosityCoefficient,
    Real decayCoefficient, Counter& floatingPointOperations) {
    Real const localScaleFactor = scaleFactor;
    Real const localDecayCoefficient = decayCoefficient;
    cullen_and_dehnen::Result<Real> result{};
    if (!(localScaleFactor > Real(0)) || !(smoothingLength > Real(0))) {
        return result;
    }
    result.inverseDecayTime =
        localDecayCoefficient * soundSpeed / (localScaleFactor * smoothingLength);
    Real const velocityDivergence =
        velocityGradient[0][0] + velocityGradient[1][1] + velocityGradient[2][2];
    Real const curlX = velocityGradient[2][1] - velocityGradient[1][2];
    Real const curlY = velocityGradient[0][2] - velocityGradient[2][0];
    Real const curlZ = velocityGradient[1][0] - velocityGradient[0][1];
    Real const curlSquared = curlX * curlX + curlY * curlY + curlZ * curlZ;
    floatingPointOperations += 10;    // Velocity trace and squared curl.
    result.divergence = velocityDivergence / (localScaleFactor * localScaleFactor);
    /*
     * Equation (53) in the stored variables, then equations (47), (46), and (48).
     */
    result.sampleDivergence = result.divergence;
    Real const divergenceSquared = velocityDivergence * velocityDivergence;
    Real const limiterDenominator = divergenceSquared + curlSquared;
    Real const shearLimiter =
        limiterDenominator > Real(0) ? divergenceSquared / limiterDenominator : Real(0);
    Real const shockIndicator =
        shearLimiter * (divergenceDerivative < Real(0) ? -divergenceDerivative : Real(0));
    /*
     * Equation (48)'s dimensionless prefactor; the paper gives no further justification.
     */
    constexpr Real kShockPrefactor = Real(10);
    Real const scaledSmoothingLengthSquaredShockIndicator =
        kShockPrefactor * smoothingLength * smoothingLength * shockIndicator;

    Real const soundSpeedSquared = soundSpeed * soundSpeed;
    Real localBulkViscosityTarget = Real(0);
    if (scaledSmoothingLengthSquaredShockIndicator > Real(0)) {
        localBulkViscosityTarget = scaledSmoothingLengthSquaredShockIndicator <
                maximumBulkViscosityCoefficient * soundSpeedSquared ?
            scaledSmoothingLengthSquaredShockIndicator / soundSpeedSquared :
            maximumBulkViscosityCoefficient;
    }
    result.targetBulkViscosityCoefficient = localBulkViscosityTarget;
    floatingPointOperations += 14 + (limiterDenominator > Real(0) ? 1 : 0) +
        (scaledSmoothingLengthSquaredShockIndicator > Real(0) ? 2 : 0);
    return result;
}

template <typename Real, typename Counter>
CUDA_EXPORT inline cullen_and_dehnen::Result<Real> diagnostics(
    cullen_and_dehnen::GradientSums<Real> const& sums, Real scaleFactor, Real expansionRate,
    Real smoothingLength, Real soundSpeed, Real maximumBulkViscosityCoefficient,
    Real decayCoefficient, Counter& floatingPointOperations) {
    Real velocityGradient[3][3];
    Real accelerationGradient[3][3];
    cullen_and_dehnen::reconstructGradients(
        sums, velocityGradient, accelerationGradient, floatingPointOperations);
    Real derivative = Real(3) * expansionRate * expansionRate;
    for (int i = 0; i < 3; ++i) {
        derivative += scaleFactor * accelerationGradient[i][i];
        for (int j = 0; j < 3; ++j) {
            derivative -= velocityGradient[i][j] * velocityGradient[j][i];
        }
    }
    floatingPointOperations += 26;
    return diagnosticsFromGradient(velocityGradient, scaleFactor, derivative, smoothingLength,
        soundSpeed, maximumBulkViscosityCoefficient, decayCoefficient, floatingPointOperations);
}

// Appendix B3 time differences retain this model's target and limiter unchanged.
template <typename Real, typename Counter>
CUDA_EXPORT inline cullen_and_dehnen::Result<Real> diagnosticsFromHistory(
    cullen_and_dehnen::VelocityGradientSums<Real> const& sums, Real scaleFactor, Real expansionRate,
    SphCullenDehnenState const& state, double sampleInterval, Real smoothingLength, Real soundSpeed,
    Real maximumBulkViscosityCoefficient, Real decayCoefficient, Counter& floatingPointOperations) {
    Real velocityGradient[3][3];
    int const rank = cullen_and_dehnen::reconstructVelocityGradient(
        sums, velocityGradient, floatingPointOperations);
    Real const divergence =
        (velocityGradient[0][0] + velocityGradient[1][1] + velocityGradient[2][2]) /
        (scaleFactor * scaleFactor);
    Real const sampleDivergence =
        divergence - Real(3) * expansionRate / (scaleFactor * scaleFactor);
    floatingPointOperations += 8;
    Real const derivative = rank == 0 ?
        Real(0) :
        cullen_and_dehnen::divergenceTimeDerivative(
            sampleDivergence, scaleFactor, state, sampleInterval, floatingPointOperations);
    auto result =
        diagnosticsFromGradient(velocityGradient, scaleFactor, derivative, smoothingLength,
            soundSpeed, maximumBulkViscosityCoefficient, decayCoefficient, floatingPointOperations);
    result.sampleDivergence = sampleDivergence;
    return result;
}

template <typename Real>
CUDA_EXPORT inline cullen_and_dehnen::Result<Real> diagnostics(
    cullen_and_dehnen::GradientSums<Real> const& sums, Real scaleFactor, Real expansionRate,
    Real smoothingLength, Real soundSpeed, Real maximumBulkViscosityCoefficient,
    Real decayCoefficient) {
    unsigned long long floatingPointOperations = 0;
    return diagnostics(sums, scaleFactor, expansionRate, smoothingLength, soundSpeed,
        maximumBulkViscosityCoefficient, decayCoefficient, floatingPointOperations);
}

}    // namespace price_etal
