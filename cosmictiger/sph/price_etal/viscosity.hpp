/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/sph/cullen_and_dehnen/viscosity.hpp>

// D. J. Price et al., "Phantom: A smoothed particle hydrodynamics and
// magnetohydrodynamics code for astrophysics", PASA 35 (2018), e031,
// doi:10.1017/pasa.2018.25, https://arxiv.org/abs/1702.03930, section 2.2.9.
// Their variation on the Cullen & Dehnen (2010) switch: the shock indicator of
// equation (46) is limited by the Balsara-type ratio (47) instead of the sign-weighted
// tr(S S^T) form, the target (48) is 10 h² A/c_s² capped at α_max = 1, the decay time
// uses the sound speed with σ_decay = 0.1, and the decay step (50) is backward Euler.
// The quadratic coefficient β^AV of equation (41) stays constant, independent of α.
// The paper's h is the smoothing length, with support R_kern h = 2h for the cubic
// spline (section 2.1.6); the caller supplies the calibrated viscosity length h_M of
// docs/monaghan_sph.md, which equals H/2 for the cubic spline.
namespace price_etal {

// Equation (50) with the instantaneous increase that precedes equation (49).
float evolvedAlpha(float alpha, float targetAlpha, float inverseDecayTime, float timeStep);

// NOLINTBEGIN(readability-identifier-naming)

// Equations (46)-(48) and (51)-(54), evaluated from the Cullen & Dehnen appendix-B sums
// accumulated by the density pass (the paper's equations (51)-(52) are that gradient
// estimate). Cosmological variables follow cullen_and_dehnen::diagnostics: with
// stored velocity w = a v_pec, the reconstructed velocity gradient is V' = a² ∇_r v_phys
// and the acceleration gradient G = ∇_x F, so
//
//   a⁴ D(div v_phys)/Dt = a tr(G) + 3 a³ d²a/dt² − tr(V'²),
//
// equation (53), and the ratio h² A/c_s² is invariant when h is comoving and c_s is a
// stored (comoving) velocity. ξ of equation (47) uses |∇·v|² and |∇×v|² from the same
// V', whose common a² scaling cancels. inverseDecayTime is the conformal-time rate
// σ_decay c_s/(a h) from the text following equation (45). The counter follows the
// repository's source-operation convention and may be FlopCounter<>.
template <typename Real, typename Counter>
CUDA_EXPORT inline cullen_and_dehnen::Result<Real> diagnostics(
    cullen_and_dehnen::GradientSums<Real> const& sums, Real scaleFactor, Real expansionAcceleration,
    Real viscosityLength, Real soundSpeed, Real alphaMaximum, Real decayCoefficient,
    Counter& floatingPointOperations) {
    Real const a = scaleFactor;
    Real const h_a = viscosityLength;
    Real const c_s_a = soundSpeed;
    Real const σ_decay = decayCoefficient;
    Real const α_max = alphaMaximum;
    cullen_and_dehnen::Result<Real> result{};
    if (!(a > Real(0)) || !(h_a > Real(0))) {
        return result;
    }
    result.inverseDecayTime = σ_decay * c_s_a / (a * h_a);
    Real V[3][3];
    Real G[3][3];
    floatingPointOperations += 3;
    if (cullen_and_dehnen::reconstructGradients(sums, V, G, floatingPointOperations) == 0) {
        return result;
    }
    // Equation (54): tr(V²) is Σ_ij V_ij V_ji, not the squared Frobenius norm.
    Real const div_v_a = V[0][0] + V[1][1] + V[2][2];
    Real const div_F_a = G[0][0] + G[1][1] + G[2][2];
    Real tr_V2 = Real(0);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            tr_V2 += V[i][j] * V[j][i];
        }
    }
    Real const curl_v_a_x = V[2][1] - V[1][2];
    Real const curl_v_a_y = V[0][2] - V[2][0];
    Real const curl_v_a_z = V[1][0] - V[0][1];
    Real const curl_v_a2 =
        curl_v_a_x * curl_v_a_x + curl_v_a_y * curl_v_a_y + curl_v_a_z * curl_v_a_z;
    floatingPointOperations += 30;    // Two traces, tr(V²), and the squared curl.
    result.divergence = div_v_a / (a * a);
    // Equation (53) in the stored variables, then equations (47), (46), and (48).
    Real const dot_div_v_a = a * div_F_a + Real(3) * expansionAcceleration - tr_V2;
    Real const div_v_a2 = div_v_a * div_v_a;
    Real const ξ_denominator = div_v_a2 + curl_v_a2;
    Real const ξ_a = ξ_denominator > Real(0) ? div_v_a2 / ξ_denominator : Real(0);
    Real const A_a = ξ_a * (dot_div_v_a < Real(0) ? -dot_div_v_a : Real(0));
    // Equation (48)'s dimensionless prefactor; the paper gives no further justification.
    constexpr Real kShockPrefactor = Real(10);
    Real const h2_A_a = kShockPrefactor * h_a * h_a * A_a;

    Real const c_s_a2 = c_s_a * c_s_a;
    Real α_loc_a = Real(0);
    if (h2_A_a > Real(0)) {
        α_loc_a = h2_A_a < α_max * c_s_a2 ? h2_A_a / c_s_a2 : α_max;
    }
    result.targetAlpha = α_loc_a;
    floatingPointOperations += 13 + (ξ_denominator > Real(0) ? 1 : 0) + (h2_A_a > Real(0) ? 2 : 0);
    return result;
}

template <typename Real>
CUDA_EXPORT inline cullen_and_dehnen::Result<Real> diagnostics(
    cullen_and_dehnen::GradientSums<Real> const& sums, Real scaleFactor, Real expansionAcceleration,
    Real viscosityLength, Real soundSpeed, Real alphaMaximum, Real decayCoefficient) {
    unsigned long long floatingPointOperations = 0;
    return diagnostics(sums, scaleFactor, expansionAcceleration, viscosityLength, soundSpeed,
        alphaMaximum, decayCoefficient, floatingPointOperations);
}

// NOLINTEND(readability-identifier-naming)

}    // namespace price_etal
