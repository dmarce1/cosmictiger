/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cmath>

#include <cosmictiger/sph/viscosity/cullen_and_dehnen.hpp>

// Hu et al. (2014), SPHGal, MNRAS 443, 1173, equations (16)-(19).
// https://arxiv.org/abs/1402.1788
namespace hu_etal {

// Equation (17): compare with the unlimited target, then limit the entire update.
// Applying the limiter only to the target would leave old viscosity in shear flows.
// clang-format off
/*
 * Hu et al. (2014), SPHGal, MNRAS 443, 1173, https://arxiv.org/abs/1402.1788,
 * equation (17):
 *   α_iⁿ⁺¹ = ξ_i α_tar,i                                  if α_iⁿ ≤ α_tar,i,
 *   α_iⁿ⁺¹ = ξ_i [α_tar,i + (α_iⁿ − α_tar,i)e^(−Δt/τ_i)]   otherwise.
 * Symbol directory:
 *   α_iⁿ → bulk viscosity coefficient; α_tar,i → target bulk viscosity coefficient
 *   ξ_i → limiter; Δt → time step; τ_i⁻¹ → inverse decay time
 *   bracket before ξ_i → updated bulk viscosity coefficient; α_iⁿ⁺¹ → return value.
 *   Code names: α_tar_i = α_tar,i; τ_i_inv = τ_i⁻¹.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real evolvedBulkViscosityCoefficient(
    Real α_iⁿ, Real α_tar_i, Real τ_i_inv, Real ξ_i, Real Δt) {
    Real const updatedBulkViscosityCoefficient =
        α_iⁿ <= α_tar_i ? α_tar_i : α_tar_i + (α_iⁿ - α_tar_i) * std::exp(-Δt * τ_i_inv);
    return ξ_i * updatedBulkViscosityCoefficient;
}

// The acceleration-gradient reference uses CD's corrected reconstruction:
// V = a^2 grad_r(v_phys), G = grad_x(F), and a3_ä = a^3 d^2a/dt^2.
// The unscaled material derivative in these variables is a^4 D(div v_phys)/Dt.
// H is Hu's full kernel support radius. Stored c = a c_phys makes both H^2 Q/c^2
// and the regularized limiter invariant; inverseDecayTime is a conformal-time rate.
// Production history instead supplies V = ∇_x w and a² D(tr V)/Dt, following
// GADGET-4's convention of differencing its stored gradient before conversion.
// clang-format off
/*
 * Hu et al. (2014), equations (16), (18), (19); citation above:
 *   S_i = max(0, −D(∇·v_i)/Dt), α_tar,i = α_max h_i² S_i/(h_i² S_i + c_i²), (16)
 *   v_dec = max_j [c̄_ij − min(0, v_ij · x̂_ij)],                          (18)
 *   ξ_i = |∇·v_i|²/(|∇·v_i|² + |∇×v_i|² + 10⁻⁴(c_i/h_i)²).             (19)
 * The paper uses τ_i = 10h_i/v_dec; the configurable form is h_i/(2ℓ v_dec), ℓ = 0.05.
 * Symbol directory in the stored cosmological variables:
 *   h_i → smoothing length; a c_i → sound speed; a v_dec → signal speed
 *   α_max → maximum bulk viscosity coefficient; α_tar,i → target bulk viscosity coefficient
 *   a → scale factor; a³ ä → expansion acceleration; ℓ → decay coefficient
 *   a² ∇_r v_phys → velocity gradient; ∇_x F → acceleration gradient
 *   a² ∇·v_i → divergence; a⁴ D(∇·v_i)/Dt → divergence derivative
 *   a⁴ S_i → shock indicator; h_i² a⁴ S_i → radius squared shock
 *   |a² ∇×v_i|² → curl squared; ξ_i → limiter
 *   τ_conformal⁻¹ → inverse decay time.
 *   Code names: a_c_i = a c_i; a_v_dec = a v_dec; α_tar_i = α_tar,i; a3_ä = a³ ä;
 *     a2_grad_r_v_phys = a² ∇_r v_phys; grad_x_F = ∇_x F; a2_div_v_i = a² ∇·v_i;
 *     a4_D_div_v_i_Dt = a⁴ D(∇·v_i)/Dt; a4_S_i = a⁴ S_i; h_i2_a4_S_i = h_i² a⁴ S_i;
 *     abs_a2_curl_v_i2 = |a² ∇×v_i|².
 * The neighbor maximum (18) is accumulated in the CPU/CUDA density pass.
 */
// clang-format on
template <typename Real, typename Counter>
CUDA_EXPORT inline cullen_and_dehnen::Result<Real> diagnosticsFromGradient(
    Real const (&a2_grad_r_v_phys)[3][3], Real a, Real a4_D_div_v_i_Dt, Real h_i, Real a_c_i,
    Real a_v_dec, Real α_max, Real ℓ, Counter& floatingPointOperations) {
    cullen_and_dehnen::Result<Real> result{};
    result.limiter = Real(0);
    if (!(a > Real(0)) || !(h_i > Real(0))) {
        return result;
    }
    result.inverseDecayTime = Real(2) * ℓ * a_v_dec / (a * h_i);
    floatingPointOperations += 4;
    Real const a2_div_v_i =
        a2_grad_r_v_phys[0][0] + a2_grad_r_v_phys[1][1] + a2_grad_r_v_phys[2][2];
    Real const curlX = a2_grad_r_v_phys[2][1] - a2_grad_r_v_phys[1][2];
    Real const curlY = a2_grad_r_v_phys[0][2] - a2_grad_r_v_phys[2][0];
    Real const curlZ = a2_grad_r_v_phys[1][0] - a2_grad_r_v_phys[0][1];
    Real const abs_a2_curl_v_i2 = curlX * curlX + curlY * curlY + curlZ * curlZ;
    result.divergence = a2_div_v_i / (a * a);
    Real const a4_S_i = a4_D_div_v_i_Dt < Real(0) ? -a4_D_div_v_i_Dt : Real(0);
    Real const h_i2_a4_S_i = h_i * h_i * a4_S_i;
    Real const targetDenominator = h_i2_a4_S_i + a_c_i * a_c_i;
    result.targetBulkViscosityCoefficient =
        targetDenominator > Real(0) ? α_max * h_i2_a4_S_i / targetDenominator : Real(0);
    Real const divergenceSquared = a2_div_v_i * a2_div_v_i;
    Real const soundSpeedOverRadius = a_c_i / h_i;
    Real const limiterDenominator = divergenceSquared + abs_a2_curl_v_i2 +
        Real(1.0e-4) * soundSpeedOverRadius * soundSpeedOverRadius;
    result.limiter =
        limiterDenominator > Real(0) ? divergenceSquared / limiterDenominator : Real(0);
    floatingPointOperations +=
        23 + (targetDenominator > Real(0) ? 2 : 0) + (limiterDenominator > Real(0) ? 1 : 0);
    return result;
}

template <typename Real, typename Counter>
CUDA_EXPORT inline cullen_and_dehnen::Result<Real> diagnostics(
    cullen_and_dehnen::GradientSums<Real> const& sums, Real a, Real a3_ä, Real h_i, Real a_c_i,
    Real a_v_dec, Real α_max, Real ℓ, Counter& floatingPointOperations) {
    Real a2_grad_r_v_phys[3][3];
    Real grad_x_F[3][3];
    cullen_and_dehnen::reconstructGradients(
        sums, a2_grad_r_v_phys, grad_x_F, floatingPointOperations);
    Real a4_D_div_v_i_Dt = Real(3) * a3_ä;
    for (int i = 0; i < 3; ++i) {
        a4_D_div_v_i_Dt += a * grad_x_F[i][i];
        for (int j = 0; j < 3; ++j) {
            a4_D_div_v_i_Dt -= a2_grad_r_v_phys[i][j] * a2_grad_r_v_phys[j][i];
        }
    }
    floatingPointOperations += 26;
    return diagnosticsFromGradient(a2_grad_r_v_phys, a, a4_D_div_v_i_Dt, h_i, a_c_i, a_v_dec, α_max,
        ℓ, floatingPointOperations);
}

// Appendix B3 time differences use peculiar gradients in production. The target
// and limiter formulas are unchanged; the physical-gradient acceleration route
// above is retained only as a reference estimator.
template <typename Real, typename Counter>
CUDA_EXPORT inline cullen_and_dehnen::Result<Real> diagnosticsFromHistory(
    cullen_and_dehnen::VelocityGradientSums<Real> const& sums, Real a, Float16Storage const& state,
    double sampleInterval, double previousSampleTimestep, Real h_i, Real a_c_i, Real a_v_dec,
    Real α_max, Real ℓ, Counter& floatingPointOperations) {
    Real a2_grad_r_v_phys[3][3];
    int const rank = cullen_and_dehnen::reconstructVelocityGradient(
        sums, a2_grad_r_v_phys, floatingPointOperations);
    Real const a2_div_v_i =
        a2_grad_r_v_phys[0][0] + a2_grad_r_v_phys[1][1] + a2_grad_r_v_phys[2][2];
    floatingPointOperations += 2;
    Real const a4_D_div_v_i_Dt = rank == 0 ?
        Real(0) :
        cullen_and_dehnen::divergenceTimeDerivative(
            a2_div_v_i, a, state, sampleInterval, previousSampleTimestep, floatingPointOperations);
    auto result = diagnosticsFromGradient(a2_grad_r_v_phys, a, a4_D_div_v_i_Dt, h_i, a_c_i, a_v_dec,
        α_max, ℓ, floatingPointOperations);
    result.divergence = a2_div_v_i;
    return result;
}

template <typename Real>
CUDA_EXPORT inline cullen_and_dehnen::Result<Real> diagnostics(
    cullen_and_dehnen::GradientSums<Real> const& sums, Real a, Real a3_ä, Real h_i, Real a_c_i,
    Real a_v_dec, Real α_max, Real ℓ) {
    unsigned long long operations = 0;
    return hu_etal::diagnostics(sums, a, a3_ä, h_i, a_c_i, a_v_dec, α_max, ℓ, operations);
}

}    // namespace hu_etal
