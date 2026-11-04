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
 *   α_iⁿ → bulkViscosityCoefficient; α_tar,i → targetBulkViscosityCoefficient
 *   ξ_i → limiter; Δt → timeStep; τ_i⁻¹ → inverseDecayTime
 *   bracket before ξ_i → updatedBulkViscosityCoefficient; α_iⁿ⁺¹ → return value.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real evolvedBulkViscosityCoefficient(Real bulkViscosityCoefficient,
    Real targetBulkViscosityCoefficient, Real inverseDecayTime, Real limiter, Real timeStep) {
    Real const updatedBulkViscosityCoefficient =
        bulkViscosityCoefficient <= targetBulkViscosityCoefficient ?
        targetBulkViscosityCoefficient :
        targetBulkViscosityCoefficient +
            (bulkViscosityCoefficient - targetBulkViscosityCoefficient) *
                std::exp(-timeStep * inverseDecayTime);
    return limiter * updatedBulkViscosityCoefficient;
}

// The acceleration-gradient reference uses CD's corrected reconstruction:
// V = a^2 grad_r(v_phys), G = grad_x(F), and expansionAcceleration = a^3 d^2a/dt^2.
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
 *   h_i → supportRadius; a c_i → soundSpeed; a v_dec → signalSpeed
 *   α_max → maximumBulkViscosityCoefficient; α_tar,i → result.targetBulkViscosityCoefficient
 *   a → scaleFactor; a³ ä → expansionAcceleration; ℓ → decayCoefficient
 *   a² ∇_r v_phys → velocityGradient; ∇_x F → accelerationGradient
 *   a² ∇·v_i → divergence; a⁴ D(∇·v_i)/Dt → divergenceDerivative
 *   a⁴ S_i → shockIndicator; h_i² a⁴ S_i → radiusSquaredShock
 *   |a² ∇×v_i|² → curlSquared; ξ_i → result.limiter
 *   τ_conformal⁻¹ → result.inverseDecayTime.
 * The neighbor maximum (18) is accumulated in the CPU/CUDA density pass.
 */
// clang-format on
template <typename Real, typename Counter>
CUDA_EXPORT inline cullen_and_dehnen::Result<Real> diagnosticsFromGradient(
    Real const (&velocityGradient)[3][3], Real scaleFactor, Real divergenceDerivative,
    Real supportRadius, Real soundSpeed, Real signalSpeed, Real maximumBulkViscosityCoefficient,
    Real decayCoefficient, Counter& floatingPointOperations) {
    cullen_and_dehnen::Result<Real> result{};
    result.limiter = Real(0);
    if (!(scaleFactor > Real(0)) || !(supportRadius > Real(0))) {
        return result;
    }
    result.inverseDecayTime =
        Real(2) * decayCoefficient * signalSpeed / (scaleFactor * supportRadius);
    floatingPointOperations += 4;
    Real const divergence =
        velocityGradient[0][0] + velocityGradient[1][1] + velocityGradient[2][2];
    Real const curlX = velocityGradient[2][1] - velocityGradient[1][2];
    Real const curlY = velocityGradient[0][2] - velocityGradient[2][0];
    Real const curlZ = velocityGradient[1][0] - velocityGradient[0][1];
    Real const curlSquared = curlX * curlX + curlY * curlY + curlZ * curlZ;
    result.divergence = divergence / (scaleFactor * scaleFactor);
    Real const shockIndicator = divergenceDerivative < Real(0) ? -divergenceDerivative : Real(0);
    Real const radiusSquaredShock = supportRadius * supportRadius * shockIndicator;
    Real const targetDenominator = radiusSquaredShock + soundSpeed * soundSpeed;
    result.targetBulkViscosityCoefficient = targetDenominator > Real(0) ?
        maximumBulkViscosityCoefficient * radiusSquaredShock / targetDenominator :
        Real(0);
    Real const divergenceSquared = divergence * divergence;
    Real const soundSpeedOverRadius = soundSpeed / supportRadius;
    Real const limiterDenominator = divergenceSquared + curlSquared +
        Real(1.0e-4) * soundSpeedOverRadius * soundSpeedOverRadius;
    result.limiter =
        limiterDenominator > Real(0) ? divergenceSquared / limiterDenominator : Real(0);
    floatingPointOperations +=
        23 + (targetDenominator > Real(0) ? 2 : 0) + (limiterDenominator > Real(0) ? 1 : 0);
    return result;
}

template <typename Real, typename Counter>
CUDA_EXPORT inline cullen_and_dehnen::Result<Real> diagnostics(
    cullen_and_dehnen::GradientSums<Real> const& sums, Real scaleFactor, Real expansionAcceleration,
    Real supportRadius, Real soundSpeed, Real signalSpeed, Real maximumBulkViscosityCoefficient,
    Real decayCoefficient, Counter& floatingPointOperations) {
    Real velocityGradient[3][3];
    Real accelerationGradient[3][3];
    cullen_and_dehnen::reconstructGradients(
        sums, velocityGradient, accelerationGradient, floatingPointOperations);
    Real derivative = Real(3) * expansionAcceleration;
    for (int i = 0; i < 3; ++i) {
        derivative += scaleFactor * accelerationGradient[i][i];
        for (int j = 0; j < 3; ++j) {
            derivative -= velocityGradient[i][j] * velocityGradient[j][i];
        }
    }
    floatingPointOperations += 26;
    return diagnosticsFromGradient(velocityGradient, scaleFactor, derivative, supportRadius,
        soundSpeed, signalSpeed, maximumBulkViscosityCoefficient, decayCoefficient,
        floatingPointOperations);
}

// Appendix B3 time differences use peculiar gradients in production. The target
// and limiter formulas are unchanged; the physical-gradient acceleration route
// above is retained only as a reference estimator.
template <typename Real, typename Counter>
CUDA_EXPORT inline cullen_and_dehnen::Result<Real> diagnosticsFromHistory(
    cullen_and_dehnen::VelocityGradientSums<Real> const& sums, Real scaleFactor,
    Float16Storage const& state, double sampleInterval, double previousSampleTimestep,
    Real supportRadius, Real soundSpeed, Real signalSpeed, Real maximumBulkViscosityCoefficient,
    Real decayCoefficient, Counter& floatingPointOperations) {
    Real velocityGradient[3][3];
    int const rank = cullen_and_dehnen::reconstructVelocityGradient(
        sums, velocityGradient, floatingPointOperations);
    Real const divergence =
        velocityGradient[0][0] + velocityGradient[1][1] + velocityGradient[2][2];
    floatingPointOperations += 2;
    Real const derivative = rank == 0 ?
        Real(0) :
        cullen_and_dehnen::divergenceTimeDerivative(divergence, scaleFactor, state, sampleInterval,
            previousSampleTimestep, floatingPointOperations);
    auto result = diagnosticsFromGradient(velocityGradient, scaleFactor, derivative, supportRadius,
        soundSpeed, signalSpeed, maximumBulkViscosityCoefficient, decayCoefficient,
        floatingPointOperations);
    result.divergence = divergence;
    return result;
}

template <typename Real>
CUDA_EXPORT inline cullen_and_dehnen::Result<Real> diagnostics(
    cullen_and_dehnen::GradientSums<Real> const& sums, Real scaleFactor, Real expansionAcceleration,
    Real supportRadius, Real soundSpeed, Real signalSpeed, Real maximumBulkViscosityCoefficient,
    Real decayCoefficient) {
    unsigned long long operations = 0;
    return hu_etal::diagnostics(sums, scaleFactor, expansionAcceleration, supportRadius, soundSpeed,
        signalSpeed, maximumBulkViscosityCoefficient, decayCoefficient, operations);
}

}    // namespace hu_etal
