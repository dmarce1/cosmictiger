/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cmath>
#include <limits>

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/matrix_eigen.hpp>

// L. Cullen and W. Dehnen, "Inviscid smoothed particle hydrodynamics",
// MNRAS 408 (2010), 669-683, doi:10.1111/j.1365-2966.2010.17158.x.
// https://arxiv.org/pdf/1006.1524
namespace cullen_and_dehnen {

float evolvedBulkViscosityCoefficient(float bulkViscosityCoefficient,
    float targetBulkViscosityCoefficient, float inverseDecayTime, float timeStep);

template <typename Real>
struct GradientSums {
    Real moment[3][3]{};
    Real velocity[3][3]{};
    Real acceleration[3][3]{};
};

template <typename Real>
struct Result {
    Real divergence;
    Real targetBulkViscosityCoefficient;
    Real inverseDecayTime;
    Real limiter = Real(1);    // Hu et al. applies this after the alpha update.
};

namespace detail {

    /*
     * Equation (B7) requires T^-1. A symmetric pseudoinverse also handles planar or
     * collinear neighborhoods: unresolved gradient components are zero, while affine
     * fields are recovered exactly on the resolved subspace. The normalized adjugate
     * handles well-conditioned 3D neighborhoods; Jacobi diagonalization handles the
     * remaining cases without assigning an artificial length scale to missing axes.
     */
    template <typename Real, typename Counter>
    CUDA_EXPORT inline int inverseMoment(Real const (&moment)[3][3],
        math::SymmetricMatrix<Real, 3>& inverse, Counter& floatingPointOperations) {
        math::SymmetricMatrix<Real, 3> momentMatrix;
        inverse = math::SymmetricMatrix<Real, 3>{};
        Real scale = Real(0);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                Real const magnitude = std::abs(moment[i][j]);
                scale = magnitude > scale ? magnitude : scale;
            }
        }
        floatingPointOperations += 9;
        if (!(scale > Real(0))) {
            return 0;
        }
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j <= i; ++j) {
                momentMatrix(i, j) = (Real(0.5) * moment[i][j] + Real(0.5) * moment[j][i]) / scale;
            }
        }
        floatingPointOperations += 24;    // Six independent symmetric entries.
        auto const momentAdjugate = math::adjugate(momentMatrix);
        Real const momentDeterminant = momentMatrix(0, 0) * momentAdjugate(0, 0) +
            momentMatrix(0, 1) * momentAdjugate(0, 1) + momentMatrix(0, 2) * momentAdjugate(0, 2);
        constexpr Real kRoundoff = std::numeric_limits<Real>::epsilon();
        constexpr Real kDeterminantTolerance = Real(128) * kRoundoff;
        constexpr Real kOffDiagonalTolerance = Real(4) * kRoundoff;
        constexpr Real kEigenvalueTolerance = Real(64) * kRoundoff;
        floatingPointOperations += 24;
        if (std::abs(momentDeterminant) > kDeterminantTolerance) {
            if constexpr (math::DefaultMatrixPolicy::kSymmetricInverseThreshold >= 3) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j <= i; ++j) {
                        inverse(i, j) = (momentAdjugate(i, j) / momentDeterminant) / scale;
                    }
                }
                floatingPointOperations += 12;
                return 3;
            } else {
                // Keep the physical rank gate above while honoring the build's
                // algorithm selection for full-rank neighborhoods.
                size_t operations = 0;
                auto const status = math::tryInversePivoted(momentMatrix, inverse,
                    math::MatrixOptions<Real>{Real(0), Real(0)}, &operations);
                floatingPointOperations += operations;
                if (status == math::MatrixStatus::Success) {
                    inverse /= scale;
                    floatingPointOperations += 6;
                    return 3;
                }
            }
        }

        math::Matrix<Real, 3> eigenvectors;
        math::SymmetricEigenOptions<Real> options;
        options.pivot = math::JacobiPivot::Cyclic;
        options.maxIterations = 8;
        options.absoluteOffDiagonalTolerance = kOffDiagonalTolerance;
        options.relativeOffDiagonalTolerance = Real(0);
        options.useHypot = false;
        math::jacobiDiagonalize(momentMatrix, eigenvectors, options, floatingPointOperations);
        Real maximumEigenvalueMagnitude = Real(0);
        for (int k = 0; k < 3; ++k) {
            Real const magnitude = std::abs(momentMatrix(k, k));
            maximumEigenvalueMagnitude =
                magnitude > maximumEigenvalueMagnitude ? magnitude : maximumEigenvalueMagnitude;
        }
        floatingPointOperations += 3;
        int rank = 0;
        for (int k = 0; k < 3; ++k) {
            Real const eigenvalue = momentMatrix(k, k);
            floatingPointOperations += 2;
            if (std::abs(eigenvalue) > kEigenvalueTolerance * maximumEigenvalueMagnitude) {
                ++rank;
                Real const inverseEigenvalue = (Real(1) / eigenvalue) / scale;
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j <= i; ++j) {
                        inverse(i, j) +=
                            eigenvectors(i, k) * eigenvectors(j, k) * inverseEigenvalue;
                    }
                }
                floatingPointOperations += 20;
            }
        }
        return rank;
    }

}    // namespace detail

/*
 * Equations (B2), (B4), and (B7): D = Σ_j v_ij ⊗ r_ij w̄_ij,
 * T = Σ_j r_ij ⊗ r_ij w̄_ij, V = D T^-1. Apply the same inverse to the
 * acceleration sums for equation (B12). All three sums use identical weights.
 */
// clang-format off
/*
 * Cullen & Dehnen (2010), MNRAS 408, 669, doi:10.1111/j.1365-2966.2010.17158.x,
 * https://arxiv.org/abs/1006.1524, equations (B2), (B4), (B7), (B12):
 *   D = Σ_j v_ij ⊗ r_ij w̄_ij,  T = Σ_j r_ij ⊗ r_ij w̄_ij,  V = D T⁻¹.
 * Derived acceleration analogue: C = Σ_j a_ij ⊗ r_ij w̄_ij, G = C T⁻¹.
 * Symbol directory:
 *   T, D, C → sums.moment, sums.velocity, sums.acceleration
 *   T⁻¹ → inverseMomentMatrix; D → velocityMoment; C → accelerationMoment
 *   V, G → velocityGradientMatrix, accelerationGradientMatrix
 *   V_ij, G_ij → velocityGradient[i][j], accelerationGradient[i][j].
 * The inverseMoment helper uses T⁺ = Σ_k q_k q_kᵀ/λ_k on resolved eigenmodes:
 *   normalized T → momentMatrix; adj(T) → momentAdjugate; det(T) → momentDeterminant
 *   q_k → eigenvectors(:, k); λ_k → eigenvalue; λ_max → maximumEigenvalueMagnitude.
 * This pseudoinverse extension is a code-derived treatment of deficient neighborhoods.
 */
// clang-format on
template <typename Real, typename Counter>
CUDA_EXPORT inline int reconstructGradients(GradientSums<Real> const& sums,
    Real (&velocityGradient)[3][3], Real (&accelerationGradient)[3][3],
    Counter& floatingPointOperations) {
    math::SymmetricMatrix<Real, 3> inverseMomentMatrix;
    int const rank =
        detail::inverseMoment(sums.moment, inverseMomentMatrix, floatingPointOperations);
    math::Matrix<Real, 3> velocityMoment;
    math::Matrix<Real, 3> accelerationMoment;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            velocityMoment(i, j) = sums.velocity[i][j];
            accelerationMoment(i, j) = sums.acceleration[i][j];
        }
    }
    auto const velocityGradientMatrix = velocityMoment * inverseMomentMatrix;
    auto const accelerationGradientMatrix = accelerationMoment * inverseMomentMatrix;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            velocityGradient[i][j] = velocityGradientMatrix(i, j);
            accelerationGradient[i][j] = accelerationGradientMatrix(i, j);
        }
    }
    floatingPointOperations += 90;    // Two products: three multiplies/two adds per entry.
    return rank;
}

template <typename Real>
CUDA_EXPORT inline int reconstructGradients(GradientSums<Real> const& sums,
    Real (&velocityGradient)[3][3], Real (&accelerationGradient)[3][3]) {
    unsigned long long floatingPointOperations = 0;
    return reconstructGradients(
        sums, velocityGradient, accelerationGradient, floatingPointOperations);
}

/*
 * Equations (13)-(18) and (B12), with comoving coordinates x, stored velocity
 * w = a v_pec, conformal time dτ = dt/a, and stored acceleration F = dw/dτ.
 * The velocity sums use w_ij + a da/dt x_ij, giving V' = a² ∇_r v_phys;
 * the acceleration sums give G = ∇_x F. Therefore
 *
 *   a⁴ D(div v_phys)/Dt = a tr(G) + 3 a³ d²a/dt² − tr(V'²).
 *
 * expansionAcceleration supplies a³ d²a/dt². The limiter is invariant under the
 * common a² scaling of V'. Equation (17)'s weighted sign average R_i is supplied
 * by the density loop using an immutable snapshot of previous divergences.
 * This paper's h is the full support radius H (its footnote 2), not M&M's H/2.
 * The optional counter follows the repository's source-operation convention:
 * arithmetic and math results cost one, FMA costs two, and selections/signs cost zero.
 * Its type may be FlopCounter<> so disabled instrumentation optimizes away.
 */
// clang-format off
/*
 * Cullen & Dehnen (2010), equations (13)–(18), (B12); citation above:
 *   A_i = ξ_i max(−D(∇·v_i)/Dt, 0),                                  (13)
 *   α_loc,i = α_max h_i² A_i/(v_sig,i² + h_i² A_i),                     (14)
 *   τ_i = h_i/(2ℓ v_sig,i),                                          (16)
 *   R_i = ρ_i⁻¹ Σ_j m_j W_ij sign(∇·v_j),                            (17)
 *   ξ_i = |2(1 − R_i)⁴ ∇·v_i|²/(|2(1 − R_i)⁴ ∇·v_i|² + tr(S_i S_iᵀ)). (18)
 * Derived comoving conversion, also used by the code below:
 *   V′ = a² ∇_r v_phys, G = ∇_x F,
 *   a⁴ D(∇·v_phys)/Dt = a tr(G) + 3a³ ä − tr(V′²).
 * Symbol directory:
 *   a → scaleFactor/localScaleFactor; h_i → supportRadius/kernelSupportRadius
 *   v_sig,i → signalSpeed/localSignalSpeed; ℓ → decayCoefficient
 *   α_max → maximumBulkViscosityCoefficient; α_loc,i → result.targetBulkViscosityCoefficient
 *   R_i → signAverage/divergenceSignAverage; ξ_i → shearLimiter
 *   V′, G → velocityGradientMatrix, accelerationGradientMatrix
 *   tr(V′) → velocityDivergence; tr(G) → accelerationDivergence
 *   tr(V′²) → gradientSquareTrace; S′_ij → shearComponent
 *   tr(S′ S′ᵀ) → shearMagnitudeSquared; a³ ä → expansionAcceleration
 *   a⁴ D(∇·v_phys)/Dt → divergenceDerivative; a⁴ A_i → shockIndicator
 *   h_i² a⁴ A_i → radiusSquaredShockIndicator; τ_conformal⁻¹ → result.inverseDecayTime.
 * The support and signal speed passed here are comoving; v_sig,stored = a v_sig,phys.
 * The common scaling cancels in ξ_i and the target ratio.
 */
// clang-format on
template <typename Real, typename Counter>
CUDA_EXPORT inline Result<Real> diagnostics(GradientSums<Real> const& sums, Real scaleFactor,
    Real expansionAcceleration, Real signAverage, Real supportRadius, Real signalSpeed,
    Real maximumBulkViscosityCoefficient, Real decayCoefficient, Counter& floatingPointOperations) {
    Real const localScaleFactor = scaleFactor;
    Real const kernelSupportRadius = supportRadius;
    Real const localSignalSpeed = signalSpeed;
    Result<Real> result{};
    if (!(localScaleFactor > Real(0)) || !(kernelSupportRadius > Real(0))) {
        return result;
    }
    result.inverseDecayTime =
        Real(2) * decayCoefficient * localSignalSpeed / (localScaleFactor * kernelSupportRadius);
    Real velocityGradientMatrix[3][3];
    Real accelerationGradientMatrix[3][3];
    floatingPointOperations += 4;
    if (reconstructGradients(sums, velocityGradientMatrix, accelerationGradientMatrix,
            floatingPointOperations) == 0) {
        return result;
    }
    Real const velocityDivergence =
        velocityGradientMatrix[0][0] + velocityGradientMatrix[1][1] + velocityGradientMatrix[2][2];
    Real const accelerationDivergence = accelerationGradientMatrix[0][0] +
        accelerationGradientMatrix[1][1] + accelerationGradientMatrix[2][2];
    Real gradientSquareTrace = Real(0);
    Real shearMagnitudeSquared = Real(0);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            gradientSquareTrace += velocityGradientMatrix[i][j] * velocityGradientMatrix[j][i];
            Real const shearComponent =
                Real(0.5) * (velocityGradientMatrix[i][j] + velocityGradientMatrix[j][i]) -
                (i == j ? velocityDivergence / Real(3) : Real(0));
            shearMagnitudeSquared += shearComponent * shearComponent;
        }
    }
    floatingPointOperations += 70;    // Two traces and the full tr(V²), S, and tr(SS^T) loop.
    result.divergence = velocityDivergence / (localScaleFactor * localScaleFactor);
    Real const divergenceDerivative = localScaleFactor * accelerationDivergence +
        Real(3) * expansionAcceleration - gradientSquareTrace;
    Real const divergenceSignAverage = signAverage;
    Real const squaredSignComplement =
        (Real(1) - divergenceSignAverage) * (Real(1) - divergenceSignAverage);
    Real const limiterCompression =
        Real(2) * squaredSignComplement * squaredSignComplement * velocityDivergence;
    Real const limiterCompressionSquared = limiterCompression * limiterCompression;
    Real const limiterDenominator = limiterCompressionSquared + shearMagnitudeSquared;
    Real const shearLimiter =
        limiterDenominator > Real(0) ? limiterCompressionSquared / limiterDenominator : Real(0);
    Real const shockIndicator =
        shearLimiter * (divergenceDerivative < Real(0) ? -divergenceDerivative : Real(0));
    Real const radiusSquaredShockIndicator =
        kernelSupportRadius * kernelSupportRadius * shockIndicator;
    Real const denominator = localSignalSpeed * localSignalSpeed + radiusSquaredShockIndicator;
    result.targetBulkViscosityCoefficient = denominator > Real(0) ?
        maximumBulkViscosityCoefficient * radiusSquaredShockIndicator / denominator :
        Real(0);
    floatingPointOperations +=
        19 + (limiterDenominator > Real(0) ? 1 : 0) + (denominator > Real(0) ? 2 : 0);
    return result;
}

template <typename Real>
CUDA_EXPORT inline Result<Real> diagnostics(GradientSums<Real> const& sums, Real scaleFactor,
    Real expansionAcceleration, Real signAverage, Real supportRadius, Real signalSpeed,
    Real maximumBulkViscosityCoefficient, Real decayCoefficient) {
    unsigned long long floatingPointOperations = 0;
    return diagnostics(sums, scaleFactor, expansionAcceleration, signAverage, supportRadius,
        signalSpeed, maximumBulkViscosityCoefficient, decayCoefficient, floatingPointOperations);
}

}    // namespace cullen_and_dehnen
