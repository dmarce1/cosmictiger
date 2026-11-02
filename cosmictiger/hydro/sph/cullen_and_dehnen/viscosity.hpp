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

float evolvedAlpha(float alpha, float targetAlpha, float inverseDecayTime, float timeStep);

template <typename Real>
struct GradientSums {
    Real moment[3][3]{};
    Real velocity[3][3]{};
    Real acceleration[3][3]{};
};

template <typename Real>
struct Result {
    Real divergence;
    Real targetAlpha;
    Real inverseDecayTime;
};

namespace detail {

    // NOLINTBEGIN(readability-identifier-naming)

    // Equation (B7) requires T^-1. A symmetric pseudoinverse also handles planar or
    // collinear neighborhoods: unresolved gradient components are zero, while affine
    // fields are recovered exactly on the resolved subspace. The normalized adjugate
    // handles well-conditioned 3D neighborhoods; Jacobi diagonalization handles the
    // remaining cases without assigning an artificial length scale to missing axes.
    template <typename Real, typename Counter>
    CUDA_EXPORT inline int inverseMoment(Real const (&moment)[3][3],
        math::SymmetricMatrix<Real, 3>& inverse, Counter& floatingPointOperations) {
        math::SymmetricMatrix<Real, 3> T;
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
                T(i, j) = (Real(0.5) * moment[i][j] + Real(0.5) * moment[j][i]) / scale;
            }
        }
        floatingPointOperations += 24;    // Six independent symmetric entries.
        auto const C = math::adjugate(T);
        Real const det_T = T(0, 0) * C(0, 0) + T(0, 1) * C(0, 1) + T(0, 2) * C(0, 2);
        constexpr Real kRoundoff = std::numeric_limits<Real>::epsilon();
        constexpr Real kDeterminantTolerance = Real(128) * kRoundoff;
        constexpr Real kOffDiagonalTolerance = Real(4) * kRoundoff;
        constexpr Real kEigenvalueTolerance = Real(64) * kRoundoff;
        floatingPointOperations += 24;
        if (std::abs(det_T) > kDeterminantTolerance) {
            if constexpr (math::DefaultMatrixPolicy::kSymmetricInverseThreshold >= 3) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j <= i; ++j) {
                        inverse(i, j) = (C(i, j) / det_T) / scale;
                    }
                }
                floatingPointOperations += 12;
                return 3;
            } else {
                // Keep the physical rank gate above while honoring the build's
                // algorithm selection for full-rank neighborhoods.
                size_t operations = 0;
                auto const status = math::tryInversePivoted(
                    T, inverse, math::MatrixOptions<Real>{Real(0), Real(0)}, &operations);
                floatingPointOperations += operations;
                if (status == math::MatrixStatus::Success) {
                    inverse /= scale;
                    floatingPointOperations += 6;
                    return 3;
                }
            }
        }

        math::Matrix<Real, 3> Q;
        math::SymmetricEigenOptions<Real> options;
        options.pivot = math::JacobiPivot::Cyclic;
        options.maxIterations = 8;
        options.absoluteOffDiagonalTolerance = kOffDiagonalTolerance;
        options.relativeOffDiagonalTolerance = Real(0);
        options.useHypot = false;
        math::jacobiDiagonalize(T, Q, options, floatingPointOperations);
        Real λ_max = Real(0);
        for (int k = 0; k < 3; ++k) {
            Real const magnitude = std::abs(T(k, k));
            λ_max = magnitude > λ_max ? magnitude : λ_max;
        }
        floatingPointOperations += 3;
        int rank = 0;
        for (int k = 0; k < 3; ++k) {
            Real const λ = T(k, k);
            floatingPointOperations += 2;
            if (std::abs(λ) > kEigenvalueTolerance * λ_max) {
                ++rank;
                Real const inverse_λ = (Real(1) / λ) / scale;
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j <= i; ++j) {
                        inverse(i, j) += Q(i, k) * Q(j, k) * inverse_λ;
                    }
                }
                floatingPointOperations += 20;
            }
        }
        return rank;
    }

    // NOLINTEND(readability-identifier-naming)

}    // namespace detail

// NOLINTBEGIN(readability-identifier-naming)

// Equations (B2), (B4), and (B7): D = Σ_j v_ij ⊗ r_ij w̄_ij,
// T = Σ_j r_ij ⊗ r_ij w̄_ij, V = D T^-1. Apply the same inverse to the
// acceleration sums for equation (B12). All three sums use identical weights.
template <typename Real, typename Counter>
CUDA_EXPORT inline int reconstructGradients(GradientSums<Real> const& sums,
    Real (&velocityGradient)[3][3], Real (&accelerationGradient)[3][3],
    Counter& floatingPointOperations) {
    math::SymmetricMatrix<Real, 3> T_inverse;
    int const rank = detail::inverseMoment(sums.moment, T_inverse, floatingPointOperations);
    math::Matrix<Real, 3> D;
    math::Matrix<Real, 3> C;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            D(i, j) = sums.velocity[i][j];
            C(i, j) = sums.acceleration[i][j];
        }
    }
    auto const V = D * T_inverse;
    auto const G = C * T_inverse;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            velocityGradient[i][j] = V(i, j);
            accelerationGradient[i][j] = G(i, j);
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

// Equations (13)-(18) and (B12), with comoving coordinates x, stored velocity
// w = a v_pec, conformal time dτ = dt/a, and stored acceleration F = dw/dτ.
// The velocity sums use w_ij + a da/dt x_ij, giving V' = a² ∇_r v_phys;
// the acceleration sums give G = ∇_x F. Therefore
//
//   a⁴ D(div v_phys)/Dt = a tr(G) + 3 a³ d²a/dt² − tr(V'²).
//
// expansionAcceleration supplies a³ d²a/dt². The limiter is invariant under the
// common a² scaling of V'. Equation (17)'s weighted sign average R_i is supplied
// by the density loop using an immutable snapshot of previous divergences.
// This paper's h is the full support radius H (its footnote 2), not M&M's H/2.
// The optional counter follows the repository's source-operation convention:
// arithmetic and math results cost one, FMA costs two, and selections/signs cost zero.
// Its type may be FlopCounter<> so disabled instrumentation optimizes away.
template <typename Real, typename Counter>
CUDA_EXPORT inline Result<Real> diagnostics(GradientSums<Real> const& sums, Real scaleFactor,
    Real expansionAcceleration, Real signAverage, Real supportRadius, Real signalSpeed,
    Real alphaMaximum, Real decayCoefficient, Counter& floatingPointOperations) {
    Real const a = scaleFactor;
    Real const H_i = supportRadius;
    Real const v_sig_i = signalSpeed;
    Result<Real> result{};
    if (!(a > Real(0)) || !(H_i > Real(0))) {
        return result;
    }
    result.inverseDecayTime = Real(2) * decayCoefficient * v_sig_i / (a * H_i);
    Real V[3][3];
    Real G[3][3];
    floatingPointOperations += 4;
    if (reconstructGradients(sums, V, G, floatingPointOperations) == 0) {
        return result;
    }
    Real const div_v_i = V[0][0] + V[1][1] + V[2][2];
    Real const div_F_i = G[0][0] + G[1][1] + G[2][2];
    Real tr_V2 = Real(0);
    Real tr_SS = Real(0);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            tr_V2 += V[i][j] * V[j][i];
            Real const S_ij =
                Real(0.5) * (V[i][j] + V[j][i]) - (i == j ? div_v_i / Real(3) : Real(0));
            tr_SS += S_ij * S_ij;
        }
    }
    floatingPointOperations += 70;    // Two traces and the full tr(V²), S, and tr(SS^T) loop.
    result.divergence = div_v_i / (a * a);
    Real const dot_div_v_i = a * div_F_i + Real(3) * expansionAcceleration - tr_V2;
    Real const R_i = signAverage;
    Real const one_minus_R2 = (Real(1) - R_i) * (Real(1) - R_i);
    Real const ξ_numerator = Real(2) * one_minus_R2 * one_minus_R2 * div_v_i;
    Real const ξ_numerator2 = ξ_numerator * ξ_numerator;
    Real const ξ_denominator = ξ_numerator2 + tr_SS;
    Real const ξ_i = ξ_denominator > Real(0) ? ξ_numerator2 / ξ_denominator : Real(0);
    Real const A_i = ξ_i * (dot_div_v_i < Real(0) ? -dot_div_v_i : Real(0));
    Real const H2_A_i = H_i * H_i * A_i;
    Real const denominator = v_sig_i * v_sig_i + H2_A_i;
    result.targetAlpha = denominator > Real(0) ? alphaMaximum * H2_A_i / denominator : Real(0);
    floatingPointOperations +=
        19 + (ξ_denominator > Real(0) ? 1 : 0) + (denominator > Real(0) ? 2 : 0);
    return result;
}

template <typename Real>
CUDA_EXPORT inline Result<Real> diagnostics(GradientSums<Real> const& sums, Real scaleFactor,
    Real expansionAcceleration, Real signAverage, Real supportRadius, Real signalSpeed,
    Real alphaMaximum, Real decayCoefficient) {
    unsigned long long floatingPointOperations = 0;
    return diagnostics(sums, scaleFactor, expansionAcceleration, signAverage, supportRadius,
        signalSpeed, alphaMaximum, decayCoefficient, floatingPointOperations);
}

// NOLINTEND(readability-identifier-naming)

}    // namespace cullen_and_dehnen
