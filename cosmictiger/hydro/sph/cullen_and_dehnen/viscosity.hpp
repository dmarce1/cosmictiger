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
    CUDA_EXPORT inline int inverseMoment(
        Real const (&moment)[3][3], Real (&inverse)[3][3], Counter& floatingPointOperations) {
        Real T[3][3];
        Real scale = Real(0);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                Real const magnitude = std::abs(moment[i][j]);
                scale = magnitude > scale ? magnitude : scale;
                inverse[i][j] = Real(0);
            }
        }
        floatingPointOperations += 9;    // Absolute values in the matrix scale.
        if (!(scale > Real(0))) {
            return 0;
        }
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                T[i][j] = (Real(0.5) * moment[i][j] + Real(0.5) * moment[j][i]) / scale;
            }
        }
        floatingPointOperations += 36;    // Symmetrization and normalization.
        Real C[3][3];
        C[0][0] = T[1][1] * T[2][2] - T[1][2] * T[2][1];
        C[0][1] = T[0][2] * T[2][1] - T[0][1] * T[2][2];
        C[0][2] = T[0][1] * T[1][2] - T[0][2] * T[1][1];
        C[1][0] = C[0][1];
        C[1][1] = T[0][0] * T[2][2] - T[0][2] * T[2][0];
        C[1][2] = T[0][2] * T[1][0] - T[0][0] * T[1][2];
        C[2][0] = C[0][2];
        C[2][1] = C[1][2];
        C[2][2] = T[0][0] * T[1][1] - T[0][1] * T[1][0];
        Real const det_T = T[0][0] * C[0][0] + T[0][1] * C[0][1] + T[0][2] * C[0][2];
        constexpr Real kRoundoff = std::numeric_limits<Real>::epsilon();
        constexpr Real kDeterminantTolerance = Real(128) * kRoundoff;
        constexpr Real kOffDiagonalTolerance = Real(4) * kRoundoff;
        constexpr Real kEigenvalueTolerance = Real(64) * kRoundoff;
        floatingPointOperations += 24;    // Six cofactors, determinant, and its absolute value.
        if (std::abs(det_T) > kDeterminantTolerance) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    inverse[i][j] = (C[i][j] / det_T) / scale;
                }
            }
            floatingPointOperations += 18;    // Two divisions per inverse entry.
            return 3;
        }

        Real Q[3][3]{};
        for (int i = 0; i < 3; ++i) {
            Q[i][i] = Real(1);
        }
        for (int sweep = 0; sweep < 8; ++sweep) {
            for (int p = 0; p < 2; ++p) {
                for (int q = p + 1; q < 3; ++q) {
                    Real const T_pq = T[p][q];
                    floatingPointOperations += 1;
                    if (std::abs(T_pq) <= kOffDiagonalTolerance) {
                        continue;
                    }
                    Real const θ = (T[q][q] - T[p][p]) / (Real(2) * T_pq);
                    Real const t = (θ < Real(0) ? Real(-1) : Real(1)) /
                        (std::abs(θ) + std::sqrt(Real(1) + θ * θ));
                    Real const c = Real(1) / std::sqrt(Real(1) + t * t);
                    Real const s = t * c;
                    T[p][p] -= t * T_pq;
                    T[q][q] += t * T_pq;
                    T[p][q] = T[q][p] = Real(0);
                    for (int k = 0; k < 3; ++k) {
                        if (k != p && k != q) {
                            Real const T_kp = T[k][p];
                            Real const T_kq = T[k][q];
                            T[k][p] = T[p][k] = c * T_kp - s * T_kq;
                            T[k][q] = T[q][k] = s * T_kp + c * T_kq;
                        }
                        Real const Q_kp = Q[k][p];
                        Real const Q_kq = Q[k][q];
                        Q[k][p] = c * Q_kp - s * Q_kq;
                        Q[k][q] = s * Q_kp + c * Q_kq;
                    }
                    // Rotation parameters (14), two diagonal updates (4),
                    // one symmetric off-diagonal pair (6), and Q's three pairs (18).
                    floatingPointOperations += 42;
                }
            }
        }
        Real λ_max = Real(0);
        for (int k = 0; k < 3; ++k) {
            Real const magnitude = std::abs(T[k][k]);
            λ_max = magnitude > λ_max ? magnitude : λ_max;
        }
        floatingPointOperations += 3;    // Eigenvalue absolute values.
        int rank = 0;
        for (int k = 0; k < 3; ++k) {
            Real const λ = T[k][k];
            floatingPointOperations += 2;
            if (std::abs(λ) > kEigenvalueTolerance * λ_max) {
                ++rank;
                Real const inverse_λ = (Real(1) / λ) / scale;
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        inverse[i][j] += Q[i][k] * Q[j][k] * inverse_λ;
                    }
                }
                floatingPointOperations +=
                    29;    // Reciprocal scaling and nine outer-product updates.
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
    Real T_inverse[3][3];
    int const rank = detail::inverseMoment(sums.moment, T_inverse, floatingPointOperations);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Real V_ij = Real(0);
            Real G_ij = Real(0);
            for (int k = 0; k < 3; ++k) {
                V_ij += sums.velocity[i][k] * T_inverse[k][j];
                G_ij += sums.acceleration[i][k] * T_inverse[k][j];
            }
            velocityGradient[i][j] = V_ij;
            accelerationGradient[i][j] = G_ij;
        }
    }
    floatingPointOperations += 108;    // Two 3x3 products, with three multiply-adds per entry.
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
