/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cmath>
#include <limits>

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/matrix_eigen.hpp>
#include <cosmictiger/sph/viscosity/scaled_divergence.hpp>

// L. Cullen and W. Dehnen, "Inviscid smoothed particle hydrodynamics",
// MNRAS 408 (2010), 669-683, doi:10.1111/j.1365-2966.2010.17158.x.
// https://arxiv.org/pdf/1006.1524
namespace cullen_and_dehnen {

float evolvedBulkViscosityCoefficient(float bulkViscosityCoefficient,
    float targetBulkViscosityCoefficient, float inverseDecayTime, float timeStep);

template <typename Real>
struct VelocityGradientSums {
    math::Matrix<Real, kSpatialDimensions> moment{};
    math::Matrix<Real, kSpatialDimensions> velocity{};
};

// Common corrected reconstruction for direct acceleration-gradient sensors.
template <typename Real>
struct GradientSums : VelocityGradientSums<Real> {
    math::Matrix<Real, kSpatialDimensions> acceleration{};
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
    CUDA_EXPORT inline int inverseMoment(math::Matrix<Real, kSpatialDimensions> const& moment,
        math::SymmetricMatrix<Real, kSpatialDimensions>& inverse,
        Counter& floatingPointOperations) {
        math::SymmetricMatrix<Real, kSpatialDimensions> T_norm;
        inverse = math::SymmetricMatrix<Real, kSpatialDimensions>{};
        Real scale = Real(0);
        for (int i = 0; i < kSpatialDimensions; ++i) {
            for (int j = 0; j < kSpatialDimensions; ++j) {
                Real const magnitude = std::abs(moment[i][j]);
                scale = magnitude > scale ? magnitude : scale;
            }
        }
        floatingPointOperations += kSpatialDimensions * kSpatialDimensions;
        if (!(scale > Real(0))) {
            return 0;
        }
        for (int i = 0; i < kSpatialDimensions; ++i) {
            for (int j = 0; j <= i; ++j) {
                T_norm(i, j) = (Real(0.5) * moment[i][j] + Real(0.5) * moment[j][i]) / scale;
            }
        }
        floatingPointOperations += 2 * kSpatialDimensions * (kSpatialDimensions + 1);
        auto const adj_T = math::adjugate(T_norm);
        Real det_T = T_norm(0, 0) * adj_T(0, 0);
        for (int dim = 1; dim < kSpatialDimensions; ++dim) {
            det_T += T_norm(0, dim) * adj_T(0, dim);
        }
        constexpr Real kRoundoff = std::numeric_limits<Real>::epsilon();
        constexpr Real kDeterminantTolerance = Real(128) * kRoundoff;
        constexpr Real kOffDiagonalTolerance = Real(4) * kRoundoff;
        constexpr Real kEigenvalueTolerance = Real(64) * kRoundoff;
        floatingPointOperations += kSpatialDimensions == 3 ? 24 : 2 * kSpatialDimensions;
        if (std::abs(det_T) > kDeterminantTolerance) {
            if constexpr (math::DefaultMatrixPolicy::kSymmetricInverseThreshold >=
                kSpatialDimensions) {
                for (int i = 0; i < kSpatialDimensions; ++i) {
                    for (int j = 0; j <= i; ++j) {
                        inverse(i, j) = (adj_T(i, j) / det_T) / scale;
                    }
                }
                floatingPointOperations += kSpatialDimensions * (kSpatialDimensions + 1);
                return kSpatialDimensions;
            } else {
                // Keep the physical rank gate above while honoring the build's
                // algorithm selection for full-rank neighborhoods.
                size_t operations = 0;
                auto const status = math::tryInversePivoted(
                    T_norm, inverse, math::MatrixOptions<Real>{Real(0), Real(0)}, &operations);
                floatingPointOperations += operations;
                if (status == math::MatrixStatus::Success) {
                    inverse /= scale;
                    floatingPointOperations += kSpatialDimensions * (kSpatialDimensions + 1) / 2;
                    return kSpatialDimensions;
                }
            }
        }

        math::Matrix<Real, kSpatialDimensions> q;
        math::SymmetricEigenOptions<Real> options;
        options.pivot = math::JacobiPivot::Cyclic;
        options.maxIterations = 8;
        options.absoluteOffDiagonalTolerance = kOffDiagonalTolerance;
        options.relativeOffDiagonalTolerance = Real(0);
        options.useHypot = false;
        math::jacobiDiagonalize(T_norm, q, options, floatingPointOperations);
        Real λ_max = Real(0);
        for (int k = 0; k < kSpatialDimensions; ++k) {
            Real const magnitude = std::abs(T_norm(k, k));
            λ_max = magnitude > λ_max ? magnitude : λ_max;
        }
        floatingPointOperations += kSpatialDimensions;
        int rank = 0;
        for (int k = 0; k < kSpatialDimensions; ++k) {
            Real const λ_k = T_norm(k, k);
            floatingPointOperations += 2;
            if (std::abs(λ_k) > kEigenvalueTolerance * λ_max) {
                ++rank;
                Real const inverseEigenvalue = (Real(1) / λ_k) / scale;
                for (int i = 0; i < kSpatialDimensions; ++i) {
                    for (int j = 0; j <= i; ++j) {
                        inverse(i, j) += q(i, k) * q(j, k) * inverseEigenvalue;
                    }
                }
                floatingPointOperations +=
                    2 + 3 * kSpatialDimensions * (kSpatialDimensions + 1) / 2;
            }
        }
        return rank;
    }

}    // namespace detail

/*
 * Cullen & Dehnen (2010), Appendix B2, equations (B2), (B4), (B7):
 *   T = Σ_j r_ij ⊗ r_ij w̄_ij, D = Σ_j v_ij ⊗ r_ij w̄_ij, V = D T⁻¹.
 * T, D → moment, velocity sums; V → velocity gradient.
 * This velocity-only reconstruction also supplies the shear/vorticity limiters.
 */
template <typename Real, typename Counter>
#if defined(__GNUC__) && !defined(__clang__) && !defined(__CUDACC__)
// GCC's fast-math vectorization of this 3x3 product can spill two floats,
// reload four, then evaluate unused stack lanes. Those lanes can overflow
// with floating-point traps enabled even though every matrix entry is finite.
__attribute__((optimize("no-tree-loop-vectorize", "no-tree-slp-vectorize")))
#endif
CUDA_EXPORT inline int reconstructVelocityGradient(
    VelocityGradientSums<Real> const& sums, Real (&V)[3][3], Counter& floatingPointOperations) {
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            V[row][column] = Real(0);
        }
    }
    math::SymmetricMatrix<Real, kSpatialDimensions> T_inv;
    int const rank = detail::inverseMoment(sums.moment, T_inv, floatingPointOperations);
    // Keep the product here: an inlined generic matrix operator can carry
    // vectorized instructions produced before this function's GCC safeguard.
    for (int row = 0; row < kSpatialDimensions; ++row) {
        for (int column = 0; column < kSpatialDimensions; ++column) {
            Real component = sums.velocity[row][0] * T_inv(0, column);
            for (int inner = 1; inner < kSpatialDimensions; ++inner) {
                component += sums.velocity[row][inner] * T_inv(inner, column);
            }
            V[row][column] = component;
        }
    }
    floatingPointOperations +=
        kSpatialDimensions * kSpatialDimensions * (2 * kSpatialDimensions - 1);
    return rank;
}

/*
 * Cullen & Dehnen (2010), Appendix B3's first alternative: difference successive
 * divergence estimates along the same particle trajectory. As in GADGET-4's
 * artificial_viscosity.cc, difference G = ∇_x·w before converting to a physical
 * rate: https://wwwmpa.mpa-garching.mpg.de/gadget4/doxygen/artificial__viscosity_8cc_source.html
 * Derived first-order conversion for conformal time dτ = dt/a:
 *   G_previous = q_previous/Δτ_previous,
 *   S = a⁻² DG/Dt ≈ (G_n − G_previous)/(a³ Δτ),
 *   a⁴ S ≈ a (G_n − G_previous)/Δτ.
 * G → sample divergence; Δτ → sample interval; a → scale factor;
 * q_previous → state; Δτ_previous → previous sample timestep.
 * All three cosmological sensors use the peculiar stored-velocity gradient.
 * Since G = a² θ_pec, this estimates Dθ_pec/Dt + 2Hθ_pec, not Dθ_pec/Dt alone.
 * Pure Hubble flow has G = 0 without subtracting a background after differencing.
 * A zero interval suppresses the startup derivative. Density retries never modify
 * the previous endpoint sample used here. Δτ is the completed conformal step;
 * the endpoint difference estimates its midpoint derivative and drives the
 * following kick with first-order viscosity timing.
 */
template <typename Real, typename Counter>
CUDA_EXPORT inline Real divergenceTimeDerivative(Real G, Real a, Float16Storage const& q_previous,
    double Δτ, double Δτ_previous, Counter& floatingPointOperations) {
    if (!sph_viscosity::detail::isFinite(Δτ) || !sph_viscosity::detail::isFinite(Δτ_previous) ||
        !(Δτ > 0.0) || !(Δτ_previous > 0.0)) {
        return Real(0);
    }
    double const derivative = (double(G) - sphDecodeDivergence(q_previous, Δτ_previous)) / Δτ;
    floatingPointOperations += 4;
    return Real(double(a) * derivative);
}

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
 *   T, D, C → moment, velocity, acceleration sums
 *   T⁻¹ → inverse moment matrix; D → velocity moment; C → acceleration moment
 *   V, G → velocity gradient matrix, acceleration gradient matrix
 *   V_ij, G_ij → velocity gradient, acceleration gradient entries.
 * The inverseMoment helper uses T⁺ = Σ_k q_k q_kᵀ/λ_k on resolved eigenmodes:
 *   normalized T → moment matrix; adj(T) → moment adjugate; det(T) → moment determinant
 *   q_k → eigenvector k; λ_k → eigenvalue; λ_max → maximum eigenvalue magnitude.
 *   Code names: T_inv = T⁻¹; T_norm = normalized T; adj_T = adj(T); det_T = det(T);
 *     q(:, k) = q_k.
 * This pseudoinverse extension is a code-derived treatment of deficient neighborhoods.
 */
// clang-format on
template <typename Real, typename Counter>
#if defined(__GNUC__) && !defined(__clang__) && !defined(__CUDACC__)
// As in reconstructVelocityGradient, avoid arithmetic on unused SIMD stack lanes.
__attribute__((optimize("no-tree-loop-vectorize", "no-tree-slp-vectorize")))
#endif
CUDA_EXPORT inline int reconstructGradients(GradientSums<Real> const& sums, Real (&V)[3][3],
    Real (&G)[3][3], Counter& floatingPointOperations) {
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            G[row][column] = Real(0);
        }
    }
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            V[row][column] = Real(0);
        }
    }
    math::SymmetricMatrix<Real, kSpatialDimensions> T_inv;
    int const rank = detail::inverseMoment(sums.moment, T_inv, floatingPointOperations);
    for (int row = 0; row < kSpatialDimensions; ++row) {
        for (int column = 0; column < kSpatialDimensions; ++column) {
            Real velocityComponent = sums.velocity[row][0] * T_inv(0, column);
            Real accelerationComponent = sums.acceleration[row][0] * T_inv(0, column);
            for (int inner = 1; inner < kSpatialDimensions; ++inner) {
                velocityComponent += sums.velocity[row][inner] * T_inv(inner, column);
                accelerationComponent += sums.acceleration[row][inner] * T_inv(inner, column);
            }
            V[row][column] = velocityComponent;
            G[row][column] = accelerationComponent;
        }
    }
    floatingPointOperations += 2 * kSpatialDimensions * kSpatialDimensions *
        (2 * kSpatialDimensions - 1);    // Two products: three multiplies/two adds per entry.
    return rank;
}

template <typename Real>
CUDA_EXPORT inline int reconstructGradients(
    GradientSums<Real> const& sums, Real (&V)[3][3], Real (&G)[3][3]) {
    unsigned long long floatingPointOperations = 0;
    return reconstructGradients(sums, V, G, floatingPointOperations);
}

// clang-format off
/*
 * Derived peculiar-gradient conversion of Cullen & Dehnen (2010), equation (B12),
 * https://arxiv.org/abs/1006.1524, preserving the history sensor's convention:
 *   w = a v_pec, dτ = dt/a, dx/dτ = w/a, F = dw/dτ, V = ∇_x w, C = ∇_x F,
 *   D tr(V)/Dτ = tr(C) − tr(V²)/a,
 *   a⁴ [Dθ_pec/Dt + 2Hθ_pec] = a tr(C) − tr(V²),  θ_pec = tr(V)/a².
 * F includes the full hydrodynamic, gravitational and imposed acceleration at
 * the same stage as w. No homogeneous expansion term is added to either gradient.
 * Symbol directory:
 *   a → scale factor; V → velocity gradient; C → acceleration gradient;
 *   a⁴ [Dθ_pec/Dt + 2Hθ_pec] → derivative (returned scaled shock sensor).
 *   Code names: a4_Dθ_pec_Dt_plus_2Hθ_pec = a⁴ [Dθ_pec/Dt + 2Hθ_pec].
 */
// clang-format on
template <typename Real, typename Counter>
CUDA_EXPORT inline Real peculiarDivergenceDerivative(
    Real const (&V)[3][3], Real const (&C)[3][3], Real a, Counter& floatingPointOperations) {
    Real a4_Dθ_pec_Dt_plus_2Hθ_pec = Real(0);
    for (int row = 0; row < kSpatialDimensions; ++row) {
        a4_Dθ_pec_Dt_plus_2Hθ_pec += a * C[row][row];
        for (int column = 0; column < kSpatialDimensions; ++column) {
            a4_Dθ_pec_Dt_plus_2Hθ_pec -= V[row][column] * V[column][row];
        }
    }
    floatingPointOperations += 24;
    return a4_Dθ_pec_Dt_plus_2Hθ_pec;
}

/*
 * Equations (13)-(18) and (B12), with comoving coordinates x, stored velocity
 * w = a v_pec, conformal time dτ = dt/a, and stored acceleration F = dw/dτ.
 * The acceleration-gradient reference estimator uses w_ij + a da/dt x_ij,
 * giving V' = a² ∇_r v_phys;
 * the acceleration sums give G = ∇_x F. Therefore
 *
 *   a⁴ D(div v_phys)/Dt = a tr(G) + 3 a³ d²a/dt² − tr(V'²).
 *
 * Production uses peculiar gradients V' = ∇_x w = a² ∇_r v_pec with either
 * peculiarDivergenceDerivative or the legacy GADGET-4 history derivative above.
 * The physical-gradient reference path supplies a³ ä.
 * The limiter is invariant under the
 * common a² scaling of V'. Equation (17)'s weighted sign average R_i is supplied
 * by the density loop using an immutable snapshot of stage divergences (direct
 * mode) or previous endpoint divergences (legacy mode).
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
 * Derived comoving conversion for the acceleration-gradient reference:
 *   V′ = a² ∇_r v_phys, G = ∇_x F,
 *   a⁴ D(∇·v_phys)/Dt = a tr(G) + 3a³ ä − tr(V′²).
 * Production supplies V′ = a² ∇_r v_pec with the derived peculiar conversion above,
 * or legacy endpoint differences; the target and limiter keep the same scaled form.
 * Symbol directory:
 *   a → scale factor; h_i → smoothing length
 *   v_sig,i → signal speed; ℓ → decay coefficient
 *   α_max → maximum bulk viscosity coefficient; α_loc,i → target bulk viscosity coefficient
 *   R_i → sign average; ξ_i → shear limiter
 *   V′, G → velocity gradient matrix, acceleration gradient matrix
 *   tr(V′) → velocity divergence; tr(G) → acceleration divergence
 *   tr(V′²) → gradient square trace; S′_ij → shear component
 *   tr(S′ S′ᵀ) → shear magnitude squared; a³ ä → expansion acceleration
 *   a⁴ D(∇·v_phys)/Dt → divergence derivative; a⁴ A_i → shock indicator
 *   h_i² a⁴ A_i → radius squared shock indicator; τ_conformal⁻¹ → inverse decay time.
 *   Code names: v_sig_i = v_sig,i; α_loc_i = α_loc,i; Vʹ = V′; tr_Vʹ = tr(V′); Sʹ_ij = S′_ij;
 *     tr_SʹSʹᵀ = tr(S′ S′ᵀ); a3_ä = a³ ä; a4_D_div_v_phys_Dt = a⁴ D(∇·v_phys)/Dt; a4_A_i = a⁴ A_i;
 *     h_i2_a4_A_i = h_i² a⁴ A_i.
 * The support and signal speed passed here are comoving; v_sig,stored = a v_sig,phys.
 * The common scaling cancels in ξ_i and the target ratio.
 */
// clang-format on
template <typename Real, typename Counter>
CUDA_EXPORT inline Result<Real> diagnosticsFromGradient(Real const (&Vʹ)[3][3], Real scaleFactor,
    Real a4_D_div_v_phys_Dt, Real signAverage, Real smoothingLength, Real signalSpeed, Real α_max,
    Real ℓ, Counter& floatingPointOperations) {
    Real const a = scaleFactor;
    Real const h_i = smoothingLength;
    Real const v_sig_i = signalSpeed;
    Result<Real> result{};
    if (!(a > Real(0)) || !(h_i > Real(0))) {
        return result;
    }
    result.inverseDecayTime = Real(2) * ℓ * v_sig_i / (a * h_i);
    floatingPointOperations += 4;
    Real const tr_Vʹ = Vʹ[0][0] + Vʹ[1][1] + Vʹ[2][2];
    Real tr_SʹSʹᵀ = Real(0);
    for (int i = 0; i < kSpatialDimensions; ++i) {
        for (int j = 0; j < kSpatialDimensions; ++j) {
            Real const Sʹ_ij = Real(0.5) * (Vʹ[i][j] + Vʹ[j][i]) -
                (i == j ? tr_Vʹ / Real(kSpatialDimensions) : Real(0));
            tr_SʹSʹᵀ += Sʹ_ij * Sʹ_ij;
        }
    }
    floatingPointOperations += 48;    // Velocity trace, symmetric shear and its squared norm.
    result.divergence = tr_Vʹ / (a * a);
    Real const R_i = signAverage;
    Real const squaredSignComplement = (Real(1) - R_i) * (Real(1) - R_i);
    Real const limiterCompression = Real(2) * squaredSignComplement * squaredSignComplement * tr_Vʹ;
    Real const limiterCompressionSquared = limiterCompression * limiterCompression;
    Real const limiterDenominator = limiterCompressionSquared + tr_SʹSʹᵀ;
    Real const ξ_i =
        limiterDenominator > Real(0) ? limiterCompressionSquared / limiterDenominator : Real(0);
    Real const a4_A_i = ξ_i * (a4_D_div_v_phys_Dt < Real(0) ? -a4_D_div_v_phys_Dt : Real(0));
    Real const h_i2_a4_A_i = h_i * h_i * a4_A_i;
    Real const denominator = v_sig_i * v_sig_i + h_i2_a4_A_i;
    result.targetBulkViscosityCoefficient =
        denominator > Real(0) ? α_max * h_i2_a4_A_i / denominator : Real(0);
    floatingPointOperations +=
        19 + (limiterDenominator > Real(0) ? 1 : 0) + (denominator > Real(0) ? 2 : 0);
    return result;
}

template <typename Real, typename Counter>
CUDA_EXPORT inline Result<Real> diagnostics(GradientSums<Real> const& sums, Real a, Real a3_ä,
    Real R_i, Real h_i, Real v_sig_i, Real α_max, Real ℓ, Counter& floatingPointOperations) {
    Real Vʹ[3][3];
    Real G[3][3];
    reconstructGradients(sums, Vʹ, G, floatingPointOperations);
    Real a4_D_div_v_phys_Dt = Real(3) * a3_ä;
    for (int i = 0; i < kSpatialDimensions; ++i) {
        a4_D_div_v_phys_Dt += a * G[i][i];
        for (int j = 0; j < kSpatialDimensions; ++j) {
            a4_D_div_v_phys_Dt -= Vʹ[i][j] * Vʹ[j][i];
        }
    }
    floatingPointOperations += 1 + 2 * kSpatialDimensions * (kSpatialDimensions + 1);
    return diagnosticsFromGradient(
        Vʹ, a, a4_D_div_v_phys_Dt, R_i, h_i, v_sig_i, α_max, ℓ, floatingPointOperations);
}

// Production time differences of G = ∇_x·w; no acceleration samples are read.
template <typename Real, typename Counter>
CUDA_EXPORT inline Result<Real> diagnosticsFromHistory(VelocityGradientSums<Real> const& sums,
    Real a, Float16Storage const& state, double sampleInterval, double previousSampleTimestep,
    Real R_i, Real h_i, Real v_sig_i, Real α_max, Real ℓ, Counter& floatingPointOperations) {
    Real Vʹ[3][3];
    int const rank = reconstructVelocityGradient(sums, Vʹ, floatingPointOperations);
    Real const divergence = Vʹ[0][0] + Vʹ[1][1] + Vʹ[2][2];
    floatingPointOperations += 2;
    Real const a4_D_div_v_phys_Dt = rank == 0 ?
        Real(0) :
        divergenceTimeDerivative(
            divergence, a, state, sampleInterval, previousSampleTimestep, floatingPointOperations);
    auto result = diagnosticsFromGradient(
        Vʹ, a, a4_D_div_v_phys_Dt, R_i, h_i, v_sig_i, α_max, ℓ, floatingPointOperations);
    result.divergence = divergence;
    return result;
}

template <typename Real>
CUDA_EXPORT inline Result<Real> diagnostics(GradientSums<Real> const& sums, Real a, Real a3_ä,
    Real R_i, Real h_i, Real v_sig_i, Real α_max, Real ℓ) {
    unsigned long long floatingPointOperations = 0;
    return diagnostics(sums, a, a3_ä, R_i, h_i, v_sig_i, α_max, ℓ, floatingPointOperations);
}

}    // namespace cullen_and_dehnen
