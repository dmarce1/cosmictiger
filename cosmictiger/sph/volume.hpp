/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

/*
 * Support-radius, weighted-volume and grad-H correction algebra for SPH, after
 *
 *   P. F. Hopkins, "A general class of Lagrangian smoothed particle hydrodynamics
 *   methods and implications for fluid mixing problems", MNRAS 428 (2013), 2840,
 *   arXiv:1206.5006, equations 8-11 (y_i and its derivatives) and equation 13
 *   (f_ij, with the constraint scalar x~ = 1), and
 *   D. J. Price, "Smoothed particle hydrodynamics and magnetohydrodynamics",
 *   J. Comput. Phys. 231 (2012), 759, arXiv:1012.1885, equation 27 (Omega_a).
 *
 * The length here is the stored compact support radius H, with the selected kernel
 * W(r, H) = H^-3 w(q) and q = r/H. It is not the papers' smoothing length h, whose
 * normalization differs by kernel; only ratios H dW/dH enter, so the formulas are
 * unchanged. Particle i carries a volume weight x_i and a mass m_i. With
 *
 *   n_i = sum_j W_ij(H_i),   y_i = sum_j x_j W_ij(H_i),
 *   V_i = x_i / y_i,         rho*_i = m_i y_i / x_i,
 *
 * the resolution constraint H_i^3 n_i = constant uses the unweighted count n_i. With
 * x_i = m_i, y_i is the ordinary mass-summed density. For free supports
 *
 *   Omega_i = 1 + H_i/(3 n_i) dn_i/dH_i,
 *   zeta_i  = H_i/(3 n_i Omega_i) dy_i/dH_i.
 *
 * The Hopkins factor f_ij = 1 - zeta_i/x_j reduces to 1/Omega_i for equal-mass
 * density SPH. A support held at a constant bound has dH_i/dr = 0, so zeta_i = 0
 * there. The pressure force and thermal work that use these corrections are in
 * cosmictiger/sph/variational/equations.hpp. docs/variational_sph.md derives the
 * equations and cosmological scale factors. Sums are taken in units of H, using
 * H^4 dW/dH = -3 w - q w', so that no power of H multiplies a correction.
 */
#pragma once

#include <cstdint>
#include <cstring>

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/simd_math.hpp>

namespace sph_volume {

// Corrections with Omega below this are reported as ill-conditioned: the
// support-solve residual and the summation error are amplified by 1/Omega.
inline constexpr float kMinimumWellConditionedOmega = 0.1f;

// dW/dH = H^-4 [-3 w(q) - q w'(q)] for W(r, H) = H^-3 w(r/H).
template <typename Real>
CUDA_EXPORT inline Real kernelSupportDerivative(
    Real kernelValue, Real kernelDerivative, Real normalizedRadius, Real inverseSupportRadius) {
    Real const inverseSupportSquared = inverseSupportRadius * inverseSupportRadius;
    return -(simdBroadcast<Real>(3) * kernelValue + normalizedRadius * kernelDerivative) *
        inverseSupportSquared * inverseSupportSquared;
}

// grad_i W(r_ij, H) = F r_ij with F = w'(q)/(H^4 r); the caller supplies r > 0.
template <typename Real>
CUDA_EXPORT inline Real kernelGradientFactor(
    Real kernelDerivative, Real inverseSupportRadius, Real radius) {
    Real const inverseSupportSquared = inverseSupportRadius * inverseSupportRadius;
    return kernelDerivative * inverseSupportSquared * inverseSupportSquared / radius;
}

// One sink's neighbor sums at fixed H, including the self term.
template <typename Real>
struct SupportSums {
    Real numberSum;         // sum_j w(q_ij) = H^3 n_i
    Real numberMoment;      // sum_j q_ij w'(q_ij) = -H^4 dn_i/dH - 3 H^3 n_i
    Real weightedSum;       // sum_j x_j w(q_ij) = H^3 y_i
    Real weightedMoment;    // sum_j x_j q_ij w'(q_ij) = -H^4 dy_i/dH - 3 H^3 y_i
};

template <typename Real>
CUDA_EXPORT inline SupportSums<Real> emptySupportSums() {
    Real const zero = simdBroadcast<Real>(0);
    return SupportSums<Real>{zero, zero, zero, zero};
}

template <typename Real>
CUDA_EXPORT inline void accumulateSupportSums(SupportSums<Real>& sums, Real weight,
    Real kernelValue, Real kernelDerivative, Real normalizedRadius) {
    Real const moment = normalizedRadius * kernelDerivative;
    sums.numberSum += kernelValue;
    sums.numberMoment += moment;
    sums.weightedSum += weight * kernelValue;
    sums.weightedMoment += weight * moment;
}

// Production gas has one mass; its weighted sums are the number sums times x.
template <typename Real>
CUDA_EXPORT inline SupportSums<Real> equalWeightSupportSums(
    Real numberSum, Real numberMoment, Real weight) {
    return SupportSums<Real>{numberSum, numberMoment, weight * numberSum, weight * numberMoment};
}

template <typename Real>
CUDA_EXPORT inline Real numberDensity(SupportSums<Real> const& sums, Real inverseSupportRadius) {
    return sums.numberSum * inverseSupportRadius * inverseSupportRadius * inverseSupportRadius;
}

template <typename Real>
CUDA_EXPORT inline Real numberDensityDerivative(
    SupportSums<Real> const& sums, Real inverseSupportRadius) {
    Real const inverseSupportSquared = inverseSupportRadius * inverseSupportRadius;
    return -(simdBroadcast<Real>(3) * sums.numberSum + sums.numberMoment) * inverseSupportSquared *
        inverseSupportSquared;
}

template <typename Real>
CUDA_EXPORT inline Real weightedDensity(SupportSums<Real> const& sums, Real inverseSupportRadius) {
    return sums.weightedSum * inverseSupportRadius * inverseSupportRadius * inverseSupportRadius;
}

template <typename Real>
CUDA_EXPORT inline Real weightedDensityDerivative(
    SupportSums<Real> const& sums, Real inverseSupportRadius) {
    Real const inverseSupportSquared = inverseSupportRadius * inverseSupportRadius;
    return -(simdBroadcast<Real>(3) * sums.weightedSum + sums.weightedMoment) *
        inverseSupportSquared * inverseSupportSquared;
}

// V_i = x_i / y_i.
template <typename Real>
CUDA_EXPORT inline Real volume(Real weight, Real weightedDensityValue) {
    return weight / weightedDensityValue;
}

// rho*_i = m_i y_i / x_i, the density implied by the weighted volume.
template <typename Real>
CUDA_EXPORT inline Real volumeDensity(Real mass, Real weight, Real weightedDensityValue) {
    return mass * weightedDensityValue / weight;
}

// Omega_i = 1 + H/(3n) dn/dH = -sum q w' / (3 sum w), written without the
// cancellation of the defining form. Nonnegative for nonincreasing kernels.
template <typename Real>
CUDA_EXPORT inline Real omega(SupportSums<Real> const& sums) {
    return -sums.numberMoment / (simdBroadcast<Real>(3) * sums.numberSum);
}

// zeta_i = H/(3 n Omega) dy/dH = (3 sum x w + sum x q w') / sum q w', in units of x.
// For equal weights this is x (1 - 1/Omega_i).
template <typename Real>
CUDA_EXPORT inline Real zeta(SupportSums<Real> const& sums) {
    return (simdBroadcast<Real>(3) * sums.weightedSum + sums.weightedMoment) / sums.numberMoment;
}

// Hopkins (2013) equation 13 with x~_j = 1: f_ij = 1 - zeta_i/x_j. Forces use x_j - zeta_i.
template <typename Real>
CUDA_EXPORT inline Real correctionFactor(Real zetaI, Real weightJ) {
    return simdBroadcast<Real>(1) - zetaI / weightJ;
}

// How the stored support depends on the particle positions.
enum class SupportConstraint : int {
    Free,          // H follows H^3 n = constant
    LowerBound,    // held at H_min with more than the target neighbor sum
    UpperBound,    // held at H_max (or the periodic cap) with fewer
    Fixed          // H_min = H_max
};

// relativeResidual is (sum_j w - target)/target at the stored H. A residual within
// the solve tolerance satisfies the constraint and is free even at a bound.
CUDA_EXPORT inline SupportConstraint classifySupport(float supportRadius,
    float minimumSupportRadius, float maximumSupportRadius, float relativeResidual,
    float tolerance) {
    if (minimumSupportRadius == maximumSupportRadius) {
        return SupportConstraint::Fixed;
    }
    if (supportRadius <= minimumSupportRadius && relativeResidual > tolerance) {
        return SupportConstraint::LowerBound;
    }
    if (supportRadius >= maximumSupportRadius && relativeResidual < -tolerance) {
        return SupportConstraint::UpperBound;
    }
    return SupportConstraint::Free;
}

template <typename Real>
struct GradHCorrection {
    Real omega;    // Omega_i of a free support; one when the support is held
    Real zeta;     // zeta_i in units of x; zero when the support is held
};

template <typename Real>
CUDA_EXPORT inline GradHCorrection<Real> gradHCorrection(
    SupportSums<Real> const& sums, SupportConstraint constraint) {
    if (constraint != SupportConstraint::Free) {
        return GradHCorrection<Real>{simdBroadcast<Real>(1), simdBroadcast<Real>(0)};
    }
    return GradHCorrection<Real>{omega(sums), zeta(sums)};
}

// Inspect the exponent bits: -ffast-math may fold std::isfinite to true.
CUDA_EXPORT inline bool isFiniteValue(float value) {
#ifdef __CUDA_ARCH__
    std::uint32_t const bits = __float_as_uint(value);
#else
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
#endif
    return (bits & 0x7f800000U) != 0x7f800000U;
}

// A correction that cannot be used: non-finite, or a nonpositive Omega.
CUDA_EXPORT inline bool isUsable(GradHCorrection<float> const& correction) {
    return isFiniteValue(correction.omega) && isFiniteValue(correction.zeta) &&
        correction.omega > 0.0f;
}

CUDA_EXPORT inline bool isWellConditioned(GradHCorrection<float> const& correction) {
    return isUsable(correction) && correction.omega >= kMinimumWellConditionedOmega;
}

}    // namespace sph_volume
