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
 *   J. Comput. Phys. 231 (2012), 759, arXiv:1012.1885, equation 27 (Ω_a).
 *
 * The length here is the stored compact support radius H, with the selected kernel
 * W(r, H) = H^-D w(q) and q = r/H. It is not the papers' smoothing length h, whose
 * normalization differs by kernel; only ratios H dW/dH enter, so the formulas are
 * unchanged. Particle i carries a volume weight x_i and a mass m_i. With
 *
 *   n_i = Σ_j W_ij(H_i),   y_i = Σ_j x_j W_ij(H_i),
 *   V_i = x_i / y_i,         ρ*_i = m_i y_i / x_i,
 *
 * the resolution constraint H_i^D n_i = constant uses the unweighted count n_i. With
 * x_i = m_i, y_i is the ordinary mass-summed density. For free supports
 *
 *   Ω_i = 1 + H_i/(D n_i) dn_i/dH_i,
 *   ζ_i  = H_i/(D n_i Ω_i) dy_i/dH_i.
 *
 * The Hopkins factor f_ij = 1 - ζ_i/x_j reduces to 1/Ω_i for equal-mass
 * density SPH. A support held at a constant bound has dH_i/dr = 0, so ζ_i = 0
 * there. The pressure force and thermal work that use these corrections are in
 * cosmictiger/sph/variable_smoothing_lengths.hpp. docs/variational_sph.md derives the
 * equations and cosmological scale factors. Sums are taken in units of H, using
 * H^(D+1) dW/dH = -D w - q w', so that no power of H multiplies a correction.
 */
#pragma once

#include <cstdint>
#include <cstring>

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/simd_math.hpp>
#include <cosmictiger/sph/geometry.hpp>

namespace sph_volume {

// Corrections with Ω below this are counted as ill-conditioned: Ω is the
// normalized Jacobian of the constraint H^D n = constant, so a relative residual
// left by the support solve is worth about residual/Ω in density.
inline constexpr float kMinimumWellConditionedOmega = 0.1f;

// dW/dH = H^-(D+1) [-D w(q) - q w'(q)] for W(r, H) = H^-D w(r/H).
// clang-format off
/*
 * Derived from W = H⁻ᴰ w(q), q = r/H: ∂W/∂H = −H⁻⁽ᴰ⁺¹⁾[Dw + qw′].
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   w → kernel value; w′ → kernel derivative; q → normalized radius; H⁻¹ → inverse smoothing length.
 *   Code names: wʹ = w′; H_inv = H⁻¹.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real kernelSupportDerivative(Real w, Real wʹ, Real q, Real H_inv) {
    return -(simdBroadcast<Real>(kSpatialDimensions) * w + q * wʹ) *
        sph_geometry::inverseGradientScale(H_inv);
}

// grad_i W(r_ij, H) = F r_ij with F = w'(q)/(H^(D+1) r); the caller supplies r > 0.
// clang-format off
/*
 * Derived radial gradient: ∇_i W_ij = [w′(q)/(H⁽ᴰ⁺¹⁾ r)] r_ij.
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   w′ → kernel derivative; H⁻¹ → inverse smoothing length; r → radius; bracket → return value.
 *   Code names: wʹ = w′; H_inv = H⁻¹.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real kernelGradientFactor(Real wʹ, Real H_inv, Real r) {
    return wʹ * sph_geometry::inverseGradientScale(H_inv) / r;
}

// One sink's neighbor sums at fixed H, including the self term.
template <typename Real>
struct SupportSums {
    Real numberSum;         // sum_j w(q_ij) = H^D n_i
    Real numberMoment;      // sum_j q_ij w'(q_ij) = -H^(D+1) dn_i/dH - D H^D n_i
    Real weightedSum;       // sum_j x_j w(q_ij) = H^D y_i
    Real weightedMoment;    // sum_j x_j q_ij w'(q_ij) = -H^(D+1) dy_i/dH - D H^D y_i
};

template <typename Real>
CUDA_EXPORT inline SupportSums<Real> emptySupportSums() {
    Real const zero = simdBroadcast<Real>(0);
    return SupportSums<Real>{zero, zero, zero, zero};
}

// clang-format off
/*
 * Hopkins (2013), equations (8)–(11): Hᴰ n_i = Σ_j w, Hᴰ y_i = Σ_j x_j w.
 * H⁽ᴰ⁺¹⁾ ∂n_i/∂H = −DΣ_j w − Σ_j qw′; H⁽ᴰ⁺¹⁾ ∂y_i/∂H = −DΣ_j x_j w − Σ_j x_j qw′.
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   x_j → weight; w → kernel value; w′ → kernel derivative; q → normalized radius
 *   Σw, Σqw′ → number sum, number moment
 *   Σxw, Σxqw′ → weighted sum, weighted moment; qw′ → one neighbor's number moment.
 *   Code names: wʹ = w′; qwʹ = qw′.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline void accumulateSupportSums(
    SupportSums<Real>& sums, Real x_j, Real w, Real wʹ, Real q) {
    Real const qwʹ = q * wʹ;
    sums.numberSum += w;
    sums.numberMoment += qwʹ;
    sums.weightedSum += x_j * w;
    sums.weightedMoment += x_j * qwʹ;
}

// Production gas has one mass; its weighted sums are the number sums times x.
// Σw, Σqw′ → number sum, number moment; x → common volume weight. Code names: Σqwʹ = Σqw′.
template <typename Real>
CUDA_EXPORT inline SupportSums<Real> equalWeightSupportSums(Real Σw, Real Σqwʹ, Real x) {
    return SupportSums<Real>{Σw, Σqwʹ, x * Σw, x * Σqwʹ};
}

// clang-format off
/*
 * n_i = H⁻ᴰ Σ_j w(q_ij).
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   Σw → number sum; H⁻¹ → inverse smoothing length; n_i → return value.
 *   Code names: H_inv = H⁻¹.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real numberDensity(SupportSums<Real> const& sums, Real H_inv) {
    return sums.numberSum * sph_geometry::inverseVolume(H_inv);
}

// clang-format off
/*
 * ∂n_i/∂H = −H⁻⁽ᴰ⁺¹⁾(DΣw + Σqw′).
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   Σw → number sum; Σqw′ → number moment; H⁻¹ → inverse smoothing length.
 *   Code names: H_inv = H⁻¹.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real numberDensityDerivative(SupportSums<Real> const& sums, Real H_inv) {
    return -(simdBroadcast<Real>(kSpatialDimensions) * sums.numberSum + sums.numberMoment) *
        sph_geometry::inverseGradientScale(H_inv);
}

// clang-format off
/*
 * y_i = H⁻ᴰ Σ_j x_j w(q_ij).
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   Σxw → weighted sum; H⁻¹ → inverse smoothing length; y_i → return value.
 *   Code names: H_inv = H⁻¹.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real weightedDensity(SupportSums<Real> const& sums, Real H_inv) {
    return sums.weightedSum * sph_geometry::inverseVolume(H_inv);
}

// clang-format off
/*
 * ∂y_i/∂H = −H⁻⁽ᴰ⁺¹⁾(DΣxw + Σxqw′).
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   Σxw → weighted sum; Σxqw′ → weighted moment; H⁻¹ → inverse smoothing length.
 *   Code names: H_inv = H⁻¹.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real weightedDensityDerivative(SupportSums<Real> const& sums, Real H_inv) {
    return -(simdBroadcast<Real>(kSpatialDimensions) * sums.weightedSum + sums.weightedMoment) *
        sph_geometry::inverseGradientScale(H_inv);
}

// V_i = x_i / y_i.
// clang-format off
/*
 * Hopkins (2013), equation (8): ΔV_i = x_i/y_i.
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   x_i → weight; y_i → weighted density value; ΔV_i → return value.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real volume(Real x_i, Real y_i) {
    return x_i / y_i;
}

// ρ*_i = m_i y_i / x_i, the density implied by the weighted volume.
// clang-format off
/*
 * Derived mass density: ρ_i = m_i/ΔV_i = m_i y_i/x_i.
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   m_i → mass; x_i → weight; y_i → weighted density value; ρ_i → return value.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real volumeDensity(Real m_i, Real x_i, Real y_i) {
    return m_i * y_i / x_i;
}

// Ω_i = 1 + H/(Dn) dn/dH = -sum q w' / (D sum w), written without the
// cancellation of the defining form. Nonnegative for nonincreasing kernels.
// clang-format off
/*
 * Price (2012), equation (27); derived equal-weight form:
 * Ω_i = 1 + H_i (∂n_i/∂H_i)/(Dn_i) = −Σ_j q_ij w′/(DΣ_j w).
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   Σw → number sum; Σqw′ → number moment; Ω_i → return value.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real supportJacobian(SupportSums<Real> const& sums) {
    return -sums.numberMoment / (simdBroadcast<Real>(kSpatialDimensions) * sums.numberSum);
}

// ζ_i = H/(D n Ω) dy/dH = (D sum x w + sum x q w') / sum q w', in units of x.
// For equal weights this is x (1 - 1/Ω_i).
// clang-format off
/*
 * Hopkins (2013), equation (13), with x̃_j = 1, derived notation:
 * ζ_i = H_i (∂y_i/∂H_i)/(Dn_i Ω_i) = (DΣxw + Σxqw′)/Σqw′.
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   Σxw → weighted sum; Σxqw′ → weighted moment
 *   Σqw′ → number moment; ζ_i → return value.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real weightedSupportCorrection(SupportSums<Real> const& sums) {
    return (simdBroadcast<Real>(kSpatialDimensions) * sums.weightedSum + sums.weightedMoment) /
        sums.numberMoment;
}

// Hopkins (2013) equation 13 with x~_j = 1: f_ij = 1 - ζ_i/x_j. Forces use x_j - ζ_i.
// clang-format off
/*
 * Hopkins (2013), equation (13), with x̃_j = 1: f_ij = 1 − ζ_i/x_j.
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   ζ_i → sink grad-H correction; x_j → neighbor volume weight; f_ij → return value.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real correctionFactor(Real ζ_i, Real x_j) {
    return simdBroadcast<Real>(1) - ζ_i / x_j;
}

// How the stored support depends on the particle positions.
enum class SupportConstraint : int {
    Free,          // H follows H^D n = constant
    LowerBound,    // held at H_min with more than the target neighbor sum
    UpperBound,    // held at H_max (or the periodic cap) with fewer
    Fixed          // H_min = H_max
};

// relativeResidual is (sum_j w - target)/target at the stored H. A residual within
// the solve tolerance satisfies the constraint and is free even at a bound.
// H → stored support radius; H_min, H_max → its bounds.
CUDA_EXPORT inline SupportConstraint classifySupport(
    float H, float H_min, float H_max, float relativeResidual, float tolerance) {
    if (H_min == H_max) {
        return SupportConstraint::Fixed;
    }
    if (H <= H_min && relativeResidual > tolerance) {
        return SupportConstraint::LowerBound;
    }
    if (H >= H_max && relativeResidual < -tolerance) {
        return SupportConstraint::UpperBound;
    }
    return SupportConstraint::Free;
}

template <typename Real>
struct GradHCorrection {
    Real supportJacobian;              // Ω_i of a free support; one when the support is held
    Real weightedSupportCorrection;    // ζ_i in units of x; zero when the support is held
};

template <typename Real>
CUDA_EXPORT inline GradHCorrection<Real> gradHCorrection(
    SupportSums<Real> const& sums, SupportConstraint constraint) {
    if (constraint != SupportConstraint::Free) {
        return GradHCorrection<Real>{simdBroadcast<Real>(1), simdBroadcast<Real>(0)};
    }
    return GradHCorrection<Real>{supportJacobian(sums), weightedSupportCorrection(sums)};
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

// A correction that cannot be used: non-finite, or a nonpositive Ω.
CUDA_EXPORT inline bool isUsable(GradHCorrection<float> const& correction) {
    return isFiniteValue(correction.supportJacobian) &&
        isFiniteValue(correction.weightedSupportCorrection) && correction.supportJacobian > 0.0f;
}

CUDA_EXPORT inline bool isWellConditioned(GradHCorrection<float> const& correction) {
    return isUsable(correction) && correction.supportJacobian >= kMinimumWellConditionedOmega;
}

}    // namespace sph_volume
