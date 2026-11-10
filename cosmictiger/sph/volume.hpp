/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

/*
 * Smoothing-length, weighted-volume and grad-H correction algebra for SPH, after
 *
 *   P. F. Hopkins, "A general class of Lagrangian smoothed particle hydrodynamics
 *   methods and implications for fluid mixing problems", MNRAS 428 (2013), 2840,
 *   arXiv:1206.5006, equations 8-11 (yᵢ and its derivatives) and equation 13
 *   (fᵢⱼ, with the constraint scalar x~ = 1), and
 *   D. J. Price, "Smoothed particle hydrodynamics and magnetohydrodynamics",
 *   J. Comput. Phys. 231 (2012), 759, arXiv:1012.1885, equation 27 (Ω𞁑).
 *
 * The length here is the stored smoothing length h, the full compact support of the kernel
 * W(r, h) = h^-D w(q) and q = r/h. It is not the kernel-normalized length of Hopkins
 * (2013) and Price (2012), whose normalization differs by kernel; only ratios h dW/dh enter, so the
 * formulas are unchanged. Particle i carries a volume weight xᵢ and a mass mᵢ. With
 *
 *   nᵢ = Σⱼ Wᵢⱼ(hᵢ),   yᵢ = Σⱼ xⱼ Wᵢⱼ(hᵢ),
 *   Vᵢ = xᵢ / yᵢ,         ρ*ᵢ = mᵢ yᵢ / xᵢ,
 *
 * the resolution constraint hᵢ^D nᵢ = constant uses the unweighted count nᵢ. With
 * xᵢ = mᵢ, yᵢ is the ordinary mass-summed density. For free supports
 *
 *   Ωᵢ = 1 + hᵢ/(D nᵢ) dnᵢ/dhᵢ,
 *   ζᵢ  = hᵢ/(D nᵢ Ωᵢ) dyᵢ/dhᵢ.
 *
 * The Hopkins factor fᵢⱼ = 1 - ζᵢ/xⱼ reduces to 1/Ωᵢ for equal-mass
 * density SPH. A support held at a constant bound has dhᵢ/dr = 0, so ζᵢ = 0
 * there. The pressure force and thermal work that use these corrections are in
 * cosmictiger/sph/variable_smoothing_lengths.hpp. docs/variational_sph.md derives the
 * equations and cosmological scale factors. Sums are taken in units of h, using
 * h^(D+1) dW/dh = -D w - q w', so that no power of h multiplies a correction.
 */
#pragma once

#include <cstdint>
#include <cstring>

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/simd_math.hpp>
#include <cosmictiger/sph/geometry.hpp>

namespace sph_volume {

// Corrections with Ω below this are counted as ill-conditioned: Ω is the
// normalized Jacobian of the constraint h^D n = constant, so a relative residual
// left by the support solve is worth about residual/Ω in density.
inline constexpr float kMinimumWellConditionedOmega = 0.1f;

// dW/dh = h^-(D+1) [-D w(q) - q w'(q)] for W(r, h) = h^-D w(r/h).
// clang-format off
/*
 * Derived from W = h⁻ᴰ w(q), q = r/h: ∂W/∂h = −h⁻⁽ᴰ⁺¹⁾[Dw + qw′].
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   w → kernel value; w′ → kernel derivative; q → normalized radius; h⁻¹ → inverse smoothing length.
 *   Code names: wʹ = w′; ih = h⁻¹.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real kernelSupportDerivative(Real w, Real wʹ, Real q, Real ih) {
    return -(math::broadcast<Real>(kSpatialDimensions) * w + q * wʹ) *
        sph_geometry::inverseGradientScale(ih);
}

// grad_i W(rᵢⱼ, h) = F rᵢⱼ with F = w'(q)/(h^(D+1) r); the caller supplies r > 0.
// clang-format off
/*
 * Derived radial gradient: ∇ᵢ Wᵢⱼ = [w′(q)/(h⁽ᴰ⁺¹⁾ r)] rᵢⱼ.
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   w′ → kernel derivative; h⁻¹ → inverse smoothing length; r → radius; bracket → return value.
 *   Code names: wʹ = w′; ih = h⁻¹.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real kernelGradientFactor(Real wʹ, Real ih, Real r) {
    return wʹ * sph_geometry::inverseGradientScale(ih) / r;
}

// One sink's neighbor sums at fixed h, including the self term.
template <typename Real>
struct SupportSums {
    Real numberSum;         // sum_j w(qᵢⱼ) = h^D nᵢ
    Real numberMoment;      // sum_j qᵢⱼ w'(qᵢⱼ) = -h^(D+1) dnᵢ/dh - D h^D nᵢ
    Real weightedSum;       // sum_j xⱼ w(qᵢⱼ) = h^D yᵢ
    Real weightedMoment;    // sum_j xⱼ qᵢⱼ w'(qᵢⱼ) = -h^(D+1) dyᵢ/dh - D h^D yᵢ
};

/// Initialize all scalar/SIMD support sums to zero.
template <typename Real>
CUDA_EXPORT inline SupportSums<Real> emptySupportSums() {
    Real const zero = math::broadcast<Real>(0);
    return SupportSums<Real>{zero, zero, zero, zero};
}

// clang-format off
/*
 * Hopkins (2013), equations (8)–(11): hᴰ nᵢ = Σⱼ w, hᴰ yᵢ = Σⱼ xⱼ w.
 * h⁽ᴰ⁺¹⁾ ∂nᵢ/∂h = −DΣ_j w − Σⱼ qw′; h⁽ᴰ⁺¹⁾ ∂yᵢ/∂h = −DΣ_j xⱼ w − Σⱼ xⱼ qw′.
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   xⱼ → weight; w → kernel value; w′ → kernel derivative; q → normalized radius
 *   Σw, Σqw′ → number sum, number moment
 *   Σxw, Σxqw′ → weighted sum, weighted moment; qw′ → one neighbor's number moment.
 *   Code names: wʹ = w′; qwʹ = qw′.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline void accumulateSupportSums(
    SupportSums<Real>& sums, Real xⱼ, Real w, Real wʹ, Real q) {
    Real const qwʹ = q * wʹ;
    sums.numberSum += w;
    sums.numberMoment += qwʹ;
    sums.weightedSum += xⱼ * w;
    sums.weightedMoment += xⱼ * qwʹ;
}

// Production gas has one mass; its weighted sums are the number sums times x.
// Σw, Σqw′ → number sum, number moment; x → common volume weight. Code names: Σqwʹ = Σqw′.
template <typename Real>
CUDA_EXPORT inline SupportSums<Real> equalWeightSupportSums(Real Σw, Real Σqwʹ, Real x) {
    return SupportSums<Real>{Σw, Σqwʹ, x * Σw, x * Σqwʹ};
}

// clang-format off
/*
 * nᵢ = h⁻ᴰ Σⱼ w(qᵢⱼ).
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   Σw → number sum; h⁻¹ → inverse smoothing length; nᵢ → return value.
 *   Code names: ih = h⁻¹.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real numberDensity(SupportSums<Real> const& sums, Real ih) {
    return sums.numberSum * sph_geometry::inverseVolume(ih);
}

// clang-format off
/*
 * ∂nᵢ/∂h = −h⁻⁽ᴰ⁺¹⁾(DΣw + Σqw′).
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   Σw → number sum; Σqw′ → number moment; h⁻¹ → inverse smoothing length.
 *   Code names: ih = h⁻¹.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real numberDensityDerivative(SupportSums<Real> const& sums, Real ih) {
    return -(math::broadcast<Real>(kSpatialDimensions) * sums.numberSum + sums.numberMoment) *
        sph_geometry::inverseGradientScale(ih);
}

// clang-format off
/*
 * yᵢ = h⁻ᴰ Σⱼ xⱼ w(qᵢⱼ).
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   Σxw → weighted sum; h⁻¹ → inverse smoothing length; yᵢ → return value.
 *   Code names: ih = h⁻¹.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real weightedDensity(SupportSums<Real> const& sums, Real ih) {
    return sums.weightedSum * sph_geometry::inverseVolume(ih);
}

// clang-format off
/*
 * ∂yᵢ/∂h = −h⁻⁽ᴰ⁺¹⁾(DΣxw + Σxqw′).
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   Σxw → weighted sum; Σxqw′ → weighted moment; h⁻¹ → inverse smoothing length.
 *   Code names: ih = h⁻¹.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real weightedDensityDerivative(SupportSums<Real> const& sums, Real ih) {
    return -(math::broadcast<Real>(kSpatialDimensions) * sums.weightedSum + sums.weightedMoment) *
        sph_geometry::inverseGradientScale(ih);
}

// Vᵢ = xᵢ / yᵢ.
// clang-format off
/*
 * Hopkins (2013), equation (8): ΔVᵢ = xᵢ/yᵢ.
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   xᵢ → weight; yᵢ → weighted density value; ΔVᵢ → return value.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real volume(Real xᵢ, Real yᵢ) {
    return xᵢ / yᵢ;
}

// ρ*ᵢ = mᵢ yᵢ / xᵢ, the density implied by the weighted volume.
// clang-format off
/*
 * Derived mass density: ρᵢ = mᵢ/ΔVᵢ = mᵢ yᵢ/xᵢ.
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   mᵢ → mass; xᵢ → weight; yᵢ → weighted density value; ρᵢ → return value.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real volumeDensity(Real mᵢ, Real xᵢ, Real yᵢ) {
    return mᵢ * yᵢ / xᵢ;
}

// Ωᵢ = 1 + h/(Dn) dn/dh = -sum q w' / (D sum w), written without the
// cancellation of the defining form. Nonnegative for nonincreasing kernels.
// clang-format off
/*
 * Price (2012), equation (27); derived equal-weight form:
 * Ωᵢ = 1 + hᵢ (∂nᵢ/∂hᵢ)/(Dnᵢ) = −Σⱼ qᵢⱼ w′/(DΣ_j w).
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   Σw → number sum; Σqw′ → number moment; Ωᵢ → return value.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real supportJacobian(SupportSums<Real> const& sums) {
    return -sums.numberMoment / (math::broadcast<Real>(kSpatialDimensions) * sums.numberSum);
}

// ζᵢ = h/(D n Ω) dy/dh = (D sum x w + sum x q w') / sum q w', in units of x.
// For equal weights this is x (1 - 1/Ωᵢ).
// clang-format off
/*
 * Hopkins (2013), equation (13), with x̃ⱼ = 1, derived notation:
 * ζᵢ = hᵢ (∂yᵢ/∂hᵢ)/(Dnᵢ Ωᵢ) = (DΣxw + Σxqw′)/Σqw′.
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   Σxw → weighted sum; Σxqw′ → weighted moment
 *   Σqw′ → number moment; ζᵢ → return value.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real weightedSupportCorrection(SupportSums<Real> const& sums) {
    return (math::broadcast<Real>(kSpatialDimensions) * sums.weightedSum + sums.weightedMoment) /
        sums.numberMoment;
}

// Hopkins (2013) equation 13 with x~ⱼ = 1: fᵢⱼ = 1 - ζᵢ/xⱼ. Forces use xⱼ - ζᵢ.
// clang-format off
/*
 * Hopkins (2013), equation (13), with x̃ⱼ = 1: fᵢⱼ = 1 − ζᵢ/xⱼ.
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   ζᵢ → sink grad-H correction; xⱼ → neighbor volume weight; fᵢⱼ → return value.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real correctionFactor(Real ζᵢ, Real xⱼ) {
    return math::broadcast<Real>(1) - ζᵢ / xⱼ;
}

// How the stored support depends on the particle positions.
enum class SupportConstraint : int {
    Free,          // h follows h^D n = constant
    LowerBound,    // held at hₘᵢₙ with more than the target neighbor sum
    UpperBound,    // held at hₘₐₓ (or the periodic cap) with fewer
    Fixed          // hₘᵢₙ = hₘₐₓ
};

// relativeResidual is (sum_j w - target)/target at the stored h. A residual within
// the solve tolerance satisfies the constraint and is free even at a bound.
// h → stored smoothing length; hₘᵢₙ, hₘₐₓ → its bounds.
CUDA_EXPORT inline SupportConstraint classifySupport(
    float h, float hₘᵢₙ, float hₘₐₓ, float relativeResidual, float tolerance) {
    if (hₘᵢₙ == hₘₐₓ) {
        return SupportConstraint::Fixed;
    }
    if (h <= hₘᵢₙ && relativeResidual > tolerance) {
        return SupportConstraint::LowerBound;
    }
    if (h >= hₘₐₓ && relativeResidual < -tolerance) {
        return SupportConstraint::UpperBound;
    }
    return SupportConstraint::Free;
}

template <typename Real>
struct GradHCorrection {
    Real supportJacobian;              // Ωᵢ of a free support; one when the support is held
    Real weightedSupportCorrection;    // ζᵢ in units of x; zero when the support is held
};

/// Return free-support grad-h factors, or neutral corrections for a constrained smoothing
/// length.
template <typename Real>
CUDA_EXPORT inline GradHCorrection<Real> gradHCorrection(
    SupportSums<Real> const& sums, SupportConstraint constraint) {
    if (constraint != SupportConstraint::Free) {
        return GradHCorrection<Real>{math::broadcast<Real>(1), math::broadcast<Real>(0)};
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

/// Require usable grad-h data and a support Jacobian above the conditioning threshold.
CUDA_EXPORT inline bool isWellConditioned(GradHCorrection<float> const& correction) {
    return isUsable(correction) && correction.supportJacobian >= kMinimumWellConditionedOmega;
}

}    // namespace sph_volume
