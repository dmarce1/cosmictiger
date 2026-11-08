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
 * density SPH. Near a bound the constraint's target depends on h itself (the bounded
 * target below); the support stiffens continuously, Ωᵢ → 1 and ζᵢ → 0 at the bound, and a
 * fixed support (hₘᵢₙ = hₘₐₓ) has ζᵢ = 0. The pressure force and thermal work that use these
 * corrections are in cosmictiger/sph/variable_smoothing_lengths.hpp. docs/variational_sph.md
 * derives the equations and cosmological scale factors. Sums are taken in units of h, using h^(D+1)
 * dW/dh = -D w - q w', so that no power of h multiplies a correction.
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
    return -(simdBroadcast<Real>(kSpatialDimensions) * w + q * wʹ) *
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

template <typename Real>
CUDA_EXPORT inline SupportSums<Real> emptySupportSums() {
    Real const zero = simdBroadcast<Real>(0);
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
    return -(simdBroadcast<Real>(kSpatialDimensions) * sums.numberSum + sums.numberMoment) *
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
    return -(simdBroadcast<Real>(kSpatialDimensions) * sums.weightedSum + sums.weightedMoment) *
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
    return -sums.numberMoment / (simdBroadcast<Real>(kSpatialDimensions) * sums.numberSum);
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
    return (simdBroadcast<Real>(kSpatialDimensions) * sums.weightedSum + sums.weightedMoment) /
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
    return simdBroadcast<Real>(1) - ζᵢ / xⱼ;
}

// clang-format off
/*
 * Bounded neighbor target (derived; no source equation). The smoothing-length solve
 * holds Σⱼ w(qᵢⱼ) = N(h)/V_D, with V_D the unit D-ball volume, for a target that
 * depends on the smoothing length itself,
 *   N(h) = N₀ f(h),
 *   f = 1/(1 − s³),  s = (h₁ − h)/(h₁ − hₘᵢₙ)    for hₘᵢₙ < h < h₁ = λ hₘᵢₙ,
 *   f = 1                                      for h₁ ≤ h ≤ h₂,
 *   f = 1 − t³,      t = (h − h₂)/(hₘₐₓ − h₂)    for h₂ = μ hₘₐₓ < h < hₘₐₓ.
 * f → ∞ at hₘᵢₙ and f → 0 at hₘₐₓ; f, f′ and f″ are continuous at both junctions
 * because each layer departs from one as a cubic. When λ hₘᵢₙ ≥ μ hₘₐₓ both junctions
 * move to (hₘᵢₙ + hₘₐₓ)/2 and the constant piece disappears. Σⱼ w is nondecreasing and
 * N(h) strictly decreasing in h, so the residual Σⱼ w − N(h)/V_D has exactly one root,
 * strictly inside (hₘᵢₙ, hₘₐₓ): the bounds are never reached, and no support is held.
 *
 * The slopes:
 *   lower layer: f′ = −3 s² f²/(h₁ − hₘᵢₙ),   upper layer: f′ = −3 t²/(hₘₐₓ − h₂).
 * The grad-h terms need the logarithmic slope χ = −d ln N/d ln h = −h f′/f through
 *   ψ = D/(D + χ) ∈ [0, 1],
 *   lower layer: ψ = D (h₁ − hₘᵢₙ)(1 − s³) / [D (h₁ − hₘᵢₙ)(1 − s³) + 3 s² h],
 *   upper layer: ψ = D (hₘₐₓ − h₂)(1 − t³) / [D (hₘₐₓ − h₂)(1 − t³) + 3 t² h],
 * which is one in the interior and zero at either bound; ψ is formed directly so that it
 * stays finite where χ diverges. A fixed support (hₘᵢₙ = hₘₐₓ) also has ψ = 0.
 *
 * Symbol directory:
 *   h → smoothing length; hₘᵢₙ, hₘₐₓ → its bounds; h₁, h₂ → layer junctions
 *   λ, μ → junction ratios; s, t → lower, upper layer coordinates
 *   N₀ → requested neighbor number; f → N(h)/N₀; f′ → df/dh
 *   χ → logarithmic slope of N(h); ψ → support stiffness D/(D + χ); D → dimension.
 *   Code names: h_1 = h₁; h_2 = h₂; kLowerLayerRatio = λ; kUpperLayerRatio = μ;
 *   one_minus_s3 = 1 − s³; one_minus_t3 = 1 − t³.
 */
// clang-format on
inline constexpr double kLowerLayerRatio = 1.5;    // λ: the lower layer ends at λ hₘᵢₙ
inline constexpr double kUpperLayerRatio = 0.8;    // μ: the upper layer starts at μ hₘₐₓ

// Iterates of the smoothing-length solve stay this fraction of a layer inside the
// bounds, where the target is singular.
inline constexpr double kSupportMargin = 1.0 / 65536.0;

template <typename Real>
struct SupportLayers {
    Real lowerJunction;    // h₁
    Real upperJunction;    // h₂
};

template <typename Real>
CUDA_EXPORT inline SupportLayers<Real> supportLayers(Real hₘᵢₙ, Real hₘₐₓ) {
    Real h_1 = Real(kLowerLayerRatio) * hₘᵢₙ;
    Real h_2 = Real(kUpperLayerRatio) * hₘₐₓ;
    if (!(h_1 < h_2)) {
        h_1 = h_2 = Real(0.5) * (hₘᵢₙ + hₘₐₓ);
    }
    return SupportLayers<Real>{h_1, h_2};
}

template <typename Real>
struct BoundedTarget {
    Real factor;       // f = N(h)/N₀
    Real slope;        // f′ = df/dh, nonpositive
    Real stiffness;    // ψ = D/(D + χ): one in the interior, zero at a bound
};

// The hydro sources are compiled with -ffast-math, which lets the compiler evaluate both
// layer branches speculatively; a layer of zero width (merged junctions at a fixed support)
// therefore takes a positive stand-in width so that no speculative 0/0 can trap.
template <typename Real>
CUDA_EXPORT inline BoundedTarget<Real> boundedTarget(
    Real h, Real hₘᵢₙ, Real hₘₐₓ, SupportLayers<Real> const& layers) {
    Real const D = Real(kSpatialDimensions);
    BoundedTarget<Real> result{Real(1), Real(0), Real(1)};
    if (h < layers.lowerJunction) {
        Real const width = layers.lowerJunction > hₘᵢₙ ? layers.lowerJunction - hₘᵢₙ : Real(1);
        Real const s = (layers.lowerJunction - h) / width;
        Real const s2 = s * s;
        Real const one_minus_s3 = Real(1) - s2 * s;
        result.factor = Real(1) / one_minus_s3;
        result.slope = -Real(3) * s2 * result.factor * result.factor / width;
        Real const interior = D * width * one_minus_s3;
        result.stiffness = interior / (interior + Real(3) * s2 * h);
    } else if (h > layers.upperJunction) {
        Real const width = hₘₐₓ > layers.upperJunction ? hₘₐₓ - layers.upperJunction : Real(1);
        Real const t = (h - layers.upperJunction) / width;
        Real const t2 = t * t;
        Real const one_minus_t3 = Real(1) - t2 * t;
        result.factor = one_minus_t3;
        result.slope = -Real(3) * t2 / width;
        Real const interior = D * width * one_minus_t3;
        result.stiffness = interior / (interior + Real(3) * t2 * h);
    }
    return result;
}

template <typename Real>
struct GradHCorrection {
    Real supportJacobian;              // Ωᵢ
    Real weightedSupportCorrection;    // ζᵢ in units of x
};

// clang-format off
/*
 * Grad-h correction of a support with stiffness ψ (derived from the bounded target
 * above by repeating the Price (2012) and Hopkins (2013) derivations with the
 * h-dependent constraint hᴰ nᵢ = N(hᵢ)/V_D):
 *   Ωᵢ = ψ Ω⁰ᵢ + (1 − ψ),                 Ω⁰ᵢ = −Σⱼ qw′/(DΣ_j w),
 *   ζᵢ = ψ (DΣxw + Σxqw′) / [ψ Σqw′ − D (1 − ψ) Σw].
 * ψ = 1 recovers Price (2012) equation (27) and Hopkins (2013) equation (13); ψ = 0
 * gives Ωᵢ = 1 and ζᵢ = 0, the fixed-support limit. For equal weights
 * ζᵢ = x (1 − 1/Ωᵢ) holds for every ψ. Both forms are finite for every ψ in [0, 1].
 * References: Hopkins (2013), https://arxiv.org/abs/1206.5006;
 * Price (2012), https://arxiv.org/abs/1012.1885; full citations above.
 * Symbol directory:
 *   ψ → support stiffness; Ω⁰ᵢ → free-support Jacobian (supportJacobian)
 *   Σw, Σqw′ → number sum, number moment; Σxw, Σxqw′ → weighted sum, weighted moment
 *   Ωᵢ, ζᵢ → return values. Code names: one_minus_ψ = 1 − ψ.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline GradHCorrection<Real> gradHCorrection(SupportSums<Real> const& sums, Real ψ) {
    Real const D = simdBroadcast<Real>(kSpatialDimensions);
    Real const one_minus_ψ = simdBroadcast<Real>(1) - ψ;
    Real const Ω = ψ * supportJacobian(sums) + one_minus_ψ;
    Real const ζ = ψ * (D * sums.weightedSum + sums.weightedMoment) /
        (ψ * sums.numberMoment - D * one_minus_ψ * sums.numberSum);
    return GradHCorrection<Real>{Ω, ζ};
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
