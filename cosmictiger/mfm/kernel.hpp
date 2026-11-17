/*
 CosmicTiger - A cosmological N-Body code
 Precision-generic compact-support cubic spline for MFM preparation.
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <numbers>

#include <cosmictiger/math/simd_math.hpp>

namespace mfm {

/*
 * The standard M4 cubic spline, written with full support h (twice the conventional
 * spline length); see Monaghan (1992), ARA&A 30, 543, section 2,
 * doi:10.1146/annurev.aa.30.090192.002551. Derived support normalization:
 * W_D(r,h)=h⁻ᴰ w_D(q), q=|r|/h,
 * w_D(q)=C_D [1−6q²+6q³] for 0≤q<1/2, 2 C_D(1−q)³ for 1/2≤q<1, zero otherwise.
 * C_1=4/3, C_2=40/(7π), C_3=8/π, so ∫ W_D dᴰx=1.
 * Symbol directory: q → normalized radius; C_D → dimension-specific normalization.
 * Code names: q2=q²; one_minus_q=1−q.
 */
template <typename Real, int dimensions>
CUDA_EXPORT constexpr Real kernelNormalization() {
    static_assert(dimensions >= 1 && dimensions <= 3);
    using Scalar = SimdScalar<Real>;
    if constexpr (dimensions == 1) {
        return Real(Scalar(4) / Scalar(3));
    } else if constexpr (dimensions == 2) {
        return Real(Scalar(40) / (Scalar(7) * std::numbers::pi_v<Scalar>) );
    } else {
        return Real(Scalar(8) / std::numbers::pi_v<Scalar>);
    }
}

template <typename Real, int dimensions>
CUDA_EXPORT inline Real kernelWeight(Real const& q) {
    Real const q2 = q * q;
    Real const one_minus_q = Real(1) - q;
    Real const inner = Real(1) - Real(6) * q2 + Real(6) * q2 * q;
    Real const outer = Real(2) * one_minus_q * one_minus_q * one_minus_q;
    Real const w = math::select(q < Real(SimdScalar<Real>(0.5)), inner, outer);
    return math::select(
        q >= Real(0) && q < Real(1), kernelNormalization<Real, dimensions>() * w, Real(0));
}

// Derived derivative of the spline above: w′=C_D(−12q+18q²) in the inner piece,
// w′=−6 C_D(1−q)² in the outer piece. Code names: one_minus_q = 1−q.
template <typename Real, int dimensions>
CUDA_EXPORT inline Real kernelDerivative(Real const& q) {
    Real const one_minus_q = Real(1) - q;
    Real const inner = q * (Real(18) * q - Real(12));
    Real const outer = -Real(6) * one_minus_q * one_minus_q;
    Real const wʹ = math::select(q < Real(SimdScalar<Real>(0.5)), inner, outer);
    return math::select(
        q >= Real(0) && q < Real(1), kernelNormalization<Real, dimensions>() * wʹ, Real(0));
}

}    // namespace mfm
