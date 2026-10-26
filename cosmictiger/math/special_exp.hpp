/*
 * CosmicTiger - A cosmological N-Body code
 * Copyright (C) 2026 Dominic C. Marcello
 * Distributed under the Boost Software License, Version 1.0.
 */
#pragma once

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/special_functions_core.hpp>

namespace special_functions {
namespace detail {

    // Degree seven on [-ln(2)/2, ln(2)/2]. Estrin evaluation exposes four
    // independent FMAs, instead of a chain of seven dependent FMAs.
    template <class Real>
#if defined(__GNUC__) && !defined(__CUDACC__)
    [[gnu::always_inline]]
#endif
    CUDA_EXPORT inline Real expPolynomial(Real r) {
        Real const r2 = r * r;
        Real const r4 = r2 * r2;
        Real const p01 = simdFma(r, simdBroadcast<Real>(1), simdBroadcast<Real>(1));
        Real const p23 = simdFma(r, simdBroadcast<Real>(1.0 / 6), simdBroadcast<Real>(0.5));
        Real const p45 = simdFma(r, simdBroadcast<Real>(1.0 / 120), simdBroadcast<Real>(1.0 / 24));
        Real const p67 =
            simdFma(r, simdBroadcast<Real>(1.0 / 5040), simdBroadcast<Real>(1.0 / 720));
        return simdFma(r4, simdFma(r2, p67, p45), simdFma(r2, p23, p01));
    }

    // x + tail is the argument. The caller supplies finite x in [-87, 88] and
    // a small rounding residual. In particular, retaining the residual of -x*x
    // avoids magnifying the square's rounding error in the Gaussian tail.
    template <class Real>
#if defined(__GNUC__) && !defined(__CUDACC__)
    [[gnu::always_inline]]
#endif
    CUDA_EXPORT inline Real expWithTail(Real x, Real tail) {
        Real const n = detail::roundEven(x * simdBroadcast<Real>(0x1.715476p+0f));
        Real r = simdFma(-n, simdBroadcast<Real>(0x1.62e430p-1f), x);
        r = simdFma(n, simdBroadcast<Real>(0x1.05c610p-29f), r);
        r += tail;
        return expPolynomial(r) * detail::powerOfTwo(n);
    }

    template <class Real>
#if defined(__GNUC__) && !defined(__CUDACC__)
    [[gnu::always_inline]]
#endif
    CUDA_EXPORT inline Real exp2Normal(Real x) {
        Real const n = detail::roundEven(x);
        Real const fraction = x - n;
        Real const r = fraction * simdBroadcast<Real>(0x1.62e430p-1f);
        return detail::expPolynomial(r) * detail::powerOfTwo(n);
    }

}    // namespace detail

// Float fast path covers every exponential evaluated by the SIMD Ewald
// kernel, including lanes outside the scalar real-space cutoff (down to -75).
template <class Real>
#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline Real exp(Real x) {
    if constexpr (std::is_arithmetic_v<Real>) {
        static_assert(std::is_same_v<Real, float>);
    } else {
        static_assert(std::is_same_v<typename Real::value_type, float>);
    }
    auto const inRange = (x >= simdBroadcast<Real>(-87)) && (x <= simdBroadcast<Real>(88));
    if (simdAllOf(inRange)) {
        return detail::expWithTail(x, simdBroadcast<Real>(0));
    }
    if constexpr (std::is_arithmetic_v<Real>) {
        return ::expf(x);
    } else {
#ifndef __CUDACC__
        Real const safe = simdSelect(inRange, x, simdBroadcast<Real>(0));
        Real result = detail::expWithTail(safe, simdBroadcast<Real>(0));
        for (size_t lane = 0; lane < Real::size(); ++lane) {
            if (!inRange[lane]) {
                simdSetLane(result, lane, std::exp(float(x[lane])));
            }
        }
        return result;
#endif
    }
}

template <class Real>
#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline Real exp2(Real x) {
    if constexpr (std::is_arithmetic_v<Real>) {
        static_assert(std::is_same_v<Real, float>);
    } else {
        static_assert(std::is_same_v<typename Real::value_type, float>);
    }
    auto const inRange = (x >= simdBroadcast<Real>(-126)) && (x <= simdBroadcast<Real>(127));
    if (simdAllOf(inRange)) {
        return detail::exp2Normal(x);
    }
    if constexpr (std::is_arithmetic_v<Real>) {
        return ::exp2f(x);
    } else {
#ifndef __CUDACC__
        Real const safe = simdSelect(inRange, x, simdBroadcast<Real>(0));
        Real result = detail::exp2Normal(safe);
        for (size_t lane = 0; lane < Real::size(); ++lane) {
            if (!inRange[lane]) {
                simdSetLane(result, lane, std::exp2(float(x[lane])));
            }
        }
        return result;
#endif
    }
}

#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline double exp(double x) {
    return ::exp(x);
}

#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline double exp2(double x) {
    return ::exp2(x);
}

}    // namespace special_functions
