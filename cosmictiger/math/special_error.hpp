/*
 * CosmicTiger - A cosmological N-Body code
 * Copyright (C) 2026 Dominic C. Marcello
 * Distributed under the Boost Software License, Version 1.0.
 */
#pragma once

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/special_exp.hpp>

namespace special_functions {
namespace detail {

    // Approximate erfcx(x)/t, t = 1/(1+x), in z = 2*t-1. This smooth transformed
    // function needs no coefficient gathers or interval selection. Coefficients
    // are a degree-15 Chebyshev interpolant converted to powers of z.
    // Reproduce with tools/generate_special_coefficients.py.
    template <class Real>
#if defined(__GNUC__) && !defined(__CUDACC__)
    [[gnu::always_inline]]
#endif
    CUDA_EXPORT inline Real erfcScaledPolynomial(Real z) {
        Real const z2 = z * z;
        Real const z4 = z2 * z2;
        Real const z8 = z4 * z4;
        Real const p01 = math::fma(
            z, math::broadcast<Real>(0.23768090801657485), math::broadcast<Real>(0.8551671515542169));
        Real const p23 = math::fma(z, math::broadcast<Real>(-0.013909097020869406),
            math::broadcast<Real>(-0.09555637894239456));
        Real const p45 = math::fma(z, math::broadcast<Real>(-0.009541354226376413),
            math::broadcast<Real>(0.025395408215796715));
        Real const p67 = math::fma(z, math::broadcast<Real>(0.00363832667429335),
            math::broadcast<Real>(-0.0012273681249999522));
        Real const p89 = math::fma(z, math::broadcast<Real>(0.0006177752573986548),
            math::broadcast<Real>(-0.0023080431800590415));
        Real const p1011 = math::fma(z, math::broadcast<Real>(-0.0006844932303980659),
            math::broadcast<Real>(0.00042624070251109616));
        Real const p1213 = math::fma(z, math::broadcast<Real>(7.516425100350753e-05),
            math::broadcast<Real>(0.0003123094208618928));
        Real const p1415 = math::fma(z, math::broadcast<Real>(2.7979785401400326e-05),
            math::broadcast<Real>(-0.00011452929646717468));
        Real const lower = math::fma(z4, math::fma(z2, p67, p45), math::fma(z2, p23, p01));
        Real const upper = math::fma(z4, math::fma(z2, p1415, p1213), math::fma(z2, p1011, p89));
        return math::fma(z8, upper, lower);
    }

    template <class Real>
#if defined(__GNUC__) && !defined(__CUDACC__)
    [[gnu::always_inline]]
#endif
    CUDA_EXPORT inline void positiveErfcexp(Real x, Real* complement, Real* exponential) {
        Real const square = x * x;
        Real const residual = math::fma(-x, x, square);
        *exponential = detail::expWithTail(-square, residual);
        Real const t = math::broadcast<Real>(1) / (math::broadcast<Real>(1) + x);
        Real const z = math::fma(math::broadcast<Real>(2), t, math::broadcast<Real>(-1));
        *complement = (t * detail::erfcScaledPolynomial(z)) * *exponential;
    }

    template <class Real>
#if defined(__GNUC__) && !defined(__CUDACC__)
    [[gnu::always_inline]]
#endif
    CUDA_EXPORT inline Real erfSmall(Real x) {
        Real const x2 = x * x;
        Real p = math::broadcast<Real>(1.6462114365889247e-6);
        p = math::fma(p, x2, math::broadcast<Real>(-0.000014925650358406251));
        p = math::fma(p, x2, math::broadcast<Real>(0.00012055332981789664));
        p = math::fma(p, x2, math::broadcast<Real>(-0.00085483270234508528));
        p = math::fma(p, x2, math::broadcast<Real>(0.0052239776254421878));
        p = math::fma(p, x2, math::broadcast<Real>(-0.026866170645131252));
        p = math::fma(p, x2, math::broadcast<Real>(0.11283791670955126));
        p = math::fma(p, x2, math::broadcast<Real>(-0.37612638903183752));
        p = math::fma(p, x2, math::broadcast<Real>(1.1283791670955126));
        return x * p;
    }

}    // namespace detail

// Both outputs refer to the exact represented input x: exp(-x*x) retains
// the multiplication residual rather than exponentiating a rounded square.
template <class Real>
#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline void erfcexp(Real x, Real* complement, Real* exponential) {
    if constexpr (std::is_arithmetic_v<Real>) {
        static_assert(std::is_same_v<Real, float>);
    } else {
        static_assert(std::is_same_v<typename Real::value_type, float>);
    }
    Real const magnitude = math::max(x, -x);
    auto const inRange = magnitude <= math::broadcast<Real>(9);
    Real const safe = math::select(inRange, magnitude, math::broadcast<Real>(0));
    detail::positiveErfcexp(safe, complement, exponential);
    *complement =
        math::select(x < math::broadcast<Real>(0), math::broadcast<Real>(2) - *complement, *complement);
    if (!math::allOf(inRange)) {
        if constexpr (std::is_arithmetic_v<Real>) {
            *complement = ::erfcf(x);
            // Outside the optimized interval use a double square to retain
            // the Gaussian's tiny representable float/subnormal results.
            *exponential = float(::exp(-double(x) * double(x)));
        } else {
#ifndef __CUDACC__
            for (size_t lane = 0; lane < Real::size(); ++lane) {
                if (!inRange[lane]) {
                    float const value = x[lane];
                    simdSetLane(*complement, lane, std::erfc(value));
                    simdSetLane(*exponential, lane, float(std::exp(-double(value) * value)));
                }
            }
#endif
        }
    }
}

template <class Real>
#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline Real erfc(Real x) {
    Real complement, exponential;
    special_functions::erfcexp(x, &complement, &exponential);
    return complement;
}

template <class Real>
#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline Real erf(Real x) {
    Real const magnitude = math::max(x, -x);
    auto const small = magnitude <= math::broadcast<Real>(0.75);
    if (math::allOf(small)) {
        return detail::erfSmall(x);
    }
    Real result = math::broadcast<Real>(1) - special_functions::erfc(magnitude);
    result = math::select(x < math::broadcast<Real>(0), -result, result);
    if (math::anyOf(small)) {
        result = math::select(
            small, detail::erfSmall(math::select(small, x, math::broadcast<Real>(0))), result);
    }
    return result;
}

#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline void erfcexp(double x, double* complement, double* exponential) {
    *complement = ::erfc(x);
    double const square = x * x;
    *exponential = ::exp(-square);
    if (square <= 81.0) {
        double const residual = math::fma(-x, x, square);
        *exponential = math::fma(*exponential, residual, *exponential);
    }
}

#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline double erfc(double x) {
    return ::erfc(x);
}

#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline double erf(double x) {
    return ::erf(x);
}

}    // namespace special_functions
