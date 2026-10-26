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
        Real const p01 = simdFma(
            z, simdBroadcast<Real>(0.23768090801657485), simdBroadcast<Real>(0.8551671515542169));
        Real const p23 = simdFma(z, simdBroadcast<Real>(-0.013909097020869406),
            simdBroadcast<Real>(-0.09555637894239456));
        Real const p45 = simdFma(z, simdBroadcast<Real>(-0.009541354226376413),
            simdBroadcast<Real>(0.025395408215796715));
        Real const p67 = simdFma(z, simdBroadcast<Real>(0.00363832667429335),
            simdBroadcast<Real>(-0.0012273681249999522));
        Real const p89 = simdFma(z, simdBroadcast<Real>(0.0006177752573986548),
            simdBroadcast<Real>(-0.0023080431800590415));
        Real const p1011 = simdFma(z, simdBroadcast<Real>(-0.0006844932303980659),
            simdBroadcast<Real>(0.00042624070251109616));
        Real const p1213 = simdFma(z, simdBroadcast<Real>(7.516425100350753e-05),
            simdBroadcast<Real>(0.0003123094208618928));
        Real const p1415 = simdFma(z, simdBroadcast<Real>(2.7979785401400326e-05),
            simdBroadcast<Real>(-0.00011452929646717468));
        Real const lower = simdFma(z4, simdFma(z2, p67, p45), simdFma(z2, p23, p01));
        Real const upper = simdFma(z4, simdFma(z2, p1415, p1213), simdFma(z2, p1011, p89));
        return simdFma(z8, upper, lower);
    }

    template <class Real>
#if defined(__GNUC__) && !defined(__CUDACC__)
    [[gnu::always_inline]]
#endif
    CUDA_EXPORT inline void positiveErfcexp(Real x, Real* complement, Real* exponential) {
        Real const square = x * x;
        Real const residual = simdFma(-x, x, square);
        *exponential = detail::expWithTail(-square, residual);
        Real const t = simdBroadcast<Real>(1) / (simdBroadcast<Real>(1) + x);
        Real const z = simdFma(simdBroadcast<Real>(2), t, simdBroadcast<Real>(-1));
        *complement = (t * detail::erfcScaledPolynomial(z)) * *exponential;
    }

    template <class Real>
#if defined(__GNUC__) && !defined(__CUDACC__)
    [[gnu::always_inline]]
#endif
    CUDA_EXPORT inline Real erfSmall(Real x) {
        Real const x2 = x * x;
        Real p = simdBroadcast<Real>(1.6462114365889247e-6);
        p = simdFma(p, x2, simdBroadcast<Real>(-0.000014925650358406251));
        p = simdFma(p, x2, simdBroadcast<Real>(0.00012055332981789664));
        p = simdFma(p, x2, simdBroadcast<Real>(-0.00085483270234508528));
        p = simdFma(p, x2, simdBroadcast<Real>(0.0052239776254421878));
        p = simdFma(p, x2, simdBroadcast<Real>(-0.026866170645131252));
        p = simdFma(p, x2, simdBroadcast<Real>(0.11283791670955126));
        p = simdFma(p, x2, simdBroadcast<Real>(-0.37612638903183752));
        p = simdFma(p, x2, simdBroadcast<Real>(1.1283791670955126));
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
    Real const magnitude = simdMax(x, -x);
    auto const inRange = magnitude <= simdBroadcast<Real>(9);
    Real const safe = simdSelect(inRange, magnitude, simdBroadcast<Real>(0));
    detail::positiveErfcexp(safe, complement, exponential);
    *complement =
        simdSelect(x < simdBroadcast<Real>(0), simdBroadcast<Real>(2) - *complement, *complement);
    if (!simdAllOf(inRange)) {
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
    Real const magnitude = simdMax(x, -x);
    auto const small = magnitude <= simdBroadcast<Real>(0.75);
    if (simdAllOf(small)) {
        return detail::erfSmall(x);
    }
    Real result = simdBroadcast<Real>(1) - special_functions::erfc(magnitude);
    result = simdSelect(x < simdBroadcast<Real>(0), -result, result);
    if (simdAnyOf(small)) {
        result = simdSelect(
            small, detail::erfSmall(simdSelect(small, x, simdBroadcast<Real>(0))), result);
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
        double const residual = simdFma(-x, x, square);
        *exponential = simdFma(*exponential, residual, *exponential);
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
