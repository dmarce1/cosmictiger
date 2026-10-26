/*
 * CosmicTiger - A cosmological N-Body code
 * Copyright (C) 2026 Dominic C. Marcello
 * Distributed under the Boost Software License, Version 1.0.
 */
#pragma once

#include <cmath>
#include <type_traits>

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/special_functions_core.hpp>

namespace special_functions {

// The fast path covers the required Ewald range of four pi in either direction.
// Use platform math outside the tested |x| <= 16 interval.
template <class Real>
#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline void sincos(Real x, Real* sine, Real* cosine) {
    if constexpr (std::is_arithmetic_v<Real>) {
        static_assert(std::is_same_v<Real, float>);
    } else {
        static_assert(std::is_same_v<typename Real::value_type, float>);
    }

    Real const bound = simdBroadcast<Real>(16.0f);
    bool outside;
    if constexpr (std::is_arithmetic_v<Real>) {
        outside = x > bound || x < -bound;
    } else {
        outside = simdAnyOf((x > bound) | (x < -bound));
    }
    if (outside) {
        if constexpr (std::is_same_v<Real, float>) {
            *sine = ::sinf(x);
            *cosine = ::cosf(x);
        } else {
#ifndef __CUDACC__
            *sine = detail::mapUnary(x, [](float lane) { return std::sin(lane); });
            *cosine = detail::mapUnary(x, [](float lane) { return std::cos(lane); });
#endif
        }
        return;
    }

    // Split pi/2 so cancellation near multiples of pi retains float accuracy.
    Real const turns = detail::roundEven(x * simdBroadcast<Real>(0x1.45f306p-1f));
    Real reduced = simdFma(-turns, simdBroadcast<Real>(0x1.921fb6p+0f), x);
    reduced = simdFma(-turns, simdBroadcast<Real>(-0x1.777a5cp-25f), reduced);
    reduced = simdFma(-turns, simdBroadcast<Real>(-0x1.ee59dap-50f), reduced);

    Real const squared = reduced * reduced;
    Real const fourth = squared * squared;
    Real const eighth = fourth * fourth;

    // Estrin form exposes independent polynomial chains for sin and cos.
    Real const sineLow =
        simdFma(simdBroadcast<Real>(1.0f / 120.0f), squared, simdBroadcast<Real>(-1.0f / 6.0f));
    Real const sineHigh = simdFma(
        simdBroadcast<Real>(1.0f / 362880.0f), squared, simdBroadcast<Real>(-1.0f / 5040.0f));
    Real const sinePoly = simdFma(sineHigh, fourth, sineLow);
    Real const cosineLow =
        simdFma(simdBroadcast<Real>(1.0f / 24.0f), squared, simdBroadcast<Real>(-1.0f / 2.0f));
    Real const cosineHigh =
        simdFma(simdBroadcast<Real>(1.0f / 40320.0f), squared, simdBroadcast<Real>(-1.0f / 720.0f));
    Real const cosinePoly = simdFma(
        simdBroadcast<Real>(-1.0f / 3628800.0f), eighth, simdFma(cosineHigh, fourth, cosineLow));
    Real const sineReduced = simdFma(reduced * squared, sinePoly, reduced);
    Real const cosineReduced = simdFma(squared, cosinePoly, simdBroadcast<Real>(1.0f));

    // turns is an exact integer in [-10, 10], so this remainder is exact.
    Real const quadrant = simdFma(
        detail::floor(turns * simdBroadcast<Real>(0.25f)), simdBroadcast<Real>(-4.0f), turns);
    auto const zero = quadrant == simdBroadcast<Real>(0.0f);
    auto const one = quadrant == simdBroadcast<Real>(1.0f);
    auto const two = quadrant == simdBroadcast<Real>(2.0f);
    Real const rotatedSine = simdSelect(zero, sineReduced,
        simdSelect(one, cosineReduced, simdSelect(two, -sineReduced, -cosineReduced)));
    *sine = simdSelect(x == simdBroadcast<Real>(0.0f), x, rotatedSine);
    *cosine = simdSelect(zero, cosineReduced,
        simdSelect(one, -sineReduced, simdSelect(two, -cosineReduced, sineReduced)));
}

template <class Real>
#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline Real sin(Real x) {
    Real sine, cosine;
    special_functions::sincos(x, &sine, &cosine);
    return sine;
}

template <class Real>
#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline Real cos(Real x) {
    Real sine, cosine;
    special_functions::sincos(x, &sine, &cosine);
    return cosine;
}

#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline void sincos(double x, double* sine, double* cosine) {
    *sine = ::sin(x);
    *cosine = ::cos(x);
}

#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline double sin(double x) {
    return ::sin(x);
}

#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline double cos(double x) {
    return ::cos(x);
}

}    // namespace special_functions
