/*
 * CosmicTiger - A cosmological N-Body code
 * Copyright (C) 2026 Dominic C. Marcello
 * Distributed under the Boost Software License, Version 1.0.
 */
#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/special_functions_core.hpp>

#ifdef __CUDACC__
// NVIDIA's documented libdevice entry point avoids the powf -> __powf rewrite
// performed by --use_fast_math, whose error exceeds this API's float tolerance.
extern "C" __device__ float __nv_powf(float base, float exponent);
#endif

namespace special_functions {
namespace detail {

    template <class Real>
    struct LogReduction {
        Real mantissa;
        Real exponent;
    };

    // For positive normal binary32 x, obtain x = mantissa * 2^exponent with
    // sqrt(1/2) <= mantissa < sqrt(2). The integer normalization is exact.
    template <class Real>
#if defined(__GNUC__) && !defined(__CUDACC__)
    [[gnu::always_inline]]
#endif
    CUDA_EXPORT inline LogReduction<Real> reduceLog(Real value) {
        constexpr uint32_t kMantissaBoundary = 0x3f3504f3U;
        constexpr uint32_t kMantissaMask = 0x007fffffU;
        if constexpr (std::is_arithmetic_v<Real>) {
            static_assert(std::is_same_v<Real, float>);
#ifdef __CUDA_ARCH__
            uint32_t const bits = __float_as_uint(value);
#else
            uint32_t const bits = std::bit_cast<uint32_t>(value);
#endif
            uint32_t const reduced = bits - kMantissaBoundary;
#ifdef __CUDA_ARCH__
            int const exponent = __float_as_int(__uint_as_float(reduced)) >> 23;
            float const mantissa = __uint_as_float((reduced & kMantissaMask) + kMantissaBoundary);
#else
            int const exponent = std::bit_cast<int32_t>(reduced) >> 23;
            float const mantissa =
                std::bit_cast<float>((reduced & kMantissaMask) + kMantissaBoundary);
#endif
            return {mantissa, float(exponent)};
        } else {
#ifndef __CUDACC__
            static_assert(std::is_same_v<typename Real::value_type, float>);
            using Unsigned = SimdVector<uint32_t, Real::size()>;
            using Integer = SimdVector<int32_t, Real::size()>;
            Unsigned const reduced = simdBitCast<Unsigned>(value) - Unsigned(kMantissaBoundary);
            Integer const exponent = simdBitCast<Integer>(reduced) >> 23;
            Real const mantissa = simdBitCast<Real>(
                (reduced & Unsigned(kMantissaMask)) + Unsigned(kMantissaBoundary));
            return {mantissa, simdCast<Real>(exponent)};
#endif
        }
    }

    // log(m) = 2 atanh(t), t = (m-1)/(m+1), |t| <= 0.171573.
    // Six terms through t^11 leave less than 3e-11 absolute truncation error.
    // Pairing coefficients shortens Horner's dependency chain from five FMAs to
    // three, at the cost of one additional multiplication.
    template <bool baseTwo, class Real>
#if defined(__GNUC__) && !defined(__CUDACC__)
    [[gnu::always_inline]]
#endif
    CUDA_EXPORT inline Real logPositiveNormal(Real value) {
        auto const reduced = reduceLog(value);
        Real const t = (reduced.mantissa - simdBroadcast<Real>(1.0f)) /
            (reduced.mantissa + simdBroadcast<Real>(1.0f));
        Real const squared = t * t;
        constexpr float kScale = baseTwo ? 2.8853900817779268f : 2.0f;
        Real const pair0 =
            simdFma(simdBroadcast<Real>(kScale / 3.0f), squared, simdBroadcast<Real>(kScale));
        Real const pair1 = simdFma(
            simdBroadcast<Real>(kScale / 7.0f), squared, simdBroadcast<Real>(kScale / 5.0f));
        Real const pair2 = simdFma(
            simdBroadcast<Real>(kScale / 11.0f), squared, simdBroadcast<Real>(kScale / 9.0f));
        Real const fourth = squared * squared;
        Real const polynomial = simdFma(simdFma(pair2, fourth, pair1), fourth, pair0);
        if constexpr (baseTwo) {
            return simdFma(t, polynomial, reduced.exponent);
        } else {
            Real const mantissaLog = t * polynomial;
            Real const low =
                simdFma(reduced.exponent, simdBroadcast<Real>(-1.904654323148236e-9f), mantissaLog);
            return simdFma(reduced.exponent, simdBroadcast<Real>(0.6931471824645996f), low);
        }
    }

#if defined(__GNUC__) && !defined(__CUDACC__)
    [[gnu::always_inline]]
#endif
    CUDA_EXPORT inline bool positiveNormal(float value) {
#ifdef __CUDA_ARCH__
        uint32_t const bits = __float_as_uint(value);
#else
        uint32_t const bits = std::bit_cast<uint32_t>(value);
#endif
        return bits >= 0x00800000U && bits < 0x7f800000U;
    }

}    // namespace detail

// Normal positive inputs follow the shared CPU/CUDA approximation. Library
// fallbacks retain the platform's behavior for zero, subnormal, negative, and
// nonfinite inputs.
#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline float log(float value) {
    return detail::positiveNormal(value) ? detail::logPositiveNormal<false>(value) : ::logf(value);
}

#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline float log2(float value) {
    return detail::positiveNormal(value) ? detail::logPositiveNormal<true>(value) : ::log2f(value);
}

#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline float pow(float base, float exponent) {
#ifdef __CUDA_ARCH__
    return ::__nv_powf(base, exponent);
#else
    return ::powf(base, exponent);
#endif
}

// Preserve precision when a caller explicitly selects a double argument.
#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline double log(double value) {
    return ::log(value);
}

#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline double log2(double value) {
    return ::log2(value);
}

#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
CUDA_EXPORT inline double pow(double base, double exponent) {
    return ::pow(base, exponent);
}

#ifndef __CUDACC__
inline SimdFloat log(SimdFloat const& value) {
    SimdFloat result = detail::logPositiveNormal<false>(value);
    using Unsigned = SimdVector<uint32_t, SimdFloat::size()>;
    Unsigned const bits = simdBitCast<Unsigned>(value);
    auto const normal = (bits >= Unsigned(0x00800000U)) && (bits < Unsigned(0x7f800000U));
    if (!simdAllOf(normal)) {
        for (size_t lane = 0; lane < SimdFloat::size(); ++lane) {
            if (!normal[lane]) {
                simdSetLane(result, lane, ::logf(float(value[lane])));
            }
        }
    }
    return result;
}

inline SimdFloat log2(SimdFloat const& value) {
    SimdFloat result = detail::logPositiveNormal<true>(value);
    using Unsigned = SimdVector<uint32_t, SimdFloat::size()>;
    Unsigned const bits = simdBitCast<Unsigned>(value);
    auto const normal = (bits >= Unsigned(0x00800000U)) && (bits < Unsigned(0x7f800000U));
    if (!simdAllOf(normal)) {
        for (size_t lane = 0; lane < SimdFloat::size(); ++lane) {
            if (!normal[lane]) {
                simdSetLane(result, lane, ::log2f(float(value[lane])));
            }
        }
    }
    return result;
}

inline SimdFloat pow(SimdFloat const& base, SimdFloat const& exponent) {
    return SimdFloat([&](auto lane) { return ::powf(float(base[lane]), float(exponent[lane])); });
}
#endif

}    // namespace special_functions
