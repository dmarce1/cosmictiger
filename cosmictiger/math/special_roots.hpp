/*
 * CosmicTiger - A cosmological N-Body code
 * Copyright (C) 2026 Dominic C. Marcello
 * Distributed under the GNU General Public License, version 2 or later.
 */
#pragma once

#include <bit>
#include <cmath>
#include <cstdint>

#if defined(__SSE__) && !defined(__CUDACC__)
#include <immintrin.h>
#endif

#include <cosmictiger/math/special_functions_core.hpp>

namespace special_functions {

// FMA has one final rounding. In particular, it must not be replaced by a
// separately rounded multiplication followed by an addition.
COSMICTIGER_SPECIAL_HD inline float fma(float first, float second, float third) {
#ifdef __CUDA_ARCH__
    return __fmaf_rn(first, second, third);
#else
    return ::fmaf(first, second, third);
#endif
}

COSMICTIGER_SPECIAL_HD inline double fma(double first, double second, double third) {
#ifdef __CUDA_ARCH__
    return __fma_rn(first, second, third);
#else
    return ::fma(first, second, third);
#endif
}

COSMICTIGER_SPECIAL_HD inline float sqrt(float value) {
#ifdef __CUDA_ARCH__
    return __fsqrt_rn(value);
#elif defined(__SSE__) && !defined(__CUDACC__)
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(value)));
#else
    return ::sqrtf(value);
#endif
}

COSMICTIGER_SPECIAL_HD inline double sqrt(double value) {
#ifdef __CUDA_ARCH__
    return __dsqrt_rn(value);
#else
    return ::sqrt(value);
#endif
}

namespace detail {

    COSMICTIGER_SPECIAL_HD inline bool rootPositiveNormal(float value) {
#ifdef __CUDA_ARCH__
        uint32_t const bits = __float_as_uint(value);
#else
        uint32_t const bits = std::bit_cast<uint32_t>(value);
#endif
        return bits >= 0x00800000U && bits < 0x7f800000U;
    }

    // Form x*y before multiplying by y: y*y can overflow at the lower end
    // of the normal input range. The correction is stable across that range.
    template <class Real>
    COSMICTIGER_SPECIAL_HD inline Real refineRsqrt(Real value, Real estimate) {
        Real const product = value * estimate;
        Real const residual = simdFma(-product, estimate, simdBroadcast<Real>(1.0f));
        return simdFma(estimate * simdBroadcast<Real>(0.5f), residual, estimate);
    }

#if defined(__SSE__) && !defined(__CUDACC__)
    inline SimdFloat rootEstimate(SimdFloat const& value) {
        alignas(32) float lanes[SimdFloat::size()];
        for (size_t lane = 0; lane < SimdFloat::size(); ++lane) {
            lanes[lane] = value[lane];
        }
#ifdef __AVX__
        _mm256_store_ps(lanes, _mm256_rsqrt_ps(_mm256_load_ps(lanes)));
#else
        _mm_store_ps(lanes, _mm_rsqrt_ps(_mm_load_ps(lanes)));
        _mm_store_ps(lanes + 4, _mm_rsqrt_ps(_mm_load_ps(lanes + 4)));
#endif
        return SimdFloat([&](auto lane) { return lanes[lane]; });
    }
#endif

}    // namespace detail

COSMICTIGER_SPECIAL_HD inline float rsqrt(float value) {
#ifdef __CUDA_ARCH__
    return __frsqrt_rn(value);
#elif defined(__SSE__) && !defined(__CUDACC__)
    if (detail::rootPositiveNormal(value)) {
        float const estimate = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(value)));
        return detail::refineRsqrt(value, estimate);
    }
    return 1.0f / special_functions::sqrt(value);
#else
    return 1.0f / special_functions::sqrt(value);
#endif
}

COSMICTIGER_SPECIAL_HD inline double rsqrt(double value) {
#ifdef __CUDA_ARCH__
    return __ddiv_rn(1.0, __dsqrt_rn(value));
#else
    return 1.0 / special_functions::sqrt(value);
#endif
}

#ifndef __CUDACC__
inline SimdFloat fma(SimdFloat const& first, SimdFloat const& second, SimdFloat const& third) {
    return simdFma(first, second, third);
}

inline SimdFloat sqrt(SimdFloat const& value) {
#if defined(__SSE__)
    alignas(32) float lanes[SimdFloat::size()];
    for (size_t lane = 0; lane < SimdFloat::size(); ++lane) {
        lanes[lane] = value[lane];
    }
#ifdef __AVX__
    _mm256_store_ps(lanes, _mm256_sqrt_ps(_mm256_load_ps(lanes)));
#else
    _mm_store_ps(lanes, _mm_sqrt_ps(_mm_load_ps(lanes)));
    _mm_store_ps(lanes + 4, _mm_sqrt_ps(_mm_load_ps(lanes + 4)));
#endif
    return SimdFloat([&](auto lane) { return lanes[lane]; });
#else
    return simdSqrt(value);
#endif
}

inline SimdFloat rsqrt(SimdFloat const& value) {
#if defined(__SSE__)
    using Unsigned = SimdVector<uint32_t, SimdFloat::size()>;
    Unsigned const bits = simdBitCast<Unsigned>(value);
    auto const normal = (bits >= Unsigned(0x00800000U)) && (bits < Unsigned(0x7f800000U));
    if (simdAllOf(normal)) {
        return detail::refineRsqrt(value, detail::rootEstimate(value));
    }
    return SimdFloat([&](auto lane) { return special_functions::rsqrt(float(value[lane])); });
#else
    return SimdFloat(1.0f) / special_functions::sqrt(value);
#endif
}
#endif

}    // namespace special_functions
