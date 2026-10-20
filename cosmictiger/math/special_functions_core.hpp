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

#include <cosmictiger/math/simd_math.hpp>

#ifdef __CUDACC__
#define COSMICTIGER_SPECIAL_HD __host__ __device__
#elif defined(__GNUC__)
// GCC otherwise leaves some polynomial helpers out of line inside large
// SIMD kernels, preventing vectorization across their call boundaries.
#define COSMICTIGER_SPECIAL_HD [[gnu::always_inline]]
#else
#define COSMICTIGER_SPECIAL_HD
#endif

namespace special_functions {
namespace detail {

    template <class Real>
    COSMICTIGER_SPECIAL_HD inline Real roundEven(Real value) {
        if constexpr (std::is_arithmetic_v<Real>) {
#ifdef __CUDA_ARCH__
            return ::nearbyintf(value);
#else
            return __builtin_roundevenf(value);
#endif
        } else {
#ifndef __CUDACC__
            return simdRound(value);
#endif
        }
    }

    template <class Real>
    COSMICTIGER_SPECIAL_HD inline Real floor(Real value) {
        if constexpr (std::is_arithmetic_v<Real>) {
            return ::floorf(value);
        } else {
#ifndef __CUDACC__
            return simdFloor(value);
#endif
        }
    }

    // n is an exactly represented integer in [-126, 127].
    template <class Real>
    COSMICTIGER_SPECIAL_HD inline Real powerOfTwo(Real n) {
        if constexpr (std::is_arithmetic_v<Real>) {
            int const bits = (int(n) + 127) << 23;
#ifdef __CUDA_ARCH__
            return __int_as_float(bits);
#else
            return std::bit_cast<float>(bits);
#endif
        } else {
#ifndef __CUDACC__
            using Integer = SimdVector<int, Real::size()>;
            return simdBitCast<Real>((simdCast<Integer>(n) + Integer(127)) << 23);
#endif
        }
    }

#ifndef __CUDACC__
    template <class Real, class Function>
    inline Real mapUnary(Real value, Function function) {
        if constexpr (std::is_arithmetic_v<Real>) {
            return function(value);
        } else {
            return Real([&](auto lane) { return function(float(value[lane])); });
        }
    }
#endif

}    // namespace detail
}    // namespace special_functions
